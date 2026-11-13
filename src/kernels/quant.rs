//! Affine Q4/Q8 kernels with BF16 activations and F32 accumulation.
//! Q4 packs eight low-nibble-first codes per u32; `w = scale*q + bias`.

use anyhow::{Result, ensure};

use crate::kernels::gemm::gemm_bf16_nt;
use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/quant.metal");

/// Rows each thread of the Q4 gather GEMV accumulates.
#[cfg(not(target_os = "linux"))]
const Q4_GEMV_ROWS: usize = 2;

/// Code bytes from which a lane of the Vulkan raw-code projection body
/// (`vulkan/quant_rows.glsl`: the Q4 GEMVs and the 8-bit register-A GEMMs)
/// computes two weight rows per block, sharing the block's activation work.
/// Measured 2026-11-10 on the M5 Max in chains of barriered levels
/// (`quant_rows_layout_probe`, us per level at one row per lane against
/// two): two rows win from the 6.5 MB projections up (the 21 MB Q4 GDN
/// input GEMV 54.2 against 49.8, the 16 MB Q8 attention output at three
/// activation rows 42.6 against 35.7), about tie at 3.3 MB (11.8 against
/// 11.1 at three rows), and lose on the 1.3 to 1.6 MB ones, whose levels
/// are bound by their ramp and want more, shorter subgroups (8.4 to 8.7
/// against 9.6 to 9.9 at three rows).
#[cfg(target_os = "linux")]
const ROWS_PAIR_MIN_CODE_BYTES: usize = 3 << 20;

/// The layout of the Vulkan raw-code projection body for a weight: log2 of
/// the lanes that walk a row set, and the weight rows a lane computes per
/// block. The lanes are the largest power of two up to a subgroup's 32 that
/// divides the row's blocks (one `uint4` of codes each), so no lane idles
/// through a step; they depend on the row length alone, which keeps a row's
/// result the same bits in every dispatch of that length (a fused stack and
/// its slices, a weight alone and among others in one dispatch).
#[cfg(target_os = "linux")]
pub(crate) fn rows_layout(w: &QuantWeights) -> (usize, usize) {
    let (n, k) = (w.out_features(), w.in_features());
    (rows_lanes_log2(k, w.bits), rows_per_lane(n * k * w.bits / 8))
}

#[cfg(target_os = "linux")]
fn rows_lanes_log2(k: usize, bits: usize) -> usize {
    let blocks = k * bits / 128;
    (blocks.trailing_zeros() as usize).min(5)
}

#[cfg(target_os = "linux")]
fn rows_per_lane(code_bytes: usize) -> usize {
    if code_bytes >= ROWS_PAIR_MIN_CODE_BYTES { 2 } else { 1 }
}

/// The grid of the Vulkan raw-code projection body: one subgroup per
/// `32 >> lanes_log2` row sets of `rows` weight rows.
#[cfg(target_os = "linux")]
pub(crate) fn rows_grid(n: usize, lanes_log2: usize, rows: usize) -> Grid {
    Grid::Threadgroups {
        groups: (n.div_ceil((32 >> lanes_log2) * rows), 1, 1),
        threadgroup: (32, 1, 1),
    }
}

/// Row-tile height and execution-simdgroup count of a grouped Q4 GEMM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeTile {
    T32Sg4,
    T64Sg4,
}

impl MoeTile {
    /// Row-tile height T — the block map's granularity.
    pub fn rows(self) -> usize {
        match self {
            Self::T32Sg4 => 32,
            Self::T64Sg4 => 64,
        }
    }

    /// Tensor-op simdgroups; dispatch uses `32 * simdgroups` threads.
    pub fn simdgroups(self) -> usize {
        match self {
            Self::T32Sg4 | Self::T64Sg4 => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::T32Sg4 => "32x4",
            Self::T64Sg4 => "64x4",
        }
    }

    fn nax_kernel(self) -> &'static str {
        match self {
            Self::T32Sg4 => "gemm_q4_nt_nax_grouped_t32x4",
            Self::T64Sg4 => "gemm_q4_nt_nax_grouped_t64x4",
        }
    }
}

/// Row count at which grouped Q4 GEMM switches to the 64x4 tile.
pub const MOE_TILE_MROUTE_DEFAULT: usize = 3072;

/// Largest prefill chunk routed to grouped GEMV instead of tiled GEMM.
const MOE_SMALLM_THRESHOLD: usize = 8;

// The small-M route is limited by per-expert register capacity.
const _: () = assert!(MOE_SMALLM_THRESHOLD > 0);
const _: () = assert!(MOE_SMALLM_THRESHOLD <= crate::kernels::moe::MOE_SMALLM_MAX_M);

/// Grouped GEMM or direct small-M GEMV execution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeRoute {
    Grouped(MoeTile),
    SmallmGemv,
}

impl MoeRoute {
    pub fn label(self) -> &'static str {
        match self {
            Self::Grouped(tile) => tile.label(),
            Self::SmallmGemv => "gemv",
        }
    }
}

/// Fixed per-chunk MoE routing policy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoeTilePolicy;

impl MoeTilePolicy {
    /// Selects small-M through 8 rows, 64x4 from 3072, otherwise 32x4.
    pub fn route_for(&self, chunk_m: usize) -> MoeRoute {
        if chunk_m > 0 && chunk_m <= MOE_SMALLM_THRESHOLD {
            MoeRoute::SmallmGemv
        } else {
            MoeRoute::Grouped(self.tile_for(chunk_m))
        }
    }

    fn tile_for(&self, chunk_m: usize) -> MoeTile {
        if chunk_m >= MOE_TILE_MROUTE_DEFAULT {
            MoeTile::T64Sg4
        } else {
            MoeTile::T32Sg4
        }
    }

    /// Upper bound for grouped block-map rows.
    pub fn alloc_tile(&self) -> MoeTile {
        MoeTile::T32Sg4
    }
}

/// Returns the fixed chunk-routing policy.
pub fn moe_tile_mroute_table() -> MoeTilePolicy {
    MoeTilePolicy
}

/// Validates a quantized weight for the GEMV kernels; returns `(out, in)`.
pub(crate) fn check_quant(w: &QuantWeights) -> Result<(usize, usize)> {
    let (n, k) = (w.out_features(), w.in_features());
    ensure!(matches!(w.bits, 4 | 8), "unsupported bit width {}", w.bits);
    // A uint4 block must not cross a quantization group.
    ensure!(
        k.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(128 / w.bits),
        "in_features {k} / group size {} not block-packable at {} bits",
        w.group_size,
        w.bits
    );
    // The Vulkan kernels take a block's quant group by a shift.
    #[cfg(target_os = "linux")]
    ensure!(
        w.group_size.is_power_of_two(),
        "group size {} is not a power of two",
        w.group_size
    );
    ensure!(w.codes.dtype() == DType::U32, "codes must be U32");
    // uint4 code loads require 16-byte alignment.
    ensure!(
        w.codes.binding().1.is_multiple_of(16),
        "codes byte offset {} not 16-aligned; the kernels read codes as uint4",
        w.codes.binding().1
    );
    ensure!(
        w.scales.shape() == [n, k / w.group_size]
            && w.biases.shape() == [n, k / w.group_size],
        "scales/biases shape mismatch for [{n}, {k}] gs={}",
        w.group_size
    );
    Ok((n, k))
}

pub fn gemv_quant(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    y: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(x.numel() == k, "x numel {} != K {k}", x.numel());
    ensure!(y.numel() == n, "y numel {} != N {n}", y.numel());
    // Vulkan: the 4-bit GEMVs are the raw-code projection body at one
    // activation row (`vulkan/quant_rows.glsl`), with its params and grid.
    #[cfg(target_os = "linux")]
    {
        if w.bits == 4 {
            return gemv_q4_rows(ctx, pass, w, x, y, n, k);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let requested_rows =
        if w.bits == 4 && y.dtype() == DType::BF16 { Q4_GEMV_ROWS } else { 1 };
    #[cfg(target_os = "linux")]
    let requested_rows = 1;
    let packed_rows = if n.is_multiple_of(requested_rows) { requested_rows } else { 1 };
    let fn_name = match (w.bits, y.dtype(), packed_rows) {
        (4, DType::BF16, 1) => "gemv_q4_bf16",
        (4, DType::BF16, 2) => "gemv_q4_bf16_2row",
        (4, DType::F32, 1) => "gemv_q4_bf16_f32out",
        (8, DType::BF16, 1) => "gemv_q8_bf16",
        (8, DType::F32, 1) => "gemv_q8_bf16_f32out",
        _ => anyhow::bail!("gemv output must be BF16 or F32"),
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            y.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threadgroups { groups: (n / packed_rows, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Vulkan: a 4-bit GEMV through the raw-code projection body at one
/// activation row (`vulkan/quant_rows.glsl`), with its params and grid.
#[cfg(target_os = "linux")]
fn gemv_q4_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    y: &Tensor,
    n: usize,
    k: usize,
) -> Result<()> {
    let (lanes_log2, rows) = rows_layout(w);
    let fn_name = match (y.dtype(), rows) {
        (DType::BF16, 1) => "gemv_q4_bf16",
        (DType::BF16, _) => "gemv_q4_bf16_2row",
        (DType::F32, 1) => "gemv_q4_bf16_f32out",
        (DType::F32, _) => "gemv_q4_bf16_f32out_2row",
        _ => anyhow::bail!("gemv output must be BF16 or F32"),
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            y.binding(),
        ],
        &[
            &u32_bytes(k),
            &u32_bytes(n),
            &u32_bytes(w.group_size),
            &u32_bytes(lanes_log2),
        ],
        rows_grid(n, lanes_log2, rows),
    )
}

/// The GEMVs of one dependency level that read the same `x`: `y = W x` for
/// each `(W, y)` of `sets`, every `y` BF16 or F32 on its own. On Metal the
/// dispatches of a level overlap and this is a GEMV per set. On Vulkan they
/// do not (measured 2026-11-10, `dense_shared_levels_timing`: a decode
/// step's router level takes as three dispatches the sum of what each takes
/// alone, 18.9 us, and 14.0 as one), so the sets of one code width, row
/// length and group size go out as one dispatch, up to four at a time
/// (`gemv_q8_sets`, `gemv_q4_sets`); each output is the bits `gemv_quant`
/// gives it.
pub fn gemv_quant_sets(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    sets: &[(&QuantWeights, &Tensor)],
) -> Result<()> {
    #[cfg(target_os = "linux")]
    for bits in [4usize, 8] {
        let same: Vec<(&QuantWeights, &Tensor)> =
            sets.iter().copied().filter(|(w, _)| w.bits == bits).collect();
        let fused = same.len() >= 2
            && same.iter().all(|(w, _)| {
                w.in_features() == same[0].0.in_features()
                    && w.group_size == same[0].0.group_size
            });
        for group in same.chunks(if fused { SETS_MAX } else { 1 }) {
            match group {
                [(w, y)] => gemv_quant(ctx, pass, w, x, y)?,
                _ => gemv_sets_dispatch(ctx, pass, x, group)?,
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    for (w, y) in sets {
        gemv_quant(ctx, pass, w, x, y)?;
    }
    Ok(())
}

/// Weights one multi-weight dispatch of the Vulkan projection kernels takes;
/// must match `QR_SETS` in `vulkan/quant_rows.glsl` and `gemv_q8_sets`.
#[cfg(target_os = "linux")]
pub(crate) const SETS_MAX: usize = 4;

/// The buffers, params and grid of a multi-weight dispatch of the Vulkan
/// projection kernels (`vulkan/quant_rows.glsl`, `QR_SETS`; `gemv_q8_sets`)
/// over `a` `[m, K]` (or `[K]`), and the weight rows per lane. `layout` is
/// the kernel's fixed lanes per row set (log2) and rows per lane; without
/// one they are the raw-code body's choice for the sets' bytes together (a
/// row's bits depend on neither). The sets share a code width, K and group
/// size.
#[cfg(target_os = "linux")]
#[allow(clippy::type_complexity)]
pub(crate) fn sets_dispatch_args<'a>(
    a: &'a Tensor,
    sets: &[(&'a QuantWeights, &'a Tensor)],
    layout: Option<(usize, usize)>,
) -> Result<(usize, Vec<(&'a crate::metal::GpuBuffer, usize)>, [u32; 8], Grid)> {
    ensure!(
        (1..=SETS_MAX).contains(&sets.len()),
        "a multi-weight projection takes 1..={SETS_MAX} weights (got {})",
        sets.len()
    );
    let first = sets[0].0;
    let (k, bits) = (first.in_features(), first.bits);
    let mut code_bytes = 0;
    let mut ns = [0u32; SETS_MAX];
    let mut f32_mask = 0u32;
    for (s, (w, c)) in sets.iter().enumerate() {
        let (n, wk) = check_quant(w)?;
        ensure!(
            wk == k && w.bits == bits && w.group_size == first.group_size,
            "the weights of a multi-weight projection differ in K, bits or group size"
        );
        ensure!(
            matches!(c.dtype(), DType::BF16 | DType::F32),
            "projection output must be BF16 or F32"
        );
        code_bytes += n * k * bits / 8;
        ns[s] = n as u32;
        f32_mask |= u32::from(c.dtype() == DType::F32) << s;
    }
    let (lanes_log2, rows) =
        layout.unwrap_or((rows_lanes_log2(k, bits), rows_per_lane(code_bytes)));
    // Unused sets repeat the first one's buffers, with no rows.
    let set = |s: usize| sets.get(s).unwrap_or(&sets[0]);
    let mut buffers = vec![
        first.codes.binding(),
        first.scales.binding(),
        first.biases.binding(),
        a.binding(),
        sets[0].1.binding(),
    ];
    for s in 1..SETS_MAX {
        let (w, c) = *set(s);
        buffers.extend([
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            c.binding(),
        ]);
    }
    let params = [
        k as u32,
        ns[0],
        first.group_size as u32,
        lanes_log2 as u32,
        ns[1],
        ns[2],
        ns[3],
        f32_mask,
    ];
    let rows_per_sg = (32 >> lanes_log2) * rows;
    let subgroups: usize =
        sets.iter().map(|(w, _)| w.out_features().div_ceil(rows_per_sg)).sum();
    let grid = Grid::Threadgroups { groups: (subgroups, 1, 1), threadgroup: (32, 1, 1) };
    Ok((rows, buffers, params, grid))
}

#[cfg(target_os = "linux")]
fn gemv_sets_dispatch(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    sets: &[(&QuantWeights, &Tensor)],
) -> Result<()> {
    let (k, bits) = (sets[0].0.in_features(), sets[0].0.bits);
    ensure!(x.numel() == k, "x numel {} != K {k}", x.numel());
    for (w, y) in sets {
        ensure!(
            y.numel() == w.out_features(),
            "y numel {} != N {}",
            y.numel(),
            w.out_features()
        );
    }
    // The 8-bit GEMV is one subgroup per row, as on Metal.
    let layout = (bits == 8).then_some((5, 1));
    let (rows, buffers, params, grid) = sets_dispatch_args(x, sets, layout)?;
    let fn_name = match (bits, rows) {
        (4, 1) => "gemv_q4_sets",
        (4, _) => "gemv_q4_sets_2row",
        _ => "gemv_q8_sets",
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    let params = params.map(u32::to_ne_bytes);
    let params: Vec<&[u8]> = params.iter().map(|p| &p[..]).collect();
    pass.dispatch_at(&pipeline, &buffers, &params, grid)
}

/// Writes `out[N, K] = dequant(W)` in bf16.
fn dequant_to_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    out: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(
        out.shape() == [n, k] && out.dtype() == DType::BF16,
        "dequant out must be BF16 [{n}, {k}], got {:?} {:?}",
        out.shape(),
        out.dtype()
    );
    let fn_name = if w.bits == 4 { "dequant_q4_bf16" } else { "dequant_q8_bf16" };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[w.codes.binding(), w.scales.binding(), w.biases.binding(), out.binding()],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threads { grid: (k / (32 / w.bits), n, 1), threadgroup: (32, 1, 1) },
    )
}

/// Dequantizes `W` into bf16 scratch, then computes
/// `C[M, N] = A[M, K] . W[N, K]^T`.
pub fn gemm_quant_bf16_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    scratch: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(
        scratch.numel() >= n * k && scratch.dtype() == DType::BF16,
        "dequant scratch holds {} bf16 elements, needs {}",
        scratch.numel(),
        n * k
    );
    let staged = scratch.view(0, &[n, k])?;
    dequant_to_bf16(ctx, pass, w, &staged)?;
    // The GEMM reads the staged dequant, and the next projection's dequant
    // must not overwrite the staging while it does: both edges get barriers
    // (no-ops on a serial encoder; the staged path is only taken for chunks
    // past the skinny kernels' row limit, where the GEMMs fill the GPU).
    pass.level_barrier(&[&staged])?;
    gemm_bf16_nt(ctx, pass, a, &staged, c)?;
    pass.level_barrier(&[c])
}

/// A layout of the Vulkan dense quantized GEMM that dequantizes in its
/// multiply loop (`vulkan/quant_dense.glsl`), over 16-row blocks of the
/// weight: with `s` 0 a subgroup per block, running `mb` 16-row blocks of
/// the activations; with `s` > 0 a threadgroup of `s` subgroups per block,
/// `s - mb` (1 to 4) of them dequantizing it for `mb` that multiply.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FusedTile {
    pub mb: usize,
    pub s: usize,
}

#[cfg(target_os = "linux")]
impl FusedTile {
    pub fn kernel(self, bits: usize) -> Result<&'static str> {
        ensure!(
            self.s == 0 || (self.s > self.mb && self.s - self.mb <= 4),
            "no fused dense GEMM tile {self:?}"
        );
        Ok(match (bits, self.s > 0, self.mb) {
            (4, false, 1) => "gemm_q4_nt_dense_b1",
            (4, false, 2) => "gemm_q4_nt_dense_b2",
            (4, false, 4) => "gemm_q4_nt_dense_b4",
            (8, false, 1) => "gemm_q8_nt_dense_b1",
            (8, false, 2) => "gemm_q8_nt_dense_b2",
            (8, false, 4) => "gemm_q8_nt_dense_b4",
            (4, true, 1) => "gemm_q4_nt_dense_pc_b1",
            (4, true, 2) => "gemm_q4_nt_dense_pc_b2",
            (4, true, 4) => "gemm_q4_nt_dense_pc_b4",
            (8, true, 1) => "gemm_q8_nt_dense_pc_b1",
            (8, true, 2) => "gemm_q8_nt_dense_pc_b2",
            (8, true, 4) => "gemm_q8_nt_dense_pc_b4",
            _ => anyhow::bail!("no fused dense GEMM for {bits} bits, tile {self:?}"),
        })
    }
}

/// Subgroups per threadgroup of the fused dense GEMM (each its own tile).
#[cfg(target_os = "linux")]
const FUSED_SUBGROUPS: usize = 2;

/// Whether the fused dense GEMM takes this weight and these tensors: 4 or 8
/// bits, K a multiple of 64, quant groups at least 64 deep, A `[M, K]` and
/// C of M x N elements, both bf16 at word-aligned addresses.
#[cfg(target_os = "linux")]
pub fn fused_takes(a: &Tensor, w: &QuantWeights, c: &Tensor) -> bool {
    a.shape().len() == 2
        && a.shape()[1] == w.in_features()
        && c.numel() == a.shape()[0] * w.out_features()
        && matches!(w.bits, 4 | 8)
        && w.in_features().is_multiple_of(64)
        && w.group_size >= 64
        && w.group_size.is_power_of_two()
        && a.dtype() == DType::BF16
        && c.dtype() == DType::BF16
        && a.binding().1.is_multiple_of(4)
        && c.binding().1.is_multiple_of(4)
}

/// The fused dense GEMM's tile for an `m`-row chunk through this weight, or
/// `None` where the staged route (dequant to bf16, `gemm_bf16_nt_nax`) is
/// the faster. Measured 2026-11-13 on the M5 Max at the model's shapes, us
/// per call in a chain of barriered calls (`dense_quant_gemm_timing`; a
/// prefill pass's dense projections at 32 rows: staged 160 ms, fused 38):
///
/// - Where N is a few hundred rows or fewer the direct tiles have few
///   subgroups, each a long chain of groups, and the producer tile (`s` > 0)
///   is the fast one: q8 320 x 10 240 at 32 rows 83 (direct 162, staged
///   420), 512 x 2 560 23 (42, 121); at up to 256 rows with four blocks of
///   activation rows, 99 and 28 at 128 rows.
/// - Where N fills the GPU, the direct tile with as many blocks of
///   activation rows as keep 512 subgroups in the grid: q4 10 240 x 2 560
///   at 128 rows 209 (staged 315), 2 560 x 6 144 145 (376). At up to 32
///   rows an N under 8 192 takes the producer tile with two multiplying
///   subgroups: 2 560 x 6 144 69 (direct 90).
/// - The staged route ties from 384 rows where N and K are both large (q4
///   10 240 x 2 560 618 against 601) and wins from 512; where N is a few
///   hundred, from 2 048 (q8 512 x 2 560 218 against 161); never where it is
///   tiny (4 x 10 240 at 4 096 rows 199 against 422).
#[cfg(target_os = "linux")]
pub fn fused_route(a: &Tensor, w: &QuantWeights, c: &Tensor) -> Option<FusedTile> {
    let (m, n, k) = (a.shape()[0], w.out_features(), w.in_features());
    if !fused_takes(a, w, c) {
        return None;
    }
    let max_m = match (n, k) {
        (..=64, _) => usize::MAX,
        (_, ..=1024) | (..=640, ..=4096) => 1024,
        (..=640, _) => 2048,
        _ => 256,
    };
    if m > max_m {
        return None;
    }
    if n <= 64 || (n <= 640 && m <= 64) {
        return Some(FusedTile { mb: 1, s: 5 });
    }
    if n <= 640 && m <= 256 {
        return Some(FusedTile { mb: 4, s: 8 });
    }
    if m <= 32 && n < 8192 {
        return Some(FusedTile { mb: 2, s: 4 });
    }
    let mb = [4, 2]
        .into_iter()
        .find(|&mb| n.div_ceil(16) * m.div_ceil(16 * mb) >= 512)
        .unwrap_or(1);
    Some(FusedTile { mb, s: 0 })
}

/// `C[M, N] = A[M, K] . dequant(W)^T` in one dispatch that dequantizes in
/// its multiply loop: the staged route's result bit for bit
/// (`gemm_quant_fused_matches_staged`), without its bf16 copy of W.
#[cfg(target_os = "linux")]
pub fn gemm_quant_fused_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    tile: FusedTile,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(fused_takes(a, w, c), "the fused dense GEMM does not take this weight");
    let m = a.shape()[0];
    ensure!(a.shape() == [m, k], "A shape {:?} != [{m}, {k}]", a.shape());
    ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
    let pipeline = ctx.pipeline(tile.kernel(w.bits)?, SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(m), &u32_bytes(w.group_size)],
        if tile.s > 0 {
            Grid::Threadgroups {
                groups: (n.div_ceil(16), m.div_ceil(16 * tile.mb), 1),
                threadgroup: (32 * tile.s, 1, 1),
            }
        } else {
            Grid::Threadgroups {
                groups: (n.div_ceil(16 * FUSED_SUBGROUPS), m.div_ceil(16 * tile.mb), 1),
                threadgroup: (32 * FUSED_SUBGROUPS, 1, 1),
            }
        },
    )
}

/// Block-mapped grouped GEMM with Q4 dequantization fused into B-tile staging.
/// `tile` must match the configuration used to build the block map.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_grouped_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    blocks: &Tensor,
    n_blocks: usize,
    tile: MoeTile,
) -> Result<()> {
    let (_, k) = check_quant(w)?;
    ensure!(
        w.bits == 4 && w.group_size == 64 && k.is_multiple_of(64),
        "grouped NAX GEMM needs 4-bit group-64 weights and K % 64 == 0"
    );
    let (s, ak) = (a.shape()[0], a.shape()[1]);
    ensure!(ak == k, "A K {ak} != weight K {k}");
    let n = c.shape()[1];
    ensure!(c.shape() == [s, n], "C shape {:?} != [{s}, {n}]", c.shape());
    ensure!(n.is_multiple_of(64), "grouped GEMM needs N % 64 == 0 (got {n})");
    ensure!(blocks.dtype() == DType::U32, "block map must be U32");
    ensure!(blocks.numel() >= n_blocks * 4 && n_blocks > 0, "bad block map");
    let pipeline = ctx.pipeline(tile.nax_kernel(), SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
            blocks.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        Grid::Threadgroups {
            groups: (n_blocks, 1, 1),
            threadgroup: (32 * tile.simdgroups(), 1, 1),
        },
    )
}

/// `out[M, K]` = dequantized embedding rows for `ids` (batched prefill
/// gather), from a 4-bit table or an 8-bit one (`--q8-embed`).
pub fn gather_rows_quant(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    ids: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let (_, k) = check_quant(w)?;
    let m = ids.numel();
    ensure!(ids.dtype() == DType::U32, "ids must be U32");
    ensure!(
        out.numel() == m * k && out.dtype() == DType::BF16,
        "gather out must be BF16 [{m}, {k}]"
    );
    let fn_name =
        if w.bits == 4 { "gather_rows_q4_bf16" } else { "gather_rows_q8_bf16" };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            ids.binding(),
            out.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        // One thread per code word: 32 / bits elements each.
        Grid::Threads { grid: (k / (32 / w.bits), m, 1), threadgroup: (32, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/quant.rs"]
mod tests;
