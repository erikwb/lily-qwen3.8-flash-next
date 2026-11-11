//! Small-M Q4 and Q8 GEMMs with staged-A and register-A variants.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/skinny.metal");

/// Weight rows per threadgroup; must match `SKINNY_SG` in skinny.metal.
const ROWS_PER_TG: usize = 4;

/// Largest row count compiled for register-A kernels.
const REG_MAX_M: usize = 8;

/// Output width from which register-A is used for every small m. Below it,
/// register-A still wins for `m <= REG_SMALL_M` rows (measured on the M5 Max
/// at Qwen3.8-Flash-Next's shapes: the staged walk is latency-bound for
/// narrow, deep projections), so the threshold only matters above that.
const WIDE_N_MIN: usize = 65536;
const REG_SMALL_M: usize = 8;

/// Simdgroups per register-A threadgroup.
const REG_SIMDGROUPS_PER_TG: usize = 2;
/// Weight rows each Q8 register-A simdgroup computes; must match
/// `SKINNY_REG_ROWS` in skinny.metal.
#[cfg(not(target_os = "linux"))]
const REG_ROWS_PER_SG: usize = 2;
/// Weight rows each Q4 register-A simdgroup computes, by m (index m - 1);
/// must match the `GEMM_SKINNY_Q4_REG` instantiations in skinny.metal. More
/// rows share a block's activation work (the bound past m = 2) across more
/// weight rows. Measured on the M5 Max: with a barrier after each projection
/// as in the model's passes, 4 rows beat 8 (which wins only on back-to-back
/// dispatches, see `skinny_rows_scaling_probe`) at every m and beat 2 from
/// m = 3; at m = 2 they sped up the big projections but slowed the
/// `lily-bench --batch-rows 2` step (16.1 against 15.6 ms), so m = 2 keeps 2.
const Q4_REG_ROWS_PER_SG: [usize; REG_MAX_M] = [2, 2, 4, 4, 4, 4, 4, 4];

/// Selects register-A when shape and packing constraints hold.
fn reg_routes(m: usize, n: usize, block_walk_ok: bool) -> bool {
    block_walk_ok && m <= REG_MAX_M && (n >= WIDE_N_MIN || m <= REG_SMALL_M)
}

/// Requires a fused stack and every slice to use the same reduction variant.
pub fn stack_route_uniform(
    m: usize,
    n_total: usize,
    slice_ns: &[usize],
    block_walk_ok: bool,
) -> bool {
    let stack_reg = reg_routes(m, n_total, block_walk_ok);
    slice_ns.iter().all(|&n| reg_routes(m, n, block_walk_ok) == stack_reg)
}

/// Vulkan: whether the dispatches of one dependency level run side by side,
/// as they do on Metal. They do on the Mesa lily runs on (Honeykrisp with
/// "hk: elide the barrier between application dispatches on M5"); on a
/// driver that runs them one after another, `LILY_FUSE_LEVELS=1` brings back
/// the encodings made for it: one multi-weight dispatch for the 8-bit
/// projections of a level (`gemm_skinny_q8_nt_sets`) and one fused GEMM plus
/// a split level for a stack (`stack_as_slices`). The outputs are the same
/// bits either way, so the switch is also the A/B of the two encodings on
/// one build.
#[cfg(target_os = "linux")]
fn level_dispatches_overlap() -> bool {
    static OVERLAP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OVERLAP.get_or_init(|| std::env::var_os("LILY_FUSE_LEVELS").is_none())
}

/// Whether a stacked projection that could go out as one fused GEMM at `m`
/// rows (`moe_ffn::stack_fused_view`) goes out as its slices instead, each a
/// dispatch of the same level writing its own output. On Metal the fused
/// GEMM is one cold start where the slices are several, and the level that
/// splits its output is free. On Vulkan a level's dispatches overlap as well
/// as one dispatch's threadgroups do, and the split level costs what any
/// level does (a barrier and a launch, 2.5 to 3 us), so the slices win at
/// every register-A row count. Measured 2026-11-11 in chains of levels at
/// m = 1, 2, 3, 4 and 8 (`verify_levels_timing`; us per level, fused and
/// split against the slices, smallest first): the 4-bit GDN input stack
/// 53.8/66.8/68.6/82.0/154.3 against 51.7/61.5/65.9/79.6/148.4; the 8-bit
/// attention stack with the indexer's projection on its level
/// 71.6/72.9/75.3/81.5/136.2 against 68.9/69.9/71.2/82.2/130.7. Past the
/// register-A rows (the staged kernels) nothing was measured and the stack
/// stays fused.
pub fn stack_as_slices(m: usize) -> bool {
    #[cfg(target_os = "linux")]
    {
        level_dispatches_overlap() && m <= REG_MAX_M
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = m;
        false
    }
}

/// Largest row count routed to the staged small-M family.
pub const DENSE_SMALLM_THRESHOLD: usize = 16;

/// Whether a nonempty dense chunk uses the staged small-M kernel.
pub fn dense_smallm_routes(m: usize) -> bool {
    m > 0 && m <= DENSE_SMALLM_THRESHOLD
}

fn staged_grid(n: usize) -> Grid {
    Grid::Threadgroups {
        groups: (n.div_ceil(ROWS_PER_TG), 1, 1),
        threadgroup: (32 * ROWS_PER_TG, 1, 1),
    }
}

/// Kernel names for the per-m register-A instantiations (index m - 1), by
/// output type: bf16 activations, f32 logits.
const Q4_REG_FNS: [&str; REG_MAX_M] = [
    "gemm_skinny_q4_bf16_reg_m1",
    "gemm_skinny_q4_bf16_reg_m2",
    "gemm_skinny_q4_bf16_reg_m3",
    "gemm_skinny_q4_bf16_reg_m4",
    "gemm_skinny_q4_bf16_reg_m5",
    "gemm_skinny_q4_bf16_reg_m6",
    "gemm_skinny_q4_bf16_reg_m7",
    "gemm_skinny_q4_bf16_reg_m8",
];
const Q8_REG_FNS: [&str; REG_MAX_M] = [
    "gemm_skinny_q8_bf16_reg_m1",
    "gemm_skinny_q8_bf16_reg_m2",
    "gemm_skinny_q8_bf16_reg_m3",
    "gemm_skinny_q8_bf16_reg_m4",
    "gemm_skinny_q8_bf16_reg_m5",
    "gemm_skinny_q8_bf16_reg_m6",
    "gemm_skinny_q8_bf16_reg_m7",
    "gemm_skinny_q8_bf16_reg_m8",
];
/// Vulkan: the same at one weight row per lane (`quant::rows_layout`).
#[cfg(target_os = "linux")]
const Q8_REG_FNS_R1: [&str; REG_MAX_M] = [
    "gemm_skinny_q8_bf16_reg_m1_r1",
    "gemm_skinny_q8_bf16_reg_m2_r1",
    "gemm_skinny_q8_bf16_reg_m3_r1",
    "gemm_skinny_q8_bf16_reg_m4_r1",
    "gemm_skinny_q8_bf16_reg_m5_r1",
    "gemm_skinny_q8_bf16_reg_m6_r1",
    "gemm_skinny_q8_bf16_reg_m7_r1",
    "gemm_skinny_q8_bf16_reg_m8_r1",
];
/// Vulkan: the register-A body over up to four weights in one dispatch, at
/// two weight rows per lane and at one (`gemm_skinny_q8_nt_sets`).
#[cfg(target_os = "linux")]
const Q8_SETS_FNS: [[&str; REG_MAX_M]; 2] = [
    [
        "gemm_skinny_q8_sets_m1_r1",
        "gemm_skinny_q8_sets_m2_r1",
        "gemm_skinny_q8_sets_m3_r1",
        "gemm_skinny_q8_sets_m4_r1",
        "gemm_skinny_q8_sets_m5_r1",
        "gemm_skinny_q8_sets_m6_r1",
        "gemm_skinny_q8_sets_m7_r1",
        "gemm_skinny_q8_sets_m8_r1",
    ],
    [
        "gemm_skinny_q8_sets_m1",
        "gemm_skinny_q8_sets_m2",
        "gemm_skinny_q8_sets_m3",
        "gemm_skinny_q8_sets_m4",
        "gemm_skinny_q8_sets_m5",
        "gemm_skinny_q8_sets_m6",
        "gemm_skinny_q8_sets_m7",
        "gemm_skinny_q8_sets_m8",
    ],
];
const Q8_REG_FNS_F32: [&str; REG_MAX_M] = [
    "gemm_skinny_q8_f32_reg_m1",
    "gemm_skinny_q8_f32_reg_m2",
    "gemm_skinny_q8_f32_reg_m3",
    "gemm_skinny_q8_f32_reg_m4",
    "gemm_skinny_q8_f32_reg_m5",
    "gemm_skinny_q8_f32_reg_m6",
    "gemm_skinny_q8_f32_reg_m7",
    "gemm_skinny_q8_f32_reg_m8",
];
const Q4_REG_FNS_F32: [&str; REG_MAX_M] = [
    "gemm_skinny_q4_f32_reg_m1",
    "gemm_skinny_q4_f32_reg_m2",
    "gemm_skinny_q4_f32_reg_m3",
    "gemm_skinny_q4_f32_reg_m4",
    "gemm_skinny_q4_f32_reg_m5",
    "gemm_skinny_q4_f32_reg_m6",
    "gemm_skinny_q4_f32_reg_m7",
    "gemm_skinny_q4_f32_reg_m8",
];

fn reg_grid(n: usize, simdgroups: usize, rows_per_sg: usize) -> Grid {
    Grid::Threadgroups {
        groups: (n.div_ceil(simdgroups * rows_per_sg), 1, 1),
        threadgroup: (32 * simdgroups, 1, 1),
    }
}

fn validate_q4(
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<(usize, usize, usize)> {
    let (n, k) = (w.out_features(), w.in_features());
    let m = a.shape()[0];
    ensure!(m > 0, "skinny GEMM needs at least one row");
    ensure!(
        m <= DENSE_SMALLM_THRESHOLD,
        "skinny GEMM is instantiated only through m={DENSE_SMALLM_THRESHOLD} (got {m})"
    );
    ensure!(w.bits == 4, "skinny quantized GEMM is 4-bit only");
    // The Metal body requires complete 64-element quantization groups.
    ensure!(
        w.group_size == 64 && k.is_multiple_of(64),
        "skinny GEMM requires group_size=64 and K % 64 == 0 (k={k}, gs={})",
        w.group_size
    );
    ensure!(a.shape() == [m, k], "A shape {:?} != [{m}, {k}]", a.shape());
    ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
    ensure!(w.codes.dtype() == DType::U32, "codes must be U32");
    ensure!(
        w.scales.shape() == [n, k / w.group_size]
            && w.biases.shape() == [n, k / w.group_size],
        "scales/biases shape mismatch for [{n}, {k}] gs={}",
        w.group_size
    );
    ensure!(a.dtype() == DType::BF16, "skinny q4 GEMM input must be BF16");
    ensure!(
        matches!(c.dtype(), DType::BF16 | DType::F32),
        "skinny q4 GEMM output must be BF16 or F32"
    );
    Ok((m, k, n))
}

/// Whether the register-A walk can consume complete quant groups.
pub fn q4_block_walk_ok(k: usize, group_size: usize) -> bool {
    group_size == 64 && k.is_multiple_of(64)
}

fn dispatch_q4_staged(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    let fn_name = match (m <= REG_MAX_M, c.dtype()) {
        (true, DType::F32) => "gemm_skinny_q4_f32_m8",
        (false, DType::F32) => "gemm_skinny_q4_f32_m16",
        (true, _) => "gemm_skinny_q4_bf16_m8",
        (false, _) => "gemm_skinny_q4_bf16_m16",
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size), &u32_bytes(m)],
        staged_grid(n),
    )
}

fn dispatch_q4_reg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    // The kernel addresses code words with 32-bit offsets.
    ensure!(
        n * (k / 8) <= u32::MAX as usize,
        "register-A q4 needs N * K / 8 < 2^32 (n = {n}, k = {k})"
    );
    let names = if c.dtype() == DType::F32 { &Q4_REG_FNS_F32 } else { &Q4_REG_FNS };
    let pipeline = ctx.pipeline(names[m - 1], SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        reg_grid(n, REG_SIMDGROUPS_PER_TG, Q4_REG_ROWS_PER_SG[m - 1]),
    )
}

/// Vulkan: the raw-code projection body (`vulkan/quant_rows.glsl`), with its
/// params and grid. The f32 kernels (the LM head) exist at two rows per lane
/// only.
#[cfg(target_os = "linux")]
fn dispatch_q8_reg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    names: &[&'static str; REG_MAX_M],
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    let (lanes_log2, rows) = crate::kernels::quant::rows_layout(w);
    let rows = if c.dtype() == DType::F32 { 2 } else { rows };
    let name = if rows == 1 { Q8_REG_FNS_R1[m - 1] } else { names[m - 1] };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[
            &u32_bytes(k),
            &u32_bytes(n),
            &u32_bytes(w.group_size),
            &u32_bytes(lanes_log2),
        ],
        crate::kernels::quant::rows_grid(n, lanes_log2, rows),
    )
}

#[cfg(not(target_os = "linux"))]
fn dispatch_q8_reg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    names: &[&'static str; REG_MAX_M],
    (m, k, n): (usize, usize, usize),
) -> Result<()> {
    let pipeline = ctx.pipeline(names[m - 1], SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        reg_grid(n, REG_SIMDGROUPS_PER_TG, REG_ROWS_PER_SG),
    )
}

/// Small-M affine-Q8 GEMM (the 8-bit tensors are the narrow routers, gates
/// and stream mixers, plus every dense projection and the LM head of a
/// `--q8-dense` checkpoint). The staged kernels have the numerics of the
/// bf16 dequant + GEMM fallback they replace; register-A (`m <= 8`, bf16
/// out, or f32 out at least `WIDE_N_MIN` wide: the LM head) dots the raw
/// codes like the decode GEMV, so verify logits match decode's numerics as
/// the Q4 head's do. The narrow f32 outputs (the routers) stay staged, as
/// every checkpoint has always run them.
pub fn gemm_skinny_q8_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<()> {
    let (n, k) = (w.out_features(), w.in_features());
    let m = a.shape()[0];
    ensure!(
        m > 0 && m <= DENSE_SMALLM_THRESHOLD,
        "skinny q8 GEMM handles 1..={DENSE_SMALLM_THRESHOLD} rows (got {m})"
    );
    ensure!(w.bits == 8, "skinny q8 GEMM needs 8-bit weights");
    ensure!(
        w.group_size.is_multiple_of(8) && k.is_multiple_of(8),
        "skinny q8 GEMM needs group_size % 8 == 0 and K % 8 == 0 (k={k}, gs={})",
        w.group_size
    );
    ensure!(a.shape() == [m, k], "A shape {:?} != [{m}, {k}]", a.shape());
    ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
    ensure!(a.dtype() == DType::BF16, "skinny q8 GEMM input must be BF16");
    // The Vulkan kernels take a block's quant group by a shift.
    #[cfg(target_os = "linux")]
    ensure!(
        w.group_size.is_power_of_two(),
        "group size {} is not a power of two",
        w.group_size
    );
    // Register-A for small m (no threadgroup staging or barriers: the
    // narrow, deep mixers and routers are latency-bound in the staged walk).
    if q8_reg_routes(m, w, c) {
        let names = if c.dtype() == DType::F32 { &Q8_REG_FNS_F32 } else { &Q8_REG_FNS };
        return dispatch_q8_reg(ctx, pass, a, w, c, names, (m, k, n));
    }
    let fn_name = match (m <= REG_MAX_M, c.dtype()) {
        (true, DType::F32) => "gemm_skinny_q8_f32_m8",
        (false, DType::F32) => "gemm_skinny_q8_f32_m16",
        (true, DType::BF16) => "gemm_skinny_q8_bf16_m8",
        (false, DType::BF16) => "gemm_skinny_q8_bf16_m16",
        _ => anyhow::bail!("skinny q8 GEMM output must be BF16 or F32"),
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size), &u32_bytes(m)],
        staged_grid(n),
    )
}

/// Whether `gemm_skinny_q8_nt` takes the register-A kernel for this weight
/// and output at `m` rows (the kernel whose rows a multi-weight dispatch
/// reproduces).
fn q8_reg_routes(m: usize, w: &QuantWeights, c: &Tensor) -> bool {
    let wide_enough = match c.dtype() {
        DType::BF16 => true,
        DType::F32 => w.out_features() >= WIDE_N_MIN,
        _ => false,
    };
    w.bits == 8
        && wide_enough
        && (1..=REG_MAX_M).contains(&m)
        && w.group_size.is_multiple_of(16)
        && w.in_features().is_multiple_of(16)
}

/// The small-M 8-bit GEMMs of one dependency level that read the same `a`
/// `[m, K]`: `C = A . W^T` for each `(W, C)` of `sets`. The dispatches of a
/// level overlap, on Metal and on the Vulkan driver lily runs on, and this is
/// `gemm_skinny_q8_nt` per set. On a Vulkan driver where they do not
/// (`level_dispatches_overlap`) it is `gemm_skinny_q8_nt_sets_fused`. Each
/// output is the same bits either way. Measured 2026-11-11 on the batched
/// router level (router, shared gate/up, shared gate; `verify_levels_timing`,
/// us per level at m = 1, 2, 3, 4, 8): 13.8/14.1/14.6/15.5/24.9 as one
/// multi-weight dispatch, 12.4/12.8/13.4/14.9/25.5 as three dispatches (the
/// multi-weight kernel picks each subgroup's weight with a dozen loads and
/// gives every weight the rows per lane of the largest).
pub fn gemm_skinny_q8_nt_sets(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    sets: &[(&QuantWeights, &Tensor)],
) -> Result<()> {
    #[cfg(target_os = "linux")]
    if !level_dispatches_overlap() {
        return gemm_skinny_q8_nt_sets_fused(ctx, pass, a, sets);
    }
    for (w, c) in sets {
        gemm_skinny_q8_nt(ctx, pass, a, w, c)?;
    }
    Ok(())
}

/// Vulkan: `gemm_skinny_q8_nt_sets` with the sets that take the register-A
/// kernel and share K and group size as one dispatch, up to four at a time
/// (`vulkan/quant_rows.glsl`, `QR_SETS`); each output is the bits
/// `gemm_skinny_q8_nt` gives it.
#[cfg(target_os = "linux")]
pub fn gemm_skinny_q8_nt_sets_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    sets: &[(&QuantWeights, &Tensor)],
) -> Result<()> {
    use crate::kernels::quant::{SETS_MAX, sets_dispatch_args};
    let m = a.shape()[0];
    let (reg, rest): (Vec<_>, Vec<_>) =
        sets.iter().copied().partition(|(w, c)| q8_reg_routes(m, w, c));
    let fused = reg.len() >= 2
        && reg.iter().all(|(w, _)| {
            w.in_features() == reg[0].0.in_features()
                && w.group_size == reg[0].0.group_size
        });
    for group in reg.chunks(if fused { SETS_MAX } else { 1 }) {
        if let [(w, c)] = group {
            gemm_skinny_q8_nt(ctx, pass, a, w, c)?;
            continue;
        }
        let k = group[0].0.in_features();
        ensure!(a.shape() == [m, k], "A shape {:?} != [{m}, {k}]", a.shape());
        ensure!(a.dtype() == DType::BF16, "skinny q8 GEMM input must be BF16");
        for (w, c) in group {
            let n = w.out_features();
            ensure!(c.numel() == m * n, "C numel {} != {m}x{n}", c.numel());
        }
        let (rows, buffers, params, grid) = sets_dispatch_args(a, group, None)?;
        let pipeline =
            ctx.pipeline(Q8_SETS_FNS[rows - 1][m - 1], SOURCE, MslVersion::V3_1)?;
        let params = params.map(u32::to_ne_bytes);
        let params: Vec<&[u8]> = params.iter().map(|p| &p[..]).collect();
        pass.dispatch_at(&pipeline, &buffers, &params, grid)?;
    }
    for (w, c) in rest {
        gemm_skinny_q8_nt(ctx, pass, a, w, c)?;
    }
    Ok(())
}

/// Small-M affine-Q4 GEMM. Register-A (`m <= 8`, `group_size == 64`,
/// `K % 64 == 0`) dots the raw codes with scale and bias applied per block,
/// like the decode GEMV; the staged fallback rounds dequantized weights to
/// bf16 first. `c` may be BF16 or F32 (the latter for logits); the f32
/// result is what the bf16 variant rounds.
pub fn gemm_skinny_q4_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<()> {
    let dims = validate_q4(a, w, c)?;
    let (m, k, n) = dims;
    if reg_routes(m, n, q4_block_walk_ok(k, w.group_size)) {
        return dispatch_q4_reg(ctx, pass, a, w, c, dims);
    }
    dispatch_q4_staged(ctx, pass, a, w, c, dims)
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/skinny.rs"]
mod tests;
