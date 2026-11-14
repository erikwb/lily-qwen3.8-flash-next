//! GLM 5.3 Flash kernels (`vulkan/glm53.comp`): GEMV over GGUF blocks with
//! f32 activations, routed experts, the router, hyper-connections, Kimi delta
//! attention, the DSA latent cache and attention, the pooled indexer.
//! The CPU reference is `crate::glm53::reference`.

use anyhow::{Result, ensure};

use crate::gguf::GgmlType;
use crate::kernels::{f32_bytes, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

/// The kernels are GLSL only so far: on macOS the Metal versions remain to
/// be written, and `ctx.pipeline` with this empty source fails there.
const SOURCE: &str = "";

/// Rows of activations a GEMV takes at once.
pub const GEMV_MAX_ROWS: usize = 4;

/// A GGUF weight on the GPU: `rows` rows of `cols` values of `ty`, read in
/// place (`t` is a byte view, I8, over the mapped file or a test upload).
#[derive(Clone)]
pub struct QWeight {
    pub t: Tensor,
    pub ty: GgmlType,
    pub cols: usize,
    pub rows: usize,
    pub row_bytes: usize,
}

impl QWeight {
    pub fn new(t: Tensor, ty: GgmlType, cols: usize, rows: usize) -> Result<Self> {
        let row_bytes = ty.row_bytes(cols)?;
        ensure!(
            t.byte_len() >= rows * row_bytes,
            "{} bytes hold no {rows} rows of {cols} {ty:?}",
            t.byte_len()
        );
        Ok(Self { t, ty, cols, rows, row_bytes })
    }

    /// Uploads raw blocks (tests).
    pub fn from_bytes(
        ctx: &MetalContext,
        bytes: &[u8],
        ty: GgmlType,
        cols: usize,
        rows: usize,
    ) -> Result<Self> {
        Self::new(
            Tensor::from_bytes(ctx, bytes, &[bytes.len()], DType::I8)?,
            ty,
            cols,
            rows,
        )
    }
}

fn gemv_name(ty: GgmlType) -> Result<&'static str> {
    Ok(match ty {
        GgmlType::Q8_0 => "glm53_gemv_q8_0",
        GgmlType::Q4K => "glm53_gemv_q4_k",
        GgmlType::Q2K => "glm53_gemv_q2_k",
        GgmlType::Iq2Xxs => "glm53_gemv_iq2_xxs",
        GgmlType::Bf16 => "glm53_gemv_bf16",
        GgmlType::F32 => "glm53_gemv_f32",
        GgmlType::F16 => anyhow::bail!("no GLM GEMV over F16"),
    })
}

fn moe_name(gate_up: bool, ty: GgmlType) -> Result<&'static str> {
    Ok(match (gate_up, ty) {
        (true, GgmlType::Iq2Xxs) => "glm53_moe_gate_up_iq2_xxs",
        (true, GgmlType::Q4K) => "glm53_moe_gate_up_q4_k",
        (true, GgmlType::Q2K) => "glm53_moe_gate_up_q2_k",
        (false, GgmlType::Q2K) => "glm53_moe_down_q2_k",
        (false, GgmlType::Q4K) => "glm53_moe_down_q4_k",
        (false, GgmlType::Iq2Xxs) => "glm53_moe_down_iq2_xxs",
        _ => anyhow::bail!("no GLM expert kernel for {ty:?}"),
    })
}

fn check_f32(ts: &[&Tensor]) -> Result<()> {
    for t in ts {
        ensure!(
            t.dtype() == DType::F32,
            "GLM kernels take f32 tensors, got {:?}",
            t.dtype()
        );
    }
    Ok(())
}

/// The GEMV's layout of its activations and outputs beyond the plain
/// `y[i] = W x[i]`.
#[derive(Clone, Copy)]
pub struct GemvLayout {
    /// Output rows that share one input vector (per-head projections); the
    /// input of row `n` is vector `n / x_group` of the activation row.
    pub x_group: usize,
    /// f32 elements between activation rows.
    pub x_row_stride: usize,
    /// f32 elements between output rows.
    pub y_row_stride: usize,
}

/// `y[i][n] = w[n] . x[i]` for `i < m` (`m <= 4`), `x` rows of `w.cols`.
pub fn gemv(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
) -> Result<()> {
    let layout =
        GemvLayout { x_group: w.rows, x_row_stride: w.cols, y_row_stride: w.rows };
    gemv_with(ctx, pass, w, x, y, m, layout)
}

/// [`gemv`] with an explicit activation/output layout.
pub fn gemv_with(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QWeight,
    x: &Tensor,
    y: &Tensor,
    m: usize,
    l: GemvLayout,
) -> Result<()> {
    gemv_dispatch(ctx, pass, &[(w, y)], x, m, l)
}

/// One dispatch of up to four GEMVs of one weight type and width over the
/// same input `x` (`m` rows): `y_s[i] = w_s x[i]`. On this driver
/// dispatches of one dependency level run one after another, so a level's
/// projections are best one dispatch.
pub fn gemv_sets(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    sets: &[(&QWeight, &Tensor)],
    x: &Tensor,
    m: usize,
) -> Result<()> {
    let rows = sets.iter().map(|(w, _)| w.rows).max().unwrap_or(0);
    let cols = sets.first().map_or(0, |(w, _)| w.cols);
    let l = GemvLayout { x_group: rows.max(1), x_row_stride: cols, y_row_stride: 0 };
    gemv_dispatch(ctx, pass, sets, x, m, l)
}

fn gemv_dispatch(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    sets: &[(&QWeight, &Tensor)],
    x: &Tensor,
    m: usize,
    l: GemvLayout,
) -> Result<()> {
    ensure!((1..=4).contains(&sets.len()), "GEMV of {} sets", sets.len());
    ensure!((1..=GEMV_MAX_ROWS).contains(&m), "GEMV of {m} rows");
    let w0 = sets[0].0;
    check_f32(&[x])?;
    let tasks = w0.cols / 32;
    ensure!(
        w0.cols.is_multiple_of(32) && (tasks >= 32 || tasks.is_power_of_two()),
        "GEMV row of {} values",
        w0.cols
    );
    let per_sg = 32 / tasks.min(32);
    let rows_per_tg = 8 * per_sg;
    let mut groups = 0;
    for &(w, y) in sets {
        ensure!(
            w.ty == w0.ty && w.cols == w0.cols && w.row_bytes == w0.row_bytes,
            "GEMV sets of different types or widths"
        );
        check_f32(&[y])?;
        let stride = if l.y_row_stride == 0 { w.rows } else { l.y_row_stride };
        ensure!(y.numel() >= (m - 1) * stride + w.rows, "GEMV output of {}", y.numel());
        let x_groups = w.rows.div_ceil(l.x_group);
        ensure!(
            x.numel() >= (m - 1) * l.x_row_stride + x_groups * w.cols,
            "GEMV input of {} values",
            x.numel()
        );
        groups += w.rows.div_ceil(rows_per_tg);
    }
    let mut bindings = vec![x.binding()];
    for s in 0..4 {
        bindings.push(sets.get(s).map_or(x.binding(), |(w, _)| w.t.binding()));
    }
    for s in 0..4 {
        bindings.push(sets.get(s).map_or(x.binding(), |(_, y)| y.binding()));
    }
    let rows = |s: usize| u32_bytes(sets.get(s).map_or(0, |(w, _)| w.rows));
    let pipeline = ctx.pipeline(gemv_name(w0.ty)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &bindings,
        &[
            &u32_bytes(w0.cols),
            &u32_bytes(m),
            &u32_bytes(w0.row_bytes),
            &rows(0),
            &rows(1),
            &rows(2),
            &rows(3),
            &u32_bytes(l.x_group),
            &u32_bytes(l.x_row_stride),
            &u32_bytes(l.y_row_stride),
        ],
        Grid::Threadgroups { groups: (groups, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Routed experts' gate and up with the clamped SwiGLU, weighted:
/// `mid[i * used + s] = w[i][s] * swiglu(gate_e x[i], up_e x[i])` for the
/// `used` experts `sel[i]` of each of the `m` rows.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QWeight,
    up: &QWeight,
    ff: usize,
    x: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    mid: &Tensor,
    m: usize,
    used: usize,
    limit: f32,
) -> Result<()> {
    check_f32(&[x, weights, mid])?;
    ensure!(gate.ty == up.ty && gate.cols == up.cols, "gate and up experts differ");
    ensure!(gate.rows.is_multiple_of(ff), "expert rows {} of {ff}", gate.rows);
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * used, "selection");
    ensure!(mid.numel() >= m * used * ff, "mid of {}", mid.numel());
    let name = moe_name(true, gate.ty)?;
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.t.binding(),
            up.t.binding(),
            x.binding(),
            sel.binding(),
            weights.binding(),
            mid.binding(),
        ],
        &[
            &u32_bytes(gate.cols),
            &u32_bytes(ff),
            &u32_bytes(gate.row_bytes),
            &u32_bytes(used),
            &f32_bytes(limit),
        ],
        Grid::Threadgroups {
            groups: (ff.div_ceil(8), m * used, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// Routed experts' down projection summed over each row's slots, plus
/// `add` when given: `out[i] = sum_s down_{sel[i][s]} mid[i * used + s]
/// (+ add[i])`; `out` may be `add`.
#[allow(clippy::too_many_arguments)]
pub fn moe_down(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    down: &QWeight,
    hidden: usize,
    mid: &Tensor,
    sel: &Tensor,
    add: Option<&Tensor>,
    out: &Tensor,
    m: usize,
    used: usize,
) -> Result<()> {
    check_f32(&[mid, out])?;
    ensure!(down.rows.is_multiple_of(hidden), "down rows {} of {hidden}", down.rows);
    ensure!(out.numel() >= m * hidden, "out of {}", out.numel());
    let name = moe_name(false, down.ty)?;
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            down.t.binding(),
            mid.binding(),
            sel.binding(),
            out.binding(),
            add.unwrap_or(out).binding(),
        ],
        &[
            &u32_bytes(down.cols),
            &u32_bytes(hidden),
            &u32_bytes(down.row_bytes),
            &u32_bytes(used),
            &u32_bytes(usize::from(add.is_some())),
        ],
        Grid::Threadgroups {
            groups: (hidden.div_ceil(8), m, 1),
            threadgroup: (256, 1, 1),
        },
    )
}

/// The router: top `k` of `sigmoid(logits) + bias` per row, weights the
/// selected probabilities normalized times `scale`.
#[allow(clippy::too_many_arguments)]
pub fn router_topk(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    bias: &Tensor,
    sel: &Tensor,
    weights: &Tensor,
    m: usize,
    n_expert: usize,
    k: usize,
    scale: f32,
) -> Result<()> {
    check_f32(&[logits, bias, weights])?;
    ensure!(n_expert <= 384 && k <= 32, "router of {n_expert} experts, top {k}");
    let pipeline = ctx.pipeline("glm53_router_topk", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), bias.binding(), sel.binding(), weights.binding()],
        &[&u32_bytes(n_expert), &u32_bytes(k), &f32_bytes(scale)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// RMSNorm of `m` rows of `n`, times `w` when given; `out` may be `x`.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: Option<&Tensor>,
    out: &Tensor,
    m: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    check_f32(&[x, out])?;
    ensure!(x.numel() >= m * n && out.numel() >= m * n, "rmsnorm of {m} x {n}");
    let pipeline = ctx.pipeline("glm53_rmsnorm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.unwrap_or(x).binding(), out.binding()],
        &[&u32_bytes(n), &f32_bytes(eps), &u32_bytes(usize::from(w.is_some()))],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// HC pre after the control projection: Sinkhorn split of `mix` into
/// `split` (pre, post, comb) and `xn = rmsnorm(sum_h pre[h] * h_streams[h],
/// norm)`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    mix: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    streams: &Tensor,
    norm: &Tensor,
    split: &Tensor,
    xn: &Tensor,
    m: usize,
    hidden: usize,
    iters: usize,
    hc_eps: f32,
    rms_eps: f32,
) -> Result<()> {
    check_f32(&[mix, scale, base, streams, norm, split, xn])?;
    ensure!(
        streams.numel() >= m * 4 * hidden && mix.numel() >= m * 24,
        "hc_pre shapes"
    );
    let pipeline = ctx.pipeline("glm53_hc_pre", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            mix.binding(),
            scale.binding(),
            base.binding(),
            streams.binding(),
            norm.binding(),
            split.binding(),
            xn.binding(),
        ],
        &[
            &u32_bytes(hidden),
            &u32_bytes(iters),
            &f32_bytes(hc_eps),
            &f32_bytes(rms_eps),
        ],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// HC post: `out[dst] = block * post[dst] + sum_src comb[dst + 4 src] *
/// streams[src]`; `out` may be `streams`.
#[allow(clippy::too_many_arguments)]
pub fn hc_post(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    streams: &Tensor,
    block: &Tensor,
    split: &Tensor,
    out: &Tensor,
    m: usize,
    hidden: usize,
) -> Result<()> {
    check_f32(&[streams, block, split, out])?;
    let pipeline = ctx.pipeline("glm53_hc_post", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[streams.binding(), block.binding(), split.binding(), out.binding()],
        &[&u32_bytes(hidden)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Threadgroups (and partial sums per row) of [`hc_post_mix`].
pub fn hc_mix_groups(hidden: usize) -> usize {
    hidden / 128
}

/// HC post of the previous sublayer (when `post` gives its output and the
/// split holding its post/comb weights; in place on `streams`) fused with
/// the partial sums of the next control projection `fn_w . streams` and of
/// `sum(streams^2)`, `[m][groups][25]` into `partial`.
#[allow(clippy::too_many_arguments)]
pub fn hc_post_mix(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    streams: &Tensor,
    post: Option<(&Tensor, &Tensor)>,
    fn_w: &QWeight,
    partial: &Tensor,
    m: usize,
    hidden: usize,
) -> Result<()> {
    check_f32(&[streams, partial])?;
    ensure!(
        fn_w.ty == GgmlType::Bf16 && fn_w.rows == 24 && fn_w.cols == 4 * hidden,
        "HC fn shape"
    );
    ensure!(hidden.is_multiple_of(128), "hidden {hidden}");
    ensure!(partial.numel() >= m * hc_mix_groups(hidden) * 25, "partials");
    let (block, split) = post.unwrap_or((streams, streams));
    let pipeline = ctx.pipeline("glm53_hc_post_mix", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            streams.binding(),
            block.binding(),
            split.binding(),
            fn_w.t.binding(),
            partial.binding(),
        ],
        &[&u32_bytes(hidden), &u32_bytes(usize::from(post.is_some()))],
        Grid::Threadgroups {
            groups: (hc_mix_groups(hidden), m, 1),
            threadgroup: (64, 1, 1),
        },
    )
}

/// HC pre from [`hc_post_mix`]'s partial sums: the split into `split`
/// (pre, post, comb) and `xn = rmsnorm(sum_h pre[h] * streams[h], norm)`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre2(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    partial: &Tensor,
    scale: &Tensor,
    base: &Tensor,
    streams: &Tensor,
    norm: &Tensor,
    split: &Tensor,
    xn: &Tensor,
    m: usize,
    hidden: usize,
    iters: usize,
    hc_eps: f32,
    rms_eps: f32,
) -> Result<()> {
    check_f32(&[partial, scale, base, streams, norm, split, xn])?;
    ensure!(hidden == 4096, "hc_pre2 is written for hidden 4096, not {hidden}");
    let pipeline = ctx.pipeline("glm53_hc_pre2", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            partial.binding(),
            scale.binding(),
            base.binding(),
            streams.binding(),
            norm.binding(),
            split.binding(),
            xn.binding(),
        ],
        &[
            &u32_bytes(hidden),
            &u32_bytes(iters),
            &f32_bytes(hc_eps),
            &f32_bytes(rms_eps),
            &u32_bytes(hc_mix_groups(hidden)),
        ],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// The mean of the four streams.
pub fn hc_mean(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    streams: &Tensor,
    out: &Tensor,
    m: usize,
    hidden: usize,
) -> Result<()> {
    check_f32(&[streams, out])?;
    let pipeline = ctx.pipeline("glm53_hc_mean", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[streams.binding(), out.binding()],
        &[&u32_bytes(hidden)],
        Grid::Threads { grid: (hidden, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// Embeds `m` tokens (U32) from a Q8_0 table into four streams each.
pub fn embed(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    table: &QWeight,
    tokens: &Tensor,
    streams: &Tensor,
    m: usize,
) -> Result<()> {
    ensure!(table.ty == GgmlType::Q8_0, "embedding table of {:?}", table.ty);
    check_f32(&[streams])?;
    let pipeline = ctx.pipeline("glm53_embed_q8_0", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[table.t.binding(), tokens.binding(), streams.binding()],
        &[&u32_bytes(table.cols), &u32_bytes(table.row_bytes)],
        Grid::Threads { grid: (table.cols, m, 1), threadgroup: (256, 1, 1) },
    )
}

/// `out = swiglu(g, u)` over `n` values with GLM's clamp.
pub fn swiglu(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    g: &Tensor,
    u: &Tensor,
    out: &Tensor,
    n: usize,
    limit: f32,
) -> Result<()> {
    check_f32(&[g, u, out])?;
    let pipeline = ctx.pipeline("glm53_swiglu", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[g.binding(), u.binding(), out.binding()],
        &[&f32_bytes(limit)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// `out = a + b` over `n` values.
pub fn add(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    n: usize,
) -> Result<()> {
    check_f32(&[a, b, out])?;
    let pipeline = ctx.pipeline("glm53_add", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[a.binding(), b.binding(), out.binding()],
        &[],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// One layer's KDA parameters on the GPU (f32).
pub struct KdaParams<'a> {
    pub q_conv: &'a Tensor,
    pub k_conv: &'a Tensor,
    pub v_conv: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub o_norm: &'a Tensor,
    pub heads: usize,
    pub lower_bound: f32,
    pub eps: f32,
}

/// The raw projections KDA consumes, `[rows][projection]` (`beta`
/// `[rows][heads]`); the prefill path overwrites q, k, v and gate in place.
pub struct KdaInputs<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub gate: &'a Tensor,
    pub beta: &'a Tensor,
    pub out_gate: &'a Tensor,
}

/// Where a run of KDA tokens also leaves the conv window and state as they
/// were after one of its rows (speculative decoding's rollback point).
pub struct KdaSnapshot<'a> {
    pub row: usize,
    pub conv: &'a Tensor,
    pub state: &'a Tensor,
}

/// KDA for `rows` consecutive tokens of one sequence: the one-token kernel
/// for a single row without a snapshot, else prepare / recurrence / output.
#[allow(clippy::too_many_arguments)]
pub fn kda(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &KdaParams<'_>,
    x: &KdaInputs<'_>,
    conv_state: &Tensor,
    state: &Tensor,
    out: &Tensor,
    rows: usize,
    snap: Option<KdaSnapshot<'_>>,
) -> Result<()> {
    let (snap_row, snap_conv, snap_state) = match &snap {
        Some(s) => {
            ensure!(s.row < rows, "KDA snapshot after row {} of {rows}", s.row);
            (s.row, s.conv, s.state)
        }
        None => (u32::MAX as usize, conv_state, state),
    };
    check_f32(&[x.q, x.k, x.v, x.gate, x.beta, x.out_gate, conv_state, state, out])?;
    let proj = p.heads * 128;
    ensure!(
        state.numel() >= p.heads * 128 * 128 && conv_state.numel() >= 9 * proj,
        "KDA state"
    );
    if rows == 1 && snap.is_none() {
        let pipeline = ctx.pipeline("glm53_kda_decode", SOURCE, MslVersion::V3_1)?;
        return pass.dispatch_at(
            &pipeline,
            &[
                x.q.binding(),
                x.k.binding(),
                x.v.binding(),
                x.gate.binding(),
                x.beta.binding(),
                x.out_gate.binding(),
                p.q_conv.binding(),
                p.k_conv.binding(),
                p.v_conv.binding(),
                p.a_log.binding(),
                p.dt_bias.binding(),
                p.o_norm.binding(),
                conv_state.binding(),
                state.binding(),
                out.binding(),
            ],
            &[&u32_bytes(p.heads), &f32_bytes(p.lower_bound), &f32_bytes(p.eps)],
            Grid::Threadgroups { groups: (1, p.heads, 1), threadgroup: (128, 1, 1) },
        );
    }
    let prepare =
        ctx.pipeline("glm53_kda_prefill_prepare", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &prepare,
        &[
            x.q.binding(),
            x.k.binding(),
            x.v.binding(),
            x.gate.binding(),
            p.q_conv.binding(),
            p.k_conv.binding(),
            p.v_conv.binding(),
            p.a_log.binding(),
            p.dt_bias.binding(),
            conv_state.binding(),
            snap_conv.binding(),
        ],
        &[
            &u32_bytes(p.heads),
            &u32_bytes(rows),
            &f32_bytes(p.lower_bound),
            &u32_bytes(snap_row),
        ],
        Grid::Threadgroups { groups: (p.heads, 1, 1), threadgroup: (128, 1, 1) },
    )?;
    pass.memory_barrier()?;
    let recurrence =
        ctx.pipeline("glm53_kda_prefill_recurrence", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &recurrence,
        &[
            x.q.binding(),
            x.k.binding(),
            x.v.binding(),
            x.gate.binding(),
            x.beta.binding(),
            state.binding(),
            out.binding(),
            snap_state.binding(),
        ],
        &[&u32_bytes(p.heads), &u32_bytes(rows), &u32_bytes(snap_row)],
        Grid::Threadgroups { groups: (p.heads, 32, 1), threadgroup: (128, 1, 1) },
    )?;
    pass.memory_barrier()?;
    let output = ctx.pipeline("glm53_kda_prefill_output", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &output,
        &[out.binding(), x.out_gate.binding(), p.o_norm.binding()],
        &[&u32_bytes(p.heads), &f32_bytes(p.eps)],
        Grid::Threadgroups { groups: (rows, p.heads, 1), threadgroup: (128, 1, 1) },
    )
}

/// Stores `rows` normalized latent rows (`rmsnorm(kv_raw, w)`, 512 wide) as
/// f16 at cache rows `pos0..`.
#[allow(clippy::too_many_arguments)]
pub fn kv_store(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    kv_raw: &Tensor,
    w: &Tensor,
    cache: &Tensor,
    pos0: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
    check_f32(&[kv_raw, w])?;
    ensure!(
        cache.dtype() == DType::F16 && cache.numel() >= (pos0 + rows) * 512,
        "latent cache"
    );
    let pipeline = ctx.pipeline("glm53_kv_store", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[kv_raw.binding(), w.binding(), cache.binding()],
        &[&u32_bytes(pos0), &f32_bytes(eps)],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (128, 1, 1) },
    )
}

/// Which cache rows a DSA query attends.
#[derive(Clone, Copy)]
pub enum Rows<'t> {
    /// Causal and dense: row `i` of the batch sees `0..=pos0 + i`.
    Causal { pos0: usize },
    /// The first `count` entries of each row of `sel` (U32 `[m][s]`).
    Selected { sel: &'t Tensor, count: usize },
}

/// Absorbed DSA attention core over the latent cache: from `qk_low`
/// (`[m][64][512]`) to the mixed latent `lat` (`[m][64][512]`), with scratch
/// `scores` (`[m][64][s]`) and `denom` (`[m][64]`); `s` bounds every row's
/// count.
#[allow(clippy::too_many_arguments)]
pub fn dsa_attend(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qk_low: &Tensor,
    cache: &Tensor,
    rows: Rows<'_>,
    scores: &Tensor,
    denom: &Tensor,
    lat: &Tensor,
    m: usize,
    s: usize,
    scale: f32,
) -> Result<()> {
    check_f32(&[qk_low, scores, denom, lat])?;
    ensure!(scores.numel() >= m * 64 * s, "scores of {}", scores.numel());
    let (pos0, sel, has_sel, n_sel) = match rows {
        Rows::Causal { pos0 } => {
            ensure!(pos0 + m <= s, "causal rows past the scores' width {s}");
            (pos0, qk_low, 0usize, 0usize)
        }
        Rows::Selected { sel, count } => {
            ensure!(count <= s && sel.numel() >= m * s, "selection");
            (0, sel, 1, count)
        }
    };
    let scores_k = ctx.pipeline("glm53_dsa_scores", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scores_k,
        &[qk_low.binding(), cache.binding(), sel.binding(), scores.binding()],
        &[
            &u32_bytes(s),
            &u32_bytes(pos0),
            &u32_bytes(has_sel),
            &u32_bytes(n_sel),
            &f32_bytes(scale),
        ],
        Grid::Threadgroups { groups: (s.div_ceil(8), m, 1), threadgroup: (64, 1, 1) },
    )?;
    pass.memory_barrier()?;
    let softmax = ctx.pipeline("glm53_dsa_softmax", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &softmax,
        &[scores.binding(), denom.binding()],
        &[&u32_bytes(s)],
        Grid::Threadgroups { groups: (64, m, 1), threadgroup: (256, 1, 1) },
    )?;
    pass.memory_barrier()?;
    let lora = ctx.pipeline("glm53_dsa_lora", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &lora,
        &[
            scores.binding(),
            denom.binding(),
            cache.binding(),
            sel.binding(),
            lat.binding(),
        ],
        &[&u32_bytes(s), &u32_bytes(pos0), &u32_bytes(has_sel), &u32_bytes(n_sel)],
        Grid::Threadgroups { groups: (64, 8, m), threadgroup: (64, 1, 1) },
    )
}

/// The pooled indexer's tensors of one layer.
pub struct PoolParams<'a> {
    pub norm_w: &'a Tensor,
    pub norm_b: &'a Tensor,
    /// BF16 `[4][128]`, as a byte view.
    pub ape: &'a Tensor,
}

/// Feeds `n_tokens` raw indexer key/gate rows at positions `pos0..` into
/// the pooled key cache (f16 `[pools][128]`) and the tail buffers.
#[allow(clippy::too_many_arguments)]
pub fn indexer_pool_update(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    p: &PoolParams<'_>,
    raw_k: &Tensor,
    gate: &Tensor,
    pooled: &Tensor,
    tail_k: &Tensor,
    tail_g: &Tensor,
    pos0: usize,
    n_tokens: usize,
) -> Result<()> {
    check_f32(&[raw_k, gate, tail_k, tail_g])?;
    ensure!(pooled.dtype() == DType::F16, "pooled cache dtype");
    ensure!(pooled.numel() >= (pos0 + n_tokens) / 4 * 128, "pooled cache too small");
    let pipeline =
        ctx.pipeline("glm53_indexer_pool_update", SOURCE, MslVersion::V3_1)?;
    // As ds4 does: a leading partial pool, which reads the tail buffers, is
    // its own dispatch before the rest, which may write them (the pass
    // orders its dispatches).
    let lead = if pos0 % 4 == 0 { 0 } else { (4 - pos0 % 4).min(n_tokens) };
    let mut done = 0;
    for chunk in [lead, n_tokens - lead] {
        if chunk == 0 {
            continue;
        }
        let at = pos0 + done;
        let k = raw_k.view(done * 128, &[chunk * 128])?;
        let g = gate.view(done * 128, &[chunk * 128])?;
        pass.dispatch_at(
            &pipeline,
            &[
                k.binding(),
                g.binding(),
                p.norm_w.binding(),
                p.norm_b.binding(),
                p.ape.binding(),
                pooled.binding(),
                tail_k.binding(),
                tail_g.binding(),
            ],
            &[&u32_bytes(at), &u32_bytes(chunk), &f32_bytes(1e-6)],
            Grid::Threadgroups {
                groups: (chunk.div_ceil(4), 1, 1),
                threadgroup: (128, 1, 1),
            },
        )?;
        done += chunk;
        pass.memory_barrier()?;
    }
    Ok(())
}

/// Indexer scores of the complete pools for `m` query rows at positions
/// `pos0..`: `scores[i][p]` (`[m][pools_cap]`, -inf past the row's pools)
/// from `iq` (`[m][32][128]`) and the head weights `w` (`[m][32]`).
#[allow(clippy::too_many_arguments)]
pub fn indexer_scores(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    iq: &Tensor,
    w: &Tensor,
    pooled: &Tensor,
    scores: &Tensor,
    m: usize,
    pools_cap: usize,
    pos0: usize,
    scale: f32,
) -> Result<()> {
    check_f32(&[iq, w, scores])?;
    ensure!((pos0 + m) / 4 <= pools_cap, "pool scores of {pools_cap} pools");
    ensure!(scores.numel() >= m * pools_cap, "scores of {}", scores.numel());
    let pipeline = ctx.pipeline("glm53_indexer_scores", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[iq.binding(), w.binding(), pooled.binding(), scores.binding()],
        &[&u32_bytes(pools_cap), &u32_bytes(pos0), &f32_bytes(scale)],
        Grid::Threadgroups {
            groups: (pools_cap.div_ceil(8), m, 1),
            threadgroup: (32, 1, 1),
        },
    )
}

/// The rows a DSA query attends past the dense limit: the token rows of its
/// `k` best pools (ascending) and of its incomplete pool, the rest of the
/// `s` slots marked unused; `count[i]` gets the row count.
#[allow(clippy::too_many_arguments)]
pub fn indexer_topk(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    scores: &Tensor,
    sel: &Tensor,
    count: &Tensor,
    m: usize,
    pools_cap: usize,
    s: usize,
    pos0: usize,
    k: usize,
) -> Result<()> {
    ensure!(sel.dtype() == DType::U32 && sel.numel() >= m * s, "selection");
    ensure!(4 * k + 3 <= s, "selection rows {s} for top {k} pools");
    let pipeline = ctx.pipeline("glm53_indexer_topk", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[scores.binding(), sel.binding(), count.binding()],
        &[&u32_bytes(pools_cap), &u32_bytes(s), &u32_bytes(pos0), &u32_bytes(k)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (1024, 1, 1) },
    )
}

#[cfg(test)]
#[path = "../../tests/unit/kernels/glm53.rs"]
mod tests;
