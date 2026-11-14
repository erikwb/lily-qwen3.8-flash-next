//! The CPU reference of GLM 5.3 Flash: every operation of the graph in plain
//! f32 (dot products accumulate in f64), written for clarity, not speed. The
//! GPU kernels are tested against these functions, and [`RefModel`] runs the
//! whole model layer by layer over a prompt on real weights so the GPU path
//! can be compared with it layer by layer.
//!
//! The semantics are ds4's (antirez, MIT): `hc_split_sinkhorn_one`,
//! `hc_weighted_sum_one` and `hc_post_one` (ds4.c), the GLM 5.3 KDA kernels
//! (`metal/glm53_kda.metal`), the pooled indexer
//! (`kernel_glm53_indexer_pool_update`, `kernel_glm53_expand_pool_selection`,
//! `kernel_glm_indexer_score_one`), the absorbed DSA attention
//! (`kernel_glm_attention_indexed_decode_exact_*`), the router
//! (`layer_glm_router_selected_experts`) and the routed/shared experts
//! (`layer_glm_routed_moe_one_f32_ref`, `swiglu`). Where ds4 stores a cache in
//! f16 (the DSA latent rows and the pooled indexer keys) the reference rounds
//! to f16 too.

use std::ops::Range;

use rayon::prelude::*;

use anyhow::{Context as _, Result, ensure};
use half::f16;

use super::config::{Config, LayerKind};
use crate::gguf::Gguf;

/// Where the reference reads weights from: decoded rows of a named tensor.
pub trait WeightSource {
    /// `(row length, rows)` of tensor `name`.
    fn shape(&self, name: &str) -> Result<(usize, usize)>;
    /// Rows `rows` of `name`, decoded to f32, row-major.
    fn rows(&self, name: &str, rows: Range<usize>) -> Result<Vec<f32>>;
}

impl WeightSource for Gguf {
    fn shape(&self, name: &str) -> Result<(usize, usize)> {
        let t = self.require(name)?;
        Ok((t.row_len(), t.rows()))
    }

    fn rows(&self, name: &str, rows: Range<usize>) -> Result<Vec<f32>> {
        let t = self.require(name)?;
        self.read_rows_f32(t, rows)
    }
}

/// A dense matrix, `rows x cols`, row-major (a GGUF tensor's rows).
#[derive(Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
}

impl Mat {
    pub fn load(src: &dyn WeightSource, name: &str) -> Result<Self> {
        let (cols, rows) = src.shape(name)?;
        let w = src.rows(name, 0..rows).with_context(|| format!("reading {name}"))?;
        Ok(Self { rows, cols, w })
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.w[r * self.cols..(r + 1) * self.cols]
    }

    /// `y = W x`.
    pub fn mv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols, "matvec width");
        (0..self.rows).into_par_iter().map(|r| dot(self.row(r), x)).collect()
    }

    /// `y[r] = W[r] . x[r / rows_per_x]`: per-head projections whose input
    /// differs per group of rows (`attn_k_b`, `attn_v_b`).
    pub fn mv_grouped(&self, x: &[f32], rows_per_x: usize) -> Vec<f32> {
        (0..self.rows)
            .into_par_iter()
            .map(|r| {
                let g = r / rows_per_x;
                dot(self.row(r), &x[g * self.cols..(g + 1) * self.cols])
            })
            .collect()
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum::<f64>() as f32
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn round_f16(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// `x / sqrt(mean(x^2) + eps)`, times `w` when given.
pub fn rmsnorm(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let scale = 1.0 / ((ss / x.len() as f64) as f32 + eps).sqrt();
    match w {
        Some(w) => x.iter().zip(w).map(|(&v, &g)| v * scale * g).collect(),
        None => x.iter().map(|&v| v * scale).collect(),
    }
}

/// The hyper-connection control split (ds4 `hc_split_sinkhorn_one`):
/// `mix[2n + n*n]` -> `pre[n]`, `post[n]`, `comb[n*n]` (`comb[dst + n*src]`
/// is read as the weight of stream `src` in output stream `dst`, ds4's
/// `[dst_hc, src_hc]` addressing).
pub fn hc_split_sinkhorn(
    mix: &[f32],
    scale: &[f32],
    base: &[f32],
    n: usize,
    iters: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let pre: Vec<f32> =
        (0..n).map(|i| sigmoid(mix[i] * scale[0] + base[i]) + eps).collect();
    let post: Vec<f32> = (0..n)
        .map(|i| 2.0 / (1.0 + (-(mix[n + i] * scale[1] + base[n + i])).exp()))
        .collect();
    let mut c = vec![0f32; n * n];
    // Row softmax over src for each dst, plus eps.
    for dst in 0..n {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n {
            let idx = src + dst * n;
            c[idx] = mix[2 * n + idx] * scale[2] + base[2 * n + idx];
            row_max = row_max.max(c[idx]);
        }
        let mut sum = 0f32;
        for src in 0..n {
            let idx = src + dst * n;
            c[idx] = (c[idx] - row_max).exp();
            sum += c[idx];
        }
        let inv = 1.0 / sum;
        for src in 0..n {
            c[src + dst * n] = c[src + dst * n] * inv + eps;
        }
    }
    let col_norm = |c: &mut [f32]| {
        for src in 0..n {
            let sum: f32 = (0..n).map(|dst| c[src + dst * n]).sum();
            let inv = 1.0 / (sum + eps);
            for dst in 0..n {
                c[src + dst * n] *= inv;
            }
        }
    };
    let row_norm = |c: &mut [f32]| {
        for dst in 0..n {
            let sum: f32 = (0..n).map(|src| c[src + dst * n]).sum();
            let inv = 1.0 / (sum + eps);
            for src in 0..n {
                c[src + dst * n] *= inv;
            }
        }
    };
    col_norm(&mut c);
    for _ in 1..iters {
        row_norm(&mut c);
        col_norm(&mut c);
    }
    (pre, post, c)
}

/// HC pre: collapse the streams `h[n * hidden]` into the sublayer input,
/// normalized with `norm`. Returns `(normed input, post, comb)`.
#[allow(clippy::too_many_arguments)]
pub fn hc_pre(
    cfg: &Config,
    h: &[f32],
    fn_w: &Mat,
    scale: &[f32],
    base: &[f32],
    norm: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = cfg.n_hc;
    let flat = rmsnorm(h, None, cfg.rms_eps);
    let mix = fn_w.mv(&flat);
    let (pre, post, comb) =
        hc_split_sinkhorn(&mix, scale, base, n, cfg.sinkhorn_iters, cfg.hc_eps);
    let x: Vec<f32> = (0..cfg.hidden)
        .map(|d| (0..n).map(|s| h[s * cfg.hidden + d] * pre[s]).sum())
        .collect();
    (rmsnorm(&x, Some(norm), cfg.rms_eps), post, comb)
}

/// HC post: `out[dst][d] = block[d] * post[dst] + sum_src comb[dst + n*src]
/// * h[src][d]`.
pub fn hc_post(
    h: &[f32],
    block: &[f32],
    post: &[f32],
    comb: &[f32],
    n: usize,
) -> Vec<f32> {
    let hidden = block.len();
    let mut out = vec![0f32; n * hidden];
    for dst in 0..n {
        for d in 0..hidden {
            let mut acc = block[d] * post[dst];
            for src in 0..n {
                acc += comb[dst + src * n] * h[src * hidden + d];
            }
            out[dst * hidden + d] = acc;
        }
    }
    out
}

/// SwiGLU with GLM's clamp: `silu(min(g, limit)) * clamp(u, -limit, limit)`.
pub fn swiglu(g: f32, u: f32, limit: f32) -> f32 {
    let (g, u) =
        if limit > 1e-6 { (g.min(limit), u.clamp(-limit, limit)) } else { (g, u) };
    silu(g) * u
}

/// The router: sigmoid scores, top-k of score + bias, weights the selected
/// scores normalized and scaled. Ties go to the lower expert id.
pub fn route(
    logits: &[f32],
    bias: &[f32],
    k: usize,
    scale: f32,
) -> (Vec<usize>, Vec<f32>) {
    let probs: Vec<f32> = logits.iter().map(|&l| sigmoid(l)).collect();
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        (probs[b] + bias[b]).total_cmp(&(probs[a] + bias[a])).then(a.cmp(&b))
    });
    let sel: Vec<usize> = order[..k].to_vec();
    let sum: f32 = sel.iter().map(|&e| probs[e]).sum::<f32>().max(6.103_515_6e-5);
    let w = sel.iter().map(|&e| probs[e] / sum * scale).collect();
    (sel, w)
}

/// One KDA layer's recurrent state.
#[derive(Clone)]
pub struct KdaState {
    /// `[q, k, v][3 history rows, oldest first][projection]`.
    pub conv: Vec<f32>,
    /// `[head][value row][key column]`.
    pub s: Vec<f32>,
}

impl KdaState {
    pub fn new(cfg: &Config) -> Self {
        let p = cfg.kda_projection();
        Self {
            conv: vec![0.0; 3 * (cfg.kda_conv - 1) * p],
            s: vec![0.0; cfg.kda_heads * cfg.kda_dim * cfg.kda_dim],
        }
    }
}

/// The per-layer KDA parameters the recurrence reads (all f32).
pub struct KdaParams<'a> {
    pub q_conv: &'a [f32],
    pub k_conv: &'a [f32],
    pub v_conv: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub o_norm: &'a [f32],
}

/// One KDA token (ds4 `kernel_glm53_kda_decode`): the inputs are the raw
/// projections; returns the gated, normalized output `[projection]` before
/// `kda_output`.
#[allow(clippy::too_many_arguments)]
pub fn kda_step(
    cfg: &Config,
    st: &mut KdaState,
    p: &KdaParams<'_>,
    q_in: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    raw_gate: &[f32],
    raw_beta: &[f32],
    out_gate: &[f32],
) -> Vec<f32> {
    let d = cfg.kda_dim;
    let proj = cfg.kda_projection();
    let hist = cfg.kda_conv - 1;
    let conv_w = [p.q_conv, p.k_conv, p.v_conv];
    let inputs = [q_in, k_in, v_in];
    let mut conv_out = vec![vec![0f32; proj]; 3];
    for t in 0..3 {
        let state = &mut st.conv[t * hist * proj..(t + 1) * hist * proj];
        for c in 0..proj {
            let mut acc = 0f32;
            for w in 0..hist {
                acc = state[w * proj + c].mul_add(conv_w[t][c * cfg.kda_conv + w], acc);
            }
            acc = inputs[t][c].mul_add(conv_w[t][c * cfg.kda_conv + hist], acc);
            for w in 0..hist - 1 {
                state[w * proj + c] = state[(w + 1) * proj + c];
            }
            state[(hist - 1) * proj + c] = inputs[t][c];
            conv_out[t][c] = silu(acc);
        }
    }
    let mut out = vec![0f32; proj];
    for h in 0..cfg.kda_heads {
        let r = h * d..(h + 1) * d;
        let q = &conv_out[0][r.clone()];
        let k = &conv_out[1][r.clone()];
        let v = &conv_out[2][r.clone()];
        let qn =
            1.0 / (dot(q, q) + 1e-6).sqrt() * std::f32::consts::FRAC_1_SQRT_2 / 8.0;
        let kn = 1.0 / (dot(k, k) + 1e-6).sqrt();
        let q: Vec<f32> = q.iter().map(|&x| x * qn).collect();
        let k: Vec<f32> = k.iter().map(|&x| x * kn).collect();
        let a = p.a_log[h].exp();
        let decay: Vec<f32> = (0..d)
            .map(|i| {
                let c = h * d + i;
                (cfg.kda_gate_lower_bound * sigmoid(a * (raw_gate[c] + p.dt_bias[c])))
                    .exp()
            })
            .collect();
        let beta = sigmoid(raw_beta[h]);
        let mut o = vec![0f32; d];
        for (vi, ov) in o.iter_mut().enumerate() {
            let row = &mut st.s[(h * d + vi) * d..(h * d + vi + 1) * d];
            for (x, dc) in row.iter_mut().zip(&decay) {
                *x *= dc;
            }
            let hk = dot(row, &k);
            let delta = (v[vi] - hk) * beta;
            for (x, kc) in row.iter_mut().zip(&k) {
                *x = kc.mul_add(delta, *x);
            }
            *ov = dot(row, &q);
        }
        let on = rmsnorm(&o, Some(p.o_norm), cfg.rms_eps);
        for i in 0..d {
            out[h * d + i] = on[i] * sigmoid(out_gate[h * d + i]);
        }
    }
    out
}

/// One DSA layer's caches.
#[derive(Clone, Default)]
pub struct DsaState {
    /// `[token][kv_lora]`, f16-rounded.
    pub latent: Vec<f32>,
    /// Pooled indexer keys `[pool][indexer_dim]`, f16-rounded.
    pub pooled: Vec<f32>,
    /// Raw indexer key and gate rows of the incomplete pool.
    pub tail_k: Vec<f32>,
    pub tail_g: Vec<f32>,
}

/// LayerNorm of one indexer key row (eps 1e-6, as ds4).
fn layernorm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-6).sqrt();
    x.iter().zip(w).zip(b).map(|((&v, &g), &bb)| (v - mean) * inv * g + bb).collect()
}

/// Appends one token's raw indexer key and gate rows; when they complete a
/// pool, pools them (per channel softmax over the pool's rows of `gate +
/// ape`, of the LayerNorm'd keys) into a new f16 row of `st.pooled`.
pub fn indexer_pool_push(
    cfg: &Config,
    st: &mut DsaState,
    ik: &[f32],
    ig: &[f32],
    norm_w: &[f32],
    norm_b: &[f32],
    ape: &[f32],
) {
    st.tail_k.extend_from_slice(ik);
    st.tail_g.extend_from_slice(ig);
    let dim = cfg.indexer_dim;
    let pool = cfg.indexer_pool;
    if st.tail_k.len() < pool * dim {
        return;
    }
    let rows: Vec<Vec<f32>> = (0..pool)
        .map(|r| layernorm(&st.tail_k[r * dim..(r + 1) * dim], norm_w, norm_b))
        .collect();
    for c in 0..dim {
        let logits: Vec<f32> =
            (0..pool).map(|r| st.tail_g[r * dim + c] + ape[r * dim + c]).collect();
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|&l| (l - m).exp()).collect();
        let denom: f32 = e.iter().sum();
        let pooled: f32 = (0..pool).map(|r| e[r] / denom * rows[r][c]).sum();
        st.pooled.push(round_f16(pooled));
    }
    st.tail_k.clear();
    st.tail_g.clear();
}

/// The token rows a DSA query at `pos` attends: all of them while
/// `pos + 1 <= top_k + pool - 1`, else the top `top_k / pool` pools by
/// indexer score and the incomplete pool's tokens.
pub fn indexer_select(
    cfg: &Config,
    st: &DsaState,
    pos: usize,
    iq: &[f32],
    weights: &[f32],
) -> Vec<usize> {
    let visible = pos + 1;
    if visible <= cfg.indexer_selected_limit() {
        return (0..visible).collect();
    }
    let dim = cfg.indexer_dim;
    let pools = visible / cfg.indexer_pool;
    let scale = 1.0 / ((dim * cfg.indexer_heads) as f32).sqrt();
    let scores: Vec<f32> = (0..pools)
        .map(|p| {
            let key = &st.pooled[p * dim..(p + 1) * dim];
            (0..cfg.indexer_heads)
                .map(|h| {
                    (dot(&iq[h * dim..(h + 1) * dim], key) * scale).max(0.0)
                        * weights[h]
                })
                .sum()
        })
        .collect();
    let mut order: Vec<usize> = (0..pools).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let keep = cfg.indexer_top_k / cfg.indexer_pool;
    let mut rows: Vec<usize> = order[..keep.min(pools)]
        .iter()
        .flat_map(|&p| p * cfg.indexer_pool..(p + 1) * cfg.indexer_pool)
        .collect();
    let tail = visible % cfg.indexer_pool;
    rows.extend(visible - tail..visible);
    rows
}

/// Absorbed MLA attention over the selected latent rows: returns the heads
/// `[n_head][value_dim]` before `attn_output`.
pub fn dsa_attend(
    cfg: &Config,
    latent: &[f32],
    rows: &[usize],
    q: &[f32],
    k_b: &Mat,
    v_b: &Mat,
) -> Vec<f32> {
    let kv = cfg.kv_lora;
    let qk_low = k_b.mv_grouped(q, kv);
    let scale = 1.0 / (cfg.key_dim as f32).sqrt();
    let mut lat = vec![0f32; cfg.n_head * kv];
    for h in 0..cfg.n_head {
        let ql = &qk_low[h * kv..(h + 1) * kv];
        let scores: Vec<f32> = rows
            .iter()
            .map(|&s| dot(ql, &latent[s * kv..(s + 1) * kv]) * scale)
            .collect();
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = scores.iter().map(|&x| (x - m).exp()).collect();
        let denom = e.iter().sum::<f32>().max(1e-20);
        for j in 0..kv {
            let acc: f64 = rows
                .iter()
                .zip(&e)
                .map(|(&s, &w)| f64::from(w) * f64::from(latent[s * kv + j]))
                .sum();
            lat[h * kv + j] = acc as f32 / denom;
        }
    }
    v_b.mv_grouped(&lat, cfg.value_dim)
}

/// The dense (non-expert) weights of one layer, decoded.
pub struct RefLayer {
    pub il: usize,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc: Option<[(Mat, Vec<f32>, Vec<f32>); 2]>,
    pub mats: std::collections::BTreeMap<&'static str, Mat>,
    pub vecs: std::collections::BTreeMap<&'static str, Vec<f32>>,
}

impl RefLayer {
    pub fn load(cfg: &Config, src: &dyn WeightSource, il: usize) -> Result<Self> {
        let name = |s: &str| format!("blk.{il}.{s}");
        let vec = |s: &str| -> Result<Vec<f32>> { Ok(Mat::load(src, &name(s))?.w) };
        let hc = if il < cfg.n_trunk {
            let side = |p: &str| -> Result<(Mat, Vec<f32>, Vec<f32>)> {
                Ok((
                    Mat::load(src, &name(&format!("hc_{p}_fn.weight")))?,
                    vec(&format!("hc_{p}_scale.weight"))?,
                    vec(&format!("hc_{p}_base.weight"))?,
                ))
            };
            Some([side("attn")?, side("ffn")?])
        } else {
            None
        };
        let mut mats = std::collections::BTreeMap::new();
        let mut vecs = std::collections::BTreeMap::new();
        let mat_names: &[&'static str] = match cfg.kinds[il] {
            LayerKind::Kda => &[
                "kda_q.weight",
                "kda_k.weight",
                "kda_v.weight",
                "kda_f_a.weight",
                "kda_f_b.weight",
                "kda_beta.weight",
                "kda_g_a.weight",
                "kda_g_b.weight",
                "kda_output.weight",
            ],
            LayerKind::Dsa => &[
                "attn_q_a.weight",
                "attn_q_b.weight",
                "attn_kv_a_mqa.weight",
                "attn_k_b.weight",
                "attn_v_b.weight",
                "attn_output.weight",
                "indexer.attn_q_b.weight",
                "indexer.attn_k.weight",
                "indexer.proj.weight",
                "indexer.pool_gate.weight",
            ],
        };
        let vec_names: &[&'static str] = match cfg.kinds[il] {
            LayerKind::Kda => &[
                "kda_q_conv.weight",
                "kda_k_conv.weight",
                "kda_v_conv.weight",
                "kda_a_log.weight",
                "kda_dt_bias.weight",
                "kda_o_norm.weight",
            ],
            LayerKind::Dsa => &[
                "attn_q_a_norm.weight",
                "attn_kv_a_norm.weight",
                "indexer.k_norm.weight",
                "indexer.k_norm.bias",
                "indexer.pool_ape.weight",
            ],
        };
        let ffn_mats: &[&'static str] = if cfg.is_moe(il) {
            &[
                "ffn_gate_inp.weight",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
            ]
        } else {
            &["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
        };
        for &m in mat_names.iter().chain(ffn_mats) {
            mats.insert(m, Mat::load(src, &name(m))?);
        }
        for &v in vec_names {
            vecs.insert(v, vec(v)?);
        }
        if cfg.is_moe(il) {
            vecs.insert("exp_probs_b.bias", vec("exp_probs_b.bias")?);
        }
        if il == cfg.mtp_layer() {
            mats.insert(
                "nextn.eh_proj.weight",
                Mat::load(src, &name("nextn.eh_proj.weight"))?,
            );
            for v in [
                "nextn.enorm.weight",
                "nextn.hnorm.weight",
                "nextn.shared_head_norm.weight",
            ] {
                vecs.insert(v, vec(v)?);
            }
        }
        Ok(Self {
            il,
            attn_norm: vec("attn_norm.weight")?,
            ffn_norm: vec("ffn_norm.weight")?,
            hc,
            mats,
            vecs,
        })
    }

    fn m(&self, name: &str) -> &Mat {
        &self.mats[name]
    }

    fn v(&self, name: &str) -> &[f32] {
        &self.vecs[name]
    }
}

/// Per-layer recurrent state of the reference.
pub enum RefLayerState {
    Kda(KdaState),
    Dsa(DsaState),
}

/// Intermediates of one token through one layer, for layer-by-layer
/// comparison with the GPU.
#[derive(Clone, Default)]
pub struct Trace {
    pub attn_in: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_in: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub selected: Vec<usize>,
    pub expert_weights: Vec<f32>,
}

/// The attention sublayer of one token: input normalized, output before HC.
pub fn attention(
    cfg: &Config,
    l: &RefLayer,
    st: &mut RefLayerState,
    xn: &[f32],
    pos: usize,
) -> Vec<f32> {
    match st {
        RefLayerState::Kda(s) => {
            let p = KdaParams {
                q_conv: l.v("kda_q_conv.weight"),
                k_conv: l.v("kda_k_conv.weight"),
                v_conv: l.v("kda_v_conv.weight"),
                a_log: l.v("kda_a_log.weight"),
                dt_bias: l.v("kda_dt_bias.weight"),
                o_norm: l.v("kda_o_norm.weight"),
            };
            let q = l.m("kda_q.weight").mv(xn);
            let k = l.m("kda_k.weight").mv(xn);
            let v = l.m("kda_v.weight").mv(xn);
            let gate = l.m("kda_f_b.weight").mv(&l.m("kda_f_a.weight").mv(xn));
            let beta = l.m("kda_beta.weight").mv(xn);
            let og = l.m("kda_g_b.weight").mv(&l.m("kda_g_a.weight").mv(xn));
            let o = kda_step(cfg, s, &p, &q, &k, &v, &gate, &beta, &og);
            l.m("kda_output.weight").mv(&o)
        }
        RefLayerState::Dsa(s) => {
            let qr = rmsnorm(
                &l.m("attn_q_a.weight").mv(xn),
                Some(l.v("attn_q_a_norm.weight")),
                cfg.rms_eps,
            );
            let q = l.m("attn_q_b.weight").mv(&qr);
            let kv = rmsnorm(
                &l.m("attn_kv_a_mqa.weight").mv(xn),
                Some(l.v("attn_kv_a_norm.weight")),
                cfg.rms_eps,
            );
            debug_assert_eq!(s.latent.len(), pos * cfg.kv_lora);
            s.latent.extend(kv.iter().map(|&v| round_f16(v)));
            let rows = if l.il < cfg.n_trunk {
                let ik = l.m("indexer.attn_k.weight").mv(xn);
                let ig = l.m("indexer.pool_gate.weight").mv(xn);
                indexer_pool_push(
                    cfg,
                    s,
                    &ik,
                    &ig,
                    l.v("indexer.k_norm.weight"),
                    l.v("indexer.k_norm.bias"),
                    l.v("indexer.pool_ape.weight"),
                );
                if pos + 1 <= cfg.indexer_selected_limit() {
                    (0..=pos).collect()
                } else {
                    let iq = l.m("indexer.attn_q_b.weight").mv(&qr);
                    let w = l.m("indexer.proj.weight").mv(xn);
                    indexer_select(cfg, s, pos, &iq, &w)
                }
            } else {
                (0..=pos).collect()
            };
            let heads = dsa_attend(
                cfg,
                &s.latent,
                &rows,
                &q,
                l.m("attn_k_b.weight"),
                l.m("attn_v_b.weight"),
            );
            l.m("attn_output.weight").mv(&heads)
        }
    }
}

/// The FFN sublayer of one token (input normalized). Routed experts are
/// read from `src` per token.
pub fn ffn(
    cfg: &Config,
    l: &RefLayer,
    src: &dyn WeightSource,
    xn: &[f32],
    trace: &mut Trace,
) -> Result<Vec<f32>> {
    let lim = cfg.swiglu_limit;
    let swiglu_mlp = |g: &Mat, u: &Mat, d: &Mat| -> Vec<f32> {
        let gv = g.mv(xn);
        let uv = u.mv(xn);
        let mid: Vec<f32> =
            gv.iter().zip(&uv).map(|(&a, &b)| swiglu(a, b, lim)).collect();
        d.mv(&mid)
    };
    if !cfg.is_moe(l.il) {
        return Ok(swiglu_mlp(
            l.m("ffn_gate.weight"),
            l.m("ffn_up.weight"),
            l.m("ffn_down.weight"),
        ));
    }
    let logits = l.m("ffn_gate_inp.weight").mv(xn);
    let (sel, w) = route(
        &logits,
        l.v("exp_probs_b.bias"),
        cfg.n_expert_used,
        cfg.expert_weight_scale,
    );
    let mut out = swiglu_mlp(
        l.m("ffn_gate_shexp.weight"),
        l.m("ffn_up_shexp.weight"),
        l.m("ffn_down_shexp.weight"),
    );
    let ff = cfg.ff_expert;
    let il = l.il;
    let mut moe = vec![0f32; cfg.hidden];
    let mut mids = Vec::with_capacity(sel.len());
    for (&e, &we) in sel.iter().zip(&w) {
        let g =
            src.rows(&format!("blk.{il}.ffn_gate_exps.weight"), e * ff..(e + 1) * ff)?;
        let u =
            src.rows(&format!("blk.{il}.ffn_up_exps.weight"), e * ff..(e + 1) * ff)?;
        let mid: Vec<f32> = (0..ff)
            .into_par_iter()
            .map(|r| {
                let gv = dot(&g[r * cfg.hidden..(r + 1) * cfg.hidden], xn);
                let uv = dot(&u[r * cfg.hidden..(r + 1) * cfg.hidden], xn);
                swiglu(gv, uv, lim) * we
            })
            .collect();
        mids.push(mid);
    }
    for (k, &e) in sel.iter().enumerate() {
        let dn = src.rows(
            &format!("blk.{il}.ffn_down_exps.weight"),
            e * cfg.hidden..(e + 1) * cfg.hidden,
        )?;
        let part: Vec<f32> = (0..cfg.hidden)
            .into_par_iter()
            .map(|r| dot(&dn[r * ff..(r + 1) * ff], &mids[k]))
            .collect();
        for (m, p) in moe.iter_mut().zip(part) {
            *m += p;
        }
    }
    for (o, m) in out.iter_mut().zip(&moe) {
        *o = m + *o;
    }
    trace.selected = sel;
    trace.expert_weights = w;
    Ok(out)
}

/// The reference model: runs a prompt layer by layer.
pub struct RefModel<'s> {
    pub cfg: Config,
    pub src: &'s dyn WeightSource,
}

impl<'s> RefModel<'s> {
    pub fn new(cfg: Config, src: &'s dyn WeightSource) -> Self {
        Self { cfg, src }
    }

    /// Embeds `tokens` into HC streams `[token][n_hc * hidden]`.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
        tokens
            .iter()
            .map(|&t| {
                let row =
                    self.src.rows("token_embd.weight", t as usize..t as usize + 1)?;
                Ok(row.repeat(self.cfg.n_hc))
            })
            .collect()
    }

    /// Runs trunk layer `il` over every token's HC state in order (positions
    /// `0..`), with fresh state, and returns the traces.
    pub fn trunk_layer(&self, il: usize, hs: &mut [Vec<f32>]) -> Result<Vec<Trace>> {
        let cfg = &self.cfg;
        ensure!(il < cfg.n_trunk, "layer {il} is not a trunk layer");
        let l = RefLayer::load(cfg, self.src, il)?;
        let mut st = match cfg.kinds[il] {
            LayerKind::Kda => RefLayerState::Kda(KdaState::new(cfg)),
            LayerKind::Dsa => RefLayerState::Dsa(DsaState::default()),
        };
        let [(afn, ascale, abase), (ffn_w, fscale, fbase)] =
            l.hc.as_ref().expect("trunk layer");
        let mut traces = Vec::with_capacity(hs.len());
        for (pos, h) in hs.iter_mut().enumerate() {
            let mut tr = Trace::default();
            let (xn, post, comb) = hc_pre(cfg, h, afn, ascale, abase, &l.attn_norm);
            let a = attention(cfg, &l, &mut st, &xn, pos);
            let h1 = hc_post(h, &a, &post, &comb, cfg.n_hc);
            let (fx, fpost, fcomb) =
                hc_pre(cfg, &h1, ffn_w, fscale, fbase, &l.ffn_norm);
            let f = ffn(cfg, &l, self.src, &fx, &mut tr)?;
            *h = hc_post(&h1, &f, &fpost, &fcomb, cfg.n_hc);
            tr.attn_in = xn;
            tr.attn_out = a;
            tr.ffn_in = fx;
            tr.ffn_out = f;
            traces.push(tr);
        }
        Ok(traces)
    }

    /// The trunk's final hidden of one HC state: the mean of the streams.
    pub fn collapse(&self, h: &[f32]) -> Vec<f32> {
        let n = self.cfg.n_hc;
        let hid = self.cfg.hidden;
        (0..hid)
            .map(|d| (0..n).map(|s| h[s * hid + d] * (1.0 / n as f32)).sum())
            .collect()
    }

    /// Logits of one HC state.
    pub fn logits(&self, h: &[f32], head: &Mat, norm: &[f32]) -> Vec<f32> {
        head.mv(&rmsnorm(&self.collapse(h), Some(norm), self.cfg.rms_eps))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/glm53/reference.rs"]
mod tests;
