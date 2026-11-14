//! GLM 5.3 Flash on the GPU: weights mapped read-only from the GGUF, the
//! per-session caches and recurrent state, and the forward pass over up to
//! [`MAX_ROWS`] consecutive tokens at a time (decode is one row; a prompt is
//! fed in runs of rows).
//!
//! The graph is ds4's (`glm_graph_forward_token`); `reference.rs` holds the
//! same graph on the CPU and `forward_traced` exposes each layer's
//! intermediates for a layer-by-layer comparison.

use std::path::Path;

use anyhow::{Context as _, Result, ensure};

use super::config::{Config, LayerKind};
use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::kernels::glm53::{
    self as k, GemvLayout, KdaInputs, KdaParams, PoolParams, QWeight, Rows,
};
use crate::metal::{Buffer, ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};

/// Rows one forward step takes (the GEMV's limit).
pub const MAX_ROWS: usize = k::GEMV_MAX_ROWS;

/// Mapped regions are at most this large (single tensors larger than it get
/// a region of their own).
const REGION_BYTES: u64 = 1 << 30;

/// The tensors of the file, mapped.
struct Mapped {
    tensors: std::collections::HashMap<String, (Buffer, usize, TensorInfo)>,
    buffers: Vec<Buffer>,
}

impl Mapped {
    fn map(ctx: &MetalContext, g: &Gguf) -> Result<Self> {
        let mut order: Vec<&TensorInfo> = g.tensors.iter().collect();
        order.sort_by_key(|t| t.offset);
        let mut tensors = std::collections::HashMap::new();
        let mut buffers = Vec::new();
        let mut i = 0;
        while i < order.len() {
            let start = order[i].offset;
            let mut end = start + order[i].bytes;
            let mut j = i + 1;
            while j < order.len()
                && order[j].offset + order[j].bytes - start <= REGION_BYTES
            {
                end = order[j].offset + order[j].bytes;
                j += 1;
            }
            let (buf, inner) = ctx
                .new_buffer_mapped(&g.path, start, usize::try_from(end - start)?)
                .with_context(|| format!("mapping {} bytes at {start}", end - start))?;
            for t in &order[i..j] {
                let at = inner + usize::try_from(t.offset - start)?;
                tensors.insert(t.name.clone(), (buf.clone(), at, (*t).clone()));
            }
            buffers.push(buf);
            i = j;
        }
        Ok(Self { tensors, buffers })
    }

    fn get(&self, name: &str) -> Result<&(Buffer, usize, TensorInfo)> {
        self.tensors.get(name).with_context(|| format!("GGUF has no tensor {name}"))
    }

    fn q(&self, name: &str) -> Result<QWeight> {
        let (buf, at, t) = self.get(name)?;
        let whole = Tensor::from_buffer(buf.clone(), &[buf.length()], DType::I8)?;
        let view = whole.view(*at, &[t.bytes as usize])?;
        QWeight::new(view, t.ty, t.row_len(), t.rows())
    }

    fn f32(&self, name: &str) -> Result<Tensor> {
        let (buf, at, t) = self.get(name)?;
        ensure!(t.ty == GgmlType::F32, "{name} is {:?}, expected F32", t.ty);
        let n = t.row_len() * t.rows();
        let whole = Tensor::from_buffer(buf.clone(), &[buf.length() / 4], DType::F32)?;
        whole.view(at / 4, &[n])
    }

    /// Raw bytes as I8 (the BF16 `pool_ape`).
    fn bytes(&self, name: &str) -> Result<Tensor> {
        let (buf, at, t) = self.get(name)?;
        let whole = Tensor::from_buffer(buf.clone(), &[buf.length()], DType::I8)?;
        whole.view(*at, &[t.bytes as usize])
    }
}

struct HcW {
    fn_w: QWeight,
    scale: Tensor,
    base: Tensor,
}

struct KdaW {
    q: QWeight,
    k: QWeight,
    v: QWeight,
    f_a: QWeight,
    f_b: QWeight,
    beta: QWeight,
    g_a: QWeight,
    g_b: QWeight,
    out: QWeight,
    q_conv: Tensor,
    k_conv: Tensor,
    v_conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    o_norm: Tensor,
}

struct IndexerW {
    q_b: QWeight,
    k: QWeight,
    proj: QWeight,
    gate: QWeight,
    norm_w: Tensor,
    norm_b: Tensor,
    ape: Tensor,
}

struct DsaW {
    q_a: QWeight,
    q_a_norm: Tensor,
    q_b: QWeight,
    kv_a: QWeight,
    kv_a_norm: Tensor,
    k_b: QWeight,
    v_b: QWeight,
    out: QWeight,
    indexer: Option<IndexerW>,
}

enum AttnW {
    Kda(KdaW),
    Dsa(DsaW),
}

enum FfnW {
    Dense {
        gate: QWeight,
        up: QWeight,
        down: QWeight,
    },
    Moe {
        router: QWeight,
        bias: Tensor,
        gate: QWeight,
        up: QWeight,
        down: QWeight,
        sh_gate: QWeight,
        sh_up: QWeight,
        sh_down: QWeight,
    },
}

struct LayerW {
    attn_norm: Tensor,
    ffn_norm: Tensor,
    hc: Option<[HcW; 2]>,
    attn: AttnW,
    ffn: FfnW,
}

/// The model's weights on the GPU.
pub struct Glm53Model {
    pub cfg: Config,
    pub gguf: Gguf,
    mapped: Mapped,
    layers: Vec<LayerW>,
    embed: QWeight,
    output: QWeight,
    output_norm: Tensor,
}

impl Glm53Model {
    /// Opens and maps `path` (the GGUF file).
    pub fn load(ctx: &MetalContext, path: &Path) -> Result<Self> {
        let gguf = Gguf::open(path)?;
        let cfg = Config::from_gguf(&gguf)?;
        let mapped = Mapped::map(ctx, &gguf)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{il}.{s}");
            let q = |s: &str| mapped.q(&n(s));
            let f = |s: &str| mapped.f32(&n(s));
            let hc = if il < cfg.n_trunk {
                let side = |p: &str| -> Result<HcW> {
                    Ok(HcW {
                        fn_w: q(&format!("hc_{p}_fn.weight"))?,
                        scale: f(&format!("hc_{p}_scale.weight"))?,
                        base: f(&format!("hc_{p}_base.weight"))?,
                    })
                };
                Some([side("attn")?, side("ffn")?])
            } else {
                None
            };
            let attn = match cfg.kinds[il] {
                LayerKind::Kda => AttnW::Kda(KdaW {
                    q: q("kda_q.weight")?,
                    k: q("kda_k.weight")?,
                    v: q("kda_v.weight")?,
                    f_a: q("kda_f_a.weight")?,
                    f_b: q("kda_f_b.weight")?,
                    beta: q("kda_beta.weight")?,
                    g_a: q("kda_g_a.weight")?,
                    g_b: q("kda_g_b.weight")?,
                    out: q("kda_output.weight")?,
                    q_conv: f("kda_q_conv.weight")?,
                    k_conv: f("kda_k_conv.weight")?,
                    v_conv: f("kda_v_conv.weight")?,
                    a_log: f("kda_a_log.weight")?,
                    dt_bias: f("kda_dt_bias.weight")?,
                    o_norm: f("kda_o_norm.weight")?,
                }),
                LayerKind::Dsa => AttnW::Dsa(DsaW {
                    q_a: q("attn_q_a.weight")?,
                    q_a_norm: f("attn_q_a_norm.weight")?,
                    q_b: q("attn_q_b.weight")?,
                    kv_a: q("attn_kv_a_mqa.weight")?,
                    kv_a_norm: f("attn_kv_a_norm.weight")?,
                    k_b: q("attn_k_b.weight")?,
                    v_b: q("attn_v_b.weight")?,
                    out: q("attn_output.weight")?,
                    indexer: if il < cfg.n_trunk {
                        Some(IndexerW {
                            q_b: q("indexer.attn_q_b.weight")?,
                            k: q("indexer.attn_k.weight")?,
                            proj: q("indexer.proj.weight")?,
                            gate: q("indexer.pool_gate.weight")?,
                            norm_w: f("indexer.k_norm.weight")?,
                            norm_b: f("indexer.k_norm.bias")?,
                            ape: mapped.bytes(&n("indexer.pool_ape.weight"))?,
                        })
                    } else {
                        None
                    },
                }),
            };
            let ffn = if cfg.is_moe(il) {
                FfnW::Moe {
                    router: q("ffn_gate_inp.weight")?,
                    bias: f("exp_probs_b.bias")?,
                    gate: q("ffn_gate_exps.weight")?,
                    up: q("ffn_up_exps.weight")?,
                    down: q("ffn_down_exps.weight")?,
                    sh_gate: q("ffn_gate_shexp.weight")?,
                    sh_up: q("ffn_up_shexp.weight")?,
                    sh_down: q("ffn_down_shexp.weight")?,
                }
            } else {
                FfnW::Dense {
                    gate: q("ffn_gate.weight")?,
                    up: q("ffn_up.weight")?,
                    down: q("ffn_down.weight")?,
                }
            };
            layers.push(LayerW {
                attn_norm: f("attn_norm.weight")?,
                ffn_norm: f("ffn_norm.weight")?,
                hc,
                attn,
                ffn,
            });
        }
        let embed = mapped.q("token_embd.weight")?;
        let output = mapped.q("output.weight")?;
        let output_norm = mapped.f32("output_norm.weight")?;
        Ok(Self { cfg, gguf, mapped, layers, embed, output, output_norm })
    }

    /// The buffers holding the mapped file.
    pub fn weight_buffers(&self) -> Vec<Buffer> {
        self.mapped.buffers.clone()
    }

    pub fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<Glm53State> {
        Glm53State::new(ctx, &self.cfg, capacity)
    }

    pub fn new_scratch(&self, ctx: &MetalContext) -> Result<Scratch> {
        Scratch::new(ctx, &self.cfg)
    }
}

/// One layer's recurrent state or caches.
pub enum LayerState {
    Kda { conv: Tensor, s: Tensor },
    Dsa { latent: Tensor, pooled: Tensor, tail_k: Tensor, tail_g: Tensor },
}

/// A session: caches and recurrent state at position `pos`.
pub struct Glm53State {
    pub pos: usize,
    pub capacity: usize,
    pub layers: Vec<LayerState>,
}

impl Glm53State {
    pub fn new(ctx: &MetalContext, cfg: &Config, capacity: usize) -> Result<Self> {
        let capacity = capacity.next_multiple_of(4);
        let p = cfg.kda_projection();
        let layers = (0..cfg.n_layer)
            .map(|il| -> Result<LayerState> {
                Ok(match cfg.kinds[il] {
                    LayerKind::Kda => LayerState::Kda {
                        conv: Tensor::zeros(
                            ctx,
                            &[3 * (cfg.kda_conv - 1) * p],
                            DType::F32,
                        )?,
                        s: Tensor::zeros(
                            ctx,
                            &[cfg.kda_heads * cfg.kda_dim * cfg.kda_dim],
                            DType::F32,
                        )?,
                    },
                    LayerKind::Dsa => LayerState::Dsa {
                        latent: Tensor::zeros(
                            ctx,
                            &[capacity * cfg.kv_lora],
                            DType::F16,
                        )?,
                        pooled: Tensor::zeros(
                            ctx,
                            &[capacity / 4 * cfg.indexer_dim],
                            DType::F16,
                        )?,
                        tail_k: Tensor::zeros(ctx, &[4 * cfg.indexer_dim], DType::F32)?,
                        tail_g: Tensor::zeros(ctx, &[4 * cfg.indexer_dim], DType::F32)?,
                    },
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { pos: 0, capacity, layers })
    }

    /// Zeroes everything (GPU idle).
    pub fn reset(&mut self) {
        for l in &self.layers {
            match l {
                LayerState::Kda { conv, s } => {
                    conv.zero_fill();
                    s.zero_fill();
                }
                LayerState::Dsa { tail_k, tail_g, .. } => {
                    tail_k.zero_fill();
                    tail_g.zero_fill();
                }
            }
        }
        self.pos = 0;
    }
}

/// Per-step intermediates, for up to [`MAX_ROWS`] rows.
pub struct Scratch {
    pub tokens: Tensor,
    pub h: Tensor,
    flat: Tensor,
    mix: Tensor,
    split: Tensor,
    pub xn: Tensor,
    pub attn_out: Tensor,
    pub ffn_out: Tensor,
    // KDA
    kq: Tensor,
    kk: Tensor,
    kv: Tensor,
    kgate: Tensor,
    kbeta: Tensor,
    kog: Tensor,
    klow: Tensor,
    kout: Tensor,
    // DSA
    qr: Tensor,
    q: Tensor,
    kv_raw: Tensor,
    qk_low: Tensor,
    scores: Tensor,
    denom: Tensor,
    lat: Tensor,
    heads: Tensor,
    ik: Tensor,
    ig: Tensor,
    iq: Tensor,
    iw: Tensor,
    pool_scores: Tensor,
    pool_scores_cap: usize,
    sel: Tensor,
    sel_count: Tensor,
    // FFN
    fg: Tensor,
    fu: Tensor,
    fmid: Tensor,
    rlogits: Tensor,
    rsel: Tensor,
    rw: Tensor,
    moe_mid: Tensor,
    moe_out: Tensor,
    // Head
    hidden: Tensor,
    pub logits: Tensor,
}

impl Scratch {
    fn new(ctx: &MetalContext, cfg: &Config) -> Result<Self> {
        let m = MAX_ROWS;
        let f = |n: usize| Tensor::zeros(ctx, &[n], DType::F32);
        let u = |n: usize| Tensor::zeros(ctx, &[n], DType::U32);
        let p = cfg.kda_projection();
        let hq = cfg.n_head * cfg.key_dim;
        let s = cfg.indexer_selected_limit().next_multiple_of(8);
        Ok(Self {
            tokens: u(m)?,
            h: f(m * cfg.hc_dim())?,
            flat: f(m * cfg.hc_dim())?,
            mix: f(m * cfg.hc_mix())?,
            split: f(m * cfg.hc_mix())?,
            xn: f(m * cfg.hidden)?,
            attn_out: f(m * cfg.hidden)?,
            ffn_out: f(m * cfg.hidden)?,
            kq: f(m * p)?,
            kk: f(m * p)?,
            kv: f(m * p)?,
            kgate: f(m * p)?,
            kbeta: f(m * cfg.kda_heads)?,
            kog: f(m * p)?,
            klow: f(m * cfg.kda_dim)?,
            kout: f(m * p)?,
            qr: f(m * cfg.q_lora)?,
            q: f(m * hq)?,
            kv_raw: f(m * cfg.kv_lora)?,
            qk_low: f(m * cfg.n_head * cfg.kv_lora)?,
            scores: f(m * cfg.n_head * s)?,
            denom: f(m * cfg.n_head)?,
            lat: f(m * cfg.n_head * cfg.kv_lora)?,
            heads: f(m * cfg.n_head * cfg.value_dim)?,
            ik: f(m * cfg.indexer_dim)?,
            ig: f(m * cfg.indexer_dim)?,
            iq: f(m * cfg.indexer_heads * cfg.indexer_dim)?,
            iw: f(m * cfg.indexer_heads)?,
            pool_scores: f(m * 1024)?,
            pool_scores_cap: 1024,
            sel: u(m * s)?,
            sel_count: u(m)?,
            fg: f(m * cfg.ff_dense)?,
            fu: f(m * cfg.ff_dense)?,
            fmid: f(m * cfg.ff_dense)?,
            rlogits: f(m * cfg.n_expert)?,
            rsel: u(m * cfg.n_expert_used)?,
            rw: f(m * cfg.n_expert_used)?,
            moe_mid: f(m * cfg.n_expert_used * cfg.ff_expert)?,
            moe_out: f(m * cfg.hidden)?,
            hidden: f(m * cfg.hidden)?,
            logits: f(cfg.vocab)?,
        })
    }

    /// Grows the pool score rows to cover `pools` pools.
    fn ensure_pools(&mut self, ctx: &MetalContext, pools: usize) -> Result<()> {
        if pools > self.pool_scores_cap {
            let cap = pools.next_power_of_two();
            self.pool_scores = Tensor::zeros(ctx, &[MAX_ROWS * cap], DType::F32)?;
            self.pool_scores_cap = cap;
        }
        Ok(())
    }
}

/// What a traced forward reads back after each layer (row-major over the
/// step's rows).
#[derive(Default, Clone)]
pub struct LayerTrace {
    pub attn_in: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_in: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub h: Vec<f32>,
}

impl Glm53Model {
    /// Feeds `tokens` (at most [`MAX_ROWS`]) at the state's position and
    /// leaves the last row's logits in `scratch.logits` when `logits`.
    pub fn forward(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
        logits: bool,
    ) -> Result<()> {
        self.forward_inner(ctx, st, sc, tokens, logits, None)
    }

    /// [`Self::forward`] committing after every trunk layer and reading its
    /// intermediates back.
    pub fn forward_traced(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
        trace: &mut Vec<LayerTrace>,
    ) -> Result<()> {
        self.forward_inner(ctx, st, sc, tokens, true, Some(trace))
    }

    /// Feeds a whole prompt in runs of [`MAX_ROWS`] rows.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
    ) -> Result<()> {
        let n = tokens.len();
        for (i, chunk) in tokens.chunks(MAX_ROWS).enumerate() {
            let last = (i + 1) * MAX_ROWS >= n;
            self.forward(ctx, st, sc, chunk, last)?;
        }
        Ok(())
    }

    fn forward_inner(
        &self,
        ctx: &MetalContext,
        st: &mut Glm53State,
        sc: &mut Scratch,
        tokens: &[u32],
        want_logits: bool,
        mut trace: Option<&mut Vec<LayerTrace>>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let m = tokens.len();
        ensure!((1..=MAX_ROWS).contains(&m), "forward of {m} rows");
        let pos0 = st.pos;
        ensure!(
            pos0 + m <= st.capacity,
            "state capacity {} for position {}",
            st.capacity,
            pos0 + m
        );
        sc.ensure_pools(ctx, (pos0 + m) / 4 + 1)?;
        sc.tokens.write_bytes(bytemuck::cast_slice(&{
            let mut t = [0u32; MAX_ROWS];
            t[..m].copy_from_slice(tokens);
            t
        }))?;
        let mut pass = ctx.begin()?;
        k::embed(ctx, &pass, &self.embed, &sc.tokens, &sc.h, m)?;
        for il in 0..cfg.n_trunk {
            let l = &self.layers[il];
            let [hc_a, hc_f] = l.hc.as_ref().expect("trunk layer has HC");
            self.hc_pre(ctx, &pass, sc, hc_a, &l.attn_norm, m)?;
            if trace.is_some() {
                pass.commit_wait()?;
                pass = ctx.begin()?;
            }
            let attn_in =
                trace.as_ref().map(|_| read(&sc.xn, m * cfg.hidden)).transpose()?;
            self.attention(ctx, &pass, sc, &st.layers[il], &l.attn, il, pos0, m)?;
            k::hc_post(
                ctx,
                &pass,
                &sc.h,
                &sc.attn_out,
                &sc.split,
                &sc.h,
                m,
                cfg.hidden,
            )?;
            self.hc_pre(ctx, &pass, sc, hc_f, &l.ffn_norm, m)?;
            if trace.is_some() {
                pass.commit_wait()?;
                pass = ctx.begin()?;
            }
            let ffn_in =
                trace.as_ref().map(|_| read(&sc.xn, m * cfg.hidden)).transpose()?;
            self.ffn(ctx, &pass, sc, &l.ffn, m)?;
            k::hc_post(
                ctx,
                &pass,
                &sc.h,
                &sc.ffn_out,
                &sc.split,
                &sc.h,
                m,
                cfg.hidden,
            )?;
            if let Some(t) = trace.as_deref_mut() {
                pass.commit_wait()?;
                pass = ctx.begin()?;
                t.push(LayerTrace {
                    attn_in: attn_in.unwrap(),
                    attn_out: read(&sc.attn_out, m * cfg.hidden)?,
                    ffn_in: ffn_in.unwrap(),
                    ffn_out: read(&sc.ffn_out, m * cfg.hidden)?,
                    h: read(&sc.h, m * cfg.hc_dim())?,
                });
            }
        }
        if want_logits {
            let last = m - 1;
            let h_last = sc.h.view(last * cfg.hc_dim(), &[cfg.hc_dim()])?;
            k::hc_mean(ctx, &pass, &h_last, &sc.hidden, 1, cfg.hidden)?;
            k::rmsnorm(
                ctx,
                &pass,
                &sc.hidden,
                Some(&self.output_norm),
                &sc.hidden,
                1,
                cfg.hidden,
                cfg.rms_eps,
            )?;
            k::gemv(ctx, &pass, &self.output, &sc.hidden, &sc.logits, 1)?;
        }
        pass.commit_wait()?;
        st.pos += m;
        Ok(())
    }

    fn hc_pre(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        hc: &HcW,
        norm: &Tensor,
        m: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        k::rmsnorm(ctx, pass, &sc.h, None, &sc.flat, m, cfg.hc_dim(), cfg.rms_eps)?;
        k::gemv(ctx, pass, &hc.fn_w, &sc.flat, &sc.mix, m)?;
        k::hc_pre(
            ctx,
            pass,
            &sc.mix,
            &hc.scale,
            &hc.base,
            &sc.h,
            norm,
            &sc.split,
            &sc.xn,
            m,
            cfg.hidden,
            cfg.sinkhorn_iters,
            cfg.hc_eps,
            cfg.rms_eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        ls: &LayerState,
        a: &AttnW,
        il: usize,
        pos0: usize,
        m: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        match (a, ls) {
            (AttnW::Kda(w), LayerState::Kda { conv, s }) => {
                k::gemv(ctx, pass, &w.q, &sc.xn, &sc.kq, m)?;
                k::gemv(ctx, pass, &w.k, &sc.xn, &sc.kk, m)?;
                k::gemv(ctx, pass, &w.v, &sc.xn, &sc.kv, m)?;
                k::gemv(ctx, pass, &w.f_a, &sc.xn, &sc.klow, m)?;
                k::gemv(ctx, pass, &w.f_b, &sc.klow, &sc.kgate, m)?;
                k::gemv(ctx, pass, &w.beta, &sc.xn, &sc.kbeta, m)?;
                k::gemv(ctx, pass, &w.g_a, &sc.xn, &sc.klow, m)?;
                k::gemv(ctx, pass, &w.g_b, &sc.klow, &sc.kog, m)?;
                let p = KdaParams {
                    q_conv: &w.q_conv,
                    k_conv: &w.k_conv,
                    v_conv: &w.v_conv,
                    a_log: &w.a_log,
                    dt_bias: &w.dt_bias,
                    o_norm: &w.o_norm,
                    heads: cfg.kda_heads,
                    lower_bound: cfg.kda_gate_lower_bound,
                    eps: cfg.rms_eps,
                };
                let x = KdaInputs {
                    q: &sc.kq,
                    k: &sc.kk,
                    v: &sc.kv,
                    gate: &sc.kgate,
                    beta: &sc.kbeta,
                    out_gate: &sc.kog,
                };
                k::kda(ctx, pass, &p, &x, conv, s, &sc.kout, m)?;
                k::gemv(ctx, pass, &w.out, &sc.kout, &sc.attn_out, m)
            }
            (AttnW::Dsa(w), LayerState::Dsa { latent, pooled, tail_k, tail_g }) => {
                k::gemv(ctx, pass, &w.q_a, &sc.xn, &sc.qr, m)?;
                k::rmsnorm(
                    ctx,
                    pass,
                    &sc.qr,
                    Some(&w.q_a_norm),
                    &sc.qr,
                    m,
                    cfg.q_lora,
                    cfg.rms_eps,
                )?;
                k::gemv(ctx, pass, &w.q_b, &sc.qr, &sc.q, m)?;
                k::gemv(ctx, pass, &w.kv_a, &sc.xn, &sc.kv_raw, m)?;
                k::kv_store(
                    ctx,
                    pass,
                    &sc.kv_raw,
                    &w.kv_a_norm,
                    latent,
                    pos0,
                    m,
                    cfg.rms_eps,
                )?;
                let kv = cfg.kv_lora;
                let hq = cfg.n_head * cfg.key_dim;
                k::gemv_with(
                    ctx,
                    pass,
                    &w.k_b,
                    &sc.q,
                    &sc.qk_low,
                    m,
                    GemvLayout {
                        x_group: kv,
                        x_row_stride: hq,
                        y_row_stride: cfg.n_head * kv,
                    },
                )?;
                let limit = cfg.indexer_selected_limit();
                let rows = match &w.indexer {
                    Some(ix) => {
                        k::gemv(ctx, pass, &ix.k, &sc.xn, &sc.ik, m)?;
                        k::gemv(ctx, pass, &ix.gate, &sc.xn, &sc.ig, m)?;
                        let pp = PoolParams {
                            norm_w: &ix.norm_w,
                            norm_b: &ix.norm_b,
                            ape: &ix.ape,
                        };
                        k::indexer_pool_update(
                            ctx, pass, &pp, &sc.ik, &sc.ig, pooled, tail_k, tail_g,
                            pos0, m,
                        )?;
                        if pos0 + m <= limit {
                            Rows::Causal { pos0 }
                        } else {
                            k::gemv(ctx, pass, &ix.q_b, &sc.qr, &sc.iq, m)?;
                            k::gemv(ctx, pass, &ix.proj, &sc.xn, &sc.iw, m)?;
                            let scale = 1.0
                                / ((cfg.indexer_dim * cfg.indexer_heads) as f32).sqrt();
                            k::indexer_scores(
                                ctx,
                                pass,
                                &sc.iq,
                                &sc.iw,
                                pooled,
                                &sc.pool_scores,
                                m,
                                sc.pool_scores_cap,
                                pos0,
                                scale,
                            )?;
                            let s = limit.next_multiple_of(8);
                            k::indexer_topk(
                                ctx,
                                pass,
                                &sc.pool_scores,
                                &sc.sel,
                                &sc.sel_count,
                                m,
                                sc.pool_scores_cap,
                                s,
                                pos0,
                                cfg.indexer_top_k / cfg.indexer_pool,
                            )?;
                            Rows::Selected { sel: &sc.sel, count: s }
                        }
                    }
                    None => {
                        ensure!(
                            pos0 + m <= limit,
                            "MTP attention window past {limit} rows (il {il})"
                        );
                        Rows::Causal { pos0 }
                    }
                };
                let s = match rows {
                    Rows::Causal { .. } => (pos0 + m).next_multiple_of(8),
                    Rows::Selected { count, .. } => count,
                };
                k::dsa_attend(
                    ctx,
                    pass,
                    &sc.qk_low,
                    latent,
                    rows,
                    &sc.scores,
                    &sc.denom,
                    &sc.lat,
                    m,
                    s,
                    1.0 / (cfg.key_dim as f32).sqrt(),
                )?;
                k::gemv_with(
                    ctx,
                    pass,
                    &w.v_b,
                    &sc.lat,
                    &sc.heads,
                    m,
                    GemvLayout {
                        x_group: cfg.value_dim,
                        x_row_stride: cfg.n_head * kv,
                        y_row_stride: cfg.n_head * cfg.value_dim,
                    },
                )?;
                k::gemv(ctx, pass, &w.out, &sc.heads, &sc.attn_out, m)
            }
            _ => anyhow::bail!("layer {il}: weights and state disagree"),
        }
    }

    fn ffn(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        sc: &Scratch,
        f: &FfnW,
        m: usize,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let lim = cfg.swiglu_limit;
        match f {
            FfnW::Dense { gate, up, down } => {
                k::gemv(ctx, pass, gate, &sc.xn, &sc.fg, m)?;
                k::gemv(ctx, pass, up, &sc.xn, &sc.fu, m)?;
                k::swiglu(ctx, pass, &sc.fg, &sc.fu, &sc.fmid, m * cfg.ff_dense, lim)?;
                k::gemv(ctx, pass, down, &sc.fmid, &sc.ffn_out, m)
            }
            FfnW::Moe { router, bias, gate, up, down, sh_gate, sh_up, sh_down } => {
                let used = cfg.n_expert_used;
                k::gemv(ctx, pass, router, &sc.xn, &sc.rlogits, m)?;
                k::router_topk(
                    ctx,
                    pass,
                    &sc.rlogits,
                    bias,
                    &sc.rsel,
                    &sc.rw,
                    m,
                    cfg.n_expert,
                    used,
                    cfg.expert_weight_scale,
                )?;
                k::moe_gate_up(
                    ctx,
                    pass,
                    gate,
                    up,
                    cfg.ff_expert,
                    &sc.xn,
                    &sc.rsel,
                    &sc.rw,
                    &sc.moe_mid,
                    m,
                    used,
                    lim,
                )?;
                k::moe_down(
                    ctx,
                    pass,
                    down,
                    cfg.hidden,
                    &sc.moe_mid,
                    &sc.rsel,
                    &sc.moe_out,
                    m,
                    used,
                )?;
                k::gemv(ctx, pass, sh_gate, &sc.xn, &sc.fg, m)?;
                k::gemv(ctx, pass, sh_up, &sc.xn, &sc.fu, m)?;
                k::swiglu(ctx, pass, &sc.fg, &sc.fu, &sc.fmid, m * cfg.ff_expert, lim)?;
                k::gemv(ctx, pass, sh_down, &sc.fmid, &sc.ffn_out, m)?;
                k::add(ctx, pass, &sc.moe_out, &sc.ffn_out, &sc.ffn_out, m * cfg.hidden)
            }
        }
    }
}

fn read(t: &Tensor, n: usize) -> Result<Vec<f32>> {
    Ok(t.to_f32()?[..n].to_vec())
}
