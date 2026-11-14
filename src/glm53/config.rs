//! GLM 5.3 Flash's shape, read from the GGUF metadata (`glm5-next.*`) and
//! checked against the numbers the kernels are written for (ds4's
//! `DS4_SHAPE_GLM53` and `config_validate_glm53_model`).

use anyhow::{Result, ensure};

use crate::gguf::{Gguf, Value};

/// The GGUF architecture name.
pub const ARCH: &str = "glm5-next";

/// What a trunk layer's attention is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Kimi delta attention (linear, recurrent state per head).
    Kda,
    /// DeepSeek sparse attention: MLA latent cache plus the pooled indexer.
    Dsa,
}

#[derive(Clone, Debug)]
pub struct Config {
    /// Trunk layers plus the MTP block.
    pub n_layer: usize,
    pub n_trunk: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub ff_dense: usize,
    pub ff_expert: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_leading_dense: usize,
    pub expert_weight_scale: f32,
    pub swiglu_limit: f32,
    pub rms_eps: f32,
    pub n_head: usize,
    /// Per-head query/key width of the absorbed attention (no RoPE part).
    pub key_dim: usize,
    pub value_dim: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub indexer_heads: usize,
    pub indexer_dim: usize,
    pub indexer_top_k: usize,
    pub indexer_pool: usize,
    pub kda_heads: usize,
    pub kda_dim: usize,
    pub kda_conv: usize,
    pub kda_gate_lower_bound: f32,
    pub n_hc: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub context_length: usize,
    /// Per layer (MTP block included).
    pub kinds: Vec<LayerKind>,
    pub bos: u32,
    pub eos: u32,
    pub eot: u32,
    pub eom: u32,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.architecture()?;
        ensure!(arch == ARCH, "GGUF architecture {arch:?}, expected {ARCH:?}");
        let u =
            |k: &str| -> Result<usize> { Ok(g.u64(&format!("{ARCH}.{k}"))? as usize) };
        let f = |k: &str| g.f32(&format!("{ARCH}.{k}"));
        let n_layer = u("block_count")?;
        let n_trunk = u("trunk_block_count")?;
        ensure!(
            n_trunk <= n_layer,
            "trunk_block_count {n_trunk} exceeds block_count {n_layer}"
        );
        ensure!(
            u("nextn_predict_layers")? == n_layer - n_trunk,
            "nextn_predict_layers does not match block counts"
        );
        let kinds = match g.get(&format!("{ARCH}.layer_types")) {
            Some(Value::Array(v)) => v
                .iter()
                .map(|t| match t.as_u64() {
                    Some(0) => Ok(LayerKind::Kda),
                    Some(1) => Ok(LayerKind::Dsa),
                    other => anyhow::bail!("layer type {other:?}"),
                })
                .collect::<Result<Vec<_>>>()?,
            _ => anyhow::bail!("{ARCH}.layer_types missing"),
        };
        let tok = |k: &str| -> Result<u32> {
            Ok(g.u64(&format!("tokenizer.ggml.{k}"))? as u32)
        };
        let c = Self {
            n_layer,
            n_trunk,
            hidden: u("embedding_length")?,
            vocab: u("vocab_size")?,
            ff_dense: u("feed_forward_length")?,
            ff_expert: u("expert_feed_forward_length")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_leading_dense: u("leading_dense_block_count")?,
            expert_weight_scale: f("expert_weights_scale")?,
            swiglu_limit: f("swiglu_limit")?,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            n_head: u("attention.head_count")?,
            key_dim: u("attention.key_length")?,
            value_dim: u("attention.value_length")?,
            q_lora: u("attention.q_lora_rank")?,
            kv_lora: u("attention.kv_lora_rank")?,
            indexer_heads: u("attention.indexer.head_count")?,
            indexer_dim: u("attention.indexer.key_length")?,
            indexer_top_k: u("attention.indexer.top_k")?,
            indexer_pool: u("attention.indexer.pool_size")?,
            kda_heads: u("linear_attention.head_count")?,
            kda_dim: u("linear_attention.head_dimension")?,
            kda_conv: u("linear_attention.conv_kernel")?,
            kda_gate_lower_bound: f("linear_attention.gate_lower_bound")?,
            n_hc: u("hyper_connection.count")?,
            sinkhorn_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            context_length: u("context_length")?,
            kinds,
            bos: tok("bos_token_id")?,
            eos: tok("eos_token_id")?,
            eot: tok("eot_token_id")?,
            eom: tok("eom_token_id")?,
        };
        c.validate()?;
        Ok(c)
    }

    /// GLM 5.3 Flash's shape as the published GGUF declares it (tests build
    /// small variants from it).
    pub fn glm53_flash() -> Self {
        Self {
            n_layer: 46,
            n_trunk: 45,
            hidden: 4096,
            vocab: 154880,
            ff_dense: 12288,
            ff_expert: 2048,
            n_expert: 288,
            n_expert_used: 8,
            n_leading_dense: 3,
            expert_weight_scale: 2.5,
            swiglu_limit: 10.0,
            rms_eps: 1e-5,
            n_head: 64,
            key_dim: 256,
            value_dim: 256,
            q_lora: 1536,
            kv_lora: 512,
            indexer_heads: 32,
            indexer_dim: 128,
            indexer_top_k: 2048,
            indexer_pool: 4,
            kda_heads: 64,
            kda_dim: 128,
            kda_conv: 4,
            kda_gate_lower_bound: -5.0,
            n_hc: 4,
            sinkhorn_iters: 20,
            hc_eps: 1e-6,
            context_length: 1 << 20,
            kinds: (0..46)
                .map(|il| {
                    if il < 45 && il % 4 != 3 { LayerKind::Kda } else { LayerKind::Dsa }
                })
                .collect(),
            bos: 154822,
            eos: 154820,
            eot: 154827,
            eom: 154829,
        }
    }

    /// The shape GLM 5.3 Flash has; the kernels assume these numbers.
    pub fn validate(&self) -> Result<()> {
        let want: [(&str, usize, usize); 22] = [
            ("block_count", self.n_layer, 46),
            ("trunk_block_count", self.n_trunk, 45),
            ("embedding_length", self.hidden, 4096),
            ("vocab_size", self.vocab, 154880),
            ("feed_forward_length", self.ff_dense, 12288),
            ("expert_feed_forward_length", self.ff_expert, 2048),
            ("expert_count", self.n_expert, 288),
            ("expert_used_count", self.n_expert_used, 8),
            ("leading_dense_block_count", self.n_leading_dense, 3),
            ("attention.head_count", self.n_head, 64),
            ("attention.key_length", self.key_dim, 256),
            ("attention.value_length", self.value_dim, 256),
            ("attention.q_lora_rank", self.q_lora, 1536),
            ("attention.kv_lora_rank", self.kv_lora, 512),
            ("attention.indexer.head_count", self.indexer_heads, 32),
            ("attention.indexer.key_length", self.indexer_dim, 128),
            ("attention.indexer.top_k", self.indexer_top_k, 2048),
            ("attention.indexer.pool_size", self.indexer_pool, 4),
            ("linear_attention.head_count", self.kda_heads, 64),
            ("linear_attention.head_dimension", self.kda_dim, 128),
            ("linear_attention.conv_kernel", self.kda_conv, 4),
            ("hyper_connection.count", self.n_hc, 4),
        ];
        for (name, got, expected) in want {
            ensure!(
                got == expected,
                "GLM 5.3 Flash {name} is {got}, lily's kernels need {expected}"
            );
        }
        ensure!(
            self.kinds.len() == self.n_layer,
            "layer_types has {} entries",
            self.kinds.len()
        );
        for (il, kind) in self.kinds.iter().enumerate() {
            let expected = if il < self.n_trunk && il % 4 != 3 {
                LayerKind::Kda
            } else {
                LayerKind::Dsa
            };
            ensure!(*kind == expected, "layer {il} is {kind:?}, expected {expected:?}");
        }
        ensure!(
            self.sinkhorn_iters >= 1,
            "Sinkhorn iterations {}",
            self.sinkhorn_iters
        );
        ensure!(self.rms_eps > 0.0 && self.hc_eps > 0.0, "epsilons must be positive");
        Ok(())
    }

    /// Tokens the indexer selects at most: `top_k` plus an incomplete pool.
    pub fn indexer_selected_limit(&self) -> usize {
        self.indexer_top_k + self.indexer_pool - 1
    }

    pub fn kda_projection(&self) -> usize {
        self.kda_heads * self.kda_dim
    }

    pub fn hc_dim(&self) -> usize {
        self.n_hc * self.hidden
    }

    /// `pre[n_hc]`, `post[n_hc]`, `comb[n_hc * n_hc]`.
    pub fn hc_mix(&self) -> usize {
        self.n_hc * (self.n_hc + 2)
    }

    /// The MTP block's layer index.
    pub fn mtp_layer(&self) -> usize {
        self.n_trunk
    }

    pub fn is_moe(&self, il: usize) -> bool {
        il >= self.n_leading_dense
    }
}

#[cfg(test)]
#[path = "../../tests/unit/glm53/config.rs"]
mod tests;
