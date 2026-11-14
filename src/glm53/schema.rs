//! The tensors GLM 5.3 Flash's kernels read, with the shape and the block
//! types each may have: [`validate`] checks a file's tensor table against
//! it before anything is mapped. The kernels index weights by raw GPU
//! address from the configured shape, so a tensor that is shorter than its
//! role, or of a type a kernel does not decode, must be refused here.

use anyhow::{Result, ensure};

use super::config::{Config, LayerKind};
use crate::gguf::{GgmlType, Gguf};

/// Types every projection path decodes: the decode GEMV and the prefill
/// GEMM.
const DENSE: &[GgmlType] =
    &[GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Bf16, GgmlType::F32];
/// Types the routed-expert kernels (per-pair and grouped) decode.
const EXPERT: &[GgmlType] = &[GgmlType::Iq2Xxs, GgmlType::Q2K];
const F32: &[GgmlType] = &[GgmlType::F32];
const BF16: &[GgmlType] = &[GgmlType::Bf16];
const Q8_0: &[GgmlType] = &[GgmlType::Q8_0];

/// Tensor offsets the kernels need: 16-byte vector loads of rows.
pub const ALIGNMENT: u64 = 32;

/// One required tensor.
#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    /// ggml order.
    pub dims: Vec<u64>,
    pub types: &'static [GgmlType],
}

fn spec(name: String, dims: &[usize], types: &'static [GgmlType]) -> Spec {
    Spec { name, dims: dims.iter().map(|&d| d as u64).collect(), types }
}

/// Every tensor the model reads.
pub fn expected(c: &Config) -> Vec<Spec> {
    let h = c.hidden;
    let mut v = vec![
        spec("token_embd.weight".into(), &[h, c.vocab], Q8_0),
        spec("output.weight".into(), &[h, c.vocab], DENSE),
        spec("output_norm.weight".into(), &[h], F32),
    ];
    let kda = c.kda_projection();
    let hq = c.n_head * c.key_dim;
    for il in 0..c.n_layer {
        let n = |s: &str| format!("blk.{il}.{s}");
        v.push(spec(n("attn_norm.weight"), &[h], F32));
        v.push(spec(n("ffn_norm.weight"), &[h], F32));
        if il < c.n_trunk {
            for p in ["attn", "ffn"] {
                v.push(spec(
                    n(&format!("hc_{p}_fn.weight")),
                    &[c.hc_dim(), c.hc_mix()],
                    BF16,
                ));
                v.push(spec(n(&format!("hc_{p}_scale.weight")), &[3], F32));
                v.push(spec(n(&format!("hc_{p}_base.weight")), &[c.hc_mix()], F32));
            }
        }
        match c.kinds[il] {
            LayerKind::Kda => {
                for t in ["kda_q", "kda_k", "kda_v"] {
                    v.push(spec(n(&format!("{t}.weight")), &[h, kda], DENSE));
                }
                for t in ["kda_q_conv", "kda_k_conv", "kda_v_conv"] {
                    v.push(spec(n(&format!("{t}.weight")), &[c.kda_conv, 1, kda], F32));
                }
                v.push(spec(n("kda_f_a.weight"), &[h, c.kda_dim], DENSE));
                v.push(spec(n("kda_g_a.weight"), &[h, c.kda_dim], DENSE));
                v.push(spec(n("kda_f_b.weight"), &[c.kda_dim, kda], DENSE));
                v.push(spec(n("kda_g_b.weight"), &[c.kda_dim, kda], DENSE));
                v.push(spec(n("kda_beta.weight"), &[h, c.kda_heads], DENSE));
                v.push(spec(n("kda_dt_bias.weight"), &[kda], F32));
                v.push(spec(n("kda_a_log.weight"), &[c.kda_heads], F32));
                v.push(spec(n("kda_o_norm.weight"), &[c.kda_dim], F32));
                v.push(spec(n("kda_output.weight"), &[kda, h], DENSE));
            }
            LayerKind::Dsa => {
                v.push(spec(n("attn_q_a.weight"), &[h, c.q_lora], DENSE));
                v.push(spec(n("attn_q_a_norm.weight"), &[c.q_lora], F32));
                v.push(spec(n("attn_q_b.weight"), &[c.q_lora, hq], DENSE));
                // No RoPE part: the latent is the whole kv_a output.
                v.push(spec(n("attn_kv_a_mqa.weight"), &[h, c.kv_lora], DENSE));
                v.push(spec(n("attn_kv_a_norm.weight"), &[c.kv_lora], F32));
                v.push(spec(
                    n("attn_k_b.weight"),
                    &[c.key_dim, c.kv_lora, c.n_head],
                    DENSE,
                ));
                v.push(spec(
                    n("attn_v_b.weight"),
                    &[c.kv_lora, c.value_dim, c.n_head],
                    DENSE,
                ));
                v.push(spec(
                    n("attn_output.weight"),
                    &[c.n_head * c.value_dim, h],
                    DENSE,
                ));
                if il < c.n_trunk {
                    let idim = c.indexer_heads * c.indexer_dim;
                    v.push(spec(
                        n("indexer.attn_q_b.weight"),
                        &[c.q_lora, idim],
                        DENSE,
                    ));
                    v.push(spec(
                        n("indexer.attn_k.weight"),
                        &[h, c.indexer_dim],
                        DENSE,
                    ));
                    v.push(spec(
                        n("indexer.proj.weight"),
                        &[h, c.indexer_heads],
                        DENSE,
                    ));
                    v.push(spec(
                        n("indexer.pool_gate.weight"),
                        &[h, c.indexer_dim],
                        DENSE,
                    ));
                    v.push(spec(n("indexer.k_norm.weight"), &[c.indexer_dim], F32));
                    v.push(spec(n("indexer.k_norm.bias"), &[c.indexer_dim], F32));
                    v.push(spec(
                        n("indexer.pool_ape.weight"),
                        &[c.indexer_dim, c.indexer_pool],
                        BF16,
                    ));
                }
            }
        }
        if c.is_moe(il) {
            v.push(spec(n("ffn_gate_inp.weight"), &[h, c.n_expert], DENSE));
            v.push(spec(n("exp_probs_b.bias"), &[c.n_expert], F32));
            for t in ["ffn_gate_exps", "ffn_up_exps"] {
                v.push(spec(
                    n(&format!("{t}.weight")),
                    &[h, c.ff_expert, c.n_expert],
                    EXPERT,
                ));
            }
            v.push(spec(
                n("ffn_down_exps.weight"),
                &[c.ff_expert, h, c.n_expert],
                EXPERT,
            ));
            v.push(spec(n("ffn_gate_shexp.weight"), &[h, c.ff_expert], DENSE));
            v.push(spec(n("ffn_up_shexp.weight"), &[h, c.ff_expert], DENSE));
            v.push(spec(n("ffn_down_shexp.weight"), &[c.ff_expert, h], DENSE));
        } else {
            v.push(spec(n("ffn_gate.weight"), &[h, c.ff_dense], DENSE));
            v.push(spec(n("ffn_up.weight"), &[h, c.ff_dense], DENSE));
            v.push(spec(n("ffn_down.weight"), &[c.ff_dense, h], DENSE));
        }
        if il == c.mtp_layer() {
            v.push(spec(n("nextn.eh_proj.weight"), &[2 * h, h], DENSE));
            for t in ["nextn.enorm", "nextn.hnorm", "nextn.shared_head_norm"] {
                v.push(spec(n(&format!("{t}.weight")), &[h], F32));
            }
        }
    }
    v
}

/// The tensors each layer reads in one fused dispatch: the decode GEMV of a
/// group, and the routed-expert gate/up kernel, take one block type for all
/// of its weights.
pub fn fused_groups(c: &Config) -> Vec<Vec<String>> {
    let mut v = Vec::new();
    for il in 0..c.n_layer {
        let group = |names: &[&str]| -> Vec<String> {
            names.iter().map(|s| format!("blk.{il}.{s}.weight")).collect()
        };
        match c.kinds[il] {
            LayerKind::Kda => {
                v.push(group(&["kda_q", "kda_k"]));
                v.push(group(&["kda_v", "kda_f_a", "kda_beta", "kda_g_a"]));
            }
            LayerKind::Dsa => {
                v.push(group(&["attn_q_a", "attn_kv_a_mqa"]));
                if il < c.n_trunk {
                    v.push(group(&[
                        "indexer.attn_k",
                        "indexer.pool_gate",
                        "indexer.proj",
                    ]));
                }
            }
        }
        if c.is_moe(il) {
            v.push(group(&["ffn_gate_exps", "ffn_up_exps"]));
            v.push(group(&["ffn_gate_shexp", "ffn_up_shexp"]));
        } else {
            v.push(group(&["ffn_gate", "ffn_up"]));
        }
    }
    v
}

/// Checks every tensor the model reads: present, of the expected rank and
/// dimensions, of a type its kernels decode, at an aligned offset; and that
/// the tensors of each fused group share a type.
pub fn validate(g: &Gguf, c: &Config) -> Result<()> {
    for s in expected(c) {
        let t = g.require(&s.name)?;
        ensure!(
            t.dims == s.dims,
            "tensor {} has dimensions {:?}, GLM 5.3 Flash needs {:?}",
            s.name,
            t.dims,
            s.dims
        );
        ensure!(
            s.types.contains(&t.ty),
            "tensor {} is {}, lily's kernels read it as one of {:?}",
            s.name,
            t.ty.name(),
            s.types.iter().map(|t| t.name()).collect::<Vec<_>>()
        );
        ensure!(
            t.offset.is_multiple_of(ALIGNMENT),
            "tensor {} is at offset {}, not a multiple of {ALIGNMENT}",
            s.name,
            t.offset
        );
    }
    for names in fused_groups(c) {
        let ty = g.require(&names[0])?.ty;
        for n in &names[1..] {
            ensure!(
                g.require(n)?.ty == ty,
                "tensors {} are read together but of different types",
                names.join(", ")
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/glm53/schema.rs"]
mod tests;
