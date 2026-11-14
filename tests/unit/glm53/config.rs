use super::*;

/// The published Q2 file's header: `LILY_GLM53_GGUF=<path> cargo test
/// --lib glm53::config -- --ignored` (reads the header only).
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn real_header_validates() {
    let path = std::env::var("LILY_GLM53_GGUF").expect("LILY_GLM53_GGUF");
    let g = Gguf::open(std::path::Path::new(&path)).expect("open");
    let c = Config::from_gguf(&g).expect("config");
    assert_eq!(c.mtp_layer(), 45);
    assert_eq!(c.kinds.iter().filter(|k| **k == LayerKind::Kda).count(), 34);
    assert_eq!((c.bos, c.eos, c.eot), (154822, 154820, 154827));
    assert_eq!(c.indexer_selected_limit(), 2051);
    assert_eq!(c.rms_eps, 1e-5);
    assert_eq!(c.expert_weight_scale, 2.5);
    assert_eq!(c.kda_gate_lower_bound, -5.0);
    assert_eq!(g.tensors.len(), 1412);
    crate::glm53::schema::validate(&g, &c)
        .expect("the published file matches the schema");
}

/// A metadata-only GGUF of `c`'s configuration, with `tokens` as the raw
/// token-id values (bos, eos, eot, eom).
fn header(c: &Config, tokens: [u64; 4]) -> Vec<u8> {
    fn key(f: &mut Vec<u8>, k: &str, ty: u32) {
        f.extend((k.len() as u64).to_le_bytes());
        f.extend(k.as_bytes());
        f.extend(ty.to_le_bytes());
    }
    let ints: [(&str, usize); 25] = [
        ("block_count", c.n_layer),
        ("trunk_block_count", c.n_trunk),
        ("nextn_predict_layers", c.n_layer - c.n_trunk),
        ("embedding_length", c.hidden),
        ("vocab_size", c.vocab),
        ("feed_forward_length", c.ff_dense),
        ("expert_feed_forward_length", c.ff_expert),
        ("expert_count", c.n_expert),
        ("expert_used_count", c.n_expert_used),
        ("leading_dense_block_count", c.n_leading_dense),
        ("attention.head_count", c.n_head),
        ("attention.key_length", c.key_dim),
        ("attention.value_length", c.value_dim),
        ("attention.q_lora_rank", c.q_lora),
        ("attention.kv_lora_rank", c.kv_lora),
        ("attention.indexer.head_count", c.indexer_heads),
        ("attention.indexer.key_length", c.indexer_dim),
        ("attention.indexer.top_k", c.indexer_top_k),
        ("attention.indexer.pool_size", c.indexer_pool),
        ("linear_attention.head_count", c.kda_heads),
        ("linear_attention.head_dimension", c.kda_dim),
        ("linear_attention.conv_kernel", c.kda_conv),
        ("hyper_connection.count", c.n_hc),
        ("hyper_connection.sinkhorn_iterations", c.sinkhorn_iters),
        ("context_length", c.context_length),
    ];
    let floats = [
        ("expert_weights_scale", c.expert_weight_scale),
        ("swiglu_limit", c.swiglu_limit),
        ("attention.layer_norm_rms_epsilon", c.rms_eps),
        ("linear_attention.gate_lower_bound", c.kda_gate_lower_bound),
        ("hyper_connection.epsilon", c.hc_eps),
    ];
    let toks = ["bos_token_id", "eos_token_id", "eot_token_id", "eom_token_id"];
    let mut f = Vec::from(&b"GGUF"[..]);
    f.extend(3u32.to_le_bytes());
    f.extend(0u64.to_le_bytes());
    f.extend(((ints.len() + floats.len() + toks.len() + 2) as u64).to_le_bytes());
    key(&mut f, "general.architecture", 8);
    f.extend((ARCH.len() as u64).to_le_bytes());
    f.extend(ARCH.as_bytes());
    for (k, v) in &ints {
        key(&mut f, &format!("{ARCH}.{k}"), 10);
        f.extend((*v as u64).to_le_bytes());
    }
    for (k, v) in floats {
        key(&mut f, &format!("{ARCH}.{k}"), 6);
        f.extend(v.to_le_bytes());
    }
    key(&mut f, &format!("{ARCH}.layer_types"), 9);
    f.extend(4u32.to_le_bytes());
    f.extend((c.kinds.len() as u64).to_le_bytes());
    for k in &c.kinds {
        f.extend(u32::from(*k == LayerKind::Dsa).to_le_bytes());
    }
    for (k, v) in toks.iter().zip(tokens) {
        key(&mut f, &format!("tokenizer.ggml.{k}"), 10);
        f.extend(v.to_le_bytes());
    }
    f
}

#[test]
fn token_ids_are_checked_before_narrowing() {
    let c = Config::glm53_flash();
    let dir =
        std::env::temp_dir().join(format!("lily-glm53-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let config = |name: &str, tokens: [u64; 4]| {
        let p = dir.join(name);
        std::fs::write(&p, header(&c, tokens)).unwrap();
        Gguf::open(&p).and_then(|g| Config::from_gguf(&g))
    };
    let ids = [c.bos, c.eos, c.eot, c.eom].map(u64::from);
    let good = config("good.gguf", ids).unwrap();
    assert_eq!([good.bos, good.eos, good.eot, good.eom], [c.bos, c.eos, c.eot, c.eom]);
    for i in 0..4 {
        // 2^32 + id would narrow to a valid id.
        let mut t = ids;
        t[i] += 1 << 32;
        let e = config("wrapped.gguf", t).unwrap_err();
        assert!(format!("{e:#}").contains("is not a token id"), "{i}: {e:#}");
        let mut t = ids;
        t[i] = c.vocab as u64;
        let e = config("vocab.gguf", t).unwrap_err();
        assert!(format!("{e:#}").contains("outside the vocabulary"), "{i}: {e:#}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
