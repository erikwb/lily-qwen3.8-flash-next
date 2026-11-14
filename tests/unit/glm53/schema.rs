use super::*;

/// The type a test file gives a role: the published file's.
fn first_type(s: &Spec) -> GgmlType {
    if s.name.ends_with("ffn_down_exps.weight") { GgmlType::Q2K } else { s.types[0] }
}

/// Writes a GGUF holding only a tensor table (`tensors`: name, dims,
/// type) at 32-byte-aligned offsets; the data is a hole of the file
/// (sparse), so the ~96 GB table costs nothing.
fn write_table(
    name: &str,
    tensors: &[(String, Vec<u64>, GgmlType)],
) -> std::path::PathBuf {
    let mut f = Vec::from(&b"GGUF"[..]);
    f.extend(3u32.to_le_bytes());
    f.extend((tensors.len() as u64).to_le_bytes());
    f.extend(0u64.to_le_bytes());
    let mut offset = 0u64;
    for (n, dims, ty) in tensors {
        f.extend((n.len() as u64).to_le_bytes());
        f.extend(n.as_bytes());
        f.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            f.extend(d.to_le_bytes());
        }
        f.extend((*ty as u32).to_le_bytes());
        f.extend(offset.to_le_bytes());
        let rows: u64 = dims[1..].iter().product();
        let bytes = ty.row_bytes(dims[0] as usize).unwrap() as u64 * rows;
        offset = (offset + bytes).next_multiple_of(32);
    }
    let data = (f.len() as u64).next_multiple_of(32);
    let dir =
        std::env::temp_dir().join(format!("lily-glm53-schema-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.gguf"));
    let file = std::fs::File::create(&path).unwrap();
    use std::io::Write;
    (&file).write_all(&f).unwrap();
    file.set_len(data + offset).unwrap();
    path
}

fn table(c: &Config) -> Vec<(String, Vec<u64>, GgmlType)> {
    expected(c)
        .iter()
        .map(|s| (s.name.clone(), s.dims.clone(), first_type(s)))
        .collect()
}

fn check(name: &str, t: &[(String, Vec<u64>, GgmlType)]) -> Result<()> {
    let path = write_table(name, t);
    let r = Gguf::open(&path).and_then(|g| validate(&g, &Config::glm53_flash()));
    std::fs::remove_file(&path).ok();
    r
}

#[test]
fn the_published_layout_validates() {
    let c = Config::glm53_flash();
    let t = table(&c);
    assert!(t.len() > 1000, "{} tensors", t.len());
    check("published", &t).unwrap();
}

#[test]
fn shortened_or_retyped_tensors_are_refused() {
    let c = Config::glm53_flash();
    let base = table(&c);
    let with = |name: &str, f: &dyn Fn(&mut (String, Vec<u64>, GgmlType))| {
        let mut t = base.clone();
        let e = t.iter_mut().find(|e| e.0 == name).unwrap_or_else(|| panic!("{name}"));
        f(e);
        t
    };
    let cases: Vec<(&str, Vec<(String, Vec<u64>, GgmlType)>, &str)> = vec![
        // One HC scale where the kernel reads three.
        (
            "hc-scale",
            with("blk.0.hc_attn_scale.weight", &|e| e.1 = vec![1]),
            "dimensions",
        ),
        (
            "norm-short",
            with("blk.7.attn_norm.weight", &|e| e.1 = vec![4064]),
            "dimensions",
        ),
        (
            "embd-vocab",
            with("token_embd.weight", &|e| e.1 = vec![4096, 154_000]),
            "dimensions",
        ),
        (
            "output-vocab",
            with("output.weight", &|e| e.1 = vec![4096, 154_879]),
            "dimensions",
        ),
        (
            "experts-287",
            with("blk.10.ffn_down_exps.weight", &|e| e.1 = vec![2048, 4096, 287]),
            "dimensions",
        ),
        (
            "gate-rows",
            with("blk.4.ffn_gate_exps.weight", &|e| e.1 = vec![4096, 1024, 288]),
            "dimensions",
        ),
        (
            "k_b-heads",
            with("blk.3.attn_k_b.weight", &|e| e.1 = vec![256, 512, 32]),
            "dimensions",
        ),
        (
            "v_b-layout",
            with("blk.3.attn_v_b.weight", &|e| e.1 = vec![256, 512, 64]),
            "dimensions",
        ),
        (
            "conv-rank",
            with("blk.1.kda_q_conv.weight", &|e| e.1 = vec![4, 8192]),
            "dimensions",
        ),
        (
            "ape-type",
            with("blk.3.indexer.pool_ape.weight", &|e| e.2 = GgmlType::F32),
            "is F32",
        ),
        ("fn-type", with("blk.5.hc_ffn_fn.weight", &|e| e.2 = GgmlType::F32), "is F32"),
        ("norm-type", with("output_norm.weight", &|e| e.2 = GgmlType::Bf16), "is BF16"),
        ("embd-type", with("token_embd.weight", &|e| e.2 = GgmlType::Bf16), "is BF16"),
        (
            "expert-type",
            with("blk.20.ffn_up_exps.weight", &|e| e.2 = GgmlType::Q4K),
            "is Q4_K",
        ),
        ("dense-type", with("blk.0.kda_v.weight", &|e| e.2 = GgmlType::Q2K), "is Q2_K"),
        (
            "gate-up-mismatch",
            with("blk.30.ffn_up_exps.weight", &|e| e.2 = GgmlType::Q2K),
            "different types",
        ),
    ];
    // Each fused group with one member of another (allowed) type.
    let mut fused = 0;
    for names in fused_groups(&c) {
        let mut t = base.clone();
        let e = t
            .iter_mut()
            .find(|e| e.0 == names[1])
            .unwrap_or_else(|| panic!("{names:?}"));
        e.2 = if e.2 == GgmlType::Iq2Xxs {
            GgmlType::Q2K
        } else if e.2 == GgmlType::Q4K {
            GgmlType::Q8_0
        } else {
            GgmlType::Q4K
        };
        let e = check("fused", &t).expect_err(&names[1]);
        assert!(format!("{e:#}").contains("different types"), "{names:?}: {e:#}");
        fused += 1;
    }
    // At least one attention and one FFN group a layer.
    assert!(fused > 2 * c.n_layer, "{fused} fused groups");
    for (name, t, want) in cases {
        let e = check(name, &t).expect_err(name);
        assert!(format!("{e:#}").contains(want), "{name}: {e:#}");
    }
    // A missing tensor.
    let t: Vec<_> =
        base.iter().filter(|e| e.0 != "blk.45.nextn.hnorm.weight").cloned().collect();
    let e = check("missing", &t).expect_err("missing");
    assert!(format!("{e:#}").contains("has no tensor"), "{e:#}");
}

#[test]
fn tokenizer_and_stop_ids_must_fit_the_vocabulary() {
    fn kv_str(f: &mut Vec<u8>, k: &str, v: &str) {
        f.extend((k.len() as u64).to_le_bytes());
        f.extend(k.as_bytes());
        f.extend(8u32.to_le_bytes());
        f.extend((v.len() as u64).to_le_bytes());
        f.extend(v.as_bytes());
    }
    fn kv_u32(f: &mut Vec<u8>, k: &str, v: u32) {
        f.extend((k.len() as u64).to_le_bytes());
        f.extend(k.as_bytes());
        f.extend(4u32.to_le_bytes());
        f.extend(v.to_le_bytes());
    }
    let header = |vocab: u32, eos: u32| {
        let mut f = Vec::from(&b"GGUF"[..]);
        f.extend(3u32.to_le_bytes());
        f.extend(0u64.to_le_bytes());
        f.extend(8u64.to_le_bytes());
        kv_str(&mut f, "tokenizer.ggml.model", "gpt2");
        kv_str(&mut f, "tokenizer.ggml.pre", "glm4");
        // Three tokens, no merges.
        let k = "tokenizer.ggml.tokens";
        f.extend((k.len() as u64).to_le_bytes());
        f.extend(k.as_bytes());
        f.extend(9u32.to_le_bytes());
        f.extend(8u32.to_le_bytes());
        f.extend(3u64.to_le_bytes());
        for t in ["a", "b", "c"] {
            f.extend(1u64.to_le_bytes());
            f.extend(t.as_bytes());
        }
        let k = "tokenizer.ggml.merges";
        f.extend((k.len() as u64).to_le_bytes());
        f.extend(k.as_bytes());
        f.extend(9u32.to_le_bytes());
        f.extend(8u32.to_le_bytes());
        f.extend(0u64.to_le_bytes());
        kv_u32(&mut f, "glm5-next.vocab_size", vocab);
        kv_u32(&mut f, "tokenizer.ggml.eos_token_id", eos);
        kv_u32(&mut f, "tokenizer.ggml.eot_token_id", 1);
        kv_u32(&mut f, "tokenizer.ggml.eom_token_id", 2);
        f
    };
    let dir =
        std::env::temp_dir().join(format!("lily-glm53-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let open = |name: &str, bytes: Vec<u8>| {
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        Gguf::open(&p).unwrap()
    };
    let good = open("good.gguf", header(3, 0));
    crate::glm53::tokenizer::tokenizer_json(&good).unwrap();
    assert_eq!(crate::glm53::tokenizer::stop_ids(&good).unwrap(), [0, 1, 2]);
    let short = open("short.gguf", header(5, 0));
    let e = crate::glm53::tokenizer::tokenizer_json(&short).unwrap_err();
    assert!(format!("{e:#}").contains("vocabulary"), "{e:#}");
    let stop = open("stop.gguf", header(3, 3));
    let e = crate::glm53::tokenizer::stop_ids(&stop).unwrap_err();
    assert!(format!("{e:#}").contains("outside the vocabulary"), "{e:#}");
    std::fs::remove_dir_all(&dir).ok();
}
