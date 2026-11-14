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
