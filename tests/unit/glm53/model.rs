use super::*;

/// Every staging path the engine uses fits the scratch it writes, at every
/// row count it is called with: a verify pass (1 or 2 rows, up to
/// MAX_ROWS), a decode step, a prefill run up to PREFILL_ROWS, the MTP
/// step. (The verify pass once wrote MAX_ROWS words into buffers sized for
/// a prefill run and failed every speculative request.)
#[test]
fn scratch_staging_fits_every_caller() {
    let ctx = MetalContext::new().expect("context");
    let cfg = Config::glm53_flash();
    let sc = Scratch::new(&ctx, &cfg).expect("scratch");
    for drafts in 0..MAX_ROWS {
        let d: Vec<u32> = (0..drafts as u32).map(|i| 100 + i).collect();
        sc.stage_spec_tokens(7, &d).unwrap();
        let got = sc.spec_tokens.to_u32().unwrap();
        assert_eq!(got[0], 7);
        assert_eq!(&got[1..=drafts], &d[..]);
        assert!(got[drafts + 1..].iter().all(|&v| v == 0));
        // Every row's draw has a slot.
        assert!(sc.draws.numel() > drafts);
    }
    assert!(sc.stage_spec_tokens(7, &[1; MAX_ROWS]).is_err());
    for n in [1, MAX_ROWS, MAX_ROWS + 1, PREFILL_ROWS] {
        let t: Vec<u32> = (0..n as u32).collect();
        sc.stage_tokens(&t).unwrap();
        assert_eq!(&sc.tokens.to_u32().unwrap()[..n], &t[..]);
    }
    assert!(sc.stage_tokens(&vec![0; PREFILL_ROWS + 1]).is_err());
    sc.stage_mtp_token(42).unwrap();
    assert_eq!(sc.mtp_io.to_u32().unwrap(), [42, 0]);
    assert!(sc.next_token.numel() >= 2);
}

/// The engine's whole speculative cycle on the published file, after a
/// prompt longer than one prefill run: `LILY_GLM53_GGUF=<path> cargo test
/// --release --lib glm53::model -- --ignored --test-threads=1`. The verify
/// pass's staging once failed here on every request.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn speculative_cycle_after_a_long_prefill() {
    use crate::engine::{DecodeStateApi, Draw, LanguageModel, LoadOptions};
    use crate::kernels::sample::SamplingParams;
    let path = std::env::var("LILY_GLM53_GGUF").expect("LILY_GLM53_GGUF");
    let ctx = MetalContext::new().expect("context");
    let options = LoadOptions { mtp_drafts: 1, ..Default::default() };
    let model = <Glm53Model as LanguageModel>::load(&ctx, std::path::Path::new(&path), &options)
        .expect("load");
    let mut st = LanguageModel::new_state(&model, &ctx, 64).expect("state");
    let mut sc = LanguageModel::new_scratch_with_capacity(&model, &ctx, 0).expect("scratch");
    let prompt: Vec<u32> = (0..PREFILL_ROWS as u32 + 37).map(|i| 1000 + i % 5000).collect();
    let params = SamplingParams { temperature: 0.7, top_p: 0.9, ..SamplingParams::greedy() };
    LanguageModel::prefill(&model, &ctx, &mut st, &mut sc, &prompt, Some(Draw { params: &params, step: 0 }))
        .expect("prefill");
    assert_eq!(st.pos(), prompt.len());
    let mut last = sc.next_token.to_u32().unwrap()[0];
    let mut proposals = model.draft_initial(&ctx, &mut st, &mut sc, last, 1, &params, 1).expect("draft");
    for step in 0..2 {
        let pos = st.pos();
        let (draws, pending) = model
            .verify(&ctx, &mut st, &mut sc, last, &proposals, &params, 2 + 2 * step, 1)
            .expect("verify");
        assert_eq!(draws.len(), 2);
        assert!(draws.iter().all(|&t| (t as usize) < model.cfg.vocab));
        let kept = usize::from(draws[0] == proposals[0]);
        proposals = model
            .finish_speculation(&ctx, &mut st, &mut sc, kept, true, pending)
            .expect("finish");
        assert_eq!(st.pos(), pos + 1 + kept);
        assert_eq!(proposals.len(), 1);
        last = draws[kept];
    }
}
