use super::*;

/// A held instance lock refuses the GLM load before the file is opened or
/// anything mapped: the refusal, not "no such file", for a path that does
/// not exist. (`Glm53Model::load` and the engine's `LanguageModel::load`,
/// which calls it, take the lock through `load_locked`.)
#[test]
fn a_held_instance_lock_refuses_the_load_before_mapping() {
    use crate::instance::{AlreadyRunning, InstanceLock};
    let dir =
        std::env::temp_dir().join(format!("lily-glm53-lock-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let lock = dir.join("instance.lock");
    let ctx = MetalContext::new().expect("context");
    let missing = dir.join("no-such-model.gguf");
    let held = InstanceLock::acquire_at(&lock).unwrap();
    let e = Glm53Model::load_locked(&ctx, &missing, || {
        InstanceLock::acquire_at(&lock).map(drop)
    })
    .err()
    .expect("refused");
    assert!(e.chain().any(|c| c.is::<AlreadyRunning>()), "{e:#}");
    assert_eq!(crate::instance::exit_status(&e), crate::instance::EXIT_ALREADY_RUNNING);
    drop(held);
    // With the lock free, the same load gets as far as opening the file.
    let e = Glm53Model::load_locked(&ctx, &missing, || {
        InstanceLock::acquire_at(&lock).map(drop)
    })
    .err()
    .expect("no file");
    assert!(!e.chain().any(|c| c.is::<AlreadyRunning>()), "{e:#}");
    assert!(format!("{e:#}").contains("no-such-model"), "{e:#}");
    std::fs::remove_dir_all(&dir).ok();
}

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
    let model = <Glm53Model as LanguageModel>::load(
        &ctx,
        std::path::Path::new(&path),
        &options,
    )
    .expect("load");
    let mut st = LanguageModel::new_state(&model, &ctx, 64).expect("state");
    let mut sc =
        LanguageModel::new_scratch_with_capacity(&model, &ctx, 0).expect("scratch");
    let prompt: Vec<u32> =
        (0..PREFILL_ROWS as u32 + 37).map(|i| 1000 + i % 5000).collect();
    let params =
        SamplingParams { temperature: 0.7, top_p: 0.9, ..SamplingParams::greedy() };
    LanguageModel::prefill(
        &model,
        &ctx,
        &mut st,
        &mut sc,
        &prompt,
        Some(Draw { params: &params, step: 0 }),
    )
    .expect("prefill");
    assert_eq!(st.pos(), prompt.len());
    let mut last = sc.next_token.to_u32().unwrap()[0];
    let mut proposals = model
        .draft_initial(&ctx, &mut st, &mut sc, last, 1, &params, 1)
        .expect("draft");
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

/// Loads the published file for the real-weight tests.
fn real_model(ctx: &MetalContext) -> Glm53Model {
    use crate::engine::{LanguageModel, LoadOptions};
    let path = std::env::var("LILY_GLM53_GGUF").expect("LILY_GLM53_GGUF");
    let options = LoadOptions { mtp_drafts: 1, ..Default::default() };
    <Glm53Model as LanguageModel>::load(ctx, std::path::Path::new(&path), &options).expect("load")
}

/// A prompt of `n` ordinary text tokens.
fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 7919) % 50_000).collect()
}

/// prefill_until stops at a run boundary before committing it and reports
/// what it fed; the rest fed afterwards gives exactly what one prefill of
/// the whole prompt gives.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn prefill_until_stops_at_run_boundaries_and_resumes_exactly() {
    use crate::engine::{DecodeStateApi, Draw, LanguageModel};
    use crate::kernels::sample::SamplingParams;
    let ctx = MetalContext::new().expect("context");
    let m = real_model(&ctx);
    assert_eq!(m.prefill_chunk_tokens(), Some(PREFILL_ROWS));
    let p = prompt(2 * PREFILL_ROWS + 100);
    let greedy = SamplingParams::greedy();
    let draw = || Some(Draw { params: &greedy, step: 0 });
    let mut sc = m.new_scratch(&ctx).unwrap();
    let mut st = LanguageModel::new_state(&m, &ctx, p.len()).unwrap();
    assert_eq!(m.prefill_until(&ctx, &mut st, &mut sc, &p, None, &|| true).unwrap(), 0);
    assert_eq!(st.pos(), 0);
    let asked = std::cell::Cell::new(0);
    let stop = || {
        asked.set(asked.get() + 1);
        asked.get() > 1
    };
    let fed = m.prefill_until(&ctx, &mut st, &mut sc, &p, None, &stop).unwrap();
    assert_eq!((fed, st.pos()), (PREFILL_ROWS, PREFILL_ROWS));
    LanguageModel::prefill(&m, &ctx, &mut st, &mut sc, &p[fed..], draw()).unwrap();
    let resumed = sc.logits_last.to_f32().unwrap();
    let resumed_draw = sc.next_token.to_u32().unwrap()[0];
    let mut whole = LanguageModel::new_state(&m, &ctx, p.len()).unwrap();
    LanguageModel::prefill(&m, &ctx, &mut whole, &mut sc, &p, draw()).unwrap();
    assert_eq!(sc.logits_last.to_f32().unwrap(), resumed, "same runs, same logits");
    assert_eq!(sc.next_token.to_u32().unwrap()[0], resumed_draw);
}
