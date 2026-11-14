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
    // Sampling, not greedy (greedy() sets top_k 1), with penalties.
    let params = SamplingParams {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        presence_penalty: 0.3,
        frequency_penalty: 0.2,
        repetition_penalty: 1.05,
        seed: 1234,
        ..SamplingParams::greedy()
    };
    assert!(!params.is_greedy());
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
    <Glm53Model as LanguageModel>::load(ctx, std::path::Path::new(&path), &options)
        .expect("load")
}

/// A prompt of `n` ordinary text tokens.
fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 7919) % 50_000).collect()
}

/// Relative L2 distance of two logit vectors.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (f64::from(x) - f64::from(y)).powi(2);
        n += f64::from(y).powi(2);
    }
    (d / n).sqrt()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b }) as u32
}

/// Ordinary decoding: the logits after each of `tokens`, fed one at a time
/// after `prompt`.
fn ordinary(
    m: &Glm53Model,
    ctx: &MetalContext,
    prompt: &[u32],
    tokens: &[u32],
) -> Vec<Vec<f32>> {
    let mut st = m.new_state(ctx, prompt.len() + tokens.len() + 8).unwrap();
    let mut sc = m.new_scratch(ctx).unwrap();
    m.prefill(ctx, &mut st, &mut sc, prompt).unwrap();
    tokens
        .iter()
        .map(|&t| {
            m.forward(ctx, &mut st, &mut sc, &[t], true).unwrap();
            sc.logits_last.to_f32().unwrap()
        })
        .collect()
}

/// Continues `st` ordinarily with `tokens` and checks each step's logits
/// against `want`: the same top token and a relative L2 distance within
/// 5e-3 (the verify pass runs KDA's multi-row kernels where ordinary
/// decoding runs the one-row kernel, so the sums differ in the last bits).
fn continue_and_compare(
    m: &Glm53Model,
    ctx: &MetalContext,
    st: &mut Glm53State,
    sc: &mut Scratch,
    tokens: &[u32],
    want: &[Vec<f32>],
    case: &str,
) {
    for (k, (&t, w)) in tokens.iter().zip(want).enumerate() {
        m.forward(ctx, st, sc, &[t], true).unwrap();
        let got = sc.logits_last.to_f32().unwrap();
        let r = rel(&got, w);
        eprintln!("{case}: step {k} relative L2 {r:.2e}");
        assert!(r < 5e-3, "{case}: step {k} logits differ (relative L2 {r:.2e})");
        assert_eq!(argmax(&got), argmax(w), "{case}: step {k} top token");
    }
}

/// Speculative steps leave the state ordinary decoding would have, for an
/// accepted draft, a rejected one, a confirmed draft cut at the first row
/// (the generation ends there), and a rejected row that completes an
/// indexer pool past the dense window; and a snapshot taken before a step
/// restores, and forks, to that state.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn speculative_steps_match_ordinary_decoding() {
    use crate::engine::{DecodeStateApi, LanguageModel};
    use crate::kernels::sample::SamplingParams;
    let ctx = MetalContext::new().expect("context");
    let m = real_model(&ctx);
    let greedy = SamplingParams::greedy();
    // 2102 = 2 mod 4 and past 2051: the verify pass's row 1 (position 2103)
    // completes a pool and the attention is sparse.
    for (n, cases) in
        [(37usize, &["accept", "reject", "first-row"][..]), (2102, &["reject"][..])]
    {
        let p = prompt(n);
        // The greedy continuation; t0 is what the prompt's logits pick.
        let mut st = m.new_state(&ctx, n + 16).unwrap();
        let mut sc = m.new_scratch(&ctx).unwrap();
        m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
        let mut toks = vec![argmax(&sc.logits_last.to_f32().unwrap())];
        for _ in 0..6 {
            let last = *toks.last().unwrap();
            m.forward(&ctx, &mut st, &mut sc, &[last], true).unwrap();
            toks.push(argmax(&sc.logits_last.to_f32().unwrap()));
        }
        drop((st, sc));
        let want = ordinary(&m, &ctx, &p, &toks);
        for &case in cases {
            let mut st = LanguageModel::new_state(&m, &ctx, n + 16).unwrap();
            let mut sc = m.new_scratch(&ctx).unwrap();
            m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
            m.draft_initial(&ctx, &mut st, &mut sc, toks[0], 1, &greedy, 1).unwrap();
            // The draft: the true next token, or a wrong one.
            let draft =
                if case == "reject" { (toks[1] + 1) % 1000 + 5 } else { toks[1] };
            assert_eq!(draft == toks[1], case != "reject");
            let (draws, pending) = m
                .verify(&ctx, &mut st, &mut sc, toks[0], &[draft], &greedy, 1, 1)
                .unwrap();
            assert_eq!(draws[0], toks[1], "{case} n={n}: row 0 draw");
            if case == "accept" {
                assert_eq!(draws[1], toks[2], "{case} n={n}: row 1 draw");
            }
            let kept = usize::from(case == "accept");
            m.finish_speculation(&ctx, &mut st, &mut sc, kept, true, pending).unwrap();
            assert_eq!(st.pos(), n + 1 + kept);
            let label = format!("{case} n={n}");
            continue_and_compare(
                &m,
                &ctx,
                &mut st,
                &mut sc,
                &toks[1 + kept..],
                &want[1 + kept..],
                &label,
            );
        }
        if n == 37 {
            let mut st = LanguageModel::new_state(&m, &ctx, n + 16).unwrap();
            let mut sc = m.new_scratch(&ctx).unwrap();
            m.prefill(&ctx, &mut st, &mut sc, &p).unwrap();
            let snap = st.snapshot(&ctx).unwrap();
            m.draft_initial(&ctx, &mut st, &mut sc, toks[0], 1, &greedy, 1).unwrap();
            let (_, pending) = m
                .verify(&ctx, &mut st, &mut sc, toks[0], &[toks[1]], &greedy, 1, 1)
                .unwrap();
            m.finish_speculation(&ctx, &mut st, &mut sc, 1, true, pending).unwrap();
            let mut fork = LanguageModel::new_state(&m, &ctx, n + 16).unwrap();
            fork.copy_prefix_from(&ctx, &st, n).unwrap();
            fork.restore(&ctx, &snap).unwrap();
            st.restore(&ctx, &snap).unwrap();
            assert_eq!((st.pos(), fork.pos()), (n, n));
            continue_and_compare(&m, &ctx, &mut st, &mut sc, &toks, &want, "restored");
            continue_and_compare(&m, &ctx, &mut fork, &mut sc, &toks, &want, "forked");
        }
    }
}

/// One ordinary sampled decode step: feeds `input`, draws with `params` at
/// request step `step`, returns the draw.
fn sampled_step(
    m: &Glm53Model,
    ctx: &MetalContext,
    st: &mut Glm53State,
    sc: &Scratch,
    input: u32,
    params: &crate::kernels::sample::SamplingParams,
    step: usize,
) -> u32 {
    use crate::engine::{DecodeStateApi, Draw, LanguageModel};
    sc.next_token
        .view(0, &[1])
        .unwrap()
        .write_bytes(bytemuck::bytes_of(&input))
        .unwrap();
    let encoded =
        m.encode_decode_step(ctx, st, sc, 0, 1, Draw { params, step }).unwrap();
    m.prepare_step_inputs(st, sc, input).unwrap();
    let pending = encoded.commit().unwrap();
    st.advance(1);
    pending.wait().unwrap();
    sc.next_token.to_u32().unwrap()[1]
}

/// Sampled speculation emits what ordinary sampled decoding emits, with
/// the same penalty counts: the same seed, steps and penalties; a forced
/// rejection (its row-1 draw is discarded and uncounted), then a forced
/// acceptance (both rows' draws kept), then ordinary steps.
#[test]
#[ignore = "requires LILY_GLM53_GGUF"]
fn sampled_speculation_matches_ordinary_sampling() {
    use crate::engine::{Draw, LanguageModel, ScratchApi};
    use crate::kernels::sample::SamplingParams;
    let ctx = MetalContext::new().expect("context");
    let m = real_model(&ctx);
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 40,
        top_p: 0.95,
        presence_penalty: 0.5,
        frequency_penalty: 0.5,
        repetition_penalty: 1.1,
        seed: 4242,
        ..SamplingParams::greedy()
    };
    assert!(!params.is_greedy());
    // The patterned prompt: after a pseudo-random one, two of layer 7's
    // experts tied to 1e-6 at the top-8 boundary, and the verify pass's
    // rounding (any two-row pass's) picked the other one.
    let p = prompt(37);
    let steps = 7;
    // Ordinary: t0 from the prompt, then one draw a step, and the counts
    // after each.
    let mut st = LanguageModel::new_state(&m, &ctx, p.len() + 16).unwrap();
    let mut sc = m.new_scratch(&ctx).unwrap();
    sc.begin_request();
    let draw = Some(Draw { params: &params, step: 0 });
    LanguageModel::prefill(&m, &ctx, &mut st, &mut sc, &p, draw).unwrap();
    let mut toks = vec![sc.next_token.to_u32().unwrap()[0]];
    let mut counts = vec![sc.sampler().counts().unwrap()];
    let mut logits = vec![sc.logits_last.to_f32().unwrap()];
    let mut off_argmax =
        usize::from(toks[0] != argmax(&sc.logits_last.to_f32().unwrap()));
    for k in 1..steps {
        let t = sampled_step(&m, &ctx, &mut st, &sc, toks[k - 1], &params, k);
        off_argmax += usize::from(t != argmax(&sc.logits_last.to_f32().unwrap()));
        toks.push(t);
        counts.push(sc.sampler().counts().unwrap());
        logits.push(sc.logits_last.to_f32().unwrap());
    }
    eprintln!("sampled: {toks:?}, {off_argmax} draws off the argmax");
    assert!(off_argmax > 0, "the run never left the argmax");
    assert!(counts[steps - 1].iter().sum::<u32>() > 0, "penalties count draws");
    drop((st, sc));
    // Speculative.
    let mut st = LanguageModel::new_state(&m, &ctx, p.len() + 16).unwrap();
    let mut sc = m.new_scratch(&ctx).unwrap();
    sc.begin_request();
    LanguageModel::prefill(&m, &ctx, &mut st, &mut sc, &p, draw).unwrap();
    assert_eq!(sc.next_token.to_u32().unwrap()[0], toks[0]);
    m.draft_initial(&ctx, &mut st, &mut sc, toks[0], 1, &params, 1).unwrap();
    // Rejected: a draft that is not t1.
    let wrong = if toks[1] == 1000 { 1001 } else { 1000 };
    let (draws, pending) =
        m.verify(&ctx, &mut st, &mut sc, toks[0], &[wrong], &params, 1, 1).unwrap();
    assert_eq!(draws[0], toks[1], "rejected step: row 0 draw");
    m.finish_speculation(&ctx, &mut st, &mut sc, 0, true, pending).unwrap();
    assert_eq!(sc.sampler().counts().unwrap(), counts[1], "counts after a rejection");
    // Accepted: the draft is t2, so rows 0 and 1 draw t2 and t3.
    let (draws, pending) =
        m.verify(&ctx, &mut st, &mut sc, toks[1], &[toks[2]], &params, 2, 1).unwrap();
    assert_eq!(draws, [toks[2], toks[3]], "accepted step: both rows' draws");
    m.finish_speculation(&ctx, &mut st, &mut sc, 1, false, pending).unwrap();
    assert_eq!(sc.sampler().counts().unwrap(), counts[3], "counts after an acceptance");
    // Ordinary steps after both.
    for k in 4..steps {
        let t = sampled_step(&m, &ctx, &mut st, &sc, toks[k - 1], &params, k);
        let got = sc.logits_last.to_f32().unwrap();
        let r = rel(&got, &logits[k]);
        eprintln!("step {k} after speculation: relative L2 {r:.2e}");
        assert!(r < 5e-3, "step {k} logits differ (relative L2 {r:.2e})");
        assert_eq!(t, toks[k], "step {k} after speculation");
        assert_eq!(sc.sampler().counts().unwrap(), counts[k], "step {k} counts");
    }
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
