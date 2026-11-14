use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref::assert_close;
use crate::gguf::dequantize;
use crate::glm53::reference as r;

fn ctx() -> MetalContext {
    MetalContext::new().expect("metal context")
}

fn f32s(rng: &mut StdRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn put_f16(b: &mut [u8], at: usize, v: f32) {
    b[at..at + 2].copy_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
}

/// Random blocks of `ty` with sane scales: random code bytes, f16 scales
/// small enough that the dot products stay well inside f32.
fn random_blocks(rng: &mut StdRng, ty: GgmlType, values: usize) -> Vec<u8> {
    let nb = values / ty.block_values();
    let bb = ty.block_bytes();
    let mut b: Vec<u8> = (0..nb * bb).map(|_| rng.r#gen()).collect();
    for blk in b.chunks_exact_mut(bb) {
        match ty {
            GgmlType::Q8_0 => put_f16(blk, 0, rng.gen_range(0.001..0.02)),
            GgmlType::Q4K => {
                put_f16(blk, 0, rng.gen_range(0.001..0.01));
                put_f16(blk, 2, rng.gen_range(0.001..0.01));
            }
            GgmlType::Q2K => {
                put_f16(blk, 80, rng.gen_range(0.001..0.02));
                put_f16(blk, 82, rng.gen_range(0.001..0.02));
            }
            GgmlType::Iq2Xxs => put_f16(blk, 0, rng.gen_range(0.001..0.01)),
            GgmlType::Bf16 => {
                let v = half::bf16::from_f32(rng.gen_range(-0.1..0.1));
                blk.copy_from_slice(&v.to_bits().to_le_bytes());
            }
            GgmlType::F32 => {
                blk.copy_from_slice(&rng.gen_range(-0.1f32..0.1).to_le_bytes())
            }
            GgmlType::F16 => unreachable!(),
        }
    }
    b
}

fn decoded(ty: GgmlType, raw: &[u8], values: usize) -> r::Mat {
    let _ = values;
    let n = raw.len() / ty.block_bytes() * ty.block_values();
    let mut w = vec![0f32; n];
    dequantize(ty, raw, &mut w).unwrap();
    r::Mat { rows: 0, cols: 0, w }
}

fn mat(ty: GgmlType, raw: &[u8], rows: usize, cols: usize) -> r::Mat {
    let mut m = decoded(ty, raw, rows * cols);
    m.rows = rows;
    m.cols = cols;
    m
}

fn run(ctx: &MetalContext, f: impl FnOnce(&ComputePass<'_>)) {
    let pass = ctx.begin().expect("pass");
    f(&pass);
    pass.commit_wait().expect("commit");
}

fn tensor(ctx: &MetalContext, v: &[f32]) -> Tensor {
    Tensor::from_f32(ctx, v, &[v.len()]).expect("tensor")
}

fn zeros(ctx: &MetalContext, n: usize) -> Tensor {
    Tensor::zeros(ctx, &[n], DType::F32).expect("zeros")
}

#[test]
fn gemv_matches_cpu_for_every_type_and_row_count() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(53);
    for ty in [
        GgmlType::Q8_0,
        GgmlType::Q4K,
        GgmlType::Q2K,
        GgmlType::Iq2Xxs,
        GgmlType::Bf16,
        GgmlType::F32,
    ] {
        for (cols, rows) in [(4096usize, 72usize), (2048, 40), (1536, 24), (256, 100)] {
            if !cols.is_multiple_of(ty.block_values()) {
                continue;
            }
            let raw = random_blocks(&mut rng, ty, rows * cols);
            let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
            let cpu = mat(ty, &raw, rows, cols);
            for m in 1..=GEMV_MAX_ROWS {
                let x = f32s(&mut rng, m * cols, -1.0, 1.0);
                let tx = tensor(&ctx, &x);
                let ty_out = zeros(&ctx, m * rows);
                run(&ctx, |p| gemv(&ctx, p, &w, &tx, &ty_out, m).unwrap());
                let expected: Vec<f32> =
                    (0..m).flat_map(|i| cpu.mv(&x[i * cols..(i + 1) * cols])).collect();
                assert_close(&ty_out.to_f32().unwrap(), &expected, 1e-4, 1e-4);
            }
        }
    }
}

#[test]
fn gemv_grouped_matches_cpu() {
    // attn_k_b's shape: 64 heads x 512 rows of 256, input per head.
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(54);
    let (heads, per, cols) = (8, 512, 256);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, heads * per * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, heads * per).unwrap();
    let cpu = mat(GgmlType::Q8_0, &raw, heads * per, cols);
    for m in [1, 3] {
        let x = f32s(&mut rng, m * heads * cols, -1.0, 1.0);
        let tx = tensor(&ctx, &x);
        let out = zeros(&ctx, m * heads * per);
        let layout = GemvLayout {
            x_group: per,
            x_row_stride: heads * cols,
            y_row_stride: heads * per,
        };
        run(&ctx, |p| gemv_with(&ctx, p, &w, &tx, &out, m, layout).unwrap());
        let expected: Vec<f32> = (0..m)
            .flat_map(|i| {
                cpu.mv_grouped(&x[i * heads * cols..(i + 1) * heads * cols], per)
            })
            .collect();
        assert_close(&out.to_f32().unwrap(), &expected, 1e-4, 1e-4);
    }
}

#[test]
fn moe_gate_up_and_down_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(55);
    let (experts, hidden, ff, used) = (12, 512, 256, 8);
    let g_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let u_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let d_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * hidden * ff);
    let gw = QWeight::from_bytes(&ctx, &g_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let uw = QWeight::from_bytes(&ctx, &u_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let dw =
        QWeight::from_bytes(&ctx, &d_raw, GgmlType::Q2K, ff, experts * hidden).unwrap();
    let gc = mat(GgmlType::Iq2Xxs, &g_raw, experts * ff, hidden);
    let uc = mat(GgmlType::Iq2Xxs, &u_raw, experts * ff, hidden);
    let dc = mat(GgmlType::Q2K, &d_raw, experts * hidden, ff);
    let m = 2;
    let x = f32s(&mut rng, m * hidden, -2.0, 2.0);
    let sel: Vec<u32> =
        (0..m * used).map(|s| ((s * 5 + s / used) % experts) as u32).collect();
    let wts = f32s(&mut rng, m * used, 0.1, 0.6);
    let tx = tensor(&ctx, &x);
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &wts);
    let mid = zeros(&ctx, m * used * ff);
    let out = zeros(&ctx, m * hidden);
    let limit = 0.05;
    run(&ctx, |p| {
        moe_gate_up(&ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, m, used, limit)
            .unwrap();
        moe_down(&ctx, p, &dw, hidden, &mid, &tsel, None, &out, m, used).unwrap();
    });
    let mut emid = vec![0f32; m * used * ff];
    let mut eout = vec![0f32; m * hidden];
    for i in 0..m {
        let xi = &x[i * hidden..(i + 1) * hidden];
        for s in 0..used {
            let e = sel[i * used + s] as usize;
            for row in 0..ff {
                let g = r::dot(gc.row(e * ff + row), xi);
                let u = r::dot(uc.row(e * ff + row), xi);
                emid[(i * used + s) * ff + row] =
                    r::swiglu(g, u, limit) * wts[i * used + s];
            }
        }
        for row in 0..hidden {
            eout[i * hidden + row] = (0..used)
                .map(|s| {
                    let e = sel[i * used + s] as usize;
                    r::dot(
                        dc.row(e * hidden + row),
                        &emid[(i * used + s) * ff..(i * used + s + 1) * ff],
                    )
                })
                .sum();
        }
    }
    assert_close(&mid.to_f32().unwrap(), &emid, 1e-5, 1e-4);
    assert_close(&out.to_f32().unwrap(), &eout, 1e-5, 1e-3);
    // With the shared expert's output added in place: exactly routed + add.
    let shared = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let both = tensor(&ctx, &shared);
    run(&ctx, |p| {
        moe_down(&ctx, p, &dw, hidden, &mid, &tsel, Some(&both), &both, m, used)
            .unwrap()
    });
    let routed = out.to_f32().unwrap();
    let expect: Vec<f32> = routed.iter().zip(&shared).map(|(a, b)| a + b).collect();
    assert_eq!(both.to_f32().unwrap(), expect);
}

#[test]
fn router_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(56);
    let (m, e, k) = (3, 288, 8);
    let logits = f32s(&mut rng, m * e, -4.0, 4.0);
    let bias = f32s(&mut rng, e, -0.05, 0.05);
    let (tl, tb) = (tensor(&ctx, &logits), tensor(&ctx, &bias));
    let sel = Tensor::zeros(&ctx, &[m * k], DType::U32).unwrap();
    let w = zeros(&ctx, m * k);
    run(&ctx, |p| router_topk(&ctx, p, &tl, &tb, &sel, &w, m, e, k, 2.5).unwrap());
    let gsel = sel.to_u32().unwrap();
    let gw = w.to_f32().unwrap();
    for i in 0..m {
        let (esel, ew) = r::route(&logits[i * e..(i + 1) * e], &bias, k, 2.5);
        assert_eq!(
            gsel[i * k..(i + 1) * k],
            esel.iter().map(|&v| v as u32).collect::<Vec<_>>()[..]
        );
        assert_close(&gw[i * k..(i + 1) * k], &ew, 1e-6, 1e-5);
    }
}

#[test]
fn rmsnorm_hc_pre_post_mean_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(57);
    let cfg = crate::glm53::Config::glm53_flash();
    let (m, hidden) = (2, cfg.hidden);
    let h = f32s(&mut rng, m * 4 * hidden, -3.0, 3.0);
    let fn_raw = random_blocks(&mut rng, GgmlType::Bf16, 24 * 4 * hidden);
    let fn_w =
        QWeight::from_bytes(&ctx, &fn_raw, GgmlType::Bf16, 4 * hidden, 24).unwrap();
    let fn_cpu = mat(GgmlType::Bf16, &fn_raw, 24, 4 * hidden);
    let scale = [0.8f32, 1.2, 1.7];
    let base = f32s(&mut rng, 24, -0.5, 0.5);
    let norm = f32s(&mut rng, hidden, 0.5, 1.5);
    let block = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let th = tensor(&ctx, &h);
    let flat = zeros(&ctx, m * 4 * hidden);
    let mix = zeros(&ctx, m * 24);
    let (tsc, tba, tno) =
        (tensor(&ctx, &scale), tensor(&ctx, &base), tensor(&ctx, &norm));
    let split = zeros(&ctx, m * 24);
    let xn = zeros(&ctx, m * hidden);
    let tblock = tensor(&ctx, &block);
    let post_out = zeros(&ctx, m * 4 * hidden);
    let mean = zeros(&ctx, m * hidden);
    run(&ctx, |p| {
        rmsnorm(&ctx, p, &th, None, &flat, m, 4 * hidden, cfg.rms_eps).unwrap();
        gemv(&ctx, p, &fn_w, &flat, &mix, m).unwrap();
        hc_pre(
            &ctx,
            p,
            &mix,
            &tsc,
            &tba,
            &th,
            &tno,
            &split,
            &xn,
            m,
            hidden,
            20,
            1e-6,
            cfg.rms_eps,
        )
        .unwrap();
        hc_post(&ctx, p, &th, &tblock, &split, &post_out, m, hidden).unwrap();
        hc_mean(&ctx, p, &th, &mean, m, hidden).unwrap();
    });
    let gxn = xn.to_f32().unwrap();
    let gsplit = split.to_f32().unwrap();
    let gpost = post_out.to_f32().unwrap();
    let gmean = mean.to_f32().unwrap();
    for i in 0..m {
        let hi = &h[i * 4 * hidden..(i + 1) * 4 * hidden];
        let (exn, epost, ecomb) = r::hc_pre(&cfg, hi, &fn_cpu, &scale, &base, &norm);
        assert_close(&gxn[i * hidden..(i + 1) * hidden], &exn, 1e-4, 1e-3);
        assert_close(&gsplit[i * 24 + 4..i * 24 + 8], &epost, 1e-5, 1e-4);
        assert_close(&gsplit[i * 24 + 8..i * 24 + 24], &ecomb, 1e-5, 1e-4);
        // Post from the GPU's own split, so it is checked on its own.
        let gp = &gsplit[i * 24 + 4..i * 24 + 8];
        let gc = &gsplit[i * 24 + 8..i * 24 + 24];
        let ep = r::hc_post(hi, &block[i * hidden..(i + 1) * hidden], gp, gc, 4);
        assert_close(&gpost[i * 4 * hidden..(i + 1) * 4 * hidden], &ep, 1e-5, 1e-5);
        let em: Vec<f32> = (0..hidden)
            .map(|d| (0..4).map(|s| hi[s * hidden + d] * 0.25).sum())
            .collect();
        assert_close(&gmean[i * hidden..(i + 1) * hidden], &em, 1e-6, 1e-6);
    }
}

#[test]
fn embed_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(58);
    let (vocab, hidden) = (50, 4096);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, vocab * hidden);
    let table = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, hidden, vocab).unwrap();
    let cpu = mat(GgmlType::Q8_0, &raw, vocab, hidden);
    let toks = [7u32, 0, 49];
    let tt = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&toks), &[3], DType::U32)
        .unwrap();
    let out = zeros(&ctx, 3 * 4 * hidden);
    run(&ctx, |p| embed(&ctx, p, &table, &tt, &out, 3).unwrap());
    let expected: Vec<f32> =
        toks.iter().flat_map(|&t| cpu.row(t as usize).repeat(4)).collect();
    assert_eq!(out.to_f32().unwrap(), expected);
}

struct KdaCase {
    q_conv: Vec<f32>,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    o_norm: Vec<f32>,
}

fn kda_case(rng: &mut StdRng, heads: usize) -> KdaCase {
    let p = heads * 128;
    KdaCase {
        q_conv: f32s(rng, p * 4, -0.6, 0.6),
        k_conv: f32s(rng, p * 4, -0.6, 0.6),
        v_conv: f32s(rng, p * 4, -0.6, 0.6),
        a_log: f32s(rng, heads, -1.0, 1.0),
        dt_bias: f32s(rng, p, -1.0, 1.0),
        o_norm: f32s(rng, 128, 0.5, 1.5),
    }
}

/// Runs `tokens` KDA tokens on the GPU in chunks of `chunk` rows and on the
/// CPU one by one.
fn kda_compare(tokens: usize, chunk: usize, seed: u64) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cfg = crate::glm53::Config::glm53_flash();
    let heads = 4;
    cfg.kda_heads = heads;
    let p = heads * 128;
    let c = kda_case(&mut rng, heads);
    let t = |v: &Vec<f32>| tensor(&ctx, v);
    let (qc, kc, vc, al, dt, on) = (
        t(&c.q_conv),
        t(&c.k_conv),
        t(&c.v_conv),
        t(&c.a_log),
        t(&c.dt_bias),
        t(&c.o_norm),
    );
    let params = KdaParams {
        q_conv: &qc,
        k_conv: &kc,
        v_conv: &vc,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound: -5.0,
        eps: 1e-5,
    };
    let conv_state = zeros(&ctx, 9 * p);
    let state = zeros(&ctx, heads * 128 * 128);
    let mut st = r::KdaState::new(&cfg);
    let cp = r::KdaParams {
        q_conv: &c.q_conv,
        k_conv: &c.k_conv,
        v_conv: &c.v_conv,
        a_log: &c.a_log,
        dt_bias: &c.dt_bias,
        o_norm: &c.o_norm,
    };
    let mut done = 0;
    while done < tokens {
        let n = chunk.min(tokens - done);
        let q = f32s(&mut rng, n * p, -2.0, 2.0);
        let k = f32s(&mut rng, n * p, -2.0, 2.0);
        let v = f32s(&mut rng, n * p, -2.0, 2.0);
        let g = f32s(&mut rng, n * p, -3.0, 3.0);
        let b = f32s(&mut rng, n * heads, -2.0, 2.0);
        let og = f32s(&mut rng, n * p, -2.0, 2.0);
        let (tq, tk, tv, tg, tb, tog) = (t(&q), t(&k), t(&v), t(&g), t(&b), t(&og));
        let out = zeros(&ctx, n * p);
        let edges = zeros(&ctx, kda_edges_len(n, heads));
        let inputs = KdaInputs {
            q: &tq,
            k: &tk,
            v: &tv,
            gate: &tg,
            beta: &tb,
            out_gate: &tog,
            edges: Some(&edges),
        };
        run(&ctx, |pass| {
            kda(&ctx, pass, &params, &inputs, &conv_state, &state, &out, n, None)
                .unwrap()
        });
        let got = out.to_f32().unwrap();
        for j in 0..n {
            let s = j * p..(j + 1) * p;
            let e = r::kda_step(
                &cfg,
                &mut st,
                &cp,
                &q[s.clone()],
                &k[s.clone()],
                &v[s.clone()],
                &g[s.clone()],
                &b[j * heads..(j + 1) * heads],
                &og[s.clone()],
            );
            assert_close(&got[s], &e, 2e-4, 2e-3);
        }
        done += n;
    }
    assert_close(&state.to_f32().unwrap(), &st.s, 1e-4, 1e-3);
}

#[test]
fn kda_decode_matches_cpu() {
    kda_compare(6, 1, 59);
}

#[test]
fn kda_prefill_matches_cpu() {
    kda_compare(37, 16, 60);
}

/// A prompt run's prepare stage in KDA_CHUNK-token chunks gives the
/// single-chunk run's results bit for bit: output, prepared q/k/v/decay,
/// recurrent state, conv window and both snapshots, from a nonzero starting
/// state, at row counts around the chunk boundaries and with snapshots on
/// both sides of one.
#[test]
fn kda_prefill_chunks_are_bit_exact() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(76);
    let heads = 4;
    let p = heads * 128;
    let c = kda_case(&mut rng, heads);
    let t = |v: &Vec<f32>| tensor(&ctx, v);
    let (qc, kc, vc, al, dt, on) = (
        t(&c.q_conv),
        t(&c.k_conv),
        t(&c.v_conv),
        t(&c.a_log),
        t(&c.dt_bias),
        t(&c.o_norm),
    );
    let params = KdaParams {
        q_conv: &qc,
        k_conv: &kc,
        v_conv: &vc,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound: -5.0,
        eps: 1e-5,
    };
    let conv0 = f32s(&mut rng, 9 * p, -1.0, 1.0);
    let state0 = f32s(&mut rng, heads * 128 * 128, -0.2, 0.2);
    let bits = |x: &Tensor| -> Vec<u32> {
        x.to_f32().unwrap().iter().map(|v| v.to_bits()).collect()
    };
    let cases = [
        (31, Some(30)),
        (32, Some(31)),
        (33, Some(32)),
        (63, None),
        (64, Some(33)),
        (65, Some(64)),
    ];
    for (n, snap_row) in cases {
        let raw: Vec<Vec<f32>> = [p, p, p, p, heads, p]
            .iter()
            .map(|&w| f32s(&mut rng, n * w, -2.0, 2.0))
            .collect();
        let mut got = Vec::new();
        for chunk in [KDA_CHUNK, n] {
            let x: Vec<Tensor> = raw.iter().map(t).collect();
            let (conv, st) = (t(&conv0), t(&state0));
            let (out, edges) = (zeros(&ctx, n * p), zeros(&ctx, n.div_ceil(3) * 9 * p));
            let (sconv, sst) = (zeros(&ctx, 9 * p), zeros(&ctx, heads * 128 * 128));
            let inputs = KdaInputs {
                q: &x[0],
                k: &x[1],
                v: &x[2],
                gate: &x[3],
                beta: &x[4],
                out_gate: &x[5],
                edges: Some(&edges),
            };
            run(&ctx, |pass| {
                let snap =
                    snap_row.map(|row| KdaSnapshot { row, conv: &sconv, state: &sst });
                kda_chunked(
                    &ctx, pass, &params, &inputs, &conv, &st, &out, n, snap, chunk,
                )
                .unwrap()
            });
            let mut all: Vec<Vec<u32>> = x[..4].iter().map(bits).collect();
            all.extend([bits(&out), bits(&st), bits(&conv), bits(&sconv), bits(&sst)]);
            got.push(all);
        }
        for (k, name) in
            ["q", "k", "v", "decay", "out", "state", "conv", "snap conv", "snap state"]
                .iter()
                .enumerate()
        {
            assert!(got[0][k] == got[1][k], "{name} differs at {n} rows");
        }
    }
}

/// Runs of more than one prepare chunk (KDA_CHUNK rows): 75-row runs of
/// three chunks each, the second run continuing the first's state.
#[test]
fn kda_prefill_long_runs_match_cpu() {
    kda_compare(150, 75, 75);
}

#[test]
fn dsa_store_and_attend_match_cpu() {
    // 42 cached rows then 3 rows of queries (SIMT kernels); 25 then a run
    // of 20 (matrix units); runs from row 0 in partial tiles of the
    // padded widths 8 and 16.
    for (n, m, s) in [(45, 3, 64), (45, 20, 64), (5, 5, 8), (13, 13, 16)] {
        dsa_causal_case(n, m, s, false);
    }
}

/// Queries at and past f16's range on both routes (a tile with one goes
/// through f32 on the matrix route): finite, and as the CPU computes.
#[test]
fn dsa_attend_large_queries_stay_finite() {
    for m in [3, 20] {
        dsa_causal_case(45, m, 64, true);
    }
}

/// The matrix route's widths are padded to 8.
#[test]
fn dsa_attend_refuses_unpadded_prompt_widths() {
    let ctx = ctx();
    let (m, s) = (5, 5);
    let cache = Tensor::zeros(&ctx, &[8 * 512], DType::F16).unwrap();
    let qk = zeros(&ctx, m * 64 * 512);
    let (scores, denom) = (zeros(&ctx, m * 64 * 8), zeros(&ctx, m * 64));
    let (lat, partial) = (zeros(&ctx, m * 64 * 512), zeros(&ctx, 4 * 8 * 64 * 512));
    let pass = ctx.begin().unwrap();
    let r = dsa_attend(
        &ctx,
        &pass,
        &qk,
        &cache,
        Rows::Causal { pos0: 0 },
        &scores,
        &denom,
        &lat,
        &partial,
        m,
        s,
        1.0,
    );
    assert!(r.is_err());
    pass.commit_wait().unwrap();
}

fn dsa_causal_case(n: usize, m: usize, s: usize, big: bool) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(61);
    let cfg = crate::glm53::Config::glm53_flash();
    let kv_raw = f32s(&mut rng, n * 512, -2.0, 2.0);
    let w = f32s(&mut rng, 512, 0.5, 1.5);
    let cache = Tensor::zeros(&ctx, &[64 * 512], DType::F16).unwrap();
    let (tkv, tw) = (tensor(&ctx, &kv_raw), tensor(&ctx, &w));
    run(&ctx, |p| kv_store(&ctx, p, &tkv, &tw, &cache, 0, n, cfg.rms_eps).unwrap());
    let latent: Vec<f32> = (0..n)
        .flat_map(|t| {
            r::rmsnorm(&kv_raw[t * 512..(t + 1) * 512], Some(&w), cfg.rms_eps)
        })
        .map(r::round_f16)
        .collect();
    let gl: Vec<f32> = cache.contents()[..n * 1024]
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect();
    let diffs = gl.iter().zip(&latent).filter(|(a, b)| a != b).count();
    assert!(diffs * 1000 < latent.len(), "{diffs} f16 latent values differ");
    let mut qk = f32s(&mut rng, m * 64 * 512, -0.2, 0.2);
    if big {
        // f16's largest value, the first values past it, far past it, and
        // large values of both signs whose products cancel in part.
        let at = |i: usize, h: usize, j: usize| (i * 64 + h) * 512 + j;
        qk[at(0, 0, 0)] = 65536.0;
        qk[at(0, 1, 5)] = -65536.0;
        qk[at(1, 2, 7)] = 65520.0;
        qk[at(m - 1, 3, 0)] = 1.0e6;
        qk[at(m - 1, 3, 1)] = -1.0e6;
        for j in 0..512 {
            qk[at(1, 4, j)] = if j % 2 == 0 { 65504.0 } else { -65504.0 };
            qk[at(2, 5, j)] = 3.0e4;
        }
    }
    let tqk = tensor(&ctx, &qk);
    let scores = zeros(&ctx, m * 64 * s);
    let denom = zeros(&ctx, m * 64);
    let lat = zeros(&ctx, m * 64 * 512);
    let partial = zeros(&ctx, 4 * 8 * 64 * 512);
    let pos0 = n - m;
    run(&ctx, |p| {
        dsa_attend(
            &ctx,
            p,
            &tqk,
            &cache,
            Rows::Causal { pos0 },
            &scores,
            &denom,
            &lat,
            &partial,
            m,
            s,
            1.0 / 16.0,
        )
        .unwrap()
    });
    let glat = lat.to_f32().unwrap();
    assert!(
        glat[..m * 64 * 512].iter().all(|v| v.is_finite()),
        "non-finite latent mix"
    );
    for i in 0..m {
        let rows: Vec<usize> = (0..=pos0 + i).collect();
        for h in 0..64 {
            let ql = &qk[(i * 64 + h) * 512..(i * 64 + h + 1) * 512];
            let sc: Vec<f32> = rows
                .iter()
                .map(|&t| r::dot(ql, &gl[t * 512..(t + 1) * 512]) / 16.0)
                .collect();
            let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = sc.iter().map(|&x| (x - mx).exp()).collect();
            let d: f32 = e.iter().sum();
            let exp: Vec<f32> = (0..512)
                .map(|j| {
                    rows.iter()
                        .zip(&e)
                        .map(|(&t, &wt)| wt * gl[t * 512 + j])
                        .sum::<f32>()
                        / d
                })
                .collect();
            assert_close(
                &glat[(i * 64 + h) * 512..(i * 64 + h + 1) * 512],
                &exp,
                1e-5,
                1e-4,
            );
        }
    }
}

/// The attention over selected rows (the sparse and MTP paths): per-row
/// lists with unused slots, against the CPU.
#[test]
fn dsa_attend_selected_matches_cpu() {
    // Decode rows (SIMT kernels, the slots in chunks) and a prompt run
    // (matrix units); a short selection and a full one (its last chunk full).
    for (m, count) in [(2, 344), (6, 344), (2, 2056), (6, 2056)] {
        dsa_selected_case(m, count);
    }
}

fn dsa_selected_case(m: usize, count: usize) {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(70);
    let (cached, s) = (700usize, 2056usize);
    let lat_rows: Vec<f32> =
        f32s(&mut rng, cached * 512, -1.5, 1.5).into_iter().map(r::round_f16).collect();
    let bytes: Vec<u8> = lat_rows
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let cache = Tensor::from_bytes(&ctx, &bytes, &[cached * 512], DType::F16).unwrap();
    let mut sel = vec![u32::MAX; m * s];
    let mut lists = Vec::new();
    for i in 0..m {
        let mut l = Vec::new();
        for slot in 0..count {
            // Every seventh slot unused; rows spread over the cache.
            if slot % 7 != 3 {
                let row = (slot * 131 + i * 17) % cached;
                sel[i * s + slot] = row as u32;
                l.push(row);
            }
        }
        lists.push(l);
    }
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[m * s], DType::U32)
            .unwrap();
    let qk = f32s(&mut rng, m * 64 * 512, -0.2, 0.2);
    let tqk = tensor(&ctx, &qk);
    let scores = zeros(&ctx, m * 64 * s);
    let denom = zeros(&ctx, m * 64);
    let lat = zeros(&ctx, m * 64 * 512);
    // Bound here: the pass reads it until run's commit_wait returns.
    let partial = zeros(&ctx, 4 * 8 * 64 * 512);
    run(&ctx, |p| {
        dsa_attend(
            &ctx,
            p,
            &tqk,
            &cache,
            Rows::Selected { sel: &tsel, count },
            &scores,
            &denom,
            &lat,
            &partial,
            m,
            s,
            1.0 / 16.0,
        )
        .unwrap()
    });
    let glat = lat.to_f32().unwrap();
    for (i, rows) in lists.iter().enumerate() {
        for h in 0..64 {
            let ql = &qk[(i * 64 + h) * 512..(i * 64 + h + 1) * 512];
            let sc: Vec<f32> = rows
                .iter()
                .map(|&t| r::dot(ql, &lat_rows[t * 512..(t + 1) * 512]) / 16.0)
                .collect();
            let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = sc.iter().map(|&x| (x - mx).exp()).collect();
            let d: f32 = e.iter().sum();
            let exp: Vec<f32> = (0..512)
                .map(|j| {
                    rows.iter()
                        .zip(&e)
                        .map(|(&t, &wt)| wt * lat_rows[t * 512 + j])
                        .sum::<f32>()
                        / d
                })
                .collect();
            assert_close(
                &glat[(i * 64 + h) * 512..(i * 64 + h + 1) * 512],
                &exp,
                1e-5,
                1e-4,
            );
        }
    }
}

/// DSA decode attention (scores, softmax, weighted latent sum) at the
/// model's shapes: a dense run at 264 rows and the sparse 2 056-slot
/// selection, `LILY_GLM53_ROWS` query rows (default 1), best of five serial
/// passes of 11 layers' calls over 11 caches.
#[test]
#[ignore = "timing only"]
fn dsa_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(71);
    let m: usize =
        std::env::var("LILY_GLM53_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let cached = 4112 + m;
    let caches: Vec<Tensor> = (0..11)
        .map(|_| {
            let bytes: Vec<u8> = f32s(&mut rng, cached * 512, -1.5, 1.5)
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect();
            Tensor::from_bytes(&ctx, &bytes, &[cached * 512], DType::F16).unwrap()
        })
        .collect();
    let s_max = 2056;
    let sel: Vec<u32> =
        (0..m * s_max).map(|k| ((k % s_max) * 2 + k / s_max) as u32).collect();
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[m * s_max], DType::U32)
            .unwrap();
    let qk = tensor(&ctx, &f32s(&mut rng, m * 64 * 512, -0.2, 0.2));
    let scores = zeros(&ctx, m * 64 * s_max);
    let denom = zeros(&ctx, m * 64);
    let lat = zeros(&ctx, m * 64 * 512);
    let partial = zeros(&ctx, 4 * 8 * 64 * 512);
    // Prompt runs: two layers (a 1 024-row call takes milliseconds), the
    // dense case causal over the run itself.
    let caches = if m > 4 { &caches[..2] } else { &caches[..] };
    for (label, sparse) in [("dense 264 rows", false), ("selected 2056", true)] {
        let s = if sparse { s_max } else { 264.max(m).next_multiple_of(8) };
        let rows = if sparse {
            Rows::Selected { sel: &tsel, count: s_max }
        } else {
            Rows::Causal { pos0: 264usize.saturating_sub(m) }
        };
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for c in caches {
                dsa_attend(
                    &ctx, &pass, &qk, c, rows, &scores, &denom, &lat, &partial, m, s,
                    0.0625,
                )
                .unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!(
            "dsa {label:16} m={m} {:8.1} us a layer",
            best / caches.len() as f64 * 1e6
        );
        // With LILY_KERNEL_PROFILE: each kernel alone, per layer.
        let mut by: Vec<(&str, f64)> = Vec::new();
        for p in crate::metal::profile::take() {
            for k in &p.kernels {
                match by.iter_mut().find(|(n, _)| *n == k.name) {
                    Some(e) => e.1 += k.gpu_secs,
                    None => by.push((k.name, k.gpu_secs)),
                }
            }
        }
        for (name, secs) in by {
            println!(
                "dsa   {name:28} {:8.1} us a layer",
                secs / (5 * caches.len()) as f64 * 1e6
            );
        }
    }
}

/// The indexer's pool scores and top-k at decode (one row): 1 024 and
/// 8 192 complete pools, the top 512 kept; best of five passes of 11 calls.
#[test]
#[ignore = "timing only"]
fn indexer_topk_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(72);
    for pools in [1024usize, 8192] {
        let cap = pools + 1;
        let scores = tensor(&ctx, &f32s(&mut rng, cap, -1.0, 1.0));
        let s = 2056;
        let sel = Tensor::zeros(&ctx, &[s], DType::U32).unwrap();
        let count = Tensor::zeros(&ctx, &[1], DType::U32).unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for _ in 0..11 {
                indexer_topk(
                    &ctx,
                    &pass,
                    &scores,
                    &sel,
                    &count,
                    1,
                    cap,
                    s,
                    4 * pools,
                    512,
                )
                .unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("indexer_topk {pools:5} pools {:8.1} us", best / 11.0 * 1e6);
        let iq = tensor(&ctx, &f32s(&mut rng, 32 * 128, -1.0, 1.0));
        let w = tensor(&ctx, &f32s(&mut rng, 32, -1.0, 1.0));
        let bytes: Vec<u8> = f32s(&mut rng, pools * 128, -1.0, 1.0)
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let pooled =
            Tensor::from_bytes(&ctx, &bytes, &[pools * 128], DType::F16).unwrap();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for _ in 0..11 {
                indexer_scores(
                    &ctx,
                    &pass,
                    &iq,
                    &w,
                    &pooled,
                    &scores,
                    1,
                    cap,
                    4 * pools,
                    0.01,
                )
                .unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("indexer_scores {pools:5} pools {:8.1} us", best / 11.0 * 1e6);
    }
}

#[test]
fn indexer_pool_update_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(62);
    let cfg = crate::glm53::Config::glm53_flash();
    let norm_w = f32s(&mut rng, 128, 0.5, 1.5);
    let norm_b = f32s(&mut rng, 128, -0.2, 0.2);
    let ape_f: Vec<f32> = f32s(&mut rng, 4 * 128, -1.0, 1.0)
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect();
    let ape_b: Vec<u8> = ape_f
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let (tw, tb) = (tensor(&ctx, &norm_w), tensor(&ctx, &norm_b));
    let tape = Tensor::from_bytes(&ctx, &ape_b, &[ape_b.len()], DType::I8).unwrap();
    let pp = PoolParams { norm_w: &tw, norm_b: &tb, ape: &tape };
    let pooled = Tensor::zeros(&ctx, &[16 * 128], DType::F16).unwrap();
    let tail_k = zeros(&ctx, 4 * 128);
    let tail_g = zeros(&ctx, 4 * 128);
    let mut st = r::DsaState::default();
    let mut pos = 0;
    // Feeds that start and end mid-pool, one token, and several pools.
    for n in [3, 1, 6, 1, 1, 9, 2] {
        let k = f32s(&mut rng, n * 128, -2.0, 2.0);
        let g = f32s(&mut rng, n * 128, -2.0, 2.0);
        let (tk, tg) = (tensor(&ctx, &k), tensor(&ctx, &g));
        run(&ctx, |p| {
            indexer_pool_update(
                &ctx, p, &pp, &tk, &tg, &pooled, &tail_k, &tail_g, pos, n,
            )
            .unwrap()
        });
        for t in 0..n {
            r::indexer_pool_push(
                &cfg,
                &mut st,
                &k[t * 128..(t + 1) * 128],
                &g[t * 128..(t + 1) * 128],
                &norm_w,
                &norm_b,
                &ape_f,
            );
        }
        pos += n;
    }
    let pools = pos / 4;
    let got: Vec<f32> = pooled.contents()[..pools * 256]
        .chunks_exact(2)
        .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect();
    assert_close(&got, &st.pooled, 2e-3, 2e-3);
}

#[test]
fn indexer_scores_and_topk_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(63);
    let cfg = crate::glm53::Config::glm53_flash();
    // Rows past the dense limit with 700 to 702 complete pools; scores
    // quantized so ties occur.
    let (m, pos0) = (3, 2803);
    let pools_cap = (pos0 + m) / 4;
    let pooled_f: Vec<f32> = f32s(&mut rng, pools_cap * 128, -1.0, 1.0)
        .iter()
        .map(|&v| r::round_f16((v * 8.0).round() / 8.0))
        .collect();
    let pooled_b: Vec<u8> = pooled_f
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();
    let pooled =
        Tensor::from_bytes(&ctx, &pooled_b, &[pooled_f.len()], DType::F16).unwrap();
    let iq: Vec<f32> = f32s(&mut rng, m * 32 * 128, -1.0, 1.0)
        .iter()
        .map(|&v| (v * 4.0).round() / 4.0)
        .collect();
    let w = f32s(&mut rng, m * 32, 0.0, 1.0)
        .iter()
        .map(|&v| (v * 4.0).round() / 4.0)
        .collect::<Vec<_>>();
    let (tiq, tw) = (tensor(&ctx, &iq), tensor(&ctx, &w));
    let scores = zeros(&ctx, m * pools_cap);
    let s = cfg.indexer_selected_limit();
    let sel = Tensor::zeros(&ctx, &[m * s], DType::U32).unwrap();
    let count = Tensor::zeros(&ctx, &[m], DType::U32).unwrap();
    let scale = 1.0 / ((128 * 32) as f32).sqrt();
    run(&ctx, |p| {
        indexer_scores(&ctx, p, &tiq, &tw, &pooled, &scores, m, pools_cap, pos0, scale)
            .unwrap();
        indexer_topk(&ctx, p, &scores, &sel, &count, m, pools_cap, s, pos0, 512)
            .unwrap();
    });
    let gsel = sel.to_u32().unwrap();
    let gcount = count.to_u32().unwrap();
    let gscores = scores.to_f32().unwrap();
    let st = r::DsaState { pooled: pooled_f.clone(), ..Default::default() };
    for i in 0..m {
        let pos = pos0 + i;
        let pools = (pos + 1) / 4;
        let iq_i = &iq[i * 4096..(i + 1) * 4096];
        let w_i = &w[i * 32..(i + 1) * 32];
        let escores: Vec<f32> = (0..pools)
            .map(|p| {
                (0..32)
                    .map(|h| {
                        (r::dot(
                            &iq_i[h * 128..(h + 1) * 128],
                            &pooled_f[p * 128..(p + 1) * 128],
                        ) * scale)
                            .max(0.0)
                            * w_i[h]
                    })
                    .sum()
            })
            .collect();
        assert_close(
            &gscores[i * pools_cap..i * pools_cap + pools],
            &escores,
            1e-5,
            1e-5,
        );
        // The selection from the GPU's own scores (the order of equal sums
        // can differ in the last bit), compared as sets.
        let mut order: Vec<usize> = (0..pools).collect();
        let gs = &gscores[i * pools_cap..];
        order.sort_by(|&a, &b| gs[b].total_cmp(&gs[a]).then(a.cmp(&b)));
        let mut expect: Vec<u32> = order[..512]
            .iter()
            .flat_map(|&p| (4 * p as u32)..(4 * p as u32 + 4))
            .collect();
        expect.sort();
        let tail = (pos + 1) % 4;
        expect.extend(((pos + 1 - tail) as u32)..((pos + 1) as u32));
        assert_eq!(gcount[i] as usize, expect.len());
        assert_eq!(&gsel[i * s..i * s + expect.len()], &expect[..], "row {i}");
        assert!(gsel[i * s + expect.len()..(i + 1) * s].iter().all(|&v| v == u32::MAX));
        // And the CPU reference picks the same set from its own scores.
        let mut cpu = r::indexer_select(&cfg, &st, pos, iq_i, w_i);
        let ctail = cpu.split_off(2048);
        cpu.sort();
        cpu.extend(ctail);
        let same = cpu.iter().zip(&expect).filter(|(a, b)| **a as u32 == **b).count();
        assert!(
            same + 8 >= expect.len(),
            "CPU selection differs in {} rows",
            expect.len() - same
        );
    }
}

#[test]
fn hc_post_mix_and_pre2_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(64);
    let cfg = crate::glm53::Config::glm53_flash();
    let (m, hidden) = (2, cfg.hidden);
    let h = f32s(&mut rng, m * 4 * hidden, -3.0, 3.0);
    let block = f32s(&mut rng, m * hidden, -1.0, 1.0);
    // A previous split: post weights and a comb near a doubly stochastic one.
    let mut prev = Vec::new();
    for _ in 0..m {
        prev.extend(f32s(&mut rng, 4, 0.0, 1.0));
        prev.extend(f32s(&mut rng, 4, 0.0, 2.0));
        prev.extend(f32s(&mut rng, 16, 0.0, 0.5));
    }
    let fn_raw = random_blocks(&mut rng, GgmlType::Bf16, 24 * 4 * hidden);
    let fn_w =
        QWeight::from_bytes(&ctx, &fn_raw, GgmlType::Bf16, 4 * hidden, 24).unwrap();
    let fn_cpu = mat(GgmlType::Bf16, &fn_raw, 24, 4 * hidden);
    let scale = [0.8f32, 1.2, 1.7];
    let base = f32s(&mut rng, 24, -0.5, 0.5);
    let norm = f32s(&mut rng, hidden, 0.5, 1.5);
    let th = tensor(&ctx, &h);
    let (tblock, tprev) = (tensor(&ctx, &block), tensor(&ctx, &prev));
    let partial = zeros(&ctx, m * hc_mix_groups(hidden) * 25);
    let (tsc, tba, tno) =
        (tensor(&ctx, &scale), tensor(&ctx, &base), tensor(&ctx, &norm));
    let split = zeros(&ctx, m * 24);
    let xn = zeros(&ctx, m * hidden);
    run(&ctx, |p| {
        hc_post_mix(&ctx, p, &th, Some((&tblock, &tprev)), &fn_w, &partial, m, hidden)
            .unwrap();
        hc_pre2(
            &ctx,
            p,
            &partial,
            &tsc,
            &tba,
            &th,
            &tno,
            &split,
            &xn,
            m,
            hidden,
            20,
            1e-6,
            cfg.rms_eps,
        )
        .unwrap();
    });
    let gh = th.to_f32().unwrap();
    let gxn = xn.to_f32().unwrap();
    let gsplit = split.to_f32().unwrap();
    for i in 0..m {
        let hi = &h[i * 4 * hidden..(i + 1) * 4 * hidden];
        let pi = &prev[i * 24..(i + 1) * 24];
        let eh = r::hc_post(
            hi,
            &block[i * hidden..(i + 1) * hidden],
            &pi[4..8],
            &pi[8..24],
            4,
        );
        assert_close(&gh[i * 4 * hidden..(i + 1) * 4 * hidden], &eh, 1e-5, 1e-5);
        let (exn, epost, ecomb) = r::hc_pre(&cfg, &eh, &fn_cpu, &scale, &base, &norm);
        assert_close(&gsplit[i * 24 + 4..i * 24 + 8], &epost, 1e-5, 1e-4);
        assert_close(&gsplit[i * 24 + 8..i * 24 + 24], &ecomb, 1e-5, 1e-4);
        assert_close(&gxn[i * hidden..(i + 1) * hidden], &exn, 1e-4, 1e-3);
    }
}

/// The HC pair of a sublayer boundary (`hc_post_mix` then `hc_pre2`, each
/// its own level as in the step) at the model's shapes: best of five
/// serial passes of `LILY_GLM53_REPEAT` (default 64) pairs over 128 copies
/// of the control weights (100 MB: not cache resident, as in a step).
/// `LILY_GLM53_ROWS` rows (default 1).
#[test]
#[ignore = "timing only"]
fn hc_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(69);
    let env = |k: &str, d: usize| -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let (repeat, m) = (env("LILY_GLM53_REPEAT", 64), env("LILY_GLM53_ROWS", 1));
    let cfg = crate::glm53::Config::glm53_flash();
    let hidden = cfg.hidden;
    let fn_raw = random_blocks(&mut rng, GgmlType::Bf16, 24 * 4 * hidden);
    let fns: Vec<QWeight> = (0..128)
        .map(|_| {
            QWeight::from_bytes(&ctx, &fn_raw, GgmlType::Bf16, 4 * hidden, 24).unwrap()
        })
        .collect();
    let h = tensor(&ctx, &f32s(&mut rng, m * 4 * hidden, -0.01, 0.01));
    let block = tensor(&ctx, &f32s(&mut rng, m * hidden, -1e-3, 1e-3));
    let partial = zeros(&ctx, m * hc_mix_groups(hidden) * 25);
    let scale = tensor(&ctx, &[0.8f32, 1.2, 1.7]);
    let base = tensor(&ctx, &f32s(&mut rng, 24, -0.5, 0.5));
    let norm = tensor(&ctx, &f32s(&mut rng, hidden, 0.5, 1.5));
    let split = zeros(&ctx, m * 24);
    let xn = zeros(&ctx, m * hidden);
    for (label, part) in [("post_mix + pre2", 0), ("post_mix", 1), ("pre2", 2)] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for r in 0..repeat {
                if part != 2 {
                    let post = (r > 0).then_some((&block, &split));
                    hc_post_mix(
                        &ctx,
                        &pass,
                        &h,
                        post,
                        &fns[r % 128],
                        &partial,
                        m,
                        hidden,
                    )
                    .unwrap();
                }
                if part != 1 {
                    hc_pre2(
                        &ctx,
                        &pass,
                        &partial,
                        &scale,
                        &base,
                        &h,
                        &norm,
                        &split,
                        &xn,
                        m,
                        hidden,
                        20,
                        1e-6,
                        cfg.rms_eps,
                    )
                    .unwrap();
                }
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("hc {label:20} m={m} {:8.1} us", best / repeat as f64 * 1e6);
    }
}

#[test]
fn gemv_sets_match_single_gemvs() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(65);
    let cols = 4096;
    for ty in [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Bf16] {
        let rows = [72usize, 8, 130, 64];
        let raws: Vec<Vec<u8>> =
            rows.iter().map(|&r| random_blocks(&mut rng, ty, r * cols)).collect();
        let ws: Vec<QWeight> = raws
            .iter()
            .zip(rows)
            .map(|(raw, r)| QWeight::from_bytes(&ctx, raw, ty, cols, r).unwrap())
            .collect();
        for m in [1, 3] {
            let x = f32s(&mut rng, m * cols, -1.0, 1.0);
            let tx = tensor(&ctx, &x);
            let ys: Vec<Tensor> = rows.iter().map(|&r| zeros(&ctx, m * r)).collect();
            let singles: Vec<Tensor> =
                rows.iter().map(|&r| zeros(&ctx, m * r)).collect();
            let sets: Vec<(&QWeight, &Tensor)> = ws.iter().zip(&ys).collect();
            run(&ctx, |p| {
                gemv_sets(&ctx, p, &sets, &tx, m).unwrap();
                for (w, y) in ws.iter().zip(&singles) {
                    gemv(&ctx, p, w, &tx, y, m).unwrap();
                }
            });
            for (a, b) in ys.iter().zip(&singles) {
                assert_eq!(a.to_f32().unwrap(), b.to_f32().unwrap(), "{ty:?} m={m}");
            }
        }
    }
}

/// Bandwidth of the GEMVs at the model's shapes: `LILY_GLM53_REPEAT`
/// (default 16) dispatches over four weight copies (so the cache does not
/// hold them) in one pass, best of five passes.
#[test]
#[ignore = "timing only"]
fn gemv_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(66);
    let repeat: usize = std::env::var("LILY_GLM53_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let shapes = [
        ("kda_v Q8_0 8192x4096", GgmlType::Q8_0, 4096, 8192, 1),
        ("kda_out Q8_0 4096x8192", GgmlType::Q8_0, 8192, 4096, 1),
        ("kda_q Q4_K 8192x4096", GgmlType::Q4K, 4096, 8192, 1),
        ("shexp Q8_0 2048x4096", GgmlType::Q8_0, 4096, 2048, 1),
        ("head Q8_0 154880x4096", GgmlType::Q8_0, 4096, 154880, 1),
        ("kda_v Q8_0 m=4", GgmlType::Q8_0, 4096, 8192, 4),
        ("router F32 288x4096", GgmlType::F32, 4096, 288, 1),
        ("kv_a Q8_0 512x4096", GgmlType::Q8_0, 4096, 512, 1),
        ("q_a Q8_0 1536x4096", GgmlType::Q8_0, 4096, 1536, 1),
        ("f_b Q8_0 8192x128", GgmlType::Q8_0, 128, 8192, 1),
        ("attn_out Q8_0 4096x16384", GgmlType::Q8_0, 16384, 4096, 1),
    ];
    for (label, ty, cols, rows, m) in shapes {
        let copies = if rows > 100_000 { 1 } else { 4 };
        let ws: Vec<QWeight> = (0..copies)
            .map(|_| {
                let raw = random_blocks(&mut rng, ty, rows * cols);
                QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
            })
            .collect();
        let x = tensor(&ctx, &f32s(&mut rng, m * cols, -1.0, 1.0));
        let y = zeros(&ctx, m * rows);
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for r in 0..repeat {
                gemv(&ctx, &pass, &ws[r % copies], &x, &y, m).unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let bytes = ws[0].row_bytes * rows;
        let per = best / repeat as f64;
        println!(
            "{label:28} {:8.1} us  {:6.1} GB/s",
            per * 1e6,
            bytes as f64 / per / 1e9
        );
    }
}

/// The routed experts' kernels at the model's shapes (8 of 288 experts,
/// gate/up IQ2_XXS 2048 x 4096, down Q2_K 4096 x 2048), one row, best of
/// five passes of `LILY_GLM53_REPEAT` layers each over four weight copies.
#[test]
#[ignore = "timing only"]
fn moe_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(67);
    let repeat: usize = std::env::var("LILY_GLM53_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let (experts, hidden, ff, used) = (16, 4096, 2048, 8);
    let mk = |rng: &mut StdRng, ty, cols, rows| {
        let raw = random_blocks(rng, ty, rows * cols);
        QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
    };
    let layers: Vec<(QWeight, QWeight, QWeight)> = (0..4)
        .map(|_| {
            (
                mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff),
                mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff),
                mk(&mut rng, GgmlType::Q2K, ff, experts * hidden),
            )
        })
        .collect();
    let x = tensor(&ctx, &f32s(&mut rng, hidden, -1.0, 1.0));
    let sel: Vec<u32> = (0..used as u32).map(|i| i * 2).collect();
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[used], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &vec![0.3f32; used]);
    let mid = zeros(&ctx, used * ff);
    let out = zeros(&ctx, hidden);
    for (label, part) in [("gate_up", 0), ("down", 1)] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let pass = ctx.begin().unwrap();
            for r in 0..repeat {
                let (g, u, d) = &layers[r % 4];
                if part == 0 {
                    moe_gate_up(
                        &ctx, &pass, g, u, ff, &x, &tsel, &tw, &mid, 1, used, 10.0,
                    )
                    .unwrap();
                } else {
                    moe_down(&ctx, &pass, d, hidden, &mid, &tsel, None, &out, 1, used)
                        .unwrap();
                }
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let bytes = if part == 0 { 2 * used * ff * 1056 } else { used * hidden * 672 };
        let per = best / repeat as f64;
        println!(
            "moe {label:24} {:8.1} us  {:6.1} GB/s",
            per * 1e6,
            bytes as f64 / per / 1e9
        );
    }
}

/// One-token KDA at the model's shape (64 heads), best of five serial
/// passes of 34 layers' calls over 34 states (146 MB: as in a step); then a
/// 1 024-row prompt run, best of three passes of three layers. (The prompt
/// run's q, k, v and gate are prepared in place: later passes see their
/// silu'd values, the same work.)
#[test]
#[ignore = "timing only"]
fn kda_decode_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(73);
    let heads = 64;
    let p = heads * 128;
    let mk = |rng: &mut StdRng, n: usize, lo: f32, hi: f32| {
        tensor(&ctx, &f32s(rng, n, lo, hi))
    };
    let (q, k, v) = (
        mk(&mut rng, p, -1.0, 1.0),
        mk(&mut rng, p, -1.0, 1.0),
        mk(&mut rng, p, -1.0, 1.0),
    );
    let (g, beta, og) = (
        mk(&mut rng, p, -1.0, 1.0),
        mk(&mut rng, heads, -1.0, 1.0),
        mk(&mut rng, p, -1.0, 1.0),
    );
    let convs: Vec<Tensor> = (0..3).map(|_| mk(&mut rng, 4 * p, -0.5, 0.5)).collect();
    let (a_log, dt, on) = (
        mk(&mut rng, heads, -1.0, 0.0),
        mk(&mut rng, p, -0.5, 0.5),
        mk(&mut rng, 128, 0.5, 1.5),
    );
    let layers: Vec<(Tensor, Tensor)> = (0..34)
        .map(|_| (zeros(&ctx, 9 * p), mk(&mut rng, heads * 128 * 128, -0.1, 0.1)))
        .collect();
    let out = zeros(&ctx, p);
    let edges = zeros(&ctx, kda_edges_len(1024, heads));
    let params = KdaParams {
        q_conv: &convs[0],
        k_conv: &convs[1],
        v_conv: &convs[2],
        a_log: &a_log,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound: -5.0,
        eps: 1e-5,
    };
    let x = KdaInputs {
        q: &q,
        k: &k,
        v: &v,
        gate: &g,
        beta: &beta,
        out_gate: &og,
        edges: Some(&edges),
    };
    let mut best = f64::MAX;
    for _ in 0..5 {
        let pass = ctx.begin().unwrap();
        for (conv, st) in &layers {
            kda(&ctx, &pass, &params, &x, conv, st, &out, 1, None).unwrap();
        }
        let t = std::time::Instant::now();
        pass.commit_wait().unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("kda_decode {:8.1} us a layer", best / 34.0 * 1e6);
    // A prompt run of 1 024 rows (prepare, recurrence, output), three layers.
    let n = 1024;
    let big = |rng: &mut StdRng, w: usize| mk(rng, n * w, -1.0, 1.0);
    let (q, k, v) = (big(&mut rng, p), big(&mut rng, p), big(&mut rng, p));
    let (g, beta, og) = (big(&mut rng, p), big(&mut rng, heads), big(&mut rng, p));
    let out = zeros(&ctx, n * p);
    let x = KdaInputs {
        q: &q,
        k: &k,
        v: &v,
        gate: &g,
        beta: &beta,
        out_gate: &og,
        edges: Some(&edges),
    };
    let mut best = f64::MAX;
    for _ in 0..3 {
        let pass = ctx.begin().unwrap();
        for (conv, st) in &layers[..3] {
            kda(&ctx, &pass, &params, &x, conv, st, &out, n, None).unwrap();
        }
        let t = std::time::Instant::now();
        pass.commit_wait().unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("kda_prefill 1024 rows {:8.1} us a layer", best / 3.0 * 1e6);
    // With LILY_KERNEL_PROFILE: each kernel alone, per layer of the run.
    let mut by: Vec<(&str, f64)> = Vec::new();
    for p in crate::metal::profile::take() {
        for k in &p.kernels {
            match by.iter_mut().find(|(n, _)| *n == k.name) {
                Some(e) => e.1 += k.gpu_secs,
                None => by.push((k.name, k.gpu_secs)),
            }
        }
    }
    for (name, secs) in by {
        if name != "glm53_kda_decode" {
            println!("kda_   {name:28} {:8.1} us a layer", secs / 9.0 * 1e6);
        }
    }
}

#[test]
fn kda_snapshot_is_the_state_after_its_row() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(68);
    let mut cfg = crate::glm53::Config::glm53_flash();
    let heads = 2;
    cfg.kda_heads = heads;
    let p = heads * 128;
    let c = kda_case(&mut rng, heads);
    let t = |v: &Vec<f32>| tensor(&ctx, v);
    let (qc, kc, vc, al, dt, on) = (
        t(&c.q_conv),
        t(&c.k_conv),
        t(&c.v_conv),
        t(&c.a_log),
        t(&c.dt_bias),
        t(&c.o_norm),
    );
    let params = KdaParams {
        q_conv: &qc,
        k_conv: &kc,
        v_conv: &vc,
        a_log: &al,
        dt_bias: &dt,
        o_norm: &on,
        heads,
        lower_bound: -5.0,
        eps: 1e-5,
    };
    let cp = r::KdaParams {
        q_conv: &c.q_conv,
        k_conv: &c.k_conv,
        v_conv: &c.v_conv,
        a_log: &c.a_log,
        dt_bias: &c.dt_bias,
        o_norm: &c.o_norm,
    };
    // Two tokens of history first, so the conv window is not all zero.
    let mut st = r::KdaState::new(&cfg);
    let conv_state = zeros(&ctx, 9 * p);
    let state = zeros(&ctx, heads * 128 * 128);
    for n in [2usize, 3] {
        let q = f32s(&mut rng, n * p, -2.0, 2.0);
        let k = f32s(&mut rng, n * p, -2.0, 2.0);
        let v = f32s(&mut rng, n * p, -2.0, 2.0);
        let g = f32s(&mut rng, n * p, -3.0, 3.0);
        let b = f32s(&mut rng, n * heads, -2.0, 2.0);
        let og = f32s(&mut rng, n * p, -2.0, 2.0);
        let (tq, tk, tv, tg, tb, tog) = (t(&q), t(&k), t(&v), t(&g), t(&b), t(&og));
        let out = zeros(&ctx, n * p);
        let edges = zeros(&ctx, kda_edges_len(n, heads));
        let inputs = KdaInputs {
            q: &tq,
            k: &tk,
            v: &tv,
            gate: &tg,
            beta: &tb,
            out_gate: &tog,
            edges: Some(&edges),
        };
        let snap_conv = zeros(&ctx, 9 * p);
        let snap_state = zeros(&ctx, heads * 128 * 128);
        let snap = (n == 3).then_some(KdaSnapshot {
            row: 0,
            conv: &snap_conv,
            state: &snap_state,
        });
        run(&ctx, |pass| {
            kda(&ctx, pass, &params, &inputs, &conv_state, &state, &out, n, snap)
                .unwrap()
        });
        for j in 0..n {
            let s = j * p..(j + 1) * p;
            r::kda_step(
                &cfg,
                &mut st,
                &cp,
                &q[s.clone()],
                &k[s.clone()],
                &v[s.clone()],
                &g[s.clone()],
                &b[j * heads..(j + 1) * heads],
                &og[s.clone()],
            );
            if n == 3 && j == 0 {
                assert_close(&snap_state.to_f32().unwrap(), &st.s, 1e-4, 1e-3);
                assert_close(&snap_conv.to_f32().unwrap(), &st.conv, 0.0, 0.0);
            }
        }
    }
    assert_close(&state.to_f32().unwrap(), &st.s, 1e-4, 1e-3);
}

#[test]
fn gemm_matches_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(69);
    for (ty, cols, rows, m) in [
        (GgmlType::Q8_0, 4096usize, 200usize, 70usize),
        (GgmlType::Q4K, 4096, 130, 5),
        (GgmlType::Bf16, 1536, 64, 129),
        (GgmlType::F32, 4096, 288, 33),
        (GgmlType::Q8_0, 128, 8192, 64),
    ] {
        let raw = random_blocks(&mut rng, ty, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let cpu = mat(ty, &raw, rows, cols);
        let x = f32s(&mut rng, m * cols, -1.0, 1.0);
        let tx = tensor(&ctx, &x);
        let y = zeros(&ctx, m * rows);
        run(&ctx, |p| gemm(&ctx, p, &w, &tx, &y, m).unwrap());
        let g = y.to_f32().unwrap();
        // bf16 operands: compare against the exact product with a tolerance
        // scaled by the row's magnitude.
        for i in 0..m {
            let e = cpu.mv(&x[i * cols..(i + 1) * cols]);
            let scale: f32 =
                (e.iter().map(|v| v * v).sum::<f32>() / rows as f32).sqrt();
            for (n, (a, b)) in g[i * rows..(i + 1) * rows].iter().zip(&e).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-2 * scale + 1e-6,
                    "{ty:?} row {i} col {n}: {a} vs {b}"
                );
            }
        }
    }
}

/// The dense GEMM walks K in 64-deep steps: widths of 64 are computed,
/// widths that are not a multiple of 64 are refused before anything is
/// encoded.
#[test]
fn gemm_k_widths() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(75);
    let (rows, m) = (40, 5);
    for cols in [32usize, 64, 96] {
        let raw = random_blocks(&mut rng, GgmlType::Q8_0, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, rows).unwrap();
        let x = f32s(&mut rng, m * cols, -1.0, 1.0);
        let tx = tensor(&ctx, &x);
        let y = zeros(&ctx, m * rows);
        let pass = ctx.begin().unwrap();
        let r = gemm(&ctx, &pass, &w, &tx, &y, m);
        pass.commit_wait().unwrap();
        if cols % 64 != 0 {
            assert!(r.is_err(), "K = {cols} accepted");
            continue;
        }
        r.unwrap();
        let cpu = mat(GgmlType::Q8_0, &raw, rows, cols);
        let g = y.to_f32().unwrap();
        for i in 0..m {
            let e = cpu.mv(&x[i * cols..(i + 1) * cols]);
            let scale: f32 =
                (e.iter().map(|v| v * v).sum::<f32>() / rows as f32).sqrt();
            for (a, b) in g[i * rows..(i + 1) * rows].iter().zip(&e) {
                assert!((a - b).abs() <= 2e-2 * scale + 1e-6, "K = {cols}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn gemm_batched_matches_grouped_gemv() {
    // attn_k_b's shape for 4 heads, 70 rows: per head a 512 x 256 block.
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(70);
    let (heads, per, cols, m) = (4, 512, 256, 70);
    let raw = random_blocks(&mut rng, GgmlType::Q8_0, heads * per * cols);
    let w = QWeight::from_bytes(&ctx, &raw, GgmlType::Q8_0, cols, heads * per).unwrap();
    let cpu = mat(GgmlType::Q8_0, &raw, heads * per, cols);
    let x = f32s(&mut rng, m * heads * cols, -1.0, 1.0);
    let tx = tensor(&ctx, &x);
    let y = zeros(&ctx, m * heads * per);
    let l = GemmLayout {
        ldx: heads * cols,
        ldy: heads * per,
        batches: heads,
        x_batch: cols,
        w_batch: per,
        y_batch: per,
    };
    run(&ctx, |p| gemm_with(&ctx, p, &w, &tx, &y, m, per, l).unwrap());
    let g = y.to_f32().unwrap();
    for i in 0..m {
        let e = cpu.mv_grouped(&x[i * heads * cols..(i + 1) * heads * cols], per);
        let scale: f32 = (e.iter().map(|v| v * v).sum::<f32>() / e.len() as f32).sqrt();
        for (a, b) in g[i * heads * per..(i + 1) * heads * per].iter().zip(&e) {
            assert!((a - b).abs() <= 2e-2 * scale, "{a} vs {b}");
        }
    }
}

#[test]
fn moe_grouped_matches_per_pair_kernels() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(71);
    let (experts, hidden, ff, used, m) = (12, 512, 256, 8, 70);
    let g_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let u_raw = random_blocks(&mut rng, GgmlType::Iq2Xxs, experts * ff * hidden);
    let d_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * hidden * ff);
    let gw = QWeight::from_bytes(&ctx, &g_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let uw = QWeight::from_bytes(&ctx, &u_raw, GgmlType::Iq2Xxs, hidden, experts * ff)
        .unwrap();
    let dw =
        QWeight::from_bytes(&ctx, &d_raw, GgmlType::Q2K, ff, experts * hidden).unwrap();
    let x = f32s(&mut rng, m * hidden, -2.0, 2.0);
    // Distinct experts per row, skewed so some experts get more than 64 pairs.
    let sel: Vec<u32> = (0..m)
        .flat_map(|i| (0..used).map(move |s| ((i * (s + 1) + s * s) % experts) as u32))
        .collect();
    let wts = f32s(&mut rng, m * used, 0.1, 0.6);
    let shared = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let tx = tensor(&ctx, &x);
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &wts);
    let grouped = tensor(&ctx, &shared);
    let reference = tensor(&ctx, &shared);
    let mid = zeros(&ctx, m * used * ff);
    let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
    run(&ctx, |p| {
        moe_grouped(
            &ctx, p, &gw, &uw, &dw, ff, hidden, &tx, &tsel, &tw, &grouped, &gs, m,
            used, experts, 0.05,
        )
        .unwrap();
        moe_gate_up(&ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, m, used, 0.05)
            .unwrap();
        moe_down(
            &ctx,
            p,
            &dw,
            hidden,
            &mid,
            &tsel,
            Some(&reference),
            &reference,
            m,
            used,
        )
        .unwrap();
    });
    let a = grouped.to_f32().unwrap();
    let b = reference.to_f32().unwrap();
    let scale = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 3e-2 * scale, "element {k}: {x} vs {y}");
    }
}

/// The prefill's grouped routed experts at the model's shapes (gate/up
/// IQ2_XXS 2048 x 4096, down Q2_K 4096 x 2048) over 64 experts, for runs
/// whose pairs per expert match a 256-token and a 4K-token prompt of the
/// model (288 experts, 8 used): best of three passes of one call.
#[test]
#[ignore = "timing only"]
fn moe_grouped_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(72);
    let (experts, hidden, ff, used) = (64, 4096, 2048, 8);
    let mk = |rng: &mut StdRng, ty, cols, rows| {
        let raw = random_blocks(rng, ty, rows * cols);
        QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
    };
    let gw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let uw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let dw = mk(&mut rng, GgmlType::Q2K, ff, experts * hidden);
    // Rows such that pairs per expert = 256 * 8 / 288 and 4096 * 8 / 288.
    for (label, m) in [("256-token density", 57), ("4K-token density", 910)] {
        let sel: Vec<u32> = (0..m)
            .flat_map(|i| {
                let start = (i * 37) % experts;
                (0..used).map(move |s| ((start + s * 7) % experts) as u32)
            })
            .collect();
        let x = tensor(&ctx, &f32s(&mut rng, m * hidden, -1.0, 1.0));
        let tsel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&sel),
            &[sel.len()],
            DType::U32,
        )
        .unwrap();
        let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.6));
        let out = zeros(&ctx, m * hidden);
        let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let pass = ctx.begin().unwrap();
            moe_grouped(
                &ctx, &pass, &gw, &uw, &dw, ff, hidden, &x, &tsel, &tw, &out, &gs, m,
                used, experts, 10.0,
            )
            .unwrap();
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("moe grouped {label:20} m {m:4}: {:8.1} us", best * 1e6);
    }
}

/// The prefill's dense GEMMs at the model's shapes for a run of 1 024 rows:
/// best of three passes of four calls.
#[test]
#[ignore = "timing only"]
fn gemm_timing() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(73);
    let m = 1024;
    for (label, ty, cols, rows) in [
        ("kda_v Q8_0 8192x4096", GgmlType::Q8_0, 4096usize, 8192usize),
        ("kda_out Q8_0 4096x8192", GgmlType::Q8_0, 8192, 4096),
        ("kda_q Q4_K 8192x4096", GgmlType::Q4K, 4096, 8192),
        ("f_b Q8_0 8192x128", GgmlType::Q8_0, 128, 8192),
        ("shexp Q8_0 2048x4096", GgmlType::Q8_0, 4096, 2048),
    ] {
        let raw = random_blocks(&mut rng, ty, rows * cols);
        let w = QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap();
        let x = tensor(&ctx, &f32s(&mut rng, m * cols, -1.0, 1.0));
        let y = zeros(&ctx, m * rows);
        let mut best = f64::MAX;
        for _ in 0..3 {
            let pass = ctx.begin().unwrap();
            for _ in 0..4 {
                gemm(&ctx, &pass, &w, &x, &y, m).unwrap();
            }
            let t = std::time::Instant::now();
            pass.commit_wait().unwrap();
            best = best.min(t.elapsed().as_secs_f64() / 4.0);
        }
        println!(
            "gemm {label:24} m {m}: {:8.1} us  {:5.1} TFLOP/s",
            best * 1e6,
            2.0 * (m * rows * cols) as f64 / best / 1e12
        );
    }
}

/// The fused gate/up/SwiGLU expert GEMM gives the separate GEMMs' and
/// SwiGLU's values bit for bit: experts with 16, 32, 48, 64 and 65 pairs
/// (one, two, three and four working subgroups, and a second tile), small
/// ones, an unused one; the clamp off, tight and loose.
#[test]
fn moe_grouped_fused_matches_separate() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(74);
    let (experts, hidden, ff, used, m) = (12, 1024, 512, 8, 40);
    let mk = |rng: &mut StdRng, ty, cols, rows| {
        let raw = random_blocks(rng, ty, rows * cols);
        QWeight::from_bytes(&ctx, &raw, ty, cols, rows).unwrap()
    };
    let gw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let uw = mk(&mut rng, GgmlType::Iq2Xxs, hidden, experts * ff);
    let dw = mk(&mut rng, GgmlType::Q2K, ff, experts * hidden);
    // 320 pairs: experts 0-4 take 16, 32, 48, 64 and 65, expert 5 none,
    // experts 6-11 the other 95, in a scrambled order.
    let counts = [16usize, 32, 48, 64, 65, 0, 16, 16, 16, 16, 16, 15];
    assert_eq!(counts.iter().sum::<usize>(), m * used);
    let mut sel: Vec<u32> = counts
        .iter()
        .enumerate()
        .flat_map(|(e, &c)| std::iter::repeat_n(e as u32, c))
        .collect();
    for i in (1..sel.len()).rev() {
        sel.swap(i, rng.gen_range(0..=i));
    }
    let x = tensor(&ctx, &f32s(&mut rng, m * hidden, -2.0, 2.0));
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &f32s(&mut rng, m * used, 0.1, 0.6));
    let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
    for limit in [0.0f32, 1e-3, 10.0] {
        let mut got = Vec::new();
        for fused in [true, false] {
            let out = zeros(&ctx, m * hidden);
            run(&ctx, |p| {
                moe_grouped_with(
                    &ctx, p, &gw, &uw, &dw, ff, hidden, &x, &tsel, &tw, &out, &gs, m,
                    used, experts, limit, fused,
                )
                .unwrap()
            });
            got.push((gs.mid.to_f32().unwrap(), out.to_f32().unwrap()));
        }
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&got[0].0), bits(&got[1].0), "mid, limit {limit}");
        assert_eq!(bits(&got[0].1), bits(&got[1].1), "out, limit {limit}");
    }
}

/// The grouped route of Q2_K gate/up experts (separate GEMMs and SwiGLU)
/// against the per-pair kernels.
#[test]
fn moe_grouped_q2_k_gate_up_matches_per_pair_kernels() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(76);
    let (experts, hidden, ff, used, m) = (12, 512, 256, 8, 70);
    let g_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * ff * hidden);
    let u_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * ff * hidden);
    let d_raw = random_blocks(&mut rng, GgmlType::Q2K, experts * hidden * ff);
    let gw =
        QWeight::from_bytes(&ctx, &g_raw, GgmlType::Q2K, hidden, experts * ff).unwrap();
    let uw =
        QWeight::from_bytes(&ctx, &u_raw, GgmlType::Q2K, hidden, experts * ff).unwrap();
    let dw =
        QWeight::from_bytes(&ctx, &d_raw, GgmlType::Q2K, ff, experts * hidden).unwrap();
    let x = f32s(&mut rng, m * hidden, -2.0, 2.0);
    let sel: Vec<u32> = (0..m)
        .flat_map(|i| (0..used).map(move |s| ((i * (s + 1) + s * s) % experts) as u32))
        .collect();
    let wts = f32s(&mut rng, m * used, 0.1, 0.6);
    let shared = f32s(&mut rng, m * hidden, -1.0, 1.0);
    let tx = tensor(&ctx, &x);
    let tsel =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&sel), &[sel.len()], DType::U32)
            .unwrap();
    let tw = tensor(&ctx, &wts);
    let grouped = tensor(&ctx, &shared);
    let reference = tensor(&ctx, &shared);
    let mid = zeros(&ctx, m * used * ff);
    let gs = MoeGroupScratch::new(&ctx, m * used, experts, ff, hidden).unwrap();
    run(&ctx, |p| {
        moe_grouped(
            &ctx, p, &gw, &uw, &dw, ff, hidden, &tx, &tsel, &tw, &grouped, &gs, m,
            used, experts, 0.05,
        )
        .unwrap();
        moe_gate_up(&ctx, p, &gw, &uw, ff, &tx, &tsel, &tw, &mid, m, used, 0.05)
            .unwrap();
        moe_down(
            &ctx,
            p,
            &dw,
            hidden,
            &mid,
            &tsel,
            Some(&reference),
            &reference,
            m,
            used,
        )
        .unwrap();
    });
    let a = grouped.to_f32().unwrap();
    let b = reference.to_f32().unwrap();
    let scale = (b.iter().map(|v| v * v).sum::<f32>() / b.len() as f32).sqrt();
    for (k, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!((x - y).abs() <= 3e-2 * scale, "element {k}: {x} vs {y}");
    }
}
