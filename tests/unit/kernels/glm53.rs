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
        let inputs =
            KdaInputs { q: &tq, k: &tk, v: &tv, gate: &tg, beta: &tb, out_gate: &tog };
        run(&ctx, |pass| {
            kda(&ctx, pass, &params, &inputs, &conv_state, &state, &out, n).unwrap()
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

#[test]
fn dsa_store_and_attend_match_cpu() {
    let ctx = ctx();
    let mut rng = StdRng::seed_from_u64(61);
    let cfg = crate::glm53::Config::glm53_flash();
    let (n, m) = (45, 3); // 42 cached rows, then 3 rows of queries
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
    let qk = f32s(&mut rng, m * 64 * 512, -0.2, 0.2);
    let tqk = tensor(&ctx, &qk);
    let s = 64;
    let scores = zeros(&ctx, m * 64 * s);
    let denom = zeros(&ctx, m * 64);
    let lat = zeros(&ctx, m * 64 * 512);
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
            m,
            s,
            1.0 / 16.0,
        )
        .unwrap()
    });
    let glat = lat.to_f32().unwrap();
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
