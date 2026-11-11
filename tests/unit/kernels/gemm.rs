use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;
use crate::tensor::DType;

fn random_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn check_gemm(seed: u64) {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(seed);
    // Tile-exact and partial-tile shapes around both kernels' tile sizes
    // (32x32xK16 and 64x64xK32): multiples, non-multiples (incl. odd K
    // exercising the scalar-load path), single-row M, the M/N/K tails one
    // past a tile boundary, and the model's real shapes.
    for (m, k, n) in [
        (5, 64, 7),
        (32, 64, 32),
        (128, 1024, 512),
        (129, 100, 33),
        (1, 17, 40),
        (33, 16, 31),
        (63, 24, 65),
        (65, 64, 63),
        (257, 512, 40),
        (64, 1024, 3584),
        (256, 3584, 1024),
        (1, 1024, 6144),
    ] {
        let a = random_vec(&mut rng, m * k);
        let b = random_vec(&mut rng, n * k);

        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[n, k]).expect("b");
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");

        let bias = random_vec(&mut rng, n);
        let tbias = Tensor::from_f32_as_bf16(&ctx, &bias, &[n]).expect("bias");
        let tcb = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c + bias");

        let pass = ctx.begin().expect("pass");
        gemm_bf16_nt(&ctx, &pass, &ta, &tb, &tc).expect("gemm");
        gemm_bf16_nt_bias(&ctx, &pass, &ta, &tb, &tbias, &tcb).expect("gemm + bias");
        pass.commit_wait().expect("commit");

        let expected = cpu_ref::gemm_nt(
            &cpu_ref::round_bf16(&a),
            &cpu_ref::round_bf16(&b),
            m,
            k,
            n,
        );
        cpu_ref::assert_close(&tc.to_f32().expect("read"), &expected, 2e-2, 2e-2);
        let rb = cpu_ref::round_bf16(&bias);
        let biased: Vec<f32> =
            expected.iter().enumerate().map(|(i, v)| v + rb[i % n]).collect();
        cpu_ref::assert_close(&tcb.to_f32().expect("read"), &biased, 2e-2, 2e-2);
    }
}

#[test]
fn gemm_bf16_nt_matches_cpu() {
    check_gemm(0);
}

/// Dispatches a dense GEMM kernel by name with the shipped kernel's
/// arguments: `tile` is the (rows, columns) of C a threadgroup of `threads`
/// computes.
#[allow(clippy::too_many_arguments)]
fn gemm_named(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    name: &'static str,
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    c: &Tensor,
    threads: usize,
    tile: (usize, usize),
) -> Result<()> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[0];
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V4_0)?;
    let mut buffers = vec![a.binding(), b.binding(), c.binding()];
    buffers.extend(bias.map(Tensor::binding));
    pass.dispatch_at(
        &pipeline,
        &buffers,
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(m)],
        Grid::Threadgroups {
            groups: (n.div_ceil(tile.1), m.div_ceil(tile.0), 1),
            threadgroup: (threads, 1, 1),
        },
    )
}

fn output_bits(t: &Tensor) -> Vec<u32> {
    t.to_f32().expect("read").iter().map(|v| v.to_bits()).collect()
}

/// The shipped kernels against the first port's (`tests/vulkan/gemm_test.comp`:
/// every operand a `coopMatLoad` from device memory, K in order). Where the
/// shipped kernel walks K in order too (K under 64, or not a multiple of 8)
/// the two match bit for bit. Past that its 64-deep groups pair K
/// differently among their multiplies (`matmul.glsl`, MAT_K64): the same
/// products, summed in fp32 in another grouping, so an output may differ by
/// the accumulation's rounding and then by one bfloat ulp, and no more.
/// Shapes on every edge of the 16-block, of the 32-row and 64-column tiles
/// and of the 64-deep group in M, N and K, odd K (rows at odd addresses),
/// and the model's.
#[test]
fn gemm_bf16_nt_matches_first_port() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(5);
    for (m, k, n) in [
        (1, 1, 1),
        (5, 64, 7),
        (16, 16, 16),
        (17, 15, 15),
        (31, 17, 33),
        (64, 64, 64),
        (65, 63, 63),
        (63, 65, 65),
        (33, 72, 129),
        (129, 100, 33),
        (1, 17, 40),
        (127, 129, 191),
        (97, 200, 130),
        (128, 1024, 512),
        (257, 513, 40),
        (3, 2560, 2560),
        (200, 2560, 640),
        (64, 1024, 3584),
        (256, 3584, 1024),
        (1, 1024, 6144),
    ] {
        let a = random_vec(&mut rng, m * k);
        let b = random_vec(&mut rng, n * k);
        let bias = random_vec(&mut rng, n);
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[n, k]).expect("b");
        let tbias = Tensor::from_f32_as_bf16(&ctx, &bias, &[n]).expect("bias");
        let out = || Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
        let (c, cb, rc, rcb) = (out(), out(), out(), out());
        let pass = ctx.begin().expect("pass");
        gemm_bf16_nt(&ctx, &pass, &ta, &tb, &c).expect("gemm");
        gemm_bf16_nt_bias(&ctx, &pass, &ta, &tb, &tbias, &cb).expect("gemm + bias");
        gemm_named(
            &ctx,
            &pass,
            "gemm_bf16_nt_nax_ref",
            &ta,
            &tb,
            None,
            &rc,
            128,
            (64, 64),
        )
        .expect("reference");
        gemm_named(
            &ctx,
            &pass,
            "gemm_bf16_nt_bias_nax_ref",
            &ta,
            &tb,
            Some(&tbias),
            &rcb,
            128,
            (64, 64),
        )
        .expect("reference + bias");
        pass.commit_wait().expect("commit");
        let in_order = k < 64 || k % 8 != 0;
        for (got, want, what) in [(&c, &rc, "gemm"), (&cb, &rcb, "gemm + bias")] {
            let (got, want) =
                (got.to_f32().expect("read"), want.to_f32().expect("read"));
            let mut differ = 0usize;
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                if g.to_bits() == w.to_bits() {
                    continue;
                }
                differ += 1;
                // One bfloat ulp is at most 2^-7 of the value; the fp32 sums
                // of K products under 1 differ by far less than 1e-6 K.
                let bound = w.abs() / 128.0 + 1e-6 * k as f32;
                assert!(
                    !in_order && (g - w).abs() <= bound,
                    "{what} M={m} K={k} N={n}: output {i} is {g}, the first port's {w}"
                );
            }
            assert!(
                differ * 100 <= got.len(),
                "{what} M={m} K={k} N={n}: {differ} of {} outputs differ from the first port",
                got.len()
            );
            assert!(want.iter().any(|&v| v != 0.0), "M={m} K={k} N={n}: empty output");
        }
    }
}

/// A row of C depends on its row of A and on B alone, a column on its row of
/// B: not on M or N, nor on where the row or the column falls in a tile
/// (inside one, on a clamped edge, in another subgroup's rows). Rows computed
/// alone and columns computed from a prefix of B match the full product bit
/// for bit, with and without a K tail.
#[test]
fn gemm_bf16_nt_results_do_not_depend_on_position() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(6);
    for (m, k, n) in [(70usize, 192usize, 150usize), (45, 200, 64), (33, 2560, 129)] {
        let a = random_vec(&mut rng, m * k);
        let b = random_vec(&mut rng, n * k);
        let run = |rows: std::ops::Range<usize>, cols: usize| -> Vec<u32> {
            let ta = Tensor::from_f32_as_bf16(
                &ctx,
                &a[rows.start * k..rows.end * k],
                &[rows.len(), k],
            )
            .expect("a");
            let tb =
                Tensor::from_f32_as_bf16(&ctx, &b[..cols * k], &[cols, k]).expect("b");
            let tc = Tensor::zeros(&ctx, &[rows.len(), cols], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_bf16_nt(&ctx, &pass, &ta, &tb, &tc).expect("gemm");
            pass.commit_wait().expect("commit");
            output_bits(&tc)
        };
        let full = run(0..m, n);
        for rows in [0..1, 17..18, 31..33, m - 1..m, 5..m - 3] {
            for cols in [1, 63, 64, 65, n - 1, n].into_iter().filter(|&cols| cols <= n)
            {
                let part = run(rows.clone(), cols);
                for (i, row) in rows.clone().enumerate() {
                    assert_eq!(
                        part[i * cols..(i + 1) * cols],
                        full[row * n..row * n + cols],
                        "M={m} K={k} N={n}: row {row} of rows {rows:?}, {cols} columns"
                    );
                }
            }
        }
    }
}

/// The dense GEMM at a prefill shape (`LILY_GEMM_SHAPE=M,K,N`, default
/// 4096,2560,2560: a 4096-token chunk through a square projection),
/// interleaved over the kernels of `LILY_GEMM_ARMS`: comma-separated
/// `name[:threads[:tile rows[:tile columns]]]`, default the first port and
/// the shipped kernel. A name with no built-in shader loads from
/// `LILY_VK_SHADER_DIR`, so a variant under work needs no rebuild. Prints
/// best and median ms and TFLOP/s per arm and how many outputs differ from
/// the first arm's. `cargo test --release --lib -- --ignored --nocapture
/// gemm_nax_timing`.
#[test]
#[ignore = "timing only"]
fn gemm_nax_timing() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(9);
    let shape: Vec<usize> = std::env::var("LILY_GEMM_SHAPE")
        .unwrap_or_else(|_| "4096,2560,2560".into())
        .split(',')
        .map(|v| v.parse().expect("LILY_GEMM_SHAPE=M,K,N"))
        .collect();
    let (m, k, n) = (shape[0], shape[1], shape[2]);
    let arms: Vec<(&'static str, usize, (usize, usize))> =
        std::env::var("LILY_GEMM_ARMS")
            .unwrap_or_else(|_| "gemm_bf16_nt_nax_ref,gemm_bf16_nt_nax".into())
            .split(',')
            .map(|arm| {
                let mut parts = arm.split(':');
                let name: &'static str = parts.next().expect("name").to_string().leak();
                let mut number = |default: usize| {
                    parts.next().map_or(default, |v| v.parse().expect("number"))
                };
                (name, number(128), (number(64), number(64)))
            })
            .collect();
    let ta = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, m * k), &[m, k])
        .expect("a");
    let tb = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, n * k), &[n, k])
        .expect("b");
    let outs: Vec<Tensor> = arms
        .iter()
        .map(|_| Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c"))
        .collect();
    let reps: usize =
        std::env::var("LILY_GEMM_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);

    let mut times = vec![Vec::new(); arms.len()];
    let warm = crate::kernels::qsa::tests::chain_warmup();
    let mut round = 0usize;
    loop {
        let timed = warm.elapsed() >= crate::kernels::qsa::tests::chain_warmup_ms();
        for j in 0..arms.len() {
            let which = (round + j) % arms.len();
            let (name, threads, tile) = arms[which];
            let pass = ctx.begin().expect("pass");
            for _ in 0..reps {
                gemm_named(
                    &ctx,
                    &pass,
                    name,
                    &ta,
                    &tb,
                    None,
                    &outs[which],
                    threads,
                    tile,
                )
                .expect("dispatch");
            }
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            if timed {
                times[which].push((t.gpu_end_secs - t.gpu_start_secs) / reps as f64);
            }
        }
        if timed {
            round += 1;
            if round == 9 {
                break;
            }
        }
    }
    let flop = 2.0 * (m * k * n) as f64;
    let want = output_bits(&outs[0]);
    for ((arm, t), out) in arms.iter().zip(&mut times).zip(&outs) {
        t.sort_by(f64::total_cmp);
        let (best, med) = (t[0], t[t.len() / 2]);
        let got = output_bits(out);
        let differ = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        eprintln!(
            "{:28} M={m} K={k} N={n}: best {:.3} ms ({:.1} TFLOP/s), median {:.3} ms ({:.1}); {differ} outputs differ from {}'s",
            arm.0,
            best * 1e3,
            flop / best / 1e12,
            med * 1e3,
            flop / med / 1e12,
            arms[0].0
        );
    }
}
