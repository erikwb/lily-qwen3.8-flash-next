const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/skinny.metal"),
    "\n",
    include_str!("../../metal/skinny_test.metal")
);

/// Test-only explicit staged variant at K-chunk `kc`.
fn gemm_skinny_q4_nt_staged(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    kc: usize,
) -> Result<()> {
    let (m, k, n) = validate_q4(a, w, c)?;
    if kc == 256 {
        return dispatch_q4_staged(ctx, pass, a, w, c, (m, k, n));
    }
    ensure!(m <= 8, "K-chunk {kc} is instantiated only for m <= 8 (m = {})", m);
    let fn_name = match kc {
        512 => "gemm_skinny_q4_bf16_m8_kc512",
        1024 => "gemm_skinny_q4_bf16_m8_kc1024",
        _ => anyhow::bail!("no test skinny q4 instantiation for K-chunk {kc}"),
    };
    let pipeline = ctx.pipeline(fn_name, TEST_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size), &u32_bytes(m)],
        staged_grid(n),
    )
}

/// Test-only register-A variant with an explicit simdgroups-per-threadgroup
/// width (each simdgroup computes `Q4_REG_ROWS_PER_SG[m - 1]` rows).
fn gemm_skinny_q4_nt_reg(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    simdgroups: usize,
) -> Result<()> {
    let (m, k, n) = validate_q4(a, w, c)?;
    ensure!(m <= REG_MAX_M, "register-A kernels cover m <= 8 (m = {m})");
    ensure!(
        k.is_multiple_of(32) && w.group_size.is_multiple_of(32),
        "register-A q4 needs K % 32 == 0 and group size % 32 == 0 (k = {} gs = {})",
        k,
        w.group_size
    );
    ensure!((1..=32).contains(&simdgroups), "simdgroups {simdgroups} out of 1..=32");
    let pipeline = ctx.pipeline(Q4_REG_FNS[m - 1], SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        reg_grid(n, simdgroups, Q4_REG_ROWS_PER_SG[m - 1]),
    )
}

/// Test-only dispatch of the register-A Q4 kernels as shipped before the
/// explicit-order rewrite (`tests/metal/skinny_test.metal`): two weight rows
/// per simdgroup, two simdgroups per threadgroup.
fn gemm_skinny_q4_nt_previous(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
) -> Result<()> {
    const BF16: [&str; REG_MAX_M] = [
        "gemm_skinny_q4_bf16_reg_m1_previous",
        "gemm_skinny_q4_bf16_reg_m2_previous",
        "gemm_skinny_q4_bf16_reg_m3_previous",
        "gemm_skinny_q4_bf16_reg_m4_previous",
        "gemm_skinny_q4_bf16_reg_m5_previous",
        "gemm_skinny_q4_bf16_reg_m6_previous",
        "gemm_skinny_q4_bf16_reg_m7_previous",
        "gemm_skinny_q4_bf16_reg_m8_previous",
    ];
    const F32: [&str; REG_MAX_M] = [
        "gemm_skinny_q4_f32_reg_m1_previous",
        "gemm_skinny_q4_f32_reg_m2_previous",
        "gemm_skinny_q4_f32_reg_m3_previous",
        "gemm_skinny_q4_f32_reg_m4_previous",
        "gemm_skinny_q4_f32_reg_m5_previous",
        "gemm_skinny_q4_f32_reg_m6_previous",
        "gemm_skinny_q4_f32_reg_m7_previous",
        "gemm_skinny_q4_f32_reg_m8_previous",
    ];
    let (m, k, n) = validate_q4(a, w, c)?;
    ensure!(m <= REG_MAX_M, "register-A kernels cover m <= 8 (m = {m})");
    let names = if c.dtype() == DType::F32 { &F32 } else { &BF16 };
    let pipeline = ctx.pipeline(names[m - 1], TEST_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        Grid::Threadgroups { groups: (n.div_ceil(4), 1, 1), threadgroup: (64, 1, 1) },
    )
}

use half::bf16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

const GROUP_SIZE: usize = 64;

/// M values covering register-A and both sides of the staged-A boundary.
const M_SWEEP: [usize; 6] = [1, 2, 5, 8, 9, 16];

/// Tolerance against an f32 reference over BF16-rounded operands (the
/// staged kernels).
const ATOL: f32 = 2e-2;
const RTOL: f32 = 2e-2;
/// Tolerance of the register-A kernels against an f32 reference over the
/// unrounded dequantized weights: the bf16 output rounding (2^-9) and the
/// f32 accumulation order.
const ATOL_REG: f32 = 4e-3;
const RTOL_REG: f32 = 4e-3;

/// The f32 reference of `gemm_skinny_q4_nt` at this shape and its
/// tolerance: the register-A route (m <= 8) dots the unrounded dequantized
/// weights like the decode GEMV, the staged route rounds them to bf16
/// first.
fn q4_reference(
    a: &[f32],
    dequant: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (Vec<f32>, f32, f32) {
    let a_r = cpu_ref::round_bf16(a);
    if reg_routes(m, n, q4_block_walk_ok(k, GROUP_SIZE)) {
        (cpu_ref::gemm_nt(&a_r, dequant, m, k, n), ATOL_REG, RTOL_REG)
    } else {
        (cpu_ref::gemm_nt(&a_r, &cpu_ref::round_bf16(dequant), m, k, n), ATOL, RTOL)
    }
}

fn assert_q4_close(
    got: &[f32],
    a: &[f32],
    dequant: &[f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let (expected, atol, rtol) = q4_reference(a, dequant, m, k, n);
    cpu_ref::assert_close(got, &expected, atol, rtol);
}

fn random_vec(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

/// A random 4-bit affine weight plus its exact f32 dequant image (the
/// quant.rs test-fixture recipe).
fn random_quant(
    ctx: &MetalContext,
    rng: &mut StdRng,
    n: usize,
    k: usize,
    gs: usize,
) -> (QuantWeights, Vec<f32>) {
    let words = k / 8;
    let groups = k / gs;
    let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
    let scales: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(0.01f32..0.5)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|_| bf16::from_f32(rng.gen_range(-2.0f32..0.0)).to_f32())
        .collect();
    let dequant = cpu_ref::dequant_q4(&codes, &scales, &biases, n, k, gs);
    (quant_tensors(ctx, codes, &scales, &biases, n, k, gs), dequant)
}

/// Uploads CPU-side q4 arrays as a [`QuantWeights`] (scales/biases are
/// already bf16-rounded f32).
fn quant_tensors(
    ctx: &MetalContext,
    codes: Vec<u32>,
    scales: &[f32],
    biases: &[f32],
    n: usize,
    k: usize,
    gs: usize,
) -> QuantWeights {
    let to_bf16 =
        |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, k / 8],
            DType::U32,
        )
        .expect("codes"),
        scales: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(scales)),
            &[n, k / gs],
            DType::BF16,
        )
        .expect("scales"),
        biases: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&to_bf16(biases)),
            &[n, k / gs],
            DType::BF16,
        )
        .expect("biases"),
        group_size: gs,
        bits: 4,
    }
}

/// Deterministic Q4 weights for tests that dequantize sampled rows on demand.
fn hash_quant_raw(
    ctx: &MetalContext,
    n: usize,
    k: usize,
    gs: usize,
) -> (QuantWeights, Vec<u32>, Vec<f32>, Vec<f32>) {
    let words = k / 8;
    let groups = k / gs;
    let codes: Vec<u32> = (0..n * words)
        .map(|i| {
            let h1 = (i as u32).wrapping_mul(2654435761).wrapping_add(97);
            let h2 = (i as u32).wrapping_mul(0x9E37_79B9).wrapping_add(0xA5A5_A5A5);
            (h1 & 0xFFFF_0000) | (h2 >> 16)
        })
        .collect();
    let unit = |i: usize, salt: u32| -> f32 {
        let h = (i as u32).wrapping_mul(0x9E37_79B9).wrapping_add(salt);
        (h >> 8) as f32 / (1u32 << 24) as f32
    };
    let scales: Vec<f32> = (0..n * groups)
        .map(|i| bf16::from_f32(0.01 + 0.49 * unit(i, 5)).to_f32())
        .collect();
    let biases: Vec<f32> = (0..n * groups)
        .map(|i| bf16::from_f32(-2.0 + 2.0 * unit(i, 9)).to_f32())
        .collect();
    let w = quant_tensors(ctx, codes.clone(), &scales, &biases, n, k, gs);
    (w, codes, scales, biases)
}

/// Checks register/staged routing, N tails, and K-chunk tails against f32.
#[test]
fn gemm_skinny_q4_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(41);
    for (n, k) in [
        (16usize, 1024usize),
        (512, 2048),
        (2048, 512),
        (33, 320),
        (256, 64),
        (4, 256),
        (64, 256),
        (1024, 256),
        (256, 512),
        (1, 256),
        (2, 64),
    ] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in M_SWEEP {
            if m * n * k > 150_000_000 {
                continue;
            }
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
            pass.commit_wait().expect("commit");
            assert_q4_close(&tc.to_f32().expect("read"), &a, &dequant, m, k, n);
        }
    }
}

/// The 16-byte A staging path needs a 16B-aligned base; a bf16 view at an
/// odd 4-byte offset must take the scalar staging path and still agree.
#[test]
fn gemm_skinny_q4_unaligned_a_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(43);
    let (n, k, m) = (48usize, 320usize, 5usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let mut padded = vec![0.0f32; 2];
    padded.extend_from_slice(&a);
    let backing =
        Tensor::from_f32_as_bf16(&ctx, &padded, &[m * k + 2]).expect("backing");
    // Byte offset 4: valid for Metal (4B) but not 16B-aligned.
    let ta = backing.view(2, &[m, k]).expect("view");
    let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
    pass.commit_wait().expect("commit");
    assert_q4_close(&tc.to_f32().expect("read"), &a, &dequant, m, k, n);
}

/// Samples boundary rows and a prime-stride walk across wide outputs.
fn sample_rows(n: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..16).chain(n - 16..n).collect();
    rows.extend((0..n).step_by(9973));
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// Checks the wide-output register range, final-row tail, and staged fallback.
/// The f32 reference evaluates sampled rows at dispatch boundaries.
#[test]
fn gemm_skinny_q4_vocab_shape_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(44);
    let (k, gs) = (2048usize, 64usize);
    let (words, groups) = (k / 8, k / gs);
    for n in [248320usize, 248317] {
        let (w, codes, scales, biases) = hash_quant_raw(&ctx, n, k, gs);
        let rows = sample_rows(n);
        for m in 1..=9usize {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
            pass.commit_wait().expect("commit");
            let got_full = tc.to_f32().expect("read");
            let a_r = cpu_ref::round_bf16(&a);
            let mut got = Vec::new();
            let mut want = Vec::new();
            for &r in &rows {
                let deq = cpu_ref::dequant_q4(
                    &codes[r * words..(r + 1) * words],
                    &scales[r * groups..(r + 1) * groups],
                    &biases[r * groups..(r + 1) * groups],
                    1,
                    k,
                    gs,
                );
                let reg = reg_routes(m, n, q4_block_walk_ok(k, gs));
                let w_r = if reg { deq } else { cpu_ref::round_bf16(&deq) };
                for i in 0..m {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a_r[i * k + kk] * w_r[kk];
                    }
                    want.push(sum);
                    got.push(got_full[i * n + r]);
                }
            }
            let reg = reg_routes(m, n, q4_block_walk_ok(k, gs));
            let (atol, rtol) = if reg { (ATOL_REG, RTOL_REG) } else { (ATOL, RTOL) };
            cpu_ref::assert_close(&got, &want, atol, rtol);
        }
    }
}

/// Wide N used to exercise register-A routing.
const WIDE_N: usize = 65536;

#[test]
fn reg_route_covers_small_m_everywhere_and_wide_n_up_to_the_ceiling() {
    // Narrow layer widths take register-A for small m (the staged walk is
    // latency-bound there) and nothing past the instantiation ceiling.
    assert!(reg_routes(1, WIDE_N - 1, true));
    assert!(reg_routes(REG_SMALL_M, WIDE_N - 1, true));
    assert!(!reg_routes(REG_MAX_M + 1, WIDE_N - 1, true));
    assert!(reg_routes(1, WIDE_N, true));
    assert!(!reg_routes(1, WIDE_N, false));
}

#[test]
fn fused_stack_requires_one_route_for_stack_and_slices() {
    assert!(stack_route_uniform(1, WIDE_N - 1, &[1024, 2048], true));
    // Small m: every width routes register-A, so any stack is uniform.
    assert!(stack_route_uniform(1, WIDE_N, &[WIDE_N - 1, 1], true));
    assert!(stack_route_uniform(REG_MAX_M + 1, WIDE_N, &[WIDE_N - 1, 1], true));
    assert!(stack_route_uniform(1, WIDE_N, &[WIDE_N], false));
}

/// The route boundary itself: the register body serves m up to its
/// instantiation ceiling and nothing past it.
#[test]
fn gemm_skinny_q4_wide_route_boundary_matches_reference() {
    assert!(reg_routes(1, WIDE_N, true));
    assert!(reg_routes(REG_MAX_M, WIDE_N, true));
    assert!(
        !reg_routes(REG_MAX_M + 1, WIDE_N, true),
        "m past the register instantiations must stage"
    );
    assert!(
        !reg_routes(REG_MAX_M, WIDE_N, false),
        "a failed block-walk precondition must stage"
    );

    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(45);
    let k = 256usize;
    let check = |w: &QuantWeights, dequant: &[f32], n: usize, m: usize| {
        let a = random_vec(&mut StdRng::seed_from_u64((n + m) as u64), m * k);
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
        let pass = ctx.begin().expect("pass");
        gemm_skinny_q4_nt(&ctx, &pass, &ta, w, &tc).expect("skinny q4");
        pass.commit_wait().expect("commit");
        assert_q4_close(&tc.to_f32().expect("read"), &a, dequant, m, k, n);
    };
    for (n, ms) in [(WIDE_N - 4, &[8usize][..]), (WIDE_N, &[1usize, 8, 9][..])] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for &m in ms {
            check(&w, &dequant, n, m);
        }
    }
}

/// The register kernel's scalar-A fallback: a bf16 view at an odd 4-byte
/// offset is not 16B-aligned, so the wide route's per-lane A loads must
/// take the element path and still agree.
#[test]
fn gemm_skinny_q4_wide_unaligned_a_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(47);
    let (n, k, m) = (WIDE_N, 256usize, 3usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let mut padded = vec![0.0f32; 2];
    padded.extend_from_slice(&a);
    let backing =
        Tensor::from_f32_as_bf16(&ctx, &padded, &[m * k + 2]).expect("backing");
    let ta = backing.view(2, &[m, k]).expect("view");
    let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &tc).expect("skinny q4");
    pass.commit_wait().expect("commit");
    assert_q4_close(&tc.to_f32().expect("read"), &a, &dequant, m, k, n);
}

/// Checks explicit staged K-chunk and register threadgroup-width variants.
#[test]
fn gemm_skinny_q4_variants_match_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(48);
    for (n, k) in [(1024usize, 1280usize), (64, 320)] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in [1usize, 5, 8] {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            // The explicit staged variants round the weights to bf16.
            let expected = cpu_ref::gemm_nt(
                &cpu_ref::round_bf16(&a),
                &cpu_ref::round_bf16(&dequant),
                m,
                k,
                n,
            );
            for kc in [512usize, 1024] {
                let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
                let pass = ctx.begin().expect("pass");
                gemm_skinny_q4_nt_staged(&ctx, &pass, &ta, &w, &tc, kc)
                    .expect("skinny q4 kc");
                pass.commit_wait().expect("commit");
                cpu_ref::assert_close(
                    &tc.to_f32().expect("read"),
                    &expected,
                    ATOL,
                    RTOL,
                );
            }
        }
    }
    let (n, k, m) = (WIDE_N, 256usize, 5usize);
    let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
    let a = random_vec(&mut rng, m * k);
    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
    // The register-A variants dot the unrounded weights.
    let expected = cpu_ref::gemm_nt(&cpu_ref::round_bf16(&a), &dequant, m, k, n);
    for rows in [2usize, 4, 8] {
        let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
        let pass = ctx.begin().expect("pass");
        gemm_skinny_q4_nt_reg(&ctx, &pass, &ta, &w, &tc, rows).expect("skinny q4 reg");
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(
            &tc.to_f32().expect("read"),
            &expected,
            ATOL_REG,
            RTOL_REG,
        );
    }
}

/// Creates a row-range view of a Q4 weight.
fn quant_view_rows(q: &QuantWeights, start: usize, rows: usize) -> QuantWeights {
    let words = q.codes.shape()[1];
    let groups = q.scales.shape()[1];
    QuantWeights {
        codes: q.codes.view(start * words, &[rows, words]).expect("codes view"),
        scales: q.scales.view(start * groups, &[rows, groups]).expect("scales"),
        biases: q.biases.view(start * groups, &[rows, groups]).expect("biases"),
        group_size: q.group_size,
        bits: q.bits,
    }
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: bit mismatch at {i} ({g} vs {w})"
        );
    }
}

/// Fused-stack and per-slice dispatches must be bit-identical, including
/// column splitting. Uneven widths and K=320 cover row and K-chunk tails.
#[test]
fn gemm_skinny_fused_stack_matches_per_slice_bits() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(49);
    let k = 320usize;
    // Starts must be even: gs = 64 gives an odd group count (5), and a
    // bf16 scale view at an odd row start would not be 4-byte aligned.
    let widths = [40usize, 24, 4, 4];
    let n_total: usize = widths.iter().sum();
    let (w, _) = random_quant(&ctx, &mut rng, n_total, k, GROUP_SIZE);

    for m in [1usize, 5, 8, 16] {
        let a_vals = random_vec(&mut rng, m * k);
        let a = Tensor::from_f32_as_bf16(&ctx, &a_vals, &[m, k]).expect("a");
        // The register kernels are instantiated for m <= 8 only.
        for use_reg in [false, true] {
            if use_reg && m > REG_MAX_M {
                continue;
            }
            let name = if use_reg { "reg2" } else { "routed" };
            let run = |wq: &QuantWeights, n: usize| -> Vec<f32> {
                let c = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
                let pass = ctx.begin().expect("pass");
                if use_reg {
                    gemm_skinny_q4_nt_reg(&ctx, &pass, &a, wq, &c, 2)
                        .expect("reg dispatch");
                } else {
                    gemm_skinny_q4_nt(&ctx, &pass, &a, wq, &c).expect("dispatch");
                }
                pass.commit_wait().expect("commit");
                c.to_f32().expect("read")
            };
            let fused = run(&w, n_total);
            let mut start = 0usize;
            let mut slice_outs: Vec<Vec<f32>> = Vec::new();
            for &rows in &widths {
                let ws = quant_view_rows(&w, start, rows);
                slice_outs.push(run(&ws, rows));
                start += rows;
            }
            let mut off = 0usize;
            for (si, slice) in slice_outs.iter().enumerate() {
                let mut seg = Vec::with_capacity(m * widths[si]);
                for i in 0..m {
                    for j in 0..widths[si] {
                        seg.push(fused[i * n_total + off + j]);
                    }
                }
                assert_bits_eq(
                    &seg,
                    slice,
                    &format!("{name} fused vs slice {si} at m={m}"),
                );
                off += widths[si];
            }
            // The production epilogue: split the fused output back into
            // the contiguous per-projection tensors, bit-exact.
            let fused_t =
                Tensor::from_f32_as_bf16(&ctx, &fused, &[m, n_total]).expect("c");
            let dsts: Vec<Tensor> = widths
                .iter()
                .map(|&rows| Tensor::zeros(&ctx, &[m, rows], DType::BF16))
                .collect::<Result<_>>()
                .expect("dsts");
            let pass = ctx.begin().expect("pass");
            crate::kernels::elementwise::split_cols_bf16(
                &ctx,
                &pass,
                &fused_t,
                &dsts.iter().collect::<Vec<_>>(),
            )
            .expect("split");
            pass.commit_wait().expect("commit");
            for (si, (d, slice)) in dsts.iter().zip(&slice_outs).enumerate() {
                assert_bits_eq(
                    &d.to_f32().expect("read"),
                    slice,
                    &format!("{name} split segment {si} at m={m}"),
                );
            }
        }
    }
}

/// Hashed Q4 weights for the bit-identity sweeps: every code pattern, scales
/// from 2^-9 to 2^3 and biases of both signs, cheap enough for the LM head.
fn bits_quant(ctx: &MetalContext, n: usize, k: usize, salt: u32) -> QuantWeights {
    let words = k / 8;
    let groups = k / GROUP_SIZE;
    let hash = |i: usize, s: u32| -> u32 {
        let h = (i as u32 ^ s.wrapping_mul(0x85EB_CA6B)).wrapping_mul(0x9E37_79B9);
        h ^ (h >> 15)
    };
    let codes: Vec<u32> = (0..n * words)
        .map(|i| hash(i, salt).wrapping_mul(2_654_435_761).rotate_left(11))
        .collect();
    let value = |i: usize, s: u32, sign: bool| -> f32 {
        let h = hash(i, s);
        let mant = 1.0 + (h & 0xFF) as f32 / 256.0;
        let exp = ((h >> 8) % 13) as i32 - 9;
        let v = mant * 2f32.powi(exp);
        if sign && h & (1 << 20) != 0 { -v } else { v }
    };
    let scales: Vec<f32> =
        (0..n * groups).map(|i| value(i, salt ^ 0x51, false)).collect();
    let biases: Vec<f32> =
        (0..n * groups).map(|i| value(i, salt ^ 0xB1, true)).collect();
    quant_tensors(ctx, codes, &scales, &biases, n, k, GROUP_SIZE)
}

/// Activations for the bit-identity sweeps; `wide` spreads the exponents
/// over 2^-24..2^24 so a changed summation order would round differently,
/// and `zeros` makes whole 32-element blocks -0.0, mixed signed zeros or
/// values, with the first row (`k` elements) all -0.0, so a sum started
/// differently would show in the sign of a zero.
fn bits_activations(
    rng: &mut StdRng,
    len: usize,
    k: usize,
    wide: bool,
    zeros: bool,
) -> Vec<f32> {
    let mut a: Vec<f32> = (0..len)
        .map(|_| {
            let v = rng.gen_range(-1.0f32..1.0);
            if wide { v * 2f32.powi(rng.gen_range(-24i32..24)) } else { v }
        })
        .collect();
    if zeros {
        for (bi, block) in a.chunks_mut(32).enumerate() {
            let pick = if bi * 32 < k { 0 } else { rng.gen_range(0..3) };
            for v in block.iter_mut() {
                match pick {
                    0 => *v = -0.0,
                    1 => *v = if rng.gen_range(0..2) == 0 { -0.0 } else { 0.0 },
                    _ => {}
                }
            }
        }
    }
    a
}

/// The register-A Q4 kernels compute bit for bit what they computed before
/// the explicit-order rewrite (`*_previous`, main 0446016), for every m they
/// serve, both output types, the model's projection shapes (the GDN input
/// stack, the attention qkv-and-gate stack, the output projections, the LM
/// head with f32 logits), row counts that leave a partial row group, K values
/// whose last pass splits or does not, blocks of signed zeros, and an
/// activation pointer that is not 16-byte aligned.
#[test]
fn gemm_skinny_q4_reg_matches_previous_bits() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(97);
    // (K, N, output types)
    let both = [DType::BF16, DType::F32];
    let shapes: [(usize, usize, &[DType]); 12] = [
        (2560, 16480, &[DType::BF16]),
        (2560, 13312, &[DType::BF16]),
        (6144, 2560, &[DType::BF16]),
        (2560, 248320, &[DType::F32]),
        (2560, 1029, &both),
        (6144, 37, &both),
        (64, 5, &both),
        (320, 43, &both),
        (640, 29, &both),
        (1088, 21, &both),
        (3072, 13, &both),
        (5120, 3, &both),
    ];
    for (si, &(k, n, dtypes)) in shapes.iter().enumerate() {
        let w = bits_quant(&ctx, n, k, si as u32 + 1);
        for m in 1..=REG_MAX_M {
            for (case, wide, zeros, unaligned) in [
                ("plain", false, false, false),
                ("wide", true, false, false),
                ("signed zeros", true, true, false),
                ("unaligned", true, false, true),
            ] {
                // The large shapes skip the unaligned case.
                if unaligned && n > 4096 {
                    continue;
                }
                let a = bits_activations(&mut rng, m * k, k, wide, zeros);
                let pad = if unaligned { 2 } else { 0 };
                let mut padded = vec![0.0f32; pad];
                padded.extend_from_slice(&a);
                let backing = Tensor::from_f32_as_bf16(&ctx, &padded, &[m * k + pad])
                    .expect("backing");
                let ta = backing.view(pad, &[m, k]).expect("a view");
                for &dt in dtypes {
                    let got = Tensor::zeros(&ctx, &[m, n], dt).expect("c");
                    let want = Tensor::zeros(&ctx, &[m, n], dt).expect("c");
                    let pass = ctx.begin().expect("pass");
                    gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &got).expect("shipped");
                    gemm_skinny_q4_nt_previous(&ctx, &pass, &ta, &w, &want)
                        .expect("previous");
                    pass.commit_wait().expect("commit");
                    assert_bits_eq(
                        &got.to_f32().expect("read"),
                        &want.to_f32().expect("read"),
                        &format!("K={k} N={n} m={m} {case} {dt:?}"),
                    );
                }
            }
        }
    }
}

/// Checks explicit register-A kernels across regular and boundary shapes.
#[test]
fn gemm_skinny_reg_layer_shapes_match_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(51);
    for (n, k) in [
        (16usize, 1024usize),
        (512, 2048),
        (2048, 512),
        (256, 64),
        (4, 256),
        (1, 256),
        (2, 64),
        (64, 320),
    ] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in 1..=REG_MAX_M {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let expected =
                cpu_ref::gemm_nt(&cpu_ref::round_bf16(&a), &dequant, m, k, n);
            let tc = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt_reg(&ctx, &pass, &ta, &w, &tc, 2).expect("q4 reg");
            pass.commit_wait().expect("commit");
            cpu_ref::assert_close(
                &tc.to_f32().expect("read"),
                &expected,
                ATOL_REG,
                RTOL_REG,
            );
        }
    }
}

/// The fixed routing predicate includes its threshold.
#[test]
fn dense_smallm_routing_boundary() {
    assert!(dense_smallm_routes(1));
    assert!(dense_smallm_routes(DENSE_SMALLM_THRESHOLD));
    assert!(!dense_smallm_routes(DENSE_SMALLM_THRESHOLD + 1));
    assert!(!dense_smallm_routes(0));
}

#[test]
fn skinny_rejects_partial_group_tail() {
    let ctx = MetalContext::new().expect("metal context");
    let (m, n, k) = (1usize, 4usize, 96usize);
    let w = quant_tensors(&ctx, vec![0; n * k / 8], &[1.0; 4], &[0.0; 4], n, k, 64);
    let a = Tensor::zeros(&ctx, &[m, k], DType::BF16).expect("a");
    let c = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let pass = ctx.begin().expect("pass");
    let err = gemm_skinny_q4_nt(&ctx, &pass, &a, &w, &c).expect_err("K=96 must reject");
    assert!(err.to_string().contains("K % 64 == 0"), "{err:#}");
}

/// The f32-output variants (used for batched logits) compute exactly what the
/// bf16 variants round: same reduction order, only the final store differs.
/// Covers both the staged family and the wide register-A family.
#[test]
fn gemm_skinny_q4_f32_out_is_the_unrounded_bf16_result() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(77);
    for (n, k) in [(512usize, 256usize), (WIDE_N_MIN + 64, 256)] {
        let (w, dequant) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        for m in [1usize, 3, 8, 12] {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let c_bf = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c bf16");
            let c_f32 = Tensor::zeros(&ctx, &[m, n], DType::F32).expect("c f32");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &c_bf).expect("bf16 out");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &c_f32).expect("f32 out");
            pass.commit_wait().expect("commit");
            let got_f32 = c_f32.to_f32().expect("read f32");
            let got_bf = c_bf.to_f32().expect("read bf16");
            assert_bits_eq(
                &cpu_ref::round_bf16(&got_f32),
                &got_bf,
                &format!("m={m} n={n} rounded f32 vs bf16"),
            );
            assert_q4_close(&got_f32, &a, &dequant, m, k, n);
        }
    }
}

/// The Q8 staged kernel matches the CPU reference over bf16-rounded
/// dequantized weights, in bf16 and f32 output, for the narrow router/mixer
/// shapes it serves.
#[test]
fn gemm_skinny_q8_matches_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(78);
    for (n, k) in [(512usize, 2560usize), (320, 10240), (4, 10240), (1, 2560), (33, 64)]
    {
        let gs = 64;
        let words = k / 4;
        let groups = k / gs;
        let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
        let scales: Vec<f32> = (0..n * groups)
            .map(|_| bf16::from_f32(rng.gen_range(0.001f32..0.02)).to_f32())
            .collect();
        let biases: Vec<f32> = (0..n * groups)
            .map(|_| bf16::from_f32(rng.gen_range(-2.0f32..0.0)).to_f32())
            .collect();
        let dequant = cpu_ref::dequant_affine(&codes, &scales, &biases, n, k, gs, 8);
        let to_bf16 =
            |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
        let w = QuantWeights {
            codes: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&codes),
                &[n, words],
                DType::U32,
            )
            .expect("codes"),
            scales: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&to_bf16(&scales)),
                &[n, groups],
                DType::BF16,
            )
            .expect("scales"),
            biases: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&to_bf16(&biases)),
                &[n, groups],
                DType::BF16,
            )
            .expect("biases"),
            group_size: gs,
            bits: 8,
        };
        for m in [1usize, 3, 4, 8, 13] {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let c_bf = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
            let c_f32 = Tensor::zeros(&ctx, &[m, n], DType::F32).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q8_nt(&ctx, &pass, &ta, &w, &c_bf).expect("q8 bf16");
            gemm_skinny_q8_nt(&ctx, &pass, &ta, &w, &c_f32).expect("q8 f32");
            pass.commit_wait().expect("commit");
            // The f32 output takes the staged kernel (bf16-rounded weights).
            let a_r = cpu_ref::round_bf16(&a);
            let expected =
                cpu_ref::gemm_nt(&a_r, &cpu_ref::round_bf16(&dequant), m, k, n);
            let got_f32 = c_f32.to_f32().expect("read");
            cpu_ref::assert_close(&got_f32, &expected, ATOL, RTOL);
            // bf16 output takes the register-A kernel for m <= 8, which dots
            // the unrounded weights.
            let got_bf = c_bf.to_f32().expect("read");
            if m <= REG_MAX_M {
                let expected_reg = cpu_ref::gemm_nt(&a_r, &dequant, m, k, n);
                cpu_ref::assert_close(&got_bf, &expected_reg, ATOL_REG, RTOL_REG);
            } else {
                cpu_ref::assert_close(&got_bf, &expected, ATOL, RTOL);
            }
        }
    }
}

/// A `--q8-dense` LM head: f32 output at least `WIDE_N_MIN` wide takes the
/// Q8 register-A kernel for m <= 8, which dots the unrounded weights like
/// the decode GEMV (so verify logits carry decode's numerics); m = 1 is
/// checked against that GEMV directly.
#[test]
fn gemm_skinny_q8_wide_f32_takes_register_a() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(79);
    let (n, k) = (WIDE_N_MIN + 4, 128);
    let w = random_quant_bits(&ctx, &mut rng, n, k, GROUP_SIZE, 8);
    let codes = w.codes.to_u32().expect("codes");
    let scales = w.scales.to_f32().expect("scales");
    let biases = w.biases.to_f32().expect("biases");
    let dequant =
        cpu_ref::dequant_affine(&codes, &scales, &biases, n, k, GROUP_SIZE, 8);
    for m in [1usize, 2, 5, 8, 11] {
        let a = random_vec(&mut rng, m * k);
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let c = Tensor::zeros(&ctx, &[m, n], DType::F32).expect("c");
        let pass = ctx.begin().expect("pass");
        gemm_skinny_q8_nt(&ctx, &pass, &ta, &w, &c).expect("q8 f32");
        pass.commit_wait().expect("commit");
        let a_r = cpu_ref::round_bf16(&a);
        let got = c.to_f32().expect("read");
        if m <= REG_MAX_M {
            let expected = cpu_ref::gemm_nt(&a_r, &dequant, m, k, n);
            cpu_ref::assert_close(&got, &expected, ATOL_REG, RTOL_REG);
        } else {
            let expected =
                cpu_ref::gemm_nt(&a_r, &cpu_ref::round_bf16(&dequant), m, k, n);
            cpu_ref::assert_close(&got, &expected, ATOL, RTOL);
        }
        if m == 1 {
            let tx = Tensor::from_f32_as_bf16(&ctx, &a, &[k]).expect("x");
            let y = Tensor::zeros(&ctx, &[n], DType::F32).expect("y");
            let pass = ctx.begin().expect("pass");
            gemv_quant(&ctx, &pass, &w, &tx, &y).expect("gemv");
            pass.commit_wait().expect("commit");
            cpu_ref::assert_close(&got, &y.to_f32().expect("y"), ATOL_REG, RTOL_REG);
        }
    }
}

/// The verify-pass question of the optimization notes (§ 2.2): the same
/// dense bytes through the m-row register-A kernel against the decode
/// GEMV. Streams 64 weight sets per shape (past the SLC) and prints µs per
/// dispatch and the weight bandwidth for m = 1..4, on the model's attention
/// and GDN projection shapes. Run with
/// Random affine weights at `bits` (4 or 8) for timing; no CPU dequant.
fn random_quant_bits(
    ctx: &MetalContext,
    rng: &mut StdRng,
    n: usize,
    k: usize,
    gs: usize,
    bits: usize,
) -> QuantWeights {
    let words = k * bits / 32;
    let groups = k / gs;
    let codes: Vec<u32> = (0..n * words).map(|_| rng.r#gen()).collect();
    let bf = |v: Vec<f32>| -> Vec<bf16> { v.into_iter().map(bf16::from_f32).collect() };
    let scales = bf((0..n * groups).map(|_| rng.gen_range(0.001f32..0.05)).collect());
    let biases = bf((0..n * groups).map(|_| rng.gen_range(-2.0f32..0.0)).collect());
    let upload = |v: &[bf16]| {
        Tensor::from_bytes(ctx, bytemuck::cast_slice(v), &[n, groups], DType::BF16)
            .expect("bf16")
    };
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, words],
            DType::U32,
        )
        .expect("codes"),
        scales: upload(&scales),
        biases: upload(&biases),
        group_size: gs,
        bits,
    }
}

/// Times the decode GEMV against the one-row register-A kernel on every
/// projection shape a Qwen3.8-Flash-Next decode step dispatches through
/// `gemv_quant` (Q4 unless noted; the LM head writes f32), plus register-A
/// at 2 to 4 rows. Weight sets rotate so the weights stream from memory
/// rather than the last-level cache, as in a decode step.
/// `cargo test --release -- --ignored --nocapture skinny_reg_vs_gemv_timing`.
#[test]
#[ignore = "timing only"]
fn skinny_reg_vs_gemv_timing() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(61);
    let iters = 128;
    for (bits, n, k, f32_out, what) in [
        (4usize, 248320usize, 2560usize, true, "lm_head"),
        (4, 16480, 2560, false, "gdn in_proj"),
        (4, 2560, 6144, false, "gdn out_proj / attn o_proj"),
        (4, 13312, 2560, false, "attn qkv_proj"),
        (4, 1280, 2560, false, "shared gate_up"),
        (4, 2560, 640, false, "shared down"),
        (8, 512, 2560, false, "moe gate (q8)"),
        (8, 640, 2560, false, "indexer qk_proj (q8)"),
        (8, 10240, 2560, false, "ple key_proj (q8)"),
        (8, 2560, 2560, false, "ple value_proj / mtp fc (q8)"),
        (8, 1, 2560, false, "shared_expert_gate (q8)"),
    ] {
        let bytes = n * k * bits / 8 + 2 * n * (k / GROUP_SIZE) * 2;
        let sets = (256usize << 20).div_ceil(bytes).clamp(1, 64);
        let weights: Vec<QuantWeights> = (0..sets)
            .map(|_| random_quant_bits(&ctx, &mut rng, n, k, GROUP_SIZE, bits))
            .collect();
        let out = if f32_out { DType::F32 } else { DType::BF16 };
        let x =
            Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, k), &[k]).expect("x");
        let y = Tensor::zeros(&ctx, &[n], out).expect("y");
        let time = |name: &str, f: &dyn Fn(&ComputePass<'_>, &QuantWeights)| {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let pass = ctx.begin_concurrent().expect("pass");
                for i in 0..iters {
                    f(&pass, &weights[i % sets]);
                }
                let start = std::time::Instant::now();
                pass.commit_wait().expect("commit");
                best = best.min(start.elapsed().as_secs_f64() * 1e6 / iters as f64);
            }
            eprintln!(
                "{what} [{n} x {k}] {name}: {best:.1} us per dispatch, {:.0} GB/s of weights",
                bytes as f64 / best / 1e3
            );
        };
        time("decode GEMV (m = 1)", &|pass, w| {
            gemv_quant(&ctx, pass, w, &x, &y).unwrap()
        });
        let rows = if f32_out { 1 } else { 4 };
        for m in 1..=rows {
            let a =
                Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, m * k), &[m, k])
                    .expect("a");
            let c = Tensor::zeros(&ctx, &[m, n], out).expect("c");
            time(&format!("register-A skinny (m = {m})"), &|pass, w| {
                if bits == 4 {
                    gemm_skinny_q4_nt(&ctx, pass, &a, w, &c).unwrap()
                } else {
                    gemm_skinny_q8_nt(&ctx, pass, &a, w, &c).unwrap()
                }
            });
        }
    }
}

/// The one-row register-A kernel and the decode GEMV are the same
/// construction (raw-code dot, scale and bias per block or word), so they
/// agree to the f32 accumulation order on the model's projection shapes.
#[test]
fn gemm_skinny_reg_m1_matches_decode_gemv() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(63);
    for (n, k) in [(2560usize, 2560usize), (320, 6144), (48, 2560), (7680, 2560)] {
        let (w, _) = random_quant(&ctx, &mut rng, n, k, GROUP_SIZE);
        let x = random_vec(&mut rng, k);
        let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[k]).expect("x");
        let ta = Tensor::from_f32_as_bf16(&ctx, &x, &[1, k]).expect("a");
        let y = Tensor::zeros(&ctx, &[n], DType::BF16).expect("y");
        let c = Tensor::zeros(&ctx, &[1, n], DType::BF16).expect("c");
        let pass = ctx.begin().expect("pass");
        gemv_quant(&ctx, &pass, &w, &tx, &y).expect("gemv");
        gemm_skinny_q4_nt(&ctx, &pass, &ta, &w, &c).expect("skinny");
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(
            &c.to_f32().expect("c"),
            &y.to_f32().expect("y"),
            ATOL_REG,
            RTOL_REG,
        );
    }
}

/// On Vulkan a row of the 8-bit register-A GEMM (`vulkan/quant_rows.glsl`)
/// depends on nothing but the row: each activation row's output at m = 2..8
/// is the bits the one-row GEMM gives it, a row range dispatched alone gives
/// the bits it has in the whole matrix, and so do one and two weight rows
/// per lane (the small and the large shapes here take one and the other).
/// K = 640 walks a row with 8 lanes per set and K = 2560 with 32.
#[cfg(target_os = "linux")]
#[test]
fn gemm_skinny_q8_reg_rows_do_not_depend_on_the_dispatch() {
    use crate::kernels::quant::rows_layout;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(81);
    for (n, k, rows) in [(515usize, 2560usize, 1usize), (2562, 640, 1), (1283, 2560, 2)] {
        let w = random_quant_bits(&ctx, &mut rng, n, k, GROUP_SIZE, 8);
        assert_eq!(rows_layout(&w).1, rows, "rows per lane at [{n}, {k}]");
        // Slices at an odd and an even row, one small enough to take one
        // row per lane where the whole matrix takes two.
        let slices = [(0usize, n), (3, 100), (n - 258, 258)];
        for m in 1..=REG_MAX_M {
            let a = random_vec(&mut rng, m * k);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let run = |ta: &Tensor, m: usize, wq: &QuantWeights, rows: usize| -> Vec<f32> {
                let c = Tensor::zeros(&ctx, &[m, rows], DType::BF16).expect("c");
                let pass = ctx.begin().expect("pass");
                gemm_skinny_q8_nt(&ctx, &pass, ta, wq, &c).expect("skinny q8");
                pass.commit_wait().expect("commit");
                c.to_f32().expect("read")
            };
            let whole = run(&ta, m, &w, n);
            for i in 0..m {
                let one = run(&ta.view(i * k, &[1, k]).expect("row"), 1, &w, n);
                assert_bits_eq(
                    &whole[i * n..(i + 1) * n],
                    &one,
                    &format!("[{n}, {k}] m={m} row {i} against the one-row GEMM"),
                );
            }
            for (start, len) in slices {
                // A bf16 scale view must start 4-byte aligned.
                if (start * (k / GROUP_SIZE)) % 2 != 0 {
                    continue;
                }
                let got = run(&ta, m, &quant_view_rows(&w, start, len), len);
                for i in 0..m {
                    assert_bits_eq(
                        &got[i * len..(i + 1) * len],
                        &whole[i * n + start..i * n + start + len],
                        &format!("[{n}, {k}] m={m} row {i} rows {start}..+{len}"),
                    );
                }
            }
        }
    }
}

/// The 8-bit projections of a level as one multi-weight dispatch where the
/// backend has one (Vulkan: `gemm_skinny_q8_nt_sets_fused`, whatever
/// `gemm_skinny_q8_nt_sets` picks for the driver), else as
/// `gemm_skinny_q8_nt_sets` runs them.
fn q8_sets_one_dispatch(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    sets: &[(&QuantWeights, &Tensor)],
) -> Result<()> {
    #[cfg(target_os = "linux")]
    return gemm_skinny_q8_nt_sets_fused(ctx, pass, a, sets);
    #[cfg(not(target_os = "linux"))]
    gemm_skinny_q8_nt_sets(ctx, pass, a, sets)
}

/// The multi-weight dispatch of `gemm_skinny_q8_nt_sets` gives each output
/// the bits `gemm_skinny_q8_nt` gives it, for the batched router level's
/// shapes (a 512-row router, the shared expert's 1280-row gate|up stack, its
/// one-row gate) at every row count, with a set that stays out of the
/// register-A dispatch (a narrow f32 output takes the staged kernel) and,
/// past eight rows, all of them.
#[test]
fn gemm_skinny_q8_sets_match_separate_bits() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(83);
    let k = 2560usize;
    // (N, f32 output)
    let shapes = [(512usize, false), (1280, false), (1, false), (33, true), (515, false)];
    let weights: Vec<QuantWeights> = shapes
        .iter()
        .map(|&(n, _)| random_quant_bits(&ctx, &mut rng, n, k, GROUP_SIZE, 8))
        .collect();
    for m in [1usize, 2, 3, 5, 8, 9] {
        let a = random_vec(&mut rng, m * k);
        let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
        let outputs = || -> Vec<Tensor> {
            shapes
                .iter()
                .map(|&(n, f32_out)| {
                    let dtype = if f32_out { DType::F32 } else { DType::BF16 };
                    Tensor::zeros(&ctx, &[m, n], dtype).expect("c")
                })
                .collect()
        };
        let (together, apart) = (outputs(), outputs());
        let pass = ctx.begin().expect("pass");
        let pairs: Vec<(&QuantWeights, &Tensor)> = weights.iter().zip(&together).collect();
        q8_sets_one_dispatch(&ctx, &pass, &ta, &pairs).expect("sets");
        for (w, c) in weights.iter().zip(&apart) {
            gemm_skinny_q8_nt(&ctx, &pass, &ta, w, c).expect("skinny q8");
        }
        pass.commit_wait().expect("commit");
        for (s, (got, want)) in together.iter().zip(&apart).enumerate() {
            assert_bits_eq(
                &got.to_f32().expect("read"),
                &want.to_f32().expect("read"),
                &format!("m={m} set {s} {:?}", shapes[s]),
            );
        }
    }
}

/// Q4 weights for timing only: cheap hashed codes, plausible scales and
/// biases, no CPU image.
fn timing_quant(ctx: &MetalContext, n: usize, k: usize, salt: u32) -> QuantWeights {
    let words = k / 8;
    let groups = k / GROUP_SIZE;
    let codes: Vec<u32> = (0..n * words)
        .map(|i| (i as u32 ^ salt).wrapping_mul(2_654_435_761).rotate_left(7))
        .collect();
    let scales: Vec<f32> =
        (0..n * groups).map(|i| 0.01 + (i % 97) as f32 * 0.004).collect();
    let biases: Vec<f32> =
        (0..n * groups).map(|i| -2.0 + (i % 89) as f32 * 0.02).collect();
    quant_tensors(ctx, codes, &scales, &biases, n, k, GROUP_SIZE)
}

/// How the shared-weight projections of a batched decode step scale with
/// its row count (`docs/architecture.md`, "Continuous batching"): every row
/// reads the same weights, so a bandwidth-bound kernel would take about as
/// long at four rows as at one. Measured 2026-10-06 in the kernel profile,
/// the two-row register-A skinny GEMM took about 44 % longer at m = 4 than
/// at m = 1 (its per-block activation work bound it); the shipped kernel
/// shares that work across more rows per simdgroup. For Qwen3.8-Flash-Next's
/// Q4 shapes (the GDN input stack, the attention qkv-and-gate stack, the two
/// output projections, the LM head with f32 logits), this times each kernel
/// variant at m = 1..8 over enough distinct copies of the matrix (1.2 GB and
/// up) that the system cache does not serve them, best and median of five
/// passes, and prints the effective weight bandwidth and the time relative
/// to m = 1. Variants: the shipped route (`gemm_skinny_q4_nt`), the two-row
/// kernel it replaced (`*_previous`, bit-identical results), both again
/// with a barrier after every dispatch (as in the model's passes), register-A
/// with 4 simdgroups per threadgroup instead of 2, the staged route
/// (activations staged in threadgroup memory, weights rounded to bf16), and
/// the decode GEMV at m = 1 as the bandwidth reference. Compare with
/// `gemv_q4_bandwidth_probe` and `memory_bandwidth_probe`.
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn skinny_rows_scaling_probe() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    // (name, K, N, f32 output)
    let shapes: [(&str, usize, usize, bool); 5] = [
        ("gdn in stack 2560->16480", 2560, 16480, false),
        ("attn qkv+gate 2560->13312", 2560, 13312, false),
        ("gdn out 6144->2560", 6144, 2560, false),
        ("attn out 6144->2560", 6144, 2560, false),
        ("lm head 2560->248320 f32", 2560, 248320, true),
    ];
    let target_bytes = 1.2e9;
    let mut rng = StdRng::seed_from_u64(31);
    for (name, k, n, f32_out) in shapes {
        let probe = timing_quant(&ctx, n, k, 0);
        let per_matrix = (probe.codes.byte_len()
            + probe.scales.byte_len()
            + probe.biases.byte_len()) as f64;
        drop(probe);
        let copies = ((target_bytes / per_matrix).ceil() as usize).max(2);
        let weights: Vec<QuantWeights> =
            (0..copies).map(|c| timing_quant(&ctx, n, k, c as u32 + 1)).collect();
        let total = per_matrix * copies as f64;
        eprintln!(
            "\n{name}: {copies} copies of {:.1} MB ({:.2} GB per pass)",
            per_matrix / 1e6,
            total / 1e9
        );
        let out_dtype = if f32_out { DType::F32 } else { DType::BF16 };
        let base: std::cell::Cell<Option<f64>> = std::cell::Cell::new(None);
        let time = |encode: &dyn Fn(&ComputePass<'_>) -> Result<()>| -> (f64, f64) {
            let mut times = Vec::new();
            for _ in 0..5 {
                let pass = ctx.begin_concurrent().expect("pass");
                encode(&pass).expect("encode");
                let done = pass.commit().expect("commit").wait_retain().expect("wait");
                let t = done.timing().expect("timing");
                times.push(t.gpu_end_secs - t.gpu_start_secs);
            }
            times.sort_by(f64::total_cmp);
            (times[0], times[times.len() / 2])
        };
        let report = |label: &str, m: usize, (best, median): (f64, f64)| {
            let rel = base.get().map_or(String::new(), |b| {
                format!(", x{:.2} vs shipped m=1", best / b)
            });
            eprintln!(
                "  {label:<26} m={m}: {:7.3} ms best, {:7.3} median, {:4.0} GB/s{rel}",
                best * 1e3,
                median * 1e3,
                total / best / 1e9
            );
        };
        if !f32_out {
            let x: Vec<f32> = (0..k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[k]).expect("x");
            let ys: Vec<Tensor> = (0..copies)
                .map(|_| Tensor::zeros(&ctx, &[n], DType::BF16).expect("y"))
                .collect();
            let t = time(&|pass| {
                for (w, y) in weights.iter().zip(&ys) {
                    gemv_quant(&ctx, pass, w, &tx, y)?;
                }
                Ok(())
            });
            report("decode gemv (reference)", 1, t);
        }
        for m in 1..=REG_MAX_M {
            let a: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let cs: Vec<Tensor> = (0..copies)
                .map(|_| Tensor::zeros(&ctx, &[m, n], out_dtype).expect("c"))
                .collect();
            let shipped = time(&|pass| {
                for (w, c) in weights.iter().zip(&cs) {
                    gemm_skinny_q4_nt(&ctx, pass, &ta, w, c)?;
                }
                Ok(())
            });
            if m == 1 {
                base.set(Some(shipped.0));
            }
            report("shipped route", m, shipped);
            let previous = time(&|pass| {
                for (w, c) in weights.iter().zip(&cs) {
                    gemm_skinny_q4_nt_previous(&ctx, pass, &ta, w, c)?;
                }
                Ok(())
            });
            report("two-row kernel (previous)", m, previous);
            // A barrier after every dispatch, as in the model's passes: each
            // dispatch then fills the GPU on its own and pays its own tail.
            let shipped_b = time(&|pass| {
                for (w, c) in weights.iter().zip(&cs) {
                    gemm_skinny_q4_nt(&ctx, pass, &ta, w, c)?;
                    pass.level_barrier(&[])?;
                }
                Ok(())
            });
            report("shipped, barriers", m, shipped_b);
            let previous_b = time(&|pass| {
                for (w, c) in weights.iter().zip(&cs) {
                    gemm_skinny_q4_nt_previous(&ctx, pass, &ta, w, c)?;
                    pass.level_barrier(&[])?;
                }
                Ok(())
            });
            report("previous, barriers", m, previous_b);
            if !f32_out {
                let wide = time(&|pass| {
                    for (w, c) in weights.iter().zip(&cs) {
                        gemm_skinny_q4_nt_reg(&ctx, pass, &ta, w, c, 4)?;
                    }
                    Ok(())
                });
                report("register-A, 4 sg/tg", m, wide);
                let staged = time(&|pass| {
                    for (w, c) in weights.iter().zip(&cs) {
                        gemm_skinny_q4_nt_staged(&ctx, pass, &ta, w, c, 256)?;
                    }
                    Ok(())
                });
                report("staged (bf16 weights)", m, staged);
            }
        }
    }
}

/// Hashed affine weights at `bits` for timing only (no CPU image).
fn timing_quant_bits(
    ctx: &MetalContext,
    n: usize,
    k: usize,
    bits: usize,
    salt: u32,
) -> QuantWeights {
    let words = k * bits / 32;
    let groups = k / GROUP_SIZE;
    let codes: Vec<u32> = (0..n * words)
        .map(|i| (i as u32 ^ salt).wrapping_mul(2_654_435_761).rotate_left(7))
        .collect();
    let scales: Vec<f32> =
        (0..n * groups).map(|i| 0.001 + (i % 97) as f32 * 0.0004).collect();
    let biases: Vec<f32> =
        (0..n * groups).map(|i| -2.0 + (i % 89) as f32 * 0.02).collect();
    let bf = |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
    let upload = |v: &[f32]| {
        Tensor::from_bytes(ctx, bytemuck::cast_slice(&bf(v)), &[n, groups], DType::BF16)
            .expect("bf16")
    };
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, words],
            DType::U32,
        )
        .expect("codes"),
        scales: upload(&scales),
        biases: upload(&biases),
        group_size: GROUP_SIZE,
        bits,
    }
}

/// The dense projections as the model's passes run them: for every dense
/// shape of the q4-xl checkpoint (8-bit attention, shared expert, routers,
/// PLE and LM head; 4-bit GDN), a chain of dispatches over distinct weight
/// sets with a barrier after each, through the decode GEMV (m = 1) and the
/// skinny GEMM at the rows of `LILY_DENSE_ROWS` (default "3", the verify
/// pass of two drafts). Prints microseconds per level and the weight
/// bandwidth, best of several passes after a warm-up pass.
/// `LILY_DENSE_SHAPES` filters shapes by substring.
/// `cargo test --release -- --ignored --nocapture dense_chain_timing`.
#[test]
#[ignore = "timing only"]
fn dense_chain_timing() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(71);
    let rows: Vec<usize> = std::env::var("LILY_DENSE_ROWS")
        .unwrap_or_else(|_| "3".into())
        .split(',')
        .map(|v| v.parse().expect("rows"))
        .collect();
    let filter = std::env::var("LILY_DENSE_SHAPES").ok();
    let mut total_gemv = 0.0;
    let mut total_rows = vec![0.0; rows.len()];
    // (name, bits, N, K, f32 out, dispatches per decode step)
    for (what, bits, n, k, f32_out, per_step) in [
        ("attn qkv", 8usize, 13312usize, 2560usize, false, 12usize),
        ("attn o", 8, 2560, 6144, false, 12),
        ("indexer qk", 8, 640, 2560, false, 12),
        ("shared gate_up", 8, 1280, 2560, false, 48),
        ("shared down", 8, 2560, 640, false, 48),
        ("moe gate", 8, 512, 2560, true, 48),
        ("ple key", 8, 10240, 2560, false, 1),
        ("ple value", 8, 2560, 2560, false, 1),
        ("gdn in", 4, 16480, 2560, false, 36),
        ("gdn out", 4, 2560, 6144, false, 36),
        ("lm head", 8, 248320, 2560, true, 1),
    ] {
        if filter.as_ref().is_some_and(|f| !f.split(',').any(|p| what.contains(p))) {
            continue;
        }
        let bytes = n * k * bits / 8 + 2 * n * (k / GROUP_SIZE) * 2;
        let sets = (768usize << 20).div_ceil(bytes).clamp(3, 48);
        let weights: Vec<QuantWeights> = (0..sets)
            .map(|s| timing_quant_bits(&ctx, n, k, bits, s as u32 + 1))
            .collect();
        let out = if f32_out { DType::F32 } else { DType::BF16 };
        let x =
            Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, k), &[k]).expect("x");
        let y = Tensor::zeros(&ctx, &[n], out).expect("y");
        // A pass as long as a decode step (the sets repeated to
        // `LILY_DENSE_LEVELS` levels, default 512): short passes with the
        // host between them do not hold the GPU's clock.
        let levels: usize = std::env::var("LILY_DENSE_LEVELS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let repeats = levels.div_ceil(sets).max(1);
        let time = |f: &dyn Fn(&ComputePass<'_>, &QuantWeights)| -> f64 {
            // The GPU's clock ramps: 300 ms of the same work first.
            let mut best = f64::MAX;
            let start = std::time::Instant::now();
            let mut measured = 0;
            while measured < 8 {
                let pass = ctx.begin_concurrent().expect("pass");
                for _ in 0..repeats {
                    for w in &weights {
                        f(&pass, w);
                        pass.level_barrier(&[]).expect("barrier");
                    }
                }
                let done = pass.commit().expect("commit").wait_retain().expect("wait");
                let t = done.timing().expect("timing");
                if start.elapsed().as_millis() >= 300 {
                    best = best.min(
                        (t.gpu_end_secs - t.gpu_start_secs) / (repeats * sets) as f64,
                    );
                    measured += 1;
                }
            }
            best
        };
        let gemv = time(&|pass, w| gemv_quant(&ctx, pass, w, &x, &y).unwrap());
        total_gemv += gemv * per_step as f64;
        let mut line = format!(
            "{what:<14} q{bits} [{n} x {k}] {:5.1} MB: gemv {:6.1} us ({:3.0} GB/s)",
            bytes as f64 / 1e6,
            gemv * 1e6,
            bytes as f64 / gemv / 1e9
        );
        for (ri, &m) in rows.iter().enumerate() {
            // The narrow f32 outputs of a batched pass are bf16 (the routers'
            // logits are read by the batched top-k as bf16).
            let c_dtype = if f32_out && n >= WIDE_N_MIN { DType::F32 } else { DType::BF16 };
            let a =
                Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, m * k), &[m, k])
                    .expect("a");
            let c = Tensor::zeros(&ctx, &[m, n], c_dtype).expect("c");
            let t = time(&|pass, w| {
                if bits == 4 {
                    gemm_skinny_q4_nt(&ctx, pass, &a, w, &c).unwrap()
                } else {
                    gemm_skinny_q8_nt(&ctx, pass, &a, w, &c).unwrap()
                }
            });
            total_rows[ri] += t * per_step as f64;
            line += &format!(
                " | m={m} {:6.1} us ({:3.0} GB/s)",
                t * 1e6,
                bytes as f64 / t / 1e9
            );
        }
        eprintln!("{line}");
    }
    let mut line = format!("per step: gemv {:.2} ms", total_gemv * 1e3);
    for (&m, t) in rows.iter().zip(&total_rows) {
        line += &format!(" | m={m} {:.2} ms", t * 1e3);
    }
    eprintln!("{line}");
}

/// The dense projections on the levels they share in a decode or verify
/// pass (`moe_ffn.rs`, `model.rs`): the router with the shared expert's
/// gate/up projection and gate; the shared expert's down projection beside
/// the expert gather (stood in for by an 18 MB 4-bit projection through the
/// `*_previous` test kernel, alone on its level as the reference); the
/// attention qkv projection with the indexer's; and, for what the separate
/// dispatches of a level cost, one GEMV over as many rows as the level's
/// matrices have together. Chains of barriered levels
/// over distinct weights in passes as long as a decode step; microseconds
/// per level at one row (the decode GEMVs) and at `LILY_DENSE_ROWS` rows
/// (the skinny GEMMs), best of eight passes after a warm-up.
#[test]
#[ignore = "timing only"]
fn dense_shared_levels_timing() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(75);
    let rows: Vec<usize> = std::env::var("LILY_DENSE_ROWS")
        .unwrap_or_else(|_| "3".into())
        .split(',')
        .map(|v| v.parse().expect("rows"))
        .collect();
    let sets = 24usize;
    let make = |bits: usize, n: usize, k: usize, salt: u32| -> Vec<QuantWeights> {
        (0..sets)
            .map(|s| timing_quant_bits(&ctx, n, k, bits, salt * 1000 + s as u32))
            .collect()
    };
    let gate = make(8, 512, 2560, 1);
    let gate_up = make(8, 1280, 2560, 2);
    let shared_gate = make(8, 1, 2560, 3);
    let down = make(8, 2560, 640, 4);
    let partner = make(4, 14336, 2560, 5);
    let qkv = make(8, 13312, 2560, 6);
    let idx = make(8, 640, 2560, 7);
    let m_max = *rows.iter().max().expect("rows").max(&1);
    let a = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, m_max * 2560), &[m_max * 2560])
        .expect("a");
    let a_down =
        Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, m_max * 640), &[m_max * 640])
            .expect("a");
    let out = |n: usize, dtype: DType| Tensor::zeros(&ctx, &[m_max * n], dtype).expect("c");
    let (c_gate_f32, c_gate) = (out(512, DType::F32), out(512, DType::BF16));
    let c_gate_up = out(1280, DType::BF16);
    let (c_sg_f32, c_sg) = (out(1, DType::F32), out(1, DType::BF16));
    let c_down = out(2560, DType::BF16);
    let c_partner = out(14336, DType::BF16);
    let c_qkv = out(13312, DType::BF16);
    let c_idx = out(640, DType::BF16);
    // One projection at m rows: the decode GEMV at m = 0, else the skinny
    // GEMM (bf16 out).
    let project = |pass: &ComputePass<'_>,
                   w: &QuantWeights,
                   a: &Tensor,
                   c_f32: Option<&Tensor>,
                   c: &Tensor,
                   m: usize| {
        let (n, k) = (w.out_features(), w.in_features());
        if m == 0 {
            let y = c_f32.unwrap_or(c).view(0, &[n]).expect("y");
            gemv_quant(&ctx, pass, w, &a.view(0, &[k]).expect("x"), &y).unwrap();
        } else {
            let am = a.view(0, &[m, k]).expect("a");
            let cm = c.view(0, &[m, n]).expect("c");
            gemm_skinny_q8_nt(&ctx, pass, &am, w, &cm).unwrap();
        }
    };
    let partner_dispatch = |pass: &ComputePass<'_>, w: &QuantWeights| {
        let am = a.view(0, &[1, 2560]).expect("a");
        let cm = c_partner.view(0, &[1, 14336]).expect("c");
        gemm_skinny_q4_nt_previous(&ctx, pass, &am, w, &cm).unwrap();
    };
    let time = |f: &dyn Fn(&ComputePass<'_>, usize)| -> f64 {
        let repeats = 512usize.div_ceil(sets);
        let mut best = f64::MAX;
        let start = std::time::Instant::now();
        let mut measured = 0;
        while measured < 8 {
            let pass = ctx.begin_concurrent().expect("pass");
            for _ in 0..repeats {
                for s in 0..sets {
                    f(&pass, s);
                    pass.level_barrier(&[]).expect("barrier");
                }
            }
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            if start.elapsed().as_millis() >= 300 {
                best =
                    best.min((t.gpu_end_secs - t.gpu_start_secs) / (repeats * sets) as f64);
                measured += 1;
            }
        }
        best * 1e6
    };
    let alone = time(&|pass, s| partner_dispatch(pass, &partner[s]));
    eprintln!("the 18 MB partner alone: {alone:.1} us per level");
    // What the levels' dispatches would take as one: the router level's
    // three matrices as one of their rows together, and the attention
    // level's two.
    let router_rows = make(8, 512 + 1280 + 1, 2560, 8);
    let attn_rows = make(8, 13312 + 640, 2560, 9);
    let c_rows = out(13312 + 640, DType::BF16);
    let router_one = time(&|pass, s| project(pass, &router_rows[s], &a, None, &c_rows, 0));
    let attn_one = time(&|pass, s| project(pass, &attn_rows[s], &a, None, &c_rows, 0));
    eprintln!(
        "as one decode GEMV of the same rows: router level {router_one:.1} us | attn level {attn_one:.1} us"
    );
    // The levels through `gemv_quant_sets` (one dispatch on Vulkan).
    let y = |c: &Tensor, n: usize| c.view(0, &[n]).expect("y");
    let x = a.view(0, &[2560]).expect("x");
    let router_sets = time(&|pass, s| {
        crate::kernels::quant::gemv_quant_sets(
            &ctx,
            pass,
            &x,
            &[
                (&gate[s], &y(&c_gate_f32, 512)),
                (&gate_up[s], &y(&c_gate_up, 1280)),
                (&shared_gate[s], &y(&c_sg_f32, 1)),
            ],
        )
        .unwrap()
    });
    let attn_sets = time(&|pass, s| {
        crate::kernels::quant::gemv_quant_sets(
            &ctx,
            pass,
            &x,
            &[(&qkv[s], &y(&c_qkv, 13312)), (&idx[s], &y(&c_idx, 640))],
        )
        .unwrap()
    });
    eprintln!(
        "through gemv_quant_sets: router level {router_sets:.1} us | attn level {attn_sets:.1} us"
    );
    for m in std::iter::once(0).chain(rows.iter().copied()) {
        let router = time(&|pass, s| {
            project(pass, &gate[s], &a, Some(&c_gate_f32), &c_gate, m);
            project(pass, &gate_up[s], &a, None, &c_gate_up, m);
            project(pass, &shared_gate[s], &a, Some(&c_sg_f32), &c_sg, m);
        });
        let gather = time(&|pass, s| {
            partner_dispatch(pass, &partner[s]);
            project(pass, &down[s], &a_down, None, &c_down, m);
        });
        let attn = time(&|pass, s| {
            project(pass, &qkv[s], &a, None, &c_qkv, m);
            project(pass, &idx[s], &a, None, &c_idx, m);
        });
        if m > 0 {
            // The router level as one multi-weight dispatch (Vulkan).
            let router_sets = time(&|pass, s| {
                let am = a.view(0, &[m, 2560]).expect("a");
                let c = |c: &Tensor, n: usize| c.view(0, &[m, n]).expect("c");
                q8_sets_one_dispatch(
                    &ctx,
                    pass,
                    &am,
                    &[
                        (&gate[s], &c(&c_gate, 512)),
                        (&gate_up[s], &c(&c_gate_up, 1280)),
                        (&shared_gate[s], &c(&c_sg, 1)),
                    ],
                )
                .unwrap()
            });
            eprintln!(
                "skinny m={m} as one multi-weight dispatch: router level {router_sets:.1} us"
            );
        }
        let label = if m == 0 { "decode GEMVs".to_string() } else { format!("skinny m={m}") };
        eprintln!(
            "{label}: router + shared gate/up + shared gate {router:.1} us | partner + shared down {gather:.1} us (+{:.1}) | attn qkv + indexer {attn:.1} us",
            gather - alone
        );
    }
}

/// One or two weight rows per lane in the Vulkan raw-code projection body
/// (`vulkan/quant_rows.glsl`), the choice `quant::rows_layout` makes by the
/// weight's size: every dense shape of the q4-xl checkpoint at one
/// activation row (the 4-bit GEMV kernels; for the 8-bit shapes the one-row
/// register-A GEMM, since their GEMV is another kernel) and, for the 8-bit
/// ones, at three, as chains of barriered levels over distinct weights in
/// passes as long as a decode step. Prints microseconds per level, best of
/// eight passes after a warm-up.
/// `LILY_DENSE_SHAPES` filters shapes by substring.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn quant_rows_layout_probe() {
    use crate::kernels::quant::{rows_grid, rows_layout};
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(73);
    let filter = std::env::var("LILY_DENSE_SHAPES").ok();
    for (what, bits, n, k) in [
        ("attn qkv", 8usize, 13312usize, 2560usize),
        ("attn o", 8, 2560, 6144),
        ("indexer qk", 8, 640, 2560),
        ("shared gate_up", 8, 1280, 2560),
        ("shared down", 8, 2560, 640),
        ("moe gate", 8, 512, 2560),
        ("ple key", 8, 10240, 2560),
        ("ple value", 8, 2560, 2560),
        ("gdn in", 4, 16480, 2560),
        ("gdn out", 4, 2560, 6144),
        ("lm head", 8, 248320, 2560),
    ] {
        if filter.as_ref().is_some_and(|f| !f.split(',').any(|p| what.contains(p))) {
            continue;
        }
        let bytes = n * k * bits / 8 + 2 * n * (k / GROUP_SIZE) * 2;
        let sets = (768usize << 20).div_ceil(bytes).clamp(3, 48);
        let weights: Vec<QuantWeights> = (0..sets)
            .map(|s| timing_quant_bits(&ctx, n, k, bits, s as u32 + 1))
            .collect();
        let (lanes_log2, shipped_rows) = rows_layout(&weights[0]);
        // Passes as long as a decode step, as in `dense_chain_timing`.
        let repeats = 512usize.div_ceil(sets);
        let a3 = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, 3 * k), &[3, k])
            .expect("a");
        let c3 = Tensor::zeros(&ctx, &[3, n], DType::BF16).expect("c");
        let mut line = format!(
            "{what:<14} q{bits} [{n} x {k}] {:5.1} MB, {} lanes per set, ships {shipped_rows}:",
            bytes as f64 / 1e6,
            1 << lanes_log2
        );
        for m in [1usize, 3] {
            if m == 3 && bits == 4 {
                continue;
            }
            for rows in [1usize, 2] {
                let name: &'static str = match (m, bits, rows) {
                    (1, 4, 1) => "gemv_q4_bf16",
                    (1, 4, _) => "gemv_q4_bf16_2row",
                    (1, _, 1) => "gemm_skinny_q8_bf16_reg_m1_r1",
                    (1, _, _) => "gemm_skinny_q8_bf16_reg_m1",
                    (_, _, 1) => "gemm_skinny_q8_bf16_reg_m3_r1",
                    _ => "gemm_skinny_q8_bf16_reg_m3",
                };
                let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1).expect("pipeline");
                let mut best = f64::MAX;
                let start = std::time::Instant::now();
                let mut measured = 0;
                while measured < 8 {
                    let pass = ctx.begin_concurrent().expect("pass");
                    for w in weights.iter().cycle().take(repeats * sets) {
                        pass.dispatch_at(
                            &pipeline,
                            &[
                                w.codes.binding(),
                                w.scales.binding(),
                                w.biases.binding(),
                                a3.binding(),
                                c3.binding(),
                            ],
                            &[
                                &u32_bytes(k),
                                &u32_bytes(n),
                                &u32_bytes(w.group_size),
                                &u32_bytes(lanes_log2),
                            ],
                            rows_grid(n, lanes_log2, rows),
                        )
                        .expect("dispatch");
                        pass.level_barrier(&[]).expect("barrier");
                    }
                    let done = pass.commit().expect("commit").wait_retain().expect("wait");
                    let t = done.timing().expect("timing");
                    if start.elapsed().as_millis() >= 300 {
                        best = best
                            .min((t.gpu_end_secs - t.gpu_start_secs) / (repeats * sets) as f64);
                        measured += 1;
                    }
                }
                line += &format!(" m={m} x{rows} {:.1} us", best * 1e6);
            }
        }
        eprintln!("{line}");
    }
}

/// A 4-bit register-A kernel by name with an explicit layout: `rr` weight
/// rows per simdgroup, `sg` simdgroups per threadgroup. For the probe below;
/// the name may be a shader that exists only in `LILY_VK_SHADER_DIR`.
#[cfg(target_os = "linux")]
fn dispatch_q4_reg_named(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    name: &'static str,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    (rr, sg): (usize, usize),
) -> Result<()> {
    let (_, k, n) = validate_q4(a, w, c)?;
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        reg_grid(n, sg, rr),
    )
}

/// The 4-bit register-A GEMM of the verify pass (the GDN projections at
/// three rows) against variants of itself, as the model's passes run them:
/// chains of barriered levels over distinct weights in passes as long as a
/// decode step, microseconds per level, best of eight passes after a warm-up.
/// Lines: the decode GEMV at one row (the bandwidth reference), the two-row
/// kernel the shipped one is pinned to (`*_previous`), the shipped route,
/// and each entry of `LILY_Q4X`, a comma-separated list of
/// `kernel:rows per simdgroup:simdgroups per threadgroup[:m]` (the kernel's
/// shader may exist only in `LILY_VK_SHADER_DIR`: a variant under work, no
/// rebuild). Each entry is first compared with the shipped route bit for bit
/// on hashed weights and wide-exponent activations (the count of differing
/// outputs is printed, not asserted: a stripped variant differs on purpose).
/// `LILY_DENSE_SHAPES` filters shapes by substring (`gdn in`, `gdn out`,
/// `draft head`); `LILY_DENSE_ROWS` is the row count of the shipped and
/// previous lines (default 3); `LILY_Q4X_PER_LEVEL` puts that many dispatches
/// on a level (default 1).
///
/// What it showed on 2026-11-11 (M5 Max, us per level at m = 3, the GDN
/// input [16480 x 2560] and output [2560 x 6144] projections; the decode
/// GEMV at one row 49.2 and 21.7): the shipped kernel 65.7 and 27.6 before
/// the one-store epilogue, 63.6 and 27.1 with it. With the products left out
/// 52.2 and 22.7, with constant codes in place of the weight loads 51.9 and
/// 23.8, with constant activations 59.3 and 25.1: the kernel is bound by its
/// instruction count (about 1080 per lane and block of four weight rows,
/// 480 of them the pinned products and sums, 256 the code conversions) and
/// by the loads in between, which a thread waits out one after another,
/// not by the weights' bandwidth. Two, four or eight weight rows per
/// simdgroup 64.3/27.9, 65.5/27.7, 76.5/39.2; one, two or four simdgroups
/// per threadgroup the same within 0.3; the code words as 16-byte loads in
/// an unrolled block 75.2/26.8 (the body outgrows what K = 2560's split
/// pass leaves of the instruction cache), as 8-byte loads in a half-rolled
/// one 67.1/27.3, loaded a word ahead of their use 72.3/33.0; 16 and 32
/// more live registers 68.2 and 72.6 on the input projection (residency),
/// nothing on the output's single generation of 640 subgroups. Four
/// dispatches on a level run at 55.4 and 23.7 each: what a level of its own
/// loses to its ramp and tail.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn q4_reg_variants_probe() {
    use crate::kernels::quant::gemv_quant;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(79);
    let filter = std::env::var("LILY_DENSE_SHAPES").ok();
    let m_ref: usize = std::env::var("LILY_DENSE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    // (kernel, rows per simdgroup, simdgroups per threadgroup, m)
    let variants: Vec<(&'static str, usize, usize, usize)> = std::env::var("LILY_Q4X")
        .unwrap_or_default()
        .split(',')
        .filter(|e| !e.is_empty())
        .map(|e| {
            let p: Vec<&str> = e.split(':').collect();
            let name: &'static str = Box::leak(p[0].to_string().into_boxed_str());
            let num = |i: usize, d: usize| p.get(i).map_or(d, |v| v.parse().expect("number"));
            (name, num(1, 4), num(2, 2), num(3, m_ref))
        })
        .collect();
    for (what, n, k, f32_out) in [
        ("gdn in", 16480usize, 2560usize, false),
        ("gdn out", 2560, 6144, false),
        ("draft head", 248320, 2560, true),
    ] {
        if filter.as_ref().is_some_and(|f| !f.split(',').any(|p| what.contains(p))) {
            continue;
        }
        let dtype = if f32_out { DType::F32 } else { DType::BF16 };
        // Bits against the shipped route.
        let wb = bits_quant(&ctx, n, k, 7);
        for &(name, rr, sg, m) in &variants {
            let a = bits_activations(&mut rng, m * k, k, true, false);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let got = Tensor::zeros(&ctx, &[m, n], dtype).expect("c");
            let want = Tensor::zeros(&ctx, &[m, n], dtype).expect("c");
            let pass = ctx.begin().expect("pass");
            gemm_skinny_q4_nt(&ctx, &pass, &ta, &wb, &want).expect("shipped");
            dispatch_q4_reg_named(&ctx, &pass, name, &ta, &wb, &got, (rr, sg))
                .expect("variant");
            pass.commit_wait().expect("commit");
            let (got, want) = (got.to_f32().expect("read"), want.to_f32().expect("read"));
            let differ =
                got.iter().zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
            eprintln!("{what} {name} rr={rr} sg={sg} m={m}: {differ} of {} outputs differ from the shipped route", got.len());
        }
        drop(wb);
        let bytes = n * k / 2 + 2 * n * (k / GROUP_SIZE) * 2;
        let sets = (768usize << 20).div_ceil(bytes).clamp(3, 48);
        let weights: Vec<QuantWeights> =
            (0..sets).map(|s| timing_quant(&ctx, n, k, s as u32 + 1)).collect();
        // Dispatches per level (`LILY_Q4X_PER_LEVEL`, default 1): more than
        // one shows what the kernel costs beside a copy of itself.
        let per_level: usize = std::env::var("LILY_Q4X_PER_LEVEL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let levels: usize = if f32_out { 32 } else { 512 };
        let repeats = levels.div_ceil(sets).max(1);
        let time = |f: &dyn Fn(&ComputePass<'_>, &QuantWeights)| -> f64 {
            let mut best = f64::MAX;
            let start = std::time::Instant::now();
            let mut measured = 0;
            while measured < 8 {
                let pass = ctx.begin_concurrent().expect("pass");
                for _ in 0..repeats {
                    for (i, w) in weights.iter().enumerate() {
                        f(&pass, w);
                        if (i + 1) % per_level == 0 {
                            pass.level_barrier(&[]).expect("barrier");
                        }
                    }
                }
                let done = pass.commit().expect("commit").wait_retain().expect("wait");
                let t = done.timing().expect("timing");
                if start.elapsed().as_millis() >= 300 {
                    best = best
                        .min((t.gpu_end_secs - t.gpu_start_secs) / (repeats * sets) as f64);
                    measured += 1;
                }
            }
            best * 1e6
        };
        let report = |label: &str, us: f64| {
            eprintln!(
                "{what} [{n} x {k}] {:5.1} MB {label}: {us:6.1} us ({:3.0} GB/s)",
                bytes as f64 / 1e6,
                bytes as f64 / us / 1e3
            );
        };
        let x = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, k), &[k]).expect("x");
        let y = Tensor::zeros(&ctx, &[n], dtype).expect("y");
        report("decode gemv m=1", time(&|pass, w| gemv_quant(&ctx, pass, w, &x, &y).unwrap()));
        let rows_of = |m: usize| {
            let a = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut StdRng::seed_from_u64(5), m * k), &[m, k])
                .expect("a");
            let c = Tensor::zeros(&ctx, &[m, n], dtype).expect("c");
            (a, c)
        };
        let (a, c) = rows_of(m_ref);
        report(
            &format!("previous m={m_ref}"),
            time(&|pass, w| gemm_skinny_q4_nt_previous(&ctx, pass, &a, w, &c).unwrap()),
        );
        report(
            &format!("shipped m={m_ref}"),
            time(&|pass, w| gemm_skinny_q4_nt(&ctx, pass, &a, w, &c).unwrap()),
        );
        for &(name, rr, sg, m) in &variants {
            let (a, c) = rows_of(m);
            report(
                &format!("{name} rr={rr} sg={sg} m={m}"),
                time(&|pass, w| {
                    dispatch_q4_reg_named(&ctx, pass, name, &a, w, &c, (rr, sg)).unwrap()
                }),
            );
        }
    }
}

/// The dense levels of a verify pass in the encodings they can take
/// (`gdn_batched`, `attn_batched_theta`, `prefill_moe`'s router and gather
/// levels; `LILY_DENSE_ROWS` rows, default 3; `LILY_DENSE_SHAPES` filters by
/// `gdn`, `attn`, `down`, `router`): chains of levels over distinct weights
/// in passes as long as a decode step, microseconds per level, best of eight
/// passes after a warm-up. What `skinny::stack_as_slices` and
/// `gemm_skinny_q8_nt_sets` choose on Vulkan was measured here.
///
/// - The GDN input stack (4-bit, 16480 rows: qkv, z, a, b): one fused GEMM
///   and a level that splits its output (Metal's encoding); the fused GEMM
///   alone (what the split level costs); the four slices as dispatches of
///   one level, in stack order and smallest first (Vulkan's).
/// - The attention input (8-bit: the q, k, v stack of 13312 rows and the
///   indexer's 640): the fused stack beside the indexer's GEMM and a split
///   level; the four projections as one multi-weight dispatch into their
///   own outputs; as four dispatches, the indexer's first or last.
/// - The shared expert's down projection beside a stand-in for the expert
///   gate/up kernel, at one weight row per lane and at two, dispatched
///   before the partner and after it.
/// - The router level (8-bit: router, shared gate/up, shared gate): one
///   multi-weight dispatch; three dispatches.
///
/// Every encoding of a level writes the same bits (a fused stack and its
/// slices, a weight alone and among others in one dispatch:
/// `gemm_skinny_fused_stack_matches_per_slice_bits`,
/// `stack_slices_match_the_fused_stack_bits`,
/// `gemm_skinny_q8_sets_match_separate_bits`,
/// `gemm_skinny_q8_reg_rows_do_not_depend_on_the_dispatch`).
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing only"]
fn verify_levels_timing() {
    use crate::kernels::elementwise::split_cols_bf16;
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(85);
    let m: usize = std::env::var("LILY_DENSE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let sets = 24usize;
    let k = 2560usize;
    let a = Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, m * k), &[m, k]).expect("a");
    let out = |n: usize| Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("c");
    let time = |f: &dyn Fn(&ComputePass<'_>, usize)| -> f64 {
        let repeats = 512usize.div_ceil(sets);
        let mut best = f64::MAX;
        let start = std::time::Instant::now();
        let mut measured = 0;
        while measured < 8 {
            let pass = ctx.begin_concurrent().expect("pass");
            for _ in 0..repeats {
                for s in 0..sets {
                    f(&pass, s);
                }
            }
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            if start.elapsed().as_millis() >= 300 {
                best =
                    best.min((t.gpu_end_secs - t.gpu_start_secs) / (repeats * sets) as f64);
                measured += 1;
            }
        }
        best * 1e6
    };
    let barrier = |pass: &ComputePass<'_>| pass.level_barrier(&[]).expect("barrier");
    let filter = std::env::var("LILY_DENSE_SHAPES").ok();
    let wanted = |what: &str| filter.as_ref().is_none_or(|f| f.split(',').any(|p| what.contains(p)));

    if wanted("gdn") {
        let widths = [10240usize, 6144, 48, 48];
        let n: usize = widths.iter().sum();
        let stacks: Vec<QuantWeights> =
            (0..sets).map(|s| timing_quant(&ctx, n, k, s as u32 + 1)).collect();
        let slices: Vec<Vec<QuantWeights>> = stacks
            .iter()
            .map(|w| {
                let mut start = 0;
                widths
                    .iter()
                    .map(|&rows| {
                        let v = quant_view_rows(w, start, rows);
                        start += rows;
                        v
                    })
                    .collect()
            })
            .collect();
        let c_stack = out(n);
        let outs: Vec<Tensor> = widths.iter().map(|&w| out(w)).collect();
        let out_refs: Vec<&Tensor> = outs.iter().collect();
        let fused_split = time(&|pass, s| {
            gemm_skinny_q4_nt(&ctx, pass, &a, &stacks[s], &c_stack).unwrap();
            barrier(pass);
            split_cols_bf16(&ctx, pass, &c_stack, &out_refs).unwrap();
            barrier(pass);
        });
        let fused = time(&|pass, s| {
            gemm_skinny_q4_nt(&ctx, pass, &a, &stacks[s], &c_stack).unwrap();
            barrier(pass);
        });
        let in_order = time(&|pass, s| {
            for (w, c) in slices[s].iter().zip(&outs) {
                gemm_skinny_q4_nt(&ctx, pass, &a, w, c).unwrap();
            }
            barrier(pass);
        });
        let small_first = time(&|pass, s| {
            for (w, c) in slices[s].iter().zip(&outs).rev() {
                gemm_skinny_q4_nt(&ctx, pass, &a, w, c).unwrap();
            }
            barrier(pass);
        });
        eprintln!(
            "gdn in m={m}: fused + split level {fused_split:.1} us | fused alone {fused:.1} | slices in stack order {in_order:.1} | slices smallest first {small_first:.1}"
        );
    }

    if wanted("attn") {
        let widths = [12288usize, 512, 512];
        let n: usize = widths.iter().sum();
        let stacks: Vec<QuantWeights> =
            (0..sets).map(|s| timing_quant_bits(&ctx, n, k, 8, s as u32 + 1)).collect();
        let idx: Vec<QuantWeights> =
            (0..sets).map(|s| timing_quant_bits(&ctx, 640, k, 8, s as u32 + 101)).collect();
        let slices: Vec<Vec<QuantWeights>> = stacks
            .iter()
            .map(|w| {
                let mut start = 0;
                widths
                    .iter()
                    .map(|&rows| {
                        let v = quant_view_rows(w, start, rows);
                        start += rows;
                        v
                    })
                    .collect()
            })
            .collect();
        let c_stack = out(n);
        let c_idx = out(640);
        let outs: Vec<Tensor> = widths.iter().map(|&w| out(w)).collect();
        let out_refs: Vec<&Tensor> = outs.iter().collect();
        let current = time(&|pass, s| {
            gemm_skinny_q8_nt(&ctx, pass, &a, &stacks[s], &c_stack).unwrap();
            gemm_skinny_q8_nt(&ctx, pass, &a, &idx[s], &c_idx).unwrap();
            barrier(pass);
            split_cols_bf16(&ctx, pass, &c_stack, &out_refs).unwrap();
            barrier(pass);
        });
        let stack_alone = time(&|pass, s| {
            gemm_skinny_q8_nt(&ctx, pass, &a, &stacks[s], &c_stack).unwrap();
            barrier(pass);
        });
        let one = time(&|pass, s| {
            let sl = &slices[s];
            gemm_skinny_q8_nt_sets_fused(
                &ctx,
                pass,
                &a,
                &[(&sl[0], &outs[0]), (&sl[1], &outs[1]), (&sl[2], &outs[2]), (&idx[s], &c_idx)],
            )
            .unwrap();
            barrier(pass);
        });
        let one_small_first = time(&|pass, s| {
            let sl = &slices[s];
            gemm_skinny_q8_nt_sets_fused(
                &ctx,
                pass,
                &a,
                &[(&idx[s], &c_idx), (&sl[2], &outs[2]), (&sl[1], &outs[1]), (&sl[0], &outs[0])],
            )
            .unwrap();
            barrier(pass);
        });
        let four = time(&|pass, s| {
            let sl = &slices[s];
            gemm_skinny_q8_nt(&ctx, pass, &a, &idx[s], &c_idx).unwrap();
            for i in (0..3).rev() {
                gemm_skinny_q8_nt(&ctx, pass, &a, &sl[i], &outs[i]).unwrap();
            }
            barrier(pass);
        });
        let four_idx_last = time(&|pass, s| {
            let sl = &slices[s];
            for i in (0..3).rev() {
                gemm_skinny_q8_nt(&ctx, pass, &a, &sl[i], &outs[i]).unwrap();
            }
            gemm_skinny_q8_nt(&ctx, pass, &a, &idx[s], &c_idx).unwrap();
            barrier(pass);
        });
        eprintln!(
            "attn in m={m}: fused stack + indexer + split level {current:.1} us | the stack alone {stack_alone:.1} | one multi-weight dispatch {one:.1} (smallest first {one_small_first:.1}) | four dispatches, smallest first {four:.1}, the indexer last {four_idx_last:.1}"
        );
    }

    if wanted("down") {
        // The shared expert's down projection (8-bit, 2560 x 640) on the
        // level of the expert gate/up kernel, which a 4-bit GEMM of 18 MB
        // stands in for: at one weight row per lane (the route) and at two.
        use crate::kernels::quant::{rows_grid, rows_layout};
        let kd = 640usize;
        let down: Vec<QuantWeights> =
            (0..sets).map(|s| timing_quant_bits(&ctx, 2560, kd, 8, s as u32 + 201)).collect();
        let partner: Vec<QuantWeights> =
            (0..sets).map(|s| timing_quant(&ctx, 14336, k, s as u32 + 301)).collect();
        let a_down =
            Tensor::from_f32_as_bf16(&ctx, &random_vec(&mut rng, m * kd), &[m, kd]).expect("a");
        let (c_down, c_partner) = (out(2560), out(14336));
        let (lanes_log2, _) = rows_layout(&down[0]);
        let by_rows = |pass: &ComputePass<'_>, w: &QuantWeights, rows: usize| {
            let name = if rows == 1 { Q8_REG_FNS_R1[m - 1] } else { Q8_REG_FNS[m - 1] };
            let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1).expect("pipeline");
            pass.dispatch_at(
                &pipeline,
                &[
                    w.codes.binding(),
                    w.scales.binding(),
                    w.biases.binding(),
                    a_down.binding(),
                    c_down.binding(),
                ],
                &[
                    &u32_bytes(kd),
                    &u32_bytes(2560),
                    &u32_bytes(w.group_size),
                    &u32_bytes(lanes_log2),
                ],
                rows_grid(2560, lanes_log2, rows),
            )
            .expect("dispatch");
        };
        let alone = time(&|pass, s| {
            gemm_skinny_q4_nt(&ctx, pass, &a, &partner[s], &c_partner).unwrap();
            barrier(pass);
        });
        let mut line = format!("gather level m={m}: the partner alone {alone:.1} us");
        for rows in [1usize, 2] {
            let first = time(&|pass, s| {
                by_rows(pass, &down[s], rows);
                gemm_skinny_q4_nt(&ctx, pass, &a, &partner[s], &c_partner).unwrap();
                barrier(pass);
            });
            let last = time(&|pass, s| {
                gemm_skinny_q4_nt(&ctx, pass, &a, &partner[s], &c_partner).unwrap();
                by_rows(pass, &down[s], rows);
                barrier(pass);
            });
            let lone = time(&|pass, s| {
                by_rows(pass, &down[s], rows);
                barrier(pass);
            });
            line += &format!(
                " | down at {rows} per lane: before it {first:.1}, after it {last:.1}, alone {lone:.1}"
            );
        }
        eprintln!("{line}");
    }

    if wanted("router") {
        let make = |n: usize, salt: u32| -> Vec<QuantWeights> {
            (0..sets)
                .map(|s| timing_quant_bits(&ctx, n, k, 8, salt * 1000 + s as u32))
                .collect()
        };
        let (gate, gate_up, shared_gate) = (make(512, 1), make(1280, 2), make(1, 3));
        let (c_gate, c_gate_up, c_sg) = (out(512), out(1280), out(1));
        let one = time(&|pass, s| {
            gemm_skinny_q8_nt_sets_fused(
                &ctx,
                pass,
                &a,
                &[(&gate[s], &c_gate), (&gate_up[s], &c_gate_up), (&shared_gate[s], &c_sg)],
            )
            .unwrap();
            barrier(pass);
        });
        let three = time(&|pass, s| {
            gemm_skinny_q8_nt(&ctx, pass, &a, &gate[s], &c_gate).unwrap();
            gemm_skinny_q8_nt(&ctx, pass, &a, &gate_up[s], &c_gate_up).unwrap();
            gemm_skinny_q8_nt(&ctx, pass, &a, &shared_gate[s], &c_sg).unwrap();
            barrier(pass);
        });
        let three_small_first = time(&|pass, s| {
            gemm_skinny_q8_nt(&ctx, pass, &a, &shared_gate[s], &c_sg).unwrap();
            gemm_skinny_q8_nt(&ctx, pass, &a, &gate[s], &c_gate).unwrap();
            gemm_skinny_q8_nt(&ctx, pass, &a, &gate_up[s], &c_gate_up).unwrap();
            barrier(pass);
        });
        eprintln!(
            "router level m={m}: one multi-weight dispatch {one:.1} us | three dispatches {three:.1} (smallest first {three_small_first:.1})"
        );
    }
}

/// A stack projected as `moe_ffn::project_stack_or_slices` encodes it (on
/// Vulkan the slices, each into its own output: `stack_as_slices`) is, bit
/// for bit, the fused GEMM split into those outputs, at the model's stack
/// shapes: the 4-bit GDN input (qkv, z, a, b) and the 8-bit attention input
/// (q, k, v), at the row counts of a draft pass, a verify pass and the widest
/// register-A batch, with activations of wide exponents.
#[test]
fn stack_slices_match_the_fused_stack_bits() {
    use crate::kernels::elementwise::split_cols_bf16;
    use crate::moe_ffn::{project_stack_fused, project_stack_or_slices};
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(87);
    let k = 2560usize;
    let dequant = Tensor::zeros(&ctx, &[1], DType::BF16).expect("scratch");
    for (bits, widths) in
        [(4usize, vec![10240usize, 6144, 48, 48]), (8, vec![12288, 512, 512])]
    {
        let n: usize = widths.iter().sum();
        let stack = if bits == 4 {
            bits_quant(&ctx, n, k, 11)
        } else {
            timing_quant_bits(&ctx, n, k, 8, 11)
        };
        let mut start = 0;
        let slices: Vec<QuantWeights> = widths
            .iter()
            .map(|&rows| {
                let v = quant_view_rows(&stack, start, rows);
                start += rows;
                v
            })
            .collect();
        for m in [1usize, 3, 8] {
            let a = bits_activations(&mut rng, m * k, k, true, false);
            let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[m, k]).expect("a");
            let outputs = || -> Vec<Tensor> {
                widths
                    .iter()
                    .map(|&w| Tensor::zeros(&ctx, &[m, w], DType::BF16).expect("c"))
                    .collect()
            };
            let (got, want) = (outputs(), outputs());
            let stack_c = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("stack c");
            let pass = ctx.begin().expect("pass");
            // The fused GEMM and the split, spelled out.
            let fused = project_stack_fused(&ctx, &pass, &ta, &stack, &stack_c, &widths)
                .expect("fused")
                .expect("the stack fuses at this shape");
            split_cols_bf16(&ctx, &pass, &fused, &want.iter().collect::<Vec<_>>())
                .expect("split");
            // The model's encoding.
            let stack_c2 = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("stack c");
            if bits == 4 {
                project_stack_or_slices(
                    &ctx,
                    &pass,
                    &ta,
                    &stack,
                    &stack_c2,
                    [
                        (&slices[0], &got[0]),
                        (&slices[1], &got[1]),
                        (&slices[2], &got[2]),
                        (&slices[3], &got[3]),
                    ],
                    &dequant,
                )
                .expect("stack or slices");
            } else {
                project_stack_or_slices(
                    &ctx,
                    &pass,
                    &ta,
                    &stack,
                    &stack_c2,
                    [(&slices[0], &got[0]), (&slices[1], &got[1]), (&slices[2], &got[2])],
                    &dequant,
                )
                .expect("stack or slices");
            }
            pass.commit_wait().expect("commit");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_bits_eq(
                    &g.to_f32().expect("read"),
                    &w.to_f32().expect("read"),
                    &format!("q{bits} stack m={m} slice {i}"),
                );
            }
        }
    }
}
