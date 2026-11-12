const TEST_SOURCE: &str = concat!(
    include_str!("../../../src/kernels/metal/elementwise.metal"),
    "\n",
    include_str!("../../metal/elementwise_test.metal")
);

fn mul_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
) -> Result<()> {
    binary_op_source(ctx, pass, "mul_bf16", DType::BF16, a, b, out, TEST_SOURCE)
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::*;
use crate::cpu_ref;

#[test]
fn elementwise_ops_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(3);
    let n = 1000;
    let a: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
    let (ra, rb) = (cpu_ref::round_bf16(&a), cpu_ref::round_bf16(&b));

    let ta = Tensor::from_f32_as_bf16(&ctx, &a, &[n]).expect("a");
    let tb = Tensor::from_f32_as_bf16(&ctx, &b, &[n]).expect("b");

    let pair = |f: &dyn Fn(f32, f32) -> f32| -> Vec<f32> {
        ra.iter().zip(&rb).map(|(&x, &y)| f(x, y)).collect()
    };
    let cases = [
        ("add", pair(&|x, y| x + y)),
        ("mul", pair(&|x, y| x * y)),
        ("silu_mul", pair(&|x, y| cpu_ref::silu(x) * y)),
    ];
    for (name, expected) in cases {
        let out = Tensor::zeros(&ctx, &[n], DType::BF16).expect("out");
        let pass = ctx.begin().expect("pass");
        match name {
            "add" => add_bf16(&ctx, &pass, &ta, &tb, &out).expect("add"),
            "mul" => mul_bf16(&ctx, &pass, &ta, &tb, &out).expect("mul"),
            _ => silu_mul_bf16(&ctx, &pass, &ta, &tb, &out).expect("silu_mul"),
        }
        pass.commit_wait().expect("commit");
        cpu_ref::assert_close(&out.to_f32().expect("read"), &expected, 2e-2, 2e-2);
    }
}

#[test]
fn argmax_f32_stages_match_first_max_reference() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(8);
    let n = 4097;
    let mut x: Vec<f32> = (0..n).map(|_| rng.gen_range(-8.0f32..8.0)).collect();
    x[513] = 9.5;
    x[3501] = 9.5;
    let expected = 513u32;

    let tx = Tensor::from_f32(&ctx, &x, &[n]).expect("x");
    let partials =
        Tensor::zeros(&ctx, &[ARGMAX_GROUPS * 2], DType::F32).expect("partials");
    let out = Tensor::zeros(&ctx, &[1], DType::U32).expect("out");

    let partial_pass = ctx.begin().expect("partial pass");
    argmax_f32_partial(&ctx, &partial_pass, &tx, &partials)
        .expect("argmax_f32_partial");
    partial_pass.commit_wait().expect("partial commit");

    let chunk = n.div_ceil(ARGMAX_GROUPS);
    let bytes = partials.raw_bytes();
    for group in 0..ARGMAX_GROUPS {
        let start = group * chunk;
        let end = ((group + 1) * chunk).min(n);
        let (expected_value, expected_index) = x[start..end].iter().enumerate().fold(
            (f32::NEG_INFINITY, 0usize),
            |(best_value, best_index), (offset, &value)| {
                let index = start + offset;
                if value > best_value {
                    (value, index)
                } else {
                    (best_value, best_index)
                }
            },
        );
        let base = group * 8;
        let actual_value = f32::from_ne_bytes(
            bytes[base..base + 4].try_into().expect("partial value"),
        );
        let actual_index = u32::from_ne_bytes(
            bytes[base + 4..base + 8].try_into().expect("partial index"),
        );
        assert_eq!(actual_value, expected_value, "group {group} value");
        assert_eq!(actual_index as usize, expected_index, "group {group} index");
    }

    let final_pass = ctx.begin().expect("final pass");
    argmax_f32_final(&ctx, &final_pass, &partials, &out).expect("argmax_f32_final");
    final_pass.commit_wait().expect("final commit");
    assert_eq!(out.to_u32().expect("read staged result")[0], expected);

    out.zero_fill();
    let combined_pass = ctx.begin().expect("combined pass");
    argmax_f32(&ctx, &combined_pass, &tx, &partials, &out).expect("argmax_f32");
    combined_pass.commit_wait().expect("combined commit");
    assert_eq!(out.to_u32().expect("read combined result")[0], expected);
}

#[test]
fn copy_words_copies_any_dtype_and_rejects_mismatch() {
    let ctx = MetalContext::new().expect("metal context");
    let src: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5 - 7.0).collect();
    let t_src = Tensor::from_f32(&ctx, &src, &[10, 100]).expect("src");
    let t_dst = Tensor::zeros(&ctx, &[1000], DType::F32).expect("dst");
    let pass = ctx.begin().expect("pass");
    copy_words(&ctx, &pass, &t_src, &t_dst).expect("copy");
    pass.commit_wait().expect("commit");
    assert_eq!(t_dst.to_f32().expect("read"), src);

    let ids: Vec<u32> = (0..6).collect();
    let t_ids = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[6], DType::U32)
        .expect("ids");
    let t_out = Tensor::zeros(&ctx, &[6], DType::U32).expect("out");
    let pass = ctx.begin().expect("pass");
    copy_words(&ctx, &pass, &t_ids, &t_out).expect("copy u32");
    pass.commit_wait().expect("commit");
    assert_eq!(t_out.to_u32().expect("read"), ids);

    let short = Tensor::zeros(&ctx, &[999], DType::F32).expect("short");
    let pass = ctx.begin().expect("pass");
    assert!(copy_words(&ctx, &pass, &t_src, &short).is_err());
}

/// Raw unified-memory bandwidth from the GPU: a 4 GiB coalesced read and a
/// 4 GiB copy, best of several passes. The number to hold the weight-streaming
/// kernels against (the decode step's effective GB/s is bytes moved / GPU ms).
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn memory_bandwidth_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let bytes = 4usize << 30;
    let u4 = bytes / 16;
    let per_thread = 16usize;
    let threads = u4 / per_thread;
    let src = Tensor::zeros(&ctx, &[bytes / 4], DType::U32).expect("src");
    let dst = Tensor::zeros(&ctx, &[bytes / 4], DType::U32).expect("dst");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let grid = Grid::Threads { grid: (threads, 1, 1), threadgroup: (256, 1, 1) };
    let per = u32_bytes(per_thread);
    {
        let fill =
            ctx.pipeline("bw_fill_u4", TEST_SOURCE, MslVersion::V3_1).expect("fill");
        let pass = ctx.begin().expect("pass");
        pass.dispatch_at(
            &fill,
            &[src.binding()],
            &[],
            Grid::Threads { grid: (u4, 1, 1), threadgroup: (256, 1, 1) },
        )
        .expect("dispatch");
        pass.commit_wait().expect("fill");
    }
    let read = ctx.pipeline("bw_read_u4", TEST_SOURCE, MslVersion::V3_1).expect("read");
    let copy = ctx.pipeline("bw_copy_u4", TEST_SOURCE, MslVersion::V3_1).expect("copy");
    for (name, kernel, traffic) in [("read", &read, bytes), ("copy", &copy, 2 * bytes)]
    {
        let mut times = Vec::new();
        for _ in 0..5 {
            let pass = ctx.begin().expect("pass");
            let second: &Tensor = if name == "read" { &out } else { &dst };
            pass.dispatch_at(
                kernel,
                &[src.binding(), second.binding()],
                &[&per[..]],
                grid,
            )
            .expect("dispatch");
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            times.push(t.gpu_end_secs - t.gpu_start_secs);
        }
        times.sort_by(f64::total_cmp);
        let best = times[0];
        let median = times[times.len() / 2];
        eprintln!(
            "bandwidth {name}: {:.0} GB/s best, {:.0} GB/s median ({} bytes of traffic, {:.1} ms best)",
            traffic as f64 / best / 1e9,
            traffic as f64 / median / 1e9,
            traffic,
            best * 1e3
        );
    }
}

/// `silu_mul_rows_bf16` over a `[m, 2n]` gate|up stack equals `silu_mul_bf16`
/// on the split halves bit for bit (the small-m MoE path's shared expert
/// reads the fused stack GEMM output in place instead of splitting it).
#[test]
fn silu_mul_rows_matches_split_silu_mul_bitwise() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(97);
    for (m, n) in [(1usize, 640usize), (3, 640), (8, 96), (16, 1280)] {
        let gu: Vec<f32> =
            (0..m * 2 * n).map(|_| rng.gen_range(-4.0f32..4.0)).collect();
        let gu_ref = &gu;
        let (gate, up): (Vec<f32>, Vec<f32>) = (0..m)
            .flat_map(|r| {
                (0..n).map(move |c| (gu_ref[r * 2 * n + c], gu_ref[r * 2 * n + n + c]))
            })
            .unzip();
        let t_gu = Tensor::from_f32_as_bf16(&ctx, &gu, &[m, 2 * n]).expect("gu");
        let t_gate = Tensor::from_f32_as_bf16(&ctx, &gate, &[m, n]).expect("gate");
        let t_up = Tensor::from_f32_as_bf16(&ctx, &up, &[m, n]).expect("up");
        let rows_out = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("rows out");
        let split_out = Tensor::zeros(&ctx, &[m, n], DType::BF16).expect("split out");
        let pass = ctx.begin().expect("pass");
        silu_mul_rows_bf16(&ctx, &pass, &t_gu, &rows_out).expect("rows");
        silu_mul_bf16(&ctx, &pass, &t_gate, &t_up, &split_out).expect("split");
        pass.commit_wait().expect("commit");
        let bits = |t: &Tensor| -> Vec<u32> {
            t.to_f32().expect("read").iter().map(|x| x.to_bits()).collect()
        };
        assert_eq!(bits(&rows_out), bits(&split_out), "m={m} n={n}");
    }
}

/// The textbook form in f64, so the reference carries no f32 cancellation of
/// its own in the tail.
fn gelu_tanh(x: f32) -> f32 {
    let x = x as f64;
    let u = (2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
    (0.5 * x * (1.0 + u.tanh())) as f32
}

/// erf by Simpson integration of the Gaussian in f64 (2 000 steps, error far
/// below 1e-9): deliberately not the closed-form approximation the kernel
/// uses, so a coefficient slip there is caught.
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let steps = 2000;
    let h = x / steps as f64;
    let f = |t: f64| (-t * t).exp();
    let mut sum = f(0.0) + f(x);
    for i in 1..steps {
        sum += f(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    (sum * h / 3.0 * 2.0 / std::f64::consts::PI.sqrt()) as f32
}

fn gelu_erf(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (1.0 + erf((x / std::f64::consts::SQRT_2) as f32) as f64)) as f32
}

#[test]
fn gelu_forms_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(7);
    let n = 1500;
    let x: Vec<f32> = (0..n).map(|_| rng.gen_range(-5.0f32..5.0)).collect();
    let rx = cpu_ref::round_bf16(&x);

    for (form, f) in [(Gelu::Tanh, gelu_tanh as fn(f32) -> f32), (Gelu::Erf, gelu_erf)]
    {
        let tx = Tensor::from_f32_as_bf16(&ctx, &x, &[n]).expect("x");
        let pass = ctx.begin().expect("pass");
        gelu_bf16(&ctx, &pass, &tx, form).expect("gelu");
        pass.commit_wait().expect("commit");
        let expected: Vec<f32> = rx.iter().map(|&v| f(v)).collect();
        let got = tx.to_f32().expect("read");
        // Output rounded to bf16 on values up to 5.
        cpu_ref::assert_close(&got, &expected, 2e-2, 1e-2);
        // Where the tail is tiny the two forms differ by nearly a factor of
        // two (gelu_erf(-4) = -1.27e-4, gelu_tanh(-4) = -7.0e-5), which the
        // absolute tolerance above cannot see; a relative check there pins
        // which form ran. bf16 carries 0.4 %, the erf approximation 1.5e-7
        // absolute on values of 1e-4 and above, so 5 % is generous.
        let mut tails = 0;
        for (i, &v) in rx.iter().enumerate() {
            if (-4.2..-3.5).contains(&v) {
                tails += 1;
                let (g, e) = (got[i], expected[i]);
                assert!(
                    (g - e).abs() <= 0.05 * e.abs(),
                    "{form:?} tail at {i}: {g} vs {e}"
                );
            }
        }
        assert!(tails > 10, "the input must reach into the tail");
    }
}

/// The row gather on both of its paths: eight elements per thread for
/// 16-byte-aligned rows (the MoE input gather's 2 560-wide rows, and a view
/// at an aligned offset) and one element per thread otherwise (a 6-wide
/// table, an unaligned output view).
#[test]
fn gather_rows_bf16_both_paths_match_cpu() {
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(33);
    for (rows, h, out_offset) in
        [(64usize, 2560usize, 0usize), (37, 2560, 8), (20, 6, 0), (20, 16, 2)]
    {
        let table: Vec<f32> =
            (0..rows * h).map(|_| rng.gen_range(-2.0f32..2.0)).collect();
        let table = cpu_ref::round_bf16(&table);
        let m = 50usize;
        let ids: Vec<u32> = (0..m).map(|_| rng.gen_range(0..rows as u32)).collect();
        let t_table =
            Tensor::from_f32_as_bf16(&ctx, &table, &[rows, h]).expect("table");
        let t_ids =
            Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[m], DType::U32)
                .expect("ids");
        let backing =
            Tensor::zeros(&ctx, &[out_offset + m * h], DType::BF16).expect("out");
        let out = backing.view(out_offset, &[m, h]).expect("view");
        let pass = ctx.begin().expect("pass");
        gather_rows_bf16(&ctx, &pass, &t_table, &t_ids, &out).expect("gather");
        pass.commit_wait().expect("commit");
        let got = out.to_f32().expect("read");
        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(
                &got[i * h..(i + 1) * h],
                &table[id as usize * h..(id as usize + 1) * h],
                "rows {rows} h {h} offset {out_offset}: row {i}"
            );
        }
    }
}

/// Per-dispatch GPU time of the row gather at the MoE input gather's chunk
/// shape (4 096 tokens x 10 experts rows of 2 560) on the profile
/// transport, eight elements per thread against one.
#[test]
#[ignore = "timing only"]
fn gather_rows_bf16_timing() {
    let ctx = MetalContext::new_with_profile(true).expect("metal context");
    let mut rng = StdRng::seed_from_u64(35);
    let (rows, h, m) = (4096usize, 2560usize, 40960usize);
    let table = Tensor::zeros(&ctx, &[rows, h], DType::BF16).expect("table");
    let ids: Vec<u32> = (0..m).map(|_| rng.gen_range(0..rows as u32)).collect();
    let t_ids = Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[m], DType::U32)
        .expect("ids");
    let out = Tensor::zeros(&ctx, &[m, h], DType::BF16).expect("out");
    crate::metal::profile::take();
    for _ in 0..8 {
        for name in ["gather_rows_bf16_x8", "gather_rows_bf16"] {
            let pipeline =
                ctx.pipeline(name, SOURCE, MslVersion::V3_1).expect("pipeline");
            let (n, w) =
                if name.ends_with("_x8") { (m * h / 8, h / 8) } else { (m * h, h) };
            let pass = ctx.begin().expect("pass");
            pass.dispatch_at(
                &pipeline,
                &[table.binding(), t_ids.binding(), out.binding()],
                &[&crate::kernels::u32_bytes(w)],
                Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
            )
            .expect("dispatch");
            pass.commit_wait().expect("commit");
        }
    }
    let passes = crate::metal::profile::take();
    for name in ["gather_rows_bf16_x8", "gather_rows_bf16"] {
        let mut ms: Vec<f64> = passes
            .iter()
            .flat_map(|p| p.kernels.iter())
            .filter(|s| s.name == name)
            .map(|s| s.gpu_secs * 1e3)
            .collect();
        ms.sort_by(|a, b| a.total_cmp(b));
        let n = ms.len();
        eprintln!(
            "{name}: min {:.2} ms, median {:.2} ms over {n} dispatches ({:.0} GB/s at the minimum)",
            ms[0],
            ms[n / 2],
            2.0 * (m * h * 2) as f64 / ms[0] / 1e6
        );
    }
}

/// Achievable bandwidth as a function of the bytes one dependency level
/// reads: the decode step is a chain of about 450 levels, most reading one
/// weight matrix of 2 to 40 MB, and a level cannot start before the previous
/// one drains. For each level size this streams `K` distinct regions of a
/// 4 GiB buffer, one dispatch per level with a barrier between (the decode
/// shape), then the same dispatches with no barriers (what overlapping
/// would give back), then two dispatches per level of half the size. Prints
/// GB/s and the microseconds per level; run with `--ignored --nocapture`.
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn level_size_bandwidth_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let bytes = 4usize << 30;
    let src = Tensor::zeros(&ctx, &[bytes / 4], DType::U32).expect("src");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    {
        let fill =
            ctx.pipeline("bw_fill_u4", TEST_SOURCE, MslVersion::V3_1).expect("fill");
        let pass = ctx.begin().expect("pass");
        pass.dispatch_at(
            &fill,
            &[src.binding()],
            &[],
            Grid::Threads { grid: (bytes / 16, 1, 1), threadgroup: (256, 1, 1) },
        )
        .expect("dispatch");
        pass.commit_wait().expect("fill");
    }
    let read = ctx.pipeline("bw_read_u4", TEST_SOURCE, MslVersion::V3_1).expect("read");
    let per_thread: usize = std::env::var("LILY_LEVEL_PROBE_PER_THREAD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let per = u32_bytes(per_thread);
    // Regions far apart so no two levels share cache lines in the system cache.
    let stride = 96usize << 20;
    let region =
        |i: usize, size: usize| ((i * (stride + size)) % (bytes - size)) / 16 * 16;
    for &mb in &[1usize, 2, 4, 8, 16, 32, 64, 128, 512] {
        let size = mb << 20;
        let levels = (1usize << 30) / size;
        let levels = levels.clamp(2, 128);
        let dispatch = |pass: &ComputePass<'_>, offset: usize, size: usize| {
            let view = src.view(offset / 4, &[size / 4]).expect("view");
            let threads = size / 16 / per_thread;
            pass.dispatch_at(
                &read,
                &[view.binding(), out.binding()],
                &[&per[..]],
                Grid::Threads { grid: (threads, 1, 1), threadgroup: (256, 1, 1) },
            )
            .expect("dispatch");
        };
        let mut results = Vec::new();
        for (name, barriers, split) in [
            ("barriered", true, 1usize),
            ("concurrent", false, 1),
            ("2 per level", true, 2),
        ] {
            let mut times = Vec::new();
            for _ in 0..5 {
                let pass = ctx.begin_concurrent().expect("pass");
                for i in 0..levels {
                    let base = region(i, size);
                    for j in 0..split {
                        dispatch(&pass, base + j * (size / split), size / split);
                    }
                    if barriers {
                        pass.level_barrier(&[&out]).expect("barrier");
                    }
                }
                let done = pass.commit().expect("commit").wait_retain().expect("wait");
                let t = done.timing().expect("timing");
                times.push(t.gpu_end_secs - t.gpu_start_secs);
            }
            times.sort_by(f64::total_cmp);
            let best = times[0];
            results.push(format!(
                "{name} {:.0} GB/s ({:.1} us/level)",
                (levels * size) as f64 / best / 1e9,
                best / levels as f64 * 1e6
            ));
        }
        eprintln!("level {mb:>3} MB x {levels:>3}: {}", results.join(" | "));
    }
}

/// What a tiny dependency level costs on this backend, by grid and
/// threadgroup shape and by what its threads do (the probe kernels of
/// `tests/vulkan/levels_test.comp`): chains of `LEVELS` dispatches in one
/// concurrent pass, with a level barrier after each and with none, then the
/// same tiny level between 4 MB streaming levels (what the decode step has:
/// the cost that matters is what the tiny level adds to its neighbours).
/// Prints microseconds per level, best of five passes. Run with `--ignored
/// --nocapture`; `LILY_LEVEL_PROBE_LEVELS` sets the chain length.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn tiny_level_floor_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let levels: usize = std::env::var("LILY_LEVEL_PROBE_LEVELS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let words = 1usize << 16;
    let src = Tensor::zeros(&ctx, &[words], DType::U32).expect("src");
    let dst = Tensor::zeros(&ctx, &[words], DType::U32).expect("dst");
    let four = [u32_bytes(2560), u32_bytes(0), u32_bytes(0), u32_bytes(0)];
    // (kernel, threads of the grid, threadgroup, params)
    let cases: &[(&'static str, usize, usize, usize)] = &[
        ("lv_nop", 1, 1, 0),
        ("lv_nop", 32, 32, 0),
        ("lv_nop", 256, 256, 0),
        ("lv_nop", 1024, 1024, 0),
        ("lv_nop", 2560, 256, 0),
        ("lv_nop", 2560, 1024, 0),
        ("lv_nop", 10240, 256, 0),
        ("lv_nop", 10240, 1024, 0),
        ("lv_guard", 2560, 256, 0),
        ("lv_guard", 10240, 256, 0),
        ("lv_copy1", 1, 1, 0),
        ("lv_copy1", 256, 256, 0),
        ("lv_copy1", 2560, 64, 0),
        ("lv_copy1", 2560, 256, 0),
        ("lv_copy1", 2560, 1024, 0),
        ("lv_copy1", 10240, 256, 0),
        ("lv_copy1", 10240, 1024, 0),
        ("lv_copy4", 640, 32, 0),
        ("lv_copy4", 640, 128, 0),
        ("lv_copy4", 640, 256, 0),
        ("lv_copy4", 640, 640, 0),
        ("lv_copy4", 2560, 256, 0),
        ("lv_copy4", 2560, 1024, 0),
        ("lv_copy1_params", 2560, 256, 4),
        ("lv_copy1_params", 10240, 256, 4),
    ];
    let dispatch = |pass: &ComputePass<'_>, &(name, threads, tg, params): &(&'static str, usize, usize, usize)| {
        let kernel = ctx.pipeline(name, "", MslVersion::V3_1).expect("kernel");
        let params: Vec<&[u8]> = four[..params].iter().map(|p| &p[..]).collect();
        pass.dispatch_at(
            &kernel,
            &[src.binding(), dst.binding()],
            &params,
            Grid::Threads { grid: (threads, 1, 1), threadgroup: (tg, 1, 1) },
        )
        .expect("dispatch");
    };
    let span = |pass: ComputePass<'_>| {
        let done = pass.commit().expect("commit").wait_retain().expect("wait");
        let t = done.timing().expect("timing");
        t.gpu_end_secs - t.gpu_start_secs
    };
    for case in cases {
        let mut best = [f64::INFINITY; 2];
        for round in 0..6 {
            for (mode, barriers) in [true, false].into_iter().enumerate() {
                let pass = ctx.begin_concurrent().expect("pass");
                for _ in 0..levels {
                    dispatch(&pass, case);
                    if barriers {
                        pass.level_barrier(&[&dst]).expect("barrier");
                    }
                }
                let secs = span(pass);
                if round > 0 {
                    best[mode] = best[mode].min(secs / levels as f64);
                }
            }
        }
        let (name, threads, tg, _) = case;
        eprintln!(
            "{name:<16} {threads:>6} threads in {:>3} x {tg:>4}: {:>6.2} us per level barriered, {:>6.2} with no barriers",
            threads.div_ceil(*tg),
            best[0] * 1e6,
            best[1] * 1e6
        );
    }

    // The tiny level between streaming levels.
    let bytes = 1usize << 30;
    let big = Tensor::zeros(&ctx, &[bytes / 4], DType::U32).expect("big");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let read = ctx.pipeline("bw_read_u4", TEST_SOURCE, MslVersion::V3_1).expect("read");
    let per = u32_bytes(8);
    let (size, stream_levels) = (4usize << 20, 64usize);
    let stream = |pass: &ComputePass<'_>, i: usize| {
        let view = big.view(i * (bytes / stream_levels) / 4, &[size / 4]).expect("view");
        pass.dispatch_at(
            &read,
            &[view.binding(), out.binding()],
            &[&per[..]],
            Grid::Threads { grid: (size / 16 / 8, 1, 1), threadgroup: (256, 1, 1) },
        )
        .expect("dispatch");
        pass.level_barrier(&[&out]).expect("barrier");
    };
    let between: &[Option<(&'static str, usize, usize, usize)>] = &[
        None,
        Some(("lv_nop", 1, 1, 0)),
        Some(("lv_copy1", 2560, 256, 0)),
        Some(("lv_copy1", 2560, 1024, 0)),
        Some(("lv_copy1", 10240, 256, 0)),
        Some(("lv_copy1", 10240, 1024, 0)),
        Some(("lv_copy4", 640, 128, 0)),
        Some(("lv_copy4", 2560, 256, 0)),
        Some(("lv_copy1_params", 10240, 256, 4)),
    ];
    let mut best = vec![f64::INFINITY; between.len()];
    for round in 0..7 {
        for k in 0..between.len() {
            let which = (round + k) % between.len();
            let pass = ctx.begin_concurrent().expect("pass");
            for i in 0..stream_levels {
                stream(&pass, i);
                if let Some(case) = &between[which] {
                    dispatch(&pass, case);
                    pass.level_barrier(&[&dst]).expect("barrier");
                }
            }
            let secs = span(pass);
            if round > 0 {
                best[which] = best[which].min(secs / stream_levels as f64);
            }
        }
    }
    for (case, secs) in between.iter().zip(&best) {
        let what = match case {
            None => "nothing".to_string(),
            Some((name, threads, tg, _)) => format!("{name} {threads} threads / {tg}"),
        };
        eprintln!(
            "4 MB stream level + {what}: {:.2} us per pair ({:+.2} us)",
            secs * 1e6,
            (secs - best[0]) * 1e6
        );
    }
}

/// What bounds a streaming level on this backend: the ALU rate (`lv_alu`,
/// multiply-adds only), then 4 MB levels read by `lv_read_w<W>` in every
/// shape of threads x rounds x W loads issued together, with and without
/// ALU work after each round. Chains of 64 levels over distinct regions, a
/// barrier between; microseconds per level, best of five.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn level_shape_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let bytes = 1usize << 30;
    let big = Tensor::zeros(&ctx, &[bytes / 4], DType::U32).expect("big");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let levels = 64usize;
    let span = |pass: ComputePass<'_>| {
        let done = pass.commit().expect("commit").wait_retain().expect("wait");
        let t = done.timing().expect("timing");
        t.gpu_end_secs - t.gpu_start_secs
    };
    let alu = ctx.pipeline("lv_alu", "", MslVersion::V3_1).expect("alu");
    for threads in [256usize, 10240, 324 * 128, 2560 * 32] {
        for n in [0usize, 16, 64, 256] {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let pass = ctx.begin_concurrent().expect("pass");
                for _ in 0..levels {
                    pass.dispatch_at(
                        &alu,
                        &[big.binding(), out.binding()],
                        &[&u32_bytes(n)],
                        Grid::Threads { grid: (threads, 1, 1), threadgroup: (256, 1, 1) },
                    )
                    .expect("dispatch");
                    pass.level_barrier(&[&out]).expect("barrier");
                }
                best = best.min(span(pass) / levels as f64);
            }
            eprintln!(
                "lv_alu {threads} threads x {n:>3} rounds of 8 fma: {:.2} us per level",
                best * 1e6
            );
        }
    }
    if std::env::var_os("LILY_LEVEL_SHAPE_ALU_ONLY").is_some() {
        return;
    }
    let size = 4usize << 20;
    // (W, rounds, alu rounds after each round's loads, alu rounds between
    // their issue and their use, 64 more live registers)
    let shapes: &[(usize, usize, usize, usize, usize)] = &[
        (1, 1, 0, 0, 0),
        (1, 8, 0, 0, 0),
        (4, 1, 0, 0, 0),
        (4, 4, 0, 0, 0),
        (8, 2, 0, 0, 0),
        (4, 1, 0, 0, 1),
        (4, 4, 0, 0, 1),
        (4, 1, 16, 0, 0),
        (4, 1, 0, 16, 0),
        (4, 1, 64, 0, 0),
        (4, 1, 0, 64, 0),
        (4, 4, 16, 0, 0),
        (4, 4, 0, 16, 0),
        (4, 4, 16, 0, 1),
        (8, 2, 32, 0, 0),
        (8, 2, 0, 32, 0),
        (1, 4, 16, 0, 0),
        (1, 4, 0, 16, 0),
    ];
    let mut best = vec![f64::INFINITY; shapes.len()];
    for round in 0..6 {
        for k in 0..shapes.len() {
            let which = (round + k) % shapes.len();
            let (w, rounds, alu, pre, fat) = shapes[which];
            let name: &'static str = match w {
                1 => "lv_read_w1",
                2 => "lv_read_w2",
                4 => "lv_read_w4",
                _ => "lv_read_w8",
            };
            let kernel = ctx.pipeline(name, "", MslVersion::V3_1).expect("read");
            let threads = size / 16 / w / rounds;
            let pass = ctx.begin_concurrent().expect("pass");
            for i in 0..levels {
                let view =
                    big.view(i * (bytes / levels) / 4, &[size / 4]).expect("view");
                pass.dispatch_at(
                    &kernel,
                    &[view.binding(), out.binding()],
                    &[&u32_bytes(rounds), &u32_bytes(alu), &u32_bytes(pre), &u32_bytes(fat)],
                    Grid::Threads { grid: (threads, 1, 1), threadgroup: (256, 1, 1) },
                )
                .expect("dispatch");
                pass.level_barrier(&[&out]).expect("barrier");
            }
            let secs = span(pass);
            if round > 0 {
                best[which] = best[which].min(secs / levels as f64);
            }
        }
    }
    for ((w, rounds, alu, pre, fat), secs) in shapes.iter().zip(&best) {
        let threads = size / 16 / w / rounds;
        eprintln!(
            "4 MB level, {threads:>6} threads x {rounds} rounds x {w} loads, alu {alu:>2} after / {pre:>2} before use, fat {fat}: {:>6.2} us ({:.0} GB/s; {:.2} M multiply-adds)",
            secs * 1e6,
            size as f64 / secs / 1e9,
            (threads * rounds * (alu + pre) * 8) as f64 / 1e6
        );
    }
}

/// The decode step's small kernels at the model's shapes (H 2560, 4 streams,
/// 48 GDN heads, conv over 10240 channels), each as decode runs it: alone in
/// a chain of barriered levels, and between 4 MB streaming levels (what it
/// adds to its neighbours is its cost in a step). Microseconds per level,
/// best of several passes. Run with `--ignored --nocapture`.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn small_kernels_chain_timing() {
    use crate::kernels::gdn::{conv_window_rollback, conv1d_step};
    use crate::kernels::hc::{hc_inject_bf16, rmsnorm_grouped_bf16};
    use crate::kernels::norm::rmsnorm_bf16;
    use crate::kernels::spec::copy_row;
    let ctx = MetalContext::new().expect("metal context");
    let (h, g, c, heads) = (2560usize, 4usize, 10240usize, 48usize);
    let bf = |shape: &[usize]| Tensor::zeros(&ctx, shape, DType::BF16).expect("bf16");
    let per_state = heads * 128 * 128;
    let mid = Tensor::zeros(&ctx, &[2, per_state], DType::F32).expect("mid");
    let st = Tensor::zeros(&ctx, &[per_state], DType::F32).expect("state");
    let (hyper1, branch1, inj1) = (bf(&[1, g * h]), bf(&[1, h]), bf(&[1, g]));
    let (hyper3, branch3, inj3) = (bf(&[3, g * h]), bf(&[3, h]), bf(&[3, g]));
    let (win, win2, x1, x3) = (bf(&[c, 3]), bf(&[c, 3]), bf(&[c]), bf(&[3, c]));
    let (conv_w, conv_out) = (bf(&[4, c]), bf(&[c]));
    let (hidden, hidden2) = (bf(&[g * h]), bf(&[g * h]));
    let (gu, act) = (bf(&[3, 2 * 1024]), bf(&[3, 1024]));
    let (split_src, s0, s1, s2) =
        (bf(&[3, c]), bf(&[3, 4096]), bf(&[3, 4096]), bf(&[3, 2048]));
    let (norm_x, norm_w, norm_out) = (bf(&[3, h]), bf(&[h]), bf(&[3, h]));
    let (gn_w, gn_out) = (bf(&[g * h]), bf(&[3, g * h]));
    type Run<'a> = Box<dyn Fn(&ComputePass<'_>) + 'a>;
    let cases: Vec<(&str, Run<'_>)> = vec![
        (
            "hc_inject_bf16 m=1",
            Box::new(|p| hc_inject_bf16(&ctx, p, &hyper1, &branch1, &inj1, h, g).unwrap()),
        ),
        (
            "hc_inject_bf16 m=3",
            Box::new(|p| hc_inject_bf16(&ctx, p, &hyper3, &branch3, &inj3, h, g).unwrap()),
        ),
        (
            "conv1d_step_bf16",
            Box::new(|p| conv1d_step(&ctx, p, &win, &x1, &conv_w, &conv_out).unwrap()),
        ),
        (
            "conv_window_rollback_bf16",
            Box::new(|p| conv_window_rollback(&ctx, p, &win, &x3, &win2, 2).unwrap()),
        ),
        (
            "copy_row_u32 3.1 MB state",
            Box::new(|p| copy_row(&ctx, p, &mid, crate::kernels::Arg::Const(1), &st).unwrap()),
        ),
        (
            "copy_u32 hyper row",
            Box::new(|p| copy_words(&ctx, p, &hidden, &hidden2).unwrap()),
        ),
        (
            "silu_mul_gu_rows_bf16 m=3",
            Box::new(|p| silu_mul_rows_bf16(&ctx, p, &gu, &act).unwrap()),
        ),
        (
            "split_cols_bf16 m=3",
            Box::new(|p| split_cols_bf16(&ctx, p, &split_src, &[&s0, &s1, &s2]).unwrap()),
        ),
        (
            "rmsnorm_bf16 m=3",
            Box::new(|p| rmsnorm_bf16(&ctx, p, &norm_x, &norm_w, &norm_out, 1e-6, 1.0).unwrap()),
        ),
        (
            "rmsnorm_grouped_bf16 m=3",
            Box::new(|p| {
                rmsnorm_grouped_bf16(&ctx, p, &hyper3, &gn_w, &gn_out, h, g, 1e-6, 1.0).unwrap()
            }),
        ),
    ];
    let span = |pass: ComputePass<'_>| {
        let done = pass.commit().expect("commit").wait_retain().expect("wait");
        let t = done.timing().expect("timing");
        t.gpu_end_secs - t.gpu_start_secs
    };
    let bytes = 1usize << 30;
    let big = Tensor::zeros(&ctx, &[bytes / 4], DType::U32).expect("big");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let read = ctx.pipeline("bw_read_u4", TEST_SOURCE, MslVersion::V3_1).expect("read");
    let per = u32_bytes(8);
    let (size, levels) = (4usize << 20, 64usize);
    let stream = |pass: &ComputePass<'_>, i: usize| {
        let view = big.view(i * (bytes / levels) / 4, &[size / 4]).expect("view");
        pass.dispatch_at(
            &read,
            &[view.binding(), out.binding()],
            &[&per[..]],
            Grid::Threads { grid: (size / 16 / 8, 1, 1), threadgroup: (256, 1, 1) },
        )
        .expect("dispatch");
        pass.level_barrier(&[&out]).expect("barrier");
    };
    let mut alone = vec![f64::INFINITY; cases.len()];
    let mut between = vec![f64::INFINITY; cases.len()];
    let mut base = f64::INFINITY;
    for round in 0..6 {
        let pass = ctx.begin_concurrent().expect("pass");
        for i in 0..levels {
            stream(&pass, i);
        }
        let secs = span(pass);
        if round > 0 {
            base = base.min(secs / levels as f64);
        }
        for k in 0..cases.len() {
            let which = (round + k) % cases.len();
            let pass = ctx.begin_concurrent().expect("pass");
            for _ in 0..4 * levels {
                (cases[which].1)(&pass);
                pass.level_barrier(&[]).expect("barrier");
            }
            let a = span(pass) / (4 * levels) as f64;
            let pass = ctx.begin_concurrent().expect("pass");
            for i in 0..levels {
                stream(&pass, i);
                (cases[which].1)(&pass);
                pass.level_barrier(&[]).expect("barrier");
            }
            let b = span(pass) / levels as f64;
            if round > 0 {
                alone[which] = alone[which].min(a);
                between[which] = between[which].min(b);
            }
        }
    }
    eprintln!("4 MB stream level alone: {:.2} us", base * 1e6);
    for ((name, _), (a, b)) in cases.iter().zip(alone.iter().zip(&between)) {
        eprintln!(
            "{name:<28} {:>6.2} us per level alone, {:>+6.2} us between stream levels",
            a * 1e6,
            (b - base) * 1e6
        );
    }
}

/// What a thread's dependent load rounds and a threadgroup's barriers cost
/// (`lv_rounds`): chains of barriered levels of 48 threadgroups of 128
/// threads (the GDN step's shape) and of 3 of 256 (a three-row norm's), over
/// a cache-resident region and a DRAM-sized one. Microseconds per level.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn load_round_and_barrier_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let big = Tensor::zeros(&ctx, &[1usize << 28], DType::F32).expect("big");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let kernel = ctx.pipeline("lv_rounds", "", MslVersion::V3_1).expect("kernel");
    let levels = 128usize;
    // (threadgroups, threadgroup, rounds, words of the region, barrier pairs, wide)
    let mut cases: Vec<(usize, usize, usize, usize, usize, usize)> = Vec::new();
    for (tgs, tg) in [(48usize, 128usize), (3, 256), (1, 128)] {
        for (rounds, words, pairs, wide) in [
            (0usize, 1usize << 12, 0usize, 0usize),
            (8, 1 << 12, 0, 0),
            (32, 1 << 12, 0, 0),
            (32, 1 << 12, 0, 1),
            (8, 1 << 26, 0, 0),
            (32, 1 << 26, 0, 0),
            (32, 1 << 26, 0, 1),
            (0, 1 << 12, 1, 0),
            (0, 1 << 12, 3, 0),
            (0, 1 << 12, 8, 0),
        ] {
            cases.push((tgs, tg, rounds, words, pairs, wide));
        }
    }
    let mut best = vec![f64::INFINITY; cases.len()];
    for round in 0..5 {
        for k in 0..cases.len() {
            let which = (round + k) % cases.len();
            let (tgs, tg, rounds, words, pairs, wide) = cases[which];
            let pass = ctx.begin_concurrent().expect("pass");
            for i in 0..levels {
                // A different part of the large region per level.
                let at = if words > (1 << 20) { (i * 1_000_003) % (1 << 26) } else { 0 };
                let view = big.view(at, &[words]).expect("view");
                pass.dispatch_at(
                    &kernel,
                    &[view.binding(), out.binding()],
                    &[
                        &u32_bytes(rounds),
                        &u32_bytes(words - 1),
                        &u32_bytes(pairs),
                        &u32_bytes(wide),
                    ],
                    Grid::Threadgroups { groups: (tgs, 1, 1), threadgroup: (tg, 1, 1) },
                )
                .expect("dispatch");
                pass.level_barrier(&[&out]).expect("barrier");
            }
            let done = pass.commit().expect("commit").wait_retain().expect("wait");
            let t = done.timing().expect("timing");
            if round > 0 {
                best[which] =
                    best[which].min((t.gpu_end_secs - t.gpu_start_secs) / levels as f64);
            }
        }
    }
    for ((tgs, tg, rounds, words, pairs, wide), secs) in cases.iter().zip(&best) {
        eprintln!(
            "{tgs:>2} x {tg:>3} threads, {rounds:>2} load rounds{} over {:>6} KB, {pairs} barrier pairs: {:>6.2} us",
            if *wide != 0 { " (8 per round)" } else { "" },
            words * 4 / 1024,
            secs * 1e6
        );
    }
}

/// Which association order the hardware's subgroup sum and its 4-lane
/// clustered sum use, by comparing them bit for bit with candidate orders
/// on random values of mixed magnitude (`lv_reduce`). Run with `--ignored
/// --nocapture`.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "probe; run with --ignored --nocapture"]
fn subgroup_sum_order_probe() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let ctx = MetalContext::new().expect("metal context");
    let mut rng = StdRng::seed_from_u64(77);
    let rows = 256usize;
    let x: Vec<f32> = (0..rows * 32)
        .map(|i| {
            // Some rows leave their last lanes zero, as a short GEMV row does.
            if (i / 32) % 2 == 1 && i % 32 >= 20 {
                0.0
            } else {
                rng.gen_range(-1.0f32..1.0) * 2f32.powi(rng.gen_range(-12..12))
            }
        })
        .collect();
    let t_x = Tensor::from_f32(&ctx, &x, &[rows * 32]).expect("x");
    let out = Tensor::zeros(&ctx, &[rows * 16], DType::F32).expect("out");
    let kernel = ctx.pipeline("lv_reduce", "", MslVersion::V3_1).expect("kernel");
    let pass = ctx.begin().expect("pass");
    pass.dispatch_at(
        &kernel,
        &[t_x.binding(), out.binding()],
        &[],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (32, 1, 1) },
    )
    .expect("dispatch");
    pass.commit_wait().expect("commit");
    let got = out.to_f32().expect("read");
    fn tree(v: &[f32]) -> f32 {
        if v.len() == 1 { v[0] } else { tree(&v[..v.len() / 2]) + tree(&v[v.len() / 2..]) }
    }
    // Lanes i and i + n/2 first (a butterfly), then recursively.
    fn butterfly(v: &[f32]) -> f32 {
        if v.len() == 1 {
            return v[0];
        }
        let h = v.len() / 2;
        let folded: Vec<f32> = (0..h).map(|i| v[i] + v[i + h]).collect();
        butterfly(&folded)
    }
    let candidates: [(&str, fn(&[f32]) -> f32); 4] = [
        ("ascending", |v| v.iter().fold(0.0, |a, b| a + b)),
        ("descending", |v| v.iter().rev().fold(0.0, |a, b| a + b)),
        ("adjacent pairs tree", tree),
        ("butterfly (i, i + n/2)", butterfly),
    ];
    for (name, f) in candidates {
        let full = (0..rows)
            .filter(|&r| f(&x[r * 32..r * 32 + 32]).to_bits() == got[r * 16].to_bits())
            .count();
        let quad = (0..rows * 8)
            .filter(|&q| {
                f(&x[q * 4..q * 4 + 4]).to_bits() == got[q / 8 * 16 + 1 + q % 8].to_bits()
            })
            .count();
        eprintln!(
            "{name}: subgroup sum equal in {full} of {rows} rows, quad sum in {quad} of {}",
            rows * 8
        );
    }
    // The subgroup sum from the quad sums, as an adjacent-pairs tree.
    let from_quads = (0..rows)
        .filter(|&r| tree(&got[r * 16 + 1..r * 16 + 9]).to_bits() == got[r * 16].to_bits())
        .count();
    eprintln!("tree of the eight quad sums equals the subgroup sum in {from_quads} of {rows} rows");
}

/// What an instruction costs a thread (`lv_op_*`, eight dependent operations
/// of a kind per round): nanoseconds per operation over the loop alone, with
/// 320 subgroups (eight a core) and with 1 280. Run with `--ignored
/// --nocapture`.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn instruction_cost_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let src = Tensor::zeros(&ctx, &[1024], DType::F32).expect("src");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let kinds: [&'static str; 28] = [
        "lv_op_loop",
        "lv_op_xor2",
        "lv_op_andx",
        "lv_op_shrx",
        "lv_op_bfex",
        "lv_op_orx",
        "lv_op_addx",
        "lv_op_cvt",
        "lv_op_mul",
        "lv_op_unpack",
        "lv_op_unpackm",
        "lv_op_fma",
        "lv_op_add",
        "lv_op_u2f",
        "lv_op_f2u",
        "lv_op_bfe",
        "lv_op_imul",
        "lv_op_sel",
        "lv_op_exp",
        "lv_op_rcp",
        "lv_op_rne",
        "lv_op_bcast",
        "lv_op_quad",
        "lv_op_red",
        "lv_op_lload",
        "lv_op_lstore",
        "lv_op_dload",
        "lv_op_branch",
    ];
    let levels = 64usize;
    for subgroups in [320usize, 1280] {
        let mut per_round = Vec::new();
        for name in kinds {
            let kernel = ctx.pipeline(name, "", MslVersion::V3_1).expect("kernel");
            let mut t = [f64::INFINITY; 2];
            for _ in 0..4 {
                for (k, rounds) in [32usize, 160].into_iter().enumerate() {
                    let pass = ctx.begin_concurrent().expect("pass");
                    for _ in 0..levels {
                        pass.dispatch_at(
                            &kernel,
                            &[src.binding(), out.binding()],
                            &[&u32_bytes(rounds)],
                            Grid::Threadgroups {
                                groups: (subgroups / 8, 1, 1),
                                threadgroup: (256, 1, 1),
                            },
                        )
                        .expect("dispatch");
                        pass.level_barrier(&[&out]).expect("barrier");
                    }
                    let done =
                        pass.commit().expect("commit").wait_retain().expect("wait");
                    let timing = done.timing().expect("timing");
                    t[k] = t[k]
                        .min((timing.gpu_end_secs - timing.gpu_start_secs) / levels as f64);
                }
            }
            per_round.push((t[1] - t[0]) / 128.0 * 1e9);
        }
        for (name, ns) in kinds.iter().zip(&per_round) {
            eprintln!(
                "{subgroups:>4} subgroups, {name:<14} {ns:>7.1} ns per round, {:>6.2} ns per operation over the loop",
                (ns - per_round[0]) / 8.0
            );
        }
    }
}

/// Whether what a thread holds in registers changes what its instructions
/// cost (`lv_fat<N>`: the Q8 unpack-and-dot loop with N more registers live
/// across it), by threadgroup size: nanoseconds per unpacked word.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn register_footprint_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let src = Tensor::zeros(&ctx, &[1024], DType::F32).expect("src");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let levels = 64usize;
    for (tgs, tg) in [(40usize, 256usize), (160, 256), (40, 640), (80, 640), (640, 32), (2560, 32)] {
        for name in ["lv_fat0", "lv_fat32", "lv_fat64", "lv_fat96"] {
            let kernel = ctx.pipeline(name, "", MslVersion::V3_1).expect("kernel");
            let mut t = [f64::INFINITY; 2];
            for _ in 0..4 {
                for (k, rounds) in [8usize, 40].into_iter().enumerate() {
                    let pass = ctx.begin_concurrent().expect("pass");
                    for _ in 0..levels {
                        pass.dispatch_at(
                            &kernel,
                            &[src.binding(), out.binding()],
                            &[&u32_bytes(rounds)],
                            Grid::Threadgroups { groups: (tgs, 1, 1), threadgroup: (tg, 1, 1) },
                        )
                        .expect("dispatch");
                        pass.level_barrier(&[&out]).expect("barrier");
                    }
                    let done =
                        pass.commit().expect("commit").wait_retain().expect("wait");
                    let timing = done.timing().expect("timing");
                    t[k] = t[k]
                        .min((timing.gpu_end_secs - timing.gpu_start_secs) / levels as f64);
                }
            }
            eprintln!(
                "{tgs:>4} x {tg:>3} threads ({:>4} subgroups), {name:<8}: {:>6.1} ns per word, {:>5.2} us at 8 rounds",
                tgs * tg / 32,
                (t[1] - t[0]) / 32.0 / 8.0 * 1e9,
                t[0] * 1e6
            );
        }
    }
}

/// Whether a shader's threadgroup memory or barrier changes how fast its
/// subgroups run (`lv_tg_*`), by threadgroup size at 320 subgroups:
/// nanoseconds per unpacked word of the loop that follows.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "timing probe; run with --ignored --nocapture"]
fn threadgroup_mode_probe() {
    let ctx = MetalContext::new().expect("metal context");
    let src = Tensor::zeros(&ctx, &[1024], DType::F32).expect("src");
    let out = Tensor::zeros(&ctx, &[1024], DType::U32).expect("out");
    let levels = 64usize;
    for total in [320usize, 1280] {
        for tg in [32usize, 128, 256, 640] {
            for name in ["lv_tg_plain", "lv_tg_shared", "lv_tg_barrier"] {
                let kernel = ctx.pipeline(name, "", MslVersion::V3_1).expect("kernel");
                let tgs = total * 32 / tg;
                let mut t = [f64::INFINITY; 2];
                for _ in 0..4 {
                    for (k, rounds) in [8usize, 40].into_iter().enumerate() {
                        let pass = ctx.begin_concurrent().expect("pass");
                        for _ in 0..levels {
                            pass.dispatch_at(
                                &kernel,
                                &[src.binding(), out.binding()],
                                &[&u32_bytes(rounds)],
                                Grid::Threadgroups {
                                    groups: (tgs, 1, 1),
                                    threadgroup: (tg, 1, 1),
                                },
                            )
                            .expect("dispatch");
                            pass.level_barrier(&[&out]).expect("barrier");
                        }
                        let done =
                            pass.commit().expect("commit").wait_retain().expect("wait");
                        let timing = done.timing().expect("timing");
                        t[k] = t[k].min(
                            (timing.gpu_end_secs - timing.gpu_start_secs) / levels as f64,
                        );
                    }
                }
                eprintln!(
                    "{tgs:>4} x {tg:>3} threads, {name:<14}: {:>6.1} ns per word, {:>5.2} us at 8 rounds",
                    (t[1] - t[0]) / 32.0 / 8.0 * 1e9,
                    t[0] * 1e6
                );
            }
        }
    }
}
