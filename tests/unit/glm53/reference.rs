use super::*;

#[test]
fn sinkhorn_comb_is_doubly_stochastic() {
    let n = 4;
    let mix: Vec<f32> = (0..24).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.3).collect();
    let scale = [0.7, 1.3, 2.0];
    let base: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.05).collect();
    let (pre, post, c) = hc_split_sinkhorn(&mix, &scale, &base, n, 20, 1e-6);
    for i in 0..n {
        assert!(pre[i] > 0.0 && pre[i] < 1.0 + 1e-5);
        assert!(post[i] > 0.0 && post[i] < 2.0);
        let row: f32 = (0..n).map(|s| c[s + i * n]).sum();
        let col: f32 = (0..n).map(|d| c[i + d * n]).sum();
        assert!((col - 1.0).abs() < 1e-4, "column {i} sums to {col}");
        assert!((row - 1.0).abs() < 1e-2, "row {i} sums to {row}");
    }
}

#[test]
fn hc_post_with_identity_comb_adds_block() {
    let n = 4;
    let h: Vec<f32> = (0..n * 3).map(|i| i as f32).collect();
    let block = [1.0, 2.0, 3.0];
    let mut comb = vec![0.0; n * n];
    for i in 0..n {
        comb[i + i * n] = 1.0;
    }
    let post = [1.0, 0.5, 0.0, 2.0];
    let out = hc_post(&h, &block, &post, &comb, n);
    for dst in 0..n {
        for d in 0..3 {
            assert_eq!(out[dst * 3 + d], h[dst * 3 + d] + block[d] * post[dst]);
        }
    }
}

#[test]
fn route_picks_biased_top_k_and_normalizes() {
    let logits = [0.0f32, 3.0, -1.0, 1.0, 2.0];
    let bias = [0.0f32, 0.0, 5.0, 0.0, 0.0];
    let (sel, w) = route(&logits, &bias, 2, 2.5);
    assert_eq!(sel, [2, 1]);
    let p = |l: f32| sigmoid(l);
    let s = p(-1.0) + p(3.0);
    assert!((w[0] - p(-1.0) / s * 2.5).abs() < 1e-6);
    assert!((w.iter().sum::<f32>() - 2.5).abs() < 1e-5);
}

#[test]
fn swiglu_clamps_gate_above_and_up_both_sides() {
    assert_eq!(swiglu(20.0, 30.0, 10.0), silu(10.0) * 10.0);
    assert_eq!(swiglu(-20.0, -30.0, 10.0), silu(-20.0) * -10.0);
}

#[test]
fn indexer_selects_all_rows_up_to_the_limit() {
    let cfg = test_config();
    let st = DsaState::default();
    assert_eq!(indexer_select(&cfg, &st, 2050, &[], &[]).len(), 2051);
}

#[test]
fn indexer_selects_top_pools_and_tail() {
    let mut cfg = test_config();
    cfg.indexer_top_k = 8; // 2 pools
    cfg.indexer_heads = 1;
    cfg.indexer_dim = 2;
    let mut st = DsaState::default();
    // 5 complete pools with keys scoring 0.1, 0.5, 0.3, 0.9, 0.2 on head 0.
    for s in [0.1f32, 0.5, 0.3, 0.9, 0.2] {
        st.pooled.extend([s, 0.0]);
    }
    let rows = indexer_select(&cfg, &st, 21, &[1.0, 0.0], &[1.0]);
    // visible 22: 5 pools, tail 2 (tokens 20, 21); pools 3 then 1.
    assert_eq!(rows, [12, 13, 14, 15, 4, 5, 6, 7, 20, 21]);
}

#[test]
fn kda_first_token_matches_closed_form() {
    // One head of width 2, zero state: S = k * v * beta (outer), o = S q.
    let mut cfg = test_config();
    cfg.kda_heads = 1;
    cfg.kda_dim = 2;
    let mut st = KdaState::new(&cfg);
    let conv = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let ones = [1.0f32, 1.0];
    let p = KdaParams {
        q_conv: &conv,
        k_conv: &conv,
        v_conv: &conv,
        a_log: &[0.0],
        dt_bias: &[0.0, 0.0],
        o_norm: &ones,
    };
    let q = [1.0f32, 2.0];
    let k = [3.0f32, -1.0];
    let v = [0.5f32, 2.0];
    let out =
        kda_step(&cfg, &mut st, &p, &q, &k, &v, &[0.0, 0.0], &[0.0], &[100.0, 100.0]);
    let sq: Vec<f32> = q.iter().map(|&x| silu(x)).collect();
    let sk: Vec<f32> = k.iter().map(|&x| silu(x)).collect();
    let sv: Vec<f32> = v.iter().map(|&x| silu(x)).collect();
    let nq = (sq[0] * sq[0] + sq[1] * sq[1] + 1e-6).sqrt();
    let nk = (sk[0] * sk[0] + sk[1] * sk[1] + 1e-6).sqrt();
    let qd: f32 =
        sq.iter().zip(&sk).map(|(a, b)| a / nq / 2f32.sqrt() / 8.0 * b / nk).sum();
    let o: Vec<f32> = sv.iter().map(|&x| x * 0.5 * qd).collect(); // beta = sigmoid(0)
    let expect = rmsnorm(&o, Some(&ones), cfg.rms_eps);
    for i in 0..2 {
        assert!((out[i] - expect[i]).abs() < 1e-5, "{} vs {}", out[i], expect[i]);
    }
}

pub(crate) fn test_config() -> Config {
    Config {
        n_layer: 46,
        n_trunk: 45,
        hidden: 4096,
        vocab: 154880,
        ff_dense: 12288,
        ff_expert: 2048,
        n_expert: 288,
        n_expert_used: 8,
        n_leading_dense: 3,
        expert_weight_scale: 2.5,
        swiglu_limit: 10.0,
        rms_eps: 1e-5,
        n_head: 64,
        key_dim: 256,
        value_dim: 256,
        q_lora: 1536,
        kv_lora: 512,
        indexer_heads: 32,
        indexer_dim: 128,
        indexer_top_k: 2048,
        indexer_pool: 4,
        kda_heads: 64,
        kda_dim: 128,
        kda_conv: 4,
        kda_gate_lower_bound: -5.0,
        n_hc: 4,
        sinkhorn_iters: 20,
        hc_eps: 1e-6,
        context_length: 1 << 20,
        kinds: (0..46)
            .map(
                |il| {
                    if il < 45 && il % 4 != 3 { LayerKind::Kda } else { LayerKind::Dsa }
                },
            )
            .collect(),
        bos: 154822,
        eos: 154820,
        eot: 154827,
        eom: 154829,
    }
}
