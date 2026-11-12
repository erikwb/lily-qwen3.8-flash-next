// What gdn.comp shares with the unit tests' kernels
// (tests/vulkan/gdn_test.comp): the head dimension and, for a file that
// defines GDN_STEP_BODY before including this, the decode step's body
// (`gdn_step_body` of src/kernels/metal/gdn.metal). Include after common.glsl.
#define DIM 128

#ifdef GDN_STEP_BODY
// Rows of the state a decode-step thread loads per register block.
#define GDN_STEP_RB 8

shared float q_norm[DIM];
shared float k_norm[DIM];
shared float part_q[DIM / 32];
shared float part_k[DIM / 32];
shared float gates[4];

// One GDN decode step of value head h; each thread owns one state column.
// `state` is the address of the fp32 [H, DIM, DIM] state.
float gdn_step_body(BF16 q, BF16 k, BF16 v, BF16 a, BF16 b, F32 a_log, BF16 dt_bias, uint64_t state,
                    float scale, uint vpk, uint h, uint tid, uint sg, uint lane) {
    uint hk = h / vpk;
    float qv = float(q.v[hk * DIM + tid]);
    float kv = float(k.v[hk * DIM + tid]);
    float sq = subgroupAdd(qv * qv);
    float sk = subgroupAdd(kv * kv);
    if (lane == 0) {
        part_q[sg] = sq;
        part_k[sg] = sk;
    }
    barrier();
    if (tid == 0) {
        float tq = 0.0;
        float tk = 0.0;
        for (uint i = 0; i < DIM / 32; ++i) {
            tq += part_q[i];
            tk += part_k[i];
        }
        gates[0] = 1.0 / max(sqrt(tq), 1e-12);
        gates[1] = 1.0 / max(sqrt(tk), 1e-12);
        float x = float(a.v[h]) + float(dt_bias.v[h]);
        float softplus = x > 20.0 ? x : log(1.0 + exp(x));
        gates[2] = exp(-exp(a_log.v[h]) * softplus);
        gates[3] = 1.0 / (1.0 + exp(-float(b.v[h])));
    }
    barrier();
    q_norm[tid] = qv * gates[0] * scale;
    k_norm[tid] = kv * gates[1];
    barrier();

    float decay = gates[2];
    float beta = gates[3];
    F32 st = F32(state + 4 * (uint64_t(h) * (DIM * DIM)));

    float kv_pred = 0.0;
    for (uint k0 = 0; k0 < DIM; k0 += GDN_STEP_RB) {
        float sv[GDN_STEP_RB];
        for (uint j = 0; j < GDN_STEP_RB; ++j) sv[j] = st.v[(k0 + j) * DIM + tid];
        for (uint j = 0; j < GDN_STEP_RB; ++j) kv_pred += k_norm[k0 + j] * sv[j] * decay;
    }
    float v_new = (float(v.v[h * DIM + tid]) - kv_pred) * beta;
    float o = 0.0;
    for (uint k0 = 0; k0 < DIM; k0 += GDN_STEP_RB) {
        float sv[GDN_STEP_RB];
        for (uint j = 0; j < GDN_STEP_RB; ++j) sv[j] = st.v[(k0 + j) * DIM + tid];
        for (uint j = 0; j < GDN_STEP_RB; ++j) {
            float updated = sv[j] * decay + k_norm[k0 + j] * v_new;
            st.v[(k0 + j) * DIM + tid] = updated;
            o += q_norm[k0 + j] * updated;
        }
    }
    return o;
}
#endif
