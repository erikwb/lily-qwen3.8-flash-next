// The dense quantized GEMM C[M, N] = A[M, K] . dequant(W[N, K])^T on the
// matrix units, dequantizing in the kernel: the route of prefill chunks too
// short to pay for staging the weight in bf16 (quant.rs, `fused_route`). It
// computes the transpose, C^T = W . A^T, so the weight is the left operand:
// a lane of a left operand holds two weight rows and, per 64-deep group in
// matmul.glsl's MAT_K64 pairing, the even or the odd codes of 32 consecutive
// ones of each, which are one 16-byte load of a 4-bit row (two of an 8-bit
// one) with one scale and one bias (a quant group is at least 64 deep). As
// the right operand (the staged route's layout) it would take four rows'
// codes, scales and biases per lane and group.
//
// Bit for bit the staged route's result (dequant_q*_bf16, then
// gemm_bf16_nt_nax): each weight is that kernel's bf16(q * s + b), every
// output the same chain of 16-deep multiplies in the same K64 pairing (a
// product's operands trade sides, which the matrix units' sums do not see:
// `gemm_quant_fused_matches_staged`), rounded by the same store.
//
// Two layouts, both over 16-row blocks of W:
// - direct (QD_PC undefined): a subgroup dequantizes its block and runs the
//   chains of QD_MB (1, 2 or 4) 16-row blocks of A; the threadgroup's
//   subgroups take consecutive blocks of W. For an N that fills the GPU.
// - producer (QD_PC): a threadgroup per block of W and QD_MB blocks of A,
//   its subgroups split between dequantizing and multiplying (below). For a
//   short N, where the direct layout's few subgroups each run a long chain.
// Define QD_BITS (4 or 8) and QD_MB before including; for QD_PC also
// QD_PC_MAX_P. Buffers: 0 codes, 1 scales, 2 biases, 3 A, 4 C; params 5 K,
// 6 N, 7 M, 8 GS. K is a multiple of 64, GS a power of two, at least 64; A's
// and C's addresses are multiples of 4.
//
// What did not pay (2026-11-13, us per call at 32 rows): a subgroup taking
// two blocks of W (more registers, fewer threads: q4 10 240 x 2 560 105
// against 73), and the direct layout issuing a group's loads a group ahead
// (q8 12 288 x 2 560 147 against 124).
#include "matmul.glsl"

// Weights k .. k + 4 of four codes' floats q: bf16(q * s + b), rounded as
// dequant_q*_bf16 rounds (bf16_weight), each bfloat in its word's high half.
uvec4 qd_weights(vec4 q, float s, float b) {
    uvec4 u = floatBitsToUint(q * s + b);
    return u + 0x7FFFu + ((u >> 16) & 1u);
}

// The codes of multiply t of a group as four floats: from the lane's codes
// of its row (shifted right by its parity's 4 or 8 bits), the code of
// element 2 t of each of four 8-code vectors.
#if QD_BITS == 4
#define QD_CODES_T(v0, v1, t) vec4(((v0) >> (8 * (t))) & 0xFu)
#define QD_ROW_VECTORS (K / 32)
#else
#define QD_CODES_T(v0, v1, t)                                                                 \
    vec4((((t) < 2 ? uvec4((v0).x, (v0).z, (v1).x, (v1).z) : uvec4((v0).y, (v0).w, (v1).y, (v1).w)) \
          >> (16 * ((t) & 1))) & 0xFFu)
#define QD_ROW_VECTORS (K / 16)
#endif

// The lane's two rows of the block's weights at group g8 / 8 (g8 counts
// 8-element vectors of K): their codes, scales and biases, all loaded before
// the first is used; and the four rounded weights of each for multiply t.
#if QD_BITS == 4
#define QD_W_LOAD(g8)                                                                         \
    uint qd_gv = (g8) / 4, qd_gi = (8 * (g8)) >> gs_log2;                                     \
    uvec4 wc0 = cw.v[wv0 + qd_gv], wc1 = cw.v[wv1 + qd_gv], wd0 = uvec4(0u), wd1 = uvec4(0u);
#else
#define QD_W_LOAD(g8)                                                                         \
    uint qd_gv = (g8) / 2, qd_gi = (8 * (g8)) >> gs_log2;                                     \
    uvec4 wc0 = cw.v[wv0 + qd_gv], wd0 = cw.v[wv0 + qd_gv + 1];                               \
    uvec4 wc1 = cw.v[wv1 + qd_gv], wd1 = cw.v[wv1 + qd_gv + 1];
#endif
#define QD_SB_LOAD                                                                            \
    float ws0 = float(scales.v[wg0 + qd_gi]), wb0 = float(biases.v[wg0 + qd_gi]);             \
    float ws1 = float(scales.v[wg1 + qd_gi]), wb1 = float(biases.v[wg1 + qd_gi]);
#define QD_W_ROW0(t) qd_weights(QD_CODES_T(wc0 >> odd_bits, wd0 >> odd_bits, t), ws0, wb0)
#define QD_W_ROW1(t) qd_weights(QD_CODES_T(wc1 >> odd_bits, wd1 >> odd_bits, t), ws1, wb1)

// What both layouts set up: the params and the lane's place in the operands.
#define QD_SETUP                                                                              \
    uint64_t c = arg(4);                                                                      \
    U32x4 cw = U32x4(arg(0)), xw = U32x4(arg(3));                                             \
    BF16 scales = BF16(arg(1)), biases = BF16(arg(2));                                        \
    uint K = arg_u32(5), N = arg_u32(6), M = arg_u32(7), GS = arg_u32(8);                     \
    uint gs_log2 = uint(findLSB(GS)), groups = K >> gs_log2;                                  \
    uint lane_row = mat_lane_row(), lane_col = mat_lane_col();                                \
    uint K8 = K / 8, x_vec = lane_row / 2;
// The lane's 32 codes of a row at a group are its half's of the 64; of
// them it takes the even or the odd ones.
#if QD_BITS == 4
#define QD_LANE_CODES uint a_vec = mat_a_half(), odd_bits = 4 * mat_a_odd();
#else
#define QD_LANE_CODES uint a_vec = 2 * mat_a_half(), odd_bits = 8 * mat_a_odd();
#endif
// The lane's two weight rows of the block at n0 (rows past N read row
// N - 1, their products not stored): their first code vector and scale.
#define QD_W_ROWS                                                                             \
    QD_LANE_CODES                                                                             \
    uint wr0 = min(n0 + lane_row, N - 1), wr1 = min(n0 + lane_row + 1, N - 1);                \
    uint wv0 = wr0 * QD_ROW_VECTORS + a_vec, wv1 = wr1 * QD_ROW_VECTORS + a_vec;              \
    uint wg0 = wr0 * groups, wg1 = wr1 * groups;
// The lane's four rows of A of block b, from row m (the right operand's
// columns): their vectors' indices at the first group.
#define QD_X_ROWS(b, m)                                                                       \
    uint xv##b##0 = min((m), M - 1) * K8 + x_vec, xv##b##1 = min((m) + 1, M - 1) * K8 + x_vec; \
    uint xv##b##2 = min((m) + 2, M - 1) * K8 + x_vec, xv##b##3 = min((m) + 3, M - 1) * K8 + x_vec;

// A lane's part of an accumulator: weight rows n, n + 1 of rows m .. m + 4
// of A; a row of C takes the two as one word where it can.
void qd_store(uint64_t c, uint N, uint M, uint n, uint m, MatC acc) {
    uvec2 lo = mat_bf16_words(vec4(acc[0], acc[4], acc[1], acc[5]));
    uvec2 hi = mat_bf16_words(vec4(acc[2], acc[6], acc[3], acc[7]));
    uint words[4] = uint[4](lo.x, lo.y, hi.x, hi.y);
    bool pairs = (N & 1) == 0 && (c & 3) == 0;
    if (n >= N) return;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        if (m + j >= M) break;
        uint at = (m + j) * N + n;
        if (pairs) {
            U32(c).v[at / 2] = words[j];
        } else {
            U16(c).v[at] = uint16_t(words[j]);
            if (n + 1 < N) U16(c).v[at + 1] = uint16_t(words[j] >> 16);
        }
    }
}

#ifdef QD_PC
// The producer layout. For a short N (the hyper-connection down and inject
// projections: 320 and 4 rows of K 10 240) the direct layout's subgroups
// are few and each runs a long chain of groups, a wait for the group's loads
// and some 450 dependent instructions of dequantization before its four
// multiplies: 1.2 us a group. Here a threadgroup takes one 16-row block of
// W: of its S subgroups, P = S - QD_MB (at most QD_PC_MAX_P) produce, each
// dequantizing a group per step into threadgroup memory (the lane's four
// left operands as it holds them), and QD_MB consume, subgroup b running the
// groups of the step before, in order, through the accumulator of rows
// 16 b .. of A. Two buffers: one barrier per step, the producers a step
// ahead. Each output is the direct layout's chain, so the bits stay.
//
// What bounds it: with the producers' work removed it takes as long (q8 320
// x 10 240 at 16 rows, 2026-11-13: 107.6 us against 107.9), so it is the
// consumer's chain. A 16x16x16 multiply waits for the one before on its
// accumulator, about 115 ns, so a chain of 640 (K 10 240) takes about 75 us
// whatever feeds it. Only splitting K, which changes the sums, would go
// below. 84 us at 32 rows, the direct layout 207.
shared uvec4 qd_ops[2 * QD_PC_MAX_P * 4 * 32];

// A left operand's words: the four weights of the lane's first row, then
// of its second, in element order (MAT_OF_WORDS).
uvec4 qd_operand_words(uvec4 r0, uvec4 r1) {
    return uvec4((r0.x >> 16) | (r0.y & 0xFFFF0000u), (r0.z >> 16) | (r0.w & 0xFFFF0000u),
                 (r1.x >> 16) | (r1.y & 0xFFFF0000u), (r1.z >> 16) | (r1.w & 0xFFFF0000u));
}
#define QD_PC_STORE(t)                                                                        \
    qd_ops[((buf * QD_PC_MAX_P + p) * 4 + (t)) * 32 + lane] = qd_operand_words(QD_W_ROW0(t), QD_W_ROW1(t));
// Group j of a consumer's step: the lane's activation vectors, loaded with
// the step's others (past G a group reads the last one's and multiplies
// nothing), then its four operands from threadgroup memory and the four
// multiplies.
#define QD_PC_X_LOAD(j)                                                                       \
    uint g8##j = 8 * min(first + (j), G - 1);                                                 \
    uvec4 x##j##0 = xw.v[xv00 + g8##j], x##j##1 = xw.v[xv01 + g8##j];                         \
    uvec4 x##j##2 = xw.v[xv02 + g8##j], x##j##3 = xw.v[xv03 + g8##j];
#define QD_PC_STEP(j, t)                                                                      \
    {                                                                                         \
        MatA ma;                                                                              \
        MAT_OF_WORDS(ma, o##t)                                                                \
        MatB mb;                                                                              \
        MAT_K64_B(mb, t, x##j##0, x##j##1, x##j##2, x##j##3)                                  \
        acc = coopMatMulAdd(ma, mb, acc);                                                     \
    }
#define QD_PC_GROUP(j)                                                                        \
    if ((j) < count) {                                                                        \
        uint at = (rbuf * QD_PC_MAX_P + (j)) * 128 + lane;                                    \
        uvec4 o0 = qd_ops[at], o1 = qd_ops[at + 32], o2 = qd_ops[at + 64], o3 = qd_ops[at + 96]; \
        QD_PC_STEP(j, 0) QD_PC_STEP(j, 1) QD_PC_STEP(j, 2) QD_PC_STEP(j, 3)                   \
    }

void main() {
    QD_SETUP
    uint n0 = threadgroup_position_in_grid.x * 16, m0 = threadgroup_position_in_grid.y * (16 * QD_MB);
    uint sg = gl_SubgroupID, lane = gl_SubgroupInvocationID;
    uint P = gl_NumSubgroups - QD_MB, p = sg - QD_MB, G = K / 64, steps = (G + P - 1) / P;
    QD_W_ROWS
    // A consumer's rows of A: block sg of the tile.
    uint m = m0 + 16 * sg + lane_col;
    QD_X_ROWS(0, m)
    bool consumer = sg < QD_MB, working = consumer && m0 + 16 * sg < M;
    MatC acc = MatC(0.0);

    [[dont_unroll]] for (uint i = 0; i <= steps; ++i) {
        uint buf = i & 1, rbuf = buf ^ 1;
        uint g = i * P + p;
        if (!consumer && g < G) {
            QD_W_LOAD(8 * g)
            QD_SB_LOAD
            QD_PC_STORE(0) QD_PC_STORE(1) QD_PC_STORE(2) QD_PC_STORE(3)
        } else if (working && i > 0) {
            // Step i - 1. A thread waits for its loads where it first uses
            // one: the step's activations are loaded together, one wait a
            // step (107 us for 320 x 10 240 with each group's loaded a group
            // ahead, 74 with no activation loads at all, 84 this way).
            uint first = (i - 1) * P, count = min(P, G - first);
            QD_PC_X_LOAD(0) QD_PC_X_LOAD(1) QD_PC_X_LOAD(2) QD_PC_X_LOAD(3)
            QD_PC_GROUP(0) QD_PC_GROUP(1) QD_PC_GROUP(2) QD_PC_GROUP(3)
        }
        barrier();
    }

    if (working) qd_store(c, N, M, n0 + lane_row, m, acc);
}
#else
#if QD_MB == 1
#define QD_EACH_B(X) X(0)
#elif QD_MB == 2
#define QD_EACH_B(X) X(0) X(1)
#else
#define QD_EACH_B(X) X(0) X(1) X(2) X(3)
#endif
#define QD_X_ROWS_B(b) QD_X_ROWS(b, m0 + 16 * (b) + lane_col)
#define QD_ACC(b) MatC acc##b = MatC(0.0);
// Block b of A at the group: the lane's vector of each of its four rows;
// then, per multiply t, its right operand and the product.
#define QD_X_LOAD(b)                                                                          \
    uvec4 x##b##0 = xw.v[xv##b##0 + g8], x##b##1 = xw.v[xv##b##1 + g8];                       \
    uvec4 x##b##2 = xw.v[xv##b##2 + g8], x##b##3 = xw.v[xv##b##3 + g8];
#define QD_MUL(b, t)                                                                          \
    {                                                                                         \
        MatB mb;                                                                              \
        MAT_K64_B(mb, t, x##b##0, x##b##1, x##b##2, x##b##3)                                  \
        acc##b = coopMatMulAdd(ma, mb, acc##b);                                               \
    }
#define QD_MUL_0(b) QD_MUL(b, 0)
#define QD_MUL_1(b) QD_MUL(b, 1)
#define QD_MUL_2(b) QD_MUL(b, 2)
#define QD_MUL_3(b) QD_MUL(b, 3)
// Multiply t of the group: the left operand from the weights, then its
// product with every block of A.
#define QD_STEP(t)                                                                            \
    {                                                                                         \
        MatA ma;                                                                              \
        uvec4 r0 = QD_W_ROW0(t), r1 = QD_W_ROW1(t);                                           \
        ma[0] = MAT_HI(r0.x); ma[1] = MAT_HI(r0.y); ma[2] = MAT_HI(r0.z); ma[3] = MAT_HI(r0.w); \
        ma[4] = MAT_HI(r1.x); ma[5] = MAT_HI(r1.y); ma[6] = MAT_HI(r1.z); ma[7] = MAT_HI(r1.w); \
        QD_EACH_B(QD_MUL_##t)                                                                 \
    }
#define QD_STORE(b) qd_store(c, N, M, n0 + lane_row, m0 + 16 * (b) + lane_col, acc##b);

void main() {
    QD_SETUP
    uint n0 = (threadgroup_position_in_grid.x * gl_NumSubgroups + gl_SubgroupID) * 16;
    uint m0 = threadgroup_position_in_grid.y * (16 * QD_MB);
    if (n0 >= N) return;
    QD_W_ROWS
    QD_EACH_B(QD_X_ROWS_B)
    QD_EACH_B(QD_ACC)

    // A group's loads are issued together, at its start.
    [[dont_unroll]] for (uint g8 = 0; g8 < K8; g8 += 8) {
        QD_W_LOAD(g8)
        QD_EACH_B(QD_X_LOAD)
        QD_SB_LOAD
        QD_STEP(0) QD_STEP(1) QD_STEP(2) QD_STEP(3)
    }

    QD_EACH_B(QD_STORE)
}
#endif
