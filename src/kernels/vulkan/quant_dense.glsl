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
// The bits hold with K split too (QD_SK, below) except for the order of
// the partial sums: within an fp32 accumulation's error of them
// (`gemm_quant_split_matches_reference`).
//
// Two layouts, both over 16-row blocks of W:
// - direct: a subgroup dequantizes its block and runs the chains of QD_MB
//   (1, 2 or 4) 16-row blocks of A; the threadgroup's subgroups take
//   consecutive blocks of W. For an N that fills the GPU.
// - split K (QD_SK): a threadgroup per block of W and QD_MB blocks of A,
//   its subgroups each running a share of the groups (below). For a short
//   N or a long K, where the direct layout's chains are long.
// Define QD_BITS (4 or 8) and QD_MB before including. Buffers: 0 codes, 1
// scales, 2 biases, 3 A, 4 C; params 5 K, 6 N, 7 M, 8 GS. K is a multiple
// of 64, GS a power of two, at least 64; A's and C's addresses are
// multiples of 4.
//
// What did not pay (2026-11-13, us per call at 32 rows): a subgroup taking
// two blocks of W (more registers, fewer threads: q4 10 240 x 2 560 105
// against 73; with four blocks of A, the dense GEMM's 32 x 64 subgroup
// tile, 99 registers spilled, at 512 rows 807 against 788), and the direct
// layout issuing a group's loads a group ahead (q8 12 288 x 2 560 147
// against 124). A producer layout (a threadgroup per block of W, some
// subgroups dequantizing groups into threadgroup memory for the others to
// multiply: q8 320 x 10 240 at 32 rows 83, the direct layout 158) ran the
// same chain as the direct one, and the chain was its floor; split K is
// faster at every shape measured.
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
// A group: with four blocks of A, block by block (the four left operands
// first, then each block's four multiplies, its loads issued while the
// block before it multiplies); with one or two, multiply by multiply with
// every load issued at the group's start (QD_STEP). The same chain per
// output either way. Block by block keeps fewer loaded vectors live: q4
// b4 spills 19 registers where it spilled 50, and at 512 rows q4 10 240 x
// 2 560 takes 750 us against 784, q8 12 288 x 2 560 946 against 1 032 (a
// split b4 at 256 rows, the five big projections' pass 1.29 ms against
// 1.42); with two blocks it lost (b2, q4 10 240 x 2 560 at 32 rows 98
// against 70), 2026-11-13.
#define QD_MA(name, row0, row1)                                                               \
    MatA name;                                                                                \
    {                                                                                         \
        uvec4 r0 = row0, r1 = row1;                                                           \
        name[0] = MAT_HI(r0.x); name[1] = MAT_HI(r0.y); name[2] = MAT_HI(r0.z); name[3] = MAT_HI(r0.w); \
        name[4] = MAT_HI(r1.x); name[5] = MAT_HI(r1.y); name[6] = MAT_HI(r1.z); name[7] = MAT_HI(r1.w); \
    }
#define QD_MUL1(b, t)                                                                         \
    {                                                                                         \
        MatB mb;                                                                              \
        MAT_K64_B(mb, t, x##b##0, x##b##1, x##b##2, x##b##3)                                  \
        acc##b = coopMatMulAdd(ma##t, mb, acc##b);                                            \
    }
#define QD_BLOCK1(b) QD_MUL1(b, 0) QD_MUL1(b, 1) QD_MUL1(b, 2) QD_MUL1(b, 3)
#define QD_MAS                                                                                \
    QD_MA(ma0, QD_W_ROW0(0), QD_W_ROW1(0)) QD_MA(ma1, QD_W_ROW0(1), QD_W_ROW1(1))             \
    QD_MA(ma2, QD_W_ROW0(2), QD_W_ROW1(2)) QD_MA(ma3, QD_W_ROW0(3), QD_W_ROW1(3))
#if QD_MB == 4
#define QD_GROUP(g8)                                                                          \
    QD_W_LOAD(g8) QD_SB_LOAD                                                                  \
    QD_X_LOAD(0) QD_MAS QD_X_LOAD(1) QD_BLOCK1(0) QD_X_LOAD(2) QD_BLOCK1(1) QD_X_LOAD(3)      \
    QD_BLOCK1(2) QD_BLOCK1(3)
#else
#define QD_GROUP(g8)                                                                          \
    QD_W_LOAD(g8)                                                                             \
    QD_EACH_B(QD_X_LOAD)                                                                      \
    QD_SB_LOAD                                                                                \
    QD_STEP(0) QD_STEP(1) QD_STEP(2) QD_STEP(3)
#endif

#ifdef QD_SK
// Split K (QD_SK): a threadgroup per 16-row block of W and QD_MB blocks of
// A, its KS subgroups (a power of two, KS * QD_MB <= 32) each running the
// direct layout's chain over every KS-th group, then summing their fp32
// partial accumulators in a fixed tree through threadgroup memory
// (subgroup s adds s + KS / 2, then s + KS / 4, ...). For a short N with a
// long K, where the bit-exact chain is the floor (a 16x16x16 multiply waits
// about 115 ns for the one before on its accumulator: q8 320 x 10 240, 640
// multiplies, 84 us at 32 rows in the producer layout). The sums group the
// products differently from the staged route, so the bits move within the
// fp32 accumulation's error (`gemm_quant_split_matches_reference`).
shared vec4 qd_part[16 * 2 * 32];
#define QD_PART_AT(slot, i) (((slot) * 2 + (i)) * 32 + lane)
#define QD_PART_PUT(b)                                                                        \
    {                                                                                         \
        uint at = (s - span) * QD_MB + (b);                                                   \
        qd_part[QD_PART_AT(at, 0)] = vec4(acc##b[0], acc##b[1], acc##b[2], acc##b[3]);        \
        qd_part[QD_PART_AT(at, 1)] = vec4(acc##b[4], acc##b[5], acc##b[6], acc##b[7]);        \
    }
#define QD_PART_ADD(b)                                                                        \
    {                                                                                         \
        uint at = s * QD_MB + (b);                                                            \
        vec4 p0 = qd_part[QD_PART_AT(at, 0)], p1 = qd_part[QD_PART_AT(at, 1)];                \
        acc##b[0] += p0.x; acc##b[1] += p0.y; acc##b[2] += p0.z; acc##b[3] += p0.w;           \
        acc##b[4] += p1.x; acc##b[5] += p1.y; acc##b[6] += p1.z; acc##b[7] += p1.w;           \
    }

void main() {
    QD_SETUP
    uint s = gl_SubgroupID, KS = gl_NumSubgroups, lane = gl_SubgroupInvocationID;
    uint n0 = threadgroup_position_in_grid.x * 16;
    uint m0 = threadgroup_position_in_grid.y * (16 * QD_MB);
    QD_W_ROWS
    QD_EACH_B(QD_X_ROWS_B)
    QD_EACH_B(QD_ACC)

    [[dont_unroll]] for (uint g8 = 8 * s; g8 < K8; g8 += 8 * KS) {
        QD_GROUP(g8)
    }

    // Partial sums of subgroups s in [span, 2 span) into subgroup s - span.
    [[dont_unroll]] for (uint span = KS >> 1; span > 0; span >>= 1) {
        if (s >= span && s < 2 * span) {
            QD_EACH_B(QD_PART_PUT)
        }
        barrier();
        if (s < span) {
            QD_EACH_B(QD_PART_ADD)
        }
        barrier();
    }

    if (s == 0) {
        QD_EACH_B(QD_STORE)
    }
}
#else
void main() {
    QD_SETUP
    uint n0 = (threadgroup_position_in_grid.x * gl_NumSubgroups + gl_SubgroupID) * 16;
    uint m0 = threadgroup_position_in_grid.y * (16 * QD_MB);
    if (n0 >= N) return;
    QD_W_ROWS
    QD_EACH_B(QD_X_ROWS_B)
    QD_EACH_B(QD_ACC)

    [[dont_unroll]] for (uint g8 = 0; g8 < K8; g8 += 8) {
        QD_GROUP(g8)
    }

    QD_EACH_B(QD_STORE)
}
#endif
