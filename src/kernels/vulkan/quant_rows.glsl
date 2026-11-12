// The raw-code projection body of the Q4 decode GEMVs (quant.comp) and the
// 8-bit register-A GEMMs (skinny.comp): C[MB, N] = A[MB, K] . W[N, K]^T with
// W affine Q4 or Q8, each block of codes dotted raw and scaled once,
// s * dot(q, x) + b * sum(x), so the weights are never dequantized element by
// element (quant.metal's dot_word_q4). Include after common.glsl and
// quant_common.glsl with
//
//     QR_BITS 4 or 8   the code width; a block is one uint4 of codes (32 or
//                      16 elements) and must not straddle a quant group
//     QR_MB            activation rows (the kernel requires M == QR_MB)
//     QR_RR            weight rows a lane computes per block (1 or 2)
//     QR_F32           defined: f32 output instead of bf16
//     QR_SETS          defined: up to four weights in one dispatch (below)
//
// Buffers: 0 codes, 1 scales, 2 biases, 3 A (bf16), 4 C; params: 5 K, 6 N,
// 7 GS (a power of two), 8 log2 of the lanes per row set. The grid is one
// subgroup per (32 >> param 8) * QR_RR weight rows.
//
// With QR_SETS the dispatch computes up to four projections of the same A
// (weights of one code width, K and GS; each its own N, output and output
// type), set after set along the grid, each set starting a subgroup:
// buffers 5 + 4 * (s - 1) and the three after are the codes, scales, biases
// and C of set s = 1..3 (an unused set: N = 0), and the params move to 17 K,
// 18 N of set 0, 19 GS, 20 the lanes' log2, 21..23 N of sets 1..3, 24 a mask
// whose bit s says set s's output is f32. Dispatches that share a dependency
// level do not overlap on this driver as they do on Metal (quant.comp,
// gemv_q8_sets): the batched router level's three projections at three
// activation rows take 22.8 us as three dispatches and 15.7 as one
// (`dense_shared_levels_timing`, 2026-11-10).
//
// This is not the Metal kernels' layout (one subgroup per row or per two
// rows, 32 lanes each: quant.metal's Q4 GEMV bodies and skinny.metal's
// register-A Q8 body), and nothing pins their arithmetic (Metal compiles
// them with fast math; the Q8 GEMV, which hc.comp reproduces, is not this
// body), so the order here is this file's own. What differs and why,
// measured 2026-11-10 on the M5 Max in chains of barriered levels over
// distinct weights, in passes as long as a decode step
// (`dense_chain_timing`, `quant_rows_layout_probe`; us per level):
//
// - A set of 2^ls lanes walks QR_RR weight rows, and a subgroup holds
//   32 >> ls sets. The caller picks ls from K alone so that the lanes divide
//   the row's blocks: K = 2560 at 4 bits is 80 blocks, which 16 lanes walk
//   in five steps each, where 32 lanes take a third step for half of them
//   and the idle half still occupies the subgroup's ALU time; K = 640 at 8
//   bits is 40 blocks, 8 lanes. A row's arithmetic depends on K, GS and the
//   row's own data and on nothing else, so its result is the same bits
//   whatever N, QR_RR, QR_MB, QR_SETS, the row's index or the threadgroup
//   size: a fused stack and its slices agree, and so do a weight alone and
//   among others in one dispatch.
// - With QR_RR = 2 the two rows share the block's activation work (the
//   loads, the widening, the sum). It pays from about 3 MB of codes (the
//   24 MB 4-bit GDN input GEMV 54.2 at one row per lane, 49.8 at two; the
//   17 MB 8-bit attention output at three activation rows 42.6 and 35.7).
//   Below that a level is bound by its ramp, more and shorter subgroups
//   win, and QR_RR = 1 is used (the 1.4 to 1.7 MB 8-bit projections at
//   three rows: 8.4 to 8.7 against 9.6 to 9.9). Four rows per lane were
//   slower than two wherever tried (their registers cost residency).
// - A block's quant group is a shift: a division by the runtime group size
//   is about twenty instructions on this compiler, and the Metal kernels
//   have one per block (the ported 4-bit GEMV on the 24 MB level, in an
//   experiment of that day: 59.9 with it, 54.1 without).
// - Activations are read as 16 bytes through a 4-byte aligned reference,
//   with no branch on A's alignment (skinny_common.glsl, SKINNY_LOAD_ROWS).
// - The sums go an element at a time across the rows, so one activation per
//   row is live in registers, not eight (the 17 MB 8-bit level at three
//   rows, same experiments: 53.8 with a row's eight elements widened first,
//   37.1 this way), and scale and bias are applied once per block, not per
//   code word.
//
// Against the kernels as ported from Metal: the 4-bit GDN GEMVs 53.7 to 49.5
// (in, 24 MB) and 22.0 to 21.4 (out, 9 MB); the 8-bit GEMMs at three rows
// 74.1 to 70.4 (36 MB), 39.6 to 35.2 (17 MB), 13.5 to 11.1 (3.5 MB) and 9.1
// to 10.7 down to 7.8 to 8.5 (1.4 to 1.7 MB).

#if QR_BITS == 4
#define QR_BLOCK_LOG2 5
#define QR_STEPS 4
#else
#define QR_BLOCK_LOG2 4
#define QR_STEPS 2
#endif

// Each operation in its place (a precise dot() is lowered to the chain of
// fused multiply-adds in component order, and its `acc * 1.0` folds away;
// GLSL's fma() is fused only where the compiler chooses): fma(q, x, acc),
// acc + x, and a block folded into a lane's partial,
// fma(b, xs, fma(s, qx, acc)).
float qr_fma(float acc, float q, float x) {
    precise float d = dot(vec2(acc, q), vec2(1.0, x));
    return d;
}
float qr_add(float acc, float x) {
    precise float s = acc + x;
    return s;
}
float qr_fold(float acc, float s, float qx, float b, float xs) {
    precise float r = dot(vec3(acc, s, b), vec3(1.0, qx, xs));
    return r;
}

#ifdef QR_SETS
#define QR_PARAMS 17
#else
#define QR_PARAMS 5
#endif

void main() {
    uint64_t codes = arg(0), scales_at = arg(1), biases_at = arg(2), c = arg(4);
    U32x4 a = U32x4(arg(3));
    uint K = arg_u32(QR_PARAMS), N = arg_u32(QR_PARAMS + 1), GS = arg_u32(QR_PARAMS + 2);
    uint ls = arg_u32(QR_PARAMS + 3);
    uint lane = thread_index_in_simdgroup;
    uint set_in_sg = lane >> ls, sublane = lane & ((1u << ls) - 1u);
    uint sg = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
#ifdef QR_SETS
    // The weight this subgroup belongs to: each takes the subgroups its rows
    // need, in order (a subgroup holds (32 >> ls) * QR_RR rows).
    uint f32_mask = arg_u32(QR_PARAMS + 7);
    bool f32_out = (f32_mask & 1u) != 0u;
    uint sg_rows_log2 = 5u - ls + (QR_RR / 2);
    [[unroll]] for (uint s = 1; s < 4; ++s) {
        uint before = (N + (1u << sg_rows_log2) - 1u) >> sg_rows_log2;
        if (sg >= before) {
            sg -= before;
            codes = arg(1 + 4 * s);
            scales_at = arg(2 + 4 * s);
            biases_at = arg(3 + 4 * s);
            c = arg(4 + 4 * s);
            N = arg_u32(QR_PARAMS + 3 + s);
            f32_out = (f32_mask >> s & 1u) != 0u;
        }
    }
#endif
    BF16 scales = BF16(scales_at), biases = BF16(biases_at);
    uint row0 = ((sg << (5u - ls)) + set_in_sg) * QR_RR;
    // Sets past N do nothing; the sums below cover the lanes that remain.
    if (row0 >= N) return;
    uint blocks = K >> QR_BLOCK_LOG2, gs_log2 = uint(findLSB(GS));
    uint gshift = gs_log2 - QR_BLOCK_LOG2, groups = K >> gs_log2;
    U32x4 w = U32x4(codes + uint64_t(row0) * (blocks * 16));
    // Rows past N repeat the last row (the loads stay in bounds) and are not
    // stored.
    uint woff[QR_RR], soff[QR_RR];
    [[unroll]] for (uint r = 0; r < QR_RR; ++r) {
        uint row = min(row0 + r, N - 1);
        woff[r] = (row - row0) * blocks;
        soff[r] = row * groups;
    }

    float acc[QR_RR][QR_MB];
    [[unroll]] for (uint r = 0; r < QR_RR; ++r) {
        [[unroll]] for (uint i = 0; i < QR_MB; ++i) acc[r][i] = 0.0;
    }

    for (uint blk = sublane; blk < blocks; blk += 1u << ls) {
        uint g = blk >> gshift;
        uvec4 wv[QR_RR];
        float s[QR_RR], b[QR_RR];
        [[unroll]] for (uint r = 0; r < QR_RR; ++r) {
            wv[r] = w.v[woff[r] + blk];
            s[r] = float(scales.v[soff[r] + g]);
            b[r] = float(biases.v[soff[r] + g]);
        }
        float qx[QR_RR][QR_MB], xs[QR_MB];
        [[unroll]] for (uint i = 0; i < QR_MB; ++i) {
            xs[i] = 0.0;
            [[unroll]] for (uint r = 0; r < QR_RR; ++r) qx[r][i] = 0.0;
        }
        // Eight elements per step: one 16-byte activation load per row.
        [[unroll]] for (uint st = 0; st < QR_STEPS; ++st) {
            uvec4 raw[QR_MB];
            [[unroll]] for (uint i = 0; i < QR_MB; ++i) raw[i] = a.v[i * (K >> 3) + blk * QR_STEPS + st];
            [[unroll]] for (uint j = 0; j < 8; ++j) {
                float x[QR_MB];
                [[unroll]] for (uint i = 0; i < QR_MB; ++i) {
                    // A bf16 is the high half of its float.
                    uint xw = raw[i][j / 2];
                    x[i] = uintBitsToFloat(j % 2 == 0 ? xw << 16 : xw & 0xFFFF0000u);
                    xs[i] = qr_add(xs[i], x[i]);
                }
                [[unroll]] for (uint r = 0; r < QR_RR; ++r) {
#if QR_BITS == 4
                    float q = float((wv[r][st] >> (4 * j)) & 0xFu);
#else
                    float q = float((wv[r][2 * st + j / 4] >> (8 * (j % 4))) & 0xFFu);
#endif
                    [[unroll]] for (uint i = 0; i < QR_MB; ++i) qx[r][i] = qr_fma(qx[r][i], q, x[i]);
                }
            }
        }
        [[unroll]] for (uint r = 0; r < QR_RR; ++r) {
            [[unroll]] for (uint i = 0; i < QR_MB; ++i) acc[r][i] = qr_fold(acc[r][i], s[r], qx[r][i], b[r], xs[i]);
        }
    }

    // Each set's lanes summed apart (-0.0 adds nothing), its first lane
    // storing.
    for (uint t = 0; t < (32u >> ls); ++t) {
        bool mine = set_in_sg == t;
        [[unroll]] for (uint r = 0; r < QR_RR; ++r) {
            [[unroll]] for (uint i = 0; i < QR_MB; ++i) {
                float sum = subgroupAdd(mine ? acc[r][i] : -0.0);
                if (mine && sublane == 0 && row0 + r < N) {
#if defined(QR_SETS)
                    if (f32_out) {
                        F32(c + uint64_t(i) * (4 * N)).v[row0 + r] = sum;
                    } else {
                        BF16(c + uint64_t(i) * (2 * N)).v[row0 + r] = bfloat16_t(sum);
                    }
#elif defined(QR_F32)
                    F32(c + uint64_t(i) * (4 * N)).v[row0 + r] = sum;
#else
                    BF16(c + uint64_t(i) * (2 * N)).v[row0 + r] = bfloat16_t(sum);
#endif
                }
            }
        }
    }
}
