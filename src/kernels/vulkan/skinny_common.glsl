// The bodies of the small-M Q4/Q8 GEMMs of skinny.metal, for skinny.comp's
// kernels and the test kernels of tests/vulkan/skinny_test.comp. Include
// after common.glsl and quant_common.glsl, with one family selected:
//
//     SKINNY_STAGED 4 or 8   staged-A body at that code width, with SKINNY_MB
//                            rows and K chunks of SKINNY_KC (default 256)
//     SKINNY_REG_Q4 m        4-bit register-A body at m rows
//
// (none: the helpers alone), and SKINNY_F32 for f32 output instead of bf16.
// The 8-bit register-A body is quant_rows.glsl.
//
// The group size GS is a power of two here (skinny.rs checks it): a block's
// quant group is a shift. Metal divides by GS per block, which this compiler
// turns into about twenty instructions each time.

// c[row, col] of the [M, N] output, in its element type.
#ifdef SKINNY_F32
#define SKINNY_STORE(c, row, N, col, value) F32((c) + uint64_t(row) * (4 * (N))).v[col] = (value)
#else
#define SKINNY_STORE(c, row, N, col, value) BF16((c) + uint64_t(row) * (2 * (N))).v[col] = bfloat16_t(value)
#endif

// raw[i], i in 0..MB: eight activations of row i at element `at` of the row
// (a multiple of 8), as their bits: one 16-byte load each. Metal takes that
// load only where A is 16-byte aligned and eight 2-byte loads elsewhere,
// choosing per code word; a uvec4 here is read through a 4-byte aligned
// reference, which is all a tensor's offset is held to, so the one load
// serves both and the loops carry no branch on the alignment. With the group
// shift above, measured 2026-11-10 in chains of barriered levels at m = 3
// (`dense_chain_timing`): the 4-bit GDN projections 81.1 to 69.1 us per
// level (in, 24 MB) and 33.6 to 27.7 (out, 9 MB), the same bits.
#define SKINNY_LOAD_ROWS(raw, at)                                                             \
    [[unroll]] for (uint i = 0; i < MB; ++i) raw[i] = U32x4(a).v[(i * K + (at)) / 8];

#ifdef SKINNY_STAGED
#ifndef SKINNY_KC
#define SKINNY_KC 256
#endif
#define SKINNY_SG 4

// The A tile, zero padded: SKINNY_MB rows of SKINNY_KC bf16, eight to a
// uvec4 (Metal reads and writes its bfloat tile through uint4 and bfloat4
// pointers; here a thread also stages eight elements where A is not 16-byte
// aligned, from eight loads).
shared uvec4 a_tile[SKINNY_MB * SKINNY_KC / 8];

void main() {
    BF16 scales = BF16(arg(1)), biases = BF16(arg(2));
    uint64_t a = arg(3), c = arg(4);
    uint K = arg_u32(5), N = arg_u32(6), GS = arg_u32(7), M = arg_u32(8);
    uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x * SKINNY_SG + simdgroup_index_in_threadgroup;
    uint m0 = threadgroup_position_in_grid.y * SKINNY_MB;
    uint m_rem = min(M - m0, uint(SKINNY_MB));
    uint words = K / (32 / SKINNY_STAGED), gs_log2 = uint(findLSB(GS)), groups = K >> gs_log2;
    bool a_vec = (a & 15) == 0;
    U32 wrow = U32(arg(0) + uint64_t(row) * (4 * words));
    a += uint64_t(m0) * (2 * K);

    float acc[SKINNY_MB];
    [[unroll]] for (uint i = 0; i < SKINNY_MB; ++i) acc[i] = 0.0;

    for (uint k0 = 0; k0 < K; k0 += SKINNY_KC) {
        uint kc = min(uint(SKINNY_KC), K - k0);
        for (uint i = tid; i < SKINNY_MB * (SKINNY_KC / 8); i += 32 * SKINNY_SG) {
            uint r = i / (SKINNY_KC / 8);
            uint col = (i % (SKINNY_KC / 8)) * 8;
            uvec4 v = uvec4(0);
            if (r < m_rem && col < kc) v = load_raw8(a, r * K + k0 + col, a_vec);
            a_tile[i] = v;
        }
        barrier();

        if (row < N) {
            for (uint col = lane * 8; col < kc; col += 256) {
                uint g = (k0 + col) >> gs_log2;
                float s = float(scales.v[row * groups + g]);
                float b = float(biases.v[row * groups + g]);
#if SKINNY_STAGED == 4
                uint word = wrow.v[(k0 + col) / 8];
                vec4 qlo = q4_lo(word), qhi = q4_hi(word);
#else
                // Two code words of four elements per eight-element step.
                uint wi = (k0 + col) / 4;
                vec4 qlo = q8_codes(wrow.v[wi]), qhi = q8_codes(wrow.v[wi + 1]);
#endif
                // The dequantized weights are rounded to bf16, as the
                // dequant + GEMM fallback's are.
                vec4 wlo = vec4(bf16_weight(qlo * s + b)), whi = vec4(bf16_weight(qhi * s + b));
                [[unroll]] for (uint i = 0; i < SKINNY_MB; ++i) {
                    uvec4 x = a_tile[i * (SKINNY_KC / 8) + col / 8];
                    acc[i] += dot(wlo, bf4_bits(x.xy)) + dot(whi, bf4_bits(x.zw));
                }
            }
        }
        barrier();
    }

    if (row < N) {
        [[unroll]] for (uint i = 0; i < SKINNY_MB; ++i) {
            float sum = subgroupAdd(acc[i]);
            if (lane == 0 && i < m_rem) SKINNY_STORE(c, m0 + i, N, row, sum);
        }
    }
}
#endif

// The arithmetic of the 4-bit register-A kernels, each operation in its place
// (Metal: safe math, no contraction), so the shipped kernel and its reference
// in tests/vulkan/skinny_test.comp round alike. GLSL's fma() is fused only
// where the compiler chooses, and the driver re-associates what it leaves
// split; a precise dot() is lowered to the chain of fused multiply-adds, in
// component order, and left alone.

// Four products summed left to right; each product is exact, so the fma is
// that sum's rounding, not an extra fusion.
float skinny_dot4(vec4 q, vec4 x) {
    precise float d = dot(q, x);
    return d;
}

float skinny_sum4(vec4 x) {
    precise float s = ((x.x + x.y) + x.z) + x.w;
    return s;
}

// Folds a block into a lane partial: fma(s, qx, fma(b, xs, acc)) where `sb`,
// else fma(b, xs, fma(s, qx, acc)).
float skinny_fold(bool sb, float s, float qx, float b, float xs, float acc) {
    precise float r = sb ? dot(vec3(acc, b, s), vec3(1.0, xs, qx)) : dot(vec3(acc, s, b), vec3(1.0, qx, xs));
    return r;
}

// 4-bit register-A body: RR weight rows per subgroup (2 up to m = 2, 4 from
// m = 3), a uint4 block of 32 codes of each per lane and step, and a last
// partial pass split between the half-subgroups. Requires M == MB, GS % 32
// == 0, K % 32 == 0 and N * K / 8 < 2^32 (32-bit code word offsets).
#ifdef SKINNY_REG_Q4
#define MB SKINNY_REG_Q4
#if MB <= 2
#define RR 2
#else
#define RR 4
#endif
#define HR (RR / 2)

// Adds a code word of a block to qx[r][i] (row r's codes dotted with
// activation row i) and xs[i] (activation row i summed), for NR rows with
// code words wv[r] and the word's activations at element `at`. Half a word
// (four elements) at a time. (Metal first spreads a word's nibbles into
// bytes for its conversions; here a code is one bit-field extract.)
#define SKINNY_Q4_WORD(NR, wv, at)                                                            \
    {                                                                                         \
        uvec4 raw[MB];                                                                        \
        SKINNY_LOAD_ROWS(raw, at)                                                             \
        [[unroll]] for (uint h = 0; h < 2; ++h) {                                             \
            vec4 x[MB];                                                                       \
            [[unroll]] for (uint i = 0; i < MB; ++i) {                                        \
                x[i] = bf4_bits(h == 0 ? raw[i].xy : raw[i].zw);                              \
                xs[i] = xs[i] + skinny_sum4(x[i]);                                            \
            }                                                                                 \
            [[unroll]] for (uint r = 0; r < NR; ++r) {                                        \
                vec4 q = h == 0 ? q4_lo(wv[r]) : q4_hi(wv[r]);                                \
                [[unroll]] for (uint i = 0; i < MB; ++i) qx[r][i] = qx[r][i] + skinny_dot4(q, x[i]); \
            }                                                                                 \
        }                                                                                     \
    }

// qx and xs over block `blk` of NR rows whose code words start at woff[r].
// As on Metal the word loop is unrolled only while the body is small (NR x MB
// <= 8); unrolled, a row's four code words are one 16-byte load.
#define SKINNY_Q4_BLOCK(NR, woff, blk)                                                        \
    {                                                                                         \
        [[unroll]] for (uint i = 0; i < MB; ++i) {                                            \
            xs[i] = 0.0;                                                                      \
            [[unroll]] for (uint r = 0; r < NR; ++r) qx[r][i] = 0.0;                          \
        }                                                                                     \
        if (NR * MB <= 8) {                                                                   \
            uvec4 w4[NR];                                                                     \
            [[unroll]] for (uint r = 0; r < NR; ++r) w4[r] = U32x4(arg(0)).v[woff[r] / 4 + (blk)]; \
            [[unroll]] for (uint wi = 0; wi < 4; ++wi) {                                      \
                uint wv[NR];                                                                  \
                [[unroll]] for (uint r = 0; r < NR; ++r) wv[r] = w4[r][wi];                   \
                SKINNY_Q4_WORD(NR, wv, (blk) * 32 + wi * 8)                                   \
            }                                                                                 \
        } else {                                                                              \
            [[dont_unroll]] for (uint wi = 0; wi < 4; ++wi) {                                 \
                uint wv[NR];                                                                  \
                [[unroll]] for (uint r = 0; r < NR; ++r) wv[r] = codes.v[woff[r] + (blk) * 4 + wi]; \
                SKINNY_Q4_WORD(NR, wv, (blk) * 32 + wi * 8)                                   \
            }                                                                                 \
        }                                                                                     \
    }

void main() {
    U32 codes = U32(arg(0));
    BF16 scales = BF16(arg(1)), biases = BF16(arg(2));
    uint64_t a = arg(3), c = arg(4);
    uint K = arg_u32(5), N = arg_u32(6), GS = arg_u32(7);
    uint lane = thread_index_in_simdgroup;
    uint row0 = (threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup) * RR;
    if (row0 >= N) return; // whole subgroup
    uint words = K / 8, blocks = words / 4, gs_log2 = uint(findLSB(GS));
    uint gshift = gs_log2 - 5, groups = K >> gs_log2;
    // Rows past N repeat the last row (the loads stay in bounds) and are not
    // stored.
    uint rows[RR], woff[RR];
    [[unroll]] for (uint r = 0; r < RR; ++r) {
        rows[r] = min(row0 + r, N - 1);
        woff[r] = rows[r] * words;
    }

    precise float acc[RR][MB];
    [[unroll]] for (uint r = 0; r < RR; ++r) {
        [[unroll]] for (uint i = 0; i < MB; ++i) acc[r][i] = 0.0;
    }

    uint rem = blocks % 32;
    bool split = rem != 0 && rem <= 16;
    uint full = split ? blocks - rem : blocks;
    for (uint blk = lane; blk < full; blk += 32) {
        uint g = blk >> gshift;
        precise float qx[RR][MB];
        precise float xs[MB];
        SKINNY_Q4_BLOCK(RR, woff, blk)
        [[unroll]] for (uint r = 0; r < RR; ++r) {
            float s = float(scales.v[rows[r] * groups + g]);
            float b = float(biases.v[rows[r] * groups + g]);
            bool sb = MB == 1 || (r & 1) != 0;
            [[unroll]] for (uint i = 0; i < MB; ++i) acc[r][i] = skinny_fold(sb, s, qx[r][i], b, xs[i], acc[r][i]);
        }
    }
    if (split) {
        // Lanes l and l + 16 both take block full + l % 16: the lower half
        // for rows [0, RR/2), the upper half for rows [RR/2, RR) on behalf of
        // lane l, which owns all of them. Lanes without a block compute the
        // pass's first block and discard it, keeping the shuffles uniform.
        uint hv = lane / 16;
        bool has_block = lane % 16 < rem;
        uint blk = has_block ? full + lane % 16 : full;
        uint g = blk >> gshift;
        uint hrows[HR], hoff[HR];
        [[unroll]] for (uint r = 0; r < HR; ++r) {
            hrows[r] = hv == 0 ? rows[r] : rows[HR + r];
            hoff[r] = hv == 0 ? woff[r] : woff[HR + r];
        }
        precise float qx[HR][MB];
        precise float xs[MB];
        SKINNY_Q4_BLOCK(HR, hoff, blk)
        [[unroll]] for (uint r = 0; r < HR; ++r) {
            float s = float(scales.v[hrows[r] * groups + g]);
            float b = float(biases.v[hrows[r] * groups + g]);
            bool sb = MB == 1 || ((hv * HR + r) & 1) != 0;
            [[unroll]] for (uint i = 0; i < MB; ++i) {
                // Metal's shuffle up and down by 16: the other half's lane.
                float up = subgroupShuffleXor(acc[HR + r][i], 16);
                float from = hv == 0 ? acc[r][i] : up;
                float to = has_block ? skinny_fold(sb, s, qx[r][i], b, xs[i], from) : from;
                float back = subgroupShuffleXor(to, 16);
                if (hv == 0 && has_block) {
                    acc[r][i] = to;
                    acc[HR + r][i] = back;
                }
            }
        }
    }

    // The subgroup's RR x MB sums leave in one store, lane j carrying output
    // j (activation row j / RR, weight row j % RR; RR x MB is at most a
    // subgroup's 32 lanes). Metal's lane 0 stores each sum as it is reduced;
    // here every such store is a branch of its own for the whole subgroup,
    // and at m = 3 the twelve of them cost about a twentieth of a short
    // subgroup's time. Measured 2026-11-11 in chains of barriered levels at
    // m = 3 (`q4_reg_variants_probe`): the GDN input projection (4120
    // subgroups of 2.5 blocks a lane) 65.6 to 63.6 us per level, the output
    // projection (640 subgroups of 6) 27.4 to 27.2, the same bits.
    float out_sum = 0.0;
    [[unroll]] for (uint i = 0; i < MB; ++i) {
        [[unroll]] for (uint r = 0; r < RR; ++r) {
            float sum = subgroupAdd(acc[r][i]);
            if (lane == i * RR + r) out_sum = sum;
        }
    }
    uint out_row = row0 + lane % RR;
    if (lane < RR * MB && out_row < N) SKINNY_STORE(c, lane / RR, N, out_row, out_sum);
}
#endif
