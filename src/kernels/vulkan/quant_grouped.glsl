// The grouped Q4 GEMM of quant.metal on the matrix units (matmul.glsl): the
// body of quant.comp's kernels. Before including, define QG_BM, the block
// map's row tile (32 or 64). K must be a multiple of 64 and GS a power of
// two, at least 64 (the Rust wrapper passes 64).
//
// A threadgroup of four subgroups computes a QG_BM x 64 tile of C. Where the
// port differs from Metal's kernel (one matmul2d per 128-deep K step over
// the tile, A from device memory, the B tile dequantized by weight rows into
// threadgroup memory), and why. Ms per layer of gate and up, and of down, at
// the model's shapes and 4 096 tokens (`grouped_expert_gemm_timing`,
// 2026-11-10); Metal 6.79 and 3.60; the first port, which staged A in
// threadgroup memory too and gave a subgroup 16 columns of every block of
// A, 14.4 and 7.3 (kept as the tests' reference,
// tests/vulkan/quant_grouped_first.glsl: the two match bit for bit):
//
// - A subgroup owns 16 of the tile's rows and all 64 columns: one block of
//   A against four of B. A block of A is then read by one subgroup, straight
//   from device memory (matmul.glsl, MAT_A_OF_ROWS: two 16-byte loads per
//   lane and K step), and the B tile, which every subgroup needs whole, is
//   the one operand in threadgroup memory. No staging of A, half the
//   barriers.
// - A subgroup whose rows are all past the tile's multiplies nothing: a test
//   outside the 16-deep steps. (Metal picks the product's height among BM,
//   BM / 2 and BM / 4; the first port ran every block, a branch per block
//   having cost it 30%.) Giving the idle subgroups a share of the busy ones'
//   columns instead (each subgroup rows 16 (s % H).., 64 / H columns, for H
//   blocks of rows) is slower, 12.6 and 13.1 ms at H from 2 and from 1: the
//   same multiplies with more loads of A. What counts is the work, not how
//   long a threadgroup stays.
// - The B tile is stored in the lanes' order, one 16-byte vector per lane
//   and 16-deep step: a B operand is one load from threadgroup memory
//   (MAT_OF_WORDS) where a tile stored by weight rows needs four. A load
//   from or a store to threadgroup memory costs a quarter of a multiply, so
//   they are what to count: per subgroup and 64 of K, 4 stores and 16 loads
//   against the first port's 16 and 48. A thread therefore dequantizes
//   across weight rows (codes k .. k + 8 of four rows, a scale and a bias
//   each), the arithmetic per weight unchanged.
// - K steps are 64 deep with two tiles in turn, so one barrier per step: the
//   threads that write the next step's tile cannot overrun a subgroup still
//   reading this step's. 11.9 and 5.8 ms; with one tile and two barriers
//   12.4; 128-deep steps (16 KB per tile) 14.1 with one tile and 19.3 with
//   two, 32-deep 14.6.
// - A 16-deep step issues its six loads before it uses any (QG_K_LOOP), and
//   a lane's four columns of C are one store: 10.5 and 5.0 ms.
//
// Where the time is (ablations of the kernel at 11.9 ms): the multiplies and
// their moves 6.0, at the matrix units' rate for the blocks that hold rows;
// the dequantization 4.6, about 8.5 instructions per weight (extract,
// convert, multiply-add, three to round to bfloat, 1.5 to pack two into a
// word), of which Metal's bfloat conversion would save the three; the loads
// 1.4.
#include "matmul.glsl"

// One K step.
#define QG_STEP 64
// A step's B tile in the lanes' order: vector (k - k0) / 2 * 16 + n / 4 holds
// weights k and k + 1 of the tile's weight rows n .. n + 3, which are a
// lane's eight elements of a B operand (matmul.glsl): rows (k - k0) % 16 and
// the next of the step's block (k - k0) / 16, columns n % 16 .. + 4 of
// column block n / 16. Two tiles, for the even steps and the odd ones.
shared uvec4 qg_tile[2 * QG_STEP * 8];

// Unused: quant.comp still defines the first port's qg_fill after this file
// and it has to compile. The dequantization is qg_dequant.
void qg_store_word(uint word, float s, float b, uint at) {}

// bf16_weight's rounding (quant_common.glsl) with the bfloat left in the
// word's high half, and two such bfloats as one word of the tile.
uvec4 qg_round(vec4 w) {
    uvec4 u = floatBitsToUint(w);
    return u + 0x7FFFu + ((u >> 16) & 1u);
}
uint qg_pair(uint first, uint second) { return (first >> 16) | (second & 0xFFFF0000u); }

// Thread t dequantizes weights k0 + 8 (t / 16) .. + 8, one code word, of the
// tile's weight rows 4 (t % 16) .. + 4 into four vectors of the tile at
// `tile_at`. `codes` are the code words of the tile's first weight row,
// b_row0.
void qg_dequant(uint64_t codes, BF16 scales, BF16 biases, uint words, uint groups, uint g_shift,
                uint b_row0, uint k0, uint tid, uint tile_at) {
    uint quad = tid & 15, k = k0 + 8 * (tid >> 4);
    uint p = (b_row0 + 4 * quad) * groups + (k >> g_shift);
    vec4 s = vec4(float(scales.v[p]), float(scales.v[p + groups]), float(scales.v[p + 2 * groups]),
                  float(scales.v[p + 3 * groups]));
    vec4 b = vec4(float(biases.v[p]), float(biases.v[p + groups]), float(biases.v[p + 2 * groups]),
                  float(biases.v[p + 3 * groups]));
    U32 code = U32(codes);
    uint c = 4 * quad * words + k / 8;
    uvec4 word = uvec4(code.v[c], code.v[c + words], code.v[c + 2 * words], code.v[c + 3 * words]);
    uint at = tile_at + (k - k0) * 8 + quad;
    [[unroll]] for (uint i = 0; i < 4; ++i) {
        uvec4 w0 = qg_round(vec4((word >> (8 * i)) & 0xFu) * s + b);
        uvec4 w1 = qg_round(vec4((word >> (8 * i + 4)) & 0xFu) * s + b);
        qg_tile[at + 16 * i] =
            uvec4(qg_pair(w0.x, w0.y), qg_pair(w0.z, w0.w), qg_pair(w1.x, w1.y), qg_pair(w1.z, w1.w));
    }
}

#define QG_EACH_BLOCK(X) X(0) X(1) X(2) X(3)
#define QG_LOAD(i) uvec4 b##i = qg_tile[b_at + 4 * i];
#define QG_MULTIPLY(i)                                                                        \
    MAT_OF_WORDS(mb, b##i)                                                                    \
    acc##i = coopMatMulAdd(ma, mb, acc##i);
// The K loop; `vector(at)` loads the uvec4 at vector `at` of the tile's
// rows of A. A 16-deep step's six loads are issued before any is used: a
// thread waits where it first uses a loaded value, for all in flight
// (matmul.glsl). 12.4 ms with each operand loaded where it is multiplied,
// 10.5 this way. (Issuing a step's loads ahead of the multiplies of the step
// before, the four steps written out, as the dense GEMM does: no further.)
#define QG_K_LOOP(vector)                                                                     \
    for (uint k0 = 0; k0 < K; k0 += QG_STEP) {                                                \
        uint tile_at = (k0 / QG_STEP & 1) * (QG_STEP * 8);                                    \
        qg_dequant(codes, scales, biases, words, groups, g_shift, b_row0, k0, tid, tile_at);  \
        barrier();                                                                            \
        if (working) {                                                                        \
            for (uint kk = 0; kk < QG_STEP; kk += 16) {                                       \
                uint at = (k0 + kk) / 8 + a_half;                                             \
                uint b_at = tile_at + lane_at + kk * 8;                                       \
                uvec4 v0 = vector(a_at0 + at), v1 = vector(a_at1 + at);                       \
                QG_EACH_BLOCK(QG_LOAD)                                                        \
                MatA ma;                                                                      \
                MatB mb;                                                                      \
                MAT_A_OF_ROWS(ma, v0, v1, odd16)                                              \
                QG_EACH_BLOCK(QG_MULTIPLY)                                                    \
            }                                                                                 \
        }                                                                                     \
    }
#define QG_VECTOR_WIDE(at) aw.v[at]
#define QG_VECTOR_NARROW(at) load_raw8(a, 8 * (at), false)
// A lane holds four consecutive columns of two rows of a block: one store
// each.
#define QG_STORE(i)                                                                           \
    [[unroll]] for (uint h = 0; h < 2; ++h) {                                                 \
        uint row = row0 + lane_row + h;                                                       \
        if (row < rows) {                                                                     \
            vec4 v = vec4(acc##i[4 * h], acc##i[4 * h + 1], acc##i[4 * h + 2], acc##i[4 * h + 3]); \
            if (c_wide) {                                                                     \
                MAT_STORE_HALF(c_rows, row * N + col + 16 * i, v);                            \
            } else {                                                                          \
                MAT_STORE_HALF_UNALIGNED(c_rows, row * N + col + 16 * i, v);                  \
            }                                                                                 \
        }                                                                                     \
    }

void main() {
    uint64_t codes = arg(0), a = arg(3), c = arg(4);
    BF16 scales = BF16(arg(1)), biases = BF16(arg(2));
    uvec4 blk = U32x4(arg(5)).v[threadgroup_position_in_grid.x];
    if (blk.x >= blk.w) return; // sentinel entry from the GPU-built block map
    uint K = arg_u32(6), N = arg_u32(7), GS = arg_u32(8);
    uint m0 = blk.x, b_row0 = blk.y, n0 = blk.z, m_end = blk.w;
    uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    // `words` is also the uvec4 of a row of A: K / 8 either way.
    uint words = K / 8, groups = K / GS, g_shift = findMSB(GS);

    uint rows = min(m_end - m0, uint(QG_BM));
    a += uint64_t(m0) * (2 * K);
    codes += uint64_t(b_row0) * (4 * words);
    // A at an address the 16-byte loads cannot take is read two bytes at a
    // time.
    bool a_wide = (a & 3) == 0;
    U32x4 aw = U32x4(a);

    // The subgroup's rows of the tile, and its lanes' two rows of A: a row
    // past the tile's reads the tile's last (and is not stored).
    uint row0 = 16 * sg;
    bool working = row0 < rows;
    uint lane_row = mat_lane_row(), a_half = mat_a_half(), odd16 = 16 * mat_a_odd();
    uint a_at0 = min(row0 + lane_row, rows - 1) * words, a_at1 = min(row0 + lane_row + 1, rows - 1) * words;
    // The lane's vector of the tile's first 16-deep step and column block.
    uint lane_at = lane_row * 8 + mat_lane_col() / 4;

    MatC acc0 = MatC(0.0), acc1 = MatC(0.0), acc2 = MatC(0.0), acc3 = MatC(0.0);
    if (a_wide) {
        QG_K_LOOP(QG_VECTOR_WIDE)
    } else {
        QG_K_LOOP(QG_VECTOR_NARROW)
    }

    if (working) {
        uint64_t c_rows = c + uint64_t(m0) * (2 * N);
        bool c_wide = (c_rows & 3) == 0;
        uint col = n0 + mat_lane_col();
        QG_EACH_BLOCK(QG_STORE)
    }
}
