// The grouped Q4 GEMM of quant.metal on the matrix units (matmul.glsl): the
// body quant.comp's kernels and the reference kernels of
// tests/vulkan/quant_test.comp share. Before including, define
//
//     QG_BM   the block map's row tile (32 or 64),
//     QG_BK   the K step (a multiple of 64),
//     QG_LD   the B tile's row stride in elements (a multiple of 4);
//
// after it, define qg_fill, which dequantizes one K step of the expert's 64
// weight rows into qg_tile (qg_store_word).
//
// A threadgroup of four subgroups computes a QG_BM x 64 tile of C: subgroup s
// its columns 16 s.., in 16-row blocks, as many as the tile has rows for
// (where Metal picks the product's height among BM, BM/2 and BM/4).
//
// Unlike Metal's, the product does not read A from device memory: the
// threadgroup stages its rows 64 elements of K at a time, as it does B's.
// Every load from a device address costs a dozen instructions on this driver
// (its sparse-residency guard), and an operand block is two rows per lane, so
// a lane loading its own elements pays that 8 times per K step of 16 and
// block where staging pays it once per 64 and row tile; measured on the
// model's shapes, 24.4 against 16.7 ms per layer of gate and up.
#include "matmul.glsl"

// The B tile, 64 rows of QG_LD elements. As bf16vec4 a thread stores four
// dequantized weights at once; coopMatLoad counts in these vectors.
shared bf16vec4 qg_tile[64 * QG_LD / 4];
// The A tile: QG_BM rows of 64 elements, zero past the tile's rows. A lane's
// elements of an operand block are four of a row, every other one of eight
// (mat_row, mat_a_col), so each eight are stored as two vectors: the even
// elements, then the odd ones.
shared bf16vec4 qg_a[QG_BM * 16];

// Dequantizes one Q4 word into the B tile, at vector `at`.
void qg_store_word(uint word, float s, float b, uint at) {
    qg_tile[at] = bf16_weight(q4_lo(word) * s + b);
    qg_tile[at + 1] = bf16_weight(q4_hi(word) * s + b);
}

// `codes` are the code words of the tile's first weight row, b_row0.
void qg_fill(uint64_t codes, BF16 scales, BF16 biases, uint words, uint groups, uint GS,
             uint b_row0, uint k0, uint depth, uint tid);

// The column of an A block a lane's element e holds, as mat_col is an
// accumulator's (its row is mat_row(e)): the driver's layout of the operand.
uint mat_a_col(uint e) {
    uint lane = gl_SubgroupInvocationID;
    return ((lane >> 3) & 1) | ((lane & 1) << 3) | ((e & 3) << 1);
}

#define QG_BLOCKS (QG_BM / 16)

void main() {
    uint64_t codes = arg(0), a = arg(3), c = arg(4);
    BF16 scales = BF16(arg(1)), biases = BF16(arg(2));
    uvec4 blk = U32x4(arg(5)).v[threadgroup_position_in_grid.x];
    if (blk.x >= blk.w) return; // sentinel entry from the GPU-built block map
    uint K = arg_u32(6), N = arg_u32(7), GS = arg_u32(8);
    uint m0 = blk.x, b_row0 = blk.y, n0 = blk.z, m_end = blk.w;
    uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    uint words = K / 8, groups = K / GS;

    uint rows = min(m_end - m0, uint(QG_BM));
    a += uint64_t(m0) * (2 * K);
    codes += uint64_t(b_row0) * (4 * words);
    bool a_wide = (a & 3) == 0;
    // The lane's first vector of a block in the A tile.
    uint lane_at = mat_row(0) * 16 + mat_a_col(0) / 8 * 2 + (mat_a_col(0) & 1);

    for (uint i = tid; i < QG_BM * 16; i += 128) {
        if (i >= rows * 16) qg_a[i] = bf16vec4(0.0);
    }

    MatC acc[QG_BLOCKS];
    [[unroll]] for (uint i = 0; i < QG_BLOCKS; ++i) acc[i] = MatC(0.0);
    for (uint k0 = 0; k0 < K; k0 += QG_BK) {
        uint depth = min(uint(QG_BK), K - k0);
        qg_fill(codes, scales, biases, words, groups, GS, b_row0, k0, depth, tid);
        for (uint k1 = 0; k1 < depth; k1 += 64) {
            for (uint i = tid; i < rows * 8; i += 128) {
                uvec4 v = load_raw8(a, (i / 8) * K + k0 + k1 + (i % 8) * 8, a_wide);
                qg_a[2 * i] = uintBitsToBFloat16EXT(u16vec4(v));
                qg_a[2 * i + 1] = uintBitsToBFloat16EXT(u16vec4(v >> 16));
            }
            barrier();
            for (uint kk = k1; kk < k1 + 64; kk += 16) {
                MatB mb;
                MAT_LOAD_B(mb, qg_tile, 16 * sg * (QG_LD / 4) + kk / 4, QG_LD / 4);
                [[unroll]] for (uint i = 0; i < QG_BLOCKS; ++i) {
                    MatA ma;
                    uint at = 16 * 16 * i + lane_at + (kk - k1) / 4;
                    [[unroll]] for (uint h = 0; h < 2; ++h) {
                        bf16vec4 v = qg_a[at + 16 * h];
                        [[unroll]] for (uint j = 0; j < 4; ++j) ma[4 * h + j] = v[j];
                    }
                    acc[i] = coopMatMulAdd(ma, mb, acc[i]);
                }
            }
            barrier();
        }
    }

    // A lane holds four consecutive columns of two rows of a block
    // (mat_row, mat_col): one store each.
    BF16x4 c_rows = BF16x4(c + uint64_t(m0) * (2 * N));
    uint col = n0 + 16 * sg + mat_col(0);
    [[unroll]] for (uint i = 0; i < QG_BLOCKS; ++i) {
        [[unroll]] for (uint h = 0; h < 2; ++h) {
            uint row = 16 * i + mat_row(4 * h);
            if (row < rows) {
                c_rows.v[(row * N + col) / 4] =
                    bf16(vec4(acc[i][4 * h], acc[i][4 * h + 1], acc[i][4 * h + 2], acc[i][4 * h + 3]));
            }
        }
    }
}
