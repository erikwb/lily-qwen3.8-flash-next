// The GPU's matrix units, which the Metal kernels reach through Metal 4
// tensor ops (MetalPerformancePrimitives matmul2d): here 16x16x16 cooperative
// matrices, one per subgroup, with bfloat operands and an fp32 accumulator:
//
//     MatC c = MatC(0.0);
//     for (k0 = 0; k0 < K; k0 += 16) { load a, b; c = coopMatMulAdd(a, b, c); }
//
// computes C[16, 16] += A[16, 16 of K] * B[16 of K, 16]. A threadgroup of S
// subgroups runs S blocks at a time; a subgroup holds as many accumulators as
// it has blocks of the tile.
//
// Include after common.glsl. A kernel that stages edges declares
// `shared bfloat16_t mat_stage[MAT_STAGE_ELEMENTS(subgroups)];`.
//
// What a matrix kernel costs here (Honeykrisp on AGX3, measured 2026-11-10
// on the dense GEMM at M=4096, N=K=2560, `gemm_nax_timing`): the multiplies
// alone, operands in registers, run at 57 TFLOP/s; what a kernel adds to that
// is how its operands reach the registers.
//
// - `coopMatLoad` from device memory multiplies row and stride in 64 bits for
//   every load and reads an A block as eight 2-byte loads per lane, a B block
//   as four 4-byte ones: 11 TFLOP/s. The loaders below index a
//   loop-invariant base with 32 bits and load 16 bytes where the layout
//   allows.
// - The cost is the loads, by count and by the bytes they stream, not the
//   register moves that assemble an operand (300 moves per 16 multiplies
//   cost 9%). Per lane and four multiplies, one 16-byte load from device
//   memory costs 13% of the multiplies' time and two 21%, running beside the
//   matrix units; five cost as much as the multiplies and ten three times as
//   much. Five loads from threadgroup memory cost 28%. A store to
//   threadgroup memory costs about what a load from it does.
// - So: reuse operands (a subgroup with a x b accumulators loads a + b
//   operands for a b multiplies; eight accumulators are 64 registers), give
//   every lane whole 16-byte vectors it uses all of (MAT_K64), and keep
//   branches out of the K loop (clamp indices at the edges instead).
// - Loads are issued in source order and a thread waits at the first use of
//   any loaded value, for all that are in flight (up to nine, and never
//   across a branch). So write every load of a step, of both operands and
//   from both memories, into variables before the first is used, and pass
//   the variables to the macros below: the grouped GEMM 12.4 to 10.5 ms,
//   attention over three rows 2.09 to 1.61. Loads issued ahead of work that
//   does not use them stay in flight through it: the dense GEMM issues a
//   column block's loads before the block before it multiplies (41 to 44
//   TFLOP/s).
// - Where a kernel's time is a chain of steps (few threadgroups, each a long
//   loop), a step should be one wait and no arithmetic between its loads and
//   its multiply. An operand whose bfloats are loaded as its halves needs
//   none; MAT_A_OF_ROWS shifts the lane's even or odd bfloats out, the
//   compiler puts the shifts between the step's loads, and there coopMatLoad
//   (eight 2-byte loads in a row straight into the operand's registers) is
//   the faster: gdn_chunk_scan 1.64 ms a dispatch with it, 2.2 with the
//   16-byte loads.
// - A bf16vec4 through a 2-byte-aligned reference (BF16x4) is four loads or
//   four stores: move four bfloats as two words (MAT_HALF, MAT_STORE_HALF).
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#pragma use_vulkan_memory_model

#define MatA coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MatB coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MatC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

// How a 16x16 matrix lies over the 32 lanes of a subgroup (the driver's
// layout, agx_nir_lower_cmat.c; the GEMM tests would catch it changing). A
// lane holds eight elements, m[e] for e in 0..8, in four 32-bit registers:
// element e is the low (e even) or high half of register e / 2, so a 16-byte
// load whose bfloats are in element order is an operand as it stands.
//
//               rows                        columns
//   A operand   mat_lane_row() + (e >> 2)   8 mat_a_half() + mat_a_odd() + 2 (e & 3)
//   B operand   mat_lane_row() + (e >> 2)   mat_lane_col() + (e & 3)
//   accumulator mat_lane_row() + (e >> 2)   mat_lane_col() + (e & 3)
//
// Every lane holds two adjacent rows (the first even). Of a B operand or an
// accumulator it holds four adjacent columns of each; of an A operand every
// other column of one eight-column half. In C = A * B the columns of A and
// the rows of B are the K index.
uint mat_lane_row() {
    uint lane = gl_SubgroupInvocationID;
    return (lane & 6) | ((lane >> 1) & 8);
}
uint mat_lane_col() {
    uint lane = gl_SubgroupInvocationID;
    return ((lane & 1) << 2) | (lane & 8);
}
uint mat_a_half() { return gl_SubgroupInvocationID & 1; }
uint mat_a_odd() { return (gl_SubgroupInvocationID >> 3) & 1; }
// Element e's row, and its column in a B operand or an accumulator
// (mat_col) or in an A operand (mat_a_col).
uint mat_row(uint e) { return mat_lane_row() | (e >> 2); }
uint mat_col(uint e) { return mat_lane_col() + (e & 3); }
uint mat_a_col(uint e) { return mat_a_odd() | (mat_a_half() << 3) | ((e & 3) << 1); }

// ---- coopMatLoad -----------------------------------------------------------
// A block straight from memory, every element of it inside its tensor:
// `rows` is a bfloat16_t array (BF16(address).v, or a shared one), `at` the
// index of the block's first element and `stride` the elements between rows.
// A's rows are the block's rows; B's rows are its columns. Slow from device
// memory (above); fine from threadgroup memory for a B operand whose rows
// are the array's (MAT_LOAD_B_ROWS: two 8-byte loads per lane).
#define MAT_LOAD_A(a, rows, at, stride) coopMatLoad(a, rows, at, stride, gl_CooperativeMatrixLayoutRowMajor)
#define MAT_LOAD_B(b, rows, at, stride) coopMatLoad(b, rows, at, stride, gl_CooperativeMatrixLayoutColumnMajor)
#define MAT_LOAD_B_ROWS(b, rows, at, stride) coopMatLoad(b, rows, at, stride, gl_CooperativeMatrixLayoutRowMajor)

// ---- Operands from wide loads ---------------------------------------------
// The loaders below take the memory as an array of words: `U32x4(address).v`
// (16 bytes, eight bfloats, per element), `U32x2(address).v` (four bfloats)
// or `U32(address).v` (two), or a `shared` array of uvec4, uvec2 or uint.
// The address must be a multiple of 4 and the rows a whole number of vectors
// apart; index with 32 bits from an address computed once (a 64-bit index
// costs three instructions per add). The caller computes the lane's indices,
// which is where an edge is made safe: clamp a row past the tensor to its
// last one (its products are not stored), so no load leaves the tensor and
// no branch enters the K loop.
#define MAT_LO(w) uintBitsToBFloat16EXT(uint16_t(w))
#define MAT_HI(w) uintBitsToBFloat16EXT(uint16_t((w) >> 16))

// Bfloats `at` .. of the bfloat tensor at `address` as words: two (a pair),
// four (a half) or eight (a vector) of them, `at` a multiple of as many and
// the address of 4. The _UNALIGNED forms move two bytes at a time, for a
// tensor at any address: a kernel tests its addresses once and runs its
// loop with one set of loaders or the other (a test at every load would end
// the basic block there, and the loads would no longer be in flight
// together).
#define MAT_PAIR(address, at) U32(address).v[(at) / 2]
#define MAT_HALF(address, at) U32x2(address).v[(at) / 4]
#define MAT_VECTOR(address, at) U32x4(address).v[(at) / 8]
#define MAT_PAIR_UNALIGNED(address, at) (uint(U16(address).v[at]) | (uint(U16(address).v[(at) + 1]) << 16))
#define MAT_HALF_UNALIGNED(address, at) uvec2(MAT_PAIR_UNALIGNED(address, at), MAT_PAIR_UNALIGNED(address, (at) + 2))
#define MAT_VECTOR_UNALIGNED(address, at)                                                     \
    uvec4(MAT_PAIR_UNALIGNED(address, at), MAT_PAIR_UNALIGNED(address, (at) + 2),             \
          MAT_PAIR_UNALIGNED(address, (at) + 4), MAT_PAIR_UNALIGNED(address, (at) + 6))
// To store: four floats rounded to bfloats `at` .. of the tensor at
// `address`, in one 8-byte store (or four of two bytes).
uvec2 mat_bf16_words(vec4 f) {
    // common.glsl's bf16(vec4) with each bfloat left in its word's high half.
    uvec4 u = floatBitsToUint(f);
    uvec4 r = u + 0x7FFFu + ((u >> 16) & 1u);
    r = mix(r, u, greaterThan(u & 0x7FFFFFFFu, uvec4(0x7F800000u)));
    return uvec2((r.x >> 16) | (r.y & 0xFFFF0000u), (r.z >> 16) | (r.w & 0xFFFF0000u));
}
#define MAT_STORE_HALF(address, at, f) U32x2(address).v[(at) / 4] = mat_bf16_words(f)
#define MAT_STORE_HALF_UNALIGNED(address, at, f) BF16x4(address).v[(at) / 4] = bf16(f)

// A lane's eight elements of any matrix from 16 bytes that hold them in
// element order (memory written for the purpose: a tile in threadgroup
// memory laid out by lanes, quant_grouped.glsl): one load per operand.
#define MAT_OF_WORDS(m, v)                                                                    \
    {                                                                                         \
        uvec4 mat_v = v;                                                                      \
        m[0] = MAT_LO(mat_v[0]); m[1] = MAT_HI(mat_v[0]); m[2] = MAT_LO(mat_v[1]); m[3] = MAT_HI(mat_v[1]); \
        m[4] = MAT_LO(mat_v[2]); m[5] = MAT_HI(mat_v[2]); m[6] = MAT_LO(mat_v[3]); m[7] = MAT_HI(mat_v[3]); \
    }

// The same from two 8-byte halves, the lane's four elements of its first row
// and of its second.
#define MAT_OF_HALVES(m, h0, h1)                                                              \
    {                                                                                         \
        uvec2 mat_h0 = h0, mat_h1 = h1;                                                       \
        m[0] = MAT_LO(mat_h0.x); m[1] = MAT_HI(mat_h0.x); m[2] = MAT_LO(mat_h0.y); m[3] = MAT_HI(mat_h0.y); \
        m[4] = MAT_LO(mat_h1.x); m[5] = MAT_HI(mat_h1.x); m[6] = MAT_LO(mat_h1.y); m[7] = MAT_HI(mat_h1.y); \
    }

// A B block from memory whose rows are the block's rows (16 columns
// contiguous in a row: the right operand of C = A * B). `h0` and `h1` are
// the uvec2 holding the lane's four columns, bfloats mat_lane_col() .. + 4,
// of row mat_lane_row() and of the next: every byte loaded is used.
#define MAT_B_OF_ROWS(m, h0, h1) MAT_OF_HALVES(m, h0, h1)

// An A block from memory whose rows are the block's (16 columns of K
// contiguous in a row), in K order. `v0` and `v1` are the uvec4 holding the
// lane's half of its two rows: bfloats 8 mat_a_half() .. + 8 of row
// mat_lane_row() of the block, and of the next row. `odd16` is
// 16 * mat_a_odd(), taken once outside the loop: a lane uses a vector's even
// bfloats or its odd ones, so two lanes load each vector.
#define MAT_A_OF_ROWS(m, v0, v1, odd16)                                                       \
    {                                                                                         \
        uvec4 mat_v0 = v0, mat_v1 = v1;                                                       \
        mat_v0 >>= (odd16);                                                                   \
        mat_v1 >>= (odd16);                                                                   \
        m[0] = MAT_LO(mat_v0[0]); m[1] = MAT_LO(mat_v0[1]); m[2] = MAT_LO(mat_v0[2]); m[3] = MAT_LO(mat_v0[3]); \
        m[4] = MAT_LO(mat_v1[0]); m[5] = MAT_LO(mat_v1[1]); m[6] = MAT_LO(mat_v1[2]); m[7] = MAT_LO(mat_v1[3]); \
    }
// The same from threadgroup memory staged for it: `e0` and `e1` are the
// bf16vec4 of the lane's four elements of its two rows (a tile stored with
// a row's even elements and its odd ones apart).
#define MAT_A_OF_ELEMENTS(m, e0, e1)                                                          \
    {                                                                                         \
        bf16vec4 mat_e0 = e0, mat_e1 = e1;                                                    \
        m[0] = mat_e0[0]; m[1] = mat_e0[1]; m[2] = mat_e0[2]; m[3] = mat_e0[3];               \
        m[4] = mat_e1[0]; m[5] = mat_e1[1]; m[6] = mat_e1[2]; m[7] = mat_e1[3];               \
    }

// A B block from memory whose rows are the block's columns (the right
// operand of C = A * B^T, B^T row-major; or a left operand A^T by
// MAT_A_OF_PAIRS), in K order. `p0`..`p3` are the words holding the lane's
// two rows, K = mat_lane_row() and the next, of its four columns: bfloats
// mat_lane_row(), + 1 of memory rows mat_lane_col() + j. Four 4-byte loads
// per operand: the loads this layout cannot avoid without MAT_K64.
#define MAT_B_OF_PAIRS(m, p0, p1, p2, p3)                                                     \
    {                                                                                         \
        uint mat_p0 = p0, mat_p1 = p1, mat_p2 = p2, mat_p3 = p3;                              \
        m[0] = MAT_LO(mat_p0); m[1] = MAT_LO(mat_p1); m[2] = MAT_LO(mat_p2); m[3] = MAT_LO(mat_p3); \
        m[4] = MAT_HI(mat_p0); m[5] = MAT_HI(mat_p1); m[6] = MAT_HI(mat_p2); m[7] = MAT_HI(mat_p3); \
    }
// An A block from memory whose rows are the block's columns: `p0`..`p3` the
// words holding bfloats mat_lane_row(), + 1 of memory rows
// 8 mat_a_half() + mat_a_odd() + 2 j.
#define MAT_A_OF_PAIRS(m, p0, p1, p2, p3) MAT_B_OF_PAIRS(m, p0, p1, p2, p3)

// An A block from a bfloat16_t array in threadgroup memory, as fast as a B
// block is: a row's sixteen columns stored at their places mat_a_place(c),
// where a lane's four are neighbours (the place of column c of any block
// of 16 is mat_a_place(c)), and loaded as a B operand's rows. Two 8-byte
// loads where MAT_LOAD_A takes eight of two bytes.
uint mat_a_place(uint c) { return (c & ~15u) | ((c & 8u) >> 1) | ((c & 1u) << 3) | ((c >> 1) & 3u); }
// MAT_LOAD_A_PLACED loads; or load `placed`, a MatB, with MAT_LOAD_B_ROWS
// beside the step's other loads and take the operand from it after
// (MAT_A_OF_PLACED).
#define MAT_A_OF_PLACED(a, placed)                                                            \
    for (uint mat_e = 0; mat_e < 8; ++mat_e) a[mat_e] = placed[mat_e];
#define MAT_LOAD_A_PLACED(a, rows, at, stride)                                                \
    {                                                                                         \
        MatB mat_placed;                                                                      \
        MAT_LOAD_B_ROWS(mat_placed, rows, at, stride);                                        \
        MAT_A_OF_PLACED(a, mat_placed)                                                        \
    }

// ---- MAT_K64: K walked 64 at a time ----------------------------------------
// For C = A * B^T with both tensors row-major (K contiguous in the rows of
// both): the four multiplies of a 64-deep group pair K in an order that lets
// every lane load whole 16-byte vectors. Multiply t (0..3) of the group at
// G takes, of each 8-bfloat vector of a row, bfloats 2 t and 2 t + 1:
//
//     operand column c (of A) and row c (of B)  <->  k = G + 8 (c >> 1) + 2 t + (c & 1)
//
// A lane of B then needs, per column, one vector for all four multiplies
// (its two rows are 2 t and 2 t + 1 of vector mat_lane_row() / 2), where the
// K order needs four 4-byte loads per multiply; a lane of A needs four
// vectors per row (vectors 4 mat_a_half() .. + 4, of each its even or its
// odd bfloats). The dense GEMM: 30 TFLOP/s in K order, 42 this way.
// (gemm.comp is the example: loads, clamped edges, the K tail.)
//
// The sum over K is the same set of products grouped differently into the
// 16-deep multiplies, so a result can round differently from a K-order
// walk's (3 in 10 000 outputs of the dense GEMM by one bfloat ulp): not for
// a kernel that must match another bit for bit.
//
// The lane's part of A's multiplies t = 0..3: `v0x` are the four uvec4 of
// its first row (vector indices G / 8 + 4 mat_a_half() + x), `v1x` of its
// second; `odd16` as in MAT_A_OF_ROWS.
#define MAT_K64_A_STEP(m, t, v00, v01, v02, v03, v10, v11, v12, v13, odd16)                   \
    {                                                                                         \
        m[0] = MAT_LO(v00[t] >> (odd16)); m[1] = MAT_LO(v01[t] >> (odd16));                   \
        m[2] = MAT_LO(v02[t] >> (odd16)); m[3] = MAT_LO(v03[t] >> (odd16));                   \
        m[4] = MAT_LO(v10[t] >> (odd16)); m[5] = MAT_LO(v11[t] >> (odd16));                   \
        m[6] = MAT_LO(v12[t] >> (odd16)); m[7] = MAT_LO(v13[t] >> (odd16));                   \
    }
#define MAT_K64_A(m0, m1, m2, m3, v00, v01, v02, v03, v10, v11, v12, v13, odd16)              \
    MAT_K64_A_STEP(m0, 0, v00, v01, v02, v03, v10, v11, v12, v13, odd16)                      \
    MAT_K64_A_STEP(m1, 1, v00, v01, v02, v03, v10, v11, v12, v13, odd16)                      \
    MAT_K64_A_STEP(m2, 2, v00, v01, v02, v03, v10, v11, v12, v13, odd16)                      \
    MAT_K64_A_STEP(m3, 3, v00, v01, v02, v03, v10, v11, v12, v13, odd16)
// The lane's part of B's multiply t: `v0`..`v3` are the uvec4 at vector
// index G / 8 + mat_lane_row() / 2 of memory rows mat_lane_col() + j.
#define MAT_K64_B(m, t, v0, v1, v2, v3)                                                       \
    {                                                                                         \
        m[0] = MAT_LO(v0[t]); m[1] = MAT_LO(v1[t]); m[2] = MAT_LO(v2[t]); m[3] = MAT_LO(v3[t]); \
        m[4] = MAT_HI(v0[t]); m[5] = MAT_HI(v1[t]); m[6] = MAT_HI(v2[t]); m[7] = MAT_HI(v3[t]); \
    }

// ---- Threadgroup memory ----------------------------------------------------
// An order within one subgroup for its shared memory: its lanes' writes
// before this, their reads after.
void mat_subgroup_barrier() {
    controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
}

// A block at a tensor's edge that must read as zero past it (a K tail;
// where the products past the edge are not stored, clamp instead): memory
// past the tensor is not read (Metal's tensor slices check bounds;
// coopMatLoad does not). The subgroup copies the block's `rows` x `cols`
// valid elements into its 16x16 of `mat_stage`, zero elsewhere, each lane
// eight elements of a row; then MAT_LOAD_STAGED_A or _B. For the rare block:
// eight guarded 2-byte loads per lane.
#define MAT_STAGE_ELEMENTS(subgroups) ((subgroups) * 256)
#define MAT_STAGE(address, stride, rows, cols)                                                \
    {                                                                                         \
        uint mat_r = gl_SubgroupInvocationID >> 1, mat_c0 = (gl_SubgroupInvocationID & 1) * 8; \
        BF16 mat_source = BF16((address) + 2 * uint64_t(mat_r) * (stride));                   \
        for (uint mat_j = 0; mat_j < 8; ++mat_j) {                                            \
            bfloat16_t mat_v = bfloat16_t(0.0);                                               \
            if (mat_r < (rows) && mat_c0 + mat_j < (cols)) mat_v = mat_source.v[mat_c0 + mat_j]; \
            mat_stage[gl_SubgroupID * 256 + mat_r * 16 + mat_c0 + mat_j] = mat_v;             \
        }                                                                                     \
        mat_subgroup_barrier();                                                               \
    }
#define MAT_LOAD_STAGED_A(a) MAT_LOAD_A(a, mat_stage, gl_SubgroupID * 256, 16)
#define MAT_LOAD_STAGED_B(b) MAT_LOAD_B(b, mat_stage, gl_SubgroupID * 256, 16)
