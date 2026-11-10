// The GPU's matrix units, which the Metal kernels reach through Metal 4
// tensor ops (MetalPerformancePrimitives matmul2d): here 16x16x16 cooperative
// matrices, one per subgroup, with bfloat operands and an fp32 accumulator:
//
//     MatC c = MatC(0.0);
//     for (k0 = 0; k0 < K; k0 += 16) { load a, b; c = coopMatMulAdd(a, b, c); }
//
// computes C[16, 16] += A[16, 16 of K] * B^T, where A's block is 16 rows of
// an [M, K] row-major tensor and B's 16 rows of an [N, K] one (the Metal
// kernels' transpose_right). A threadgroup of S subgroups runs S blocks at a
// time; a subgroup holds as many accumulators as it has blocks of the tile.
//
// Include after common.glsl. A kernel that stages edges declares
// `shared bfloat16_t mat_stage[MAT_STAGE_ELEMENTS(subgroups)];`.
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#pragma use_vulkan_memory_model

#define MatA coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA>
#define MatB coopmat<bfloat16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB>
#define MatC coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>

// A block straight from memory, every element of it inside its tensor:
// `rows` is a bfloat16_t array (BF16(address).v, or a shared one), `at` the
// index of the block's first element and `stride` the elements between rows.
// A's rows are the block's rows; B's rows are its columns.
#define MAT_LOAD_A(a, rows, at, stride) coopMatLoad(a, rows, at, stride, gl_CooperativeMatrixLayoutRowMajor)
#define MAT_LOAD_B(b, rows, at, stride) coopMatLoad(b, rows, at, stride, gl_CooperativeMatrixLayoutColumnMajor)

// The elements an accumulator's lane holds: c[i], i in 0..8, is element
// (mat_row(i), mat_col(i)) of the block. This is the driver's layout of a
// 16x16 matrix over a subgroup (Honeykrisp on AGX3), which the GEMM tests
// would catch changing.
uint mat_row(uint i) {
    uint lane = gl_SubgroupInvocationID;
    return (lane & 6) | ((lane >> 1) & 8) | (i >> 2);
}
uint mat_col(uint i) {
    uint lane = gl_SubgroupInvocationID;
    return ((lane & 1) << 2) + (lane & 8) + (i & 3);
}

// An order within one subgroup for its shared memory: its lanes' writes
// before this, their reads after.
void mat_subgroup_barrier() {
    controlBarrier(gl_ScopeSubgroup, gl_ScopeSubgroup, gl_StorageSemanticsShared, gl_SemanticsAcquireRelease);
}

// A block at a tensor's edge: memory past the tensor is not read (Metal's
// tensor slices check bounds; coopMatLoad does not). The subgroup copies the
// block's `rows` x `cols` valid elements into its 16x16 of `mat_stage`, zero
// elsewhere, each lane eight elements of a row; then MAT_LOAD_STAGED_A or _B.
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
