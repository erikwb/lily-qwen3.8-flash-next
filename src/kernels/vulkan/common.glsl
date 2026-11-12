// What every Vulkan kernel shares (src/vulkan.rs, "How the Metal 4 model maps
// to Vulkan"; the porting rules are in src/kernels/vulkan/PORTING.md).
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_bfloat16 : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require

// The threadgroup size of the dispatch (Metal's threadsPerThreadgroup).
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Memory by GPU address: `F32(address).v[i]` is element i of the floats at
// `address`. bfloat16_t is Metal's bfloat: float(x) widens it exactly and
// bfloat16_t(f) rounds to nearest even.
layout(buffer_reference, scalar, buffer_reference_align = 1) buffer U8 { uint8_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 1) buffer I8 { int8_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer U16 { uint16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer I16 { int16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer BF16 { bfloat16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer BF16x2 { bf16vec2 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer BF16x4 { bf16vec4 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer F16 { float16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer U32 { uint v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer U32x2 { uvec2 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer U32x4 { uvec4 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer I32 { int v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer F32x4 { vec4 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer U64 { uint64_t v[]; };

// float to bfloat, rounded to nearest even as Metal's bfloat(f) is. The
// driver's own conversion truncates (Vulkan leaves the rounding of a float
// conversion to the implementation), so `bfloat16_t(f)` is this function
// everywhere: the macro takes the constructor's place, the type's other
// uses stay. A vector is `bf16(v)`; `bf16vec4(v)` of a vec4 would truncate.
bfloat16_t bf16(float f) {
    uint u = floatBitsToUint(f);
    uint r = u + 0x7FFFu + ((u >> 16) & 1u);
    // A NaN keeps its high bits: rounding up could carry it into infinity.
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) r = u;
    return uintBitsToBFloat16EXT(uint16_t(r >> 16));
}
bf16vec4 bf16(vec4 f) {
    uvec4 u = floatBitsToUint(f);
    uvec4 r = u + 0x7FFFu + ((u >> 16) & 1u);
    r = mix(r, u, greaterThan(u & 0x7FFFFFFFu, uvec4(0x7F800000u)));
    return uintBitsToBFloat16EXT(u16vec4(r >> 16));
}
bfloat16_t bf16(bfloat16_t f) { return f; }
#define bfloat16_t(f) bf16(f)

// A dispatch's arguments: `grid` is its thread grid and `args[i]` the address
// Metal binds at [[buffer(i)]]: a buffer at its offset, or a `constant T&`
// param's value. The compiler reads push constants once per dispatch, so
// arg(i) costs a thread nothing; what it points at is a load.
layout(push_constant, scalar) uniform LilyDispatch {
    uvec3 grid;
    // Bit i: args[i] is a four-byte param's value itself, not its address.
    uint by_value;
    uint64_t args[30];
} lily_dispatch;

uint64_t arg(uint index) { return lily_dispatch.args[index]; }
// A `constant uint&`, `constant int&` or `constant float&` param at binding
// `index`. Read a four-byte param through these and nothing else: it may
// arrive by value.
uint arg_u32(uint index) {
    if ((lily_dispatch.by_value >> index & 1u) != 0u) return uint(lily_dispatch.args[index]);
    return U32(lily_dispatch.args[index]).v[0];
}
int arg_i32(uint index) { return int(arg_u32(index)); }
float arg_f32(uint index) { return uintBitsToFloat(arg_u32(index)); }

// Metal's dispatchThreads runs exactly the grid's threads; Vulkan runs whole
// threadgroups. False for a thread past the grid of a Grid::Threads dispatch,
// which must do nothing. (Of a Grid::Threadgroups dispatch every thread is
// inside.)
#define LILY_THREAD_IN_GRID all(lessThan(gl_GlobalInvocationID, lily_dispatch.grid))

// Metal's kernel inputs.
#define thread_position_in_grid gl_GlobalInvocationID
#define threadgroup_position_in_grid gl_WorkGroupID
#define thread_position_in_threadgroup gl_LocalInvocationID
#define thread_index_in_threadgroup gl_LocalInvocationIndex
#define simdgroup_index_in_threadgroup gl_SubgroupID
#define thread_index_in_simdgroup gl_SubgroupInvocationID
#define threads_per_threadgroup gl_WorkGroupSize
