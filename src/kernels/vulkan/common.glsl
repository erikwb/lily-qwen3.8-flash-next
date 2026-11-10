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

// A dispatch's arguments: `table` is the address of its address table,
// entry i the address Metal binds at [[buffer(i)]] (a buffer at its offset,
// or a `constant T&` param's value), and `grid` its thread grid.
layout(push_constant, scalar) uniform LilyDispatch {
    uint64_t table;
    uvec3 grid;
} lily_dispatch;

uint64_t arg(uint index) { return U64(lily_dispatch.table).v[index]; }
// A `constant uint&` / `constant float&` param at binding `index`.
uint arg_u32(uint index) { return U32(arg(index)).v[0]; }
float arg_f32(uint index) { return F32(arg(index)).v[0]; }

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
