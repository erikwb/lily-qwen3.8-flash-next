# Porting a Metal kernel file to Vulkan

On Linux the engine runs on Vulkan (Honeykrisp on an Apple M5): `src/vulkan.rs`
is `crate::metal` there, with the Metal backend's API, so the Rust dispatch
code in `src/kernels/*.rs` is used unchanged. What is ported is the shader
source: each `src/kernels/metal/<file>.metal` becomes
`src/kernels/vulkan/<file>.comp`, GLSL, holding the same kernels under the
same names, doing the same arithmetic in the same order.

`norm.comp`, `elementwise.comp` and `gemm.comp` are finished examples: read
one beside its `.metal` before starting.

## The contract

- **One file, many kernels.** `build.rs` compiles a `.comp` once per kernel:
  every `KERNEL_<name>` the file mentions is a kernel `<name>` (exactly the
  Metal kernel function's name, which the Rust side passes to
  `ctx.pipeline("<name>", ...)`), built with `-DKERNEL_<name>`. Put each
  kernel's `main()` (and what only it needs) under `#ifdef KERNEL_<name>`;
  kernels of one family (a Metal template or macro instantiated under several
  names) share code under `#if defined(KERNEL_a) || defined(KERNEL_b)` and
  differ by `#ifdef`. Use the prefix `KERNEL_` for nothing else. A kernel
  name must be unique across all `.comp` files. Helpers shared by several
  files go in a `.glsl` include.
- **Test kernels.** Some unit tests run kernels of their own from
  `tests/metal/<file>_test.metal` (an older form a shipped kernel is checked
  against). Their ports go in `tests/vulkan/<file>_test.comp`, same rules,
  includes from `src/kernels/vulkan`. The Rust test passes the Metal source
  to `ctx.pipeline`, which ignores it here, so the test itself needs no
  change.
- **Arguments.** `arg(i)` (common.glsl) is the 64-bit GPU address of what
  Metal binds at `[[buffer(i)]]`: a `device T*` pointer (already at its
  offset), or the value of a `constant T&` param. A four-byte param (`uint`,
  `int`, `float`) is read with `arg_u32(i)`, `arg_i32(i)` or `arg_f32(i)`
  and in no other way: the backend may pass it by value in place of its
  address. A larger one (a `uint4`, a struct) is read through a buffer
  reference at `arg(i)`. Read and write memory
  through the buffer references of common.glsl: `BF16(arg(0)).v[i]`,
  `U32(address).v[i]`, `F32x4(...)`, ...; declare more where a kernel has a
  struct. All are `scalar` layout: fields at their C offsets (declare the
  struct's fields in the Metal order and check its size against the Rust
  side's).
- **64-bit offsets.** A GLSL array index is 32 bits and the expression
  `x.v[uint64]` does not compile. Where the Metal kernel indexes with a
  `ulong` (`x[(ulong)row * H + i]`), put the wide part into the address:
  `BF16(arg(0) + 2 * uint64_t(row) * H).v[i]` (times the element size in
  bytes). Do this wherever the product could pass 2^32 elements; tensors here
  reach tens of GB.
- **Types.** `bfloat` is `bfloat16_t` (GL_EXT_bfloat16): `float(x)` widens
  exactly and `bfloat16_t(f)` rounds to nearest even, as Metal's `bfloat`
  does. That rounding is common.glsl's: the driver's own conversion
  truncates, so `bfloat16_t(f)` is a macro for the function `bf16(f)`. A
  vector conversion must be written `bf16(v)` (`bf16vec4(v)` of a `vec4`
  truncates; `bf16vec4(a, b, c, d)` of four bfloats is fine), and a constant
  initializer cannot use either (write the bits). `half` is `float16_t`. `bfloat4`/`float4`/`uint4` are
  `bf16vec4`/`vec4`/`uvec4`. Metal `uchar`/`ushort`/`ulong` are
  `uint8_t`/`uint16_t`/`uint64_t`. `as_type<T>(x)` is the
  `floatBitsToUint`/`uintBitsToFloat` family (for bfloat:
  `bfloat16BitsToUintEXT`, `uintBitsToBFloat16EXT`).
- **Grid.** The Metal threadgroup grid is the Vulkan workgroup grid; the
  threadgroup size of a dispatch is the shader's local size (the backend sets
  it from the dispatch, never write `local_size` yourself). common.glsl
  defines the Metal input names: `thread_position_in_grid`,
  `threadgroup_position_in_grid`, `thread_index_in_threadgroup`,
  `thread_position_in_threadgroup`, `simdgroup_index_in_threadgroup`,
  `thread_index_in_simdgroup`, `threads_per_threadgroup` (all `uvec3` or
  `uint` as in GLSL: take `.x` etc. where Metal declares a `uint` or
  `uint2`). `threads_per_grid` is `lily_dispatch.grid`.
- **`Grid::Threads` dispatches** (Metal `dispatchThreads`: look at the Rust
  wrapper) run exactly the grid's threads on Metal. Vulkan runs whole
  threadgroups, so such a kernel must start with
  `if (!LILY_THREAD_IN_GRID) return;`. After it, a subgroup may be partly
  inactive: subgroup operations cover the active lanes only, and "lane 0"
  may not be among them (use
  `gl_SubgroupInvocationID == subgroupBallotFindLSB(subgroupBallot(true))`
  for "one lane of the subgroup"). A kernel that uses `barrier()` must not
  return before it in some threads only: guard the body instead.
  `Grid::Threadgroups` dispatches need no guard.
- **Subgroups.** A Metal simdgroup is a Vulkan subgroup, 32 lanes on this
  GPU. `simd_sum/min/max/or/xor/and` are `subgroupAdd/Min/Max/Or/Xor/And`;
  `simd_prefix_exclusive_sum` is `subgroupExclusiveAdd`;
  `simd_shuffle(v, lane)` is `subgroupShuffle`; `simd_shuffle_xor(v, m)` is
  `subgroupShuffleXor`; `simd_shuffle_down/up(v, d)` are
  `subgroupShuffle(v, lane +/- d)` where the source lane exists (a reduction
  written as shuffle-downs by 16, 8, 4, 2, 1 folded into lane 0 is the same
  tree with `subgroupShuffleXor`, which is defined for every lane);
  `simd_broadcast(v, lane)` is `subgroupBroadcast` (constant lane) or
  `subgroupShuffle`; `simd_broadcast_first` is `subgroupBroadcastFirst`;
  `simd_ballot`, `simd_is_first`, `simd_all/any` are the ballot/vote
  functions (enable `GL_KHR_shader_subgroup_vote` if needed).
- **Threadgroup memory.** `threadgroup T x[N]` is a global `shared T x[N];`
  (declared at file scope, under the kernel's `#ifdef`);
  `threadgroup_barrier(mem_flags::mem_threadgroup)` is `barrier()`. Atomics
  are `atomicAdd/Xor/...` on shared variables or on a buffer reference's
  member.
- **Arithmetic.** Keep the Metal kernel's order of operations and where it
  rounds to bf16; the tests compare against CPU references with tight bounds
  and some results must be bit-identical between kernels. Mark an
  accumulation whose order matters `precise` (the compiler may otherwise
  reassociate or contract it differently between two kernels). Metal's
  `fast::`/`precise::` functions are the plain GLSL ones; `rsqrt` is
  `inversesqrt`; `INFINITY` is `1.0 / 0.0`; `fma`, `exp`, `log`, `tanh`,
  `clamp`, `select` (`mix` with a bool) exist. Do not write a `half`
  through `packHalf2x16` or `float16_t(f)`: both truncate on this driver.
  `precise` does not reach `exp` and the other extended instructions.
- **Params are loads.** `arg(i)` is a push constant, read once per
  dispatch; `arg_u32(i)` loads through it unless the param came by value:
  take each once, outside loops (the compiler does not hoist the load).
- **Templates and macros.** MSL is C++: templates, lambdas, references,
  `static inline` helpers. GLSL has none of that: functions take
  `in`/`out`/`inout` parameters, families become `#define`d constants per
  kernel or function-like macros. GLSL inlines everything and has no
  recursion.
- **Tensor ops.** Kernels compiled as MSL 4 use MetalPerformancePrimitives
  (`matmul2d`, `metal::tensor`, cooperative tensors) to reach the GPU's
  matrix units. Here those are 16x16x16 cooperative matrices: read
  `matmul.glsl` and `gemm.comp`. A threadgroup of S simdgroups running one
  `matmul2d` of a BM x BN tile becomes S subgroups each running the tile's
  16x16 blocks it owns, with the same K walk; `acc[i]` over the destination
  cooperative tensor becomes the lane's eight elements of each block
  (`mat_row`, `mat_col`). Never let `coopMatLoad` read past a tensor's end:
  stage edge blocks (`MAT_STAGE`). Where a Metal kernel has both a tensor
  path and an older `simdgroup_matrix` or scalar fallback, port the path the
  Rust wrapper dispatches on a family-10 GPU (`apple_gpu_family() >= 10`).
- **The one-job limit.** The kernel driver gives a GPU job about two seconds.
  A direct port never comes near it; a kernel far slower than its original
  might.

## Driver notes

Things this driver and compiler do that a plain translation trips over:

- **Wide bfloat loads.** A `bf16vec4` read through `BF16x4` is four 16-bit
  loads (the reference is 2-byte aligned). Where the Metal kernel reads
  `bfloat4`s for speed, load `uvec4` words through `U32x4` (16 bytes, eight
  bfloats) and widen them: `uintBitsToFloat(w << 16)` and
  `uintBitsToFloat(w & 0xffff0000u)` are a word's two bfloats, exactly.
- **Lookup tables.** Declare a table as a global with an initializer
  (`uint kTable[256] = uint[256](...);`), not as a `const` array: indexing a
  `const` array copies it into every function that does, and so into every
  inlined use, which the driver compiles badly or not at all.
- **Code size.** Everything is inlined and a kernel is compiled once per
  threadgroup size it is dispatched with. Unrolling a loop over a body with a
  dozen format branches makes megabytes of SPIR-V; keep unrolled bodies small
  and leave rare paths in loops.
- **Cooperative matrices** are 16x16x16 with `bfloat16_t` operands and a
  `float` accumulator, one per subgroup (matmul.glsl); `coopMatLoad` takes a
  `bfloat16_t` array (a buffer reference's `.v` or a `shared` array), an
  element offset and a stride. Arrays of matrices and `acc[i]` element access
  work. Subgroup-scope matrices need the whole subgroup in uniform control
  flow around the matrix operations.
- **Subgroup shuffles** with an out-of-range source lane are undefined, not
  zero.
- **Reads past a buffer** fault the GPU job; Metal tensor slices clamp where
  a pointer walk does not.

Start with a direct, readable translation that passes the tests. Then, where
the Rust side or `docs/performance.md` says a kernel is on the decode or
prefill hot path, look at what the translation costs: the Metal kernels were
tuned (loads by `uint4`, work split across lanes, no divisions in loops) and
the port should keep those shapes, not undo them. Leave a comment only where
the port differs from the Metal kernel and why; the reasoning of the kernel
itself stays in the `.metal` file, which the `.comp` header names.

## Build and test

The tree is on the build host; it builds and runs on the Mac.

    ~/aurora-m5/lily-check.sh <file>.comp          # compile every kernel of a file here (glslc), fast
    ~/aurora-m5/lily-test.sh kernels::<module>     # the module's unit tests on the Mac's GPU

Both take `LILY_OVERLAY=<dir>` and `LILY_NAME=<name>`: a directory whose
`src/kernels/vulkan/` and `tests/vulkan/` files are laid over the main tree
(`~/aurora-m5/lily`), for work on a kernel file apart from the tree.

`lily-test.sh` copies the tree (and the overlay) to the Mac's
`~/ai/lily-<name>`, builds, and
runs `cargo test --release --lib <filter> -- --test-threads=1` under the GPU
lock (one GPU user at a time, across everyone), printing the failures and the
summary; the whole output stays in the Mac's `/tmp/lily-test-<name>.log`
(read it with `ssh -i ~/.ssh/id_agent ewb@192.168.1.110 'tail -100 /tmp/lily-test-<name>.log'`).
A kernel's unit tests are `tests/unit/kernels/<module>.rs`, each against the
f32 CPU reference in `tests/support/cpu_ref.rs`; tests marked
`#[ignore = "requires LILY_MODEL_DIR_FLASH"]` need a checkpoint and are not
run. A kernel without a shader fails with "has no Vulkan shader".

`LILY_VK_SHADER_DIR=<dir>` makes the backend load `<dir>/<kernel>.spv` in
place of the built-in shader, for trying a variant without a rebuild.

A GPU fault or a job over its time shows as "Vulkan command buffer failed"
(or the test hangs until its timeout): do not loop on it. Never reboot the
Mac, never kill processes there that are not yours, and never use
`pkill -f`/`pgrep -f` over ssh with a pattern that matches your own command
line.
