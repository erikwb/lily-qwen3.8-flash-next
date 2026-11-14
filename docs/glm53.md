# GLM 5.3 Flash

lily also serves GLM 5.3 Flash, from antirez's published Q2 GGUF loaded as it
is, on the Linux/Vulkan backend. The implementation follows antirez's ds4
(MIT, credited in `NOTICE`), which runs the same file on Metal, CUDA and ROCm.

## Checkpoint and backend

- File: `GLM-5.3-Flash-Q2.gguf` from `antirez/glm-5.3-flash-gguf`, 96.5 GB
  (routed experts IQ2_XXS gate/up and Q2_K down, the rest Q8_0, Q4_K, BF16
  and F32). Other GLM GGUFs load only if every tensor has the shape and one of
  the block types the kernels read, and the tensors one dispatch reads
  together share a type (`src/glm53/schema.rs`); the loader refuses anything
  else before mapping it.
- Backend: Vulkan only (GLSL kernels in `src/kernels/vulkan/glm53*.comp`),
  written and measured on an M5 Max under Honeykrisp. There are no Metal
  kernels yet.
- The weights are mapped read-only from the page cache, not copied. Run with
  `hk_host_memory_read_only=true hk_host_memory_import_read_only=true`.

## Running it

```sh
lily --model ~/models/GLM-5.3-Flash-Q2/GLM-5.3-Flash-Q2.gguf \
    --cache-bytes 9G --pin-weights off [--mtp-drafts 1] [--max-seq 65536]
```

A `.gguf` path as `--model` selects GLM (the file's architecture is the
model type; its metadata stands in for `config.json`, its tokenizer and chat
template are read from it).

- `--cache-bytes 9G`: the session cache's floor for models with bf16 caches is
  8.6 GB; GLM's sessions are small (about 13 KB a token of context plus
  0.3 GB of recurrent state), so this is a cap, not an allocation.
- `--pin-weights off`: pinning would lock all 96.5 GB of weights in memory.
- `--mtp-drafts 1`: speculative decoding with the model's own MTP block, one
  draft a step.
- One model on the machine at a time: the GLM loader takes the instance lock
  (`src/instance.rs`) before mapping anything, like every other loader, and
  GLM and Qwen3.8-Flash-Next do not fit in 128 GB together.

On the M5 Max a long compute command holds other GPU clients back until it
ends; a per-layer prefill pass processing 1 024 rows is one, and without
Mesa's compute kick (aurora-silicon/mesa #260) the desktop's work can wait
past the kernel driver's timeout. Run GLM on a Mesa with the kick.

Memory: the 96.5 GB of weights in the page cache, about 1.5 GB of scratch
for prefill runs of 1 024 rows, and the sessions.

## What runs

`src/gguf.rs` reads the file (hardened against malformed headers) and decodes
its blocks on the CPU. `src/glm53/` holds the configuration, the schema check,
the model (45 trunk layers of KDA linear attention and DSA sparse attention,
3:1, three dense FFN layers then 288 routed experts and a shared one, four
hyper-connection streams with the Sinkhorn split), the MTP draft block, the
engine interface, the tokenizer and template built from the GGUF, and a CPU
reference of every operation. Decode and the speculative verify pass use the
GEMV kernels on f32 activations (up to four rows); prompts run in runs of
1 024 rows on the matrix units, routed experts grouped by expert, one pass a
layer. A pass of several rows rounds differently from a one-row decode step;
where two experts tie at the top-8 boundary (seen to 1e-6 after a
pseudo-random prompt) the two can route differently, so speculative decoding
gives plain decoding's tokens only up to such ties.

## Differences from ds4

- The MTP head attends at most the last 2 051 rows of its run (ds4: the whole
  run). It changes only how often drafts are accepted, never the output.
- Prefill GEMMs round both operands to bfloat16 (f32 accumulators); decode
  works on f32 activations. Quality is close to ds4's (below).
- Prompt runs compute DSA attention on the matrix units: the f32 queries
  and attention weights are split into an f16 high part (the value with its
  mantissa cut to ten bits) and the f16 rounding of the rest, multiplied
  with the f16 latent rows into f32 accumulators: about 22 significant bits
  for values in f16's normal range, an absolute 2^-24 below it. A score
  tile whose queries leave f16's range (or are not finite) is computed in
  f32. `LILY_GLM53_DSA_F32=1` keeps prompt runs on the f32 kernels. Decode
  and verify rows compute attention in f32.
- The hyper-connection control projection scales its sums by the RMS after
  the dot products (ds4 normalizes first): the products differ in rounding
  only.

## Tests and tools

```sh
cargo test --release --lib -- glm53 gguf    # kernel tests need a GPU, no file
LILY_GLM53_GGUF=<path> cargo test --release --lib glm53 -- --ignored --test-threads=1
```

The ignored tests load the real file: the header against the schema,
speculative steps against ordinary decoding (accepted and rejected drafts,
rollback across an indexer pool past the sparse window, snapshot restore and
fork), sampled speculation against ordinary sampling (the same draws and
penalty counts after a rejection and an acceptance), and cancelled prefill. `lily-glm53` (`src/bin/lily-glm53.rs`) compares
the GPU with the CPU reference layer by layer (`compare`), scores ds4's
official GLM 5.3 Flash continuations from its `gguf-tools/quality-testing`
(`score`, `--legacy` for ds4's reference rendering), and measures prefill,
decode and speculative decoding (`bench`, `spec`).

Measured on the M5 Max, 2026-11-13: ds4's 100 official cases NLL 0.466 per
token and 89/100 first tokens (ds4's own references 0.458 to 0.462 and 89 to
90); its 8 long cases (3K to 7K tokens) 0.666 and 6/8 (ds4 0.662 to 0.680,
6/8). Decode about 26 tok/s; speculative 27 to 37 tok/s depending on how
predictable the text is (greedy runs, the same tokens as plain decoding);
prefill 220 tok/s at 1K tokens and 160 at 4K.

## Not implemented

Batching across sessions (the server decodes one request at a time), sessions
saved to disk, vision, Metal kernels.
