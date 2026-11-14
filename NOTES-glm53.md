# GLM 5.3 Flash in lily: notes (BRIEF10)

Working notes, newest phase last. Tree: `framework:~/aurora-m5/lily-glm53`
(branch `glm53` from fork/main 52cea08). Reference: antirez's ds4 (MIT),
`framework:~/aurora-m5/ref/ds4` at b3f5018.

## Phase 1: the plan (2026-11-13)

### The file

`ewb-m5:~/models/GLM-5.3-Flash-Q2/GLM-5.3-Flash-Q2.gguf`, GGUF v3, 1412
tensors, 50 metadata keys, data at byte 9 515 200, tensor data ends at
96 505 814 584 (the file is 96 505 816 384: alignment padding). Header
dumped with `scratchpad/gguf_dump.py` while the body was still downloading.

Metadata (arch `glm5-next`), all equal to ds4's `DS4_SHAPE_GLM53`:
block_count 46 (trunk 45 + 1 nextn), embedding 4096, vocab 154 880, dense FF
12 288, expert FF 2048, 288 experts, 8 used + 1 shared, 3 leading dense
blocks, expert_weights_scale 2.5, expert_weights_norm true, swiglu_limit 10,
rms eps 1e-5, 64 heads, key_length 256, value_length 256, q_lora 1536,
kv_lora 512, rope_dimension_count **0** (no RoPE anywhere; ds4 says "raw
DSA cache row: 512 latent dims, no RoPE tail"), indexer 32 heads x 128,
top_k 2048, pool_size 4; KDA 64 heads x 128, conv 4, gate lower bound -5;
layer_types (0 = KDA, 1 = DSA) = `il % 4 == 3` for the trunk, layer 45
(MTP) is DSA; hyper-connections 4, Sinkhorn 20 iterations, eps 1e-6;
context_length 1 048 576. Tokenizer: gpt2 BPE, pre `glm4`, 154 880 tokens,
321 649 merges, eos 154820, pad 154821, bos 154822, eot 154827, eom 154829;
`tokenizer.chat_template` (8.6 KB jinja: `[gMASK]<sop>`, reasoning effort
low/high/max, tools).

Tensor inventory (types per role; GiB = 2^30 bytes):

| tensor | type | shape (ggml order: ne0 = row length) | count | GiB |
|---|---|---|---|---|
| token_embd, output | Q8_0 | 4096 x 154880 | 1+1 | 0.63 each |
| output_norm, attn_norm, ffn_norm | F32 | 4096 | | |
| hc_{attn,ffn}_fn | BF16 | 16384 x 24 | 45+45 (trunk) | 0.07 |
| hc_{attn,ffn}_scale / _base | F32 | 3 / 24 | | |
| kda_q, kda_k | **Q4_K** | 4096 x 8192 | 34 | 0.60 each |
| kda_v, kda_output | Q8_0 | 4096 x 8192, 8192 x 4096 | 34 | 1.13 each |
| kda_f_a, kda_g_a | Q8_0 | 4096 x 128 | 34 | |
| kda_f_b, kda_g_b | Q8_0 | 128 x 8192 | 34 | |
| kda_beta | Q8_0 | 4096 x 64 | 34 | |
| kda_{q,k,v}_conv | F32 | 4 x 1 x 8192 | 34 | |
| kda_dt_bias / a_log / o_norm | F32 | 8192 / 64 / 128 | 34 | |
| attn_q_a / q_b | Q8_0 | 4096 x 1536 / 1536 x 16384 | 12 | |
| attn_kv_a_mqa | Q8_0 | 4096 x 512 | 12 | |
| attn_k_b | Q8_0 | 256 x 512 x 64 (per head: 512 rows of 256) | 12 | 0.10 |
| attn_v_b | Q8_0 | 512 x 256 x 64 (per head: 256 rows of 512) | 12 | 0.10 |
| attn_output | Q8_0 | 16384 x 4096 | 12 | 0.80 |
| indexer.attn_q_b / attn_k / proj | BF16 | 1536 x 4096 / 4096 x 128 / 4096 x 32 | 12 | |
| indexer.pool_gate / pool_ape | BF16 | 4096 x 128 / 128 x 4 | 12 | |
| indexer.k_norm weight / bias | F32 | 128 | 12 | |
| ffn_gate/up/down (dense, layers 0-2) | Q8_0 | 4096 x 12288 | 3 | 0.15 each |
| ffn_gate_inp (router) | F32 | 4096 x 288 | 43 | 0.19 |
| exp_probs_b.bias | F32 | 288 | 43 | |
| ffn_gate_exps, ffn_up_exps | **IQ2_XXS** | 4096 x 2048 x 288 | 43 | 24.9 each |
| ffn_down_exps | **Q2_K** | 2048 x 4096 x 288 | 43 | 31.7 |
| ffn_{gate,up,down}_shexp | Q8_0 | 4096 x 2048 | 43 | 0.36 each |
| nextn.eh_proj | BF16 | 8192 x 4096 | 1 | 0.06 |
| nextn.{enorm,hnorm,shared_head_norm} | F32 | 4096 | 1 | |

Totals: 89.87 GiB (96.50 GB); routed experts 81.6 GiB of it. Block types to
support: F32, BF16, Q8_0, Q4_K, Q2_K, IQ2_XXS.

### The forward graph (ds4: `glm_graph_forward_token`, ds4.c ~57855)

State per token: 4 HC streams `H[4][4096]` f32. Embedding: Q8_0 row ->
copied into all 4 streams.

Per trunk layer `il` (0..44):
1. **HC pre (attn)**: `flat = rmsnorm_noweight(H as 16384)`; `mix[24] =
   hc_attn_fn(bf16) . flat`; Sinkhorn split (ds4 `hc_split_sinkhorn_one`):
   `pre[i] = sigmoid(mix[i]*s0 + b[i]) + eps`, `post[i] = 2*sigmoid(mix[4+i]*s1
   + b[4+i])`, `comb[4x4]` = row softmax of `mix[8+]*s2 + b[8+]` plus eps, then
   column-normalize, then 19 more row/column normalizations (each `/(sum+eps)`);
   `x = sum_h pre[h] * H[h]`; `xn = rmsnorm(x, attn_norm)`.
2. **Attention**, KDA (il % 4 != 3) or DSA (il % 4 == 3), output `a[4096]`.
3. **HC post**: `H'[dst][d] = a[d]*post[dst] + sum_src comb[dst + 4*src] *
   H[src][d]`.
4. **HC pre (ffn)** on H' with hc_ffn_* and ffn_norm -> `fn`.
5. **FFN**: layers 0-2 dense SwiGLU (12288); else MoE: router logits (F32
   4096x288), `p = sigmoid(logit)`, select top-8 of `p + exp_probs_b`,
   weights `p[sel] / max(sum, 6.1e-5) * 2.5`; each routed expert:
   `silu(min(g, 10)) * clamp(u, -10, 10)` times its weight, down (Q2_K), sum;
   plus the shared expert (Q8_0 SwiGLU with the same clamp, weight 1).
6. **HC post** of the FFN output into H.

Output: `h = mean of the 4 streams` (weighted sum with 0.25 each),
`rmsnorm(output_norm)`, `output` Q8_0 4096 -> 154880.

**KDA** (ds4 `metal/glm53_kda.metal`, from the kimi-k3 branch), per token:
q, k, v = Q4_K/Q4_K/Q8_0 projections of xn (8192 each); `raw_gate =
f_b(f_a(xn))` (8192), `raw_beta = beta(xn)` (64), `out_gate = g_b(g_a(xn))`
(8192). Per channel c: causal depthwise conv over the last 3 inputs + the new
one (weights `conv[c*4 + w]`, w = 3 is the newest), then SiLU; conv state keeps
the last 3 raw inputs per channel (q, k, v separately). Per head (64 x 128):
`q = l2norm(q) / sqrt(128)` (eps 1e-6 inside the rsqrt), `k = l2norm(k)`;
decay per key channel `exp(-5 * sigmoid(exp(a_log[h]) * (raw_gate + dt_bias)))`;
`beta = sigmoid(raw_beta)`. State `S[h][value 128][key 128]` f32:
`S *= decay (per key column)`; `hk = S[v] . k`; `S[v] += k * (v[v] - hk) *
beta`; `o[v] = S[v] . q`. Output: per head RMSNorm of o (eps 1e-5) times
`o_norm` times `sigmoid(out_gate)`; then `kda_output` Q8_0 8192 -> 4096.

**DSA** (MLA without RoPE, sparse via the pooled indexer):
`qr = rmsnorm(q_a . xn, q_a_norm)` (1536); `q = q_b . qr` (64 x 256);
`kv = rmsnorm(kv_a . xn, kv_a_norm)` (512), stored to the latent cache as
**f16**. Absorbed: `qk_low[h][j] = sum_i q[h][i] * k_b[h][j][i]` (512 per
head); score of cached row s = `qk_low[h] . kv[s] / sqrt(256)`; softmax over
the selected rows (denominator floored at 1e-20); `lat[h] = sum_s w_s kv[s]`;
`head[h][d] = v_b[h][d] . lat[h]` (256); `attn_output` Q8_0 16384 -> 4096.

**Pooled indexer** (`kernel_glm53_indexer_pool_update`): per token, `ik =
indexer.attn_k . xn` (128), `ig = pool_gate . xn` (128) (both BF16 weights).
Every 4 tokens form a pool: each row LayerNorm'd (mean/var over 128, eps
1e-6, weight + bias), then per channel softmax over the 4 rows of `ig +
pool_ape[r]` weights the rows; pooled key stored (f16). Incomplete pool rows
wait in a tail buffer (raw k, gate, f32). Selection at decode, visible =
pos + 1: if visible <= 2048 + 3 = 2051, attend all rows (dense). Else
`iq = indexer.attn_q_b . qr` (32 x 128), `w = indexer.proj . xn` (32),
score(pool p) = `sum_h max(iq[h] . pk[p] / sqrt(128*32), 0) * w[h]` over the
`visible / 4` complete pools, top 512 pools -> 2048 token rows, plus the
`visible % 4` tail tokens (the incomplete pool) always.

**MTP block** (layer 45, ds4 `glm_graph_mtp_step`): no HC. Input
`concat(rmsnorm(embed(next_token), enorm), rmsnorm(mean of H, hnorm))` ->
`eh_proj` (BF16 8192 -> 4096) -> plain pre-norm DSA layer (dense attention
over its own latent cache window, no indexer) and MoE FFN with residuals ->
`shared_head_norm` -> the shared `output` head. To be read in full in phase 5.

### What lily has, what is new

lily's activations are bf16 and its weights are its own q4/q8/bf16 layouts;
the kernels are written for those. GLM's weights are GGUF blocks and ds4 runs
the residual path in f32. Reuse is therefore at the level of patterns (the
GEMV/MoE kernels' load layouts, PORTING.md's rules, the backend, the server),
not of kernels. New:

- `src/gguf.rs`: GGUF reader (header, metadata, tensor table), block types.
- dequant + GEMV/GEMM kernels for Q8_0, Q4_K, Q2_K, IQ2_XXS, BF16 (f32
  activations), with a bit-exact CPU dequant (ported from ggml/ds4) as the
  reference; routed-expert gather kernels (IQ2_XXS gate/up pair + SwiGLU,
  Q2_K down + weighted combine).
- KDA conv + step (decode) and the chunked prefill; DSA latent store, qk_low,
  dense and indexed attention, pooled indexer update/score/top-k; HC pre with
  Sinkhorn and HC post (f32, 4 streams); router sigmoid top-8 with bias.
- `src/glm53/`: config (from GGUF metadata, checked against the numbers
  above), weights (one mapped buffer per tensor region), CPU reference,
  model (decode, prefill, MTP), `LanguageModel` implementation.
- Tokenizer: lily's `Tokenizer` reads `tokenizer.json` + a jinja template; for
  GLM build the tokenizer.json in memory from the GGUF vocab/merges with the
  glm4 pre-tokenizer regex (ds4 `bpe_tokenize_text_glm4`), and use the GGUF's
  template. Model selection: `checkpoint_model_type` learns that a `.gguf`
  file (or a directory holding one) of arch `glm5-next` is GLM.

### Memory budget (GPU clients get 85% of 128 GB; lily sets heap 0.84)

| | bytes |
|---|---|
| weights (imported read-only from the page cache) | 96.5 GB |
| KDA state 34 x 64 x 128 x 128 f32 | 143 MB |
| KDA conv state 34 x 3 x 3 x 8192 f32 | 10 MB |
| DSA latent cache, 13 layers x 512 f16 | 13.3 KB/token |
| pooled indexer keys, 12 layers x 128 f16 / 4 | 0.8 KB/token |
| decode scratch | < 10 MB |
| prefill scratch per chunk row (q/k/v/gates/hc/ffn f32) | ~0.4 MB/row |

At 32K context the caches are 0.45 GB, at 128K 1.8 GB; with 512-row prefill
chunks the total stays near 98 GB, under the ~107 GB heap. The page cache
must hold the 96.5 GB file, so nothing else large (lily-server's Qwen) can run.

### Order of work

1. GGUF reader + CPU dequant (bit-exact vs ggml) + tests on framework.
2. GLSL: dequant/GEMV kernels per type, tests vs CPU (Mac GPU).
3. CPU reference forward (one layer at a time, real weights) and GPU decode
   compared layer by layer.
4. Prefill, tokenizer, server. 5. MTP. 6. Speed.

GPU access: blocked until the lead frees it (2026-11-13: a Mesa agent runs
CTS). Until then: CPU-side code, kernel source compiled with lily-check.sh,
CPU tests on framework.

## Phases 2-3: loader, kernels, decode (2026-11-13)

- `src/gguf.rs`: GGUF v3 reader + exact CPU decoders; bit-identical to values
  computed with ds4's own formulas (`tests/goldens/gguf_dequant.bin`,
  generator `tools/reference/glm53/gguf_dequant_golden.c`; blocks from ds4's
  quantizer and random bytes). The real header validates (test on the Mac,
  `LILY_GLM53_GGUF=... cargo test --lib glm53::config -- --ignored`).
- `src/glm53/reference.rs`: CPU reference of every op + a layer-by-layer
  model on GGUF weights (rayon matvecs).
- `vulkan/glm53.comp` (31 kernels), `src/kernels/glm53.rs`: 11 unit tests vs
  the CPU reference, all pass on the M5 GPU (first try): GEMV for six types
  x 1-4 rows and grouped, MoE gate/up + down, router, HC pre/post/mean,
  embed, KDA decode + prefill (37 tokens in runs of 16), DSA store+attend,
  pooled indexer update, indexer scores + top-k (vs CPU set).
- `src/glm53/model.rs`: GGUF mapped read-only (1 GiB regions, 9 s cold, 0.7 s
  warm), serial pass, up to 4 rows per step.
- `lily-glm53 compare` (prompt "[gMASK]<sop>The capital of France is", 7
  tokens): all 45 trunk layers vs the CPU reference from the GPU's own
  input: relative L2 <= 5e-5 (DSA attention out, f16 latent), ~1e-7
  elsewhere; identical experts. Greedy: " Paris. Paris is the capital and
  largest city of France. It is located on the Seine River, in the north of".
- Runner: `~/aurora-m5/lily-wt/glm53/run.sh <bin> args` (sync, build, flock,
  lily-server's Mesa + read-only import env; log in the Mac's
  /tmp/lily-glm53-<bin>.log). Stop lily-server before model runs.

## Phase 4: tokenizer, template (2026-11-13)

- `src/glm53/tokenizer.rs`: tokenizer.json built in memory from the GGUF
  (byte-level BPE, glm4 split regex, control tokens special, user-defined
  (<think>, <tool_call>, <arg_key>...) added non-special); the GGUF template
  with an `enable_thinking` switch on the generation prompt (`<think>` vs
  `<think></think>`); stop ids eos/eot/eom = <|endoftext|>, <|user|>,
  <|observation|>. `Tokenizer::from_gguf`; `from_model_dir` takes a .gguf.
  minijinja needs `loop_controls` for the template's `{% break %}`.
- `lily-glm53 chat`: Fibonacci prompt answered correctly, stops on <|user|>.

### End-to-end quality (ds4's official GLM 5.3 Flash continuations)

`lily-glm53 score` teacher-forces each official continuation (Z.AI FP8,
temperature 0) after the prompt; 100 cases, 11 550 tokens:

| prompt rendering | avg NLL/token | first-token match | avg greedy prefix | greedy agreement |
|---|---|---|---|---|
| ds4 legacy (`<|user|>..<|assistant|><think></think>`) | 0.4650 | 90/100 | 7.39 | 84.3% |
| ds4 reference, Metal, same file | 0.4580 | 89/100 | 7.37 | |
| ds4 reference, CUDA Spark | 0.4618 | 90/100 | 7.49 | |
| rendered (effort Low + API reasoning) | 0.4171 | | | 85.4% |

lily is within ds4's own backend-to-backend spread. (Prompt counts are one
above the API's usage count, which leaves `</think>` to the completion.)

**GPU released 2026-11-13 ~20:15** at the lead's request: no lily-glm53
process, no lock held; lily-server left as it was (stopped; I had stopped it
for the model runs, and per the lead's note did not restart it).

## First speed numbers (2026-11-13, commit a1c26ab = GPU path of 62e12cd)

`lily-glm53 bench` on ewb-m5 (lily-server stopped; Mesa
`~/ai/mesa-plfix-r3next/install`, `hk_host_memory_read_only=true
hk_host_memory_import_read_only=true heap_memory_percent=0.84`), prompt =
`[gMASK]<sop>` + ds4's `speed-bench/promessi_sposi.txt` truncated, greedy,
3 runs each, fresh state per run. Decode is the model's forward with the
logits read back and argmax on the host (~0.6 MB per token); the server's
path samples on the GPU instead.

| prompt | prefill tok/s (3 runs) | decode 256 tok/s (3 runs) | ms/token |
|---|---|---|---|
| 32 | 49.0 / 48.6 / 49.7 | 24.48 / 24.43 / 24.45 | 40.9 |
| 1024 | 44.7 / 44.9 / 46.9 | 22.35 / 22.49 / 22.53 | 44.5 |
| 4096 | 42.6 / 36.9 / 36.2 | 20.79 / 19.98 / 20.13 | 49.7 |

Untuned: every kernel is the first correct version, a serial pass with a
barrier after each of ~1 200 dispatches per token, and prefill is the decode
GEMV path over runs of 4 rows (no matrix-unit GEMM yet), so prefill is only
~2x decode. Bytes read per token are ~10.3 GB (KDA projections 3.7, routed
experts 2.4, DSA 1.7, shared experts 1.15, head 0.67), so the bandwidth floor
is ~21 ms (~48 tok/s plain). Reference points: ds4 records DeepSeek V4 Flash
Q2 on an M5 Max (Metal) at ~39 tok/s decode and ~790 tok/s prefill at 2K
(`speed-bench/m5_max.csv`); GLM 5.3 Flash is not in that file.

Next for speed, in order: concurrent pass with barriers only between
dependency levels (written, numerically identical), the GEMV kernels' load
shapes (Q8_0 blocks are 2-byte aligned: 17 u16 loads per 32 values now),
a matrix-unit GEMM for prefill, then MTP speculative decoding.

## Speed round 1 (2026-11-13 evening)

| commit | change | effect (interleaved A/B, 256-token prompt, 128 tokens) |
|---|---|---|
| 7502abc | concurrent pass, barriers only between levels; shared expert added in the down kernel | prefill 49.3 -> 53.2 tok/s; decode 41.3 -> 41.1 ms (noise); bit-identical |
| 3697551 | HC post fused into the next control projection (32-TG partial sums) + hc_pre2 (16-lane Sinkhorn, sums in registers) | step median 40.7/41.3/38.0 -> 37.5/38.1/37.6 ms; prefill +3%; same digest |
| 662d839 | one GEMV dispatch per level's same-input projections (sets) | no measurable change (same digest) |

Probe (`kernels::glm53::tests::gemv_timing`, `moe_timing`): big Q8_0 GEMVs
450-565 GB/s, Q4_K ~290, small Q8_0 (shared expert) 95-300 (noisy); routed
gate/up 163 us a layer (212 GB/s), down 99 us (222 GB/s). Tried, no gain:
two rows a subgroup with activations in registers, gate+up sharing the
activation load, Q2_K codes as 16-byte loads. IQ2 grid lookup ablated: -18 us
of 163. The routed gate/up is instruction-bound (~2 600 instr/lane x 16 384
subgroups ~ 158 us at the measured rate): next attempt there is fewer ALU
ops per weight (scale once per 32, sign by XOR of float bits, grid in a
buffer), or repacking the experts.

## Server (2026-11-13 evening, commit dbe25ff)

`lily --model <GLM .gguf> --bind 127.0.0.1:8091 --max-seq 32768 --cache-bytes
9G --pin-weights off --stats-db off --machine-stats off` (bf16-format models
need >= 8.6 GB of cache budget; pin off because pinning would mlock all
96.5 GB). Loads in 0.8 s (mapped). Checked: direct answer (thinking off),
streamed reasoning split from content (thinking on), a GLM tool call parsed
into OpenAI tool_calls, the follow-up turn with the tool result (template
tool rendering; session cache reused 175 of 257 prompt tokens). Server
decode 27.8-28.1 tok/s (GPU-side sampling). Server stopped afterwards.

## Phase 5: MTP speculative decoding (2026-11-13, commit 6f1f781)

One draft a step (ds4 does the same for GLM 5.3). Verify = 2 rows with both
draws sampled on the GPU (exact under sampling: the emitted token is always a
target draw). Rejection costs no replay: the verify pass's KDA kernels snapshot
each layer's conv window + state after row 0 into a second buffer that becomes
current (`snap_row`); the pooled indexer keeps tail rows so a rolled-back row is
re-fed correctly. MTP head attends its own run of rows (capped at the dense
window, a deviation from ds4 that only affects draft quality).

`lily-glm53 spec` (greedy; tokens identical to plain decoding in every run):

| prompt | accepted | plain | speculative |
|---|---|---|---|
| code (chat) | 92% | 26.5 tok/s | 36.6 tok/s |
| English prose (chat) | 69% | 26.4 | 31.9 |
| raw Italian novel text | 51% | 26.1 | 26.7 |

Server `--mtp-drafts 1`: 35.3 tok/s greedy (93/106 drafts), 33.5 at
temperature 1 (86/113), code prompt. A speculative step costs ~52 ms against
~38 for a plain one: three passes with host round trips per step (verify,
catch-up, draft); folding the draft into the verify pass is the next step there.

## Prefill on the matrix units (2026-11-13 night, commit 5492603)

Runs of up to 1 024 rows: dequantizing GEMM (64x64 tiles, bf16 operands in
threadgroup memory, f32 accumulators; batched form for attn_k_b/v_b) and the
routed experts grouped by expert (sort pairs -> 64-pair tiles -> gate/up/down
GEMMs on each expert's weights -> SwiGLU -> combine with the shared expert).

| prefill | GEMV runs of 4 | GEMM, 256 rows | + grouped MoE, 256 | 512 | 1 024 rows |
|---|---|---|---|---|---|
| 1 024 tokens | 45 tok/s | 84 | 125 | 174 | 220 |
| 4 096 tokens | 37-43 | 71 | 103 | 137 | 164 |

Quality: 100 official cases NLL 0.4659, 89/100 first token (ds4 Metal 0.458 /
89); **8 long official cases (3-7K tokens, sparse attention, several runs): NLL
0.6657, 6/8 first token (ds4 ROCm 0.6617, CUDA 0.6801, both 6/8)**.

At 4K the expert GEMMs are 48% of the prefill: each tile decodes its expert's
whole weight for ~28 rows (decode-ALU bound). Next: decode IQ2 to unscaled
grid values from a table and apply the 32-value scales to the accumulators
per K step; dsa_scores/dsa_lora (16%) to the matrix units.

## Checks of the rest of lily (2026-11-13 night)

- Whole lib test suite on the Mac: 724 passed, 0 failed, 80 ignored.
- Qwen3.8 lily-bench (1 024-token prompt, 96 steps): digest d31e5265cf174669 on
  fork/main 52cea08 and on this branch (prefill 1643 / 1674 tok/s, decode
  61.9 / 67.2 tok/s: noise).
- NOTICE credits ds4.

## Where things stand / next

Works: GGUF load (0.8 s warm), decode, MTP speculation, prefill on matrix units,
tokenizer + template + tool calls, server (incl. --mtp-drafts 1), session cache.
Decode ~38 ms/token plain at short context (26 tok/s; 42 ms at 4K), speculative
26.7-36.6 tok/s by content (server 33-35 on code). Prefill 220 tok/s at 1K.

Next, by expected gain: (1) routed-expert decode ALU (GEMV and GEMM: table +
scales on the accumulators, or repacking the experts); (2) fold the MTP draft
into the verify pass (one pass a step instead of three); (3) Q4_K/IQ2 GEMV load
shapes; (4) DSA attention for prefill on the matrix units; (5) cross-session
batching (max_batch_rows) and session persistence (not implemented);
(6) Metal kernels (none written: GLSL only, as asked).
Server notes: needs --cache-bytes >= 8.6G (bf16 format floor) and should run
with --pin-weights off (pin would mlock all 96.5 GB).
