// GGUF block decoders for GLM 5.3 Flash (src/gguf.rs is the CPU version;
// layouts and value formulas are ggml's as antirez's ds4, MIT, has them).
// Every decoder turns one "task" of 32 consecutive values of a row into
// floats: one Q8_0 block, one 32-value sub-block of Q4_K, two 16-value groups
// of Q2_K, one 32-value group of IQ2_XXS, or 32 BF16/F32 values. Each value
// is a product of small exact factors (f16 scale times small integers), so
// the decode is exact whatever the compiler contracts.

#define GLM53_TASK 32u

// Type ids (GGUF): the kernels take the type as a compile-time constant.
#define GLM53_F32 0u
#define GLM53_Q8_0 8u
#define GLM53_Q2_K 10u
#define GLM53_Q4_K 12u
#define GLM53_IQ2_XXS 16u
#define GLM53_BF16 30u

float glm53_h2f(uint bits) { return unpackHalf2x16(bits).x; }

// Bytes of a 32-value task inside a row: the tasks of one block are laid out
// at the block's offset; this gives the block's byte offset for task t.
// Q8_0 34-byte blocks of 32; K-quants 256-value blocks.

// --- Q8_0: f16 d, 32 int8. Block base is 2-byte aligned. ---
void glm53_decode_q8_0(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 34u;
    const float d = glm53_h2f(uint(U16(b).v[0]));
    [[unroll]] for (uint i = 0u; i < 16u; i++) {
        const uint pair = uint(U16(b + 2u).v[i]);
        w[2u * i] = d * float(int(pair << 24) >> 24);
        w[2u * i + 1u] = d * float(int(pair << 16) >> 24);
    }
}

// --- Q4_K: f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes (144-byte
// blocks, 16-byte aligned in these rows). Task t: block t / 8, sub-block
// j = t % 8: 6-bit scale and min, the low (even j) or high nibbles of bytes
// 32 * (j / 2).. ---
void glm53_decode_q4_k(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 144u;
    const uint j = t & 7u;
    const uint dd = U32(b).v[0];
    const float d = glm53_h2f(dd & 0xffffu);
    const float dmin = glm53_h2f(dd >> 16u);
    // Scale bytes q[0..12] as three words.
    const uvec3 sw = uvec3(U32(b + 4u).v[0], U32(b + 8u).v[0], U32(b + 12u).v[0]);
    #define Q4K_BYTE(i) ((sw[(i) >> 2u] >> (8u * ((i) & 3u))) & 0xffu)
    uint sc, m;
    if (j < 4u) {
        sc = Q4K_BYTE(j) & 63u;
        m = Q4K_BYTE(j + 4u) & 63u;
    } else {
        sc = (Q4K_BYTE(j + 4u) & 0xfu) | ((Q4K_BYTE(j - 4u) >> 6u) << 4u);
        m = (Q4K_BYTE(j + 4u) >> 4u) | ((Q4K_BYTE(j) >> 6u) << 4u);
    }
    #undef Q4K_BYTE
    const float scale = d * float(sc);
    const float minv = dmin * float(m);
    const uint shift = (j & 1u) * 4u;
    const uint64_t qs = b + 16u + (j >> 1u) * 32u;
    [[unroll]] for (uint q = 0u; q < 2u; q++) {
        const uvec4 v = U32x4(qs + 16u * q).v[0];
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            const uint word = v[c] >> shift;
            [[unroll]] for (uint k = 0u; k < 4u; k++) {
                w[16u * q + 4u * c + k] = scale * float((word >> (8u * k)) & 0xfu) - minv;
            }
        }
    }
}

// --- Q2_K: 16 scale bytes (low nibble scale, high nibble min), 64 bytes of
// 2-bit codes, f16 d, f16 dmin (84-byte blocks, 4-byte aligned). Task t:
// block t / 8, groups g = 2 (t % 8) and g + 1, which read the 32 bytes at
// 32 * ((t % 8) / 4) at bit 2 * (t % 4). ---
void glm53_decode_q2_k(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 84u;
    const uint j = t & 7u;
    const uint dd = U32(b + 80u).v[0];
    const float d = glm53_h2f(dd & 0xffffu);
    const float dmin = glm53_h2f(dd >> 16u);
    const uint sc2 = uint(U16(b + 2u * j).v[0]);
    const uint shift = (j & 3u) * 2u;
    const uint64_t qs = b + 16u + (j >> 2u) * 32u;
    [[unroll]] for (uint h = 0u; h < 2u; h++) {
        const uint sc = (sc2 >> (8u * h)) & 0xffu;
        const float scale = d * float(sc & 0xfu);
        const float minv = dmin * float(sc >> 4u);
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            const uint word = U32(qs + 16u * h + 4u * c).v[0] >> shift;
            [[unroll]] for (uint k = 0u; k < 4u; k++) {
                w[16u * h + 4u * c + k] = scale * float((word >> (8u * k)) & 3u) - minv;
            }
        }
    }
}

// --- IQ2_XXS: f16 d, then per 32 values two words: four grid indices and
// four 7-bit sign indices plus a 4-bit scale (66-byte blocks, 2-byte
// aligned). The sign byte of index s is s with an even-parity eighth bit. ---
uint glm53_iq2xxs_grid[512] = uint[512](
#include "glm53_iq2xxs_grid.glsl"
);

void glm53_decode_iq2_xxs(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t >> 3u) * 66u;
    const uint ib = t & 7u;
    const float d = glm53_h2f(uint(U16(b).v[0]));
    const uint64_t q = b + 2u + 8u * ib;
    const uint aux0 = uint(U16(q).v[0]) | (uint(U16(q).v[1]) << 16u);
    const uint aux1 = uint(U16(q).v[2]) | (uint(U16(q).v[3]) << 16u);
    const float scale = 0.125 * d * float(2u * (aux1 >> 28u) + 1u);
    [[unroll]] for (uint l = 0u; l < 4u; l++) {
        const uint g = (aux0 >> (8u * l)) & 0xffu;
        const uint s7 = (aux1 >> (7u * l)) & 127u;
        const uint signs = s7 | ((bitCount(s7) & 1u) << 7u);
        const uint lo = glm53_iq2xxs_grid[2u * g];
        const uint hi = glm53_iq2xxs_grid[2u * g + 1u];
        [[unroll]] for (uint k = 0u; k < 8u; k++) {
            const float v = float(((k < 4u ? lo : hi) >> (8u * (k & 3u))) & 0xffu);
            w[8u * l + k] = scale * ((signs >> k & 1u) != 0u ? -v : v);
        }
    }
}

// --- BF16 and F32 rows (16-byte aligned here). ---
void glm53_decode_bf16(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 64u;
    [[unroll]] for (uint q = 0u; q < 4u; q++) {
        const uvec4 v = U32x4(b + 16u * q).v[0];
        [[unroll]] for (uint c = 0u; c < 4u; c++) {
            w[8u * q + 2u * c] = uintBitsToFloat(v[c] << 16u);
            w[8u * q + 2u * c + 1u] = uintBitsToFloat(v[c] & 0xffff0000u);
        }
    }
}

void glm53_decode_f32(uint64_t row, uint t, out float w[32]) {
    const uint64_t b = row + uint64_t(t) * 128u;
    [[unroll]] for (uint q = 0u; q < 8u; q++) {
        const vec4 v = F32x4(b + 16u * q).v[0];
        [[unroll]] for (uint c = 0u; c < 4u; c++) w[4u * q + c] = v[c];
    }
}

// The decoder of a compile-time type id.
void glm53_decode(uint ty, uint64_t row, uint t, out float w[32]) {
    if (ty == GLM53_Q8_0) glm53_decode_q8_0(row, t, w);
    else if (ty == GLM53_Q4_K) glm53_decode_q4_k(row, t, w);
    else if (ty == GLM53_Q2_K) glm53_decode_q2_k(row, t, w);
    else if (ty == GLM53_IQ2_XXS) glm53_decode_iq2_xxs(row, t, w);
    else if (ty == GLM53_BF16) glm53_decode_bf16(row, t, w);
    else glm53_decode_f32(row, t, w);
}

// The dot of a decoded task with 32 f32 activations at `x` (16-byte aligned).
float glm53_task_dot(in float w[32], uint64_t x) {
    float acc = 0.0;
    [[unroll]] for (uint q = 0u; q < 8u; q++) {
        const vec4 v = F32x4(x + 16u * q).v[0];
        acc += w[4u * q] * v.x + w[4u * q + 1u] * v.y + w[4u * q + 2u] * v.z + w[4u * q + 3u] * v.w;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Dot products straight from the blocks, for the decode GEMVs of the routed
// experts, which are bound by instructions and loads per weight: the codes
// become exact f16 values through a table, signs are an XOR of the f16 sign
// bits, and a group's scale multiplies its sum once instead of every
// weight. On G17C an f32 fmul widens an f16 source for free but an ffma
// takes two moves for it, so the products are fmuls and the sums separate
// adds (`precise`). The products round in another order than
// glm53_decode + glm53_task_dot; the decoded values are the same.
// ---------------------------------------------------------------------------
// A dynamically indexed table must be a plain global with an initializer
// (the compiler then reads it from the shader's constant data); a `const`
// array is copied to per-thread scratch first.
uvec4 glm53_iq2xxs_grid_h[256] = uvec4[256](
#include "glm53_iq2xxs_grid_h.glsl"
);
uvec4 glm53_iq2xxs_signs_h[128] = uvec4[128](
#include "glm53_iq2xxs_signs_h.glsl"
);

// IQ2_XXS task t of `row` against the 32 activations xv (as decoded by
// glm53_decode_iq2_xxs, then dotted). Loads index the row as 16-bit words
// (66-byte blocks are 33 words).
float glm53_dot_iq2_xxs(uint64_t row, uint t, in vec4 xv[8]) {
    const uint bw = (t >> 3u) * 33u;
    const uint qw = bw + 1u + 4u * (t & 7u);
    const float d = glm53_h2f(uint(U16(row).v[bw]));
    const uint aux0 = uint(U16(row).v[qw]) | (uint(U16(row).v[qw + 1u]) << 16u);
    const uint aux1 = uint(U16(row).v[qw + 2u]) | (uint(U16(row).v[qw + 3u]) << 16u);
    float s = 0.0;
    [[unroll]] for (uint l = 0u; l < 4u; l++) {
        const uint gi = (aux0 >> (8u * l)) & 0xffu;
        const uint si = (aux1 >> (7u * l)) & 127u;
#if defined(GLM53_SIGN_ALU)
        // Signs of values p and p + 4 to bits 15 and 31 of word p: q holds
        // sign k at bits k and 12 + k (the parity bit, sign 7, at 19).
        const uint q = si + (si << 12u) + ((bitCount(si) & 1u) << 19u);
        const uvec4 sg = (uvec4(q) << uvec4(15u, 14u, 13u, 12u)) & 0x80008000u;
#else
        const uvec4 sg = glm53_iq2xxs_signs_h[si];
#endif
        const uvec4 g = glm53_iq2xxs_grid_h[gi] ^ sg;
        const vec4 a = xv[2u * l], b = xv[2u * l + 1u];
        const vec2 p0 = unpackHalf2x16(g.x), p1 = unpackHalf2x16(g.y);
        const vec2 p2 = unpackHalf2x16(g.z), p3 = unpackHalf2x16(g.w);
        precise float e = ((p0.x * a.x + p0.y * b.x) + (p1.x * a.y + p1.y * b.y))
                        + ((p2.x * a.z + p2.y * b.z) + (p3.x * a.w + p3.y * b.w));
        s += e;
    }
    return 0.125 * d * float(2u * (aux1 >> 28u) + 1u) * s;
}
