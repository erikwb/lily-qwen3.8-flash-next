/* Golden dequantization data for lily's GGUF block decoders. Blocks come from
 * ds4's quantizer (gguf-tools/quants.c) and from random bytes; every value is
 * decoded with ds4's reference formulas (ds4.c: q2_k_value_f32,
 * ds4_vec_dot_q4_K_f32, ds4_vec_dot_iq2_xxs_f32, the Q8_0 matvec reference). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "quants.h"
#define QK_K 256
#include "iq2tables.inc"
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; } block_q2_K;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct { uint16_t d; uint16_t qs[32]; } block_iq2_xxs;
static float h2f(uint16_t h) { return ds4q_f16_to_f32(h); }
static float q2k(const block_q2_K *xb, uint32_t idx) {
    const uint32_t group = idx / 16u, l = idx - group * 16u;
    const uint32_t q_base = 32u * (group / 8u) + 16u * (group & 1u);
    const uint32_t shift = ((group / 2u) & 3u) * 2u;
    const uint32_t q = ((uint32_t)xb->qs[q_base + l] >> shift) & 0x03u;
    const uint32_t sc = xb->scales[group];
    return h2f(xb->d) * (float)(sc & 0x0fu) * (float)q - h2f(xb->dmin) * (float)(sc >> 4u);
}
static void q4sm(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
    else { *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}
static void q4k(const block_q4_K *x, float *out) {
    const float d = h2f(x->d), dmin = h2f(x->dmin);
    for (int j = 0; j < 8; j++) {
        uint8_t sc, m; q4sm(j, x->scales, &sc, &m);
        const int byte_off = (j >> 1) * 32, shift = (j & 1) * 4;
        const float scale = d * (float)sc, minv = dmin * (float)m;
        for (int l = 0; l < 32; l++) out[j * 32 + l] = scale * (float)((x->qs[byte_off + l] >> shift) & 0x0F) - minv;
    }
}
static void iq2(const block_iq2_xxs *x, float *out) {
    const float d = h2f(x->d);
    const uint16_t *q2 = x->qs;
    uint32_t aux32[2]; const uint8_t *aux8 = (const uint8_t *)aux32;
    for (int ib32 = 0; ib32 < 8; ib32++) {
        memcpy(aux32, q2, 8); q2 += 4;
        const float scale = 0.125f * d * (float)(2u * (aux32[1] >> 28) + 1u);
        for (int l = 0; l < 4; l++) {
            const uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
            const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
            for (int j = 0; j < 8; j++) {
                const int v = (signs & kmask_iq2xs[j]) ? -(int)grid[j] : (int)grid[j];
                out[ib32 * 32 + l * 8 + j] = scale * (float)(int8_t)v;
            }
        }
    }
}
static void q80(const uint8_t *b, float *out) {
    uint16_t dbits; memcpy(&dbits, b, 2); const float d = h2f(dbits);
    for (int i = 0; i < 32; i++) out[i] = d * (float)(int8_t)b[2 + i];
}
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }
static float rndf(void) { return ((float)(rnd() & 0xffffff) / 16777216.0f - 0.5f) * 2.0f; }
/* A finite, normal-ish f16 scale for random-byte blocks. */
static uint16_t rnd_h(void) { return (uint16_t)((rnd() & 0x83ff) | ((5 + rnd() % 20) << 10)); }
static void emit(FILE *f, uint32_t type, uint32_t nblk, uint32_t blk_bytes, uint32_t blk_vals, const uint8_t *raw, const float *vals) {
    fwrite(&type, 4, 1, f); fwrite(&nblk, 4, 1, f);
    fwrite(raw, blk_bytes, nblk, f); fwrite(vals, 4, (size_t)nblk * blk_vals, f);
}
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "wb");
    enum { N = 2048, RB = 8 }; /* N quantized values (8 blocks of 256), RB random blocks */
    static float src[N], imat[N], vals[N + RB * 256];
    static uint8_t raw[N * 2];
    for (int i = 0; i < N; i++) { src[i] = rndf() * (i % 97 == 0 ? 8.0f : 1.0f); imat[i] = 0.5f + (rnd() % 100) / 100.0f; }
    /* Q8_0: quantized + random. */
    ds4q_quantize_chunk(DS4Q_TYPE_Q8_0, src, raw, 0, 1, N, NULL);
    for (int b = 0; b < N / 32; b++) q80(raw + b * 34, vals + b * 32);
    emit(f, 8, N / 32, 34, 32, raw, vals);
    for (int b = 0; b < RB; b++) { uint16_t h = rnd_h(); memcpy(raw + b * 34, &h, 2); for (int i = 0; i < 32; i++) raw[b * 34 + 2 + i] = rnd(); q80(raw + b * 34, vals + b * 32); }
    emit(f, 8, RB, 34, 32, raw, vals);
    /* Q4_K */
    ds4q_quantize_chunk(DS4Q_TYPE_Q4_K, src, raw, 0, 1, N, imat);
    for (int b = 0; b < N / 256; b++) q4k((block_q4_K *)(raw + b * 144), vals + b * 256);
    emit(f, 12, N / 256, 144, 256, raw, vals);
    for (int b = 0; b < RB; b++) { block_q4_K *x = (block_q4_K *)(raw + b * 144); for (int i = 0; i < 144; i++) raw[b * 144 + i] = rnd(); x->d = rnd_h(); x->dmin = rnd_h(); q4k(x, vals + b * 256); }
    emit(f, 12, RB, 144, 256, raw, vals);
    /* Q2_K */
    ds4q_quantize_chunk(DS4Q_TYPE_Q2_K, src, raw, 0, 1, N, imat);
    for (int b = 0; b < N / 256; b++) for (int i = 0; i < 256; i++) vals[b * 256 + i] = q2k((block_q2_K *)(raw + b * 84), i);
    emit(f, 10, N / 256, 84, 256, raw, vals);
    for (int b = 0; b < RB; b++) { block_q2_K *x = (block_q2_K *)(raw + b * 84); for (int i = 0; i < 84; i++) raw[b * 84 + i] = rnd(); x->d = rnd_h(); x->dmin = rnd_h(); for (int i = 0; i < 256; i++) vals[b * 256 + i] = q2k(x, i); }
    emit(f, 10, RB, 84, 256, raw, vals);
    /* IQ2_XXS */
    ds4q_quantize_init(DS4Q_TYPE_IQ2_XXS);
    ds4q_quantize_chunk(DS4Q_TYPE_IQ2_XXS, src, raw, 0, 1, N, imat);
    for (int b = 0; b < N / 256; b++) iq2((block_iq2_xxs *)(raw + b * 66), vals + b * 256);
    emit(f, 16, N / 256, 66, 256, raw, vals);
    for (int b = 0; b < RB; b++) { block_iq2_xxs *x = (block_iq2_xxs *)(raw + b * 66); for (int i = 0; i < 66; i++) raw[b * 66 + i] = rnd(); x->d = rnd_h(); iq2(x, vals + b * 256); }
    emit(f, 16, RB, 66, 256, raw, vals);
    fclose(f);
    return 0;
}
