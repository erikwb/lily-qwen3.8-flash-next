// Conversions qsa.comp needs that common.glsl does not have. Include after
// common.glsl.
//
// Metal's bfloat(f) rounds f to the nearest bfloat, ties to even. The
// driver's bfloat16_t(f) truncates (Mesa lowers f2bf to the high half of the
// float's bits), so the kernels round on the bits: the same value for every
// finite f and the infinities, and a NaN stays one.
uint qsa_bf16_bits(float f) {
    uint u = floatBitsToUint(f);
    uint r = u + 0x7FFFu + ((u >> 16) & 1u);
    // The carry could run through the sign of an all-ones NaN.
    return (u & 0x7FFFFFFFu) > 0x7F800000u ? u | 0x00400000u : r;
}
uvec4 qsa_bf16_bits(vec4 f) {
    uvec4 u = floatBitsToUint(f);
    uvec4 r = u + 0x7FFFu + ((u >> 16) & 1u);
    return mix(r, u | 0x00400000u, greaterThan(u & 0x7FFFFFFFu, uvec4(0x7F800000u)));
}
bfloat16_t qsa_bf16(float f) {
    return uintBitsToBFloat16EXT(uint16_t(qsa_bf16_bits(f) >> 16));
}
bf16vec4 qsa_bf16(vec4 f) {
    return uintBitsToBFloat16EXT(u16vec4(qsa_bf16_bits(f) >> 16));
}
// Four bfloats as the two words they are in memory (a `bf16vec4` through a
// 2-byte aligned reference is four 16-bit accesses).
uvec2 qsa_bf16_words(vec4 f) {
    uvec4 r = qsa_bf16_bits(f) >> 16;
    return uvec2(r.x | (r.y << 16), r.z | (r.w << 16));
}
bf16vec4 qsa_bf16_of_words(uvec2 w) {
    return uintBitsToBFloat16EXT(u16vec4(unpack16(w.x), unpack16(w.y)));
}
// float(bfloat(f)): f at bfloat precision.
float qsa_round_bf16(float f) {
    return uintBitsToFloat(qsa_bf16_bits(f) & 0xFFFF0000u);
}
// qsa_bf16 of values that are never NaN (a q8 byte times its scale), without
// the NaN test.
bf16vec4 qsa_bf16_finite(vec4 f) {
    uvec4 u = floatBitsToUint(f);
    return uintBitsToBFloat16EXT(u16vec4((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16));
}
