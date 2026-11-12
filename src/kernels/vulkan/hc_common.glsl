// f32 -> bf16 rounded to nearest even, as Metal's bfloat(f) does. On this
// driver (Mesa's agx3 backend sets lower_bfloat16_conversions) the GLSL
// conversion bfloat16_t(f) truncates instead, which breaks the tests that
// compare against CPU values rounded to nearest. NaN stays a quiet NaN.
uint bf16_rne_bits(uint u) {
    return (u & 0x7FFFFFFFu) > 0x7F800000u ? (u >> 16) | 0x40u
                                            : (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
bfloat16_t bf16_rne(float f) {
    return uintBitsToBFloat16EXT(uint16_t(bf16_rne_bits(floatBitsToUint(f))));
}
bf16vec4 bf16_rne(vec4 v) {
    uvec4 u = floatBitsToUint(v);
    return uintBitsToBFloat16EXT(u16vec4(bf16_rne_bits(u.x), bf16_rne_bits(u.y),
                                         bf16_rne_bits(u.z), bf16_rne_bits(u.w)));
}
