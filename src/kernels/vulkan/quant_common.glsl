// What the affine Q4/Q8 kernels share (quant.comp, skinny.comp and their
// test kernels). Include after common.glsl.

// A Q4 code word's eight codes, low nibble first, as two float4.
vec4 q4_lo(uint word) { return vec4((uvec4(word) >> uvec4(0, 4, 8, 12)) & 0xFu); }
vec4 q4_hi(uint word) { return vec4((uvec4(word) >> uvec4(16, 20, 24, 28)) & 0xFu); }
// A Q8 code word's four codes, low byte first.
vec4 q8_codes(uint word) { return vec4((uvec4(word) >> uvec4(0, 8, 16, 24)) & 0xFFu); }

// bf16(w) of dequantized weights w = s * q + b, without bf16()'s test for a
// NaN (5 to 10% of the grouped GEMM's time): a NaN of this arithmetic is the
// default one or a bfloat operand's, its low half zero, and the rounding
// leaves such a NaN one.
bf16vec4 bf16_weight(vec4 w) {
    uvec4 u = floatBitsToUint(w);
    return uintBitsToBFloat16EXT(u16vec4((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16));
}

// Four bf16 values held as the bits of two words (Metal's
// as_type<bfloat4>(uint2)), widened: a bf16 is the high half of its float.
vec4 bf4_bits(uvec2 v) {
    return uintBitsToFloat(uvec4(v.x << 16, v.x & 0xFFFF0000u, v.y << 16, v.y & 0xFFFF0000u));
}

// Eight bf16 values at element `at` (a multiple of 8) of `p`, as their bits:
// one 16-byte load where `vec` says p is aligned for it.
uvec4 load_raw8(uint64_t p, uint at, bool vec) {
    if (vec) return U32x4(p).v[at / 8];
    U16 u = U16(p);
    return uvec4(uint(u.v[at]) | (uint(u.v[at + 1]) << 16), uint(u.v[at + 2]) | (uint(u.v[at + 3]) << 16),
                 uint(u.v[at + 4]) | (uint(u.v[at + 5]) << 16), uint(u.v[at + 6]) | (uint(u.v[at + 7]) << 16));
}
