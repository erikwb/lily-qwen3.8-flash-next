// Conversions attention.comp needs that common.glsl does not have: Metal's
// half(f) for the q8 caches' scales. Include after common.glsl.

// The bits of half(f) for 0 <= f <= 65504 (half's largest finite value),
// rounded to nearest even. The driver's float16_t(f) is not used: PORTING.md
// has it truncating, and float(float16_t(f)) of a register value comes back
// as f itself, not rounded at all (the q8 scatter test caught that).
uint f16_rne_bits(float f) {
    uint u = floatBitsToUint(f);
    // A normal half: the float's 13 low mantissa bits rounded away, the carry
    // free to reach the exponent, whose bias is 112 less.
    if (u >= (113u << 23)) return ((u + 0xFFFu + ((u >> 13) & 1u)) >> 13) - (112u << 10);
    // A subnormal one (or zero, or the smallest normal): a count of 2^-24.
    return uint(roundEven(f * 16777216.0));
}

// float(h) of the finite, non-negative half with the bits `bits`.
float f16_bits_value(uint bits) {
    uint e = bits >> 10, m = bits & 0x3FFu;
    return e == 0u ? float(m) * 5.9604644775390625e-8 : uintBitsToFloat(((e + 112u) << 23) | (m << 13));
}
