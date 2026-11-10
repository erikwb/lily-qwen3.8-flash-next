// What moe.comp and tests/vulkan/moe_test.comp share (on Metal the tests'
// source is moe.metal followed by moe_test.metal). Include after common.glsl.
//
// Arithmetic that a test demands bit-identical between two kernels is
// `precise` in both: this compiler otherwise fuses and reassociates each
// kernel's sums as that kernel's surroundings suggest.

const uint MOE_MAX_E = 1024;
const uint MOE_MAX_K = 16;
const uint MOE_ROUTER_MAX_K = 32;
const uint MOE_SMALLM_MAX_S = 256;

#define MOE_NAN uintBitsToFloat(0x7FC00000u)
#define MOE_NEG_INFINITY (-1.0 / 0.0)

// Masked Q4 nibbles are rescaled before affine dequantization.
vec2 qdot_word_masked(uint word, vec4 xlo, vec4 xhi) {
    const vec4 inv = vec4(1.0, 1.0 / 16.0, 1.0 / 256.0, 1.0 / 4096.0);
    uint wl = word & 0xFFFFu;
    uint wh = word >> 16;
    vec4 ql = vec4(float(wl & 0x000Fu), float(wl & 0x00F0u), float(wl & 0x0F00u), float(wl & 0xF000u));
    vec4 qh = vec4(float(wh & 0x000Fu), float(wh & 0x00F0u), float(wh & 0x0F00u), float(wh & 0xF000u));
    precise vec2 d = vec2(dot(ql, xlo * inv) + dot(qh, xhi * inv), dot(xlo, vec4(1.0)) + dot(xhi, vec4(1.0)));
    return d;
}

// The eight Q4 codes of a word, low and high half.
vec4 q4_lo(uint word) {
    return vec4(float((word >> 0) & 0xFu), float((word >> 4) & 0xFu), float((word >> 8) & 0xFu), float((word >> 12) & 0xFu));
}
vec4 q4_hi(uint word) {
    return vec4(float((word >> 16) & 0xFu), float((word >> 20) & 0xFu), float((word >> 24) & 0xFu), float((word >> 28) & 0xFu));
}

// Eight consecutive bf16 of an activation row as floats, from one 16-byte
// load `x` (a bf16 widens to the float whose high half it is). The Metal
// kernels read them as two bfloat4; on this driver the number of loads is
// what a gather costs, so the port reads the same bytes in half as many (as
// words: a row starts on a 4-byte boundary, where bfloat4 asks for 8).
vec4 bf16x8_lo(uvec4 x) {
    return uintBitsToFloat(uvec4(x.x << 16, x.x & 0xFFFF0000u, x.y << 16, x.y & 0xFFFF0000u));
}
vec4 bf16x8_hi(uvec4 x) {
    return uintBitsToFloat(uvec4(x.z << 16, x.z & 0xFFFF0000u, x.w << 16, x.w & 0xFFFF0000u));
}

// moe.metal's MOE_SMALLM_DOWN_COMBINE_KERNEL(NAME, R4, PSG): the including
// file defines MOE_SMALLM_DOWN_R4 and MOE_SMALLM_DOWN_PSG for a kernel that
// is one.
//
// One threadgroup per (2 * R4 output columns, token row), 2 * PSG subgroups:
// subgroup (cg, pp) owns columns cg * R4.. of the pairs k = pp, pp + PSG, ...
#ifdef MOE_SMALLM_DOWN_R4
#define R4 MOE_SMALLM_DOWN_R4
#define PSG MOE_SMALLM_DOWN_PSG
shared float part[2 * MOE_MAX_K * R4];
void main() {
    uint K = arg_u32(9), GS = arg_u32(10), N = arg_u32(11), TOPK = arg_u32(12);
    uint64_t codes = arg(0), scales_at = arg(1), biases_at = arg(2), x = arg(3);
    U32 indices = U32(arg(4));
    F32 scores = F32(arg(5));
    // The subgroup's own position (Metal: the thread index over and modulo 32).
    uint sgi = simdgroup_index_in_threadgroup, sl = thread_index_in_simdgroup;
    uint cg = sgi / PSG, pp = sgi % PSG;
    uint row = threadgroup_position_in_grid.y;
    uint row0 = (threadgroup_position_in_grid.x * 2 + cg) * R4;
    uint words = K / 8, blocks = K / 16, bpg = GS / 16, groups = K / GS;
    for (uint k = pp; k < TOPK; k += PSG) {
        uint pair = row * TOPK + k;
        uint e = indices.v[pair];
        // The expert's R4 rows from row0 are consecutive: row r4's block i is
        // wrow.v[r4 * blocks + i], its group g scales.v[r4 * groups + g]. A
        // row number fits 32 bits (the offsets it scales to do not).
        uint grow = e * N + row0;
        U32x2 wrow = U32x2(codes + 4 * (uint64_t(grow) * words));
        BF16 scales = BF16(scales_at + 2 * (uint64_t(grow) * groups)), biases = BF16(biases_at + 2 * (uint64_t(grow) * groups));
        precise float acc[R4];
        [[unroll]] for (uint r4 = 0; r4 < R4; ++r4) acc[r4] = 0.0;
        U32x4 xv0 = U32x4(x + 2 * (uint64_t(pair) * K));
        for (uint i = sl; i < blocks; i += 32) {
            uint g = i / bpg;
            uvec2 w2[R4];
            float s[R4], b[R4];
            [[unroll]] for (uint r4 = 0; r4 < R4; ++r4) {
                w2[r4] = wrow.v[r4 * blocks + i];
                s[r4] = float(scales.v[r4 * groups + g]);
                b[r4] = float(biases.v[r4 * groups + g]);
            }
            uvec4 xa = xv0.v[2 * i], xb = xv0.v[2 * i + 1];
            vec4 x0 = bf16x8_lo(xa), x1 = bf16x8_hi(xa), x2 = bf16x8_lo(xb), x3 = bf16x8_hi(xb);
            [[unroll]] for (uint r4 = 0; r4 < R4; ++r4) {
                precise vec2 d = qdot_word_masked(w2[r4].x, x0, x1);
                d += qdot_word_masked(w2[r4].y, x2, x3);
                acc[r4] += s[r4] * d.x + b[r4] * d.y;
            }
        }
        [[unroll]] for (uint r4 = 0; r4 < R4; ++r4) {
            float r = subgroupAdd(acc[r4]);
            if (sl == 0) part[(cg * MOE_MAX_K + k) * R4 + r4] = float(bfloat16_t(r));
        }
    }
    barrier();
    if (pp == 0 && sl < R4) {
        uint r4 = sl;
        precise float combined = 0.0;
        for (uint k = 0; k < TOPK; ++k) combined += scores.v[row * TOPK + k] * part[(cg * MOE_MAX_K + k) * R4 + r4];
        precise float gsig = 1.0 / (1.0 + exp(-float(BF16(arg(7)).v[row])));
        uint64_t at = 2 * (uint64_t(row) * N + row0 + r4);
        precise float v = float(bfloat16_t(combined)) + gsig * float(BF16(arg(6) + at).v[0]);
        BF16(arg(8) + at).v[0] = bfloat16_t(v);
    }
}
#undef R4
#undef PSG
#endif
