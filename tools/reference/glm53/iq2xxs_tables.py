#!/usr/bin/env python3
"""Writes the f16 IQ2_XXS tables of src/kernels/vulkan/glm53_quant.glsl:

    tools/reference/glm53/iq2xxs_tables.py src/kernels/vulkan/glm53_iq2xxs_grid.glsl \
        src/kernels/vulkan/glm53_iq2xxs_grid_h.glsl src/kernels/vulkan/glm53_iq2xxs_signs_h.glsl

from the grid in glm53_iq2xxs_grid.glsl (ggml's, as u32 pairs)."""
import re, struct, sys
src = open(sys.argv[1]).read()
words = [int(w, 16) for w in re.findall(r'0x([0-9a-fA-F]+)u', src)]
assert len(words) == 512
def h(v): return struct.unpack('<H', struct.pack('<e', float(v)))[0]
out = []
out.append("// ggml's IQ2_XXS grid (glm53_iq2xxs_grid.glsl) as f16: entry g holds its")
out.append("// eight unsigned values, value p in the low and p + 4 in the high half of")
out.append("// word p. Generated from glm53_iq2xxs_grid.glsl.")
ents = []
for g in range(256):
    lo, hi = words[2*g], words[2*g+1]
    vals = [(lo >> (8*k)) & 0xff for k in range(4)] + [(hi >> (8*k)) & 0xff for k in range(4)]
    ws = [h(vals[p]) | (h(vals[p+4]) << 16) for p in range(4)]
    ents.append("uvec4(" + ", ".join("0x%08xu" % w for w in ws) + ")")
for i in range(0, 256, 2):
    out.append("    " + ", ".join(ents[i:i+2]) + ("," if i + 2 < 256 else ""))
open(sys.argv[2], 'w').write("\n".join(out) + "\n")
# signs: s7 -> 8 sign bits with even parity -> XOR masks on f16 pairs
out = ["// IQ2_XXS sign masks: entry s (the 7-bit sign index) holds the f16 sign bits",
       "// of its eight values (s plus the even-parity eighth bit) laid out as the",
       "// grid words of glm53_iq2xxs_grid_h.glsl. Generated."]
ents = []
for s in range(128):
    b = s | ((bin(s).count('1') & 1) << 7)
    ws = [((b >> p) & 1) * 0x8000 | ((b >> (p+4)) & 1) * 0x80000000 for p in range(4)]
    ents.append("uvec4(" + ", ".join("0x%08xu" % w for w in ws) + ")")
for i in range(0, 128, 2):
    out.append("    " + ", ".join(ents[i:i+2]) + ("," if i + 2 < 128 else ""))
open(sys.argv[3], 'w').write("\n".join(out) + "\n")
