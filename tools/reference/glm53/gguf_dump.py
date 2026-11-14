import struct, sys, json
f = open(sys.argv[1], 'rb')
def rd(fmt):
    s = struct.calcsize(fmt); return struct.unpack('<'+fmt, f.read(s))
def rstr():
    (n,) = rd('Q'); return f.read(n).decode('utf-8', 'replace')
T = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rval(t):
    if t == 8: return rstr()
    if t == 9:
        (at,) = rd('I'); (n,) = rd('Q')
        if at == 8: return [rstr() for _ in range(n)]
        fmt = T[at]; return list(struct.unpack('<%d%s' % (n, fmt), f.read(n*struct.calcsize(fmt))))
    return rd(T[t])[0]
magic, ver = rd('4sI'); nt, nkv = rd('QQ')
kv = {}
for _ in range(nkv):
    k = rstr(); (t,) = rd('I'); kv[k] = rval(t)
tensors = []
for _ in range(nt):
    name = rstr(); (nd,) = rd('I'); dims = rd('%dQ' % nd); (ty,) = rd('I'); (off,) = rd('Q')
    tensors.append((name, dims, ty, off))
align = kv.get('general.alignment', 32)
pos = f.tell(); data0 = (pos + align - 1)//align*align
out = {'kv': {}, 'data_offset': data0, 'tensors': tensors}
for k, v in kv.items():
    if isinstance(v, list) and len(v) > 16:
        out['kv'][k] = {'len': len(v), 'head': v[:8], 'tail': v[-4:]}
    else: out['kv'][k] = v
json.dump(out, sys.stdout)
if len(sys.argv) > 2:
    json.dump({k: kv[k] for k in kv if k.startswith('tokenizer.')}, open(sys.argv[2], 'w'))
