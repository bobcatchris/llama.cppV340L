#!/usr/bin/env python3
# SPLIT-BALANCE DESK A2 instrument: grid-search --tensor-split ratios under
# the exact loader boundary math (floors + granularity + per-layer rotation)
# to minimize the predicted per-boundary wall max_d(s_d * t_d).
# t_d (per-byte time, die3=1) from the census: [0.893, 0.846, 0.964, 1.000].
import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'gguf-py'))
import gguf

GGUF = '/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf'
NDEV = 4
T = [0.893, 0.846, 0.964, 1.000]

# tensor classes present (per the GGUF dump + qwen35.cpp):
# (count, ne_s, nr, g, bytes_each, rot_cycle_uniform)
CLASSES = []
def add(count, ne_s, nr, g, bytes_each):
    CLASSES.append((count, ne_s, nr, g, bytes_each))

K = 256.0  # bytes per row per 256-blck element basis varies; use RELATIVE weights
# relative byte weight per row-unit per class (bpw x base row length, arbitrary units)
# attn layers (16 trunk + 1 mtp), gdn layers 48:
#   attn: q [5120,12288] Q4_K-class ~4.5-6.5bpw; use per-tensor byte inventory
# exact per-tensor bytes come from the audit run; approximate here with dims x bpw:
BPW = {'Q4_K':4.5,'Q5_K':5.5,'Q6_K':6.5625,'Q3_K':3.4375,'IQ3_XXS':3.0625,'IQ3_S':3.24,'IQ4_XS':4.25,'IQ2_XS':2.0625,'IQ2_XXS':2.0625,'F32':32,'Q8_0':8.5}
# class byte weights are mixed per layer; use the audit totals instead:
# from audit: per-block sharded per-die totals ~equal; the WIDTH math is
# byte-weight-proportional, so weight each class by bytes-per-width-unit
# computed from the GGUF inventory below.

r = gguf.GGUFReader(GGUF)
ts = sorted(r.tensors, key=lambda t: t.data_offset)
fsize = os.path.getsize(GGUF)
sizes = {}
for i, t in enumerate(ts):
    end = ts[i+1].data_offset if i+1 < len(ts) else fsize
    sizes[t.name] = end - t.data_offset

GDN_N, ATTN_N = 48, 16
MTP = 1
# per-class entries: (layers, ne_s, nr, g, bytes_total_all_layers, axis_base)
# bytes per width unit = total_bytes / (ne_s * nr) per layer
def cls(layers, ne_s, nr, g, suffix, total_bytes):
    per_layer = total_bytes / (layers + (1 if suffix.startswith('attn') and False else 0))
    return (layers, ne_s, nr, g, per_layer)

def add_class(entries, layers, ne_s, nr, g, total_bytes):
    entries.append((layers, ne_s, nr, g, total_bytes / layers))

E = []
# attn-class tensors (16 trunk + 1 mtp = 17 each; q/k/v/out/gates only on attn blocks)
q_b   = sum(v for k, v in sizes.items() if k.endswith('.attn_q.weight'))
k_b   = sum(v for k, v in sizes.items() if k.endswith('.attn_k.weight'))
v_b   = sum(v for k, v in sizes.items() if k.endswith('.attn_v.weight'))
o_b   = sum(v for k, v in sizes.items() if k.endswith('.attn_output.weight'))
qn_b  = sum(v for k, v in sizes.items() if k.endswith('.attn_q_norm.weight'))   # mirrored
gdn   = [k for k in sizes.items()]
add_class(E, 17, 12288, 1, 3072, q_b)
add_class(E, 17, 1024,  1, 256,  k_b)
add_class(E, 17, 1024,  1, 256,  v_b)
add_class(E, 17, 6144,  1, 1536, o_b)
# gdn-class tensors (48 layers)
qkv_b  = sum(v for k, v in sizes.items() if k.endswith('.attn_qkv.weight'))
gate_b = sum(v for k, v in sizes.items() if k.endswith('.attn_gate.weight'))
so_b   = sum(v for k, v in sizes.items() if k.endswith('.ssm_out.weight'))
cv_b   = sum(v for k, v in sizes.items() if k.endswith('.ssm_conv1d.weight'))
dt_b   = sum(v for k, v in sizes.items() if k.endswith('.ssm_dt.bias')) + sum(v for k, v in sizes.items() if k.endswith('.ssm_a'))
al_b   = sum(v for k, v in sizes.items() if k.endswith('.ssm_alpha.weight')) + sum(v for k, v in sizes.items() if k.endswith('.ssm_beta.weight'))
add_class(E, 48, 2048, 5, 256, qkv_b)
add_class(E, 48, 2048, 3, 256, gate_b)
add_class(E, 48, 2048, 3, 256, so_b)
add_class(E, 48, 2048, 5, 256, cv_b)
add_class(E, 48, 16,   3, 2,   dt_b)
add_class(E, 48, 16,   3, 2,   al_b)
# ffn tensors on ALL 65 blocks (fine granularity 256 on 17408)
fd_b = sum(v for k, v in sizes.items() if k.endswith('.ffn_down.weight'))
fg_b = sum(v for k, v in sizes.items() if k.endswith('.ffn_gate.weight')) + sum(v for k, v in sizes.items() if k.endswith('.ffn_up.weight'))
add_class(E, 65, 17408, 1, 256, fd_b)
add_class(E, 65, 17408, 1, 256, fg_b)
# output.weight (non-block, rot 0 always, g=1)
ow_b = sizes['output.weight']

def widths(ne_s, nr, g, rot, w):
    scan = []; tot = 0.0
    for j in range(NDEV):
        tot += w[(j + rot) % NDEV]; scan.append(tot)
    ws = [0]*NDEV; low = 0
    for j in range(NDEV - 1):
        high = int(ne_s * scan[j] / tot)
        if g and high % g: high -= high % g
        ws[(j + rot) % NDEV] += (high - low) * nr
        low = high
    ws[(NDEV - 1 + rot) % NDEV] += (ne_s - low) * nr
    return ws

def shares(w):
    num = [0.0]*NDEV
    for layers, ne_s, nr, g, per_layer in E:
        acc = [0.0]*NDEV
        for rot in range(4):
            ws = widths(ne_s, nr, g, rot, w)
            acc = [a + x * layers / 4.0 for a, x in zip(acc, ws)]  # rotations uniform
        unit = per_layer / (ne_s * nr)
        num = [n + a * unit for n, a in zip(num, acc)]
    # output.weight: rot 0 only
    ws = widths(129272, 1, 1, 0, w)
    unit = ow_b / 129272
    num = [n + x * unit for n, x in zip(num, ws)]
    stot = sum(num)
    return [x / stot for x in num]

def objective(w):
    s = shares(w)
    return max(s[d] * T[d] for d in range(NDEV)), s

best = None
import numpy as _np
grid = sorted(set([round(x,3) for x in _np.arange(0.96, 1.241, 0.01)]))
s0, _ = objective([1,1,1,1])
print(f'default [1,1,1,1]: max s_d t_d = {s0:.5f} -> wall compute = {s0*4*579.4:.1f} us/boundary (should be ~579.4)')
print('reference: default max compute = 579.4 us; balanced ideal = 534.2 us')
for w0, w1, w2 in itertools.product(grid, repeat=3):
    w = [w0, w1, w2, 1.00]
    if sum(w[:3]) < 2.7 or sum(w[:3]) > 3.5:
        continue
    obj, s = objective(w)
    if best is None or obj < best[0]:
        best = (obj, w, s)
print(f'best: ratios {best[1]} max s_d t_d = {best[0]:.5f} -> predicted wall compute {best[0]*4*579.4:.1f} us/boundary (today 579.4)')
print('achieved shares: ' + ', '.join(f'die{d}={best[2][d]*100:.2f}%' for d in range(NDEV)))
sv = 579.4 - best[0]*4*579.4
print(f'saving/boundary = {sv:.1f} us; x128 = {sv*128/1000:.2f} ms/round verify; x136 incl draft ~{sv*136/1000:.2f} ms')
