#!/usr/bin/env python3
# SPLIT-BALANCE DESK P0 instrument: per-die weight bytes for the TP4
# name-keyed axis sharding (zero-GPU).
#
# Rule implemented from src/llama-model.cpp (llama_meta_device_get_split_state):
#   :333-357  tensor-name pattern table
#   :383-501  axis table (AXIS_0 / AXIS_1 / MIRRORED) + segments fallback
#   :386-393  get_il_eff: layer index within its (is_recr, is_swa) class
#   :403-407  rotation = get_il_eff(il) % n_devices (non-blk: n_layer % n_devices)
#   :503-559  get_split_segments (QWEN35 branch: qkv/gate/dt/a/alpha/beta/conv1d
#             interleave at K-scale)
#   :566-645  get_split_granularity (recr: lcm(blck,128)-class; attn: head-based;
#             ffn: lcm(blck,128); else 1)
#   :649-683  boundary math:
#               scan[j] = cumsum(tensor_split[(j + rotation) % n_devices])
#               high_j  = ne_s*(j+1)/n_devices           (default, all-zero)
#                       = ne_s*scan[j]/sum               (given split)
#               if high % g_s: high -= high % g_s        (floor to granularity)
#               die (j + rotation) % n_devices gets [low, high); last gets rest
# Tensor bytes from GGUF offset deltas (offset-delta law).
# MIRRORED tensors: full copy on every die.
import sys, os, re, math, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'gguf-py'))
import gguf

GGUF = sys.argv[1] if len(sys.argv) > 1 else '/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf'
NDEV = 4
SPLIT = None
if len(sys.argv) > 2:
    SPLIT = [float(x) for x in sys.argv[2].split(',')]
    assert len(SPLIT) == NDEV

r = gguf.GGUFReader(GGUF)
kv = {name: r.fields[name] for name in r.fields}

def u32(name):
    f = kv.get(name)
    return int(f.parts[-1][0]) if f is not None else None

n_embd   = u32('qwen35.embedding_length')
n_ff     = u32('qwen35.feed_forward_length')
n_head   = u32('qwen35.attention.head_count')
n_head_kv= u32('qwen35.attention.head_count_kv')
head_k   = u32('qwen35.attention.key_length')
blk_cnt  = u32('qwen35.block_count')
n_nextn  = u32('qwen35.nextn_predict_layers') or 0
d_state  = u32('qwen35.ssm.state_size')
n_group  = u32('qwen35.ssm.group_count')
dt_rank  = u32('qwen35.ssm.time_step_rank')
d_conv   = u32('qwen35.ssm.conv_kernel')
n_vocab  = None  # derived from token_embd dims

# qwen35.cpp: no recurrent_layer_count array in this GGUF -> interval fallback
# (src/models/qwen35.cpp:21-26): is_recr[i] = (i < n_layer()) && ((i+1)%interval != 0)
# n_layer() = 64 trunk blocks, blk.64 = MTP (dense attn, non-recr), n_layer_all = 65
full_attn_interval = u32('qwen35.full_attention_interval') or 4
n_layer_trunk = blk_cnt - n_nextn            # 64
n_layer_all   = blk_cnt                      # 65
is_recr = [ (i < n_layer_trunk) and ((i + 1) % full_attn_interval != 0) for i in range(n_layer_all) ]
is_swa  = [ False ] * n_layer_all

# get_il_eff (llama-model.cpp:386-393)
def il_eff(il):
    return sum(1 for p in range(il) if is_recr[p] == is_recr[il] and is_swa[p] == is_swa[il])

# derived dims (src/models/qwen35.cpp:60-63, 85-97)
key_dim  = d_state * n_group          # 128*16 = 2048
value_dim= d_state * dt_rank          # 128*48 = 6144
conv_dim = key_dim*2 + value_dim      # 10240
head_ratio = dt_rank // n_group       # 3
n_embd_q_full = head_k * n_head       # 6144
n_embd_gqa    = head_k * n_head_kv    # 1024
gqa           = n_head // n_head_kv   # 6
n_embd_q      = gqa * head_k          # 1536 (llama-model.cpp:575-576 naming)

# ---- tensor inventory: name, dims, type, bytes via offset-delta law ----
tensors = []
for t in r.tensors:
    tensors.append((t.name, [int(x) for x in t.shape], t.tensor_type.name, t.data_offset, int(t.n_bytes)))
tensors.sort(key=lambda x: x[3])
fsize = os.path.getsize(GGUF)
for i in range(len(tensors)):
    off = tensors[i][3]
    end = tensors[i+1][3] if i+1 < len(tensors) else fsize
    tensors[i] = (tensors[i][0], tensors[i][1], tensors[i][2], off, end - off)
assert sum(t[4] for t in tensors) <= fsize

QBLCK = {'Q4_K':256,'Q5_K':256,'Q6_K':256,'Q3_K':256,'IQ3_XXS':256,'IQ4_XS':256,'IQ3_S':256,'Q2_K':256,'IQ2_XS':256,'Q8_0':32,'F32':1,'F16':1,'BF16':1}

# ---- axis + segments + granularity per the served rule ----
AX = {}
def classify(name, dims):
    m = re.match(r'blk\.(\d+)\.(.*)', name)
    if m and not name.startswith('blk.0.cache'):
        il = int(m.group(1)); rest = m.group(2)
    else:
        il = None; rest = name
    recr = is_recr[il] if il is not None else None
    axis = 'MIR'; segs = None; gran = None
    if rest == 'attn_q.weight':
        axis = 'A1'; segs = [(dims[1], 1)]
        gran = [math.lcm(2*n_embd_q, 256)]       # qwen35 q-gate doubling
    elif rest in ('attn_k.weight','attn_v.weight'):
        axis = 'A1'; segs = [(dims[1], 1)]
        gran = [math.lcm(n_embd_q, 256)//gqa]    # granularity_kv
    elif rest == 'attn_qkv.weight':
        axis = 'A1'; segs = [(key_dim, 2 + head_ratio)]
        gran = [math.lcm(math.lcm(256,128), d_state)]
    elif rest == 'attn_gate.weight':
        axis = 'A1'; segs = [(key_dim, head_ratio)]
        gran = [math.lcm(math.lcm(256,128), d_state)]
    elif rest == 'attn_output.weight':
        axis = 'A0'; segs = [(dims[0], 1)]
        gran = [math.lcm(n_embd_q, 256)]
    elif rest in ('attn_q_norm.weight','attn_k_norm.weight'):
        axis = 'MIR'
    elif rest == 'ssm_conv1d.weight':
        axis = 'A1'; segs = [(key_dim, 2 + head_ratio)]
        gran = [math.lcm(math.lcm(256,128), d_state)]
    elif rest == 'ssm_dt.bias' or rest == 'ssm_a':
        axis = 'A0'; segs = [(n_group, head_ratio)]
        gran = [math.lcm(math.lcm(256,128), d_state)//d_state]
    elif rest in ('ssm_alpha.weight','ssm_beta.weight'):
        axis = 'A1'; segs = [(n_group, head_ratio)]
        gran = [math.lcm(math.lcm(256,128), d_state)//d_state]
    elif rest == 'ssm_norm.weight':
        axis = 'MIR'
    elif rest == 'ssm_out.weight':
        axis = 'A0'; segs = [(key_dim, head_ratio)]
        gran = [math.lcm(math.lcm(256,128), d_state)]
    elif rest in ('ffn_gate.weight','ffn_up.weight'):
        axis = 'A1'; segs = [(dims[1], 1)]
        gran = [math.lcm(256, 128)]
    elif rest == 'ffn_down.weight':
        axis = 'A0'; segs = [(dims[0], 1)]
        gran = [math.lcm(256, 128)]
    elif rest == 'nextn.eh_proj.weight':
        axis = 'MIR'
    elif rest in ('attn_norm.weight','post_attention_norm.weight'):
        axis = 'MIR'
    elif rest in ('nextn.enorm.weight','nextn.hnorm.weight','nextn.shared_head_norm.weight'):
        axis = 'MIR'
    elif name == 'output.weight':
        axis = 'A1'; segs = [(dims[1], 1)]; gran = [1]
    elif name == 'output_norm.weight':
        axis = 'MIR'
    elif name == 'token_embd.weight':
        axis = 'MIR'
    else:
        raise SystemExit(f'unclassified tensor: {name}')
    return il, axis, segs, gran

def rot_of(name, il):
    if il is not None:
        return il_eff(il) % NDEV
    return n_layer_trunk % NDEV   # 64 % 4 = 0 (llama-model.cpp:407 non-blk branch)

# GGUF tensor dims vs created-tensor ne: gguf stores [ne0, ne1] same as code
rows_out = []
die_bytes = [0]*NDEV
per_layer = {}
for (name, dims, ttype, off, nbytes) in tensors:
    il, axis, segs, gran = classify(name, dims)
    widths = [0]*NDEV
    if axis == 'MIR':
        for d in range(NDEV): widths[d] = dims[0] if len(dims) == 1 else None
        # mirrored: full tensor on each die
        for d in range(NDEV): die_bytes[d] += nbytes
        share = [nbytes]*NDEV
    else:
        ne_s_total = dims[0] if axis == 'A0' else dims[1]
        rot = rot_of(name, il)
        share = [0]*NDEV
        for (ne_s, nr), g_s in zip(segs, gran):
            scan = []
            tot = 0.0
            for j in range(NDEV):
                w = (SPLIT[(j + rot) % NDEV] if SPLIT else 0.0)
                tot += w
                scan.append(tot)
            low = 0
            for j in range(NDEV - 1):
                if SPLIT and tot > 0:
                    high = int(ne_s * scan[j] / tot)
                else:
                    high = ne_s * (j + 1) // NDEV
                if g_s and high % g_s != 0:
                    high -= high % g_s
                widths[(j + rot) % NDEV] += (high - low) * nr
                low = high
            widths[(NDEV - 1 + rot) % NDEV] += (ne_s - low) * nr
        # bytes: fraction of axis width per die (row-uniform quant tensors)
        ne_ax = dims[0] if axis == 'A0' else dims[1]
        base = dims[1] if (axis == 'A0' and len(dims) > 1) else dims[0]
        for d in range(NDEV):
            share[d] = nbytes * widths[d] // ne_ax if False else int(round(nbytes * (widths[d] / ne_ax)))
            die_bytes[d] += share[d]
    rows_out.append((name, il, axis, ttype, nbytes, widths, share))

tot = sum(t[4] for t in tensors)
mir = sum(t[4] for t in tensors if classify(t[0], t[1])[1] == 'MIR')
print(f'GGUF: {GGUF}')
print(f'tensors: {len(tensors)}  total bytes: {tot} ({tot/1e9:.3f} GB)')
print(f'mirrored bytes (per-die full copy): {mir} ({mir/1e6:.1f} MB)')
print(f'sharded bytes: {(tot-mir)/1e9:.3f} GB -> /4 = {(tot-mir)/4/1e9:.4f} GB')
print(f'per-die TOTAL weight bytes (default split): ' + ', '.join(f'die{d}={die_bytes[d]} ({die_bytes[d]/1e9:.4f} GB)' for d in range(NDEV)))
mx, mn = max(die_bytes), min(die_bytes)
print(f'imbalance: max-min = {mx-mn} B ({(mx-mn)/mn*100:.4f} % of min), max/mean-1 = {(mx/(sum(die_bytes)/NDEV)-1)*100:.4f} %')
if SPLIT:
    print(f'split used: {SPLIT}')

# per-layer table
print()
print('per-block byte table (default split, bytes per die):')
print('blk | class | ffn_down | ffn_gate+up | qkv/gdn-attn | other | die totals equal?')
for il in range(n_layer_all):
    sel = [t for t in rows_out if t[1] == il]
    s = sum(t[4] for t in sel)
    sh = [sum(t[6][d] for t in sel) for d in range(NDEV)]
    cls = 'gdn' if is_recr[il] else ('mtp-attn' if il >= n_layer_trunk else 'attn')
    spread = max(sh) - min(sh)
    fd = sum(t[4] for t in sel if t[0].endswith('ffn_down.weight'))
    fu = sum(t[4] for t in sel if t[0].endswith(('ffn_gate.weight','ffn_up.weight')))
    at = sum(t[4] for t in sel if t[0].endswith(('attn_qkv.weight','attn_gate.weight','attn_q.weight','attn_k.weight','attn_v.weight','attn_output.weight','ssm_out.weight','ssm_conv1d.weight','ssm_dt.bias','ssm_a','ssm_alpha.weight','ssm_beta.weight')))
    ot = s - fd - fu - at
    print(f'{il:3d} | {cls:8s} | {fd:9d} | {fu:11d} | {at:12d} | {ot:8d} | die spread {spread}')

# verify a few against per-tensor listing
print()
print('sample per-tensor shares (default split):')
for nm in ('blk.0.ffn_down.weight','blk.0.ffn_gate.weight','blk.3.attn_q.weight','blk.3.attn_k.weight','blk.3.attn_output.weight','blk.0.attn_qkv.weight','blk.0.attn_gate.weight','blk.0.ssm_out.weight','blk.0.ssm_dt.bias','blk.0.ssm_a','output.weight','token_embd.weight','blk.64.nextn.eh_proj.weight','blk.0.ssm_conv1d.weight'):
    for t in rows_out:
        if t[0] == nm:
            print(f'  {nm:34s} {t[3]:7s} total={t[4]:12d} axis={t[2]} widths={t[5]} per-die={[f"{x/1e6:.2f}MB" for x in t[6]]}')
