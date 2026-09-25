#!/usr/bin/env python3
"""E5 tap analysis: layer-wise x dumps (NINFER_C_LAYERS) for T=64 (PASS) vs T=65 (FAIL).

For every layer's post-residual x dump (rank 0):
  - per-row fp32-from-bf16 stats for the LAST rows + row 0
  - NaN/Inf counts in the last row
  - cross-run diff of the identical-token prefix rows: bit-mismatch count + max |delta|
Verdict grammar: ulp-class drift (small deltas, scattered) vs LARGE divergence (semantic).
"""
import array, glob, os, struct, sys

D = "/tmp/coh_tap"
HIDDEN = 5120

def load_rows(path):
    a = array.array('H')
    with open(path, 'rb') as f:
        blob = f.read()
    assert blob[:4] == b'MBP1', path
    a.frombytes(blob[20:])   # 20-byte MBP1 provenance header, then raw bf16
    n = len(a)
    assert n % HIDDEN == 0, (path, n)
    return a, n // HIDDEN

def row_f32(a, r):
    lo = r * HIDDEN
    out = array.array('f', bytes(4 * HIDDEN))
    for i in range(HIDDEN):
        out[i] = struct.unpack('<f', struct.pack('<I', a[lo + i] << 16))[0]
    return out

def stats(a, rows):
    res = {}
    T = len(a) // HIDDEN
    for r in rows:
        if r >= T: continue
        v = row_f32(a, r)
        s = sum(x * x for x in v)
        nan = sum(1 for x in v if x != x)
        inf = sum(1 for x in v if x in (float('inf'), float('-inf')))
        mx = max(abs(x) for x in v)
        res[r] = (s ** 0.5, nan, inf, mx)
    return res

def diff_rows(a, b, r):
    """bit-mismatch count + max |f32 delta| for one row across two dumps."""
    lo = r * HIDDEN
    mism = 0; mx = 0.0
    for i in range(HIDDEN):
        wa, wb = a[lo + i], b[lo + i]
        if wa != wb:
            mism += 1
            fa = struct.unpack('<f', struct.pack('<I', wa << 16))[0]
            fb = struct.unpack('<f', struct.pack('<I', wb << 16))[0]
            d = abs(fa - fb)
            if d > mx: mx = d
    return mism, mx

def files_for(n_elems, lane="l00"):
    fs = glob.glob(f"{D}/LA_r*_p*_&_{lane}_n{n_elems:08d}_*.bin".replace('&', '*'))
    out = {}
    for f in fs:
        base = os.path.basename(f)
        rnd = int(base.split('_r')[1].split('_')[0])
        pss = int(base.split('_p')[1].split('_')[0])
        out.setdefault((rnd, pss), f)
    return out

def latest_by_layer(n_elems):
    """one file per layer: the LATEST pass for that n (real request, not warmup/replay)."""
    fs = glob.glob(f"{D}/LA_r*_p*_l00_n{n_elems:08d}_*.bin")
    best = {}
    for f in fs:
        base = os.path.basename(f)
        rnd = int(base.split('_r')[1].split('_')[0])
        pss = int(base.split('_p')[1].split('_')[0])
        if rnd not in best or pss >= best[rnd][0]:
            best[rnd] = (pss, f)
    return {r: v[1] for r, v in best.items()}

def main():
    t64 = latest_by_layer(HIDDEN * 64)
    t65 = latest_by_layer(HIDDEN * 65)
    nl = min(len(t64), len(t65))
    print(f"layers: T64={len(t64)} T65={len(t65)} -> comparing {nl}")
    print("\n=== LAST-ROW NORM TRAJECTORY (T64 row63 vs T65 row64) + NaN ===")
    print(f"{'layer':>5} {'|x63|@T64':>12} {'|x64|@T65':>12} {'nan64':>6} {'max|x64|':>12}")
    for r in range(nl):
        a64, T1 = load_rows(t64[r]); a65, T2 = load_rows(t65[r])
        s64 = stats(a64, [T1 - 1])[T1 - 1]
        s65 = stats(a65, [T2 - 1])[T2 - 1]
        print(f"{r:>5} {s64[0]:>12.3f} {s65[0]:>12.3f} {s65[1]:>6} {s65[3]:>12.4f}")
    print("\n=== PREFIX DIFF (identical tokens, rows 0..62): first LARGE divergence ===")
    for r in range(nl):
        a64, _ = load_rows(t64[r]); a65, _ = load_rows(t65[r])
        worst = (0, 0.0, -1)
        for row in range(63):
            mism, mx = diff_rows(a64, a65, row)
            if mism > worst[0]:
                worst = (mism, mx, row)
        tag = ""
        if worst[0] > 0 and worst[1] > 0.25:
            tag = "  <-- LARGE"
        print(f"layer {r:>2}: worst-row {worst[2]:>2} mism {worst[0]:>5}/5120 maxd {worst[1]:.4f}{tag}")

if __name__ == "__main__":
    main()
