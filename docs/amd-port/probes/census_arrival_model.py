#!/usr/bin/env python3
# SPLIT-BALANCE DESK A1 instrument: per-die compute-arrival model from the
# TP4 kernel census (zero-GPU; reads the coordinator trace read-only).
#
# Tests: (1) reproduce W8 per-die NCCL medians (validates agent->die map);
# (2) per-boundary arrival spread (Start_Timestamp of the i-th NCCL kernel
# per die) -> in-kernel peer wait per die; (3) per-die serve-state compute
# rate: identical-kernel duration medians per die (bytes are EQUAL per P0,
# so duration differences are rate differences); (4) correlation of NCCL
# duration with own arrival lag.
import sys, csv, statistics as st

TRACE = sys.argv[1] if len(sys.argv) > 1 else \
    '/media/chris/ssd128/llamacpp/llama.cpp/docs/amd-port/results/TP4_kernel_census_2026-09-23_kernel_trace.csv'
NDEV = 4
WINDOW_S = 5.0

rows = []  # (agent, name, start, end)
with open(TRACE, newline='') as f:
    rd = csv.reader(f)
    hdr = next(rd)
    ia = hdr.index('Agent_Id'); ik = hdr.index('Kernel_Name')
    isx = hdr.index('Start_Timestamp'); iex = hdr.index('End_Timestamp')
    kind = hdr.index('Kind')
    for row in rd:
        if row[kind] != 'KERNEL_DISPATCH':
            continue
        try:
            rows.append((int(row[ia]), row[ik], int(row[isx]), int(row[iex])))
        except ValueError:
            continue

t0 = min(r[2] for r in rows); t1 = max(r[3] for r in rows)
dec_lo = t1 - int(WINDOW_S * 1e9)
drows = [r for r in rows if r[2] >= dec_lo]
print(f'trace span {(t1-t0)/1e9:.2f} s, decode window last {WINDOW_S} s, dispatches in window {len(drows)}')

nccl = {d: [] for d in range(NDEV)}
comp = {d: [] for d in range(NDEV)}
for ag, name, s, e in drows:
    d = ag - 1
    if d < 0 or d >= NDEV:
        continue
    if 'nccl' in name.lower():
        nccl[d].append((s, e))
    else:
        comp[d].append((s, e, name))

print()
print('per-die NCCL (allreduce) kernel durations in window:')
med = {}
for d in range(NDEV):
    dur = [e - s for s, e in nccl[d]]
    med[d] = st.median(dur)
    print(f'  die {d}: n={len(dur):5d} median={med[d]:7.1f} us avg={st.mean(dur):7.1f} p90={sorted(dur)[int(len(dur)*0.9)]:7.1f}')
print(f'  W8 of record: die0=191.6 die1=210.8 die2=151.9 die3=128.8 (agent=die+1 by Location_Id)')

# boundary alignment: i-th NCCL per die in time order
nmin = min(len(nccl[d]) for d in range(NDEV))
starts = [[nccl[d][i][0] for i in range(nmin)] for d in range(NDEV)]
ends   = [[nccl[d][i][1] for i in range(nmin)] for d in range(NDEV)]
durs   = [[nccl[d][i][1] - nccl[d][i][0] for i in range(nmin)] for d in range(NDEV)]

# per-agent clock skew calibration: a collective completes near-simultaneously
# on all ranks, so per-die END offsets vs the cross-die median end are the
# per-agent clock offsets (durations stay in-die, immune; cross-die starts are not)
skew = {}
for d in range(NDEV):
    sk = [ends[d][i] - st.median([ends[x][i] for x in range(NDEV)]) for i in range(nmin)]
    skew[d] = st.median(sk)
print(f'\nper-agent clock skew vs cross-die median end (ns): ' +
      ', '.join(f'die{d}={skew[d]:+.0f}' for d in range(NDEV)))
cstarts = [[starts[d][i] - skew[d] for i in range(nmin)] for d in range(NDEV)]

cspreads = [max(cstarts[d][i] for d in range(NDEV)) - min(cstarts[d][i] for d in range(NDEV)) for i in range(nmin)]
print(f'corrected start-spread us: median={st.median(cspreads)/1000:.1f} p90={sorted(cspreads)[int(nmin*0.9)]/1000:.1f} max={max(cspreads)/1000:.1f}')

# arrival model on corrected starts: wait_d(i) = max_corr_start(i) - corr_start_d(i)
last_arr = {d: 0 for d in range(NDEV)}
waits = {d: [] for d in range(NDEV)}
corr_pairs = {d: [] for d in range(NDEV)}
for i in range(nmin):
    mx = max(cstarts[d][i] for d in range(NDEV))
    for d in range(NDEV):
        w = mx - cstarts[d][i]
        waits[d].append(w)
        if w < 5000:  # within 5 us of last -> this die arrived last
            last_arr[d] += 1
        corr_pairs[d].append((w, durs[d][i]))
print('who arrives last per boundary (corrected, wait < 5 us):')
for d in range(NDEV):
    print(f'  die {d}: {last_arr[d]} of {nmin} ({last_arr[d]/nmin*100:.0f}%)  median wait {st.median(waits[d])/1000:6.1f} us')
print('NCCL duration vs corrected arrival-wait (pearson r, slope; ideal slope 1.0 + ring):')
for d in range(NDEV):
    ws = [p[0] for p in corr_pairs[d]]; ds = [p[1] for p in corr_pairs[d]]
    mw, md = st.mean(ws), st.mean(ds)
    cov = sum((w-mw)*(x-md) for w, x in corr_pairs[d]) / len(ws)
    r_ = cov / (st.pstdev(ws)*st.pstdev(ds) + 1e-18)
    slope = cov / (st.pvariance(ws) + 1e-18)
    print(f'  die {d}: r={r_:+.2f} slope={slope:5.2f}  duration median {med[d]/1000:6.1f} vs wait median {st.median(waits[d])/1000:6.1f} -> ring est {med[d]/1000 - st.median(waits[d])/1000:6.1f} us')

raw_spreads = [max(starts[d][i] for d in range(NDEV)) - min(starts[d][i] for d in range(NDEV)) for i in range(nmin)]
print(f'RAW (uncorrected) start-spread us: median={st.median(raw_spreads)/1000:.1f} - clock-skew driven, see calibration above')

# stability: split the window in halves, busy ratio per die must hold in both
hmid = (dec_lo + t1) // 2
half = {0: {d: 0 for d in range(NDEV)}, 1: {d: 0 for d in range(NDEV)}}
for d in range(NDEV):
    for s, e, _ in comp[d]:
        half[0 if s < hmid else 1][d] += e - s
print('\nrate stability (non-NCCL busy ratio vs die3, first half / second half):')
for h in (0, 1):
    print(f'  half {h}: ' + ', '.join(f'die{d}={half[h][d]/half[h][3]:.4f}' for d in range(NDEV)))

# serve-state compute rate: identical-kernel duration medians per die
from collections import defaultdict
kdur = {d: defaultdict(list) for d in range(NDEV)}
for d in range(NDEV):
    for s, e, name in comp[d]:
        kdur[d][name].append(e - s)
names = sorted(set.intersection(*[set(kdur[d]) for d in range(NDEV)]),
               key=lambda n: -sum(sum(kdur[d][n]) for d in range(NDEV)))
print('\nidentical-kernel duration medians per die (top 12 classes by total device time, us):')
print(f'  {"kernel":46s} {"die0":>8s} {"die1":>8s} {"die2":>8s} {"die3":>8s}  ratio d1/d3  n/die')
for name in names[:12]:
    ms = [st.median(kdur[d][name])/1000 for d in range(NDEV)]
    tot = sum(sum(kdur[d][name]) for d in range(NDEV))/1e6
    print(f'  {name[:46]:46s} ' + ' '.join(f'{m:8.1f}' for m in ms) +
          f'  {ms[1]/ms[3]:8.3f}  {len(kdur[0][name])} ({tot:.1f} ms tot)')

# aggregate device-busy per die in window (non-NCCL)
print('\nnon-NCCL device-busy per die (sum of kernel durations, window):')
busy = {}
for d in range(NDEV):
    busy[d] = sum(e - s for s, e, _ in comp[d])
for d in range(NDEV):
    print(f'  die {d}: {busy[d]/1e6:8.1f} ms  ratio vs die3: {busy[d]/busy[3]:.4f}')

# per-boundary compute between consecutive NCCL boundaries per die:
# device-busy time in (start_i, start_{i+1}) - measures compute arrival spacing
print('\nper-boundary inter-NCCL device-busy per die (us):')
seg = {d: [] for d in range(NDEV)}
for d in range(NDEV):
    cs = sorted(comp[d])
    # two-pointer over boundary starts
    import bisect
    cst = [c[0] for c in cs]
    for i in range(nmin - 1):
        lo, hi = cstarts[d][i], cstarts[d][i+1]
        a = bisect.bisect_left(cst, lo); b = bisect.bisect_left(cst, hi)
        seg[d].append(sum(c[1]-c[0] for c in cs[a:b]))
meds = [st.median(seg[d]) for d in range(NDEV)]
for d in range(NDEV):
    print(f'  die {d}: median {meds[d]/1000:7.1f} us  avg {st.mean(seg[d])/1000:7.1f} us  ratio vs die3 {meds[d]/meds[3]:.3f}')

# A2: rate-weighted split that equalizes per-boundary compute time
# compute_d(proposal) = W * s_d * t_d ; equalize with s_d proportional to 1/t_d
W = sum(meds)                       # total device compute per boundary (today)
t = [meds[d] / meds[3] for d in range(NDEV)]   # per-byte time, die3 = 1.0
inv = [1.0 / x for x in t]
ssum = sum(inv)
shares = [x / ssum for x in inv]
ring = meds[3] * 0 + st.median([durs[3][i] for i in range(nmin)]) / 1000
print('\nA2 model (per-boundary, verify):')
print(f'  W (total compute) = {W/1000:.1f} us; today max = {max(meds)/1000:.1f} us (die 3); ring floor = med NCCL on last arriver = {ring:.1f} us')
print(f'  per-byte time t_d (die3=1): ' + ', '.join(f'die{d}={t[d]:.3f}' for d in range(NDEV)))
print(f'  balancing shares (% of sharded bytes): ' + ', '.join(f'die{d}={shares[d]*100:.2f}' for d in range(NDEV)))
ts_ratios = [shares[d] / shares[3] for d in range(NDEV)]
print(f'  --tensor-split ratios (die3=1): ' + ', '.join(f'{x:.3f}' for x in ts_ratios))
C = 4 * meds[3] / ssum  # equalized per-die compute: K*s_d*t_d with K = 4*meds[3]
print(f'  equalized per-die compute = {C/1000:.1f} us; wall/boundary today = {max(meds)/1000 + ring:.1f} us -> balanced = {C/1000 + ring:.1f} us (ring unchanged)')
sv = (max(meds)/1000 + ring) - (C/1000 + ring)
print(f'  verify saving/boundary = {sv:.1f} us; x128 = {sv*128/1000:.2f} ms/round; draft+catchup 8 boundaries ~{8*sv/1000:.2f} ms')
print(f'  stacked with ch4 ring (-11%, W9): ring 128.9 -> ~114.7; total saving ~{(max(meds)/1000 + 128.9) - (C/1000 + 114.7):.1f} us/boundary = ~{((max(meds)/1000 + 128.9) - (C/1000 + 114.7))*136/1000:.2f} ms/round')
