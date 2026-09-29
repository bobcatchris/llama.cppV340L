#!/usr/bin/env python3
# Offline profiler for LLAMA_LAUNCH_TIMELINE=1 logs (run with -lv 4).
# Consumes the [launch-timeline] lines (meta per ubatch, cuda per die replay)
# plus the regular [decode-timeline] / [spec-timeline] lines and prints the
# launch/replay cost table: per-ubatch meta decomposition, per-replay cuda
# bookkeeping (check/launch host cost; the min launch is the enqueue floor),
# and per-round sweep counts (backend drains, split input copies).
# Usage: python3 profile_launch_timeline.py <log> [<log> ...]
import re, sys, statistics as st

ts_re  = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.(\d+) ')
meta_re = re.compile(r'\[launch-timeline\] meta nodes = (\d+), subs = (\d+), replays = (\d+), bounds = (\d+), comm = (\d+), fb = (\d+), host = ([\d.]+) ms, ar = ([\d.]+) ms')
cuda_re = re.compile(r'\[launch-timeline\] cuda dev = (\d+), nodes = (\d+), mode = (\w+), host = ([\d.]+) ms, check = ([\d.]+) ms, launch = ([\d.]+) ms')
sched_re = re.compile(r'\[launch-timeline\] sched splits = (\d+), host = ([\d.]+) ms, tot \{cpy = (\d+), csync = (\d+), syncs = (\d+), sync_us = (\d+)\}')
dec_re  = re.compile(r'\[decode-timeline\] n_tokens = (\d+), reused = (\d+), build = ([\d.]+) ms, inputs = ([\d.]+) ms, issue = ([\d.]+) ms')
drain_re = re.compile(r'\[decode-timeline\] drain = ([\d.]+) ms')

def ts_us(line):
    m = ts_re.match(line)
    if not m:
        return None
    mn, s, ms, us = map(int, m.groups())
    return ((mn * 60 + s) * 1000 + ms) * 1000 + us

def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    i = min(len(xs) - 1, max(0, int(q * (len(xs) - 1) + 0.5)))
    return xs[i]

def row(name, xs, unit='ms'):
    if not xs:
        return f"  {name:<44} n=0"
    return (f"  {name:<44} n={len(xs):<5} med {st.median(xs):9.3f} {unit:<3} "
            f"min {min(xs):9.3f}  p90 {pct(xs, 0.9):9.3f}  max {max(xs):9.3f}")

def load(path):
    evs = []
    with open(path, errors='replace') as f:
        for line in f:
            t = ts_us(line)
            if t is None:
                continue
            if '[launch-timeline] meta' in line:
                m = meta_re.search(line)
                evs.append((t, 'meta', dict(nodes=int(m.group(1)), subs=int(m.group(2)),
                    replays=int(m.group(3)), bounds=int(m.group(4)), comm=int(m.group(5)),
                    fb=int(m.group(6)), host=float(m.group(7)), ar=float(m.group(8)))))
                continue
            if '[launch-timeline] cuda' in line:
                m = cuda_re.search(line)
                evs.append((t, 'cuda', dict(dev=int(m.group(1)), nodes=int(m.group(2)),
                    mode=m.group(3), host=float(m.group(4)), check=float(m.group(5)),
                    launch=float(m.group(6)))))
                continue
            if '[launch-timeline] sched' in line:
                m = sched_re.search(line)
                evs.append((t, 'sched', dict(splits=int(m.group(1)), host=float(m.group(2)),
                    cpy=int(m.group(3)), csync=int(m.group(4)), syncs=int(m.group(5)),
                    sync_us=int(m.group(6)))))
                continue
            m = dec_re.search(line)
            if m:
                evs.append((t, 'dec', dict(n=int(m.group(1)), issue=float(m.group(5)),
                    build=float(m.group(3)), inputs=float(m.group(4)))))
                continue
            m = drain_re.search(line)
            if m:
                evs.append((t, 'drain', dict(ms=float(m.group(1)))))
                continue
    evs.sort(key=lambda e: e[0])
    return evs

def analyze_decode_only(path, evs):
    # fallback for logs captured before LLAMA_LAUNCH_TIMELINE=1 existed: the
    # decode-timeline lines still give the per-ubatch issue decomposition and
    # the drain sweep counts, just not the launch/replay split
    decs = [d for _, k, d in evs if k == 'dec']
    drains = [d['ms'] for _, k, d in evs if k == 'drain']
    if not decs:
        print("  no [decode-timeline] lines either (need -lv 4 + LLAMA_DECODE_TIMELINE=1)")
        return
    print("  decode-only table (no [launch-timeline] meta lines; boot with")
    print("  LLAMA_LAUNCH_TIMELINE=1 for the launch/replay split):")
    for kind, sel in (("draft (n=2)",          [d for d in decs if d['n'] == 2]),
                      ("catchup (n=4, issue<=40)", [d for d in decs if d['n'] == 4 and d['issue'] <= 40]),
                      ("verify (n=4, issue>40)",   [d for d in decs if d['n'] == 4 and d['issue'] > 40])):
        if not sel:
            continue
        print(f"  {kind}: n={len(sel)}")
        print(row("issue (whole ubatch)", [d['issue'] for d in sel]))
        print(row("build (graph reuse)", [d['build'] for d in sel]))
        print(row("inputs (host setup)", [d['inputs'] for d in sel]))
    if drains:
        print(row("drain (per call)", drains))
        print(row("drain (sum per log, ms)", [sum(drains)]))

def analyze(path):
    evs = load(path)
    print(f"== {path}: {len(evs)} timeline events ==")
    if not any(k == 'meta' for _, k, _ in evs):
        analyze_decode_only(path, evs)
        return

    # classify each meta line by the NEXT decode-timeline line (issued inside it)
    ubatches = []
    for i, (t, k, d) in enumerate(evs):
        if k != 'meta':
            continue
        kind = '?'
        for t2, k2, d2 in evs[i+1:]:
            if k2 == 'dec':
                if d2['n'] == 4 and d2['issue'] > 40:
                    kind = 'verify'
                elif d2['n'] == 4:
                    kind = 'catchup'
                elif d2['n'] == 2:
                    kind = 'draft'
                else:
                    kind = f"n{d2['n']}"
                break
            if k2 == 'meta':
                break
        ubatches.append((t, kind, d))

    for kind in ('verify', 'catchup', 'draft'):
        sel = [d for _, kd, d in ubatches if kd == kind]
        if not sel:
            continue
        print(f"  {kind}: n={len(sel)}")
        print(row("meta host (whole graph_compute)", [d['host'] for d in sel]))
        print(row("meta allreduce host (comm enqueues)", [d['ar'] for d in sel]))
        print(row("meta replays/ubatch", [float(d['replays']) for d in sel], ''))
        print(row("meta bounds/ubatch", [float(d['bounds']) for d in sel], ''))
        print(row("meta comm/ubatch (RCCL served)", [float(d['comm']) for d in sel], ''))
        print(row("meta fb/ubatch (butterfly fallback)", [float(d['fb']) for d in sel], ''))
        print(row("graph nodes", [float(d['nodes']) for d in sel], ''))

    # cuda per-replay bookkeeping
    cudas = [d for _, k, d in evs if k == 'cuda']
    for mode in ('replay', 'capture', 'direct'):
        sel = [d for d in cudas if d['mode'] == mode]
        if not sel:
            print(f"  cuda mode={mode}: none")
            continue
        print(f"  cuda mode={mode}: n={len(sel)}")
        print(row("cuda host per call", [d['host'] for d in sel]))
        print(row("cuda check (compat+props scan)", [d['check'] for d in sel]))
        print(row("cuda launch (graph launch, incl. backpressure)", [d['launch'] for d in sel]))
        print(row("cuda nodes", [float(d['nodes']) for d in sel], ''))

    # sweeps per round: deltas of the cumulative sched counters across rounds
    # a round starts at a verify meta line
    rounds = []
    cur = None
    for t, kind, d in ubatches:
        if kind == 'verify':
            if cur:
                rounds.append(cur)
            cur = dict(sync0=None, cpy0=None, csync0=None, verify=0, host=0.0)
        if cur is None:
            continue
        cur['verify'] += kind == 'verify'
    scheds = [(t, d) for t, k, d in evs if k == 'sched']
    drains = [(t, d['ms']) for t, k, d in evs if k == 'drain']
    if rounds and scheds:
        # use cumulative counters from sched lines inside each round window
        windows = []
        vt = [t for t, kind, _ in ubatches if kind == 'verify']
        for a, b in zip(vt, vt[1:]):
            ss = [d for t, d in scheds if a <= t < b]
            dd = [ms for t, ms in drains if a <= t < b]
            if len(ss) >= 2:
                windows.append(dict(
                    cpy   = ss[-1]['cpy']   - ss[0]['cpy'],
                    csync = ss[-1]['csync'] - ss[0]['csync'],
                    syncs = ss[-1]['syncs'] - ss[0]['syncs'],
                    sync_us = ss[-1]['sync_us'] - ss[0]['sync_us'],
                    ndrain = len(dd), drain_ms = sum(dd)))
        if windows:
            print(f"  per round (n={len(windows)}, verify->verify windows):")
            print(row("sched input copies (cpy delta)", [float(w['cpy']) for w in windows], ''))
            print(row("sched input drains (csync delta)", [float(w['csync']) for w in windows], ''))
            print(row("backend syncs (all, delta)", [float(w['syncs']) for w in windows], ''))
            print(row("backend sync blocked ms (delta)", [w['sync_us']/1e3 for w in windows]))
            print(row("drain lines/round", [float(w['ndrain']) for w in windows], ''))
            print(row("drain ms/round (sum)", [w['drain_ms'] for w in windows]))

    # verify decomposition: issue vs meta host
    verify_sel = [(t, d) for t, kind, d in ubatches if kind == 'verify']
    issues = []
    for t, d in verify_sel:
        for t2, k2, d2 in evs:
            if t2 >= t and k2 == 'dec':
                if d2['n'] == 4:
                    issues.append(d2['issue'])
                break
    if issues:
        print(f"  verify decode issue: n={len(issues)} med {st.median(issues):.3f} ms")
        hosts = [d['host'] for _, d in verify_sel]
        n = min(len(issues), len(hosts))
        share = [h/i*100 for h, i in zip(hosts[:n], issues[:n]) if i > 0]
        print(row("meta host share of verify issue (%)", share, '%'))

for path in sys.argv[1:]:
    analyze(path)
