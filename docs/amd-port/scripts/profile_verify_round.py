#!/usr/bin/env python3
# Offline profiler for the verify-round host slice from -lv 4 timeline logs.
# Usage: python3 profile_verify_round.py <log> [<log> ...]
import re, sys, statistics as st

ts_re = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.(\d+) ')
dec_re = re.compile(r'\[decode-timeline\] n_tokens = (\d+), reused = (\d+), build = ([\d.]+) ms, inputs = ([\d.]+) ms, issue = ([\d.]+) ms')
out_re = re.compile(r'\[decode-timeline\] outputs = ([\d.]+) ms \(n_outputs = (\d+), backend sampled = (\d+)\)')
drain_re = re.compile(r'\[decode-timeline\] drain = ([\d.]+) ms')
proc_re = re.compile(r'\[spec-timeline\] process: n_tokens = (\d+), heads = (\d+), decode_issue = ([\d.]+) ms')
draft_re = re.compile(r'\[spec-timeline\] draft: steps = (\d+), decode_issue = ([\d.]+) ms/step, sample\+batch = ([\d.]+) ms/step, total = ([\d.]+) ms')

def ts_us(line):
    m = ts_re.match(line)
    if not m:
        return None
    mn, s, ms, us = map(int, m.groups())
    return ((mn * 60 + s) * 1000 + ms) * 1000 + us

def load(path):
    evs = []
    with open(path, errors='replace') as f:
        for line in f:
            t = ts_us(line)
            if t is None:
                continue
            m = dec_re.search(line)
            if m:
                n, reused, build, inputs, issue = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
                evs.append((t, 'dec', (n, reused, build, inputs, issue)))
                continue
            m = out_re.search(line)
            if m:
                evs.append((t, 'out', (float(m.group(1)), int(m.group(2)))))
                continue
            m = drain_re.search(line)
            if m:
                evs.append((t, 'drain', (float(m.group(1)),)))
                continue
            m = proc_re.search(line)
            if m:
                evs.append((t, 'proc', (int(m.group(1)), float(m.group(3)))))
                continue
            m = draft_re.search(line)
            if m:
                evs.append((t, 'draft', (int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))))
    evs.sort(key=lambda e: e[0])
    return evs

def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    i = min(len(xs) - 1, max(0, int(q * (len(xs) - 1) + 0.5)))
    return xs[i]

def stats_row(name, xs, unit='ms'):
    if not xs:
        return f"  {name:<34} n=0"
    return (f"  {name:<34} n={len(xs):<4} mean {st.mean(xs):8.3f}  med {st.median(xs):8.3f}  "
            f"p10 {pct(xs,0.10):8.3f}  p90 {pct(xs,0.90):8.3f}")

def analyze(path):
    evs = load(path)
    print(f"== {path}: {len(evs)} timeline events ==")

    # segment rounds: a round begins at a verify decode (n_tokens=4 whose next
    # 'out' event has n_outputs=4) and ends at the next 'draft' line
    rounds = []
    i = 0
    n = len(evs)
    while i < n:
        ev = evs[i]
        if ev[1] == 'dec' and ev[2][0] == 4:
            # find the following 'out' before the next 'dec'
            j = i + 1
            nxt = None
            while j < n and evs[j][1] != 'dec':
                if evs[j][1] == 'out':
                    nxt = evs[j]
                    break
                j += 1
            if nxt and nxt[2][1] > 0:
                # this is the verify decode; collect until next 'draft' line
                k = i + 1
                R = {'verify': (evs[i][0], evs[i][2]), 'out4': None, 'drain_v': None,
                     'catchup': None, 'proc': None, 'drains_mid': [], 'steps': [],
                     'draft_line': None}
                while k < n:
                    t2, kind, d = evs[k]
                    if kind == 'draft':
                        R['draft_line'] = (t2, d)
                        break
                    if kind == 'dec':
                        if d[0] == 4 and R['verify'] is not None and R['catchup'] is None:
                            R['catchup'] = (t2, d)
                        elif R['verify'] is None:
                            R['verify'] = (t2, d)
                    elif kind == 'out':
                        if d[1] > 0 and R['out4'] is None:
                            R['out4'] = (t2, d)
                    elif kind == 'drain':
                        R['drains_mid'].append((t2, d[0]))
                    elif kind == 'proc':
                        R['proc'] = (t2, d)
                    k += 1
                if R['draft_line']:
                    rounds.append((i, R))
        i += 1

    V = lambda r: r[1]
    verify_issue, catchup_issue, proc_issue = [], [], []
    build_v, inputs_v = [], []
    out4_ms, drain_v_ms = [], []
    g1, g2, g3 = [], [], []     # draft_line->verify entry ; verify drain->catchup entry ; proc line->step1 entry
    step_issue, step_samp, draft_total = [], [], []
    wall = []
    drains_mid_ms = []

    rounds_v = [V(r) for r in rounds]
    for idx, (start_i, R) in enumerate(rounds):
        if R['verify']:
            verify_issue.append(R['verify'][1][4])
            build_v.append(R['verify'][1][2])
            inputs_v.append(R['verify'][1][3])
        if R['out4']:
            out4_ms.append(R['out4'][1][0])
        # first drain after out4
        if R['out4'] and R['drains_mid']:
            t_out = R['out4'][0]
            after = [x for x in R['drains_mid'] if x[0] >= t_out]
            if after:
                drain_v_ms.append(after[0][1])
        if R['catchup']:
            catchup_issue.append(R['catchup'][1][4])
        if R['proc']:
            proc_issue.append(R['proc'][1][1])
        if R['draft_line']:
            step_issue.append(R['draft_line'][1][1])
            step_samp.append(R['draft_line'][1][2])
            draft_total.append(R['draft_line'][1][3])
        # gaps
        if R['draft_line'] and idx + 1 < len(rounds_v):
            nxt = rounds_v[idx + 1]
            if nxt['verify']:
                entry = nxt['verify'][0] - (nxt['verify'][1][2] + nxt['verify'][1][3] + nxt['verify'][1][4]) * 1000
                g1.append((entry - R['draft_line'][0]) / 1000)
            wall.append((rounds[idx + 1][1]['draft_line'][0] if rounds_v[idx + 1]['draft_line'] else None, 0))
        if R['out4'] and R['drains_mid'] and R['catchup']:
            t_out = R['out4'][0]
            after = [x for x in R['drains_mid'] if x[0] >= t_out]
            if after:
                entry = R['catchup'][0] - (R['catchup'][1][2] + R['catchup'][1][3] + R['catchup'][1][4]) * 1000
                g2.append((entry - (after[0][0] + after[0][1] * 1000)) / 1000)
        if R['proc'] and R['draft_line']:
            pass  # draft steps belong to the NEXT segment in our ordering; handled below

    # step1 entry needs the draft steps: rescan for dec n_tokens=1 right after proc
    for idx, (start_i, R) in enumerate(rounds):
        if not R['proc']:
            continue
        t_proc = R['proc'][0]
        # find first dec n_tokens=1 after proc line
        j = start_i + 1
        found = None
        while j < n and evs[j][0] < (R['draft_line'][0] if R['draft_line'] else evs[n-1][0] + 1):
            if evs[j][1] == 'dec' and evs[j][2][0] == 1:
                found = evs[j]
                break
            j += 1
        if found:
            entry = found[0] - (found[2][2] + found[2][3] + found[2][4]) * 1000
            g3.append((entry - t_proc) / 1000)

    print("device blocking (inside graph_compute):")
    print(stats_row("verify issue (ctx_tgt, 4 tok)", verify_issue))
    print(stats_row("catchup issue (ctx_dft, 4 tok)", catchup_issue))
    print(stats_row("process line (proc total)", proc_issue))
    print(stats_row("draft step issue/step", step_issue))
    print("host gaps:")
    print(stats_row("G1: draft line -> verify entry", g1))
    print(stats_row("G2: verify drain -> catchup entry", g2))
    print(stats_row("G3: proc line -> step1 entry", g3))
    print("other:")
    print(stats_row("verify build", build_v))
    print(stats_row("verify inputs", inputs_v))
    print(stats_row("outputs extract (4 rows)", out4_ms))
    print(stats_row("verify drain", drain_v_ms))
    print(stats_row("draft sample+batch/step", step_samp))
    print(stats_row("draft total", draft_total))
    if wall:
        walls = [b for a, b in wall]
    # wall from draft line to next draft line using raw events
    dl = [e[0] for e in evs if e[1] == 'draft']
    dw = [(b - a) / 1000 for a, b in zip(dl, dl[1:])]
    print(stats_row("round wall (draft line deltas)", dw))
    if verify_issue and catchup_issue and dw:
        med = st.median(dw)
        print(f"\n  median budget: wall {med:.1f} = verify {st.median(verify_issue):.1f}"
              f" + catchup {st.median(catchup_issue):.1f} + draft {st.median(draft_total):.1f}"
              f" + G1 {st.median(g1) if g1 else float('nan'):.1f} + G2 {st.median(g2) if g2 else float('nan'):.2f}"
              f" + G3 {st.median(g3) if g3 else float('nan'):.2f}"
              f" + outputs/drain {st.median(out4_ms) + st.median(drain_v_ms):.2f}"
              f" -> residual {med - st.median(verify_issue) - st.median(catchup_issue) - st.median(draft_total) - (st.median(g1) if g1 else 0) - (st.median(g2) if g2 else 0) - (st.median(g3) if g3 else 0) - (st.median(out4_ms) + st.median(drain_v_ms)):.2f}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyze(p)
        print()
