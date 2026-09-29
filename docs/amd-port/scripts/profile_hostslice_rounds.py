#!/usr/bin/env python3
# HOST-SLICE A1: per-round decomposition of the W12 host-slice P0 timeline log.
# Parses [decode-timeline]/[launch-timeline]/[spec-timeline] lines with their
# wall timestamps and reconstructs, for every decode round:
#   verify issue (host) + meta host + mode, drain (device wait),
#   slice A = drain-end -> catchup-issue-end (the all-dies-idle window),
#   catchup issue + meta host, accept+bookkeeping (catchup-end -> draft start),
#   draft section, post-draft gap -> next verify issue start.
import re, sys, statistics as st

ts_re   = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.(\d+) ')
dec_re  = re.compile(r'\[decode-timeline\] n_tokens = (\d+), reused = (\d+), build = ([\d.]+) ms, inputs = ([\d.]+) ms, issue = ([\d.]+) ms')
meta_re = re.compile(r'\[launch-timeline\] meta nodes = (\d+), subs = (\d+), replays = (\d+), bounds = (\d+), comm = (\d+), fb = (\d+), host = ([\d.]+) ms, ar = ([\d.]+) ms')
drain_re = re.compile(r'\[decode-timeline\] drain = ([\d.]+) ms')
proc_re = re.compile(r'\[spec-timeline\] process: n_tokens = \d+, heads = \d+, decode_issue = ([\d.]+) ms')
draft_re = re.compile(r'\[spec-timeline\] draft: steps = (\d+), decode_issue = ([\d.]+) ms/step, sample\+batch = ([\d.]+) ms/step, total = ([\d.]+) ms')
catch_re = re.compile(r'\[spec-timeline\] catchup: rows = (\d+), decode_issue = ([\d.]+) ms')
cuda_re = re.compile(r'\[launch-timeline\] cuda dev = \d+, nodes = \d+, mode = (\w+), host = ([\d.]+) ms')

def t_us(line):
    m = ts_re.match(line)
    if not m: return None
    mn, s, ms, us = map(int, m.groups())
    return ((mn*60 + s)*1000 + ms)*1000 + us

events = []  # (t, kind, data, line)
with open(sys.argv[1], errors='replace') as f:
    for line in f:
        t = t_us(line)
        if t is None: continue
        if '[decode-timeline] n_tokens' in line:
            m = dec_re.search(line)
            events.append((t, 'dec', dict(n=int(m.group(1)), build=float(m.group(3)),
                inputs=float(m.group(4)), issue=float(m.group(5)), reused=int(m.group(2))), line))
        elif '[launch-timeline] meta' in line:
            m = meta_re.search(line)
            events.append((t, 'meta', dict(nodes=int(m.group(1)), subs=int(m.group(2)),
                host=float(m.group(7)), ar=float(m.group(8))), line))
        elif '[decode-timeline] drain' in line:
            m = drain_re.search(line)
            events.append((t, 'drain', dict(ms=float(m.group(1))), line))
        elif '[spec-timeline] process' in line:
            m = proc_re.search(line)
            events.append((t, 'process', dict(ms=float(m.group(1))), line))
        elif '[spec-timeline] draft' in line:
            m = draft_re.search(line)
            events.append((t, 'draftsec', dict(steps=int(m.group(1)), dec=float(m.group(2)),
                samp=float(m.group(3)), total=float(m.group(4))), line))
        elif '[spec-timeline] catchup' in line:
            m = catch_re.search(line)
            events.append((t, 'catchupsec', dict(rows=int(m.group(1)), ms=float(m.group(2))), line))
        elif '[launch-timeline] cuda' in line:
            m = cuda_re.search(line)
            events.append((t, 'cuda', dict(mode=m.group(1), host=float(m.group(2))), line))

# a round = a big meta (nodes > 1000) verify pass; the dec line right after it
# must be n_tokens = 4 (prefill ubatches are the same graph but dec n_tokens=512)
rounds = []
cur = None
pending_big = None
for t, k, d, line in events:
    if k == 'meta' and d['nodes'] > 1000:
        pending_big = (t, d)
        continue
    if pending_big is not None:
        # the next dec line decides: n=4 -> verify round, else prefill tail
        if k == 'dec':
            tb, db = pending_big
            pending_big = None
            if d['n'] == 4:
                if cur: rounds.append(cur)
                cur = dict(t_verify_issue_end=tb, verify_issue=d, verify_meta=db,
                           capture=False, capture_host=0.0, drain_end=None, drain_ms=None,
                           catchup_issue=None, catchup_meta=None, process_ms=None,
                           draft=None, t_draft_end=None)
            continue
        if k in ('drain', 'process', 'draftsec', 'catchupsec'):
            pending_big = None
            if cur is not None and k == 'drain' and cur['drain_ms'] is None and d['ms'] > 1.0:
                cur['drain_ms'] = d['ms']; cur['drain_end'] = t
            elif cur is not None and k == 'process' and cur['process_ms'] is None:
                cur['process_ms'] = d['ms']
            continue
    if cur is None: continue
    if k == 'dec' and cur['verify_issue'] is None and d['n'] in (2, 4):
        cur['verify_issue'] = d
    if k == 'cuda' and cur['verify_issue'] is None:
        if d['mode'] == 'capture':
            cur['capture'] = True
            cur['capture_host'] += d['host']
    if k == 'drain' and cur['drain_ms'] is None and d['ms'] > 1.0:
        cur['drain_ms'] = d['ms']
        cur['drain_end'] = t
    if k == 'dec' and cur['drain_end'] is not None and cur['catchup_issue'] is None and d['n'] == 4:
        cur['catchup_issue'] = d
        cur['t_catchup_issue_end'] = t
        cur['in_catchup'] = True
        continue
    if k == 'meta' and cur.get('in_catchup') and cur['catchup_meta'] is None and d['nodes'] < 1000:
        cur['catchup_meta'] = d
    if k == 'process' and cur['process_ms'] is None:
        cur['process_ms'] = d['ms']
    if k == 'draftsec':
        cur['draft'] = d
        cur['t_draft_end'] = t
        rounds.append(cur)
        cur = None

if cur: rounds.append(cur)

# attach the NEXT round's verify issue start = t_verify_issue_end - issue
for r in rounds:
    if r['verify_issue']:
        r['t_verify_issue_start'] = r['t_verify_issue_end'] - r['verify_issue']['issue']*1000
for i, r in enumerate(rounds):
    if i+1 < len(rounds) and r['t_draft_end'] and rounds[i+1].get('t_verify_issue_start'):
        r['gap_draft_to_next_issue'] = (rounds[i+1]['t_verify_issue_start'] - r['t_draft_end'])/1000.0
    if r['drain_end'] and r.get('t_catchup_issue_end'):
        r['sliceA_ms'] = (r['t_catchup_issue_end'] - r['drain_end'])/1000.0
    if r['drain_end'] and r.get('catchup_issue'):
        # host-only part inside catchup issue before its graph_compute: issue - meta host - build - inputs
        ci, cm = r['catchup_issue'], r.get('catchup_meta')
        if cm:
            r['catchup_sched_resid_ms'] = ci['issue'] - ci['build'] - ci['inputs'] - cm['host']
    if r['verify_meta']['host'] > 100:
        r['capture'] = True

def row(name, xs, unit='ms'):
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return f"  {name:<46} n=0"
    return (f"  {name:<46} n={len(xs):<4} med {st.median(xs):8.3f} {unit:<3} "
            f"min {min(xs):8.3f}  p90 {pct(xs,0.9):8.3f}  max {max(xs):8.3f}")

def pct(xs, q):
    xs = sorted(xs)
    i = min(len(xs)-1, max(0, int(q*(len(xs)-1)+0.5)))
    return xs[i]

print(f"rounds parsed: {len(rounds)}")
cap = sum(1 for r in rounds if r['capture'])
print(f"capture-mode verify passes: {cap} / {len(rounds)}")
print("")
print("== per-round host decomposition ==")
print(row("verify issue (whole llama_decode)", [r['verify_issue']['issue'] for r in rounds]))
print(row("verify meta host (graph_compute)", [r['verify_meta']['host'] for r in rounds]))
print(row("verify meta ar (RCCL enqueues)", [r['verify_meta']['ar'] for r in rounds]))
print(row("verify capture-mode die-capture host", [r['capture_host'] for r in rounds if r['capture']]))
print(row("verify drain (device wait, from issue end)", [r['drain_ms'] for r in rounds]))
print(row("SLICE A: drain-end -> catchup-issue-end", [r.get('sliceA_ms') for r in rounds]))
print(row("catchup issue (whole llama_decode)", [r['catchup_issue']['issue'] for r in rounds if r.get('catchup_issue')]))
print(row("catchup meta host (graph_compute)", [r['catchup_meta']['host'] for r in rounds if r.get('catchup_meta')]))
print(row("catchup meta ar", [r['catchup_meta']['ar'] for r in rounds if r.get('catchup_meta')]))
print(row("catchup sched residual (issue-meta-b-i)", [r.get('catchup_sched_resid_ms') for r in rounds]))
print(row("process decode_issue (drain incl)", [r['process_ms'] for r in rounds]))
print(row("draft section total", [r['draft']['total'] for r in rounds if r['draft']]))
print(row("draft decode_issue/step", [r['draft']['dec'] for r in rounds if r['draft']]))
print(row("draft sample+batch/step", [r['draft']['samp'] for r in rounds if r['draft']]))
print(row("gap: draft-end -> next verify issue start", [r.get('gap_draft_to_next_issue') for r in rounds]))
print("")
# round period
for a, b in zip(rounds, rounds[1:]):
    pass
periods = []
for a, b in zip(rounds, rounds[1:]):
    if a.get('t_verify_issue_start') and b.get('t_verify_issue_start'):
        periods.append((b['t_verify_issue_start'] - a['t_verify_issue_start'])/1000.0)
print(row("round period (verify issue start -> next)", periods))
# replay-only period
per_rep = [p for p, r in zip(periods, rounds[1:]) if not r['capture']]
print(row("round period (replay rounds only)", per_rep))
print("")
tot_slice = [r['sliceA_ms'] for r in rounds if r.get('sliceA_ms') is not None]
gaps = [g for g in [r.get('gap_draft_to_next_issue') for r in rounds] if g is not None]
print(f"SUM: median slice A {st.median(tot_slice):.3f} ms; median gap draft->next "
      f"{(st.median(gaps) if gaps else float('nan')):.3f} ms")
