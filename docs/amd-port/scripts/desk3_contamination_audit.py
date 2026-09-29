#!/usr/bin/env python3
"""desk3_contamination_audit.py - scan served arm logs for foreign requests.

Each guard battery is strictly serial: 7 tasks
  [prefill_2k, decode_10k, det1, det2, needle1, needle2, needle3]
The diag boot served exactly 1 task. Contamination signatures:
  (a) task count != expected
  (b) temporal overlap: a task launched before the previous one released
  (c) duplicate/unknown prompt sizes
  (d) KV-retry / retry lines
"""
import re, sys, glob

TS = r"(\d+)\.(\d+)\.(\d+)\.(\d+)\s+[IWE]\s+\S+\s+(\w+):\s+id\s+(\d+) \| task (-?\d+) \| (.*)"

def secs(m_, s, ms, us):
    # log timestamps are M.SS.mmm.uuu since process start
    return int(m_) * 60 + int(s) + int(ms) / 1000.0 + int(us) / 1e6

def audit(path):
    events = []  # (t, kind, slot, task, extra)
    for line in open(path, errors="replace"):
        m = re.search(TS, line)
        if m:
            m_, s, ms, us, comp, slot, task, rest = m.groups()
            t = secs(m_, s, ms, us)
            events.append((t, comp, int(slot), int(task), rest.strip()))
            continue
        if re.search(r"retry|Retry|RETRY", line):
            events.append((None, "RETRY_LINE", -1, -1, line.strip()[:120]))
    launches, releases, evals = {}, {}, []
    for t, comp, slot, task, rest in events:
        if comp == "launch_slot_":
            launches[(slot, task)] = t
        elif comp == "release":
            releases[(slot, task)] = t
        elif comp == "print_timing" and rest.startswith("prompt eval time"):
            mt = re.search(r"/\s*(\d+) tokens", rest)
            evals.append((t, slot, task, int(mt.group(1)) if mt else -1))
    # serial-overlap check over wall-clock ordered launches
    ordered = sorted(launches.items(), key=lambda kv: kv[1])
    overlaps = []
    for i in range(len(ordered) - 1):
        (s0, t0), l0 = ordered[i]
        (s1, t1), l1 = ordered[i + 1]
        r0 = releases.get((s0, t0))
        if r0 is not None and l1 < r0:
            overlaps.append((t0, t1, round(l1 - l0, 2)))
    retries = [e for e in events if e[1] == "RETRY_LINE"]
    return dict(n_tasks=len(launches), prompts=[e[3] for e in evals], overlaps=overlaps, retries=len(retries))

def main():
    paths = sorted(glob.glob("docs/amd-port/results/server_tp3_200k_20260921_1[9]*.log")) + \
            sorted(glob.glob("docs/amd-port/results/server_tp3_200k_20260921_2*.log")) + \
            sorted(glob.glob("docs/amd-port/results/desk3_diag_q81_*.log"))
    print(f"{'log':<44} {'tasks':<6} {'overlaps':<20} retries  prompt tokens per task")
    for p in paths:
        if "213215" in p or "212711" in p or "211103" in p:
            continue  # crash/void rows, audited separately
        r = audit(p)
        flag = ""
        if r["retries"]: flag += " RETRY!"
        if r["overlaps"]: flag += " OVERLAP!"
        print(f"{p.split('/')[-1]:<44} {r['n_tasks']:<6} {str(r['overlaps'])[:20]:<20} {r['retries']:<7} {r['prompts']}{flag}")

if __name__ == "__main__":
    main()
