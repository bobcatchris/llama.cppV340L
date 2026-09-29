#!/usr/bin/env python3
"""vram_reduce.py - reduce a vram_sampler.py CSV into per-phase per-die VRAM stats.

Usage: vram_reduce.py <vram.log> [more logs...]

Prints for each phase window and card: mean used MiB and peak used MiB.
Phases are delimited by '#PHASE <name> <unix_ts>' marker lines.
"""

import sys
from collections import defaultdict

MARKS = []
ROWS = []  # (unix_ts, card, used_mib)

for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#PHASE"):
                parts = line.split()
                MARKS.append((parts[1], float(parts[2]), path))
            elif line.startswith("ts_iso"):
                continue
            else:
                parts = line.split(",")
                try:
                    ts = parts[0]
                    el = float(parts[1])
                    card = parts[2]
                    used = float(parts[3])
                    ROWS.append((el, card, used, ts))
                except (ValueError, IndexError):
                    continue

if not MARKS or not ROWS:
    print("no data")
    sys.exit(0)

# order marks by elapsed time using their wall ts relative to first row
# marker lines carry unix epoch; rows carry elapsed seconds. Convert marks
# to elapsed by anchoring on the first data row's wall ts minus its elapsed.
t0_wall = None
for el, card, used, ts in ROWS:
    try:
        from datetime import datetime
        t0_wall = datetime.fromisoformat(ts).timestamp() - el
        break
    except Exception:
        continue

phases = []
for name, wall_ts, path in MARKS:
    phases.append((name, wall_ts - t0_wall))
phases.sort(key=lambda x: x[1])

cards = sorted({r[1] for r in ROWS})
print(f"{'phase':<22}" + "".join(f"{c + ':mean':>12}{c + ':peak':>12}" for c in cards))

bounds = []
for i, (name, el) in enumerate(phases):
    end = phases[i + 1][1] if i + 1 < len(phases) else float("inf")
    bounds.append((name, el, end))

for name, start, end in bounds:
    stats = defaultdict(list)
    for el, card, used, _ in ROWS:
        if start <= el < end:
            stats[card].append(used)
    row = f"{name:<22}"
    for c in cards:
        vals = stats.get(c, [])
        if vals:
            row += f"{sum(vals)/len(vals):>12.1f}{max(vals):>12.1f}"
        else:
            row += f"{'-':>12}{'-':>12}"
    print(row)
