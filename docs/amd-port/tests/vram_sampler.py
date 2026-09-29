#!/usr/bin/env python3
"""vram_sampler.py - background per-die VRAM/GPU-use sampler for the TP2 feasibility arms.

Writes CSV rows: ts_iso,elapsed_s,card,used_vram_mib,gpu_pct
Marks: lines beginning with '#' are phase markers written by the runner.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--cards", type=str, default="card0,card1,card2,card3")
    args = ap.parse_args()

    cards = args.cards.split(",")
    t0 = time.time()
    # append mode: the runner's #PHASE markers share this file via shell >>;
    # 'w' here would clobber them (writer offsets race)
    with open(args.out, "a") as f:
        if f.tell() == 0:
            f.write("ts_iso,elapsed_s,card,used_vram_mib,gpu_pct\n")
        f.flush()
        while True:
            try:
                out = subprocess.check_output(
                    ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                    text=True, stderr=subprocess.DEVNULL)
                data = json.loads(out)
            except Exception:
                data = {}
            ts = datetime.now().isoformat()
            el = round(time.time() - t0, 1)
            for c in cards:
                cd = data.get(c, {})
                used_b = cd.get("VRAM Total Used Memory (B)")
                gpu_pct = cd.get("GPU use (%)")
                try:
                    used_mib = round(int(used_b) / (1024 * 1024), 1)
                except (TypeError, ValueError):
                    used_mib = ""
                f.write(f"{ts},{el},{c},{used_mib},{gpu_pct or ''}\n")
            f.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
