#!/usr/bin/env python3
"""probe_2k_decode.py - decode t/s probe for 2k-context arms.

The standard guard decode cell uses a 7,857-token probe, which cannot fit
a -c 2048 boot, so the 2k A/B (draft-device vs in-split) needs its own
short-prompt decode cell. Greedy, cache_prompt off; decode t/s and MTP
acceptance come from the server's own timings + speculative fields.

Usage: python3 probe_2k_decode.py --port 8081 [--reps 2] [--n-predict 256]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_prompts import UNIT_SENTENCE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--sentences", type=int, default=90, help="~10 tok each -> ~1k-token prompt")
    args = ap.parse_args()

    prompt = (f"Summarize the following text in one sentence. Text: "
              f"{UNIT_SENTENCE * args.sentences}\nQuestion: What was logged?")
    payload = json.dumps({
        "prompt": prompt, "n_predict": args.n_predict, "temperature": 0.0,
        "cache_prompt": False,
    }).encode()

    for rep in range(1, args.reps + 1):
        t0 = time.time()
        with urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{args.port}/completion", data=payload,
                headers={"Content-Type": "application/json"}), timeout=300) as r:
            body = json.loads(r.read())
        wall = time.time() - t0
        t = body.get("timings", {})
        n = t.get("predicted_n", 0)
        ms = t.get("predicted_ms", 0.0)
        tps = n / ms * 1000 if ms else 0.0
        spec = body.get("speculative", {}) or {}
        drafted = spec.get("n_drafted") or spec.get("draft_n")
        accepted = spec.get("n_accepted") or spec.get("draft_n_accepted")
        print(f"rep{rep}: decode {tps:.2f} t/s ({n} tok in {ms:.0f} ms, "
              f"wall {wall:.1f}s) drafted={drafted} accepted={accepted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
