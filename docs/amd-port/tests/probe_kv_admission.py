#!/usr/bin/env python3
"""probe_kv_admission.py - Served validation for the E-054 server-exposures fix.

Fires two ~7,857-token prompts concurrently at one llama-server (10k-class
-c, kv unified) and checks the unified KV admission + FIFO fill behavior:

  fixed build (--kv-admission, default): one request defers with the log line
  "unified KV occupancy is too high, defer task", runs after the other
  finishes, both return HTTP 200, and the server log has zero
  "Context size has been exceeded" lines and zero 500s.

  control build (--no-kv-admission): the legacy failure signature is expected
  (context-exceeded retries / HTTP 500 / aborted slots).

Usage:
  python3 probe_kv_admission.py --port 8081 --log /path/to/server.log \
      [--control] [--n-predict 16]

The server must already be up (boot + lock handling stay with the run
scripts). Exit 0 = verdict PASS for the selected mode.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request

HERE = __import__("pathlib").Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_prompts import get_10k_prompt  # noqa: E402

DEFER_MARK = "unified KV occupancy is too high, defer task"
EXCEED_MARK = "Context size has been exceeded"


def fire(port: int, prompt: str, n_predict: int) -> dict:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=payload,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        body = e.read()
        code = e.code
    except Exception as e:  # noqa: BLE001
        return {"code": 0, "s": time.time() - t0, "err": repr(e)}
    out = ""
    try:
        out = json.loads(body).get("content", "")
    except Exception:  # noqa: BLE001
        pass
    return {"code": code, "s": time.time() - t0, "out": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True, help="server log to scan")
    ap.add_argument("--control", action="store_true",
                    help="expect the legacy failure signature (--no-kv-admission build)")
    ap.add_argument("--n-predict", type=int, default=16)
    args = ap.parse_args()

    p1 = get_10k_prompt()
    p2 = p1 + " Explain step by step."  # break exact prefix sharing

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1, f2 = ex.submit(fire, args.port, p1, args.n_predict), \
                  ex.submit(fire, args.port, p2, args.n_predict)
        r1, r2 = f1.result(), f2.result()

    print(f"request A: http {r1['code']} in {r1['s']:.1f}s")
    print(f"request B: http {r2['code']} in {r2['s']:.1f}s")

    log = open(args.log, encoding="utf-8", errors="replace").read()
    n_defer = log.count(DEFER_MARK)
    n_exceed = log.count(EXCEED_MARK)
    n_500 = len(re.findall(r"HTTP 500|\" 500 ", log))
    print(f"log: defer-lines={n_defer} exceeded-lines={n_exceed} http500~={n_500}")

    ok = True
    if args.control:
        if n_exceed == 0 and r1["code"] == 200 and r2["code"] == 200 and n_defer == 0:
            print("CONTROL INCONCLUSIVE: legacy build did not reproduce the "
                  "failure signature - check -c size (must be ~10k, both "
                  "prompts must exceed it together)")
            ok = False
        elif n_exceed > 0 or n_500 > 0:
            print("CONTROL PASS: legacy failure signature reproduced")
        else:
            print("CONTROL FAIL: unexpected signature")
            ok = False
    else:
        if not (r1["code"] == 200 and r2["code"] == 200):
            print("FIX FAIL: non-200 response"); ok = False
        if n_exceed != 0:
            print("FIX FAIL: context-exceeded line present"); ok = False
        if n_defer == 0:
            print("FIX FAIL: admission did not defer (engagement unproven)"); ok = False
        slower = r2 if r2["s"] > r1["s"] else r1
        if slower["s"] < 1.5 * min(r1["s"], r2["s"]) and r1["s"] > 30:
            print(f"FIX WARN: fill order not visibly serialized "
                  f"({r1['s']:.1f}s vs {r2['s']:.1f}s)")
        if ok:
            print("FIX PASS: both 200, zero exceeded, admission defer engaged")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
