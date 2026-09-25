#!/usr/bin/env python3
"""Host check for guard_battery.py flag-mode honesty (E-145 replicate).

E-145: --decode-only runs silently omitted the prefill/canary/determinism/
needle rows, so downstream adjudicators read "row absent" as "voided"
(w23env 'text_sha256 diverged' when the sha was never measured; w28nmax
ACCEPT-MISSING). The fix emits SKIP rows with a reason plus a GUARD-CAPS
line at battery start.

Runs main() end-to-end with the network, lock, provenance, and thermal
sampler stubbed out: zero GPU, zero sockets, zero lock contact.

Usage: python3 test_guard_battery_skiprows.py   (exit 0 = all cases pass)
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import guard_battery  # noqa: E402


def fake_provenance(server_binary, launch_config, allow_dirty):
    prov = {
        "commit": "0" * 40, "tree_dirty": False, "diff_hash": "",
        "binary": "/fake/llama-server", "binary_sha256": "f" * 16,
        "cmake_cache_sha256": "a" * 16, "launch_config": "", "dirty_files": []
    }
    return prov, []


class FakeSampler:
    def __init__(self, log_path, interval=5.0):
        self.log_path = log_path

    def start(self):
        pass

    def stop(self):
        return {"thermal_log": str(self.log_path), "samples_count": 1,
                "thermal_drift_c": 0.0, "max_edge_c": 40.0, "max_junction_c": 45.0}


def fake_decode_pass(base_url, baseline_cfg, server_log=None, timeout=300.0):
    decode = {"guard": "decode_guard", "cell": "decode_8k_10k", "status": "PASS",
              "measured_tps": 24.5, "baseline_tps": 15.57, "delta_pct": 57.0,
              "accept_ratio": 0.67, "mean_len": 2.9,
              "notes": "24.50 t/s vs baseline 15.57 t/s (+57.00%)"}
    canary = {"guard": "mtp_canary", "cell": "mtp_canary", "status": "PASS",
              "accept_ratio": 0.67, "min_acceptance_ratio": 0.63,
              "notes": "acceptance 0.6700 (gate: >=0.63)"}
    return decode, canary


def fake_decode_fail(base_url, baseline_cfg, server_log=None, timeout=300.0):
    raise RuntimeError("connection refused")


def fake_guard(name, cell, status="PASS"):
    def run(base_url, baseline_cfg=None, timeout=300.0):
        return {"guard": name, "cell": cell, "status": status, "notes": "fake"}
    return run


def run_battery(argv, decode_behavior=fake_decode_pass):
    out = io.StringIO()
    receipt = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    receipt.unlink()
    real_decode = guard_battery.run_decode_guard
    real_argv = sys.argv
    guard_battery.run_decode_guard = decode_behavior
    sys.argv = argv + ["--output-jsonl", str(receipt),
                       "--baseline", str(TESTS / "baseline_tp3_200k.json"),
                       "--idle-wait", "0"]
    try:
        with contextlib.redirect_stdout(out):
            rc = guard_battery.main()
    finally:
        sys.argv = real_argv
        guard_battery.run_decode_guard = real_decode
    rows = []
    receipt_full = {}
    if receipt.is_file():
        with open(receipt, "r", encoding="utf-8") as f:
            receipt_full = json.loads(f.read().strip())
        rows = receipt_full["results"]
        receipt.unlink()
    return out.getvalue(), rc, rows, receipt_full


def check(label, cond):
    print(("PASS: " if cond else "FAIL: ") + label)
    return cond


def main() -> int:
    ok = True
    guard_battery.collect_provenance = fake_provenance
    guard_battery.check_server_health = lambda url, timeout=5.0: True
    guard_battery.ThermalSampler = FakeSampler
    guard_battery.acquire_session_lock = lambda fp: None
    guard_battery.run_prefill_guard = fake_guard("prefill_guard", "prefill_2k")
    guard_battery.run_determinism_guard = fake_guard("determinism_guard", "determinism_greedy")
    guard_battery.run_needle_recall_guard = fake_guard("needle_recall_guard", "needle_recall_8k")

    # 1. decode-only: SKIP rows present with reason, caps line exact
    out, rc, rows, receipt = run_battery(["prog", "--decode-only"])
    ok &= check("decode-only GUARD-CAPS line exact",
                "GUARD-CAPS: decode=1 prefill=0 determinism=0 canary=0 needle=0" in out)
    ok &= check("decode-only emits 5 rows", [r["status"] for r in rows] ==
                ["SKIP", "PASS", "SKIP", "SKIP", "SKIP"])
    ok &= check("decode-only SKIP rows carry reason",
                all(r.get("reason") == "decode-only" and
                    r["notes"] == "%s SKIP (decode-only)" % r["guard"]
                    for r in rows if r["status"] == "SKIP"))
    ok &= check("decode-only row order prefill,decode,canary,determinism,needle",
                [r["guard"] for r in rows] == ["prefill_guard", "decode_guard",
                                               "mtp_canary", "determinism_guard",
                                               "needle_recall_guard"])
    ok &= check("decode-only table keeps guard+metric names, no fake values",
                "SKIP     determinism_guard  text_sha256        -" in out and
                "determinism_guard SKIP (decode-only)" in out and "diverged" not in out)
    ok &= check("decode-only verdict from real rows only (PASS, rc 0)",
                "OVERALL VERDICT: PASS" in out and rc == 0)
    ok &= check("decode-only receipt guard_caps matches caps line",
                receipt.get("guard_caps") == {"decode": 1, "prefill": 0,
                                              "determinism": 0, "canary": 0,
                                              "needle": 0})

    # 2. full battery: no SKIP rows, caps all enabled
    out, rc, rows, receipt = run_battery(["prog"])
    ok &= check("full battery GUARD-CAPS all enabled",
                "GUARD-CAPS: decode=1 prefill=1 determinism=1 canary=1 needle=1" in out)
    ok &= check("full battery no SKIP rows",
                all(r["status"] == "PASS" for r in rows) and len(rows) == 5)
    ok &= check("full battery receipt carries guard_caps all ones",
                receipt.get("guard_caps") == {"decode": 1, "prefill": 1,
                                              "determinism": 1, "canary": 1,
                                              "needle": 1})

    # 3. determinism-only: decode+canary skipped, others measured
    out, rc, rows, receipt = run_battery(["prog", "--determinism-only"])
    ok &= check("determinism-only caps",
                "GUARD-CAPS: decode=0 prefill=0 determinism=1 canary=0 needle=0" in out)
    ok &= check("determinism-only decode+canary SKIP with reason",
                [r["status"] for r in rows] == ["SKIP", "SKIP", "SKIP", "PASS", "SKIP"] and
                all(r.get("reason") == "determinism-only" for r in rows if r["status"] == "SKIP"))

    # 4. decode failure under --decode-only: real FAIL row wins, SKIPs stay SKIP
    out, rc, rows, receipt = run_battery(["prog", "--decode-only"],
                                         decode_behavior=fake_decode_fail)
    ok &= check("decode-only failure keeps verdict FAIL rc 1",
                "OVERALL VERDICT: FAIL" in out and rc == 1 and
                rows[1]["status"] == "FAIL" and rows[1]["guard"] == "decode_guard")
    ok &= check("decode-only failure keeps canary SKIP (not FAIL)",
                rows[2]["status"] == "SKIP" and rows[2]["reason"] == "decode-only")

    print("SKIPROWS-VERDICT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
