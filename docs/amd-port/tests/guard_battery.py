#!/usr/bin/env python3
"""guard_battery.py - Minimal served regression guard suite for llama.cpp TP3.

Implements the house testing law for the AMD/V340L port:
  - Evaluates against live llama-server (config of record: /home/chris/launch_tp3_200k.sh).
  - Cell order:
      1. Prefill Guard: cold-stamped 2k prefill probe (after --idle-wait cooldown).
      2. Decode Guard: decode tok/s at 8k-10k primed context vs baseline.
      3. MTP Acceptance Canary: fails if speculative acceptance < 0.63.
      4. Determinism Guard: consecutive greedy runs at temp 0 must be byte-identical.
      5. Needle Recall Guard: exact 5-digit code retrieval at 25%, 50%, 75% depth.
  - Hardening invariants:
      (a) BATTERY_VERSION row in receipt (sha256 of code + baseline) & session lock.
      (b) Thermal sideband: background sampler every 5s of sclk + edge/junc/mem temps.
      (c) cache_prompt=false on all requests (zero cross-cell cache hits).
      (d) Monotonic baseline ratcheting (--ratchet).
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_BASELINE = HERE / "baseline_tp3_200k.json"
DEFAULT_RECEIPT_DIR = REPO_ROOT / "docs" / "amd-port" / "results"
LOCK_FILE = HERE / ".battery.lock"

from probe_prompts import get_2k_prompt, get_10k_prompt, get_determinism_prompt, get_needle_prompt


# ---------------------------------------------------------------------------
# Session Lock
# ---------------------------------------------------------------------------
def acquire_session_lock(fingerprint: str) -> None:
    """Ensure no concurrent or mid-flight battery modifications."""
    if LOCK_FILE.is_file():
        try:
            with open(LOCK_FILE, "r", encoding="utf-8") as f:
                info = json.load(f)
            prev_pid = info.get("pid")
            if prev_pid and os.path.isdir(f"/proc/{prev_pid}"):
                print(f"ERROR: Another battery session (PID {prev_pid}, fingerprint {info.get('fingerprint')}) is currently in flight!", file=sys.stderr)
                print(f"Concurrent runs are prohibited by session-lock law. Exiting.", file=sys.stderr)
                sys.exit(1)
        except Exception:
            pass

    lock_data = {
        "pid": os.getpid(),
        "start_time": datetime.now().isoformat(),
        "fingerprint": fingerprint
    }
    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)


def release_session_lock() -> None:
    """Release the session lock on termination."""
    try:
        if LOCK_FILE.is_file():
            LOCK_FILE.unlink()
    except Exception:
        pass


atexit.register(release_session_lock)


# ---------------------------------------------------------------------------
# Version & Provenance
# ---------------------------------------------------------------------------
def compute_battery_version() -> Dict[str, str]:
    """Compute sha256 checksums of test files to establish battery version identity."""
    files = {
        "guard_battery": HERE / "guard_battery.py",
        "probe_prompts": HERE / "probe_prompts.py",
        "baseline": DEFAULT_BASELINE
    }
    meta: Dict[str, str] = {}
    combo = hashlib.sha256()
    for name, p in sorted(files.items()):
        content = p.read_bytes() if p.is_file() else b""
        s = hashlib.sha256(content).hexdigest()
        meta[f"{name}_sha256"] = s
        combo.update(s.encode())
    meta["battery_fingerprint"] = combo.hexdigest()[:16]
    return meta


def get_git_provenance() -> Dict[str, Any]:
    """Capture current git commit, tree dirty state, and diff hash."""
    prov: Dict[str, Any] = {
        "commit": "unknown",
        "tree_dirty": False,
        "diff_hash": ""
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        prov["commit"] = commit
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
        ).strip()
        prov["tree_dirty"] = bool(status)
    except Exception:
        pass

    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=str(REPO_ROOT)
        )
        prov["diff_hash"] = hashlib.md5(diff).hexdigest()[:16]
    except Exception:
        pass

    return prov


def collect_provenance(server_binary: str, launch_config: str, allow_dirty: bool) -> Tuple[Dict[str, Any], List[str]]:
    """Collect mandatory run provenance and return (provenance, errors).

    A guard result is only attributable if we can say WHICH binary, built
    from WHICH commit and build config, served it, and that the tree the
    binary claims to represent is clean. Missing hashes or a dirty tree
    with tracked modifications FAIL the battery before any cell runs.
    """
    errors: List[str] = []
    prov: Dict[str, Any] = get_git_provenance()
    prov["binary"] = ""
    prov["binary_sha256"] = ""
    prov["cmake_cache_sha256"] = ""
    prov["launch_config"] = launch_config or ""
    prov["dirty_files"] = []

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True
    ).stdout.splitlines()
    tracked_dirty = [l for l in status if not l.startswith("??")]
    untracked = [l for l in status if l.startswith("??")]
    prov["dirty_files"] = status

    if tracked_dirty:
        (errors if not allow_dirty else []).append(
            "tracked tree is dirty (results may not match HEAD): " + " | ".join(tracked_dirty[:10])
        )
    if untracked:
        # untracked files cannot change a built binary; recorded, not fatal
        prov["untracked"] = untracked[:10]

    if server_binary:
        p = Path(server_binary)
        if p.is_file():
            prov["binary"] = str(p)
            try:
                prov["binary_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            except Exception as e:
                errors.append(f"could not hash server binary: {e}")
            cache = p.parents[1] / "CMakeCache.txt"
            if cache.is_file():
                try:
                    prov["cmake_cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()[:16]
                except Exception:
                    pass
            else:
                errors.append(f"CMakeCache.txt not found next to binary ({cache}); build config unattributable")
        else:
            errors.append(f"server binary not found: {server_binary}")
    else:
        errors.append("--server-binary is required (the guard refuses unattributable results)")

    if allow_dirty and tracked_dirty:
        print("PROVENANCE OVERRIDE: --allow-dirty used; dirty-tree failure suppressed but RECORDED.")
    return prov, errors


def get_model_metadata(model_path: str) -> Dict[str, Any]:
    """Capture model file metadata for receipt provenance."""
    meta: Dict[str, Any] = {"model_path": model_path}
    p = Path(model_path)
    if p.is_file():
        try:
            st = p.stat()
            meta["model_size_bytes"] = st.st_size
            meta["model_mtime"] = datetime.fromtimestamp(st.st_mtime).isoformat()
        except Exception as e:
            meta["model_stat_error"] = str(e)
    return meta


# ---------------------------------------------------------------------------
# Thermal Sideband Sampler
# ---------------------------------------------------------------------------
class ThermalSampler(threading.Thread):
    """Background sampler recording temperatures and clocks every 5 seconds."""

    def __init__(self, log_path: Path, interval: float = 5.0):
        super().__init__(daemon=True)
        self.log_path = log_path
        self.interval = interval
        self.running = True
        self.samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def run(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,elapsed_s,c0_edge,c0_junc,c0_mem,c0_sclk,c1_edge,c1_junc,c1_mem,c1_sclk,c2_edge,c2_junc,c2_mem,c2_sclk\n")

        t0 = time.time()
        while self.running:
            sample = self._sample(time.time() - t0)
            if sample:
                with self._lock:
                    self.samples.append(sample)
                self._append_log(sample)
            time.sleep(self.interval)

    def _sample(self, elapsed: float) -> Optional[Dict[str, Any]]:
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showtemp", "--showclocks", "--json"],
                text=True, stderr=subprocess.DEVNULL
            )
            data = json.loads(out)
            row: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_s": round(elapsed, 1),
                "cards": {}
            }
            for c in ("card0", "card1", "card2"):
                cd = data.get(c, {})
                try:
                    edge = float(cd.get("Temperature (Sensor edge) (C)", 0))
                    junc = float(cd.get("Temperature (Sensor junction) (C)", 0))
                    mem = float(cd.get("Temperature (Sensor memory) (C)", 0))
                    sclk_str = cd.get("sclk clock speed:", "").strip("()")
                    sclk_mhz = int(re.sub(r"[^\d]", "", sclk_str) or 0)
                except Exception:
                    edge, junc, mem, sclk_mhz = 0.0, 0.0, 0.0, 0

                row["cards"][c] = {
                    "edge": edge,
                    "junction": junc,
                    "memory": mem,
                    "sclk_mhz": sclk_mhz
                }
            return row
        except Exception:
            return None

    def _append_log(self, s: Dict[str, Any]) -> None:
        try:
            c = s["cards"]
            c0, c1, c2 = c.get("card0", {}), c.get("card1", {}), c.get("card2", {})
            line = (
                f"{s['timestamp']},{s['elapsed_s']},"
                f"{c0.get('edge', 0)},{c0.get('junction', 0)},{c0.get('memory', 0)},{c0.get('sclk_mhz', 0)},"
                f"{c1.get('edge', 0)},{c1.get('junction', 0)},{c1.get('memory', 0)},{c1.get('sclk_mhz', 0)},"
                f"{c2.get('edge', 0)},{c2.get('junction', 0)},{c2.get('memory', 0)},{c2.get('sclk_mhz', 0)}\n"
            )
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def stop(self) -> Dict[str, Any]:
        self.running = False
        with self._lock:
            if not self.samples:
                return {"thermal_log": str(self.log_path), "samples_count": 0}

            first = self.samples[0]["cards"]
            last = self.samples[-1]["cards"]

            max_edge = max(max(s["cards"].get(k, {}).get("edge", 0) for k in ("card0", "card1", "card2")) for s in self.samples)
            max_junc = max(max(s["cards"].get(k, {}).get("junction", 0) for k in ("card0", "card1", "card2")) for s in self.samples)
            max_mem = max(max(s["cards"].get(k, {}).get("memory", 0) for k in ("card0", "card1", "card2")) for s in self.samples)

            summary = {
                "thermal_log": str(self.log_path),
                "samples_count": len(self.samples),
                "start_edge_c": {k: first[k]["edge"] for k in first},
                "end_edge_c": {k: last[k]["edge"] for k in last},
                "max_edge_c": max_edge,
                "max_junction_c": max_junc,
                "max_mem_c": max_mem,
                "thermal_drift_c": round(max(last[k]["edge"] - first[k]["edge"] for k in first), 2)
            }
            return summary


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------
def http_post_json(url: str, payload: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
    """Send HTTP POST request with JSON payload."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raw = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {resp.status}: {raw}")
        return json.loads(resp.read().decode("utf-8"))


def check_server_health(base_url: str, timeout: float = 5.0) -> bool:
    """Check whether llama-server is healthy."""
    url = f"{base_url}/health"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("status") == "ok"
    except Exception:
        return False
    return False


def parse_server_log_tail(log_path: Optional[str], seek_offset: int) -> Dict[str, Any]:
    """Parse draft acceptance line from server stdout if log path is provided."""
    info: Dict[str, Any] = {}
    if not log_path or not os.path.isfile(log_path):
        return info

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(seek_offset)
            tail = f.read()

        match = re.search(
            r"draft acceptance = ([0-9.]+) \(\s*([0-9]+) accepted /\s*([0-9]+) generated\), mean len =\s*([0-9.]+)",
            tail
        )
        if match:
            info["log_draft_ratio"] = float(match.group(1))
            info["log_draft_accepted"] = int(match.group(2))
            info["log_draft_generated"] = int(match.group(3))
            info["log_mean_acc_len"] = float(match.group(4))
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Guard Cells
# ---------------------------------------------------------------------------
def run_prefill_guard(
    base_url: str,
    baseline_cfg: Dict[str, Any],
    timeout: float = 300.0
) -> Dict[str, Any]:
    """Execute prefill guard cell against /completion (cold-stamped)."""
    cell_meta = baseline_cfg.get("cells", {}).get("prefill_2k", {})
    warn_pct = baseline_cfg.get("thresholds", {}).get("prefill_tps_warn_pct", 5.0)
    fail_pct = baseline_cfg.get("thresholds", {}).get("prefill_tps_fail_pct", 10.0)
    base_tps = cell_meta.get("prompt_per_second", 96.5)

    prompt = get_2k_prompt()
    req = {
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0.0,
        "cache_prompt": False
    }

    t0 = time.time()
    resp = http_post_json(f"{base_url}/completion", req, timeout=timeout)
    wall_s = round(time.time() - t0, 3)

    timings = resp.get("timings", {})
    prompt_tokens = timings.get("prompt_n", 0)
    measured_tps = timings.get("prompt_per_second", 0.0)

    delta_pct = 0.0
    if base_tps > 0:
        delta_pct = round(100.0 * (measured_tps - base_tps) / base_tps, 2)

    status = "PASS"
    notes = [f"{measured_tps:.2f} t/s vs baseline {base_tps:.2f} t/s ({delta_pct:+.2f}%)"]
    if delta_pct < -fail_pct:
        status = "FAIL"
        notes.append(f"exceeds fail threshold (-{fail_pct}%)")
    elif delta_pct < -warn_pct:
        status = "WARN"
        notes.append(f"exceeds warn threshold (-{warn_pct}%)")

    return {
        "guard": "prefill_guard",
        "cell": "prefill_2k",
        "status": status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": timings.get("predicted_n", 0),
        "measured_tps": round(measured_tps, 2),
        "baseline_tps": base_tps,
        "delta_pct": delta_pct,
        "wall_s": wall_s,
        "notes": " | ".join(notes)
    }


def run_decode_guard(
    base_url: str,
    baseline_cfg: Dict[str, Any],
    server_log: Optional[str] = None,
    timeout: float = 300.0
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute decode guard cell and compute MTP canary metrics."""
    cell_meta = baseline_cfg.get("cells", {}).get("decode_8k_10k", {})
    canary_meta = baseline_cfg.get("cells", {}).get("mtp_canary", {})
    warn_pct = baseline_cfg.get("thresholds", {}).get("decode_tps_warn_pct", 5.0)
    fail_pct = baseline_cfg.get("thresholds", {}).get("decode_tps_fail_pct", 10.0)
    canary_min_accept = canary_meta.get("min_acceptance_ratio", 0.63)
    base_decode_tps = cell_meta.get("decode_per_second", 15.57)

    prompt = get_10k_prompt()
    req = {
        "prompt": prompt,
        "n_predict": 128,
        "temperature": 0.0,
        "cache_prompt": False
    }

    log_offset = 0
    if server_log and os.path.isfile(server_log):
        log_offset = os.path.getsize(server_log)

    t0 = time.time()
    resp = http_post_json(f"{base_url}/completion", req, timeout=timeout)
    wall_s = round(time.time() - t0, 3)

    log_stats = parse_server_log_tail(server_log, log_offset)

    timings = resp.get("timings", {})
    prompt_tokens = timings.get("prompt_n", 0)
    completion_tokens = timings.get("predicted_n", 0)
    decode_tps = timings.get("predicted_per_second", 0.0)

    draft_n = timings.get("draft_n", 0)
    draft_n_accepted = timings.get("draft_n_accepted", 0)
    accept_ratio = 0.0
    if draft_n > 0:
        accept_ratio = round(draft_n_accepted / draft_n, 5)

    mean_len = log_stats.get("log_mean_acc_len")
    if mean_len is None:
        draft_rounds = draft_n / 3.0 if draft_n > 0 else 0
        mean_len = round((draft_n_accepted / draft_rounds) + 1.0, 2) if draft_rounds > 0 else None

    delta_tps_pct = 0.0
    if base_decode_tps > 0:
        delta_tps_pct = round(100.0 * (decode_tps - base_decode_tps) / base_decode_tps, 2)

    decode_status = "PASS"
    decode_notes = [f"{decode_tps:.2f} t/s vs baseline {base_decode_tps:.2f} t/s ({delta_tps_pct:+.2f}%)"]
    if delta_tps_pct < -fail_pct:
        decode_status = "FAIL"
        decode_notes.append(f"exceeds fail threshold (-{fail_pct}%)")
    elif delta_tps_pct < -warn_pct:
        decode_status = "WARN"
        decode_notes.append(f"exceeds warn threshold (-{warn_pct}%)")

    decode_res = {
        "guard": "decode_guard",
        "cell": "decode_8k_10k",
        "status": decode_status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "measured_tps": round(decode_tps, 2),
        "baseline_tps": base_decode_tps,
        "delta_pct": delta_tps_pct,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "accept_ratio": accept_ratio,
        "mean_len": mean_len,
        "wall_s": wall_s,
        "notes": " | ".join(decode_notes)
    }

    # MTP Canary verdict
    canary_status = "PASS"
    canary_notes = [f"acceptance {accept_ratio:.4f} (gate: >={canary_min_accept:.2f})"]
    if accept_ratio < canary_min_accept:
        canary_status = "FAIL"
        canary_notes.append(f"canary tripped: acceptance {accept_ratio:.4f} < {canary_min_accept:.2f}")
    elif accept_ratio < canary_min_accept + 0.02:
        canary_status = "WARN"
        canary_notes.append("close to canary threshold")

    canary_res = {
        "guard": "mtp_canary",
        "cell": "mtp_canary",
        "status": canary_status,
        "accept_ratio": accept_ratio,
        "min_acceptance_ratio": canary_min_accept,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "mean_len": mean_len,
        "notes": " | ".join(canary_notes)
    }

    return decode_res, canary_res


def run_determinism_guard(
    base_url: str,
    timeout: float = 300.0
) -> Dict[str, Any]:
    """Execute determinism guard: two identical greedy requests must be byte-identical."""
    prompt = get_determinism_prompt()
    req = {
        "prompt": prompt,
        "n_predict": 64,
        "temperature": 0.0,
        "cache_prompt": False
    }

    t0 = time.time()
    resp1 = http_post_json(f"{base_url}/completion", req, timeout=timeout)
    resp2 = http_post_json(f"{base_url}/completion", req, timeout=timeout)
    wall_s = round(time.time() - t0, 3)

    text1 = resp1.get("content", "")
    text2 = resp2.get("content", "")

    hash1 = hashlib.sha256(text1.encode("utf-8")).hexdigest()[:16]
    hash2 = hashlib.sha256(text2.encode("utf-8")).hexdigest()[:16]

    is_identical = (text1 == text2)
    status = "PASS" if is_identical else "FAIL"

    if is_identical:
        notes = f"byte-identical across 2 runs (sha256: {hash1}, len {len(text1)} chars)"
    else:
        div_idx = 0
        while div_idx < len(text1) and div_idx < len(text2) and text1[div_idx] == text2[div_idx]:
            div_idx += 1
        notes = f"DIVERGED at char {div_idx}: {hash1} vs {hash2}"

    return {
        "guard": "determinism_guard",
        "cell": "determinism_greedy",
        "status": status,
        "identical": is_identical,
        "sha256_1": hash1,
        "sha256_2": hash2,
        "tokens_1": resp1.get("timings", {}).get("predicted_n", 0),
        "tokens_2": resp2.get("timings", {}).get("predicted_n", 0),
        "wall_s": wall_s,
        "notes": notes
    }


def run_needle_recall_guard(
    base_url: str,
    depths: Optional[List[float]] = None,
    timeout: float = 300.0
) -> Dict[str, Any]:
    """Execute needle-in-a-haystack recall test across multiple context depths."""
    if depths is None:
        depths = [0.25, 0.50, 0.75]

    codes = ["48291", "71503", "92618"]
    passes = 0
    sub_results = []
    t0 = time.time()

    for idx, depth in enumerate(depths):
        code = codes[idx % len(codes)]
        prompt, expected_code = get_needle_prompt(depth, code, target_tokens=8000)
        req = {
            "prompt": prompt,
            "n_predict": 32,
            "temperature": 0.0,
            "cache_prompt": False
        }
        resp = http_post_json(f"{base_url}/completion", req, timeout=timeout)
        content = resp.get("content", "")
        recalled = expected_code in content
        if recalled:
            passes += 1
        sub_results.append({
            "depth": depth,
            "code": expected_code,
            "recalled": recalled,
            "output_snippet": content.strip()[:60]
        })

    wall_s = round(time.time() - t0, 3)
    status = "PASS" if passes == len(depths) else "FAIL"
    depth_pcts = [int(d * 100) for d in depths]
    notes = f"{passes}/{len(depths)} depths recalled exact ({depth_pcts}%)"
    if status == "FAIL":
        failed_depths = [f"{int(sr['depth']*100)}%" for sr in sub_results if not sr["recalled"]]
        notes += f" | FAILED at depths: {', '.join(failed_depths)}"

    return {
        "guard": "needle_recall_guard",
        "cell": "needle_recall_8k",
        "status": status,
        "recalled_count": passes,
        "total_depths": len(depths),
        "depth_details": sub_results,
        "wall_s": wall_s,
        "notes": notes
    }


def update_baseline_ratchet(
    baseline_path: Path,
    baseline_cfg: Dict[str, Any],
    results: List[Dict[str, Any]]
) -> int:
    """Ratchet baseline values upward if measured numbers exceed baseline."""
    n_raised = 0
    cells = baseline_cfg.get("cells", {})

    for r in results:
        cell_name = r.get("cell")
        if cell_name == "prefill_2k":
            old_pp = cells.get("prefill_2k", {}).get("prompt_per_second", 0.0)
            new_pp = r.get("measured_tps", 0.0)
            if new_pp > old_pp and r.get("status") == "PASS":
                cells["prefill_2k"]["prompt_per_second"] = new_pp
                n_raised += 1
        elif cell_name == "decode_8k_10k":
            old_dec = cells.get("decode_8k_10k", {}).get("decode_per_second", 0.0)
            new_dec = r.get("measured_tps", 0.0)
            if new_dec > old_dec and r.get("status") == "PASS":
                cells["decode_8k_10k"]["decode_per_second"] = new_dec
                cells["decode_8k_10k"]["mtp_accept_ratio"] = r.get("accept_ratio")
                if r.get("mean_len"):
                    cells["decode_8k_10k"]["mtp_mean_len"] = r.get("mean_len")
                n_raised += 1

    if n_raised > 0:
        baseline_cfg["ratchets"] = baseline_cfg.get("ratchets", 0) + 1
        baseline_cfg["last_ratchet_ts"] = datetime.now().isoformat()
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(baseline_cfg, f, indent=2)
            f.write("\n")

    return n_raised


def main() -> int:
    parser = argparse.ArgumentParser(description="llama.cpp TP3 200k Served Guard Battery")
    parser.add_argument("--port", type=int, default=8080, help="llama-server port (default: 8080)")
    parser.add_argument("--server-url", type=str, default="", help="Override full server base URL")
    parser.add_argument("--baseline", type=str, default=str(DEFAULT_BASELINE), help="Path to baseline json")
    parser.add_argument("--server-log", type=str, default="", help="Path to server log for timing details")
    parser.add_argument("--output-jsonl", type=str, default="", help="Path to append receipt JSONL")
    parser.add_argument("--idle-wait", type=int, default=180, help="Seconds to wait before prefill cell for cold-stamping (default: 180)")
    parser.add_argument("--prefill-only", action="store_true", help="Run only the 2k prefill guard")
    parser.add_argument("--decode-only", action="store_true", help="Run only the 8k-10k decode guard")
    parser.add_argument("--canary-only", action="store_true", help="Run only MTP canary evaluation")
    parser.add_argument("--determinism-only", action="store_true", help="Run only the determinism guard")
    parser.add_argument("--needle-only", action="store_true", help="Run only the needle-recall guard")
    parser.add_argument("--ratchet", action="store_true", help="Ratchet baseline upward on verified gains")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-guard HTTP timeout in seconds")
    parser.add_argument("--server-binary", type=str, default="",
                        help="Path to the llama-server binary under test (REQUIRED: hashed into provenance)")
    parser.add_argument("--launch-config", type=str, default="",
                        help="Full server launch command + env, recorded verbatim in provenance")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Suppress the dirty-tree failure (recorded loudly; tracked dirt still reported)")
    args = parser.parse_args()

    # 0. Provenance Gate (fail closed): no binary hash, no build-config
    #    hash, or a dirty tracked tree -> the battery refuses to run.
    provenance, prov_errors = collect_provenance(args.server_binary, args.launch_config, args.allow_dirty)
    if prov_errors:
        print("=== PROVENANCE FAIL - battery refused ===", file=sys.stderr)
        for e in prov_errors:
            print(f"  {e}", file=sys.stderr)
        print("Every guard number must be attributable: commit hash, clean tree,")
        print("binary sha256, build-config sha256, launch config.", file=sys.stderr)
        return 2
    print("Provenance Gate:     PASS (commit {}, binary {}, config {})".format(
        provenance.get("commit", "?")[:12],
        provenance.get("binary_sha256", "?"),
        provenance.get("cmake_cache_sha256", "?")))
    if provenance.get("tree_dirty"):
        print("WARNING: untracked files present (recorded, cannot affect the binary):")
        for l in provenance.get("dirty_files", []):
            if l.startswith("??"):
                print(f"  {l}")

    # 1. Battery Version Identity
    version_meta = compute_battery_version()
    fingerprint = version_meta["battery_fingerprint"]

    # 2. Session Lock
    acquire_session_lock(fingerprint)

    base_url = args.server_url or f"http://127.0.0.1:{args.port}"
    baseline_path = Path(args.baseline)

    if not baseline_path.is_file():
        print(f"ERROR: Baseline file not found: {baseline_path}", file=sys.stderr)
        return 1

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_cfg = json.load(f)

    print(f"=== llama.cpp TP3 200k Guard Battery ===")
    print(f"Battery Fingerprint: {fingerprint}")
    print(f"Target Server:       {base_url}")
    print(f"Baseline File:       {baseline_path.name}")

    if not check_server_health(base_url):
        print(f"ERROR: llama-server at {base_url} is not healthy (/health did not respond ok).", file=sys.stderr)
        return 1

    # Initialize Receipt Path & Thermal Sampler
    receipt_dir = DEFAULT_RECEIPT_DIR
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    receipt_path = Path(args.output_jsonl) if args.output_jsonl else receipt_dir / f"tp3_guards_{stamp}.jsonl"
    thermal_log_path = receipt_path.parent / f"{receipt_path.stem}_thermal.log"

    print(f"Starting thermal sideband sampler -> {thermal_log_path.name}...")
    thermal_sampler = ThermalSampler(thermal_log_path, interval=5.0)
    thermal_sampler.start()

    # 3. Idle Cooldown if requested (to cold-stamp prefill)
    if args.idle_wait > 0:
        print(f"\n--> [idle-cooldown] Resting GPUs for {args.idle_wait}s to ensure clean cold-stamp prefill...")
        for remaining in range(args.idle_wait, 0, -10):
            print(f"    Cooldown: {remaining}s remaining...")
            time.sleep(min(10, remaining))
        print("    Cooldown complete.")

    guard_results: List[Dict[str, Any]] = []

    # 4. Cell Execution in strict order:
    #    (1) Prefill Guard (COLD-STAMPED)
    #    (2) Decode Guard & MTP Canary
    #    (3) Determinism Guard
    #    (4) Needle Recall Guard

    # (1) Prefill Guard
    if not (args.decode_only or args.canary_only or args.determinism_only or args.needle_only):
        print("\n--> Running Prefill Guard (2k prompt, cold-stamped)...")
        try:
            r_prefill = run_prefill_guard(base_url, baseline_cfg, timeout=args.timeout)
            guard_results.append(r_prefill)
        except Exception as e:
            guard_results.append({
                "guard": "prefill_guard",
                "cell": "prefill_2k",
                "status": "FAIL",
                "error": str(e),
                "notes": f"Request failed: {e}"
            })

    # (2) Decode Guard & MTP Canary
    if not (args.prefill_only or args.determinism_only or args.needle_only):
        print("--> Running Decode Guard & MTP Canary (8k-10k primed context)...")
        try:
            r_decode, r_canary = run_decode_guard(
                base_url, baseline_cfg, server_log=args.server_log, timeout=args.timeout
            )
            if not args.canary_only:
                guard_results.append(r_decode)
            if not args.decode_only:
                guard_results.append(r_canary)
        except Exception as e:
            err_entry = {
                "guard": "decode_guard",
                "cell": "decode_8k_10k",
                "status": "FAIL",
                "error": str(e),
                "notes": f"Request failed: {e}"
            }
            if not args.canary_only:
                guard_results.append(err_entry)
            if not args.decode_only:
                guard_results.append({
                    "guard": "mtp_canary",
                    "cell": "mtp_canary",
                    "status": "FAIL",
                    "error": str(e),
                    "notes": f"Canary evaluation failed: {e}"
                })

    # (3) Determinism Guard
    if not (args.prefill_only or args.decode_only or args.canary_only or args.needle_only):
        print("--> Running Determinism Guard (consecutive greedy runs at temp 0)...")
        try:
            r_det = run_determinism_guard(base_url, timeout=args.timeout)
            guard_results.append(r_det)
        except Exception as e:
            guard_results.append({
                "guard": "determinism_guard",
                "cell": "determinism_greedy",
                "status": "FAIL",
                "error": str(e),
                "notes": f"Determinism test failed: {e}"
            })

    # (4) Needle Recall Guard
    if not (args.prefill_only or args.decode_only or args.canary_only or args.determinism_only):
        print("--> Running Needle Recall Guard (25%, 50%, 75% depth of primed context)...")
        try:
            r_needle = run_needle_recall_guard(base_url, timeout=args.timeout)
            guard_results.append(r_needle)
        except Exception as e:
            guard_results.append({
                "guard": "needle_recall_guard",
                "cell": "needle_recall_8k",
                "status": "FAIL",
                "error": str(e),
                "notes": f"Needle recall test failed: {e}"
            })

    # Stop Thermal Sampler and gather thermal stats
    thermal_summary = thermal_sampler.stop()

    # Stamp provenance into every cell row
    for r in guard_results:
        r["provenance"] = provenance

    # Compute overall verdict
    worst = "PASS"
    for r in guard_results:
        st = r.get("status", "FAIL")
        if st == "FAIL":
            worst = "FAIL"
            break
        elif st in ("WARN", "CONFIG") and worst == "PASS":
            worst = "WARN"

    # Display Summary Table
    print("\n" + "=" * 92)
    print(f"{'STATUS':<8} {'GUARD':<18} {'METRIC':<18} {'MEASURED':<14} {'BASELINE':<14} {'NOTES'}")
    print("-" * 92)
    for r in guard_results:
        g = r.get("guard", "")
        st = r.get("status", "")
        notes = r.get("notes", "")

        if g == "prefill_guard":
            m_val = f"{r.get('measured_tps', 0):.2f} t/s"
            b_val = f"{r.get('baseline_tps', 0):.2f} t/s"
            metric = "prompt_tps"
        elif g == "decode_guard":
            m_val = f"{r.get('measured_tps', 0):.2f} t/s"
            b_val = f"{r.get('baseline_tps', 0):.2f} t/s"
            metric = "decode_tps"
        elif g == "mtp_canary":
            m_val = f"{r.get('accept_ratio', 0):.4f}"
            b_val = f">={r.get('min_acceptance_ratio', 0):.2f}"
            metric = "draft_accept"
        elif g == "determinism_guard":
            m_val = "identical" if r.get("identical") else "diverged"
            b_val = "identical"
            metric = "text_sha256"
        elif g == "needle_recall_guard":
            m_val = f"{r.get('recalled_count', 0)}/{r.get('total_depths', 3)}"
            b_val = f"{r.get('total_depths', 3)}/{r.get('total_depths', 3)}"
            metric = "exact_recall"
        else:
            m_val, b_val, metric = "-", "-", g

        print(f"{st:<8} {g:<18} {metric:<18} {m_val:<14} {b_val:<14} {notes}")
    print("=" * 92)
    print("PROVENANCE: commit={} tree_clean={} binary={} config={}".format(
        provenance.get("commit", "?")[:12],
        not provenance.get("tree_dirty"),
        provenance.get("binary_sha256", "?"),
        provenance.get("cmake_cache_sha256", "?")))
    print(f"OVERALL VERDICT: {worst}")
    print(f"THERMAL DRIFT:   +{thermal_summary.get('thermal_drift_c', 0.0)}°C (Max Edge: {thermal_summary.get('max_edge_c')}°C, Max Junc: {thermal_summary.get('max_junction_c')}°C)\n")

    # Build Provenance Receipt
    receipt = {
        "timestamp": datetime.now().isoformat(),
        "verdict": worst,
        "battery_version": version_meta,
        "git": provenance,
        "model": get_model_metadata(baseline_cfg.get("config", {}).get("model", "")),
        "thermal": thermal_summary,
        "config": baseline_cfg.get("config", {}),
        "results": guard_results
    }

    try:
        with open(receipt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(receipt) + "\n")
        print(f"Receipt written to: {receipt_path}")
        print(f"Thermal log:        {thermal_log_path}")
    except Exception as e:
        print(f"WARNING: Could not write receipt: {e}", file=sys.stderr)

    if args.ratchet:
        raised = update_baseline_ratchet(baseline_path, baseline_cfg, guard_results)
        if raised > 0:
            print(f"Ratchet: baseline updated (+{raised} metrics raised) -> {baseline_path.name}")
        else:
            print(f"Ratchet: no metrics exceeded baseline; baseline unchanged.")

    return {"PASS": 0, "WARN": 2, "FAIL": 1}[worst]


if __name__ == "__main__":
    sys.exit(main())
