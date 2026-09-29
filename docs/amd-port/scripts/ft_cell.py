#!/usr/bin/env python3
"""Fixed-text acceptance cell driver (W34 instrument).

One server boot = one cell. This driver speaks to the booted llama-server
and produces the per-request JSONL that ft_score.py consumes.

Passes per cell:
  GEN      one greedy continuation per corpus item (n_predict = --gen-np,
           temperature 0). Records served acceptance (timings.draft_n /
           draft_n_accepted), the per-round accept lines and the text sha.
           On freeze cells also writes the frozen corpus (token ids).
  CASCADE  (only with --frozen) for each frozen item and each grid point j,
           POST /completion with prompt = full_ids[:prompt_len + j] as a
           TOKEN-ID ARRAY (exact fixed prefix; server-common.cpp tokenize_mixed
           passes ids through) and n_predict = --cascade-np. Round 1 of each
           request drafts off TRUE frozen history; its "accepted k/n" line is
           the fixed-text datum. Token-id prompts make the input text
           IDENTICAL across arms BY CONSTRUCTION (no L3 divergence).

Server surfaces used (all in build-hip/bin/libllama-server-impl.so):
  POST /completion  {"prompt": str | [ids], n_predict, temperature: 0, ...}
    resp.timings.draft_n / draft_n_accepted        server-task.cpp:255-258
  server log per round (LLAMA_TRACE=1):            server-context.cpp:4016/4046
    "accepted k/n draft tokens [(restore checkpoint)]"
  server log per request:                          server-context.cpp:627-630
    "draft acceptance = x (a accepted / g generated), mean len = m"
  per-position (needs -lv 5):                      server-context.cpp:632-633
    "acc per pos = (p1, p2, ...)"
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

ROUND_RE = re.compile(r"accepted\s+(\d+)\s*/\s*(\d+)\s+draft tokens")
ACCEPT_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*([0-9]+) accepted /\s*([0-9]+) generated\), mean len =\s*([0-9.]+)")
PERPOS_RE = re.compile(r"acc per pos = \(([^)]*)\)")


def http_post_json(url, payload, timeout=300.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def log_window(path, lo, hi):
    """Read the server log byte window [lo, hi)."""
    try:
        with open(path, "rb") as f:
            f.seek(lo)
            return f.read(max(0, hi - lo)).decode("utf-8", errors="replace")
    except OSError:
        return ""


def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def parse_rounds(chunk):
    """Return [(accepted, proposed), ...] in order of appearance."""
    return [(int(m.group(1)), int(m.group(2))) for m in ROUND_RE.finditer(chunk)]


def load_corpus(path):
    with open(path, "r", errors="replace") as f:
        c = json.load(f)
    items = c.get("items", [])
    if not items:
        raise SystemExit("corpus has no items: %s" % path)
    for it in items:
        for k in ("id", "kind", "prompt"):
            if k not in it or not it[k]:
                raise SystemExit("corpus item missing %s: %r" % (k, it))
    return c


def tokenize(base, text):
    r = http_post_json(base + "/tokenize",
                       {"content": text, "add_special": True})
    ids = r.get("tokens")
    if not isinstance(ids, list) or not ids:
        raise SystemExit("tokenize returned no tokens for %d chars" % len(text))
    return [int(t) for t in ids]


def completion(base, payload, server_log, meta, out):
    """One request with log-window attribution; returns response json."""
    lo = log_size(server_log)
    t0 = time.time()
    resp = http_post_json(base + "/completion", payload)
    wall = round(time.time() - t0, 3)
    time.sleep(0.05)  # let the final per-request log lines flush
    chunk = log_window(server_log, lo, log_size(server_log))
    t = resp.get("timings", {})
    rec = dict(meta)
    rec.update({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_s": wall,
        "prompt_n": t.get("prompt_n"),
        "predicted_n": t.get("predicted_n"),
        "predict_tps": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n", 0),
        "draft_n_accepted": t.get("draft_n_accepted", 0),
        "rounds": parse_rounds(chunk),
    })
    rec["round1_accepted"], rec["round1_proposed"] = (rec["rounds"][0] if rec["rounds"] else (None, None))
    m = ACCEPT_RE.search(chunk)
    if m:
        rec["log_accept_ratio"] = float(m.group(1))
        rec["log_mean_acc_len"] = float(m.group(4))
    m = PERPOS_RE.search(chunk)
    if m:
        rec["acc_per_pos"] = [float(x) for x in m.group(1).split(",")]
    if "content" in resp:
        rec["text_sha256"] = hashlib.sha256(
            resp["content"].encode("utf-8")).hexdigest()
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return resp


def gen_pass(base, corpus, args, server_log, out):
    texts = {}
    for it in corpus["items"]:
        resp = completion(base, {
            "prompt": it["prompt"],
            "n_predict": args.gen_np,
            "temperature": 0.0,
            "cache_prompt": False,
        }, server_log, {
            "mode": "gen", "cell": args.tag, "arm": args.arm,
            "item": it["id"], "kind": it["kind"], "j": None,
            "n_predict": args.gen_np,
        }, out)
        texts[it["id"]] = resp.get("content", "")
        print("  GEN %-12s kind=%-11s cont_len=%4d chars"
              % (it["id"], it["kind"], len(texts[it["id"]])))
    return texts


def freeze(args, corpus, texts):
    frozen = {"corpus_id": corpus.get("corpus_id"),
              "frozen_by_cell": args.tag,
              "frozen_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "items": []}
    for it in corpus["items"]:
        text = texts.get(it["id"], "")
        if len(text.strip()) < 64:
            raise SystemExit("refusing to freeze short continuation for %s "
                             "(%d chars)" % (it["id"], len(text)))
        full = tokenize(args.base_url, it["prompt"] + text)
        plen = len(tokenize(args.base_url, it["prompt"]))
        if len(full) <= plen + args.grid_first + args.grid_step:
            raise SystemExit("frozen continuation too short for grid: %s "
                             "(full=%d plen=%d)" % (it["id"], len(full), plen))
        frozen["items"].append({
            "id": it["id"], "kind": it["kind"], "prompt": it["prompt"],
            "text": text, "full_ids": full, "prompt_len": plen,
            "cont_tokens": len(full) - plen,
        })
    with open(args.freeze_out, "w") as f:
        json.dump(frozen, f, indent=1)
    print("  FROZEN -> %s (%d items, cont tokens: %s)"
          % (args.freeze_out,
             len(frozen["items"]),
             ",".join(str(i["cont_tokens"]) for i in frozen["items"])))


def cascade_pass(base, args, frozen, server_log, out):
    n_req = 0
    for it in frozen["items"]:
        full, plen, cont = it["full_ids"], it["prompt_len"], it["cont_tokens"]
        js = list(range(args.grid_first, cont + 1, args.grid_step))
        for j in js:
            ids = full[:plen + j]
            completion(base, {
                "prompt": ids,
                "n_predict": args.cascade_np,
                "temperature": 0.0,
                "cache_prompt": False,
            }, server_log, {
                "mode": "cascade", "cell": args.tag, "arm": args.arm,
                "item": it["id"], "kind": it["kind"], "j": j,
                "n_predict": args.cascade_np, "prompt_ids": len(ids),
            }, out)
            n_req += 1
        print("  CASCADE %-12s kind=%-11s points=%d (j=%d..%d)"
              % (it["id"], it["kind"], len(js), js[0], js[-1]))
    return n_req


def summarize(out_path):
    n_cas = n_r1 = 0
    per_kind = {}
    with open(out_path, errors="replace") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("mode") != "cascade" or not r.get("rounds"):
                continue
            k = r["rounds"][0][1]
            if k < 1:
                continue
            n_cas += 1
            hit1 = 1 if r["rounds"][0][0] >= 1 else 0
            n_r1 += hit1
            kd = r.get("kind", "?")
            a, b = per_kind.get(kd, (0, 0))
            per_kind[kd] = (a + hit1, b + 1)
    if n_cas:
        print("  cell p1 (round-1 fixed-text hit rate): %d/%d = %.3f"
              % (n_r1, n_cas, n_r1 / n_cas))
        for kd in sorted(per_kind):
            a, b = per_kind[kd]
            print("    kind %-11s %d/%d = %.3f" % (kd, a, b, a / b if b else 0.0))
    else:
        print("  (no cascade records with round-1 lines yet)")


def selftest(args):
    c = load_corpus(args.corpus)
    ids = [i["id"] for i in c["items"]]
    if len(set(ids)) != len(ids):
        raise SystemExit("corpus item ids not unique: %s" % ids)
    non_ascii = 0
    with open(args.corpus, "rb") as f:
        non_ascii = sum(1 for ch in f.read() if ch > 0x7E or (ch < 0x20 and ch not in (0x09, 0x0A, 0x0D)))
    if non_ascii:
        raise SystemExit("corpus contains %d non-ASCII/control bytes" % non_ascii)
    print("SELFTEST-OK corpus=%s items=%d kinds=%s" %
          (args.corpus, len(ids), ",".join(sorted({i["kind"] for i in c["items"]})) ))
    if args.frozen and os.path.isfile(args.frozen):
        with open(args.frozen, errors="replace") as f:
            fz = json.load(f)
        for it in fz.get("items", []):
            need = it["prompt_len"] + args.grid_first
            if len(it["full_ids"]) < need:
                raise SystemExit("frozen item %s shorter than grid start" % it["id"])
        print("SELFTEST-OK frozen=%s items=%d" %
              (args.frozen, len(fz.get("items", []))))
    print("SELFTEST grid: first=%d step=%d -> points per item ~%d" %
          (args.grid_first, args.grid_step,
           max(1, (args.gen_np - args.grid_first) // args.grid_step + 1)))
    return 0


def main():
    ap = argparse.ArgumentParser(description="fixed-text acceptance cell driver")
    ap.add_argument("--base-url", default="http://127.0.0.1:8082")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--frozen", default="")
    ap.add_argument("--freeze-out", default="")
    ap.add_argument("--server-log", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--arm", default="")
    ap.add_argument("--gen-np", type=int, default=160)
    ap.add_argument("--cascade-np", type=int, default=4)
    ap.add_argument("--grid-first", type=int, default=8)
    ap.add_argument("--grid-step", type=int, default=12)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest(args)

    for req in ("out", "tag", "arm"):
        if not getattr(args, req):
            ap.error("--%s is required for a real (non-selftest) run" % req)
    if not args.server_log:
        ap.error("--server-log is required for a real (non-selftest) run")

    corpus = load_corpus(args.corpus)
    print("FT-CELL %s arm=%s base=%s" % (args.tag, args.arm, args.base_url))

    with open(args.out, "w") as out:
        texts = gen_pass(args.base_url, corpus, args, args.server_log, out)
        if args.freeze_out:
            freeze(args, corpus, texts)
        if args.frozen:
            if not os.path.isfile(args.frozen):
                raise SystemExit("frozen corpus missing: %s" % args.frozen)
            with open(args.frozen, errors="replace") as f:
                frozen = json.load(f)
            n = cascade_pass(args.base_url, args, frozen, args.server_log, out)
            print("  cascade requests: %d" % n)

    print("  records -> %s" % args.out)
    summarize(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
