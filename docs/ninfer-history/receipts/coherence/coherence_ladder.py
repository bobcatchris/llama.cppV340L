#!/usr/bin/env python3
"""COHERENCE LADDER — deterministic serve-coherence test for NVFP4@TP4 (RED capture cell).

Grades known-answer prompts at graded prompt lengths. A leg PASSES iff:
  - http 200, finish in {stop, length}
  - expected answer substring present in content
  - no mixed-script mojibake in content+reasoning (CJK/Cyrillic/Arabic when
    the answer is ASCII)

Usage:
  coherence_ladder.py [--port 8098] [--max-tokens 64] [--long] [--tag NAME]

Writes: <tag>_row.txt + per-leg JSON alongside this script.
Exit code: 0 if all legs PASS, 1 if any FAIL (RED), 2 if service unreachable.
"""
import argparse, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# (leg name, question, filler-word count to add, expected substring in content)
# Filler ' alpha' repeats pad the prompt; the server reports true prompt_tokens.
LEGS_FAST = [
    ("w56",  "What is water?",                                              0, "water"),
    ("a60",  "What is 2 plus 2?",                                           0, "4"),
    ("b61",  "Count from 1 to 5.",                                          0, "5"),
    ("c62",  "Count from 1 to 10.",                                         0, "10"),
    ("d64",  "Reply with exactly the single word BLUE. Context:",           2, "BLUE"),
    ("e66",  "Reply with exactly the single word BLUE. Context:",           4, "BLUE"),
    ("f68",  "Reply with exactly the single word BLUE. Context:",           6, "BLUE"),
    ("g72",  "Reply with exactly the single word BLUE. Context:",          10, "BLUE"),
    ("h80",  "Reply with exactly the single word BLUE. Context:",          18, "BLUE"),
    ("i96",  "Reply with exactly the single word BLUE. Context:",          34, "BLUE"),
    ("j128", "Reply with exactly the single word BLUE. Context:",          66, "BLUE"),
    ("k160", "Reply with exactly the single word BLUE. Context:",          98, "BLUE"),
    ("l224", "Reply with exactly the single word BLUE. Context:",         162, "BLUE"),
    ("m288", "Reply with exactly the single word BLUE. Context:",         226, "BLUE"),
]
LEGS_LONG = [
    ("n1k",  "Reply with exactly the single word BLUE. Context:",   938, "BLUE"),
    ("o2k",  "Reply with exactly the single word BLUE. Context:",  1938, "BLUE"),
]

MOJIBAKE = re.compile(r'[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]')

def ask(port, question, max_tokens, timeout=1800):
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload, time.time() - t0

def grade(payload, expected):
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = payload.get("usage", {})
    text = content + reasoning
    checks = {
        "http_ok": True,
        "finish_ok": ch.get("finish_reason") in ("stop", "length"),
        "answer_ok": expected.lower() in content.lower(),
        "script_ok": not MOJIBAKE.search(text),
        "nonempty_ok": bool(content.strip()) or bool(reasoning.strip()),
    }
    return checks, content, reasoning, usage, ch.get("finish_reason")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--long", action="store_true", help="include 1k/2k legs (slow)")
    ap.add_argument("--tag", default="L1")
    args = ap.parse_args()

    legs = LEGS_FAST + (LEGS_LONG if args.long else [])
    rows, npass = [], 0
    for name, q, k, expected in legs:
        question = q + " alpha" * k
        try:
            payload, wall = ask(args.port, question, args.max_tokens)
        except Exception as e:
            rows.append(dict(leg=name, error=str(e)))
            print(f"leg {name}: SERVICE ERROR {e}")
            continue
        checks, content, reasoning, usage, finish = grade(payload, expected)
        ok = all(checks.values())
        npass += ok
        row = dict(leg=name, prompt_tokens=usage.get("prompt_tokens"),
                   gen=usage.get("completion_tokens"), finish=finish, wall=round(wall, 1),
                   expected=expected, verdict="PASS" if ok else "FAIL", **checks,
                   content_head=content[:80], reasoning_head=reasoning[:60])
        rows.append(row)
        print(f"leg {name}: prompt={row['prompt_tokens']} gen={row['gen']} fin={finish} "
              f"-> {'PASS' if ok else 'FAIL ' + str([k for k, v in checks.items() if not v])}")
        print(f"        content={content[:70]!r}")
        print(f"        reason ={reasoning[:70]!r}")

    tag = args.tag
    with open(os.path.join(HERE, f"{tag}_row.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(HERE, f"{tag}_row.txt"), "w") as f:
        f.write(f"== coherence ladder tag={tag} legs={len(rows)} pass={npass} ==\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"== {npass}/{len(rows)} legs PASS ==")
    sys.exit(0 if npass == len(rows) and rows else 1)

if __name__ == "__main__":
    main()
