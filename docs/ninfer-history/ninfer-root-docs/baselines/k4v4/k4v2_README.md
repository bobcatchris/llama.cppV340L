# k4v2 `<4,2>` regression gate — the baseline MANIFEST.txt did not provide one

**Why this file exists.** `MANIFEST.txt` seals `<5,4>` as three sha256'd raw outputs
(`oracle.txt`, `dispatch.txt`, `known_answer.txt`), but for the k4v2 side it records only
`PASS rc=0` for three tests. That is not a byte-identity gate: a refactor that changed a
`max_err` from 0.0019 to 0.0021 while staying under tolerance would still print
`ninfer_slice4_kvarn_test: PASS` and still exit 0. The Step C gate was specified as
"`<4,2>` byte-identical to the sealed baseline" and, as written, **that gate was
unfalsifiable** — there was nothing sealed to be identical to.

So Step C's equivalence was proven by **direct pre/post comparison** instead, and this file
seals the result so Step D has a real gate to compare against.

## Method (reproducible)

1. Build `ninfer_slice4_kvarn_test`, `ninfer_kvarn_batched_ops_test`, `ninfer_kvarn_gqa_test`
   at the **post**-Step-C tree → capture → `/tmp/c4post/`.
2. `git checkout 28100ec7 -- src/ops/kernel/gqa_decode_slice4_kvarn.cuh` (the pre-Step-C copy),
   rebuild the same three targets, run → capture → `/tmp/c4pre/`.
3. Compare sha256 of the full stdout+stderr of each test.
4. Restore the Step-C header (verified by sha256 of the file itself), rebuild, re-run →
   `/tmp/c4post2/`, confirming the POST hashes reproduce (determinism, not luck).

`<5,4>` was re-checked independently after Step C, because slice4 and slice6 now share
`gqa_decode_kvarn_prologues.cuh` — so a Step C mistake in the shared header *would* move the
`<5,4>` hash. It did not: all three `MANIFEST.txt` hashes still MATCH.

## Result — Step C is data-equivalent for `<4,2>`

All three outputs byte-identical pre/post, and all **10** `max_err=`/`tol=` values identical.

## sha256 — the sealed `<4,2>` gate for Step D

```
ffc87f87fc6e7c4c521aa4b46aa048a913333805f6f98063d63834f6bf269728  k4v2_slice4_kvarn.txt
ac03b6f37788e0fe5c36dec8f323f98fb109358d48ddf47e79fb8b40b692ecbe  k4v2_batched_ops.txt
ee03b2c42e7f285f922332b323433d8b4afb6293d6c6afdc9f242b822a92a33c  k4v2_gqa.txt
```

Captured 2026-09-05 on `wo/k4v4-prologue` at Step C, RTX 5060 Ti device 0, lease
`agent1-k4v4`, harnesses and tests rebuilt from source (never reused binaries).

**To re-verify after any later edit:** rebuild the three targets, run with
`CUDA_VISIBLE_DEVICES=0`, and compare each output's sha256 to the block above. If a hash
moves, the extraction is wrong — not this baseline.
