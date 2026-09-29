# k4v4 `<4,4>` — Step D acceptance evidence

Captured 2026-09-05 on `wo/k4v4-prologue` at Step D, RTX 5060 Ti device 0, lease `agent1-k4v4`.
Harnesses rebuilt from source against the merged tree; no binaries reused.

## sha256 of raw outputs

```
eb661b130fa9acf96fd680b6370c6ed722ec18e11cf92d0af89d037f53f9ad4e  k4v4_oracle.txt
31da68e0ff4afb5b65fa08882bb720a6e4c49eae3a35050115ed321329383f15  k4v4_dispatch_route.txt
a47bd314d041bc399b34835feb0e523da6ebe55ef90c01feb5e27242d9d5fb9c  k5v4_oracle_reverified.txt
```

The third is the **sealed `<5,4>` oracle hash re-run after Step D** — it equals
`MANIFEST.txt`'s `oracle.txt` exactly, which is what proves the oracle harness was
parameterised without disturbing the shipped tier.

## How to reproduce

One harness serves both single-buffered 4-bit-V tiers (`docs/117` Step D chose parameterisation
over a 510-line clone — the same anti-fork rule as the prologue extraction):

```bash
NVCC=/usr/local/cuda-13.1/bin/nvcc
C="-std=c++20 -O2 -I include -I src -I tests -I third_party --expt-relaxed-constexpr \
   -diag-suppress 20050 -diag-suppress 177 -gencode arch=compute_120a,code=sm_120a"
L="build/src/libninfer_ops.a build/src/libninfer_core.a build/src/libninfer_nvfp4_tma.a \
   -lcudart -lcublasLt -ldl"
# k4v4:
$NVCC $C -DNINFER_KVARN_ORACLE_KBITS=4 tools/i4_oracle/k5v4_oracle.cu $L -lcuda -o /tmp/k4v4_oracle
# k5v4 (default, must match MANIFEST.txt):
$NVCC $C tools/i4_oracle/k5v4_oracle.cu $L -lcuda -o /tmp/k5v4_oracle
# routing (covers all three tiers in one run):
$NVCC $C tools/i4_oracle/k5v4_dispatch.cu $L -lcuda -o /tmp/kvarn_dispatch
```

## Results

**Oracle — 10/10 PASS, 0 FAIL.** Worst `max_err=0.0254412` against `tol=0.0610617`. For
comparison k5v4's worst is 0.0311902 / 0.0616363, so k4v4 lands in the same band — which is the
expected result for a tier composed from two already-proven sides with zero new bit-manipulation
code.

**Routing — 15 PASS / 0 FAIL, rc=0.** `rel(k4v4 vs k4v2)=1.001803` and
`rel(k4v4 vs k5v4)=0.666765` over identical bytes, so the launcher genuinely selects a third
route rather than falling through. Native `(4,4)` fixture: `err(k4v4 vs bf16)=0.116836`, against
k5v4's 0.105060 — same band, as expected.

## Two things this file deliberately does NOT claim

1. **`err(k4v4 vs bf16)=0.672207` in the route section is NOT a bug and is not a quality figure.**
   The routing fixture is committed at `k5v4` and then *misread* as `(4,4)` — that is the whole
   point of a routing test, and it is why the sealed run scores `err(k4v2 vs bf16)` at 13.75.
   Only the **native** fixture row is a plausibility figure. An earlier draft of this check scored
   the misread against a 0.25 bar and failed; the assertion was wrong, not the kernel.
2. **`ninfer_slice7_kvarn_k4v4_test` is still RED, and that is correct.** Its green path
   (`run_case` / `run` under `HAS_SLICE7_K4V4`) is `return 0;` with no work — defining
   `NINFER_HAS_KVARN_K4V4` would flip 11 oracle cases to green without executing anything. Its
   public-op path IS implemented. See the Step D report; the fix belongs with the test's owner.

## Known cosmetic defect, left unfixed on purpose

`tools/i4_oracle/k5v4_oracle.cu` prints its verdict line as
`ninfer_slice4_kvarn_test: PASS` — a stale label from the harness it was cloned from. It is the
same string the real slice4 CMake test prints, so results are confusable. NOT fixed here because
that line is inside the sealed `<5,4>` `oracle.txt`, and moving a sealed hash for cosmetics would
destroy the one anchor this whole lane depends on. Fix by re-sealing deliberately, not in
passing.
