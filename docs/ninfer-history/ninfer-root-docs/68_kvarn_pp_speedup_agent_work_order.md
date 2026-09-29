# 68 — KVarN beyond-wall pp speedup (no decode loss): Agent Work Order

**Status:** CURRENT — work order for a single implementer agent.
**Re-scoped 2026-08-25 after step 0** (results/step0_report.md, commit
50127288): pass-2 flash is 97.8% of attention and latency/occupancy-bound;
option A's payoff is not the lever; beyond-wall **decode** is 3.7× degraded
and is now in scope as priority #1.
**Mission:** maximize beyond-wall prompt-processing rate (toward the
staged-path ~700 tok/s) AND fix the beyond-wall decode collapse
(65.6 → 17.5 tok/s from 10k → 40k context), with zero regression elsewhere.
Done = a live 250k KVarN MTP server with: 40k-context decode within 2× of the
10k in-capacity rate (docs/66 step-2 gate); 60k–88k pp ≥1.25× step-0
baselines (T19 ≥450 tok/s floor unchanged, no threshold edits); decode guard
table within 2% at 10k MTP-on / 10k MTP-off with MTP acceptance within 5pp;
must-pass battery green; startup VRAM unchanged at 250k (14,635 MiB/rank ±
a few MiB).
Read this document fully before writing code.

---

## 1. Context (60-second version)

D-18 (docs/66, merged `879bc70d`/`ad95d5af`) fixed the beyond-capacity prefill
collapse with a 2-pass KVarN prefill: pass 1
(`gqa_attention_kvarn_materialize_kernel`) dequants the ENTIRE visible history
into a paged BF16 temp every chunk; pass 2 runs the stock BF16 flash on it.
The cost is O(T²/C) rematerialization, and pass 2's grid
`(q_blocks, q_heads)` makes each kv_head's temp tile be DRAM-read once per
q-head in its GQA group (×6). Rate decays 720 → ~360 tok/s between 30.7k and
225k. docs/67 held the option analysis (A: GQA-shared pass 2, B: segmented
pipeline, C: skip shadow round-trip, D: coalesce scale gathers); step 0
re-prioritized it (§5): decode fix first (docs/66 step 2), pass-2 occupancy
(Br sweep) second, A conditional, B killed, C/D deferred. The **decode
guard** applies at every step (§5.10).

**Already built and verified (do not redo):**
- 2-pass KVarN prefill, temp layout `{D,G,H,P}` (d fastest), tail/code/shadow
  branches bit-exact vs BF16 reference (unit rel_l2 = 0.000000) — `879bc70d`.
- MMA GQA-shared kernel `gqa_attention_kvarn_mma.cuh` (pattern template only —
  see §5.2) and bench harness `tests/bench_kvarn_mma.cpp`.
- KVarN battery T16–T19 (T19: 88k prefill ≥450 tok/s avg; currently ~600).
- `run_ci.sh --full` (one command: verify + serve + int8 T1–T12 + KVarN
  battery + T14 zone) — `5e7c79c8`.

**What you are doing:** steps 1–5 of §6 — step 0 is DONE (report in
`results/step0_report.md`, commit `50127288`: pp baseline 663/641/607 @
40k/60k/88k, gate PASS; decode baseline 65.6/17.5 @ 10k/40k; attribution:
pass-2 = 97.8%, pass-1 = 2.2% DERIVED). Remaining: beyond-wall decode kernel
(docs/66 step 2), pass-2 occupancy (Br sweep), conditional GQA-shared
pass-2 variant, closeout.

## 2. Environment & build/test

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`).
- **Work protocol (MANDATORY):** ALL work happens in a worktree + branch —
  never edit the main tree directly (2026-08-24 `git add -A` incident).
  ```bash
  cd /home/intel/ninfer/repo
  git worktree add ~/ninfer/worktrees/wo-kvarn-pp -b wo/kvarn-pp
  cd ~/ninfer/worktrees/wo-kvarn-pp
  cmake -S . -B build && cmake --build build -j 16
  ```
  Commit per step to `wo/kvarn-pp` and push. **Merging to main is done by the
  main-side agent/user** — do not merge or push to main yourself.
- Build: `cmake --build build -j 16` (CUDA 13.1, arch forced `sm_120a`,
  2× RTX 5060 Ti 16 GB).
- Unit tests: **`/usr/bin/ctest`** (PATH `ctest` is a broken wrapper). From
  `build/`: `/usr/bin/ctest -R "kvarn"` (expect `test_kvarn_gqa` parity
  0.000000 for shadow+code+tail).
- Bench: `build/tests/bench_kvarn_mma` (kQ=4096, 1024 pages).
- Battery: `tools/smoke/serve_correctness_ci.sh` (env-overridable:
  `KV_DTYPE=kvarn_k4v2 KV_CAPACITY=250000 MAX_CONTEXT=250000`, `--only
  T16,T17,T18,T19`, `--mode full`). Full pipeline: `tools/ops/run_ci.sh
  --full`.
- Model artifact: `/home/intel/models/qwen3_8_27b.ninfer`.
- KVarN server config for ALL live tests: `--port 8091 --devices 0,1 --spec
  mtp --draft-tokens 3 --kv-dtype kvarn_k4v2 --kv-capacity 250000
  --max-context 250000`. Launch per LAUNCH.md via `setsid`.
- Probes: `tools/bench/pp_probe.sh`, `tools/bench/decode_guard.sh`,
  `tests/bench_kvarn_2pass.cu` — all committed at step 0; extend them, do not
  recreate. Raw logs to `~/ninfer/logs/`; committed numbers to `results/`.

## 3. Architecture facts (verified — do not re-derive)

- Engine path is **TPEngine** (`src/runtime/tp2/tp_engine.h`), NOT
  ConcurrentExecutor (REPO.md §2a).
- Geometry: **24 query heads, 4 kv heads → GQA group size 6**, `head_dim=256`
  (`src/targets/qwen3_6_27b/impl/config.h:28-30`). NOTE: docs/67's "12 CTAs /
  2 kv_heads" was a typo — the real redundancy factor is 6 (one kv_head's
  temp tile read by its 6 q-head CTAs).
- Pass 1: `gqa_attention_kvarn_materialize_kernel`, grid `(n_tiles,
  kv_heads=4)`, warp-FWHT dequant of codes+scales → temp; temp allocated in
  `src/ops/launcher/gqa_attention_kvarn.cu` as
  `{kKvarnAttnD, kKvarnAttnG, kv_heads, n_tiles}` — **d fastest** (the
  g-fastest allocation was a real bug: misaligned 16B tail copies).
- Pass 2: `gqa_attention_prefill_bf16_kernel`, grid `(ceil(tokens/64),
  q_heads=24)`, `kv_head = q_head / 6` (line ~134) — every q-head CTA loops
  ALL KV blocks itself → ×6 redundant temp reads per kv_head.
- GQA-shared pattern (target structure for option A):
  `src/ops/kernel/gqa_attention_kvarn_mma.cuh` lines 106–133 — one CTA per
  `(q_block, kv_head)`, inner loop over the group's q-heads, KV tile shared
  in smem across the loop.
- Staged-shadow budget: `kKvarnStagedBudgetBytes` = 1 GiB
  (`include/ninfer/ops/kvarn_workspace.h:30`) → ~30.7k tokens staged at the
  250k config. In-capacity (≤30.7k) uses the staged BF16 flash directly
  (~720 tok/s); beyond the wall it is the 2-pass path.
- Decode (`tokens == 1`) is the **legacy per-q-head fused kernel, unchanged
  by D-18** (`src/ops/launcher/gqa_attention_kvarn.cu:79`). It does NOT
  execute the materialize kernel. This is the **target of step 1** (docs/66
  step 2 scope, now in this work order); no other step may touch it.
- Transient temp allocations come from the 1 GiB/rank arena
  (`src/runtime/tp2/tp2_backend.cpp:247`); nothing here may become persistent.
- sm_120 smem: 100 KB/SM, 99 KB dynamic opt-in; a ~96 KB smem kernel = 1
  CTA/SM. Check smem fit BEFORE writing any kernel change.
- Token counts must come from `usage.prompt_tokens`, never char estimates
  (calibration: 1 unit ≈ 5.33 chars/token; the wrong direction once produced
  a 12.7M-token prompt).
- The BF16 verify battery pp number drifted 267.6 → 246 → 225 tok/s on the
  SAME binary between 06:08 and 07:07 (2026-08-25) — environmental clock
  drift, not code. Record `nvidia-smi` clocks with every probe; do not chase
  this number.

## 4. Key call sites (anchors — verify line numbers before editing)

- `src/ops/launcher/gqa_attention_kvarn.cu` — launcher: materialize →
  temp_view → BF16 flash; temp allocation; `tokens == 1` decode branch at
  line ~79 (untouchable).
- `src/ops/kernel/gqa_attention_prefill_bf16.cuh` — pass 2 flash kernel,
  grid geometry lines ~100–137 (option A instantiates a KVarN-only variant
  of this, or a new kernel sharing its math).
- `src/ops/kernel/gqa_attention_kvarn_flash.cuh` — materialize kernel: code
  branch (option D target: per-lane strided scale gathers `T[F * idx]`,
  1152-float = 4.5 KiB field), shadow/tail branches (plain int4 loads/stores;
  `cp.async` is SMEM-only — global use is a real bug we already hit).
- `src/ops/kernel/gqa_attention_kvarn_mma.cuh` — GQA-shared pattern template
  (option A), NOT a prefill workhorse (see §5.2).
- `src/runtime/tp2/tp2_backend.cpp:247` — arena sizing (do not grow).
- `tests/bench_kvarn_mma.cpp` — bench harness; `tests/test_kvarn_gqa.cpp` —
  unit parity (expect 0.000000).
- `tools/smoke/test_serve_correctness.py` — battery T16–T19 (do not modify
  T19's threshold).

## 5. Design decisions (FINAL — do not re-litigate)

1. **Step-0 evidence reshapes the plan** (results/step0_report.md):
   pass-2 flash = 97.8% of attention (36.43 of 37.24 ms/layer); pass-1
   materialize = 2.2% (0.80 ms — DERIVED by subtraction, step 2 must event-
   measure it directly); pass-2 is latency/occupancy-bound (98 KB smem → 1
   CTA/SM, 128 threads, ~20 GB/s effective vs ~448 GB/s peak); attention is
   ~20% of chunk time at 60k, ~40% by 220k. Consequences: pass-2 kernel
   efficiency is the pp lever; the 3.7× beyond-wall decode collapse is the
   bigger user-visible defect and is now in scope.
2. **Beyond-wall decode is IN SCOPE, priority #1** (previously: "decode
   branch untouchable, docs/66 step 2 is a separate work order"). That
   prohibition is lifted ONLY for step 1; every other step still must not
   touch the decode branch.
3. **Br sweep is the first pass-2 experiment** (template constants only, no
   new kernel): Br 64 → 32 → 16 plus KV-tile sizing, targeting 2 CTAs/SM
   (Br=32 ⇒ ~50 KB ⇒ 2 CTAs/SM ⇒ doubled latency hiding). This is the cheap
   discriminator for the occupancy hypothesis before any structural change.
4. **Option A DEMOTED (conditional, step 3):** GQA-shared pass 2 is pursued
   ONLY if the Br sweep plateaus below the 1.25× gate. Constraint for the
   agent: 6-head acc at Br=64/D=256 is 6×64×256×4B = 384 KB ≫ 99 KB —
   impossible; any variant must show its smem math first (realistic floor:
   2-heads/CTA at Br=32 ≈ 96 KB). A is not a pure bandwidth play (it
   amortizes tile-load latency across heads) — but only within smem budget.
5. **REJECTED (killed by step 0): option B (segmented producer/consumer)** —
   its payoff was hiding pass-1 behind pass-2; pass-1 is 2.2% of attention ⇒
   B's maximum payoff is <1% of chunk time. No feasibility proof will be
   attempted.
6. **DEFERRED: options C and D** — they now target pass-1 (2.2%) and the
   scale gathers; execute only if steps 1–2 plateau below their gates, else
   report skipped.
7. **REJECTED as prefill workhorse: the MMA kernel**
   (`gqa_attention_kvarn_mma.cuh`) — measured ~218 tok/s @59k vs ~600 for
   2-pass bf16 flash (docs/66 history, cliff fixed but rate lower). Use its
   GQA-shared *pattern* only (it is the template for step 1's decode kernel).
8. **REJECTED: grid-sync cooperative fusion** (docs/66 §5.9).
9. **REJECTED: persistent dequant mirror** (VRAM headroom at 250k is ~zero;
   14,635 MiB/rank; the 2.4 GiB budget bump OOM'd and was reverted
   `152e7ba9`).
10. **Decode guard (binding, unchanged in spirit):** every step must keep
    10k-ctx MTP-on decode within 2% of step-0 (65.6 tok/s) and 10k-ctx
    MTP-off plain within 2%, MTP acceptance within 5pp (docs/50 baseline
    82.0%); 48 generated tokens, 3-run median from serve.log. **40k MTP-on
    (step-0: 17.5 tok/s) is the TARGET of step 1, not a guard** — it must
    improve (≥2×, i.e. ≥32.8 tok/s), it must not be defended. Steps 2–5
    must keep it from regressing vs step 1. The decode kernel change affects
    in-capacity decode too — 10k within 2% is what enforces bit-exactness
    there (plus T10 + in-capacity A/B byte compare).
6. **Determinism:** in-capacity outputs bit-identical at every step (the
   A/B byte compares enforce it — a step that breaks in-capacity bit-exact
   is a math-order regression, not a perf change). Beyond-wall outputs may
   differ from the `879bc70d` build only in steps that change attention
   summation order (decode kernel, Br change, GQA-shared pass 2); each such
   change is fixed order so results are run-to-run deterministic. T19's
   450 tok/s threshold is never modified.
7. **`--prefill-chunk` experiments stay manual-only** (docs/67 §4) — not part
   of any step.
8. **Pre-existing, out of scope but merge-blocking:** verify battery
   `"MTP long-prefill >512 tok (MTP==plain)"` FAIL (1793-tok prompt; present
   in the `06de4282` baseline). Root-cause it and report; do not let it block
   the pp steps, do not let it be merged past.

## 6. Execution order (commit + test each step before the next)

**Testing standard (applies to every step):** tests must simulate REAL
behavior. A step touching any server-facing path is NOT done on unit tests
alone — spin up the server with the new code and execute the real call
sequence (chunked prefill past the wall, MTP rounds, request lifecycle).
Kernel/unit isolation is necessary, never sufficient (the 2026-08-24
"append jumped pages" defect passed every kernel test and died on the first
live request).

### Step 0 — DONE (do not redo)
Report: `results/step0_report.md` (commit `50127288`). Baselines: pp 663.4 /
640.7 / 606.7 tok/s @ 40k/60k/88k (gate PASS); decode 65.6 / 17.5 tok/s @
10k/40k MTP-on. Attribution: pass-2 = 97.8%, pass-1 = 2.2% (derived). Note
for step 2: event-measure pass-1 directly to close the derived number.

### Step 1 — Beyond-wall decode: GQA-shared decode kernel (docs/66 step 2)
Implement the docs/66 §5.2 GQA-shared decode kernel (pattern:
`gqa_attention_kvarn_mma.cuh:106-133` — one CTA per `(q_block, kv_head)`,
group loop over the 6 q-heads, KV tile shared in smem) and switch the KVarN
decode branch (`gqa_attention_kvarn.cu:79`, `tokens == 1`) to it.
**Tests (must pass before moving on):**
- `test_kvarn_gqa` parity: decode path vs BF16 reference 0.000000 (or
  recorded rel_l2 within existing tolerance).
- **In-capacity A/B:** 30k-token request, decode byte-identical old vs new
  (the kernel change is shared by in-capacity and beyond-wall decode — the
  10k guard below enforces this, but byte compare is the proof).
- **Target gate: 40k-ctx MTP-on decode ≥ 32.8 tok/s (2× the 10k in-cap
  rate / 2, per docs/66 step-2 "within 2× of in-capacity rate").**
- Decode guard: 10k MTP-on within 2% of 65.6; 10k MTP-off plain within 2%;
  MTP acceptance within 5pp. T10 determinism green; T19 unchanged
  (decode-only change; if T19 moves >10%, STOP and report).
- Must-pass subset (T1 T2 T3 T5 T8 T10 T11 + T16 T17 T18) green.

### Step 2 — Pass-2 occupancy: Br sweep (no new kernel)
Template-constant sweep of the existing pass-2 kernel (KVarN variant if a
separate instantiation is needed; stock path must stay bit-identical or
pass the A/B gates below): Br 64 → 32 → 16, KV-tile sizing to fit,
targeting 2 CTAs/SM. Record per-Br: smem, CTAs/SM, occupancy, 60k/88k
median.
Also: extend `tests/bench_kvarn_2pass.cu` to event-time pass-1 directly
(closes the derived 0.80 ms).
**Tests (must pass before moving on):**
- Per-Br table committed to `results/`; in-capacity A/B: chosen Br gives
  byte-identical responses (if fp summation order changes at some Br: that
  Br is rejected — Br is a performance knob, not a math change), OR rel_l2
  ≤ tolerance + T10 green + documented.
- 60k/88k medians ≥1.25× step-0 (641/607 → ~801/759) — OR a documented
  ceiling with the smem/occupancy table showing why (e.g. 2 CTAs/SM
  achieved but no gain ⇒ the hypothesis is wrong, say so).
- T19 green; decode guard (40k MTP-on must not regress below step 1);
  must-pass subset green.

### Step 3 — GQA-shared pass 2 (old option A; CONDITIONAL)
ONLY if step 2 plateaus below the 1.25× gate: new KVarN-only pass-2
instantiation with KV tile shared across heads (grid `(q_blocks,
kv_heads=4)`, or 2-heads/CTA if smem math says so — §5.4: show the smem
budget in the step's first commit, kernel second). Stock pass 2 untouched
for BF16/I8.
**Tests (must pass before moving on):**
- `test_kvarn_gqa` parity unchanged (0.000000 or recorded rel_l2).
- In-capacity A/B: 30k-token request, responses byte-identical old vs new.
- 60k/88k medians ≥1.25× step-0. Decode guard + must-pass subset.

### Step 4 — Options C and D (DEFERRED)
Execute ONLY if steps 1–2 plateau below their gates (C: skip the shadow
round-trip — flash `[0, staged_pages)` from the staged view, materialize+
flash only packed pages ≥ staged; D: coalesce the materialize scale gathers
via smem). Otherwise report skipped with the step-0 numbers as the reason.
**Tests (if executed):** C — 60k probe ≥1.2× step-2/3 median, T19,
in-capacity byte-identical, decode guard + must-pass. D — bit-exact
(`test_kvarn_gqa` 0.000000), dequant-overhead delta recorded, decode guard +
must-pass subset.

### Step 5 — Closeout
`bash tools/ops/run_ci.sh --full` (int8 T1–T12 + KVarN battery @250k + T14
@200k/250k zone) on the final build; all green or documented exceptions;
commit all probe/battery outputs to `results/`.
**Tests (must pass before done):** full CI green; report per §8.

## 7. Constraints (non-negotiable)

- **Worktree only:** no edits in `/home/intel/ninfer/repo` outside your
  worktree, ever (§2).
- **Live end-to-end before "done":** every server-facing step requires the
  real server executing the real path (§6 testing standard).
- **No damage:** commit per step with a message naming the step; tree
  buildable at every commit; no untested code.
- **GPU/server:** one live server at a time; port 8091 serves an active
  conversation — swap protocol: ask the user → `pkill -x ninfer-serve`
  (exact name; NEVER `pkill -f`) → wait GPUs <500 MiB → launch per §2 with
  the KVarN flags → run → restore. Check `pgrep -x ninfer-serve` +
  `nvidia-smi` before every launch (a stale ~14 GB/rank server OOMs startup).
- **No new persistent device allocations.** Startup VRAM at 250k stays
  14,635 MiB/rank ± a few MiB. Transient arena only; if a design needs
  device memory, stop and report.
- **Do NOT touch:** I8/BF16 KV paths, staged-shadow path, prefix-tail (D-16)
  machinery, `kKvarnStagedBudgetBytes`, T19 threshold, arena sizing. The
  decode `tokens == 1` branch is step 1's target and is off-limits to every
  other step. KVarN is opt-in via `--kv-dtype`; all other modes
  bit-identical before/after every step.
- **Decode guard is a gate, not a suggestion:** a step that moves 10k decode
  >2% or MTP acceptance >5pp, or regresses 40k decode below step 1, is
  reverted, not tuned (§5.10).
- **Measurement data is project data:** every probe/battery run committed
  under `results/` with its server config + clocks.

## 8. Definition of done

1. Steps 1–5 committed to `wo/kvarn-pp` with passing tests at each step —
   including the live-path test for every server-facing step.
2. **Live proof:** fresh 250k KVarN server launch with the final code; real
   88k and 225k prompts prefilled past the wall; T19 green; launch command
   recorded in the report.
3. **Decode fixed:** 40k-ctx MTP-on decode vs step-0's 17.5 tok/s —
   improvement reported, gate ≥32.8 tok/s met; 10k MTP-on unchanged within
   2% (65.6), 10k MTP-off within 2%, MTP acceptance within 5pp.
4. **pp maximized per gates:** final 60k/88k medians vs step-0 baselines
   reported with the profile evidence for each landed item (decode kernel,
   Br sweep, GQA-shared variant landed or skipped-with-smem-math, C/D
   landed or deferred-with-reason).
5. All measurement data committed to `results/` (including the direct
   event-timing of pass-1); pre-existing verify long-prefill FAIL
   root-caused and reported (§5.13); report = one paragraph per step + the
   key numbers table (pp per size, decode per ctx, acceptance, VRAM,
   smem/occupancy per Br, clocks).
