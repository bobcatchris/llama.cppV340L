# Debrief — A2 DFlash2 lane, session 01a0762a
### Read with `HANDOFF_A2_dflash2.md` (standing state). The previous session's debrief is `HANDOFF_A2_dflash2_debrief.md`; its §6 traps and §9 "do not re-derive" list still apply and are NOT repeated here except where this session changed them.

**This file is written to be executed, not admired.** §1 is the first ten minutes. Every
number has the command that produced it next to it, because §6.11 of the last debrief is
true and this file already caught its own author out twice (§7).

---

## 0. The one-paragraph state

The DFlash2 drafter chain now **executes end to end on device and produces real draft
token ids** (`lane=0 n=5 ids: 220 11 430 274 198`), but nothing consumes them yet: the 2a
loud gate still stands, so a DFlash2 server still refuses to decode, by design. Along the
way the chain's first-ever execution found **blocker H** — three of the drafter's own
kernels indexed activations as the transpose of every producer in the repo, one of them
invisible because the matrix was square at the only geometry ever tested. That is fixed,
gated and mutation-proved. **One piece of work stands between here and DFlash2 decoding:
2b-iv(b), planned to the line number in §22.54 and inlined in §3 below.**

---

## 1. First ten minutes, in order

```bash
cd /home/intel/ninfer/worktrees/wo-dflash2

# 1. Reality check. tip must equal remote; disk was 26G free at this debrief.
git log --oneline -3 && git status --short && git ls-remote github wo/dflash2-scope
df -h / | tail -1

# 2. Baseline green BEFORE you touch anything. 10/10, ~3 s.
/usr/bin/ctest --test-dir build -E "ninfer_dflash2_block_test" -R \
 "^(ninfer_speculative_backend_enum_test|ninfer_tp2_budget_test|ninfer_dflash2_config_test|\
ninfer_dflash2_block_ref_test|ninfer_dflash2_sidecar_test|ninfer_dflash2_conv_layout_test|\
ninfer_diag_sources_compile_check|ninfer_request_log_test|ninfer_serve_options_test|\
ninfer_dflash2_symbols_check)$"

# 3a. The three gates. NEEDS A GRANT (it calls cudaSetDevice), 1.3 s.
cmake --build build --target ninfer_ops ninfer_dflash2_block_test -j4 && ./build/tests/ninfer_dflash2_block_test

# 3b. The container refusals. NO grant needed, 0.0 s, creates no context.
./tools/smoke/diag/build_d2_probe.sh && /tmp/d2_probe --mode negatives --quick

# 3c. The rope one-hot (§22.31 item 4, already DONE — re-run only if you touch rope).
#     NEEDS A GRANT, 1.0 s. Build line is in the driver's own header comment.
g++ -std=c++20 -I src -I include -I third_party -I src/targets/qwen3_6/export \
  -I src/targets/qwen3_6_27b/export -I /usr/local/cuda-13.1/include \
  tools/smoke/diag/dflash2_rope_onehot.cpp -Wl,--start-group build/src/libninfer_ops.a \
  build/src/libninfer_nvfp4_tma.a build/src/libninfer_core.a build/src/libninfer_artifact.a \
  -Wl,--end-group -L/usr/local/cuda-13.1/lib64 -lcudart -lcuda -ldl -lpthread \
  -Wl,-rpath,/usr/local/cuda-13.1/lib64 -o /tmp/rope_onehot && /tmp/rope_onehot

# 4. Ask for a grant by SESSION ID, not the name "coordinator"
#    (name resolution is ambiguous — see §6). Then read §22.54 and start on §3 below.
```

**Step 3's negatives arm creates no CUDA context** — verified: `nvidia-smi
--query-compute-apps` is empty before and after it. It is a CPU check now. Do not wait for
a window to run it.

---

## 2. Measured iteration costs (these change the plan — the old handoff's numbers were wrong)

| what | measured | the old claim |
|---|---|---|
| full TpBackend load + `--mode kvarn` run | **14.1 s** | "~60 s" |
| the three gates (`ninfer_dflash2_block_test`) | **1.3 s** | "~4 s" |
| rope one-hot driver | **1.0 s** | never run |
| `--mode negatives --quick` | **0.0 s**, no context | needed a window |
| `cmake --build --target ninfer_engine` (1 TU dirty) | **12 s** | — |
| `emit_sidecar.py --encoding w8g32_f16s` | **178 s** | — |
| CPU suites (10 by name) | **2.5 s** | — |

**Consequence:** a device iteration loop is ~15 s, not minutes. 2b-iv(b) is debuggable in
this session's time even with several red runs. Budget accordingly — the expensive thing
is thinking, not waiting.

---

## 3. The next action: 2b-iv(b). Six fork points, inlined so you need not open a 4,500-line doc

Full reasoning in **docs/151 §22.54**. The two measured facts that make it tractable:

- `if (mtp)` spans **tp2_backend.cpp:3734-4611** (brace-matched, 878 lines) and its
  `st.*` dependencies are exactly `decoder, dev_draft_vocab_ptr, draft_vocab_ids,
  kvarn_lane_ws, lanes, text, work`. **No `st.mtp_view`, no `st.round`.** The round is a
  generic batched verify/accept/rewind with an MTP-shaped draft tail.
- §17.2 already answers the one question that looked open — the drafter frontier after a
  partial accept: *"restoring the frontier to the block anchor suffices … a frontier
  integer should suffice; VERIFY at wiring."* So `hist_hi = lane.cur_F`,
  `hist_lo = max(0, hist_hi - 2048)`, block positions `cur_F + j`. **One integer**, which
  is also §22.30's invariant.

| # | site | change | risk |
|---|---|---|---|
| 1 | `:2631` `const bool mtp = !dflash2 && ...` | add `const bool drafting = mtp \|\| dflash2;`, use at `:3734`. **Leave `mtp` itself false** — it gates `make_rank`'s MTP buffers and the bind plan | low |
| 2 | `:3996` `target_verify_batch(...)` | this overload has **no sink argument**. Bind `dflash2_batch_feature_sink` as `:3315` does, or `pending_features` freezes at round-0 taps and every later round drafts from a stale stream | **HIGH — silent** |
| 3 | `:4467`-`:4607` draft tail (both `mtp_forward_decode_batch` sites, 4467 and 4580) | replace with the chain when `dflash2` | medium |
| 4 | `:4599`-`:4606` `if (rank == 0) { lane.drafts.assign(k, 0); ... }` | keep the rank-0 write, but **both ranks must still run the chain** — the head allgather is a collective and a rank that skips it deadlocks the other | medium |
| 5 | `:3867` INV-7 draft-vocab domain check | **bypass for DFlash2, do not relax it**: the walk emits full-vocab ids and `st.dev_draft_vocab_ptr` is null on this path. Applying the MTP test rejects every draft | low |
| 6 | `:4322` `kvarn_rewind_lane` | the drafter pool's `hist_*` must track the committed frontier | **HIGH — first non-empty history ever** |

`dflash2_round.cu`'s two entry points are already written and device-executed; what is missing is a caller that runs per round. The code to lift is **`tp2_backend.cpp:3390-3708`** (fuse → block_stack → head+allgather → select → evidence). Lift it into a helper and call it from the draft tail; **do not paste it twice.** What must STAY in the probe block: the drafter state alloc (`:3193`-`:3233`), the sink bind (`:3315`) and the tap dump (`:3370`) — that is round 0 and DT2's evidence.

**Then, and only then:** delete the trailing throw at **`:3712`-`:3716`** in the same commit,
with a device run attached.

### The two places this will actually bite

**Point 2** is the one to get right first. If you wire everything and forget the sink, the
run will *work* — round 0 drafts correctly, and every later round drafts the same tokens
from frozen taps. The symptom is a plausible-but-degenerate acceptance curve, not a crash.
Check it by printing the fused-stream hash per round (§22.48's line already does this per
run): **if the hash repeats across rounds, the sink is not bound.** That is a one-line
diagnostic worth adding before you start.

**Point 6** is the first time the attention kernel's history path executes inside the
chain. Every drafter run to date had `hist_lo == hist_hi == 0`, i.e. zero history
iterations. §22.31 gate 1 covers that path on a fixture that populates it deliberately,
but the chain never has. If the first multi-round run produces garbage from round 1
onward while round 0 is clean, **this is the suspect, not the kernels** — and the failure
mode §22.30 warns about (attending a window it never wrote) catches nothing, so check it
by printing `hist_lo/hist_hi/cur_F` per lane per round and reading them.

### Pass condition, stated before the number exists

2b-iv(b) is green when: it decodes, it terminates, drafts are in-domain every round, the
tap dump still prints, and the acceptance number is **recorded, not judged**. A low
acceptance rate is a result. A hang, a refusal, or a repeated fused hash is a bug.
Do not chase acceptance in this commit — that is (d.4)'s job with n stated.

### Three rules that survive the whole change (§6.8, §22.16)

1. **Keep the tap-evidence dump.** It is what gemini's DT2 reads (`grep -n "dflash2 taps
   layer=" src/runtime/tp2/tp2_backend.cpp`).
2. **Keep every admission refusal.** The probe asserts them (`--mode kvarn`: window 0/8, window 6/7 at the TokenTile ceiling, single-lane; `--mode bf16`: non-kvarn before the per-lane fallback; plus the dispatch batch-of-one). Do not count them from prose — `grep -n 'refus\|refuse' tools/smoke/diag/dflash2_tap_probe.cpp`.
3. **Every new line must be gated on `dflash2`/`drafting`.** The MTP path is the shipping
   path; §22.16's argument is what makes an 878-line edit reviewable.

---

## 4. What landed this session (7 commits, all pushed; tip `40f700d3`)

| commit | what | evidence |
|---|---|---|
| `eb72ac61` | **2b-i** sidecar bound at backend init (§22.47) | 1950 MiB/rank on the right card, asserted off `cudaPointerGetAttributes`; mutation: drop `cudaSetDevice` → "landed on 1 but the rank is on 0" |
| `36d9f26e` | **2b-ii** the fuse (§22.48) | `fused` non-zero/lane-distinct/rank-identical; lane-blind mutation goes red |
| `23379373` | **BLOCKER H** activation layout (§22.50) | 3 kernels + oracle + fixture → feature-inner; gates rc=0; transpose mutation 7.8e-3 → **3.9e+0** |
| `eb86c38e` | **2b-iii** 5-block stack executes, behind the gate (§22.51) | `block_stack lane=0 40931/40960`, lanes differ, ranks identical |
| `bea09dc7` | **2b-iv(a)** head + selector + walk execute, behind the gate (§22.52) | `drafts lane=0 n=5 ids: 220 11 430 274 198` |
| `b6fa79d2` | **§22.31 item 4** [I5] RoPE on device (§22.53) | partner energy at dim 64, dim 1 zero |
| `40f700d3` | **§22.54** the integration plan above | measured, not guessed |

Plus `tools/smoke/diag/build_d2_probe.sh` (this debrief's commit): the probe's link line
was 20 load-bearing tokens in prose. §22.50's lesson generalised — a recipe that only
exists in prose is pre-staged code and rots the same way.

---

## 5. §22.31 device checklist — current state

| item | gate | status |
|---|---|---|
| 1-3 | attention / top-k / assemble vs FP64 | **GREEN** (§22.46, pre-session). Re-verified green after blocker H at `23379373` |
| 4 | [I5] rope device one-hot | **DONE** `b6fa79d2` |
| 5 | 2b-i sidecar-at-init | **DONE** `eb72ac61` |
| 6 | 2b-ii fuse | **DONE** `36d9f26e` |
| 7 | BF16 conv re-check under the real chain | **EXECUTED, NOT VALIDATED.** The conv runs 20×/round inside the chain and the output is non-degenerate, but there is no chain-level FP64 reference to compare against. Building one is gemini's stage-C job (§21.2), not a by-product of a window. Do not mark this DONE until a reference exists |

---

## 6. Coordination and resources

- **GPU: RELEASED at session end.** Ask the coordinator in writing to YOUR session id.
  `send` to the name `coordinator` can fail with "multiple disconnected sessions named
  coordinator" — **address the id `01a07140`.** Re-guard at the moment of claim
  (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` → empty, both cards
  15 MiB / 0 %); a clear guard is not a release.
- **Disk 26 G free** at this debrief (`df -h /`). A full-tree build costs ~15 G — build
  TARGETS. Containers, all regenerable (`emit_sidecar.py`, 178 s):
  `/home/intel/models/dflash2_w8g32.sidecar.{json,bin}` (1.9 G, the probe's default),
  `.trunc.sidecar.*` (100 M) and `.drift.sidecar.json` — **the last two are what
  `--mode negatives` consumes; deleting them turns two green cells into a false pass**, so
  keep them or update the arm.
- **Three notices queued for gemini, none delivered (his mesh is down; broker works):**
  1. §22.29 — tensor-class rows are BF16. Re-emit any container of his.
  2. §22.44 — `dflash2_ref::AttnHistory.lo/hi` are per-lane vectors.
  3. **NEW, blocker H:** `dflash2_ref::conv` is now **feature-inner**. Any stage-A/C cell
     that hand-builds a conv operand must pack it the same way, and the right way is
     `pack_2d` in `tests/test_dflash2_block_cuda.cu`, not a hand-written expression.
- **A1:** the docs/152 rule from blocker H was relayed by the coordinator at 10:55Z —
  *a custom kernel's operand index must match its producer's Tensor layout, and the
  fixture must be constructed through Tensor so the match is not a shared expression.*
  He is in a fresh session (`01a07628`) on the (vi) write-time-hashing hunt; his edits
  touch `tp2_backend.cpp`'s prefill/MTP-forward region, which is **adjacent to §3's fork
  points** — coordinate before you both edit the round loop.

---

## 7. New traps this session (the previous §6 list still stands; these are additions)

1. **A byte stride is not an element stride.** `Tensor::nb` is in BYTES
   (`set_contiguous_strides` seeds it from `dtype_size`). Using `nb[1]` as an element
   index corrupts the heap before anything prints — it bit the fixture written to FIX
   blocker H: `malloc(): mismatching next->prev_size`, core dump, zero diagnostics.
2. **A landed header with no in-tree includer has no compile gate.** `dflash2_bind.h`
   said "DEVICE-VERIFIED" in its commit message and had never been compiled by the build;
   its only includer was a scratch TU carrying a `using namespace` that made it work.
   The diag-compile check covers `tools/smoke/diag/*.cpp`, not headers.
3. **A print of an input is not a measurement.** The bind printed `st.ctx.device` — what
   we *asked for*. Delete the `cudaSetDevice` above it and it still prints "device 0"
   while the bytes land on card 1, and it will not even OOM to announce itself.
4. **A square matrix hides a transpose.** `slot*C + col` vs `col*kTopK + slot` are
   indistinguishable when `C == kTopK`, which was true at the only geometry ever run.
   **Make a fixture's two axes different sizes whenever index order is under test.**
5. **A refusal placed after the expensive setup is not a startup refusal.** §22.17's
   literal checked the container after `materialize_tp`. Moving it to the top of
   `create()` made all four negatives CPU-only.
6. **A handoff's numbers go stale inside one session.** This file said "disk ~19 G" and
   "load ~60 s" in its first draft; both were wrong by the time it was committed (26 G,
   14.1 s), and the 60 s figure came from the previous debrief rather than from a
   measurement. §6.11 confirmed again — **measure, then write, then re-read.**

---

## 8. Settled this session — do not re-open without new evidence

- Activation layout: **feature-inner**, `element(feature, token) at feature + token*K`
  (§22.50). The repo's `ops::linear`, `causal_conv1d.cuh:35` and the drafter's own
  attention kernel all agree; the three outliers were the bug.
- The conv **base** keeps `c*K*2 + k*2 + side` — it is a weight the binder lays out
  deliberately (§22.32's transpose), not a stream. Do not "fix" it to match.
- The drafter's cyclic append is `launch_block_kv_append`; the shared op's 4096
  requirement is a contract with its caller, not a defect (A1's blocker-C precedent).
- [I5] RoPE split-half at rot=128 is now a **device measurement** (§22.53). Still
  inference: "the q/k rows need no llama.cpp-style permute".
- The (d.5) drafter cost is **1950 MiB/rank measured**, so 11,089 MB/rank at T=6/L=2
  (§22.47). §22.19's ladder cells were derived from a smaller assumed figure — that is
  (d.5)'s job to re-derive with measured-at-load numbers, **not** yours to patch here.
- The 2a gate stays until the drafts are consumed (coordinator, 10:51Z).
