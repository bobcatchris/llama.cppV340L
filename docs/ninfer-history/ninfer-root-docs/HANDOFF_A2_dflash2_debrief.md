# Debrief — A2 DFlash2 lane, session 01a07435
### Read this with `HANDOFF_A2_dflash2.md`. That file is the lane's standing state; this one is the hand-off from a 98-commit session and is written to get you productive in one read.

Every claim below was checked against the tree at `c2e353be`, not recalled. Where a
number came from a command, the command is printed with it.

**The review pass after the first draft caught four of my own claims already stale** —
commit count 98→99, the round file 517→522 lines (a later commit grew it), and
§22.31's verdict column still reporting gate 1 red after §22.46 resolved it. Three of
those are §6.8 firing while I wrote §6.8, which is the argument for the review being a
separate pass rather than a reread.

---

## 1. Sixty-second state

```
branch        wo/dflash2-scope        tip c2e353be == remote (verify: git ls-remote github wo/dflash2-scope)
main          cfca00ca                FOLDED into this branch, zero conflicts, verified
this session  99 commits (git log --oneline --no-merges 6459bac6..HEAD | wc -l)
gates 1-3     ALL GREEN               (./build/tests/ninfer_dflash2_block_test -> rc=0, ALL PASS)
CPU suites    9/9 by NAME             (see §5; never a bare -R dflash2)
oracle        34/34 self-checks       (python3 tools/convert/qwen3_8_27b/dflash2/block_graph_ref.py)
GPU           NOT held by this lane. No standing need until you run something.
disk          ~21G free. A full-tree build costs ~15G. Build TARGETS.
```

**The lane's position in one sentence:** every DFlash2 kernel the batched round needs
now exists and is individually verified against an FP64 oracle; what remains is
**wiring**, in a known order, with one decision still pending from A1.

---

## 2. The next action, concretely

**Wire 2b-i (sidecar at backend init).** This is the first unblocked piece and
everything downstream depends on it. Verified current state:

| piece | status | how I checked |
|---|---|---|
| `DFlash2Sidecar::total_row_bytes()` | **DONE** (§22.17 prereq 3) | `grep -n total_row_bytes src/targets/qwen3_6_27b/impl/load/dflash2_sidecar.h` → line 196 |
| `TpBackendOptions::dflash2_sidecar` field | **NOT DONE** (prereq 1) | `grep -c dflash2_sidecar src/runtime/tp2/tp2_backend.h` → 0 |
| `TpRankState` members (sidecar/backing/arena) | **NOT DONE** (prereq 4) | same grep, 0 |
| `tp_engine.cpp` one-line pass-through | **NOT DONE** (prereq 2) | `grep -c b_opts.dflash2_sidecar src/runtime/tp2/tp_engine.cpp` → 0 |
| bind call site in `TpBackend::create` | **NOT DONE** | — |

The literal diff for all of it is **§22.17**, and the binding logic it calls is
already landed and CPU-tested (`dflash2_bind.h` + `dflash2_conv_layout.h` + the
`dflash2_require_bf16` guard), so 2b-i is **the call site and the arena, not the
binding**. That is a correction §22.39 records: §22.17 was written before the
transpose and the guard landed.

**Then 2b-ii** (fuse) — §22.18, literal, unaffected by anything this session changed.
The fuse's output is BF16, which after §22.29 is the *right* dtype rather than an
accident.

**Then 2b-iii** (the chain). See §3.

---

## 3. 2b-iii: what is ready, what is not, and the one decision pending

**Ready and verified:**
- `launch_block_attention` — gate GREEN. Compile-time window = pool capacity, BF16
  storage / FP32 accumulate, per-query admission, non-causal within block.
- `launch_top_candidates` — gate GREEN at the real 248320 domain.
- `launch_block_input_assemble` — gate GREEN, all 4 cells.
- `launch_conv` / `launch_edge_scores` / `launch_repeat_single_pred` — BF16 operands,
  device-verified (§22.26).
- The chain file itself: **`src/runtime/tp2/dflash2_round.cu` (381) + `.h` (141) = 522
  lines, builds into `ninfer_engine`.** Block stack + selector/walk are implemented.
- The blessed shape (§22.39, coordinator 05:57Z): new file, **ONE call** from
  `tp2_backend.cpp`. A1's (vi) work edits that same file, so the overlap is one line.

**Not done, deliberately:**
- The call site is **not wired** and the 2a loud gate is **still in place**
  (`grep -n "drafter chain not bound" src/runtime/tp2/tp2_backend.cpp` → line 3297).
  Wiring it means *removing that gate in the same commit*, and that is the one edit in
  this lane where a half-finished state is actively unsafe: it lets a DFlash2 server
  decode. Do it in one commit with a device run attached, or not at all.
- `dflash2_round.cu` has never executed. It compiles; that is all that is claimed.

**The pending decision:** `launch_block_input_assemble`'s kernel half was routed to A1
with blocker F (§22.30). It is **landed as a skeleton and gate-green**, so the ruling
is now moot in substance — but the *oracle-side* consequence was not: **§22.44's
per-lane `hist_lo`/`hist_hi` change to `dflash2_ref::AttnHistory` is a signature
change to the reference**, and gemini's stage-A/C cells must be written against the
per-lane form. Nobody has told gemini (his mesh is down; see §7).

---

## 4. Blockers found this session — status, so you don't re-find them

| | what | status |
|---|---|---|
| **[I5]** | oracle's rope wrong on pairing, width **and** lattice | **CLOSED** §22.24. Engine was right. Device one-hot still owed (§22.31 item 4) — driver written, compile-gated |
| **C** | `ops::swa` is DFlash v1's attention with v1's 4096 window baked in | **RESOLVED** §22.25. A1 adopted (b): drafter-side kernel. swa untouched |
| **[I9]** | "FP32 staging streams" adopted last night | **RETIRED** as unbuildable. Intent survives as **FP32 accumulators inside every first-cut op** — check against that sentence, not the retired one |
| **D** | no op returns a ranked top-k set | **RESOLVED** §22.27. A1 adopted (i): `launch_top_candidates` |
| **E** | container tensor-class rows were FP16, consumers need BF16; E2 was silent | **RESOLVED** §22.28/22.29. **Re-emit any sidecar on disk** |
| **F** | block-input assembly missing from plan **and** oracle | **RESOLVED both halves** §22.30/22.44, gate green. Oracle signature changed — see §3 |
| **G** | `ops::embedding` has no `Q4G64_F16S` case → codebook gather throws at the **shipping** encoding, works at the measurement one | **OPEN, recommendation stands** §22.41: pin codebooks to `w8g32_f16s` via `--set` (+64.4 MiB for the pair). Routed to A1, non-urgent. **Do not discover this at (d.4)** — the ladder flips manifest fields by design |

**T=8 vs T=6 is settled, no ruling needed** (§22.30): the target's verify is
`T = k+1` (admitted ≤6); the drafter's block pass is **always 8 columns/lane** because
the conv requires `T % block_size == 0`. Blocker B costs acceptance breadth, not conv
geometry.

---

## 5. Verification commands — copy these, do not improvise

```bash
# CPU suites: BY NAME, with the device test excluded. A bare `-R dflash2` matches
# #147, which creates a CUDA context, and this lane has no standing right to that
# without a grant (§22.26 is the story).
/usr/bin/ctest --test-dir build -E "ninfer_dflash2_block_test" -R \
  "^(ninfer_speculative_backend_enum_test|ninfer_tp2_budget_test|ninfer_dflash2_config_test|\
ninfer_dflash2_block_ref_test|ninfer_dflash2_sidecar_test|ninfer_dflash2_conv_layout_test|\
ninfer_diag_sources_compile_check|ninfer_request_log_test|ninfer_serve_options_test)$"

# the FP64 oracle's own self-checks (no artifact, no cards)
python3 tools/convert/qwen3_8_27b/dflash2/block_graph_ref.py            # 34/34
python3 tools/convert/qwen3_8_27b/dflash2/block_graph_ref.py --feedcheck <manifest> <rows>

# the three gates — NEEDS A GRANT, ~4 seconds
cmake --build build --target ninfer_dflash2_block_test -j4 && ./build/tests/ninfer_dflash2_block_test

# targeted builds only; never a full-tree build on this disk
cmake --build build --target ninfer_engine ninfer_ops ninfer ninfer_serve -j4
```

`~/bin/ctest` is a broken shim. Use `/usr/bin/ctest`.

---

## 6. Traps that will bite you, condensed from the eight that bit this session

1. **A `ctest -R` pattern is not a list of tests you chose.** Check what it matches
   before running it on a shared box.
2. **A failed build leaves the previous binary on disk.** Capture build rc *before*
   test rc, or you will read a green from a mutation that never compiled.
3. **`git commit` must be `&&`-chained to the edit it describes.** Twice this session
   a message claimed a change whose script had aborted. Both were caught by noticing a
   *result* disagreed with the claim — not by checking the diff, which is the check
   that works. After any commit saying "updated X": `git show | grep X`.
4. **A 2-byte type is not a type.** FP16/BF16/uint16 are interchangeable to every
   existing check: same size, same shape, no error. This caused E2, three test
   buffers, and one oracle probe. The fix is an assertion at the boundary
   (`dflash2_require_bf16`), not care.
5. **A pre-staged diff is code, and code gets compiled.** §22.23's helper sat
   uncompiled through two rulings and went stale twice. §22.32 is the corrected form
   and it was `-fsyntax-only`'d before being written down.
6. **A green gate is not evidence until you know *why* it passes.** Four non-vacuity
   failures this session, all the same shape: a cell that could not fail in the
   configuration it was handed. §22.36's probe — shrink the fixture along the
   dimension that gates the path, expect red — is the general test.
7. **An aggregate only proves what its dependencies cover.** Three fingerprints
   agreed while the thing they don't touch (V) was broken. See §22.46's table.
8. **Wiring 2b-iii must not delete the 2a tap-evidence dump.** It is what gemini's
   DT2 reads (`grep -n "dflash2 taps layer=" src/runtime/tp2/tp2_backend.cpp`). Keep
   the dump, keep every admission refusal, replace only the trailing
   `"drafter chain not bound"` throw (line 3297). Serve/cli exposure stays last
   (2b-v, A1's §21.3(b) sequencing).
9. **The window ceiling is still `k <= 5`** (T = k+1 ≤ 6) until blocker B closes —
   that is A1's kernel-lane decision, not this lane's. Proposals go to the existing
   verify at T = k+1; the walk emits up to 7 and the caller takes the first k.
10. **`[hist_lo, hist_hi)` and the §17.2 rollback frontier must be ONE number read
    from ONE place** (§22.30). If they diverge the drafter attends a window it never
    wrote, and **no gate catches that** — they check acceptance and arithmetic, not a
    stale bound. Enforce in review. The coordinator has it on the Phase C queue.
11. **Handoff numbers go stale faster than handoff prose.** This file's own counts were
   wrong twice in one day. Verify with the printed command, not from memory —
   including mine above.

---

## 7. Coordination state

- **INTERCOM is the channel** (coordinator `01a07140`, A1 `01a0740f`). The coordinator
  does **not** read `agent_comm`. Verify liveness with `intercom({action:"list"})`.
- **gemini's mesh sends fail** ("Agent not found"); broker delivery works. Two things
  are queued for him and exist only in docs so far:
  1. §22.29 — sidecar tensor-class encoding moved **F16 → BF16**. Not a regression.
     Re-emit containers; re-derive any tolerance pinned to old f16 output.
  2. §22.44 — `dflash2_ref::AttnHistory.lo/hi` are now **per-lane vectors**. Any
     stage-A/C cell written against the old single interval needs updating.
- **A1's (vi)** work edits `tp2_backend.cpp`'s round/prefill region. His six
  `NINFER_MB_HASHPT` diagnostic sites are env-gated and drop-or-keep free at a fold.
  Zero `tp_engine.cpp` changes from him tonight.
- Report a one-line status per commit; pushed SHAs are the only truth.

---

## 8. Where things are recorded (docs/151 is the law; the handoff is the quick-start)

| section | what |
|---|---|
| §22.24 | [I5] closed — rope is split-half, rot=128 |
| §22.25 + addendum | blocker C, A1's five conditions on the drafter attention kernel |
| §22.26 | [I9]→BF16 landed + the UNGRANTED one-off device run and its corrective action |
| §22.27 | 2b-iv pre-stage + blocker D |
| §22.28 / §22.29 | blocker E found / resolved. **§22.29 corrects the argument §22.28 made** |
| §22.30 | 2b-iii pre-stage + blocker F + the T=8/T=6 settlement |
| **§22.31 + addendum** | **the device-window checklist: gates, commands, paired mutations, and the verdict column** |
| §22.32 | §22.23's pre-staged [I8] diff is SUPERSEDED — apply this form |
| §22.33 / §22.35 / §22.36 | docs/152 review pass; the mirrored-oracle catch; the four-gate C2 audit |
| §22.34 | the (d.4) measurement harness |
| §22.37 | what REPO.md rule 14 does **not** cover (fixture-reachability) |
| §22.38 | blocker E re-checked on the real artifact + the diag-compile gate |
| §22.39 | fold-prep: new file, one call site, the width chain |
| §22.41 | blocker G |
| §22.42–§22.45 | the gate runs, the C2 repair constraint, the narrowing |
| **§22.46** | **gate 1 resolved — the oracle bug, and why three matching aggregates were blind to it** |

---

## 9. What I would NOT re-derive

Six causes of gate 1's red were eliminated with evidence, and one of them (the
exclusion predicate) cost several cycles of reading correct code. The list is in
§22.45's table and the answer is in §22.46. If you are looking at a red gate-1-style
number again, start from §22.46's dependency table, not from the kernel.

Also settled, do not re-open without new evidence:
- the container's tensor-class dtype (BF16, §22.29)
- the rope convention (split-half rot=128, §22.24)
- the conv/edge storage dtype (BF16 + FP32 accumulators, §22.26)
- the drafter block width (always 8 columns/lane, §22.30)
- the fold shape (new file + one call, §22.39, coordinator-blessed)

---

## 10. (d.4) / (d.5), so you don't rediscover the rules

- **(d.4)** harness is §22.34: 32 prompts / 3 families with **n stated before any
  number exists**, a non-vacuity mutation per metric, and **the requant ladder and the
  encoding decision must NOT share a run** (the ladder changes the weights; the
  decision is made from the ladder's curve). Blocker G's codebook pin is a precondition
  of running the ladder at all (§22.41).
- **(d.5)** must **REPLACE** §22.19's delta-derived ladder with measured-at-load
  numbers, and read the **per-rank `materialized: NNNN MB` line** (tp_load.cpp:152),
  never a binder-plan delta. That is the specific error §22.19 was written to correct.
- **L=2 is the only admissible concurrency point** on today's geometry (§22.15(b)),
  and closing blocker B does not change that.

## 11. Open questions I'd ask the coordinator first

1. **Blocker G** — is the (i) first-cut (pin codebooks to `w8g32_f16s`, +64.4 MiB)
   ratified, or does A1 want (ii) `Q4G64` in `ops::embedding`? It is a manifest-field
   decision with a throw on the other side of it, and (d.4) flips manifest fields by
   design.
2. **The 2b-iii wiring commit** — does the coordinator want the call site + gate
   removal + a device run as one commit (my recommendation, and the only safe shape),
   and is a grant available for the run?
3. **§22.31 item 4** — the [I5] device one-hot is a formality that has never run.
   Worth folding into the same grant as (2) rather than a separate window.
4. **gemini's two queued notices** (§7) — should they block his stage-A/C authoring,
   or is a heads-up enough?

---

*Written at session end, then reviewed twice. Pass 1 caught four stale claims (§1's
note). Pass 2 caught three things that were true and recorded in docs/151 but absent
here — the tap-dump preservation rule, the k<=5 ceiling, and the one-number frontier
invariant — which is the failure mode of a handoff written from what I was working on
rather than from what the next session will need. Re-run §1's commands and §2's greps
before trusting them; §6.11 is why.*
