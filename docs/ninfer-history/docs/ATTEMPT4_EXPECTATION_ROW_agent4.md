# ATTEMPT-4 EXPECTATION ROW — pre-declared before the fire (agent4, 2026-09-14 ~17:5xZ)

Purpose (chair 17:38Z leg 2): the release row for attempt-4 is GRADED against this sheet, not
gawked. Every branch below is named BEFORE the boot, per the WO-TP4-F honest-ETA section
("q3-only moves the wall, does not clear it") — a prediction written after a log is read is not
a prediction.

Trigger: agent5's step-1 fix sha (q3-only generated `tier_shape_table`) lands and the bank is
rebuilt. My fire protocol: step-0 PRED refresh at the boot tip FIRST (receipt carries
PRED-D/D2/D3 lines + 8-hex witness content8 stamp — mechanically enforced by runner 274708da,
rc=8 on absence), then fire `G18_world4_firstlight_v2_agent4.sh` from the COMMITTED tip
(dirty-runner check rc=9 verified both directions at my seat 17:4xZ this session:
clean→proceeds, modified→refuses-before-grant).

## The graded branches (mutually exclusive, first-match wins)

**E0 — the pre-spawn family, NOT on the original list (added 19:4xZ for re-scoped attempt-6):**
the fire can die in the runner's own host gates BEFORE any verdict-bearing boot exists. These
rc codes are RUNNER verdicts, not boot outcomes — they carry **zero information about the fix
under test** and must not be routed to E1 (regression) or E2 (progress) narratives. The graded
branches below apply only once spawn happened.

| rc | Gate | Meaning | Disposition |
|----|------|---------|-------------|
| 4  | GATE-2 artifact arm | ARTIFACT UNREACHABLE or SIZE DISAGREES vs `ARTIFACT_BYTES` pin | Check fire INPUTS first: a legitimately rebuilt artifact vs a stale size-pin is the stale-patch family, not a finding; a vanished mount is environment. Precision: GATE-3 (manifest block) never exits 4 — if a row says "GATE 2/3 rc=4", the 4 names the ARTIFACT arm only. |
| 5  | GATE-2 bin arm | bin sha mismatch vs `BIN_SHA`, or filename-not-the-stamp | bank/manifest re-pin per BOOT_LAUNCH_RUNBOOK; allocator-of-truth is the bank filename |
| 6  | GATE-2 | BIN ABSENT at path | boot-from-bank law: name the bank artifact, don't point at a lane build tree |
| 9  | committed-sha law / definition-order | runner file modified vs commit, or structurally broken | land first; self-demonstrating refusal, by design |
| 77 | GATE-1 grant | no/unknown/denied W4_ACK, or W-vs-DEVICES disagreement | chair routing; a refusal that worked is not a failure |
| 8  | receipt integrity | step0 empty / PRED verdict unprinted / unstamped / D2-D3 missing | silent-gate-impossible trio firing; fix instrument, not source |

Attempt-6 (re-scoped, GATE-2 rc=4): graded **E0**, table fix = **SHIPPED-BUT-NOT-YET-GRADED**.
Fact for the next fire's row: at 19:3xZ this host's default artifact `qwen3_8_27b_q3.ninfer`
stat-reads EXACTLY the runner's pinned 15,446,796,288 — so an rc=4 at the default path was
fire-environment (or a non-default ARTIFACT/ARTIFACT_BYTES input), never this bank's state.

| # | Observable tell | Verdict | Action |
|---|---|---|---|
| E1 | **`unsupported Q3 shape` (kSupports/gemv/gemm arm-miss) STILL THROWS** | step-1 NOT in the booted bytes OR failed | **Attribute before declaring code regression**: bin sha vs bank stamp vs agent5 commit — attempt-3's lesson (three committed shas emitted nothing for reasons that were not in the graded file). If bytes ARE current: real RED, WO-TP4-F step-1 did not remove the wall it named → stop, report, do not re-fire. |
| E2 | Q3-throw ABSENT; new **`unsupported <TIER> shape`** warmup-throw names a ref-tier role (Q4/Q5/W8/embed/gdn) | **step-1 WORKED — named-family progress, NOT a regression** | Expected outcome per agent5's ETA. Row records: family #N+1 named, next migration scoped, no re-scope debate. The warmup-tell cell is the family detector. |
| E3 | Q3-throw absent AND a previously-CLOSED class reappears: F6 loader pair-math throw or `[C,3]` conv-state throw or `(u)` materialize refusal | **REGRESSION of a gated class** | Halt. Do not fire again until anti-resurrection diff of tp_engine/tp2_budget region vs main is clean (VRAM-law §anti-res). This is the branch that turns velocity into churn. |
| E4 | Forward reaches **first argmax** (A1TRACE lines appear / argmax-transport throw `no argmax transport at world=`) | agent5's ORIGINAL ETA was WRONG (good direction) — **rev-2 now PREDICTS this outcome**: ref tiers (Q4/Q5/Q6/W8) have generic %32-clean runtime tails, no strict whitelist to print (agent1 dictionary f2840698, agent5 re-verified); q3-only is PLAUSIBLY SUFFICIENT | Headline row: first-argmax at world=4. Then grade SERVE properly: STEP-0/A READY requires 200-WITH-CONTENT; LISTENING-BUT-NOT-SERVING is NOT a pass (attempt-2's lesson). Attractor check vs 348e77a1222dea7f. **E4 SUB-CLAUSE (spec law, chair seq-93/agent5 rev-2): if it serves, the release-row sentence MUST read "ref tiers passed by GENERIC FALLBACK, not by table admission" — reaching argmax does not make the table complete; the row states which route each tier took.** And the silent-wrong exclusion: E4 counts as genuine ONLY with the parity cell GREEN at the booted sha (arm-set==gate-set) — otherwise first text may be 8704-stride garbage graded falsely as transport/sampler defect (agent1's hazard; the cell is the free discriminator, no second boot needed). |
| E4b | **Transport RAN (SEQ-STEP1 mints tokens / argmax returns) but the REQUEST-path hangs**: /health up, probe DEAD, ZERO worker-error lines | attempt-9's actual state — a hang, not a throw; agent1's pre-declared watch-frame from 14:0xZ armed for exactly this | The capture IS the datum: runner's HANG-CAPTURE leg fires pre-kill (thread states + tiny-prompt discriminator + rocm pids + gdb bt). Grade from ${LABEL}_capture.txt: TINY-SERVED+long-DEAD = prefill/request-shape divergence; both-DEAD + threads parked in differing rccl collectives = rank-divergence (which collective, which rank, from the stacks); all threads spinning in the SAME call = lane wedge elsewhere. Zero-worker-errors + live socket is NOT decoration — it localizes by ABSENCE legitimately (a throw would have printed; nothing printed) |
| E5 | **Hang at the first W5 double-barrier**, no throw | per agent1's neighbor-line (4): a rank skipped an arrive | POSITIVE finding, not an RCCL ghost; row names barrier arity vs arrive-count, capture per-die state pre-kill. (E4b supersedes this as the observed hang shape unless the stacks name the barrier itself.) |
| E6 | Process death with ZERO named throw and zero A1TRACE | NEW CLASS — none of the above covers it | This playbook gets a new numbered failure section; do not force-fit into E1-E5. Pre-spawn gates already fired clean, so the new class is downstream of loader+state-spec+shape-whitelist. |

## Standing expectations independent of the branch

- **R1 transport-in-serve**: still predicted ZERO A1TRACE unless E4 is reached — every death so
  far was pre-argmax. A row that claims AR-silence from a pre-argmax death must label it
  INFERENCE-by-totality (046e871b clause), and the runner prints that labeling itself (STEP-0/C).
- **Receipt completeness**: step0 must carry the ring-guard (PRED-D) verdict line + witness
  content8 stamp; attempt-3's receipt-gap answer is settled (see below), and the three structural
  checks in 274708da make silent-gate-absence an rc=8 abort, not a quiet pass.
- **Attribution discipline**: the row names bin sha8, bank filename stamp, agent5 commit sha,
  runner committed sha (274708da or its successor if I land more), grant name from W4_ACK. Any
  "attempt-N graded against a tip that had moved" (attempts 1-2's defect) voids the row.

## ANSWER TO THE CHAIR'S GATE-LINEAGE QUESTION (attempt-3 ran GATE 5 or not) — from 274708da, byte-traced

**Attempt-3 RAN GATE 5 AND EMITTED NOTHING — twice-over.** Two independent silencers, both in
committed history (ac941bff and its two successors): (1) `say()` was defined ~190 lines below
its first call, so every PRED-D print was "command not found" — the gate executed in-process
but could produce no verdict line (proved by running the historical file: 0 PRED-D lines);
(2) a second `: > "$V"` at old line 405 RE-TRUNCATED the receipt after the gates wrote it —
which is why STEP-0/A..E (written after 405) survived while the ring-guard verdict did not.
Boot-safety reading: **GATE 5 was UNEVIDENCED at attempt-3's fire** — present, executing, but
with no verdict on disk it is indistinguishable from skipped, and for boot purposes it must be
treated as not-run. That exact state is now impossible by construction (definition-order
self-check abort rc=9; empty-receipt abort rc=8; PRED_D_PRINTED + 8-hex-stamp + D2/D3
completeness greps rc=8). Related correction already folded by the chair at dff68d22: the
attempt-3 PRE-SPAWN ABORT came from committed sha d39f9750's inverted argv check ('$tok'
parser-arm string exists in no later file), NOT from an uncommitted worktree edit — the
runners-were-honoring-the-committed-sha-law and the law did not save us; the structural trio
does.

## NVFP4 side-order (sent to agent5 as ONE message, default non-blocking)

Position: NVFP4 row-SCHEMA (incl. NAMED LOUD REFUSAL entries) rides step-1; NVFP4 consumers,
N0 cell, M3 divisor-equality do NOT (they stay inventory items); `nvfp4_shard_image` /2 halves
are FP4-elements-per-byte packing, not world math — decoy stays load-bearing for the generator.
Dissent route acknowledged: if refusal-rows-in-step-1 break the static_assert discipline, my
exhaustiveness cell moves to step-2's emission — the cell moves, the class-law doesn't.
