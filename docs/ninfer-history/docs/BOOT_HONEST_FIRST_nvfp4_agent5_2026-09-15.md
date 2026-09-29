# THE HONEST-FIRST BOOT — minted into the queue (agent5, plan owner; chair 09:50Z item 4)

Cheapest meaningful boot on the board, and it needs NO device code to be meaningful: it proves the
admission layer is the thing refusing, before any SIMT path exists. Two boots, one grep, both
directions. Zero-card legs run first; the boot legs need the usual written grant (this document mints
the SEQUENCE, not permission to run it).

## Vocabulary, per the chair's ruling while reviewing
Capability gates are **G1-G6** (warm-breath / determinism-bar / corpus-diversity / long-generation /
cross-format / sustained). The context ladder stays **H1-H6** (1k→200k) and runs *under* G4/G6.
This boot is a **G-row-adjacent admission leg**, not a rung: it produces no token stream, so it can
never be quoted as an H-rung result.

## BOOT-0 (pre-device, zero card, grant-free) — the refusal must be seen BEFORE any kernel exists
1. `census-vs-acceptsets gate` cold at the deploy tip (`tools/v340l/nvfp4/nvfp4_census_vs_acceptset_gate.py`,
   v3.1 @ b0309248 lineage; today's main blob 4fba2cc4) → expect `VERDICT=0`, warnings exactly the two
   ruled C6 rows. This is the "admission green" half.
2. `nvfp4_dispatch_table` host test (CUDA-free, `tests/nvfp4_dispatch_table_test.cpp`) → drives
   `resolve_nvfp4_route` across the matrix incl. the A16Only expectation; expect PASS. This is the
   "which runtime governs the guard" witness — the guard is host-side, and the only CUDA-family-bound
   NVFP4 surface is the TMA/W4A4 sm120 archive (`src/CMakeLists.txt:72-80`), so a row citing a refusal
   must name its path.
3. Static refusal audit (no boot): `grep -n "unsupported shape" src/ops/linear/nvfp4/nvfp4_dispatch.cpp
   src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp` → the pinned texts at dispatch :19-21 and :60-62
   are THROW sites that fire before any launch. Text is the contract: the boot rows below grep it
   verbatim, `nvfp4 linear: unsupported shape`.

## BOOT-A (before the device port lands) — the CLEAN refusal, with both numbers
Boot the serve path on the NVFP4 artifact with the TP gate still closed and expect, in this order:
1. `[C-ADMISSION]`/load lines green (artifact identity + geometry accepted — that is the §8g A-row set
   applied at boot), then
2. `nvfp4 linear: unsupported shape` (or, if the gate closes earlier, the `NINFER_ALLOW_NVFP4_TP2`
   refusal at `tp2_backend.cpp:1333-1345` — **and note the world-name defect there: the env says TP2
   while the artifact rides world=4; the row must name which guard fired, not which env was set**), then
3. NO device-side symptom: no `tp2::TPEngine` ctor line past the guard, no per-rank H2D of NVFP4
   weights, no collective join. The point of the boot is that the refusal is *host-side and first*.
Failure shape to name if seen: a refusal that arrives AFTER weight placement, or an empty response with
no throw text, or a throw naming the wrong world — each is a different defect and the row must say which.

## BOOT-B (after the G2/G3/G4 SIMT ports land) — the INVERSE
Same argv, same bank, same prompts. Now `nvfp4 linear: unsupported shape` must appear **zero** times and
the serve must produce tokens; the six-conjunct confirmation table then runs (raw RULING-4 digest
grammar, `sha256(reasoning_content + content)` over the RAW JSON fields — never the display-wrapped
`<<REASONING:…>>` form, which is the wrapper bug's home and is why digest rows must state their field).

## Two-direction discipline, because one boot is a story and two are a test
BOOT-A and BOOT-B are the SAME assertion inverted, so a runner that can print both has actually measured
the port landing; a runner that only prints B has measured nothing (an always-green serve looks identical).
Both rows carry: bin sha (bank path = filename = stamp, per the boot-from-bank law), tree sha, artifact
sha `eaf8ad12…56d2`, GPU grant id, and the refusal-text grep count (A: ≥1, B: 0).

## What this boot does NOT claim
It is not first-light (no token correctness claim), not a TP4 serve-green (that needs the six conjuncts
plus sustained legs), and not a codec-artifact statement: the mounted artifact is the QAT one, and the
real-decode artifact row waits on the convert-side step with `expected-sha-at-landing = TBD at
production` (naming a sha before bytes exist is the phantom-pin class).

## Owner/queue placement
Sequence owner: chair (window). Legs: BOOT-0 this desk any time (zero card); BOOT-A this desk on a
granted window; G2/G3/G4 device SIMT ports are agent4's, queued behind his decode/MTP calls per the
09:50Z order; BOOT-B runs the first time a post-port boot is granted, and I'll pair it with BOOT-A
retroactively if the windows differ — a pair split across days is still a pair as long as both rows
name the same bin and the same grep.

## ADDENDUM — the H-rung vocabulary is not satisfied by the r28 ladder row (verified at bytes, zero boot)
The chair's rename says capability gates are G1-G6 and the ladder H1-H6 is the CONTEXT axis (1k→200k).
Measured: `G18r28_hladder_g1000_1.json` and `G18r28_hladder_g10000_1.json` carry `prompt_tokens: 54`
both, `completion_tokens: 58`, `reason_len 185`, `content_len 52`, and the same raw-field digest
`e764922b0dfe1a7f`. The runner says why: `G18r28_hladder_agent4.sh:3-4` — *"GEN per rung = the plan's
'contexts' = generation DEPTH — short prompt, depth"* — with `max_tokens:$G` at `:68`.
So the r28 row's "rungs" vary the GENERATION BUDGET, not the context length. It is legitimate evidence
for **G2 (determinism-bar)** at two depths — legs agreeing on the same prompt is exactly a determinism
pair, and it arrived free — and it is NOT evidence for any H-rung, because the prompt never grew past
54 tokens. Consequence for the ACCEPT table: citing H2-10k (the user-named first number) from this row
would be the vocabulary error the rename exists to prevent, and the H-ladder still needs a payload whose
PROMPT length is the axis. Flagged to the chair and to agent4 in the same beat, at the source lines
rather than by inference — the identical digests across "rungs" were the tell, and the honest reading of
a surprise is to go find out why, not to quote it more carefully.
