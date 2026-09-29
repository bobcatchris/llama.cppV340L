# WO-07 — Gfx900 PERMANENCE: 12 unwhitelisted shuffle sites + the sub-group invariant audit (agent 5) — issued 2026-09-12 ~19:5xZ, coordinator C441

**Worktree:** `/home/chris/worktrees/amd-wo-gfx900-perm`, branch `amd/wo-gfx900-perm`, already created at `amd/t3-wip` tip b91d3bf0 (pull `amd/main` in first — cite `amd/main` by name, it moves; your base carries the merged shim + donor adoption + G-AMD-15 evidence).
**Mission (regression defense the user named you for):** make the line's cross-lane-reduce correctness PERMANENT, so no future change can re-break what G-AMD-13 cost a full night to fix. The known open surface: 12 of the 27 in-build shuffle sites sit in files NOT HIP-whitelisted (agent3's addendum named 2 of them: `cudaHostGetDevicePointer`, `cudaDevAttrMultiProcessorCount`; 28 `warp_max<N<32>`/`warp_sum<N<32>` sub-group sites exist with 0 in-build, the agent2 v340l/03 invariant that must not rot), and the line's own ISA gates key on isolated probes — which let the promote-carried `__CUDACC__` guard AND the syncwarp host-break pass a green gate while the real build was dead.

## 1. Goal & gates (goal → gate → evidence)
G1. **Site census, measured not inferred:** for every unwhitelisted file with shuffle/wavefront semantics (the 12 + the 28 sub-group sites + any your grep finds): HIP-compilable TODAY or first-real-error pasted, `-x hip --offload-arch=gfx900 -fsyntax-only`, PASS LABELED which compile pass (host/device — defect text differs, counts must not be pass-confounded). Zero-GPU.
G2. **Tripwire per reduce-site class:** extend agent2's ISA pattern (docs/amd/v340l/03 + `results/amd/` receipts scripts — reuse, don't reinvent) so EACH file family in G1 gets: isolated-probe compile + ds_bpermute count + narrowed-EXEC divergence count + negative control that FIRES on the pre-fix guard style (their 10/11 discipline: a green without a demonstrated-red control is decoration).
G3. **Real-build cell wired where it's missing** — see §5: the durable one is gemini's lane (their build-cell merge-gate already in flight with the three-mode lesson); yours is the SPEC + a runnable script they can land (mode list: pure-host .cpp TU, -x hip both passes, full cmake build; negative: the pre-fix shim style fails it).

## 2. Test entry points
- CPU-only gates; `bash tools/ops/run_ci_amd.sh --zero-gpu` stays green on your tree (check-(a) registered exceptions expected-RED with ticket, everything else hard-fail).
- Every claim ships a receipt script IN `tools/v340l/` (agent2's syncwarp_isa_receipts.sh is the template) — rerunnable by a cold reader, output pasted in results/amd/, mtimes/claims inadmissible.
- Device: NONE without written coordinator grant; your G2 negative-controls are compile/ISA cells, they do not need cards.

## 3. Design decisions (FINAL — do not re-litigate)
- The shim is CUDA-semantics-exact and gate-grade (width forwarded, no early-return, cndmask selects — 496/1520/inv 0.088388 hardware-verified at G-AMD-13); this WO makes coverage permanent, it does NOT revisit the fix.
- Gates test ISA behavior, not call-site semantics (the multi-step butterfly lesson: two single-step enumerations cleared the broken shim); your G2 cells inherit that or they're theater.
- Whole-kernel divergence counts are caller-side-safe where EXEC regions are warp-uniform (rmsnorm 114/114, argmax 20/20 unchanged post-fix — agent2's documented reasoning); gates on ISOLATED reduces, CFG-grade whole-file checks are gemini's authorship if/when wanted.
- REJECTED: re-whitelisting any file without G1 evidence first; editing tools/ops/gate* or tests/** (gemini's lane, §7.x); a from-scratch probe suite (extend the receipts pattern; §F applies to kernels, and you write gates not kernels).

## 4. File ownership (exclusive; do not cross)
YOURS: `tools/v340l/*_receipts.sh`, `tools/v340l/shfl_*`, new probe/cell scripts you add there; `docs/amd/v340l/07*` (your audit doc — 07_ prefix reserved for you).
NEVER: `src/common/hip_shim/*` (agent2 — STOP-ask with a receipt, don't edit), `src/ops/launcher/*` + `src/core/multi_gpu/*` + `src/targets/**` (agent4's bring-up surface, and agent3's launcher T3 edits — coordinate via me before touching anything that isn't a probe), `tools/ops/gate*` + `run_ci_amd.sh` + `tests/**` (gemini, §7.x; your spec rides to them through me), the canonical pair (`src/runtime/tp2/tp_engine.cpp`, `tp2_budget.h` — zero-diff law).
GPU: none (stamps via coordinator only). df -h / before any >5 GB build; / sits ~38–44 G free today, two-builds-max, stagger.

## 5. Known-state inputs (cite, do not re-derive)
- Merge-gate hole + three-mode lesson (device-only probes missed the host break; agent2's confession + their R5 receipts script) — your G3 spec builds ON that, don't duplicate their cell.
- v340l/03: masks-ignored verdict, 28 sub-group sites, 12/27 unwhitelisted (their tables + the read-the-matched-line law: receipts agreeing ≠ evidence, disagreeing = a gift).
- v340l/04/05/06: probe port, TB2 landing map (P2 passed, q3 fully absorbed), artifact discipline.
- Donor adoption is on your base (RULING-1 lanes, full-family mma guard, funcattr durable shim merged) — your G1 compiles run against THAT, not a pre-fix tree.
- gemini: un-ACKed since 15:00Z, gate branch tip at 68358e08, 3 items still open (check-(a) .h-pair, D3 fingerprint, q3_ci) — surface-to-user is queued at the next wake if still dark; your G3 spec goes to them via me regardless, so their silence doesn't stall G1/G2.

## 6. Execution order (commit + test each step)
S1 pull amd/main into your branch, run zero-GPU gate, record expected-RED baseline. S2 the 12-site + 28 sub-group G1 table (one file per row: class, pass-labeled compile result, first error text if any). S3 per-class G2 probes with negative controls, receipts in tools/v340l/, outputs in results/amd/. S4 G3 spec doc (the three-mode cell + negatives, written FOR gemini's landing) → to me. S5 report gap verdict: for every G1 failure, classify (enablement gap / new defect / already-fixed) and STOP-ask anything needing a product edit — routing to the owning lane, you audit not author.

## 7. Definition of done
G1 table complete and honest (a file that compiles says so; one that errors pastes the error), G2 receipt scripts committed + each with a demonstrated-negative-control row, G3 spec delivered to coordinator for gemini's lane, docs/amd/v340l/07 audit doc naming the residual permanence gaps, zero-GPU CI green on your tree, handoff section so a cold reader can re-run everything from one command list. Report blockers by name — both lanes you neighbor (agent3's decode block, agent4's serve link) are the sprint's live critical path; your work is what makes the fixes outlive the sprint.
