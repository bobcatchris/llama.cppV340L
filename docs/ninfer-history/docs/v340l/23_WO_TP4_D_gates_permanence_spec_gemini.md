# WO-TP4-D: Gates and Permanence Specification for the TP4 Surface

**Author**: Gemini (`b4791a54-9948-4610-9ee7-ab96a077c0cb`)  
**Parent Work Order**: `docs/amd/WO_TP4_all_lanes.md` (§WO-TP4-D)  
**Parent Delta Map**: `docs/amd/v340l/17_tp4_artifact_shape_answer.md`, `TP4_DELTA_MAP_v1.md`  
**Related Audit Receipts**: `docs/amd/v340l/21_shim_verifier_poison_check_is_textual.md` (Agent 2), `TRIPWIRE_GATE_CLASS_SPEC_agent5.md` (Agent 5)  
**Baseline**: `origin/amd/main` @ `a38976f9`  
**Execution Posture**: Zero GPU execution, zero product code edits in `src/`, CPU-only gate verifications, three-state exit discipline.

---

## 1. Executive Summary & Lane Mandate

Per Coordinator directive `COORDINATOR-SUCCESSOR.md` and `WO_TP4_all_lanes.md`, `WO-TP4-D` is the **exclusive lane** of Gemini for authoring, maintaining, and defending the gate suites, invariant tripwires, and permanence verifiers across the incoming TP4 (world=4) surface.

TP4 does not alter model artifacts or export formats; it is a runtime parameterization of engine, transport, and geometry across four physical GPUs. All code landing across WO-TP4-A (engine parameterization), WO-TP4-B (transport), WO-TP4-C (geometry), and WO-TP4-E (budget/VRAM) must merge cleanly through the gate family established here.

---

## 2. TP4 Exception Taxonomy & Registration Architecture

### 2.1 Four-Category Exception Model
All files introduced or touched for TP4 must register under the established `(i)–(iv)` taxonomy in `tools/ops/gate_pg1_whitelist.sh` and `tools/ops/verify_registered_exception.py` **in the exact commit that introduces them**:

1. **Category (i) — Shim Headers (`src/common/hip_shim/*`)**:
   - Must satisfy Bare Compilation Syntax Check under `hipcc -fsyntax-only`.
   - Must satisfy Keyword & Macro Poison Refusal (position-independent regex).
   - Must satisfy Candidate B Semantic Post-Preprocessor Witness (`verify_shim_consumer_witness`).
   - Must satisfy Logical Warp Width Contract (`width=32` constant lowering).
   - Must satisfy Object-Derived Tree Validation (refusing borrowed includes with exit code 2 `INSTRUMENT-ERROR`).

2. **Category (ii) — Documentation & Audit Specifications (`docs/amd/*`)**:
   - Registered under `verify_documentation_file()`.
   - Verified clean of executable macro injections, unshielded macro definitions, or toxic shell code snippets outside fenced markdown blocks.

3. **Category (iii) — Two-Hop & Seam-Adjacent Sources**:
   - Bounded-poll and status-word modifications must maintain `__HIP__` isolation.
   - P2P and IPC handles must observe strict nullability and lifetime management.

4. **Category (iv) — Data-Row & Roster Invariants**:
   - Whitelist parity (name-sets equal via `SOURCES_FILE`).
   - Extension vocabulary roster completeness (`.inc` actively tracked).

### 2.2 Near-Capacity Synthetic 4-Rank Geometry Cell
Following the WO-VRAM-1 precedent (where constant-powered relief was abolished in favor of measured-basis allocation), TP4 geometry validation must:
- Implement a synthetic 4-rank geometry test (`test_tp4_geometry_shapes.py` or equivalent host test) that **derives shapes dynamically** (`full / 4`) and verifies divisibility of all QKVZ ranges, NS16/NS32 panel alignments, and head counts ($48 / 4 = 12$).
- **No Constant Refusal**: The check must compute exact memory requirements based on measured tensor descriptors. Any failure to evaluate must exit with `code 2` (`INSTRUMENT-ERROR`), never reporting a vacuous green pass.

### 2.3 WO-TP4-A Co-Land Exception Registration Rows & Verification Verdict
Candidate A-series commits (`a34d378b`, `c4b30bf3`, `e611c398` on `origin/amd/wo-p3-serve`) reviewed against `origin/amd/main` (@ `f18ba195`):

| File | Category | Diff (+/-) | Gate Status | Evidence Summary & Contract Compliance |
| :--- | :--- | :--- | :--- | :--- |
| `src/targets/qwen3_6_27b/impl/load/tp_load.cpp` | (iii) Preflight & Layout | +127 / -45 | **BLESSED** | Single declared part table `tp_part_sizes` shared by `multi_ranges` & `tp_local_shape`. Divisibility enforced by `require_parts_divisible`. Table sum == `full_rows` check rejects artifact/table mismatches. Dynamic derivation (`full/world`), zero static literals. Clean of banned constants. |
| `src/runtime/tp2/tp_engine.cpp` | (iii) Preflight & Engine | +59 / -37 | **BLESSED** | Vectorization over `b_opts.devices` replaces `dev0`/`dev1` fixed pair. Pre-device capability refusal (`tp_world != 2`). Placement shards by `tp_world`. Live free queries iterate `b_opts.devices` computing `live_free_min`. Binding rank identified in refusal. Zero banned constants (`16310ULL`/`1200ULL`/`headroom_bytes`/`kGateReliefBytes`). |
| `src/runtime/tp2/tp2_backend.h` | (iii) Backend Interface | +8 / -10 | **BLESSED** | `dev0`/`dev1` fields deleted; unified `std::vector<int> devices{0, 1}`. Zero remaining consumers. Structurally prevents caller pair-truncation. |
| `src/runtime/tp2/tp2_backend.cpp` | (iii) Backend Runtime | +36 / -9 | **BLESSED** | `TpGroup` built from `options.devices`. Rank creation loop over `backend_world`. Pre-device capability guard (`backend_world != 2`). `d2_sidecar` bindings indexed from `ranks` vector. Zero banned patterns. |

#### Pre-Authored Registration Row Text (for co-landing in `gate_pg1_whitelist.sh` and `verify_registered_exception.py`):
```bash
  # WO-TP4-A1: Unified fused part table (tp_part_sizes), divisibility guards & manifest sum check
  "src/targets/qwen3_6_27b/impl/load/tp_load.cpp"
  # WO-TP4-A2: Device vector world resolution, world-generic live free min, binding rank telemetry & capability gate
  "src/runtime/tp2/tp_engine.cpp"
  # WO-TP4-A3: TpBackendOptions devices vector migration, dev0/dev1 alias retirement
  "src/runtime/tp2/tp2_backend.h"
  # WO-TP4-A3: TpBackend rank construction loop over backend_world vector, backend-side capability guard
  "src/runtime/tp2/tp2_backend.cpp"
```

---

## 3. Tripwire & Seam Permanence Extensions

### 3.1 Deleted-Overload Tripwire (Spelling-3 Invariant)
The T3 `ldmatrix` deleted-overload protection enforces that raw smem addresses cannot bypass type-safe handles.
- **Fleet Scope**: Check (l) in `gate_pg1_whitelist.sh` scans all translation units in `src/` to assert `VIOLATIONS=0` against the 152-site ground truth.
- **TP4 Transport Protection**: If WO-TP4-B adopts multi-peer mesh or tree-based AllReduce using shared-memory staging buffers, those buffers must strictly conform to type-safe pointer abstractions.
- The gate asserts that zero untyped `__shared__` pointer casts or raw `smem_addr` overloads appear in newly whitelisted TP4 paths.

### 3.2 TpGroup Collective API Routing Permanence (RED→GREEN Closure Law)
In accordance with the **RED→GREEN Closure Law** (`AGENTS.md @ 53c09d17`) and the GATE-2 finding (Agent 1 runbook, Chair-verified at `tp_group.cpp:297-302`):
- **Defect Class**: `allreduce_argmax` contained a silent null-guard (`if (impl_->one_shot_argmax) { ... }`) with no `else`, fallback implementation (e.g. RCCL), or loud throw. At `world=4`, `impl_->one_shot_argmax` is null, causing `out_token` to silently never be written while the function returns cleanly — the "green-that-did-not-run" failure class.
- **Permanent Host-Parseable Arm (`tools/ops/check_tpgroup_routing.py`)**:
  Statically parses all public collective APIs in `src/core/multi_gpu/tp_group.cpp` (`allreduce_*`, `allgather_*`), verifying that no collective method terminates on a silent null-guard without an unconditional execution path (e.g. RCCL fallback) or a loud throw.
- **Falsifier Suite (`--falsify`)**: 5 negative and positive test cases verifying deterministic detection of silent null-guards, certification of return-fallback patterns, certification of `else-throw` and subsequent `throw`, and three-state exit discipline (`rc=2` on missing files).
- **Pre-Fix RED Capture Receipt**:
  - **Ref**: `origin/amd/main @ 53c09d17`
  - **Command**: `python3 tools/ops/check_tpgroup_routing.py`
  - **Diagnostic Output**:
    ```
    ======================================================================
      RED CAPTURE: TpGroup Collective API Routing Defect Detected
      Target: src/core/multi_gpu/tp_group.cpp
    ======================================================================
      [Line 297] TpGroup::allreduce_argmax: Silent null-guard detected on condition '(impl_->one_shot_argmax)' with no else branch, fallback path, or loud throw
    ======================================================================
      STATUS: RED (pre-fix reproduction under RED->GREEN Closure Law)
    ======================================================================
    ```
  - **Exit Code**: `1` (RED).
- **Post-Fix Closure**: The arm is wired into `gate_pg1_whitelist.sh` Check (q). Upon Agent 4's co-landing of the A-4 fix (providing an unconditional fallback or loud throw), the check will transition deterministically from RED to GREEN.

---

## 4. Phase-Gate Parity Family (Item 7 Adjudication & Bar Formulation)

### 4.1 Ground-Truth Corpus State & Item 7 Adjudication
Per Coordinator audit, commit records (`bc122675`, `66a5fcb3`), and Agent 3's decisive Item 7 verdict row (`1d0ff3c6`, Chair STATE 13:5xZ):
1. **Measured Historical Corpus**: Across four process boots with identical model weights and configuration, three discrete output classes were recorded (`17f == 17g5` byte-exact, `17g4` discrete).
2. **Item 7 Canonical Finding (Verdict Row `1d0ff3c6`)**: Agent 3's intra-server 5× repeat test demonstrated **5/5 distinct text outputs** within a single server instance, binary, artifact, flags, and greedy prompt. Sampler window was exonerated for 4/5 forks by pre-declared attribution rules, while S1 monotonic epoch proof held intact (327/327 observed == expected). The live failure class is identified as upstream carry-in (uninitialized/recycled page read-before-write, notably `gdn_gating` slice-coverage :275/:307/:330 and `tp2_backend:768`).
3. **P1 Prerequisite Law Resolution**: Hard prerequisite P1 in `docs/amd/WO_TP4_all_lanes.md` is **SATISFIED**. Within-boot cross-request bit-identity is dead as an acceptance bar under the current uninitialized carry-in regime.

### 4.2 TP4 Parity Bar Formulation & Activation Status
With Item 7 adjudicated:
- **Active Parity Bar (WO-TP4-D.4 Core)**: Coherent-class bars apply unconditionally for TP4 bring-up and first-light (G-AMD-18):
  1. Zero-fault execution (no SIGSEGV, no KFD aborts, no hang, clean shutdown).
  2. Syntactic & token-sequence coherence under greedy decode.
  3. Strict rank-4 AllReduce trace coverage and shape census invariance ($8704 \times 5120$, $5120 \times 4352$, $3584 \times 5120$, 12 GDN heads/rank).
- **Bit-Exact Parity Bar Activation & D.4 Cross-Boot Leg (Chair RULING 4 @ 60713b0b / G-AMD-34 @ 26a21f6c)**:
  - **Within-Boot Repeat**: Promoted to permanent hard gate upon G-AMD-33 25/25 text-identical closure row (`db09c924`, merged `f29bafea`).
  - **Cross-Boot Text Identity**: Formally **ACTIVATED** under Chair RULING 4's explicit field and scope clause: *cross-boot text-identity at greedy same-prompt world=2* (concatenated `reasoning_content` + `content`, whole-JSON vacuity law cited). Measured across 50 served requests across two distinct server process boots (G18d trace-ON × 25 + G18e trace-OFF × 25): **50/50 text-identical** (`348e77a1222dea7f` × 50). This formally closes the honest open carry-in residual datum as a latency/publish-lag class, proving numerical bit-exact reproducibility across boots.
    - *Erratum (2026-09-14, annotate-never-delete per Chair RULING 4)*: The phrase "proving numerical bit-exact reproducibility across boots" is strictly scoped to **text-field identity** (`reasoning_content` + `content` sha16 `348e77a1222dea7f` × 50). Raw JSON differs at non-text metadata headers (`created:`, `usage:`), and no intermediate numerical surface (logits, sampling tensors) was measured cross-boot; Chair RULING 4's scope clause (*greedy text identity at world=2*) binds this finding verbatim.
    - *Count-Hygienics Note (2026-09-14, Chair / Agent 5 audit finding)*: The world-2 comparator retry count across dial states is **1,841** = SUM (G18d 1,010 + G18e 831). The two repeat-summary files employ distinct sha reporting formats (G18d verdict-line `"DISTINCT TEXTS"` vs G18e per-request `"rep N: text_sha16=<sha>"` lines); multi-bin analysis instruments must parse both formats to avoid undercounting (e.g. naive single-pattern greps reporting 25/50).
- **Cross-Topology Invariance**: Structural parity and numeric tolerance bounded by $\epsilon_{\text{bf16}}$ apply across alternate rank orderings or hardware device enumerations.

---

## 5. Addressing Agent 2's Residual: Semantic AST Width-32 Witness

### 5.1 The Authoring Mutation Gap
In `docs/amd/v340l/21_shim_verifier_poison_check_is_textual.md`, Agent 2 correctly observed that while macro shadowing (`#define __shfl*`) is caught by Gemini's `:112` static regex and Candidate B consumer witness, a mutation inside the C++ wrapper body itself:
```cpp
// Mutated wrapper body passing 16 or warpSize instead of 32:
template <class T> __device__ __forceinline__ T __shfl_down_sync(unsigned m, T v, unsigned d) {
    return __shfl_down_sync(m, v, d, 16); 
}
```
can evade textual regexes if rewritten across multiple lines or refactored into helper templates, while `hipcc -fsyntax-only` and `hipcc -E` remain blind (since `-E` only expands preprocessor macros, not C++ function templates).

### 5.2 The Clang AST / LLVM IR Consumer Witness
To provide 100% semantic verification without executing code on a GPU:
1. We construct a consumer translation unit that invokes all 4 default 3-argument wrappers:
   ```cpp
   __device__ void __gemini_default_wrapper_witness() {
       int v = 0;
       int r1 = __shfl_sync(0xffffffff, v, 0);
       int r2 = __shfl_down_sync(0xffffffff, v, 1);
       int r3 = __shfl_up_sync(0xffffffff, v, 1);
       int r4 = __shfl_xor_sync(0xffffffff, v, 1);
   }
   ```
2. We compile the consumer TU with `hipcc -fsyntax-only -Xclang -ast-dump` (or `-emit-llvm -S`).
3. We inspect the resulting Clang AST:
   - For every instantiation of `__shfl_*_sync`, the instantiated callee `CallExpr` must have an argument matching `IntegerLiteral ... 'int' 32`.
   - Any literal other than `32` (e.g. `16`, `64`, or a non-constant `warpSize` variable) triggers immediate refusal (`EXIT_FAIL` / `rc=1`).
4. **Negative Falsifier (Test 10)**: A synthetic header mutation altering `32` to `16` or `64` in the wrapper body is fed to `run_falsifier()`, asserting that Test 10 turns **RED**.

---

## 6. Verification Status & Roadmap

| Gate Component | Status | Falsifier Test | Verification Baseline |
| :--- | :--- | :--- | :--- |
| Static Whitelist & Anti-Resurrection | **ACTIVE** | Test 1, 2, 5 | `origin/amd/main` |
| Macro Shadowing Refusal (`__shfl*`) | **ACTIVE** | Test 6, 7 | Static regex |
| Preprocessor Witness (`-E`) | **ACTIVE** | Test 8 | Candidate B (`hipcc -E`) |
| Semantic AST Width-32 Witness | **ACTIVE (WO-TP4-D.1)** | Test 10 | Clang AST Dump |
| Synthetic 4-Rank Geometry Cell | **ACTIVE (WO-TP4-D.2)** | Test 11 / 12 | CPU Geometry Runner (`test_tp4_geometry_shapes.py`) |
| Host Runtime & Preflight Exception Pairs | **ACTIVE (WO-TP4-D.3)** | Test 11 | `verify_host_runtime_source` (VRAM Law & Banned Constant Guard) |
| Item 7 Parity Gate Family | **ACTIVE (Coherent/Trace & Cross-Boot Text) (WO-TP4-D.4)** | Test 11 / 12 | P1 SATISFIED; within-boot promoted; 50/50 cross-boot text-identical (`348e77a1222dea7f`) activated per Chair RULING 4 @ `60713b0b` |
| WO-TP4-A Real Files Co-Land Review | **BLESSED & MERGED (WO-TP4-A1..A4)** | `verify_registered_exception.py` | A1-A3 + A-4 (`81563732`) blessed; `tp_group.*`, `argmax_routing.h`, `rank_index.h` registered as HOST-RUNTIME; F-B/ring_props/routing tests (#158-#160, 179 total) registered |
| TpGroup Collective Routing Arm | **ACTIVE / GREEN CO-LANDED (WO-TP4-D.7)** | `check_tpgroup_routing.py` / `t3_gate2` | `ENFORCE_A4_ROUTING=1` active hard gate; pre-fix RED captured, A-4 GREEN certified; dual-parser parity GREEN/GREEN |
| Determinism G-CELL Suite Wiring | **ACTIVE / PROMOTED HARD GATE (WO-TP4-D.5)** | Test 1-4 (`check_determinism_cells.py --falsify`) | Promoted upon G-AMD-33 25/25 closure row `db09c924` (merged `f29bafea`); triple `1d0ff3c6` / `db09c924` / wiring commit; `--enforce-promoted` permanent gate in Check (r) |
| Census Attribution Totality | **ACTIVE (WO-TP4-D.6)** | Test 1-5 (`check_census_retry_attribution.py --falsify`) | Enforces total retry attribution across AR and argmax rings; pre-fix RED (`c3895bdb`), live GREEN, both-way falsifiers (Check s) |

### 6.1 Amendment Trail
- **2026-09-13 (Chair Ruling on §4.1/4.2)**: §4.1 corrected to cite measured 4-boot/3-class corpus (`17f == 17g5` byte-exact, `17g4` alone, `bc122675`/`66a5fcb3`); §4.2 marked explicitly as PRE-REGISTERED pending Agent 3's decisive item-7 verdict row; `17g6` clarified as an unexecuted planned duty from Agent 5's `#757` proposal. Hard prerequisite P1 law (`WO_TP4_all_lanes.md`) binding.
- **2026-09-13 (Chair STATE 13:5xZ Item 7 Adjudication)**: P1 Prerequisite Law satisfied via Agent 3 canonical verdict row `1d0ff3c6` (5/5 distinct outputs under single server/boot). Within-boot cross-request bit-identity declared dead as an acceptance bar under current uninitialized carry-in regime. Active parity bar defined as coherent/zero-fault/arms-traced; bit-exact parity activation gated on Agent 4 carry-in remediation.
- **2026-09-13 (Chair Dispatch 14:3xZ Co-Land Review Blessing & Carry-In Gate Refinement)**: §2.3 added delivering BLESSING on Agent 4's WO-TP4-A1..A3 commits (`a34d378b`, `c4b30bf3`, `e611c398` on `origin/amd/wo-p3-serve`), with pre-authored registration rows for all 4 modified `src/` files. §4.2 amended clarifying sampler exoneration for 4/5 forks and gating bit-exact parity specifically on request-4 carry-in residual disposition.
- **2026-09-13 (Chair Dispatch 15:3xZ §4.2 Activation Gating Re-Derivation)**: §4.2 amended per Agent 3's self-correction (`232d1797`, merged `4f1f8bd7`, chair-verified): the three KAR REJECTs are warmup-phase, exonerating the sampler window for 5/5 served forks. The bit-exact parity activation predicate is formally re-named to gate on 'warmup rank-step asymmetry + K-blind publish-tear instrument gap' (the two live suspects under zero-card analysis). Core gate content and within-boot/cross-boot bar definitions remain intact.
- **2026-09-13 (Chair Dispatch 16:1xZ GATE-2 TpGroup Routing Arm & RED Capture)**: Implemented permanent host static scan `tools/ops/check_tpgroup_routing.py` enforcing that every `TpGroup` collective API has an unconditional path or loud throw under the user RED→GREEN Closure Law (`AGENTS.md @ 53c09d17`). Pre-fix RED captured on `tp_group.cpp:297-302` (`allreduce_argmax` silent null-guard at world=4). Wired into `gate_pg1_whitelist.sh` Check (q) with 5/5 falsifier suite.
- **2026-09-13 (Agent 4 G-AMD-30a Discovery & status_transport.h Whitelist Registration)**: Registered `src/core/multi_gpu/status_transport.h` in `gate_pg1_whitelist.sh` Check (b). Provides single-definition volatile-RMW status helper to remedy PCIe/gfx900 device atomicOr host-mapped memory non-crossing defect.
- **2026-09-13 (Chair Dispatch 21:5xZ G-CELL Wiring GO & 10/11 Tracked Split Integration)**: Wired both determinism cells (`tools/smoke/diag/determinism_5x_cell.sh` and `tools/smoke/diag/determinism_inproc_repeat_cell.sh`) into standing suite via `tools/ops/check_determinism_cells.py` as `gate_pg1_whitelist.sh` Check (r). Enforces stamp gate refusal (`rc=77` before network connect), argument validation (`rc=2`), and the 10/11 exit-code split contract (`ASSERT=tracked` -> `rc=10` keeps CI green while item 7 is open; `rc=11` signals promotion upon 25/25 closure). Audits banked evidence `results/amd/p3/G18c_release_row.txt` (19/19 text-identical banked, carrier line not claimed at 19<25, tracking-pending).
- **2026-09-13 (Chair Merge f29bafea & G-AMD-33 25/25 Closure Promotion)**: Item 7 closed with carrier claimed (`db09c924` + evidence `64859c20`, merged `f29bafea`). 25/25 text-identical (`348e77a1222dea7f` x 25). Determinism in-proc repeat cell promoted from TRACKED-RED to permanent gate via 10/11 exit-code split contract. Check (r) in `gate_pg1_whitelist.sh` updated to `--enforce-promoted` (hard gate requiring 25/25 closure row). Full triple complete: RED `1d0ff3c6` / GREEN `db09c924` / WIRED promote commit.
- **2026-09-13 (Chair Ruling 4 & G-AMD-34 Census Merge 26a21f6c / 60713b0b)**: D.4 cross-boot parity bar activated under scope: *cross-boot text-identity at greedy same-prompt world=2* (concatenated text-sha `348e77a1222dea7f` x 50 across G18d and G18e). Proves degradation-not-dial. Check (s) wired into `gate_pg1_whitelist.sh` enforcing census reader retry attribution totality under RED→GREEN Closure Law (Agent 5 audit finding (a)). Full gate suite counts 19/19 checks. A-4 co-land scope expanded to require +2 `ctest -N` inventory receipt for F-B and ring pairing properties registration.
- **2026-09-14 (Chair Dispatch 23:2xZ Erratum on §4.2 Text-Field Scope)**: Annotated §4.2 in place per Chair RULING 4 scope clause (annotate-never-delete): clarified that the 50/50 cross-boot reproducibility finding is strictly text-field identity (`reasoning_content` + `content` sha16 `348e77a1222dea7f` x 50), with non-text JSON headers (`created:`, `usage:`) differing and intermediate numerical surfaces unmeasured cross-boot.
- **2026-09-14 (Chair / Agent 5 Landing Audit @ a1243eee & Check (s) Convention Fix)**: Applied `drafts/PG1_CHECK_S_sete_convention_fix.patch` updating Check (s) to the `if ! python3 ...` convention matching the rest of `gate_pg1_whitelist.sh` (preventing `set -euo pipefail` from masking the named FAIL line). Documented count-hygienics rules: 1,841 total retry comparator across dial states and multi-format parsing across G18d/G18e repeat summaries.
- **2026-09-14 (WO-TP4-A4 Co-Land Blessing & ENFORCE_A4_ROUTING=1 Hard Gate)**: Merged Agent 4's WO-TP4-A4 (`81563732`). Registered `src/core/multi_gpu/tp_group.cpp`, `tp_group.h`, `argmax_routing.h`, `rank_index.h` in `gate_pg1_whitelist.sh` and `verify_registered_exception.py` as HOST-RUNTIME exceptions. Registered `argmax_routing.h` and `rank_index.h` in Check (b) header additions whitelist. Flipped `ENFORCE_A4_ROUTING=1` to permanent hard gate. Verified dual-parser parity GREEN/GREEN (`check_tpgroup_routing.py` rc=0 + `t3_gate2_silent_nullpath_check.py` rc=0). Verified test registrations (#158 parity tag falsifier, #159 ring pairing props, #160 argmax routing host test; 179 total tests). Verified Agent 1 F1-F7 audit. PG-1 gate suite: 19/19 PASS.
- **2026-09-14 (Decision-Aware Leg-2 Anti-Resurrection Guard Triple)**: Implemented decision-aware regression classifier in `tools/ops/check_anti_resurrection.sh` to resolve the historical runbook §1b / Chair `ad069459` false-RED hazard (diff-shape removed-line count conflating capability guard narrowing with stale-region budget resurrection). Filter classifies removed lines: genuine regressions (stale patterns, decision-path math, or `tp2_budget.h` code alterations) fail loudly; non-decision canonical edits (e.g. capability guard relax `!= 2` -> `< 2`, error strings) are permitted. Authoring verified across the full triple: (1) RED: Banked predicted-RED transcript `results/amd/p3/G35_antires_leg2_predictedRED_83d1b00e.txt`; (2) GREEN: `check_anti_resurrection.sh --baseline 393ce73d` rc=0 (0 decision regressions, 4 benign guard edits recognized); (3) FALSIFIER: `--falsify` rc=1 with all 3 internal arms passing, and mutation discrimination verified via `tests/test_anti_resurrection_triple.py` (5/5 PASS, wired into Check (d)).

---
**Agent-ID**: `b4791a54-9948-4610-9ee7-ab96a077c0cb (Gemini)`

