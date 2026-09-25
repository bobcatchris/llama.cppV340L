# T3 HANDOFF DEBRIEF — read this first (agent3 → any agent, 2026-09-12 late)

Purpose: any agent picking up the T3/attention lane (or the decode-verification
block) hits the ground running in ~15 minutes. Everything below is verified
against the tree unless labeled otherwise. If a claim here conflicts with the
coordinator's doc (docs/amd/COORDINATOR.md), the coordinator wins.

## 1. One-paragraph state

The T3 goal (WO-05: the attention op family compiles and runs on gfx900 for the
plain bf16-KV route) is ~70% done on branch `amd/t3-wip`. DONE: donor SIMT
bf16 decode+prefill kernels adopted (590 LOC), upstream mma kernels preserved
as CUDA-only, the launchers route `__HIP__` → donor SIMT / CUDA → upstream mma
kernels, the D3-bridge helpers landed durably in math.cuh, CPU-oracle tables
generated, and the **prefill path is device-validated** (outputs track the fp64
oracle at bf16-quantization + fp32-order scale). OPEN (the only red): the
**decode path produces all-zero output** — narrowed to the donor decode
kernel's key-visibility loop resolving zero keys on my runner path; the next
step is a kernel-body read, then a runner q-shape fix, then a re-stamped
device window.

## 2. Branches and where the work lives

NOTE: `amd/t3-wip` is a BRANCH in the existing worktree
`/home/chris/worktrees/amd-wo-q3hip` (no separate worktree). To continue:
`git checkout amd/t3-wip` there. My lane branch `amd/wo-q3hip` holds the
GREEN WO-04 state if you need the pre-T3 baseline.

| Branch | State | Contents |
|---|---|---|
| `amd/wo-q3hip` | GREEN, pushed | WO-04 complete (q3/q2 HIP bring-up steps 0–4: guard patch `f5035616`, goldens `fae07b3c`, whitelist+build `8bf9a1ef`), TA1 census, TA2 adoption prep, both plan docs, oracle fix |
| `amd/t3-wip` | WIP, pushed, RED-by-design (decode path) | T3 adoption: donor kernels + launcher routing + math.cuh durable landing + prefill adoption `88eeba93` + oracle fix `e0c1dbb8` + G-AMD-15 evidence `210622d7` + decode diagnostics `54eaab52`, `0fd3e6a8`, `b91d3bf0`, `54423cc3`, oracle OOB fix + this debrief |
| `amd/main` | coordination tip | All of the above merged through `88eeba93`; G-AMD-14/15 grants + rulings in docs/amd/COORDINATOR.md (grep "GRANT G-AMD-14/15") |
| `/tmp/t3_device_runner_g15` | built binary | The linked device runner (rebuild recipe §4) |

Key docs on `amd/t3-wip`:
- `docs/amd/FORK_VELOCITY_PLAN.md` — the fork-adoption plan + §2b reuse census.
- `docs/amd/TP2_AMD_SUBSET_PLAN.md` — the full 2-agent+gemini TP2 plan.
- `results/amd/p1/g14_fault_mechanism.md` — the G-AMD-14 root cause + contract
  read + surviving suspect (the width-semantics analysis).
- `results/amd/t3_oracle*.table` + `t3_oracle.cpp` + `t3_oracle_run.log` — the
  fp64 oracle tables (sha-pinned in the run log) with the seq-31 bands.

## 3. Environment — exact working commands

```
# include roots (ALL of these; missing any = phantom errors)
INC="-Isrc -Iinclude -Isrc/common/hip_shim \
     -Isrc/targets/qwen3_6_27b/export -Isrc/targets/qwen3_6/export \
     -Isrc/targets/qwen3_6_35b_a3b/export -Ithird_party"
H="/opt/rocm-6.2.0/bin/hipcc -O2 -x hip --offload-arch=gfx900 -std=c++20 $INC"

# syntax census (host pass — LABEL IT; the device pass fails differently)
$H -fsyntax-only <file>

# standalone cell build+link (staged compile then link — the single-command
# hipcc link has a quirk; staged is the reliable recipe)
$H -c <cell>.cu -o <cell>.o
/opt/rocm-6.2.0/bin/hipcc -O2 --offload-arch=gfx900 <objs...> -o <bin> \
    -Lbuild-hip-amd/src -lninfer_hip_host

# real build gate
export PATH=/home/chris/opt/cmake/bin:$PATH   # cmake NOT on system PATH
cmake -S . -B build-hip-amd -DNINFER_BACKEND=hip \
      -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0 \
      -DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF
cmake --build build-hip-amd --target ninfer_hip_host -j4
# NOTE: the flag is NINFER_BACKEND (not NINFER_BUILD_BACKEND — WO-04 §4's name
# was wrong and silently defaulted, sending project() hunting for nvcc)

# CPU goldens (q3): std=c++20 required for the two tests including
# tests/ops/quantized_weight.h (std::span); the others are c++17-clean
g++ -std=c++20 -O2 -Isrc -Itests -o <t> tests/ops/<t>.cpp && /tmp/<t>
```

# oracle cell build+run (pure host, no HIP):
g++ -std=c++20 -O2 -o /tmp/t3_oracle results/amd/t3_oracle.cpp && /tmp/t3_oracle
# (writes results/amd/t3_oracle_*.table — regenerate = re-pin sha + disclose)

# CPU goldens (all must stay exit 0): test_q3_decode_golden,
test_q3_embed_gather_golden, quantized_weight_q3_test (Q3_A16_QUANT),
quant_recipe_test (QWEN38_RECIPE_CONSISTENCY). Baseline logs:
`results/amd/wo_q3hip_step2_cpu_goldens.log`.

## 4. GREEN — verified, with evidence

| Item | Evidence |
|---|---|
| WO-04 steps 0–4 (q3/q2 guard fix, goldens 4/4, whitelist, HIP build rc=0) | `f5035616`, `fae07b3c`, `8bf9a1ef`; merged to amd/main @ `039ab28b`+ |
| Donor kernel adoption: linear/gfx906 family (1,332 LOC) compiles on gfx900 via fallback arms | TA2, in t3-wip history; goldens byte-identical |
| math.cuh D3-bridge helpers (durable, ruled 17:0xZ) | `88eeba93`+; interim copies deleted same commit |
| Prefill path device-validated (outputs = fp64 oracle at bf16-quant + fp32-order scale, 0.001–0.017) | G-AMD-15, `210622d7`, `results/amd/p1/g15_device_run.log` |
| Oracle tables + loud-band comparator | `a50eecde`, results/amd/t3_oracle_*.table (sha in run log) |
| T3 census + defect list | `b8bd286c`, `results/amd/t3_census_attention.log` |
| G-AMD-14/15 evidence | `results/amd/p1/` (merged to amd/main by coordinator) |

## 5. OPEN — the decode-zeros anomaly (full evidence chain)

Symptom: `gqa_attention_cached_small_t_launch` runs (no fault after the fixes
below, splits=8 launched, partials WRITTEN) but partials are m=-inf/l=0 for
every head/split and out stays 0.0 → zero visible keys resolved.

Fixed already (runner-side, in-tree on t3-wip):
- block_table padded 2 → 64 entries (donor kernels bulk-load PageIds=64);
- cache planes sized to page capacity 128 slots (was 96 — the fill writes 104);
- decode query position 95 (< capacity 96 — was 96 = capacity → neutral-write);
- envelope [1, 96] (min 0 trips the split-capacity profile guard);
- 3-D q {kD, kQH, 1} (2-D q → ne[2]=0 → width=0 → kernel no-op).

STILL zero after all that. Ruled out by the C441 fresh-eyes read + my contract
read (results/amd/p1/g14_fault_mechanism.md): cache layout divergence
(append vs donor read use the same paged_kv_element_offset contract — DEAD),
causal-predicate deadlock (cleared), nullptr traps (cleared), allocation OOB in
my control (fixed above).

### §5 ADDENDUM 2026-09-12 ~21:10Z — G-AMD-16 series CLOSED (window 20:59:00Z); the §5 premise is STALE

Key visibility was ALREADY fixed before this session's work (uncommitted g15 log
showed healthy partials m≈18–19, l≈5–10 — §5's "STILL zero" no longer held).
The G-AMD-16 series then ran the full anomaly to ground under stamped windows:

- **Run3's "got=0" was a runner readback bug** (bf16 bit-cast into the LOW half of
  f32, missing `<<16`) — fixed + C441 comparator upgrade, commit 0bbce5b8.
- **G-AMD-16 (run 0bbce5b8, tree sha 55fceb38…2277): values are REAL but the buffer
  layout is wrong**: oracle element k lands at bf16 slot 2k+1 (byte 4k+2), even
  slots explicitly 0x0000 — a 4-byte-stride writer emitting f32 bit-patterns of
  bf16-rounded values, 1536 dwords (heads 0..5, cos 0.99975), hard canary wall at
  byte 6144. Repeatable across double-run with distinct canaries.
- **G-AMD-16c (rocprofv3, evidence results/amd/p1/g16c_trace/)**: dispatch is
  correct (grid threads 3072=12×256, wg 256, kernel-name+template pinned), zero
  D2D copies, and the mutation window between canary fill and readback contains
  EXACTLY ONE kernel — the reduce kernel. The fatbin's kernel (388-instruction
  disasm, object↔binary diff ZERO, both code objects arch-tagged gfx900+xnack)
  writes a single `global_store_short` at out+512h+2d with guards covering all
  3072 slots. Executing effect ≠ in-file ISA. Mystery stands.
- **G-AMD-16d (probe cells)**: P2 — a full 30-line replica of the reduce store
  path (same addressing, same conversion, same guard, same launch shape) — lands
  3072/3072 CLEAN SHORT stores. Box, runtime, store/conversion/addressing ALL
  INNOCENT in isolation. The anomaly needs the real kernel body (LDS reduce,
  partial reads, syncthreads) to appear.
- **Instrument-bug tally this session: 3** (readback `<<16` shift; canary
  byte-vs-slot detection; probe dump() observer-erase — the v1/v2 "dead launches"
  were the dump path re-filling the canary between launch and readback). LANE
  RULEBOOK LINE: an observer must never touch the thing it observes before
  observing it.

### §5 ADDENDUM 2 — G-AMD-18/19 results (2026-09-12 ~21:53Z): body innocent, launcher args innocent, anomaly alive post-merge

G-AMD-18 (dev2, closed): freshness rebuild sha 8217d149; PC re-derivation
old==new (store 0xD7D50 stable — layout-stable datum); **stride-4 PERSISTS
POST-MERGE** (A1 fresh bytes). rocgdb UNAVAILABLE on this box ('unable to
enable GPU debug', zero KFD PIDs — capability gap, gemini permanence material)
⇒ watchpoint/BP branch matrix unrunnable as designed.

G-AMD-19 (dev2, closed): bisection ladder v2/v3/v4 + **v5 = REAL reduce body
(header-included, exact instantiation) in clean context: ALL SLOTS CLEAN
SHORTS, stride-4 NEVER APPEARS in any replica**. v3/v4 value-noise = my own
guard-placement defects (disclosed; tid>=64 returns before reduce[tid] writes).
LD_PRELOAD hipLaunchKernel capture on the live runner: func ptr = correct
__device_stub, grid/block correct, ALL 11 ARGS == the runner's own printed
pointer map (capture run itself still mangles) ⇒ LAUNCHER ARG PLUMBING CORRECT.
Residue suspect set: (a) data-state fidelity (real partial-kernel outputs as
reduce inputs — v5b replay), (b) execution-context (single-object full-link
probe vs staged multi-object runner link) — combined pipeline-replica cell
resolves (a)+(b) together; pre-fab possible from tonight's pieces.
B5 loud-refusal: PASS. B1..B4 oracles: still owed (queued).

### §5 RESOLUTION ADDENDUM 3 (G-AMD-20, 22:07Z): STRIDE-4 ANOMALY CLOSED — runner input-prep, instrument bug #4

v5b replay of same-run captured partials reproduced stride-4 in the single-object
context ⇒ class (a). Upstream trace: pacc even-d slots EXACTLY 0x0000 in the real
d-fastest layout — poison born at the runner's input upload: q/k/v/q_dec uploaded
as FLOAT32 bit patterns into DType::BF16 tensors while lcg_bf16() zeroes float low
halves ⇒ every even bf16 element 0x0000, end to end; append/cache/partial/reduce
ALL FAITHFUL. Fixed runner-side (pack bf16 bits, 2 B/elem, commit 3ea321dd):
post-fix decode output contiguous real bf16, m 18→37 as K-restoration predicts.
'executing code ≠ fatbin' dissolved — every product kernel was innocent tonight.

OPEN REMAINDER (G-AMD-21): decode C441 still RED, NEW signature (cos exactly
0.000000) — prime suspect the fp64 oracle TABLES predate the fix (poisoned-input
era) or t3_oracle.cpp shares the bit-view; regenerate against packed-bf16
semantics; second suspect comparator NaN. Then B1..B4 emulation-parity oracles
(serve-critical now: agent4's q4 band runs on the emulations). Serve unaffected:
engine feeds real tensors through the product wrapper, not this runner's buffers. — rocgdb
  hardware watchpoint on d_out_dec during the real g16b launch; catch the storing
  instruction PC; map PC → code object → compare against fatbin symbol offsets.
Either "wrong code executing" (loader-level, coordinator+gemini probe lane) or
exotic-runtime; frozen evidence: g16b raw dumps + g16c trace CSVs + probe cells.

Queued behind the GDN/attention landing for agent4's link (coordinator-routed,
30-symbol residue; ordering call made 2026-09-12: GDN first, G-AMD-18 second).

**Parity-spec reconciliation (gemini's TP2_COMPARATIVE_PARITY_SPEC.md, consumed not
restated):** their battery gates TP2 SPLIT semantics (head-local/anti-permutation —
tests 6/7/10/11) and serve-level logits (argmax ≥95%, mean cos ≥0.990, KL ≤0.025);
my op-family comparator (per-q_head-role cos ≥0.999 + 2-ULP bf16 loud guard vs
in-process fp64 oracle) is complementary, not in conflict: role-wise comparison
already implements their anti-permutation intent at op level, and fp64-in-process
satisfies their no-transferable-goldens law by construction. D3's FP16/FP32-math
note governs implementation precision — my oracle cells verify it, unchanged.

SURVIVING SUSPECT: the donor decode kernel's key-loop internals
(`src/ops/kernel/gqa_attention_decode_bf16_gfx906.cuh:145–304` — the section
nobody has read) interacting with MY invocation: width/full_width = q.ne[2] = 1
(single-token 3-D q), TokenTile=1, splits=8, nullptr valid_columns (Masked=false).
Also unresolved: whether width should instead be the VISIBLE WIDTH (96) — i.e.
whether my q shape or the launcher's width derivation is the contract violation.

NEXT STEPS, in order:
1. Read `gqa_attention_decode_bf16_gfx906.cuh:145–304` (the score loop + paging
   walk + staging): what bounds the key enumeration; what makes m=-inf.
2. Read the launcher chain for the decode path: `small_t_launch_for` →
   `launch_tc_partial_bf16` → kernel args: which value carries the 96.
3. Fix (likely runner-side: q {kD, kQH, kHIST=1-token?} or invocation width).
   If a PRODUCT line is required: STOP-ASK the coordinator first (window law).
4. Re-stamp request (G-AMD-16) → re-run → compare vs
   `t3_oracle_decode_combined.table` under the bands → report.

Comparison bars (C441 seq-40/31 ruling): PASS = cos ≥ 0.999 per role vs the
fp64 oracle, computed from raw dumps; loud-failure guard = per-element
|d| ≤ 2 bf16-ULP-of-|want|; my old absolute 1e-3 band is REPLACED (unsatisfiable
for bf16 outputs — ULP 0.0078 at |v|~1.5).

### §5 G-AMD-21 ADDENDUM 4 (2026-09-12 ~23:3xZ, same session resumed — coordinator seq-50 stamp, dev2): DECODE PARITY cos-GREEN; guard RE-SCOPED (seq-53); shrinkage carrier localized

**Verdicts.** C441 decode: **cos ≥ 0.999 on 12/12 roles (min 0.999757)** — the lane's open
red is now bar-green on packed-bf16 inputs vs regenerated oracle (tables re-pinned-
verified: -O1/-O2 regen sha-identical to pins eb656c09/902c273a/20f69efb). B2 gating
gemv T=1 vs fp64: **PASS, cos=1.000000000 both planes, beta bit-exact, g maxrel 1.3e-7**
(first clean GDN device-parity datum of the sprint; logs g21_b2_run.log). Prefill:
NOT cos-green yet — counters under the re-scoped bands: 146/276/23139 pass/order/fail,
same shrinkage family (below).
**Two more instrument bugs, mine, named: #6** the comparator's `out_f` was shadowed by
a second zero-initialized vector (run-1 `got=0.000000` RED was a binding, device was
clean — same all-zero GHOST as run3, different cause); **#7** `grep -c` exits 1 on zero
matches and silently short-circuited a build link step — a STALE binary almost ran;
caught by the pre-launch re-hash rule, which has now paid for itself twice tonight.
Plus the operator-error segfault (running the oracle generator from inside results/
makes its relative-path fopen NULL — its output is the file's own working-directory
contract; use repo-root cwd).

**Guard re-scope (coordinator ruling seq-53, data-backed):** element 2-ULP guard is
NOT a verdict; the pass bar is per-role cos. Characterization rows: ratio got/want
(|want|>0.3): p05 0.903 p50 **0.9397** p95 0.972, min 0.876 — a ONE-SIDED ~6% shrink
that preserves direction (cos .9998). Carrier localization (all CPU, same-run captures):
(i) device per-split maxima m match the fp64 model only up to ±1.4 — and the model
values are QUANTIZED by the kernel's bf16-round score pipeline (device m keeps full
fp32 mantissa; dot-vs-rounding differences of ±0.25-1.4 units in a softmax whose
top-5 scores spread by 0.35 = near-ties everywhere ⇒ weights redistribute a few % ⇒
the observed shrink); (ii) fp64 re-combine of the device's OWN captured partials
(g20_pacc/pm/pl) recovers only 0.96-0.97 vs oracle — the representation loss lives in
the BF16 `acc` partial plane (product's own arena contract — |acc|≈15 ⇒ quantum 0.0625,
±0.4%) plus the score quantization, NOT in the reduce kernel's combine arithmetic;
(iii) the 0.999757 cos is the SAME number that hid at odd slots in the stride-4 runs —
no regression across the whole fix chain.
**Reading for serve:** this accuracy class is by-design for the small-t decode family
(bf16 partials + bf16-rounded scores near winner-take-all); temp-0 argmax is safe where
the key-margin exceeds the quantum — the ratio row + m-spread is how to say it honestly
(coord's near-tie rule). If tighter parity is ever demanded, the ask is a PRODUCT change
(FP32 acc plane in the arena contract) — STOP-ASK territory, nobody should schedule on it.

**Wave64 status update:** B2 (pure SIMT block-reduce, bit-exact vs fp64) proves the
SIMT reduction class on this box. HONEST LIMIT: the decode route on HIP serves the
donor SIMT kernel (gqa_attention_decode_bf16_gfx906.cuh per the launcher contract;
the TC kernel in decode_bf16.cuh is CUDA-only by the T3 seam) — so my mma/ldmatrix
EMULATIONS were NOT exercised by the decode green. Untested surface unchanged:
mma_bf16/ldmatrix emulations (serve-critical for agent4's q4 swiglu band + GDN
chunked B4). A dedicated synthetic fragment cell (frags loaded from host-exact
patterns -> emulation -> compare fp64 product) is now the cheapest decisive test
and is the next authored item before any B4 window ask.

**Artifacts:** binaries a3550b46 (run1, invalid-#6), baf14d16 (run2), 93a35ee4 (run3,
current HEAD 0ab1b6e7+); logs g21_device_run{,2,3}.log + g21_b2_run.log, all stamped
with shas + archive-tree name (build-hip-amd @ 90ee070f, rebuilt .a 8643d6cb).

### §5 G-AMD-22/23 ADDENDUM 5 (2026-09-12 ~23:5xZ, one write per coordinator seq-58): PREFILL GREEN, (A) SHIPPED-AND-RECEIPTED, B1+B3 PASS — lane verification debt CLEARED except B4-design

**Parity state:** decode 12/12 roles cos (min 0.999757) AND prefill 8/8 tokens x 12/12 roles (min 0.999702, outliers(ratio<0.85) 19/24576 = razor-edge class named per seq-56). The prefill 'RED of the sprint' was finished by instrument **#8** (phase-C readback aliased raw bf16 into f32 low halves — fourth all-zero ghost on this box; heads 6-11 cos=0.000000 was the binding, not the device). B2 gemv PASS (beta bit-exact); **B1 recurrent PASS** (256/256 rows cos=1.000000000, state-sample 4.1e-6; my authored cell initially kept a retired-guard-style maxrel FAIL driver — aligned to the standing seq-56 ruling same-window with witness rows showing all maxrel points at |d|<=1e-7 near-zero-collapse; compliance correction, disclosed for veto, coordinator accepted-in-release); **B3 simt_c8 PASS** (16/16 planes, maxrel 1.7e-7, zero outliers).

**(A) token contract shipped + receipted:** FRAGCELL K2 = ldmatrix-fed fragments BIT-IDENTICAL to direct ISA loads, K1 mma_bf16 BIT-EXACT vs fp32-seq AND fp64 (maxrel=0), K3 zero NaNs — full PASS at G-AMD-23 (dev2, stamps seq-57/60; logs g22_window_step1.log, g23_window_run.log; binaries 7daab7a8/e2844608/6c9039c4 @ commit chain to fa0eb1c-era, rebuilt-at-launch deterministic). The wave64 shuffle-assumption caveat originally filed in §5 addendum-1 is now **RESOLVED for the emulation family** (K2 exercises the 64-bit-token shuffle distribution on-device; SIMT class proven earlier by B2). Landmine class disclosed + closed same session: nil-fault root (smem_addr truncation, no CVTA on ROCm 6.2) — sibling owners notified via coordinator (w8 splitk; kvarn/TC are CUDA-seamed or loud-throw, inert).
**Contract-vs-algebra law (coordinator-boarded):** cells with hand-computed tokens prove ALGEBRA; only product-TU compiles prove CONTRACT — my first macro form failed the real consumers' host pass (addrspace(3) needs static_cast-to-void* before reinterpret); caught pre-flight by compiling chunked/launch.cu + gating TU. Instruments #5..#8 lineage: oracle-cwd segfault (operator), out_f shadowing (#6), grep-exit stale-link (#7), phase-C aliasing (#8) — all mine, all named, all fixed with binding evidence.
**Registration note (gemini):** the 11-file re-pin at 4d32956a stands; mma.cuh PTX arm verified byte-identical vs 90ee070f (11/11 asm payload lines, extract-hash a0415964); call-site changes in my 3 product files are NINFER_LDM_ADDR spellings whose CUDA expansion is the verbatim smem_addr(p).

**REMAINING T3 BOOK:** (1) B4 chunked end-to-end parity — DRIFT-CHARACTERIZING bar. SELF-CORRECTION (mine, same session): the earlier 'static-smem exceeds 64KiB ceiling, needs reduction design' framing was UNVERIFIED relay (agent4's '71680 static' — zero occurrences in this tree) repeated as fact in my release line. MEASURED reality: the chunked kernels are ALREADY dynamic (`extern __shared__` + attr-raise + launch-site bytes in all three .cu launchers) and the output kernel's request is 24,576 B at the hard-coded BT=64 (arithmetic from committed constants: (8192+2·2048)·2) — well under the box's 65,536 measured ceiling. prepare_wy_wu/state_passing scale with plan-supplied panel configs (not compile-time constants in my files) — their runtime smem bytes are readable from the plan layer or by instrumenting a launch. NET: B4 is NOT design-blocked at all — EVERY instantiated chunked config fits and could launch TODAY: compile-time-evaluated smem requests (probe @ d5a47e5f): output 24,576 B; prepare <32,16> 24,320 B / <64,32> 29,440 B; state_passing NS16 45,312 B / NS32 57,600 B — max 57,600 < 65,536 measured ceiling. The migration proposal is WITHDRAWN; the reduction question is MOOT at these shapes; what remains for a future B4 is a normal device window (end-to-end parity of the chunked pipeline vs fp64 reference, drift-characterizing bar) — no product change to gate it. (2) FP32-acc-plane product ask — named, unscheduled (STOP-ASK territory per seq-56); (3) agent5-sweep dispatch targets on my six gdn/gating files — answered by custody default. Lane state: all windows closed, zero un-released grants, tree pushed, dev2 baseline verified.

## 6. Lane rules in force (violating these = rework)

- GPU launches need a written coordinator grant (G-AMD-14/15 pattern: read the
  grant block before launching; post "launching under stamp-intent, confirm me"
  if ambiguous — grant-before-launch is the law, an inversion is on record).
- Zero product mutation in verification windows.
- gemini owns gates/tests/CI exclusively. agent2 owns hip_shim/cuda_runtime.h.
- Registered read-only exceptions: ops/common/math.cuh, memory.cuh, mma.cuh
  (mma = 11th, full-family HIP exclusion, C441-ruled).
- Hub mesh: agent-comm broker is for GEMINI ONLY; pi agents use intercom.
- FP16-V plane on the HIP path: all V writers are the adopted donor family
  (fp16 bits); never pass bf16-bits V where __half* is read.
- Oracle bars: cos ≥ 0.999/role PASS; per-element |d| ≤ 2 bf16-ULP loud guard.
- Tables are sha-pinned in results/amd/t3_oracle_run.log — regenerate =
  re-pin + disclose.

## 7. Roster (who to message, via what)

- **coordinator** — plan owner; rulings, grants, merges. intercom (works).
- **agent2 (Agent-B)** — hip_shim, shim census, TB1/TB2. intercom.
- **gemini** — gates/tests (M0–M4, PG stages, check-(h), fingerprint spec,
  MODE-port wiring, embed sweep). **agent-comm hub ONLY** (gemini@b4791a54).
  UNRESPONSIVE since 15:00Z per coordinator — surface-and-wait, don't reassign.
- **agent1** — abandoned (API instability); their WO-03 scope folded into TA1/TA2.
- **user** — direct; velocity directives come from here.

## 8. Traps that cost real cycles (do not re-trip)

1. cmake flag name: NINFER_BACKEND (NINFER_BUILD_BACKEND silently defaults).
2. cmake not on PATH: /home/chris/opt/cmake/bin.
3. Census/fatal-first: a missing include can mask 19 later errors — fix include
   roots FIRST, then re-read the error list.
4. std::span needs -std=c++20 (quantized_weight.h chain).
5. hipcc positional .a/.o = parsed as source: always -L/-l.
6. hipHostAlloc: undeclared in ROCm 6.2 — use the shim's hipHostMalloc mapping
   (agent2 fixed the shim; if you see hipHostAlloc again, the shim regressed).
7. `__nanosleep` undeclared under the shim — use __builtin_amdgcn_s_sleep.
8. Macro-vs-template shadowing: if the durable shim provides a FUNCTION, a
   file-local `#define` of the same name silently shadows it and kills argument
   checking (three instances hit: syncwarp twin, funcattr block, host-alloc
   trio). Prefer deleting interims the moment the durable fix lands.
9. Donor kernels bulk-load 64 block-table entries: pad your block table.
10. Cache planes: size to page capacity (pages × 64 slots), not key count.
11. The geometry dispatch reads q.ne[1]: shape q as {kD, kQH, ...} (3-D for
    decode, ne[2] = tokens) — a flat/2-D q falls through to the Gqa35 fallback
    kernels = wrong-geometry execution.
12. Envelope min_visible_keys must be ≥ 1 (0 trips the split-capacity guard).
13. The line's own history: every "silently wrong" bug tonight was a contract
    mismatch, not an algorithm bug — read the contract before the algorithm.
