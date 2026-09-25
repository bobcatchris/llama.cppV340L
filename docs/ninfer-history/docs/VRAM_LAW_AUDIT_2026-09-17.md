# VRAM LAW AUDIT — 2026-09-17 (no-GPU sweep, tree `amd/tp4-cure`)

**Law audited against:** user order 2026-09-10, ABSOLUTE — *"NO ESTIMATED VRAM CHARGE MAY
EVER REFUSE A LAUNCH, ANYWHERE, EVER. Preflight/budget code reports MEASURED numbers only.
The allocator is the gate: if it doesn't fit, cudaMalloc says so, cleanly, in real time.
Estimate-based refusal constants (fixed budgets, prefix/ws reserve charges, safety
multipliers, slack terms) are BANNED."* Plus the ANTI-RESURRECTION rule (same order) and
LITH (2026-09-07: no phantom VRAM tax, ever).

**Scope:** full sweep of `src/` + `include/ninfer/` (723 files) in worktree
`/home/chris/worktrees/amd-tp4-cure` (branch `amd/tp4-cure`), emphasis
`src/runtime/tp2/`, `src/core/multi_gpu/`, `src/serve/`. NO-GPU audit: no binary run, no
build, no process touched. `src/ops/linear/nvfp4/nvfp4_small_t_hip.cu`,
`nvfp4_launch.h`, `results/amd/coherence/nvfp4_smallt_roofline_bench.cu` are under
in-flight edits by another desk — read-only grep only (no VRAM-refusal shape found in
them; they are excluded from pins/allowlist because their line numbers are moving).

**Regression cell:** `tools/guards/check_vram_refuse_constants.py` (zero-GPU, house
pattern of `check_bf16_low_guard.py`: `--selftest` mutation coverage both directions,
`--root` override). GREEN = every finding is a banked/allowlisted instance printed loud;
RED = a NEW constant-in-refusal-position instance, or a banked pin changed (forces
allowlist + this doc to be re-classified in the same commit). Run: `python3
tools/guards/check_vram_refuse_constants.py` (CI farm / PG-1 class — host-only).

---

## 1. Methodology

1. Anchor on the banked instances (task brief): tp_engine preflight charge
   (~:1147-1149 at bank time) and `tp2_budget.h` reserve constant.
2. Pattern sweep: all `cudaMemGetInfo`/`hipMemGetInfo` call sites (the LEGAL-gate
   signature); refusal-wording files (`refus|reject|deny`); fixed MiB-literal density
   (`* 1024 * 1024`, `<< 20`); budget vocabulary (`reserve|headroom|slack|safety|margin|
   allowance`); refusal-shaped messages (`throw` + insufficient/exceeds/no room/not
   enough/out of memory) — then manual classification of every hit against the law.
3. Decision rule (per law + task brief):
   - **LEGAL** — live-measured gate (`cudaMemGetInfo`/actual free vs actual composition),
     or comparison against the size of something ACTUALLY ALLOCATED (arena capacity,
     staged pool pages, buffer `ne[]`), or the allocator itself (cudaMalloc).
   - **VIOLATION** — a compile-time/config CONSTANT (fixed MiB reserve, headroom,
     allowance, slack term) participates in a decision that refuses/denies a launch,
     allocation, or boot.
   - **GRAY** — constant used for sizing or reporting only (no refusal reachable from
     it), or a composition term feeding a legal live-gate with banked measured-floor
     provenance. Logged, watched, not indicted.
   - Routing/wire-shape capability gates (e.g. one-shot-AR world bound,
     `tp_group.cpp:136` class) are OUT OF CLASS per the brief — a routing gate refuses
     nothing that fits the wire; flagged nowhere.
4. Everything found was re-derived from the tree at HEAD of `amd/tp4-cure` today; line
   numbers below are TODAY's (the banked tp_engine citation ~:1147-1149 now sits at
   :1144-1146 — same site, shifted by in-flight edits).

## 2. Inventory

### VIOLATIONS (constant refuses — owner-desk fix list in §3)

| ID | Site (today's lines) | Constant | What it refuses | Banked? |
|----|----------------------|----------|-----------------|---------|
| V1 | `src/runtime/tp2/tp2_budget.h:89` | `kTp2RuntimeReserveBytes = 1536 MiB` | feeds `fixed_bytes()` -> both TP2 gates; a config passing the composition by <1536 MiB is refused instead of being tried at `cudaMalloc` | BANKED (docs/154 §8 lineage; the default-KV mc=4 false refusal: fixed 8483 > 8160 free) |
| V2 | `src/runtime/tp2/tp_engine.cpp:1144-1146` (preflight charge) and `:996-999` (auto-capacity probe charge; same constant, no-divergence rule) | charge of V1, keyed `is_bf16_kv && conc>=2` | the preflight throw `:1306-1325` (`total_required_vram > live_free_min`) and the probe's `fitting == 0` throw — both inherit the 1536 MiB fixed term | BANKED (same instance as V1; gate SHAPE is legal — live min-over-world read vs composition — but the composition carries the fixed reserve) |
| V3 | constant `include/ninfer/types.h:141` (`kDefaultKvCapacityHeadroomBytes = 1024 MiB`, default of `KvCapacityPolicy::automatic()`); refuse site `src/runtime/engine/kv_capacity.cpp:92` | fixed 1024 MiB headroom | auto-KV boot refused whenever live-after-weights < 1024 MiB — the exact "fixed headroom reserve charge" class; this is the surviving SIBLING of the WO-VRAM-1 constant #1 (`headroom_bytes` 1024 MiB) that was deleted from `tp2_budget.h` on 2026-09-13 | NEW (unbanked) |
| V4 | compare `src/targets/qwen3_6/impl/runtime/program_impl.h:1447`, throw `:1449-1450`; constants `src/targets/qwen3_6/impl/runtime/layouts_impl.h:672-703` | `graph_allowance_bytes` = fixed 12/82/64/96 MiB per graph-topology class | boot refused when MEASURED CUDA-graph capture consumption exceeds the FIXED planned allowance — a safety-allowance refuse; the measured side is real, the threshold is an estimate | NEW (unbanked) |

Exposure note: V3/V4 live on the qwen3_6 non-TP2 engine path (`registry.cpp` ->
`resolve_kv_capacity` -> Program), not on the AMD TP2 boot path — but they are the same
phantom CLASS the law bans, and `types.h` is shared by every line. **No finding is more
severe than the banked two** (those sit on the ACTIVE AMD TP2 boot path); V3 is the most
notable: it is the resurrected twin of a constant this project already deleted once
(anti-resurrection rule territory — the deletion was scoped to `tp2_budget.h` and missed
the shared header).

### GRAY (constant present, no constant-refuse reachable; watched by the cell's philosophy)

| Site | Constant(s) | Why not a violation |
|------|-------------|---------------------|
| `tp2_budget.h:66-70` | `decoder_fixed 592 MiB`, `workspace default 2 GiB`, `staging 200 MiB`, `arena_padding 160 MiB` | composition terms with measured-floor provenance (calibrated serve logs; receipts printed per-term in the preflight composition line). Feeds a LIVE-measured gate (V2's throw). Itemization refinement is the owner desk's standing item. |
| `tp2_budget.h:99-101, 216-224` | drafter charges: MTP `+768 MiB` (2026-08-25 bisect), DFlash2 `1134 + lanes*45 MiB` | same class as above; keyed by backend enum, not blanket; unit-tested seam (`drafter_fixed_bytes`). |
| `tp2_budget.h:37-44` | `static_weights_bytes` default `9059 MiB` | launch-dead: TPEngine overwrites from the MEASURED manifest placement (2026-09-12); the reader-throw fallback to the default is dead (WO-VRAM-1 path 2). Survives only as the pure unit-model default for tests. |
| `layouts_impl.h:672-703` (sizing side) | same 12/82/64/96 MiB terms | when only SIZING the reservation plan they are GRAY; they become the V4 violation only at the `program_impl.h:1447` refuse. |
| `tp2_backend.cpp:4024, 6166` | fixed `64 MiB` scratch `DeviceBuffer` | pure allocation SIZING — the allocator is the gate. |
| `types.h:142-156` (`KvCapacityPolicy`) | `explicit_tokens 2048` default | config default, user-overridable, no capacity verdict. |

### LEGAL (verified live-measured or actual-bytes gates — no action)

| Site | Gate |
|------|------|
| `tp_engine.cpp:1200-1216` (live reads), `:1306` (throw) | preflight: LIVE `cudaMemGetInfo` min-over-WORLD vs actual composition; failed read = INSTRUMENT ERROR (never a fabricated zero); `NINFER_PREFLIGHT_DISABLE` operator override. |
| `tp_engine.cpp:1001-1014, 1044-1052` | auto-KV probe: same live min-over-world discipline; refuses only when `fitting == 0` against the live read. |
| `tp_engine.cpp:966-975` | placement-reader failure = INSTRUMENT ERROR, explicitly NOT a VRAM verdict. |
| `targets/registry.cpp:69-89, 105, 115` | weights measured from the load plan + LIVE `cudaMemGetInfo`; `:38` headroom-shape hit is config-CONSISTENCY validation (explicit mode must carry zero headroom) — allowlisted in the cell as NOT-a-refuse. |
| `core/arena.cu` (`DeviceArena`/`DeviceBuffer`) | `cudaMalloc` is the gate; `require_range` compares against ACTUAL allocated bytes. |
| `core/paged_kv_cache.cpp:149-161` | `can_reserve`/`reserve` against actually-reserved pool pages. |
| `ops/wrapper/sparse_moe.cpp:228-232`, `ops/kvarn/kvarn_workspace.cu:141` | compare vs ACTUALLY allocated workspace/staged capacity. |
| `runtime/engine/request_memory.cpp:61` | request transient vs the STARTUP-FROZEN (actually allocated) arena. |
| `runtime/engine/kv_capacity.cpp:99-146` (other arms) | curve bounds are structural plan figures; `available_runtime_bytes` side is live. |
| `core/multi_gpu/*` (`tp_group.cpp:131,262`, `argmax_r1.h`, `one_shot_allreduce.cu:703`, `one_shot_argmax.cu:500`, `gather_capacity.h`) | routing/wire-shape capability gates (OUT OF CLASS per brief) and real allocator/NCCL errors; several self-document as "CAPABILITY refusal, not a capacity verdict". |

### OUT OF SCOPE (not VRAM)

- Host-RAM/HTTP budgets: `serve_options.h:18-20` (384/256 MiB request/store), media
  cache/live caps, `media/decode` + `product/media_acquire` `BudgetExceeded` (upload
  byte limits), `serve/response_store.cpp:94`. Their `1ULL << 20` overflow guards are
  unit-scale conversions (the cell excludes division-by-unit shapes).
- Hardware launch-config bounds: rmsnorm/l2norm/layer_norm/causal_conv1d grid-limit
  throws (device-property facts, not VRAM estimates).
- Integer-range guards (int32/uint32 overflow), architecture bounds
  (`layouts_impl.h:545` native context), request/capacity consistency throws.

## 3. Owner-desk fix list (each: the LEGAL replacement)

1. **V1+V2 (BANKED — tp2 1536 MiB reserve).** Legal replacement: delete the fixed
   charge from the decision path; keep it ONLY as a REPORTED term (the preflight
   composition line already prints it with provenance). The gate stays exactly as it is:
   actual composition vs LIVE `cudaMemGetInfo` min-over-world. The reserve's real
   function (early legibility on bf16-MTP shapes) is preserved by the printed receipt +
   the allocator's own OOM message. If the desk prefers an interim keying narrowing
   (conc-shape/dtype), that is still a constant refuse — do not. Red/green rows required
   per closure law: RED = the banked default-KV mc=4 refusal repro; GREEN = same config
   booting with the charge gone and the suite cell pinning the composition print.
2. **V3 (types.h 1024 MiB headroom — NEW).** Legal replacement: `resolve_kv_capacity`
   must size auto-KV against the LIVE available bytes with NO headroom subtraction —
   `max_context_fitting`-style solve (tp2_budget.h already has the shape); if a
   post-proposal margin is wanted, REPORT the measured margin, never refuse on it. The
   allocator is the gate for the margin's truth. Delete
   `kDefaultKvCapacityHeadroomBytes` or re-purpose it to a report-only field.
   Anti-resurrection: this constant is the deleted WO-VRAM-1 #1's twin — its removal
   must come with the CI cell (already in place) staying green with the allowlist row
   dropped.
3. **V4 (graph allowance — NEW).** Legal replacement: `program_impl.h:1447` must REPORT
   (log measured consumed vs planned allowance) and PROCEED; the true gate is graph
   instantiation/capture itself, where `cudaMalloc` failure is the real, measured
   verdict. If a hard stop is genuinely required (capture corruption risk), the stop must
   key on the MEASURED delta against the device's LIVE free bytes
   (`cudaMemGetInfo` before/after capture, refuse only `consumed > live_free_after` is
   false — i.e. only when capture actually exhausted the device), never against the
   fixed 12/82/64/96 MiB allowance.

## 4. Cell semantics (read before editing the allowlist)

- `KNOWN_INSTANCES` (allowlist): keyed `(file, anchor)`; each hit prints LOUD with its
  citation every run. Today: V3 refuse site (kv_capacity.cpp), V4 refuse site
  (program_impl.h), registry.cpp config-consistency (NOT a refuse — cited).
- `PINS` (banked-citation integrity): PIN-1 tp2_budget.h:89 constant, PIN-2
  tp_engine.cpp preflight charge site, PIN-3 types.h headroom constant, PIN-4
  kv_capacity.cpp refuse compare, PIN-5 program_impl.h refuse compare. A pin going RED
  means the banked line moved or the fix landed — update this doc + allowlist in the
  SAME commit as the fix (RED->GREEN closure law; a fix PR without its cell/doc update
  is rejectable on sight).
- Known limit (stated honestly): the scanners are line-local (decision shape = constant
  + comparison + refusal arm within 4 lines). A constant flowing through an intermediate
  variable from far away is invisible to them — that shape is exactly what PIN-2/PIN-4/
  PIN-5 pin manually. The cell is a regression NET, not a dataflow proof.

## 5. Counts (for the coordinator ledger)

- VIOLATION rows: 4 (V1, V2 = the two banked instances; V3, V4 = NEW, unbanked).
- GRAY rows: 6 clusters; LEGAL gates verified: 11 clusters; OUT-OF-SCOPE: 3 clusters.
- Severity call: **nothing more severe than the known two** — V3/V4 are real
  constant-refuses but sit off the AMD TP2 boot path; V3 flagged as
  resurrection-adjacent (twin of a deleted constant surviving in a shared header).

*Audited by the no-GPU audit agent, 2026-09-17, worktree `amd-tp4-cure`. Cell:
`tools/guards/check_vram_refuse_constants.py` — selftest PASS (5 bad/mutation detected,
17 pristine/routing clean); tree scan GREEN (0 new, 4 allowlisted loud, 5 pins intact,
723 files).*
