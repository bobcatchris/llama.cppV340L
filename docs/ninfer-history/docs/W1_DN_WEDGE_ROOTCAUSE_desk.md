# W1 DN-WEDGE ROOT-CAUSE desk — SIMT chunked-prefill trio, T=128/chunks=2 device wedge

**Desk:** static-analysis (no GPU, no builds, read-only), Team Red W1, 2026-09-18.
**Scope read:** all of
`src/ops/linear_attention/gated_delta_net/chunked/` (three `*_simt.cuh` kernels + their
`.cu` launchers + the mma originals + `launch.{h,cu}` + `simt_gate.h`), the op entry
`gated_delta_net.cpp`, `core/arena.h`, `core/prefill_body_trace.h`,
`core/multi_gpu/one_shot_allreduce.cu`, plus the evidence row and serve log.

## R2 rule — closest prior ledger failure and what this desk does differently

Closest prior failure: **the arm-A wedge row itself**,
`results/amd/coherence/W6_row_armA_gdn_RED.txt` (2026-09-18 06:3x, window 6): ARM-A
(`NINFER_GDN_SIMT=1`) RED by device wedge — 3 of 4 dies pinned 100% for 300 s on the
FIRST chunked call at T=128/chunks=2, no fault, no abort, no completion; probe TIMEOUT
rc=28; recovery needed `rocm-smi --gpureset` on dies 0/2/3. That row is an *attempt*
ledger (route proof + observables + verdict, no mechanism). This desk does the different
thing: **root-cause from source alone** — every loop, barrier, index, and buffer in the
trio and its hosts read line-by-line against the mma originals and the workspace layout —
and ships (a) the defect set with FILE:LINE arithmetic, (b) concrete fix diffs for the
lane owner to apply and bank, (c) a device-cell design that turns the wedge into a
RED→GREEN pair per the CLOSURE LAW. No re-attempt is proposed anywhere in this report.

## 0. Executive summary

The trio contains **three real device-addressing defects**, two of which fire only at
`chunk >= 1` and one at every chunk count:

1. **`h_chunk` base built from the token offset instead of the chunk index** — in BOTH
   `state_passing_simt.cuh:126-128` (WRITE side) and `output_simt.cuh:70,77` (READ side).
   At chunk 1 this reads/writes **~24 MiB past a 768 KiB tensor** (64x the intended
   offset). This is the ONLY chunks>=2-dependent defect in the trio and the prime wedge
   candidate.
2. **`v_row` double-offset in prepare's U product** — `prepare_wy_wu_simt.cuh:259,265`:
   V is read at token `cs+t+s` instead of `cs+s`; semantically wrong AND out-of-bounds by
   up to ~190 KiB at **every** chunk count.
3. (provenance, not a defect line) The trio's **first device execution ever** was the
   wedging T=128 call: the "T=51/chunks=1 GREEN" datapoint in the evidence row never ran
   the SIMT trio at all (§1) — there is no chunks=1-green leg to contrast against.

The trio's instruction streams are provably spin-free and deadlock-free (§3.3): every
loop is constant-bounded, every barrier uniform. The no-fault 300 s wedge therefore
enters through the **memory system**, not the control flow: the chunk-1 `h_chunk` OOB
write (~22.3 MiB past the end of the 1.88 MiB bump-arena stage slice) either (E) lands
past the 96 MiB work arena and VM-faults the queue into a silent halt (the fault text
lives in dmesg/KFD, not the serve log — the W6 recovery reset the dies without capturing
dmesg, so that observation is lost), or (i) lands inside the arena and silently corrupts
~384 KiB of live chunk-graph tensors, after which the stream's only no-progress
primitive — the one-shot AR flag poll, `one_shot_allreduce.cu:176-181` — spins forever at
100%. Both continuations produce exactly the W6 signature; the device cell in §7 decides
between them and provides the RED capture either way.

---

## 1. Premise correction: the T=51 leg never exercised the SIMT trio

The evidence row's central contrast — "the SAME trio at T=51 tokens with chunks=1 runs
CLEAN and fast (dn=12.0 ms)" — is **false**, and this changes the desk's shape.

- `gated_delta_net.cpp:271-274`:
  ```cpp
  const std::int32_t T_full =
      (force_recurrent || T < detail::gated_delta_net::kChunkSize)
          ? 0
          : (T / detail::gated_delta_net::kChunkSize) * detail::gated_delta_net::kChunkSize;
  ```
  At T=51 < kChunkSize(64), `T_full = 0` → `launch_chunked` is never called
  (`gated_delta_net.cpp:291-301` runs only `if (T_full > 0)`); the whole probe runs the
  **recurrent tail path** (`launch_recurrent_inout`, line 315). `T=51` cannot take the
  chunked path — `stage_validator::check_full_chunks` (`chunked/launch.h:121-130`)
  rejects any T that is not a multiple of 64.
- Serve log `W6_boot_armA_gdn_serve.log` confirms: the tok=51 self-probe block (lines
  40-55, `dn=12.021`) is printed with **no `[GDN]` lines anywhere before it**. The
  `[GDN]` gate trace and dispatch receipt (which fire on the FIRST chunked consult that
  finds the env var, `simt_gate.h:52-67` and `prepare_wy_wu.cu:45-47`) appear at lines
  160-161 — **at the T=128 request**, naming `T=128 chunks=2`. The consult itself, hence
  the trio's first device execution in the process, is the wedging call.
- The dn=12.0 ms matches the baseline default-off probe (11.7-12.3) precisely *because it
  is the same recurrent code*, not because the SIMT trio is fast.

Consequences: (a) the "chunks=1 works / chunks=2 hangs" delta in the row's PRIME SUSPECT
note must be re-derived from source, not from that datapoint; (b) defect 2 above has
NEVER been masked by a passing chunks=1 run — the trio has never once completed on
device at any shape; (c) the desk's cell must use **T=64** (not T=51) for its chunks=1
leg (see §7).

## 2. The defects, with FILE:LINE and arithmetic

### 2.1 `h_chunk` base: `cs` (token offset) used where `chunk` (chunk index) is required

Workspace layout, `chunked/launch.h:32-46`: `h_chunk` is
`{kStateDim=128, kStateDim=128, value_heads, chunks}` — ggml ordering, so the **chunk is
the outermost index**: element (d, k, h, c) lives at
`((c*H_v + h)*128 + d)*128 + k`. Capacity at T=128: `128*128*12*2 = 393,216` elements
(768 KiB).

The mma originals index it by chunk:
- `state_passing.cuh:280`: `hc_chunk_stride = H_v * kStateDim * kStateDim;`
  `:287`: `hc_block_base = h_v * kStateDim * kStateDim;` advanced per chunk by
  `:516`: `hc_base += hc_chunk_stride;`
- `output.cuh:185-186`: `hc_base = (chunk * H_v + h_v) * kStateDim * kStateDim;`

Both SIMT ports instead use `cs = chunk*BT` (the TOKEN offset — correct for the
token-major W/U/v_new/g_cumsum tensors, wrong for the one chunk-major tensor):
- **`state_passing_simt.cuh:126-128`** (inside the serial chunk loop, line 125):
  ```cuda
  const int64_t cs = static_cast<int64_t>(chunk) * BT;
  const int64_t hc_base =
      (cs * H_v + static_cast<int64_t>(h_v)) * kStateDim * kStateDim;
  ```
  consumed by the snapshot WRITE at `:131-137` (`h_chunk + hc_base + (d_off + r)*128 + c2`).
- **`output_simt.cuh:70,77`**:
  ```cuda
  const int64_t cs     = static_cast<int64_t>(chunk) * BT;
  const int64_t hc_base = (cs * H_v + static_cast<int64_t>(h_v)) * kStateDim * kStateDim;
  ```
  consumed by the h_panel READS at `:122-127` inside the 8-panel loop.

Arithmetic at T=128 (H_v=12, BT=64, chunks=2):

| | intended base (elts) | actual base (elts) | actual bytes from h_chunk start |
|---|---|---|---|
| chunk 0, h_v | (0*12+h_v)*16384 = h_v*16384 | same (cs=0) — **coincidentally correct** | 0..360 KiB (in bounds) |
| chunk 1, h_v | (1*12+h_v)*16384 = 196,608..393,215 | **(64*12+h_v)*16384 = 12,582,912..12,779,263** | **24.02..24.40 MiB** (buffer is 0.75 MiB) |

So chunk 1 writes/reads at **64x the intended offset — 32x past the whole tensor**.
Volume: state_passing writes 96 CTAs x 4 KiB = 384 KiB (scattered 4-KiB strips across a
~360 KiB window at +24 MiB); output reads 48 chunk-1 CTAs x 8 panels x 4 KiB = 1.5 MiB
from the same window (so even with no fault, `attn_out` for chunk 1 is computed from
garbage). The dn stage (`scratch.stage`, `gated_delta_net.cpp:206-207`) is 1,972,224 B =
1.88 MiB at T=128 (`launch.h:36-44`), with h_chunk its LAST tensor — so the OOB window
sits **~22.3 MiB past the end of the stage allocation**, at
`stage_base + ~23.4 MiB` absolute.

### 2.2 `v_row` double-offset in prepare's U product (all chunk counts)

`prepare_wy_wu_simt.cuh`, Phase G:

```cuda
245    const int64_t v_st = H_v * kStateDim;                  // (line 109 region; v_st = token stride)
247    const int64_t v_base       = cs * v_st + static_cast<int64_t>(h_v) * kStateDim;
248    const __nv_bfloat16* v_stp = v_in + v_base + d_off;
...
259    const __nv_bfloat16* v_row = v_stp + static_cast<int64_t>(t) * v_st;   // <-- extra t*v_st
...
261        for (int s = 0; s < BT; ++s) {
...
264            const __nv_bfloat162 vp = *reinterpret_cast<const __nv_bfloat162*>(
265                v_row + static_cast<int64_t>(s) * v_st + d2);              // <-- + s*v_st again
```

The address is `v_in + (cs + t + s)*v_st + ...` — V read at token **cs+t+s** where the
math requires **cs+s** (`U[t][d] = sum_s T_inv[t][s] * (beta_s * V[cs+s][d])`). The mma
original stages the chunk's V panel once at `v_in + v_base + row*v_stride_t` with
`row` chunk-local (`prepare_wy_wu.cuh:453-454, 528-535, 699-702`) — no `t` term. Effects:

- **Numerics:** every U cell with `t+s >= 64` sums the wrong V rows — roughly half the
  product is garbage at any shape. The declared acceptance (rel-L2 < 1e-2, header lines
  47-51) cannot pass on device. It was never checked on device: the only parity
  instrument shipped with FIX-A is the **host-emulated** logic cell
  (`tests/test_gdn_simt_parity.cpp`, GDN_SIMT_NOTES.md §5), which mirrors the algebra,
  not the device addresses, and its own honest boundary note says exactly that.
- **Bounds:** at chunks=1 (T=64) the read reaches token 126 vs a 64-token slice
  (`v_full = v.slice(2, 0, T_full)`, `gated_delta_net.cpp:294`) → up to
  62 rows x 1536 elts x 2 B ≈ **190 KiB past the v allocation**; at chunks=2 (T=128) it
  reaches token 190 vs 128 → the same ~190 KiB class of overshoot. These are READS —
  silent garbage when they land in mapped memory (almost always; v's neighbors are
  activation tensors), a VM fault only if they cross the end of an allocation.

### 2.3 What is NOT wrong (verified so the desk doesn't re-chase)

- prepare's W/U/g_cumsum stores (`:245-246, 277`), its K loads (`:115-122`), the g scan
  and publish (`:126-147`) — all token-strided and in bounds at chunks=2 (max k offset
  65,534 of 65,536; max W/U offset < 196,608; max g offset 1,023 of 1,536).
- state_passing's W/K/g prefetch (`:189-204`) and U/v_new accesses (`:153-160`) — in
  bounds at chunk 1 (token indices <= 127 of 128).
- The `if (x == 0.0f) continue;` skip in Phase G (`:263`) is a mask, not a loop.

## 3. The wedge mechanism

### 3.1 Why "no fault, clean 100% spin" is exactly what this bug class produces here

The dn stage is a bump-arena slice: `DeviceArena` is a single device allocation with
scope save/restore (`core/arena.h:42-89`), sized `NINFER_WORKSPACE_MIB=96` in this boot
(serve log line 2). The stage's offset inside the arena is whatever the enclosing chunk
scope watermark is at the dn call — static analysis cannot pin it, and the two
continuations differ by exactly that:

- **(E) fault-halt branch:** if the stage sits within ~24 MiB of the arena top
  (offset > ~72 MiB), the chunk-1 h_chunk writes land past the arena's hipMalloc end →
  unmapped VA → VM fault on gfx900. On this stack the fault text goes to **dmesg/KFD,
  not the serve log**; a faulted compute queue halts with waves gone or CP wedged and
  the host's `ctx_.synchronize()` blocks indefinitely — no userspace fault line, no
  abort, process alive and ignoring SIGTERM. The W6 recovery receipt ("KFD=0 after")
  reflects a post-kill check, not a dmesg capture; **the decisive dmesg observation was
  destroyed by the gpureset before anyone looked** (recovery runbook gap — see §7.6).
- **(i) corruption-cascade branch:** if the stage sits lower, the 384 KiB stomp lands on
  live chunk-graph tensors ~22 MiB into the arena → silent numeric corruption. The
  corrupted chunk's kernels still complete — the trio has no spin (§3.3) — so the
  no-progress state forms downstream, where the prefill stream's ONLY unbounded wait
  lives: the one-shot AR's device flag poll, `one_shot_allreduce.cu:176-181`,
  `while (*peer_flag < expected_flag)` on pinned-host flags (the W4-guard bounded arm is
  env-gated; the default arm is unbounded). A rank whose stream died — or whose
  choreography partner never publishes — spins there at 100% GPU use, no fault, forever.

Both branches produce the W6 observables: zero `[PREFILL-BODY]` lines (those print only
after the chunk-end `ctx_.synchronize()` drains the event ring,
`prefill_body_trace.h:60-63, 204-208`), 100% on the wedged/spinning dies, no abort line,
host alive (the host raced 336+ `[TILED]` launches ahead into the async queue — serve
log lines 159-165 — then sat in the sync). The 0%/100% split across ranks (rank 1 at 0%
"waiting in AR") is a snapshot of *where each rank's stream had reached* when the first
stream died — not evidence of a rank-specific code path; the trio and its placement are
identical on all four ranks.

### 3.2 Why this is the chunks>=2 delta — and why chunks=1 "worked" in the row

There is **no chunks=1 device datapoint** (§1). From source: defects 2.1's two kernels
are byte-identical to correct behavior at chunk 0 **because cs=0 collapses the wrong
formula to the right one** (`(0*H_v + h_v)` vs `(0*BT*H_v + h_v)` — both `h_v*16384`).
A chunks=1 run (T=64) would therefore complete with correct h_chunk/state and only the
2.2 garbage in U — a silent-wrongness class, not a wedge. The wedge requires chunk 1,
i.e. **chunks >= 2, i.e. T >= 128** — precisely the W6 probe shape. That is the precise
delta, derived from source instead of the row's T=51 misread.

### 3.3 Negative proof: the trio cannot spin or deadlock by its own control flow

Every loop in the three kernels has a compile-time-constant bound and a uniform trip
count across the CTA; the only runtime bound is state_passing's
`for (int chunk = 0; chunk < chunks; ++chunk)` (`state_passing_simt.cuh:125`) with
`chunks` a by-value kernel arg printed as 2 in the dispatch receipt. Barrier audit:

- prepare: `__syncthreads()` at `:148, 167, 181, 201, 239` — unconditional;
  Phase E's per-pair barrier `:229` sits in loops whose trip count (i=1..3, j<i → 6
  barriers/thread) is tid-independent; Phase D's `__syncwarp()` (`:196,198`) lives in the
  `tid < 4*kWarpSize` region, which contains whole warps 0-3 only — warp-uniform, and
  `__syncwarp` is warp-scoped anyway.
- state_passing: barriers at `:123, 138, 166, 175, 186, 205` — all unconditional; the
  only conditional block, the Phase D' prefetch (`:189-204`, `chunk + 1 < chunks`), has
  its trailing barrier at `:205` **outside** the if — uniform by construction.
- output: barriers at `:95, 116, 136, 157` — all unconditional; the 8-panel loop is
  uniform.

No `while`, no global-memory flags, no data-dependent exits anywhere in the trio. A
wedged kernel in this trio is therefore impossible without the memory system poisoning
either the waves (fault branch E) or the downstream choreography (cascade branch i).

## 4. Proposed fix (diffs for the lane owner to apply and bank)

Three minimal edits; each restores the mma original's addressing semantics.

**(1) `state_passing_simt.cuh:127-128`** — chunk index, not token offset:

```diff
         const int64_t cs = static_cast<int64_t>(chunk) * BT;
         const int64_t hc_base =
-            (cs * H_v + static_cast<int64_t>(h_v)) * kStateDim * kStateDim;
+            (static_cast<int64_t>(chunk) * H_v + static_cast<int64_t>(h_v)) * kStateDim * kStateDim;
```
(`cs` stays: it still feeds the Phase B' `u_off` and the Phase D' prefetch.)

**(2) `output_simt.cuh:77`** — same correction on the read side:

```diff
-    const int64_t hc_base = (cs * H_v + static_cast<int64_t>(h_v)) * kStateDim * kStateDim;
+    const int64_t hc_base =
+        (static_cast<int64_t>(chunk) * H_v + static_cast<int64_t>(h_v)) * kStateDim * kStateDim;
```

**(3) `prepare_wy_wu_simt.cuh:259 + 265`** — drop the stray `t*v_st` term:

```diff
-        const __nv_bfloat16* v_row = v_stp + static_cast<int64_t>(t) * v_st;
...
         const float2 v0 = bf16x2_to_float2(vp);
...
-            const __nv_bfloat162 vp = *reinterpret_cast<const __nv_bfloat162*>(
-                v_row + static_cast<int64_t>(s) * v_st + d2);
+            const __nv_bfloat162 vp = *reinterpret_cast<const __nv_bfloat162*>(
+                v_stp + static_cast<int64_t>(s) * v_st + d2);
```
(After edit (3), `v_row` has no remaining uses — delete the line. The `ti_row[s]`,
`beta_smem[s]`, `bg_smem[s]`, and `k_smem[s][d2]` operands are already chunk-local and
correct.)

Reasoning: (1)+(2) restore the mma ground truth
(`state_passing.cuh:280,287,516`; `output.cuh:185-186`) for the one chunk-major tensor;
(3) restores the mma V-panel semantics (`prepare_wy_wu.cuh:528-535`) and simultaneously
removes the out-of-bounds read class. No grid, barrier, or occupancy changes — the
T=51-vs-T=128 delta the row suspected in the *loop* is not where the defect lives; it
lives in the chunk-1 *address arithmetic*.

## 5. Ruled-out candidates (so the desk does not re-chase them)

1. **chunks=2 serial state_passing loop as a liveness bug** (the row's prime suspect
   #1): bound is the kernel arg (2); all inner work constant-bounded; the only
   conditional block (prefetch) has its barrier outside the if (`:205`). Ruled out —
   §3.3.
2. **Barrier divergence in the "block inversion + Schur with DECLARED barrier/ordering
   constraints"** (the design note the row points at — `GDN_SIMT_NOTES.md:34,140-141`,
   prepare header Phase E `:203-232`): the Schur constraints (a)/(b) are
   ordering-for-correctness constraints on read-before-overwrite, not liveness
   mechanisms; every thread executes the identical 6-barrier sequence. The Phase D
   in-place inversion (`:186-200`) uses `__syncwarp` only, inside whole warps. Ruled out
   as a hang source. (Whether the Schur order is numerically right is covered by cell
   leg A, not by this desk.)
3. **A `last-chunk-partial` branch diverging at exactly T=128**: no such branch exists
   anywhere in the trio — `check_full_chunks` (`launch.h:121-130`) rejects non-multiples
   of 64, and T<64 never reaches the chunked launcher (§1). T=128 takes the fully
   generic path; there is nothing special about "exact" T=128 other than chunks=2.
4. **Stride-skip / grid-shape loop exit** (hang class #1 in the desk briefing): all idx
   loops are `for (idx = tid; idx < CONST; idx += 256)` — exit condition reachable for
   every tid at every grid shape; grid dims enter only through global addresses.
5. **Producer/consumer flag spin between chunks** (hang class #2): no device flags exist
   in the trio; inter-stage ordering is stream order (`launch.cu:49,69,86`). The only
   flag spin in the prefill stream is the AR's (ruled in only as the SECONDARY cascade,
   §3.1-i — and arms B/C/D ran the same AR GREEN on the same window/binary, so the AR is
   not the primary).
6. **VRAM-law refusal constant / preflight estimate**: the SIMT launchers contain no
   budget logic; the wedge is not a refusal.
7. **chunks kernel-arg corruption**: the dispatch receipt printed `chunks=2` from the
   same value (`prepare_wy_wu.cu:46` passes `(int)NT` from `cfg.L/BT`).
8. **mma-path regression**: untouched by the gate (`simt_gate.h` strict parse; launchers
   route only when armed); the W6 baseline leg was GREEN on the same binary.
9. **output.cuh:362 barrier-inside-if** (`if (chunk + chunk_stride < chunks)
   __syncthreads();` in the mma MULTI_JOB loop): CTA-uniform condition (all tid-invariant
   values) — and mma-path anyway, not under test.

## 6. What static analysis cannot decide (honest boundary)

Which continuation branch (E vs i) the server took — i.e. whether the +22.3 MiB window
was unmapped (fault-halt) or mapped arena bytes (silent corruption → AR cascade) —
depends on the runtime arena watermark, which is not derivable from source. The cell
below is designed so that **both branches produce a deterministic, box-safe RED** and the
optional tight arm can reproduce branch (E) deliberately in a dedicated window.

## 7. Device cell — RED→GREEN per the CLOSURE LAW

**File:** `tools/v340l/dn_simt_chunk_cell.cu` (this lane worktree). Single die is
sufficient — the trio is per-rank code; the cell runs the exact wedge geometry
(H_qk=4, H_v=12, D=128, BT=64).

### 7.1 Structure (legs run in this order)

- **Provenance banner first:** `getpid()`, `hipGetDevice` + `hipDeviceProp_t::pciBusID`
  (die index), hip runtime version, and the `git rev-parse HEAD` + `sha256sum` of the
  three `*_simt.cuh` files (captured into `-D` macros at compile time and echoed at
  runtime) — the row's three shas per closure law.
- **LEG A (T=64, chunks=1) FIRST — the sanity leg, corrected:** the task sketch said
  "T=51/chunks=1"; T=51 cannot run chunked (§1), so the leg is **T=64**. Runs the mma
  trio (gate unset) as reference, then the SIMT trio (gate armed by `setenv` in-process —
  legal: `gdn_simt_enabled()` getenvs fresh, `simt_gate.h:52-67`). Compares
  `{W, U, v_new, h_chunk, state_out, attn_out}` rel-L2 < 1e-2 (the declared gate) and
  scans poison (7.2). Pre-fix RED expectation: **numerics RED on U/attn_out (defect 2.2)
  and the v-tail poison touched** — no hang, no fault.
- **LEG B (T=128, chunks=2) — the wedge leg:** same A/B comparison at the wedge geometry
  with the SIMT trio under the watchdog (7.4). Pre-fix RED expectation: workspace-poison
  stomp receipt whose first offending offset is
  `h_chunk_start + 12,582,912 elts` (~24.02 MiB — the smoking gun; print offset and diff
  against the prediction), plus `h_chunk`/`attn_out` numerics RED; if the box reproduces
  the server's hang class in isolation, the watchdog fires WEDGE first. Post-fix GREEN:
  completes, all tensors rel-L2 < 1e-2 vs the mma chunks=2 reference, poison clean.
- **LEG C (handoff cross-check, runs post-fix as part of GREEN):** SIMT T=128/chunks=2
  vs two SIMT T=64/chunks=1 calls with explicit state chaining
  (`state_out(leg1) → state_in(leg2)`) — rel-L2 < 1e-2 on `v_new/h_chunk/attn_out`.
  Validates the cross-chunk handoff independently of the mma reference.
- **Optional `--tight` arm (flag-gated, NOT part of the standing suite):** exact-size
  allocations (no poison tail) so the chunk-1 OOB hits unmapped VA — reproduces server
  branch (E) deliberately. Prints the fault warning and the FLR recipe BEFORE running;
  requires `--i-know-this-can-fault`. Run only in a dedicated card window.

### 7.2 Fault-free-by-construction allocation (ops-law compliance)

One inputs slab and one workspace slab per leg, each `needed_bytes + 64 MiB` with the
tail memset to `0xC5C5C5C5`. Rationale: the pre-fix chunk-1 h_chunk stomp lands ~22.3 MiB
past the stage end — INSIDE the 64 MiB poison region — so the default cell can never
fault the box, yet the stomp is caught deterministically by a post-run scan (report the
first offending byte offset + the tensor it pollutes). Input tensors get 2 MiB poison
tails each (catches defect 2.2's ~190 KiB v overrun). The mma reference legs run in
plain exact-size workspaces (mma is in-bounds by production history).

### 7.3 Synthetic tensors

Deterministic LCG fill, fp32-then-round: k, q, v in [-1, 1); g = -0.01*(1 + |x|)
(monotone-decreasing cumsum, the realistic decay class); beta in [0.01, 1];
scale = 1/sqrt(128); state_in fp32 zeros (`[128,128,12]`, the recurrent layout). dtypes:
q/k/v bf16, g/beta fp32, state fp32 in/out (the `<float,float>` launcher arm).

### 7.4 Watchdog mechanics (host-side bounded wait)

After the third `launch_output` enqueue: `hipEventRecord(done_ev, stream)`, then poll
`hipEventQuery(done_ev)` against a `steady_clock` 10 s deadline. On timeout: print
`WEDGE leg=B pid=<pid> die=<idx> pci=<bus>` + last HIP error state, then a BEST-EFFORT
bounded `hipDeviceReset()` wrapped in `alarm(30)` (it may itself hang on a wedged
context — that is expected and documented), print
`RECOVERY: operator runs PLOG-047/049 FLR — sudo rocm-smi --gpureset -d <die>; verify rocminfo gfx900 + 4x0% before any next boot`,
`exit(2)` = RED-wedge. `exit(1)` = numerics/canary RED; `exit(0)` = GREEN. No pkill, no
cross-process kills, nothing outside the cell's own PID.

### 7.5 Compile (standalone, gfx900)

```
hipcc --offload-arch=gfx900 -std=c++20 -O2 -I src -I include \
  tools/v340l/dn_simt_chunk_cell.cu \
  src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu \
  src/ops/linear_attention/gated_delta_net/chunked/state_passing.cu \
  src/ops/linear_attention/gated_delta_net/chunked/output.cu \
  -o bin/dn_simt_chunk_cell
```
Link closure is header-only for the repo bits the launchers pull
(`sku_launch_config.h`, `prefill_body_trace.h`, `simt_gate.h` are headers; if
`core/device.h` grows a TU dependency at first link, append that one `.cu`). The cell TU
includes `chunked/launch.h` for the config structs and calls
`chunked::launch_prepare_wy_wu / launch_state_passing / launch_output` directly.

### 7.6 Recovery runbook addition (cheap, decisive, learned the hard way here)

Before ANY `--gpureset` on a wedged box: `sudo dmesg | grep -iE 'vm fault|amdgpu|kfd' >
<wedge-dmesg.capture>` — W6's reset destroyed the one observation that would have named
branch (E) outright. Add to the PLOG-047/049 playbook.

### 7.7 RED/GREEN rows (closure law)

- RED (pre-fix): leg A numerics RED (U/attn_out) + v-poison touched; leg B canary stomp
  at predicted offset and/or WEDGE rc=2. Cite: cell binary sha + the three `*_simt.cuh`
  shas at the pre-fix commit + this report.
- GREEN (post-fix): legs A/B/C all-clean per 7.1; same three sha classes at the post-fix
  commit. The cells then join the per-window boot battery permanently (device-touching
  class → per-window cell, banked per LABEL).

## 8. Confidence and falsifier

**Confidence: HIGH** on the defect set and the root cause — the hc_base `cs`-vs-`chunk`
arithmetic is line-verified against the mma ground truth and the workspace layout, it is
the ONLY chunks>=2-dependent code in the trio, and the "chunks=1 green" contrast that
would soften it is provably illusory (§1). HIGH that the trio's own control flow cannot
spin/deadlock (§3.3 negative proof). **MEDIUM-HIGH** end-to-end on which memory-system
continuation produced the exact 300 s no-fault signature (fault-halt vs corruption→AR
cascade) — that branch is arena-placement-dependent and is precisely what the cell's
canary-vs-watchdog outcome decides.

**Single device observation that would falsify the root cause:** run pre-fix cell LEG B
(T=128/chunks=2, poisoned slab). If the poison scan comes back CLEAN and the SIMT
outputs match the mma reference at chunks=2 — i.e. no out-of-bounds traffic exists at
chunk 1 — this root cause is dead and the desk re-opens. The confirm-side twin: first
offending poison offset ≈ `h_chunk_start + 24.02 MiB` (= element 12,582,912, the
predicted `(64*12+h_v)*16384` window). Equivalently on the server: applying §4's diffs
and completing the T=128 probe with a `[PREFILL-BODY]` dn line falsifies any residual
"second wedge inside the trio" claim.
