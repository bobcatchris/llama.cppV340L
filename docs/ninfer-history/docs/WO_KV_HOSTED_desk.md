# WO_KV_HOSTED_desk.md — hosted-KV mechanism design + integration gate (W7, Team Red)

**Mission: price and prove a system-DRAM-hosted KV cache (or cold portion) for the V340L line —
OFF BY DEFAULT (`NINFER_KV_HOSTED=1`), prefill-centric. This doc pre-registers the Phase 2
integration gate and the standalone cell's self-bar BEFORE the cell runs (2026-09-19 05:45 CDT).**
Companion checkpoint log: docs/amd/W7_KVHOSTED_desk.md.

## 0. Ground truth: how KV lives today (mapped 2026-09-19, worktree amd/wo-w7-body)

- Target qwen3_6_27b (serving line qwen3_8_27b_nvfp4 @ TP4): 64 layers hybrid = **16 full-attention**
  + 48 GDN linear (no paged KV) + 1 MTP attention layer. Full-attn: kv_heads=4, head_dim=256.
- **Allocator:** ONE persistent `DeviceArena` per rank (`program_impl.h:271`:
  `backing = persistent.alloc_bytes(plan.persistent.bytes, 256)`); all KV pool planes bind from
  it. **KV is 100% VRAM-resident today**, sized at plan time from capacity — no page faulting,
  no host tier.
- **Layout** (`src/core/paged_kv_cache.h`, `decoder_state.cpp plan_cache`): paged, **page = 64
  tokens**; KVarN tiers = 2 U8 code planes/layer (K channel-major `D*kb/8` B/token,
  V token-major `D*vb/8` B/token) + fp32 scale side-table `[layers, heads, pages, 1152]`
  (4,608 B/tile = 72 B/token/head).
- **Append (prefill):** `gqa_kv_append_kvarn[_and_commit]` (`src/ops/kvarn/kvarn_workspace.cpp`)
  → scatter bf16 tile → per-(layer,head,page) quantize into plane slabs + scale tile → commit
  via block_table.
- **Read (attention):** `gqa_attention_kvarn*.cuh` (prefill), `gqa_decode_slice4_kvarn.cuh` /
  `gqa_decode_slice6_kvarn_k5v4.cuh` (decode) read U8 codes + 1152-float scale tiles through
  `PagedKVLayerView` via block_table; dequant in-kernel.
- **Existing host affordance:** `PagedKVPool::copy_pages_to_host / copy_pages_from_host`
  (docs/156 §18 P2 host-KV safety net) — page-major host mirror, PageMajor pools only. Any
  hosted-KV design builds on this or alongside it.

## 1. KV budget table (kvarn k4v2 unless stated; MEASURED geometry, not estimates)

Per token per full-attn layer per head: K 128 B (256ch × 4b) + V 64 B (256ch × 2b) + scales 72 B
= **264 B**. Scales share: 27% of bytes — a hosted design must move scale tiles WITH their code
pages (the pool's page-major host mirror already concatenates plane slabs per page; the kvarn
scale side-table must ride the same granularity).

| tier | B/token/head/layer | text KV B/token (4h × 16L) | +MTP (1L) | total paged B/token |
|---|---|---|---|---|
| kvarn k4v2 | 264 | 16,896 | +1,056 | **17,952** |
| kvarn k5v4 | 320+72+... ≈ 392* | 25,088 | +1,568 | 26,656 |
| bf16 | 1,024 | 65,536 | +4,096 | 69,632 |
*estimate from widths; k4v2 row is exact.

**Total KV (all 4 ranks summed) at k4v2, text+MTP = 17,952 B/token:**

| context | text KV | +MTP total | per-rank @TP4 (1 kv head/rank, text) |
|---|---|---|---|
| 10k | 169.0 MB | 179.5 MB | 42.2 MB |
| 50k | 844.8 MB | 897.6 MB | 211.2 MB |
| 100k | 1.690 GB | 1.795 GB | 422.4 MB |
| 200k | 3.379 GB | 3.590 GB | 844.8 MB |
| 500k | 8.448 GB | 8.976 GB | 2.112 GB |

**VRAM fit today:** per-die HBM 16.4 GB; NVFP4 weights + workspace (96 MiB) + graph allowances
leave roughly 10-12 GB/die for KV+transients (the 2026-09-10 int8 incident: 16,679 MiB claimed,
15,069 ran). k4v2@TP4 puts 200k ctx at only ~0.9 GB/die of KV — the binding pressure at high
context is bf16/nvfp4 KV tiers, concurrency, and TP2-class worlds (2 kv heads/rank → 2×),
NOT k4v2 single-batch. Hosted-KV's wins, concretely:
(i) **beyond-VRAM prefill** — contexts whose KV (+weights+workspace) exceed the die at ANY tier;
(ii) **headroom** — free die GBs at 100k-500k for batch, MTP, or larger prefill chunks;
(iii) it is purely capacity-ADDING — never a refusal path (VRAM LAW).

## 2. The three mechanisms, priced on THIS box's measured numbers

Measured/quoted anchors: pinned PCIe ~10-12 GB/s class; SHM-staged relay **3.14-3.17 GiB/s**
(W7_ksplit log, `canAccess=0` on ALL pairs both directions); device HBM ~full BW (V340 spec
class ~484 GB/s; the cell measures actual). Per-rank k4v2 = 4,224 B/context-token (text).

Common math used below, per rank @TP4 k4v2:
- **Prefill chunk read:** a chunk of S=2048 tokens attending over C prior tokens reads
  C × 4,224 B. Largest read at 100k ctx = 422 MB; at 2k prefill-chunk wall ~1.5-2 s class on
  this box, a 422 MB PCIe read at 11 GB/s = **38 ms ≈ 2-2.5% of one chunk's wall, once** — and
  only the FINAL chunk pays the full C; the average chunk pays ~half.
- **Decode tax (the honest number):** decode is bandwidth-random over ALL KV every token.
  Hosting a cold tail of C tokens costs, per generated token, per rank:
  `C × 4,224 B / BW`. At pinned 11 GB/s: C=10k → **3.8 ms/tok**; C=50k → **19.2 ms/tok**;
  C=100k → **38.4 ms/tok**; C=200k → **76.8 ms/tok** (≈13 tok/s ceiling — from ~150+ tok/s
  device-resident). At SHM-staged 3.38 GB/s these triple (~125 ms/tok @100k). Device-resident
  100k read ≈ 1 ms. **The tax is ~35-40× at PCIe and ~120× at SHM, and it is UNAVOIDABLE in
  every mechanism** — once KV must live in DRAM, each decode token re-reads it. Prefetch/paging
  games do not help decode: there is no reuse window to hide a full-context re-read into.

**(a) Mapped-pinned zero-copy cold KV.** Cold pages live in `hipHostMallocMapped` host memory;
kernels receive host-backed device pointers for cold pages, device pointers for hot ones.
Prefill's sequential page walk streams at pinned PCIe BW with hardware read-ahead — no staging
buffer, no copy orchestration, no second code path beyond pointer selection. Decode pays the
full tax above. Integration surface: allocator carve-out (host slab) + block-table-side flag or
pointer table + kernels take `const U8*` either way (already pointer-driven).
Cost: decode tax (full, unavoidable); pointer-indirection in kvarn kernels (small, arm-gated).

**(b) Page-spill + streaming prefetch.** Cold pages in a pinned DRAM pool; before chunk i,
async H2D copies the pages chunk i touches, overlapping chunk i-1's compute. Same decode tax
as (a) — the cold set must be re-read every token whether it is mapped (a) or staged (b);
(b) additionally cannot keep the whole cold set device-resident (that's the premise), so decode
either pays per-token full re-streams (identical to (a), minus zero-copy's simplicity) or
fails beyond-VRAM decode entirely. Extra machinery vs (a): pinned staging pool sized to the
prefetch window, double-buffer state machine, per-chunk copy scheduling, policy for eviction.
Cost: decode tax (same) + a state machine whose only benefit over (a) is hiding copy launch
latency that (a) doesn't have in the first place.

**(c) Allocator overflow (spill on VRAM pressure).** Identical substrate to (a) (mapped-pinned
host pages), but pages land in DRAM only when the device arena hits pressure. It is (a) with a
lazy policy. Strictly it makes access patterns WORSE for prefill (cold-ness discovered at
allocation time, not temperature time) and adds a pressure sensor to the allocator — exactly
the kind of live-threshold logic that must stay a cudaMemGetInfo compare, never a constant.

### Recommendation: **(a) mapped-pinned zero-copy cold-KV tail**, optionally adopting (c)'s
"only host what needn't be in VRAM" as the default *policy* (host the cold prefix beyond a
keep-in-VRAM window; env `NINFER_KV_HOSTED=1` + window knob). Rationale:
1. Prefill-centric: prefill reads are sequential, compute-dominated (38 ms transfer vs
   1.5-2 s compute per 2k chunk at 100k — ≤2.5% even UNHID; ≈0 with read-ahead/overlap).
2. Smallest diff and fewest moving parts of the three; kernels stay pointer-driven; no
   prefetch state machine; reuses the docs/156 page-major host layout thinking.
3. Decode tax is IDENTICAL across (a)/(b)/(c) beyond VRAM — the tax cannot be engineered away,
   only measured and documented (gate requires it at 600+ generated).
4. (b) is dominated: it pays (a)'s decode tax plus machinery to hide a cost (a) doesn't incur.
   (c) is (a) with a policy — adopt the policy, skip the separate mechanism.

**Decode tax by mechanism, stated plainly:**
- (a): full tax on the hosted tail, every decode token (table above). Prefill: ≤2.5% wall at
  100k unhid, ~0 with streaming read-ahead. RECOMMENDED.
- (b): identical decode tax; prefill ≈ (a) with more code. Not chosen.
- (c): decode tax proportional to spilled fraction (worst case = (a), best = 0). Same
  substrate; adopted only as the sizing policy, not as a distinct mechanism.

## 3. Standalone parity+perf cell — spec (the mechanism decision data)

File: `tools/v340l/w7_kvhosted_cell.cu` (standalone hipcc, gfx900, NO tree build).
Build: `/opt/rocm/lib/llvm/bin/clang++ -x hip --offload-arch=gfx900 -O3 -std=gnu++20
tools/v340l/w7_kvhosted_cell.cu -o <tmp>/w7_kvhosted_cell` (pattern: tools/v340l/*.cu).
Run pinned to a free die: `HIP_VISIBLE_DEVICES=2` (microbench law) + clocks note
(rocm-smi --showclocks/--showtemp before and after; with every perf number).

- Slab: N (layer,head,page) kvarn-k4v2 tiles of 16,896 B (K 8,192 U8 + V 4,096 U8 + scales
  1,152 fp32), deterministically patterned (LCG from tile id — every byte exercised).
- Two allocations: device slab (`hipMalloc`) vs host slab (`hipHostMallocMapped` +
  `hipHostGetDevicePointer`). Same bytes both sides.
- Attention-shaped read kernel: grid over tiles; each block streams its K codes, V codes, and
  scale tile with position-dependent weighting (mixes EVERY byte, memory-bound), reduces into
  an output word per tile.
- Arms: (1) bit-parity — kernel output over device slab vs host slab must be WORD-IDENTICAL;
  plus raw `hipMemcpy` slab compare; (2) full-sweep timing at slab sizes 10/50/100/200 MB,
  both backs, GB/s each; (3) bonus probe: async pinned→device copy overlapping a dummy compute
  kernel (evidences the overlap premise both mechanisms lean on).

### SELF-BAR (pre-registered 2026-09-19 05:45 CDT, BEFORE first cell run)
The mechanism clears its own bar, and Phase 2 may start (given its other preconditions), iff:
1. **Parity (mandatory):** hosted-mapped vs device kernel outputs bit-identical at every size;
   raw slab memcmp equal. Any mismatch = NO GO, regardless of speed.
2. **Hosted read BW:** sustained full-sweep read BW ≥ **6.0 GB/s** at the 200 MB working set
   (~half the pinned-PCIe class floor; below this even unhid prefill tax stops being ≈2% and
   the streaming-read-ahead premise dies).
3. **Prefill tax sanity from measured numbers:** a 2k-token chunk's cold read per rank
   (8.65 MB) at measured hosted BW must be ≤ 5% of the measured 2k prefill chunk wall
   (~1.5-2 s class on this box) — expected to clear by two orders.
4. Device-arm numbers are RECORDED as reference (no floor beyond ≫ hosted, sanity only).
Decode-tax acceptability is NOT a cell bar: it is a documented cost, measured end-to-end at
the integration gate (600+ generated). The cell measures its BW component.

## 4. Integration gate (pre-registered for Phase 2; Phase 2 starts only after ALL of:
V-arm desk has landed its build + window free; cell bar §3 cleared; build-claim protocol run)

- **Env arm:** `NINFER_KV_HOSTED=1` enables the hosted cold-tail; **unset = byte-identical
  current path** (no new allocations, no layout change, no behavior change — verified by
  boot-log diff on a fixed request).
- **Capacity-adds-only:** the arm may never refuse a launch; with the arm OFF, behavior is
  exactly today's allocator outcome (VRAM LAW; hosted KV ADDS capacity).
- **RED→GREEN (per the closure law):** RED = pre-integration bin booted with
  `NINFER_KV_HOSTED=1` shows ZERO hosted bytes and unchanged VRAM reservation (env unknown →
  dead arm — captured as the red row); GREEN = post-integration bin, same env, hosted bytes
  > 0, device KV reservation drops by the hosted size (live-measured baseline diff), and
  hosted-vs-device parity is bit-exact on patterned data (cell §3 row + serve fixed-prompt
  output-id equality). Shas named for both runs.
- **Serve-leg A/B, ordinal-paired** (PLOG-060 design): same prompt set, same order, OFF vs ON;
  per-request wall-time ordinal pairing; boot battery per runbook.
- **Decode tax measured at 600+ generated tokens** with the arm ON (documentation, not a bar).
- **Context-length sweep proving the VRAM win:** 10k/50k/100k/200k (+1 beyond-VRAM point)
  with measured per-boot device reservation, OFF vs ON; the beyond-VRAM point must
  cudaMalloc-fail OFF and boot+prefill ON — the "beyond-VRAM context" proof.
- Bank bin per BANK-BEFORE-RELINK: `/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin`;
  PLOG row via `tools/guards/plog_append.py` (chain head 0d4c30c3272d3092); serving retire =
  EXACTLY `pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"`; restore via
  `bash /home/chris/serve_10k.sh` + health-verify at window close; serving window claimed
  against docs/amd/W7_VARM_INTEGRATION_desk.md and docs/amd/W7_ROUNDWALL_desk.md.

---

## 5. CELL RESULTS (2026-09-19 05:50 CDT, run1 — bar was pre-registered above BEFORE this run)

Banked: results/amd/coherence/W7_kvhosted_cell{.cu,_run1.log}, W7_kvhosted_row.txt.
GPU die 2, HIP_VISIBLE_DEVICES=2; clocks: mclk 945 MHz throughout, sclk sampled at 1500 MHz
(level 7 TOP) mid-run; 40-47 C. gfx900:xnack-, unified pointer (host==devptr).

| size | dev_GB/s (byte/vec) | hosted_GB/s (byte/vec) | tok-equiv/rank |
|---|---|---|---|
| 10 MiB | 134.4 / 306.4 | 6.38 / 6.44 | 2,480 |
| 50 MiB | 178.8 / 300.4 | 6.34 / 6.48 | 12,412 |
| 100 MiB | 191.0 / 342.9 | 6.33 / 6.49 | 24,824 |
| 200 MiB | 199.0 / 340.8 | 6.33 / 6.50 | 49,648 |

- **PARITY: bit-exact everywhere** — kernel outputs word-identical device-vs-hosted at all four
  sizes, both kernel shapes, plus raw slab memcmp equal. GREEN.
- **Hosted BW 6.5 GB/s, load-width-insensitive → link-bound** (dev arm: byte 134-199 vs vec
  300-343 GB/s shows width mattered for HBM, not for PCIe — the hosted arm is at the link).
- **Bar verdict: GO** (parity PASS; 6.50 ≥ 6.0 PASS; prefill tax 1.30 ms/2k-chunk ≈ 0.08% of a
  ~1.5-2 s chunk wall, PASS ~60× under the 5% bar; full 100k cold sweep 422 MB = 65 ms ≈ 3-4%
  of one chunk wall UNHID).
- **Measured decode-tax unit: 32.25 ms per generated token per 200 MiB/rank hosted** (~0.65 ms
  per 1k hosted tokens; C=100k → ~65 ms/token; ~25-50× the device-resident read). Documented
  price, gated at 600+ generated in Phase 2.
- **Overlap probe measured FALSE on this stack** (21.8 ms concurrent vs 22.3 ms sum): pinned
  async H2D does NOT overlap compute on Vega 10 HIP → mechanism (b)'s prefetch premise is not
  just dominated, it is UNMEASURED-FALSE here. (a) needs no copy concurrency: read-ahead lives
  inside the kernel's own memory pipeline.

**§2 recommendation stands on measured data: (a) mapped-pinned zero-copy cold-KV tail.**

## 6. PHASE 2 STATUS: NOT STARTED (preconditions unmet)

- W7_VARM_INTEGRATION_desk.md does not exist anywhere (checked 05:28 and again at cell close)
  → no landed V-arm build to piggyback, no serving-window counterpart to claim against.
- Cell bar IS cleared (§5), so the desk's Phase 2 obligation reduces to: wait for the V-arm
  build + free window, then execute §4's pre-registered gate (env arm, capacity-adds-only,
  RED/GREEN rows, ordinal A/B, 600+ decode-tax leg, context sweep, BANK-BEFORE-RELINK,
  PLOG row chain 0d4c30c3272d3092).
