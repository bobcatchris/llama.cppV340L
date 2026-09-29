> **Landed on main 2026-08-26 as docs/70** (QA review + numbering map: docs/77).
> Original: `wo/kvarn-pp` @ 52af56e5, file `docs/69_kvarn_full_matrix_work_order.md` (content as of landing).
> Status as of landing (QA-verified): **CLOSED** — steps A–F + B2 matrix done, regression fixed c289dd06; gates re-checked. Known gap: wide-tier decode path (see docs/77 G-4).
> In-text references to docs/69–74 use the BRANCH numbering: 69=full-matrix(docs/70), 70=prefix-reuse(docs/72), 71=direct-read(docs/73), 72=attn-work-order(docs/74), 73=staged-layout(docs/75), 74=attn-scope(docs/76).

---

# 70 — KVarN full width matrix (K,V independently 2–8 bit): Agent Work Order

**Status:** **CLOSED** — landed on main `fe3d8857` (2026-08-26) as docs/70; steps A–F + B2 matrix done, regression fixed `c289dd06`; gates re-checked. Known gap: wide-tier decode path (docs/77 G-4, H2).
**Mission:** implement the complete KVarN width matrix — K and V cache
independently configurable at 2, 3, 4, 5, 6, 7, and 8 bits (49 meaningful
pairings) — behind per-side flags, with quality validation per tier and zero
regression on existing `kvarn_k4v2` behavior or decode/pp performance.
This generalizes the D-18/D-19 KVarN implementation to the same surface
beellama exposes (`--cache-type-k` / `--cache-type-v`, kvarn2…kvarn8).
Done = all widths selectable at server start; capacity accounting exact for
every pairing; unit gates green for every implemented width pair used by the
tests; decode guard within 2% at the current default (`kvarn_k4v2`); a
quality (KLD/perplexity) table published for at least the ladder tiers;
must-pass battery green.

Read this document fully before writing code. Follows docs/66 → 67 → 68 → 69.

---

## 1. Context (60-second version)

KVarN stores KV pages as variance-normalized quantized records: Hadamard
rotation after RoPE, per-64-key-page scale/zero-point metadata along both
axes, packed integer codes. The current implementation hardcodes exactly one
tier — **K @ 4-bit, V @ 2-bit** (`kvarn_k4v2`):
- Encode: `quantize_tile_kernel<bool IsK>` (src/ops/kvarn/*), two hardcoded
  specializations.
- Decode: nibble extraction in the attention score loops
  (`byte & 0x0F`, two keys/byte) and `kvarn_dequant_v`
  (`byte & 0x03`, four values/byte).
- Metadata layout is **width-independent**: every page carries s_col/zp/s_row
  arrays per dim/key. Only the packed code payload changes with width.

beellama ships the same algorithm with independent K/V widths kvarn2–kvarn8
and publishes a measured quality ladder (Qwen 3.6 27B, RTX 3090, Wikitext-2,
64K ctx): kvarn8/kvarn8 = 52% of bf16 size at near-q8_0 quality; kvarn5/kvarn4
= 33% ("balanced default"); kvarn4/kvarn3 = 27%; kvarn3/kvarn2 = 24%
("emergency"). Our kvarn_k4v2 sits past their smallest recommended tier —
maximum capacity, lowest fidelity. The full matrix lets deployments trade
quality vs capacity deliberately instead of accepting one point.

D-19 (commits through `f76f00bd`) removed the beyond-wall decode cliff and
validated pp at baseline + 200k context. Do not regress those results:
the gates in §5 apply at every step.

---

## 2. Non-goals

- No new attention algorithms. All existing routes (staged shadow, TC small-T,
  split-K direct read, tiled, flash prefill over materialized temp) keep their
  structure; only code unpacking widens.
- No retraining/calibration — KVarN is calibration-free by design.
- No mixed-width *within* one cache side (one K width, one V width per server).
- No per-layer width overrides in this order (SWA-specific pairs are a stretch
  goal, §6).

---

## 3. Architecture of the change

### 3.1 Width-parameterized record format

Introduce `inline constexpr int kKvarnBitsMin = 2, kKvarnBitsMax = 8;` and a
compile-time table:

```
struct KvarnWidth { int bits; };   // per side: K width, V width
```

Code bytes per page per kv head: `ceil(D * G * bits / 8)` where D=256, G=64.
Current: K 4b → 8192 B ✓ matches `kKvarnKCodeBytes`; V 2b → 4096 B ✓
matches `kKvarnVCodeBytes`. Make both computed from the active width instead
of constants. Page geometry (64 keys) is fixed — do NOT vary it.

Metadata floats per page (unchanged): s_col[D], zp_col[D], s_row[G],
zp_row[G] per side, at their existing offsets in `kvarn_scale_pages`.
NOTE: verify whether zp is stored per-dim only for some sides today; if any
offset assumptions differ per side, encode them in one place
(`kvarn_meta_offsets(bits)`).

### 3.2 Templated kernels

Template on `<int KBits>` / `<int VBits>` (or one `KvarnWidths<K,V>` struct):

1. **Encode**: generalize `quantize_tile_kernel` to N levels =
   `1 << bits`. Levels map symmetric-normalized magnitudes; keep the exact
   rounding used today for (4,2) so `kvarn_k4v2` stays bit-identical.
   Odd widths (3,5,6,7): pack MSB-first into `ceil(G*bits*K/8)`-byte rows;
   define the padding rule once and use it everywhere.
2. **Decode, K-score path** (attention kernels: staged TC small-T, split-K
   direct read, tiled, v2 shared): replace the nibble extraction with a
   constexpr unpacker:
   `value(key j, dim ch) = (word >> ((j % keysPerByte) * bits)) & mask`.
   For bits==8 skip unpacking entirely (load bf16/fp8-style direct).
   For bits>=4 consider byte-aligned fast paths (2/4 keys per byte load).
3. **Decode, V path**: `kvarn_dequant_v` → `kvarn_dequant_v<VBits>`
   (same shape as today's 2-bit version).

Instantiate only the widths the configuration can produce (49 pairings would
explode compile time and binary size — see §4 for the dispatch strategy).

### 3.3 Runtime selection & plumbing

- Flags: add `--cache-type-k kvarn{N}` / `--cache-type-v kvarn{N}`
  (N ∈ 2..8). Keep `--kv-dtype kvarn_k4v2` working as an alias that sets
  K=4,V=2 (must remain the default; do not change defaults).
- `KvCacheStorage` enum gains parametric kvarn entries or (preferred) becomes
  `{ family: Kvarn, k_bits, v_bits }` — audit every switch on the enum.
- Capacity accounting (`kv-capacity` resolution, startup VRAM log line) must
  compute bytes/page from the active widths. Add a unit test asserting
  capacity math for all 49 pairings.
- Graph capture / execution envelopes: confirm no shape depends on the old
  constants (grep `kKvarnKCodeBytes|kKvarnVCodeBytes` everywhere).

### 3.4 Dispatch strategy (compile-time explosion control)

Do NOT instantiate 49 kernel pairs. Instantiate per-side widths lazily:
- Attention kernels need `<KBits>` for scores and `<VBits>` for PV — that's
  KBits ∈ {2..8} × VBits ∈ {2..8} = up to 49 instantiations IF fully static.
- Preferred: make the unpackers RUNTIME-branched on width inside a single
  instantiation (`switch (bits)` around the innermost unrolled loops), and
  only fully specialize the hot tiers {2,3,4,5} measured to matter.
  Measure before specializing: if runtime-switch costs <3% on the decode
  guard, ship runtime-only.
- Binary size budget: kernel .rodata/.text growth must stay under ~15 MB
  total; document final numbers in results.

---

## 4. Implementation steps (commit after each; live-verify each gate)

1. **Step A — format refactor, no behavior change.** Replace hardcoded
   constants with computed sizes from a `(4,2)` config; all tests green,
   decode guard within 2%, pp probe unchanged. This commit must be a pure
   no-op — it de-risks everything after.
2. **Step B — templated decode paths.** Unpackers parameterized; instantiate
   (4,2) plus (8,8) first. Unit gates: isolation rel_l2 vs BF16-dequant ref
   ≤5e-3 per width; determinism enforced.
3. **Step C — templated encode.** Quantizer for arbitrary N; round-trip test:
   quantize→dequant→KLD vs bf16 reference per single width, synthetic data +
   real activation dump if available.
4. **Step D — plumbing + capacity math** (§3.3), flag parsing, alias, tests
   for all pairings' size accounting.
5. **Step E — quality ladder.** Publish results table: for ladder tiers
   (8/8, 6/6, 6/5, 5/5, 5/4, 4/4, 4/3, 4/2, 3/3, 3/2, 2/2) run Wikitext-2 or
   equivalent KLD harness at 64k ctx on this hardware. Table goes in
   `results/kvarn_width_ladder.md` mirroring beellama's columns (size vs
   bf16, median KLD, 99.9% KLD).
6. **Step F — performance sweep.** Decode guard + pp probe at default tier;
   spot-check decode tps at (8,8) (expected ≥ current: less unpacking) and
   (2,2) (expected ≤ current). Record in the ladder doc.

---

## 5. Hard gates (every step)

Inherited from docs/68 §5, unchanged:
- Decode guard within 2% at 10k MTP-on / 10k MTP-off, acceptance ±5pp —
  measured on the DEFAULT tier (kvarn_k4v2).
- T19 pp floor ≥450 tok/s at 60k–88k unchanged; no threshold edits anywhere.
- Startup VRAM at 250k unchanged for the default tier (14,635 MiB/rank ± few
  MiB); new-tier VRAM must match computed capacity math exactly.
- In-capacity staged path and I8/BF16 caches bit-identical to current.
- Determinism: repeated identical requests byte-identical outputs.
- One live server at a time; provenance check before any measurement
  (server start time > binary mtime, exe path == worktree build).
- Short test cycles: build (~2 min) → unit gate → single live probe; abort
  and report on anomaly instead of iterating blind.

New gates specific to this order:
- Round-trip property test per width: quantize(dequant(x)) idempotent within
  1 LSB; dequant of stored reference vectors bit-exact vs CPU model.
- Capacity unit test across all 49 pairings.
- Default-tier outputs bit-identical pre/post refactor (step A).

---

## 6. Stretch goals (explicitly optional, do not block Done)

- SWA-layer-specific K/V pairs (`--cache-type-k-swa` etc., beellama parity).
- Mixed standard/KVarN sides (e.g., q8_0 K + kvarn V).
- Runtime width override via API for A/B testing without restart.

---

## 7. Risks

- **Compile time / binary size** from 49 specializations → mitigation §3.4
  (runtime switch, selective specialization).
- **Odd-width packing bugs** (3/5/6/7-bit) → dedicated bit-pattern unit tests
  with known vectors; round-trip property test catches most.
- **Quality surprises at low widths on V** — our current kvarn_k4v2 already
  lives below beellama's recommended floor; the ladder (step E) makes the
  tradeoff visible rather than fixing it.
- **Perf regression on default tier** from generalized inner loops → step A
  no-op contract + guard; specialize (4,2) back if runtime-switch costs >2%.

---

## 8. Deliverables

1. Merged branch `wo/kvarn-matrix` with per-step commits.
2. `results/kvarn_width_ladder.md` — quality + perf table.
3. Updated serve docs (`docs/serving.md`) with flag reference and ladder.
4. CI green including new capacity/determinism tests; pushed to remote.
