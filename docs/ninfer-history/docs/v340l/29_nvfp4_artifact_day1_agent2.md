# NVFP4 ARTIFACT DAY-1 — real-bytes audit receipt (agent2, chair seq-152 leg 2)

**Artifact:** `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer` — **18,324,067,840 B** (grew live
during this beat, 2.45 → 6.6 → 18.32 GB; stable across 60 s × 3 samples).
**SHA-256 PINNED FULL 64-HEX at my seat** (2 m 04 s read pass):
`eaf8ad124256d0a0c1ebbbca442ca58eee4f97ab34a60a0b4d57e2b41e2c56d2`. Chair's expected
`eaf8ad12…56d2` — **prefix `eaf8ad12` MATCH, suffix `56d2` MATCH — VERIFIED against expected**
(the desk's first full-hash record; from this beat the artifact's identity stamp is the 64-hex
above and any shorter cite is a content8-style prefix, name the namespace).
Zero card, host-only reads. The file itself is the new namespace: **A-row "stamps" for the
ARTIFACT = path + this full sha; NOT git-resolvable, deliberately.**

## A-ROW RESULTS AT REAL BYTES (manifest header + divisor probes; A5/A6 device legs stay hour-2)

| row | expected datum | measured HERE | verdict |
|---|---|---|---|
| **A1** | `Reader(real_file)` opens, identity parses; "the file is real v2, identity qwen3.8-27b/nvfp4" | magic `NINFER` + JSON header parses standalone (python raw_decode — no runtime needed for the header); identity = `{"model_id":"qwen3.8-27b","weights_id":"nvfp4"}` EXACT; 1307 objects; size arithmetic closes: highest tensor extent 18,323,858,944 + 208,896 B header = file size exactly | **PASS (header-level)** — the C++ Reader open is the boot-battery's leg now that identity+geometry close externally |
| **A2** | profile resolves `Qwen38Nvfp4`; name-diff control GREEN | resolve arm SHIPPED at amd/main already: package.cpp:108-110 (`qwen3_8_model_id` + `"nvfp4"` → `WeightsProfile::Qwen38Nvfp4`, enum split-form package.h:32-37, consumed bindings.cpp:44/:522, gated tp2_backend.cpp:1340) — see STALE-PATCH section: the relayed `/tmp/nvfp4_support.patch` is NVIDIA-line-era bytes and MUST NOT walk the chain, the support already landed | **PASS at bytes; runtime resolve = boot-battery leg** |
| **A3** | geometry duel-close, EXACT compare, zero slack | **247/247 NVFP4 tensors**: law-derived `encoded_bytes` (el/2 code + align256 + el/16 scale + 4-byte divisor slot) == manifest bytes EVERY tensor, 0 mismatch; all 247 satisfy N%128==0 && K%64==0 (admission refuses none); every tensor offset 256-aligned | **PASS — the plan's stated first-RED stop-condition did NOT fire: exporter lineage == layouts.py law at real shapes** |
| **A4** | mixed-region census vs manifest-derived | census TABLE: 247 NVFP4 / 591 BF16 / 343 FP32 / 55 Q4G64_F16S / 54 Q5G64_F16S / 9 W8G32_F16S / 1 Q6G64_F16S / 1 I32 / 6 resource (no format). **NVFP4 divisor pairing: 247 in-tensor words + 247 separate `input_scale_divisor` objects = EXACT 1:1, zero unpaired.** Divisor-value census (real bytes, LE u32→f32): weight words min 2.22e8 max 2.91e29, input objects min 8.74e8 max 1.10e27, **0 violations of require_positive_finite, 0 subnormals**. Distinct NVFP4 shapes: {5120×6144, 5120×17408, 14336×5120, 16384×5120, 34816×5120} — fused-vs-split hazard RESOLVED toward fused: `gdn/query_key_value_z [16384,5120]` is real. NOTE **zero FP8_E4M3FN_ROW_BF16S tensors** — the plan's "{NVFP4 rows, FP8 rows…}" mixed-region expectation gets its FP8 count from the manifest as 0; the W8 vocabulary in this file is W8G32_F16S (9 rows: embedding/head), a different enum. §8's A4 expectation text needs the 0-FP8 correction at agent5's desk (annotate, not delete) | **PASS with one expectation-text correction** |
| **A5** | world-2/4 shard projection, reassemble full, divisors verbatim per-rank-equal | hour-2: needs `nvfp4_shard_image` in the HIP build (device grant) — but the HOST-SIDE twin is now un-phantomed: my python re-derivation of per-tensor geometry (A3 leg) is the rank-0 full-image truth the shard test compares against | **NOT-YET-RUN, prereq named; no longer linkability-blocked** (see doc 28 retraction) |
| **A6** | divisor-word presence per NVFP4 tensor through real bindings path | presence + value + boundary legs MEASURED above at real offsets (247 in-tensor words at their geometry-declared `weight_divisor_offset`, all read positive-finite; 247 sibling objects same). The BIND-TIME leg (real `bind_nvfp4_weight` execution) rides the boot battery | **PARTIAL-PASS at bytes; runtime leg cued** |
| **A7** | pricing cell over real tier set, report-only | real tier set = {NVFP4, BF16, W8G32_F16S, Q4/Q5/Q6G64_F16S, FP32, I32} cache-side enums unaffected — check_kv_tier_priced ran GREEN at wo-p3-serve tip a32b4883 (doc 28 beat); still ABSENT from main → still a land-with-join row. Report-only per VRAM law, never a refusal | **PASS at lane bytes; registration debt open** |

## FALSE ALARM, CAUGHT BEFORE FILING (the receipt's honest half)

My first pairing probe printed **247 "PAIR BYTE MISMATCH"** — I did NOT report it until re-deriving
the shipping law: `bind_nvfp4_weight` (bindings.cpp:82-103) reads TWO DISJOINT quantities — the
in-tensor WEIGHT divisor at `geometry.weight_divisor_offset` and a SEPARATE INPUT-scale object —
`weight_scale_divisor_bits` vs `input_scale_divisor_bits`, both validated independently, NO
byte-equality between them by design. My expectation of equality was the wrong model — the
mismatch count was **my probe's phantom, not the artifact's**. The datum that SURVIVES the
re-derivation is the real one: 1:1 presence pairing (247/247) + both value classes finite-positive
+ exact geometry offsets (A3 closes ⇒ the divisor slot is the LAST 4 bytes of each encoded image,
structurally). Filed so the next grep-head doesn't "fix" the artifact for a law it misread.

## STALE-PATCH FLAG (sent urgent 21:0xZ, repeated for the ledger)

`/tmp/nvfp4_support.patch` = 17 lines against base blob `8369f4f` = pre-image of **5d2c1f55**
(2026-08-17, NVIDIA-line `origin/main`) returning `WeightsProfile::Nvfp4` — a token that does
NOT exist in amd/main's split-enum world (`git grep -c "WeightsProfile::Nvfp4\b"` = the arm at
:105-106 uses `Qwen36Nvfp4`; zero hits for the bare name). The patch's FUNCTION is SUPERSEDED
BY LANDING at amd/main; applying it = dead-token break. Chain action: mark superseded, do not
merge. Name-diff control intact: profiles are per-model typed enums; 3.6/3.8 cross-resolution
cannot compile.

## RE-PINNED RUNBOOK STATE (the chair's "flag phantom-pins" ask, final form)

- **ON MAIN c218547f (resolvable, verified-shipped):** 0fe01f38, 20eb99a7, 81aced51, 41d5e020,
  8deaa216, 1fa3c219 — plus in-tree corpus a024627b at `tests/multi_gpu/nvfp4_shard_fixture.h`.
- **STILL OFF-MAIN (named cures, no phantoms):** af31aecd synth cell (desk-only — ship or re-cite
  A1/A2 to main's admission+corpus pair); check_kv_tier_priced e6db3027@656aa7f6 / tip a32b4883
  (wo-p3-serve-only — land with join); 143650e4 export consumer (correctly desk-only, torch row);
  bccdcb39 (lineage-only, resolves nowhere by design).
- **NEW NAMESPACE:** the artifact — `qwen3_8_27b_nvfp4.ninfer` @
  `eaf8ad124256d0a0c1ebbbca442ca58eee4f97ab34a60a0b4d57e2b41e2c56d2` (path + full sha, external
  drive, not in git BY DESIGN — pin rows against THIS, verify by 3-probe: ls size, sha pass,
  identity header read; all three done this beat).

— agent2, zero-card, zero build in shared tree; reads were sequential-seek host I/O only;
no processes left, no worktree touched.
