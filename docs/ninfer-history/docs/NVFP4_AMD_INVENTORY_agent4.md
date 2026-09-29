# NVFP4-on-AMD inventory (product side) — agent4, 2026-09-14 ~15:2xZ

**Scope, stated so nobody re-derives it:** this is the *product-side* route/capacity/gate map, at
`amd/main` bytes as checked from my lane tip. Artifact-layer recon is **agent5's** and it is already
done well — `docs/amd/NVFP4_AMD_PLAN_agent5.md` @ `amd-wo-nvfp4` `e96a26b0` (worktree present, +1
commit ahead of main, clean). This document does not duplicate §0 of that plan; it **verifies its
cites** and adds the world-math/capacity axis, which is my class after GATE-3.

**Zero build, zero card, zero artifact.** Everything below was checked on source at a tip, and the
artifact does not exist on this box yet.

---
## 1. ROUTE MAP — what `--kv-dtype nvfp4` actually reaches, hop by hop

| hop | site | state at bytes |
|---|---|---|
| CLI name → enum | `include/ninfer/types.h:101` `if (v == "nvfp4") return KvCacheStorage::Nvfp4Group16;` | **REAL** — the enum exists and parses; usage prints it |
| enum → name (round trip) | `types.h:115` | REAL, so an echo-back identity check is possible |
| KV consumer branch | `tp_engine.cpp:839`, `tp2_backend.cpp:735` (`is_nvfp4_kv = ...`) | **REAL but only two call sites** — see §3 M1 |
| request logging | `serve/request_log.cpp:107` | REAL |
| weights profile resolve | `targets/qwen3_6_27b/export/.../package.h:36` + `impl/load/bindings.cpp:44,525` (`Qwen38Nvfp4`) | REAL |
| container geometry | `src/artifact/storage_layouts.cpp:143-149` dispatch rowsplit / **block_scale** / row_scale; `:175` `encoded_bytes` | **REAL, exact-arithmetic** (`checked_mul/checked_add` — refuses loud). Agent5's read confirmed. |
| TP shard of an NVFP4 tensor | `impl/load/tp_load.cpp:141` `nvfp4_shard_image` | **REAL and constraint-validating** — throws on blockscale-violating shape/offset (`dst_rows%128`, `columns%64`, `col0%64`, `col0+columns>full_cols`) |
| device dequant on gfx900 | E2M1/E4M3 SIMT LUT, W4A16 GEMV/GEMM | **ABSENT** — this is the real gap (agent5 G1-G4, N2). Nothing here is inventable product-side. |
| boot gate | `tp2_backend.cpp:1336` `NINFER_ALLOW_NVFP4_TP2` env-present → allow; default fail-closed | **REAL, and see §2 — the name is a trap** |

**Correcting one cite in the plan before anyone greps for it:** agent5's N0.1 names
`package.cpp:109-111` for the profile resolution. **There is no `src/artifact/package.cpp` in this
tree** — `find` confirms absent. The resolution lives in
`src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h:36` (enum) and
`src/targets/qwen3_6_27b/impl/load/bindings.cpp:44,525` (switch). Likewise the header comment at
`types.h:45` cites `src/ops/kv_cache/nvfp4_g16_codec.cuh` and `docs/maintainer/kv-nvfp4-yarn.md`;
**neither exists here** — `src/ops/kv_cache/` is not a directory and `docs/maintainer/` has no such
file. Verdict: those are **NVIDIA-line paths** (this checkout is the AMD line; the docs namespace is
shared and the NVIDIA side has files we don't). Not rot in the enum, and not a bug to file — but
**N0 must be written against the paths in the table above**, or the admission cell will cite
absent files and report a false ABSENT, which is exactly the "negative from a wrong path" failure I
hit three times today.

## 2. THE GATE NAME IS A WORLD=2 ASSUMPTION IN A SHARED PATH — my class, one layer over

`tp2_backend.cpp:1333-1347` gates NVFP4 weights on `NINFER_ALLOW_NVFP4_TP2`. Two product-side
problems, both the lineage I've been closing since A-4:

- **The name bakes a world.** We are minutes from a world=4 boot on the same `TpBackend` ctor. An env
  called `*_TP2` either (a) is checked against `tp_world == 2` somewhere, in which case a world=4
  NVFP4 run is refused by a *name* rather than a finding, or (b) is not checked against world at all,
  in which case it silently authorizes world=4 under a label that says TP2. **Either way the
  audit trail is wrong** — a row citing "gated by `NINFER_ALLOW_NVFP4_TP2`" does not state which
  world it authorized. This is GATE-3's `n_vocab/2` and `tp_group.cpp:164`'s hardcoded `{0,1}` in a
  new costume: a pair assumption living in a path that serves more than two.
- **Env-presence as the authorization mechanism is not a grant.** `getenv(...) != nullptr` accepts
  `NINFER_ALLOW_NVFP4_TP2=0`, `=off`, `=""`. The fail-closed *default* is right; the fail-open *on
  any string* is a footgun that will eventually produce "I didn't mean to enable that." The
  plan's own N7 says the end state is a real `--allow-nvfp4-weights` flag — that is the correct fix
  and it is product-side and zero-build.

**Recommendation (mine to propose, agent5's to review, chair's to route):** rename/extend to a
world-aware form at the same time as the flag flip, and have the refusal message **print the world it
was given and the world it accepted** — the same "name both numbers" rule GATE-2/GATE-3 carry.

## 3. MISSING HOMES (things with no place to live yet, distinct from "not implemented")

- **M1 — RETRACTED. My claim was wrong, and the way it was wrong is the lesson.** I wrote that
  preflight placement is "tier-blind," citing a count: 85 lines (`tp_engine.cpp:913..973`) with ZERO
  mentions of `kv_cache`/`Nvfp4`/storage tier. **The count is accurate. The inference was false.**
  The tier is applied one function UPSTREAM at `:872`:
  `tier_kv_bytes_per_token = tp2::Budget::kv_bytes_for_tier(...)`, and that value is passed INTO the
  placement builder at `:911` — with a comment there stating it is the "SAME BUILDER as the preflight
  budget ... the probe cannot diverge from the preflight again." So the region is silent *because the
  tiering is already resolved*, which is the opposite of absent. Verified further:
  `tp2_budget.h:224 kv_bytes_for_tier` handles `Nvfp4Group16` **explicitly**
  (`nvfp4_kv_bytes_per_token()`), alongside Int8Group64/Int4Group64/kvarn, with BF16 as the
  documented default — so a 4-bit KV has a real home, and it is single-sourced by design.

  **The transferable part:** absence of a token in a window is evidence about the WINDOW, not about
  the FEATURE. I have written "a negative from my own command is not a measurement" four times today
  about malformed greps; this one was a WELL-FORMED grep whose result I over-read, which is harder to
  catch and the reason the fix is to test the claim's CONSEQUENCE (does placement actually size a
  4-bit cache differently? -> read the function the value comes from) rather than the claim's
  evidence. A "measured, not inferred" label does not make an inference measured.

  What survives of the concern, as a small residual rather than a missing home: worth one line in
  N0 is that the BF16 fall-through at `tp2_budget.h` (`return 34ULL * 1024ULL`) is reached by any
  tier not explicitly listed, so a NEW cache enum added without a branch would silently price as
  BF16 -- the same "guard whose default is optimistic" shape as GATE-3's `out_capacity_elems == 0`.
  Whether that deserves a cell is agent5's call as plan owner, not mine to assert.

- **M2 — no admission cell.** N0 as specced (identity → geometry recompute → mixed-region census →
  world-N projection) has no test file today. Host-only, synthetic fixtures, zero artifact: fully
  startable now.
- **M3 — the divisor word has no product-side receipt.** `storage_layouts.cpp` models a 4-byte FP32
  weight-divisor word per NVFP4 tensor and `nvfp4_shard_image` copies it **verbatim to every rank**
  (`// tensor-global word; every rank's copy is...`). Correct for sharding, but nothing asserts the
  per-rank copies are *equal* to each other — a torn rank copy is silent. Cheap invariant, host-side,
  my class.

### NAMED PARKS (chair order 19:17Z: "the park is the task — no work queued against a byte that does not exist")

- **M2 / N0 admission cell: PARKED**, reason = artifact-dependent. The scaffold could be written
  against pure synthetics, but its DECISIVE arms (manifest geometry recompute, mixed-region census)
  assert against NVFP4 artifact byte layouts that do not exist on this host; a cell whose fixtures
  nobody can validate against a shipped object is prose with a `main()`. Re-arms the moment the
  artifact lands (chair's flow). Not counted as open work until then.
- **M3 / divisor-equality invariant: PARKED**, same reason — the invariant is over per-rank copies
  of a word that only a real NVFP4 artifact materializes; synthetic fixtures would test my fixture,
  not the loader. Re-arm trigger identical.
- The host bytes this inventory recorded (route map, cite corrections, the `NINFER_ALLOW_NVFP4_TP2`
  world-name finding, the /2-decoy distinction) stay LANDED — they were derived from shipped
  source and need no artifact to be true.

## 4. WHAT I CAN START TODAY — no artifact, no build, no card

1. **~~Capacity/world-math map for a 4-bit KV tier~~ SUPERSEDED BY M1's RETRACTION** — the home
   exists (`kv_bytes_for_tier`). Replacement work for this slot, smaller and real: a **host cell
   pinning the tier->bytes identity for NVFP4 KV** (so a future enum cannot fall through to the BF16
   default silently), which is a 10-line addition on the `gather_capacity.h` pattern, not a map. VRAM-law constraint on myself: **the output may
   only ever inform a report; it must never refuse a launch.** Placement math that can say "won't
   fit" is the banned class regardless of how it's derived — allocator is the gate.
   **SHIPPED (chair 19:17Z re-task, this session): `tools/ops/check_kv_tier_priced.py`** — TWO
   paired arms, five+two directions observed at seat via built-in `--selftest`/`--falsify`:
   (a) PRICING arm — every `KvCacheStorage` enumerator (single source: the ENUM in types.h, not any
   hand-list) must have an explicit `kv_bytes_for_tier`/`kvarn_tier_widths` arm or BE the named
   BF16 tier; planted-PhantomTier RED, current-tree GREEN (7/7 priced), unreadable rc=2. Live
   measured discriminator that the class was genuinely unguarded: `tests/test_tp2_budget.cpp:247`
   iterates a hand-maintained six-name list while the enum has SEVEN members — `KvarnK4V4` is in
   the enum, absent from the matrix list; today it is priced (types.h:67 widths{4,4}) so this is a
   class guard with NO live instance, exactly as the ring-guard shipped. The existing matrix
   cannot catch a tier forgotten in BOTH places; this cell can.
   (b) TABLE PAIR arm (`--table`) — grades agent5's generated `tier_shape_table` emission against
   the WO-TP4-F documented row-key schema: every schema tier must carry a row or a NAMED LOUD
   REFUSAL (the fold's law), NVFP4 may not receive silent geometry without a gfx900 route proof,
   and a non-C++ subject or unparseable emission is rc=2 SCHEMA-HANDSHAKE not a verdict. Two
   instrument bugs were caught by RUNNING it, both directions banked in the selftest: a
   comment-stripping version FALSE-REDed a legal comment-form named refusal (v7 over-conviction
   shape), and a prose-scan version GREENed this checker's own source (false-acquittal cousin —
   both now fixtures). c587965f is not yet fetchable from any local ref, so the first LIVE table
   run rides the moment agent5's sha reaches origin; rc=2 there = parser meets real syntax, the
   handshake is designed for it. Zero card, zero build, zero grant.
2. **N0 admission cell scaffold with synthetic fixtures** (M2): geometry recompute vs manifest,
   corrupted-divisor and off-by-one-offset as the **RED positive controls**, honest fixture GREEN.
   Both directions must be observed at my seat per law-C, named in the receipt.
3. **Per-rank divisor equality invariant** (M3).
4. **Gate-language fix proposal** for §2, as a patch for whoever owns `tp2_backend.cpp`.

**Not mine:** codec host references / test lane = **GEMINI** (untouched, per the standing rule);
SIMT device kernels = agent3 (N2); loader/codec registration = agent2 (N1).

## 5. TP4 STILL COMES FIRST
This inventory is parked-work-while-blocked, not a hand-off. Runner + playbook are current at
`ac941bff` (F6 + STEP-0/F landed), and I stay TP4-hot: step-0 refresh at agent5's fix sha, and the
playbook branches ready if attempt #2 dies differently. **If agent5 stalls, I revert to TP4
immediately** — the runner/playbook context is mine and nobody else has it.
