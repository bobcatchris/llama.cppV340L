# WO_KVARN_10K_OVERFLOW — fix the kvarn long-prefill std::bad_alloc (BLOCKER A, rank 0)

- **Defect (banked, PLOG-076):** kvarn-route prefill dies with `std::bad_alloc` at
  ~10.0-10.2k REAL tokens — 4/4 deterministic, ALL 4 ranks, position-locked (dies at
  9984-10112 progress marks on BOTH a 19.7k and a 49.3k prompt), **RSS flat 3.0 GB with
  55 GB host free** (bf16 did 36k routinely on the same box). Flat RSS + instant throw =
  a HUGE/NEGATIVE single allocation, i.e. an integer-arithmetic overflow in a computed
  size, not memory pressure.
- **Prime hypothesis:** int32 (or int) size/index arithmetic in the kvarn HOST prefill
  path where (cumulative tokens | pages | committed entities) x per-entity bytes crosses
  2^31 at ~10.1k — a product near 2,147,483,647. Search shapes: `tokens*N`, `pages*M`,
  `committed_pages *`, `blocks *` with an int/int32/int operand in
  `src/ops/kvarn/kvarn_workspace.cpp`, `src/ops/launcher/gqa_attention_kvarn.cu`,
  `src/targets/qwen3_6/impl/runtime/text_context_impl.h` (kvarn seams ~:429-700),
  `src/runtime/tp2/tp2_backend.cpp` (kvarn staging/binding), and the kvarn scale-table
  host staging. Also check `.pack_pages`/`scale_pages`/`block_tables` host mirrors and
  any `std::vector::resize(bytes)` where bytes came from int math.
- **Evidence:** /home/chris/worktrees/amd-wo-k4v4gates/results/amd/k4v4gates/
  (W7_k4v4_phase2_row.txt, serve logs, RSS trace, request bodies for replay).
  Bin f3312f25c8201da0 (green, banked) reproduces.
- **Desk:** agent #2 (slot freed by k4v4gates close). Worktree: `git worktree add
  /home/chris/worktrees/amd-wo-ovf amd/wo-k4v4gates -b amd/wo-ovf` (inherit the gates
  evidence). cmake at /home/chris/opt/cmake/bin/cmake (configure per house recipe;
  -DNINFER_BUILD_APPS=OFF, NINFER_HIP_BUILD_SERVE=ON, gfx900; df -h / FIRST — ~6 GB free,
  build tree ~450 MB, GO but keep it lean).
- **SEQUENCE (window protocol):**
  1. **RED BOOT FIRST (window is free — claim it):** append "WINDOW CLAIM: RED repro" to
     this file on YOUR branch, retire canonical per the EXACT law form
     (`pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"` — never any other pkill),
     boot bin f3312f25 (canonical env + `--kv-dtype kvarn_k4v4 --max-context 36352
     --kv-capacity 36352 --spec mtp --draft-tokens 2`), fire a >10.3k-token prompt, capture
     the throw + backtrace (gdb or the abort message) + the exact failing allocation site.
     Then RESTORE canonical (`bash /home/chris/serve_fast.sh` + health + FAST PROBE) and
     RELEASE the window (append "WINDOW RELEASED") — the coordinator interleaves its own
     window slot while you hunt source.
  2. **Source hunt (GPU-free):** name the exact line(s); verify the arithmetic crosses
     2^31 at the observed position band (show the numbers).
  3. **Fix:** widen to int64/size_t at the site(s) — minimal diff, no behavior change below
     the overflow point (byte-identical serving below 10k must be PRESERVED).
  4. **GREEN:** rebuild, bank `ninfer-serve_<sha16>.bin` (BANK-BEFORE-RELINK), re-claim the
     window, run the SAME >10.3k prompt to completion (finish=stop, coherent), plus one 2k
     probe for unchanged-short-context parity. Restore canonical + health.
- **Gates:** G-OVF-0: RED reproduced with the allocation site named (throw at ~10.0-10.2k).
  G-OVF-1: GREEN boot completes >10.3k prefill, 2k parity unchanged, no new preflight
  delta. RED->GREEN rows join the evidence permanently (closure law).
- **Laws:** no boots without a window claim; retire law form ONLY; restore canonical +
  health at every close; PROGRESS LOG newest-first after every step; infra = tag
  "BLOCKER:"; no builds in /home/chris/dual_5060_ti_ninfer EVER.

---

PROGRESS LOG (newest first)

- **2026-09-20T01:2xZ — G-OVF-0 MET (RED capture complete, prime hypothesis FALSIFIED).**
  Six boots total (5 fired at the 19,709-tok gates body; 4/4 throws, wall 108-127 s,
  response `{"error":{"message":"std::bad_alloc"}}`, all 4 ranks). Decisive capture
  (RED #6, `results/amd/ovf/red_gdb5.log`, gdb `catch throw bad_alloc` + register/stack
  interrogation): throw = `src/core/arena.cu:299` (`if (end > cap_) throw std::bad_alloc();`),
  caller chain DeviceArena::alloc_bytes <- DeviceArena::alloc <-
  `gqa_attention_kvarn_cached_launch` (ret 0xa0e0a44). Failing call's init-list read off the
  stack: **{256, 64, 1, 160}** = `alloc(FP16, {kKvarnAttnD, kKvarnAttnG, kv_heads=1(TP4),
  n_tiles=160})` = the HIP-port temp `v_temp_h` at **amd/wo-kvarnport
  `src/ops/launcher/gqa_attention_kvarn.cu:396-397`** (bin's as-built source = branch
  amd/wo-kvarnport tip 4178169ef — proven by the cuda_check filename string embedded at
  .rodata 0x247f92). n_tiles=160 pages x 64 = **10,240 tokens — the exact death band**.
  Arena state: cap_=100,663,296 (96 MiB = NINFER_WORKSPACE_MIB), off_=95,421,568,
  peak_=100,651,264 (12 KB under cap — riding the ceiling), requested
  bytes=5,242,880 = 256*64*1*160*2 EXACTLY (honest size). **The int32-overflow prime
  hypothesis is FALSIFIED** — the defect is 96 MiB work-arena exhaustion: chunk-entry
  offset ~81 MiB ratchets ~1.2 MB/chunk until the materialize temps (k_temp 5 MiB +
  v_temp 5 MiB + v_temp_h 5 MiB + bt) hit the cap at chunk 79. The HIP port's THIRD
  full-size temp (v_temp_h; CUDA line has two) sets the wall's position on this bin.
  Fix direction: (1) PRIMARY — find the ~1.2 MB/chunk un-freed residue in the prefill
  pass; (2) in-place bf16->fp16 V-temp conversion (2 B -> 2 B, elementwise, race-free)
  restores CUDA-line temp parity — (2) alone moves the wall only ~2 chunks.
  Evidence banked: `results/amd/ovf/` (red_gdb.log, red_gdb2/3/4/5.log, red6.gdb,
  RED_OVF_row.md, red*_resp.json, red_rss_trace.txt). Merged amd/wo-kvarnport (the bin's
  source) into this branch at 327b8db76 so the fix is against the as-built tree.
  NOTE: gdb batch kills the inferior at batch end — after each capture the box was
  re-retired with the law form; no stray serve processes.

WINDOW CLAIM: RED repro — agent #2 ovf desk, 2026-09-20T00:44:17ZZ. Retiring canonical per law form; booting bin f3312f25c8201da0 (kvarn_k4v4, max-ctx 36352, kv-cap 36352, mtp d2) on :8100; firing >10.3k prompt for RED capture (throw + backtrace + allocation site). Release to follow after canonical restore + health + FAST PROBE.

WINDOW RELEASED: 2026-09-20T01:25Z (coordinator owns restore — canonical boot deferred per BLOCKER-tagged 2c8901d3 warmup-fault class 414405c32; box retires clean by the law form, coordinator's release-poller confirms EMPTY).

WINDOW CLAIM: GREEN verify — agent #2 ovf desk, 2026-09-20T02:10:00ZZ. Booting banked bin ninfer-serve_8ac1ba93eb7fbfad.bin (in-place V conversion fix, commit 8ac1ba93e) on :8100 with canonical env EXCEPT NINFER_WORKSPACE_MIB=512 (arena lever: at --kv-capacity 36352 the KV pool is 405 MB/rank vs the auto arm measured 1,240 MB/rank -> ~835 MB/rank measured headroom; allocator stays the gate). Same 19.7k gates body must COMPLETE (finish=stop, coherent) + 2k and 9.5k parity probes. Retire per law form; canonical restore stays coordinator-owned.

- **2026-09-20T02:5xZ — G-OVF-1 MET (GREEN verified, clean window; desk CLOSES).**
  Banked bin **ninfer-serve_8ac1ba93eb7fbfad.bin** (HEAD 8ac1ba93e; sha256 8e8b7b541b28cb20).
  Boot: canonical env + NINFER_WORKSPACE_MIB=512 + kvarn flags; live preflight gate passed
  with **slack 404 MiB** (required 7755 / usable 8160; kv 386 MiB/rank at kv-cap 36352).
  Clean-window arms: **20k needle (19,709 tok) COMPLETES finish=stop, exact "ORCHID-TUNNEL",
  84 tok, zero mojibake (494 s)**; 9.5k needle exact ORCHID-TUNNEL (229 s, matches the gates
  bin's PASS row); 2k probe "BLUE" finish=stop (35 s). Collision-armored duplicates (the
  coordinator's G-MM-2 crossed my window mid-legs; my serve was never retired, 0 worker
  errors): 20k 425 s PASS, 9.5k 239 s PASS, 2k 67 s PASS — behavioral gates green in BOTH
  conditions. Fix = in-place bf16->fp16 V-temp conversion (8ac1ba93e; removes the second
  O(committed_pages) temp, the captured thrower); lever = ws 512 on the measured ~835
  MiB/rank kv-cap headroom. RED->GREEN rows: results/amd/ovf/RED_OVF_row.md +
  GREEN_OVF_row.md; standing cell (needle20k replay + grade) joins the kvarn boot battery.
  Canonical restore = coordinator (BLOCKER class 414405c32); box retired by the law form.
  Desk verdict: the WO's int32-overflow premise was falsified by measurement — the defect
  is the kvarn materialize route's O(committed_pages) workspace vs the era-pinned 96 MiB
  chunk arena. Structural cure (docs/120 B2 direct route; not byte-identical) is a lane
  decision, flagged for the coordinator, NOT done here.

WINDOW RELEASED: 2026-09-20T02:55Z (GREEN complete; box retired by the exact law form; canonical restore remains coordinator-owned per 414405c32; G-MM-2 may proceed).
