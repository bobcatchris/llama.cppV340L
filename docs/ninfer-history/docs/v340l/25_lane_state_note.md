# 25 — Lane state note (successor entry point), 2026-09-13 ~15:4xZ

Author: agent2 (pi 01a09af0), hip_shim lane. Supplements, does not replace, `23_lane_handoff_at_session_end.md`
(on this branch @ ea3e1e7b) — everything in 23's "do these first" list was executed and its open items closed:

- **v340l/23 open item: tps_probe per-token curve** → DONE `3e818d23` (+ `--arrival-jsonl`), fake-SSE both-ways,
  rule is max/MEAN (median self-masked the planted spike — lesson in the file's non-vacuity block).
- **v340l/23 open item: README stale shfl citations ("never landed" hunk)** → DONE `c2e6106b`, symbol-grep form,
  `tools/ops/verify_registered_exception.py` rc=0 (dir-bearing path — file lives in tools/ops/, cite with dir per
  agent5's decay-law point). If the chair never merged the lane, re-verify:
  `git show origin/amd/main:src/common/hip_shim/README.md | grep -c '441/:451/:466'` (1 = still stale, mine pending).
- **Chair queue (1) host suite** → GREEN three times: 9/0/0/0 at 88af9c78 (main 174b1d84), at 9cbcbadf (main f18ba195),
  and at 83fba46e (main bfa04b1f; receipt addendum banked same file `results/amd/host_suite/2026-09-13_merged_tip_88af9c78.txt`).
  Shas reported to chair.
- **Chair queue (2) tp1-control = G-AMD-28** → COMPLETE as a DEATH ROW, refuted-by-bytes: the ref artifact is
  **binder-dead on this tree** (66 unconsumable dflash2/* container objects; 27B consumption set has zero dflash2
  bind sites; both single AND tp2 routes call the same `bind_artifact`). Everything measured + repro commands:
  `results/amd/p3/G28_death_row.md` @ e67483b2. DO NOT re-fire the same argv — deterministic 0.2 s repro, it burns
  the window for a line you can read in the row. The cell re-opens only at a chair product ruling (strip-export
  ref / sidecar-shape boot / fixture-class control). Binary banked: `/home/chris/artifacts_bin/tp1control_a28_9dc56c390815bda0.bin`.
- **Chair queue (3) USER-TASK real-context TPS cell on dev2,3** → **SUPERSEDED — it FIRED and CLOSED.
  Chair cross-map #978, all four shas byte-verified at this seat 03:5xZ:** fired as **A2Q3-1 / G-AMD-32**
  (granted 18:44Z per `results/amd/p3/A2Q3_grant_request.txt`; four-addendum close at **ce5c52b4**, in
  amd/main; AUTO capacity printouts + k5v4 stub-death banked; residence/harness instruments promoted to
  standing law). `:132 '15 vs 0'` self-closed by agent4 at **d701fe8f** (in amd/main). The lane-merge
  decision below was **exercised — merged**; host-suite four-count restated 9/0/0/0 at ff8ff70e-era and
  again at **c1fa658c** (in amd/main). G-AMD-28 product ruling: chair's, still parked — the binder-dead
  `G28_death_row.md` IS the closure record. RESTART-kit gemini-line: superseded by
  `docs/amd/RESTART_2026-09-14/` front door. **DO NOT re-fire or re-request any of the above from this
  note's wording — this bullet's "STILL UNFIRED" state is the /tmp replay band the chair just classified
  as stale relay.** Live carry from the relay: HAZARD ONLY — disk measured **8.8G free (93%)** at
  03:5xZ this session; the df<10G=STOP clause of G-AMD-32's grant generalizes: any build, anywhere, on
  this host, audit df FIRST. *(annotation per chair #978; original text kept verbatim below the fold of
  this bullet as the provenance of the error.)*
  ORIGINAL (pre-supersession): STILL MINE, STILL UNFIRED (pair held by agent4's
  item-7/carry-in block per chair STATE 13:5xZ). When it frees: written request with host:port+window, chair ack
  in row text, manifest BEFORE spawn, kill recorded pgid only (contract clause), artifact EMTEC256 q3 (Desktop copy
  TRUNCATED — never touch), `HIP_VISIBLE_DEVICES=2,3` + `--devices 0,1` (mask/ordinal law), agent5's gen-64 +
  second-request cells attach zero-boot, cycle class + `--request-log-jsonl` on every row, tps_probe now carries
  the arrival curve — report the CURVE, decode_tps is a summary of a shape.
  - **GEN-64 reader cell, chair-APPROVED (ITEM6 §3 @ 142e9714, status #797): fires INSIDE this boot, reader-only,
    zero reservation, do-not-fire-before-my-window-announcement.** Arm A: `tps_probe.py --prompt-tokens 54
    --max-tokens {4,32,64,128} --repeats 2` — 32→64/64→128 are the first beyond-all-observed points (kv cap 260,
    5 pages: engage-after-N ∈ (32,128] catchable ONLY here). Arm B: `--prompt-tokens 1800 --max-tokens {32,128}
    --repeats 2` in stable ctx-2048+k5v4 (1800+128=1928≤2048, re-checked). Flatness rule PRE-DECLARED:
    marginal(32→128) within ±15% of marginal(4→32) = flat, else cliff relocated WITH its N; marginals by
    consecutive pairs, division chains printed. Plus the second-request determinism reader (chair #790 slot):
    same prompt ×2-3 identical-length, every generation incl warmups captured + request-log-jsonl — data source
    for agent4's carry-in hunt, nothing more from this seat.
  - Provenance settled at my seat (#764 item 2): banked `bc122675:results/amd/p3/G17g3_serve.log` reads
    decode=0.62 s (gen=4, line 710) / 6.42 s (gen=32, line 3885) — my measured grep this session; the 0.63/6.48
    pair came from the since-CLOBBERED `/tmp/g3_serve.log`@bb984de4 (kit single-path defect, ruling #797, fixed
    by kit v6 per-LABEL banks). The dead-sha was a one-digit slip: verdict row is `8d090f6b`, cat-file says
    commit, landed via bc122675. Split retired exactly as agent5 predicted — "landing at main retires the split
    in one fetch."
- **v340l/24** (mine, 04c483da): item-7 decisive-cell review — A1TRACE-K sees flags not payload, so agent3's
  outcome (iii) could not falsify their own tear theory. Status: superseded-in-substance by events (chair
  adjudicated per-request nondeterminism live, 1d0ff3c6; sampler window exonerated 4/5 by agent3's own rule).
  The instrument-law content stands for the record: an observation channel must be able to see the mechanism it
  is declared decisive for.

## Verified facts a successor should not re-derive unless deciding from them

- `Binder::finish()` (src/artifact/binder.cpp:149-156) throws on first unconsumed manifest object; ValidateOnly
  counts as consumed; ref's dflash2 rows have NO consumer in `src/targets/qwen3_6_27b/` (bindings.cpp, tp_load.cpp
  — grep -c both = 0); sidecar path is `--dflash2-sidecar <manifest-file>` (tp2_backend.cpp:1306 requires it under
  spec DFlash2). q3 artifact: dflash2 count 0 (mmap+JSON parse of manifest, 1124 objs). ref: 1190 objs.
- make_engine: `devices.size()==2` → TPEngine, else single Engine (engine.cpp:308-313) — a "tp1" boot with
  `--device N` never reaches tp2 code at all.
- Watcher (`results/amd/watch_events/`, 30-min tsv) still runs from the predecessor session; bank-commit its rows
  periodically, it's my lane's file.

## Hygiene the chair has (do not re-litigate)

Shared-tree `G28_pid.txt` = my drifted-cwd tee, disclosed, chair told to keep-or-sweep. Shared scratch `/tmp/ht`
detached at my merged tip (the suite's TREE design). `/tmp/sv_G17*.log` = my corpus copies, transient, fine to die.
Board law reminder for myself as much as anyone: cite banked path+sha, never a /tmp name (#797).
