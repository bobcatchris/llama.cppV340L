# NEW MODEL LANDING CHECKLIST — what happens the day the principal names the smaller model

Written 2026-09-20 at the pivot (PLOG-090: smaller model, NVFP4 dropped). This is the
dispatch-ready recipe so "here's the model" converts to a serving boot the same session.
Nothing here needs the model identity; fill the three blanks at the bottom and execute
top to bottom.

## 0. THE THREE BLANKS (needed from the principal)
- MODEL: name / weights location / HF repo.
- TARGET: context? latency? throughput? (decides every posture below).
- STACK: ninfer or external (llama.cpp)? If external — the parked probe
  (WO parked 2026-09-20, llamacpp) becomes the first desk; its entry blockers
  (docker, source weights, disk) were pre-solved on paper; storage/cooling
  arrivals make it trivial.

## 1. ACQUIRE + BAKE (if ninfer)
1. Weights to /home/chris/models/<new_model>/ (disk: 8.6 GB free on / at writing;
   use /media/chris/EMTEC256 (157 GB) for anything large; df -h / first).
2. Bake path: the NVFP4 bake is 27B-specific tooling — for a non-NVFP4 model the
   serving line is bf16/fp16 weights (serve_options default) — the stack boots
   unquantized models natively (the bf16 route is the oldest, most-tested path).
3. Draft assets (if spec mtp is wanted): the 27B draft vocab/heads are model-specific;
   WITHOUT them boot with --spec omitted (no MTP) — decode is 1 token/step but
   everything else works. MTP-for-new-model is a separate desk.

## 2. FIRST BOOT (allocator is the gate; no estimated refusals)
1. Boot shape: `serve_fast.sh` pattern with the new model path, NO --kv-dtype
   (bf16 KV default), canonical env otherwise.
2. Read the preflight lines: fixed side (the new placement), auto-KV capacity, slack.
   THAT is the new context ceiling, measured — the whole PLOG-089 ladder question
   answers itself here.
3. TP world choice: TP4 (all 4 dies) if the model needs it; TP2 opens per-die budget
   2x if weights/rank <= ~6 GB (TP2 receipt: the 27B overflowed at 10,262 MiB/rank —
   results/amd/k4v4fin/boot_refused_k4v4_300k_receipt.log addendum).

## 3. BASELINE BATTERY (the new model's first row)
1. BLUE sanity + 5-probe behavioral battery (tools exist: w7 probes; bodies need the
   new model name in the "model" field).
2. Prefill/decode census: tools/v340l/wo_census_leg.sh (EXTRACT-BEFORE-RESTORE built in;
   NINFER_PREFILL_OPTRACE=3). NOTE: the OPTRACE column partition is bf16-route-shaped —
   for a bf16-model boot the columns are valid as-is (PLOG-087's gap was kvarn-route-only).
3. Thermal sideband: tools/v340l/w7_gap_clock_sampler.sh (RED->GREEN fixed 2026-09-20,
   live-validated — usable for the state-function check on the new model).
4. Bank: results/amd/<new_model>/ + PLOG row. The pre-registered bars start at the
   model's OWN baseline — no inherited 27B bars (the pk16 lesson: bars belong to goals).

## 4. ONLY IF WANTED LATER (all parked, all keyed)
- kvarn KV quantization (context x3.1) — the family is model-agnostic; the HIP port
  needs the new model's head geometry in the launchers (small port, same pattern as
  the 27B port; WO_Q4KV_K4V4.md documents every step of the last one).
- MTP drafting for the new model.
- llamacpp comparison leg.

## LAWS THAT DO NOT CHANGE
Retire law EXACT form; restore + health + +3-min liveness; no cmake in the shared
checkout; df before >1G; <=2 agents; PLOG chain via tools/guards/plog_append.py.

## 5. THE 27B RETIREMENT PLAN (execute when the new model's weights need the disk)
- ARCHIVE to /media/chris/EMTEC256 (157 GB, 30 % used): the NVFP4 artifact
  /home/chris/models/qwen3_8_27b_nvfp4.ninfer (18 GB — the single biggest reclaim;
  frees / for the new weights). Verify sha after move before deleting the original.
- KEEP FOREVER (provenance, KBs-MBs): results/ (all banked receipts), docs/amd/
  (PLOG chain, WOs, the explainer), tools/v340l/ (runners), the banked BINS in
  /home/chris/artifacts_bin/ (small; they are the boot stamps the ledger cites —
  cc122cc0, 8ac1ba93, 058a7b85, 37bc9b25, 05628598...).
- RELEASE ON EXIT CHECK: stale worktrees (git worktree list — all merged as of
  7e0e884c0), /tmp scratch (census/request files), pip caches if pip was used.
- The serving line reverts to whatever the new model's canonical is — the 27B
  serve_10k/serve_fast lines get commented (not deleted) as the rollback record.
