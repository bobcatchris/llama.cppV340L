# TP3 V340L served-validation desk - 3-candidate verdicts (2026-09-21)

Desk: served A/B on dies 0,1,2, canonical boot (/home/chris/launch_tp3_200k.sh,
ub512, TP3/200k, draft-mtp). One rebuild of build-hip at 19:14-19:15 from the
merged tree (HEAD b4ad90c5a lineage); ALL arms below ran that single binary
(libggml-hip.so mtime 19:14, llama-server 19:15, sha c1319b01615fd7c5 /
044d214dfda75d51 / 4f3407f9a98cbc34). Per-boot receipts: tp3_guards_*.jsonl +
manifest docs/amd-port/results/desk3_ab_manifest.jsonl (pre/post edge temps per
boot beside every row). Battery fingerprint aa336055d4d73b00 throughout.
Greedy reference sha 4beb1ba25219ee9b (E-027 gate of record).

## VERDICT TABLE

| cand | env | arm | decode t/s (reps)          | mean decode | prefill mean | accept  | verdict        |
|------|-----|-----|----------------------------|-------------|--------------|---------|----------------|
| C1 T4 q8_1 cache | GGML_CUDA_Q81_ACT_CACHE=1 | OFF x3 | 14.95 / 14.98 / 14.92 | 14.950 | 115.36 | 0.66667 | PASS          |
| C1  |     | ON x3 | 14.97 / 14.97 / 15.07      | 15.003      | 115.39       | 0.66667 | PASS          |
| C1  |     | delta |                            | +0.36%      | +0.03%       | =       | NOT PROMOTED (gate >= +2% decode) |
| C2 a8 tile | GGML_CUDA_TILE_FP16=1 | OFF x1 | 15.10 | 15.100 | 115.17 | 0.66667 | PASS (also C2 env-unset GREEN cell) |
| C2  |     | ON    | CRASHED on request 1       | n/a         | n/a          | n/a     | FAIL - STOPPED |
| C3 T3 grouped mmvq | GGML_CUDA_MMVQ_GROUP=1 | OFF x3 | 15.05 / 15.10 / 15.02 | 15.057 | 115.46 | 0.66667 | PASS          |
| C3  |     | ON x3 | 14.89 / 15.11 / 15.06      | 15.020      | 115.41       | 0.66667 | PASS          |
| C3  |     | delta |                            | -0.24%      | -0.04%       | =       | NOT PROMOTED (gate >= +2% decode) |

No candidate promotes; combined-arm check not applicable.
All 13 completed batteries: 5/5 guards PASS, needle 3/3.

## C1 (T4 q8_1 activation cache)

- 3-rep alternating OFF,ON,OFF,ON,OFF,ON fresh boots, 180s cold-stamp idle per
  arm (settle window: edge <= 35C, min 120s rest; per-boot temps in manifest).
- Engagement PROVEN (diagnostic boot at -lv 4; ggml INFO is verbosity-filtered
  at the default thold, which is why server logs show no gate lines):
  "q8_1 activation cache enabled" + begin_compute stats
  "401 hits / 427 misses over 1024 computes" (~48% hit rate). The cache
  populates and hits; the flat delta is a real negative at ub512/TP3 scale.
- Determinism sha byte-identical to reference in every arm (contract holds).
- Clean-shutdown smoke (b) GREEN: 0 GGML_ASSERT lines of any kind in all 6
  server logs; the E-018 pool_size teardown fix is confirmed served.
- NOT PROMOTED on the +2% decode gate (+0.36% observed).

## C2 (a8 tile)

- env-unset GREEN arm PASS (decode 15.10, prefill 115.17).
- ON arm: env delivery verified via /proc/<pid>/environ; boot healthy; SIGABRT
  on the FIRST prefill request - "ROCm error: out of memory" at
  ggml-cuda.cu:452 ggml_cuda_pool_leg::alloc inside ggml_cuda_tile_fp16_mul_mat
  <- ggml_cuda_op_mul_mat_cublas (core dump). One off-whitelist WARN
  ("shape off the census whitelist") fired 4s earlier: draft-model dense shapes
  are off-list at the canonical config. E-031 puts boot-ready free VRAM at
  ~200-400 MiB/die; the tile's per-call f16 pool allocs abort die 0 at 200k.
  Candidate STOPPED per protocol; no re-run (not a transient - the alloc path
  is tile-specific and deterministic).
- Census cell: census_decode.sh (-p 8) reaches NO dense GEMM under either arm
  (mmvq covers M <= MMVQ_MAX_BATCH_SIZE 8; zero GEMM-class kernels both arms).
  Supplementary -p 512 -sm tensor rocprofv3 pair DOES show the replacement:
  OFF = 2976 rocblas Cijk launches / 6729 ms; ON = 1632
  tile_fp16_gemm<128,64,4,8,32> (4325 ms) + 1632 tile_fp16_reduce (331 ms),
  the dominant MT128x128x16 rocblas class fully displaced, zero off-whitelist
  WARN at -p 512. Traces: desk3_c2_p512_{off,on}_kernel_trace.csv(.gz).
- FAIL - STOPPED. Re-try direction: load-time f16 residency (E-036 desk)
  to delete the per-call dequant alloc before the next served attempt.

## C3 (T3 grouped mmvq)

- 3-rep A/B: decode -0.24%, prefill -0.04% - FAILS +2%.
- Determinism byte-identical in every arm (contract holds); env-unset GREEN.
- Census launch-count cell NOT MET: quantize_q8_1 358413 and mul_mat_vec_q
  358413 launches IDENTICAL OFF vs ON (census_decode_20260921_224002 vs
  _224043 traces), zero mul_mat_vec_q_grouped launches. The grouping never
  forms in this topology: ggml_cuda_try_group_mmvq declines when
  stream_context().concurrent_events is non-empty ("never reorder around the
  multi-stream machinery" - active under the TP3 meta-backend butterfly) and/
  or no eligible same-src1 window survives the eligibility filters under split.
- NOT PROMOTED: the mechanism is inert on the canonical config.

## Contamination audit (post-alert, mandate 2)

Two mid-battery SIGKILLs (21:11:15, 21:27:16) and one serving kill (21:34:14)
were collisions with the tp2-feasibility lane's boots on shared port 8080
(journal: Gemini WINDOW START 21:10:54, GAP CLAIM 21:25:09, GO boot 21:34:16).
Affected rows are marked VOID-RETRIED in the manifest; retries clean.
Audit of all 14 completed arm logs (tests/desk3_contamination_audit.py):
exactly 7 tasks per battery, prompt-token signature
[3122, 7857, 32, 32, 6043, 6043, 6043] identical in every log, zero serial
task overlaps, zero KV-retry lines -> NO foreign requests landed on any
completed arm; the C1/C3 deltas and the C2 crash stand as measured.
The lane adopted the campaign GPU lock convention (/tmp/campaign_gpu_boot.lock)
mid-run: wrapper check-and-waits, holds through teardown.

## Harness changes banked by this desk

- scripts/desk3_ab_boot.sh: one served A/B boot (settle window, env-delivery
  proof via /proc, assert scan, campaign lock, manifest row with temps).
- scripts/desk3_diag_q81.sh: engagement diagnostic (-lv 4).
- scripts/desk3_contamination_audit.py: foreign-request audit.
- scripts/desk3_census_supplemental.sh: -p 512 tile census pair.
