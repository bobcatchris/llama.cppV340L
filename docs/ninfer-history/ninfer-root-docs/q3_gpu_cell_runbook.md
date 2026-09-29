# q3 v1a GPU cell — queued runbook (execute ONLY on coordinator RELEASE + re-grant)

## ARTIFACT LOCATION (2026-09-11 late): /media/intel/models/qwen3_8_27b_q3.ninfer
md5 e57258df899e5c4590edeae353564775 (matched pre-move local copy); local deleted to free space.
NTFS read of 14.4G during load adds ~1-3 min; all serve commands below use this path. The
reference /home/intel/models/qwen3_8_27b.ninfer CANNOT load on this machine (needs 17.09G >
16.5G free) — 27B single-card was FIRST enabled by this Q3 artifact.

## GATING BUG PROVENANCE (aa576027, 'TP2 Agent', Aug 29): the exact 'setattribute returns
success but does not apply' failure class was ALREADY hit+fixed once (cached -> re-issue). Our
:372 invalidValue is the same symptom family on this 36-SM/5060Ti+13.1 combo, for the MmaUnsplit
(Stages=4, 80KiB) launch that TP2-era tests never reached on this box. Candidate fix if the
next-window LAUNCH_BLOCKING test indicts gating (not my GEMM): instantiate unsplit at Stages=2
(40KiB < 48KiB default; attr-free) — perf irrelevant until serving, correctness first. My W2
test now includes T=9/17 (this exact boundary) so one op-test run discriminates in ~60s.

## STEP 0 (FIRST, replaces everything below on failure): OWNERSHIP PROBE
**UPDATED 23:5x — superseded by the isolation repro; run THIS first:**
`CUDA_VISIBLE_DEVICES=<granted> /tmp/gqa_repro` (source /tmp/gqa_repro.cu, rebuild if lost:
nvcc -rdc=true + cudadevrt, include src/include/third_party, sm_120a). Prints the REAL
Gqa27Geometry bf16-prefill kernel's attrs, setattr result, launch+sync errors. My probe5 launched
96 KiB/92.5 KiB configs CLEAN on this box — if the real kernel throws invalidValue here, the
attrs line will show what differs (regs/staticSmem/maxDyn) and the fix follows directly. If it
LAUNCHES clean in isolation, the difference is state the serve builds before this point (capture
with --no-prefix-reuse and a post-load immediate-sync to bisect).

**Family-wide fact (LAUNCH_BLOCKING-proven, both KV dtypes):** :73 is the shared trailing check
of gqa_attention_prompt_attention_launch_for; i8 branch (92,672 B static_assert) AND bf16 branch
(98,304 B) both die there with warmup prompt (~9-17 template tokens, > small-t threshold).
Warmup with kvarn instead fails EARLIER in a different way (fused-append throw — that route is
kvarn-incompatible; production serves must use the append+cached pair — flag/route question,
see wrapper gqa_attention.cpp:592 + cached path ~916).

**Control that settles Q3-vs-environment:** serve the REFERENCE /home/intel/models/qwen3_8_27b.ninfer
with --devices 0,1 (TP2 — only fits across both cards). If reference-TP2 warmup ALSO hits :73 or
a kvarn-route throw, the prefill family is pre-existing-broken HERE and the q3 cell reduces to
re-verify after env fix; if it warms clean, Q3 artifacts are implicated and the repro MUST show
why.

### (old STEP-0 text, kept for history) OWNERSHIP PROBE
`./build/apps/ninfer-serve /media/intel/models/qwen3_8_27b_q3.ninfer --devices 1 --port 8125 --greedy
 --kv-dtype bf16 --no-cuda-graph --max-context 8 --default-max-tokens 4`
Rationale: warmup 'hi' prefill ≤8 tokens routes bf16 gdn_gating to SmallTGemv (NO dynamic smem).
The 80KiB dynamic-smem MmaUnsplit launch (T>=9) is the ONLY 27B route whose <<<>>> needs the
per-block smem opt-in, and the code's own comment (kernels.cu:316-319) says sm_120a cached-
setAttribute is unreliable — this box is a 36-SM RTX 5060 Ti and the REFERENCE ARTIFACT CANNOT
LOAD HERE AT ALL (preflight: needs 17.09 GiB > 16.5 free), so the 27B T>=9 gating route has
NEVER executed on this card. If this probe UPs + prompts: gating-smem is sole blocker (fix W-DIR
below). If it throws :372 again: my Q3 kernels implicated too — capture LAUNCH_BLOCKING trace.

## ROOT-CAUSE FILE (pre-existing, NOT Q3): bf16_gdn_gating_proj_kernels.cu:480 instantiates
launch_bf16_prefill_mma<...,Stages=4> = 80 KiB dyn smem via plain <<<>>>; device opt-in
apparently insufficient on this SKU/CUDA-13.1 combo. Fix direction (awaiting stamp, one-shot
rule): drop MmaUnsplit to Stages<=2 (kSmem 40 KiB <= 48 KiB default) OR swap the plain launch
for cudaLaunchKernelEx path used by SplitK>1. A CPU-side compile + residency re-audit both.

Preconditions each step: `touch /home/intel/ninfer/worktrees/wo-q3-gemv/.a3_gpu_lease`
at claim; heartbeat into results/q3_gpu_window_2026-09-11.md; NO foreign-context
guard bypass (gpu_guard.sh repo copy); kill only own PIDs.

1. **Artifact verify** (10 s): if local artifacts/qwen3_8_27b_q3.ninfer was
   removed post-drive-move, work directly from /media/intel/models copy
   (slow USB: mmap page-ins cost once; acceptable) OR copy back after md5 match.
   Expect: identity qwen3.8-27b/groupwise-q3, 129 Q3, 15,446,796,288 bytes.
2. **Op-test relink** (no card):
   `cmake --build build -j6 --target ninfer_linear_q3_a16_test ninfer_linear_q2_a16_test ninfer_artifact_reader_test`
3. **Numeric gate on card** (mine, single-device):
   - `CUDA_VISIBLE_DEVICES=<granted> ./build/tests/ninfer_linear_q3_a16_test` → expect OK Q3_A16 Linear
   - same q2 — unchanged kernels, regression guard.
4. **Artifact-load gate** (the real W4 proof):
   `CUDA_VISIBLE_DEVICES=<g> timeout 900 ./build/apps/ninfer-serve artifacts/... --devices <g> --kv-capacity 4096 --max-context 4096 --port 8125 --greedy --kv-dtype kvarn_k4v2`
   - PASS = "loading model..." then listen line, NO "no registered target"/binder throw.
   - cudaMalloc near-cap failure = per VRAM Law a clean measured refusal: record
     the reported GiB, drop --kv-capacity and relaunch once; second failure = stop+report.
5. **Prompt cell**:
   `curl -s localhost:8125/v1/chat/completions -d '{"model":"q","messages":[{"role":"user","content":"What is 27 times 43? Answer with the number only."}],"max_tokens":16,"temperature":0}'`
   capture: text, finish_reason, first-token latency + decode tok/s from server log.
6. **Embed-gather proof**: the prompt exercises it (prefill embeds tokens); pass =
   coherent output. If serve throws inside embedding/Q3G64 = W1 bug, verbatim report.
7. Bank rows into results/q3_gpu_window_2026-09-11.md; release lease + RELEASE row.

Known-bad fallbacks (do NOT improvise around silently; report):
- gdn layers: value_z bound Q5 ✓ (v1a deferral) — if binder complains shape/role, it
  means the artifact tier drifted; stop.
- attn_input_proj full-attn layers qk/gate_value Q4/Q5 ✓ untouched.
- draft/MTP/vision objects copied byte-exact from reference ✓ untouched.

## Step-0 probe FALSIFIED (2026-09-11, CPU-only, /tmp/smem_probe): 80KiB dyn-smem + fresh
setAttribute + plain <<<>>> launches clean on this 5060Ti/CUDA-13.1 (optin_max=99KiB). The
MmaUnsplit-smem theory is dead. Corrected shortlist for :372's cudaErrorInvalidValue:
 (a) STICKY-ERROR misattribution (no LAUNCH_BLOCKING in the repros) — culprit may be an EARLIER
     launch; prime suspect = my W2 SIMT GEMM (T=2 warmup prefill executes it for the first time
     EVER — never ran standalone: op-tests only cover T=1 GEMV; multi-token math tested only at
     CPU-reference level).
 (b) 27B prefill has NEVER run on this machine (reference CANNOT load: needs 17.09 GiB > 16.5)
     -> both prefill-phase throws (kvarn fused-append, gating) are first-here paths; if LAUNCH_
     BLOCKING exonerates my GEMM, the gating throw is a pre-existing card-environment bug to
     report, not lane-fix.
NEXT (needs card, ~5 min): CUDA_LAUNCH_BLOCKING=1 serve attempt, ctx 512; the trace names the
REAL kernel. If it's mine: fix SIMT launch config (1280x1 grid, 128 thr, 0 smem looks legal —
more likely an alignment/pointer arg; audit against require()s). If gating: file env-bug row.
