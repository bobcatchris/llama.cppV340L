#!/usr/bin/env bash
# HOST-SLICE P0 diagnostic: does the TP4 warmup graph run survive
# (a) canonical env only, (b) canonical + launch/decode/spec timeline envs?
# Each boot: lock-compliant, killed by PID at the verdict. Exit code captured.
set -u
BIN=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin/llama-server
MODEL=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
RES=/media/chris/ssd128/llamacpp/wt-host-slice/docs/amd-port/results
LOCK=/tmp/campaign_gpu_boot.lock
PORT=8083

BASEENV="GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_MMVQ_IQ3S_SHARE=1 GGML_CUDA_MMVQ_IQ3XXS_SHARE=1 GGML_CUDA_MMVQ_Q3K_SHARE=1 GGML_CUDA_MMVQ_Q4K_SHARE=1 GGML_CUDA_MMVQ_Q5K_SHARE=1 GGML_CUDA_MMVQ_Q6K_SHARE=1 LLAMA_DRAFT_FAST_TOPK=1 LLAMA_DRAFT_PACKED_GET=1 LLAMA_DRAFT_LIGHT_SYNC=1 LLAMA_VERIFY_ROW_SAMPLING=1 LLAMA_ASYNC_INPUT=1 GGML_PINNED_DEV_COPY=1"
TLENV="LLAMA_LAUNCH_TIMELINE=1 LLAMA_DECODE_TIMELINE=1 LLAMA_SPEC_TIMELINE=1"

try_boot() {  # try_boot <tag> <extra-env>
  local tag="$1" extra="$2"
  local log="$RES/W12_hostslice_diag_${tag}_$(date +%H%M%S).log"
  while [ -f "$LOCK" ]; do
    local HPID; HPID=$(sed -n 's/.*"pid":\([0-9]*\).*/\1/p' "$LOCK" 2>/dev/null)
    if [ -n "$HPID" ] && ! kill -0 "$HPID" 2>/dev/null; then
      echo "[$tag] stale lock (pid $HPID dead), removing: $(cat "$LOCK")"
      rm -f "$LOCK"; break
    fi
    echo "[$tag] lock held, waiting 150 s"; sleep 150
  done
  printf '{"desk":"host-slice","arm":"diag-%s","ts":"%s","pid":%d}\n' "$tag" "$(date +%s)" "$$" > "$LOCK"
  echo "[$tag] boot start $(date '+%T') extra=[$extra]"
  env HIP_VISIBLE_DEVICES=0,1,2,3 $BASEENV $extra "$BIN" -m "$MODEL" \
    -ngl 999 -sm tensor -c 32768 --batch-size 512 --ubatch-size 512 \
    -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
    --device ROCm0,ROCm1,ROCm2,ROCm3 --port $PORT -t 8 -lv 4 > "$log" 2>&1 &
  local SRV=$!
  local WARMED=0 ALIVE=1
  for i in $(seq 1 8); do
    sleep 15
    grep -q "warming up" "$log" && WARMED=1
    if ! kill -0 "$SRV" 2>/dev/null; then ALIVE=0; break; fi
    grep -q "listening" "$log" && break
  done
  wait "$SRV" 2>/dev/null; local RC=$?
  echo "[$tag] VERDICT: warmed=$WARMED survived_loop=$ALIVE wait_rc=$RC log=$log"
  echo "[$tag] last lines:"; tail -4 "$log" | sed 's/^/[$tag]   /'
  grep -q '"desk":"host-slice"' "$LOCK" 2>/dev/null && rm -f "$LOCK"
  echo "[$tag] lock released"
}

try_boot ctrl ""
try_boot tl "$TLENV"
echo "DIAG DONE"
