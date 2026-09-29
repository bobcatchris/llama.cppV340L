#!/usr/bin/env bash
# desk3_diag_q81.sh - diagnostic (NOT a measured arm): engagement evidence for
# GGML_CUDA_Q81_ACT_CACHE on the canonical TP3 200k config. Boots with -lv 4 so
# ggml INFO lines pass the verbosity filter, serves one 400-token greedy request
# (>= 256 graph computes -> the periodic hits/misses stats line prints), greps
# the log, shuts down cleanly, releases the campaign lock.
set -uo pipefail
REPO=/media/chris/ssd128/llamacpp/llama.cpp
RES=$REPO/docs/amd-port/results
LOCK=/tmp/campaign_gpu_boot.lock
STAMP=$(date +%Y%m%d_%H%M%S)
DIAGLOG=$RES/desk3_diag_q81_$STAMP.log

cleanup() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; wait "${SRV:-}" 2>/dev/null; rm -f "$LOCK" 2>/dev/null; }
trap cleanup EXIT INT TERM

while [ -f "$LOCK" ]; do
  echo "[diag] campaign GPU lock held, waiting: $(cat "$LOCK" 2>/dev/null)"
  sleep 15
done
echo "holder=served-validation desk3 label=diag_q81_lv4 start=$(date +%s) eta_min=10" > "$LOCK"

# canonical config of record + -lv 4 (verbosity only; documented diagnostic)
HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_Q81_ACT_CACHE=1 "$REPO"/build-hip/bin/llama-server \
  -m /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf \
  -ngl 999 -sm tensor -c 200000 --batch-size 512 --ubatch-size 512 \
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
  --port 8080 -t 8 -lv 4 > "$DIAGLOG" 2>&1 &
SRV=$!

for i in $(seq 1 72); do
  curl -s -m 5 "http://127.0.0.1:8080/health" 2>/dev/null | grep -q '"status":"ok"' && break
  kill -0 $SRV 2>/dev/null || { echo "[diag] BOOT FAILED - see $DIAGLOG"; exit 1; }
  sleep 5
done
curl -s -m 5 "http://127.0.0.1:8080/health" | grep -q '"status":"ok"' || { echo "[diag] BOOT TIMEOUT"; exit 1; }
echo "[diag] boot ready"

curl -s -m 300 "http://127.0.0.1:8080/completion" \
  -d '{"prompt":"The quick brown fox jumps over the lazy dog. Count: one two three four five.","n_predict":400,"temperature":0,"cache_prompt":false}' \
  -o /tmp/desk3_diag_q81_resp.json
echo "[diag] request done: $(python3 -c "import json;r=json.load(open('/tmp/desk3_diag_q81_resp.json'))['timings'];print('predict_n',r.get('predicted_n'),'tps',round(r.get('predicted_per_second',0),2))")"

# clean shutdown (SIGTERM), then scan
kill "$SRV"; wait "$SRV" 2>/dev/null; SRV=""
echo "[diag] === engagement evidence ==="
grep -E "q8_1 activation cache enabled|hits / .* misses" "$DIAGLOG" | head -8
echo "[diag] pool0 asserts: $(grep -c 'GGML_ASSERT(pool_size == 0)' "$DIAGLOG")"
echo "[diag] full log: $DIAGLOG"
