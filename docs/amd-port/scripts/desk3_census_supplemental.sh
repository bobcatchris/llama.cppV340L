#!/usr/bin/env bash
# desk3_census_supplemental.sh - C3 gate reachability + C2 tile-vs-SGEMM
# supplemental traces (-p 512 reaches the dense cublas branch; -p 8 cannot).
set -u
cd /media/chris/ssd128/llamacpp/llama.cpp
LOCK=/tmp/campaign_gpu_boot.lock
while [ -f "$LOCK" ]; do echo "lock held: $(cat "$LOCK")"; sleep 15; done
echo "holder=served-validation desk3 label=census_supplemental start=$(date +%s) eta_min=15" > "$LOCK"
cleanup() { rm -f "$LOCK"; }
trap cleanup EXIT INT TERM
export HIP_VISIBLE_DEVICES=0,1,2
BENCH=build-hip/bin/llama-bench
GGUF=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
ROCP=/opt/rocm-6.2.0/bin/rocprofv3
ARGS="-p 512 -n 32 -r 1 -t 8 -ctk q4_0 -ctv q4_0 -fa 1 -ngl 99 -sm tensor"

echo "== C3 gate reachability (direct bench):"
GGML_CUDA_MMVQ_GROUP=1 "$BENCH" -m "$GGUF" -p 8 -n 16 -r 1 -t 8 -ctk q4_0 -ctv q4_0 -fa 1 -ngl 99 -sm tensor 2>&1 | grep -iE "grouped|group " | head -3
echo "-- end gate check (blank above = enable line filtered at bench verbosity)"

echo "== C2 -p 512 OFF pass:"
"$ROCP" --kernel-trace --output-format csv -o /tmp/c2_p512_off -- "$BENCH" -m "$GGUF" $ARGS > /tmp/c2_p512_off.log 2>&1
echo "off rc=$?"
echo "== C2 -p 512 ON pass:"
GGML_CUDA_TILE_FP16=1 "$ROCP" --kernel-trace --output-format csv -o /tmp/c2_p512_on -- "$BENCH" -m "$GGUF" $ARGS > /tmp/c2_p512_on.log 2>&1
echo "on rc=$?"
grep -iE "whitelist|WARN" /tmp/c2_p512_on.log | head -3
ls -la /tmp/c2_p512_off*.csv /tmp/c2_p512_on*.csv 2>/dev/null
