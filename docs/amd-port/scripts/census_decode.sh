#!/usr/bin/env bash
# W0 census: kernel time ranking of no-spec decode (the W2/MMVQ target).
# rocprof v1 around llama-bench -p 8 -n 256 -> decode-dominated pass.
set -eu
cd /media/chris/ssd128/llamacpp/llama.cpp
BIN=build-hip/bin/llama-bench
GGUF=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
OUT=docs/amd-port/results/census_decode_$(date +%Y%m%d_%H%M%S)
CSV=${OUT}.csv
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2}
echo "census start $(date) cold-stamp:" | tee "${OUT}.stamp"
rocm-smi --showtemp --showclock 2>/dev/null | grep -E 'GPU\[' | head -12 | tee -a "${OUT}.stamp"
/opt/rocm-6.2.0/bin/rocprofv3 --kernel-trace --output-format csv -o "${CSV%.csv}" -- "$BIN" -m "$GGUF" -p 8 -n 256 -ngl 99 -sm tensor \
  -ctk q4_0 -ctv q4_0 -fa 1 -r 1 -t 8 2>&1 | tail -20 | tee "${OUT}.bench.log"
echo "census end $(date)" | tee -a "${OUT}.stamp"
ls -la "${OUT}"* | tee -a "${OUT}.stamp"
