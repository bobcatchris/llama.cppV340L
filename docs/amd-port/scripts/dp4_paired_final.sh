#!/usr/bin/env bash
# DP-3 DEFINITIVE paired window (supervised). One boot, 4 cells, USR1 toggle
# between cells, receipt per cell.
set -u
export HOME=/home/chris
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=*
export GGML_CUDA_AR_TREE=1
cd /media/chris/ssd128/llamacpp/llama.cpp
T=docs/amd-port/tests/guard_battery.py
BIN=build-hip-therock/bin/llama-server
LCFG="BOOT_SCRIPT=/home/chris/launch_tp3_200k_therock.sh port=8080 stack=therock7.14"
W=docs/amd-port/results/dp4_paired_final

/home/chris/launch_tp3_200k_therock.sh > ${W}_server.log 2>&1 &
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && break; sleep 3; done
SRV=$(pgrep -f "llama-serve[r]" | head -1)
echo "$(date -Is) server pid $SRV"

cell() {
  runuser -u chris -- env HOME=/home/chris GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*' \
    python3 $T --port 8080 --server-binary $BIN --launch-config "$LCFG tree=$1" \
    --server-log ${W}_server.log \
    --output-jsonl ${W}.jsonl > ${W}_cell_$1.log 2>&1
  chown chris:chris ${W}_cell_$1.log 2>/dev/null
  grep -E "OVERALL|decode_tps" ${W}_cell_$1.log | head -2
}

cell ON;  kill -USR1 $SRV; sleep 10
cell OFF; kill -USR1 $SRV; sleep 10
cell ON;  kill -USR1 $SRV; sleep 10
cell OFF; kill -USR1 $SRV; sleep 10
echo "$(date -Is) WINDOW COMPLETE"
