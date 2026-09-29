#!/usr/bin/env bash
# DP-3 paired window v2: one boot, direct guard_battery.py cells (no runner
# teardown between cells), SIGUSR1 toggles the tree between adjacent cells.
set -u
export HOME=/home/chris
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=*
export GGML_CUDA_AR_TREE=1
export BOOT_SCRIPT=/home/chris/launch_tp3_200k_therock.sh
cd /media/chris/ssd128/llamacpp/llama.cpp
T=docs/amd-port/tests/guard_battery.py
BIN=build-hip-therock/bin/llama-server
LCFG="BOOT_SCRIPT=/home/chris/launch_tp3_200k_therock.sh port=8080"

echo "=== boot (tree ON)"
/home/chris/launch_tp3_200k_therock.sh > /tmp/dp4_pw2_server.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8080/health | grep -q ok && break; sleep 3; done
SRV=$(pgrep -f "bin/llama-server" | head -1)
echo "server pid $SRV"

cell() {
  python3 $T --port 8080 --server-binary $BIN --launch-config "$LCFG toggle=$1" \
    --server-log /tmp/dp4_pw2_server.log \
    --output-jsonl docs/amd-port/results/dp4_paired.jsonl 2>&1 | grep -E "OVERALL|decode_tps|prefill_tps" | head -3
  sleep 20
}

echo "=== A1 (ON)"; cell ON
kill -USR1 $SRV; sleep 3
echo "=== R1 (OFF)"; cell OFF
kill -USR1 $SRV; sleep 3
echo "=== A2 (ON)"; cell ON
kill -USR1 $SRV; sleep 3
echo "=== R2 (OFF)"; cell OFF
kill -USR1 $SRV; sleep 3
echo "=== WINDOW COMPLETE"
kill $SRV 2>/dev/null
