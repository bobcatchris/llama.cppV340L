#!/usr/bin/env bash
# DP-3 thermally-matched paired window: ONE boot, adjacent R/A cells toggled
# via SIGUSR1 - thermal confound collapsed (cells are minutes apart).
# Order: A1 ON -> R1 OFF -> A2 ON -> R2 OFF (battery = full guard set each).
set -u
export HOME=/home/chris
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=*
export GGML_CUDA_AR_TREE=1
export BOOT_SCRIPT=/home/chris/launch_tp3_200k_therock.sh
cd /media/chris/ssd128/llamacpp/llama.cpp
R=docs/amd-port/tests/run_tp3_guards.sh

echo "=== boot (tree ON by default)"
$R > /tmp/dp4_pw_a1.log 2>&1
echo "=== A1 done: $(grep OVERALL /tmp/dp4_pw_a1.log)"

SRV=$(pgrep -f "build-hip-therock/bin/llama-server" | head -1)
echo "=== toggle OFF (USR1 -> $SRV)"
kill -USR1 $SRV; sleep 5
$R --no-boot > /tmp/dp4_pw_r1.log 2>&1
echo "=== R1 done: $(grep OVERALL /tmp/dp4_pw_r1.log)"

echo "=== toggle ON"
kill -USR1 $SRV; sleep 5
$R --no-boot > /tmp/dp4_pw_a2.log 2>&1
echo "=== A2 done: $(grep OVERALL /tmp/dp4_pw_a2.log)"

echo "=== toggle OFF"
kill -USR1 $SRV; sleep 5
$R --no-boot > /tmp/dp4_pw_r2.log 2>&1
echo "=== R2 done: $(grep OVERALL /tmp/dp4_pw_r2.log)"
echo "=== WINDOW COMPLETE"
