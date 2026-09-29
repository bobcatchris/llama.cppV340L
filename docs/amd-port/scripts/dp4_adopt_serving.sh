#!/usr/bin/env bash
# dp4_adopt_serving.sh - DP program adoption: switch serving to the winning
# configuration (dp2 kernel + THE ROCK 7.14 runtime + AR_TREE), then re-stamp
# the guard baseline.  Run as: sudo docs/amd-port/scripts/dp4_adopt_serving.sh
#
# What it does:
#   1. verifies the dp2 kernel is running (the P2P substrate prerequisite)
#   2. backs up /home/chris/launch_tp3_200k.sh -> launch_tp3_200k.pre_dp4.bak
#   3. installs the THE ROCK launcher as the serving launch script
#      (build-hip-therock binary + GGML_CUDA_AR_TREE=1)
#   4. runs the full guard battery; on OVERALL PASS it rattles the baseline
#      (adopts the new numbers as of-record)
#
# Revert: restore the .bak over launch_tp3_200k.sh; the dp2 kernel stays
# (it is strictly better: P2P substrate + whitelist built-in), or re-select
# the previous kernel in grub.
set -u
if [ "$(id -u)" -ne 0 ]; then echo "run as root (sudo)" >&2; exit 1; fi

KREL=$(uname -r)
echo "== kernel: $KREL"
case "$KREL" in
  *dp2*) echo "   dp2 kernel active - P2P substrate prerequisite OK";;
  *) echo "REFUSED: not on a dp2 kernel (the P2P whitelist is in that kernel). Boot it first."; exit 1;;
esac

LAUNCH=/home/chris/launch_tp3_200k.sh
THEROCK=/home/chris/launch_tp3_200k_therock.sh
REPO=/media/chris/ssd128/llamacpp/llama.cpp

[ -x "$THEROCK" ] || { echo "missing $THEROCK - create it first (build-hip-therock launcher)"; exit 1; }
[ -x "$REPO/build-hip-therock/bin/llama-server" ] || { echo "missing the therock llama-server build"; exit 1; }

if [ ! -f "$LAUNCH.pre_dp4.bak" ]; then
  cp "$LAUNCH" "$LAUNCH.pre_dp4.bak"
  echo "== backed up: $LAUNCH.pre_dp4.bak"
else
  echo "== backup already exists (keeping the original)"
fi

cp "$THEROCK" "$LAUNCH"
echo "== serving launcher switched to the THE ROCK binary + AR_TREE=1"

echo "== running the full guard battery (boot + ~8 min)..."
cd "$REPO"
export HOME=/home/chris
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0=*
docs/amd-port/tests/run_tp3_guards.sh --ratchet
RC=$?
echo "== battery exit: $RC"
echo
echo "On OVERALL PASS: the adoption is of-record (baseline rattled)."
echo "Revert any time: cp $LAUNCH.pre_dp4.bak $LAUNCH"
