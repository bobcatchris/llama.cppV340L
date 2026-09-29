#!/usr/bin/env bash
# A/B helper: boot server with an env override, run guard battery twice
# (fresh-boot + no-boot rep), bank labeled rows, tear down.
# usage: ab_arm.sh <label> [ENV=VAL ...]
set -uo pipefail
LABEL=$1; shift
TESTS=/media/chris/ssd128/llamacpp/llama.cpp/docs/amd-port/tests
RES=/media/chris/ssd128/llamacpp/llama.cpp/docs/amd-port/results
STAMP=$(date +%H%M%S)
LOG=$RES/ab_${LABEL}_${STAMP}.log

env "$@" /home/chris/launch_tp3_200k.sh > "$LOG" 2>&1 &
SRV=$!
cleanup() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 72); do
  curl -s -m 2 "http://127.0.0.1:8080/health" 2>/dev/null | grep -q '"status":"ok"' && break
  kill -0 $SRV 2>/dev/null || { echo "BOOT FAILED - see $LOG"; exit 1; }
  sleep 3
done
curl -s -m 2 "http://127.0.0.1:8080/health" | grep -q '"status":"ok"' || { echo "BOOT TIMEOUT"; exit 1; }
echo "[$LABEL] boot ready, rep1 (fresh-boot battery)"
python3 "$TESTS/guard_battery.py" --port 8080 --server-log "$LOG" \
  --output-jsonl "$RES/ab_battery.jsonl" | grep -E "^(PASS|WARN|FAIL|OVERALL)" | sed "s/^/[$LABEL rep1] /"
echo "[$LABEL] rep2 (warm, no-boot battery)"
python3 "$TESTS/guard_battery.py" --port 8080 --server-log "$LOG" \
  --output-jsonl "$RES/ab_battery.jsonl" | grep -E "^(PASS|WARN|FAIL|OVERALL)" | sed "s/^/[$LABEL rep2] /"
echo "[$LABEL] done"
