#!/usr/bin/env bash
# w7_oneshot_repro.sh — WO_ONESHOT_ROOTLOOK repro boot (bounded, stack-capturing).
# Usage: bash w7_oneshot_repro.sh <bin_path> <label> [timeout_s=120]
# Boots <bin> with NINFER_TP_ONESHOT_AR=1 + canonical env, watches warmup:
#   CLEAN   -> "listening on http" within timeout: health-probe, retire, verdict CLEAN.
#   WEDGED  -> [AR-WEDGE-WATCHDOG] line OR timeout silence: gdb thread-applied bt,
#              then SIGTERM (law: wedged = SIGTERM; they exit clean).
# Log + stacks banked under results/amd/coherence/w7_oneshot_rootlook/.
set -u
BIN=${1:?bin path}
LABEL=${2:?label}
TMO=${3:-120}
OUT=/home/chris/worktrees/amd-wo-w7-body/results/amd/coherence/w7_oneshot_rootlook
LOG=$OUT/${LABEL}_serve.log
STK=$OUT/${LABEL}_stacks.txt
mkdir -p "$OUT"

export NINFER_ALLOW_NVFP4_TP2=1
export NINFER_WORKSPACE_MIB=96
export NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/multi_gpu/data/qwen38_draft_vocab_ids.json
export NINFER_MTP_TAIL_ASYNC=1
export NINFER_H2D_PINNED_STAGE=1
export NINFER_VOCAB_COUNT_DIR=/home/chris/vocab_counts
export NINFER_TP_ONESHOT_AR=1   # THE ARM UNDER ROOT-LOOK

pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve" 2>/dev/null
sleep 2
# VRAM-quiet gate (boot fits with ~1 MiB/die slack — a dying predecessor strands it):
for i in $(seq 1 30); do
  BUSY=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "VRAM%\": [0-9]+|VRAM%\): [0-9]+" | grep -oE "[0-9]+$" | sort -rn | head -1)
  [ "${BUSY:-0}" -le 2 ] && break
  sleep 2
done
T0=$(date +%s)
setsid nohup "$BIN" /home/chris/models/qwen3_8_27b_nvfp4.ninfer --port 8100 --devices 0,1,2,3 \
  --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 --greedy \
  --default-max-tokens 16 --spec mtp --draft-tokens 2 --allow-nvfp4-weights \
  > "$LOG" 2>&1 < /dev/null &
SRV=$!
echo "boot $LABEL bin=$(basename "$BIN" | sed 's/ninfer-serve_//;s/\.bin//') pid=$SRV t0=$(date -Ins)"

VERDICT=UNKNOWN
STOP_ON_WARMUP=${STOP_ON_WARMUP:-0}
if [ "$STOP_ON_WARMUP" = "1" ]; then
  # Freeze the whole process (watchdog included) the moment warmup starts, so the
  # +2s B3 _exit(72) cannot race the stack capture. Then: stacks -> SIGTERM.
  for i in $(seq 1 $((TMO * 10))); do
    sleep 0.2
    PID=$(pgrep -f "^$BIN" | head -1)
    [ -n "${PID:-}" ] || continue
    if grep -q "warming up" "$LOG" 2>/dev/null; then
      sleep 0.4   # let the first collectives get issued
      kill -STOP "$PID" 2>/dev/null
      sleep 1
      {
        echo "=== stacks (SIGSTOP-frozen) $LABEL pid=$PID $(date -Ins) ==="
        for t in /proc/$PID/task/*; do
          tid=$(basename "$t")
          echo "--- tid=$tid wchan=$(cat $t/wchan 2>/dev/null) stat=$(awk '{print $3}' $t/stat 2>/dev/null) syscall=$(cat $t/syscall 2>/dev/null | awk '{print $1}')"
        done
        timeout 60 gdb -p "$PID" -batch -ex "set pagination off" -ex "thread apply all bt 15" 2>/dev/null
      } > "$STK" 2>&1
      VERDICT=STOPCAP
      break
    fi
  done
else
  for i in $(seq 1 "$TMO"); do
    sleep 1
    if grep -q "listening on http" "$LOG" 2>/dev/null; then VERDICT=CLEAN; break; fi
    if grep -q "AR-WEDGE-WATCHDOG" "$LOG" 2>/dev/null; then sleep 3; VERDICT=WEDGED_B3; break; fi
    if grep -q "AR-FAILOUT\|AR-DEADLINE\|AR-DESYNC" "$LOG" 2>/dev/null; then sleep 2; VERDICT=AR_LOUD_DEATH; break; fi
    NOW=$(( $(date +%s) - T0 ))
    if [ "$NOW" -ge "$TMO" ]; then VERDICT=WEDGED_SILENT; break; fi
  done
fi
T1=$(date +%s)

PID=$(pgrep -f "^$BIN" | head -1)
if [ -n "${PID:-}" ] && [ "$VERDICT" != "CLEAN" ] && [ "$VERDICT" != "STOPCAP" ]; then
  {
    echo "=== stacks for $LABEL pid=$PID verdict=$VERDICT at t+$((T1-T0))s $(date -Ins) ==="
    for t in /proc/$PID/task/*; do
      tid=$(basename "$t")
      echo "--- tid=$tid wchan=$(cat $t/wchan 2>/dev/null) stat=$(awk '{print $3}' $t/stat 2>/dev/null)"
    done
    timeout 30 gdb -p "$PID" -batch -ex "set pagination off" -ex "thread apply all bt 12" 2>/dev/null
  } > "$STK" 2>&1
  echo "stacks banked: $STK ($(wc -l < "$STK") lines)"
fi

if [ -n "${PID:-}" ]; then
  kill -TERM "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
  sleep 3
  kill -9 "$PID" 2>/dev/null
fi
sleep 1
echo "VERDICT[$LABEL]=$VERDICT wall=$((T1-T0))s"
grep -E "AR-WEDGE|AR-FAILOUT|listening|warming|VERDICT" "$LOG" | tail -12
exit 0
