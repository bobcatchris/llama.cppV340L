#!/usr/bin/env bash
# w7_oneshot_gdb_repro.sh v5 — LAST diagnostic boot design:
#   gdb parent; queued -ex chain: break _exit; run; bt-all; kill; quit.
#   Shape A: B3 watchdog calls _exit at warmup+2s -> breakpoint -> bt-all.
#   Shape B: no watchdog fire (silent spin) -> monitor SIGABRTs the child ->
#            gdb catches the signal, stops target, runs the queued bt-all.
# Usage: bash w7_oneshot_gdb_repro.sh <bin_path> <label>
set -u
BIN=${1:?bin path}
LABEL=${2:?label}
OUT=/home/chris/worktrees/amd-wo-w7-body/results/amd/coherence/w7_oneshot_rootlook
LOG=$OUT/${LABEL}_serve.log
GDBC=$OUT/${LABEL}_gdb_console.txt
mkdir -p "$OUT"

export NINFER_ALLOW_NVFP4_TP2=1
export NINFER_WORKSPACE_MIB=96
export NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/multi_gpu/data/qwen38_draft_vocab_ids.json
export NINFER_MTP_TAIL_ASYNC=1
export NINFER_H2D_PINNED_STAGE=1
export NINFER_VOCAB_COUNT_DIR=/home/chris/vocab_counts
export NINFER_TP_ONESHOT_AR=1

pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve" 2>/dev/null
sleep 2
for i in $(seq 1 30); do
  BUSY=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "VRAM%\): [0-9]+" | grep -oE "[0-9]+$" | sort -rn | head -1)
  [ "${BUSY:-0}" -le 2 ] && break
  sleep 2
done

T0=$(date +%s)
timeout 300 gdb -q -nx \
  -ex "set pagination off" -ex "set confirm off" -ex "set debuginfod enabled off" \
  -ex "set args /home/chris/models/qwen3_8_27b_nvfp4.ninfer --port 8100 --devices 0,1,2,3 --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 --greedy --default-max-tokens 16 --spec mtp --draft-tokens 2 --allow-nvfp4-weights > $LOG 2>&1 < /dev/null" \
  -ex "break _exit" \
  -ex "run" \
  -ex "echo \n ===== WEDGE-STATE STACK DUMP =====\n" \
  -ex "info threads" \
  -ex "thread apply all bt 20" \
  -ex "kill" \
  -ex "quit" \
  "$BIN" > "$GDBC" 2>&1 &
GDBPID=$!

# monitor: SIGABRT the child if warmup started but nothing loud happened within 6s
CHILD=""
for i in $(seq 1 600); do
  sleep 0.5
  NOW=$(( $(date +%s) - T0 ))
  if [ -z "$CHILD" ]; then CHILD=$(pgrep -P "$(pgrep -P $GDBPID | head -1)" 2>/dev/null | head -1); [ -z "$CHILD" ] && CHILD=$(pgrep -f "^$BIN" | head -1); fi
  if [ -n "$CHILD" ] && grep -q "warming up" "$LOG" 2>/dev/null; then
    sleep 6
    if ! grep -q "AR-WEDGE-WATCHDOG\|AR-FAILOUT\|listening on" "$LOG" 2>/dev/null; then
      echo "monitor: silent 6s post-warmup at t+${NOW}s -> SIGABRT child $CHILD" >&2
      kill -ABRT "$CHILD" 2>/dev/null
    fi
    break
  fi
  [ "$NOW" -ge 240 ] && break
done

wait $GDBPID 2>/dev/null
pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve" 2>/dev/null
sleep 2
echo "DONE $LABEL threads_dumped=$(grep -c '^Thread' "$GDBC") bphits=$(grep -c 'Breakpoint 1,' "$GDBC") sigabrt=$(grep -c 'SIGABRT' "$GDBC")"
