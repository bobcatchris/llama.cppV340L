#!/usr/bin/env bash
# PERSISTENT-SERVER BATTERY PROTOCOL (prototype runner, W21 / ledger E-134c).
# Boots llama-server ONCE per arm, runs N guard-battery cells against it
# with an explicit slot erase between cells, tears down ONCE. Cuts KFD SVM
# allocation churn per decision from N boot/teardown cycles to 1 per arm
# (E-134: every hard death clustered at boot/teardown churn).
#
# Design + owner brief: docs/amd-port/results/W21_persistent_server_brief_2026-09-24.md
#   Usage: run_persistent_battery.sh --arm <name> [--cells N] [--ctx C]
#            [--extra-env "A=1 B=2"] [--extra-args "..."] [--decode-only]
#            [--port P] [--dry-run]
#   Cell 1 is a SCRUB cell (run, stamped VOID, never banked): it absorbs
#   first-call jitter (workspace pools, draft shape cache warmup) so banked
#   cells 2..N share one warm regime. Position-matched cells compare across
#   arms; interleaved pairing stays at the promotion tier (one classic
#   window per E-119 before any decision banks).
#   Every cell still passes guard_battery.py's provenance gate (commit +
#   binary sha + cmake sha + launch config) - unchanged verdict lines.
# Zero-GPU law: --dry-run prints the full plan and touches nothing.
set -u

ARM="regress"
CELLS=4
CTX=200000
EXTRA_ENV=""
EXTRA_ARGS=""
PORT=8081
DECODE_ONLY=0
DRY_RUN=0
SETTLE=180          # inter-cell churn spacing (E-134c), seconds
IDLE_WAIT=5         # in-battery cooldown, as the classic windows pass it

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --cells) CELLS="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    --extra-env) EXTRA_ENV="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --decode-only) DECODE_ONLY=1; shift ;;
    --settle) SETTLE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "PERSIST-FAIL: unknown arg $1" >&2; exit 2 ;;
  esac
done

REPO=/media/chris/ssd128/llamacpp/llama.cpp
BIN="$REPO/build-hip/bin/llama-server"
MODEL=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
TESTS="$REPO/docs/amd-port/tests"
BASEENV="GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_MMVQ_IQ3S_SHARE=1 GGML_CUDA_MMVQ_IQ3XXS_SHARE=1 GGML_CUDA_MMVQ_Q3K_SHARE=1 GGML_CUDA_MMVQ_Q4K_SHARE=1 GGML_CUDA_MMVQ_Q5K_SHARE=1 GGML_CUDA_MMVQ_Q6K_SHARE=1 LLAMA_DRAFT_FAST_TOPK=1 LLAMA_DRAFT_PACKED_GET=1 LLAMA_DRAFT_LIGHT_SYNC=1 LLAMA_VERIFY_ROW_SAMPLING=1 LLAMA_ASYNC_INPUT=1 GGML_PINNED_DEV_COPY=1"
LOCK=/tmp/campaign_gpu_boot.lock
SLOT_SAVE_DIR=/tmp/persist_slot_save
LOG="/home/chris/persist_${ARM}"
BASE_URL="http://127.0.0.1:${PORT}"
BATTERY_ARGS="--baseline baseline_tp3_200k.json --idle-wait ${IDLE_WAIT}"
[ "$DECODE_ONLY" = 1 ] && BATTERY_ARGS="$BATTERY_ARGS --decode-only"
LAUNCH_CONFIG="HIP_VISIBLE_DEVICES=0,1,2,3 $BASEENV $EXTRA_ENV llama-server -m $MODEL -ngl 999 -sm tensor -c $CTX --batch-size 512 --ubatch-size 512 -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp --device ROCm0,ROCm1,ROCm2,ROCm3 --slot-save-path $SLOT_SAVE_DIR --port $PORT -t 8 $EXTRA_ARGS"

hog_count() {  # same per-boot KFD hog counter the svm watchdog reads
  journalctl -b --no-pager -k 2>/dev/null \
    | grep -oE "hogged CPU for >[0-9]+us [0-9]+ times" \
    | grep -oE "[0-9]+ times" | grep -oE "[0-9]+" | sort -n | tail -1
}

echo "== PERSISTENT BATTERY PLAN (arm=$ARM cells=$CELLS ctx=$CTX port=$PORT) =="
echo "launch: $LAUNCH_CONFIG"
echo "cells : 1=SCRUB(VOID) then $((CELLS-1)) banked cells, slot erase + ${SETTLE}s settle between, teardown once"
if [ "$DRY_RUN" = 1 ]; then
  echo "-- DRY-RUN: no GPU touch, no lock write, no server, no curl --"
  echo "per-cell steps:"
  echo "  1. wait while $LOCK held (svm-watchdog pause point)"
  echo "  2. [cells 2..N] curl -X POST $BASE_URL/slots/0?action=erase  (witness: n_erased)"
  echo "  3. settle ${SETTLE}s + hog check (hog>=3 -> wait, watchdog owns churn)"
  echo "  4. guard_battery.py --port $PORT $BATTERY_ARGS --server-binary $BIN \\"
  echo "       --launch-config \"$LAUNCH_CONFIG\" --server-log $LOG.server \\"
  echo "       --output-jsonl $LOG.cell<k>.jsonl"
  echo "  5. grep verdict lines into $LOG; stamp cell_pos + uptime + hog"
  echo "teardown: kill server once, rm lock, final settle ${SETTLE}s"
  echo "churn math: server cycles per window = 1 (this arm) vs 1-per-cell today"
fi

# --- fail-loud gates (E-133 arm-identity law, inherited verbatim) ---
GATE="OK"
gate_fail() { echo "GATE-FAIL: $1" >&2; GATE="FAIL"; }
STAMP="$(git -c safe.directory="$REPO" -C "$REPO" log --oneline -1 2>/dev/null)"
[ -n "$STAMP" ] || gate_fail "empty git stamp"
[ -x "$BIN" ] || gate_fail "missing server binary $BIN"
[ -f "$MODEL" ] || gate_fail "missing model $MODEL"
[ -f "$TESTS/baseline_tp3_200k.json" ] || gate_fail "missing baseline"
[ -f "$TESTS/guard_battery.py" ] || gate_fail "missing guard_battery.py"
BIN_TS="$(stat -c %Y "$BIN" 2>/dev/null)" || gate_fail "cannot stat $BIN"
HEAD_TS="$(git -C "$REPO" log -1 --format=%ct)"
if [ -n "$HEAD_TS" ] && [ -n "$BIN_TS" ] && [ "$BIN_TS" -lt "$HEAD_TS" ]; then
  gate_fail "binary predates HEAD ($STAMP); REBUILD before serving"
fi
if [ "$GATE" = "FAIL" ]; then
  echo "DRY-RUN-VERDICT: FAIL (gates failed above; a real run would have exited before boot)"
  [ "$DRY_RUN" = 1 ] && exit 2
  exit 2
fi
echo "ARM-IDENTITY-OK: $STAMP"

if [ "$DRY_RUN" = 1 ]; then
  echo "DRY-RUN-OK: plan is complete; binary/model/baseline/battery present, gates pass"
  exit 0
fi

# --- lock check-and-hold (sibling convention; watchdog pauses us here) ---
while [ -f "$LOCK" ]; do sleep 60; done

# --- boot ONCE ---
printf '{"desk":"persist-battery","arm":"%s","ts":"%s"}\n' "$ARM" "$(date +%s)" > "$LOCK"
{
  echo "== ARM IDENTITY (persistent epoch $(date +%s)) =="
  echo "arm: $ARM  extra_env: ${EXTRA_ENV:-none}  extra_args: ${EXTRA_ARGS:-none}"
  echo "cells: $CELLS (cell 1 = SCRUB/VOID)  settle: ${SETTLE}s  port: $PORT"
  echo "BASEENV: $BASEENV"
  echo "bin: $BIN  (built from: $STAMP)"
} > "$LOG"
mkdir -p "$SLOT_SAVE_DIR"
: > "$LOG.server"

env HIP_VISIBLE_DEVICES=0,1,2,3 $BASEENV $EXTRA_ENV "$BIN" -m "$MODEL" \
  -ngl 999 -sm tensor -c "$CTX" --batch-size 512 --ubatch-size 512 \
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
  --device ROCm0,ROCm1,ROCm2,ROCm3 --slot-save-path "$SLOT_SAVE_DIR" \
  --port "$PORT" -t 8 $EXTRA_ARGS >> "$LOG.server" 2>&1 &
SRV=$!
BOOT_OK=0
for i in $(seq 1 40); do
  sleep 20
  curl -s -m 5 "$BASE_URL/health" | grep -q ok && { BOOT_OK=1; break; }
  kill -0 $SRV 2>/dev/null || break
done
if [ "$BOOT_OK" != 1 ]; then
  echo "PERSIST-FAIL: boot-timeout arm=$ARM" >> "$LOG"
  echo "PERSIST-FAIL: boot-timeout arm=$ARM" >&2
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  rm -f "$LOCK"
  exit 1
fi
BOOT_TS=$(date +%s)
echo "PERSIST-BOOT-OK: arm=$ARM pid=$SRV ts=$BOOT_TS" | tee -a "$LOG"

WORST="PASS"
ERASED=""
DECODE_TPS_LAST=""
for k in $(seq 1 "$CELLS"); do
  # watchdog pause point: never churn into a held lock
  while [ -f "$LOCK" ] && ! grep -q persist-battery "$LOCK" 2>/dev/null; do sleep 30; done

  TAG="c${k}"
  if [ "$k" -gt 1 ]; then
    # explicit cell-boundary reset: clear slot 0 KV (target + draft) + tokens
    ERASED=""
    for try in 1 2 3 4 5; do
      ERASED="$(curl -s -m 30 -X POST "$BASE_URL/slots/0?action=erase")"
      echo "$ERASED" | grep -q n_erased && break
      sleep 20
    done
    echo "$ERASED" | grep -q n_erased || { echo "PERSIST-FAIL: slot erase refused (cell $k)" | tee -a "$LOG"; break; }
    echo "PERSIST-ERASE: cell=$TAG $ERASED" >> "$LOG"
    sleep "$SETTLE"
  else
    sleep 30   # post-boot settle for the scrub cell
  fi
  HOG="$(hog_count)"; HOG=${HOG:-0}
  UPTIME=$(( $(date +%s) - BOOT_TS ))
  MARK="BANKED"; [ "$k" = 1 ] && MARK="SCRUB-VOID"
  {
    echo "== CELL $TAG ($MARK) pos=$k uptime_s=$UPTIME hog=$HOG ts=$(date +%s) =="
    echo "cell_provenance: arm=$ARM cell_pos=$k server_uptime_s=$UPTIME hog_before=$HOG erase=$ERASED"
  } >> "$LOG"
  (cd "$TESTS" && python3 guard_battery.py --port "$PORT" \
     $BATTERY_ARGS \
     --server-binary "$BIN" \
     --launch-config "$LAUNCH_CONFIG" \
     --server-log "$LOG.server" --output-jsonl "$LOG.${TAG}.jsonl" 2>&1 \
     | tee "$LOG.${TAG}.battery_full" \
     | grep -E "PROVENANCE|prompt_tps|decode_guard|draft_accept|text_sha256|exact_recall|VERDICT" >> "$LOG")
  CELL_WORST="$(grep -E "^OVERALL VERDICT" "$LOG.${TAG}.battery_full" | tail -1 | awk '{print $3}')"
  CELL_WORST=${CELL_WORST:-UNKNOWN}
  echo "PERSIST-CELL-DONE: cell=$TAG verdict=$CELL_WORST mark=$MARK" | tee -a "$LOG"
  case "$CELL_WORST" in
    FAIL) WORST="FAIL" ;;
    WARN) [ "$WORST" = "PASS" ] && WORST="WARN" ;;
  esac
  [ "$k" = 1 ] || DECODE_TPS_LAST="$(grep -E 'decode_guard' "$LOG.${TAG}.battery_full" | tail -1)"
done

# --- teardown ONCE ---
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
rm -f "$LOCK"
sleep "$SETTLE"
echo "PERSISTENT-VERDICT: arm=$ARM cells=$CELLS banked=$((CELLS-1)) worst=$WORST boot_cycles=1"
echo "PERSISTENT-DONE arm=$ARM (log: $LOG)"
