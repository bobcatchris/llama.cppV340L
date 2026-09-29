#!/usr/bin/env bash
# W27 Q40-PREFILL DESK - staged oracle + timing, die 3 only (PCI 0000:0d:00.0).
# ONE COMMAND: bash docs/amd-port/scripts/run_q40prefill_bench.sh
#
# Laws: /tmp/campaign_gpu_boot.lock check-and-hold (single sleeps <= 150 s,
# JSON line desk/arm/ts/pid, stale-holder removal only with a dead pid),
# PCI resolution at runtime (never trust a hardcoded HIP index - W17's
# "=3" annotation does not match the rocminfo/rocm-smi enumeration), one
# short bench process per invocation, no background children to kill,
# lock released with evidence at exit. E-117 engagement line check: the
# GGML_CUDA_FATTN_TILE_Q40_PREFILL WARN line must appear once per run.
#
# Oracle rules (W24 2.3 / W16-W17): v11p must be BIT-EXACT, or DUST with
# max_rel <= 1.3e-3 (owner sign-off required before serve); basedupp must be
# bit-exact (bench exits 1 otherwise); staged KT/VT/KQ must be "=ok" - any
# "KT-X" is the W17 miscompile signature => GARBAGE => STOP, do not sweep
# more variants (scalar stores are the only exact shape).
#
# Timing: --prefill arm set (basep = served f16-pool <8,2>, v11p, basedupp;
# --prefill implies the solo 3-arm discipline), rep 0 discarded in-bench,
# niter + BENCH_WARMUP sized per depth (a 199936 M=512 launch is seconds,
# not the 90 ms W24 hoped - keep warmup tiny or the run takes hours).
# Depth 199936 (256-multiple) maps the served ~199000 cell; 199000 itself is
# rejected by the instrument (must be a multiple of 256).
set -u
set -o pipefail

WORKTREE=/media/chris/ssd128/llamacpp/wt-q40prefill
BENCH=/tmp/fa40_bench
BENCH_DBG=/tmp/fa40_bench_dbg
MODEL=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
RES=$WORKTREE/docs/amd-port/results
LOCK=/tmp/campaign_gpu_boot.lock
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$RES/W27_q40prefill_bench_$STAMP.log
STAMPF=$RES/W27_q40prefill_stamp_$STAMP.txt

stamp() { echo "$(date +%H%M%S) $*" | tee -a "$STAMPF" >> "$LOG"; }

# --- preconditions -----------------------------------------------------------
for f in "$BENCH" "$BENCH_DBG" "$MODEL"; do
  [ -e "$f" ] || { echo "MISSING $f"; exit 1; }
done
stamp "BIN $BENCH -> $(ldd "$BENCH" 2>/dev/null | grep ggml-hip)"
[ "$(ldd "$BENCH" | grep -c "$WORKTREE/build-bench/bin")" = "1" ] \
  || { echo "BENCH-NOT-FROM-THIS-TREE"; exit 1; }

# --- svm-hog advisory --------------------------------------------------------
if pgrep -fi "svm.hog|svm_hog" > /dev/null 2>&1; then
  echo "SVM-HOG ACTIVE - aborting (run later)"; exit 1
fi

# --- resolve the bench die by PCI 0000:0d:00.0 --------------------------------
# rocminfo GPU-agent order = HIP device order; HIP idx = agent - 2 (2 CPUs).
DIE3=$(rocminfo 2>/dev/null | awk '
  /^Agent [0-9]+[ \t]*$/ {ag=$2; isgpu=0}
  /Device Type:[ \t]*GPU/ {isgpu=1}
  /BDFID/ {if (isgpu) {bus=int($2/256)%256; if (bus==13) print ag-2; isgpu=0}}' | head -1)
[ -n "$DIE3" ] || { echo "PCI-RESOLVE-FAIL (0d:00.0 not found)"; exit 1; }
rocm-smi --showbus 2>/dev/null | grep -E "^GPU\[$DIE3\]" | grep -qi "0D:00.0" \
  || { echo "PCI-CROSSCHECK-FAIL idx=$DIE3"; exit 1; }
stamp "PCI-RESOLVE 0000:0d:00.0 -> HIP_VISIBLE_DEVICES=$DIE3"

# card of the bench die, for the VRAM idle gate
CARD=$(for c in /sys/class/drm/card[0-9]*; do
  [ "$(basename "$(readlink -f "$c/device")" 2>/dev/null)" = "0000:0d:00.0" ] && basename "$c" && break
done)
[ -n "$CARD" ] || { echo "CARD-RESOLVE-FAIL"; exit 1; }

die_busy_mib() {  # vram used on the bench die, MiB
  echo $(( $(cat "/sys/class/drm/$CARD/device/mem_info_vram_used" 2>/dev/null || echo 999999999) / 1048576 ))
}
die_hot_c() {     # max junction temp across the 4 dies (house law)
  local p h t m=0
  for p in 0000:05:00.0 0000:08:00.0 0000:0d:00.0 0000:10:00.0; do
    h=$(ls -d /sys/bus/pci/devices/$p/hwmon/hwmon* 2>/dev/null | head -1)
    t=$(($(cat "$h/temp2_input" 2>/dev/null || cat "$h/temp1_input" 2>/dev/null || echo 0)/1000))
    [ "$t" -gt "$m" ] && m=$t
  done
  echo $m
}

# --- lock check-and-hold ------------------------------------------------------
WAITED=0
while [ -f "$LOCK" ]; do
  HOLDER=$(cat "$LOCK" 2>/dev/null)
  HPID=$(echo "$HOLDER" | sed -n 's/.*"pid":\([0-9]*\).*/\1/p')
  [ -z "$HPID" ] && HPID=$(echo "$HOLDER" | sed -n 's/.*\bpid=\([0-9]*\).*/\1/p')
  if [ -n "$HPID" ] && ! kill -0 "$HPID" 2>/dev/null; then
    echo "LOCK: stale holder pid $HPID dead, removing with evidence: $HOLDER"
    rm -f "$LOCK"
    break
  fi
  [ "$WAITED" -ge 14400 ] && { echo "LOCK-TIMEOUT (4 h)"; exit 1; }
  echo "LOCK: held, waiting ($WAITED s): $HOLDER"
  sleep 150
  WAITED=$((WAITED+150))
done
printf '{"desk":"q40-prefill","arm":"W27-oracle-timing","ts":"%s","pid":%d}\n' \
  "$(date +%s)" "$$" > "$LOCK"
grep -q '"desk":"q40-prefill"' "$LOCK" || { echo "LOCK-LOST"; exit 1; }
trap 'echo "LOCK: released by pid $$"; rm -f "$LOCK"' EXIT
stamp "LOCK-HELD pid $$ die=$DIE3 card=$CARD"

# --- cool / idle gate ----------------------------------------------------------
WAITED=0
while [ "$(die_hot_c)" -ge 60 ] || [ "$(die_busy_mib)" -ge 200 ]; do
  [ "$WAITED" -ge 1500 ] && { echo "GATE-TIMEOUT hot=$(die_hot_c)C vram=$(die_busy_mib)MiB"; exit 1; }
  stamp "GATE-WAIT hot=$(die_hot_c)C vram=$(die_busy_mib)MiB"
  sleep 150
  WAITED=$((WAITED+150))
done
stamp "GATE-PASS hot=$(die_hot_c)C vram=$(die_busy_mib)MiB"

export HIP_VISIBLE_DEVICES=$DIE3

# --- phase 1: staged oracle (dbg build, one process per cell) ------------------
stamp "PHASE1 ORACLE begin"
ORC=0
oracle_cell() {  # depth M used label
  local d=$1 m=$2 u=$3 lbl=$4
  local args="--oracle --prefill --depth $d --M $m"
  local cell=/tmp/q40p_cell_$lbl.log
  [ "$u" != "0" ] && args="$args --used $u"
  stamp "CELL $lbl: $args"
  echo "== CELL $lbl ==" >> "$LOG"
  if ! GGML_FATTN_Q40_DBG=1 "$BENCH_DBG" "$MODEL" 2 1 $args | tee -a "$LOG" > "$cell"; then
    stamp "CELL $lbl: BENCH-FAIL (basedupp nondeterministic or hard error)"
    ORC=1
    return
  fi
  grep -E "ORACLE v11p|STAGED v11p" "$cell" >> "$STAMPF"
  local cls rel ktok
  cls=$(grep "ORACLE v11p" "$cell" | sed -n 's/.*: \([A-Z-]*\) .*/\1/p' | tail -1)
  rel=$(grep "ORACLE v11p" "$cell" | sed -n 's/.*max rel \([0-9.e+-]*\).*/\1/p' | tail -1)
  ktok=$(grep -c "STAGED v11p vs base: KT=ok VT=ok KQ=ok" "$cell")
  if [ "$ktok" != "1" ]; then
    stamp "CELL $lbl: GARBAGE (staged tile mismatch - W17 signature) => STOP"
    ORC=1; return
  fi
  if [ "$cls" = "BIT-EXACT" ]; then
    stamp "CELL $lbl: EXACT"
  elif [ "$cls" = "DUST" ] && awk -v r="$rel" 'BEGIN{exit !(r+0 <= 1.3e-3)}'; then
    stamp "CELL $lbl: DUST-PASS (max_rel $rel <= 1.3e-3, owner sign-off needed)"
  else
    stamp "CELL $lbl: GARBAGE (class $cls max_rel $rel > 1.3e-3) => STOP"
    ORC=1; return
  fi
  sleep 30   # settle between invocations, single sleep <= 150 s
}
oracle_cell  7168 512 0     m512_d7168
oracle_cell 10240 512 0     m512_d10240
oracle_cell 32768 512 0     m512_d32768
oracle_cell 65536 512 0     m512_d65536
oracle_cell 102400 512 0    m512_d102400
oracle_cell 199936 512 0    m512_d199936
oracle_cell  7168 268 0     m268_d7168
oracle_cell 10496 268 10235 m268_d10496_used10235
oracle_cell 10496 512 10235 m512_d10496_used10235
if [ "$ORC" != "0" ]; then
  stamp "PHASE1 ORACLE FAIL - timing NOT run; see $LOG"
  exit 1
fi
stamp "PHASE1 ORACLE PASS"
sleep 60

# --- phase 2: timing (clean binary, rep 0 discarded in-bench) ------------------
stamp "PHASE2 TIMING begin (prefill arm set = basep/v11p/basedupp)"
time_cell() {  # depth niter warmup label
  local d=$1 ni=$2 wu=$3 lbl=$4
  stamp "TIMING $lbl d=$d niter=$ni BENCH_WARMUP=$wu reps=5"
  echo "== TIMING $lbl ==" >> "$LOG"
  BENCH_WARMUP=$wu "$BENCH" "$MODEL" "$ni" 5 --prefill --solo --depth "$d" --M 512 >> "$LOG" 2>&1 \
    || { stamp "TIMING $lbl: BENCH-FAIL"; exit 1; }
  sleep 30
}
time_cell  10240 4 12 d10240
time_cell  51200 2  8 d51200
time_cell 102400 2  6 d102400
time_cell 199936 1  4 d199936
stamp "PHASE2 TIMING done - see $LOG"
stamp "DONE $STAMP"
echo "W27 bench complete: $LOG"
