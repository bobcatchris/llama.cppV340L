#!/usr/bin/env bash
# run_w25_wides2r_bench.sh - one-command die-3 window for the wide-s2r desk.
# W25/C3 (W23 receipt B2): ws2r arm vs base/share/s2r at T=4, oracle-gated.
#
# Lock law: check-and-hold /tmp/campaign_gpu_boot.lock (atomic noclobber
# create, desk stamp with pid), run on die 3 ONLY
# (HIP_VISIBLE_DEVICES=3), release in the EXIT trap - and only if the
# lock we see at release is still our own stamp.
#
# usage: docs/amd-port/tests/run_w25_wides2r_bench.sh [gguf] [maxwait_s]
#   gguf defaults to /media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
#   maxwait_s defaults to 21600 (6 h); the script polls every 60 s
set -u

GGUF="${1:-/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf}"
MAXWAIT="${2:-21600}"
LOCK=/tmp/campaign_gpu_boot.lock
BENCH=/tmp/bench_mmvq_real_w25
OUT=/tmp/w25_wides2r_bench_$(date +%H%M%S).log
DIE=3

if [ ! -x "$BENCH" ]; then
    echo "bench binary missing: $BENCH (build it first, see receipt W25)" | tee "$OUT"
    exit 1
fi
if [ ! -f "$GGUF" ]; then
    echo "gguf missing: $GGUF" | tee "$OUT"
    exit 1
fi

# --- lock: poll every 60 s until free, then take it atomically -----------
STAMP=$(printf '{"desk":"wide-s2r","pid":%s,"ts":%s,"start":"%s","arm":"w25-wides2r-bench"}' \
    "$$" "$(date +%s)" "$(date '+%Y-%m-%d %H:%M:%S')")
waited=0
while true; do
    if (set -C; printf '%s\n' "$STAMP" > "$LOCK") 2>/dev/null; then
        break
    fi
    if [ "$waited" -ge "$MAXWAIT" ]; then
        echo "w25: lock still held after ${MAXWAIT}s, giving up" | tee -a "$OUT"
        exit 2
    fi
    sleep 60
    waited=$((waited + 60))
done

echo "w25: lock acquired pid=$$ -> $OUT"
trap 'if grep -q "\"pid\":$$" "$LOCK" 2>/dev/null; then rm -f "$LOCK"; echo "w25: lock released"; fi' EXIT

# --- die 3 only ----------------------------------------------------------
export HIP_VISIBLE_DEVICES=$DIE
export LD_LIBRARY_PATH=/media/chris/ssd128/llamacpp/wt-wides2r/build-bench/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

run() { echo "+ $*" | tee -a "$OUT"; "$@" 2>&1 | tee -a "$OUT"; echo "step-exit:${PIPESTATUS[0]}" | tee -a "$OUT"; }

# oracle + timing, same session, base-dup drift control at the end;
# 30 iters x 4 reps per arm per the W23 bench plan
for t in iq4_xs q4_K iq3_s iq3_xxs q5_K; do
    run "$BENCH" "$GGUF" "$t" 4 base,share,s2r,ws2r,base 30 4
done

# occupancy / VGPR answer per arm (the C3 risk item: reg growth vs CTAs/CU)
for t in iq4_xs q4_K iq3_s iq3_xxs q5_K; do
    run "$BENCH" "$GGUF" "$t" 4 base,s2r,ws2r 1 1 --occupancy
done

echo "w25: bench complete -> $OUT"
