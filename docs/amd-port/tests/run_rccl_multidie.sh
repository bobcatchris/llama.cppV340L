#!/usr/bin/env bash
# RCCL TRANSPORT DESK runner: check-and-hold the campaign GPU boot lock
# (JSON line), cool-die gate on ALL FOUR dies, launch ONE PROBE PROCESS PER
# DIE (clean exec from this shell - see probe header note), collect exit
# codes, release. One invocation = one config = one lock window; the caller
# exports any NCCL_*/RCCL_* env for the A2 sweep and passes a label.
#
# usage: [NCCL_*=... ] run_rccl_multidie.sh <label> [--inproc] [probe args...]
set -u

LOCK=/tmp/campaign_gpu_boot.lock
DESK=rccl-transport
WT=/media/chris/ssd128/llamacpp/wt-rccl-transport
BIN=/tmp/bin_rccl_multidie_probe
DIES=(0 1 2 3)
LABEL=${1:?usage: run_rccl_multidie.sh <label> [--inproc] [probe args...]}; shift
OUT=$WT/docs/amd-port/results/rccl_multidie_${LABEL}_$(date +%Y%m%d_%H%M%S).log

hold_lock() {
    while true; do
        if ( set -o noclobber; echo "{\"desk\":\"$DESK\",\"arm\":\"$LABEL\",\"ts\":\"$(date -Is)\",\"pid\":$$}" > "$LOCK" ) 2>/dev/null; then
            return 0
        fi
        echo "lock held; sleeping 150 s" >&2
        sleep 150
    done
}

release_lock() {
    rm -f "$LOCK"
}
trap release_lock EXIT

hold_lock
echo "lock held by $DESK arm=$LABEL at $(date -Is)"

# cool-die gate < 60 C on every die used (card0/1/3/4 = HIP 0..3)
rocm-smi --showtemp --showuse > /tmp/rccl_multidie_temps.txt 2>&1
cat /tmp/rccl_multidie_temps.txt
MAXT=$(grep -o '(C): [0-9.]*' /tmp/rccl_multidie_temps.txt | grep -o '[0-9.]*$' | sort -n | tail -1)
if [ -z "$MAXT" ]; then echo "no temp sensors read; aborting"; exit 5; fi
if awk -v t="$MAXT" 'BEGIN{exit !(t >= 60)}'; then echo "cool-die gate FAIL (${MAXT} C >= 60); aborting"; exit 5; fi
echo "cool-die gate PASS (max ${MAXT} C)"

# die identity of record: card <-> HIP mapping
rocm-smi --showbus >> /tmp/rccl_multidie_temps.txt 2>&1

[ -x "$BIN" ] || { echo "no probe binary at $BIN; compile first:"; \
    echo "  hipcc -O2 -x hip $WT/docs/amd-port/tests/rccl_multidie_probe.cu -o $BIN -lrccl"; exit 6; }

echo "ENV: $(env | grep -E '^(NCCL|RCCL|HIP|GGML)_' | sort | tr '\n' ' ')"

SHM=/rccl_multidie_$(date +%s)
pids=()
cleanup() {
    for p in "${pids[@]:-}"; do kill -9 "$p" 2>/dev/null; done
    rm -f "/dev/shm$SHM" 2>/dev/null
    release_lock
    trap - EXIT
}
trap cleanup EXIT

{
    if [ "${1:-}" = "--inproc" ]; then
        "$BIN" --inproc "$@"
    else
        for r in 0 1 2 3; do
            HIP_VISIBLE_DEVICES=${DIES[$r]} "$BIN" --worker --rank "$r" --die "${DIES[$r]}" \
                --shm "$SHM" --init-delay "${INIT_DELAY_S:-0}" "$@" &
            pids+=($!)
        done
        fail=0
        for p in "${pids[@]}"; do
            wait "$p" || fail=1
        done
        [ "$fail" -eq 0 ] || { echo "a worker FAILED"; exit 2; }
    fi
} 2>&1 | tee "$OUT"
echo "PROBE-EXIT:${PIPESTATUS[0]}"

cleanup
echo "log: $OUT"
