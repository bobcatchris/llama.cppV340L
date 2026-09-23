#!/usr/bin/env bash
# RCCL TRANSPORT DESK runner: check-and-hold the campaign GPU boot lock
# (JSON line), cool-die gate on ALL FOUR dies, run the multi-die probe
# (one process per die), release. One invocation = one config = one lock
# window; the caller exports any NCCL_*/RCCL_* env for the A2 sweep and
# passes a label for the session log name.
#
# usage: [NCCL_*=... ] run_rccl_multidie.sh <label> [probe args...]
set -u

LOCK=/tmp/campaign_gpu_boot.lock
DESK=rccl-transport
WT=/media/chris/ssd128/llamacpp/wt-rccl-transport
BIN=/tmp/bin_rccl_multidie_probe
LABEL=${1:?usage: run_rccl_multidie.sh <label> [probe args...]}; shift
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

[ -x "$BIN" ] || { echo "no probe binary at $BIN; compile first (hipcc -O2 -x hip"; \
    echo "  $WT/docs/amd-port/tests/rccl_multidie_probe.cu -o $BIN -lrccl)"; exit 6; }

echo "ENV: $(env | grep -E '^(NCCL|RCCL|HIP|GGML)_' | sort | tr '\n' ' ')"
"$BIN" "$@" 2>&1 | tee "$OUT"
echo "PROBE-EXIT:${PIPESTATUS[0]}"

release_lock
trap - EXIT
echo "log: $OUT"
