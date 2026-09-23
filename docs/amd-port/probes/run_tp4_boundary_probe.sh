#!/usr/bin/env bash
# TP4 boundary desk P0 probe runner: check-and-hold the campaign GPU boot
# lock (JSON line), cool-die gate, compile + run on die 3 ONLY, release.
set -u

LOCK=/tmp/campaign_gpu_boot.lock
DESK=tp4-bound
WT=/media/chris/ssd128/llamacpp/wt-tp4-bound
OUT=$WT/docs/amd-port/results/tp4_boundary_probe_d3_$(date +%Y%m%d_%H%M%S).log

hold_lock() {
    while true; do
        if ( set -o noclobber; echo "{\"desk\":\"$DESK\",\"arm\":\"P0-rccl-probe\",\"ts\":\"$(date -Is)\"}" > "$LOCK" ) 2>/dev/null; then
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
echo "lock held by $DESK at $(date -Is)"

# cool-die gate < 60 C on die 3
rocm-smi --showtemp --showuse -d 3 2>/dev/null | tee /tmp/tp4_boundary_temps.txt
MAXT=$(grep -o '(C): [0-9.]*' /tmp/tp4_boundary_temps.txt | grep -o '[0-9.]*$' | sort -n | tail -1)
if [ -z "$MAXT" ]; then echo "no temp sensors read; aborting"; exit 5; fi
if awk -v t="$MAXT" 'BEGIN{exit !(t >= 60)}'; then echo "cool-die gate FAIL (${MAXT} C >= 60); aborting"; exit 5; fi
echo "cool-die gate PASS (max ${MAXT} C)"

# no other die may be touched: HIP_VISIBLE_DEVICES=3
export HIP_VISIBLE_DEVICES=3

cd "$WT/docs/amd-port/probes"
hipcc -O2 -x hip rccl_tp4_boundary_probe.cpp -o rccl_tp4_boundary_probe -lrccl 2>&1
echo "COMPILE-EXIT:$?"
[ -x rccl_tp4_boundary_probe ] || { echo "no binary; abort"; exit 6; }

./rccl_tp4_boundary_probe --ranks 4 --iters 2000 --pipe 16 \
    --sizes 4096,8192,16384,20480,24576,32768 \
    --nsweep 4,3,2,1 --timeout 500 2>&1 | tee "$OUT"
echo "PROBE-EXIT:${PIPESTATUS[0]}"

release_lock
echo "log: $OUT"
