#!/usr/bin/env bash
# P2P-ALLREDUCE desk W31 probe runner: check-and-hold the campaign GPU
# boot lock (JSON line), cool-die gate, compile + run, release.
#
# STAGED, NOT RUN while U1 (arm=u1-depth) holds the machine.  Run only
# from the coordinator window or when the lock is free.  Budget: well
# under 15 min wall (FEAS seconds, bench ~2-3 min GPU time).
#
# Usage:
#   ./run_p2p_allreduce.sh          # full 4-die session (real thing)
#   ./run_p2p_allreduce.sh 2        # early 2-die enablement check
#                                   # (dies 0,1 = same-switch pair under
#                                   # PM8533 switch A) if a pair frees
set -u

LOCK=/tmp/campaign_gpu_boot.lock
DESK=p2p-ar
WT=/media/chris/ssd128/llamacpp/wt-p2p-ar
RANKS="${1:-4}"
OUT=$WT/docs/amd-port/results/p2p_allreduce_probe_r${RANKS}_$(date +%Y%m%d_%H%M%S).log

hold_lock() {
    while true; do
        if ( set -o noclobber; echo "{\"desk\":\"$DESK\",\"arm\":\"W31-p2p-probe\",\"pid\":$$,\"ts\":\"$(date -Is)\"}" > "$LOCK" ) 2>/dev/null; then
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
echo "lock held by $DESK pid $$ at $(date -Is)"

# cool-die gate < 60 C on every visible die
rocm-smi --showtemp --showuse 2>/dev/null | tee /tmp/p2p_ar_temps.txt
MAXT=$(grep -o '(C): [0-9.]*' /tmp/p2p_ar_temps.txt | grep -o '[0-9.]*$' | sort -n | tail -1)
if [ -z "$MAXT" ]; then echo "no temp sensors read; aborting"; exit 5; fi
if awk -v t="$MAXT" 'BEGIN{exit !(t >= 60)}'; then echo "cool-die gate FAIL (${MAXT} C >= 60); aborting"; exit 5; fi
echo "cool-die gate PASS (max ${MAXT} C)"

cd "$WT/docs/amd-port/probes"
hipcc -O2 -x hip p2p_allreduce_probe.cu -o p2p_allreduce_probe 2>&1
echo "COMPILE-EXIT:$?"
[ -x p2p_allreduce_probe ] || { echo "no binary; abort"; exit 6; }

# run in the background so a wedged handshake can be killed by PID only
./p2p_allreduce_probe --ranks "$RANKS" --iters 2000 \
    --sizes 20480,5120 --arms p2p,host --pacing lockstep,burst --timeout 600 \
    > "$OUT" 2>&1 &
PROBE_PID=$!
echo "probe pid $PROBE_PID (kill by PID only: kill \$PROBE_PID)"
wait "$PROBE_PID"
echo "PROBE-EXIT:$?"

release_lock
echo "log: $OUT"
