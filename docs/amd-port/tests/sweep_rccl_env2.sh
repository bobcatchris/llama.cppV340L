#!/usr/bin/env bash
# RCCL TRANSPORT DESK A2 combo mini-sweep: refine the one winning knob
# (NCCL_MIN_NCHANNELS=4) and run the reproducibility repeats.
set -u
cd /media/chris/ssd128/llamacpp/wt-rccl-transport

PROBE_ARGS="--inproc --iters 2000 --warmup 300 --burst 256 --timeout 600"

run_cfg() { # label, env assignments...
    local label="$1"; shift
    echo "=== config $label: $* ==="
    if [ "$#" -eq 0 ]; then
        docs/amd-port/tests/run_rccl_multidie.sh "$label" $PROBE_ARGS
    else
        env "$@" docs/amd-port/tests/run_rccl_multidie.sh "$label" $PROBE_ARGS
    fi
    sleep 10 # cool gap between configs
}

run_cfg a2b_ch4ll128       NCCL_MIN_NCHANNELS=4 NCCL_PROTO=LL128
run_cfg a2b_ch4ll          NCCL_MIN_NCHANNELS=4 NCCL_PROTO=LL
run_cfg a2b_ch5            NCCL_MIN_NCHANNELS=5
run_cfg a2b_ch6            NCCL_MIN_NCHANNELS=6
run_cfg a2b_ch12           NCCL_MIN_NCHANNELS=12
run_cfg a2b_ch4max4        NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
run_cfg a2b_ch4nt1024      NCCL_MIN_NCHANNELS=4 NCCL_NTHREADS=1024
run_cfg a2b_ch4            NCCL_MIN_NCHANNELS=4
run_cfg a2b_ch4_rep2       NCCL_MIN_NCHANNELS=4
run_cfg a2b_ch4nt1024_rep2 NCCL_MIN_NCHANNELS=4 NCCL_NTHREADS=1024
echo "SWEEP2 DONE"
