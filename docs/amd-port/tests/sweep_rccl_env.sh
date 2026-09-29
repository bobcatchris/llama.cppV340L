#!/usr/bin/env bash
# RCCL TRANSPORT DESK A2 env sweep: one lock window per config, released
# between configs. Each config = run_rccl_multidie.sh (which takes the
# campaign lock, cool-die gate, runs the probe, releases).
# The A1 baseline config (empty env) is repeated at the end as a control.
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

run_cfg a1_default
run_cfg a2_proto_ll        NCCL_PROTO=LL
run_cfg a2_proto_ll128     NCCL_PROTO=LL128
run_cfg a2_proto_simple    NCCL_PROTO=Simple
run_cfg a2_algo_ring       NCCL_ALGO=Ring
run_cfg a2_algo_tree       NCCL_ALGO=Tree
run_cfg a2_tree_ll128      NCCL_ALGO=Tree NCCL_PROTO=LL128
run_cfg a2_tree_simple     NCCL_ALGO=Tree NCCL_PROTO=Simple
run_cfg a2_ch1             NCCL_MIN_NCHANNELS=1
run_cfg a2_ch4             NCCL_MIN_NCHANNELS=4
run_cfg a2_ch8             NCCL_MIN_NCHANNELS=8
run_cfg a2_nt256           NCCL_NTHREADS=256
run_cfg a2_nt1024          NCCL_NTHREADS=1024
run_cfg a2_p2poff          NCCL_P2P_DISABLE=1
run_cfg a2_shmoff          NCCL_SHM_DISABLE=1
run_cfg a2_msccl           RCCL_MSCCL_ENABLE=1
run_cfg a2_ll128force      RCCL_LL128_FORCE_ENABLE=1
run_cfg a2_noaffinity      NCCL_IGNORE_CPU_AFFINITY=1
echo "SWEEP DONE"
