// LLAMA_LAUNCH_TIMELINE=1: host-side launch/replay instrumentation.
// Counts die-level graph replays, allreduce boundaries, backend drains and
// scheduler input copies, and times the host slice of each. All counters are
// cumulative; the meta backend graph_compute (one per served ubatch) emits one
// line carrying the totals so offline analysis can diff consecutive lines.
// Env unset = no counters, no logs, byte-identical behavior.
#pragma once

#include <cstdint>
#include <cstdlib>

struct ggml_launch_timeline_state {
    uint64_t meta_computes   = 0; // meta backend graph_compute calls (one per ubatch)
    uint64_t meta_replays    = 0; // die-level graph_compute calls issued by meta
    uint64_t meta_boundaries = 0; // allreduce points between subgraphs
    uint64_t meta_ar_comm    = 0; // boundaries served by the backend comm (RCCL)
    uint64_t meta_ar_fallback= 0; // boundaries served by the butterfly fallback
    uint64_t ar_comm_us      = 0; // host time inside comm_allreduce
    uint64_t sched_computes  = 0; // scheduler compute_splits calls
    uint64_t sched_inputs    = 0; // split input copies issued
    uint64_t sched_csyncs    = 0; // split backend drains done for input copies
    uint64_t backend_syncs   = 0; // all ggml_backend_synchronize calls
    uint64_t backend_sync_us = 0; // host time blocked in ggml_backend_synchronize
    uint64_t cuda_computes   = 0; // die-level CUDA graph_compute calls
    uint64_t cuda_replays    = 0; // steady cudaGraphLaunch calls
    uint64_t cuda_captures   = 0; // capture + instantiate/update passes
    uint64_t cuda_direct     = 0; // node-loop executions (no graph)
    uint64_t cuda_check_us   = 0; // host time in compat/update bookkeeping
    uint64_t cuda_launch_us  = 0; // host time in cudaGraphLaunch (incl. backpressure)
};

inline ggml_launch_timeline_state g_launch_tl;

inline bool ggml_launch_timeline_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_LAUNCH_TIMELINE");
        return e != nullptr && atoi(e) == 1;
    }();
    return on;
}
