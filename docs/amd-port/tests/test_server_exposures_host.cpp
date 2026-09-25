// Server exposures desk: host-side logic tests (no GPU, no ggml linkage).
//
// Reproduces the two E-039 server exposures from the E-037/E-038 forensics
// (a 7857-token request starved to death while the server served concurrent
// traffic on a unified KV cache) and validates the fixes added to
// tools/server/server-context.cpp:
//   1. Per-slot n_ctx admission ignores global unified KV occupancy.
//      Under kv_unified, n_ctx_seq == n_ctx, so every slot admits requests up
//      to the FULL cache size. The new admission check
//      kv_unified_cells_needed() + kv_unified_admission_fits() requires a
//      task's remaining prompt plus the remaining prompts of all in-flight
//      requests to fit in the free cells together (idle cached prompts are
//      purged first, impossible requests are 400s, everything else is
//      deferred and retried on the next slot release).
//   2. Slot fill order starves an older mid-prompt request for a newer one.
//      The fill loop now iterates slots in FIFO order of task id (oldest
//      first) instead of slot index order.
//
// The sims mirror the update_slots() loop arithmetic: the fill phase adds up
// to n_batch tokens per round in fill order (server counts cells at fill
// time), the decode phase commits views to the cache, halves n_batch on
// KV-full (restoring it after each success) and aborts EVERY slot at
// n_batch == 1 with zero free cells - the "Context size has been exceeded"
// HTTP 500 that killed both requests in the incident.
//
// Build + run:
//   g++ -std=c++17 -Wall -Wextra -o /tmp/test_server_exposures_host
//       docs/amd-port/tests/test_server_exposures_host.cpp && /tmp/test_server_exposures_host

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// ---- mirrors of the fix (server-context.cpp) ---------------------------->

// mirror: kv_unified_cells_needed - cells a task still needs after reusing
// the candidate slot's cached prefix (1 token entry = 1 cell, media included)
static int32_t kv_cells_needed(bool cache_prompt, size_t n_prefix, size_t n_tokens) {
    const size_t n_past = cache_prompt ? n_prefix : 0;

    return std::max<int32_t>(0, (int32_t) n_tokens - (int32_t) n_past);
}

// mirror: kv_unified_admission_fits
static bool kv_admission_fits(int32_t n_need, int32_t n_ctx_total, int32_t n_used, int32_t n_pending) {
    return n_need + n_pending <= n_ctx_total - n_used;
}

// ---- incident-scale simulation ------------------------------------------>

// incident of record (E-037/E-038): 10240-cell unified cache (n_parallel
// auto = 4 slots, kv_unified = true), 512-token batch, two concurrent
// 7857-token prompts
static const int32_t N_CTX    = 10240;
static const int32_t N_BATCH  = 512;
static const int32_t N_PROMPT = 7857;

struct slot_sim {
    int  task_id     = -1;   // monotonic arrival order
    int  slot_id     = -1;
    int  n_placed    = 0;    // server-side count (grows at fill time)
    int  n_committed = 0;    // cells actually held in the cache
    bool admitted    = false;
    bool prompt_done = false;
    bool aborted     = false;

    int32_t n_remaining() const { return N_PROMPT - n_placed; }
    int32_t n_pending()   const { return n_placed - n_committed; }
};

struct server_sim {
    std::vector<slot_sim> slots;
    std::vector<slot_sim *> fill_order; // order used by the last fill()
    int32_t used = 0;                   // committed cells
    int32_t idle_cache = 0;             // cached cells on idle slots (part of used)
    bool fatal = false;
    int32_t fatal_off = -1;

    int32_t free_cells() const { return N_CTX - used; }

    // mirror: pre_decode fill phase - slot index order (legacy) or task id
    // order (FIFO fix); returns the batch size
    int32_t fill(bool fifo, int32_t n_batch) {
        fill_order.clear();
        for (auto & s : slots) {
            if (s.admitted && !s.prompt_done && !s.aborted && s.n_remaining() > 0) {
                fill_order.push_back(&s);
            }
        }
        if (fifo) {
            std::sort(fill_order.begin(), fill_order.end(),
                    [](const slot_sim * a, const slot_sim * b) { return a->task_id < b->task_id; });
        } else {
            std::sort(fill_order.begin(), fill_order.end(),
                    [](const slot_sim * a, const slot_sim * b) { return a->slot_id < b->slot_id; });
        }

        int32_t batch = 0;
        for (auto * s : fill_order) {
            if (batch >= n_batch) {
                break; // batch is full, skip remaining slots
            }
            const int32_t take = std::min(n_batch - batch, s->n_remaining());
            s->n_placed += take;
            batch += take;
        }
        return batch;
    }

    // mirror: decode() retry loop - commit views while they fit, purge idle
    // caches before halving n_batch on KV-full (restore after each success),
    // abort everything at n_batch == 1 with zero free cells
    void decode(int32_t total) {
        int32_t n_batch = N_BATCH;
        int32_t off = 0;

        while (off < total) {
            const int32_t view = std::min(n_batch, total - off);

            if (view <= free_cells()) {
                // commit the view: slots own the batch in fill order
                int32_t left = view;
                for (auto * s : fill_order) {
                    if (left == 0) {
                        break;
                    }
                    const int32_t take = std::min(left, s->n_pending());
                    s->n_committed += take;
                    used += take;
                    left -= take;
                    if (s->n_committed >= N_PROMPT) {
                        s->prompt_done = true;
                    }
                }
                off += view;
                n_batch = N_BATCH; // restored on success
                continue;
            }

            // mirror: try_clear_idle_slots() before halving
            if (idle_cache > 0) {
                used -= idle_cache;
                idle_cache = 0;
                continue;
            }

            if (n_batch == 1) {
                fatal = true; // "Context size has been exceeded. off = %d"
                fatal_off = off;
                for (auto & s : slots) {
                    if (s.admitted) {
                        s.aborted = true; // HTTP 500 for every in-flight request
                    }
                }
                return;
            }
            n_batch /= 2;
        }
    }

    void run_round(bool fifo) {
        const int32_t batch = fill(fifo, N_BATCH);
        if (batch > 0) {
            decode(batch);
        }
    }
};

// mirror of the new admission flow in process_single_task(); unlike the
// legacy per-slot check it accounts for the global unified occupancy and
// purges idle cached prompts (a real side effect) before deferring
enum admission_outcome { ADMIT, DEFER, REJECT_TOOBIG };

static admission_outcome admit_task(server_sim & srv, int32_t cand_task_id,
        int32_t n_need, int32_t n_tokens_raw, bool kv_unified, bool kv_admission) {
    if (!kv_unified || !kv_admission) {
        return ADMIT; // legacy path (or non-unified cache: per-slot regions)
    }

    int32_t n_pending = 0;
    for (const auto & cur : srv.slots) {
        if (cur.admitted && !cur.aborted && cur.task_id != cand_task_id) {
            // cells still to be placed for in-flight prompts
            n_pending += std::max(0, N_PROMPT - std::max(cur.n_placed, cur.n_committed));
        }
    }

    // mirror: purge idle cached prompts before giving up
    while (!kv_admission_fits(n_need, N_CTX, srv.used, n_pending) && srv.idle_cache > 0) {
        srv.used -= srv.idle_cache;
        srv.idle_cache = 0;
    }

    if (!kv_admission_fits(n_need, N_CTX, srv.used, n_pending)) {
        return n_tokens_raw > N_CTX ? REJECT_TOOBIG : DEFER;
    }
    return ADMIT;
}

static slot_sim make_task(int task_id, int slot_id) {
    slot_sim s;
    s.task_id = task_id;
    s.slot_id = slot_id;
    return s;
}

// the incident geometry: task 14 placed 6075 + 512 in-flight = 6587 cells on
// slot 1; the newer task 35 sits on slot 0 (lower fill priority pre-fix)
static server_sim incident() {
    server_sim srv;
    slot_sim a = make_task(14, 1);
    a.admitted = true;
    a.n_placed = 6587;
    a.n_committed = 6587;
    slot_sim b = make_task(35, 0);
    srv.slots = { b, a };
    srv.used = 6587;
    return srv;
}

int main() {
    // ---- 1: unit mirrors of the new admission helpers ----

    // prefix credit: 7857-token task on a slot with a 3584-token cached prefix
    CHECK(kv_cells_needed(true, 3584, 7857) == 4273);
    // cache_prompt = false: no credit, full prompt must fit
    CHECK(kv_cells_needed(false, 3584, 7857) == 7857);
    // slot cache diverges beyond the task prompt: clamp at 0
    CHECK(kv_cells_needed(true, 9000, 7857) == 0);

    // exact fit admits, one cell over rejects
    CHECK(kv_admission_fits(1000, 10240, 9000, 240));
    CHECK(!kv_admission_fits(1001, 10240, 9000, 240));

    // ---- 2: defect 1 reproduction (legacy admission, incident numbers) ----
    //
    // E-037/E-038 log of record: task 35 is admitted because its 7857 tokens
    // fit the per-slot n_ctx (10240 under kv_unified) with no check of the
    // global occupancy. The cache can only supply 3653 more cells.
    {
        server_sim srv = incident();
        slot_sim & a = srv.slots[1];
        slot_sim & b = srv.slots[0];
        b.admitted = true; // legacy admission: zero free-cell check

        CHECK(2 * N_PROMPT > N_CTX); // demand 15714 > supply 10240

        int guard = 0;
        while (!srv.fatal && !b.prompt_done && guard++ < 32) {
            srv.run_round(/*fifo*/ false);
        }

        // the starvation arithmetic closes to the exact cell
        CHECK(b.n_committed == 3653);              // 3584 + 64 + 4 + 1 trickle
        CHECK(srv.used == 10240);                  // zero free cells
        CHECK(srv.fatal);                          // "Context size has been exceeded"
        CHECK(srv.fatal_off == 69);                // batch offsets 0/64/68/69
        CHECK(a.aborted && b.aborted);             // BOTH requests die with HTTP 500
        CHECK(a.n_committed == 6587);              // task 14 never got another cell
        CHECK(a.n_remaining() > 0);                // starved mid-prompt (1270 to go)
    }

    // ---- 3: defect 2 reproduction (legacy fill order) ----
    //
    // Same state, one round: slot index order hands the whole batch to the
    // NEWER request (slot 0) while the older mid-prompt request (slot 1, task
    // 14, needs 1782 more cells) receives none.
    {
        server_sim srv = incident();
        slot_sim & a = srv.slots[1];
        slot_sim & b = srv.slots[0];
        b.admitted = true;

        const int32_t batch = srv.fill(/*fifo*/ false, N_BATCH);
        CHECK(batch == N_BATCH);   // the whole round goes to the newer task
        CHECK(b.n_placed == N_BATCH);
        CHECK(a.n_placed == 6587); // the older request got nothing

        // the fix: oldest task id fills first
        srv.fill(/*fifo*/ true, N_BATCH);
        CHECK(srv.fill_order.front()->task_id == 14);
    }

    // ---- 4: admission decision units (the fix) ----
    {
        server_sim srv = incident();
        const int32_t n_need = kv_cells_needed(true, 0, N_PROMPT);
        CHECK(n_need == 7857);

        // incident admission: need 7857, pending 1270 (task 14 still needs
        // 7857 - 6587), free 3653 - the newcomer must NOT be admitted
        CHECK(!kv_admission_fits(n_need, N_CTX, 6587, 1270));
        CHECK(admit_task(srv, 35, n_need, N_PROMPT, /*kv_unified*/ true, /*kv_admission*/ true) == DEFER);
        CHECK(srv.used == 6587); // nothing purged: no idle caches existed

        // the same request on an empty server admits
        server_sim empty;
        empty.slots = { make_task(35, 0) };
        CHECK(admit_task(empty, 35, n_need, N_PROMPT, true, true) == ADMIT);

        // prefix reuse shrinks the need; with the incident occupancy (free
        // 3653 minus pending 1270) only a large enough cache flips to ADMIT
        server_sim small_reuse = incident();
        CHECK(admit_task(small_reuse, 35, kv_cells_needed(true, 3000, N_PROMPT), N_PROMPT, true, true) == DEFER);
        CHECK(kv_admission_fits(kv_cells_needed(true, 5600, N_PROMPT), N_CTX, 6587, 1270));
        server_sim big_reuse = incident();
        CHECK(admit_task(big_reuse, 35, kv_cells_needed(true, 5600, N_PROMPT), N_PROMPT, true, true) == ADMIT);

        // knobs off: legacy behavior (documented escape hatch)
        server_sim legacy = incident();
        CHECK(admit_task(legacy, 35, n_need, N_PROMPT, false, true) == ADMIT);
        CHECK(admit_task(legacy, 35, n_need, N_PROMPT, true, false) == ADMIT);

        // a request larger than the whole cache is rejected outright
        server_sim big;
        big.slots = { make_task(35, 0) };
        CHECK(admit_task(big, 35, N_CTX + 1, N_CTX + 1, true, true) == REJECT_TOOBIG);
    }

    // ---- 5: purge-then-admit (idle cached prompts are reclaimable) ----
    {
        server_sim srv;
        srv.used = 7865;       // a finished request keeps its cache on the idle slot
        srv.idle_cache = 7865;
        CHECK(admit_task(srv, 35, N_PROMPT, N_PROMPT, true, true) == ADMIT);
        CHECK(srv.idle_cache == 0); // mirror of try_clear_idle_slots() relief
        CHECK(srv.used == 0);
    }

    // ---- 6: fixed trajectory (admission + FIFO) ----
    //
    // Same incident geometry, fixes on: task 35 is deferred at admission;
    // task 14 completes its prompt and finishes; the release pops the
    // deferred task, the stale cache is purged and task 35 runs to
    // completion. No request ever sees HTTP 500.
    {
        server_sim srv = incident();
        slot_sim & a = srv.slots[1];
        slot_sim & b = srv.slots[0];

        // admission of task 35 on arrival
        CHECK(admit_task(srv, 35, N_PROMPT, N_PROMPT, true, true) == DEFER);

        int guard = 0;
        while (!a.prompt_done && !a.aborted && guard++ < 32) {
            srv.run_round(/*fifo*/ true);
        }
        CHECK(a.prompt_done);
        CHECK(!srv.fatal);
        CHECK(a.n_committed == N_PROMPT);          // 7857, the full prompt
        CHECK(b.n_committed == 0);                 // newcomer placed nothing yet

        // task 14 generates a few tokens, hits EOS, releases its slot
        for (int i = 0; i < 8; i++) {
            a.n_committed++;
            srv.used++;
        }
        CHECK(srv.used <= N_CTX);

        // release keeps the prompt cache on the idle slot; the deferred task
        // is re-admitted, purging the stale cache first
        srv.idle_cache = a.n_committed;
        CHECK(admit_task(srv, 35, N_PROMPT, N_PROMPT, true, true) == ADMIT);
        b.admitted = true;

        guard = 0;
        while (!b.prompt_done && !b.aborted && guard++ < 32) {
            srv.run_round(/*fifo*/ true);
        }
        CHECK(b.prompt_done);
        CHECK(!srv.fatal);
        CHECK(b.n_committed == N_PROMPT);
        CHECK(!a.aborted);                         // task 14's response survived
        CHECK(srv.used <= N_CTX);
    }

    // ---- 7: FIFO alone is not enough (documents why admission is primary) --
    //
    // With the fill-order fix only, the older request wins the remaining
    // cells and completes its prompt, but the still-admitted newcomer
    // eventually drains the cache to zero and the nb=1 abort fires during
    // generation - the defect 1 overcommit must be prevented at admission.
    {
        server_sim srv = incident();
        slot_sim & a = srv.slots[1];
        slot_sim & b = srv.slots[0];
        b.admitted = true; // legacy admission let it in

        int guard = 0;
        while (!srv.fatal && guard++ < 32) {
            srv.run_round(/*fifo*/ true);
        }
        CHECK(a.prompt_done);      // the older request now completes its prompt
        CHECK(srv.fatal);          // but the overcommit still ends in the abort
        CHECK(srv.used == N_CTX);  // drained to zero free cells again
    }

    // ---- 8: non-unified cache is out of scope for the admission check ----
    //
    // kv_unified = false gives each slot its own cells region sized
    // n_ctx_seq = n_ctx / n_seq_max, so per-slot n_ctx admission is already
    // exact there - the knob must not engage (mirrored by the early-out).
    {
        server_sim srv;
        srv.slots = { make_task(1, 0) };
        CHECK(admit_task(srv, 1, N_CTX + 1, N_CTX + 1, /*kv_unified*/ false, true) == ADMIT);
    }

    if (n_fail == 0) {
        fprintf(stdout, "ALL PASS (test_server_exposures_host)\n");
        return 0;
    }
    fprintf(stdout, "%d FAILURES (test_server_exposures_host)\n", n_fail);
    return 1;
}
