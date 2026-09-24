// Draft shape-cache desk: replicate-the-defect suite for the single graph
// reuse slot (no GPU, no ggml linkage).
//
// The W12 host-slice capture named the mechanism: the draft context decodes
// alternate shapes 4 (catchup) -> 1,1,1 (steps) every speculative round while
// llama_context keeps ONE graph reuse slot (gf_res_prev,
// src/llama-context.cpp process_ubatch), so 2 of the 4 draft decodes per round
// miss and re-split from scratch (issue 2.593 ms vs 1.108 ms reused). The P0
// correction: the paid cost is NOT the llama-context-level walk - it is the
// ggml uid chain under the re-split: ggml_backend_sched_reset wipes the
// tensor->backend assignments (ggml-backend.cpp ggml_backend_sched_reset),
// every re-split stamps fresh graph uids (ggml-backend.cpp split_graph /
// split-view ids) and the meta backend rebuilds per-die subgraphs + stamps
// fresh per-die graph uids on any uid change (ggml-backend-meta.cpp
// graph_compute needs_rebuild / per-die uid assignment), which defeats the
// HIP device-graph replay keyed by uid (ggml-cuda.cu
// ggml_cuda_graph_update_required) and forces a recapture.
//
// This suite mirrors those exact host decision functions and proves:
//   1. DEFECT: the single-slot policy produces reused=0 on the catchup and on
//      the first step of EVERY round (2 misses/round), with a graph rebuild
//      and fresh per-die uids (HIP recapture class) on every miss.
//   2. FIX: the shape-keyed policy (LLAMA_DRAFT_SHAPE_CACHE) with the same
//      reuse predicate hits on re-entry from the second round on (0
//      steady-state misses), each shape is built exactly once, and the per-die
//      uids stay stable (HIP replay class).
//   3. Identical schedule decisions: the params signature the split would see
//      per shape is identical between the fresh build and the cached re-entry
//      (same predicate, same fields), and the single-split graph uid is stable
//      across re-splits of the same graph (ggml-backend.cpp split_graph).
//   4. Invalidation mirrors: a memory update (context shift / KV defrag /
//      resize path, llama-context.cpp memory_update) and a graph reserve
//      reset ALL cached entries; the next decode rebuilds.
//   5. Predicate safety: a third shape never gets a stale hit; it degrades to
//      rebuilds (today's behavior), never to a wrong reuse.
//
// Build + run:
//   hipcc -O2 -std=c++17 -DGGML_COMMON_DECL_HIP -DGGML_USE_HIP
//       -I ggml/include -I ggml/src docs/amd-port/tests/test_draft_shape_cache_host.cpp
//       -o /tmp/test_draft_shape_cache_host -pthread && /tmp/test_draft_shape_cache_host
//
// Provenance of every mirrored rule is cited inline as file:line at the
// commit this suite was added in.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// ---- mirrored host state --------------------------------------------------

// llm_graph_params::allow_reuse (src/llama-graph.h:706-775), reduced to the
// fields the draft alternation actually exercises; the mirror keeps the same
// field-by-field comparison so a hit decision here implies a hit decision in
// the real predicate for these ubatches.
struct GParams {
    // ubatch shape (llama-graph.h:708-718)
    uint32_t n_tokens     = 0;
    uint32_t n_seq_tokens = 0;
    uint32_t n_seqs       = 0;
    uint32_t n_seqs_unq   = 0;
    bool     equal_seqs   = true;
    bool     has_token    = true;
    // output + graph identity (llama-graph.h:738, 765-774)
    uint32_t n_outputs    = 0;
    int      gtype        = 0;
    uint64_t nextn_offset = 0;
    void *    adapter     = nullptr; // cvec/loras/cross stand-in

    bool allow_reuse(const GParams & o) const {
        const bool ub =
            equal_seqs == o.equal_seqs &&
            n_tokens   == o.n_tokens &&
            n_seq_tokens == o.n_seq_tokens &&
            n_seqs     == o.n_seqs &&
            n_seqs_unq == o.n_seqs_unq &&
            has_token  == o.has_token;
        if (!ub) {
            return false;
        }
        return n_outputs    == o.n_outputs &&
               gtype        == o.gtype &&
               nextn_offset == o.nextn_offset &&
               adapter      == o.adapter;
    }

    // signature of everything the schedule split would be built from
    std::string signature() const {
        char buf[128];
        snprintf(buf, sizeof(buf), "t%u_st%u_sq%u_o%u_g%d", n_tokens, n_seq_tokens, n_seqs, n_outputs, gtype);
        return buf;
    }
};

// llm_graph_result stand-in: a built graph with its params copy
// (src/llama-graph.cpp llm_graph_result::reset/set_params, can_reuse :1396)
struct GfResult {
    GParams  params;       // saved after build (set_params)
    uint64_t uid = 0;      // gf->uid
    int      builds = 0;   // times this graph was (re)built
    bool     alive = true; // false after reset() until rebuilt

    bool can_reuse(const GParams & g) const {
        return alive && params.allow_reuse(g);
    }
    void reset() {
        alive = false;
        params = GParams{};
        uid = 0; // real reset() re-creates the graph: fresh uid (llama-graph.cpp:1305)
    }
};

// ggml_backend_sched stand-in for the parts the reuse path touches
struct Sched {
    bool     is_alloc   = false;
    int      splits     = 0;
    uint64_t last_split_uid = 0; // uid the split views carried
    int      n_reset = 0;
    int      n_alloc = 0;

    void reset() { // ggml_backend_sched_reset: wipes assignments, is_alloc = false
        is_alloc = false;
        n_reset++;
    }
    // ggml_backend_sched_alloc_graph -> split_graph: fresh uid unless the
    // graph already carries one (the A1 change at ggml-backend.cpp:1099), and
    // single-split views carry the graph uid (ggml-backend.cpp:1551)
    void alloc(GfResult & res) {
        if (res.uid == 0) {
            res.uid = next_uid();
        }
        splits = 1; // meta backend owns the whole graph: single split
        last_split_uid = splits == 1 ? res.uid : next_uid();
        is_alloc = true;
        n_alloc++;
    }
    static uint64_t next_uid() {
        static uint64_t uid = 0;
        return ++uid;
    }
};

// ggml-backend-meta stand-in (needs_rebuild + per-die uid memo): on any uid
// change the per-die subgraphs are re-derived and their graphs get fresh uids
// (ggml-backend-meta.cpp:1862, 2126) unless the uid was seen before and the
// memo holds its per-die uids (the A1 memo); HIP replays per-die graphs whose
// uid is unchanged (ggml-cuda.cu:3457)
struct Meta {
    uint64_t cur_uid = 0;                 // backend_ctx->uid
    int      rebuilds = 0;
    std::vector<uint64_t> die_uids;       // current per-die graph uids
    std::vector<uint64_t> memo_uid;       // remembered top-level uids (2 slots)
    std::vector<std::vector<uint64_t>> memo_die;
    std::vector<int> memo_nodes;
    size_t memo_next = 0;
    static constexpr size_t n_seen = 2;

    void compute(const GfResult & res, int n_die) {
        const bool needs_rebuild = (res.uid == 0) || (res.uid != cur_uid);
        if (!needs_rebuild) {
            return; // replay: meta host does not re-derive anything
        }
        rebuilds++;
        cur_uid = res.uid;
        die_uids.assign(n_die, 0);
        int seen = -1;
        for (size_t s = 0; s < n_seen; s++) {
            if (s < memo_uid.size() && memo_uid[s] == res.uid && memo_nodes[s] == 49) {
                seen = (int) s;
                break;
            }
        }
        for (int j = 0; j < n_die; j++) {
            if (seen >= 0) {
                die_uids[j] = memo_die[seen][j];
            } else {
                die_uids[j] = Sched::next_uid();
            }
        }
        if (seen < 0) {
            memo_uid.resize(n_seen, 0);
            memo_nodes.resize(n_seen, 0);
            memo_die.resize(n_seen);
            const size_t s = memo_next;
            memo_uid[s] = res.uid;
            memo_nodes[s] = 49;
            memo_die[s] = die_uids;
            memo_next = (s + 1) % n_seen;
        }
    }
};

// ---- the two process_ubatch policies ---------------------------------------

struct Trace {
    std::vector<int> reused;      // per decode
    int              rebuilds = 0;
    int              resplits = 0; // alloc without build (cached re-entry)
    int              meta_rebuilds = 0;
    bool             die_uids_stable = true;
    uint64_t         split_uid_a = 0; // graph uid seen for shape A re-splits
    std::vector<std::string> sig_per_decode; // params signature at every decode
};

struct UCtx {
    std::vector<GfResult> slots;  // 1 = single slot (default), 2 = shape cache
    size_t   active = 0;
    GfResult * sched_res = nullptr; // entry the sched is allocated with
    Sched    sched;
    Meta     meta;
};

// mirrors llama_context::process_ubatch decision tree (src/llama-context.cpp)
static void process(UCtx & ctx, const GParams & g, Trace & tr) {
    GfResult * res = &ctx.slots[ctx.active];

    bool can_reuse = res->can_reuse(g);

    if (ctx.slots.size() > 1) {
        for (size_t i = 0; i < ctx.slots.size() && !can_reuse; i++) {
            GfResult * cand = &ctx.slots[i];
            if (cand == res) {
                continue;
            }
            if (cand->can_reuse(g)) {
                ctx.active = i;
                res = cand;
                can_reuse = true;
            }
        }
        if (!can_reuse) {
            ctx.active = (ctx.active + 1) % ctx.slots.size(); // LRU victim
            res = &ctx.slots[ctx.active];
        }
    }

    const bool sched_is_res = ctx.slots.size() == 1 || res == ctx.sched_res;

    if (can_reuse && sched_is_res) {
        tr.reused.push_back(1); // pure replay: no split, no meta rebuild
        ctx.meta.compute(*res, 4);
        ctx.sched_res = res;
    } else if (can_reuse) {
        // shape re-entry: keep the graph, re-split the sched for it; the
        // graph uid is stable so the meta memo keeps the per-die uids
        tr.reused.push_back(1);
        tr.resplits++;
        ctx.sched.reset();
        ctx.sched.alloc(*res);
        ctx.meta.compute(*res, 4);
        ctx.sched_res = res;
    } else {
        // miss: rebuild into the slot, re-split from scratch
        tr.reused.push_back(0);
        tr.rebuilds++;
        res->reset();
        ctx.sched.reset();
        res->builds++;
        res->alive = true;
        res->params = g; // set_params after build
        ctx.sched.alloc(*res);
        ctx.meta.compute(*res, 4);
        ctx.sched_res = ctx.slots.size() > 1 ? res : nullptr;
    }
    tr.sig_per_decode.push_back(res->params.signature());
}

static void memory_update(UCtx & ctx) { // context shift / KV defrag / resize
    for (auto & r : ctx.slots) {
        r.reset();
    }
    ctx.sched_res = nullptr;
}

// ---- the draft round the W12 capture measured ------------------------------

static GParams catchup_shape() { // n_tokens = 4
    GParams g;
    g.n_tokens = 4; g.n_seq_tokens = 4; g.n_seqs = 1; g.n_seqs_unq = 1; g.n_outputs = 4; g.gtype = 1;
    return g;
}

static GParams step_shape() { // n_tokens = 1
    GParams g;
    g.n_tokens = 1; g.n_seq_tokens = 1; g.n_seqs = 1; g.n_seqs_unq = 1; g.n_outputs = 1; g.gtype = 1;
    return g;
}

// one speculative round: catchup + 3 draft steps
static void round(UCtx & ctx, Trace & tr) {
    process(ctx, catchup_shape(), tr);
    process(ctx, step_shape(), tr);
    process(ctx, step_shape(), tr);
    process(ctx, step_shape(), tr);
}

int main(int argc, char ** argv) {
    // 1. DEFECT: single slot thrashes across the 4 -> 1,1,1 alternation
    {
        UCtx ctx; ctx.slots.resize(1);
        Trace tr;
        std::vector<uint64_t> die_before;
        bool die_uids_churn = false;
        for (int r = 0; r < 10; r++) {
            round(ctx, tr);
            if (r == 2) {
                die_before = ctx.meta.die_uids;
            }
            if (r == 3) {
                die_uids_churn = ctx.meta.die_uids != die_before; // recapture class
            }
        }
        // steady-state round trace = 0,0,1,1 (catchup and step 1 miss)
        bool steady_ok = true;
        for (int r = 0; r < 10; r++) {
            const size_t o = (size_t) r * 4;
            steady_ok &= tr.reused[o + 0] == 0 && tr.reused[o + 1] == 0 &&
                         tr.reused[o + 2] == 1 && tr.reused[o + 3] == 1;
        }
        CHECK(steady_ok);
        CHECK(tr.rebuilds == 20);       // 2 graph rebuilds x 10 rounds
        CHECK(ctx.meta.rebuilds == 20); // each miss re-derives the meta subgraphs
        CHECK(die_uids_churn);          // fresh per-die uids every round -> HIP recapture
        printf("DEFECT replicated: single-slot steady round reused = 0,0,1,1 "
               "(2 misses/round, %d graph + meta rebuilds in 10 rounds, per-die uids churn)\n",
               tr.rebuilds);
    }

    // 2. FIX: shape cache hits on re-entry, per-die uids stable
    {
        UCtx ctx; ctx.slots.resize(2);
        Trace tr;
        std::vector<uint64_t> die_before;
        for (int r = 0; r < 10; r++) {
            round(ctx, tr);
            if (r == 2) {
                die_before = ctx.meta.die_uids; // after warmup
            }
            if (r > 2) {
                tr.die_uids_stable &= ctx.meta.die_uids == die_before;
            }
        }
        // steady-state round trace = 1,1,1,1 (warmup spent the first round)
        bool steady_ok = true;
        for (int r = 4; r < 10; r++) {
            const size_t o = (size_t) r * 4;
            steady_ok &= tr.reused[o + 0] == 1 && tr.reused[o + 1] == 1 &&
                         tr.reused[o + 2] == 1 && tr.reused[o + 3] == 1;
        }
        CHECK(steady_ok);
        CHECK(tr.rebuilds == 2);  // each shape built exactly once (round 1)
        CHECK(tr.resplits == 18); // 2 cached re-entries x 9 steady rounds
        CHECK(ctx.meta.rebuilds == 20); // re-splits still re-derive (remap), see receipt
        CHECK(tr.die_uids_stable); // HIP replay class: no recapture after warmup
        // both shapes keep distinct stable per-die uids
        CHECK(ctx.meta.die_uids.size() == 4);
        CHECK(ctx.meta.die_uids[0] != ctx.meta.die_uids[1]);
        printf("FIX proven: shape-cache steady round reused = 1,1,1,1 "
               "(0 misses/round, %d total builds, %d cached re-splits, per-die uids stable)\n",
               tr.rebuilds, tr.resplits);
    }

    // 3. identical schedule decisions: the params signature the split sees per
    // shape must match the single-slot fresh-decision signature, and the
    // single-split graph uid must be stable across re-splits of one graph
    {
        UCtx single; single.slots.resize(1);
        UCtx cached; cached.slots.resize(2);
        Trace tr_s, tr_c;
        for (int r = 0; r < 6; r++) {
            round(single, tr_s);
            round(cached, tr_c);
        }
        // the round-2 catchup (index 4) is the first shape-A re-entry: the
        // split decision must see the same signature as its round-1 fresh
        // build (index 0) - identical schedule decisions
        CHECK(tr_c.sig_per_decode.size() == 24);
        CHECK(tr_c.sig_per_decode[4] == catchup_shape().signature());
        CHECK(tr_c.sig_per_decode[4] == tr_c.sig_per_decode[0]);
        CHECK(tr_c.sig_per_decode[4] == "t4_st4_sq1_o4_g1");
        // shape A built exactly once in the cached path, with a stable uid;
        // in the single-slot path it is rebuilt every round (fresh uid each)
        GfResult * a = nullptr;
        for (auto & s : cached.slots) {
            if (s.params.n_tokens == 4) {
                a = &s;
            }
        }
        CHECK(a != nullptr);
        CHECK(a->builds == 1);
        CHECK(a->uid != 0);
        CHECK(single.slots[0].builds == 12); // defect contrast: 2 rebuilds x 6 rounds, one slot
        printf("SCHEDULE identity: cached re-entry signature '%s' matches the fresh build; "
               "shape-A built once, uid stable (%llu); single-slot rebuilt every round (%d builds)\n",
               tr_c.sig_per_decode[4].c_str(), (unsigned long long) a->uid, single.slots[0].builds);
    }

    // 4. invalidation mirrors: memory update (context shift) resets all slots
    {
        UCtx ctx; ctx.slots.resize(2);
        Trace tr;
        for (int r = 0; r < 4; r++) {
            round(ctx, tr);
        }
        const int builds_before = ctx.slots[0].builds + ctx.slots[1].builds;
        memory_update(ctx);
        round(ctx, tr);
        const int builds_after = ctx.slots[0].builds + ctx.slots[1].builds;
        CHECK(builds_after == builds_before + 2); // both shapes rebuilt
        // the round after the shift misses twice, then hits (warmup again)
        const size_t o = tr.reused.size() - 4;
        CHECK(tr.reused[o + 0] == 0 && tr.reused[o + 1] == 0);
        CHECK(tr.reused[o + 2] == 1 && tr.reused[o + 3] == 1);
        printf("INVALIDATION: memory update resets both slots -> both shapes rebuild, then reuse resumes\n");
    }

    // 5. predicate safety: a third shape never gets a stale hit
    {
        UCtx ctx; ctx.slots.resize(2);
        Trace tr;
        for (int r = 0; r < 3; r++) {
            round(ctx, tr);
        }
        GParams odd = catchup_shape();
        odd.n_tokens = 2; // never seen
        const int builds_before = ctx.slots[0].builds + ctx.slots[1].builds;
        process(ctx, odd, tr);
        CHECK(tr.reused.back() == 0); // no wrong reuse
        CHECK(ctx.slots[0].builds + ctx.slots[1].builds == builds_before + 1);
        printf("PREDICATE safety: unseen shape misses (degrades to today's rebuild), no stale hit\n");
    }

    // 6. ENGAGEMENT wiring source pin (E-126: the dc1 arm's server log had no
    //    engagement line - the verdict could not distinguish engaged from
    //    inert, E-117 law). Pins the real call sites: the gate env name, the
    //    canonical text, and the log level - WARN survives the served filter
    //    (common_log_default_callback drops ggml INFO at the served verbosity
    //    3, common/log.cpp common_get_verbosity), INFO does not. The real
    //    routing chain is exercised by test_engagement_routing_host.
    {
        const char * src_path = argc > 1 ? argv[1] : "src/llama-context.cpp";
        FILE * f = fopen(src_path, "rb");
        CHECK(f != nullptr);
        if (f) {
            std::string s;
            char buf[65536];
            size_t n;
            while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
                s.append(buf, n);
            }
            fclose(f);
            const size_t gate = s.find("getenv(\"LLAMA_DRAFT_SHAPE_CACHE\")");
            CHECK(gate != std::string::npos);
            if (gate != std::string::npos) {
                const size_t window_end = gate + 900 < s.size() ? gate + 900 : s.size();
                const std::string window = s.substr(gate, window_end - gate);
                CHECK(window.find("LLAMA_LOG_WARN") != std::string::npos);
                CHECK(window.find("draft shape cache enabled (%d slots)") != std::string::npos);
                CHECK(window.find("LLAMA_LOG_INFO") == std::string::npos);
                // the mechanism witness must be greppable served too (W13 A4 gate 2)
                CHECK(s.find("LLAMA_LOG_WARN(\"[decode-timeline] n_tokens = %d, reused = %d") != std::string::npos);
            }
            printf("ENGAGEMENT source pin: gate env + WARN level + canonical text verified in %s\n", src_path);
        }
    }

    if (n_fail == 0) {
        printf("ALL PASS: draft shape cache host suite\n");
    } else {
        printf("FAILURES: %d\n", n_fail);
    }
    return n_fail != 0;
}
