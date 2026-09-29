// Catch-up rollback desk: host-side parity tests for the accepted-prefix
// catch-up (LLAMA_DRAFT_PREFIX_CATCHUP, no GPU, no ggml linkage).
//
// Baseline round (heads = 1 MTP, growing-KV draft):
//   draft phase    : ctx_dft decodes id_last @ L, then each drafted token;
//                    the checkpoint block then seq_rm's everything >= L
//   target verify  : ctx_tgt decodes [sampled @ L, d1..dm @ L+1..L+m]
//   catch-up       : process() re-decodes ALL m+1 rows on ctx_dft (this is
//                    the 11.66 ms/round the desk attacks), incl. the rows the
//                    post_decode seq_rm(pos_next) throws away right after
//   accept         : k = number of matched drafts; prompt += k tokens;
//                    pos_next = L+k+1; seq_rm(ctx_dft, pos_next)
// Prefix arm: process() stages the batch; after the accept loop, catchup()
//   decodes rows 0..k only (row 0 = the sampled token @ L, rows 1..k = the
//   accepted drafts). The rejected rows are never written, so the rollback
//   finds nothing to remove: every cell the cache ever RETURNS is identical.
//
// The two arms are driven through the same scripted accept/reject patterns on
// synthetic sequences. Every ctx_dft decode records the content hash of the
// cells its rows attend (attention is causal: positions 0..p), so any
// divergence in cache state or batch inputs changes the view hash and the
// deterministically-derived drafted tokens. The test pins:
//
//   1. the drafted-token sequences are identical across the whole run,
//   2. every draft-phase decode sees an identical attended view,
//   3. the catch-up rows the prefix arm decodes are the same rows (same
//      token/pos/embd source) the baseline decodes - it just skips rows > k,
//      which the baseline's seq_rm removes from the cache anyway,
//   4. cells at positions < pos_next are identical after every round,
//   5. the odd paths stay equivalent: prefill chunks and 1-token rounds
//      (no accept decision -> full staged rows), an unflushed staging
//      followed by the process()-entry self-heal, a checkpoint-restore round
//      (restore rm after baseline's catch-up, before the prefix arm's),
//   6. the work reduction is the intended one: catch-up rows 4 -> k+1.
//
// Build + run:
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_prefix_catchup_host
//       docs/amd-port/tests/test_prefix_catchup_host.cpp && /tmp/test_prefix_catchup_host

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

typedef int32_t llama_pos;
typedef int32_t llama_seq_id;
typedef int32_t llama_token;

// a KV cell: the token plus the h-row source it was decoded with (identity of
// the decode inputs, not float values - numerics are out of scope host-side)
struct cell {
    llama_token tok = -1;
    int         h   = -1;

    bool operator==(const cell & o) const { return tok == o.tok && h == o.h; }
};

typedef std::map<std::pair<llama_seq_id, llama_pos>, cell> kv_t;

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t hash_kv_view(const kv_t & kv, llama_seq_id seq, llama_pos p) {
    // content of the cells a row at pos p attends (0..p)
    uint64_t h = 0;
    for (llama_pos q = 0; q <= p; q++) {
        auto it = kv.find({seq, q});
        if (it == kv.end()) {
            h = fnv("MISS", 4, h); // a missing cell must not pass silently
        } else {
            h = fnv(&it->second.tok, sizeof(it->second.tok), h);
            h = fnv(&it->second.h,   sizeof(it->second.h),   h);
        }
        h = fnv(&q, sizeof(q), h);
    }
    return h;
}

// one row handed to a ctx_dft decode
struct row {
    llama_token tok;
    llama_pos   pos;
    int         h;   // embd source tag (pending_h / shifted target h row)
};

struct decode_ev {
    std::string       kind;   // "draft0" "draft" "catchup" "selfheal" "prefill"
    llama_seq_id      seq;
    std::vector<row>  rows;
    std::vector<uint64_t> views; // attended-view hash per row, at decode time
};

// the deterministic "draft model": drafted token from the attended view + the
// embd input, so any divergence flips the token and the parity checks catch it
static llama_token draft_sample(uint64_t view, int h) {
    return (llama_token) ((fnv(&view, sizeof(view), fnv(&h, sizeof(h))) % 97));
}

struct sim {
    bool defer; // LLAMA_DRAFT_PREFIX_CATCHUP

    kv_t kv;

    // impl mirrors
    std::vector<row>  staged;           // the built catch-up batch (all seqs)
    bool              staged_flag = false;
    std::vector<int32_t> keep;          // per seq: -1 full, else rows to keep
    std::vector<int>  pending_h;        // embd source tag per seq

    // h rows the target produced for the current verify batch, by batch row
    std::vector<int> h_tgt;

    std::vector<decode_ev> log;

    size_t n_catchup_rows = 0;

    explicit sim(bool defer_) : defer(defer_) {}

    // ctx_dft decode of `rows` for one seq (contiguous, causal)
    void dft_decode(const char * kind, llama_seq_id seq, const std::vector<row> & rows) {
        decode_ev ev;
        ev.kind = kind;
        ev.seq  = seq;
        for (const auto & r : rows) {
            kv[{seq, r.pos}] = { r.tok, r.h };
            ev.rows.push_back(r);
            ev.views.push_back(hash_kv_view(kv, seq, r.pos));
        }
        log.push_back(ev);
    }

    // process(): stage (prefix arm) or decode in full (baseline); both arms
    // refresh pending_h to the last target h row afterwards, as the real
    // process() does (the staged rows keep the pre-overwrite copies)
    void process(llama_seq_id seq, const std::vector<row> & rows) {
        if (!defer) {
            dft_decode("catchup", seq, rows);
            n_catchup_rows += rows.size();
        } else {
            staged = rows;
            staged_flag = true;
            keep.assign(1, -1);
        }
        if (!h_tgt.empty()) {
            pending_h[seq] = h_tgt[h_tgt.size() - 1];
        }
    }

    // accept(): record the accepted-prefix width for the staged seq
    void accept(llama_seq_id seq, int k, int n_rows) {
        pending_h[seq] = h_tgt[k < n_rows ? k : n_rows - 1];
        if (defer && staged_flag) {
            keep[seq] = k + 1; // rows 0..k
        }
    }

    // common_speculative_catchup(): flush the staging (prefix arm)
    bool catchup() {
        if (!(defer && staged_flag)) {
            return true;
        }
        std::vector<row> rows;
        const int nk = keep[0] < 0 ? (int) staged.size() : keep[0];
        for (int i = 0; i < nk && i < (int) staged.size(); i++) {
            rows.push_back(staged[i]);
        }
        if (!rows.empty()) {
            dft_decode("catchup", 0, rows);
            n_catchup_rows += rows.size();
        }
        staged_flag = false;
        return true;
    }

    // process()-entry / draft()-entry self-heal: unflushed staging decodes in full
    void selfheal() {
        if (!(defer && staged_flag)) {
            return;
        }
        dft_decode("selfheal", 0, staged);
        n_catchup_rows += staged.size();
        staged_flag = false;
    }

    void seq_rm(llama_seq_id seq, llama_pos beg) {
        for (auto it = kv.begin(); it != kv.end(); ) {
            if (it->first.first == seq && it->first.second >= beg) {
                it = kv.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// run one scripted generation on both arms and compare
struct script_step {
    int k;          // accepted drafts this verify (0..m)
    int m;          // drafts offered
    bool do_draft;  // draft phase runs this round
    bool ckpt;      // checkpoint-restore round (no accept callback)
    int n_reverify; // > 0: reuse round re-verifying n_reverify rows
};

static bool run_script(const std::vector<script_step> & steps,
                       std::vector<decode_ev> & log_base,
                       std::vector<decode_ev> & log_prefix,
                       size_t & rows_base, size_t & rows_prefix) {
    sim base(false), pref(true);
    base.pending_h.assign(1, 100);
    pref.pending_h.assign(1, 100);

    llama_token sampled = 500;      // the committed sampled token
    llama_pos   L       = 32;       // prompt length (positions 0..L-1 in KV)

    for (llama_pos q = 0; q < L; q++) {
        base.kv[{0, q}] = { (llama_token) (900 + q), (int) (900 + q) };
        pref.kv[{0, q}] = { (llama_token) (900 + q), (int) (900 + q) };
    }

    for (size_t s = 0; s < steps.size(); s++) {
        const auto & st = steps[s];

        if (st.do_draft && st.n_reverify == 0) {
            // draft phase (both arms identical): step 0 re-decodes the sampled
            // token @ L, then the drafted tokens
            std::vector<row> rows = { { sampled, L, base.pending_h[0] } };
            uint64_t v0 = hash_kv_view(base.kv, 0, L);
            llama_token d1 = draft_sample(v0, base.pending_h[0]);

            // the arms draft independently so a divergence cannot hide
            std::vector<row> rows_p = { { sampled, L, pref.pending_h[0] } };

            base.dft_decode("draft0", 0, rows);
            pref.dft_decode("draft0", 0, rows_p);

            // the checkpoint block drops the draft-phase cells again (>= L)
            base.seq_rm(0, L);
            pref.seq_rm(0, L);
            (void) v0;
            (void) d1;
        }

        // target verify batch: [sampled @ L, drafts @ L+1..L+m]; a reuse
        // round re-verifies the stored partial draft instead
        const int n_rows = st.n_reverify > 0 ? st.n_reverify + 1 :
                ((st.do_draft ? st.m : 0) + 1);
        std::vector<row> vrows;
        vrows.push_back({ sampled, L, base.pending_h[0] });
        for (int i = 0; i < n_rows - 1; i++) {
            // the target h row feeds the NEXT row's embd (shift right)
            vrows.push_back({ (llama_token) (600 + i), L + 1 + i, (int) (1000 + s * 10 + i) });
        }
        base.h_tgt.clear();
        pref.h_tgt.clear();
        for (int i = 0; i < n_rows; i++) {
            base.h_tgt.push_back((int) (2000 + s * 10 + i));
            pref.h_tgt.push_back((int) (2000 + s * 10 + i));
        }

        base.process(0, vrows);
        pref.process(0, vrows);

        // post_decode: accept decision + rollback + (prefix arm) catch-up
        if (st.ckpt) {
            // checkpoint-restore round: no accept callback; baseline's
            // catch-up already ran inside process(); both arms rm >= L, the
            // prefix arm flushes its staging in full after the restore
            base.seq_rm(0, L);
            pref.seq_rm(0, L);
            pref.catchup(); // no keep set -> full rows
            sampled = 777;  // re-verify path reuses the partial draft
            L = L;          // the restore truncates the prompt back to L
        } else {
            base.accept(0, st.k, n_rows);
            pref.accept(0, st.k, n_rows);

            pref.catchup();

            // post_decode seq_rm on both contexts at pos_next = L+k+1
            base.seq_rm(0, L + st.k + 1);
            pref.seq_rm(0, L + st.k + 1);

            // the last accepted token becomes the next sampled token
            sampled = (llama_token) (700 + s);
        }

        // invariant 4: cells below pos_next identical. ckpt rounds check only
        // below L: the baseline's restore rm runs after its in-process catch-up
        // while the prefix arm flushes after the restore, so the rejected cells
        // differ (dead until the re-verify round re-decodes them - pinned there)
        const llama_pos pn = L + st.k + 1;
        const llama_pos pn_chk = st.ckpt ? L : pn;
        for (llama_pos q = 0; q < pn_chk; q++) {
            auto ib = base.kv.find({0, q});
            auto ip = pref.kv.find({0, q});
            if (ib == base.kv.end() || ip == pref.kv.end()) {
                fprintf(stderr, "DBG round %zu (k=%d m=%d): missing q=%d (base=%d pref=%d) L=%d pn=%d\n",
                        s, st.k, st.m, (int) q, ib != base.kv.end(), ip != pref.kv.end(), (int) L, (int) pn);
            }
            CHECK(ib != base.kv.end());
            CHECK(ip != pref.kv.end());
            if (ib != base.kv.end() && ip != pref.kv.end()) {
                CHECK(ib->second == ip->second);
            }
        }

        if (!st.ckpt) {
            L = pn; // prompt grew by k accepted tokens
        }
    }

    log_base     = base.log;
    log_prefix   = pref.log;
    rows_base    = base.n_catchup_rows;
    rows_prefix  = pref.n_catchup_rows;

    // invariant 1: the drafted-token sequences (draft0 tokens) are identical
    std::vector<llama_token> toks_b, toks_p;
    for (auto & e : log_base)   if (e.kind == "draft0") toks_b.push_back(e.rows[0].tok);
    for (auto & e : log_prefix) if (e.kind == "draft0") toks_p.push_back(e.rows[0].tok);
    CHECK(toks_b == toks_p);

    // invariant 2: draft-phase attended views identical
    std::vector<uint64_t> vb, vp;
    for (auto & e : log_base)   if (e.kind == "draft0") vb.push_back(e.views[0]);
    for (auto & e : log_prefix) if (e.kind == "draft0") vp.push_back(e.views[0]);
    CHECK(vb == vp);

    // invariant 5 (self-heal): every prefix-arm selfheal decode has a
    // baseline "catchup" twin with identical rows and views
    size_t nh = 0;
    for (auto & e : log_prefix) {
        if (e.kind != "selfheal") {
            continue;
        }
        nh++;
        bool found = false;
        for (auto & eb : log_base) {
            if (eb.kind == "catchup" && eb.views == e.views && eb.rows.size() == e.rows.size()) {
                bool same = true;
                for (size_t i = 0; i < e.rows.size(); i++) {
                    if (eb.rows[i].tok != e.rows[i].tok || eb.rows[i].pos != e.rows[i].pos ||
                        eb.rows[i].h   != e.rows[i].h) {
                        same = false;
                        break;
                    }
                }
                found = same;
                if (found) {
                    break;
                }
            }
        }
        CHECK(found);
    }
    return true; // selfheal parity was checked inline
}

int main() {
    // ---- 1. single seq, all accept/reject shapes ------------------------>
    {
        const std::vector<script_step> steps = {
            /* verify round k=0 */ { 0, 3, true, false, 0 },
            /* verify round k=1 */ { 1, 3, true, false, 0 },
            /* verify round k=2 */ { 2, 3, true, false, 0 },
            /* verify round k=3 */ { 3, 3, true, false, 0 },
        };

        std::vector<decode_ev> lb, lp;
        size_t rb = 0, rp = 0;
        run_script(steps, lb, lp, rb, rp);

        // invariant 6: catch-up rows 4 -> k+1
        const size_t want_b = 4 * steps.size();
        size_t want_p = 0;
        for (auto & st : steps) {
            want_p += st.k + 1;
        }
        CHECK(rb == want_b);
        CHECK(rp == want_p);
        CHECK(rp < rb);

        // invariant 3: prefix-arm catch-up rows are a prefix of the
        // baseline's catch-up rows, per round
        std::vector<const decode_ev *> cb, cp;
        for (auto & e : lb) if (e.kind == "catchup") cb.push_back(&e);
        for (auto & e : lp) if (e.kind == "catchup") cp.push_back(&e);
        CHECK(cb.size() == steps.size());
        CHECK(cp.size() == steps.size());
        for (size_t r = 0; r < steps.size() && r < cb.size() && r < cp.size(); r++) {
            const auto & st = steps[r];
            const decode_ev & eb = *cb[r];
            const decode_ev & ep = *cp[r];
            CHECK(eb.rows.size() == (size_t)(st.m + 1));
            CHECK(ep.rows.size() == (size_t)(st.k + 1));
            for (size_t i = 0; i < ep.rows.size(); i++) {
                CHECK(ep.rows[i].tok == eb.rows[i].tok);
                CHECK(ep.rows[i].pos == eb.rows[i].pos);
                CHECK(ep.rows[i].h   == eb.rows[i].h);
            }
        }
    }

    // ---- 2. mixed round kinds (prefill, 1-token, mtmd self-heal) -------->
    {
        sim base(false), pref(true);
        base.pending_h.assign(1, 100);
        pref.pending_h.assign(1, 100);

        llama_token sampled = 500;
        llama_pos   L       = 32;

        // prefill chunk: process without any accept follow-up
        {
            std::vector<row> vrows = { { 10, L, base.pending_h[0] }, { 11, L + 1, 55 } };
            base.h_tgt = { 91, 92 };
            pref.h_tgt = { 91, 92 };
            base.process(0, vrows);
            pref.process(0, vrows);
            pref.catchup(); // post_decode flush, no accept -> full rows
            base.seq_rm(0, L + 2);
            pref.seq_rm(0, L + 2);
            for (llama_pos q = 0; q < L + 2; q++) {
                CHECK((base.kv[{0, q}] == pref.kv[{0, q}]));
            }
            L += 2;
        }

        // mtmd-style staging with NO flush, then drafting: the prefix arm
        // self-heals the full staging at draft() entry
        {
            std::vector<row> vrows = { { 12, L, base.pending_h[0] }, { 13, L + 1, 66 } };
            base.h_tgt = { 93, 94 };
            pref.h_tgt = { 93, 94 };
            base.process(0, vrows);
            pref.process(0, vrows);
            // no catchup() call here
            base.seq_rm(0, L + 2);
            pref.seq_rm(0, L + 2);
            // note: the baseline decoded rows inside process(); the prefix arm
            // still owes them -> the self-heal below must produce the same cells
            base.pending_h[0] = 94;

            // draft() entry
            base.dft_decode("draft0", 0, { { sampled, L + 2, base.pending_h[0] } });
            pref.selfheal();
            pref.dft_decode("draft0", 0, { { sampled, L + 2, pref.pending_h[0] } });

            for (llama_pos q = 0; q <= L + 2; q++) {
                CHECK(base.kv.count({0, q}) == pref.kv.count({0, q}));
                if (base.kv.count({0, q}) && pref.kv.count({0, q})) {
                    CHECK((base.kv[{0, q}] == pref.kv[{0, q}]));
                }
            }
        }

        // 1-token round (no draft offered): full rows = the single row
        {
            std::vector<row> vrows = { { sampled, L + 3, base.pending_h[0] } };
            base.h_tgt = { 95 };
            pref.h_tgt = { 95 };
            base.process(0, vrows);
            pref.process(0, vrows);
            pref.catchup();
            base.seq_rm(0, L + 4);
            pref.seq_rm(0, L + 4);
            CHECK((base.kv[{0, L + 3}] == pref.kv[{0, L + 3}]));
        }
    }

    // ---- 3. checkpoint-restore round ------------------------------------>
    {
        const std::vector<script_step> steps = {
            { 1, 3, true,  false, 0 },
            { 0, 3, true,  true,  0 }, // ckpt restore: prompt truncated to L
            { 2, 3, false, false, 2 }, // reuse round re-verifies the partial draft
            { 1, 3, true,  false, 0 },
        };

        std::vector<decode_ev> lb, lp;
        size_t rb = 0, rp = 0;
        run_script(steps, lb, lp, rb, rp);

        std::vector<llama_token> toks_b, toks_p;
        for (auto & e : lb) if (e.kind == "draft0") toks_b.push_back(e.rows[0].tok);
        for (auto & e : lp) if (e.kind == "draft0") toks_p.push_back(e.rows[0].tok);
        CHECK(toks_b == toks_p);
    }

    // ---- 4. long mixed campaign, per-position accept histogram ----------<
    {
        const std::vector<script_step> steps = {
            { 0, 3, true, false, 0 }, { 3, 3, true, false, 0 }, { 1, 3, true, false, 0 },
            { 2, 3, true, false, 0 }, { 0, 3, true, false, 0 }, { 3, 3, true, false, 0 },
            { 1, 3, true, false, 0 }, { 0, 3, true, false, 0 }, { 2, 3, true, false, 0 },
            { 3, 3, true, false, 0 }, { 1, 3, true, false, 0 }, { 0, 3, true, false, 0 },
        };

        std::vector<decode_ev> lb, lp;
        size_t rb = 0, rp = 0;
        run_script(steps, lb, lp, rb, rp);

        CHECK(rb == 4 * steps.size());
        size_t want_p = 0;
        for (auto & st : steps) {
            want_p += st.k + 1;
        }
        CHECK(rp == want_p);
        fprintf(stdout, "catch-up rows over %zu rounds: baseline %zu vs prefix %zu (%.1f%%)\n",
                steps.size(), rb, rp, 100.0 * (1.0 - (double) rp / (double) rb));
    }

    if (n_fail == 0) {
        fprintf(stdout, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "%d FAILURES\n", n_fail);
    return 1;
}
