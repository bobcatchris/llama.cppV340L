// W6 full draft isolation: host-side logic tests (no GPU, no ggml linkage).
//
// Mirrors the pure host logic added on top of --spec-mtp-device:
//   1. The draft-visible shared-tensor rule of llama-model-loader.cpp
//      (create_tensor): a whole copy is created on the MTP device iff the
//      tensor is TOKEN_EMBD or OUTPUT; copies dedup by name (tied embeddings
//      load once) and their bytes go to the loader progress accounting
//      (size_data), not to n_created.
//   2. The tensors_by_name exclusion of llama-model.cpp: the MTP copies must
//      not be visible to get_tensor() (meta split-state queries answer by
//      name and must resolve the originals).
//   3. The backend-list leading rule of the llama_context ctor: the extra
//      device is inserted at index 0; skipped when already among the model
//      devices.
//   4. The scheduler placement rule of ggml-backend.cpp
//      (sched_backend_from_buffer): a pre-allocated weight resolves to the
//      first backend that supports its buffer type; graph inputs enter at the
//      last backend (CPU). With every draft-visible weight in the MTP buft
//      and the list [MTP, meta, CPU], every op resolves to the MTP backend;
//      a weight still on the meta split (output_norm fallback case) resolves
//      to meta - correct but not isolated (the audit covers it).
//   5. The isolation audit bucketing of sched_reserve: buffer sizes are summed
//      per extra device vs the rest (CPU excluded); LLAMA_SPEC_MTP_STRICT
//      turns a nonzero rest into a boot failure.
//
// Build + run:
//   g++ -std=c++17 -Wall -Wextra -o /tmp/test_w6_isolation_host
//       docs/amd-port/tests/test_w6_isolation_host.cpp && /tmp/test_w6_isolation_host

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// ---- mirror 1: create_tensor dup rule ---------------------------------->

enum llm_tensor { T_OTHER, T_TOKEN_EMBD, T_OUTPUT };

// mirror: llama-model-loader.cpp create_tensor full-isolation block
static bool creates_mtp_copy(llm_tensor t, bool dev_mtp_set) {
    return dev_mtp_set && (t == T_TOKEN_EMBD || t == T_OUTPUT);
}

// name-keyed MTP context (ggml_get_tensor dedup), plus the loader counters the
// dup path must (not) touch
struct mtp_mirror {
    std::vector<std::string> names;
    size_t size_data = 0;
    size_t n_created = 0;

    void create(const char * name, size_t nbytes) {
        for (const auto & n : names) {
            if (n == name) {
                return; // dedup: whole copy already in the MTP ctx
            }
        }
        names.emplace_back(name);
        // dup bytes are loaded in addition to the original: progress
        // accounting only, never n_created (done_getting_tensors guard)
        size_data += nbytes;
    }
};

// ---- mirror 3: backend list leading ------------------------------------->

static std::vector<const void *> build_backend_list(const void * extra, const std::vector<const void *> & model_devices) {
    std::vector<const void *> backends(model_devices);
    // ctor rule: skip when the extra device is already among the model devices
    for (auto * d : model_devices) {
        if (d == extra) {
            return backends; // dup: ignore
        }
    }
    backends.insert(backends.begin(), extra); // LEAD
    return backends;
}

// ---- mirror 4: scheduler placement -------------------------------------->

// backend list [MTP, meta, CPU]; bufts: 0 = MTP device buffer, 1 = meta buffer
// supports[buft][backend]: each GPU backend supports only its own buffer type,
// the CPU backend supports neither (weights on CPU are not modeled here)
static const bool supports[2][3] = {
    /* buft 0 (MTP) */ {true, false, false},
    /* buft 1 (meta) */ {false, true, false},
};

// mirror: ggml_backend_sched_backend_id_from_cur -> backend_from_buffer for an
// op over a pre-allocated weight; is_input models GGML_TENSOR_FLAG_INPUT
static int sched_backend_for(int weight_buft, bool is_input) {
    if (is_input) {
        return 2; // last backend (assumed CPU)
    }
    for (int i = 0; i < 3; ++i) {
        if (supports[weight_buft][i]) {
            return i; // first backend supporting the weight's buffer type
        }
    }
    return -1;
}

// ---- mirror 5: audit bucketing ------------------------------------------>

struct audit_result {
    size_t size_extra;
    size_t size_rest;
    bool strict_trip;
};

// mirror: sched_reserve isolation audit; devs[2] is the CPU backend (bucketed
// out like the real check skips CPU devices)
static audit_result isolation_audit(const void * dev_extra, const std::vector<const void *> & backend_devs,
        const std::vector<size_t> & buf_sizes, bool strict) {
    audit_result r{0, 0, false};
    for (size_t i = 0; i < backend_devs.size(); ++i) {
        if (backend_devs[i] == dev_extra) {
            r.size_extra += buf_sizes[i];
        } else if (backend_devs[i] != (const void *) 0xC) {
            r.size_rest += buf_sizes[i];
        }
    }
    if (r.size_rest > 0 && strict) {
        r.strict_trip = true;
    }
    return r;
}

// ---- mirror 6: draft-cache independence (defect E-038 route-back) ------->

// mirror: llama-model.cpp create_memory - which context/memory combinations
// pass mem_other into the cache (the ONLY cell-aliasing path). qwen35 MTP is
// NOT one of them: the draft cache is a separate cells object.
enum cache_case { T_QWEN35_MTP, T_QWEN35_TARGET, T_GEMMA4_ASSISTANT };

static bool shares_cells_with_target(cache_case c) {
    switch (c) {
        case T_GEMMA4_ASSISTANT: return true;  // llama_kv_cache_iswa(mem_other)
        case T_QWEN35_MTP:       return false; // plain llama_kv_cache, no mem_other
        case T_QWEN35_TARGET:    return false; // llama_memory_hybrid, no mem_other
    }
    return false;
}

// ---- mirror 7: unified-cache capacity (defect E-038 incident) ----------->

// mirror of the TP2@10k incident arithmetic: kv_unified=true gives ALL slots
// one shared 10240-cell array; two concurrent 7857-token prompts overcommit it
// regardless of build. The server's n_ctx admission check is per-slot.
struct unified_cache {
    uint32_t size;
    uint32_t used = 0;

    bool can_place(uint32_t n) const { return size - used >= n; }
    void place(uint32_t n) { used += n; }
};

int main() {
    // ---- 1: dup rule ----
    CHECK(creates_mtp_copy(T_TOKEN_EMBD, true));
    CHECK(creates_mtp_copy(T_OUTPUT, true));
    CHECK(!creates_mtp_copy(T_OTHER, true));       // e.g. OUTPUT_NORM stays single-copy
    CHECK(!creates_mtp_copy(T_TOKEN_EMBD, false)); // flag unset: nothing duplicated

    // loader accounting, untied model (Qwen3.8-27B-ASCII-P1M): two whole copies
    {
        mtp_mirror m;
        const size_t n_tensors_gguf = 866; // done_getting_tensors guard baseline
        m.n_created = n_tensors_gguf;
        m.create("token_embd.weight", 351619840); // IQ4_XS, GGUF offset delta
        m.create("output.weight", 542942400);     // Q6_K,   GGUF offset delta
        CHECK(m.names.size() == 2);
        CHECK(m.size_data == 351619840ull + 542942400ull);
        CHECK(m.n_created == n_tensors_gguf); // dups must not trip n_created
    }

    // tied model: output dedups onto the token_embd copy by name
    {
        mtp_mirror m;
        m.create("token_embd.weight", 351619840);
        m.create("token_embd.weight", 351619840); // the TENSOR_DUPLICATED output
        CHECK(m.names.size() == 1);
        CHECK(m.size_data == 351619840ull);       // counted once
    }

    // exact sizes of record: token_embd.weight IQ4_XS 351619840 B = 335.33 MiB,
    // output.weight Q6_K 542942400 B = 517.79 MiB; combined 853.12 MiB fits the
    // ~0.9 GiB budget (all values from GGUF offset deltas, never formulas)
    CHECK(351619840ull / 1024 / 1024 == 335);
    CHECK(542942400ull / 1024 / 1024 == 517);
    CHECK((351619840ull + 542942400ull) / 1024 / 1024 == 853);
    CHECK((351619840ull + 542942400ull) < 900ull * 1024 * 1024);

    // ---- 2: tensors_by_name exclusion ----
    {
        const void * meta_buft = (const void *) 0x1;
        const void * mtp_buft  = (const void *) 0x2;
        struct entry { const void * buft; const char * name; };
        const std::vector<entry> all = {
            {meta_buft, "output.weight"}, {meta_buft, "token_embd.weight"},
            {mtp_buft,  "output.weight"}, {mtp_buft,  "token_embd.weight"},
        };
        std::vector<const char *> by_name;
        for (const auto & e : all) {
            if (e.buft == mtp_buft) {
                continue; // mirror: skip buft_mtp
            }
            by_name.push_back(e.name);
        }
        CHECK(by_name.size() == 2); // originals only; get_tensor() stays honest
    }

    // ---- 3: backend list leading ----
    {
        const void * meta = (const void *) 0x20;
        const void * d3   = (const void *) 0x13;
        const auto lead = build_backend_list(d3, {meta});
        CHECK(lead.size() == 2);
        CHECK(lead[0] == d3);   // extra device LEADS
        CHECK(lead[1] == meta);
        const auto dup = build_backend_list(meta, {meta}); // already a target device
        CHECK(dup.size() == 1 && dup[0] == meta);          // ignored, no double entry
        const auto solo = build_backend_list(d3, {});      // CPU-only model edge
        CHECK(solo.size() == 1 && solo[0] == d3);
    }

    // ---- 4: scheduler placement ----
    // every weight of the isolated draft graph sits in the MTP buft (0):
    // blk.64 block weights, the duplicated output.weight / token_embd.weight
    // and the draft KV buffers -> every op resolves to the MTP backend
    for (int i = 0; i < 4; ++i) {
        CHECK(sched_backend_for(0, false) == 0);
    }
    // inputs enter at the CPU backend and are copied into the MTP split
    CHECK(sched_backend_for(-1, true) == 2);
    // fallback case: a weight still on the meta split (output_norm fallback
    // when the GGUF lacks blk.64.nextn.shared_head_norm) resolves to meta -
    // correct, not isolated; the audit WARN catches it
    CHECK(sched_backend_for(1, false) == 1);

    // ---- 5: audit bucketing ----
    {
        const void * meta = (const void *) 0x20;
        const void * d3   = (const void *) 0x13;
        const void * cpu  = (const void *) 0xC;
        const std::vector<const void *> devs = {d3, meta, cpu};
        const size_t mib = 1024 * 1024;

        // full isolation: everything on the extra device, meta silent
        const std::vector<size_t> clean_bufs = {1643 * mib, 0, 0};
        const auto clean = isolation_audit(d3, devs, clean_bufs, true);
        CHECK(clean.size_extra == 1643 * mib);
        CHECK(clean.size_rest == 0);
        CHECK(!clean.strict_trip);

        // spillover (e.g. output_norm fallback): rest nonzero
        const std::vector<size_t> spill_bufs = {1643 * mib, 5 * mib, 0};
        const auto spill = isolation_audit(d3, devs, spill_bufs, true);
        CHECK(spill.size_extra == 1643 * mib);
        CHECK(spill.size_rest == 5 * mib);
        CHECK(spill.strict_trip); // LLAMA_SPEC_MTP_STRICT=1 fails the boot

        const auto warn_only = isolation_audit(d3, devs, spill_bufs, false);
        CHECK(warn_only.size_rest == 5 * mib);
        CHECK(!warn_only.strict_trip); // default: WARN only
    }

    // ---- 6: draft-cache independence (E-038 route-back) ----
    // the TP2@10k "Context size has been exceeded. off = 69" incident suspected
    // draft-cache cell aliasing into the target's unified cache; the draft
    // cache is independent by construction and the incident was unified-cache
    // capacity overcommit from concurrent foreign requests
    CHECK(!shares_cells_with_target(T_QWEN35_MTP));
    CHECK(!shares_cells_with_target(T_QWEN35_TARGET));
    CHECK(shares_cells_with_target(T_GEMMA4_ASSISTANT)); // the ONLY sharing path

    // incident arithmetic of record: 10240-cell unified cache, two concurrent
    // 7857-token prompts (battery request + foreign client on the shared port)
    {
        unified_cache kv{10240};
        kv.place(6075);                 // request A, starved mid-prompt
        CHECK(kv.can_place(3584));      // request B chunks 1..7 placed
        kv.place(3584);
        CHECK(kv.used == 9659);
        // request B's next 512-token chunk fails because request A's in-flight
        // [6075, 6587) chunk holds the last 512 cells: 69 free remain
        kv.place(512);                  // request A in-flight chunk
        CHECK(kv.used == 10171);
        CHECK(!kv.can_place(512));
        // request B trickles 64 + 4 + 1 tokens (batch offsets 0 -> 64 -> 68
        // -> 69, then "Context size has been exceeded. off = 69" at nb=1)
        kv.place(64);
        kv.place(4);
        kv.place(1);
        CHECK(!kv.can_place(1));
        CHECK(kv.used == 6587u + 3653u); // exactly 10240 - zero free
        // demand vs supply: two 7857-token prompts need 15714 cells
        CHECK(2u * 7857u > 10240u);
    }

    if (n_fail == 0) {
        fprintf(stdout, "ALL PASS (test_w6_isolation_host)\n");
        return 0;
    }
    fprintf(stdout, "%d FAILURES (test_w6_isolation_host)\n", n_fail);
    return 1;
}
