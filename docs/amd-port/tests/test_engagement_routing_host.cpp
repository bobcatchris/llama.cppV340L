// test_engagement_routing_host.cpp - the E-126 dc-window engagement defect,
// replicated and closed in CI (zero GPU; links the built tree libs).
//
// What happened: the dc1 arm's server log had NO "draft shape cache enabled"
// line, so the paired window could not distinguish "engaged with no effect"
// from "inert env" (E-117 law: engagement REQUIRED before any verdict). The
// desk's P0 found TWO stacked root causes:
//   1. STALE SERVED BINARY: the window launched build-hip/bin/llama-server
//      built BEFORE the draft-cache merge - the gate code was not in the
//      served process at all (strings scan: zero occurrences of the
//      engagement text in every lib of the served bin dir).
//   2. LOG-VISIBILITY: even on a fresh build the line cannot reach the served
//      log - library INFO is dropped by the server's own callback at the
//      served verbosity: LLAMA_LOG_INFO -> llama_log_internal (src/
//      llama-impl.cpp) -> common_log_default_callback (installed at
//      common/common.cpp:373) maps ggml INFO to LOG_LEVEL_TRACE = 4
//      (common/log.cpp common_get_verbosity) and the server runs at thold 3.
//
// This suite drives the REAL chain end to end, at served conditions:
//   - freshness: the built libllama under test must contain the engagement
//     text (a stale build-hip fails here - the E-126 root cause class);
//   - wiring: llama_log_set(common_log_default_callback, ...) observable
//     via llama_log_get (the exact server wiring);
//   - filter: at thold 3 the WARN engagement line ARRIVES through
//     llama_log_internal and an INFO probe does NOT (direction proof);
//     at thold 4 both arrive (thold semantics sanity).
//
// Build (see run_premerge_ci.sh section 2b):
//   hipcc -O2 -std=c++17 -x c++ -I include -I ggml/include -I src -I common \
//     test_engagement_routing_host.cpp -o /tmp/test_engagement_routing_host \
//     -pthread -L build-hip/bin -lllama -lllama-common -Wl,-rpath,<abs bin>
// Run (CWD = tree root, or pass the bin dir as argv[1]):
//   /tmp/test_engagement_routing_host [build-hip/bin]

#include "llama.h"
#include "ggml.h"
#include "log.h"
#include "llama-impl.h"

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

static std::string read_file(const char * path, bool * ok) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        if (ok) { *ok = false; }
        return "";
    }
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        s.append(buf, n);
    }
    fclose(f);
    if (ok) { *ok = true; }
    return s;
}

static bool binary_contains(const char * path, const char * needle) {
    bool ok = false;
    const std::string s = read_file(path, &ok);
    return ok && s.find(needle) != std::string::npos;
}

static std::string capture_path;

static void emit_probe(ggml_log_level level, const char * tag) {
    llama_log_internal(level, "DCENG-PROBE %s level=%d\n", tag, (int) level);
}

static int count_in_capture(const char * needle) {
    bool ok = false;
    const std::string s = read_file(capture_path.c_str(), &ok);
    if (!ok) {
        return -1;
    }
    int count = 0;
    for (size_t pos = s.find(needle); pos != std::string::npos; pos = s.find(needle, pos + 1)) {
        count++;
    }
    return count;
}

int main(int argc, char ** argv) {
    const char * bindir = argc > 1 ? argv[1] : "build-hip/bin";

    // 0. freshness: the served-stack lib under test must carry the gate code
    //    (the E-126 window served a pre-merge binary - env delivered, code
    //    absent, arm inert BY CONSTRUCTION)
    {
        const std::string lib = std::string(bindir) + "/libllama.so";
        CHECK(binary_contains(lib.c_str(), "draft shape cache enabled"));
        printf("FRESHNESS: %s carries the engagement gate text\n", lib.c_str());
    }

    // 1. wiring: the server installs common_log_default_callback as the llama
    //    log sink (common/common.cpp:373); llama_log_get must report it back
    {
        llama_log_set(common_log_default_callback, nullptr);
        ggml_log_callback cb = nullptr;
        void * ud = nullptr;
        llama_log_get(&cb, &ud);
        CHECK(cb == common_log_default_callback);
        printf("WIRING: llama_log_get reports common_log_default_callback installed\n");
    }

    // capture through the common_log file sink (same rendering path as the
    // server's redirected stderr stream)
    capture_path = "/tmp/dc_eng_routing_probe.log";
    common_log_set_file(common_log_main(), capture_path.c_str());

    // 2. served conditions: verbosity 3 (the dc1 log line 1:
    //    "verbosity = 3"; the of-record launch passes no -lv)
    {
        common_log_set_verbosity_thold(3);
        emit_probe(GGML_LOG_LEVEL_INFO, "INFO-MUST-NOT-APPEAR"); // the OLD level: dropped at thold 3
        llama_log_internal(GGML_LOG_LEVEL_WARN, "draft shape cache enabled (%d slots)\n", 2);
        common_log_flush(common_log_main());
        CHECK(count_in_capture("DCENG-PROBE INFO-MUST-NOT-APPEAR") == 0);
        CHECK(count_in_capture("draft shape cache enabled (2 slots)") == 1);
        printf("FILTER thold=3: engagement WARN line arrives, INFO probe dropped (the E-126 visibility defect)\n");
    }

    // 3. thold semantics sanity: at 4 (trace) the INFO probe arrives too
    {
        common_log_set_verbosity_thold(4);
        emit_probe(GGML_LOG_LEVEL_INFO, "INFO-AT-THOLD4");
        common_log_flush(common_log_main());
        CHECK(count_in_capture("DCENG-PROBE INFO-AT-THOLD4") == 1);
        printf("FILTER thold=4: INFO probe arrives (filter direction confirmed)\n");
    }

    common_log_set_file(common_log_main(), nullptr);
    remove(capture_path.c_str());

    if (n_fail == 0) {
        printf("ALL PASS: engagement routing suite\n");
    } else {
        printf("FAILURES: %d\n", n_fail);
    }
    return n_fail != 0;
}
