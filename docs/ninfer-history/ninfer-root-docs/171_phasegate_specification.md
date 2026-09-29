# DOC 171 — Full-Pipeline Phase-Gate Specification for Speculative Decoding

**Status:** ACTIVE · **Author:** Gemini (test lane) · **Coordinator Tag:** C441 · **Target:** Twin speculative decode (`run_tp2_request_dflash2`)

---

## 1. Principle & Objectives

Per user directive ("create tests to PHASE-GATE our approach, something is failing somewhere"), we replace sequential single-hypothesis probes with a full-pipeline cross-rank phase gate.

The KVarN Phase-Gate Rule: **The first failing phase is the point of failure.**

Speculative decoding on TP2 executes lock-step iterations across Rank 0 and Rank 1. When a hang or desynchronization occurs (e.g. at step ~13), comparing cross-rank stage boundaries across the execution order immediately isolates whether divergence originated in:
1. Prefill context
2. Fusion feed / draft proposal
3. Draft token identities
4. Target verification forward / argmax
5. Accept kernel inputs or outputs
6. KV / linear-state commit
7. Tail termination decision (`hit_stop`, `gen_before`, `done`)
8. Next-round context window feeding the chain

---

## 2. Stage Boundaries (P0 .. P7)

Every stage boundary produces a single stdout/stderr line per rank per step:

| Phase | Name | Scope / Payload | Hash Source |
|---|---|---|---|
| **P0** | `P0_PREFILL` | Prefill taps / prompt context | Prefix activations / prompt context slice |
| **P1** | `P1_SEED_FUSE` | Seed-fuse buffer (`d2h_fused`) | Fused features prior to initial proposal |
| **P2** | `P2_CHAIN_DRAFTS` | Draft token IDs | Array of `k` draft token IDs (`chain_drafts`) |
| **P3** | `P3_VERIFY_LOGITS` | Target verify logits & global argmax | `d2_tgt` device tokens (length `T * sizeof(int)`) |
| **P4** | `P4_ACCEPT_IN` | Inputs to accept kernel | Transposed targets (`mb_tgt_rm`), drafts (`mb_drafts`), extents (`mb_ext`) |
| **P4** | `P4_ACCEPT_OUT` | Outputs of accept kernel | `a = acc_h[0]`, `lic = lic_h[0..a]` |
| **P5** | `P5_COMMIT` | KV & Linear-state commit | GDN recurrent/conv slot-0 state hash |
| **P6** | `P6_TAIL_DECISION` | Loop termination flags | `hit_stop`, `gen_before`, `done`, `out_count` |
| **P7** | `P7_NEXT_CONTEXT` | Next-round context window | Drafter pool slice at read time (`c_lo`, `c_hi`) |

---

## 3. Printf Specification

Lines MUST conform to the machine-parseable format:
```text
PG step=<STEP> phase=<PHASE> rank=<RANK> h=<16_HEX_CHARS> extra=<KEY_VALUE_PAIRS>
```

### Grammar
- `PG`: literal header token.
- `step=<STEP>`: 1-indexed integer decode round / step.
- `phase=<PHASE>`: one of `P0_PREFILL`, `P1_SEED_FUSE`, `P2_CHAIN_DRAFTS`, `P3_VERIFY_LOGITS`, `P4_ACCEPT_IN`, `P4_ACCEPT_OUT`, `P5_COMMIT`, `P6_TAIL_DECISION`, `P7_NEXT_CONTEXT`.
- `rank=<RANK>`: integer `0` or `1`.
- `h=<HEX64>`: 16-character lowercase hexadecimal 64-bit FNV-1a hash over raw byte payload.
- `extra=<EXTRA>`: human-readable and machine-parseable metadata for immediate diagnosis without re-running (e.g. `a=2 lic=[248046,248046,999]` or `hit_stop=0 done=0`).

### Examples
```text
PG step=12 phase=P2_CHAIN_DRAFTS rank=0 h=5a3b9f1012ca34ef extra=drafts=[248046,248046,198,8839,220]
PG step=12 phase=P2_CHAIN_DRAFTS rank=1 h=5a3b9f1012ca34ef extra=drafts=[248046,248046,198,8839,220]
PG step=12 phase=P4_ACCEPT_OUT rank=0 h=81f0a3e8bc491200 extra=a=2 lic=[248046,248046,999]
PG step=12 phase=P4_ACCEPT_OUT rank=1 h=81f0a3e8bc491200 extra=a=2 lic=[248046,248046,999]
PG step=12 phase=P6_TAIL_DECISION rank=0 h=0000000000000000 extra=hit_stop=0 gen=29 done=0
PG step=12 phase=P6_TAIL_DECISION rank=1 h=0000000000000000 extra=hit_stop=0 gen=29 done=0
```

---

## 4. C++ Implementation Helper

House standard 64-bit FNV-1a hash implementation:

```cpp
// FNV-1a 64-bit hash over raw memory bytes
inline std::uint64_t fnv64(const void* data, std::size_t bytes, std::uint64_t h = 0xcbf29ce484222325ULL) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Environment-gated phase gate emission
#define EMIT_PHASEGATE(step, phase, rank, data_ptr, bytes, extra_fmt, ...) \
    do { \
        static const bool pg_enabled = (getenv("NINFER_D2_PHASEGATE") != nullptr); \
        if (pg_enabled) { \
            std::uint64_t h = fnv64((data_ptr), (bytes)); \
            char extra_buf[256]; \
            std::snprintf(extra_buf, sizeof(extra_buf), extra_fmt, ##__VA_ARGS__); \
            std::fprintf(stderr, "PG step=%d phase=%s rank=%d h=%016llx extra=%s\n", \
                         (step), (phase), (rank), static_cast<unsigned long long>(h), extra_buf); \
            std::fflush(stderr); \
        } \
    } while (0)
```

---

## 5. Automated Reconciliation Tool (`phasegate_report.py`)

The companion tool [`tools/ops/phasegate_report.py`](file:///home/intel/ninfer/repo/tools/ops/phasegate_report.py) parses any server log, matches `(step, phase)` pairs between Rank 0 and Rank 1, and reports:

1. **Exact match count**: number of verified byte-identical phase boundaries.
2. **First Divergent Phase**: if `h0 != h1`, outputs the earliest divergence in execution order.
3. **Reachability Asymmetry**: if Rank 1 vanishes (e.g. silently breaks from the loop), flags the exact step where Rank 1 was missing and prints the previous step's final phase decisions.

### Usage
```bash
# Analyze serve log
tools/ops/phasegate_report.py serve.log

# Show only diverging or missing stages
tools/ops/phasegate_report.py serve.log --diff-only

# Machine-readable JSON output for CI
tools/ops/phasegate_report.py serve.log --json
```

---

## 6. N-in-1 Repeats Harness Oracle Contract (`p1n1`)

A2's `p1n1.sh` executes repeated sequential requests of `m2b_10k_a` against the same server. The oracle evaluates:

1. **Per-request status**: HTTP 200 return code, client text finish reason `stop`.
2. **Wall time bound**: `< 3.0s` per 30-token request (`> 10.0 tok/s`).
3. **Stats row integrity**: `[D2-SS-STATS]` present, `rounds == steps` (verifies double-count fix `dc92bddf`), acceptance rate `acc_rate` within `0.20 <= acc <= 0.50`.
4. **Zero desynchronization / zero hang**: no server hang at step 13; if a hang occurs, `phasegate_report.py` is invoked to dump the decisive verdict before teardown.
