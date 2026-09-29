# TP4 world=4 DETERMINISTIC TOKEN SOUP — diagnostic dossier for external counsel

**Purpose:** this is a complete, self-contained capture of one bug: code, evidence, fired history, exonerations, ranked hypotheses, and specific questions. The reader is assumed to have NO access to this repo beyond what is quoted here. All line numbers refer to `origin/amd/wo-p3-serve` @ `05012259` unless stated. Verbatim strings are marked.

**Hardware/runtime:** 4× Vega10 (gfx900, MI25x2 dual-die boards, 8 GiB/die), ROCm 6.2, HIPCC compiling CUDA-source (nccl = RCCL 2.20.5; `cuda*`/`__nv_bfloat16` are HIP shims). bf16-only (no fp8/mma on gfx900 — every fast arm is SIMT or link-stubbed). Model: `qwen3_8_27b` (27B, 248320 vocab, hidden 5120, 64 layers, GQA 24q/4kv heads, thinking-mode model).

## 1. The symptom (exact)

Greedy (temp=0) serve at **world=4**: the request completes — `finish_reason=stop`, usage present, `reasoning_len=52` — but the generated text is **deterministic garbage, byte-identical across all 4 ranks AND across repeated identical requests**:

```
<<REASONING:AM)}./r ikhail和睦 ifar和睦ikhailftype只求你ftype和睦sFL和睦>>
```
(fire G18w4q, 6 response legs shown, all identical; also seen at GEN=512 and short probes; the `REASONING:` prefix is the runner's wrapper for reasoning_content, the model is in thinking mode.)

Prior fire (with breadcrumb env `NINFER_BATCH_DBG=1` on, G18w4p) produced the same soup — determinism holds with/without observation (stderr prints in play).

**Contrasts:**
- **world=2 through the SAME binary** is byte-correct all night (a long-standing deterministic attractor `348e77a1222dea7f` over a fixed corpus; the TP2 era served real text).
- **Warmup at world=4** (an internal `run()` of ~51 prompt tokens, max_tokens=4, greedy, through the SAME R1 argmax route) prints per-step decode tokens:
  `[SEQ-STEP1] step=1 next=93 (cpos=53)` / `[SEQ-STEP1] step=2 next=107 (cpos=54)` — these are TOKEN IDs coming out of the R1 reduce. 93 and 107 are very low vocab ids; on the same warmup prompt, world=2's ring route produces the (correct) attractor baseline — **the w=2 warmup token IDs for the same prompt are NOT yet recorded in this dossier; see Q1.**
- Control-flow is PERFECT at w=4: four ranks materialize (3820 MB each == preflight placement), all four `TextContext ready`, collectives complete on the first try (RCCL banners clean, zero warnings), requests terminate, bounded, accounted. **The corruption is in the token VALUES, not the machinery.**

## 2. The route in question (R1), and its contract

world>2 argmax is a NEW code path (`WO-TP4-B/R1`), world=2 uses the old fused ring (untouched, guarded-return dispatch). The route: per-rank shard argmax → `ncclAllGather` of a 12-byte-per-token champion wire → per-rank local reduce over W champions over ALL tokens → draft-vocab remap LAST. The determinism claim (this is the contract to audit):

- champion = `{float val; int tok(global); float sumexp}` — `ArgmaxChampion`, 12 B, moved VERBATIM (no pad; a padded 16B ring struct exists and must never be read at world>2).
- total order = (val desc, GLOBAL tok asc) — all ranks reduce allgathered data identically; equality with full-row-set argmax is by construction, PROVIDED each rank's shard rows really are the global range `[rank*n_rows,(rank+1)*n_rows)` and `n_rows=248320/world`.
- conf = `1/Σ_r sumexp_r·exp(val_r − M_GLOBAL)` ascending-rank combine (single-home `argmax_reduce.h` used by both ring and R1).
- remap LAST: `out = draft_vocab_ids[w.tok]` when a draft table exists (else identity) — remapping before the reduce would compare global ids against shard-local = the named silent class.

### 2a. Verbatim — host shape law (`src/core/multi_gpu/argmax_r1.h`, condensed; full comments retained where load-bearing)

```cpp
static_assert(sizeof(ArgmaxChampion) == 12,
    "R1 wire = ArgmaxChampion {float val; int tok; float sumexp} ...");
inline constexpr std::size_t kR1MaxTokens = 16;
inline constexpr int         kR1MaxWorld  = 8;
inline constexpr std::size_t kR1ConfSlots = 32;

constexpr std::size_t r1_send_bytes(std::size_t T) { return T * sizeof(ArgmaxChampion); }
constexpr std::size_t r1_recv_bytes(std::size_t T, int world) { return r1_send_bytes(T) * world; }

// ncclAllGather lays rank r's block at byte offset r*sendcount — so token t of rank r lives at
// element r1_gather_index(T, r, t). ONE home for the stride, producer and consumer must agree
// by CONSTRUCTION, and the cell pins both directions.
constexpr std::size_t r1_gather_index(std::size_t T, int r, std::size_t t) {
    return static_cast<std::size_t>(r) * T + t;
}

inline void require_r1_shape(int world, std::size_t T, int n_rows) {
    if (world < 2 || world > kR1MaxWorld) throw std::invalid_argument("R1 argmax: world=... outside [2,8] ...");
    if (T == 0 || T > kR1MaxTokens)       throw std::invalid_argument("R1 argmax: T=... outside [1,16] ...");
    if (n_rows <= 0)                      throw std::invalid_argument("R1 argmax: shard rows = ... refusing before any gather");
}

// The per-rank SHARD champion, one home, host/testable (kernel mirrors fold-for-fold):
// winner by (val desc, LOCAL idx asc) — equals (val desc, GLOBAL tok asc) within a shard
// because global = local + rank*n_rows is order-preserving — and the ring's online-lse sumexp
// relative to the shard winner.
inline ArgmaxChampion r1_shard_champion(const float* row, int n_rows, int rank) {
    ArgmaxChampion c{0.0f, 0, 0.0f};
    if (n_rows <= 0) return c;
    float max_v = -1e30f; int max_idx = -1;
    float m = -1e30f, s = 0.0f;
    for (int i = 0; i < n_rows; ++i) {
        const float v = row[i];
        if (v > max_v || (v == max_v && i < max_idx)) { max_v = v; max_idx = i; }
        if (v > m) { s = s * std::exp(m - v) + 1.0f; m = v; }
        else       { s += std::exp(v - m); }
    }
    c.val = max_v;
    c.tok = max_idx + rank * n_rows;     // GLOBAL vocab coords (ring :218 convention)
    c.sumexp = s;
    return c;
}
```

### 2b. Verbatim — device code (`src/core/multi_gpu/r1_argmax.cu`)

```cpp
// kernel 1: shard argmax -> one ArgmaxChampion per token column
__global__ void r1_shard_argmax_kernel(const __nv_bfloat16* __restrict__ logits, int n_rows,
                                       int T, int rank, ArgmaxChampion* __restrict__ out) {
    const int t = blockIdx.x;                       // token column
    if (t >= T) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, num_warps = blockDim.x >> 5;
    const __nv_bfloat16* row = logits + (std::size_t)t * (std::size_t)n_rows;   // *** SHAPE ASSUMPTION: logits is [T, n_rows] row-major ***
    float max_v = -1e30f; int max_idx = -1;
    float lse_m = -1e30f, lse_s = 0.0f;
    for (int i = tid; i < n_rows; i += blockDim.x) {          // 256 threads, strided
        const float v = __bfloat162float(row[i]);
        if (v > max_v || (v == max_v && i < max_idx)) { max_v = v; max_idx = i; }
        if (v > lse_m) { lse_s = lse_s * __expf(lse_m - v) + 1.0f; lse_m = v; }
        else           { lse_s += __expf(v - lse_m); }
    }
    // xor-butterfly (val,tok) with matched (m,s) rescale (rescale uses the PAIRED partner's m/s):
    for (int mask = 16; mask > 0; mask /= 2) {
        const float ov = __shfl_xor_sync(~0u, max_v, mask);
        const int   oi = __shfl_xor_sync(~0u, max_idx, mask);
        const float oms = __shfl_xor_sync(~0u, lse_m, mask);
        const float oss = __shfl_xor_sync(~0u, lse_s, mask);
        const bool take = (ov > max_v || (ov == max_v && oi < max_idx));
        const float hi = take ? ov : max_v, m_hi = take ? oms : lse_m, s_hi = take ? oss : lse_s;
        const float m_lo = take ? lse_m : oms, s_lo = take ? lse_s : oss;
        lse_s = s_hi * __expf(m_hi - hi) + s_lo * __expf(m_lo - hi);
        lse_m = hi;
        if (take) { max_v = ov; max_idx = oi; }
    }
    __shared__ float s_val[32]; __shared__ int s_idx[32]; __shared__ float s_lse[32];
    if (lane == 0) { s_val[warp] = max_v; s_idx[warp] = max_idx; s_lse[warp] = lse_s; }
    __syncthreads();
    if (warp == 0) {                                // winner butterfly over warp leaders only
        max_v = (lane < num_warps) ? s_val[lane] : -1e30f;
        max_idx = (lane < num_warps) ? s_idx[lane] : -1;
        // lse does NOT fold here: each warp's partial stays relative to its own s_val[w];
        // thread 0 combines AFTER the winner is known (ring :220-226 shape).
        for (int mask = 16; mask > 0; mask /= 2) {
            const float ov = __shfl_xor_sync(~0u, max_v, mask);
            const int   oi = __shfl_xor_sync(~0u, max_idx, mask);
            const bool take = (ov > max_v || (ov == max_v && oi < max_idx));
            if (take) { max_v = ov; max_idx = oi; }
        }
    }
    __syncthreads();
    if (tid == 0) {
        float sumexp = 0.0f;
        for (int w = 0; w < num_warps; ++w) sumexp += s_lse[w] * __expf(s_val[w] - max_v);
        ArgmaxChampion c; c.val = max_v;
        c.tok = max_idx + rank * n_rows;            // GLOBAL coords
        c.sumexp = sumexp;
        out[t] = c;                                  // plain device staging, no publish/flag
    }
}

// kernel 3: gather-reduce over W champions, remap LAST, conf to pinned ring
__global__ void r1_gather_reduce_kernel(const ArgmaxChampion* __restrict__ gathered, int world,
                                        int T, const int* __restrict__ draft_vocab_ids,
                                        int* __restrict__ out_token,
                                        float* __restrict__ conf_ring_mapped, std::uint64_t step) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    ArgmaxChampion c[kR1MaxWorld];
    for (int r = 0; r < world; ++r) c[r] = gathered[r1_gather_index(T, r, t)];
    const ArgmaxChampion w = reduce_champions(c, world);        // SINGLE HOME total order
    out_token[t] = draft_vocab_ids ? draft_vocab_ids[w.tok] : w.tok;
    if (t == 0 && conf_ring_mapped)
        conf_ring_mapped[step % kR1ConfSlots] = champion_softmax_conf(c, world);
}

// launcher (runs on the CALLER's per-rank engine stream — per-rank threading law, see §6)
void r1_allreduce_argmax(void* comm, int rank, int world, const void* logits_bf16, int n_rows,
                         std::size_t T, const int* draft_vocab_ids, void* out_token_i32,
                         bool emit_conf, std::uint64_t step, const R1RankBuffers& bufs,
                         void* vstream) {
    require_r1_shape(world, T, n_rows);
    cudaStream_t stream = (cudaStream_t)vstream;
    ArgmaxChampion *send = (ArgmaxChampion*)bufs.send, *recv = (ArgmaxChampion*)bufs.recv;
    constexpr int kThreads = 256;
    r1_shard_argmax_kernel<<<(unsigned)T, kThreads, 0, stream>>>(
        (const __nv_bfloat16*)logits_bf16, n_rows, (int)T, rank, send);
    CUDA_CHECK(cudaGetLastError());
    R1_NCCL(ncclAllGather(send, recv, r1_send_bytes(T), ncclInt8, (ncclComm_t)comm, stream));
    r1_gather_reduce_kernel<<<(T + 31) / 32, 32, 0, stream>>>(
        recv, world, (int)T, draft_vocab_ids, (int*)out_token_i32,
        emit_conf ? bufs.conf_dev : nullptr, step);
    CUDA_CHECK(cudaGetLastError());
}
```

### 2c. Verbatim — dispatch site (`src/core/multi_gpu/tp_group.cpp`, `TpGroup::allreduce_argmax`)

```cpp
    if (impl_->one_shot_argmax) {   // world==2 attractor path, byte-frozen behind its own guard
        require_argmax_transport(size(), true);
        impl_->one_shot_argmax->allreduce_argmax(rank, logits, out_token, draft_vocab_ids, stream, emit_conf);
        return;
    }
    const int world = size();
    const bool r1_ready = impl_->r1_staged && [&]{ for (auto& k : impl_->ranks) if (!k->comm) return false; return true; }();
    require_argmax_transport(world, argmax_route(world, /*ring_present=*/false, r1_ready) == ArgmaxRoute::Rccl);
    const int T = (logits.ne[1] > 1) ? logits.ne[1] : 1;      // *** token count read from dim 1 ***
    r1_allreduce_argmax(impl_->ranks[checked_rank_index(rank, ...)], comm, rank, world,
                        logits.data, logits.ne[0],            // *** n_rows read from dim 0 ***
                        (std::size_t)T, draft_vocab_ids, out_token.data, emit_conf,
                        impl_->r1_step[i], impl_->r1[i], stream);
    impl_->r1_step[i]++;                                       // commit only on success
```

### 2d. Engine-side shapes (the caller of 2c) — `src/runtime/tp2/tp2_backend.cpp`

Staging construction (post loader-fix, world-derived):
```cpp
state->logits      = a_bf(n_vocab);          // n_vocab = per-rank shard length (world-derived now)
state->full_logits = a_bf(tp_w * n_vocab);   // sendcount x WORLD (GATE-3 sized, was 2*n_vocab literal)
```
Decode greedy call site (:2625 region) — with the in-source comment that IS the coordinate contract:
```cpp
    // S3: logits are ColumnN-sharded (each rank owns 124160 of 248320 vocab     [NOTE: comment
    // rows). ncclSum across ranks is WRONG (it sums disjoint token slices).      // written at w=2;
    //   - temp==0 (greedy): fused allreduce_argmax is exact and matches the      // verify at w=4
    ...                                                                            // geometry]
    backend.group().allreduce_argmax(rank, st.logits, st.token, nullptr, s);       // draft=nullptr
```
and the tensor passed in is written by the decoder: `st.text->ordinary_decode_batch(..., st.hidden, st.logits)`. **`st.logits` is constructed as a FLAT 1-D staging buffer (`a_bf(n_vocab)`); the dispatch at 2c reads `ne[0]` as `n_rows` and `ne[1]` as `T`.** Whether the Tensor view handed to `allreduce_argmax` at this call site carries shape `{n_vocab}` or `{1, n_vocab}` or `{n_vocab,1}` was NOT resolved inside this dossier — see Q2 (the two consumers, ring kernel `logits + t*n_rows` and R1 kernel identical stride arithmetic, agree with each other; whether BOTH silently normalize a 1-D tensor to "row 0" at T=1 vs an uninitialized `ne[1]` reading depends on Tensor internals not pasted here: `ne` defaults, ndim).

## 3. Fire history (what has been ruled out, and how)

Nine families found and killed on this path before the soup appeared; each death was LOUD at first frame and each class is now permanently gated (CI/boot-battery cells — so none of them is alive):

| fire | died at | class | cure + permanent cell |
|---|---|---|---|
| 1 | materialize OOM (7102 vs 3820 MB) | loader pair-math (`/2` literals, TP2 template tags) | world-derived loader, check (u) + STEP-0/F |
| 2 | READY+listen, warmup threw `conv_state must have shape [C,3]` | GDN state-spec pair literals | world-derived state spec, class cell |
| 3 | `linear: unsupported Q3 shape` | shape-table TP2-halves whitelists | generated tier_shape_table, Check (v), arm-existence constexpr law |
| 3b | catch-all `ELSE launch<5120,8704>` found PRE-fire by audit | widen-gate-without-arm = silent wrong-stride, `lastError=SUCCESS` | per-K arms + terminal throw; parity cell + compiled collision assert |
| 4 | `gqa_attention: unsupported Q/KV head geometry` (4 ranks) | head-ladder literal {24,16,12} | pair-keyed table arms |
| 5 | second throw in same file `{4,2}` kv whitelist | same family, next organ | one-commit both ladders |
| 6 | `hipErrorInvalidValue` at FIRST post-argmax readback | `plain_tok_pinned[2]` pair-minted, ranks 2/3 read past | world-sized vectors (shape-level) |
| 7-9 | "hang" (probe DEAD) | PROBE GRAMMAR: model_id env landed after those fires → instant model-less 400s swallowed; bodies proven fully-read (httplib dispatch order); zero device work (CU=0) | verdict-grammar fixed; 4-site then 6-edge breadcrumb chain bisected; liveness trio (models+health ALIVE vs chat DEAD) |
| 10 | `EMPTY finish=stop reasoning_len=52` at 8-token probe | thinking-budget truncation (model behavior, NOT a bug) | larger-probe grammar: print finish/reasoning/usage |
| 11 | **SERVES — deterministic soup at GEN=512, all legs, repeats** | **OPEN — this dossier** | — |

Exonerated by positive observation, not absence: loader/state/shape/heads/pinned (all silent 3 fires running); sampler-tear and ring-parity classes (determinism: 4 fresh processes, identical bytes — and R1 route shares NO machinery with the ring: no epoch/flag/tag, RCCL stream ordering); HTTP transport/framing (bodies proven complete at dispatch); engine mutex/locks/capacity ledger (6-edge chain traversed end-to-end twice: `[WDBG] 1→1a→1b(cap-mutex=0x… printed)→1c→1d→1e→2→3(engine-mutex=0x…)→4`). The RCCL exchange itself completes: **no collective error, no hang, and the per-step tokens are PRODUCED — the numbers are just wrong.**

## 4. Ranked hypotheses

**H1 — loader vs argmax DISAGREE on the vocab-row split at w=4 (mint/consume mismatch).** The arithmetic `tok_global = local_idx + rank*n_rows` is only true if rank r's `st.logits` actually contains global rows `[r·n_rows, (r+1)·n_rows)` AND the `output_head` weight rows loaded onto rank r are the same rows. The w=2 comment above the call site ("124160 of 248320") is TP2-era; the loader fix made sizes world-derived but row-BLOCK-vs-alternative-split (e.g. per-rank blocks vs stride-interleave vs head-group-split) was never cross-checked against the `+rank*n_rows` convention — the exact class banked tonight as *"a migrated constant is half-migrated until the MINT side reads the same source as the CONSUME side"* (three prior instances found by the same census). **Signature match:** deterministic, uniform across ranks (all ranks read the SAME wrongly-placed region if the split permutes blocks identically per rank — e.g. every rank's rows are rank-0's block, and `rank*n_rows` then maps them to 4 disjoint wrong regions producing one shared wrong stream after reduce), plausible low token ids (93/107 = wrong region, near table start where untrained/pad weights can peak).

**H2 — the reduce consumes the right bytes but the wrong LAYOUT: `r1_gather_index(T,r,t)=r*T+t` vs RCCL's per-rank sendcount block order.** The allgather `sendcount=r1_send_bytes(T)` bytes of `ncclInt8` — one home exists (`r1_gather_index`) but the PRODUCER side is RCCL's implicit layout; if RCCL places rank blocks in its own order or pads the per-block send region to a multiple (alignment), the reduce reads rank r's champion for token t from a wrong offset — deterministic soup, same across all ranks **only if** the misreading permutes identically everywhere (it would: `gathered[4][T]` of 12-B structs with any consistent skew).

**H3 — T/n_rows shape swap at the dispatch (Q2 below).** Flat `st.logits` read as `{n_rows=ne[0]=62080, T=ne[1]=?}`; if the Tensor wrapper's `ne[1]` is uninitialized/0/1-dependent, kernel 1 strides `t*n_rows` over garbage or computes the reduce for `T>1` tokens of a [1, n_vocab] tensor: token t>0's champion region is OUT-OF-BUFFER (device reads adjacent staging: struct-shaped garbage → low ids). Warmup's step-1 (t=0) uses ONLY row 0 → t=0 is the one clean read — yet step-1 token 93 already looks wrong, which weighs AGAINST H3 unless 93 IS the correct thinking-mode first token (Q1).

**H4 — bf16 decode scale/type misread in kernel 1** (`__bfloat162float` on FP8-requantized shards?) — the served tier is BF16 (`state->logits = a_bf`), so the read matches the buffer; but if the OUTPUT HEAD matmul wrote a requant layout this run (the q3 artifact has W8G32 endpoint rows), the 'floats' are codes-as-bf16 = garbage with structure — consistent with the soup looking like markup (`AM)}`, `sFL`). This indicts the prefill path's logits production, not R1.

**H5 — correct machinery, wrong model content** (thinking-mode soup on a broken KV-prefix for w=4): would fight determinism across fresh processes only weakly — a wrong-but-stable attention state CAN produce stable text. Weakest prior given ring-route w=2 is byte-correct on the same KV code (KVarN paths differ though).

## 4b. UPDATE (same evening, post agent1 static re-read @ 45da0bc1 — annotate, not delete)

- **H1 RESOLVED-BY-READ: FALSE as a suspect** (agent1 seven-leg table): classify_tp gives text/output_head ColumnN; shard_row_split copies the contiguous block [r·62080,(r+1)·62080) for all three planes with the same row0; pass-1/pass-2 geometry-compare THROWS on mismatch; consume side derives 248320/tp_w at tp2_backend.cpp:1177 and kernel-1's `+rank*n_rows` lands exactly on the minted slice. Divisibility fine (62080=485·128). Two w2-era `124160` COMMENTS in the serve path are prose-drift only (fix owed, round-13) — do not chase them.
- **H2 STATICALLY DEAD** (gather-index one home, bijectivity cell-pinned). **H3 LOUD-BOUNDED** (T∉[1,16] throws pre-enqueue; all 7 callsites pass single-token; filed as a standing parity-ARM, not a suspect). Tie-break + conf arms CLEAN (single-home comparator verified at the wire; global-shift softmax re-derived for W=4).
- **MEASURED SOUP SIGNATURE (25-response census at agent1's seat): a TIGHT PERIOD-3–4 CYCLE.** 25/25 responses byte-identical; within each: 和睦 ×4, Mikhail/ifar-family ×3, ftype ×2, stable prefix `AM)}./`; completions 8 tokens, reasoning_len 17–52, finish=stop. This argues AGAINST field-crosstalk (H2-style skew would per-step drift, not cycle-stably survive 25 requests × 4 fresh processes) and FOR **wrong numbers entering a clean wire (H4/H5)**. Counsel: weigh a period-3-4 attractor — that looks like a model decoding its own corrupted-but-stable prefill state (garbage KV → stable nonsense loop), not a permutation of correct logits.
- **Q6 ANSWERED (sharpened [B], agent4 holds)**: print the champion TUPLE (val,tok) per rank for ONE known-input step — warmup's 93/107 sequence is a free probe. **Tuple agreement across ranks = exchange is fine, bug is upstream (H4/H5 with evidence); disagreement = the exchange after all.** Cheap discriminator ordered first.
- **THE GAP STATIC CANNOT CLEAR (Q7-adjacent, counsel's best target)**: between `ordinary_decode_batch(..., st.logits)` finishing the logits write and R1 kernel-1 reading `st.logits`, the ONLY ordering guarantee is stream order on `s`. If prefill/decode writes logits on a different stream or the matmul is async and unjoined on `s`, R1 reads PARTIAL logits — deterministic if the race is deterministically scheduled. Check: same-stream enqueue chain, or a missing event/wait at the boundary. (world=2's ring route consumes the SAME buffer the same way and is correct — so either the race is w4-specific by scheduling, or H4 content-wrongness predates the read.)
- **Cross-artifact datum**: the q3 (currently booted) artifact's endpoint triple measured = W8G32/row-split/1,350,860,800 — IDENTICAL to nvfp4's plan; 'endpoint format surprise' RETIRED as a differentiator between artifacts' serve paths.
- Retraction held at register: draft-remap bounds was filed-then-withdrawn same-edit (booted config: MTP k=0, plain decode passes nullptr draft ids at :2625) — a LATENT-only suspect; do NOT spend counsel cycles on it.


- **Q1** — is `[SEQ-STEP1] step=1 next=93 (cpos=53)` PLAUSIBLE as the correct first continuation token of a thinking-mode Qwen3-family model at all (token 93 near BOS/specials: what do ids 93, 107 typically decode to in HF Qwen2/3 vocabularies)? If 93/107 are structurally-impossible first tokens, H1/H2/H3 harden; if they're plausible (`!`, newline-ish), the soup may start LATER than step-1 and prefill (H4) rises.
- **Q2** — given ONLY the pasted code, what is `logits.ne[1]` when `st.logits` is a flat `a_bf(n_vocab)` staging? Trace both consumers: ring kernel (:95 region, `token_logits = logits + t*n_rows`) and R1 dispatch (`T = ne[1] > 1 ? ne[1] : 1`). Does world=2 pass for the same reason only because its fused ring kernel takes T as a separate template/constant? If world=2's call reads T from `kMaxTokens` or a fixed 1, the two routes DISAGREE on where T comes from — that asymmetry alone can be the bug (H3).
- **Q3** — audit `r1_gather_index(T,r,t)=r*T+t` against RCCL's documented AllGather layout for `sendcount` BYTES with `ncclInt8`: per-rank block size = `sendcount*datatypebytes` — is any per-rank alignment/padding documented (e.g. block stride rounded to 512B / word) that could skew a 12-byte wire at world=4 (12·16=192 B send per rank → block padding candidate)? If yes: H2 with the padding as the offset.
- **Q4** — in the loader contract quoted (2d: sizes world-derived), what is the ONE place that asserts rank r's `output_head` ROWS are `rows[r·n_rows:(r+1)·n_rows]`? (grep the materializer for the head's row-slicing: if it slices by `n_draft_local`/`n_prop_vocab` arithmetic or a `dst_geom` from a plan whose axis was fixed at w=2, H1 is proven; if the slice is correct but the ORDER of `st.logits` is per-head-GROUP rather than contiguous, same conclusion by another route.)
- **Q5** — determinism-shape reading: all 4 ranks emit the SAME final text. R1's reduce is a per-rank LOCAL function of the gathered array, so identical outputs prove identical gathered arrays OR identical winners. Which failures make gathered data identical-but-wrong (H1: yes if same block misplacement everywhere; H2: yes; H4: yes — prefill garbage shared post-reduce)? Design the ONE print that separates them (see Q6).
- **Q6** — proposed decisive next instrument (already queued on this board): per-rank, at the staging site, print `(rank, step, t, local_idx, tok_global, val, sumexp)` for champion-send AND a per-rank digest of the gathered array post-recv. Is there a strictly cheaper discriminator — e.g. a host-side unit simulation replaying the exact kernel folds on synthetic logits with known split, or running the same prompt at w=4 with a SYNTHETIC identity-permutation head (row-slice test pattern) so wrong-region reads announce themselves as recoverable indices rather than soup?
- **Q7** — anything about `ncclInt8` allgather with `sendcount=192` bytes on RCCL-2.20.5/gfx900 P2P-less cross-package (the box is 2 dual-die boards, `distinct_board_keys=2`, package-crossing at {0,2}) that could silently ZERO-PAD a small transfer? B-1 measured this exact geometry alive (first-ever comms established) but never graded 192-byte payloads' contents.

## 5b. SAME-EVENING AMENDMENT — the request lane's first sample EXISTS (agent1 @ 07bb35dd; fold law from the chair):

`[SEQ-STEP1] step=1 next=93021 (cpos=54)` / `step=2 next=14 (cpos=55)` — G18w4p serve-log :1414/:1607, the REQUEST lane's genuine first tokens (the warmup lane's `93/107` pair sits in the same log; line numbers are the lane discriminator, and a grep without them mis-ships the values story). **93021 lands inside rank-1's shard territory [62080,124160)** as a first generated token — either a legitimately-high first token or the wrong-region signature the soup implies; the 93021→14 transition is cycle material. Counsel target: map id 93021 in the qwen3.8 tokenizer — real-word vs pad/marker-dense region discriminates 'wrong argmax picked a plausible far token' from 'argmax over garbage landed mid-range'. Also ruled BEFORE the next rows exist: a 'missing join' hypothesis is NOT-OBSERVABLE-IN-PRINCIPLE on any boot that printed step-2 (allgather all-or-nothing ⇒ step-1 joined four-way); the residual ordering tell is per-rank step-COUNTER skew, which the next trace print carries (step= per rank, all-equal ⇒ permanent counter-join cell).

## 5c. SAME-EVENING AMENDMENT 2 — tokenizer decode grid (agent1 @ f386cd7c, artifact-carried frontend/tokenizer.json, 12.8 MB resource object, payload at the 4096-align law):

**Every emitted id is a REAL vocabulary token — the wire transports faithfully; H2's strong form is DEAD BY OBSERVATION.** Grid: request-lane `93021=')}'` (rank-1 shard), `14='/'`; soup's leading `AM'=1354`; warmup-lane `93='~'`, `107='¯'` (rank-0 low region). The soup is an **ATTRACTOR CYCLE THROUGH REAL PUNCTUATION** and it is **LANE-AGNOSTIC** (both the warmup and request lanes emit region-junk from their own shards) — narrowing H4/H5 to **shared per-step state or weights that BOTH lanes read**. Prime suspect as revised at the chair desk, stated for challenge: **GQA attention consumers at kv_local=1** — the heads-axis fix DERIVED the geometry, but no value-grade verdict exists for KV cardinality at 1 kv-head/rank (world=2's ring era only ever ran kv=2; the soup's stability fits stable-wrong-attention-state, H5 surviving WITH H4's prefill provenance). The [B] tuple print's remaining job is unchanged and singular: do champion VALUES match the logits prefill claims wrote — values DISAGREE ⇒ wire re-lives (named step); AGREE ⇒ proof-of-wrong-inputs and the diagnostic turns to a bounded logits-row dump (top-8 + ids, sized before printing).
- NEW PERMANENT-CELL CANDIDATE (agent1, NOT-suspect-as-cause, filed anyway): vocab_size 248044 < endpoint rows 248320 ⇒ a 276-id PAD REGION that decodes to NOTHING, fault-free, glyph-less; gate = `token_id < vocab_size` at the argmax consumer (require_positive-finite analogy on the ID axis). Related accept-set question routed to the NVFP4 desk: does any declared row carry the vocab extent so pad arithmetic is graded, not assumed?
- Register line from the lookup itself: the first read hit ZEROS — resource payloads are 4096-ALIGNED like tensors; reader.cpp's own comment is the cure; next desk skips the retry.

## 5d. FINAL SAME-EVENING STATE (post-[B], pre-[C]) — the shortlist is now exactly ONE organ:

- **[B] tuples (request-lane, G18w4s):** rank0 local `(v=11.125 tok=51910)`; gathered view shows faithful shard placement `(51910)(62080+3806)(124160+30803)(186240+30310)` — the **+rank*n_rows contract holds TO THE ID through the exchange** (mechanics exonerated with evidence; rank3's printed t=228176 is an arithmetic slip on the print — 186240+30310=216550 — suspected print-path bug, flagged, does not touch the exchange).
- **THE UNRAULITED LEG:** no trace line yet shows the reduce's ANSWER and the token WRITTEN as one pair — warmup-lane step-1 shows emitted next=107 with reduce max at tok=1076 — a candidate post-reduce disconnect (or a counter-base mispair: r1_step is per-rank cumulative across lanes, SEQ-STEP1 is per-request — the mint/consume law, one organ later). The next fire carries `-> tok=X out[0]=Y` in one print with a lane/step mapping; a mismatch will then convict itself.
- **step=0 all-zero champions on all four ranks**: benign-init vs read-before-write (the stream-order gap made real) — first-value check ordered same fire.
- **agent1's pre-dark static read (ba42523c) — [C]'s first line:** every serve-reachable attention ladder is TERMINAL-LOUD at kv_local=1 (eight ladders enumerated, each ends in a throw naming its numbers; the wrapper is a table view with dual refusals; `head_key_collision_free()` static_assert fences the q-keyed projection; the q=16 dual-claim landmine is NOT-YET-EMITTED with the migration path written). And [B] PROVED the (6,1,2) kernel is SELECTED (tuples ran through it). What remains ungraded is exactly: **the NUMERICS of that selected kernel — DecodeSplits=170 partials folding over a single kv-head, and k4v2 plane widths at kv_local=1.** 'Right-kernel-or-refusal never proves kernel-computes-truth.'
- **Model-context note for counsel:** the served artifact is QAT NVFP4 evaluated by fake-quant round-trips (weights W4 group128 + FP8 scale-plane + bf16 requanted activations; endpoints W8G32); numerics failures would present as stable garbage on W4 while W8 (endpoints) stays sane — 'the endpoints moved but the stream didn't' is consistent with that reading, and `reuse=full_reset` on every request means no cross-request KV carry is involved.
- **Status:** nine families closed at shape level, family #10 (this one) narrowed to a single named organ with a pre-declared discriminator one fire away. Counsel questions Q1/Q4/Q5 are now ANSWERED by observation (kept for provenance); Q2/Q3/Q6/Q7 remain live-open.

## 5e-pointer (2026-09-15 ~07:4xZ, chair pen): THE COUNSEL FOLD SHIPPED EARLY AND WITHOUT COUNSEL — agent1's `S5E_counsel_fold_answered_by_observation_agent1_2026-09-15.md` (landed in-tree, content8 b64712f2, authored @ 75e455d0) retires Q2/Q3/Q6/Q7 ANSWERED-BY-OBSERVATION with banked proofs (Q2 dead thrice incl. r23's zero-ambiguity channel map; Q3 exonerated by four independent bit-consistent witnesses; Q6 superseded twice — invariant-scalar channels graded cross-world float from EXISTING prints; Q7 mooted by positive localization: bit-clean arrivals + bit-disjoint states, transport explains neither and both). Counsel's text APPENDS when/if it lands — observation-wins-with-dissent-recorded, the same grammar §5d applied to Q1/Q4/Q5. THE R23 VERDICT SUPERSEDES §5e'S PREDICTION ROWS WHERE THEY DIFFER: organ NAMED = GDN recurrent update at world=4 (M1/M2/M3), channel map as data (512-block interleave), fix PR with pre-registered counter-tests — see docs/amd/R23_plane_map_verdict_agent1_2026-09-15.md.

## 5e. MORNING AMENDMENT (2026-09-15, chair-authored from two-desk measured rows) — THE HUNT ARRIVED: PREFILL NUMERICS CONVICTED AT STEP ZERO; COUNSEL'S OPEN QUESTIONS MOSTLY RETIRED BY OBSERVATION

- **[C] fired and verdicted (G18w2u_c / G18w4u_c, r18/r19 bin, banked at results/amd/p3/ + cdump_w2/ + cdump_w4/):** the PH tap (LM-head INPUT, prefill) DISAGREES cross-world at element 0, step 0, on BOTH paired prompts. w2's prefill winner: 760='The' with ~10.5-logit lead (exactly agent1's corpus-predicted opener — 50/50). w4's: 1076/1354 at a 0.56 lead. THE FLIP IS STRUCTURAL, NOT NUMERICAL-NOISE: reassociation (the w2!=w4 kernel caveat, §4b/agent1 T1) is KILLED BY LEAD MARGINS. H5 (wrong-model-content) dead; LM-arm branch (PH-small/PL-flip) not reached — its input is already wrong, sufficient alone. **FAMILY #10 = PREFILL LAYER-STACK NUMERICS AT kv_local=1, UPSTREAM OF THE LM-HEAD.**
- **What exonerated IN THE SAME TABLE (all value-grade, file-parsed):** exchange = 88/88 G5 self-convictions (tok==out[0], remap=NONE), zero pad winners (G6 + counter-join K6 host leg), and the HIDDEN ALLREDUCE is SOUND BOTH WORLDS — CH cross-rank BIT-EXACT at every step (first-ever value-grade of that organ). §4b read-before-write: closed statically (agent1 total-write + one-stream) AND empirically (no triple-law shape in the captures). The dossier's indictment chain is now: H1 dead (read), H2 dead (observed + self-conviction), H3 bounded (read), H4's 'prefill logits production' branch CONFIRMED GUILTY at its attention/layer-stack source, H5 dead (margins).
- **Instrument honesty rows (why two table legs don't convict):** agent4's grader G2 DISAGREE lines are stderr-interleave artifacts (gathered record = 3+ fprintfs vs local's 1: locals 1992/1992 parse, gathered 0/498 four-rank-complete; cure = one-sprintf print site, PIPE_BUF-proven, rides r21) and G3's 24 'mismatches' are a KEY-SPACE MISJOIN (filename round is request-relative, trace step is cumulative r1_step — under ordinal join, trace and dump AGREE digit-for-digit at all 4 ranks, which itself CLOSES the claim-vs-float loop [B] could never reach: kernel-1's printed tuples are float-truthful). Adjudicated by agent1's arithmetic cross-check (eac03f0f) + agent3's r15 measurements; a grader that can false-convict two ways has both fixes ordered with closure cells.
- **COUNSEL QUESTION STATUS (so nobody waits on answers we've outgrown): Q2 ANSWERED-BY-OBSERVATION** (T/shape dispatch: seven-leg table, H3 loud-bounded, and route-selection now proven by the (6,1,2) tuples running to argmax); **Q3/Q7 MOOT** (12-B wire layout/padding — exchange self-convicted clean end-to-end; the RCCL block never falsifies); **Q6 ANSWERED AND SURPASSED** (the champion-tuple print was built, fired, agreed — the cheaper discriminator question retired by its own success). Q1/Q4/Q5 were already answered @ §5d. Counsel's remaining unique value: the §4b summation-order question at kv_local=1 — the SIMT prefill kernel's fold order vs the split-KV law — which the LAYER BISECTION (WO-r21, agent1 authoring, one boot) localizes empirically regardless of counsel return. THE BOARD NO LONGER BLOCKS ON COUNSEL.
- **NEXT INSTRUMENT (named, funded, single-boot):** per-layer prefill taps hunting the FIRST-BAD LAYER — invariants-first (fixed-point row-norm/lse scalars into the existing int32 dump; full hidden dump only at the bisected layer), w4-internal rank-consistency always-valid, cross-world tolerance-graded, first-bad-layer id becomes a permanent register row. If layer-0-adjacent (embed/rope/KV-CONTENT entering prefill), cure class changes from kernel-math to data-placement — the dossier's last named fork.

## 6. Reproduction / consultation pointers

- Every fire's rows: `results/amd/p3/G18w4*_p3serve_{step0,serve,responses,capture,manifest}.*` in lane `amd-wo-p3-serve` (worktree `/home/chris/worktrees/amd-wo-p3-serve`). Boot bytes: `ninfer-serve_4349e204d9275663.bin` (bank `/home/chris/artifacts_bin/`, filename==sha256-prefix).
- Arming battery (author-absent callable, main tree): `bash tools/v340l/tp4_arming_battery.sh <ref> /home/chris/dual_5060_ti_ninfer` — 19 host legs incl. the ring-vs-R1 guard, arm-existence and rank-state censuses. All green at the soup tip: **the bug is invisible to every existing static cell** (value bug, not shape) — that fact is itself evidence for H1/H2/H4 over H3-adjacent shape errors.
- The route contract in full: `docs/amd/WO_TP4_B_R1_DESIGN_HANDOFF_agent4.md` (port spec §1, tie-break/conf laws, remap-last), the wire law in `argmax_r1.h` header comments above, single-home reduce in `argmax_reduce.h`.
- The world=2 oracle corpus + attractor: 25 fixed requests; w=4 grading contract: same corpus, per-leg digest vs `348e77a1222dea7f` (w=2) — currently failing, deterministically.

**Constraint notes for counsel:** this project bans estimate-based refusals and trusts measured allocator answers (VRAM law — budget advice of the 'reserve X MiB' shape is out of scope); determinism-equality is the acceptance bar for the transport (any fix must keep it or replace the grading); world=2 is byte-frozen — proposals must not touch the ring.

*Assembled by the coordinator desk, 2026-09-14 ~23:5xZ, from: agent4 (fire history, breadcrumbs, loader fixes), agent1 (dict/audit lineage, static re-read pending — their pass will append to THIS file), chair seat (code capture above). Claims here are quoted code + pasted strings, not inference; the ranked-hypotheses section IS inference and marked as such.*
