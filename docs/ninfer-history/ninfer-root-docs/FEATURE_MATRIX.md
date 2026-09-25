# ninfer Feature & Divergence Comparison Matrix

## START HERE — what this is (3 sentences)

This project is a homemade, high-performance serving system for one specific large language
model (Qwen3.8-27B) that runs on TWO consumer graphics cards (RTX 5060 Ti, 16 GB each)
working as a team — hardware that normally could not hold or run a 27-billion-parameter
model at all. It talks the standard OpenAI-style chat API, so any chat program can use it,
and it is packed with speed tricks (guessing several words ahead, compressing the model's
memory, reusing previous work) that push past what the stock open-source tools can do on
the same hardware. This document is the honest feature list: what works, what's
half-built, what's parked, and what's next — each explained for someone who has never
run a local AI model.

---

## 1. Plain-English Feature Dictionary (The "Grandmother Test")

Every core feature, explained for someone who has never run a local AI model. The rule:
after the name and the description, you should know what the thing does FOR YOU and why
you would care.

- **Tensor Parallelism (TP2) — two cards acting as one**
  *Plain English*: The AI model is too big to fit on one graphics card, so it is sliced
  in half across two cards that work in perfect lockstep — like two people jointly
  carrying one very long ladder. Every word of output is computed by both cards
  together. **Why you care**: without this, a 27-billion-parameter model simply cannot
  run on consumer hardware.
- **Speculative Decoding (MTP) — guessing ahead, then checking**
  *Plain English*: Normally the AI writes one word, then the next, then the next — one
  slow step at a time. MTP ("Multi-Token Prediction") makes the model guess the next
  SEVERAL words in one go, then checks all the guesses in a single pass: every guessed
  word that matches what the model would have said anyway is kept for free. It is like
  a fast typist typing a whole phrase while a proofreader only needs to skim it.
  **Why you care**: roughly 2–3× faster answers with zero change in quality.
- **Batched Serving — helping several people at once**
  *Plain English*: Serving multiple conversations simultaneously: while one person is
  reading their answer, the cards are already working on the next person's request in
  the very same pass. **Why you care**: more total work done per second when more than
  one person (or tab, or app) is using the server.
- **KV Cache — the model's memory of the conversation so far**
  *Plain English*: An AI model has no memory between words — to "remember" the
  conversation it must carry notes about every previous word. Those notes are called
  the KV cache ("Key-Value cache"), and they grow with every word of your document and
  every word of the answer. **Why you care**: the size of this memory decides how long
  a document the model can read and how long it can keep talking.
- **KV Cache Storage Tiers (BF16, INT8, INT4, NVFP4, KVarN) — shrinking the notes**
  *Plain English*: The memory notes can be written in smaller handwriting. "BF16" is
  full-size notes; "INT8"/"INT4" are half- and quarter-size; "NVFP4" is a special
  quarter-size format supported natively by the newest graphics cards; "KVarN" is a
  custom codec from this project. Smaller notes = 2–4× more conversation fits on the
  same card. **Why you care**: a longer memory (longer documents, longer chats) on
  hardware you already own.
- **Host-KV Offload Safety Net — an overflow tank for the memory**
  *Plain English*: When a conversation grows so long that its notes no longer fit on
  the graphics cards, the system automatically parks the oldest notes in regular
  computer memory (system RAM) and brings them back seamlessly when the conversation
  needs them again — like moving boxes you rarely open into the garage. **Why you
  care**: very long sessions stop crashing the server.
- **Linear Attention State (GDN) — a smarter kind of memory for some layers**
  *Plain English*: Some layers of this model don't keep a note per word; instead they
  keep one compact rolling summary that is updated word by word (like keeping a running
  grocery list instead of every receipt). This makes very long documents extremely
  cheap to read. **Why you care**: it is a big part of why huge contexts are possible
  on small cards.
- **Determinism & Reproducibility — same question, same answer, every time**
  *Plain English*: Ask the identical question with identical settings twice, and you
  get the exact same answer down to the last character. **Why you care**: repeatable
  tests, and bugs can't hide behind "the AI was in a different mood."
- **VRAM Budget Model & Refusal Gate — the bouncer that reads the room**
  *Plain English*: Before starting, the system measures what the model actually needs
  (by weighing it, not by guessing from a formula) and refuses to launch configurations
  that would really run out of graphics-card memory. A recent recalibration replaced
  guessed numbers with measured ones, which removed ~1.5 GB of imaginary memory
  requirements — a 200,000-word context that used to be refused now runs. **Why you
  care**: maximum conversation length without mysterious "insufficient memory" errors.
- **Prefill and Decode — reading vs writing**
  *Plain English*: "Prefill" is the AI reading your prompt (fast — thousands of words
  per second); "Decode" is the AI writing its answer (slower — tens of words per
  second). "TTFT" (time-to-first-token) is how long you wait before the first word
  appears. **Why you care**: these are the two speeds you actually experience.
- **Prefix Caching & State Reuse — never re-read the same thing twice**
  *Plain English*: If two requests start with the same long instructions (the same
  system prompt, the same attached document), the system remembers the reading work
  and skips straight past it. **Why you care**: near-instant first answers for
  repeated setups.
- **Adaptive MTP Depth — the guesser that learns from its own hit rate** (its own
  feature; docs/69)
  *Plain English*: After each round of guessing, the model counts how many of its
  guesses were accepted. If it keeps guessing well, it tries LONGER chains next round;
  if it guesses poorly, it SHORTENS them. This is separate from the confidence break
  below — it tunes how many words to ATTEMPT, based on results. **Why you care**: more
  speed exactly when the text is easy, no wasted checking when it isn't. Status:
  **LANDED** in single-request mode (off by default; the switch is `mtp_adaptive`).
- **Confidence Break (W5) — the bail-out switch mid-guess** (a SEPARATE feature; not
  part of adaptive depth)
  *Plain English*: While the guess chain is being produced, if confidence in the
  current guess sinks below a threshold, the rest of the chain is abandoned NOW
  instead of checking guesses that were heading nowhere. **Status**: the two-rank
  race in the mechanism was fixed and proven; a SECOND defect (hangs on long
  documents with it on) keeps it switched OFF — see the roadmap's BLOCKED row.
- **N-gram Pool & Context Lookup Drafting — the copyist**
  *Plain English*: When the text contains repeats (names, code snippets, quotes), the
  system notices and copies those fragments directly as its guesses instead of
  re-deriving them. **Why you care**: repetitive content (lists, code, logs) is
  predicted almost perfectly and for free.
- **Magic Dictionary — teaching the guesser your own vocabulary**
  *Plain English*: The built-in guesser uses a generic word list; this feature builds a
  specialized word list from YOUR documents and the model's own accepted answers (with
  a verified byte-exact match to the official tokenizer across 256,916 lines), so
  guesses on your domain's names and jargon hit more often. Includes an output-trained
  phrase pool. **Why you care**: agent-style and domain-heavy work guess better.
  **Status**: the vocabulary builder, counter, and serving hook are LANDED and proven
  observation-only (zero behavior change); the acceptance A/B (does it clear the
  +3-point bar?) is designed but not yet run.
- **DFlash — a second, bigger guesser (BUILT, ON HOLD)**
  *Plain English*: An alternative to the built-in guesser (MTP): a small separate
  helper model, built into this project (a 6-layer drafter that can guess up to 15
  words ahead one at a time), reading hints from the main model. **Status: fully
  built, ON HOLD by user decision** (the switch is off in the model's config: the
  drafter is disabled, zero draft words). **Why you care**: when switched on it could
  guess longer chains than MTP; the follow-up "DFlash2" design (below) would improve
  on it further.
- **DFlash2 (planned) — guessing a whole block in one shot**
  *Plain English*: The next generation of the idea: instead of guessing word-by-word,
  a small 1.9-billion-parameter helper model produces an entire BLOCK of ~7 words in a
  single parallel pass (like dictating a sentence instead of spelling it). On other
  hardware it measured ~34% more guessed words per round than MTP. **Why you care**:
  potentially the biggest remaining speed multiplier. Status: fully scoped and
  designed; the official building blocks (the drafter's own trained files and a
  reference implementation) are public.
- **CI Exit Contracts & Mutation Testing — the alarm system that tests itself**
  *Plain English*: The project's automated tests deliberately introduce fake bugs
  ("mutations") to prove the alarms actually ring — a test suite that catches a
  planted bug is trustworthy; one that misses it isn't. **Why you care**: it is why
  the speed features above can be shipped without quietly breaking your answers.
- **Teardown & Process Hygiene — leaving the playground tidy**
  *Plain English*: Test scripts used to shut down AI servers by shouting a name that
  matched EVERY server on the machine — including other projects'. All shutdowns are
  now surgical (each script stops only the exact server it started, verified by
  process ID and network port). **Why you care**: nothing you are running gets
  accidentally killed.

---

## 2. Feature Comparison Matrix

*Audit State: main @ `55a41b81` (VRAM refusal-gate correction landed) + local `e278bd4f`
(`wo/single-parity` & `wo/radiance-launch-gap` integrated). Columns: this project's
single-request mode ("Single-Seq"), this project's multi-conversation mode ("Batched"),
the original pre-rewrite baseline of this project, vLLM (the industry-standard serving
stack, from public documentation), and llama.cpp (verified in local checkouts).*

| Feature & Plain-Language Summary | US Single-Seq | US Batched | ORIGINAL NINFER | vLLM | llama.cpp |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TP2 — two cards as one** | **Present** — fused two-card handshake, custom peer-to-peer link. | **Present** — same primitives across multi-lane batches. | **Present** — basic two-card engine, slower handshakes, double overhead. | **Present** — industry-standard multi-card via NCCL. | **Partial** — row/layer splitting across cards, no fused two-card reductions. |
| **MTP — guess ahead, check once** | **Present (full surface)** — native guesses (k=3), adaptive depth, n-gram pool, context lookup, confidence break (W5: race fixed @ `e278bd4f`, default OFF pending user flip + long-prefill defect, see roadmap). | **Present (uniform)** — fixed guess depth for all conversations; no adaptive depth, no n-gram pool, no lookup, no confidence break. | **Partial** — guessing present but flawed by late memory rebinding and launch-gap stalls. | **Partial** — speculative decoding via separate draft models; the native MTP-head style used here needs custom support. | **Partial** — separate draft-model speculation; local fork has MTP-style context support for related architectures. |
| **Batched Serving** | **Absent (by design)** — dedicated single-conversation mode with zero contention. | **Present** — multi-lane serving (2 lanes × 40,960 tokens), queueing, arrival-order fairness proven. | **Partial** — batching present but order-invariance was broken and memory budget was over-charged. | **Present** — continuous batching, dynamic queueing. | **Present** — multi-slot server queueing. |
| **KV Cache Tiers** | **Present (all tiers verified)** — BF16, INT8, INT4, NVFP4 (7/7 unit tests), KVarN (k4v2, k4v4, k5v4). | **Partial** — BF16/INT8/INT4/KVarN verified; NVFP4 is the one unverified tier (no batched prefill kernel — see roadmap). | **Partial** — BF16 and INT8 only. | **Partial** — FP16/BF16/FP8/INT8/INT4; no NVFP4 KV format. | **Present** — broad integer quantization; no NVFP4 KV format. |
| **Host-KV Offload Safety Net** | **Present, LANDED & VERIFIED** — 45k-token × 3-session park/restore proven byte-identical; wild-write and scale-plane defects fixed. | **Present, LANDED & VERIFIED** — per-lane park/restore on main; CI gate passed. | **Absent** — over-long sessions crashed or were refused. | **Present** — CPU swap space with LRU eviction. | **Absent** — CPU KV placement exists but no dynamic VRAM↔RAM paging during serving. |
| **GDN Linear Attention** | **Present** — fused by-value params, zero-copy views, mid-prefill checkpoints. | **Present** — multi-ring GDN state across lanes. | **Partial** — three extra device-to-device copies per decode step. | **Partial** — hybrid Mamba/RNN models supported, but no fused two-card GDN path. | **Present** — recurrent/conv state per sequence, standard scheduling. |
| **Determinism** | **Present** — bit-exact greedy; reproducible randomness. | **Present** — arrival-order invariance formally proven under concurrency. | **Broken** — output changed depending on the order requests arrived. | **Partial** — deterministic on one card; multi-card can reorder floating-point math. | **Partial** — deterministic single-threaded; parallel streams can reorder sums slightly. |
| **VRAM Budget Model & Refusal Gate** | **Present (corrected @ `55a41b81`)** — gate charges the MEASURED floor + context + 256 MiB margin; int8 at 200k context now launches without override (measured 15,069 MiB, 780 MiB spare). | **Present** — per-conversation memory charge honest within 26 MiB; physical fences tightened (50 MiB / 20 MiB baselines). | **Defective** — imaginary overhead multiplier and ~1.5 GB of never-touched reserves falsely capped capacity. | **Present** — dynamic startup profiling assigns all leftover space to KV. | **Present** — static buffer estimation at context creation. |
| **Prefill / Decode Throughput** | **Present (lowest latency)** — prefill 810 tok/s; MTP decode 81 tok/s; int8@200k verified at 617.8 prefill / 59.0 decode. | **Present (max total throughput)** — 86.2 tok/s aggregate on 2 conversations (2.36× plain decode). | **Partial** — slower prefill; decode throttled by launch gaps and extra copies. | **Present** — tuned for datacenter cards; on consumer cards the card-to-card link limits decode. | **Partial** — strong single-card; consumer multi-card scaling limited by host sync. |
| **Prefix Caching & Reuse** | **Present (full)** — longest-common-prefix detection, state restore, mid-prefill checkpoints, open-page tail restore. | **Absent** — cache invalidated on entry; every conversation re-reads from scratch. | **Defective** — late memory rebinding corrupted concurrent requests. | **Present** — automatic prefix caching across requests. | **Present** — prompt cache + explicit KV cache copy/remove calls. |
| **DFlash — second, bigger guesser** | **BUILT, ON HOLD by user decision** — the 6-layer drafter mechanism exists end-to-end in the runtime (context, feature feed, rewrite checkpoints, CUDA-graph profiles); the 27B model's config has it switched off (zero draft words). Would guess up to 15 words ahead vs MTP's 7. | **Absent.** | **Absent.** | **Absent** (no equivalent separate 6-layer in-tree drafter). | **Absent.** |
| **Adaptive MTP Depth (its own feature)** | **Present (LANDED)** — acceptance-feedback-driven chain length, bounded [min, max]; off by default (`mtp_adaptive`). | **Absent** — fixed depth only (Inverse-Gap Lane, roadmap). | **Absent.** | **Partial** — some speculative stacks adapt draft length; not this form. | **Absent.** |
| **Confidence Break (W5, separate feature)** | **Present (mechanism fixed; OFF — BLOCKED, see roadmap)** — mid-chain early exit when confidence sinks; the two-rank race is fixed, but a SECOND defect hangs long prompts with it on. | **Absent.** | **Absent.** | **Absent.** | **Absent.** |
| **N-gram Pool & Context Lookup** | **Present** — shared 24-gram pool + direct prompt/history fragment drafting. | **Absent.** | **Absent.** | **Partial** — prefix/n-gram style reuse exists in some stacks. | **Absent.** |
| **CI Contract Guard & Mutation Testing** | **Present** — static contract linter (28/28) + Rule 14 deliberate-break tests per cell. | **Present** — formal CI cells with mandatory mutation tests. | **Partial** — smoke tests without formal gates or mutation checks. | **Absent** — standard CI, no runtime mutation gating. | **Absent** — standard CI, no mutation gating. |
| **Teardown & Process Hygiene** | **Present (PID-scoped, landed @ `48bff27f`)** — every shutdown stops only its own recorded process; system-wide name kills eradicated repo-wide. | **Present** — per-child PID verification, clean card release (≤20 MiB). | **Dangerous** — uncoordinated system-wide kills across 5+ scripts hit other agents' live servers. | **Present** — standard process-group/Ray supervision. | **Present** — single-process, clean signal handling. |

## 3. Single-Sequence vs Batched Serving — Full Parity Audit

### 3.1 Executive Summary & Plain-Language Motivation

For an introductory overview of ninfer and plain-language definitions of all core concepts without engineering jargon, refer to the **START HERE** overview and **Plain-English Feature Dictionary** in [`FEATURE_MATRIX.md`](file:///home/intel/ninfer/repo/docs/FEATURE_MATRIX.md).

In dual RTX 5060 Ti serving, multi-sequence batching (`max_concurrency >= 2`) improves aggregate token throughput under concurrent load (e.g. 86.2 tok/s aggregate at concurrency=2 vs 81.3 tok/s sequential MTP). However, concurrency imposes structural and memory trade-offs:
1. **KV Capacity Partitioning**: High-speed graphics memory (VRAM) dedicated to conversation memory (the KV cache) is partitioned across lanes (`pages_per_lane = cap / lanes`). A single-sequence request at `lanes=1` receives **2× the effective conversation memory** (e.g., 81,920 tokens vs 40,960 tokens/lane).
2. **Memory Staging & Reserve**: MultiBatch staging buffers scale with lane count. Following the VRAM ledger calibration ([`docs/VRAM_LEDGER.md`](file:///home/intel/ninfer/repo/docs/VRAM_LEDGER.md) @ commit `55a41b81`), the refusal gate charges the true measured floor (12,998 MiB) plus context and a 256 MiB margin, retiring the ~1.5 GB phantom reservation tax.
3. **Feature Asymmetry**: Single-sequence mode currently supports the full algorithmic surface (adaptive MTP draft depth, shared n-gram candidate pool, context lookup drafting, prefix caching with mid-prefill GDN restoration, and in-chain confidence break). The batched runner operates with fixed draft depths, disables prefix reuse, and invalidates cached states. Closing this gap is scheduled in the roadmap as the *Inverse-Gap Lane*.

---

### 3.2 Option Plumbing Audit Matrix

*Current State: main@`55a41b81` + local `e278bd4f` (`wo/vram-ledger` & `wo/radiance-launch-gap` integrated).*

| Feature / Option | Single-Sequence Runner (`run_tp2_request`) | Batched Runner (`run_tp2_requests_batched`) | Parity Status | File & Line Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **KV Cache: BF16** | **Supported**<br>Standard 16-bit unquantized KV storage; fused append and decode attention. | **Supported**<br>Dtype-generic fused-append route (Phase C, commit `e804971f`). | **Parity** *(verified)* | Single: [`tp2_backend.cpp:1831`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1831)<br>Batched: [`tp2_backend.cpp:2582`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L2582) |
| **KV Cache: INT8** | **Supported**<br>Group-64 symmetric quantization; fused append + slice3/slice4 decode. | **Supported**<br>Rides fused-append batched MultiBatch route (Phase C). | **Parity** *(verified)* | Single: [`tp2_backend.cpp:391`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L391)<br>Batched: [`tp2_backend.cpp:2582`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L2582) |
| **KV Cache: INT4** | **Supported**<br>Group-64 symmetric 4-bit codes with FP16 group scales. | **Supported**<br>Rides fused-append batched MultiBatch route (`launch_tc_partial_unified_i4`). | **Parity** *(verified)* | Single: [`tp2_backend.cpp:393`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L393)<br>Batched: [`gqa_attention_decode.cu:309`](file:///home/intel/ninfer/repo/src/ops/launcher/gqa_attention_decode.cu#L309) |
| **KV Cache: NVFP4** | **Supported**<br>Group-16 E2M1 codes + E4M3 scales; in-smem Q rotation + on-the-fly dequant. Verified via CTests (7/7 pass). | **REVERSE GAP (Unverified / Broken)**<br>Batched kernel template instantiated, but device tests are 100% batch=1. No batched prefill kernel; `gqa_attention_cached_batched` throws `invalid_argument`. | **Asymmetric Defect** *(unverified/red)* | Single: [`test_nvfp4_attention.cu:184`](file:///home/intel/ninfer/repo/tests/test_nvfp4_attention.cu#L184)<br>Batched: [`gqa_attention.cpp:936`](file:///home/intel/ninfer/repo/src/ops/wrapper/gqa_attention.cpp#L936) |
| **KV Cache: KVarN** | **Supported**<br>All tiers (k4v2, k4v4, k5v4); per-sequence workspace tile and open-page tail management. | **Supported**<br>Dedicated batched attend (`kvarn_attend_text_batched`), per-lane workspaces, tail migration. | **Parity** *(verified)* | Single: [`tp2_backend.cpp:1288`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1288)<br>Batched: [`tp2_backend.cpp:2852`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L2852) |
| **MTP Speculative Decoding** | **Supported**<br>Multi-Token Prediction verify rounds (k=3); 1-shot candidate expansion. | **Supported**<br>Uniform MTP verify rounds across live lanes; verify width T = k + 1. | **Parity** *(verified)* | Single: [`tp2_backend.cpp:1894`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1894)<br>Batched: [`tp2_backend.cpp:2600`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L2600) |
| **DFlash Speculative Drafting** | **BUILT / ON-HOLD (User Decision)**<br>In-tree implementation (~1134 MiB weights + 40 MiB draft KV, zero tap store); held inactive by user decision prioritizing native MTP. | **Absent (Blocked by Single-Seq Hold)**<br>Not plumbed into batched runner. | **Single-Seq Only** *(on-hold)* | Single: [`tp2_budget.h:137`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_budget.h#L137)<br>Batched: [`tp2_backend.cpp`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp) |
| **Adaptive Draft Depth** | **Supported**<br>`MtpAdaptiveController` dynamically adapts draft depth `rk` between `mtp_draft_min_adaptive` and `mtp_draft_max`. | **Absent**<br>Fixed draft depth `k = backend.options().mtp_k`. No adaptive controller logic exists in batched runner. | **Single-Seq Only** *(verified)* | Single: [`tp2_backend.cpp:1165`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1165)<br>Batched: [`tp2_backend.cpp:2601`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L2601) |
| **N-gram Pool Drafting** | **Supported**<br>Shared 24-gram pool (`MtpNgramMod`); rolls candidates when history matches >= `ngram_mod_n_min`. | **Absent**<br>Batched runner completely omits n-gram pool consultation. | **Single-Seq Only** *(verified)* | Single: [`tp2_backend.cpp:1171`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1171)<br>Batched: lines 2555–4324 (0 refs) |
| **Context Lookup Drafting** | **Supported**<br>`find_context_lookup_drafts` extracts recurring token n-grams directly from prompt/history. | **Absent (Rejected)**<br>Explicitly excluded by admission `eligible()` in `tp_engine.cpp:259`; throws in runner if passed. | **Single-Seq Only** *(verified)* | Single: [`tp2_backend.cpp:1753`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1753)<br>Batched: [`tp_engine.cpp:259`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp_engine.cpp#L259) |
| **Prefix Caching & Reuse** | **Supported**<br>LCP match, GDN slot copy, mid-prefill GDN checkpoint restore, KVarN open-page tail snapshot restore. | **Absent (Invalidated)**<br>Invalidates cache on entry (`st.cache_valid = false`); forces full re-prefill for every lane from token 0. | **Single-Seq Only** *(verified)* | Single: [`tp2_backend.cpp:1225`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1225)<br>Batched: [`tp2_backend.cpp:2693`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L2693) |
| **Mid-Prefill GDN Checkpoint** | **Supported**<br>Captures/restores linear attention state within 1 chunk of prompt end (`st.gdn_ckpt_conv/rec`). | **Absent**<br>Frontier reset to -1 (`st.text->set_prefill_rewrite_checkpoint_frontier(-1)`). | **Single-Seq Only** *(verified)* | Single: [`tp2_backend.cpp:1405`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1405)<br>Batched: [`tp2_backend.cpp:2703`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L2703) |
| **Confidence Break (W5)** | **Supported (Rank-Race Fixed @ `6be15a2e` / `e278bd4f`)**<br>In-chain confidence product early exit (`MtpConfidenceBreak conf_break`). Two-barrier consume at round-top closes pending_depth race. **Active Blocker**: Prompts >= 5k exhibit second defect (relaxed read race on shared `conf_break`). Default strictly OFF (`NINFER_MTP_CONF_TAU=0`), pending user flip decision. | **Absent**<br>No confidence-break evaluation or early round shortening in batched loop. | **Single-Seq Only** *(verified; pending user flip)* | Single: [`tp2_backend.cpp:1141`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1141)<br>Batched: lines 2555–4324 (0 refs) |
| **DFlash Backend (second drafter)** | **BUILT, ON HOLD by user decision**<br>The 6-layer drafter mechanism exists end-to-end in the runtime (context, feature feed, rewrite checkpoints, CUDA-graph profiles); the 27B model's config has it switched off (`supported=false`, 0 draft tokens). Would draft up to 15 tokens vs MTP's 7. | **Absent.** | **Absent.** | Single: [`src/targets/qwen3_6/impl/runtime/dflash_context.h`](file:///home/intel/ninfer/repo/src/targets/qwen3_6/impl/runtime/dflash_context.h)<br>Config: [`src/targets/qwen3_6_27b/impl/config.h:73`](file:///home/intel/ninfer/repo/src/targets/qwen3_6_27b/impl/config.h#L73) |
| **Host-KV Safety Net** | **Supported (Landed on Main & Verified @ `25fab77d`)**<br>Host memory park/restore safety net (docs/156) for context extension. Fixed t6 straddle wild-write and i8 scale-plane preservation. 3/3 byte-identical at 45k tokens. | **Supported (Landed on Main & Verified @ `25fab77d`)**<br>Per-lane park/restore safety net integrated into main. CI Cell C4 passed. | **Parity** *(verified)* | Single: [`tools/ops/host_kv_gate_ci.sh`](file:///home/intel/ninfer/repo/tools/ops/host_kv_gate_ci.sh)<br>Batched: [`tools/ops/host_kv_gate_ci.sh`](file:///home/intel/ninfer/repo/tools/ops/host_kv_gate_ci.sh) |
| **Sampling & Temperature** | **Supported**<br>Greedy (`temp<=0`): fused `allreduce_argmax`. Sampled (`temp>0`): allgather local logits + `ops::sample`. | **Supported**<br>Phase S (`cbf613fa`): per-lane `SamplingConfig`, mixed greedy + sampled lanes supported in one batch. | **Parity** *(verified)* | Single: [`tp2_backend.cpp:1841`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1841)<br>Batched: [`tp2_backend.cpp:4146`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L4146) |
| **Penalties (Freq / Presence)** | **Supported**<br>Zeroes `st.token_counts_buf` and passes to `host_cfg.token_counts`. | **Supported**<br>Per-lane `token_counts` allocated at `tc_base + b * 248320` and passed to lane configs. | **Parity** *(verified)* | Single: [`tp2_backend.cpp:1202`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1202)<br>Batched: [`tp2_backend.cpp:3113`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L3113) |
| **Cancellation & Errors** | **Supported (Global)**<br>Single cancellation token `cancellation.requested()`. | **Supported (Per-Lane)**<br>`lane_cancellations` vector; isolated callbacks (one failing callback terminates only its own lane). | **Batched Superior** *(verified)* | Single: [`tp2_backend.cpp:1806`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1806)<br>Batched: [`tp_engine.cpp:364`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp_engine.cpp#L364) |
| **Process Teardown & Clean Fences** | **Supported (Landed @ `48bff27f` & `55a41b81`)**<br>PID-scoped teardown; blanket `pkill -x ninfer-serve` eradicated. Pre-launch fence verifies <=20 MiB clean card state. | **Supported (Landed @ `48bff27f` & `55a41b81`)**<br>PID-scoped process verification and listener checks across all multi-lane test runners. | **Parity** *(verified)* | Single: [`tools/smoke/diag/gpu_guard.sh`](file:///home/intel/ninfer/repo/tools/smoke/diag/gpu_guard.sh)<br>Batched: [`tools/smoke/serve_batched_ci.sh`](file:///home/intel/ninfer/repo/tools/smoke/serve_batched_ci.sh) |
| **Prefill Progress Callback** | **Supported**<br>`req.prefill_progress(cursor, plen)` called per chunk. | **Absent**<br>Progress callback is ignored during per-lane prefill. | **Single-Seq Only** *(verified)* | Single: [`tp2_backend.cpp:1399`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L1399)<br>Batched: lines 2827–2845 (0 calls) |

---

### 3.3 NVFP4 Functionality in Single-Sequence

Answering user and maintainer inquiries regarding NVFP4 KV cache functionality in single-sequence mode:

1. **Append**:
   - **Prefill**: Handled by [`gqa_attention_prefill_fill_nvfp4_kernel`](file:///home/intel/ninfer/repo/src/ops/launcher/gqa_attention_prefill.cu#L129). Fills code planes (U8, D/2 bytes) and scale planes (U8 E4M3, D/16 bytes) with normalized Hadamard H256 rotation on K.
   - **Decode**: Handled by [`gqa_decode_slice7_nvfp4_kernel`](file:///home/intel/ninfer/repo/src/ops/kernel/gqa_decode_slice7_nvfp4.cuh#L199) with `CacheInput::writes_cache=true`. Fused append rotates K and quantizes both K and V into physical cache pages in lockstep with decoding.
2. **Attend**:
   - **Prefill**: Handled by [`gqa_attention_prefill_nvfp4_kernel`](file:///home/intel/ninfer/repo/src/ops/launcher/gqa_attention_prefill.cu#L53). Dequant-staged flash kernel with in-smem Q rotation.
   - **Decode**: Small-T decode route dispatches to `launch_tc_partial_unified_nvfp4` with `MultiBatch=false, Masked=false` ([`gqa_attention_decode.cu:330`](file:///home/intel/ninfer/repo/src/ops/launcher/gqa_attention_decode.cu#L330)), executing on-the-fly E2M1 dequantization and scaled dot-product attention.
3. **Quant / Dequant**:
   - Implemented in [`src/ops/kernel/gqa_attention_kv_quant_nvfp4.cuh`](file:///home/intel/ninfer/repo/src/ops/kernel/gqa_attention_kv_quant_nvfp4.cuh). All unit test golden vectors pass bit-exact in CTests (`ninfer_nvfp4_kv_codec_ref`).
4. **Eviction & Lifecycle**:
   - In single-sequence mode, NVFP4 cache pages remain resident in `st.kv_alloc`. Because dense NVFP4 writes codes directly to paged memory without an uncommitted open-page tail (unlike KVarN), cache lines are immediately stable. Upon new non-matching prompts, `st.text->set_text_kv_base(0)` resets the append cursor to 0, cleanly overwriting prior allocations without page fragmentation.
5. **Conclusion**: NVFP4 is **fully functional, verified, and complete in single-sequence mode**.

---

### 3.4 The Critical Reverse Gap: NVFP4 in Batched Serving

While single-sequence NVFP4 is fully functional, **batched NVFP4 contains an unverified reverse gap**:

1. **Prefill Kernel Lacks Batch Dimension**:
   [`gqa_attention_prefill_nvfp4_kernel`](file:///home/intel/ninfer/repo/src/ops/launcher/gqa_attention_prefill.cu#L51) hardcodes the third grid dimension to `1u`:
   ```cpp
   const dim3 attention_grid(div_up(tokens, kGqaPrefillBr), Geometry::QHeads, 1u);
   ```
   The kernel cannot execute batched prefill across sequences simultaneously. The batched runner only succeeds here because it executes prefill **serially per-lane** ([`tp2_backend.cpp:2786`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp2_backend.cpp#L2786)).
2. **Decode MultiBatch Kernel is Completely Untested on Device**:
   While `gqa_decode_slice7_nvfp4_kernel` includes a `MultiBatch` template parameter ([`gqa_decode_slice7_nvfp4.cuh:66`](file:///home/intel/ninfer/repo/src/ops/kernel/gqa_decode_slice7_nvfp4.cuh#L66)), all test suites in the repository ([`tests/test_nvfp4_attention.cu`](file:///home/intel/ninfer/repo/tests/test_nvfp4_attention.cu)) test **exclusively `batch_size = 1`**. There has never been an on-device multi-lane test of NVFP4 MultiBatch decode.
3. **Cached Batched Route Throws**:
   If any path routes NVFP4 into `gqa_attention_cached_batched`, [`src/ops/wrapper/gqa_attention.cpp:936-940`](file:///home/intel/ninfer/repo/src/ops/wrapper/gqa_attention.cpp#L936-L940) throws an explicit exception:
   ```cpp
   if (cache.dtype != DType::KVARN_K4V2) {
       throw std::invalid_argument(
           "gqa_attention_cached_batched: only KVarN is supported here; bf16/int8 use "
           "gqa_attention (fused-append)");
   }
   ```
4. **Status**: **UNVERIFIED / RED** in batched serving mode.

---

### 3.5 Server & Application Routing Mechanics

Requests are routed between `run_tp2_request` and `run_tp2_requests_batched` in [`src/runtime/tp2/tp_engine.cpp`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp_engine.cpp) according to strict admission logic:

```
                          Incoming Request
                                 |
                                 v
                     [lanes > 1 && !BATCH_DISABLE]
                             /                                  No           Yes
                          /                     Single-Seq (run_tp2_request)   Check mtp_batch_ok (need <= have)
                                             /                                                    False         True
                                          /                                             Single-Seq Fallback     Hold door (batch_window_ms)
                             (Log: need > have)            |
                                                     Filter eligible:
                                                     - !use_lookup
                                                     - (mtp_k > 0) == self_mtp
                                                           |
                                                     [batch.size() > 1]
                                                      /                                                                  No               Yes
                                                    /                                                       Single-Seq Runner            Batched Runner
                                    (run_tp2_request)       (run_tp2_requests_batched)
```

### Specific Routing Boundaries:
- **`--max-concurrency 1`**: Sets `lanes = 1`, making `can_batch = false` ([`tp_engine.cpp:205`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp_engine.cpp#L205)). All requests route to `run_tp2_request`.
- **`NINFER_BATCH_DISABLE=1`**: Environment override forcing 100% single-sequence routing regardless of concurrency settings.
- **GDN Verify Ring Exhaustion**: For MTP requests, ring geometry requires `need = lanes * (k + 1) <= have = 2*lanes + 2k + 1`. If `max_concurrency=4, k=3`, `need = 16 > have = 15`. The server explicitly logs:
  `[tp2] batch admission: batched MTP unavailable at max_concurrency=4 (need=16 > have=15); serving single-sequence`
  and routes to single-sequence execution without failing.
- **Lookup Incompatibility**: Any request specifying `use_lookup = true` is barred from joining a batch ([`tp_engine.cpp:259`](file:///home/intel/ninfer/repo/src/runtime/tp2/tp_engine.cpp#L259)) and executed sequentially.

---

### 3.6 MultiBatch Cost Surface & Footprint Economics (Post-Ledger Calibration)

The physical memory profile was empirically audited on dual RTX 5060 Ti GPUs in [`docs/VRAM_LEDGER.md`](file:///home/intel/ninfer/repo/docs/VRAM_LEDGER.md) (commit `55a41b81`):

| Cost Component | Single-Sequence (`lanes=1`) | Batched Serving (`lanes=2`) | Economic & Operational Impact |
| :--- | :--- | :--- | :--- |
| **KV Cache Capacity per Request** | **100% of KV Pool** (`cap`) | **50% of KV Pool** (`cap / 2`) | Single-seq provides **2× maximum context length** in the same physical VRAM. |
| **Measured VRAM Floor** | **12,998 MiB** (at 40k BF16) | **14,508–14,510 MiB** (at 2×40k BF16) | Lane-2 adds **+1,510 MiB** measured vs +1,536 claimed (honest within 26 MiB). Batched decode adds only +2 MiB. |
| **Refusal Gate Formula (`55a41b81`)** | `measured_floor + ctx + 256 MiB` | `measured_floor + ctx + 256 MiB` | Retires ~1.5 GB phantom reservation tax; unlocks `int8@200k` without operator override flags. |
| **BF16 Physical VRAM @ 68k ctx** | **15,546 MiB** (Fits 16 GB card) | **17,082 MiB** (OOM on 16 GB card) | BF16 Line M requires single-sequence mode to fit 68k context on dual RTX 5060 Ti. |
| **GDN Linear Attention State Slots** | `k + 1` slots (need <= 2k + 3) | `2 * lanes + (2k + 1)` slots | MultiBatch requires sizing GDN state pool for multiple ring geometries. |
| **KVarN Workspace Allocations** | 1 workspace tile set | `lanes` workspaces (`kvarn_lane_ws`) | MultiBatch allocates `lanes` separate persistent workspace structs and device pointer arrays. |
| **Per-Round Staging Arena** | Sized for `[*, 1]` | Sized for `[*, N]` (`mb_vhid`, `mb_vlog`, etc.) | MultiBatch inflates staging memory consumption proportionally to concurrency. |
| **LITH Rule-14 Invariant** | Pinned: 2048 B/tok/layer (BF16), 8976 B/tok (k4v2) | Pinned: 2048 B/tok/layer (BF16), 8976 B/tok (k4v2) | Any lane scaling must strictly maintain `kv_unit_bytes() == kv_bytes_per_token()`. |

---


---

## 4. What Changed Since Last Audit (delta vs `9f453037`)

1. **Host-KV Safety Net landed & verified (`25fab77d`)**: t6 wild-write (a park/restore
   memory-straddle bug) and the missing i8 dequant-scale planes fixed; 45k×3-session
   park/restore proven byte-identical with engagement; CI cell C4 green on main.
2. **VRAM Refusal Gate recalibration (`55a41b81`)**: refusal now charges the MEASURED
   floor + context + 256 MiB margin (retiring ~1.5 GB of never-touched reservations);
   int8@200k launches without override (measured 15,069 MiB, 780 MiB spare, 199.8k
   tokens @ 59.0 tok/s); fences tightened (50/20 MiB baselines).
3. **pkill era terminated (`48bff27f`)**: all 9 system-wide name-kill sites across 5
   files replaced with PID-scoped/port-scoped shutdowns (live smoke: target killed,
   neighbor survived).
4. **W5 rank-race fix folded (`6be15a2e` / `e278bd4f`)**: two-barrier consume + atomic
   break closed the two-rank desync (matrix 4/4 clean, byte-identical). W5 default
   stays OFF — **a SECOND defect hangs very long prompts with W5 on** (below), and the
   activation decision is the user's.
5. **CI Exit Contract Guard landed (`b0e9a8f2`)**: static linter (28/28) as the CI
   prerun gate.

---

## 5. Engineering Roadmap: Landed / In Progress / Queued / Blocked / On Hold / Horizon

*Every item: status, what it unlocks for the user, what it depends on. Nothing dropped —
this consolidates docs/56, docs/59, docs/151, docs/152, the VRAM ledger (docs/VRAM_LEDGER.md)
and the coordinator ledger.*

**BLOCKED (active defect):**

| Item | Status | What it unlocks | Dependencies |
| :--- | :--- | :--- | :--- |
| **W5 Activation — SECOND defect (long-prefill hang)** | **BLOCKED — being dug now** (WO §18, instrument-don't-fix) | Turning W5 on for real: 10–15% decode speedup on confident text, ON by default. | The first fix (two-barrier + atomic break) is proven necessary but NOT sufficient: with W5 on, prompts ≳5k tokens hang the server at decode start (dtype-independent; the 2k-token validation matrix was too narrow to catch it). Mechanism candidate named from code-read (the two ranks read the shared break-decision flag unsynchronized → they disagree on when to stop guessing → one rank waits forever); discriminator trace running. W5 stays OFF and `e278bd4f` stays unpushed until this closes. User decision on the flip comes after. |

**LANDED (done, verified):**

| Item | Status | What it unlocked | Notes |
| :--- | :--- | :--- | :--- |
| **VRAM Refusal Gate recalibration** | **DONE** (`55a41b81`) | 200k-token contexts on 16 GB cards without manual override; retired ~1.5 GB of imaginary requirements. | Measured ledger (docs/VRAM_LEDGER.md); user LITH sign-off. |
| **Host-KV Safety Net** | **DONE** (`25fab77d`) | 45k×3-session contexts with seamless GPU↔RAM parking; no more OOM aborts on long sessions. | t6 wild-write + i8 scale-plane defects fixed and validated (3/3 byte-identical). |
| **Radiance W0–W4a launch-gap reduction** | **DONE** (`9c665401`) | 15.3% fewer multi-card dispatch gaps; zero-copy GDN kernels. | PCIe peer sync, folded argmax. |
| **W5 rank-race fix (first defect)** | **DONE** (`e278bd4f`, local) | Closed the two-rank desync hang for short/medium prompts; matrix 4/4 clean, byte-identical. | Local-only until the second defect closes (see BLOCKED). |
| **PID-scoped process hygiene** | **DONE** (`48bff27f`) | No script can kill another project's server anymore; 9 sites swept, live-smoked. | Replaced every system-wide name kill. |
| **CI Exit Contract Guard** | **DONE** (`b0e9a8f2`) | No more 90-minute CI runs wasted by mis-declared test results. | Static linter (28/28) as the CI prerun gate. |

**IN PROGRESS:**

| Item | Status | What it unlocks | Dependencies |
| :--- | :--- | :--- | :--- |
| **Single-Seq Parity Port (WO #1)** | **IN PROGRESS** (A1, branch `wo/single-parity` — resumed after the VRAM ledger landed) | Carries single-request-only features (draft-skip, coverage instrumentation) into shared code so both serving modes support the same surface. | VRAM ledger (landed `55a41b81`). |

**QUEUED (scheduled, in order):**

| Item | Status | What it unlocks | Dependencies |
| :--- | :--- | :--- | :--- |
| **Inverse-Gap Lane (batched MTP parity)** | **QUEUED** | Brings the single-request-only speed features (adaptive guessing depth, the shared n-gram pool, context lookup, confidence break) into multi-conversation serving. | WO #1 completion; the W5 blocker closing; W5 flip decision. |
| **DFlash Path B + DFlash2** | **QUEUED (NEXT BIG FEATURE per docs/56/59)** | Turns the built-but-parked DFlash drafter on for the 27B model, then upgrades to the DFlash2 block drafter (~34% more guessed words per round than MTP measured on reference hardware). | The 1.92B drafter's official trained files + reference implementation are PUBLIC (llama.cpp PR #27342); needs VRAM headroom (the recalibrated gate helps). Config today: `supported=false`, 0 draft words. |
| **PPL / KLD Quality Tooling** | **QUEUED** | A permanent quality ruler: measures how much the compressed-memory tiers (and future changes) actually hurt answer quality, not just speed. | The `top_logprobs` endpoint (small serve-layer work); scope in docs/65. |
| **All-Q4 Model Artifact Validation** | **QUEUED** | A smaller model file (18.2 GB → ~13.5 GB) validated for quality — more VRAM headroom on the same cards. | The PPL/KLD tooling (above) + the quality battery. |
| **Host-RAM Ledger** | **QUEUED** | Exact measured numbers for system-RAM use (the host-KV park holds up to 14 GiB of pinned host memory) so long sessions can't exhaust the computer's RAM either. | Host-KV safety net (landed). |
| **Rank-Race Class Audit** | **QUEUED** | A systematic sweep for the same unsynchronized-shared-state pattern that caused both W5 hangs (batched runner `tp2_backend.cpp:3853` / `observe()` sites), so the bug class dies, not just the instance. | W5 §18 dig conclusions. |

**ON HOLD (user decision):**

| Item | Status | What it unlocks | Notes |
| :--- | :--- | :--- | :--- |
| **DFlash (in-tree drafter)** | **BUILT, ON HOLD by user decision** | A second, larger guesser (6 layers, up to 15 words ahead) as an alternative to MTP. | Mechanism complete in the runtime (context, feature feed, CUDA-graph profiles); the 27B config has it switched off (`supported=false`, 0 draft words). Reason for hold: user decision. |
| **KV Defrag & Dynamic Compaction** | **DISCUSSION (direction stage, no work order)** | Runtime cleanup of fragmented conversation memory without restarting or re-reading the document. | User wants to discuss the approach first; WO recalled from A1. |

**HORIZON (bigger futures):**

| Item | Status | What it unlocks | Notes |
| :--- | :--- | :--- | :--- |
| **KVarN D-18/D-19 wall fix** | **CLOSED (2026-08-26, `acd6f791`)** | The long-document reading-speed collapse (724 → 4 words/s past the shadow capacity) is FIXED: over-capacity short-batch attention routes through the proven tensor-core split-K kernel — 31.5k tokens: 39.4→62.8 words/s; 40k: 35.1→60.6; T19 (80k prefill ≥450 tok/s) achieved; yesterday's decode-guard re-proved 160k–250k prefill green. The staged-shadow premise was later removed entirely (docs/83). | docs/50 D-18/D-19 entries; docs/66 spec. |
| **Single-Card Support (single RTX 5090 / PRO 5000)** | **QUEUED (scope in docs/62)** | Runs the model on ONE newer card (32 GB) without the two-card team. | Build is card-generation-specific (sm_120a vs sm_100) — port + re-baseline, not a flip; needs the compressed-memory tiers first. |
| **TP4+ (four or more cards)** | **HORIZON** | Even bigger models on consumer hardware (5090×4). | Weight-sharding math is already card-count-generic; the two-card handshake generalizes to N cards — scope as its own work order. |
| **v340l lead-up** | **HORIZON** | This whole project builds the skills and patterns for the next machine (v340l). | Docs preserved in `~/comfy_templates/v340l_optimization/`. |

---

## 6. Verification Timestamps & Confidence

| Column | Verification Status | Last Verified | Basis |
| :--- | :--- | :--- | :--- |
| **US Single-Seq** | **Verified** (code + hardware tests) | 2026-09-08 | `e278bd4f` on dual RTX 5060 Ti; int8@200k run complete (`dg200k_gate_decode_20260908_071914.json`). |
| **US Batched** | **Verified** (BF16, INT8, KVarN); **Unverified** (NVFP4) | 2026-09-08 | `e278bd4f`; `results/batched_serving_20260907_123832.json`. |
| **ORIGINAL NINFER** | **Verified** (code inspection) | 2026-09-07 | `1ae30d91` inspected directly. |
| **vLLM** | **Knowledge / semi-verified** | 2026-09-07 | Public architecture docs; per-cell notes distinguish verified core vs knowledge. |
| **llama.cpp** | **Verified** (local checkouts) | 2026-09-07 | `/home/intel/llama.cpp-combined` and `/home/intel/complete-fix-llama-cpp` inspected. |

---

## 4. SPEC/DECODING PROVENANCE LEDGER (added 2026-09-09, user order; git-verified only)

Every origin below is backed by a git-greppable reference (tree comment, docs header,
manifest field, or commit). Where provenance is genuinely unknown the row says
**unattributed** — a confident wrong origin is worse than an honest blank (lesson,
W5/Radiance gap, 2026-09-09). Dates: "verified" = when THIS repo confirmed the origin
against evidence, not when upstream shipped.

| Feature | Origin (verified) | Origin-verified | Landed | DEFAULT STATE | Habitat / notes |
|---|---|---|---|---|---|
| MTP base drafter | target-model native (Qwen3.8 MTP head); engine integration unattributed (internal) | — | early core (pre-docs/69) | ON under `--spec mtp` | — |
| Adaptive MTP draft depth | llama.cpp **PR #27210** (tree refs incl. mtp_adaptive.h header) | 2026-08-26 | 7d78bb55 (2026-08-26) | opt-in `--mtp-adaptive` (battery runs ON) | docs/69 WO = landing record |
| ngram-mod draft pool | llama.cpp **PR #19164** ("ngram-mod", MIT; pool table verbatim per src comment) | 2026-08-26 | ea852a01 seed loader (2026-08-26); docs/69 step 3a | opt-in `--mtp-ngram-mod` | — |
| W5 confidence-product early break | **vLLM Radiance project** = radiance-vllm-mxfp4, doc-44 §4.2 "Dynamic draft depth (MTP)" (codeberg ggz14/radiance-vllm-mxfp4; τ=0.35, on-device same-round, +5.3% mostly-at-concurrency) | 2026-09-09 (user question restored lineage) | first code 92e6ebd6 (radiance lane); dormant merge into wo/trunk-integration e143f0be (2026-09-09) | **OFF** (`NINFER_MTP_CONF_TAU` unset; merge = "code in, dark until proof") | MTP habitat CLOSED arithmetically-infeasible with receipts (chain-share 5.8% envelope d1ae9b75; Row P −7.7/−11.1% d62aaa6d). DFlash2 habitat BLOCKED upstream: §22.18 lane-blind fuse at round-0 (92d274ce); inline dual-layout discriminant armed 1ef9f740. See docs/151 §20 (b91ffbe5). |
| DFlash2 block drafter | upstream vLLM **PR #27342** (MERGED per docs/151 §0 network evidence, head z-lab:dflash2 2f3923bc81) + artifact **z-lab/Qwen3.8-27B-DFlash2-GGUF** (sidecar manifest `source_gguf` field; local: dflash2_w8g32 = Q4_K_M route) | 2026-09-05 (docs/151 §0) | wo/dflash2-scope lineage; merged into trunk-integration 8599ef55 (2026-09-09) | opt-in `--spec dflash2` (requires `--dflash2-sidecar`, k4v2 tier, N≥2 multibatch-native) | 2026-09-09 21:00Z state: boot WORKS — the §22.18 'lane-blind fuse' was a transposed-slice artifact in the EVIDENCE GATES (2b-ii + 2b-iii, fixed 4c6c4ac6/c19358c6 on wo/dflash2-w5-habitat; fuse+stack always lane-distinct). Live ghost = ACCEPT 0.003/draft with the 09-06 Q4_K_M sidecar; the provenance pair test awaits A1's driver re-emit (v2_correct manifest currently carries source-HF schema incl. target_layers=5 vs taps [6,20,34,48,62] — loader correctly rejects; weight-tap semantics suspect, manifest-only patching would mask it). Draft ceiling 5 here (T=6 TokenTile, §22.12.6) |
| FirstMissHistogram / verify-depth logging | unattributed (internal diagnostic) | — | 699ee6cd (2026-09-06) | OFF (`NINFER_VERIFY_DEPTH_LOG`) | the measurement that made W5's economics visible |
| Prefix-GDN checkpoint buffer | semantics validated against llama.cpp **PR #24891** checklist (docs/58 §2); upstream invalidation-mode tracking **PR #25819** | 2026-09 (docs/58 refs in tree) | pre-dates ledger window | ON | W5's 22.59 suspect-list neighbor |
| Host-KV safety net | unattributed (internal design, docs/156) | — | 6e558163 (2026-09-06) | operator-sized via `--host-kv-mib` / `--host-kv-min-pages` | separate feature; do not conflate with W5 |

### Branch ownership (the 19:56Z divergence is why this exists)
Single-writer rule per branch; force-push only by the owner WITH a backup ref pushed first.
- `main` — coordinator cut; doc/CI commits flow here (this table included).
- `wo/trunk-integration` — **A1 sole owner** (19:56Z union-merge in flight; agent2's 814ae97e..209da59b ride his merge, never rebased-over).
- `wo/radiance-launch-gap` — agent2 (rebased onto e479a06a @ 083e3776; granular history preserved at `backup/radiance-pre-rebase-3dc8053c`).
- `wo/dflash2-w5-habitat` — agent2 for D2TIMING/fuse-discriminant instrumentation; §22.18 FIX ownership pending coordinator assignment (A1 assessed).
- `ci/full-20260910` — agent2 chain (#26/#137 closes); frozen-for-content post-rebase, picks already on ci tip cb9f5d82.

### Footer: instrumentation-as-argument (2026-09-09)
The D2FUSEDIAG dual-layout discriminant (agent2, 1ef9f740) resolved its own blocker-class
in one run — the §22.18 tripwire it was built to adjudicate turned out to be the bug, and
the printed interpA-vs-interpB hash pair settled it on both ranks without archaeology.
Standing pattern for evidence gates: any assertion about CONTENT must name the LAYOUT it
reads through — a gate that hashes the wrong slice certifies nothing (the 22.18-class
self-trip; same family as every vacuous-green of the day).
