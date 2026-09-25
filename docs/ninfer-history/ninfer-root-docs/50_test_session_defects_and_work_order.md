# 50 — Test Session Defects & Work Order (2026-08-23)

**Status:** CURRENT

## 1. Context

First full user test session of the TP2 `ninfer-serve` stack
(2× 5060 Ti 16 GB, Qwen3.8-27B Q4, MTP k=3, q8 KV 100k cap / 80k context, port 8091).

Last updated: 2026-08-26 — D-18 FIXED via D-19 wall fix `acd6f791` (62.8/60.6 t/s, full CI 20260826_095551 green); D-19 CLOSED B2 hazard class `c289dd06`; D-20 OPEN (VRAM pre-check under-estimates prefix-capacity cost); **D-21 FIXED 2026-08-28 (`84ba2c8b` anchor off-by-one + `1d3a0533` gate route; H5 k=3 + short prompt byte-identical; residual width-dependent attention precision accepted Option C; pending temp>0 re-run / full-width H5 / full CI — `results/d21_mtp_verify_anchor_fix.md`)**. Prior 2026-08-24: D-14 FIXED (KVarN T16 live append jump on `wo/kvarn-live`, 250k must-pass T16,T1,T2,T3,T5,T8,T10,T11 green, docs/54 §9 P2c step 4); D-11 CLOSED (`setsid` mitigation final per user); new: D-15/D-16 KVarN perf defects from wo/kvarn-live review, D-17 prefix-restore finding from docs/63.

User sent, in **separate conversations**: only "hi", "hello", "what is your name".
Observed model outputs included content **never requested**: a fake file read
(`server.py`), a full Python CSV-filter script, a `</function_results>` repetition
loop, and — for "what is your name" — a verbatim replay of the CSV script from an
earlier unrelated exchange.

Reference launch: see `LAUNCH.md`.

## 2. Defect register

| ID | Sev | Owner | Status | Summary |
|----|-----|-------|--------|---------|
| D-01 | P0 | GPU | **FIXED** (712c0fd5) | Cross-request content leakage. Root cause: partial prefix hit restored GDN from end of longer cached prompt; work arena not scoped per request. Fix: skip GDN restore when `prefix_len < cached_tokens.size()`; `work.scope()`; alias mtp_ph/mtp_mh to resident buffers. Verified: T1/T3/T10 PASS, `det2.py` n2==n3. |
| D-02 | P1 | REMOTE | **MERGED** (`v1-integrate` 2026-08-23; code on `wo/auto-kv`) | `--kv-capacity auto` + borderline configs pass the TP2 preflight then `cudaMalloc` OOM at startup. Root cause: preflight under-estimated per-rank VRAM — omitted `prefill_dummy {5120,N}`, the MTP KV pool (1/16 of text KV, same page count), the decoder-fixed portion (GDN/tables), and any headroom. (TP2 ring is never sized from free VRAM — ring = max_context.) Fix: calibrated pure budget model `src/runtime/tp2/tp2_budget.h` (I8 80k fits with ~240 MiB margin; 100k/110k I8 and 80k BF16 rejected; `auto` reports max-feasible ≈ 86k I8 and rejects a max_context that doesn't fit) wired into `tp_engine.cpp` preflight + unit test `tests/test_tp2_budget.cpp` (green locally, registered in CTest). **GPU agent verification:** battery T8 + startup OOM check (explicit 100k I8 must be cleanly rejected with "max feasible is N"; `auto` at 80k I8 must start). |
| D-03 | P1 | GPU | **FIXED** (712c0fd5) | No admission control for prompt+max_tokens > max_context. Fix: reject in `tp_engine.cpp` submit (`ContextLengthExceeded` → 400). Verified: T8(b) PASS. |
| D-04 | P1 | GPU | **FIXED** (this commit) | Streaming crash `json.exception.type_error.316` — incomplete UTF-8 at SSE chunk boundary. Root cause: TP2 streaming path published raw token bytes without UTF-8 coalescing; non-tool streams also dropped terminal `remaining` content. Fix: `Utf8StreamBuffer` in `generation_service.cpp` + `tp_engine.cpp`; flush all `remaining` in `http_server.cpp`. Verified: T4 PASS (stream, 👋+你好, thinking off). |
| D-05 | P0 | — | **FIXED** (5a270a85) | Arena over-commit: `prefill_dummy {5120, max_context}` allocated from a `WorkspaceArena` that didn't account for it → bad_alloc for max_context > 16k. Fixed in `src/runtime/tp2/tp2_backend.cpp` (`prefill_dummy_bytes` added to arena size). |
| D-06 | P0 | — | **FIXED** (5a270a85) | A-3 long-prefill crash: MTP module prefill not chunked. Battery PASS, byte-identical vs greedy. |
| D-07 | P2 | REMOTE | **IMPLEMENTED** (Phase P2 complete; D-07) | 200k unreachable: per-rank 37 KB/token (q8 KV 16.5 + `cached_ph` 10.24 + `prefill_dummy` 10.24; per-request `mtp_ph`/`mtp_mh` already aliased in 712c0fd5). Implemented: prefix-cache cap P=16k (`--prefix-cache-capacity P`) + interleaved chunked prefill (`ph_scratch` staging scratch) for `plen > P` + updated budget model in `tp2_budget.h` + T13 battery test. |
| D-08 | P3 | GPU | OPEN | `--no-prefix-reuse` does not actually free the buffers (still allocated) |
| D-09 | P2 | REMOTE | **CLOSED — machinery pre-existing** (verified 2026-08-23, `wo/tool-calls`) | OpenAI `tool_calls` emission already exists end-to-end: `parse_qwen_tool_call_output` + `ToolCallStreamFilter` (`src/serve/tool_call_parser.{h,cpp}`) wired `generation_service.cpp` → `http_server.cpp` (non-stream + SSE `tool_calls` chunk, `finish_reason=tool_calls`) → `openai_schema` (`make_chat_completion_tool_response`, `make_chat_chunk_tool_calls`). Native content is kept as fallback (malformed blocks fall back verbatim). Verified 2026-08-23: `tests/test_tool_call_parser.cpp` compiles + passes standalone (single/multi call, JSON params, malformed fallback, name limits, incremental stream filter). **The T6 failure (11:16 run) is NOT a missing tool_calls field** — it is the separate early-stop failure mode (`content` = `"We\n\n</think>"`, model stops mid-thought, raw `</think>` leaked past `derive_think_parts`); tracked under §7.1 control, owner GPU. |
| D-10 | P1 | GPU | **FIXED** (712c0fd5) | Large prompt (~24k tokens) → bad_alloc: per-request `{5120,plen}` mtp_ph/mtp_mh exhausted persistent arena. Fix: alias mtp_ph→cached_ph, mtp_mh→prefill_dummy; mtp_pos/ids in staging. Verified: T8(a) PASS (~12k-token prompt). |
| D-11 | P3 | OPS | **CLOSED — MITIGATED, NO ACTION** (2026-08-24) | Server process dies silently (no crash, no log line) when the shell session that launched it goes away. Observed twice this morning. **Always launch via `setsid`** (see §5) or it will vanish between tool calls/sessions. User decision 2026-08-24: keep the `setsid` mitigation, no external process management (no systemd auto-restart). |
| D-12 | P1 | GPU | **FIXED** (4e85dcd2) | Long-prefill decode early stop on TP2: prompt ≳580 tokens → `gen=2`, `finish=stop_token`, answer prefix `The`/`We`/`GL` + im_end; needle tests fail. Repro without MTP (`speculative=off`). Shorter prompts (≈436 tok) pass. §7.1 control: identical on fresh server → not cross-request state. Fix: chunked long-prefill prefill + `OutputSession` stop handling in `tp2_backend`/`tp_engine`. Verified 2026-08-23: T9 PASS (168.8s) on live 80k/I8/MTP server. |
| D-13 | P1 | GPU | **LAYER 1 VERIFIED** (ce7077c5; T14 PASS 2026-08-23: 90,439-tok prompt served on 80k/100k server, generated as requested; 125k-tok prompt clean 400 naming kv capacity; T1/T2/T3/T5/T8 regression green) | Context overshoot hard-rejected: prompt > `max_context` → 400 even when KV pool has room; root cause: `--kv-capacity` did not size the ring (always `max_context + mtp_k + 4`). Pi auto-compaction at 83.7k/80k failed → session stuck; sessions routinely run over the window (observed 102–108% of 200k). Fix per docs/55: **layer 1** tolerance zone — ring sized to `max(max_context, kv_capacity)`, admission up to capacity, per-request effective limit; **layer 2** IMPLEMENTED 2026-08-23: `POST /v1/compact` recursive chunked summarization (chunk = 0.65·window from oldest end, measured chars/token ratio, convergence guard) — `src/serve/compaction.{h,cpp}` + unit test `ninfer_compaction_test`; live T15 pending restart. |
| D-14 | P0 | GPU (`wo/kvarn-live`) | **FIXED** (2026-08-24) | KVarN live: every request failed with `gqa_kv_append_kvarn: append jumped pages…` at 250k (`--kv-dtype kvarn_k4v2`). Gate: battery **T16**. Root causes: stale MTP/text tile across re-prefill; MTP verify page-cross without hydrate; unaligned prefix reuse after lossy commit; attention inferring packed pages from envelope. Fix: `gqa_kvarn_reset_inflight` / rewind; hydrate via `gqa_kvarn_prepare_page_for_append`; unaligned prefix → re-prefill from 0; `committed_pages` for attend; skip worker MTP redo after `prepare_mtp`. Verified: units + T16 + must-pass at 250k (docs/54 §9 P2c step 4). |
| D-18 | P1 | GPU | **FIXED** (D-19 wall fix, `acd6f791`, verified 62.8/60.6 t/s + full CI 20260826_095551) | Prefill/decode collapse beyond staged-shadow capacity: shadow covers ~30.7k tokens (481 pages @ 1 GiB budget); past it, attention falls back to the step-3 fused kernel which re-dequants every packed page per query token (O(T×P)) — measured 724→652 tok/s up to 29.7k, then **12.6 → 4.0 tok/s** at 31–32k. Decode beyond capacity is affected too (fused read every step). This blocks KVarN's primary use case (long context). Interim budget bump (1→2.4 GiB) was tried and **REVERTED** (`3ed1320d`→revert): it pushed the 250k config past the real VRAM ceiling (startup OOM); the ~16,310 usable estimate was too optimistic — do not raise the shadow budget without measuring free VRAM at startup. **Fix (D-19 wall fix, `acd6f791`):** route over-shadow small-T (T≤6) attention through the proven TC split-K small-T kernel instead of the chunk-prefill flash whose grid collapses to 12 CTAs at T=4 — live 31.5k 39.4→62.8 tok/s, 40k 35.1→60.6 tok/s; MTP acceptance unchanged (85.7%). Full CI 20260826_095551 green. Gate: T19 (80k-token prefill ≥450 tok/s avg). |
| D-19 | P1 | GPU | **CLOSED — B2 hazard class** (`c289dd06`, merged to main `fe3d8857`) | Semantically-neutral edits to the legacy materialize kernel body flip MTP acceptance on prefix-restored requests (same hazard class as docs/70 m2 arena incident). Pre-B2 prefix-restore decode 60.1→60.2 t/s @81.0% acceptance; the B2 refactor silently regressed it to 55.7 t/s @75.6% (same output md5 `21fd`). Fix: default tier dispatches to the **byte-identical** pre-B2 kernel; a separate width-generic `_w` variant serves non-default tiers. **Process response:** `tools/build_and_serve.sh` (git hash + sha256 + mtime provenance, `--kill-only` verified-shutdown) added to prevent stale-binary A/B poisoning. Lesson documented in docs/77 G-4: do not rewrite kernel bodies for default-tier paths — append a new kernel instead. |
| D-20 | P3 | GPU | **OPEN** (found during docs/74 merge review 2026-08-26) | The TP2 VRAM pre-check under-estimates prefix-capacity cost: a 98k context OOM'd at startup **after** the pre-check reported PASS, while 48k was the working max. The budget model omits the prefix-reuse working buffers (GDN checkpoint, open-page tails) needed when the prefix cache is large relative to max_context. Do not trust a pre-check PASS for prefix-reuse-heavy configs; measure free VRAM at startup (see the D-18 revert lesson, `3ed1320d`). |
| D-21 | P2 | GPU | **FIXED (2026-08-28, `84ba2c8b` anchor + `1d3a0533` gate route; pending temp>0 re-run, full-width H5 matrix, full CI — server items in `results/d21_mtp_verify_anchor_fix.md`)** | Root cause: round-0 MTP verify **anchor off-by-one** (`cur_F_mtp = plen-1` → `plen` in `tp2_backend.cpp`) wrote t0's KV at the prompt's last position, shifting the whole verify batch one position early — the first-token signature below; plus an independent GDN gate route split by token count (fixed to `MmaUnsplit`). Verified byte-identical MTP-vs-plain: H5 1024-token case at k=3 (`tools/ops/h5_retest_mtp_long.sh`), short-prompt repro, 6-prompt battery at k=2. Remaining: separate width-dependent batched-attention precision tolerance, accepted Option C (plan owner 2026-08-28) — not this defect. Full record: `results/d21_mtp_verify_anchor_fix.md`. (Original signature, superseded by the fix:) MTP chunked-prefill diverges from plain decode at the **first generated token** for a prompt of **exactly 1024 tokens** (= 2 full 512-token chunks = 16×64 pages), on a **non-degenerate** space-joined prompt. Both paths are fully deterministic (plain A/B exact; MTP A/B exact). Clean bracketing (tools/ops/h5_retest_mtp_long.sh, clean prompts): 250/509/655/751/844/937/987/1005/1015 tokens ALL MATCH (MTP==Plain); **only 1024 diverges** (plain generates `astronomer`, MTP `farmer` at word 814). The original `mtp_long_ok` gate (LONGPROMPT2 = 128x "The capital of France…" = 1793 tok, strict `==`) was triaged as a *degenerate near-tie flaky gate* — **that triage is now SUPERSEDED**: the real signature is a deterministic first-token divergence at an exact 1024/chunk-aligned boundary. Significance: chunked MTP prefill produces a subtly different KV/state than the plain path at the exact lookahead-BS/2-chunk boundary; plain vs MTP diverge only for the full-2-chunk case. Fix is in the MTP chunked-prefill path (`tp2_backend.cpp` `mtp_forward_batch` / KV commit) or a gate relax to compare the generated continuation (not the echo) with an entropy/prefix-aware check. Tracked as the MTP-path risk for docs/74 C6. |
| D-22 | P1 | GPU | **FIXED (2026-08-28, `6ad60d42`)** | MTP sampling-acceptance collapse (llama.cpp >70% parity vs 9–16% measured; 08-21 known-good 46–47%). Root cause: the MTP **accept** ran with `token_domain=124160` (per-rank vocab shard, `st.verify_logits`) while MTP drafts are **GLOBAL** tokenizer ids. Cross-half drafts (id ≥ 124160) were never found → `pd=0` → **permanently rejected** (DRAFT_OUTSIDE_LOCAL, ~2.9%); and reject-corrections were drawn from the impovеrished per-rank half, so a rejected draft was corrected to an **off-distribution token → garbage cascade** (the MTP head stopped tracking → acceptance collapse). Fix: `allgather_local_bf16` the two rank shards into `verify_logits_full[248320,k+1,1]` (mirroring the plain temp>0 path, S3 note) and run the accept with `token_domain=248320`. Result: seed-42 temp1 acceptance **16.1%→39.6%**; `verify_mtp_sampling.sh` **1.1%→40.6% (PASS, 40% floor)**; MTP output **garbage→coherent**; greedy **80.4%** (A2 identity MTP==plain OK); `samp_s42`/`samp_s43` det+div OK; doc-41 TC-P leak gate OK; full CI `20260828_224030` **PASS** (verify 0 / serve 0 / correctness 0). Residual: r0-col1 `p(argmax)~0.2165` (the accepted Option-C batched-vs-single attention precision tolerance) — limits high-entropy trajectories (seed-dependent 25–67%) but no longer causes the collapse. Full record: `results/d22_accept_instrumentation_findings.md`. Also fixed D-23/S3 (serve seed divergence) — `9a33b896`: S3 compares effective text (`reasoning_content`) since serve defaults to thinking=on with empty `content`. |
| D-17 | P3 | GPU | **OPEN (found by docs/63 audit 2026-08-24; renumbered from D-14 — ID collision with the T16 bug)** | Aborted request + subsequent full-match prefix restore can carry stale KVarN workspace state: request A decodes mid-page (partial tile resident), client disconnects; request B fully matches A's cached prompt → restore path appends from `plen` without resetting the workspace → possible "jumped pages" throw or a commit quantizing a tile with foreign slots. Re-prefill path is safe (reset added 2026-08-24); only full-restore exposed, KVarN storage only. Fix needs a live repro (test-first policy) — deferred with the KVarN work order (docs/61). |
| D-15 | P2 | GPU | **FIXED + INDEPENDENTLY VERIFIED** (2026-08-24, `dca3ca7f`, merged to main `6b535c76`) | KVarN prefill was O(T×P): fused kernel re-dequanted every packed page per query. T8 was ~111→22 tok/s. Fix: ~1 GiB bf16 staged shadow (481 pages), stage-on-commit + prefix stage-from-codes, attend via existing BF16 GQA kernels. Verified: unit bit-exact; T8 curve 678→720 tok/s; ~12k prefill 722 tok/s; must-pass green; VRAM 14,635 MiB/rank. **Main-side verification (2026-08-24): fast battery on the MAIN binary at 250k KVarN — 11 passed / 0 failed (T1,T2,T3,T5,T8,T10,T11,T15,T16,T17,T18); cold prefill 683–728 tok/s, cache-hit restore 23.5–24.2k tok/s.** |
| D-16 | P2 | GPU | **FIXED + INDEPENDENTLY VERIFIED** (2026-08-24, `523e54b4`, merged to main `6b535c76`) | KVarN prefix reuse discarded unless `prefix_len % 64 == 0` (interim: unaligned → full re-prefill). Multi-turn ~94% unaligned → prefix cache off (T17). Fix: snapshot open-page bf16 tails + MTP seed at prefix-cache save (end of prefill); restore on full hit before decode can commit the open page lossily. Verified: T17 PASS; T2/T16 green. **Main-side verification: `test_prefix_snapshot_roundtrip` (unit, bit-exact vs direct-append control) + new battery T18 (restore equivalence — greedy outputs bit-identical cold vs warm for BOTH restore shapes: full hit and delta-after-restore; T17 checks speed only). All green on the main binary at 250k KVarN (`6b535c76`).** |

**Owner key:** `GPU` = agent with the live server (main worktree `/home/intel/ninfer/repo`).
`REMOTE` = code-only agent (worktree `/home/intel/ninfer/repo-remote`, branch
`wo/remote-ports`; its full brief is **`docs/51_remote_agent_work_order.md`**).
New defects MUST get an owner when registered.

### D-01 detail (the one users hit)

**Findings after the test battery (2026-08-23, fresh server, zero user involvement):**

- **T3 (prefix divergence) FAILS on a fresh server** — cold vs warm outputs differ (thinking text drift).
- **Cross-request non-determinism reproduced in 3 requests** (`/tmp/det2.py`, any server):
  same prompt twice consecutively → **bit-identical**; same prompt after one
  intervening decoy request → **diverges at ~token 40** (inside the thinking
  phase). Both with and without `seed=42`. So: identical-prompt warm restore is
  bit-exact; the bug is **residual state from an intervening request** — suspect
  slot zeroing, MTP draft/verify columns (slots 1..3), GDN conv/recurrent state
  in slot 0 after the decoy, one-shot-allreduce epoch flags, or sampling token
  buffers. Not a general nondeterminism (consecutive runs are exact).
- **Semantic bleed observed**: in T9 (a no-tools, no-system conversation), the
  model spontaneously talked about using the `read_file` tool — the exact tool
  name used in T6, which had just run. (Model-habit vs state bleed: unproven,
  but consistent with D-01.)
- User's original repro (below) remains the severe case: full content replay.

Evidence — `/tmp/serve_100k.log`, 07:51–07:55:

- req 2 (07:51:38): msgs=4, prompt=4139, gen=335 → fake file-read + repetition loop
- req 4 (07:53:30): msgs=7, prompt=4726, gen=769, `finish=cancelled client disconnected` → CSV script
- req 5 (07:54:32): msgs=2, prompt=4125 ("what is your name"), **`prefix hit: 4117 tokens (slot 7)`** from req 4 → generated the CSV script again (972 tokens, clean stop)
- req 5's prompt = system (~4117) + 8 tokens; 4117-token match = shared pi system prompt. If the restore is *correct*, the model's context is exactly system + "what is your name" — it should not produce the CSV script.

Candidate root causes (ranked, all to be verified):

1. **Prefix restore restores the wrong state** — `copy_slot(cache_slot→0)` + `cached_ph → mtp_ph` restore at the matched boundary is not bit-exact (e.g., slot state is end-of-previous-request, not boundary state).
2. **Attention KV for prefix positions is stale** — new request's block table maps old pages beyond the matched prefix, so the model attends to the previous conversation.
3. **MTP hidden-state restore offset** — `mtp_ph` restore misaligned vs `set_text_kv_base(prefix_len)`.
4. **Client-side history replay** (cheapest to rule out first) — pi or the web client resending session history the user didn't send. Note the anomaly: req 2 has `msgs=4` for what the user believes was a first "hi" in a fresh conversation.

Fix rule: **no fix accepted without T3 (below) passing.** ✓ Verified 2026-08-23 (712c0fd5).

**Root cause (confirmed):** When `prefix_len < cached_tokens.size()` (shorter prompt sharing a prefix with a longer cached prompt — req 5 after req 4), `copy_slot(cache_slot→0)` restored GDN state from the *end* of the longer cached prefill. Secondary: work arena carried scratch across requests (T10/decoy/`det2.py` divergence).

## 3. Work order

**SPLIT (2026-08-23): two agents. Do not take the other agent's tasks.**
Remote agent's full brief + hard boundaries: **`docs/51`**.

### 3.1 GPU agent (owns the live server, port 8091, main worktree)

~~1. **P0 — D-01.**~~ **DONE** (712c0fd5).
~~2. **P0 — D-10.**~~ **DONE** (712c0fd5).
~~3. **P1 — D-03**~~ **DONE** (712c0fd5).

1. **P1 — D-04** ~~streaming UTF-8 json 316~~ **DONE** (UTF-8 stream coalescing + remaining flush; T4 PASS).
2. **P1 — D-12** ~~long-prefill early stop~~ **DONE** (4e85dcd2 chunked prefill + OutputSession stops; T9 PASS 168.8s).
3. **P1 — D-13** context overshoot / always-compact: **layer 1 DONE+verified** (T14 PASS); **layer 2 DONE** (`/v1/compact` implemented + unit-tested; live T15 pending).
4. **P3 — D-08**, ~~**T12**~~ **DONE** (wired; use `pkill -x ninfer-serve` in battery — broad `-f apps/ninfer-serve` kills the test runner when `--restart` embeds the launch cmd), ~~**D-11**~~ **CLOSED 2026-08-24** (mitigated via `setsid`; no external process management per user).
3. **Verification & merges:** remote agent pushes branches (`wo/auto-kv`,
   `wo/tool-calls`, `wo/ci-wiring`, …). Merge **one at a time**, run the full
   battery per §4, update §7 baseline. Nothing is done until it is merged
   AND battery-green. **D-02/D-09/D-07 unblocked for remote** (GPU tasks above landed).

**GPU agent MUST NOT start:** D-07, D-02, D-09, CI wiring (remote tasks —
docs/51 §2). If you find a new defect that is code-only, register it with
owner `REMOTE` and leave it.

### 3.2 Remote agent (code-only, no server — see docs/51)

Tasks in order: **D-07** design doc → **D-02** auto-KV port → **D-09**
tool_calls emission → CI wiring → docs upkeep.

**Remote agent MUST NOT:** D-01/D-10/D-03/D-04/T12/D-08, launch or probe
the server, edit `tp2_backend.cpp` / `text_context.h` / `admission_*` /
`request_memory*`, merge or push to `main`.

### 3.3 Shared rules

- Housekeeping done: D-05/D-06 fixes committed (`5a270a85`), battery + docs
  committed (`bab85197`).
- **Policy: every new user-visible bug gets a register entry here (with an
  owner) AND a test in the battery before its fix is accepted.**

## 4. Work protocol (per task)

Follow this for every work-order item, in order:

1. **Take a task** — pick the topmost OPEN item in §3. Mark its §2 row
   `IN PROGRESS (by <name>, <date>)`. Do not start a new item until the
   current one has reached step 7.
2. **Test first** — if the defect has no failing battery test, add one and
   confirm it FAILS against the current build (repro on the live server).
   Any new defect discovered gets a §2 register entry before its fix.
3. **Fix** — minimal, at the root-cause site. Rebuild:
   `cmake --build build -j24`. Restart the server per §5 (kill,
   `setsid` relaunch, wait for "listening").
4. **Verify** — run the full battery:
   `python3 tools/smoke/test_serve_correctness.py --base-url http://127.0.0.1:8091 --model qwen3.8-27b`
   - the **specific test(s) for this defect must PASS**;
   - **every test that passed in the §7 baseline must still PASS** (no
     regressions). If a test itself needs a parameter fix, say why in §7.
   - also run `python3 /tmp/det2.py` (D-01 sanity: expect `n2==n3: True`).
   - if the fix touches memory/sizing, also run `tools/smoke/serve_battery.py`
     S1–S5.
   - a server death during verify is a finding: log it in §2, relaunch, continue.
5. **Update the doc** — §2 row: status + one-line finding; defect detail:
   append root cause + what was verified; §7 baseline table: new results +
   timestamp; §1: bump "last updated".
6. **Commit & push** — one commit per work-order item:
   ```
   git add -A
   git commit -m "fix(<area>): D-NN <summary> (battery: <test ids> pass)"
   git push github main
   ```
   Never push with a red battery.
7. **Close** — mark the §2 row `FIXED (commit <sha>)`, take the next task.

## 5. Execution context (for whoever picks this up)

- Repo: `/home/intel/ninfer/repo` (git clean except the 2 modified files + 1 untracked test — see §3.8). Build: `cmake --build build -j24` (Release). Binary: `build/apps/ninfer-serve`.
- Model: `/home/intel/models/qwen3_8_27b.ninfer` (Qwen3.8-27B Q4, 64 layers = 16 full-attn + 48 GDN; 4 KV heads, 2/rank, head_dim 256).
- **Launch (must be `setsid`, see D-11):**
  ```
  cd /home/intel/ninfer && setsid nohup ./repo/build/apps/ninfer-serve \
    /home/intel/models/qwen3_8_27b.ninfer --port 8091 --devices 0,1 \
    --spec mtp --draft-tokens 3 --kv-dtype int8 --kv-capacity 100000 \
    --max-context 80000 > /tmp/serve_test2.log 2>&1 < /dev/null &
  ```
  (~30 s to "listening on http://127.0.0.1:8091"; model id `qwen3.8-27b`, auth off. Kill with `pkill -x ninfer-serve` — do **not** use `pkill -f apps/ninfer-serve` from a shell whose argv embeds the launch command, e.g. the battery `--restart` flag.)
- Current log: `/tmp/serve_test2.log` (rotates on restart). Old session log with the user repro: `/tmp/serve_100k.log`.
- **Test battery:** `cd /home/intel/ninfer/repo && python3 tools/smoke/test_serve_correctness.py --base-url http://127.0.0.1:8091 --model qwen3.8-27b [--only T3,T10]` (T9 is slow, ~2–4 min; exit code = # failures). Full baseline: §6.
- Determinism repro: `python3 /tmp/det2.py` (3 requests; expect `n2==n3: True` after the fix).
- Upstream reference code: `/tmp/ninfer-up` (Neroued/ninner — `src/runtime/engine/admission_policy.{h,cpp}`, `request_memory.{h,cpp}`, `kv_capacity.h`).
- User context: wants **short** chat answers (1–2 sentences); detail goes in docs. Reference llama.cpp command (200k on same GPUs, the bar for KV cost): `/home/intel/complete-fix-llama-cpp/build/bin/llama-server --model ~/models/Qwen3.8-27B-UD-Q5_K_M.gguf --port 8080 --ctx-size 200111 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 -ngl 999 -b 4096 -ub 512 -sm tensor -ts 1,1 --spec-type draft-mtp --spec-draft-n-max 3 -n -1 -np 1`.

## 6. Test suite expansion

### 6.1 Current inventory (what exists)

- **102 C++ unit tests**: arena, tensor, kv_cache, kv_capacity, sampling defaults, tool_call_parser, openai/anthropic/responses schemas, serve_options, request_log, request_memory, admission_policy, http_error_handler, state_store, response_store, media_decode, gdn_replay_records, decode_graph, artifact.
- **TP2**: test_tp_group, test_tp_kernel, test_weight_shard, one_shot_allreduce correctness, tp2_decode/load/shard_plan (2-GPU).
- **Serve smoke**: `serve_battery.py` (S1 mixed traffic, S2 prefix multi-turn, S3 sampling API, S4 cancellation, S5 VRAM flatness), `serve_contract.py`, `serve_thinking_preservation.py`, `test_serve_corpus.py`, `test_bench_matrix.py`.
- **Parity**: `tools/parity` (vision only — no text/MTP parity harness in-tree).
- **Gap**: none of the above would have caught D-01..D-04. There is no cross-request state test, no prefix-restore correctness test, no streaming UTF-8 test, no repetition-loop guard, no tool-call round-trip test, no long-context needle test.

### 6.2 New serve correctness battery

Location: `tools/smoke/test_serve_correctness.py` (run against a live server; greedy
sampler for determinism; asserts log lines where marked).

| # | Test | Catches |
|---|------|---------|
| T1 | **Prompt isolation** — separate conversations: "hi", "hello", "what is your name", "what is 2+2", "list US presidents". Assert no answer contains content from any other conversation and no forbidden corpus tokens (`python`, `csv`, `function_results`, `def `, `import`). | D-01 (today's exact repro) |
| T2 | **Prefix-reuse equivalence** — same prompt twice; 2nd must be a cache hit (log assert) and output token-identical. | prefix restore sanity |
| T3 | **Prefix divergence** — conversation A long multi-turn; conversation B shares ~60% of A then diverges. Run B cold (baseline) then B warm (A's cache present). Outputs must be token-identical. | **D-01 (the gate test)** |
| T4 | **Streaming UTF-8 integrity** — force multi-byte content (emoji 👋, CJK, German umlauts); stream; every SSE chunk valid JSON; concatenation valid UTF-8; no split code point at chunk boundaries. | D-04 |
| T5 | **Mid-stream disconnect** — kill client at ~50% of a long generation; subsequent T1/T3 must pass clean. | state bleed after cancel (req 4 today) |
| T6 | **Tool-call round trip** — pi-style 14-tool session, 5 turns with real tool calls (read/bash); tool_calls JSON must parse; args valid. | pi agent viability |
| T7 | **Repetition guard** (global, applied to all responses in the battery) — no output line repeated >8×. | the `</function_results>` loop |
| T8 | **Admission/oversize** — (a) ~40k prompt + 8192 max_tokens (< cap) → 200; (b) max_tokens > cap → expect 4xx, server alive, next request 200. | D-03 |
| T9 | **Long context** — 8k/32k/64k prefill; needle-in-haystack at 25/50/75% depth; 80k prefill + 256 decode at the end. | A-3 class regressions, q8 KV at scale |
| T10 | **Determinism** — fixed seed, 3 runs (one of them a cache-hit run) token-identical. | MTP/sampling drift |
| T11 | **Leak/flatness** — 50 mixed requests; GPU memory before vs after within 200 MB (extends S5). | slow leaks, D-08 class |
| T12 | **Clean restart** — kill server, restart, T1 passes. | stale-state assumptions |
| T13 | **Prefix-cache boundary** — plen ≤ P prefix hit speedup; plen > P full interleaved prefill determinism (D-07). | 200k prefix boundary |

### 6.3 Wiring & policy

- `scripts/run_serve_tests.sh` gains the correctness battery; T1–T8 run in CI
  (`run_ci.sh`), T9–T11 nightly, TP2 tests marked 2-GPU-only.
- **Policy: every user-visible bug from a test session gets a test in this battery
  before its fix is accepted.** (T3 is D-01's test, T13 is D-07's test.)

## 7. Test battery baseline (2026-08-23, main, post D-13 layer 1, live 80k/100k I8/MTP server)

| # | Result | Note |
|---|--------|------|
| T1 | PASS | prompt isolation (hi/hello/name/2+2/presidents) |
| T2 | PASS | prefix hit, output identical |
| T3 | PASS | cold/warm B identical (D-01 gate) |
| T4 | PASS | stream UTF-8 👋+你好 (`enable_thinking: false`; D-04 fix) |
| T5 | PASS | disconnect clean |
| T6 | PASS | native tool_call round trip (`enable_thinking: false`, short system prompt) |
| T7 | built into all | |
| T8 | PASS | (a) ~12k prompt → 200; (b) max_tokens=1M → 400 |
| T9 | PASS | long-prefill needle all depths green (D-12 FIXED 4e85dcd2), 168.8s |
| T10 | PASS | cross-request determinism (D-01 gate); `det2.py` n2==n3 ✓ |
| T11 | PASS | VRAM flat over 20 requests |
| T12 | PASS | clean-state probe (wrapper-orchestrated restart; T1+T2 on fresh instance) |
| T13 | PASS | prefix-cache boundary at 200k (D-07) |
| T14 | PASS | overshoot tolerance zone (D-13): 90,439-tok prompt served on 80k/100k; 125k-tok → 400 naming kv capacity; 130.2s |

**No-regression gate:** T1–T14 must stay PASS.

### 7.1 Expected results (what "battery-green" means)

**Must PASS every run, no exceptions:** T1, T2, T3, T5, T7 (all), T8, T10, T11, T12.
**Expected to PASS:** T4, T6, T9, T13, T14. T9/T13/T14 are long (68–169s) and run nightly; T12 on server-restart work only. T14 requires a tolerance-zone config (`--kv-capacity > --max-context`).

**KVarN servers (`--kv-dtype kvarn_k4v2`):** T16, T17, T18 must PASS (live smoke + prefix speed + restore equivalence). **T19 (80k prefill ≥450 tok/s) achieved by the D-18/D-19 wall fix (`acd6f791`)** — full CI 20260826_095551 green including T19.

**MTP acceptance gate clarification (2026-08-24):** the 82.0% baseline is workload-specific (test-driver prompt with `--no-draft-vocab`), not a KV-type property. Same-workload comparison measured 2026-08-24 on this server: 512-token essay, KVarN k4v2 = **50.3%** acceptance / 2.51 tok/round vs int8 = **50.1%** / 2.50 — KVarN does NOT degrade MTP acceptance (draft and target see the same KV). The <78% gate is a regression tripwire on like-for-like workloads, not a cross-KV comparison.

**A FAIL is only closed as non-regression if it has a control**:
re-run the failing test (a) twice in a row and (b) immediately after a
`pkill -x ninfer-serve` + §5 relaunch. If the control passes but the
original context fails, it is a **state-dependent server bug — file a defect
entry, do not attribute to the model.** If all runs fail identically, suspect
model/prompt behavior; still document the control evidence in the table note.

**No-regression gate:** every test that PASSed in the latest table must PASS
in the next run. A test that PASSed once and later FAILs is a **regression
by default**, even if the failure text looks model-like.

**§7.1 control (2026-08-23, T4/T6/T9, pre D-04 fix):** two consecutive runs +
two runs after fresh relaunch — all failed identically (thinking-on early stop /
`We`/`The`+im_end). **Not** cross-request state. Mitigations applied:
- T4/T9: `enable_thinking: false` in battery
- T6: short system prompt (40× repeat blew prefill budget → `GL` stop)
- T4 server fix (D-04): UTF-8 stream coalescing — T4 now PASS
- T9: still FAIL — server-side long-prefill decode (D-12), not prompt attribution

**Remote-agent verification run (11:16, post-rebase, `wo/remote-ports`):**
T1 T2 T3 T5 T8 T10 T11 PASS (matches table); T4 FAIL (early stop, same mode);
T6 FAIL (content `"We \\n</think>"` — stop token inside thinking, same mode);
T9 FAIL (content `"We \\n</think>"`, same mode). The 11:07 table did not run T9.
**T4/T6/T9 share one failure mode — see §7.1.**

### 7.1 Expected results (what "battery-green" means)

**Must PASS every run, no exceptions:** T1, T2, T3, T5, T7 (all), T8, T10, T11.
**Expected to PASS:** T4, T6, T9, T12 (when wired). T9 runs nightly; T12 on
server-restart work only.

**A FAIL is only closed as non-regression if it has a control**:
re-run the failing test (a) twice in a row and (b) immediately after a
`pkill -f apps/ninfer-serve` + §5 relaunch. If the control passes but the
original context fails, it is a **state-dependent server bug — file a defect
entry, do not attribute to the model.** If all runs fail identically, suspect
model/prompt behavior; still document the control evidence in the table note.

**No-regression gate:** every test that PASSed in the latest table must PASS
in the next run. A test that PASSed once and later FAILs is a **regression
by default**, even if the failure text looks model-like.

**Known open (11:16):** T4/T6/T9 early-stop failure mode — unattributed
pending control. Owner: **GPU** (server-side until model-quirk is proven by
the §7.1 control). Related: D-09 (remote) covers tool-call *format*, not this.
Remote hardened the T4/T6/T9 prompts (imperative phrasing; see
`test_serve_correctness.py`, commit after c905b7fa) — after merge, the first
full run sets the new §7 baseline; any still-failing test goes through the
§7.1 control before attribution.

## 8. D-01 user repro (exact)

1. Launch per §5 (q8, 100k cap, 80k context).
2. Three separate conversations (fresh sessions), one message each:
   - "hi" → got fake file read + `</function_results>` loop (log req 2)
   - "hello" → got CSV script (log req 4, cancelled at 769)
   - "what is your name" → got CSV script again (log req 5, prefix hit 4117)
3. Log file: `/tmp/serve_100k.log`.
