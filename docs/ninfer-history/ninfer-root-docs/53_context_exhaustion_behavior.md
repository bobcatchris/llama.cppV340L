# 53 — Context exhaustion: what happens, what it should do

Status: **ROOT CAUSE FOUND + FIXED** (2026-08-23).
Trigger: pi session at ~79k/80k context; request with 31,518-token prompt;
server SIGKILLed ("Killed") mid-decode at ~8k generated tokens.

## 0. Root cause (final)

The artifact loader mmaps the **entire 16.7 GB model file**
(`src/artifact/reader.cpp`, `MappedFile`), and `TpBackend::create()` kept the
`Reader` alive for the server's lifetime (`tp2_backend.h: reader_`). The mmap
is only used during load — nothing is served from it afterwards — but holding
it kept 16.7 GB of page cache counted against process RSS **forever**:
baseline RSS was **19 GB on a 32 GB machine** that also runs pi, a browser and
elasticsearch. Any transient spike (pi streaming buffers, page-cache churn,
swap pressure) pushed the box over the edge and the OOM killer took the
largest process — always ninfer-serve.

Reproduction confirmed: RSS was **flat at 19027 MB** through prefill+decode
(no per-token host growth; the decode loop is clean), and smaps attributed
16.7 GB to the `.ninfer` file mapping.

**Fix:** drop the reader at the end of `TpBackend::create()` (all weights are
copied to device by then). Baseline RSS: **19027 MB → 2278 MB**. Battery
T1 T2 T3 T5 T8 T10 T11 all pass on the fixed build; MTP acceptance healthy
(3.6–4.0 tok/round).

Residual exposure (machine-level, not server code): pi + browser + elasticsearch
still share 32 GB. If OOMs recur, next steps are host-side (close tabs,
elasticsearch heap) or `--kv-dtype`/context sizing — the server is no longer
the top target.

## 1. What happened (evidence)

- Progress lines worked perfectly on the 31k prefill (2%/16%/31%…/97%, ~800 tok/s).
- Decode started with limit 48,466 (= 80000 − 31518 − 16, the clamp — correct).
- At decode token ~8k (16:21:50) the process received **SIGKILL** → bash printed `Killed`.
- `earlyoom` (installed on this box, `-r 3600`) did **not** kill it — no "Killing
  process" line in syslog.
- No cgroup limit, no managed-memory (`cudaMallocManaged`) usage, `/dev/shm` empty.
- Kernel OOM killer is the leading suspect (dmesg not readable without root;
  syslog is quiet on this box so a gap is not conclusive).
- Machine: 32 GB RAM, ~25 GB available after the crash; disk **97% full** (8 GB free).

## 2. What was ruled out by code inspection

| Suspect | Verdict |
|---|---|
| KV pool growth during decode | No — `PagedKVPool` is device-backed, pre-allocated at startup for `--kv-capacity` |
| Resident buffers (cached_ph etc.) | No — fixed `{5120, max_context}` device slices (D-10), allocated at startup |
| Allreduce staging | No — 128 slots × 128 KB pinned, pre-allocated (~16 MB/rank) |
| GDN linear state | No — fixed-size per layer |
| Host vectors in decode loop | No — `generated_ids` ≈ 192 KB at 48k tokens; `local_history` ≈ 32 KB |
| Context overflow logic | No — both plain and MTP decode loops stop at `max_context` (`FinishReason::OutputLimit`); admission clamps/rejects oversized requests |
| Self-kill / abort in serve path | No — no `kill()`, no `abort()` outside `cuda_check` (which prints first) |

**Resolved:** nothing grew during decode — the 19 GB baseline itself was the
problem (§0). No per-token host allocation exists in the TP2 path.

## 3. How the reference servers handle this

### llama.cpp (`llama-server`)
- Fixed `n_ctx` per slot. If `prompt + max_tokens > n_ctx`: recent versions
  **clamp** max_tokens to fit and log a warning; prompt alone too long → 400.
- During generation, once the context is full the request finishes with
  `finish_reason: length` (or an error if even one token won't fit).
- **The server never dies.** Conversation compaction is a *client* concern —
  the app summarizes history and sends a shorter prompt.

### vLLM
- `max_model_len` bound. At admission: `prompt + max_tokens > max_model_len`
  → effective max_tokens clamped to fit (logged); prompt alone too long → 400.
- During generation, KV budget pressure is handled by the **scheduler**:
  requests are preempted (swap-to-CPU or recompute) and requeued. A request
  that cannot fit simply waits. GPU memory is partitioned at startup so the
  scheduler can never over-allocate → **the server never OOMs.**

### Common properties (target behavior)
1. Admission guarantees `prompt + output ≤ max_context` (clamp or 400).
2. All per-request memory comes from pools sized at startup — no growth path.
3. If anything still fails, the *request* fails with a clear reason; the
   *server* keeps running.
4. The client sees a machine-readable signal (finish reason / error) so it can
   compact and retry.

## 4. Where we stand against that target

| Property | Status |
|---|---|
| Admission clamp/reject | ✅ done (2026-08-23): omitted max_tokens clamped to `max_context − prompt − 16`; explicit oversized values still 400 |
| Fixed pools at startup | ✅ KV pool, resident buffers, AR staging all pre-allocated |
| Request-level failure instead of process death | ❌ **`cuda_check()` → `abort()`** on any CUDA error; a host OOM kills the process regardless. This is the "server dies" behavior |
| Client-visible signal | ⚠️ `FinishReason::ContextCapacity` exists in the enum but nothing sets it; need to verify what pi sees when a request dies mid-stream (connection reset) |

## 5. Plan (execute after server access returns)

1. **Reproduce with instrumentation** (no code change first):
   - Launch server as usual; background sampler logging every second:
     `VmRSS/VmHWM` from `/proc/<pid>/status`, `nvidia-smi` per-GPU MiB,
     swap usage — while running a 31k-prompt / long-decode request.
   - Identify which allocation site grows (host RSS vs GPU) and by how much.
2. **Make failure request-scoped:**
   - Replace blanket `abort()` in the serve path with catch-and-report:
     CUDA errors during a request → finish that request with
     `FinishReason::ContextCapacity` + log, keep serving. (Startup-time
     allocation failures still abort — a server that can't load is useless.)
3. **Client-visible signal:** map the failure to an OpenAI-compatible error /
   finish reason so pi compacts instead of hanging on a dead socket.
4. **Battery test** (T13, read-only client): long generation near the context
   edge must return a clean finish — never a connection reset; server stays up.

## 6. Notes for the reproduction run

- Do NOT use `--kv-capacity auto` (D-02); keep explicit 100k/80k for this test.
- The pi session that triggered this had thinking=on and ~79k context; a plain
  31k-prompt request reproduced the *shape* of the run (prefill + long decode).
- If RAM does not spike on reproduction, the trigger may be cumulative across
  many requests (prefix-cache churn) — in that case replay the pi session.
