# Doc 41 — TC-P Issue 2: bisection data + handover

**Status: Issue 1 FIXED & verified. Issue 2 narrowed to one code region + one kernel-path question. Ready for agent pickup.**

---

## 1. TL;DR

- TC-P gate = 2nd request's generated output must be **bit-identical** between prefix-cache ON and OFF runs.
- **Issue 1** (one-shot allreduce reset desync): root-caused, fixed, verified. See §3.
- **Issue 2** (remaining TC-P failure): divergence enters the forward pass **after the GDN mixer output of the first GDN triplet and before/at the MLP partial** of that triplet, at a token proven to be identical in both runs. Two explanations remain, both actionable (§5).
- All probe instrumentation is **uncommitted** in `/tmp/ninfer`. Do NOT revert before reading §7 (cleanup checklist).
- Pushes go to `origin` = `git@github.com:chrisconcepcion/dual_5060_ti_ninfer.git` ONLY. Never to the upstream ninfer repo.

## 2. Background

- Branch `mtp-perf`, HEAD at handover time: `184e9d15` (doc 40).
- Battery: `/home/intel/verify_battery.sh`. TC-P cases: `prefix_on` / `prefix_off` (L111-112); gate `prefix_identity_ok` (L184: 2nd-request output text equal).
- True perf baseline (do not regress): MTP k=3 **81.02 t/s**, acceptance 68.9%, plain 35.49 t/s, pp 32.0 t/s.
- Box: Ubuntu 22.04, 2× RTX 5060 Ti (36 SMs), TP2, ranks are **threads in one process** (`tp2_backend.cpp:1293 std::thread th1`) → all C++ `static` counters are **shared between ranks** (matters for probes, see §6 caveat).
- Artifact: `/home/intel/models/qwen3_8_27b.ninfer`
- Test binary: `/tmp/ninfer/build/tests/ninfer_tp2_decode_test`
- Build: `cd /tmp/ninfer/build && make -j24 ninfer_tp2_decode_test`

## 3. Issue 1 — CLOSED (one-shot AR reset desync)

**Symptom:** req2 prefix-hit run had stale one-shot-AR flag on rank1 (flag1=6).
**Root cause:** `reset_one_shot_step` (called between requests) had **no stream synchronize** — the reset raced the in-flight AR kernels.
**Fix (uncommitted, `src/runtime/tp2/tp2_backend.cpp` L829-837):**
```cpp
cudaStreamSynchronize(s);
sync_bar.arrive_and_wait();   // existing barrier, now guaranteed post-kernel
```
**Verified:** post-fix, req2 GDN tensors are bit-identical between ON/OFF runs (v1probe).
**Do not remove** the sync. Also uncommitted: debug accessors in `one_shot_allreduce.{h,cu}`, `tp_group.{h,cpp}` (OneShotDebug) — keep for now, remove with probes per §7.

## 4. Issue 2 — what is PROVEN (data)

### 4.1 Token alignment (trusted anchor — use ONLY these)

v1probe c-counter (per-gidx, fires window `[plen1e, plen1e+2)` via `NINFER_REQ1PLEN`):

| Run | env | counter values = req2 local tokens |
|---|---|---|
| ON (prefix) | `NINFER_REQ1PLEN=224` | c=224,225 = req2 local 224, 225 |
| OFF (no-prefix) | `NINFER_REQ1PLEN=448` | c=448,449 = req2 local 224, 225 |

Token identity proven by value bridge: ON c=225 gidx2 `x=1200c74a7c31f2d0` == OFF c=449 gidx2 `x=1200c74a7c31f2d0`.

> ⚠️ The attnprobe **t-counter is NOT reliable** (increments at gidx=0 mid-token; fidx=0 lines land at t−1; possible rank-related drift — weird t values 411/426/428/434/441 observed in OFF log). Never anchor a comparison on an attnprobe t-label. The last wide-window run (222/444) was **misaligned** (ON c=222/223 = req1 tail; OFF c=444/445 = req2 local 220/221) — discard it.

### 4.2 Layer-chain facts (at the anchored token)

Layer pattern `[Full, 3×GDN]`: fidx_k = model layer 4k; gidx_{3k+j} = model layer 4k+1+j.

1. **v1probe (GDN mixer, pre-AR `p` and post-AR `x`, 9 tensors each):** gidx 0,1,2 (layers 1,2,3) **bit-identical** ON vs OFF at both anchored tokens. **gidx3 (layer 5) diverges** (e.g. h: ON `669504fb…` vs OFF `2ce51187…`).
2. **mlpprobe (local SwiGLU pre-AR partial):** at gidx2, the **local partial DIFFERS** (ON `7241bb0d…` vs OFF `970f54c5…`) even though v1probe says the mixer output (MLP input) is bit-identical. All 4 mlp partials in the group differ.
3. **attnprobe (t-labels untrusted, qualitative only):** fidx xin values differ between runs at the nominal same token — consistent with (2) if the residual entering the MLP region is already wrong, but NOT usable as an anchor.
4. **mtp_ph hash:** restored `74ab3db5…` vs freshly computed `d5c65f32…` — they differ.

### 4.3 The leak gate (strong discriminator)

Bisect A/B (earlier, verified):
- `--mtp 0` → clean. `--mtp 3 --tokens 1` → clean. `--mtp 3 --tokens 64` → dirty.
- → The leak **requires >1 MTP-generated token**. Any correct hypothesis must explain this gate.

### 4.4 Eliminated (each empirically verified, do not re-investigate)

GDN kernel purity (T=1 pure) · GDN state zeroing/restore · MTP head GDN layers (none exist) · Envelope · T=1 GDN workspace · conv state (in-place) · KV page over-read · probe race (v1probe syncs) · GEMV determinism in isolation · Small-T kernel masking · stale KV · prefix KV pages · block table (fixed via materialize_pages) · GDN state restore · valid_tokens · cache write race · CUDA graph (default OFF) · prefill non-determinism (prefill IS prefix-stable).

## 5. Remaining suspects (ranked, all actionable)

### A. Local SwiGLU GEMV partial differs with identical input (PRIMARY, data-backed)

mlpprobe shows partial divergence with a v1probe-identical input at gidx2. **Both probes are stream-synced before hashing (verified: `text_context_impl.h` L1563-1580 does `cudaStreamSynchronize(s)` before the partial hash)** → the measurement is clean. So this is a **real** divergence in the local SwiGLU path, not a probe race. Mechanisms to check (in order):

1. **Cross-stream arena aliasing race:** `partial = work_.alloc(...)` (arena) on the compute stream, while one-shot AR / P2P worker threads run on other streams. If a previous AR (or P2P send/recv buffer) aliases this arena region and is still in flight, the GEMV input or output can be corrupted nondeterministically. Check arena bump allocator vs. one-shot AR slot lifetimes; note the leak gate (§4.3) — heavier MTP activity = more in-flight ARs = more aliasing windows.
2. **Unordered read in the SwiGLU GEMV:** the GEMV reading a buffer (up-proj/gate-proj output, silu_mul input) whose prior write is not stream-ordered with the read.
3. **Bisect the MLP tail** to find the first divergent op: hash up-proj out, gate-proj out, silu_mul in/out separately (extend mlpprobe into `Variant::post_mixer_tp`).

### B. mtp_ph seeding in the prefix-hit restore path (SECONDARY)

`tp2_backend.cpp` L790-975: hit path does `copy_slot(cache,0)` + mtp_ph copy + zero slots 1..n, then prefills `i=prefix_len..plen` via `ordinary_decode_batch`. If the first new token's residual is seeded from **restored mtp_ph** (≠ fresh, hash-proven in §4.2.4) instead of the embedding, divergence starts at layer 0. This also naturally explains the §4.3 gate (mtp=0 → mtp_ph all-zero in both paths → clean).
**Check:** print/hash what `ordinary_decode_batch` receives as hidden/x for the first hit-path token vs the fresh path. If identical → suspect B is dead; A stands alone.

> Order of work: do B's 2-line print first (cheapest), then A's sync fix. They are independent.

## 6. Quick reproduction

```bash
BIN=/tmp/ninfer/build/tests/ninfer_tp2_decode_test
ART=/home/intel/models/qwen3_8_27b.ninfer
LONGPROMPT="The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today. The capital of France is Paris, and the weather is nice today."   # = 224 tokens
P2="$LONGPROMPT And to summarize in one word."   # req2 = 231 tokens
```

### 6.1 TC-P gate (the thing that must flip to PASS)

Full battery (runs TC-P + everything else):
```bash
bash /home/intel/verify_battery.sh
# gate in report: "Prefix identity TC-P(a) on==off"  (prefix_identity_ok)
```
Or the two TC-P cases only (~1-2 min total):
```bash
timeout 300 "$BIN" --artifact "$ART" --mtp 3 --tokens 64 --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2" > /tmp/tcp_on.log 2>&1
timeout 300 "$BIN" --artifact "$ART" --mtp 3 --tokens 64 --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2" --no-prefix-cache > /tmp/tcp_off.log 2>&1
# compare the SECOND "---- output ----" block of each log (req2)
diff <(grep -A6 '^---- output ----' /tmp/tcp_on.log  | sed -n '7,$p' | head -6) \
     <(grep -A6 '^---- output ----' /tmp/tcp_off.log | sed -n '7,$p' | head -6)
```
(Req2's output is the 2nd output block; the battery's python does the exact same comparison.)

### 6.2 Probe pair (full hash dumps, ~1 min each)

```bash
export NINFER_ATTPROBE=1
# ON: window catches req2 local 224/225 at c=224/225
NINFER_KVPROBE=1 NINFER_REQ1PLEN=224 timeout 300 "$BIN" --artifact "$ART" --mtp 3 --tokens 64 --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2" > /home/intel/verify_logs/attp_on.log 2>&1
# OFF: same semantic tokens at c=448/449
NINFER_KVPROBE=1 NINFER_REQ1PLEN=448 timeout 300 "$BIN" --artifact "$ART" --mtp 3 --tokens 64 --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2" --no-prefix-cache > /home/intel/verify_logs/attp_off.log 2>&1
```
Extract:
```bash
grep '^\[v1probe\] rank=0' /home/intel/verify_logs/attp_on.log  | grep -E 'c=22[45] '   # ON anchor
grep '^\[v1probe\] rank=0' /home/intel/verify_logs/attp_off.log | grep -E 'c=44[89] '   # OFF anchor
grep '^\[mlpprobe\]' /home/intel/verify_logs/attp_on.log  | head -10
grep '^\[mlpprobe\]' /home/intel/verify_logs/attp_off.log | head -10
grep '^\[v1ar\]\|^\[v1pre\]' /home/intel/verify_logs/attp_on.log | head
```
Existing logs from handover time (already contain this data): `/home/intel/verify_logs/attp_on.log`, `attp_off.log` (good 224/448 pair); `attp_on2.log`/`attp_off2.log` (the misaligned 222/444 pair — ignore for anchoring).

### 6.3 Fast A/B checks (no probes needed)

```bash
timeout 300 "$BIN" --artifact "$ART" --mtp 0 --tokens 64 --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2"  # clean
timeout 300 "$BIN" --artifact "$ART" --mtp 3 --tokens 1  --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2"  # clean
timeout 300 "$BIN" --artifact "$ART" --mtp 3 --tokens 64 --ctx 4096 --prompt "$LONGPROMPT" --prompt2 "$P2"  # dirty
```

## 7. For the agent — work order

1. **Read this doc §4-§5 first.** Do not re-run the eliminated checks (§4.4).
2. **Suspect B check (2-line print):** in the prefix-hit path (`tp2_backend.cpp` ~L895), hash-print the hidden/x vector passed to the first `ordinary_decode_batch` for `i=prefix_len`, plus the same in the fresh path (a `--no-prefix-cache` run, token `i=prefix_len` equivalent). If they differ → root cause found: fix the seeding (use embedding, or make mtp_ph restore bit-exact) → rebuild → §6.1.
3. **Suspect A (measurement already verified clean):** the divergence is real. (a) Check arena/one-shot-AR cross-stream aliasing (§5A.1): audit `work_` arena lifetime vs AR slot lifetime; if suspect, force-separate buffers and re-run. (b) Otherwise bisect the MLP tail: extend mlpprobe into `Variant::post_mixer_tp` — hash up-proj out, gate-proj out, silu_mul in/out — to find the first divergent op.
4. **Fix → §6.1 must show `prefix_identity_ok` PASS** AND full battery 0 fail (perf gates vs baseline 81.02 t/s).
5. **Cleanup (only after battery passes):** remove all probe code — `v1probe/v1pre/v1ar/attnprobe/mlpprobe` + token counter in `text_context_impl.h`; `[gqa]` logging in `ops/wrapper/gqa_attention.cpp`; debug accessors in `one_shot_allreduce.{h,cu}` + `tp_group.{h,cpp}`; kv_probe lambda in `tp2_backend.cpp`. **KEEP:** the AR reset fix (§3) and `materialize_pages` fix. Rebuild, re-run full battery one final time, update `/home/intel/verify_baseline.json` if perf moved.
6. Commit with specific file paths (NOT `git add -A`). Push to origin only.
7. Remaining worklist after TC-P: kv_i8 gate recalibration (absolute→relative), then WO-4..WO-7 per doc 39.

## 8. Uncommitted state at handover (all in /tmp/ninfer)

| File | Contains |
|---|---|
| `src/runtime/tp2/tp2_backend.cpp` | **AR reset fix (KEEP)**, materialize_pages fix (KEEP), kv_probe lambda (remove) |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | v1probe, v1pre, v1ar, attnprobe, mlpprobe, token counter (remove) |
| `src/core/multi_gpu/one_shot_allreduce.{h,cu}` | debug accessors (remove) |
| `src/core/multi_gpu/tp_group.{h,cpp}` | OneShotDebug + forwarders (remove) |
| `src/ops/wrapper/gqa_attention.cpp` | [gqa] wrapper logging (remove) |
| `src/ops/kernel/gqa_attention_decode_bf16.cuh` | minor (check diff before cleanup) |
| `src/runtime/tp2/tp_engine.cpp` | cache_ph_bytes VRAM estimate (keep) |

## 9. Reference values (trusted-numbers review)

- Perf: 81.02 t/s MTP k=3 (acceptance 68.9%), plain 35.49 t/s, pp 32.0 t/s
- v1probe anchor token (req2 local 225): gidx0-2 all 9 tensors identical; gidx2 `x=1200c74a7c31f2d0` (both runs)
- gidx3 first divergence: h ON `669504fb` vs OFF `2ce51187`
- mlpprobe gidx2 partial: ON `7241bb0d` vs OFF `970f54c5` (input per v1probe identical; probe sync verified → **real** divergence, §5A)
- mtp_ph: restored `74ab3db5` vs fresh `d5c65f32`
- Battery run with current (unfixed) state: expect TC-P FAIL + kv_i8 warn only
