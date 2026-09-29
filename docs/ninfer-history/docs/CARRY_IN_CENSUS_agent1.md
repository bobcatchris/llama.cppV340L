# CARRY-IN CENSUS — buffers on the decode path: allocated-once, written-per-request, read-before-guaranteed-write (agent1, WO-SUPPORT-1 Task B)

**Desk**: agent1 · **Date**: 2026-09-13 ~15:4xZ · **Tree for all bare line numbers**: `amd/main` @ `dfd5b7ec`
**Datum being explained** (canonical, `1d0ff3c6` on `origin/amd/t3-wip`): ONE server, identical
bin/prompt/flags, greedy, 5 sequential requests → **5/5 distinct**; first-fork chars 37–107; two forks
repeat the just-emitted token; `reuse=full_reset` and `cache=0` on every serve line.

**Seat discipline (WO §3/§5):** this is **support input, not adjudication**. Every row carries
`file:line@ref` and a one-command re-derivation; anything I could not re-derive is marked
`unverified` rather than promoted. **No patch proposals appear in this file** (board law: no patch
before the decisive measurement). agent4's pass-1 (`docs/amd/CARRYIN_HUNT_agent4_pass1.md` @
`origin/amd/wo-p3-serve:96e5a58a`, **not** on `amd/main`) explicitly asks this desk for the
files:line table that "decides which arm is cheapest" — §5 answers that question and nothing else.

## 1. Census format

`suspect | file:line@ref | alloc lifetime | per-request write? | read-before-write? | what would falsify`

`read-before-write?` ∈ {**N** = provably covered by a write on every path, **Y** = a read exists whose
path is not preceded by a write of the same bytes, **unverified** = I could not settle it statically}.

## 2. EXONERATED BY MEASUREMENT (not by reading) — the D2 uninit-read class is OFF the served path

The `NINFER_D2_POISON` probe (agent4's own instrument) targets `st.work` + the `mb_ext` envelope inside
the **dflash2 single-seq** block: `tp2_backend.cpp:3651-3655` (poison sites), reached only from the
runner whose comment is at `:3472/:3483/:3569` ("DFlash2 single-seq…", `worker` lambda at `:3497`),
with `d2_N = 1` at `:3585`.

**Measured, all three banked corpus logs** (`git show <ref>:<path> | grep -ci`):

    G17v (51dc3dd6)  G17g5 (wo-p3-serve)  G17g3 (wo-p3-serve)
    dflash: 0        dflash: 0            dflash: 0
    D2SS:   0        D2SS:   0            D2SS:   0
    mtp:    4        mtp:    4            mtp:    4

    Serve line that fixes the arm: "[tp2] single-seq lane 0: acceptance=0.00 tok/round=0.00 rounds=0
    gen=32 …" — acceptance 0.00 / rounds 0 on BOTH generations, in every boot.

**Verdict: N — with a named limit.** The speculative/drafting scratch class (fresh 64 MiB
`chain_scratch_backing`, `st.work` beyond the `a>0` envelope) is **not reached** by the geometry that
produced the 5/5-distinct datum, so it cannot be that datum's carrier. **Limit, stated so nobody
over-reads this:** `dflash:0/D2SS:0` proves *that block* did not run; `mtp:4` proves MTP-related lines
*did* print, so this is NOT "all drafting code is dead". A future cell that enables dflash2/MTP puts this
class back on the live list — the re-derivation is the same three greps, and they cost nothing.

## 3. The two rows where the zero IS per-boot but NOT per-request (my strongest finding)

### 3a. `cache_slot` is deliberately EXCLUDED from every per-request slot zero

| field | value |
|---|---|
| suspect | GDN recurrent/conv **`cache_slot`** value, per layer |
| file:line@ref | zero loops `tp2_backend.cpp:1903-1906` and `:2129-2132` (both read `if (sl != st.cache_slot) { zero_slot(sl, s); }`); slot-0 zero `:1894`/`:2124`; binding `state->cache_slot = cache_slot` at `:725-726`, literal `= mtp ? 2*k_buf+1 : 1`; consumer `set_linear_state_slots(0, 1)` at `:1936`, `:2162`, `:3547` (@ `dfd5b7ec`) |
| alloc lifetime | the containing `state_span` is memset **once per rank per boot** — `:812` `cudaMemsetAsync(state_span.data, 0, state_span.bytes, ctx.stream)`, comment at `:811`: "Pristine GDN slots (verify round 0 reads the empty slot 4) and clean KV pages" |
| per-request write? | slot 0 yes (`:1894`); all slots **except** `cache_slot` yes (`:1904`); `cache_slot` **no** — only `gdn_ckpt_*` restores write it (`:1839-1846` `cudaMemcpyAsync` conv/recurrent pairs) |
| read-before-write? | **N — RESOLVED at static, and the resolver was cheap enough to run.** Agent4's pass-1 §(a) left this as "the unread line"; it reads without a boot. (1) Single-seq binds **current = 0** (`:1936`, `:2162`, `:3547` are all `set_linear_state_slots(0, 1)`; the default binding at `text_context_impl.h:284` is the same shape) and slot 0 **is** zeroed per request (`:1894`). (2) The batched runner binds current = `b` (`:5272-5273`, `current=b, rewrite=st.lanes+b`) and zeroes exactly that slot one line earlier, at **`:5269 zero_slot(b, s)`**. (3) `set_linear_state_slots` **throws** when `current_slot == rewrite_checkpoint_slot` (`text_context_impl.h:291-295`), so current and the excluded checkpoint slot cannot alias by construction. (4) The only read of `cache_slot` is `copy_slot(st.cache_slot, 0, s)` at `:1851` — on the **prefix-hit restore** arm, which the corpus never entered (`--no-prefix-reuse` → `prefix_len = 0` → the `:1893` else-arm). Net: every slot read as *current* on the served path is zeroed before the read. |
| what would falsify | Delivered, and it is still the re-derivation: `grep -rn "set_linear_state_slots" src/ \| grep -v "(0, 1)"` returns only `:5272` (batched `b`, zeroed at `:5269`), the declaration `text_context.h:369`, the definition `text_context_impl.h:290`, and the default binding `:284` — **no binding passes `cache_slot` as current**. The row re-opens only for a geometry that enables prefix reuse AND takes `copy_slot(cache_slot, 0)` with a `cache_slot` last written by a different session's decode — i.e. a prefix-reuse cell, not this corpus. |

### 3b. `zero_pages` has exactly ONE caller in the whole tree, and it is not the rewind

| field | value |
|---|---|
| suspect | KV physical pages reused across requests while the **rewind** path zeroes nothing |
| file:line@ref | `src/core/paged_kv_cache.cpp:167-201` (`zero_pages` → `zero_run`, `cudaMemsetAsync`/`memset2DAsync` on planes) @ `dfd5b7ec`; **sole in-tree caller** `src/targets/qwen3_6/impl/runtime/program_impl.h:1185` (`cache.pool().zero_pages(pages, device.stream)`) |
| the rewind actually used | `tp2_backend.cpp:1852-1853` `kvarn_rewind_text(prefix_len)` / `kvarn_rewind_mtp(prefix_len)`; the `prefix_len==0` arm additionally calls `kvarn_reset_inflight()` (`:1898`, `:2125`, `:3546`) and `publish_mapping(s)` (`:1908`) — **`zero_pages` appears in none of them** |
| alloc lifetime | pool planes allocated once per boot (part of the `state_span`/pool construction at `:807-812`) |
| per-request write? | pages are re-**written** by the prefill that fills them; a page is only *read* by attention up to `valid_v`/`extents` (`:1198-1199` `valid = k_buf + 1`, `ext` written at `:1203`) |
| read-before-write? | **N for within-envelope reads** (attention is bounded by the envelope published per request); **unverified beyond it** — the D2 comment block itself names "the kernel reads columns beyond the envelope" as a live pattern at `:3639-3641` (that instance is off the served path per §2, but the *habit* is the precedent that keeps this row open) |
| what would falsify | `grep -n "zero_pages" src/ -r` → still one caller at `program_impl.h:1185`, and read its enclosing condition: if that caller runs on every allocation the served path can reach, then every reused page is zeroed before first read and the row closes **N**. The zero-card falsifier for the residue is agent4's own §(b) form: table every read of a KV page whose generation < this request's write set — one pass over the kvarn stage/attend call sites, no boot. |

## 4. The five WO-named starting points, resolved (with two dead pointers reported)

| WO-named starting point | result at `dfd5b7ec` |
|---|---|
| `tp2_backend.cpp:768` comment thread | **Stale line, real thread.** At this tip `:758-770` is the `decoder_spec()` layout block whose live content is the **FP32 recurrent-state** warrant ("FP16 here quantized the state at the 225 boundary, drifting one late argmax", `:767-769`) — which is *itself* a documented instance of state-width → late-argmax drift, i.e. the same failure family as the datum. The zeroing thread I needed is at `:811-812` instead. Cite `:811`, not `:768`. |
| `gdn_gating_proj_gemm_mma.cuh:275/:307/:330` slice coverage | **File path in the WO is short by two components.** Real: `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh`. Coverage answer at the named lines: `:270-277` split-k store `partial[(split*t+token)*kLogicalRows + {row, kHeads+row}]`; `:279-292` `FullTokens` unguarded store vs `if (col0 < t)` / `if (col1 < t)` **guarded** store; `:307-322` `NormalizeInput` reduce reading `norm_partial[s*t+token]` for `s < SplitK`; `:330-343` the av/bv combine, fixed order `#pragma unroll for (s = 0; s < SplitK; ++s)`. **Read-before-write = N on coverage, and provably so:** the write guard is `token_count = min(kBlockN, t - token0)` (`:103`) over `token_local < NormTokenCapacity` (`:109`) and launcher capacity is selected to *contain* t — `kernels.cu:608-616`: `t<=6→6, t<=8→8, t<=12→12, t<=16→16, else throw "fused BF16 GDN norm/control requires T=1..16"`, with `static_assert(SplitK == 32)` at `:98`. So every index the reduce reads was written by the split that owns it. This **supports** agent4's §1 "fixed-order, no order-varying float combine" exoneration, adding the capacity-dispatch proof it did not cite. |
| arena park/restore paths | `host_kv_parked.cpp` — restore is **driven by the IMAGE's component list, never the live `snap.valid`** (`:12-14`, `:78` "never from the live snap.valid"), i.e. already hardened against exactly the stale-view class; `net.park(...)` before re-resolving the target (`tp2_backend.cpp:1812`) with the reason in-comment (`:1806-1808`: park's eviction can drop the matched entry). Copy paths: `:1839-1846` (conv+recurrent `cudaMemcpyAsync` D2D). No zero-on-restore is needed because restore writes every byte it claims — but **over-capture is explicitly called safe** at `host_kv_parked.cpp:9-10` ("restored pages beyond the flow's rewind point are unreferenced") — that sentence is an *assumption*, and it is the cheapest thing in this file for agent4 to attack: it is the same "beyond the envelope" habit as §3b. |
| `kv_bytes_per_token=18496` ring init | Lives as a **default in a header, not a ring init**: `tp2_budget.h:65` `kv_bytes_per_token = 18496`, documented `:26` "per context token KV ring 18496 B (I8) / 34816 B (BF16)". The **measured** corpus number is the BF16 one — G-AMD-26's own printout: `kv(34816 B/t x 128 tok + mtp 0) 4` MiB. So a census that used 18496 against this boot would be charging the I8 tier to a BF16 serve; the predicate that decides which is `cache_type_{k,v}_bits` at the request, not a constant. **No ring-init memset found at this ref** — the ring's zero state comes from the `state_span` memset at `:812` (boot-once) plus §3b's `zero_pages` (single caller). Marked `unverified`: whether any per-request path re-inits ring *metadata* only. |
| workspace 96 MiB consumers | Env `NINFER_WORKSPACE_MIB` (`tp_engine.cpp:746-753`), default work arena sized `1 GiB + 2 MiB·(chunk-512)` at `tp_engine.cpp:734-735` — so the `96` in the kit's env is an **override**, and its consumption site is `state->work` (`tp2_backend.cpp:1189` capacity throw = a *capability* check that names its predicate; `work_scope` per-request at `:1693`, `:3502`, `:5159`). Read-before-write: **unverified by me, and it is the one place an instrument already exists** — `:3651-3655` can poison the entire arena to 0x5A under `NINFER_D2_POISON`, env-gated, zero-cost unset. But per §2, on the served greedy geometry the dflash2 block that arms it did not run, so re-using that probe requires either the batched/greedy runner to poison `st.work` too, or the datum moves to a cell that does reach it. **Stated as a census observation, not a patch.** |

## 4b. Consequence for the hunt — stated as an instrument fact, not a verdict

Agent4's pass-1 §(a) proposed the discriminating arm `NINFER_GDN_SLOTZERO=1`: "at session-init,
force-memset the new session's committed GDN slots + conv window before first step". Against §3a as
resolved, **that probe is already-covered ground** — the slots it would force-zero are zeroed today at
`:1894` (slot 0), `:1903-1906` (every slot but the checkpoint), and `:5269` (batched lanes). A probe whose
action duplicates an existing write **cannot discriminate**: if the forks survive it — and mechanically
they should — the row would read "(a) dead" while having tested nothing new. Naming this before a boot
is the census's most useful output, and it retires a device-time arm at zero cost.

## 5. Which arm is cheapest — the answer to agent4's question, ordered by zero-card cost

1. **§3b envelope-assumption table** (static, one pass over the kvarn stage/attend call sites) — now the
   **top** row precisely because §3a resolved to N. It attacks the one place this tree states a safety
   property in prose instead of in code: `host_kv_parked.cpp:9-10` — "restored pages beyond the flow's
   rewind point are unreferenced … over-capturing is safe" — next to `zero_pages`' single caller
   (`program_impl.h:1185`, invoked at `:1203` text / `:1205` mtp / `:1207` dflash). If any served pool
   takes a rewind without a matching `zero_pages`, its reused pages carry the previous request's bytes
   into a read bounded only by an envelope integer.
2. **§2's D2 class** — measured OFF the current path (`dflash:0`, `D2SS:0`, `acceptance=0.00`); do not
   spend window minutes on it unless a cell enables dflash2/MTP. If such a cell runs, the probe already
   exists (`NINFER_D2_POISON`, `:3651-3655`) and costs one env var.
3. **agent4's `NINFER_GDN_SLOTZERO` arm — DEPRIORITIZED per §4b** (duplicates writes that already
   happen; a null result would be uninformative). If a GDN-shaped probe is still wanted, the
   discriminating targets are the **conv-window restore** (`:1839-1842`, `gdn_ckpt_conv` → `conv_slot`)
   and the beyond-envelope reads — not slot zeroing.
4. **§4's split-k / fused-norm combine** — closed **N** here with the capacity-dispatch proof. This
   retires the last "logits-producer numerics" branch that agent3's #784 handed forward as the next
   hypothesis after the sampler window.
5. **§3a** — closed. Kept in place rather than deleted, because the *route* to N (three bindings, a
   throws-on-alias guard, one unreachable restore arm) is what makes it re-checkable when A-4 moves the
   slot code.

Net zero-card cost of the census's remaining open set: **two static passes, no boot.**

## 6. What I did NOT establish (no over-claiming)

- **No `Y` (guilty) verdict anywhere.** Two rows opened `unverified` and **both closed at static** (§3a → N;
  §4's split-k/norm → N); one class was **excluded by measurement**, not by argument (§2); and exactly one
  row stays open (§3b, beyond-envelope KV reads), whose resolver is a static pass, not a boot. A census
  that had handed the hunt a `Y` from a read of 4000 lines would have been the more likely kind of wrong,
  so the nulls are reported as nulls.
- Whether the 5 forks *must* come from carry-in — that is agent4's adjudication seat (WO §5), unchanged.
- `st.work`'s per-request read/write coverage in full (I only located its consumers and its existing
  poison instrument).
- All line numbers move at merge speed: **re-derive with the greps in each row, not by trusting this
  file** — that is the whole point of the `file:line@ref` column. `dfd5b7ec` is where every bare
  number above is true.
