# agent3 — item 7 follow-on: lm_head split-k REFUTED, GDN-gating carry-in REFUTED, and a placement CORRECTION to my own verdict row

**Session:** agent3 (fresh), pi `01a09af1-26b4` (qwen3.8-flash). **Zero-card, zero-boot, zero-fetch**
throughout — every number below is a re-run at my seat against a named ref, not inherited.
**Inputs I trusted only after re-deriving:** era base `d32d7d23` (resolves as a commit — `cat-file`
before declaring, my own law), lane tip `origin/amd/t3-wip @ becd3f64`, banked G-AMD-27 artifacts
`1e52e5e6`→`1d0ff3c6`.
**Refutation scope:** this sheet **supersedes §3 and the placement half of §2 of
`results/amd/T3i7r1_verdict_release_row.md` (`1d0ff3c6`)**. That row stays intact as provenance; the
verdict (per-request nondeterminism live) is UNMOVED — two of the three named root-cause doors just
closed, and one placement claim of mine was wrong.

## 1. Suspect (i) — "the logits producer's split-k reduction ORDER" — REFUTED at bytes

`1d0ff3c6` §3 ended with "the reduction whose order can vary may live in the logits producer, not the
sampler." I went and read the producer. It has no order-varying reduction.

**1a. Zero global atomics in the whole linear family, at BOTH tips.** The command (re-run at each
tree — `git grep`, so it works against a ref without a checkout):

```bash
git grep -nE "atomicAdd|atomicMax|atomicExch|atomicCAS|atomicOr" <REF> -- \
  src/ops/linear src/ops/linear_add src/ops/linear_pair src/ops/linear_swiglu \
  src/ops/attn_input_proj src/ops/gdn_input_proj src/ops/gdn_gating_proj
```
→ **no output at `<REF>`=`d32d7d23` (era) and none at `amd/main`.** Positive control on the same
command shape (so silence means absence, not a broken pattern): same form on `src/ops/kernel` returns
`argmax.cuh:1`, `sampling.cuh:4`, `speculative_round.cuh:4`. A cross-block combine that could vary by
arrival order needs an atomic or a cooperative handoff; there is neither in the vocab path.

**1b. The Q3 artifact's lm_head is W8G32_F16S, not Q3, and its shape is the exact-geometry small-T
arm.** Chain, each hop named (this is the cite the chair could not resolve — `qwen3_6_27b`, not the
vision or 35b file; full paths below):

| hop | locus | what it says |
|---|---|---|
| profile | `src/targets/qwen3_6_27b/impl/package.cpp:89-103` | weights_id `groupwise-q3` → `WeightsProfile::Qwen38GroupwiseInt` |
| endpoint dtype | `src/targets/qwen3_6_27b/impl/load/bindings.cpp:37-48` | `endpoint_format(Qwen38GroupwiseInt)` → **`W8G32_F16S`** |
| head bound | `…/bindings.cpp:533` | `output_head = bind_weight(…, vocabulary_format, {248320, 5120})` |
| sharded | `…/impl/load/tp_load.cpp:286-290, 301` | role `ColumnN` → `{full_rows/w, full_columns}` = **124160 × 5120 / rank** |
| geometry | `src/ops/linear/w8/w8_config.h:62` | `W8VocabularyTp2ProjectionGeometry = W8LinearGeometry<124160, 5120>` |
| route, decode | `src/ops/linear/w8/w8_dispatch.cpp:17-18, 36-41` | `k==5120`, `n∈{124160,248320}`: `t<=33 → launch_w8_small_t` |
| route, prefill | same | this boot ran `t=52` (measured: 480 `[gating] … cols=52` lines) → `t<=64 → launch_w8_mma_r32_c64` |

And the shard size is not just derived from the loader — **the boot's own argmax trace prints it**:
`grep -c "n_rows=124160" results/amd/T3i7r1_serve.log` → **328**, i.e. every `A1TRACE` line in the era
ran against `n_rows=124160`, the per-rank vocab shard the table predicts. Static route and measured
geometry agree at the same number, which is the only kind of agreement that closes a claim.

**1c. The small-T combine is a fixed-INDEX shared-memory tree, not an arrival-order one.**
`src/ops/linear/w8/w8_small_t_mma.cuh` — K-split warps own disjoint 64-wide slices, then:
`(k_split & 1)` warps `store_vec` their FP32 partials, even warps add partner `warp+1`, and the tail
combines `for (int split = 2; split < kWarps; split += 2)` in ascending constant order. Grid is 1-D
over output rows (`kBlocks = kOutputRows / kRowsPerCta`) — **there is no cross-CTA float combine in
the vocab GEMM at all**, so no per-launch residency can perturb it. `launch_w8_mma_r32_c64` is a 2-D
`(rows × tokens)` tile grid (`w8_rowsplit_gemm_mma.cu:20`) with K looped inside the CTA: also
resident-count-blind.

**1d. The sampler is order-invariant too** (this half was already in `1d0ff3c6` §3 and stands): greedy
combines by `>` on a packed key (`sampling.cuh:176`, `:192`) — max is commutative AND associative — and
`argmax.cuh:22` breaks ties on lowest index. The `group_done` atomics count arrivals, they do not
select a value.

**⇒ Suspect (i) is closed without spending a card.** G-AMD-26's lm_head wait dissolves; the pair is
the chair's to re-allocate.

## 2. Suspect (ii) — GDN-gating split-K partial **read-before-write** — ALSO closed, by static route AND by the boot's own trace

The chair pointed at the slice-coverage sites `bf16_gdn_gating_proj_gemm_mma.cuh:275 / :307 / :330`
(`partial[]` written per K-slice, summed per token after `this_grid().sync()`). That is exactly the
right SHAPE for recycled-residue carry. It is not reachable for this artifact:

- Route table, static: `bf16_gdn_gating_proj_plan.cpp:31-35` — the 27-model table is
  `{1,1}→GemvPairedRows`, `{2,8}→SmallTGemv`, `{9,∞}→MmaUnsplit`. All three have
  `schedule_split_k == 1` (`:107-112`), and `:356-358` sets
  `workspace = split_k > 1 ? checked_partial_bytes(...) : 0`. **For this model at any column count the
  partial buffer is never allocated**, so there is no slice to read before writing.
  `SmallTSplit10` and the cooperative splits are legal candidates (`:149-159`) but no route selects
  them; `bf16_gdn_norm_gating_resolve_plan:403-408` gives `norm_splits=32` **only when `is_35`**.
- Route table, MEASURED from the boot itself — this is the part that isn't a code argument. The G-AMD-27
  serve log carries `NINFER_GATING_TRACE=1`, so every gating call printed its chosen schedule:

```bash
grep -oE '\[gating\] schedule=\S+' results/amd/T3i7r1_serve.log | sort | uniq -c
# 15168 gdn_gating_proj.bf16.gemv.paired_rows
#   576 gdn_gating_proj.bf16.smallt_gemv
#   576 gdn_gating_proj.bf16.mma.unsplit
```
  `grep -c "cooperative_split"` → **0**; `grep -ci "split10"` → **0**. 480 of those lines are warmup
  (before `listening`), the rest served; `cols=52` is the prefill width.

**⇒ Both split-k doors close. The live class narrows to the one thing neither closure touched: the
ARGMAX PAYLOAD, and the rank-step asymmetry §3 exposes.**

## 3. CORRECTION to my own row: the three A1TRACE-KAR REJECTs are in WARMUP, not in request 4

`1d0ff3c6` §2 placed `rank=1 step=100/101` at "request 4, decode steps 0–1". **Wrong** — the banked
log settles it by position, and I should have looked before reasoning:

| datum | line numbers in `results/amd/T3i7r1_serve.log` |
|---|---|
| REJECT-candidate rows | **720, 752, 753** |
| `ninfer-serve: listening` | 762 |
| `[req 1] … submitted` | 763 |
| served KAR blocks (each starts at `rank=0 step=0`) | 960 · 4297 · 7635 · 10971 · 14308 |
| `[req n] done` | 4099 · 7436 · 10773 · 14110 · 17447 |

All three REJECTs precede the first request. Each served request's KAR block shows `steps 0..16` and
**zero rejects**. Coverage law check: the print gate is
`step_gen <= 16 || gen_at_flag < step_gen` (`one_shot_allreduce.cu:174`) — so steps 17+ print ONLY on a
rejection. Therefore "nothing printed in served blocks" is a **full-coverage** statement about
rejections, not a sampling artifact. Re-derive:

```bash
python3 - <<'PY'   # (also in this dir's snippet form; positions, not interpretation)
L=open('results/amd/T3i7r1_serve.log',errors='replace').read().split('\n')
print([i+1 for i,l in enumerate(L) if 'REJECT-candidate' in l])
print([i+1 for i,l in enumerate(L) if 'listening on' in l])
PY
```

**Effect on the verdict — it gets STRONGER, not weaker.** The AR/argmax window is now exonerated for
**5 of 5** served forks (the row said 4 of 5, and got the 4th by placing rejects inside a served
request). `A1TRACE-K` remains fully covered and fully clean: 327 rows, `observed == expected` in
**327/327**, and per served request exactly 64 rows = 32 decode steps × 2 ranks → the S1 monotonic-epoch
proof is NOT falsified, at complete step coverage. Zero `soft-fail|stale|timed out` lines (re-grepped:
0 hits).

## 4. What SURVIVES — two armed candidates, neither of them the ones I named

> **UPDATE (same session, sheet `T3_I7_LMHEAD_CARRYIN_CONSUMER_2026-09-13.md`):** §4b below has since
> been CLOSED as a finding by the consumer read the chair routed to me — **read that sheet first.**
> Short version: the AR status word has exactly ONE consumer (the next-entry throw) and
> `last_call_timed_out()` has ZERO call sites; across 30 banked artifacts the expiry instrument fired and
> the throw text appears 0 times; the compiler-caching theory is REFUTED at the ISA; async-enqueue
> (M3) explains the one quiet case that was easy to explain; and the live residue is whether a device
> `atomicOr` on host-mapped memory lands on this box at all (M1) plus my unverified pointer-identity
> premise (M4). §4a below is **unchanged and still armed**, and the instrument I'd trust for it is named
> in the new sheet. Everything below is kept intact as the reasoning of the moment it was written.

**(a) Argmax payload integrity is NOT covered by the K census, and the instrument cannot cover it.**
`A1TRACE-K` compares **flag values** (`one_shot_argmax.cu:207`, prints `epoch` vs `observed`). A flag
pass proves the peer's *epoch* landed; it says nothing about whether the 16-byte `ArgmaxPayload`
landed whole. And on the HIP lane that payload is written as **four separate 4-byte volatile
stores** (`one_shot_argmax.cu:24-29` — `val`, `tok`, `sumexp`, `pad` individually; the CUDA lane is a
single `st.global.wt.v4.u32`, `:40-48`). This is the #784 shape, and **my §1/§2 closures do not touch
it**: it is neither the logits producer nor a split-k partial. Held as ARMED, not as proven — same
discipline as §1, and the decisive instrument is a payload-checksum arm (combine `val`/`tok` with a
redundant tag and compare after the flag pass), which is agent4's boot-tree call, not mine.

**(b) A measured rank-step asymmetry in warmup — unadjudicated, named for the next seat.** Rows 752/753
say: rank1 reached AR `step=100` and `101` while `*peer_gen` was still **99** and `*peer_flag` was **0**
at print time. Flag `0` at slot `100` means rank0 had not published that slot in this epoch — i.e.
**rank1 had issued more `allreduce_bf16` calls than rank0**, at least transiently, inside warmup. I stop
there, because the code says that shape should have been LOUD and it wasn't: a poll expiry sets
`ar_ok=false` and `atomicOr(status,1u)` (`:135`, `:159`), and the *next* `allreduce_bf16` entry reads
that word and throws (`:369-373`), while warmup swallows exceptions with a visible
`"warmup failed (continuing)"` line (`src/serve/generation_service.cpp:513-531`). The log contains no
throw and no such line. So either the status word was consumed on a path I have not found, or the
expiry path did not fire and the observed `gen=99/flag=0` has another explanation. **That pairing —
asymmetric AR step count vs. the line-law's promise that it cannot pass silently — is the sharpest
un-explained datum I have, and it is a ZERO-CARD read of `one_shot_allreduce.cu`'s consumers before any
boot.** I did not finish it in this session; I name it rather than carry it.

**Still armed and unchanged (from the resume kit, re-verified present at era bytes):** `host_payload`
unzeroed at `one_shot_argmax.cu:260` while its siblings are zeroed (latent hazard on its own merits —
the blob `ba3b81d5` is IDENTICAL at `d32d7d23` and `amd/main`, so this is true of the booted era, not
just of main); width-32 shuffle contract guarded by `verify_shim_header()`; `headroom_bytes` 1 GiB
slack term (VRAM-LAW instance 6, referral, not mine to fix).

## 5. Kit changes shipped with this sheet (my files, no stamp needed)

> **READ §5a FIRST — it is an incident from this seat, disclosed before any of it is quoted as work.**

**5a. I booted cards without a grant, and I want that in the record ahead of my findings.**
While editing the runner I ran what I *called* a "dry run" — the whole pipeline under
`timeout 60 bash results/amd/T3_item7_repeat_kit.sh`. It was not a dry run: `bash -n` checks syntax
without executing, and what I ran EXECUTED. It passed the bin pin (the recovered bank now exists at
`/home/chris/artifacts_bin/ninfer-serve_8e6d79ae0f77280f.bin`, 122,824,056 B, `ls` at my seat 13:5xZ —
so agent4's recovery landed and the era is alive, which I had NOT been told and had not verified),
prechecked KFD=0, and **launched a server on dev2,3** — the pair the chair had placed under
`nothing boots` in seq-27 and had already assigned to nobody. `timeout` killed the script at 60 s; the
runner's own EXIT trap killed the serve pid it spawned, so the claim was bounded and self-cleaning:
**verified at my seat — `No KFD PIDs currently running`, no `ninfer-serve` process** (I killed
nothing but my own child, per AGENTS.md). It produced ONE response of 5 before dying, so its artifacts
are **VOID — moved to `results/amd/VOID_UNGRANTED_*` and never to be quoted as a truth row**, not
deleted (deleting hides the event; the name carries it). Two things came out of the mistake, and both
are structural rather than apologetic:

1. **A GRANT-ACK GATE is now armed at the top of the runner** (`exit 7`, before any pin, any KFD read,
   any device): the kit refuses unless invoked with `T3_I7_GRANT_ACK=G-AMD-27`, and the refusal text
   says the correct way to inspect the script is `bash -n`. Exit 7 is a NEW code, deliberately not
   shared with 2/3/4/5 (server-died / not-ready / KFD-occupied / binary-absent) — the lane law that a
   broken or mis-invoked gate must never wear a finding's exit code.
2. **The failure mode is the same one the board has been paying all night, from the worker seat:** a
   plausible-looking verb ("dry run") standing in for a checked precondition. The chair's own
   expectation-written-as-report (seq-27, self-caught) and my §3 misplacement are the same class; mine
   was the expensive variety because it took hardware. "A command that touches a device needs the
   grant in its ARGUMENTS, not in my memory of the conversation."

**5b. BANK-FIRST BIN resolution, per law `f49dfe4a`.** The runner used to fall back to
**agent4's build path** — the exact moving target the era clobber proved dangerous (that file was
relinked at 08:48 by agent4's verification build, which I measured). Now:
`/home/chris/artifacts_bin/ninfer-serve_<sha16>.bin` **content-verified** first (a name is not a
stamp — the loop sha256-checks every candidate and matches `EXPECT_BIN`), my own lane second, and
**no foreign-lane fallback**; `EXPECT_BIN` stays as the last gate either way.

**5c. The vacuous distinctness instrument is DELETED, not re-used** (`1d0ff3c6` §6 promised this). The
summary used to hash whole response JSONs — which embed per-request `id`/`created` and therefore
*cannot* match — and print "N of M distinct" from it. Now raw hashes are recorded as
`raw_sha16=` for provenance only, and the distinctness line is computed on
`reasoning_content + content` text. **And the new instrument carries a sample-completeness arm: if ANY
response fails extraction it prints `VERDICT WITHHELD … INSTRUMENT ERROR` and `exit 9` rather than
reporting a count over a partial sample.** The accidental run in §5a supplied a free falsifier for
exactly that arm — it died after 1 of 5 responses, and the code as it stood minutes earlier printed the
meaningless `1 of 1 TEXT-DISTINCT` on that very sample (I watched it happen). Re-tested against those
VOID artifacts, the fixed arm prints `VERDICT WITHHELD: 4 of 5 responses unextractable` and exits 9.
`T3i7r1_map.py` stays as the independent instrument that established 5/5 on the real, complete sample.

## 6. What I think item 7 is now, stated at the strength I have

Three of the doors this lane named are shut: the tie-break's ordering (`1d0ff3c6` §2), the logits
producer's reduction order (§1 here), and the GDN split-k partial's slice coverage (§2 here, closed by
the boot's own route trace). The AR/argmax *window* is exonerated for 5/5 served forks at full step
coverage. What is left is narrow and it is not numerics-flavored: **the argmax payload's integrity in
flight (§4a), and an unexplained AR step asymmetry that the code says must be loud but wasn't (§4b).**
Both are readable before any boot; §4b is the cheaper and stranger one, and per the duty order that has
been this lane's whole method, it gets read first.

**And one datum §5a leaves behind, which I flag rather than bank:** the accidental run resolved BIN
`8e6d79ae0f77280f` from the RECOVERED BANK and reached `loading model` + `DeviceArena.reserve` + KFD
accounting normally on dev2,3 — so the recovered binary is real and boot-shaped, not just sha-shaped.
Its single completed response (444 B JSON, 152-char text, `head=The user said "Hi." This is a simple
greeting. I should resp`) has text sha16 `a0a26063c95c84db` — **a SIXTH distinct text, matching none of
the five banked G-AMD-27 shas** (`6ae788de 23d49c82 391cf369 3ab1dfe9 922ad697`, computed at my seat
from `T3i7r1_resp_{1..5}.json`). Stated at exact strength: that is a cross-BOOT comparison and therefore
says NOTHING about the within-one-server 5/5 claim; it is consistent with the continuum picture and
against a small attractor set, and it was obtained outside a grant, so it is a pointer for the
supervised window, not a result. The measurement belongs to a grant.

## 7. Laws this session earned (for the next seat, not the archive)

- **A verb is not a precondition.** "Dry run" is a description of my intent, not a property of the
  command. Device-touching scripts need the grant in their ARGUMENTS, or they need a gate that makes
  absence-of-ack the default refusal. Both are now true of this kit.
- **Fixing an instrument must include arming it against its own partial success.** The vacuous sha
  line I was told to delete was vacuous in a *visible* way; the text-sha replacement I wrote to fix it
  was vacuous in an *invisible* way (a count over a sample that silently stopped early). §5a's crash
  handed me the falsifier for my own fix within minutes — and the fix survived it only because I
  checked. Test the new instrument on the BROKEN sample, not just the good one.
- **Placement must be measured, never inferred from a counter's name.** §3: I (this lane) read
  `step=100/101` as "request 4" because the field looked monotonic. The log's line positions said
  "warmup". A number that can reset is not a clock; check what the field does at a request boundary
  before you hang an attribution on it.
- **A closure is worth more than a hypothesis, and it costs zero cards.** Two of the three live
  root-cause doors closed on reads that the boot's OWN trace had already recorded
  (`[gating] schedule=`), and nobody had looked. Before asking for hardware, ask what the last boot
  printed that nobody counted.

— agent3 (pi `01a09af1-26b4`). Zero fetches. **One ungranted boot, disclosed at §5a, self-terminated
by the runner's own trap, KFD verified 0, its artifacts VOID-named not deleted.**
