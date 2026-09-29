# agent3 — §4b closes: what consumes an AR expiry, and the corpus fact that it never has

Follow-on to `T3_I7_LMHEAD_REFUTAL_AND_CARRY_IN_2026-09-13.md` (same session, pi `01a09af1-26b4`).
That sheet armed two candidates: §4a (K-blind 4-store publish) and §4b (a warmup AR expiry that "cannot
pass silently" yet did). The chair routed §4b to me and agent5 flagged the swallow-path alternative.
**This is the consumer read. Zero-card, zero-boot, zero-fetch. It closes §4b as a FINDING and leaves the
mechanism OPEN between two named candidates, each with its cheap discriminator.**

## 1. The consumer chain, read end to end (the answer to "what consumes those flags?")

| link | locus | measured |
|---|---|---|
| expiry sets the bit | `one_shot_allreduce.cu:135` (flag poll) and `:159` (gen poll): `atomicOr(status, 1u); ar_ok = false;` | present |
| block skips its work | `:180-185` — `if (!ar_ok) atomicOr(&s_fail,1u); … if (s_fail != 0u) return;` → **this call produces NO output** | present |
| device writes via | `:408` `reinterpret_cast<unsigned int*>(impl_->dev_status_view[rank])`, alias from `hipHostGetDevicePointer` of a `cudaHostAlloc(MAPPED)` word (`:301-305`) | present |
| **the ONLY consumers** | `:369-373` next-entry `if (*host_status[rank] != 0) { *host_status[rank]=0; throw …"prior output invalid" }` and `:347-351 last_call_timed_out()` | — |
| is `last_call_timed_out()` ever CALLED? | `grep -rn last_call_timed_out src/ apps/` → 6 hits, **every one a definition, declaration, or comment; ZERO call sites** | verified |
| does a per-request reset clear the bit instead? | `reset_step` (`:336-345`) resets `rank_step`, `host_epoch`, `dev_epoch`, and every slot flag — **it does NOT touch `host_status`** | verified |
| do the AR call sites differ by rank? | all 9 `allreduce_local_bf16` sites in `text_context_impl.h` are guarded only by `tp_group_ != nullptr && tp_rank_ >= 0` — same predicate both ranks; no rank-conditional AR call exists | verified |

So by the code's own design the sequence is: **rank1's step 101 entry MUST throw**, because rank1's
step 100 already set the bit and nothing between them clears it. Two entries were consumed (step 101's
kernel demonstrably RAN — it printed its own REJECT row at serve.log line 753), so **the check at :369
failed to see a bit the same rank's previous kernel had already written.** That is the finding: not
"the throw was swallowed on a path I haven't found" — the throw's PRECONDITION demonstrably did not
observe its own input, one call later, in the same rank.

## 2. The corpus fact, with window and filter named

    window: results/ (whole tree)   filter: grep -rl  (file-presence, not a count)
      artifacts containing a REJECT-candidate row .......... 30 files
      artifacts containing the throw text .................  0 files
      (patterns: "previous allreduce timed out", "prior output invalid",
                 "that token's output is stale", "timed out waiting for peer")
      docs/ tree, same patterns ...........................  0 files
    window: T3i7r1_serve.log only
      "warmup failed" (the Warning that generation_service.cpp:528-530 would print) ... 0

Re-derive in one command from the repo root:

```bash
grep -rl "REJECT-candidate" results/ | wc -l
grep -rlE "previous allreduce timed out|prior output invalid|token's output is stale" results/ docs/ | wc -l
```

**Completeness note on the KAR census (I checked the rows that are NOT rejects, not just the rejects):**
203 rows carry `flag_obs=1`, 3 carry `flag_obs=0` — and the 3 are exactly the REJECTs at 720/752/753, all
warmup. Of the 203 ACCEPTs, exactly **one** has `gen_at_flag != step_gen`: line 721, `rank=1 step=1
gen_at_flag=2` — the peer running one call AHEAD, the harmless direction (a later commit stamp still
implies this call's payload is visible), and also warmup. So the served blocks contain no anomaly in
either field, in either direction.

**Across thirty banked artifacts in which the expiry instrument fired, the loud path it exists to arm
has produced output ZERO times.** The "never silent" guarantee is unverified in precisely the situation
it was written for — and this lane's own row (`1d0ff3c6`) cited that guarantee as reason to trust the
instrument. The chair's seq-39 framing was right: agent5's swallow-path alternative is the correct
direction, but the swallow is one link EARLIER than anyone guessed — between the device write and the
host read, not between the throw and the logger.

## 3. Two candidates for the dead link, and the discriminator for each. Neither is proven.

**(M1) The atomic never lands in host memory on this box.** Every other field in this protocol crosses
the device→host boundary with a **plain volatile store** — flags (`*my_flag = expected_epoch`, `:119`),
gen (`:117`), payload (`st_volatile_payload`, four stores) — and all of THOSE demonstrably work. The
status word is the single field moved with an **atomic**, and `status` is even declared non-volatile
(`:75 unsigned int* status`). Consumer PCIe hosts commonly ship with **PCIe atomic operations disabled
by platform firmware**, in which case a GPU atomic to system memory is dropped or returns garbage while
ordinary reads/writes to the same buffer work fine. If that is what this box does, the entire
line-law soft-fail surface (AR `:135/:159` AND argmax `one_shot_argmax.cu:192`) is structurally dead,
on every boot, on this hardware — and has been all along, which is exactly what the 30-versus-0 tally
looks like.
- **Discriminator (one boot, ~60 s, and it is NOT an item-7 boot — it is an instrument boot):** write
  the status with a plain `*((volatile unsigned*)status) = 1u;` alongside the `atomicOr`, keep
  everything else identical, and fire the SAME shape that already produces REJECT rows (warmup is
  enough — nothing client-side is needed). If the next-entry throw now appears, M1 held. Note the
  sha-pin consequence honestly: this changes the binary, so it is a NEW era, not a G-AMD-27 replay.

**(M2) — RUN AND REFUTED, compile-only, zero-card.** The hypothesis: `host_status` is `int*` (`:253`),
not `volatile int*`, so an optimizer might cache the `:369` read across the enqueue and never re-read the
device's write. I tested it directly rather than reasoning about it: a minimal TU replicating the exact
shape (`if (*impl->host_status[rank] != 0) { *impl->host_status[rank] = 0; thrower(); }` before AND after
an opaque `launch()` call), compiled at -O0/-O2/-O3.

**Result: the word is RE-LOADED after the call at -O2 and -O3** — `call launch@PLT` followed by
`movq (%rbx,%r13,8),%rax` then `movl (%rax),%edx`. The call is an optimization barrier over memory, so
the compiler cannot carry a cached value across it. **M2 is dead: the read side is sound.** Which is
what makes M1 the last candidate standing, and why I looked for a way to test M1 without a card:

**(M1), and the sysfs route I tried (documented so the next seat doesn't repeat it).** This host exposes
**no** amdgpu atomics knob — `/sys/module/amdgpu/parameters/` yields `pcie_gen2`, `pcie_gen_cap`,
`pcie_lane_cap`, `pcie_p2p` and nothing about atomics; there are **no** `*atomic*` nodes under
`/sys/bus/pci/devices`. And the obvious follow-on probe is **also closed to an unprivileged seat**:
`lspci -s 05:00.0 -vv` prints **zero** `DevCtl`/`LnkCtl`/`Express` capability lines without root, so the
`AtomicOps` block cannot be read from here at all (the four Vega endpoints enumerate fine — `05:00.0`,
`08:00.0`, `0d:00.0`, `10:00.0` — it is the config-space *detail* that needs privilege). `dmesg` is
likewise unreadable (documented host limitation: no passwordless sudo).
**So M1 is NOT decidable from this seat by any no-card route available to me** — which is a result about
the hunt, not a dead end: it means the cheap discriminator is the probe cell in the ordering below, and
the sudo `lspci` line is offered to any human at the keyboard.

**(M3) — the candidate my own reading produced, and it is why the picture is not simply "the atomic is
dead".** The deferred check is late BY DESIGN: `allreduce_bf16` enqueues asynchronously, so rank1's
step-101 entry check can run while step-100's kernel is still inside its 65-80 ms poll loop and has not
written anything yet. **That fully explains the printed step-101 row existing with no throw** — so my
original "cannot pass silently" framing was sharper than warranted, and I own that. What M3 does NOT
explain is the rest: the plain decode loop stream-syncs every token (`tp2_backend.cpp:2477
CUDA_CHECK(cudaStreamSynchronize(s))`), so the first post-sync entry should observe the bit and throw —
and across 30 artifacts it never once did. **So the anomaly narrows from "never loud" to "never loud
after a sync", and that residual is M1's or a broken premise of mine.**

**(M4) my premise, corrected in the open.** I claimed `gen_final=99 < step_gen=100` proves the `:159`
break executed, hence `atomicOr` ran on the device. `gen_obs` is a post-loop local, so the break did
happen and the device-side write did execute. What I have **NOT** verified is that the object the kernel
writes (`dev_status_view[r]`, via `hipHostGetDevicePointer` at `:304-305`) is the object the host reads
(`host_status[r]`). On ROCm those are normally the same address for host-alloc'd mapped memory — and if
that identity does NOT hold here, it is M1 wearing a different hat. **Held as: device write executed,
delivery to the host-visible word unverified.** That is the honest line, and it is exactly why I name no
fix.

**Ordered cheapest-first, for whoever picks this up:** (1) `lspci -vv | grep -i atomic` on the Vega
endpoints **run with sudo by a human** — unreadable from my seat (shown above), and it may settle M1 with
zero cards and zero boots; (2) a **CPU-side host-mapped-atomic probe cell**: `hipHostAlloc(MAPPED|
PORTABLE)` a word, launch a ONE-thread kernel doing `atomicOr` on the `hipHostGetDevicePointer` alias,
host reads back, and the same cell drives a **volatile-store control** into a second word — this decides
M1-vs-M4 and needs **no server, no artifact, no era pin, no client**: seconds of an idle device context,
an order of magnitude cheaper than any boot. **I am not requesting even that — this session holds to zero
boots and zero cards, and a stamp ask belongs on the chair's table, not smuggled in mid-hunt**; (3) the
instrument boot in M1 only if (1) and (2) converge there. Sha consequence stated up front: it changes the
binary, so it is a NEW era, not a G-AMD-27 replay.

## 3b. STATUS OF THIS SHEET AFTER agent4's G-AMD-30 (annotate, never delete)

**M1 is now MEASURED FACT, not candidate** — device `atomicOr` → host-mapped never crosses (5/5; a
volatile store to the same word crosses 5/5; alias identity YES, so my **M4 freebie is settled** and
M1 stands whole). The 30-versus-0 census above is explained exactly as this sheet predicted. One
correction to my own §3 confidence: I wrote that M2's refutation left M1 "the last candidate standing"
— correct — but I did NOT foresee that the *new* instrument agent4 built to replace the dead channel
would raise its own finding on the same `atomicOr` (`one_shot_argmax.cu:249`), inheriting the defect one
layer down. That row, and the voided-claim ledger sweep this sheet triggered, are in
**`T3_VOIDED_INFERENCE_INVENTORY_2026-09-13.md`** — including my OWN `8afd84c3` refutation of the
`host_payload` mechanism, which that inventory voids: **a mechanism this lane killed on a grep of a
line that cannot be emitted is not killed, it is unmeasured.** Read that sheet before quoting §4's
channel table below.

## 4. What this does and does NOT do to item 7

- **Does NOT move the verdict.** `1d0ff3c6` stands: per-request nondeterminism is live, 5/5 distinct on
  text-only comparison, independently matched by the chair's own measurement.
- **Does NOT re-open the AR exoneration of the served forks.** §3 of my previous sheet placed all three
  REJECTs in WARMUP by line position (720/752/753 < `listening`@762 < first submit@763), and the served
  blocks each print steps 0..16 with **zero rejects** — with the print gate `step_gen<=16 ||
  gen_at_flag<step_gen` (`:174`) making any later rejection printable at any step, that silence is full
  coverage. The 5/5 exoneration is untouched, and it survives this finding for the reason in the next
  bullet.
- **DOES knock one support out from under that exoneration, and the honest way to say it is mine to
  say.** If M1 or M2 holds, the AR *expiry* instrument is dead-on-arrival on this box, so "no REJECT
  rows in the served blocks" only exonerates the AR window **as far as that instrument can see**. What
  still carries the exoneration independently is the K-family: **327/327 `A1TRACE-K` rows with
  `observed == expected`** — plain volatile flag reads, demonstrably working, 64 rows per served request
  = 32 steps × 2 ranks, and per the S1 comment (`one_shot_argmax.cu:339-357`) a future/stale pass is
  unreachable by dependency, so a hit there would mean a broken pairing. That argument does not depend
  on the status word at all. I flag the scope limit rather than let a still-good conclusion rest on an
  instrument I just showed may be dead.
- **The warmup consequence is real but not item 7's cause.** If rank1's step 100/101 ARs really
  expired, those two calls skipped their combine → rank1's residual for them is missing rank0's partial
  → silently-wrong hidden state **in warmup**, whose output is discarded. Consistent with all served
  requests being clean, and with `1d0ff3c6`'s exoneration of the served forks. Most likely benign
  explanation for the asymmetry itself (not for the dead bit): async enqueue means a merely-SLOW peer
  (rank0 doing host-side prefill work longer than the ~65-80 ms poll ceiling) reads as an expiry. That
  would make the ceiling the real defect for warmup and the dead status word the real defect for
  everything. Both belong to agent4's file lineage (#784), not mine — this is a referral with a
  discriminator attached, not a patch proposal.

## 5. §4a: the instrument I'd trust, named as the chair asked (no patch, no boot)

The tear is payload-internal and `A1TRACE-K` compares flag epochs only (`one_shot_argmax.cu:208`) — it
cannot see a mixed-half payload, so a different instrument is required before anyone talks about
fixing (a). The field already exists: `ArgmaxPayload` has `float pad` (`:14-19`), and the CUDA lane
already publishes all four components as one 16-byte `st.global.wt.v4.u32` (`:40-48`) while the HIP
lane uses four independent volatile stores (`:24-29`) — which is where a half-payload can be consumed.

**Trustworthy instrument: a self-describing payload, verified by the consumer, printed on mismatch.**
Write `pad` as a checksum over the other three components *in the same publish* (e.g. the raw bit-sum
of `val`, `tok`, `sumexp`), and have the reader, immediately after `ld_volatile_payload` (`:212`),
recompute and print `[A1TRACE-PAYLOAD] rank=… t=… tag_obs=… tag_exp=… verdict=…`. Properties that make
it the right instrument, each one buying off a mistake this board has already paid for:
- **Detects exactly the event in question** — a torn/stale/mixed-half payload — which an epoch compare
  provably cannot, instead of inferring it from a symptom.
- **Self-validating rather than silent**: like §3's finding shows, an instrument whose negative is the
  only output is unauditable. So the tag must be reported as a CENSUS (N verified / M mismatches) in
  the release row, not merely printed on a hit — a `grep -c` must be able to distinguish "checked and
  clean" from "never checked". (The same law that made me re-grep for the throw text here.)
- **Cannot mask the event it measures** — the reader still consumes what it consumed; the tag ride is
  payload-invisible. It DOES change the binary sha, so it is its own era and its own truth stamp, and
  must not be folded into a G-AMD-27 replay. That constraint is the chair's seq-8(2) rule, restated so
  nobody discovers it at the release row.
- **Positive control required before anyone believes its silence**: deliberately corrupt one component
  on one rank in a CPU unit cell and show the mismatch line fires — a mutation arm proves sensitivity,
  never coverage (this lane's most expensive law).

## 6. Files touched, and the state of the lane

`docs/amd/T3_I7_LMHEAD_CARRYIN_CONSUMER_2026-09-13.md` (this sheet). The previous sheet gets a
one-line pointer from me at commit, not a rewrite — §4b there said "the decisive next read is
`one_shot_allreduce`'s CONSUMERS, zero-card"; it is done and the answer is in §1-§2 here.
**Cards: none held, none used. Zero grants, zero fetches, KFD 0. No boot proposed by me at all** — the
M1 discriminator needs a stamp and I am explicitly not asking for one; M2 is compile-only and I take it
next unless the chair re-sequences.

— agent3 (pi `01a09af1-26b4`).
