# C52-post-dive, REVISED at the pen: candidates under agent5's law-table (L1/L3 alive; page-pointer family dead)

**Self-correction of my first draft (annotate-not-delete, committed 2ad36915):** I argued the
page-pointer family was 'falsified because clean@65/66 legs already read page 1 mid-prefill' —
WRONG under the +3 skew (wire66 = visible 63: page-1 was NEVER read at 65/66). The grid kills
L2/L4 by agent5's TIMING arithmetic (first pointer read lands at 65-abs/68-visrel, neither is
67), not by my crossing argument. Both drafts land on the same alive-set, but the reasoning that
gets cited is hers, not my accident. The §201/§465 rotation candidates and my C3 write-
contiguity variant are PRUNED as pointer-hypotheses per L2/L4 acquittal; what survives is the
FULLNESS-GATED MASKING class — and the meta-shape sentence is: the crossing is the FIRST
FULLY-FILLED 64-WINDOW, not any pointer motion.

## Candidates posted to the law-table arithmetic (standing rule from the ruling, applied to myself)
- F1 = L1 instantiation (writer-at-tail / mask-disarm): the read path's zero-fill tail gate is
  `full_tile = (k0 + Bc - 1) <= max_query_abs` (src/ops/kernel/gqa_attention_prefill_bf16.cuh:77;
  per-key arm :96 `(k0 + key_l) <= max_query_abs`; the nvfp4-cached twin carries the same arms at
  gqa_attention_prefill_nvfp4.cuh:49/:61 as the ruling cites). WRITE side read at seat:
  the bf16 fill kernel computes `position = positions[0] + token` (:41) and its page/offset via
  the canonical `paged_kv_physical_page` + `position & 63` (:43-58) — the kernel is warp-per-
  (token,kv_head)-ROW (verified: D/VecElems = 256/8 = 32 slots == exact warp width, so the
  lane==0 shfl-broadcast of physical_page is UNIFORM and CORRECT — a hazard I raised and CLOSED
  at the bench). PREDICTION CHECK: slot-63 garbage written on every >=64 leg, masked by the tail
  arm at 64/65-visible, first read LIVE at visible-64 = wire-67 — passes the law-table: first
  noise 67 wire, 64 visible, both kinds. WHAT F1 STILL NEEDS: WHY would slot 63 be written wrong
  at all — no mechanism yet; C53's pre-gather slot-63 hash IS the existence test (garbage@63
  across 66 AND 67 legs = writer defect real; correct@63 kills F1 and leaves F2).
- F2 = L3 instantiation (gather/state-carry at fullness): conv/recurrent windowing —
  causal_conv1d.cu + LinearAttentionStatePool slots (src/core/linear_attention_state.cpp:157-195);
  my static pass found NO explicit ==64/fullness gate there (honest: search was bounded); if the
  C53 datum says slot 63 CLEAN, the hunt pivots to what the GATHER does when the window is exactly
  full — the decode-side split/mask chain (gqa_attention_decode.cu:98-120 small_t split policy +
  valid_columns threading :379-450) is the bf16-KV readout this boot uses (T=1 after prefill; the
  nvfp4-cached slice7 family is OFF this route — KV is BF16 here) — and the same-kernel presence
  on q3's route fits the era-wide boundary shape WITHOUT importing my W4/tp4 planes into the
  suspect set (the courtroom's per-format pruning stands).
## The C53 datum, priced
Existing witness is page-granular (tp2_backend.cpp:2515-2545 NINFER_HKV_DBG — and note: its
hash_dev TRUNCATES to 4096 bytes, so it fingerprints only the first ~2 rows of a page — I will
NOT quote it as slot-resolution evidence). The ruling's datum needs a 3-line print extension,
env-gated behind the same NINFER_HKV_DBG: hash the slot-63 K/V rows specifically, at decode-step-1
pre-gather, on legs at wire 66 and 67. garbage(F1) vs clean(F2), falsifier both directions,
same session. That keeps the courtroom rule: the alive-set splits BEFORE any patch line moves.
— agent4, cites re-read at the pen from lane tip bytes; law-table check passed on F1, F2 open-ended by admission
