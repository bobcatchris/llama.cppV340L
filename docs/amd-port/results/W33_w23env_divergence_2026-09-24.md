# W33 w23env served-divergence root cause: the EXACT-VOID never measured anything - 2026-09-24

Desk W23ENV-DIVERGENCE (2nd dispatch), main tree amd/v340-port-v2, ZERO-GPU.
Question of record: the postu1 w23env window verdicted EXACT-VOID with the
ratchet "arm VOID - text_sha256 diverged" (/home/chris/postu1_window_results.txt)
although every gate in the arm stack was previously validated bit-exact. Which
env diverges, by what mechanism, and what class is it?

VERDICT: no divergence was observed, because no text_sha256 was ever measured.
The EXACT-VOID is a harness false-positive: the window ran all four cells
--decode-only, which structurally cannot produce the determinism guard the
w23env adjudicator keys on, and the adjudicator treats a MISSING guard the
same as a FAILED one. The kernel gates stand exonerated on standing evidence;
the measured perf case for promotion is noise-level and not worth pursuing.

## 1. A2: what the four cells actually contain

Cells of record (final battery per cell; each also holds one earlier jsonl
line from a first dispatch at older commits):

| cell | env | decode_tps | verdict | determinism line | text sha |
|------|-----|-----------|---------|------------------|----------|
| postu1_w23r1 | none (BASEENV) | 24.69 | PASS | absent | none |
| postu1_w23a1 | IQ4XS_SHARE + IQ3XXS_S2R + IQ3S_S2R | 24.80 | PASS | absent | none |
| postu1_w23a2 | same arm env | 24.73 | PASS | absent | none |
| postu1_w23r2 | none (BASEENV) | 24.68 | PASS | absent | none |

- grep across all 4 cell logs + .server logs: zero matches for
  determinism / text_sha256 / sha256 (case-insensitive).
- The jsonl witnesses carry no text field at all; decode_guard records only
  tps and draft-chain counters.
- The dispatch's key question (a1 sha == a2 sha? r1 == r2?) is unanswerable
  from these cells: no sha exists anywhere in the window.

Indirect in-window text witness (greedy decode_guard, temperature 0.0,
fixed 10k prompt, n_predict 128): draft_n=126, draft_n_accepted=84,
accept_ratio=0.66667, mean_len=3.0, prompt_tokens=7857,
completion_tokens=128 - IDENTICAL in all four cells. Divergent greedy
trajectories over 128 tokens would not reproduce identical acceptance
counts four times; this is draft-chain invariance across arm and control,
consistent with identical served text. Not proof - witness only.

Paired decode (matches the machine line mean +0.32%):
PAIR1 a1-r1 = +0.11 t/s (+0.45%), PAIR2 a2-r2 = +0.05 (+0.20%).
Noise yardstick from inside the window itself: cell w23r1 ran its battery
twice in one boot at one env - 24.41 then 24.69, a 0.28 t/s within-boot
repeat spread. Both paired arm deltas are smaller than that spread. The
W23 static prediction (-3.4 ms/round -> ~+2.6%) is not in the data.

## 2. The verdict mechanism (file:line, owner-owned files, read-only)

1. run_post_u1_battery.sh window_w23env() launches all four cells with
   battery_args --decode-only (lines 440-447) while the w23env adjudicator
   config declares exactness=True (line 271, comment at line 33 promises
   "any arm cell determinism_guard (text_sha256) not PASS -> arm VOID").
2. docs/amd-port/tests/guard_battery.py:895 - the determinism guard runs
   only if NOT (prefill_only or decode_only or canary_only or needle_only).
   Under --decode-only it never runs, never prints, never hashes.
3. run_post_u1_battery.sh adjudicate(): det is parsed with
   re.search(r"^(\w+)\s+determinism_guard", log, re.M) -> None for every
   cell; the w23env branch `any(cells[t]["det"] != "PASS")` is True for
   None, so verdict = EXACT-VOID and the ratchet string printed is
   "text_sha256 diverged" (lines 360-363). Reproduced read-only on the
   actual logs: all four cells det=None -> EXACT-VOID.

Two design notes compound the mislabel:
- the determinism guard is a WITHIN-cell check (two consecutive greedy
  runs in the same boot, guard_battery.py:597-643); it cannot measure
  arm-vs-control divergence. The campaign's cross-cell exactness law
  (W7 A4 / E-102 era) compares the guard's sha ACROSS cells by hand.
- the guard's env-gate engagement lines print at GGML_LOG_INFO
  (mmvq.cu:1066 via mmvq_share_env_flag), and the served log drops ggml
  INFO (dossier Wall 8: engagement must print at WARN). Consistent with
  the mining: zero MMVQ gate lines in any w23 cell or in any earlier
  combowin cell that used these gates; only the WARN-level FATTN line
  appears (320x in w23a1). The arm therefore also lacks served
  engagement evidence - engagement rests on code reading + the paired
  deltas + the W7 A4 host proofs.

## 3. A3: served-path trace of the three gates

Env parse (all static-init on first MMVQ launch of the type):
- GGML_CUDA_MMVQ_IQ4XS_SHARE -> mmvq_share_env_enabled(IQ4_XS),
  mmvq.cu:1082-1085.
- GGML_CUDA_MMVQ_IQ3XXS_S2R / GGML_CUDA_MMVQ_IQ3S_S2R ->
  mmvq_s2r_env_enabled, mmvq.cu:1147-1154 (both reuse
  mmvq_share_env_flag, hence the "s2r decode-once share enabled" text).

Dispatch (mmvq.cu:1218-1286 mul_mat_vec_q_switch_fusion):
- s2r_on = s2r && env, gated to c_ncols_dst 2..4 (1230-1232).
- share = (share_env || s2r_on) && ncols 2..4 (1233-1234) - s2r implies
  the share body; composition share+S2R is the normal s2r arm, not a new
  state.
- aln_on suppressed whenever s2r_on (1237) and dead anyway here
  (LLAMA_MMVQ_ALN unset; IQ4_XS not in mmvq_type_has_aln 1124-1136).
- launch_variant (1256-1263) instantiates mul_mat_vec_q<type, ncols,
  false, small_k, kAln, kS2r, 1> with the runtime share flag; lb arms
  (1273-1281) need GGML_CUDA_MMVQ_LB=2/4, unset in this window.

Shape/routing gate at the entry, mmvq.cu:1713-1716 (ggml_cuda_mul_mat_vec_q):
mmvq_s2r = !mmvq_aln && ids == nullptr && ne11 in [2,4]. The served
verify band is T=4 (FATTN log: "T 4"), dense projections are plain
MUL_MAT (ids == nullptr) - in band.

Serving topology check (the would-be second routing gap): the TP4 config
is -sm tensor, which builds a meta device (src/llama.cpp:131-176) that
shards the graph per die and computes each shard on the device backend
with per-device simple buffers (ggml-backend-meta.cpp:1875+,
2250/2298/2322). On each die src0 is therefore NOT in a cuda split
buffer: split=false at ggml-cuda.cu:2675 and MUL_MAT reaches
ggml_cuda_mul_mat_vec_q at ggml-cuda.cu:2748 (or the GLU fusion
handshake at 4240/4277/4342, same entry) - the s2r-capable path. The one
path that hardcodes s2r=false is the legacy split-buffer op
ggml_cuda_op_mul_mat_vec_q (mmvq.cu:1767-1769, reached only via
ggml_cuda_op_mul_mat when split=true, i.e. -sm row): not this serving
config. No second E-117-class gap found.

Ledger era, for the exactness trail:
- E-104 (plan 2742-2750): the promoted 6-type share set validated served
  "determinism byte-identical (sha 793bf51b - the share kernels change
  nothing numerically)" - with IQ4_XS deliberately OFF on replica bench
  numbers. So IQ4XS_SHARE had never been served before w23a; its
  exactness evidence is host-side only.
- E-117 fix (commit 303c92546): switch_type dropped the s2r flag at all
  21 type cases; wired through, verified by W7 A4.
- W7 A4 (results/W7_ldsy_a4_checks_2026-09-23.txt): on the REAL kernel,
  real GGUF bytes, served geometry, T=1..4, the exact composition
  superset - ALL share gates including IQ4XS_SHARE plus ALL s2r gates
  including IQ3XXS_S2R and IQ3S_S2R - dumps byte-compared:
  "DEFAULT-PATH byte-identity: PASS", "SHARE-ARMS value-identity vs base
  tree: PASS", "s2r(engaged) bit-exact vs default: PASS", and every
  bench arm "ORACLE BITEXACT". The w23a arm stack is a SUBSET of the
  validated dump, and the three gates are type-disjoint (iq4_xs share
  vs iq3_xxs/iq3_s2r s2r) with a shared producer only in the q8_1
  quantization, which s2r does not touch (legacy 36 B layout, no
  producer change).
- W30 arms (36ac2f47c: GLU fusion T4 + lb) are env-gated OFF in this
  window; the armed kernel templates equal the oracle-validated ones.

Numerics class: the gates are integer-order-identical by construction
(decode-once share mmvq.cu:1058-1061; s2r preload-and-replay
mmvq.cu:1138-1141) - the BIT-EXACT class, not the fp32 sum-order dust
class of E-090/E-094 prefill. There is no mechanism here that can change
a served float.

## 4. A4: classification and disposition

- The anomaly as framed ("text_sha256 mismatch") did not occur. At least
  one of the mission's four hypotheses is CONFIRMED but it is the first
  one in its harness form: a host-test gap - not in the kernels, in the
  window: --decode-only + exactness=True is self-contradictory, and the
  adjudicator cannot distinguish "guard absent" from "guard failed".
- d1 (fp-order dust, re-propose as numerics-class): NO. The gates are
  bit-exact class; there is no dust to sign off. And independently of
  exactness, the measured +0.32% mean sits inside the window's own
  0.28 t/s within-boot repeat spread - the perf case does not merit
  promotion or further GPU spend. Recommendation: leave
  IQ4XS_SHARE / IQ3XXS_S2R / IQ3S_S2R unset in BASEENV (E-104 status
  quo for iq4_xs; s2r stays unserved). Re-open only if a future schedule
  change (C3/C4 class) re-prices the pool.
- d2 (routing/composition bug): none in ggml. The fixable bug is in the
  owner-owned harness /home/chris/run_post_u1_battery.sh, spec'd below,
  NOT touched per desk rules.
- d3 (unprovable host-side -> isolation cells): the exactness question is
  closable cheaply if the owner ever wants a served receipt for these
  gates (it would be their first served determinism evidence); spec'd
  below. It is not needed to clear the EXACT-VOID - that verdict is
  already explained in full by the harness defect.

## 5. Specs (for the owner; no /home/chris file was modified)

S5.1 Harness fix (run_post_u1_battery.sh):
  a) adjudicate(): distinguish missing from failed -
     det_missing = (c["det"] is None); det_fail = (c["det"] == "FAIL").
     EXACT-VOID only on det_fail; det_missing on an arm -> verdict
     "EXACT-UNMEASURED" with ratchet "determinism guard absent
     (--decode-only); rerun with the guard before adjudicating
     exactness".
  b) window_w23env(): either drop exactness=True from the w23env config
     or launch cells with battery args that include the determinism
     guard (e.g. --decode-only --determinism-only if the battery grows
     that combination, or the full battery). Config and cells must not
     contradict.
  c) optional, law-consistent: if cross-arm exactness is the intent,
     have the adjudicator compare the guard shas across the four cells
     (the W7 A4 law) instead of only arm-internal PASS.

S5.2 Three single-env isolation cells (only if a served exactness
receipt is wanted; ~10 min each, run_combined_window.sh arm pattern,
owner executes):
  run_combined_window.sh w23iso4xs "GGML_CUDA_MMVQ_IQ4XS_SHARE=1"
  run_combined_window.sh w23isoxxs "GGML_CUDA_MMVQ_IQ3XXS_S2R=1"
  run_combined_window.sh w23isos  "GGML_CUDA_MMVQ_IQ3S_S2R=1"
  against a regress control arm. The 200k cell of each run carries the
  full battery incl. determinism_guard. PASS bar: every arm's
  determinism sha equals the control sha (byte-identical text class,
  W7 A4 law). Any single-env sha mismatch then isolates the diverging
  gate directly. Cheaper variant if the 200k full battery is too heavy:
  boot per run_combined_window.sh's pattern and call guard_battery.py
  with --determinism-only (~2 min of requests per cell).

## 6. Desk record

- CHECKIN.log: A1 17:03:32, A4 17:09:56 (root-caused line).
- Cells mined read-only: /home/chris/postu1_w23{r1,a1,a2,r2}{,.jsonl,.server}.
- Verdict repro (read-only, exact adjudicator regex + logs): all four
  det=None -> EXACT-VOID; paired +0.11/+0.05 t/s.
- Zero-GPU: no lock touch, no card work, guard_battery.py untouched.
