# A/B Receipt: GGML_CUDA_DISABLE_FUSION probe - 2026-09-21 14:0x-14:1x CDT

Question: is fused-GLU MMVQ a net win on gfx900 despite running 2.4x off the
byte floor per call (census W0_census_decode_receipt.md)?
Method: ab_arm.sh, one boot per arm, 2 battery reps each (rep1 fresh-boot,
rep2 warm no-boot). Baseline gates file of record.
Battery version caveat: guard_battery.py gained determinism+needle guards
MID-SESSION (14:05-14:07); control rep1 ran the old battery, all later reps
the new one. Content differs across reps; arm-vs-arm comparison below uses
same-position reps only.

## Results

| cell          | control rep1 | control rep2 | nofusion rep1 | nofusion rep2 |
|---------------|-------------|--------------|---------------|----------------|
| prefill_2k    | 49.77       | 52.07        | 54.46         | 55.89          |
| decode_8k_10k | 15.81       | 16.60        | 14.56         | 14.76          |
| accept        | 0.6795      | 0.6944       | 0.6825        | 0.7333         |
| determinism   | -           | PASS         | PASS          | PASS           |
| needle 3/8k   | -           | 3/3          | 3/3           | 3/3            |

## Verdicts

1. DECODE - fusion STAYS. Removing fusion costs 14.66 vs 16.21 t/s mean
   (-8.0%, consistent sign across both reps = unambiguous class). The census
   per-call reading was incomplete: fused MMVQ also eliminates one
   quantize_q8_1 pass, an intermediate f32 write+read, and the separate
   silu-mul kernel. Acceptance moved +0.01..+0.04 in the nofusion arm
   (numerics changed -> expected; within canary gate).
2. PREFILL - VOID for arm attribution. Values climb monotonically
   49.77 -> 55.89 across all four reps regardless of arm = the thermal
   integrator dominates (box was hot from the preceding validation battery;
   docs: prefill is a heat state function, 1.65-2.8x cold/hot spread).
   Also, GLU fusion should not even engage at prefill (ncols_dst>1 returns
   false in ggml_cuda_should_fuse_mul_mat_vec_q), so no arm effect is
   expected there - consistent with what we see.
3. The 49.77 vs 93.77 prefill gap twenty minutes apart re-demonstrates the
   THARM1 law on the new cooling: prefill cell REQUIRES cold/warm stamping
   and cell order (prefill first after >=3 min idle). Decode is mildly
   thermal (16.6 warm observed, above its own baseline - decode gate center
   stays 15.57).

## Follow-ups

- F1: battery hardening (assigned Gemini): thermal sideband sampler,
  cell order prefill-first, idle-wait knob, cache_prompt=false everywhere,
  battery VERSION row in every receipt, no mid-session edits.
- F2: cold-stamp baseline re-run with the hardened battery to restore a
  clean prefill reference (93.77 -> expected ~96+ cold).
- F3: scoped fusion ablation per iq type stays CLOSED unless a future cell
  shows a specific iq type losing to fusion (env kill switch is too blunt;
  a positive probe would be needed first).
