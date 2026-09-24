# W7 FINALIZE DESK RESUME — post-reboot pickup point (written by owner from the desk's banked evidence after the desk session died on a model-request failure 2026-09-18 ~13:5x; nothing was lost — all evidence was committed/banked before death)

## Desk mission (unchanged)
Split and cure the finalize-chunk anomaly: chunk 17 (tok=27) carries ~1394 ms unbracketed
gap per request, rank staircase 1362-2053 ms (rotating straggler: rank0 was the 2051 in the
12:08 leg, rank1 the 2052 in the 13:41 leg). REV2 §3.

## Measured so far (banked)
- 12:08 OPTRACE=3 leg (pre-pin): chunk17 wall 2051, gemm 446 / ar 44 / body 167 / gap 1394.
- 13:41 bracket leg (v2 bin 8dce792f, post-pin + post-gpureset): GREEN probe BLUE/stop/52,
  **prefill 110.0 tok/s, mtp 2.65 tok/round** (single instrumented leg, post-recovery box —
  NOT a pin verdict; the pinned-vs-auto A/B on the SAME bin is still owed).
- Bracket readout (v2): chunk-17 prep/final_fwd/arsteps_tail all 0.000 AND extent==0 with
  **no LMHEAD print inside the finalize window, no ar-step loop executed**.

## THE TWIST (changes the diagnosis)
The is_last MTP finalize path (final mtp_prefill_chunk + ar-step loop) **never executes in
the serve path** — mtp_proposal_extent_==0 at the finalize chunk. The 1394 ms gap therefore
lives OUTSIDE the MTP finalize region: in the unbracketed span between the layer-loop end
and wall_end — candidates now ranked: (1) the T=27 tail's recurrent-dn path (T<64 => the
recurrent GDN, 27 serial steps x 64 layers — CHECK ITS COST FIRST), (2) request-end work
(checkpoint rewrite, KV finalize, response assembly), (3) a rotating-straggler collective
drain at request end (the staircase rotates across ranks between legs).
v3 instrumentation (commit f3bdae593) records marks in the ELSE branch (the seeding
mtp_prefill_chunk + sync + D2H + H2D) so the next leg's prep column carries exactly that
span — plus the existing [PREFILL-OP]/[PREFILL-BODY] columns for the recurrent-dn check.

## Wedge saga (box-state; playbook row material)
Onset 12:47 after the pk16 bench GPU session; every boot 12:47-13:33 wedged at the FIRST
worker-MTP-prefill-redo with no-retry TCP-read VM faults (status 0x00801031, ring:24, RW=0;
host-VA-class fault addresses — consistent with IOMMU DMA-FQ translation state). Exonerated:
binary (certified 1e03e5af wedged 13:32 after serving GREEN 12:03), commit (c1c6b4cb and
8dce792f both wedged pre-fix-point), env, clock pin (12:48/12:53 pre-pin). CLEARED by root
gpureset of all 4 dies at ~13:38 (dmesg "reset succeeded"); 13:41 leg GREEN. The operator's
iommu=pt reboot (runbook 408d04a4d) may harden against the class — note the correlation in
the row. New unit of the playbook: a bench session with copy-hammer loops followed by serve
boots wedging at first collective+device-buffer read => gpureset BEFORE blaming code.

## NEXT STEPS (in order, post-reboot)
1. Operator reboots with iommu=pt (GRUB line in docs/amd/W7_MAINT_WINDOW_RUNBOOK.md);
   verify: IOMMU groups read identity, rocminfo 4x gfx900, then bash /home/chris/serve_10k.sh.
2. Boot v3 bin 6a684e7f2988adf7 (banked, UNBOOTED): bash
   results/amd/coherence/W7_finbracket_serve.sh /home/chris/artifacts_bin/ninfer-serve_6a684e7f2988adf7.bin
   (script retires anchored ^/home/chris/artifacts_bin/ninfer-serve, boots, probes plen-2075
   mt64 temp0, OPTRACE=2 env included; do NOT pipe through head — SIGPIPE kills it).
3. Read [PREFILL-FIN] chunk-17 (now with else-branch prep) + [PREFILL-BODY] chunk-17 dn
   column (recurrent tail cost) => name the 1394 ms owner.
4. If recurrent-dn tail: the cure candidates are a chunk-boundary flush (pad tail into the
   chunked path at T<64 via NINFER_GDN_FORCE_RECURRENT-style gate inversion) or a
   T=27-specific recurrent tune; if request-end work: bracket it (same fin-mark pattern).
5. Fix + A/B on the same bin, same band (pinned clocks now standing); parity gate content
   'BLUE' finish=stop; bank row + PLOG + merge.
6. ALSO OWED: the pinned-vs-auto clock A/B (same bin both legs; the 110 single leg is NOT
   evidence either way) and the AR floor bench (docs/amd/W2_AR_FLOOR_desk.md).

## Artifacts
Row: results/amd/coherence/W7_finbracket_row.txt (full leg table + wedge receipts).
Logs: W7_finbracket_serve.log (13:41 GREEN leg), W7_optrace3_serve.log (12:08 staircase
source), W7_dn_default_serve.log (12:03). Bins: 1e03e5af (serving-era), c1c6b4cb (v1),
8dce792f (v2), 6a684e7f (v3, next). Commits: ac3dfc1b8 (v1) -> 8dce792f (v2) -> f3bdae593 (v3).
