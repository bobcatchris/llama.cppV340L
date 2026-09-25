# A2 lane — GHOST hunt state + discriminants (external-review packet sections)

Timeboxed handout for the 00:00Z pivot (coordinator seq-29). Written by agent2
(01a08823), 2026-09-09 ~23:5xZ. Everything citable to a committed artifact in
`results/` on `wo/dflash2-w5-habitat` (tree incl. 23ac8b3a tap-fix + 291c28a1
LOGDUMP hook). All numbers below are RE-DERIVABLE from the committed dumps +
`tools/diag/block0_replay.py` / `stage0_fuse_reference.py`.

## THE GHOST, defined
DFlash2 draft-speculation acceptance ~0 (tok/round 1.02-1.07, accept 0.3-0.9%)
on a healthy boot: reproducible across TWO artifact provenances (09-06 Q4_K_M
route, z-lab-safetensors v2 route), BOTH tap-site spellings ({6,20,34,48,62} and
trunk-parity {5,19,33,47,61}), and K=5 AND K=7. W5-for-DFlash2 is blocked behind it.

## (1) K=7 cell verdict (suspect #2 CLOSED)
`results/k7_cell_20260909_191744/` — tree 754abab1, binary 498419b2ecf1,
v3_trunkparity sidecar (payload sha 58e056847bcf…, VERIFIED byte-identical to
v2_correct), NINFER_DFLASH2_ALLOW_K7=1 + --draft-tokens 7, same two bodies.
acceptance=0.01/0.00, tok/round=1.07/1.04 per lane — statistically identical
to K=5 on every prior capture. decode/round also moved 6.4→8.2s (T=8 TokenTile
first-contact cost, recorded, not analyzed — regime is off the table as THE fix).

## (2) Full evidence chain, first-divergent-stage method (all BITWISE unless noted)
Stage-local references computed from the ENGINE'S OWN dumped bytes at each seam
(so one stage's noise cannot cascade), bf16-bitwise with ULP accounting; every
named read-layout per the standing rule:

| stage | result | layout note |
|---|---|---|
| loader (arena fc vs sidecar bytes) | BITWISE-EQUAL, both ranks, both artifacts | raw w8g32 planes: codes int8 [5120,25600]@0, f16 scales @align256 |
| fuse `enc_norm(fc@taps)` | ≤3.4 bf16-ULP on 100% of 61440 elems (f32-accum noise class) | **fused/taps: row = lane*6+t, k = layer*5120+hidden; the [hidden,cols] "interpB" read MISRESOLVES everything — §1.4's lane-distinctness verdicts must be re-read under this layout (T6)** |
| x_in assembly | BITWISE (81920) | block cols [16,5120], col=lane*8+j, col0←fused row lane*6 |
| h, c_in (norms+convs) | BITWISE (each 81920) | conv base needs the BINDER PERMUTATION: kernel slot (c,k,s) = container flat c+5120*(k+2s); a naive container read "fails" 68-93% and FAKES a divergence |
| proj | 90.5% ≤4ULP, stragglers p99 27ULP | f64-vs-MMA accumulation class |
| qn/kn | position-fit scan: engine rotated cols 0..7 at pos **70+j**, cols 8..15 at **62+j** — pairing of rope positions to fused lanes is lane-order-AMBIGUOUS from logs; internally consistent either way (both built from same lanes[] vector) | DflashText1D rope = kDflashRopeInvFrequency table 10^(-i/10)-geometry, pairs (j, j+64), f64-angle→f32-reduced→sincosf — NOT base^(-2i/128) |
| attention `a` (from dumped qn/kn/v) | ≤1.8 ULP | in-block-only at first chain (c_lo=c_hi=0), per-lane non-causal softmax, GQA 4:1 |
| 5-block stack → final | cos 0.9997, per-col cos 1.000 | f64 stack replay |
| **head logits (d2_vlog)** | **maxdiff 0.061, corr 0.999997, top-16 SETS+ORDER equal (16/16, 15/16, 16/16 on cols 1/2/9)** | `NINFER_DFLASH2_LOGDUMP` dump: bf16 {2*124160, 16} feature-INNER, element (v,c) at byte 2*(c*248320+v); v = GLOBAL token id |
| candidate-domain (A1 suspect #4) | 0 intruders ≥248077 in 256 slots — faithfulness nit, not ghost | |
| **selector lattice + walk** | **DIVERGENT — see below** | |

### The unreachable-chain finding (THE datum)
Engine's own first-chain drafts (killcell + vlog captures, IDENTICAL chains —
deterministic): lane0 = [264,369,369,60,264], lane1 = [351,15,220,220,220].
My rebuild of the EXACT `dflash2_select_candidates` pipeline from verified inputs:
- `gate = selector_hidden(w8-dequant) @ final` — equation from block_ref.py
  (llama.cpp-pinned), inputs both bitwise-matched.
- candidates+unary = top-16 of the ENGINE'S OWN dumped logits (order = value desc) —
  matches kernel contract.
- `scores[p,c] = Σ_r ss[c][r]·sp[p][r]·gate[r,col] + unary[c]` (bf16 storage +
  bf16 gate variants both tried).
- walk: anchor brute-forced over ALL 248320 possible global ids — 3988 (lane0) /
  94234 (lane1) reproduce the engine's d1; EVERY ONE of them then walks to
  [874,13,271,16] / [16,15,321,351] — the engine walks [369,369,60,264] /
  [15,220,220,220]. Chain-reachability table: engine's d2=369 at col2 is the
  winner from pred-slot 0 ONLY; the engine's own d1=264 sits at slot 5 in its
  (value-descending) candidate order — **under ANY scoring consistent with my
  gate/candidates, the engine's chain is unreachable from ANY anchor.**
Surviving explanations (the dump cell separates them):
- E1: the engine's `gate` TENSOR VALUES differ from any correct-equation rebuild
  (its upstream: ops::linear on d2_final with selector_hidden w8 — the one operand
  pair never dumped bitwise; if the w8 GEMV dequant of sel_hidden or the gate's
  bf16 staging differs, every lattice edge shifts → chain unreachable).
- E3: one of MY rebuild assumptions is wrong in a way the verified stages don't
  cover (prime candidate: the prev-candidate SLOT order the engine's walk feeds at
  pos≥2 — my ordering is value-desc; if `launch_top_candidates` emits a different
  internal slot order while `cand_ids` values stay correct, sets match, transitions
  do not — this is SQUARE-C 6/16-class invisible).
Decisive next cell (instrumentation exists for logits; needs 20 lines for tensors):
`NINFER_DFLASH2_SELDUMP` = dump cand_ids/unary/gate/scores from
dflash2_round.cu select on first chain; compare each against the rebuild;
first divergent tensor names E1 vs E3 definitively.

## (1b) E1-vs-E3 call, FINALIZED with the vlog datum (coordinator seq-32 question)
The right comparison is WITHIN one run: vlog capture (K=5, engine's own logits
dumped at the selector seam) — engine drafts [264,369,369,60,264]/[351,15,220,220,220]
lie INSIDE the vlog's per-column top-16 sets (all 10 slots), and vlog == my
independent lm_head@final replay 16/16 in sets AND order. The K7-trace-vs-my-logits
6/16 overlap was an E3 artifact on MY side (I mapped fused rows with the K=5
lane*6 stride while K=7 is lane*8 — chain_front = b*T), confirmed benign: the
K5 within-run comparison above is the clean test and it passes.
=> HEAD SEAM FULLY EXONERATED end-to-end. The unreachable-chain finding therefore
localizes to the ONLY operand pair never dumped: `gate` (and the derived
`scores`). E1, named: gate VALUES as consumed by launch_edge_scores differ from
gate = selector_hidden @ final computed by any correct-equation rebuild from the
verified bytes. E3 survives only as: cand SLOT-order (sets+order of K5 verified,
so this is now the WEAKER survivor: order verified within K5 ⇒ E3 effectively
cleared for the head path). One cell left: SELDEDUMP (cand_ids/unary/gate/scores
at first chain) — first divergent tensor closes the hunt. Expected shape if
right: gate bf16 [256,16] will differ from my rebuild by a structural amount
(e.g. transposed cols / wrong final view), scores accordingly.

 (committed this file's commit; `results/`)
Per capture dir: `serve.log` (boot + per-lane acceptance lines), `body_a.json`/
`body_b.json` (the two DISTINCT prompts — EN-reasoning 62 tok, ZH-translation
70 tok; standing recipe since seq-9: identical prompts make lane-verdicts vacuous),
`respA/B.json`; `cap.r{0,1}.taps` (u16 [25600,12] row-major, row=lane*6+t after
reshape(12,5120); k=layer*5120+hidden), `.fc` OMITS — sha in FC_OMITTED_SHA.txt
(verified equal to sidecar fc.weight region; re-derivable), `.fused` (u16
[12,5120] row=lane*6+t), `.mask` (u16 [5120]), `.final` (u16 [16,5120] col=lane*8+j),
`.logits` (vlog capture only; contract above); `blk0.r{0,1}.{x_in,h,proj,c_in,qn,kn,v,a,o,x_after_attn,x_out}`
(all u16 row-major [cols, rows-dim] per engine Tensor views; shapes 16×5120 /
16×1280 / 16×4096 / 16×1024; conv internals need the binder permute — see table);
`trace.lane{0,1}` (k7 capture; engine's per-pos candidate-id lattice, ASCII).
Binary provenance per dir is in serve.log header (sha[:12] + mtime + tree).
Captures: 18:11 (v2_fixed, old tap-sites, binary 29fcb750cd65) / 18:29 killcell
(v2_fixed, tap-fix 23ac8b3a-era, 515d1e03bebc) / 19:17 k7 (v3, 754abab1,
498419b2ecf1) / 19:35 vlog (v2_fixed, 291c28a1+4600274bb070, +LOGDUMP).

## What is NOT the ghost (closed, each with receipts above)
quant convention (0.55% = ideal int8); loader/staging (bitwise); fuse compute;
tap-site spelling; K-regime; candidate vocab-domain; the W5-MTP machinery
(separate lane, closed-infeasible arithmetic); attention/conv/norm/rope math
(all bitwise or noise-floor).

## Standing lesson (this session earned it twice)
Every content assertion must name its read-layout: interpB [hidden,cols] hashing
manufactured the §22.18 lane-blind verdict AND my first "FFN divergence" (f64-ULP
metric) AND a "c_in structural failure" (naive conv-base view) — all three were
REFERENCE bugs, not kernel bugs. The same trap on the reference side is exactly
what E3 encodes; the SELDEDUMP cell removes the ambiguity by construction.
## (3) Dump inventory (per-file sha256[:16]; layouts named above; regenerates any analysis)

### results/ghost_t1_capture_20260909_181101
```
002cebd2c1548e2e          34  FC_OMITTED_SHA.txt
97ca8207d3991e62      131072  blk0.r0.a
62386c68fbc6492a      163840  blk0.r0.c_in
c3b69946d2e1731c      163840  blk0.r0.h
3da9c68f8d3db6a5       32768  blk0.r0.kn
5faeafd5af143c7c      163840  blk0.r0.o
7d28c37657eec3a0       40960  blk0.r0.proj
4c9cd97f788d3373      131072  blk0.r0.qn
e1f8ded3e79fa5c8       32768  blk0.r0.v
33976e8ebb3470f4      163840  blk0.r0.x_after_attn
6ba07c51050271e9      163840  blk0.r0.x_in
82136a86aa029140      163840  blk0.r0.x_out
97ca8207d3991e62      131072  blk0.r1.a
62386c68fbc6492a      163840  blk0.r1.c_in
c3b69946d2e1731c      163840  blk0.r1.h
3da9c68f8d3db6a5       32768  blk0.r1.kn
5faeafd5af143c7c      163840  blk0.r1.o
7d28c37657eec3a0       40960  blk0.r1.proj
4c9cd97f788d3373      131072  blk0.r1.qn
e1f8ded3e79fa5c8       32768  blk0.r1.v
33976e8ebb3470f4      163840  blk0.r1.x_after_attn
6ba07c51050271e9      163840  blk0.r1.x_in
82136a86aa029140      163840  blk0.r1.x_out
7896439a62a59478         372  body_a.json
abbe13c7a3a3e2e9         549  body_b.json
2388526e7f20f632   139264000  cap.r0.fc
fec8a53850320426      163840  cap.r0.final
5829bf5177074de9      122880  cap.r0.fused
073b279864bb884f       10240  cap.r0.mask
7e04b3843774aca3      614400  cap.r0.taps
2388526e7f20f632   139264000  cap.r1.fc
fec8a53850320426      163840  cap.r1.final
5829bf5177074de9      122880  cap.r1.fused
073b279864bb884f       10240  cap.r1.mask
7e04b3843774aca3      614400  cap.r1.taps
dd757c965f49b1aa         331  capture.out
1fde951201133867         496  respA.json
af8dac1d24c40634         534  respB.json
b7342c76f2c89ba9        6635  serve.log
```

### results/ghost_t1_killcell_20260909_182957
```
002cebd2c1548e2e          34  FC_OMITTED_SHA.txt
5b848202e380118f      131072  blk0.r0.a
e3b368f10f218510      163840  blk0.r0.c_in
6d609f45e136d95b      163840  blk0.r0.h
fe14a06e6aeff4a6       32768  blk0.r0.kn
d491268a5ae08412      163840  blk0.r0.o
4727b03d43df0919       40960  blk0.r0.proj
45d434563e41a60c      131072  blk0.r0.qn
952017dd6498e975       32768  blk0.r0.v
b6d3251e6eb870c7      163840  blk0.r0.x_after_attn
f079c6936209f8ec      163840  blk0.r0.x_in
d7fe5db34cce0cfb      163840  blk0.r0.x_out
5b848202e380118f      131072  blk0.r1.a
e3b368f10f218510      163840  blk0.r1.c_in
6d609f45e136d95b      163840  blk0.r1.h
fe14a06e6aeff4a6       32768  blk0.r1.kn
d491268a5ae08412      163840  blk0.r1.o
4727b03d43df0919       40960  blk0.r1.proj
45d434563e41a60c      131072  blk0.r1.qn
952017dd6498e975       32768  blk0.r1.v
b6d3251e6eb870c7      163840  blk0.r1.x_after_attn
f079c6936209f8ec      163840  blk0.r1.x_in
d7fe5db34cce0cfb      163840  blk0.r1.x_out
7896439a62a59478         372  body_a.json
abbe13c7a3a3e2e9         549  body_b.json
2388526e7f20f632   139264000  cap.r0.fc
a43313585c473605      163840  cap.r0.final
bbbb1c6da2793a32      122880  cap.r0.fused
073b279864bb884f       10240  cap.r0.mask
329350520d99144e      614400  cap.r0.taps
2388526e7f20f632   139264000  cap.r1.fc
a43313585c473605      163840  cap.r1.final
bbbb1c6da2793a32      122880  cap.r1.fused
073b279864bb884f       10240  cap.r1.mask
329350520d99144e      614400  cap.r1.taps
14075c417e472419         331  capture.out
c61bfc362a006161         496  respA.json
da4f0cc7b181678e         534  respB.json
f2a31b82e604c2fe        6639  serve.log
```

### results/k7_cell_20260909_191744
```
002cebd2c1548e2e          34  FC_OMITTED_SHA.txt
7896439a62a59478         372  body_a.json
abbe13c7a3a3e2e9         549  body_b.json
2388526e7f20f632   139264000  cap.r0.fc
a43313585c473605      163840  cap.r0.final
2d80ab9488911f67      163840  cap.r0.fused
073b279864bb884f       10240  cap.r0.mask
139f5173446e4a22      819200  cap.r0.taps
2388526e7f20f632   139264000  cap.r1.fc
a43313585c473605      163840  cap.r1.final
2d80ab9488911f67      163840  cap.r1.fused
073b279864bb884f       10240  cap.r1.mask
139f5173446e4a22      819200  cap.r1.taps
dec427603c89da8d          79  capture.out
f28828c91e98547f         496  respA.json
0a02909586c8d15b         534  respB.json
09ed8a6064692676       18840  serve.log
c64b4c874ce01071        2269  trace.lane0
7f124f34c0648990        2281  trace.lane1
```

### results/vlog_capture_20260909_193536
```
002cebd2c1548e2e          34  FC_OMITTED_SHA.txt
5b848202e380118f      131072  blk0.r0.a
e3b368f10f218510      163840  blk0.r0.c_in
6d609f45e136d95b      163840  blk0.r0.h
fe14a06e6aeff4a6       32768  blk0.r0.kn
d491268a5ae08412      163840  blk0.r0.o
4727b03d43df0919       40960  blk0.r0.proj
45d434563e41a60c      131072  blk0.r0.qn
952017dd6498e975       32768  blk0.r0.v
b6d3251e6eb870c7      163840  blk0.r0.x_after_attn
f079c6936209f8ec      163840  blk0.r0.x_in
d7fe5db34cce0cfb      163840  blk0.r0.x_out
5b848202e380118f      131072  blk0.r1.a
e3b368f10f218510      163840  blk0.r1.c_in
6d609f45e136d95b      163840  blk0.r1.h
fe14a06e6aeff4a6       32768  blk0.r1.kn
d491268a5ae08412      163840  blk0.r1.o
4727b03d43df0919       40960  blk0.r1.proj
45d434563e41a60c      131072  blk0.r1.qn
952017dd6498e975       32768  blk0.r1.v
b6d3251e6eb870c7      163840  blk0.r1.x_after_attn
f079c6936209f8ec      163840  blk0.r1.x_in
d7fe5db34cce0cfb      163840  blk0.r1.x_out
7896439a62a59478         372  body_a.json
abbe13c7a3a3e2e9         549  body_b.json
2388526e7f20f632   139264000  cap.r0.fc
a43313585c473605      163840  cap.r0.final
bbbb1c6da2793a32      122880  cap.r0.fused
85e8919750656b93     7946240  cap.r0.logits
073b279864bb884f       10240  cap.r0.mask
329350520d99144e      614400  cap.r0.taps
2388526e7f20f632   139264000  cap.r1.fc
a43313585c473605      163840  cap.r1.final
bbbb1c6da2793a32      122880  cap.r1.fused
85e8919750656b93     7946240  cap.r1.logits
073b279864bb884f       10240  cap.r1.mask
329350520d99144e      614400  cap.r1.taps
5542ffb714e3bf38         331  capture.out
f0e1cc483d829556         496  respA.json
bac0fabf2ca66a91         534  respB.json
1faf480cd895a562        6638  serve.log
```


## (4) POST-PACKET ADVANCE (same night, commits after 23ad7ed0) — GHOST LAYERED, LAYER-1 DEAD
1. RACE (layer 1, FIXED + PROVEN): selector-walk D2H was a bare null-stream memcpy
   against a non-blocking in.stream (§22.57 class, instance #4). c8cd0af2 fix +
   eae596e2 proof: two uninstrumented runs bitwise-identical chains, and the
   fixed-code chain equals the oracle chain the hook-sync had been forcing.
2. FIX-C (M1, FIXED): block-input slot0 = tok_embd[anchor] (a6a564f2). Verdict
   ad5e92cb: P1 slot0 BITWISE == tok_embd[1206]/[332] (the real t0s); anchor-col
   entropy 12.38/12.00 -> 11.33/7.89; accept 0.02/0.00 -> 0.03/0.01 (1.5-2.9%);
   ws-top1s 14/16 -> 3/16. Pre-reg honored: modest lift = P3 corridor, >>5% not
   crossed => M2-necessity stands.
3. THE SHARPEST DATUM (for outsiders): post-FIX-C lane0 masked-column entropy
   ROSE ~5.5 -> 12.3 nats. Pre-fix the drafter was CONFIDENTLY WRONG (low entropy,
   whitespace chains); now it is honestly uncertain — the fused-at-col0 input was
   an OOD signal the model compensated for with garbage confidence. Confidence
   moving toward uniform after removing a poisoned input is the signature of
   conditioning-starved proposals: layer 2 = M2 (no c_t->K/V context store exists;
   pools hold self-generated K/V only). FIX-D design + A2's 4 ring-math answers:
   drafts/157_fixd_design_prereg.md (§4 + §4-ANSWERS); falsification harness
   tools/diag/kctx_replay.py (golden K_ctx/V_ctx bytes; pool-compare arms when
   FIX-D's dump lands).
Entropy tables (pre, seldump/racefix captures — identical): anchor cols 12.38/12.00,
masked cols 3.4-10.3, 14/16 whitespace top-1s. Post-FIX-C: see ad5e92cb dirs
(fixc_run1/2, anchors.json + cap.r0.logits committed; fc omitted-with-sha).
