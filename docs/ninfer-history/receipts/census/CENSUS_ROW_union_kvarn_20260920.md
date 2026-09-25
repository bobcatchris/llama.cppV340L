# POST-PROMOTION CENSUS — union bin 058a7b85, kvarn_k4v4@65536/ws512, plen-2075 mt64 x3 + fast probe (NINFER_PREFILL_OPTRACE=3), 2026-09-20 ~10:5x
# SEQUENCING ERROR (stamped): the raw OPTRACE lines were lost — serve_fast truncates its log
# on every boot and the canonical restore ran BEFORE extraction. Only the per-request steady
# table below survives (computed live from the trace before restore). Repro = one instrumented
# boot (~2 min); EXTRACT BEFORE RESTORE from now on (runbook candidate rule).
## The table (rank=0, steady = chunks 2+, ms/chunk):
req1  chunks=1   (fast probe; per-chunk trace boundaries noisy — excluded)
req2  chunks=1   (excluded, same)
req3  chunks=17  steady: wall=661.1 gemm=179.0(27%) ar=143.1(22%) body=41.9(6%) gap=41.9(6%)  <- warming
req4  chunks=17  steady: wall=667.7 gemm=114.5(17%) ar=142.0(21%) body=24.1(4%) gap=24.1(4%)
req5  chunks=17  steady: wall=674.4 gemm=115.8(17%) ar=143.1(21%) body=24.1(4%) gap=24.1(4%)
req6  chunks=17  steady: wall=684.2 gemm=114.3(17%) ar=144.8(21%) body=23.9(3%) gap=23.9(3%)
req7  chunks=17  steady: wall=699.2 gemm=112.7(16%) ar=145.8(21%) body=28.2(4%) gap=28.2(4%)
## FINDINGS (client walls corroborate: 18.9/19.1/19.7 s request walls, finish=stop x3):
1. **Steady chunk wall ~661-699 ms = ~183-194 tok/s prefill class at plen-2075** — ~1.5x the
   REV2-era ~128 tok/s class (and request walls 18.7-19.7 s vs the 41-46 s pairs era). The
   promoted line (mad-mix + kvarn) is the fastest measured serving state on this box.
2. **The OPTRACE column partition is STALE on the kvarn route**: columns sum to ~305 ms of a
   ~670 ms wall — ~356 ms (~54%) is UNTRACED device work (the kvarn attention/dequant kernels
   sit outside the tracer's gemm/ar/body sections; "gemm" now reads only the traced nvfp4
   launches). NO column-share claim is valid on this route until the tracer partition is
   extended to the kvarn kernels. The REV2 books ("gemm 76% of wall") do NOT describe this route.
3. AR (~143 ms, 21% of wall) is the largest TRACED term — but with >50% untraced, the real
   critical-path ranking is UNKNOWN pending the tracer fix (PLOG-085's rule: no window spends
   until the map is honest).
## NEXT (the only executable path this opens): extend NINFER_PREFILL_OPTRACE sections to the
kvarn route's kernels (tracer desk, zero-GPU design + one instrumented boot), THEN re-census.
