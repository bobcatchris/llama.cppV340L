(c) CHECK-(i) SCOPE FINDING - one paragraph, for routing to gemini (their gate file; my measurement).

PG-1 check (i) certifies D3 compliance by compiling exactly ONE file (src/ops/launcher/l2norm.cu) and
asserting zero `__ocml_*` PLUS presence of `v_fma_f32`/`v_add_f32`. Measured here: it covers 1 of the
27 whitelisted `.cu` device TUs (3.7%), and 23 of those 27 reference bf16 types (counting rule: one
regex, `__nv_bfloat16|__hip_bfloat16`, occurrences per file; most affected rmsnorm.cu 25,
silu_and_mul.cu 20, gdn_projected_conv.cu 20, embed_gather.cu 11). That alone would argue "widen the
list" - but the assertion as written CANNOT be generalised, which is the stronger finding. I ran
check (i)'s exact two clauses over a sample of six unchecked TUs: five PASS, and `embed_gather.cu`
FAILS the fp32-presence clause, because it is a pure-integer gather (add_co_u32 97, mul_lo_u32 71,
zero float ALU by design - correct code, wrong expectation). So "fp32 ALU instructions present"
encodes *this file does float math*, a property of the file, not a D3 criterion: applying check (i)
to embed_gather yields a FALSE RED, and not applying it leaves 26 of 27 un-covered while the gate's
own PASS line reads "D3 ISA ... Certified". Combined with my S1 §2 reproduction - a genuine D3
violation (`__hadd2`/`__hmul2` on gfx900) emits `__ocml_*`=0 AND fp32 ALU >0, i.e. it SATISFIES
check (i) exactly as written on the one file it does look at - check (i) is simultaneously too narrow
to cover the build and too weak to catch the violation. Suggested shape (design is theirs): (a)
replace the presence-of-fp32 clause with a per-TU EXPECTATION table naming which whitelisted TUs
legitimately do float math, so an integer-only TU is neither a false red nor a silent skip; (b) gate
on the ABSENCE of an emulation fingerprint - bf16-typed values feeding fp32 ops feeding a repack
(`v_bfi`/`v_cndmask` RNE) - since the emulation path never calls libm, `__ocml_*` cannot see it.
Zero-GPU, ISA-only. A working negative control already exists and FIRES:
tools/v340l/wo07_pg1_check_census_receipts.sh cell C3.
