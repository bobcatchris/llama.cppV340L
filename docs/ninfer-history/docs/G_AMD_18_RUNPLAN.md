# G-AMD-18 written run plan (agent3, 2026-09-12 ~21:30Z) — one grant, two cells

Per coordinator seq-19/25/27: plan before grant; bars named before running;
drift-characterizing vs pass/fail chosen up front; release rows; budget up front.

**Device: dev2 (or dev3) — NOT 0/1, G-AMD-17 serve holds them.** Pre-launch rule:
rocm-smi KFD check; any foreign context on the chosen device → stop + report,
no silent device switch. HIP_VISIBLE_DEVICES set to the granted device.

**Freshness protocol (both cells):** binaries rebuilt from committed HEAD
(1726c918 or the exact tip cited at grant time), sha256 recorded pre-launch.
Note: the merged amd/main shim (cuda_runtime.h hostalloc rework) COULD change
device-path behavior — the rebuilt runner doubles as a post-merge re-verification
of the stride-4 anomaly's persistence. sha-delta vs tonight's 4da6e9b8 g16b =
merge inheritance, disclosed.

---

## CELL A — the stride-4 writer, caught in the act (rocgdb)

A0. Rebuild runner from HEAD (same staged recipe: runner + dec + pref .o,
    -lninfer_hip_host). Verify fatbin arch tag gfx900. New sha in report.
A1. PLAIN run first (no debugger, ≤90 s): does stride-4 persist post-merge?
    (If the anomaly VANISHES → the merge itself is the datum; report + no A2-A4,
    window returns early.)
A2. rocgdb (/opt/rocm-6.2.0/bin/rocgdb, batch -ex mode, script logged):
    load rebuilt runner, run to the decode reduce launch.
A3. BREAKPOINT at the predicted store PC, from the frozen code object
    (carved_A.co, symbol `_ZN6ninfer3ops42gqa_attention_small_t_reduce_output_kernelI…
    GqaGeometryILi12ELi2ELi2EEELi64ELb0ELb0ELb0ELb0EEE…`, .text 0xD7600, size 0x75C):
      **the ONLY store: `global_store_short v3, v0, v[0:1]` at file-vaddr 0xD7D50
      = load_base + 0xD7600 + 0x750.**
    Expected outcome per the ISA-in-file: hit, repeatedly (3072 active stores).
    At first hits dump: $pc (disassemble-at-pc to confirm executed mnemonic ==
    global_store_short), the v_data reg (expect bf16 in LOW 16 bits), the 64-bit
    vaddr (expect byte offsets 2d+512h, 2-byte aligned, contiguous).
A4. Verdict matrix, fixed BEFORE running:
    - BP hits, regs clean (short width, 2-byte addr), buffer STILL stride-4
      post-run ⇒ corruption is BELOW register semantics (memory-system/loader
      class) ⇒ hand to coordinator+gemini deep-probe lane with the register
      receipts.
    - BP hits, regs DIRTY (dword-pattern data in v_data or 4-stride vaddr) ⇒
      upstream divergence inside the kernel ⇒ single-step backward from store
      to first dirty def — compiler-level finding for the ISA-probe lane.
    - BP NEVER hits yet buffer mutates ⇒ executed code ≠ fatbin ⇒ `rocgdb
      examine` the executing code-object bytes at the actually-executing PC
      (get via watchpoint below or `info targets`/code-object list) → diff vs
      carved .co — loader-level, deepest class.
    - Device-memory watchpoint (`watch *(unsigned*)0x…` on d_out_dec[0..64))
      is the FALLBACK instrument if the bp can't be placed before module load;
      same verdict mapping (watchpoint catch gives the PC directly).
A5. Timeout 90 s per launched run; rocgdb session ≤ 5 min wall; own-pid kills.

## CELL B — GDN/gating emulation parity vs fp64 oracle

B0. New probe binary `t3_parity_g18` (source committed pre-launch): CPU fp64
    oracles + D2H comparator, C441 seq-40/31 comparator shape (readback via
    correct bf16<<16 path — the 0bbce5b8 lesson encoded).
B1. GDN recurrent T=1 (launch_recurrent; pure SIMT original — NO mma) —
    **PASS/FAIL bar: per-role cos ≥ 0.999, loud guard |d| ≤ 2 bf16-ULP.**
B2. gating gemv (T=1 LIVE arm; no mma) — **PASS/FAIL, same bars.**
B3. gating simt_c8 (SIMT route; no mma) — **PASS/FAIL, same bars.**
B4. GDN chunked (launch_chunked → output/prepare_wy_wu/state_passing — the
    mma-emulation consumers) — **DRIFT-CHARACTERIZING, decided now:**
    report per-role cos distribution, max|d|, |d| in ULP histogram; pass/fail
    vs the C441 bars is INFORMATIONAL in the report, never re-scoped after
    seeing results. Accumulation-order drift vs HW-mma is a real effect at
    f32-eps scale; bf16-rounded outputs should still sit inside 2 ULP unless
    the fragment layouts are wrong — a distribution OUTSIDE 2 ULP on many
    roles IS the emulation-bug signal either way.
B5. gating mma_split2 → assert the LOUD host refusal fires (cudaLaunchKernelEx
    shim message + error return; zero device execution). PASS = refusal observed.
B6. Per-launch timeout 90 s; shapes = q3 (Gqa27Tp-like GDN dims, H_qk/H_v from
    the target's real config); seeds fixed and printed pre-run (no seed
    shopping).

## Budget & closure

- Total device ask: **≤ 15 min** (A ≤ 6 incl. rocgdb drag; B ≤ 6; 3 min margin).
- Release rows: results/amd/p1/g18_window_{start,end}.txt + intercom row line.
- Evidence committed before report: logs, raw dumps, gdb transcript, probe source.
- If G-AMD-17's serve throws a stride-4-lookalike in production decode (agent4
  stop-kill-report per coord), that witness attaches to CELL A's framing —
  A's verdict matrix is written to consume either outcome without re-planning.
- Queue after G-AMD-18: gemini parity-spec reconciliation of the runner
  comparator (their file, my call — coordinator standing note).
