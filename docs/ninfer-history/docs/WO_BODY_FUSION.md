# WO_BODY_FUSION — body-op fusion/harvest (queue #4, priced ~21 ms/chunk flat)

- **Target mass (PLOG-078-era pricing, arfix BODYSUM):** the flat fusible body column ≈
  gate 11.6 + conv 4.6 + norms ~3.6 + upk 1.6 ≈ ~21 ms/chunk (~2% of the 990 ms chunk).
  `gqa` (+3.2 ms/chunk, attention-vs-depth) and `dn` (~30 ms, delta-net trio) are
  algorithmic — NOT fusion targets.
- **Desk:** agent #2. Worktree: `git worktree add /home/chris/worktrees/amd-wo-fuse amd/main
  -b amd/wo-fuse`. cmake at /home/chris/opt/cmake/bin/cmake (house recipe: NINFER_BACKEND=hip,
  NINFER_BUILD_APPS=OFF, NINFER_HIP_BUILD_SERVE=ON, gfx900; -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0
  REQUIRED). df -h / first (~5 GB free — keep the build lean; serve build only if you reach
  the serving leg).
- **PHASE 1 — decisive measurement (2 s of GPU when the window frees; GPU-free prep first):**
  a standalone bench binary is ALREADY LINKED at /tmp/gdn_gating_proj_bench (built from
  bench/ops/gdn_gating_proj_bench.cu against libninfer_hip_host — rebuild it in YOUR tree if
  /tmp was cleaned). Run: `--norm-control --tokens 128 --warmup 5 --repeat 50` (and tokens 1
  for the decode-route sanity). This names the kernel-level split of the 11.6 ms `gate`
  column (rmsnorm vs the 96x5120x128 MmaUnsplit control projection) at serving geometry.
  Expected from first principles: ~0.5 ms class — if it measures ~11 ms the gating MMA is
  ~20x off its floor and the fix is LAUNCH-SHAPE, not fusion. Either result decides the
  implementation. A serve up = dies are a packed TP4: do NOT run the bench while a serve is
  up (check `pgrep -f '^/home/chris/artifacts_bin/ninfer-serve'` first; the coordinator's
  G-MM-2 pairs own the window until its row lands — poll, don't interleave).
- **PHASE 2 — implement the top item the measurement names**, in YOUR worktree, with an env
  kill-switch (default ON acceptable only with a runtime rollback path):
  - if bench ~11 ms: fix the gating MMA launch shape (grid/block for 48-row x 128-token
    problem) — likely the whole prize;
  - if bench ~0.5 ms: fuse the flat chain instead — candidates in body order:
    gdn_norm_control_projection's norm+projection seam (text_context_impl.h:2807-2811),
    rmsnorm folds around GDN (gnorm/mnorm), unpack-in-GEMM absorption (upk).
- **PRE-REGISTERED GATES:** G-BF-0 (cell): the gated op's kernel time improves >=10% at
  T=128, relL2 vs fp64 golden <= 1e-2 with a both-directions falsifier, no VGPR/scratch
  regression (census before/after). G-BF-1 (serving): ordinal-paired fresh boots
  (PLOG-060), 3x plen-2075 mt64 per arm, +-2% within-pair, wall -0.5% minimum => promote to
  BOOT_BATTERY + PLOG row; miss => bank, close, receipt.
- **Laws:** retire = EXACTLY `pkill -9 -f "^/home/chris/artifacts_bin/ninfer-serve"` (only
  boots you started); restore canonical serve_fast + health at every close; window claims
  appended to THIS file on YOUR branch ("WINDOW CLAIM: <leg>"); no builds in
  /home/chris/dual_5060_ti_ninfer EVER; PROGRESS LOG newest-first after every step; clocks
  sidebands on every number; infra = tag "BLOCKER:" and continue.

### 2026-09-20 09:2x local — WINDOW CLAIM: G-BF-1 (coordinator executing personally) — canonical now = union bin 058a7b85859c5cd0 (kvarn promoted); arms unchanged per pre-registration: BASE cc122cc0 vs TIP 37bc9b250bfada70 (both bf16 legs — the pair isolates ksplit)
- The battery closed 08:58 (4/4 + 4/4 exact); the KV promotion flipped the default to the union bin (see WO_Q4KV_K4V4.md close entry). G-BF-1 runs NOW per this claim; the chunk256 watcher is blocked by this line (by design) and I launch its runner myself after this window.
- Restore after this window = the NEW canonical (union kvarn line via serve_fast), not cc122cc0.
