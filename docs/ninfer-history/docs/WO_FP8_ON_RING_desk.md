# DESK FILE: fp8-on-ring AR (WO_FP8_ON_RING.md) — desk amd-wo-fp8ar

Owner: fp8ar desk agent. Branch: `amd/wo-fp8ar` (worktree `/home/chris/worktrees/amd-wo-fp8ar`,
branched from `amd/main` @ 3780414e0). Build: `build-hip-amd/` in that worktree only
(HIP lane, serve-only, mirrors the w7-body config: NINFER_BACKEND=hip,
NINFER_HIP_BUILD_SERVE=ON, APPS=OFF, gfx900).

## WINDOW CLAIMS (coordinate here; k4v4 desk has FIRST SERVE boots per the WO)

- **Claim 2026-09-19: standalone transport cell.** Serving window verified DOWN
  (no `ninfer-serve` process; rocm-smi 0% GPU + 0% VRAM on all 4 dies immediately before).
  I run `tools/v340l/w7_fp8_ring_cell.cu`:
  - RCCL arms (bf16 anchor, fp8, int8) need **4 physical dies** — RCCL refuses
    duplicate-GPU ranks at 2 visible devs (banked evidence: WO_COPYADD_row.txt, rc=5,
    2026-09-19). Die-law deviation from the WO's "(standalone, dies 2/3)" is REQUIRED for
    the decisive RCCL arms; run is guarded (`alarm`), seconds long, window re-checked
    immediately before launch. If the window comes UP mid-run: the cell's RCCL arms abort,
    host-staged arms on dies 2,3 only, and the RCCL legs re-run in the next down window.
  - Host-staged arms (shapes a/b) run pinned to dies 2,3 (`HIP_VISIBLE_DEVICES=2,3`),
    the pre-registered geometry.
- **SERVE boots (G-FP8-2/3) are NOT claimed yet.** k4v4 takes first boots; I claim a boot
  slot by appending a claim here when my build + cell gates are done.

## STATUS

- [x] Worktree + branch + build config (this worktree, no shared-checkout builds)
- [x] Survey: tp_group.cpp AR entry, copyadd codec + row, RCCL fp8 types declared
- [x] Cell built + run (staged rerun VALID after g_n fix; rccl VALID on 4 dies)
- [x] **G-FP8-0: FAIL (all arms) — G-FP8-1: FAIL (all quant arms) => DESK CLOSED per the
      pre-registered law. Verdict tables: WO_FP8_ON_RING.md PROGRESS LOG step 5.**
- [x] Serve arm 4a7533a7b5f4e16c: NEVER BOOTED (boot-gated on the cell verdict; the
      ncclFp8E4M3 route it implements is slower than bf16 — do not ship)
- [x] Close: rows banked (results/amd/coherence/WO_FP8_ring_row.txt), window found DOWN
      and left DOWN, no foreign process touched, watcher exited, health clean

## RE-EVALUATION MANDATE (what would reopen this item — coordinator/principal call)

1. A numerics-budget call: rig-I8P passes kappa=1 at ~2e-2 (measured 0.0160) and delivers
   -39% AR in-cell — but the current 1e-2 budget kills it, and rightly: 1.6% element-level
   error feeds every attn/mlp input. Requires behavioral evidence of the G-FP8-2 class
   before any principal sign-off.
2. RCCL fixed upstream: ncclFp8E4M3 allreduce exists but runs software-slow (slower than
   bf16) on RCCL 2.20.5 — reprice if RCCL ever ships a real fp8 reduction path.
3. Wider-code schemes (>=8-bit codes) cannot halve bytes — no byte win, no desk.

## NOTES FOR THE COORDINATOR

- 15:47 disk: `df -h /` = 5.9G free (95%), down from 11G at ~15:05. My tree accounts
  2.1G (build 446M + inherited sources); /tmp traces (tg_trace* ~0.6G) and the kvarnport
  desk tree (2.2G) are the visible movers. No action from this desk; flagged per the
  disk-is-binding law.
- `git push origin amd/wo-fp8ar` is REJECTED by the pre-receive hook: the branch's
  history (inherited from amd/main's merge of w7-body) carries >100MB blobs
  (results/amd/coherence/w7_cure_tl/*.csv, w7_draft_arm_bench). My commits are local;
  coordinator may want the hook's large-file list resolved before chair merge.
