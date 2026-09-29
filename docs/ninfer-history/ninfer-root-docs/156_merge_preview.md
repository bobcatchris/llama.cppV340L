# §18.16 MERGE CONFLICT MAP — host-kv → wo/integration-ci (refreshed 2026-09-07 ~20:00Z)

Re-run per coordinator (the 6-file preview @ 78fde32f predated kvarn + radiance + nvfp4
landing). Live dry merge: `git worktree add /tmp/merge_prev cbad568d --detach; git merge
--no-commit --no-ff 441e484c` (my HEAD at the time).

## RESULT: 1 textual conflict, doc-only. ALL 6 CODE FILES AUTO-MERGE CLEAN.

Auto-merged, no markers (semantic build check still owed at integration):
  include/ninfer/types.h            (nvfp4 dtype enum + my RuntimeStats host_kv_* — disjoint hunks)
  src/runtime/tp2/tp2_backend.cpp   (nvfp4 surfaces + my net/trigger/fence/probes — disjoint hunks)
  src/runtime/tp2/tp_engine.cpp
  src/serve/request_log.cpp
  src/serve/serve_options.cpp
  tests/CMakeLists.txt              (nvfp4 test target + my host-mirror test target)

## THE one conflict:
  docs/156_host_kv_safety_net_agent_work_order.md — ADD/ADD (the doc was created after the
  stack point; both lineages appended sections). Resolution: keep-both (their sections,
  then mine) — validated: 911-line resolved doc, no orphan markers.

## Residual risk:
  textual auto-merge ≠ semantic merge. The integration lane should build + run
  ninfer_host_kv_arena_test + ninfer_paged_kv_host_mirror_test + the unit suite on the
  merged tree before my lane is declared merged. nvfp4's dtype.h surface and my
  RuntimeStats share types.h — disjoint hunks, low risk.
