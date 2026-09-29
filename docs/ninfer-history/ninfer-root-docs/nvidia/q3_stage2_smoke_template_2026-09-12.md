# Stage-2 serve-smoke — pre-staged template (agent3 lane, 2026-09-12, coord seq-40 CPU fill 1)

**Honest gap statement:** promotion is NOT done until a post-land serve row exists on the
merged tree. Everything else (content, audit, link) is triple-GREEN at `github/main = ebf08aaa`;
this doc freezes the exact command + expected rows so the granted window executes, not decides.

## Binary under test (tip-faithful, proven)
- `/home/intel/ninfer/worktrees/wo-q3-on-main/build/apps/ninfer-serve`
- 361,756,856 B, md5 `7af40c2209674e78c3e3952cd6995707`
- built [300/300] RC=0 AT tip ebf08aaa; commits ebf08aaa..cd6ce13d = docs-only ⇒ md5 stays
  tip-faithful UNLESS main moves src before the slot — re-link + re-md5 + restate if so.

## Command (GATE RESOLVED geometry, frozen 2026-09-12 07:02)
```
cd /home/intel/ninfer/worktrees/wo-q3-on-main
setsid nohup ./build/apps/ninfer-serve artifacts/qwen3_8_27b_q3.ninfer \
  --devices <GRANTED> --port 8155 --greedy --kv-dtype bf16 --max-context 4096 \
  > results/serve_q3_main_smoke_$(date +%Y-%m-%d).log 2>&1 < /dev/null & disown
```
Artifact: local `artifacts/qwen3_8_27b_q3.ninfer` (15,446,796,288 B, sha256 `7f26a0eb…`;
drive copy `/media/intel/models/` if local pruned — adds ~1–3 min load, honest note in row).

## Expected rows (parity target = GATE RESOLVED, results/q3_gpu_window_2026-09-11.md)
| field | anchor | smoke verdict rule |
|---|---|---|
| identity | `qwen3.8-27b / groupwise-q3` | exact string match |
| loaded weights | 13.35 GiB, ~27–30 s local | +-50% wall (IO class only) |
| KV auto | 4096 tok, runtime 654.84 MiB, free-after-weights 2.02 GiB | MEASURED restated; refusal = allocator answer, per VRAM LAW record verbatim, no estimate-relaunch loops |
| `'27 times 43'` | `1161` (32-tok cap = thinking-era empty-content row, re-run 300) | content match |
| `'capital of Japan'` | `Tokyo` finish=stop | match |
| decode | 18.3 tok/s, ttft ~0.7–0.9 s | record, don't judge (perf not the smoke's question) |
| fallback-loud audit | NO 'unsupported qtype', NO silent Q4/Q5 substitution lines | A3 cold-audits log vs frozen-contract grammar |

## Protocol reminders that ride the window
written grant from coord BEFORE launch; re-guard nvidia-smi at claim (foreign CUDA contexts,
not ports); gpu_guard.sh = repo copy absolute path; `pgrep -x ninfer-serve` only; heartbeat
rows <=5 min into lane window log; kill only own PIDs; RELEASE row + lease rm at teardown;
smoke binary/worktree retire into the prune plan below immediately after.

## Prune plan (coord seq-40 fill 3 — PLAN ONLY, execute AFTER smoke row banks)
| target | size | disposition |
|---|---|---|
| `worktrees/wo-q3-on-main/` (worktree+build) | ~1.8 G | RETIRE after smoke (tree == main; md5 banked) |
| `worktrees/promote-q3-verify/` (worktree+build) | ~1.4 G | REMOVE first — tree superseded (promote ref pruned) |
| `repo/build` | 811 M | REMOVE (repo checkout is a worktree host; nobody builds there) |
| `worktrees/wo-158-host-compaction-staging/` build | (none measured) | already absent — verify, then remove stale worktree pointer if owner-idles 24h per FROZEN rule |
| `worktrees/wo-q3-gemv/build` | 2.2 G | KEEP — live lane, TP2 device rows need it |
| NOT MINE: `wo-dflash2-w5-habitat/build` 23 G, `wo-mb-prefix-cache/build` 1.3 G, `wo-chat-template-port/build` 761 M | 25 G total | inventory only — owners decide; flagged to coord as the big lever vs the 25 G farm floor |
Post-smoke df projection: ~29–30 G free without touching other lanes.

## ARMS-PROOF GAP FINDING (2026-09-12 13:1x, from A3 q3_ci.sh review -> cold-verified)
The verified-good GATE RESOLVED serve log contains **ZERO** occurrences of `Q3G64_F16S` /
`q3_rowsplit` — serve logs never name per-tensor formats. Consequences:
1. A3's strict verifier key 'Q3G64_F16S token > 0' is FALSE-RED on a perfect serve; the
   q3_ci.sh arms-proof is simultaneously near-vacuous (bare `q3` = path echo) — defect row
   with evidence sent to gemini (their file, coord seq-22 boundary).
2. Honest log-level proof of 'no silent fallback' on THIS stack is STRUCTURAL, not grep-level:
   reader throws on unknown format (loud), typed_binding has no Q3->Q4 fallthrough (loud),
   and the 1161/Tokyo QUALITY rows would be garbage on any fallback (the 129-tensor tier map
   cannot produce '1161' from a wrong decode). Negative-guards (zero 'unknown tensor
   format|unsupported qtype|fallback|Q2' occurrences) stay in the verifier.
3. PROPOSED 1-LINE FIX (post-smoke, src is mine): loader already counts formats at bind
   (storage_layouts path); emit `load: formats {Q3G64_F16S: 129, Q5G64_F16S: N, ...}` in the
   load-complete log line. That converts structural proof into greppable arms-proof for the
   bench cell + every future CI row, and makes A3's strict key TRUE-positive-capable. Sized
   at one std::printf over the existing per-format tally; rides a follow-up commit, NOT the
   smoke (frozen binary md5 rule).

## *** SMOKE ROW — EXECUTED & GREEN (2026-09-12 09:17-09:22, coord grant seq-42, SINGLE card dev1) ***
Launch-time md5 RE-VERIFIED on disk = 7af40c2209674e78c3e3952cd6995707 (frozen-contract pass;
main pull showed docs-only delta since link proof). Artifact: wo-q3-gemv local copy, same
15,446,796,288 B identity as drive. Log: results/serve_q3_main_smoke_2026-09-12.log (this tree).
- load: 13.35 GiB / 27.5 s; KV auto 4096, runtime **654.84 MiB**, free-after-weights **2.02 GiB**
  = BIT-IDENTICAL geometry to GATE RESOLVED anchor (07:02, lane binary)
- '27 times 43' -> **'1161'** finish=stop_token gen=51 ttft=767ms
- 'capital of Japan' -> **'Tokyo'** finish=stop_token gen=29 ttft=839ms
- decode **18.3 tok/s** BOTH requests (anchor 18.3 — parity), prefill 75.7/78.7 tok/s
- negative-guard: grep -ciE 'unknown tensor format|unsupported qtype|fallback|Q2' = **0**
- release: own PID killed, dev1 to 18 MiB / dev0 241 (rustdesk), lease rm'd
**Q3-ON-MAIN PROMOTION: DONE. Receipt, not promise.** A3 cold-audit pinged on the log.
