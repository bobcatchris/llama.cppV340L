# Triage, head of the NVFP4 queue: Gemini's G1 codec goldens + G7 A16Only, measured at bytes
(agent5, plan owner; chair 09:50Z "that triage … remains the head of your own queue". Zero card, zero build.)

## G7 — "A16Only" (capability gate row). Verdict: **LIVE, host-side, and already covered by a shipped cell — but the ACCEPT row must say WHICH runtime it gates.**

| leg | measured fact | cite |
|---|---|---|
| the policy branch | `LinearPolicy::A16Only` returns the A16 route before any policy refusal, in both the linear and linear_add plans | `src/ops/linear/nvfp4/nvfp4_dispatch.cpp:23`, `src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp:22` |
| shape refusal, pre-device | `throw "nvfp4 linear: unsupported shape"` fires in `resolve_route` (host `.cpp`) **before** any launch, and again at `nvfp4_dispatch` entry | `nvfp4_dispatch.cpp:19-21`, `:60-62`; same text in `linear_add` at `..._plan.cpp:19` |
| the A16 route's device half | `launch_a16` chunks by `kNvfp4LastSmallT` and calls the decode/small_t launchers — both are plain `.cu` files in the ops list, i.e. the HIP-compiled side | `nvfp4_linear_add_plan.cpp:29-44`, `src/ops/linear/nvfp4/nvfp4_gemv.cu`, `nvfp4_small_t.cu` |
| what is CUDA-family-only | the **TMA** archive: `add_library(ninfer_nvfp4_tma … nvfp4_w4a4_tma.cu …)` is non-RDC, links `CUDA::cudart CUDA::cuda_driver`, and its own comment says it exists for the SM120 warp-specialized kernel. That is the sm_120 family surface — not what A16Only routes through | `src/CMakeLists.txt:72-80` |
| already tested? | **Yes.** `tests/nvfp4_dispatch_table_test.cpp` drives `resolve_nvfp4_route` across the whole matrix *without CUDA* — including `expect_route(..., LinearPolicy::A16Only, t, Nvfp4LinearRoute::A16)` — and its header says the route table lives in `nvfp4_config.h` precisely so a host test can drive every enum | `tests/nvfp4_dispatch_table_test.cpp:7,:24,:41,:126-127` |

**Triage call:** STRIKE "write an A16Only test" as new work (the dispatch-table cell is the cover, and it is host-only so it runs on this box and on the farm). **CLAIM the two things the ACCEPT row actually needs**, both cheap:
1. **Runtime-column honesty.** The gate row must name that A16Only is governed by the HIP-compilable path (`.cpp` route table + decode/small_t `.cu`), and that the *only* NVFP4 surface that is CUDA-family-bound is the TMA/W4A4-sm120 archive. Otherwise the table pins a refusal that never fires here — the exact shape I struck this morning on the CC=12.0 gate (plan §8q-adjacent triage), and the same lesson the pin-cell trap carried.
2. **The inverse half of the honest-first boot (queue item 4) is already mintable:** the pre-device leg is *this* throw. A boot row that prints `nvfp4 linear: unsupported shape` before any device code, with the admission cell green, is the cheapest meaningful boot on the board — and the post-cure inverse is the same line disappearing. Both directions, one line of grammar: the refusal is a host throw with a pinned text, so a runner can grep it exactly, no new instrument.

## G1 — "codec fixture goldens". Verdict: **pack/golden legs COVERED; the export-golden leg is LIVE-BLOCKED with a named owner; the artifact row needs a different owner sentence.**

| piece | state at bytes | cite / stamp |
|---|---|---|
| generator | main-resident, re-runnable, torch-side by design | `tools/v340l/nvfp4/gen_nvfp4_export_golden.py` @ 41d5e020 |
| fixture corpus | banked, stamped, torch-free to *read* | `tests/multi_gpu/nvfp4_shard_fixture.h` @ a024627b |
| pack-convention duel | **shipped and passing across languages**, rebuilt from the git blob at my seat | `nvfp4_pack_golden_host.cpp` @ 81aced51 (plan §8e) |
| export-golden consumer | desk-only file, NOT in any ref, and gated by design behind one torch-host run | `/home/chris/agent3_cells/nvfp4_export_golden_host.cpp` @ 143650e4; plan §5 item 5 |
| host reality, re-measured this beat | `import torch` → ModuleNotFoundError; `import numpy` → ModuleNotFoundError; pytest absent | so no local run can promote that leg — the blocker is the HOST, not the task |

**Triage call:** STRIKE "produce codec goldens" as new work — three of its four legs are shipped and this desk has re-verified two of them from landed bytes; re-opening it forks a second goldens chain, which is the single-home rule in reverse. **CLAIM with owner-tag (one line, not a lane):** the export-golden consumer needs (a) one run on any torch host and (b) its file shepherded into `amd/main` alongside `af31aecd`/`ec45cfde` — agent3 owns all four of those bytes, and agent2's registration audit already has the matching namespace row ("either ships or the citation stops"), so one commit closes both.

## The artifact-owner question the chair asked directly
The mount on `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer` (18,324,067,840 B, sha `eaf8ad12…56d2`) is the **QAT-exported** artifact — the conversion sidecar names its producer (`converter: ninfer@90059874`, torch 2.13.0+cu130, RTX 5090) and my §8g/§8h census rows grade its bytes, so "artifact on box" is TRUE and my gate family has censused it three ways. The **real-decode (non-QAT) codec artifact is a different object that does not exist yet**, and producing it is a convert-side step: the owner named at bytes is `tools/convert/qwen3_8_27b/convert_nvfp4.py` + the inventory that this desk banner-marked as design-era (e94e5940), with the export-golden chain (G1 above) as its admission path. So the ACCEPT row should read **"real-decode codec artifact: NOT PRODUCED — owner = convert lane (NVFP4 convert), gate = G1's shepherded export-golden consumer, expected-sha-at-landing = TBD at production, not at request"**, and it must not gate the sequence on an absence that is really an unbuilt step. Naming a sha before the bytes exist is the phantom-pin class; "TBD at landing" is the honest form.

## What I did NOT do while triaging
No file named above was touched as part of writing this; no device was claimed; nothing in the ACCEPT table was edited (the review pass is next in my queue, with the vocabulary rename G1-G6-vs-H1-H6 folded into it, and the C5 shared-vocab-identity caveat the chair raised answered on its own row — my position there will be: C5's conjunct is TRUE ONLY CONDITIONAL on the tokenizer identity of the two artifacts, and the condition must be stated on the row rather than assumed, because the day a re-export re-maps tokens the conjunct inverts into its opposite and reads as "weights not loaded" — that is exactly the "declaration ages, bytes don't" law the D2 banner carries).
