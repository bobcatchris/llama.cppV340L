# 10 — Host-test inventory & zero-GPU runnability (WO-07 follow-through on F3) + the three-state exit rule

**Author:** agent5 · **Date:** 2026-09-12 · **Tree:** `amd/wo-gfx900-perm` @ `754ed088` (code identical to
`amd/main` for `src/`+`tools/`), `amd/main` tip cited as `0ae1f903`.
**Cost:** zero device time, no grant. `tests/**` and `tools/ops/` **untouched** (§4/§7.x — gemini's).
Everything here is *inventory, classification, and a blocked run*, reported not fixed.

---

## 1. HEADLINE: the suite is not merely unwired on this host — it cannot be CONFIGURED

F3 said no CI stage runs a test. The stronger fact found while executing the assigned run:

```
configure: cmake -DNINFER_BACKEND=hip -DBUILD_TESTING=ON ...  ->  rc=1
  Checking for module 'libcurl>=7.81'
    Package 'libcurl', required by 'virtual:world', not found
  CMake Error at .../FindPkgConfig.cmake:645: required packages were not found: libcurl>=7.81
  -- Configuring incomplete, errors occurred!
```

The binding chain, each link verified rather than assumed:

1. Tests require `BUILD_TESTING=ON` (`tests/CMakeLists.txt` is only entered via CTest's gate).
2. `CMakeLists.txt:76-80` — `set(NINFER_BUILD_MEDIA_ACQUIRE OFF)` then
   `if(NINFER_BUILD_APPS OR BUILD_TESTING) set(NINFER_BUILD_MEDIA_ACQUIRE ON)`.
   **So turning testing on turns media-acquire on**, and because it is `set()` not `option()`,
   `-DNINFER_BUILD_MEDIA_ACQUIRE=OFF` is *silently ignored* — I tried it; configure still failed.
   (This is the same trap agent2 flagged for the build cell, now shown to also gate the test suite.)
3. `CMakeLists.txt:106` — `pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.81)`.
4. This host: `pkg-config --list-all | grep -c curl` → **0**; no `/usr/include/curl/curl.h`;
   only runtime libs (`libcurl4t64`, `libcurl3t64-gnutls` installed). The **`-dev` package is absent**
   (`apt-cache policy libcurl4-openssl-dev` → `Installed: (none)`, `Candidate: 8.5.0-2ubuntu10.13`).

**Classification: enablement gap, installable, not a defect.** One `apt install libcurl4-openssl-dev`
plausibly unblocks the whole suite. I did **not** install it — a system package is outside my lane and
outside any grant I hold, and per the sprint's own rule that's an ask, not a workaround.
**Ask for the coordinator:** approve the install (or have a lane with box rights do it), because every
number below is otherwise unexecutable on this box and it is the only pre-existing evidence agent4's
serve layer can get.

## 2. Inventory, and a correction to the number in circulation

| figure | how it was counted |
|---|---|
| `add_test(` occurrences in `tests/CMakeLists.txt` | **98** (what F3 and the brief cite) |
| `add_test(NAME …)` literal hits | 19 |
| **distinct test names** registered (direct + `ninfer_add_test`/`_op_test`/`_linear_test`/`_fused_linear_test`) | **152** |

So **"98 tests" is an undercount of the registered set** — the helper functions expand to more
`add_test` calls than the literal text count, and the `add_test(` grep counts lines including
function-internal ones. I could not reproduce 98 as a count of distinct tests; the reproducible
numbers are **98 literal `add_test(` lines** and **152 distinct registered names**. Do not gate on
"98" as a test count; cite one of these two with its rule. (Same lesson as §3 of the 07 doc: the
counting rule must ship with the number.)

## 3. Host-only vs device-dependent — classification method, and the honest limit

`tests/CMakeLists.txt` declares **no LABELS at all** (grep: zero `LABELS` properties), so there is no
supported way to select a host subset with `ctest -L`. The only machine-usable discriminator is
**what each test links**:

- `LIBRARIES ninfer_serve` → HTTP/schema/parse logic, host-only by construction.
- `LIBRARIES ninfer_ops` / `ninfer_add_op_test` → op kernels, device-backed at run time.
- default `ninfer_core` → mixed; some are pure host logic (arena, tensor, admission policy),
  some are plan/config logic.

**agent4's serve surface is a clean, contiguous block — `tests/CMakeLists.txt:320-346`, 9 tests,
all `LIBRARIES ninfer_serve`, none requiring a card:**

| test | covers (whitelisted `src/serve/*.cpp`) |
|---|---|
| `ninfer_openai_schema_test` | `openai_schema.cpp` |
| `ninfer_responses_schema_test` | `responses_schema.cpp` |
| `ninfer_anthropic_schema_test` | `anthropic_schema.cpp` |
| `ninfer_response_store_test` | `response_store.cpp` |
| `ninfer_tool_call_parser_test` | `tool_call_parser.cpp` |
| `ninfer_compaction_test` | `compaction.cpp` |
| `ninfer_serve_options_test` | `serve_options.cpp` |
| `ninfer_request_log_test` | `request_log.cpp` |
| `ninfer_http_error_handler_test` | `http_server.cpp` (+ `third_party/cpp-httplib`) |

Coverage of the 13 whitelisted `src/serve/*.cpp` TUs, checked by grepping `tests/` for each name
rather than by assuming the 9 tests map 1:1:

- **Directly named by a test:** 9 (the rows above).
- **Reached indirectly** — `#include`d inside a test binary that already links `ninfer_serve`, so
  the code is compiled and can be exercised even though no test is named for it:
  `translate.cpp` (4 test files), `generation_service.cpp` (test_responses_schema.cpp),
  `console_log.cpp` (test_request_log.cpp).
- **Genuinely uncovered:** `responses_http.cpp` — **0 references anywhere in `tests/`**, the one real
  hole, and it is an HTTP route file (i.e. serve-surface).

My first draft claimed 4 uncovered including `generation_service.cpp` "has no host test at all".
Wrong in the optimistic-for-my-finding direction, caught by grepping instead of reasoning: it is
included by test_responses_schema.cpp, so its code is linked into a binary that does run. Indirect
inclusion is not the same as dedicated coverage, so `generation_service.cpp` — the file agent4 is
bring-up-ing — still has **no test aimed at it**, which is the accurate way to say it. The clean
statement for the board: **12 of 13 serve TUs are reachable from the host test suite, 1
(`responses_http.cpp`) is not referenced at all; and the single most important file on agent4's path
has no dedicated test.**

All 9 serve tests are host-only by construction (pure schema/parse/HTTP logic, link `ninfer_serve`,
no card needed) — precisely the CPU surface the first real q3 serve runs through, and precisely what
nothing exercises tonight, because the suite cannot be configured (§1).

**A second trap that would have made any "green suite" a lie:** `tests/CMakeLists.txt:74` sets
`SKIP_RETURN_CODE 77` on every op test. Without a device, many tests *skip* and ctest reports them
under "Not Run"/skipped, not failed — so a headline "N tests passed" can be mostly skips. Any run
here must count `exit 0` and `exit 77` **separately**, and a stage that reports only a pass count is
uninterpretable. I could not execute to measure the real split; that number is the first thing a
runnable host will produce.

I stopped classification here rather than asserting a 108/44 host/device split: I derived it from a
**name heuristic**, and it produced obvious garbage (a phantom test named `ninfer_`, serve tests
landed in the "host-only" bucket only by luck of spelling). Publishing it would be exactly the
confidently-wrong table this WO has caught me building three times today. The link-based rule above
is sound and small; the rest needs the suite actually configured, or gemini adding labels.

## 4. What I did and did not do

Did: read declarations; ran `cmake` into a **/tmp** dir (never `/`, per instruction; disk measured
first at 26-27 G free); proved the configure failure and its cause chain; verified the missing
package three independent ways (`pkg-config`, header path, `dpkg`/`apt-cache`); cross-checked the
serve-to-TU mapping against `src/HipSources.cmake`.
Did **not**: install any system package; edit `tests/**`, `tools/ops/run_ci_amd.sh`, or
`tools/ops/gate*`; modify `CMakeLists.txt` to dodge libcurl (that would be editing a shared build
file to make my own report look executable — the failure *is* the finding); run any device stage;
claim any pass.

## 5. THREE-STATE EXIT — the rule, as text (coordinator asked for a citable form)

> **Every CI cell exits `0` = clean, `4` = hazard detected, `5` = inconclusive (analyzed, but the
> cell could not see the thing it was asked about). `0` is reserved for a cell that both ran and
> discriminated.** No cell may exit `0` on a path that did not execute, on a probe that produced no
> verdict, or on a suite whose non-zero results were skips.
>
> Rationale, all measured on this host tonight: PG-1 check (e) exits 0 while both D3 gates are blind
> to a real violation; a red check (a) reported "Failed Stages: 1" while eight checks never ran; the
> test-count guard protects a call site that does not exist; `SKIP_RETURN_CODE 77` means a ctest
> "pass" count can be mostly skips. Each of those is a **green that did not run**. A two-state exit
> cannot express the difference, and the difference is the whole content of the week.
>
> Corollaries for cell authors: (i) ship the cell with a **negative control that must fire**, and
> abort loudly (not green) if it stops firing; (ii) print reached/not-reached for every sub-check so
> sequential `exit 1`s cannot mask coverage; (iii) report the counting rule next to any number.

## 6. One-line stage design for gemini (asked-for ending; not implemented by me)

**Stage 2-alt "host ctest"**: zero-GPU, `BUILD_TESTING=ON` in a scratch dir, `-L host` **after labels
are added** (none exist today), explicitly reporting `passed / skipped(77) / failed / not-reached`
with three-state exit — the serve-label block of §3 being the highest-value first subset, since those
9 tests cover agent4's CPU surface and need no card. Prerequisite: the `libcurl4-openssl-dev`
enablement ask in §1, or decoupling `NINFER_BUILD_MEDIA_ACQUIRE` from `BUILD_TESTING`
(`CMakeLists.txt:76-80`) so testing does not drag the media fetcher in — the latter is arguably the
real fix, and it is the same `set()`-vs-`option()` shape already flagged twice.
