# 12 — Authoritative test-suite inventory for the HIP lane (supersedes `results/amd/wo07_s1/host_test_pertable.txt` drafts)

**Author:** agent5 · **Date:** 2026-09-12 · **Ask:** coordinator seq-29 items (1) number
reconciliation and (2) the 136 re-run with the correct walker.
**Method:** the **generated** `tests/CTestTestfile.cmake` from a scratch `/tmp` cmake of agent4's
`10e4a7cf` is treated as the authority on *what exists*; `tests/CMakeLists.txt` text is used only to
recover each test's `SOURCES`/`LIBRARIES`. Zero device time, zero product builds, no shared file
edited.

---

## 1. THE NUMBER, with one rule — and both of my earlier numbers were wrong

| claim | rule | status |
|---|---|---|
| `98` | count of literal `add_test(` **lines** in `tests/CMakeLists.txt` | true but not a test count |
| `144` / `145` | my regex walk of declarations (`145` = with one repeat; `144` = distinct names) | **both undercounts** |
| **`174`** | `ctest -N` on the generated file — **distinct registered tests** | **AUTHORITATIVE** |

**The reconciliation the coordinator asked for, and it is not a footnote — it is a bug in my walk:**
`174 − 144 = 30` (my walk found 144 distinct; `ctest` reports 174 names, 174 − 145 raw matches with
the repeat = **30** net, **31** names genuinely missed minus `ninfer_`, a regex artifact of matching
`add_test(NAME` — see §4). Cause of the miss:

```cmake
# tests/CMakeLists.txt:381-385
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test ...)
endforeach()
```

`ninfer_add_test`-style declarations inside a `foreach` expand at generate time, so **no text
scanning can enumerate them** — 23 op tests (`ninfer_argmax_test`, `ninfer_l2norm_test`,
`ninfer_layer_norm_test`, `ninfer_cast_test`, `ninfer_embedding_test`, …) plus the
`${op}`-templated forms were invisible to my walker. I had written the caveat *"a test defined via a
loop/foreach would be missed; none found"* — **that statement was not merely unmet, it was false**,
which is worse than having no caveat, because the caveat is what made the number look checked.

**Rule to cite from now on: 174 registered tests (`ctest -N` on a HIP configure with
`BUILD_TESTING=ON`).** `98`/`144`/`145` retired; if quoted at all, quote them with their counting
rule and the word "not the test count."

## 2. Classification over the authoritative 174

| class | count | meaning |
|---|---|---|
| **HOST-OK** — builds on the HIP lane today | **9** | links no CUDA-only target, no `.cu` source |
| BLOCKED on a missing target only | 141 | needs an alias/target (`ninfer_core`, `ninfer_ops`, …) |
| BLOCKED on target **and** language | 24 | `.cu` source with no `LANGUAGE HIP` + missing target |
| BLOCKED on language only | **0** | *see §3* |

**Demand across the 174** (a test may carry several):
`ninfer_core` 84 · `ninfer_ops` 73 · `.cu` source 24 · `ninfer_engine` 21 · `CUDA::cudart` 17 ·
`ninfer_artifact` 14 · `crypto` 1 · `ninfer_media_decode` 1.

## 3. My earlier remediation advice was wrong, and this is the actionable part

Previous draft proposed two cheap class-fixes — *guard `CUDA::cudart`* and *add `LANGUAGE HIP` to
`.cu` tests* — as if each unlocked a tranche of tests. **Measured: each buys ZERO tests on its own.**
`BLOCK-lang-only = 0`: every `.cu`-sourced test *also* links a missing target, so fixing the
language alone leaves it unbuildable, and identically for the cudart guard. The single lever is
**target aliases**, largest first: `ninfer_core` (84 demand) and `ninfer_ops` (73). An
`add_library(ninfer_core ALIAS ninfer_hip_host)`-shaped move is the decision worth evaluating —
with the §5 symbol-closure caveat, which I have *not* tested and which is the difference between
clearing a CMake error and having a runnable suite.

## 4. Two findings that fell out of the reconciliation

1. **`ninfer_dflash2_symbols_check` is registered twice** — `tests/CMakeLists.txt:816-826`, an
   `if(NM_TOOL)/else()` pair of `add_test` with the **same name**. CMake keeps one (generated file
   shows the `nm` branch, backtrace to `:817`). Harmless today, but a same-name double
   registration is precisely how a check silently stops being run: the second definition wins or is
   dropped depending on CMake version. Worth one line to gemini.
2. **The `else()` branch is a self-declared no-op that passes.** Same block:
   ```cmake
   else()
     add_test(NAME ninfer_dflash2_symbols_check COMMAND ${CMAKE_COMMAND} -E true)
     message(WARNING "nm not found: ninfer_dflash2_symbols_check is a no-op")
   endif()
   ```
   If `nm`/`llvm-nm` is absent, a **symbol-verification test exits 0 while checking nothing**. Here
   `/usr/bin/nm` exists, so today it is real — but this is the *fourth* instance of tonight's theme
   (dead ctest guard, phantom stages 3–5, a green configure registering nothing) and the only one
   **written on purpose with a warning**, i.e. the pattern's shape: a pass path for the missing-tool
   case that reports success. The honest variants are skip-with-nonzero-status or a hard fail; `-E
   true` is neither. My own classifier nearly repeated it (false "none found" caveat), so I am not
   going to soft-pedal it in someone else's file.

## 5. Limits (stated with the failure mode, not just the gap)

- **HOST-OK = the CMake graph can build it.** Not that it passes, not that it avoids a device at run
  time: `tests/CMakeLists.txt:74` sets `SKIP_RETURN_CODE 77`, so a real host run must report
  **passed / skipped(77) / failed / not-reached** separately. Four-count is law per coordinator.
- **Alias demand is name-level.** `ninfer_hip_host` carries only whitelisted sources, so an alias may
  clear the CMake error and still fail at link on undefined symbols. **Unverified; it is the next
  question, and nobody should plan 84 tests off §2 without it.**
- Generated-file authority is specific to this configure (`10e4a7cf`, `BUILD_TESTING=ON`, root-free
  libcurl tree). A later commit adding targets changes the numbers; re-running is deterministic and
  cheap, and §1's rule (`ctest -N`, not grep) is what survives the change.
- The `ninfer_` phantom name was my regex matching `add_test(NAME` on a line where the name token
  began with `$`; dropped as noise, noted so nobody re-finds it as a real test.

## 6. What the serve window actually needs (the point of the exercise)

**9 tests, all `LIBRARIES ninfer_serve`, no `.cu` sources — the entire set is agent4's serve CPU
surface**: `openai_schema`, `responses_schema`, `response_store`, `anthropic_schema`,
`tool_call_parser`, `compaction`, `serve_options`, `request_log`, `http_error_handler`. They need no
new target work: agent4's `10e4a7cf` already supplies `ninfer_serve`, so the only barrier is the
**136 others failing generate** (§2/§3 of the v340l/11 inventory). A subset door — enter only these 9
on the HIP branch, or gate the rest out — converts the coordinator's porting-project worry into a
small scoped edit, and 12 of the 13 whitelisted `src/serve/*.cpp` TUs become exercised for the first
time. Kept visible as this task's lesson, per the coordinator's instruction: I shipped a 9-only
draft with the 136 *named as pending* rather than publish a table I knew undercounted 7:1 — and the
final numbers still exceeded *both* my earlier counts, which is the argument for making
`ctest -N`, not text scanning, the rule.
