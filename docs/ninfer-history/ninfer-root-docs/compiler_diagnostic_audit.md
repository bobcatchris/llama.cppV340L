# Compiler-diagnostic reliance audit (2026-09-05)

Read-only pass, greenlit by the coordinator after the `-Wswitch` finding. Question: where does this
repo rely on a compiler diagnostic to catch a defect, and does the build actually enable it?

## Headline: the build enables **zero** warning flags

- `CMakeLists.txt`, `cmake/`, `src/CMakeLists.txt`: the only `target_compile_options` in the repo
  is `-lineinfo` for CUDA. No `-W` flag anywhere.
- `build/compile_commands.json`: contains no `-W*` option at all.
- `tools/ops/run_ci.sh` and every script under `tools/`, `ci/`, `.github/`: **no** `-Werror`,
  `-Wall`, or `-Wextra`. There is no strict-warning build path in the repository.
- `-Wswitch` is **not** default-on in this GCC. Minimal repro: `enum class E{A,B,C}` with a switch
  covering A and B is silent under default flags and warns only with `-Wswitch` or `-Wall`.

Consequence: any correctness property stated as "the compiler will catch it" is currently
unenforced, and would stay unenforced even if the code were deleted entirely.

## Findings

### 1. `-Wswitch` reliance — UNFOUNDED (found and corrected)

`d3e88039` (A2) and my own review recommendation both rested on "exhaustive switch with no
default, so `-Wswitch` fires when an enumerator goes unmapped." Verified false in this build: the
same TU that warns under `-Wswitch` is silent under the project's real command line.

This claim survived a commit message, a doc section (docs/151 §18) and two reviewers, and was
falsified by one compile.

Status: A2's exhaustive-switch *shape* is still correct and worth keeping (it is right, and helps
if the flag is ever enabled), but the **load-bearing** guard is the enumeration test, which is
build-flag independent.

### 2. "clean under `-Wall -Wextra -Werror`" claims — TRUE but UNMAINTAINED

`fab93b97` and `0ac03096` both record tests passing "clean under -Wall -Wextra -Werror and
ASan/UBSan". These were almost certainly true when written, from a manual invocation. But:

- no build or CI path enables those flags, so nothing prevents a later edit from regressing them;
- the claim is scoped to **those specific test files**, not the repo. It reads broader than it is.
  For scale, `-Wextra` on the 27B variant TU produces several pre-existing
  `missing-field-initializer` warnings in `program_impl.h`, so a repo-wide `-Werror` build would
  fail today.

Not a false claim. A snapshot described in the present tense, which is how it will be misread.

### 3. No repo-wide warning hygiene exists

Worth stating plainly since it is the generalizable result: this repository has no mechanism that
keeps any warning clean. Warning-based claims are therefore all point-in-time.

## Recommendation

1. **Add `-Wswitch` narrowly.** Measured: `-Wswitch` alone on the 27B variant TU yields exactly
   **1** warning — the real, currently-pending `DFlash2` unhandled case. No flood. One line in
   `src/CMakeLists.txt`. This is defense-in-depth, not the safety net.
2. **Do not add `-Wall -Wextra -Werror` repo-wide** without a cleanup pass — it fails today.
3. **Treat enumeration tests, not compiler flags, as the guard** for "every enumerator is
   handled". The A2 §19.1(a) test is the right instrument precisely because it does not depend on
   a build flag anyone can forget to pass.
4. When writing a claim of the form "the compiler catches X", state **which flag** and **where it
   is enabled**. Otherwise the claim is about a hypothetical build.
