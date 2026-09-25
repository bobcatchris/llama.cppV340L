# 11 — (a)+(b) target inventory, updated against agent4's `10e4a7cf`, with the generate run

**Author:** agent5 · **Date:** 2026-09-12 · **Ask:** coordinator seq-24, "complete missing-target
inventory so my one-line edit becomes one informed landing."
**Method:** read-only scans of `tests/CMakeLists.txt` / `src/CMakeLists.txt` / agent4's branch, plus
**one scratch `cmake` generate in /tmp on agent4's own commit** (no edits to any shared file, no
build of any product target, no device). Supersedes an earlier draft of this file whose textual
conclusion was wrong in the dangerous direction — see §5.

---

## 1. Board-state correction first (this changes who owns (a))

agent4's `10e4a7cf` — *"expose ninfer_serve STATIC target HIP-side + tests door under
BUILD_TESTING"* — **already lands both (a) and (b)**, on `amd/wo-p3-serve`. It answers the
coordinator's seq-20 ask and cites my root-free libcurl proof in its own comment. So the board line
is not "coord holds (a) until agent4 answers (b)" — it is **(a)+(b) exist in agent4's branch, and as
written the suite does not generate** (§3). My inventory is therefore for **agent4's second edit**,
and the value is unchanged.

## 2. What `tests/` links vs what exists HIP-side

`src/CMakeLists.txt` **returns at :14** on the HIP lane, so every `add_library(ninfer_*)` at lines
25–397 is CUDA-only. Reachability (not text presence) is what matters — see §5.

| target | tests linking it | main | agent4's `10e4a7cf` |
|---|---|---|---|
| `ninfer_serve` | 9 | ✗ | **✓ added** (`apps/CMakeLists.txt`, 13 serve TUs, links `ninfer_hip_host`) |
| `ninfer_ops` | 53 | ✗ | ✗ |
| `ninfer_core` (default) | 29 | ✗ | ✗ |
| `ninfer_engine` | 21 | ✗ | ✗ |
| **`CUDA::cudart`** | **17–21** | ✗ (imported only under `find_package(CUDAToolkit)`, CUDA path) | ✗ |
| `ninfer_artifact` | 14 | ✗ | ✗ |
| `ninfer_media_decode` | 1 | ✗ | ✗ |
| `crypto` | 1 | ? (OpenSSL, unassessed) | ✗ |

Note `ninfer_ops`/`core`/`engine`/`artifact` are *partially* represented on HIP — their whitelisted
sources compile into `ninfer_hip_host` — so these are **not** missing code, they're missing *CMake
targets*. An alias `add_library(ninfer_ops ALIAS ninfer_hip_host)` is the shape worth evaluating,
and it is agent4's/gemini's call, not mine.

## 3. THE EMPIRICAL PART: generate on agent4's branch **FAILS**, reproducibly

Scratch worktree at `10e4a7cf` → `cmake -DNINFER_BACKEND=hip -DBUILD_TESTING=ON` (curl via the
documented PKG_CONFIG_PATH tree): **rc=1, 72 `CMake Error` lines**, twice in independent fresh dirs
(runs reproduced: `rc=1 errors=72` both times).

```
CMake Generate step failed.  Build files cannot be regenerated correctly.
```

Error taxonomy: **25 × "Cannot determine link language for target"** + **21 ×
`target_link_libraries` naming `CUDA::cudart`** + 1 × `add_test` (tests/CMakeLists.txt:817). Offending
sites include tests/CMakeLists.txt **:60, :167, :298, :315, :770**.

Two distinct causes, and only one is about libraries:
- **`CUDA::cudart` (21×)** — a CUDA-only imported target named unconditionally. One guard fixes the
  class.
- **24 tests have `.cu` SOURCES** (e.g. `bench_kvarn_2pass.cu`) → on the HIP lane CMake assigns them
  **no language**, hence "cannot determine link language". This is the same silent-drop class
  `HipSources.cmake:208-215` documents and guards for the product build — arriving in the test dir,
  where nothing guards it. `LANGUAGE HIP` per `.cu` test is the analogue of the product fix.

## 4. The good news, and it's exactly what the sprint needs

**Zero of the 72 errors reference any of the 9 serve tests** (grep for
`serve|schema|response_store|tool_call|compaction|request_log|http_error` in the error output: **0**).
agent4's `ninfer_serve` target resolves cleanly, and the serve tests drag **no** second missing
target — they are gated on (b) alone, and (b) is done.

So the minimum path to the serve CPU-layer evidence the coordinator wants tonight is **not** "port
the suite": it is to stop the non-serve tests from generating. Options, all other-lane edits:
1. **guard `CUDA::cudart`** (21 errors) on `NINFER_BACKEND`, and give `.cu` test sources
   `LANGUAGE HIP` (25 errors) — full suite, biggest surface.
2. **a subset door**: enter only the host-only/serve tests on HIP (the 9, plus the pure-`.cpp`
   ones linking `ninfer_core`→alias), leaving the rest CUDA-only — smallest change, unblocks the
   serve evidence, matches what WO-07 actually needs.
3. agent4's own documented fallback (`ninfer_serve` before `ninfer_hip_host` in link order, drop
   serve rows from HipSources) — theirs, for symbol-collision risk, not for generate.

I implemented none of these. `CMakeLists.txt`, `tests/**`, `apps/**` are not my files, and the
anti-resurrection law makes shared build files the last place an auditor should be editing.

## 5. My own error in this task, stated because it was the dangerous kind

The draft of this file concluded from a **textual** `grep add_library(ninfer_serve` that all eight
targets were "DEFINED" on HIP — i.e. it told the coordinator (a) needed *no* target work at all.
Wrong: the definitions exist at lines 25–397, **after** the `return()` at :14, so they are
unreachable. A name appearing in a file is not a target in the build graph; past a `return()` that
distinction is exactly a **green that did not run**, which is this WO's whole theme, caught in my own
inventory. I then nearly over-corrected the other way: my second run printed `Generating done` and I
treated that as success — but CMake prints it for the root directory **before** descending, and the
tail says `CMake Generate step failed`. Both errors were caught by checking the artifact and the
exit code rather than the reassuring line, per the coordinator's own `grep -c`/`echo $?` lore.
Verified now by rc + fresh-dir reproducibility, twice.

## 6. Reproduction (cold reader)

```sh
export PATH=/home/chris/opt/cmake/bin:$PATH
E=/tmp/agent5_probe/enablement; mkdir -p $E/tree; cd $E
apt-get download libcurl4-openssl-dev && dpkg-deb -x libcurl4-openssl-dev_*.deb $E/tree
# rewrite prefix= AND includedir= in $E/tree/usr/lib/x86_64-linux-gnu/pkgconfig/libcurl.pc
# repoint the dangling libcurl.so -> /lib/x86_64-linux-gnu/libcurl-gnutls.so.4  (link-time only)
export PKG_CONFIG_PATH=$E/tree/usr/lib/x86_64-linux-gnu/pkgconfig
git worktree add -f /tmp/wt-a4 10e4a7cf
cmake -S /tmp/wt-a4 -B /tmp/b-a4 -DNINFER_BACKEND=hip -DBUILD_TESTING=ON -DCMAKE_PREFIX_PATH=/opt/rocm-6.2.0
echo $?    # 1 ; grep -c '^CMake Error' -> 72 ; tail -1 -> "Generate step failed"
```
Zero device time; the only cmake invocations were `-B /tmp/...` generate attempts, never `--build`.
