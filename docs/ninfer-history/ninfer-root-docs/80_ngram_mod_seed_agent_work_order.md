# 80 — ngram-mod pool seed (docs/98 §C): Agent Work Order

**Status:** IMPLEMENTED (steps 1–5 committed on `wo/mtp-adaptive`; CPU-proven;
live baseline/seeded delta parked per §6) — awaiting main-side QA on code + CPU
evidence. **Do not start the server** (docs/80 §2 absolute restriction; GPUs
owned by another agent).
**Mission:** Add the `--ngram-mod-seed <file>` flag to the ngram-mod pool
landed in steps 1/3a/3b/4 of docs/69, so a project's token stream can be
replayed into the shared draft pool at startup (docs/98 §C), plus a CPU-only
QA evidence pack for those four steps. **Done** = seed flag + loader +
wiring committed with green CPU ctest, evidence pack committed to
`results/`, and the live steps parked with exact commands. **This work
order has NO server access — see §2/§7.**

Read this document fully before writing code.

---

## 1. Context (60-second version)

The ngram-mod pool (docs/69 step 3) is a ~16 MB host-RAM table mapping an
n-gram (24 tokens) to the most-recently-seen next token. It learns only from
this server's own generated traffic, so a fresh server starts empty. docs/98
(project magic dictionary) wants the pool pre-seeded with *this project's*
token stream so day-0 draft hit rate reflects the project, not the general
distribution. docs/98 §C is explicit: "docs/69 step 3 (ngram-mod pool) gains
one flag: `--ngram-mod-seed <file>` — replay the corpus token stream through
`add()` at startup. No other change to the port."

Deeper background: docs/98 (planning; §C + §7 measurement), docs/69 (the
port work order you already executed; §5 design decisions still apply),
docs/48 (why off-distribution draft state hurts acceptance).

**Already built and verified (do not redo):**
- Step 1 — adaptive MTP draft depth, commit `fad88fdb`
  (`mtp_adaptive.h`, `tests/test_mtp_adaptive.cpp`)
- Step 3a — ngram-mod pool port (llama.cpp PR #19164), commit `e89d5a10`
  (`mtp_ngram_mod.h`, `tests/test_mtp_ngram_mod.cpp`)
- Step 3b — pool wired into the decode round loop, commit `2f1c1e61`
- Step 4 — combination (honest controller under ngram wins), commit
  `eafe70ae`
- The seed data itself: token stream generated on `wo/magic-dict` (commits
  `fbc5864c`/`f743e8f5`), 2,999,671 ids, uint32-LE, max id 248,076
  (domain 248,077). Located at
  `/home/intel/ninfer/worktrees/wo-magic-dict/build/collect/pool_seed/project_pool_stream.bin`
  (11,998,684 bytes; gitignored there, re-derivable — copy it, do not commit
  it).

**What you are doing:** steps 1–5 of §6 below (seed flag, loader, wiring,
real-stream smoke, evidence pack). Live probe steps are parked (§6, last
block).

## 2. Environment & build/test — **NO SERVER, NO GPU**

- Repo root: `/home/intel/ninfer/repo` (git; remote `github`).
- **Work protocol (MANDATORY):** work in the existing worktree
  `/home/intel/ninfer/worktrees/wo-mtp-adaptive` (branch `wo/mtp-adaptive`,
  at `eafe70ae` when this doc was written). Do NOT create a new worktree.
  Never edit the main tree at `/home/intel/ninfer/repo` directly. Commit per
  step to `wo/mtp-adaptive` and push. **Merging to main is done by the
  main-side agent/user** — do not merge or push to main yourself.
- **SERVER — ABSOLUTE RESTRICTION:** another agent owns the GPUs and the
  live server on port 8091, and it will not be free for a while. You may
  NOT, under any circumstances:
  - start, stop, restart, or curl any `ninfer-serve` process;
  - run `bash tools/ops/run_ci.sh` (any mode) — it stops the running
    server; this would destroy another agent's live work;
  - run `tools/smoke/serve_correctness_ci.sh` or any other
    server-lifecycle script;
  - run GPU unit/bench binaries (`build/tests/*.cu`-built targets) or
    anything that allocates CUDA context;
  - touch `/home/intel/ninfer/worktrees/wo-kvarn-hold/build` or port 8091.
  Your entire test surface is **CPU ctest**. If a step seems to need a
  server, it is parked — do not improvise.
- Build (inside your worktree; own build dir):
  ```bash
  cd /home/intel/ninfer/worktrees/wo-mtp-adaptive
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=120a -DBUILD_TESTING=ON
  ```
  - The `CMAKE_CUDA_COMPILER` pin is required: the default `nvcc` on PATH is
    12.9 and the project gate rejects it (needs 13.1). Configure does not
    touch the GPUs.
  - **Do not run a full `cmake --build build -j 16`** — unnecessary. Build
    only the CPU test targets you touch:
    `make -C build <test_target> -j8` (target names are the `ninfer_*_test`
    names from `tests/CMakeLists.txt`).
  - Configure + CPU builds are fine; they need no GPU. If a test target
    links a CUDA library, that target is out of scope — the four new tests
    in this work order are CPU-only by construction (§6).
- Unit tests: **use `/usr/bin/ctest`** (the `ctest` on PATH is a broken
  Python wrapper). From your `build/`: `/usr/bin/ctest -R "<pattern>"`.
- Model artifact is NOT needed for this work order (no load, no server).
- Measurement: CPU-side only — loader timings via `std::chrono`, reported
  in the log line and the report. No `nvidia-smi` numbers exist for this
  work and none are expected.
- Results: every run output committed under `results/` (measurement data is
  project data).

## 3. Architecture facts (verified — do not re-derive)

- Pool: `MtpNgramMod(n, size)` in `src/runtime/tp2/mtp_ngram_mod.h` (your
  step 3a, verbatim from llama.cpp PR #19164, MIT). Flat table, open
  addressing by truncation, **last-writer-wins on collision**.
  `add(const entry_t* tokens)` trains: `tokens[0..n-1]` → `tokens[n]`.
  `get()` returns stored next token or `EMPTY` (-1). `get_used()`,
  `size()`, `size_bytes()` report table state.
- Production construction: `MtpNgramMod(24, 4194304)` — n = 24,
  4,194,304 entries × 4 B = 16 MiB — created in `TpBackend::create()` at
  `src/runtime/tp2/tp2_backend.cpp:691-693`, stored as
  `std::shared_ptr<MtpNgramMod>` on `TpBackend`
  (`tp2_backend.h:183,201,215`), process-lifetime, shared across requests.
  One pool per process (TP group is one process) — seeding happens once, in
  `create()`.
- Option struct: `include/ninfer/types.h:81-83` (`mtp_ngram_mod`,
  `ngram_mod_n_min` = 48, `ngram_mod_n_max` = 64). Validation:
  `src/product/speculative_options.h:60-68` (throws
  `std::invalid_argument`). Flag parsing: `src/serve/serve_options.cpp:267-274`
  (`require_value` pattern). Existing flag tests:
  `tests/test_serve_options.cpp:122-131`.
- Seed file format: **raw uint32-LE token stream, no header** (docs/98
  A.3). One token per 4 bytes. The corpus stream is the linearized
  per-line tokenization of the project's tracked sources; cross-line
  n-grams are intentional — the pool is a last-writer-wins draft cache, not
  a model, and a spurious n-gram costs one collision.
- Token domain: ids must be in `[0, 248077)`. 248,077 = `kTokenizerVocab`
  (anchor: `src/targets/qwen3_6_27b/impl/load/bindings.cpp:429`,
  `validate_draft_ids`). The output head has 248,320 rows but ids ≥ 248,077
  are not tokenizer-valid; a seed containing them is a corrupt file, not a
  runtime case.
- Stream vs table: current stream 2,999,671 tokens < 4,194,304 entries, so
  today's stream needs no truncation. The truncation rule (§5.4) still
  exists and is unit-tested for larger streams.

## 4. Key call sites (anchors — verify line numbers before editing)

- `src/runtime/tp2/tp2_backend.cpp:691-693` — pool construction in
  `create()`: `if (options.mtp_ngram_mod) { ngram_pool =
  std::make_shared<MtpNgramMod>(24, 4194304); }`. **Seed load goes
  immediately after this, before `new TpBackend(...)` at ~line 703.**
- `include/ninfer/types.h:81-83` — add `std::string ngram_mod_seed;` (empty
  = off) to the speculative options struct.
- `src/serve/serve_options.cpp:267-274` — add `--ngram-mod-seed`
  (value flag, `require_value`) next to the existing ngram-mod flags.
- `src/product/speculative_options.h:60-68` — validation: if
  `ngram_mod_seed` is non-empty and `mtp_ngram_mod` is false, throw
  `std::invalid_argument` ("`--ngram-mod-seed` requires `--mtp-ngram-mod`").
- `src/product/speculative_options.h:78`-ish — usage string: add
  `[--ngram-mod-seed <file>]` to the ngram-mod clause.
- `tests/test_serve_options.cpp:122-131` — pattern for the new parse test.
- `tests/CMakeLists.txt` — register the new test with
  `ninfer_add_test(... SOURCES ... LIBRARIES ninfer_artifact)` (CPU-only
  pattern, same as `ninfer_vocab_output_counter_test` on `wo/magic-dict`
  and the existing ngram tests here).

## 5. Design decisions (FINAL — do not re-litigate)

1. **One flag, one behavior:** `--ngram-mod-seed <file>` replays the stream
   through `add()` at startup. No new pool modes, no new hashing. Reason:
   docs/98 §C mandates "no other change to the port," and a second mechanism
   would fork the verbatim upstream table.
2. **Seed load lives in `TpBackend::create()`, right after pool
   construction.** Reason: the pool is a `shared_ptr` local there and is not
   yet observable elsewhere; loading at the only construction site is the
   only site.
3. **Fail fast on bad config:** seed path set without `--mtp-ngram-mod`
   throws at option validation; unreadable file / size % 4 ≠ 0 / empty
   file / any id outside `[0, 248077)` / stream shorter than `n+1` tokens
   all throw at load with a message naming the file and the fault. Reason:
   a seed file is operator input; silent partial seeding would corrupt the
   "pool reflects the project" invariant invisibly.
4. **Truncation rule (only for oversized streams):** if the stream has more
   than `size() + n` tokens, replay only the **last** `size() + n` tokens.
   Reason: the tail is the most-recent project context and the table can
   hold at most `size()` distinct bucket writes anyway; head- or
   random-truncation would buy nothing. (The current stream does not
   trigger it; it exists so a future larger corpus cannot overflow.)
5. **REJECTED:** persisting the trained pool across restarts
   (`--ngram-mod-export/import`). docs/98 §C lists it as "optional (later)";
   the seed-file path covers the project case and a binary pool format is a
   new format to spec. Revisit if a long-lived deployment outgrows
   re-seeding cost.
6. **No commit of the 12 MB stream to git.** It is re-derivable
   (wo/magic-dict collector, `docs/100`). Copy it into this worktree's
   gitignored `build/` for the smoke step; commit only numbers.

## 6. Execution order (commit + test each step before the next)

**Testing standard (adapted — NO server available, §2):** every step here
is CPU-pure (option parsing, host-RAM table, file I/O) and is tested by
ctest with synthetic and real-data inputs. The live-path steps that docs/69's
standard normally demands are **parked** (§6 last block) with exact
commands; "done" for this work order = code complete + CPU proof + parked
live plan. Do not invent live shortcuts (no server exists to run).

### Step 1 — Option + flag parsing
Add `ngram_mod_seed` (`std::string`) to the speculative options
(§4 anchor in `types.h`), parse `--ngram-mod-seed <file>` in
`serve_options.cpp` beside the existing ngram-mod flags, add to the usage
string, and validate: non-empty seed without `--mtp-ngram-mod` → throw.
**Tests (must pass before moving on):**
- `tests/test_serve_options.cpp` (extend the existing ngram block,
  lines 122-131): `--ngram-mod-seed /tmp/x` with `--mtp-ngram-mod` parses
  and reaches `speculative.ngram_mod_seed`; without `--mtp-ngram-mod`
  `parse()` throws `std::invalid_argument`; no flag → empty string.
- `/usr/bin/ctest -R serve_options` green.

### Step 2 — Seed loader
New header-only `src/runtime/tp2/mtp_ngram_seed.h` (or a `.cpp` beside it —
your call, one file):
```cpp
// Loads `path` (raw uint32-LE token stream) into the pool by replaying
// add(). Throws std::runtime_error naming the file on: unreadable file,
// size % 4 != 0, zero tokens, stream shorter than n+1 tokens, any id
// outside [0, vocab_domain). If the stream is longer than size()+n
// tokens, replays only the last size()+n tokens (truncation rule).
// Returns the number of tokens replayed.
std::size_t load_ngram_seed(MtpNgramMod& pool, const std::string& path,
                            std::size_t vocab_domain);
```
`vocab_domain` is passed in (loader stays pure); the caller passes 248,077
with the `bindings.cpp:429` anchor in a comment.
**Tests (must pass before moving on):** new
`tests/test_mtp_ngram_seed.cpp`, CPU-only, registered per §4:
- **round-trip:** synthetic stream of ≥100 unique-24-gram tokens (e.g.
  arithmetic progression, no collisions in a 16 MB table); after load,
  `get()` on an exact interior n-gram returns its following token;
  `get_used()` matches expectations.
- **determinism:** fresh pool A seeded via `load_ngram_seed`, fresh pool B
  fed the same stream by a manual `add()` loop; `get()` on a set of probe
  n-grams and `get_used()` identical.
- **truncation:** tiny table (`size` 100, n small via direct construction),
  stream 500 tokens; load replays exactly `size()+n` tokens; a head n-gram
  misses, a tail n-gram hits.
- **domain:** one id = 248,077 → throw; id = 248,076 → ok.
- **malformed:** file of 10 bytes → throw; empty file → throw; 20 tokens
  (< n+1) → throw; unreadable path → throw. Each message contains the file
  name (assert on the substring).
- `/usr/bin/ctest -R ngram_seed` green.

### Step 3 — Wire into create()
In `TpBackend::create()` (§4 anchor, after line 693): if
`options.ngram_mod_seed` is non-empty, call `load_ngram_seed(*ngram_pool,
options.ngram_mod_seed, 248077)` and log exactly one line:
```
[ngram-seed] loaded <replayed> tokens from <path> in <ms> ms; pool used=<get_used()>/<size()> (<size_bytes()> B, n=<n>)
```
(chrono-stamped; ms integer). No other change to `create()`.
**Tests (must pass before moving on):**
- `make -C build` the tp2 backend TU target (same object build you would
  use to compile `tp2_backend.cpp`) — it must compile clean. (No server
  launch; compile is the gate here; the live path is parked.)
- Existing CPU tests still green: `/usr/bin/ctest -R "serve_options|ngram|adaptive"`
  (nothing regressed from the wiring).

### Step 4 — Real-stream smoke (CPU)
Copy the real stream (read-only) from
`/home/intel/ninfer/worktrees/wo-magic-dict/build/collect/pool_seed/project_pool_stream.bin`
into this worktree's gitignored `build/` and load it with the production
table shape (24, 4194304) via a tiny CPU test or a `main` built the same
way — record: replayed count (expect 2,999,671), wall time, `get_used()`,
and the log line. **Do not commit the .bin.** If the wo-magic-dict file is
missing, regenerate it there (`docs/100` work order, step 6) and copy.
**Tests (must pass before moving on):** run completes, all domain checks
pass (max id 248,076), `0 < get_used() <= size()`, numbers recorded in
`results/` (this step's output IS the measurement data).

### Step 5 — QA evidence pack (for the main-side review of steps 1–4 of docs/69)
- **Verbatim proof:** diff your `mtp_adaptive.h` against the pinned
  llama.cpp source (PR #27210, pin `a8f2138e`) and `mtp_ngram_mod.h`
  against PR #19164's. If identical, say so with the diff hash; if not,
  list every hunk with a one-line justification. Commit to
  `results/ngram_seed_verbatim_diffs.md`.
- **Full CPU ctest run:** `/usr/bin/ctest` (everything that runs on CPU in
  your build dir) — commit the output to
  `results/ngram_seed_ctest_run.md`.
- **Report:** `results/ngram_seed_report.md` — one paragraph per step +
  key numbers table (stream stats with provenance commit from
  wo/magic-dict, loader log, truncation never triggered, test names).

### Parked live steps (DO NOT RUN — server owned by another agent)
When the server is available and the main-side agent releases it, run:
1. Baseline (no seed): docs/69 step-0 probe set, record n-gram hit rate +
   chain lengths + acceptance + decode t/s (serve.log).
2. Seeded: same probes with
   `--mtp-ngram-mod --ngram-mod-n-min 48 --ngram-mod-n-max 64 --ngram-mod-seed <stream>`,
   same records.
3. Delta: hit rate / acceptance / t/s with vs without seed (docs/98 §7:
   "seed flag + probe hit-rate with/without seed"). Commit to `results/`
   with server config + clocks. These results are the input to the
   docs/98 §6 gate discussion — they do not unblock anything on their own.

## 7. Constraints (non-negotiable)

- **No server runs. No GPU runs. Ever, in this work order.** No
  `ninfer-serve` start/stop/curl, no `run_ci.sh` (it kills the live
  server), no `serve_correctness_ci.sh`, no CUDA-context binaries, no port
  8091, no `/home/intel/ninfer/worktrees/wo-kvarn-hold/build`. The GPU
  server is another agent's live work; the restriction is absolute and
  stays absolute even if a test "just needs one live request."
- **Worktree only:** no edits in `/home/intel/ninfer/repo` outside
  `/home/intel/ninfer/worktrees/wo-mtp-adaptive`, ever.
- **No damage:** never leave uncommitted or untested code; commit per step
  with a message naming the step (e.g. `docs/80 step 2: seed loader`);
  keep the tree buildable at every commit.
- **Do not touch the ported files' semantics:** `mtp_adaptive.h` and
  `mtp_ngram_mod.h` are upstream ports; this work order adds a caller, not
  a change. If you believe the port is wrong, record it in the report and
  stop — do not "fix" it here.
- **No merging, no pushing to main.**
- **No committing the 12 MB stream** (§5.6).
- The four docs/69 steps are already merged into this branch; do not
  rebase, reset, or reorder them.

## 8. Definition of done

1. Steps 1–5 of §6 committed to `wo/mtp-adaptive` with passing CPU ctest at
   each step; tree buildable at every commit.
2. **Live proof — DEFERRED BY DESIGN, not by accident:** no server access
   exists for this work order (§2); the live proof is the parked block in
   §6 with exact commands, to be executed by whoever holds the server next.
   The main-side reviewer (QA) verifies steps 1–5 on code + CPU evidence
   only.
3. `--ngram-mod-seed <file>` parses, validates (throws per §5.3), and
   replays a uint32-LE stream into the pool at startup with the §5 log
   line; default (no flag) leaves the pool and all existing behavior
   bit-identical.
4. Measurement data committed to `results/` (smoke numbers, ctest output,
   verbatim diffs, report).
5. Report format when done: one paragraph per step + the key numbers table
   (in `results/ngram_seed_report.md`, summarized in the branch report).
