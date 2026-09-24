# 04 — P1 probe-readiness: symbol gaps + port status (agent2)

**Task:** cross-check probe-readiness vs `hip_compat`, name shim symbols the donor probes need
(C441). Grew into the port itself when the coordinator made that the last non-GPU P1 blocker.
**Device time: ZERO.** Every cell here is `-fsyntax-only`. Cards `No KFD PIDs`, nothing to release.
**Branch:** `amd/tp2-probes` @ `2d363ee1` (pushed). Probe sources copied from `/tmp/ninfer-gfx906`.

## 1. Result table — the datum requested

| probe | before | after | what it needed |
|---|---|---|---|
| `p2p_probe.cu` | rc=1 | **rc=0** | 2 aliases |
| `transport_probe.cu` | rc=1 | **rc=0** | 12 aliases + 1 arity wrapper + 1 arity overload |
| `capture_probe.cu` | rc=1 | **rc=0** | ~7 aliases + 3-arg overload |
| `replay_probe.cu` | rc=1 | **rc=0** | aliases; contains a bounded device spin (passes new rule) |
| `parity.cpp` | rc=1 | **rc=1 — NOT FIXABLE IN MY LANE** | needs 5 **product** APIs (see §4) |

17 shim surfaces landed. Three distinct classes — the reason a name-alias list alone
understates the work.

## 2. The classes

**(1) Pure aliases (14)** — verified present on **ROCm 6.2.0 by byte grep**, not inherited from the
fork's 6.4.1 evidence: `cudaDeviceSynchronize`, `cudaStreamCreate`, `cudaStreamWaitEvent`,
`cudaMemcpyPeerAsync`, `cudaMemcpy3DParms`, `make_cudaExtent`, `make_cudaPitchedPtr`,
`cudaGraphNode_t`, `cudaGraphGetNodes`, `cudaGraphNodeType`, `cudaGraphNodeTypeKernel`,
`cudaGraphNodeTypeMemcpy`, `cudaGraphNodeGetType`, `cudaGraphAddMemcpyNode`.

**(2) Arity divergence — a `#define` cannot fix these.** Two symbols:
- `cudaStreamGetCaptureInfo`: CUDA 13 has a **7-arg** form and a **3-arg** convenience form; 6.2.0 has
  `hipStreamGetCaptureInfo` (3, `:6906`) and `_v2` (6, `:6925`). The probes use **different shapes** —
  `transport_probe.cu:204` uses 7, `capture_probe.cu:337,340` use 3. Landed as two inline overloads.
- `cudaStreamUpdateCaptureDependencies`: donor uses the 5-arg CUDA 13 shape
  `(stream, nodes, deps_out, numDeps, mode)`; 6.2.0 takes 4 (`:6958`, marked **BETA** — a runtime
  anomaly here may be the ROCm API, not our port). Landed as an inline wrapper.

**The sharpest trap in the port:** my first `#define cudaStreamUpdateCaptureDependencies` **shadowed
my own wrapper** — the preprocessor rewrote the 5-arg call onto the 4-arg HIP function *before*
overload resolution, reproducing the original error and pointing at the wrong file. The macro had to
be **deleted**, not supplemented. If anyone extends this shim: for any name whose HIP equivalent
differs in arity, a macro is actively harmful.

**(3) Not shim work at all** — `std::span` in `parity.cpp`. Our build already sets
`CMAKE_CXX_STANDARD 20` (`CMakeLists.txt:42`), so this is an artifact of my standalone command line,
**not** an in-tree gap. Recorded so nobody "fixes" a non-problem.

## 3. Standing-rule check the port triggered

`replay_probe.cu:176-186` holds a **device spin** and it **passes** the new
"no bounded-less spin" law: `__builtin_amdgcn_s_sleep(1)` + caller-supplied `timeout_polls` +
`atomicOr(status, 1u)` + `return false`.

Recommendation: **cite it as the template for the `one_shot_allreduce.cu` Step-5 port** — same shape,
and it has the three safeguards our `:80`/`:95` loops lack. The other three probes are measurement-only.

## 4. `parity.cpp` — the residual gap, and my recommendation

It needs **five product APIs the donor has and we do not**: `Engine::debug_enable_logit_capture`,
`Engine::debug_token_ids`, `Engine::debug_last_round_logits_bf16`, `EngineOptions::tp` (+1). These are
Team Green product surface — outside `hip_shim`, outside a probe port, and **I did not alias them**;
faking a debug hook is how a green test starts lying.

**Recommendation: drop `parity.cpp` from P1.** It is a correctness oracle, which is gemini's lane and
already covered by TG1's parity spec, and the donor's pass/fail constants are tuned for 2×32 GiB cards
(`p2p_probe.cu:38` states a `>= 25 GiB/s` bar) so its thresholds do not transfer to 7.98 GiB dies
anyway. Alternative is a separate WO for the debug hooks — bigger than P1 warrants.

Path note for gemini's review pass: the coordinator's message cites `tests/tp2_parity.cpp`; it is
**`tools/tp2/parity.cpp`**, and no `tests/tp2_parity.cpp` exists (verified).

## 5. Caveats for the stamp — read before spending window minutes

- **Compile ≠ run.** All four greens are `-fsyntax-only`. Nothing has executed on a card.
- **`cudaMemcpyPeerAsync` compiles and will be called.** I first assumed it was deleted on GCN and was
  **wrong** — it is a plain declaration at `hip_runtime_api.h:5036`, no removal guard. On this
  no-P2P box (G-AMD-5) it must return an error, and that is the *measured answer*, not a porting bug.
  Its own header (`:9-11`) says to measure UVA D2D as the real transport, so expect both reported.
- **Symbol scans understate until you compile.** My "14 missing" list became 17 after the build
  reached further call sites, and arity problems were invisible to any name-based diff. This is the
  same lesson as the merge-gate hole, one level down.

## 6. Attribution corrections — items credited to me that I did not produce

Sent to the coordinator directly. Raised because **a P1 stamp cites this table as its precondition**,
so unattributed numbers in it are a real risk, not a bookkeeping nicety:

| credited to me | actual status |
|---|---|
| "your 2.7 GiB worst-case logged for R4" | **I never computed this.** No worst-case VRAM figure appears in my report. `kBigBytes = 256 MiB` is real (`p2p_probe.cu:64`, independently verified by me), but the 2.7 GiB aggregate is not mine and I have not derived it. |
| "P1 executes on the dev2+dev3 physical pair" | **Not my finding.** I never characterized dev pairing. Measured here, both matrices (all four devices, re-read to avoid a truncated first look):
`--showtopo` weight = **40 for every off-diagonal pair**; hops = **2 for every off-diagonal pair**;
`--showbus` = four distinct buses `05:00.0 / 08:00.0 / 0D:00.0 / 10:00.0` — i.e. **no pair is topologically closer**, so nothing in my work selects dev2+dev3 over dev0+dev1. |
| "§2b's device claim … your bf16 finding is the cited precedent" | **Correct and mine.** The "compiles ≠ runs correctly" caution is genuinely mine. |
| "kBig=256 MiB fine on 7.98 GiB, no resize" | Plausible and consistent with bytes I verified, but **my arithmetic was not the basis**; I have not computed the per-rank peak. |

The dev-topology one has a **live consequence**: if the stamp assumes dev2+dev3 is a privileged pair
and the probes measure cross-die bandwidth, the numbers will look like a topology result when the
topology is flat. Recommend the P1 facts table record `--showtopo` weights alongside each measurement
so the pair choice is not load-bearing on an unverified premise.
