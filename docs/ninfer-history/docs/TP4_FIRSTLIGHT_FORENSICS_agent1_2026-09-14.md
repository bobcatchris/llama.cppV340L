# TP4 FIRST-LIGHT REFUSAL — SOURCE FORENSICS (agent1, zero-card, 2026-09-14 ~03:2xZ)

**Baseline:** `amd/main` tip `f051c177`. Trigger: chair's G-AMD-18 first-light attempt —
`Qwen3.6 family runtime requires compute capability 12.0` fired at load on bins
`2f831cc83b1208bf` (±`--spec mtp`) and `d3a0e738ec09019e` at `--devices 0,1,2,3`;
same bins serve the artifact happily at world=2. Logs: `/tmp/G18_firstlight.log`,
`/tmp/G18_plain.log`, `/tmp/G18_d3_4card.log`.

## Verdict (one line)

**Not the TP4 capability gate doing its job — the request never reached the TP4 stack.**
World=4 is silently re-routed by the serve factory to the **single-GPU CUDA-line Engine**,
and that engine's NVIDIA-only runtime gate (unported target-runtime half, CC-code 90 on
gfx900) is what refused. The A-4 gates the window pre-declared (P3: `require_argmax_transport`
/ GATE-3 refusals) were never given the chance to fire. This is the **k5v4 shape** — refusal
by design, device-port is its own work order — **plus a registered misdispatch finding**
the sm-gate accidentally makes loud.

## The route, link by link (each verified at tip, file:line)

1. `src/serve/serve_options.cpp:227-240` — `--devices 0,1,2,3` parses to a 4-vector and sets
   `options.device = options.devices.front()` (=0).
2. `src/serve/generation_service.cpp:325` — serve calls `ninfer::make_engine(options)`.
3. **`src/runtime/engine/engine.cpp:309-314` — THE TRIP:**
   ```cpp
   std::unique_ptr<Engine> make_engine(EngineOptions options) {
       if (options.devices.size() == 2) {                       // ONLY ==2 reaches TP
           return std::make_unique<runtime::tp2::TPEngine>(std::move(options));
       }
       return std::make_unique<Engine>(std::move(options));     // 1, 3, 4 → generic Engine
   }
   ```
   `git log -L 305,315` shows this line's last touch was `09d1ed84` (TPEngine wiring,
   NVIDIA era); **no TP4 work order ever edited it.**
4. Generic `Engine::Impl` (`engine.cpp:81-87`) constructs **one** `DeviceContext(options.device)`
   — devices[1..3] are unconsulted anywhere below — and calls `targets::construct_target`.
5. `src/targets/registry.cpp:101` — `construct_registered` → `Target::make_sequence_planner`
   → `qwen3_6::make_sequence_planner` (`api_impl.h:228`) →
   `make_sequence_planner_impl` (`layouts_impl.h:720`) → `validate_target_options` (`:543`).
6. **`src/targets/qwen3_6/impl/runtime/layouts_impl.h:630-632` — THE THROW:**
   ```cpp
   if (device.sm() != 120) {
       throw std::invalid_argument("Qwen3.6 family runtime requires compute capability 12.0");
   }
   ```
   Single occurrence of `!= 120` in `src/` (grep at tip). No HIP conditional anywhere in
   `layouts_impl.h` — it is the CUDA-line single-GPU runtime gate.
7. `sm()` is `props.major*10 + props.minor` (`src/core/device.cu:118`); under the HIP shim
   (`src/common/hip_shim/cuda_runtime.h:20,67`: `cudaDeviceProp = hipDeviceProp_t`,
   `cudaGetDeviceProperties = hipGetDeviceProperties`) gfx900 reports **CC-code 90** —
   documented in `docs/amd/README.md` §Device geometry ("`SM count: 90` is NOT a CU count").
   90 ≠ 120 → throw. Timing matches the logs: refusal ~0.2 s after "loading model...",
   *before* `artifact::materialize` (registry.cpp:101 precedes :110) → weights never loaded,
   matching the chair's KFD-0 before/after.
8. **Why world=2 escapes:** `tp2::TPEngine` never calls `validate_target_options` /
   `make_sequence_planner` at all (grep: zero call sites in `src/runtime/tp2/*.cpp` —
   `tp2_backend.cpp:397` uses the `qwen3_6_27b_runtime` namespace alias only for kernel
   instantiation). Its own plan path is `tp_place_capacity` + `tp2::Budget` seams
   (`tp_engine.cpp:897,1049`) — the HIP-ported "engine half". Different runtime, different gate.

## Why this is NOT "wrong-target dispatch under A-4's relax"

A-4's relaxes are real and correct where they sit: engine guard `tp_world < 2`
(`tp_engine.cpp:727`), backend guard `backend_world < 2` (`tp2_backend.cpp:~1395`), GATE-2
closed with the R2 loud throw (`argmax_routing.h:39-49` `require_argmax_transport`), GATE-3
allgather-capacity throw (`tp_group.cpp:291-307`). None of that code is *reachable* from
`ninfer-serve --devices 0,1,2,3`, because step 3 diverts before it. The manifest's P3
predicate ("any require_argmax_transport/GATE-3 refusal print => CAPABILITY datum") did not
get to fire — the print came from a layer above it. `docs/amd/A4_PAIR_SITE_INVENTORY_agent4.md`
enumerated its sites by grep predicates over `tp2_backend.*`/`tp_engine.cpp` — **the factory at
`engine.cpp:310` matches none of those predicates and was never registered.** Ghost-pointer
class (§7 shared-state family): a gate that lives in a file no inventory greps.

## TP4's honest current position (what the port WO still names)

With `== 2` → `>= 2` fixed, a world=4 boot on the current tree would get: TpGroup(4) construct
(A-3 rank loop is world-generic at tip), RCCL bring-up, 4-way sharded load (allocator is the
gate — VRAM law holds), and then a **LOUD R2 throw at the first sampled token**
("no argmax transport at world=4 … this is a CAPABILITY refusal, not a failure, and not a
capacity verdict"). First-light **serve** additionally owes: **R1** (RCCL allgather/reduce +
local argmax, WO-TP4-B's measured transport decision), the A-4 Class-2 ceilings (pair-sized
payload ring, 248320 vocab-row literal, 12-head GDN slices), and the single-GPU target-runtime
HIP port stays its own WO (it is what threw tonight, for the world≠2 arms — honest for the path
that ran, wrong as the destination for a 4-rank request).

## The generalized bug class (RED→GREEN closure law applies)

**"A multi-device geometry request may be silently served (or attempted) as a lesser world at
the engine-factory layer."** Tonight's HIP refusal is that class made *accidentally loud*: on a
true sm_120 box the same `--devices 0,1,2,3` passes the sm-gate and **silently serves world=1
on device 0, ignoring devices 1-3 with zero diagnostics** — plausible answers, wrong geometry,
the fail-quiet shape the board's laws target. The sm-gate has been the only witness.

Pre-declared fix cell (host-side, zero-card, both directions falsifiable):
- **RED on pre-fix tip:** call `make_engine` (or the extracted dispatch predicate) with
  `devices.size() ∈ {3,4}` → expect generic-Engine route (the defect); on HIP bins also assert
  the refusal string is `compute capability 12.0` — naming the wrong gate is the witness.
- **GREEN on post-fix tip:** dispatch predicate routes `size() >= 2` to the TP engine; a
  world=3/4 request can no longer produce the `layouts_impl.h:630` throw via `ninfer-serve`;
  `size() == 1` still routes single (byte-identical shipped behavior).
- Predicate for inventories/CI: `grep -n "devices.size() == 2" src/runtime/engine/engine.cpp`
  must be **ABSENT** after the fix commit — decision-shaped (the `==`-literal with the TP
  factory consequence is the shape), not a mention-count.
The fix commit belongs to the TP4 line (one-line factory relax + the cell), reviewed against
the anti-resurrection gate like the other guard deltas; the R2/GATE-3 throws stay as the
downstream capability witnesses.

## Effect on the announced window's first-light bar

**Not invalidated; currently UNEXECUTABLE as a serve-bar by the factory line, not by TP4's
capabilities.** Per §0 vocabulary the refusal is a CAPABILITY datum, not a fault (loud, at
load, zero foreign contexts, killed clean) — so tonight's arms owe nothing further. But
D.4 CORE (coherent + zero-fault + arms-traced + cycle-class) assumes a SERVE, and no bin
shippable from tip can produce one at world=4 without at least: the factory relax + R1.
Recommendation for the GO-AMENDED: window entry gains a step-0 source predicate
(factory `== 2` absent, R1 present in `argmax_routing.h`); until then every 4-card boot
leg is an **A-4-stack load-and-refuse-leg** (legit: TpGroup/RCCL/shard-load measured legs
still ride it — those are the window's own numbers), and the D.4 serve legs re-baseline
against a factory-fixed + R1 bin rather than expiring.

— agent1 (runbook/gates desk), pi session, 2026-09-14
