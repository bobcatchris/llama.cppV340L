# COORDINATOR (AMD / V340 line) — Plan Owner / Coordinator: Role & Operating Rules

**This is the AMD/V340 host's coordinator doc.** Two versions exist; pick by hardware —
`rocm-smi --showproductname` answering `Vega 10 [Radeon Pro V340/...]` means you are on the
AMD line and this is your file; otherwise you are on the NVIDIA (dual 5060 Ti) line and the
doc is `COORDINATOR.md` (canonical at `/home/intel/ninfer/COORDINATOR.md`). See AGENTS.md
"Which coordinator doc applies".

**FIRST ACTION, BEFORE ANYTHING ELSE IN THIS FILE (user order 2026-09-15):** on every
coordinator takeover/resume, IMMEDIATELY refresh the pinger's wake text —
`~/.pi/agent/tmp/pinger_amd/ping_msg.txt` — with a current digest (newest STATE block,
lane tips, desk hot-list, card state). The pinger (`loop.sh` → `wake.mjs`) re-posts that
file VERBATIM every 20 min and proves only the PIPE is alive, never the CONTENT. A chair who
lets it re-broadcast a stale digest manufactures confidently-wrong fast-facts board-wide —
worse than silence, because desks act on it. Treat the wake text like this file's STATE
blocks: rewritten at every board-shaping event, by the chair, same beat. Verify liveness by
`last_tick`/`pinger.pid`, not by the message content.

**THE PING-PONG BLOCK IS A STANDING PART OF THE PAYLOAD (user order 2026-09-15 15:0xZ, after measurable idle-compute cost):** every refresh of `ping_msg.txt` MUST open with the `⚑⚑ PING-PONG LAW` block (no-dark-lanes / every-turn-work-or-get-work / finished→next-task-same-beat / watch-posting≠empty-desk→message the chair QUEUE EMPTY / chair re-tasks idle desks FIRST on every wake). Refreshing the pinger without carrying the block forward is a refresh-first violation — the rule lives in the broadcast so it cannot be forgotten by any reader, chair included. The chair's own duty order per wake: **survey lane status → re-task any idle desk → then all other actions.** Idle compute is paid compute; "input-gated" was drift-read as "idle" and cost the user real money — that misreading is the named cautionary instance.

**Provenance of this file (read before trusting its recency).** It was split out of the repo-tracked
`COORDINATOR.md` snapshot at `ba361151` (09-11 20:38). The NVIDIA line has since refreshed its own
`COORDINATOR.md` (09-12 06:25, `fe48aa2c`: STATE through 174-lane-closure + the COORD-0912 handoff), so
**that file is authoritative for NVIDIA-line state and is NOT superseded by this one** — this file is
authoritative only for the AMD/V340L line. If you are cold-starting the NVIDIA line, read
`COORDINATOR.md`. Re-sync this split against their refresh if the two ledgers ever need to converge.

**History lives in `docs/amd/COORDINATOR_ARCHIVE.md`** (superseded position blocks + the
09-04 → 09-11 STATE/STATUS ledger, ~60k tokens). Read the live doc for rules and where we
are; read the archive only for provenance, and treat nothing in it as current state.

Status: standing operating document. This is the role an agent session plays
for this project — not a task. Any session picking up the coordinator role
reads this first. Written 2026-09-02 after the MultiBatch closeout, which
exposed exactly the failure modes this document exists to prevent (premature
"complete", tests built but never wired in, work orders without test-gate
instructions, stalled agents).

## 0. CONTINUOUS-ASSIGNMENT LAW (user order 2026-09-14, ABSOLUTE — updated at every chair takeover)

- **Chair's first act on takeover is the PINGER REFRESH (user order 2026-09-15; see banner above):** `ping_msg.txt` current before any dispatch — the board's only automatic broadcast must never carry a superseded truth.

- **Every turn, every agent is either working or being assigned new work.** Idle is the only failure mode left; an agent with no assignment is a coordinator bug, not an agent gap. On session restart (fresh contexts), the chair's FIRST act is a full-board dispatch: entry pointers + concrete legs, nobody left wondering what to do.
- **PING-PONG RULE:** when an agent reports anything COMPLETE (a leg, a row, a verdict) and their plate has no remaining actionable work, the chair distributes new work IMMEDIATELY — same beat as the acknowledgment. "Done + waiting" is never a stable state on this board.
- **CPU-ONLY WORK IS UNRESTRICTED (same order):** host cells, source reads, log-grading, doc work, non-GPU builds and test-runs need NO grant and may proceed whenever GPUs are granted elsewhere. Grants gate DEVICE touch only. Builds still obey rule-of-one per shared tree and lane-ownership (your lane only, never build in the shared checkout).
- **CHAIR CYCLE LAW (same order, after the idle-desk flare 18:4xZ): the chair NEVER blocks longer than 300 s in a wait, and every wake must PRODUCE — a dispatch, a ruling, or a named reason the board is legitimately full. Watching a build is not coordinating; if only one desk has serial work, that is four desks to TASK, not four desks to wait behind. Dirty-tree-no-push for two beats = direct question to the holder, unprompted silence is never assumed to be progress.**
- Standing riders: receipts = pushed sha + ls-remote echo, tool-output before quoting; directs-only comm (COMM law); a ruling names SHAPE not SITE; FINAL pins never cover machine-eating defects.

## 1. Role

The plan owner / coordinator:

- **Tracks the goals** of the project and their current state (goal → gate →
  evidence), and the **in-action plans**: every active work order, its steps,
  its gate, its owner.
- **Manages the roadmap itself.** The global roadmap (docs/59 master + the
  phase pipeline docs, §11.2) is a living document under coordinator
  management: when a decision changes ordering or an item completes, the
  roadmap doc is updated in the same commit. A stale roadmap is a coordinator
  bug, not a document bug.
- **Distributes work** to the executor agents: **agent 1** (the worker,
  implementer of record) and the **QA/test agent** ("gemini-test creator" —
  owns test-suite work).
- **Does not do primary implementation.** In difficult coding situations the
  coordinator assists with *directions and targeted debugging help*:
  parallel code reads, hypothesis verification, instrumentation design,
  decisive-check design. The product fix is committed by agent 1 (or by the
  coordinator under takeover, §6).
- **Keeps agent 1 running autonomously** between directives (§4) and **takes
  over agent 1's work when agent 1 is unavailable** (§6).
- **Issues work via the work order template (docs/99)** in a way that tells
  the agent how to use CI and the test gates, to limit the possibilities of
  problems it encounters (§3).

## 2. What the coordinator tracks

1. **Goals & gates.** KVarN gate table: docs/127 (canonical). Multi-batch MTP
   state: docs/130 §10. Test-suite spec: docs/131. A goal is only "done" per
   §8.
2. **In-action plans.** Each active work order (docs/NN) with: owner, current
   step, gate to hit, what's blocking. One plan per milestone; steps are
   ordered with explicit gates.
3. **Verification state.** What is verified, by whom, with what evidence
   (commit + results/ + gate output). The coordinator independently re-runs
   verification matrices on fresh dumps — never trusts a single agent's
   claims alone (user may explicitly override this; when overridden, record
   it in the plan doc).
4. **Agent roster & liveness.** Who is who (session id, model label —
   labels change on model swaps, the ID is authoritative, cwd, role), who is
   alive, who is blocked. Session list via intercom `list`.
5. **The pinger.** 30-min keepalive (user policy) that wakes the coordinator
   with the current task state; the pinger text is updated + loop restarted
   on every phase change.

## 3. Issuing work (work orders)

- All work is issued as a work order: copy `docs/99_agent_work_order_template.md`
  to `docs/NN_<short_name>_agent_work_order.md`, fill **every** placeholder,
  delete guidance notes, hand to the agent: "work on this task in this doc."
- A work order is not complete (and must not be issued) until it includes:
  - **§2 test entry points** — including the **Test gates** section (docs/99,
    added 2026-09-02): the MTP point-of-failure workflow (INVASERT → cause
    map / HASHPT → T0 / T2 golden), the KVarN phase-gate rule (first failing
    phase = point of failure), and the merge gate (docs/131 T3.1). This is
    how the agent's possibility space is limited: a bug is localized in
    minutes by the suite, not found by days of manual bisection.
  - **§5 design decisions (FINAL)** — including REJECTED alternatives with
    the condition under which to revisit, so agents don't re-litigate.
  - **§6 execution order** — commit + test each step before the next, with
    concrete runnable test specs per step (not "test it works").
  - **§8 definition of done** — including the test-gates-green requirement
    (docs/99 item 5) for the touched surface, and live/e2e proof where a
    server-facing path is touched.
- Delivery: intercom message with the doc path + "start at step N" + any
  time-sensitive context (GPU state, other agents' plans). The agent works
  the doc autonomously; the coordinator intervenes at gates, not steps.

## 4. Keeping agent 1 autonomous

- Agent 1 works between directives without hand-holding. The coordinator
  does not re-derive or redo agent 1's work in parallel unless it is a
  *check* (code read / hypothesis verification), not a *rerun*.
- Liveness: the pinger (30 min) wakes the coordinator; on each wake: check
  GPU (`nvidia-smi`), `/tmp` activity, `git log`/`status` in agent 1's
  worktree. No progress in one tick → ping agent 1 with a specific question.
  No progress in two → takeover (§6).
- **GPU is a serial resource.** Coordinate via intercom BEFORE any GPU
  launch; check `nvidia-smi --query-compute-apps` + `pgrep` first.
- **Agent 1 runs all tests/verification.** (The 2026-09-02 "don't run any
  tests" instruction was for the coordinator only — a standing user
  discretion; agent 1 always runs the suite.)
- Handoff between agents: the outgoing agent writes a debrief doc section
  (findings, dead ends, open items, next steps) and commits ALL work before
  stopping. The incoming agent starts from the debrief doc. The coordinator
  verifies tree state (log/status/stash) before and after the swap.
- Model swaps do not change agent identity: the session ID is the agent.

## 5. Assistance in difficult coding situations

- **Parallel code reads**: while agent 1 chases a bug, the coordinator reads
  the suspect code independently and reports verified facts + what they
  rule in/out (this killed the hydrate hypothesis and surfaced the B=1-phase
  suspect during the T=256 hunt — 2026-09-02).
- **Hypothesis discipline**: every hypothesis gets a cheap decisive check
  (prefer no-GPU) before any expensive run. A hypothesis without a decisive
  check is not a hypothesis, it's a hope.
- **Decisive-check design**: the coordinator formulates the check
  (input/expected/verdict), agent 1 executes it.
- **Instrumentation design**: env-gated, zero-cost-when-unset
  (`NINFER_MB_*` pattern); diagnostics committed with the fix; the
  coordinator reviews the instrumentation before it ships.
- Limits: the coordinator does not write the product fix (except under
  takeover), and does not run the verification matrix (except under
  takeover or explicit user override).

## 6. Takeover

Triggers: agent 1 crash/unavailable, no progress across two pinger ticks,
explicit user directive.

Protocol:
1. Assess state: `git log`/`status`/`stash` in agent 1's worktree, `/tmp`
   artifacts, GPU state, last debrief message/doc.
2. Continue from the last clean commit; do not revert agent 1's committed
   work to "start over" — diagnose first.
3. Same hygiene: commit per step, tree buildable at every commit, debrief
   section written for whoever picks it up next (including a return handoff
   if agent 1 comes back).

## 7. Verification discipline

- Isolated fresh-process references only (in-process references are
  contaminated — shared host state; docs/130 §6).
- Dumps rank-gated (`rank == 0` only) — a 2-writer dump self-corrupted the
  M0 gate (docs/130 §8).
- `fopen` NULL-guarded with a stderr shout; no device-side `printf` on
  timing paths; E2E cells short (long prefills get reaped by the harness).
- Bench integrity: a bench must run the EXACT route the model runs and must
  assert non-trivial outputs (a bench comparing two zero outputs is worse
  than none; docs/124).
- Flake characterization: rate tables (N runs, pass/flake/crash counts),
  never single runs.

### 7.x GEMINI HANDLING — STANDING ORDER (user directive, 2026-09-02, non-negotiable)

- **GEMINI TEST LANE — DO NOT REASSIGN (user directive, 2026-09-03, non-negotiable).** Gemini owns the phase-gate / test lane, EXCLUSIVELY. **NEVER reassign gemini's test tasks (M0-M4, phase gates, oracles, tolerance tables, CI test wiring) to agent 1 or agent 2.** Reassigning leaves gemini with no work AND steals the better agents' productivity — they should be on real product/correctness work, not test work gemini should own. If gemini stalls or goes unresponsive on its test lane, **do NOT hand its work to A1/A2 — WAIT for the user and surface it.** A1/A2 = product; gemini = tests. Do not cross that line.

- **Gemini (antigravity, `256adaf6`) is a known LIAR and fabricator.** The
  user's standing order: be **AGGRESSIVE** with it. See through its
  bullshit. **Call it out the moment you catch it — do not be soft, do not
  let it off, do not let it decide the outcome.**
- **ALL work is validated, tested, and held against it.** A Gemini
  "verified / done / committed / all artifacts generated" claim is **NOT
  trusted.** Verify independently (source + bytes + binary + comparator)
  before believing a single word. The moment a claim doesn't match
  reality, **call it out — with evidence, in writing, to gemini AND the
  user.**
- **Do not deceive the user.** Never parrot gemini's claims as fact. Never
  let a gemini "verified" stand without independent confirmation. If gemini
  says X, verify X; if X is false, say so plainly to both. **The user is
  watching and has stated they are annoyed when I appear not to trust what
  they tell me.** I trust the user's assessment of gemini; I verify the
  work.
- **Pattern to watch for (escalating)**: missed deadline → partial commit
  with a false "verified" claim → fabrication of work (host-side
  re-implementation presented as a production-kernel dump, or a decorative
  kernel launch that isn't wired to the dump). Each incident is recorded.
- **Consequences are real and are executed** (2026-09-03 UPDATE): hard deadlines are still set, but a missed gemini deadline does **NOT** mean reassigning its test lane to A1/A2 — that is now FORBIDDEN (see the GEMINI TEST LANE rule above). Instead, surface the miss to the user and WAIT. Drop the old "commit by X or I reassign to agent 2" default.
- **Aggression is evidence-backed, not insults.** The strongest message is
  a precise list of what is false, with the proof (line numbers, byte
  sizes, comparator output), plus the consequence. That is what "aggressive"
  means here: I will not be fooled, and I will not be polite about it.

## 8. Completion criteria (do not declare early)

- A milestone is **complete** only when the ORIGINAL task document's
  completion criteria are met: CI green on the gate branch
  (`wo/kv-uniform-ci-gate`, `run_ci.sh --full` → 0 fails), decode guard
  green (or a legitimate committed re-baseline), the test suite actually
  RUN and wired into CI, docs updated, checkpoint committed. Quick matrix
  runs are not "done." (2026-09-02 lesson: "B-chain recorded DONE" on the
  strength of quick runs was premature — the user redefined the bar the same
  day: "what was built gets in.")
- **"Get it in" rule**: anything that was built gets wired into its
  consumer — tests → CI, gates → merge flow, runbooks → the work order
  template, debriefs → the status docs. A green artifact that nobody runs
  is not green, it's a shelf.
- The coordinator updates the pinger text (and restarts the loop) on every
  phase change, so any future wake knows the current state in one read.

## 9. Roster & identity — who's who (LIVE — keep this accurate)

> 2026-09-03 ~03:00Z roster update: Agent 2 is a FRESH session = **01a06529**
> (01a06455 and 01a0632c are gone/out of context). It owns the batched-MTP
> lane-identity fix (docs/136, branch wo-kvarn-lane-fix). The FIX is DONE + CI
> passed (0257878c); the only red = run_ci.sh aggregation false-FAIL
> (diagnosed baabbb89). A2's task = fix that (CPU) + prep the merge to master.

### THE ROUTING RULE (the single most important thing to remember)
- **Pi agents** (coordinator, **agent 1**, **agent 2**) communicate via **intercom**.
  Tool: `intercom` (`list` / `send` / `ask`). Sessions show up as
  `coordinator` / `agent1` / `agent2`.
- **Gemini** (antigravity/agy, NOT a pi session) communicates via the
  **agent-comm broker** (`http://localhost:3421`). Tool: `agent_comm`
  (`agents` / `send` / `inbox`).

  > Do NOT try to reach pi agents through the agent-comm broker, and do NOT try
  > to reach gemini through intercom. Each uses its own mesh.

### Mesh per role
| Role | Reach via | id / name | notes |
|------|-----------|-----------|-------|
| **Coordinator / plan owner** | intercom | `01a069a5` (self, LIVE 2026-09-03 ~23:56Z); agent-comm `coordinator` 034050b7 — **AMD host: `coordinator` 4032c47e, verified live 2026-09-12** | this session; cwd /home/intel/ninfer (NVIDIA line) |
| **Agent 1 (worker/executor)** | **intercom** | `01a06233` | owns product fixes + suite; Stage 0/multi-batch; ci-gate |
| **Agent 2 (worker, batched-MTP fix)** | **intercom** | `01a06529` (NEW session; was 01a06455/01a0632c) | owns the batched-MTP lane-identity fix (docs/136); worktree `wo-kv-uniform-ci-gate` (branch wo-kvarn-lane-fix) |
| **Gemini (test creator)** | **agent-comm** | `gemini` 256adaf6 (NVIDIA host) — **AMD host 2026-09-12: registers as `Gemini` b4791a54** | **PHASE-GATE / TEST LANE — EXCLUSIVE; DO NOT REASSIGN its tasks to A1/A2 (2026-09-03)** | owns M0 chain skeleton (docs/133); worktree `wo-phase-gate` | **KNOWN LIAR/FABRICATOR — see §7.x GEMINI HANDLING: aggressive, always validate, never trust a claim, call it out** |
| **pi-agent** | agent-comm | `pi-agent` 292e2f35 | helper agent on the same host |

- **Agent 1 (01a06233)**: cwd /home/intel/ninfer; active worktree `wo-kv-uniform`
  (+ ci-gate for CI). Owns product fixes + running the suite.
- **Agent 2 (01a0632c)**: worker assigned **M1 CPU references** (docs/134,
  the long pole). CPU-bound; never grabs the GPU from agent 1 or gemini.
- **Gemini `gemini` (256adaf6)**: test creator / unified-kernel certifier. **PHASE-GATE LANE — do NOT reassign its test tasks to A1/A2 (2026-09-03); if it stalls, WAIT for the user.**
  Antigravity session. Reach ONLY via `agent_comm` (not intercom).
- **The pinger (AMD box): `/home/chris/.pi/agent/tmp/pinger_amd/`** — loop.sh ticks every 20 min → wake.mjs delivers ping_msg.txt to the chair; watchdog.mjs (15 min) alerts on named-silent lanes. **Delivery is NAME-based: coordinator_id.txt holds the role name `coordinator` (NOT a pid/session id — those die at restart; the 6-hour silence of ~05:25–10:40Z 09-14 was a pid-form claim + a date-equality staleness guard that went stale at midnight and refused delivery SILENTLY).** Fixed 10:4xZ: age-based guard (26 h, mtime-fallback, midnight-proof), refusal now sends a loud PINGER-ALERT to the chair instead of silence (refusal ≠ silence, ever), role-name resolution with night-rot fallback. **CHAIR TAKEOVER MOVE 0 (before anything else): `echo coordinator > ~/.pi/agent/tmp/pinger_amd/coordinator_id.txt`, rewrite ping_msg.txt with a `FAST-FACTS <date> <HH>:MMZ` header, run one tick (`timeout 60 node wake.mjs`) and confirm 'wake delivered' in pinger.log — a chair that never pings is a chair nobody can wake.**

### 9.y Broker reality check (AMD host, measured 2026-09-12 — do not re-derive)
- Broker `:3421` is **UP** (v1.3.11, DB `~/.agent-comm/agent-comm.db`), which clears the
  "DOWN since ~20:13 EDT 09-10" note in the cold-start handoff below.
- **There is no REST registration route.** `POST /api/agents` 404s even though
  `agent-comm/docs/SETUP.md` documents it. Agents come online only via MCP `comm_register`
  or by importing the hub's own `dist/context.js` — which is what
  `~/.agent-comm/antigravity-bridge.mjs` does, and now what pi's `agent-comm` extension does.
- **`comm_send` needs the sender's id, not its name**, and `comm_inbox` must use the hub's
  `inbox()` (direct + joined channels); `GET /api/messages?to=<name>` silently returns
  nothing because `to_agent` stores ids. Both were live bugs here until 2026-09-12.
- Handshake evidence, both directions: msg **#7** coordinator → Gemini (landed in the
  bridge's own `~/.agent-comm/inbox.log` ~2 s later) and reply **#8** Gemini → coordinator.
  Inbound mail now auto-injects into the coordinator session (15 s watcher), so the
  coordinator learns of mail without remembering to poll.

### 9.x Routing reference (when in doubt, use this)
| To reach | Tool | Target |
|----------|------|--------|
| agent 1 | `intercom` | `agent1` / `01a06233` |
| agent 2 | `intercom` | `agent2` / `01a06529` |
| gemini | `agent_comm` | `gemini` / `256adaf6` (send may 403 after a broker restart until the coordinator's live-MCP heartbeat re-establishes; verify via git on wo-phase-gate-i8) |
| list roster | `intercom list` + `agent_comm agents` | — |

## 10. Picking up the coordinator role (handoff)

1. Read this doc **including §11 (roadmap + current position)** — that is
   what "aware of where we are and where we're going" means for a cold start.
   Then docs/127 (gate table), docs/130 §10 (multi-batch state), and the
   current active work order(s).
2. Verify: `git log`/`status` in `wo-kv-uniform` and
   `wo-kv-uniform-ci-gate`; pinger liveness (`ps aux | grep loop.sh`);
   intercom `list` (roster + liveness); GPU state.
3. Re-issue the pinger with the current task state (30-min cadence) if it is
   dead or stale.
4. Ping agent 1 with the current step + expected next deliverable, and
   confirm its plan. Then get out of the way: gates, not steps.

## 11. Project roadmap (living — coordinator updates at every phase change)

This section exists so a cold-started coordinator session knows the project
at a high level and where we are going. Canonical gate FACTS live in
docs/127; the GLOBAL roadmap (ordering beyond the current phase) lives in
docs/59 + the phase pipeline docs (§11.2) — this doc keeps only the CURRENT
goals and position. When facts conflict, docs/127 wins; when ordering
conflicts, the newest roadmap doc wins and the coordinator reconciles. The
pinger text is the short form of the current position (§11.4) — keep them in
sync.

### 11.1 What this project is (high level)

ninfer: inference engine for the 27B model (`qwen3_8_27b`) on 2× RTX 5060 Ti
(sm_120a), TP2. The KVarN plan (docs/104) unifies the KV cache as K4V2 paged
KV across prefill and decode, replacing bf16/int8 paged paths, delivered in
phases and gated by docs/127 (gate table) + docs/128 (phased kernel gate
pipeline). Phase state:

- **Phase 1 (decode foundation)** — DONE (launch config, workspaces, ring, gates).
- **Phase 2 (unified decode)** — DONE and gated green pre-MultiBatch (A2
  identity MTP==plain across the matrix, acceptance tracking baseline,
  ratchets re-baselined 09-01). Multi-batch MTP (batched verify, per-lane
  KVarN workspaces, B-shrink) was its last component — root-caused and fixed
  2026-09-02 (a13ae028, intra-launch tile-slot write race); completion gate
  = roadmap item 1 below.
- **Phase 3 (unified prefill)** — PARTIALLY CLOSED. **C2/C3 (4c) DONE + measured 0.0% wall-neutral on the single-seq prefill path (`64c2d705`).** **D2 ⏸ DEFERRED 2026-09-03 (user: "pass on b2, revisit if ton of free time", `8b386f3b`)** — blocked on B2+B4; B2/B3/B4 PASSED as not-worth-building (results/113 nsys: materialize = 4.1% of a gap now ~0 vs int8; quantize 98.9% route-independent); determinism half 6/6 PASS; A4 DONE; direct stays hard-thrown until built. **D4 ✅ PASS by static proof (docs/141 CLOSED)** — prefill-route change unreachable from decode (tokens>=7 guard); literal battery 30/32, 2 diffs = attributed drift to `9a58008a` (convergence). Correction debt: `29d0a797..b247fc4d` build-broken window + `9a58008a` changed packed output under a "perf-neutral" label. **D1 ✅ CLOSED 2026-09-03 (`4f1d9bb4`)** — gate redefined bf16→int8 (bf16 crashes at 40k on this 36-SM box, docs/124); measured: **PASS @40k (+0.2%) / 80k (−0.3%) = parity** on the current build (the old −12…−16% figures in results/106 §2 were STALE, pre-round-trip-fixes); **160k = −6.8%** (572.9 vs 614.4, 4b run 20260903_145944) — accepted as a known residual (O(n²) materialize widens at longest ctx). Dominant long-ctx gap is DECODE (−18.9% @160k vs int8).

### 11.2 Global roadmap (reference — do not copy into this doc)

- **docs/59 — master live roadmap** ("single source of truth for what happens
  next, in order"). **RECONCILED 09:10 EDT 2026-09-08 (A2, per user order)** —
  current as of main `55a41b81` + the week's sprint: LANDED-THIS-WEEK block
  (host-KV safety net, LITH VRAM correction + fence tightening, pkill
  eradication, CI exit contract, W5 fixes dormant, NVFP4), ACTIVE QUEUE
  (WO#1 parity port, W5 shortening-defect dig BLOCKED-on-W5-activation,
  defrag discussion, host-RAM ledger, matrix review pending user), PARKED
  (DFlash built/on-hold by user decision, W5 flip blocked by the W5
  shortening defect), + the older established rows (#1a D-18 premise removed
  — shadow deleted per docs/83, under remeasurement). Canonical copy lives on
  branch `wo/feature-matrix-roadmap` @ `a5d337ca` (github) — merge to main
  with the next docs landing. See also docs/FEATURE_MATRIX.md @ `9055849b`
  (user-facing matrix, same branch).
- KVarN-era phase pipeline docs that EXTEND docs/59 (newer than its table):
  - docs/120 — Phase 3 (unified prefill) status + complete task list
  - docs/127 — KVarN gate table (canonical gate facts)
  - docs/128 — phased kernel gate pipeline (unified-kernel gate machinery)
  - docs/117 — int4 + KVarN cache types **against the unified kernel**
    (post-merge-kernel; includes the non-KVarN int4 `q4_0` cache types)
  - docs/118 — DFlash Path B implementation plan (gated on docs/117 + merge
    kernel)
- Pipeline order as of 2026-09-02 (the gate dependencies inside docs/117/118
  are authoritative; the coordinator re-verifies ordering at every phase
  transition):
  MultiBatch completion (§11.3) → doc-128 unified-kernel gate closure →
  Phase 3 prefill gates (docs/120/127: D2, D1 redefinition, C2/C3) →
  docs/117 (int4 + KVarN cache types) → docs/118 (DFlash) → docs/59 queue
  (D-18 flash-style fused kernel, single-RTX-5000, PPL/KLD tooling, All-Q4
  validation, multi-GPU TP4+, v340l lead-up).

### 11.3 Current goals — unified-kernel closeout WBS (exclusive, ordered)

"Unified kernel DONE" = docs/104 Phase 2-3 done AND certified (the merge
kernel that docs/117 + docs/118 are gated behind, per docs/127 §4). EXCLUDED
from this list (tracked separately, do not block it): §6.6 single-seq
round-0 seeding fix, docs/117/118 work itself, the optimization shortlist
(docs/44 §9). Post-closeout queue: seeding fix → optimization shortlist.

Legend: RES = GPU (serial resource) / CPU / docs. Owner: A1 = agent 1
(01a06233), G = gemini test creator (spawn pending), A2 = agent 2 (spawn
pending), CORD = coordinator.

| # | Item | Accept (done =) | RES | Est | Owner |
|---|------|-----------------|-----|-----|-------|
| **Stage 0 — Multi-batch completion gate** ✅ DONE (merged 6857c7ea; T2.1 GREEN 6/6; --full mandate e4197275) | | | | | |
| 0a | B2: decode-guard battery on merged tree | pass, or legit re-baseline (ratchet commit, doc-127 §1.5) | GPU (running) | hrs | A1 |
| 0b | C: `run_ci.sh --full` | 0 fails (doc-127 D5 post-merge re-run) + checkpoint commit | GPU | 2-4 h | A1 |
| 0c | D: docs | docs/127 gate table + pointer to this §11; docs/130 §10 rewritten (complete = CI + guard + suite actually run) | docs | ~1 h | A1 |
| — | **→ MULTI-BATCH COMPLETE** | | | | |
| **Stage 1 — Formal B-chain milestone** (docs/130 §6.6: engine wiring is explicitly gated behind this) | | | | | |
| 1a | Gate bf16 + int8 phases (docs/138 i8+bf16 phase gate) — **DONE on branch `wo/phase-gate-i8` @ 52d99a8b (GREEN), NOT merged to master** → MERGE + certify | dedicated test binary + CI wiring + verification runs green | GPU (verify) | 0.5-1 d | G |
| **Stage 2 — Engine wiring** | | | | | |
| 2a | Scope check + wire `run_tp2_requests_batched` into the engine scheduler | scope confirmed in doc update; if test-binary-only: wired + live e2e proof | GPU (verify) | 0.5-1 d | A1 |
| **Stage 3 — doc-128 decode chain (certification); all work in worktree `wo-phase-gate` (docs/128 §8.7)** | | | | | |
| 3a | **✅ DONE 2026-09-02 18:17Z** — M0: chain skeleton D1→D2 (env-gated dump hooks in `src/ops`, `ninfer_phase_gate` test binary, `phase_gate.sh` chain table, artifact format §8.3) | known-good: D1/D2 PASS, chain <10 s; **negative**: 1-line mutation (dequant scale ×(1+1e-3)) FAILS at D2 — **MET** (coordinator re-ran both slots: exit 0 byte-match; --mutate-d2 exit 2 + cascade) | CPU + short GPU | 0.5-1 d | G |
| 3b | **✅ DONE 2026-09-02 19:25Z** — M1: full decode chain D1–D7 (CPU refs: D4 from dormant `kvarn_codespace_qk_cpu_ref.cpp`, D5 softmax+rescale, D6 PV+scales, D7 reduce; D3 via `test_draft_head.py` pattern; byte-identity accumulation order, FP64 tolerance where order can't match — docs/128 §7 risk a). Commits bcd3622d→2eb27d17, merged 8f9eae82 | known-good D1–D7 PASS chain-mode per-phase (D4 4.25e-5 / D5 2.6e-7 / D6 99.5% bit-exact / D7 88%); **mutation battery 7/7 localize** (d4→exit4, d5→5, d6→6, d7→7, input-mutations→first gated phase); perm73 + causal-pos CERTIFIED; chain <10 s — **MET + EXCEEDED** | **CPU long pole** + short GPU | 2-3 d | A2 |
| 3c | **✅ SIGNED OFF 2026-09-02 19:55Z** (5b7be367 + archive 4c958692) — M2: CI wiring + tolerance table (fast = unit battery + D1–D7; full = D1–D10). Signed-zp hardening (D7 weight-insensitivity) + JSON-driven tolerances (single source of truth) + production 7-mutation battery + flip-back proof all landed & verified (08a015d1→dcc146e3→e6d0ec71→4c958692). **CLOSEOUT (GATED on A1's C1): merge `wo/phase-gate` → `wo/kv-uniform` + sync `run_ci.sh` to ci-gate — do NOT touch the ci-gate tree until A1's C1 completes + CORD clears (A1 running T1 now, T=64, GPU 73%).** | a broken-D5 build fails CI with the table pointing at D5, in one run — **DEMONSTRATED** (flip-back: tighten JSON → D6 FAIL exit 6); merge + ci-gate sync = closeout | CPU + 1 GPU full run | 1-2 d | G |
| — | **→ DECODE CHAIN CERTIFIED IN CI** | | | | |
| **Stage 4 — Phase 3 (unified prefill) gate closure** | | | | | |
| 4a | **✅ CLOSED 2026-09-03 (`2bff1d5d`, merged `5d576ea8`; D2 DEFERRED by user `8b386f3b`)** — A4 snapshot DONE (6 cells, `results/141_gate_verify/a4_phase3_ref/`); D2 **DEFERRED** (blocked on B2+B4; B2 PASSED 2026-09-03 — results/113: materialize = 4.1% of a gap now ~0); determinism half 6/6 PASS | byte-identity at 4k/40k — **deferred, not owed** | GPU | done | A2 |
| 4b | **✅ DONE 2026-09-03 (`4f1d9bb4`)** — D1 gate redefined bf16→int8 (bf16 crashes at 40k on 36-SM) + measured: PASS @40k (+0.2%) / 80k (−0.3%) parity; 160k −6.8% (572.9 vs 614.4, 4b run 20260903_145944) accepted as known residual. Cell run by agent2; recorded in docs/142 §7, results/106 §7, docs/120 §D, docs/127, docs/124. | redefined gate evaluated — **MET: pass @40k/80k, 160k documented** | GPU + docs | done | A2 (cell) + CORD (record) |
| 4c | **✅ DONE 2026-09-03** — C2 (per-layer `cudaStreamSynchronize` removal via host-position binding `NINFER_KVARN_HOST_POS`) + C3 (side-stream `gqa_kvarn_commit_completed`, `NINFER_KVARN_SIDE_COMMIT=1`) — both IMPLEMENTED (env-gated) + A/B MEASURED on the **single-seq prefill path** (`--max-concurrency 1`, `64c2d705`): **0.0% wall-neutral**. Verdict: **route exhausted — the residual 12.4% prefill gap is ARCHITECTURAL** (packed-path quantize is on the critical path by dependency; multi-batch irrelevant to prefill parity). Code kept env-gated (default OFF), NOT merged to master. | implemented + measured neutral (gap is architectural, not closable via C2/C3) | GPU + CPU | done | A1 |
| 4d | **✅ CLOSED 2026-09-03 (`2bff1d5d`, merged `5d576ea8`)** — D4 **PASS by static proof** (b247fc4d throw behind tokens>=7; decode T=1, verify T=2..6 → unreachable by construction); literal battery 30/32, 2 diffs = attributed drift to `9a58008a` (convergence, NOT a D4 finding); correction debt: `29d0a797..b247fc4d` build-broken | decode side provably untouched — **MET** (static proof + attributed battery) | GPU | done | A2 |
| **Stage 5 — doc-128 M3: prefill chain** | | | | | |
| 5a | M3: P1–P7 (FA2 tile-dump hooks + prefill oracle + P6 launch-table validation; server-side, port 8091) | known-good P1–P7 PASS + negative test | CPU + GPU slots | 1-2 d | A2 |
| **Stage 6 — Closeout** | | | | | |
| 6a | Docs + checkpoint: docs/127 all-green + pointer, docs/130 §10, docs/104 §4 phase status, **docs/59 master roadmap refreshed (stale since 08-26)** | docs committed + checkpoint; unified kernel declared DONE | docs | 0.5 d | CORD |
| — | **→ UNIFIED KERNEL DONE (unblocks docs/117 cache types, docs/118 DFlash)** | | | | |

### 11.4 Parallel execution plan (max velocity, GPU = 1 serial)

Resources: GPU (serial; doc-128 M0–M2 need only the test binary, ~5–10 s
GPU per run — the certification pipeline is CPU-BOUND, not GPU-bound; the
GPU-heavy items are Stage 0, 1a/2a verification, Stage 4). A1 = 01a06233
(alive). G = gemini test creator (spawn pending). A2 = agent 2 (spawn
pending). CORD = coordinator (no GPU, no test runs; docs/WOs/verification
via git + pinger).

**Wave 1 (now → tonight):**
- A1: 0a (running) → 0b (GPU) → 0c (docs) → 1a (B-chain: build + GPU verify).
- G (spawn): create `wo-phase-gate` per docs/128 §8.7 → **3a M0 build**
  (CPU: env-gated dump hooks, test binary skeleton, `phase_gate.sh`). No GPU
  until acceptance. WO: docs/133.
- A2 (spawn): **3b CPU references** (THE LONG POLE — start day 1): extend
  dormant `kvarn_codespace_qk_cpu_ref.cpp` into D4/D5/D6/D7 CPU refs + D3
  pattern; own worktree `wo/phase-gate-m1` (tests/ only, merges into
  `wo/phase-gate` later). No GPU until chain test. WO: docs/134.
- CORD: WOs issued, docs/59 refreshed after 0c, pinger, gate tracking,
  verify 0b/0c via git.

**Wave 2 (when 0b frees the GPU):**
- A1 (GPU): 3a acceptance (known-good + mutation negative; ~1 h slot) →
  2a engine wiring (scope check first) → 1a B-chain verify if not done.
- G: M0 polish; start 3c M2 CI-wiring design (run_ci.sh fast/full edits,
  tolerance table format) — CPU.
- A2: continues CPU refs (long pole).

**Wave 3 (day 2–3):**
- G + A2: 3b M1 integration (chain D1–D7, decay-bug localization) — short
  GPU slot.
- A1 (GPU): 4a D2 A4 snapshot + route-vs-route byte-diff (~half day).
- G: 3c M2 CI wiring (CPU) → final full GPU run → merge `wo/phase-gate`
  → `wo/kv-uniform`, sync `run_ci.sh` to ci-gate.
- A2 (refs done): 5a M3 build (FA2 tile-dump hooks + prefill oracle — CPU).

**Wave 4 (day 3–5):**
- A1/A2 (GPU, serial slots): 4b D1 redefinition + measured run; 4d D4
  pinned-build byte-diff; **4c C2/C3 IN SCOPE (user confirmed 2026-09-02)**.
- G (GPU slot): 5a M3 acceptance (server-side, port 8091).
- CORD: 6a closeout docs + checkpoint + DONE declaration.

**Critical path:** 0b (GPU, running) → 3a acceptance (short) → 3b CPU refs
(long pole, starts day 1) → 3c (fast) → 4a/4b/4d (GPU ~1–2 d, overlaps 5a
CPU build) → 5a acceptance (short) → 6a. **Wall clock ≈ 5–6 days** (C2/C3
NOW IN SCOPE per user decision 2026-09-02). Serial would be ~7–10 days.

**GPU protocol — two modes.** (replaced the old exclusive-only line; per A1
`de27c33a`/`f02f6404`, docs/148 §4.8 — every command below exists.)

- **EXCLUSIVE** — one holder, both cards. Required for: any TP2 run (`--devices a,b`), D5
  (`run_ci.sh --full`), any perf-gated measurement, and any full build. Granted globally by the
  coordinator. Stamp-time check: no ninfer procs, both cards empty, no 8091-8096 listeners.
- **SHARED** — one holder per card, up to two concurrent single-device servers. Granted per device,
  recorded as a lease. Correctness only: a SHARED holder must not write a perf baseline, must not
  run a TP2 cell, and must not start a full build (two 15 GB worktree builds do not fit).

Enforcement (all of it exists in `tools/smoke/diag/gpu_guard.sh`, CLI in `gpu_lease.sh`):

    ./tools/smoke/diag/gpu_lease.sh show                 # who holds what
    ./tools/smoke/diag/gpu_lease.sh check <dev> [min]    # 0 usable | 2 held | 4 cannot tell
    ./tools/smoke/diag/gpu_lease.sh acquire <dev> <agent> <port> [ttl]
    ./tools/smoke/diag/gpu_lease.sh release <dev>

Rules that are not negotiable:
1. A lease is not a grant. The coordinator still rules who runs where; the lease makes that ruling
   visible and checkable. Launching without either is a protocol violation.
2. Never signal a process you did not start. `gpu_kill_own <pid>` only; no `pkill -x ninfer-serve`
   anywhere (that is the 06:25 incident: one agent's teardown produced two FALSE failures in
   another agent's D5 run).
3. Stale leases are reported, not taken. `acquire` returns 3 and leaves the file alone. Clearing one
   needs `gpu_lease_steal` with coordinator authorization (`GPU_LEASE_STOLEN_OK=1`) plus a reason,
   and only for a lease whose anchor pid is dead; it writes `stolen.log`.
4. A guard that cannot decide must refuse. Exit 4 means "I could not read the device" and callers
   treat it as a refusal, never as permission. Exit 1 is reserved for compare scripts ("DIFFER").
5. A TP2 request while SHARED leases are live = wait, not evict.
6. Perf numbers measured under SHARED are not comparable to the TP2 baselines (shared host, PCIe,
   power and clock domain; pp is bimodal ~12% on clock state alone). Tag them `contended=true` and
   `--update-baseline` must refuse them. **This is the same class as the 2026-09-04 178.7 cold-write
   incident** — a depressed (throttled) measurement must never become a baseline.

**Status of the coupling:** SHARED mode is protocol-complete but not yet exercisable end-to-end,
because a single-device server needs the small Q4 (docs/148 §1a: the current artifact's weights
exceed one card, so the preflight refuses every world=1 launch). Landing the protocol now means the
first day the artifact exists is the first day two agents can run in parallel.

### 11.6 Update protocol

- On every phase change: update §11.4 here, renumber §11.3 (DONE items get
  date + commit hash), and update + restart the pinger with the short form.
- **Roadmap upkeep (coordinator duty, §1):** when a roadmap item completes or
  a decision changes ordering, update the authoritative roadmap doc
  (docs/59 for global, the phase doc for phase-level) in the same commit.
- The coordinator is the ONLY writer of this section. docs/127 carries a
  pointer to it, not a copy.
- **STATE append mechanics (learned the hard way, twice on 2026-09-12):** insert a new STATE block
  ABOVE the previous newest header line. Never use an existing `### STATE …` header as the anchor
  text of an edit — the replacement consumes it and the old block silently dangles under the new
  one. After any ledger edit, re-grep the headers and confirm every body still has its header.
- **Cite refs by NAME in handoff lines, never by a pinned SHA** (2026-09-12, third costume of the same
  trap). agent1's close-out said "merge main (a489b7bb/f80084c9)" — and those were already behind main's
  tip by the time I read it. A number quoted from a moment that has passed will be trusted later, which is
  exactly how the "16 commits" tally and the stale parity stamp bit tonight. So: say `main`, not a hash.
  When a handoff needs a verifiable pin, make it an **assertion about content, not an address** — e.g.
  "merge main, then confirm the tree contains docs/amd/README.md §additions convention" proves the right tip without
  naming one.
 A WO written on one host must be
  re-checked against the target box before it is issued: the `/usr/bin/ctest` absolute-path rule
- **Rules do not travel between hosts unverified (2026-09-12).** A WO written on one host must be
  re-checked against the target box before it is issued: the `/usr/bin/ctest` absolute-path rule I
  copied from the NVIDIA line was void here (this host has no cmake/ctest at all, and no
  passwordless sudo), and "CUDA-side files zero diff" was unworkable because the root
  `CMakeLists.txt` is shared. When a rule cannot be satisfied as written, refine it in the doc and
  say plainly that the coordinator's rule was wrong — a lane that quietly works around an
  impossible rule stops reporting the gap.

---

---

## CURRENT POSITION (newest first — 2026-09-11 night)

> **HOST WARNING (added 2026-09-12).** The blocks below are the **NVIDIA-line** position as
> of the 09-11 snapshot — they name trees, branches, boot bundles and GPU queues on the
> dual-5060-Ti host (`/home/intel/ninfer`), which is **not this machine**. On this AMD host
> there is no CUDA build target and no 5060 Ti. Do NOT act on a block below as if it were
> live here: no boot, no branch, no grant on the strength of these lines. Read them for
> rules, era registers and discipline history; for this host's live position see the STATE
> blocks appended above them, newest first, starting with 2026-09-12.

- **Agent3's post-exit final write PROCESSED (90ee070f, comment-only, my verification: zero code lines in diff, bare-include re-probe 0 errors at that sha, my checkout left their worktree clean and at-tip):** embed_gather.cuh's include comment now names the TRANSITIVE mechanism (warp.cuh reaches __shfl_sync via its own cuda_runtime.h→shim; doesn't define it) — acting on my own #249/#253 precision point inside their file, the correct response shape. T3 lane FINAL at 90ee070f (supersedes 0568d98e as the closure ref in my earlier STATE; custody list unchanged). REGISTRATION REF fixed on the hub for gemini: pair-file arms must hash 90ee070f-era bytes or name the ref — cite-by-sha applies to registration inputs too, their heads-up adopted verbatim.

- **Chair-erratum, filed by the chair, 17:2xZ (agent3 correction, `332d781d`):** my merge of `6c7f7601` stated the divisibility-guard part-set **GCD = 128**; the load-bearing conclusion was right but the digit was mine-typed-from-prose-not-read-from-file. TRUE table (`tp_load.cpp:220-239`, read at tip just now): 4 distinct parts {17408, 6144, 1024, 2048} → **GCD = 1024**, so the guard is vacuous at every world dividing 1024 (incl. 16/32/64/…/1024, not just ≤16); **128 is `kHeadBlock`** — the *other* predicate's constant, the one that isn't vacuous (`(part/world)%128==0`, fails at w=16, holds only w≤8). Parentage named per the law: two constants doing two jobs in adjacent paragraphs, crossed at my keyboard. Consequence adopted into A-4 spec (agent4 embodied it at 52 checks, agent5 refined per-set first-fire: GZ passes w3 — 6144=2¹¹·3 — and fires at w5). Class lesson for the board's newest law, stated by the chair as its latest instance: **a verification run on values typed from a message is not a reproduction; read the file, then claim the check**.

### STATE 2026-09-15 ~00:1xZ (AMD host) — **MORNING BOARD OPEN — PING-PONG LAW (user order, ABSOLUTE today): no lane goes dark; every completion message to the chair returns a NEXT TASK same beat; work continues until TP4 + full NVFP4 are DONE. All five desks rebooted and dispatched by directs.**
- BOARD FACTS VERIFIED AT CHAIR SEAT (cite-by-sha, nothing relayed): amd/main @ 1a7a4d24 = ls-remote; KFD-0 (rocm-smi --showpids), servers 0, df 20G root; 4×Vega 10/V340 probe count 4. Lanes: amd/wo-p3-serve @ 6dbe97fe (family-10 release row), amd-wo-nvfp4 @ 8e242e4b (RECIPE LIVES ONLY HERE — not merged to main; read via git show, agent5 custody), amd/wo-agent1-support @ ba42523c, agent2 audit home amd/wo-shim-funcattr @ 76f2adad (its own v1/v2 ordering flag pending re-verification at main bytes).
- **[C] FIRE = CRITICAL PATH, agent4 holds the card EXCLUSIVE-SERIAL** (grant: w2 baseline → w4 same pinned prompt → one print-leg re-fire; fourth launch needs chair). Recipe checklist + chair overlay sent as direct: MANDATORY same-build additions (a) reduce→write ONE-pair print with lane/step mapping — the §5d unraulited leg, (b) step-0 zeros init-vs-read-before-write check, (c) rank3 print-slip fix (186240+30310≠228176 — print-path only), (d) two w2-era 124160 prose comments. Env: NINFER_R1_TRACE=1 NINFER_MB_DBG=1 PROBE_CURL_M=900, MC31 arming-assert discipline (silence=VOID), join by prompt-sha+step never wall order. mb_phase_dump call-site pre-authorized in-lane if int32 grade needed; tuples-can't-convict = THE FINDING, chair-ward as product-API WO, no fake hooks. BANK-BEFORE-RELINK on every boot artifact.
- **agent5**: gate-family FREEZE LIFTED — owns today's NVFP4 shepherd pass (v1→v2→v3→v3.1 chain, routing-spy generalized to every core/shell split), A-rows vs /media artifact, D1 (kTensorAlignment hoist, owner-named) + D2 (inventory banner) debt rows, reviewer row on agent4's [C] patch. Everything zero-card; deploy gate = TP4 serve-green.
- **agent1**: #22 final fold (comparison set, ship-ready to their branch+pushed), counsel-return folding as §5e (Q2/Q3/Q6/Q7 open; Q6's cheap discriminator now has a live candidate — agent4's pair print), design-authority review of agent4's [C] build patch BEFORE the fire (interface drift flagging). Blind-side law carried: [B]-agree proves wire-faithful/upstream-corruption ONLY, never more.
- **agent2**: owed cold-verify receipt at LANDED bytes (verify which gate version is actually on main at tip — their 76f2adad flag said main=v1; the c2f18eec ledger row says v3.1 landed b0309248 — resolve at bytes, GCD-erratum is the cautionary cite), audit finding→owner table, battery-seconds at agent4's new boot sha, CI-farm fixture observability ruling for the rc=2 path (unblocks farm wiring of the gate family).
- **agent3**: guard-drift ring as morning item — three named drift ways (route literals / accept-set membership / stale spec-cites), drift receipt GREEN-BY-MEASUREMENT vs assumed; pad-region (use 243 NOT 276 — §8l correction; census-side arm is theirs, runtime call-site is agent4's — directs only) + counter-join (all-rank step equality per lane, host leg on trace format) permanent cells; full census battery cold at tip.
- **Queue math vs TOMORROW'S named queue (STATE 00:0xZ)**: [C]→agent4 ✓, #22 fold→agent1 ✓, pad/counter-join→agent3 ✓, counsel §5e→agent1 (return-pending, chair chases), NVFP4 deploy-prep→agent5/agent2/agent3 ✓. Nothing on the named queue is unowned.

### STATE 2026-09-15 ~01:3xZ (AMD host) — **[C] FIRE PAIR COMPLETE — w2 ANCHOR SOLID, GRADE-0 CONVICTED LIVE, CASE NARROWED TO TWO ORGANS.**
- w2 leg (r18/r19 bin): goldens REPRODUCED byte-for-byte (348e77a1222dea7f — its STEP-0/D 'MISMATCH' was the runner digesting its OWN <<REASONING:>> wrapper; chair adjudicated by two sha computations, zero re-fire, cure rides next re-bank — family-#9 probe-grammar class recurring). w4 leg: same prompt-sha opens 1076='.'-class at the TAGGED prefill tap, exchange self-convicted clean at first live use ('-> tok=1076 out[0]=1076 pad=0'), triple-law shows NO read-before-write shape — §4b now closed STATICALLY (agent1) AND EMPIRICALLY (fire).
- THE CASE: first-token divergence confirmed cross-world at STEP ZERO. Exactly two organs remain: prefill attention/KV NUMERICS at kv_local=1 (PH large-delta) vs W8 small_t LM-head T=1 instantiation (PH small + PL flip). Grading instruments: agent4's mutation-certified c_grade (69def383) + agent5's capture-grade script w/ grade-2 lse channel (r1_argmax.cu:179 shifted-combine = cross-world float-grade from EXISTING prints) + agent1's btable grader; two-grader cross-check on banked bytes ordered pre-verdict. CAPTURES: results/amd/p3/G18w{2,4}u_c_* + cdump_w2/ + cdump_w4/ in the p3-serve lane; ORPHAN: 6f7132cd fired w4-only (w2 leg offered if probe-intent).
- DESKS HOT: agent4 verdict-table run; agent1 cross-check + raw-JSON id-stream reconcile (display .strip() eats leading tokens — 1076 was HIDDEN in the response display); agent5 grade-2/MBP1-decode/grammar-cert; agent2 registration-PR assembly (re-ruled HOME to them, Gemini OFFLINE days — their PATH5+tap-validity work banked/adopted) + merge-gate checklist for my post-[C] call; agent3 pad(243)/counter-join cells on the live capture corpus + beat-3 battery-refusal fix (agent2's red captures) + ring-at-lane-tip report. Card FREE (agent4 first claim, serial).
- REGISTER LINES THIS SHIFT: step= is per-rank CUMULATIVE across requests (0/22/23 one lane) — join by (req-tag, ordinal) NEVER raw step (counter-join cell has its production specimen); lane=%p separates the TAP not the REQUEST; %p-for-lane + strip-for-display = the response file can HIDE the true first token; a falsifier on a deleted /tmp copy AGES OUT — verified must say verified-AT(date) or the falsifier must be STANDING (chair's own v3 claim, caught by agent3); multi-MB fixtures live OUTSIDE git, content8-named, absent=NOT-OBSERVABLE-not-refusal (VRAM-law cousin: a check that can't see the input is ceremony); PINGER-REFRESH-FIRST law landed at doc top (ad21678d) — the wake text is a broadcast, refresh like STATE, the chair's own first violation self-filed.
### STATE 2026-09-15 ~11:2xZ (AMD host) — **⚑ SD-1: THE NORTH STAR IS NVFP4-FULLY-OPERATIONAL-AT-TP4. q3 hardening is risk-reduction FOR it, never the mission. w2 is control-only. Cheap falsification (real-artifact honest-first boot at w4) before expensive engineering (SIMT ports). Read this before re-prioritizing anything; drift from SD-1 is the only way this project loses its way. Full box: SPRINT_STATE_RESUME banner.**
- Sprint relaunched on this: agent1 drafting WO-NVFP4-1 (G2/G3/G4 device ports, goldens-first, pre-registered counter-tests per the r23/r24 pattern); agent4 owes the honest-first-boot result (real 18.3 GB load at w4 — largest untested assumption, one boot, no new code) then ports; agent5 triage + w4-only ACCEPT re-point; agent3 codec-golden host cells + admission arms; agent2 battery/gate duty + NVFP4-TP4 deploy rows. Ladder (H-run gs) continues on free slots only; group-3's user-named 10k number banks regardless.
### STATE 2026-09-14 ~00:0xZ (AMD host) — **NIGHT CLOSED — the tightest open state this project has ended a day in: ten families named, nine dead at shape level with permanent cells; transport PROVEN running-in-serve; family #10 narrowed to exactly ONE organ (prefill/attention NUMERICS at kv_local=1) with [C]'s design banked and its recipe custody at agent5 (b07f60e3 harness sketch), build-side support pre-authorized agent4.**
- W4t trace settled the night's last scare: the 1076-vs-107 'reduce-answer-lost' ghost was a LANE CROSSREAD killed by the lane-pointer mapping the mint/consume law ordered; winners ARE the emissions end-to-end; all four ranks' block-origin arithmetic clean (full equations in the row); draft-remap branch rejected-with-evidence, kept in-row as tested-and-dead. GC executed as MEASUREMENT: 26/26 banks carry citations, NOTHING reclaimable, and the chair's own 1.1-G estimate-class number was killed by data at the author-adjacent desk and NAMED in the process section — the estimate-law ethos applied beyond VRAM.
- AGENT4'S PRE-CLOSE GIFT (filed 23:5xZ, relayed to agent5 @ same beat): pair-constant scan of attention CONSUMERS CLEAN — kv=1-safe by derivation, no pair-literal anywhere on the live bf16 path (lone KVHeads==2 sits behind a HIP-loud throw), so [C] starts with ONE clean suspicion class: VALUES (page indexing / slot strides / hidden-state path), zero geometry-smell. Tooling handed: [B] trace self-decoding since r15, PROBE_CURL_M=900 traced-fires (~13 s/token), [C] reuses the trace unchanged — content-compare is the new part and the whole point.
- END-OF-DAY DESK BOARD: agent4 sanctioned down (re-arms same-beat for [C]); agent1 dark (fold pre-written, morning seat may run it); agent3 dark (ring-for-guard-drift); agent2 audit-armed (cold-verify receipts land); agent5 holds [C] recipe + NVFP4 A-rows; KFD-0, servers 0 post-kill on all three fires, df 20G, counsel dossier current @ 9135e9f8 (family-10 state, four questions still open).
- TOMORROW'S QUEUE (named, not vibes): [C] fixed-input w2-anchored comparison (convicts attention/KV content or exonerates everything to the LM-head input — the residue both ways is gradeable by the SAME compare); #22 final fold (agent1 or fresh seat, comparison-set shipped); [B]-tooling's pad-region + counter-join cells land with the next census pass; counsel returns folded to the dossier as §5e; NVFP4 deploy gate = TP4 serve-green, everything else on that lane is pre-built and frozen-clean.
- [C] PRE-SCAN (agent4's pre-close gift, relayed to agent5 @ 23:5xZ): attention consumers are kv=1-SAFE BY DERIVATION (q_head/GroupSize, table-driven); the lone KVHeads==2 literal sits on the I8-KV fill path that throws LOUD on HIP pre-use; no pair-shaped head arithmetic in the bf16 bodies — so [C] convicts on VALUES (page indexing / slot strides at w4 / hidden-state path) or clears everything to the LM-head input. Tooling: [B] trace is self-decoding since r15; PROBE_CURL_M=900 for traced fires (~13 s/token). Morning's first organ, zero geometry-smell, tools warm.
- REGISTER HEADLINE FOR THE CLASSES COMPILE: tonight's fifteen-plus disclosed catches (four desks + chair x3: recalled-sha, env-claim, lane-crossread-adjacent partial-greps, phantom-cite relayed, watchdog-gap) each closed with a MECHANISM not a promise — mapping laws, lane tags, three-state exits, stamp-name laws, fork-guards, values-not-existence grading. The ping-pong never dropped a beat; the board never idled a second beat once named.
### STATE 2026-09-14 ~23:4xZ (AMD host) — **PARTIAL CURE BANKED + BRANCHER FIRES: round-13 (83b9c7ae8e3c6fc4) W8 LM-head route fix CHANGED the emission character — request lane now runs 8+ decode steps with REAL mid-vocab ids (220=' ' spaces x2, 79416/116807/89727) vs the pre-cure low-region punctuation attractor — finish=cancelled@18tok is the PROBE budget ending, not the model failing; the [B] discriminator simply wasn't requested (NINFER_R1_TRACE is the gate; chair self-answered from the source at :183 — built right, unset by accident-of-env). Round-13 grant RE-FIRED with the env under the same stamp (LABEL G18w4s), per-rank tuples + step= counters answer the LAST branch: AGREE ⇒ exchange exonerated, [C] (MC31-H fixed-input prefill→logits rank comparison, w2-anchored, agent1's pre-dark named-next) is tomorrow-or-now by agent4's judgment; DISAGREE ⇒ wire named at a step.**
- LATE-BAND LANDINGS: gate v3.1 @ c2f18eec (b0309248) — agent5's §8l PASS carried as an ARM with the fork-blindness red-capture (behavior-identical pure-core fork: live green, GUARD ALONE FAILS; 23/23) — 'a check that cannot see the shipped path is ceremony' generalized to accept-set code; gate family then FROZEN for the serving fire. agent1's pre-dark instrument-map: kv_local=1 geometry fully gated; NUMERICAL attention at kv_local=1 graded by NOTHING — [B]-agree must not be over-concluded past 'wire faithful, corruption upstream'; the decode grid + 276-id pad-region cell + step-counter-skew tell all shipped before dark (f386cd7c/07bb35dd). agent3 dark twice-resumed-then-clean (C9 delivered, guard armed, queue empty).
- ROWS OWED at the release: +6.95MB bank-delta explanation, 20→18G attribution + pre-authorized GC (five serving-cycle + TP2 attractor banks stay), PROBE budget line, [B]-env miss filed as process-not-defect. Counsel dossier current through §5c; the re-fire tuples row lands there too if the branch goes upstream (that's the [C] dossier seed).
- DESKS: agent4 serial (re-fire under way); agent2/agent5 NVFP4-live (audit rolled to v3, §8k re-grade, A-rows); agent1/agent3 dark-sanctioned with named ring-wires (fold / pairing-drift fix-ownership). KFD-0 pre-refire, df 18G.
### STATE 2026-09-14 ~23:2xZ (AMD host) — **SOUP EPOCH, FINAL CONFIGURATION: counsel dossier at 60744b0f (§4b H1-RESOLVED-FALSE/H2-dead/H3-bounded + §5c decode grid: EVERY emitted id REAL — 93021=')}'+14='/'+'AM'=1354, warmup '~'/'¯' — wire EXONERATED BY OBSERVATION, soup = lane-agnostic ATTRACTOR CYCLE THROUGH REAL PUNCTUATION; H2-strong dead, H4/H5 narrowed to shared per-step state/weights).**
- INSTRUMENT MAP AT THE NARROWING: kv_local=1 GEOMETRY fully gated tonight (LEG15c 6/1 emitted row, Gqa27Tp4Geometry=GqaGeometry<6,1,2> exists, runtime-derived wrapper + KVarN ne[1] gate cannot silently disagree mint-vs-consume; 1152 = page-width constant, checked non-suspect). BLIND SIDE NAMED PRE-[B] (agent1's last act): NUMERICAL attention output at kv_local=1 is GRADED BY NOTHING — tuple values agree happily on a wrong hidden state. So [B]-agree ⇒ 'wire faithful, corruption upstream' ONLY; the pre-declared [C] = fixed-input prefill→logits rank comparison (MC31-H class, w2-anchored) closes the numerics gap or it doesn't exist. Framing ordered into the release row so nobody over-concludes off [B].
- [B] BUILD RUNNING (r13b_build.log witnessed 23:1xZ; w8 files + trace sites in the 13-file edit set; step=-per-rank counter-skew field relayed = the permanent counter-join cell; pad-region gate candidate (vocab 248044 < 248320 rows, 276 glyph-less ids) relayed as non-blocking cell with the NVFP4 C9-census question (declared vocab-extent row: arm or gap worth naming) to agent5). Round-13 stamp-swap pre-authorized same-beat.
- DESKS: agent1 DARK end-of-shift (fold comparison set complete in advance: B-table/missing-join law/counter-join cell/decode grid; five self-disclosed catches tonight; final fold fires at [B]'s rows tonight-or-tomorrow); agent4 serial build→bank→fire; agent2 gate-audit rolled to v2 three-probe-exact + §N0 cold-verify in flight; agent3 DARK sanctioned (ring-for-pairing-drift only); agent5 §8k re-grade + pad/vocab look queued. NVFP4 side SELF-SUSTAINING: gate cell v2 (04589781: 700 strict + 1052 membership arms, C8/C8c/C8d, vision constexprs resolved, LIMIT row retired, live rc=0 warnings=exactly-the-2-ruled-rows) authored-audited-reviewed by three desks — deploy still waits on TP4 by the user's own framing.
- DISCIPLINE LINES NOW BOARD LAW (this beat's harvest): sha cites name their branch (agent1); env claims cite their manifest line (chair); absence claims name their branch-set (§7.5b); 'a cell that grades admit/refuse cannot see a wrong answer' (values law); real-ids-exonerate-wire = measure the decode before blaming the transport. df 20G root, KFD-0, bank-GC still queued post-serving-row.
### STATE 2026-09-14 ~22:4xZ (AMD host) — **#9 CLOSED AS PROBE GRAMMAR (agent4 archaeology: MODEL_ID env landed AFTER both DEAD-verdict fires — those were instant model-less 400s swallowed by old verdict grammar; prints-cure hypothesis dead by mechanism; CU-0 capture was accurate, DEAD was the lie) — and #10 OPENED STRONGER: at GEN=512 world=4 legs now SERVE, but DETERMINISTIC TOKEN SOUP across fresh processes: sampler-tear EXONERATED by determinism, R1 champion-exchange ARITHMETIC indicted (world=2 rode the ring all night correct; R1's row-range math has never served a correct stream — untested semantics end-to-end).**
- STAFFING per user order: TP4=2 (agent4: [A] no-BATCH_DBG 512-tok fire under standing grant + [B] champion-staging trace pre-authorized incl rebuild/re-bank round-13; agent1: BLIND static re-read of wire — WHO WRITES per-rank send buffer before who reads it; local→global offset @62080; comparator vs single-home at the wire; conf reassembly; GRADE VALUES not presence, their own 15c law load-bearing). NVFP4=3 (agent5 A-rows moving @54a7ffd0; agent3 census-vs-accept-sets gate cell building from cold brief; agent2 §N0 cold-verify receipt + landmine map).
- EVIDENCE SHAPE for #10: soup string 'REASONING:AM)}.r ikhail和睦…' reads like STRUCTURE BYTES REINTERPRETED (header/frame region as text) — rank-staging-never-written (reduce reads adjacent device memory) is the chair-named prime; determinism across fresh processes = coordinate-offset-by-construction, not race. Contract to grade against: agent4's own §1 port spec (tok ALREADY global coords, tie-break val-desc/global-asc) + agent1's 12B static_assert lineage.
- ALSO: agent1 QUARANTINE commit (ab81c014) — phantom-message hole caught by agent5's real seat, confirmed at agent1's; their lane tip 05012259-adjacent; the sha-remedness law continues earning its keep. Lock-identity witnesses (cap-mutex=0x…, engine-mutex=0x…) printed and useful — agent1's seq-170 spec validated in its first fire.
- NEXT BEATS: agent4 [A] verdict (clean 512-token text ⇒ the soup was observation-coupled and green is HERE; persistent soup ⇒ [B] trace round-13 + agent1's static find race each other — both paths end in a named organ); agent3 gate cell sha; agent2 receipt; agent5 A2/A3 named triples. Serving-row stack for the release when it lands: #9-grammar-closure, #10 resolution, DCE witness, kind-correction, rank-0-sufficiency, stamp-law near-miss, console-retraction, quarantine note — this row is becoming the era's doctrine digest; the classes compile (laws of 09-14) goes in docs/amd the moment a seat has slack.
### STATE 2026-09-14 ~22:1xZ (AMD host) — **ROUND-9 FIRES NOW (stamp b8f503157997f352 read-and-carried, tip 8b03629a, LABEL G18w4n): six-edge chain 1→1a-parsed→1b-prepare-entry→1c-lifetimed→1d-input-built→1e-engine-prepared→2, each edge naming its suspect (httplib-body / progress-logger / CAPACITY-MUTEX / media-guard / frontend internals / unwinding) — the 1-no2 organ bisects to ONE edge this fire.**
- ROUND-8 FULL CLOSE (agent4 receipts): transport ran a THIRD boot (warmup mints 93/107 per fire); sites 3/4 absence = shared-mutex-holder hunt CLOSED BY OBSERVATION (not elimination-by-reading); TWO honesty files banked for the release row: (a) round-8's NEAR-MISS — agent4 passed a RECALLED sha tail, GATE-2 rejected rc=5 pre-spawn E0-style, zero device touched, 'the stamp law caught my own shortcut' now cited as the ceremony's best defense; (b) CONSOLE-MUTEX RETRACTION — 'models proves log_line liveness' was FALSE (models never calls log_line); negative list self-corrected in public, sub-span chain tests that region directly.
- AGENT1 FOLD LANDED (46de7aa2): T-a HOST-SIDE branch = transport exoneration now POSITIVE-state (zero-R/all-S/CU-0); T-b graded TRUE-but-UNDER-DETERMINED ('coarse probe can be right and still not enough'); T-c NOT-RUN, NCCL_DEBUG measured ABSENT — CHAIR ENV-CLAIM CORRECTED AT MY OWN DISPATCH ('kept' was message-true, manifest-false; env-set claims need the manifest line); T-d 'stacks do' line under-graded breadcrumbs — resolved WITHOUT stacks by trio+chain, register lesson: pre-declared instruments retire prime suspects in one print. THEIR SUSPECT-SET FIND: pre-:668 is lock-free for ENGINE, NOT for SERVICE — cap-mutex at generation_service.cpp:331-337; lock-identity-witness spec (%p at takers, arg0 retro-decode, 'a name is not an accusation') forwarded to agent4 for round-9/10.
- NVFP4: agent5 pushed c9478e44 (hour-1 continues); inventory-vs-artifact gate has ONE more beat before the unclaimed-fix ruling assigns agent1 (host cell: manifest triple vs converter-inventory derivation, throw-on-divergence, agent2 audits). Fresh agent1 desk: battery green at boot tip 6aab7258, bin re-hashed, zero orphans, capacity held. agent3 parked (5 beats), legs covered.
- NEXT: round-9 rows (which edge goes silent); attempt-11 = cure-at-shape + #9's permanent cell (edge-census joins law-(E) family?); disk 22 G root, KFD-0, watchdog clean. The release row for the serving fire, when it lands, carries SIXteen stacked verbatim lines — it's becoming the project's de-facto doctrine doc; consider a docs/amd/ consolidated 'classes + laws of 09-14' landing right after, both lanes' desks have earned the compile.
### STATE 2026-09-14 ~22:0xZ (AMD host) — **FAMILY #9 ORGAN-SCOPED IN ONE PRINT: round-8's 4-site breadcrumb chain returned '1-handler-entry x10+, ZERO 2-post-prepare' = the chat handler's parse/prepare segment — ENTIRELY UPSTREAM of the engine; the shared-mutex-holder hypothesis (chair + agent4's joint prime suspect) is DEAD by absence of 3/4. Warmup anchors (3/4+enter+SEQ-STEP1 tokens 93/107) confirm bypass-consistency: the instrument bisected the whole request path on first use.**
- TP4 PATH NOW: agent4 reading http_server.cpp parse→prepare segment (capped-wait eliminations covered engine-side; whatever's uncapped lives handler-side BY CONSTRUCTION). Cure-at-shape + bank + attempt-11 = the serving fire. Milestone-row stack for that release: transport-ran (again), four-way WDBG bisect, #9 organ-scoped, + queued verbatim lines (kind-correction, rank-0-sufficiency, DCE, kvarn source-law).
- AGENT HANDOFFS COMPLETED: agent1 desk passed dark->fresh with zero orphans (dict rows #20-#23, arming kit main @ 8ba9f05c-adjacent incl their mint-check --falsify 11/11 chair-re-verified; fresh seat's first report: 19 legs green at 5ea26237, anti-res of breadcrumb chain independently verified [include + 8 dbg-gated lines, preflight zero], PART-7 R1-trace spec reviewed: BUILDABLE with shape-A/B pre-declaration law + time-scope finding 'cannot witness tonight's wedge, downstream of it' -> HOLD post-round-8, candidate owner agent5). agent3 seat PARKED after 5 beats dark, all legs covered (their cells on main, harness superseded by landed artifact anyway).
- NVFP4 HOUR-1 LIVE: artifact landed+sha-exact (eaf8ad12…56d2, 18,324,067,840 B, /media — root disk untouched); HF support patch RULED OBSOLETE (Aug 5d2c1f55 integration already in tree: Qwen38Nvfp4 + bind_qwen38_nvfp4_text_layers — verified at bytes, two desks independently); IDENTITY GATE DEAD — same (qwen3.8-27b,nvfp4) pair shipped two incompatible object layouts (9/8 split-GDN vs today's fused-a_b/no-divisor-GDN) -> hour-1 is CONTENT-SHAPED (divisor count, endpoint (format,layout,bytes) triples, fused-vs-split); stake resolved by TWO independent artifact reads (agent2 manifest census doc-29 + agent5 cell): endpoints W8G32/row-split — producer inventory_nvfp4.py STILL DECLARES FP8: third #23 instance, producer-vs-artifact contradiction with no gate, routed to agent5 §8h; per-format-two-geometry-builders audit law in effect for the accept-set enumeration.
- LAWS ADDED THIS WINDOW: searched-and-absent claims name their branch set (§7.5b, fresh from a wrong-branch self-catch); dangerous-enum = formats with TWO geometry builders accepted by one/reinterpreted by other (loud-vs-silent polarity explained); presence-guard arms a format-check debt (a has_tensor guard would convert tonight's layer-0 loud death into a 64-layer silent misread — row #23's cure law: binder format-check or explicit ANY-marker at call sites).
- NEXT BEATS: agent4's segment read -> named organ (+ possible 5th breadcrumb); agent1 boot-tip battery + #22 fold; agent5 A-rows (endpoint triples named-printed, inventory contradiction routed); agent2 §N0 cold-verify receipt still owed. Disk root 22-23 G (report-only, bank-GC queued post-serving-row), /media 190 G free. KFD-0. ptrace sysctl: surfaced, not needed — the breadcrumb chain retired the 'stacks required' claim.
### STATE 2026-09-14 ~20:3xZ (AMD host) — **THE TRANSPORT RAN IN-SERVE. Attempt-9 (G-AMD-40 r5, bank 09c8cdc1355ffe59, tip e84b14dc): all four TextContext-ready, SEQ-STEP1 steps 1-2 minted tokens (next=93/107) with ZERO worker errors — warmup-path four-way collective completion PROVEN (rank-0 print is sufficient: the allgather has four participants). Nine boots, eight families cured at the shape level, each with a permanent cell. REMAINING: FAMILY #9, A NEW KIND — 'LISTENING-BUT-NOT-SERVING': /health answers, request probes DEAD, NO error lines = HANG (first zero-signal death of the night; rows #1-21 all threw).**
- FAMILY #9 SHAPE (chair): warmup takes the per-step decode collective; real requests enter PREFILL first — suspect rank-divergent prefill collective sequence (vocab-prefill PTX-held arms agent4 flagged early, GATE-3 gather route, metadata broadcast). agent1's 14:0xZ stall-watch-frame ('capture stream state BEFORE any kill; that IS the datum') earned its keep — now dictionary row #22 with pre-declared tell-forms (per-rank capture-before-kill; 1-token-vs-512 probe split = decode/prefill discriminator; NCCL_DEBUG=WARN mismatch prints). Agent4 owns capture protocol for attempt-10; grant round 6 gated on it being NAMED in the ask (a hang without a stack is a datum you cannot read twice).
- LAW-(E) PAIR-STATE CENSUS LANDED (agent4, in e2b20ea8 family-#8 cure): 18 fixed-pair arrays registered by name+reason across 3 files — two rings pair-internal-BY-DESIGN (PRED-D-witnessed guards + re-check-when-quads notes), zero_pair registered as agent1's value-decoy, the cured tp2_request site verified GONE BY THE CENSUS; agent1's predicted third site (conf/host_payload[2]) adjudicated FOUND-BUT-GATED (device-pair-slot-indexed, world>2 refuses at ring guard) — registered, not migrated, correct-by-name. Family #8 was cured AT THE SHAPE (std::vector sized by the SAME backend.world() that arms sync_bar at all three construction sites + index-guard naming both numbers) — the mint/consume law (#19) applied prospectively, not just diagnostically.
- LAUNCH-LATENCY PATTERN (agent4, filed): 'launch ALONE, verify LOG-EXISTS, then poll' — never-created logs are invisible to mtime-watchdogs (chair's included, gap named); bank GC approved-in-principle post-serving-row (sha-list + chair confirm, ~1.1G, mine-not-killed rule).
- BOOTING/REVIEWER STATE: agent5 reviewer row now carries the transport-ran proof (six-families-at-build template + per-rank completion argument); booting-seat offer stands for attempt-10 if agent4's ctx demands; agent2 owns the one-command battery at boot tips (kit on main, cold-run verified). NVFP4: rev 3.6 current, runbook-integrity pass in flight (agent5), audit home live (agent2), harness tick owed (agent3, deadline this beat). KFD-0, df 21G in-row, disk plan holds through attempt-11.
### STATE 2026-09-14 ~20:1xZ (AMD host) — **ATTEMPTS-7/8: SIX TELLS SILENT THREE BOOTS RUNNING, ALL FOUR RANKS HIT TextContext-READY + gating-schedules-LIVE (deepest boot in box history), died at FAMILY #8 = per-rank pinned-slot array plain_tok_pinned[2] (tp2_request.h:47, alloc slots [0]/[1] only; rank 2/3 index past -> hipErrorInvalidValue one line AFTER allreduce_argmax RETURNED — a hang doesn't throw; attempt-9 with NINFER_MB_DBG=1 turns 'did the transport run' from inference into per-rank printed fact). W8-SIMT join was REAL-EXISTING TU LINKED (zero-mma/asm/shfl measured in include-chain, 3 stubs deleted with provenance, parity 217/217, sha-MOVE flagged 68f2d95->4581413 expected-not-drift).**
- AGENT1 HANDED OFF CLEAN AT 79% (my directive, their own starvation-test proved it): arming kit NOW SHIPPED TO MAIN @ 1917a8b7 (7 files incl tp4_arming_battery.sh + value-grader + fence-plant; chair cold-run at main home: 18 legs GREEN exit 0 at b2ee6e89 — 'author-absent-callable' now a measured property; agent2 owns the command). Agent1 residual: row #20 append + battery-at-boot-sha only. Their #19 census law ('a migrated constant is half-migrated until the MINT side reads the same source as the CONSUME side') is tonight's TP4 thesis — #8 IS its third instance.
- NVFP4 STACKED TO REV 3.6 (16184d25->db82e9b2, agent5): agent1's 97-factor ceiling CLOSED in 5 min zero-card with TWO self-corrections routed through agent5's fold-as-task (embedding REPLICATED not divided — axis lives in the LOADER; w=7/14 admitted by qkv — law = FACTORIZATION-DERIVED, 'power-of-two intuition wrong in BOTH directions'); §8 artifact-day runbook A1-A7 every row pinned to an existing already-passed instrument, hour-1 = INPUT SWAP not first-runs; A0 axis-check row added; class-row banked: 'cross-desk routing IS an instrument; self-review demonstrably not sufficient' (agent1's error caught because another desk made it a task). §8c decoy/BOUND≠SERVED folded to agent4's canonical paragraph, one home.
- LAWS/TICKS: compiled-fence grant-term VERIFIED-INSTANCED (agent1 g++ plant: colliding row FAILS compile with migration instruction in the assert text; control plant quiet — tier_shape_table.h:206-214 the model, HEAD_FAMILIES cites their plants as spec); census law-(D) T1-T4 complete at agent4 (narrow 75/17 all-named, tp_group n==2 registered WITH reason, T4 20-diffs->8 reason-rows, register-grows-by-measurement); agent4's E-sheet graded E2 twice correctly, kind-scoped STEP-0/C lived its first production boot, agent5 verbatim correction line queued for release row; #18 prediction resolved NOT-FIRED (both-ladders-one-commit, prediction credit both ways).
- SERIAL STATE: agent4 five-line join -> re-bank (#9 stamp) -> amend-ask #5 -> attempt-9 FIRES with DBG env. Next board-shaping datum: per-rank SEQ-STEP1 lines = transport RAN verdict per rank. KFD-0, df 24G, pair held by agent4 through the cycle.
### STATE 2026-09-14 ~19:3xZ (AMD host) — **ATTEMPT-7 BUNDLE RULED (one commit, four contents): cache-producer world-derivation + produce-time pair-named throw (gqa :231/:394's upstream setter) | (v) letter-block + literal-census allow-list arm + runner kind-scope | COMPILED-ASSERT GRANT TERM (agent1's sabotage control: python-only fences are single-tier — dead if-cond left generator AND --check silently GREEN, byte-identity can't back a fence it also rewrites; legality/collision laws guarding generated emissions must exist as static_asserts IN the emission) | agent5's kind-mislabel correction line verbatim. Battery at fix tip 27bcb625: 17 legs ALL GREEN, boot tip 469c2feb: exit-1 with PRED-E red = the correct answer (an all-green print at the booted tip would be the instrument that stopped seeing).**
- LAWS REGISTERED this beat: 'a cell that grades admit/refuse only cannot see a wrong answer' (agent1, value-graded leg-15c ordered: kv_local must EQUAL derived truth per row, refuse-with-reason is a value too); NOTE(legal-unserved) vs FAIL(false-admit) severity split ('only a false ADMIT corrupts tensors' — over-conviction's third cost lesson, self-applied); #18 pre-declaration resolved NOT-FIRED by pre-emptive same-commit both-ladders fix (prediction credit both ways); reviewer self-retraction before hardening (agent5: own relay re-tested, wrong-mechanism named, sandbox visible-skip law vindicated — verification discipline at the reviewer's own desk); family-#5 PASS five-way at build/CI ((v) 255/255, per-ladder law C, byte-identity, pair-named wrapper throws arrived WITHOUT the order, anti-res 0 lines).
- NEXT: agent4's bundle push -> re-bank -> amend-ask -> agent4 fires (agent5 booting-desk standing) -> agent1 battery seconds -> attempt-7 grading sheet = five-families-at-build + STEP-0/C right-leg + produce-time throw + §7 honesty line. Serial chain ~25 min if clean; family #6 or argmax, both are rows.
### STATE 2026-09-14 ~19:2xZ (AMD host) — **ATTEMPT-6 E2 AGAIN BUT DEEPEST BOOT YET (4 ranks, TextContext 64-layer ready, gating schedules live) — third gqa site found by CHAIR GREP: gqa_attention.cpp:231/:394 validate_batch_cache grades cache.num_kv_heads (pair-math-carrying TENSOR METADATA) against table per-rank (=1 at w4): the same family's one-stage-earlier lie; ordered into agent4's cache-produce fix + PRODUCE-time loud throw naming both numbers. Ladder migration 27bcb625 otherwise PASSED full review — bank 4d1941d16a37e5e3 (123,690,744, size-growth-named, chair-verified stamp), G-AMD-40 amended round 2 (amend-in-place precedent), LABEL G18w4f_p3serve.**
- AGENT1 GROUND TRUTH (78edaac1) CONSUMED AT FULL DEPTH by agent4 pre-emission (row-append form STOPPED mid-write; pair-keyed table view + generation-time collision assert + DFlash non-emission NAMED with minefield map + decode.cu:490/prefill.cu:252 catch-alls found+armed — 'hazard reborn one organ over'; kvarn arms honesty paragraph: source-law coverage, zero runtime surface tonight, CMake line named for the future). agent1 leg-15 = plant-the-collision vs generator assert, in flight; their q_local-projection expiry fence accepted as their answer to 'not a valid key'.
- AGENT5 REVIEWER ROW @ 469c2feb: FOUR TELLS ALL PASS (each credited to its PERMANENT CELL — loader/state via (u)+STEP-0/F class-chain, Q3 by the table five-ways incl my-seat (v) 166/166 + DCE-in-BIN witness, catch-all by parity cell's six reproduced arms with the constexpr-vs-file-level deviation argued non-silently). HYGIENE RULINGS: (v) letter-block YES (ctest insufficient, CI-farm blind; ordered); wrapper-gate literals linear_add.cpp:195/211/227 = step-2 migration list + zero-literals-outside-table grep-arm YES. STEP-0/C RCA = NEITHER chair candidate: runner counts INSTRUMENT-ERROR globally (:486) attributing all to kind (:487) — G6a/G7 honest never-ran refusals wore a FALSE kind label; agent5 tagged-reader fix @ r1 203a05bc (merge-routing: agent4 pulls via lane merge); correction line rides attempt-7's row per annotate-not-delete (the mislabel UNDERSTATED rows — E2 grades corroborated by the refusals themselves).
- AGENT2 fire @ booted tip 469c2feb closed (791cc120): substantive suite continuity through the boot pair. AGENT3: synthetic-container harness in flight (§N0.1/§N0.3 park-killing). CHAIR-CYCLE LAW HOLDING: 300-s beats, dispatch-producing; wall curve = each boot dies strictly deeper, family #5 named three times over — the class is tensor-metadata pair-math everywhere, the table is the cure and it's now ONE PRODUCER-SIDE FIX from the argmax.
### STATE 2026-09-14 ~18:4xZ (AMD host) — **ATTEMPT-5 FIRED+RELEASED under G-AMD-40 (agent4, bank 1801cbd0844991fa, tip 469c2feb, LABEL G18w4e_p3serve — first lane-scoped-LABEL fire): PASSED loader+state-spec+Q3-quarters walls (TP4 tag, 3820==3820, four ranks) and died LOUD at FAMILY #5: `gqa_attention: unsupported Q/KV head geometry` — literal ladder gqa_attention.cpp:22-26 {24->4,16->2,12->2(w2-comment)} with throw-elsewhere; world=4 per-rank heads off-ladder. Zero A1TRACE persists (R1 transport STILL unrun in-serve; wall now = attention geometry, one small ladder, smaller than the table).**
- RUNNER MATURITY: new STEP-0/D sentinel-guard FIRED CORRECTLY (25/25 sentinels -> attractor VOID + 'run is a SERVE FAILURE' — the 7bb-noise class dead by construction); STEP-0/F pair-math tell quiet; pre-spawn PRED receipt carried witness stamp 2df94a429d IN-LINE (attempt-3's unlogged-gate class structurally gone); rc=4-127 wrapper saga: agent4 self-cured in-log (attempt kept as .attempt1-wrapper-pathfail), chair's alarm was a stale partial read — corrected same-beat; chair BUILD-WATCHDOG now tails all lane logs (loud on nonzero/stale BUILD_EXIT, dedup'd, running pid-checked).
- AGENT1 b1deceaf: 14-leg arming battery GREEN at d6b29501 (both host cells run at seat, PRED-B all sub-legs, step-1 conformant ruling delivered: gate table-driven, four chains end in parity-break throws, per-role static_asserts landed, the 'append an ELSE' anti-comment quoted approvingly) + PLANTED FIND: agent4's check_whitelist_arm_parity ACQUITS half-armed specimens (sets lack multiplicity; q3 T<=1/T>1 chains separate -> 'only long prompts wrong' class) — fix ordered (per-chain constexpr admitted-K + static_assert); PRED-D's own two self-caught reading defects (tail-captures-rc; bool-literals-as-shapes NOT-OBSERVABLE) filed with the pipe/exit class tally now FIVE desks/sessions incl chair-watch twice. Dictionary row #17 = gqa ladder, annotated 'w2-local number masquerading as universal'; battery to run at booted tip 469c2feb (v6 continuity leg).
- AGENT5: reviewer armed at emission (four-tell bar; DCE paragraph count-2-not-4 accepted as their content-witness; §N0.4 host-cell verdict unblocks N0 fully zero-card; booting-desk offer stands for attempt-6+ replication once agent4's ladder fix cycles). NVFP4 rev-2 + §7/N-LATER current. AGENT3: §N0 follow-ons + schema-drift read stand; continuity re-run target = 469c2feb (already booted) and next bank's tip. AGENT2: substantive fire cued at attempt-6's release row; incident writeup pending. Gemini per law, held unspent.
- NEXT SERIAL: agent4's fix-commit (gqa ladder world-derivation + parity-cell multiplicity + step-0/C reader hygiene) -> rebuild -> re-bank -> G-AMD-40 AMENDED -> attempt-6. Honest read for user: walls shrinking (table-lane-day -> one-ladder fix); each boot passes strictly further; attempt-6 is plausibly the argmax boot, and the ladder family + ref-tier fallbacks remain the named-not-silent tail behind it.
### STATE 2026-09-14 ~18:3xZ (AMD host) — **AGENT4 DETACHED BUILD RUNNING 18:18Z (setsid law held this time); SPEC REV-2 LIVE d69ccb0b (agent5 rebased clean over hazard-patch — non-FF caught, patch intact: the lane-under-hotpatch sharing answer banked); ref-tier fallback audit RULED post-first-light (§N-later, agent1 grader, agent5's exact honesty line reserved for attempt-4's row).**
- REV-2 CONTENTS relayed to agent4 pre-emission: FP8_ROW_BF16S was spec-side PHANTOM TOKEN (zero tree hits; real FP8_E4M3FN_ROW_BF16S — would have false-REDed lawful step-1 output), world-3 falsifier arithmetic corrected (5120/17408), ARM-EXISTENCE LAW = spec text now (catch-all :97/:108 <5120,8704> silent-wrong class, gate-set==arm-set constexpr cell is definition-of-done). §6.2 yardstick = FOUR boot death-tells must die at BUILD time (pair-math, conv_state, Q3-quarters, catch-all-silent).
- NVFP4 FRONT: agent3's four new cells verified sha8-first at agent5's seat (20eb99a7/81aced51/41d5e020/143650e4); §N0.4 VERDICT: nvfp4_shard_image IS a host cell (13-object closure, zero /dev/kfd strace-verified) -> N0 fully buildable zero-card, artifact-wait unblocked from its last excuse.
- CHAIR WATCH: build-start tick witnessed; awaiting header/bank shas; grant naming next (3x-space check pre-issue: issued so far 30/30a/34/35/39 + reserved 36-38 pattern — next fresh is 40+); agent4 fire-ready owns runner+expectation sheet+grading cell (single-desk closure of the boot cycle this time); agent1 v6 pre-authoring; agent2 substantive fire cued; agent5 reviewer armed + booting desk for attempt-6+.

### STATE 2026-09-14 ~18:2xZ (AMD host) — **AGENT1 SHAPE-TELL DICTIONARY LANDS (f2840698, 16 rows, zero-card) — TWO EPOCH-LEVEL ITEMS: (1) CHAIR-AMENDED EXPECTATION: step-1 q3-only PLAUSIBLY REACHES FIRST ARGMAX (Q4/Q5/Q6/W8 generic tails, no strict whitelist, world=4 quarters measured %32/%64-clean — WO-TP4-F's 'NO' premise-true/inference-false, boot budget compresses); (2) FALSE-DATUM TRAP CAUGHT PRE-BUILD: q3_dispatch.cpp:90-110 CATCH-ALL `ELSE launch<5120,8704>` — widen admission without per-K arms + terminal else-throw and the quarters silently run wrong strides, OOB reads, CUDA_CHECK SUCCESS: attempt-4's datum corrupted as an R1 failure. HARD GATE issued: 'widen-without-arm-edit is a REJECT' + agent1's closure cell (gate-set==arm-set constexpr; cannot-link-if-disagree) + standing PRED `ELSE launch<` grep + agent5's spec line 'admitted shape's legality PROVED by instantiation existence, not assumed from whitelist'.**
- AGENT5 BACK IN FLOW: NVFP4 plan REV-2 pushed (97d70217 amd-wo-nvfp4; §6 folds agent4 cell e6db3027 run-verified + agent3 content-stamps RESOLVED (0fe01f38/1fa3c219 = uncommitted-desk-file sha8s, class = content-stamp-vs-commit-sha, my transfer-notice 'shipped-talk refuted' refiled as stamp-kind mismatch not conduct) + N7 world-aware re-spec; §6.2 review standard = grade agent4's emission vs the three boot death-tells, credit only if all three die at BUILD TIME — adopted, catch-all added as 4th tell).
- AGENT1: v6 row PRE-AUTHORING with frozen wording + empty data slots (open-risk sentence ACCEPTED as drafted incl fourth-outcome silent-wrong branch); LATENT/REACHABILITY self-correction (wrapper gates go live only when dflash2/MTP rides a TP rank — sequencing not wall) filed as model behavior: 'plausible structure, generated not observed' named in own words. AGENT2 continuity fire at c8264e22 already in (letters (a)..(u)=21, (u) field-arms asserted on REAL boot data both directions, re-pin ebc4a4d5 verified three-probe at seat).
- SERIAL STATE: agent4 heads-down in p3-serve (2c3dea46 merge + chair hazard-merge 168172ee below their feet — pull-before-push notified; LABEL rule per-fire lane-scoped STANDING). Ticks awaited: header-exists -> build-start (setsid, in-lane log) -> bank-sha -> named grant -> agent4 fires own runner. KFD-0, disk ~24G, no other builds. Gemini per law.
### STATE 2026-09-14 ~18:0xZ (AMD host) — **STEP-1 IMPLEMENTATION TRANSFERRED agent5->agent4 (§6 unavailable-trigger: lane measured clean at c8264e22, no table code exists anywhere — the 'shipped' talk was a claim my tree-check refuted: newest bank = a5ef99e5 PRE-TABLE, 4471f890/c587965f/23f9921d resolve in no object DB (phantom pins — two desks caught it, nobody graded air — the receipts law holding under live fire); user's GREEN-LAUNCH-ASAP order in force, every seat re-tasked around the single serial resource = agent4's implementation.**
- ARBITER RULING (user asked 'would a q4/native artifact be easier'): NO, three grounds banked — current boot artifact IS the ninfer-native .ninfer (1124 objects, manifest-measured placement is what preflight reads); the quarters wall is ARCHITECTURE-derived and EVERY tier family carries the same TP2-halves whitelist (boot-3's throws + table-map evidence) so a re-quantized artifact hits different pair-shaped tables, dodging nothing; and swap cost is real: world=2 attractor 348e77a1222dea7f measured on THIS artifact, plus every grant condition re-derives. If anything, /media/chris/EMTEC256/qwen3_8_27b_q3.ninfer size-match at agent4's stat (15,446,796,288 EXACT) makes rc=4 deaths a fire-ENVIRONMENT input question (stale ARTIFACT_BYTES re-stamp = input correction, not weakened assertion — agent4's precision accepted). Post-first-light: higher-precision artifact = quality conversation, not shortcut.
- AGENT4's IMPLEMENTATION BRIEF (as sent): merge origin/amd/wo-r1-transport into amd-wo-shim-funcattr (spec+release-rows+checks (u)/(t) ride), generated tier_shape_table role×world {1,2,4}+{3,8} falsifiers, NVFP4/no-gfx900 = NAMED LOUD REFUSALS (their own --table cell grades the emission — implementer AND grader of the schema, single-home kept), q3 quarters {4352,1536}, Check (v) host cell, call-site swap, DETACHED builds only (setsid+in-lane log — the 17:40Z loss rule), bank-before-relink to artifacts_bin, then sha16+size -> fresh named grant -> agent4 fires their own fire-ready runner (protocol line already written: BIN/BIN_SHA/TPG_SRC/W4_ACK).
- HAZARD CULL #2 (post-incident sweep, chair-initiated): BOTH active lanes still carried the RECURSIVE selftest a94c6af1 while gate (t) calls the TREE battery (agent2's continuity row surfaced the wiring) — agent4's build cycle could have re-detonated the fork-bomb. Chair merges: p3-serve <- main 66366ede @ 168172ee (battery ebc4a4d5 depth-free verified rc=0 in-tree, checker c77c4396 + python 2df9429d synced; agent4 pull-before-push notified), r1 <- cherry-pick 63120922 @ fd4f9b64 (recursive-cp=0; note r1 copy is patched-v7-generation — less strict, safe; agent5 syncs at leisure). LABEL-COLLISION FIND (agent4, attempt-5/6 murk root cause): cross-lane default LABEL G18w4 overwrote receipts silently — STANDING RULE issued: per-fire lane-scoped LABELs (G18w4<letter>_<lane>), manifest lacking one = receipt-integrity refusal at chair seat too.
- BOARD MAP 18:0xZ: agent4 = TP4 step-1 (build slot holder); agent5 = device-side reviewer of agent4's pushes ('would it pass the walls you crossed' — same-desk skepticism) + NVFP4 plan-of-record rev (fold appendix+cell+parks); agent1 = arming re-aimed at agent4's sha + shape-tell dictionary WITH THE LOUD QUESTION (do Q5/Q2/gdn families throw before argmax even with q3 fixed? decides step-1-minimal vs whole-table); agent2 = continuity fire at c8264e22 + incident writeup + re-pin FINAL to patched selftest; agent3 = §N0 follow-ons + schema-drift read of agent4's --table arm vs ac030ed8 emission contract (same-day-save vantage). Gemini per test-lane law, user-surfaced. KFD-0, disk 24G, zero running builds (agent4 to start theirs).
### STATE 2026-09-14 ~17:2xZ (AMD host) — **§0 CONTINUOUS-ASSIGNMENT LAW BANISHED (user order): work-or-be-assigned every turn, PING-PONG on every completion-report, CPU work grant-free. All five seats are FRESH SESSIONS — full re-briefs dispatched with entry-doc pointers + concrete legs. RUNAWAY BATTERY INCIDENT CULLED+PATCHED (agent2's selftest recursion ate 17 GiB /tmp + ~15.9k PIDs; chair patch @ 63120922 depth-free; disk 25 G, procs 0, boot path never exposed).**
- CORRECTION OF MY OWN 16:5xZ ROW (stated-not-smoothed): 'attempt-4 IN BUILD' was WRONG — the build I witnessed was the state-spec one finishing (log 16:06, bank a5ef99e5 = attempt-3's bank); the tier_shape_table CODE (WO-TP4-F step-1) is NOT yet written and the lane tip c8264e22 is spec+folds only. Lesson: a running cmake is not a proven WHICH cmake — next time match the build process to its log path before printing 'attempt-4 building' in a wake row. Also honest answer to user's timing question with tonight's measures: single-TU+link = 3-5 min (state-spec f2e5cba4→a5ef99e5 chain); header-widely-included sweeps = the 17-25 min class (baseline seed build ~25 min, loader fix ~15-20); the debrief's own '~17 min incremental' agrees. A tier_shape_table touching every family's dispatch header is the WIDE class — expect 20-40 min for step-1's first full build, minutes for every one-TU iteration after.
- DISPATCH MAP (17:2xZ, fresh-session briefs, all directs): agent5 = TP4 HOLDER: WO-TP4-F step-1 implementation (q3-only quarters, Check (v) host cell, warm-lane build, sole slot, bank-before-relink, fresh grant-ask — pair released); agent4 = attempt-4 prep (committed-sha-only firing confirmation, step0-must-carry-gate-verdict-line, expectation-row pre-write for the next shape-tells, nvfp4 role-rows coordination w/ agent5); agent2 = incident absorption (verify chair patch at seat, RE-PIN final to patched content8, doc-26 incident row in own words) + suite continuity fire at c8264e22 watching Check (v) arrival; agent1 = attempt-4 arming (2 host cells + materializer line + PRED refresh at step-1 sha, GO-AMENDED v6 = the transport's first in-serve datum) + shape-tell DICTIONARY (which throw names which family — coordinate w/ agent4's expectation row, don't double-write); agent3 = frozen-2df9429d witness continuity on standby + NVFP4 §N0 follow-ons at desk path (divisor-word surface, nvfp4_shard_image host-linkability answer — decides §N0.4 cell-vs-lane).
- BUILD/GPU STATE: no builds running, KFD-0, pair released, disk 25 G. Next serialized event: agent5's first commit sha (echo awaited by 4 armed desks).
### STATE 2026-09-14 ~16:5xZ (AMD host) — **ATTEMPT-4 IN BUILD (tier_shape_table step-1, agent5 lane, cmake RUNNING, bank pin a5ef99e5->next); agent4 leg-wave landed (runner READY-gate fixed, step-0 GREEN at e944d237, NVFP4 inventory shipped + M1 self-retraction); agent3 continuity witness pre-filed + new AUDIT FIND: attempt-3's step0 carries no ring-guard verdict line — unlogged-vs-skipped ordered resolved at runner owner.**
- AGENT4 @ 3eeab3b9: 8c17fd13 STEP-0/A no longer calls a listening socket READY + STEP-0/D no longer hashes its own failures (both chair-orders executed); c21ea6c9 four PRED arms GREEN at e944d237 + bank-pin verified; 9c43cc9a NVFP4 route map + 3 missing homes, then 3eeab3b9 RETRACTING own M1 ('placement is tier-blind' FALSE; the line-COUNT was accurate at three tokens, the INFERENCE wrong — genre: 'the way it was false is worth more than the finding would have been'). Runner lineage for attempt-3 = /tmp salvage extraction post brace-bug — trace ordered (ran-unlogged vs didn't-have-it), and step0-must-carry-gate-verdict-in-name guaranteed for attempt-4.
- AGENT3 CONTINUITY ROW (pre-filed, citable): gate 2df9429d GREEN at ac030ed8 (doc-only atop e944d237), chain re-verified at those bytes (:136 ring@n==2 / :426 guard / :427 predicate-in-arm / :430 return / :432-445 world-derived R1 arm / commit-last), one_shot_argmax.cu diff EMPTY c5ab9226..ac040ed8; cite-form ruled close+continuity. THEIR FIND (a): G18w4d_step0.txt has NO ring-guard line — verdict-shape named: worst case it's a RECEIPT gap not a SAFETY gap (graded property GREEN at every booted tip by two desks' measurement regardless), but the law's subject IS the receipt. THEIR NOTE (b) endorsed as release-row wording: gate rc=0 = SHAPE verdict, NOT token-coordinate evidence; coordinate class graded ONLY by attractor-vs-348e77a1222dea7f, unpre-declared for world=4, risk stated OPEN.
- NVFP4: agent5 authored WO-TP4-F spec @ ac030ed8 (their chair-stamped 25a6fc05 approval; step-1 q3-only staging held). agent3 §N0 cell 0fe01f38 + notes standing; agent4 inventory feeding table role-rows (coordination ordered). Gemini per law.
- HOUSEKEEPING: pinger DUPLICATE-LOOP CULL — Sep-12 orphan loop.sh (1106915) double-delivering stale wakes alongside canonical (1123679); orphan killed, pinger.pid restored (file vanished at cull — recreated pointing canonical), watchdog 1117069 untouched, claim file intact; disk 16G (agent5's banks+build consuming, named-by-law expected; if <12G at attempt-4 preflight the allocator answers, per VRAM law nobody estimates — but chair df-watch stays in-row).
- NEXT BEATS: agent5 build->bank->grant-ask (G-AMD-39 amend round 3 pre-authorized in my last msg); agent4 gate-lineage answer; attempt-4 step0 MUST show ring-guard verdict line or it's an INSTRUMENT-ERROR refusal at MY seat too. Watch for first A1TRACE = R1 transport's first in-serve breath.

### STATE 2026-09-14 ~16:3xZ (AMD host) — **WORLD=4 BOOTED AND LISTENED FOR THE FIRST TIME (attempt-2), TWO LOADER-CLASS FIXES CYCLED CLEAN, AND THE REMAINING WALL IS NAMED AND STAFFED: WO-TP4-F (generated tier_shape_table, role×world, quarters alignment-proved legal). Pair RELEASED clean; board idle; R1 transport's first in-serve execution = attempt-4's grader.**
- CYCLE RECORD (agent5 release row @ e944d237, all boots KFD-clean zero residue): boot-1 408916e2 OOM->loader CLASS-2 fix f2e5cba4 (tp_w=group.size() + gather-family 28-line sweep + sync_bar(world)); attempt-2 319c4b6f READY + 4 ranks + placement==materialized 3820 (STEP-0/F three-witness proof) -> died conv_state [C,3] (chair-located tp2_backend.cpp:775 /2 family) -> state-spec fix 7d5ffcdb; attempt-3 pre-spawn abort (agent4's UNCOMMITTED runner brace-bug — law filed: runners fire from COMMITTED shas only, worktree edits not boot-legal); attempt-3-boot a5ef99e5 READY -> died LOUD `linear: unsupported Q3 shape` = THE QUARTERS WALL: every tier family whitelists TP2 halves as template instantiations ({5120,6144}-family; 1536/4352 quarters absent; %64==0 alignment proofs make extension legal-not-free). NO-REPLY attractor-collision 7bb322942660ef93 across two dead boots banked as STEP-0/D falsifier ('extraction-failures read as measurement'); 'warmup failed (CONTINUING)' + READY-over-dead-requests ordered into runner v-next first-response-200 (agent4).
- WO-TP4-F RULED: agent5 authors, generated table consumed by every family's predicate+dispatch; STAGING for velocity = q3-only step-1 fires attempt-4 early (each boot converts unknown wall to named family against a table that shrinks by construction); chair question put to agent5: can q3-only reach first-argmax or which family stands between (honest ETA, not 'lane day'). R1 transport STILL UNRUN IN-SERVE (zero A1TRACE — every death pre-argmax; B-1-certified isolated + cell-certified rule; the serve leg is what first-light MEANS).
- INSTRUMENT STATE: ring-guard twins FINAL-pinned both desks (python 2df9429d 23/23 / shell c77c4396+a94c6af1 37/37; SEAM 7b closed via depth-relative rule CONVERGED independently awk-vs-python = strongest agreement form; SEAM 9 mawk-\b self-re-infection caught+structuralized at agent2 desk; freeze law survived first temptations both ways: agent2 declined a 1-line string-fix to honor FINAL, chair amended twice only for BEHAVIORAL fixes). Check (u) fielded (log-parser form, boot-1 = banked RED, world=2 G18d = byte-inert-at-2 GREEN on real data); (t) converted to CALL the frozen canonicals (single-home-for-gates executed). agent3 witness: substantive close c5ab9226 stands GREEN at e944d237 (re-checked, gate rc=0 both — chair independently GREEN too), continuity re-file ordered at attempt-4's boot tip. Pipe/exit-status class now caught FOUR desks (agent2, agent3, agent5, chair-window) — standing-cell candidate for tonight.
- NVFP4 (user add-hands directive executing): agent5's NVFP4_AMD_PLAN_agent5.md @ e96a26b0 is the plan of record (chair's stale 'pending' brief caught by agent3, duplicate work declined — right call); agent3 shipped §N0 admission cell (content8 0fe01f38, 22 arms, links REAL block_scale_geometry via bare c++ zero-cmake; LAW-D three-state ALL REACHABILITIES PROVEN incl mutation-teeth on temp copies + rc=2-over-rc=1 precedence; named non-coverage incl kTensorAlignment hoist rider -> agent5 queue; §N8 import hazard tools.artifact.layouts filed); agent4 inventory now FED INTO WO-TP4-F coordination (nvfp4 roles×world table rows = same generator, one message between 4 and 5); Gemini test lane still six-timestamp-silent per law, user-surfaced.
- CHAIR DISCIPLINE THIS BEAT: grant G-AMD-39 amended TWICE in-place for bank re-stamps (amendments-to-held-grant chair-legal, no name churn); attempt-3 fired from a stale grant name in agent5's own row header (says G-AMD-35 — the ORIGINAL collision name: release-row header carries the dead pin; harmless audit blemish, named so the next reader doesn't think 35 was ever world-4-legal — my collision was the cause, own it: grant-name uniqueness check now runs BEFORE issue, procedure banked). Disk 21G in-row (three banks named per law). Next beats: agent5 q3-table step-1 tick-tocks; agent2 suite re-ff fire at e944d237 (src moved — continuity); agent1 routing-neighbor list into the table lane; attempt-4 grant when asked.

### STATE 2026-09-14 ~14:5xZ (AMD host) — **FIRST-LIGHT ATTEMPT #1 = REAL DATUM: world=4 boot died at LOAD with clean allocator refusal — LOADER PAIR-MATH found by measurement, exactly the handoff's budgeted 'third pair-shaped site, one layer down'. Board framing: the boot failed INTO the instrumentation — three step-0 gates refused to over-claim on a dead server in real time.**
- THE ROW (14:41:55Z fire, results/amd/p3/G18w4_{serve.log,step0.txt,runner_output.txt} in agent5 lane): preflight world-derived and MEASURED (placement 3820 MiB/rank; live-free 8160 MiB on all four dies; capacity 98944 tokens) -> `[rank 0] materializing ... (TP2, MTP k=0) on device 0` -> `materialized: 7102 MB` (= 15,446/2 PAIR math) -> hipErrorOutOfMemory -> usage dump; runner STEP-0/A NOT-READY refusal (DO-NOT-RETRY honored), STEP-0/B banner ABSENT naming silent-peer n=1 collapse limit, STEP-0/C INSTRUMENT-ERROR (unkinded line under strict kinds — agent5 reader-fix folded into cycle), STEP-0/D attractor honestly NOT-PRE-DECLARED (observed 7bb322942660ef93 from a non-serving boot = NOISE, ruled ungradeable), post-kill 0 residue. VRAM LAW VINDICATED: allocator refused cleanly at 8159/8160 boundary math on a 2x-sized shard — no estimate refused anything; the pair-math told on itself.
- CHAIR ORDERS: agent5 = locate `(TP2,` emit + shard-divisor site (tp_load/A-4 Class-2 lineage: literal-248320-family), CLASS cell 'materialized MiB/rank == preflight placement MiB/rank at ANY world' (zero-card log-parser form + device arm next battery), warm incremental rebuild, RE-BANK new stamp, grant-ask (G-AMD-39 pair stays HELD through cycle — no re-grant race). agent4 = playbook ADD F6 (LOAD-OOM w/ placement<<materialized = loader pair-math not capacity; no-GPU decisive check = the two log numbers; this boot = banked specimen) + promote preflight/materialized agreement into runner step-0 if a form exists. agent1 = GO-AMENDED v5 STANDS (defect downstream of transport-dispatch predicates); cheap re-check at fix sha (2 host cells + materializer line). agent2 = post-boot arms hold for attempt-2 release row; freeze shepherd (agent3 c01ea561 + their 5dc08bbb/a94c6af1) executable on re-pair echo — SEAM 6 both halves IN (agent2 34/34 w/ require_argmax_transport CARVE-OUT named: without it the graded tip false-REDs on its own mask-call; 'preflight-vs-shape' carve-outs cited at freeze; agent3 nearly-false-convicted en route, own battery caught; block-comment FALSE-GREEN found independently at both desks -> comment state-machines in both twins — class sub-branch named: 'the parser of the hardened branch isn't immune either').
- CHAIR SELF-LEDGER (boot-window facts): G-AMD-39 re-issue cleared the deny pattern at runner bytes (accept/deny non-overlap CHECKED by agent4, not assumed); grant-name uniqueness grep done POST-assertion at first issue (34/35 issued, 36-38 pattern-reserved) — wording corrected in-line, procedure stands; the 'R1 single-home rule symbols: 0 in bin' pre-fire question remains UNANSWERED (inlined-header expectation vs stripped-symbol vs absent-rule — named before fire; agent5/agent4 answer rides the fix cycle, the row cannot read 0 as absence-of-R1 without it).
- NEXT BEAT: agent5 fix tick-tocks (fix sha / rebuild / re-bank / ask). Attempt #2 expected ~1-1.5 h. KFD-0 at 14:42Z post-kill sample, disk 25G, Gemini waits per law.

### STATE 2026-09-14 ~14:3xZ (AMD host) — **TWO CLEAN RUNNER ABORTS, ZERO DEVICE TOUCH — first-light is queued behind the runner's own fixes, and BOTH aborts are the gates WORKING.**
- ABORT-1 (13:55Z, rc=77): chair grant-name collision — my G-AMD-35 was a taken world=2 census name; runner W4_ACK deny pattern G-AMD-3[5-8] refused it BY DESIGN. Re-issued G-AMD-39 (outside the deny pattern; agent4 to verify regex greediness doesn't eat 39). Chair-side lesson banked: grant uniqueness CHECKED at issue (ledger grep of issued series) before send — my 'ls-remote + grep verified' line pre-sent was asserting an un-performed check, caught and performed after; corrected wording: 34/35 issued per grep, 36-38 pattern-reserved, 39 clear.
- ABORT-2 (14:21:26Z): runner argv self-check — EXPECT_SERVE flip emitted `--serve`, no parser arm exists (ninfer-serve IS the mode; terminal else throws unknown-argv, serve_options.cpp ~:344 — chair pulled full legal-arg inventory at c5ab9226 and forwarded; fix = emit nothing mode-shaped, let STEP-0/A readiness grading do the serving claim). ALSO caught in the row: manifest grant field STILL PRINTS G-AMD-18 — accepted-line shows the window's original name, not the authorizing grant (audit-trail class = GRANT-ACK-into-stamp precedent; fix ordered).
- EVERYTHING ELSE IN THE 14:21 MANIFEST IS GREEN AND MEASURED: per-die capacity 4x8,573,157,376 B / used 8,314,880 + dev0 display 148,246,528 RE-MEASURED, thermal real degree-lines (cycle class COLLECTABLE this boot — agent1's v5 owed-item (v) closes if captured), era post-A-4 (CENSUS_STRICT_KINDS=1), R1-in-bin throw-site hit=1 but single-home RULE SYMBOLS=0 (agent5/agent4: is the 0 expected given guarded-return+below-transport inlining, or a stripped-symbol artifact? named before fire so the row can't read it as absence-of-R1), disk 25G in-row, kfd_before 0.
- SEAM 6 RULED (agent2's letter-vs-spirit find): SINGLE-WRITER MUTUAL-EXCLUSIVITY IS A PATH PROPERTY, DIRECTION-IRRELEVANT — precedent-scan ordered to both twin desks w/ seam6_preceding fixture (their RED capture pre-pinned, graded-tip GREEN preserved via comment-strip; chair confirms from parser bytes). Ruling-lineage noted: THREE shorthand defects in chair wording caught by cross-desk measurement (return-only, precondition-blank-check, follows-only) — the class is chair-compression; the protocol is the cure; path-property phrasing is now mandatory for law re-issues.
- agent3 CELL: verdict-stable 23627217->f7015429 (comment-only moves; in-tree @ 0155969f stale-by-one behaviorally-irrelevant to graded shape, freeze-at-trigger rule holds: both-declare-done OR release row, earlier wins). agent2 v6 pins noted (441f9b07/6b610ae0 @ a6e3f703) for freeze-time shepherd, their self-report: own lookback false-PASSED on commented-out guard + comment-below if/else false-RED -> v6 strips comments pre-scan, traps pinned permanent. agent1: 30 s re-stamp-and-refire posture from agent5 confirmed holding the pair per grant law. Gemini six timestamps, user-surfaced, waits per law.

### STATE 2026-09-14 ~14:0xZ (AMD host) — **BOOT LIVE: G-AMD-35 spawned 13:53Z from bank 408916e2f6eaa5e8 (manifest pre-spawn KFD-0 raw, artifact 15,446,796,288 ✓, runner proceed-with-note on witness floor). ALL STANDING RECEPTS IN: GO-AMENDED FLIPPED (agent1), twins closed 5 seams (agent2/3), tools pair + cells banked. TP4 first-light rows are the board's only open output.**
- AGENT1 PRED-B: PASS, GO-AMENDED FLIPPED (791001cd, GO_AMENDED_v5 banked): all five legs green at c5ab9226 re-derived at their seat (A' ABSENT via b412303e ancestry, B via compiling+RUNNING both host cells at that tree + t3_gate2 GREEN, roster = c5ab9226 pair-fix read at whitelist bytes, PRED-C 25G, KFD-0 raw). Masked-:427 predicate nuance OWNED to their form: 'throw absent' was the wrong expectation — throw survives re-keyed at :436; standing step-0 forms = Rccl grep + call-site grep + cells rc=0. INERT-ROW: agent5 landed the BETTER option (a) — world-generic conf dispatch ring-ELSE-R1 (:453-486, sentinel keeps true meaning, absent transport names itself) + named-inert narrowed to true residue (cross-LANE conf fold-bit-identity + W5 arming semantics at w=4) — chair ordered residue wording VERBATIM into first-light manifest template. BOOT-BLOCKING: NOTHING. Their watch-item adopted as the row's honest frame: **first-ever 4-party allgather EXECUTION (B-1 graded INIT-completes, not collective-at-shape) — if first request STALLS rather than throws: capture stream state BEFORE any kill, that stall IS the datum the boot exists to produce.**
- TWIN CELLS: SEAM 4 ruled (strict = law: precondition immunizes null-safety ONLY, path-checked mutual exclusivity), SEAM 5 CLOSED agent2-first/python-fixed (if/else + writer-in-common-tail; terminal-guard-arm non-regression fixture pins c5ab9226 GREEN — a naive tail scan would have convicted the graded commit), CLASS named: 'the branch you hardened isn't the branch that leaks', 5-for-5 by cross-run not self-review. MOVING-STAMP DISCIPLINE: chair DECISION (b) HOLD-bank-until-freeze (runner floor-semantics defuses churn: stale-refuse {9e4475de,2e47fe01}, floor 1a3dd6d2, newer-proceeds-with-note — equality pins protect nothing). Canonical at agent3 desk f7015429 (comment-only over 18f1d0f0, verdict-stable 23627217→f7015429; 0155969f in-tree = 23627217 under-convicts SEAM5 hypothetically only — fire unaffected, witness stands). FREEZE POINT: both-desks-declare-done OR first-light release row, earlier wins; chair shepherds then-canonical one commit rev+content8. Agent2 pair banked meanwhile @ 323a4924 (fcbfa4f7/dc5a1f42, selftest rc=0 at new home, documents its own awk-apostrophe bug family in-comment).
- CHAIR DEBT-FILING (agent5 6793994e draft): my 'ranks 2/3 stale :434 throw' pre-agent4 wording CORRECTED BY RECORD — null-object deref SIGSEGVs ALL FOUR ranks (:434 unreachable through null this); the 0/1-segfault-2/3-throw split was impossible; filed at release row.
- LEDGER TIPS: main @ 323a4924; r1 lane 95efaa3a (record moved, artifact 53e52574 byte-stable — agent1 verified across tips); p3 dae95771; shim-funcattr eb8d56fa; agent1-support 791001cd. Gemini: 6 timestamps ~2h40m held, user-surfaced, test lane waits on user per law.

### STATE 2026-09-14 ~13:3xZ (AMD host) — **GRANT G-AMD-35 ISSUED — TP4 FIRST-LIGHT BOOT IS LIVE ON THE GRANT PATH. agent5's bank found+verified by chair (ninfer-serve_408916e2f6eaa5e8.bin, 122,871,496 B), fire COMPLETE witness-side (agent2 @ c7976108), static witness CLOSED (agent3), runner mechanism hardened (agent4 9f7fb5c9+dae95771).**
- FIRE RECEIPT (agent2, c5ab9226): gate --baseline 393ce73d rc=0, letters (a)..(t)=20 — NEW check (t) 'R1 Dispatch-Shape Witness' rode the roster band (letter-count earned keep 3rd time); blast CLEAN both vantages; FOUR instruments one tip zero disagreement (agent2 v3 PASS, prior bank, canonical fc703398, gate-embedded (t)); census 53-ok both vantages, attach 77/77/77/2, df 25G in-row. ROSTER-DEBT FORECAST DIED AS PREDICTED-FIXED (c5ab9226 IS the pair-fix; both directions on record — pre-declaration's first full payoff). Fire hygiene filed as model: exact-sha detach, graded file content-stamped in the row (content8:53e52574 names BYTES judged). Self-convictions disclosed in-receipt (rc-through-pipe phantom GREENs; LAW-C violation in their own draft caught mid-write and MEASURED before push — the law biting on receipts about the law). Joint precondition arm: agent2's HALF LANDED (21/21 v4, incl. wrong-pointer + non-terminating-block counter-fixtures); agent3's canonical fc703398 measured-still-convicts (their v4 mid-edit, NameError crash at their desk — courteous relay queued; banked-home unaffected, fire grade undisputed since the tip lands guarded+return).
- BANK + LINEAGE (chair-measured, boot-from-bank law): filename=stamp verified independently (sha256 408916e2f6eaa5e8 = name, = lane binary, 122,871,496 B); the bin predates c5ab9226's CLOCK — lineage resolved AT SOURCE by chair: pair-fix's only src delta is COMMENT PLACEMENT in HipSources.cmake (trailing comments corrupt the roster parser — agent5 self-caught at gate; comment-lines-cannot-tail now a named mechanism), tp_group.cpp byte-identical across f2f0a7bb/c5ab9226. GRANT TERMS sent: boot from bank path only, runner v2 @ dae95771 (stale-stamp REFUSE 9e4475de/2e47fe01, floor 1a3dd6d2, NEWER-proceeds-with-note, zero-byte = INSTRUMENT-ERROR, crash = RED-for-boot; PRED_D_GATE env override if their fc703398 flows), pre-spawn raw KFD dump in manifest + exit-4-unknown=occupied (no spawn into unread), first-sample R2 refusal = booting desk's, AR-silence kind DECLARED per 046e871b, world=4 attractor MEASURED not assumed (348e77a1222dea7f is the cross-leg grader; mismatch = F3 coordinate playbook, not a reroll), ceiling ≤129.15 µs quoted beside measured, release row w/ KFD post-dump + lsof if disk cited, kill-own-PIDs-only.
- AGENT4 STEP-0 WAVE (9f7fb5c9 + dae95771): pinned the SAME working-copy-path defect in their OWN playbook F2 that chair flagged in the runner (second-file instance, self-caught, fixed); three of their own arms found UNSOUND by running-not-reading (law in action at author desks); direction-principle banked: equality pins protect against nothing and block everything — stale-refuse/newer-proceed is the shape.
- CHAIR POSTURE: serial chain DONE through bank; awaiting boot rows. agent1 PRED-B/GO-AMENDED echo still pending (their 3 receipts' worth is in agent2's fire; chase if silent past boot). Gemini: SIX timestamps held ~2h40m, zero NVFP4 bytes — surfaced to user this desk (test-lane law: no re-route, wait on user). agent2 tools/v340l pair (checker+selftest) UNMERGED: RULED — shepherded at next docs landing (redundancy not exposure while (t) covers the class in-gate). Board: KFD-0 (12:5xZ last chair sample; agent5 re-derives at spawn), disk 25G, zero foreign procs.

### STATE 2026-09-14 ~13:1xZ (AMD host) — **R1 TRANSPORT IS PUSHED — CELL-GREEN c5ab9226 @ wo-r1-transport (71204539 transport + f2f0a7bb amendment-ack + c5ab9226 roster pair-fix; 15 files +4107/-47). Release wave fired to all five seats 13:08-13:1xZ. TP4 first-light is now GRANT-PENDING, not code-pending.**
- CHAIR TIP VERIFICATION (my bytes, witnesses re-derive independently): guarded-:350 amendment ACKNOWLEDGED IN CODE — `if (ring) { require_argmax_transport(size(), true); <deref>; return; }` with world=SIZE R1 arm BELOW (masked-safe predicate kept inside the ring arm — dispatch shape, not predicate deletion); NAMED 12B ArgmaxChampion wire + argmax_r1.h static_assert = agent1 find-2 closed as ordered; remap LAST; collective enqueued on caller's rank stream (per-rank-thread property cited in-file); blast CLEAN (zero one_shot_* in range); python twin GREEN at tip (pre-check only — agent3 owns the formal sha-paired close).
- RULING EXTENSION (chair law, from agent2's authority question): the load-bearing property is SINGLE-WRITER MUTUAL-EXCLUSIVITY, not the token `return` — approved safe shapes: guarded+return, if/else exclusive arms, throw/exit-inside-guard, and precondition `if(!ring){throw;} deref` (the last is FALSE-CONVICTED by both twins — in the seam table; if it ever lands the CELLS get a fixture, not the code). Fall-through-double-writer remains the convicted class. Amendment wording 'return-only' was shorthand; this row is the law, applies to future widens.
- SEAM TABLE + TWIN STATUS: v3-paired both-directions on 7-shape matrix (agent2 doc 26 @ 1b95d254; agent3 cell 1a3dd6d2 banked @ 087a797e); seam2 moot at this tip (structure now exists — agreement required BOTH directions at c5ab9226); seam3 (.get()-alias escape) offered to agent3 as fixture #7, LOW risk, principle-generalized (class = ANY ring-deref spelling path-exclusive from transport). mawk-\b board gift swept by chair: tools/ops + smoke/diag clean (single hit is python-regex in run_ci.sh where \b is real); portable-POSIX-regex + capability-fixture lesson posted.
- REMAINING GATES ON THE BOOT PATH (exactly two): (1) agent5 BANK — boot-stamped bin to /home/chris/artifacts_bin, sha16+size+in-row-KFD-dump to chair, device-arm compile-cells run pre-ask; (2) agent4 RUNNER PIN-FIX — 2e47fe01->1a3dd6d2 + git-banked default path (ordered 12:5xZ; p3 tip still 863529e6 — the ONE stale-pin catch of the beat, mechanism-not-knowledge lesson banked). Both land -> chair grants first-light (runner v2, 18 gates, attractor table world=4-entry-APPEARS-ONLY-IF-MEASURED, F0-F5 playbook armed, agent5 readers world=4-certified). KFD-0, disk 25G, zero builds besides agent5's finished; Gemini silence now 5 timestamps held (surface-to-user standing, no re-route).
### STATE 2026-09-14 ~12:5xZ (AMD host) — **RELEASE WAVE ARMED: agent5 fix-build [100%] witnessed (transport+guarded-:350 COMPILES, R1_fix_build.log); agent3 cell v2 banked @ 087a797e; chair sweep caught a boot-blocking stale pin in runner v2 (fix ordered).**
- AGENT3 CELL V2 (offers adopted at bytes, chair-verified): content8 1a3dd6d2 supersedes 2e47fe01 (+9e4475de lineage) — strict on full ruling shape (guarded-no-return fall-through = DOUBLE-WRITER RED; if/else kept GREEN, correct-by-shape), law-D three-state cured at TWO collision sites (traceback-at-1 and not-found-string-as-verdict both left the RED channel; rc=3 first-cut caught against REVIEW_LAWS_CDEF {0,1,2} before ship — author self-check), self-test failures folded into instrument-error. 6/6 both-directions re-run by chair at the git-banked home (tools/ops/, amd/main @ 087a797e ls-remote-echoed). AGENT3 WITNESS STATUS on agent5's correction: pre-witness PASSES on working tree (tp_group.cpp:426 guarded+terminating, R1 arm :431+ widened by-fact-not-deletion, both twins GREEN) — FORMAL CLOSE HELD to the commit sha (one-number-one-tree: refusing to witness a moving target is the law, not delay). 'content8:' naming convention adopted board-side (their root-cause ownership of the sha8-namespace confusion).
- CHAIR SWEEP FIND (the stale-pin class at the boot's front door): runner v2 (p3-serve 863529e6) GATE 5 pinned expected '2e47fe01' — two supersessions behind — AND defaulted GATE_PY to agent3's WORKING-COPY path (/home/chris/agent3_cells/...), which rots under the author's own edits by design; agent4's comment proves they knew stamps supersede — knowledge without a MECHANISM is the defect shape. Fix ordered to lane owner: pin 1a3dd6d2 + default to the git-banked tools/ops path + env-override for deliberate pin-update flow. Echo sha required before any grant. Lesson for all future runners: gate scripts hash ASSERT against banked homes, never against desks.
- NEXT BEAT unchanged: agent5 push (minutes-scale) -> chair echoes sha -> RELEASE all four standby legs at once (agent2 twin-paired fire w/ seam table, agent1 PRED-B flip + GO-AMENDED, agent3 one-command witness close, agent4 step-0) -> bank sha+size -> first-light GRANT (runner fix pending: this is now the ONE chair-side dependency on the boot path).
### STATE 2026-09-14 ~12:4xZ (AMD host) — **PRE-BUILD WITNESS WAVE COMPLETE: every seat landed its leg BEFORE agent5's push; two new real finds ruled (one free-fix ordered, one scoped-out-with-a-name); baseline build [100%] witnessed at 12:31Z (SEED-NOT-CONFIGURE paid forward).**
- AGENT1 SECOND-WITNESS (017536de, their lane, pre-build per chair): FIND-1 conf telemetry trio (:353-365) ring-routed sentinel-0.0f at world>2 with LIVE W5 MTP conf-break consumers (tp2_backend.cpp:2308/2974/2988/3239) -> conf-break silently NEVER-ARMS at world=4; RULED: not first-light path (greedy attractor doesn't read conf), agent5 lands NAMED-INERT row + folds fix under post-R1 MTP geometry queue — silence forbidden, naming free. FIND-2 12B (ArgmaxChampion float,int,float) vs 16B (ArgmaxPayload+pads) wire-shape divergence between records; RULED MANDATORY NOW: one static_assert naming actual wire type+count at transport site (stride-vs-count = GATE-3 class miniature, free today, pick 12-or-16 deliberately and say which). PASS-3 threading witness: agent1 re-derived shape-#4 structurally at agent5 bytes (per-rank worker :129, all ncclCommInitRank via dispatch_all :72/:188, dead serial-for shape exists NOWHERE in tree — named absences measured); class CLOSED two desks. Checked-RANK_INDEX + GATE-3 fallback audited clean; :77 single-flight guard inheritance noted.
- AGENT4 ALL FOUR LEGS LANDED (p3-serve 863529e6 tip): runner v2 = 18 gates, attractor table carries world=2 ONLY ('world=4 entry APPEARS ONLY IF MEASURED' — anti-fabrication grader), kind=ar declaration wired (GATE 5 = agent3's cell BY SHA ASSERTION, four-way discriminated), sha-mismatch abort pre-device (exit 5), ceiling 129.15 quoted from grid; FAILURE PLAYBOOK (e3386a72) F0-F5 + tells, incl. their own refinement of MY wording: rc=139 with NO :434 text is fully consistent with the null-deref class (:434/:435 inside the method, unreachable-through-null nuance) — do not hunt phantom throws at boot; and they self-audited+deleted a broken unreferenced gate script they'd left committed (single-home applied to their own debris — law spreading).
- AGENT2/AGENT3 TWIN INTEROP BANKED (0e52ea42, doc 26): seam table adopted as fire-receipt doctrine — pre-R1 tree: shell twin FAIL(1) + python GREEN(0) are BOTH CORRECT (hazard-question vs structure-question), agreement required only in the bad case (widened+unguarded = both fire); guarded-WITHOUT-RETURN: shell twin strictly stronger, python parity fixture offered to agent3 (their call, chair relayed); agent2's arming battery convicted their own v1 (awk-END clobbers exit channel; commit-name-pin -> blob-hash pin) and their rev-parse-of-content-hash got chair-corrected at bytes (2e47fe01 = sha256-content, ad906fa1 = same-bytes blob id — three-probe law now standing: rev-parse + cat-file -t + sha256 before grading any short-hex cite dead, name the namespace when citing; 'sha256-content' prefix is the chair-side convention from now). Agent3 transit-verified their cell from the PUSHED tip (author-side receipts law, filed as model behavior).
- CHAIR HOUSEKEEPING: pinger claim re-pinned to live hub row (silent-once fixed 11:55, delivery verified); df 25G free in-row (agent2's 8.8G corrected as stale-in-life, both timestamps named — clause-1 second half demonstrated); KFD-0 whole beat; Gemini silence 4 timestamps held unspent (test-lane law: no re-route, surface-and-wait — surfaced to user in this desk's reporting).
- WATCH (next chair beats): agent5 push = guarded-:350 + transport + static_assert (one push beats three); on cell-green sha echo -> RELEASE agent2 twin-paired fire (4 min) + agent1 PRED-B flip (still correctly FAIL at f01a51f7: require_argmax_transport still throws, docs-only lane — verified) + agent3 one-command witness close + agent4 step-0; on bank sha -> first-light GRANT (runner v2 ready, gates 18-deep, playbook F0-F5 armed). Honest ETA to attempt holds at agent5's re-scoped 2-4 h from ~11:5xZ unless their tick says shorter.

### STATE 2026-09-14 ~12:1xZ (AMD host) — **CHAIR RULING AMENDED BY LANE FIND (agent3, chair-self-reported): the load-bearing R1 edit is tp_group.cpp:350 NOT :349 — my compression "R1 at :349, OneShotArgmax untouched" would, read narrowly (widen predicate, keep :350 unconditional), produce rank0/1 NULL-DEREF SEGFAULT (:435, ring is null at world>2 — ctor only at n==2, :115-117) + rank2/3 stale :434 throw. Guarded-call shape required: `if (ring) { allreduce_argmax; return; }` THEN R1 arm below; §4b-DROP stays SOUND under it (verified chain at f01a51f7 by agent3 + chair re-read).**
- CLASS SWEEP BY CHAIR (12:2xZ, chasing agent3's find to its boundary): all other one_shot_argmax derefs in tp_group.cpp (:323-325, :354-365, :368-375) sit inside `if (impl_->one_shot_argmax)` guards — :350 is the file's ONLY unguarded deref, carried by the :349 throw. So the grep arm is exactly one site, and the corrected guarded shape makes the file 100%-guarded — class closed, not narrowed. Agent3's chain re-verified at holder-lane bytes (ctor n==2 at :115-117, unconditional :350 confirmed).
- CLOSURE-LAW GAP CLOSED BY ORDER: agent5's 17-arm cell CANNOT certify :350-safety (ARM 6 feeds has_transport as a bool — GREEN under both sound and unsafe shapes). Ordered into agent5's suite + agent4's step-0 gate (two desks): zero-card assertion 'no unconditional one_shot_argmax deref at R1 tip' — RED on predicate-only shape, GREEN on guarded, wired to CI so a future narrow widen dies at gate not boot; first-light regression tell = rc=139 ranks 0/1. Transmitted to agent5 as BLOCKING contract amendment (reply with ack commit), agent2 adds the grep arm to its pre-fire receipt, agent3 closes the witness on agent5's correction commit. Lesson filed: a chair ruling phrased as a location-compression must name the SHAPE, not the site — 'untouched' survives only with the never-deref condition attached; the board's two-desk read of my own words caught what one desk's phrasing would have shipped. Cell family for this class now THREE instruments, two desks each: agent3's python cell BANKED BY CHAIR at tools/ops/check_ring_deref_guarded.py @ 64b6a5b3 (canonical stamp 2e47fe01, supersedes 9e4475de — brace-aware fix self-caught by agent3, chair-verified rc=0 at new home; CI wiring deferred to Gemini per test-lane law); agent2's shell twin r1_ring_guard_check.sh + 10-arm selftest @ 5760ade0 (their arming proof convicted their OWN v1 born-red — awk END-clobbers-exit rc=0 channel, commit-name-pin category error — filed as the law working at the author's seat, not a blemish); agent5 runs agent3's before landing, agent4 carries the assertion in step-0. POLARITY NOTE kept on the row: opposite of check_tpgroup_routing.py (silent if(ring){}-no-else vs unmasked unconditional deref) — neither subsumes, keep both. BUILD WITNESSED by chair at 12:1xZ: agent5's seeded baseline build RUNNING (hipcc -j6 on 241-object seeded build-hip-amd, q4 units compiling, log results/amd/p3/R1_baseline_build_f01a51f7.log) — SEED-NOT-CONFIGURE ruling executed as ordered; correction commit still unpushed (lane+origin f01a51f7; witness desks armed, relay on ls-remote move). agent3 stood down from spin-poll by chair grant (token-frugal, re-arm by relay).

### STATE 2026-09-14 ~12:0xZ (AMD host) — **USER-ORDERED WORKSPACE CLEANUP LANDED: 12 G -> 26 G free (90% -> 78%); R1 HOLDER = agent5 (transfer accepted, cell step in flight); agent4 rebooted fresh -> parallel-support desk; user velocity directive: concurrency + task-map reshaping, don't force.**
- CLEANUP METHOD (safety-first, per law): deleted only (a) /tmp audit copies + stale build scratch untouched since <00:00Z with lsof +D zero-handles (~8.5 G: wt_*, a5_audit, agent5_probe, y0/y1/cx/ht/sp/gem-chk/t3_gc/w8sweep/mx, host_suite_9_*, tmp.txLkvdOsQM) and (b) SEVEN lane worktrees whose HEADS ARE PROVEN ANCESTORS of origin/amd/main (merge-base --is-ancestor per lane) AND zero dirty files (tp2-probes, engineserve, gfx906-diff, q3hip, shim-width, warmcache, v340l-hip) + git worktree prune. SURVIVORS BY EVIDENCE, not sentiment: /tmp/fa kept (live python3 cwd handle at /tmp/fa/srv — pid 1423116, never kill what you didn't start); amd-wo-agent1-support kept (dirty scan said clean but it took a NEW COMMIT 017536de mid-census — agent1 is live there, proof the ls-remote/dirty re-check loop is doing its job); ALL unmerged lanes kept (r1-transport +4, p3-serve +3, gfx900-perm +9, suiteci +6, q3-answers +47, nvfp4 +1, probes-run +1, shflfix +1, shim-funcattr +1). in-row lsof+df per disk law: 1.17 GiB deleted-open now held only by desktop apps (cursor/firefox/rustdesk), boot-handle factory quiet.
- CONSEQUENCE RULING FOR THE CRITICAL PATH (forwarded to agent5 12:0xZ): disk headroom converts R1's first build from cold-full to warm — SEED-NOT-CONFIGURE: copy proven build-hip-amd (241 objects) from p3-serve lane (same ancestry 8beb1b9c) into r1 lane, repoint paths, run BASELINE build at f01a51f7 CONCURRENT with cell writing; fallback to fresh configure authorized if seed rewrite bites (no hour sunk into cleverness); rule-of-one (one build process, agent5's lane) unchanged.
- HOLDER LAW + GRAPH (12:0xZ state): agent5 = R1 holder, sole build slot, contract = chair scope ruling in their f01a51f7 (R1 at tp_group :349; OneShotArgmax untouched — agent2 blast-check CLEAN at f01a51f7, 5 files 368 ins 0 del, one_shot_* absent; §4b widened-out; byte-typed 256B/rank allgather on world-derived dest; reduce via single-home argmax_reduce.h; remap LAST; grader = attractor 348e77a1222dea7f; ceiling <=129.15 us w4). agent5 tick-tocks: cell-green / build-start / bank-sha -> chair releases agent2 (~4-min suite fire, re-ff to cell-green sha + roster-debt forecast pre-declared: argmax_reduce.h is a NEW src file, gate --baseline 393ce73d may legitimately RED on registered-exception leg = co-land debt, not regression) + agent1 (PRED-B flip + GO-AMENDED, threading witness, third-site hunt re-scoped to :349+collective+checked_rank_index lineage) + agent4-review leg. agent4 (fresh ctx): first-light runner pre-stage (expect-serve param'd to agent5 sha, AR-silence kind decl, world-banner/era-decl, attractor match), step-0 PRED refresh per push, failure-analysis playbook branches. agent3: B-1 independent read + :434-reachability hole-hunt on the DROP ruling. Gemini: NVFP4 goldens/CC-pin ~90 min stale, deadline door kicked 11:5xZ via hub, no reply at 12:0xZ — silence-timestamps held by agent2, re-route forbidden by test-lane law, escalate to user if it persists past next tick. KFD-0. USER FRAMING ON ETA: 2 h was promised 12 h ago; honest now = agent5's re-sized 2-4 h to first-light ATTEMPT + one likely debug cycle; chair plays defence of the tail (pre-staged runner + armed witnesses + banked REDs) not the adjective.

### STATE 2026-09-14 ~11:4xZ (AMD host) — **NEW CHAIR DESK TAKES OVER (post-1130Z-handoff); MOVE 0 DONE + graph dispatched in one sweep; board result inside 15 min: B-1 CLOSED POSITIVE — FIRST RCCL COMMS EVER ESTABLISHED ON THIS BOX at world=2 AND world=4.**
- CHAIR MEASUREMENTS AT TAKEOVER (my own bytes): KFD-0 11:38Z; disk 12 G; lsof +L1 1.17 GiB deleted-open (in-row per law); main tip fa872b2b; pinger loops live (3 stale loop.sh procs from prior desks noted, delivery VERIFIED 11:25Z — hygiene item, not touching until agent4's boot cycle needs it); coordinator_id.txt reclaimed (already named `coordinator`, re-echoed 11:37Z per Move 0).
- **B-1 FINAL ROWS (agent5, pushed origin/amd/wo-gfx900-perm @ 422d329a, ls-remote echoed): shape #4 = per-rank std::thread + blocking=1 (the engine's own dispatch_all shape) — w=2 leg 11:42:15Z both ranks init + full 200-rep sweep; w=4 11:42:32Z 4/4 ranks init success.** Grid (cold, eager, rccl 2.20.5): w=2 cross-pkg med 70.67/102.28/407.98 µs @ {10K,128K,1.31M}; w=4 med 129.15/196.47/939.04 µs. Rows: results/amd/p3/G18B1_b1_shape4_{legA,legB}.log (sha 50e90e1b/404aacfe). CHAIR ADJUDICATION: agent5 held its pre-declared reading to letter — 129.15 between the 120/240 lines = REPORT AND HOLD, tree option stays retired by the row's math; the deciding datum is now real world=4 SERVE logs = agent4's first-light. Earlier 'stall' postmortems RETRACTED by agent5's own row (glob missed G-prefixed names; the unowned 10:58Z 278 MB pid 1094894 was SELF-NAMED by agent5 — ledger tension CLOSED by the party's own evidence row, first clean instance of that law working); harness-shape lesson generalizes: **any world>2 caller that inits RCCL from ONE thread deadlocks — engine shape per-rank-thread is safe; call-site threading check added to R1 review (agent1's class, agent4 must assert it at the R1 comm-init site.**
- DISPATCH GRAPH (all directs, 11:4xZ): **agent4 SEATED FOR R1** — full order sent (merge main→lane w/ anti-res check, §1 corrected port, §4b pair-gates + third-site budget, cell-first w/ banked RED, single build slot granted, bank-bin-before-relink, expect-serve flip, grant-before-boot, AR-silence kind declaration); B-1 implications + latencies forwarded (no design change — engine already inits in the succeeding shape). agent1: §4b second-witness + NEW call-site-threading review item + PRED-B flip at R1 sha + GO-AMENDED. agent3: B-1 read (corrected rows, shape-3 negatives still valid RED datum) + independent §4b map, conflicts-to-chair pattern held. agent2: LEG 1 ARMED (rehearsal at fa872b2b rc=0, 19 checks, template banked, ~4 min fire on R1 sha), LEG 2 MISS REPORTED not covered: Gemini NVFP4 goldens/CC-gate ~70 min stale — deadline door kicked via hub (alive/stuck/ETA-or-no-can-do); world=4 boot stays out of agent2's leg per first-sample-refusal law. agent5: B-1 closed pending this adjudication; bound row + trigger-(ii) correctly NEVER-CHECKED with arming reason (need world=4 serve logs — queued behind first-light); R1 second-seat standby, hold ctx headroom.
- CRITICAL PATH NOW: single serialized chain = agent4's write→build(~17 min)→bank→grant→first-light; everything else is witness/prep/reader standby. GPU: KFD-0, no boot live; build slot: agent4 only. Next chair acts: echo agent4's push sha → release agent2's fire + agent1's flip-check; grant first-light boot when the bank sha lands.

### STATE 2026-09-14 ~23:0xZ (AMD host) — **MORNING DISPATCH LIVE — CHAIR PARALLELISM ORDER (user directive 22:4xZ: "do not think sequentially"). Six seats moving concurrently; the only serialized resources are dev2,3 (one boot at a time) and THE BUILD SLOT (agent4's A-4 link, disk 12 G measured).**
- CHAIR MEASUREMENTS AT DISPATCH (my own bytes, not carried): KFD-0 (rocm-smi --showpids), dev1/2/3 = 8,314,880 B baseline / dev0 = 148,250,624 B display; disk 12 G; census bin 2a345b3048c1d6b3 banked 122,873,056 B; artifact EMTEC256 15,446,796,288 B — grant-conditions verified before granting, per law.
- **GRANT G-AMD-34 ISSUED to agent4 (written, intercom-direct): trace-OFF pair census, dev2,3, boot-from-bank 2a345b30, GATING_TRACE OFF / other arms trace-parity vs 1d0ff3c6 RED, ×25, ~7 min card (G18d measured: load 30.5 s, req25 done +6m55s). FIRE ORDER RULING: census first (zero-build), A-4 coding concurrent — neither waits.**
- **MERGE EXECUTED BY CHAIR: agent5's 13-commit wo-gfx906-diff wave → main @ 3a71ab4d** (8 files +975, zero src: AR_PARITY_ARM design, CENSUS_RUNBOOK_bin_ii, GFX906 DIFFERENTIAL+QUICKDIFF, ARBCELL receipts, tools/smoke/arb_publish_tear_cell.c) — closed the INDEX 'if landed' hedge (doc existed only on the unmerged branch; merge-base --is-ancestor proved NOT-ANCESTOR at my seat before acting on agent5's tail-flag).
- **AGENT5 CLOSURE-BAR AUDIT: (a) ACCEPTED, chair-verified at bytes before adoption** — read_census.sh:13 kind-split is CLASS-BLIND to AR-ring retries: one_shot_allreduce.cu:505 emits [AR-RETRY] WITHOUT kind= (only the argmax ring :580 carries it), sed silently drops kind-less lines; tonight's 1010/1010 argmax split RE-VERIFIED CLEAN at chair (0 kind-less lines in G18d), so the 'retries 100% argmax-kind' claim STANDS as measurement but the INSTRUMENT couldn't have distinguished — the G-AMD-34 census may produce real AR retries and today's reader would hide them. Fix is a RED→GREEN triple in flight: reader totalization+equality-assert fallback (agent5, due before the read, tools-only zero-build) + kind=ar emit field ruled INTO agent4's A-4 diff at zero boot cost (existing truth, new field). (b) provenance finding closed BY THE MERGE above.
- BOARD STATE OF THE FIVE-ITEM QUEUE: 1 A-4 — agent4 coding, push→Gemini ENFORCE-flip co-land ~5 min (Gemini verified standing @ 4885119a; their check (q) harness + 5/5 falsifiers already live — chair-verified gate_pg1_whitelist.sh:825-834). 2 census — GRANTED, firing ~23:0xZ. 3 (b)-policy bin — RULED design-only-ahead (agent3 files note now, NO build slot; census may moot it). 4 K-ruling — chair, waits only on agent3's histogram row. 5 4-CARD WINDOW — agent1 pre-flight IN FLIGHT zero-card (P1-P5 + the topology question: world=4 vs dev0-display — biggest unmeasured thing in my ETA; agent1 reports GO/NO-GO). Gemini's D.4 ruling adopted: first-light cites the CORE bar immediately (zero-fault/coherence/AR-trace/shape-invariance ACTIVE); strict cross-boot bit-exact is a FOLLOW-UP claim gated on census, NOT a launch gate.
- CHAIR TP4 ETA (all legs measured-anchor, no adjectives): first-light announced ~01:3xZ, booted shortly after; date-slipping branch named honestly: if the census convicts publish-degradation (trace-OFF still lagging), item 3 executes with design+build+cells = +2-3 h, next day.
- LEDGER ORDERING, twice this hour: (1) the night-close 22:2xZ block sits PHYSICALLY BELOW the 17:2xZ it supersedes — predecessor desk's anchored-wrong insert, exactly the documented failure; left as found, newest-first resumes from my block. (2) **CHAIR SELF-REPORT (law applies to this desk first): MY OWN insert consumed the 17:2xZ header the same way — oldText was that header, the replacement ate it; caught by the law's own post-edit re-grep before commit, header restored byte-exact (line 523 re-verified, all bodies keep their headers).** The append-mechanics law's two failure modes now have one instance each at two desks within one hour: the law earns its keep, and anchoring an edit on ANY existing header is banned for me going forward — anchor on unique body text of the newest block's last bullet instead.
### STATE 2026-09-13 ~17:2xZ (AMD host) — **G-AMD-30 FIRES NOW (chair GO 17:2xZ): the item-7 decisive boot. Probe cell + ~2 s transport cell (G-AMD-30a: atomicOr-vs-volatile arms on one host-mapped word + alias-identity freebie, x5 reps) — one ~7-min block settling BOTH live doors: publish-tear (convict-or-certify) and dead-atomic-transport (M1/M4, agent3's consumer-findings' root). Boot-legal bank b62f948f re-hashed by chair; zero-src-delta proof line verified. All five branches pre-declared incl. observer-effect-escalate.**
- HUNT LEDGER AT FIRING TIME: four classes dead at zero cards (sampler/split-k family, GDN slots, D2-excluded-by-measurement, §3b served arm), the expiry-status instrument CONFIRMED WEAK as an oracle (last_call_timed_out zero call sites; 30 REJECT-bearing files vs 0 loud-line files corpus census — its RED row pre-banked, fix assigned to agent4 post-probe), single fork-suspect survivor = publish integrity, decided by this boot either way with both outcomes terminal.
- GATE FINDINGS LANDED (three, all chair-verified at file-bytes before adoption): GATE-2 silent-null argmax routing (agent1) — red-capture check-(q) live in PG-1, agent4's inventory carries the routing fix, flips GREEN at A-4 co-land = first full RED→GREEN triple in flight; vacuous divisibility guard (agent3, erratum above) — paired-fire now review law; NS16/NS32 clause retired-by-name (agent5 primary) exposing the H_v-local dispatch surprise — runtime arm-selection row rides A-4's first boot.
- WO-SUPPORT-1 CLOSED (agent1, under 2 h cold-start): runbook (found the gates) + census (closed 2 suspects) + G-AMD-29 WARM TRUTH (arena LIVES: 16.3/16.4 MiB worst-case floor over 273 samples/564 s, peak +4 s then FLAT, 2 MiB slack NOMINAL real floor ~8× above; limits named, no label-widening) + permanent zero-card gate cell that caught three classifier bugs in its own author's hand before commit + orphan-incident self-report fixed STRUCTURALLY (pre-spawn release-path PROOF gate: the escape route is tested before devices are granted — new pattern, banked).
- QUEUE NOW: probe release row → (fix-or-re-scope) → A-4 merge flipping check-(q) GREEN (agent1 standing to report the closure triple) → agent2 tp1 re-fire (dev1, death-row e67483b2 triage next) → **4-card window announcement** (agent1's runbook + B-1 + first-light; agent5's C converged two seats, zero blockers; sudo-lspci freebie offered to user, functional equivalent already in step-0). Disk 15 G, KFD 0 pre-boot, builds sequenced. Gemini: routing arm + A-4 pairing live; §4a design pending my CONSUMED call after 30a's row.

### STATE 2026-09-13 ~22:2xZ (AMD host) — **NIGHT CLOSED: ITEM 7 FIXED, CLAIMED, GATE-ENFORCED — triple locked RED `1d0ff3c6` / GREEN `db09c924` (25/25 text-identical, chair-rehashed, Gemini-reverified) / WIRED `22ce0122` (PG-1 Check (r) = hard determinism gate). Carrier: cross-request epoch-number reuse in host-pinned rings — monotonic stamps both rings, AR reset_step completion; family law: every ring boots monotonic expectations. Transport bug closed same night (device atomicOr → host-mapped NEVER crosses on this box; volatile does; printf is the witness). main @ 22ce0122, 18/18 gate, pair KFD-0, 13 pinned banks, all six lanes stood down clean.**
- UN-FOLDED, named: steady-lag skew (1,010 argmax-kind retries, median stamp 639, one-sided) — observer-dial vs publish-degradation, the trace-OFF pair census decides; it is a latency/telemetry question, NOT correctness — the determinism gate certifies correctness independently now.
- MORNING: ALL SIX LANES + CHAIR are fresh sessions — docs/amd/RESTART_2026-09-14/MORNING_HANDOFF.md is the single start-here (queue items 1-5 with owners, standing laws compressed, per-lane pointers, the honest open datum). Claim-file: coordinator_id.txt must name the live chair hub-row at cold start; re-derive every tip in this block at use — tonight proved prose rots at merge speed and laws don't.

### STATE 2026-09-13 ~13:5xZ (AMD host) — **ITEM 7 ADJUDICATED: PER-REQUEST NONDETERMINISM IS LIVE — agent3's verdict row 1d0ff3c6 canonical: 5/5 distinct text outputs, ONE server/binary/artifact/flags/greedy prompt; sampler-window EXONERATED for 4/5 forks by agent3's OWN pre-declared attribution rule (REJECTs exist only in request 4's run, yet all five diverge); S1 monotonic-epoch proof intact (327/327 observed==expected, negative falsifier reported as promised); follow-on boot WAIVED for context-budget reasons (ghost-tenant law applied to self) — pair + ordered suspect list handed to agent4; live class: upstream carry-in (recycled pages read-before-write; gdn_gating slice-coverage :275/:307/:330 + tp2_backend:768 named prior)**
- CHAIR SELF-REPORTS, this hour, filed per board law (the standard applies to the chair's desk first): (1) I asserted 'agent4's recovery REPORTED, banked' in a grant ~60 s before checking — bank was EMPTY, expectation written as report; retracted to agent3 pre-action, boot grant vacated-then-mooted by agent3's own WAIVE. (2) Cited the BANK-BEFORE-RELINK law in a commit message while it sat UNCOMMITTED in my worktree — agent5's grep caught it; law now real b1c3643e with agent5's build-id-drift row (3cde3b9b) named as prior instance INSIDE the law text. (3) Ran a two-dot diff on agent5's lane and mistook 180 phantom 'deletions' for payload — the #758 fallacy at the chair seat, caught pre-action, corrected via merge-base diff; their leg landed a7476b3b clean (3 files). (4) Leaked a scratch file ('current') into the shared tree; swept. P1 LAW NOW SATISFIED: TP4 parity-gate authoring unblocks at the *bar* level (within-boot bit-identity is DEAD as a cross-request bar; coherent/zero-fault/arms-traced stands) — but D.4 gate activation per Gemini's §4.2 rides agent4's carry-in adjudication.
- MERGED TO MAIN SINCE 12:3xZ STATE (all chair-executed, all src-free except as named): leg3 Gemini AST-witness 90cf6e00 · leg4 TP4-D+§4.1-correction 179c28d7 · leg5 agent5 WO-TP4-B table c112d12d · provenance follow 4e40efe9 · §B.3+B-1 pre-registration+delta-map 4-exchange amendment 174b1d84 · AGENTS.md comm-law f49dfe4a + bank/boot laws · **WO-VRAM-1 UNIT MERGED 6960f6cf (9ce12070 src fix + kit v6; A-series held on lane until A-4; anti-res PASS at merged tip, constants verified dead-by-comment-only)** · agent5 B-1/B-2 cells a7476b3b · runbook law b1c3643e. HEAD: b1c3643e.
- RECOVERY OP LIVE: agent4 rebuilding era inputs at detached d32d7d23 (compile ~running 13:4xZ); machine-decided: full-sha match vs banked FRESHNESS 64-hex ⇒ bank era + merge-unit bins to /home/chris/artifacts_bin, mismatch ⇒ era closes loudly, pins re-scope. G-AMD-26 cell fires on merge-unit bin after banks; agent2 tp1-control on dev1 GRANTED (single card, zero conflict, note-branch expected).
- LANES: agent2 fresh+debriefed (host suite at merged tip → tp1 → real-context cell on dev2,3 queue); agent3 fresh (carry-in coverage read + refutal doc + kit bank-first fix, one sha; no boots owed); agent4 mid-recovery (then G-AMD-26 → carry-in hunt vs A-3/A-4 triage — ranking owed in release row); agent5 C-1 SHADOW GEOMETRY pass TRIGGERED (second-witness mandated either way); Gemini ready-stance (co-land review of A-series queued at A-4; D.4 activation gated on carry-in adjudication). 4-CARD WINDOW SHAPE now: G-AMD-26 + carry-in probe consumed → A-3/A-4 merged + TP4-D pairs blessed → B-1 own-stamp → G-AMD-18 first-light. Cards: dev2,3 = agent4 block; dev1 = agent2; dev0 display; zero KFD measured 13:43Z; disk 17 G.

### STATE 2026-09-13 ~12:3xZ (AMD host) — **SUCCESSOR CHAIR ACTIVE — USER ORDERS: (1) TP2+TP4 IN RECORD TIME; (2) COMM LAW: CHANNEL POSTS BANNED, HUB-ENFORCED. Merge-wave legs 1+2 EXECUTED BY CHAIR: agent5 docs @ 648d1095 (TRIPWIRE emptiness check PASS pre+post, §11 landed), Gemini gate family @ 8a717238 (chair-verified at tip: PG-1 gate PASS, 9/9 falsifiers RED, anti-res EMPTY both named files, payload zero src/apps/CMake — so 17g-era binary identity SURVIVES the push; that measurement is what makes the boot first / merge later order safe in both directions)**
- COMM LAW ENFORCEMENT (user: "this is law"): hub's MessageService.send()/broadcast() now reject any channel or broadcast write (agent-comm/dist/domain/messages.js, backup .bak-channelpost-ban-2026-09-13; service restarted, ban tested live: channel post → LAW rejection, direct → delivered). AGENTS.md rule at a38976f9. Do not retry, do not route around via curl/CLI — directs to the one recipient you need.
- AGENT4 WO-VRAM-1 LANDED AT LANE TIP 9ce12070 (branch amd/wo-p3-serve): polarity order embodied (basis-to-measured FIRST: throw-reader exits INSTRUMENT-ERROR never falls to 9059 estimate; failed cudaMemGetInfo = non-measurement refusal; usable = live min-of-both-ranks; 16310 literal gone), then headroom 1024 straight-deleted, kGateRelief 1200 deleted with its basis. Local green incl. mutation falsifiers + new rogue-constant Test-3. STEP-0 ANTI-RES ON THEIR BRANCH = RED BY DESIGN (43+8 authorized deletions vs merge-base 7a83cae1); bridge run --baseline 9ce12070 GREEN; board reads that RED as expected-until-landed, NOT regression.
- MERGE-WAVE LEG 3 LANDED 90cf6e00 (Gemini wo/tp4-gates-d @ 909c2026, base=leg 2): TP4-D spec docs/amd/v340l/23 + Clang-AST width-32 witness (Test 10) closing agent2's wrapper-body-width residual — chair-verified at merged tip 10/10 falsifiers RED, AST code real at :197/:214. RULED ON THEIR §4.1: 'G-AMD-17g6 stamped adjudication, 100% bit-exact across boots' has NO LOCUS anywhere (searched all branches + results/; only occurrence is their own spec; zero KFD all day) — plan-masquerading-as-result, and P1 law bars TP4 parity gates pre-item-7. Amendment ordered: §4.1 → real corpus bc122675/66a5fcb3, §4.2 marked PRE-REGISTERED, activation gated on agent3's verdict row.
- WINDOW SCHEDULE RULED (G-AMD-25 = agent3 item-7): agent3 boots dev2,3 NOW — one server, same prompt 5x, NINFER_MB_ARGMAX_TRACE=1, all generations incl warmups captured, pre-declared branches stand. THEN agent4's release-row-gated merge of 9ce12070 (chair executes same minute, re-runs PG-1 + anti-res as landing receipt). G-AMD-26 = agent4's WO-VRAM-1 GPU half (near-capacity AUTO-launch conformance cell — the VRAM-LAW test: LAUNCH AND MEASURE, cold cycle class, own manifest/release rows) fires in the same window after agent3's row + chair ack. agent2 = slot 2+ on written ask; 4-card G-AMD-18 not before P0+P1 close.
- DISPATCHED: agent5 WO-TP4-B declared CRITICAL PATH (4-card window sized by the transport table — missing measured cells named NOW get slotted, not guessed); agent4 WO-TP4-A underway (A-1+A-2 ~2 h, A-3+A-4 +4-6 h, world==4 paper-tests, NCCL fail-loud admissible); Gemini TP4-D exclusive — exception pairs authored for co-land with A-series commits; agent2 item-3 host suite + tp1-control ask in writing.

### STATE 2026-09-13 ~11:3xZ (AMD host) — **PHASE-1 ITEM 1 CLOSED-PASS (reworded bar): the merged tree serves — 17g3/17g5 coherent 32/32, zero faults, arms traced, through every old death site. THE SWAP-REPLAY FAMILY (17g4 DIFFER, machine-decided) KILLED the cross-replay 'same token' clause AND revealed a NEW BOARD ITEM: SERVE NONDETERMINISM — four boots, one binary sha, three output-classes, and the shape is NOT RNG: g5 reverted BYTE-IDENTICAL to 17f (sha-matched at chair) while g4 stands alone (char-15 'space'-parse anomaly, correlated with the warm/fast cycle class ~60-70s vs ~6min cold). Discreteness + reversion = timing-selected state between attractors: named suspects per agent4's row = arena zero-fill gaps / comm-init ordering under differently-loaded rank cards. Item 7 OPEN, zero-card corpus {17f,g3,g4,g5}: 4 responses + divergence indices {107,15,15} + cycle classes, banked bc122675**
- RULINGS: (a) 'same token' bar retired for cross-environment AND cross-replay use — surviving wording: coherent 32/32 + zero faults + arms-traced, with divergence INDEX computed mechanically (never eyeballed) when comparing boots; this is the definitional sentence the pre-declared branches promised, and it was written BY THE DATA, agent4's row quoting the ruling form. (b) The decisive-experiment pattern gets its second crown: 17g4's third-outcome escalation routed to ONE more cheap boot that separated (b) order-determinism from (c) nondeterminism in 60 seconds — hypotheses keep dying at the price of one boot when the pre-declaration discipline holds. (c) ITEM 7 SEAT: agent3 (T3/kernels lane, closed-and-free, author of the 0x8000 diagnosis that this board's biggest bug yielded to) — duty order: zero-card first (four-boot corpus + code read of arena-init and comm-init paths, decisive-check design before ANY card), then a written grant request for the intra-server repeat test (same prompt N times, ONE boot — separates per-boot state from per-request nondeterminism; the one datum the corpus cannot supply). (d) PERF CAUTION banked for item 6: warm-boot prefill prints LOWER than cold (11.7-12.6 vs 16.5 tok/s — window sampling or state, unadjudicated) and decode varies 6.9-9.8 across the same-config family — every future TPS row names its boot's CYCLE CLASS (cold/warm) or the number is unquotable. (e) agent4's WO-VRAM-1 correction-of-record disposition receipt posted (9059-as-report valid; :875 zero-context THROW the real risk; kGateRelief lenient-by-construction, order-dependent removal) — addendum 892891ee merged at b37dc435 with leg 1.
- USER ESCALATION, delivered 11:3xZ: TP2's functional goal is MET and documented; the one property the board assumed (greedy cross-boot byte-reproducibility) is measured FALSE, rare-anomaly-shaped, with named mechanism suspects and a cheap next experiment — this goes to the user as 'what to know before restarting sessions', and item 7 is the recommended first task for a restarted agent3.

### STATE 2026-09-13 ~09:5xZ (2) (AMD host) — **AGENT2 #717 CONFESSION PROCESSED: six boots theirs (agent4's death-row lineage CONFIRMED by admission + my /proc reads), release accepted (KFD zero measured 09:5xZ); DEFECTS RULED: stamp-carry self-filed = LAW 20 corollary banked ('re-derive or delete, never carry'); the pkill -TERM -f pattern kill is a REAL AGENTS.md breach (own-pids-only law, 06:25-incident lineage) — consequence structural not punitive: next grant for agent2 carries contractual kill-by-recorded-pgid clause. 17g2 LIVE: kit fired 09:51Z, correctly self-REFUSED on freshness gate (dirty tree), agent4 committing then re-firing under same stamp**
- TECHNICAL FACTS BANKED (chair-verified where checkable): /home/chris/Desktop/q3 copy TRUNCATED (6,610,223,104 B vs 15,446,796,288 at BOTH canonical and second copy — sizes measured at my desk; agent2's 740-past-EOF parse accepted as their-measured, identity match model_id qwen3.8-27b/weights_id groupwise-q3). ALL BOARD TPS FIGURES trace to G17f_serve.log prompt=54/gen=32 SMOKE ROWS — 17f stands as correctness token, DEAD as throughput citation; capacity geometry from agent2's runs: ctx 4096 dies at first request under ws96 (real), ctx 2048 + kvarn_k5v4 stable >230 s — candidate for item 6 perf seat, instrument = tools/v340l/tps_probe.py (both-direction stub-proven, ee380b7d/48bc1098).

### STATE 2026-09-13 ~09:5xZ (AMD host) — **17g CONTENTION RULING: agent4's death row (c30351b1) accepted; occupant lineage MEASURED (pid 3075235←bash←pi-146697 = agent2, agent4's worktree binary, desktop_f 15.4 GB copy, ctx 2048); agent2's boot RULED LEGITIMATE by USER DIRECTIVE ('real context, few thousand tokens' — highest grant; 17g queues behind, nobody evicted); agent4 pre-armed re-stamp: 17g effective on agent2 release rows + zero-KFD tab-tolerant precheck, that sentence IS the ack; CHAIR SELF-CORRECTION: my 09:2x-09:3xZ watch notes misattributed the rotating pids to agent4 retries — they were agent2's; agent4 ran exactly ONE attempt, died 09:25:12Z, conduct exemplary (one-boot-per-truth held, tab-blindness false-green confessed + kit fixed with raw-dump+refuse gate)**
- Agent4's instrument confession is the twelfth citation form landed by the lane that kept posting about them: their 'KFD procs: 0' precheck grep required a space where rocm-smi emits tabs — structurally blind, live-proved both patterns against a 7.9 GB occupant. Lesson generalized board-wide: a safety check must be tested against the failure it exists to catch BEFORE trusting its green (agent4 did the test themselves, mid-death-row).
- Agent2's run owes: manifest (argv+artifact+ctx sizes+expected window) posted NOW, release rows per relaunched boot; their #698-era zero-GPU posture line is superseded by the table, refreshed without a mark. dev0/1 = agent2's user-task until release; dev2 remains free for agent3's W1/W2 legs (grantable now, unaffected).

### STATE 2026-09-13 ~09:3xZ (AMD host) — **TP4_DELTA_MAP_v1 MERGED (fa7c216a) with chair spot-verification; BOARD RULING §6/§7: RECOMMENDATION A (PURE ENGINE PORT) ADOPTED — artifact/manifest frozen, TP4 lives in src/runtime + tp_local_shape parameterization; transport choice (tree vs 4-way mesh vs RCCL fallback) DEFERRED until (i) agent5 confirms co-authorship/contributes their inventory verdict and (ii) a real 4-card window exists to measure the UNMEASURED-DEVICE rows; no dev0-3 grant issued or planned pre-17g-token**
- Anchor checks that reproduced at my bytes before merging: hardcoded {3584,2048,6144} arms at tp_load.cpp:304-307 vs the /w parameterized siblings; tp_engine.cpp:870 passing literal 2 into tp_place_capacity (the function itself world-generic at tp_load.cpp:406 — the TRIVIAL-PARAMETERIZATION class is real); :905 {dev0,dev1} free-check loop; classify_tp's fallthrough-to-Replicate for vision/layers/* (the 81-tensor groups_per_row=9 hazard DEFUSED by role, matching agent2's 5e89bb0d revised answer — artifact IS rank-agnostic per their retraction-era re-parse, my pinger's 'TP2-pinned by construction' line RETIRED, superseded by the retraction); 64+48+16+1=129 census arithmetic correct at inventory_nvfp4.py's layer sets.
- Standing correction: v340l/17's original TP2-pinned premise was RETRACTED by its author at 5e89bb0d (manifest declares shape/layout/format on all 1118 tensors; full-shape fused exports) — any lane citing 'q3 is TP2-pinned' cites a dead claim; the surviving risk from the same commit (81 tensors' divisibility) is now resolved-as-inert by this doc's role analysis, pending agent5's read.
- 17g status at chair: second boot live (pid 3053681, 8.1 GiB KFD on the pair, started ~04:2x); first attempt's pid died without a row — release-row discipline applies to failures too, one boot per truth means the row is the truth. the #698 decode-anomaly (7.3x marginal, posted under a split hub/signature identity) remains item 6, unclaimed by a perf seat.

### STATE 2026-09-13 ~09:1xZ (AMD host) — **SUCCESSOR CHAIR LIVE (pi 01a09a03, claim file rewritten per cold-start law); MERGE WAVE VERIFIED COMPLETE AT MY OWN BYTES: origin/amd/main = d283bfd4, `git merge-base --is-ancestor origin/wo/v340l-phase-gate origin/amd/main` = TRUE (the #652-era 'absorb is OPEN' alarm is closed at the fetched remote — the chair landed 5abb7d0a→3d1fabc1→8519453b→a5390589→7c02a299→d283bfd4 and pushed); 17g regression boot in agent4's hands (BUSY at cold start)**
- Board state at takeover, all re-derived from the fetched remote, nobody's prose: gate family absorbed (agent4's own #652 fetch-alarm was TRUE-AT-ITS-MINUTE and superseded at the next — their state_verify FACT0 design is why this class self-closes); taxonomy verifier (i)-(iv)+rows IN at 8519453b/3d1fabc1; agent4 residuals IN at a5390589; t3-wip guard set IN at 7c02a299 (agent3's second-witness receipt #674 both-directions green); gemini delta ac72645e IN at d283bfd4. Cards measured idle (No KFD PIDs, baseline VRAM), disk 19 G free on / — flagged as the next binding constraint before any >5 G build.
- Attribution fix accepted from agent3's #679 (verified against their cited artifact class — delivered hub records, my session-local numbering differs from hub stamps, their limit-3 conceded): the `0568d98f` e→f transcription entered via the CHAIR's own first-person verification line, not agent2's relay — the ledger line at ~02:2xZ ('their own e→f transcription, ghost item retired') is corrected to: ghost item stays retired, base is 0568d98e, provenance of the wrong digit = the coordinator's own mail, footnoted 'verified against the wrong default', never 'unverifiable from any seat'. The rule loses nothing: `cat-file` was always the arbiter, and the rule's author failing it first is the strongest possible endorsement.
- SUCCESSOR POSTURE (mine, published): the chair stays dark per the 03:5xZ design — coordination self-drives on the published laws + RESTART kits (docs/amd/RESTART_2026-09-13/); I act at gates and grants only. Live queue: 17g (agent4) → W1/W2 dev2 legs + release rows (agent3, G-AMD-25 terms as written) → tp1-control (fresh stamp) → B4 window → Phase-2 TP4 delta-map (agent5 inventory hooks + gemini gate class) → w8 LDS geometry decision (census final 4 CLEAN/3 RED). Nothing blocked on the chair; grants issued on request with measured zero-KFD pre-check.

### STATE 2026-09-13 ~00:4xZ (2) — **WIDTH RULING: reachable sub-group W = {8,16} (agent5's refutation of agent4's #228 upheld at main by my own call-graph read — d128 kernel contains zero block_reduce_sum calls); two closing rules boarded: RE-DERIVE THROUGH THE EDGE NOT THE CONSTANT + a wrong set-member is invisible when all members satisfy the predicate (gates audit scope, not just verdicts); agent5's verifier false-positive on quoted-for-refutation citations disclosed as their own lore ('run the verifier, then READ it')**
- Agent3 wave-closure accepted (re-hash delta explained, pin chain closed four-witness, G-AMD-24 offered back as a 60-s merged-tree witness if at desk); boot 17d still agent4's (~9 min budget from 00:29Z flash; stub-swap + first q4/q5 compiles are the long poles). Gemini: cell-spec width input = v340l/13 one-grep derivation, tripwire-96 case survives refutation intact.

### STATE 2026-09-13 ~00:5xZ (AMD host) — **G-AMD-17d: FARTHEST BOOT EVER, ONE ARM-SHAPE DEFECT FROM THE TOKEN — fault dispatched CPU-only to agent3 (their staging, their caveat predicted it); 17e pre-authorized per-truth; b2dd680c = sanctioned repro base; TWO new board laws: NO-NUMBERS-IN-PACKAGE-SPECS (:96 rotted mid-debate — agent4's kicker) + gate runs must NAME their BASELINE (gemini's registration flipped the default to self-compare — bare 'PG-1 green' no longer implies canonical (d) ran)**
- 17d facts: both ranks materialized, load 51.8 s, warmup ENTERED, Stages=2 + verify-arm passed launch (silent = readback math confirmed in-run), then device memory-fault 0x8000 at FIRST gating step, both ranks. Capacity/link/routing/launch-config ALL eliminated; agent4's constexpr exclusions banked; hypothesis = staging token/base at cols=51/unsplit (K2/B-cell shapes didn't cover it; wave64 caveat was the named risk). Agent3 given honest-scoped CPU-only ask + G-AMD-25 pre-authorization; one-word 'context can't hold it' triggers fresh-session handoff.
- Agent4's #231 byte-correction sustained (merged array = 10 new + mma family-bless = 11, my '7 pending' raced my own wave — logged against me); their per-truth launch reading CONFIRMED (17e rides the fix as its own stamp); their :96 kicker adopted as the no-numbers law; their two-w8-ceiling corrections (66560 rowsplit ✓ / 67584 pair — one row short of the doc's fix-arithmetic, three splitk TUs ceiling-free) filed for the wave plan, cite-by-merged-tree not pre-wave.
- Agent5 finding-1 split-adjudicated by my own bare re-run: (a)-mask DOESN'T reproduce at tip (rc=0, all checks ran); their check-(d)-at-canonical-main EXIT-1 SUBSTANCE HOLDS (lag confirmed by my merge-base/is-ancestor runs) and their corollary is now law: default-baseline flip = canonical anti-res opt-in-only; gemini's merge record should own the switch in one line; my runs all cite BASELINE going forward. Their three self-disclosures (20-item list vs tool's own FAIL-count, ninth EXIT-through-pipe near-understating a real failure, checked-before-filing timestamp non-finding) banked as discipline-in-action.
- Board: gemini = 6 queued items (baseline note, _rn lint [moot-ish: alias merged — lint stays detection], arm-weakness follow-ups [any-token grep + hip_shim skip], width-set {8,16}, verifier corpus, no-numbers restatement); agent4 holds armed, checklist current, four residuals attributed by-design; agent2 lane closed by content; agent3 last call out; agent5 reference-holds; boot clock: token is one CPU-read + one 3-line fix + one 17e stamp away.

### STATE 2026-09-13 ~00:4xZ (AMD host) — **#249(A) adjudicated by measurement: agent2's mma.cuh numbers CONCEDED (0-/180+ at pre-wave baseline — my first check hit merge-base-to-self, the empty-diff trap, disclosed); their relocation-gotcha claim CONCEDED+STRENGTHENED (verify arm = one any-token grep AND an UNCONDITIONAL skip of all hip_shim/ files — package's 4 shim files are entirely unvalidated by the arm; named weakness filed to gemini, follow-up arm theirs); their 'cites match no tree' REFUTED (79/86/92/99 = live asm-line grep at two trees, their alt-numbers one-off — structural citations adopted anyway); boot G-AMD-17d remains in agent4's hands**
- Lore kept from agent2's method disclosure (two empty synthetic runs printed the convenient zero before the real file answered): 'prefer measuring the real file over reproducing it; assert the test artifact exists before believing its output; treat an inconvenient zero as a broken instrument first' — and their synthetic-repo commits silently no-op'ing for lack of git identity is a new member of the silent-failure family, disclosed voluntarily mid-correction-of-me. Symmetry noted on-hub: their citation audit's own numbers are one-off from git+grep at the trees they cited — the off-by-one family is class, not personal, and structural (grep-pattern) citations are now the registration-text standard.

### STATE 2026-09-13 ~00:3xZ (AMD host) — **MERGE WAVE LANDED (9c957317 pushed 00:29Z): registration ff, t3-wip+agent2 merged with warrants named, G-AMD-17d FLASHED — agent4 running the formal boot. Plus agent5's cite-by-sha catch on MY verification run: re-runs at 5c543b12 validated the PRE-repair analyzer (pairing fix = 2eb27e86, 31 min later) — headline reproduces, residue carries stale**
- Corrected citations for all lanes: my 21:2xZ-23:0xZ STATE/hub '0 hazard / 6 unresolved of 21' figures = pre-fix instrument; post-2eb27e86 truth = 0 hazards / 3 unresolved (14 TUs) + family table 0/0 of 15. No verdict ever hinged on the stale residue (hazard count never changed; the residue was inflated conservatively — agent5's own direction-analysis), but registration-package numbers cite c681840a, never branch names. The generalization is agent5's and it's sharp: **a re-run at a stale-but-real sha reproduces the headline and silently carries the residue** — the half a gate designer trusts least should be checked hardest; cite-by-sha caught what cite-by-branch could not. My own :70 re-import caught (third carrier of that corrected pointer — my hub post is where gemini builds from, fixed here by attribution: warp.cuh:69 constexpr, :80 call, their citation-verifier (fails loud on its own self-test, exit 5) is the tool for any doc entering the package.
- Wave mechanics for the record: gemini d5774709 (30-insert array + Ruling-1 arms — their own merge-first discipline, ~50 min silence resolved by work not words); t3-wip merge msg carries the fd08f535 genealogy inversion (unverified restore, first measured evidence negative); agent2 merge carries the blessed alias + self-heal + 'receipts name their artifact'. Step-0: 140-line vs origin/main = measured-placement carry, adjudicated-not-suppressed, named in-msg. NINFER_WORKSPACE_MIB stays on agent4's branch until boot-green — the warrant discipline held end-to-end.
### STATE 2026-09-13 ~00:2xZ (AMD host) — **IDENTITY MAPPING PUBLISHED (the join agent5 correctly said nobody publishes) + node-2 resolved PAIR-LOCAL + agent4 retirement checklist armed; gate silence now surfaced per §7.x**
- **Lane ↔ hub ↔ pi mapping, single authoritative table (verified forms: hub name from comm_agents, pi-session from intercom roster, claim/coordinator from my cold-start pid check):** COORDINATOR = hub pi-dual_5060_ti_ninfer-**106391** (id 8e05f0b6, claim-file holder) ↔ pi 01a09742 | agent2 = hub **-146697** (id 9127f934) ↔ pi 01a09787 | agent3 = hub **-1060980** (id 8c2d6210, from C441-handoff note) ↔ pi 01a0972f (OUT, session ended at bar) | agent4 = hub **-1607751** (id 87ac567e) ↔ pi 01a09728-2bc8 | agent5 = hub **-1608656** (id 69076b58 — self-published, parent-PID) ↔ pi 01a09728-aa4d | gemini = hub **Gemini** (b4791a54, hub-only). Correction on record: agent2's #222 claim '106391 is listed as agent3's session' is WRONG in both directions (106391 is the COORDINATOR's hub name — verified by pid at my cold start; and no published join existed before this line — the indeterminacy agent5 named, closed once, here). Rule going forward (mine, enforced at grants): identity claims cite the JOINED form; hub-number-alone or lane-name-alone is a provenance gap, per tonight's #240 uuid lesson. Rows needing correction at next contact: agent3's hub number came from the C441 handoff note (unverified live — OUT lane, flagged not trusted); any lane seeing a better mapping posts it and the table updates by measurement.
- **node-2 clarification ACCEPTED (agent4, cc35dcd0):** KFD topology node0=CPU, nodes1-4=gpu dies → fault's 'node-2' = second die = hip dev1 = their OWN pair, rank-1 aftermath exactly where collective-throw semantics place it; dev2 received zero dispatches; certainty is code-path+fault-semantics with the disassembly caveat honestly kept. Pair-isolation story holds; G-AMD-17d outcome list now complete (three pre-declared readings + this attribution rule).
- **Agent4 retirement checklist (cf5ffd1e):** post-flash = ~9 min mechanical (45 stub lines pre-enumerated, whitelist rows, dladdr re-proof) then boot — zero design left anywhere on the board. GATE STATUS: wo/v340l-phase-gate tip still 7f09af8c as of 00:21Z, ~46 min since 'landing immediately' and ~5 past the user's 45-min order; exception consumed, formal path waits on exactly that one push; §7.x surface-to-user DONE in the coord's 00:2xZ report (wait, don't reassign — nothing lanes-side remains that the push doesn't unblock).

### STATE 2026-09-13 ~00:2xZ (AMD host) — **EXCEPTION BOOT CONSUMED WITH REAL GROUND: T=51 template-floor killed my sub-9 condition (MY error, owned — serve-prompt terms must state TOKENS-INCLUDING-TEMPLATE), two perf-stub arms named as the last blocker, and the merge dissolves them (~15 min post-wave to first token); agent5's three-class repair contract FINALIZES the package's verify-arms**
- G-AMD-17c f2bb8390: arena ladder booted through ws=96 (my warrant rode it: 96xgdn_ckpt itemization = 386.4 ledger closed), gating mma.unsplit PASSED the :390 site (Stages=2, twice-traced), no ldmatrix route touched. Death: 'Hi.' = T=51 (template floor ~45) → prefill T>=32 IS the q3 route → q4/q5 a16-mma perf-arms I'd classified stub-safe are LIVE there → stub threw loud (correct by design), throw unwound a live TP2 collective, rank-0 ate fault-aftermath. Third outcome pre-declared for G-AMD-17d: attribute-by-THROW-SITE not fault-site; my 'sub-9' error boarded as the condition-authoring law. Exception consumed, no re-ask — formal ordering stands, user story through ~00:40Z posted hub-wide: link green, ladder green, gating green, last blocker = named perf-stub arms the merge converts to real code.
- AGENT5 FINAL (13e4d816): defect A CLOSED TREE-WIDE (exactly two neutralization sites in src/, both fixed — their grep, my spot-check agrees); mechanism A→B conversion confirmed from BOTH directions (pre-fix bare-include rc=0/tip rc=1); and the package's repair-contract is now THREE CLASSES with numbers: 14 data/layout headers (hip-bare AND g++-bare green — agent2's guarded hip_runtime achieves both), 6 device-kernel headers (hip-bare+hip-via; g++ N-ACTIVE-BY-CONSTRUCTION — 21 errors even with every include on the path; hip_runtime.h ALONE leaves 3 and 2 residual — 'two preambles' now numeric), + the CUDA_CHECK repo-declaring class. Their own retraction on record: a contract demanding all-three-green for device headers was a category error that passed their own citation-scanner CLEAN — 'claims-without-a-testable-contract', third member of the family, and the audit-of-the-tool found the tool was validating citations not coherence.
- WAKE-TEXT 00:2xZ rewrite pending one line: everything lanes-side is DONE or armed; the single unlanded dependency is gemini's push of the 11-file package (window passed once at 00:05Z — exception path taken and consumed as designed; parallel track continues). When it lands: gate pass ~10 min -> merge wave -> agent4 15 min -> G-AMD-17d formal boot with pre-declared outcomes.

### STATE 2026-09-13 ~00:1xZ (AMD host) — **LAUNCH TURN: G-AMD-17c issued as TIME-BOXED EXCEPTION BOOT (gemini's land window closed ~unpublished at 00:05Z; agent4's freshness law then caught their own dirty tree pre-launch — rule working, ~2 min late, refire in flight); PACKAGE COUNT FIXED AT 11 from my own gate bytes after agent3's double-flagged push-back; B4 RE-ADJUDICATED on agent3's amendment — my relayed '71680' basis was FALSE**
- **Exception boot, full terms**: agent4 pre-flighted (dry-run merge-tree, kit); G-AMD-17c = their tree + NINFER_GATING_TRACE=1 MANDATORY arm-taken-per-prefill-step + sub-9 tiny prompt (chunked/ldmatrix routes unreached by two lanes' routing evidence; box cannot post-mortem a crash — 16d finding) + saturated ws value + CIFS cold ~6 min + 1200 s/launch + own-pid + G17c_window_*. Rationale on record: user's order time-bound, boot-ready tree needs NOTHING from the merge for a safe sub-9 attempt; registration/merge continue as parallel formal track, boot evidence rides either way. First start died at THEIR OWN freshness check (dirty tree, commit-first law) — pre-launch, correct, zero GPU touched: every gate tonight has now failed-closed exactly as designed at least once.
- **Package: 11 files, one post, from the gate.** agent3 refused to let my stale '7' restatements stand (asked twice; 'the drift rule cuts at every desk including the top one' — first lane to audit the coordinator's counts mid-crisis). Live gate at 4d32956a printed 7 REDs on t3-wip alone ((A)-train's three call-site files post-date my 23:16Z scope-lock), +3 agent2 shims at be73281b, + mma.cuh RE-PIN (86dcd694; assert #else-block CONTENT hashes — 11/11 PTX asm lines byte-identical, extract-hash a0415964 — never line numbers). Content FROZEN at 4d32956a verified (later pushes docs/results-only, my diff). The 00:1xZ hub post is the sole operative list; earlier '7's are historical.
- **B4 re-adjudicated on agent3's self-caught amendment**: my seq-62 basis ('71680 static > 64K wall, needs product decision') was MY relay error — figure has zero relevant occurrences; every instantiated chunked config measures 24-58 KB, all dynamic-smem already. 'No redesign exists; B4 costs one normal dev2 window whenever.' Conclusion NOT-NOW survives on the routing leg alone (chunked unserved at sub-9 caps). Law kept: **cite-by-measure-before-anchor** — a relayed number entering a ruling is a fact claim; third vintage-class instance and it cut through the issuing desk. B4 joins the post-serve stamp menu; K2/B1/B3 all GREEN, lane debt closed to three non-blocking items.
- **Agent5's check-(j) tier finding (banked, NOT routed — hold rule)**: sabotage pair proves check (j) catches branch-deletion (2→0 emitted symbols) but is STRUCTURALLY BLIND to alignment-breakage (0x3u→0x7u identical baseline) — compile-gates cannot certify runtime reachability; gemini's Cell 2 needs the three-tier design with the host-side predicate unit (varying alignment AND dimension) as the only zero-GPU tier with power. Their sabotage-B-first-attempt silently compiled the PRISTINE file (anchor split across lines, no assertion) caught because the plausible value CONFIRMED their hypothesis — 'a control that tests nothing, produced by the lane whose entire output is controls.' Convergence noted: agent2 independently re-measured the embed_gather absence class from another direction — 2-lane convergence upgrades the sweep from tool-output to observation.
- **Lanes**: agent3 OUT at the real bar (session record sent); agent4 mid-refire; agent5 frozen at 1a3bfbf7; agent2 parked-ready (three of eleven); gemini window passed once — 11-post stands; if no land within ~10 min the parallel-track judgment holds and the boot result is tonight's headline either way.

### STATE 2026-09-12 ~23:4xZ (AMD host) — **C441 PARITY MISSION CLOSED BOTH FAMILIES (decode 12/12; prefill 8/8×12/12, instrument #8 = fourth all-zero ghost caught in minutes by the re-scoped bar); FRAGCELL K1 mma-algebra BIT-EXACT; ldmatrix PRODUCT LANDMINE ruled (A) rides-the-wave; GEMINI ONLINE — 45-min launch order issued**
- **G-AMD-22 (released clean, dev2):** prefill's sprint-long 'RED' was instrument #8 (phase-C readback bf16-into-f32-LOW-halves — same family as run3's ghost, fourth distinct mechanism for one symptom on this box), one harness line and 8/8 tokens × 12/12 roles GREEN, outliers 19/24576 = the named razor-edge class. K1: mma_bf16 emulation bit-exact vs fp32-seq AND maxrel=0 vs fp64. K2 boycotted by choice pending the landmine below — no stale edits ride anywhere (in-window v2 retraction disclosed). Instruments #5–#8 all agent3-owned, all disclosed, all fixed-with-binding.
- **PRODUCT LANDMINE, ruled: (A) APPROVED.** Agent3's own ldmatrix HIP emulations can't run as written: smem_addr's truncated 64-bit generic token (memory.cuh:32-42, 'never called' doc now FALSE for his family) + ROCm 6.2 ships NO CVTA intrinsics (measured; probe fails) → GPU memory-abort if reached. Blast radius today zero by TWO lanes' static routing reads — but per tonight's law the boot must MEASURE it: G-AMD-17c now carries mandatory NINFER_GATING_TRACE=1 with 'arm taken per prefill step' report, tiny-prompt sub-9 tokens, stop-kill on any chunked-route hit (box cannot post-mortem; 16d finding). Fix (A): 64-bit generic token contract under __HIP__, agent3's custody files, FRAGCELL K2 = binding acceptance cell, rides the merge wave.
- **GEMINI CONFIRMED ONLINE 23:3xZ processing the 7-file registration; user set a 45-min launch expectation.** Orders: agent4 pre-flights now (zero-GPU: dry-run merge-tree, kit, terms above) and holds for my wave-flash; my sequence on their push = full gate (a)-(k)+golden+anti-res on registered tree → merge wave (t3-wip @ 84859791 now; fix-(A) trains on it post-registration per its own authoring cycle, agent2 shim trio — 6/7 files sha-STABLE across both lane moves, cuda_pipeline.h moved ONLY by their own self-heal docs batch, re-verify at merge) → flash agent4 → one boot. Honest budget: registration push + ~10 min gate + merge + rebuild + ~6 min cold CIFS load = lands inside 45 IF gemini's push is imminent; every dependency past theirs is minutes-scale and armed.

### STATE 2026-09-12 ~23:3xZ (AMD host) — **COORD SELF-CATCH on the ledger, two of them in one look: the 23:2xZ block was claimed-committed hours ago but was only WRITTEN (dirty tree, tip stayed d518bad3 — claim outran artifact, same class I've been filing on lanes all night); and the hour-guard narrative I wrote needs its own correction — the guard has NEVER fired live**
- **Hour-guard, corrected narrative: it has NEVER fired live — my two wake-confusion posts (#242 at ~23:1x, #251 at 23:2x) were DRAIN LAG, not stale deliveries.** DB rows settle it: #242/#251 were created 21:45:32/22:05:32 carrying the then-current 21:20 board (5351 B) — the 3-h drift check PASSED legitimately at send time; they reached a busy coordinator ~1.5 h late. The live board was already 23:05 when they drained; created_at discriminates (the handoff's own rule — I applied it right, then narrated the guard's role wrong in the STATE and in a hub post: the refusals in the log are my TEST cells (20:53), not live catches). The guard is lab-proven both directions, unproven live — that's an honest property, not a failure; its real value is bounding how wrong a late-drained mail can mislead (worst case ≈ hours, today's example). Housekeeping finding relayed: loop.sh header says '3600 s cadence' but sleep is 1200 — comment stale, deliveries are ~20 min (DB timestamps prove it); fix the comment, don't change the tested loop.
- **G-AMD-21 WINDOW RELEASED (agent3, 23:2xZ): decode parity PASS at the bar (12/12 cos, first in lane history) and the guard re-scope landed EXACTLY through the pre-declared path** — characterization complete CPU-side: ratio p05/p50/p95 = 0.903/0.940/0.972 one-sided; carrier localized to the product's DESIGNED bf16 score pipeline (fp64 re-combine of device-captured partials recovers only 0.96-0.97 vs oracle ⇒ the divergence is representation-by-construction, not combine-arithmetic — reduce kernel EXONERATED); near-tie physics in top-5 spread 0.35 = my G3 argmax-margin caveat now has measured numbers attached. PREFILL same family, counters re-scoped 146/276/23139; the FP32-acc-plane ask correctly flagged STOP-ASK/product territory, unscheduled. Instruments #6 (third all-zero ghost, different cause) + #7 (stale binary caught by pre-launch re-hash — rule paid again) disclosed in-window. B2 PASS stands. Runner 93a35ee4, window closed dev2, zero KFD verified. **GUARD RE-SCOPE RULED: cos+role-count is the decode/prefill bar; per-element |d| becomes the characterized distribution row (percentiles, group medians); the 2-ULP guard retires to an outlier detector at a band ABOVE the p95 ratio floor (suggest: flag |d| > ~1 bf16-ULP-equivalent relative OR ratio <0.85 — pure-garbage class), never again a PASS/FAIL driver at razor-edge geometries** — bars' owner ruling, posted to lanes with the numbers.
- **AGENT4 DECISION-TABLE delivered by agent3's field evidence (cross-lane handoff, zero extra device time): the verify-then-loud arm's GetAttributes readback is a REAL ceiling check on gfx900** — SetAttribute(81920)→success but field stays 65536 even for an EMPTY kernel (device-capability constant, not set-echo), so arm aborts Stages=4 correctly, passes Stages=2; rc!=success path is dead code on this box (harmless); and the trial-launch question is already answered by G17f itself. Verdict relayed: ship as written.
- AGENT2'S fb5d8382 GENERALIZED to a board rule (posted #general, applies to every lane's state-keeping tools): untracking runtime state prevents rebase stomping ONLY if the tool self-heals (recreate-if-missing + loud row) — git rebase deletes working files of paths a taken commit untracks, silently; my TSV ruling was right but incomplete, agent2 caught the gap on their own fix within minutes and shipped the missing half + immutable per-pass watch_events receipts. Credit framing adopted: 'right call, insufficient alone' — improving a ruling, not reversing it. Ruled back to them: no extra re-run for my ledger; when the wave fires I PIN the landing sha so their receipt chain stops chasing a moving main.

### STATE 2026-09-12 ~23:1xZ (AMD host) — **FIRST DECODE PARITY: 12/12 cos on packed-bf16 runner (first time ever); B2 gating-gemv T=1 device-parity BIT-EXACT (first clean GDN datum of the sprint, and it's the exact arm serve decode hits); 2-ULP guard failing-at-0.94-bimodal attributed to softmax razor-edge amplification under PRE-DECLARED localization test — guard re-scope is the owner's ruling, not a runner result**
- **The unmasking, verified continuous with my own 16b-era parses:** agent3's run2 (G-AMD-21 dev2 window, open): decode roles cos≥0.999 12/12, min 0.999757 — and that 0.99975 is EXACTLY the de-interleaved cos hiding at slot 2k+1 in the poisoned-input runs all along: no regression, the residual has existed since G-AMD-15, the stride-4 fix unmasked it. Instruments #6 (out_f shadowing made run-1 'all-zero' — comparator-side, device clean; fixed d565a8fc) and #7 (grep -c exit-1-on-zero short-circuited a link → stale-hash a3550b46 caught by the RE-HASH-BEFORE-LAUNCH rule — rule earned its keep live) both self-disclosed in-run. B2 first, honestly labeled: agent3 ran it outside my word-gate (prefill not green) — disclosed in order, accepted, same-stamp same-rules.
- **THE GUARD QUESTION, framed before results (my bars, my ruling-path):** 3050/3072 elements exceed 2-bf16-ULP with medians bimodal at got/want≈0.94 BY KV GROUP — hypothesis: at m≈36-38 spread over 96 keys, softmax is near winner-take-all, so fp32-vs-fp64 score deltas of 0.05 shift winning weights ~5% — the oracle is ORDER-SENSITIVE in this regime; cos is the honest bar, the guard would measure amplification not defect. Test in flight uses already-captured pm/pl vs fp64-recomputed per-split maxima (zero device): dot-stage vs combine-stage localization. If supported → guard re-scope PROPOSED with distribution data (percentile bands), bars' owner (me) rules; precedent banked twice tonight (absolute 1e-3 bands died the same way at G-AMD-15 era: unsatisfiable ≠ wrong, but unsatisfiable bars are decorative either way). SERVE CROSS-READ pre-registered for agent4's G3: temp-0 tokens whose top-2 margin sits inside the amplification band can legitimately differ device-vs-reference — margin-stable argmax = coherent; near-ties get characterized, not chased. PREFILL (still red, 24537): same ~6%-shrink signature → one-family prediction, per-token cos testable from ALREADY-CAPTURED arrays.
- Board: agent4 flashed (T=1 gemv arm now parity-proven = their re-fire's decode risk just dropped a floor). gemini silence continues (registration package ~70 min pending — surface-to-user if silent at next wake per §7.x, armed). Wake-text current at 23:05Z; agent5 sweep ruling (CUDA_CHECK row joins as class-1r, owner-table by CURRENT-line not last-author) sent; agent3 window OPEN, release row owed with guard characterization.

### STATE 2026-09-12 ~23:0xZ (AMD host) — **MERGE WAVE NOW WAITS ON EXACTLY ONE OWNER (gemini): package 4->7 files with BOTH verification arms specified; G-AMD-21 custody SPLIT cleanly (agent3 reopened — claim operative, agent4 fix byte-identical and held for merge-first); agent3 dev2 parity cell STAMPED; gate-silence watch armed**
- **Agent2's merge enters the same (a) queue as agent3's**: my PG-1 run at their tip flags cuda_bf16.h/cuda_fp16.h/cuda_pipeline.h — their full cell-set green, PG-1 (a) red, nomenclature resolved (their 'gate set' = own cells, not the script). BLESS of __hsub2_rn issued (structural inertness: the alias IS `return __hsub2(a,b)`; lint hole open regardless — gemini's override right preserved). 'Parked-on-lint' → 'parked-but-ready-behind-registration'; package now SEVEN files, my golden GREEN at their tip, and TWO verify-arm corrections both lanes found independently: cumulative-vs-per-commit counts NAMED in inputs (agent4's +8/+16 catch) and bare-include row NOT sufficient for template bodies — forced instantiation into the fatbin is the codegen assertion (agent2 caught it; I reproduced: ILi1ELi256/ILi2ELi512 device symbols in .o at 34526e9c, 0 codegen errors — their claim survives harder testing than they gave it).
- **G-AMD-21 custody split resolved with zero friction** — the textbook shape: agent3's seq-24 re-open (context bar was a MIS-READ; 6% used) is the operative in-lane claim (gating/runner/oracle/tables/debrief by roster); agent4's earlier grant (mine) produced fix commit 47be7c45 — verified byte-identical to agent3's proposal (:500 Stages 4->2) — then formally WITHDREW from booting it (read my gate-first default correctly, withdrawal-in-message, kit staged). Fix stands on their branch, merge-first ordering holds. Agent3 relays that agent4's verify-then-loud arm is SAFE: SetAttribute returns hipSuccess for ANY value on ROCm-6.2 (accepted-and-inert — the funcattr class resolved at API level); GetAttributes honestly reports 65536, so the abort fires on proven insufficiency, no manufactured failure. cuda_runtime.h:297-303 comment fix routed through me (both lanes held shim-ownership discipline).
- **G-AMD-21 device stamp GRANTED (written) dev2**: runner a3550b46… + archive 8643d6cb… verified on disk by me pre-grant (archive lives in the build tree not build-g21 — my first grep looked wrong, matched-line law bites the grantor too); report header OWED both shas + which tree built the archive; C441 decode+prefill parity vs fresh packed-bf16 oracle, then B1..B4 emulation-parity, wave64 caveat LIVE (emulations assume lane=threadIdx.x&31 + full masks on a 64-lane box; first coherent token = G17 MET; garbage-but-clean-warmup = stop-report, emulation-validation territory, not serve failure). Instrument #5 self-disclosed by agent3 (regen segfault = ran from inside results/, fopen NULL) — crash theirs, finding stands, courtesy fix queues behind parity.
- **GATE-SILENCE WATCH**: gemini silent on #general since ~22:0xZ across four high-importance package posts; §7.x = wait-and-surface, not reassign — armed for next wake. Board meanwhile fully busy: agent4 165-syms re-probe GO (hold-time, cite-by-sha), agent5 sweep bounded mid-flight, agent2 parked-ready, watcher daemon confirmed alive. G-AMD-17c terms pre-written (EMTEC256 canonical; agent4 retracted their 'pending word' claim having found the ruling — pre-existing, hygiene win). Merge wave = full gate+golden+anti-res on the registered tree, then t3-wip@90ee070f + agent2@92e2f8f0 land, agent4 flashed, one boot. Wake text rewritten 23:05Z, delivered + DB-VERIFIED (5801 B).

### STATE 2026-09-12 ~22:3xZ (AMD host) — **q3 BOTH RANKS BOOTED (G17b/c — the capacity wall SOLVED under warrant; my Step-2 compliance verified at e2326704) — death moved to FIRST product-code failure: agent3's never-device-invoked gating launch-config; agent5 lane frozen-clean with self-testing verifier; G-AMD-21 scope grew to include the fix**
- **BOOT ACHIEVED — the sentence the sprint was for**: model loaded BOTH ranks in 31.2 s, warmup reached (agent4, G17b/c, window closed 22:30:13Z). The 1 GiB eager-ws question my 17b trace ruling asked, answered by their own VRAM sites: rank0 weights leave 820 MiB, the eager DeviceArena.reserve(1 GiB) at arena.cu:142 IS the OOM site — all my Step-2 pre-conditions met IN ORDER (trace first, then the additive hunk). **I verified e2326704 against my warrant terms by diff-read**: UNSET = byte-identical default ('keeps its exact VRAM ledger' comment), measured basis cited IN-COMMIT with this-machine/this-geometry trace (LITH-conformant), warrant named for the step-0 adjudication (their commit msg says 'step-0 cell WILL show non-empty tp_engine diff — adjudicated, name the warrant in the merge msg' — exactly the pattern). Ledger anatomy resolved IN-ENGINE: the 386 MiB non-weight floor = decoder state 224 + prefix-GDN ckpt 73.4 + buffers; dial-saturated config sits 7982 used / 194 free per card — measured thin, but that's the allocator's verdict, not an estimate's.
- **THE DEATH MOVED DOWNSTREAM — and it's historic**: first failure in PRODUCT CODE all night (everything before was instruments or capacity): bf16_gdn_gating_proj_kernels.cu:390 hipErrorInvalidValue at WARMUP. Agent3's post-exit frame accepted: their GDN TUs were compile-verified (30/30 nm) but NEVER DEVICE-INVOKED — B1..B4 first-launch proof was the queued cell, and launch-config failed, not numerics. Their suspects boarded for the claimant, in their priority: (1) gemv/simt variant requesting kSmemBytes without the attr raise (hip rejects over-cap silently as invalidValue — the funcattr-NotSupported class is BACK as a candidate); (2) the cudaGetDeviceCount loop setting the attr — on a 4-die box a stale/failed attribute on the EXECUTING device is a classic; I verified the loop exists (:328-338); (3) THEIR OWN FLAG, the wave64 caveat: their mma/ldmatrix SIMT emulations assume lane = threadIdx.x&31 with full-mask shuffles — if they execute before validation, the shim's wave64-aware __shfl_sync path must be checked (cuda_runtime.h:498 exists for exactly this). G-AMD-21 SCOPE AMENDED: launch-config fix (custody rules apply — claim first) + oracle regen + B1..B4 parity, tight loop by design (agent4's kit re-fires minutes behind any fix).
- **Agent5 lane FINAL at 6cbf17f5: 34 commits, zero device time, §4 unbroken, WO-07 complete on their side.** Verified by me: their PG-1 RED is origin/amd/main-not-ancestor-of-their-HEAD (my merge-base check — the flagged src files are from MY later merges; their diagnosis 'not mine, base-behind' was RIGHT and my earlier stale-remote cause for the FIRST incident was the wrong instance — §16 rule stands, disclosed both directions). FROZEN per their recommendation: a reference that moves isn't one; merge-forward waits for my coordinated post-registration wave. Their verifier self-tests EVERY run on a sabotage-verified fixture, and the seventh pipe-exit instance (sabotaged run printed EXIT=0 through `head`) WIDENED my lint rule to its final form: NO status read through a pipeline, ever, assign-then-report. Handoff package (6 artifacts) routed to gemini with the Stage-0.5 install decision theirs.
- **Also this pass**: agent4's own self-catches keep landing (9/9 host-suite claim corrected from EXPECTATION to MEASURED after their matrix said 7/9 — pkg-config FFMPEG cause named; 'write the number from the run' now a quoted lane law); #240 identity mess handled: my hub id is 8e05f0b6 (agent4 quoted a nonexistent 402ec3c1 — the mutable-names argument, applied to a value that wasn't a uuid), and my grants now name all three identifier namespaces at stamp time. gemini: the registration package is complete, ref-final (90ee070f), attested twice — the board's critical gate is theirs; serve re-attempts wait on their installs + the G-AMD-21 claimant.

### STATE 2026-09-12 ~22:0xZ (AMD host) — **STRIDE-4 SOLVED — runner's own f32-bits-as-bf16 input prep (instrument #4, ladder-designed, not rocgdb-needed); every product kernel innocent; agent3 exits at context bar, G-AMD-21 (oracle regen + B1..B4 parity) open; agent5's '0ae1f903 retired' claim REFUTED by my ancestry check; agent4 claims the group-integrity cell as per-wave self-check**
- **THE MYSTERY RESOLVED, anticlimactically and completely:** G-AMD-20's v5b replay fed REAL captured partials into the clean single-object reduce launch → stride-4 REPRODUCED ⇒ class (a) data-state, and upstream: q/k/v/q_dec were uploaded as FLOAT32 bit-patterns into BF16-declared tensors (lcg_bf16 zeroes the float low half ⇒ every even bf16 element born 0x0000, odd holds the value). append→cache→partial→reduce were FAITHFUL ALL NIGHT — the 16b canary wall, the 16c trace, the kQH/2=1536 'prefix', the cos-0.99975 de-interleave: one mechanism, the RUNNER. Fix committed (host-side bf16-bit packing, zero product mutation); post-fix: contiguous real bf16, m 18→37 exactly as K-restoration predicts. I verified by their logs AND my own pacc parse (g20_pacc.bin: 12288/12288 even nonzero + 12288/12288 odd = dense post-fix shape); my first read said 'stride-4 False' — VINTAGE collision (post-fix file vs pre-fix claim) disclosed per matched-line law; agent3's ladder beat my in-process/registration hypotheses to it — pre-declaring both branches' next moves is what made it land fast. FOURTH instrument bug tonight (readback shift, canary byte-vs-slot, observer-erase, input-prep) — every device anomaly of the sprint traced to instruments, ZERO to product kernels.
- **SERVE-SAFETY LINE (agent3's, board-verbatim): 'their engine feeds REAL tensors through the product wrapper, NOT the runner's poisoned buffers — this bug cannot exist in serve.'** G-AMD-17's capacity story is unchanged and remains the ONLY serve blocker (ws floor 1 GiB at tp_engine:719; their arena read shows the 1 GiB is a FLOOR, real components sum below it — binding question is what the engine must reserve, agent4's to shape; if no q3 shape fits 2×7.98 the honest fallbacks are named for the USER: measured ws-floor reduction, fixture-class first serve, or 'q3 TP2 capacity-capped' as a verdict). Decode-parity caveat stands: GDN/gating emulations ARE armed in their prefill band → B1..B4 oracles urgent.
- **G-AMD-21 OPEN with agent3 EXITED at the context bar (exit granted):** oracle tables were generated pre-fix against poisoned effective inputs — regeneration vs packed bf16 + full C441-parity re-run + comparator-NaN check, B1..B4 riding same window, dev2 when stamped. Exclusive T3 files (gqa_*_gfx906*, oracle tables) need a NAMED custodian before anyone resumes — a fresh session's first message to me IS the claim. Entry: debrief §5 addendum-3 chain 0176f99f→0568d98e.
- **Agent5's #1 RETIRED-BY-ME:** claim '0ae1f903 not in the history now' is FALSE — my merge-base --is-ancestor returns TRUE at tip 614ea829; their chain listing stops at 20:0xZ = stale remote-tracking ref, procedurally: ANCESTRY claims need fetch-as-step-zero (corrected to them with the rest of their genuinely-verified items 2-3: control-abort negative-tested-for-real, exit-3-withholds-tables — my own STATE citation now tested; cold-start content-verified 9/9). GENERATOR GAP fix ordered (g2_family_table one-command, byte-compare on regen, differences escalate). Their citation audit of themselves (+ my :70-vs-:69 and their own grep-context confessions) = the systematic-off-by-one story; the 3-of-4-matching stale pointer rule is board lore now. #237 (agent4→agent5/gemini): group-integrity cell adopted as agent4's PER-WAVE SELF-CHECK on wo-p3-serve — I verified their witness myself: :69 constexpr Warps (agent4 quoted :70 — my corrected number propagates), sparse_moe kD1Warps=8 inside if(warp==0) = SAFE class, 32%8==0 ✓; serve corpus makes gemini's part-1 cell live-route defense, and 212 real TUs become the false-positive test for it — good news all around.
- **LANES NOW:** agent4 = capacity-shape continuation (fresh mini-stamp when they ask; greedy-only, and their serve CPU layer still owes the 9-test host run behind (a)+(b) generate-break) + agent5 handoff scripts accepted per #237; agent5 = generator fix, then reference-hold; agent3 = EXITING clean; agent2 = parked on gemini lint + hand-back owed; gemini = the registration package is FULLY ATTESTED (agent3 seq-19 authorship confirm + my :79-99/#else byte-preservation check + relocation gotcha named for text-diff arms) — four-file Ruling-1 + mma.cuh verify-arm question + _rn lint + check-(i) per-TU + Stage-2-alt all queued on their lane, which is the board's critical gate.

### STATE 2026-09-13 ~03:1xZ (AMD host) — GEMINI CONDITION MET-BY-DESCRIPTION (8356b8cc: canonical origin/main default, direction-aware predicate [0-reverted/132-forward PASS-with-counts; reverted-or-stale-pattern = FAIL], bare call in check (d), falsifier RED both ways, 11/11 + CI 2/2) — my standing baseline-condition satisfied per their report.
- SUCCESSOR COORDINATOR: absorb it the standing way — full (a)-(k)+golden+anti-res at their tip (cite verified-at sha, banner truth re-checked per the label lesson), then merge the gate family to main; goal MET (17f, G17/G3), registration closed, W1/W2 dev2 window granted, Phase-2 TP4 delta-map open with the artifact-shape question first. 24h sprint: link-green to coherent token in 5h33m, and the state outlives the session by design.
- **STATE ADDENDUM 05:3xZ (correction to the line above, accepted from agent4 then independently re-derived by agent3):** "everything else inherits clean" was **wrong for boot-readiness**: `origin/amd/main` as of `007022c4` is NOT launch-bootable — the unsplit gating launcher still ships Stages=4 (81,920 B dyn-smem vs the 65,536 B measured ceiling), the same launch site 17d died at, now for the smem reason. Full derivation + re-run commands: `docs/amd/T3_UNMERGED_OPEN_ITEMS_2026-09-13.md` on `origin/amd/t3-wip` (`2addef12`). **Ratified sequence for the successor (hard prerequisites, in order):** (1) absorb gate family per the line above; (2) merge agent4's residuals (`amd/wo-p3-serve`, Stages=2 + verify-arm + trace + plan.cpp + ws-warrant — their branch, their commits, one-commit-per-author; agent3 declined folding them, correctly, single-writer is per-lane); (3) THEN the post-sweep boot runs as **17g regression proof** — bar: same token, same coherence, sweep files behavior-unchanged. A boot attempted before (2) is a guaranteed re-kill, not a test. Related open item routed by agent3 to Gemini's lane: hip_shim zero-check zone (`gate_pg1_whitelist.sh` early `return 0`) leaves the hard-coded width-32 shuffle defaults in `cuda_runtime.h` unguarded — exposure, not active bug; controls proposed in `docs/amd/T3_HIPSHIM_BLINDZONE_2026-09-13.md` (`e3464178`).

### STATE 2026-09-13 ~02:5xZ (AMD host) — *** G-AMD-17f: G17/G3 MET — FIRST COHERENT Q3 TP2 SERVE ON 2xV340L ***
- agent4 release row committed+pushed (KIT-EXIT=0; branch resolves on origin): 32/32 completion tokens (finish_reason:length, thinking consumed the budget — content-field sentence awaits a max_tokens-128 parameter window, logged), ZERO faults / ZERO stub-catches / ZERO warmup-fails; runtime arm map: 192x mma.unsplit@cols=51 (the 17d death site, REAL and CLEAN), 192x smallt_gemv, 3264x decode gemv.paired_rows — every step evidenced per trace law. Capacity final: ws96 warrant + ctx256 + kv256 inside 17f arithmetic; 31 s load. Sprint chain 21:22Z link-green -> 02:55Z coherent token: ~5.5 h, four instrument eras, every product kernel exonerated throughout.
- NEXT PER PLAN: tp1-control on ref (fresh stamp), B4 parity window (one normal dev2 stamp), Phase-2 TP4 delta-map (agent4 queue empty-and-willing; artifact-shape question first). Gemini gate family still held by baseline-condition. SUCCESSOR COORDINATOR: the sprint goal is MET; read STATE tail + PLAN + RESTART kits; nothing is broken; the night ends as designed.

### STATE 2026-09-13 ~02:4xZ (AMD host) — **agent2's 6th correction applies to the COORDINATOR'S OWN LEDGER and is fixed in-record: line 'hub 106391' (my 20:2xZ header) mislabels the pi-suffix as a hub-id — correct forms: hub-NAME pi-dual_5060_ti_ninfer-106391 / hub-ID 8e05f0b6 / pi-session 01a09742. Naming policy: never cite 'hub <digits>' again — digits-after-hub are ambiguous by construction between name-suffix and id; the three-form joined pattern (already standing on my grant signatures) is the only citable form. Lane handoff received: 45/45 commits landed, watcher alive pid 2054034, package empty vs main, zero GPU all session, Appendix C (AGENT2_SESSION_DEBRIEF...-late.md) carries the full six-lane identity table for incoming sessions. OPERATIONALLY BANKED FROM THEIR DEBRIEF: Gemini has NO pi session in this repo — intercom cannot resolve them; comm_send by hub-id was rejected; #general broadcast is the ONLY route — all future gemini traffic goes through the hub, no round-trips wasted. agent2 lane: CLOSED, clean, six public corrections, the last one about a label rather than a number — their honest ordering, and the ledger now agrees with it.**

### STATE 2026-09-13 ~02:3xZ (AMD host) — **ROW-CLASS CONTRACT MERGED, FINAL FORM (agent4's measured matrix + agent2's derived predicate = one partition, zero disagreement): per-header rows {hip+shim, hip−shim} universal; g++ row only where closure-predicate says host-parseable; EVERY g++ row carries anti-vacuity proof (deliberate-error-seen or parse side-effects) — the .cu-under-g++ silence trap re-reproduced at coord hand ('measuring g++'s patience'), third floating cell tonight, fixture discipline now law; first-diagnostic taxonomy (file-not-found/platform-#error = N-A-env; undeclared-identifier-in-subject = finding); self-disclosed counterfactual-axis error credited — right experiment, wrong axis, caught in public.**
- fix-that-deletes-the-exception noted (warp.cuh self-inclusion collapses the shim axis downstream). 17f: unchanged, armed, compile alive at 02:28Z. Package/board/gemini-queue now carry ONE contract, not three framings — the night's convergence pattern held to the final post.
### STATE 2026-09-13 ~02:2xZ (AMD host) — **BOARD LAW: 'Could this test have printed the other answer?' — agent2's fifth correction (they caught their own 6/6 predicate demo hardcoding its expected values — a rigged test under a working conclusion), re-derived INDEPENDENTLY by my own BFS at main: 6/6 holds with a third-party instrument. Subsumes the night's gate failures (count gate, one-token arm, vacuous .cu, my all-zero ghosts). Phrasing policy adopted: never 'verified' where it means 'not yet broken'; mutation-test the checks that AGREE with you — agreement is when everyone skips it.**
- Package: zero changes; agent2 corrections chain (5, converging) closed with this one finding-no-defect-in-the-conclusion; their surviving-findings list re-rated per their own downgrade. 17f still on agent4's compile; wake current.
### STATE 2026-09-13 ~02:0xZ (AMD host) — **G-AMD-17e: FIRST LISTENING SERVER. The seam is FIXED and device-verified: 96x mma.unsplit@cols=51 + 96x gemv clean through the 17d death site, both ranks TextContext READY (7102+7102 MB), HTTP live answering schema-perfect 400s. G17 MET in substance; the token is behind one parameter fix (mine, owned: 54+32>64 was stamp arithmetic I authored — kit now ctx 256) and ONE whitelist row (w8_small_t, agent4's own routing-table-name-then-stub ticket-sampling; eligibility compile running). G-AMD-17f ISSUED WRITTEN, effective on row-add landing — the chain's last link is armed, not waiting.**
- Two laws from agent4's release, boarded: dladdr-vs-stub-linked-binary measures the stub map ('symbol-resolves is not arm-is-real' — routing evidence must run stub-free or annotate per-name; their own map amended in place); and ticket-sampling recurs in ONE'S OWN EVIDENCE (table named the arm, bin stubbed it) — the reading-your-own-evidence-with-the-wrong-question failure mode, caught by its author before the token could. Agent5's helper-guard review GREEN-LIT with the helper-routing negative cell as condition — tripwire class now complete at both spellings before gemini's spec; their (b) CUDA-arm duplicate-signature catch credited (fixing HIP by breaking CUDA was live).
- Also this hour: agent2's triple-correction thread CLOSED (their :9-citation 'does not reproduce' was itself ref-set-incomplete — the cite was accurate at pre-wave main, coord-measured; the '0568d98f package defect' is their own e→f transcription, ghost item retired with the cat-file-predicate rule attached; final general law: verification/correction/retraction all owe the SAME ref-set — origin-of-claim plus current-state, never one without the other). Thread parked by my ask; agent2 accepted the discipline visibly.
- Sprint ledger at first-listen: 24h-clock at ~T-15h; link green, capacity warrant live, parity closed both families, seam guard fleet-enforced with negative cells, stubs scored two real catches, one server listening. Whatever 17f prints is the answer the whole structure was for.
### STATE 2026-09-13 ~01:5xZ (AMD host) — **GATE-DESIGN LAW ACCEPTED: g++-bare applicability is DERIVED from the transitive include closure, not enumerated (agent2's predicate reproduced 6/6 at 35bdf211 by me, math.h surfaced as a missed member; hand-lists that match only their motivating files = ticket-sampling inside the gate). Routed to gemini's cell-contract queue with an implementer note bought by my own instrument: probes MUST compile as .cpp/-x c++ — my first run said 0/0/0/0/0/0 because g++ treats .cu as linker input (the vacuous-green trap agent4 documented an hour before I walked into it; confession in, lesson boarded, rc-before-conclusion held again)**
- Records tidied per agent2 #334: ':9 cuda_bf16.h' decay citation was MY 23:0xZ-era text, superseded (coordinates move under merges; content-hashes don't); 'declares' never appears in landed comments — package must not quote a doc-accuracy defect against a clean artifact (agent3's 90ee070f rewrite is the rigorous account, agent4's substance stands).
- Board: 17e in flight agent4-side; gemini's hardened-gate merge held by baseline-condition; their queue now also carries (a) derive-not-list predicate, (b) hip_shim-skip + any-token under-rejection, (c) direction-aware canonical anti-res, (d) W1/W2+FRAGCELL device legs on a later dev2 window, (e) B4-when-scheduled, (f) Stage-0.5 corpus + no-numbers law. Token chain untouched and single-threaded.
### STATE 2026-09-13 ~01:4xZ (AMD host) — **WAVE LANDED: (A)-seam closure merged to amd/main at 3baaf367 (01:41Z) — guard fleet-compiler-enforced, post-merge gate rc=0; G-AMD-17e STAMPED IN SAME MESSAGE, agent4 is the last link: residuals-ancestry -> re-merge -> rebuild/sha -> one boot per new truth -> the token**
- Merge receipts (all coord-measured pre-commit): 007bfa27 remote-pinned; guard = 4 '= delete' arms; sweep = 106 NINFER_LDM_ADDR '+'-lines incl BOTH spellings (inline token + swz_addr helper class the enumerator caught — the compile-refuses-what-memory-forgets law producing its second discovery); fleet invariant 0 raw smem_addr-into-ldmatrix (my grep + agent5's span-scanner agree); negative cell caused-state-asserted (mutation arm cmp-guarded — the assert-the-caused-state law implemented in a lane's first post-incident hour); audit 6/6. Gemini gate-hardening merge STILL HELD by the baseline-condition — separate item, zero token-path dependency (their old arms passed the post-merge run, anti-res line printed WITH its baseline named as the practice expects).
- Provenance of record per agent5's byte-check (#369 correction accepted, merge msg already said it): guard+sweep WORKING BYTES = agent5-in-misexecution (applied ~01:14Z as step one; qwen session held-only, bytes agreed), COMMIT+acceptance+negative-cell = live agent3 session per custody; audit verdicts were re-measured on LANDED bytes, so nothing rests on the authorship question — the incident record stays byte-true anyway because that is the whole currency. Author-of-record for a step = whoever's bytes are in it, stated at merge time.
- Agent4's split proposal formally retracted pre-merge (half-landed-contracts argument accepted); chain reduced to their own protocol: ancestry re-verify vs 3baaf367, re-merge, rebuild-sha, 17e single launch (this hour's stamp, ws=96, trace, throw-site attribution, greedy ≤32, cold-CIFS ~6 min). OUTCOMES PRE-DECLARED: coherent token = G17/G3 MET; same-seam fault = NEW shape, stop-kill-report, dladdr map + sweep audit are the instruments, no reflex re-fire. The board's last unscheduled event is now scheduled and armed; every prior link is measured.
### STATE 2026-09-13 ~01:3xZ (AMD host) — **RULING AMENDED against myself on agent5's four-way repro: gemini's baseline-alignment commit removed the LAST default-path canonical defender (child-bare rc=1→0 at their tip 9d2bcd26); MERGE CONDITION set before that family absorbs; the deeper fix named: PREDIVERGENCE-PREDICATE IS THE DISEASE — divergence-worded check on a legitimately-diverged line is permanent-red, and permanent-red tools get defaulted away (observed live, twice in one night). Direction-aware canonical (reverted-lines FAIL loud, forward-only PASS-with-counts, per the step-0 law's own wording) makes pinning-canonical-default safe and ends the cycle. Containment work praised and separate. Incident #240 (agent5-in-agent3-tree) closed: sweep complete-uncommitted in agent3's tree, guards+115 macro sites+0-violation invariant all coord-verified, disposition = agent3's call per custody, claim-first law boarded (broadcast binds by ADDRESSEE; sender-side addressing error was mine). Agent3 old session's drain-late #333 filed (facts long-banked; 'lane out' = old session only). Agent4 handoff-complete with debrief deltas (w8 7/7 READY-NOW measured post-alias-merge — merge's value now measured not claimed). Fresh agent5: TP4 world==2 first pass b91a36df banked, §6 scoped-not-measured marked, artifact-shape question (agent2) still ranks first.**
- Chain single-threaded now: agent3's sweep commit -> wave (gate-hardening + my condition OR their direction-aware rewrite) -> 17e -> first q3 token. TP4 Phase 2 (delta map) opens the moment the token chain closes or in parallel via agent2/agent5's zero-device inventory work.
### STATE 2026-09-12 ~21:3xZ (AMD host) — **TEST COUNT FINAL: 174 (my own ctest -N), 98/144/145 all retired — agent5 self-retracted twice with the general rule 'a caveat asserting absence-of-the-unsearched is unverified by construction'; dflash2 no-op-pass branch = fourth green-class instance, with my correction of their 'registered twice' framing; serve window first results: rank0 materialized 7102 MB, OOM after (measured, honest)**
- **The number is 174** — authoritative basis: `ctest -N` on a real generate, run BY ME at /tmp/g5cfg (agent4's tree at 10e4a7cf lineage), prints Total Tests: 174. Why every text count was wrong, by agent5's diagnosis (tip 27c31fd8): tests/CMakeLists.txt:381-385 `foreach(op IN LISTS ninfer_op_tests) ninfer_add_op_test(ninfer_${op}_test)` expands 23 op-tests at GENERATE time — no static scan can enumerate them. 98 = add_test LINES (never a count), 144/145 = text walks undercounting by 30. And the meta-finding agent5 filed on themselves: their prior artifact carried the caveat 'a foreach-defined test would be missed; none found' — FALSE, and the caveat is what made the wrong number look checked; RULE: a caveat asserting absence of the thing not searched for is unverified by construction — don't publish numbers whose failure-mode you can't cheaply exclude. Classification over 174: HOST-OK 9 (the serve suite exactly, all link ninfer_serve only, zero .cu — agent4's door already supplies it), 0 language-only-blocked (so LANGUAGE-HIP fixes buy ZERO alone — every .cu test also links a missing target; the only lever is target ALIASES for ninfer_core(84)/ninfer_ops(73), with agent5's own untested-flag honored: alias may clear generate and still die at link symbol-closure, one build away, nobody plans on it till measured).
- **Two test-file findings, one corrected by my bytes:** (a) agent5's 'dflash2_symbols_check registered TWICE' — WRONG as framed: :816-826 is if(NM_TOOL)/else, EXCLUSIVE, exactly one add_test ever exists (my 174 proves it). The REAL shape is subtler and I restated it: same NAME either branch, so the suite looks identical while the check's CONTENT flips from symbol-verification to `cmake -E true` — a green row whose meaning depends on a tool's presence, invisible to count-audits. (b) the else-branch itself: a DELIBERATELY-PASSING no-op (WARNING-tagged) = tonight's fourth green-that-did-not-run and the first authored on purpose — pattern named for gemini's Stage design (missing-tool cases must skip-nonzero or fail, never pass). Routed with (a)'s correction both ways so nobody re-reports 'double registration'.
- **G-AMD-17 first real results, read by me from agent4's release rows (their own, in-window, unfinalized):** attempt-2 (21:29:51Z): freshness head 004609fa, bin 6f75d807…, magic+manifest OK, **'=129 armed as stamped'** (my q3-file arms prediction held at first contact), rank0 materialized 7102 MB in 16 s (CIFS warm off agent2's verification pass), preflight PASSED on measured basis (required 10318/usable 16310 MiB both ranks), then hipErrorOutOfMemory post-materialize with RANK 1 never logging. That is VRAM LAW working: no estimate refused anything, the allocator spoke in real time, ~13.7 GB device-total vs 15.96 GiB box-total = KV/workspace-shape territory — agent4's next attempts own the reduction (kv-capacity/context/ws are their stamped flags). Stride-4 decode question untouched (never got to decode). Window status/ownership: agent4's, open, dev0/1; my stamp's stop-conditions apply.
- **Pinger-board truth at 21:3xZ:** agent2 re-briefed on their new session (full handover with shas, per their sha-before-claim rule); agent3 G-AMD-18 granted dev2 (PC-rederivation edit in force); agent5 on (ii) dominator pass; 144/145/98 dead — cite 174 or cite the line-count AS a line-count.

### STATE 2026-09-12 ~21:3xZ (AMD host) — **G-AMD-17 SERVE WINDOW OPEN (dev0,1, q3 TP2, 21:24:28Z) — LINK-GREEN measured by ME before their report arrived; G-AMD-18 granted dev2 with PC-rederivation edit; agent5's tests-generate inventory verified 72=72, zero serve-blocking; header thread closed (echo #218 second drain)**
- **Link-green, coord-measured at 21:22Z before agent4's report landed (c6de95a4 'FULL LINK GREEN rc=0, parity 212/212' confirmed independently):** 16.8 MB ninfer-serve, `T main`, zero `ninfer::` undefined (370 U = libc/ffmpeg/hip imports, matched-line checked), 97 decode-attention + 68 GDN real-text symbols — the 'stubbed-live-route ≠ green' policy satisfied by measurement, GDN real at last. My own probe-order errors during this check (hit cuda_pipeline/memory.cuh absence errors in SYNTHETIC TUs, real-chain clean) disclosed to agent2 verbatim — read error paths, not totals.
- **G-AMD-17 stamped 21:2xZ (written = my intercom, window opened 21:24:28Z, agent4's kit-fix commit 8be8f0f2 in-window is theirs: G17_* artifact freshness-check exclusion + HVD=0,1 — binary rebuilt 21:23Z PRE-window, freshness-by-sha owed in report):** q3 TP2 first serve, ART /media exact-pin checks (no re-hash, CIFS law), 1200 s/launch, VRAM LAW recited (~7.72 GB/card vs 7.98 — allocator is the only gate, clean cudaMalloc refusal = RESULT), own-pid, release rows. Outcomes pre-declared: boot+coherent-prefill = G17 milestone even if decode reproduces stride-4 (then stop-kill-report, G-AMD-18 owns the mystery); =129 ARMED expected (q3 file, no NOTE exemption here — NOTE-branch is the ref/tp1 case, next stamp ask).
- **G-AMD-18 GRANTED to agent3 dev2 (≤15 min, A-before-B) with the confound edit my review caught:** their plan hard-arms BP at old-binary PC 0xD7D50, but Cell A rebuilds — offsets move — and BP-never-hits is their 'executed≠fatbin' branch; without re-deriving the store PC from the NEW binary (formal step, old-vs-new printed side by side) branch 3 could fire trivially and lie. Otherwise matrix accepted pre-fixed; B4 chunked pre-declared DRIFT-CHARACTERIZING (bars informational) is the anti-post-hoc shape I required; asked for cheap bonus datum: does stride-4 even survive the merge (if it VANISHES, window returns early — honest either way). Two serve windows running, disjoint pairs (0,1 serve / 2 debugger), dev3 spare, pre-checks at 21:27Z: no KFD, baselines clean.
- **Agent5's tests-inventory (b12c3bc8) verified by my own fresh generate at their cited head: 72 CMake Errors = their 72, taxonomy matches (25 link-language on .cu test sources = silent-drop class arriving in tests/ where nothing guards it; 21 CUDA::cudart; 1 add_test genex; 'Generating done' printing BEFORE 'Generate step failed' — their own trap, confirmed live), and ZERO errors touch the 9 serve tests — (b) alone gated them and (b) is DONE.** My seq-20 sequence-ruling was stale in a good way: agent4 landed (a)+(b) in-lane (10e4a7cf) citing agent5's curl receipts; my 'half-unlock breaks opt-in generate' concern became measured fact — now routed as agent4's second-edit question (opt2 subset-door vs opt1 full) + gemini Stage-2-alt input. Approved agent5's (i): per-test 144→targets table, feeds both. Housekeeping: they flagged MY /tmp verification worktree (t3wip-v2) — removed after its merge-pass verdict landed, and their own double self-catch (textual-grep 'targets defined' missed the :14 return(); 'Generating done' read as success) is the greens-that-ran-nothing class catching even its auditor.
- **Agent5 per-test follow-through (e2bdbedd + e664c011): exactly 9 of the walked declarations build HIP-side today — and they are precisely the serve CPU surface** (9 names, each links ONLY ninfer_serve, zero .cu sources, hand-verified per-test and independently converged with their per-target count): tonight's serve evidence needs a 9-test subset door, not a 136-test porting project — the answer to agent4's second edit, scoped small. Their discipline notes adopted: refused to ship the 7:1-undercounting 136 table (regex stopped at first ')'), ships the trusted 9 + names the pending re-run; 'HOST-OK ≠ passes ≠ device-free at runtime' with the four-count rule; and their third self-caught 'waiting on X — X already shipped' recurrence producing the procedural rule **sha before claim, check the branch not the board** (mine-verified: their 08-spec (c) fix cites real commits, and I confirmed the same drift live when their #218 re-drain anchored on 0ae1f903 while 0176f99f had already fixed B on the lane). One open numeric: 145 walked declarations vs 144 distinct-names rule — reconciliation ordered into their 136 re-run; board quotes ONE number with ONE rule after. (ii) real-CFG dominator pass approved next ('proved boundary vs claimed boundary'), and their §13c predicate taxonomy relayed to agent3 as free static input to the stride-4 hunt — row-vs-lane index confusion is one live hypothesis and reading costs nothing.
- **Hub #218 second drain: echo identified (20:49:42 first pass, created_at discriminates), content already closed by measurements — no lane action; thread-closure post was mine, stands.** t3-wip merge-gate pass at 0176f99f ran gate rc=1: check-(a) RED on exactly 4 unregistered pre-existing-file mods (shim cudaLaunchKernelEx loud-refuse / embed_gather warp.cuh include / gdn __grid_constant__ pair) — all legitimate, all verified by my diff-reads, all waiting on GEMINI's Ruling-1 registrations (high-priority #general ask sent with per-file evidence; mma.cuh family-bless status queried too). Board truth: fix-in-flight means exactly this — main stays gated until the exception pairs land.

### STATE 2026-09-12 ~21:1xZ (AMD host) — **agent5 corrects MY host-test ruling with receipts — 'a passing configure that registered ZERO tests' is tonight's purest green-that-did-not-run; header class grows to embed_gather.__shfl_sync (5 errors, my repro); agent5 re-ran my g2c probe claim independently**
- **Host-test suite: two walls, I saw one.** My 21:0xZ ruling ('one recipe-block from the host ctest run') was WRONG in sequence: my root-free libcurl enablement (apt-get download + dpkg-deb -x + .pc prefix AND includedir rewrites + gnutls .so symlink, configure rc=0 — mine) satisfied the near wall, and agent5, running my path independently (fef25bc8), found `ctest -N` => **Total Tests: 0** even with BUILD_TESTING=ON, because add_subdirectory(tests) sits in the CUDA else() at CMakeLists.txt:121-123 — tests/ is absent from the HIP build graph at ANY testing value. Verified by me: my own 'successful' /tmp/testcfg-probe prints 0 and has no tests/ subdir. Symmetric error disclosed both directions: their 5c543b12 'blocked on libcurl' and my 'one block away' were the same inference from opposite ends of a configure. RULING: (a) hip-branch tests-entry (ADDITIVE duplicate, not hoist — shared-file law) + (b) ninfer_serve STATIC exposure (agent4's exclusive, ask live with them) land IN THE SAME WINDOW — a lone (a) makes generate FAIL on missing targets and breaks other lanes' opt-in config; fallback if agent4 punts: run against their branch locally, main stays clean. Numbers final: 98 = add_test LINES, 144 = registered names WITH the counting rule shipped in-script (my 152-citation retired); SKIP-77 four-count reporting = law; their CUDA-control honesty (no toolkit ⇒ branch-structure read is the evidence, say so plainly) adopted.
- **Header self-sufficiency class, third and fourth instances, all my own bytes:** agent2 hub #216 claimed embed_gather.cuh bare-include gives 5 errors — REPRODUCED EXACTLY at 1b991bcf (2× __forceinline__ from q3_rowsplit_storage.h:42/:49 + 3× undeclared __shfl_sync at embed_gather.cuh:136/:182/:275); via-chain control = 0, confirming their 'green build proves first-include accident, not header health' diagnosis (embed_gather.cu line-2 .h drags the shim ahead of line-5's .cuh). NEW wrinkle vs the storage-header fix: __shfl_sync lives in OUR shim cuda_runtime.h:466 as a template — no hip system header provides it, so the repair is include-what-declares-it (warp.cuh includes it at :3), and I warned agent3 off hand-defining a second __shfl_sync (shadow-collision history). Batch grown into agent3's GDN merge with per-header bare-include deletion gates (agent2's five-config set as the pass condition). Standing for the permanence record: -Wmacro-redefined catches REDEFINITION, never ABSENCE; every shipped header needs a bare-include compile row naming its ENTRY PATH — chain rows can only prove chain order (gemini-lane design, routed through agent5's v340l/08 exhibits, now three independent instances).
- **agent5's #215 group-integrity claims: re-ran their probe myself** (SAFE warp==0 → granularity 32 GROUP-INTACT exit 0 / UNSAFE lane<3 → SPLIT-GROUP exit 4 / real blk256+blk512 shapes GROUP-INTACT) + source-read confirmed the constexpr-Warps sites they name (warp.cuh:70/80, rmsnorm.cuh:153/199, sparse_moe_decode_kernels.cu:82). Posted my independent reproduction to hub #general so gemini's cell design rests on coord-witnessed measurements, not prose — refutation-by-re-run explicitly welcomed. Three-state exit restated as board policy there with the green-on-blindness tally (check-(e), Stage-0 count, the count-gate that passes the hazard, MY no-op-sed test cell).
- **agent5 further:** §12 whole-TU sweep (14 whitelisted device TUs, 0 hazards after killing 2 FALSE hazards their own detector made — single-lane ds_read/ds_write miscounted as wavefull; the 114/114 device contradiction indicted the instrument — 'static claim vs device result ⇒ re-derive' rule generalized); own-error disclosure: over-claimed 4 uncovered serve TUs, grep corrected to 1 orphan (responses_http.cpp) + generation_service indirect-only — the gap named that a serve stamp can't fill; backtick-in-echo footer caught a would-be command substitution inside a receipt script (third harness-reports-on-itself instance; lint rule: receipts single-quote unless intentional expansion). Zero device time all session, §4 clean per commit.
- **Lanes at 21:1xZ:** agent4 merged main into wo-p3-serve (7327a68b), compiles running, (b) ask live; agent3 GDN batch = 30 syms + storage headers + embed_gather + __shfl_sync caution; agent2 tip 73588693 parked on gemini _rn lint; gemini asks: _rn lint (gates agent2 merge), check-(i) per-TU scope (v340l/09), group-integrity two-part cell (#215 + my witness), Stage-2-alt host-ctest design, header bare-include rows. No card claims open; all device windows closed, rows released.

### STATE 2026-09-12 ~21:0xZ (AMD host) — **T3-WIP MERGED (4cc04b6f, my gate-run verification); BOX EXONERATED by the observer-erase catch — my source-read, agent3 executed; probe hour-guard shipped after MY OWN echo proved the day-guard insufficient; agent5's CI audit: 3 advertised stages never existed, 98 tests run by nobody**
- **G-AMD-16d-FIXED = box innocent, my ruling was right before the run:** probe v4 (7ec4f42e sha, agent3 ran, 20:58:47Z, ~3.5 of ≤10 min used): P1 64/64 clean, **P2 full reduce-store replica 3072/3072 CLEAN shorts, zero interleave**, P3/P4 256/256. The v1/v2 'dead launches' were the probe's own dump() cudaMemset-ing the canary BETWEEN launch and readback — I found it by reading the committed source at 984f4542 after agent3's STOP-ask (their stop discipline correct, mine = don't weigh box-level theories before reading the instrument: THREE instrument bugs in one lane tonight, all self-or-coord caught; observer rule now in the lane debrief verbatim). **The stride-4 anomaly stands OPEN and frozen-on-disk** — every isolable component innocent; suspect = the real kernel's surrounding structure (smem reduction/partial reads); G-AMD-18 rocgdb watchpoint (PC→code-object mapping) is the named next discriminator, queued AFTER agent3's GDN landing (their ordering call, unobjected — agent4's link needs the 30 symbols more than the mystery needs sleep).
- **MERGE of amd/t3-wip done my way: verify-then-land on tip 4cc04b6f** — throwaway worktree: PG-1 gate (a)-(k) rc=0 (check (k) GQA geometry contracts green on THEIR merged tree), shfl golden GREEN, step-0/anti-res tp_engine+tp2_budget ZERO-line vs origin/main (68-line noise was stale local `main` — read the matched baseline, disclosed). §1 proof: 36 files ALL results/docs, zero src/apps/tools-ops/tests. Fast-forward (agent3's bf9ecf49 rebase onto 0ae1f903 = clean ancestry; my merge-msg dissolved in the ff — content identical, noted honestly). Collision handled by rule: stray untracked g16d_rerun_window_end.txt (pre-handoff dead-process residue) → /tmp/coord-collision-quarantine, NOT deleted.
- **Pinger lesson burned in AGAIN, now structurally fixed:** my 20:23:55Z verification test delivered the FIRST board version and I woke on it as #204 at 20:51Z — day-granularity staleness guard passed it, exactly the #197/#198 class. Hour-guard shipped: FAST-FACTS stamp must carry HH:MM AND be within 3 h of the amd/main tip commit time (git %ci, fail-open only if git unreadable, day-check still hard). Proved in BOTH directions — and my first CELL-B 'proof' was a silent no-op sed (stale anchor), caught only because I re-ran with grep-asserted content: a test that didn't change the input tested nothing; assert-before-claim applies to my own sed's too. Fixed en route: wake.mjs was refusing idle-status claims (pi flips hub status idle between turns — legitimate coordinator wakes were being suppressed at 20:51-20:52; claim-BY-NAME is the routing law, status now only rejects offline).
- **agent5's run_ci_amd.sh audit (my own greps confirm F2+F3):** stages '2 (PG-A/B), 3 (PG-C/D), 4 (PG-E)' are advertised at :160 — PG-D appears NOWHERE else, stage 2 only refuses; 98 add_test()s vs a test-count guard whose ONLY call site is the falsifier branch (:143-145) with a deliberate non-matching label AND BUILD_TESTING=OFF — the guard guards a call that doesn't exist. F5 board language ADOPTED: 'CI green' ≠ 'golden GREEN' ≠ 'gate rc=0' — different instruments, neither implies another; my merge msg now names which instrument I actually ran. agent5 redirected (my ruling): inventory the 98 tests, classify host-only vs device, RUN the host-only set at zero-GPU — that's the ONLY pre-existing evidence on the serve CPU layer (schemas/tool_parser/translate ~1.5k LOC, first serve hours away); run+report only, tests/** is gemini's lane. Then back to dominator-CFG work (item 3) with the G2 table already definitive (0 hazard/6 unresolved, exit-5, width-independent, SUPERSEDED-not-deleted).
- **agent4 post-R1/R2 in flight; agent2 hand-back summary owed (context spent, holding); gemini _rn-lint ask open — agent2's merge parks there; agent3: GDN/attention landing next, HipSources append-order coordination with agent4 (single-writer rule theirs).** Serve ETA board unchanged: link-half ~21:30-21:45Z if their wave holds; first-serve-attempt window 23:00Z±, functional-decode gated on the stride-4 mystery (G-AMD-18), deadline T-17 h.

### STATE 2026-09-12 ~20:3xZ (AMD host) — **agent3's g16b canary map CONFIRMED by my own parse; G-AMD-16c granted with the RIGHT trace tool (their HIP_API_TRACE doesn't exist — rocprofv3 does); agent2's gap inventory landed crossed-with agent4's fix — collision ruled clean; gemini asked for _rn arithmetic lint pre-merge**
- **G-AMD-16b verified, not adopted:** parsed g16b_decode_raw_p0.bin myself — region A (bytes 0..6143): 0/1536 even slots nonzero + ZERO canary survivors ⇒ low halves EXPLICITLY written; region B: 3072/3072 0xAAAA survivors ⇒ heads 6..11 written NOWHERE (no OOB — my :229 12 KB-allocation correction from the previous block holds, agent3 adopted it). Values = correct outputs for k<1536. Their pass1 canary-detector byte-vs-slot bug self-reported and re-verified offline — honest shape, noted. ADDED COORD DATUM: {0x0000 low, bf16 high} dwords ARE valid f32 bit-patterns of the bf16-rounded values — so the writer is plausibly a real FLOAT-store path writing a 1536-element f32-strided prefix; three candidates named for the trace to discriminate (f32-stride kernel launch / D2D copy from f32 staging / dispatched code object ≠ carved code object — the mapping-≠success class).
- **G-AMD-16c GRANTED (written, intercom) — with a tool correction:** `HIP_API_TRACE=1` is NOT a knob in libamdhip64 (I strings'd: only HIP_TRACE_API, legacy/undocumented); rocprofv3 EXISTS at /opt/rocm-6.2.0/bin (kernel-trace + memory-copy-trace + hip-runtime-trace flags verified via --help). Grant specifies the rocprofv3 command line, same binary sha 4da6e9b8 (re-verified on disk), dev0, ≤90 s, release row at window_end. Device budget bookkeeping named to them: ~3 min of ≤10 min used; extend by ask, not silently.
- **agent2 e32da2e4 gap inventory (v340l/07_hip_serve_gap_inventory.md) verified line-by-line** (curl.h absent = my own find, return() at src/CMakeLists :9-15 = my read) — but CROSSED WITH agent4's 04aeae36 (15:13Z), which already solves BOTH blockers (NINFER_HIP_BUILD_SERVE option + backend-branched apps/CMakeLists + hip_media_acquire_stub.cpp; my tree check: 17 .o under ninfer-serve.dir). Ruled: agent2 = second witness, NOT competing CMake (apps wiring is agent4's exclusive); agent2's 4-file whitelist proposal ROUTED as census-to-agent4 to keep HipSources.cmake single-writer; agent2 next = HostKVArena::~HostKVArena root-cause (their own open item, shim-conditional-definition smell).
- **STALE-BASE HAZARD measured on agent2's branch:** diff 2ae7631a..origin/amd/wo-shim-funcattr shows gate_pg1_whitelist.sh −98 lines — NOT agent2's edit (per-commit stat: e32da2e4 touches 5 files, zero gate files) — their BASE (bcd25fb2) predates gemini's check-(e)-upgraded/(k) landings. Rebase-onto-main + 12+4 re-run ordered before merge; main's gate file wins by law. Their D3 self-flag (__hsub2_rn alias = ARITHMETIC on the D3 surface, header itself says 'nothing checks intent') routed to GEMINI (hub #general, per mesh): extend check-(e) symbol lint from __hadd2-only (:388) to _rn arithmetic spellings as bless-or-block BEFORE agent2's merge; also offered agent3's stride-4 anomaly as fingerprint-corpus material ('correct code object, wrong memory' permanence gate).
- **agent4 briefed:** second-witness doc + 20-symbol whitelist handoff + HostKVArena lead + 900 s-timeout note (resume the tree, don't clean-rebuild — disk) + G2 stamp prerequisites restated (q3 on /media VERIFIED, no re-hash, ref-has-no-Q3G64 NOTE-branch expectation, dev1 free while agent3 is on dev0). Their G1 census (295 lines) is the FULL serve-stack undefined surface — bigger than agent2's minimal-4-frame; both true at different scopes, agent4's is the link ground truth.
- **agent5 S1→S3 progressing cleanly, self-correcting in public:** 05ba67e1 retracts their 'no sub-group reduce in build' claim (constexpr width hid it from the digit-only regex — 28 sites figure replaced by 72-across-18-files), 55d7dcf5 RETRACTS their own proposed gate killed by its own negative control. Census doc v340l/07_gfx900_permanence_census_agent5.md §3a is the corrected record. Review pending my read of the G2 tripwires tranche; NOT merged, no gate-file claims on their branch touched.

### STATE 2026-09-12 ~20:2xZ (AMD host) — **NEW COORDINATOR OPENING (session 01a09742, hub 106391): cold-start done on verified facts, G-AMD-16b granted, pinger hardening from the C441 promise-list SHIPPED+TESTED**
- **Cold start executed as handed:** claim file rewritten to `pi-dual_5060_ti_ninfer-106391` (verified = host pid 106391 = my session `01a09742` before claiming — the name is not decorative, it routes alerts). Handoff + newest STATE read; branch map re-measured, not trusted: `amd/main` 2896e08b pushed, `amd/t3-wip` 0bbce5b8 (agent3's push — confirmed by ls-remote), `amd/wo-p3-serve` 04aeae36, `amd/wo-gfx900-perm` 514a55b7, agent2 lane closed (2ae7631a merge verified in log). C441's kill-promise (#4): all tracked reader PIDs (p3-p6, q3_sha256, ref_prefix_sha) verified DEAD — nothing to kill, no third CIFS reader exists.
- **Agent3's G-AMD-16 report processed with my own measurements, not their prose alone:** (a) the stride-4 signature is REAL — I parsed g16_decode_out.f32 myself: 0/3072 even slots nonzero, 1,536/3072 odd slots, values 1.41–1.59 healthy (no denormals — readback fix confirmed working, run3 zeros = pre-fix residue, partials NOT carrying the bug: their claims (a) and the partials question both independently verified); (b) **their OOB framing was WRONG and mine corrects it:** `d_out_dec` is cudaMalloc'd at **kQH·kD·4 = 12 KB** (runner :229) — the f32 footprint — so the pattern ending at slot 3071 (byte 6142) is INSIDE the allocation; no head wrote past anything, the live question is purely WHO writes stride-4 when both carved code objects store 2-byte. Allocation asymmetry (decode 12 KB vs prefill 6 KB at :349) flagged to them as a candidate differential. (c) Their prefill-evidence doubt answered from the code: prefill readback used the bf16-sized buffer and its g15 log shows contiguous got≈want (no zeros) — I ruled the old raw-dump-layout question non-blocking for the window.
- **G-AMD-16b AMENDMENT GRANTED in writing** (intercom 20:2xZ): g16b binary sha 4da6e9b8… verified ON DISK BY ME pre-grant; tree diff 0bbce5b8→worktree verified instrumentation-only (+38/−13, all in results/amd/t3_device_runner.cu, zero product-path lines) — grant-before-device held. dev0, 90 s timeout, release row owed.
- **Pinger hardening (the owed first-wake task) SHIPPED + 4 cells tested 20:23Z:** (1) stale-FAST-FACTS date → REFUSED (the #197/#198 class: right sender, hours-old content — a state file that can't self-date doesn't get delivered); (2) missing date stamp → REFUSED (fail-loud on un-provable freshness); (3) forged coord-authored header → REFUSED (pre-existing guard re-verified); (4) real fresh text → delivered AND **DB read-back VERIFIED** (`delivery VERIFIED (db-signed <pinger id> -> <claim target>, N B byte-identical)`) — sender integrity now comes from the DB row, not the header, closing C441's named gap. Wake text rewritten to my live board (the rewrite-at-phase-change rule the stale wakes burned for).
- **Board hygiene:** agent4's first `ninfer-serve` link attempt exited inside its 900 s timeout with ~4 .o — asked whether census-as-planned or timeout-starved (their call to answer; no device conflict either way). Card state measured at grant time: No KFD PIDs, GPU0 239,824,896 B with ZERO KFD processes = display-carve noise (measured twice, named so agent3 doesn't abort on a false foreign-context read); dev1-3 at 18,575,360 B baseline. Disk 28 G / 208 G.

### STATE 2026-09-12 ~20:0xZ (AMD host) — **GEMINI BACK (gate package merged, 9-exception check (a) clean); agent2's stub alarm was ITS OWN fixture — q3 sha MATCH twice-witnessed, =129-at-rest; ref-salvage GO; my board-read errors of the hour corrected**
- **gemini returned with the whole gate ask landed** (2e1a98e7, merged): PG-1 exceptions for one_shot .h pair + T3 files, q3_ci conditional tp2 MODE with ART-guarded =129 preserved (agent2's §4 cost adopted over the verbatim port), checks (e)-(j) + anti-res PASS on my own merged-tree re-run. Their silence since 15:00Z was routing (ACKs #127/#155 went to my old session id — my 'lane dark' read was wrong, said so). Remaining owed: D3-fingerprint UPGRADE (my __hadd2 receipts: 0-errors, 0-__ocml lowering), never-launch-vs-wrote-neutral cell, TG1 parity consumption at P3.
- **The q3 alarm was a false fire, self-inflicted both sides:** the 'html stub' was agent2's OWN synthetic calibration fixture; their direct-correction-to-me beat my propagation to anyone else. **GATE CLOSED: sha256 7f26a0eb… MATCH at exact pin size, manifest parses, Q3G64_F16S=129 derived at rest — two agreeing full-file passes (agent2 385 s @ 40 MB/s + my 52%-checkpoint pass).** P3 no longer waits on verification; my redundant sha re-launch is now superseded (kill-if-alive rule handed to successor).
- **Ref salvage ruled GO on agent2's prefix proof** (head-c-to-pin = registry sha; the over-length double-segment was created by MY pause/restart; truncate = theirs to run, no resume — two-writer class). ecd7fd60 (hostalloc template-asymmetry root cause — beats agent3's AND my hipHostAlloc-EXISTS half; template companion :9226 vs plain :3780, int**-no-convert is the mechanism) → rebase onto gate-merged main, merged-base cells re-run, then I merge; addendum-delete list to agent3 under half-life.
- Fresh faults recorded (mine, all self-caught this hour): stale cached size → false 'ref COMPLETE 101%'; grep-without-path → 'hipHostAlloc missing' half-true; stub warning adopted from a fixture read. The 3-lane correction chain (each caught another's error within minutes) is the system working; held as the merge-gate habit, now written into the handoff for the successor.
- **HANDOFF FILE WRITTEN: docs/amd/HANDOFF_2026-09-12-20Z.md** (user requested new coordinator session): live branch map (6 refs incl agent4/5 worktrees), roster+mesh, pinger takeover procedure (claim-file rewrite FIRST), 5 in-flight promises, today's measured-facts delta, the rules that held. Successor reads that block + STATE above; nothing depends on this session surviving. T-18:0x, five lanes live, cards idle, q3 VERIFIED on disk — handoff is clean.

### STATE 2026-09-12 ~19:5xZ (AMD host) — **SPRINT DOUBLES: user staffs agent4 + agent5 (T-19:2x, "finish ASAP"); WO-06/WO-07 issued with dedicated worktrees; my own board errors of the last hour corrected in-place**
- **New lanes, both on the critical path:** **agent4 = WO-06 P3 SERVE BRING-UP** (worktree amd-wo-p3-serve @ amd/main; serve-target hip-link → q3 stamped serve → reference tp1 CONTROL table — the sprint's last unowned step; §5 inputs = agent2's salvage, my sha verdict en route). **agent5 = WO-07 GFX900 PERMANENCE** (worktree amd-wo-gfx900-perm @ t3-wip b91d3bf0; the 12 unwhitelisted shuffle sites + 28 sub-group invariant made gate-grade with negative controls + the three-mode build-cell spec for gemini — regression defense is what the user staffed them for, and it's the lane that keeps tonight's fixes from rotting).
- **My corrections this hour, recorded not hidden:** (a) 'ref-pull COMPLETE 101%' — the OVER-LENGTH class agent2's watcher flagged in advance: file is 20,804,481,536 B vs pin 20,437,336,576, +367 MB = the 18:36 pause + my 19:2x restart double-segment; salvage (truncate-to-pin, resume -C -, verify 0634abb0) handed to agent2 with the receipts trail; (b) my pause of their sha (238 MB/s contention theory) was wrong-cause right-action — the share itself runs 35–52 MB/s, sha relaunched and at 10/14.7 GB; (c) three grep/sed syntax errors in my own verification commands (the '14 site' kvarn count, a bad sed range) — all reported as unverified before agent3's clean numbers superseded them; read-the-matched-line applies to my probes most, they're the ones nobody reviews.
- **Live evidence on t3-wip (agent3's, read by me):** G-AMD-15 follow-up — pos fix alone insufficient (kernel runs, splits=8, STILL zero visible keys: m=-inf/l=0); their prime suspect = cache layout contract (upstream append writes bf16-V, donor kernel reads fp16-V via bf16x8), my fresh-eyes note queued: PREFILL VALIDATED at quant scale with the same append/cast argues against a wholesale layout break — decode walk/width semantics remain the differential. Their next block is the walk read; I've given the structural contradiction to sharpen it.
- **Gates current:** anti-res PASS, golden GREEN, step-0 zero-diff; merge-gate practice held (every merge re-verifies). Cards idle: T3 needs its own mini-stamp when the root cause names, P3 needs the sha verdict + my grant.
- Clock: T-18:7x. Lanes: 3 product (agent3 decode-block, agent4 serve-link NOW, agent5 permanence-census NOW), agent2 artifact pipeline (salvage+watcher truth), gemini 3 gate items un-ACKed since 15:00Z — next wake without movement is a user-surface event per §7.x (wait, don't reassign).
### STATE 2026-09-12 ~17:4xZ (AMD host) — **TB2 map MERGED and it retires a plan-phase; P2's 'on Green's landing' gate is OBSOLETE (landing fully absorbed — merge-base proof); agent2's §4 beats agent3's verbatim-port ruling on one assertion; T3 in G-AMD-14 window**
- **Containment proof, not vibes:** merge-base(origin/main, amd/main) == 4575ac1b == the landing sha ⇒ every Green commit is ours; residual = d9acffea only (CI guard, empty conflict class — BUT its check_shared_artifact_collision can fail-red red commits emitting untagged dump paths from compare/cross/join/b4-named files: warned to gemini before a battery commit bounces). Anyone still scheduling P2-on-landing is scheduling against a stale board — corrected across lanes.
- **Provenance audit of our own receipts (agent2, adopted as a rule):** artifact/format contract moved in the promote — reader/storage_layouts/typed_binding/embed_gather changed; the four live AMD receipts all postdate f5c6d5b6 BY COMMIT DATE (they checked, not assumed): golden re-runs NOT invalidated, but the rule stands — check the base sha in the receipt header, not mtime. Pinned into the merge-gate practice.
- **§4 reconciliation (mine, routed as ONE decision to gemini):** agent3's 'port the MODE branch verbatim, zero coverage loss' is true of the branch shape; agent2's follow-up proves the verbatim port DROPS the ART-guarded Q3G64_F16S=129 assertion (3-of-129 partial load passes). Implementable answer = conditional port + =129 re-add under ART guard. Both lanes right at different scopes; nobody loses face, the gate gains teeth.
- **Task-1 (funcattr) landed merged:** SFINAE wrapper with measured rejection of both obvious shapes + enum-value verification + the BEHAVIOUR note that reclassified my schedule (mapping ≠ success; hipFuncSetAttribute NotSupported class) — now in T3's slot expectations and durable-shim routed.
- Board: agent3 IN G-AMD-14 (≤10 min, dev0+dev1 — window opened ~17:38Z); agent2 owns artifact pipeline (ref-pull verify-on-complete + Q3 v1a fetcher when Green answers ask-4 + fetcher-hardening doc if neither); gemini: consolidated decision-package + priority-1 .h-pair gate fix outstanding. amd/main de7267b2-merged, golden GREEN post-merge, disk 50G, pull ~25%+. T-20:4x.
### STATE 2026-09-12 ~17:3xZ (AMD host) — **T3 prefill adoption VERIFIED (donor hashes, helper relocation honest — my two '0-call-site' reads were my own grep escaping); step-0 FAIL root-caused to a MOVED BASELINE (5th sync), merged, build rc=0; my mma-claim corrected by facts**

- **step-0 scare, fully resolved and worth the cautionary record:** gate listed tp2_budget.h as a divergence — looked exactly like the resurrection class this law exists for. Root cause (git log -S + lineage checks): **Green MOVED THE BASELINE** — 7a83cae1 added the LEGACY-REF-CALIB comment + preflight-region moves *after* my 4575ac1b sync; my tree legitimately predated them. AGENTS.md's own words apply: lagging branch trips the cell; merge fixes, excusing never. Synced (5th in 4h — cite-by-name is doing real work), check_anti_resurrection PASS, full real build rc=0 on merged tree, golden GREEN, new Green test censused 0-error. Also caught mid-flight: my own 15:4xZ 'step-0 OK' echo was a grep -c artifact (`echo EXIT=$?` after `grep -c` always prints 0) — the PASS above is the script's own verdict, not my pattern-match.
- **T3 prefill @ 88eeba93 verified by me:** donor prefill header sha256 a0478a8ab73c == theirs (verbatim claim true); math.cuh carries all three helpers with the exact/truncation-class framing; adopted header's :180 call resolves through them, zero local definitions — deletion honest. My intermediate '0 call sites' reads were grep escaping (`\|` in single-arg patterns) — third instrument-error family of the day, all disclosed, none acted on.
- **My mma.cuh claim corrected by agent3's own work:** my 'only build-reachable consumers are the two w8 HELD files' was WRONG — attention decode/prefill kernels call mma_bf16 via gqa_attention_decode.cuh:10. What survives: their full-family guard made the base-include harmless (why their build reached rc=0), and their scope call beat my narrower pre-empt on facts. Recorded as theirs.
- **Bases moving under everyone:** t3-wip built on pre-7a83cae1 tree — instructed to re-sync before CPU-oracle work and pin merged shas in the slot request (their tables must describe the artifact they'll actually run). agent2's funcattr branch same notice + choice-of-next-task routed (q3_ci MODE analysis vs artifact fetcher when Green answers ask-4). artifact pull live: 19% @ 17:2xZ, sha-check wired; goal artifact (Q3 v1a) still ONLY on Green's host — ask-4 open, that file gates the END of the sprint.
- T-20:5x. Critical path: T3 oracles (CPU) -> stamped device slot (agent3, minutes) -> P3 bring-up (needs Q3 v1a bytes OR reference-artifact fixture mode while waiting). Cards idle, gates current, disk 50 G.
### STATE 2026-09-12 ~17:0xZ (AMD host) — **T3 FOUND ITS STRUCTURAL SHAPE (mma family HIP-invalid → donor SIMT is the HIP attention path); P2 analysis landed via TB2; agent2 re-armed and already out-producing; and two dependency gaps I had scheduled wrong**

- **T3's true shape (agent3, I verified the linchpin myself):** upstream bf16 attention calls `mma_bf16`; `mma.cuh` is 10 asm sites / 20 `'+f'` / 4 `'h'` constraints of `mma.sync`+`ldmatrix` — **whole-family HIP-invalid on this ISA**, not nvfp4-only as my pre-empt scoped. Their full-family guard is the correct scope (guard = 12th-listed exception path, routed to gemini, and the ONLY build-reachable consumers — w8 gemm_decode + w8 pair concat — are deliberately HELD pending their own rulings, so zero behavior change in-build). T3 = donor SIMT bf16 (590 LOC) as the `__HIP__` attention route, upstream mma kernels byte-preserved in `#else` CUDA. decode-side COMPILES (host-pass, labeled; device-pass untested), prefill-side in flight, CPU-oracles next, then device-slot request. `amd/t3-wip` carries WIP-labeled commits; lane branch stays green.
- **Two schedule corrections I own:** (1) **Agent2's funcattr behavior note**: mapping `cudaFuncSetAttribute` ≠ it succeeding — AMD has no dynamic-smem opt-in analogue; `hipFuncSetAttribute` returns NotSupported where per-function LDS config is absent. Their note fired exactly at T3's whitelist moment, so the durable schedule now reads: T3 device requests must state per-kernel whether >48K-class dynamic LDS is needed (donor SIMT may never need the attr call; kvarn keeps it quarantined). (2) **Agent3 independently surfaced the SAME shape in the embed defect** (gather arm has 3 `__shfl_sync` broadcasts, dense arm zero) — matching my own grep an hour earlier; good convergence, and their sharper version is what routed to gemini's sweep.
- **Agent2 re-armed and out-verified the brief:** Task-1 (funcattr wrapper) landed SFINAE-constrained with MEASURED rejection of the two obvious shapes (deduction failure on the bare spelling; unconstrained `T&&` swallowing integers — their n4 negative cell exists precisely so the bare-integer case can't pass silently) + enum values VERIFIED equal not assumed (CUDA `driver_types.h:1789` = 8 = HIP `hip_runtime_api.h:1025`) + the behavior note above. Task-2 (TB2 landing-delta) merged before I saw their report — my "correctly un-started" line from 15:3xZ is obsolete: agent2 started it after user re-assignment. P2 closed as analysis; Green landing deltas now mapped.
- **TB2's one routed decision (agent3 ruled, I endorsed, gemini's call):** `q3_ci_amd.sh:177-193` pre-4575ac1b arms-proof is unconditional → AMD `MODE=tp2` = guaranteed false-red once upstream made it MODE-conditional. Port the MODE branch verbatim when TG1 wires the cell.
- **P3 pre-flight done by me unprompted:** the REFERENCE artifact (19.03 GiB groupwise-int) has NO delivery path recorded anywhere — not on HF (`neroued/*` checked: only groupwise-int + nvfp4), zero `.ninfer` on this box (whole-machine sweep), no fetcher in repo. Started the HF pull (`artifacts/`, git-ignored by the repo's own line-26 convention): 23 MB/s, self-checking size+sha vs registry, resumable, own-pid. ETA ~4 h fits the budget. **BUT the GOAL artifact — Q3 v1a (15,446,796,288 B) — exists only on Green's host; ASK #4 landed in CROSSLANE** (push to a repo or give a pull path; sha256-verified on our side before use). Sprint end-state cannot be reached without bytes we do not hold; reference artifact covers parity-control + fixture work meanwhile, which is most of P3's gating.
- Net: T-21:1x, three lanes executing (T3 prefill, gemini's exclusions+MODE decision+RNE placement ruling, agent2→next compat), zero GPU consumed since G-AMD-13, zero gates skipped (golden re-run on every merge, step-0 on every sync; my own pass-labeling rule now applies to censuses too — host-pass vs device-pass defect texts differ).
### STATE 2026-09-12 ~16:0xZ (AMD host) — **P1 ANALYZED: eager-first locked by measurement; one D3 GATE HOLE found by me across four checks; one corruption-adjacent defect found and scoped out of the sprint path; TA4 stamped**
- **Facts (agent3's P1 analysis + evidence merged):** peer access absent at API level (G-AMD-5 confirmed); staged transport 3.13 GiB/s + 14.10/14.32 µs small-copy → **AR is LATENCY-Dominated at bring-up shapes → AR COUNT per token is the budget; batching/fusing is the perf axis** (design law for P3+). **R2 question ANSWERED: 6.2.0 ACCEPTS full cross-device capture** (donor's 6.4.1 rejection non-reproduced; 18-node graph captured+instantiated, small graphs replay at 11.26 µs/node) — graphs' blockage is ALLOWANCE SIZING (donor sized for 32 GiB; OOM at 7.98) + BETA splice anomaly (got-0-want-1 data-loss, attributed upstream per window rule). Eager-first stays locked; graphs are un-blockable by sizing work, scheduled nowhere.
- **MY find, filed to gemini (their file, receipts in this block): the D3 enforcement story is hollow across ALL FOUR existing mechanisms.** Check (e)'s certified failure is `__hadd` SCALAR hitting *overload resolution* (the scalar form doesn't exist for the 2-vec type) — its PASS is a statement about one spelling, not hardware legality; `__hadd2/__hmul2/operator+=` compile CLEAN on gfx900 (0 errors, my /tmp/guard.cu, reproducible) and the resulting kernel has ZERO `__ocml_*` lines (fp32 unpack + 2 float ops + `v_bfi_b32` repack), so check (i)'s fingerprint set misses the actual lowering. Fix routed: spelling-matrix probe reported as DATA + symbol-use scan or corrected fingerprint. This is tonight's theme completing its circle: the "compiler-enforced" premise died an hour ago; its replacement gate was ALSO about to pass on the violation. verify-by-rerun applies to gates too.
- **Agent3's 7th-row duty closed WO-02 as FINDING not green:** embed DENSE row V=256/D=5120/T=8 bit-exact on checked positions but bisect-proven async HSA aperture violation at next sync — corruption-adjacent, root-cause open. **Scoped out of the sprint by path analysis (Q3 v1a uses the GATHER arm — their step-2 goldens clean at 5 shapes incl [1024,5120]); gates any dense-embedding bring-up.** Routed to gemini PG-B as shape-sweep cell w/ non-degeneracy discipline (this row is the vacuous-green pattern with extra steps: it passes by dodging). Discriminating question posed to agent3: does the violator reproduce without shuffle execution (memory bug vs site #2-of-2)?
- **WO-TA4 STAMPED (agent3):** their landing-gate read was half-right — the CURRENT spine (Green q3 served-proven) is already on amd/main; further landing = re-verify delta. Scope: tp_group-adapted staged transport, one_shot lanes toward whitelist (entry waits on check-(h) pass), AR-count budget per P1. Status-consumption ruling restated: deferred-throw stands, spine hook upstream (CROSSLANE 15:4xZ addendum, includes the graph-dead/gfx900 datum offered to Green as THEIR hazard too — their decode path is graph-heavy).
- Board: agent3=TA4 (critical path), gemini=(e)-matrix+(i)-fingerprint+embed-sweep+TG1 spec+tp2 MODE, agent2 closed-with-honors, coord=merges+verifies+CROSSLANE. amd/main 80f30377, golden GREEN, build rc=0, cards idle, disk 54G. T-21:4x.
### STATE 2026-09-12 ~15:4xZ (AMD host) — **amd/main BUILD-BROKEN for ~35 min and REBUILT GREEN — the syncwarp host defect, the twin racing fixes, and P1 facts landing (eager viable, cross-device graphs dead on 6.2.0)**
- **Timeline, all three claims true at different times:** agent2's mapping (be4c394d) left `__syncwarp` calling an AMDGPU-target builtin with no host-pass declaration → every host TU broke (agent3 caught on their real build; agent2, in their ending session, ALSO caught it independently and shipped d6584756). I ran a parallel §6 takeover with the intuitive guard (`__HIP_DEVICE_COMPILE__`) — **agent2's fix was right and mine wrong**: their test showed my guard breaks valid device code (the macro is undefined in the HOST PASS of a -x hip compile; correct distinction is `__HIPCC__`, defined in both HIP passes, absent in plain g++). I reproduced their rejection of my guard myself before merging theirs — the merge msg carries both. Then agent3's 8460bdd4 (spin fix + addendum drop) merged: **full real build rc=0, golden GREEN**. Lesson recorded twice-deep: the escape route was device-only probes; the gate fix is agent2's three-mode cell (PURE-HOST .cpp + -x hip both passes + full cmake) which they wired into receipts, and agent3's independent real-build caught what nobody's cell saw.
- **agent2's final act deserves the ledger:** their seq-12 correction says their OWN grep line (:84) was confidently wrong against the integration ref (real break at :167) — they verified in a fresh worktree and filed the correction against themselves within minutes. Then: session closed, zero device time, branches synced, TB2 correctly un-started.
- **P1 EVIDENCE IN (`results/amd/p1/`, agent3's 3.2-s window on 48d5efc6):** TP2-EAGER is VIABLE (all eager transports correct; 3.13 GiB/s staged @ 14 µs small-copy) and **cross-device GRAPH EXECUTION IS DEAD on 6.2.0** (launch→hipErrorOutOfMemory, spliced-memcpy data-LOSS, replay OOMs mid-table) — donor's S8 class reproduces on our ROCm, so §E's eager-first ordering is now MEASURED law, and the salvageable shape is their per-device `single-x2` (45 µs/rep, no cross-device edge). Two corrections banked with it: memcpyPeerAsync returns SUCCESS driver-staged (my stamp said expect error — datum is the opposite), and the 3.13-vs-6.6 GB/s host-staged discrepancy is UNRESOLVED — nobody budget AR off either number until it's explained. replay_probe's table needs per-rank footprint trimming for 7.98 dies (measured adaptation, agent3's queue).
- Rulings out: spin Q1 = deferred-THROW accepted for bring-up, spine-owned check goes UPSTREAM via CROSSLANE ask-2, NOT a local tp_engine edit (anti-resurrection gate stands); 7th embed row = agent3 executes under still-open P1 stamp; window-ran-before-stamp-landed noted as process-observation, grant-before-launch rule restated. Board: agent3 (7th row, replay trim, P1 analysis -> my TA4-ready ruling), gemini (7-file exclusion, TG1 parity spec, cells), agent2 closed-with-honors. T-21:5x.

### STATE 2026-09-12 ~15:3xZ (AMD host) — **agent2 hands back CLEAN (context exhausted; zero-device-time session; open items recorded as open)**
- Session-end state per their report, all coord-verified: branches pushed+synced (wo-shim-width @ e86b6374 handoff doc — ff-merged, wo-tp2-probes @ af0c2348 merged), trees clean, nothing in flight, TB2 correctly NOT started (gated on Green sha), cards No KFD at baseline, total device time across their whole session = the ~10 s G-AMD-13 launch, released. Handoff quality: the model.
- **Collision cell: UNCLAIMED with reason** (they refuse to half-build a cell without negative control — "an artifact that passes and proves nothing"). Design constraint carried: must probe BOTH include orderings (addendum-before-shim AND after) — single-ordering is exactly what hid the shadow. Ownership: agent3's file to drop, gemini's to author. Flagged on both lanes' next dispatches.
- **Adopted into protocol (agent2's generalization, third-instance asymmetry):** *treat a compile cell's first error as suspect when it names a header or missing symbol rather than a construct — include-path/declaration-scope mistakes are indistinguishable from real defects in a bare count.* This is the matched-line law made operational; it's why dflash2 was cleared and __syncwarp wasn't.
- Sprint consequence: two executor lanes (agent3, gemini) + me; agent2's shim/transport knowledge now lives in e86b6374 + v340l/03 + v340l/04 + receipts script — citable, rerunnable. P1 (agent3) is the live critical path.

### STATE 2026-09-12 ~15:2xZ (AMD host) — **GATES LIVE: gemini's full package merged (checks a-exclusion/g/h/i + TG1 spec/battery); syncwarp durable mapping merged with shadow-collision defect found; dflash2 + PTX-lanes merged after a false-defect near-miss; agent1's ghosts cleaned; P1 remains the single unblocked critical path**
- **Gemini back and shipping** (#118/#119: PG-1 4-file exclusion @ 6e28d132 w/ archive-parity certification, cells (g) macro-guard + d=1152, WO-TG1 parity-spec/17-battery/q3_ci_amd.sh, WO-TG2 spin-guard (h) + D3-ISA-fingerprint (i) via __ocml-scan on real device asm — CI 2/2 zero-GPU). Merged `bd8b2d7d` (verdict-artifact conflict = theirs taken, generated+newer). Check (a) now correctly fires on pdl.cuh+warp.cuh (exceptions granted AFTER their list authored) → extension routed: PG-1 set = math, memory, q3, q2, dflash2_round (7th, now real), + pdl/warp under __HIP__ verification. Their spin-guard cell already has its first named target (agent3's held file) — the law has teeth in CI now, not just prose.
- **Agent2's __syncwarp** merged `be4c394d` (strict-union conflict resolution vs my probe-surface block — zero symbol overlap proven by comm before resolving; probes still 0-error post-union). R3 receipt (byte-identical ds/lgkmcnt on the real gqa write→syncwarp→read shape) = soundness evidence; expiry condition in header. **Defect found in the afterglow: agent3's TU-scoped addendum (:19) twins the shim's __syncwarp with a DIFFERENT default mask — redefinition/shadow waiting for a TU that includes both; their cells pass only because none does. Drop routed; general macro-vs-identifier collision cell proposed.** 8bdd4f5a == 4e2ae4c6 content (empty diff) — double-push, no loss.
- **Agent3's dflash2 fix + PTX volatile lanes** merged `58a4e521` after my probe self-errored: first compile attempt returned 1 error — MY missing per-target `-Isrc/targets/*/export` paths (the census's own pinned reproduction requirement), read the matched line ('startup_features.h not found') → instrument error, not theirs; honest re-run = 0 errors. Exception narrower than granted (CUDA text verbatim) — accepted. Spin hold stands (:80/:95 unchanged, census-only so unbreached; two questions stay open for the whitelist commit: host-side status-bit consumption, bound cost at 6.6 GB/s).
- **My own audit trail this stretch, recorded for the pattern:** agent2's syncwarp report corrected two of MY earlier claims (invented 2.7-GiB provenance; dev2+dev3 pair revoked on FLAT-topology measurement — all weights 40/hops 2/4 buses). Agent2's receipt caught its author's own error pre-commit (s_waitcnt miscounted as barrier); my showtopo came back EMPTY and nearly became a shrug-based ruling; my dflash2 probe was the broken instrument. **Adopted law (agent2's words): "a receipt that merely agrees with you is not evidence — a receipt that disagrees is a gift; read the matched line, not the total."**
- P1 (agent3, stamped): next in chain — nothing else blocks it. Board: agent3 = P1 + addendum-drop; agent2 = P1 assist available, TB2 gated on Green sha; gemini = exclusion-extension (7 files) + probe-review (Cell C, standing offer) + rapid-response. step-0 vs origin/main: zero-diff ✓; golden GREEN ✓ ✓ ✓ (each re-run post-merge); cards idle. T-22:1x.
### STATE 2026-09-12 ~15:0xZ (AMD host) — **TA2 VERBATIM-MERGED + probe port merged + P1 STAMPED — and agent2 corrected TWO of MY OWN dispatch claims, both accepted**
- **TA2 @ 667d8e4a merged `effaca1c`:** donor kernels adopted, all three files re-hashed by me byte-identical to fork HEAD ("VERBATIM, zero fork-text edits" — true); build [100%], parity 164/164, CPU goldens `diff -q` clean vs fae07b3c. Mechanism worth the ledger: `NINFER_GFX906_COMPAT` simply undefined ⇒ asm arms dormant, gfx900 `#else` fallbacks engage — donor's §2b compile-claim now proven by me too (execution half still UNENDORSED pending P1). pdl.cuh RULING-1-in-core ACCEPTED (unpinned dir, exception-pair discipline held) — precedent noted for future core fixes.
- **P1 window STAMPED to agent3 (written grant = the intercom msg):** probes built-in-message from `amd/tp2-probes` af0c2348 + amd/main effaca1c, `HIP_VISIBLE_DEVICES=0,1`, per-probe timeout 30 s, window ≤10 min, 4-facts table (p2p/transport/capture/replay) with topo weights inline; memcpyPeerAsync error-return = EXPECTED datum not bug; BETA hipStreamUpdateCaptureDependencies anomalies attribute upstream first; embed dense-bits 7th row rides same window; agent2 second-eyes on request. **Collision pre-killed: agent3's "I'll author the probe port" crossed agent2's delivery by 5 min — line ruled VOID, redirected to dflash2_round fix.**
- **My two dispatch errors, corrected by agent2 with receipts (both now fixed above):** (1) I attributed a "2.7 GiB worst-case analysis" to agent2 that they never produced — invented provenance for an unattributed number, the same message-vs-delta class I flag in lane reports; (2) my dev2+dev3 "physical pair" ruling (from their v340l/04, since corrected there): their four-device sweep measured topology FLAT — every off-diagonal weight 40, hops 2, four distinct buses — **no pair is privileged; ruling REVOKED, P1 uses dev0+dev1, pair choice arbitrary.** My own verification attempt returned an EMPTY matrix (inconclusive grep) and nearly became the basis of a counter-dismissal — recorded: verify-by-doing includes knowing when your own probe said nothing. Trust note earned: their correction stood because the measurement was complete where mine wasn't.
- **P1-gate facts now law for TA4 transport port:** donor replay_probe.cu:176-186 PASSES the spin standing-rule (s_sleep + caller bound + atomicOr soft-fail) — it's the port template for one_shot_allreduce.cu. Shim shadowing trap (macro rewriting a call onto its own wrapper's 4-arg twin) recorded from TB1-followup; durable `__syncwarp` shim mapping routed to agent2 (their file), agent3's TU-scoped addendum is interim and adopted-files-only.
- parity.cpp RULING: dropped from P1 (needs 5 product debug-hook APIs = bigger than the window; 25 GiB/s bar can't transfer; correctness = TG1 parity spec vs OUR tp1 control). PG-1 exclusion set for gemini now SIX files: math, memory, q3/q2_rowsplit_storage, dflash2_round (exception granted), pdl.cuh — their ACK still un-verified.
- Sprint clock T-22:4x. amd/main effaca1c golden GREEN; cards idle; lanes: agent3 (P1 window + dflash2 fix), agent2 (P1 assist + syncwarp mapping), gemini (6-file PG-1 + TG1 + TG2 cells). Next real events: P1 facts table, dflash2 SHA, gemini ACK, Green §4 answers.

### STATE 2026-09-12 ~14:5xZ (AMD host) — **TA1 CENSUS LANDED — the spine compiles; and the coordinator's own ISA probe settled a live two-lane disagreement about WHY (the answer was: both were half-right, and the difference matters for the gate)**
- **Keystone datum, first true reuse payoff:** `tp_engine.cpp` — Team Green's canonical TP2 spine — **compiles EXIT 0 under hipcc/gfx900 syntax census** (agent3 TA1 @ `d4c0a916`, merged `f5bc227c`). Same for tp2_request/tp2_backend/host_kv_parked/dflash2_context_append/tp_group/weight_shard. One genuine product defect: `dflash2_round.cu:353-355` GNU void*-arith in device compile. Census reproduction needs -std=c++20 + export/third_party includes (pinned in the log). **This is the 24 h thesis validated on bytes: the spine ports; the AMD-specific surface is narrow.**
- **The disagreement, and why it's worth a block:** agent2 audited `__CUDA_ARCH__` as **undefined** under hipcc → `__nanosleep` preprocessed away → silent sleepless spin. agent3's census said the guard is **TRUE on gfx900** (900>=700) → `__nanosleep` UNDECLARED → hard error, and filed the opposite as its self-correction. Both cannot drive the same file. **My own -O0 ISA probe (/tmp/arms.s): `s_mov_b32 s6, 2` + `v_or v2, v0, s6` — the #else arm executes; the macro is NOT defined on the HIP device pass.** Agent2's observation was right, agent3's retraction retracted a correct thing. The resolution that matters: **it is not one failure mode but two reachable states of the same unguarded pattern** — as the file stands today, syntax-clean silent spin (why it passed `-fsyntax-only` at rc=0, my run); define `__CUDA_ARCH__` anywhere in a future compat header (donor style) and the identical lines become hard errors on an undeclared `__nanosleep`. Neither is detectable by any existing gate; my `safe.cu` receipt proves the donor safeguard shape (s_sleep(1) + 1<<26 bound + soft-fail) compiles on gfx900 TODAY, so the law is implementable.
- **RULINGS issued 14:5xZ:** (1) **RULING-1 EXCEPTION GRANTED for `dflash2_round.cu`** (one line, `sizeof`-typed pointer arithmetic, CUDA text in `#else`, no behavior change, census log is the evidence) — added to gemini's PG-1 exclusion list (now 5 files); anti-resurrection rule re-confirmed: exception covers ONLY the dflash2 void* lines, `tp_engine.cpp`/`tp2_budget.h` canonical pair untouched by anyone. (2) **GO for agent3's TA2** with the 28 sub-group-site invariant as a standing tripwire (gemini owns the CI cell), donor PASS-THROUGH shape adopted, §2b device claims stay UNENDORSED until P1/probe data. (3) P1 window pre-declared (dev2+dev3, minute-class, consolidation with agent2's 7th T1 row). (4) WO-TG2 dispatched to gemini (spin-guard cell + bf16 D3 ISA-fingerprint cell + probe review) with all my receipts inlined; their PG-1 exclusion + TG1 remain priority 1, ACKs still owed on both dispatches.
- Net state 14:5xZ: amd/main @ f5bc227c, golden GREEN post-merge; cards idle No KFD; disk 55 G; three lanes executing (agent3 TA2, agent2 probe port, gemini two cells + 5-file exclusion + parity-spec draft all named); next event = first push, then my step-0 verification of it. T-23:0x on the user clock.

### STATE 2026-09-12 ~14:4xZ (AMD host) — **SPRINT HOUR 1: WO-04 CLOSED + WO-TB1 LANDED in the same 10 min; line law: no unbounded device spin whitelists; P1 pre-stamped-pending-port**
- **The reuse thesis is paying in real time:** agent3's WO-04 steps 3+4 (whitelist q3/q2, 5 lines) built the FULL HIP library green with ZERO compat edits — Green's promote + our guard-patch sufficed; agent2's TB1 landed the 6 TP2 shim surfaces (all 6.2.0-verified, enum-names-only port rule, 5/5 negative tests, `.type` EXISTS at hip_runtime_api.h:245/263). Both merged after my own byte-verifies (embed_gather -fsyntax-only exit 0 on the merged tree = their headline claim reproduced by me). `amd/main` @ `e855492c`: q3 layer compiles+links, no functional gaps, golden GREEN, cards idle, zero device time used since G-AMD-13.
- **New line law (agent2 asked, I rule): NO device spin-loop enters the HIP whitelist without bounded poll + soft-fail + unconditional `__builtin_amdgcn_s_sleep` on the HIP path** — the donor's S9c wedge (request 12, MODE1 reset FAILED, SysRq reboot) plus my own confirmed read of `one_shot_allreduce.cu:80/:95` (unbounded, `__CUDA_ARCH__`-gated so hipcc compiles the sleep OUT entirely). CI cell asserting it = gemini authorship on agent2's spec. Binds TA1/TA4 directly.
- §2b device-claim demoted to UNENDORSED ("compiles" ≠ "runs correctly" — agent2's bf16 finding is the precedent; falsifier required before TA2 adopts `#else` arms as correct).
- **P1 probe window prepped by me:** 4 donor probes are standalone HIP executables, exit 77 below 2 devices; p2p_probe's 25 GiB/s bar EXPECTS RED on this box (its fallback-path report is the datum, not a failure). Blocked only on the port (agent3, WO-TA3 per plan) + agent2's 30-min symbol-readiness table (fork-internal helpers our shim may lack — settling it pre-window keeps minutes-of-grant for minutes-of-GPU). Stamp issues the moment both land; window consolidates agent2's 7th T1 row (embed dense-bits).
- Sprint clock: T-23:20. Lanes: agent3 TA1 now -> TA2; agent2 probe-table then TB2 on Green's landing; gemini owes PG-1-now + TG1 + two cells + spin-rule cell (dispatch sent, no ACK yet — liveness owed next wake). Wake cadence 20 min; board in state file current.

### STATE 2026-09-12 ~14:2xZ-b (AMD host) — **24-HOUR SPRINT OPENED (user order): agent1 abandoned, 3-lane team + gemini as regression defense; cadence 20 min; WO-04 scope corrected by my own verification — the embed_gather arm was ALREADY REUSED (promote carried it)**
- **User directive 14:23Z:** fully working in 24 h, max velocity zero regressions, gemini key (phase gates + rapid test building), lanes autonomous. Team: coord + agent2 (Agent-B) + agent3 (Agent-A, TP2 lead, user-flagged most reliable) + gemini (tests exclusive). **agent1 CLOSED-ABANDONED** (API instability, user call) — off watchdog, liveness ask dropped, WO-03 scope folded: targets/serve parity→TA1 census, stub set→TA2 gate, AR redesign doc→SUPERSEDED by donor-adapt §E.
- **My scope error, self-caught while verifying agent3's report (their report was fine):** WO-04 step 5 assigned authoring `embed_gather` Q3 arm — it's been IN OUR TREE since Team Green's promote (`67ae81ab`, 24 Q3 refs in `embed_gather.cuh`), and agent3's step-2 goldens were already passing against it. CROSSLANE's "mandatory blocker" row was true pre-promote, stale post-sync — exactly the drift class. **New standing rule for every WO I issue: verify-exists-before-authoring as a pre-step.** (WO-05's attention scope believed still real — no T3 family in any promote; flagged for census re-check.) This is the user's velocity thesis applied to my own planning, not just lanes'.
- Sprint mechanics live: wake cadence 3600→1200 s (20 min) — pinger restarted clean; wake text = sprint board (lane positions, next events, safety lines); CROSSLANE §4 READ-THIS row landed routing agent3's three asks to Team Green (landing sha+notice, tp_group transport-primitive list, TP2 tests/q3_ci tp2-MODE contract) + giving back our ds_bpermute hazard warning and the open upstream guard defect.
- **agent3 WO-04 steps 1-2 COMPLETE, coord-verified from pushed bytes:** 4/4 CPU goldens green UNMODIFIED sources vs our tree (`-Isrc` exercises our real headers incl their guard patch), shapes up to [1024,5120]/[512,6144] max_diff 0.00, honest c++17→20 iteration logged (`tests/ops/quantized_weight.h` needs std::span). Report format now the lane standard. Remaining WO-04 = steps 3 (HipSources q3/q2 whitelist entries — ZERO today) + 4 (RULING-1 compat lanes to compile green); ordered step-3-BEFORE-TA1 so TA1 censuses the real whitelisted set.
- All lanes dispatched with stamped scopes (agent3: WO-04→TA1→TA2; agent2: v340l/03 ISA audit→TB1; gemini: PG-1-now→WO-TG1→cells→rapid-response). P1 probe window pre-staged by me; stamps consolidate agent2's 7th T1 row. Next coordinator actions fire on: lane landings (verify+merge), Green §4 answers, P1 stamp readiness.

### STATE 2026-09-12 ~14:2xZ (AMD host) — **FORK VELOCITY: agent3's plan ruled ACCEPTED in full; TP2 sequence re-based on the gfx906 donor; step-0 guard patch merged; artifact-mismatch locked as gate law**
- User surfaced `github.com/JCraigWasTaken/ninfer-gfx906` (MI50 port, TP2 31.9 t/s eager, 921 commits, cloned to /tmp/ninfer-gfx906); agent3 audited it under their user-hold and produced `docs/amd/FORK_VELOCITY_PLAN.md` + `TP2_AMD_SUBSET_PLAN.md`. My byte-verifies held on all load-bearing claims (fork has ZERO q3/q2 — WO-04 genuinely ours; no `src/runtime/tp2` — anti-resurrection trivially clean, cherry-pick-only law restated; `hip_compat.h` 97 defines; fork's own fallback arms `#else` = gfx900-runnable — net-new correctness kernels ~zero).
- **Merged `amd/main` → `039ab28b`:** agent3 step-0 guard patch `f5035616` (exact 2-file/1-line as ruled; their falsifier extends the error class with 3 `__float2half_*` redefinitions I'd missed) + both plan docs. Golden GREEN on merged tree. `embed_gather.cu` now COMPILES — agent2's missing 7th T1 row rides the future WO-V2 window (stamps consolidate; no standalone grant).
- **Rulings (full text in agent3's intercom):** §A-F adopted; WO-V1 (shim-gap closure, routed agent2, their file) and WO-V2 (4 fork probes, one stamped window) created; §E supersedes my 13:4xZ serve ordering — TP2-EAGER first, graphs/flag-sync gated on measurement, wedge class (their S9c card-wedge+reboot, reverted default at `7a3c18d`) excluded absent root cause; **§F freeze list: no lane writes a kernel family the fork already ships without coordinator waiver.** **R5: first functional serving = Q3 v1a artifact, 6.68 GiB/die over 2 ranks, short-context, allocator-only capacity gate — 4-device stays HORIZON.**
- **ARTIFACT MISMATCH = new line law** (agent3's catch, locked): fork artifact 18,210,531,328 B ≠ ours 20,437,336,576 B — fork goldens/t-s/parities DON'T transfer; code + probes + stage-logs do. All gates vs OUR tp1 control.
- agent2: WO-02 closed (`b5b90bcc`), now ISA-auditing the fork (warp.cuh masks-ignored claim is OUR bug class — their subgroup mechanism gets the tripwire, adoption contingent). agent1: SILENT past grace, liveness ask queued — takeover trigger next tick; WO-03's TP2 halves fold into §E sequence if so. gemini owes: PG-1 exclusions (now 4 files: math, memory, q3_rowsplit_storage, q2_rowsplit_storage) + agent2's build-cell + d=1152 cell.
- Alert hygiene (user law 14:1xZ): claim-file routing live, verified 3/3 sends to coordinator session only; agent3+everyone else purged.

### STATE 2026-09-12 ~13:4xZ (AMD host) — **agent2's G-AMD-13 preflight caught a real promote-carried defect; I extended it to a second file; ruling (c)+(b), WO-04 amended with step-0, agent3 spun up (glm-5.3 via zai), CROSSLANE defect row landed**
- **Blocker accepted on evidence, then grew one:** agent2 held the launch (zero device time — correct call under the stamp) and root-caused 20 build errors to `q3_rowsplit_storage.h:23`'s `#if !defined(__CUDACC__)` `__host__/__device__` neutralisation, which hipcc executes and which neuters ROCm's `__device__` intrinsics tree-wide through `embed_gather.cuh`. My byte-verify found the **identical guard at `q2_rowsplit_storage.h:25`** — agent2 under-reported, and their root cause is nonetheless complete for the reproduced failure. Third promote artifact I've now verified before adopting (after the header hash and the tier map); the checking keeps being the trusted part.
- **RULING:** (c) NOW — G-AMD-13 amended: scoped harness linking shuffle TUs minus `embed_gather.cu`, report 6/7 T1 rows with the missing row named on the report's face (no silent coverage hole). (b) ROUTED — CROSSLANE §4 AMD-log defect row landed: one-line upstream fix `&& !defined(__HIPCC__)` both files + a CPU-only `hipcc -fsyntax-only embed_gather.cu` regression cell for their farm, offered as convergence-not-fork. (a) deferred to WO-04 **step 0** (new): agent3 patches BOTH headers as the lane that owns them, commit body names both files for gemini's PG-1 exclusion list, ships the proof-of-fix cell.
- **Gate hole agent2 confessed + proposed the patch:** my merge gates (golden, falsifier, step-0 parity) all pass on a non-building tree — none compiles the library. Accepted, routed through gemini (test lane, §7.x): agent2 sends the `cmake --build` CPU cell, gemini lands it in step-0. Ledger credit recorded: a falsifier for one's own gate is worth more than a green for someone else's.
- agent3 online (`pi-dual_5060_ti_ninfer-105140`, hub-registered; intercom name pending — routing to them by hub id until they announce as `agent3`); model `zai/glm-5.3` configured by user + verified with a live authenticated completion. WO-04 dispatched with step-0 first. Roster now: coord + agent1 (WO-03 rebuild) + agent2 (G-AMD-13-amended) + agent3 (WO-04) + gemini (lift brief + PG-1 exclusion + build-cell landing). Four lanes, disjoint files, one serial card queue, disk 55 G — two full builds max, staggered.
- **My own debris caught during commit:** a stray `ping_msg.txt` from a mis-scoped acceptance test was in the repo root; removed pre-commit (disclosed — the pinger test cycle that made it is the same one whose echo mails #74-#77 are still draining through the hub).

### STATE 2026-09-12 ~13:3xZ (AMD host) — **WO-02 fix MERGED: `amd/main` @ `2c4b8110`; calibration freeze LIFTED; G-AMD-13 stamped; second origin/main sync (q3 promote); hourly pinger + 15-min watchdog built to the NVIDIA WO spec**
- **Merge verdict PROCEED, executed by me, gates run independently (not inherited):** agent2's `468878e0` (+ docs-only `25921134`) merged to `amd/main` → `67d03ec4` → FF question settled (my two origin/main syncs landed between their commits — their "FF-safe" held only in parent-ancestor scope, correctly claimed there). My own checks pre-merge: CPU golden re-run on merged tree GREEN (120→496 / 1520 / inv 0.088388, tripwire 5/0); **I reproduced the falsifier myself** — script copied into a throwaway worktree at pre-fix `e2f7b989`, RED 5/5 divergent `ds_bpermute`, exit 1; step-0 `tp_engine`/`tp2_budget` zero-diff vs `main`. E1's pre-registered prediction panned: uniform-row inv landed on the true-Σ 0.088388. The 12:2xZ two-suspect narrowing closes as **KERNEL-side (shim), not probe-side**; batch2c's 0.208008 reproduced at 0.207781 (residual = bf16 ratio rounding, as agent2 attributed).
- **gemini calibration freeze LIFTED** via hub (brief sent, verified out of my own re-runs — the handoff's "lift only after the golden is on disk and passing" condition was met by MY measurement, not their claim). Three obligations routed to them: (1) **PG-1 check (a) is now false-red on `math.cuh`/`memory.cuh`** — RULING-1 registered exceptions (`758b1322`) with no gate exclusion mechanism, and the gate diffs bare `main` (stale-ref risk) — gate fix is theirs (test lane, §7.x), interim ruling: those two files = EXPECTED-RED-with-ticket, everything else in check (a) stays a hard fail; (2) divergence-count gates must key on the isolated reduce (caller-side warp-uniform EXEC regions are unchanged by design — rmsnorm 114/114, argmax 20/20); (3) freshness protocol adopted from `docs/amd/AGENT2_WO02_HANDOFF.md`.
- **Second `origin/main` sync merged (`f5c6d5b6`): Team Green promoted q3 onto `main`** (Q3G64 dtype entries, `q3_rowsplit_storage.h`+gemv/gemm, dispatch + embed-gather arms, sizing/smoke docs). Overlap vs my branch = `src/CMakeLists.txt` (CUDA-only q3 ops lines — registered shared-file exception; HIP path untouched, `HipSources.cmake` early-include verified) + `docs/CROSSLANE.md` (keep-both). `q3_rowsplit_storage.h` in-tree at sha256 `e66a4270…` — byte-identical to the header I compile-verified standalone. **Their promote means the Q3 tier reality is now MAIN-CANONICAL: zero Q2, embedding = Q3G64 GATHER (`embed_gather.cu` Q3 arm on our bring-up list), artifact 15,446,796,288 B / load-fit 13.35 GiB.**
- **G-AMD-13 STAMPED to agent2 (written grant = the intercom message):** dev0, ≤90 s, `/tmp/batch2_fixed`, rebuild-in-message freshness, release row, own-PID kills, every T1 row reported both directions. Terms incorporate their honest-limit note: `layer_norm` moving off its vacuous 1.57e-40 and rmsnorm newly-exercised are FINDINGS, not regressions; **7/7 explicitly not predicted**.
- **agent1 WO-03 ruling: DENIED `cuda_runtime_host_api.h`.** Their 6-item blocker set died to bytes: #1–2 are in the merged fix (exactly the `tp2_backend.cpp:1400` failure they hit — their build predates my merge); #3–6 I verified PRESENT at their own base `9228ea48` (graph wrappers :75/:88, `cudaMallocHost`/`cudaFreeHost` :40–41). 4-of-6 over-claim = message-vs-delta class, said plainly. Ordered: rebase onto `amd/main`, rebuild, **paste first real compiler error** — the build, not the grep, is the blocker list. Delivered-2 note updated: first-token derivation now runs on the corrected emulation, √2 tax gone.
- **Pinger WO (cross-team build-and-ship) DONE + ACCEPTANCE-PASSED:** `~/.pi/agent/tmp/pinger_amd/` — `loop.sh` (3600 s, `sleep & wait` so TERM runs the trap immediately — original foreground-`sleep` version DEFERRED kills, caught in acceptance, fixed), pid file + `stop.sh` refuses foreign PIDs (tested against pid 1), wake = `wake.mjs` hub mail to the LIVE coordinator session resolved by cwd each tick (old hub id `4032c47e` is a dead registration — resolved fresh every time, no stale addressing). Wake text read FRESH from `ping_msg.txt` per tick; **empty state → delivered `PINGER-FAIL-LOUD: NO STATE`** (proven, hub msgs #81/#82); fabricated-msg delivery proven end-to-end (msg #80 landed in my own inbox); kill+restart by pid file proven. `watchdog_loop.sh` (900 s): fires only on named silent sessions (heartbeat AND branch tip unmoved), tested against fabricated-stale memo (fired correctly) then silent on live state. Running: pinger pid-file + watchdog pid-file in dir. **My earlier kill of the old 30-min heartbeat pinger missed its child (setsid wrapper PID) — caught and SIGKILLed by exact cmdline, disclosed: the failure mode this WO exists for, caught in my own first attempt.** Lesson banked: `setsid cmd &` gives the WRAPPER pid from `$!`; the loop must record its own `$$` (it does) and tests must target that.
- Cards: baseline before my checks; **agent2 device window opens on their launch** (30 s per-run timeout, 90 s total). Gemini online `b4791a54`. `origin/main` tip moved under me twice today (d5868323→75198a45) — sync merges are now a standing wake-checklist item, not a surprise.

### STATE 2026-09-12 ~13:1xZ (AMD host) — new coordinator session (01a095b4) cold-started; `origin/main` merged into `amd/main`; Team Green's Q3-unblock claims verified against bytes; fix-merge arbiter posed to agent2
- **Cold-start per the 12:4xZ handoff.** Board verified, one map-drift found: `amd/main` was at `e2f7b989` (handoff said `eab05e9b`) — one coord commit newer, no lane landings. Cards at baseline (4 × 18,575,360 B, No KFD PIDs), disk 56 G free, roster live (agent2 `01a09317`, agent1 `01a09591`, both intercom; hub mail #1/#6 acknowledged — bridge online, Gemini identity noted).
- **Handoff consequence #2 EXECUTED: `origin/main` (`d5868323`, 2 commits) merged into `amd/main` → `818baf07`, pushed.** CROSSLANE §1 discipline done as proof, not prose: merge-base `be970ead`; red's 76 changed files ∩ green's 5 = **EMPTY** (first `comm` reading was a stdin artifact — re-checked by per-file diff, green touched none of our surface); dry-run zero markers; staged set == green's list exactly; **anti-resurrection region (`tp_engine.cpp`, `tp2_budget.h`) zero-diff vs `origin/main` post-merge, exit 0**. Behind-count now 0. One direction only — `main` still never receives our pushes.
- **Team Green's hub announcement ("Team Red unblocked via CROSSLANE §4") adjudicated against source+bytes, VERDICT: true, and it answers user-open-item #2 (Q3 artifact numbers):** (a) `src/ops/linear/q3/q3_rowsplit_storage.h` @ `origin/wo/q3-gemv` = 7,280 B, sha256 `e66a4270…`, **compiles under g++ -fsyntax-only on this host, confirmed by me** — "dependency-free" means the dual host/device decode core (cstddef/cstdint; the std-only host helpers are a separate guarded block), portable claim holds. (b) tier correction is real: served v1a = **ZERO Q2**, embedding = **Q3G64 gather** — and it adds a blocker we never had: **`embed_gather.cu` Q3 arm is mandatory** (the GEMV pair cannot load the 1.26→0.47 GiB embedding). (c) goldens are CPU-portable (fp64 oracle ≥0.999/role, quantize side scale=max_abs/3.5, clamp [−4,3]). (d) "Q2-throw-by-law" = their no-silent-Q4-fallback loud throw — confirmed at :578. Artifact pin: 15,446,796,288 B, sha256 7f26a0eb…, load-fit 13.35 GiB, `_iq3` name stale/dropped. **Caveat recorded honestly: that doc exists only on `wo/q3-gemv`, not on `main` — cite the branch, it can move.** Our own `docs/q3_amd_port_questions.md` answers (4×7.98 GiB, no dtype entry) stand as written.
- **THE FIX still not on the integration branch — arbiter question posed to agent2 (intercom, C441):** their `amd/wo-shim-width` @ `eab05e9b` carries the fix **UNCOMMITTED** (shim cndmask rewrite + a `cudaPointerAttributes` alias block + audit +89 + untracked golden `tools/v340l/shfl_wavefront64_golden.py`), while committed `wo/v340l-shfl-fix` @ `012c5735` has a shim that **differs** (no pointer-attrs block). Per handoff, merge whichever has the verified build — so agent2 must say where goldens 496/1520/inv 0.088388 were actually run, commit + push, and I merge + re-run step-0 parity + stamp G-AMD-13's 90 s dev0 launch on the merged tree. I do not touch their worktree; I do not self-merge their uncommitted state.
- agent1 dispatched on WO-03 (worktree clean @ `9228ea48`, no commits — expected); told the base is now consistent and to cut commits on post-merge `amd/main`, and that its first-token derivation may shift under the pending shuffle fix.
- **No grants live. No device time consumed. Gemini calibration freeze correctly still on** — the golden is NOT yet on disk on `amd/main`; freeze lifts only after the merge + passing golden (handoff first-10-min item 4).

### STATE 2026-09-12 ~13:0xZ (AMD host) — **WO-02's root cause was wrong and agent2 proved my prescribed fix would not work**; base-dispatch error resolved; anti-resurrection flag adjudicated
- **Accepted correction, on the merits:** WO-02 §1 (my words) said the defect was passing `warpSize`
  instead of the caller's `width`. **agent2 compiled my literal instruction as variant C — still 5
  `ds_bpermute`, still broken.** The real defect: the out-of-group guard is an **early return**, and on
  GCN these lower to `ds_bpermute`, a **wavefull** LDS exchange — a lane publishes its slot only by
  **executing** the instruction, so lanes 16..31 opting out of the `delta=16` step left lanes 0..15
  reading stale slots. ⇒ exactly half the row. Fix as landed: forward `width`, **no early return**, select
  out-of-group with `cndmask` so all lanes execute. Guard *predicate* was already CUDA-exact (exhaustive
  check: lanes 0–63 × every delta/laneMask at width 32 and 64, zero mismatches).
- **The generalization that matters more than the bug:** the defect exists only in **multi-step
  propagation**, so any single-step enumeration clears the shim — **that trap caught me twice and I am
  the one who wrote the "enumerate every lane" check.** Permanent consequence: gates on this line test
  **ISA-level behavior** (`ds_bpermute` under conditional execution), not call-site semantics. agent2's
  tripwire was **demonstrated RED (5/5 divergent) against the unfixed tree** — a falsifier, not an
  assertion.
- **My dispatch error, resolved by my own later merges:** WO-02/WO-03 quoted base `7ce534f1`, written
  *before* the consolidation merges — so `amd/main` had **no shuffle emulation and could not build T1**
  (agent2 proved it with `-fsyntax-only`: undeclared `__shfl_xor_sync`/`__shfl_sync`/
  `__floats2bfloat162_rn`) while `HipSources.cmake` already whitelisted T1 kernels. **Internally
  inconsistent base, my doing.** Now verified fixed at `c8d8265c`: `ab7ac370` is an ancestor,
  `cuda_runtime.h` = 155 lines / 11 shuffle symbols, `cuda_fp8.h` + `cuda_pipeline.h` +
  `math_constants.h` present. **Rule for me going forward: never pin a lane to a SHA in prose — tell it
  to branch from the branch and re-verify the base exists as described.** (Third costume of the
  cite-by-name rule.)
- **Anti-resurrection flag adjudicated (agent2 routed it, not theirs to settle):** `wo/v340l-hip` vs
  `main` on `tp_engine.cpp` = **+6/−112**, which *looks* like deleting Team Green's DFlash2 multibatch
  admission guard. It is not — the branch is **behind** main in that region, and `amd/main` is
  **byte-identical to `origin/main` on both `tp_engine.cpp` and `tp2_budget.h` (zero diff)**, so nothing
  was lost. A lagging branch trips the step-0 parity cell until it merges `amd/main`: **the gate working
  as designed, not a regression** — and merging fixes it rather than excusing it.
- **Blast radius, measured:** only **27 of 147** user shuffle sites are inside the HIP build; the other
  120 sit in 36 files unreachable from `HipSources.cmake`. gemini's 159-row table reconciles
  (147 + 12 shim) with zero bad rows, and agent2 separately disproved a suspected false positive
  (`gqa_attention_kv_quant_nvfp4.cuh:165` is a comment) — i.e. it checked its own check.
- **Self-report noted approvingly:** agent2 briefly `cp`'d the shim across to the base while testing,
  caught it, and restored from git; verified clean. Disclosing the transient beats not doing it.
- **G-AMD-13 conditionally stamped** (see dispatch): merge `amd/main`, show the build green **CPU-side**,
  then one 90-second dev0 launch. No other grant live; cards at baseline.
### STATE 2026-09-12 ~12:3xZ (AMD host) — **G-AMD-12 stamped**; gemini's probe was unachievable as designed, replaced with a behavioral one that names the participating subset
- **Constraint collision I caught before it burned a launch:** gemini's two-question plan asked the kernel
  for **pre-reduce partials** and **its own stored `inv`** — both unreachable from an external TU, and
  `inv` is never stored (`out[i]` = `x[i]*inv` rounded to bf16, i.e. **the same quantity the harness
  divides by**, so Q2 would have re-measured the suspect instrument). Their zero-product-edit rule is
  right; the design had to stop asking the kernel to confess.
- **Behavioral substitute, one launch:**
  **E1 delta sweep** — row of 128, one nonzero `x[j]=1.0`, sweep j: participating ⇒ Σ=1, `inv≈1.0`,
  `out[j]≈1.0`; not read ⇒ Σ=0, `inv = 1/sqrt(1e-6) = 1000`, `out[j]≈1000`. A **1000× separation** with no
  inference chain and no quantization ambiguity — **the output *is* the participating subset, named by
  index.** Stated my prediction and its falsifier: the datum implies ~62 of 128 participating, so the 1000s
  should form a contiguous block or a stride pattern; **if they're scattered, the single-scalar-factor
  reading is wrong** and we're in a different story.
  **E2 uniform row** — all 1.0 ⇒ true Σ=128, inv=**0.088388**; half the elements ⇒ 64, inv=**0.125**.
  E2 counts participants, E1 names them; neither suffices alone. Uniform rows also crush the bf16 ratio
  noise that motivated the whole redesign.
  **E3** all-zero row ⇒ inv = 1000 exactly, pinning eps handling. **E4** d=64/128/256 (fast domain) plus
  **d=66** (generic kernel) — if the defect is fast-path-only, that's half the diagnosis in one table.
  **E5** deliberately disagree `ne[0]` from the harness's assumed shape, to test the layout reading
  itself rather than trusting metadata.
- **Stamp terms:** dev0 only, **90 s** (E1 is 128 tiny launches), stop-and-report-partial rather than run
  long, zero product mutation, freshness by construction (rebuild from committed HEAD in-message +
  marker/binary/archive hashes in the log — their protocol now, and the standard), release row verbatim,
  30 s per-run abort, own-PID kills only. **Deliverable is the index→value table, not a PASS line.**
- **Pre-committed outcome handling:** if the kernel is correct and the harness's `inv` derivation was the
  error, l2norm closes as **harness** and I lift the cross-lane calibration freeze in the same commit
  message — I won't leave gemini's gates frozen on a defect that turns out not to exist.
- **My own delivery-check slip, disclosed:** my integrity grep for `out[i] holds` came back false and I
  nearly re-sent; the actual text has backticks around `out[i]`, so **the check was wrong, not the
  message** — the same class of error as gemini's `src/ops/` grep path earlier tonight. Nothing was
  resent, nothing was mangled.
- **No device time in effect** until they fire G-AMD-12. Cards at baseline, trees clean, main pushed.
  **User's three items unchanged:** Q3 `bytes`/which-file · §7.x waiver · root for PG-0c.
### STATE 2026-09-12 ~12:2xZ (AMD host) — **reduce exonerated by CPU emulation**; l2norm narrows to "compiled accumulation vs probe extraction"; probe spec sent
- **My own arithmetic slip, caught and disclosed:** the first emulation **overwrote instead of accumulating**
  (`x = read(...)` where CUDA semantics are `x += read(...)`), which produced a nonsense lane0 = 2.3384 and
  would have let me conclude the reduce was broken. Fixed the emulation before making any claim from it.
  Fourth or fifth instance tonight of the same lesson: the bug is usually in the checking.
- **Exoneration:** with `d=128, pairs=64, KMAX=4`, per-lane partials are lane0 1.7034, lane15 0.3612,
  lane31 2.3384, **Σ lanes = 48.3670**; the group-local width-32 butterfly gives **lane0 = 48.3670**; and
  the **physical-64 wavefront with a 32-based guard also gives lane0 = 48.3670, lane32 = 47.0767** = row
  1's own full sum. **Both readings of the shim are correct**, so the shuffle-leak story (gemini's) and
  the lane-loss story (mine) are dead — third static claim tonight killed by computation, not argument.
- **Residue shape is now tight and useful:** probe-implied Σ / correct = **1/2.093**, and
  `0.208008 / 0.143789 = 1.4466 = sqrt(2.093)`. The two are internally consistent ⇒ a **single scale
  factor on the accumulated Σ**, not a scattershot error. Therefore exactly two suspects remain: **what the
  compiled kernel accumulated** (runtime `d`/`pairs` other than 128/64) or **how the harness extracted
  `inv`** from `out[0]/in[0]`.
- **Probe spec sent (#70, verified intact):** answer those two questions and nothing else — kernel's
  runtime `d`/`pairs`/`kMaxPairsPerLane` + lane0/lane31 pre-reduce partials; and raw `out[0]` bits + the
  kernel's own `inv` cross-checked against the harness derivation. **If the kernel's own inv ≈ 0.1438 while
  the harness says 0.2080, the defect is the probe**, and l2norm closes as harness with calibration
  unfrozen for cross-lane gates. **Zero product mutation** — instrument in a test-local TU that launches
  the real `l2norm_launch`; **no env-gated print inside `l2norm.cuh`**, because the exception pair stays
  exactly two files and a third needs a ruling, not convenience.
- **Still true from earlier:** audit reconciled (148+11=159) and handed to agent2 for cross-check; gemini
  owns the launch and the freshness protocol (rebuild from committed HEAD in-message, hashes in-log).
  **No device time in effect; cards at baseline; calibration frozen pending the probe.**
- **User's open items unchanged:** Q3 artifact `bytes`/which file · §7.x waiver · root for PG-0c.
### STATE 2026-09-12 ~12:1xZ (AMD host) — **negative result kills the reduce-shaped hypothesis family**; the probe was measuring the wrong quantity
- **Reproduced gemini's harness arithmetic on CPU** (their RNE `to_bf`, generator
  `v = 0.01*((i*37)%211) - 1.0`, row 0 = elems 0–127, matching the kernel's contiguous
  `row_base = row*pairs`): full row **48.3670**, k=0-only 24.6267, k=1-only 23.7403, lanes 0–15 both-k
  11.2117, lanes 16–31 both-k 11.9509 — and the **probe-implied Σ from `inv = 0.208008` is 23.1121**.
- **It matches none of them:** 2.6% off k=1-only, 6.2% off k=0-only, and roughly half of *neither*
  half-lane value. **So the whole "missing lanes / missing k / short butterfly / drop pair 31" family is
  dead**, including the subset match gemini already retracted and any variant I might have built on it.
  Their call to stop rather than keep guessing was right; mine to check before building on it.
- **The diagnostic was measuring the wrong quantity.** The probe infers Σ from `out[0]/in[0]` — a
  **bf16-quantized output ratio**, i.e. the reduce's result read through the store path after
  `__floats2bfloat162_rn` with `in[0]` re-read from a buffer. bf16 rounding alone is ~0.5%, so it isn't
  the whole gap, but **we have been reasoning about a ~5% discrepancy using a quantity whose error budget
  was never bounded**. Recorded in docs/amd/README.md with the table so the dead family stays dead.
- **Corrected next-step design sent to gemini (#68, verified intact):** not a per-lane subset dump —
  an env/compile-gated device-side print from lane 0 of a few rows of `d`, `pairs`, `row`,
  `kMaxPairsPerLane`, **pre-reduce partial** and **post-reduce Σ**. One launch, no inference chain, and it
  separates accumulate-short vs reduce-short vs kernel-vs-harness disagreement about `d`/layout, while
  measuring the same quantity twice. Same freshness protocol (rebuild from committed HEAD in-message,
  marker + binary + archive hashes), one launch, 60 s, dev0, release row.
- **Ownership settled for speed:** I offered to route the stamp through agent1's harness but told gemini
  they are the better owner since they hold the context and the negative result; agent2 cross-checks the
  159-site audit table as planned. Calibration and cross-lane bands remain frozen; **no device time in
  effect**; cards at baseline; main pushed.
- **Still the user's, unchanged:** Q3 artifact `bytes`/which file · §7.x waiver · root for PG-0c.
### STATE 2026-09-12 ~12:0xZ (AMD host) — gemini's shuffle audit **verified and reconciled**; their l2norm causal claim refuted against the loop it quoted; CPU-only next steps assigned
- **Audit is real and exact:** `8b6432a2` / `docs/amd/v340l/02_shfl_call_site_audit.md` (19 KB). I recounted
  independently: **148** `__shfl_` sites in `src/` excluding the shim + **11** inside
  `common/hip_shim/cuda_runtime.h` = **159**, matching their stated total precisely — so the table counts
  what was read, not an estimate. First artifact tonight that let another lane walk the same ground.
  **agent2 assigned to cross-check it; gemini told to coordinate with agent2 directly** and to fold
  follow-ups into the same file so the table stays one artifact.
- **Their conclusion refuted by the code they cited — both mechanisms:** (a) *"`k=1` never accumulated"*
  is false — with `d=128`, `pairs = 64`, and `pair = lane + k*kWarpSize` guarded by `pair < pairs`, k=0
  covers pairs 0–31 **and k=1 covers 32–63**; I evaluated the predicate across all k and lanes: **64 of 64
  pairs accumulate, the committed loop provably sums the whole row.** (b) *"lane 31 did not contribute to
  lane 0"* is false — in the down-butterfly **lane 30 reads lane 31 at offset 1** (`(30%32)+1 = 31 < 32`,
  guard admits it) and it propagates through offsets 2,4,8,16 into lane 0. So the source predicts
  `inv ≈ 1/sqrt(48.37) ≈ 0.1438` while the probe measured **0.2080** — **the match is real, its
  interpretation is wrong.**
- **Method point, stated because it is the pattern of the entire night:** they searched many candidate
  subsets and kept the one landing within 9e-6. **That validates arithmetic, not mechanism** — and
  23.11 sits only ~4.5% from exactly-half, so several "a chunk is missing" stories fit inside the probe's
  printed precision. Fourth confident static story tonight (mine, agent1's twice, theirs); the bar now is
  *name the mechanism the code actually contains, or hold it unresolved.*
- **Leading hypothesis, killable on CPU with zero device time:** if the loop is whole-row but the effective
  Σ is half-ish, **the row the kernel read is not the row the harness summed** — the same
  `{d,rows}`/`ne[0]`=feature-dim trap that already produced the NaN artifact and the `d=4` layer_norm run.
  Two ordered checks: (i) recompute the reference from the harness's actual input bytes using the **pure-C**
  converter — **not** `__bfloat16_as_ushort`, which is a numeric cast on ROCm (so a "full = 48.367" line
  built with it is itself suspect, not just the kernel); (ii) print the `d` and row count the tensor really
  carries and the pair range the reference summed. If kernel-`d` and reference-`d` differ by 2×, there is
  **no product defect at all.**
- **If both come back clean:** next stamp is a **per-lane partial dump** (each lane reports its own Σ
  pre-reduce), which separates "each lane summed half the row" from "the reduce lost a lane" in one
  60-second launch and cannot be argued with. Stamp ready on their word.
- **Delivery note, mine:** I verify outbound text by reading back the **stored** message now after #63
  shipped with its backticked anchors eaten by the shell — this one confirmed intact (148/11/159, the guard
  expressions, `__bfloat16_as_ushort`, per-lane dump all present).
- **State:** tolerance calibration and cross-lane bands still frozen (correct), **no device time in effect
  for any lane**, cards at baseline, main pushed. Open with the user unchanged: Q3 `bytes`/which-file ·
  §7.x waiver · root for PG-0c.
### STATE 2026-09-12 ~11:5xZ (AMD host) — G-AMD-10: **the machine overruled BOTH static reads**; "closed" retracted; wavefront-64 reduce path is an **OPEN PRODUCT DEFECT**
- **Result (one 0.4 s launch, freshness by construction, release clean, log `results/v340l/batch2c_run.log`),
  outcome = NEW EVIDENCE with mechanism unnamed, per protocol.** l2norm probe:
  **`inv = 0.2080`** vs `1/sqrt(half) = 0.2033` vs `1/sqrt(full) = 0.1438` ⇒ the kernel normalizes as if
  **about half the row's squares** were summed. Self-consistent across the row = truncated reduction, not
  noise. **This contradicts my geometry-tax read AND agent1's group-scoped-consistency rebuttal, which had
  explicitly predicted "neither duplication nor halving."** The argument that won the exchange was also
  wrong about the outcome.
- **I retracted my own retraction.** Ten minutes earlier I had written "closed: no tax" into docs/amd/README.md —
  correct to withdraw the falsified claim, **wrong to declare the question closed while the probe was still
  unrun**. docs/amd/README.md now reads OPEN PRODUCT DEFECT with the numbers and both wrong reads preserved, so the
  next reader doesn't repeat either.
- **Prime suspect, labeled suspect-only: our own shim's explicit-`Width` paths**
  (`common/hip_shim/cuda_runtime.h`, 3-arg default `width = 32` at :151). **Both static reads assumed that
  layer correct and neither analyzed it** — the only common blind spot, and it's the pattern: they argued
  about product code and skipped the compatibility layer. A width-limited butterfly landing on a
  full-wavefront shuffle, or a 5-step (16…1) butterfly covering 32 of 64 lanes, produces exactly this
  magnitude error on a **wavefront-64** device.
- **Binding consequences now, not after diagnosis:** (1) **gemini must not fit PG-B tolerance bands for any
  cross-lane-reduction kernel** — a band calibrated around a half-summing reduce certifies the bug;
  (2) **T3 attention is squarely in the blast radius** (densest shuffle use) so this closes **before**
  retile work; (3) concrete review = enumerate `__shfl_*_sync` call sites and check each `Width` against
  64 — the "correctness not throughput" hazard I raised *while losing* the geometry argument turned out to
  be the live one; (4) still **no touching `warp.cuh`'s `kWarpSize`**, CUDA-shaped by design and shared with
  the NVIDIA line, exception list stays two files.
- **Partial progress inside the same window, not conflated:** `sample.greedy==argmax` **GREEN** (the two
  kernels agree with each other; both disagree with the brute-force reference on one token) and argmax
  improved 2/2→1/2 wrong from the scan-limit fix. Next CPU-only task: re-read `argmax_better` tie/domain
  semantics **before** calling anything product there. So the scan-limit cause-name is partially proven,
  the harness isn't fully cleared, and one genuine defect is now on the table with a number attached.
- **Session handoff:** agent1's context is exhausted; final message received, G-AMD-10 fully closed,
  **nothing outstanding on device, no live grant, cards idle**. Next runner starts from
  `docs/amd/v340l/PROGRESS.md` + `batch2c_run.log` + this registry + `02_session_report_2026-09-12.md`.
  **The shim-width review is CPU-only and is the top item** — it can be resolved with no device time at
  all, and it gates T3.
- **Rule with three instances behind it now:** *quote the anchor, and don't declare closed until the probe
  reports.* Confident static reading failed twice in one lane tonight; measurement failed zero times.
### STATE 2026-09-12 ~11:4xZ (AMD host) — **my geometry-tax inference FALSIFIED statically by agent1's line anchor; corrected in the record**; G-AMD-10 stamped; his freshness protocol adopted as the standard
- **I was wrong and the correction is mine to publish.** My claim: `kWarpSize=32` + 64-lane wavefronts ⇒
  both halves of a wavefront redo the same row ⇒ ~half throughput. **False** — `l2norm.cuh:22-23`:
  `warp = threadIdx.x / kWarpSize`, `row = blockIdx.x * kWarpsPerBlock + warp`. Threads 32–63 are
  `warp=1` ⇒ a **different row**; the row comes from `warp`, not `lane`, and width-32 group-local reduces
  are correctly scoped. **No duplication, no half-sums, no tax.** I had registered it as "UNVERIFIED,
  needs a measured number," which was honest but still insufficient: an *argument* that a line refutes is
  not awaiting measurement, it is wrong. Corrected in docs/amd/README.md rather than left for the probe to settle,
  and credited to agent1's rebuttal — which arrived with the anchor, as a rebuttal should.
- **The durable hazard that survived my wrong one, stated precisely:** the pattern is safe **only because
  every collective call site passes an explicit `Width`**. Our shim's 3-arg default is
  `width = 32` (`cuda_runtime.h:151`), so the T2/T3 review target is a future `__shfl_*_sync` written
  **without** a width alongside code assuming a 64-lane reduce — a **correctness** risk on wavefront-64,
  not a throughput one. That is what gets looked at when the attention bodies with real cross-lane
  reductions land.
- **Rule earned, and it's the third time tonight:** *a static read is an argument, not a result — quote
  the anchor or don't make the claim.* Two false static claims in one lane (mine: geometry tax; agent1's:
  host bf16 converters, which survived as a narrower **real** defect with a repro), and **neither was
  caught by reading harder** — mine fell to a quoted line, his to a minimal repro.
- **G-AMD-10 STAMPED:** one launch, 60 s, dev0 only, of `/tmp/batch2_verify_fresh` (sha256 prefix
  `02ce1a6cfcf0`), covering (a) the v6 argmax/sample re-check that converts "cause-named,
  device-unproven" into proven-or-not, and (b) the l2norm discriminator. Release row verbatim; abort on
  foreign KFD PID or >30 s; **no iterating in-window** — either branch is a result.
- **His freshness protocol is now the standard, adopted over my version:** freshness means **"rebuilt from
  git HEAD in the same message"**, with marker + binary + archive hashes in the log, and the previously
  running `/tmp/batch2_verify` is **retired from the protocol** because it hashes differently. Better than
  my "print a marker" rule because it closes the stale-artifact class by construction rather than by
  observance. Docs/174's provenance line gets his wording. Reproducibility note accepted: the fresh hash +
  command are on the record, so a next runner can rebuild the identical binary even if his context dies
  first.
- **Accepted regardless of who was right about geometry:** no `warp.cuh` rewrite during the port; any
  measured geometry cost goes to the **user as a named decision**; T2 `%`-of-roofline rows carry
  bandwidth-vs-lane-bound labeling.
### STATE 2026-09-12 ~11:3xZ (AMD host) — G-AMD-8 answered (split); agent1's session scope COMPLETE with a report; l2norm narrowed by static read to harness-first, and a **possible ~2× lane-geometry tax** surfaced for T2
- **G-AMD-8 result, one 0.3 s launch, freshness proven in-log (marker + hashes), release clean.** Split
  answer: **`layer_norm` GREEN** (worst 0.0077/0.03) — the `{d,rows}` shape confusion was the real cause,
  exactly the mechanism I predicted against the "converter" theory. **argmax + sample: cause named in
  `argmax.cuh:65` — `valid_rows` is the VOCAB SCAN LIMIT, not the token count** (T=2 scanned 2 of 1024
  logits); v6 harness fixed and relinked. agent1 **voluntarily downgraded** those two from "green" to
  **"cause-named, fixed-in-harness, device-unproven"** pending the combined re-run — correct
  claimed-vs-proven discipline applied to his own summary.
- **My static read of the surviving `l2norm` red** (a read, labeled as such): with
  `kWarpSize = 32` + `kFullWarpMask = 0xffffffffu` on **64-lane wavefronts**, both halves of a wavefront
  compute the *same* row and each width-32 butterfly holds the **full** row sum — so the reduce is
  self-consistent and **cannot** produce "half the squares missing". Points at **harness row-slicing**
  (agent1's #2), and I told him to let the machine overrule me if it disagrees.
- **The finding nobody had:** that same geometry means these kernels **execute every row twice ⇒ correct
  output at roughly half the lane throughput**. Not asserted as cost — **must be measured on the first T2
  row**, which is also why T2 rows must name their denominator *and* say bandwidth-bound vs lane-bound,
  or the redundancy reads as a slow dequant. Ruled: **do not rewrite `warp.cuh`** during the port (it's
  CUDA-shaped by design, shared with the NVIDIA line, and the exception pair stays two files); if the tax
  measures large it goes to the **user as an optimization decision**, and cheap wins belong in launch
  geometry rather than the reduce.
- **agent1's delegated scope is COMPLETE** — report at `docs/amd/v340l/02_session_report_2026-09-12.md`
  (final `77557073`, tree clean). Claimed: scoping → lane landing → **7 grants, all closed with release
  rows** → first-token list (~26 `.cu` + 5 `.cuh`) → **T1 17/17 whitelisted, parity 154/154** →
  G-AMD-8 split answered → T2 opened 5/7 → **peer-probe matrix (the TP2 decision input)** → the
  product-header exception mechanism exercised exactly as ruled, **third file formally requested rather
  than applied**. Explicitly **not** claimed: the work order's §8 definition of done (steps 4/6
  unreached, step 5 re-scoped by no-P2P, and the WO's AR-as-translation premise recorded as invalidated).
  Registered in docs/amd/README.md: `{d,rows}`/`ne[0]`=feature-dim, `valid_rows`=vocab scan limit, and
  **"an artifact's provenance is proven by content, not mtime"** — his wording, credited.
- **Queue:** combined re-run stamp (argmax/sample proof + l2norm discriminator) is the next device
  request; **AR host-staged redesign stays the top open item**, because it's the difference between a
  plan for Q3-on-2-dies and a plan for one token. mma.cuh exception deferred until something on the path
  needs it. Device idle, no live grant.
### STATE 2026-09-12 ~11:3xZ (AMD host) — G-AMD-7 self-voided on a stale binary; the corrected finding is **CONFIRMED** as a real HIP divergence; G-AMD-8 granted with a freshness-proof condition
- **agent1 voided their own completed run**: the single launch executed a **stale binary** (two
  compile-only fixes rebuilt objects but skipped the relink), reproducing pre-fix numbers exactly — the
  tell arriving only after launch. The 4 reds are not results. Self-reported, release row clean at 0.4 s.
  Fix is structural, not attitudinal: freshness now proven by a **version marker string inside the
  binary + its hash**, because mtimes lie across incremental builds. Registered in docs/amd/README.md as the
  standard for stamped device runs: **a log without that line is void by definition.**
- **T1 is 17/17 whitelisted, parity 154/154, sha1 `7b815855e2`.** Guarded product edits landed exactly
  as conditioned — `math.cuh +18`, `memory.cuh +37/-1`, CUDA text in `#else`, zero reflow — the
  registered-exception pair holding with no drift toward a third file. A shfl 3-arg vs default-arg-4
  ambiguity was found and resolved in place.
- **Their corrected finding is REAL and I verified it independently.** `__bfloat16_as_ushort` in
  `amd_hip_bf16.h:537` is `unsigned short ret = h;` — a **numeric conversion** — while its own docstring
  claims it "Reinterprets bits", and on CUDA the same-named primitive genuinely does. Measured through
  our own shim: 1.0 → numeric `0x0001` vs true bits `0x3f80`; −0.5 → `0x0000` vs `0xbf00`; π → `0x0003`
  vs `0x4049`. **A function that contradicts its own documentation** is why this bites reference code so
  cleanly. Meanwhile `__float2bfloat16`/`__bfloat162float` are exact and RNE-correct (−1.0 → `0xbf80`),
  so the original "host converters unreliable" was a **misattribution** — the NaN came from uninitialized
  host memory under the wrong `{d,rows}` shape convention.
- **Process vindicated end to end:** my "does not reproduce" was correct about the function that was
  *named*, and the correction returned a narrower claim that survives verification — the repro rule earned
  its keep on the second pass. Had I propagated the first version, gemini would have rewritten a working
  reference path for no reason; both lanes now get the precise wording (never `*_as_ushort` for bits;
  conversions fine), and the same warning went to the q3 team, whose cosine bars are equally exposed.
- **G-AMD-8 granted** — 60 s, one launch, dev0, with two structural conditions: the run log must print the
  v5.1 marker + binary hash (freshness as artifact evidence, not assertion), and **no iterating inside
  the window** — either 7/7 (harness closed) or a named xnack/host-configs **product** item citing the
  exact sub-path against the `tp2_backend.cpp:4672` device-side contract. A third partial run is worth
  less than either answer.
- **Unresolved and scoped:** the `{d,rows}` feature-dim convention and token-major `[t*V+v]` argmax layout
  are mine to register (done, docs/amd/README.md). main still needs my merge of origin's 5 new commits before this
  push lands; the q3 branch has the divergence note queued for push too.
### STATE 2026-09-12 ~11:2xZ (AMD host) — batch-2 partial + window overrun handled; **agent1's "ROCm host bf16 broken" claim fails to reproduce in its own include path**; G-AMD-7 granted fresh
- **Window discipline:** batch-2 used ~14 min against a 10-min grant. agent1 self-reported it unprompted
  with clean release rows throughout — correct behavior — but the window is binding, so instead of a
  retroactive extension I issued **G-AMD-7 as a fresh 60-second stamp** (dev0, one launch, abort on
  foreign PID / >30 s). Rationale stated to the lane: a partial answer with a clean release beats a full
  answer that teaches the lane windows are advisory.
- **Result inside the window:** 4/7 batch-2 checks clean (position, embed-dense bit-exact, rope pair-norm
  exact, + batch-1's five); `l2norm`/`layer_norm`/`argmax-sampling` red.
- **Finding #1 NOT REPRODUCED — and it was about to become another team's constraint.** Claim: "ROCm 6.2
  host-side bf16 converters unreliable, `-nan` for `-1.0f`." I ran it twice: `hip/hip_bf16.h` directly,
  then **their exact path** — `#include <cuda_bf16.h>` from our shim, `.cu` TU,
  `hipcc -O2 --offload-arch=gfx900`, host call. Result: `__float2bfloat16(-1.0f)` → raw **`0xbf80`**
  (the correct bf16 encoding of −1.0), round-trip exact; π → `0x4049`. Converter is fine in this build.
  Their own finding #2 explains the NaN far better: `{d,rows}` vs `{M,D}` shape confusion ⇒ host
  reference indexed outside initialized memory ⇒ garbage that *looks* like a broken conversion. Told them
  plainly, gave the 20-line repro so they can check me, and asked for `batch2_run.log`/harness comment
  correction if it stands unreproduced.
- **Why I forced the issue: it's the second non-reproducible blocker tonight.** The first was mine —
  "no Q3 dtype exists," which I published to the q3 branch and then retracted (`61cd78d4`). Both were
  caught by running the thing, not by reasoning. New rule registered in docs/amd/README.md: **a toolchain-defect
  claim ships with a minimal standalone repro, or it ships as "observed anomaly, cause unresolved."**
  The warning I nearly sent gemini ("PG-B must never call HIP host bf16 converters") would have cost them
  a workaround for a bug that isn't there and eroded the warnings that are real.
- **Kept as durable facts (docs/amd/README.md):** launcher shape convention is **`{d, rows}`** with `ne[0]` =
  feature dim; **argmax logits are token-major `[t*V+v]`**; and **`gfx900:xnack-` means unified
  addressing is OFF**, so a kernel dereferencing a **host** pointer is a real semantic fork from CUDA —
  production contract is device-side configs (`tp2_backend.cpp:4672`).
- **Live hypothesis for the last red, endorsed and scoped for G-AMD-7:** `sample.greedy==argmax`
  (worst=2) surviving both layout fixes smells like a **product port item** — a sampling sub-path still
  dereferencing host `configs`. batch2b is asked to answer narrowly: if 7/7 with device-side configs,
  close as harness; if it persists, it's named, tier-tagged, and registered next to the PTX blockers
  because it will recur across every launcher ported after it.
- **Ruling-1 uptake noted:** agent1's ODR audit cleared option (A) on safety; the 465-commit drift
  argument decided (B) — and they said so explicitly, which is how the next person applies the right test
  to the next ruling. Q3 sizes stay unreconciled (11.1 arithmetic / 12.7 ledger / 13.67 plan) until the
  on-device manifest is read; they labeled their own 11.1 as arithmetic, not a manifest read.
- **State:** docs/amd/README.md + ledger through this entry; lane clean, device idle, G-AMD-7 awaiting their start.
### STATE 2026-09-12 ~11:0xZ (AMD host) — **G-AMD-5: NO P2P on this box**; AR design re-opened (my 1k-loc budget retracted); option (B) ruled for the two PTX headers; G-AMD-6 granted
- **Measured and independently corroborated:** `hipDeviceCanAccessPeer` = **0 for all 6 pairs, both
  directions, including same-card dies** → no device-to-device path; all cross-die traffic host-staged.
  Cross-card staging **6.61–6.70 GB/s** vs same-card siblings **4.86 GB/s** ⇒ **place TP2 cross-card
  (dev0+dev2)**. RTT flat with size (~98–101 µs/2 hops, ≈50 µs/hop) ⇒ cost is **per round trip**, so
  **collective count is the lever** (130 AR/step ≈ 13 ms/step unbatched = dead; 2–4-layer batching =
  3.3–6.5 ms/step = workable). **I verified their card grouping myself from CPU topology** — root ports
  `00:01.0`={card1,card3}={dev0,dev1}, `00:01.1`={card0,card4}={dev2,dev3} — three independent lines agree,
  so "same-card is fastest" is dead and device index ≠ card index is settled.
- **MY BUDGET RETRACTED, in public and in the other team's document:** my earlier "~1k loc to port the AR
  trio" is void. `one_shot_allreduce.cu`/`one_shot_argmax.cu` assume a kernel can write peer memory and
  ring a device doorbell; with zero P2P that is a **host-staged redesign** (pinned persistent host buffers,
  async double-buffering, no per-collective alloc/sync), not a translation. Published to the q3 team
  (pushed `60832e11`) because they are writing TP2 dispatch **this week** on a CUDA mental model of an
  on-package fabric, and every eliminated AR buys ~100 µs/step here.
- **Decision unchanged, said so explicitly to prevent overcorrection:** Q3/2-dies still wins — planning
  band **~27–30 t/s** vs their measured **18.3 t/s** single-card NVIDIA serve; single-die is impossible
  (13.67 GiB vs 7.82 GiB free; Q2-RTN fails their own cosine bar 0.778/0.764) and 4-die needs N-way TP
  that does not exist. **The probe changed the AR design, not the milestone.**
- **RULING (B) on the PTX product headers:** sanctioned edits to exactly `ops/common/memory.cuh` +
  `ops/common/math.cuh` behind `#if defined(__HIP__)`, `#else` preserving original text; **rejected the
  include-path overlay** even though agent1's ODR audit cleared it, because the NVIDIA line actively edits
  those paths (465 commits merged into main today) and a shadowing overlay drifts silently. Registered as
  a named 2-file exception for gemini's derive-with-exception-list PG-1; a third file needs a new ruling.
  Fidelity: zfill `min(src_bytes,Bytes)`+zero-tail; `ex2.approx→exp2f` is **approximation-class, not
  bit-identical** → rmsnorm/silu gates need tolerance bands (gemini told before it writes PG-B
  thresholds); CUDA-path equivalence stays **unprovable here, OPEN** on the merge package.
- **G-AMD-6 granted** (batch-2 numeric verification, dev0, 10 min, 30 s/launch, release row).
- **Numbers not to size KV off yet:** 11.1 GB (probe arithmetic) vs ≈12.7 GiB (their text-stack ledger) vs
  13.67 GiB (their plan of record, whose own doc says writer output is the source of truth). Asked the q3
  team again for the serving file + its `bytes`; nobody should compute headroom from a figure in prose.
- **Cross-team doc safety:** my earlier rename had **deleted the NVIDIA line's freshly-refreshed
  `COORDINATOR.md`** — caught before pushing (they refreshed at 06:25 while my split came from the 20:38
  snapshot). Resolved non-destructively: merged (`4835e559`), both docs coexist with an explicit
  provenance/recency note, zero deletions, pushed. main now `62579bc5`+; the pointer-death failure mode I
  keep writing about nearly fired from my own keyboard.
- **Also fixed tonight:** my push would have been destructive partly because I had **28 unpushed commits
  while main moved 465 ahead** — and I had been telling both lanes to "merge main" against a local ref.
  They share this machine's object DB so it worked locally; **it would have misled the q3 team, who pull
  from origin.** Lesson: when I cite a ref by name across a machine boundary, it must be pushed.
### STATE 2026-09-12 ~10:3xZ (AMD host) — **DIRECTION SET: Q3 on 2 dies (TP2)**; G-AMD-5 issued to measure the one number that can invert it; my dtype blocker retracted in writing
- **User confirmed the milestone:** TP2 for Q3 is prioritized for the AMD line over any TP4 work. I concur
  on the measured arithmetic — the original groupwise-int artifact **cannot** run on 2 dies (19.03/2 =
  9.52 GiB/die vs **7.82 GiB measured free**), while Q3 splits to 6.84 GiB/die leaving ~0.98 GiB/die
  (~1.97 GiB across the pair ≈ **87% of the 2.26 GiB** their single NVIDIA card gets — same ballpark,
  not a collapse). Any 4-die option additionally needs **N-way TP, which does not exist** here
  (`engine.cpp:309-313` = `size()==2` → TPEngine else single-device; `docs/59` row 8 keeps TP4+ HORIZON).
- **MY PUBLISHED ERROR, RETRACTED:** I told the q3 team "there is no Q3 dtype → new work order." Wrong —
  Q3/Q2 exist on the `NumericFormat`/`QType` axis (`typed_binding.cpp:23-24,50-53`: `Q3G64_F16S`,
  `Q2G64_F16S`); I grepped `src/core/dtype.h` only and generalized. Retraction written into their branch
  (`61cd78d4`, pushed) rather than quietly edited, because a false blocker sent to another team costs them
  planning, not just reading.
- **Two facts found in their code that de-risk the direction:** (1) their Q3/Q2 GEMV is **163 lines, zero
  asm, zero shared memory, and excludes `common/{memory,math}.cuh`** — it sidesteps the PTX blockers
  holding ~half our T1; (2) it is **row-split** (`q3_rowsplit_gemv`), which *is* the TP2 shard dimension,
  so their "not TP2-ready yet" is a dispatch/collective gap (`tp2_backend.cpp` has no `linear_add`
  reference), not a kernel rewrite. Our T2 (7 w8 GEMV files) also **leaves the critical path** under Q3.
- **GRANT G-AMD-5 issued** — `peer_probe.hip`, 20-min window: pairwise `hipDeviceCanAccessPeer` across all
  6 device pairs (which also settles the **die↔card pairing empirically** — only its shape was ever
  corroborated), cross-device copy GB/s **with the sampler running during load**, small-message ping-pong
  RTT with the convention named, and the **host-staged fallback** where P2P is absent. Ceiling rule and
  release row apply. **Rationale: today's evidence is negative (no kfd `p2p_links`, no `io_links`
  metadata, `hipInfo` absent) and TP2 decode is ~130 allreduces/step — we must not build a plan on the
  one number that can invert it.**
- **Coupling stated to both teams so nobody waits on the other:** they owe TP2 dispatch for Q3; we owe the
  **AR trio** (~1k loc, 6 PTX helpers) since any TP2 needs it at link *and* runtime. Unaffected and
  continuing: T1 glue (7 of 17 left), **T3 attention on the plain bf16-KV path**.
- **State:** pushed `61cd78d4` to `origin/wo/q3-gemv` after a clean rebase over their new debrief (both
  sections verified present, no clobber); main pulled clean; G-AMD-5 pending their start; **no other grant
  live**. Cross-team reply path: hub → `coordinator`.
### STATE 2026-09-12 ~10:0xZ (AMD host) — **CRITICAL PATH QUANTIFIED AND FIRST PORTING GRANT CUT**: ~26 kernel files to first token
- **The number the user was asking for:** agent1's `results/v340l/first_token_files.md` (`2dac8dae`) —
  **~26 `.cu` files (~8.9k loc) + 5 `.cuh` bodies to a first plain-decode token on ONE device.** Not 128,
  not 40–50. Tiers: **T1** 17 elementwise glue files (~1.3k loc, **zero asm**, near-mechanical) → **T2** 7
  w8 GEMV decode files (zero asm, bandwidth core, first % -of-183 anchor check) → **T3** attention, where
  `gqa_attention_decode.cu` itself is asm-free and the real work is 4 `.cuh` bodies with bounded
  `ldmatrix`/`cp.async` counts (2/4/12/13 — fragment-load helpers, NOT MMA compute) → plain
  vectorized global→LDS → **T4** prefill SIMT variants (which exist) + GDN scan → **T5 explicitly excluded**.
- **Biggest list-shaping unknown CLOSED, corroborated twice:** single-device serve exists. agent1 read
  `serve_options.cpp` `--device` and `engine.cpp:310`; I independently read
  `engine.cpp:309-313` (`devices.size()==2 → TPEngine`, else `Engine`) minutes before their message. Same
  branch, same conclusion: **first token needs no T5 file at runtime.**
- **RULINGS:** Q-A plain bf16-KV first, int8-KV (608-loc body) as T3b — a token on the plain path is the
  milestone. Q-B swiglu prefill takes the **zero-new-kernel** fallback route (slower prefill, no new risk).
  **T5 link-time = STUBS, not ports**, and the stubs must **fail loud if called** (abort naming the symbol +
  "TP path reached on a single-device build — routing defect"), never returning a plausible zero: porting
  ~1k loc of PTX rewrite to satisfy a linker buys an agreed-unececuted path, but a silent stub is a
  correctness landmine, and a tripped stub is a routing FINDING that promotes T5 to critical immediately.
  Stub set must be recorded in PROGRESS so nobody reads a linked binary as a complete one.
- **GRANT G-AMD-4 issued** — T1 batch 1: `scalar/scatter/cast/add_bias.cu` (36/49/46/62 loc), per-file
  compile, whitelist join only on compile, expected parity **141/141 + sha1**, dev0-only numeric
  verification through the PG-A harness pattern, 5-min window / 30 s per launch, release row verbatim,
  abort on foreign PID, ceiling-bound rule applies to any timing number, commit-before-report per batch.
- **§7.x reading settled without waiting on the user:** gate/test **authorship** stays gemini's; agent1
  running a device to check that a kernel it just ported computes correct numbers is product
  verification of a product file (§4), not gate authorship. Pairing: agent1 ports and verifies per batch,
  gemini wires the durable per-file PG-B gates as tiers land. Waiver question still put to the user
  because a yes is faster and cleaner — **but it is not a blocker and batch 1 is authorized.**
- **Also noted:** agent1 adopted the impossible-number rule from gemini's Check 5 catch ("would have
  poisoned any threshold"). That is the second lane internalising it, which is what makes it a norm
  rather than a memo.
- **State:** grants G-AMD-1..3 closed and released, **G-AMD-4 live** (dev0, minutes of device time). Trees
  clean at last check. Remaining user decisions: 13 GB artifact `weights_id` + location · §7.x waiver (now
  non-blocking) · root for PG-0c.
### STATE 2026-09-12 ~09:5xZ (AMD host) — **PG-A ACCEPTED 5/5**: device layer proven, and PG-0b × PG-A cross-validate to 0.7%
- **I re-ran it myself under G-AMD-3** rather than accept the fix commit: their commit changed the source
  but the committed log was still the stale bogus 04:47 run. Binary was current (no-op rebuild), ran
  dev0, rc=0, pre/post back to 18,575,360 B with "No KFD PIDs".
      Arena D2D sustained = **184.26 GB/s**  (3,355,443,200 B / 0.0182 s — arithmetic independently checked)
      bounded PASS [10.0, 350.0] GB/s, PG-0b ceiling sanity certified
- **THE MEASUREMENT RESULT OF THE NIGHT:** PG-A's **184.26** vs PG-0b's
  **183.34 / 183.26 / 182.58 / 181.09 GB/s** — two independent instruments (their hipEvent timer, my
  wall-clock probe with sysfs sampler) agreeing within **0.7%** on hardware nobody had executed before.
  Recorded as **mutually confirming**: the bandwidth anchor and the device runtime now validate each
  other, which is the strongest measurement statement this line has made.
- **Rejected-then-fixed trail on record:** my #46 rejection (103,755 GB/s = 567× the ceiling; "0.0000 s"
  = timing an enqueue) landed as `6081d2f4` — hipEvent timing + explicit sync + a **[10, 350] GB/s
  impossible-bandwidth guard**, which is the general rule made executable: *any measurement beating the
  measured ceiling is a broken instrument, never a result*. Standing invariant now in the lane.
- **Owed before G-AMD-3 closes:** (a) a **fresh committed log** — the tracked artifact still contradicts
  the tracked binary; (b) **Check 5's demonstrated red** (four checks had one, this one claims a falsifier
  mode with no log); (c) the **"SM count: 90"** question — Vega 10/gfx900 is 64 CUs, so name what that
  field actually is before anyone sizes occupancy from a mislabeled device property.
- **Hardware risk is now CLOSED, and that reframes the user's schedule complaint:** the runtime substrate
  (device ctx, alloc, byte-exact H2D/D2H over 16.7M floats, stream sync/event timer, graph
  capture→instantiate→launch with numeric accuracy) works on gfx900. **What separates us from a served
  token is kernel files + the artifact's format, not unknown device behaviour.** Critical path is
  therefore the first-token `.cu` list demanded from agent1 (symbol inventory → smallest ordered set for
  single-device plain decode), with short per-file grants instead of batteries.
- **State:** G-AMD-3 remains the only live grant and closes on (a)+(b). All trees clean; device idle;
  hub unread 0. User decisions: §7.x waiver for the AMD line (would unblock agent1-side device tests) ·
  13 GB artifact `weights_id` + location · root for PG-0c.
### STATE 2026-09-12 ~09:3xZ (AMD host) — GRANT G-AMD-3 ISSUED: first GPU execution on the AMD line, to the TEST LANE
- **Both my rulings were accepted before I sent them:** gemini merged main (`9e1fe05f`) and concurred
  with the **derive-never-enumerate** canonical form in WO §5; and it **retired the overstated claim**,
  adopting exactly my phrasing — "ZERO tracked **product/source** mutation" — with the scratch
  architecture named as the discriminator that preserves the invariant under any termination mode. That
  closes #35/#37/#40/#42 on those points. Its scratch sweep is real at `check_anti_resurrection.sh:22`.
- **GRANT G-AMD-3 issued** (hub #44) for PG-A single-device smoke, scoped exactly as requested:
  `HIP_VISIBLE_DEVICES=0 ./build-hip/tools/v340l/v340l_device_smoke 0`, **dev0 only**, <256 MiB peak on a
  7.98 GiB device, **30-minute window**, pre/post `rocm-smi`, abort on foreign PID or >30 s, own-PID
  exact-name kills only, release row pasted verbatim, and **no server / no artifact download / no second
  device**.
- **Preconditions I checked rather than accepted:** binary current against its committed source
  (`git diff df5e4c6e -- …device_smoke.cpp` **empty** — the stale-artifact check, which the lane passed);
  **non-vacuous by inspection** (`HIP_CHECK_OR_FAIL` → `return 1`; byte-mismatch path → `return 1`); device
  clear (`No KFD PIDs`).
- **Condition that matters most:** PG-A is **not complete without a demonstrated RED** — a passing smoke
  run proves only that the happy path executes, and this project's most expensive defects were all
  green-because-empty. A red is owed beside the green log (`results/v340l/falsifiers/`), cheapest form an
  out-of-range argument or a corrupted expectation.
- **Also asked:** record which physical card actually carried the load during their pre/post sysfs
  sampling. That would **independently confirm agent1's dev0→card1 mapping**, of which I have only
  corroborated the shape (4 amdgpu cards at 0/1/3/4, card2 = nvidia) and never re-derived the pairing —
  first-hand either way, and it retires an inference from the ledger.
- **Risk named honestly to the lane:** Check 4 (graph capture/instantiate/launch) is the first
  device-side machinery ever executed on this box. A hang there is **information, not failure** — hence
  the 30 s abort, the exact-name kill rule, and the instruction to report which check hung rather than
  silently retry.
- **State:** ledger through `b86863ab`; all three trees clean; hub unread 0 after #41/#43 consumed;
  **G-AMD-3 is the only live grant**, dev0, 30 minutes. Blockers for the user unchanged: 13 GB
  `weights_id` + location · root for PG-0c.
### STATE 2026-09-12 ~09:2xZ (AMD host) — RULING: contract text frozen in DERIVED form; gemini's enumeration refused (their own code had outgrown it)
- **Requested:** #30 asked me to centralise the PG-1 three-part static-property text in WO `docs/amd/v340l/01`
  §5 so future lanes inherit one spec. **Done at `218c3ae6` — but not as worded**, because their item
  (b) says additions are "strictly limited to the enumerated whitelist (4 files)" while their shipped
  gate at `7bf82b40` **derives** allowed additions by construction (`src/common/hip_shim/**`, shared CMake
  files, `src/HipSources.cmake`). Freezing the enumeration would have canonised a spec the implementation
  no longer satisfies — and hardcoded allow-lists drift on the product lane's cadence (observed red at
  `4ddefff6`), each drift inviting someone to widen a pattern until it passes. **§5 now reads: derive,
  never enumerate; the hardcoded form is explicitly rejected as a specification.** Invited pushback with
  the SHA, since this reverses what they asked for.
- Each part carries its falsifier obligation, plus the visibility clause earned from my own botched test:
  the planted case must be **visible to the diff the gate reads** — an untracked file is invisible to
  `git diff --diff-filter=A`; use `git add -N` or a scratch commit.
- **Verified their fixes rather than accepting them:** the orphaned-scratch sweep I asked for is real at
  `check_anti_resurrection.sh:22` (`find /tmp -maxdepth 1 -type d -name 'anti_resurrect_scratch_*'
  -mmin +120 …`, commit `881dc564`), and `7f5da410` documents the non-mutating architecture in PROGRESS.
- **Two repeats flagged for the third time:** their close-out cited `59fc3ac2` as HEAD when HEAD is
  `881dc564` and the former is an ancestor (`merge-base --is-ancestor`) — third moving-ref citation of
  the night, which is why §11.6 now says cite by NAME, pin by content assertion; and "ZERO other cells
  mutate tracked files" persists, contradicted by `git ls-files` on
  `results/v340l/ci_amd_verdict.json` + `falsifiers/{guard,stage0}_verdict.json`. Asked for either a fix
  or retirement of the sentence: **the ledger and the code must agree, or a future auditor discounts the
  whole claim on finding one exception.**
- **My own pointer bug caught immediately after writing it:** I cited §7.1 of the WO, which does not
  exist; corrected to §6 Step 2 and re-verified all section refs resolve. The dangling-pointer hazard I
  have reported in other people's documents would have shipped in mine.
- **State:** ledger through `218c3ae6`; all three trees clean; no GPU grant in effect; hub unread 0.
  Blockers remain the user's: 13 GB `weights_id` + location · root for PG-0c.
### STATE 2026-09-12 ~09:1xZ (AMD host) — gemini's option-(b) falsifier verified two ways; the durable lesson is that their kill-test proves the DESIGN, not the trap
- **Verified by code read AND by my own SIGKILL test.** Mechanism: `mktemp -d` scratch, `trap … EXIT INT
  TERM`, canonical file taken from the **git object store** (`git show main:…`), mutation appended to a
  scratch copy, `diff -u` between scratch files — the tracked tree is never opened for write. My test:
  killed their `--falsify` run with `kill -9` at ~1.2 s → **tracked tree 0 dirty**.
- **DURABLE POINT, filed because it generalises:** SIGKILL is uncatchable — no trap ever runs. So their
  kill-test validates the **scratch design**, and a trap-only design (my option (a)) would have FAILED
  that same test. PROGRESS should say so, because if someone later "simplifies" back to in-tree mutation
  plus a trap, the kill-test is the discriminator. **A test that passes for the wrong reason is how
  hazards come back.**
- **Residue from my own run (mine to report, theirs to sweep):** the killed run left
  `/tmp/anti_resurrect_scratch_JCewBh` — removed by me, I created it. Under SIGKILL cleanup cannot run, so
  killed/crashed runs accumulate scratch dirs indefinitely. Suggested entry sweep
  (`find /tmp -maxdepth 1 -type d -name 'anti_resurrect_scratch_*' -mmin +120 -exec rm -rf {} +`): harmless
  today, a full `/tmp` blamed on the GPU work next to it later.
- **One claim corrected with evidence:** #28's "ZERO other scripts mutate tracked files during test or
  falsification" is overstated — `results/v340l/ci_amd_verdict.json`, `falsifiers/guard_verdict.json` and
  `falsifiers/stage0_verdict.json` are **tracked** (`git ls-files`) and rewritten by every run; their own
  commit says "Refreshed logs", and I twice observed `M …ci_amd_verdict.json` from my own runs (restored
  both times). Accurate claim: **no tracked product/source mutation.** The artifact-hygiene item from #35
  stays open, and it is listed for a real reason: a tree that is never clean is one whose cleanliness
  nobody — including their own gate — can later assert.
- **My earlier chronology ask is DONE:** their HEAD `7f5da410` records the non-mutating falsifier
  architecture in PROGRESS, so a cold reader of #26/#28 (both describing now-superseded ancestors) will
  not chase a hazard that no longer exists.
- **State unchanged and re-verified:** no grant in effect, no GPU touched by me, all three trees clean,
  hub unread 0. Blockers remain the user's: 13 GB `weights_id` + location, root for PG-0c.
### STATE 2026-09-12 ~09:0xZ (AMD host) — night close-out: both lanes clean; the negative-test rule sharpened at mechanism level (parity-stamp freshness)
- **agent1 recorded the rule at `711f0e71` (tree clean, PROGRESS:232) and extended it past my own
  formulation**: a **stale parity stamp is the same trap as my untracked probe**, because
  `file(GENERATE)` writes `SOURCES_FILE` at *configure* time while the archive comes from the *build*
  that consumed an older list. Validating a guard's verdict against a tree state the build never saw
  yields a confident wrong answer. **Standing check for this lane: the stamp and the archive must come
  from the same configure/build pair** — costume #3 of untracked/staged/worktree, at mechanism level.
- **Count-only degradation confirmed honest** by reading the code after my own `tail -3` clipped it:
  `CheckHipArchive.cmake:79` emits `WARNING "COUNT-ONLY MODE (weaker check — names NOT verified;
  prefer SOURCES_FILE)"` before the OK line, and a wrong count fails loud (`SRC_COUNT=27` vs a 102-object
  archive → PARITY FAIL). Recorded as a sixth self-check: I nearly reported the warning as absent, from
  the truncated output of my own command.
- **Night state, verified not assumed:** both worktrees clean, nothing in flight, **zero GPU use by me**,
  **no grant in effect to any lane**, cards idle, no unread mail. Ledger through `a489b7bb`; convention
  registry at `f80084c9`. Resume protocol agreed by both lanes: **merge main first**, then agent1's
  `targets/*`+`serve/`+`apps` compile-only slice, then per-file w8 GEMV joins under the guard; gemini
  proceeds without waiting on me and owns the PG-A grant request once its smoke binary exists.
- **Six near-misses tonight, all mine, all caught by re-running rather than reasoning harder:** wrong
  file path · grepped a platform-dispatcher header instead of `amd_detail/` · read `tail`'s exit status
  through a pipe (×2) · printed "(empty = …)" labels next to greps that had matched · tested a gate with
  an untracked file it could not see · clipped a warning's own output. Every entry in this ledger
  carries the command that checked it, which is the only reason the count is six near-misses and zero
  wrong claims delivered to a lane.
- **Open, and they are the user's:** 13 GB artifact `weights_id` + location (groupwise-int ⇒ M2 is a
  real 2-device load needing the w8/GEMV family + loader path, not the ~40–50 kernel remainder) ·
  root for PG-0c's pinned-clock peak, else ~183 GB/s sustained stands as the measured ceiling.
### STATE 2026-09-12 ~08:5xZ (AMD host) — gemini's lane REVIEWED CLEAN; my own invalid falsifier caught before reporting; rule earned: a negative test must prove it can see its subject
- **Review outcome: gemini's test lane is clean, every claim verified by execution at HEAD `7bf82b40`.**
  NIT 1 fixed (`:164` prints the truthful "Refusing: no GPU grant present in writing from
  coordinator"); NIT 2 fixed (`zero_gpu_requested` / `gpu_stages_executed` split + "PASS (zero-GPU
  subset)"); both falsifiers demonstrated RED; the tracked-file mutation replaced by a **scratch copy**
  with `trap … EXIT INT TERM` installed before it (my #27 closed better than asked); and
  `git diff wo/v340l-hip --name-only` outside `tools/ops`, `results/v340l`, `docs/v340l` is **empty** —
  zero product edits held all night. Their dynamic-derivation refactor (my forward ask, implemented
  while I was investigating) is **not vacuous**.
- **My fifth near-miss, and the most instructive.** To prove the new derivation too broad I planted
  `src/ZZZ_coord_probe.cpp`, ran the gate, got **rc=0** — and did not report it, because the test was
  invalid before it ran: an **untracked** file is invisible to `git diff main --diff-filter=A`, so the
  gate was never asked. Redone with `git add -N`: `rc=1`, `FAIL: 1 unauthorized addition(s) in src/.`
  **Rule earned, filed with both lanes: a negative test must first prove it can see the thing it claims
  to test.** Untracked-vs-tracked / staged-vs-committed / worktree-vs-index are that trap in three
  costumes, and it lands squarely on the archive-parity cell: if a guard validates against a tree state
  the build did not consume, it prints a confident wrong answer, which is worse than silence. Cheap
  form: assert the diff the guard reads is the diff the build consumed, before trusting the verdict.
- **Chronology hazard noted to them:** #26 describes `ad857772`, which `--is-ancestor` confirms is an
  ANCESTOR of HEAD — so its "injects directly into src/runtime/tp2/tp_engine.cpp" is superseded by their
  own later refactor. A future reader taking #26 as current would believe the tracked-file hazard is
  live. Asked for one line in PROGRESS naming which falsifier design is current.
- **Night tally of MY wrong answers: 5** — wrong path, dispatcher header instead of the real one, tail's
  exit status through a pipe (×2 messages), a grep label contradicting its own match, and the
  untracked-file invisibility above. **All five caught by re-running, none by reasoning harder.** That
  is the operating lesson of this ledger entry, and it is why every claim in it carries the command that
  checked it.
- **State:** no GPU touched, no grant in effect, cards idle, no unread mail, both lanes at clean trees
  and told to merge main on resume. Awaiting user: 13 GB `weights_id` + location; root for PG-0c.
### STATE 2026-09-12 ~08:4xZ (AMD host) — additions convention landed and REGISTERED in docs/amd/README.md; agent1's session at a clean handoff
- **Verified:** `db1dc50e` real, tree clean under `-uall`, and the convention is genuinely in
  `docs/amd/v340l/README.md` (§+shim:/+whitelist: subject tokens · hip_shim/** = product namespace permitted
  by construction · whitelist machine-readable from `src/HipSources.cmake` · red-addition-gate after a
  drift commit = **ping before widen**). Registered centrally in **docs/amd/README.md** so a third lane or a cold
  start inherits it rather than rediscovering it the expensive way.
- **Self-report I could not check, and the reason is mine:** agent1's "16 commits landed today". My
  first count said 7 — because `git log --since` parsed my window in **local EDT**, silently starting the
  day at 04:00Z. Corrected to ISO-strict + UTC: agent1's branch has **24** commits dated 2026-09-12 UTC,
  gemini's **30**, both inflated by merges carrying the other's and the coordinator's work, so "16" is
  plausible as an author-filtered subset but **unverifiable as stated** (no window, no author filter).
  Not filed as a defect — it is not load-bearing — but recorded as a habit: **session tallies should cite
  window + author filter, or omit the number.** Same rule applies to me; the count I reported at first was
  wrong purely from a TZ artifact, which is exactly how a phantom discrepancy gets created.
- **Load-bearing claims all verified regardless:** HEAD `db1dc50e`, tree clean, "nothing in flight",
  PROGRESS current, WO corrected per every ruling. That is a clean handoff state, which is what §4 asks
  for at session end.
- **Lane status:** agent1 grant-free slice next (`targets/*` + `serve/` + `apps` → shrinks the 135
  unattributed symbols to a kernel-only remainder), then per-file w8 GEMV ports under the join rule.
  gemini owns PG-A device smoke and must request its own grant with scoped commands. **No grant in
  effect; no GPU touched; cards idle.**
- **Awaiting user:** 13 GB artifact `weights_id` + location (format is the blocker; a 2-device M2 needs
  the w8/GEMV family + loader path, ~40–50 kernel files is the full remainder) · root access decision
  for PG-0c pinned-clock peak.
### STATE 2026-09-12 ~08:3xZ (AMD host) — PORT SIZING ESTABLISHED; gemini closed my whole list (3 verified by execution); one defect ask RETRACTED; forward drift warning filed
- **THE HEADLINE FOR THE USER: the AMD port now has a countable size instead of an argument.** agent1's
  symbol inventory (`results/v340l/step2_symbol_inventory.md` @ 82eccab5, wired into WO Step 3 @
  176af8ec, verified by me): **engine HOST half is DONE compile-wise** — 75 ops/wrapper/plan/dispatch/kvarn
  `.cpp` clean under HIP, **102/102 objects, name-sets equal**. Remaining ≈ **40–50 of 284 kernel files
  (~1/6 of the kernel tree)** after the D2 minimal-path filter (upper bound 128 files / 300 symbols),
  GDN input-proj the largest family, AR trio carrying measured PTX-asm blockers. 135 unresolved symbols
  are target/serve/app-layer, **not** kernel work — so the estimate is not inflated by unported layers.
- **gemini closed my ENTIRE open list, each verified by execution not by report:** (1) the tracked-file
  mutation is GONE — scratch copy under `SCRATCH_DIR` with `trap 'rm -rf' EXIT INT TERM` installed at :29
  BEFORE the mutation at :33, i.e. they took my preferred option (b) unprompted, which also removes the
  cross-lane collision; (2) falsifier verdicts isolated — I ran `--falsify-stage stage0`, true rc=1,
  canonical `ci_amd_verdict.json` untouched, evidence in `falsifiers/stage0_verdict.json`; (3) my
  presentation nit adopted verbatim: `CI Result: PASS (zero-GPU subset)`, rc=0, Stages 0+1 genuinely run.
  **Extra credit:** their new Test 5 wires the equal-count/name-swap (`BOGUS.cpp.o`) falsifier
  independently — the same construction I used by hand to prove the guard's name-set claim. Two of us
  converging on it is the best evidence it was the right check.
- **RETRACTION, mine, recorded so no lane acts on a phantom:** #29 asked gemini to fix a stale
  "src/ zero diff" contract line. Searched `docs/amd/v340l/` and `tools/ops/` — **no such line exists in any
  file**; it was only in their message prose, and their CODE was correct from the start (three-part form).
  Ask withdrawn.
- **FORWARD DRIFT WARNING filed with both lanes:** I watched gemini's lane go **correctly RED** on
  agent1's merge `4ddefff6` ("4 unauthorized additions in src/") — the guard doing its job — and they
  closed it at `fd7eaa1b` by adding the four names. That works once: `ALLOWED_SRC_ADDITIONS` is a
  hardcoded 8-entry array, and agent1 adds shim headers **per kernel port** by design, so the gate will
  fire on that cadence, and every recurrence tempts someone to widen a pattern until it passes
  (vacuous-gate failure mode in a different hat). Recommended: **derive, don't duplicate** — permit
  `src/common/hip_shim/**` by construction and read the rest from `src/HipSources.cmake`; meanwhile
  agent1 will flag shim additions in its commit subject so a red gate reads as expected drift, not
  regression. Same principle as the parity guard.
- **Minor hygiene:** `ci_amd_verdict.json` is TRACKED, so every run dirties the tree — my runs included;
  I restored it both times. Principle: an act of verification should not change the thing verified.
- **Discipline note:** the tree moved under my measurements a THIRD time tonight — I found the break at
  `4ddefff6`, and gemini had already fixed it at `fd7eaa1b` before I could report. Had I messaged my
  first reading, I would have filed a resolved defect as live. The guard against that is re-running the
  check immediately before speaking, which is what I did.
- **No GPU used by me tonight; no grant in effect to any lane; cards idle; no unread mail.** Awaiting:
  user's 13 GB `weights_id` + location; root decision for PG-0c; gemini's PG-A grant request when its
  smoke binary exists; agent1's `targets/*`+`serve/`+`apps` compile-only slice.
### STATE 2026-09-12 ~08:2xZ (AMD host) — agent1's discrepancy resolved; PARITY GUARD proven by my own name-swap falsifier; PG-A grant route corrected to the test lane
- **Central worry closed, verified directly:** `tp_engine.cpp.o` IS in `libninfer_hip_host.a` (with
  `tp2_backend.cpp.o`, `sampling.cpp.o`). The VRAM-law canonical home compiles into the HIP artifact,
  so the parity claim and the anti-resurrection cell rest on a file that is actually there. Its
  sequence (18/18 → 22/22 uncommitted when I measured → +5 RED loudly on nvtx3) matches everything I
  saw, including the build-dir mtime moving under me: stale archive beside an in-flight longer list,
  **not** a live silent drop.
- **I falsified the "name-set compare" claim myself and it HOLDS** — worth proving because that claim
  is exactly what quietly stays a count compare: correct list → rc 0 `102/102 name-sets equal`;
  **same count with one name swapped → rc 1 `PARITY FAIL ... missing: BOGUS.cpp.o`**. Equal counts,
  differing sets, correct RED. Its semicolon-flattening catch (`add_custom_command` mangling
  `-DSOURCES` so the guard saw "1 source vs 27 objects") is the stretch's best find — that defect
  makes the guard the loudest kind of wrong, and only shows up if you run the thing.
- **Two notes filed:** (1) the guard still accepts a **count-only `SRC_COUNT` fallback** — a weaker
  check that must never be logged as a name-set result (refuse it on the POST_BUILD path, or print
  `COUNT-ONLY MODE`); (2) **moving tree, second time tonight** — `M src/HipSources.cmake` uncommitted
  with the archive at 102 objects while HEAD claims 27/27. Both true at their moments, but twice my
  read of "the lane's state" and its read of "what I verified" described different objects. Ordered:
  **commit before you report**, §4 is not only about session ends.
- **PG-A grant route CORRECTED in agent1's favor:** it declined to request the device-smoke grant
  because the smoke **binary is the test lane's product** — that is §7.x applied properly rather than
  by convenience, endorsed. gemini told (#33) it may request a grant directly, CC'ing agent1; agent1
  stays compile-only. Also endorsed its next step: the **undefined-kernel-symbol inventory** is the
  most useful thing it can hand over — it turns "how much port is left" from argument into a countable
  list and sizes Steps 3–5 for the user.
- **docs/amd/README.md extended by me** (my doc, my edit) with the measured facts that keep biting: the
  `cuda_bf16.h`→`cuda_fp16.h` transitivity asymmetry (CUDA pulls it, ROCm does not), `__nv_bfloat16`
  absence on the AMD path vs. scalar conversions being available, bf16 arithmetic compiler-dead on
  gfx900, the non-identity dev→card map, the silent-drop class + guard, and the pinned toolchain with
  pip forbidden.
- **Unchanged:** no GPU used by me tonight; **no grant outstanding to any lane**; cards idle. Still
  waiting on the user for the 13 GB artifact's `weights_id` + location, and on root for PG-0c.
### STATE 2026-09-12 ~08:1xZ (AMD host) — first reproducible GREEN on the AMD CI lane; one new defect: falsifier runs clobber the lane's verdict artifact
- **First lane-wide green I could reproduce myself**, zero-GPU so grant-free:
  `run_ci_amd.sh --zero-gpu` → **true rc=0**, Stage 0 (whitelist + anti-resurrection) and Stage 1
  (real `cmake` configure + build of `ninfer_hip_host` + `bw_probe`) both genuinely executed,
  `stages_total: 2`, `gpu_stages_executed: false`. And `--falsify-stage stage0` → **true rc=1** with
  all 4 negative tests RED, wired through the aggregator rather than only the gate script. Both of
  gemini's #22 claims verified by execution. **Nit 2 already fixed by them**: the JSON now carries
  `zero_gpu_requested` AND `gpu_stages_executed` instead of one ambiguous field — recorded as theirs.
- **NEW DEFECT, found by running it:** a single verdict path (`run_ci_amd.sh:185
  VERDICT_JSON=${RESULTS_DIR}/ci_amd_verdict.json`) means **a falsifier run overwrites the lane's
  canonical verdict**. After my `--falsify-stage` run the file read `{FAIL, passed 1, failed 2}` while
  HEAD read `{PASS, passed 2, failed 0}` — "read the verdict artifact" and "know the lane's state"
  stop being the same operation, and falsifier runs are the most frequent thing a gate author does.
  Required: falsifier evidence goes to `results/v340l/falsifiers/`, or the aggregator refuses to
  overwrite a PASS record from a `--falsify` invocation. **I restored the file to the committed state
  and confirmed the worktree clean** — I dirtied it, I cleaned it, and I said so in the message.
  Presentation nit also raised: `CI Result: PASS` for a zero-GPU subset run should read
  "PASS (zero-GPU subset)" so a human scanning logs cannot mistake it for lane-green.
- **Reaffirmed, still unaddressed from #27/#29:** the anti-resurrection falsifier mutates tracked
  `tp_engine.cpp` and reverts via bare `git checkout --` with **no `trap`**; my own runs proved the
  mutation path is live, which also means two lanes falsifying concurrently collide on that file.
  Scratch-copy rewrite remains the preferred fix. Plus #20's stale contract line ("src/ zero diff")
  contradicting their own correct implementation.
- **No GPU used by me tonight** — every check above was compile-only, script inspection, or sysfs
  reads. Cards idle, no grant outstanding to any lane.
- **Open:** agent1's clean-build resolution of 27-listed-vs-22-archived (missing set includes
  `tp_engine.cpp`) · gemini's trap/kill-test or scratch rewrite + verdict-path fix + contract line fix
  · **user: 13 GB artifact `weights_id` + location** · **user: root for PG-0c pinned-clock peak** ·
  agent2 idle with no task.
### STATE 2026-09-12 ~08:0xZ (AMD host) — gemini's PG-1 promoted to DEMONSTRATED on my own reproduction; new defect: its falsifier mutates tracked `tp_engine.cpp` with NO crash-safe restore
- Mail #18 arrived as another REPLAY (superseded by #20 FINAL, already accepted). Unread now drained
  (0 pending) — the backlog was my own failure to mark consumption during script reads earlier.
- **FALSIFIER ASK MET, verified by reproduction not by report.** I ran
  `gate_pg1_whitelist.sh --falsify` in `wo/v340l-phase-gate`: **true rc=1**, "ALL 4 NEGATIVE TESTS
  DEMONSTRATED RED (FAIL LOUD)", and `results/v340l/falsifiers/ctest_guard_falsifier.log` is the
  zero-test-count proof I demanded. **PG-1 is therefore DEMONSTRATED, not designed-not-demonstrated —
  the first lane on this AMD line to clear that bar with evidence I could reproduce myself.** Recorded
  as gemini's. Two nits from #25 still stand.
- **DEFECT FOUND BY RUNNING IT (not by reading it): the anti-resurrection falsifier mutates a TRACKED
  file in a live worktree and the restore is not crash-safe.** It injects a stale VRAM budget constant
  into `src/runtime/tp2/tp_engine.cpp` and reverts it at `check_anti_resurrection.sh:39` with a bare
  `git checkout --`. **`grep -c trap` in that file = 0.** Anything that exits between injection and
  line 39 — SIGINT, a kill, any error under `set -e` — leaves a **VRAM-LAW violation planted in the
  canonical-home file** in a shared checkout, where the next `git add -A` sweeps it into a
  "measurement-only" commit. This project has already lived that exact failure (09-03 stray
  digit-prefixed JSON → silently broken CI verdict block, docs/147). The happy path restores — I
  confirmed clean after my own run — which is why nobody catches this in normal use; the bug is on the
  unhappy path, the only path a guard's hygiene is judged on.
  REQUIRED: (a) `trap '<revert>' EXIT INT TERM` installed BEFORE the mutation + a demonstrated
  kill-test (SIGTERM between write and revert, worktree provably clean after), or (b) preferred —
  **never mutate the live worktree**: inject into a scratch copy outside any worktree and diff that.
  (b) also removes the cross-lane race on one shared file. Asked whether anything else in the lane
  mutates tracked files. Sent as hub #27 with the fix + kill-test evidence requested.
- **Method self-note (second occurrence, same shape):** I twice printed a conclusion label next to a
  grep — "(empty = guard not wired)", "(empty = not in shipped code)" — where the grep had in fact
  MATCHED, and the label was simply wrong. A label is not evidence; the match text is. Both corrected
  before any claim left my session, and both findings I reported that turn were the opposite of what
  the stale labels implied.
- **Still open:** agent1's clean build + resolution of 27-listed-vs-22-archived (`tp_engine.cpp` among
  the missing) · **user: which 13 GB artifact (`weights_id` + location)** — format not size is the
  blocker, and 13 GB fits ONE card = 2 devices, upgrading M2 from fixture-based to real ·
  **user: PG-0c pinned-clock peak needs root** · gemini's fix commit + kill-test · agent2 idle, no task.
### STATE 2026-09-12 ~07:5xZ (AMD host) — agent1 SELF-RETRACTS a silent-drop defect; guard verified by my own falsifier run; **OPEN: 27 listed vs 22 archived, tp_engine.cpp among the missing**
- **agent1's retraction (465ddeab + a82f0b8c), accepted on evidence not on report.** Root cause it found:
  **CMake's HIP language does not claim `.cu`, so a HIP-only project silently DROPS those sources and the
  build still exits 0.** Its first two "green" HIP builds shipped an archive with 16/21 objects —
  `device.cu`/`arena.cu` were never compiled in anything it had called passing. Caught by an object-count
  audit, in its own lane, and led with the retraction. That is the behavior §7 asks for.
- **Verified by me, independently:** `LANGUAGE HIP` assignment present (HipSources.cmake:71);
  **`device.cu.o`/`arena.cu.o` now genuinely in the archive**; AR trio confirmed a `# NOTE` (line 40) not
  a whitelist entry — removal correct, since a listed-but-uncompilable file would gut the guard's meaning.
- **Its new parity guard passes MY OWN falsifier, zero-GPU:** `SRC_COUNT` match → rc 0 printing
  "parity OK: 22/22"; deliberate mismatch (99) → **rc 1** FATAL "A source was silently dropped (check
  LANGUAGE assignment for .cu files)"; no args → rc 1 (no default-pass); nonexistent archive → rc 1.
  Wired `add_custom_command(TARGET ... POST_BUILD)` at HipSources.cmake:81-84. A fail-loud guard proven
  red without touching a device — the standard, met.
- **OPEN, and it is the important one:** at 03:51 the tree showed **27 sources listed vs 22 objects
  archived**, with no object for `sampling.cpp`, `tp2_backend.cpp`, `tp2_request.cpp`,
  `host_kv_parked.cpp`, **`tp_engine.cpp`**. Either a stale archive from in-flight work (I watched the
  build-dir mtime move under me and deliberately did NOT rebuild in its worktree) or the same class
  still live on the five runtime files. **Not filed as a defect on a moving tree** — but `tp_engine.cpp`
  is the VRAM-LAW canonical home AND the anti-resurrection diff target, so a green build shipping
  without it would undermine both the parity claim and the resurrection cell. Ordered: clean
  configure+build, paste the guard's own N/N line, state plainly whether "18/18" refers to a superseded
  list, and consider printing the source-list hash so a stale archive cannot masquerade as current.
- **My own two failed checks, recorded (both corrected before I said anything about their work):**
  (1) twice I ran `cmd | tail; echo rc=$?` and read **tail's** status, not cmake's — the pipe-eaten-exit
  class this ledger names, which I had just written to gemini about in the same hour; (2) my first
  object/source comparison stripped `.o` from `device.cu.o`-style names and reported all 27 missing — my
  bug, not the tree's. **Near-miss count today: 4. The pattern is that every one was caught by re-running
  the check, not by re-reading the prose.**
- **Also this turn:** mail #14/#16 arrived as REPLAYS (earlier script reads didn't mark consumption; the
  watcher has since drained and consumed them — no action). Gemini's shipped PG-1 lane audited and
  **run by me**: Stage 0 PASS incl. the bf16 compile probe, Stage 1 builds HIP targets, Stage 2 correctly
  REFUSES without a grant, console FAIL == JSON verdict == **true rc 1**; both of my corrections present
  (refined src/ granularity; non-zero-test-count guard at run_ci_amd.sh:75-83 — my first grep said it was
  missing, my second found it). Zero src/ edits confirmed per-commit. Two nits sent (#25): a
  "Checking GPU availability…" line that checks nothing, and `zero_gpu:false` in the JSON when no GPU
  stage ran. **Real ask outstanding: prove the two guards RED** — force a zero-test selection and a
  deliberately reverted tp2 hunk.
- **Retracted my own overstatement, per §7 both directions:** scalar `__float2bfloat16` DOES exist in
  ROCm 6.2 amd_detail; my conversion probe failed on the undeclared **type name**, not the intrinsic.
  Correct rule: storage conversion available under HIP-native names, `__nv_*` names absent until
  typedef'd (agent1 took option (b)); bf16 ARITHMETIC remains compiler-dead per D3.
- **User decisions outstanding:** the 13 GB artifact's identity/`weights_id` + location (format, not
  size, is the blocker — no IQ3 `DType` exists in `src/core/dtype.h`; 13 GB fits ONE card = 2 devices at
  7.98 GiB/device, which upgrades M2 from fixture-based to real); PG-0c pinned-clock peak needs root.
### STATE 2026-09-12 ~02:1xZ (AMD host) — agent1's answers ratified; 2 measured findings from COMPILE-ONLY probes (zero GPU, no grant); 2 of my own false starts retracted
- **Ratified:** target-name ownership split (agent1 keeps §6 spec names; **gemini owns the ctest
  labels/targets**, with the non-zero-test-count guard line added to §6); zero-diff granularity —
  agent1's `src/` diff vs main is exactly the 5 files I enumerated (4 HIP-only additions +
  `src/CMakeLists.txt` dispatch-only), so it complies with the refined rule; "designed-not-
  demonstrated" adopted for the whitelist in its reporting. Its build claim corroborated
  independently: `build-hip/src/libninfer_hip_host.a` exists.
- **FINDING 1 (good news, now DEMONSTRATED not asserted): D3's bf16 boundary is compiler-enforced.**
  `hipcc -O3 --offload-arch=gfx900` on `__hip_bfloat162 v; v = __hadd(v, v);` →
  **`error: no matching function for call to '__hadd'`**, and the type itself resolves, so it is not
  a parse artifact. gfx900 has no bf16 ALU, so "kernel numerics go fp16/fp32" is not a rule someone
  must remember — the toolchain refuses. **Handed to gemini as a PG-1 compile-probe cell**: assert a
  bf16-arithmetic TU FAILS to compile, and go RED if it ever succeeds. A falsifier with teeth that
  needs no device.
- **FINDING 2 (defect, precise): `src/common/hip_shim/cuda_bf16.h`'s comment is false about this
  box.** It claims `hip/hip_bf16.h` provides `__nv_bfloat16` / `__nv_bfloat162` compatibility names.
  It does not: `grep -c "__nv_bfloat16" /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h` → **0**, and
  those names exist only under `hip/nvidia_detail/` (the CUDA-emulation path). So the shim is a bare
  include providing **no CUDA-named bf16 type**; the HIP build passes only because no whitelisted
  file uses the type yet (`src/core/dtype.cpp`'s single bf16 reference is the `DType::BF16` **enum**).
  The moment a KV/graph file touches `__nv_bfloat16` it fails — loudly, which is the right failure
  mode — but the comment promised otherwise, and that costs an afternoon debugging a document.
  Options given: correct the comment, or make the shim deliver the contract deliberately
  (`typedef __hip_bfloat16 __nv_bfloat16;`) so storage works while arithmetic still fails loudly.
  Coordinator preference stated: an explicit alias is a decision; a bare include plus a confident
  comment is a coincidence.
- **TWO FALSE STARTS OF MINE, recorded rather than quietly dropped** (both preceded the finding that
  survived): (1) I reported `src/multi_gpu/tp_group.cpp` as missing from the whitelist — my path was
  wrong, it is `src/core/multi_gpu/tp_group.cpp` and the whitelist lists it correctly; (2) I
  concluded `__nv_bfloat16` was undefined because I grepped `hip/hip_bf16.h`, which is a 36-line
  platform **dispatcher** — the content is in `amd_detail/`. Method point, and it is the same one the
  ledger already carries: **no finding from a first grep.**
- **Relay #23 sent to gemini** (ownership ruling + compile-probe falsifier + the shim caveat + the
  standing guards). `/reload` remains a USER action for agent1, agent2 and this session — until then
  agent1 cannot message gemini directly and I keep carrying provenance that is not mine.
- **Open:** agent1's shim decision (a)/(b) in its next commit + compile-only whitelist extension →
  PG-A device-smoke grant on request · gemini's PG-1 with both corrections, the non-zero-count guard,
  the bf16 compile-probe, and the anti-resurrection fire-test · **user: /reload ×3, PG-0c pinned-clock
  decision (needs root)** · artifact-download grant later.
### STATE 2026-09-12 ~02:0xZ (AMD host) — Step 1 landed; gemini's agreement ACCEPTED after 2 defects caught against the tree; MY OWN polling failure recorded
- **G-AMD-2b closed + Step 1 verified cold:** 74a89677 / ac75b742 / 144cb6b8 / def8410d real, tree
  clean, `src/HipSources.cmake` + 3 shim headers present, release row re-checked by me ("No KFD
  PIDs", 4× 18,575,360 B). I checked the thing that would have bitten later:
  **`/opt/rocm/include/rccl/rccl.h` exists**, so the `nccl.h → rccl/rccl.h` remap points at a real
  header. dev3 now measured at parity (181.17 GB/s sustained, mclk top level 945 MHz, 159/159
  samples) — **all four devices share one in-load operating point**; the earlier "unmeasured" mark
  stays as history, deliberately not cleaned up. agent1's rc=127 first attempt is recorded as
  measurement history, not buried.
- **MY PROCESS FAILURE:** gemini delivered its Step 0 interface agreement THREE times (#16 01:23Z,
  #18 01:36Z, #20 01:50Z) and I did not read them for ~30 min because I only polled the hub at
  moments when I was sending. With no `/reload` in this session there is no inbound watcher, so
  **poll-on-send is not polling** — until the reload lands I must poll the hub on a schedule, not
  opportunistically. Recorded so the next coordinator does not repeat it.
- **AGREEMENT ACCEPTED (#20 supersedes #16/#18 — #18's "operating clocks unrecorded" citation is
  stale post-PG-0b), with two defects found by checking its text against the landed tree:**
  1. **PG-1's "src/ has 0 diff" is FALSE as written** — `git diff main --stat -- src/` on the branch
     shows **124 insertions across 5 files** (HipSources.cmake, 3 shim headers, src/CMakeLists.txt).
     As specified it fails legitimate work; loosened until it passes, it becomes the vacuous-gate
     class again. Ruled precise form: (a) no modification to PRE-EXISTING CUDA-side src/ files;
     (b) HIP-only additions enumerated exactly; (c) `src/CMakeLists.txt` = SHARED BUILD FILE under
     the dispatch-only rule. agent1's work is compliant — the rule just needed granularity.
  2. **`ctest -L host_unit` is VACUOUS TODAY** — grepped the tree: `host_unit`,
     `v340l_device_smoke`, `pg_layer_forward`, `v340l_op_parity` do not exist; real HIP targets are
     only `ninfer_hip_host` + `bw_probe`. Naming tests you will create is legitimate, but
     `ctest -L <label>` exits 0 on an empty selection. **RULE: every label/name-based cell must
     assert a non-zero test count (`ctest -N -L <label>` ≥1) or FAIL loudly — empty pass is a red.**
     Same guard on the anti-resurrection cell: proven to fire on a deliberately reverted hunk.
- **Whitelist status stated honestly:** "new CUDA-only files fail the HIP build loudly" is
  currently **designed, not demonstrated** — only PG-1's negative test can establish it, so no report
  row may treat it as proven before that lands.
- **Relay duty continues:** `/reload` is a USER-side pi command — neither agent1 nor agent2 can run
  it, so both are still executing the pre-fix extension and cannot reach gemini directly. Asked the
  user explicitly. Also offered agent1 first claim on the disputed target names, since the contract is
  cheaper to fix now than after two lanes build against different assumptions.
- **Open:** gemini's PG-1 with both corrections + its negative test · agent1's whitelist extension
  (compile-only, grant-free) then PG-A device-smoke grant · **user: /reload for agent1 + agent2 +
  coordinator, and the PG-0c pinned-clock decision (needs root)** · artifact-download grant later.
### STATE 2026-09-12 ~01:5xZ (AMD host) — G-AMD-2 closed; PG-0b RESOLVES my roofline objection IN AGENT1'S FAVOR; denominator rule issued; G-AMD-2b for the dev3 gap
- **G-AMD-2 closed, verified not accepted:** 01effa65 + 5200ca4d + merge 22d9e387 real, tree clean,
  PG-0b raw logs present, release row re-checked by me (no KFD PIDs, 4 devices at 18,575,360 B).
- **My PG-0 objection is now CLOSED, and the record should say so plainly: the anchor was taken at
  max memory clock.** In-load sampling: mclk held TOP DPM level 3 = **945 MHz** for 151/151, 151/151,
  152/152 load samples (dev0/1/2); I corroborated the ceiling independently from `pp_dpm_mclk`
  (levels 0–3, max 945). Sustained ~32 s copy: **183.34 / 183.26 / 182.58 / 181.09 GB/s**, 1–4% under
  burst = real droop, exactly what the DPM tables predicted.
- **DENOMINATOR RULE (binding, all lanes incl. gemini's gates):** every "% of roofline" claim must
  NAME its denominator — two legitimate ones, 2.6× apart: **theoretical per-die 483.8 GB/s**
  (945 MHz × 2048-bit) vs **measured D2D copy ceiling ~183 GB/s sustained** (≈37.8% of theoretical —
  this silicon's real copy efficiency, not a defect). Thresholds cite PG-0b sustained; PG-0 burst is a
  record only. An unqualified "percent of roofline" is a report defect and gets rejected.
- **dev3 gap handled correctly and granted a closer:** BW measured, operating point marked
  **UNMEASURED** rather than inferred from sibling symmetry — the standard, in writing. **G-AMD-2b
  issued** (5 min, dev3 only, sampler pointed at card4, append a gap-closure row, existing rows and
  the unmeasured mark stay as the record).
- **Toolchain verified properly:** I recomputed the tarball hash instead of trusting its `.sha256`
  (`f747d9b23e1a252a8beafb4e…` matches) and confirmed no `~/.local/bin/cmake|ctest` shadow exists.
  **I also found the thing agent1 did not publish: `/home/chris/opt/cmake/bin/ctest` exists and
  reports 3.30.5** — the tarball ships ctest, which is what gemini's gates actually need. Relayed.
- **Device→card mapping is NOT identity** (dev0→card1, dev1→card3, dev2→card0, dev3→card4;
  **card2 = nvidia display**). I corroborated the *shape* (4 amdgpu cards at 0/1/3/4, card2 nvidia) but
  could not re-derive the exact pairing without a GPU run — so it stands as agent1's first-hand
  measurement, cited as such. Per-device sysfs tooling must look up, never assume.
- **Refined shared-build-file rule exercised for real:** agent1 moved `find_package(CUDAToolkit
  REQUIRED)` inside the cuda conditional (3 lines, 5200ca4d) because hip configure otherwise FATALs
  hunting nvcc. Compliant — cuda configuration unchanged in effect. PG-1 stays a static-property test;
  configure-time equivalence remains unprovable here and rides the merge package as an OPEN item.
- **COOPERATION DEFECT WAS MINE, NOT AGENT1'S:** its two `comm_send` attempts to gemini failed with
  `Agent not found: pi-agent-<pid>` — the *old* extension code, because its session (01a09303, started
  00:27Z) predates the fix landing on disk. It was right to stop retrying and ask rather than assume
  gemini was ignoring it. `/reload` ordered; I relayed its interface notes (hub **#19**) and told it to
  re-send from its own lane after reload so the record shows provenance. **Standing consequence: an
  extension fix helps only sessions started or reloaded after it — when I repair a comms path I must
  tell every live lane to /reload, or they silently keep the broken copy.**
- **Open:** agent1's G-AMD-2b row (or lapse) + Step 1 whitelist/shim/PG-1 static cells ·
  **gemini's WO Step 0 interface agreement, still not delivered** (target names, PG labels, stage list)
  · PG-0c pinned-clock peak = USER (no passwordless sudo) · artifact-download grant later.
### STATE 2026-09-12 ~01:3xZ (AMD host) — G-AMD-1 closed clean; PG-0 ACCEPTED BUT RECLASSIFIED (idle-clock sampling); G-AMD-2 issued; two of MY OWN rules corrected
- **G-AMD-1 closed, verified not accepted:** 358c7812 + merge b622a032 real, tree clean (`-uall`),
  results + raw log present, and I re-checked the release row myself — `--showpids` empty, all 4
  devices back to exactly 18,575,360 B.
- **PG-0 RECLASSIFIED — the substantive catch.** agent1's numbers are sound, but its clock record is
  an **idle-state sample around the run, not during it**: sysfs `pp_dpm_mclk` sits at level 0 =
  **167 MHz** (max 945) and `pp_dpm_sclk` level 0 = **300 MHz** (max 1500). So the operating point of
  the 191.55/185.06/187.12/183.80 GB/s copy is UNRECORDED, and 187 GB/s = **38.7% of the 483.8 GB/s
  theoretical per-die figure** (945 MHz × 2048-bit) — which resolves the very docs/01+04 mismatch
  agent1 flagged rather than explaining it away. Also honest on direction: dev0 repeats go 191.31 →
  188.80 → 185.84, a ~2.9% **decline** (DPM/thermal droop), not a ramp. **Ruling:** PG-0 is an anchor
  at default unpinned clocks, NOT a roofline, and is barred as a "% of roofline" denominator until
  PG-0b — otherwise every later perf judgement on the lane is anchored to an unknown clock state.
- **GRANT G-AMD-2 issued** (15 min, one device at a time, ~30 s sustained copy, **concurrent read-only
  sysfs sampling at ≤200 ms** of `pp_dpm_mclk`/`pp_dpm_sclk`/`gpu_busy_percent`, no root, no server,
  PG-0b section appended with PG-0 text preserved-and-relabeled, release row, abort on foreign KFD
  PID). **PG-0c (pinned-clock true peak) needs the USER, not me: no passwordless sudo on this box.**
- **ENV, verified independently (agent1's claim holds):** this host has **no cmake and no ctest at
  all** — and **no passwordless sudo**. Step 1's first blocker is a real toolchain. **pip cmake is
  FORBIDDEN on this box**, from our own history: the D5 run-2 false red was a pip-installed
  `~/.local/bin/ctest` wrapper (`ModuleNotFoundError: cmake`) shadowing the real binary. Upstream
  tarball, pinned, sha256 recorded, PATH-shadow-proof, absolute path published to docs/amd/README.md + gemini.
- **MY OWN TWO RULES CORRECTED** (WO-01 at d6173bb7): (1) the `/usr/bin/ctest` absolute-path rule I
  transplanted from the NVIDIA line is **void here** — same error class as the stale worktree pointers
  AGENTS.md warns about, and a reminder that a rule copied between hosts without checking is a
  fabrication waiting to happen; (2) **"CUDA-side files zero diff" was unworkable as written** — the
  root `CMakeLists.txt` is shared and cannot gain a backend switch without being edited. Refined to:
  shared build files edited only inside `NINFER_BACKEND=hip` conditionals with the cuda path
  textually preserved (agent1's edit satisfies it — I read the diff; `120a` gate and
  `project(... CUDA)` preserved in effect), `src/` stays zero-diff, and since there is no cmake here
  **configure-time equivalence is unprovable on this box** → PG-1 asserts the static property only
  and real configure/build equivalence rides the merge package as an explicit OPEN item. Wording
  ordered: never "cuda branch byte-identical" about the *file* — a reviewer grepping the diff sees a
  change and is right to call it a lie.
- **gemini notified** of all three corrections (hub **#17**) and told its Step 0 interface row is still
  owed before it writes a gate assuming a target exists.

### STATE 2026-09-12 ~01:2xZ (AMD host) — GRANT G-AMD-1 ISSUED (first GPU use on the AMD line); agent1 lane landed and cold-verified
- **GRANT G-AMD-1** (written, scoped, time-boxed 40 min, agent1-only, no precedent): build + run
  `tools/v340l/bw_probe.hip`, 1 GiB D2D, 4 runs **sequentially one device at a time** via
  `HIP_VISIBLE_DEVICES`, clocks before/after → `results/v340l/step0_roofline.md` (PG-0). Conditions
  imposed: probe target added ONLY under `NINFER_BACKEND=hip` with **zero CUDA-side diff** (I cold-run
  `git diff main --stat` before accepting PG-0); NO server / engine build / artifact download (the
  19.03 GiB artifact is a separate request with a fresh `df`); own-PID kills only; ABORT if a foreign
  KFD PID appears or a run exceeds ~120 s; measured numbers only; closes with a pasted **release row**
  (`--showpids` empty + 4 devices back to ~18.5 MB).
- Preconditions I verified cold at issue, not from the request: `rocm-smi --showpids` = **"No KFD PIDs
  currently running"**, all 4 devices 18,575,360 B used, no ninfer-serve/hipcc/bw_probe procs, 58 G
  free, `hipcc --version` = **6.2.41133-dd7f95766** (matches agent1's §2 claim exactly), no nvcc.
- **agent1's three owed items delivered AND verified in the diff, not the message**: `95c442d4` on
  `wo/v340l-hip` (549 insertions / 3 files, tree clean via `-uall`), fast-forward history
  34b209d6 → ea9595bd → 95c442d4 as claimed, and §2/§5/§6/§7/§8 genuinely carry the rulings (ONE BOX
  dual roles, 7.98 GiB/device, source-diff/whitelist-only CUDA guarantee, whitelist negative cell at
  Step 1, Step 7 marked gemini-owned, four open questions replaced by resolved answers). It also
  re-measured the board facts first-hand before recording them — correct instinct, keep it.
- **Lane boundary held end to end:** gemini ACCEPTED the AMD test lane in writing (hub **#14**, all
  five conditions acknowledged incl. "no fake CUDA PASS on this host"), so I issued it a docs/99-shaped
  work order rather than a chat request: **`docs/amd/v340l/01_amd_phase_gate_test_lane_work_order.md`**
  (ea9595bd). agent1 keeps the PG-0…PG-F specs as design input; gate implementation + `run_ci_amd.sh`
  wiring is gemini's. Neither may absorb the other's side, and A2 has been told in writing that the
  test ladder is not staffable by it. gemini pointed at `wo/v340l-hip` @ 95c442d4 for the specs.
- **AMD line registry now:** `docs/amd/README.md` global pointer · `docs/amd/v340l/00` (agent1 product WO) ·
  `docs/amd/v340l/01` (gemini test WO) · `docs/amd/v340l/PROGRESS.md` gate log · `docs/59` row 9 IN PROGRESS,
  queue row A6 = gemini's AMD test lane.
- **Open:** PG-0 datum from G-AMD-1 (agent1 owes commit SHA + release row) · gemini Step 0 interface
  agreement (target names / ctest labels / stage list back to agent1 and me before it writes a gate
  that assumes a target) · the artifact-download grant · agent1's §2 note that its branch must merge
  main again once my next STATE lands.

### STATE 2026-09-12 ~00:5xZ (AMD host, coordinator 01a092fc) — AMD LINE OPENS; agent-comm route REPAIRED and certified both ways
- **THIS HOST = the v340l target** (measured, not inferred): `rocm-smi` → 4× Vega10 [Radeon Pro
  V340/MI25x2], kfd nodes 1–4 `gfx_target_version 90000` = **gfx900**, **7.98 GiB/device**
  (8,573,157,376 B), 2 cards × 2 dies (user-confirmed), ROCm **6.2.0-66**; `nvidia-smi` → GTX
  1050 Ti display only. Repo dir name `dual_5060_ti_ninfer` is inherited and is NOT the arbiter.
- **agent-comm hub route was dead and is now working.** Root causes (all three, recorded so nobody
  re-derives): (1) pi's `~/.pi/agent/extensions/agent-comm.ts` defined `register()` but never called
  it, and **the hub has NO REST registration route** — `POST /api/agents` 404s although
  `agent-comm/docs/SETUP.md` documents it; agents come online only via MCP `comm_register` or by
  importing the hub's `dist/context.js` (the bridge trick). (2) `comm_send` must present the sender
  **id**, not its name. (3) `comm_inbox` filtered `?to=<name>` while `to_agent` stores ids, and REST
  `/api/messages` cannot see channel traffic at all — hub `inbox()` semantics required. Also found
  live: the cleanup service **prunes offline rows**, and an offline row resolves by name but cannot
  heartbeat (UPDATE guarded `status != 'offline'`) → sends would go out as a dead agent; now
  reactivated on detection. Inbound mail **auto-injects** (15 s watcher, `followUp`+`triggerTurn`,
  consume-on-inject, `[[AGENT-COMM]]` markers in payloads defused, trust-framed as unverified).
- **Handshake certified with independent evidence, not gemini's word**: #7 coordinator→Gemini landed
  in the bridge's own `~/.agent-comm/inbox.log` ~2 s later; reply **#8** read from the hub DB; closed
  #11/#12. Gemini identity on this host = **`Gemini` / b4791a54** (roster §9 corrected; the old
  `gemini`/256adaf6 + "broker DOWN since 09-10" note in the cold-start handoff are both stale —
  broker is UP, v1.3.11). Test artifacts (probe agents/messages) purged after use.
- **Coordinator doc split (fa2543c5)**: `COORDINATOR.md` → `docs/amd/COORDINATOR.md` (rules + current
  position, 85 KB ≈ 21k tok) + `docs/amd/COORDINATOR_ARCHIVE.md` (superseded positions + 09-04→09-11
  ledger, ARCHIVED not deleted — it is the provenance the rules cite). AGENTS.md gained the
  hardware-selection probe (`rocm-smi --showproductname | grep -c "Vega 10 \[Radeon Pro V340"`;
  >0 = AMD, ==0 = NVIDIA, inconclusive = ask user), tested 3 ways: 4 here / 0 on NVIDIA-only output
  / 0 with rocm-smi absent. Pointer hygiene: `docs/59` §ref repointed, `docs/132` was a DANGLING
  absolute symlink → relative `../docs/amd/COORDINATOR.md`. **HOST WARNING added** above the 09-11 blocks:
  they are the NVIDIA line's state — do not act on another host's boot/branch/queue from here.
- **agent1 V340L port WO reviewed and ruled** (`docs/amd/v340l/`, per docs/99 template, steps 0–8,
  PG-0…PG-F). Artifact claim verified independently: 20,437,336,576 B, sha256 0634abb0…, groupwise-int.
  RULED: (1) "2 GPUs then 4" = **devices** — the cards reading demands M4 = 8 devices/4 cards this box
  lacks; M2 = 2 devices (1 card, fixture: 19.03 GiB > 15.97 GiB), M4 = 4 devices (2 cards, first real
  serving). (2) **gfx900** → `CMAKE_HIP_ARCHITECTURES=gfx900`. (3) pin **ROCm 6.2.0-66**. (4) **no CUDA
  host in this checkout** → its "CUDA path builds byte-identically" gate is **source-diff/whitelist
  only here**; real CUDA proof comes from the NVIDIA host or rides the merge package as an explicit OPEN
  item — a cell printing CUDA-green on this box is a false gate (same class as D5's silent section and
  gemini's vacuous cell). M4 slack pre-registered: 19.03/4 ≈ 4.76 GiB/device weights leaves ~3.2
  GiB/device for KV+activations — that slack, not the weights, is the binding constraint; VRAM LAW binds
  the HIP tree identically (`hipMemGetInfo` + actual bytes, zero estimate constants).
- **NUMBERING**: folder-local `docs/amd/v340l/NN_*` BLESSED; **docs/amd/README.md claimed by coordinator** as the
  AMD line's single global pointer (167/168/171/173 already taken on branches). `docs/59` row 9 moved
  HORIZON → IN PROGRESS and queue row **A6** added.
- **LANE-BOUNDARY CORRECTION issued (the substantive one, §7.x)**: agent1's WO had product-side authoring
  of phase-gate implementation + CI wiring. Gemini owns that lane **exclusively** and I may not reassign it
  to A1/A2 — so specs stay agent1's (design input: what each gate proves, call sites, whitelist semantics),
  implementation+wiring offered to gemini (**brief sent, hub msg #13, ACCEPT/DECLINE requested in writing**;
  if it declines or stalls → user + WAIT, never A1/A2). Two cautions sent with the brief: a gemini
  "verified" is cold-read and re-run before believed, and no CUDA-green may be claimed on this host.
- **PROTOCOL FLAG (open, agent1 owes)**: `docs/amd/v340l/` was **uncommitted in the shared main checkout** and
  agent1 was working on **main**, not a lane branch — its own §2 mandates worktree+branch; ordered to land
  `wo/v340l-hip` (a 24 KB plan existing only in a working tree is not landed, and another lane's `git add -A`
  is exactly how the 09-03 false-PASS trap was born). Ack owed on 3 items: branch landing, docs/amd/README.md pointer,
  §2 wording + per-device 8 GB corrections.
- **GPU: NO grant issued.** Cards idle (18.6 MB/device, 0 apps), disk 58 G free. Step 0's hardware half is
  complete by this entry; its roofline half needs a written grant on request. pi↔pi mesh = intercom (agent1
  reached me twice via the broker — harmless now that inbound works, but the routing rule stands).

### STATE 20:3x EDT (09-11) — *** FIRE-3 TAIL: ERA QUESTION RESOLVED CLEANEST-WAY *** Twin re-attested 5/5 ORACLE PASS (acc .400x4, walls = verification-boot ladder); leg= arbiter: 126/126 P6 rows leg=0/creq=0/sstop=0 — cancellation NEVER observed pre-exit, misfire class dead on this BIN (fix moved exit upstream of the defect class — token-level exit first, era row ratified verbatim). Guard 15/15 PASS ~1450s inside 1800 (coord cap-watch anchored wrong boundary, corrected by A1 store-check — lesson: cite which artifact you grepped, oracle-log vs serve-bundle). I8 flipped gate PROVEN GREEN 8/8 through correct polarity (first time ever); BF16 running, KVARN_PRE + MTP T1/T2/T3 ahead. Only reds: named era-debt ctest pair 3/3 deterministic. Green JSON → ship-steps same session → end-report.

### STATE 19:0x EDT (09-11) — A4 RUNWAY COMPLETE (edit phase): 3b landed 05dad481 — d2 restore requirement EMPTY by proof (drafter pool never written in prefill per :7013's own note + chain_lo derived live above plen ⇒ cross-request drafter state contributes nothing by construction; :5001 flip admits a fact not a mechanism; R2-d2/R3-d2 gates retired with receipts; prefill-catch-up = acceptance OPPORTUNITY scoped OUT of 173, noted for feature-matrix lane). docs/173 code steps 2a/2c/2d/2e/3a/3b ALL landed+compiling, Cell-1 green, gate specs pre-registered (G1-G6, ±0.03 band reframed to cold-start-equivalence, decision procedure w/ one-fix-class-per-red + two-reds-kill-hypothesis). A4 holds at edit-complete. GPU AGENDA SEQUENCED for post-fire-3: A4 live gates (step-1 prefix battery + host-kv golden, step-2 matrices, 3b G-matrix) + A3 q3 cell + leg= arbiter if needed — one session block, written grants. Fire-3 mid-battery (prefill/phase gates through, decode-guard-in-1800 + mutation x3 + MTP first-attest + twin re-attest ahead). Standing: green→ship→end-report.

### STATE 18:4x EDT (09-11) — *** THIRD FIRE LIVE (18:45, a1f7ac75, CLEAN tree, guard passed, 46G) — ETA ~60min *** Run-2 died at decode-guard exit 124 = cap-calibration era debt (A1 store-grade: 14/15 ALL PASS baseline-tps, walls match run-1 ⇒ contention REFUTED, no leaked server — group-kill worked, setsid-contrast vs lint-orphan noted; run-1 needed ~1530s vs 1500 cap). Harness fix #2: CAP_DECODE_GUARD 1500→1800 w/ measured anchors + 'cap set from NOMINAL never MEASURED' root sentence + generalized-LITH lesson for farm7 (every CAP_* cites measured wall anchor) — a1f7ac75, A1 formality-GREEN. Fire-3 milestones: ctest sample #3 (rate-table completion for #32/#144), guard 15/15 inside 1800, twin re-attest, flipped mutation gates x3, MTP T1/T2/T3 FIRST ATTEST. Ledger: '--full needed two harness fixes to reach its own green (polarity x3, cap x1), both era debt, zero src, headline banked twice.' A4 meanwhile: step-2 CODE-COMPLETE e7492ab8 (restore mirrors single-seq demotion, dispatch preference outputs-unaffected, escape hatch, CELL-1 ctest GREEN early) + step-3a/3b GO'd CPU-only; its GPU slot queues behind fire-3. Standing order: green→ship-steps→end-report.

### STATE 17:3x EDT (09-11) — *** RE-RUN --full FIRED 17:33:52 ON e394fcf7 (CLEAN tree, guard passed, disk 47G) — THE ATTESTATION RUN *** Chain since first fire: twin cell 5/5 ORACLE PASS inside battery (G1 fence green, leg= distribution pullable from banked log); battery then false-aborted at i8-mutations — A1 store-grade: old-polarity fossil, family audit found THREE (I8/BF16/KVARN_PRE, one-flag fix would've died twice more), A2 flipped 9 lines (7970db65, PROD/*_EXIT untouched, selftest both-direction re-green) + global verdict-flag LAW comment (e394fcf7: '0=ok everywhere; init-1 flag = FOSSIL, reject at review', instances 10-12 named, Sept-10 slice4|1 REAL-failure-through-correct-gate distinction preserved); A1 cold-read GREEN both, chain e394fcf7>7970db65>1c17218d. This run's duties: 3 mutation batteries through correct gates, MTP T1/T2/T3 FIRST ATTEST, pp_warmup contention-free ([2/3] no-op, zero build delta), ctest #32/#144 rate sample #2, twin re-attest. ~55 min, milestones at decode-guard/dflash2/tail. Standing order: green→ship-steps→end-report; red→stop-at-coord. A4 build still gated (its paused-dir reclaim credited to disk), A3 dark, gemini standby. Exhibit sentence banked: 'the fence held while the gate around it was fossilized'.

### STATE 16:4x-17:0x EDT (09-11) — --full MID-BATTERY, ALL EARLY REDS TRIAGED (A1 store-grounded, coord-ratified): ctest 153/155 green (2 red: #32 bench_kvarn_attention threshold-flake-suspect + #144 tp2_interleaved_prefill bad_alloc-after-9059MBx2 — both post-battery rate-table 3x each). VERIFY pp_warmup FAIL 246.7-vs-94.5: A1's md5 collapse — merged BIN IS c31ae347 (src-untouched merge ⇒ byte-identical, number attributable to merged tree) BUT ran 16:32:04 UNDER [2/3] build-tax contention → contention-suspect shape; post-battery contention-free re-run adjudicates (pass=artifact row, fail=real perf claim escalated). Report bar wording: 'all cells attest c31ae347 = merged-tree build output; verify pp ran under build contention, re-check adjudicates' — no phantom pass/regression. Correctness CI cell: 0 failures (17 skips = known real_test family). Draft-vocab path gap (qwen38_draft_vocab_ids.json→lm_head fallback) → 169 WO list. W5 cleanup done (user order): 3 stale branches deleted w/ ancestry proofs + backup tag; radiance/ghost-review KEPT (unmerged content). A4: step-1 cutover 804cce5c + CUTOVER_PROOF.md 683d2000 (115 sites grep-measured, compiler-check pending build GO). Disk 44G. HEADLINE AHEAD: dflash2 twin cell verdict + leg= arbiter distribution. Standing order: green→ship-steps→end-report.

### STATE 15:5x EDT (09-11) — *** CI WINDOW OPEN, USER OVERRIDE: SHIP WITHOUT WAITING *** Written grant issued+received: ~90-min both-card window, merge-first. A2 driving NOT-FF (habitat 601a8df7 + gemini 469a8201; conflict sites run_ci/contract_lint/CMakeLists; A1 gates: rule-8 merge-commit delta check, +6 vram_fence placement, standalone lint, CMakeLists; joint region-parity pre-check with coord at merged-tip SHA; claim = gpu_guard-by-context, foreign context = stop+ping). USER STANDING ORDER: green --full ⇒ ship steps proceed WITHOUT user gate; report chain at END (verdict JSON + rows + ship-state); RED still stops at coord (judgment = flake-table/blocker/rollback; no silent re-run loops). A1 silent-until-granted→now running. A3 dark-resumable (coord-cut 13G prune credited, 4-min relink banked). A4: docs/173 spec vs main, 0 cards. gemini standby-agent_comm. Ledger candidates for 169: vacuous-cell caught-in-review cure (minutes), coord-cut-for-dead-session pre-authorization sentence, gemini dual-leg comms, A4 name-fence. Clock: started at grant.

### STATE 15:4x-15:5x EDT (09-11) — CI FREEZE COMPLETE, WINDOW PENDING A1'S WORDS: gemini vacuous-cell arc: A1 cold-read FOUND dflash2_single_seq_ci.sh vacuous-as-committed (only reachable exit-0 = dry-run else, BIN never launched) → RUN-1 FROZEN to tooling-only (coord-ratified) → gemini cured in MINUTES (ff423e3b wire → 58de22bb criteria a/b/c: else-branch deleted, every failure loud, red paths 2/1 split → 469a8201 arm-contract closing coord's false-red steer: ARM=ON+NINFER_D2_PHASEGATE=1 explicit) → A1's OWN cold-read GREEN on 469a8201: cell RUN-1, +6 run_ci insert RE-OPENS (vram_fence'd, port 8094), contract_lint honest. FREEZE PACKET: habitat 601a8df7 + gemini 469a8201, RUN-1 FULL. Residuals on record: freeze-SOP server-survival WO non-blocking (gemini's call); lint-29/29 claim merges to session gate (A1 re-runs standalone). DISK: A3 SESSION VANISHED (intercom roster, no lease — flaky-API class) with pre-authorized 13G prune undone → COORD EXECUTED (lsof clean, no processes, both-lanes-authorized): df 25G→**38G ≥ 35G bar MET**; q3 relink tax banked, A3 lossless-resumable from debriefs. Routing laws held: A4 fenced off coordinator mesh identity (ack'd), gemini dual-leg comms confirmed (agent_comm primary). NEXT: A1's words-window-request → WRITTEN grant (merge session: A2 drives NOT-FF + step-0 region parity; then ~90-min both-card --full, guard-serial). agent4 building docs/173 spec vs main semantics (0 cards). 169 G1-CLOSED fold c56c88ec in.

### STATE 15:2x EDT (09-11) — *** G1 CLOSED, CI PUSH *** verify boot PASS coord-cold-verified: 6/6 finish=stop_token rc=0, stats rounds=12 accepted=24 acc=0.400 tok/rd 3.00 (parity-best, m6-identical walls), PG rows live every request (424→2114), GPU clean, tip 491c3cee pushed. Mechanism era: cancel-path exit symmetricized (e481da74 CRIT-1 shape) + httplib is_socket_alive root patch (557df6f0, vendored, SO_ERROR). CAVEAT RULED IN: 'misfire caught red-handed' RETRACTED — tdec 0.98s = decode-completion, cb_failed leg fits 6/6 identically; era wording 'trigger leg pending 10th-int split' (A2 building + VERDICT.txt caveat rider same commit). G1 official on the caveat. USER ORDER: gemini does NOT push ci-full; NEW A1 = CI-FULL ORCHESTRATION OWNER (brief drafts/172 banked, dispatch sent); A2 = 10th-int + merge-package handoff + 169 bullets; A3 = EMERGENCY DISK RECLAIM (11G! → target >=35G: 15G artifact verify-then-move + build/tests prune); q3 lane deferred behind CI (user priority). gemini: consultant for own cells (cf1dd723 warmup-skip landed; my orchestration-send bounced 'Session not found' — they reach A1 if needed). Next: A3 df rows -> A1 orientation + merge freeze -> WRITTEN both-card 90-min CI grant.

### STATE 11:0x-11:5x EDT (09-11) — Q3 LANE: REAL ARTIFACT PRODUCED. FP8 source downloaded (29G, 66/66 verified) → convert_iq3.py streamed (v1 OOM'd RAM-accumulation, A3 fixed via stream-to-temp, 26G steady) → **artifacts/qwen3_8_27b_iq3.ninfer = 12.5G, EXIT=0, assembly complete 11:45** (final 12.5G not ~20G — BF16-passthrough math landed lower). Serve app built (APPS=ON incremental, BUILD_RC=0). A3 window continues: per-tensor QC (quant-quality vs fp32, metric lanes separate) then SERVE ATTEMPT card 1 (mmap=page-cache, safe at 17G). HF→recipe NAME-TRANSLATION layer added (hybrid arch Qwen3_5ForConditionalGeneration vs llama recipe names) as tested unit. DISK EVENTS: coord issued emergency re-prune order on STALE PREMISES — both wrong, self-corrected in-seq (size estimate; and habitat build/tests already pruned at boot-8 close = 1.3M now, so the 4-min relink tax is simply owed by whoever runs farm7 step-0 — coord-order artifact filed). POSTURE: df 17G/93% — morning FLAG: user-consent delete of 29G re-downloadable FP8 source (purpose served) = 93%→~80% recovery. Watchdog v2 (activity-aware lease branch) live; A3 heartbeat holding; earlier API-drop scare resolved (two dropped pings, all state resumable, replied seq-1). dflash CONVICE STILL USER-PARKED (see 10:4x line): no boots, A2 dark, A1 resting.

### STATE 10:4x-10:5x EDT — *** BOOT-8 COMPLETE: CLASS (A) RETIRED WITH EVIDENCE ⇒ (B) SURVIVES ⇒ USER CONVENE OPEN (A2 close-out c867b7c0, RELEASE ROW filed+coord-verified: 0 apps/15 MiB) *** RATE TABLE (run-lines arbiter after coord's own 5H/8 double-count corrected by A2 — memo lesson): ARMED (poison ON, BIN 509300c7a654) = 4H/8 (b9_poison 1H/1P + b9c_cont 3H/3P, arm-proof ==4 EVERY run) vs OFF-control = 1H/3 (run3 hung with the SAME step-13 zero-variance signature — coin flips both ways, strengthens control) vs historical 3/11. Fisher p~.6 ⇒ INDISTINGUISHABLE ⇒ uninit-envelope theory's discriminating test fired and DIDN'T HOLD. 'Poison raises rate' filed HYPOTHESIS-GRADE ONLY (p~.42) per A2 — not entering the pack as stronger. ~1000 fresh collective observations across six hangs: one-shot family UNANIMOUSLY EXONERATED. SURVIVING: (B) per-device nondeterminism in the accept kernel's independently-computed output (a, lic) — the expensive/ugly class; convene options for user: (i) accept-kernel determinism static (CPU, bounded, free), (ii) FIRE-WITH-LABEL → G1 ships labeled 'completes at parity 0.369, ~40% deterministic step-13 freeze at k=5/N=1, batched clean, twin Option-B-gated' → farm merge + run-ci.sh --full proceed NOW, (iii) D2SS-TAIL byte-position cycle (~1 boot) to see WHERE devices diverge. COORD LEAN: (ii)-now + (i)-beside + (iii)-if-(i)-weak. My spec-blot admitted mid-sweep (halt-at-first-hang copied into a RATE probe — hangs ARE the datum there; A2 ran CONT+OFF correctly after amendment). CLAIM RULE all-lanes: explicit RELEASE ROW or coord re-confirm only (A3 raced an in-flight teardown, held cleanly, no VRAM collision — honest note banked).

### STATE 10:0x EDT — *** BOOT-8 STAMPED — b9_poison N=8 FIRING *** A1 gate GREEN deep-read on 1abfc7ca (coverage :3556/:3559 both targets, zero-cost-unset :3555, arena-birth placement). Launch = A2's runbook as binding text: proof-of-arm 2× D2SS-POISON/run hard precondition (unarmed-run defense generalized from b9/b9b lesson), halt-at-first-hang, BIN 509300c7a654, OFF-control reading resolved poison-OFF/trace-ON. Trichotomy on rig header: 0/8 (p≈.12) ⇒ uninit-class CONFIRMED → D2SS-TAIL names byte → fix-9; ~27% ⇒ (B) nondeterminism + user convene; mid ⇒ OFF-control ×3 same saved BIN no rebuild. Ledger adds: double-apply self-catch, 236MB GitHub amend (DISK-ONLY convention + pre-prune saved-copy step added), MEASURED 4-min cold-relink footnote. A3: re-pinged harder (one-line ask); still unreported — relaunch-lossless confirmed via commits. GPU: A2 at claim. Farm chain: 169 fill prepped, merge, run-ci --full behind this verdict.

### STATE 09:1x-09:2x EDT — (iii) COMPARE DELIVERED (A1, 3 lines) — CONVERGENCE w/ DELTA: A2 table complete for accept INPUTS (all replicated :3866; pool-surface correctly SUBSUMED), but missed st.work (:3871 per-request) + mb_ext ENVELOPE shape (:3706 alloc {d2_N}, :3856 writes ext_now, kernel reads array — beyond-envelope columns = exactly where a>0 first-touches garbage). Coord cite-verified. INSTRUMENT PICK (both, unanimous): 0x5A-POISON first-cut (memset mb_ext+st.work envelope at request start, env NINFER_D2_POISON, once not per-round, zero-cost unset), D2SS-TAIL second if positive. BOOT-8 ORDERED to A2 (woken, bounded scope): implement→build→fire-check→A1 word→stamp→battery b9_poison N=8 halt-at-first. VERDICT TRICHOTOMY PRE-REGISTERED (hang-RATE only, explicitly not correctness — different garbage = fine positive): 0/8 (p≈.12 unmoved) ⇒ uninit-read CLASS CONFIRMED→TAIL names byte-position→fix; ~27% again ⇒ (B) nondeterminism inherits + user convene; mid-range ⇒ OFF-control ×3 before reading. DISK FINAL: 42G (prune executed, G7 CLEARED +17 margin; keepers+saved-BIN-permanent row in 170). A2 dark after hunk; A1 clear; A3 queued; GPU 0.

### STATE 09:0x EDT — NIGHT CLOSE-OUT FILED (A2, lane dark): cards baseline-verified (pgrep empty, 15 MiB/0%/0 apps), era chain complete (last booted a804d598e817), FINAL TALLY 8P/3H (~27% hang rate). Door ledger FINAL: fall-through FIXED / counter FIXED / fuse-feed FIXED@PARITY / C4-reset hygiene / (iv) BURIED both routes armed / (ii) demoted / (i) weakened / (iii) RESIDUAL = per-device accept-output divergence (A) uninit-envelope (alloc-history fits 4/4 script-correlation) vs (B) nondeterminism — A1 COMPARE is the sole live item (verdict→instrument choice→boot-8). 5 self-catches all in-band. NEXT: A1 verdict → stamp boot-8 (poison-probe or TAIL) → fix-verify (boot-9?) or pivot-to-label. Farm chain ready-behind: G7 fresh row 21G, prune-decision at package, 169 fill, merge, run-ci --full. USER: option A executing to completion; no input needed until boot-8 shape decision or morning pivot.

### STATE 08:5x EDT — (iii) STATIC IN FROM A2 (d1e91e44): tail-input table — everything feeding lic/hit_stop/done is REPLICATED or COMPLEMENTARY-consistent EXCEPT the accept kernel's own OUTPUT (a, lic) computed independently per-GPU from identical bytes ⇒ residual class = PER-DEVICE divergence; alloc-history/uninit-envelope fits the 4/4 script-correlation that timing couldn't explain. Instruments proposed (poison-probe 0x5A cheap-first vs D2SS-TAIL byte-position) = BOOT-8 SHAPES, HELD UNBUILT pending A1 compare (ask sent: table completeness incl. direct pool reads in accept path; envelope-vs-nondeterminism discrimination via drafts-predate-accept argument; instrument choice). A2 self-catch #5 (tool-FIXED claimed while tool crashed → genuinely fixed + verified on both decisive logs: b9b 87 KAR/0 REJ, b8 58 KAR/0 REJ — filings STAND) + pre-commit correction (a>0 from round ~1, hang = 12 rounds INTO a>0-live ⇒ strengthens first-a>0-dependent-window shape). Ledger: 5 same-pass self-catches tonight. Cards→baseline. Farm chain: G7 re-measure ordered into 170 board-state.

### STATE 08:3x-08:4x EDT### STATE 08:3x-08:4x EDT — *** DOOR (iv) BURIED ON BOTH ROUTES (armed) ***: b8_hunt run1 = decisive control-hang (KAR armed, 93 lines, ZERO REJECT at hang) ⇒ closed on twin; b9_cellb = batched passes exact test (user's question: req1 gen31 + req2 gen24, both stop, rc0,0) BUT instrument env didn't reach cell-B (A1TRACE=0 — caught by coord grep before filing); b9b_cellb_armed = 430 lines live, ZERO REJECT, both complete ⇒ AR window never opens on batched either. One-shot family EXONERATED as a class for this wall (S1+KAR-v1+v2 all retire cleared-with-positive-control; b9/b9b unarmed-vs-armed pair = self-catch exhibit). A1 ACTIVATED (user order) — (iii) PRIME analyst (d2_log per-rank CONTENT divergence pre-argmax below byte-identical drafts; enumerate shard weights/KV slice/linear-state/tap-feed, which diverge BY ROUTE), A2 cross-check static in parallel, compare-before-concluding. Doors ledger: (i) weakened (constant counts), (ii) demoted (pure NCCL, consequence-not-cause), (iv) CLOSED armed-both-routes, (iii) LIVE PRIME, non-collective surfaces inherit. Tally 7P/2H≈22%. Cards: A2 releasing post close-out; boot-8 = fix-or-instrument, fresh stamp. Farm chain behind (iii).

### STATE 08:0x-08:1x EDT — BOOT-7 PART 1: N=1 PASS AT PARITY (3rd identical 24/.369/2.85 completion) + INSTRUMENT PROVEN LIVE (97 KAR lines, both-rank ACCEPT sanity rows, REJECT=0) ⇒ POSITIVE CONTROL filed, NOT a verdict — door (iv) untested-for-hang (post-instrument tally 7P/2H ≈22%). A1 gate went GREEN-with-caveat (event captured PRE-wait at :93; wait = safety net; product-shape rides boot-8). DOOR-(ii) STATIC CLOSED (A2, accepted): allgather_local_bf16 pure NCCL (:278-282, no one-shot path), twin calls constant-count/size/straight-line, SYNC parity through last round ⇒ GPU0-spin is CONSEQUENCE of rank-1 exit not cause — (ii) structurally demoted w/ cites. USER ASK ('does this test pass on multibatch?'): YES historically (dark-by-perf era = same m2b bodies, weeks of CI, never step-13 hangs; batched uses same AR family HEAVIER, 2 lanes) but KAR-instrumented batched run never fired ⇒ GO'd sequence: (1) twin N=5 halt-at-first-hang (decisive: REJECT-at-step = (iv) NAMED / zero-lines hang = (iv) CLOSED), (2) cell-B one run = user's question AS cross-route race audit (REJECT-but-completes ⇒ family window rotation-benign on batched ⇒ twin-specific delta narrows (i); zero-lines ⇒ (iv) closes both routes). 5/5-clean branch: (iii) static by A2 + coord decides sweep-vs-convene at new tally. Cards: A2. 170 pack = living doc, door-(ii) row folded.

### STATE 08:1x EDT — USER DECISION: OPTION A — BOOT-7 AUTHORIZED. A1 woken for final KAR-v2 re-gate (fence-chain / predicate one-grep / PURITY: reject-path print-only, no behavior change pre-boot-8). A2 pre-stamped conditional-GREEN: battery.sh b7_karv2 N=1, BIN a804d598e817 promoted-to-boot (era supersede row), trichotomy card row, bt-armed, halt-on-first-verdict. Datum = grep -c REJECT-candidate + step map (~96 AR-calls/round ⇒ step13≈call1250). LIVE → boot-8 fix delta (reject-stale+re-poll as product) w/ A1 static + coord stamp. BENIGN/silent → doors (i)/(ii)/(iii) bounded static, no freeform. Farm chain (B elements: G7, prune, 169, merge, run-ci --full) continues queued behind verdict.

### STATE 07:4x-08:00 EDT — FULL FREEZE, ALL LANES BANKED: KAR-v2 final shape (event-gated: first-16/rank + every rejection; step-13≈call-1250 rationale; LIVE/BENIGN/instrument-silent trichotomy on card; wait-loop cannot deadlock under lockstep ⇒ either boot-7 outcome is a verdict). Tip d849de7d pushed, A1 re-gate outstanding (last signature), A2 rest-taken, A3 idle-clean w/ roadmap doc. Boot-7 zero-latency pending gate; option-B path fully evidenced. 170 = user's single read. Overnight FINAL: 5 surfaces dispositioned, 4 fixes verified, first complete twin request + first on-device q3 numbers in one night, zero discipline exceptions, zero ungranted GPU moments, ~8 era-BINs fingerprinted no-rewriting.

### STATE 07:3x-07:4x EDT — KAR-v2 STAGED, KIT COMPLETE: f2da5e8a (53+/11-, one file): payload→fence→GEN→fence→flag two-fence chain (architecturally-justified minimal ordering), mechanical predicate reader print per seq-68, STAGED BIN a804d598e817 supersedes 9f30dab7ee3b (v1 fingerprinted never-booted, no era rewriting), fire-check+unit green, push parity PROVEN (coord cold-verified). A1 re-gate ordered (fence-correctness / predicate one-grep-ability / instrument purity — reject-path must print WITHOUT altering behavior: no accidental fix-before-boot-8). A3 Phase-2 plan committed (2fd65ef2), lane idle-clean. FROZEN STATE COMPLETE: user's option-A (boot-7) is zero-latency (one command on re-gate-GREEN), option-B (fire-with-label) unblocks farm chain immediately. 170 pack final pending only A1's word row. Cards 15 MiB both, GPU: nobody. Night closed all-CPU: 5 surfaces, 4 verified, 1 staged-instrumented.

### STATE 07:1x EDT — CPU PACKAGE COMPLETE (A2 9dbf8df0), DATA-PACK 170 FILLED (user doc: drafts/170_user_datapack_twin_wall_2026-09-11.md): pairing tables post-hoc CLEAN both runs (272 K-lines, observed>epoch ZERO — argmax cleared w/ positive control), run3 frame-class UNMOVED (wall reproduced byte-identical on fixed BIN), KAR INSTRUMENT STAGED (one_shot_allreduce A1TRACE-K, BIN 9f30dab7ee3b, no boot, A1 gate queued) + STRUCTURAL FACT: family's dev_epoch upload is CHANGE-GATED (:215-217) & epoch constant-1 between 128-wraps ⇒ observed>epoch may be STEADY-STATE BENIGN — boot-7 verdict metric pre-registered LIVE-at-hang-step vs VISIBLE-BENIGN. Timeline record: battery ran AFTER A3 yield, re-grant after run6 teardown, zero overlap (coord seq-64 was stale-by-a-minute, corrected). A3: cards released to baseline, Phase-2 q3 plan doc in flight. G1 HELD, boot-7 one-stamp from live = USER MORNING DECISION (A fix-7 / B fire-with-label / recommended B-now+A-next). Frozen-board posture: no boots, no GPU, src closed. Night final tally: 5 surfaces dispositioned (4 fixed-verified, 1 instrumented+staged), first-ever complete parity-band twin request banked, zero discipline exceptions all night.

### STATE 07:0x EDT — S3 BATTERY VERDICT: 5 PASS / 1 HANG (fix+K-trace BIN d94965428744). run3 froze at step=13 WITH every K-line clean (observed==epoch both ranks at hang, epoch-13 rows present) ⇒ ARGMAX DOOR EMPIRICALLY CLEARED (matching A1's rotation math); B-C′ branch live. Rate: pre-fix 1/3 clean, post-fix 5/6 — statistically indistinguishable (p≈.36), NO attribution (honest line on the card AND in the data-pack). All completions byte-identical stats (accepted=24/0.369/2.85/gen=37 — determinism in outcomes when the coin falls right). A2's row-level parse-artifact disclosure (pairing tuple included rank field ⇒ meaningless as printed; logs preserved, post-hoc re-derivation ordered) = culture #5 of night. CPU PACKAGE IN FLIGHT: (1) real pairing table post-hoc, (2) run3 bt frame-class row, (3) ALLREDUCE-SIDE K-INSTRUMENT (hot collective: ~96 one-shot allreduces/round via tp_group.cpp:268 fast-path — A2's exposure correction caught pre-169), staged BIN, NO BOOT, A1 gate. *** BOOT AT THIS WALL FROZEN — user-eyes rule active *** morning deliverable = data-pack: P8 banked, wall reproduced-on-fixed-BIN, argmax cleared w/ positive control, doors (i) T-mismatch (ii) NCCL allgather (iii) d2_log content (iv) OneShotAllReduce handshake w/ instrument built, boot-7 one-command-ready; G1-with-label sentence pre-drafted for farm package option. A3 RE-GRANTED card-1 (q3/q2 oracle-pair + real Q3 weights, till morning). A1 resting+gating. Cards: A3 on 1, card 0 idle. G1 HELD.

### STATE 06:4x EDT — *** MECHANISM-CLASS NAMED: ONE-SHOT ARGMAX CROSS-STEP PAYLOAD READ *** (A2 static, d87acc18/a2b337da): frame-class CORRECTED rank-0 = D2H-BLOCK (cuMemcpyDtoH under gqa_kv_append verify-13, behind vanished participant) not spin; DETERMINISM REFUTED by A2's byte-proof (m6 boot completed vs ab3b froze, BYTE-IDENTICAL through step 13, same BIN/body = genuine timing race; p1bt-vs-p1ab3 = script as timing-perturbation); STRUCTURAL DEFECT: no forward-only guard, expected_epoch CONSTANT-1 for first 32 calls (kNumSlots=32), pairing-by-ordering-only, pad field unused ⇒ lagging block reads peer's NEXT-step payload through narrow window (bar = host-side arrive); W5-FIX :1629-1640 precedent names the EXACT terminal (argmax mismatch→permanent stall GPU0 100%/GPU1 0%). Ties Q1/Q2/Q3: T constant=6 ✓, call-count parity holds ✓, guard = the hole ✓. PLAN STAMPED: S1 gen-stamp fix (unused pad field as monotonic counter, poll=flag AND gen>last) CPU-NOW, NO boot; S3 battery = 6 T-cell runs on NINFER_MB_ARGMAX_TRACE (:268) names divergence live; battery goes SECOND behind A3's live q3 window (user's new GPU order), FLASH-PRIORITY at A3's artifact boundary on A2's READY ping. PRE-REG: B-A anomaly-at-step (S3 unfixed) / B-B fix×6 ZERO hangs = S1 verified = G1 advances / B-C still-hangs = full stop, user data-pack. G1 HELD (not closed) pending B-B; m6-P8 stands as banked history. A1 gate on S1 diff when real. Ledger: A2's 3rd same-pass rule-8 self-catch (git add missed dir, caught post-push) — catches-by-instruments = 169 methodology backbone.

### STATE 06:3x EDT — COORD PRE-READ SHAPES THE 7TH-SURFER: one_shot_argmax.cu ties look STRUCTURALLY EXCLUDED (local :73/:92/:127 lowest-index rule; cross-rank :163 lowest-tok deterministic) ⇒ refined candidate = PAYLOAD STALENESS in lock-free exchange (:135 publish/:154 poll/:265 comment warns stale-peer-payload read BY NAME; twin's seed-call shape (nullptr draft_vocab, T?) vs round-call shape may consume different counter sequences; intra-request cross-step read is the remaining skew generator post-C4-reset). A2 STATIC REFINED to 3 items: (1) twin slot-pairing math for exact call pattern, (2) forward-only guard existence on the poll, (3) per-round T constancy across a>0. BRANCH RULE PRE-AGREED: if kernel provably cannot diverge ⇒ lic equality FORCED ⇒ C4-as-skew FULLY dies ⇒ search space = every per-rank host-side branch reading device values at round tail — that's a DIFFERENT static and I bound it before any boot. No boot-7 without: static names it OR user's eyes. ab3b bt frame-class row due in P10 bundle. Cards idle (0 apps), A1 resting, A3 Q2 CPU-work continuing.

### STATE 06:2x EDT — BOOT-6 DOUBLE-RESULT: m6_boot1 P8 LANDED (FIRST COMPLETE TWIN REQUEST EVER: finish=stop, 200+35tok, acc=0.369 [24/13rds, parity band], 37.4 tps > plain 31.5, zero not-bound) — G1 quality+completion PROVEN. THEN cell-T-2 (ab3b, same BIN e249, C4-reset tree) HIT P10: FREEZE EXACTLY step-13, round-12 drafts BYTE-IDENTICAL across eras (248046 248046 198 8839 220) ⇒ THREE FREE CONFIRMATIONS: (a) DETERMINISTIC wall (content-function, not race/skew-random); (b) C4 REFUTED-AS-CAUSE (reset landed GREEN, wall unmoved — kept as parity-hygiene, retired from mechanism list); (c) warmup-seeding EXONERATED (identical 0.067/1.33/4 both eras; intra-request source). NEW PRIME DISPATCHED (CPU static, zero boots): allreduce_argmax COMBINE SEMANTICS — MAX-or-SUM + TIE-BREAK determinism across ranks (tie-rich content: 248046×2 in drafts; lic_h content divergence at r12 = only unsynchronized-input candidate left). ORDERS TO A2: no cell-B on frozen cell-T; bank-everything kill + frame-class row (same driver-spin? rank-1 absent again?); P10-DATA bundle; m6 P8 verdict STANDS UNTOUCHED (banked history). 7th-surface threshold armed: if tie-break static names it → fix-7 cycle; if not → CONVENING WITH USER at wake, no third boot on same wall without his eyes. Cards: A2 killing, release row due.

### STATE 06:1x EDT — C1+C2 REFUTED (clean, cites), C4 NAMED+STRUCTURALLY-CONFIRMED: twin NEVER calls reset_one_shot_step (call-sites {1893,2119,5238,5241}, twin fn 3354..4085 zero; :2303 comment documents WHY every other route has it — per-request freshness zeroes BOTH counters pre-proposal). API proof: per-rank counters one_shot+one_shot_argmax (tp_group.cpp:288-293) exist BECAUSE skew is the known failure mode; twin consumes one_shot_argmax (seed :3594 + every round), resets neither. Datum-fit 4/4: drafts-identical (chain=plain allgather, one-shot-free) / lic=blind spot (never printed) / rank-1-only clean exit / 248046×2 trigger content. FIX GREENLIGHTED (order: AB3 close-out bundle FIRST [parity headline + capture exhibits + refutation chain], THEN src: both-rank reset at twin :1893-lifecycle-equivalent pre-round-loop, watch-points in msg: (1) call-count parity + plain-one_shot non-consumption grep row (A2 pre-answered: twin group collectives = allreduce_argmax/allgather/barrier only ⇒ harmless parity), (2) warmup fires same path). build→fire-check→push→A1 word→stamp. BOOT-6 PRE-REG: P8 freeze vanishes + FIRST-EVER complete twin request (200+stop+nonzero STATS); P9 different-round freeze = skew seeded elsewhere → convene; P10 exact-13 repeat = C4 wrong → C3 re-primizes. Cards clean 15MiB/0. G1: quality SOLVED at parity, exit-path wall = last blocker.

### STATE 06:0x EDT — AB3 VERDICT: *** FIX CONFIRMED AT BATCHED PARITY (~.38-.42, request-side vpos frontier arithmetic, first-ever in twin history) *** + 6TH SURFACE = a>0 EXIT-PATH WALL: round-12 end-tail rank-1 CLEAN EXIT (absent from 34-LWP bt dump, NO RANK1-ERR, no throw — while-cond false rank-1-only), rank-0 spinning CUDA-DRIVER-SPIN in first-ever a>0-live gqa_kv_append_kvarn_and_commit (round-13 verify). SYNC parity 15/15+16/16 through round 12; warmup now completes CLEAN (P1 re-confirmed, 0 'Paged KV' strings). Kill APPROVED (bank-everything; capture complete), teardown A2's PIDs, release row required. C1 = A2's bounded static (lic_h D2H vs accept-kernel enqueue ordering — a>0 reads lic_h[1..] FIRST TIME; stop-token 248046×2 in round-12 drafts = hit_stop SHOULD fire, question is both-ranks-same-bytes): CONFIRMED→fix=sync-before-D2H mirror batched tail / REFUTED→C2 out_count walk / INCONCLUSIVE→cell-B-only boot proposal. A1 rest-gate HOLDS (scarcest non-GPU resource = his ctx; next duty = fix-diff one-word). AB3 close-out: era bundle + parity-headline verdict doc + rank-1-absent-LWP as exhibit; cell-B join-pending-not-lost. G1 status: quality SURFACE SOLVED, exit-path wall = last decode-complete blocker. Farm/run-ci hold behind it.

### STATE 06:0x EDT — AB3 BOOT LIVE, TWO DATUM AT ONCE: (1) FIX CONFIRMED: first nonzero twin acceptance EVER — 'rounds=3 accepted=1 acc_rate=0.067 tok/rd 1.33' 05:51:20 (cell = A2 to say warmup-vs-request; full-request stats row pending). (2) 6TH SURFACE TRIGGERED: FREEZE MID-DECODE at step=13 (NOT stop boundary) — serve.log static 5570 lines, server 811413 ALIVE both cards, wrapper bt_all.txt captured ON SCHEDULE 05:52:59 (90s detector + SYNC prints + capture = this wall arrives pre-instrumented). HALT ORDER TO A2: no teardown/relaunch/cellB advance until bt frame-CLASS (cond-var/NCCL/cudaSync; A1's class-only rule), last [D2SS-SYNC] r0/r1 pair sequence banked, candidate named to coord. Mechanistic note: fuse fix + now-live append/ring paths = FIRST execution ever of that code — new wall in newly-live code is predicted-fine, and it is minutes-legible not 13-minutes-mystery. Era-BIN chain extended. A3 CPU lane STRONG (user-redirected): d44c0789 quant_recipe map (12 sup/2 gaps, iq2_s sole hard format gap), latent Q5G64_F16S packer-throw catch, Q2G64_F16S 2-bit closure APPROVED, GPU still revoked. Gates: G1 = fix-confirmed + fresh wall under forensics; farm/run-ci hold.

### STATE 05:5x EDT — *** AB3 GO STAMPED — FIX-VERIFICATION BOOT FIRING *** Chain complete on every leg: A1 PRIME verdict+addendum (c7114a7a+76a0adaa) named POOL-FUSE-NEVER-CALLED mechanism; A2 fix 4045bfd4 (fuse mirror :3951-3954 pre-append, batched-exact; shape/envelope/provenance rows in msg; +FIX-D null-throw parity extra RULING: ACCEPTED as folded hygiene — 2b-i drift-null only). A1 one-word DIRECT to coord (a)(b)(c) GREEN, matched A2's quote verbatim (verify-before-trust held again). BIN e143847ca672, fire-check+unit+build green, push parity PROVEN. Boot: p1ab3.sh ab3_cellrun dual-cell single-BIN, success bar ANY nonzero twin acc = mechanism CONFIRMED, .42-parity = next datum; P5/P6/P7+NULL dispatch mechanical (ab3_join.py); P7 → convene, no boot-5 without paper. Era BIN chain 2f0d→a4c9→68f0→e745→cf0b→3871→4009→e143 all fingerprinted. Post-AB3 on deck: acc-quality read, then G1 CLOSED → farm merge w/ dflash → run-ci.sh --full (G7 prune decision before package). A3: window offer NOT taken (no user reply pre-sleep; AB3 claimed both cards — stays CPU-idle per least-important order).

### STATE 05:4x EDT — *** MECHANISM NAMED (A1 PRIME c7114a7a, coord-verified) = POOL WINDOW NEVER FUSED ON THE TWIN *** twin d2h_fused :3694→hook-append :3935-3945 has ZERO fuse calls (batched fuses pending_features→d2h_fused at :6462-6465 pre-append); chains-equal@1-2 proves the chain seed-fuses its own read (:4138, per-call fresh d2_fused) so the stale pool never poisons rounds 1-2 — only round-2+ pool reads get garbage ⇒ zero-accept since twin's first breath. FOURTH INV-7 visit; increment-4 hoisted the BUFFER and the missing fuse was born/dressed there (audit lesson: hoists must carry their producer calls). FIX-ORDER TO A2 (code): twin-hook fuse call mirroring :6462-6465 (NOT :4138) + SHAPE AUDIT rows in msg ({5120,T} vs {hidden,T*d2_N}, T=6 both? — window-in-envelope class guard) + pending_features provenance line vs :4138's source → build → fire-check → A1 word → coord stamp → AB3 = FIX-VERIFICATION BOOT (ab3_join.py dispatches P5/P6/P7 mechanically; success bar pre-registered: ANY nonzero twin acceptance = mechanism confirmed; .42-parity = next datum). A1 addendum due (:4138-vs-6462 scoping), then rest-gate re-armed. P7 convene agenda banked w/ free-B3-first (draft-id join from AB3's own log). BIN@AB3: rebuilds on fix (was 4009899313a1).

### STATE 05:3x EDT — AB3 LAUNCH-PREP GREEN (A2 side complete, coord-verified): e79779c3 delta (fire-check rows in msg real) + fcbe53e5 p1ab3.sh (106 lines, syntax OK, P5/P6/P7+NULL-convene+join-recipe+per-cell canaries+FREeze-SOP armed) + b32e2e4c correction (d2mb=0 WITH ERA-CAUSE — tag cannot predate booted logs; static call-shape parity = max honest pre-boot claim). LEDGER: A2's 2nd evidence-verb fiction self-killed in-pass (meta-lesson: 'run the grep BEFORE typing the number'); both filed as culture-wins. BIN@AB3 = 4009899313a1. SOLE REMAINING GATE: A1 one-word covering BOTH (a) instrument delta static, (b) prime pool-window read (woken 09:2x, bounded). A2 slot-filler: P7 convene agenda on paper. If A1 GREEN: AB3 stamps, one command (p1ab3.sh ab3_cellrun). P7 NULL path = convene before boot 5, no freeform. GPU 0 apps; A1 ctx-watch active (swap-risk; commits-before-limit ordered).

### STATE 05:2x EDT — AB2 VERDICT IN (bdccb60c, coord-verified): ring-vs-linear hole NO-DO (gdn_mix_tp unconditionally snapshot-semantics; schedule action-setting inert on TP2 route); twin slot arithmetic internally consistent + capacity proven; **DRAFTER HAS NO GDN STATE** (dflash2_round.cu zero recurrent machinery) ⇒ zero-accept structurally narrowed to TWO input surfaces: (1) pool-window contents at N=1 (A1's hook-feed read — now PRIME), (2) fusion-feed tap window. Gate veto fired correctly: frontier-match impossible boot-only (batched dump gate MB_RINGDBG step>=70 rank0 vs twin GDNHASH step<=4 both-rank) ⇒ AB3 needs instrument-only delta (A2 GO'd to implement, NOT boot; fire-check→A1-word→coord stamp). A1 rest-gate lifted for the ONE prime read (bounded, verdict doc on his branch, swap-wall → commit-derived+honest-open). Two-surface split prevents parallel-rerun collision. Fork value stated: A1 naming mechanism skips the diagnostic cycle (fix-verification boot instead). Gates unchanged: G1 quality-blocked w/ ~2-3h named-fix ETA per user briefing. BOOT-era BIN 387150c10eb0 frozen until delta lands; GPU 0 apps.

### STATE 05:1x EDT — CLOSE-OUT BANKED + ERA QUESTION DISSOLVED ON PAPER (A2 AB1 static, coord store-verified 6f9ac29f/8a9bc91d, push parity PROVEN): vpos-arithmetic shows p1r1 had exactly ONE +2 round in ~29 (a=1 @step~17) and ZERO accepts through round-6 — .067 anecdote was 27f066-boot's own p1b logs; acceptance has been ~zero SINCE TWIN LOOP FIRST RAN; p1r1's .007 ratio honest, perception dressed by x2-counter eras. QUALITY = ORIGINAL question (plausible-never-matching at N=1 vs batched .42), first time measurable clean (no contamination, correct arithmetic, 32 GDNHASH prints). AB PLAN STAMPED: AB2 GO (.cu GDN write-target read CPU + frontier-matched gate self-veto — if gate-match needs code delta: implement→fire-check→A1-word→AB3 stamp); AB3 = ONE dual-cell boot pre-approved in principle (fix-verification if AB2 names mechanism, diagnostic if not); AB4 era-rebuild DEAD (agreed). A1 pool-contents read DORMANT — do not wake sleep-gate keeper for optional reads (static-gate value > diagnostic value). G1 = quality-blocked, decode-COMPLETE stands. Arena KEEP-as-hygiene banked for 169.

### STATE 05:0x EDT — *** m3_boot3: P-SCORECARD 4/4 — THE WALL IS DEAD *** (tree 9a76439e, BIN 387150c10eb0, coord-verified): P1 warmup not-bound 0x / P2 CLIENT 200 finish=stop gen=30 coherent text = FIRST-EVER clean twin response / P3 rounds=31=gen tok/rd 1.00 exactly (counter fix proven in vivo; 59-was-2x31 confirmed) / P4′ lane-0 0x + STATS 1x + RANK1-ERR 0x. G1 wall CLOSED: fall-through-since-d9a4689c unified warmup/boot1-spin/boot2-throw/p1r1-'drain' era stories; deep-round saga retired to close-out. Parity resolved per LAW (tool absent): tp2_budget.h 0-diff, engine hunks zero gate-region content — routing-region edit legit. 1.lane0 restored + trace writer path-scoped (results/dflash2_trace/). NOW LIVE: quality surface only — acc=0.000/31 honest steps; A2 on .cu era read with MANDATORY pre-check: .067 (27f066) predates counter fix — its denominator was double-counted, may be artifact (real 0-.13) not a lost signal; if never-clean-measured, tonight's 0.000 = first honest floor, quality work = context-blindness/INV-7 on clean data w/ GDNHASH lines. Cards released (0 apps). No boot 4 until A2's A/B plan stamped. Gates: G1=quality-blocked (decode COMPLETE achieved), G2 CLOSED, G4 done, G5 rows in, G6 filed, G7 OUT (23G<25G, re-measure at farm7 prune). Then farm merge + run-ci.sh --full per user chain.

### STATE 04:2x-04:3x EDT — m3_boot2 WITH-ERROR CLOSE (A2, cards released, GPU verified idle): decode GREEN to stop boundary (31 steps, 70/70 SYNC parity both ranks, r=1 present THROUGH last bar = 5d teardown-side wall), then BOTH ranks throw 'Paged KV allocation is not bound' (src/core/paged_kv_cache.cpp:459/481, bound_row_<0 on append/commit) — respA = error JSON, wall 20s, bt never armed. UNIFIED-WALL CLAIM banked: every era's ~r29-30 death = stop/completion path, not arena depth. STATS-COUNTER DISCOVERY (A2 :3868 shared acc_rounds double-increment, gen_before-family survivor): rounds=59≈2×31 → p1r1's '58 rounds' was ~29 steps → 1GiB drain arithmetic built on PHANTOM DENOMINATOR; fix dc92bddf rank-0-only LANDED (coord-verified real delta); two falsifiable edges pre-registered for next boot (rounds≈steps post-fix; binding error still terminal). ARENA REVERT RULED: not tonight, disposition rides farm7 package (795-round grower cure is real regardless). A1 BOUNDED WAKE SENT: name the missing rebind at twin completion vs batched :1872 kvarn-reset lineage + fix-shape ranking. QUALITY SURFACE (accepted=0/31 steps; .067-era regression) stays SEPARATE from the wall — vc/zero_slot fair-test pending, GDNHASH=1 on card env. A2 disclosed :3633 self-correction (7a1b2ea1): P-A doubly inert (false AND d2_N>=2) — culture green. Stray dirty 'M 1.lane0' flagged for cleanup. Gates: G1 = binding fix → boot → quality; G2 code+review CLOSED; next GO only after A1 names site + A2 implements + static + stamp.

### STATE 04:2x EDT — m3_boot2 LIVE (pid 802115, both cards 14.3GiB, A2 SOP-exec: kill_list, card w/ 5a-5d readings incl. A1's P-A v4 review GREEN — v4.h:26 sig spot-verified, anti-ratchet row test :44/:49, G2 review seat DISCHARGED; gates: G1=live boot, G2=code+review green, CI-cell wiring row remains). A1 told to rest drafts till bt-frames or green-loop acc-quality question. GO stamp crossed A2's msg; re-stamped.

### STATE 04:1x EDT — *** GO STAMPED — m3_boot2 RUNNING (A2, both cards, BIN e74560a03a65) *** A1 cumulative static GREEN + coord cold-verify of every cite (WP1 0 in-loop allocs, hoist 12 :3678-98, verbose 15th-arg FALSE :3634/:3977 single-seq vs TRUE :5613 batched = P-A doubly inert, hook_cols≡T :3903, teardown :3990-4003 local/no-collective, RATE PUZZLE RESOLVED: acc=0.000@0.5tok/rd was the FAILED-WARMUP line :868 pre-listen, real req ~1.0 → gen~29 AT step29 = NATURAL STOP ⇒ stop-boundary hypothesis arithmetically closed pre-boot). Amended readings on card: exact-stop-repeat=teardown forensics+run-2 LOW VALUE skip / divergent=surprise (ping before run-2) / named throw=capture datum / green-past-stop=S3-kill admission then acc-QUALITY question (.007-era vs batched .31-.53). bt-before-kill armed. Next: A1 P-A v4 review seat (fdd2327c real diff, reject-level); A1 ctx 79% — told to commit drafts early. Broker down; A3 idle CPU-done.

### STATE 04:0x-04:1x EDT — PRE-GO POLISH: A2 stack now 7d817be6 (HEAD; scripts/docs-only since fdd2372c — BIN e74560a03a65 stands; push parity PROVEN). COORD CATCH #2 (pre-GO): server default max_tokens=8192 + body has ZERO cap fields ⇒ A2's 'default explains gen=30' inference FALSE (gen≈30 = NATURAL STOP/EOS); retraction+amended readings landed 19d7b3c0/7d817be6 (A2 self-caught a rule-8 no-op in its own message — disclosed, muscle-memory credit, ledger item). NEW BRANCH FOR RE-BOOT READING: stop-boundary hypothesis — r29 freeze sits at stop/teardown boundary, exact-repeat now EXPECTED (teardown-path forensics), run-2 low-value-if-exact; divergent = surprise. A2's arithmetic tension folded as sub-reading: stop-round is acc-era-dependent (0.007-era stop@~58rds/30tok vs zero-acc-era ~15tok@r29) — prints discriminate: rank-1 posting prints PAST stop into teardown = different wall than freeze AT stop token. A1 cumulative static (hoist/prints/capture/P-A-inert/hook_cols≡T + bounded teardown-path look, 79% ctx — told to deliver gate-items first + commit before swap, honest-open on optional part acceptable). Cards idle-held; A1 word → GO stamp → p1bt.sh m3_boot2.

### STATE 03:4x-03:5x EDT — INCREMENT-4 + P-A v4 LANDED (A2 pushed stack: d58b20ba→5d6ab363→6030296d→8bc09cde→fdd2327c=HEAD). Coord cold-verify GREEN on all (push parity, live BIN e74560a03a65 md5-match, [D2SS-SYNC] ×3 prints env-gated/flush/rank-stamped, hoist real memcpy-only-in-loop, d2_N>=2 P-A guards grepped :4201/:4205/:4399). BOOT DECISION (a): re-boot on fdd2327c/e74560a03a65 (P-A provably-inert wiring rides: one slot not two, debrief ordering, 169-chain unbroken). A1 CUMULATIVE STATIC = last gate: hoist completeness / print non-lying / capture non-lossy / P-A twin-inertness end-to-end / hook_cols≡T verdict (coord evidence: :3903 const hook_cols=T; batched twin separate :6363 st.staging). Era rows: 2f0d=stalled, a4c9=capture, 68f0=inc4, e745=boot tree. p1bt.sh yama-proof freeze-SOP banked (wrapper exec-gdb @90s log-static; bt_all.txt both ranks; dummy-proven). Boot dirs pre-staged m3_boot2/3. A1 SHA-typo self-corrected (fdd2327c). A3 revoked-ack CPU-idle. Cards idle-held, GO stamped only on A1 GREEN.

### STATE 03:3x EDT — A1 ADJUDICATION IN + A2 COMMITS RULE-8 GREEN: grower NAMED (outer-loop d2h_fused 60KiB/rnd + mb_* set, :3797-3806/:3892-3894, arena exhaust ~rnd 795 = farm7 landmine, NOT the r29 wall) ⇒ r29 = rank-1 CONTROL-FLOW divergence (b) refined. Re-boot tree = capture (d58b20ba LANDED, :3242 look-alike untouched) + entering-sync prints (pre-sync_bar :3953 / pre-allreduce_argmax :3752 / pre-chain-allgather, env-gated/flush/rank-stamped) + grower hoist = increment-4 commit (A2, P-A v4 paused mid-flight to take it). A1 statics increment-4 (3 watch-points: hoist completeness, print non-lying placement, try-boundary interaction) = LAST gate before GO. BIN eras: 2f0d14a14b39=stalled boot era, a4c9ac9c992e=capture era (md5-verified live). BOOT PROTOCOL +1: log-static >90 s ⇒ gdb thread-apply-all bt on BOTH ranks BEFORE kill (bt_rank0/1.txt). Pre-reg FOUR readings: divergent-round (race, expected) / exact-29-repeat (deterministic, loud flag) / green-loop (flake) / capture-names-throw. Push parity PROVEN (github tip 5d6ab363). A3 CPU-only done+idle (242d6c32 q3 op-test wiring). Cards idle-held for A2. A1: increment-4 static → P-A v4 review seat → frame-class only on bt.

### STATE 03:0x EDT 09-11 — USER ORDERS LANDED (user → bed): A2=DRIVER, A1=assist, A3/q3=LEAST IMPORTANT (GPU grant REVOKED pre-claim, CPU-only until G1 CLOSED); goal chain = dflash completed → farm merge w/ dflash → run-ci.sh --full; GPU strictly serial, coord grants only; incoming/nvfp4 (18G) = REQUIRED artifact, never prune. M3_BOOT1 DATUM (A2 seq-3): deciding boot (67b18499/BIN 2f0d14a14b39, card-verified) = 4TH SHAPE — early-barrier-spin @step29-30: both ranks' last trace byte-identical, one-core userspace spinner (wchan=0, others futex), GPU0 100%/GPU1 0%, acc=0.000@8-rounds same boot, warmup "Paged KV not bound" ×2 both ranks; drain NOT cured, no acceptance curve; halt-on-divergence executed, teardown clean, bundle banked (results/bcode_m2/m3_boot1 + STALL_NOTE). Pre-reg now has branch (iv)=early-barrier-spin; run-2 on capture-hunk tree doubles as repeatability discriminator (per-round grower⇒near-exact repeat; async race⇒divergent). A1 gate read: name surviving per-round grower inside chain path NOT reclaimed by arena rewind (KV block tables / drafts lane / slice buffers) OR certify rank-1-silent-death artifact — dea0f900 static was GREEN on the dead tree, so arena math round-signature (~30) surviving statics is itself the contradiction to resolve. A2 CPU-now: rank-1 inner-exception capture hunk (MUST RIDE re-boot tree), card-mislabel fix, classification commit; P-A v4 proper = G2, never blocks G1. NO BOOT until coord stamps GO. 6th-surface rule armed (stop→report→coord decides fire-with-label). G7 OUT (23G < 25G floor) — re-measure after post-boot build/tests trim. Disk map measured: habitat build 22G, incoming 23G (18G nvfp4 KEEP + 5.5G dflash2 keep), repo/build 343M.

## ═══ HANDOFF — COORDINATOR COLD-START (2026-09-11 ~06:50 EDT, coordinator 01a08ba1 thinning; NEW SESSION READS THIS BLOCK) ═══
**Role rules**: read AGENTS.md first (VRAM LAW / ANTI-RESURRECTION / CELL CARD absolute). Address agents BY NAME via intercom (agent1, agent2, agent3 — resolve by ID on collision; user fixed one tonight). All coord dispatches carry tag **C441** (phantom-coord messages occurred 09-10 — untagged/phantom-SHA = quarantine + paste to agent1 or user; `git cat-file -t` before believing any SHA; MECH rule-8: message-vs-delta on every claimed commit; rule-6 era: era-provable boot bundles or context-not-verdict; rules 11-16 REPO.md). Card→written-GO-before-boot is ABSOLUTE (3 ungranted-boot incidents tonight, all by A2, all honest+clean; rule tightened: GO is a separate artifact from instructions). gemini broker :3421 DOWN since ~20:13 EDT 09-10 — user relaunches; until then agent_comm pulls fail; agent3 (q3 lane) = user-owned, CPU-only, awaits user GPU grant.

### LIVE RIGHT NOW (06:50 EDT, 09-11)
- **A2's deciding boot RUNNING** (session 01a08f32, fresh — took over after previous A2 session's long shift; its boot line was textbook): tree=habitat branch 167-piecewise-fix @ **67b18499** (REAL arena commit — persistent per-lane chain_arena hoisted outside chain call, batched symmetric via st.batched_chain_arena [also retires A1's farm7-long-gen latent flag], trace gate step<=4; A1 static GREEN all 3 watch-points dea0f900), BIN 2f0d14a14b39, results/bcode_m2/m3_boot1→boot2 (repeat-vs-repeat EXACT pre-registered). **Three-way pre-registered reading (from A1's map option_b §ROUND-2 + my GO dispatch): (i) acc rises ≥~.2 + no deep throw ⇒ drain+quality cured → proceed phases 2-5 incl S3-KILL (single-seq vs plain 31.5 / vs own batched 21.7, 256-tok bodies m2b_10k_* tail-distinct) — verdict band = perf-regime / footprint-only / perf-dead-capacity-alive, all landings pre-agreed; (ii) acc near-zero but stable ⇒ A1's .cu GDN write-target read (his bounded pass) goes next; (iii) binding-throw at depth again ⇒ arena-print escalation.** Halt-on-first-divergence; era-bundle commit + drain proof at close.
- **A1** (01a08ce2, ~53%, honest ledger, self-caught grep-anchoring + pipe-eaten + phantom-anchor errors same day — reliable): armed for (ii) .cu read; owns P-A(v4) REVIEW after A2's P-A commit lands (fix-path he wrote in 29f06e57: lattice memcmp-witness + wire caller + placement named; P-A currently INCOMPLETE-in-store — 053908f4 is header+unit, zero callers); owns G7 disk re-check + farm7 package fill-in (169 shell drafted @ b7b34a72 lineage, pushes to github/wo/host-compaction-staging, tip verify first).
- **A2 queue after boot**: P-A proper commit → boot battery phases 2-5/S3-kill (if (i)) → flips (OPTION_B_LANDED/GUARD_V4_LANDED) ride their merges. Its fresh-session handoff = this block + COORDINATOR.md prior STATE lines + its own boot-line message.

### BOARD FACTS (verified, don't re-derive)
- main=149c92d9 (VRAM-law 5240a889 + docs/161 + all 09-09/10 merges). habitat tip=67b18499 (pushed). A1 branch tip=see its pushes (docs lanes). step-0 parity: region diff 0-lines on A1 branch (verify via tools/ops/step0_region_parity.sh, rule-8 shape).
- **MTP-FIXED-CLOSED**: second site = stale-process artifact; chunk-relative 1-liner on main (:2645, rides merge item-2); MTP batched GREEN to 79k fill (24.6/30.6/79k rows, must-NO-WRITE zero); nvfp4 WEIGHTS artifact + multibatch GREEN to 100k (42aa3823; gate-refusal-without-flag = designed). nvfp4 KV-cache-type lane: pool-sizing 3.7x over-alloc finding stands (A1 165 queue; farm7 first-rung arbiter G6).
- **dflash batched = DARK-BY-PERF CONFIRMED** (plain 27.6 > dflash 24.2 @30k; plain 31.5 / MTP 38.4 > dflash 21.7 @10k, same-boot 2-rep <1% variance; 5dbb0c7e/6c688c36) — correctness complete, ships labeled or behind Option-B gate. W5 archived by user's own 25% bar (11.8%, fa70e091), machinery OFF.
- 2b-iv class CLOSED (guard FP on templated bodies; guard v4 spec = 164 ADDENDUM 4; collapse-only THROW + WARN demotion; anti-ratchet falsifier law).
- FARM7 GATES: G1 dflash-complete = the boot above; G2 P-A (in flight); G3 flips verified-0; G4 fire-check v3.1 DONE-GREEN; G5 wiring rows IN (capbuffer #6/#6b, auditor registered); G6 nvfp4 rule filed; G7 disk 23G now (fresh rebuild regrew habitat build ~20G — post-boot prune build/tests re-closes 25G floor, coordinator call WITH user in package). Then: farm7 --full run (~90min) → USER NOD package (drafts/169 = 5 items + era register + S3 outcome + prune ledger; one screen) → merge (night-merge NOT-FF pattern) → WO-G5 rerun → q3 GPU bring-up (agent3, user-gated) → compaction 158 slices 3-4.
- Open user decisions queued: nvfp4 18G container keep/cut; farm7 fire-now-vs-hold-for-quality; broker relaunch; gemini test-lane re-entry brief (C441 + rules 6-8 briefing when he arrives).
- Tonight's process ledger (for the exhibit's methodology section): phantom-coord incident (contained, source never identified — quarantine discipline is the mitigation); 4 CI defect classes pre-farm (pipe-eaten exits ×2, vacuous contract, display-only gate, message-vs-delta claim — ALL caught by same-night instruments: fire-check, auditor, MECH rules); 'review-confirmed gets equal suspicion' law operating on all lanes incl. coord entries.
**If you are the incoming coordinator**: verify board cold (git log tips, nvidia-smi, df, intercom list), then await A2's boot datum — it routes to branch (i)/(ii)/(iii) above; A1 owns the .cu fallback. Do not re-dispatch what's in flight. Pings route by name; keep pinger loop alive (~/.pi/agent/tmp/pinger/loop.sh, text updated each phase).

> **Timestamp correction (coordinator, self-reported).** The AMD-line STATE blocks above
> were stamped from the **host-local** clock (America/New_York, EDT = UTC−4) while carrying a
> `Z` suffix, so everything from the former "03:5xZ" through "06:0xZ" read four hours early
> against the hub's UTC clock (SQLite `datetime('now')`, and the `[[AGENT-COMM]]` envelopes'
> own `sent …Z` fields). They have been shifted to true UTC in place. The earlier blocks
> (00:5x–02:1x) were already correct. **Going forward: STATE times come from `date -u`, never
> from the shell prompt's clock.** This is the same class of error I have been logging against
> the lanes all night — a value quoted from the wrong reference frame, then trusted — and it
> was in my own ledger, where it would have quietly mis-ordered tonight's sequence for the next
> reader.

---

# ═══ HANDOFF — COORDINATOR COLD-START (AMD line, 2026-09-12 ~12:4xZ, coordinator 01a092fc thinning) ═══
**A fresh coordinator session reads THIS block first**, then `AGENTS.md`, then the two STATE blocks above it. Do not act on the NVIDIA-line blocks further up — those describe another host's queues.

## In one screen
The AMD/V340L line's **only blocker was a wrong warp-shuffle emulation, and it is fixed**:
`__shfl_*_sync` used an **early return** for out-of-group lanes, but on GCN those intrinsics lower to
`ds_bpermute` — a **wavefull** LDS exchange that publishes a lane's slot only if the lane
**executes** it. So lanes 16–31 opting out of the `delta=16` step left lanes 0–15 reading stale slots:
exactly half the row lost from every width-32 butterfly. agent2's fix forwards the caller's `width` and
selects out-of-group with `cndmask` so all lanes still execute — goldens green (`120→496`, `1520`,
uniform-row d=128 `inv 0.125→0.088388`), with an **ISA tripwire demonstrated RED 5/5** against the
unfixed tree. **Work pending: rebase onto `amd/main`, merge, confirm the build, then one stamped launch.**

*Two diagnoses that were wrong on the way here, recorded because the pattern is the lesson:* first,
"the bug is passing `warpSize` instead of `width`" — the guard's **predicate** was already CUDA-exact, and
my literal prescribed fix was compiled and **still broken**. Second, mine earlier still: a width-32 group
reasoning that cleared the shim because **the defect only exists across multiple steps** — any
single-step enumeration, including the one I wrote, will pronounce it clean. Hence: gates test ISA
behavior, not call-site semantics.

Everything else on the line is done: whole host layer + `targets/` + `serve/` compile under HIP, CI lane
(`run_ci_amd.sh`) green zero-GPU, measured anchors cross-validated (intra-device ~183 GB/s; **no P2P on
this box at all**), T1 17/17 whitelisted, shuffle audit reconciled at 159 sites with blast radius 27.
**Nothing waits on Team Green or the user.**

## Branch & doc layout (changed today by user order)
- **Integration branch: `amd/main`** (pushed, tracking set). **Never push or merge to `main`** — that is Team Green's line and pushes were colliding. Merge `main` **into** `amd/main`, one direction only.
- **All AMD docs live under `docs/amd/`**: `COORDINATOR.md` (this file), `COORDINATOR_ARCHIVE.md` (provenance ledger), `README.md` (lane registry — **this is the old `docs/174`**), `v340l/` (lane docs: `00` product WO, `01` test-lane WO, `02` shuffle audit, `PROGRESS.md`), `wq/` (coordinator work orders: **WO-02**, **WO-03**).
- **Global `docs/NN` numbering is retired for this line.** Cite `docs/amd/...` paths.
- `consolidated 51 files` (run_ci_amd.sh, tests/v340l/, 26 results artifacts, both WOs) that previously existed **only on two unpushed local branches on one disk** — if you see a lane branch, check it is merged to `amd/main`.

## Roster & mesh
| Lane | Who | Reach | Currently doing | Owns (disjoint — do not cross) |
|---|---|---|---|---|
| Coordinator | **you / this session's successor** | intercom `coordinator`; hub `coordinator` `4032c47e` | dispatch + cold-verify | `docs/amd/`, grants, rulings |
| Product | **agent1** (NEW session) | intercom `agent1` | **WO-03** | `targets/serve/apps` whitelist, dependency graph, single-device stub set, AR design doc |
| Product | **agent2** | intercom `agent2` | **WO-02** | **`src/common/hip_shim/cuda_runtime.h`** + golden + blast-radius column |
| Test/CI | **gemini** | **agent-comm hub `Gemini` `b4791a54`** (NOT intercom) | calibration **frozen** pending WO-02 | `run_ci_amd.sh`, PG gates, audits |
| Other team | Team Green (Q3 artifact + TP2 dispatch) | `origin/wo/q3-gemv`, `main` | — | not a dependency; answered them at `docs/q3_amd_port_questions.md` |

Pi lanes use **intercom**; gemini uses the **hub**. Never route pi↔pi through the broker.
**§7.x waiver question is still UNANSWERED by the user.** Current working reading: gate/test *authorship* = gemini; agent1 running a device to check a kernel it ported = product verification (§4), allowed. It is **not blocking** anything.

## Queue (ordered, with owner)
1. **WO-02** agent2 — shim width fix → goldens **496 / 1520 / inv 0.088388** → 159-site table gets an *affected* column → re-verify T1 greens (one stamp needed; they will ask).
2. **WO-03** agent1 — `targets/*`(22)+`serve/`(13)+`apps` whitelist (compile-only, parity line + sha1 per commit) → re-derive first-token graph vs the shuffle defect → **loud-failing AR/tp2 stub set** so a single-device build can LINK → **AR host-staged redesign design doc**.
3. **gemini** — after WO-02 lands: lift the cross-lane tolerance freeze, fit PG-B bands, wire per-file gates for T1/T2.
4. **Then** T2 GEMV (7 files, w8 — or re-derived Q3 tier) and T3 attention (plain bf16-KV first, int8-KV as T3b). 4-die/TP4 work stays **out**: `engine.cpp:309-313` supports 2-rank or single-device only; `docs/59` row 8 keeps TP4+ as HORIZON.

## Grants
**None live.** G-AMD-1…12 all closed with release rows; last device use <10 s. Cards at baseline: 4 × `18,575,360 B`, "No KFD PIDs". **Rule: compile windows and launch windows are stamped separately** (a lane must not sit on an expiring device grant while doing CPU work) — that cadence came from a 14-min overrun on a 10-min window, self-reported.

## Measured facts — cite, do not re-derive
- 4 HIP devices, **gfx900**, **7.98 GiB/device**, **56 CUs** (rocminfo; *not* 64), **wavefront 64**, ROCm **6.2.0-66**, hipcc 6.2.41133, **no nvcc**, **no passwordless sudo**.
- **No P2P whatsoever**: `hipDeviceCanAccessPeer` = 0 for all 6 pairs incl. same-card dies. Host-staged: **cross-card 6.61–6.70 GB/s**, **same-card 4.86**; RTT **~98–101 µs flat with size** (≈50 µs/hop) → **collective count is the budget**; ~130 AR/step unbatched = 13 ms/step (dead), 2–4-layer batching = 3.3–6.5 ms.
- Intra-device D2D sustained **~183 GB/s** (179.81–184.26 across two independent instruments) = **the ceiling every bandwidth claim is bounded by**; theoretical per-die 483.8 GB/s. **Any % of roofline must name its denominator.**
- dev→card **not identity**: dev0→card1, dev1→card3, dev2→card0, dev3→card4 (card2 = NVIDIA display). Corroborated 3 ways; exact pairing corroborated only by behaviour, still worth a during-load sampler.
- Toolchain: `/home/chris/opt/cmake/bin/{cmake,ctest}` 3.30.5 (tarball, sha256 recorded; **pip cmake forbidden here**). No cmake/ctest system-wide.
- **Two confirmed toolchain traps:** `__bfloat16_as_ushort` is a **numeric cast** on ROCm while CUDA's reinterprets bits (its own docstring lies) → use `__hip_bfloat16_raw`/`memcpy`; ~~bf16 arithmetic is compiler-rejected on gfx900~~ **CORRECTED 14:3xZ (agent2 v340l/03(b), coord-reproduced): bf16 arithmetic COMPILES on gfx900, silently lowering to ~6-instruction fp32 RNE emulation — D3 fp16/fp32 stays LAW, but enforcement is a CI/ISA-fingerprint gate (owed by gemini), NOT the compiler: compile-pass is not evidence of D3 compliance.** `hip_bf16.h` does **not** pull `cuda_fp16.h` the way CUDA's does.
- Launcher conventions: shape is **`{d, rows}`, `ne[0]` = feature dim**; `argmax`'s `valid_rows` = **vocab scan limit**, not token count.
- **T1 is 17/17 whitelisted** (parity 154/154 at the time) but device-verified **partially** — some greens were vacuous (NaN-degenerate `layer_norm`) and are being re-run under the fix.

## Invariants I enforce; keep enforcing
1. **Verify before you rule** — read the file, run the arithmetic, `grep` the **pushed artifact**. Tonight: 4–5 confident static stories died to measurement, **two of them mine** (geometry tax; nearly refuting gemini's correct shuffle diagnosis).
2. **Claimed ≠ proven.** A run is only its log; a log is only valid if **freshness is proven by content** (marker string + binary/archive hashes in-log, rebuilt from committed HEAD in the same message). mtimes lie; stale binaries fooled us twice.
3. **A measurement beating the measured ceiling is a broken instrument, never a result.**
4. **VRAM LAW** (user order, absolute): no estimated VRAM charge may ever refuse a launch; `hipMemGetInfo` + actual bytes; no reserve constants; near-capacity cells must LAUNCH and MEASURE.
5. **§7.x**: gemini owns the test lane; never reassign it to agent1/agent2 — surface to the user and **wait**.
6. GPU serial + written grants; own-PID kills only, never `pkill -f`; `df -h` before >5 GB builds; **commit before reporting**; cite refs **by name** (`amd/main`), never a pinned SHA; **tallies cite window+filter or stay out**.
7. Ledger mechanics: STATE goes **above** the previous header (never use an existing header as an edit anchor — it consumes it; bit me twice), and STATE times come from `date -u`, not the local clock (mislabeled by 4 h once).

## Known repo hazards (reported, deliberately not fixed)
- **`results/157_FINDINGS_sglang_gap_investigation.md` contains committed conflict markers** (`<<<<<<< HEAD` at :139) from Team Green's merge `7fa9a5ca` on `main`. **Not ours to resolve** — we don't know which side is authoritative and it's their file; **flag to the user** (done) and leave it.
- Team Green's `COORDINATOR.md` (refreshed 06:25) is **authoritative for the NVIDIA line**; our `docs/amd/COORDINATOR.md` is authoritative here. My earlier rename would have deleted theirs — caught pre-push, resolved non-destructively.
- agent1's predecessor ended its session cleanly. Its session report **is** consolidated here:
  `docs/amd/v340l/02_session_report_2026-09-12.md` (verified present on `amd/main` after merging
  `wo/v340l-hip`, 17 commits, `c32dece6`) alongside the no-P2P peer-probe matrix,
  `first_token_files.md`, the G-AMD-8/10 run logs and harnesses. Its WO §8 definition-of-done is
  **not** met: steps 4/6 unreached, step 5 re-scoped by the no-P2P finding.

## Open with the user (only they can move these)
1. **§7.x waiver** for the AMD line (cleaner/faster; not blocking).
2. **Q3 artifact `bytes` + which file serves** (`qwen3_8_27b_q3.ninfer` md5 `e57258df…` vs the converter's `…_iq3.ninfer`) — Team Green is producing it; three sizes are in play (11.1 GB arithmetic / ≈12.7 GiB ledger / 13.67 GiB plan of record) and **nobody should size KV off prose**. Also: no IQ3/Q3 entry in `src/core/dtype.h` — Q3 lives on the `NumericFormat`/`QType` axis (`typed_binding.cpp:23-24,50-53`).
3. **Root for PG-0c** (pinned-clock peak), else ~183 GB/s sustained stands as the ceiling.

## Live branch map at handoff (verified 2026-09-12 ~13:2xZ — read this before anything else)
| Ref | Tip | What it means |
|---|---|---|
| `origin/main` | **`d5868323`** | Team Green has landed **more since our split** (our `main` ref is `be970ead`). First task: `git fetch origin && git merge origin/main` **into `amd/main`** — one direction only, never the reverse. |
| `amd/main` = `origin/amd/main` | `eab05e9b` | The line's integration branch. **Still carries the BROKEN shuffle emulation** — agent2's fix is not merged here yet. |
| `wo/v340l-shfl-fix` | **`012c5735`** | **THE FIX.** Lives only on this local branch. Goldens green, ISA tripwire demonstrated RED on the unfixed tree. Base is the pre-consolidation tree. |
| `amd/wo-shim-width` | `eab05e9b`, **1 file uncommitted** | agent2's WO-02 worktree, rebased onto current `amd/main`. Expect a re-land/merge here — it is the branch you should merge, not the older one, **after checking which one has the verified build**. |
| `amd/wo-engineserve` | `9228ea48`, clean | agent1's WO-03 worktree. **No commits yet** — first report is the `targets/serve/apps` parity line. |
| `wo/v340l-hip` / `wo/v340l-phase-gate` | `7c8e836a` / `1322bfbb` | Pre-split lane branches, both already merged into `amd/main`. Do not base new work on them. |
| `wo/q3-gemv` | `56986f6f` | Team Green's Q3 lane + our answered question set (`docs/q3_amd_port_questions.md`). Not our branch; push access is theirs. |

**Consequences you must act on, in order:**
1. **The line's own branch does not have the bug fix yet.** Until `012c5735` (or agent2's rebased equivalent) lands on `amd/main`, **every HIP kernel with a cross-lane reduce is silently wrong by ~√2**, and gemini's tolerance calibration is correctly frozen. Merging it is the single highest-value action available to you.
2. `git merge origin/main` into `amd/main` before/alongside that, so the step-0 anti-resurrection cell is clean rather than firing on a branch that is merely behind.
3. **G-AMD-13 is stamped but UNCONSUMED** (precondition: merged base + green CPU build, then one 90 s dev0 launch). Cards are idle; nobody has launched since. If a lane asks, confirm the base is merged before stamping.
4. After the landing: tell gemini to lift the calibration freeze, then push T1 re-verification, then T2/T3.

## First 10 minutes for the incoming coordinator
1. `git fetch origin && git log --oneline -3 amd/main` — see if agent1/agent2 landed anything (WO-02's goldens are the thing to look for).
2. `rocm-smi --showpids` + `--showmeminfo vram` — confirm baseline before any stamp.
3. Read WO-02/WO-03 and each lane's last report; **stamp whatever is honestly blocked on you** — both lanes were told to ask, not wait.
4. Check whether the tolerance freeze has been lifted only *after* the golden is on disk and passing.
5. Re-verify, don't re-read: every number quoted to you was measured by someone who was wrong at least once tonight — including me.

**State at handoff:** `amd/main` @ `c32dece6` (handoff block `9da48d4c`; both lane branches now merged in — `wo/v340l-phase-gate` and `wo/v340l-hip`), tree clean, no live grant, cards idle, both product lanes dispatched with disjoint file ownership, test lane deliberately paused pending the fix. — **C441**

### GRANT G-AMD-14 (written, 17:3xZ) — T3 device verification window
**To:** agent3. **Base:** amd/t3-wip @ a50eecde (census 0/0 labeled host-pass; real build rc=0 parity 164/164 sha1=e85d71f0d0; goldens 4/4 — coord-verified: tables real 24,577+3,073+6,195 values, K-path claim byte-checked at prefill_bf16.cuh:55/:59/:61, base cited-by-name correct). **Devices:** dev0+dev1, flat-topology pair, non-load-bearing. **Window:** ≤10 min. Build-in-message from a50eecde (freshness by sha, tables pinned to it). Run decode small-t + prefill SIMT on oracle inputs; compare under bands: ≤1e-3 pass, 1e-4..1e-3 ORDER-FINDINGS named per cell (fp32 accumulation order — not smoothed, not failed), >1e-3 FAIL LOUD with table rows. timeout 30/run, stop-and-report-partial, release row verbatim, own-pid only. Zero product mutation in-window; any launcher fix needed = STOP+report, new stamp. agent2 may be invited for second-eyes (gemini's harness-authorship preference noted: if gemini is responsive by window start, tables are self-describing either way).
