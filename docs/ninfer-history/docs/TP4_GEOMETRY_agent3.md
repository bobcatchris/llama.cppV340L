# WO-TP4-C — agent3's geometry rows: world=4 is CLEAN, and the A-1 divisibility guard is VACUOUS at 4

**Session:** agent3, pi `01a09af1-26b4`. **Zero-card, zero-boot, zero-fetch.** Derived at
`amd/main` tip from the LIVE part table, not from this sheet's own numbers — every row carries the
command that re-derives it.
**Scope:** `WO_TP4_all_lanes.md` §WO-TP4-C items 1–3. Parent: `TP4_DELTA_MAP_v1.md` (RULED: pure engine
port), `TP4_world2_inventory_agent5.md`. Consumes A-1 as landed (`a34d378b`, chair-merged):
`tp_local_shape` + `multi_ranges` now derive from ONE `tp_part_sizes` table.

> **HEADLINE, and it is a gate finding more than a geometry finding:** at world=4 every fused part is
> divisible AND every resulting slice is a whole number of 128-row head blocks, so **the geometry is
> clean — and precisely because it is clean, `require_parts_divisible` can NEVER fire at 4.** A
> paper-test written to satisfy A-1's "assert divisibility loudly" clause **passes vacuously** and
> certifies nothing. The mutation arm that makes the assertion mean something is **world=3**, and it is
> not in any spec yet. This is this lane's most expensive law — *a mutation arm proves sensitivity,
> never coverage* — applied to a gate that had not yet been asked whether it could fail.

## 1. C-1 — fused-range math at world=4: DERIVED, and the derivation lands on the spec's numbers

Source of truth read at tip (`tp_load.cpp:220-239`):

```
GateUp [17408,17408]  QKV [6144,1024,6144,1024]  QK [6144,1024]  GV [6144,1024]
GQK    [2048,2048]    GZ  [6144,6144]            GQKV [2048,2048,6144,6144]
```

`multi_ranges` splits **inside each part** (`rank r` takes `[base + r·h, h]`, `h = part/world`) — the
A1-review-5563465b semantics: a contiguous row split would give rank0 all q/k and rank1 none. Computed
local slices at w=4:

| role | parts | `part/4` each | local rows (Σ) | heads@128 |
|---|---|---|---|---|
| GateUp | 17408, 17408 | 4352, 4352 | 8704 | 34, 34 |
| QKV | 6144, 1024, 6144, 1024 | 1536, 256, 1536, 256 | 3584 | 12, 2, 12, 2 |
| QK / GV | 6144, 1024 | 1536, 256 | 1792 | 12, 2 |
| GQK | 2048, 2048 | 512, 512 | 1024 | 4, 4 |
| GZ | 6144, 6144 | 1536, 1536 | 3072 | 12, 12 |
| **GQKV (fused gdn)** | 2048, 2048, 6144, 6144 | **512, 512, 1536, 1536** | **4096** | 4, 4, 12, 12 |

**The WO's pre-declared expectation — "q_l=k_l=512, v_l=z_l=1536 at world=4" — is CONFIRMED by the
derivation, not assumed to be.** And the skews are *preserved*, which is what "DERIVED, not divided"
means operationally: 6144:1024 = 6:1 → 1536:256 = 6:1; GQKV 1:1:3:3 → 512:512:1536:1536 = 1:1:3:3.
Non-fused roles at w=4: ColumnN vocab 248320/4 = **62080** (remainder 0); RowK gdn/output K 6144/4 =
**1536** (rem 0); GdnConv cols 10240/4 = **2560** (rem 0).

Re-derive: `sed -n '/tp_part_sizes(TpRole role)/,/^}/p' src/targets/qwen3_6_27b/impl/load/tp_load.cpp`
then divide. 48 attn q-heads → 12/rank; 8 kv-heads (1024/128) → 2/rank — both integers, which is the
inventory's §2 worry resolved by arithmetic rather than by confidence.

## 2. C-2 — where the guard actually fires, i.e. the falsifier the A-1 test needs

`require_parts_divisible` throws iff `∃ part : part % world ≠ 0`. GCD of all 14 distinct part values =
**1024**. So:

```
world : guard at w=4?      which roles would throw
1     : vacuous            —
2     : vacuous            —          (world=2 history: the guard NEVER protected the shipped config)
3     : FIRES              GateUp, QKV, QK, GV, GQK, GQKV   (2048%3=2 is the smallest trigger)
4     : VACUOUS            —           ← our target
5,7   : FIRES              all seven roles
6     : FIRES              all seven roles
8     : vacuous            —
```

**Consequences, stated as gate policy not as opinion:**
1. A TP4 paper-test that asserts the loud-throw behavior **at world=4 cannot fail**. If it is written as
   "construct at 4, expect throw-free, PASS", its PASS carries no information about the guard. It must be
   paired with a **firing arm at world=3** (or 5/6/7) on the same table, and the test must NAME which arm
   produced which result. World=3 is the cheapest firing case: one modulus, `2048 % 3`.
2. The sufficiency gap is real even though w=4 is safe: divisibility is NOT sufficient for head
   ownership. At **world=16** the guard passes for **every** role, yet four of them produce
   head-misaligned slices — `QKV/QK/GV` give a 64-row kv slice (1024/16) and `GateUp` gives 1088
   (= 8.5 × 128) — none a whole number of 128-row head blocks. **A silently-wrong placement that no
   current check catches.** (Computed: `[(p//16) for p in parts if p%16==0 and (p//16)%128]` → GateUp
   1088, QKV/QK/GV 64; GQK/GZ/GQKV are aligned at 16.) **Recommendation (a spec amendment for the
   A-series owner, not a silent fix, per WO-TP4-C's own rule):** the guard should assert
   `(part/world) % kHeadBlock == 0`, not `part % world == 0`. Not TP4's path today (w=4 is clean on both
   tests), so this is filed as hardening, priority below the 4-card window, and **it does not block
   A-1**.

## 3. C-3 — Replicate fallthrough and the census row

`classify_tp` returns `TpRole::Replicate` as its **terminal fallthrough by NAME** (`tp_load.cpp:312,
350`), so the world-parameterized path never divides a replicated tensor: gdn control (a_log, dt_bias,
a/b), norms, and **embedding** are read whole by every rank. At w=4 the fallthrough is unchanged because
it does not consult `world` at all — **role-based, so the map's "hazard inert" ruling holds at 4 without
new arithmetic** (verified by reading the function; there is no `/w` on that path to re-derive).
The WO's "81-tensor" figure is agent5's inventory count and is **not re-derived here** — it is a census
of the artifact manifest, not of this function, and I am not restating a number I did not measure.
`bf16_gdn_norm_gating`'s `norm_splits=32` is gated on `is_35` (problem.heads==32) and so is **inert for
this 48-head artifact at any world** — same conclusion my §2 of the refutal sheet reached from the
opposite direction, and the reason the split-k family cannot be item 7's carrier.

## 4. What I did NOT do, so nobody thinks it is done

- **No device numbers.** NS16/NS32 panel divisibility per kernel: `grep -rn "NS16|NS32|panel"` over
  `w8_config.h` and `w8_rowsplit_gemm_mma.cu` returned **zero hits** — the panel names in the inventory
  are not identifiers in this tree, so C-1's "panel divisibility per kernel" clause is **UNLOCATED at
  tip by me** and needs agent5's inventory row for the vocabulary before it can be derived. Flagged, not
  skipped silently. Per-kernel smem/occupancy at 12-head GDN slices (A-4's compile-check) likewise not
  attempted — it needs a build, and law `f49dfe4a` says build only in my own lane worktree.
- Q3G64 arm tallies at world=4: the 129 census being architecture-invariant is the map's claim, not
  mine; I only verified there is no `check_div`-style guard on the Replicate path to re-derive.

## 5. Overlap with agent5's shadow pass — stated so the board does not pay twice

agent5 has chair pre-authorization for a zero-card shadow staging pass on C-1. **§1 above IS C-1**;
rather than have two seats derive the same table, the useful split is: agent5's inventory keeps the
*per-kernel panel/NS vocabulary* (the thing I could not locate, §4) and their `check_div` sweep, and
**my rows become the second-witness check on §1/§2** — if agent5's independent `part/4` numbers differ
from §1 at any row, that discrepancy is the finding, and it is resolvable in one command each. My §2.1
guard-vacuity result and §2.2 world=16 sufficiency gap are new to both sheets and belong to whichever
A-series test lands first.

— agent3 (pi `01a09af1-26b4`). Zero cards, zero boots, zero fetches, own branch `amd/t3-wip`.
