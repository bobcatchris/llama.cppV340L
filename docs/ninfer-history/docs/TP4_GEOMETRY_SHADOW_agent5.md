# TP4 geometry — SHADOW pass (agent5), WO-TP4-C trigger armed 13:57Z

**Status**: shadow/second-witness by chair designation (seq-30 item b). If agent3's
`TP4_GEOMETRY_agent3.md` lands first → this file is the mandated second witness; if it slips →
this becomes primary, theirs the check. Either way the derivations stand on their own bytes.
**Baselines (re-derive at use)**: part table read from agent4's pushed lane ref
`origin/amd/wo-p3-serve @ a34d378b` (WO-TP4-A1 — **NOT yet on amd/main**, verified
`merge-base --is-ancestor` = false at 13:5xZ; the live main `multi_ranges` still carries the
remembered-half literals at :214-217, which is WHY the A-1 paper test exists); artifact shapes
from `classify_tp()` comments @ `origin/amd/main`; ceiling 65,536 B static smem MEASURED on this
box (agent3's probe @ `d5a47e5f`, cited via T3_HANDOFF §B4 correction — the honest-annotation
form, since the earlier "71,680 needed" relay was struck there).
**Instrument**: `results/amd/TP4_C1_shadow_geom.py`, pure host Python, rc=0, 29/29 `ok` rows
(post-S3-resolution count; was 26/26 with the flag open), runnable at any seat (zero build, zero
device). Every row below carries predicate + provenance.

## S1 — qkvz fused-range at world=4 (C-1 headline): CONFIRMED by derivation

Declared parts `[q 2048; k 2048; v 6144; z 6144]` (A-1 table, = artifact `gdn/query_key_value_z`
[16384,5120], sum verified). Per-part independent split `part/world` (head-block law from A1
review 5563465b — a contiguous 16384/4 block would give rank0 all q/k, garbage GDN):

| part | w2 (shipped, equality pin) | **w4 (derived)** | predicate |
|---|---|---|---|
| q_l | 1024 | **512** | 2048 % 4 == 0 → part/world |
| k_l | 1024 | **512** | same |
| v_l | 3072 | **1536** | 6144 % 4 == 0 |
| z_l | 3072 | **1536** | same |
| Σ local rows | 8192 | **4096** | = 16384/world ✓ (consistency, not the derivation) |

The WO's "(q_l=k_l=512, v_l=z_l=1536 at world=4)" reproduces to the digit from the table — the WO
text and the A-1 derivation AGREE; nothing inherited was trusted.

## S2 — divisibility census of every sharded family at world=4

MultiRange parts: gate_up [4352,4352], attn-qkv [1536,256,1536,256], QK/GV [1536,256],
GQK [512,512], GZ [1536,1536] — **all parts % 4 == 0** (26/26 instrument rows). RowK cols:
down 4352, attn/out + gdn/out 1536. GdnConv 10240 → 2560. KV channels 512/world = 128 (map §5).
**Zero throw-paths at w=4 for the text model.** The 81-tensor class (vision `groups_per_row=9`)
remains inert by the SAME mechanism the map settled: role-based `Replicate` fallthrough, verified
in current main's `classify_tp` — re-confirmed, not re-quoted (C-2 duty).

## S3 — RESOLVED (self-disposition per chair 14:33Z; artifact config.h row-math, zero-card):
**the map's prose is wrong at the head-count level, and my 8×128 guess was wrong too — the real
geometry is head_dim 256.** `src/targets/qwen3_6_27b/impl/config.h:28-30` (committed, this tree):
`query_heads = 24, kv_heads = 4, head_dim = 256` → q part 24×256 = **6144 ✓**, kv part 4×256 =
**1024 ✓** — the part sizes are exact with 4 kv-heads of 256 dims. Neither "8 heads × 128" (my
shadow guess, built on the GDN head_dim) nor the map's "48 q-heads / 16 kv-heads" (those are the
**GDN** head counts, config.h:22-25: gdn_value_heads=48, gdn_key_heads=16, gdn head_dim=128 —
2048/6144 parts ✓ with THOSE) describes attention. The map row conflated the GDN head pair with
the attention parts. NAMED AMENDMENT REQUEST to `TP4_DELTA_MAP_v1.md` (the 4-exchange precedent
shape, found-not-fixed so far by me): replace "48 q-heads / 16 kv-heads attn" with
"24 q / 4 kv attn at dim-256; 48 v / 16 k GDN at dim-128" — and note map §5's "24 heads/rank
(48 total) → 12 heads/rank" IS correct for GDN v-heads (48/4=12, matching `check_div`-class
head maps), it just belongs to the GDN row, not the attention one.
CONSEQUENCE FOR SHARDS (the question the row actually gates): per-rank head counts at w4 are
attn 6q+1kv/rank (24/4, 4/4 — group ratio 6 PRESERVED; GQA needs q_heads%kv_heads, holds), GDN
12v+4k/rank (48/4, 16/4 — head_map G = 12/4 = 3 UNCHANGED, `common.cuh:105-110` fastdiv stays
exact). Divisibility of the parts was already shown (S2); this row upgrades it from
"bytes-divide" to "heads-divide too" — the stronger statement kernels actually consume.
Original flag text kept below as written (annotate, never delete):

Artifact k-part 1024 rows = **8 heads × 128 dims** OR 16 heads × 64 — the map's prose says
"16 kv-heads" (48 q/16 kv) while 1024/128 = 8. Divisibility at w=4 is unaffected (1024 % 4 = 0
either way) and NOTHING in this pass changes a shape on account of it. Flagged for the map owner
(gemini) + agent3's primary to resolve explicitly — a head-count that disagrees with its own row
math is exactly the class this shadow pass exists to catch before a kernel author spends a boot on it.

## S4 — NS16/NS32 panel geometry at world=4: PASS, by the right reason — **CLAUSE RETIRED BY
NAME (C-scope 1, chair 17:10Z): "NS16/NS32" is MY OWN inventory vocabulary (world2_inventory
§-reference + the map's §5 row), NOT code identifiers — agent3's grep-0 at tip is correct and
complete. The names map to real constructs as follows, and that mapping IS the retirement:**
- `kernel_dims<16>` / `kernel_dims<32>` specializations, `state_passing.cuh:28-46` — the template
  argument is **NStrip** (N_STRIP_PER_BLOCK = 16 or 32 state-dim columns per block);
  `D_STRIPS = kStateDim(128)/NStrip` ∈ {8,4}, `THREADS` derives from it. THESE are the "NS16/NS32".
- The DISPATCH that picks them (`state_passing.cu:57-64`): `cfg.H_v >= 48 → launch_fixed<16>`,
  else `<32>` — **world-relevant!** H_v is per-rank (`launch.cu:47`, from `v.ne[1]` of the LOCAL
  shard): TP2 ships H_v=24/rank → takes the <32> arm; at world=4 H_v=12/rank → still <32>. The
  <16> arm is selected only at per-rank H_v>=48, i.e. TP1 or non-sharded vision routes — **the
  4-rank port never enters the arm whose smem row my S4 originally quoted.** Grid math
  (`check_grid(H_v * D_STRIPS)`): 12×4=48 blocks, no divisibility term; head_map G = H_v/H_qk =
  12/4 = 3 exact (fastdiv legal) — all under S2's heads-divide row.
- `prepare_wy_wu.cu:42` `cfg.H_v == 32 → <32,16,4>` is an EQUALITY dispatch on per-rank H_v: at
  world=4 H_v=12 does NOT match, falls to the default pair — a SILENT dispatch change across
  worlds that no check_div guards, worth one row at A-4's warmup/cols derivation (agent4's seat):
  verify which arm TP4 actually selects for ALL THREE chunked kernels, at RUNTIME, first boot.
- Real `check_*` sites are `stage_validator::check_shape` (head-count validity
  `are_head_counts_valid`: H_v>=H_qk && H_v%H_qk==0 — passes at 12/4) and `check_full_chunks`
  (T%64 — world-independent); there is NO per-world divisibility gate in the GDN kernels at all
  — the divisibility gate lives in tp_load (S2), the kernel guards are head-ratio guards.
Original S4 text below stands as written (annotate-never-delete), superseded in READING: the
"panel cols × BC, not head count" claim was right for smem BYTES but missed that the ARM CHOICE
is head-count-driven — which is the row that actually matters at 4 ranks.

state_passing `static_assert(kStateDim == 128)` + panel-cols ∈ {16,32}/{32,64}
(`prepare_wy_wu.cuh:50-51`); compile-time-evaluated smem (agent3's probe): output 24,576 /
prepare <32,16> 24,320 / <64,32> 29,440 / **state NS16 45,312 / NS32 57,600** — max 57,600 <
65,536 measured. **The bytes are functions of panel columns × BC, not of per-rank head count** —
world=4 quarters the GRID (24 heads/rank → 12, per map §5) which adds blocks, not smem. So the
panel divisibility check at 4 ranks is satisfied by construction for the same reason the shipped
kernels fit today; the residual is only that the launch-extent math in the plan layer be
world-parameterized (agent4's A-series territory, WO-TP4-A-4 "warmup/cols arithmetic — DERIVE at
4, do not extrapolate" — this pass notes the constraint, the derivation belongs to A-4).

## S5 — 2:1 row skew re-check (the WO's :214-226 citation)

At LIVE main those lines are the remembered-halves literals (the pre-A-1 form the refactor exists
to kill); at a34d378b the same ranges derive from the table, and the skew that is *designed-in* is
the attn q:k 6:1 (6144:1024) and the gdn 1:1 (2048:2048, 6144:6144) — per-rank PRESERVED at w=4
because parts split independently. The G1-era measurement stands: a perfect even split is
impossible for gdn/query_key's 2:1 fused qkvz (G17_window_report (D) VOID note) — unchanged at 4,
the per-part order is what carries the semantics, and it survives.

## S6 — what a slip-vs-land means for the window

Nothing in S1-S5 blocks; no deviations from the map found EXCEPT S3's prose-vs-row-head mismatch
(flagged, zero shape impact) and the map's tree-sketch error already amended (174b1d84). The
4-card window's shape inputs are therefore **converged from two seats pending agent3's primary** —
if their doc agrees on S1/S2/S4 digits, the A-series + window plan need no amendment; any digit
disagreement between this file and theirs is the trigger to stop and re-derive at the table source
before ANY A-4 warmup/cols commit lands.
