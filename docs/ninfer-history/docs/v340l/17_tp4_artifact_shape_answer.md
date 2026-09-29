# TP4 Phase 2, blocker-question #1: is the q3 artifact TP2-shaped? — measured answer

**Who:** agent2. **When:** 2026-09-13 ~01:1xZ. **Cost:** zero GPU; two bounded head-reads of manifest
JSON, no artifact hashing (CIFS law respected — the manifest is a JSON directory at a fixed prefix,
so reading it is not re-hashing 15 GB). **Refs:** q3 at
`/media/chris/EMTEC256/qwen3_8_27b_q3.ninfer` (canonical per the path ruling), ref at
`artifacts/qwen3_8_27b.ninfer`. Code read at `origin/amd/main` @ `426e2ff2`.

## Answer, in one line

**No — q3 is not TP2-shaped.** It is rank-agnostic, and so is the reference artifact. TP4 is therefore
an **engine port**, not an export decision; it does not cross Team Green's promote machinery. The port
**fails loudly** today rather than silently corrupting, and the geometry divides cleanly by 4, so the
only substantive work is four hard-coded `TpRole` cases.

## 1. The artifact carries no rank structure at all (measured, not inferred)

| property | q3 | ref |
|---|---|---|
| `weights_id` | `groupwise-q3` | `groupwise-int` |
| objects | **1124** (6 resource + 1118 tensor) | 1190 (6 + 1184) |
| manifest object fields | `name, kind, offset, bytes, encoding` | same |
| encoding values | `raw-bytes-v1` | same |
| names containing rank/shard/half/split/part | **0** | **0** |

* **q3 ⊂ ref, strictly**: `q3-only = 0`, shared = **1124**, ref-only = **66** — and all 66 are
  `dflash2/**` (the speculative-decoding module). The 66-object delta the count invites someone to
  explain is a *feature absence*, not a shard difference.
* **No shape field exists in the manifest.** Objects carry byte-offsets and lengths only; matrix
  geometry lives in the product's plan layer. So a "per-rank K-half structure" is not something the
  artifact *could* encode — this is the load-bearing structural fact, and it is why the question
  resolves in the loader rather than the export.
* 995 of 1124 shared objects are byte-length-identical; the 129 that differ are the quantized tensors
  whose width follows `weights_id`, i.e. expected.
* Same six `frontend/*` resources in both, so tokenizer/chat-template payload is shared too.

## 2. Where the split actually happens: `shard_row_split(..., rank, num_ranks, ...)`

`src/core/multi_gpu/weight_shard.cpp:56` takes a **full** source payload and derives rank-local
geometry at load time — `rows/num_ranks` for ColumnN (`:65-66`) and `groups_per_row/num_ranks` for
RowK (`:70-73`). `src/targets/qwen3_6_27b/impl/load/tp_load.cpp:192` `multi_ranges(role, rank, world)`
computes ranges arithmetically. Nothing anywhere indexes objects by rank. Consequence for TP4: the
artifact needs no re-export and no promote-side change — `num_ranks` is already a runtime parameter of
the sharding function.

## 3. The actual blocker: 4 of 8 roles bake in world=2, and they are NOT parameterised

`tp_load.cpp:206` and `:222` do it right — `MultiRangeQKV` and `MultiRangeGQKV` divide by `world`
(`6144/world`, `1024/world`, `2048/world`, `6144/world`). The four in `:214-217` do not:

```
case TpRole::MultiRangeQK:  return {{r * 3072, 3072},        {6144 + r * 512,  512}};
case TpRole::MultiRangeGV:  return {{r * 3072, 3072},        {6144 + r * 512,  512}};
case TpRole::MultiRangeGQK: return {{r * 1024, 1024},        {2048 + r * 1024, 1024}};
case TpRole::MultiRangeGZ:  return {{r * 3072, 3072},        {6144 + r * 3072, 3072}};
```

`3072 = 6144/2` and `512 = 1024/2` are the **TP2 halves written as literals**. There is no
`static_assert`, no runtime `world == 2` check, and `default:` returns `{}`; the only related comment is
`tp_engine.cpp:862`, a prose assertion that the engine is "TP2 (world==2) unconditionally" — a remembered
list, not a compile-refusing mechanism.

### Severity: LOUD, not silent — verified against the real fused dims

My first read said "silent wrong weights." That was wrong, and the correction matters: the guard
`weight_shard.cpp:129-131` throws `shard: row range exceeds source rows` against **`src.rows`**, the
*whole fused* tensor's row count. Resolving the tensors from the role map at `:260-268`
(`attention/query_key` → `[7168,5120]`, `gdn/query_key_value_z`-family, etc.) and replaying the
arithmetic at world=4:

| role | fused rows | first offending rank | outcome |
|---|---|---|---|
| MultiRangeQK / GV | 7168 | r2: `6144+3072 = 9216 > 7168` | **throws** |
| MultiRangeGZ | 12288 | r2: side-2 `12288+3072 > 12288` | **throws** |
| MultiRangeGQK | 4096 | r2: side-2 `4096+1024 > 4096` | **throws** |

So a naive `world=4` attempt **aborts at load** rather than training on rank-0's q-block. Worth stating
plainly because it is the good news: nobody has to audit a TP4 run for silent corruption, and the
existing row guard is what makes that true.

**But note the guard is coarse**, and this is the one place I would not rest easy: it bounds against
*total* rows, so a range that stays inside the fused tensor while landing in the **wrong block** would
pass. Here ranks ≥2 happen to overrun the total and so are caught, but that is arithmetic coincidence —
a parameterisation that "fixed" the overruns by clamping instead of dividing would convert a loud
failure into a silent cross-block read. Two distinct invariants need asserting, not one: a rank's
ranges must stay inside **their own block**, and their union must **partition** it exactly.

## 4. What the delta-map should do with this

1. **Do not open an export/promote work item.** No Team Green dependency for TP4 from the artifact side;
   q3 and ref are both rank-agnostic and shared-name.
2. **The engine port is 4 cases**, and they already have a correct pattern to copy from two lines away
   (`:206`, `:222` divide by `world`). Parameterise `6144/world`, `1024/world`, `2048/world` and drop the
   literals.
3. **Add the two assertions this analysis shows are missing** — per-block containment and exact
   partition — in the form that *refuses to compile or fails at load*, per the standing law that a
   contract change needs a mechanism rather than a remembered list. Today's protection is an accident of
   the total-rows bound.
4. **The `weights_id` divergence is orthogonal** to TP4 and already gated: the q3 file's `=129`
   (`Q3G64_F16S`) arm is armed per CI, and ref carries no Q3G64 — that pin distinction is recorded in
   `docs/amd/v340l/06` and my closeout.
5. **Group divisibility for a 4-way RowK split: checked, and it passes.** `columns = 5120` with the
   q3 group size of 64 gives `groups_per_row = 80`, and `80 % 4 == 0` — so `check_div(groups_per_row,
   num_ranks)` holds for TP4. On the row side every relevant fused tensor divides by 4:
   `attention/query_key` 7168, `attention/gate_value` 7168, `gdn qkvz` 16384, `mlp/gate_up` 34816.
   So TP4 is **arithmetically viable** and the blocker is purely the four hard-coded `TpRole` cases —
   not a shape or divisibility obstacle. Not established: per-tensor columns for *every* one of the
   1118 tensors (I checked the TP-partitioned widths), and `check_div` remains the real gate at boot.

   Caveat kept rather than hidden: a probe for `encoding == 'raw-bytes-v1'` on tensor objects returned
   **0** — tensors carry no `encoding` field, only the six `frontend/*` resources do. So byte lengths
   are the only per-tensor geometry the manifest itself provides, and the widths above come from the
   plan layer. This does not change the divisibility result, which follows from published geometry.

## 5. Method notes, since this is a measured answer meant to be re-run

Manifests read by parsing the 16-byte prefix then the declared JSON span; both objects and shapes taken
from the file, not from memory — the coordinator's "1124-object manifest" checked out exactly, and the
`66` delta turned out to be a different thing than a shard split, which is the one place a remembered
summary would have misled. The severity reversal in §3 is a self-correction: the first pass read
per-block totals where the guard tests total rows, and produced a confidently wrong "silent corruption"
claim that only the role→tensor lookup fixed. Both the wrong and the right version are recorded because
the difference between them is the finding.


---

## RETRACTION OF §1'S CENTRAL PREMISE — the manifest DOES carry shape, layout and format

This document's headline rested on "the manifest objects carry ONLY name/kind/offset/bytes/encoding —
there is **no shape field**", and I repeated that to the coordinator as the reason the TP4 question
resolves in the loader. **It is false.** Re-parsed the q3 manifest directly:

    tensors=1118   keys present: name 1118, kind 1118, offset 1118, bytes 1118,
                  shape 1118, layout 1118, format 1118, encoding 6 (the six frontend resources)

So **every tensor declares `shape`, `layout` and `format`**, and layouts are not decorative:
`row-split-k128-v1` on 439 tensors, `contiguous-le-v1` on 679. My earlier probe asked tensors for
`encoding`, got 0, and I concluded the manifest held no per-tensor geometry. That is the
wrong-token-in-the-wrong-place error I have now diagnosed in four other people's posts: an absent key
read as an absent concept. The one true part of my sentence is that *encoding* is only on resources —
I simply generalised from that to "no geometry".

### What the corrected data changes, and what it does not

**The conclusion survives, now positively rather than by absence.** `text/layers/*/mlp/gate_up` is
declared `shape=[34816,5120]` — the full fused gate+up, not a per-rank half — and
`attention/query_key` is `[7168,5120]` full. Nothing is halved, so the artifact is still
rank-agnostic and TP4 is still an engine port, not an export/promote decision. It is better argued as
"declared shapes are full" than "shapes are absent".

**But the corrected fields expose a real TP4 risk my answer missed.** Under the declared group size of
128, four `(rows, cols)` shapes have `groups_per_row` NOT divisible by 4, so a 4-way RowK split
would hit `check_div(groups_per_row, num_ranks)` and throw:

    (1152,1152) (3456,1152) (4304,1152)   groups_per_row = 9   -> 81 tensors, 9 % 4 = 1
    (1152,4304)                            4304 % 128 != 0      -> not an integer group count at all

The last one is also a warning about my own reading: for that family either the `k128` tag does not
carry the meaning I assumed, or its group size differs. All rows are 4-divisible, so the ColumnN axis
is clear; only RowK is at risk, and which tensors take which axis is decided by **role in the plan**,
not by the artifact. So the honest statement of the answer is now:

> **The artifact is rank-agnostic — but `layout` is a real field, and a 4-way RowK over the
> `row-split-k128` population is not trivially even. Before parameterising the four `TpRole` cases,
> resolve which of the 81 `groups_per_row = 9` tensors are RowK-split; if any are, TP4 needs a
> group-size or role decision that crosses the export, not only the loader.**

That is a materially different plan input than the one I filed, and it should be re-derived rather
than trusted — the command is in the commit that added this section. I also mis-axis the first pass
(this uses the columns side, per `kGroupsPerRow = kSplitK / group_size`; my first computation used
`shape[0]` and produced a wrong pass/fail list), so treat the enumeration as the thing to re-run.

Two lessons I would keep, both earned: **an absent key is not an absent concept** (probe for the key
you mean before declaring the data absent), and the answer that unblocks a delta-map deserves the same
adversarial re-read you would give a claim you were disputing — mine got it only because another lane
asked me to verify a claim about my own receipts.
