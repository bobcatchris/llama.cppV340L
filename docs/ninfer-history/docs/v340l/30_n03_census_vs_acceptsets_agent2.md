# §N0.3-CENSUS VS ACCEPT-SETS — formal arm list at the REAL artifact + hour-1 landmine map (agent2, chair seq-182 leg 2)

Subject: `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer` @ sha256
`eaf8ad124256d0a0c1ebbbca442ca58eee4f97ab34a60a0b4d57e2b41e2c56d2` (verified day-1, doc 29) —
1307 objects. Acceptor side: **`bind_qwen38_nvfp4_text_layers`** + `bind_artifact` at amd/main
(bytes read post `origin/amd/main` merge state; the 3.6 binder `bind_nvfp4_text_layers` is NOT
this artifact's path). Method: header parse + per-leg presence/routing check in host python,
zero card, read-only. This doc is the **arm list** agent3's incoming gate cell
(census-vs-accept-sets, chair scope-call (b)) should be built to compute mechanically — the
numbers below are what it must reproduce at green.

## ADJUDICATION OF THE CHAIR'S CITED LANDMINE (said plainly, because the row cites a retracted claim)

The seq-182 phrasing — "layers 0..2 gdn/query_key_value_z divisors ABSENT per agent1's §7.4" —
**does not hold at the real bytes**: every gdn layer INCLUDING 0, 1, 2 carries
`gdn/query_key_value_z` (NVFP4, 16384×5120) AND its paired
`gdn/input_projection/input_scale_divisor` (FP32, 4 B). Zero hard-leg absences anywhere.
agent1's own §7.5 re-audit already retracted §7.4's table for THIS file (that column was
producer-inventory-derived and described the 09-08 sibling export). The chair's brief propagated
the dead row; the landmine map below replaces it with the MEASURED routing, so the hour-1 map
is built on bytes, not on an inventory that was already corrected once.

## ACCEPT-SET ROUTING CENSUS (per 64 layers; 80 routed legs total)

- **NVFP4 legs taken: 71 · BF16→W8G32-requant legs taken: 9 · hard-leg absences: 0.**
- The 9 requant legs, exact names (these are the hour-1 landmines — different CODE, needs the
  tp_load pass-2 writer green):
  - `text/layers/{3,7,11,15,19,23}/attention/query_key_gate_value` — BF16 [14336,5120]
    (guard absent ⇒ `bind_weight(BF16)` + `requant_target=W8G32_F16S`; matches the source's own
    comment "layers {3,7,11,15,19,23} (manifest-verified)" — **comment and bytes AGREE here**)
  - `text/layers/{3,7}/attention/output` — BF16 [5120,6144] (guard absent ⇒ requant)
  - `text/layers/4/gdn/output` — BF16 [5120,6144] (guard absent ⇒ requant; source comment
    "layer {4}" agrees)
- Layers 27,31,35,39,43,47,51,55,59,63 attention legs: NVFP4 path (divisors PRESENT); layers
  11,15,19,23 split their way (qkv BF16-requant, output NVFP4) — the mixed routing is REAL and
  per-layer, which is why the gate cell must grade per (layer, leg), never per family.
- mlp: every layer took the NVFP4 leg (all 64 `mlp/*_projection/input_scale_divisor` present);
  the `kFp8` else-leg (`FP8_E4M3FN_ROW_BF16S`-declared mlp bind) is UNREACHED by this artifact —
  its zero-FP8 census (doc 29) means that leg's silent-shape-only form never fires HERE, but the
  GATE must exist before any re-export carries FP8 endpoints (agent5 §8h accept-set law, and
  chair's audit law at plan :429 already says this).

## TOP-LEVEL & MTP LEGS (all PRESENT; strictness labelled)

```
text/token_embedding      W8G32_F16S [248320,5120]  declared FP8 by endpoint_format(Qwen38Nvfp4)
text/final_norm           BF16       [5120]         strict leg (require_tensor: format+layout+shape)
text/output_head          W8G32_F16S [248320,5120]  declared FP8 — same class as embedding
text/draft_head           Q4G64_F16S [131072,5120]
text/draft_head_token_ids I32        [131072]       + validate_draft_ids SCANS PAYLOAD at load
mtp/{input_projection W8G32 5120×10240, embedding_norm BF16, hidden_norm BF16, layer/input_norm
BF16, layer/attention/{query_key_gate_value W8G32 14336×5120, query_norm/key_norm BF16 256,
output W8G32 5120×6144}}  — all PRESENT as declared.
6 frontend resources present (tokenizer.json, tokenizer_config, chat_template, generation_config,
preprocessor_config, video_preprocessor_config).
```

## THE SILENT-CLASS PAIR, NAMED FOR THE GATE CELL (this is the real §7.4 mechanism, surviving its own re-audit)

`endpoint_format(Qwen38Nvfp4)` **declares `FP8_E4M3FN_ROW_BF16S`** for the vocabulary endpoints;
the artifact carries **`W8G32_F16S`**. The check that catches this DOES NOT EXIST on this leg:
`bind_weight` → `require_weight_tensor` (binder.cpp:66-80) compares **SHAPE ONLY** and OVERWRITES
the declared format with the descriptor's (`resolved_format = tensor->format`). So boot does not
throw and does not misread-quant-math either (the plan carries the ACTUAL format downstream) —
what it does is let a **declared-vs-real format lie ride through silently at the FIRST tensor
read** (token_embedding is hour-1 object #1 by offset: 12,837,376). agent1's §7.4 cure proposal
(strict check in `require_weight_tensor` when a format was declared, or explicit ANY-marker at
call sites) remains the named fix; until it lands, **the census-vs-accept-sets gate cell is the
only organ that sees this class at merge time** — that is the argument FOR chair scope-call (b)
and for agent5 re-pinning the A-rows to the gate cell's sha (my registration-grade when it
lands: three-state legs must include THIS pair as a planted-declaration falsifier: FP8-declared/
W8-actual must convict as a NAMED mismatch, not as silence).

## HOUR-1 LANDMINE MAP — objects a boot touches FIRST (offset order = read order; all names exact)

1. `text/token_embedding` — W8G32 vs FP8-declared (silent-pass, item above) — **the #1 datum**.
2. `text/layers/0/input_norm` (BF16, strict) → `gdn/a_log`, `gdn/dt_bias` (FP32),
   `gdn/convolution`, `gdn/a_projection`, `gdn/b_projection` (BF16; split-form leg — fused
   `a_b_projection` ABSENT in this file, guard routes to split correctly).
3. `text/layers/0/gdn/query_key_value_z` + `text/layers/0/gdn/input_projection/input_scale_divisor`
   — NVFP4 UNGUARDED leg, **PRESENT** (the chair's cited absence: REJECTED at bytes).
4. `text/layers/0/gdn/output` + `gdn/output_projection/input_scale_divisor` — NVFP4 leg (layer 0
   is NOT the bf16-4).
5. `text/layers/0/mlp/gate_up` + divisor, `mlp/down` + divisor — NVFP4.
6. First REQUANT landmine at read position ~11 layers deep: `text/layers/3/attention/`
   `query_key_gate_value` (BF16 [14336,5120]) — consumer must use the W8G32 planes the tp_load
   pass-2 writer mints, else this is the garbage-GDN false-negative class wearing an
   attention-layer mask. Layers {3,7} then also demand `attention/output` requant;
   {11,15,19,23} demand qkv requant ONLY.
7. `text/draft_head_token_ids` — the ONLY leg with a PAYLOAD validator at bind time
   (`validate_draft_ids`: ids < 248077 and unique over 131072) — first real-bytes run of that
   function happens at this boot; it reads 512 KB from the artifact; no failure expected
   (exporter-side ids presumably lawful) but it is hour-1 untested code on real bytes — named.
8. MTP block ValidateOnly by default (`features.optimized_propose()` off) — validate-only legs
   still SHAPE-check; all present.

## GATE-CELL ARM LIST (what agent3's cell should enumerate mechanically — chair brief says BUILDING; my grade on landing)

A. per-leg presence: name ∈ manifest ⇔ acceptor leg reachable (guard truth-value computed from
   manifest, not hardcoded layer lists — the source comments {3,7,11,15,19,23} / {4} become
   OUTPUTS to check, not inputs to trust).
B. declared-format vs actual-format per leg, with the strictness of the ACTUAL check the code
   performs (shape-only vs strict) as an arm input — a declared≠actual pair where the code is
   shape-only is a NAMED SILENT row (today: exactly 2 — the vocabulary endpoints), NOT a pass.
C. requant-leg census: which layers route to BF16→W8G32 (today 9) — each needs the pass-2
   writer's admission shape-set {14336×5120, 5120×6144} coverage CHECKED (the A2-requant
   admission wall from the 9b9efefb-era note lives here now: my old NVFP4-prefill lane's
   `bf16_dispatch` admitted EXACTLY the shapes 14336×5120 and 5120×6144 — both are the ones the
   9 legs need ⇒ lane parked but its census datum transfers, cited).
D. three-state: absent required leg rc=1; format-class overlap on a shape-only leg = rc=1 NAMED;
   instrument cannot resolve a guard pattern = rc=2 (never a green).
E. pairing: NVFP4 tensor ⇔ sibling divisor object, 1:1 (today 247/247 exact — my day-1 row).

Registration grade when it lands: same battery as the §N0 family (content8, build+run at my
seat, falsifiers planted, gate-letter/CTest registration named, three-state both-directions) —
and per phantom-pins law I re-resolve agent5's A-row cites to the GATE CELL'S sha at landing,
flagging any that stay pinned to desk-path drafts.

— agent2, zero card, host read-only; census script transient (/tmp/accept_set_census.txt
raw table this beat, numbers above are its transcript).
