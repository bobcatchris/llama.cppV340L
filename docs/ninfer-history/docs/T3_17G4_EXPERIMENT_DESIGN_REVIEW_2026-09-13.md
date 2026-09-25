# 17g4 char-15 divergence: the decisive experiment is confounded, and the evidence is generated text

**From:** agent3 (pi `01a0984f-cd62`), read-only analysis of agent4's banked artifacts at
`amd-wo-p3-serve/results/amd/p3/`. **No boot taken, no device touched, no edits to agent4's lane.**
**Why it matters:** agent4 is about to spend a stamped device window on a replay whose result cannot
discriminate the hypotheses it was designed to separate — and the one measurement that *would*
discriminate them costs nothing and was already supported by the binary.

## 1. char 15 is inside the model's self-report, not a telemetry field
```
17f  [0:30] 'The user said "Hi." This is a '
17g4 [0:30] 'The user said " " which is a v'
first differing char index = 15
```
The divergence sits inside `The user said "…"` — **generated prose in which the model describes its
own prompt.** The verdict row reads it as *"an input-reconstruction-level difference,"* i.e. it treats
the model's claim about the input as a measurement of the input. LLMs misreport their own prompts in
reasoning text routinely; and note the model also calls it *"a very short, casual greeting"*, which is
a sane reading of a message it allegedly saw as nothing.

## 2. `prompt_tokens` is **54 in both boots**
| boot | prompt_tokens | rendered self-report |
|---|---|---|
| 17f | **54** | `The user said "Hi."` |
| 17g4 | **54** | `The user said " "` |

A genuine `"Hi."` → `" "` substitution should move the token count, and it doesn't. This is
**suggestive, not conclusive** — template overhead dominates a 54-token prompt, and equal counts are
compatible with differing IDs. But it points the same way as §1: the input probably did not differ.

## 3. The record that would settle it was never captured — and it needs no new boot
`ninfer-serve` supports `--request-log-jsonl FILE` (*"appends full-precision server/request
records"* — visible in the binary's own usage dump at `/tmp/tps_window/control.log`). It is **not**
present in `G17g4_window.log`, and the serve log never echoes the received prompt:
```
grep -oE "request-log-jsonl[^ ]*" G17g4_window.log     -> nothing
grep -iE 'hi|" "' G17g4_serve.log                       -> nothing
```
So the single field that answers "was the input different" was available and uncollected. **Adding it
to an existing boot does not create a new truth** — it is an extra field on the truth already being
measured — so it should be free inside the current stamp rather than costing one.

## 4. THE DESIGN PROBLEM: the replay cannot separate (a) from (b)
agent4's decisive experiment is `REPLAY dev3,2 IDENTICAL`, against hypotheses
(a) rank0-card determinism · (b) order-dependent numerics · (c) nondeterminism.

Replaying one identical pair order separates **(c) from {a,b}** — deterministic-or-not. It **cannot
separate (a) from (b)**, because across the three reference boots, rank0 identity, rank1 identity,
pair identity and reduction order all move together. In a 2-rank launch, "which card is rank0" and
"which shard talks first" are the *same* degree of freedom, so the stated perfect correlation with
rank0 identity is **equally well correlated with rank1 identity and with pair identity**. A
reproducible char-15 will be read as *"determinism per order"* when all it proves is
*"determinism per pair."*

**The discriminating run is a cross, not a replay:** hold rank0 and vary rank1 — `(dev3,dev2)` vs
`(dev3,dev0)`.
- outputs **match** → rank0 alone determines it → **(a)**.
- outputs **differ** → the far side matters → **(b)** (pair/order), and (a) is a special case of it.
Then one same-order replay still serves to kill (c). That is a 2×2 over two boots instead of one boot
that answers a question it can't answer — worth noting because this lane already named the shape:
agent4's own `dladdr`-stub-map case and my `#6` compound-cause shadowing were both *"one green, two
possible causes, only a witness row separates them."*

## 5. This lane's actual prior on (b) — it is live, and here is its mechanism
My sweep **cannot** explain an input-side difference: acts 1/2/3 touched **zero** tokenizer/prep/embed
files (`git show --name-only | grep -iE "token|prep|embed|input"` → nothing beyond
`w8_gdn_input_gemm_splitk.cu`, a kernel). **But** `launcher/gqa_attention_prefill.cu` and
`gqa_attention_decode.cu` are whitelisted **and** carry swept code, so the (A)-seam numerics sit squarely
in the reduction path of every boot in question — (b) is not dismissible from my side.

The mechanism is already measured in this lane's history. Parity produced **19/24576 outliers**,
classified as razor-edge physics — near-ties within ~1 ULP — and the **2-ULP guard was retired to a
`ratio<0.85` outlier detector for exactly this reason**: at low token indices, near-tied logits sit
close enough that a small change in summation order can flip the argmax. **An order-dependent
reduction flipping an *early* token is precisely a char-15-scale effect**, so (b) has a specific,
previously-observed mechanism here rather than being a generic worry. Conversely, if (c) came back
live, that would be a genuinely new finding and much bigger than the char-107 note it displaces.

## 6. Referral
All of §1–§4 is agent4's experimental design and the chair's stamp call; I have opinions and no
custody here. Nothing in this note reopens a closed item — boot-block, two-hop, and width-32 remain
verified closed on `origin/amd/main`.

— agent3 (pi 01a0984f-cd62). *Read-only inference from banked artifacts; the two claims I could have
fabricated — char index 15 and prompt_tokens 54/54 — were computed from the JSON with `python3`,
not transcribed.*
