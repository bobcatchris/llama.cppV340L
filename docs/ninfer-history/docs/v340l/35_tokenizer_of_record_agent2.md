# TOKENIZER-OF-RECORD — agent2's titled leg, contract row (chair ruling seq-74 item 2)

One row, as ruled. Everything under it is either that row or the evidence that the row's instrument
does what the row says.

## THE ROW

| field | value |
|---|---|
| **TITLE** | tokenizer-of-record for the NVFP4 artifact of record (declaration side of the id axis) |
| **INPUT** | a disputed token **id** (`--id N`, repeatable), a surface **token string** (`--token S`), or nothing at all when the question is "what is the extent/pad today" |
| **OUTPUT** | `LEGAL` / `ILLEGAL — pad region` per id, **plus the measurement path on every run**: which artifact, its size vs size-of-record, the 1 MiB head-sha (explicitly *not* a full hash), directory length, data base, tokenizer offset+length, object count, `vocab entries`, `added_tokens` count+range, `LEGAL EXTENT = max(union)+1`, `holes`, `endpoint rows`, `PAD`, and the verdict's SOURCE (artifact vs fixture) |
| **MEASUREMENT** | `tools/v340l/nvfp4/tokenizer_of_record.py` — reads the artifact's own **in-band** `frontend/tokenizer.json` through the census directory (the same raw_decode + data-base route doc 29 established), then the union `set(model.vocab.values()) ∪ {t["id"] for t in added_tokens}`. Numbers are read per run, never recalled: this beat's run printed extent **248,077**, rows **248,320**, pad **243**, holes **0**, vocab entries **248,044**, added **33** @ `[248044,248076]` |
| **IDENTITY OF RECORD** | `/media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer`, 18,324,067,840 B, sha256 `eaf8ad124256d0a0c1ebbbca442ca58eee4f97ab34a60a0b4d57e2b41e2c56d2` — the 64-hex was measured once, at the mount seat, 2 m 04 s (doc 29). **This leg re-reads 1 MiB and prints its head-sha only; it does not re-hash 18 GB and does not inherit that credit** — same declined-hash discipline as agent3's beat-4 row |
| **LIMIT (in the title's own words)** | **Declaration side only.** Whether a legal id is the *right* argmax, and every runtime bound, is **agent4's organ** — the gate's C9 `LIMIT` arm already prints that split ("census-side declaration-vs-arithmetic ONLY; runtime argmax-legality = agent4's gate"), so this title inherits no runtime claim. An id can be LEGAL here and still be wrong; an id ILLEGAL here cannot be anything but a pad artifact |
| **EXIT (law D)** | 0 answered (legal *or* illegal — both are answers) / 1 at least one disputed id is ILLEGAL (a conviction, with the path printed) / 2 **NOT-OBSERVABLE**: file absent, `model.vocab` layout moved, or **size ≠ size-of-record without `--fixture-declared`** |
| **CONSUMER** | any first-token / decode-grid dispute; the [C]/r21 graders and the counsel packet route *legality* questions here and *correctness* questions to agent4's organ |

## Why a seat has to own this at all

The pad number is the difference between **243 and 276**, and the delta is exactly the
`added_tokens` block (33 ids: 248044..248076). `model.vocab` values are **not** the whole
vocabulary — the special-token array lives in a separate field, and the runtime domain is the
**union**. An extent computed from `len(vocab)` or `max(vocab.values())` alone under-declares by 33
and reports `248320 − 248044 = 276`. That is the mint/consume law in a tokenizer costume: a
self-derived number self-reporting a total. The row exists so nobody re-derives it in prose, and the
instrument prints **both** numbers with the delta named, so the trap stays visible instead of being
re-fallen-into by the next seat that reads only one field.

Printed verbatim by the leg on every run:

```
TRAP ROW: rows - vocab_entries = 276  <- what an extent computed from model.vocab ALONE reports;
the delta 33 IS the added_tokens block. Cite the union, never len(vocab).
```

## Arming evidence (measured this beat, not asserted)

Four boundary ids, one call — the instrument's own both-directions check:

| id | answer | why it is the interesting case |
|---|---|---|
| 248043 | LEGAL | last ordinary-vocab id before the added block |
| 248044 | LEGAL | **first added_token** — the number a `len(vocab)`-only extent would call the boundary |
| 248076 | LEGAL | last added_token = `extent − 1`, the true edge |
| 248077 | **ILLEGAL — pad region** | first id that fits the tensor (248077 < 248320 rows) but decodes to NOTHING — precisely the ids the pad-region cell exists to catch |

Three states, all exercised: `--id 248076` → rc=0 (artifact); `--id 248100` → rc=1 with the pad
explanation (artifact *and* `--fixture-declared` variants, each labelling its source);
absent file → rc=2; **fixture passed without declaring it → rc=2** because the leg printed
"verdict void" and my first draft then exited 1 as if it had ruled — **a banner/code disagreement at
my own desk, caught by running my own falsifier**, same family as the two I filed this morning
(zero-evidence bar, partial-evidence cell). An undeclared non-record size is now a refusal with the
cause named, and `--fixture-declared` is the explicit opt-in, because the fixture's tokenizer payload
*is* the artifact's bytes while its offset columns are not — and this leg reads only the payload.

## Two rows I am NOT claiming

1. **Not the runtime bound.** `argmax.cu:18 kFullValidRows = 248077` and `frontend.h:15
   kTokenDomain = 248077` exist and are the runtime's own statements; the single-home question for
   that literal family is doc 32 row 8 and belongs to whoever owns the vocabulary constants, not to
   a declaration-side title. If the merge window wants the literals consolidated, this leg is the
   measurement oracle for the check, not the check.
2. **Not the artifact's byte-truth.** Size-of-record is checked; the 64-hex is cited-with-provenance,
   never re-run here (18 GB, 2+ minutes, and a cold seat should not spend it to answer an id
   question). The mount seat owes the full-hash row per re-export, exactly as §8 of the plan says.

— agent2, zero card, host-only reads (8 B magic + 206,636 B directory + 12,809,320 B tokenizer +
1 MiB head for the prefix hash), no build, no server, `/tmp` scratch reaped. Instrument:
`tools/v340l/nvfp4/tokenizer_of_record.py`.
