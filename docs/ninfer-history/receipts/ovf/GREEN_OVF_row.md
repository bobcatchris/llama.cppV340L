# GREEN_OVF_row — G-OVF-1 GREEN verification (kvarn >10.3k completes; parity unchanged)

- Date: 2026-09-19 21:10-21:4x local. Desk: ovf (Team Red agent #2). Window: claimed 8f95828b2.
- Bin: /home/chris/artifacts_bin/ninfer-serve_8ac1ba93eb7fbfad.bin (banked from
  build-hip-amd/apps/ninfer-serve, this tree @ 8ac1ba93e — the in-place V-conversion fix;
  sha256 prefix 8e8b7b541b28cb20e8da, 172,440,368 B).
- Boot: canonical env EXCEPT NINFER_WORKSPACE_MIB=512 (arena lever) + same kvarn flags
  (--kv-dtype kvarn_k4v4 --max-context 36352 --kv-capacity 36352 --spec mtp --draft-tokens 2
  --prefill-chunk 128 --no-prefix-reuse --greedy). Log: green_serve.log.
- Preflight (live gate, measured): required 7755 MiB / usable 8160 MiB, **slack 404 MiB**
  (fixed 7363 = w 5131 + dec 1360 + ws 512 + staging 200 + pad 160; kv 11152 B/t x 36352
  = 386 MiB). Boot clean, tail-width warmup OK, 0 worker errors all session.

## Arms (same bodies as the gates/RED legs)

| arm | body | wall | result |
|-----|------|------|--------|
| 20k needle (fire 1, mid coordinator-collision) | needle20k_req.json (19,709 tok) | 425 s | finish=stop, content="ORCHID-TUNNEL" (exact), prompt=19709 completion=84, zero mojibake |
| 2k probe (collision) | BLUE/'alpha'x2013, mt=64 | 67 s | finish=stop, "BLUE", zero mojibake |
| 9.5k needle (collision) | needle9500_req.json (9,473 tok) | 239 s | finish=stop, "ORCHID-TUNNEL" exact, zero mojibake |
| 2k probe v2 (clean window) | same | 35 s | finish=stop, "BLUE", zero mojibake |
| 9.5k needle v2 (clean window) | same | 229 s | finish=stop, "ORCHID-TUNNEL" exact, zero mojibake |
| 20k needle v2 (clean window) | same | 494 s | finish=stop, "ORCHID-TUNNEL" exact, prompt=19709 completion=84, zero mojibake |

Coordinator's G-MM-2 collided with fires 1-3 (its law-form retire hit its own arm; my serve
was never retired — PID 1031848 served the whole session, 0 tp2 worker errors). Fires 4-6
are the clean-window evidence; fires 1-3 banked as collision-armored duplication: the
behavioral gates (finish/content/mojibake) passed in BOTH conditions.

## Verdict

- G-OVF-1 MET: the exact >10.3k prompt that threw std::bad_alloc 4/4 on bin f3312f25
  (RED_OVF_row.md) now COMPLETES on 8ac1ba93eb7fbfad — prefill crosses the old wall
  (n_ctx 10,240, n_tiles 160) to 19,709 tokens, finish=stop, exact needle, zero mojibake.
- 2k parity unchanged (finish=stop, "BLUE", clean). 9.5k parity = same exact retrieval as
  the gates bin's PASS row (ORCHID-TUNNEL).
- RED->GREEN closure (law form): RED = f3312f25c8201da0 4/4 throws at n_ctx 9,984-10,240;
  GREEN = 8ac1ba93eb7fbfad 2/2 clean-window completions at 19,709 + 2/2 parity probes.
  Standing cell: needle20k_req.json replay + grade (finish=stop + needle + mojibake scan)
  joins the kvarn boot battery — every future kvarn window ends by running this cell.

## Fix + lever (for the record)

1. Code (8ac1ba93e): the HIP port's V-temp bf16->fp16 conversion is now IN-PLACE
   (same-width 2 B elementwise; bit-identical output, race-free) — removes the second
   O(committed_pages) temp per materialize call (the captured thrower:
   alloc(FP16,{256,64,1,160}) at n_ctx 10,240).
2. Env lever: NINFER_WORKSPACE_MIB=512 for kvarn long-prefill boots on this box. Basis:
   the 96 MiB value was set in the 2-rank VRAM-squeeze era; at --kv-capacity 36352 the KV
   pool is 386 MiB/rank (vs the auto-113k arm's measured 1,240 MiB/rank), i.e. ~835
   MiB/rank of measured headroom; the live preflight gate verified ws=512 with slack 404
   MiB (no estimate in the refusal path — the allocator/gate stays the arbiter).
   NOTE for the coordinator: canonical serve_10k.sh still carries 96; kvarn long-context
   legs need the 512 lever (or a per-boot rule) until the O(pages) materialize design is
   replaced (docs/120 B2 direct route = the structural cure; NOT byte-identical, so it is
   a lane decision, not this desk's).
