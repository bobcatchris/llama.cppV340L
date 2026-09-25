# M1 — SPACED CENSUS PROTOCOL (agent3; grant: chair `e468ab9e` reply, window battery LEG 1.5)

**Why this leg exists, in the manifest's own words (chair asked for this sentence):** in every
back-to-back census on disk, **request ordinal, elapsed wall time and rank1's clock are mutually
monotone** (corr(rank1 sclk, ordinal) = -0.925; corr(retries, ordinal) = +0.947 over 12 windows at
G18e). No re-reading of existing logs can separate *"after N requests"* (cumulative in-server state:
arena / KV / slot history — mine) from *"after T seconds"* (machine state: DVFS/thermal). **Sleeping between reps is what breaks the degeneracy**, which is the entire reason a grant-holder
spends 50 extra seconds on the pair (25 reps x 2 s; G18e's busy wall was 361 s, so M1 lands near
6.9 min against G18e's ~6.0 — inside a normal pair grant). It buys a causal answer with one line in
the runner.

## 1. Boot — one delta from G-AMD-34, and that delta is SLEEP, not code

Reuse the G-AMD-34 runner (`results/amd/p3/G18e_traceoff_census.sh` in agent4's lane) unchanged
except the marked lines. Bin, arms, mask, workload are **byte-equal to G18e** so the comparison is
legitimate:

    python3 results/amd/p3/residence_probe.py "$(cat results/amd/p3/Q3_CANONICAL_PATH)"   # RESIDENCE BEFORE SPAWN
    # re-warm if the probe is under ~90%: dd if=/media/chris/EMTEC256/qwen3_8_27b_q3.ninfer of=/dev/null bs=8M
    # >>> GRANT-ACK env-in-arguments per kit v7; manifest row BEFORE spawn <<<

    HIP_VISIBLE_DEVICES=2,3 NINFER_WORKSPACE_MIB=96 GRANT_ACK=G-AMD-M1 \
      NINFER_MB_ARGMAX_TRACE=1 NINFER_AR_PARITY=1 NINFER_MC31=1 \
      setsid nohup /home/chris/artifacts_bin/ninfer-serve_2a345b3048c1d6b3.bin \
      "$(cat results/amd/p3/Q3_CANONICAL_PATH)" --port 8093 --devices 0,1 \
      --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 --no-cuda-graph --greedy \
      > results/amd/p3/G19spaced_serve.log 2>&1 < /dev/null &
    # NINFER_GATING_TRACE deliberately ABSENT — trace-OFF parity with G18e, the arm that priced the dial.

    # WITNESS (mine, runs on the chair's/holder's shell, NOT the server's — zero device context):
    python3 results/amd/sysfs_witness_sampler.py results/amd/p3/G19spaced_witness.tsv \
        --secs 900 --ival 2 --cards card0,card4 --label G19spaced &

    for i in $(seq 1 25); do
      curl -s localhost:8093/v1/chat/completions -H 'content-type: application/json' \
        -d '{"model":"q","messages":[{"role":"user","content":"Hi."}],"temperature":0,"max_tokens":32}' \
        > results/amd/p3/G19spaced_resp_$i.json
      sleep 2                                   # <<< THE ONE DELTA of this leg
    done

Bank: `text_sha16` per rep into `G19spaced_repeat_summary.txt` (TEXT-sha only — whole-JSON is the
vacuous instrument, `1d0ff3c6`), plus the counts file and the release row. Kill own pgid only; KFD
recheck; release row includes deaths as datums.

**Coverage note, and it is a fix for my own G18e limitation:** start the sampler **before spawn** and
keep it running through the post-run cooldown. At G18e my sampler came up at request 13, so it could
describe the tail but never see the onset — the single biggest hole in my own read.

## 2. Pre-declared readings — decided NOW, before any number exists

Primary predicate: `retries per request` and `corr(retries, request ordinal)` from
`tools/census/census_read_analysis.py <log> <witness.tsv>`. n = 25 requests, ~1,600 argmax calls.

| # | observed shape | verdict | consequence |
|---|---|---|---|
| **B1** | retries collapse (total ≤ ~80, i.e. ≤ 10 % of G18e's 831; median stamp no longer late-loaded) | **MACHINE STATE** — duty-coupled; but see B7 for *which* machine state, since thermal and power-cap are now BOTH excluded | publish-degradation hunt **closes with no product patch**; every rate number off a long back-to-back census gets labelled duty-coupled (agent2's rule becomes law); no arena/KV re-seat |
| **B2** | retries persist (total ≥ ~600, i.e. ≥ 72 % of G18e's 831; ordinal-corr ≥ +0.8, one-sided rank0 survives) | **CUMULATIVE IN-SERVER STATE** | agent3 re-seats hypothesis-first on arena/KV/slot history; **M2 is the promoted arm** (see B7) and is the only thing that separates thread-architecture starvation from a per-die difference; **no patch before the mechanism is named** |
| **B3** | in between | **BOTH contribute** | report the split as two named fractions with the fit, do NOT average them into one story |
| **B4** | FALSIFIER OF MY OWN READ (fires either way): rank1's sclk does **not** sit ~15 % under rank0's inside a degraded tail — or it drops **while** retries vanish | my G18e clock↔retries correlation was **two monotone series coinciding** | I say so in the row that carries it and the clock story is void, not amended |
| **B5** | any `[AR-FAILOUT]`, or `[ARTAG]`, or a KAR `REJECT-candidate` in the served window, or a LONE MC31-H step | **new datum, not this cell's branch** | bank it, name it, stop; it is a carrier-class event and outranks the rate question |
| **B6** | sampler yields < 20 aligned windows, or prints `ALIGN-ERROR`/`SAMPLER-ERROR` | **INSTRUMENT-ERROR, not a result** | B1/B2/B7 must NOT be read from it; re-run with the witness fixed. An unarmed leg never reads clean (agent5's rule, adopted) |
| **B7** | **CLOCK CHANNEL — added on agent2's cross-link; read it BESIDE B1/B2. Different channel, and they CAN DISAGREE.** rank1's sclk across the spaced reps. Basis: thermal and power-cap both excluded on the G18e witness, verified at my bytes (13/15 rows show EQUAL edge temp with the dies **186 MHz** apart; rank1 peaks **81 W against a 110 W cap** and `corr(sclk,pwr) = +0.635`, i.e. clock and power fall TOGETHER — a governor giving cycles, not a limit clawing them back) | three sub-readings | **recovers between reps** ⇒ duty-accumulation starvation (machine state, physics flavor); **holds depressed ~15 % with retries gone** ⇒ the gap is NOT retry-driven, so the thread-architecture candidate (rank1's worker born per request, `:3325`) needs **M2** to separate it from a plain per-die difference; **B1/B2 and B7 disagree** ⇒ the most informative outcome this leg can produce, reported AS a disagreement, never averaged |

### 2a. ERA RULE for M1's kind field — measured from the binary, because git ancestry LIED about it

The bin is `d3a0e738ec09019e` (A-4's artifact, `A4_ARTIFACT_RECORD_agent4.txt`). `git merge-base
--is-ancestor 6b98623f d3a0e738` returns **false**, which reads as "this bin predates A-4's `kind=ar`"
— and it is the wrong test: the bin's tree is agent4's **lane tip**, which is not an ancestor
relationship to main's merge commit. The decisive measurement is the binary's own format strings:

    $ strings -a /home/chris/artifacts_bin/ninfer-serve_d3a0e738ec09019e.bin | grep AR-RETRY
      [AR-RETRY] rank=%d kind=ar gen=%llu slot=%zu try=%d status=%d
      [AR-RETRY] rank=%d kind=argmax stamp=%llu slot=%zu try=%d status=%d

**Both emitters name their kind in M1's bin.** Consequences for how M1's log is read, and they differ
from the five banked pre-A-4 bins:

- `kind-less AR-RETRY > 0` in M1 is **a finding about the EMITTER SET** (an unknown emitter printed a
  retry), NOT "the AR ring retried, unlabelled" as it means on G18c/d/e. Read with
  `CENSUS_STRICT_KINDS=1` (agent5's mode) so a non-zero count refuses loudly instead of bucketing —
  the chair's "must FAIL rather than vanish" clause is mechanical now, which is why my table does not
  have to remember the era; the tool carries it.
- **AR-ring retries are now directly countable as `kind=ar` for the first time.** On the pre-A-4 bins
  the AR ring printed zero retries and the class-blindness cost nothing, so tonight's "0 AR retries"
  was rescued by totality rather than by evidence of distinction. M1 gets the real thing: if AR-ring
  retries appear at spaced-census geometry where they were absent back-to-back, that is a NEW datum
  about the wait structure, and it is B5-class (a new channel firing) not a B1/B2 reading.
- My `stamp=`/`gen=` legs must be read per kind, which agent5's reader already does and names.

**And the same discipline applies to my own number from §4 of the read row:** the 72-79 ms price was
measured on PRE-A-4 bins, whose retry legs were `stamp=`-only for argmax. The price is a property of
the bounded poll, which A-4 did not change, so it carries — but M1's row must state the era of the
bin it is compared against, not assume continuity of format.

Also recorded either way, no branch attached: 25/25 text-sha (correctness is *not* this cell's
question), `rb` line counts per rank (presence-witness only — **`rb n=` is inadmissible as branch
evidence**, ruled `60713b0b` + my prediction test), `MC31`/`MC31-H` arming counts, kind-less retry
count via agent5's reader.

## 3. What M1 does NOT settle (so nobody over-reads one leg)

- **Not the cause of the clock asymmetry**, only whether the *lag* is duty/time-coupled or
  request-cumulative. M2 (solo die, world=1, no pairing in the loop) is the arm for that.
- **Not K=16.** The chair's re-open trigger (`max try ≥ 12`, or any death at K=16, or world=4's fresh
  distribution) stands; M1's tail is expected to stay ≤ 8, and if it *rises* under spacing that is
  itself a B2-flavoured datum.
- **Not the absorb price** — 72-79 ms/retry drift-controlled, 87-96 ms uncontrolled (three boots, two dial states),
  not a rate claim; M1 can only say how often that price gets paid.
- **Not any baseline.** A throttled or spaced measurement is a description of that arm, never a
  platform rate.

## 4. World=4 cousin — my call, named for the chair

**Pre-register it, as a print-cadence-only variant, and let it inherit these branches unchanged.**
Reason: at world=4 both the retry distribution *and* the barrier capacity are new
(`std::barrier(2)` at three sites is pair-hardcoded), so a spaced 4-rank census is the cheapest place
to learn whether B1/B2 is a property of the box or of the pairing — but it must not be *designed*
before the window exists, and it must not be merged into agent5's try-histogram row, because that row
answers "is K=16 the right bound" (a **bound** question) while M1 answers "what causes the lag" (a
**mechanism** question) — different predicates, different failure shapes, one shared argv skeleton.
Concretely: same protocol, `--devices 0,1,2,3`, `--cards` widened to all four drm nodes, 4-rank
witness, and **B5 gains one arm**: any rank-asymmetric step count at a reset boundary is a
carrier-class event at world=4 by definition. If the chair wants it not to be a separate leg, the
right merge target is the window's own boot battery, not agent5's row.

## 5. Lineage

Bin `2a345b3048c1d6b3` (item-7 census bin, `db09c924` chain) · dial arm `d752eb54`/G18e · baseline
G18d trace-ON (`db09c924`) · my read + corrections `c2145b99` + `e468ab9e` · witness instrument
`results/amd/sysfs_witness_sampler.py` · analysis `tools/census/census_read_analysis.py` (rc contract
cellified: `tests/check_census_tool_exit_contract.py`, RED 3-arms on c2145b99, GREEN all-arms here).

— agent3 (pi `01a09cdc-ceb0`), 2026-09-14 ~00:2xZ. Protocol only; no boot, no build, no card claimed.
