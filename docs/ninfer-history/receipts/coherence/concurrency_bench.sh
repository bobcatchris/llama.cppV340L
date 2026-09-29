#!/usr/bin/env bash
# concurrency_bench.sh — N=1/2/4 concurrent count600-class streams on the MTP serve line.
# Design + laws: results/amd/coherence/CONCURRENCY_BENCH.md (read before firing).
#
# LAWS EMBEDDED (do not weaken):
#   - CONC_BENCH_GRANT=1 required: operator attests the coordinator's WRITTEN grant (intercom).
#   - GPU guard: canonical gpu_guard.sh, refuse-if-busy, PID-scoped kill only, never pkill.
#   - Bank law: BIN must be /home/chris/artifacts_bin/<name>_<sha16>.bin, sha recomputed==stamp.
#   - KFD: rocm-smi --showpids raw dumps pre/post, tab-tolerant; ends only on verified KFD=0.
#   - Thermal: arms 1->2->4, >=150 s idle between arms, clocks sideband 2 s, sampler killed
#     by its OWN PID (REALK23 correction). Burst walls projected <60 s; CONC_MT=300 if void.
#
# Usage:
#   CONC_BENCH_GRANT=1 ./concurrency_bench.sh \
#     -b /home/chris/artifacts_bin/ninfer-serve_<sha16>.bin -a <artifact>.ninfer [-p 8100]
# Env knobs: CONC_MT (600), CONC_STAGGER (0.25), CONC_IDLE_S (150), CONC_BODY
#            (count|prose; default count), CONC_ALLOW_UNBANKED=1 (loud escape, bank law).
set -euo pipefail

PORT=8100; DEVS="0,1,2,3"; MT="${CONC_MT:-600}"; STAGGER="${CONC_STAGGER:-0.25}"
IDLE_S="${CONC_IDLE_S:-150}"; BODYSEL="${CONC_BODY:-count}"
BIN=""; ARTIFACT=""; RUNDIR="$(cd "$(dirname "$0")" && pwd)"
GUARD=/home/chris/dual_5060_ti_ninfer/tools/smoke/diag/gpu_guard.sh
TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "CONC-BENCH REFUSE: $*" >&2; exit 1; }

[ "${CONC_BENCH_GRANT:-0}" = "1" ] || die "set CONC_BENCH_GRANT=1 only after the coordinator's WRITTEN grant (intercom). 'Clear to claim' is not a grant."
[ -f "$GUARD" ] || die "canonical gpu_guard.sh not found at $GUARD"
# shellcheck source=/dev/null
. "$GUARD"

while getopts "b:a:p:d:" o; do
  case "$o" in
    b) BIN=$OPTARG ;; a) ARTIFACT=$OPTARG ;; p) PORT=$OPTARG ;; d) DEVS=$OPTARG ;;
    *) die "usage: -b bin -a artifact [-p port] [-d devices]" ;;
  esac
done
[ -x "$BIN" ] || die "-b <ninfer-serve bin> required"
[ -n "$ARTIFACT" ] && [ -f "$ARTIFACT" ] || die "-a <artifact.ninfer> required"

# --- Bank law ---------------------------------------------------------------
case "$BIN" in
  /home/chris/artifacts_bin/*_*)
    SHA16="$(basename "$BIN")"; SHA16="${SHA16%.bin}"; SHA16="${SHA16##*_}"
    [ ${#SHA16} -eq 16 ] || die "bin filename lacks <name>_<sha16>.bin stamp: $BIN"
    ACT="$(sha256sum "$BIN" | cut -c1-16)"
    [ "$ACT" = "$SHA16" ] || die "bank law: sha256sum $BIN = $ACT..., filename stamp $SHA16 — mismatch"
    ;;
  *) [ "${CONC_ALLOW_UNBANKED:-0}" = "1" ] || die "BIN is not a banked artifact path (bank law). CONC_ALLOW_UNBANKED=1 to override, loudly."
    SHA16="$(sha256sum "$BIN" | cut -c1-16)"
    echo "CONC-BENCH WARNING: UNBANKED binary in use, sha16=$SHA16 (stamped loud in the row)" >&2 ;;
esac

gpu_refuse_if_busy || die "foreign GPU holder present — never evict (gpu_guard law)"
echo "== df -h / (shared constraint):" >&2; df -h / >&2

# --- KFD precheck (raw, tab-tolerant reader reads the dump, not the pipe) ---
PRE="$RUNDIR/CONC_kfd_pre.txt"
rocm-smi --showpids 2>/dev/null | sed -n "/KFD/,/^====*$/p" > "$PRE" 2>/dev/null || : > "$PRE"
KFD_PRE_LINES=$(grep -c "$(printf '\t')" "$PRE" 2>/dev/null || true)
echo "== KFD precheck: $KFD_PRE_LINES tab lines (raw: $PRE)" >&2
[ "$KFD_PRE_LINES" = "0" ] || die "foreign KFD compute context present pre-boot ($KFD_PRE_LINES lines, raw $PRE) — guard on contexts, not ports; do NOT evict, wait for the window"

# --- Serve line: CURRENT recommended line + --max-concurrency 4 (N7 flag) ---
# GPU-owner fix 2026-09-17 (fired window): the old line passed `--allow-nvfp4-weights` as an
# env(1) ARGUMENT (`env --allow-nvfp4-weights NINFER_WORKSPACE_MIB=96 ...`) — env exits RC=125
# "unrecognized option" and the boot dies instantly. Flag moved to the CLI args; the mandated
# NINFER_ALLOW_NVFP4_TP2=1 + NINFER_DRAFT_VOCAB env vars added (HOLE1 CWD-trap mandate).
# mc=4 KV REALITY (measured 2026-09-17, MELTDOWN boot attempts): default (auto) KV at mc=4
# REFUSES at preflight on 8 GB dies (fixed 8483 MiB incl the 1536 MiB bf16 mc>=2 reserve >
# 8160 free); forcing --kv-dtype int8 routes KV planning to I8_KV which is DEFERRED on the HIP
# lane -> half-initialized cache, every request fails instantly. The workable mc=4 posture is
# EXPLICIT --kv-dtype bf16 with an explicit --kv-capacity (owner-verified 2026-09-17). Until
# that line is standard here, CONC_MC may be lowered to 2.
SLOG="$RUNDIR/CONC_serve.log"; : > "$SLOG"
gpu_spawn SERVE_PID env NINFER_ALLOW_NVFP4_TP2=1 \
  NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/multi_gpu/data/qwen38_draft_vocab_ids.json \
  NINFER_WORKSPACE_MIB=96 \
  "$BIN" "$ARTIFACT" --port "$PORT" --devices "$DEVS" --prefill-chunk 128 \
  --no-prefix-reuse --prefix-cache-capacity 256 --greedy --default-max-tokens 16 \
  --spec mtp --draft-tokens 2 --allow-nvfp4-weights --max-concurrency 4 >> "$SLOG" 2>&1
echo "== serve pid $SERVE_PID (bin sha16 $SHA16), waiting for /health" >&2
for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -30 "$SLOG" >&2; die "serve died during boot"; }
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || die "no /health after 180 s"
grep -m1 "\[tp2\] auto KV" "$SLOG" || true
grep -m1 "\[preflight\] composition" "$SLOG" || true
grep -m1 "draft vocabulary IDs" "$SLOG" || true
grep -m1 "draft output head ready" "$SLOG" || true

# --- Body (count600 class; CONC_BODY=prose -> REALK sky-blue shape, mt honored) ---
if [ "$BODYSEL" = "prose" ]; then
  PROMPT="Why is the sky blue?"   # REALK23-class prose body (real-prose acceptance shape)
else
  PROMPT="Count from 1 to 200, one number per line, no commentary."
fi
BODY_JSON="$(printf '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"temperature":0}' "$PROMPT" "$MT")"

# --- Clocks sideband: 2 s sysfs sampler; killed by ITS OWN PID (REALK23 fix) ---
CARDS=""
for c in /sys/class/drm/card*/device; do
  [ "$(cat "$c/vendor" 2>/dev/null)" = "0x1002" ] && [ -f "$c/pp_dpm_sclk" ] && \
    [ -f "$c/unique_id" ] && CARDS="$CARDS $(basename "$(dirname "$c")")"
done
[ -n "$CARDS" ] || die "no AMD sclk-capable cards found for the sideband"
start_sampler() { # $1 outfile -> sets SAMPLER_PID
  ( while :; do
      line="$(date +%H:%M:%S)"
      for card in $CARDS; do
        d="/sys/class/drm/$card/device"
        sclk="$(grep '\*' "$d/pp_dpm_sclk" 2>/dev/null | head -1 | awk '{print $2}')"
        edge="$(for h in "$d"/hwmon/hwmon*; do cat "$h/temp1_input" 2>/dev/null; done | head -1)"
        line="$line $card=${sclk:-None}MHz,${edge:-None}mC"
      done
      echo "$line" >> "$1"; sleep 2
    done ) &
  SAMPLER_PID=$!
}
stop_sampler() { [ -n "${SAMPLER_PID:-}" ] && { kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""; }; }
trap stop_sampler EXIT

# --- Arms: 1 -> 2 -> 4, coldest first, >=IDLE_S between arms -----------------
for N in 1 2 4; do
  echo "== idle ${IDLE_S}s before arm N=$N (thermal integrator)" >&2
  sleep "$IDLE_S"
  CLK="$RUNDIR/CONC_N${N}_clocks.tsv"; : > "$CLK"; start_sampler "$CLK"
  T0="$(date +%s.%N)"; PIDS=""
  for i in $(seq 1 "$N"); do
    curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' -d "$BODY_JSON" > "$RUNDIR/CONC_N${N}_resp_${i}.json" &
    PIDS="$PIDS $!"; sleep "$STAGGER"
  done
  FAIL=0; for p in $PIDS; do wait "$p" || FAIL=1; done
  T1="$(date +%s.%N)"
  stop_sampler
  WALL="$(awk -v a="$T0" -v b="$T1" 'BEGIN{printf "%.1f", b-a}')"
  TOT=0
  for i in $(seq 1 "$N"); do
    c="$(grep -o '"completion_tokens":[0-9]*' "$RUNDIR/CONC_N${N}_resp_${i}.json" 2>/dev/null | head -1 | grep -o '[0-9]*$')"
    TOT=$((TOT + ${c:-0}))
  done
  AGG="$(awk -v t="$TOT" -v w="$WALL" 'BEGIN{printf "%.2f", (w>0)? t/w : 0}')"
  echo "== ARM N=$N: wall ${WALL}s; total_completion=$TOT; aggregate=${AGG} tok/s; curl_failures=$FAIL" \
    > "$RUNDIR/CONC_N${N}_row.txt"
  { echo "-- dispatched line:";   grep "batched decode: dispatched" "$SLOG" | tail -"$N" || echo "(none — lanes ran single-seq; a SPLIT is a finding, record it)";
    echo "-- per-lane lines:";    grep -E "batched lane|single-seq lane" "$SLOG" | tail -"$N" || echo "(none)";
    echo "-- done lines:";        grep "done finish=" "$SLOG" | tail -"$N" || echo "(none)";
    echo "-- clocks sideband: $CLK ($(wc -l < "$CLK") samples; sclk collapse 1500-><800 MHz = throttle, arm VOID)";
  } >> "$RUNDIR/CONC_N${N}_row.txt"
  echo "== ARM N=$N banked -> CONC_N${N}_row.txt" >&2
done

# --- Teardown: PID-scoped only, then verified KFD=0 --------------------------
gpu_kill_own "$SERVE_PID" || true
sleep 3
kill -0 "$SERVE_PID" 2>/dev/null && die "serve survived gpu_kill_own — investigate by PID, do NOT pkill"
POST="$RUNDIR/CONC_kfd_post.txt"
rocm-smi --showpids 2>/dev/null | sed -n "/KFD/,/^====*$/p" > "$POST" 2>/dev/null || : > "$POST"
KFD_POST_LINES=$(grep -c "$(printf '\t')" "$POST" 2>/dev/null || true)
[ "$KFD_POST_LINES" = "0" ] || die "KFD postcheck nonzero ($KFD_POST_LINES tab lines, raw $POST) — window not clean"
stop_sampler
echo "== RELEASE OK: serve killed anchored, sampler dead, KFD=0 ($POST)" >&2
echo "== NEXT: fill the manifest template (CONCURRENCY_BENCH.md §6) from the CONC_N*_row.txt files and bank the pair." >&2
