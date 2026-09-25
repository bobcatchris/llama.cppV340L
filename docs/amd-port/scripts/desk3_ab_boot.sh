#!/usr/bin/env bash
# desk3_ab_boot.sh - one served A/B boot for the TP3 V340L desk.
# usage: desk3_ab_boot.sh <candidate> <arm> <rep> [ENV=VAL ...]
# Boots via run_tp3_guards.sh (fresh boot + full guard battery with 180s
# cold-stamp idle), records per-boot GPU temps, verifies the arm signature
# in the server log, checks for GGML_ASSERT lines, appends a manifest row.
set -uo pipefail
REPO=/media/chris/ssd128/llamacpp/llama.cpp
TESTS=$REPO/docs/amd-port/tests
RES=$REPO/docs/amd-port/results
CAND=$1; ARM=$2; REP=$3; shift 3
LABEL=c${CAND}_${ARM}_rep${REP}
MANIFEST=$RES/desk3_ab_manifest.jsonl

# pre-boot thermal window: wait for edge temps to settle (<= 35C, max 300s)
PRE_TMP=/tmp/desk3_${LABEL}_pre.json
for i in $(seq 1 30); do
  rocm-smi --showtemp --json > "$PRE_TMP" 2>/dev/null
  MAXEDGE=$(python3 - "$PRE_TMP" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    vals = [float(v.get("Temperature (Sensor edge) (C)", 0)) for v in d.values() if isinstance(v, dict)]
    print(int(max(vals)) if vals else 0)
except Exception:
    print(0)
EOF
)
  [ "$MAXEDGE" -le 35 ] && break
  [ $i -eq 1 ] && echo "[$LABEL] pre-boot cooldown: waiting for edge <= 35C (now ${MAXEDGE}C)"
  sleep 10
done
echo "[$LABEL] pre-boot temps: edge max ${MAXEDGE}C"

# launch the guarded boot + battery with the arm env, and sample the server
# process environ in parallel (ggml INFO lines are verbosity-filtered from the
# server log, so env delivery is proven via /proc instead)
cd "$REPO"
ENVF=/tmp/desk3_${LABEL}_env.txt
( while kill -0 $$ 2>/dev/null; do
    PID=$(pgrep -f 'build-hip/bin/llama-server' | head -1)
    if [ -n "$PID" ]; then
      sleep 2  # let the final process settle
      tr '\0' '\n' < "/proc/$PID/environ" 2>/dev/null | grep '^GGML_CUDA' > "$ENVF" || true
      break
    fi
    sleep 2
  done ) &
ENVSampler=$!

RUNLOG=$RES/desk3_${LABEL}_$(date +%Y%m%d_%H%M%S).log
if [ "$#" -gt 0 ]; then
  env "$@" bash "$TESTS/run_tp3_guards.sh" > "$RUNLOG" 2>&1
else
  bash "$TESTS/run_tp3_guards.sh" > "$RUNLOG" 2>&1
fi
EXIT=$?
kill "$ENVSampler" 2>/dev/null
ENVSEEN=$(tr '\n' ';' < "$ENVF" 2>/dev/null || echo "")

# locate this boot's receipt and server log
RECEIPT=$(grep -oE 'Receipt banked at: .*' "$RUNLOG" | awk '{print $NF}' | tail -1)
STAMP=$(basename "${RECEIPT:-x}" .jsonl | sed 's/^tp3_guards_//')
SLOG=$(grep -oE 'server_tp3_200k_[0-9_]+\.log' "$RUNLOG" | head -1)
SLOG=$RES/${SLOG:-missing}
VERDICT=$(grep -oE 'OVERALL VERDICT: .*' "$RUNLOG" | tail -1 | awk '{print $3}')
GUARDEXIT=$EXIT

# arm signature: env delivery via /proc + ggml WARN scan in server log
EXPECT=""
case "$CAND:$ARM" in
  1:on)  EXPECT="GGML_CUDA_Q81_ACT_CACHE=1" ;;
  2:on)  EXPECT="GGML_CUDA_TILE_FP16=1" ;;
  3:on)  EXPECT="GGML_CUDA_MMVQ_GROUP=1" ;;
esac
if [ -z "$EXPECT" ]; then
  [ -z "$ENVSEEN" ] && SIG=clean || SIG=unexpected_env
elif [ "$ENVSEEN" = "${EXPECT};" ] || [ "$ENVSEEN" = "$EXPECT" ]; then
  SIG=delivered
else
  SIG=NOT_DELIVERED
fi
ASSERTS=$(grep -c "GGML_ASSERT" "$SLOG" 2>/dev/null); ASSERTS=${ASSERTS:-0}
POOL0=$(grep -c "GGML_ASSERT(pool_size == 0)" "$SLOG" 2>/dev/null); POOL0=${POOL0:-0}
TILEWARN=$(grep -c "shape off the census whitelist" "$SLOG" 2>/dev/null); TILEWARN=${TILEWARN:-0}

# post-shutdown temps
POST_TMP=/tmp/desk3_${LABEL}_post.json
rocm-smi --showtemp --json > "$POST_TMP" 2>/dev/null
POSTEDGE=$(python3 - "$POST_TMP" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    vals = [float(v.get("Temperature (Sensor edge) (C)", 0)) for v in d.values() if isinstance(v, dict)]
    print(int(max(vals)) if vals else 0)
except Exception:
    print(0)
EOF
)

# receipt summary metrics (decode tps, prefill tps, accept, det sha)
METRICS=$(python3 - "$RECEIPT" <<'EOF'
import json, sys, os
p = sys.argv[1]
try:
    rows = [json.loads(l) for l in open(p) if l.strip()]
    r = rows[-1]
    dec = pre = acc = sha = "na"
    for g in r.get("results", []):
        if g.get("guard") == "decode_guard": dec = g.get("measured_tps")
        if g.get("guard") == "prefill_guard": pre = g.get("measured_tps")
        if g.get("guard") == "mtp_canary": acc = g.get("accept_ratio")
        if g.get("guard") == "determinism_guard": sha = g.get("sha256_1")
    ndl = next((g.get("recalled_count") for g in r.get("results", []) if g.get("guard") == "needle_recall_guard"), "na")
    print(f'"{dec}","{pre}","{acc}","{ndl}","{sha}"')
except Exception as e:
    print('"err","err","err","err","err"')
EOF
)
IFS=, read DEC PRE ACC NDL SHA <<< "$METRICS"

echo "[$LABEL] exit=$GUARDEXIT verdict=${VERDICT:-na} dec=$DEC pre=$PRE acc=$ACC needle=$NDL sha=$SHA sig=$SIG env_seen=${ENVSEEN:-none} tilewarn=$TILEWARN asserts=$ASSERTS pool0=$POOL0 temps=$MAXEDGE/$POSTEDGE C"
ROW=$(python3 - "$CAND" "$ARM" "$REP" "$LABEL" "$GUARDEXIT" "${VERDICT:-na}" "$DEC" "$PRE" "$ACC" "$NDL" "$SHA" "$SIG" "$ENVSEEN" "$TILEWARN" "$ASSERTS" "$POOL0" "$MAXEDGE" "$POSTEDGE" "$RECEIPT" "$SLOG" "$RUNLOG" <<'EOF'
import json, sys
keys = ["candidate","arm","rep","label","battery_exit","verdict","decode_tps","prefill_tps","accept","needle","det_sha","sig","env_seen","tile_warn","assert_lines","pool0_asserts","pre_edge_max_c","post_edge_max_c","receipt","server_log","run_log"]
row = dict(zip(keys, sys.argv[1:]))
for k in ("battery_exit","tile_warn","assert_lines","pool0_asserts","pre_edge_max_c","post_edge_max_c","rep","candidate"):
    row[k] = int(row[k])
print(json.dumps(row))
EOF
)
echo "$ROW" >> "$MANIFEST"
echo "[$LABEL] manifest row banked"
exit $GUARDEXIT
