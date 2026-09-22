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

# launch the guarded boot + battery with the arm env
cd "$REPO"
RUNLOG=$RES/desk3_${LABEL}_$(date +%Y%m%d_%H%M%S).log
if [ "$#" -gt 0 ]; then
  env "$@" bash "$TESTS/run_tp3_guards.sh" > "$RUNLOG" 2>&1
else
  bash "$TESTS/run_tp3_guards.sh" > "$RUNLOG" 2>&1
fi
EXIT=$?

# locate this boot's receipt and server log
RECEIPT=$(grep -oE 'Receipt banked at: .*' "$RUNLOG" | awk '{print $NF}' | tail -1)
STAMP=$(basename "${RECEIPT:-x}" .jsonl | sed 's/^tp3_guards_//')
SLOG=$(grep -oE 'server_tp3_200k_[0-9_]+\.log' "$RUNLOG" | head -1)
SLOG=$RES/${SLOG:-missing}
VERDICT=$(grep -oE 'OVERALL VERDICT: .*' "$RUNLOG" | tail -1 | awk '{print $3}')
GUARDEXIT=$EXIT

# arm signature + assert scan
SIG=absent
case "$CAND:$ARM" in
  1:on)  grep -q "q8_1 activation cache enabled" "$SLOG" 2>/dev/null && SIG=present ;;
  2:on)  grep -q "GGML_CUDA_TILE_FP16=1, packed-fp16 tile GEMM route enabled" "$SLOG" 2>/dev/null && SIG=present ;;
  3:on)  grep -q "grouped mmvq decode path enabled" "$SLOG" 2>/dev/null && SIG=present ;;
  *)     grep -q "GGML_CUDA_.*enabled" "$SLOG" 2>/dev/null || SIG=clean ;;
esac
ASSERTS=$(grep -c "GGML_ASSERT" "$SLOG" 2>/dev/null || echo 0)
POOL0=$(grep -c "GGML_ASSERT(pool_size == 0)" "$SLOG" 2>/dev/null || echo 0)

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

echo "[$LABEL] exit=$GUARDEXIT verdict=${VERDICT:-na} dec=$DEC pre=$PRE acc=$ACC needle=$NDL sha=$SHA sig=$SIG asserts=$ASSERTS pool0=$POOL0 temps=$MAXEDGE/$POSTEDGE C"
ROW=$(python3 - "$CAND" "$ARM" "$REP" "$LABEL" "$GUARDEXIT" "${VERDICT:-na}" "$DEC" "$PRE" "$ACC" "$NDL" "$SHA" "$SIG" "$ASSERTS" "$POOL0" "$MAXEDGE" "$POSTEDGE" "$RECEIPT" "$SLOG" "$RUNLOG" <<'EOF'
import json, sys
keys = ["candidate","arm","rep","label","battery_exit","verdict","decode_tps","prefill_tps","accept","needle","det_sha","sig","assert_lines","pool0_asserts","pre_edge_max_c","post_edge_max_c","receipt","server_log","run_log"]
row = dict(zip(keys, sys.argv[1:]))
for k in ("battery_exit","assert_lines","pool0_asserts","pre_edge_max_c","post_edge_max_c","rep","candidate"):
    row[k] = int(row[k])
print(json.dumps(row))
EOF
)
echo "$ROW" >> "$MANIFEST"
echo "[$LABEL] manifest row banked"
exit $GUARDEXIT
