#!/usr/bin/env bash
# W1 A/B: draft backend sampling on/off, served metric + acceptance.
# usage: server_ab.sh <on|off> <rep> <port>   (boots one server, 1 warmup + 1 measured run)
set -eu
MODE=$1; REP=$2; PORT=${3:-8012}
BIN=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin/llama-server
GGUF=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
PROBE=/media/chris/ssd128/llamacpp/probe_2k.txt
RES=/media/chris/ssd128/llamacpp/llama.cpp/docs/amd-port/results/W1_server_ab.csv
FLAG="--spec-draft-backend-sampling"
[ "$MODE" = "off" ] && FLAG="--no-spec-draft-backend-sampling"
LOG=/tmp/server_ab_${MODE}_${REP}.log

export HIP_VISIBLE_DEVICES=0,1,2
"$BIN" -m "$GGUF" -ngl 999 -sm tensor -c 4096 -b 512 -ub 128 \
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp $FLAG \
  --port "$PORT" > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  if curl -s "http://127.0.0.1:${PORT}/health" | grep -q '"status":"ok"'; then break; fi
  sleep 2
done

# warmup (small)
curl -s "http://127.0.0.1:${PORT}/completion" -d "{\"prompt\":\"hello\",\"n_predict\":8,\"temperature\":0}" > /dev/null
sleep 1

# measured run: fixed ~2k-token prose prompt, greedy, 128 new tokens
python3 - "$PROBE" << 'EOF' > /tmp/ab_req.json
import json, sys
text = open(sys.argv[1]).read()
print(json.dumps({"prompt": text, "n_predict": 128, "temperature": 0, "cache_prompt": False}))
EOF
curl -s "http://127.0.0.1:${PORT}/completion" -d @/tmp/ab_req.json > /tmp/ab_resp.json

python3 - "$MODE" "$REP" "$RES" << 'EOF'
import json, sys, os
mode, rep, res = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open("/tmp/ab_resp.json"))
t = r.get("timings", {})
row = {"mode": mode, "rep": rep,
       "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
       "pp_tps": t.get("prompt_per_second"), "tg_tps": t.get("predicted_per_second"),
       "draft_n_accepted": t.get("draft_n_accepted"), "draft_n_total": t.get("draft_n_total"),
       "draft_ratio": t.get("draft_ratio"), "mean_acc_len": t.get("mean_acc_len")}
keys = list(row)
new = not os.path.exists(res)
with open(res, "a") as f:
    if new: f.write(",".join(keys) + "\n")
    f.write(",".join(str(row[k]) for k in keys) + "\n")
print(row)
EOF
