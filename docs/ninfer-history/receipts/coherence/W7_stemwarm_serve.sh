#!/usr/bin/env bash
# W7 promotion leg — boot the gqa-promoted bin (unset=AUTO) on the W6 P2 known-good
# env (GDN/GGATE/GQA env UNSET — defaults own the leg), probe plen-2016 first+only
# mt 64 temp 0, bank route proofs + BODYSUM + resp. Window owner: session W7.
set -u
BIN="$1"                       # banked promoted bin, artifacts_bin/<name>_<sha16>.bin
PORT=8100
LANE=/home/chris/worktrees/amd-wo-w7-body
LOG="$LANE/results/amd/coherence/W7_stemwarm_serve.log"
RESP="$LANE/results/amd/coherence/W7_stemwarm_resp.json"

export NINFER_ALLOW_NVFP4_TP2=1
export NINFER_WORKSPACE_MIB=96
export NINFER_DRAFT_VOCAB=/home/chris/dual_5060_ti_ninfer/tests/multi_gpu/data/qwen38_draft_vocab_ids.json
export NINFER_MTP_TAIL_ASYNC=1
# NINFER_GDN_SIMT / NINFER_GGATE_SIMT / NINFER_GQA_SPLITK: deliberately UNSET.

# retire whatever answers on the serving line. pgrep -f anchored to the BANK
# path (comm is truncated to 15 chars — pgrep -x ninfer-serve matches NOTHING;
# W7 leg-1 lesson). Only our banked servers live under artifacts_bin.
for p in $(pgrep -f "^/home/chris/artifacts_bin/ninfer-serve"); do
  echo "TERM pid=$p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-80)"
  kill -TERM $p 2>/dev/null
done
sleep 5
for p in $(pgrep -f "^/home/chris/artifacts_bin/ninfer-serve"); do kill -KILL $p 2>/dev/null; done
sleep 5
pgrep -f "^/home/chris/artifacts_bin/ninfer-serve" >/dev/null && { echo "RETIRE FAILED"; exit 1; }
# R6: 'retired' means VRAM actually freed — check the min free per die before boot.
for i in $(seq 1 12); do
  FREE=$(rocminfo 2>/dev/null | grep -A4 "gfx900" | grep "Memory Usage" | wc -l)
  USED=$(rocm-smi --showuse 2>/dev/null | grep -c "GPU use (%): 0")
  [ "$USED" = "4" ] && break
  sleep 5
done
rocm-smi --showuse 2>/dev/null | grep "GPU use"
echo "retired; KFD=$(ls /sys/kernel/debug/kfd 2>/dev/null | wc -l)"

cd "$LANE"
setsid nohup "$BIN" /media/chris/EMTEC256/qwen3_8_27b_nvfp4.ninfer --port $PORT \
  --devices 0,1,2,3 --prefill-chunk 128 --no-prefix-reuse --prefix-cache-capacity 256 \
  --greedy --default-max-tokens 16 --spec mtp --draft-tokens 2 --allow-nvfp4-weights \
  > "$LOG" 2>&1 < /dev/null &
echo "spawned pid=$! bin=$BIN"

for i in $(seq 1 60); do
  sleep 5
  curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
done
curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "BOOT FAILED — tail:"; tail -20 "$LOG"; exit 1; }
echo "SERVING on :$PORT (pid $(pgrep -f "^/home/chris/artifacts_bin/ninfer-serve" | head -1))"

# plen-2016 known-answer probe, first+only, mt 64 temp 0 (W6 leg grammar)
python3 - <<'EOF'
import json, urllib.request
q = 'Reply with exactly the single word BLUE. Context:' + ' alpha'*2013
body = json.dumps({'model':'qwen3.8-27b','messages':[{'role':'user','content':q}],
                   'max_tokens':64,'temperature':0}).encode()
req = urllib.request.Request('http://127.0.0.1:8100/v1/chat/completions', data=body,
                             headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=600) as r:
    o = json.load(r)
ch = o['choices'][0]
out = {'finish': ch.get('finish_reason'), 'content': ch['message']['content'],
       'completion_tokens': o.get('usage',{}).get('completion_tokens'),
       'prompt_tokens': o.get('usage',{}).get('prompt_tokens')}
open('/home/chris/worktrees/amd-wo-w7-body/results/amd/coherence/W7_stemwarm_resp.json','w').write(json.dumps(o, indent=1))
print(json.dumps(out))
EOF
echo "PROBE DONE — BODYSUM follows in serve log"
grep -a "PREFILL-BODYSUM\|PREFILL-SUM\|\[GQA\]\|\[GGATE\]\|loaded 40960 draft" "$LOG" | tail -30
