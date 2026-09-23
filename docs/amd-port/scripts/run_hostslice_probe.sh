#!/usr/bin/env bash
# HOST-SLICE DESK P0: one TP4 decode probe with launch-timeline instrumentation.
# Boots the canonical stack on lane 8083 with LLAMA_LAUNCH_TIMELINE=1 +
# LLAMA_DECODE_TIMELINE=1 + LLAMA_SPEC_TIMELINE=1 at -lv 4, runs ONE decode
# probe (4k prompt, 128 tokens, temp 0), tears down by PID, releases the lock.
# Laws: boot lock check-and-hold (JSON line desk/arm/ts/pid), single sleeps
# up to 150 s, cool-die gate < 60 C, PID-targeted teardown, prompt release.
set -u
BIN=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin/llama-server
MODEL=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
RES=/media/chris/ssd128/llamacpp/wt-host-slice/docs/amd-port/results
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RES/W12_hostslice_p0_server_$STAMP.log"
STAMPF="$RES/W12_hostslice_p0_stamp_$STAMP.txt"
PROBEJSON="$RES/W12_hostslice_p0_probe_$STAMP.json"
PROMPTF=/tmp/hostslice_prompt_4k.txt
LOCK=/tmp/campaign_gpu_boot.lock
PORT=8083

BASEENV="GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_MMVQ_IQ3S_SHARE=1 GGML_CUDA_MMVQ_IQ3XXS_SHARE=1 GGML_CUDA_MMVQ_Q3K_SHARE=1 GGML_CUDA_MMVQ_Q4K_SHARE=1 GGML_CUDA_MMVQ_Q5K_SHARE=1 GGML_CUDA_MMVQ_Q6K_SHARE=1 LLAMA_DRAFT_FAST_TOPK=1 LLAMA_DRAFT_PACKED_GET=1 LLAMA_DRAFT_LIGHT_SYNC=1 LLAMA_VERIFY_ROW_SAMPLING=1 LLAMA_ASYNC_INPUT=1 GGML_PINNED_DEV_COPY=1"
TLENV="LLAMA_LAUNCH_TIMELINE=1 LLAMA_DECODE_TIMELINE=1 LLAMA_SPEC_TIMELINE=1"

die_hot() {  # prints max die junction temp (C)
  local c h t m=0
  for c in card0 card1 card3 card4; do
    h=$(for hw in /sys/class/hwmon/hwmon*; do [ "$(cat $hw/name 2>/dev/null)" = amdgpu ] && [ "$(basename $(readlink -f $hw/device))" = "$(basename $(readlink -f /sys/class/drm/$c/device))" ] && echo $hw && break; done)
    t=$(($(cat $h/temp2_input 2>/dev/null || echo 0)/1000)); [ "$t" -gt "$m" ] && m=$t
  done
  echo $m
}
die_idle() {  # all four dies < 200 MiB vram used
  local c v
  for c in card0 card1 card3 card4; do
    v=$(cat /sys/class/drm/$c/device/mem_info_vram_used 2>/dev/null || echo 999999999)
    [ "$v" -ge 200000000 ] && return 1
  done
  return 0
}

echo "== HOST-SLICE P0 probe $STAMP =="
# --- lock check-and-hold ---
WAITED=0
while [ -f "$LOCK" ]; do
  HOLDER=$(cat "$LOCK" 2>/dev/null)
  HPID=$(echo "$HOLDER" | sed -n 's/.*"pid":\([0-9]*\).*/\1/p')
  if [ -n "$HPID" ] && ! kill -0 "$HPID" 2>/dev/null; then
    echo "LOCK: stale holder pid $HPID dead, removing with evidence: $HOLDER"
    rm -f "$LOCK"
    break
  fi
  echo "LOCK: held, waiting ($WAITED s): $HOLDER"
  sleep 150
  WAITED=$((WAITED+150))
done
while ! die_idle; do echo "GATE: dies busy, waiting"; sleep 60; done
while [ "$(die_hot)" -ge 60 ]; do echo "GATE: dies hot ($(die_hot) C), waiting"; sleep 30; done
printf '{"desk":"host-slice","arm":"P0-launch-timeline","ts":"%s","pid":%d}\n' \
  "$(date +%s)" "$$" > "$LOCK"
echo "LOCK: held by pid $$"

cleanup() {
  if [ -n "${SRV:-}" ] && kill -0 "$SRV" 2>/dev/null; then
    echo "TEARDOWN: killing server pid $SRV"
    kill "$SRV" 2>/dev/null
    for i in $(seq 1 30); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
  fi
  [ -f "$LOCK" ] && grep -q '"desk":"host-slice"' "$LOCK" 2>/dev/null && rm -f "$LOCK" && echo "LOCK: released"
}
trap cleanup EXIT

# --- stamp + identity ---
{
  echo "== HOST-SLICE P0 stamp $STAMP =="
  echo "desk=host-slice arm=P0-launch-timeline lane=$PORT"
  echo "bin=$BIN"
  echo "bin_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "base_commit=$(git -C /media/chris/ssd128/llamacpp/wt-host-slice log --oneline -1)"
  echo "BASEENV: $BASEENV"
  echo "TLENV:   $TLENV"
  echo "cmdline: llama-server -m $MODEL -ngl 999 -sm tensor -c 32768 --batch-size 512 --ubatch-size 512 -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp --device ROCm0,ROCm1,ROCm2,ROCm3 --port $PORT -t 8 -lv 4"
  echo "server_log=$LOG"
  echo "pre_temps: max_die=$(die_hot) C  $(date '+%F %T')"
} > "$STAMPF"

# --- boot ---
env HIP_VISIBLE_DEVICES=0,1,2,3 $BASEENV $TLENV "$BIN" -m "$MODEL" \
  -ngl 999 -sm tensor -c 32768 --batch-size 512 --ubatch-size 512 \
  -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
  --device ROCm0,ROCm1,ROCm2,ROCm3 --port $PORT -t 8 -lv 4 > "$LOG" 2>&1 &
SRV=$!
echo "BOOT: server pid $SRV, log $LOG"

OK=0
for i in $(seq 1 50); do
  sleep 15
  curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { OK=1; break; }
  kill -0 "$SRV" 2>/dev/null || break
  echo "BOOT: waiting ($(($(i)*15)) s)"
done
if [ "$OK" != 1 ]; then
  echo "BOOT FAILED - see $LOG"; tail -30 "$LOG"
  exit 1
fi
echo "BOOT: ready after $((SECONDS)) s"
echo "boot_ready_s=$((SECONDS))" >> "$STAMPF"

# --- 4k prompt (deterministic) ---
python3 - "$PROMPTF" <<'PYEOF'
import sys, random
rng = random.Random(20260923)
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform victor "
         "whiskey xray yankee zulu data cache token stream kernel graph device "
         "buffer sample decode verify draft accept batch issue drain slice").split()
out = []
n = 0
while n < 4200:
    s = " ".join(rng.choice(words) for _ in range(12))
    out.append("Record %04d: %s." % (len(out), s))
    n += 13
prompt = ("Below is a long reference document. Read it and then continue.\n\n"
          + "\n".join(out)
          + "\n\nSummarize the document in one sentence, then list the first ten record names verbatim:\n")
open(sys.argv[1], "w").write(prompt)
print("prompt chars:", len(prompt))
PYEOF
PTOK=$(python3 -c "import json;print(len(open('$PROMPTF').read()))")
echo "prompt_chars=$PTOK" >> "$STAMPF"

# --- ONE decode probe: 4k prompt, 128 tokens, temp 0 ---
echo "PROBE: firing (4k prompt, n_predict 128, temp 0)"
T0=$(date +%s.%N)
python3 - "$PORT" "$PROMPTF" "$PROBEJSON" <<'PYEOF'
import sys, json, urllib.request
port, promptf, outf = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = open(promptf).read()
body = json.dumps({"prompt": prompt, "n_predict": 128, "temperature": 0.0,
                   "cache_prompt": True, "stream": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    resp = json.load(r)
open(outf, "w").write(json.dumps(resp, indent=2))
print("timings:", json.dumps(resp.get("timings", {})))
print("text head:", resp.get("content", "")[:200].replace("\n", " "))
PYEOF
RC=$?
T1=$(date +%s.%N)
echo "probe_rc=$RC wall_s=$(echo "$T1 - $T0" | bc)" >> "$STAMPF"
echo "post_temps: max_die=$(die_hot) C  $(date '+%F %T')" >> "$STAMPF"
sleep 5

# --- settle, then teardown by PID ---
echo "SETTLE: letting queues drain (30 s)"
sleep 30
cleanup
trap - EXIT
echo "P0 DONE"
echo "outputs:"
echo "  server log : $LOG"
echo "  stamp      : $STAMPF"
echo "  probe json : $PROBEJSON"
