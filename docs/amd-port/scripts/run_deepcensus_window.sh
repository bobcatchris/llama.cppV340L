#!/usr/bin/env bash
# W15 DEEP-CENSUS DESK - U1 deep-context decode window.
# One boot: TP4 canonical stack, ctx 131072, rocprofv3 kernel trace.
# Probe A: ~64k-token prefill + 128 gen. Probe B: +~55k more (total ~120k KV) + 128 gen.
# Lock law: check-and-hold /tmp/campaign_gpu_boot.lock (single sleeps <= 150 s,
# cool-die < 60 C), JSON line desk/arm/ts/pid. PID-targeted teardown.
set -u
BIN=/media/chris/ssd128/llamacpp/llama.cpp/build-hip/bin/llama-server
MODEL=/media/chris/ssd128/gguf/Qwen3.8-27B-ASCII-P1M.gguf
OUT=/media/chris/ssd128/llamacpp/wt-deep-census/docs/amd-port/results/W15_deepcensus_2026-09-23
LOCK=/tmp/campaign_gpu_boot.lock
WINDOW_MAX=1980   # 33 min hard cap from lock grant
BASEENV="GGML_CUDA_ALLREDUCE=nccl GGML_CUDA_MMVQ_IQ3S_SHARE=1 GGML_CUDA_MMVQ_IQ3XXS_SHARE=1 GGML_CUDA_MMVQ_Q3K_SHARE=1 GGML_CUDA_MMVQ_Q4K_SHARE=1 GGML_CUDA_MMVQ_Q5K_SHARE=1 GGML_CUDA_MMVQ_Q6K_SHARE=1 LLAMA_DRAFT_FAST_TOPK=1 LLAMA_DRAFT_PACKED_GET=1 LLAMA_DRAFT_LIGHT_SYNC=1 LLAMA_VERIFY_ROW_SAMPLING=1 LLAMA_ASYNC_INPUT=1 GGML_PINNED_DEV_COPY=1"

stamp() { echo "$(date +%H%M%S) $*" >> "${OUT}.stamp"; }

die_temps_ok() {
  for p in 0000:05:00.0 0000:08:00.0 0000:0d:00.0 0000:10:00.0; do
    hw=$(ls -d /sys/bus/pci/devices/$p/hwmon/hwmon* 2>/dev/null | head -1)
    t=$(cat "$hw/temp1_input" 2>/dev/null)
    [ -z "$t" ] && { echo "DIE-MISSING $p"; return 1; }
    [ "$t" -ge 60000 ] && { echo "DIE-HOT $p $((t/1000))C"; return 1; }
  done
  return 0
}

# --- lock check-and-hold ---
waited=0
while [ -f "$LOCK" ]; do
  [ "$waited" -ge 5400 ] && { echo "LOCK-TIMEOUT"; exit 1; }
  sleep 150; waited=$((waited+150))
done
printf '{"desk":"deep-census","arm":"W15-u1-deepctx","ts":"%s","pid":%d}\n' \
  "$(date +%Y%m%d_%H%M%S)" "$$" > "$LOCK"
# race guard: if we lost the grab, back off and retry once per 150 s
if ! grep -q '"desk":"deep-census"' "$LOCK"; then
  while [ -f "$LOCK" ]; do sleep 150; done
  printf '{"desk":"deep-census","arm":"W15-u1-deepctx","ts":"%s","pid":%d}\n' \
    "$(date +%Y%m%d_%H%M%S)" "$$" > "$LOCK"
  grep -q '"desk":"deep-census"' "$LOCK" || { echo "LOCK-LOST"; exit 1; }
fi
T0=$SECONDS
: > "${OUT}.stamp"; stamp "LOCK-HELD pid $$"

# --- cool-die gate ---
cool=0
for i in 1 2 3 4 5 6; do
  if die_temps_ok; then cool=1; break; fi
  stamp "COOL-WAIT $i"; sleep 150
done
[ "$cool" = 1 ] || { stamp "VOID DIES-NOT-COOL"; rm -f "$LOCK"; exit 1; }
stamp "DIES-COOL $(for p in 05:00 08:00 0d:00 10:00; do hw=$(ls -d /sys/bus/pci/devices/0000:$p/hwmon/hwmon* | head -1); printf "%s=%sC " "$p" "$(( $(cat $hw/temp1_input) / 1000 ))"; done)"

# --- boot (up to 2 attempts inside the window) ---
boot_server() {
  env HIP_VISIBLE_DEVICES=0,1,2,3 $BASEENV \
    /opt/rocm-6.2.0/bin/rocprofv3 --kernel-trace --output-format csv -o "${OUT}" -- \
    "$BIN" -m "$MODEL" -ngl 999 -sm tensor -c 131072 --batch-size 512 --ubatch-size 512 \
    -ctk q4_0 -ctv q4_0 -fa on --spec-type draft-mtp \
    --device ROCm0,ROCm1,ROCm2,ROCm3 --port 8083 -t 8 >> "${OUT}.server" 2>&1 &
  echo $!
}

teardown() {
  SRVPID="$1"
  kill "$SRVPID" 2>/dev/null
  sleep 3
  # PID-targeted: any llama-server child left on port 8083
  for p in $(pgrep -f "llama-server.*--port 8083"); do
    stamp "TEARDOWN-KILL $p"; kill "$p" 2>/dev/null
  done
  sleep 5
  pgrep -f "llama-server.*--port 8083" >/dev/null && {
    for p in $(pgrep -f "llama-server.*--port 8083"); do kill -9 "$p" 2>/dev/null; done
  }
}

SRV=""
ok=0
for attempt in 1 2; do
  [ "$attempt" = 2 ] && stamp "BOOT-RETRY (first attempt failed)"
  SRV=$(boot_server)
  stamp "BOOT-START attempt $attempt srvpid $SRV"
  for i in $(seq 1 30); do
    sleep 20
    [ $((SECONDS - T0)) -gt "$WINDOW_MAX" ] && break
    curl -s -m 5 http://127.0.0.1:8083/health | grep -q ok && { ok=1; break; }
    kill -0 "$SRV" 2>/dev/null || break
  done
  [ "$ok" = 1 ] && break
  stamp "BOOT-FAIL attempt $attempt"
  teardown "$SRV"
  [ $((SECONDS - T0)) -gt "$WINDOW_MAX" ] && break
done
if [ "$ok" != 1 ]; then
  stamp "VOID BOOT-FAILED-TWICE"
  rm -f "$LOCK"
  exit 1
fi
stamp "BOOT-OK after $((SECONDS - T0)) s"

# --- probes ---
probe() {  # $1 = label, $2 = python expr file
  stamp "PROBE-$1-START"
  python3 "$2" >> "${OUT}.probes" 2>&1
  rc=$?
  stamp "PROBE-$1-END rc=$rc t+$((SECONDS - T0))s"
}

cat > /tmp/w15_probeA.py << 'PYEOF'
import json, time, urllib.request
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango ")
prompt = ("Summarize the following reference text in one sentence.\n\n" + words * 2114)
body = {"messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128, "temperature": 0}
req = urllib.request.Request("http://127.0.0.1:8083/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=1500))
print("probeA wall %.1f s usage %s" % (time.time() - t0, r.get("usage")))
PYEOF

cat > /tmp/w15_probeB.py << 'PYEOF'
import json, time, urllib.request
words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango ")
# identical first ~64k so the slot prefix-cache reuses probe A's KV;
# only the delta (~55k) is prefill-processed -> total KV ~120k
prompt = ("Summarize the following reference text in one sentence.\n\n"
          + words * 2114 + words * 1833
          + "\n\nNow summarize ALL of the reference text above in one sentence.")
body = {"messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128, "temperature": 0}
req = urllib.request.Request("http://127.0.0.1:8083/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=1500))
print("probeB wall %.1f s usage %s" % (time.time() - t0, r.get("usage")))
PYEOF

probe A /tmp/w15_probeA.py
[ $((SECONDS - T0)) -lt "$WINDOW_MAX" ] || { stamp "WINDOW-EXPIRED post-A"; teardown "$SRV"; rm -f "$LOCK"; stamp "LOCK-RELEASED"; exit 1; }
probe B /tmp/w15_probeB.py

# --- teardown + records ---
grep -E "print_timing|slot |n_ctx|cache" "${OUT}.server" | tail -40 >> "${OUT}.stamp" 2>/dev/null
teardown "$SRV"
stamp "TEARDOWN-DONE t+$((SECONDS - T0))s"
rm -f "$LOCK"
stamp "LOCK-RELEASED"
ls -la "${OUT}"* >> "${OUT}.stamp" 2>&1
echo "WINDOW-DONE"
