#!/usr/bin/env bash
# probe_rccl_prefill.sh - run the RCCL prefill-size probe under the campaign
# GPU boot lock. rccl-ext desk variant of verify-transport's probe_rccl.sh.
# Lock protocol (E-082 law + E-092 LOCK SETTLE RULE): check-and-hold on
# /tmp/campaign_gpu_boot.lock (150 s poll cadence), 60 s settle, per-die VRAM
# verify (< 200 MiB) before touching dies, release at exit. Probe self-caps at
# 100 s (SIGALRM) and 140 s here.
# usage: probe_rccl_prefill.sh <tag> [probe args...]
set -uo pipefail
WT=/media/chris/ssd128/llamacpp/wt-rccl-ext
PROBE=$WT/docs/amd-port/probes/rccl_prefill_probe
RES=$WT/docs/amd-port/results
LOCK=/tmp/campaign_gpu_boot.lock
TAG=${1:?tag required}; shift
DESK=rccl-ext

cleanup_lock() { rm -f "$LOCK" 2>/dev/null; }
trap cleanup_lock EXIT INT TERM

while [ -f "$LOCK" ]; do
  HOLDER=$(cat "$LOCK" 2>/dev/null)
  if [ "$HOLDER" = "$DESK" ]; then
    echo "[$TAG] lock already ours, reusing"
    break
  fi
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || date +%s) ))
  if [ "$AGE" -gt 1800 ]; then
    echo "[$TAG] lock stale (${AGE}s), stealing"
    cleanup_lock; break
  fi
  echo "[$TAG] campaign GPU lock held by [${HOLDER}], poll cadence 150 s"
  sleep 150
done

if [ ! -f "$LOCK" ] || [ "$(cat "$LOCK" 2>/dev/null)" != "$DESK" ]; then
  # no live llama-server may survive from a raced teardown
  if pgrep -f llama-server > /dev/null 2>&1; then
    echo "[$TAG] live llama-server present, refusing to boot over it"
    exit 1
  fi
  echo "{\"desk\": \"$DESK\", \"arm\": \"$TAG\", \"booted\": \"$(date +%Y%m%d_%H%M%S)\"}" > "$LOCK"
  echo "[$TAG] lock held"
fi

# LOCK SETTLE RULE: a released lock does not imply a drained die
echo "[$TAG] settle 60 s before touching dies"
sleep 60

VRAM=$(rocm-smi --showmeminfo vram --json 2>/dev/null | grep -o '"VRAM Total Used Memory (B)": "[0-9]*"' | grep -o '[0-9]*')
OK=1
DIE=0
for B in $VRAM; do
  MB=$(( B / 1048576 ))
  echo "[$TAG] die $DIE vram ${MB} MiB"
  if [ "$DIE" -le 2 ] && [ "$MB" -ge 200 ]; then
    echo "[$TAG] die $DIE not drained (${MB} MiB >= 200), aborting probe"
    OK=0
  fi
  DIE=$(( DIE + 1 ))
done
[ "$OK" = "1" ] || exit 1

mkdir -p "$RES"
OUT=$RES/rccl_prefill_${TAG}_$(date +%Y%m%d_%H%M%S).log
echo "[$TAG] probe -> $OUT"
( timeout 140 "$PROBE" "$@" 2>&1; echo "probe-exit: ${PIPESTATUS[0]}" ) | tee "$OUT"
RC=$(grep -oE "probe-exit: [0-9]+" "$OUT" | tail -1 | cut -d' ' -f2)
echo "[$TAG] probe rc=$RC"
[ "$RC" = "0" ] || exit 1
exit 0
