#!/usr/bin/env bash
# probe_rccl.sh - run the RCCL transport probe under the campaign GPU boot lock.
# usage: probe_rccl.sh <tag> [probe args...]
# Lock protocol (E-082 law): check-and-wait on /tmp/campaign_gpu_boot.lock,
# hold with our desk name, release at exit. Probe is capped at 110 s by the
# binary itself (SIGALRM) and once more at 140 s here.
set -uo pipefail
WT=/media/chris/ssd128/llamacpp/wt-verify-transport
PROBE=$WT/docs/amd-port/probes/rccl_probe
RES=$WT/docs/amd-port/results
LOCK=/tmp/campaign_gpu_boot.lock
TAG=${1:?tag required}; shift
DESK=verify-transport

cleanup_lock() { rm -f "$LOCK" 2>/dev/null; }
trap cleanup_lock EXIT INT TERM

while [ -f "$LOCK" ]; do
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || date +%s) ))
  if [ "$AGE" -gt 1800 ]; then
    echo "[$TAG] lock stale (${AGE}s), stealing"
    cleanup_lock; break
  fi
  [ ${LOCKWAIT:-0} -eq 0 ] && echo "[$TAG] campaign GPU lock held, waiting: $(cat "$LOCK" 2>/dev/null)"
  LOCKWAIT=1
  sleep 20
done

echo "{\"desk\": \"$DESK\", \"arm\": \"$TAG\", \"booted\": \"$(date +%Y%m%d_%H%M%S)\"}" > "$LOCK"
echo "[$TAG] lock held"

# drain a moment before touching the dies
sleep 5

OUT=$RES/rccl_probe_${TAG}_$(date +%Y%m%d_%H%M%S).log
echo "[$TAG] probe -> $OUT"
( timeout 140 "$PROBE" "$@" 2>&1; echo "probe-exit: ${PIPESTATUS[0]}" ) | tee "$OUT"
RC=$(grep -oE "probe-exit: [0-9]+" "$OUT" | tail -1 | cut -d' ' -f2)
echo "[$TAG] probe rc=$RC"
[ "$RC" = "0" ] || exit 1
exit 0
