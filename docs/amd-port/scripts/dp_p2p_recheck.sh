#!/usr/bin/env bash
# dp_p2p_recheck.sh - DP-2 STEP A first-signal check (run AFTER dp_acs_clear.sh
# + amdgpu reload). Read-only. Exit 0 iff every vega10 KFD node has a non-empty
# p2p_links directory.

set -u
rc=0
for n in /sys/class/kfd/kfd/topology/nodes/*/; do
    gid=$(cat "$n/gpu_id" 2>/dev/null)
    [ "$gid" = "0" ] && continue   # CPU node
    links=$(ls "$n/p2p_links" 2>/dev/null | grep -c .)
    echo "node $(basename "$n") gpu_id=$gid p2p_links=$links"
    [ "$links" -eq 0 ] && rc=1
done
if [ "$rc" -eq 0 ]; then
    echo "SIGNAL: peer links present - proceed to the W31 probes (same-card pairs first)"
else
    echo "NO LINKS: check 'sudo dmesg | grep -i P2P' for remaining whitelist refusals"
fi
echo "full dmesg P2P verdict lines this boot:"
dmesg 2>/dev/null | grep -i "P2P access" | tail -4 || echo "(dmesg needs sudo)"
exit $rc
