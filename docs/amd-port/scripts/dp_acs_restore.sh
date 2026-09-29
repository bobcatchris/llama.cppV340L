#!/usr/bin/env bash
# dp_acs_restore.sh - restore ACS control registers saved by dp_acs_clear.sh
# Usage: sudo dp_acs_restore.sh [backup-file]   (default: newest in /tmp/dp_acs_state)

set -u
STATE_DIR=/tmp/dp_acs_state
if [ "$(id -u)" -ne 0 ]; then echo "REFUSED: run as root" >&2; exit 1; fi
BK="${1:-$(ls -1t $STATE_DIR/acs_backup_*.txt 2>/dev/null | head -1)}"
[ -f "$BK" ] || { echo "no backup found" >&2; exit 1; }
echo "restoring from $BK"
while read -r b reg val; do
    setpci -s "$b" "$reg.w=$val" 2>/dev/null
    rb=$(setpci -s "$b" "$reg.w" 2>/dev/null)
    echo "$b $reg: -> $rb (was $val)"
done < "$BK"
