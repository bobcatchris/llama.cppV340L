#!/usr/bin/env bash
# dp_acs_clear.sh - DP-2 STEP A: clear ACS P2P redirect bits on the GPU fabric
# bridges so the KFD topology can create peer links (see
# docs/amd-port/results/dp_p0_p2p_gates_2026-09-25.md for the full chain).
#
# OWNER-GO REQUIRED on first application (system-state write). No server may be
# running (churn law). Reversible: dp_acs_restore.sh, or any reboot (config
# space resets to firmware defaults).
#
# Usage: dp_acs_clear.sh --go        (without --go: dry run, prints the plan)

set -u
BRIDGES="00:01.0 00:01.1 01:00.0 02:00.0 02:01.0 03:00.0 04:00.0 06:00.0 07:00.0 09:00.0 0a:00.0 0a:01.0 0b:00.0 0c:00.0 0e:00.0 0f:00.0"
STATE_DIR=/tmp/dp_acs_state
GO="${1:-}"

if [ "$GO" != "--go" ]; then
    echo "DRY RUN (use --go to apply). Bridges to clear ACS control on:"
    for b in $BRIDGES; do
        off=$(lspci -v -s $b 2>/dev/null | awk '/Access Control Services/ {gsub(/[^0-9a-f]/,"",$2); print $2; exit}')
        if [ -z "$off" ]; then echo "  $b: NO ACS CAP (skip)"; continue; fi
        cur=$(setpci -s $b $(printf '0x%x' $((0x$off + 6))).w 2>/dev/null)
        [ -z "$cur" ] && cur="<unreadable without sudo>"
        echo "  $b: acs_cap=0x$off ctrl_reg=0x$(printf %x $((0x$off + 6))) current=0x$cur"
    done
    echo "After clearing: amdgpu reload (sudo modprobe -r amdgpu && sudo modprobe amdgpu),"
    echo "then check /sys/class/kfd/kfd/topology/nodes/*/p2p_links/"
    exit 0
fi

# fail-loud preconditions
if pgrep -f llama-server >/dev/null 2>&1; then
    echo "REFUSED: llama-server is running (churn law; stop it first)" >&2; exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "REFUSED: run as root (sudo)" >&2; exit 1
fi
mkdir -p "$STATE_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SAVE="$STATE_DIR/acs_backup_${STAMP}.txt"
: > "$SAVE"

fail=0
for b in $BRIDGES; do
    off=$(lspci -v -s $b 2>/dev/null | awk '/Access Control Services/ {gsub(/[^0-9a-f]/,"",$2); print $2; exit}')
    if [ -z "$off" ]; then echo "SKIP $b (no ACS cap)"; continue; fi
    ctrl=$((0x$off + 6))
    cur=$(setpci -s $b $(printf '0x%x' $ctrl).w 2>/dev/null)
    if [ -z "$cur" ]; then echo "FAIL read $b" >&2; fail=1; continue; fi
    echo "$b 0x$(printf %x $ctrl) $cur" >> "$SAVE"
    if [ "$cur" = "0x0000" ] || [ "$cur" = "0000" ]; then echo "OK   $b already 0x0000"; continue; fi
    setpci -s $b $(printf '0x%x' $ctrl).w=0x0000 2>/dev/null
    rb=$(setpci -s $b $(printf '0x%x' $ctrl).w 2>/dev/null)
    if [ "$rb" = "0x0000" ] || [ "$rb" = "0000" ]; then
        echo "OK   $b: $cur -> 0x0000 (verified)"
    else
        echo "FAIL $b: readback=$rb (restoring)" >&2
        setpci -s $b $(printf '0x%x' $ctrl).w=$cur 2>/dev/null
        fail=1
    fi
done

echo "backup: $SAVE"
if [ "$fail" -ne 0 ]; then echo "RESULT: PARTIAL FAILURE (see above); nothing left half-cleared" >&2; exit 2; fi
cat <<'EOF'

ACS cleared. Next (manual, one churn event):
  1. sudo modprobe -r amdgpu && sudo modprobe amdgpu
  2. ls /sys/class/kfd/kfd/topology/nodes/*/p2p_links/   <- expect non-empty
  3. re-run the W31 peer probes for the same-card pairs first
  4. sudo dmesg | grep -i "P2P"  (the 'not supported by the chipset' lines
     should be gone for same-card pairs; cross-card pairs still hit the
     whitelist until DP-2 STEP B)
EOF
