#!/usr/bin/env bash
# dp_uv_apply.sh - DP-1: apply a guarded undervolt to one die via the standard
# amdgpu overdrive interface (pp_od_clk_voltage). E-149 hazard law applies:
# dp_uv_guard.sh MUST run in another terminal for the whole session.
#
# Stock curves (receipt dp_p0_power_2026-09-25, identical on all 4 dies):
#   sclk: 0:300@800mV 1:560@950 2:775@1000 3:991@1050 4:1138@1100
#         5:1269@1150 6:1350@1200 7:1500@1250
#   mclk: 0:167@800 1:500@800 2:800@1100 3:945@1150
# Decode band is levels 4-7; the 991MHz W18 throttle latch is level 3.
# First profile (CONSERVATIVE): shave 100mV off levels 4-7, 50mV off mclk 2-3.
#
# Usage: sudo dp_uv_apply.sh <pci-slot like 0000:10:00.0> [--go]
# Without --go: prints the planned writes (dry run).

set -u
SLOT="${1:-}"
GO="${2:-}"
[ -z "$SLOT" ] && { echo "usage: $0 <pci-slot> [--go]" >&2; exit 1; }
DEV=$(for d in /sys/class/drm/card*/device; do
    grep -q "PCI_SLOT_NAME=$SLOT" "$d/uevent" 2>/dev/null && { basename "$(dirname "$d")"; break; }
done)
[ -z "$DEV" ] && { echo "no DRM card for slot $SLOT" >&2; exit 1; }
OD="/sys/class/drm/$DEV/device/pp_od_clk_voltage"

plan=(
    "s 4 1138 1000"
    "s 5 1269 1050"
    "s 6 1350 1100"
    "s 7 1500 1150"
    "m 2 800 1050"
    "m 3 945 1100"
)
echo "die $SLOT ($DEV) pp_od_clk_voltage writes:"
printf '  %s\n' "${plan[@]}"
if [ "$GO" != "--go" ]; then
    echo "DRY RUN. Commit line ('c') NOT sent. Re-run with --go."
    exit 0
fi
[ "$(id -u)" -ne 0 ] && { echo "REFUSED: root required" >&2; exit 1; }
if ! pgrep -f dp_uv_guard.sh >/dev/null 2>&1; then
    echo "REFUSED: dp_uv_guard.sh is not running (E-149 hazard law)" >&2; exit 1
fi
for line in "${plan[@]}"; do echo "$line" > "$OD" || exit 2; done
echo "c" > "$OD" || exit 2
echo "== after commit:"
cat "$OD"
echo "revert anytime: echo r > $OD ; echo c > $OD"
