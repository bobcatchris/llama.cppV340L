#!/usr/bin/env python3
# VOID-GATE adjudicator for guard-battery thermal sideband logs (ledger E-133).
#
# A DECISION cell is VOID iff, over its decode window, the die-3 sclk
# act-mean < 1150 MHz OR the fraction of loaded samples at <= 991 MHz > 20%
# (W18 far-die soak: decode 23.2+ t/s iff act-mean >= ~1190 with low-band
# duty 0-20%; 17.3-17.7 t/s iff act-mean 1068-1135 with duty 44-53%).
#
# CAVEAT - column groups are CARD-indexed, not die-indexed: the sampler
# (guard_battery.py ThermalSampler) hardcodes rocm-smi card0/card1/card2.
# In the banked evidence set card0/1/2 == PCI 05:00/08:00/0D:00 == dies
# 1/2/3, so "die 3" is the LAST group (c2). After a boot that reshuffles
# the card order this default is wrong (e.g. card1 can be the NVIDIA
# display GPU); use --die3-group to pin the die-3 group explicitly, and
# treat cross-boot comparisons with suspicion unless PCI mapping holds.
# Die 4 (10:00) is never sampled - it throttles together with die 3 per
# W15/W18, so die 3 stands in for the far-die pair.
import argparse
import csv
import json
import re
import sys

DEFAULT_ACT_MEAN = 1150.0
DEFAULT_DUTY = 0.20
LOW_BAND = 991          # throttle band top (991/775/560 MHz), MHz
LOADED = 500            # samples at or below this sclk are idle/desktop, MHz


def detect_layout(path):
    """Return ("csv", groups) with groups = {n: {suffix: colidx}},
    ("text", n_groups) for the W15 4-die plain-text format, or None."""
    with open(path, "r", errors="replace") as f:
        first = f.readline()
    if re.match(r"timestamp\s*,", first) and "_sclk" in first:
        cols = first.strip().split(",")
        groups = {}
        for i, c in enumerate(cols):
            m = re.match(r"c(\d+)_(edge|junc|mem|sclk)$", c.strip())
            if m:
                groups.setdefault(int(m.group(1)), {})[m.group(2)] = i
        if not any("sclk" in g for g in groups.values()):
            return None
        return "csv", groups
    if re.match(r"\d+\s+(\d+C\s+\d+MHz\s*){2,}$", first):
        n = (len(first.split()) - 1) // 2
        return "text", n
    return None


def read_cells(path, layout):
    """Yield (elapsed_s, [sclk per group]) rows; groups with no reading None."""
    kind, spec = layout
    if kind == "csv":
        sclk_col = {n: g["sclk"] for n, g in spec.items() if "sclk" in g}
        with open(path, "r", errors="replace") as f:
            for row in csv.reader(f):
                if not row or row[0].strip() == "timestamp":
                    continue
                try:
                    t = float(row[1])
                except (IndexError, ValueError):
                    continue  # header or truncated line
                sclk = []
                for n in sorted(sclk_col):
                    try:
                        sclk.append(float(row[sclk_col[n]]))
                    except (IndexError, ValueError):
                        sclk.append(None)
                yield t, sclk
    else:
        t0 = None
        with open(path, "r", errors="replace") as f:
            for line in f:
                tok = line.split()
                if len(tok) < 3:
                    continue
                try:
                    t = float(tok[0])
                except ValueError:
                    continue
                if t0 is None:
                    t0 = t
                sclk = []
                for d in range(spec):
                    try:  # tokens: <edge>C <sclk>MHz per die
                        sclk.append(float(tok[1 + 2 * d + 1].rstrip("MHz")))
                    except (IndexError, ValueError):
                        sclk.append(None)
                yield t - t0, sclk


def auto_window(dur):
    # Battery shape: decode guard sits early; the 200k needle_recall at the
    # END of cells with wall > 120 s is the soak driver for the NEXT cell.
    if dur > 120.0:
        return 10.0, 95.0
    return max(0.0, dur - 25.0), dur + 1.0


def group_stats(samples):
    loaded = [s for s in samples if s is not None and s > LOADED]
    if not loaded:
        return None
    return {
        "n_loaded": len(loaded),
        "act_mean": sum(loaded) / len(loaded),
        "low_duty": sum(1 for s in loaded if s <= LOW_BAND) / len(loaded),
        "min": min(loaded),
    }


def adjudicate(label, path, args):
    out = {"cell": label, "path": path, "verdict": "ERROR", "groups": {}}
    try:
        layout = detect_layout(path)
    except OSError as e:
        out["reason"] = "unreadable: %s" % e
        return out
    if layout is None:
        out["verdict"] = "UNSUPPORTED"
        out["reason"] = ("no per-group sclk column found; the E-133 rule "
                         "needs per-die sclk - not guessing, cell not adjudicated")
        return out
    kind, spec = layout
    out["layout"] = kind
    ng = len(spec) if kind == "csv" else spec
    rows = [(t, s) for t, s in read_cells(path, layout)]
    if not rows:
        out["verdict"] = "NODATA"
        out["reason"] = "no sample rows"
        return out
    if args.win_lo is not None:
        lo, hi = args.win_lo, args.win_hi
    elif args.whole:
        lo, hi = rows[0][0], rows[-1][0] + 1.0
    else:
        lo, hi = auto_window(rows[-1][0])
    out["window_s"] = [round(lo, 1), round(hi, 1)]
    for d in range(ng):
        out["groups"][str(d)] = group_stats(
            [s[d] for t, s in rows if lo <= t <= hi])
    # csv groups are card-indexed (card0/1/2 = dies 1/2/3): die 3 = last
    # group. text groups are die-ordered over 4 dies: die 3 = group 2.
    if args.die3_group is not None:
        d3 = args.die3_group
    else:
        d3 = ng - 1 if kind == "csv" else 2
    out["die3_group"] = d3
    st = out["groups"].get(str(d3))
    if st is None:
        out["verdict"] = "NOCLKS"
        out["reason"] = ("no loaded sclk sample (>%d MHz) in group c%d in "
                         "window" % (LOADED, d3))
        return out
    am, duty = st["act_mean"], st["low_duty"]
    out["act_mean"] = am
    out["low_duty"] = duty
    void_am = am < args.act_mean_thresh
    void_duty = duty > args.duty_thresh
    if void_am or void_duty:
        out["verdict"] = "VOID"
        out["reason"] = []
        if void_am:
            out["reason"].append("die-3 act-mean %.0f < %.0f MHz" % (am, args.act_mean_thresh))
        if void_duty:
            out["reason"].append("die-3 <=%d MHz duty %.0f%% > %.0f%%"
                                 % (LOW_BAND, 100 * duty, 100 * args.duty_thresh))
        out["reason"] = "; ".join(out["reason"])
    else:
        out["verdict"] = "PASS"
        out["reason"] = ("die-3 act-mean %.0f MHz >= %.0f, <=%d duty %.0f%%"
                         % (am, args.act_mean_thresh, LOW_BAND, 100 * duty))
    return out


def main():
    p = argparse.ArgumentParser(
        description="E-133 VOID gate: adjudicate guard-battery thermal "
                    "sideband logs (die-3 sclk soak rule).",
        epilog="COLUMN GROUPS ARE CARD-INDEXED, NOT DIE-INDEXED: the sampler "
               "hardcodes rocm-smi card0/card1/card2. In the banked logs "
               "card0/1/2 == PCI 05:00/08:00/0D:00 == dies 1/2/3, so the "
               "default die-3 group is the LAST group. After a boot reshuffle "
               "(e.g. card1 = NVIDIA display GPU) pin --die3-group. Die 4 is "
               "never sampled; die 3 stands in for the far-die pair.\n"
               "Windows: auto = [10,95] s for cells with wall > 120 s "
               "(battery shape), else the last 25 s; --whole for whole-log "
               "steady-state traces. Exit: 0 all PASS, 1 a DECISION cell is "
               "VOID, 2 any cell unadjudicable (UNSUPPORTED/NODATA/NOCLKS).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", metavar="CELL=PATH",
                   help="thermal sideband log; CELL=LABEL prefix optional")
    p.add_argument("--act-mean-thresh", type=float, default=DEFAULT_ACT_MEAN,
                   help="VOID if die-3 sclk act-mean below this MHz "
                        "(default %(default)g)")
    p.add_argument("--duty-thresh", type=float, default=DEFAULT_DUTY,
                   help="VOID if die-3 <=991 MHz duty fraction above this "
                        "(default %(default)g)")
    p.add_argument("--die3-group", type=int, default=None,
                   help="column group treated as die 3 (default: last group; "
                        "group 2 for the 4-die text layout)")
    p.add_argument("--win-lo", type=float, default=None,
                   help="override window start (elapsed s; use with --win-hi)")
    p.add_argument("--win-hi", type=float, default=None,
                   help="override window end (elapsed s)")
    p.add_argument("--whole", action="store_true",
                   help="adjudicate the whole log, not the auto decode window")
    p.add_argument("--observe", action="store_true",
                   help="observational mode: report verdicts but never gate "
                        "the exit code on VOID")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = p.parse_args()

    cells = []
    for spec in args.logs:
        label, _, path = spec.partition("=")
        if not _:
            path = label
            label = path.rsplit("/", 1)[-1].replace("_thermal.log", "")
        cells.append((label, path))
    if args.win_lo is not None and args.win_hi is None:
        p.error("--win-lo requires --win-hi")

    results = [adjudicate(l, pa, args) for l, pa in cells]
    gateable = [r for r in results
                if r["verdict"] in ("PASS", "VOID")]
    unadjudicable = [r for r in results if r["verdict"] not in ("PASS", "VOID")]
    voids = [r for r in results if r["verdict"] == "VOID"]

    if args.json:
        print(json.dumps({
            "rule": ("VOID if die-3 sclk act-mean < %g MHz or <=%d MHz duty "
                     "> %g over the decode window (ledger E-133, W18 far-die "
                     "soak)" % (args.act_mean_thresh, LOW_BAND, args.duty_thresh)),
            "thresholds": {"act_mean_mhz": args.act_mean_thresh,
                           "low_band_mhz": LOW_BAND, "duty": args.duty_thresh,
                           "loaded_mhz": LOADED},
            "observational": bool(args.observe),
            "cells": results,
        }, indent=1))
    else:
        names = sorted({g for r in results for g in r["groups"]},
                       key=lambda s: int(s))
        hdr = ("%-22s %-6s %-11s %-6s " % ("cell", "layout", "win_s", "verdict")
               + " ".join("%-15s" % ("c%s actmean/duty/min" % n) for n in names)
               + "  why")
        print(hdr)
        for r in results:
            stats = []
            for n in names:
                st = r["groups"].get(n)
                stats.append("%4.0f/%3.0f%%/%4.0f" % (
                    st["act_mean"], 100 * st["low_duty"], st["min"])
                    if st else "%15s" % "-")
            print("%-22s %-6s %-11s %-6s " % (
                r["cell"], r.get("layout", "-"),
                "%s-%s" % tuple(r["window_s"]) if "window_s" in r else "-",
                r["verdict"]) + " ".join(stats) + "  " + r.get("reason", ""))

    code = 0
    for r in unadjudicable:
        print("void-gate: cell %s: %s (%s)" % (
            r["cell"], r["verdict"], r.get("reason", "")), file=sys.stderr)
        code = 2
    if voids and not args.observe:
        print("void-gate: %d/%d DECISION cell(s) VOID (E-133): %s" % (
            len(voids), len(gateable),
            ", ".join(r["cell"] for r in voids)), file=sys.stderr)
        code = 1
    elif voids:
        print("void-gate: observe mode, %d VOID ignored: %s" % (
            len(voids), ", ".join(r["cell"] for r in voids)))
    if code == 0 and not unadjudicable and not voids:
        print("void-gate: all %d cell(s) PASS (E-133)" % len(results))
    return code


if __name__ == "__main__":
    sys.exit(main())
