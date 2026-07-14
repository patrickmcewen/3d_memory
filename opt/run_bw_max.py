#!/usr/bin/env python3
"""Run the 3d_memory phase-1 BW-max MINLP for a YAML-selected config.

Reads opt/config.yaml, deep-merges a named config onto `defaults`, renders an
AMPL data file, and solves opt/bw_max.mod with the bundled StreamHLS
ampl+gurobi. Generated artifacts land in opt/build/ (gitignored).

Examples:
    python opt/run_bw_max.py --list
    python opt/run_bw_max.py --config dram_100Mb
    python opt/run_bw_max.py --all
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent          # .../3d_memory/opt
MODEL = HERE / "bw_max.mod"
REPORT = HERE / "report.run"
BUILD = HERE / "build"

# Default location of the bundled ampl+gurobi (override with --ampl or $AMPL_BIN).
DEFAULT_AMPL = Path(
    "/nfs/pool0/pmcewen/rsgvm13dir/codesign2/Stream-HLS/ampl.linux-intel64/ampl"
)

# Every param the model declares, by block. Used to render the .dat and to
# assert a config is complete (fail loud on a missing/extra key).
PARAMS = {
    "problem": ["C", "A", "L", "t_layer"],
    "technology": ["k_dec", "k_WL", "k_BL", "t_SA0", "t_restore", "destructive",
                   "t_sw", "v_cell", "v_sa0", "k_vdec", "v_sel"],
    "bounds": ["NBL_min", "NBL_max", "NWL_min", "NWL_max", "margin_min",
               "margin_max", "Nshare_min", "Nshare_max", "Nindep_max",
               "tcyc_max", "BW_max"],
}
# fix-key -> (min_param, max_param) bound pair it collapses.
FIXABLE = {
    "margin":  ("margin_min", "margin_max"),
    "N_BL":    ("NBL_min", "NBL_max"),
    "N_WL":    ("NWL_min", "NWL_max"),
    "N_share": ("Nshare_min", "Nshare_max"),
}


def merge(defaults: dict, cfg: dict) -> dict:
    """Block-level deep-merge of one config onto defaults."""
    out = {}
    for block in ("problem", "technology", "bounds", "solver", "fix"):
        base = dict(defaults.get(block) or {})
        base.update(cfg.get(block) or {})
        out[block] = base
    return out


def apply_fixes(merged: dict) -> None:
    """Collapse a variable's [min,max] bounds onto a pinned value."""
    for var, val in (merged["fix"] or {}).items():
        assert var in FIXABLE, f"cannot fix unknown variable '{var}' (fixable: {list(FIXABLE)})"
        lo, hi = FIXABLE[var]
        merged["bounds"][lo] = merged["bounds"][hi] = val


def fmt(v) -> str:
    """AMPL-friendly literal: ints stay ints, floats keep full precision."""
    return str(int(v)) if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and abs(v) < 1e6) else repr(float(v))


def render_dat(merged: dict) -> str:
    lines = []
    for block, keys in PARAMS.items():
        have = merged[block]
        missing = [k for k in keys if k not in have]
        assert not missing, f"config block '{block}' is missing params: {missing}"
        lines.append(f"# {block}")
        for k in keys:
            lines.append(f"param {k} := {fmt(have[k])};")
        lines.append("")
    return "\n".join(lines)


def render_run(dat_path: Path, solver: dict) -> str:
    opts = " ".join(f"{k}={v}" for k, v in solver.items())
    return "\n".join([
        "option solver gurobi;",
        f"option gurobi_options '{opts}';",
        f"model {MODEL};",
        f"data {dat_path};",
        "solve;",
        f"commands {REPORT};",
        "",
    ])


REPORT_KEYS = {
    "BW":       r"objective\s+BW\s*=\s*([-\d.eE+]+)\s*B/s",
    "t_cycle":  r"cycle time t_cycle\s*=\s*([-\d.eE+]+)",
    "N_BL":     r"N_BL x N_WL\s*=\s*([-\d.eE+]+)\s*x",
    "N_WL":     r"N_BL x N_WL\s*=\s*[-\d.eE+]+\s*x\s*([-\d.eE+]+)",
    "b_acc":    r"sense width b_acc\s*=\s*([-\d.eE+]+)",
    "margin":   r"margin\s*=\s*([-\d.eE+]+)",
    "N_share":  r"sharing  N_share\s*=\s*([-\d.eE+]+)",
    "N_indep":  r"indep sets N_indep\s*=\s*([-\d.eE+]+)",
    "vol_pct":  r"volume used / budget.*\(([-\d.eE+]+)%\)",
}


def parse_report(text: str) -> dict:
    out = {}
    for key, pat in REPORT_KEYS.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    return out


def solve_config(name: str, defaults: dict, configs: dict, ampl: Path) -> dict:
    assert name in configs, f"unknown config '{name}' (available: {list(configs)})"
    merged = merge(defaults, configs[name])
    apply_fixes(merged)

    BUILD.mkdir(exist_ok=True)
    dat_path = BUILD / f"{name}.dat"
    run_path = BUILD / f"{name}.run"
    dat_path.write_text(render_dat(merged))
    run_path.write_text(render_run(dat_path, merged["solver"]))

    desc = configs[name].get("description", "").strip()
    print(f"\n########## config: {name} ##########")
    if desc:
        print(f"# {desc}")

    proc = subprocess.run([str(ampl), str(run_path)], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    (BUILD / f"{name}.log").write_text(proc.stdout + proc.stderr)
    assert proc.returncode == 0, f"ampl exited {proc.returncode} for config '{name}'"

    result = parse_report(proc.stdout)
    result["config"] = name
    return result


def print_summary(rows: list) -> None:
    if len(rows) < 2:
        return
    hdr = f"{'config':<26}{'BW[B/s]':>13}{'t_cyc[ns]':>11}{'N_BLxN_WL':>16}{'margin':>8}{'N_sh':>6}{'vol%':>7}"
    print("\n" + "=" * len(hdr))
    print("SUMMARY")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        geo = f"{r['N_BL']:.0f}x{r['N_WL']:.0f}" if r["N_BL"] else "-"
        print(f"{r['config']:<26}{r['BW']:>13.4g}{r['t_cycle']:>11.4g}{geo:>16}"
              f"{r['margin']:>8.3g}{r['N_share']:>6.0f}{r['vol_pct']:>7.1f}")
    print("=" * len(hdr))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="name of the config to solve")
    ap.add_argument("--all", action="store_true", help="solve every config in the file")
    ap.add_argument("--list", action="store_true", help="list config names and exit")
    ap.add_argument("--config-file", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--ampl", type=Path,
                    default=Path(os.environ.get("AMPL_BIN", DEFAULT_AMPL)))
    args = ap.parse_args()

    doc = yaml.safe_load(args.config_file.read_text())
    defaults, configs = doc["defaults"], doc["configs"]

    if args.list:
        for name, cfg in configs.items():
            print(f"  {name:<26} {cfg.get('description', '').strip()}")
        return 0

    assert args.ampl.exists(), f"ampl binary not found at {args.ampl} (set --ampl or $AMPL_BIN)"

    if args.all:
        names = list(configs)
    elif args.config:
        names = [args.config]
    else:
        ap.error("give --config <name>, --all, or --list")

    rows = [solve_config(n, defaults, configs, args.ampl) for n in names]
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
