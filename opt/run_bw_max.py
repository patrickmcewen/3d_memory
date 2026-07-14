#!/usr/bin/env python3
"""Run the 3d_memory phase-1 BW-max MINLP for a YAML-selected config.

Reads opt/config.yaml, deep-merges a named config onto `defaults`, builds the
Pyomo model in opt/model.py, and solves it with the bundled StreamHLS Gurobi
ASL driver. Generated artifacts (.nl / .sol / solver log) land in opt/build/
(gitignored).

Examples:
    python opt/run_bw_max.py --list
    python opt/run_bw_max.py --config dram_100Mb
    python opt/run_bw_max.py --all
"""
import argparse
import math
import os
import re
import sys
from pathlib import Path

import yaml
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import TerminationCondition as TC

from model import ProblemSpec, TechSpec, Bounds, build_model, energy_coeffs

HERE = Path(__file__).resolve().parent          # .../3d_memory/opt
BUILD = HERE / "build"

# Default location of the bundled Gurobi 13 ASL driver. Its $ORIGIN rpath
# resolves libgurobi130.so, so no LD_LIBRARY_PATH is needed. Override with
# --gurobi or $GUROBI_ASL.
DEFAULT_GUROBI = Path(
    "/nfs/pool0/pmcewen/rsgvm13dir/codesign2/Stream-HLS/ampl.linux-intel64/gurobi"
)

# Every param the model declares, by block. Used to assert a config is complete
# (fail loud on a missing/extra key) before constructing the dataclass specs.
PARAMS = {
    "problem": ["C", "A", "L", "t_layer", "P_max"],
    "technology": ["k_dec", "k_wire_WL", "k_cell_WL", "t_SA0", "t_restore",
                   "destructive", "t_sw", "v_cell", "v_sa0", "k_vdec", "v_sel",
                   "v_periph", "sense_mode", "settle_frac", "c_bl", "r_bl",
                   "i_read", "n_ut", "r_pullup", "v_ratio", "c_cell", "margin_sa",
                   "v_read", "v_sense", "e_periph", "e_write_cell", "p_leak_bit",
                   "write_fraction"],
    "bounds": ["NBL_min", "NBL_max", "NWL_min", "NWL_max",
               "Nshare_min", "Nshare_max", "Nindep_max", "BW_max"],
}
# Params that are NOT numeric (skip float() coercion in make_specs).
STRING_PARAMS = {"sense_mode"}
# fix-key -> (min_param, max_param) bound pair it collapses.
FIXABLE = {
    "N_BL":    ("NBL_min", "NBL_max"),
    "N_WL":    ("NWL_min", "NWL_max"),
    "N_share": ("Nshare_min", "Nshare_max"),
}

# Terminations that may carry a usable (possibly non-proven-optimal) incumbent.
SOLVED = {TC.optimal, TC.locallyOptimal, TC.feasible, TC.maxTimeLimit,
          TC.maxIterations}


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


def make_specs(merged: dict):
    """Turn a merged config dict into (ProblemSpec, TechSpec, Bounds).

    Assert each block holds exactly the params the model needs -- an extra or
    missing key is a config bug, so fail loud rather than silently dropping it.
    """
    specs = {}
    for block, keys in PARAMS.items():
        # PyYAML reads unsigned-exponent literals like "1.0e9" as strings; every
        # numeric param is coerced to float (dataclasses re-narrow the integer
        # ones L/Nshare_* in __post_init__). String params (sense_mode) pass through.
        have = {k: (v if k in STRING_PARAMS else float(v))
                for k, v in merged[block].items()}
        assert set(have) == set(keys), (
            f"config block '{block}' key mismatch: "
            f"missing={sorted(set(keys) - set(have))}, extra={sorted(set(have) - set(keys))}"
        )
        specs[block] = have
    return (ProblemSpec(**specs["problem"]),
            TechSpec(**specs["technology"]),
            Bounds(**specs["bounds"]))


def make_solver(merged: dict, gurobi_asl: Path):
    """Build the ASL/Gurobi solver interface with the config's solver knobs."""
    opt = pyo.SolverFactory("asl", executable=str(gurobi_asl))
    opt.options["solver"] = "gurobi"        # names the driver's option namespace
    for k, v in (merged["solver"] or {}).items():
        opt.options[k] = v
    return opt


# Fields lifted straight out of the solved model (Var or Expression); pyo.value
# works on both. Keep this list == report layout below.
VALUE_KEYS = ["N_BL", "N_WL", "b_acc", "N_share", "N_indep",
              "t_cycle", "BW", "t_dec", "t_WL", "t_BL", "t_SA", "sum_dev",
              "cells_arr", "N_tot", "N_SA", "total_cells", "cells_x_indep",
              "sel_term", "E_bit", "P_dyn"]


def collect_values(m, problem: ProblemSpec, tech: TechSpec) -> dict:
    """Dict of solved values plus derived report quantities."""
    d = {k: pyo.value(getattr(m, k)) for k in VALUE_KEYS}
    vol_used = (tech.v_cell * d["total_cells"] + tech.v_sa0 * d["N_SA"]
                + tech.k_vdec * d["cells_x_indep"] + tech.v_sel * d["sel_term"]
                + tech.v_periph * d["N_tot"])
    d["vol_arrays"] = tech.v_cell * d["total_cells"]
    d["vol_sas"] = tech.v_sa0 * d["N_SA"]
    d["vol_dec"] = tech.k_vdec * d["cells_x_indep"]
    d["vol_sel"] = tech.v_sel * d["sel_term"]
    d["vol_periph"] = tech.v_periph * d["N_tot"]
    d["vol_used"] = vol_used
    d["vol_budget"] = problem.vol_budget
    d["vol_pct"] = 100 * vol_used / problem.vol_budget
    d["C"] = problem.C
    d["t_sw"] = tech.t_sw
    d["sense_mode"] = tech.sense_mode
    # Power density [uW/um^2 == W/mm^2] over the shared 3D footprint A_used.
    d["A_used"] = vol_used / (problem.L * problem.t_layer)
    d["P_leak"] = tech.p_leak_bit * d["total_cells"]
    d["p_density"] = (d["P_dyn"] + d["P_leak"]) / d["A_used"]
    d["P_max"] = problem.P_max
    d["p_density_pct"] = 100 * d["p_density"] / problem.P_max
    # Per-access dynamic-energy breakdown (the four E_access terms; see
    # energy_coeffs / build_model). Sum == E_access == E_bit * b_acc.
    k_col, k_arr = energy_coeffs(tech)
    d["E_access"] = d["E_bit"] * d["b_acc"]                 # [fJ] per single-array access
    d["e_bitcell"] = k_col * d["N_BL"]                      # per-bitline cell/access CV^2
    d["e_blwire"] = k_arr * d["cells_arr"]                  # distributed BL wire CV^2 over the row
    d["e_periph"] = tech.e_periph                           # fixed decode/WL-drive energy
    d["e_write"] = tech.write_fraction * tech.e_write_cell * d["b_acc"]  # write-only per-cell term
    d["overfetch"] = d["N_BL"] / d["b_acc"]                 # K: whole row swings, only b_acc bits leave
    # Every pyo.Var value, keyed by name (declaration order) -- future-proof: new
    # model variables show up in the report without touching this function.
    d["all_vars"] = {var.name: pyo.value(var)
                     for var in m.component_data_objects(pyo.Var, active=True)}
    # Data-wire connection pitch: N_SA sense amps distributed over the chip
    # footprint A -> each owns A/N_SA area, so adjacent data wires sit sqrt() apart.
    d["conn_pitch"] = math.sqrt(problem.A / d["N_SA"])
    return d


def print_report(name: str, v: dict) -> None:
    """Design-point report -- mirrors the old report.run layout.

    BW is solved in bit/ns; 1 bit/ns = 1.25e8 B/s.
    """
    print("\n================= 3d_memory :: BW-max design point =================")
    print(f"  objective  BW        = {v['BW'] * 1.25e8:12.4g}  B/s   ({v['BW'] * 0.125e-3:.4g} TB/s)")
    print(f"  cycle time t_cycle    = {v['t_cycle']:12.4g}  ns")
    print("  ------------------------------------------------------------------")
    print(f"  array   N_BL x N_WL   = {v['N_BL']:8.1f} x {v['N_WL']:<8.1f}  (K = N_BL/b_acc = {v['N_BL'] / v['b_acc']:.2f})")
    print(f"  sense width b_acc     = {v['b_acc']:12.1f}  bit")
    print(f"  sense mode            = {v['sense_mode']:>12}")
    print(f"  sharing  N_share      = {v['N_share']:12.0f}")
    print(f"  indep sets N_indep    = {v['N_indep']:12.1f}")
    print(f"  sense amps N_SA       = {v['N_SA']:12.4g}")
    print("  ------------------------------------------------------------------")
    print(f"  t_dec / t_WL / t_BL   = {v['t_dec']:.4g} / {v['t_WL']:.4g} / {v['t_BL']:.4g} ns")
    print(f"  t_SA+t_sw (floor)     = {v['t_SA'] + v['t_sw']:.4g} ns   develop sum = {v['sum_dev']:.4g} ns")
    print(f"  single-array t_cycle  = {v['sum_dev']:12.4g}  ns   (before N_share amortization)")
    print("  ------------------------------------------------------------------")
    print(f"  volume used / budget  = {v['vol_used']:.4g} / {v['vol_budget']:.4g} um^3  ({v['vol_pct']:.1f}%)")
    print(f"    arrays   = {v['vol_arrays']:.4g}   SAs = {v['vol_sas']:.4g}   dec = {v['vol_dec']:.4g}   sel = {v['vol_sel']:.4g}   periph = {v['vol_periph']:.4g} um^3")
    print(f"    n_arrays (N_tot)      = {v['N_tot']:.4g}")
    print(f"  capacity  stored/target = {v['total_cells']:.4g} / {v['C']:.4g} bit")
    print("  ------------------------------------------------------------------")
    print(f"  energy/access E_access = {v['E_access']:.4g}  fJ   (overfetch K = N_BL/b_acc = {v['overfetch']:.2f})")
    print(f"    bitcell CV^2 = {v['e_bitcell']:.4g} fJ ({100 * v['e_bitcell'] / v['E_access']:.1f}%)"
          f"   BL wire = {v['e_blwire']:.4g} fJ ({100 * v['e_blwire'] / v['E_access']:.1f}%)")
    print(f"    periph       = {v['e_periph']:.4g} fJ ({100 * v['e_periph'] / v['E_access']:.1f}%)"
          f"   write   = {v['e_write']:.4g} fJ ({100 * v['e_write'] / v['E_access']:.1f}%)")
    print(f"  energy/bit E_bit      = {v['E_bit'] * 1e-3:12.4g}  pJ/bit  (= E_access / b_acc)")
    print(f"  power  dyn / leak     = {v['P_dyn'] * 1e-6:.4g} / {v['P_leak'] * 1e-6:.4g} W")
    print(f"  power density         = {v['p_density']:12.4g}  W/mm^2  ({v['p_density_pct']:.1f}% of P_max = {v['P_max']:.3g})")
    print("  ------------------------------------------------------------------")
    print(f"  data-wire conn pitch  = {v['conn_pitch']:12.4g}  um    (sqrt(A / N_SA))")
    print("  ------------------------------------------------------------------")
    print("  all model variables (pyo.Var):")
    for name, val in v['all_vars'].items():
        print(f"    {name:<16} = {val:.6g}")
    print("====================================================================")


def solve_config(name: str, defaults: dict, configs: dict, gurobi_asl: Path) -> dict:
    """Build, solve, and report one config. Returns a summary-row dict."""
    assert name in configs, f"unknown config '{name}' (available: {list(configs)})"
    merged = merge(defaults, configs[name])
    apply_fixes(merged)
    problem, tech, bounds = make_specs(merged)

    m = build_model(problem, tech, bounds)

    desc = configs[name].get("description", "").strip()
    print(f"\n########## config: {name} ##########")
    if desc:
        print(f"# {desc}")

    BUILD.mkdir(exist_ok=True)
    TempfileManager.tempdir = str(BUILD)     # keep .nl/.sol/.opt under build/
    opt = make_solver(merged, gurobi_asl)
    logfile = BUILD / f"{name}.log"
    results = opt.solve(m, load_solutions=False, tee=True, keepfiles=True,
                        symbolic_solver_labels=True, logfile=str(logfile))

    tc = results.solver.termination_condition
    solved = tc in SOLVED and len(results.solution) > 0
    row = {"config": name, "termination": str(tc)}
    if not solved:
        print(f"  [no incumbent: termination = {tc}]")
        row.update({k: None for k in ("BW", "t_cycle", "N_BL", "N_WL",
                                      "N_share", "vol_pct")})
        return row

    m.solutions.load_from(results)
    v = collect_values(m, problem, tech)
    print_report(name, v)
    row.update({"BW": v["BW"] * 1.25e8, "t_cycle": v["t_cycle"],
                "N_BL": v["N_BL"], "N_WL": v["N_WL"], "mode": v["sense_mode"],
                "N_share": v["N_share"], "vol_pct": v["vol_pct"],
                "gap": read_gap(logfile)})
    return row


def read_gap(logfile: Path):
    """Best-known optimality gap from the Gurobi log, or None."""
    if not logfile.exists():
        return None
    matches = re.findall(r"gap\s+([-\d.eE+]+)%", logfile.read_text())
    return float(matches[-1]) if matches else None


def print_summary(rows: list) -> None:
    if len(rows) < 2:
        return
    hdr = f"{'config':<26}{'BW[B/s]':>13}{'t_cyc[ns]':>11}{'N_BLxN_WL':>16}{'mode':>9}{'N_sh':>6}{'vol%':>7}"
    print("\n" + "=" * len(hdr))
    print("SUMMARY")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("BW") is None:
            print(f"{r['config']:<26}{'(' + r['termination'] + ')':>60}")
            continue
        geo = f"{r['N_BL']:.0f}x{r['N_WL']:.0f}"
        print(f"{r['config']:<26}{r['BW']:>13.4g}{r['t_cycle']:>11.4g}{geo:>16}"
              f"{r['mode']:>9}{r['N_share']:>6.0f}{r['vol_pct']:>7.1f}")
    print("=" * len(hdr))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="name of the config to solve")
    ap.add_argument("--all", action="store_true", help="solve every config in the file")
    ap.add_argument("--list", action="store_true", help="list config names and exit")
    ap.add_argument("--config-file", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--gurobi", type=Path,
                    default=Path(os.environ.get("GUROBI_ASL", DEFAULT_GUROBI)))
    args = ap.parse_args()

    doc = yaml.safe_load(args.config_file.read_text())
    defaults, configs = doc["defaults"], doc["configs"]

    if args.list:
        for name, cfg in configs.items():
            print(f"  {name:<26} {cfg.get('description', '').strip()}")
        return 0

    assert args.gurobi.exists(), f"gurobi ASL driver not found at {args.gurobi} (set --gurobi or $GUROBI_ASL)"

    if args.all:
        names = list(configs)
    elif args.config:
        names = [args.config]
    else:
        ap.error("give --config <name>, --all, or --list")

    rows = [solve_config(n, defaults, configs, args.gurobi) for n in names]
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
