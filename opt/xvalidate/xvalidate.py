#!/usr/bin/env python3
"""Cross-validate the 3d_memory subarray timing/energy model vs DESTINY.

Single-subarray only: DESTINY is forced to 1x1 bank / 1x1 mat / one subarray of
exactly ``numRow x numColumn`` sensing ``b_acc`` bits/access, so mats=banks=1 and
its H-tree collapses to a small "Non-H-Tree" residual (reported separately as
overhead the model omits).

For each (config, geometry) point:
  1. generate a DESTINY .cell + .cfg, run it, parse the per-component tree;
  2. forward-evaluate opt/model.py's develop/energy expressions at the SAME
     (N_BL=numColumn, N_WL=numRow, b_acc, N_share=1);
  3. print a per-component side-by-side with ratios; dump comparison.csv.

Component mapping (see tables printed at runtime):
  LATENCY  decode      DESTINY Predecoder + Row Decoder   <-> model t_dec + t_WL
           bitline     DESTINY Bitline                    <-> model t_BL
           senseamp    DESTINY Senseamp                   <-> model t_SA
           mux         DESTINY Mux (+ Precharge off-path) <-> model t_sw (mostly unmodeled)
           TOTAL       DESTINY Predecoder + Subarray      <-> model sum_dev
  ENERGY   bitline+cell DESTINY Subarray - periph leaves  <-> model k_col*N_BL + k_arr*cells
           periph       DESTINY RowDec+MuxDec+SA+Mux+Prech <-> model e_periph
           TOTAL        DESTINY Subarray Dynamic Energy    <-> model E_access

Run (needs the opt venv for pyyaml/pyomo-backed model import):
  opt/.venv/bin/python opt/xvalidate/xvalidate.py            # default sweep, all mapped configs
  opt/.venv/bin/python opt/xvalidate/xvalidate.py --configs sram_16nm reram_16nm
"""
import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
MEMROOT = OPT.parent.parent                       # .../memory
DESTINY_DIR = MEMROOT / "destiny_3d_cache"
DESTINY_BIN = DESTINY_DIR / "destiny"

sys.path.insert(0, str(OPT))
from model import TechSpec, develop_coeffs, energy_coeffs  # noqa: E402
from tech_map import build_cell_params, cell_text, CONFIG_TO_DTYPE  # noqa: E402

# Default geometry sweep: numColumn = b_acc * muxSenseAmp (mux a power of two);
# capacity_bits = numRow*numColumn is an integer #KB for all of these.
B_ACC = 64
DEFAULT_COLS = [256, 1024]
DEFAULT_ROWS = [256, 512, 1024, 2048, 4096]
NODE_NM = 22

# Device roadmap per DESTINY cell family. HP transistors leak too much for a
# tall bitline of leaky/passive cells (BITLINE_LEAKAGE_TOLERANCE=1), so dense
# memory realistically uses low-leakage devices -- LSTP for charge-share and
# resistive arrays, HP only for the SRAM latch (matches DESTINY sample configs).
DTYPE_ROADMAP = {"SRAM": "HP", "eDRAM": "EDRAM", "memristor": "LSTP",
                 "MRAM": "HP", "PCRAM": "HP"}
# eDRAM has a dedicated device technology in DESTINY that is only modeled at the
# 32/45 nm nodes; the others run at the 22 nm cross-check node.
DTYPE_NODE = {"eDRAM": 32}


# --------------------------------------------------------------------------- #
# config loading (deep-merge onto defaults, same rule as run_bw_max.py)
# --------------------------------------------------------------------------- #
def load_configs(config_file: Path):
    doc = yaml.safe_load(config_file.read_text())
    return doc["defaults"], doc.get("configs", {})


def merged_tech(defaults: dict, configs: dict, name: str) -> dict:
    """Deep-merge one config's technology block onto defaults.technology."""
    tech = dict(defaults["technology"])
    if name != "__defaults__":
        assert name in configs, f"unknown config {name!r}"
        tech.update((configs[name] or {}).get("technology", {}) or {})
    return tech


# --------------------------------------------------------------------------- #
# DESTINY: generate, run, parse
# --------------------------------------------------------------------------- #
CFG_TEMPLATE = """\
-DesignTarget: RAM
-ProcessNode: {node}
-Capacity (KB): {cap_kb}
-WordWidth (bit): {b_acc}
-DeviceRoadmap: {roadmap}
-Routing: Non-H-tree
-InternalSensing: true
-MemoryCellInputFile: {cell_name}
-Temperature (K): {temp}
{retention}-OptimizationTarget: ReadLatency
-EnablePruning: No
-BufferDesignOptimization: latency
-ForceBank (Total AxB, Active CxD): 1x1, 1x1
-ForceMat (Total AxB, Active CxD): 1x1, 1x1
-ForceMuxSenseAmp: {mux_sa}
-ForceMuxOutputLev1: 1
-ForceMuxOutputLev2: 1
-StackedDieCount: 1
-MonolithicStackCount: 1
"""

_UNIT_TIME = {"s": 1e12, "ms": 1e9, "us": 1e6, "ns": 1e3, "ps": 1.0}   # -> ps
_UNIT_NRG = {"J": 1e15, "mJ": 1e12, "uJ": 1e9, "nJ": 1e6, "pJ": 1e3, "fJ": 1.0}  # -> fJ
_LINE = re.compile(r"---\s*(.+?)\s*=\s*([-\d.eE]+)\s*([a-zA-Z]+)")


def _to_ps(val, unit):
    return float(val) * _UNIT_TIME[unit]


def _to_fj(val, unit):
    return float(val) * _UNIT_NRG[unit]


def run_destiny(work: Path, config_name: str, tech: dict, n_row: int, n_col: int, temp: float):
    """Write .cell/.cfg for one forced subarray, run DESTINY, parse its tree.

    Returns a dict of parsed latencies [ps] / energies [fJ] plus the geometry
    DESTINY actually built (so a mismatch vs the request is caught loudly).
    """
    assert n_col % B_ACC == 0, f"numColumn {n_col} must be a multiple of b_acc {B_ACC}"
    mux_sa = n_col // B_ACC
    assert mux_sa & (mux_sa - 1) == 0, f"muxSenseAmp {mux_sa} must be a power of two"
    cap_bits = n_row * n_col
    assert cap_bits % 8192 == 0, f"{n_row}x{n_col} is not an integer #KB"
    cap_kb = cap_bits // 8192

    cell, dtype = build_cell_params(config_name, tech)
    stem = f"{config_name}_{n_row}x{n_col}"
    cell_name = f"{stem}.cell"
    retention = "-RetentionTime (us): 40\n" if dtype == "eDRAM" else ""
    (work / cell_name).write_text(cell_text(cell))
    (work / f"{stem}.cfg").write_text(CFG_TEMPLATE.format(
        node=DTYPE_NODE.get(dtype, NODE_NM), cap_kb=cap_kb, b_acc=B_ACC, cell_name=cell_name,
        temp=int(temp), mux_sa=mux_sa, roadmap=DTYPE_ROADMAP[dtype], retention=retention))

    proc = subprocess.run([str(DESTINY_BIN), f"{stem}.cfg"], cwd=work,
                          capture_output=True, text=True, timeout=180)
    out = proc.stdout
    if "Subarray Latency" not in out or "invalid" in out.lower():
        raise RuntimeError(f"DESTINY produced no valid subarray for {stem}:\n{out[-800:]}")

    # geometry actually built
    m = re.search(r"Subarray Size\s*:\s*(\d+)\s*Rows?\s*x\s*(\d+)\s*Columns?", out)
    assert m, f"could not parse subarray size for {stem}"
    got_row, got_col = int(m.group(1)), int(m.group(2))
    assert (got_row, got_col) == (n_row, n_col), \
        f"{stem}: DESTINY built {got_row}x{got_col}, requested {n_row}x{n_col}"

    # per-component leaves (read side). Parse only the Timing+Power read blocks.
    read_block = out[out.index("Timing:"):out.index("Finished!")]
    d = {}
    for label, val, unit in _LINE.findall(read_block):
        key = label.strip()
        if unit in _UNIT_TIME:
            d.setdefault("t_" + key, _to_ps(val, unit))
        elif unit in _UNIT_NRG:
            d.setdefault("e_" + key, _to_fj(val, unit))
    return d, dtype


def destiny_components(d: dict) -> dict:
    """Collapse DESTINY leaves into the comparison buckets (ps / fJ)."""
    t_decode = d.get("t_Predecoder Latency", 0.0) + d.get("t_Row Decoder Latency", 0.0)
    t_bl = d.get("t_Bitline Latency", 0.0)
    t_sa = d.get("t_Senseamp Latency", 0.0)
    t_mux = d.get("t_Mux Latency", 0.0)
    t_prech = d.get("t_Precharge Latency", 0.0)                 # off critical path
    t_total = d.get("t_Predecoder Latency", 0.0) + d.get("t_Subarray Latency", 0.0)

    e_sub = d["e_Subarray Dynamic Energy"]
    e_periph = (d.get("e_Row Decoder Dynamic Energy", 0.0)
                + d.get("e_Mux Decoder Dynamic Energy", 0.0)
                + d.get("e_Senseamp Dynamic Energy", 0.0)
                + d.get("e_Mux Dynamic Energy", 0.0)
                + d.get("e_Precharge Dynamic Energy", 0.0))
    e_bitline = e_sub - e_periph                                # bitline+cell CV^2 remainder
    return dict(t_decode=t_decode, t_bitline=t_bl, t_senseamp=t_sa, t_mux=t_mux,
                t_precharge=t_prech, t_total=t_total,
                e_bitline=e_bitline, e_periph=e_periph, e_total=e_sub,
                t_nonhtree=d.get("t_Non-H-Tree Latency", 0.0),
                e_nonhtree=d.get("e_Non-H-Tree Dynamic Energy", 0.0))


# --------------------------------------------------------------------------- #
# model.py forward evaluation at a fixed geometry (N_share = N_indep = 1)
# --------------------------------------------------------------------------- #
def model_components(tech: TechSpec, n_bl: int, n_wl: int, b_acc: int) -> dict:
    """Evaluate the model's develop/energy expressions in ps / fJ.

    Mirrors build_model(): sum_dev = t_dec + t_WL + t_BL + t_SA + t_sw; with a
    single subarray (N_share=1) t_cycle=sum_dev and BW=b_acc/sum_dev.
    """
    f_margin, a_lin, a_quad = develop_coeffs(tech)
    t_dec = tech.k_dec * math.log(n_wl) / math.log(2)
    t_wl = tech.k_wire_WL * n_bl**2 + tech.k_cell_WL * n_bl
    t_bl_rc = f_margin * (a_lin * n_wl + a_quad * n_wl**2)         # intrinsic, slew-free
    t_bl = math.sqrt(t_bl_rc**2 + tech.c_slew * t_bl_rc * t_wl)    # Horowitz WL-slew coupling
    t_sa = tech.t_SA0 + tech.destructive * tech.t_restore
    t_sw = tech.t_sw
    sum_dev = t_dec + t_wl + t_bl + t_sa + t_sw
    NS = 1e3  # ns -> ps

    k_col, k_arr = energy_coeffs(tech)
    e_bitline = k_col * n_bl + k_arr * (n_bl * n_wl)            # fJ
    e_periph = (tech.e_periph + tech.e_periph_col * n_bl
                + tech.e_sa_read * b_acc                    # sense amp sits in DESTINY's periph bucket
                + tech.write_fraction * tech.e_write_cell * b_acc)
    e_total = e_bitline + e_periph

    # charge-share row-collapse cap: flag geometries the model would forbid.
    nwl_cap = None
    if tech.sense_mode == "charge_share":
        nwl_cap = tech.c_cell * (1.0 - tech.margin_sa) / (tech.c_bl * tech.margin_sa)
    return dict(t_decode=(t_dec + t_wl) * NS, t_bitline=t_bl * NS, t_senseamp=t_sa * NS,
                t_mux=t_sw * NS, t_precharge=0.0, t_total=sum_dev * NS,
                e_bitline=e_bitline, e_periph=e_periph, e_total=e_total,
                bw_bit_per_ns=b_acc / sum_dev, nwl_cap=nwl_cap)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _ratio(dst, mdl):
    return "n/a" if dst == 0 else f"{mdl / dst:6.2f}x"


def print_point(name, dtype, n_row, n_col, dcmp, mcmp):
    print(f"\n=== {name}  [DESTINY cell: {dtype}]   subarray {n_row} rows x {n_col} cols "
          f"(b_acc={B_ACC}, mux={n_col // B_ACC}) ===")
    if mcmp.get("nwl_cap") is not None and n_row > mcmp["nwl_cap"]:
        print(f"    !! model charge-share signal collapse: N_WL cap = {mcmp['nwl_cap']:.0f} "
              f"< {n_row} rows (model would FORBID this geometry)")
    print(f"  {'LATENCY [ps]':<16}{'DESTINY':>12}{'model':>12}{'model/DST':>12}")
    for k, lbl in [("t_decode", "decode+WL"), ("t_bitline", "bitline"),
                   ("t_senseamp", "senseamp"), ("t_mux", "mux/sw"), ("t_total", "TOTAL(dev)")]:
        print(f"  {lbl:<16}{dcmp[k]:>12.2f}{mcmp[k]:>12.2f}{_ratio(dcmp[k], mcmp[k]):>12}")
    print(f"  {'(precharge*)':<16}{dcmp['t_precharge']:>12.2f}{'--':>12}{'off-path':>12}")
    print(f"  {'(non-htree*)':<16}{dcmp['t_nonhtree']:>12.2f}{'--':>12}{'overhead':>12}")
    print(f"  {'ENERGY [fJ]':<16}{'DESTINY':>12}{'model':>12}{'model/DST':>12}")
    for k, lbl in [("e_bitline", "bitline+cell"), ("e_periph", "periph"), ("e_total", "TOTAL(access)")]:
        print(f"  {lbl:<16}{dcmp[k]:>12.1f}{mcmp[k]:>12.1f}{_ratio(dcmp[k], mcmp[k]):>12}")
    ebit_d = dcmp["e_total"] / B_ACC
    ebit_m = mcmp["e_total"] / B_ACC
    print(f"  {'per-bit':<16}{ebit_d:>12.2f}{ebit_m:>12.2f}{_ratio(ebit_d, ebit_m):>12}")
    print(f"  {'(non-htree E*)':<16}{dcmp['e_nonhtree']:>12.1f}{'--':>12}{'overhead':>12}")


CSV_COLS = ["config", "dtype", "n_row", "n_col", "b_acc",
            "t_decode", "t_bitline", "t_senseamp", "t_mux", "t_total",
            "e_bitline", "e_periph", "e_total"]


def csv_rows(name, dtype, n_row, n_col, dcmp, mcmp):
    def row(src, c):
        return [name, dtype, n_row, n_col, B_ACC, src] + \
               [f"{c[k]:.4f}" for k in CSV_COLS[5:]]
    return [row("destiny", dcmp), row("model", mcmp)]


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", type=Path, default=OPT / "config.yaml")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="config names to test (default: all mapped ones present)")
    ap.add_argument("--rows", nargs="+", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--cols", nargs="+", type=int, default=DEFAULT_COLS)
    ap.add_argument("--work", type=Path, default=HERE / "_work")
    ap.add_argument("--csv", type=Path, default=HERE / "comparison.csv")
    args = ap.parse_args()

    assert DESTINY_BIN.exists(), f"destiny binary not found at {DESTINY_BIN}"
    args.work.mkdir(parents=True, exist_ok=True)
    defaults, configs = load_configs(args.config_file)

    if args.configs:
        names = args.configs
    else:
        names = [n for n in configs if n in CONFIG_TO_DTYPE]
    assert names, "no configs to test"

    header = "config,dtype,n_row,n_col,b_acc,source," + ",".join(CSV_COLS[5:])
    out_rows = [header]
    for name in names:
        assert name in CONFIG_TO_DTYPE, f"config {name!r} has no DESTINY cell mapping"
        tech_dict = merged_tech(defaults, configs, name)
        tech_spec = TechSpec(**tech_dict)
        temp = 350
        for n_col in args.cols:
            for n_row in args.rows:
                try:
                    draw, dtype = run_destiny(args.work, name, tech_dict, n_row, n_col, temp)
                except (RuntimeError, AssertionError) as e:
                    print(f"\n=== {name}  {n_row}x{n_col}: SKIPPED ({str(e).splitlines()[0]})")
                    continue
                dcmp = destiny_components(draw)
                mcmp = model_components(tech_spec, n_col, n_row, B_ACC)
                print_point(name, dtype, n_row, n_col, dcmp, mcmp)
                out_rows += [",".join(map(str, r)) for r in
                             csv_rows(name, dtype, n_row, n_col, dcmp, mcmp)]

    args.csv.write_text("\n".join(out_rows) + "\n")
    print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
