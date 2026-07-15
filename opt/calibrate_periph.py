#!/usr/bin/env python3
"""Calibrate 3d_memory peripheral-volume coefficients from destiny_3d_cache.

The phase-1 BW-max model (opt/model.py) charges each peripheral circuit block
against the volume budget with its own scaling law (v_wldrv per row, v_sa0 per
sensed bit, v_pre/v_pass per column, v_periph fixed per array). This script
derives those coefficients from the ``destiny_3d_cache`` submodule
(memory/3d_memory/destiny_3d_cache) -- the same DESTINY build used as the
cross-validation reference in opt/xvalidate.

Method: force DESTINY to a single ``n_row x n_col`` subarray with one sense amp
per column (1x1 bank / 1x1 mat / muxSenseAmp=1, i.e. WordWidth = numColumn), sweep
the rows/cols, and linearly regress each component's area to separate the per-row
/ per-column SLOPE from the fixed (predecoder / control) INTERCEPT. DESTINY emits
the per-component subarray areas on stderr (``PERIPHFIT ...``, added via
SubArray::PrintAreaBreakdown so the normal stdout report stays untouched). Areas
are normalized to F^2 (F = ProcessNode, node-independent) and then to 16 nm volume
via the tool convention  v = area_F2 * (16e-3 um)^2 * t_layer(1 um)  -- the same
currency as v_cell.

The CFG/cell generation is shared with opt/xvalidate (imported below), so a cell
family DESTINY lays out (SRAM latch, charge-share DRAM) calibrates the CMOS
periphery shared across cell techs (the config `defaults`). Resistive SET/RESET
write drivers, HV pass gates, and current-sense IV converters are NOT laid out by
this SRAM sweep and stay EMBER/DESTINY-anchored in the per-type config overrides.

Run:  opt/.venv/bin/python opt/calibrate_periph.py
(non-mutating: writes scratch .cell/.cfg into the gitignored opt/xvalidate/_work).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE
MEMROOT = OPT.parent.parent                                  # .../memory
DESTINY_DIR = MEMROOT / "3d_memory" / "destiny_3d_cache"     # the git submodule
DESTINY_BIN = DESTINY_DIR / "destiny"
WORK = OPT / "xvalidate" / "_work"                           # gitignored scratch dir

# Reuse xvalidate's proven DESTINY cfg/cell generation and config loading.
sys.path.insert(0, str(OPT))
sys.path.insert(0, str(OPT / "xvalidate"))
from xvalidate import (CFG_TEMPLATE, DTYPE_ROADMAP, DTYPE_NODE, NODE_NM,  # noqa: E402
                       load_configs, merged_tech)
from tech_map import build_cell_params, cell_text  # noqa: E402

CONV16 = (16e-3) ** 2 * 1.0          # um^3 per F^2 at 16 nm, t_layer = 1 um (= 2.56e-4)
# DESTINY rejects tall SRAM bitlines (>512 rows) and very wide words (>1024 cols)
# as invalid, so the row/column levers use different, separately-valid ranges.
ROW_SIZES = [64, 128, 256, 512]      # sweep rows @ cols=256
COL_SIZES = [128, 256, 512, 1024]    # sweep cols @ rows=256
_PERIPHFIT = re.compile(
    r"PERIPHFIT numRow=(\d+) numColumn=(\d+) subarray_um2=(\S+) "
    r"cells_um2=(\S+) rowdec_um2=(\S+) senseamp_um2=(\S+)")


def run(config_name, tech, dtype, node, n_row, n_col):
    """Force one n_row x n_col subarray (one SA per column) in destiny_3d_cache,
    parse the stderr PERIPHFIT per-component areas [um^2]."""
    assert n_row * n_col % 8192 == 0, f"{n_row}x{n_col} is not an integer #KB"
    cap_kb = n_row * n_col // 8192
    b_acc = n_col                                       # WordWidth = columns => muxSenseAmp = 1
    stem = f"calib_{config_name}_{n_row}x{n_col}"
    cell_name = f"{stem}.cell"
    cell, _ = build_cell_params(config_name, tech)
    (WORK / cell_name).write_text(cell_text(cell))
    (WORK / f"{stem}.cfg").write_text(CFG_TEMPLATE.format(
        node=node, cap_kb=cap_kb, b_acc=b_acc, cell_name=cell_name,
        temp=300, mux_sa=1, roadmap=DTYPE_ROADMAP[dtype], retention=""))

    proc = subprocess.run([str(DESTINY_BIN), f"{stem}.cfg"], cwd=WORK,
                          capture_output=True, text=True, timeout=180)
    hits = [m for m in _PERIPHFIT.findall(proc.stderr)
            if (int(m[0]), int(m[1])) == (n_row, n_col)]
    assert hits, (f"{stem}: no PERIPHFIT for {n_row}x{n_col} "
                  f"(built {[(m[0], m[1]) for m in _PERIPHFIT.findall(proc.stderr)]}); "
                  f"stdout tail:\n{proc.stdout[-400:]}")
    total, cells, rowdec, sa = (float(x) for x in hits[0][2:])
    return dict(total=total, cells=cells, rowdec=rowdec, sa=sa,
                blstrip=total - cells - rowdec - sa)


def slope_intercept(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return slope, (sy - slope * sx) / n


def fit(config_name, defaults, configs):
    tech = merged_tech(defaults, configs, config_name)
    _, dtype = build_cell_params(config_name, tech)
    node = DTYPE_NODE.get(dtype, NODE_NM)               # ProcessNode = feature size F [nm]

    row_runs = [(r, run(config_name, tech, dtype, node, r, 256)) for r in ROW_SIZES]
    col_runs = [(c, run(config_name, tech, dtype, node, 256, c)) for c in COL_SIZES]

    m_wl, b_wl = slope_intercept([r for r, _ in row_runs], [d["rowdec"] for _, d in row_runs])
    m_sa, _ = slope_intercept([c for c, _ in col_runs], [d["sa"] for _, d in col_runs])
    m_bl, b_bl = slope_intercept([c for c, _ in col_runs], [d["blstrip"] for _, d in col_runs])
    v_cell_bit = run(config_name, tech, dtype, node, 256, 256)["cells"] / (256 * 256)

    f_um2 = (node * 1e-3) ** 2                          # um^2 per F^2 at this node
    to_vol = lambda a: a / f_um2 * CONV16               # um^2/unit -> F^2/unit -> um^3/unit @16nm
    print(f"\n===== {config_name}  [DESTINY cell: {dtype} @ {node} nm] =====")
    print(f"  v_wldrv (per row)    = {to_vol(m_wl):8.4f} um^3   ({m_wl / f_um2:8.0f} F^2)")
    print(f"  v_sa0   (per col)    = {to_vol(m_sa):8.4f} um^3   ({m_sa / f_um2:8.0f} F^2)  [voltage latch]")
    print(f"  v_pre+v_pass(per col)= {to_vol(m_bl):8.4f} um^3   ({m_bl / f_um2:8.0f} F^2)")
    print(f"  v_periph(fixed/array)= {to_vol(max(0.0, b_bl)):8.4f} um^3   [BL-strip intercept]")
    print(f"  v_cell  (per bit)    = {to_vol(v_cell_bit):8.5f} um^3   ({v_cell_bit / f_um2:6.1f} F^2)  [cross-check]")


if __name__ == "__main__":
    assert DESTINY_BIN.exists(), f"destiny binary not found at {DESTINY_BIN} (run `make` in the submodule)"
    WORK.mkdir(parents=True, exist_ok=True)
    defaults, configs = load_configs(OPT / "config.yaml")
    for cfg in ("sram_16nm", "gaincell_100Mb"):
        fit(cfg, defaults, configs)
