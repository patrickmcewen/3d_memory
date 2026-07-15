#!/usr/bin/env python3
"""Calibrate 3d_memory peripheral-volume coefficients from destiny_3d_cache.

The phase-1 BW-max model (opt/model.py) charges each peripheral circuit block
against the volume budget with its own scaling law (v_wldrv per row, v_sa0 per
sensed bit, v_pre/v_pass per column, v_periph fixed per array). This script
derives those coefficients from the ``destiny_3d_cache`` submodule
(memory/3d_memory/destiny_3d_cache) -- the same DESTINY build opt/xvalidate uses.

Calibrated at the MODEL'S OPERATING REGIME, not an idealized single-subarray:
the model senses ``b_acc`` bits from ``N_BL`` columns, i.e. a column mux of
``N_BL/b_acc > 1``, and the optimizer favors wide-short arrays (N_BL >> N_WL).
So the sweep runs at ``b_acc = B_ACC`` (mux = n_col/b_acc, matching xvalidate),
and the WL-driver/row-decoder lever is fit at WIDE columns where its per-row area
has saturated. (An earlier revision fit at mux=1 / one-SA-per-column, which
under-counts the sense-amp and bl-strip area the model actually sees at mux>1 and
the WL-driver area at wide arrays -- see opt/xvalidate AREA section.)

Per-component subarray areas come from DESTINY's stderr ``PERIPHFIT`` line
(SubArray::PrintAreaBreakdown), parsed and normalized to 16 nm volume [um^3] by
xvalidate.run_destiny / destiny_components -- the same v_cell currency the volume
budget uses. The fit separates each block's per-row / per-column SLOPE from the
fixed (predecode/control) INTERCEPT (-> v_periph).

This SRAM sweep calibrates the CMOS periphery shared across cell techs (config
`defaults`). Resistive SET/RESET write drivers, HV pass gates, and current-sense
IV converters are NOT laid out by it and stay EMBER/DESTINY-anchored in the
per-type config overrides.

Run:  opt/.venv/bin/python opt/calibrate_periph.py
(non-mutating: writes scratch .cell/.cfg into the gitignored opt/xvalidate/_work).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE
MEMROOT = OPT.parent.parent                                  # .../memory
DESTINY_DIR = MEMROOT / "3d_memory" / "destiny_3d_cache"     # the git submodule
DESTINY_BIN = DESTINY_DIR / "destiny"
WORK = OPT / "xvalidate" / "_work"                           # gitignored scratch dir

# Reuse xvalidate's DESTINY runner (forces b_acc=B_ACC, mux=n_col/B_ACC) and its
# PERIPHFIT area parse + 16 nm-volume normalization (destiny_components).
sys.path.insert(0, str(OPT))
sys.path.insert(0, str(OPT / "xvalidate"))
from xvalidate import (run_destiny, destiny_components, B_ACC,  # noqa: E402
                       load_configs, merged_tech)

TEMP = 350                            # match xvalidate's operating temperature
FIXED_ROWS = 256                      # rows held fixed while sweeping columns
WIDE_COLS = 1024                      # WL driver saturates by ~512 cols; fit v_wldrv here
ROW_SIZES = [256, 512, 1024]          # sweep rows @ WIDE_COLS (DESTINY-valid range)
COL_SIZES = [128, 256, 512, 1024]     # sweep cols @ FIXED_ROWS (mux = cols/B_ACC)


def areas(name, tech, n_row, n_col):
    """Per-component subarray areas [um^3 @16nm] at the operating mux."""
    draw, _ = run_destiny(WORK, name, tech, n_row, n_col, TEMP)
    return destiny_components(draw)   # a_array / a_senseamp / a_decode / a_blstrip / a_total


def sa_read_energy(name, tech, n_row, n_col):
    """DESTINY voltage sense-amp read energy per SENSED bit [fJ/bit].

    DESTINY's Senseamp Dynamic Energy = capLoad*vdd^2*numSenseAmp
    (destiny_3d_cache SenseAmp::CalculatePower, SenseAmp.cpp:166-170), and the
    sweep forces numSenseAmp = numColumn/mux = b_acc, so it is a per-sensed-bit
    constant, geometry-independent -> per bit = e_SA / B_ACC. This is the model's
    e_sa_read for voltage latches (split out of the DESTINY periph bucket that
    e_periph is fit to; see opt/provenance.yaml energy_read_senseamp)."""
    draw, _ = run_destiny(WORK, name, tech, n_row, n_col, TEMP)
    return draw["e_Senseamp Dynamic Energy"] / B_ACC


def slope_intercept(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return slope, (sy - slope * sx) / n


def fit(name, defaults, configs):
    tech = merged_tech(defaults, configs, name)
    row_runs = [(r, areas(name, tech, r, WIDE_COLS)) for r in ROW_SIZES]
    col_runs = [(c, areas(name, tech, FIXED_ROWS, c)) for c in COL_SIZES]

    v_wldrv, _ = slope_intercept([r for r, _ in row_runs], [a["a_decode"] for _, a in row_runs])
    v_pre, v_periph = slope_intercept([c for c, _ in col_runs], [a["a_blstrip"] for _, a in col_runs])
    v_sa0 = sum(a["a_senseamp"] for _, a in col_runs) / len(col_runs) / B_ACC   # per sensed bit (constant)
    e_sa_read = sa_read_energy(name, tech, FIXED_ROWS, 256)                     # voltage SA read energy, per sensed bit
    a256 = areas(name, tech, 256, 256)
    v_cell = a256["a_array"] / (256 * 256)

    print(f"\n===== {name} (b_acc={B_ACC}, operating mux) =====")
    print(f"  v_sa0   (per sensed bit) = {v_sa0:8.4f} um^3   [voltage latch @ mux={256 // B_ACC}]")
    print(f"  v_pre+v_pass (per col)   = {v_pre:8.4f} um^3   [bl-strip slope]")
    print(f"  v_wldrv (per row)        = {v_wldrv:8.4f} um^3   [WL driver + row decoder @ {WIDE_COLS} cols]")
    print(f"  v_periph (fixed/array)   = {max(0.0, v_periph):8.4f} um^3   [bl-strip intercept]")
    print(f"  v_cell  (per bit)        = {v_cell:8.5f} um^3   [cross-check]")
    print(f"  e_sa_read (per sensed bit) = {e_sa_read:8.4f} fJ    [voltage SA CV^2; subtract {e_sa_read * B_ACC:.3g} fJ from e_periph intercept]")


if __name__ == "__main__":
    assert DESTINY_BIN.exists(), f"destiny binary not found at {DESTINY_BIN} (run `make` in the submodule)"
    WORK.mkdir(parents=True, exist_ok=True)
    defaults, configs = load_configs(OPT / "config.yaml")
    fit("sram_16nm", defaults, configs)
