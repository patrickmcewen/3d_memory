#!/usr/bin/env python3
"""Calibrate 3d_memory peripheral-volume coefficients from the memory_model tool.

The phase-1 BW-max model (opt/model.py) charges each peripheral circuit block
against the volume budget with its own scaling law (v_wldrv per row, v_sa0 per
sensed bit, v_pre/v_pass per column, v_periph fixed per array). This script
derives those coefficients from the sibling ``memory_model`` subarray tool
(memory/memory_model), which lays out the real NVSim/DESTINY-style periphery.

Method: sweep the subarray rows/cols, parse each component's area, and linearly
regress it to separate the per-row / per-column SLOPE from the fixed (predecoder
/ control) INTERCEPT. Areas are normalized to F^2 (F = Lgate, node-independent)
and then to 16 nm volume via the tool convention  v = area_F2 * (16e-3 um)^2 *
t_layer(1 um)  -- the same currency as v_cell.

memory_model only lays out SRAM_GEN / gcDRAM_GEN, so this calibrates the CMOS
periphery shared across cell techs (the config `defaults`). Resistive SET/RESET
write drivers, HV pass gates, and current-sense IV converters are NOT modeled by
memory_model and stay EMBER/DESTINY-anchored in the per-type config overrides.

Run:  opt/.venv/bin/python opt/calibrate_periph.py
(non-mutating: restores the swept configs to 256x256 on exit).
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.normpath(os.path.join(HERE, "..", "..", "memory_model"))  # sibling tool
F_NM = 40.0                          # GENFET example Lgate (the model node)
F_UM2 = (F_NM * 1e-3) ** 2           # um^2 per F^2 at that node
CONV16 = (16e-3) ** 2 * 1.0          # um^3 per F^2 at 16 nm, t_layer = 1 um (= 2.56e-4)
SIZES = [128, 256, 512, 1024]


def run(cfg, rows, cols):
    """Set the subarray geometry, run memory_model, parse component areas [um^2]."""
    path = os.path.join(MM, "config", cfg)
    txt = open(path).read()
    txt = re.sub(r"-SubarrayRows: \d+", f"-SubarrayRows: {rows}", txt)
    txt = re.sub(r"-SubarrayColumns: \d+", f"-SubarrayColumns: {cols}", txt)
    open(path, "w").write(txt)
    out = subprocess.run([os.path.join(MM, "memory_model"), os.path.join("config", cfg)],
                         cwd=MM, capture_output=True, text=True).stdout

    def dims(pat):
        m = re.search(pat, out)
        assert m, f"could not parse '{pat}' from memory_model output"
        return float(m.group(1)) * float(m.group(2))

    total = dims(r"Area = ([\d.]+)um x ([\d.]+)um")
    cells = dims(r"lenWordline \* lenBitline = ([\d.]+)um \* ([\d.]+)um")
    rowdec = dims(r"Row Decoder Area:([\d.]+)um x ([\d.]+)um")
    sa = dims(r"Sense Amplifier Area:([\d.]+)um x ([\d.]+)um")
    return dict(total=total, cells=cells, rowdec=rowdec, sa=sa,
                blstrip=total - cells - rowdec - sa)


def slope_intercept(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return slope, (sy - slope * sx) / n


def fit(cfg):
    row_runs = [(r, run(cfg, r, 256)) for r in SIZES]   # sweep rows @ cols=256
    col_runs = [(c, run(cfg, 256, c)) for c in SIZES]   # sweep cols @ rows=256

    m_wl, b_wl = slope_intercept([r for r, _ in row_runs], [d["rowdec"] for _, d in row_runs])
    m_sa, _ = slope_intercept([c for c, _ in col_runs], [d["sa"] for _, d in col_runs])
    m_bl, b_bl = slope_intercept([c for c, _ in col_runs], [d["blstrip"] for _, d in col_runs])
    v_cell_bit = run(cfg, 256, 256)["cells"] / (256 * 256)

    to_vol = lambda a: a / F_UM2 * CONV16       # um^2/unit -> F^2/unit -> um^3/unit @16nm
    print(f"\n===== {cfg} =====")
    print(f"  v_wldrv (per row)    = {to_vol(m_wl):8.4f} um^3   ({m_wl / F_UM2:8.0f} F^2)")
    print(f"  v_sa0   (per col)    = {to_vol(m_sa):8.4f} um^3   ({m_sa / F_UM2:8.0f} F^2)  [voltage latch]")
    print(f"  v_pre+v_pass(per col)= {to_vol(m_bl):8.4f} um^3   ({m_bl / F_UM2:8.0f} F^2)")
    print(f"  v_periph(fixed/array)= {to_vol(max(0.0, b_bl)):8.4f} um^3   [BL-strip intercept]")
    print(f"  v_cell  (per bit)    = {to_vol(v_cell_bit):8.5f} um^3   ({v_cell_bit / F_UM2:6.1f} F^2)  [cross-check]")


if __name__ == "__main__":
    for cfg in ("sram_subarray.cfg", "gaincell_subarray.cfg"):
        fit(cfg)
    for cfg in ("sram_subarray.cfg", "gaincell_subarray.cfg"):
        run(cfg, 256, 256)          # restore
    print("\nconfigs restored to 256x256")
