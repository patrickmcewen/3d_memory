#!/usr/bin/env python3
"""Cross-validate the 3d_memory area model vs CACTI-3DD's 3D-DRAM stack.

Companion to xvalidate_cacti.py (2D subarray) and xvalidate.py (DESTINY). This
drives CACTI-3DD (../../cacti-3DD) in its 3D-DRAM mode -- a whole stacked-die
commodity-DRAM chip -- and cross-checks the ONE quantity the two models robustly
share at that level: **data-array area efficiency** (cell area / total array
area), i.e. how much the peripheral circuitry inflates the array. That directly
exercises the model's peripheral-volume coefficients (v_sa0, v_pre, v_pass,
v_wldrv, v_periph relative to v_cell) in a DRAM context.

Two hard limitations of THIS cacti-3DD build/model are honored, not papered over:
  * Chip-level timing/energy (t_RCD, t_RAS, read/refresh energy, TSV latency &
    energy) blow up to a ~1e107 sentinel -- a numerical bug in this build's TSV/
    membus delay path (uca.cc delay_TSV_tot). They are reported RAW and flagged
    UNRELIABLE; they are NOT compared. (Subarray-level area/geometry is sane.)
  * The 3d_memory model has NO TSV / power-delivery-via term yet (the PDN model
    is still proposed -- see the 3d_memory PDN-model design note). So CACTI-3DD's
    TSV area overhead is reported for REFERENCE / future PDN calibration only,
    with no model counterpart to divide against.

For each 3D point (capacity, stacked-die count, node) it runs CACTI-3DD unforced
(letting it pick a valid organization), reads back the subarray geometry + die
area efficiency it built, then evaluates opt/model.py (via xvalidate.
model_components) at that SAME subarray geometry and compares area efficiency.

Run:
  opt/.venv/bin/python opt/xvalidate/xvalidate_cacti3dd.py
  opt/.venv/bin/python opt/xvalidate/xvalidate_cacti3dd.py --config dram_cacti --dies 2 4 8
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
MEMROOT = OPT.parent.parent
CACTI3DD_DIR = MEMROOT / "cacti-3DD"
CACTI3DD_BIN = CACTI3DD_DIR / "cacti"

sys.path.insert(0, str(OPT))
sys.path.insert(0, str(HERE))
from model import TechSpec                                          # noqa: E402
from xvalidate import (load_configs, merged_tech, model_components,   # noqa: E402
                       _Tee, _ratio)

NODE_NM = 50            # CACTI-3DD's DRAM tech band (its sample point); comm-dram
BIGNUM = 1e30           # cacti-3DD const.h sentinel; parsed values above this are invalid
B_ACC = 64

# 3d_memory DRAM-family config -> CACTI-3DD comm-dram cell (only DRAM maps here).
CONFIG_TO_CELL = {"dram_cacti": "comm-dram", "dram_100Mb": "comm-dram",
                  "edram_dest": "lp-dram"}

# Full CACTI-3DD 3D-DRAM cfg; only the stack knobs are templated (rest is the
# sample's working boilerplate, incl. the DDR3 IO block this build requires).
CFG_3DD = """\
-size (Gb) {size_gb}
-block size (bytes) 128
-associativity 1
-read-write port 1
-exclusive read port 0
-exclusive write port 0
-single ended read ports 0
-UCA bank count {banks}
-technology (u) {node_um}
-burst length 4
-internal prefetch width 1
-Data array cell type - "{cell_type}"
-Data array peripheral type - "itrs-lstp"
-Tag array cell type - "itrs-hp"
-Tag array peripheral type - "itrs-hp"
-output/input bus width 64
-operating temperature (K) {temp}
-cache type "3D memory or 2D main memory"
-page size (bits) 8192
-burst depth 8
-IO width 4
-system frequency (MHz) 677
-stacked die count {stacked_die}
-partitioning granularity 0
-TSV projection 1
-tag size (b) "default"
-access mode (normal, sequential, fast) - "fast"
-design objective (weight delay, dynamic power, leakage power, cycle time, area) 0:0:0:0:100
-deviate (delay, dynamic power, leakage power, cycle time, area) 50:100000:100000:100000:1000000
-NUCAdesign objective (weight delay, dynamic power, leakage power, cycle time, area) 0:0:0:0:100
-NUCAdeviate (delay, dynamic power, leakage power, cycle time, area) 10:10000:10000:10000:10000
-Optimize ED or ED^2 (ED, ED^2, NONE): "NONE"
-Cache model (NUCA, UCA)  - "UCA"
-NUCA bank count 0
-Wire signaling (fullswing, lowswing, default) - "Global_30"
-Wire inside mat - "semi-global"
-Wire outside mat - "global"
-Interconnect projection - "conservative"
-Core count 8
-Cache level (L2/L3) - "L3"
-Add ECC - "true"
-Print level (DETAILED, CONCISE) - "DETAILED"
-Print input parameters - "false"
-Force cache config - "false"
-Ndwl 16
-Ndbl 32
-Nspd 1
-Ndcm 1
-Ndsam1 1
-Ndsam2 1
-dram_type "DDR3"
-io state "WRITE"
-addr_timing 1.0
-mem_density 4 Gb
-bus_freq 800 MHz
-duty_cycle 1.0
-activity_dq 1.0
-activity_ca 0.5
-num_dq 72
-num_dqs 18
-num_ca 25
-num_clk  2
-num_mem_dq 2
-mem_data_width 8
-rtt_value 10000
-ron_value 34
-tflight_value
-num_bobs 1
-capacity 80
-num_channels_per_bob 1
-first metric "Cost"
-second metric "Bandwidth"
-third metric "Energy"
-DIMM model "ALL"
-mirror_in_bob "F"
"""

_VAL = lambda label: re.compile(re.escape(label) + r"\s*:?\s*([-\d.eE+]+)")


def run_cacti3dd(work: Path, cell_type: str, size_gb: int, banks: int,
                 stacked_die: int, temp: float) -> dict:
    stem = f"{cell_type}_{size_gb}Gb_{stacked_die}die"
    cfg = work / f"{stem}.cfg"
    cfg.write_text(CFG_3DD.format(cell_type=cell_type, size_gb=size_gb, banks=banks,
                                  node_um=NODE_NM / 1000.0, stacked_die=stacked_die,
                                  temp=int(temp)))
    proc = subprocess.run([str(CACTI3DD_BIN), "-infile", str(cfg)], cwd=CACTI3DD_DIR,
                          capture_output=True, text=True, timeout=300)
    out = proc.stdout
    if "Area efficiency" not in out or "rows in subarray" not in out:
        raise RuntimeError(f"CACTI-3DD gave no 3D array for {stem}:\n{out[-500:]}")

    def val(label):
        m = _VAL(label).search(out)
        assert m, f"{stem}: could not parse {label!r}"
        return float(m.group(1))

    return dict(
        n_row=int(val("# rows in subarray")), n_col=int(val("# columns in subarray")),
        eff=val("Area efficiency"), area_die=val("DRAM area per die"),
        tsv_area=val("TSV area (mm2)"), banks=int(val("Number of banks")),
        dies=int(val("Stacked die count")),
        t_rcd=val("t_RCD (Row to column command delay)"),   # BIGNUM in this build
    )


def print_point(name, cell_type, d, model_eff):
    dies, banks, ncol, nrow = d["dies"], d["banks"], d["n_col"], d["n_row"]
    print(f"\n=== {name}  [CACTI-3DD cell: {cell_type} @ {NODE_NM} nm]  "
          f"{dies} dies x {banks} banks, subarray {nrow} rows x {ncol} cols ===")
    print(f"  {'AREA EFFICIENCY':<22}{'CACTI-3DD':>12}{'model':>12}{'model/CAC':>12}")
    print(f"  {'cells/total array':<22}{d['eff']:>11.2f}%{model_eff * 100:>11.2f}%"
          f"{_ratio(d['eff'], model_eff * 100):>12}")
    tsv_frac = 100.0 * d["tsv_area"] / d["area_die"]
    print(f"  -- 3D reference (no model counterpart yet) --")
    print(f"     DRAM area / die     : {d['area_die']:.3f} mm2")
    print(f"     TSV area overhead   : {d['tsv_area']:.4f} mm2  ({tsv_frac:.2f}% of die)")
    unreliable = "  <-- UNRELIABLE (cacti-3DD TSV/membus overflow)" if d["t_rcd"] > BIGNUM else ""
    print(f"     t_RCD (raw)         : {d['t_rcd']:.3g} ns{unreliable}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", type=Path, default=OPT / "config.yaml")
    ap.add_argument("--config", default="dram_cacti", help="DRAM-family config for the model side")
    ap.add_argument("--size-gb", type=int, default=8)
    ap.add_argument("--banks", type=int, default=8)
    ap.add_argument("--dies", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--work", type=Path, default=HERE / "_work_cacti3dd")
    ap.add_argument("--report", type=Path, default=HERE / "report_cacti3dd.txt")
    args = ap.parse_args()

    assert CACTI3DD_BIN.exists(), f"cacti-3DD binary not found at {CACTI3DD_BIN}"
    assert args.config in CONFIG_TO_CELL, \
        f"{args.config!r} is not a DRAM-family config (CACTI-3DD models only DRAM); " \
        f"choose from {sorted(CONFIG_TO_CELL)}"
    args.work.mkdir(parents=True, exist_ok=True)
    report_fh = open(args.report, "w")
    sys.stdout = _Tee(sys.stdout, report_fh)
    cell_type = CONFIG_TO_CELL[args.config]

    defaults, configs = load_configs(args.config_file)
    tech_spec = TechSpec(**merged_tech(defaults, configs, args.config))

    print(f"CACTI-3DD 3D-DRAM area-efficiency cross-check  (model config: {args.config})")
    print("NOTE: chip-level timing/energy excluded (cacti-3DD build overflow); "
          "TSV area is reference-only (no model PDN term yet).")
    for dies in args.dies:
        try:
            d = run_cacti3dd(args.work, cell_type, args.size_gb, args.banks, dies, 350)
        except (RuntimeError, AssertionError) as e:
            print(f"\n=== {args.config} {dies} dies: SKIPPED ({str(e).splitlines()[0]})")
            continue
        # model area efficiency at CACTI-3DD's built subarray geometry
        m = model_components(tech_spec, d["n_col"], d["n_row"], B_ACC)
        model_eff = m["a_array"] / m["a_total"]
        print_point(args.config, cell_type, d, model_eff)

    print(f"\nwrote {args.report}")
    sys.stdout = sys.stdout.stream
    report_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
