#!/usr/bin/env python3
"""Cross-validate the 3d_memory subarray timing/energy/area model vs CACTI.

The sibling xvalidate.py does this against DESTINY; this does the same against
CACTI (v7.0.3DD, ../../cacti). CACTI models only bulk-CMOS SRAM and commodity/
low-power DRAM, so this harness covers just the SRAM- and DRAM-family configs
(the `*_cacti` calibration targets in config.yaml + the SRAM/DRAM `*_dest`
points); the emerging-memory configs remain DESTINY-only.

For each (config, geometry) point:
  1. Force CACTI to build one 2x2-subarray mat of exactly n_row x n_col cells
     (tech_map_cacti.build_cfg_params), run it, parse the per-component read tree.
  2. Forward-evaluate opt/model.py at the SAME (N_BL=n_col, N_WL=n_row, b_acc,
     N_share=1) by reusing xvalidate.model_components (no solver).
  3. Print a per-component side-by-side with ratios; dump comparison_cacti.csv.

Comparison granularity (CACTI's smallest valid array is a 4-subarray mat):
  LATENCY  per-subarray critical path (one subarray of n_row x n_col), ns->ps.
           CACTI Decoder+wordline / Bitline / Sense Amp  <-> model t_dec+t_WL /
           t_BL / t_SA. H-tree in/out is reported separately as interconnect
           overhead the single-subarray model omits (like DESTINY's Non-H-Tree).
  ENERGY   per delivered bit (energy/access / bits/access), nJ->fJ. Count-
           invariant, so it sidesteps how many of the 4 subarrays a mat access
           activates. CACTI Bitlines <-> model e_bitline (bitline+cell CV^2);
           everything else (decoder, wordline, precharge, SA, muxes, output
           driver) <-> model e_periph.
  AREA     per subarray, from CACTI's reported Subarray Height x Length; cell
           area via the reported area efficiency. Normalized area_um2 @node ->
           F^2 -> 16 nm volume [um^3] -- the same v_cell currency as xvalidate.

Run (needs the opt venv for pyyaml + the model import):
  opt/.venv/bin/python opt/xvalidate/xvalidate_cacti.py
  opt/.venv/bin/python opt/xvalidate/xvalidate_cacti.py --configs sram_cacti dram_cacti
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPT = HERE.parent
MEMROOT = OPT.parent.parent                       # .../memory
CACTI_DIR = MEMROOT / "cacti"
CACTI_BIN = CACTI_DIR / "cacti"

sys.path.insert(0, str(OPT))
sys.path.insert(0, str(HERE))
from model import TechSpec                                         # noqa: E402
from xvalidate import (load_configs, merged_tech, model_components,  # noqa: E402
                       _Tee, _ratio, CONV16)
from tech_map_cacti import (CONFIG_TO_CACTI, NODE_NM, CFG_TEMPLATE,  # noqa: E402
                            build_cfg_params)

B_ACC = 64
DEFAULT_COLS = [256, 1024]
DEFAULT_ROWS = [256, 512, 1024, 2048, 4096]

_VAL = lambda label: re.compile(re.escape(label) + r"\s*[:\-]?\s*([-\d.eE]+)")


# --------------------------------------------------------------------------- #
# CACTI: generate one forced mat, run, parse
# --------------------------------------------------------------------------- #
def run_cacti(work: Path, config_name: str, n_row: int, n_col: int, b_acc: int, temp: float):
    """Write a forced-config .cfg, run CACTI, parse the data-array read tree."""
    cell_type, _family = CONFIG_TO_CACTI[config_name]
    p = build_cfg_params(config_name, n_row, n_col, b_acc)
    stem = f"{config_name}_{n_row}x{n_col}"
    cfg = work / f"{stem}.cfg"
    cfg.write_text(CFG_TEMPLATE.format(
        cell_type=cell_type, node_um=NODE_NM / 1000.0, temp=int(temp), **p))

    proc = subprocess.run([str(CACTI_BIN), "-infile", str(cfg)], cwd=CACTI_DIR,
                          capture_output=True, text=True, timeout=180)
    out = proc.stdout
    if "no valid data array" in out or "Access time (ns)" not in out:
        raise RuntimeError(f"CACTI built no valid array for {stem}:\n{out[-500:]}")

    # confirm CACTI honored the forced organization (guarantees the geometry)
    for key, want in [("Best Ndwl", p["ndwl"]), ("Best Ndbl", p["ndbl"]),
                      ("Best Ndsam L1", p["ndsam1"])]:
        m = _VAL(key).search(out)
        assert m and int(float(m.group(1))) == want, \
            f"{stem}: CACTI {key}={m and m.group(1)} != forced {want}"

    # data-side read tree (scratch RAM => no tag block to disambiguate)
    tree = out[out.index("Time Components:"):out.index("Wire Properties:")]

    def val(label):
        m = _VAL(label).search(tree)
        assert m, f"{stem}: could not parse {label!r}"
        return float(m.group(1))

    return dict(
        t_decode=val("Decoder + wordline delay (ns)"),
        t_bitline=val("Bitline delay (ns)"),
        t_senseamp=val("Sense Amplifier delay (ns)"),
        t_htree=val("H-tree input delay (ns)") + val("H-tree output delay (ns)"),
        e_decoder=val("Decoder (nJ)"), e_wordline=val("Wordline (nJ)"),
        e_bl_mux=val("Bitline mux & associated drivers (nJ)"),
        e_sa_mux=val("Sense amp mux & associated drivers (nJ)"),
        e_precharge=val("Bitlines precharge and equalization circuit (nJ)"),
        e_bitlines=val("Bitlines (nJ)"), e_sa=val("Sense amplifier energy (nJ)"),
        e_outdrv=val("Sub-array output driver (nJ)"),
        e_total=val("Total dynamic read energy/access  (nJ)"),
        a_sub_h=val("Subarray Height (mm)"), a_sub_l=val("Subarray Length (mm)"),
        a_eff=val("Area efficiency (Memory cell area/Total area) -"),
        out_w=p["out_w"],
    )


def cacti_components(d: dict) -> dict:
    """Collapse the CACTI leaves into the comparison buckets (ps / fJ/bit / um^3)."""
    NS, NRG = 1e3, 1e6                              # ns->ps, nJ->fJ
    t_decode = d["t_decode"] * NS
    t_bl = d["t_bitline"] * NS
    t_sa = d["t_senseamp"] * NS
    t_total = t_decode + t_bl + t_sa                # excludes H-tree, like model sum_dev

    e_bitline = d["e_bitlines"] * NRG               # bitline+cell CV^2 remainder
    e_periph = (d["e_decoder"] + d["e_wordline"] + d["e_bl_mux"] + d["e_sa_mux"]
                + d["e_precharge"] + d["e_sa"] + d["e_outdrv"]) * NRG
    e_total = d["e_total"] * NRG
    per_bit = e_total / d["out_w"]                  # energy per delivered bit

    # per-subarray area: mm^2 -> um^2 @node -> F^2 -> 16 nm volume [um^3]
    a_sub_um2 = d["a_sub_h"] * d["a_sub_l"] * 1e6
    to_vol = lambda a: a / (NODE_NM * 1e-3) ** 2 * CONV16
    a_total = to_vol(a_sub_um2)
    a_array = to_vol(a_sub_um2 * d["a_eff"] / 100.0)
    return dict(t_decode=t_decode, t_bitline=t_bl, t_senseamp=t_sa, t_mux=0.0,
                t_total=t_total, t_htree=d["t_htree"] * NS,
                e_bitline=e_bitline / d["out_w"], e_periph=e_periph / d["out_w"],
                e_total=e_total / d["out_w"], per_bit=per_bit,
                a_array=a_array, a_total=a_total)


# --------------------------------------------------------------------------- #
# reporting  (mirrors xvalidate.print_point; energy shown per delivered bit)
# --------------------------------------------------------------------------- #
def print_point(name, cell_type, n_row, n_col, b_acc, ccmp, mcmp):
    print(f"\n=== {name}  [CACTI cell: {cell_type} @ {NODE_NM} nm]   subarray "
          f"{n_row} rows x {n_col} cols (b_acc={b_acc}, mux={n_col // b_acc}) ===")
    if mcmp.get("nwl_cap") is not None and n_row > mcmp["nwl_cap"]:
        print(f"    !! model charge-share signal collapse: N_WL cap = "
              f"{mcmp['nwl_cap']:.0f} < {n_row} rows (model would FORBID this geometry)")
    print(f"  {'LATENCY [ps]':<16}{'CACTI':>12}{'model':>12}{'model/CAC':>12}")
    for k, lbl in [("t_decode", "decode+WL"), ("t_bitline", "bitline"),
                   ("t_senseamp", "senseamp"), ("t_total", "TOTAL(dev)")]:
        print(f"  {lbl:<16}{ccmp[k]:>12.2f}{mcmp[k]:>12.2f}{_ratio(ccmp[k], mcmp[k]):>12}")
    print(f"  {'(h-tree*)':<16}{ccmp['t_htree']:>12.2f}{'--':>12}{'overhead':>12}")

    # model buckets are per-subarray (b_acc bits); normalize to per-bit to match
    mbit = lambda k: mcmp[k] / b_acc
    print(f"  {'ENERGY [fJ/bit]':<16}{'CACTI':>12}{'model':>12}{'model/CAC':>12}")
    for k, lbl in [("e_bitline", "bitline+cell"), ("e_periph", "periph"),
                   ("e_total", "TOTAL(access)")]:
        print(f"  {lbl:<16}{ccmp[k]:>12.2f}{mbit(k):>12.2f}{_ratio(ccmp[k], mbit(k)):>12}")

    print(f"  {'AREA [um^3]':<16}{'CACTI':>12}{'model':>12}{'model/CAC':>12}")
    for k, lbl in [("a_array", "array(cells)"), ("a_total", "TOTAL(subarray)")]:
        print(f"  {lbl:<16}{ccmp[k]:>12.4f}{mcmp[k]:>12.4f}{_ratio(ccmp[k], mcmp[k]):>12}")


CSV_COLS = ["config", "cell", "n_row", "n_col", "b_acc", "source",
            "t_decode", "t_bitline", "t_senseamp", "t_total",
            "e_bitline_perbit", "e_periph_perbit", "e_total_perbit",
            "a_array", "a_total"]


def csv_rows(name, cell_type, n_row, n_col, b_acc, ccmp, mcmp):
    def row(src, c, perbit_div):
        vals = [c["t_decode"], c["t_bitline"], c["t_senseamp"], c["t_total"],
                c["e_bitline"] / perbit_div, c["e_periph"] / perbit_div,
                c["e_total"] / perbit_div, c["a_array"], c["a_total"]]
        return [name, cell_type, n_row, n_col, b_acc, src] + [f"{v:.4f}" for v in vals]
    # ccmp energies already per-bit (div 1); mcmp energies per-subarray (div b_acc)
    return [row("cacti", ccmp, 1.0), row("model", mcmp, float(b_acc))]


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", type=Path, default=OPT / "config.yaml")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="config names to test (default: all CACTI-mapped ones present)")
    ap.add_argument("--rows", nargs="+", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--cols", nargs="+", type=int, default=DEFAULT_COLS)
    ap.add_argument("--b-acc", type=int, default=B_ACC)
    ap.add_argument("--work", type=Path, default=HERE / "_work_cacti")
    ap.add_argument("--csv", type=Path, default=HERE / "comparison_cacti.csv")
    ap.add_argument("--report", type=Path, default=HERE / "report_cacti.txt")
    args = ap.parse_args()

    assert CACTI_BIN.exists(), f"cacti binary not found at {CACTI_BIN} (run make in {CACTI_DIR})"
    args.work.mkdir(parents=True, exist_ok=True)
    report_fh = open(args.report, "w")
    sys.stdout = _Tee(sys.stdout, report_fh)
    defaults, configs = load_configs(args.config_file)

    names = args.configs or [n for n in configs if n in CONFIG_TO_CACTI]
    assert names, "no CACTI-mapped configs to test"

    out_rows = [",".join(CSV_COLS)]
    for name in names:
        assert name in CONFIG_TO_CACTI, f"config {name!r} has no CACTI cell mapping"
        cell_type = CONFIG_TO_CACTI[name][0]
        tech_dict = merged_tech(defaults, configs, name)
        tech_spec = TechSpec(**tech_dict)
        for n_col in args.cols:
            for n_row in args.rows:
                try:
                    craw = run_cacti(args.work, name, n_row, n_col, args.b_acc, 350)
                except (RuntimeError, AssertionError) as e:
                    print(f"\n=== {name}  {n_row}x{n_col}: SKIPPED ({str(e).splitlines()[0]})")
                    continue
                ccmp = cacti_components(craw)
                mcmp = model_components(tech_spec, n_col, n_row, args.b_acc)
                print_point(name, cell_type, n_row, n_col, args.b_acc, ccmp, mcmp)
                out_rows += [",".join(map(str, r)) for r in
                             csv_rows(name, cell_type, n_row, n_col, args.b_acc, ccmp, mcmp)]

    args.csv.write_text("\n".join(out_rows) + "\n")
    print(f"\nwrote {args.csv}")
    print(f"wrote {args.report}")
    sys.stdout = sys.stdout.stream
    report_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
