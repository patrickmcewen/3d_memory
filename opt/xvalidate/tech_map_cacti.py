"""Map a merged 3d_memory config onto a CACTI cell type + a forced single-mat cfg.

CACTI (v7.0.3DD) models only bulk-CMOS SRAM and commodity/low-power DRAM -- it has
no ReRAM/MRAM/PCM/FeFET cell (unlike DESTINY). So only the SRAM- and DRAM-family
configs map here; everything else stays a DESTINY-only cross-check.

Unlike the DESTINY tech_map, we do NOT inject cell physics into CACTI: its cells
are fixed by (cell-type, process node) in technology.cc / tech_params. The
comparison therefore pits the model coefficients in the `*_cacti` config against
CACTI's own built-in cell -- so a residual is attributable to the model's
coefficients (which is what the `*_cacti` configs are calibrated to close).

Forced geometry (see xvalidate_cacti.py for the derivation): CACTI's smallest
valid data organization is a 2x2-subarray mat, so we force Ndwl=Ndbl=2, one
column-mux stage (Ndcm=1, Ndsam1=n_col/b_acc, Ndsam2=1) and set the mat output
width to out_w = (Ndwl*Ndbl)*b_acc, which builds four subarrays of exactly
n_row x n_col each. The per-component read tree CACTI dumps is then parsed at the
subarray level (latency) / per-bit level (energy) / reported-subarray level (area).
"""

# 3d_memory config -> (CACTI data-array cell type, family). `family` only picks
# the process node band; both families run at CACTI's 22 nm floor (16 nm.dat is
# rejected as "Invalid technology nodes"), matching the DESTINY cross-check node.
CONFIG_TO_CACTI = {
    "sram_cacti":  ("itrs-hp",   "SRAM"),
    "dram_cacti":  ("comm-dram", "DRAM"),
    # convenience: the DESTINY-calibrated SRAM/DRAM points are valid CACTI cells too
    "sram_dest":   ("itrs-hp",   "SRAM"),
    "dram_100Mb":  ("comm-dram", "DRAM"),
    "edram_dest":  ("lp-dram",   "DRAM"),
}

NODE_NM = 22   # CACTI's usable floor (== DESTINY cross-check node)


# Full CACTI cfg. Everything the parser needs is present; only the array-shaping
# knobs are templated. The IO/DRAM/MemCAD tail is fixed boilerplate CACTI still
# requires even for a scratch-RAM run.
CFG_TEMPLATE = """\
-size (bytes) {size_bytes}
-Array Power Gating - "false"
-WL Power Gating - "false"
-CL Power Gating - "false"
-Bitline floating - "false"
-Interconnect Power Gating - "false"
-Power Gating Performance Loss 0.01
-block size (bytes) {block_bytes}
-associativity 1
-read-write port 1
-exclusive read port 0
-exclusive write port 0
-single ended read ports 0
-UCA bank count 1
-technology (u) {node_um}
-page size (bits) 8192
-burst length 8
-internal prefetch width 8
-Data array cell type - "{cell_type}"
-Data array peripheral type - "itrs-hp"
-Tag array cell type - "itrs-hp"
-Tag array peripheral type - "itrs-hp"
-output/input bus width {out_w}
-operating temperature (K) {temp}
-cache type "ram"
-tag size (b) "default"
-access mode (normal, sequential, fast) - "normal"
-design objective (weight delay, dynamic power, leakage power, cycle time, area) 0:0:0:100:0
-deviate (delay, dynamic power, leakage power, cycle time, area) 20:100000:100000:100000:100000
-NUCAdesign objective (weight delay, dynamic power, leakage power, cycle time, area) 100:100:0:0:100
-NUCAdeviate (delay, dynamic power, leakage power, cycle time, area) 10:10000:10000:10000:10000
-Optimize ED or ED^2 (ED, ED^2, NONE): "ED^2"
-Cache model (NUCA, UCA)  - "UCA"
-NUCA bank count 0
-Wire signaling (fullswing, lowswing, default) - "Global_30"
-Wire inside mat - "semi-global"
-Wire outside mat - "semi-global"
-Interconnect projection - "conservative"
-Core count 8
-Cache level (L2/L3) - "L3"
-Add ECC - "false"
-Print level (DETAILED, CONCISE) - "DETAILED"
-Print input parameters - "true"
-Force cache config - "true"
-Ndwl {ndwl}
-Ndbl {ndbl}
-Nspd {nspd}
-Ndcm {ndcm}
-Ndsam1 {ndsam1}
-Ndsam2 {ndsam2}
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
-num_clk 2
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


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def build_cfg_params(config_name: str, n_row: int, n_col: int, b_acc: int) -> dict:
    """Solve the forced-organization knobs for a single-mat n_row x n_col subarray.

    With Ndwl=Ndbl=2, Nspd=1 (CACTI's minimum valid mat = 2x2 subarrays):
        num_c_subarray = 8*block_bytes / Ndwl  ->  block_bytes = n_col*Ndwl/8 = n_col/4
        num_r_subarray = size_bytes / (block_bytes*Ndbl) -> size = n_row*n_col/2
    and the sense-amp column mux Ndsam1 = n_col/b_acc gives b_acc output bits per
    subarray, so the mat output width out_w = (Ndwl*Ndbl)*b_acc = 4*b_acc.
    """
    assert config_name in CONFIG_TO_CACTI, f"{config_name!r} has no CACTI cell mapping"
    assert n_col % 4 == 0, f"n_col {n_col} must be a multiple of 4 (integer block bytes)"
    assert n_col % b_acc == 0, f"n_col {n_col} must be a multiple of b_acc {b_acc}"
    mux = n_col // b_acc
    assert _is_pow2(mux), f"column mux n_col/b_acc = {mux} must be a power of two"
    assert mux <= 256, f"column mux {mux} exceeds CACTI MAX_COL_MUX (256)"
    ndwl = ndbl = 2
    return dict(
        block_bytes=n_col // 4,
        size_bytes=n_row * n_col // 2,
        out_w=(ndwl * ndbl) * b_acc,
        ndwl=ndwl, ndbl=ndbl, nspd=1,
        ndcm=1, ndsam1=mux, ndsam2=1,
        subarrays_per_mat=ndwl * ndbl,
    )
