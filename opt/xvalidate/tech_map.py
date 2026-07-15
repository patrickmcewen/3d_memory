"""Map a merged 3d_memory config onto a DESTINY .cell + process node.

The 3d_memory model is abstract (c_bl, r_bl, i_read, ...); DESTINY needs a
physical cell. We keep DESTINY's *native* cell physics (resistances, caps,
transistor widths taken from its own sample cells) and only align the two knobs
the configs genuinely share:
  * CellArea (F^2)  -- recovered from v_cell (v_cell = F^2 * 2.56e-4 @16nm),
  * ReadVoltage (V) -- set to the config's v_read so the CV^2 swing matches,
  * DRAMCellCapacitance -- set to c_cell for charge-share cells.
Everything else is DESTINY's own value, so a residual in the comparison is
attributable to either (a) different per-cell RC inputs or (b) different
formulas -- which is exactly what the cross-check is meant to expose.

DESTINY's finest process node is 22 nm; the model is nominally 16 nm. We run
DESTINY at 22 nm and record the gap (F=22 vs 16 -> ~1.4x cell pitch, so
DESTINY's internally-derived bitline RC/energy run a bit high vs a true 16 nm).
"""

# v_cell [um^3/bit] = CellArea_F2 * (16nm)^2 * t_layer(=1um) = F2 * 2.56e-4
F2_PER_VCELL = 1.0 / 2.56e-4

# DESTINY MemCellType + native physical params per cell family. CellArea and
# ReadVoltage are overwritten per-config below; the rest are DESTINY's own.
BASE_CELLS = {
    "SRAM": {
        "MemCellType": "SRAM",
        "CellAspectRatio": "1.46",
        "SRAMCellNMOSWidth (F)": "2.08",
        "SRAMCellPMOSWidth (F)": "1.23",
        "AccessCMOSWidth (F)": "1.31",
        "AccessType": "CMOS",
        "MinSenseVoltage (mV)": "80",
        "Stitching": "16",
    },
    "eDRAM": {
        "MemCellType": "eDRAM",
        "CellAspectRatio": "2.39",
        "ReadMode": "voltage",
        "AccessType": "CMOS",
        "AccessCMOSWidth (F)": "1.31",
        "SetVoltage (V)": "vdd",
        "ResetVoltage (V)": "vdd",
        "MinSenseVoltage (mV)": "10",
        # DRAMCellCapacitance (F) injected from c_cell.
    },
    "memristor": {  # ReRAM (1T1R / 1D1R)
        "MemCellType": "memristor",
        "CellAspectRatio": "1",
        "ResistanceOnAtReadVoltage (ohm)": "1000000",
        "ResistanceOffAtReadVoltage (ohm)": "10000000",
        "ResistanceOnAtSetVoltage (ohm)": "100000",
        "ResistanceOffAtSetVoltage (ohm)": "15000000",
        "ResistanceOnAtResetVoltage (ohm)": "100000",
        "ResistanceOffAtResetVoltage (ohm)": "15000000",
        "ResistanceOnAtHalfResetVoltage (ohm)": "500000",
        "CapacitanceOn (F)": "1e-16",
        "CapacitanceOff (F)": "1e-16",
        "ReadMode": "current",
        "ReadVoltage (V)": "0.3",          # native resistive read (not the CV^2 rail)
        "ReadEnergy (pJ)": "0.1",
        "ResetMode": "voltage", "ResetVoltage (V)": "2.0", "ResetPulse (ns)": "4.0", "ResetEnergy (pJ)": "0.6",
        "SetMode": "voltage", "SetVoltage (V)": "2.0", "SetPulse (ns)": "4.0", "SetEnergy (pJ)": "0.6",
        # 1T1R (CMOS access), matching reram_16nm; diode-access is the xpoint variant.
        "AccessType": "CMOS", "VoltageDropAccessDevice (V)": "0.2", "AccessCMOSWidth (F)": "4",
        "ReadFloating": "false",
    },
    "MRAM": {  # STT / SOT
        "MemCellType": "MRAM",
        "CellAspectRatio": "0.57",
        "ResistanceOn (ohm)": "6000",
        "ResistanceOff (ohm)": "12000",
        "ReadMode": "current",
        "ReadVoltage (V)": "0.25",         # native resistive read (not the CV^2 rail)
        "MinSenseVoltage (mV)": "25",
        "ReadPower (uW)": "23.4",
        "ResetMode": "current", "ResetCurrent (uA)": "40.82", "ResetPulse (ns)": "4", "ResetEnergy (pJ)": "0.252",
        "SetMode": "current", "SetCurrent (uA)": "40.82", "SetPulse (ns)": "4", "SetEnergy (pJ)": "0.252",
        "AccessType": "CMOS", "VoltageDropAccessDevice (V)": "0.15", "AccessCMOSWidth (F)": "8.5",
    },
    "PCRAM": {
        "MemCellType": "PCRAM",
        "CellAspectRatio": "0.5",
        "ResistanceOn (ohm)": "1000",
        "ResistanceOff (ohm)": "1000000",
        "ReadMode": "voltage",
        "ReadCurrent (uA)": "40",
        "ReadEnergy (pJ)": "20",
        "ResetMode": "current", "ResetCurrent (uA)": "300", "ResetPulse (ns)": "40",
        "SetMode": "current", "SetCurrent (uA)": "150", "SetPulse (ns)": "150",
        "AccessType": "CMOS", "VoltageDropAccessDevice (V)": "0.3", "AccessCMOSWidth (F)": "2",
    },
}

# 3d_memory config name -> DESTINY cell family. Configs that share a family map
# to the same base cell; CellArea/ReadVoltage/c_cell still differentiate them.
CONFIG_TO_DTYPE = {
    "gaincell_100Mb": "SRAM",
    "sram_16nm": "SRAM",
    "dram_100Mb": "eDRAM",
    "edram_16nm": "eDRAM",
    "nvm_100Mb": "memristor",
    "reram_16nm": "memristor",
    "reram_xpoint_16nm": "memristor",
    "sttmram_16nm": "MRAM",
    "sotmram_16nm": "MRAM",
    "pcram_16nm": "PCRAM",
    "fefet_16nm": "SRAM",  # FeFET = transistor/voltage read; nearest DESTINY analog
}


def build_cell_params(config_name: str, tech: dict) -> dict:
    """Return the DESTINY .cell key->value dict for a merged config's technology."""
    dtype = CONFIG_TO_DTYPE[config_name]
    cell = dict(BASE_CELLS[dtype])
    cell["CellArea (F^2)"] = f"{tech['v_cell'] * F2_PER_VCELL:.3f}"
    # Only align the read rail for cells whose read IS a full/partial CV^2 swing
    # (SRAM latch, charge-share DRAM). Resistive current-sense cells keep their
    # native low read voltage -- overriding it breaks the access-device margin.
    if dtype in ("SRAM", "eDRAM"):
        cell["ReadVoltage (V)"] = f"{tech['v_read']:.3f}"
    if dtype == "eDRAM":
        cell["DRAMCellCapacitance (F)"] = f"{tech['c_cell'] * 1e-15:.4e}"  # fF -> F
    return cell, dtype


# DESTINY derives the cell's width/height the moment it parses CellAspectRatio,
# using whatever CellArea it has read so far -- so CellArea MUST precede
# CellAspectRatio, or the cell pitch (and every pitch-limited driver) is wrong
# and tall subarrays fail transistor folding. Front-load the geometry keys.
_KEY_ORDER = ["MemCellType", "CellArea (F^2)", "CellAspectRatio", "ProcessNode"]


def cell_text(cell: dict) -> str:
    """Render a .cell dict to DESTINY's ``-Key: Value`` text, geometry-keys first."""
    ordered = [k for k in _KEY_ORDER if k in cell]
    ordered += [k for k in cell if k not in _KEY_ORDER]
    return "\n".join(f"-{k}: {cell[k]}" for k in ordered) + "\n"
