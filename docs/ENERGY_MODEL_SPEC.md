# Energy / Power-Density Model — Design Spec

Adds an energy-per-bit model and a **power-density constraint** to the BW-max
optimizer (`opt/model.py`). Status: **implemented 2026-07-14** (`energy_coeffs`,
`E_bit`/`P_dyn` vars + `def_E_bit`/`def_P_dyn`/`power_density`; params in
`config.yaml`; magnitude tests in `tests/test_solve.py`). With the placeholder
calibration the constraint binds in every config (power-density-limited BW).

Goal: `power_density = P_total / footprint ≤ P_max` (default **1.1 W/mm²**).
Since the tool *maximizes* BW and `P_dyn ∝ BW`, this turns into a binding
trade-off (each bit/s costs energy over a fixed footprint) and is the bridge to
the project's ΔT / thermal scope.

## Provenance

Derived from a DESTINY source read + a 4-paper literature cross-check
(2026-07-14). Sources and what each contributed:

- **DESTINY `SubArray::CalculatePower`** (`destiny_3d_cache/SubArray.cpp:685-775`)
  — the uniform per-mode energy *structure* (CV² bitline, refresh leak, cell
  set/reset). Constants: `DRAM_REFRESH_PERIOD=64ms`, `SHAPER_EFFICIENCY_*`.
- **NVSim** (Dong 2012) — confirms `E_dyn = C·V_dd²` per component; NVM cell write
  by Joule heating `E_SET=I²·R·t`. Crosspoint sneak-path leakage (Eq. 1-4).
- **Ha 2018** (DRAM energy, DreamRAM's own source [5]) — BLSA dominates DRAM
  energy (47% row, 68% refresh), scales with rows (bitline cap). Charge-share
  sense `ΔV = V_BLP/(1+C_BL/C_S)` == our `margin_sa` collapse term. Refresh
  Arrhenius in T (`t_REF ∝ e^{Ea/kT}`, Ea≈0.3-0.8 eV).
- **O'Connor 2017** (Fine-Grained DRAM) — validates the *whole formulation*: its
  "max E/bit for a BW within a power budget" curve IS `E_bit = P_budget/BW`.
  Magnitude anchors: GDDR5 14, HBM2 3.9, FGDRAM ~2 pJ/bit. Row-overfetch =
  the `b_acc/N_BL` efficiency lever.
- **NVMExplorer** (Pentecost 2022) — per-bit anchors per technology; write≫read
  (100-680×) for NVM, so write energy MUST be modeled.
- **Thermal** (Akgun 2016, Lee 2025, Pan 2025 Stratum) — `P_max` anchor and the
  ΔT=P·R_th stack model for the later thermal coupling.

## Energy equations

Uniform per-access form (only `V_read²` differs by mode):

```
E_read_access = (c_cell + c_bl·N_WL)·V_read² · N_BL      # bitline/BLSA — dominant, geometry-coupled
              + E_periph                                  # decode + WL drive, ~const per access
              + E_cellread                                # NVM only: 2·readPower·t_SA
E_bit_read    = E_read_access / b_acc                     # ∝ (c_cell + c_bl·N_WL)·N_BL / b_acc
```

`V_read²` per mode:
- SRAM: `V_dd²`               (full-swing precharge)   [DESTINY:687]
- DRAM/eDRAM: `senseVoltage·vdd` (partial swing; Ha ΔV) [DESTINY:699]
- NVM (voltage-sense): `V_pre² − V_on²`                 [DESTINY:714]

Write energy (needed once `write_fraction > 0`; essential for NVM):

```
DRAM/eDRAM restore: (c_bl·N_WL)·vdd² · N_BL              # full-rail write-back  [DESTINY:702]
NVM:                max(E_set, E_reset)/shaper_eff + CV²_parasitic
                    E_set = I_set²·R·t_set,  E_reset = I_reset²·R·t_reset   # NVSim Joule, per-tech
```

The `N_BL/b_acc` ratio is O'Connor's row-overfetch efficiency: a whole row (all
`N_BL` bitlines) swings per activate but only `b_acc` bits are output. Couples to
the existing `width_cap` (`b_acc ≤ N_BL`) — model will be pushed toward `b_acc→N_BL`.

## Power

```
P_dyn  = BW · [ (1 − f_w)·E_bit_read + f_w·E_bit_write ]   # f_w = write_fraction; matches NVMExplorer Σaccess·E
P_leak = total_cells · p_leak_bit
```

`p_leak_bit` per mode:
- SRAM:       `V_dd · I_leak_cell`                        (gate leak ∝ cells)     [DESTINY:691]
- DRAM/eDRAM: `E_bit_refresh / t_REF(T)`                  (refresh; Arrhenius T)  [DESTINY:703]
- NVM:        selector/sneak-path leak (small; DESTINY zeroes it — fill from NVSim Eq.1-4)

## Power-density constraint (formulates with NO division)

```
A_used = volume_used / (L · t_layer)                      # footprint [mm²]; monolithic-3D shares one footprint
⇒  P_dyn + P_leak  ≤  (P_max / (L·t_layer)) · volume_used
```

`volume_used` is already the LHS polynomial of the `m.volume` constraint, so the
RHS is a constant × that expression — no new geometry. `P_dyn = BW × E_bit` is the
only nonconvex piece (var × polynomial); handle with the existing aux-var +
Gurobi `nonconvex=2` pattern (~4-6 new `def_*` product vars). **Main solver-cost
risk to watch during implementation.**

## P_max presets (thermal-anchored)

| preset | W/mm² | basis |
|---|---|---|
| air (conservative) | 0.4 | Akgun 2016 — 3D-integration cost/thermal crossover; Tj=100°C, Tamb=30°C |
| **liquid (default)** | **1.1** | Pan 2025 Stratum — 144.5 W / 121 mm² = 1.19, vapor-chamber liquid cooling, real workload |
| microfluidic (ceiling) | 2.0 | Pan 2025 — 200 W/cm² state-of-art in-stack cooling |

**Confirmed default: `P_max = 1.1 W/mm²`.**

Later ΔT coupling (out of scope for first cut): `ΔT = P_density · R_th`, Akgun
Eq.13 stack accumulation (each layer sees cumulative power above it); hybrid-bond
interface R_th ≈ 0.8-1.2 mm²·K/W (Lee 2025); DRAM Tj_max ≈ 85-95°C (refresh-driven),
Tamb ≈ 30-45°C.

## Validation targets (magnitudes the model must reproduce)

- DRAM `E_bit` ≈ 2-4 pJ/bit (O'Connor: HBM2 3.9, FGDRAM ~2, GDDR5 14).
- Read `E_bit`: SRAM ~0.5, STT ~0.12, PCM ~0.10, RRAM ~0.115, FeFET ~0.3 pJ/bit
  (NVMExplorer Fig. 5, 2 MB arrays).
- NVM write ~0.26 nJ (STT) → 6.3 nJ (PCM) per word (NVSim §VII).
- Power density at a target BW ≈ O(1) W/mm² for HBM-class points.

## Implementation order (when unblocked)

1. Energy params in `config.yaml` per tech: `V_read`, `E_set`/`E_reset` (NVM),
   `p_leak_bit`, `write_fraction`, plus `P_max` (default 1.1) in `defaults`.
2. `E_bit_read` / `E_bit_write` / `P_leak` Pyomo `Expression`s + aux product vars
   in `model.py`.
3. The `power_density ≤ P_max` constraint (RHS form above).
4. Magnitude-validation test in `tests/` against the pJ/bit anchors.

## Open decisions (resolved by the investigation)

1. Write mix — **modeled** via per-config `write_fraction` (default 0 for read-BW
   studies; NVM configs set it > 0). NVM write from `E_set/E_reset` + shaper derate.
2. Leakage — **unified** `total_cells·p_leak_bit`; DRAM refresh Arrhenius, NVM ≈ 0.
3. Area denominator — `A_used = volume_used/(L·t_layer)`, RHS form (no division).
4. Ceiling vs thermal coupling — flat `P_max` first; ΔT=P·R_th coupling deferred.
