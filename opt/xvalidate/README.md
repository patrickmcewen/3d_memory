# DESTINY cross-validation harness

Compares `opt/model.py`'s **single-subarray** timing/energy against DESTINY
(`../../destiny_3d_cache`) at an identical array geometry, so we can see *how big*
the mismatch is and *which component* it comes from. Numbers are not expected to
match — this is an attribution tool, not a calibration.

## Run

```
opt/.venv/bin/python opt/xvalidate/xvalidate.py                 # all mapped configs, default sweep
opt/.venv/bin/python opt/xvalidate/xvalidate.py --configs sram_16nm reram_16nm
opt/.venv/bin/python opt/xvalidate/xvalidate.py --rows 128 256 512 --cols 256 1024
```

Writes per-point tables to stdout and a tidy `comparison.csv`
(one `destiny` row + one `model` row per point). Generated DESTINY `.cell`/`.cfg`
files land in `_work/`.

## How it works

1. **Force one subarray.** DESTINY is pinned to `1x1` bank / `1x1` mat with
   `MuxSenseAmp = numColumn / b_acc`, so it builds exactly `numRow x numColumn`
   sensing `b_acc` bits/access. `Capacity(KB) = numRow*numColumn/8192`,
   `WordWidth = b_acc`. Mats=banks=1 ⇒ the H-tree collapses to a small
   "Non-H-Tree" residual, reported separately as overhead the model omits.
2. **Parse DESTINY's tree.** The console dump gives a hierarchical per-component
   latency/energy breakdown; we parse the read block into ps / fJ.
3. **Evaluate the model** at the same `(N_BL=numColumn, N_WL=numRow, b_acc,
   N_share=1)` by calling `model.develop_coeffs` / `energy_coeffs` and
   re-assembling `build_model`'s expressions (no solver).

## Config → DESTINY cell mapping (`tech_map.py`)

We keep DESTINY's **native** cell physics (resistances, caps, transistor widths
from its own sample cells) and align only the shared knobs:
`CellArea(F²) = v_cell / 2.56e-4`, `ReadVoltage = v_read` (voltage/charge cells
only), `DRAMCellCapacitance = c_cell` (charge-share).

| config | DESTINY cell | sense mode | node / roadmap |
|---|---|---|---|
| `sram_16nm`, `gaincell_100Mb`, `fefet_16nm` | SRAM | voltage/latch | 22 nm HP |
| `reram_16nm`, `nvm_100Mb` | memristor (1T1R CMOS) | current | 22 nm LSTP |
| `sttmram_16nm`, `sotmram_16nm` | MRAM | current | 22 nm HP |
| `pcram_16nm` | PCRAM | voltage (resistive) | 22 nm HP |
| `edram_16nm`, `dram_100Mb` | eDRAM | charge_share | 32 nm EDRAM |

## Component buckets (DESTINY ↔ model)

| bucket | DESTINY | model |
|---|---|---|
| decode+WL | Predecoder + Row Decoder latency | `t_dec + t_WL` |
| bitline | Bitline latency | `t_BL` |
| senseamp | Senseamp latency | `t_SA` |
| mux/sw | Mux latency (Precharge is off critical path) | `t_sw` |
| TOTAL(dev) | Predecoder + Subarray latency | `sum_dev` |
| bitline+cell E | Subarray energy − periph leaves | `k_col·N_BL + k_arr·cells` |
| periph E | RowDec+MuxDec+SA+Mux+Precharge dyn E | `e_periph` |
| TOTAL(access) E | Subarray dynamic energy | `E_access` |

## Decisions & caveats

- **Node:** DESTINY's finest is 22 nm; the model is nominally 16 nm. So DESTINY's
  internally-derived bitline pitch/RC/energy run ~(22/16) high vs a true 16 nm.
- **Roadmap:** dense arrays of leaky/passive cells can't sustain a tall bitline
  under HP transistors (`BITLINE_LEAKAGE_TOLERANCE=1`); LSTP is the realistic and
  DESTINY-feasible choice for charge-share/resistive. eDRAM uses DESTINY's
  dedicated `EDRAM` device tech (only modeled at 32/45 nm).
- **Feasibility limits are findings, not failures.** DESTINY refuses to *build*
  much of the tall-bitline regime the model optimizes in:
  - SRAM folding limit ⇒ single subarray caps out ~512–1024 rows at 22 nm.
  - **charge-share (eDRAM):** the developed-signal check
    `dV = vdd/2·Ccell/(Ccell+Cbl) ≥ MinSenseVoltage` caps DESTINY subarrays at
    *tens* of rows, while the model's `nwl_sig` cap allows ~450 — a ~50× gap in
    the row ceiling. Forced points that DESTINY rejects are printed as `SKIPPED`.

## Headline mismatches (default sweep)

**Timing calibrated 2026-07-14 (T1/T2/T3 in config.yaml):** voltage-sensed cells
now develop the bitline to the *sense margin* (`settle_frac` 0.99→0.1, i.e.
`V_margin/V_signal`), and `t_SA0` for latch cells dropped to a pure latch-resolve
floor (0.05 ns) so it no longer double-counts the develop that `t_BL` carries.
Current-mode (`t_BL`) left untouched — DESTINY's current-sense bitline is the
untrusted number, not the model.

- **decode+WL agrees best:** model 1.0–1.7× DESTINY.
- **Bitline latency (voltage/latch): now 0.99–1.23× DESTINY** for SRAM/gaincell
  across the full sweep (was 25–44×). The wordline-slew coupling (`c_slew`, the
  Horowitz input-ramp term `t_BL = sqrt(t_BLrc² + c_slew·t_BLrc·t_WL)`) captures
  DESTINY's column-dependence — the wide-short corner (1024 cols × 256 rows) went
  from 0.35× → 0.99×. Recovers the slew-free settle as `t_WL → 0` or `c_slew → 0`.
- **`t_SA` (voltage): model 50 ps vs DESTINY 2.7 ps latch regen.** Small absolute
  gap; no longer dominates the total. Going lower is unphysical for a latch floor.
- **PCRAM left divergent on purpose:** `t_SA0=2 ns` + high `r_pullup` encode
  genuine PCM read slowness; DESTINY's generic voltage read (81 ps bitline,
  2.7 ps SA) under-models it. Same caveat class as current mode.
- **Current-mode bitline: still 25×–2000× (unchanged, by design).**
**Energy calibrated 2026-07-14 (E1/E2/E3 in config.yaml + model.py):** `e_periph`
became `e_periph` (fixed intercept ~85 fJ) + `e_periph_col`·N_BL (~0.75 fJ/col),
matching DESTINY's per-column periph growth. `e_bitline` for non-destructive
sensed reads now develops only the partial swing `v_read·v_sense` (sense margin)
instead of the full rail. **E3 (sense-amp read energy):** a NEW per-*sensed-bit*
term `e_sa_read` carries the sense-amp/IV-converter read energy (added as
`e_sa_read·b_acc` in `E_access`, so it survives overfetch amortization — one IV
converter per sensed bit). Current-sense configs set it to EMBER's measured VSA
(~0.5 pJ/bit, Ch. 5 Fig. 5.26); voltage stays 0 (SA folded in `e_periph`).

- **`e_periph`: voltage now ~1.0× DESTINY** (PCRAM ~0.65× residual). **Current-
  sense periph now ~3.3× DESTINY — on purpose:** it follows EMBER silicon
  (sense-amp-dominated, ~0.5 pJ/bit·b_acc), and DESTINY underestimates current-
  sense read (same story as the timing, App. A p130). Model current-sense E_bit
  now ~560–1190 fJ/bit, bracketing EMBER's measured ~1 pJ/bit @40 nm.
- **`e_bitline` (voltage): now 1.0–3.3× DESTINY** (was ~18×); SRAM ~1.0–1.4×.
- **`e_bitline` (current): now the Upton App. A integrated-current form** (was
  full-rail CV²). The read integrates I_read over the settle time t_s=τ·f_margin
  (τ=C_BL·n_ut/I_read), so E=V_read·C_BL·n_ut·f_margin — **I_read cancels**,
  leaving an effective swing `v_read·n_ut·f_margin` (~0.18) not the full rail.
  Result: matches DESTINY's STT/SOT-MRAM bitline (~4.4 pJ) to 0.8–2.5× at small
  arrays. DESTINY's ReRAM bitline (16 fJ) is 275× below its own MRAM number —
  the Seevinck-ported-to-resistive underestimate Upton calls out (App. A p130),
  so we go with Upton, cite `[upton2024, ember]`, and do NOT cite DESTINY here.
