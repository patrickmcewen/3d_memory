# Power-Delivery-Network (PDN) Model — Design Spec

Adds a **power-delivery constraint** to the BW-max optimizer (`opt/model.py`).
Supplying current to the stacked tiers costs vertical power vias that (a) compete
for the same footprint/volume as cells + sense amps, and (b) must stay within an
IR-drop voltage margin. Both effects bound the achievable bandwidth density.
Status: **spec approved (both floors), not yet implemented.**

Goal: the supply current `I = P_total/V_dd` drawn to run at bandwidth `BW`
requires `N_pv` power vias; those vias consume `Vol_pdn` out of the fixed volume
budget, so every extra bit/s of bandwidth steals packing area from the arrays and
SAs that produce it. This is the PDN sibling of the thermal `P_max` cap: both are
driven by the same `P_total` and both bind harder as `BW` rises.

## Depends on

The energy model (`ENERGY_MODEL_SPEC.md`, **implemented 2026-07-14**) already
exposes the supply power as solver terms — PDN reads the current off them, adding
**no new nonlinearity**:

- `m.P_dyn`  [uW]  — dynamic power `BW * E_bit`            (`model.py:327,329`)
- `tech.p_leak_bit * m.total_cells`  [uW] — leakage/refresh (`model.py:331`)
- `vol_used` polynomial — the packing LHS PDN volume adds to (`model.py:303-308`)

Composes with the in-flight ΔT=P·R_th thermal coupling (same `P_total`); see the
thermal-denominator note under Open decisions.

## Provenance

Literature cross-check (2026-07-14, `memory/literature/`). Only one paper in the
corpus carries a portable supply-PDN model; the IR-drop physics comes from the
crosspoint/array-internal analog. See `MEMORY.md :: 3d-memory-pdn-model`.

- **Pan 2025 Stratum** (§4.3, the load-bearing source) — power-via **area** model
  `A_PD = (P/V_dd)·(A_TSV/I_TSV)·redundancy`, competing in a shared area budget
  (their Eq. 2) with compute/PHY/peripheral. Calibrated: **36 mA per 25 µm² via
  (= 1.44 mA/µm²)**, **2:1 redundancy** (their [88], an EM/vendor current limit,
  *not* IR-drop-derived). Cooling ceiling 200 W/cm²; 104 W DRAM / 45 W logic die.
- **Li 2026 ATLAS** (p.3) — "with mini-TSVs directly delivering power to PBs,
  simultaneous PB ACT/PRE is supported" — concurrent bank activation (hence
  concurrent bandwidth) is gated by dedicated per-bank power delivery.
- **Lepri 2022** (crosspoint IR drop) — nodal `V=IR` accumulation along a rail
  (`V_{i,j}=V_{i,j-1}−r·ΣG·ΔV`, `r=2ρ/F`); IR severity grows with current, rail
  length, and the fraction of concurrently-active elements (worst case = all on).
- **Zhang 2013 VRRAM** — the 3D-specific effect: read margin degrades with **layer
  count** (90%→65% for 2→64 layers); per-branch current cap (≤10 µA) bounds array
  size; "parallel operation within a local array remains questionable."
- **O'Connor 2017** (FGDRAM) — tFAW (concurrent-activate window) "effectively
  eliminated due to power delivery constraints" once per-activation charge drops:
  per-access energy sets the power-delivery-limited concurrency. HBM2 3.9 pJ/bit.
- Corroborating anchors: Akgun 0.4 W/mm² air ceiling; Jiang Cu-Cu <10 mΩ/bond,
  Black EM exponent n≈2.1–2.5, Ren sub-10 µm pitch → +20% PD efficiency; Lee 2025
  `P_I/O=I²R_I/O`; Kim 2014 HBM V_dd 1.2 V.
- **Gap:** no paper in the corpus pins the IR-drop budget as a % of V_dd (`δ_pdn`)
  or a per-tier via resistance (`r_seg`) — both are `TODO(calibration)` defaults.
  Lee/Ha both note DRAM performance is PDN-voltage-drop-limited but **unmodeled**
  in public tools, which is why this is built from scratch.

## Equations

Supply current (linear in the existing energy vars; `1 uW/V = 1 uA`):

```
I_supply = ( m.P_dyn + p_leak_bit·m.total_cells ) / V_dd          [uA]
```

Power-via count — the larger of two floors (both linear in I_supply):

```
(EM) N_pv ≥ redundancy · I_supply / I_via                         # current-capacity / electromigration
(IR) N_pv ≥ I_supply · r_seg · L · 1e-6 / (δ_pdn · V_dd)          # IR-drop margin, worst-case top tier
```

- **EM floor** (Stratum): each via carries at most `I_via` (36 mA = 3.6e4 uA over
  `a_via` = 25 µm²), inflated by `redundancy` (2:1).
- **IR floor**: worst-case current to the top tier traverses `L` series via
  segments of `r_seg` [Ω] each, over `N_pv` parallel vias, so the rail droop is
  `V_IR = I_supply · r_seg·L / N_pv` [uA·Ω = uV]; requiring `V_IR ≤ δ_pdn·V_dd`
  gives the floor (the `1e-6` converts uV→V). The **`·L`** is what makes taller
  stacks worse — the central 3D tension (`docs/model.tex` §7).

Two `≥` constraints replace a `max()`: `N_pv` costs volume, so the solver drives
it down to whichever floor binds (no integrality or `max` needed).

Volume — power vias span the full stack, so they subtract from the packing budget:

```
Vol_pdn = a_via · (L · t_layer) · N_pv                            [um^3]
```

Added to the `vol_used` LHS of `m.volume` (`model.py:308`) — **not** to the
`m.power_density` RHS (see Open decisions).

## Parameters (new; all TechSpec unless noted, all TODO(calibration))

| param | meaning | default | basis |
|---|---|---|---|
| `v_dd` | PDN rail supply voltage [V] (distinct from `v_read` CV² swing) | 1.1 | Kim'14 HBM 1.2 V; Lee'26 M3D |
| `i_via` | current a single power via carries [uA] | 36000 | Stratum (36 mA/via) |
| `a_via` | footprint of one power via [µm²] | 25.0 | Stratum (25 µm²/via → 1.44 mA/µm²) |
| `pdn_redundancy` | via over-provisioning factor [-] | 2.0 | Stratum 2:1 |
| `r_seg` | power-via resistance per tier [Ω] | 0.02 | Jiang Cu-Cu <10 mΩ/bond + TSV (range 0.01–0.05) |
| `delta_pdn` | IR-drop budget as fraction of V_dd [-] | 0.05 | industry default (no corpus source; range 0.05–0.10) |

Decoupling-cap area is folded into `pdn_redundancy` at this altitude (not a
separate term). New solver var `m.N_pv` with a derived tight upper bound
`N_pv_ub = max(EM, IR)` evaluated at `I_supply_ub = (P_max·A + p_leak_bit·
total_cells_ub)/V_dd` (keeps spatial B&B fast, matching the existing aux-var
pattern).

## Integration points in `model.py`

1. `TechSpec`: 6 new fields + `__post_init__` asserts (`>0`; `0<delta_pdn<1`).
2. `build_model`: `I_supply` as a `pyo.Expression`; `m.N_pv` `Var`; two `≥`
   `Constraint`s (`pdn_em`, `pdn_ir`); `Vol_pdn` term added into `vol_used`
   (so it flows into `m.volume` automatically).
3. `run_bw_max.py`: add the 6 params to `PARAMS["technology"]`; add `N_pv` /
   `vol_pdn` to the report layout.
4. `config.yaml`: 6 params in `defaults.technology`; per-config overrides where a
   tech differs (e.g. crosspoint weak rails → higher `r_seg`).

## Validation targets (magnitudes the model must reproduce)

- Default 800 mm² / 1.1 W/mm² config: **EM floor binds** (~4·10⁴ vias, `Vol_pdn`
  ≈ 0.14 % of the volume budget) — PDN is a small but nonzero packing tax.
- IR-drop floor should overtake EM only at finer via pitch (small `a_via`), taller
  stacks (large `L`), or weaker rails (large `r_seg`) — verify with a sweep.
- Stratum sanity check: 104 W DRAM / 1.1 V / 36 mA → ~2600 vias × 2 redundancy ≈
  0.13 mm² (Stratum reports 0.21 mm² for DRAM+logic); order-of-magnitude match.

## Open decisions

1. **Both floors included** (user, 2026-07-14) — EM (paper-calibrated) + IR-drop
   (carries the L-dependence), even though EM binds at default configs.
2. **Thermal denominator** — `Vol_pdn` enters the packing budget (`m.volume`) but
   is kept **out** of the `m.power_density` RHS: power vias occupy footprint but
   dissipate ~no heat, so counting them in the `A_used` proxy would spuriously
   relax the thermal cap. This intersects the in-flight ΔT=P·R_th work — final
   call belongs to the thermal formulation.
3. **`v_dd` reuse** — kept as a separate param rather than reusing `v_read`, since
   the rail supply ≠ the CV² read swing (partial-swing modes differ).

## Implementation order (when unblocked, after the thermal edits land)

1. `TechSpec` params + asserts.
2. `I_supply` Expression, `m.N_pv` Var (derived bound), `pdn_em`/`pdn_ir`
   Constraints, `Vol_pdn` into `vol_used`.
3. `run_bw_max.py` PARAMS + report; `config.yaml` defaults + per-tech overrides.
4. Magnitude test in `tests/` (EM-binds default; IR-binds at large L/r_seg; the
   Stratum 0.21 mm² order-of-magnitude check).
