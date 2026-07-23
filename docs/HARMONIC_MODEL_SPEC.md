# Harmonic Memory Structure — Modeling Spec

A standalone analytical evaluator (`opt/harmonic.py`) for the **harmonic memory
structure**: it reuses the timing (`develop_coeffs`) and energy (`energy_coeffs`)
physics from `opt/model.py`, but replaces the volume-coefficient area law with a
**closed-form geometric** area/routing model tailored to the harmonic tiling.
Status: **evaluator + MINLP**. The evaluator gives a fast sweep over the harmonic
depth `n` and the sense-amp-bank depth `k_bank`; `--solve` hands the same physics
to a Pyomo BW-max MINLP (`build_model`, solved with the shared Gurobi ASL driver)
that maximizes sustained bandwidth over `(n, k_bank, W_BL)` for a given `C, A, L`
(see "MINLP solve mode" below).

Units follow `model.py`: time [ns], length [um], area [um^2], volume [um^3],
capacity [bit]. Sense-model params (cap [fF], R [ohm], I [uA], V [V]) are consumed
only through the imported `develop_coeffs`/`energy_coeffs`.

## The structure (physics = clarification "b")

- One shared **wordline** (WL) runs horizontally and, when it fires, activates a
  group of cells simultaneously. Each cell has its own (length-1) **bitline** that
  develops in parallel with all the others during the WL-settle window.
- A **sense amp** (SA) is pitch-limited and expensive, so it is *amortized*: a
  `k`:1 mux gives one SA a **segment** of `k` sibling bitlines, and the SA
  time-multiplexes — senses them one after another — *within a single WL
  activation*. This works because `t_SA < t_WL` (the SA resolves faster than the
  WL settles), so the SA would otherwise sit idle.
- **Harmonic grading.** The physical mux-tree layout produces segments graded
  `1, 2, ..., n` (the staircase in the sketches), so `Sigma k = n(n+1)/2` cells sit
  behind `n` SAs — **`(n+1)/2` cells per SA on average**, i.e. `(n+1)/2x` fewer
  (expensive) SAs than a one-SA-per-cell array of the same capacity.
- **SA-bank extension (`k_bank`).** A demux round-robins the read stream across a
  bank of `k_bank` parallel SAs (recombined by a bottom mux), giving each SA
  `k_bank x` longer to resolve → the reach ceiling lifts `R -> k_bank * R`. Cost:
  `k_bank` SA layers in Z. Flagged *unverified* — modeled, reported separately.

## Decisions pinned here (please confirm — marked [D#])

**[D1] Fixed-window amortization.** The sensing window `t_settle` is a WL-settle
budget computed from `model.py`'s WL law at a reference WL width `W_wl` (a knob),
*not* grown with `n`. The reach ceiling uses the **settle window only** (the SA cannot
sense during decode):

```
t_SA_eff = t_SA0 + destructive * t_restore + t_sw_mux         # per-sense time
R        = t_settle / t_SA_eff                                # senses per SA in the settle window
n_max    = floor(k_bank * R)                                  # depth ceiling
```

with the toggle `amort_mode in {wl_only, wl_bl}`:
`t_settle = t_WL` or `t_WL + t_BL` (does BL develop overlap the SA multiplexing?).

**Row decode (`--w-bl` dependent).** A bitline group stacks `W_BL` cells over `W_BL`
wordlines, so one WL activation must first select the accessed row: `t_dec = k_dec *
ceil(log2 W_BL)` (`W_BL=1` -> one WL per group -> `t_dec=0`, driver only). Decode runs
once per activation and is amortized over the `(n+1)/2` senses, so it taxes the **BW
period** but not reach: `t_amort = t_settle + t_dec` is the throughput denominator,
while `R` stays on `t_settle`. `t_dec` is independent of `n`, so the `(n+1)/2` BW
scaling is preserved at fixed `W_BL`.

**[D2] Grading → average `(n+1)/2` cells per SA.** The physical mux staircase yields
segments graded `1..n`, so a fixed set of `N_SA` sense amps serves `N_SA*(n+1)/2`
cells. The headline is **SA economy**: delivering that capacity/BW with a
one-SA-per-cell array would need `(n+1)/2x` the sense-amp footprint (`sa_area_ratio`).

**[D3] Layer count, not abstract volume (mux-topology toggle).** The design's cost is
reported as **in-plane footprint area + vertical layer count**, not a lumped volume:

```
L_design(n)  = k_bank + L_mux(n) + cell_layers + L_wldrv
L_mux(n)     = ceil(log2 n)   (tree)       # one Z-layer per binary mux stage
             = n              (flat k:1)   # a flat n:1 pass-gate bank stacks ~n
cell_layers  = capacity * a_cell / A       # cell planes stacked over the SA footprint
L_wldrv      = (capacity / W_wl) * v_wldrv / A   # one WL driver+decoder per wordline
```

`t_sw_mux` matches the topology: `ceil(log2 n) * t_sw` (tree) or `t_sw` (flat).
`L_design <= L` (the layer budget) is the packing check (`fits`). The WL-driver term is
**per-wordline, not per-SA**: there are `capacity/W_wl` wordlines (total cells / cells
per WL), each with one `v_wldrv` (driver + row decoder bundled). `L_wldrv` grows with
`n` (more cells -> more wordlines), a real overhead partly offsetting the amortization.

**[D4] Bandwidth at a fixed sense-amp budget.** The SAs are the scarce, pitch-limited
resource, so they tile the footprint. Each SA drains its `(n+1)/2`-avg segment per
window, so `BW = N_SA * (n+1)/2 / t_amort` (bit/ns) and the amortization headline is
`BW_per_SA = (n+1)/2 / t_amort`. Reported alongside: the thermal-limited BW
`bw_thermal = BW * min(1, P_max/p_density)` (usually the binding number).

## Geometry (closed-form, 3D monolithic)

A triangle unit's footprint is the **max** of its sense-amp plane and the upper
cell plane that feeds it (the cells have their own pitch; deep amortization makes
that plane wider than the SAs beneath it). Cells and mux stack above the SAs in Z:

```
a_sa        = v_sa0 / t_layer                # SA in-plane footprint [um^2]
a_cell      = v_cell / t_layer               # cell in-plane footprint [um^2]
area_per_sa = max(a_sa, (n+1)/2 * a_cell)    # per-SA unit footprint = max(SA, its cells)
N_SA        = A / area_per_sa                 # sense amps over the footprint
capacity    = N_SA * (n+1)/2 * W_BL           # cells served [bit]  [D2]
cell_layers = capacity * a_cell / A          # ~ (n+1)/2 * W_BL * a_cell/area_per_sa   Z tiers
```

The `max` creates a depth knee at `(n+1)/2 = a_sa/a_cell`: **SA-bound** below it
(N_SA fixed, BW grows with amortization), **cell-bound** above it (capacity/BW
plateau at `A/a_cell`, N_SA falls — deeper amortization stops paying).

**Bitline length `W_BL` (cells per vertical bitline, `--w-bl`, default 1).** Each of
the `(n+1)/2`-avg bitlines behind an SA holds `W_BL` cells stacked in **Z**, so
`W_BL` multiplies **capacity, leak, BL-develop time, and cell tiers** but leaves the
**footprint and per-window BW untouched** — only one cell per bitline is sensed per WL
activation (the other `W_BL-1` sit on other wordlines). It's a pure density knob: the
Z-tier cost grows as `~n * W_BL` (`cell_layers` above), traded for `W_BL x` capacity at
constant sense-amp count and bandwidth.

## Timing (reuse `model.py::develop_coeffs`)

```
f_margin, a_lin, a_quad = develop_coeffs(tech)
t_WL = k_wire_WL * W_wl^2 + k_cell_WL * W_wl                  # WL-settle at ref width
t_BL = f_margin * (a_lin * W_BL + a_quad * W_BL^2)           # bitline develop over W_BL cells
t_amort = t_WL (+ t_BL if amort_mode == wl_bl)               # [D1]
```

## Footprint + layers (closed-form; replaces model.py's volume coefficients)

The design cost is the fixed in-plane footprint `A` plus the vertical layer count
`L_design(n)` (see [D3]) — no abstract um^3 budget. The SA plane sets the footprint;
mux stages and the (usually <1) cell planes stack above it. `fits = L_design <= L`.
`sa_area_ratio = (n+1)/2` is the SA-footprint saving vs one-SA-per-cell.

## Energy / power density (reuse `model.py::energy_coeffs`)

`energy_coeffs(tech)` gives `(k_col, k_arr)`; a harmonic access reads one cell per
sense so `N_WL_along_bitline = 1`:

```
E_access = k_col + k_arr * W_BL + e_periph + e_periph_col * W_wl
           + e_sa_read + write_fraction * e_write_cell         # per sensed bit (BL wire spans W_BL cells)
E_bit    = E_access                                            # b_acc = 1 (one bit per sense)
P_dyn    = BW * E_bit                                          # [uW]
p_density = (P_dyn + p_leak_bit * capacity) / A <= P_max       # over the fixed footprint
```

## Outputs

Per `(n, k_bank)` point: `R`, reach feasibility (`n <= k_bank*R`), `N_SA`,
`capacity`, `BW`, `BW_per_SA`, `sa_area_ratio` (SA saving), `L_design` + `fits`,
`E_bit`, `p_density`, and `bw_thermal` — plus a sweep over `n` showing the
BW-per-SA / SA-saving gains against the layer + thermal costs (the prototype's
`bw.png`, made rigorous).

## MINLP solve mode (`--solve`, `build_model`)

`harmonic.py --solve` maximizes the **sustained** bandwidth for a given capacity
`C`, footprint `A`, and layer budget `L` (from the config's `ProblemSpec`), with
the harmonic depth `n` and SA-bank depth `k_bank` (integer) and the bitline
stacking `W_BL` (continuous, as in the evaluator) as the free decision variables.
`W_wl`, `amort_mode`, and `mux_topology` stay fixed knobs. It reuses the shared
Gurobi ASL driver and solver plumbing from `run_bw_max.py`.

**Sustained vs peak BW.** The evaluator's `BW = N_SA*(n+1)/2/t_amort` is the
*peak* rate; the array can be throttled (run at a cycle `t_cycle >= t_amort`) to
stay within the thermal budget. The MINLP makes `t_cycle` a free variable with
`t_cycle >= t_amort` and `BW = prod/t_cycle` (`prod = N_SA*(n+1)/2`), so the
power-density constraint `P_dyn + P_leak <= P_max*A` drives `t_cycle` up when the
design is thermal-bound. This reproduces the evaluator's `bw_thermal =
BW*min(1, P_max/p_density)` (mirrors `model.py`'s free `t_cycle`), and the solved
objective equals `evaluate()`'s `bw_thermal` at `(n*, k_bank*, W_BL*)` as a check.

**Structure.** A nonconvex MIQCP in `model.py`'s style — bounded auxiliary
product variables tie the bilinear couplings:

```
prod * area_per_sa = A * (n+1)/2      # N_SA = A/area_per_sa (max via two lower bounds)
capacity           = prod * W_BL      # [D2] x W_BL stacking
BW * t_cycle       = prod             # sustained rate at the throttled cycle
P_dyn              = BW * E_bit        # E_bit = e0 + k_arr*W_BL (linear in W_BL)
```

plus univariate `log` function constraints for the decode depth `log2(W_BL)`
(and, tree mux, the mux depth `log2 n`); `ceil(.)` relaxes to the continuous
`log2`, the same simplification `model.py` makes for its decode depth. The three
givens enter as `capacity >= C`, `L_design <= L`, and the thermal constraint.
Variable bounds are derived from `L` (`k_bank, W_BL <= L`; `n` from the reach
ceiling), so no extra config is needed. `W_BL` is the vertical lever that makes
`L` bind capacity — with `W_BL` fixed, capacity is footprint-bound at `A*W_BL/a_cell`.

**Lexicographic lean stage.** When a design is thermal-bound (the common case),
sustained BW `= (P_max*A - P_leak)/E_bit` depends only on `W_BL` and capacity, so
`n`/`k_bank` are degenerate. A second solve minimizes `L_design` subject to
`BW >= BW* (1 - 1e-4)`, collapsing the degenerate depths to their leanest feasible
values and keeping the design off the layer bound (where the continuous-`log2`
relaxation would otherwise disagree with `evaluate()`'s `ceil`). The reported
point is regenerated by `evaluate()` at the solved `(n*, k_bank*, W_BL*)` — one
source of truth for the physics.
