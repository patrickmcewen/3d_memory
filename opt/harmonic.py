#!/usr/bin/env python3
"""Analytical evaluator for the harmonic memory structure (docs/HARMONIC_MODEL_SPEC.md).

Reuses the timing (`develop_coeffs`) and energy (`energy_coeffs`) physics from
`model.py` and the YAML technology/footprint configs from `config.yaml`
(loaded via `run_bw_max`), but adds a closed-form geometric area/routing model
for the harmonic tiling. Phase-1 goal: sweep the harmonic depth `n` and the
sense-amp-bank depth `k_bank`, reporting the bandwidth / capacity-overhead
trade-off the structure buys per sense amp.

    python opt/harmonic.py --config gaincell_100Mb --sweep
    python opt/harmonic.py --config sram_dest --n 4 --amort wl_bl --mux flat

Units follow model.py: time [ns], length [um], area [um^2], volume [um^3].
"""
import argparse
import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager

from model import develop_coeffs, energy_coeffs, ProblemSpec, TechSpec
from run_bw_max import merge, make_specs, make_solver, DEFAULT_GUROBI, SOLVED, BUILD

HERE = Path(__file__).resolve().parent


@dataclass
class HarmonicSpec:
    """Harmonic-structure knobs (everything not already in ProblemSpec/TechSpec)."""
    W_wl: float        # reference wordline width [cells] -> sets t_WL (amortizable window)
    k_bank: int        # time-interleaved sense-amp-bank depth (>=1; 1 = no bank)
    amort_mode: str    # 'wl_only' (t_amort=t_WL) | 'wl_bl' (t_amort=t_WL+t_BL)
    mux_topology: str  # 'tree' (log2 stages) | 'flat' (single n:1 bank)
    W_BL: float = 1.0  # cells sharing one vertical bitline (>=1); >1 stacks in Z -> more capacity+tiers, no BW gain

    def __post_init__(self):
        assert self.W_wl >= 1, f"W_wl must be >= 1, got {self.W_wl}"
        assert self.W_BL >= 1, f"W_BL must be >= 1, got {self.W_BL}"
        assert self.k_bank >= 1 and float(self.k_bank).is_integer(), f"k_bank must be a positive integer, got {self.k_bank}"
        self.k_bank = int(self.k_bank)
        assert self.amort_mode in ("wl_only", "wl_bl"), f"amort_mode must be 'wl_only'|'wl_bl', got {self.amort_mode!r}"
        assert self.mux_topology in ("tree", "flat"), f"mux_topology must be 'tree'|'flat', got {self.mux_topology!r}"


def mux_switch_time(n: int, tech: TechSpec, topology: str) -> float:
    """Per-sense mux settling added to t_SA (see spec [D3]): a binary tree pays
    one t_sw per stage (ceil(log2 n)); a flat n:1 bank pays a single t_sw."""
    if n <= 1:
        return 0.0
    return math.ceil(math.log2(n)) * tech.t_sw if topology == "tree" else tech.t_sw


def mux_layers(n: int, topology: str) -> int:
    """Z-layers the mux stack consumes (spec [D3]): log2(n) for a binary tree,
    ~n for a flat n:1 pass-gate bank."""
    if n <= 1:
        return 0
    return max(1, math.ceil(math.log2(n))) if topology == "tree" else n


def reach(tech: TechSpec, harm: HarmonicSpec):
    """Amortizable window, per-sense time, and reach ratio R (spec [D1]).

    Returns (t_WL, t_BL, t_amort, t_SA_eff_ref, R) where R = t_amort/t_SA_eff at
    the fan-in that its own n implies -- so R is evaluated per-n by the caller;
    here we return the window and the fixed floor, and R for a given n via r_of_n.
    """
    f_margin, a_lin, a_quad = develop_coeffs(tech)
    t_WL = tech.k_wire_WL * harm.W_wl**2 + tech.k_cell_WL * harm.W_wl       # WL-settle at ref width
    t_BL = f_margin * (a_lin * harm.W_BL + a_quad * harm.W_BL**2)           # bitline develop over W_BL cells
    t_settle = t_WL + (t_BL if harm.amort_mode == "wl_bl" else 0.0)         # sensing window the SA multiplexes in
    t_sa_floor = tech.t_SA0 + tech.destructive * tech.t_restore
    return t_WL, t_BL, t_settle, t_sa_floor


def energy_per_bit(tech: TechSpec, harm: HarmonicSpec) -> float:
    """Per-sensed-bit access energy [fJ/bit] (spec Energy; one bit per sense so
    b_acc=1). A function of W_BL/W_wl only -- constant in n and k_bank, so it is a
    coefficient (not a variable) in the MINLP."""
    k_col, k_arr = energy_coeffs(tech)
    return (k_col + k_arr * harm.W_BL                          # bitcell + BL wire (spans W_BL cells)
            + tech.e_periph + tech.e_periph_col * harm.W_wl    # fixed + per-column periphery
            + tech.e_sa_read                                   # sense-amp read
            + tech.write_fraction * tech.e_write_cell)         # write-only per-cell term


def evaluate(problem: ProblemSpec, tech: TechSpec, harm: HarmonicSpec, n: int) -> dict:
    """Evaluate one harmonic design point (depth n). Returns a report dict."""
    assert n >= 1 and float(n).is_integer(), f"n must be a positive integer, got {n}"
    n = int(n)

    t_WL, t_BL, t_settle, t_sa_floor = reach(tech, harm)
    t_sw_mux = mux_switch_time(n, tech, harm.mux_topology)
    t_sa_eff = t_sa_floor + t_sw_mux                                        # per-sense time
    R = t_settle / t_sa_eff                                                 # senses per SA in the settle window (decode excluded)
    n_max = math.floor(harm.k_bank * R)                                     # depth ceiling (spec [D1])
    feasible = n <= n_max
    amort = (n + 1) / 2.0                                                   # avg cells per SA [D2]

    # ---- geometry: a triangle unit's footprint is the MAX of its SA plane and
    # the cell plane that feeds it (cells in the upper layers have their own pitch;
    # deep amortization makes that plane wider than the SAs beneath it) ----
    a_sa = tech.v_sa0 / problem.t_layer                                     # SA in-plane footprint [um^2]
    a_cell = tech.v_cell / problem.t_layer                                  # cell in-plane footprint [um^2]
    area_per_sa = max(a_sa, amort * a_cell)                                 # per-SA unit footprint: SA vs its (n+1)/2 cells
    cell_bound = amort * a_cell > a_sa                                      # which plane binds
    N_SA = problem.A / area_per_sa                                         # SAs filling one footprint plane
    capacity = N_SA * amort * harm.W_BL                                     # cells served (bit): (n+1)/2 bitlines x W_BL cells each

    # ---- row decode: one row-select per WL activation among the W_BL wordlines in a
    # group (W_BL=1 -> one WL/group -> no decode, driver only). Amortized over the
    # (n+1)/2 senses, so it taxes the activation period (BW) but not the reach window. ----
    t_dec = tech.k_dec * math.ceil(math.log2(harm.W_BL)) if harm.W_BL > 1 else 0.0
    t_amort = t_settle + t_dec                                             # activation period the SA amortizes over

    # ---- bandwidth at fixed footprint (spec [D4]) ----
    BW = N_SA * amort / t_amort                                             # bit/ns (each SA drains its segment/window)
    BW_per_SA = amort / t_amort                                            # amortization headline
    # SA economy: one-SA-per-cell serves the same capacity with `amort`x the SA count;
    # in footprint terms the saving saturates at a_sa/a_cell once cell-bound.
    sa_count_ratio = amort                                                  # (n+1)/2x fewer sense amps
    footprint_ratio = amort * a_sa / area_per_sa                           # footprint vs 1-SA/cell baseline

    # ---- vertical layer count (replaces the abstract volume) ----
    L_mux = mux_layers(n, harm.mux_topology)
    cell_layers = capacity * a_cell / problem.A                            # cell tiers over footprint ~ (n+1)/2*W_BL*a_cell/area_per_sa
    n_wl = capacity / harm.W_wl                                            # wordlines = total cells / cells-per-WL
    L_wldrv = n_wl * tech.v_wldrv / problem.A                             # one WL driver+decoder (v_wldrv) per wordline, as Z tiers
    L_design = harm.k_bank + L_mux + cell_layers + L_wldrv                # SA bank + mux + cells + WL drivers
    fits = L_design <= problem.L

    # ---- energy / power density (reuse energy_coeffs; one bit per sense) ----
    E_bit = energy_per_bit(tech, harm)                                     # b_acc = 1
    P_dyn = BW * E_bit                                                     # [uW]
    P_leak = tech.p_leak_bit * capacity
    p_density = (P_dyn + P_leak) / problem.A                               # [uW/um^2 == W/mm^2] over the fixed footprint
    bw_thermal = BW * min(1.0, problem.P_max / p_density)                  # P_max-throttled BW

    return dict(
        n=n, k_bank=harm.k_bank, W_BL=harm.W_BL, feasible=feasible, fits=fits, n_max=n_max, R=R, amort=amort,
        t_WL=t_WL, t_BL=t_BL, t_settle=t_settle, t_dec=t_dec, t_amort=t_amort, t_sa_eff=t_sa_eff, t_sw_mux=t_sw_mux,
        a_sa=a_sa, a_cell=a_cell, area_per_sa=area_per_sa, cell_bound=cell_bound,
        N_SA=N_SA, capacity=capacity, n_wl=n_wl,
        L_mux=L_mux, cell_layers=cell_layers, L_wldrv=L_wldrv, L_design=L_design, L_budget=problem.L,
        footprint=problem.A, sa_count_ratio=sa_count_ratio, footprint_ratio=footprint_ratio,
        BW=BW, BW_per_SA=BW_per_SA,
        E_bit=E_bit, P_dyn=P_dyn, P_leak=P_leak,
        p_density=p_density, P_max=problem.P_max, bw_thermal=bw_thermal,
    )


def print_point(name: str, v: dict) -> None:
    feas = "OK" if v["feasible"] else f"INFEASIBLE (n>{v['n_max']}=floor(k_bank*R))"
    fits = "OK" if v["fits"] else f"OVER (needs {v['L_design']:.1f} > L={v['L_budget']})"
    print(f"\n=========== harmonic :: {name}  n={v['n']}, k_bank={v['k_bank']} ===========")
    print(f"  reach   R = t_settle/t_SA_eff = {v['R']:.3g}   n_max = {v['n_max']}   [{feas}]")
    print(f"  timing  t_WL={v['t_WL']:.4g}  t_BL={v['t_BL']:.4g}  t_settle={v['t_settle']:.4g}  +t_dec={v['t_dec']:.4g}  -> t_amort={v['t_amort']:.4g} ns")
    print(f"          t_SA_eff={v['t_sa_eff']:.4g} ns  (mux switch {v['t_sw_mux']:.4g} ns)  decode over log2(W_BL={v['W_BL']:.0f})")
    bind = "cell-bound (upper cell plane)" if v["cell_bound"] else "SA-bound (sense-amp plane)"
    print(f"  footprint/SA    = {v['area_per_sa']:.3g} um^2 = max(a_sa {v['a_sa']:.3g}, (n+1)/2*a_cell {v['amort']*v['a_cell']:.3g})  [{bind}]")
    print(f"  SAs             = {v['N_SA']:.4g}  over footprint {v['footprint']:.3g} um^2")
    print(f"  capacity served = {v['capacity']:.4g} bit  ((n+1)/2*W_BL = {v['amort']*v['W_BL']:.1f} cells/SA, W_BL={v['W_BL']:.0f})")
    print(f"  BW              = {v['BW']*1.25e8:.4g} B/s ({v['BW']*0.125e-3:.4g} TB/s)")
    print(f"  BW / sense amp  = {v['BW_per_SA']:.4g} bit/ns  ((n+1)/2 amortization)")
    print(f"  vs 1-SA/cell    = {v['sa_count_ratio']:.1f}x fewer SAs; footprint {v['footprint_ratio']:.1f}x smaller")
    print(f"  layers  L_design = {v['L_design']:.2f}  (SA bank {v['k_bank']} + mux {v['L_mux']} + cells {v['cell_layers']:.2f} + WLdrv {v['L_wldrv']:.2f})  vs L={v['L_budget']}  [{fits}]")
    print(f"  wordlines       = {v['n_wl']:.3g} (= capacity/W_wl), 1 driver+decoder each")
    print(f"  energy/bit E_bit= {v['E_bit']*1e-3:.4g} pJ/bit   P_dyn={v['P_dyn']*1e-6:.4g} W  P_leak={v['P_leak']*1e-6:.4g} W")
    print(f"  power density   = {v['p_density']:.4g} W/mm^2 ({100*v['p_density']/v['P_max']:.0f}% of P_max)"
          f"   -> thermal-limited BW = {v['bw_thermal']*1.25e8:.4g} B/s")


def print_sweep(name: str, rows: list) -> None:
    print(f"\n########## harmonic sweep :: {name} ##########")
    hdr = (f"{'n':>4}{'R':>8}{'feas':>6}{'bind':>6}{'BW/SA':>9}{'cap[bit]':>11}{'N_SA':>10}"
           f"{'L_dsn':>7}{'fits':>6}{'BW_raw':>11}{'pdens/Pmax':>12}{'BW_thermal':>12}")
    print(hdr)
    print("-" * len(hdr))
    for v in rows:
        print(f"{v['n']:>4}{v['R']:>8.2f}{('y' if v['feasible'] else 'n'):>6}"
              f"{('cell' if v['cell_bound'] else 'SA'):>6}{v['BW_per_SA']:>9.3g}"
              f"{v['capacity']:>11.3g}{v['N_SA']:>10.3g}{v['L_design']:>7.2f}"
              f"{('y' if v['fits'] else 'n'):>6}{v['BW']*1.25e8:>11.2e}"
              f"{v['p_density']/v['P_max']:>12.2e}{v['bw_thermal']*1.25e8:>12.2e}")


def print_thermal_sweep(name: str, rows: list, pmax_levels: list) -> None:
    """Matrix of thermal-limited BW [B/s]: rows = harmonic depth n, columns =
    power-density budget P_max (last column = unlimited). bw_thermal for any P_max
    is BW*min(1, P_max/p_density) -- p_density is P_max-independent, so the n-sweep
    rows are reused directly. The best n per budget is starred."""
    print(f"\n########## harmonic thermal-budget sweep :: {name} ##########")
    print("  thermal-limited BW [B/s]  (P_max = power-density budget, W/mm^2; * = best n in column)")
    cols = "".join(("unlim" if math.isinf(p) else f"{p:.0f}").rjust(12) for p in pmax_levels)
    hdr = f"{'n':>4}{'bind':>6}" + cols
    print(hdr)
    print("-" * len(hdr))
    # best n (max bw_thermal) per column, over feasible+fitting rows only
    ok = [r for r in rows if r["feasible"] and r["fits"]]
    best_n = [max(ok, key=lambda r: r["BW"] * min(1.0, p / r["p_density"]))["n"]
              if ok else None for p in pmax_levels]
    for r in rows:
        cells = ""
        for j, p in enumerate(pmax_levels):
            bwt = r["BW"] * min(1.0, p / r["p_density"]) * 1.25e8
            star = "*" if (r["feasible"] and r["fits"] and r["n"] == best_n[j]) else " "
            cells += f"{bwt:>11.2e}{star}"
        print(f"{r['n']:>4}{('cell' if r['cell_bound'] else 'SA'):>6}{cells}")
    print(f"\n  (thermal-bound columns are flat in n at ~P_max*A/E_bit; the raw-BW knee in n"
          f"\n   reappears only once the budget releases the constraint -> the 'unlim' column.)")


def build_model(problem: ProblemSpec, tech: TechSpec, harm: HarmonicSpec) -> pyo.ConcreteModel:
    """Harmonic BW-max MINLP: maximize sustained bandwidth for a given capacity
    `C`, footprint `A`, and layer budget `L` (spec [D1]-[D4]).

    Free decision variables: the harmonic depth `n` and sense-amp-bank depth
    `k_bank` (both integer), and the bitline stacking `W_BL` (continuous, as in
    the evaluator) -- the vertical lever that lets the layer budget `L` bind
    capacity. `W_wl`, `amort_mode`, and `mux_topology` stay fixed in `harm`.

    Nonconvex MIQCP in `model.py`'s style: explicit bounded auxiliary product
    variables (`prod = N_SA*amort`, `capacity = prod*W_BL`, `P_dyn = BW*E_bit`)
    tie the bilinear couplings, plus univariate `log` function constraints for the
    decode depth `log2(W_BL)` (and the tree-mux depth `log2 n`). Gurobi 13
    (nonconvex=2) handles both, exactly as `model.py` does. `ceil(.)` is relaxed
    to the continuous `log2`, the same simplification `model.py` makes for decode.
    """
    m = pyo.ConcreteModel(name="harmonic_bw_max")
    A, L, P_max, C = problem.A, problem.L, problem.P_max, problem.C
    tree = harm.mux_topology == "tree"
    wl_bl = harm.amort_mode == "wl_bl"

    # ---- constants (W_wl / tech fixed) ----
    f_margin, a_lin, a_quad = develop_coeffs(tech)
    t_WL = tech.k_wire_WL * harm.W_wl**2 + tech.k_cell_WL * harm.W_wl   # WL-settle at ref width [ns]
    t_sa_floor = tech.t_SA0 + tech.destructive * tech.t_restore        # per-sense latch floor [ns]
    a_sa = tech.v_sa0 / problem.t_layer                                # SA in-plane footprint [um^2]
    a_cell = tech.v_cell / problem.t_layer                             # cell in-plane footprint [um^2]
    k_col, k_arr = energy_coeffs(tech)
    e0 = (k_col + tech.e_periph + tech.e_periph_col * harm.W_wl        # E_bit terms independent of W_BL
          + tech.e_sa_read + tech.write_fraction * tech.e_write_cell)  # (E_bit = e0 + k_arr*W_BL)

    # ---- variable bounds (finite -> nonconvex spatial B&B). k_bank <= L (one
    # additive term of L_design >= 0). W_BL <= L: W_BL cells stacked on a bitline
    # occupy W_BL vertical layers, so it cannot exceed the layer budget. n's ceiling
    # is the reach bound n <= k_bank*R at its loosest (t_sa_eff = t_sa_floor, and
    # t_settle at its largest W_BL). ----
    kbank_ub = L
    W_ub = float(L)
    t_settle_max = t_WL + (f_margin * (a_lin * W_ub + a_quad * W_ub**2) if wl_bl else 0.0)
    t_settle_min = t_WL + (f_margin * (a_lin + a_quad) if wl_bl else 0.0)
    n_ub = max(1, math.floor(kbank_ub * t_settle_max / t_sa_floor))
    amort_hi = (n_ub + 1) / 2.0
    area_hi = max(a_sa, amort_hi * a_cell)
    prod_hi = A * amort_hi / a_sa                                      # area_per_sa >= a_sa => prod <= A*amort/a_sa
    log2W_ub = math.log2(W_ub) if W_ub > 1 else 0.0
    ebit_max = e0 + k_arr * W_ub                                       # E_bit at W_BL = W_ub
    # t_cycle spans the compute floor (t_amort) up to the slowest useful throttle
    # (the thermal-bound cycle at max prod / min power headroom); 2x margin.
    t_cycle_ub = 2.0 * max(t_settle_max, prod_hi * ebit_max / (P_max * A))

    # ---- decision variables ----
    m.n = pyo.Var(domain=pyo.Integers, bounds=(1, n_ub))             # harmonic depth
    m.k_bank = pyo.Var(domain=pyo.Integers, bounds=(1, kbank_ub))    # SA-bank depth
    m.W_BL = pyo.Var(bounds=(1, W_ub))                               # cells per vertical bitline (Z stack)
    m.area_per_sa = pyo.Var(bounds=(a_sa, area_hi))                  # per-SA footprint = max(a_sa, amort*a_cell)
    m.prod = pyo.Var(bounds=(0, prod_hi))                           # N_SA*amort (= capacity/W_BL = N_SA served)
    m.capacity = pyo.Var(bounds=(0, prod_hi * W_ub))               # cells served [bit]
    m.t_cycle = pyo.Var(bounds=(t_settle_min, t_cycle_ub))         # sustained activation period (>= t_amort)
    m.BW = pyo.Var(bounds=(0, prod_hi / t_settle_min))            # sustained bit/ns
    m.E_bit = pyo.Var(bounds=(e0, ebit_max))                     # [fJ/bit]
    m.P_dyn = pyo.Var(bounds=(0, P_max * A))                     # [uW]; <= thermal budget
    m.log2W = pyo.Var(bounds=(0, log2W_ub))                      # decode depth log2(W_BL)
    m.def_log2W = pyo.Constraint(expr=m.log2W * math.log(2) == pyo.log(m.W_BL))

    amort = (m.n + 1) / 2.0                                          # avg cells/SA [D2]

    # mux depth: a flat n:1 bank stacks ~n layers and pays one t_sw; a binary tree
    # stacks/pays log2(n) (log2(1)=0 recovers the n=1 base case).
    if tree:
        m.log2n = pyo.Var(bounds=(0, math.log2(n_ub) if n_ub > 1 else 0.0))
        m.def_log2n = pyo.Constraint(expr=m.log2n * math.log(2) == pyo.log(m.n))
        L_mux, t_sw_mux = m.log2n, tech.t_sw * m.log2n
    else:
        L_mux, t_sw_mux = m.n, tech.t_sw

    # sensing window: wl_only -> t_WL (W_BL-independent); wl_bl adds the BL develop
    # over the W_BL-cell stack (needs W_BL^2 -> the Wsq auxiliary keeps it degree-2).
    if wl_bl:
        m.Wsq = pyo.Var(bounds=(1, W_ub**2))
        m.def_Wsq = pyo.Constraint(expr=m.Wsq == m.W_BL * m.W_BL)
        t_settle = t_WL + f_margin * (a_lin * m.W_BL + a_quad * m.Wsq)
    else:
        t_settle = t_WL
    t_amort = t_settle + tech.k_dec * m.log2W                        # min activation period (settle + decode)

    # ---- geometry: area_per_sa = max(a_sa, amort*a_cell), enforced by the two
    # lower bounds + maximize-prod pressure (prod = A*amort/area_per_sa). ----
    m.area_ge_sa = pyo.Constraint(expr=m.area_per_sa >= a_sa)
    m.area_ge_cell = pyo.Constraint(expr=m.area_per_sa >= amort * a_cell)
    m.def_prod = pyo.Constraint(expr=m.prod * m.area_per_sa == A * amort)  # N_SA=A/area_per_sa; prod=N_SA*amort
    m.def_capacity = pyo.Constraint(expr=m.capacity == m.prod * m.W_BL)    # (n+1)/2 bitlines x W_BL cells
    # BW is SUSTAINED: the array can be throttled (run at a cycle >= the physical
    # t_amort floor) to stay within the thermal budget, so t_cycle is a free
    # variable that the power constraint drives up when the design is thermal-bound
    # (mirrors the evaluator's bw_thermal = BW*min(1,P_max/p_density); see model.py).
    m.cyc_floor = pyo.Constraint(expr=m.t_cycle >= t_amort)                # can't beat settle+decode
    m.def_BW = pyo.Constraint(expr=m.BW * m.t_cycle == m.prod)             # each SA drains its segment per cycle

    # ---- reach ceiling n <= k_bank*R, R = t_settle/t_sa_eff (spec [D1]); written
    # division-free as n*t_sa_eff <= k_bank*t_settle. ----
    m.reach = pyo.Constraint(expr=m.n * (t_sa_floor + t_sw_mux) <= m.k_bank * t_settle)

    # ---- the three givens: capacity floor, layer budget, thermal budget ----
    m.capacity_floor = pyo.Constraint(expr=m.capacity >= C)               # "given capacity"
    cell_layers = m.capacity * a_cell / A                                 # cell tiers over footprint
    L_wldrv = m.capacity * tech.v_wldrv / (harm.W_wl * A)                 # WL driver+decoder per wordline
    m.L_design = pyo.Expression(expr=m.k_bank + L_mux + cell_layers + L_wldrv)  # total Z tiers
    m.layers = pyo.Constraint(expr=m.L_design <= L)                       # "given #layers"
    m.def_E_bit = pyo.Constraint(expr=m.E_bit == e0 + k_arr * m.W_BL)     # BL wire spans W_BL cells
    m.def_P_dyn = pyo.Constraint(expr=m.P_dyn == m.BW * m.E_bit)          # dynamic power [uW]
    m.power = pyo.Constraint(expr=m.P_dyn + tech.p_leak_bit * m.capacity <= P_max * A)  # thermal

    m.objective = pyo.Objective(expr=m.BW, sense=pyo.maximize)
    return m


def solve_harmonic(name: str, problem: ProblemSpec, tech: TechSpec, harm: HarmonicSpec,
                   merged: dict, gurobi_asl: Path) -> dict:
    """Build, solve, and report the harmonic BW-max MINLP. The solver returns the
    optimal (n*, k_bank*, W_BL*); the full physics report is then regenerated by
    the analytical `evaluate()` at that point -- one source of truth, and a
    cross-check that the model and evaluator agree (they do up to the ceil-vs-
    continuous-log2 relaxation of the decode / tree-mux depth)."""
    assert problem.mode == "bw_max", f"harmonic --solve is BW-max only, got mode={problem.mode!r}"
    m = build_model(problem, tech, harm)

    BUILD.mkdir(exist_ok=True)
    TempfileManager.tempdir = str(BUILD)
    opt = make_solver(merged, gurobi_asl)
    opt.options["numericfocus"] = 3           # large capacity/energy coefficients -> tighten tolerances

    def _solve(tag):
        logfile = BUILD / f"harmonic_{name}_{tag}.log"
        res = opt.solve(m, load_solutions=False, tee=True, keepfiles=True,
                        symbolic_solver_labels=True, logfile=str(logfile))
        tc = res.solver.termination_condition
        assert tc in SOLVED and len(res.solution) > 0, f"harmonic {tag} solve failed: termination={tc}"
        m.solutions.load_from(res)

    _solve("bw")
    bw_star = pyo.value(m.BW)
    # Stage 2 (lexicographic): the thermal-bound sustained BW depends only on W_BL
    # (via E_bit) and capacity, leaving n/k_bank degenerate -- so among BW-optimal
    # designs pick the LEANEST vertical stack. This collapses the degenerate depths
    # to their minimal feasible values and keeps the design off the layer bound
    # (where the model's continuous log2 would otherwise disagree with evaluate's ceil).
    m.objective.deactivate()
    m.bw_lock = pyo.Constraint(expr=m.BW >= bw_star * (1.0 - 1e-4))
    m.lean = pyo.Objective(expr=m.L_design, sense=pyo.minimize)
    _solve("lean")

    n_star, kbank_star = int(round(pyo.value(m.n))), int(round(pyo.value(m.k_bank)))
    W_star = pyo.value(m.W_BL)
    print(f"\n  MINLP optimum: n*={n_star}, k_bank*={kbank_star}, W_BL*={W_star:.4g}  "
          f"(sustained BW = {pyo.value(m.BW) * 1.25e8:.4g} B/s, {pyo.value(m.L_design):.1f} of {problem.L} layers)")
    v = evaluate(problem, tech,
                 dataclasses.replace(harm, k_bank=kbank_star, W_BL=W_star), n_star)
    print_point(f"{name} [MINLP-optimal]", v)
    return v


def load_specs(config_file: Path, name: str):
    doc = yaml.safe_load(config_file.read_text())
    defaults, configs = doc["defaults"], doc["configs"]
    assert name in configs, f"unknown config '{name}' (available: {list(configs)})"
    merged = merge(defaults, configs[name])
    problem, tech, _bounds = make_specs(merged)
    return problem, tech, merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="gaincell_100Mb", help="config name in config.yaml")
    ap.add_argument("--config-file", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--n", type=int, help="harmonic depth (single point)")
    ap.add_argument("--sweep", action="store_true", help="sweep n = 1..n_max")
    ap.add_argument("--thermal", action="store_true",
                    help="sweep n against a range of P_max budgets (up to unlimited)")
    ap.add_argument("--solve", action="store_true",
                    help="solve the BW-max MINLP: max BW over (n, k_bank) at given C, A, L")
    ap.add_argument("--gurobi", type=Path,
                    default=Path(os.environ.get("GUROBI_ASL", DEFAULT_GUROBI)),
                    help="Gurobi ASL driver (for --solve)")
    ap.add_argument("--k-bank", type=int, default=1, help="sense-amp-bank depth (>=1)")
    ap.add_argument("--w-wl", type=float, default=1024.0, help="reference wordline width [cells]")
    ap.add_argument("--w-bl", type=float, default=1.0, help="cells per vertical bitline (>=1; >1 adds Z tiers, no BW gain)")
    ap.add_argument("--amort", default="wl_only", choices=["wl_only", "wl_bl"])
    ap.add_argument("--mux", default="tree", choices=["tree", "flat"])
    args = ap.parse_args()

    problem, tech, merged = load_specs(args.config_file, args.config)
    harm = HarmonicSpec(W_wl=args.w_wl, k_bank=args.k_bank,
                        amort_mode=args.amort, mux_topology=args.mux, W_BL=args.w_bl)

    if args.solve:
        assert args.gurobi.exists(), f"gurobi ASL driver not found at {args.gurobi} (set --gurobi or $GUROBI_ASL)"
        solve_harmonic(args.config, problem, tech, harm, merged, args.gurobi)
        return 0

    if args.sweep or args.thermal:
        _, _, t_settle, t_sa_floor = reach(tech, harm)
        # sweep up to the reach ceiling (mux floor at n=1); rows self-flag layer fit.
        n_hi = max(1, math.floor(harm.k_bank * t_settle / t_sa_floor))
        rows = [evaluate(problem, tech, harm, n) for n in range(1, n_hi + 1)]
        print_sweep(args.config, rows)
        ok = [r for r in rows if r["feasible"] and r["fits"]]
        best = max(ok, key=lambda r: r["BW"], default=rows[0])
        print(f"\n  best BW (feasible & fits) at n={best['n']}: {best['BW']*1.25e8:.4g} B/s, "
              f"{best['sa_count_ratio']:.1f}x fewer SAs, {best['BW']/rows[0]['BW']:.2f}x the n=1 baseline BW "
              f"({'cell-bound' if best['cell_bound'] else 'SA-bound'})")
        if args.thermal:
            # budgets spanning default P_max up to just past the tightest p_density,
            # then unlimited -- so the thermal-bound -> raw-BW transition is visible.
            pdens_hi = max(r["p_density"] for r in rows)
            decades = math.ceil(math.log10(pdens_hi / problem.P_max)) + 1
            levels = [problem.P_max * 10**k for k in range(0, decades + 1, max(1, decades // 4))]
            levels.append(float("inf"))
            print_thermal_sweep(args.config, rows, levels)
        return 0

    n = args.n if args.n is not None else 1
    print_point(args.config, evaluate(problem, tech, harm, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
