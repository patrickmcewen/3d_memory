"""Pyomo model for the 3d_memory phase-1 bandwidth-maximization MINLP.

Descended from ``opt/bw_max.mod`` (retained as a phase-1 validation oracle).
Phase 2 (2026-07-14) replaced the margin/Pelgrom sense-area law with a
technology-dependent **sense-settling** develop-time model derived from Upton
2024 (App. A current-mode, App. B voltage-mode); see :func:`develop_coeffs`.
Technology branches (destructive read; current- vs voltage-sense settling)
are plain Python ``if`` statements here -- the reason the model moved off AMPL.

The AMPL ``param`` declarations carried validation (``> 0``, ``integer``,
``binary``); the dataclass ``__post_init__`` hooks re-add exactly those checks
so a bad config fails loud before the solver runs.

Units: time [ns], length [um], area [um^2], volume [um^3], capacity [bit].
Sense-model unit system (see develop_coeffs): cap [fF], resistance [ohm],
current [uA], voltage [V]  ->  1 fF*V/uA = 1 ns and 1 ohm*fF = 1e-6 ns.
"""
import math
from dataclasses import dataclass

import pyomo.environ as pyo


@dataclass
class ProblemSpec:
    """Given problem inputs (AMPL ``param C, A, L, t_layer``)."""
    C: float          # target capacity                [bit]
    A: float          # footprint (area per layer)     [um^2]
    L: int            # layer budget                   [layers]
    t_layer: float    # physical thickness per layer   [um]

    def __post_init__(self):
        assert self.C > 0, f"C must be > 0, got {self.C}"
        assert self.A > 0, f"A must be > 0, got {self.A}"
        assert self.L > 0 and float(self.L).is_integer(), f"L must be a positive integer, got {self.L}"
        self.L = int(self.L)
        assert self.t_layer > 0, f"t_layer must be > 0, got {self.t_layer}"

    @property
    def vol_budget(self) -> float:
        """AMPL ``param VolBudget := A * L * t_layer``  [um^3]."""
        return self.A * self.L * self.t_layer


@dataclass
class TechSpec:
    """Technology / calibration coefficients.

    WL develop keeps the phase-1 wire/cell split (k_wire_WL, k_cell_WL); the BL
    develop time is now the sense-settling model, so the old k_wire_BL/k_cell_BL
    are replaced by the physical params (c_bl, r_bl, and the per-mode set) that
    :func:`develop_coeffs` turns into settle-time coefficients.
    """
    k_dec: float       # decoder delay per address bit               [ns]
    k_wire_WL: float   # WL wire self-RC coeff       (per column^2)   [ns]
    k_cell_WL: float   # WL cell/gate loading coeff  (per column)     [ns]
    t_SA0: float       # strongARM latch/regeneration floor          [ns]
    t_restore: float   # restore time for a destructive read         [ns]
    destructive: int   # 1 => fold restore into t_SA (DRAM-like)
    t_sw: float        # selector switching time                     [ns]
    v_cell: float      # volume per stored bit                       [um^3/bit]
    v_sa0: float       # sense-amp volume (constant per amp)         [um^3]
    k_vdec: float      # decoder volume per array cell               [um^3/cell]
    v_sel: float       # selector volume per (share x sense-amp)     [um^3]
    # ---- sense-settling model (Upton 2024 App. A/B) ----
    sense_mode: str    # 'current' (App. A) or 'voltage' (App. B)
    settle_frac: float # Delta: fraction of steady state to settle to (e.g. 0.99)
    c_bl: float        # BL parasitic capacitance per cell            [fF/cell]
    r_bl: float        # BL wire resistance per cell pitch            [ohm/cell]
    i_read: float      # steady-state read current  (current mode)   [uA]
    n_ut: float        # subthreshold product n*Ut  (current mode)   [V]
    r_pullup: float    # pull-up resistance         (voltage mode)   [ohm]
    v_ratio: float     # V_BL,CELL / V_READ divider  (voltage mode)  [-]

    def __post_init__(self):
        assert self.k_dec > 0, f"k_dec must be > 0, got {self.k_dec}"
        assert self.k_wire_WL > 0, f"k_wire_WL must be > 0, got {self.k_wire_WL}"
        assert self.k_cell_WL >= 0, f"k_cell_WL must be >= 0, got {self.k_cell_WL}"
        assert self.t_SA0 > 0, f"t_SA0 must be > 0, got {self.t_SA0}"
        assert self.t_restore >= 0, f"t_restore must be >= 0, got {self.t_restore}"
        assert self.destructive in (0, 1), f"destructive must be 0 or 1, got {self.destructive}"
        assert self.t_sw >= 0, f"t_sw must be >= 0, got {self.t_sw}"
        assert self.v_cell > 0, f"v_cell must be > 0, got {self.v_cell}"
        assert self.v_sa0 > 0, f"v_sa0 must be > 0, got {self.v_sa0}"
        assert self.k_vdec > 0, f"k_vdec must be > 0, got {self.k_vdec}"
        assert self.v_sel > 0, f"v_sel must be > 0, got {self.v_sel}"
        assert self.sense_mode in ("current", "voltage"), f"sense_mode must be 'current'|'voltage', got {self.sense_mode!r}"
        assert 0 < self.settle_frac < 1, f"settle_frac (Delta) must be in (0,1), got {self.settle_frac}"
        assert self.c_bl > 0, f"c_bl must be > 0, got {self.c_bl}"
        assert self.r_bl >= 0, f"r_bl must be >= 0, got {self.r_bl}"
        assert self.i_read > 0, f"i_read must be > 0, got {self.i_read}"
        assert self.n_ut > 0, f"n_ut must be > 0, got {self.n_ut}"
        assert self.r_pullup > 0, f"r_pullup must be > 0, got {self.r_pullup}"
        assert self.v_ratio > 0, f"v_ratio must be > 0, got {self.v_ratio}"


@dataclass
class Bounds:
    """Finite variable bounds (required for nonconvex spatial branch-and-bound)."""
    NBL_min: float
    NBL_max: float
    NWL_min: float
    NWL_max: float
    Nshare_min: int
    Nshare_max: int
    Nindep_max: float
    BW_max: float

    def __post_init__(self):
        assert 0 < self.NBL_min <= self.NBL_max, f"need 0 < NBL_min <= NBL_max, got {self.NBL_min}, {self.NBL_max}"
        assert 0 < self.NWL_min <= self.NWL_max, f"need 0 < NWL_min <= NWL_max, got {self.NWL_min}, {self.NWL_max}"
        assert float(self.Nshare_min).is_integer() and float(self.Nshare_max).is_integer(), "Nshare bounds must be integer"
        self.Nshare_min, self.Nshare_max = int(self.Nshare_min), int(self.Nshare_max)
        assert 0 < self.Nshare_min <= self.Nshare_max, f"need 0 < Nshare_min <= Nshare_max, got {self.Nshare_min}, {self.Nshare_max}"
        assert self.Nindep_max > 0, f"Nindep_max must be > 0, got {self.Nindep_max}"
        assert self.BW_max > 0, f"BW_max must be > 0, got {self.BW_max}"


def develop_coeffs(tech: TechSpec):
    """Sense-settling develop-time coefficients (Upton 2024, App. A/B).

    Returns ``(f_margin, a_lin, a_quad)`` for the bitline settle time

        t_develop = f_margin * (a_lin * N_WL + a_quad * N_WL**2)      [ns]

    with ``N_WL`` = cells along the bitline (rows). The physics is derived here,
    in tested Python, rather than as solver algebra.

    Unit system: cap [fF], resistance [ohm], current [uA], voltage [V]; then
    ``1 fF*V/uA = 1 ns`` and ``1 ohm*fF = 1e-6 ns`` (RC in fs -> ns).

    Quadratic term = distributed BL wire self-RC ``1/2 * r_bl * c_bl`` (the
    shared wire-stack term, mode-independent to first order). Linear term is the
    per-mode signal-development time constant per row:
      * current (App. A, Eq. A.12): tau_C = C_BL * n*Ut / I_read, C_BL = c_bl*N
        -> a_lin = c_bl * n_ut / i_read.  f_margin = -ln(1-Delta) (first-order
        settle of the current signal to within (1-Delta) of steady state).
      * voltage (App. B, Eqs. B.7/B.8): Elmore tau = v_ratio*(R_pullup+R_BL/2)*C_BL;
        pull-up dominates the linear term -> a_lin = v_ratio * r_pullup * c_bl,
        and the divider ratio also scales the wire term. f_margin = -2 ln(1-Delta)
        (B.8: T_clk >= -2 tau ln(1-Delta)).
    """
    d = tech.settle_frac
    a_quad = 0.5 * tech.r_bl * tech.c_bl * 1e-6          # BL wire self-RC [ns/cell^2]
    if tech.sense_mode == "current":
        a_lin = tech.c_bl * tech.n_ut / tech.i_read      # tau_C per row   [ns/cell]
        f_margin = -math.log(1.0 - d)
    else:  # 'voltage' (validated in __post_init__)
        a_lin = tech.v_ratio * tech.r_pullup * tech.c_bl * 1e-6   # [ns/cell]
        a_quad *= tech.v_ratio                            # divider prefactor on wire term
        f_margin = -2.0 * math.log(1.0 - d)
    return f_margin, a_lin, a_quad


def build_model(problem: ProblemSpec, tech: TechSpec, bounds: Bounds) -> pyo.ConcreteModel:
    """Assemble the BW-max MINLP as a Pyomo ConcreteModel.

    A nonconvex MIQCP plus one univariate function constraint (log in t_dec)
    that Gurobi 13 (nonconvex=2) handles directly. (Phase 2 removed the
    1/margin^2 function constraint along with the margin variable.)
    """
    m = pyo.ConcreteModel(name="bw_max")
    b = bounds

    # ---- decision variables ------------------------------------------------
    m.N_BL = pyo.Var(bounds=(b.NBL_min, b.NBL_max))       # bitlines  (array columns)
    m.N_WL = pyo.Var(bounds=(b.NWL_min, b.NWL_max))       # wordlines (array rows)
    m.b_acc = pyo.Var(bounds=(1, b.NBL_max))              # bits per access (sense width)
    m.N_share = pyo.Var(domain=pyo.Integers, bounds=(b.Nshare_min, b.Nshare_max))  # arrays sharing one periph set
    m.N_indep = pyo.Var(bounds=(1, b.Nindep_max))         # independent peripheral sets
    m.t_cycle = pyo.Var(bounds=(tech.t_SA0 * 0.5, None))  # steady-state cycle time (lower bound only)
    m.BW = pyo.Var(bounds=(0, b.BW_max))                  # objective

    # ---- defined timing terms as Expressions -------------------------------
    # Pure functions of the decision vars (no free variable of their own).
    f_margin, a_lin, a_quad = develop_coeffs(tech)
    m.t_dec = pyo.Expression(expr=tech.k_dec * pyo.log(m.N_WL) / math.log(2))       # decode depth ~ log2(rows)
    m.t_WL = pyo.Expression(expr=tech.k_wire_WL * m.N_BL**2 + tech.k_cell_WL * m.N_BL)  # WL: wire self-RC + cell load
    # BL develop = sense-signal settling (Upton App. A/B): mode-dependent linear
    # term + shared wire self-RC, times the Delta settle factor. Replaces the
    # phase-1 k_wire_BL/k_cell_BL polynomial and the constant read time.
    m.t_BL = pyo.Expression(expr=f_margin * (a_lin * m.N_WL + a_quad * m.N_WL**2))
    # t_SA is the un-hideable latch floor; restore folds in iff destructive (a
    # build-time constant, not a solver decision).
    t_sa_val = tech.t_SA0 + tech.destructive * tech.t_restore
    m.t_SA = pyo.Expression(expr=t_sa_val)
    m.sum_dev = pyo.Expression(expr=m.t_dec + m.t_WL + m.t_BL + m.t_SA + tech.t_sw) # full serial develop latency

    # ---- auxiliary product variables (tight bounds keep spatial B&B fast) --
    m.cells_arr = pyo.Var(bounds=(0, b.NBL_max * b.NWL_max))
    m.N_tot = pyo.Var(bounds=(0, b.Nindep_max * b.Nshare_max))
    m.N_SA = pyo.Var(bounds=(0, b.Nindep_max * b.NBL_max))
    m.total_cells = pyo.Var(bounds=(0, b.NBL_max * b.NWL_max * b.Nindep_max * b.Nshare_max))
    m.cells_x_indep = pyo.Var(bounds=(0, b.NBL_max * b.NWL_max * b.Nindep_max))
    m.sel_term = pyo.Var(bounds=(0, b.Nshare_max * b.Nindep_max * b.NBL_max))

    m.def_cells_arr = pyo.Constraint(expr=m.cells_arr == m.N_BL * m.N_WL)
    m.def_Ntot = pyo.Constraint(expr=m.N_tot == m.N_indep * m.N_share)
    m.def_NSA = pyo.Constraint(expr=m.N_SA == m.N_indep * m.b_acc)          # = N_indep * bits/access
    m.def_total_cells = pyo.Constraint(expr=m.total_cells == m.cells_arr * m.N_tot)
    m.def_cells_x_indep = pyo.Constraint(expr=m.cells_x_indep == m.cells_arr * m.N_indep)
    m.def_sel_term = pyo.Constraint(expr=m.sel_term == m.N_share * m.N_SA)

    # ---- kernel constraints ------------------------------------------------
    m.cyc_dec = pyo.Constraint(expr=m.t_cycle >= m.t_dec)                    # decode floor
    m.cyc_sa = pyo.Constraint(expr=m.t_cycle >= m.t_SA + tech.t_sw)          # latch+switch floor
    m.cyc_dev = pyo.Constraint(expr=m.t_cycle * m.N_share >= m.sum_dev)      # develop amortized over shared arrays
    m.bw_def = pyo.Constraint(expr=m.BW * m.t_cycle == m.N_SA)               # BW = N_SA / t_cycle
    m.width_cap = pyo.Constraint(expr=m.b_acc <= m.N_BL)                     # cannot sense more bits than bitlines
    m.capacity = pyo.Constraint(expr=m.total_cells >= problem.C)             # meet target capacity
    m.volume = pyo.Constraint(                                              # 3D volume packing budget
        expr=tech.v_cell * m.total_cells
        + tech.v_sa0 * m.N_SA                                               # SA volume: constant per amp
        + tech.k_vdec * m.cells_x_indep
        + tech.v_sel * m.sel_term
        <= problem.vol_budget
    )

    # ---- objective ---------------------------------------------------------
    m.bandwidth = pyo.Objective(expr=m.BW, sense=pyo.maximize)
    return m
