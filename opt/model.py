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
    P_max: float      # cooling power-density budget    [uW/um^2 == W/mm^2]

    def __post_init__(self):
        assert self.C > 0, f"C must be > 0, got {self.C}"
        assert self.A > 0, f"A must be > 0, got {self.A}"
        assert self.L > 0 and float(self.L).is_integer(), f"L must be a positive integer, got {self.L}"
        self.L = int(self.L)
        assert self.t_layer > 0, f"t_layer must be > 0, got {self.t_layer}"
        assert self.P_max > 0, f"P_max must be > 0, got {self.P_max}"

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
    # Peripheral-component volumes (see periph_volume): each named circuit block
    # is charged against the volume budget with its own physical scaling law.
    # A block a technology does not have is disabled by setting its coeff to 0
    # (e.g. charge-share DRAM writes back through the sense amp -> v_wdrv = 0).
    v_cell: float      # volume per stored bit                       [um^3/bit]
    v_sa0: float       # sense-amp volume, per sensed bit            [um^3/bit]
    v_wdrv: float      # write-driver volume, per access bit         [um^3/bit]
    v_pass: float      # BL/SL pass-gate + column-mux, per column    [um^3/col]
    v_pre: float       # precharger, per column                      [um^3/col]
    v_wldrv: float     # WL-driver + row-decoder, per row            [um^3/row]
    v_sel: float       # inter-array sharing selector, per (extra shared array x sense-amp) [um^3]
    v_periph: float    # predecode/control/DAC overhead, per array   [um^3/array]
    # ---- sense-settling model (Upton 2024 App. A/B) ----
    sense_mode: str    # 'current' (App. A) | 'voltage' (App. B) | 'charge_share' (DRAM)
    settle_frac: float # Delta: fraction of steady state to settle to (e.g. 0.99)
    c_bl: float        # BL parasitic capacitance per cell            [fF/cell]
    r_bl: float        # BL wire resistance per cell pitch            [ohm/cell]
    i_read: float      # steady-state read current  (current mode)   [uA]
    n_ut: float        # subthreshold product n*Ut  (current mode)   [V]
    r_pullup: float    # pull-up resistance         (voltage mode)   [ohm]
    v_ratio: float     # V_BL,CELL / V_READ divider  (voltage mode)  [-]
    c_cell: float      # storage cap per cell  (charge-share mode)   [fF]
    margin_sa: float   # min charge-share signal fraction (chg mode) [-]
    # ---- energy / power-density model (see energy_coeffs) ----
    v_read: float      # read/precharge supply swing  (CV^2)         [V]
    v_sense: float     # developed sense signal (charge-share only)  [V]
    e_periph: float    # fixed decode/WL-drive energy per access     [fJ/access]
    e_write_cell: float # per-cell write energy beyond CV^2 (NVM/DRAM) [fJ/bit]
    p_leak_bit: float  # leakage/refresh power per stored bit         [uW/bit]
    write_fraction: float # fraction of accesses that are writes      [-]

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
        assert self.v_wdrv >= 0, f"v_wdrv must be >= 0, got {self.v_wdrv}"   # 0 => no dedicated write driver (charge-share)
        assert self.v_pass >= 0, f"v_pass must be >= 0, got {self.v_pass}"
        assert self.v_pre >= 0, f"v_pre must be >= 0, got {self.v_pre}"
        assert self.v_wldrv >= 0, f"v_wldrv must be >= 0, got {self.v_wldrv}"
        assert self.v_sel > 0, f"v_sel must be > 0, got {self.v_sel}"
        assert self.v_periph >= 0, f"v_periph must be >= 0, got {self.v_periph}"
        assert self.sense_mode in ("current", "voltage", "charge_share"), f"sense_mode must be 'current'|'voltage'|'charge_share', got {self.sense_mode!r}"
        assert 0 < self.settle_frac < 1, f"settle_frac (Delta) must be in (0,1), got {self.settle_frac}"
        assert self.c_bl > 0, f"c_bl must be > 0, got {self.c_bl}"
        assert self.r_bl >= 0, f"r_bl must be >= 0, got {self.r_bl}"
        assert self.i_read > 0, f"i_read must be > 0, got {self.i_read}"
        assert self.n_ut > 0, f"n_ut must be > 0, got {self.n_ut}"
        assert self.r_pullup > 0, f"r_pullup must be > 0, got {self.r_pullup}"
        assert self.v_ratio > 0, f"v_ratio must be > 0, got {self.v_ratio}"
        assert self.c_cell > 0, f"c_cell must be > 0, got {self.c_cell}"
        assert 0 < self.margin_sa < 1, f"margin_sa must be in (0,1), got {self.margin_sa}"
        assert self.v_read > 0, f"v_read must be > 0, got {self.v_read}"
        assert self.v_sense > 0, f"v_sense must be > 0, got {self.v_sense}"
        assert self.e_periph >= 0, f"e_periph must be >= 0, got {self.e_periph}"
        assert self.e_write_cell >= 0, f"e_write_cell must be >= 0, got {self.e_write_cell}"
        assert self.p_leak_bit >= 0, f"p_leak_bit must be >= 0, got {self.p_leak_bit}"
        assert 0 <= self.write_fraction <= 1, f"write_fraction must be in [0,1], got {self.write_fraction}"


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

    def __post_init__(self):
        assert 0 < self.NBL_min <= self.NBL_max, f"need 0 < NBL_min <= NBL_max, got {self.NBL_min}, {self.NBL_max}"
        assert 0 < self.NWL_min <= self.NWL_max, f"need 0 < NWL_min <= NWL_max, got {self.NWL_min}, {self.NWL_max}"
        assert float(self.Nshare_min).is_integer() and float(self.Nshare_max).is_integer(), "Nshare bounds must be integer"
        self.Nshare_min, self.Nshare_max = int(self.Nshare_min), int(self.Nshare_max)
        assert 0 < self.Nshare_min <= self.Nshare_max, f"need 0 < Nshare_min <= Nshare_max, got {self.Nshare_min}, {self.Nshare_max}"
        assert self.Nindep_max > 0, f"Nindep_max must be > 0, got {self.Nindep_max}"


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
      * current (App. A Eq. A.9 / EMBER Eq. 1): tau_C = C_BL * n*Ut / I_read,
        C_BL = c_bl*N -> a_lin = c_bl * n_ut / i_read. The margin factor
        f_margin = -ln(1-Delta) is the App. A Eq. (A.12) settling-time log-bracket
        K evaluated in the thesis-recommended limit I_bias = I_read: with d' = 1-Delta,
        K = ln[(1 - I_read/I_D0) * (1+d')/d'] -> ln((1+d')/d') ~= -ln(1-Delta) for
        I_D0 >> I_read (Delta=0.99: K=ln(101)=4.615 vs -ln(0.01)=4.605, 0.2% apart).
        The omitted (1 - I_read/I_D0) finite-initial-current term would need an I_D0
        (initial BL current) param we do not carry, so we use the leading-order factor.
      * voltage (App. B, Eq. B.7): Elmore tau = v_ratio*(R_pullup+R_BL/2)*C_BL;
        pull-up dominates the linear term -> a_lin = v_ratio * r_pullup * c_bl,
        and the divider ratio also scales the wire term. f_margin = -ln(1-Delta).
        NB: B.8 writes T_clk >= -2 tau ln(1-Delta), but that 2x is a dual-edge
        clock-PERIOD convention (Phase-1 settling occupies a half cycle), not the
        develop LATENCY. sum_dev is a serial latency, so we use the single-tau
        settling time -ln(1-Delta) here -- same factor as current/charge modes,
        which also keeps the SHARED BL wire self-RC (a_quad) consistent across
        modes (else voltage cells would carry a spurious 2x wire term, biasing
        the optimal N_WL differently for voltage vs current techs).
      * charge_share (DRAM/eDRAM, DESTINY SubArray.cpp:542): passive charge
        redistribution. TWO additive components:
          (a) lumped charge-redistribution time R_BL*C_cell (a_lin). This is the
              CACTI-D / CACTI-5.1 / 3D-DATE / DESTINY DRAM convention -- a single
              series-RC whose series cap C_cell*C_BL/(C_cell+C_BL) saturates at
              ~C_cell for C_BL >> C_cell, so it is ~linear in rows. Dominates for
              short bitlines (access-device-limited develop).
          (b) distributed BL wire self-RC 1/2*r_bl*c_bl*N_WL^2 (a_quad, the shared
              default). The DRAM-specific tools drop this, but it overtakes (a)
              once N_WL > ~2*c_cell/c_bl -- exactly the long, thin, resistive
              3D-BEOL bitline regime this tool explores -- so we keep it, matching
              the SRAM/current/voltage modes which all carry the same term.
        The series-cap ratio C_cell/(C_cell+C_BL) governs SIGNAL AMPLITUDE, not
        settling speed, so it enters ONLY as the N_WL collapse bound in
        build_model -- never as a develop-time term.
    """
    d = tech.settle_frac
    a_quad = 0.5 * tech.r_bl * tech.c_bl * 1e-6          # [eq:bl_wire_selfrc]
    if tech.sense_mode == "current":
        a_lin = tech.c_bl * tech.n_ut / tech.i_read      # [eq:bl_settle_current]
        f_margin = -math.log(1.0 - d)
    elif tech.sense_mode == "voltage":
        a_lin = tech.v_ratio * tech.r_pullup * tech.c_bl * 1e-6   # [eq:bl_settle_voltage]
        a_quad *= tech.v_ratio                            # divider prefactor on wire term
        f_margin = -math.log(1.0 - d)                     # settling latency (no dual-edge 2x; see docstring)
    else:  # 'charge_share' (validated in __post_init__)
        a_lin = tech.r_bl * tech.c_cell * 1e-6           # [eq:bl_settle_charge_share]
        f_margin = -math.log(1.0 - d)                    # a_quad kept at default: distributed wire self-RC
    return f_margin, a_lin, a_quad


def energy_coeffs(tech: TechSpec):
    """Per-access dynamic-energy coefficients (DESTINY SubArray::CalculatePower).

    Returns ``(k_col, k_arr)`` for the bitline/BLSA read energy

        E_read_access = k_col * N_BL + k_arr * (N_BL * N_WL)          [fJ]

    from the uniform CV^2 form ``E = (c_cell + c_bl*N_WL) * V^2 * N_BL``, where
    the ``c_bl*N_WL`` term is the whole activated row's distributed bitline cap
    (grows with rows) and ``c_cell`` is the per-bitline cell/access cap. Unit
    system: cap [fF] * V^2 -> fF*V^2 = 1 fJ.

    ``V^2`` per sense mode:
      * voltage / current: full precharge swing ``v_read^2`` (SRAM full-swing
        latch; resistive-read bitline precharge).                  [DESTINY:687/714]
      * charge_share: a DESTRUCTIVE read restores the row at the full rail, so
        it dissipates ``v_read^2`` (write-back dominates DRAM/eDRAM); a hypothetical
        non-destructive charge read develops only the partial signal
        ``v_read*v_sense`` (senseVoltage*vdd).                      [DESTINY:699/702]

    Peripheral (decode/WL) energy, per-cell write energy, and leakage are flat
    per-tech params consumed directly in :func:`build_model`.
    """
    if tech.sense_mode == "charge_share" and not tech.destructive:
        v2 = tech.v_read * tech.v_sense                  # partial-swing charge develop
    else:
        v2 = tech.v_read * tech.v_read                   # full-rail swing (or destructive restore)
    k_col = tech.c_cell * v2                             # [eq:energy_read_access]
    k_arr = tech.c_bl * v2
    return k_col, k_arr


def build_model(problem: ProblemSpec, tech: TechSpec, bounds: Bounds) -> pyo.ConcreteModel:
    """Assemble the BW-max MINLP as a Pyomo ConcreteModel.

    A nonconvex MIQCP plus one univariate function constraint (log in t_dec)
    that Gurobi 13 (nonconvex=2) handles directly. (Phase 2 removed the
    1/margin^2 function constraint along with the margin variable.)
    """
    m = pyo.ConcreteModel(name="bw_max")
    b = bounds

    # Charge-share (DRAM/eDRAM) signal collapse enters as a constant N_WL upper
    # bound (a tightened bound, not a develop-time term); derivation in the ledger.
    nwl_hi = b.NWL_max
    if tech.sense_mode == "charge_share":
        nwl_sig = tech.c_cell * (1.0 - tech.margin_sa) / (tech.c_bl * tech.margin_sa)  # [eq:charge_share_signal_collapse]
        assert nwl_sig >= b.NWL_min, (
            f"charge-share signal margin unsatisfiable: N_WL cap {nwl_sig:.1f} "
            f"< NWL_min {b.NWL_min} (raise c_cell, or lower margin_sa/c_bl)")
        nwl_hi = min(b.NWL_max, nwl_sig)

    # ---- decision variables ------------------------------------------------
    m.N_BL = pyo.Var(bounds=(b.NBL_min, b.NBL_max))       # bitlines  (array columns)
    m.N_WL = pyo.Var(bounds=(b.NWL_min, nwl_hi))          # wordlines (array rows; charge-share signal-capped)
    m.b_acc = pyo.Var(bounds=(1, b.NBL_max))              # bits per access (sense width)
    m.N_share = pyo.Var(domain=pyo.Integers, bounds=(b.Nshare_min, b.Nshare_max))  # arrays sharing one periph set
    m.N_indep = pyo.Var(bounds=(1, b.Nindep_max))         # independent peripheral sets
    m.t_cycle = pyo.Var(bounds=(tech.t_SA0 * 0.5, None))  # steady-state cycle time (lower bound only)
    m.BW = pyo.Var(bounds=(0, None))                  # objective

    # ---- defined timing terms as Expressions -------------------------------
    # Pure functions of the decision vars (no free variable of their own).
    f_margin, a_lin, a_quad = develop_coeffs(tech)
    m.t_dec = pyo.Expression(expr=tech.k_dec * pyo.log(m.N_WL) / math.log(2))       # [eq:decode_depth]
    m.t_WL = pyo.Expression(expr=tech.k_wire_WL * m.N_BL**2 + tech.k_cell_WL * m.N_BL)  # [eq:wl_develop]
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
    m.bl_edge = pyo.Var(bounds=(0, b.NBL_max * b.Nindep_max * b.Nshare_max))   # columns summed over every physical array
    m.wl_edge = pyo.Var(bounds=(0, b.NWL_max * b.Nindep_max * b.Nshare_max))   # rows summed over every physical array
    m.sel_term = pyo.Var(bounds=(0, (b.Nshare_max - 1) * b.Nindep_max * b.NBL_max))

    m.def_cells_arr = pyo.Constraint(expr=m.cells_arr == m.N_BL * m.N_WL)
    m.def_Ntot = pyo.Constraint(expr=m.N_tot == m.N_indep * m.N_share)
    m.def_NSA = pyo.Constraint(expr=m.N_SA == m.N_indep * m.b_acc)          # = N_indep * bits/access
    m.def_total_cells = pyo.Constraint(expr=m.total_cells == m.cells_arr * m.N_tot)
    m.def_bl_edge = pyo.Constraint(expr=m.bl_edge == m.N_BL * m.N_tot)      # per-array column strips, replicated over all arrays
    m.def_wl_edge = pyo.Constraint(expr=m.wl_edge == m.N_WL * m.N_tot)      # per-array row strips, replicated over all arrays
    m.def_sel_term = pyo.Constraint(expr=m.sel_term == (m.N_share - 1) * m.N_SA)  # selectors only for sharing beyond the first array (0 at N_share=1)

    # ---- kernel constraints ------------------------------------------------
    m.cyc_dec = pyo.Constraint(expr=m.t_cycle >= m.t_dec)                    # decode floor
    m.cyc_sa = pyo.Constraint(expr=m.t_cycle >= m.t_SA + tech.t_sw)          # latch+switch floor
    m.cyc_dev = pyo.Constraint(expr=m.t_cycle * m.N_share >= m.sum_dev)      # [eq:cycle_amortization]
    m.bw_def = pyo.Constraint(expr=m.BW * m.t_cycle == m.N_SA)               # [eq:bandwidth_def]
    m.width_cap = pyo.Constraint(expr=m.b_acc <= m.N_BL)                     # cannot sense more bits than bitlines
    m.capacity = pyo.Constraint(expr=m.total_cells >= problem.C)             # meet target capacity
    # Per-component peripheral volume (shared by the volume + power-density
    # constraints). Two scaling classes, set by whether a block is time-shared
    # across the N_share arrays that share one peripheral set:
    #   * SHARED read/write ENDPOINTS -- sense amp + write driver. Only one array
    #     is active per cycle (develop amortized in cyc_dev), so a single bank of
    #     b_acc amps/drivers serves all N_share arrays -> scale with N_SA
    #     (= N_indep * b_acc), NO N_share factor. EMBER pairs the SA and write
    #     driver in one column-muxed slot, so both ride N_SA.
    #   * PER-ARRAY IN-ARRAY STRUCTURES -- BL-side per-column strip (pass gate +
    #     column mux + precharger) and WL-side per-row strip (WL driver + row
    #     decoder). Each physical array owns its own rows and columns, so these
    #     replicate in every array -> scale with bl_edge/wl_edge (carry N_tot).
    #     (Sharing sense amps commits the arrays to bitline-direction stacking,
    #     which precludes sharing WL drivers -- see design notes.)
    #   * v_sel is the distinct INTER-array selector that muxes N_share arrays
    #     onto the shared endpoints; v_periph is the fixed predecode/control/DAC.
    vol_used = (tech.v_cell * m.total_cells                                 # cell array  [eq:periph_volume_scaling]
                + (tech.v_sa0 + tech.v_wdrv) * m.N_SA                       # shared sense-amp + write-driver interface
                + (tech.v_pass + tech.v_pre) * m.bl_edge                    # BL-side per-column strip, every array
                + tech.v_wldrv * m.wl_edge                                  # WL-side per-row strip, every array
                + tech.v_sel * m.sel_term                                   # inter-array sharing selectors
                + tech.v_periph * m.N_tot)                                  # predecode/control/DAC, fixed per array
    m.volume = pyo.Constraint(expr=vol_used <= problem.vol_budget)          # 3D volume packing budget

    # ---- energy / power-density (see energy_coeffs) ------------------------
    # E_access is the blended per-array-access energy: the CV^2 row activation
    # (dominant, geometry-coupled) plus fixed periphery, plus the write-only
    # per-cell term weighted by the write fraction. E_bit amortizes it over the
    # b_acc bits actually delivered (O'Connor row-overfetch: a whole N_BL row
    # swings, only b_acc bits leave). P_dyn = BW * E_bit is the total dynamic
    # power; leakage is flat per stored bit. The power density
    # (P_dyn + P_leak)/A_used stays <= P_max, written division-free by moving
    # A_used = vol_used/(L*t_layer) to the RHS -- reusing the volume polynomial.
    k_col, k_arr = energy_coeffs(tech)
    f_w = tech.write_fraction
    m.E_access = pyo.Expression(
        expr=k_col * m.N_BL + k_arr * m.cells_arr + tech.e_periph
        + f_w * tech.e_write_cell * m.b_acc)                                # [fJ] per single-array access
    E_bit_ub = (k_col * b.NBL_max + k_arr * b.NBL_max * nwl_hi + tech.e_periph
                + f_w * tech.e_write_cell * b.NBL_max)                       # b_acc >= 1 => E_bit <= E_access_max
    m.E_bit = pyo.Var(bounds=(0, E_bit_ub))                                  # [fJ/bit]
    m.P_dyn = pyo.Var(bounds=(0, problem.P_max * problem.A))                 # [uW]; <= P_max * footprint
    m.def_E_bit = pyo.Constraint(expr=m.E_bit * m.b_acc == m.E_access)      # [eq:energy_per_bit_overfetch]
    m.def_P_dyn = pyo.Constraint(expr=m.P_dyn == m.BW * m.E_bit)            # dynamic power [uW] (BW in bit/ns, E in fJ)
    m.power_density = pyo.Constraint(
        expr=m.P_dyn + tech.p_leak_bit * m.total_cells
        <= (problem.P_max / (problem.L * problem.t_layer)) * vol_used)     # [eq:power_density]

    # ---- objective ---------------------------------------------------------
    m.bandwidth = pyo.Objective(expr=m.BW, sense=pyo.maximize)
    return m
