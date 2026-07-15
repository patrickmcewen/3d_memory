"""Regression + physics tests for the BW-max model.

Two layers:
  * ``develop_coeffs`` unit tests -- the phase-2 sense-settling derivation
    (Upton 2024 App. A/B). These need no solver and check the physics against
    literature magnitudes (PCM ~66 ns, SOT-MRAM ~4 ns at a 512-row subarray).
  * End-to-end solve -- guards that the wired-up model builds and solves and
    that every def_* product-variable equality holds at the solution.

Run from the repo (solver tests need the bundled Gurobi ASL driver on this node):
    opt/.venv/bin/python -m pytest opt/tests -q
"""
import math
import sys
from pathlib import Path

import pytest
import pyomo.environ as pyo
import yaml

OPT = Path(__file__).resolve().parents[1]        # .../3d_memory/opt
sys.path.insert(0, str(OPT))

from run_bw_max import (merge, apply_fixes, make_specs, make_solver,  # noqa: E402
                        DEFAULT_GUROBI, BUILD, SOLVED)
from model import build_model, develop_coeffs, energy_coeffs         # noqa: E402
from pyomo.common.tempfiles import TempfileManager                  # noqa: E402

@pytest.fixture(scope="module")
def cfg():
    doc = yaml.safe_load((OPT / "config.yaml").read_text())
    return doc["defaults"], doc["configs"]


def _spec(name, defaults, configs, block=None):
    merged = merge(defaults, configs[name])
    apply_fixes(merged)
    specs = make_specs(merged)
    return specs if block is None else specs[{"problem": 0, "tech": 1, "bounds": 2}[block]]


# ---- develop_coeffs physics (no solver) --------------------------------------

def test_develop_coeffs_settle_magnitudes(cfg):
    """t_develop at a 512-row subarray tracks cited read times."""
    defaults, configs = cfg
    def t_dev(name, N=512):
        _, tech, _ = _spec(name, defaults, configs)
        f, a_lin, a_quad = develop_coeffs(tech)
        return f * (a_lin * N + a_quad * N * N)
    assert t_dev("pcram_16nm") == pytest.approx(61.5, rel=0.1)    # NVSim ~66 ns
    assert t_dev("sotmram_16nm") == pytest.approx(5.2, rel=0.15)  # DESTINY ~3.8 ns
    assert 4.0 < t_dev("reram_16nm") < 12.0                       # EMBER ~5 ns, weak 1T1R read


def test_develop_coeffs_margin_and_mode(cfg):
    """f_margin is the settling latency -ln(1-Delta): mode-INDEPENDENT (no B.8
    dual-edge 2x), so two modes at the same settle_frac share it; tighter
    settling costs more time."""
    defaults, configs = cfg
    _, cur, _ = _spec("reram_16nm", defaults, configs)     # current, Delta=0.99
    _, volt, _ = _spec("pcram_16nm", defaults, configs)    # voltage, Delta=0.99
    f_cur, _, _ = develop_coeffs(cur)
    f_volt, _, _ = develop_coeffs(volt)
    assert f_volt == pytest.approx(f_cur, rel=1e-9)              # same Delta -> same factor across modes
    assert f_cur == pytest.approx(-math.log(1 - cur.settle_frac), rel=1e-9)

    _, base, _ = _spec("gaincell_100Mb", defaults, configs)
    _, tight, _ = _spec("gaincell_tight_settle", defaults, configs)
    assert develop_coeffs(tight)[0] > develop_coeffs(base)[0]     # 99.9% > 99% settle


def test_charge_share_keeps_distributed_quadratic_and_caps_rows(cfg):
    """DRAM/eDRAM charge-share: lumped charge-redistribution linear term PLUS the
    distributed BL wire self-RC quadratic (kept for the long 3D-bitline regime,
    same term the other modes carry), and the signal-collapse row cap tightens
    N_WL below the raw NWL_max (DESTINY SubArray.cpp:542)."""
    defaults, configs = cfg
    _, tech, bounds = _spec("edram_16nm", defaults, configs)
    f, a_lin, a_quad = develop_coeffs(tech)
    assert a_quad == pytest.approx(0.5 * tech.r_bl * tech.c_bl * 1e-6, rel=1e-9)  # distributed wire self-RC
    assert a_lin == pytest.approx(tech.r_bl * tech.c_cell * 1e-6, rel=1e-9)       # lumped charge-redistribution

    # Signal C_cell/(C_cell + c_bl*N_WL) >= margin_sa caps N_WL to a constant.
    nwl_cap = tech.c_cell * (1 - tech.margin_sa) / (tech.c_bl * tech.margin_sa)
    m = build_model(_spec("edram_16nm", defaults, configs)[0], tech, bounds)
    assert m.N_WL.ub == pytest.approx(min(bounds.NWL_max, nwl_cap), rel=1e-9)
    assert m.N_WL.ub < bounds.NWL_max                            # actually binding


# ---- energy_coeffs physics (no solver) ---------------------------------------

def test_energy_bit_magnitudes(cfg):
    """E_bit at a reference row-overfetch access tracks literature pJ/bit.

    Evaluated at a fixed narrow-column subarray (activate 512 bitlines, read 8)
    -- the commodity-access regime the O'Connor/NVMExplorer anchors are quoted
    for. The BW-max solver drives b_acc->N_BL (wide access), which is strictly
    MORE energy-efficient per bit, so these are upper references, not the optimum.
    """
    defaults, configs = cfg
    def e_bit_pj(name, N_BL=512, N_WL=512, b_acc=8):
        _, tech, _ = _spec(name, defaults, configs)
        k_col, k_arr = energy_coeffs(tech)
        e_access = (k_col * N_BL + k_arr * N_BL * N_WL + tech.e_periph
                    + tech.write_fraction * tech.e_write_cell * b_acc)
        return e_access / b_acc * 1e-3                       # fJ/bit -> pJ/bit
    assert 1.0 < e_bit_pj("dram_100Mb") < 20.0               # O'Connor: FGDRAM ~2, GDDR5 14
    assert 1.0 < e_bit_pj("edram_16nm") < 20.0
    assert 0.1 < e_bit_pj("sram_16nm") < 20.0                # NVMExplorer SRAM read O(pJ) narrow access


def test_charge_share_read_uses_full_rail_restore(cfg):
    """A destructive charge-share read restores at the full rail, so its CV^2
    swing is v_read^2 (not the partial v_read*v_sense develop signal)."""
    defaults, configs = cfg
    _, tech, _ = _spec("dram_100Mb", defaults, configs)
    assert tech.destructive == 1
    k_col, _ = energy_coeffs(tech)
    assert k_col == pytest.approx(tech.c_cell * tech.v_read**2, rel=1e-9)


# ---- end-to-end golden solves ------------------------------------------------

def _solve_bw(name, defaults, configs):
    problem, tech, bounds = _spec(name, defaults, configs)
    m = build_model(problem, tech, bounds)
    BUILD.mkdir(exist_ok=True)
    TempfileManager.tempdir = str(BUILD)
    opt = make_solver(merge(defaults, configs[name]), DEFAULT_GUROBI)
    results = opt.solve(m, load_solutions=False, tee=False)
    assert results.solver.termination_condition in SOLVED
    assert len(results.solution) > 0, f"{name}: no incumbent"
    m.solutions.load_from(results)
    return m, pyo.value(m.BW)


@pytest.mark.skipif(not DEFAULT_GUROBI.exists(),
                    reason="bundled Gurobi ASL driver not present")
def test_aux_var_equalities_feasible(cfg):
    """Every def_* product-variable equality must hold at the solution."""
    defaults, configs = cfg
    m, _ = _solve_bw("reram_16nm", defaults, configs)
    from pyomo.core.expr import identify_variables
    for c in m.component_objects(pyo.Constraint):
        if c.local_name.startswith("def_"):
            resid = abs(pyo.value(c.body) - pyo.value(c.lower))
            # Relative tolerance: these product vars reach ~1e12, so an absolute
            # bound is below Gurobi's own feasibility tolerance at that magnitude.
            scale = max([1.0] + [abs(v.value) for v in identify_variables(c.body)])
            assert resid < 1e-9 * scale, f"{c.local_name} rel residual {resid/scale:.3e}"
