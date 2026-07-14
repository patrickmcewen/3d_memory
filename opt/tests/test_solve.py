"""Regression + physics tests for the BW-max model.

Two layers:
  * ``develop_coeffs`` unit tests -- the phase-2 sense-settling derivation
    (Upton 2024 App. A/B). These need no solver and check the physics against
    literature magnitudes (PCM ~66 ns, SOT-MRAM ~4 ns at a 512-row subarray).
  * Golden-objective solves -- guard the wired-up model end to end. The golden
    BW numbers are current phase-2 optima; if a later change moves them
    unintentionally, a test breaks.

Run from the repo (solver tests need the bundled Gurobi ASL driver on this node):
    opt/.venv/bin/python -m pytest opt/tests -q
"""
import sys
from pathlib import Path

import pytest
import pyomo.environ as pyo
import yaml

OPT = Path(__file__).resolve().parents[1]        # .../3d_memory/opt
sys.path.insert(0, str(OPT))

from run_bw_max import (merge, apply_fixes, make_specs, make_solver,  # noqa: E402
                        DEFAULT_GUROBI, BUILD, SOLVED)
from model import build_model, develop_coeffs                        # noqa: E402
from pyomo.common.tempfiles import TempfileManager                  # noqa: E402

# config name -> golden raw BW [bit/ns] (objective value of `var BW`).
GOLDEN_BW = {
    "reram_16nm":   8.870453e8,   # current sense
    "pcram_16nm":   3.745733e8,   # voltage sense, slow
    "sttmram_16nm": 6.230094e7,   # current sense, high latch floor
}
REL_TOL = 2e-3


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
    """Voltage f_margin = 2x current; tighter settling costs more time."""
    defaults, configs = cfg
    _, cur, _ = _spec("reram_16nm", defaults, configs)
    _, volt, _ = _spec("pcram_16nm", defaults, configs)
    f_cur, _, _ = develop_coeffs(cur)
    f_volt, _, _ = develop_coeffs(volt)
    assert f_volt == pytest.approx(2 * f_cur, rel=1e-9)           # B.8 vs A.12 factor

    _, base, _ = _spec("gaincell_100Mb", defaults, configs)
    _, tight, _ = _spec("gaincell_tight_settle", defaults, configs)
    assert develop_coeffs(tight)[0] > develop_coeffs(base)[0]     # 99.9% > 99% settle


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
@pytest.mark.parametrize("name,golden", GOLDEN_BW.items())
def test_objective_matches_golden(name, golden, cfg):
    defaults, configs = cfg
    _, bw = _solve_bw(name, defaults, configs)
    assert bw == pytest.approx(golden, rel=REL_TOL), f"{name}: BW={bw:.6g} vs golden {golden:.6g}"


@pytest.mark.skipif(not DEFAULT_GUROBI.exists(),
                    reason="bundled Gurobi ASL driver not present")
def test_aux_var_equalities_feasible(cfg):
    """Every def_* product-variable equality must hold at the solution."""
    defaults, configs = cfg
    m, _ = _solve_bw("reram_16nm", defaults, configs)
    for c in m.component_objects(pyo.Constraint):
        if c.local_name.startswith("def_"):
            resid = abs(pyo.value(c.body) - pyo.value(c.lower))
            assert resid < 1e-4, f"{c.local_name} residual {resid:.3e}"
