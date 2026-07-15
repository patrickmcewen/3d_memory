"""Provenance-ledger regression: config.yaml stays consistent with its cites.

Guards that every calibrated value in config.yaml has a provenance entry, every
cited source id exists in sources.yaml, and every `derive` expression still
evaluates to the value it justifies (catches a number and its derivation drifting
apart). Pure sidecar check -- needs no solver. See opt/check_provenance.py.
"""
import sys
from pathlib import Path

OPT = Path(__file__).resolve().parents[1]        # .../3d_memory/opt
sys.path.insert(0, str(OPT))

from check_provenance import check                # noqa: E402


def test_provenance_consistent():
    summary = check()                            # raises AssertionError on any violation
    assert summary["entries"] > 0, f"no provenance entries loaded: {summary}"
