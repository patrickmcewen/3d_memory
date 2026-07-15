#!/usr/bin/env python3
"""Provenance checker for the 3d_memory config.

Cross-checks ``opt/config.yaml`` against two sidecar ledgers -- ``opt/sources.yaml``
(the canonical bibliography) and ``opt/provenance.yaml`` (per-value src/loc/derive)
-- and fails loud on any of:

  * a provenance ``src`` id that is not defined in sources.yaml (dangling cite),
  * a provenance path that no longer resolves in config.yaml (a value was
    renamed or removed but its provenance was left behind),
  * a ``derive`` arithmetic expression that does not evaluate to the config
    value it justifies (the number and its derivation have drifted apart),
  * a calibrated config value (``problem``/``technology`` block, in ``defaults``
    or any named config) with NO provenance entry and not on the exempt list.

Equations are cited the same way: ``provenance.yaml`` has an ``equations`` section
(id -> src/loc/form), and ``model.py`` marks each implemented equation with a
``[eq:<id>]`` comment. The checker cross-validates both directions -- every tag
resolves to an entry, and every entry is referenced by at least one tag (no orphan
derivations) -- plus each equation's ``src`` exists in sources.yaml.

The runtime model (run_bw_max.py / model.py) never imports this -- provenance is
a pure sidecar. Run it standalone or via tests/test_provenance.py:

    opt/.venv/bin/python opt/check_provenance.py            # check, print a report
    opt/.venv/bin/python opt/check_provenance.py --by-source upton2024   # what cites X

Path grammar for provenance/exempt keys:  ``<scope>.<block>.<param>`` where
``<scope>`` is ``defaults`` or a named config, e.g. ``sram_16nm.technology.v_cell``.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
# Source files scanned for [eq:<id>] equation tags.
CODE_PATHS = (HERE / "model.py",)
# An equation reference in a code comment, e.g. "# tau_C per row  [eq:bl_settle_current]".
EQ_TAG = re.compile(r"\[eq:\s*([a-z0-9_]+)\s*\]")
# Only these blocks hold physically-calibrated numbers that need a citation;
# bounds/solver/fix are structural solver knobs, not literature-derived.
REQUIRED_BLOCKS = ("problem", "technology")
# `derive` expressions may quote rounded literature anchors (e.g. 146*2.56e-4
# stored as 0.0374), so allow 1% slack -- still an order of magnitude tighter
# than any real transcription drift (a wrong feature-size is >=10% off).
DERIVE_RTOL = 1e-2
# math-only namespace for evaluating derive expressions (no builtins).
_DERIVE_NS = {k: getattr(math, k) for k in ("log", "log2", "log10", "exp",
                                            "sqrt", "pi", "e")}


def _load(config_path: Path, sources_path: Path, prov_path: Path):
    doc = yaml.safe_load(config_path.read_text())
    sources = yaml.safe_load(sources_path.read_text())
    prov_doc = yaml.safe_load(prov_path.read_text())
    entries = prov_doc.get("provenance") or {}
    equations = prov_doc.get("equations") or {}
    exempt = set(prov_doc.get("exempt") or [])
    return doc, sources, entries, equations, exempt


def _bad_srcs(label: str, src, sources: dict):
    """Yield error strings for a missing/dangling ``src`` list."""
    if not src:
        yield f"{label}: no 'src' listed"
    for sid in (src or []):
        if sid not in sources:
            yield f"{label}: src '{sid}' not in sources.yaml"


def _scan_eq_refs(code_paths):
    """Map equation id -> list of files that reference it via a [eq:<id>] tag."""
    refs = {}
    for path in code_paths:
        for eid in EQ_TAG.findall(path.read_text()):
            refs.setdefault(eid, []).append(path.name)
    return refs


def _resolve(doc: dict, path: str):
    """Value at ``<scope>.<block>.<param>``, or None if the path is dangling."""
    scope, block, param = path.split(".", 2)
    root = doc["defaults"] if scope == "defaults" else (doc["configs"].get(scope) or {})
    return (root.get(block) or {}).get(param, None)


def _is_number(val) -> bool:
    """True if a YAML leaf is numeric (PyYAML reads '1.6e9' as a str)."""
    if isinstance(val, bool):
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _covered_leaves(doc: dict):
    """Every calibrated leaf that needs provenance: ``defaults`` plus every
    named config's explicit overrides, restricted to REQUIRED_BLOCKS. Yields
    (path, value). 'description' is prose, not a calibrated value -> skipped."""
    scopes = [("defaults", doc["defaults"])]
    scopes += [(name, cfg) for name, cfg in doc["configs"].items()]
    for scope, blocks in scopes:
        for block in REQUIRED_BLOCKS:
            for param, val in (blocks.get(block) or {}).items():
                yield f"{scope}.{block}.{param}", val


def check(config_path=HERE / "config.yaml",
          sources_path=HERE / "sources.yaml",
          prov_path=HERE / "provenance.yaml",
          code_paths=CODE_PATHS) -> dict:
    """Run every provenance check. Raises AssertionError listing all violations
    found; returns a summary dict on success."""
    doc, sources, entries, equations, exempt = _load(config_path, sources_path, prov_path)
    errors = []

    # 1. Every value-provenance entry is well-formed and consistent with config.yaml.
    for path, e in entries.items():
        errors += list(_bad_srcs(path, e.get("src"), sources))
        val = _resolve(doc, path)
        if val is None:
            errors.append(f"{path}: path does not resolve in config.yaml (renamed/removed?)")
            continue
        if "derive" in e:
            assert _is_number(val), f"{path}: has 'derive' but config value {val!r} is non-numeric"
            got = eval(e["derive"], {"__builtins__": {}}, _DERIVE_NS)  # noqa: S307 (math-only ns)
            ref = float(val)
            if abs(got - ref) > DERIVE_RTOL * max(abs(ref), 1e-30):
                errors.append(f"{path}: derive '{e['derive']}' = {got:.6g} "
                              f"!= config {ref:.6g} (rtol {DERIVE_RTOL})")

    # 2. Every calibrated config value is covered (has provenance or is exempt).
    for path, val in _covered_leaves(doc):
        if path in entries or path in exempt:
            continue
        errors.append(f"{path}: value {val!r} has no provenance entry (add to "
                      f"provenance.yaml, or list under 'exempt' if intentional)")

    # 3. Equations: sources resolve, and the [eq:<id>] tags in code match the
    #    ledger both ways (no dangling tag, no orphan entry).
    for eid, e in equations.items():
        errors += list(_bad_srcs(f"equation '{eid}'", e.get("src"), sources))
    eq_refs = _scan_eq_refs(code_paths)
    for eid, files in eq_refs.items():
        if eid not in equations:
            errors.append(f"[eq:{eid}] in {files} has no 'equations' entry in provenance.yaml")
    for eid in equations:
        if eid not in eq_refs:
            errors.append(f"equation '{eid}' is never referenced by a [eq:{eid}] tag in {[p.name for p in code_paths]}")

    assert not errors, "provenance check failed:\n  - " + "\n  - ".join(sorted(errors))
    return {"entries": len(entries), "sources": len(sources), "exempt": len(exempt),
            "covered": sum(1 for _ in _covered_leaves(doc)),
            "equations": len(equations), "eq_refs": len(eq_refs)}


def _report_by_source(sources: dict, entries: dict, equations: dict, sid: str) -> None:
    assert sid in sources, f"unknown source '{sid}' (known: {sorted(sources)})"
    print(f"# {sid}: {sources[sid].get('cite', '')}")
    vals = [p for p, e in entries.items() if sid in (e.get("src") or [])]
    print(f"  values ({len(vals)}):")
    for p in sorted(vals):
        print(f"    {p:<43} {entries[p].get('loc', '')}")
    eqs = [k for k, e in equations.items() if sid in (e.get("src") or [])]
    print(f"  equations ({len(eqs)}):")
    for k in sorted(eqs):
        print(f"    {k:<43} {equations[k].get('loc', '')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--sources", type=Path, default=HERE / "sources.yaml")
    ap.add_argument("--provenance", type=Path, default=HERE / "provenance.yaml")
    ap.add_argument("--by-source", metavar="ID",
                    help="list every config value that cites source ID, then exit")
    args = ap.parse_args()

    if args.by_source:
        sources = yaml.safe_load(args.sources.read_text())
        prov_doc = yaml.safe_load(args.provenance.read_text())
        _report_by_source(sources, prov_doc.get("provenance") or {},
                          prov_doc.get("equations") or {}, args.by_source)
        return 0

    summary = check(args.config, args.sources, args.provenance)
    print(f"provenance OK: {summary['covered']} calibrated values covered by "
          f"{summary['entries']} entries across {summary['sources']} sources "
          f"({summary['exempt']} exempt); {summary['equations']} equations, all "
          f"{summary['eq_refs']} model.py [eq:] tags resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
