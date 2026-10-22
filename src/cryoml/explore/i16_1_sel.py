"""I16-1: deployment test (Part B, I15-1 rules) of the I15-2 selected
bias-only correction: symmetric g, penalty lambda selected per outer fold by
nested cross-validation on the training devices; deployed full-data model
lambda* = 1 V (``I15_2/lodo_sym_e1_l1`` ``full|sym_e1_l1``).

Pre-registration: ``out/research/explore/I16_1/prereg.md`` / ``prereg.json``.

Pure helpers: sources of the selected held-out pairs, the sign-change counts
of the restriction items, the validity rule and the outcome table.  The
decks, replica, module text, references and Part B checks are the I14-1 /
I15-1 ones (``i14_1_chan``, ``i15_1_serp``, ``i12_1_symm.decide_deploy``).
"""

from __future__ import annotations

import numpy as np

# lambda* (V) -> (source directory relative to out/research/explore, correction key prefix,
#                 held-out cards key or None for a flat {tag: card} file)
SOURCES = {
    0.0: ("I15_2/lodo_sym_e1_l0", "sym_e1_l0"),
    1.0: ("I15_2/lodo_sym_e1_l1", "sym_e1_l1"),
    2.0: ("I15_2/lodo_sym_e1_l2", "sym_e1_l2"),
    5.0: ("I15_2/lodo_sym_e1_l5", "sym_e1_l5"),
    10.0: ("I13_1/lodo", "M_bias_v2"),        # stored g v2 = sym_e1_l10
}
DEPLOYED = {"dir": "I15_2/lodo_sym_e1_l1", "key": "full|sym_e1_l1", "variant": "sym_e1_l1",
            "inputs": "sym_v2", "eps": 0.01, "lambda": 1.0}


def source_for(lam: float) -> tuple[str, str]:
    """(directory, key prefix) of the 5-seed LODO variant sym_e1_l{lam}."""
    lam = float(lam)
    if lam not in SOURCES:
        raise KeyError(f"no I15-2 variant for lambda {lam}")
    return SOURCES[lam]


def fold_key(lam: float, tag: str) -> str:
    return f"{source_for(lam)[1]}|{tag}"


# ---------------------------------------------------------------------------
# sign-change counts (restriction items)
# ---------------------------------------------------------------------------
def neg_count(x, y, mask) -> int:
    """Number of points of the central-difference derivative dy/dx < 0
    inside ``mask`` (x sorted by |x|, as the dense sweeps)."""
    from .i11_1_deploy import central_diff
    d = central_diff(np.asarray(x, float), np.asarray(y, float))
    m = np.asarray(mask, bool) & np.isfinite(d)
    return int(np.sum(d[m] < 0))


def neg_range(x, y, mask) -> list[float] | None:
    """[min, max] of |x| at the points counted by ``neg_count`` (None if none)."""
    from .i11_1_deploy import central_diff
    x = np.asarray(x, float)
    d = central_diff(x, np.asarray(y, float))
    m = np.asarray(mask, bool) & np.isfinite(d) & (d < 0)
    if not m.any():
        return None
    ax = np.abs(x[m])
    return [float(ax.min()), float(ax.max())]


# ---------------------------------------------------------------------------
# decision (pure, pre-registered)
# ---------------------------------------------------------------------------
THR_VALID = {"v1_rrms": 1e-9}


def decide_validity(v: dict) -> dict:
    """V1 card-only deck x exp(fold g) reproduces the I15-2 selected rows
    (max |diff| <= 1e-9, 18 devices); V2 unit tests pass; V3 18
    element-deck accuracy rows and 0 failed M2 refits; V4 the deployed g is
    sym_v2 / EPS 0.01 / lambda 1 and the I15-2 full-data lambda* is 1."""
    fails = []
    if not (v["v1_max_rrms_diff"] <= THR_VALID["v1_rrms"]) or v.get("v1_n_rows", 18) != 18:
        fails.append("V1 card-only deck reproduces the I15-2 selected rows")
    if not v["v2_unit_tests_pass"]:
        fails.append("V2 unit tests")
    if v["v3_n_rows"] != 18 or v["v3_m2_failed"] != 0:
        fails.append("V3 18 rows / 0 failed M2 refits")
    if not v["v4_deployed_ok"]:
        fails.append("V4 deployed g (sym_v2, EPS 0.01, lambda 1; lambda* full 1)")
    return {"valid": not fails, "fails": fails}


def decide_outcome(validity: dict, dep: dict) -> dict:
    """I16-1 outcome table: invalid / deployable / not deployable."""
    if not validity["valid"]:
        return {"outcome": "invalid", "action": "stop: fix and re-register"}
    if dep["deploy"] == "not deployable":
        return {"outcome": "not deployable", "action": "stop; diagnose the failed core checks"}
    return {"outcome": "deployable", "label": dep["deploy"], "restrictions": list(dep["restrictions"]),
            "action": "the I15-2 selected variant is the deployable add-on candidate; report restrictions"}
