"""I18-1: gated correction.  The correction arm's held-out row is used for
the held-out devices of an outer fold iff the fold's training-only gate
statistic S3_f (I7-1 definition, computed with the gated arm's recipe) is
negative; otherwise the C0_refit row.

Pre-registration: ``out/research/explore/I18_1/prereg.md`` / ``prereg.json``.

Pure helpers (no training, no simulation): gate decision, gated rows,
dataset-level reference gate, per-fold summaries, the pre-registered rule
and validity table.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

THRESHOLD = 0.0                 # apply the correction iff S3_f < THRESHOLD (strict)
S3_STEPS = 500                  # I7-1 S3_HYPER (seed 0, 500 steps)
S3_SEED = 0
V2_TOL = 1e-9                   # V2: |S3 (I18-1 code, v1 recipe) - stored I7-1 S3|
SKY = "sky130_77K"
EXPECTED_ROWS = {"sky130_77K": 18, "nist_4K": 53, "nist_rt_card_4K": 87, "ihp_300K": 18, "ihp_233K": 18,
                 "ihp_343K": 18, "ihp_398K": 18, "tudelft_4K": 45, "tudelft_295K": 45}
PRIMARY = tuple(EXPECTED_ROWS)
V2_DATASETS = ("sky130_77K", "ihp_300K", "tudelft_4K", "tudelft_295K")


def apply_gate(s3: float, threshold: float = THRESHOLD) -> bool:
    """True (use the correction) iff S3_f is finite and < threshold."""
    return bool(s3 is not None and np.isfinite(s3) and float(s3) < threshold)


def fold_s3(log_ratios: Sequence[float]) -> float:
    """S3_f from the inner held-out log ratios of the fold (nanmean, I7-1
    ``run_s3``); NaN when no value is finite."""
    v = np.asarray(list(log_ratios), dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def gated_rows(corr_rows: Mapping[str, float], c0_rows: Mapping[str, float], fold_of: Callable[[str], str],
               s3_by_fold: Mapping[str, float]) -> tuple[dict, dict]:
    """Gated per-device rows and the per-fold decisions.

    ``corr_rows`` / ``c0_rows``: {tag: RRMS}; the gated row of ``tag`` is
    corr_rows[tag] if apply_gate(s3_by_fold[fold_of(tag)]) else
    c0_rows[tag].  Tags: those present in both row sets."""
    tags = sorted(set(corr_rows) & set(c0_rows))
    folds = sorted({fold_of(t) for t in tags})
    missing = [f for f in folds if f not in s3_by_fold]
    if missing:
        raise KeyError(f"no S3 for folds {missing[:5]}")
    dec = {f: apply_gate(s3_by_fold[f]) for f in folds}
    rows = {t: (corr_rows[t] if dec[fold_of(t)] else c0_rows[t]) for t in tags}
    return rows, dec


def dataset_gate(s3_by_fold: Mapping[str, float], n_held: Mapping[str, int]) -> dict:
    """I8-2 gate G (reference only): sign of the held-out-weighted mean of
    the finite S3_f values."""
    f = [k for k, v in s3_by_fold.items() if np.isfinite(v)]
    w = np.array([n_held[k] for k in f], float)
    v = np.array([s3_by_fold[k] for k in f], float)
    s = float(np.sum(w * v) / np.sum(w)) if len(f) else float("nan")
    return {"S3_D": s, "apply": apply_gate(s)}


def fold_y(corr_rows: Mapping[str, float], c0_rows: Mapping[str, float], fold_of: Callable[[str], str]) -> dict:
    """Per fold: mean over held-out devices of ln(corr / C0_refit) (finite pairs)."""
    acc: dict = {}
    for t in sorted(set(corr_rows) & set(c0_rows)):
        a, b = corr_rows[t], c0_rows[t]
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            acc.setdefault(fold_of(t), []).append(math.log(a / b))
    return {f: float(np.mean(v)) for f, v in acc.items()}


def n_finite_pairs(a: Mapping[str, float], b: Mapping[str, float]) -> int:
    return int(sum(1 for t in set(a) & set(b) if np.isfinite(a[t]) and np.isfinite(b[t]) and a[t] > 0 and b[t] > 0))


# ---------------------------------------------------------------------------
# rule and validity (pure, pre-registered)
# ---------------------------------------------------------------------------
def decide_rule(gmr: Mapping[str, dict], sky: str = SKY) -> dict:
    """C1: CI lower <= 1 on every dataset in ``gmr``; C2: CI upper < 1 on
    ``sky``.  Outcome label as in prereg.json."""
    worse = [d for d, g in gmr.items() if not (np.isfinite(g.get("ci_low", np.nan)) and g["ci_low"] <= 1.0)]
    c1 = not worse
    hs = gmr[sky].get("ci_high", np.nan)
    c2 = bool(np.isfinite(hs) and hs < 1.0)
    if not c1:
        out = f"not supported (worse than C0_refit on {', '.join(worse)})"
    elif not c2:
        out = "not supported (no gain on sky130)"
    else:
        out = "supported"
    return {"C1_never_worse": c1, "C1_fail_datasets": worse, "C2_sky130_gain": c2, "outcome": out}


def decide_validity(v: dict) -> dict:
    """V1 rows per dataset and g v2 rows equal stored selected rows; V2 S3
    code reproduces stored I7-1 values; V3 S3_v2 finite on every fold of
    the g v2 datasets; V4 unit tests."""
    fails = []
    bad = [d for d, n in EXPECTED_ROWS.items() if v["v1_rows"].get(d) != n]
    if bad or not (v["v1_selected_max_diff"] <= 0.0):
        fails.append(f"V1 rows {bad}" if bad else "V1 g v2 rows differ from stored selected rows")
    if not (v["v2_max_diff"] <= V2_TOL and v["v2_n"] == 2):
        fails.append("V2 S3 code reproduces stored I7-1 S3")
    if not v["v3_complete"]:
        fails.append("V3 S3_v2 finite on every fold")
    if not v["v4_unit_tests_pass"]:
        fails.append("V4 unit tests")
    return {"valid": not fails, "fails": fails}
