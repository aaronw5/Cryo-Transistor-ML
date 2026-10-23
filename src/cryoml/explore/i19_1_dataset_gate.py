"""I19-1: dataset-level gate.  One decision per dataset: the correction is
applied to every device of a dataset iff S3_D < 0, where S3_D is the I7-1 S3
statistic computed once with the training set = all DEV devices of the
dataset (inner leave-one-group-out over all DEV records); otherwise every
device gets C0_refit.

Pre-registration: ``out/research/explore/I19_1/prereg.md`` / ``prereg.json``.

Pure helpers (no training, no simulation): inner plan, S3_D, decision, gated
rows, Part R / Part C rules, overall outcome, validity.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

THRESHOLD = 0.0                 # apply iff S3_D < THRESHOLD (strict)
SENS_THRESHOLD = 0.02455        # reported only: zero crossing of the I7-1 OLS y = -0.0298 + 1.214 S3 (20 simulated)
S3_STEPS = 500                  # I7-1 S3 hyper (seed 0, 500 steps)
S3_SEED = 0
V2_TOL = 1e-9
C_MIN_CORRECT = 19              # Part C: sign accuracy >= 19 of 20
PRIMARY = ("sky130_77K", "nist_4K", "nist_rt_card_4K", "ihp_300K", "ihp_233K", "ihp_343K", "ihp_398K",
           "tudelft_4K", "tudelft_295K")
EXPECTED_ROWS = {"sky130_77K": 18, "nist_4K": 53, "nist_rt_card_4K": 87, "ihp_300K": 18, "ihp_233K": 18,
                 "ihp_343K": 18, "ihp_398K": 18, "tudelft_4K": 45, "tudelft_295K": 45}


def inner_plan(order: Sequence[str], group_of: Callable[[str], str],
               train_of: Callable[[str], Sequence[str]]) -> list[tuple[str, list, list]]:
    """(inner group k, inner-training tags, scored tags) for S3_D.

    ``order``: all DEV record tags of the dataset in load order;
    ``train_of(k)``: the training tags of the leave-one-group-out fold that
    holds out k (load order).  The scored tags of k are the records of group
    k.  Raises if a training set contains a record of k."""
    out = []
    for k in sorted({group_of(t) for t in order}):
        train = list(train_of(k))
        if any(group_of(t) == k for t in train):
            raise ValueError(f"training set of group {k} contains its own records")
        out.append((k, train, [t for t in order if group_of(t) == k]))
    return out


def s3_dataset(log_ratios: Sequence[float]) -> float:
    """S3_D = nanmean of the per-record log ratios; NaN when none is finite."""
    v = np.asarray(list(log_ratios), dtype=float)
    return float(np.nanmean(v)) if v.size and np.isfinite(v).any() else float("nan")


def apply_gate(s3: float, threshold: float = THRESHOLD) -> bool:
    """True (apply the correction on the whole dataset) iff S3_D finite and < threshold."""
    return bool(s3 is not None and np.isfinite(s3) and float(s3) < threshold)


def gated_rows(corr: Mapping[str, float], c0: Mapping[str, float], apply: bool) -> dict:
    """Dataset-level gated rows over the tags present in both row sets."""
    return {t: (corr[t] if apply else c0[t]) for t in sorted(set(corr) & set(c0))}


def pooled(ratios: Sequence[float]) -> float:
    """exp(mean ln ratio) over datasets (equal weight)."""
    v = [math.log(r) for r in ratios]
    return float(math.exp(np.mean(v))) if v else float("nan")


def part_r_rule(gmr_gated: Mapping[str, float], gmr_corr: Mapping[str, float]) -> dict:
    """R1: GMR(gated_D, C0_refit) <= 1 on every dataset; R2: pooled gated_D
    below pooled correction always."""
    worse = sorted(d for d, r in gmr_gated.items() if not (np.isfinite(r) and r <= 1.0))
    pg = pooled([gmr_gated[d] for d in gmr_gated])
    pc = pooled([gmr_corr[d] for d in gmr_gated])
    r1, r2 = not worse, bool(pg < pc)
    fails = ([f"R1 (GMR > 1 on {', '.join(worse)})"] if not r1 else []) + (["R2"] if not r2 else [])
    return {"R1_never_worse": r1, "R1_fail_datasets": worse, "R2_pooled_below_correction": r2,
            "pooled_gated_D": pg, "pooled_correction_always": pc,
            "supported": r1 and r2,
            "outcome": "supported (retrospective)" if r1 and r2
            else f"not supported (retrospective; {'; '.join(fails)})"}


def sign_correct(s3: float, y: float, threshold: float = THRESHOLD) -> bool:
    """Decision correct iff (S3_D < threshold) == (y < 0)."""
    return apply_gate(s3, threshold) == bool(np.isfinite(y) and y < 0)


def part_c_rule(s3: Mapping[str, float], y: Mapping[str, float], min_correct: int = C_MIN_CORRECT) -> dict:
    keys = sorted(set(s3) & set(y))
    hits = {k: sign_correct(s3[k], y[k]) for k in keys}
    n = int(sum(hits.values()))
    return {"n": len(keys), "correct": n, "required": min_correct, "holds": n >= min_correct,
            "constant_apply_correct": int(sum(1 for k in keys if y[k] < 0)),
            "wrong": sorted(k for k, v in hits.items() if not v)}


def overall(valid: bool, r_ok: bool, c_ok: bool) -> str:
    if not valid:
        return "invalid"
    if r_ok and c_ok:
        return "supported"
    if r_ok or c_ok:
        return "partially supported"
    return "not supported"


def decide_validity(v: dict) -> dict:
    """V1 rows; V2 S3 code reproduces stored fold values; V3 S3_D complete;
    V4 unit tests; V5 TEST closed."""
    fails = []
    bad = [d for d, n in EXPECTED_ROWS.items() if v["v1_rows"].get(d) != n]
    if bad or not (v["v1_rows_max_diff"] <= 0.0):
        fails.append(f"V1 rows {bad}" if bad else "V1 rows differ from I18-1 rows")
    if not (v["v2_max_diff"] <= V2_TOL and v["v2_n"] == 3):
        fails.append("V2 S3 code reproduces stored fold values")
    if not v["v3_complete"]:
        fails.append("V3 S3_D complete")
    if not v["v4_unit_tests_pass"]:
        fails.append("V4 unit tests")
    if not v["v5_test_closed"]:
        fails.append("V5 TEST closed")
    return {"valid": not fails, "fails": fails}
