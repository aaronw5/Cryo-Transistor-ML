"""I2-1: a compact-model correction trained jointly with the training cards.

E1 trained the correction ``g`` on residuals of cards that were fit without
it.  Here the training cards and ``g`` alternate (``joint_rounds``):

* round 0: ``g_0`` given (the E1 fold correction) or trained on the
  starting records;
* round r = 1..K: every training card is refit with ``g_{r-1}`` inside the
  simulator call (``refit_map``), the refit cards are simulated once more to
  give new training records, and ``g_r`` is trained from scratch on them
  (``train_map``).

``refit_map`` and ``train_map`` take a list of tasks and return a list of
results in the same order, so the caller can run them serially (tests) or
on a process pool with rounds synchronized across folds (driver
``scripts/research/explore/i2_1_joint.py``).

``CorrectedProblem`` wraps a ``DeviceProblem`` / ``Pdk2Problem`` so that
``ngspice_fit.polish`` sees corrected currents; ``correlation_stats`` is the
simulated-arm identifiability diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from .e1_correction import HYPER, Correction, CurveRec, DeviceRec, score_device


# ---------------------------------------------------------------------------
# Correction inside the simulator call
# ---------------------------------------------------------------------------
def curve_key(curve) -> str:
    """Canonical ``kind@bias`` key of a problem curve (as E1 records)."""
    meta = getattr(curve, "meta", None) or {}
    fixed = meta.get("canonical_fixed", curve.fixed)
    return f"{curve.kind}@{abs(float(fixed)):g}"


class CorrectedProblem:
    """Delegates to ``base``; ``simulate`` returns ``I_sim * exp(g)``."""

    def __init__(self, base, corr: Correction, family: str, L_um: float,
                 W_um: float, vdd: float, source: str):
        self.base = base
        self.corr = corr
        self._shape = (base.tag, family, float(L_um), float(W_um), float(vdd), source)
        self._keys = [curve_key(c) for c in base.curves]

    def __getattr__(self, name):
        return getattr(self.base, name)

    def record(self, sims) -> DeviceRec:
        tag, family, L_um, W_um, vdd, source = self._shape
        rec = DeviceRec(tag, family, L_um, W_um, vdd, source)
        for key, c, s in zip(self._keys, self.base.curves, sims):
            rec.curves.append(CurveRec(key, c.kind, np.asarray(c.Vg, float),
                                       np.asarray(c.Vd, float), np.asarray(s, float),
                                       np.asarray(c.Id, float), 0, 0.0, False))
        return rec

    def simulate(self, local):
        sims = self.base.simulate(local)
        if self.corr.arm == "C0":
            return sims
        return self.corr.apply(self.record(sims))


def with_sims(rec: DeviceRec, sims: Sequence[np.ndarray], **meta) -> DeviceRec:
    """Copy of ``rec`` with new simulated currents (same curve order)."""
    if len(sims) != len(rec.curves):
        raise ValueError(f"{rec.tag}: {len(sims)} sims for {len(rec.curves)} curves")
    curves = [replace(c, sim=np.asarray(s, dtype=np.float64))
              for c, s in zip(rec.curves, sims)]
    return replace(rec, curves=curves, meta={**rec.meta, **meta})


def corrected_score(corr: Correction, rec: DeviceRec) -> float:
    return float(score_device(rec, corr.apply(rec))["rrms"])


def mean_score(corr: Correction, recs: Sequence[DeviceRec]) -> float:
    vals = [corrected_score(corr, r) for r in recs]
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Alternating loop
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    name: str
    arm: str                                   # "M" | "T1" | ...
    train: list[DeviceRec]
    corr: Correction | None = None             # current correction (g_0 if given)
    corrs: list = field(default_factory=list)  # g_0 .. g_K
    cards: list = field(default_factory=list)  # per round: {tag: params}
    history: list = field(default_factory=list)


def joint_rounds(folds: list[Fold], k_rounds: int,
                 refit_map: Callable[[list], list],
                 train_map: Callable[[list], list],
                 log: Callable[[str], None] | None = None) -> list[Fold]:
    """Run ``k_rounds`` alternating refit/retrain rounds on every fold.

    ``refit_map([(corr, rec), ...]) -> [rec', ...]``: refit the card of
    ``rec`` with ``corr`` in the loop and return the record of the refit
    card (``meta['params']`` holds the card).
    ``train_map([(arm, recs), ...]) -> [Correction, ...]``: train from
    scratch.  Folds whose ``corr`` is None get ``g_0`` from ``train_map``.
    Mutates and returns ``folds``."""
    need = [f for f in folds if f.corr is None]
    if need:
        for f, corr in zip(need, train_map([(f.arm, f.train) for f in need])):
            f.corr = corr
    for f in folds:
        f.corrs = [f.corr]
        f.cards = [{r.tag: r.meta.get("params") for r in f.train}]
        f.history = [{"round": 0, "train_rrms_corrected": mean_score(f.corr, f.train),
                      "train_rrms_uncorrected": mean_score(Correction("C0"), f.train)}]
    for r in range(1, k_rounds + 1):
        tasks = [(f.corr, rec) for f in folds for rec in f.train]
        out = refit_map(tasks)
        if len(out) != len(tasks):
            raise RuntimeError("refit_map returned a different number of records")
        pos = 0
        for f in folds:
            n = len(f.train)
            f.train = list(out[pos:pos + n])
            pos += n
        before = [mean_score(f.corr, f.train) for f in folds]
        new = train_map([(f.arm, f.train) for f in folds])
        for f, corr, b in zip(folds, new, before):
            f.corr = corr
            f.corrs.append(corr)
            f.cards.append({rec.tag: rec.meta.get("params") for rec in f.train})
            f.history.append({
                "round": r,
                "train_rrms_prev_corr_on_refit_cards": b,
                "train_rrms_corrected": mean_score(corr, f.train),
                "train_rrms_uncorrected": mean_score(Correction("C0"), f.train),
                "refit_n_sim": [rec.meta.get("refit_n_sim") for rec in f.train]})
        if log:
            log(f"round {r}: " + " ".join(
                f"{f.name}={f.history[-1]['train_rrms_corrected']:.4f}" for f in folds))
    return folds


def split_train_tasks(tasks, hyper):
    """One pool task per M seed (each seed trains independently from
    ``torch.manual_seed(seed)`` on the same standardized batch, so merging the
    single-seed nets reproduces ``fit_arm`` with all seeds); other arms whole."""
    expanded, owner = [], []
    for i, (arm, recs) in enumerate(tasks):
        if arm == "M":
            seeds = {**HYPER["M"], **hyper.get("M", {})}["seeds"]
            for seed in seeds:
                expanded.append((arm, recs, {**hyper, "M": {**hyper.get("M", {}),
                                                           "seeds": (seed,)}}))
                owner.append(i)
        else:
            expanded.append((arm, recs, hyper))
            owner.append(i)
    return expanded, owner


def merge_train_results(n_tasks, owner, results):
    merged = [None] * n_tasks
    for i, corr in zip(owner, results):
        if merged[i] is None:
            merged[i] = corr
        else:
            if corr.arm != "M" or not (np.array_equal(corr.mu, merged[i].mu)
                                       and np.array_equal(corr.sd, merged[i].sd)):
                raise RuntimeError("per-seed corrections do not share feature scaling")
            merged[i] = Correction("M", merged[i].mu, merged[i].sd,
                                   nets=merged[i].nets + corr.nets)
    return merged


# ---------------------------------------------------------------------------
# Identifiability diagnostic (simulated arm)
# ---------------------------------------------------------------------------
COMPONENTS = ("c_param", "c_rs", "c_heat", "c_gain")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def correlation_stats(rows: list[dict]) -> dict:
    """Pooled statistics of ``g`` against known misspecification components.

    ``rows``: one dict per held-out target with equal-length arrays ``g``,
    ``resid`` (ln|target / I_card|) and the ``COMPONENTS``; non-finite points
    must already be removed."""
    if not rows:
        return {}
    cat = {k: np.concatenate([r[k] for r in rows]) for k in ("g", "resid") + COMPONENTS}
    cat["c_sum"] = sum(cat[c] for c in COMPONENTS)
    out = {"n_points": int(len(cat["g"])), "n_targets": len(rows),
           "pearson": {k: _pearson(cat["g"], cat[k]) for k in COMPONENTS + ("c_sum", "resid")},
           "std": {k: float(np.std(cat[k])) for k in ("g", "resid", "c_sum") + COMPONENTS}}
    x = np.column_stack([np.ones(len(cat["g"]))] + [cat[c] for c in COMPONENTS])
    coef, *_ = np.linalg.lstsq(x, cat["g"], rcond=None)
    pred = x @ coef
    ss = float(np.sum((cat["g"] - cat["g"].mean()) ** 2))
    out["ols"] = {"intercept": float(coef[0]),
                  **{c: float(v) for c, v in zip(COMPONENTS, coef[1:])},
                  "r2": float(1.0 - np.sum((cat["g"] - pred) ** 2) / ss) if ss > 0 else float("nan")}
    per = [_pearson(r["g"], sum(r[c] for c in COMPONENTS)) for r in rows]
    per = [p for p in per if np.isfinite(p)]
    out["mean_per_target_pearson_c_sum"] = float(np.mean(per)) if per else float("nan")
    return out


def finite_mask(arrays: Sequence[np.ndarray]) -> np.ndarray:
    m = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def log_ratio(a: np.ndarray, b: np.ndarray, floor: float = 1e-13) -> np.ndarray:
    """ln|a| - ln|b|, NaN where either magnitude is below ``floor``."""
    a, b = np.abs(np.asarray(a, float)), np.abs(np.asarray(b, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(a) - np.log(b)
    out[(a < floor) | (b < floor) | ~np.isfinite(out)] = math.nan
    return out
