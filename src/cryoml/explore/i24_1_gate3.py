"""I24-1: three-way dataset-level gate {C0_refit, selected g v2, B1} and
selected g v2 on top of B1 (IHP SG13G2).

Pre-registration: ``out/research/explore/I24_1/prereg.md`` (``prereg.json``).

Pure helpers (no simulation, no training): leave-one-group-out B1 screen
selection, the S_B1 statistic, the three-way choice rule and its sensitivity
variants, the leave-group-out per-device gate, stratified pooled bootstrap,
and the Part A / Part B outcome rules.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

from .i22_1_baselines import screen_select

ARMS = ("C0_refit", "g", "B1")
OLS_A = -0.021454                  # I19-1 Part C OLS (prereg; rounded as registered)
OLS_B = 1.217286
THRESHOLD = 0.0
MAX_NONFINITE = 0.10
N_BOOT = 10000
SEED = 0
VA1_TOL = 5e-5
VA3_TOL = 1e-12
VB1_TOL = 1e-12
VB7_TOL = 1e-9
S_G_FREEZE = {"sky130": -0.1777, "nist4": -0.1172, "nistrt": -0.0730, "ihp233": 0.1866, "ihp300": 0.3114,
              "ihp343": 0.3669, "ihp398": 0.3667, "tud4": 0.0055, "tud295": 0.2537}


# ---------------------------------------------------------------------------
# B1 statistic
# ---------------------------------------------------------------------------
def logo_selection(per_device: Mapping[str, Mapping[str, dict]], train_tags: Sequence[str]) -> list[str]:
    """Edge-screen selection over the stored per-device screen entries of the
    inner-training devices only."""
    sub = {t: per_device[t] for t in train_tags}
    return list(screen_select(sub)["selected"])


def s_b1(log_ratios: Mapping[str, float], max_nonfinite: float = MAX_NONFINITE) -> dict:
    """S_B1 = mean of the finite per-record log ratios; invalid (NaN) when
    more than ``max_nonfinite`` of the records are non-finite or none is
    finite."""
    v = np.asarray(list(log_ratios.values()), float)
    n = len(v)
    nf = int(np.sum(~np.isfinite(v)))
    ok = n > 0 and nf < n and nf <= max_nonfinite * n
    return {"S_B1": float(np.mean(v[np.isfinite(v)])) if ok else float("nan"), "n": n, "n_nonfinite": nf,
            "valid": bool(ok)}


def log_ratio(b1: float, c0: float) -> float:
    if not (np.isfinite(b1) and np.isfinite(c0) and b1 > 0 and c0 > 0):
        return float("nan")
    return float(math.log(b1 / c0))


# ---------------------------------------------------------------------------
# choice rules
# ---------------------------------------------------------------------------
def yhat_g(s_g: float, a: float = OLS_A, b: float = OLS_B) -> float:
    return float(a + b * s_g)


def _fin(x) -> bool:
    return x is not None and np.isfinite(x)


def choose(s_g: float, s_b1: float | None, threshold: float = THRESHOLD) -> dict:
    """Primary three-way rule.  ``s_b1`` None / NaN: B1 not buildable, the
    two-way frozen rule applies."""
    g_c = _fin(s_g) and s_g < threshold
    b_c = _fin(s_b1) and s_b1 < threshold
    if not _fin(s_b1):
        return {"arm": "g" if g_c else "C0_refit", "reason": "B1 not buildable: two-way frozen rule",
                "candidates": ["g"] if g_c else []}
    cands = [a for a, c in (("g", g_c), ("B1", b_c)) if c]
    if not cands:
        return {"arm": "C0_refit", "reason": "no candidate", "candidates": []}
    if len(cands) == 1:
        return {"arm": cands[0], "reason": "one candidate", "candidates": cands}
    yg = yhat_g(s_g)
    arm = "g" if yg < s_b1 else "B1"
    return {"arm": arm, "reason": f"both candidates: yhat_g {yg:+.4f} vs S_B1 {s_b1:+.4f}", "candidates": cands,
            "yhat_g": yg}


def choose_two_way(s_g: float, threshold: float = THRESHOLD) -> str:
    return "g" if _fin(s_g) and s_g < threshold else "C0_refit"


def choose_raw(s_g: float, s_b1: float | None) -> str:
    """S1: smallest of {C0_refit: 0, g: S_g, B1: S_B1}; strict; ties to
    C0_refit, then B1."""
    best, arm = 0.0, "C0_refit"
    if _fin(s_b1) and s_b1 < best:
        best, arm = s_b1, "B1"
    if _fin(s_g) and s_g < best:
        best, arm = s_g, "g"
    return arm


def chosen_rows(rows: Mapping[str, Mapping[str, float]], arm: str, tags: Sequence[str]) -> dict:
    return {t: float(rows[arm][t]) for t in tags}


def per_device_gate(tags: Sequence[str], group_of: Callable[[str], str], lr_g: Mapping[str, float],
                    lr_b1: Mapping[str, float]) -> dict:
    """S3: per record d, both statistics without the records of d's group,
    then the primary rule. ``lr_g`` / ``lr_b1``: per-record log ratios."""
    out = {}
    for d in tags:
        k = group_of(d)
        rest = [t for t in tags if group_of(t) != k]
        sg = float(np.nanmean([lr_g[t] for t in rest])) if rest else float("nan")
        vb = {t: lr_b1[t] for t in rest}
        sb = s_b1(vb)["S_B1"] if rest else float("nan")
        c = choose(sg, sb)
        out[d] = {"arm": c["arm"], "S_g": sg, "S_B1": sb}
    return out


# ---------------------------------------------------------------------------
# pooled bootstrap
# ---------------------------------------------------------------------------
def _ds_log_ratios(rows: Mapping[str, float], ctrl: Mapping[str, float], tags: Sequence[str]) -> np.ndarray:
    a = np.array([rows[t] for t in tags], float)
    b = np.array([ctrl[t] for t in tags], float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out = np.full(len(tags), np.nan)
    out[ok] = np.log(a[ok] / b[ok])
    return out


def pooled_bootstrap(data: Mapping[str, Mapping], series: Sequence[str], n_boot: int = N_BOOT,
                     seed: int = SEED) -> dict:
    """Stratified bootstrap of exp(mean over datasets of ln GMR(series, C0_refit)).

    ``data[ds] = {"tags": [...], "clusters": {tag: c} | None, "rows": {series: {tag: rrms}},
    "control": {tag: rrms}}``.  Per replicate every dataset is resampled
    independently (devices, or clusters when given), with the same indices for
    every series.  Per-dataset GMR uses the finite pairs of the resample.
    Returns per series: point, ci_low, ci_high, plus the bootstrap ln samples
    (key ``_boot``) for paired ratios."""
    rng = np.random.default_rng(seed)
    dss = list(data)
    lr = {ds: {s: _ds_log_ratios(data[ds]["rows"][s], data[ds]["control"], data[ds]["tags"]) for s in series}
          for ds in dss}
    point = {s: float(np.exp(np.mean([np.nanmean(lr[ds][s]) for ds in dss]))) for s in series}
    boot = {s: np.zeros(n_boot) for s in series}
    for ds in dss:
        tags = data[ds]["tags"]
        cl = data[ds].get("clusters")
        n = len(tags)
        if cl is None:
            idx = rng.integers(0, n, size=(n_boot, n))
            for s in series:
                v = lr[ds][s][idx]
                boot[s] += np.nanmean(v, axis=1)
        else:
            names = sorted({cl[t] for t in tags})
            members = [np.array([i for i, t in enumerate(tags) if cl[t] == c]) for c in names]
            pick = rng.integers(0, len(names), size=(n_boot, len(names)))
            for s in series:
                x = lr[ds][s]
                sums = np.array([np.nansum(x[m]) for m in members])
                cnt = np.array([np.sum(np.isfinite(x[m])) for m in members], float)
                boot[s] += sums[pick].sum(axis=1) / cnt[pick].sum(axis=1)
    out = {}
    for s in series:
        b = boot[s] / len(dss)
        lo, hi = np.percentile(b, [2.5, 97.5])
        out[s] = {"pooled": point[s], "ci_low": float(np.exp(lo)), "ci_high": float(np.exp(hi)), "_boot": b}
    return out


def paired_pooled(boot: Mapping[str, dict], a: str, b: str) -> dict:
    """Pooled ratio a / b with the paired bootstrap CI (same replicates)."""
    d = boot[a]["_boot"] - boot[b]["_boot"]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"ratio": float(boot[a]["pooled"] / boot[b]["pooled"]), "ci_low": float(np.exp(lo)),
            "ci_high": float(np.exp(hi))}


def strip_boot(b: Mapping[str, dict]) -> dict:
    return {s: {k: v for k, v in x.items() if k != "_boot"} for s, x in b.items()}


# ---------------------------------------------------------------------------
# outcome rules
# ---------------------------------------------------------------------------
def part_a_outcome(gmr3: Mapping[str, float], pooled3: float, pooled_b1: float, pooled_g: float,
                   ratio_3_2: Mapping, valid: bool) -> dict:
    if not valid:
        return {"label": "invalid", "R1": None, "R2": None, "R3": None}
    worse = sorted(d for d, r in gmr3.items() if not (np.isfinite(r) and r <= 1.0))
    r1 = not worse
    r2 = bool(np.isfinite(ratio_3_2.get("ci_high", np.nan)) and ratio_3_2["ci_high"] < 1.0)
    r3 = bool(pooled3 <= pooled_b1 and pooled3 <= pooled_g)
    if r1 and r2 and r3:
        lab = "three-way gate supported (retrospective)"
    elif r2:
        lab = "partially supported (retrospective)"
    else:
        lab = "not supported (retrospective)"
    return {"label": lab, "R1": r1, "R1_fail": worse, "R2": r2, "R3": r3}


def label_ci(cmp: Mapping) -> str:
    lo, hi = cmp.get("ci_low", np.nan), cmp.get("ci_high", np.nan)
    if np.isfinite(hi) and hi < 1.0:
        return "gain"
    if np.isfinite(lo) and lo > 1.0:
        return "worse"
    return "inconclusive"


def part_b_outcome(per_t: Mapping[str, Mapping], valid: bool) -> dict:
    """``per_t[T]`` = paired_ratio(B1+g v2, B1)."""
    if not valid:
        return {"label": "invalid"}
    labs = {t: label_ci(c) for t, c in per_t.items()}
    if len(labs) == 4 and all(v == "gain" for v in labs.values()):
        lab = "g v2 adds to B1 on IHP"
    elif len(labs) == 4 and all(v == "worse" for v in labs.values()):
        lab = "g v2 worse than B1 on IHP"
    else:
        lab = "g v2 gain over B1 not shown"
    return {"label": lab, "per_temperature": labs}
