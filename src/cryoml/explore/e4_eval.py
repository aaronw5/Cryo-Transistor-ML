"""E4 evaluation: held-out-curve RRMS, interval coverage and learned predictors.

Inputs are the per-device records of ``e4_heldout`` (members, Laplace draws,
their simulated curves).  Every quantity is computed on the held-out curves'
trimmed points (``metrics.curve_trim``), so held-out RRMS is the paper
per-curve RRMS averaged over the held-out curves of the device.  A device
whose held-out prediction has a non-finite point is NaN for that method
(strict convention); aggregates report the number of valid devices.

Methods (point predictor / interval):

* ``single``          lowest-train-objective card / none
* ``ens``             pointwise median of near-optimal members / Gaussian
                      mixture of members, each with sd = member train sigma x
                      mean |member prediction| of the curve; ``ens_env`` is
                      the member min-max envelope
* ``laplace``         mean of Laplace draws / mixture of draws with the single
                      card's train sigma (``laplace_param``: empirical draw
                      quantiles, no noise term)
* ``conformal``       single / split conformal over the other DEV devices
                      (leave-one-device-out), score |y - p| / (sigma x scale)
* ``crossinterp``     single + residual measured on train curves at the same
                      bias points, linearly interpolated along the sweep
                      (non-learned data control) / conformal as above
* ``learned``         single + kernel-ridge discrepancy learned from the other
                      DEV devices' held-out curves (features include the
                      cross-curve residual) / conformal on inner
                      leave-one-device-out residuals
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from ..card_eval import device_by_tag
from ..data_io import load_device_curves
from ..metrics import curve_trim
from .e4_heldout import curve_key

LEVELS = (0.5, 0.8, 0.9, 0.95)
PRIMARY_LEVEL = 0.9


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_device(run_dir: Path, tag: str, near_tol: float) -> dict:
    rec = json.loads((run_dir / f"{tag}.json").read_text())
    arrays = np.load(run_dir / f"{tag}.npz")
    members, draws = arrays["members"], arrays["draws"]
    device = device_by_tag(tag)
    curves = load_device_curves(device)
    keys = [curve_key(c) for c in curves]
    rows = {}
    for t in rec["train_tags"] + rec["test_tags"]:
        i = keys.index(t)
        c = curves[i]
        start, cleaned, den = curve_trim(device.dev_type, device.L_um,
                                         device.W_um, c.kind, c.Id)
        sweep = np.abs(np.asarray(c.Vd if c.kind == "idvd" else c.Vg, dtype=float))
        rows[t] = {"index": i, "start": int(start), "den": float(den),
                   "meas": np.asarray(cleaned, dtype=float),
                   "raw": np.asarray(c.Id, dtype=float), "sweep": sweep,
                   "kind": c.kind, "bias": abs(float(c.fixed)),
                   "n": len(c.Id)}
    s = rec["single_index"]
    best = rec["members"][s]["train_obj"]
    near = [k for k, m in enumerate(rec["members"])
            if m["train_obj"] <= (1 + near_tol) * best]
    return {"tag": tag, "rec": rec, "rows": rows, "members": members,
            "draws": draws, "single": s, "near": near,
            "sigma": rec["members"][s]["train_sigma"],
            "member_sigma": np.array([m["train_sigma"] for m in rec["members"]]),
            "family": device.dev_type, "L": device.L_um, "W": device.W_um}


def curve_of(arr: np.ndarray, row: dict) -> np.ndarray:
    return arr[..., row["index"], :row["n"]]


# ---------------------------------------------------------------------------
# Predictive quantiles
# ---------------------------------------------------------------------------
def mixture_quantiles(means: np.ndarray, sds: np.ndarray, levels) -> dict:
    """Pointwise quantiles of an equal-weight Gaussian mixture.
    ``means``/``sds``: (K, n).  Returns {level: (lo, hi)} central intervals."""
    sds = np.maximum(sds, 1e-30)
    lo_b = np.min(means - 8 * sds, axis=0)
    hi_b = np.max(means + 8 * sds, axis=0)
    out = {}
    for a in levels:
        bounds = []
        for p in ((1 - a) / 2, (1 + a) / 2):
            lo, hi = lo_b.copy(), hi_b.copy()
            for _ in range(70):
                mid = 0.5 * (lo + hi)
                cdf = np.mean(ndtr((mid[None] - means) / sds), axis=0)
                below = cdf < p
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
            bounds.append(0.5 * (lo + hi))
        out[a] = tuple(bounds)
    return out


def empirical_quantiles(samples: np.ndarray, levels) -> dict:
    return {a: (np.quantile(samples, (1 - a) / 2, axis=0),
                np.quantile(samples, (1 + a) / 2, axis=0)) for a in levels}


def scale_of(pred: np.ndarray) -> np.ndarray:
    """Mean |prediction| over the whole simulated curve (no measured data)."""
    return np.mean(np.abs(pred), axis=-1)


# ---------------------------------------------------------------------------
# Cross-curve residual interpolation (non-learned control)
# ---------------------------------------------------------------------------
def cross_residual(dev: dict, test_tag: str, pred_curves: dict) -> np.ndarray:
    """Residual (meas - pred) on the held-out curve's sweep, interpolated
    from the train curves that cross it at the same bias point, with a zero
    anchor at sweep voltage 0."""
    row = dev["rows"][test_tag]
    other = "idvg" if row["kind"] == "idvd" else "idvd"
    xs, rs = [0.0], [0.0]
    for t in dev["rec"]["train_tags"]:
        r = dev["rows"][t]
        if r["kind"] != other:
            continue
        k = int(np.argmin(np.abs(r["sweep"] - row["bias"])))
        if abs(r["sweep"][k] - row["bias"]) > 5e-3:
            continue
        xs.append(r["bias"])
        rs.append(float(r["raw"][k] - pred_curves[t][k]))
    order = np.argsort(xs)
    xs, rs = np.asarray(xs)[order], np.asarray(rs)[order]
    return np.interp(row["sweep"], xs, rs), xs


# ---------------------------------------------------------------------------
# Per-device method predictions
# ---------------------------------------------------------------------------
def device_predictions(dev: dict) -> dict:
    """{method: {test_tag: {"point": arr, "q": {level: (lo, hi)}}}} for the
    methods that need no other device."""
    s, near = dev["single"], dev["near"]
    out = {m: {} for m in ("single", "ens", "ens_env", "laplace",
                           "laplace_param", "crossinterp")}
    single_curves = {t: curve_of(dev["members"][s], r) for t, r in dev["rows"].items()}
    for t in dev["rec"]["test_tags"]:
        row = dev["rows"][t]
        single = single_curves[t]
        out["single"][t] = {"point": single, "q": {}}
        mem = curve_of(dev["members"][near], row)
        msd = dev["member_sigma"][near][:, None] * scale_of(mem)[:, None] \
            * np.ones_like(mem)
        out["ens"][t] = {"point": np.median(mem, axis=0),
                         "q": mixture_quantiles(mem, msd, LEVELS)}
        out["ens_env"][t] = {"point": np.median(mem, axis=0),
                             "q": {a: (mem.min(0), mem.max(0)) for a in LEVELS}}
        if dev["draws"].size:
            dr = curve_of(dev["draws"], row)
            dr = dr[np.all(np.isfinite(dr), axis=1)]
        else:
            dr = single[None]
        sd = dev["sigma"] * scale_of(single) * np.ones_like(dr)
        out["laplace"][t] = {"point": dr.mean(0),
                             "q": mixture_quantiles(dr, sd, LEVELS)}
        out["laplace_param"][t] = {"point": dr.mean(0),
                                   "q": empirical_quantiles(dr, LEVELS)}
        corr, _ = cross_residual(dev, t, single_curves)
        out["crossinterp"][t] = {"point": single + corr, "q": {}}
    return out


# ---------------------------------------------------------------------------
# Learned discrepancy (kernel ridge) and conformal intervals
# ---------------------------------------------------------------------------
def point_table(dev: dict) -> dict:
    """Features and targets on the trimmed held-out points of one device."""
    s = dev["single"]
    single_curves = {t: curve_of(dev["members"][s], r) for t, r in dev["rows"].items()}
    feats, targets, scales, meta = [], [], [], []
    for t in dev["rec"]["test_tags"]:
        row = dev["rows"][t]
        st = row["start"]
        pred = single_curves[t]
        sc = float(scale_of(pred))
        corr, anchors = cross_residual(dev, t, single_curves)
        v = row["sweep"]
        dist = np.min(np.abs(v[:, None] - anchors[None, :]), axis=1)
        f = np.stack([
            np.full_like(v, 1.0 if dev["family"] == "nmos" else -1.0),
            np.full_like(v, math.log10(dev["L"])),
            np.full_like(v, math.log10(dev["W"])),
            np.full_like(v, row["bias"] / 1.85),
            v / 1.85,
            corr / sc,
            dist / 1.85,
            np.full_like(v, math.log10(max(dev["sigma"], 1e-6))),
        ], axis=1)[st:]
        feats.append(f)
        targets.append((row["meas"] - pred)[st:] / sc)
        scales.append(np.full(len(v) - st, sc))
        meta.append((t, st, len(v)))
    return {"X": np.concatenate(feats), "y": np.concatenate(targets),
            "scale": np.concatenate(scales), "meta": meta,
            "sigma": dev["sigma"]}


class KernelRidge:
    def __init__(self, length: float, alpha: float):
        self.length, self.alpha = length, alpha

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.X = (X - self.mu) / self.sd
        K = self._k(self.X, self.X)
        self.coef = np.linalg.solve(K + self.alpha * np.eye(len(K)), y)
        return self

    def _k(self, A, B):
        d2 = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None]
              - 2 * A @ B.T)
        return np.exp(-0.5 * np.maximum(d2, 0) / (self.length ** 2 * A.shape[1]))

    def predict(self, X):
        return self._k((X - self.mu) / self.sd, self.X) @ self.coef


GRID = [(ell, a) for ell in (0.5, 1.0, 2.0) for a in (1e-2, 1e-1, 1.0)]
STRIDE = 3


def _stack(tables, idx, stride=STRIDE):
    X = np.concatenate([tables[i]["X"][::stride] for i in idx])
    y = np.concatenate([tables[i]["y"][::stride] for i in idx])
    w = np.concatenate([np.full(len(tables[i]["y"][::stride]),
                                1.0 / len(tables[i]["y"][::stride])) for i in idx])
    return X, y, w


def _lodo_predictions(tables, idx, length, alpha):
    """Leave-one-device-out predictions for each device in ``idx``."""
    out = {}
    for i in idx:
        rest = [j for j in idx if j != i]
        X, y, _ = _stack(tables, rest)
        out[i] = KernelRidge(length, alpha).fit(X, y).predict(tables[i]["X"])
    return out


def _device_mse(tables, preds):
    return float(np.mean([np.mean((tables[i]["y"] - p) ** 2)
                          for i, p in preds.items()]))


def weighted_quantile(values: list[np.ndarray], q: float) -> float:
    """Quantile of pooled values with each device's points summing to equal
    weight; finite-sample conformal correction (n_dev + 1) / n_dev."""
    n = len(values)
    v = np.concatenate(values)
    w = np.concatenate([np.full(len(x), 1.0 / len(x)) for x in values])
    order = np.argsort(v)
    cw = np.cumsum(w[order]) / n
    level = min(1.0, q * (n + 1) / n)
    k = int(np.searchsorted(cw, level))
    return float(v[order][min(k, len(v) - 1)])


def learned_and_conformal(devs: list[dict], base: dict[str, dict]) -> dict:
    """Leave-one-device-out over ``devs`` (DEV devices only).  Returns
    {method: {tag: {test_tag: {"point", "q"}}}} for conformal,
    crossinterp_conformal and learned, plus the chosen hyperparameters."""
    tables = [point_table(d) for d in devs]
    n = len(devs)
    out = {"conformal": {}, "crossinterp_conformal": {}, "learned": {},
           "learned_hyper": {}}
    # Nonconformity scores for the non-learned point predictors.
    raw = {}
    for name, method in (("conformal", "single"),
                         ("crossinterp_conformal", "crossinterp")):
        raw[name] = []
        for i, d in enumerate(devs):
            scores = []
            for t, st, _ in tables[i]["meta"]:
                row = d["rows"][t]
                p = base[d["tag"]][method][t]["point"]
                sc = float(scale_of(base[d["tag"]]["single"][t]["point"]))
                scores.append(np.abs(row["meas"] - p)[st:] / (d["sigma"] * sc))
            raw[name].append(np.concatenate(scores))
    for i, d in enumerate(devs):
        rest = [j for j in range(n) if j != i]
        for name, method in (("conformal", "single"),
                             ("crossinterp_conformal", "crossinterp")):
            qs = {a: weighted_quantile([raw[name][j] for j in rest], a)
                  for a in LEVELS}
            out[name][d["tag"]] = {}
            for t in d["rec"]["test_tags"]:
                p = base[d["tag"]][method][t]["point"]
                half = d["sigma"] * float(scale_of(base[d["tag"]]["single"][t]["point"]))
                out[name][d["tag"]][t] = {
                    "point": p, "q": {a: (p - qs[a] * half, p + qs[a] * half)
                                      for a in LEVELS}}
        # Learned discrepancy: hyperparameters by inner LODO on ``rest``.
        if len(rest) < 3:
            continue
        best = None
        for length, alpha in GRID:
            mse = _device_mse(tables, _lodo_predictions(tables, rest, length, alpha))
            if best is None or mse < best[0]:
                best = (mse, length, alpha)
        _, length, alpha = best
        inner = _lodo_predictions(tables, rest, length, alpha)
        scores = [np.abs(tables[j]["y"] - inner[j]) / tables[j]["sigma"] for j in rest]
        qs = {a: weighted_quantile(scores, a) for a in LEVELS}
        X, y, _ = _stack(tables, rest)
        model = KernelRidge(length, alpha).fit(X, y)
        out["learned_hyper"][d["tag"]] = {"length": length, "alpha": alpha,
                                          "inner_mse": best[0]}
        out["learned"][d["tag"]] = {}
        for t, st, npts in tables[i]["meta"]:
            row = d["rows"][t]
            single = base[d["tag"]]["single"][t]["point"]
            sc = float(scale_of(single))
            corr, anchors = cross_residual(
                d, t, {tt: curve_of(d["members"][d["single"]], r)
                       for tt, r in d["rows"].items()})
            # Predict on all sweep points (features are defined everywhere).
            v = row["sweep"]
            dist = np.min(np.abs(v[:, None] - anchors[None, :]), axis=1)
            F = np.stack([
                np.full_like(v, 1.0 if d["family"] == "nmos" else -1.0),
                np.full_like(v, math.log10(d["L"])),
                np.full_like(v, math.log10(d["W"])),
                np.full_like(v, row["bias"] / 1.85), v / 1.85, corr / sc,
                dist / 1.85,
                np.full_like(v, math.log10(max(d["sigma"], 1e-6)))], axis=1)
            p = single + model.predict(F) * sc
            half = d["sigma"] * sc
            out["learned"][d["tag"]][t] = {
                "point": p, "q": {a: (p - qs[a] * half, p + qs[a] * half)
                                  for a in LEVELS}}
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def device_metrics(dev: dict, preds: dict) -> dict:
    """Held-out RRMS, coverage, relative width and interval score."""
    rrms, cover, width, wink = [], {a: [] for a in LEVELS}, {a: [] for a in LEVELS}, []
    has_q = all(p["q"] for p in preds.values())
    finite = True
    for t, p in preds.items():
        row = dev["rows"][t]
        st, y, den = row["start"], row["meas"][row["start"]:], row["den"]
        pt = np.asarray(p["point"])[st:]
        if not np.all(np.isfinite(pt)):
            finite = False
            continue
        rrms.append(math.sqrt(float(np.mean((pt - y) ** 2))) / den)
        if has_q:
            for a in LEVELS:
                lo, hi = (np.asarray(b)[st:] for b in p["q"][a])
                cover[a].append(((y >= lo) & (y <= hi)))
                width[a].append((hi - lo) / den)
            lo, hi = (np.asarray(b)[st:] for b in p["q"][PRIMARY_LEVEL])
            al = 1 - PRIMARY_LEVEL
            wink.append(((hi - lo) + 2 / al * np.maximum(lo - y, 0)
                         + 2 / al * np.maximum(y - hi, 0)) / den)
    out = {"heldout_rrms": float(np.mean(rrms)) if finite and rrms else math.nan,
           "heldout_curve_rrms": rrms}
    if has_q and finite:
        for a in LEVELS:
            out[f"coverage_{a:g}"] = float(np.mean(np.concatenate(cover[a])))
            out[f"width_{a:g}"] = float(np.mean(np.concatenate(width[a])))
        out["interval_score_0.9"] = float(np.mean(np.concatenate(wink)))
    return out


def family_means(values: dict[str, float]) -> dict:
    vals = {t: v for t, v in values.items() if np.isfinite(v)}
    nm = [v for t, v in vals.items() if t.startswith("nmos")]
    pm = [v for t, v in vals.items() if t.startswith("pmos")]
    return {"all_device_mean": float(np.mean(list(vals.values()))) if vals else math.nan,
            "nmos": float(np.mean(nm)) if nm else math.nan,
            "pmos": float(np.mean(pm)) if pm else math.nan,
            "combined": float((np.mean(nm) + np.mean(pm)) / 2) if nm and pm else math.nan,
            "n_valid": len(vals), "n_devices": len(values)}
