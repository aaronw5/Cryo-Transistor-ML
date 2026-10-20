"""I7-2: hierarchical bias-only correction with few-shot per-device adaptation.

Pre-registration: ``out/research/explore/I7_2/prereg.json``.

``g_d(x) = g_shared(x) + phi(x) . delta_d``: ``g_shared`` is a stored I3-2
``M_bias`` fold correction (function of is_pmos, |Vgs|/VDD, |Vds|/VDD);
``phi`` is the quadratic basis in a = 2|Vgs|/VDD - 1, b = 2|Vds|/VDD - 1
(``basis``; 6 coefficients).  ``I_corr = I_card * exp(g_d)``.

Prior over delta (empirical Bayes): per fold, ``delta_j`` of every training
device is fitted on its starting record with the card fixed
(``fit_delta_full``); ``prior_from_deltas`` gives N(m, S) with
S = 0.5 Sigma_hat + 0.5 diag(Sigma_hat) + 1e-6 I.  Whitened coordinates
``z = Lc^-1 (delta - m)``.  The MAP objective on the K adaptation curves is
``J = mean curve RRMS + mu |z|^2``; ``mu`` is chosen per fold and K on the
training devices (``select_mu``; card fixed, leave-one-out prior).

Few-shot split rule (``draws``): Id-Vg curves sorted by |Vd|, Id-Vd curves
by |Vg|; fixed index positions per draw (K2a, K2b, K4a, K4b).

Held-out joint fit of card and z: ``HierProblem`` (card simulation cached,
so finite-difference columns in z cost no simulation) and ``polish_prior``
(``ngspice_fit.polish`` with the prior residual and J-based best tracking).

Also: ``subset_problem`` (a fit problem restricted to the adaptation
curves), ``restrict`` (a record with inclusion limited to a key set),
``score_keys`` (strict paper RRMS on a key subset), ``crossinterp_sims``
(the E4 non-learned cross-curve residual control) and ``decide``.
"""

from __future__ import annotations

import copy
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np

from ..metrics import clean_current
from .e1_correction import Correction, DeviceRec, raw_features, score_device

N_BASIS = 6
Z_MAX = 5.0
DELTA_MAX = 3.0
MU_GRID = (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, math.inf)
SHRINK = 0.5
JITTER = 1e-6
RIDGE_FULL = 1e-6
CROSS_TOL = 5e-3
DRAWS = ("K2a", "K2b", "K4a", "K4b")
DRAW_K = {"K2a": 2, "K2b": 2, "K4a": 4, "K4b": 4}
MIN_HELDOUT = 2


# ---------------------------------------------------------------------------
# Basis and corrections
# ---------------------------------------------------------------------------
def basis(phi: np.ndarray) -> np.ndarray:
    """(n, 6) quadratic basis from E1 raw features (columns 1, 2 =
    |Vgs|/VDD, |Vds|/VDD): 1, a, b, a^2, a b, b^2 with a, b = 2 x - 1."""
    phi = np.asarray(phi, dtype=np.float64)
    a = 2.0 * phi[:, 1] - 1.0
    b = 2.0 * phi[:, 2] - 1.0
    return np.column_stack([np.ones(len(a)), a, b, a * a, a * b, b * b])


@dataclass
class ShiftedCorrection(Correction):
    """g = base.g + basis(phi) @ delta."""
    base: Correction | None = None
    delta: np.ndarray | None = None

    def g(self, phi, group_tag=None, group_kind=None):
        g0 = self.base.g(phi, group_tag, group_kind)
        return g0 + basis(phi) @ np.asarray(self.delta, dtype=np.float64)

    def to_json(self) -> dict:
        return {"arm": self.arm, "kind": "shifted", "base": self.base.to_json(),
                "delta": np.asarray(self.delta, float).tolist(),
                "basis": "1, a, b, a^2, a b, b^2; a = 2|Vgs|/VDD - 1, b = 2|Vds|/VDD - 1"}


# ---------------------------------------------------------------------------
# Records: inclusion subsets, scoring, curve ordering and draws
# ---------------------------------------------------------------------------
def restrict(rec: DeviceRec, keys) -> DeviceRec:
    """Copy of ``rec`` whose inclusion is limited to ``keys`` (frozen
    inclusion AND key in keys)."""
    keys = set(keys)
    curves = [replace(c, included=bool(c.included and c.key in keys)) for c in rec.curves]
    return replace(rec, curves=curves)


def score_keys(rec: DeviceRec, sims, keys) -> dict:
    """Strict paper RRMS of ``sims`` on the included curves in ``keys``."""
    return score_device(restrict(rec, keys), sims)


def fixed_bias(c) -> float:
    """|fixed terminal bias| of a curve: |Vd| for Id-Vg, |Vg| for Id-Vd."""
    arr = c.vd if c.kind == "idvg" else c.vg
    return float(np.median(np.abs(np.asarray(arr, dtype=np.float64))))


def ordered_keys(rec: DeviceRec) -> tuple[list[str], list[str]]:
    """Included Id-Vg keys by ascending |Vd|, included Id-Vd keys by
    ascending |Vg| (stable on ties by record order)."""
    inc = [c for c in rec.curves if c.included and c.den > 0]
    G = sorted([c for c in inc if c.kind == "idvg"], key=fixed_bias)
    D = sorted([c for c in inc if c.kind == "idvd"], key=fixed_bias)
    return [c.key for c in G], [c.key for c in D]


def draw_indices(name: str, nG: int, nD: int) -> tuple[list[int], list[int]]:
    if name == "K2a":
        return [nG - 1], [nD - 1]
    if name == "K2b":
        return [0], [(nD - 1) // 2]
    if name == "K4a":
        return [0, nG - 1], [(nD - 1) // 2, nD - 1]
    if name == "K4b":
        return ([(nG - 1) // 3, math.ceil(2 * (nG - 1) / 3)],
                [(nD - 1) // 3, math.ceil(2 * (nD - 1) / 3)])
    raise KeyError(name)


def draws(rec: DeviceRec, names=DRAWS) -> dict:
    """{draw: {"adapt": [keys], "heldout": [keys]}} or {draw: None} when the
    draw is not defined for this device (prereg curve_split_rule)."""
    G, D = ordered_keys(rec)
    all_inc = G + D
    out = {}
    for name in names:
        if not G or not D:
            out[name] = None
            continue
        ig, id_ = draw_indices(name, len(G), len(D))
        if len(set(ig)) != len(ig) or len(set(id_)) != len(id_) or min(ig + id_) < 0:
            out[name] = None
            continue
        adapt = [G[i] for i in ig] + [D[i] for i in id_]
        held = [k for k in all_inc if k not in set(adapt)]
        out[name] = {"adapt": adapt, "heldout": held} if len(held) >= MIN_HELDOUT else None
    return out


# ---------------------------------------------------------------------------
# Delta fits with the card fixed (numpy objective and gradient)
# ---------------------------------------------------------------------------
@dataclass
class PointSet:
    """Trimmed included points of a record subset, with g_shared and the
    basis evaluated once."""
    sim: list = field(default_factory=list)
    tgt: list = field(default_factory=list)
    g0: list = field(default_factory=list)
    B: list = field(default_factory=list)
    den: list = field(default_factory=list)
    keys: list = field(default_factory=list)

    @property
    def n_curves(self) -> int:
        return len(self.den)


def point_set(rec: DeviceRec, corr: Correction, keys=None) -> PointSet:
    ps = PointSet()
    for c in rec.curves:
        if not c.included or not (c.den > 0) or (keys is not None and c.key not in keys):
            continue
        s = slice(c.start, None)
        sim = np.asarray(c.sim, dtype=np.float64)[s]
        phi = raw_features(rec, np.asarray(c.vg)[s], np.asarray(c.vd)[s], sim)
        n = len(sim)
        ps.sim.append(sim)
        ps.tgt.append(clean_current(c.target)[s])
        ps.g0.append(corr.g(phi, [f"{rec.family}|{c.key}"] * n, [f"{rec.family}|{c.kind}"] * n))
        ps.B.append(basis(phi))
        ps.den.append(float(c.den))
        ps.keys.append(c.key)
    return ps


def finite_points(ps: PointSet) -> bool:
    return all(np.all(np.isfinite(s)) and np.all(np.isfinite(g)) for s, g in zip(ps.sim, ps.g0))


def curve_values(ps: PointSet, delta: np.ndarray) -> np.ndarray:
    out = []
    for sim, tgt, g0, B, den in zip(ps.sim, ps.tgt, ps.g0, ps.B, ps.den):
        y = sim * np.exp(g0 + B @ delta)
        out.append(math.sqrt(float(np.mean((y - tgt) ** 2))) / den)
    return np.asarray(out)


def mean_rrms_and_grad(ps: PointSet, delta: np.ndarray) -> tuple[float, np.ndarray]:
    """mean over curves of RRMS_c(delta) and its gradient in delta."""
    total, grad = 0.0, np.zeros(N_BASIS)
    for sim, tgt, g0, B, den in zip(ps.sim, ps.tgt, ps.g0, ps.B, ps.den):
        with np.errstate(over="ignore", invalid="ignore"):
            y = sim * np.exp(g0 + B @ delta)
        r = y - tgt
        mse = float(np.mean(r * r))
        rms = math.sqrt(mse + 1e-300)
        total += rms / den
        grad += (B.T @ (r * y)) / (len(r) * den * rms)
    n = max(1, ps.n_curves)
    return total / n, grad / n


def fit_delta_full(ps: PointSet) -> np.ndarray:
    """ML delta (+ RIDGE_FULL |delta|^2) on all points of ``ps``,
    |delta_k| <= DELTA_MAX, L-BFGS-B from 0."""
    from scipy.optimize import minimize

    def fun(d):
        v, g = mean_rrms_and_grad(ps, d)
        return v + RIDGE_FULL * float(d @ d), g + 2 * RIDGE_FULL * d
    res = minimize(fun, np.zeros(N_BASIS), jac=True, method="L-BFGS-B",
                   bounds=[(-DELTA_MAX, DELTA_MAX)] * N_BASIS, options={"maxiter": 500})
    return np.asarray(res.x, dtype=np.float64)


@dataclass
class Prior:
    m: np.ndarray
    S: np.ndarray
    Lc: np.ndarray
    n: int

    def delta(self, z) -> np.ndarray:
        return self.m + self.Lc @ np.asarray(z, dtype=np.float64)

    def to_json(self) -> dict:
        return {"m": self.m.tolist(), "S": self.S.tolist(), "n_devices": self.n,
                "sd": np.sqrt(np.diag(self.S)).tolist()}


def prior_from_deltas(deltas: np.ndarray) -> Prior:
    d = np.asarray(deltas, dtype=np.float64)
    m = d.mean(axis=0)
    if len(d) > 1:
        sig = np.cov(d, rowvar=False, ddof=1)
    else:
        sig = np.zeros((N_BASIS, N_BASIS))
    S = (1 - SHRINK) * sig + SHRINK * np.diag(np.diag(sig)) + JITTER * np.eye(N_BASIS)
    return Prior(m, S, np.linalg.cholesky(S), len(d))


def fit_z(ps: PointSet, prior: Prior, mu: float, z0=None) -> np.ndarray:
    """argmin_z mean RRMS(m + Lc z) + mu |z|^2, z in [-Z_MAX, Z_MAX]^6
    (card fixed).  mu = inf gives z = 0."""
    from scipy.optimize import minimize
    if math.isinf(mu):
        return np.zeros(N_BASIS)

    def fun(z):
        v, g = mean_rrms_and_grad(ps, prior.delta(z))
        return v + mu * float(z @ z), prior.Lc.T @ g + 2 * mu * z
    x0 = np.zeros(N_BASIS) if z0 is None else np.asarray(z0, float)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   bounds=[(-Z_MAX, Z_MAX)] * N_BASIS, options={"maxiter": 300})
    return np.asarray(res.x, dtype=np.float64)


def mean_rrms(ps: PointSet, delta) -> float:
    v = curve_values(ps, np.asarray(delta, float))
    return float(np.mean(v)) if len(v) and np.all(np.isfinite(v)) else float("nan")


def select_mu(train: list[DeviceRec], corr: Correction, K: int, grid=MU_GRID) -> dict:
    """Empirical-Bayes choice of mu for one fold and K (prereg
    mu_selection_empirical_bayes).  Returns the chosen mu, the per-mu
    criterion (mean log held-out RRMS over training devices and draws of
    that K) and the number of (device, draw) tasks."""
    full = []
    for rec in train:
        ps = point_set(rec, corr)
        if ps.n_curves and finite_points(ps):
            full.append((rec, fit_delta_full(ps)))
    names = [d for d in DRAWS if DRAW_K[d] == K]
    logs = {mu: [] for mu in grid}
    n_tasks = 0
    for j, (rec, _) in enumerate(full):
        prior = prior_from_deltas(np.array([d for i, (_, d) in enumerate(full) if i != j]))
        for dr in draws(rec, names).values():
            if dr is None:
                continue
            pa = point_set(rec, corr, set(dr["adapt"]))
            ph = point_set(rec, corr, set(dr["heldout"]))
            n_tasks += 1
            z = None
            for mu in grid:
                z = fit_z(pa, prior, mu, None if math.isinf(mu) else z)
                v = mean_rrms(ph, prior.delta(z))
                logs[mu].append(math.log(v) if v > 0 and np.isfinite(v) else math.nan)
    crit = {mu: float(np.nanmean(v)) if v and np.any(np.isfinite(v)) else math.nan
            for mu, v in logs.items()}
    finite = [(c, mu) for mu, c in crit.items() if np.isfinite(c)]
    if not finite:
        chosen = math.inf
    else:
        best = min(c for c, _ in finite)
        chosen = max(mu for c, mu in finite if c <= best)       # ties: larger mu
    return {"mu": chosen, "criterion": {repr(k): v for k, v in crit.items()},
            "n_tasks": n_tasks, "n_train_devices": len(full)}


def fold_prior(train: list[DeviceRec], corr: Correction) -> tuple[Prior, dict]:
    deltas, tags = [], []
    for rec in train:
        ps = point_set(rec, corr)
        if ps.n_curves and finite_points(ps):
            deltas.append(fit_delta_full(ps))
            tags.append(rec.tag)
    prior = prior_from_deltas(np.array(deltas))
    return prior, {t: d.tolist() for t, d in zip(tags, deltas)}


# ---------------------------------------------------------------------------
# Fit problems restricted to the adaptation curves; joint card + z problem
# ---------------------------------------------------------------------------
def subset_problem(prob, keys):
    """Shallow copy of a fit problem (``layout`` / ``curves`` interface of
    ``ngspice_fit.DeviceProblem`` and the I5-1 / I6-1 ``MeasProblem``) that
    simulates and scores only the layout rows whose tag is in ``keys``."""
    keys = set(keys)
    rows = [r for r in prob.layout if r["tag"] in keys]
    if {r["tag"] for r in rows} != keys:
        raise KeyError(f"{prob.tag}: keys {sorted(keys - {r['tag'] for r in rows})} not in layout")
    idx = sorted({r["index"] for r in rows})
    remap = {old: new for new, old in enumerate(idx)}
    q = copy.copy(prob)
    q.curves = [prob.curves[i] for i in idx]
    if hasattr(prob, "meas"):
        q.meas = q.curves
    q.layout = [{**r, "index": remap[r["index"]]} for r in rows]
    q.n_curves = len(q.layout)
    q.n_sim = 0
    return q


def z_names() -> tuple[str, ...]:
    return tuple(f"z{k}" for k in range(N_BASIS))


def extend_with_z(box):
    from .i4_1_baselines import extend_box
    return extend_box(box, {n: {"value": 0.0, "bounds": [-Z_MAX, Z_MAX], "log": False}
                            for n in z_names()})


class HierProblem:
    """Card local coordinates followed by z / Z_MAX.  ``wrapped`` is a
    ``CorrectedProblem`` over the subset problem (used for ``record``);
    ``simulate`` returns the subset sims times exp(g_shared + phi . delta(z)).
    Card simulations are cached (LRU) by the card local vector."""

    def __init__(self, wrapped, g_shared: Correction, prior: Prior, card_box,
                 cache_size: int = 256):
        self.wrapped = wrapped
        self.base = wrapped.base
        self.g_shared = g_shared
        self.prior = prior
        self.card_box = card_box
        self.box = extend_with_z(card_box)
        self.n_card = card_box.n_parameters
        self._cache: OrderedDict = OrderedDict()
        self._size = cache_size
        self.n_cache_hits = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def split(self, local) -> tuple[np.ndarray, np.ndarray]:
        local = np.asarray(local, dtype=np.float64)
        z = np.clip(local[self.n_card:], -1.0, 1.0) * Z_MAX
        return local[:self.n_card], z

    def card_sims(self, card_local):
        key = np.asarray(card_local, dtype=np.float64).tobytes()
        if key in self._cache:
            self._cache.move_to_end(key)
            self.n_cache_hits += 1
            return self._cache[key]
        sims = self.base.simulate(card_local)
        self._cache[key] = sims
        if len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return sims

    def correction(self, z) -> ShiftedCorrection:
        return ShiftedCorrection("hier", base=self.g_shared, delta=self.prior.delta(z))

    def simulate(self, local):
        card, z = self.split(local)
        sims = self.card_sims(card)
        return self.correction(z).apply(self.wrapped.record(sims))

    def curve_rrms(self, sims):
        return self.base.curve_rrms(sims)

    def residual_vector(self, sims, weights):
        return self.base.residual_vector(sims, weights)


def polish_prior(problem: HierProblem, local0, *, mu: float, max_nfev: int = 400,
                 diff_step: float = 0.01, irls_rounds: int = 2) -> dict:
    """``ngspice_fit.polish`` on a ``HierProblem`` with the prior: residual
    = polish residual + sqrt(2 s mu) z (s = mean curve RRMS in IRLS round 1,
    harmonic mean of curve RRMS in later rounds, at the round's start
    point); the returned point is the lowest J = mean curve RRMS + mu |z|^2
    seen.  With mu = 0 the residual and tracking equal polish's."""
    from scipy.optimize import least_squares
    t0 = time.time()
    active = np.flatnonzero(problem.box.active)
    local = np.asarray(local0, dtype=np.float64).copy()

    def J(curves, loc):
        _, z = problem.split(loc)
        return float(np.mean(curves)) + (mu * float(z @ z) if mu > 0 else 0.0)

    curves0 = problem.curve_rrms(problem.simulate(local))
    best = {"local": local.copy(), "objective": J(curves0, local)}
    history = [best["objective"]]
    weights = np.ones(problem.n_curves)
    s = float(np.mean(curves0))
    for _ in range(max(1, irls_rounds)):
        scale = math.sqrt(2.0 * s * mu) if mu > 0 else 0.0

        def residual(values):
            trial = local.copy()
            trial[active] = values
            sims = problem.simulate(trial)
            curves = problem.curve_rrms(sims)
            value = J(curves, trial)
            if value < best["objective"]:
                best.update(local=trial.copy(), objective=value)
            r = problem.residual_vector(sims, weights)
            if mu > 0:
                _, z = problem.split(trial)
                r = np.concatenate([r, scale * z])
            return r

        def jacobian(values):
            base = residual(values)
            columns = np.empty((base.size, values.size))
            for index in range(values.size):
                shifted = values.copy()
                step = diff_step if shifted[index] + diff_step <= 1.0 else -diff_step
                shifted[index] += step
                columns[:, index] = (residual(shifted) - base) / step
            return columns

        budget = max(2, max_nfev // max(1, irls_rounds) // (len(active) + 2))
        x0 = np.clip(local[active], -1 + 1e-9, 1 - 1e-9)
        try:
            least_squares(residual, x0, method="trf", jac=jacobian,
                          bounds=(-np.ones(len(active)), np.ones(len(active))),
                          xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=budget)
        except ValueError:
            pass
        local = best["local"].copy()
        curves = problem.curve_rrms(problem.simulate(local))
        history.append(J(curves, local))
        weights = 1.0 / np.sqrt(np.maximum(curves, 1e-3))
        weights *= math.sqrt(problem.n_curves / float(np.sum(weights ** 2)))
        c = np.maximum(curves, 1e-3)
        s = float(len(c) / np.sum(1.0 / c))
    return {"local": best["local"], "objective": best["objective"], "history": history,
            "n_sim": problem.n_sim, "n_cache_hits": problem.n_cache_hits,
            "runtime_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# E4 cross-curve residual control
# ---------------------------------------------------------------------------
def crossinterp_sims(rec: DeviceRec, adapt, heldout, tol: float = CROSS_TOL
                     ) -> tuple[list[np.ndarray], dict]:
    """Simulated currents of ``rec`` plus, on each held-out curve, the
    residual (cleaned measured - simulated) of the adaptation curves of the
    other kind at their crossing with it, linearly interpolated along the
    held-out sweep (|V|) with a zero anchor at 0 V (E4 ``cross_residual``).
    Returns the arrays (one per record curve) and per-held-out-curve
    crossing counts."""
    by_key = {c.key: c for c in rec.curves}
    adapt, heldout = list(adapt), set(heldout)
    out, crossings = [], {}
    for c in rec.curves:
        sim = np.asarray(c.sim, dtype=np.float64)
        if c.key not in heldout:
            out.append(sim)
            continue
        other = "idvg" if c.kind == "idvd" else "idvd"
        bias = fixed_bias(c)
        sweep = np.abs(np.asarray(c.vd if c.kind == "idvd" else c.vg, dtype=np.float64))
        xs, rs = [0.0], [0.0]
        for k in adapt:
            a = by_key[k]
            if a.kind != other:
                continue
            a_sweep = np.abs(np.asarray(a.vd if a.kind == "idvd" else a.vg, dtype=np.float64))
            i = int(np.argmin(np.abs(a_sweep - bias)))
            if abs(a_sweep[i] - bias) > tol:
                continue
            meas = clean_current(a.target)
            xs.append(fixed_bias(a))
            rs.append(float(meas[i] - np.asarray(a.sim, float)[i]))
        crossings[c.key] = len(xs) - 1
        order = np.argsort(xs)
        corr = np.interp(sweep, np.asarray(xs)[order], np.asarray(rs)[order])
        out.append(sim + corr)
    return out, crossings


# ---------------------------------------------------------------------------
# Aggregation and decision
# ---------------------------------------------------------------------------
ARMS = ("a_C0", "b_shared", "c_hier", "c_ml", "d_scratch", "e_crossinterp")


def device_values(rows: list[dict], arm: str, k: int | None = None) -> dict:
    """{tag: geometric mean over the device's draws (of K = k, or all) of
    held-out RRMS}; NaN if any such draw failed for ``arm``."""
    by: dict = {}
    for r in rows:
        if k is not None and DRAW_K[r["draw"]] != k:
            continue
        by.setdefault(r["tag"], []).append(r["arms"].get(arm, {}).get("heldout_rrms", math.nan))
    out = {}
    for tag, vals in by.items():
        v = np.asarray(vals, dtype=float)
        out[tag] = float(np.exp(np.mean(np.log(v)))) if np.all(np.isfinite(v)) and np.all(v > 0) \
            else math.nan
    return out


RULE_DATASETS = ("sky130", "ihp300", "tud4", "tud295")


def decide(cmp: dict, failed: dict) -> dict:
    """``cmp[ds][name]``: paired_ratio dicts for names 'c_vs_b', 'c_vs_a',
    'c_vs_d'; ``failed[ds]`` = {"n_failed": .., "n": .., "n_paired": ..}
    for c_hier.  Prereg criterion C1-C4 and the outcome."""
    ds = [d for d in RULE_DATASETS if d in cmp]
    hi = lambda d, n: cmp[d][n].get("ci_high", math.nan)
    pt = lambda d, n: cmp[d][n].get("ratio", math.nan)
    c1_n = sum(1 for d in ds if hi(d, "c_vs_b") < 1.0)
    c2_n = sum(1 for d in ds if hi(d, "c_vs_a") < 1.0)
    c3_pt = sum(1 for d in ds if pt(d, "c_vs_d") < 1.0)
    c3_ci = sum(1 for d in ds if hi(d, "c_vs_d") < 1.0)
    c4 = {d: bool(failed[d]["n_failed"] <= 0.10 * failed[d]["n"] and failed[d]["n_paired"] >= 8)
          for d in ds}
    complete = len(ds) == len(RULE_DATASETS)
    C = {"C1": complete and c1_n >= 3, "C2": complete and c2_n == 4,
         "C3": complete and c3_pt == 4 and c3_ci >= 3, "C4": complete and all(c4.values())}
    no_gain = sum(1 for d in ds if pt(d, "c_vs_b") >= 0.98)
    if not complete:
        outcome = "incomplete"
    elif sum(1 for v in c4.values() if not v) >= 2:
        outcome = "invalid"
    elif all(C.values()):
        outcome = "supported_continue"
    elif no_gain >= 2:
        outcome = "delta_no_gain_stop"
    elif C["C1"] and C["C2"] and not C["C3"]:
        outcome = "sharing_no_gain"
    else:
        outcome = "inconclusive"
    return {"conditions": C, "counts": {"C1_ci_lt1_c_vs_b": c1_n, "C2_ci_lt1_c_vs_a": c2_n,
                                        "C3_point_lt1_c_vs_d": c3_pt, "C3_ci_lt1_c_vs_d": c3_ci,
                                        "c_vs_b_point_ge_0.98": no_gain},
            "C4_per_dataset": c4, "outcome": outcome, "datasets": ds}
