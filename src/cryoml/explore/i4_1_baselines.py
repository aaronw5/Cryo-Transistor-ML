"""I4-1: baselines for the I3-2 bias-only correction, and the I4-2 analysis.

Pre-registration: ``out/research/explore/I4_1/prereg.json``.

Every arm refits the held-out card in the loop with the I3-2 settings
(``polish`` max_nfev 400, irls 2, diff_step 0.01) and is scored with the
strict paper RRMS (``e1_correction.score_device``).

* ``B1`` wider parameter set: the fitted box plus extra card parameters
  (``b1_extra_box``; list fixed by the edge screen ``edge_response``).
* ``B2`` physical augmentation: an extrinsic series resistance per side
  (``rext_ohm``, a netlist resistor outside the card on drain and source)
  and a first-order self-heating factor ``exp(kappa * |Vds I| / P_max)``
  (``heat_factor``), both fitted per device (``AugProblem``).
* ``B3`` shared polynomial correction of total degree 2 or 3 in
  (|Vgs|/VDD, |Vds|/VDD) per polarity (``PolyCorrection``, ``fit_poly``).
* ``B4`` Kennedy-O'Hagan-type discrepancy: GP posterior mean of the
  log-current residual on (is_pmos, |Vgs|/VDD, |Vds|/VDD)
  (``GPCorrection``, ``fit_gp``).
* ``B5`` per-device ``M_bias``-form MLP fitted on the held-out device's own
  curves (in-sample reference; trained with ``i3_2_bias.fit_bias``).

B1 and B2 iteration budgets are matched to C0_refit (``matched_nfev``).

The I4-2 helpers (``g_grid``, ``linear_vth``, ``region_masks``,
``region_stats``, ``t1_comparison``) read stored corrections only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..metrics import clean_current
from ..parameter_space import ParameterBox
from .e1_correction import OUT_SCALE, Correction, DeviceRec, _loss_fn, make_batch

# ---------------------------------------------------------------------------
# B1: wider parameter set
# ---------------------------------------------------------------------------
B1_FACTOR = 2.0            # nonzero value v: log-uniform |v| / 2 .. 2 |v|, sign kept
B1_THRESHOLD = 1e-4        # edge-screen response threshold (as the 43-set screen)
# BSIM4 parameters whose card value is zero: absolute linear range (extended
# to include the card value).  Used when |v| <= 1e-3 * max(|lo|, |hi|).
# "rdsw" means: upper bound = the device's starting-card RDSW.
BSIM4_ZERO_RANGE: dict[str, tuple] = {
    "ud": (0.0, 1e17),        # 1/m^2; mobmod 0 term ud*Vth^2*(toxe/(Vgsteff+2Vth))^2 ~0.26 at 1e17
    "up": (0.0, 0.5),         # mobility factor 1 - up*exp(-Leff/lp)
    "a1": (0.0, 0.5),         # 1/V, non-saturation factor Lambda = a1*Vgsteff + a2
    "pvag": (0.0, 1.0),       # gate dependence of the Early voltage
    "fprout": (0.0, 200.0),   # V/m^0.5, pocket-implant Rout degradation
    "pdits": (0.0, 1.0),      # 1/V, drain-induced Vth shift on Rout
    "pditsd": (0.0, 1.0),     # 1/V
    "pditsl": (0.0, 1e6),     # 1/m
    "alpha1": (0.0, 100.0),   # 1/V, length term of impact ionization
    "voffl": (-1e-8, 1e-8),   # V*m, length dependence of voff
    "minv": (-2.0, 2.0),      # moderate-inversion smoothing
    "dvtp0": (0.0, 1e-7),     # m, pocket-implant DIBL length
    "dvtp1": (0.0, 1.0),      # 1/V, pocket-implant DIBL
    "rdswmin": (0.0, "rdsw"),  # ohm*um^WR
    "agidl": (0.0, 1e-6),     # 1/ohm, GIDL prefactor
    "dvt0w": (0.0, 1.0),      # narrow-width short-channel coefficient
    "pdiblc1": (0.0, 1.0),    # DIBL output-resistance coefficient
    "cdsc": (0.0, 2.5e-3),    # F/m^2, about 0.3 Coxe
    "phin": (-0.2, 0.2),      # V, non-uniform vertical doping
    "b0": (0.0, 1e-7),        # m, bulk-charge width effect
    "b1": (0.0, 1e-7),        # m
}
# PSP 103 (IHP) candidate DC parameters (zero-valued ones are left inactive).
PSP_B1_CANDIDATES = (
    "lap", "wot", "dwq", "lvaro", "lvarl",
    "vfbl", "vfbw", "vfblw", "toxo", "nsubo", "nsubw", "wseg", "npck", "npckw",
    "wsegp", "lpck", "lpckw", "fol1", "fol2", "dnsubo", "dphiblexp", "dphibw",
    "dphiblw", "npo", "npl",
    "cto", "ctlexp", "ctw", "ctlw", "cflexp", "cfw", "cfbo",
    "fbet1", "fbet1w", "lp1", "lp1w", "fbet2", "lp2", "betw1", "betw2", "wbet",
    "mueo", "cso", "csl", "cslexp", "csw", "cslw", "thecso", "xcoro", "xcorl",
    "xcorw", "xcorlw", "fetao",
    "rsw2", "rsbo", "rsgo",
    "thesatlexp", "thesatw", "thesatlw", "thesatbo", "thesatgo",
    "axl", "alpl", "alplexp", "alpw", "alp1l1", "alp1lexp", "alp1l2", "alp1w",
    "alp2l1", "alp2lexp", "alp2l2", "alp2w", "vpo",
    "a1o", "a1l", "a1w", "a2o", "a3o", "a3l", "a3w", "a4o", "a4l", "a4w",
    "gcoo", "igovw", "gc2o", "gc3o", "chibo",
    "agidlw", "bgidlo", "cgidlo",
)


def b1_param_box(name: str, value: float, zero_range: dict | None,
                 rdsw: float | None = None) -> tuple[list[float], bool]:
    """(bounds, log flag) of one B1 extra parameter at card value ``value``.

    Zero-range rule: if ``name`` has an entry and |value| <= 1e-3 * the
    entry's largest magnitude, the box is that linear range extended to
    include ``value``.  Otherwise a nonzero value gets the log-uniform box
    |v| / B1_FACTOR .. B1_FACTOR |v| with the sign kept, and a zero value
    without an entry is inactive ([0, 0])."""
    v = float(value)
    if zero_range and name in zero_range:
        lo, hi = zero_range[name]
        if hi == "rdsw":
            hi = float(rdsw) if rdsw is not None else 0.0
        mag = max(abs(lo), abs(hi))
        if mag > 0 and abs(v) <= 1e-3 * mag:
            return [min(lo, v), max(hi, v)], False
    if v == 0.0:
        return [0.0, 0.0], False
    a, b = v / B1_FACTOR, v * B1_FACTOR
    return [min(a, b), max(a, b)], True


def extend_box(base: ParameterBox, extras: dict[str, dict]) -> ParameterBox:
    """``base`` plus extra parameters ``{name: {"value", "bounds", "log"}}``
    (appended in the given order; ``published`` = card value)."""
    names = tuple(base.names) + tuple(extras)
    published = {**{n: float(base.published[n]) for n in base.names},
                 **{n: float(e["value"]) for n, e in extras.items()}}
    bounds = {**{n: list(base.bounds[n]) for n in base.names},
              **{n: list(e["bounds"]) for n, e in extras.items()}}
    base_log = list(base.log_scale) if base.log_scale is not None else \
        [False] * len(base.names)
    logs = tuple(base_log) + tuple(bool(e["log"]) for e in extras.values())
    return ParameterBox(names=names, published=published, bounds=bounds,
                        log_scale=logs)


def b1_extra_box(names, values: dict, zero_range: dict | None,
                 rdsw: float | None = None) -> dict[str, dict]:
    """Extras for ``extend_box`` from card ``values`` (names absent from
    ``values`` are skipped)."""
    out = {}
    for name in names:
        if name not in values:
            continue
        bounds, log = b1_param_box(name, values[name], zero_range, rdsw)
        out[name] = {"value": float(values[name]), "bounds": bounds, "log": log}
    return out


def edge_response(reference, candidate, layout) -> float:
    """Max over the included curves of RMS(candidate - reference) / the paper
    denominator, on the trimmed points (NaN if the candidate is non-finite)."""
    vals = []
    for row in layout:
        ref = np.asarray(reference[row["index"]], float)[row["start"]:]
        cand = np.asarray(candidate[row["index"]], float)[row["start"]:]
        if not np.all(np.isfinite(cand)):
            return float("nan")
        vals.append(math.sqrt(float(np.mean((cand - ref) ** 2))) / row["denominator"])
    return float(max(vals)) if vals else 0.0


def matched_nfev(n_base_active: int, n_arm_active: int, max_nfev: int = 400,
                 irls: int = 2) -> int:
    """``max_nfev`` for an arm with ``n_arm_active`` active parameters such
    that ``ngspice_fit.polish`` gets the same per-IRLS-round TRF budget
    (``max(2, max_nfev // irls // (n + 2))``) as the base box."""
    budget = max(2, max_nfev // max(1, irls) // (n_base_active + 2))
    return int(budget * max(1, irls) * (n_arm_active + 2))


# ---------------------------------------------------------------------------
# B2: extrinsic series resistance and first-order self-heating
# ---------------------------------------------------------------------------
B2_NAMES = ("rext_ohm", "heat_kappa")
B2_REXT_FACTOR = 4.0        # rext upper bound = 4 * series_r_per_side(start) / W
B2_KAPPA = (-0.5, 0.5)


def heat_factor(vd: np.ndarray, sim: np.ndarray, kappa: float, pmax: float) -> np.ndarray:
    """exp(kappa * |Vds * I_sim| / P_max)."""
    p = np.abs(np.asarray(vd, float) * np.asarray(sim, float))
    with np.errstate(invalid="ignore", over="ignore"):
        return np.exp(kappa * p / pmax)


def measured_pmax(rec: DeviceRec) -> float:
    """Largest |Vds * I_target| over the included trimmed points."""
    best = 0.0
    for c in rec.curves:
        if not c.included:
            continue
        s = slice(c.start, None)
        p = np.abs(np.asarray(c.vd, float)[s] * clean_current(c.target)[s])
        p = p[np.isfinite(p)]
        if len(p):
            best = max(best, float(p.max()))
    return best if best > 0 else 1.0


def b2_box(base: ParameterBox, rext_max: float) -> ParameterBox:
    """``base`` + rext_ohm in [0, rext_max] (start 0) + heat_kappa in
    B2_KAPPA (start 0), both linear."""
    return extend_box(base, {
        "rext_ohm": {"value": 0.0, "bounds": [0.0, float(rext_max)], "log": False},
        "heat_kappa": {"value": 0.0, "bounds": list(B2_KAPPA), "log": False}})


class AugProblem:
    """``DeviceProblem`` interface over a card box plus the B2 elements.

    ``sim_fn(card_params, rext_ohm) -> list of arrays`` (one per
    ``base.curves`` entry) runs NGSpice; the heating factor uses each curve's
    ``Vd`` array.  ``curve_rrms`` / ``residual_vector`` / ``layout`` come
    from ``base``."""

    def __init__(self, base, box: ParameterBox, sim_fn, pmax: float):
        self.base = base
        self.box = box
        self.sim_fn = sim_fn
        self.pmax = float(pmax)
        self.n_sim = 0
        self._card = [n for n in box.names if n not in B2_NAMES]

    def __getattr__(self, name):
        return getattr(self.base, name)

    def split(self, local) -> tuple[dict, float, float]:
        params = self.box.local_to_params(np.asarray(local, float))
        card = {n: params[n] for n in self._card}
        return card, float(params["rext_ohm"]), float(params["heat_kappa"])

    def simulate(self, local):
        self.n_sim += 1
        card, rext, kappa = self.split(local)
        sims = self.sim_fn(card, rext)
        out = []
        for c, s in zip(self.base.curves, sims):
            s = np.asarray(s, float)
            out.append(s * heat_factor(np.asarray(c.Vd, float)[:len(s)], s, kappa, self.pmax)
                       if kappa != 0.0 else s)
        return out

    def curve_rrms(self, sims):
        return self.base.curve_rrms(sims)

    def residual_vector(self, sims, weights):
        return self.base.residual_vector(sims, weights)


# ---------------------------------------------------------------------------
# B3: shared polynomial correction
# ---------------------------------------------------------------------------
HYPER_POLY = {"l2": 1e-4, "lr": 3e-3, "steps": 2000}


def poly_exponents(degree: int) -> list[tuple[int, int]]:
    """(i, j) with i + j <= degree, ordered by total degree then -i."""
    return [(t - j, j) for t in range(degree + 1) for j in range(t + 1)]


def poly_features(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([u ** i * v ** j for i, j in poly_exponents(degree)])


@dataclass
class PolyCorrection(Correction):
    """g = OUT_SCALE * tanh(P_pol(u, v)); u, v = standardized |Vgs|/VDD,
    |Vds|/VDD; separate coefficients for nMOS (``coef[0]``) and pMOS
    (``coef[1]``)."""
    degree: int = 2
    coef: np.ndarray | None = None

    def g(self, phi, group_tag=None, group_kind=None):
        phi = np.asarray(phi, float)
        u = (phi[:, 1] - self.mu[0]) / self.sd[0]
        v = (phi[:, 2] - self.mu[1]) / self.sd[1]
        f = poly_features(u, v, self.degree)
        p = phi[:, 0]
        h = (1.0 - p) * (f @ self.coef[0]) + p * (f @ self.coef[1])
        return OUT_SCALE * np.tanh(h)

    def to_json(self) -> dict:
        return {"arm": self.arm, "kind": "poly", "degree": self.degree,
                "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.coef.tolist(), "out_scale": OUT_SCALE,
                "exponents": poly_exponents(self.degree)}

    @classmethod
    def from_json(cls, row: dict) -> "PolyCorrection":
        return cls(row["arm"], np.asarray(row["mu"], float), np.asarray(row["sd"], float),
                   degree=int(row["degree"]), coef=np.asarray(row["coef"], float))


def fit_poly(arm: str, degree: int, train: list[DeviceRec], hyper: dict | None = None,
             threads: int = 1) -> PolyCorrection:
    """E1 loss (mean paper per-device RRMS of I_sim * exp(g)) + l2 * sum of
    squared non-constant coefficients; Adam, zero initialization, float64."""
    import torch
    torch.set_num_threads(threads)
    hp = {**HYPER_POLY, **(hyper or {})}
    batch = make_batch(train)
    loss = _loss_fn(batch)
    x = batch.phi[:, [1, 2]]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    f = torch.tensor(poly_features((x[:, 0] - mu[0]) / sd[0], (x[:, 1] - mu[1]) / sd[1],
                                   degree), dtype=torch.float64)
    p = torch.tensor(batch.phi[:, 0], dtype=torch.float64)
    coef = torch.zeros((2, f.shape[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([coef], lr=hp["lr"])
    for _ in range(hp["steps"]):
        opt.zero_grad()
        h = (1.0 - p) * (f @ coef[0]) + p * (f @ coef[1])
        value = loss(OUT_SCALE * torch.tanh(h)) + hp["l2"] * (coef[:, 1:] ** 2).sum()
        value.backward()
        opt.step()
    return PolyCorrection(arm, mu, sd, degree=degree, coef=coef.detach().numpy().copy())


# ---------------------------------------------------------------------------
# B4: GP discrepancy (Kennedy-O'Hagan-type)
# ---------------------------------------------------------------------------
GP_POINTS = 2000           # total training points per fold (equal share per device)
GP_FLOOR = 1e-13           # |I| floor of the log residual


def gp_training_data(train: list[DeviceRec], total: int = GP_POINTS,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """x = (is_pmos, |Vgs|/VDD, |Vds|/VDD), y = ln|I_target| - ln|I_sim| at the
    included trimmed points with both magnitudes >= GP_FLOOR; per device a
    uniform random subset of floor(total / n_devices) points
    (``default_rng(seed)``, devices in the given order)."""
    rng = np.random.default_rng(seed)
    per = max(1, total // max(1, len(train)))
    xs, ys = [], []
    for dev in train:
        dx, dy = [], []
        for c in dev.curves:
            if not c.included or not (c.den > 0):
                continue
            s = slice(c.start, None)
            sim = np.abs(np.asarray(c.sim, float)[s])
            tgt = np.abs(clean_current(c.target)[s])
            ok = np.isfinite(sim) & np.isfinite(tgt) & (sim >= GP_FLOOR) & (tgt >= GP_FLOOR)
            n = len(sim)
            x = np.column_stack([np.full(n, 1.0 if dev.family == "pmos" else 0.0),
                                 np.abs(np.asarray(c.vg, float)[s]) / dev.vdd,
                                 np.abs(np.asarray(c.vd, float)[s]) / dev.vdd])
            with np.errstate(divide="ignore"):
                y = np.log(tgt) - np.log(sim)
            dx.append(x[ok]); dy.append(y[ok])
        dx, dy = np.concatenate(dx), np.concatenate(dy)
        if len(dy) > per:
            pick = np.sort(rng.choice(len(dy), size=per, replace=False))
            dx, dy = dx[pick], dy[pick]
        xs.append(dx); ys.append(dy)
    return np.concatenate(xs), np.concatenate(ys)


def _se_kernel(a, b, log_sf2, log_ell):
    import torch
    d = (a[:, None, :] - b[None, :, :]) / torch.exp(log_ell)
    return torch.exp(log_sf2) * torch.exp(-0.5 * (d ** 2).sum(-1))


@dataclass
class GPCorrection(Correction):
    """g(x) = m + k(x, X) alpha; k = sf2 * exp(-0.5 sum_j (dx_j / ell_j)^2),
    x = (is_pmos, |Vgs|/VDD, |Vds|/VDD) (unscaled)."""
    X: np.ndarray | None = None
    alpha: np.ndarray | None = None
    mean: float = 0.0
    log_sf2: float = 0.0
    log_ell: np.ndarray | None = None
    log_sn2: float = 0.0
    info: dict = field(default_factory=dict)

    def g(self, phi, group_tag=None, group_kind=None):
        x = np.asarray(phi, float)[:, :3]
        out = np.empty(len(x))
        ell = np.exp(self.log_ell)
        for i in range(0, len(x), 4096):
            d = (x[i:i + 4096, None, :] - self.X[None, :, :]) / ell
            k = math.exp(self.log_sf2) * np.exp(-0.5 * (d ** 2).sum(-1))
            out[i:i + 4096] = self.mean + k @ self.alpha
        return out

    def to_json(self) -> dict:
        return {"arm": self.arm, "kind": "gp", "X": self.X.tolist(),
                "alpha": self.alpha.tolist(), "mean": self.mean,
                "log_sf2": self.log_sf2, "log_ell": self.log_ell.tolist(),
                "log_sn2": self.log_sn2, "info": self.info,
                "kernel": "sf2 * exp(-0.5 * sum_j ((x_j - x'_j) / ell_j)^2) + sn2 * delta; "
                          "x = (is_pmos, |Vgs|/VDD, |Vds|/VDD); constant mean"}

    @classmethod
    def from_json(cls, row: dict) -> "GPCorrection":
        return cls(row["arm"], X=np.asarray(row["X"], float),
                   alpha=np.asarray(row["alpha"], float), mean=float(row["mean"]),
                   log_sf2=float(row["log_sf2"]), log_ell=np.asarray(row["log_ell"], float),
                   log_sn2=float(row["log_sn2"]), info=dict(row.get("info", {})))


def fit_gp(arm: str, train: list[DeviceRec], total: int = GP_POINTS, seed: int = 0,
           maxiter: int = 200, threads: int = 1) -> GPCorrection:
    """Type-II maximum likelihood (L-BFGS-B on the exact log marginal
    likelihood, torch gradients, float64) of (m, log sf2, log ell_1..3,
    log sn2); init m = mean y, sf2 = var y, ell = (1, 0.3, 0.3),
    sn2 = 0.1 var y; bounds log ell in [log 0.02, log 50],
    sf2 in [1e-4, 1e2] var y, sn2 in [1e-6, 10] var y."""
    import torch
    from scipy.optimize import minimize
    torch.set_num_threads(threads)
    x_np, y_np = gp_training_data(train, total, seed)
    var = float(np.var(y_np)) if np.var(y_np) > 0 else 1.0
    X = torch.tensor(x_np, dtype=torch.float64)
    Y = torch.tensor(y_np, dtype=torch.float64)
    n = len(y_np)
    eye = torch.eye(n, dtype=torch.float64)

    def nll_t(theta):
        m, lsf, l1, l2, l3, lsn = theta
        K = _se_kernel(X, X, lsf, torch.stack([l1, l2, l3])) \
            + (torch.exp(lsn) + 1e-8 * torch.exp(lsf)) * eye
        L = torch.linalg.cholesky(K)
        r = (Y - m)[:, None]
        a = torch.cholesky_solve(r, L)
        return 0.5 * (r * a).sum() + torch.log(torch.diagonal(L)).sum() \
            + 0.5 * n * math.log(2 * math.pi)

    def fun(v):
        t = torch.tensor(v, dtype=torch.float64, requires_grad=True)
        try:
            val = nll_t(t)
        except RuntimeError:          # Cholesky failure
            return 1e30, np.zeros_like(v)
        val.backward()
        return float(val.detach()), t.grad.numpy().copy()

    lv = math.log(var)
    x0 = np.array([float(np.mean(y_np)), lv, 0.0, math.log(0.3), math.log(0.3),
                   lv + math.log(0.1)])
    ell_b = (math.log(0.02), math.log(50.0))
    bounds = [(None, None), (lv + math.log(1e-4), lv + math.log(1e2)), ell_b, ell_b, ell_b,
              (lv + math.log(1e-6), lv + math.log(10.0))]
    res = minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter})
    th = torch.tensor(res.x, dtype=torch.float64)
    m, lsf, lsn = th[0], th[1], th[5]
    lell = th[2:5]
    K = _se_kernel(X, X, lsf, lell) + (torch.exp(lsn) + 1e-8 * torch.exp(lsf)) * eye
    L = torch.linalg.cholesky(K)
    alpha = torch.cholesky_solve((Y - m)[:, None], L)[:, 0]
    info = {"n_points": n, "nll": float(res.fun), "converged": bool(res.success),
            "n_iter": int(res.nit), "y_var": var,
            "ell": np.exp(res.x[2:5]).tolist(), "sf2": float(np.exp(res.x[1])),
            "sn2": float(np.exp(res.x[5])), "mean": float(res.x[0])}
    return GPCorrection(arm, X=x_np, alpha=alpha.numpy().copy(), mean=float(res.x[0]),
                        log_sf2=float(res.x[1]), log_ell=np.asarray(res.x[2:5], float),
                        log_sn2=float(res.x[5]), info=info)


def correction_from_json_any(row: dict) -> Correction:
    """PolyCorrection / GPCorrection / BiasCorrection / E1 Correction."""
    kind = row.get("kind")
    if kind == "poly":
        return PolyCorrection.from_json(row)
    if kind == "gp":
        return GPCorrection.from_json(row)
    from .i3_2_bias import correction_from_json
    return correction_from_json(row)


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------
RULE_BASELINES = ("B1", "B2", "B3_d2", "B3_d3", "B4")
RULE_SIM_RATIO = 0.95


def decide_rule(real_cmp: dict, gf_cmp: dict, ihp_cmp: dict) -> dict:
    """``*_cmp``: ``{baseline: paired_ratio(M_bias, baseline)}`` per dataset.

    Baseline X is beaten if the REAL_DEV CI upper bound of GMR(M_bias, X)
    is < 1, or GMR(M_bias, X) < 0.95 on both GF180 and IHP.  Verdict
    "supported against baselines" if every baseline in RULE_BASELINES is
    beaten, else "matched by baseline(s)" with the list."""
    per, matched = {}, []
    for b in RULE_BASELINES:
        r, g, i = real_cmp.get(b, {}), gf_cmp.get(b, {}), ihp_cmp.get(b, {})
        real_ok = bool(np.isfinite(r.get("ci_high", np.nan)) and r["ci_high"] < 1.0)
        sim_ok = bool(np.isfinite(g.get("ratio", np.nan)) and np.isfinite(i.get("ratio", np.nan))
                      and g["ratio"] < RULE_SIM_RATIO and i["ratio"] < RULE_SIM_RATIO)
        per[b] = {"real_ci_high": r.get("ci_high"), "gf180_ratio": g.get("ratio"),
                  "ihp_ratio": i.get("ratio"), "real_condition": real_ok,
                  "sim_condition": sim_ok, "beaten": real_ok or sim_ok}
        if not (real_ok or sim_ok):
            matched.append(b)
    return {"per_baseline": per, "matched_by": matched,
            "verdict": "supported against baselines" if not matched
            else "matched by baseline(s) " + ", ".join(matched)}


# ---------------------------------------------------------------------------
# I4-2 analysis helpers (stored corrections only)
# ---------------------------------------------------------------------------
def g_grid(corr: Correction, is_pmos: float, n: int = 41) -> np.ndarray:
    """g on an n x n grid of (|Vgs|/VDD, |Vds|/VDD) in [0, 1]^2;
    rows = |Vgs|, columns = |Vds|.  Features 3-5 of the E1 vector are set to
    0 (not read by bias-only corrections)."""
    a = np.linspace(0.0, 1.0, n)
    u, v = np.meshgrid(a, a, indexing="ij")
    phi = np.column_stack([np.full(u.size, float(is_pmos)), u.ravel(), v.ravel(),
                           np.zeros(u.size), np.zeros(u.size), np.zeros(u.size)])
    return corr.g(phi).reshape(n, n)


def linear_vth(rec: DeviceRec) -> float:
    """|Vth| by maximum-transconductance linear extrapolation on the measured
    Id-Vg curve with the smallest |Vds|: |Vg*| - |I*| / gm* - |Vds| / 2."""
    curves = [c for c in rec.curves if c.kind == "idvg"]
    if not curves:
        return float("nan")
    c = min(curves, key=lambda c: abs(float(c.key.split("@")[1])))
    vg = np.abs(np.asarray(c.vg, float))
    i = np.abs(clean_current(c.target))
    order = np.argsort(vg)
    vg, i = vg[order], i[order]
    gm = np.gradient(i, vg)
    k = int(np.nanargmax(gm))
    vds = abs(float(np.median(np.asarray(c.vd, float))))
    return float(vg[k] - i[k] / gm[k] - vds / 2.0)


def region_masks(vgs: np.ndarray, vds: np.ndarray, vth: float, vdd: float) -> dict:
    """Boolean masks on |Vgs|, |Vds|: subthreshold |Vgs| < Vth; saturation
    |Vgs| >= Vth and |Vds| >= |Vgs| - Vth; linear otherwise; high_vds
    |Vds| / VDD >= 0.6."""
    vgs, vds = np.abs(vgs), np.abs(vds)
    sub = vgs < vth
    sat = (~sub) & (vds >= vgs - vth)
    return {"subthreshold": sub, "saturation": sat, "linear": (~sub) & (~sat),
            "high_vds": vds / vdd >= 0.6}


def _pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def region_stats(g: np.ndarray, masks: dict, resid: np.ndarray | None = None) -> dict:
    """Mean g per region, point-biserial correlation of g with each region
    indicator, and Pearson(g, resid) when given."""
    out = {"n": int(len(g)), "mean_g": float(np.mean(g)) if len(g) else float("nan")}
    for k, m in masks.items():
        out[k] = {"n": int(m.sum()),
                  "mean_g": float(np.mean(g[m])) if m.any() else float("nan"),
                  "corr_indicator": _pearson(g, m.astype(float))}
    if resid is not None:
        out["pearson_g_resid"] = _pearson(g, resid)
    return out


def t1_comparison(rows: list[dict]) -> dict:
    """rows: {"group", "t1", "g_mean"}; Pearson, mean |diff|, per kind."""
    if not rows:
        return {}
    t1 = np.array([r["t1"] for r in rows], float)
    gm = np.array([r["g_mean"] for r in rows], float)
    out = {"n": len(rows), "pearson": _pearson(t1, gm),
           "mean_abs_diff": float(np.nanmean(np.abs(t1 - gm)))}
    for kind in ("idvd", "idvg"):
        sel = [k for k, r in enumerate(rows) if f"|{kind}@" in r["group"]]
        if sel:
            out[kind] = {"n": len(sel), "pearson": _pearson(t1[sel], gm[sel]),
                         "mean_abs_diff": float(np.nanmean(np.abs(t1[sel] - gm[sel])))}
    return out
