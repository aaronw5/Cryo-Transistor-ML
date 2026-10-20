"""I7-1: a training-set statistic that predicts the held-out gain of the
shared bias-only correction (I3-2 ``M_bias``).

Pre-registration: ``out/research/explore/I7_1/prereg.json``.

Residual of a training device ``d`` at an included, trimmed bias point ``p``
of its starting-card record (``e1_correction.DeviceRec``)::

    r_p = clip(ln((|I_target,p| + I_EPS) / (|I_sim,p| + I_EPS)), -1, 1)
    w_p = (|I_target,p| / den_c)^2 / (n_c * C_d)

``den_c`` is the curve's paper denominator, ``n_c`` its number of trimmed
points and ``C_d`` the device's number of included curves, so that
``sum_p w_p (r_p - g_p)^2`` is the quadratic (first-order) form of the
M_bias training loss on device ``d`` (mean over curves of trimmed RMSE /
den).  The clip is the range of ``g = tanh(.)``.

Common normalized bias grid: square bins of width ``BIN_WIDTH`` in
(|Vgs|/VDD, |Vds|/VDD), index ``min(floor(x / BIN_WIDTH), N_BINS - 1)``,
separately per polarity.  A shared function ``m`` on a device set is the
weighted bin mean of the residuals of the devices of that polarity
(``shared_function``); empty bins give ``m = 0``.

Statistics on a set of training devices ``D`` (no held-out information):

* ``S1_lodo`` (primary): ``1 - sum_d E_d(m_{D minus d}) / sum_d E_d(0)``,
  ``E_d(m) = sum_p w_p (r_p - m(bin_p))^2``: the leave-one-device-out
  fraction of residual energy explained by the shared function.
* ``S1_pooled``: the same with ``m_D`` (in-sample).
* ``S2``: mean over same-polarity device pairs of the weighted uncentered
  cosine between the two devices' bin-mean residual vectors on their common
  bins (bin weight ``sqrt(W_d(b) W_e(b))``).
* ``S3`` (``s3_inner``): inner leave-one-group-out post-hoc gain of the
  M_bias recipe on the training devices (reduced recipe ``S3_HYPER``):
  mean over devices of ln(RRMS of the corrected starting simulation /
  RRMS of the starting simulation).

Also: Spearman rank correlation, the linear rule, truth-error injection
(random Fourier feature functions) for the prospective simulated datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..metrics import clean_current
from .e1_correction import I_EPS, DeviceRec, score_device

BIN_WIDTH = 0.1
N_BINS = 12            # x >= 1.1 falls in the last bin
R_CLIP = 1.0
S3_HYPER = {"seeds": (0,), "steps": 500}


# ---------------------------------------------------------------------------
# Residual sufficient statistics
# ---------------------------------------------------------------------------
@dataclass
class DevResid:
    tag: str
    pmos: bool
    group: str
    bins: np.ndarray       # (P,) flat bin index b1 * N_BINS + b2
    w: np.ndarray          # (P,)
    r: np.ndarray          # (P,)
    W: np.ndarray = field(init=False)   # (N_BINS^2,) sum w
    R: np.ndarray = field(init=False)   # sum w r
    Q: np.ndarray = field(init=False)   # sum w r^2

    def __post_init__(self) -> None:
        n = N_BINS * N_BINS
        self.W = np.bincount(self.bins, self.w, minlength=n)
        self.R = np.bincount(self.bins, self.w * self.r, minlength=n)
        self.Q = np.bincount(self.bins, self.w * self.r ** 2, minlength=n)

    def energy(self, m: np.ndarray | None = None) -> float:
        """sum_p w_p (r_p - m(bin_p))^2 (``m`` None: zero function)."""
        if m is None:
            return float(self.Q.sum())
        return float(np.sum(self.Q - 2.0 * m * self.R + m ** 2 * self.W))


def bin_index(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.floor(np.abs(x) / BIN_WIDTH).astype(int), N_BINS - 1)


def device_residual(dev: DeviceRec, group: str | None = None) -> DevResid:
    """Residual sufficient statistics of one starting-card record."""
    bins, ws, rs = [], [], []
    n_curves = 0
    for c in dev.curves:
        if not c.included or not (c.den > 0):
            continue
        s = slice(c.start, None)
        t = clean_current(c.target)[s]
        sim = np.asarray(c.sim, dtype=np.float64)[s]
        vg = np.asarray(c.vg, dtype=np.float64)[s]
        vd = np.asarray(c.vd, dtype=np.float64)[s]
        n = len(t)
        if n == 0:
            continue
        n_curves += 1
        ok = np.isfinite(t) & np.isfinite(sim)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log((np.abs(t[ok]) + I_EPS) / (np.abs(sim[ok]) + I_EPS))
        rs.append(np.clip(r, -R_CLIP, R_CLIP))
        ws.append((np.abs(t[ok]) / c.den) ** 2 / n)
        bins.append(bin_index(vg[ok] / dev.vdd) * N_BINS + bin_index(vd[ok] / dev.vdd))
    if not n_curves:
        raise ValueError(f"{dev.tag}: no included curve")
    w = np.concatenate(ws) / n_curves
    return DevResid(dev.tag, dev.family == "pmos", group if group is not None else dev.tag,
                    np.concatenate(bins), w, np.concatenate(rs))


def shared_function(devs: list[DevResid], pmos: bool) -> np.ndarray:
    """Weighted bin mean of the residuals of the devices of one polarity."""
    W = np.zeros(N_BINS * N_BINS)
    R = np.zeros(N_BINS * N_BINS)
    for d in devs:
        if d.pmos == pmos:
            W += d.W
            R += d.R
    return np.divide(R, W, out=np.zeros_like(R), where=W > 0)


def s1(devs: list[DevResid], lodo: bool = True) -> float:
    """Fraction of residual energy explained by the shared function
    (``lodo``: estimated without the scored device)."""
    num = den = 0.0
    pooled = {p: shared_function(devs, p) for p in (False, True)}
    for i, d in enumerate(devs):
        m = shared_function(devs[:i] + devs[i + 1:], d.pmos) if lodo else pooled[d.pmos]
        num += d.energy(m)
        den += d.energy()
    return float(1.0 - num / den) if den > 0 else float("nan")


def s2(devs: list[DevResid]) -> float:
    """Mean weighted uncentered cosine of same-polarity device pairs."""
    vals = []
    for a, b in combinations(devs, 2):
        if a.pmos != b.pmos:
            continue
        common = (a.W > 0) & (b.W > 0)
        if not common.any():
            continue
        u = np.sqrt(a.W[common] * b.W[common])
        xa = a.R[common] / a.W[common]
        xb = b.R[common] / b.W[common]
        den = math.sqrt(float(np.sum(u * xa ** 2)) * float(np.sum(u * xb ** 2)))
        if den > 0:
            vals.append(float(np.sum(u * xa * xb)) / den)
    return float(np.mean(vals)) if vals else float("nan")


def stats_for(train: list[DeviceRec], group_of=None) -> dict:
    """S1_lodo, S1_pooled, S2 of one training set."""
    devs = [device_residual(d, group_of(d) if group_of else None) for d in train]
    return {"S1_lodo": s1(devs, True), "S1_pooled": s1(devs, False), "S2": s2(devs),
            "n_train": len(devs), "n_train_pmos": int(sum(d.pmos for d in devs))}


# ---------------------------------------------------------------------------
# S3: inner leave-one-group-out post-hoc gain (reduced M_bias recipe)
# ---------------------------------------------------------------------------
def s3_inner_tasks(train: list[DeviceRec], group_of) -> list[tuple]:
    """(inner group, inner-training records, scored records) per inner fold."""
    groups = sorted({group_of(d) for d in train})
    return [(g, [d for d in train if group_of(d) != g], [d for d in train if group_of(d) == g])
            for g in groups]


def s3_fold_value(corr, scored: list[DeviceRec]) -> list[float]:
    """ln(RRMS corrected / RRMS start) per scored record (post-hoc, no refit)."""
    out = []
    for d in scored:
        before = score_device(d, [c.sim for c in d.curves])["rrms"]
        after = score_device(d, corr.apply(d))["rrms"]
        ok = np.isfinite(before) and np.isfinite(after) and before > 0 and after > 0
        out.append(float(math.log(after / before)) if ok else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Outcomes, rank correlation, rule
# ---------------------------------------------------------------------------
def log_ratios(method: dict, control: dict, tags) -> list[float]:
    out = []
    for t in tags:
        a, b = method.get(t, np.nan), control.get(t, np.nan)
        ok = np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0
        out.append(float(math.log(a / b)) if ok else float("nan"))
    return out


def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float(np.sum(rx ** 2) * np.sum(ry ** 2)))
    return float(np.sum(rx * ry) / den) if den > 0 else float("nan")


def spearman_perm_p(x, y, n_perm: int = 10000, seed: int = 0) -> float:
    """One-sided permutation p-value for a negative Spearman correlation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    obs = spearman(x, y)
    rng = np.random.default_rng(seed)
    hits = sum(spearman(x, rng.permutation(y)) <= obs for _ in range(n_perm))
    return float((hits + 1) / (n_perm + 1))


def ols(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    A = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"a": float(a), "b": float(b), "n": int(len(x)),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "threshold_S": float(-a / b) if b != 0 else float("nan")}


def rule_predict(rule: dict, s: float) -> float:
    """Predicted log GMR(M_bias, C0_refit) of the linear rule."""
    return rule["a"] + rule["b"] * s


def sign_accuracy(rule: dict, s, y) -> dict:
    s, y = np.asarray(s, float), np.asarray(y, float)
    pred = np.array([rule_predict(rule, v) for v in s])
    ok = np.isfinite(pred) & np.isfinite(y)
    hit = (np.sign(pred[ok]) == np.sign(y[ok])) & (y[ok] != 0)
    return {"n": int(ok.sum()), "correct": int(hit.sum()),
            "fraction": float(hit.mean()) if ok.any() else float("nan")}


# ---------------------------------------------------------------------------
# Prospective simulated data: shared + device-specific multiplicative error
# ---------------------------------------------------------------------------
RFF_K = 16
RFF_LENGTH = 0.35
REF_GRID = 21


@dataclass
class RandomFunction:
    """f(x1, x2) = c * sum_k a_k cos(omega_k . x + beta_k), c such that the
    RMS over the REF_GRID x REF_GRID grid of [0, 1]^2 is 1."""
    omega: np.ndarray      # (K, 2)
    a: np.ndarray          # (K,)
    beta: np.ndarray       # (K,)
    scale: float = 1.0

    @classmethod
    def draw(cls, seed: int, k: int = RFF_K, length: float = RFF_LENGTH) -> "RandomFunction":
        rng = np.random.default_rng(seed)
        f = cls(rng.normal(0.0, 1.0 / length, size=(k, 2)), rng.normal(size=k),
                rng.uniform(0.0, 2.0 * math.pi, size=k))
        g = np.linspace(0.0, 1.0, REF_GRID)
        x1, x2 = np.meshgrid(g, g, indexing="ij")
        f.scale = 1.0 / math.sqrt(float(np.mean(f(x1.ravel(), x2.ravel()) ** 2)))
        return f

    def __call__(self, x1, x2) -> np.ndarray:
        x = np.column_stack([np.ravel(x1), np.ravel(x2)])
        return self.scale * (np.cos(x @ self.omega.T + self.beta) @ self.a)

    def to_json(self) -> dict:
        return {"omega": self.omega.tolist(), "a": self.a.tolist(),
                "beta": self.beta.tolist(), "scale": self.scale}


def injected_log_error(x1, x2, lam: float, amp: float, shared: RandomFunction,
                       specific: RandomFunction) -> np.ndarray:
    """e = amp * (sqrt(lam) * shared + sqrt(1 - lam) * specific)."""
    return amp * (math.sqrt(lam) * shared(x1, x2) + math.sqrt(1.0 - lam) * specific(x1, x2))
