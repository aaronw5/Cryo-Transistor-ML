"""I9-2: function-class ablation for the shared bias-only correction.

``I_corr = I_card * exp(g)``; ``g`` depends on (is_pmos, u = |Vgs|/VDD,
v = |Vds|/VDD) only.  The study compares classes of ``g`` at the I3-2
protocol (stored folds, training on the stored starting-card records with the
E1 loss, held-out card refit with ``g`` frozen in the loop, strict paper
RRMS).  Pre-registration draft: ``out/research/explore/I9_2/prereg_draft.json``.

Classes (``CLASS_SPECS``):

* ``const``, ``lin``, ``poly2`` .. ``poly5``: ``g = tanh(P_pol(u, v))`` with a
  polynomial of total degree 0 .. 5 in the standardized u, v, one coefficient
  vector per polarity (``i4_1_baselines.fit_poly``; ``poly2`` / ``poly3`` are
  the I4-1 baselines B3_d2 / B3_d3).
* ``gam``: additive spline with a low-rank interaction, per polarity,
  ``g = tanh(c + B(u) a + B(v) b + sum_r (B(u) p_r)(B(v) q_r))`` with a cubic
  B-spline basis ``B`` (``GAMCorrection``, ``fit_gam``).
* ``mlp8``, ``mlp16``, ``mlp32``, ``mlp64``: the I3-2 ``M_bias`` recipe
  (3-w-w-1 tanh MLP, 5 seeds, mean of members) with width w
  (``i3_2_bias.fit_bias``); ``mlp16`` is ``M_bias``.
* ``mlp16_s1``: ``M_bias`` with seed 0 only (member 0 of the stored ensemble,
  ``ensemble_member``).
* ``gp``: GP posterior mean of the log residual (I4-1 B4, ``fit_gp``).

Statistics: ``pooled_ratio`` (equal-weight pooled geometric-mean ratio over
datasets, stratified device bootstrap), ``label_class`` / ``decide`` (the
pre-registered rule), ``training_subsets`` (learning curve) and
``estimate_cpu`` (compute plan).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .e1_correction import OUT_SCALE, Correction, DeviceRec, _loss_fn, make_batch

# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------
SEEDS5 = (0, 1, 2, 3, 4)
GAM_BASIS = 8          # cubic B-spline basis functions per input
GAM_RANK = 1           # rank of the u-v interaction
HYPER_COMMON = {"l2": 1e-4, "lr": 3e-3, "steps": 2000}
GP_TOTAL = 2000

CLASS_SPECS: dict[str, dict] = {
    "const": {"kind": "poly", "degree": 0},
    "lin": {"kind": "poly", "degree": 1},
    "poly2": {"kind": "poly", "degree": 2},
    "poly3": {"kind": "poly", "degree": 3},
    "poly4": {"kind": "poly", "degree": 4},
    "poly5": {"kind": "poly", "degree": 5},
    "gam": {"kind": "gam", "n_basis": GAM_BASIS, "rank": GAM_RANK},
    "mlp8": {"kind": "mlp", "hidden": 8, "seeds": SEEDS5},
    "mlp16": {"kind": "mlp", "hidden": 16, "seeds": SEEDS5},
    "mlp16_s1": {"kind": "mlp", "hidden": 16, "seeds": (0,)},
    "mlp32": {"kind": "mlp", "hidden": 32, "seeds": SEEDS5},
    "mlp64": {"kind": "mlp", "hidden": 64, "seeds": SEEDS5},
    "gp": {"kind": "gp", "total": GP_TOTAL},
}
REFERENCE = "mlp16"                       # = stored I3-2 M_bias
CONTRASTS = tuple(c for c in CLASS_SPECS if c != REFERENCE)   # 12 primary contrasts


def n_params(name: str) -> int:
    """Number of trained values in the exported g (all ensemble members;
    intercepts included; GP: the alpha vector plus the 6 hyperparameters,
    the stored training inputs not counted)."""
    s = CLASS_SPECS[name]
    if s["kind"] == "poly":
        d = s["degree"]
        return 2 * (d + 1) * (d + 2) // 2
    if s["kind"] == "gam":
        k, r = s["n_basis"], s["rank"]
        return 2 * (1 + 2 * k + 2 * r * k)
    if s["kind"] == "mlp":
        w = s["hidden"]
        per = 3 * w + w + w * w + w + w + 1
        return per * len(s["seeds"])
    if s["kind"] == "gp":
        return s["total"] + 6
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Additive spline class
# ---------------------------------------------------------------------------
def bspline_basis(x: np.ndarray, lo: float, hi: float, n_basis: int,
                  degree: int = 3) -> np.ndarray:
    """(len(x), n_basis) cubic B-spline basis on [lo, hi] with uniform
    interior knots and clamped (repeated) end knots; x is clipped to
    [lo, hi] (constant extrapolation).  Rows sum to 1."""
    if n_basis < degree + 1:
        raise ValueError("n_basis must be >= degree + 1")
    x = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    n_inner = n_basis - degree - 1
    inner = np.linspace(lo, hi, n_inner + 2)[1:-1]
    t = np.concatenate([np.full(degree + 1, lo), inner, np.full(degree + 1, hi)])
    # degree-0 bases; the last non-empty interval is closed on the right
    B = np.zeros((len(x), len(t) - 1))
    for i in range(len(t) - 1):
        if t[i + 1] > t[i]:
            B[:, i] = (x >= t[i]) & (x < t[i + 1])
    last = max(i for i in range(len(t) - 1) if t[i + 1] > t[i])
    B[x >= hi, last] = 1.0
    for k in range(1, degree + 1):
        nb = np.zeros((len(x), len(t) - 1 - k))
        for i in range(len(t) - 1 - k):
            d1, d2 = t[i + k] - t[i], t[i + k + 1] - t[i + 1]
            a = (x - t[i]) / d1 * B[:, i] if d1 > 0 else 0.0
            b = (t[i + k + 1] - x) / d2 * B[:, i + 1] if d2 > 0 else 0.0
            nb[:, i] = a + b
        B = nb
    return B


@dataclass
class GAMCorrection(Correction):
    """g = OUT_SCALE * tanh(h_pol); h_pol = c + B(u) a + B(v) b
    + sum_r (B(u) p_r) (B(v) q_r); u, v raw |Vgs|/VDD, |Vds|/VDD clipped to
    the training range [lo, hi] per input; index 0 nMOS, 1 pMOS."""
    lo: np.ndarray | None = None          # (2,) for u, v
    hi: np.ndarray | None = None
    n_basis: int = GAM_BASIS
    rank: int = GAM_RANK
    c: np.ndarray | None = None           # (2,)
    a: np.ndarray | None = None           # (2, K)
    b: np.ndarray | None = None           # (2, K)
    p: np.ndarray | None = None           # (2, R, K)
    q: np.ndarray | None = None           # (2, R, K)
    info: dict = field(default_factory=dict)

    def bases(self, phi) -> tuple[np.ndarray, np.ndarray]:
        phi = np.asarray(phi, float)
        return (bspline_basis(phi[:, 1], self.lo[0], self.hi[0], self.n_basis),
                bspline_basis(phi[:, 2], self.lo[1], self.hi[1], self.n_basis))

    def h(self, phi) -> np.ndarray:
        phi = np.asarray(phi, float)
        bu, bv = self.bases(phi)
        out = np.zeros(len(phi))
        pol = phi[:, 0]
        for k in (0, 1):
            hk = self.c[k] + bu @ self.a[k] + bv @ self.b[k]
            hk = hk + np.sum((bu @ self.p[k].T) * (bv @ self.q[k].T), axis=1)
            out += (pol if k else 1.0 - pol) * hk
        return out

    def g(self, phi, group_tag=None, group_kind=None):
        return OUT_SCALE * np.tanh(self.h(phi))

    def to_json(self) -> dict:
        return {"arm": self.arm, "kind": "gam", "out_scale": OUT_SCALE,
                "lo": self.lo.tolist(), "hi": self.hi.tolist(), "n_basis": self.n_basis,
                "rank": self.rank, "c": self.c.tolist(), "a": self.a.tolist(),
                "b": self.b.tolist(), "p": self.p.tolist(), "q": self.q.tolist(),
                "info": self.info,
                "form": "tanh(c + B(u) a + B(v) b + sum_r (B(u) p_r)(B(v) q_r)), cubic "
                        "B-splines, uniform interior knots on [lo, hi], inputs clipped"}

    @classmethod
    def from_json(cls, row: dict) -> "GAMCorrection":
        f = lambda k: np.asarray(row[k], float)
        return cls(row["arm"], lo=f("lo"), hi=f("hi"), n_basis=int(row["n_basis"]),
                   rank=int(row["rank"]), c=f("c"), a=f("a"), b=f("b"), p=f("p"), q=f("q"),
                   info=dict(row.get("info", {})))


def fit_gam(arm: str, train: list[DeviceRec], hyper: dict | None = None,
            n_basis: int = GAM_BASIS, rank: int = GAM_RANK, seed: int = 0,
            threads: int = 1) -> GAMCorrection:
    """E1 loss + l2 * (|a|^2 + |b|^2 + |p|^2 + |q|^2) (c not penalized);
    Adam, float64; c, a, b, q start at 0, p at N(0, 0.1^2) (``torch``
    generator seeded with ``seed``) so the bilinear term has a gradient."""
    import torch
    torch.set_num_threads(threads)
    hp = {**HYPER_COMMON, **(hyper or {})}
    batch = make_batch(train)
    loss = _loss_fn(batch)
    lo = batch.phi[:, 1:3].min(axis=0)
    hi = batch.phi[:, 1:3].max(axis=0)
    hi = np.where(hi > lo, hi, lo + 1.0)
    bu = torch.tensor(bspline_basis(batch.phi[:, 1], lo[0], hi[0], n_basis), dtype=torch.float64)
    bv = torch.tensor(bspline_basis(batch.phi[:, 2], lo[1], hi[1], n_basis), dtype=torch.float64)
    pol = torch.tensor(batch.phi[:, 0], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    c = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    a = torch.zeros((2, n_basis), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((2, n_basis), dtype=torch.float64, requires_grad=True)
    p = (0.1 * torch.randn((2, rank, n_basis), generator=gen, dtype=torch.float64)).requires_grad_()
    q = torch.zeros((2, rank, n_basis), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([c, a, b, p, q], lr=hp["lr"])

    def h_of():
        hs = []
        for k in (0, 1):
            inter = ((bu @ p[k].T) * (bv @ q[k].T)).sum(dim=1)
            hs.append(c[k] + bu @ a[k] + bv @ b[k] + inter)
        return (1.0 - pol) * hs[0] + pol * hs[1]

    for _ in range(hp["steps"]):
        opt.zero_grad()
        pen = (a ** 2).sum() + (b ** 2).sum() + (p ** 2).sum() + (q ** 2).sum()
        value = loss(OUT_SCALE * torch.tanh(h_of())) + hp["l2"] * pen
        value.backward()
        opt.step()
    with torch.no_grad():
        final = float(loss(OUT_SCALE * torch.tanh(h_of())))
    d = lambda t: t.detach().numpy().copy()
    return GAMCorrection(arm, lo=lo, hi=hi, n_basis=n_basis, rank=rank, c=d(c), a=d(a),
                         b=d(b), p=d(p), q=d(q), info={"final_loss": final, **hp})


# ---------------------------------------------------------------------------
# Fitting dispatch, ensemble members, JSON
# ---------------------------------------------------------------------------
def fit_class(name: str, train: list[DeviceRec], overrides: dict | None = None,
              threads: int = 1) -> Correction:
    """Train class ``name`` on ``train`` with the pre-registered settings;
    ``overrides`` (smoke only): ``steps``, ``seeds``, ``gp_total``,
    ``gp_maxiter``."""
    ov = dict(overrides or {})
    s = CLASS_SPECS[name]
    hyper = {k: ov[k] for k in ("steps",) if k in ov}
    if s["kind"] == "poly":
        from .i4_1_baselines import fit_poly
        return fit_poly(name, s["degree"], train, hyper=hyper, threads=threads)
    if s["kind"] == "gam":
        return fit_gam(name, train, hyper=hyper, n_basis=s["n_basis"], rank=s["rank"],
                       threads=threads)
    if s["kind"] == "mlp":
        from .i3_2_bias import fit_bias
        seeds = tuple(ov.get("seeds", s["seeds"])) if len(s["seeds"]) > 1 else s["seeds"]
        corr = fit_bias("M_bias", train, hyper={**hyper, "hidden": s["hidden"], "seeds": seeds},
                        threads=threads)
        corr.arm = name
        return corr
    if s["kind"] == "gp":
        from .i4_1_baselines import fit_gp
        return fit_gp(name, train, total=int(ov.get("gp_total", s["total"])),
                      maxiter=int(ov.get("gp_maxiter", 200)), threads=threads)
    raise KeyError(name)


def ensemble_member(corr, k: int, arm: str | None = None):
    """``BiasCorrection`` holding member ``k`` of ``corr`` only (same feature
    scaling).  Members are trained independently from
    ``torch.manual_seed(seed)`` on the same batch (``i3_2_bias.fit_bias``),
    so member 0 of the 5-seed M_bias equals the 1-seed fit with seed 0."""
    from .i3_2_bias import BiasCorrection
    return BiasCorrection(arm or corr.arm, corr.mu, corr.sd, nets=[corr.nets[k]],
                          cols=tuple(corr.cols))


def correction_from_json_i9(row: dict) -> Correction:
    if row.get("kind") == "gam":
        return GAMCorrection.from_json(row)
    from .i4_1_baselines import correction_from_json_any
    return correction_from_json_any(row)


# ---------------------------------------------------------------------------
# Statistics and the decision rule
# ---------------------------------------------------------------------------
N_BOOT = 20000
ALPHA = 0.05
MARGIN = 1.03                # non-inferiority / equivalence margin on GMR(X, M_bias)
PER_DATASET_MAX = 1.05       # Q2: point GMR(X, M_bias) on every primary dataset
LC_TIE = 0.005               # learning-curve class choice: GMR tie tolerance


def paired_logs(method: dict, control: dict) -> tuple[np.ndarray, list, dict]:
    """ln(method / control) over shared tags with both values finite and > 0,
    the used tags, and failure counts."""
    tags = sorted(set(method) & set(control))
    a = np.array([method[t] for t in tags], float)
    b = np.array([control[t] for t in tags], float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    info = {"n": len(tags), "n_used": int(ok.sum()),
            "n_method_failed": int(np.sum(~np.isfinite(a))),
            "n_control_failed": int(np.sum(~np.isfinite(b))),
            "wins": int(np.sum(a[ok] < b[ok]))}
    return np.log(a[ok] / b[ok]), [t for t, k in zip(tags, ok) if k], info


def pooled_ratio(data: dict, clusters: dict | None = None, n_boot: int = N_BOOT,
                 seed: int = 0, alpha: float = ALPHA) -> dict:
    """Equal-weight pooled GMR over datasets.

    ``data``: {dataset: (method {tag: rrms}, control {tag: rrms})}.  Pooled
    log ratio L = mean over datasets of the mean paired log ratio.  CI:
    percentile (alpha/2, 1 - alpha/2) of L over ``n_boot`` stratified
    resamples (devices resampled within each dataset, ``default_rng(seed)``;
    with ``clusters[dataset]`` = {tag: cluster}, clusters are resampled for
    that dataset)."""
    rng = np.random.default_rng(seed)
    per, means, boots = {}, [], np.zeros(n_boot)
    for name in sorted(data):
        lr, used, info = paired_logs(*data[name])
        if not len(lr):
            raise ValueError(f"{name}: no finite pairs")
        cl = (clusters or {}).get(name)
        if cl is None:
            idx = rng.integers(0, len(lr), size=(n_boot, len(lr)))
            b = lr[idx].mean(axis=1)
        else:
            names = sorted({cl[t] for t in used})
            members = [np.array([i for i, t in enumerate(used) if cl[t] == c]) for c in names]
            sums = np.array([lr[m].sum() for m in members])
            sizes = np.array([len(m) for m in members], float)
            pick = rng.integers(0, len(names), size=(n_boot, len(names)))
            b = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
        per[name] = {**info, "ratio": float(np.exp(lr.mean())),
                     "ci_low": float(np.exp(np.percentile(b, 100 * alpha / 2))),
                     "ci_high": float(np.exp(np.percentile(b, 100 * (1 - alpha / 2))))}
        means.append(lr.mean())
        boots += b
    boots /= len(means)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ratio": float(np.exp(np.mean(means))), "ci_low": float(np.exp(lo)),
            "ci_high": float(np.exp(hi)), "alpha": alpha, "n_boot": n_boot,
            "n_datasets": len(means), "per_dataset": per}


def label_class(cmp: dict, margin: float = MARGIN) -> str:
    """Label of GMR(X, M_bias) from its (multiplicity-adjusted) CI:
    'beats' (upper < 1), 'worse' (lower > 1), 'equivalent' (CI inside
    [1/margin, margin]), 'non-inferior' (upper <= margin), else
    'inconclusive'."""
    lo, hi = cmp["ci_low"], cmp["ci_high"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "invalid"
    if hi < 1.0:
        return "beats"
    if lo > 1.0:
        return "worse"
    if lo >= 1.0 / margin and hi <= margin:
        return "equivalent"
    if hi <= margin:
        return "non-inferior"
    return "inconclusive"


def decide(pooled: dict, failures: dict, margin: float = MARGIN,
           per_dataset_max: float = PER_DATASET_MAX) -> dict:
    """Pre-registered rule.

    ``pooled``: {class: pooled_ratio(X, M_bias) at the Bonferroni level}
    over the primary datasets.  ``failures``: {class: {dataset: n_failed(X)
    - n_failed(M_bias)}}.

    Q1 'class-dependent' if some class is 'beats' or 'worse'; 'no class
    dependence detected' if every class is 'equivalent'; else
    'inconclusive'.  Q2: candidate classes are 'beats', 'equivalent' or
    'non-inferior', with point GMR(X, M_bias) <= per_dataset_max on every
    primary dataset and no more failed devices than M_bias on any dataset;
    the answer is the candidate with the fewest parameters (ties: lower
    pooled GMR), 'none' if there is no candidate."""
    labels = {c: label_class(v, margin) for c, v in pooled.items()}
    if any(lab in ("beats", "worse") for lab in labels.values()):
        q1 = "class-dependent"
    elif all(lab == "equivalent" for lab in labels.values()):
        q1 = "no class dependence detected"
    else:
        q1 = "inconclusive"
    cands = []
    for c, v in pooled.items():
        if labels[c] not in ("beats", "equivalent", "non-inferior"):
            continue
        if c != REFERENCE and n_params(c) >= n_params(REFERENCE):
            continue
        if any(d["ratio"] > per_dataset_max for d in v["per_dataset"].values()):
            continue
        if any(x > 0 for x in failures.get(c, {}).values()):
            continue
        cands.append(c)
    cands.sort(key=lambda c: (n_params(c), pooled[c]["ratio"]))
    q2 = cands[0] if cands else "none"
    return {"labels": labels, "Q1": q1, "Q2_smallest_matching_class": q2,
            "Q2_candidates": cands,
            "Q2_beats_with_fewer_parameters": bool(cands and labels[q2] == "beats"),
            "margin": margin, "per_dataset_max": per_dataset_max,
            "n_params": {c: n_params(c) for c in pooled}}


def bonferroni_alpha(n_contrasts: int = len(CONTRASTS), alpha: float = ALPHA) -> float:
    return alpha / n_contrasts


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------
LC_SIZES = (2, 4, 8)
LC_REPS = 3
LC_SEED = 20261021


def training_subsets(pool: dict, n: int, reps: int = LC_REPS, key: tuple = ()) -> list[list[str]]:
    """``reps`` training subsets of size ``n`` from ``pool`` {tag: family}.

    Polarity-stratified: n // 2 nMOS and n - n // 2 pMOS when both families
    have enough devices, otherwise the remainder from the other family.
    Draws use ``default_rng([LC_SEED, n, r, *key])`` on the sorted tags of
    each family; ``n >= len(pool)`` returns the whole pool once."""
    tags = sorted(pool)
    if n >= len(tags):
        return [tags]
    fams = {f: sorted(t for t in tags if pool[t] == f) for f in ("nmos", "pmos")}
    out = []
    for r in range(reps):
        rng = np.random.default_rng([LC_SEED, n, r, *[int(k) for k in key]])
        want = {"nmos": n // 2, "pmos": n - n // 2}
        for f, o in (("nmos", "pmos"), ("pmos", "nmos")):
            short = want[f] - len(fams[f])
            if short > 0:
                want[f] -= short
                want[o] += short
        pick = []
        for f in ("nmos", "pmos"):
            if want[f]:
                pick += list(rng.choice(fams[f], size=want[f], replace=False))
        out.append(sorted(pick))
    return out


def select_lc_classes(pooled_vs_c0: dict, k: int = 2, tol: float = LC_TIE) -> list[str]:
    """The ``k`` classes with the lowest pooled GMR(X, C0_refit); GMRs within
    ``tol`` of each other count as tied and the class with fewer parameters
    ranks first."""
    order = sorted(pooled_vs_c0, key=lambda c: pooled_vs_c0[c]["ratio"])
    out = []
    while order and len(out) < k:
        best = pooled_vs_c0[order[0]]["ratio"]
        tied = [c for c in order if pooled_vs_c0[c]["ratio"] <= best + tol]
        pick = min(tied, key=lambda c: (n_params(c), pooled_vs_c0[c]["ratio"]))
        out.append(pick)
        order.remove(pick)
    return out


# ---------------------------------------------------------------------------
# Compute plan
# ---------------------------------------------------------------------------
def estimate_cpu(n_refits: int, refit_s: float, train_tasks: int, train_s: float,
                 workers: int = 6, efficiency: float = 0.85) -> dict:
    """CPU seconds and wall minutes on ``workers`` processes."""
    cpu = n_refits * refit_s + train_tasks * train_s
    return {"cpu_s": round(cpu, 1), "cpu_h": round(cpu / 3600, 2),
            "wall_min": round(cpu / (workers * efficiency) / 60, 1)}


def seeds_of(name: str) -> int:
    s = CLASS_SPECS[name]
    return len(s["seeds"]) if s["kind"] == "mlp" else 1

