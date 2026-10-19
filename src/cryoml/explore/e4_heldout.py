"""E4: held-out bias-curve prediction and predictive intervals on measured devices.

A device's included metric curves (frozen published-baseline inclusion) are
split into TRAIN curves, used for fitting, and HELD-OUT curves, used only for
evaluation.  Splits (``split_tags``):

* ``vd``: train = every included Id-Vg curve + the included Id-Vd curve at the
  highest gate bias; held out = the other included Id-Vd curves.
* ``vg``: train = every included Id-Vd curve + Id-Vg at Vd = 0.01 and 1.85;
  held out = the other included Id-Vg curves.

Fit per device (``fit_device``), all NGSpice, inside a cold-start round box
(multiplier-one center, so no card fitted to the held-out curves enters):

1. CMA-ES on the mean train-curve RRMS from the box center (seed s0), plus
   ``n_cma_extra`` further CMA-ES runs with other seeds;
2. bounded TRF Gauss-Newton polish of each CMA result with batched NGSpice
   finite-difference Jacobians in local box coordinates;
3. ``n_perturb`` further starts: best polished card + N(0, perturb_sigma^2)
   on the active local coordinates, each polished the same way;
4. every polished card is re-simulated exactly (``simulate_pdk``) on all
   curves; these are the ensemble MEMBERS;
5. at the member with the lowest train objective (the SINGLE card): an
   all-curve Jacobian, its singular values, and a Gauss-Newton Laplace
   approximation with a N(0, prior_sd^2) prior in local coordinates;
   ``n_draws`` Laplace draws are simulated.

All predictions are stored as full simulated curves; evaluation, the
held-out metrics and the learned predictors are in ``e4_eval``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..card_eval import device_by_tag, include_tags
from ..cold_start import device_bins
from ..config import OUT_DIR
from ..data_io import load_device_curves
from ..metrics import curve_trim
from ..parameter_space import ParameterBox
from ..spice_batch import simulate_pdk_multi
from ..spice_pdk import simulate_pdk

PENALTY = 10.0
E4_DIR = OUT_DIR / "research" / "explore" / "E4"
SPLIT_JSON = OUT_DIR / "research" / "explore" / "real_split.json"
SPLITS = ("vd", "vg")


def real_split() -> dict:
    return json.loads(SPLIT_JSON.read_text())


def curve_key(curve) -> str:
    return f"{curve.kind}@{abs(float(curve.fixed)):g}"


def split_tags(include: set[str], split: str) -> tuple[list[str], list[str]]:
    """(train tags, held-out tags) of one device's included curves."""
    include = set(include)
    idvd = sorted((t for t in include if t.startswith("idvd")),
                  key=lambda t: float(t.split("@")[1]))
    idvg = sorted((t for t in include if t.startswith("idvg")),
                  key=lambda t: float(t.split("@")[1]))
    if split == "vd":
        train = idvg + idvd[-1:]
        test = idvd[:-1]
    elif split == "vg":
        anchors = [t for t in idvg if t in ("idvg@0.01", "idvg@1.85")]
        train = idvd + anchors
        test = [t for t in idvg if t not in anchors]
    else:
        raise ValueError(f"unknown split {split!r}")
    return train, test


def box_from_spec(spec: dict, tag: str) -> ParameterBox:
    names = tuple(spec["selected_parameters"])
    flags = spec["log_scale_by_device"][tag]
    return ParameterBox(names=names, published=spec["published_by_device"][tag],
                        bounds=spec["bounds_by_device"][tag],
                        log_scale=tuple(bool(flags[n]) for n in names))


@dataclass
class SplitProblem:
    tag: str
    box: ParameterBox
    split: str
    n_sim: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.device = device_by_tag(self.tag)
        self.curves = load_device_curves(self.device)
        self.bin_index = device_bins()[self.tag]
        self.train_tags, self.test_tags = split_tags(include_tags(self.tag),
                                                     self.split)
        self.rows = {}
        for index, curve in enumerate(self.curves):
            key = curve_key(curve)
            if key not in self.train_tags and key not in self.test_tags:
                continue
            start, cleaned, den = curve_trim(
                self.device.dev_type, self.device.L_um, self.device.W_um,
                curve.kind, curve.Id)
            self.rows[key] = {"index": index, "start": int(start),
                              "target": np.asarray(cleaned, dtype=np.float64),
                              "den": float(den)}
        self.train_rows = [self.rows[t] for t in self.train_tags]
        self.test_rows = [self.rows[t] for t in self.test_tags]
        self.active = np.flatnonzero(self.box.active)

    # -- simulation ---------------------------------------------------------
    def sim_one(self, local: np.ndarray) -> list[np.ndarray]:
        self.n_sim += 1
        return simulate_pdk(self.device.dev_type, self.device.L_um,
                            self.device.W_um, self.curves,
                            params=self.box.local_to_params(local),
                            bin_index=self.bin_index)

    def sim_many(self, locals_: list[np.ndarray], method: str = "parallel"):
        self.n_sim += len(locals_)
        cards = [self.box.local_to_params(np.asarray(x)) for x in locals_]
        return simulate_pdk_multi(self.device.dev_type, self.device.L_um,
                                  self.device.W_um, self.curves, cards,
                                  bin_index=self.bin_index, method=method)

    # -- train objective ----------------------------------------------------
    @staticmethod
    def _curve(sims, row) -> np.ndarray:
        return np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]

    def curve_rrms(self, sims, rows) -> np.ndarray:
        out = []
        for row in rows:
            sim = self._curve(sims, row)
            target = row["target"][row["start"]:]
            if sim.shape != target.shape or not np.all(np.isfinite(sim)):
                out.append(PENALTY)
                continue
            out.append(math.sqrt(float(np.mean((sim - target) ** 2))) / row["den"])
        return np.asarray(out)

    def objective(self, sims) -> float:
        return float(np.mean(self.curve_rrms(sims, self.train_rows)))

    def residual(self, sims) -> np.ndarray:
        """Squared norm = mean over train curves of squared curve RRMS."""
        scale = 1.0 / math.sqrt(len(self.train_rows))
        pieces = []
        for row in self.train_rows:
            sim = self._curve(sims, row)
            target = row["target"][row["start"]:]
            n = len(target)
            if sim.shape != target.shape or not np.all(np.isfinite(sim)):
                pieces.append(np.full(n, PENALTY * scale / math.sqrt(n)))
            else:
                pieces.append((sim - target) * scale / (row["den"] * math.sqrt(n)))
        return np.concatenate(pieces)

    def point_errors(self, sims, rows) -> np.ndarray:
        """Unweighted per-point relative errors (sim - meas) / den."""
        return np.concatenate([
            (self._curve(sims, r) - r["target"][r["start"]:]) / r["den"]
            for r in rows])

    # -- Jacobians (local coordinates, active parameters) ---------------------
    def fd_jacobian(self, local: np.ndarray, step: float = 0.01):
        """Forward differences of the full simulated curves.  Returns
        (base sims, list of perturbed sims, signed steps)."""
        local = np.asarray(local, dtype=np.float64)
        steps, cards = [], [local.copy()]
        for j in self.active:
            h = step if local[j] + step <= 1.0 else -step
            trial = local.copy()
            trial[j] += h
            cards.append(trial)
            steps.append(h)
        sims = self.sim_many(cards)
        return sims[0], sims[1:], np.asarray(steps)

    def jac_rows(self, base, perturbed, steps, rows, weighted: bool) -> np.ndarray:
        if weighted:
            f0 = self.residual(base)
            cols = [(self.residual(p) - f0) / h for p, h in zip(perturbed, steps)]
        else:
            f0 = self.point_errors(base, rows)
            cols = [(self.point_errors(p, rows) - f0) / h
                    for p, h in zip(perturbed, steps)]
        return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
def cma_fit(problem: SplitProblem, x0: np.ndarray, evals: int, seed: int,
            sigma0: float = 0.3) -> tuple[np.ndarray, float]:
    import cma
    act = problem.active
    base = np.asarray(x0, dtype=np.float64).copy()
    es = cma.CMAEvolutionStrategy(
        base[act], sigma0,
        {"bounds": [-1, 1], "maxfevals": evals, "seed": seed, "verbose": -9,
         "popsize": 4 + int(3 * np.log(len(act)))})
    best_x, best_f = base.copy(), math.inf
    while not es.stop():
        xs = es.ask()
        trials = []
        for x in xs:
            t = base.copy()
            t[act] = np.clip(x, -1, 1)
            trials.append(t)
        values = [problem.objective(s) for s in problem.sim_many(trials)]
        es.tell(xs, values)
        k = int(np.argmin(values))
        if values[k] < best_f:
            best_f, best_x = float(values[k]), trials[k].copy()
    return best_x, best_f


def gn_polish(problem: SplitProblem, x0: np.ndarray, max_nfev: int,
              step: float = 0.01) -> tuple[np.ndarray, float]:
    """Bounded TRF on the train residual with batched FD Jacobians."""
    act = problem.active
    local = np.asarray(x0, dtype=np.float64).copy()
    cache: dict = {}
    best = {"x": local.copy(), "f": math.inf}

    def full(values):
        t = local.copy()
        t[act] = values
        return t

    def fun(values):
        key = values.tobytes()
        if key not in cache:
            sims = problem.sim_one(full(values))
            f = problem.objective(sims)
            if f < best["f"]:
                best.update(x=full(values).copy(), f=f)
            cache.clear()
            cache[key] = problem.residual(sims)
        return cache[key]

    def jac(values):
        base, pert, steps = problem.fd_jacobian(full(values), step)
        f = problem.objective(base)
        if f < best["f"]:
            best.update(x=full(values).copy(), f=f)
        return problem.jac_rows(base, pert, steps, None, weighted=True)

    x = np.clip(local[act], -1 + 1e-9, 1 - 1e-9)
    try:
        least_squares(fun, x, jac=jac, method="trf",
                      bounds=(-np.ones(len(act)), np.ones(len(act))),
                      xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=max_nfev)
    except ValueError:
        pass
    return best["x"], best["f"]


# ---------------------------------------------------------------------------
# One device
# ---------------------------------------------------------------------------
def fit_device(tag: str, spec: dict, split: str, opts: dict) -> dict:
    t0 = time.time()
    problem = SplitProblem(tag, box_from_spec(spec, tag), split)
    rng = np.random.default_rng(opts["seed"] + hash_tag(tag, split))
    center = np.zeros(problem.box.n_parameters)
    starts = []
    for k in range(1 + opts["n_cma_extra"]):
        evals = opts["cma_evals"] if k == 0 else opts["cma_evals_extra"]
        x, f = cma_fit(problem, center, evals, opts["seed"] + 1000 * k,
                       opts["sigma0"])
        starts.append(("cma", k, x, f))
    calls_cma = problem.n_sim
    members = []
    for kind, k, x, _f in starts:
        xp, fp = gn_polish(problem, x, opts["gn_nfev"])
        members.append({"origin": f"{kind}{k}", "local": xp, "train_obj": fp})
    best = min(members, key=lambda m: m["train_obj"])
    for k in range(opts["n_perturb"]):
        x = best["local"].copy()
        x[problem.active] = np.clip(
            x[problem.active] + rng.normal(0, opts["perturb_sigma"],
                                           len(problem.active)), -1, 1)
        xp, fp = gn_polish(problem, x, opts["perturb_nfev"])
        members.append({"origin": f"perturb{k}", "local": xp, "train_obj": fp})
    calls_fit = problem.n_sim
    # Exact re-simulation of each member on all curves.
    member_sims = [problem.sim_one(m["local"]) for m in members]
    for m, sims in zip(members, member_sims):
        m["train_obj"] = problem.objective(sims)
        m["train_curve_rrms"] = problem.curve_rrms(sims, problem.train_rows).tolist()
        m["heldout_curve_rrms"] = problem.curve_rrms(sims, problem.test_rows).tolist()
        e = problem.point_errors(sims, problem.train_rows)
        m["train_sigma"] = float(np.sqrt(np.mean(e ** 2))) if np.all(np.isfinite(e)) else math.nan
    order = int(np.argmin([m["train_obj"] for m in members]))
    single = members[order]
    # Jacobian, identifiability and Laplace at the single card.
    base, pert, steps = problem.fd_jacobian(single["local"], opts["jac_step"])
    Jw = problem.jac_rows(base, pert, steps, None, weighted=True)
    Je_train = problem.jac_rows(base, pert, steps, problem.train_rows, False)
    Je_test = problem.jac_rows(base, pert, steps, problem.test_rows, False)
    ident = identifiability(Jw, Je_train, Je_test)
    sigma = single["train_sigma"]
    precision = Je_train.T @ Je_train / max(sigma, 1e-12) ** 2 \
        + np.eye(len(problem.active)) / opts["prior_sd"] ** 2
    cov = np.linalg.pinv(precision, hermitian=True)
    cov = 0.5 * (cov + cov.T)
    w, V = np.linalg.eigh(cov)
    chol = V * np.sqrt(np.maximum(w, 0.0))
    draws = []
    for _ in range(opts["n_draws"]):
        x = single["local"].copy()
        x[problem.active] = np.clip(
            x[problem.active] + chol @ rng.standard_normal(len(problem.active)),
            -1, 1)
        draws.append(x)
    draw_sims = problem.sim_many(draws) if draws else []
    post_sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    ident["laplace_post_sd_local"] = post_sd.tolist()
    ident["laplace_n_sd_gt_0p1"] = int(np.sum(post_sd > 0.1))
    near = [m for m in members if m["train_obj"] <= (1 + opts["near_tol"]) * single["train_obj"]]
    spread = np.std(np.stack([m["local"][problem.active] for m in near]), axis=0) \
        if len(near) > 1 else np.zeros(len(problem.active))
    ident["near_optimal_count"] = len(near)
    ident["near_optimal_spread_local"] = spread.tolist()
    ident["near_optimal_n_spread_gt_0p1"] = int(np.sum(spread > 0.1))
    names = [problem.box.names[j] for j in problem.active]
    return {
        "tag": tag, "split": split, "train_tags": problem.train_tags,
        "test_tags": problem.test_tags, "active_names": names,
        "members": [{**{k: v for k, v in m.items() if k != "local"},
                     "local": np.asarray(m["local"]).tolist()} for m in members],
        "single_index": order, "identifiability": ident,
        "n_draws": len(draws), "calls_cma": calls_cma, "calls_fit": calls_fit,
        "n_sim": problem.n_sim, "runtime_s": round(time.time() - t0, 1),
        "_member_sims": member_sims, "_draw_sims": draw_sims,
    }


def identifiability(Jw: np.ndarray, Je_train: np.ndarray,
                    Je_test: np.ndarray, rel: float = 1e-2) -> dict:
    """Singular values of the weighted train Jacobian, and the fraction of the
    held-out sensitivity that lies in its sloppy subspace (right singular
    vectors with s / s_max < rel)."""
    ok = np.all(np.isfinite(Jw)) and np.all(np.isfinite(Je_test))
    if not ok:
        return {"finite": False}
    _, s, Vt = np.linalg.svd(Jw, full_matrices=False)
    rel_s = s / s[0] if s[0] > 0 else s
    sloppy = Vt[rel_s < rel]
    proj = Je_test @ sloppy.T if len(sloppy) else np.zeros((Je_test.shape[0], 0))
    total = float(np.sum(Je_test ** 2))
    return {"finite": True, "singular_values": s.tolist(),
            "rel_singular_values": rel_s.tolist(),
            "eff_rank_1e-2": int(np.sum(rel_s >= 1e-2)),
            "eff_rank_1e-3": int(np.sum(rel_s >= 1e-3)),
            "condition": float(s[0] / s[-1]) if s[-1] > 0 else math.inf,
            "heldout_sens_in_sloppy_frac": float(np.sum(proj ** 2) / total) if total > 0 else math.nan}


def hash_tag(tag: str, split: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(f"{tag}|{split}")) % 100000


def save_record(record: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = record["tag"]
    member_sims = record.pop("_member_sims")
    draw_sims = record.pop("_draw_sims")

    def stack(sims_list):
        if not sims_list:
            return np.zeros((0, 0, 0))
        n_curves = len(sims_list[0])
        n_pts = max(len(c) for s in sims_list for c in s)
        arr = np.full((len(sims_list), n_curves, n_pts), np.nan)
        for i, s in enumerate(sims_list):
            for j, c in enumerate(s):
                c = np.asarray(c, dtype=np.float64)
                arr[i, j, :len(c)] = c
        return arr

    np.savez_compressed(out_dir / f"{tag}.npz", members=stack(member_sims),
                        draws=stack(draw_sims))
    (out_dir / f"{tag}.json").write_text(json.dumps(record, indent=1) + "\n")
