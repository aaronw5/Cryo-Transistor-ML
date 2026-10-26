"""I22-1: matched-budget baselines.

Pre-registration: ``out/research/explore/I22_1/prereg.md`` (``prereg.json``).

Part A reruns the I4-1 correction-stage baselines (B1 wider card set, B2
extrinsic resistance and self-heating, B3_d3 cubic polynomial g, B4 GP
discrepancy) under the selected g v2 protocol on the 9 measured DEV datasets.
This module holds the dataset-independent pieces: B1 screen selection, labels,
outcome rule and pooled statistics.

Part C compares the C0_refit least-squares refit with CMA-ES and TuRBO-1 at the
same NGSpice-call budget.  This module holds the best-so-far trace, the budget
counter, the CMA-ES loop (pycma) and the TuRBO-1 loop (BoTorch), written
against a plain objective ``f(x) -> float`` on the local box [-1, 1]^d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Datasets and rules (prereg)
# ---------------------------------------------------------------------------
DATASETS = ("sky130", "nist4", "nistrt", "ihp233", "ihp300", "ihp343", "ihp398", "tud4", "tud295")
NAMES = {"sky130": "sky130 77 K (18, LODO)", "nist4": "NIST sky130 4 K lvt (53)",
         "nistrt": "NIST sky130 4 K RT card (87)", "ihp233": "IHP SG13G2 233 K", "ihp300": "IHP SG13G2 300 K",
         "ihp343": "IHP SG13G2 343 K", "ihp398": "IHP SG13G2 398 K", "tud4": "TU Delft 4.13 K",
         "tud295": "TU Delft 295.0 K"}
FREEZE_KEYS = {"sky130": "sky130_77K", "nist4": "nist_4K", "nistrt": "nist_rt_card_4K", "ihp233": "ihp_233K",
               "ihp300": "ihp_300K", "ihp343": "ihp_343K", "ihp398": "ihp_398K", "tud4": "tudelft_4K",
               "tud295": "tudelft_295K"}
BASELINES = ("B1", "B2", "B3_d3", "B4")
GAIN_SET = ("sky130", "nist4", "nistrt")
NI_MARGIN = 1.05
B1_THRESHOLD = 1e-4
VA1_TOL = 1e-12
MAX_FAIL_FRACTION = 0.10
F_FAIL = 10.0                       # objective value of a non-finite simulation (Part C)


def screen_select(rows: Mapping[str, Mapping[str, dict]], threshold: float = B1_THRESHOLD) -> dict:
    """I4-1 edge-screen selection over per-device rows ``{tag: {name: entry}}``
    (entry: ``active`` and, when active, ``response_lo`` / ``response_hi``).
    A candidate is kept if its max over devices (active entries only; a
    non-finite response counts as 0) of max(response_lo, response_hi) is
    >= ``threshold``."""
    names = sorted({n for r in rows.values() for n in r})
    per = {}
    for n in names:
        resp = []
        for r in rows.values():
            e = r.get(n, {})
            if e.get("active"):
                lo, hi = e.get("response_lo", 0.0), e.get("response_hi", 0.0)
                lo = lo if lo is not None and np.isfinite(lo) else 0.0
                hi = hi if hi is not None and np.isfinite(hi) else 0.0
                resp.append(max(lo, hi))
        per[n] = {"max_response": float(max(resp)) if resp else 0.0, "n_active_devices": len(resp),
                  "n_devices_ge_threshold": int(sum(x >= threshold for x in resp))}
    selected = [n for n in names if per[n]["max_response"] >= threshold]
    return {"threshold": threshold, "candidates": names, "selected": selected, "per_parameter": per}


def screen_calls(rows: Mapping[str, Mapping[str, dict]]) -> dict:
    """NGSpice calls of a B1 screen: per device 1 reference + 2 per active
    candidate."""
    per = {t: 1 + 2 * sum(1 for e in r.values() if e.get("active")) for t, r in rows.items()}
    return {"per_device": per, "total": int(sum(per.values()))}


def label_vs_control(cmp: Mapping) -> str:
    """'gain' (CI upper < 1), 'worse' (CI lower > 1), else 'inconclusive'."""
    lo, hi = cmp.get("ci_low", float("nan")), cmp.get("ci_high", float("nan"))
    if np.isfinite(hi) and hi < 1.0:
        return "gain"
    if np.isfinite(lo) and lo > 1.0:
        return "worse"
    return "inconclusive"


def label_vs_reference(cmp: Mapping, x: str = "X", ref: str = "g v2") -> str:
    lo, hi = cmp.get("ci_low", float("nan")), cmp.get("ci_high", float("nan"))
    if np.isfinite(hi) and hi < 1.0:
        return f"{x} better than {ref}"
    if np.isfinite(lo) and lo > 1.0:
        return f"{ref} better than {x}"
    return "inconclusive"


def non_inferior(cmp: Mapping, margin: float = NI_MARGIN) -> bool:
    hi = cmp.get("ci_high", float("nan"))
    return bool(np.isfinite(hi) and hi < margin)


def outcome(vs_g: Mapping[str, Mapping[str, Mapping]], valid: Mapping[str, bool],
            baselines: Sequence[str] = BASELINES, gain_set: Sequence[str] = GAIN_SET) -> dict:
    """Pre-registered Part A outcome.

    ``vs_g[ds][X]``: paired_ratio(X, selected g v2) on dataset ``ds``;
    ``valid[ds]``: validity of the dataset.  X matches g v2 iff non-inferior
    (CI upper < 1.05) on every dataset of the gain set; g v2 is ahead of X
    iff CI lower > 1 on every dataset of the gain set.  An invalid dataset of
    the gain set makes the outcome 'not evaluated'."""
    bad = [d for d in gain_set if not valid.get(d, False)]
    per = {}
    for x in baselines:
        ni = {d: non_inferior(vs_g[d][x]) for d in gain_set if d in vs_g and x in vs_g[d]}
        ahead = {d: bool(np.isfinite(vs_g[d][x].get("ci_low", np.nan)) and vs_g[d][x]["ci_low"] > 1.0)
                 for d in gain_set if d in vs_g and x in vs_g[d]}
        per[x] = {"non_inferior": ni, "g_v2_better": ahead,
                  "matches_g_v2": len(ni) == len(gain_set) and all(ni.values()),
                  "g_v2_ahead": len(ahead) == len(gain_set) and all(ahead.values())}
    if bad:
        return {"label": "not evaluated", "invalid_gain_datasets": bad, "per_baseline": per, "matched_by": []}
    matched = [x for x in baselines if per[x]["matches_g_v2"]]
    label = ("selected g v2 not matched by a correction-stage baseline" if not matched
             else "matched by " + ", ".join(matched))
    return {"label": label, "matched_by": matched, "per_baseline": per, "invalid_gain_datasets": []}


def pooled_gmr(ratios: Sequence[float]) -> float:
    """exp(mean ln GMR), equal weight per dataset (finite values only)."""
    r = np.asarray([x for x in ratios if x is not None and np.isfinite(x) and x > 0], float)
    return float(np.exp(np.mean(np.log(r)))) if len(r) else float("nan")


def family_means(rows: Mapping[str, float], is_pmos: Callable[[str], bool]) -> dict:
    """all-device mean, combined (mean of nMOS / pMOS family means), n failed
    (non-finite)."""
    v = {t: float(x) for t, x in rows.items()}
    fin = {t: x for t, x in v.items() if np.isfinite(x)}
    n = [x for t, x in fin.items() if not is_pmos(t)]
    p = [x for t, x in fin.items() if is_pmos(t)]
    fam = [np.mean(a) for a in (n, p) if a]
    return {"all_device_mean": float(np.mean(list(fin.values()))) if fin else float("nan"),
            "combined": float(np.mean(fam)) if fam else float("nan"),
            "n": len(v), "n_failed": len(v) - len(fin)}


def max_abs_diff(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """max |a - b| over the tags of ``a`` (inf if a tag is missing in ``b`` or
    exactly one side is non-finite; two non-finite values count as equal)."""
    out = 0.0
    for t, x in a.items():
        if t not in b:
            return float("inf")
        y = b[t]
        fx, fy = np.isfinite(x), np.isfinite(y)
        if fx != fy:
            return float("inf")
        if fx:
            out = max(out, abs(float(x) - float(y)))
    return out


# ---------------------------------------------------------------------------
# Part C: traces and budgets
# ---------------------------------------------------------------------------
class BudgetExhausted(Exception):
    """Raised by ``Traced`` when the call budget is used up."""


@dataclass
class Traced:
    """Objective wrapper: counts calls, records f per call, stops at ``budget``.

    ``f(x)`` returns the device paper RRMS of local point ``x``; a non-finite
    value is replaced by ``F_FAIL``."""
    f: Callable[[np.ndarray], float]
    budget: int
    values: list = field(default_factory=list)
    xs: list = field(default_factory=list)
    best_x: np.ndarray | None = None
    best_f: float = float("inf")

    def __call__(self, x) -> float:
        if len(self.values) >= self.budget:
            raise BudgetExhausted
        x = np.asarray(x, float)
        v = float(self.f(x))
        if not np.isfinite(v):
            v = F_FAIL
        self.values.append(v)
        self.xs.append(x.copy())
        if v < self.best_f:
            self.best_f, self.best_x = v, x.copy()
        return v

    @property
    def n(self) -> int:
        return len(self.values)


def best_so_far(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, float)
    v = np.where(np.isfinite(v), v, F_FAIL)
    return np.minimum.accumulate(v) if len(v) else v


def value_at(values: Sequence[float], budget: int) -> float:
    """Best-so-far after ``budget`` calls (the last value if fewer calls)."""
    b = best_so_far(values)
    if not len(b):
        return float("nan")
    return float(b[min(int(budget), len(b)) - 1])


def seed_geomean(vals: Sequence[float]) -> float:
    v = np.asarray(vals, float)
    return float(np.exp(np.mean(np.log(v)))) if len(v) and np.all(v > 0) else float("nan")


def curve_on_grid(values: Sequence[float], f0: float, budget: int, grid: Sequence[float]) -> np.ndarray:
    """best-so-far f / f0 at calls = round(g * budget) for g in ``grid``
    (at least 1 call)."""
    b = best_so_far(values)
    out = []
    for g in grid:
        k = max(1, int(round(g * budget)))
        out.append(b[min(k, len(b)) - 1] / f0 if len(b) else np.nan)
    return np.asarray(out, float)


# ---------------------------------------------------------------------------
# Part C: CMA-ES
# ---------------------------------------------------------------------------
CMA_SIGMA0 = {"primary": 0.1, "secondary": 0.03}


def run_cma(f: Callable[[np.ndarray], float], x0: np.ndarray, budget: int, seed: int,
            sigma0: float = CMA_SIGMA0["primary"]) -> dict:
    """pycma on ln f in the box [-1, 1]^d; the start point is the first call;
    stops after exactly ``budget`` calls (mid-generation if needed)."""
    import cma
    tr = Traced(f, budget)
    x0 = np.clip(np.asarray(x0, float), -1.0, 1.0)
    try:
        tr(x0)
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"bounds": [-1.0, 1.0], "seed": int(seed), "verbose": -9,
                                                  "maxfevals": 10 ** 9, "tolfun": 0, "tolx": 0,
                                                  "tolfunhist": 0, "tolstagnation": 10 ** 9})
        while True:
            xs = es.ask()
            ys = []
            for x in xs:
                ys.append(math.log(max(tr(x), 1e-300)))
            es.tell(xs, ys)
    except BudgetExhausted:
        pass
    return {"values": tr.values, "xs": tr.xs, "best_x": tr.best_x, "best_f": tr.best_f, "n": tr.n}


def best_x_at(values: Sequence[float], xs: Sequence[np.ndarray], budget: int) -> tuple[np.ndarray, float]:
    """(x, f) of the lowest value among the first ``budget`` calls (first
    occurrence)."""
    v = np.asarray(values[:budget], float)
    v = np.where(np.isfinite(v), v, F_FAIL)
    k = int(np.argmin(v))
    return np.asarray(xs[k], float), float(v[k])


# ---------------------------------------------------------------------------
# Part C: TuRBO-1 (Eriksson et al. 2019; BoTorch tutorial form)
# ---------------------------------------------------------------------------
TURBO = {"q": 4, "n_init": 10, "n_candidates": 5000, "L_init": {"primary": 0.1, "secondary": 0.03},
         "L_min": 0.5 ** 7, "L_max": 1.6, "success_tol": 3}


@dataclass
class TurboState:
    dim: int
    q: int = TURBO["q"]
    length_init: float = TURBO["L_init"]["primary"]
    length: float = TURBO["L_init"]["primary"]
    length_min: float = TURBO["L_min"]
    length_max: float = TURBO["L_max"]
    success_tol: int = TURBO["success_tol"]
    failure_tol: int = 0
    success: int = 0
    failure: int = 0
    best: float = -float("inf")
    restarts: int = 0

    def __post_init__(self):
        self.failure_tol = int(math.ceil(max(4.0 / self.q, float(self.dim) / self.q)))
        self.length = self.length_init


def turbo_update(state: TurboState, y_next: Sequence[float]) -> TurboState:
    """TuRBO-1 trust-region update (maximization of y)."""
    y = float(np.max(y_next))
    if y > state.best + 1e-3 * abs(state.best):
        state.success += 1
        state.failure = 0
    else:
        state.success = 0
        state.failure += 1
    if state.success == state.success_tol:
        state.length = min(2.0 * state.length, state.length_max)
        state.success = 0
    elif state.failure == state.failure_tol:
        state.length /= 2.0
        state.failure = 0
    state.best = max(state.best, y)
    if state.length < state.length_min:          # prereg: restart at L_init around the best point
        state.length = state.length_init
        state.success = state.failure = 0
        state.restarts += 1
    return state


def tr_bounds(center: np.ndarray, lengthscale: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Trust region in [0, 1]^d: side length ``length`` scaled per dimension
    by lengthscale / geometric mean of lengthscales (TuRBO weights)."""
    ls = np.asarray(lengthscale, float).ravel()
    w = ls / ls.mean()
    w = w / np.prod(np.power(w, 1.0 / len(w)))
    lo = np.clip(center - w * length / 2.0, 0.0, 1.0)
    hi = np.clip(center + w * length / 2.0, 0.0, 1.0)
    return lo, hi


def run_turbo(f: Callable[[np.ndarray], float], x0: np.ndarray, budget: int, seed: int,
              length_init: float = TURBO["L_init"]["primary"], n_candidates: int = TURBO["n_candidates"],
              q: int = TURBO["q"], n_init: int = TURBO["n_init"], log: list | None = None) -> dict:
    """TuRBO-1 on y = -ln f over local x in [-1, 1]^d (scaled to u in [0, 1]^d).

    Initial design: the start point and ``n_init - 1`` scrambled Sobol points
    in the initial trust region (side ``length_init``) around it.  Then batches
    of ``q`` points by Thompson sampling over ``n_candidates`` perturbed Sobol
    candidates in the trust region (perturbation probability min(1, 20 / d)).
    Stops after exactly ``budget`` calls."""
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.quasirandom import SobolEngine

    dtype = torch.double
    torch.manual_seed(int(seed))
    x0 = np.clip(np.asarray(x0, float), -1.0, 1.0)
    d = len(x0)
    tr = Traced(f, budget)
    U, Y = [], []
    state = TurboState(dim=d, q=q, length_init=length_init)

    def evaluate(u: np.ndarray) -> float:
        y = -math.log(max(tr(2.0 * u - 1.0), 1e-300))
        U.append(np.asarray(u, float).copy())
        Y.append(y)
        return y

    u0 = (x0 + 1.0) / 2.0
    it = 0
    try:
        evaluate(u0)
        sob = SobolEngine(d, scramble=True, seed=int(seed) * 1000)
        pts = sob.draw(max(0, n_init - 1)).to(dtype).numpy()
        lo, hi = np.clip(u0 - length_init / 2.0, 0, 1), np.clip(u0 + length_init / 2.0, 0, 1)
        for p in pts:
            evaluate(lo + (hi - lo) * p)
        state.best = max(Y)
        prob_perturb = min(1.0, 20.0 / d)
        while True:
            it += 1
            X = torch.tensor(np.asarray(U), dtype=dtype)
            Yt = torch.tensor(np.asarray(Y), dtype=dtype).unsqueeze(-1)
            model = SingleTaskGP(X, Yt, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception:                         # keep the prior hyperparameters
                pass
            center = np.asarray(U[int(np.argmax(Y))])
            ls = model.covar_module.lengthscale.detach().cpu().numpy().ravel() \
                if hasattr(model.covar_module, "lengthscale") else \
                model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            lo, hi = tr_bounds(center, ls, state.length)
            sob = SobolEngine(d, scramble=True, seed=int(seed) * 1000 + it)
            pert = torch.tensor(lo, dtype=dtype) + (torch.tensor(hi - lo, dtype=dtype)) * sob.draw(n_candidates).to(dtype)
            mask = torch.rand(n_candidates, d, dtype=dtype) <= prob_perturb
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            if len(ind):
                mask[ind, torch.randint(0, d, size=(len(ind),))] = True
            cand = torch.tensor(center, dtype=dtype).expand(n_candidates, d).clone()
            cand[mask] = pert[mask]
            with torch.no_grad():
                ts = MaxPosteriorSampling(model=model, replacement=False)
                nxt = ts(cand, num_samples=q).cpu().numpy()
            ys = [evaluate(np.clip(u, 0.0, 1.0)) for u in nxt]
            turbo_update(state, ys)
            if log is not None:
                log.append({"it": it, "n": tr.n, "length": state.length, "best_f": tr.best_f})
    except BudgetExhausted:
        pass
    return {"values": tr.values, "xs": tr.xs, "best_x": tr.best_x, "best_f": tr.best_f, "n": tr.n,
            "restarts": state.restarts, "final_length": state.length, "iterations": it}
