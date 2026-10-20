"""I3-1 hybrid temperature law (pre-registration out/research/explore/I3_1/prereg.json).

For a device with a per-device par3 law fit (mu, c, lg) on the O4 anchors,

    x_hyb(T) = law_x(mu, c, lg, T) + sigma_h * mean_e NN_e(features)

where the network is shared across devices and trained on densely
characterised devices (per-temperature fits x_hat(T), Gauss-Newton matrices
H(T)) of the same generator.  The output layer is initialised to zero, the
output is penalised toward zero and early stopping may return the
zero-correction state, so the hybrid can reproduce par3.

Inputs (addendum out/research/explore/I3_1/prereg_addendum.json): each
ensemble member trains two networks on the same split and seed, input set
``geo_t`` (geometry, temperature feature) and ``geo_t_res`` (geometry, par3
residuals at the O4 anchors x_anchor - law_x(T_anchor), temperature
feature), and keeps the one with the lower validation criterion.

* ``HybridLaw``: the network version (``hyb``);
* ``HybridMeanLaw``: par3 plus the population-mean residual per device type
  and temperature (``hyb_mean``, no network);
* ``gmr``: paired geometric-mean ratio with eps shift and bootstrap CI;
* ``decide``: the pre-registered I3-1 rule on the three tier statistics.
"""

from __future__ import annotations

import numpy as np

from ..temperature import temperature_feature
from .e3_temp import N_PAR, law_x

SEED = 20261019
EPS = 0.01
EPS_SENSITIVITY = 0.001


def law_theta(law: dict) -> np.ndarray:
    return np.concatenate([np.asarray(law[k], dtype=np.float64) for k in ("mu", "c", "lg")])


def law_base(law: dict, temp_K: float) -> np.ndarray:
    return law_x(np.asarray(law["mu"], dtype=np.float64), np.asarray(law["c"], dtype=np.float64),
                 np.asarray(law["lg"], dtype=np.float64), temp_K)


FEATURE_SETS = ("geo_t", "geo_t_res")


def anchor_residuals(anchor_x, law, anchors) -> np.ndarray:
    """par3 residuals at the anchors, x_anchor - law_x(T_anchor), flattened."""
    return (np.asarray(anchor_x, dtype=np.float64)
            - np.array([law_base(law, t) for t in anchors])).ravel()


def hybrid_rows(records, laws, anchors, temps) -> list[dict]:
    """One row per (training device, dense temperature) with finite H and
    finite fit RRMS; anchors are included."""
    rows = []
    for d, (rec, law) in enumerate(zip(records, laws)):
        ax = np.array([rec["fits"][f"{t:g}"]["x"] for t in anchors])
        theta = law_theta(law)
        res = anchor_residuals(ax, law, anchors)
        for t in temps:
            fit = rec["fits"][f"{t:g}"]
            H = np.asarray(fit["H"], dtype=np.float64)
            if not np.all(np.isfinite(H)) or not np.isfinite(fit["rrms"]):
                continue
            rows.append({"geometry": np.asarray(rec["geometry"], dtype=np.float64),
                         "anchor_x": ax, "theta": theta, "anchor_res": res,
                         "temp": float(t),
                         "target": np.asarray(fit["x"], dtype=np.float64), "H": H,
                         "base": law_base(law, t), "device": d})
    return rows


def residual_sigma(rows) -> np.ndarray:
    resid = np.array([r["target"] - r["base"] for r in rows])
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid, 0)), 0)
    return np.maximum(mad, 0.02)


class HybridMeanLaw:
    """par3 plus the population-mean residual (x_hat - base) per device type
    (geometry[0]) and dense temperature, interpolated linearly in T."""

    def __init__(self, records, laws, anchors, temps):
        rows = hybrid_rows(records, laws, anchors, temps)
        self.sigma = residual_sigma(rows)
        self.n_rows = len(rows)
        self.table = {}
        for kind in sorted({float(r["geometry"][0]) for r in rows}):
            sel = [r for r in rows if float(r["geometry"][0]) == kind]
            grid = sorted({r["temp"] for r in sel})
            self.table[kind] = (np.array(grid), np.array([
                np.mean([r["target"] - r["base"] for r in sel if r["temp"] == t], 0)
                for t in grid]))

    def predict(self, geometry, anchor_x, law, temps) -> np.ndarray:
        grid, values = self.table[float(geometry[0])]
        return np.array([law_base(law, t) + np.array([np.interp(t, grid, values[:, k])
                                                      for k in range(N_PAR)])
                         for t in temps])


class HybridLaw:
    """MLP ensemble for the residual of the per-device par3 law."""

    def __init__(self, records, laws, anchors, temps, seeds=5, width=64, steps=3000,
                 lr=3e-3, weight_decay=1e-4, ridge=0.1, shrink=0.01,
                 feature_sets=FEATURE_SETS):
        import torch
        self.anchors = tuple(anchors)
        rows = hybrid_rows(records, laws, anchors, temps)
        self.n_rows = len(rows)
        self.sigma = residual_sigma(rows)
        res = np.array([r["anchor_res"] for r in rows])
        self.r_mean, self.r_std = res.mean(0), np.maximum(res.std(0), 1e-3)
        Xs = {fs: torch.tensor(self._features(fs, [r["geometry"] for r in rows],
                                              [r["anchor_res"] for r in rows],
                                              [r["temp"] for r in rows]),
                               dtype=torch.float32) for fs in feature_sets}
        target = torch.tensor(np.array([r["target"] for r in rows]), dtype=torch.float32)
        base = torch.tensor(np.array([r["base"] for r in rows]), dtype=torch.float32)
        Hn = torch.tensor(np.array([7.0 * r["H"] / max(np.trace(r["H"]), 1e-30)
                                    for r in rows]), dtype=torch.float32)
        sig = torch.tensor(self.sigma, dtype=torch.float32)
        device_of = np.array([r["device"] for r in rows])
        self.models, self.history = [], []   # models: (feature set, network)

        def objective(model, X, idx, penalty):
            out = model(X[idx])
            d = base[idx] + sig * out - target[idx]
            value = (torch.mean(torch.einsum("ni,nij,nj->n", d, Hn[idx], d))
                     + ridge * torch.mean((d / sig) ** 2))
            if penalty:
                value = value + shrink * torch.mean(out ** 2)
            return value

        for seed in range(seeds):
            rng = np.random.default_rng(SEED + seed)
            devices = np.unique(device_of)
            val_dev = rng.choice(devices, max(1, len(devices) // 5), replace=False)
            is_val = np.isin(device_of, val_dev)
            tr = torch.tensor(np.flatnonzero(~is_val))
            va = torch.tensor(np.flatnonzero(is_val))
            candidates = []
            for fs in feature_sets:
                X = Xs[fs]
                torch.manual_seed(SEED + seed)
                model = torch.nn.Sequential(
                    torch.nn.Linear(X.shape[1], width), torch.nn.SiLU(),
                    torch.nn.Linear(width, width), torch.nn.SiLU(),
                    torch.nn.Linear(width, N_PAR))
                torch.nn.init.zeros_(model[-1].weight)
                torch.nn.init.zeros_(model[-1].bias)
                opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                        weight_decay=weight_decay)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
                with torch.no_grad():
                    best = float(objective(model, X, va, False))
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                best_step = -1
                for step in range(steps):
                    opt.zero_grad()
                    value = objective(model, X, tr, True)
                    value.backward()
                    opt.step()
                    sched.step()
                    if step % 25 == 0 or step == steps - 1:
                        with torch.no_grad():
                            v = float(objective(model, X, va, False))
                        if v < best:
                            best, best_step = v, step
                            best_state = {k: t.detach().clone()
                                          for k, t in model.state_dict().items()}
                model.load_state_dict(best_state)
                model.eval()
                candidates.append((best, fs, best_step, model))
            best, fs, best_step, model = min(candidates, key=lambda c: c[0])
            self.models.append((fs, model))
            self.history.append({"best_val": best, "best_step": best_step,
                                 "feature_set": fs,
                                 "candidates": {c[1]: {"best_val": c[0], "best_step": c[2]}
                                                for c in candidates}})

    def _features(self, feature_set, geometries, anchor_res, temps):
        tf = np.array([[temperature_feature(t)] for t in temps])
        if feature_set == "geo_t":
            return np.concatenate([np.array(geometries), tf], 1)
        if feature_set == "geo_t_res":
            r = (np.array(anchor_res) - self.r_mean) / self.r_std
            return np.concatenate([np.array(geometries), r, tf], 1)
        raise ValueError(feature_set)

    def correction(self, geometry, anchor_x, law, temps) -> np.ndarray:
        import torch
        n = len(temps)
        res = anchor_residuals(anchor_x, law, self.anchors)
        outs = []
        with torch.no_grad():
            for fs, model in self.models:
                X = torch.tensor(self._features(fs, [geometry] * n, [res] * n, list(temps)),
                                 dtype=torch.float32)
                outs.append(model(X).numpy())
        return np.mean(outs, 0).astype(np.float64) * self.sigma

    def predict(self, geometry, anchor_x, law, temps) -> np.ndarray:
        corr = self.correction(geometry, anchor_x, law, temps)
        return np.array([law_base(law, t) + c for t, c in zip(temps, corr)])


# ------------------------------------------------------------ statistics

def gmr(a, b, eps=EPS, n=4000, seed=0) -> dict:
    """Paired geometric-mean ratio exp(mean log((a + eps) / (b + eps))) over
    devices finite for both, percentile bootstrap 95% CI, wins = #(a < b)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return {"gmr": float("nan"), "ci95": [float("nan")] * 2, "wins": 0,
                "n_devices": 0, "eps": eps}
    r = np.log((a + eps) / (b + eps))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), (n, len(r)))
    values = np.exp(r[idx].mean(1))
    return {"gmr": float(np.exp(r.mean())),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "wins": int(np.sum(a < b)), "n_devices": int(len(r)), "eps": eps}


def decide(tiers: dict) -> dict:
    """Pre-registered I3-1 rule.  ``tiers[T]`` for T in M, W, H holds
    ``valid`` (u0 check), ``failed_fraction`` {arm: fraction} and ``gmr``
    {"hyb_vs_par3": gmr(...), "hyb_vs_lrn_mean": ..., "hyb_vs_pt_interp": ...}."""
    M, W, H = tiers["M"]["gmr"], tiers["W"]["gmr"], tiers["H"]["gmr"]
    conditions = {
        "M1": M["hyb_vs_par3"]["gmr"] <= 0.80 and M["hyb_vs_par3"]["ci95"][1] < 1.0,
        "M2": M["hyb_vs_lrn_mean"]["ci95"][1] < 1.0,
        "M3": M["hyb_vs_pt_interp"]["gmr"] < 1.0,
        "W1": W["hyb_vs_par3"]["ci95"][1] <= 1.25,
        "H1": H["hyb_vs_par3"]["ci95"][1] < 1.0,
        "H2": H["hyb_vs_lrn_mean"]["gmr"] < 1.0 and H["hyb_vs_pt_interp"]["gmr"] < 1.0,
        "F": all(max(tiers[t]["failed_fraction"][m] for m in
                     ("hyb", "par3", "lrn_mean", "pt_interp")) <= 0.10 for t in "MWH"),
    }
    conditions = {k: bool(v) for k, v in conditions.items()}
    kill = {"M_gmr_ge_0.95": bool(M["hyb_vs_par3"]["gmr"] >= 0.95),
            "W_gmr_ge_1.50": bool(W["hyb_vs_par3"]["gmr"] >= 1.50),
            "H_gmr_ge_1.00": bool(H["hyb_vs_par3"]["gmr"] >= 1.00)}
    valid = {t: bool(tiers[t]["valid"]) for t in "MWH"}
    if not all(valid.values()):
        verdict = "INVALID"
    elif all(conditions.values()):
        verdict = "CONTINUE"
    elif any(kill.values()):
        verdict = "KILL"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict_rule_output": verdict, "continue_conditions": conditions,
            "kill_conditions": kill, "valid": valid}
