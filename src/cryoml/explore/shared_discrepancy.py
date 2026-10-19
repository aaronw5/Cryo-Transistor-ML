"""E5: shared multiplicative discrepancy across devices on top of per-device cards.

I_corr = I_sim * exp(delta(x)), delta = tanh(MLP(x)).  x per bias point:
family (+-1), log10 L, log10 W, |Vg|, |Vd|, log10 |I_sim|.  The training loss is
the first-order expansion of the per-curve squared paper RRMS:

    (I_sim e^delta - I_meas) / den  ~=  (|I_meas| / den) * (delta - r),
    r = log|I_meas| - log|I_sim|,

weighted so each curve and each device has equal total weight.
See out/research/explore/E5/prereg.json.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..card_eval import device_by_tag, include_tags
from ..cold_start import device_bins
from ..config import OUT_DIR
from ..data_io import load_device_curves
from ..metrics import curve_trim, score_device_new
from ..spice_pdk import simulate_pdk

SPLIT_JSON = OUT_DIR / "research" / "explore" / "real_split.json"
I_FLOOR = 1e-15
GEOMETRY_COLUMNS = (1, 2)
N_FEATURES = 6


def real_split() -> dict:
    return json.loads(SPLIT_JSON.read_text())


def dev_tags() -> list[str]:
    return list(real_split()["REAL_DEV"])


def assert_dev(tags) -> None:
    test = set(real_split()["REAL_TEST"])
    bad = sorted(set(tags) & test)
    if bad:
        raise RuntimeError(f"REAL_TEST devices requested in an exploration probe: {bad}")


def curve_features(device, curve, i_sim: np.ndarray) -> np.ndarray:
    n = min(len(curve.Id), len(i_sim))
    fam = 1.0 if device.dev_type == "nmos" else -1.0
    x = np.empty((n, N_FEATURES))
    x[:, 0] = fam
    x[:, 1] = math.log10(device.L_um)
    x[:, 2] = math.log10(device.W_um)
    x[:, 3] = np.abs(np.asarray(curve.Vg, dtype=np.float64)[:n])
    x[:, 4] = np.abs(np.asarray(curve.Vd, dtype=np.float64)[:n])
    x[:, 5] = np.log10(np.maximum(np.abs(np.asarray(i_sim, dtype=np.float64)[:n]),
                                  I_FLOOR))
    return x


@dataclass
class DeviceData:
    tag: str
    device: object
    curves: list
    sims: list            # base-card simulated currents, one array per curve
    x: np.ndarray         # training points (included, trimmed)
    r: np.ndarray         # log-ratio targets
    w: np.ndarray         # weights (sum = 1 per device)


def simulate_base(tag: str, params: dict) -> tuple[object, list, list]:
    device = device_by_tag(tag)
    curves = load_device_curves(device)
    sims = simulate_pdk(device.dev_type, device.L_um, device.W_um, curves,
                        params=params, bin_index=device_bins()[tag])
    return device, curves, [np.asarray(s, dtype=np.float64) for s in sims]


def training_points(tag: str, device, curves, sims) -> tuple[np.ndarray, ...]:
    tags = include_tags(tag)
    rows = []
    for curve, sim in zip(curves, sims):
        key = f"{curve.kind}@{abs(float(curve.fixed)):g}"
        if key not in tags:
            continue
        start, cleaned, den = curve_trim(device.dev_type, device.L_um,
                                         device.W_um, curve.kind, curve.Id)
        if not (den > 0 and np.isfinite(den)):
            continue
        n = min(len(cleaned), len(sim))
        meas = np.abs(cleaned[start:n])
        s = np.abs(sim[start:n])
        if not np.all(np.isfinite(s)):
            raise RuntimeError(f"{tag} {key}: base card curve is non-finite")
        eps = 1e-3 * den
        r = np.log(np.maximum(meas, eps)) - np.log(np.maximum(s, eps))
        w = (meas / den) ** 2 / len(meas)
        x = curve_features(device, curve, sim)[start:n]
        rows.append((x, r, w))
    x = np.concatenate([a for a, _, _ in rows])
    r = np.concatenate([b for _, b, _ in rows])
    w = np.concatenate([c for _, _, c in rows]) / len(rows)
    return x, r, w / w.sum()


def load_device(tag: str, params: dict) -> DeviceData:
    device, curves, sims = simulate_base(tag, params)
    x, r, w = training_points(tag, device, curves, sims)
    return DeviceData(tag, device, curves, sims, x, r, w)


# ---------------------------------------------------------------- models

class DeltaModel:
    """Ensemble of small tanh MLPs; numpy inference for use inside NGSpice loops."""

    def __init__(self, mean, std, layers_list, drop_cols=()):
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)
        self.layers_list = layers_list   # list (ensemble) of [(W, b), ...]
        self.drop_cols = tuple(drop_cols)

    def _prep(self, x):
        z = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        if self.drop_cols:
            z = np.delete(z, self.drop_cols, axis=1)
        return z

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z0 = self._prep(x)
        outs = []
        for layers in self.layers_list:
            h = z0
            for i, (W, b) in enumerate(layers):
                h = h @ W + b
                if i < len(layers) - 1:
                    h = np.tanh(h)
            outs.append(np.tanh(h[:, 0]))
        return np.mean(outs, axis=0)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "drop_cols": list(self.drop_cols),
                "layers": [[(W.tolist(), b.tolist()) for W, b in layers]
                           for layers in self.layers_list]}

    @classmethod
    def from_json(cls, d: dict) -> "DeltaModel":
        layers_list = [[(np.asarray(W), np.asarray(b)) for W, b in layers]
                       for layers in d["layers"]]
        return cls(d["mean"], d["std"], layers_list, d.get("drop_cols", ()))


class ConstModel:
    """Per-family constant delta."""

    def __init__(self, values: dict[float, float]):
        self.values = values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        fam = np.asarray(x)[:, 0]
        return np.array([self.values.get(float(f), 0.0) for f in fam])


def fit_const(x: np.ndarray, r: np.ndarray, w: np.ndarray) -> ConstModel:
    values = {}
    for fam in (1.0, -1.0):
        m = x[:, 0] == fam
        if m.any():
            values[fam] = float(np.clip(np.sum(w[m] * r[m]) / np.sum(w[m]), -1, 1))
    return ConstModel(values)


def fit_mlp(x: np.ndarray, r: np.ndarray, w: np.ndarray, *, drop_cols=(),
            hidden: int = 64, steps: int = 3000, lr: float = 3e-3,
            weight_decay: float = 1e-4, seeds=(0, 1, 2, 3, 4),
            device: str | None = None) -> DeltaModel:
    import torch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    model_stub = DeltaModel(mean, std, [], drop_cols)
    z = torch.tensor(model_stub._prep(x), dtype=torch.float32)
    rt = torch.tensor(r, dtype=torch.float32)
    wt = torch.tensor(w / w.sum(), dtype=torch.float32)
    # Ensemble members are trained as one batched computation: each seed has
    # its own parameters (default nn.Linear init under that seed), the loss is
    # the sum of the per-seed losses, and AdamW is elementwise, so every member
    # follows exactly its own independent optimization trajectory.
    params = []
    for seed in seeds:
        torch.manual_seed(seed)
        lins = [torch.nn.Linear(z.shape[1], hidden), torch.nn.Linear(hidden, hidden),
                torch.nn.Linear(hidden, 1)]
        params.append([(m.weight.detach().T.clone(), m.bias.detach().clone())
                       for m in lins])
    z, rt, wt = z.to(device), rt.to(device), wt.to(device)
    Ws = [torch.stack([p[i][0] for p in params]).to(device).requires_grad_()
          for i in range(3)]
    bs = [torch.stack([p[i][1] for p in params])[:, None, :].to(device).requires_grad_()
          for i in range(3)]
    opt = torch.optim.AdamW(Ws + bs, lr=lr, weight_decay=weight_decay)
    zb = z[None].expand(len(seeds), -1, -1)
    for _ in range(steps):
        opt.zero_grad()
        h = torch.tanh(torch.bmm(zb, Ws[0]) + bs[0])
        h = torch.tanh(torch.bmm(h, Ws[1]) + bs[1])
        d = torch.tanh(torch.bmm(h, Ws[2]) + bs[2])[:, :, 0]
        loss = torch.sum(wt[None] * (d - rt[None]) ** 2)
        loss.backward()
        opt.step()
    layers_list = [[(Ws[i][s].detach().cpu().numpy().astype(np.float64),
                     bs[i][s, 0].detach().cpu().numpy().astype(np.float64))
                    for i in range(3)] for s in range(len(seeds))]
    return DeltaModel(mean, std, layers_list, drop_cols)


# ---------------------------------------------------------------- scoring

def corrected_sims(device, curves, sims, model) -> list[np.ndarray]:
    out = []
    for curve, sim in zip(curves, sims):
        sim = np.asarray(sim, dtype=np.float64)
        if model is None or not np.all(np.isfinite(sim)):
            out.append(sim)
            continue
        x = curve_features(device, curve, sim)
        corr = sim.copy()
        corr[:len(x)] = sim[:len(x)] * np.exp(model(x))
        out.append(corr)
    return out


def score(tag: str, device, curves, sims, model) -> dict:
    res = score_device_new(device.dev_type, device.L_um, device.W_um, curves,
                           corrected_sims(device, curves, sims, model),
                           include_tags=include_tags(tag))
    return {"rrms": float(res["rrms"]), "n_failed_curves": res["n_failed_curves"],
            "per_curve": {k: v["rrms"] for k, v in res["per_curve"].items()
                          if v.get("included")}}


def aggregate(per_device: dict[str, float]) -> dict:
    vals = np.array(list(per_device.values()), dtype=float)
    fam = {f: [v for t, v in per_device.items() if t.startswith(f)]
           for f in ("nmos", "pmos")}
    n_mean = float(np.mean(fam["nmos"])) if fam["nmos"] else float("nan")
    p_mean = float(np.mean(fam["pmos"])) if fam["pmos"] else float("nan")
    return {"all_dev_mean": float(np.mean(vals)), "nmos": n_mean, "pmos": p_mean,
            "combined": 0.5 * (n_mean + p_mean),
            "n_failed": int(np.sum(~np.isfinite(vals))), "n": int(len(vals))}


# ---------------------------------------------------------------- stage 2 helper

def corrected_problem_class():
    """DeviceProblem whose simulate() returns delta-corrected currents."""
    from ..ngspice_fit import DeviceProblem

    class CorrectedProblem(DeviceProblem):
        def __init__(self, tag, box, model):
            super().__init__(tag, box)
            self.model = model

        def simulate(self, local):
            sims = super().simulate(local)
            return corrected_sims(self.device, self.curves, sims, self.model)

    return CorrectedProblem


def load_base_cards(path: Path) -> dict[str, dict]:
    devices = json.loads(Path(path).read_text())["devices"]
    return {t: r["params"] for t, r in devices.items()}
