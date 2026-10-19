"""E3 learned temperature laws, shared across devices.

Inputs per training device: geometry features and per-temperature
seven-parameter fits ``x(T)`` with their Gauss-Newton matrices ``H(T)``
(``e3_temp.fit_temperature``) on the dense grid.  For an observation set O
(anchor temperatures), every model predicts

    x(T) = x_lin(T) + phi_O(T) * sigma * correction(T)

where ``x_lin`` interpolates the anchors linearly in T and ``phi_O`` is zero
at every anchor, so all models reproduce the observed-temperature fits.

* ``lrn``: an MLP ensemble for ``correction`` from (geometry, standardised
  anchors, temperature feature), loss ``d^T Hn d + 0.1 mean((d / sigma)^2)``
  with ``Hn = 7 H / tr(H)`` (sensitivity-weighted parameter error);
* ``lrn_param``: the same MLP with loss ``mean((d / sigma)^2)`` only;
* ``lrn_noH``: ``lrn`` trained without the held-out temperatures;
* ``lrn_mean``: the population-mean correction per device type and
  temperature (no network);
* ``pt_interp``: correction = 0 (linear interpolation of the anchors).
"""

from __future__ import annotations

import numpy as np

from ..temperature import temperature_feature
from .e3_temp import N_PAR, linear_interp, phi


def _samples(records, anchors, temps):
    """(geometry, anchor x, temperature, target x, H, phi, x_lin, device) rows."""
    rows = []
    for d_index, rec in enumerate(records):
        ax = np.array([rec["fits"][f"{t:g}"]["x"] for t in anchors])
        for t in temps:
            if t in anchors:
                continue
            row = rec["fits"][f"{t:g}"]
            H = np.asarray(row["H"], dtype=np.float64)
            if not np.all(np.isfinite(H)) or not np.isfinite(row["rrms"]):
                continue
            rows.append((np.asarray(rec["geometry"]), ax, t, np.asarray(row["x"]), H,
                         phi(anchors, t), linear_interp(anchors, ax, t), d_index))
    return rows


def _sigma(rows):
    resid = np.array([(r[3] - r[6]) / max(r[5], 1e-3) for r in rows])
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid, 0)), 0)
    return np.maximum(mad, 0.02)


class MeanLaw:
    """Population-mean correction per device type and temperature."""

    def __init__(self, records, anchors, temps):
        self.anchors = tuple(anchors)
        rows = _samples(records, anchors, temps)
        self.sigma = _sigma(rows)
        self.table = {}
        for kind in (-1.0, 1.0):
            sel = [r for r in rows if r[0][0] == kind]
            grid = sorted({r[2] for r in sel})
            self.table[kind] = (np.array(grid), np.array([
                np.mean([(r[3] - r[6]) / r[5] for r in sel if r[2] == t], 0)
                for t in grid]))

    def predict(self, geometry, anchor_x, temps):
        grid, values = self.table[float(geometry[0])]
        out = []
        for t in temps:
            lin = linear_interp(self.anchors, anchor_x, t)
            corr = np.array([np.interp(t, grid, values[:, k]) for k in range(N_PAR)])
            out.append(lin + phi(self.anchors, t) * corr)
        return np.array(out)


class NetLaw:
    """MLP ensemble for the correction (see module docstring)."""

    def __init__(self, records, anchors, temps, loss="sens", seeds=5, width=64,
                 steps=3000, lr=3e-3, weight_decay=1e-4, ridge=0.1):
        import torch
        self.anchors = tuple(anchors)
        rows = _samples(records, anchors, temps)
        self.sigma = _sigma(rows)
        ax = np.array([r[1].ravel() for r in rows])
        self.a_mean, self.a_std = ax.mean(0), np.maximum(ax.std(0), 1e-3)
        X = torch.tensor(self._features([r[0] for r in rows], [r[1] for r in rows],
                                        [r[2] for r in rows]), dtype=torch.float32)
        target = torch.tensor(np.array([r[3] for r in rows]), dtype=torch.float32)
        lin = torch.tensor(np.array([r[6] for r in rows]), dtype=torch.float32)
        ph = torch.tensor(np.array([r[5] for r in rows]), dtype=torch.float32)[:, None]
        Hn = np.array([7.0 * r[4] / max(np.trace(r[4]), 1e-30) for r in rows])
        Hn = torch.tensor(Hn, dtype=torch.float32)
        sig = torch.tensor(self.sigma, dtype=torch.float32)
        device_of = np.array([r[7] for r in rows])
        self.models = []
        self.history = []

        def objective(model, idx):
            d = lin[idx] + ph[idx] * sig * model(X[idx]) - target[idx]
            param = torch.mean((d / sig) ** 2)
            if loss == "sens":
                return torch.mean(torch.einsum("ni,nij,nj->n", d, Hn[idx], d)) + ridge * param
            return param

        for seed in range(seeds):
            # Each ensemble member early-stops on its own 20% of training
            # devices (device-level split, seeded).
            rng = np.random.default_rng(20261019 + seed)
            devices = np.unique(device_of)
            val_dev = rng.choice(devices, max(1, len(devices) // 5), replace=False)
            is_val = np.isin(device_of, val_dev)
            tr = torch.tensor(np.flatnonzero(~is_val))
            va = torch.tensor(np.flatnonzero(is_val))
            torch.manual_seed(20261019 + seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], width), torch.nn.SiLU(),
                torch.nn.Linear(width, width), torch.nn.SiLU(),
                torch.nn.Linear(width, N_PAR))
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
            best, best_state, best_step = float("inf"), None, 0
            for step in range(steps):
                opt.zero_grad()
                value = objective(model, tr)
                value.backward()
                opt.step()
                sched.step()
                if step % 25 == 0 or step == steps - 1:
                    with torch.no_grad():
                        v = float(objective(model, va))
                    if v < best:
                        best, best_step = v, step
                        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            self.history.append({"best_val": best, "best_step": best_step})
        self.n_rows = len(rows)

    def _features(self, geometries, anchor_xs, temps):
        a = (np.array([np.asarray(x).ravel() for x in anchor_xs]) - self.a_mean) / self.a_std
        return np.concatenate([np.array(geometries), a,
                               np.array([[temperature_feature(t)] for t in temps])], 1)

    def predict(self, geometry, anchor_x, temps):
        import torch
        X = torch.tensor(self._features([geometry] * len(temps), [anchor_x] * len(temps),
                                        list(temps)), dtype=torch.float32)
        with torch.no_grad():
            corr = np.mean([m(X).numpy() for m in self.models], 0).astype(np.float64)
        out = []
        for t, c in zip(temps, corr):
            out.append(linear_interp(self.anchors, anchor_x, t)
                       + phi(self.anchors, t) * self.sigma * c)
        return np.array(out)
