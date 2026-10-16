"""NGSpice fitting and scoring for the simulated temperature benchmark.

``TempProblem`` duck-types ``ngspice_fit.DeviceProblem`` so the same bounded
TRF polish (``ngspice_fit.polish``) runs unchanged: its "curves" are the
frozen-inclusion metric curves at every fitted temperature, each scored with
the paper's per-curve RRMS (``curve_trim`` on the simulated target).  The
objective is the mean over (temperature, curve), i.e. the mean over fitted
temperatures of the per-temperature paper RRMS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .card_eval import device_by_tag, include_tags
from .cold_start import device_bins
from .data_io import load_device_curves
from .metrics import curve_trim
from .spice_pdk import simulate_pdk
from .temperature import TempBox, card_params

PENALTY = 10.0


@dataclass
class LocalBox:
    """What ``ngspice_fit.polish`` needs from a box: every coordinate active."""
    n_parameters: int

    @property
    def active(self):
        return np.ones(self.n_parameters, dtype=bool)


@dataclass
class TempProblem:
    tag: str
    tbox: TempBox
    targets: dict              # temp_K -> concatenated (2046,) target current
    temps: tuple               # temperatures this problem fits / scores
    device: object = field(init=False)
    curves: list = field(init=False)
    layout: list = field(init=False)
    n_sim: int = field(default=0, init=False)

    def __post_init__(self):
        self.device = device_by_tag(self.tag)
        self.curves = load_device_curves(self.device)
        self.bin_index = device_bins()[self.tag]
        self.box = LocalBox(len(self.tbox.lo))
        tags = include_tags(self.tag)
        bounds, start = [], 0
        for c in self.curves:
            bounds.append((start, start + len(c.Id)))
            start += len(c.Id)
        self.bounds = bounds
        self.layout = []
        n = len(self.curves)
        for j, temp in enumerate(self.temps):
            target = np.asarray(self.targets[temp], dtype=np.float64)
            for i, curve in enumerate(self.curves):
                key = f"{curve.kind}@{abs(float(curve.fixed)):g}"
                if key not in tags:
                    continue
                lo, hi = bounds[i]
                trim, cleaned, denominator = curve_trim(
                    self.device.dev_type, self.device.L_um, self.device.W_um,
                    curve.kind, target[lo:hi])
                if denominator > 0 and np.isfinite(denominator):
                    self.layout.append({
                        "index": j * n + i, "start": trim, "tag": key,
                        "temp": temp, "target": cleaned[trim:],
                        "denominator": float(denominator)})
        self.n_curves = len(self.layout)

    def simulate(self, local: np.ndarray) -> list[np.ndarray]:
        self.n_sim += 1
        z = self.tbox.local_to_z(np.clip(local, -1, 1))
        per_temp = simulate_pdk(self.device.dev_type, self.device.L_um,
                                self.device.W_um, self.curves,
                                params=card_params(self.tbox.stock, z),
                                bin_index=self.bin_index,
                                temps_K=list(self.temps), timeout_s=20)
        return [curve for temp in per_temp for curve in temp]

    def curve_rrms(self, sims) -> np.ndarray:
        values = []
        for row in self.layout:
            sim = np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]
            if not np.all(np.isfinite(sim)):
                values.append(PENALTY)
                continue
            values.append(math.sqrt(float(np.mean((sim - row["target"]) ** 2)))
                          / row["denominator"])
        return np.asarray(values)

    def residual_vector(self, sims, weights: np.ndarray) -> np.ndarray:
        pieces = []
        scale = 1.0 / math.sqrt(self.n_curves)
        for row, weight in zip(self.layout, weights):
            sim = np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]
            n = len(row["target"])
            if not np.all(np.isfinite(sim)):
                pieces.append(np.full(n, PENALTY * weight * scale / math.sqrt(n)))
                continue
            pieces.append((sim - row["target"]) * weight * scale
                          / (row["denominator"] * math.sqrt(n)))
        return np.concatenate(pieces)

    def objective(self, local: np.ndarray) -> float:
        return float(np.mean(self.curve_rrms(self.simulate(local))))

    def per_temp(self, sims) -> dict:
        curves = self.curve_rrms(sims)
        out = {}
        for temp in self.temps:
            mask = [row["temp"] == temp for row in self.layout]
            out[f"{temp:g}"] = float(np.mean(curves[mask]))
        return out


def bank_scores(problem: TempProblem, bank_ids: np.ndarray,
                bank_temps: tuple) -> np.ndarray:
    """Objective of every bank sample (n, n_temps_bank, 2046) without NGSpice."""
    n = len(problem.curves)
    columns = []
    for row in problem.layout:
        j, i = divmod(row["index"], n)
        lo, hi = problem.bounds[i]
        block = bank_ids[:, bank_temps.index(problem.temps[j]), lo + row["start"]:hi]
        columns.append(np.sqrt(np.mean((block - row["target"][None]) ** 2, axis=1))
                       / row["denominator"])
    return np.nan_to_num(np.mean(np.stack(columns, 1), axis=1), nan=np.inf)
