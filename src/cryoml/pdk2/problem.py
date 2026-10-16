"""``Pdk2Problem``: the ``cryoml.ngspice_fit.DeviceProblem`` interface on a
pdk2 target (GF180 ``gf180_*`` or IHP ``ihp_*`` tags; the backend follows
the tag), so ``ngspice_fit.polish``, CMA-ES on ``objective`` and the
trust-region code run unchanged.

Interface (as ``DeviceProblem``): ``tag``, ``box`` (``ParameterBox`` with
``.active``), ``device`` (``dev_type``, ``L_um``, ``W_um``), ``curves``
(``Curve`` objects whose ``Id`` is the target), ``layout`` (included curves:
``index``, ``start``, ``tag``, ``target``, ``denominator``), ``bin_index``,
``n_curves``, ``n_sim``; ``simulate(local)``, ``curve_rrms(sims)``,
``residual_vector(sims, weights)``, ``objective(local)``.

Extras: ``simulate_many(locals)`` and ``residual_jacobian(local)`` run many
cards in one ngspice process (altermod, bit-identical to ``simulate``), and
``final_score`` / ``bank_starts`` mirror their ``ngspice_fit`` /
``ngspice_direct`` counterparts.  Targets come from a truth set
(``load_truth``) or are passed directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import OUT_DIR
from ..parameter_space import ParameterBox
from . import metric
from .boxes import backend, pdk_of_tag

TRUTH_ROOT = OUT_DIR / "research" / "pdk2" / "truth"          # GF180
TRUTH_ROOTS = {"gf180": TRUTH_ROOT,
               "ihp": OUT_DIR / "research" / "pdk2" / "ihp_truth"}


def load_truth(tier: str, set_index: int, root: Path | None = None
               ) -> dict[str, list[np.ndarray]]:
    """tag -> the 11 observed target curves of one truth set (``root``
    defaults to the GF180 truth root)."""
    path = Path(root or TRUTH_ROOT) / tier / f"set_{set_index}.npz"
    with np.load(path, allow_pickle=False) as data:
        tags = [str(t) for t in data["tags"]]
        return {tag: [np.asarray(data[f"{tag}__{i}"], dtype=np.float64)
                      for i in range(11)] for tag in tags}


def load_truth_meta(tier: str, root: Path | None = None) -> dict:
    return json.loads((Path(root or TRUTH_ROOT) / tier / "truth.json").read_text())


@dataclass
class Pdk2Problem:
    tag: str
    box: ParameterBox
    targets: Sequence[np.ndarray] | None = None
    temp_C: float | None = None
    device: object = field(init=False)
    curves: list = field(init=False)
    layout: list = field(init=False)
    bin_index: int = field(init=False)
    n_sim: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.targets is None:
            raise ValueError("Pdk2Problem needs target curves (see load_truth)")
        self.pdk = pdk_of_tag(self.tag)
        self.backend = backend(self.pdk)
        if self.temp_C is None:
            self.temp_C = self.backend.TEMP_C
        self.device = self.backend.device_by_tag(self.tag)
        self.curves = self.backend.device_curves(self.device.dev_type, self.targets)
        self.bin_index = self.backend.bin_index(self.device)
        self.layout = metric.curve_layout(self.curves)
        if not self.layout:
            raise RuntimeError(f"{self.tag}: no included curves")
        self.n_curves = len(self.layout)

    @classmethod
    def from_truth(cls, tag: str, box: ParameterBox, tier: str,
                   set_index: int, root: Path | None = None) -> "Pdk2Problem":
        root = root or TRUTH_ROOTS[pdk_of_tag(tag)]
        return cls(tag, box, load_truth(tier, set_index, root)[tag])

    # -- DeviceProblem interface -------------------------------------------
    def params(self, local: np.ndarray) -> dict[str, float]:
        return self.box.local_to_params(np.asarray(local, dtype=np.float64))

    def simulate(self, local: np.ndarray) -> list[np.ndarray]:
        self.n_sim += 1
        return self.backend.simulate(self.device, self.params(local),
                                     curves=self.curves, temp_C=self.temp_C)

    def curve_rrms(self, sims) -> np.ndarray:
        return metric.curve_rrms(sims, self.layout)

    def residual_vector(self, sims, weights: np.ndarray) -> np.ndarray:
        pieces = []
        scale = 1.0 / math.sqrt(self.n_curves)
        for row, weight in zip(self.layout, weights):
            sim = np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]
            n = len(row["target"])
            if not np.all(np.isfinite(sim)):
                pieces.append(np.full(n, metric.PENALTY * weight * scale
                                      / math.sqrt(n)))
                continue
            pieces.append((sim - row["target"]) * weight * scale
                          / (row["denominator"] * math.sqrt(n)))
        return np.concatenate(pieces)

    def objective(self, local: np.ndarray) -> float:
        """Mean included-curve RRMS (the fixed-inclusion paper metric)."""
        return float(np.mean(self.curve_rrms(self.simulate(local))))

    # -- batched extras ----------------------------------------------------
    def simulate_many(self, locals_: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
        self.n_sim += len(locals_)
        return self.backend.simulate_many(
            self.device, [self.params(z) for z in locals_], curves=self.curves,
            temp_C=self.temp_C)

    def residual_jacobian(self, local: np.ndarray, weights=None,
                          step: float = 1e-3, active_only: bool = True):
        """Forward-difference Jacobian of ``residual_vector`` in local
        coordinates (absolute ``step``, taken inward at the upper face as
        ``ngspice_fit.polish``), from one altermod process of 1 + n cards.
        Returns ``(r0, J, sims0)``; J columns follow the active indices."""
        local = np.asarray(local, dtype=np.float64)
        weights = np.ones(self.n_curves) if weights is None else weights
        columns = (np.flatnonzero(self.box.active) if active_only
                   else np.arange(self.box.n_parameters))
        trials, steps = [local.copy()], []
        for index in columns:
            h = step if local[index] + step <= 1.0 else -step
            shifted = local.copy()
            shifted[index] += h
            trials.append(shifted)
            steps.append(h)
        sims = self.simulate_many(trials)
        r0 = self.residual_vector(sims[0], weights)
        jac = np.empty((r0.size, len(columns)))
        for j, h in enumerate(steps):
            jac[:, j] = (self.residual_vector(sims[1 + j], weights) - r0) / h
        return r0, jac, sims[0]


class TrustRegionBackend:
    """The expensive backend ``cryoml.trust_region.TrustRegionRun`` expects
    (``n_calls``, ``jacobian_cost``, ``residual(z)``, ``jacobian(z, r)``) on
    the active coordinates ``z``.  The residual is ``residual_vector`` with
    unit weights, i.e. hybrid_tr's per-curve normalization; the Jacobian is
    one batched altermod process (forward step ``diff_step``, inward at the
    upper face).  Every simulation counts as a call."""

    def __init__(self, problem: Pdk2Problem, base_local: np.ndarray | None = None,
                 diff_step: float = 1e-3):
        self.problem = problem
        self.active = np.flatnonzero(problem.box.active)
        self.base = (problem.box.local_published if base_local is None
                     else np.asarray(base_local, dtype=np.float64)).copy()
        self.diff_step = float(diff_step)
        self.jacobian_cost = len(self.active)
        self.weights = np.ones(problem.n_curves)

    @property
    def n_calls(self) -> int:
        return self.problem.n_sim

    def local(self, z: np.ndarray) -> np.ndarray:
        local = self.base.copy()
        local[self.active] = np.clip(z, -1.0, 1.0)
        return local

    def residual(self, z: np.ndarray) -> np.ndarray:
        sims = self.problem.simulate(self.local(z))
        return self.problem.residual_vector(sims, self.weights)

    def jacobian(self, z: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Probes only (``r`` is the centre residual): n_active calls."""
        local = self.local(z)
        trials, steps = [], []
        for index in self.active:
            h = self.diff_step if local[index] + self.diff_step <= 1.0 \
                else -self.diff_step
            shifted = local.copy()
            shifted[index] += h
            trials.append(shifted)
            steps.append(h)
        sims = self.problem.simulate_many(trials)
        jac = np.empty((len(r), len(trials)))
        for j, h in enumerate(steps):
            jac[:, j] = (self.problem.residual_vector(sims[j], self.weights) - r) / h
        return jac


def final_score(problem: Pdk2Problem, local: np.ndarray) -> dict:
    """Fresh simulation scored with the frozen-inclusion paper RRMS (the
    ``ngspice_fit.final_score`` record; inclusion is target-only, so the
    dynamic score equals the fixed one)."""
    local = np.asarray(local, dtype=np.float64)
    params = problem.params(local)
    sims = problem.backend.simulate(problem.device, params,
                                    curves=problem.curves, temp_C=problem.temp_C)
    scored = metric.device_rrms(sims, problem.curves)
    if not np.isfinite(scored["rrms"]):
        scored["rrms"] = float("inf")
    scored["official_dynamic_rrms"] = scored["rrms"]
    scored["params"] = params
    scored["local"] = local.tolist()
    return scored


def bank_starts(problem: Pdk2Problem, bank_path: Path, n_starts: int,
                min_distance: float = 0.35) -> tuple[list[np.ndarray], float]:
    """Best diverse bank samples on the problem's targets (as
    ``ngspice_direct.bank_starts``)."""
    with np.load(bank_path, allow_pickle=False) as data:
        ok = np.asarray(data["ok"], dtype=bool)
        local = np.asarray(data["LOCAL"], dtype=np.float64)[ok]
        ids = np.asarray(data["IDS"], dtype=np.float64)[ok]
        slices = np.asarray(data["slices"])
        kinds = [str(k) for k in data["kinds"]]
        fixeds = np.abs(np.asarray(data["fixeds"], dtype=np.float64))
    columns = []
    for row in problem.layout:
        curve = problem.curves[row["index"]]
        match = [i for i, (k, b) in enumerate(zip(kinds, fixeds))
                 if k == curve.kind and abs(b - abs(float(curve.fixed))) < 5e-3]
        start, stop = (int(v) for v in slices[match[0]])
        block = ids[:, start + row["start"]:stop]
        columns.append(np.sqrt(np.mean((block - row["target"][None]) ** 2,
                                       axis=1)) / row["denominator"])
    scores = np.nan_to_num(np.mean(np.stack(columns, 1), axis=1), nan=np.inf)
    active = problem.box.active
    chosen: list[np.ndarray] = []
    for index in np.argsort(scores):
        candidate = local[index]
        if not chosen or min(np.sqrt(np.mean((candidate[active] - c[active]) ** 2))
                             for c in chosen) >= min_distance:
            chosen.append(candidate)
        if len(chosen) >= n_starts:
            break
    return chosen, float(scores.min())
