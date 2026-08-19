"""Generic bounded parameter coordinates for foundation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParameterBox:
    """Per-device physical bounds with a shared ordered parameter layout.

    Parameters that are inactive for a particular family/bin use ``lo == hi``
    and map to local coordinate zero.  This lets one foundation network retain
    a common maximum-size input while preventing meaningless device-specific
    degrees of freedom.
    """

    names: tuple[str, ...]
    published: dict[str, float]
    bounds: dict[str, list[float] | tuple[float, float]]
    lo: np.ndarray = field(init=False)
    hi: np.ndarray = field(init=False)
    center: np.ndarray = field(init=False)
    active: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(
            [float(self.published[name]) for name in self.names],
            dtype=np.float64)
        self.lo = np.asarray(
            [float(self.bounds[name][0]) for name in self.names],
            dtype=np.float64)
        self.hi = np.asarray(
            [float(self.bounds[name][1]) for name in self.names],
            dtype=np.float64)
        if not (np.isfinite(self.center).all() and np.isfinite(self.lo).all()
                and np.isfinite(self.hi).all()):
            raise ValueError("parameter box contains non-finite values")
        if np.any(self.lo > self.hi):
            raise ValueError("parameter lower bound exceeds upper bound")
        if np.any(self.center < self.lo) or np.any(self.center > self.hi):
            raise ValueError("published parameter lies outside its bounds")
        scale = np.maximum(np.maximum(np.abs(self.lo), np.abs(self.hi)), 1.0)
        self.active = (self.hi - self.lo) > (1e-14 * scale)

    @property
    def n_parameters(self) -> int:
        return len(self.names)

    @property
    def n_active(self) -> int:
        return int(np.sum(self.active))

    @property
    def local_published(self) -> np.ndarray:
        return self.params_to_local(self.published)

    def local_to_array(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, dtype=np.float64)
        if local.shape[-1] != self.n_parameters:
            raise ValueError("local coordinate has the wrong final dimension")
        clipped = np.clip(local, -1.0, 1.0)
        values = self.center + np.zeros_like(clipped, dtype=np.float64)
        values[..., self.active] = (
            self.lo[self.active]
            + 0.5 * (clipped[..., self.active] + 1.0)
            * (self.hi[self.active] - self.lo[self.active]))
        values[..., ~self.active] = self.center[~self.active]
        return values

    def local_to_params(self, local: np.ndarray) -> dict[str, float]:
        values = self.local_to_array(np.asarray(local, dtype=np.float64))
        if values.ndim != 1:
            raise ValueError("local_to_params expects one parameter vector")
        return {name: float(values[index])
                for index, name in enumerate(self.names)}

    def params_to_local(self, params: dict[str, float]) -> np.ndarray:
        values = np.asarray([float(params[name]) for name in self.names],
                            dtype=np.float64)
        local = np.zeros(self.n_parameters, dtype=np.float64)
        local[self.active] = (
            2.0 * (values[self.active] - self.lo[self.active])
            / (self.hi[self.active] - self.lo[self.active]) - 1.0)
        return np.clip(local, -1.0, 1.0)

