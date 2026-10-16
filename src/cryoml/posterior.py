"""Amortized I-V -> parameter models in a round-independent coordinate.

Every bank sample is mapped to z = log(value / multiplier-free value), so
banks from different cold-loop rounds, and the paper's banks, share one
parameter space and can be pooled.  Parameters whose multiplier-free value is
zero, or that a bank holds fixed, are masked out of the loss.

Two families are implemented:

* ``FlowPosterior``: conditional flow matching (rectified flow).  A velocity
  field transports N(0, I) to p(z | I-V, geometry); sampling returns many
  plausible cards, so non-identifiable directions appear as spread rather
  than as an arbitrary average.
* ``WTAPosterior``: K deterministic heads trained winner-takes-all, a cheap
  multi-hypothesis alternative without likelihoods.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data_io import bank_measured
from .devices import PAPER_DEVICES
from .foundation import (canonical_point_order, geometry_features,
                         geometry_ranges, signed_log)
from .utils import device_tag


@dataclass
class PooledBank:
    names: tuple[str, ...]
    z: np.ndarray            # (N, P) log multipliers relative to cold center
    mask: np.ndarray         # (N, P) bool, parameter varied in that bank
    currents: np.ndarray     # (N, n_points) canonical order, float32
    device_ids: np.ndarray   # (N,)
    measured: np.ndarray     # (n_devices, n_points)
    curve_scales: np.ndarray  # (n_devices, n_points) per-curve max |meas|
    geometry: np.ndarray     # (n_devices, 3)
    descriptions: list
    tags: list
    sources: list


def load_pooled_banks(bank_dirs: list[Path], names: tuple[str, ...],
                      centers: dict[str, dict]) -> PooledBank:
    ranges = geometry_ranges(PAPER_DEVICES)
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    zs, masks, currents, ids = [], [], [], []
    measured, scales, geometry = [], [], []
    descriptions = None
    for device_index, device in enumerate(PAPER_DEVICES):
        tag = tags[device_index]
        center = np.asarray([centers[tag][n] for n in names], dtype=np.float64)
        geometry.append(geometry_features(device, ranges))
        for bank_index, bank in enumerate(bank_dirs):
            with np.load(Path(bank) / f"{tag}.npz", allow_pickle=True) as data:
                bank_names = tuple(str(v) for v in data["parameter_names"]) \
                    if "parameter_names" in data.files else None
                order, desc = canonical_point_order(
                    data["kinds"], data["fixeds"], data["slices"])
                if descriptions is None:
                    descriptions = desc
                ok = np.asarray(data["ok"], dtype=bool)
                theta = np.asarray(data["THETA"], dtype=np.float64)[ok]
                if bank_names is None:
                    bank_names = names
                index = [bank_names.index(n) for n in names]
                theta = theta[:, index]
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = theta / center[None, :]
                usable = (center != 0) & np.all(ratio > 0, axis=0)
                varied = np.ptp(theta, axis=0) > 1e-14 * np.maximum(
                    np.abs(center), 1e-30)
                keep = usable & varied
                z = np.zeros_like(theta)
                z[:, keep] = np.log(ratio[:, keep])
                zs.append(z.astype(np.float32))
                masks.append(np.broadcast_to(keep, z.shape).copy())
                currents.append(np.asarray(data["IDS"], dtype=np.float32)[ok][:, order])
                ids.append(np.full(len(z), device_index, dtype=np.int64))
                if bank_index == 0:
                    meas = bank_measured(data, device)[order]
                    measured.append(meas)
                    scale = np.zeros_like(meas)
                    for row in desc:
                        block = np.abs(meas[row["start"]:row["stop"]])
                        scale[row["start"]:row["stop"]] = max(
                            float(block.max()), 1e-12)
                    scales.append(scale)
    return PooledBank(
        names=names, z=np.concatenate(zs), mask=np.concatenate(masks),
        currents=np.concatenate(currents), device_ids=np.concatenate(ids),
        measured=np.stack(measured).astype(np.float32),
        curve_scales=np.stack(scales).astype(np.float32),
        geometry=np.stack(geometry).astype(np.float32),
        descriptions=descriptions, tags=tags,
        sources=[str(b) for b in bank_dirs])


def target_measured(bank_dir: Path, tags: list[str]):
    """Current targets (measured, or the simulated override) in canonical
    order, with their per-curve max-|I| scales, for the given device tags."""
    by_tag = {device_tag(d.dev_type, d.L_um, d.W_um): d for d in PAPER_DEVICES}
    measured, scales = [], []
    for tag in tags:
        with np.load(Path(bank_dir) / f"{tag}.npz", allow_pickle=True) as data:
            order, desc = canonical_point_order(
                data["kinds"], data["fixeds"], data["slices"])
            meas = bank_measured(data, by_tag[tag])[order]
        scale = np.zeros_like(meas)
        for row in desc:
            block = np.abs(meas[row["start"]:row["stop"]])
            scale[row["start"]:row["stop"]] = max(float(block.max()), 1e-12)
        measured.append(meas)
        scales.append(scale)
    return (np.stack(measured).astype(np.float32),
            np.stack(scales).astype(np.float32))


class FeatureMap(nn.Module):
    """Standardized signed-log current, measured-scale linear current, geometry."""

    def __init__(self, slog_mean: np.ndarray, slog_std: np.ndarray,
                 curve_scales: np.ndarray, geometry: np.ndarray):
        super().__init__()
        self.register_buffer("slog_mean", torch.as_tensor(slog_mean))
        self.register_buffer("slog_std", torch.as_tensor(slog_std))
        self.register_buffer("curve_scales", torch.as_tensor(curve_scales))
        self.register_buffer("geometry", torch.as_tensor(geometry))

    @property
    def n_features(self) -> int:
        return 2 * self.slog_mean.shape[0] + 3

    def forward(self, current: torch.Tensor, device_ids: torch.Tensor):
        slog = torch.clamp((signed_log(current) - self.slog_mean)
                           / self.slog_std, -10, 10)
        linear = torch.clamp(current / self.curve_scales[device_ids], -20, 20)
        return torch.cat([slog, linear, self.geometry[device_ids]], dim=1)


def augment(current: torch.Tensor, *, gain_sigma: float, point_sigma: float,
            floor_amps: float, generator=None) -> torch.Tensor:
    """Measurement-like corruption applied to simulated training currents."""
    if gain_sigma <= 0 and point_sigma <= 0 and floor_amps <= 0:
        return current
    n, p = current.shape
    out = current
    if gain_sigma > 0:
        out = out * torch.exp(gain_sigma * torch.randn(n, 1, device=current.device))
    if point_sigma > 0:
        out = out * (1 + point_sigma * torch.randn(n, p, device=current.device))
    if floor_amps > 0:
        level = floor_amps * torch.exp(torch.randn(n, 1, device=current.device))
        out = out + level * torch.randn(n, p, device=current.device)
    return out


def encoder(n_in: int, hidden: int, context: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, context), nn.LayerNorm(context))


class FlowPosterior(nn.Module):
    def __init__(self, n_in: int, n_parameters: int, *, hidden: int = 1024,
                 context: int = 512, velocity_hidden: int = 512,
                 velocity_layers: int = 4):
        super().__init__()
        self.n_parameters = n_parameters
        self.encode = encoder(n_in, hidden, context)
        layers, width = [], n_parameters + 32 + context
        for _ in range(velocity_layers):
            layers += [nn.Linear(width, velocity_hidden), nn.GELU()]
            width = velocity_hidden
        layers.append(nn.Linear(width, n_parameters))
        self.velocity = nn.Sequential(*layers)
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(100), 16)))

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        angle = t[:, None] * self.freq[None, :]
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)

    def field(self, z_t, t, context):
        return self.velocity(torch.cat(
            [z_t, self.time_embedding(t), context], dim=1))

    def loss(self, features, z, mask):
        context = self.encode(features)
        z0 = torch.randn_like(z)
        t = torch.rand(len(z), device=z.device)
        z_t = (1 - t[:, None]) * z0 + t[:, None] * z
        error = (self.field(z_t, t, context) - (z - z0)) ** 2
        return (error * mask).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, features, n_samples: int, steps: int = 64):
        context = self.encode(features)                 # (B, C)
        batch = context.shape[0]
        context = context.repeat_interleave(n_samples, dim=0)
        z = torch.randn(batch * n_samples, self.n_parameters,
                        device=features.device)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((len(z),), step * dt, device=z.device)
            v1 = self.field(z, t, context)
            z_mid = z + dt * v1
            v2 = self.field(z_mid, t + dt, context)
            z = z + 0.5 * dt * (v1 + v2)
        return z.view(batch, n_samples, self.n_parameters)


class WTAPosterior(nn.Module):
    def __init__(self, n_in: int, n_parameters: int, *, heads: int = 16,
                 hidden: int = 1024, context: int = 512, relax: float = 0.05):
        super().__init__()
        self.n_parameters, self.heads, self.relax = n_parameters, heads, relax
        self.encode = encoder(n_in, hidden, context)
        self.head = nn.Sequential(nn.Linear(context, 1024), nn.GELU(),
                                  nn.Linear(1024, heads * n_parameters))

    def predict(self, features):
        return self.head(self.encode(features)).view(
            -1, self.heads, self.n_parameters)

    def loss(self, features, z, mask):
        prediction = self.predict(features)
        error = ((prediction - z[:, None, :]) ** 2 * mask[:, None, :]).sum(-1)
        error = error / mask.sum(-1, keepdim=True).clamp_min(1)
        return (error.min(dim=1).values + self.relax * error.mean(dim=1)).mean()

    @torch.no_grad()
    def sample(self, features, n_samples: int | None = None, steps: int = 0):
        return self.predict(features)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")
