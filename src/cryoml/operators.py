"""Forward-surrogate architectures: parameters (+ geometry, + temperature) ->
2046 standardized signed-log currents in the canonical curve order
(5 idvd then 6 idvg curves, 186 resampled points each).

* ``PaperMLP``: the cold-loop/paper dense GELU MLP (reference).
* ``ResMLP``: pre-LayerNorm residual MLP, for the "is the model too small"
  question at matched training data.
* ``DeepONet``: branch net on the card, trunk net on the bias point
  (curve kind, fixed bias, sweep position) and optionally temperature, so the
  output is a learned basis over the bias plane rather than 2046 unrelated
  outputs.
* ``FNO1d``: Fourier neural operator along each curve's sweep axis; the card
  enters as constant input channels and each curve is one 186-point function.

All take ``inputs`` of shape (B, n_in) whose last column is normalized
temperature when ``temperature=True``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

N_CURVES = 11
N_SWEEP = 186
CURVE_KINDS = (0,) * 5 + (1,) * 6                  # 0 = idvd, 1 = idvg
CURVE_BIAS = (0.37, 0.74, 1.11, 1.48, 1.85, 0.01, 0.37, 0.74, 1.11, 1.48, 1.85)


def point_coordinates() -> torch.Tensor:
    """(2046, 3): curve kind, fixed bias / 1.85, sweep position in [0, 1]."""
    s = torch.linspace(0.0, 1.0, N_SWEEP)
    rows = [torch.stack([torch.full_like(s, float(k)),
                         torch.full_like(s, b / 1.85), s], dim=1)
            for k, b in zip(CURVE_KINDS, CURVE_BIAS)]
    return torch.cat(rows, dim=0)


class PaperMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int = 2046,
                 hidden: tuple[int, ...] = (1024, 1024, 1024, 1024)):
        super().__init__()
        layers, width = [], n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.GELU()]
            width = h
        layers.append(nn.Linear(width, n_out))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


class ResBlock(nn.Module):
    def __init__(self, width: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, expansion * width), nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(expansion * width, width))

    def forward(self, x):
        return x + self.ff(self.norm(x))


class ResMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int = 2046, width: int = 1024,
                 depth: int = 6, expansion: int = 2):
        super().__init__()
        self.inp = nn.Linear(n_in, width)
        self.blocks = nn.Sequential(*[ResBlock(width, expansion)
                                      for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_out))

    def forward(self, inputs):
        return self.out(self.blocks(self.inp(inputs)))


class DeepONet(nn.Module):
    def __init__(self, n_in: int, *, width: int = 1024, depth: int = 6,
                 basis: int = 256, trunk_width: int = 256, trunk_depth: int = 4,
                 temperature: bool = False):
        super().__init__()
        self.temperature = temperature
        self.branch = ResMLP(n_in, basis, width, depth)
        n_y = 3 + (1 if temperature else 0)
        layers, w = [], n_y
        for _ in range(trunk_depth):
            layers += [nn.Linear(w, trunk_width), nn.GELU()]
            w = trunk_width
        layers.append(nn.Linear(w, basis))
        self.trunk = nn.Sequential(*layers)
        # Fourier features of the sweep position help sharp knees.
        self.register_buffer("coords", point_coordinates())
        self.point_bias = nn.Parameter(torch.zeros(self.coords.shape[0]))
        self.scale = 1.0 / math.sqrt(basis)
        self._trunk_cache = None

    def _trunk_key(self):
        # Storage and in-place version of every trunk tensor: a load_state_dict,
        # optimizer step or .to() changes the key and invalidates the cache.
        tensors = [self.coords, *self.trunk.parameters()]
        return tuple((t.data_ptr(), t._version) for t in tensors)

    def trunk_basis(self, temperature: torch.Tensor | None):
        if not self.temperature:
            if self.training or torch.is_grad_enabled():
                return self.trunk(self.coords)[None]                # (1, P, K)
            # Without temperature the trunk does not depend on the input, so
            # inference (e.g. the batch-1 extraction search) reuses it.
            key = self._trunk_key()
            if self._trunk_cache is None or self._trunk_cache[0] != key:
                self._trunk_cache = (key, self.trunk(self.coords)[None])
            return self._trunk_cache[1]
        # Temperature is an input condition, never optimized: detach it.
        unique, inverse = torch.unique(temperature.detach(), return_inverse=True)
        y = torch.cat([self.coords[None].expand(len(unique), -1, -1),
                       unique[:, None, None].expand(-1, self.coords.shape[0], 1)],
                      dim=2)
        return self.trunk(y)[inverse]                               # (B, P, K)

    def forward(self, inputs):
        coefficients = self.branch(inputs)                          # (B, K)
        basis = self.trunk_basis(inputs[:, -1] if self.temperature else None)
        out = torch.einsum("bk,bpk->bp", coefficients,
                           basis.expand(len(inputs), -1, -1))
        return out * self.scale + self.point_bias


class SpectralConv1d(nn.Module):
    """Truncated Fourier layer written with real DFT matrices: identical to
    rfft -> keep ``modes`` -> complex channel mixing -> irfft, but as GEMMs,
    which are far faster than FFT kernels on MPS for a 186-point axis."""

    def __init__(self, channels: int, modes: int, length: int = N_SWEEP):
        super().__init__()
        self.modes = modes
        scale = 1.0 / channels
        self.w_re = nn.Parameter(scale * torch.randn(modes, channels, channels))
        self.w_im = nn.Parameter(scale * torch.randn(modes, channels, channels))
        n = torch.arange(length, dtype=torch.float64)
        k = torch.arange(modes, dtype=torch.float64)
        angle = 2 * math.pi * n[:, None] * k[None, :] / length       # (L, M)
        self.register_buffer("fwd_cos", torch.cos(angle).float())
        self.register_buffer("fwd_sin", (-torch.sin(angle)).float())
        # irfft of a spectrum holding only the first ``modes`` bins.
        weight = torch.full((modes,), 2.0, dtype=torch.float64)
        weight[0] = 1.0
        if length % 2 == 0 and modes > length // 2:
            weight[length // 2] = 1.0
        self.register_buffer("inv_cos", (weight[:, None] * torch.cos(angle.T)
                                         / length).float())
        self.register_buffer("inv_sin", (-weight[:, None] * torch.sin(angle.T)
                                         / length).float())

    def forward(self, x):                                           # (N, C, L)
        a = x @ self.fwd_cos                                        # (N, C, M)
        b = x @ self.fwd_sin
        a, b = a.permute(2, 0, 1), b.permute(2, 0, 1)               # (M, N, C)
        out_re = torch.bmm(a, self.w_re) - torch.bmm(b, self.w_im)  # (M, N, O)
        out_im = torch.bmm(a, self.w_im) + torch.bmm(b, self.w_re)
        out_re, out_im = out_re.permute(1, 2, 0), out_im.permute(1, 2, 0)
        return out_re @ self.inv_cos + out_im @ self.inv_sin        # (N, O, L)


class FNO1d(nn.Module):
    """Conditioned FNO over the sweep axis of each of the 11 curves."""

    def __init__(self, n_in: int, *, channels: int = 96, modes: int = 32,
                 layers: int = 4, card_width: int = 512):
        super().__init__()
        self.card = ResMLP(n_in, channels, card_width, 3)
        self.curve_embedding = nn.Embedding(N_CURVES, channels)
        self.lift = nn.Conv1d(channels + 1, channels, 1)
        self.spectral = nn.ModuleList(SpectralConv1d(channels, modes)
                                      for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(channels, channels, 1)
                                       for _ in range(layers))
        self.film = nn.ModuleList(nn.Linear(channels, 2 * channels)
                                  for _ in range(layers))
        self.project = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.GELU(),
                                     nn.Conv1d(channels, 1, 1))
        self.register_buffer("sweep", torch.linspace(0, 1, N_SWEEP))

    def forward(self, inputs):
        batch = len(inputs)
        card = self.card(inputs)                                    # (B, C)
        curve = self.curve_embedding.weight                         # (11, C)
        cond = (card[:, None, :] + curve[None]).reshape(batch * N_CURVES, -1)
        x = torch.cat([cond[:, :, None].expand(-1, -1, N_SWEEP),
                       self.sweep[None, None].expand(len(cond), 1, -1)], dim=1)
        x = self.lift(x)
        for spectral, pointwise, film in zip(self.spectral, self.pointwise,
                                             self.film):
            gamma, beta = film(cond).chunk(2, dim=1)
            x = x + torch.nn.functional.gelu(
                (spectral(x) + pointwise(x)) * (1 + gamma[:, :, None])
                + beta[:, :, None])
        return self.project(x).reshape(batch, N_CURVES * N_SWEEP)


def build(name: str, n_in: int, temperature: bool = False) -> nn.Module:
    if name == "paper_mlp":
        return PaperMLP(n_in)
    if name == "paper_mlp_wide":
        return PaperMLP(n_in, hidden=(2048,) * 6)
    if name == "resmlp":
        return ResMLP(n_in, width=1024, depth=6)
    if name == "resmlp_large":
        return ResMLP(n_in, width=2048, depth=8)
    if name == "deeponet":
        return DeepONet(n_in, temperature=temperature)
    if name == "fno":
        return FNO1d(n_in, channels=64, modes=24)
    raise ValueError(name)


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
