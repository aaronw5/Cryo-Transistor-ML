"""E2: one-pass I-V -> seven-parameter network with misspecification-style
output augmentation (see out/research/explore/E2/prereg.json).

Training data are the cold round-1 seven-parameter banks
(``data/processed/cold_loop/seven_r1``), put in the canonical curve layout of
``foundation.canonical_point_order``: five Id-Vd curves (|Vg| = 0.37 ... 1.85,
|Vd| swept 0 ... 1.85 in 10 mV steps) then six Id-Vg curves (|Vd| = 0.01 ...
1.85, |Vg| swept 0 ... 1.85).  All 18 devices share this grid (asserted).

Augmentations act on simulated current magnitudes A >= 0 (all banks store
positive currents for both families):

* series resistance R per side:  A / (1 + R (gm + 2 gds)),
  R = rho / (W_um mult), rho log-uniform in [10, 10^4.5] ohm um,
  gm = dA/d|Vg|, gds = dA/d|Vd|.  Each family supplies its own sweep
  derivative by finite differences; the other derivative is interpolated
  from the other family as a relative conductance (g / A).
* per-curve heating: dT_i proportional to the curve's peak
  |I Vd|; |Vth| lowered by kv dT_i (factor exp(gm/A kv dT_i), exponent capped
  at 3) and mobility factor (77 / (77 + dT_i))^beta.
* measurement: per-curve gain, Vd-dependent multiplicative
  noise, additive floor.

Nothing in this module reads measured currents except ``target_currents``,
which is used only at evaluation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import OUT_DIR, PROCESSED_DIR
from ..data_io import load_device_curves
from ..devices import PAPER_DEVICES
from ..foundation import I_REF, canonical_point_order
from ..metrics import clean_current
from ..parameter_space import ParameterBox
from ..spice_pdk import device_mult
from ..utils import device_tag

ROOT = Path(__file__).resolve().parents[3]
E2_DIR = OUT_DIR / "research" / "explore" / "E2"
SPLIT_JSON = OUT_DIR / "research" / "explore" / "real_split.json"
BANK_DIR = PROCESSED_DIR / "cold_loop" / "seven_r1"
SPEC_JSON = OUT_DIR / "cold_loop" / "seven" / "round_1" / "spec.json"
TIER_A_DIR = OUT_DIR / "research" / "sim_targets"
TIER_C_DIR = OUT_DIR / "research" / "sim_targets_sky130c"
NAMES = ("vth0", "u0", "nfactor", "vsat", "delta", "rdsw", "eta0")
TAGS = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
N_IDVD, N_IDVG, N_PTS = 5, 6, 186
STEP = 0.01
REL_CAP = 200.0  # 1/V cap on relative conductances


# ---------------------------------------------------------------- boxes ---

def load_spec() -> dict:
    return json.loads(SPEC_JSON.read_text())


def box_for(spec: dict, tag: str) -> ParameterBox:
    """Same construction as ``scripts/research/cold_loop.box_for``."""
    names = tuple(spec["selected_parameters"])
    flags = spec["log_scale_by_device"][tag]
    return ParameterBox(names=names, published=spec["published_by_device"][tag],
                        bounds=spec["bounds_by_device"][tag],
                        log_scale=tuple(bool(flags[n]) for n in names))


def theta_to_local(box: ParameterBox, theta: np.ndarray) -> np.ndarray:
    """Vectorized ``ParameterBox.params_to_local`` (all-log or linear)."""
    theta = np.asarray(theta, dtype=np.float64)
    local = 2.0 * (theta - box.lo) / np.where(box.active, box.hi - box.lo, 1.0) - 1.0
    if np.any(box.log):
        mag = np.log(np.maximum(box.sign[box.log] * theta[:, box.log], 1e-300))
        lo, hi = box.log_lo[box.log], box.log_hi[box.log]
        local[:, box.log] = 2.0 * (mag - lo) / (hi - lo) - 1.0
    local[:, ~box.active] = 0.0
    return np.clip(local, -1.0, 1.0)


def load_split() -> dict:
    return json.loads(SPLIT_JSON.read_text())


# --------------------------------------------------------------- layout ---

@dataclass
class Layout:
    vg_idvd: np.ndarray   # (5,) |Vg| of the Id-Vd curves
    vd_idvg: np.ndarray   # (6,) |Vd| of the Id-Vg curves
    sweep: np.ndarray     # (186,) shared sweep grid
    abs_vd: np.ndarray    # (2046,) |Vd| per canonical point
    w_vg: np.ndarray      # (186, 5) interpolation over |Vg| from the Id-Vd biases
    w_vd: np.ndarray      # (186, 6) interpolation over |Vd| from the Id-Vg biases
    idx_vd: np.ndarray    # (6,) sweep index of each Id-Vg bias |Vd|
    idx_vg: np.ndarray    # (5,) sweep index of each Id-Vd bias |Vg|


def _interp_matrix(grid: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Linear interpolation weights with constant extrapolation."""
    out = np.zeros((len(grid), len(knots)))
    for row, x in enumerate(grid):
        if x <= knots[0]:
            out[row, 0] = 1.0
        elif x >= knots[-1]:
            out[row, -1] = 1.0
        else:
            j = int(np.searchsorted(knots, x) - 1)
            t = (x - knots[j]) / (knots[j + 1] - knots[j])
            out[row, j], out[row, j + 1] = 1 - t, t
    return out


def canonical(data) -> tuple[np.ndarray, list[dict]]:
    return canonical_point_order(data["kinds"], data["fixeds"], data["slices"])


def build_layout(data) -> Layout:
    order, desc = canonical(data)
    vg = np.abs(np.asarray(data["Vg"], dtype=np.float64))[order]
    vd = np.abs(np.asarray(data["Vd"], dtype=np.float64))[order]
    kinds = [r["kind"] for r in desc]
    assert kinds == ["idvd"] * N_IDVD + ["idvg"] * N_IDVG, kinds
    assert len(order) == (N_IDVD + N_IDVG) * N_PTS
    sweep = np.round(np.arange(N_PTS) * STEP, 10)
    vg_idvd, vd_idvg = [], []
    for c, row in enumerate(desc):
        sl = slice(c * N_PTS, (c + 1) * N_PTS)
        if row["kind"] == "idvd":
            assert np.allclose(vd[sl], sweep, atol=1e-9), "Id-Vd sweep grid"
            assert np.ptp(vg[sl]) < 1e-9
            vg_idvd.append(vg[sl][0])
        else:
            assert np.allclose(vg[sl], sweep, atol=1e-9), "Id-Vg sweep grid"
            assert np.ptp(vd[sl]) < 1e-9
            vd_idvg.append(vd[sl][0])
    vg_idvd, vd_idvg = np.asarray(vg_idvd), np.asarray(vd_idvg)
    return Layout(vg_idvd=vg_idvd, vd_idvg=vd_idvg, sweep=sweep, abs_vd=vd,
                  w_vg=_interp_matrix(sweep, vg_idvd),
                  w_vd=_interp_matrix(sweep, vd_idvg),
                  idx_vd=np.round(vd_idvg / STEP).astype(int),
                  idx_vg=np.round(vg_idvd / STEP).astype(int))


# ---------------------------------------------------------- training data ---

@dataclass
class Bank:
    currents: np.ndarray      # (N, 2046) float32 canonical
    local: np.ndarray         # (N, 7) float32
    device_ids: np.ndarray    # (N,)
    curve_scales: np.ndarray  # (18, 2046) bank-median per-curve max |I|
    r_unit: np.ndarray        # (18,) 1 / (W_um mult) [1/um]
    layout: Layout


def load_bank(spec: dict, max_per_device: int | None = None) -> Bank:
    currents, locals_, ids, scales, r_unit = [], [], [], [], []
    layout = None
    for d_index, device in enumerate(PAPER_DEVICES):
        tag = TAGS[d_index]
        with np.load(BANK_DIR / f"{tag}.npz", allow_pickle=True) as data:
            dev_layout = build_layout(data)
            if layout is None:
                layout = dev_layout
            else:
                assert np.allclose(layout.vg_idvd, dev_layout.vg_idvd)
                assert np.allclose(layout.vd_idvg, dev_layout.vd_idvg)
            order, _ = canonical(data)
            names = tuple(str(v) for v in data["parameter_names"])
            assert names == NAMES, names
            ok = np.asarray(data["ok"], dtype=bool)
            ids_ok = np.flatnonzero(ok)
            if max_per_device is not None:
                ids_ok = ids_ok[:max_per_device]
            cur = np.asarray(data["IDS"], dtype=np.float32)[ids_ok][:, order]
            theta = np.asarray(data["THETA"], dtype=np.float64)[ids_ok]
        finite = np.all(np.isfinite(cur), axis=1)
        cur, theta = cur[finite], theta[finite]
        box = box_for(spec, tag)
        local = theta_to_local(box, theta)
        currents.append(cur)
        locals_.append(local.astype(np.float32))
        ids.append(np.full(len(cur), d_index, dtype=np.int64))
        peak = np.abs(cur).reshape(len(cur), N_IDVD + N_IDVG, N_PTS).max(axis=2)
        med = np.maximum(np.median(peak, axis=0), 1e-12)
        scales.append(np.repeat(med, N_PTS).astype(np.float32))
        r_unit.append(1.0 / (device.W_um * device_mult(
            device.dev_type, device.L_um, device.W_um)))
    return Bank(currents=np.concatenate(currents), local=np.concatenate(locals_),
                device_ids=np.concatenate(ids), curve_scales=np.stack(scales),
                r_unit=np.asarray(r_unit, dtype=np.float64), layout=layout)


# ---------------------------------------------------------- augmentation ---

ARMS = ("N0", "M", "S")
AUG_PROB = 0.8


def _diff(a: torch.Tensor) -> torch.Tensor:
    """d/dV along the last axis (10 mV grid), one-sided at the ends."""
    g = torch.empty_like(a)
    g[..., 1:-1] = (a[..., 2:] - a[..., :-2]) / (2 * STEP)
    g[..., 0] = (a[..., 1] - a[..., 0]) / STEP
    g[..., -1] = (a[..., -1] - a[..., -2]) / STEP
    return g


class Augmenter(nn.Module):
    def __init__(self, layout: Layout, r_unit: np.ndarray):
        super().__init__()
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
        self.register_buffer("w_vg", f(layout.w_vg))          # (186, 5)
        self.register_buffer("w_vd", f(layout.w_vd))          # (186, 6)
        self.register_buffer("sweep", f(layout.sweep))        # (186,)
        self.register_buffer("vd_idvg", f(layout.vd_idvg))    # (6,)
        self.register_buffer("abs_vd", f(layout.abs_vd))      # (2046,)
        self.register_buffer("r_unit", f(r_unit))             # (18,)
        self.idx_vd = torch.as_tensor(layout.idx_vd, dtype=torch.long)
        self.idx_vg = torch.as_tensor(layout.idx_vg, dtype=torch.long)

    def conductances(self, a_vd: torch.Tensor, a_vg: torch.Tensor):
        """gm and gds at every point of both families (B, 5, 186)/(B, 6, 186)."""
        eps = 1e-18
        gds_vd = _diff(a_vd).clamp_min(0)                     # own, Id-Vd
        gm_vg = _diff(a_vg).clamp_min(0)                      # own, Id-Vg
        rel_gds = (gds_vd / (a_vd + eps)).clamp(0, REL_CAP)   # (B,5,186)
        rel_gm = (gm_vg / (a_vg + eps)).clamp(0, REL_CAP)     # (B,6,186)
        # gds at Id-Vg points: Id-Vd rel gds at |Vd| = Vd_k, interpolated in |Vg|.
        knots = rel_gds[:, :, self.idx_vd.to(a_vd.device)]    # (B,5,6)
        gds_vg = torch.einsum("bjk,gj->bkg", knots, self.w_vg) * a_vg
        # gm at Id-Vd points: Id-Vg rel gm at |Vg| = Vg_j, interpolated in |Vd|.
        knots = rel_gm[:, :, self.idx_vg.to(a_vg.device)]     # (B,6,5)
        gm_vd = torch.einsum("bkj,dk->bjd", knots, self.w_vd) * a_vd
        return gm_vd, gds_vd, gm_vg, gds_vg

    def structured(self, current: torch.Tensor, device_ids: torch.Tensor,
                   gen: torch.Generator) -> torch.Tensor:
        b = current.shape[0]
        dev = current.device
        u = lambda *s: torch.rand(*s, generator=gen, device=dev)  # noqa: E731
        # rho log-uniform in [10, 10^4.5] ohm um (independent of the card).
        rho = 10.0 ** (1.0 + 3.5 * u(b))
        dt_max = 40.0 * u(b, 1)
        kv = 0.5e-3 + 0.5e-3 * u(b, 1)
        beta = 1.0 + u(b, 1)
        return self.structured_fixed(current, device_ids, rho, dt_max, kv, beta)

    def structured_fixed(self, current, device_ids, rho, dt_max, kv, beta):
        """Series R (rho / (W mult) per side) then heating, given draws:
        rho (B,), dt_max, kv, beta (B, 1)."""
        b = current.shape[0]
        a = current.abs().view(b, N_IDVD + N_IDVG, N_PTS)
        a_vd, a_vg = a[:, :N_IDVD], a[:, N_IDVD:]
        gm_vd, gds_vd, gm_vg, gds_vg = self.conductances(a_vd, a_vg)
        # (1) series resistance per side, R = rho / (W mult).
        r = (rho * self.r_unit[device_ids])[:, None, None]
        den_vd = 1 + r * (gm_vd + 2 * gds_vd)
        den_vg = 1 + r * (gm_vg + 2 * gds_vg)
        a_vd, a_vg = a_vd / den_vd, a_vg / den_vg
        relgm_vd = (gm_vd / den_vd / (a_vd * den_vd + 1e-18)).clamp(0, REL_CAP)
        relgm_vg = (gm_vg / den_vg / (a_vg * den_vg + 1e-18)).clamp(0, REL_CAP)
        # (2) per-curve heating.
        p_vd = (a_vd * self.sweep).amax(dim=2)                       # (B,5)
        p_vg = a_vg.amax(dim=2) * self.vd_idvg                       # (B,6)
        power = torch.cat([p_vd, p_vg], dim=1)
        share = power / power.amax(dim=1, keepdim=True).clamp_min(1e-30)
        d_t = dt_max * share                                          # (B,11)
        mob = (77.0 / (77.0 + d_t)) ** beta
        shift = kv * d_t                                              # volts
        a_vd = a_vd * mob[:, :N_IDVD, None] * torch.exp(
            (relgm_vd * shift[:, :N_IDVD, None]).clamp(max=3.0))
        a_vg = a_vg * mob[:, N_IDVD:, None] * torch.exp(
            (relgm_vg * shift[:, N_IDVD:, None]).clamp(max=3.0))
        out = torch.cat([a_vd, a_vg], dim=1).reshape(b, -1)
        return out * torch.sign(current).where(current != 0,
                                               torch.ones_like(current))

    def measurement(self, current: torch.Tensor,
                    gen: torch.Generator) -> torch.Tensor:
        b, p = current.shape
        dev = current.device
        u = lambda *s: torch.rand(*s, generator=gen, device=dev)  # noqa: E731
        n = lambda *s: torch.randn(*s, generator=gen, device=dev)  # noqa: E731
        sg = 0.03 * u(b, 1)
        gain = (1 + sg * n(b, N_IDVD + N_IDVG)).repeat_interleave(N_PTS, dim=1)
        sn = 0.02 * u(b, 1)
        noise = 1 + sn * (1 + self.abs_vd / 1.8) * n(b, p)
        level = 10.0 ** (-13.0 + 3.0 * u(b, 1))
        return current * gain * noise + level * n(b, p)

    def forward(self, current, device_ids, arm: str, gen: torch.Generator):
        if arm == "N0":
            return current
        out = current
        if arm == "S":
            out = self.structured(out, device_ids, gen)
        out = self.measurement(out, gen)
        keep = torch.rand(current.shape[0], 1, generator=gen,
                          device=current.device) < AUG_PROB
        return torch.where(keep, out, current)


# --------------------------------------------------------------- network ---

def signed_log(x):
    return torch.sign(x) * torch.log1p(torch.abs(x) / I_REF)


class Features(nn.Module):
    def __init__(self, slog_mean, slog_std, curve_scales, n_devices=18):
        super().__init__()
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
        self.register_buffer("slog_mean", f(slog_mean))
        self.register_buffer("slog_std", f(slog_std))
        self.register_buffer("curve_scales", f(curve_scales))
        self.n_devices = n_devices

    @property
    def n_features(self):
        return 2 * self.slog_mean.shape[0] + self.n_devices

    def standardized_slog(self, current):
        return torch.clamp((signed_log(current) - self.slog_mean) / self.slog_std,
                           -10, 10)

    def forward(self, current, device_ids):
        lin = torch.clamp(current / self.curve_scales[device_ids], -20, 20)
        onehot = nn.functional.one_hot(device_ids, self.n_devices).float()
        return torch.cat([self.standardized_slog(current), lin, onehot], dim=1)


class DirectNet(nn.Module):
    def __init__(self, n_in: int, n_out: int = 7, hidden=(1024, 1024, 512)):
        super().__init__()
        layers, width = [], n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.GELU()]
            width = h
        layers.append(nn.Linear(width, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------- targets ---

GROUPS = ("real", "a0", "a1", "c0", "c1")


def override_path(group: str) -> Path | None:
    if group == "real":
        return None
    base = TIER_A_DIR if group[0] == "a" else TIER_C_DIR
    return base / f"set_{int(group[1])}.npz"


def group_tags(group: str) -> list[str]:
    return list(load_split()["REAL_DEV"]) if group == "real" else list(TAGS)


def target_currents(group: str, tag: str) -> np.ndarray:
    """Target currents (clean_current per curve) in canonical order."""
    device = PAPER_DEVICES[TAGS.index(tag)]
    if group == "real":
        curves = load_device_curves(device)
        parts = [np.asarray(c.Id, dtype=np.float64) for c in curves]
    else:
        with np.load(override_path(group)) as data:
            n = sum(1 for k in data.files if k.startswith(tag + "__")
                    and "unheated" not in k)
            parts = [np.asarray(data[f"{tag}__{i}"], dtype=np.float64)
                     for i in range(n)]
    with np.load(BANK_DIR / f"{tag}.npz", allow_pickle=True) as data:
        order, _ = canonical(data)
        slices = np.asarray(data["slices"])
        if group == "real":
            meas = np.asarray(data["meas"], dtype=np.float64)
            assert np.allclose(np.concatenate(parts), meas, rtol=1e-9,
                               atol=1e-18), f"{tag}: load order differs from bank"
    assert [len(p) for p in parts] == [int(b - a) for a, b in slices]
    full = np.concatenate([clean_current(p) for p in parts])
    return full[order]


def truth_local(group: str, tag: str, spec: dict) -> np.ndarray | None:
    if group == "real":
        return None
    base = TIER_A_DIR if group[0] == "a" else TIER_C_DIR
    truth = json.loads((base / "truth.json").read_text())
    row = truth["sets"][group[1]][tag]
    card = row["params"] if group[0] == "a" else row["card"]
    params = {n: float(card[n]) for n in NAMES}
    return box_for(spec, tag).params_to_local(params)


def truth_params(group: str, tag: str) -> dict | None:
    if group == "real":
        return None
    base = TIER_A_DIR if group[0] == "a" else TIER_C_DIR
    truth = json.loads((base / "truth.json").read_text())
    row = truth["sets"][group[1]][tag]
    card = row["params"] if group[0] == "a" else row["card"]
    return {n: float(card[n]) for n in NAMES}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=float) + "\n")


def finite_mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(vals)) if vals else math.nan
