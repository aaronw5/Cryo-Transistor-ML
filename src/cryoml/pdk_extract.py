"""Paper-exact BSIM4 7-parameter extraction against the corrected NGSpice chain.

Theta layout (order fixed): vth0, u0, nfactor, vsat, delta, rdsw, eta0.

Bounds are built *around each bin's published values* (the published fits
use wildly different scales per bin — e.g. pmos vsat up to 7e8 — so global
boxes don't work). vth0/eta0 get additive windows (sign changes allowed);
the positive scale-like params get log-space multiplicative windows.

The optimization objective includes every nonzero-denominator measured curve
and matches the paper companion notebook's all-curve RRMS definition.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .data_io import Curve
from .metrics import device_rrms
from .spice_pdk import (LIB_FILE, PDK77K_DIR, _CORNER_BASENAME, _MODEL, _SUBCKT,
                        _force_bin, ensure_pdk77k, simulate_pdk)
from .utils import get_logger

logger = get_logger("cryoml.pdk_extract")

PARAMS7 = ("vth0", "u0", "nfactor", "vsat", "delta", "rdsw", "eta0")
PENALTY = 10.0


@dataclass
class FlatCurves:
    slices: list[tuple[int, int]]
    kept: list[int]
    denominators: np.ndarray
    n_total_curves: int


def flatten_paper_curves(curves: list[Curve]) -> FlatCurves:
    """Build the paper-exact optimization layout.

    All-zero curves have RRMS 0.0 in the companion notebook, independent of
    simulation output, so they contribute no optimization residual.
    """
    slices, kept, denominators = [], [], []
    cur = 0
    for i, curve in enumerate(curves):
        meas = np.asarray(curve.Id, dtype=np.float64)
        n = len(meas)
        denominator = float(np.mean(np.abs(meas))) if n else 0.0
        if n == 0 or denominator <= 0 or not np.isfinite(denominator):
            continue
        slices.append((cur, cur + n))
        kept.append(i)
        denominators.append(denominator)
        cur += n
    return FlatCurves(
        slices=slices,
        kept=kept,
        denominators=np.asarray(denominators, dtype=np.float64),
        n_total_curves=len(curves),
    )


# ---------------------------------------------------------------------------
# Published per-bin parameter readback (authoritative, via showmod)
# ---------------------------------------------------------------------------
def read_bin_params(dev_type: str, L_um: float, W_um: float,
                    bin_index: int) -> dict[str, float]:
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    base = (PDK77K_DIR / _CORNER_BASENAME[dev_type]).read_text(errors="replace")
    text = _force_bin(base, dev_type, bin_index)
    with tempfile.TemporaryDirectory(prefix="cryoml_showmod_") as td_s:
        td = Path(td_s)
        cp = td / _CORNER_BASENAME[dev_type]
        cp.write_text(text)
        lib = LIB_FILE.read_text().replace(
            (PDK77K_DIR / _CORNER_BASENAME[dev_type]).as_posix(), cp.as_posix())
        lp = td / "lib.spice"
        lp.write_text(lib)
        sign = -1 if dev_type == "pmos" else 1
        area_um2 = W_um * 0.24
        perimeter_um = W_um + 2 * 0.24
        deck = td / "d.sp"
        deck.write_text(
            f'* showmod readback\n.options scale=1.0\n.lib "{lp.as_posix()}" tt_77k\n'
            ".options temp=-196.15\n"
            f"Xm1 nd ng 0 0 {_SUBCKT[dev_type]} l={L_um:.6g}u w={W_um:.6g}u nf=1 "
            f"ad={area_um2:.6g}p as={area_um2:.6g}p "
            f"pd={perimeter_um:.6g}u ps={perimeter_um:.6g}u\n"
            f"VG ng 0 DC {1.0 * sign}\nVD nd 0 DC {1.0 * sign}\n.op\n"
            ".control\nrun\n"
            f"showmod m.xm1.m{_SUBCKT[dev_type]} : {' '.join(PARAMS7)}\n"
            "quit\n.endc\n.end\n")
        out = subprocess.run(["ngspice", "-b", str(deck)], capture_output=True,
                             text=True, timeout=30).stdout
    vals: dict[str, float] = {}
    for p in PARAMS7:
        m = re.search(rf"^\s*{p}\s+([0-9eE.+\-]+)\s*$", out, re.MULTILINE)
        if m:
            vals[p] = float(m.group(1))
    missing = [p for p in PARAMS7 if p not in vals]
    if missing:
        raise RuntimeError(f"showmod readback missing {missing} for "
                           f"{dev_type} bin {bin_index}: {out[-500:]}")
    return vals


# ---------------------------------------------------------------------------
# Bounded theta box around the published bin values
# ---------------------------------------------------------------------------
@dataclass
class ThetaBox:
    """Sigmoid-bounded z <-> physical transform, per device bin."""
    dev_type: str
    bin_index: int
    published: dict[str, float]
    lo_t: np.ndarray = field(init=False)   # transform-space lower bounds
    hi_t: np.ndarray = field(init=False)
    is_log: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        # absolute fallback boxes for log params whose published value is
        # zero/negative (some bins ship e.g. vsat=-2.6e4 or delta=-0.012;
        # ngspice clamps them internally, but log bounds need positives)
        fallback = {
            "u0": (1e-3, 0.5),
            "nfactor": (0.5, 120.0),
            "vsat": (1e4, 5e9),
            "delta": (1e-4, 0.5),
            "rdsw": (0.05, 1e5),
        }
        lo, hi, isl = [], [], []
        for p in PARAMS7:
            v = float(self.published[p])
            if p == "vth0":
                lo.append(v - 0.6); hi.append(v + 0.6); isl.append(False)
                continue
            if p == "eta0":
                span = max(0.3, 2.0 * abs(v))
                lo.append(v - span); hi.append(v + span); isl.append(False)
                continue
            if v <= 0 or not np.isfinite(v):
                l, h = fallback[p]
            elif p == "vsat":
                l = min(0.2 * v, 3.0e4); h = max(5.0 * v, 3.0e6)
            elif p == "rdsw":
                l = max(0.05, 0.02 * v); h = max(20.0 * v, 100.0)
            elif p == "delta":
                l = 0.05 * v; h = 40.0 * v
            else:  # u0, nfactor: positive, log x[0.1, 10]
                l = 0.1 * v; h = 10.0 * v
            lo.append(np.log(l)); hi.append(np.log(h)); isl.append(True)
        self.lo_t = np.array(lo, dtype=np.float64)
        self.hi_t = np.array(hi, dtype=np.float64)
        self.is_log = np.array(isl, dtype=bool)

    def z_to_params(self, z: np.ndarray) -> dict[str, float]:
        z = np.asarray(z, dtype=np.float64)
        s = 1.0 / (1.0 + np.exp(-z))
        t = self.lo_t + (self.hi_t - self.lo_t) * s
        x = np.where(self.is_log, np.exp(t), t)
        return {p: float(x[i]) for i, p in enumerate(PARAMS7)}

    def params_to_z(self, params: dict[str, float]) -> np.ndarray:
        x = np.array([float(params[p]) for p in PARAMS7], dtype=np.float64)
        t = np.where(self.is_log, np.log(np.maximum(x, 1e-300)), x)
        frac = (t - self.lo_t) / (self.hi_t - self.lo_t)
        frac = np.clip(frac, 1e-6, 1 - 1e-6)
        return np.clip(np.log(frac / (1 - frac)), -5.5, 5.5)

    @property
    def z_published(self) -> np.ndarray:
        return self.params_to_z(self.published)


def theta_box_for(dev_type: str, L_um: float, W_um: float,
                  bin_index: int) -> ThetaBox:
    pub = read_bin_params(dev_type, L_um, W_um, bin_index)
    return ThetaBox(dev_type=dev_type, bin_index=bin_index, published=pub)


# ---------------------------------------------------------------------------
# Metric evaluation + residual against the PDK backend
# ---------------------------------------------------------------------------
def eval_params(dev_type: str, L_um: float, W_um: float, bin_index: int,
                curves: list[Curve], params: dict[str, float]):
    sims = simulate_pdk(dev_type, L_um, W_um, curves, params=params,
                        bin_index=bin_index)
    meas = [c.Id for c in curves]
    return device_rrms(sims, meas), sims


def residual_fn(z: np.ndarray, box: ThetaBox, dev_type: str, L_um: float,
                W_um: float, bin_index: int, curves: list[Curve],
                flat: FlatCurves) -> np.ndarray:
    params = box.z_to_params(z)
    sims = simulate_pdk(dev_type, L_um, W_um, curves, params=params,
                        bin_index=bin_index)
    out = np.empty(max(len(flat.kept), 1), dtype=np.float64)
    for oi, ((a, b), ci, denominator) in enumerate(
        zip(flat.slices, flat.kept, flat.denominators)
    ):
        n = b - a
        s = np.asarray(sims[ci], dtype=np.float64)[:n]
        mvals = np.asarray(curves[ci].Id, dtype=np.float64)[:n]
        rrms = float(np.sqrt(np.mean((s - mvals) ** 2)) / denominator)
        out[oi] = (
            np.sqrt(rrms / max(flat.n_total_curves, 1))
            if np.isfinite(rrms) and rrms >= 0
            else PENALTY
        )
    if not flat.kept:
        out[0] = 0.0
    return out


# ---------------------------------------------------------------------------
# Multistart finite-difference extraction (per device)
# ---------------------------------------------------------------------------
@dataclass
class PdkFitResult:
    dev_type: str
    L_um: float
    W_um: float
    bin_index: int
    params: dict[str, float]
    rrms: float
    start_rrms: float
    method: str
    n_starts: int
    runtime_seconds: float
    sims: list = field(default_factory=list)


def fd_extract_device(
    dev_type: str,
    L_um: float,
    W_um: float,
    bin_index: int,
    curves: list[Curve],
    n_starts: int = 6,
    sigma_z: float = 1.0,
    max_nfev: int = 150,
    seed: int = 0,
    extra_start_zs: list[np.ndarray] | None = None,
    method_label: str = "pdk_fd_multistart",
) -> PdkFitResult:
    t0 = time.time()
    box = theta_box_for(dev_type, L_um, W_um, bin_index)
    flat = flatten_paper_curves(curves)
    start_metrics, _ = eval_params(dev_type, L_um, W_um, bin_index, curves,
                                   box.published)
    best = {"params": box.published, "m": start_metrics, "which": "published"}

    rng = np.random.default_rng(seed)
    z0 = box.z_published
    starts = [z0]
    if extra_start_zs:
        starts += [np.asarray(z, dtype=np.float64) for z in extra_start_zs]
    while len(starts) < n_starts:
        starts.append(z0 + rng.normal(0, sigma_z, size=len(PARAMS7)))

    f = lambda z: residual_fn(z, box, dev_type, L_um, W_um, bin_index, curves, flat)
    for si, zs in enumerate(starts):
        try:
            sol = least_squares(f, zs, method="trf", jac="2-point",
                                diff_step=2e-2, max_nfev=max_nfev)
        except Exception as e:  # noqa: BLE001
            logger.debug("start %d failed: %s", si, e)
            continue
        params = box.z_to_params(np.asarray(sol.x, dtype=np.float64))
        m, _ = eval_params(dev_type, L_um, W_um, bin_index, curves, params)
        if np.isfinite(m["rrms"]) and m["rrms"] < best["m"]["rrms"]:
            best = {"params": params, "m": m, "which": f"start{si}"}

    final_m, sims = eval_params(dev_type, L_um, W_um, bin_index, curves,
                                best["params"])
    return PdkFitResult(
        dev_type=dev_type, L_um=L_um, W_um=W_um, bin_index=bin_index,
        params=best["params"],
        rrms=float(final_m["rrms"]),
        start_rrms=float(start_metrics["rrms"]),
        method=method_label, n_starts=len(starts),
        runtime_seconds=time.time() - t0, sims=sims,
    )
