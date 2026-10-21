"""I11-1: the 43-parameter card plus the shared bias-only correction (M_bias,
I3-2 recipe) as a compact-model add-on in circuit simulation (sky130 77 K).

Pre-registration: ``out/research/explore/I11_1/prereg.md`` / ``prereg.json``.

The correction is the OSDI-compiled Verilog-A module of
``i3_2_bias.to_verilog_a_bias`` (OpenVAF 23.5.0, ngspice-41 OSDI build).  One
corrected transistor is the sky130 BSIM4 subcircuit with its drain moved to an
internal node ``di`` plus the module instance ``N<name> d g s di``:

    I(d -> s) total = I(sense) * exp(g),  I(sense) = current from d into di,
    g = g(is_pmos, |V(g, s)| / 1.85, |V(d, s)| / 1.85).

This module holds the pure parts (bias grids, g and its partial derivatives,
the Python reference model, deck text, parsers, circuit metrics, the
pre-registered decision) and thin ngspice runners.  The experiment driver is
``scripts/research/explore/i11_1_deploy.py``.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

NG_OSDI = "/Users/anrunw/cryo-ng41/arm64-osdi/bin/ngspice"
OPENVAF = "/Users/anrunw/cryo-ng41/openvaf/bin/openvaf"   # OpenVAF 23.5.0 (OSDI 0.3)
OPENVAF_ARGS = ("--target", "aarch64-apple-darwin")
VA_MODULE = "i3_2_bias_corr"
VDD_NORM = 1.85          # g input normalization (part of the trained correction)
VDD_CIRCUIT = 1.8        # circuit supply
TEMP_K = 77.0
TIGHT = "reltol=1e-9 abstol=1e-18 vntol=1e-12"   # I3-2 F1 A2 tolerances + vntol
STEP = 0.025
GRID_VGS = np.round(np.arange(75) * STEP, 6)              # 0 .. 1.85
GRID_VDS = np.round(-1.85 + np.arange(149) * STEP, 6)     # -1.85 .. 1.85
GM_VDS = (0.05, 0.925, 1.85)       # |Vds| of the gm op-point sets
GDS_VGS = (0.6, 0.9, 1.2, 1.5, 1.85)   # |Vgs| of the gds op-point sets
AC_FREQ = 1e6
I_FLOOR = 1e-9           # A, current floor of the relative-difference checks
Y_NEG = 1e-12            # S, a conductance below -Y_NEG counts as negative
R4 = "reltol=1e-4"       # intermediate tolerance (descriptive)
TIMEOUT_DEVICE_S = 300.0
TIMEOUT_CIRCUIT_S = 120.0
Y_FLOOR = 1e-9           # S, conductance floor of the derivative checks

# Pre-registered thresholds (prereg.json "decision")
THR = {"q1_tight": 1e-5, "q1_default": 1e-3, "q1_vds0_abs": 1e-12,
       "q2_ac": 1e-3, "swap_asym": 1e-3, "runtime_ratio": 2.0, "ro_min_periods": 5}

CONVERGENCE_MESSAGES = ("singular matrix", "timestep too small", "no convergence",
                        "gmin stepping failed", "source stepping failed",
                        "iteration limit reached", "simulation(s) aborted", "error")


# ---------------------------------------------------------------------------
# Correction g and the Python reference model (pure)
# ---------------------------------------------------------------------------
def g_values(corr, is_pmos: bool, vgs, vds) -> np.ndarray:
    """g of a trained ``BiasCorrection`` (columns 0-2 of the E1 features) at
    signed terminal voltages V(g, s), V(d, s)."""
    vgs = np.atleast_1d(np.asarray(vgs, dtype=np.float64))
    vds = np.atleast_1d(np.asarray(vds, dtype=np.float64))
    vgs, vds = np.broadcast_arrays(vgs, vds)
    phi = np.zeros((vgs.size, 6))
    phi[:, 0] = 1.0 if is_pmos else 0.0
    phi[:, 1] = np.abs(vgs.ravel()) / VDD_NORM
    phi[:, 2] = np.abs(vds.ravel()) / VDD_NORM
    return np.asarray(corr.g(phi), dtype=np.float64).reshape(vgs.shape)


def g_partials(corr, is_pmos: bool, vgs, vds, h: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """(dg/dVgs, dg/dVds) in 1/V, central differences in the signed voltages
    (g is smooth except on the lines Vgs = 0 and Vds = 0, where abs() is
    used; there the one-sided values differ in sign)."""
    vgs = np.asarray(vgs, dtype=np.float64)
    vds = np.asarray(vds, dtype=np.float64)
    dgg = (g_values(corr, is_pmos, vgs + h, vds) - g_values(corr, is_pmos, vgs - h, vds)) / (2 * h)
    dgd = (g_values(corr, is_pmos, vgs, vds + h) - g_values(corr, is_pmos, vgs, vds - h)) / (2 * h)
    return dgg, dgd


def python_current(i_card, g) -> np.ndarray:
    """Python reference model: card current x exp(g)."""
    return np.asarray(i_card, dtype=np.float64) * np.exp(np.asarray(g, dtype=np.float64))


def python_admittance(i_card, y_card, g, dg_dv) -> np.ndarray:
    """Small-signal drain admittance of I_card * exp(g) for one excitation:
    exp(g) * Y_card + I_card * exp(g) * dg/dV (Y_card complex; the second
    term is real)."""
    eg = np.exp(np.asarray(g, dtype=np.float64))
    return eg * np.asarray(y_card) + np.asarray(i_card, dtype=np.float64) * eg * np.asarray(dg_dv)


def rel_diff(a, b, floor: float) -> np.ndarray:
    """|a - b| / |b| at points with |b| > floor (NaN elsewhere); a non-finite
    value at a counted point gives inf."""
    a = np.asarray(a)
    b = np.asarray(b)
    out = np.full(np.shape(b), np.nan)
    keep = np.isfinite(b) & (np.abs(b) > floor)
    bad = keep & ~np.isfinite(a)
    ok = keep & ~bad
    out[ok] = np.abs(a[ok] - b[ok]) / np.abs(b[ok])
    out[bad] = np.inf
    return out


def nanmax0(x) -> float:
    """max over finite-or-inf entries, 0.0 when nothing is counted."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(np.max(x)) if x.size else 0.0


def norm_y_err(y_a, y_b, floor: float = Y_FLOOR) -> np.ndarray:
    """|y_a - y_b| / max(|y_b|, floor) (pre-registered derivative error)."""
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    return np.abs(y_a - y_b) / np.maximum(np.abs(y_b), floor)


def threshold_voltage(vgs: np.ndarray, i_abs: np.ndarray, i_crit: float) -> float:
    """|Vgs| at which |I| first reaches ``i_crit`` (log-linear interpolation);
    NaN when never reached."""
    vgs = np.abs(np.asarray(vgs, dtype=np.float64))
    li = np.log10(np.maximum(np.abs(np.asarray(i_abs, dtype=np.float64)), 1e-30))
    lc = math.log10(i_crit)
    above = np.nonzero(li >= lc)[0]
    if above.size == 0:
        return float("nan")
    k = int(above[0])
    if k == 0:
        return float(vgs[0])
    t = (lc - li[k - 1]) / (li[k] - li[k - 1])
    return float(vgs[k - 1] + t * (vgs[k] - vgs[k - 1]))


def sign_changes_introduced(y_corr, y_card, mask, neg: float = Y_NEG) -> int:
    """Points in ``mask`` with y_corr < -neg where y_card >= -neg."""
    y_corr = np.asarray(y_corr)
    y_card = np.asarray(y_card)
    m = np.asarray(mask, bool) & np.isfinite(y_corr) & np.isfinite(y_card)
    return int(np.sum(m & (y_corr < -neg) & (y_card >= -neg)))


def curve_sign_change_introduced(y_corr, y_card, mask, neg: float = Y_NEG) -> bool:
    """Pre-registered curve-level rule: the corrected curve has a negative
    value (< -neg) inside ``mask`` while the card-only curve has none."""
    y_corr = np.asarray(y_corr)
    y_card = np.asarray(y_card)
    m = np.asarray(mask, bool) & np.isfinite(y_corr) & np.isfinite(y_card)
    return bool(np.any(m & (y_corr < -neg)) and not np.any(m & (y_card < -neg)))


def central_diff(x, y) -> np.ndarray:
    """dy/dx by central differences (one-sided at the ends)."""
    return np.gradient(np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------------------
# Circuit metrics (pure)
# ---------------------------------------------------------------------------
def vtc_metrics(vin, vout) -> dict:
    """Switching threshold V_M (first Vout = Vin crossing), maximum |gain|
    (central differences) and the Vin of the maximum."""
    vin = np.asarray(vin, dtype=np.float64)
    vout = np.asarray(vout, dtype=np.float64)
    d = vout - vin
    k = np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    vm = float("nan")
    if k.size:
        i = int(k[0])
        vm = float(vin[i] - d[i] * (vin[i + 1] - vin[i]) / (d[i + 1] - d[i]))
    gain = central_diff(vin, vout)
    j = int(np.nanargmax(np.abs(gain)))
    return {"vm": vm, "gain_max": float(abs(gain[j])), "vin_at_gain_max": float(vin[j]),
            "vout_min": float(np.min(vout)), "vout_max": float(np.max(vout))}


def oscillation(t, v, level: float, t_from: float = 0.0) -> dict:
    """Rising ``level`` crossings of v(t) after ``t_from``: number of full
    periods, median period, frequency, and the relative spread of the
    periods (max - min) / median."""
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    keep = t >= t_from
    t, v = t[keep], v[keep]
    up = np.nonzero((v[:-1] < level) & (v[1:] >= level))[0]
    tc = t[up] + (level - v[up]) * (t[up + 1] - t[up]) / (v[up + 1] - v[up])
    if tc.size < 2:
        return {"n_periods": 0, "period_s": float("nan"), "freq_hz": float("nan"),
                "period_spread": float("nan")}
    per = np.diff(tc)
    med = float(np.median(per))
    return {"n_periods": int(per.size), "period_s": med, "freq_hz": 1.0 / med,
            "period_spread": float((per.max() - per.min()) / med)}


def interp_at(x, y, x0: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    o = np.argsort(x)
    return float(np.interp(x0, x[o], y[o]))


# ---------------------------------------------------------------------------
# Netlist text (pure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dev:
    tag: str
    dev_type: str
    L_um: float
    W_um: float
    bin: int


def instance(name: str, dev: Dev, d: str, g: str, s: str, b: str, corr: bool) -> str:
    """sky130 instance of ``dev`` (``spice_pdk._instance_line``: parasitic
    geometry and per-device mult of the measured deck) at nodes (d, g, s, b);
    with ``corr`` the BSIM drain is ``<d>_<name>i`` and the OSDI correction
    ``N<name> d g s <d>_<name>i`` with model ``corrn`` / ``corrp`` is added."""
    from .. import spice_pdk as sp
    line = sp._instance_line(dev.dev_type, dev.L_um, dev.W_um)
    head = "Xm1 nd ng 0 0 "
    if not line.startswith(head):
        raise ValueError("unexpected instance line")
    body = line[len(head):]
    di = f"{d}_{name}i" if corr else d
    out = f"X{name} {di} {g} {s} {b} " + body
    if corr:
        model = "corrp" if dev.dev_type == "pmos" else "corrn"
        out += f"N{name} {d} {g} {s} {di} {model}\n"
    return out


def corr_models() -> str:
    return "".join(f".model {m} {VA_MODULE} is_pmos={p} VDD={VDD_NORM:.10g} RS=0\n"
                   for m, p in (("corrn", 0), ("corrp", 1)))


def deck(title: str, lib: Path, body: str, control: Sequence[str], osdi: Path | None,
         options: str = "") -> str:
    """Complete deck: scale, 77 K, the (patched) 77 K library, ``body``, the
    correction models when ``osdi`` is given, and the control block."""
    lines = [f"* {title}\n", ".options scale=1.0\n",
             f".options temp={TEMP_K - 273.15:.4f}\n"]
    if options:
        lines.append(f".options {options}\n")
    lines.append(f'.lib "{Path(lib).as_posix()}" tt_77k\n')
    lines.append(body if body.endswith("\n") else body + "\n")
    if osdi is not None:
        lines.append(corr_models())
    lines.append(".control\n")
    if osdi is not None:
        lines.append(f"pre_osdi {Path(osdi).as_posix()}\n")
    lines += ["set filetype=ascii\n", "set numdgt=12\n"]
    lines += [c if c.endswith("\n") else c + "\n" for c in control]
    lines += ["quit\n", ".endc\n", ".end\n"]
    return "".join(lines)


def device_body(dev: Dev, corr: bool, swap: bool = False) -> str:
    """Single-device test bench: VG at ng, VD at nd, source and body at 0.
    ``swap``: the instance's drain and source terminals exchanged (BSIM
    drain chain at 0, BSIM source and module ``s`` at nd, body at 0)."""
    if swap:
        x = instance("m1", dev, "0", "ng", "nd", "0", corr)
    else:
        x = instance("m1", dev, "nd", "ng", "0", "0", corr)
    return x + "VG ng 0 DC 0 AC 0\nVD nd 0 DC 0 AC 0\n"


def grid_control(dev: Dev, out: Path) -> list[str]:
    """Nested DC sweep: VD over GRID_VDS (inner), VG over the forward gate
    polarity (outer, 0 -> +-1.85)."""
    sg = -1.0 if dev.dev_type == "pmos" else 1.0
    return [f"dc VD {GRID_VDS[0]:.6g} {GRID_VDS[-1]:.6g} {STEP:.6g} "
            f"VG 0 {sg * GRID_VGS[-1]:.6g} {sg * STEP:.6g}",
            f"wrdata {Path(out).as_posix()} -i(VD)"]


def op_points(dev: Dev) -> np.ndarray:
    """(N, 2) signed (Vgs, Vds) op points: gm sets (|Vds| in GM_VDS, |Vgs| on
    GRID_VGS) then gds sets (|Vgs| in GDS_VGS, |Vds| on GRID_VGS)."""
    s = -1.0 if dev.dev_type == "pmos" else 1.0
    pts = [(s * vg, s * vd) for vd in GM_VDS for vg in GRID_VGS]
    pts += [(s * vg, s * vd) for vg in GDS_VGS for vd in GRID_VGS]
    return np.array(pts, dtype=np.float64)


def ac_control(points: np.ndarray, excite: str) -> list[str]:
    """Per op point: op (prints the drain current), AC at AC_FREQ with unit
    excitation on ``excite`` ('VG' or 'VD'); prints Re/Im of the drain
    current and Im of the gate current.  Parsed by ``parse_ac``."""
    if excite not in ("VG", "VD"):
        raise ValueError(excite)
    lines = [f"alter {excite} ac = 1"]
    for k, (vg, vd) in enumerate(points):
        lines += [f"alter VG dc = {vg:.6g}", f"alter VD dc = {vd:.6g}", "op",
                  f"let q{k}i = -i(VD)", f"print q{k}i",
                  f"ac lin 1 {AC_FREQ:g} {AC_FREQ:g}",
                  f"let q{k}r = real(-i(VD))", f"let q{k}m = imag(-i(VD))",
                  f"let q{k}g = imag(-i(VG))",
                  f"print q{k}r q{k}m q{k}g", "destroy all"]
    return lines


_PRINT = re.compile(r"^\s*q(\d+)([irmg])\s*=\s*([-+0-9.eEnaNinf]+)")


def parse_ac(text: str, n: int) -> dict[str, np.ndarray]:
    """``ac_control`` output -> arrays i0 (A), yr (S), ym (S), yg (S) of
    length n (NaN where missing)."""
    out = {k: np.full(n, np.nan) for k in ("i", "r", "m", "g")}
    for line in text.splitlines():
        m = _PRINT.match(line)
        if m:
            k = int(m.group(1))
            if k < n:
                try:
                    out[m.group(2)][k] = float(m.group(3))
                except ValueError:
                    pass
    return {"i0": out["i"], "yr": out["r"], "ym": out["m"], "yg": out["g"]}


def read_wrdata(path: Path, ncols: int = 2) -> np.ndarray:
    """Numeric rows of an ngspice ascii ``wrdata`` file (blank lines and
    non-numeric lines skipped)."""
    rows = []
    p = Path(path)
    if not p.exists():
        return np.zeros((0, ncols))
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < ncols:
            continue
        try:
            rows.append([float(x) for x in parts[:ncols]])
        except ValueError:
            continue
    return np.array(rows, dtype=np.float64).reshape(-1, ncols)


def grid_from_rows(rows: np.ndarray) -> np.ndarray:
    """Nested-sweep rows (VD inner) -> (len(GRID_VGS), len(GRID_VDS)) current
    array; NaN when the row count differs."""
    n = len(GRID_VGS) * len(GRID_VDS)
    if rows.shape[0] != n:
        return np.full((len(GRID_VGS), len(GRID_VDS)), np.nan)
    return rows[:, 1].reshape(len(GRID_VGS), len(GRID_VDS))


def convergence_messages(text: str) -> list[str]:
    out = []
    for line in text.lower().splitlines():
        if any(m in line for m in CONVERGENCE_MESSAGES):
            out.append(line.strip())
    return out


# ---------------------------------------------------------------------------
# Circuits (pure deck bodies)
# ---------------------------------------------------------------------------
def inverter_body(n: Dev, p: Dev, corr: bool, vin: str, vout: str, name: str) -> str:
    return (instance(f"{name}n", n, vout, vin, "0", "0", corr)
            + instance(f"{name}p", p, vout, vin, "vdd", "vdd", corr))


def vtc_deck_body(n: Dev, p: Dev, corr: bool) -> str:
    return (inverter_body(n, p, corr, "nin", "nout", "i1")
            + f"VDD vdd 0 DC {VDD_CIRCUIT}\nVIN nin 0 DC 0\n")


def vtc_control(out: Path) -> list[str]:
    return [f"dc VIN 0 {VDD_CIRCUIT} 0.001", f"wrdata {Path(out).as_posix()} v(nout)"]


def ring_body(n: Dev, p: Dev, corr: bool, stages: int = 5) -> str:
    if stages % 2 == 0:
        raise ValueError("ring oscillator needs an odd number of stages")
    nodes = [f"o{k + 1}" for k in range(stages)]
    body = ""
    for k in range(stages):
        body += inverter_body(n, p, corr, nodes[k - 1], nodes[k], f"s{k + 1}")
    ics = " ".join(f"v({nd})={VDD_CIRCUIT if k % 2 == 0 else 0.0}" for k, nd in enumerate(nodes))
    return body + f"VDD vdd 0 DC {VDD_CIRCUIT}\n.ic {ics}\n"


def ring_control(out: Path, tstep: float, tstop: float) -> list[str]:
    return [f"tran {tstep:.4g} {tstop:.4g}", f"wrdata {Path(out).as_posix()} v(o1)"]


def mirror_body(dev: Dev, corr: bool, iref: float) -> str:
    """1:1 mirror of two ``dev`` transistors; reference current ``iref`` into
    the diode-connected device; output drain at VOUT."""
    if dev.dev_type == "nmos":
        return (instance("m1", dev, "nref", "nref", "0", "0", corr)
                + instance("m2", dev, "nout", "nref", "0", "0", corr)
                + f"IREF 0 nref DC {iref:.6g}\nVOUT nout 0 DC 0\n")
    return (instance("m1", dev, "nref", "nref", "vdd", "vdd", corr)
            + instance("m2", dev, "nout", "nref", "vdd", "vdd", corr)
            + f"VDD vdd 0 DC {VDD_CIRCUIT}\nIREF nref 0 DC {iref:.6g}\nVOUT nout 0 DC 0\n")


def mirror_control(out: Path) -> list[str]:
    """Output current magnitude |i(VOUT)| and V(nref) over VOUT 0 -> VDD."""
    return [f"dc VOUT 0 {VDD_CIRCUIT} 0.01",
            f"wrdata {Path(out).as_posix()} abs(i(VOUT)) v(nref)"]


# ---------------------------------------------------------------------------
# ngspice and library (I/O)
# ---------------------------------------------------------------------------
def patched_library(td: Path, cards: dict) -> Path:
    """77 K library with, per polarity, one bin patched: ``cards`` maps
    dev_type -> list of (bin_index, params) (params None = published card).
    Two patches of the same bin must be identical."""
    from .. import spice_pdk as sp
    err = sp.ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    lib = sp.LIB_FILE.read_text()
    td = Path(td)
    for dev_type, patches in cards.items():
        patches = [(b, p) for b, p in patches if p]
        if not patches:
            continue
        base = sp.PDK77K_DIR / sp._CORNER_BASENAME[dev_type]
        text = base.read_text(errors="replace")
        seen = {}
        for b, p in patches:
            if b in seen:
                if seen[b] != p:
                    raise ValueError(f"two different cards for {dev_type} bin {b}")
                continue
            seen[b] = p
            text = sp._patch_params_in_bin(text, dev_type, b, p)
        corner = td / sp._CORNER_BASENAME[dev_type]
        corner.write_text(text)
        lib = lib.replace(base.as_posix(), corner.as_posix())
    path = td / "lib77k.spice"
    path.write_text(lib)
    return path


def run_deck(text: str, td: Path, name: str = "deck.sp", binary: str = NG_OSDI,
             timeout_s: float = 600.0) -> dict:
    """Run one deck in batch mode; returns stdout+stderr, convergence
    messages, wall time (s) and whether it timed out."""
    td = Path(td)
    path = td / name
    path.write_text(text)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([binary, "-b", str(path)], cwd=str(td), capture_output=True,
                              text=True, timeout=timeout_s, check=False)
        out = proc.stdout + proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired:
        out, timed_out = "", True
    wall = time.perf_counter() - t0
    msgs = convergence_messages(out) + (["timeout"] if timed_out else [])
    if "osdi" in text and ("error" in out.lower() and "osdi" in out.lower()):
        msgs.append("osdi load error")
    return {"text": out, "msgs": msgs, "wall_s": wall, "timeout": timed_out}


def compile_osdi(va_text: str, va: Path, osdi: Path) -> dict:
    from .i3_2_followup import compile_va
    return compile_va(va_text, va, osdi, openvaf=OPENVAF, extra_args=OPENVAF_ARGS)


def tempdir(prefix: str = "cryoml_i11_"):
    return tempfile.TemporaryDirectory(prefix=prefix)


# ---------------------------------------------------------------------------
# Decision (pure, pre-registered)
# ---------------------------------------------------------------------------
def decide(s: dict) -> dict:
    """Pre-registered outcome from the summary values ``s``:

    q1_tight_max, q1_default_max (max relative difference OSDI vs Python,
    |I| > 1 nA, all 18 devices); q1_bad (convergence messages + non-finite
    points + failed runs, element runs); q1_vds0_max (A); q2_ac_max
    (normalized derivative error, tight tolerance); q2_gm_sign (gm curves
    with a sign change in strong inversion introduced by the element,
    ``curve_sign_change_introduced`` on the AC points or the 1 mV sweeps);
    q2_gds_sign (same for gds curves, |Vds| >= 0.05 V); q3_fail (circuits that
    did not converge or did not oscillate with the element while the
    card-only run did); q3_fail_cardonly; swap_asym_max; cap_scaled (bool:
    drain displacement current changed by the element, max |Im Y_dd(corr) /
    Im Y_dd(card) - 1| > 1e-3); swap_asym_max (max |ln(I_swap / I_fwd)
    (corrected) - ln(I_swap / I_fwd) (card)| in forward operation, |I| > 1 nA);
    runtime_ratio_max.
    """
    core_fail = []
    if not (s["q1_tight_max"] <= THR["q1_tight"]):
        core_fail.append("Q1 tight-tolerance equivalence")
    if s["q1_bad"] > 0:
        core_fail.append("Q1 convergence / non-finite")
    if not (s["q1_vds0_max"] <= THR["q1_vds0_abs"]):
        core_fail.append("Q1 I_D at Vds = 0")
    if not (s["q2_ac_max"] <= THR["q2_ac"]):
        core_fail.append("Q2 derivative agreement")
    if s["q2_gm_sign"] > 0:
        core_fail.append("Q2 gm sign change in strong inversion")
    if s["q3_fail"] > 0:
        core_fail.append("Q3 circuit failure with the element")
    restrictions = []
    if not (s["q1_default_max"] <= THR["q1_default"]):
        restrictions.append("default tolerance above 1e-3 (tightened reltol needed for "
                            "equivalence)")
    if s["swap_asym_max"] > THR["swap_asym"]:
        restrictions.append("not source/drain symmetric (g uses the instance source terminal)")
    if s["cap_scaled"]:
        restrictions.append("drain displacement current scaled by exp(g)")
    if s["q2_gds_sign"] > 0:
        restrictions.append("gds sign changes introduced")
    if s["runtime_ratio_max"] > THR["runtime_ratio"]:
        restrictions.append("runtime overhead > 2x")
    if core_fail:
        outcome = "not deployable"
    elif restrictions:
        outcome = "deployable with restrictions"
    else:
        outcome = "deployable (simulator-level)"
    return {"outcome": outcome, "core_fail": core_fail, "restrictions": restrictions}
