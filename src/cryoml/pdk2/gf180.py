"""GF180MCU 3.3 V BSIM4 devices simulated with the pinned ngspice-41.

Second benchmark problem ("pdk2"): simulated held-out devices on a public
PDK, with the sky130 deck conventions of ``cryoml.spice_pdk``:

* one ngspice process runs all 11 DC sweeps of a device (``alter`` the fixed
  bias, ``dc`` the swept one, ``wrdata -i(VD)``), each in its own direction
  (pMOS sweeps 0 -> -3.3 V); currents are reported as positive magnitudes;
* the tuned card is the typical-corner bin that contains (L, W), extracted
  from ``sm141064.ngspice`` as one standalone ``.model`` (bit-identical to
  the full library with native bin selection), and parameter overrides patch
  that model's ``+ key = value`` lines with ``%.10g`` text;
* ``simulate_many`` runs many cards in one process with ``altermod`` (the
  ``cryoml.spice_batch`` trick: integer-mantissa literals, a BSIM4
  model-check cache reset and ``destroy all`` before every card), so each
  card's curves are bit-identical to a separate ``simulate`` call.

Devices: 8 nFET + 8 pFET geometries strictly inside the 4 x 4 (L, W) bin
grid; the nFET takes the bins with even (iL + iW), the pFET the odd ones, so
the 16 devices cover all 16 bins once and each family spans every L and W
bin.  Bias grid: the sky130 layout scaled to VDD = 3.3 V (5 Id-Vd at
Vg = 0.66..3.3 V, 6 Id-Vg at Vd = 0.05, 0.66..3.3 V, 186 points each),
simulated at 27 C (the card's TNOM is 25 C; GF180 is characterized for
-40..125 C, so no cryogenic extrapolation).
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ROOT, resolve_ngspice_bin
from ..data_io import Curve

PDK_NAME = "gf180mcuD"
PDK_VERSION = "c6d73a35f524070e85faff4a6a9eef49553ebc2b"   # volare / open_pdks
NGSPICE_DIR = (ROOT / "data" / "raw" / "pdk2" / "gf180mcu" / "volare"
               / "gf180mcu" / "versions" / PDK_VERSION / PDK_NAME / "libs.tech"
               / "ngspice")
LIBRARY = NGSPICE_DIR / "sm141064.ngspice"
DESIGN = NGSPICE_DIR / "design.ngspice"

NGSPICE_BIN = resolve_ngspice_bin()
TIMEOUT_S = float(os.environ.get("CRYOML_SPICE_TIMEOUT", "60"))
TEMP_C = 27.0
VDD = 3.3
N_POINTS = 186
LINEAR_VD = 0.05                 # the sky130 grid's 0.01 V linear Id-Vg
DIFF_EXT_UM = 0.18               # xschem symbol: ad = W*0.18u, nrd = 0.18u/W
CARD_MODEL = "pdk2card"
FAMILY_MODEL = {"nmos": "nfet_03v3", "pmos": "pfet_03v3"}
TAG_PREFIX = "gf180_"

L_BIN_EDGES_UM = (0.28, 0.5, 1.2, 10.0, 50.0)
W_BIN_EDGES_UM = (0.22, 0.5, 1.2, 10.0, 100.0)
# Two interior L values per L bin and one W value per W bin.
_L_CHOICES = ((0.3, 0.4), (0.7, 1.0), (2.0, 5.0), (15.0, 30.0))
_W_CHOICES = (0.3, 0.7, 5.0, 20.0)


@dataclass(frozen=True)
class Pdk2Device:
    dev_type: str
    L_um: float
    W_um: float

    @property
    def tag(self) -> str:
        def f(x: float) -> str:
            return f"{x:.4g}".replace(".", "p")
        return f"{TAG_PREFIX}{self.dev_type}_L{f(self.L_um)}_W{f(self.W_um)}"


def _devices() -> tuple[Pdk2Device, ...]:
    out = []
    for dev_type, parity in (("nmos", 0), ("pmos", 1)):
        for i_l, pair in enumerate(_L_CHOICES):
            w_bins = [i_w for i_w in range(4) if (i_l + i_w) % 2 == parity]
            for L_um, i_w in zip(pair, w_bins):
                out.append(Pdk2Device(dev_type, L_um, _W_CHOICES[i_w]))
    return tuple(out)


DEVICES: tuple[Pdk2Device, ...] = _devices()
TAGS: tuple[str, ...] = tuple(d.tag for d in DEVICES)


def device_by_tag(tag: str) -> Pdk2Device:
    for device in DEVICES:
        if device.tag == tag:
            return device
    raise KeyError(tag)


def parse_devices(spec: str | None) -> list[Pdk2Device]:
    """``None``/``"all"`` -> all 16; else comma-separated tags."""
    if not spec or spec == "all":
        return list(DEVICES)
    return [device_by_tag(t.strip()) for t in spec.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Bias grid (Curve objects with the target currents in ``Id``)
# ---------------------------------------------------------------------------
VD_FIXED = tuple(round(k * VDD / 5, 4) for k in range(1, 6))     # Id-Vd: Vg
VG_FIXED = (LINEAR_VD,) + VD_FIXED                                 # Id-Vg: Vd
# sky130-equivalent fixed biases (``cryoml.foundation.CANONICAL_CURVE_TAGS``)
CANONICAL_VD = (0.37, 0.74, 1.11, 1.48, 1.85)
CANONICAL_VG = (0.01, 0.37, 0.74, 1.11, 1.48, 1.85)


def device_curves(dev_type: str, targets: Sequence[np.ndarray] | None = None
                  ) -> list[Curve]:
    """The 11 curves in canonical order: Id-Vd (5) then Id-Vg (6).  ``Id``
    holds ``targets`` (positive magnitudes) or zeros."""
    s = -1.0 if dev_type == "pmos" else 1.0
    sweep = s * np.linspace(0.0, VDD, N_POINTS)
    curves = []
    for kind, fixeds, canon in (("idvd", VD_FIXED, CANONICAL_VD),
                                ("idvg", VG_FIXED, CANONICAL_VG)):
        for bias, canonical in zip(fixeds, canon):
            fixed = np.full(N_POINTS, s * bias)
            vg, vd = (fixed, sweep) if kind == "idvd" else (sweep, fixed)
            curves.append(Curve(kind=kind, fixed=s * bias, Vg=vg.copy(),
                                Vd=vd.copy(), Id=np.zeros(N_POINTS),
                                source="pdk2_grid",
                                meta={"canonical_fixed": s * canonical}))
    if targets is not None:
        if len(targets) != len(curves):
            raise ValueError("targets must have one array per curve")
        for curve, target in zip(curves, targets):
            target = np.asarray(target, dtype=np.float64)
            if target.shape != (N_POINTS,):
                raise ValueError("target curve has the wrong length")
            curve.Id = target.copy()
            curve.source = "pdk2_target"
    return curves


def curve_tag(curve: Curve) -> str:
    return f"{curve.kind}@{abs(float(curve.fixed)):g}"


# ---------------------------------------------------------------------------
# Stock card extraction and patching
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _library_text() -> str:
    if not LIBRARY.exists():
        raise RuntimeError(
            f"missing {LIBRARY}: python -m volare fetch --pdk gf180mcu "
            f"--pdk-root data/raw/pdk2/gf180mcu -l gf180mcu_fd_pr {PDK_VERSION}")
    return LIBRARY.read_text(errors="replace")


def _value(block: str, key: str) -> float:
    m = re.search(rf"^\+\s*{re.escape(key)}\s*=\s*([0-9.eE+\-]+)\s*$", block,
                  re.MULTILINE | re.IGNORECASE)
    if not m:
        raise KeyError(key)
    return float(m.group(1))


def typical_bins(dev_type: str) -> list[tuple[int, str]]:
    family = FAMILY_MODEL[dev_type]
    m = re.search(rf"^\.lib\s+{family}_t\s*$(.*?)^\.endl\s+{family}_t\b",
                  _library_text(), re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if not m:
        raise RuntimeError(f"section {family}_t not found")
    section = m.group(1)
    hits = list(re.finditer(rf"^\.model\s+{family}\.(\d+)\s", section,
                            re.MULTILINE | re.IGNORECASE))
    out = []
    for i, h in enumerate(hits):
        raw = section[h.start():hits[i + 1].start() if i + 1 < len(hits)
                      else len(section)].splitlines()
        # The header plus its ``+`` continuation lines (comments and blank
        # lines inside are kept); stop at the next statement (the section's
        # last bin is followed by unrelated subcircuits).
        keep = [raw[0]]
        for line in raw[1:]:
            stripped = line.strip()
            if stripped and not stripped.startswith(("+", "*")):
                break
            keep.append(line)
        out.append((int(h.group(1)), "\n".join(keep).rstrip() + "\n"))
    return out


@lru_cache(maxsize=None)
def stock_card(device: Pdk2Device) -> tuple[int, str]:
    """(bin index, standalone ``.model pdk2card`` text) of the typical bin
    containing (L, W).  The only symbolic values in these bins are the
    flicker-noise constants, resolved at the library default fnoicor = 0."""
    family = FAMILY_MODEL[device.dev_type]
    L, W, eps = device.L_um * 1e-6, device.W_um * 1e-6, 1e-12
    hits = [(i, b) for i, b in typical_bins(device.dev_type)
            if _value(b, "lmin") - eps <= L <= _value(b, "lmax") + eps
            and _value(b, "wmin") - eps <= W <= _value(b, "wmax") + eps]
    if len(hits) != 1:
        raise ValueError(f"{device.tag} matches bins {[h[0] for h in hits]}")
    index, block = hits[0]
    for m in re.finditer(rf"\+\s*({family}_noi[abc])\s*=\s*'\(fnoicor==0\)"
                         rf"\*([0-9.eE+\-]+)", _library_text()):
        block = re.sub(rf"\b{m.group(1)}\b", m.group(2), block)
    if "'" in block or "{" in block:
        raise RuntimeError(f"{device.tag}: unresolved expression in bin {index}")
    block = re.sub(rf"^\.model\s+{family}\.\d+", f".model {CARD_MODEL}",
                   block, count=1, flags=re.MULTILINE | re.IGNORECASE)
    return index, block.rstrip() + "\n"


def bin_index(device: Pdk2Device) -> int:
    return stock_card(device)[0]


def series_r_per_side(device: Pdk2Device) -> float:
    """Stock series resistance per side in ohm*um (RDSW/2; WR = 1)."""
    return 0.5 * _value(stock_card(device)[1], "rdsw")


def stock_params(device: Pdk2Device, names: Sequence[str]) -> dict[str, float]:
    card = stock_card(device)[1]
    return {name: _value(card, name) for name in names}


def patch_card(card: str, params: dict[str, float] | None) -> str:
    """Set ``+ key = value`` lines (``%.10g``, as ``spice_pdk``); a key the
    card lacks is inserted after the header."""
    for key, value in (params or {}).items():
        text = value if isinstance(value, str) else f"{value:.10g}"
        pat = re.compile(rf"^(\+\s*{re.escape(key)}\s*=\s*)[^\n]*",
                         re.IGNORECASE | re.MULTILINE)
        card, n = pat.subn(lambda m: m.group(1) + text, card, count=1)
        if n == 0:
            head = card.find("\n")
            card = card[:head + 1] + f"+ {key} = {text}\n" + card[head + 1:]
    return card


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------
def _instance(device: Pdk2Device, rs_ext_ohm: float = 0.0) -> str:
    area = device.W_um * DIFF_EXT_UM
    perimeter = 2 * (device.W_um + DIFF_EXT_UM)
    squares = DIFF_EXT_UM / device.W_um
    drain, source = ("ndi", "nsi") if rs_ext_ohm > 0 else ("nd", "0")
    line = (f"M1 {drain} ng {source} 0 {CARD_MODEL} l={device.L_um:.6g}u "
            f"w={device.W_um:.6g}u nf=1 ad={area:.6g}p as={area:.6g}p "
            f"pd={perimeter:.6g}u ps={perimeter:.6g}u nrd={squares:.6g} "
            f"nrs={squares:.6g} sa=0 sb=0 sd=0 m=1\n")
    if rs_ext_ohm > 0:   # extrinsic series resistance outside the card
        line += (f"RDEXT nd ndi {rs_ext_ohm:.10g}\n"
                 f"RSEXT nsi 0 {rs_ext_ohm:.10g}\n")
    return line


def _step(v0: float, v1: float, n: int) -> float:
    """Sweep step rounded toward zero at 12 digits, so ngspice never drops
    the end point and the last point sits within ~2e-9 V of it."""
    raw = (v1 - v0) / max(n - 1, 1)
    return math.copysign(math.floor(abs(raw) * 1e12) / 1e12, raw) or 0.05


def _sweep_lines(curves: list[Curve], out_dir: Path, prefix: str,
                 temps_C: Sequence[float] | None = None) -> list[str]:
    """``temps_C`` (optional, one per curve) sets ``option temp`` before each
    sweep (the self-heating-like truth tier); default: the deck temperature."""
    lines = []
    for i, curve in enumerate(curves):
        if temps_C is not None:
            lines.append(f"option temp={float(temps_C[i]):.10g}\n")
        bias, swept = ("VD", "VG") if curve.kind == "idvg" else ("VG", "VD")
        v = np.asarray(curve.Vg if curve.kind == "idvg" else curve.Vd,
                       dtype=np.float64)
        lines += [f"alter {bias} dc = {float(curve.fixed):.6g}\n",
                  f"dc {swept} {v[0]:.12g} {v[-1]:.12g} "
                  f"{_step(float(v[0]), float(v[-1]), len(v)):.12g}\n",
                  f"wrdata {(out_dir / f'{prefix}{i}.dat').as_posix()} -i(VD)\n"]
    return lines


def _header(device: Pdk2Device, card_path: Path, temp_C: float,
            rs_ext_ohm: float = 0.0) -> list[str]:
    return ["* pdk2 GF180MCU batched DC sweeps\n",
            f".options temp={temp_C:.6g}\n",
            f'.include "{card_path.as_posix()}"\n',
            _instance(device, rs_ext_ohm),
            "VG ng 0 DC 0\n", "VD nd 0 DC 0\n",
            ".control\n", "set filetype=ascii\n"]


def _read(out_dir: Path, prefix: str, curves: list[Curve], dev_type: str
          ) -> list[np.ndarray]:
    """Interpolate each sweep onto its target grid; a missing, non-finite or
    incomplete sweep gives an all-NaN curve."""
    out = []
    for i, curve in enumerate(curves):
        target = np.asarray(curve.Vg if curve.kind == "idvg" else curve.Vd,
                            dtype=np.float64)
        path = out_dir / f"{prefix}{i}.dat"
        values = np.full(len(target), np.nan)
        try:
            flat = np.array(path.read_text().split(), dtype=np.float64)
        except (OSError, ValueError):
            flat = np.empty(0)
        if flat.size >= 4 and flat.size % 2 == 0:
            table = flat.reshape(-1, 2)
            order = np.argsort(table[:, 0])
            v, i_d = table[order, 0], table[order, 1]
            if (np.all(np.isfinite(table)) and v[0] <= target.min() + 1e-6
                    and v[-1] >= target.max() - 1e-6):
                values = np.interp(target, v, i_d)
                if dev_type == "pmos":
                    values = -values
        out.append(values)
    return out


def simulate(device: Pdk2Device, params: dict[str, float] | None = None, *,
             curves: list[Curve] | None = None, temp_C: float = TEMP_C,
             rs_ext_ohm: float = 0.0, card: str | None = None,
             curve_temps_C: Sequence[float] | None = None,
             timeout_s: float = TIMEOUT_S) -> list[np.ndarray]:
    """One ngspice process: ``params`` patched into the device's stock card
    (or into ``card``); all-NaN curves on timeout or failure.
    ``rs_ext_ohm`` and ``curve_temps_C`` are truth-only misspecification."""
    curves = curves if curves is not None else device_curves(device.dev_type)
    base = card if card is not None else stock_card(device)[1]
    with tempfile.TemporaryDirectory(prefix="pdk2_sim_") as td_s:
        td = Path(td_s)
        card_path = td / "card.mod"
        card_path.write_text(patch_card(base, params))
        deck = td / "deck.sp"
        deck.write_text("".join(
            _header(device, card_path, temp_C, rs_ext_ohm)
            + _sweep_lines(curves, td, "raw_", curve_temps_C)
            + ["quit\n", ".endc\n", ".end\n"]))
        try:
            subprocess.run([NGSPICE_BIN, "-b", str(deck)], cwd=td,
                           capture_output=True, text=True, timeout=timeout_s,
                           check=False)
        except subprocess.TimeoutExpired:
            return [np.full(len(c.Id), np.nan) for c in curves]
        return _read(td, "raw_", curves, device.dev_type)


# ---------------------------------------------------------------------------
# Many cards per process (altermod; bit-identical to ``simulate``)
# ---------------------------------------------------------------------------
_CHECK_CIRCUIT = ("* BSIM4 model-check cache reset\n"
                  "Mngchk 0 0 0 0 ngchk_dummy l=1u w=1u\n"
                  ".model ngchk_dummy nmos level=54 version=4.5\n"
                  ".end\n")


def _altermod_chunk(device: Pdk2Device, cards: list[dict], curves: list[Curve],
                    temp_C: float, timeout_s: float) -> list[list[np.ndarray] | None]:
    """Run ``cards`` in one process; entries are None for cards that did not
    finish (the caller re-runs them)."""
    from ..spice_batch import ngspice_literal

    with tempfile.TemporaryDirectory(prefix="pdk2_multi_") as td_s:
        td = Path(td_s)
        card_path = td / "card.mod"
        card_path.write_text(stock_card(device)[1])
        check = td / "modelcheck.sp"
        check.write_text(_CHECK_CIRCUIT)
        progress = td / "progress.txt"
        lines = _header(device, card_path, temp_C)
        # After ``source`` the check circuit is circuit 1, the device 2.
        lines += [f"source {check.as_posix()}\n", "setcirc 2\n"]
        for k, params in enumerate(cards):
            lines += [f"altermod {CARD_MODEL} {name} = {ngspice_literal(value)}\n"
                      for name, value in params.items()]
            lines += ["setcirc 1\n", "op\n", "setcirc 2\n", "destroy all\n"]
            lines += _sweep_lines(curves, td, f"c{k}_")
            lines.append(f"echo {k} >> {progress.as_posix()}\n")
        lines += ["quit\n", ".endc\n", ".end\n"]
        deck = td / "deck_multi.sp"
        deck.write_text("".join(lines))
        try:
            subprocess.run([NGSPICE_BIN, "-b", str(deck)], cwd=td,
                           capture_output=True, text=True, check=False,
                           timeout=timeout_s + 0.5 * len(cards))
        except subprocess.TimeoutExpired:
            pass
        try:
            done = {int(t) for t in progress.read_text().split()}
        except (OSError, ValueError):
            done = set()
        return [_read(td, f"c{k}_", curves, device.dev_type) if k in done
                else None for k in range(len(cards))]


def simulate_many(device: Pdk2Device, params_list: Sequence[dict[str, float]],
                  *, curves: list[Curve] | None = None, temp_C: float = TEMP_C,
                  chunk: int = 200, timeout_s: float = TIMEOUT_S
                  ) -> list[list[np.ndarray]]:
    """``[simulate(device, p) for p in params_list]`` with up to ``chunk``
    cards per ngspice process.  Every card must override the same keys
    (altermod cannot restore a value); otherwise this falls back to separate
    processes.  A card that does not finish in a batch is re-run alone."""
    curves = curves if curves is not None else device_curves(device.dev_type)
    params_list = [dict(p or {}) for p in params_list]
    if not params_list:
        return []
    keys = {frozenset(k.lower() for k in p) for p in params_list}
    if len(keys) != 1 or not next(iter(keys)) or any(
            isinstance(v, str) for p in params_list for v in p.values()):
        return [simulate(device, p, curves=curves, temp_C=temp_C,
                         timeout_s=timeout_s) for p in params_list]
    out: list[list[np.ndarray]] = []
    pending = list(range(len(params_list)))
    while pending:
        batch = pending[:chunk]
        got = _altermod_chunk(device, [params_list[i] for i in batch], curves,
                              temp_C, timeout_s)
        n_ok = 0
        for result in got:
            if result is None:
                break
            out.append(result)
            n_ok += 1
        if n_ok < len(batch):        # the stalled card runs alone, then resume
            out.append(simulate(device, params_list[batch[n_ok]], curves=curves,
                                temp_C=temp_C, timeout_s=timeout_s))
            n_ok += 1
        pending = pending[n_ok:]
    return out
