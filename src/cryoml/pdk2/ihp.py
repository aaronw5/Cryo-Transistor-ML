"""IHP SG13G2 low-voltage MOSFETs (PSP 103.6 via OSDI) on ngspice-46.

Isolated on purpose: this is the only problem that runs Homebrew ngspice-46
(our arm64 ngspice-41 build has no OSDI support, and ngspice-41 would
accept only OSDI 0.3 anyway; OpenVAF-Reloaded emits 0.4).  It has its own
binary resolution (``CRYOML_NGSPICE46_BIN`` or ``/opt/homebrew/bin/
ngspice``, checked to report ngspice-46) and never touches
``cryoml.config.resolve_ngspice_bin``, so SkyWater and GF180 stay on the
pinned ngspice-41.

``psp103.osdi`` is compiled from the PDK's Verilog-A by the prebuilt macOS
arm64 OpenVAF-Reloaded v24.0.2mob (provenance in
``data/raw/pdk2/ihp_sg13g2/tools/openvaf-r-*/SOURCE.txt``; ``-D__NGSPICE__``).

Mirrors ``gf180`` (same backend interface, deck and curve conventions):

* card: the PDK's ``sg13g2_lv_{n,p}mos_psp`` ``.model`` block as a top-level
  ``pdk2psp`` model (PSP's global geometry scaling; no bins), with the
  ``mos_tt`` corner knobs, ``pre_layout``, ``SWSOA`` and the subcircuit's
  geometry values supplied as ``.param`` and the subcircuit's instance line
  reproduced verbatim: bit-identical to ``XM1 ... sg13_lv_nmos`` through the
  PDK library (``simulate_pdk_subckt``).  The fitted (wide-set) parameters
  are written as ``%.10g`` numbers of their evaluated stock expressions, so
  the box centre is exactly the stock card (within 1e-10 relative of the
  PDK's expression values);
* ``simulate_many``: ``altermod`` of the top-level OSDI model, ``destroy
  all`` before each card; bit-identical to separate ``simulate`` calls;
* grid: the sky130 layout scaled to the 1.2 V core supply (Id-Vd at
  Vg = 0.24..1.2 V, Id-Vg at Vd = 0.05, 0.24..1.2 V; 186 points), 27 C.

Devices: 8 nFET + 8 pFET inside the model's validity range
(L 0.13-10 um, W 0.15-10 um): four L groups x four W values with the
GF180 parity split, so each family spans every L group and W value.
"""

from __future__ import annotations

import ast
import operator
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ROOT
from ..data_io import Curve
from .gf180 import _read, _sweep_lines

PDK_NAME = "ihp-sg13g2"
PDK_VERSION = "5e6d592e4002946a4616f798c357f0f3c06cf3b6"    # IHP-Open-PDK commit
IHP_DIR = ROOT / "data" / "raw" / "pdk2" / "ihp_sg13g2"
MODELS_DIR = IHP_DIR / "repo" / "ihp-sg13g2" / "libs.tech" / "ngspice" / "models"
CORNER_LIB = MODELS_DIR / "cornerMOSlv.lib"
PARM_LIB = MODELS_DIR / "sg13g2_moslv_parm.lib"
OSDI = IHP_DIR / "osdi" / "psp103.osdi"
TEMP_C = 27.0
VDD = 1.2
N_POINTS = 186
LINEAR_VD = 0.05
TIMEOUT_S = float(os.environ.get("CRYOML_SPICE_TIMEOUT", "60"))
CARD_MODEL = "pdk2psp"
SUBCKT = {"nmos": "sg13_lv_nmos", "pmos": "sg13_lv_pmos"}
FAMILY_MODEL = {"nmos": "sg13g2_lv_nmos_psp", "pmos": "sg13g2_lv_pmos_psp"}
TAG_PREFIX = "ihp_"
# Subcircuit defaults (sg13g2_moslv_mod.lib), ng = 1, rfmode = 0.
SUBCKT_PARAMS = {"ng": 1, "m": 1, "wmin": 0.15e-6, "z1": 0.34e-6,
                 "z2": 0.38e-6, "trise": 0, "pre_layout": 1, "SWSOA": 0}
VD_FIXED = tuple(round(k * VDD / 5, 4) for k in range(1, 6))
VG_FIXED = (LINEAR_VD,) + VD_FIXED
CANONICAL_VD = (0.37, 0.74, 1.11, 1.48, 1.85)
CANONICAL_VG = (0.01, 0.37, 0.74, 1.11, 1.48, 1.85)

_L_CHOICES = ((0.13, 0.18), (0.3, 0.5), (1.0, 2.0), (4.0, 8.0))
_W_CHOICES = (0.2, 0.6, 2.0, 8.0)


@lru_cache(maxsize=None)
def ngspice46_bin() -> str:
    path = os.environ.get("CRYOML_NGSPICE46_BIN", "/opt/homebrew/bin/ngspice")
    out = subprocess.run([path, "-v"], capture_output=True, text=True,
                         timeout=30, check=False).stdout
    if "ngspice-46" not in out:
        raise RuntimeError(f"{path} is not ngspice-46: {out[:200]!r}")
    return path


@dataclass(frozen=True)
class IhpDevice:
    dev_type: str
    L_um: float
    W_um: float

    @property
    def tag(self) -> str:
        def f(x: float) -> str:
            return f"{x:.4g}".replace(".", "p")
        return f"{TAG_PREFIX}{self.dev_type}_L{f(self.L_um)}_W{f(self.W_um)}"


def _devices() -> tuple[IhpDevice, ...]:
    out = []
    for dev_type, parity in (("nmos", 0), ("pmos", 1)):
        for i_l, pair in enumerate(_L_CHOICES):
            w_idx = [i_w for i_w in range(4) if (i_l + i_w) % 2 == parity]
            for L_um, i_w in zip(pair, w_idx):
                out.append(IhpDevice(dev_type, L_um, _W_CHOICES[i_w]))
    return tuple(out)


DEVICES: tuple[IhpDevice, ...] = _devices()
TAGS: tuple[str, ...] = tuple(d.tag for d in DEVICES)


def device_by_tag(tag: str) -> IhpDevice:
    for device in DEVICES:
        if device.tag == tag:
            return device
    raise KeyError(tag)


def parse_devices(spec: str | None) -> list[IhpDevice]:
    if not spec or spec == "all":
        return list(DEVICES)
    return [device_by_tag(t.strip()) for t in spec.split(",") if t.strip()]


def device_curves(dev_type: str, targets: Sequence[np.ndarray] | None = None
                  ) -> list[Curve]:
    """Canonical order: Id-Vd (5) then Id-Vg (6); ``Id`` = targets or 0."""
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
                                source="pdk2_ihp_grid",
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


# ---------------------------------------------------------------------------
# Corner knobs, card extraction, safe expression evaluation, patching
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _mos_tt() -> str:
    text = CORNER_LIB.read_text(errors="replace")
    m = re.search(r"^\.LIB\s+mos_tt\s*$(.*?)^\.ENDL\s+mos_tt\b", text,
                  re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if not m:
        raise RuntimeError(f"mos_tt not found in {CORNER_LIB}")
    return m.group(1)


@lru_cache(maxsize=None)
def corner_knobs() -> dict[str, float]:
    return {k: float(v) for k, v in re.findall(
        r"^\s*\.param\s+(\w+)\s*=\s*([0-9.eE+\-]+)", _mos_tt(), re.MULTILINE)}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(expr: str, names: dict[str, float]) -> float:
    """Arithmetic on numbers and known names only (PDK text is data)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return float(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression {expr!r}")
    return ev(ast.parse(expr.strip(), mode="eval"))


def _model_block(dev_type: str) -> str:
    text = PARM_LIB.read_text(errors="replace")
    m = re.search(rf"^\.model\s+{FAMILY_MODEL[dev_type]}\s+psp103va\b", text,
                  re.MULTILINE | re.IGNORECASE)
    if not m:
        raise RuntimeError(f"{FAMILY_MODEL[dev_type]} not found in {PARM_LIB}")
    lines = text[m.start():].splitlines()
    keep = [lines[0]]
    for line in lines[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith(("+", "*")):
            break
        keep.append(line)
    block = "\n".join(keep).rstrip() + "\n"
    return re.sub(rf"^\.model\s+{FAMILY_MODEL[dev_type]}\b",
                  f".model {CARD_MODEL}", block, count=1,
                  flags=re.MULTILINE | re.IGNORECASE)


def _key_re(key: str) -> re.Pattern:
    return re.compile(rf"(?<![\w.]){re.escape(key)}(\s*=\s*)('[^']*'|[^\s']+)",
                      re.IGNORECASE)


def _card_value_text(card: str, key: str) -> str:
    for line in card.splitlines():
        if line.lstrip().startswith(("+", ".model")):
            m = _key_re(key).search(line)
            if m:
                return m.group(2).strip("'")
    raise KeyError(key)


def patch_card(card: str, params: dict[str, float] | None) -> str:
    """Set ``key = value`` (``%.10g``) on the card's parameter lines."""
    lines = card.splitlines()
    for key, value in (params or {}).items():
        text = value if isinstance(value, str) else f"{value:.10g}"
        for i, line in enumerate(lines):
            if not line.lstrip().startswith(("+", ".model")):
                continue
            new, n = _key_re(key).subn(lambda m: f"{key}{m.group(1)}{text}",
                                       line, count=1)
            if n:
                lines[i] = new
                break
        else:
            lines.insert(1, f"+ {key} = {text}")
    return "\n".join(lines) + "\n"


def _geometry_names(device: IhpDevice) -> dict[str, float]:
    return {**corner_knobs(), **SUBCKT_PARAMS, "w": device.W_um * 1e-6,
            "l": device.L_um * 1e-6}


def expression_card(device: IhpDevice) -> str:
    """The PDK model block, expressions intact (validation reference)."""
    return _model_block(device.dev_type)


def card_value(device: IhpDevice, key: str, card: str | None = None) -> float:
    card = card if card is not None else expression_card(device)
    return _safe_eval(_card_value_text(card, key), _geometry_names(device))


@lru_cache(maxsize=None)
def stock_card(device: IhpDevice) -> tuple[int, str]:
    """(0, card) with every fitted (wide-set) parameter as ``%.10g`` of its
    evaluated stock expression."""
    from .boxes import IHP_WIDE
    card = expression_card(device)
    values = {k: card_value(device, k, card) for k in IHP_WIDE}
    return 0, patch_card(card, values)


def bin_index(device: IhpDevice) -> int:
    return 0


def stock_params(device: IhpDevice, names: Sequence[str]) -> dict[str, float]:
    card = stock_card(device)[1]
    return {k: card_value(device, k, card) for k in names}


def series_r_per_side(device: IhpDevice) -> float:
    """Stock series resistance per side in ohm*um (PSP RSW1 at W = 1 um)."""
    return card_value(device, "rsw1", stock_card(device)[1])


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------
_INSTANCE = ("N1 {d} ngt {s} 0 {model} w=w l=l nf='ng' mult='m' "
             "as='max(w/ng,wmin)*(z1+((ng-1)/2)*z2)' "
             "ad='max(w/ng,wmin)*(z1+((ng-1)/2)*z2)' "
             "ps='2*(max(w/ng,wmin)*((ng-1)/2+1)+z1+(ng-1)/2*z2)' "
             "pd='2*(max(w/ng,wmin)*((ng-1)/2+1)+z1+(ng-1)/2*z2)' "
             "dta=trise ngcon=2 delvto=0 factuo=1\n")


def _header(device: IhpDevice, card_path: Path, temp_C: float,
            rs_ext_ohm: float = 0.0) -> list[str]:
    params = {**SUBCKT_PARAMS, "w": f"{device.W_um:.6g}u",
              "l": f"{device.L_um:.6g}u"}
    drain, source = ("ndi", "nsi") if rs_ext_ohm > 0 else ("nd", "0")
    lines = ["* pdk2 IHP SG13G2 batched DC sweeps\n",
             f".options temp={temp_C:.6g}\n",
             ".param " + " ".join(f"{k}={v}" for k, v in params.items()) + "\n"]
    lines += [f".param {k}={v!r}\n" for k, v in corner_knobs().items()]
    lines += [f'.include "{card_path.as_posix()}"\n',
              _INSTANCE.format(d=drain, s=source, model=CARD_MODEL)]
    if rs_ext_ohm > 0:
        lines += [f"RDEXT nd ndi {rs_ext_ohm:.10g}\n",
                  f"RSEXT nsi 0 {rs_ext_ohm:.10g}\n"]
    lines += ["VG ngt 0 DC 0\n", "VD nd 0 DC 0\n", ".control\n",
              f"pre_osdi {OSDI.as_posix()}\n", "set filetype=ascii\n"]
    return lines


def _run(deck: Path, cwd: Path, timeout_s: float) -> bool:
    try:
        subprocess.run([ngspice46_bin(), "-b", str(deck)], cwd=cwd,
                       capture_output=True, text=True, timeout=timeout_s,
                       check=False)
        return True
    except subprocess.TimeoutExpired:
        return False


def simulate(device: IhpDevice, params: dict[str, float] | None = None, *,
             curves: list[Curve] | None = None, temp_C: float = TEMP_C,
             rs_ext_ohm: float = 0.0, card: str | None = None,
             curve_temps_C: Sequence[float] | None = None,
             timeout_s: float = TIMEOUT_S) -> list[np.ndarray]:
    """One ngspice-46 process with all 11 sweeps (``gf180.simulate``)."""
    curves = curves if curves is not None else device_curves(device.dev_type)
    base = card if card is not None else stock_card(device)[1]
    with tempfile.TemporaryDirectory(prefix="pdk2_ihp_") as td_s:
        td = Path(td_s)
        card_path = td / "card.mod"
        card_path.write_text(patch_card(base, params))
        deck = td / "deck.sp"
        deck.write_text("".join(
            _header(device, card_path, temp_C, rs_ext_ohm)
            + _sweep_lines(curves, td, "raw_", curve_temps_C)
            + ["quit\n", ".endc\n", ".end\n"]))
        if not _run(deck, td, timeout_s):
            return [np.full(len(c.Id), np.nan) for c in curves]
        return _read(td, "raw_", curves, device.dev_type)


def _altermod_chunk(device: IhpDevice, cards: list[dict], curves: list[Curve],
                    temp_C: float, timeout_s: float) -> list:
    from ..spice_batch import ngspice_literal
    with tempfile.TemporaryDirectory(prefix="pdk2_ihp_multi_") as td_s:
        td = Path(td_s)
        card_path = td / "card.mod"
        card_path.write_text(stock_card(device)[1])
        progress = td / "progress.txt"
        lines = _header(device, card_path, temp_C)
        for k, params in enumerate(cards):
            lines += [f"altermod {CARD_MODEL} {name} = {ngspice_literal(value)}\n"
                      for name, value in params.items()]
            lines.append("destroy all\n")
            lines += _sweep_lines(curves, td, f"c{k}_")
            lines.append(f"echo {k} >> {progress.as_posix()}\n")
        lines += ["quit\n", ".endc\n", ".end\n"]
        deck = td / "deck_multi.sp"
        deck.write_text("".join(lines))
        _run(deck, td, timeout_s + 1.0 * len(cards))
        try:
            done = {int(t) for t in progress.read_text().split()}
        except (OSError, ValueError):
            done = set()
        return [_read(td, f"c{k}_", curves, device.dev_type) if k in done
                else None for k in range(len(cards))]


def simulate_many(device: IhpDevice, params_list: Sequence[dict[str, float]],
                  *, curves: list[Curve] | None = None, temp_C: float = TEMP_C,
                  chunk: int = 200, timeout_s: float = TIMEOUT_S
                  ) -> list[list[np.ndarray]]:
    """``gf180.simulate_many`` for the OSDI card (altermod, same key set)."""
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
        if n_ok < len(batch):
            out.append(simulate(device, params_list[batch[n_ok]], curves=curves,
                                temp_C=temp_C, timeout_s=timeout_s))
            n_ok += 1
        pending = pending[n_ok:]
    return out


# ---------------------------------------------------------------------------
# Validation path: the PDK's own subcircuit and corner library
# ---------------------------------------------------------------------------
def _library(knobs: dict[str, float] | None) -> str:
    body = _mos_tt()
    for key, value in (knobs or {}).items():
        body, n = re.subn(rf"^(\s*\.param\s+{re.escape(key)}\s*=\s*)\S+",
                          lambda m: m.group(1) + f"{value:.10g}", body,
                          count=1, flags=re.MULTILINE)
        if n == 0:
            raise KeyError(f"unknown mos_tt knob {key}")
    return re.sub(r"^(\s*\.include\s+)(\S+)",
                  lambda m: f'{m.group(1)}"{(MODELS_DIR / m.group(2)).as_posix()}"',
                  body, flags=re.MULTILINE)


def simulate_pdk_subckt(device: IhpDevice, knobs: dict[str, float] | None = None,
                        *, curves: list[Curve] | None = None,
                        temp_C: float = TEMP_C) -> list[np.ndarray]:
    """``XM1 ... sg13_lv_{n,p}mos`` through ``cornerMOSlv.lib`` mos_tt."""
    curves = curves if curves is not None else device_curves(device.dev_type)
    with tempfile.TemporaryDirectory(prefix="pdk2_ihp_ref_") as td_s:
        td = Path(td_s)
        lib = td / "mos_tt.lib"
        lib.write_text(_library(knobs))
        deck = td / "deck.sp"
        deck.write_text("".join(
            ["* IHP PDK reference\n", f".options temp={temp_C:.6g}\n",
             f'.include "{lib.as_posix()}"\n',
             f"XM1 nd ngt 0 0 {SUBCKT[device.dev_type]} w={device.W_um:.6g}u "
             f"l={device.L_um:.6g}u ng=1 m=1\n",
             "VG ngt 0 DC 0\n", "VD nd 0 DC 0\n", ".control\n",
             f"pre_osdi {OSDI.as_posix()}\n",
             f"pre_setcs sourcepath = ( {MODELS_DIR.as_posix()} )\n",
             "set filetype=ascii\n"]
            + _sweep_lines(curves, td, "raw_") + ["quit\n", ".endc\n", ".end\n"]))
        _run(deck, td, TIMEOUT_S)
        return _read(td, "raw_", curves, device.dev_type)
