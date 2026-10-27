"""NGSpice backend for the 77 K SKY130 model cards.

Deck and card conventions follow the confirmed setup in
``ogzamour/CryoPDK_Skywater130nm_ML`` (per-device ``sweeps.spice`` decks):
default simulator options + ``temp=-196.15``, drain current ``-i(VD)``,
the updated pFET corner card, the sweeps.spice instance line
(0.29-based ad/as/pd/ps, nrd/nrs, sa/sb/sd) and per-device ``mult``/``m``.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import (CORRECTED_REPO_DIR, DATA_DIR, NEW_REPO_DIR,
                     PROCESSED_DIR, resolve_ngspice_bin)
from .data_io import Curve
from .utils import get_logger

logger = get_logger("cryoml.spice_pdk")

_NGSPICE_BIN = resolve_ngspice_bin()
_DEFAULT_TIMEOUT_S = float(os.environ.get("CRYOML_SPICE_TIMEOUT", "60"))

PDK77K_DIR = PROCESSED_DIR / "pdk77k"
LIB_FILE = PDK77K_DIR / "sky130_77k.lib.spice"
LOD_FILE = DATA_DIR / "raw" / "lod.spice"
PDK_SPICE_DIR = (DATA_DIR / "raw" / "pdk" / "sky130A" / "libs.ref"
                 / "sky130_fd_pr" / "spice")
CORRECTED_CORNER_DIR = CORRECTED_REPO_DIR / "compressed-cryo-files"

_SUBCKT = {
    "nmos": "sky130_fd_pr__nfet_01v8_lvt",
    "pmos": "sky130_fd_pr__pfet_01v8_lvt",
}
_MODEL = {
    "nmos": "sky130_fd_pr__nfet_01v8_lvt__model",
    "pmos": "sky130_fd_pr__pfet_01v8_lvt__model",
}
_CORNER_BASENAME = {
    "nmos": "sky130_fd_pr__nfet_01v8_lvt__tt_77k.corner.spice",
    "pmos": "sky130_fd_pr__pfet_01v8_lvt__tt_77k.corner.spice",
}

# ---------------------------------------------------------------------------
# Card template (I25-1).  ``CRYOML_PDK_TEMPLATE`` is read once at import:
#   "paper77k" (default) the released 77 K corner cards (tnom -196.15 C,
#              temperature coefficients 0, one custom bin per device);
#   "stock"    the stock SkyWater sky130A tt models of the two lvt devices
#              (``libs.ref/.../*_01v8_lvt__tt.pm3.spice`` copied verbatim:
#              tnom 30 C, the stock temperature coefficients and the stock
#              L/W bins), the stock tt-corner junction multipliers, the stock
#              lod parameters and mismatch corners.  The deck is unchanged
#              (``.options temp=-196.15``), so NGSpice applies the BSIM4
#              temperature equations from tnom to 77 K and selects the bin.
# Every module that imports PDK77K_DIR / LIB_FILE / _CORNER_BASENAME after
# this module is loaded uses the selected template.
# ---------------------------------------------------------------------------
PDK_TEMPLATES = ("paper77k", "stock")
PDK_TEMPLATE = os.environ.get("CRYOML_PDK_TEMPLATE", "paper77k")
if PDK_TEMPLATE not in PDK_TEMPLATES:
    raise ValueError(f"CRYOML_PDK_TEMPLATE {PDK_TEMPLATE!r} not in {PDK_TEMPLATES}")
STOCK_DIR = PROCESSED_DIR / "pdk_stock"
STOCK_LIB_FILE = STOCK_DIR / "sky130_stock_tt.lib.spice"
_STOCK_BASENAME = {
    "nmos": "sky130_fd_pr__nfet_01v8_lvt__tt.pm3.spice",
    "pmos": "sky130_fd_pr__pfet_01v8_lvt__tt.pm3.spice",
}
_STOCK_CORNER_WRAPPER = {
    "nmos": "sky130_fd_pr__nfet_01v8_lvt__tt.corner.spice",
    "pmos": "sky130_fd_pr__pfet_01v8_lvt__tt.corner.spice",
}
PAPER77K_DIR, PAPER_LIB_FILE = PDK77K_DIR, LIB_FILE
PAPER_CORNER_BASENAME = dict(_CORNER_BASENAME)
if PDK_TEMPLATE == "stock":
    PDK77K_DIR, LIB_FILE = STOCK_DIR, STOCK_LIB_FILE
    _CORNER_BASENAME = dict(_STOCK_BASENAME)

# Corner card sources for the confirmed setup: the nFET card is the original
# published one (unchanged in the new repo); the pFET card is the new repo's
# updated ("update_...") re-fit.
_CORNER_SOURCE = {
    "nmos": CORRECTED_CORNER_DIR / _CORNER_BASENAME["nmos"],
    "pmos": NEW_REPO_DIR / ("update_" + _CORNER_BASENAME["pmos"]),
}

# Per-device parallel multiplicity (mult=m=...) from the new repo's
# per-device sweeps.spice decks; every device not listed uses 1.
_DEVICE_MULT: dict[tuple[str, float, float], int] = {
    ("pmos", 0.35, 5.0): 3,
    ("pmos", 0.5, 0.64): 2,
    ("pmos", 4.0, 7.0): 3,
    ("pmos", 8.0, 0.84): 3,
    ("pmos", 8.0, 1.6): 3,
    ("pmos", 8.0, 5.0): 3,
}


def device_mult(dev_type: str, L_um: float, W_um: float) -> int:
    for (t, l, w), m in _DEVICE_MULT.items():
        if t == dev_type and abs(l - L_um) < 1e-3 and abs(w - W_um) < 1e-3:
            return m
    return 1


def ensure_stock() -> str | None:
    """Create the stock-template copies + lib wrapper. Returns error or None.

    The two pm3 model files are copied byte for byte; the wrapper holds the
    ``.param`` block of each stock ``__tt.corner.spice`` wrapper (junction
    multipliers) and includes the stock lod parameters and mismatch corners,
    i.e. the part of ``sky130.lib.spice`` section ``tt`` that the two lvt
    devices use."""
    if not LOD_FILE.exists():
        return f"missing {LOD_FILE}"
    if not PDK_SPICE_DIR.exists():
        return f"missing Volare PDK at {PDK_SPICE_DIR}"
    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    params = []
    for dev in ("nmos", "pmos"):
        src = PDK_SPICE_DIR / _STOCK_BASENAME[dev]
        dst = STOCK_DIR / _STOCK_BASENAME[dev]
        text = src.read_text(errors="replace")
        if not dst.exists() or dst.read_text(errors="replace") != text:
            dst.write_text(text)
        wrapper = (PDK_SPICE_DIR / _STOCK_CORNER_WRAPPER[dev]).read_text(errors="replace")
        block = re.search(r"^\.param\s*\n((?:\+[^\n]*\n)+)", wrapper, re.MULTILINE)
        if not block:
            return f"no .param block in {_STOCK_CORNER_WRAPPER[dev]}"
        params.append(".param\n" + block.group(1))
    lib_text = (
        "* Stock SkyWater sky130A tt models of nfet_01v8_lvt / pfet_01v8_lvt (I25-1 template).\n"
        "* The section name tt_77k is the name the decks ask for; tnom, temperature\n"
        "* coefficients and bins are the stock ones.\n"
        ".lib tt_77k\n"
        ".param mc_mm_switch=0\n"
        ".param mc_pr_switch=0\n"
        f'.include "{LOD_FILE.as_posix()}"\n'
        + "".join(params) +
        f'.include "{(STOCK_DIR / _STOCK_BASENAME["nmos"]).as_posix()}"\n'
        f'.include "{(PDK_SPICE_DIR / "sky130_fd_pr__nfet_01v8_lvt__mismatch.corner.spice").as_posix()}"\n'
        f'.include "{(STOCK_DIR / _STOCK_BASENAME["pmos"]).as_posix()}"\n'
        f'.include "{(PDK_SPICE_DIR / "sky130_fd_pr__pfet_01v8_lvt__mismatch.corner.spice").as_posix()}"\n'
        ".endl\n")
    if not STOCK_LIB_FILE.exists() or STOCK_LIB_FILE.read_text() != lib_text:
        STOCK_LIB_FILE.write_text(lib_text)
    return None


def ensure_pdk77k() -> str | None:
    """Create the patched corner copies + lib wrapper. Returns error or None.
    With ``CRYOML_PDK_TEMPLATE=stock`` the stock template (``ensure_stock``)."""
    if PDK_TEMPLATE == "stock":
        return ensure_stock()
    if not LOD_FILE.exists():
        return f"missing {LOD_FILE}"
    if not PDK_SPICE_DIR.exists():
        return (f"missing Volare PDK at {PDK_SPICE_DIR} — run "
                "`volare enable --pdk sky130 a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b` "
                "with PDK_ROOT=data/raw/pdk")
    PDK77K_DIR.mkdir(parents=True, exist_ok=True)
    for dev, base in _CORNER_BASENAME.items():
        src = _CORNER_SOURCE[dev]
        if not src.exists():
            return f"missing corner card source {src}"
        dst = PDK77K_DIR / base
        text = src.read_text(errors="replace")
        text = re.sub(
            r'^\.include\s+"[^"]*lod\.spice"',
            f'.include "{LOD_FILE.as_posix()}"',
            text, flags=re.MULTILINE)
        if not dst.exists() or dst.read_text() != text:
            dst.write_text(text)
    lib_text = (
        "* Standalone 77K typical-corner library (cryo corners + stock mismatch params)\n"
        ".lib tt_77k\n"
        ".param mc_mm_switch=0\n"
        ".param mc_pr_switch=0\n"
        ".param my_gauss = 0\n"
        ".param m = 1\n"
        f'.include "{(PDK77K_DIR / _CORNER_BASENAME["nmos"]).as_posix()}"\n'
        f'.include "{(PDK77K_DIR / _CORNER_BASENAME["pmos"]).as_posix()}"\n'
        f'.include "{(PDK_SPICE_DIR / "sky130_fd_pr__nfet_01v8_lvt__mismatch.corner.spice").as_posix()}"\n'
        f'.include "{(PDK_SPICE_DIR / "sky130_fd_pr__pfet_01v8_lvt__mismatch.corner.spice").as_posix()}"\n'
        ".endl\n")
    if not LIB_FILE.exists() or LIB_FILE.read_text() != lib_text:
        LIB_FILE.write_text(lib_text)
    return None


# ---------------------------------------------------------------------------
# Parameter overrides: patch +key=... inside the matching bin of a corner copy
# ---------------------------------------------------------------------------
_BIN_HDR_RE = {
    dev: re.compile(
        rf"^\.model\s+{re.escape(_MODEL[dev])}\.(\d+)\s", re.IGNORECASE | re.MULTILINE)
    for dev in _MODEL
}


def _bin_spans(text: str, dev_type: str) -> list[tuple[int, int, int]]:
    """Return (bin_index, start, end) char spans of each bin's .model block."""
    hits = list(_BIN_HDR_RE[dev_type].finditer(text))
    spans = []
    for i, h in enumerate(hits):
        start = h.start()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        spans.append((int(h.group(1)), start, end))
    return spans


# Stock pm3 bins write several ``key = value`` tokens per ``+`` line; a value
# is a number or one brace expression without spaces.
_TOKEN_VALUE = r"(\{[^}\s]*\}|[^\s{}]+)"


def _token_re(key: str) -> re.Pattern:
    return re.compile(rf"(?<![\w.]){re.escape(key)}(\s*=\s*){_TOKEN_VALUE}", re.IGNORECASE)


def _bin_geometry(span_text: str) -> dict[str, float]:
    if PDK_TEMPLATE == "stock":
        return _bin_geometry_tokens(span_text)
    out = {}
    for key in ("lmin", "lmax", "wmin", "wmax"):
        m = re.search(rf"\+\s*{key}\s*=\s*([0-9.eE+\-]+)", span_text)
        if m:
            out[key] = float(m.group(1))
    return out


def _plus_lines(span_text: str) -> list[str]:
    return [ln for ln in span_text.splitlines() if ln.lstrip().startswith("+")]


def _bin_geometry_tokens(span_text: str) -> dict[str, float]:
    out = {}
    for key in ("lmin", "lmax", "wmin", "wmax"):
        for line in _plus_lines(span_text):
            m = _token_re(key).search(line)
            if m:
                out[key] = float(m.group(2))
                break
    return out


def find_bin_index(dev_type: str, L_um: float, W_um: float) -> int | None:
    """Return the model bin selected by NGspice for this native geometry."""
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    base_path = PDK77K_DIR / _CORNER_BASENAME[dev_type]
    base = base_path.read_text(errors="replace")
    L_m, W_m = L_um * 1e-6, W_um * 1e-6
    eps = 1e-12
    candidates = []
    for idx, start, end in _bin_spans(base, dev_type):
        geometry = _bin_geometry(base[start:end])
        if (len(geometry) == 4
                and geometry["lmin"] - eps <= L_m <= geometry["lmax"] + eps
                and geometry["wmin"] - eps <= W_m <= geometry["wmax"] + eps):
            candidates.append(idx)
    if len(candidates) <= 1:
        return candidates[0] if candidates else None

    sentinels = {idx: 0.101 + 0.013 * idx for idx in candidates}
    patched = base
    for idx, sentinel in sentinels.items():
        patched = _patch_params_in_bin(patched, dev_type, idx, {"eta0": sentinel})

    subckt = _SUBCKT[dev_type]
    with tempfile.TemporaryDirectory(prefix="cryoml_native_bin_") as td_s:
        td = Path(td_s)
        corner = td / _CORNER_BASENAME[dev_type]
        corner.write_text(patched)
        lib = td / "lib77k.spice"
        lib.write_text(
            LIB_FILE.read_text().replace(base_path.as_posix(), corner.as_posix())
        )
        deck = td / "native_bin.sp"
        deck.write_text(
            "* native NGspice bin selection\n"
            ".options scale=1.0 temp=-196.15\n"
            f'.lib "{lib.as_posix()}" tt_77k\n'
            + _instance_line(dev_type, L_um, W_um) +
            "VG ng 0 DC 1\n"
            "VD nd 0 DC 1\n"
            ".op\n"
            ".control\n"
            "run\n"
            f"showmod m.xm1.m{subckt} : eta0\n"
            "quit\n"
            ".endc\n"
            ".end\n"
        )
        proc = subprocess.run(
            [_NGSPICE_BIN, "-b", str(deck)],
            capture_output=True,
            timeout=_DEFAULT_TIMEOUT_S,
            text=True,
            check=False,
        )
    match = re.search(r"^\s*eta0\s+([0-9.eE+\-]+)\s*$",
                      proc.stdout, re.MULTILINE)
    if not match:
        return None
    selected = float(match.group(1))
    selected_bin = min(sentinels, key=lambda idx: abs(sentinels[idx] - selected))
    if abs(sentinels[selected_bin] - selected) > 5e-6:
        return None
    return selected_bin


def _force_bin(text: str, dev_type: str, bin_index: int) -> str:
    """Patch the subckt's m-instance to reference an explicit bin model,
    bypassing NGSpice's automatic L/W bin selection."""
    base = re.escape(_MODEL[dev_type])
    out, n = re.subn(rf"\b{base}\b(\s+l\s*=)", f"{_MODEL[dev_type]}.{bin_index}\\1",
                     text, count=1)
    if n == 0:
        raise ValueError(f"m-instance for {dev_type} not found")
    return out


def _patch_params_in_bin(text: str, dev_type: str, bin_index: int,
                         params: dict[str, float]) -> str:
    if PDK_TEMPLATE == "stock":
        return _patch_params_in_bin_tokens(text, dev_type, bin_index, params)
    for idx, s, e in _bin_spans(text, dev_type):
        if idx != bin_index:
            continue
        span = text[s:e]
        for k, v in params.items():
            # Strings are written verbatim (e.g. ``{...temper...}`` expressions).
            text_v = v if isinstance(v, str) else f"{v:.10g}"
            pat = re.compile(rf"^(\+\s*{re.escape(k)}\s*=\s*)[^\n]*", re.IGNORECASE | re.MULTILINE)
            span, n = pat.subn(lambda m: m.group(1) + text_v, span, count=1)
            if n == 0:
                hdr_end = span.find("\n")
                span = span[:hdr_end + 1] + f"+ {k} = {text_v}\n" + span[hdr_end + 1:]
        return text[:s] + span + text[e:]
    raise ValueError(f"bin {bin_index} not found for {dev_type}")


def _patch_params_in_bin_tokens(text: str, dev_type: str, bin_index: int,
                                params: dict[str, float]) -> str:
    """Stock-template patcher: replace the ``key = value`` token of each
    parameter on the bin's ``+`` lines (other tokens on the line, e.g. the
    ``lvth0`` / ``wvth0`` / ``pvth0`` size terms, are kept); a parameter the
    bin does not set is added as ``+ key = value`` after the header line."""
    for idx, s, e in _bin_spans(text, dev_type):
        if idx != bin_index:
            continue
        lines = text[s:e].split("\n")
        for k, v in params.items():
            text_v = v if isinstance(v, str) else f"{v:.10g}"
            pat = _token_re(k)
            hits = 0
            for i, line in enumerate(lines):
                if not line.lstrip().startswith("+"):
                    continue
                new, n = pat.subn(lambda m: k + m.group(1) + text_v, line)
                if n:
                    hits += n
                    lines[i] = new
            if hits > 1:
                raise ValueError(f"{k} set {hits} times in {dev_type} bin {bin_index}")
            if hits == 0:
                lines.insert(1, f"+ {k} = {text_v}")
        return text[:s] + "\n".join(lines) + text[e:]
    raise ValueError(f"bin {bin_index} not found for {dev_type}")


# Devices whose upstream sweeps.spice uses a bare instance line (no parasitic
# area/perimeter/resistance parameters) — mirrored for bit-faithfulness.
_MINIMAL_INSTANCE = {("pmos", 0.35, 1.6), ("pmos", 2.0, 5.0)}


def _instance_line(dev_type: str, L_um: float, W_um: float,
                   rs_ext_ohm: float = 0.0) -> str:
    """sweeps.spice instance, with the primed µm expressions pre-evaluated
    for our deck convention (deck scale=1.0, suffixed values): ad/as in µm²
    (p), pd/ps in µm (u), nrd/nrs unitless, per-device mult=m.

    ``rs_ext_ohm`` > 0 adds equal extrinsic resistors outside the card in
    series with the drain and the source terminals (exploration baseline
    I4-1 B2); the body stays at the external ground.  With the default 0 the
    line is unchanged."""
    subckt = _SUBCKT[dev_type]
    m = device_mult(dev_type, L_um, W_um)
    drain, source = ("ndi", "nsi") if rs_ext_ohm > 0 else ("nd", "0")
    ext = (f"RDEXT nd ndi {rs_ext_ohm:.10g}\nRSEXT nsi 0 {rs_ext_ohm:.10g}\n"
           if rs_ext_ohm > 0 else "")
    if any(dev_type == t and abs(l - L_um) < 1e-3 and abs(w - W_um) < 1e-3
           for t, l, w in _MINIMAL_INSTANCE):
        return (f"Xm1 {drain} ng {source} 0 {subckt} l={L_um:.6g}u w={W_um:.6g}u "
                f"mult={m} m={m}\n") + ext
    area_um2 = W_um * 0.29                 # floor((nf+1)/2)*w/nf*0.29, nf=1
    perimeter_um = 2 * (W_um + 0.29)       # 2*floor((nf+1)/2)*(w/nf+0.29)
    squares = 0.29 / W_um
    return (
        f"Xm1 {drain} ng {source} 0 {subckt} l={L_um:.6g}u w={W_um:.6g}u nf=1 "
        f"ad={area_um2:.6g}p as={area_um2:.6g}p "
        f"pd={perimeter_um:.6g}u ps={perimeter_um:.6g}u "
        f"nrd={squares:.6g} nrs={squares:.6g} sa=0 sb=0 sd=0 "
        f"mult={m} m={m}\n") + ext


# ---------------------------------------------------------------------------
# BSIM4 model-check clamp state (I13-1)
# ---------------------------------------------------------------------------
# ngspice-41 ``b4check.c`` (BSIM4checkModel) runs once per process per model
# name (static ``modname``) and clamps size-dependent parameters, e.g.
# A2 > 1 -> A2 = 1 and A1 = 0, A2 < 0.01 -> A2 = 0.01.  ``b4temp.c`` rebuilds the
# size-dependent parameters at every analysis.  In a deck with one BSIM4
# model and several analyses, the first analysis therefore runs with the
# clamped values and every later analysis with the card values.
#
# ``bsim4_state`` of ``simulate_pdk``:
#   "clamped"    (default) the clamp that ngspice applies in a fresh
#                single-analysis process is written into the card, so every
#                curve is simulated in that state.  Cards that the check
#                does not clamp are written unchanged (identical deck text).
#   "legacy"     the deck as used before I13-1 (first curve clamped, later
#                curves with the card values).  Reproduction only.
#   "unclamped"  a warm-up ``op`` runs before the curves, so every curve is
#                simulated with the card values.  Audit only.
BSIM4_STATES = ("clamped", "legacy", "unclamped")
_A2_LOW, _A2_HIGH = 0.01, 1.0
_SIZE_TERMS = ("la1", "wa1", "pa1", "la2", "wa2", "pa2")


def _bin_text(dev_type: str, bin_index: int) -> str:
    base = (PDK77K_DIR / _CORNER_BASENAME[dev_type]).read_text(errors="replace")
    for idx, s, e in _bin_spans(base, dev_type):
        if idx == bin_index:
            return base[s:e]
    raise ValueError(f"bin {bin_index} not found for {dev_type}")


def _bin_value(span: str, key: str) -> str | None:
    if PDK_TEMPLATE == "stock":
        for line in _plus_lines(span):
            m = _token_re(key).search(line)
            if m:
                return m.group(2)
        return None
    m = re.search(rf"^\+\s*{re.escape(key)}\s*=\s*([^\s]+)", span,
                  re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else None


def bsim4_clamp_params(dev_type: str, bin_index: int,
                       params: dict[str, float] | None) -> tuple[dict, dict]:
    """Return (params with the b4check.c A1/A2 clamp written explicitly,
    applied changes).  The effective size-dependent A2 equals the bin's a2
    because the 77 K bins have no la2/wa2/pa2/la1/wa1/pa1 terms (checked);
    the clamp is written only when it changes a value, so an unclamped card
    returns ``params`` unchanged and an empty change dict."""
    params = dict(params or {})
    span = _bin_text(dev_type, bin_index)
    for key in _SIZE_TERMS:
        if _bin_value(span, key) is not None or key in params:
            raise NotImplementedError(f"{key} present: effective A1/A2 is size dependent")
    a2 = float(params["a2"]) if "a2" in params else float(_bin_value(span, "a2") or 1.0)
    a1_raw = params.get("a1", _bin_value(span, "a1"))
    a1 = float(a1_raw) if a1_raw is not None else 0.0
    changes = {}
    if a2 > _A2_HIGH:
        changes = {"a2": (a2, _A2_HIGH)}
        params["a2"] = _A2_HIGH
        if a1 != 0.0:
            changes["a1"] = (a1, 0.0)
            params["a1"] = 0.0
    elif a2 < _A2_LOW:
        changes = {"a2": (a2, _A2_LOW)}
        params["a2"] = _A2_LOW
    return params, changes


def bsim4_check_messages(dev_type: str, L_um: float, W_um: float,
                         params: dict[str, float] | None = None,
                         bin_index: int | None = None,
                         temp_K: float = 77.0) -> list[str]:
    """Warning / fatal lines printed by the BSIM4 model check (b4check.c) for
    the card, from one ``op`` in a fresh process (no clamp written)."""
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    with tempfile.TemporaryDirectory(prefix="cryoml_b4check_") as td_s:
        td = Path(td_s)
        lib_path = LIB_FILE
        if params:
            bin_idx = bin_index if bin_index is not None else find_bin_index(dev_type, L_um, W_um)
            base = PDK77K_DIR / _CORNER_BASENAME[dev_type]
            corner = td / _CORNER_BASENAME[dev_type]
            corner.write_text(_patch_params_in_bin(base.read_text(errors="replace"),
                                                   dev_type, bin_idx, params))
            lib_path = td / "lib77k.spice"
            lib_path.write_text(LIB_FILE.read_text().replace(base.as_posix(), corner.as_posix()))
        deck = td / "check.sp"
        deck.write_text(
            "* BSIM4 model check\n.options scale=1.0\n"
            f".options temp={temp_K - 273.15:.4f}\n"
            f'.lib "{lib_path.as_posix()}" tt_77k\n'
            + _instance_line(dev_type, L_um, W_um)
            + "VG ng 0 DC 0\nVD nd 0 DC 0\n.control\nop\nquit\n.endc\n.end\n")
        proc = subprocess.run([_NGSPICE_BIN, "-b", str(deck)], cwd=str(td),
                              capture_output=True, timeout=_DEFAULT_TIMEOUT_S,
                              text=True, check=False)
    lines = (proc.stdout + "\n" + proc.stderr).splitlines()
    return [ln.strip() for ln in lines
            if ln.strip().lower().startswith(("warning:", "fatal:"))]


def _batch_deck(dev_type: str, L_um: float, W_um: float, curves: list[Curve],
                lib_path: Path, raw_dir: Path, temp_K: float,
                temps_K: list[float] | None = None,
                rs_ext_ohm: float = 0.0, warmup_op: bool = False) -> str:
    """One deck that runs every curve's DC sweep in a single ngspice process.

    With ``temps_K`` the sweeps repeat at each temperature (``option temp``
    inside the control block re-evaluates ``temper`` expressions); outputs are
    ``raw_t{j}_{i}.dat``.  ``warmup_op`` runs one ``op`` before the sweeps
    (``bsim4_state="unclamped"``)."""
    instance = _instance_line(dev_type, L_um, W_um, rs_ext_ohm)
    lines = [
        "* corrected-repository batched DC sweeps\n",
        ".options scale=1.0\n",
        f".options temp={(temps_K[0] if temps_K else temp_K) - 273.15:.4f}\n",
        f'.lib "{lib_path.as_posix()}" tt_77k\n',
        instance,
    ]
    lines += [
        "VG ng 0 DC 0\n",
        "VD nd 0 DC 0\n",
        ".control\n",
        "set filetype=ascii\n",
    ]
    if warmup_op:
        lines.append("op\n")
    if temps_K:
        for j, t in enumerate(temps_K):
            lines.append(f"option temp={t - 273.15:.4f}\n")
            lines += _sweep_lines(curves, raw_dir, f"raw_t{j}_")
    else:
        lines += _sweep_lines(curves, raw_dir, "raw_")
    lines += ["quit\n", ".endc\n", ".end\n"]
    return "".join(lines)


def _sweep_lines(curves: list[Curve], raw_dir: Path, prefix: str) -> list[str]:
    lines = []
    for i, c in enumerate(curves):
        if c.kind == "idvg":
            v = np.asarray(c.Vg, dtype=np.float64)
            fixed = float(np.median(np.asarray(c.Vd, dtype=np.float64)))
            bias_src, sweep_src = "VD", "VG"
        else:
            v = np.asarray(c.Vd, dtype=np.float64)
            fixed = float(np.median(np.asarray(c.Vg, dtype=np.float64)))
            bias_src, sweep_src = "VG", "VD"
        # Sweep in the curve's own direction (upstream pMOS sweeps run
        # 0 -> -1.85; direction changes the DC-continuation path and shifts
        # weak-inversion currents at the nA level).
        v_start, v_stop = float(v[0]), float(v[-1])
        v_step = (v_stop - v_start) / max(len(v) - 1, 1)
        if v_step == 0:
            v_step = 0.05
        lines += [
            f"alter {bias_src} dc = {fixed:.6g}\n",
            f"dc {sweep_src} {v_start:.6g} {v_stop:.6g} {v_step:.6g}\n",
            f"wrdata {(raw_dir / f'{prefix}{i}.dat').as_posix()} -i(VD)\n",
        ]
    return lines


def simulate_pdk(
    dev_type: str,
    L_um: float,
    W_um: float,
    curves: Iterable[Curve],
    params: dict[str, float] | None = None,
    temp_K: float = 77.0,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    bin_index: int | None = None,
    library_path: Path | None = None,
    temps_K: list[float] | None = None,
    rs_ext_ohm: float = 0.0,
    bsim4_state: str = "clamped",
) -> list[np.ndarray]:
    """Simulate measured-bias curves with the corrected-repository deck.

    ``params`` overrides values inside ``bin_index`` or, when omitted, the
    geometry-matching bin. ``library_path`` validates an exported combined
    card and cannot be combined with parameter injection. NGSpice always
    performs native L/W bin selection.  ``rs_ext_ohm`` > 0 adds equal
    extrinsic drain and source resistors outside the card (``_instance_line``).
    ``bsim4_state`` selects the BSIM4 model-check clamp state of the curves
    (``BSIM4_STATES``; default "clamped": every curve in the state of a fresh
    single-analysis process).
    """
    if bsim4_state not in BSIM4_STATES:
        raise ValueError(f"bsim4_state {bsim4_state!r} not in {BSIM4_STATES}")
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    if params and library_path is not None:
        raise ValueError("cannot inject params while validating an exported library")
    curves = list(curves)
    out: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="cryoml_pdk_sim_") as td_s:
        td = Path(td_s)
        lib_path = Path(library_path) if library_path is not None else LIB_FILE
        if params:
            bin_idx = bin_index
            if bin_idx is None:
                bin_idx = find_bin_index(dev_type, L_um, W_um)
            if bin_idx is None:
                raise ValueError(f"no bin for {dev_type} L={L_um} W={W_um}")
            if bsim4_state == "clamped":
                params, _ = bsim4_clamp_params(dev_type, bin_idx, params)
            base = PDK77K_DIR / _CORNER_BASENAME[dev_type]
            patched = base.read_text(errors="replace")
            patched = _patch_params_in_bin(patched, dev_type, bin_idx, params)
            corner_path = td / _CORNER_BASENAME[dev_type]
            corner_path.write_text(patched)
            lib_text = LIB_FILE.read_text().replace(
                (PDK77K_DIR / _CORNER_BASENAME[dev_type]).as_posix(),
                corner_path.as_posix())
            lib_path = td / "lib77k.spice"
            lib_path.write_text(lib_text)
        deck_path = td / "deck_batch.sp"
        deck_path.write_text(
            _batch_deck(dev_type, L_um, W_um, curves, lib_path, td, temp_K,
                        temps_K, rs_ext_ohm, warmup_op=bsim4_state == "unclamped"))
        try:
            proc = subprocess.run([_NGSPICE_BIN, "-b", str(deck_path)],
                                  cwd=str(td), capture_output=True,
                                  timeout=timeout_s, text=True, check=False)
            txt = (proc.stdout + proc.stderr).lower()
            fatal = "could not find a valid modelname" in txt
        except subprocess.TimeoutExpired:
            fatal = True
        if temps_K:
            return [_read_curves(td, f"raw_t{j}_", curves, dev_type, fatal)
                    for j in range(len(temps_K))]
        out = _read_curves(td, "raw_", curves, dev_type, fatal)
    return out


def _read_curves(td: Path, prefix: str, curves: list[Curve], dev_type: str,
                 fatal: bool) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for idx, c in enumerate(curves):
        target = np.asarray(c.Vg if c.kind == "idvg" else c.Vd, dtype=np.float64)
        raw_path = td / f"{prefix}{idx}.dat"
        sim_v = sim_id = None
        if not fatal and raw_path.exists():
            rows = []
            with raw_path.open("r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            rows.append((float(parts[0]), float(parts[1])))
                        except ValueError:
                            continue
            if rows:
                arr = np.array(rows, dtype=np.float64)
                sim_v, sim_id = arr[:, 0], arr[:, 1]
        if sim_v is None or sim_id is None or len(sim_v) != len(sim_id):
            out.append(np.full_like(target, np.nan, dtype=np.float64))
            continue
        order = np.argsort(sim_v)
        id_resampled = np.interp(target, sim_v[order], sim_id[order])
        if dev_type == "pmos":
            id_resampled = -id_resampled
        out.append(id_resampled)
    return out
