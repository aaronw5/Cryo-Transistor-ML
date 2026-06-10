"""Corrected-repository NGSpice backend for the 77 K SKY130 model cards."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import CORRECTED_REPO_DIR, DATA_DIR, PROCESSED_DIR
from .data_io import Curve
from .utils import get_logger

logger = get_logger("cryoml.spice_pdk")

_NGSPICE_BIN = os.environ.get("NGSPICE_BIN", "ngspice")
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


def ensure_pdk77k() -> str | None:
    """Create the patched corner copies + lib wrapper. Returns error or None."""
    if not LOD_FILE.exists():
        return f"missing {LOD_FILE}"
    if not PDK_SPICE_DIR.exists():
        return (f"missing Volare PDK at {PDK_SPICE_DIR} — run "
                "`volare enable --pdk sky130 a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b` "
                "with PDK_ROOT=data/raw/pdk")
    PDK77K_DIR.mkdir(parents=True, exist_ok=True)
    for dev, base in _CORNER_BASENAME.items():
        src = CORRECTED_CORNER_DIR / base
        if not src.exists():
            return f"missing corrected-repository corner {src}"
        dst = PDK77K_DIR / base
        text = src.read_text(errors="replace")
        text = re.sub(
            r'^\.include\s+"\./skywater-pdk/[^"]*lod\.spice"',
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


def _bin_geometry(span_text: str) -> dict[str, float]:
    out = {}
    for key in ("lmin", "lmax", "wmin", "wmax"):
        m = re.search(rf"\+\s*{key}\s*=\s*([0-9.eE+\-]+)", span_text)
        if m:
            out[key] = float(m.group(1))
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
    area_um2 = W_um * 0.24
    perimeter_um = W_um + 2 * 0.24
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
            f"Xm1 nd ng 0 0 {subckt} l={L_um:.6g}u w={W_um:.6g}u nf=1 "
            f"ad={area_um2:.6g}p as={area_um2:.6g}p "
            f"pd={perimeter_um:.6g}u ps={perimeter_um:.6g}u\n"
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
    for idx, s, e in _bin_spans(text, dev_type):
        if idx != bin_index:
            continue
        span = text[s:e]
        for k, v in params.items():
            pat = re.compile(rf"^(\+\s*{re.escape(k)}\s*=\s*)[^\n]*", re.IGNORECASE | re.MULTILINE)
            span, n = pat.subn(rf"\g<1>{v:.10g}", span, count=1)
            if n == 0:
                hdr_end = span.find("\n")
                span = span[:hdr_end + 1] + f"+ {k} = {v:.10g}\n" + span[hdr_end + 1:]
        return text[:s] + span + text[e:]
    raise ValueError(f"bin {bin_index} not found for {dev_type}")


def _batch_deck(dev_type: str, L_um: float, W_um: float, curves: list[Curve],
                lib_path: Path, raw_dir: Path, temp_K: float) -> str:
    """One deck that runs every curve's DC sweep in a single ngspice process."""
    subckt = _SUBCKT[dev_type]
    area_um2 = W_um * 0.24
    perimeter_um = W_um + 2 * 0.24
    instance = (
        f"Xm1 nd ng 0 0 {subckt} l={L_um:.6g}u w={W_um:.6g}u nf=1 "
        f"ad={area_um2:.6g}p as={area_um2:.6g}p "
        f"pd={perimeter_um:.6g}u ps={perimeter_um:.6g}u\n")
    lines = [
        "* corrected-repository batched DC sweeps\n",
        ".options scale=1.0\n",
        f".options temp={temp_K - 273.15:.4f}\n",
        f'.lib "{lib_path.as_posix()}" tt_77k\n',
        instance,
    ]
    lines += [
        "VG ng 0 DC 0\n",
        "VD nd 0 DC 0\n",
        ".control\n",
        "set filetype=ascii\n",
    ]
    for i, c in enumerate(curves):
        if c.kind == "idvg":
            v = np.asarray(c.Vg, dtype=np.float64)
            fixed = float(np.median(np.asarray(c.Vd, dtype=np.float64)))
            bias_src, sweep_src = "VD", "VG"
        else:
            v = np.asarray(c.Vd, dtype=np.float64)
            fixed = float(np.median(np.asarray(c.Vg, dtype=np.float64)))
            bias_src, sweep_src = "VG", "VD"
        vs = np.sort(v)
        v_start, v_stop = float(vs[0]), float(vs[-1])
        v_step = (v_stop - v_start) / max(len(vs) - 1, 1)
        if v_step == 0:
            v_step = 0.05
        lines += [
            f"alter {bias_src} dc = {fixed:.6g}\n",
            f"dc {sweep_src} {v_start:.6g} {v_stop:.6g} {v_step:.6g}\n",
            f"wrdata {(raw_dir / f'raw_{i}.dat').as_posix()} -i(VD)\n",
        ]
    lines += ["quit\n", ".endc\n", ".end\n"]
    return "".join(lines)


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
) -> list[np.ndarray]:
    """Simulate measured-bias curves with the corrected-repository deck.

    ``params`` overrides values inside ``bin_index`` or, when omitted, the
    geometry-matching bin. ``library_path`` validates an exported combined
    card and cannot be combined with parameter injection. NGSpice always
    performs native L/W bin selection.
    """
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
            _batch_deck(dev_type, L_um, W_um, curves, lib_path, td, temp_K))
        try:
            proc = subprocess.run([_NGSPICE_BIN, "-b", str(deck_path)],
                                  cwd=str(td), capture_output=True,
                                  timeout=timeout_s, text=True, check=False)
            txt = (proc.stdout + proc.stderr).lower()
            fatal = "could not find a valid modelname" in txt
        except subprocess.TimeoutExpired:
            fatal = True
        for idx, c in enumerate(curves):
            target = np.asarray(c.Vg if c.kind == "idvg" else c.Vd, dtype=np.float64)
            raw_path = td / f"raw_{idx}.dat"
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
