"""Truth-only NGSpice simulation for the misspecified SkyWater DEV tier (sky130c).

The fitting side never uses this module: every method keeps fitting with
``spice_pdk.simulate_pdk`` (stock instance line, stock non-fitted card
values, one 77 K process per call).  This module only generates targets
whose structure the fitted card cannot represent:

* ``rs_ext_ohm > 0`` adds extrinsic drain and source resistors outside the
  card.  The instance prefix ``Xm1 nd ng 0 0`` becomes ``Xm1 ndi ng nsi 0``
  and the lines ``RDEXT nd ndi R`` and ``RSEXT nsi 0 R`` are added (as in
  ``cryoml.pdk2.gf180._instance``).  The body stays at node 0.
* ``curve_params`` holds one parameter-override dict per curve (merged over
  ``params``).  Curves whose merged overrides differ run in separate ngspice
  processes; curves with identical merged overrides share one process.

With ``rs_ext_ohm == 0`` and ``curve_params is None`` the deck is the
``spice_pdk._batch_deck`` deck at 77 K (same lines in the same order), so the
result equals ``simulate_pdk`` with the same ``params`` and ``bin_index``
(checked by ``tests/test_sky130_misspec.py``).  ``spice_pdk`` is imported,
never edited.

BSIM4 model-check clamp state (I14-1): ``bsim4_state="clamped"`` (default)
writes the ``b4check.c`` A1/A2 clamp into the card
(``spice_pdk.bsim4_clamp_params``), as ``simulate_pdk``; "legacy" writes the
card unchanged (reproduction of decks made before I14-1).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .spice_pdk import (LIB_FILE, PDK77K_DIR, _CORNER_BASENAME, _NGSPICE_BIN,
                        _bin_spans, _instance_line, _patch_params_in_bin,
                        _read_curves, _sweep_lines, bsim4_clamp_params,
                        device_mult, ensure_pdk77k)

TEMP_K = 77.0
STOCK_PREFIX = "Xm1 nd ng 0 0 "
RESISTOR_PREFIX = "Xm1 ndi ng nsi 0 "

# Non-fitted BSIM4 parameters perturbed by the sky130c generator.  None of
# them is in the round-1 43-parameter fitted set (asserted by the generator).
MISSPEC = ("keta", "etab", "pdiblc1", "pdiblcb", "pscbe1", "pscbe2", "dvt2",
           "dvt0w", "dvt2w", "uc", "lint", "wint")

__all__ = ["MISSPEC", "TEMP_K", "bin_values", "device_mult", "instance_lines",
           "simulate_truth", "truth_deck"]


def instance_lines(dev_type: str, L_um: float, W_um: float,
                   rs_ext_ohm: float = 0.0) -> str:
    """``spice_pdk._instance_line``; with ``rs_ext_ohm > 0`` the drain and
    source move to internal nodes behind two extrinsic resistors."""
    line = _instance_line(dev_type, L_um, W_um)
    if rs_ext_ohm <= 0:
        return line
    if not line.startswith(STOCK_PREFIX):
        raise ValueError(f"unexpected instance line {line!r}")
    return (RESISTOR_PREFIX + line[len(STOCK_PREFIX):]
            + f"RDEXT nd ndi {rs_ext_ohm:.10g}\n"
            + f"RSEXT nsi 0 {rs_ext_ohm:.10g}\n")


def truth_deck(dev_type: str, L_um: float, W_um: float, curves: list,
               lib_path: Path, raw_dir: Path, rs_ext_ohm: float = 0.0) -> str:
    """The ``spice_pdk._batch_deck`` deck (single temperature, 77 K) with
    ``instance_lines`` in place of the stock instance line."""
    lines = [
        "* corrected-repository batched DC sweeps\n",
        ".options scale=1.0\n",
        f".options temp={TEMP_K - 273.15:.4f}\n",
        f'.lib "{lib_path.as_posix()}" tt_77k\n',
        instance_lines(dev_type, L_um, W_um, rs_ext_ohm),
        "VG ng 0 DC 0\n",
        "VD nd 0 DC 0\n",
        ".control\n",
        "set filetype=ascii\n",
    ]
    lines += _sweep_lines(curves, raw_dir, "raw_")
    lines += ["quit\n", ".endc\n", ".end\n"]
    return "".join(lines)


def _run(dev_type: str, L_um: float, W_um: float, curves: list,
         params: Mapping[str, float], bin_index: int, rs_ext_ohm: float,
         timeout_s: float, bsim4_state: str = "clamped") -> list[np.ndarray]:
    """One ngspice process for ``curves`` (mirrors ``simulate_pdk``)."""
    if bsim4_state not in ("clamped", "legacy"):
        raise ValueError(f"bsim4_state {bsim4_state!r} not in ('clamped', 'legacy')")
    with tempfile.TemporaryDirectory(prefix="cryoml_sky130c_") as td_s:
        td = Path(td_s)
        lib_path = LIB_FILE
        if params:
            if bsim4_state == "clamped":
                params, _ = bsim4_clamp_params(dev_type, bin_index, dict(params))
            base = PDK77K_DIR / _CORNER_BASENAME[dev_type]
            patched = _patch_params_in_bin(base.read_text(errors="replace"),
                                           dev_type, bin_index, dict(params))
            corner_path = td / _CORNER_BASENAME[dev_type]
            corner_path.write_text(patched)
            lib_path = td / "lib77k.spice"
            lib_path.write_text(LIB_FILE.read_text().replace(
                base.as_posix(), corner_path.as_posix()))
        deck_path = td / "deck_batch.sp"
        deck_path.write_text(truth_deck(dev_type, L_um, W_um, curves, lib_path,
                                        td, rs_ext_ohm))
        try:
            proc = subprocess.run([_NGSPICE_BIN, "-b", str(deck_path)],
                                  cwd=str(td), capture_output=True,
                                  timeout=timeout_s, text=True, check=False)
            fatal = ("could not find a valid modelname"
                     in (proc.stdout + proc.stderr).lower())
        except subprocess.TimeoutExpired:
            fatal = True
        return _read_curves(td, "raw_", curves, dev_type, fatal)


def _card_key(card: Mapping[str, float]) -> tuple:
    return tuple(sorted((str(k).lower(), repr(v)) for k, v in card.items()))


def simulate_truth(dev_type: str, L_um: float, W_um: float,
                   curves: Iterable, params: Mapping[str, float] | None, *,
                   bin_index: int, rs_ext_ohm: float = 0.0,
                   curve_params: Sequence[Mapping[str, float] | None] | None = None,
                   timeout_s: float = 20.0,
                   bsim4_state: str = "clamped") -> list[np.ndarray]:
    """Simulated currents per curve (``simulate_pdk`` sign convention and
    resampling).  A failed or timed-out process gives NaN curves.
    ``bsim4_state``: "clamped" (default) or "legacy" (module docstring)."""
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    curves = list(curves)
    base = dict(params or {})
    if curve_params is None:
        return _run(dev_type, L_um, W_um, curves, base, bin_index,
                    rs_ext_ohm, timeout_s, bsim4_state)
    if len(curve_params) != len(curves):
        raise ValueError("curve_params needs one entry per curve")
    cards = [{**base, **dict(extra or {})} for extra in curve_params]
    groups: dict[tuple, list[int]] = {}
    for index, card in enumerate(cards):
        groups.setdefault(_card_key(card), []).append(index)
    out: list[np.ndarray | None] = [None] * len(curves)
    for indices in groups.values():
        sims = _run(dev_type, L_um, W_um, [curves[i] for i in indices],
                    cards[indices[0]], bin_index, rs_ext_ohm, timeout_s, bsim4_state)
        for index, sim in zip(indices, sims):
            out[index] = sim
    return out  # type: ignore[return-value]


_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def bin_values(dev_type: str, bin_index: int,
               names: Iterable[str]) -> dict[str, float | None]:
    """Values of ``names`` in one bin of the 77 K corner file.  A plain
    number gives a float; an expression (``{...}`` or ``'...'``), a missing
    name or anything else gives None.  The first ``+ name = value`` line of
    the bin is read (the line ``_patch_params_in_bin`` replaces)."""
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    text = (PDK77K_DIR / _CORNER_BASENAME[dev_type]).read_text(errors="replace")
    span = next((text[s:e] for idx, s, e in _bin_spans(text, dev_type)
                 if idx == bin_index), None)
    if span is None:
        raise ValueError(f"bin {bin_index} not found for {dev_type}")
    out: dict[str, float | None] = {}
    for name in names:
        match = re.search(rf"^\+\s*{re.escape(name)}\s*=\s*([^\n]*)", span,
                          re.IGNORECASE | re.MULTILINE)
        raw = match.group(1).strip() if match else ""
        token = raw.split()[0] if raw else ""
        out[name] = float(token) if _NUMBER.match(token) else None
    return out
