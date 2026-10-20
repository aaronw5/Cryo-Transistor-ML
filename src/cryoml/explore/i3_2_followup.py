"""I3-2 follow-up: SPICE-native run of the exported Verilog-A correction (F1)
and helpers for the robustness variants (F2).

Pre-registration: ``out/research/explore/I3_2/followup_prereg.json``.

F1 compiles the ``to_verilog_a_bias`` text with OpenVAF-Reloaded and runs it
in ngspice-46 (the arm64 ngspice-41 build has no OSDI support) around the
sky130 BSIM4 instance of ``cryoml.spice_pdk``:

* ``osdi_va_text``: the exported text with one change.  OpenVAF treats the
  constant-zero contribution ``V(sense) <+ 0.0`` as a collapse of the two
  terminals d and di, which the ngspice netlist does not apply, so no
  current flows through the sense branch.  ``V(sense) <+ RS * I(sense)``
  with the model parameter ``RS = 0.0`` is a zero-volt source with a
  branch-current unknown.
* ``insert_correction``: the ``spice_pdk._batch_deck`` deck with the BSIM4
  drain moved to node ``ndi`` and the OSDI correction between ``nd`` and
  ``ndi``.
* ``simulate_sky130``: ``spice_pdk.simulate_pdk`` with an explicit ngspice
  binary and an optional correction; also returns convergence messages.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..config import ROOT

OPENVAF_R = (ROOT / "data/raw/pdk2/ihp_sg13g2/tools/openvaf-r-v24.0.2mob-macos-aarch64"
             / "openvaf-r-v24.0.2mob-macos-aarch64/bin/openvaf-r")
SENSE_EXPORTED = "    V(sense) <+ 0.0;"
SENSE_OSDI = "    V(sense) <+ RS * I(sense);"
RS_DECL = "  parameter real RS = 0.0;   // ohm; sense branch V = RS * I (0: zero-volt source)"
VA_MODULE = "i3_2_bias_corr"
MODEL_NAME = "i32corr"
CONVERGENCE_MESSAGES = ("singular matrix", "timestep too small", "no convergence",
                        "gmin stepping failed", "source stepping failed",
                        "iteration limit reached", "simulation(s) aborted", "error")


# ---------------------------------------------------------------------------
# Verilog-A text and netlist (pure)
# ---------------------------------------------------------------------------
def osdi_va_text(text: str) -> str:
    """Exported ``to_verilog_a_bias`` text with the sense line changed for
    OSDI (see module docstring); every other line is unchanged."""
    if text.count(SENSE_EXPORTED) != 1:
        raise ValueError("exported sense line not found exactly once")
    anchor = "  parameter real VDD = "
    if text.count(anchor) != 1:
        raise ValueError("VDD parameter line not found exactly once")
    head, tail = text.split(anchor, 1)
    vdd_line, rest = tail.split("\n", 1)
    out = head + anchor + vdd_line + "\n" + RS_DECL + "\n" + rest
    return out.replace(SENSE_EXPORTED, SENSE_OSDI)


def insert_correction(deck: str, osdi_path: Path | str, pmos: bool, vdd: float,
                      module: str = VA_MODULE, model: str = MODEL_NAME) -> str:
    """Add the OSDI correction to a ``spice_pdk._batch_deck`` deck.

    The instance ``Xm1 nd ng 0 0 ...`` becomes ``Xm1 ndi ng 0 0 ...``; the
    correction ``NCORR nd ng 0 ndi`` and its ``.model`` follow it, and
    ``pre_osdi`` is the first line of the control block."""
    lines = deck.splitlines(keepends=True)
    inst = [i for i, ln in enumerate(lines) if ln.startswith("Xm1 nd ng 0 0 ")]
    ctrl = [i for i, ln in enumerate(lines) if ln.strip() == ".control"]
    if len(inst) != 1 or len(ctrl) != 1:
        raise ValueError("deck does not have exactly one Xm1 instance and one .control")
    i = inst[0]
    lines[i] = "Xm1 ndi ng 0 0 " + lines[i][len("Xm1 nd ng 0 0 "):]
    add = [f"NCORR nd ng 0 ndi {model}\n",
           f".model {model} {module} pmos={1 if pmos else 0} VDD={vdd:.10g} RS=0\n"]
    lines[i + 1:i + 1] = add
    c = ctrl[0] + len(add)
    lines[c + 1:c + 1] = [f"pre_osdi {Path(osdi_path).as_posix()}\n"]
    return "".join(lines)


def add_options(deck: str, options: str) -> str:
    """Insert ``.options <options>`` after the deck's ``.options scale=1.0``."""
    anchor = ".options scale=1.0\n"
    if deck.count(anchor) != 1:
        raise ValueError("deck does not have exactly one '.options scale=1.0' line")
    return deck.replace(anchor, anchor + f".options {options}\n")


def convergence_messages(text: str) -> list[str]:
    """Convergence/error messages (``CONVERGENCE_MESSAGES``) found in ngspice
    output, one entry per matching line."""
    out = []
    for line in text.lower().splitlines():
        if any(m in line for m in CONVERGENCE_MESSAGES):
            out.append(line.strip())
    return out


def max_rel_diff(a: Sequence[np.ndarray], b: Sequence[np.ndarray],
                 masks: Sequence[np.ndarray] | None = None, floor: float = 0.0) -> float:
    """max |a - b| / |b| over points with |b| > ``floor`` (and ``masks`` True).

    Non-finite values in ``a`` or ``b`` at a counted point give ``inf``."""
    worst = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        keep = np.ones(len(y), bool) if masks is None else np.array(masks[i], dtype=bool)
        bad = keep & ~(np.isfinite(x) & np.isfinite(y))
        if bad.any():
            return float("inf")
        keep &= np.abs(y) > floor
        if keep.any():
            worst = max(worst, float(np.max(np.abs(x[keep] - y[keep]) / np.abs(y[keep]))))
    return worst


def scored_masks(rec) -> list[np.ndarray]:
    """Per curve of a ``DeviceRec``: True at scored points (included curve,
    index >= trim start)."""
    out = []
    for c in rec.curves:
        m = np.zeros(len(c.sim), bool)
        if c.included and c.den > 0:
            m[c.start:] = True
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Compilation and simulation
# ---------------------------------------------------------------------------
def compile_va(text: str, va: Path, osdi: Path, openvaf: Path | str = OPENVAF_R) -> dict:
    """Write ``text`` to ``va`` and compile it to ``osdi`` (skipped when the
    existing ``osdi`` was built from the same text, by sha256)."""
    va, osdi = Path(va), Path(osdi)
    va.parent.mkdir(parents=True, exist_ok=True)
    osdi.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode()).hexdigest()
    stamp = osdi.with_suffix(".sha256")
    if osdi.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        return {"va": va, "osdi": osdi, "sha256": digest, "compiled": False, "log": ""}
    va.write_text(text)
    proc = subprocess.run([str(openvaf), "-o", str(osdi), str(va)], capture_output=True,
                          text=True, timeout=300, check=False)
    log = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    if proc.returncode != 0 or not osdi.exists():
        raise RuntimeError(f"openvaf-r failed for {va}:\n{log[-2000:]}")
    stamp.write_text(digest + "\n")
    return {"va": va, "osdi": osdi, "sha256": digest, "compiled": True, "log": log}


def simulate_sky130(dev_type: str, L_um: float, W_um: float, curves: Iterable,
                    params: dict | None, bin_index: int | None, ngspice_bin: str,
                    osdi: Path | None = None, vdd: float = 1.85, options: str = "",
                    timeout_s: float = 120.0) -> tuple[list[np.ndarray], list[str]]:
    """``spice_pdk.simulate_pdk`` (same deck, card patching and read-back)
    with an explicit ngspice binary and, when ``osdi`` is given, the
    correction inserted by ``insert_correction``.  ``options`` (e.g.
    ``"reltol=1e-6 abstol=1e-15"``) adds one ``.options`` line after the
    deck's ``.options scale=1.0`` (diagnostic runs only; empty = the
    canonical deck).  Returns the currents (measured-grid order of
    ``curves``) and the convergence messages."""
    from .. import spice_pdk as sp
    err = sp.ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    curves = list(curves)
    with tempfile.TemporaryDirectory(prefix="cryoml_i32f_") as td_s:
        td = Path(td_s)
        lib_path = sp.LIB_FILE
        if params:
            bin_idx = bin_index if bin_index is not None else sp.find_bin_index(
                dev_type, L_um, W_um)
            base = sp.PDK77K_DIR / sp._CORNER_BASENAME[dev_type]
            patched = sp._patch_params_in_bin(base.read_text(errors="replace"), dev_type,
                                              bin_idx, params)
            corner = td / sp._CORNER_BASENAME[dev_type]
            corner.write_text(patched)
            lib_path = td / "lib77k.spice"
            lib_path.write_text(sp.LIB_FILE.read_text().replace(base.as_posix(),
                                                                corner.as_posix()))
        deck = sp._batch_deck(dev_type, L_um, W_um, curves, lib_path, td, 77.0)
        if options:
            deck = add_options(deck, options)
        if osdi is not None:
            deck = insert_correction(deck, osdi, dev_type == "pmos", vdd)
        deck_path = td / "deck_batch.sp"
        deck_path.write_text(deck)
        try:
            proc = subprocess.run([ngspice_bin, "-b", str(deck_path)], cwd=str(td),
                                  capture_output=True, timeout=timeout_s, text=True,
                                  check=False)
            text = proc.stdout + proc.stderr
            fatal = "could not find a valid modelname" in text.lower()
            msgs = convergence_messages(text)
        except subprocess.TimeoutExpired:
            fatal, msgs = True, ["timeout"]
        sims = sp._read_curves(td, "raw_", curves, dev_type, fatal)
    return sims, msgs


# ---------------------------------------------------------------------------
# F2 variants (pure)
# ---------------------------------------------------------------------------
F2_REAL_VARIANTS = {
    "a_seeds_5_9": {"hyper": {"seeds": (5, 6, 7, 8, 9)}, "train_cards": "start"},
    "a_seeds_10_14": {"hyper": {"seeds": (10, 11, 12, 13, 14)}, "train_cards": "start"},
    "a_seeds_15_19": {"hyper": {"seeds": (15, 16, 17, 18, 19)}, "train_cards": "start"},
    "b_width8": {"hyper": {"hidden": 8}, "train_cards": "start"},
    "b_width32": {"hyper": {"hidden": 32}, "train_cards": "start"},
    "b_depth1": {"hyper": {"layers": 1}, "train_cards": "start"},
    "c_train_round1_cards": {"hyper": {}, "train_cards": "round1"},
}
F2_GROUPS = {"a": ("a_seeds_5_9", "a_seeds_10_14", "a_seeds_15_19"),
             "b": ("b_width8", "b_width32", "b_depth1"),
             "c": ("c_train_round1_cards",)}


def network_dims(n_in: int, hyper: dict) -> list[int]:
    from .i3_2_bias import HYPER_BIAS
    hp = {**HYPER_BIAS, **hyper}
    return [n_in] + [hp["hidden"]] * hp["layers"] + [1]


def real_variant_fails(cmp_vs_c0: dict, n_failed: int) -> bool:
    """Prereg F2 REAL_DEV variant failure: CI upper bound of R(variant,
    C0_refit) >= 1 (or not finite) or any failed device."""
    hi = cmp_vs_c0.get("ci_high", float("nan"))
    return bool(n_failed > 0 or not np.isfinite(hi) or hi >= 1.0)


def tier_b_pdk_passes(cmp_vs_c0: dict, n_failed: int, ratio_max: float = 0.90) -> bool:
    """Prereg F2(d) per-PDK pass: R(M_bias, C0_refit) <= 0.90 and no failed target."""
    r = cmp_vs_c0.get("ratio", float("nan"))
    return bool(n_failed == 0 and np.isfinite(r) and r <= ratio_max)
