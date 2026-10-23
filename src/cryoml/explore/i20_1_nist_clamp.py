"""I20-1: BSIM4 A2 clamp state of the NIST decks (I8-1 lvt 77 K card, I9-1
room-temperature card), audit and fix, and the selected g v2 recipe on both
NIST DEV datasets.

Pre-registration: ``out/research/explore/I20_1/prereg.md`` / ``prereg.json``.

Clamp states (I13-1 definitions): L = "legacy" (the I8-1 / I9-1 deck: all
sweeps of a device in one ngspice process, the first analysis with the
b4check.c clamp, later analyses with the card values), C = "clamped" (the
clamp written into the card; every curve in the fresh-process state), U =
"unclamped" (one ``op`` before the curves; every curve with the card
values; audit only).  Simulator code: ``i8_1_nist.simulate`` /
``i9_1_nist_rt.simulate`` (``bsim4_state``), clamp helpers
``i8_1_nist.bsim4_clamp_params`` (77 K bins, no size terms) and
``i9_1_nist_rt.bsim4_clamp_params`` (room-temperature bins with
la2 / wa2 / pa2 terms; effective A2 of the instance geometry).

This module holds pure helpers (no training, no simulation except
``check_lines``): the fresh-process model-check deck, the b4check.c A2
message parser, selection blocks and inner folds of the nested lambda
selection, labels, the S3_D gate rules, and the amendment trigger.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .i15_2_ablate import GRID_LAMBDA, inner_folds, inner_score, select_lambda  # noqa: F401
from .i17_2_ihp import primary_label, secondary_label  # noqa: F401
from .i19_1_dataset_gate import THRESHOLD, apply_gate, s3_dataset  # noqa: F401

DATASETS = ("nist4", "nistrt")            # I8-1 lvt (77 K card), I9-1 regular-Vt (room-temperature card)
NAMES = {"nist4": "nist_4K", "nistrt": "nist_rt_card_4K"}
STATES = ("L", "C", "U")
STATE_ARG = {"L": "legacy", "C": "clamped", "U": "unclamped"}
VDD = 1.8                                 # V, NIST supply (I8-1 / I9-1 VDD)
EPS = 0.01                                # V
PEN_STEP = 0.025                          # V
PEN_N = 73                                # 0 .. 1.8 V
N_BLOCKS = 6                              # selection blocks (pre-registered deviation, see prereg)
N_INNER = 3
NI_MARGIN = 1.05
A2_LOW, A2_HIGH = 0.01, 1.0
V1_TOL = 0.0                              # state L reproduces stored rows exactly
A2_MSG_RE = re.compile(r"A2\s*=\s*([0-9eE.+\-]+)\s+is\s+(too large|too small|larger than 1)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# model check (fresh process, one op, no clamp written)
# ---------------------------------------------------------------------------
def check_lines(ds: str, dev, bin_index: int, params: dict | None) -> list[str]:
    """Warning / fatal lines of one fresh-process ``op`` of the card with
    ``params`` (changed parameters only) written into the bin; no clamp."""
    from . import i8_1_nist as b8
    from . import i9_1_nist_rt as b9
    sp = b8._sp()
    sign = -1 if dev.polarity == "pmos" else 1
    with tempfile.TemporaryDirectory(prefix="i20_1_chk_") as td_s:
        td = Path(td_s)
        if ds == "nist4":
            lib = b8._write_lib(td, dev.polarity, bin_index, params or None)
            head = (f".options temp={b8.DECK_TEMP_K - 273.15:.4f}\n", f'.lib "{lib.as_posix()}" tt_77k\n',
                    b8.instance_line(dev.polarity, dev.W_um, dev.L_um))
        else:
            lib = b9.write_lib(td, dev.polarity, bin_index, params or None)
            head = (f".options temp={b9.DECK_TEMP_C:.4f}\n", f'.lib "{lib.as_posix()}" tt_rt\n',
                    b9.instance_line(dev.polarity, dev.W_um, dev.L_um))
        deck = td / "chk.sp"
        deck.write_text("* I20-1 model check\n.options scale=1.0\n" + "".join(head)
                        + f"VG ng 0 DC {sign * 0.9}\nVD nd 0 DC {sign * 0.9}\n.control\nop\nquit\n.endc\n.end\n")
        r = subprocess.run([sp._NGSPICE_BIN, "-b", str(deck)], cwd=str(td), capture_output=True, text=True,
                           timeout=60, check=False)
    lines = (r.stdout + "\n" + r.stderr).splitlines()
    return [ln.strip() for ln in lines if ln.strip().lower().startswith(("warning", "fatal"))]


def a2_from_messages(lines: Sequence[str]) -> float | None:
    """The size-dependent A2 printed by b4check.c when it clamps, else None."""
    for ln in lines:
        mm = A2_MSG_RE.search(ln)
        if mm:
            return float(mm.group(1))
    return None


def clamp_class(a2_eff: float) -> str:
    if a2_eff > A2_HIGH:
        return "high"
    if a2_eff < A2_LOW:
        return "low"
    return "none"


# ---------------------------------------------------------------------------
# nested lambda selection with selection blocks
# ---------------------------------------------------------------------------
def selection_blocks(tags: Sequence[str], is_pmos: Mapping[str, bool], k: int = N_BLOCKS) -> list[list[str]]:
    """Devices sorted by (polarity, nMOS first; tag); the i-th device goes to
    selection block i mod k (the I15-2 inner-fold rule applied to the whole
    dataset)."""
    return inner_folds(list(tags), dict(is_pmos), k)


def block_of(blocks: Sequence[Sequence[str]]) -> dict:
    return {t: j for j, b in enumerate(blocks) for t in b}


def lambda_by_device(blocks: Sequence[Sequence[str]], lam_block: Mapping[int, float]) -> dict:
    """lambda*_h = lambda* of the selection block that holds device h."""
    out = {}
    for j, b in enumerate(blocks):
        for t in b:
            out[t] = float(lam_block[j])
    return out


def variant(lam: float) -> str:
    lam = float(lam)
    if lam not in GRID_LAMBDA:
        raise ValueError(f"lambda {lam} not in {GRID_LAMBDA}")
    return f"sym_e1_l{int(lam)}"


def pen_axis(n: int = PEN_N, step: float = PEN_STEP) -> np.ndarray:
    return np.round(np.arange(int(n)) * float(step), 6)


def grid_curve_specs(is_pmos: bool) -> list[dict]:
    """One Id-Vd sweep per u_g: V(g,s) = s u_g, V(d,s) = s u_d (0 .. 1.8 V), V(b,s) = 0."""
    s = -1.0 if is_pmos else 1.0
    ax = pen_axis()
    return [{"fixed": float(s * ug), "sweep": s * ax} for ug in ax]


# ---------------------------------------------------------------------------
# gate rules and amendment trigger
# ---------------------------------------------------------------------------
def decision_word(s3: float, threshold: float = THRESHOLD) -> str:
    return "apply" if apply_gate(s3, threshold) else "C0_refit"


def amendment_needed(committed: Mapping[str, float], new: Mapping[str, Mapping[str, float]]) -> dict:
    """Pre-registered trigger: a dated amendment of the I19-1 Part P decisions
    is written iff, for some NIST dataset, the decision from any new S3_D
    value (state C v1 recipe, g v2 recipe) differs from the committed one.

    ``committed``: {dataset: committed S3_D}; ``new``: {dataset: {label: S3_D}}."""
    changes = []
    for d, s_old in committed.items():
        w_old = decision_word(s_old)
        for lab, s_new in new.get(d, {}).items():
            w_new = decision_word(s_new)
            if w_new != w_old:
                changes.append({"dataset": d, "statistic": lab, "committed": w_old, "new": w_new,
                                "S3_D_committed": float(s_old), "S3_D_new": float(s_new)})
    return {"needed": bool(changes), "changes": changes}


def frozen_rule_decision(s3_v1_stateC: float) -> str:
    """Frozen-method gate (pre-registered, open item 9): S3_D from the v1
    recipe (I19-1 v1 definition) on the state-C starting records decides
    whether the selected g v2 correction is applied to the dataset."""
    return decision_word(s3_v1_stateC)


def max_abs_diff(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """max |a - b| over common keys; NaN in both counts as 0, in one as inf."""
    out = 0.0
    for k in set(a) & set(b):
        x, y = a[k], b[k]
        fx, fy = (x is not None and math.isfinite(x)), (y is not None and math.isfinite(y))
        if not fx and not fy:
            continue
        if fx != fy:
            return float("inf")
        out = max(out, abs(float(x) - float(y)))
    return out


def decide_validity(v: dict) -> dict:
    """V1 state L reproduces stored rows; V2 C = L = U on unclamped cards;
    V3 effective A2 = b4check.c message value on clamped cards, no A2 message
    on unclamped cards, state-C curve 1 = state-L curve 1 on clamped cards;
    V4 unit tests; V5 rows complete; V6 TEST closed."""
    fails = []
    if not (v["v1_max"] <= V1_TOL):
        fails.append("V1 state L reproduces stored rows")
    if not (v["v2_max"] == 0.0 and v["v2_n"] > 0):
        fails.append("V2 C = L = U on unclamped cards")
    if not (v["v3_a2_rel_max"] <= 1e-5 and v["v3_unclamped_msgs"] == 0 and v["v3_curve1_max"] == 0.0):
        fails.append("V3 effective A2 / curve 1")
    if not v["v4_unit_tests_pass"]:
        fails.append("V4 unit tests")
    if not v["v5_complete"]:
        fails.append("V5 rows complete")
    if not v["v6_test_closed"]:
        fails.append("V6 TEST closed")
    return {"valid": not fails, "fails": fails}
