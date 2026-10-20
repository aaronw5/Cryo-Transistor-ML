"""I6-1: the I3-2 bias-only correction on TU Delft 40-nm CMOS at 4.2 K and 295 K.

Pre-registration: ``out/research/explore/I6_1/prereg.json``.

Data: the DC folder of the 4TU dataset "Data underlying the publication:
Cryogenic Characterization of Low-Frequency Noise in 40-nm CMOS" (G. Kiene et
al., TU Delft; doi:10.4121/c41c26f7-586f-48ca-8abe-9127b2d97c60.v1, CC0),
local copy ``data/external/delft_kiene_40nm_lfn`` (I3-1 download).  Each file
is a NumPy ``.npy`` holding one pickled dict; ``read_npy_dict`` reads it with
an unpickler restricted to the numpy array-reconstruction globals, so no code
from the download is run.

Used files: campaign 1 (``*_dc_camp_1_*``), the 69 low-Vt devices with a
campaign-1 file at both ``temp_4`` (stage 4.13-4.14 K) and ``temp_300``
(stage 295.0 K).  Per device and temperature:

* ``i_d_const_d``: 12 Id-Vg curves (Vg = ``v_g_fine_sweep``, 111 points) at
  Vd = ``v_d_matrix_sweep`` (0.05 V, then 11 values up to 1.1 V);
* ``i_d_const_g``: 11 Id-Vd curves (Vd = ``v_d_fine_sweep``, 0..1.1 V, 111
  points) at Vg = ``v_g_matrix_sweep``.

Voltages are gate-source and drain-source (pMOS negative), the current is the
drain current with nMOS > 0, pMOS < 0; bulk is taken as tied to the source.
The 11-point ``i_d_diode`` / ``i_d_triode`` subsets are not used.

Curve keys: ``idvg@d0..d11`` (ascending |Vd|), ``idvd@g0..g10`` (ascending
|Vg|).  Inclusion and trimming: ``cryoml.pdk2.metric`` (frozen from the
measured current).

Compact model: BSIM4 (level 54) cards ``N1`` / ``P1`` from ngspice-41
``examples/mos/ro_17_4.cir`` ("50nm BSIM4 models VDD=1V", CMOSedu.com, R. J.
Baker, CMOS Circuit Design, Layout, and Simulation), distributed with ngspice
under the Modified BSD license; copied verbatim to ``CARD_FILE``.  The
temperature-coefficient parameters ``TC_ZERO`` are set to 0, so card values
apply at the deck temperature (``.options temp`` = stage temperature) and
temperature enters only through BSIM4's built-in kT/q, band-gap and
intrinsic-carrier terms.  Simulator: ngspice-41 native arm64 build
(``NGSPICE``).
"""

from __future__ import annotations

import io
import json
import math
import pickle
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import ROOT
from ..metrics import clean_current
from ..parameter_space import ParameterBox
from ..pdk2 import metric as pdk2_metric
from ..pdk2.ihp import _card_value_text, patch_card
from .e1_correction import CurveRec, DeviceRec
from .i5_1_ihp_real import MeasCurve, read_sweeps, sweep_lines

DATA_DIR = (ROOT / "data" / "external" / "delft_kiene_40nm_lfn" / "repository_for_upload_slim"
            / "measurement_automation" / "main" / "results" / "dc")
DOI = "10.4121/c41c26f7-586f-48ca-8abe-9127b2d97c60.v1"
OUT = ROOT / "out" / "research" / "explore" / "I6_1"
CARD_FILE = OUT / "card" / "ngspice41_ro_17_4_cmosedu50nm_bsim4.mod"
CARD_SOURCE = Path.home() / "cryo-ng41" / "src" / "ngspice-41" / "examples" / "mos" / "ro_17_4.cir"
NGSPICE = Path.home() / "cryo-ng41" / "arm64" / "bin" / "ngspice"
CARD_MODEL = "i6card"
CAMPAIGN = 1
TEMPS_K = (4, 295)                 # labels; deck temperature = stage reading
FILE_TEMP = {4: 4, 295: 300}       # file-name temperature for each label
VDD = 1.1                          # nominal core supply; sweeps reach 1.1 V
TIMEOUT_S = 10.0                   # s per ngspice process (normal: 0.03 s); at 4.2 K some cards do not converge
TC_ZERO = ("kt1", "kt1l", "kt2", "ute", "ua1", "ub1", "uc1", "at", "prt")
N_TIMEOUT = 0                      # ngspice processes stopped by TIMEOUT_S (this process)
_FILE_RE = re.compile(r"^([np])_dc_camp_(\d+)_sample_(\d+)_device_(\d+)_temp_(\d+)\.npy$")


# ---------------------------------------------------------------------------
# Restricted reading of the pickled .npy files
# ---------------------------------------------------------------------------
_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar")}


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"global {module}.{name} not allowed")
        import importlib
        return getattr(importlib.import_module(module), name)


def read_npy_dict(path: Path) -> dict:
    """The dict in a one-element object ``.npy`` (restricted unpickler)."""
    f = io.BytesIO(Path(path).read_bytes())
    version = np.lib.format.read_magic(f)
    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
              else np.lib.format.read_array_header_2_0)
    shape, _, dtype = reader(f)
    if dtype != np.dtype(object) or shape != (1,):
        raise ValueError(f"{path}: expected a (1,) object array, got {shape} {dtype}")
    arr = _NumpyOnly(f).load()
    obj = arr[0] if isinstance(arr, np.ndarray) else arr
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: payload is {type(obj)}")
    return obj


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class MeasDevice:
    tag: str
    dev_type: str              # nmos | pmos
    W_um: float
    L_um: float
    sample: int
    device: int
    sub_block: str             # LC | HC (array of the chip)
    temps_K: dict = field(default_factory=dict)      # label -> stage temperature (K)
    files: dict = field(default_factory=dict)        # label -> file name
    curves: dict = field(default_factory=dict)       # label -> list[MeasCurve]

    @property
    def group(self) -> str:
        """(polarity, W, L) group: the held-out unit of a fold."""
        return self.tag.rsplit("_d", 1)[0]

    @property
    def wl(self) -> str:
        f = lambda x: f"{x:g}".replace(".", "p")
        return f"W{f(self.W_um)}_L{f(self.L_um)}"

    @property
    def vdd(self) -> float:
        return VDD


def device_tag(dev_type: str, W_um: float, L_um: float, device: int) -> str:
    f = lambda x: f"{x:g}".replace(".", "p")
    return f"t40_{dev_type}_W{f(W_um)}_L{f(L_um)}_d{device}"


def file_curves(obj: dict) -> list[MeasCurve]:
    """The 12 Id-Vg and 11 Id-Vd curves of one file, keyed by position."""
    s, r = obj["dc_sett"], obj["dc_results"]
    vg_f = np.asarray(s["v_g_fine_sweep"], dtype=np.float64)
    vd_f = np.asarray(s["v_d_fine_sweep"], dtype=np.float64)
    vd_m = np.asarray(s["v_d_matrix_sweep"], dtype=np.float64)
    vg_m = np.asarray(s["v_g_matrix_sweep"], dtype=np.float64)
    idg, idd = list(r["i_d_const_d"]), list(r["i_d_const_g"])
    if len(idg) != len(vd_m) or len(idd) != len(vg_m):
        raise ValueError("curve count differs from the bias list")
    empty = np.zeros(0)
    out = []
    for prefix, kind, fixed, sweep, ids in (("idvg@d", "idvg", vd_m, vg_f, idg),
                                            ("idvd@g", "idvd", vg_m, vd_f, idd)):
        order = np.argsort(np.abs(fixed), kind="stable")
        for k, j in enumerate(order):
            i_d = np.asarray(ids[j], dtype=np.float64)
            if i_d.shape != sweep.shape:
                raise ValueError("current and sweep lengths differ")
            out.append(MeasCurve(f"camp{CAMPAIGN}_{kind}", kind, float(fixed[j]), 0.0,
                                 sweep.copy(), i_d.copy(), empty, empty, empty, f"{prefix}{k}"))
    return out


def discover(root: Path = DATA_DIR) -> list[MeasDevice]:
    """The campaign-1 devices with files at both temperatures."""
    found: dict[tuple, dict] = {}
    for path in sorted(Path(root).glob("*.npy")):
        m = _FILE_RE.match(path.name)
        if not m or int(m.group(2)) != CAMPAIGN:
            continue
        key = (m.group(1), int(m.group(3)), int(m.group(4)))
        found.setdefault(key, {})[int(m.group(5))] = path
    devices = []
    for (pol, sample, dev_no), by_t in sorted(found.items()):
        if not all(FILE_TEMP[t] in by_t for t in TEMPS_K):
            continue
        dev = None
        for label in TEMPS_K:
            path = by_t[FILE_TEMP[label]]
            obj = read_npy_dict(path)
            setup = obj["setup"]
            dd = setup["dev_dict"]
            if dd["type"] != pol or int(setup["campaingn"]) != CAMPAIGN or \
                    int(setup["sample"]) != sample:
                raise ValueError(f"{path.name}: header differs from the file name")
            if int(dd["NF"]) != 1 or dd["vth"] != "l":
                raise ValueError(f"{path.name}: unexpected NF / Vt flavor")
            W_um, L_um = int(dd["W"]) / 1000.0, int(dd["L"]) / 1000.0
            dev_type = {"n": "nmos", "p": "pmos"}[pol]
            if dev is None:
                dev = MeasDevice(device_tag(dev_type, W_um, L_um, dev_no), dev_type, W_um, L_um,
                                 sample, dev_no, str(dd["sub_block"]))
            elif (W_um, L_um) != (dev.W_um, dev.L_um):
                raise ValueError(f"{path.name}: geometry differs between temperatures")
            r = obj["dc_results"]
            dev.temps_K[label] = 0.5 * (float(r["temperature_meas_start"])
                                        + float(r["temperature_meas_end"]))
            dev.files[label] = path.name
            dev.curves[label] = file_curves(obj)
        devices.append(dev)
    return sorted(devices, key=lambda d: d.tag)


def fit_curves(dev: MeasDevice, temp_k: int) -> list[MeasCurve]:
    order = [f"idvg@d{k}" for k in range(12)] + [f"idvd@g{k}" for k in range(11)]
    by_key = {c.key: c for c in dev.curves[temp_k]}
    return [by_key[k] for k in order]


def inclusion(curve: MeasCurve) -> dict:
    start, _, den = pdk2_metric.curve_trim(curve.kind, curve.id)
    inc = bool(np.isfinite(den) and den > pdk2_metric.INCLUDE_MIN[curve.kind])
    return {"start": int(start), "den": float(den), "included": inc}


INDEX_COLUMNS = "device|T_label|i|key|kind|fixed_V|n_points (npz prefix d<device order>_T<label>_c<i>)"


def save_dataset(devices: list[MeasDevice], npz_path: Path) -> dict:
    arrays, index = {}, []
    for d_i, dev in enumerate(devices):
        for t in TEMPS_K:
            for c_i, c in enumerate(fit_curves(dev, t)):
                p = f"d{d_i}_T{t}_c{c_i}"
                arrays[f"{p}_sweep"] = c.sweep
                arrays[f"{p}_id"] = c.id
                index.append("|".join([dev.tag, str(t), str(c_i), c.key, c.kind,
                                       f"{c.fixed:.17g}", str(len(c.sweep))]))
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    return {"index_columns": INDEX_COLUMNS, "index": index}


def device_row(dev: MeasDevice) -> dict:
    return {"tag": dev.tag, "dev_type": dev.dev_type, "W_um": dev.W_um, "L_um": dev.L_um,
            "sample": dev.sample, "device": dev.device, "sub_block": dev.sub_block,
            "group": dev.group, "temps_K": {str(k): v for k, v in dev.temps_K.items()},
            "files": {str(k): v for k, v in dev.files.items()}}


def load_dataset(npz_path: Path, manifest: dict, tags: Sequence[str] | None = None
                 ) -> list[MeasDevice]:
    """Rebuild ``MeasDevice`` objects; with ``tags``, only those devices' arrays
    are read."""
    keep = None if tags is None else set(tags)
    devices, order = {}, {}
    for i, row in enumerate(manifest["devices"]):
        order[row["tag"]] = i
        if keep is not None and row["tag"] not in keep:
            continue
        devices[row["tag"]] = MeasDevice(
            row["tag"], row["dev_type"], row["W_um"], row["L_um"], row["sample"], row["device"],
            row["sub_block"], {int(k): v for k, v in row["temps_K"].items()},
            {int(k): v for k, v in row["files"].items()})
    empty = np.zeros(0)
    with np.load(npz_path, allow_pickle=False) as data:
        for line in manifest["index"]:
            tag, t, c_i, key, kind, fixed, _ = line.split("|")
            if tag not in devices:
                continue
            p = f"d{order[tag]}_T{t}_c{c_i}"
            devices[tag].curves.setdefault(int(t), []).append(MeasCurve(
                f"camp{CAMPAIGN}_{kind}", kind, float(fixed), 0.0, np.asarray(data[f"{p}_sweep"]),
                np.asarray(data[f"{p}_id"]), empty, empty, empty, key))
    return sorted(devices.values(), key=lambda d: d.tag)


def curve_stats(c: MeasCurve) -> dict:
    """Range, exact zeros, smallest nonzero |id|, smallest step between distinct
    values, longest run of identical values at the curve's largest |id|."""
    a = np.abs(c.id)
    nz = a[a > 0]
    u = np.unique(a)
    steps = np.diff(u)
    top = a.max()
    run = 0
    for x in a[::-1]:
        if x == top:
            run += 1
        else:
            break
    return {"max_abs_id": float(top), "min_nonzero_abs_id": float(nz.min()) if len(nz) else 0.0,
            "n_exact_zero": int(np.sum(c.id == 0.0)),
            "min_step": float(steps[steps > 0].min()) if np.any(steps > 0) else None,
            "n_identical_at_max": int(np.sum(a == top)), "tail_run_at_max": run,
            "n_sign_wrong": int(np.sum(np.sign(c.id) == -np.sign(c.fixed if c.kind == "idvg"
                                                                   else c.sweep[-1])))}


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def make_split(devices: list[MeasDevice], seed: int) -> dict:
    """W/L pairs measured in both polarities ('shared') and nMOS-only pairs;
    TEST = every device of 1 shared pair (both polarities) and 1 nMOS-only
    pair.  ``numpy.random.default_rng(seed)``: one draw from the sorted shared
    pairs, then one from the sorted nMOS-only pairs."""
    rng = np.random.default_rng(seed)
    pairs = {}
    for d in devices:
        pairs.setdefault((d.W_um, d.L_um), set()).add(d.dev_type)
    shared = sorted(p for p, s in pairs.items() if s == {"nmos", "pmos"})
    n_only = sorted(p for p, s in pairs.items() if s == {"nmos"})
    if any(s == {"pmos"} for s in pairs.values()):
        raise RuntimeError("pMOS-only W/L pair")
    test_pairs = [shared[int(rng.integers(len(shared)))], n_only[int(rng.integers(len(n_only)))]]
    test = sorted(d.tag for d in devices if (d.W_um, d.L_um) in test_pairs)
    dev = sorted(d.tag for d in devices if d.tag not in test)
    return {"seed": seed, "shared_pairs": shared, "nmos_only_pairs": n_only,
            "TEST_pairs": test_pairs, "DEV": dev, "TEST": test,
            "n_dev": len(dev), "n_test": len(test)}


# ---------------------------------------------------------------------------
# BSIM4 card
# ---------------------------------------------------------------------------
def extract_card_source(path: Path = CARD_SOURCE) -> str:
    """The N1 / P1 ``.model`` blocks of ro_17_4.cir, verbatim (text read only)."""
    text = Path(path).read_text(errors="replace")
    i = text.index("* Short channel models from CMOS Circuit Design")
    j = text.index(".end", text.index(".model  P1"))
    return text[i:j].rstrip() + "\n"


@lru_cache(maxsize=None)
def card_params(dev_type: str) -> dict[str, str]:
    """``{name: value text}`` of the N1 (nmos) or P1 (pmos) block of CARD_FILE,
    in card order (``level`` and ``version`` included)."""
    text = CARD_FILE.read_text()
    name = {"nmos": "N1", "pmos": "P1"}[dev_type]
    m = re.search(rf"^\.model\s+{name}\s+{dev_type}\b(.*?)(?=^\*|^\.model|\Z)", text,
                  re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if not m:
        raise RuntimeError(f"{name} not found in {CARD_FILE}")
    body = m.group(1).replace("\n+", " ")
    out = {}
    for k, v in re.findall(r"([A-Za-z_][\w]*)\s*=\s*([^\s=]+)", body):
        out[k.lower()] = v
    return out


@lru_cache(maxsize=None)
def model_block(dev_type: str) -> str:
    """Normalized card: one ``+ key = value`` per line, model ``CARD_MODEL``,
    ``TC_ZERO`` parameters set to 0."""
    p = dict(card_params(dev_type))
    for k in TC_ZERO:
        if k not in p:
            raise KeyError(f"{k} not in the card")
        p[k] = "0"
    head = f".model {CARD_MODEL} {dev_type} level = {p.pop('level')} version = {p.pop('version')}"
    return head + "\n" + "\n".join(f"+ {k} = {v}" for k, v in p.items()) + "\n"


def card_value(dev_type: str, key: str, card: str | None = None) -> float:
    card = card if card is not None else model_block(dev_type)
    return float(_card_value_text(card, key))


def stock_params(dev: MeasDevice, names: Sequence[str]) -> dict[str, float]:
    block = model_block(dev.dev_type)
    return {k: card_value(dev.dev_type, k, block) for k in names}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _header(dev: MeasDevice, card_path: Path, temp_K: float, rs_ext_ohm: float) -> list[str]:
    drain, source = ("ndi", "nsi") if rs_ext_ohm > 0 else ("nd", "0")
    lines = ["* I6-1 TU Delft 40-nm measured-grid DC sweeps\n",
             f".options temp={temp_K - 273.15:.6g}\n",
             f'.include "{card_path.as_posix()}"\n',
             f"M1 {drain} ngt {source} nb {CARD_MODEL} w={dev.W_um:.6g}u l={dev.L_um:.6g}u nf=1\n"]
    if rs_ext_ohm > 0:
        lines += [f"RDEXT nd ndi {rs_ext_ohm:.10g}\n", f"RSEXT nsi 0 {rs_ext_ohm:.10g}\n"]
    lines += ["VG ngt 0 DC 0\n", "VD nd 0 DC 0\n", "VB nb 0 DC 0\n", ".control\n",
              "set filetype=ascii\n"]
    return lines


def simulate(dev: MeasDevice, card: str, curves: Sequence[MeasCurve], temp_K: float,
             params: dict[str, float] | None = None, rs_ext_ohm: float = 0.0,
             timeout_s: float = TIMEOUT_S) -> list[np.ndarray]:
    """One ngspice-41 process with every sweep of ``curves`` on ``card``
    patched with ``params``; current into the drain terminal (-i(VD))."""
    with tempfile.TemporaryDirectory(prefix="i6_1_") as td_s:
        td = Path(td_s)
        card_path = td / "card.mod"
        card_path.write_text(patch_card(card, params))
        deck = td / "deck.sp"
        deck.write_text("".join(_header(dev, card_path, temp_K, rs_ext_ohm)
                                + sweep_lines(curves, td, "raw_")
                                + ["quit\n", ".endc\n", ".end\n"]))
        try:
            subprocess.run([str(NGSPICE), "-b", str(deck)], cwd=td, capture_output=True,
                           text=True, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            global N_TIMEOUT
            N_TIMEOUT += 1
            return [np.full(len(c.sweep), np.nan) for c in curves]
        return read_sweeps(td, "raw_", curves)


# ---------------------------------------------------------------------------
# Scoring and records
# ---------------------------------------------------------------------------
def score(curves: Sequence[MeasCurve], sims: Sequence[np.ndarray]) -> dict:
    """Strict paper RRMS with the frozen measured inclusion (NaN if an
    included curve is non-finite)."""
    per, failed = {}, []
    for c, s in zip(curves, sims):
        inc = inclusion(c)
        if not inc["included"]:
            continue
        sim = np.asarray(s, dtype=np.float64)[inc["start"]:]
        tgt = clean_current(c.id)[inc["start"]:]
        if not np.all(np.isfinite(sim)):
            failed.append(c.key)
            continue
        per[c.key] = math.sqrt(float(np.mean((sim - tgt) ** 2))) / inc["den"]
    rrms = float(np.mean(list(per.values()))) if per and not failed else float("nan")
    return {"rrms": rrms, "n_curves": len(per) + len(failed), "failed": failed, "per_curve": per}


def to_record(dev: MeasDevice, temp_k: int, sims: Sequence[np.ndarray],
              params: dict | None = None) -> DeviceRec:
    """E1 ``DeviceRec`` (source ``tudelft_meas``; vdd = VDD)."""
    rec = DeviceRec(dev.tag, dev.dev_type, dev.L_um, dev.W_um, VDD, "tudelft_meas",
                    meta={"group": dev.group, "T_K": temp_k, "temp_stage_K": dev.temps_K[temp_k],
                          "params": dict(params or {})})
    for c, s in zip(fit_curves(dev, temp_k), sims):
        inc = inclusion(c)
        rec.curves.append(CurveRec(c.key, c.kind, np.asarray(c.vg, float), np.asarray(c.vd, float),
                                   np.asarray(s, float), np.asarray(c.id, float), inc["start"],
                                   inc["den"], inc["included"]))
    return rec


# ---------------------------------------------------------------------------
# Fitted parameter set and box
# ---------------------------------------------------------------------------
# The 43-parameter sky130 set (out/cold_loop/full_direct/round_3/spec.json,
# selected_parameters).
BASE_SET = ("toxe", "toxm", "xj", "ndep", "nsd", "dwg", "rsh", "vth0", "k1", "k2", "k3", "dvt0",
            "dvt1", "dvt1w", "w0", "k3b", "vfb", "lpe0", "lpeb", "vsat", "ua", "ub", "rdsw",
            "prwg", "wr", "u0", "a0", "a2", "ags", "b0", "b1", "voff", "nfactor", "cit",
            "cdscd", "eta0", "dsub", "pclm", "pdiblc2", "drout", "delta", "alpha0", "beta0")
BASE_OFFSET = {"vth0": 0.5, "vfb": 0.5, "voff": 0.3}            # linear +/- (V)
BASE_FACTOR = 10.0                                               # log |v|/10 .. 10|v|
BASE_ZERO_RANGE = {"b0": (0.0, 1e-7), "b1": (0.0, 1e-7),        # i4_1 BSIM4_ZERO_RANGE
                   "cit": (0.0, 2.5e-3), "cdscd": (0.0, 2.5e-3)}  # F/m^2
ZERO_TOL = 1e-19                                                 # |v| <= ZERO_TOL counts as 0 (the card writes 1e-20 for 0)


def base_box(dev: MeasDevice) -> ParameterBox:
    """Starting-fit and refit box around the stock card value v (43-parameter
    sky130 set): VTH0 and VFB linear v +/- 0.5 V, VOFF linear v +/- 0.3 V;
    any other parameter with |v| > 1e-19 log-uniform |v|/10 .. 10|v| (sign
    of v kept); |v| <= 1e-19: linear range ``BASE_ZERO_RANGE`` (B0, B1, CIT,
    CDSCD) if listed, else inactive (fixed at v).  The box is wide because
    the stock card is a generic 50-nm card, not a card of the measured
    process."""
    stock = stock_params(dev, BASE_SET)
    bounds, logs = {}, []
    for n in BASE_SET:
        v = float(stock[n])
        if n in BASE_OFFSET:
            bounds[n] = [v - BASE_OFFSET[n], v + BASE_OFFSET[n]]
            logs.append(False)
        elif abs(v) <= ZERO_TOL:
            bounds[n] = list(BASE_ZERO_RANGE.get(n, (v, v)))
            if not bounds[n][0] <= v <= bounds[n][1]:
                bounds[n] = [min(bounds[n][0], v), max(bounds[n][1], v)]
            logs.append(False)
        else:
            a, b = v / BASE_FACTOR, v * BASE_FACTOR
            bounds[n] = [min(a, b), max(a, b)]
            logs.append(True)
    return ParameterBox(names=BASE_SET, published=stock, bounds=bounds, log_scale=tuple(logs))


class MeasProblem:
    """``ngspice_fit.polish`` interface (as ``i5_1_ihp_real.MeasProblem``)."""

    def __init__(self, dev: MeasDevice, temp_k: int, box: ParameterBox, card: str | None = None,
                 rs_ext_ohm: float = 0.0):
        self.dev = self.device = dev
        self.tag = dev.tag
        self.temp_k = temp_k
        self.temp_stage_K = dev.temps_K[temp_k]
        self.box = box
        self.card = card if card is not None else model_block(dev.dev_type)
        self.rs_ext_ohm = rs_ext_ohm
        self.meas = self.curves = fit_curves(dev, temp_k)
        self.layout = []
        for index, c in enumerate(self.meas):
            inc = inclusion(c)
            if inc["included"]:
                self.layout.append({"index": index, "start": inc["start"], "tag": c.key,
                                    "target": clean_current(c.id)[inc["start"]:],
                                    "denominator": inc["den"]})
        if not self.layout:
            raise RuntimeError(f"{dev.tag} {temp_k} K: no included curves")
        self.n_curves = len(self.layout)
        self.n_sim = 0
        self.bin_index = 0

    def params(self, local) -> dict[str, float]:
        return self.box.local_to_params(np.asarray(local, dtype=np.float64))

    def simulate_params(self, params: dict, rs_ext_ohm: float | None = None) -> list[np.ndarray]:
        self.n_sim += 1
        return simulate(self.dev, self.card, self.meas, self.temp_stage_K, params,
                        self.rs_ext_ohm if rs_ext_ohm is None else rs_ext_ohm)

    def simulate(self, local) -> list[np.ndarray]:
        return self.simulate_params(self.params(local))

    def curve_rrms(self, sims) -> np.ndarray:
        return pdk2_metric.curve_rrms(sims, self.layout)

    def residual_vector(self, sims, weights) -> np.ndarray:
        pieces = []
        scale = 1.0 / math.sqrt(self.n_curves)
        for row, w in zip(self.layout, weights):
            sim = np.asarray(sims[row["index"]], dtype=np.float64)[row["start"]:]
            n = len(row["target"])
            if not np.all(np.isfinite(sim)):
                pieces.append(np.full(n, pdk2_metric.PENALTY * w * scale / math.sqrt(n)))
                continue
            pieces.append((sim - row["target"]) * w * scale / (row["denominator"] * math.sqrt(n)))
        return np.concatenate(pieces)

    def objective(self, local) -> float:
        return float(np.mean(self.curve_rrms(self.simulate(local))))


def corrected_problem(prob: MeasProblem, corr):
    """``i2_1_joint.CorrectedProblem`` with the position keys of the curves."""
    from .i2_1_joint import CorrectedProblem
    dev = prob.dev
    cp = CorrectedProblem(prob, corr, dev.dev_type, dev.L_um, dev.W_um, VDD, "tudelft_meas")
    cp._keys = [c.key for c in prob.curves]
    return cp


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
SENS_SUMMARY = ROOT / "out" / "pdk_parameter_sensitivity" / "summary.json"


# DVT0W is excluded (prereg amendment 2026-10-20): with DVT1W = 0 in the card
# (and inactive in the base box) the BSIM4 narrow-width short-channel term
# divides by cosh(0) - 1 = 0, so any DVT0W > 0 gives no DC solution.
B1_EXCLUDE = ("dvt0w",)


def b1_universe() -> list[str]:
    """The 115-name sky130 BSIM4 sensitivity universe minus the base set,
    ``TC_ZERO`` and ``B1_EXCLUDE``, restricted to names the card defines."""
    names = list(json.loads(SENS_SUMMARY.read_text())["parameters"])
    defined = set(card_params("nmos")) & set(card_params("pmos"))
    return [n for n in names if n not in BASE_SET and n not in TC_ZERO and n not in B1_EXCLUDE
            and n in defined]


def b1_extras(dev: MeasDevice, start: dict, names: Sequence[str] | None = None) -> dict:
    """I4-1 B1 box (``b1_extra_box`` with ``BSIM4_ZERO_RANGE``) at the card values."""
    from .i4_1_baselines import BSIM4_ZERO_RANGE, b1_extra_box
    universe = b1_universe() if names is None else [n for n in b1_universe() if n in set(names)]
    block = model_block(dev.dev_type)
    values = {n: card_value(dev.dev_type, n, block) for n in universe}
    return b1_extra_box(universe, values, BSIM4_ZERO_RANGE, rdsw=start.get("rdsw"))


def rext_max(dev: MeasDevice, start: dict) -> float:
    """B2 resistor upper bound: 4 * (RDSW_start / 2) / W_um (ohm, per side; WR = 1)."""
    from .i4_1_baselines import B2_REXT_FACTOR
    return B2_REXT_FACTOR * 0.5 * float(start["rdsw"]) / dev.W_um


# ---------------------------------------------------------------------------
# Aggregates and I/O
# ---------------------------------------------------------------------------
def aggregate(values: dict) -> dict:
    """all-device mean (NaN if any failed), combined (mean of nMOS / pMOS
    family means), median, failed count."""
    v = np.array(list(values.values()), dtype=float)
    finite = v[np.isfinite(v)]
    fam = {f: [x for t, x in values.items() if f"_{f}_" in t] for f in ("nmos", "pmos")}
    out = {"n": int(len(v)), "n_failed": int(np.sum(~np.isfinite(v))),
           "all_device_mean": float(np.mean(v)) if len(v) and np.all(np.isfinite(v)) else float("nan"),
           "valid_only_mean": float(np.mean(finite)) if len(finite) else float("nan"),
           "median": float(np.median(finite)) if len(finite) else float("nan")}
    means = [float(np.mean(fam[f])) for f in ("nmos", "pmos") if fam[f]]
    out["combined"] = float(np.mean(means)) if len(means) == 2 else float("nan")
    return out


def group_of(tag: str) -> str:
    return tag.rsplit("_d", 1)[0]


def json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=json_default) + "\n")
