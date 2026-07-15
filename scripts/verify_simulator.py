#!/usr/bin/env python3
"""Verify the local harness against the confirmed-setup repository.

For every Table-6 device, re-simulate the 11 metric curves (5 idVd + 6 idVg)
on the exact voltage grids of the committed sweep outputs in
``CryoPDK_Skywater130nm_ML/{nfet,pfet}_mod/<dev>/{vd,vg}_sweep`` and report
the worst absolute / peak-relative current difference. This is the gate that
proves our decks + corner cards + binary reproduce the upstream simulation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import (NEW_REPO_COMMIT, NEW_REPO_DIR, OUT_TABLES,  # noqa: E402
                           ensure_dirs)
from cryoml.data_io import Curve  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.spice_pdk import _NGSPICE_BIN, simulate_pdk  # noqa: E402

# Device -> upstream per-device directory (names taken verbatim from the repo).
UPSTREAM_DIR = {
    ("nmos", 0.15, 1.6): "l_0p15_w_1p6",
    ("nmos", 0.19, 7.0): "l_0p19_w_7p0",
    ("nmos", 0.25, 1.6): "l_0p25_w_1p6",
    ("nmos", 1.0, 1.6): "l_1p0_w_1p6",
    ("nmos", 1.0, 3.0): "l_1p0_w_3p0",
    ("nmos", 8.0, 1.6): "l_8p0_w_1p6",
    ("nmos", 20.0, 0.64): "l_20p0_w_0p64",
    ("nmos", 100.0, 100.0): "l_100p0_w_100p0",
    ("pmos", 0.35, 0.55): "l_0p35_w_0p55",
    ("pmos", 0.35, 1.6): "l_0p35_w_1p6",
    ("pmos", 0.35, 5.0): "l_0p35_w_5p0",
    ("pmos", 0.5, 0.42): "l_0p50_w_0p42",
    ("pmos", 0.5, 0.64): "l_0p50_w_0p64",
    ("pmos", 2.0, 5.0): "l_2p0_w_5p0",
    ("pmos", 4.0, 7.0): "l_4p0_w_7p0",
    ("pmos", 8.0, 0.84): "l_8p0_w_0p84",
    ("pmos", 8.0, 1.6): "l_8p0_w_1p6",
    ("pmos", 8.0, 5.0): "l_8p0_w_5p0",
}

VD_BIASES = (0.37, 0.74, 1.11, 1.48, 1.85)          # idVd files, named by |Vg|
VG_BIASES = (0.01, 0.37, 0.74, 1.11, 1.48, 1.85)    # idVg files, named by |Vd|


def _bias_name(v: float) -> str:
    return f"{v:.3f}".replace(".", "p")


def upstream_metric_curves(dev) -> tuple[list[Curve], list[np.ndarray], list[str]]:
    """Committed sweep files -> (curves on their V grids, reference Id, names).

    Committed files interleave (scale, vector) columns: col0 = swept voltage,
    col3 = drain current (signed; negative for pMOS), col1/col5 = the fixed
    bias for idVd/idVg respectively.
    """
    fam = "nfet_mod" if dev.dev_type == "nmos" else "pfet_mod"
    root = NEW_REPO_DIR / fam / UPSTREAM_DIR[(dev.dev_type, dev.L_um, dev.W_um)]
    sign = -1.0 if dev.dev_type == "pmos" else 1.0
    curves, refs, names = [], [], []
    for vg in VD_BIASES:
        path = root / "vd_sweep" / f"idVd_vg_{_bias_name(vg)}.txt"
        arr = np.loadtxt(path)
        vd, fixed_vg, cur = arr[:, 0], float(arr[0, 1]), arr[:, 3]
        curves.append(Curve("idvd", fixed_vg, np.full_like(vd, fixed_vg), vd,
                            sign * cur, path=str(path), source="upstream_sim"))
        refs.append(sign * cur)
        names.append(path.name)
    for vd in VG_BIASES:
        path = root / "vg_sweep" / f"idVg_vd_{_bias_name(vd)}.txt"
        arr = np.loadtxt(path)
        vgs, fixed_vd, cur = arr[:, 0], float(arr[0, 5]), arr[:, 3]
        curves.append(Curve("idvg", fixed_vd, vgs, np.full_like(vgs, fixed_vd),
                            sign * cur, path=str(path), source="upstream_sim"))
        refs.append(sign * cur)
        names.append(path.name)
    return curves, refs, names


def main() -> int:
    ensure_dirs()
    upstream_commit = (NEW_REPO_DIR / ".upstream_commit").read_text().strip() \
        if (NEW_REPO_DIR / ".upstream_commit").exists() else None
    if upstream_commit != NEW_REPO_COMMIT:
        raise RuntimeError(
            f"confirmed upstream marker is {upstream_commit!r}; expected "
            f"{NEW_REPO_COMMIT}"
        )
    per_device = {}
    worst_abs = {"abs_A": 0.0, "peak_rel": 0.0, "where": None}
    worst_peak_rel = {"abs_A": 0.0, "peak_rel": 0.0, "where": None}
    for dev in PAPER_DEVICES:
        curves, refs, names = upstream_metric_curves(dev)
        sims = simulate_pdk(dev.dev_type, dev.L_um, dev.W_um, curves)
        rows = {}
        for name, sim, ref in zip(names, sims, refs):
            max_abs = float(np.max(np.abs(sim - ref)))
            peak = float(max(np.max(np.abs(ref)), 1e-12))
            rows[name] = {"max_abs_A": max_abs, "peak_rel": max_abs / peak}
            where = f"{dev.dev_type}_{dev.L_um:g}_{dev.W_um:g}:{name}"
            peak_rel = max_abs / peak
            if max_abs > worst_abs["abs_A"]:
                worst_abs = {"abs_A": max_abs, "peak_rel": peak_rel,
                             "where": where}
            if peak_rel > worst_peak_rel["peak_rel"]:
                worst_peak_rel = {"abs_A": max_abs, "peak_rel": peak_rel,
                                  "where": where}
        tag = f"{dev.dev_type}_L{dev.L_um:g}_W{dev.W_um:g}"
        per_device[tag] = {
            "max_abs_A": max(r["max_abs_A"] for r in rows.values()),
            "max_peak_rel": max(r["peak_rel"] for r in rows.values()),
            "curves": rows,
        }
        print(f"{tag:24s} max|dI| = {per_device[tag]['max_abs_A']:.3e} A   "
              f"peak-rel = {per_device[tag]['max_peak_rel']:.3e}")
    version = subprocess.run([_NGSPICE_BIN, "--version"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[1].strip()
    result = {
        "ngspice_bin": _NGSPICE_BIN,
        "ngspice_version": version,
        "upstream_repo_commit": upstream_commit,
        "n_devices": len(per_device),
        "worst_absolute": worst_abs,
        "worst_peak_relative_with_1pA_floor": worst_peak_rel,
        "per_device": per_device,
    }
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(OUT_TABLES / "simulator_verification.json", "w"),
              indent=2)
    print(f"\nWORST ABSOLUTE: {worst_abs['where']}  "
          f"|dI|={worst_abs['abs_A']:.3e} A  "
          f"peak-rel={worst_abs['peak_rel']:.3e}")
    print(f"WORST PEAK-RELATIVE (1 pA floor): {worst_peak_rel['where']}  "
          f"|dI|={worst_peak_rel['abs_A']:.3e} A  "
          f"peak-rel={worst_peak_rel['peak_rel']:.3e}")
    print(f"ngspice: {version} ({_NGSPICE_BIN})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
