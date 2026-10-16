#!/usr/bin/env python3
"""IHP SG13G2 (PSP103 via OSDI, ngspice-46) smoke test, one ngspice process
at a time:

1. standalone top-level PSP card == the PDK subcircuit + corner library,
   all 16 devices; run-to-run determinism;
2. ``simulate_many`` (altermod) == ``simulate`` and its speed-up;
3. per-curve ``option temp`` == a global temperature run (IHP and GF180);
4. ``ihp_gen_bank.py`` n=20 on 2 devices, both sets, schema vs a GF180 bank;
5. ``ngspice_fit.polish`` (unchanged) on an IHP ``Pdk2Problem``.

  .venv/bin/python -I scripts/research/pdk2/ihp_smoke.py --out <scratch dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.pdk2 import boxes, gf180, ihp  # noqa: E402
from cryoml.pdk2.problem import Pdk2Problem, final_score  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
TWO = "ihp_nmos_L0p13_W0p2,ihp_pmos_L0p5_W2"


def max_diff(a, b) -> float:
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--gf180-bank", default=None,
                        help="a GF180 bank npz for the schema comparison")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"ngspice46": ihp.ngspice46_bin()}

    # 1. extraction and determinism
    diffs, times, ion = {}, [], {}
    for device in ihp.DEVICES:
        t0 = time.perf_counter()
        sims = ihp.simulate(device)
        times.append(time.perf_counter() - t0)
        assert all(np.all(np.isfinite(c)) for c in sims), device.tag
        diffs[device.tag] = max_diff(sims, ihp.simulate_pdk_subckt(device))
        ion[device.tag] = float(sims[4][-1])
    again = ihp.simulate(ihp.DEVICES[0])
    report["standalone_vs_pdk_subckt_max_abs_A"] = max(diffs.values())
    report["repeat_bitwise_identical"] = max_diff(
        again, ihp.simulate(ihp.DEVICES[0])) == 0.0
    report["sec_per_separate_sim_median"] = round(float(np.median(times)), 4)
    report["Ion_at_VDD_A"] = {k: float(f"{v:.4g}") for k, v in ion.items()}

    # 2. altermod
    device = ihp.device_by_tag("ihp_pmos_L0p5_W2")
    box = boxes.box_for("wide", device.tag)
    rng = np.random.default_rng(5)
    cards = [box.local_to_params(rng.uniform(-1, 1, box.n_parameters))
             for _ in range(200)]
    t0 = time.perf_counter()
    separate = [ihp.simulate(device, p) for p in cards[:5]]
    t_sep = (time.perf_counter() - t0) / 5
    timing = {"separate_ms_per_card": round(1e3 * t_sep, 1)}
    for n in (50, 200):
        t0 = time.perf_counter()
        many = ihp.simulate_many(device, cards[:n], chunk=n)
        dt = (time.perf_counter() - t0) / n
        timing[f"altermod_{n}_ms_per_card"] = round(1e3 * dt, 1)
        timing[f"speedup_{n}"] = round(t_sep / dt, 2)
    report["altermod_vs_separate_max_abs_A"] = max(
        max_diff(s, m) for s, m in zip(separate, many))
    report["altermod_all_finite"] = all(np.all(np.isfinite(c))
                                        for s in many for c in s)
    report["timing"] = timing

    # 3. per-curve temperature (``option temp`` in the control block)
    for name, sim, dev in (("ihp", ihp, ihp.DEVICES[0]),
                           ("gf180", gf180, gf180.DEVICES[0])):
        hot = sim.simulate(dev, temp_C=57.0)
        per_curve = sim.simulate(dev, curve_temps_C=[57.0] * 11)
        mixed = sim.simulate(dev, curve_temps_C=[27.0] * 5 + [57.0] * 6)
        cold = sim.simulate(dev)
        report[f"curve_temps_{name}"] = {
            "uniform_vs_global_max_abs_A": max_diff(hot, per_curve),
            "mixed_matches_per_curve": bool(
                max_diff(mixed[:5], cold[:5]) == 0.0
                and max_diff(mixed[5:], hot[5:]) == 0.0),
            "hot_changes_current": max_diff(hot, cold) > 0}

    # 4. banks
    ref = np.load(args.gf180_bank) if args.gf180_bank else None
    for set_name in ("seven", "wide"):
        bank_dir = out / f"ihp_bank_{set_name}"
        t0 = time.perf_counter()
        subprocess.run([PY, "-I", str(ROOT / "scripts/research/pdk2/ihp_gen_bank.py"),
                        "--set", set_name, "--devices", TWO, "--num-samples", "20",
                        "--workers", "1", "--out-dir", str(bank_dir)],
                       capture_output=True, text=True, check=True)
        rows = {}
        for tag in TWO.split(","):
            data = np.load(bank_dir / f"{tag}.npz")
            rows[tag] = {"IDS": list(data["IDS"].shape),
                         "n_ok": int(data["ok"].sum()),
                         "n_active": int(data["active"].sum()),
                         "row0_is_stock": bool(np.allclose(data["THETA"][0],
                                                           data["published"]))}
            if ref is not None:
                rows[tag]["same_keys_as_gf180"] = set(data.files) == set(ref.files)
                rows[tag]["same_dtype_kinds"] = all(
                    data[k].dtype.kind == ref[k].dtype.kind
                    and data[k].ndim == ref[k].ndim for k in ref.files)
        report[f"bank_{set_name}"] = {"devices": rows,
                                      "wall_s": round(time.perf_counter() - t0, 2)}

    # 5. polish on an IHP problem (target: a +5% vfbo/uo card, seven box)
    tag = "ihp_nmos_L0p13_W0p2"
    box7 = boxes.box_for("seven", tag)
    dev = ihp.device_by_tag(tag)
    stock = ihp.stock_params(dev, boxes.IHP_SEVEN)
    target = ihp.simulate(dev, {"vfbo": stock["vfbo"] * 1.05,
                                "uo": stock["uo"] * 0.95})
    problem = Pdk2Problem(tag, box7, target)
    from cryoml.ngspice_fit import polish
    result = polish(problem, box7.local_published, max_nfev=150, irls_rounds=2)
    report["polish_ihp_seven"] = {
        "history": [round(v, 6) for v in result["history"]],
        "final_score": round(final_score(problem, result["local"])["rrms"], 6),
        "n_sim": result["n_sim"], "runtime_s": result["runtime_s"]}
    print(json.dumps(report, indent=1))
    (out / "ihp_smoke_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
