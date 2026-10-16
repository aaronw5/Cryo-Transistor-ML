#!/usr/bin/env python3
"""pdk2 smoke test (light: ~200 short ngspice runs).

1. extracted stock card == full GF180 library (native binning), 16 devices;
2. ``simulate_many`` (altermod) == ``simulate``, and the batched residual
   Jacobian == the sequential forward differences ``ngspice_fit.polish`` takes;
3. ``gen_bank.py`` with n=20 on one nFET and one pFET, both parameter sets:
   schema vs a cold-loop bank, ``canonical_point_order`` on
   ``fixeds_canonical``, ``pdk_gen_full_data.build_box`` on the written spec;
4. ``make_truth.py`` (1 set, the same 2 devices, both tiers);
5. ``ngspice_fit.polish`` (unchanged) on a ``Pdk2Problem`` (300 nfev).

  .venv/bin/python -I scripts/research/pdk2/smoke.py --out <scratch dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cryoml.pdk2 import boxes, gf180, metric  # noqa: E402
from cryoml.pdk2.problem import Pdk2Problem, final_score  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
COLD_BANK = ROOT / "data" / "processed" / "cold_loop" / "full_r1" / "nmos_L1_W3.npz"
TWO = "gf180_nmos_L0p3_W0p3,gf180_pmos_L1_W5"


def full_library(device) -> list[np.ndarray]:
    curves = gf180.device_curves(device.dev_type)
    with tempfile.TemporaryDirectory(prefix="pdk2_full_") as td_s:
        td = Path(td_s)
        instance = gf180._instance(device).replace(
            gf180.CARD_MODEL, gf180.FAMILY_MODEL[device.dev_type], 1)
        deck = td / "deck.sp"
        deck.write_text("".join(
            [f".options temp={gf180.TEMP_C:.6g}\n",
             f'.include "{gf180.DESIGN.as_posix()}"\n',
             f'.lib "{gf180.LIBRARY.as_posix()}" typical\n', instance,
             "VG ng 0 DC 0\n", "VD nd 0 DC 0\n", ".control\n",
             "set filetype=ascii\n"]
            + gf180._sweep_lines(curves, td, "raw_")
            + ["quit\n", ".endc\n", ".end\n"]))
        subprocess.run([gf180.NGSPICE_BIN, "-b", str(deck)], cwd=td,
                       capture_output=True, text=True, timeout=60, check=False)
        return gf180._read(td, "raw_", curves, device.dev_type)


def max_diff(a, b) -> float:
    return max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # 1. extraction
    t0 = time.perf_counter()
    diffs = {}
    for device in gf180.DEVICES:
        sims = gf180.simulate(device)
        assert all(np.all(np.isfinite(c)) for c in sims), device.tag
        diffs[device.tag] = max_diff(sims, full_library(device))
    report["extracted_vs_full_library_max_abs_A"] = max(diffs.values())
    report["sec_per_device_sim_incl_full_lib"] = round(
        (time.perf_counter() - t0) / 16, 4)

    # 2. altermod and batched Jacobian
    device = gf180.device_by_tag("gf180_pmos_L1_W5")
    box = boxes.box_for("wide", device.tag)
    rng = np.random.default_rng(1)
    params = [box.local_to_params(rng.uniform(-1, 1, box.n_parameters))
              for _ in range(4)]
    t0 = time.perf_counter()
    separate = [gf180.simulate(device, p) for p in params]
    t_sep = (time.perf_counter() - t0) / len(params)
    report["altermod_vs_separate_max_abs_A"] = max(
        max_diff(s, m) for s, m in zip(separate, gf180.simulate_many(device, params)))
    stock = gf180.simulate(device)
    problem = Pdk2Problem(device.tag, box, stock)
    local = rng.uniform(-0.5, 0.5, box.n_parameters)
    t0 = time.perf_counter()
    r0, jac, _ = problem.residual_jacobian(local, step=1e-3)
    t_jac = time.perf_counter() - t0
    weights = np.ones(problem.n_curves)
    seq = []
    for j in (0, 6, 20):          # vth0, u0, delta: polish's sequential FD
        shifted = local.copy()
        shifted[j] += 1e-3
        seq.append((problem.residual_vector(problem.simulate(shifted), weights)
                    - problem.residual_vector(problem.simulate(local), weights))
                   / 1e-3)
    report["batched_vs_sequential_jacobian_max_abs"] = float(max(
        np.max(np.abs(jac[:, j] - col)) for j, col in zip((0, 6, 20), seq)))
    report["timing_s"] = {"separate_card": round(t_sep, 4),
                          "jacobian_22_cards_one_process": round(t_jac, 3)}

    # 3. banks, n = 20 on two devices, both sets
    cold = np.load(COLD_BANK, allow_pickle=False)
    from cryoml.foundation import canonical_point_order
    import pdk_gen_full_data as pgfd
    for set_name in ("seven", "wide"):
        bank_dir = out / f"bank_{set_name}"
        t0 = time.perf_counter()
        proc = subprocess.run(
            [PY, "-I", str(ROOT / "scripts/research/pdk2/gen_bank.py"),
             "--set", set_name, "--devices", TWO, "--num-samples", "20",
             "--workers", "1", "--out-dir", str(bank_dir)],
            capture_output=True, text=True, check=True)
        wall = time.perf_counter() - t0
        spec_json = json.loads((bank_dir / "spec.json").read_text())
        rows = {}
        for tag in TWO.split(","):
            data = np.load(bank_dir / f"{tag}.npz", allow_pickle=False)
            order, desc = canonical_point_order(
                data["kinds"], data["fixeds_canonical"], data["slices"])
            box_pg = pgfd.build_box(spec_json, tag)
            rows[tag] = {
                "missing_cold_keys": sorted(set(cold.files) - set(data.files)),
                "extra_keys": sorted(set(data.files) - set(cold.files)),
                "dtype_shape_match": all(
                    data[k].dtype.kind == cold[k].dtype.kind
                    and data[k].ndim == cold[k].ndim for k in cold.files),
                "IDS": list(data["IDS"].shape), "n_ok": int(data["ok"].sum()),
                "row0_is_stock": bool(np.allclose(
                    data["THETA"][0], data["published"])),
                "canonical_order_identity": bool(
                    np.array_equal(order, np.arange(len(order)))),
                "build_box_matches": bool(
                    np.allclose(box_pg.lo, data["lo"])
                    and np.allclose(box_pg.hi, data["hi"])
                    and np.array_equal(box_pg.log, data["log_scale"])),
                "n_active": int(data["active"].sum()),
            }
        records = [json.loads(line) for line in proc.stdout.splitlines()
                   if line.startswith('{"device"')]
        report[f"bank_{set_name}"] = {
            "devices": rows, "wall_s": round(wall, 2),
            "sec_per_sample": round(sum(r["runtime_s"] for r in records)
                                    / (20 * len(records)), 4)}

    # 4. truth (1 set, 2 devices, both tiers)
    truth_root = out / "truth"
    subprocess.run([PY, "-I", str(ROOT / "scripts/research/pdk2/make_truth.py"),
                    "--sets", "1", "--devices", TWO, "--out-root",
                    str(truth_root)], capture_output=True, text=True, check=True)
    tier_rows = {}
    for tier in ("a", "b"):
        meta = json.loads((truth_root / tier / "truth.json").read_text())
        tier_rows[tier] = {}
        for tag in TWO.split(","):
            row = meta["sets"]["0"][tag]
            for set_name in ("seven", "wide"):
                box_t = boxes.box_for(set_name, tag)
                problem = Pdk2Problem.from_truth(tag, box_t, tier, 0, truth_root)
                with np.load(truth_root / tier / "set_0.npz") as data:
                    true_local = data[f"{tag}__local_{set_name}"]
                tier_rows[tier][f"{tag}/{set_name}"] = {
                    "n_curves": problem.n_curves,
                    "rrms_stock_card": round(problem.objective(
                        box_t.local_published), 5),
                    "rrms_true_fitted_params": round(problem.objective(true_local), 5),
                    "rs_ext_ohm_per_side": round(row["rs_ext_ohm_per_side"], 2)}
    report["truth"] = tier_rows

    # 5. polish (cryoml.ngspice_fit, unchanged) on a Pdk2Problem
    from cryoml.ngspice_fit import polish
    for tier, set_name in (("a", "seven"), ("b", "wide")):
        tag = "gf180_nmos_L0p3_W0p3"
        problem = Pdk2Problem.from_truth(tag, boxes.box_for(set_name, tag),
                                         tier, 0, truth_root)
        result = polish(problem, problem.box.local_published, max_nfev=300,
                        irls_rounds=3)
        report[f"polish_{tier}_{set_name}"] = {
            "history": [round(v, 5) for v in result["history"]],
            "objective": round(result["objective"], 5),
            "final_score": round(final_score(problem, result["local"])["rrms"], 5),
            "n_sim": result["n_sim"], "runtime_s": result["runtime_s"]}

    # 6. CMA-ES on ``objective`` (as ngspice_direct) and the trust-region
    #    Gauss-Newton control (cryoml.trust_region, unchanged), tier a
    import cma
    from cryoml.pdk2.problem import TrustRegionBackend
    from cryoml.trust_region import TrustRegionRun, TrustRegionSettings
    tag = "gf180_pmos_L1_W5"
    problem = Pdk2Problem.from_truth(tag, boxes.box_for("seven", tag), "a", 0,
                                     truth_root)
    es = cma.CMAEvolutionStrategy(np.zeros(7), 0.3, {
        "bounds": [-1, 1], "maxfevals": 60, "seed": 1, "verbose": -9})
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [problem.objective(np.clip(x, -1, 1)) for x in xs])
    report["cmaes_a_seven"] = {"start": round(problem.objective(np.zeros(7)), 5),
                               "best": round(float(es.result.fbest), 5),
                               "evals": int(es.result.evaluations)}
    problem = Pdk2Problem.from_truth(tag, boxes.box_for("seven", tag), "a", 0,
                                     truth_root)
    backend = TrustRegionBackend(problem)
    run = TrustRegionRun(np.zeros(7), backend, None, 1e-4 / 7 ** 0.5,
                         TrustRegionSettings())
    t0 = time.perf_counter()
    run.advance(120)
    summary = run.summary()
    report["trust_region_gn_a_seven"] = {
        "status": summary["status"], "calls": summary["calls"],
        "F_start": summary["F_start"], "F_final": summary["F_final"],
        "rrms_final": round(problem.objective(run.z_c), 6),
        "runtime_s": round(time.perf_counter() - t0, 2)}

    # 7. bank generation through the spawn pool (2 workers)
    t0 = time.perf_counter()
    subprocess.run(
        [PY, "-I", str(ROOT / "scripts/research/pdk2/gen_bank.py"), "--set",
         "seven", "--devices", TWO, "--num-samples", "20", "--workers", "2",
         "--out-dir", str(out / "bank_seven_pool")],
        capture_output=True, text=True, check=True)
    pooled = {tag: np.load(out / "bank_seven_pool" / f"{tag}.npz")["IDS"]
              for tag in TWO.split(",")}
    serial = {tag: np.load(out / "bank_seven" / f"{tag}.npz")["IDS"]
              for tag in TWO.split(",")}
    report["bank_pool_equals_serial"] = all(
        np.array_equal(pooled[t], serial[t]) for t in pooled)
    report["bank_pool_wall_s"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(report, indent=1))
    (out / "smoke_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
