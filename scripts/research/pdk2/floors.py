#!/usr/bin/env python3
"""Per-tier refit floors: ``ngspice_fit.polish`` started at the truth card's
fitted-parameter projection (``<tag>__local_<set>``), on a few devices.

For every (tier, device, fitted set): the paper RRMS of the projection
(``start``), after polish (``floor``), and of the noise alone (the truth's
own clean curves scored against its observed curves, ``noise``).  Truth
sets are generated fresh (1 set) under ``--out``; one ngspice process at a
time.

  .venv/bin/python -I scripts/research/pdk2/floors.py --pdk gf180 \
      --devices gf180_nmos_L0p3_W0p3,gf180_pmos_L1_W5 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_truth  # noqa: E402
from cryoml.ngspice_fit import polish  # noqa: E402
from cryoml.pdk2 import boxes  # noqa: E402
from cryoml.pdk2.problem import Pdk2Problem, final_score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdk", choices=("gf180", "ihp"), required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--tiers", default="a,b,c")
    parser.add_argument("--fit-sets", default="seven,wide")
    parser.add_argument("--nfev", type=int, default=300)
    parser.add_argument("--irls", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--truth-args", default="",
                        help="extra make_truth.py arguments (quoted)")
    args = parser.parse_args()
    out = Path(args.out)
    truth_root = out / "truth"
    tiers = args.tiers.split(",")
    for tier in tiers:
        make_truth.main(["--pdk", args.pdk, "--tier", tier, "--sets", "1",
                         "--devices", args.devices, "--out-root", str(truth_root),
                         *args.truth_args.split()])
    rows = []
    for tier in tiers:
        with np.load(truth_root / tier / "set_0.npz") as data:
            arrays = {k: data[k] for k in data.files}
        for tag in args.devices.split(","):
            clean = np.split(arrays[f"{tag}__clean_ids"], 11)
            for set_name in args.fit_sets.split(","):
                if tier == "a" and set_name == "wide":
                    continue          # well-specified: the floor is 0 either way
                box = boxes.box_for(set_name, tag)
                problem = Pdk2Problem.from_truth(tag, box, tier, 0, truth_root)
                start_local = arrays[f"{tag}__local_{set_name}"]
                t0 = time.time()
                start = problem.objective(start_local)
                result = polish(problem, start_local, max_nfev=args.nfev,
                                irls_rounds=args.irls)
                floor = final_score(problem, result["local"])["rrms"]
                noise = float(np.mean(problem.curve_rrms(clean)))
                row = {"tier": tier, "device": tag, "set": set_name,
                       "start": round(start, 5), "floor": round(floor, 5),
                       "noise": round(noise, 5), "n_sim": problem.n_sim,
                       "runtime_s": round(time.time() - t0, 1)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    summary = {}
    for tier in tiers:
        for set_name in args.fit_sets.split(","):
            sel = [r for r in rows if r["tier"] == tier and r["set"] == set_name]
            if sel:
                summary[f"{tier}/{set_name}"] = {
                    "mean_floor": round(float(np.mean([r["floor"] for r in sel])), 5),
                    "mean_noise": round(float(np.mean([r["noise"] for r in sel])), 5),
                    "mean_start": round(float(np.mean([r["start"] for r in sel])), 5)}
    report = {"pdk": args.pdk, "devices": args.devices.split(","),
              "nfev": args.nfev, "irls": args.irls,
              "truth_args": args.truth_args, "rows": rows, "summary": summary}
    (out / f"floors_{args.pdk}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
