#!/usr/bin/env python3
"""E1: best-fit compact-model curves for GF180 / IHP tier-c truth targets.

Per target: ``ngspice_fit.polish`` from the truth's ``<tag>__local_<fit>``
projection (default wide set, max_nfev 300, irls 3), then one fresh
simulation of the best card.  Writes ``<out>/<pdk>_<tier>_set<k>.{npz,json}``
(``DeviceRec`` records; ``meta`` holds the fitted card, set index, n_sim and
the misspecification draws of the truth).  One simulator process.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e1_collect_pdk2.py \
      --pdk gf180 --sets 0
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

from cryoml.explore.e1_correction import (CurveRec, DeviceRec, save_devices,  # noqa: E402
                                          score_device)
from cryoml.ngspice_fit import polish  # noqa: E402
from cryoml.pdk2 import boxes, metric  # noqa: E402
from cryoml.pdk2.problem import TRUTH_ROOTS, Pdk2Problem, load_truth_meta  # noqa: E402

OUT = ROOT / "out/research/explore/E1"


def collect(pdk: str, tier: str, set_index: int, fit: str, nfev: int, irls: int,
            devices: list[str] | None) -> list[DeviceRec]:
    root = TRUTH_ROOTS[pdk]
    sim = boxes.backend(pdk)
    meta = load_truth_meta(tier, root)["sets"][str(set_index)]
    recs = []
    with np.load(root / tier / f"set_{set_index}.npz") as data:
        arrays = {k: data[k] for k in data.files}
    tags = [str(t) for t in arrays["tags"]] if "tags" in arrays else sorted(meta)
    for tag in tags:
        if devices and tag not in devices:
            continue
        t0 = time.time()
        box = boxes.box_for(fit, tag)
        problem = Pdk2Problem.from_truth(tag, box, tier, set_index, root)
        start = arrays[f"{tag}__local_{fit}"]
        result = polish(problem, start, max_nfev=nfev, irls_rounds=irls)
        sims = problem.simulate(result["local"])
        dev = problem.device
        layout = {row["index"]: row for row in problem.layout}
        rec = DeviceRec(f"{tag}#s{set_index}", dev.dev_type, float(dev.L_um),
                        float(dev.W_um), float(sim.VDD), pdk,
                        meta={"truth_tag": tag, "set": set_index, "tier": tier,
                              "fit": fit, "n_sim": problem.n_sim,
                              "params": problem.params(result["local"]),
                              "truth": {k: v for k, v in meta[tag].items()
                                        if k in ("rs_ext_ohm_per_side", "rext_k", "dT_max",
                                                 "curve_gains", "misspec_params")}})
        for i, curve in enumerate(problem.curves):
            canon = curve.meta.get("canonical_fixed", curve.fixed)
            key = f"{curve.kind}@{abs(float(canon)):g}"
            start_i, _, den = metric.curve_trim(curve.kind, curve.Id)
            rec.curves.append(CurveRec(key, curve.kind, np.asarray(curve.Vg, float),
                                       np.asarray(curve.Vd, float),
                                       np.asarray(sims[i], float),
                                       np.asarray(curve.Id, float), start_i, den,
                                       i in layout))
        mine = score_device(rec, sims)
        ref = float(np.mean(problem.curve_rrms(sims)))
        if np.isfinite(mine["rrms"]) and not np.isclose(mine["rrms"], ref, rtol=1e-9):
            raise RuntimeError(f"{tag}: {mine['rrms']} != {ref}")
        rec.meta["rrms_c0"] = mine["rrms"]
        print(json.dumps({"tag": rec.tag, "start_obj": result["history"][0],
                          "rrms_c0": mine["rrms"], "n_sim": problem.n_sim,
                          "s": round(time.time() - t0, 1)}), flush=True)
        recs.append(rec)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdk", choices=("gf180", "ihp"), required=True)
    ap.add_argument("--tier", default="c")
    ap.add_argument("--sets", default="0")
    ap.add_argument("--fit", default="wide")
    ap.add_argument("--nfev", type=int, default=300)
    ap.add_argument("--irls", type=int, default=3)
    ap.add_argument("--devices", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    devices = [d for d in args.devices.split(",") if d]
    for s in (int(x) for x in args.sets.split(",")):
        recs = collect(args.pdk, args.tier, s, args.fit, args.nfev, args.irls, devices)
        suffix = "_partial" if devices else ""
        save_devices(OUT / f"{args.pdk}_{args.tier}_set{s}{suffix}", recs)
        print(f"{args.pdk} tier {args.tier} set {s}: C0 mean "
              f"{np.nanmean([r.meta['rrms_c0'] for r in recs]):.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
