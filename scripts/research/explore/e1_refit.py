#!/usr/bin/env python3
"""E1 stage 2: re-polish each held-out REAL_DEV card with a correction in the loop.

For every REAL_DEV device and every arm in ``--arms`` the card from
``out/research/polished/direct43_r3_fd`` is the start of
``ngspice_fit.polish`` (max_nfev 400, irls 2, round-3 43-parameter box)
against the measured curves, where every NGSpice simulation is passed
through that device's leave-one-device-out correction (``fold_corrections.json``
of the stage-1 run).  ``C0`` is the same polish without correction (control).
The best card is re-simulated, corrected, and scored with the strict paper
RRMS (frozen inclusion).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e1_refit.py \
      --stage1 out/research/explore/E1/real_dev_lodo --arms C0,T1,M --workers 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SPEC = ROOT / "out/cold_loop/full_direct/round_3/spec.json"
CARDS = ROOT / "out/research/polished/direct43_r3_fd/real/summary.json"
SPLIT = ROOT / "out/research/explore/real_split.json"
VDD = 1.85


def job(task):
    import torch
    torch.set_num_threads(1)
    from cold_loop import box_for
    from e1_collect_real import collect
    from cryoml.explore.e1_correction import (Correction, CurveRec, DeviceRec,
                                              score_device)
    from cryoml.ngspice_fit import DeviceProblem, polish

    tag, arm, corr_row, options = task
    spec = json.loads(SPEC.read_text())
    card = json.loads(CARDS.read_text())["devices"][tag]["params"]
    corr = Correction("C0") if corr_row is None else Correction.from_json(corr_row)

    class CorrectedProblem(DeviceProblem):
        def simulate(self, local):
            sims = super().simulate(local)
            fam = "pmos" if self.device.dev_type == "pmos" else "nmos"
            rec = DeviceRec(self.tag, fam, float(self.device.L_um),
                            float(self.device.W_um), VDD, "sky130")
            for c, s in zip(self.curves, sims):
                rec.curves.append(CurveRec(f"{c.kind}@{abs(float(c.fixed)):g}", c.kind,
                                           np.asarray(c.Vg, float), np.asarray(c.Vd, float),
                                           np.asarray(s, float), np.asarray(c.Id, float),
                                           0, 0.0, False))
            return corr.apply(rec)

    problem = CorrectedProblem(tag, box_for(spec, tag))
    start = problem.box.params_to_local(
        {**spec["published_by_device"][tag],
         **{k: v for k, v in card.items() if k in problem.box.names}})
    t0 = time.time()
    result = polish(problem, start, max_nfev=options["nfev"],
                    irls_rounds=options["irls"], diff_step=options["diff_step"])
    params = problem.box.local_to_params(result["local"])
    rec_start = collect(tag, {**card})
    rec_end = collect(tag, params)
    before = score_device(rec_start, corr.apply(rec_start))
    after = score_device(rec_end, corr.apply(rec_end))
    return tag, arm, {"rrms_start": before["rrms"], "rrms": after["rrms"],
                      "failed": after["failed"], "n_sim": problem.n_sim,
                      "history": result["history"], "params": params,
                      "runtime_s": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--arms", default="C0,T1,M")
    ap.add_argument("--nfev", type=int, default=400)
    ap.add_argument("--irls", type=int, default=2)
    ap.add_argument("--diff-step", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--devices", default="")
    ap.add_argument("--name", default="real_dev_refit")
    args = ap.parse_args()
    stage1 = Path(args.stage1)
    folds = json.loads((stage1 / "fold_corrections.json").read_text())
    tags = json.loads(SPLIT.read_text())["REAL_DEV"]
    if args.devices:
        tags = [t for t in tags if t in args.devices.split(",")]
    options = {"nfev": args.nfev, "irls": args.irls, "diff_step": args.diff_step}
    tasks = [(tag, arm, None if arm == "C0" else folds[tag][arm], options)
             for tag in tags for arm in args.arms.split(",")]
    t0 = time.time()
    rows: dict[str, dict] = {}
    with get_context("spawn").Pool(args.workers) as pool:
        for tag, arm, row in pool.imap_unordered(job, tasks):
            rows.setdefault(arm, {})[tag] = row
            print(json.dumps({"tag": tag, "arm": arm, "start": row["rrms_start"],
                              "end": row["rrms"], "n_sim": row["n_sim"],
                              "s": row["runtime_s"]}), flush=True)
    from e1_lodo import aggregate, wins
    scores = {arm: {t: r["rrms"] for t, r in v.items()} for arm, v in rows.items()}
    summary = {arm: aggregate(v) for arm, v in scores.items()}
    out = {"stage1": str(stage1), "options": options, "summary": summary,
           "rows": rows, "runtime_s": round(time.time() - t0, 1)}
    if {"M", "T1", "C0"} <= set(scores) and len(tags) == 12:
        m, t1 = summary["M"], summary["T1"]
        checks = {
            "M_mean_le_0.090": bool(m["all_device_mean"] <= 0.090),
            "M_mean_lt_T1_mean": bool(m["all_device_mean"] < t1["all_device_mean"]),
            "M_beats_T1_on_ge_8_of_12": bool(wins(scores["M"], scores["T1"]) >= 8),
            "M_beats_C0refit_on_ge_8_of_12": bool(wins(scores["M"], scores["C0"]) >= 8),
            "M_zero_failed": bool(m["n_failed"] == 0)}
        out["criterion"] = {"checks": checks, "stage2_pass": all(checks.values()),
                            "note": "C0 here is the no-correction re-polish control; "
                                    "T1 is re-polished with T1 in the loop"}
    target = stage1.parent / args.name
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"summary": summary, "criterion": out.get("criterion")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
