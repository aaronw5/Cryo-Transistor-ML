#!/usr/bin/env python3
"""E5 probe: shared multiplicative discrepancy, leave-one-device-out on REAL_DEV.

Pre-registration: out/research/explore/E5/prereg.json (do not edit).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e5_shared_discrepancy.py stage1
  PYTHONPATH=src .venv/bin/python scripts/research/explore/e5_shared_discrepancy.py stage2 --workers 4

stage1: simulate each REAL_DEV base card once (12 + 12 NGSpice calls), train the
residual arms per fold (MPS if available, float32), score held-out devices, write
out/research/explore/E5/stage1.json and the fold models.
stage2: per fold, TRF polish of the 43-parameter card with the fold's frozen
delta inside the residual, against the same polish without delta.
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

from cryoml.config import OUT_DIR  # noqa: E402
from cryoml.explore import shared_discrepancy as sd  # noqa: E402

OUT = OUT_DIR / "research" / "explore" / "E5"
BASES = {
    "b43": OUT_DIR / "research" / "polished" / "direct43_r3_fd" / "real" / "summary.json",
    "b7": OUT_DIR / "research" / "polished" / "direct7_fd" / "real" / "summary.json",
}
SPEC43 = ROOT / "out" / "cold_loop" / "full_direct" / "round_3" / "spec.json"


def decide(a0: dict, arm: dict, tags: list[str]) -> dict:
    m0 = float(np.mean([a0[t] for t in tags]))
    m1 = float(np.mean([arm[t] for t in tags]))
    red = (m0 - m1) / m0
    # NaN (failed device, strict scoring) never counts as an improvement.
    wins = int(sum(bool(arm[t] < a0[t]) for t in tags))
    n_failed = {"control": int(sum(not np.isfinite(a0[t]) for t in tags)),
                "arm": int(sum(not np.isfinite(arm[t]) for t in tags))}
    if not np.isfinite(red):
        # The prereg does not cover a failed device; report it, do not decide.
        verdict = "UNDECIDED_FAILED_DEVICE"
    elif red >= 0.15 and wins >= 8:
        verdict = "CONTINUE"
    elif red < 0.05 or wins <= 6:
        verdict = "KILL"
    else:
        verdict = "AMBIGUOUS"
    return {"control_mean": m0, "arm_mean": m1, "relative_reduction": red,
            "devices_improved": wins, "n": len(tags), "n_failed": n_failed,
            "verdict": verdict}


DEVIATIONS = [
    "optimizer is torch AdamW (decoupled weight decay 1e-4); the prereg text says Adam with "
    "weight decay 1e-4",
    "after the prereg weights (|I_meas|/den)^2/(n_points_curve*n_curves_device) the weights "
    "are renormalized to sum 1 per device (equal total weight per device, as the prereg "
    "states as the intent)",
    "training runs on MPS in float32 when available",
]


def stage1(args) -> int:
    tags = sd.dev_tags()
    if args.tags:
        tags = [t for t in tags if t in args.tags.split(",")]
    sd.assert_dev(tags)
    steps = args.steps
    seeds = tuple(range(args.seeds))
    report = {"prereg": str(OUT / "prereg.json"), "tags": tags,
              "deviations_from_prereg_text": DEVIATIONS,
              "light": bool(args.tags or args.bases or steps != 3000 or args.seeds != 5),
              "options": {"steps": steps, "seeds": list(seeds)}, "bases": {}}
    t0 = time.time()
    for base, path in BASES.items():
        if args.bases and base not in args.bases.split(","):
            continue
        cards = sd.load_base_cards(path)
        stored = json.loads(path.read_text())["devices"]
        data = {t: sd.load_device(t, cards[t]) for t in tags}
        arms = {k: {} for k in ("A0_control", "A1_const", "A2_primary",
                                "A4_no_geometry", "A3_oracle")}
        a0_check = {}
        models = {}
        for h in tags:
            d = data[h]
            arms["A0_control"][h] = sd.score(h, d.device, d.curves, d.sims, None)["rrms"]
            a0_check[h] = arms["A0_control"][h] - float(stored[h]["rrms"])
            train = [data[t] for t in tags if t != h]
            x = np.concatenate([t.x for t in train])
            r = np.concatenate([t.r for t in train])
            w = np.concatenate([t.w for t in train])
            const = sd.fit_const(x, r, w)
            mlp = sd.fit_mlp(x, r, w, steps=steps, seeds=seeds)
            nogeo = sd.fit_mlp(x, r, w, drop_cols=sd.GEOMETRY_COLUMNS,
                               steps=steps, seeds=seeds)
            oracle = sd.fit_mlp(d.x, d.r, d.w, steps=steps, seeds=seeds)
            for arm, model in (("A1_const", const), ("A2_primary", mlp),
                               ("A4_no_geometry", nogeo), ("A3_oracle", oracle)):
                arms[arm][h] = sd.score(h, d.device, d.curves, d.sims, model)["rrms"]
            models[h] = mlp.to_json()
            print(f"{base} {h}: A0 {arms['A0_control'][h]:.4f} A1 {arms['A1_const'][h]:.4f} "
                  f"A2 {arms['A2_primary'][h]:.4f} A4 {arms['A4_no_geometry'][h]:.4f} "
                  f"A3 {arms['A3_oracle'][h]:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        (OUT / "models").mkdir(parents=True, exist_ok=True)
        prefix = "light_" if report["light"] else ""
        (OUT / "models" / f"{prefix}{base}_A2_folds.json").write_text(json.dumps(models))
        report["bases"][base] = {
            "cards": str(path),
            "a0_minus_stored_rrms_max_abs": float(np.max(np.abs(list(a0_check.values())))),
            "per_device": arms,
            "aggregate": {k: sd.aggregate(v) for k, v in arms.items()},
            "vs_control": {k: decide(arms["A0_control"], v, tags)
                           for k, v in arms.items() if k != "A0_control"},
        }
    report["primary_verdict_stage1"] = (
        report["bases"]["b43"]["vs_control"]["A2_primary"] if not report["light"] else
        "not the pre-registered run (light settings); indicative only")
    report["wall_s"] = round(time.time() - t0, 1)
    name = "stage1_light.json" if report["light"] else "stage1.json"
    (OUT / name).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({b: {k: v["arm_mean"] for k, v in r["vs_control"].items()}
                      | {"A0": r["aggregate"]["A0_control"]["all_dev_mean"]}
                      for b, r in report["bases"].items()}, indent=1))
    print("primary:", json.dumps(report["primary_verdict_stage1"]))
    return 0


def _polish_job(job):
    from cold_loop import box_for
    from cryoml.ngspice_fit import DeviceProblem, polish
    tag, spec, card, model_json, options = job
    model = sd.DeltaModel.from_json(model_json) if model_json else None
    if model is None:
        problem = DeviceProblem(tag, box_for(spec, tag))
    else:
        problem = sd.corrected_problem_class()(tag, box_for(spec, tag), model)
    start = problem.box.params_to_local(
        {**spec["published_by_device"][tag],
         **{k: v for k, v in card.items() if k in problem.box.names}})
    result = polish(problem, start, max_nfev=options["max_nfev"],
                    diff_step=options["diff_step"], irls_rounds=options["irls"])
    params = problem.box.local_to_params(np.asarray(result["local"]))
    device, curves, sims = sd.simulate_base(tag, params)
    _, _, sims0 = sd.simulate_base(tag, problem.box.local_to_params(start))
    return (tag, model is not None), {
        "start_rrms": sd.score(tag, device, curves, sims0, model)["rrms"],
        "rrms": sd.score(tag, device, curves, sims, model)["rrms"],
        "uncorrected_rrms_of_refit_card": sd.score(tag, device, curves, sims, None)["rrms"],
        "n_sim": problem.n_sim, "history": result["history"],
        "runtime_s": result["runtime_s"], "params": params}


def stage2(args) -> int:
    stage1_verdict = None
    if (OUT / "stage1.json").exists():
        stage1_verdict = json.loads((OUT / "stage1.json").read_text())["primary_verdict_stage1"]
    if args.require_not_kill:
        if stage1_verdict is None:
            raise SystemExit("stage1.json missing: run stage1 first")
        verdict = stage1_verdict
        if verdict["verdict"] == "KILL":
            (OUT / "stage2_skipped.json").write_text(json.dumps(
                {"reason": "pre-registered stage-1 verdict KILL; stage 2 not run",
                 "stage1": verdict}, indent=1) + "\n")
            print("stage 1 verdict KILL: stage 2 skipped per prereg")
            return 0
    tags = sd.dev_tags()
    if args.tags:
        tags = [t for t in tags if t in args.tags.split(",")]
    sd.assert_dev(tags)
    models_path = Path(args.models) if args.models else OUT / "models" / "b43_A2_folds.json"
    models = json.loads(models_path.read_text())
    spec = json.loads(SPEC43.read_text())
    cards = sd.load_base_cards(BASES["b43"])
    options = {"max_nfev": args.max_nfev, "diff_step": 0.01, "irls": 2}
    jobs = []
    for t in tags:
        jobs.append((t, spec, cards[t], None, options))
        jobs.append((t, spec, cards[t], models[t], options))
    t0 = time.time()
    with get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        res = dict(pool.imap_unordered(_polish_job, jobs))
    control = {t: res[(t, False)]["rrms"] for t in tags}
    joint = {t: res[(t, True)]["rrms"] for t in tags}
    killed = isinstance(stage1_verdict, dict) and stage1_verdict.get("verdict") == "KILL"
    out = {"prereg": str(OUT / "prereg.json"), "tags": tags, "options": options,
           "stage1_verdict": stage1_verdict,
           "role": ("diagnostic only: the pre-registered stage-1 verdict is KILL, so this "
                    "stage-2 result does not change the E5 decision" if killed else
                    "pre-registered stage 2"),
           "light": bool(args.tags or args.max_nfev != 400 or args.models),
           "models": str(models_path),
           "control_polish": {t: res[(t, False)] for t in tags},
           "joint_refit": {t: res[(t, True)] for t in tags},
           "aggregate": {"control": sd.aggregate(control), "joint": sd.aggregate(joint)},
           "verdict_stage2": decide(control, joint, tags) if not (
               args.tags or args.max_nfev != 400 or args.models) else
           "not the pre-registered run (light settings); indicative only",
           "wall_s": round(time.time() - t0, 1)}
    name = ("stage2_light.json" if out["light"] else
            "stage2_diagnostic.json" if killed else "stage2.json")
    (OUT / name).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"aggregate": out["aggregate"], "verdict": out["verdict_stage2"],
                      "wall_s": out["wall_s"]}, indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["stage1", "stage2"])
    p.add_argument("--tags", default="", help="comma list (light runs only)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--max-nfev", type=int, default=400)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--require-not-kill", action="store_true",
                   help="stage2: run only if stage1.json verdict is not KILL")
    p.add_argument("--bases", default="", help="comma list of b43,b7 (light runs only)")
    p.add_argument("--models", default="", help="stage2 fold-model json (light runs only)")
    args = p.parse_args()
    return stage1(args) if args.stage == "stage1" else stage2(args)


if __name__ == "__main__":
    raise SystemExit(main())
