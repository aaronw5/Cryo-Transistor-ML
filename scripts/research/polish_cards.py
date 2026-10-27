#!/usr/bin/env python3
"""Finite-difference NGSpice polish of any method's predicted cards.

Each device's card (from a cold-loop extract dir + method, a posterior eval
json, or an ngspice_direct summary) is the start of bounded TRF least squares
on the fixed-inclusion per-curve residuals against the target curves
(measured, or the CRYOML_TARGET_OVERRIDE simulated devices), with an absolute
two-point NGSpice Jacobian in the ``--spec`` box (the method's own search
box).  The best card seen is kept.

  PYTHONPATH=src .venv/bin/python scripts/research/polish_cards.py \
      --cards out/cold_loop/seven/round_4/extract --method inverse_autoencoder \
      --spec out/cold_loop/seven/round_4/spec.json --run A_r4_fd
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from cryoml.card_eval import aggregate  # noqa: E402
from cryoml.config import OUT_DIR  # noqa: E402
from cryoml.data_io import target_override  # noqa: E402

RUNS = OUT_DIR / "research" / "polished"


def load_cards(path: Path, method: str | None) -> dict[str, dict]:
    if path.is_dir():
        cards = {}
        for item in sorted(path.glob("ml_*.json")):
            row = json.loads(item.read_text())
            cards[row["device"]] = row["methods"][method]["params"]
        return cards
    devices = json.loads(path.read_text())["devices"]
    return {t: r["params"] if "params" in r else r["best_card"]
            for t, r in devices.items()}


def polish_job(job):
    from cold_loop import box_for
    from cryoml.ngspice_fit import DeviceProblem, final_score, polish
    tag, spec, card, options = job
    problem = DeviceProblem(tag, box_for(spec, tag))
    start = problem.box.params_to_local(
        {**spec["published_by_device"][tag],
         **{k: v for k, v in card.items() if k in problem.box.names}})
    # Score the start with the same scorer as the end card (and as the
    # method's own score).  problem.objective() is the optimizer's penalized
    # objective: an included curve whose NGSpice sweep fails (non-finite)
    # costs PENALTY there, while card_eval.score_card silently drops it from
    # the mean.  Mixing the two made before -> after a scorer switch, so both
    # conventions and the failed-curve counts are recorded explicitly.
    before = final_score(problem, start)
    result = polish(problem, start, max_nfev=options["max_nfev"],
                    diff_step=options["diff_step"],
                    irls_rounds=options["irls"])
    scored = final_score(problem, result["local"])
    return tag, {"before_rrms": before["rrms"], "rrms": scored["rrms"],
                 "before_objective": result["history"][0],
                 "objective": result["objective"],
                 "n_included_curves": problem.n_curves,
                 "before_failed_curves": problem.n_curves - before["n_curves"],
                 "failed_curves": problem.n_curves - scored["n_curves"],
                 "official_dynamic_rrms": scored["official_dynamic_rrms"],
                 "params": scored["params"], "n_sim": problem.n_sim,
                 "history": result["history"],
                 "runtime_s": result["runtime_s"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-nfev", type=int, default=400,
                        help="approximate NGSpice simulations per device")
    parser.add_argument("--diff-step", type=float, default=0.01)
    parser.add_argument("--irls", type=int, default=2)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--out", default=None,
                        help="summary.json path (default polished/<run>/summary.json)")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    cards = load_cards(Path(args.cards), args.method)
    options = {"max_nfev": args.max_nfev, "diff_step": args.diff_step,
               "irls": args.irls}
    t0 = time.time()
    tags = sorted(cards, key=lambda t: not t.startswith("pmos"))
    with get_context("spawn").Pool(min(args.workers, len(tags))) as pool:
        results = dict(pool.imap_unordered(
            polish_job, [(t, spec, cards[t], options) for t in tags]))
    before = {t: {"rrms": r["before_rrms"]} for t, r in results.items()}
    failed = {stage: sorted(t for t, r in results.items() if r[key] > 0)
              for stage, key in (("before", "before_failed_curves"),
                                 ("after", "failed_curves"))}
    if failed["before"] or failed["after"]:
        print(f"WARNING: included curves failed to simulate (scores average "
              f"only the finite curves): {failed}", file=sys.stderr)
    out = {"cards": args.cards, "method": args.method, "spec": args.spec,
           "target_override": target_override() is not None,
           "options": options, "devices": results,
           "devices_with_failed_curves": failed,
           "before": aggregate(before), **aggregate(results),
           "mean_ngspice_calls_per_device": float(np.mean(
               [r["n_sim"] for r in results.values()])),
           "wall_s": round(time.time() - t0, 1)}
    path = Path(args.out) if args.out else RUNS / args.run / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"before": out["before"]["all_device_mean_rrms"],
                      "after": out["all_device_mean_rrms"],
                      "combined_after": out["combined_rrms"],
                      "calls": out["mean_ngspice_calls_per_device"],
                      "wall_s": out["wall_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
