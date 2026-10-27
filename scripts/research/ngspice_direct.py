#!/usr/bin/env python3
"""Non-ML reference: fit each device directly against NGSpice.

Every device gets the same procedure inside a cold-loop round box:

1. starts: the box center plus the best ``--bank-starts`` diverse samples of
   an existing NGSpice bank for that box (optional);
2. ``--cmaes-evals`` NGSpice evaluations of CMA-ES on the mean included-curve
   RRMS (the fixed-inclusion paper metric), from the best start;
3. bounded TRF least-squares polish with NGSpice finite differences.

NGSpice call counts are recorded per device so results can be compared with
the surrogate pipeline at matched simulator budgets.

  PYTHONPATH=src .venv/bin/python scripts/research/ngspice_direct.py \
      --spec out/cold_loop/seven/round_1/spec.json \
      --bank data/processed/cold_loop/seven_r1 --run seven_r1_direct
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
from cryoml.data_io import bank_measured  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import curve_trim  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

RUNS = OUT_DIR / "research" / "ngspice_direct"


def bank_starts(problem, bank_dir: Path, n_starts: int,
                min_distance: float = 0.35) -> list[np.ndarray]:
    with np.load(bank_dir / f"{problem.tag}.npz", allow_pickle=True) as data:
        ok = np.asarray(data["ok"], dtype=bool)
        local = np.asarray(data["LOCAL"], dtype=np.float64)[ok]
        ids = np.asarray(data["IDS"], dtype=np.float64)[ok]
        slices = np.asarray(data["slices"])
        kinds = [str(k) for k in data["kinds"]]
        fixeds = np.abs(np.asarray(data["fixeds"], dtype=np.float64))
        meas = bank_measured(data, problem.device)
    columns = []
    for row in problem.layout:
        curve = problem.curves[row["index"]]
        match = [i for i, (k, b) in enumerate(zip(kinds, fixeds))
                 if k == curve.kind and abs(b - abs(float(curve.fixed))) < 5e-3]
        start, stop = (int(v) for v in slices[match[0]])
        trim_start, cleaned, denominator = curve_trim(
            problem.device.dev_type, problem.device.L_um, problem.device.W_um,
            curve.kind, meas[start:stop])
        block = ids[:, start + trim_start:stop]
        target = cleaned[trim_start:]
        columns.append(np.sqrt(np.mean((block - target[None]) ** 2, axis=1))
                       / denominator)
    scores = np.nan_to_num(np.mean(np.stack(columns, 1), axis=1), nan=np.inf)
    active = problem.box.active
    chosen = []
    for index in np.argsort(scores):
        candidate = local[index]
        if not chosen or min(np.sqrt(np.mean((candidate[active] - c[active]) ** 2))
                             for c in chosen) >= min_distance:
            chosen.append(candidate)
        if len(chosen) >= n_starts:
            break
    return chosen, float(scores.min())


def load_init_cards(path: str) -> dict[str, dict]:
    path = Path(path)
    if path.is_dir():
        return {json.loads(p.read_text())["device"]:
                json.loads(p.read_text())["methods"]["surrogate_fd"]["params"]
                for p in sorted(path.glob("ml_*.json"))}
    devices = json.loads(path.read_text())["devices"]
    # posterior_study eval files keep the best NGSpice-checked sample.
    return {t: r["params"] if "params" in r else r["best_card"]
            for t, r in devices.items()}


def fit_device(job) -> tuple[str, dict]:
    import cma
    from cold_loop import box_for
    from cryoml.ngspice_fit import DeviceProblem, final_score, polish
    tag, spec, args = job
    t0 = time.time()
    problem = DeviceProblem(tag, box_for(spec, tag))
    active = np.flatnonzero(problem.box.active)
    starts = [problem.box.local_published]
    if args["init_cards"] and tag in args["init_cards"]:
        # Seed from an earlier fit (e.g. the seven-parameter result inside a
        # 43-parameter box); parameters it does not set stay at the center.
        seeded = {**spec["published_by_device"][tag],
                  **{k: v for k, v in args["init_cards"][tag].items()
                     if k in problem.box.names}}
        starts.append(problem.box.params_to_local(seeded))
    best_bank = None
    if args["bank"]:
        extra, best_bank = bank_starts(problem, Path(args["bank"]),
                                       args["bank_starts"])
        starts += extra
    start_scores = [problem.objective(s) for s in starts]
    order = np.argsort(start_scores)
    record = {"start_scores": [float(v) for v in start_scores],
              "best_bank_rrms": best_bank}
    best_local = np.asarray(starts[order[0]], dtype=np.float64)
    best_value = float(start_scores[order[0]])
    if args["cmaes_evals"] > 0:
        def f(x):
            trial = best_local.copy()
            trial[active] = np.clip(x, -1, 1)
            return problem.objective(trial)
        es = cma.CMAEvolutionStrategy(
            best_local[active], args["sigma0"],
            {"bounds": [-1, 1], "maxfevals": args["cmaes_evals"],
             "seed": args["seed"], "verbose": -9,
             "popsize": args["popsize"] or (4 + int(3 * np.log(len(active))))})
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [f(x) for x in xs])
        x = np.clip(es.result.xbest, -1, 1)
        if es.result.fbest < best_value:
            best_value = float(es.result.fbest)
            best_local = best_local.copy()
            best_local[active] = x
        record["cmaes_best"] = float(es.result.fbest)
        record["cmaes_evals"] = int(es.result.evaluations)
    if args["polish_nfev"] > 0:
        result = polish(problem, best_local, max_nfev=args["polish_nfev"],
                        irls_rounds=args["irls"])
        record["polish_history"] = result["history"]
        if result["objective"] < best_value:
            best_value, best_local = result["objective"], result["local"]
    scored = final_score(problem, best_local)
    record.update({"rrms": scored["rrms"],
                   "official_dynamic_rrms": scored["official_dynamic_rrms"],
                   "per_curve": scored["per_curve"], "params": scored["params"],
                   "local": scored["local"], "n_sim": problem.n_sim,
                   "runtime_s": round(time.time() - t0, 1)})
    return tag, record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--bank", default=None)
    parser.add_argument("--run", required=True)
    parser.add_argument("--bank-starts", type=int, default=4)
    parser.add_argument("--cmaes-evals", type=int, default=3000)
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--popsize", type=int, default=0)
    parser.add_argument("--polish-nfev", type=int, default=400)
    parser.add_argument("--irls", type=int, default=2)
    parser.add_argument("--init", default=None,
                        help="ngspice_direct summary.json or cold-loop extract "
                             "dir whose cards seed one extra start")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--out", default=None,
                        help="summary.json path (default runs/<run>/summary.json)")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    tags = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]
    if args.devices:
        tags = [t for t in tags if t in args.devices.split(",")]
    options = {k: getattr(args, k) for k in (
        "bank", "bank_starts", "cmaes_evals", "sigma0", "popsize",
        "polish_nfev", "irls", "seed", "init")}
    options["init_cards"] = load_init_cards(args.init) if args.init else None
    t0 = time.time()
    # Longest-running (pMOS) devices first keeps the pool busy.
    tags.sort(key=lambda t: not t.startswith("pmos"))
    with get_context("spawn").Pool(min(args.workers, len(tags))) as pool:
        results = dict(pool.imap_unordered(
            fit_device, [(t, spec, options) for t in tags]))
    options.pop("init_cards")
    out = {"spec": args.spec, "options": options, "devices": results,
           **aggregate(results),
           "mean_ngspice_calls_per_device": float(np.mean(
               [r["n_sim"] for r in results.values()])),
           "wall_s": round(time.time() - t0, 1)}
    path = Path(args.out) if args.out else RUNS / args.run / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in (
        "all_device_mean_rrms", "nmos_rrms", "pmos_rrms",
        "mean_ngspice_calls_per_device", "wall_s")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
