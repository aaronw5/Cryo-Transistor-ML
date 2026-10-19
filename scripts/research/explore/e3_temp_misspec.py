#!/usr/bin/env python3
"""E3: temperature generalization under misspecification (simulated only).

Pre-registration: out/research/explore/E3/prereg.json.  Model and laws:
``cryoml.explore.e3_temp`` and ``cryoml.explore.e3_learn``.

Stages (each result file is written once; a rerun resumes):

  train   per training device: truth curves on the dense grid, then
          per-temperature seven-parameter NGSpice fits at every dense
          temperature (the learned laws' training data)
  test    per test device: truth curves at 77/110/150/190/225/260/300 K;
          per-temperature fits at the O4 anchors (77/150/225/300 K), the O2
          anchors (77/300 K) and, as a floor reference only, at each
          held-out temperature (110/190/260 K, using held-out data)
  law     parametric laws fitted on NGSpice to the observed temperatures
  learn   learned laws (CPU torch) and linear interpolation
  eval    NGSpice re-simulation of every method's card at every test
          temperature, scored against the truth (strict paper RRMS)
  summary tables, paired bootstrap, pre-registered decision

Devices: only REAL_DEV geometries (out/research/explore/real_split.json).

  PYTHONPATH=src .venv/bin/python scripts/research/explore/e3_temp_misspec.py \
      --name full --tier M --n-train 6 --n-test 3 --workers 8
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

from cryoml.config import OUT_DIR  # noqa: E402
from cryoml.explore import e3_temp as E  # noqa: E402

E3_DIR = OUT_DIR / "research" / "explore" / "E3"
SPLIT = OUT_DIR / "research" / "explore" / "real_split.json"
TIER_CODE = {"W": 0, "M": 1}
SETTINGS = {"O4": ("par3", "par14", "par3_pop"), "O2": ("par3_pop", "par3_g1", "par14_g1")}
PRIMARY_CONTROL = {"O4": "par3", "O2": "par3_pop"}
LEARNED = ("lrn", "lrn_param", "lrn_noH", "lrn_mean")


def run_dir(args) -> Path:
    return E3_DIR / "runs" / args.name / f"tier_{args.tier}"


def geometry(tag: str) -> list[float]:
    from cryoml.devices import PAPER_DEVICES
    from cryoml.foundation import geometry_features, geometry_ranges
    return geometry_features(E.device_by_tag(tag), geometry_ranges(PAPER_DEVICES)).tolist()


def rng_for(seed, tier, split, tag_index, draw):
    return np.random.default_rng([seed, TIER_CODE[tier], split, tag_index, draw])


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


# ------------------------------------------------------------- stages

def train_job(job):
    tag, tag_index, i, tier, seed, dense, out = job
    path = Path(out)
    if path.exists():
        return str(path)
    t0 = time.time()
    truth = E.draw_truth(tag, tier, rng_for(seed, tier, 0, tag_index, i))
    targets = {t: E.simulate_truth_at(tag, truth, t) for t in dense}
    bad = [t for t, v in targets.items() if not np.all(np.isfinite(v))]
    fits = E.fit_sequence(tag, targets, tuple(dense))
    dump(path, {"tag": tag, "draw": i, "truth": truth, "geometry": geometry(tag),
                "nonfinite_truth_temps": bad, "fits": fits,
                "n_sim": int(sum(r["n_sim"] for r in fits.values())),
                "runtime_s": round(time.time() - t0, 1)})
    return str(path)


def test_job(job):
    tag, tag_index, j, tier, seed, out = job
    path = Path(out)
    if path.exists():
        return str(path)
    t0 = time.time()
    truth = E.draw_truth(tag, tier, rng_for(seed, tier, 1, tag_index, j))
    targets = {t: E.simulate_truth_at(tag, truth, t) for t in E.TEMPS_TEST}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), temps=np.array(E.TEMPS_TEST),
             ids=np.stack([targets[t] for t in E.TEMPS_TEST]))
    o4 = E.fit_sequence(tag, targets, E.OBS_SETS["O4"])
    x77 = np.asarray(o4["77"]["x"])
    o2 = {"77": o4["77"],
          "300": E.fit_temperature(tag, 300.0, targets[300.0],
                                   [x77] + E.bank_starts(tag, 300.0, targets[300.0], 1))}
    floor = {}
    anchors = E.OBS_SETS["O4"]
    ax = np.array([o4[f"{t:g}"]["x"] for t in anchors])
    for t in E.TEMPS_HELDOUT:
        lower = max(a for a in anchors if a < t)
        starts = [E.linear_interp(anchors, ax, t), np.asarray(o4[f"{lower:g}"]["x"])]
        starts += E.bank_starts(tag, t, targets[t], 1)
        floor[f"{t:g}"] = E.fit_temperature(tag, t, targets[t], starts)
    n_sim = sum(r["n_sim"] for r in list(o4.values()) + [o2["300"]] + list(floor.values()))
    dump(path, {"tag": tag, "draw": j, "truth": truth, "geometry": geometry(tag),
                "nonfinite_truth_temps": [t for t, v in targets.items()
                                          if not np.all(np.isfinite(v))],
                "fits": {"O4": o4, "O2": o2, "floor": floor}, "n_sim": int(n_sim),
                "runtime_s": round(time.time() - t0, 1)})
    return str(path)


def load_targets(test_json: Path) -> dict:
    with np.load(test_json.with_suffix(".npz")) as data:
        return {float(t): data["ids"][k] for k, t in enumerate(data["temps"])}


def law_job(job):
    test_path, setting, variant, pop_lg, law_iter, out = job
    path = Path(out)
    if path.exists():
        return str(path)
    t0 = time.time()
    rec = json.loads(Path(test_path).read_text())
    targets = load_targets(Path(test_path))
    temps = E.OBS_SETS[setting]
    law = E.fit_law(rec["tag"], targets, temps, variant, rec["fits"][setting],
                    pop_lg=np.asarray(pop_lg), max_iter=law_iter)
    pred = {f"{t:g}": E.law_x(np.asarray(law["mu"]), np.asarray(law["c"]),
                              np.asarray(law["lg"]), t).tolist() for t in E.TEMPS_TEST}
    dump(path, {**law, "pred": pred, "runtime_s": round(time.time() - t0, 1)})
    return str(path)


def eval_job(job):
    from cryoml.spice_batch import simulate_pdk_multi
    test_path, cards, out = job      # cards: {temp: [(key, x), ...]}
    path = Path(out)
    if path.exists():
        return str(path)
    rec = json.loads(Path(test_path).read_text())
    tag = rec["tag"]
    targets = load_targets(Path(test_path))
    device, curves, bin_index = E.device_ctx(tag)
    result = {}
    for temp_s, items in cards.items():
        temp = float(temp_s)
        scorer = E.TempScorer(tag, temp, targets[temp])
        sims = simulate_pdk_multi(device.dev_type, device.L_um, device.W_um, curves,
                                  [E.card(tag, np.asarray(x)) for _, x in items],
                                  bin_index=bin_index, method="altermod", temp_K=temp,
                                  timeout_s=20)
        for (key, _), sim in zip(items, sims):
            result.setdefault(key, {})[temp_s] = scorer.rrms(sim)
    dump(path, result)
    return str(path)


# ------------------------------------------------------------ driver

def pool_map(fn, jobs, workers):
    if not jobs:
        return []
    if workers <= 1:
        return [fn(j) for j in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return list(pool.imap_unordered(fn, jobs))


def cmd_run(args):
    base = run_dir(args)
    base.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text())
    tags = args.tags.split(",") if args.tags else split["REAL_DEV"]
    if any(t in split["REAL_TEST"] for t in tags):
        raise SystemExit("REAL_TEST geometries are not allowed in exploration probes")
    dense = tuple(float(t) for t in args.dense.split(",")) if args.dense else E.TEMPS_DENSE
    for t in set(E.OBS_SETS["O4"]) | set(E.TEMPS_HELDOUT):
        if t not in dense:
            raise SystemExit(f"dense grid must contain {t}")
    config = {"tags": tags, "tier": args.tier, "n_train": args.n_train,
              "n_test": args.n_test, "dense": dense, "seed": args.seed,
              "law_iter": args.law_iter, "learn_seeds": args.learn_seeds}
    dump(base / "config.json", config)
    t0 = time.time()

    jobs = [(tag, k, i, args.tier, args.seed, dense,
             str(base / "train" / f"{tag}__{i}.json"))
            for k, tag in enumerate(tags) for i in range(args.n_train)]
    jobs += [(tag, k, j, args.tier, args.seed, str(base / "test" / f"{tag}__{j}.json"))
             for k, tag in enumerate(tags) for j in range(args.n_test)]
    jobs.sort(key=lambda j: len(j) == 6)      # train jobs first
    train_jobs = [j for j in jobs if len(j) == 7]
    test_jobs = [j for j in jobs if len(j) == 6]
    pool_map(train_job, train_jobs, args.workers)
    print(f"train done {time.time() - t0:.0f}s", flush=True)
    pool_map(test_job, test_jobs, args.workers)
    print(f"test fits done {time.time() - t0:.0f}s", flush=True)

    train = [json.loads(Path(j[-1]).read_text()) for j in train_jobs]
    tests = {Path(j[-1]).stem: Path(j[-1]) for j in test_jobs}
    pop_lg = E.pooled_lg([r["fits"] for r in train], dense)
    dump(base / "pooled_lg.json", {"lg": pop_lg.tolist(), "g": np.exp(pop_lg).tolist()})

    law_jobs = [(str(p), s, v, pop_lg.tolist(), args.law_iter,
                 str(base / "law" / f"{s}__{v}__{dev}.json"))
                for dev, p in tests.items() for s, variants in SETTINGS.items()
                for v in variants]
    pool_map(law_job, law_jobs, args.workers)
    print(f"laws done {time.time() - t0:.0f}s", flush=True)

    preds_path = base / "preds.json"
    if not preds_path.exists():
        preds = learn(train, tests, dense, args)
        for job in law_jobs:
            s, v, dev = Path(job[-1]).stem.split("__", 2)
            preds[s][v][dev] = json.loads(Path(job[-1]).read_text())["pred"]
        dump(preds_path, preds)
    preds = json.loads(preds_path.read_text())
    print(f"learn done {time.time() - t0:.0f}s", flush=True)

    eval_jobs = []
    for dev, p in tests.items():
        rec = json.loads(p.read_text())
        cards = {}
        for s, methods in preds.items():
            for m, by_dev in methods.items():
                for t_s, x in by_dev[dev].items():
                    cards.setdefault(t_s, []).append((f"{s}|{m}", x))
        for t_s, row in rec["fits"]["floor"].items():
            cards[t_s].append(("floor|floor", row["x"]))
        eval_jobs.append((str(p), cards, str(base / "eval" / f"{dev}.json")))
    pool_map(eval_job, eval_jobs, args.workers)
    print(f"eval done {time.time() - t0:.0f}s", flush=True)
    summary = summarize(base, tests, train)
    summary["wall_s"] = round(time.time() - t0, 1)
    dump(base / "summary.json", summary)
    print_summary(summary)


def learn(train, tests, dense, args) -> dict:
    from cryoml.explore.e3_learn import MeanLaw, NetLaw
    preds = {s: {m: {} for m in ("pt_interp",) + LEARNED + SETTINGS[s]} for s in SETTINGS}
    info = {}
    for s, anchors in E.OBS_SETS.items():
        no_h = tuple(t for t in dense if t not in E.TEMPS_HELDOUT)
        models = {"lrn": NetLaw(train, anchors, dense, "sens", seeds=args.learn_seeds),
                  "lrn_param": NetLaw(train, anchors, dense, "param", seeds=args.learn_seeds),
                  "lrn_mean": MeanLaw(train, anchors, dense)}
        if any(t not in anchors for t in no_h):     # empty on a coarse smoke grid
            models["lrn_noH"] = NetLaw(train, anchors, no_h, "sens", seeds=args.learn_seeds)
        else:
            del preds[s]["lrn_noH"]
        info[s] = {k: {"sigma": m.sigma.tolist(),
                       "n_rows": getattr(m, "n_rows", None),
                       "final_loss": getattr(m, "history", None)} for k, m in models.items()}
        for dev, p in tests.items():
            rec = json.loads(p.read_text())
            ax = np.array([rec["fits"][s][f"{t:g}"]["x"] for t in anchors])
            temps = list(E.TEMPS_TEST)
            preds[s]["pt_interp"][dev] = {f"{t:g}": E.linear_interp(anchors, ax, t).tolist()
                                         for t in temps}
            for k, m in models.items():
                out = m.predict(rec["geometry"], ax, temps)
                preds[s][k][dev] = {f"{t:g}": x.tolist() for t, x in zip(temps, out)}
    dump(Path(run_dir(args)) / "learn_info.json", info)
    return preds


# ----------------------------------------------------------- summary

def bootstrap_reduction(a, b, n=4000, seed=0):
    """1 - mean(a)/mean(b) with a paired percentile 95% CI over devices."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    values = 1.0 - a[idx].mean(1) / b[idx].mean(1)
    return {"reduction": float(1.0 - a.mean() / b.mean()),
            "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "n_devices": int(len(a))}


def summarize(base, tests, train) -> dict:
    evals = {dev: json.loads((base / "eval" / f"{dev}.json").read_text()) for dev in tests}
    keys = sorted(next(iter(evals.values())).keys())
    held = [f"{t:g}" for t in E.TEMPS_HELDOUT]
    table = {}
    for key in keys:
        cells = np.array([[evals[d][key].get(t, np.nan) for t in held] for d in tests])
        per_dev = cells.mean(1)
        temps_here = sorted(next(iter(evals.values()))[key], key=float)
        row = {"heldout_all_device_mean": float(np.nanmean(per_dev)),
               "n_failed_cells": int(np.isnan(cells).sum()), "n_cells": int(cells.size),
               "per_temp": {t: float(np.nanmean([evals[d][key][t] for d in tests]))
                            for t in temps_here}}
        setting = key.split("|")[0]
        if setting in E.OBS_SETS:
            row["anchor_all_device_mean"] = float(np.nanmean(
                [[evals[d][key][f"{t:g}"] for t in E.OBS_SETS[setting]] for d in tests]))
        table[key] = row
    devs = list(tests)

    def per_device(key):
        return np.array([np.mean([evals[d][key][t] for t in held]) for d in devs])

    floor = per_device("floor|floor")
    decisions = {}
    for s in SETTINGS:
        lrn, ctrl, interp = (per_device(f"{s}|{m}") for m in
                             ("lrn", PRIMARY_CONTROL[s], "pt_interp"))
        ok = np.isfinite(lrn) & np.isfinite(ctrl) & np.isfinite(interp)
        frac_fail = {m: table[f"{s}|{m}"]["n_failed_cells"] / table[f"{s}|{m}"]["n_cells"]
                     for m in ("lrn", PRIMARY_CONTROL[s], "pt_interp")}
        vs_ctrl = bootstrap_reduction(lrn[ok], ctrl[ok])
        vs_interp = bootstrap_reduction(lrn[ok], interp[ok])
        gap = float((ctrl[ok].mean() - lrn[ok].mean())
                    / max(ctrl[ok].mean() - floor[ok].mean(), 1e-12))
        others = {m: bootstrap_reduction(lrn[ok], per_device(f"{s}|{m}")[ok])
                  for m in SETTINGS[s] + ("lrn_mean",) if m != PRIMARY_CONTROL[s]
                  and np.all(np.isfinite(per_device(f"{s}|{m}")[ok]))}
        decisions[s] = {"control": PRIMARY_CONTROL[s], "lrn_vs_control": vs_ctrl,
                        "lrn_vs_pt_interp": vs_interp, "lrn_vs_other": others,
                        "gap_closed_to_floor": gap, "failed_fraction": frac_fail,
                        "meets_continue": bool(
                            vs_ctrl["reduction"] >= 0.25 and vs_ctrl["ci95"][0] > 0
                            and vs_interp["reduction"] >= 0.10
                            and max(frac_fail.values()) <= 0.10)}
    kill = all(decisions[s]["lrn_vs_control"]["reduction"] < 0.10 for s in SETTINGS) or \
        all(decisions[s]["lrn_vs_pt_interp"]["reduction"] < 0.0 for s in SETTINGS)
    verdict = ("CONTINUE" if any(d["meets_continue"] for d in decisions.values())
               else "KILL" if kill else "INCONCLUSIVE")
    return {"table": table, "decisions": decisions, "verdict_rule_output": verdict,
            "n_test_devices": len(devs), "n_train_devices": len(train),
            "mean_train_fit_rrms": float(np.nanmean(
                [r["rrms"] for rec in train for r in rec["fits"].values()]))}


def print_summary(summary):
    print("| key | held-out mean | anchors mean | per temperature | failed |")
    for key, r in summary["table"].items():
        pt = " ".join(f"{t}:{v:.4f}" for t, v in r["per_temp"].items())
        print(f"| {key} | {r['heldout_all_device_mean']:.4f} | "
              f"{r.get('anchor_all_device_mean', float('nan')):.4f} | {pt} | "
              f"{r['n_failed_cells']}/{r['n_cells']} |")
    for s, d in summary["decisions"].items():
        print(s, json.dumps({k: d[k] for k in ("lrn_vs_control", "lrn_vs_pt_interp",
                                               "gap_closed_to_floor", "meets_continue")}))
    print("verdict (rule output):", summary["verdict_rule_output"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("M", "W"), default="M")
    parser.add_argument("--tags", default=None, help="comma list (default: REAL_DEV)")
    parser.add_argument("--n-train", type=int, default=6)
    parser.add_argument("--n-test", type=int, default=3)
    parser.add_argument("--dense", default=None, help="comma list of training temperatures")
    parser.add_argument("--law-iter", type=int, default=15)
    parser.add_argument("--learn-seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    cmd_run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
