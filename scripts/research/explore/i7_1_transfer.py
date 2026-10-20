#!/usr/bin/env python3
"""I7-1: training-set statistic S for the held-out gain of the shared bias-only
correction (I3-2 M_bias); retrospective calibration and prospective test.

Pre-registration: ``out/research/explore/I7_1/prereg.json``; frozen rule:
``out/research/explore/I7_1/prereg_amendment_*.json``.

Modes:
  retro     step 2: S1_lodo, S1_pooled, S2 per fold of the 11 stored datasets
            (no simulation), stored M_bias / C0_refit outcomes, Spearman, OLS.
            Merges retro/s3.json when present.
  s3_retro  S3 (inner leave-one-group-out, reduced M_bias recipe) per fold of
            the stored datasets (pool; queue job).
  amend     write the dated amendment with the frozen OLS rule from retro.
  run       step 3 for one (PDK, replicate): truth sets of the 5 mixing levels,
            starting fits, leave-one-geometry-out M_bias + C0_refit (I3-2 /
            F2 pipeline unchanged), per-fold S (queue job).
  s3_new    S3 per fold of the step-3 datasets (queue job).
  decide    P1 / P2 and secondary statistics on the step-3 datasets.
  figs      scatter plots of S against the outcome.

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i7_1_transfer.py --mode retro
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i7_1_transfer.py \
      --mode run --pdk gf180 --rep 0 --workers 6
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "pdk2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.e1_correction import load_devices, save_devices  # noqa: E402
from cryoml.explore.i7_1_transfer import (S3_HYPER, RandomFunction,  # noqa: E402
                                          injected_log_error, log_ratios, ols, rule_predict,
                                          s3_fold_value, s3_inner_tasks, sign_accuracy,
                                          spearman, spearman_perm_p, stats_for)

EX = ROOT / "out/research/explore"
OUT = EX / "I7_1"
PREREG = OUT / "prereg.json"
STATS = ("S1_lodo", "S1_pooled", "S2")
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
AMP = 0.10
NOISE = 0.01
BASE_SEED = 970001
PDK_SEED = {"gf180": 0, "ihp": 500000}
P1_RHO = -0.50
P2_MIN = 15
MAX_LEFT_OUT = 4
SMOKE = {"devices": 4, "levels": (0.0, 1.0), "seeds": (0,), "steps": 200}


def rel(p: Path) -> str:
    return str(Path(p).relative_to(ROOT))


# ===========================================================================
# Stored datasets (step 2)
# ===========================================================================
def retro_specs() -> list[dict]:
    specs = [{"name": "sky130_77K", "kind": "sky130", "records": [EX / "E1/real_dev"],
              "results": EX / "I3_2/real_dev/results.json", "layout": "i3_2"}]
    for t in (233, 300, 343, 398):
        specs.append({"name": f"ihp_meas_{t}K", "kind": "ihp_meas",
                      "records": [EX / f"I5_1/start/records_{t}K"],
                      "results": EX / f"I5_1/lodo_{t}K/results.json", "layout": "i3_2"})
    for t in (4, 295):
        specs.append({"name": f"tudelft_{t}K", "kind": "tudelft",
                      "records": [EX / f"I6_1/start/records_{t}K"],
                      "results": EX / f"I6_1/lodo_{t}K/results.json", "layout": "i3_2"})
    for pdk in ("gf180", "ihp"):
        specs.append({"name": f"{pdk}_c", "kind": "pdk2",
                      "records": [EX / f"E1/{pdk}_c_set{s}" for s in (0, 1, 2)],
                      "results": EX / f"I3_2/{pdk}_c_logo/results.json", "layout": "i3_2"})
    for pdk in ("gf180", "ihp"):
        d = EX / f"I3_2/followup/f2_{pdk}_b"
        specs.append({"name": f"{pdk}_b", "kind": "pdk2",
                      "records": [d / f"records/{pdk}_b_set{s}" for s in (0, 1, 2)],
                      "results": d / "results.json", "layout": "f2"})
    return specs


def group_fn(kind: str):
    """Outer and inner fold unit of a dataset kind."""
    if kind in ("sky130", "ihp_meas"):
        return lambda r: r.tag
    if kind == "tudelft":
        return lambda r: r.meta["group"]
    return lambda r: r.geometry


def folds_of(kind: str, recs: list) -> list[tuple[str, list, list]]:
    """(fold id, held-out records, training records), as in the stored runs."""
    g = group_fn(kind)
    out = []
    for f in sorted({g(r) for r in recs}):
        held = [r for r in recs if g(r) == f]
        if kind == "ihp_meas":       # I5-1: training within the held-out device's class
            cls = held[0].meta["cls"]
            train = [r for r in recs if g(r) != f and r.meta["cls"] == cls]
        else:
            train = [r for r in recs if g(r) != f]
        out.append((f, held, train))
    return out


def load_outcomes(spec: dict) -> tuple[dict, dict]:
    res = json.loads(Path(spec["results"]).read_text())
    if spec["layout"] == "f2":
        return res["M_bias"]["per_device"], res["C0_refit"]["per_device"]
    return res["per_device"]["M_bias"], res["per_device"]["C0_refit"]


_CACHE: dict = {}


def dataset_records(name: str) -> tuple[str, list]:
    if name not in _CACHE:
        if name.startswith("new:"):
            pdk, r, lev = name[4:].split("|")
            recs = load_devices(new_dir(pdk, int(r), float(lev)) / "records")
            _CACHE[name] = ("pdk2", recs)
        else:
            spec = next(s for s in retro_specs() if s["name"] == name)
            _CACHE[name] = (spec["kind"], [d for p in spec["records"] for d in load_devices(p)])
    return _CACHE[name]


def fold_rows(kind: str, recs: list, m_bias: dict, c0: dict) -> list[dict]:
    rows = []
    for f, held, train in folds_of(kind, recs):
        tags = [r.tag for r in held]
        lr = log_ratios(m_bias, c0, tags)
        ok = [v for v in lr if np.isfinite(v)]
        rows.append({"fold": f, "held": tags, "n_held": len(tags), "n_held_used": len(ok),
                     "y": float(np.mean(ok)) if ok else float("nan"), "log_ratios": lr,
                     **stats_for(train)})
    return rows


def dataset_summary(rows: list[dict], extra=("S3",)) -> dict:
    w = np.array([r["n_held"] for r in rows], float)
    out = {"n_folds": len(rows), "n_held": int(w.sum())}
    for k in STATS + tuple(extra):
        v = np.array([r.get(k, np.nan) for r in rows], float)
        ok = np.isfinite(v)
        out[k] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
    lr = np.array([x for r in rows for x in r["log_ratios"]], float)
    ok = np.isfinite(lr)
    out["y"] = float(lr[ok].mean()) if ok.any() else float("nan")
    out["gmr"] = float(math.exp(out["y"])) if ok.any() else float("nan")
    out["n_left_out"] = int((~ok).sum())
    return out


def correlations(ds: dict, fold_level: dict) -> dict:
    out = {}
    for k in STATS + ("S3",):
        xs = [v[k] for v in ds.values()]
        ys = [v["y"] for v in ds.values()]
        fx = [r.get(k, np.nan) for rows in fold_level.values() for r in rows]
        fy = [r["y"] for rows in fold_level.values() for r in rows]
        within = {n: spearman([r.get(k, np.nan) for r in rows], [r["y"] for r in rows])
                  for n, rows in fold_level.items()}
        out[k] = {"dataset_spearman": spearman(xs, ys),
                  "dataset_n": int(np.sum(np.isfinite(xs) & np.isfinite(ys))),
                  "fold_spearman_pooled": spearman(fx, fy),
                  "fold_n": int(np.sum(np.isfinite(fx) & np.isfinite(fy))),
                  "fold_spearman_within": within}
        if np.sum(np.isfinite(xs)) >= 3:
            out[k]["dataset_ols"] = ols(xs, ys)
            out[k]["fold_ols"] = ols(fx, fy)
    return out


def retro(args) -> int:
    t0 = time.time()
    s3 = {}
    s3_path = OUT / "retro" / "s3.json"
    if s3_path.exists():
        s3 = json.loads(s3_path.read_text())["per_fold"]
    fold_level, ds = {}, {}
    for spec in retro_specs():
        kind, recs = dataset_records(spec["name"])
        m_bias, c0 = load_outcomes(spec)
        rows = fold_rows(kind, recs, m_bias, c0)
        for r in rows:
            r["S3"] = s3.get(spec["name"], {}).get(r["fold"], float("nan"))
        fold_level[spec["name"]] = rows
        ds[spec["name"]] = dataset_summary(rows)
        print(f"{spec['name']:16s} folds {len(rows):3d} " + " ".join(
            f"{k} {ds[spec['name']][k]:+.3f}" for k in STATS + ("S3",))
            + f"  y {ds[spec['name']]['y']:+.4f} GMR {ds[spec['name']]['gmr']:.3f}", flush=True)
    corr = correlations(ds, fold_level)
    rule = corr["S1_lodo"]["dataset_ols"]
    rule_acc = sign_accuracy(rule, [v["S1_lodo"] for v in ds.values()],
                             [v["y"] for v in ds.values()])
    # leave-one-dataset-out sign accuracy of the rule (exploratory)
    names = list(ds)
    lodo_hits = []
    for n in names:
        rest = [m for m in names if m != n]
        r_ = ols([ds[m]["S1_lodo"] for m in rest], [ds[m]["y"] for m in rest])
        lodo_hits.append(bool(np.sign(rule_predict(r_, ds[n]["S1_lodo"])) == np.sign(ds[n]["y"])))
    result = {"prereg": rel(PREREG), "label": "retrospective calibration (outcomes already seen); "
              "exploratory", "datasets": ds, "correlations": corr,
              "rule_S1_lodo_dataset_ols": rule, "rule_in_sample_sign_accuracy": rule_acc,
              "rule_leave_one_dataset_out_sign_correct": dict(zip(names, lodo_hits)),
              "s3_included": bool(s3), "per_fold": fold_level,
              "runtime_s": round(time.time() - t0, 1)}
    (OUT / "retro").mkdir(parents=True, exist_ok=True)
    (OUT / "retro" / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items() if kk != "fold_spearman_within"}
                      for k, v in corr.items()}, indent=1, default=str))
    print(json.dumps({"rule": rule, "in_sample_sign": rule_acc,
                      "lodo_sign_correct": int(sum(lodo_hits))}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# S3 (pool)
# ---------------------------------------------------------------------------
def s3_task(task):
    from cryoml.explore.i3_2_bias import fit_bias
    name, fold, inner, hyper = task
    kind, recs = dataset_records(name)
    g = group_fn(kind)
    train = next(t for f, _, t in folds_of(kind, recs) if f == fold)
    _, inner_train, scored = next(x for x in s3_inner_tasks(train, g) if x[0] == inner)
    corr = fit_bias("M_bias", inner_train, hyper=hyper, threads=1)
    return name, fold, inner, s3_fold_value(corr, scored)


def run_s3(names: list[str], workers: int, hyper: dict, folds_limit: int = 0) -> dict:
    tasks = []
    for name in names:
        kind, recs = dataset_records(name)
        g = group_fn(kind)
        folds = folds_of(kind, recs)
        if folds_limit:
            folds = folds[:folds_limit]
        for f, _, train in folds:
            tasks += [(name, f, inner, hyper) for inner, _, _ in s3_inner_tasks(train, g)]
    print(f"S3: {len(tasks)} inner fits, hyper {hyper}", flush=True)
    vals: dict = {}
    with get_context("spawn").Pool(workers) as pool:
        for name, f, inner, v in pool.imap_unordered(s3_task, tasks, chunksize=1):
            vals.setdefault(name, {}).setdefault(f, []).extend(v)
    return {n: {f: (float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"))
                for f, v in d.items()} for n, d in vals.items()}


def s3_retro(args) -> int:
    t0 = time.time()
    hyper = dict(S3_HYPER) if not args.smoke else {"seeds": (0,), "steps": 50}
    names = [s["name"] for s in retro_specs()]
    if args.datasets:
        names = [n for n in names if n in args.datasets.split(",")]
    per_fold = run_s3(names, args.workers, hyper, 2 if args.smoke else 0)
    out = OUT / ("smoke_retro" if args.smoke else "retro")
    out.mkdir(parents=True, exist_ok=True)
    (out / "s3.json").write_text(json.dumps(
        {"prereg": rel(PREREG), "hyper": {k: list(v) if isinstance(v, tuple) else v
                                          for k, v in hyper.items()},
         "smoke": bool(args.smoke), "per_fold": per_fold,
         "runtime_s": round(time.time() - t0, 1)}, indent=1) + "\n")
    print(f"runtime {time.time() - t0:.1f} s")
    return 0


# ---------------------------------------------------------------------------
# Amendment (frozen rule)
# ---------------------------------------------------------------------------
def amend(args) -> int:
    res = json.loads((OUT / "retro" / "results.json").read_text())
    rule = res["rule_S1_lodo_dataset_ols"]
    date = _dt.date.today().isoformat()
    path = OUT / f"prereg_amendment_{date}.json"
    if path.exists():
        print(f"{rel(path)} exists; not overwritten")
        return 1
    corr = res["correlations"]
    body = {
        "amends": rel(PREREG), "written": date,
        "status": "written after the step-2 retrospective calibration (outcomes already seen) and "
                  "before any step-3 truth set, starting fit or outcome exists",
        "frozen_rule": {
            "form": "predicted y_D = a + b * S1_lodo_D; predicted sign of ln GMR(M_bias, "
                    "C0_refit) = sign(a + b * S1_lodo_D); predicted gain iff a + b * S1_lodo_D < 0",
            "a": rule["a"], "b": rule["b"], "zero_crossing_S1_lodo": rule["threshold_S"],
            "fit": "OLS over the 11 stored datasets (dataset level), retro/results.json",
            "n": rule["n"], "r2": rule["r2"],
            "slope_sign_note": ("b < 0: higher S1_lodo predicts a lower log ratio (more gain)"
                                if rule["b"] < 0 else
                                "b >= 0: the fitted direction is opposite to the expected one; "
                                "the rule is frozen as fitted")},
        "step2_summary": {
            "dataset_spearman": {k: corr[k]["dataset_spearman"] for k in corr},
            "fold_spearman_pooled": {k: corr[k]["fold_spearman_pooled"] for k in corr},
            "rule_in_sample_sign_accuracy": res["rule_in_sample_sign_accuracy"],
            "rule_leave_one_dataset_out_sign_correct": int(sum(
                res["rule_leave_one_dataset_out_sign_correct"].values())),
            "per_dataset": {n: {k: v[k] for k in ("S1_lodo", "S1_pooled", "S2", "y", "gmr",
                                                  "n_folds")}
                            for n, v in res["datasets"].items()}},
        "step3_unchanged": "the step-3 design, P1 (Spearman <= -0.50), P2 (>= 15 of 20 signs) "
                           "and the verdict rule of prereg.json are unchanged",
    }
    path.write_text(json.dumps(body, indent=1) + "\n")
    print(json.dumps(body["frozen_rule"], indent=1))
    return 0


def frozen_rule() -> dict:
    paths = sorted(OUT.glob("prereg_amendment_*.json"))
    if not paths:
        raise RuntimeError("no amendment with a frozen rule")
    return json.loads(paths[0].read_text())["frozen_rule"]


# ===========================================================================
# Step 3: new simulated datasets
# ===========================================================================
def tier_name(rep: int, lam: float, smoke: bool = False) -> str:
    return f"i7_1{'smoke' if smoke else ''}_r{rep}_l{int(round(100 * lam)):03d}"


def new_dir(pdk: str, rep: int, lam: float, smoke: bool = False) -> Path:
    return OUT / ("smoke_new" if smoke else "new") / f"{pdk}_{tier_name(rep, lam, smoke)}"


def make_truth_sets(pdk: str, rep: int, levels, n_devices: int = 0,
                    smoke: bool = False) -> dict:
    """Write the truth sets of all levels of one (PDK, replicate); one base
    simulation per device, shared by the levels."""
    from make_truth import draw_truth
    from cryoml.pdk2 import boxes, metric
    from cryoml.pdk2.problem import TRUTH_ROOTS
    sim = boxes.backend(pdk)
    devices = list(sim.DEVICES)
    if n_devices:
        devices = ([d for d in devices if d.dev_type == "nmos"][:n_devices // 2]
                   + [d for d in devices if d.dev_type == "pmos"][:n_devices - n_devices // 2])
    base = BASE_SEED + PDK_SEED[pdk] + 10000 * rep
    options = {"draw_set": "seven", "misspec_width": 0.0, "rext_range": (25.0, 100.0),
               "noise": 0.0, "temp_offset": 0.0, "c_misspec_width": 0.0,
               "c_rext_k": (1.0, 2.0), "c_dtmax": (20.0, 40.0), "c_noise": 0.0, "c_gain": 0.0}
    shared = {p: RandomFunction.draw(base + 5000 + i) for i, p in enumerate(("nmos", "pmos"))}
    rows = {}
    for dev in devices:
        idx = sim.TAGS.index(dev.tag)
        row = draw_truth(dev, "a", base + idx, options)
        curves = sim.device_curves(dev.dev_type)
        specific = RandomFunction.draw(base + 6000 + idx)
        noise_rng = np.random.default_rng(base + 7000 + idx)
        noise = [noise_rng.standard_normal(len(c.Vg)) for c in curves]
        rows[dev.tag] = (dev, row, curves, specific, noise)
    written = {}
    for lam in levels:
        out = TRUTH_ROOTS[pdk] / tier_name(rep, lam, smoke)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {"tags": np.asarray([d.tag for d in devices])}
        meta = {"tier": tier_name(rep, lam, smoke), "generator": "I7-1 (prereg out/research/explore/"
                "I7_1/prereg.json step3_prospective)", "pdk": f"{sim.PDK_NAME}@{sim.PDK_VERSION}",
                "fit_temp_C": sim.TEMP_C, "lambda": lam, "amp": AMP, "noise": NOISE,
                "base_seed": base, "replicate": rep,
                "shared_functions": {p: f.to_json() for p, f in shared.items()}, "sets": {"0": {}}}
        for tag, (dev, row, curves, specific, noise) in rows.items():
            obs = []
            for c, clean, z in zip(curves, row["clean"], noise):
                x1, x2 = np.abs(c.Vg) / sim.VDD, np.abs(c.Vd) / sim.VDD
                e = injected_log_error(x1, x2, lam, AMP, shared[dev.dev_type], specific)
                obs.append(np.asarray(clean, float) * np.exp(e) * (1.0 + NOISE * z))
            for i, c in enumerate(obs):
                arrays[f"{tag}__{i}"] = c
            arrays[f"{tag}__ids"] = np.concatenate(obs)
            arrays[f"{tag}__clean_ids"] = np.concatenate(row["clean"])
            arrays[f"{tag}__local_seven"] = row["local_seven"]
            arrays[f"{tag}__local_wide"] = row["local_wide"]
            m = dict(row["meta"])
            m["include_tags"] = metric.include_tags(sim.device_curves(dev.dev_type, obs))
            m.update(noise_rel=NOISE, noise_seed=base + 7000 + sim.TAGS.index(tag),
                     specific_seed=base + 6000 + sim.TAGS.index(tag),
                     specific_function=specific.to_json())
            meta["sets"]["0"][tag] = m
        np.savez(out / "set_0.npz", **arrays)
        (out / "truth.json").write_text(json.dumps(meta, indent=1) + "\n")
        written[lam] = str(out.relative_to(ROOT))
    return written


def collect_task(task):
    from e1_collect_pdk2 import collect
    pdk, tier, tag = task
    recs = collect(pdk, tier, 0, "wide", 300, 3, [tag])
    if len(recs) != 1:
        raise RuntimeError(f"{pdk} {tier} {tag}: {len(recs)} records")
    return recs[0]


def run(args) -> int:
    from i3_2_followup import describe, run_lodo, smoke_hyper
    from cryoml.explore.e1_correction import Correction
    pdk, rep, smoke = args.pdk, args.rep, args.smoke
    levels = SMOKE["levels"] if smoke else LEVELS
    t_start = time.time()
    todo = [lam for lam in levels
            if not (new_dir(pdk, rep, lam, smoke) / "results.json").exists()]
    if not todo:
        print("all levels done")
        return 0
    t0 = time.time()
    truth = make_truth_sets(pdk, rep, levels, SMOKE["devices"] if smoke else 0, smoke)
    print(f"truth sets ({time.time() - t0:.1f} s): {truth}", flush=True)
    with get_context("spawn").Pool(args.workers) as pool:
        for lam in todo:
            t_lev = time.time()
            out = new_dir(pdk, rep, lam, smoke)
            out.mkdir(parents=True, exist_ok=True)
            tier = tier_name(rep, lam, smoke)
            from cryoml.pdk2.problem import TRUTH_ROOTS
            with np.load(TRUTH_ROOTS[pdk] / tier / "set_0.npz") as data:
                tags = [str(t) for t in data["tags"]]
            timing = {}
            if not (out / "records.npz").exists():
                t0 = time.time()
                recs = pool.map(collect_task, [(pdk, tier, t) for t in tags], chunksize=1)
                save_devices(out / "records", recs)
                timing["collect_s"] = round(time.time() - t0, 1)
            records = load_devices(out / "records")
            key = lambda r: r.geometry
            groups = sorted({key(r) for r in records})
            hyper = smoke_hyper({}, smoke)
            rows, corr_json, tm, cards = run_lodo(pool, {"M_bias": hyper}, records, key, groups,
                                                  {"M_bias": records},
                                                  [("C0_refit", Correction("C0"))], None)
            timing.update(tm)
            m_bias, c0 = rows["M_bias"], rows["C0_refit"]
            frows = fold_rows("pdk2", records, m_bias, c0)
            for r in frows:
                r["S3"] = float("nan")
            d = describe(m_bias, c0, {r.tag: r.geometry for r in records})
            result = {"prereg": rel(PREREG), "smoke": bool(smoke), "pdk": pdk, "replicate": rep,
                      "lambda": lam, "tier": tier, "amp": AMP, "noise": NOISE,
                      "n_targets": len(records), "n_folds": len(groups),
                      "hyper_overrides": {k: list(v) if isinstance(v, tuple) else v
                                          for k, v in hyper.items()},
                      "summary": dataset_summary(frows), "M_bias": d,
                      "C0_refit": {"per_device": c0},
                      "C0_start": {"per_device": {r.tag: r.meta["rrms_c0"] for r in records}},
                      "per_fold": frows, "timing": timing,
                      "runtime_s": round(time.time() - t_lev, 1)}
            (out / "results.json").write_text(json.dumps(result, indent=1) + "\n")
            (out / "corrections.json").write_text(json.dumps(corr_json) + "\n")
            (out / "heldout_cards.json").write_text(json.dumps(cards) + "\n")
            s = result["summary"]
            print(f"{pdk} r{rep} lambda {lam}: S1_lodo {s['S1_lodo']:+.3f} S2 {s['S2']:+.3f} "
                  f"GMR {s['gmr']:.3f} start {np.nanmean(list(result['C0_start']['per_device'].values())):.4f} "
                  f"C0_refit {np.nanmean(list(c0.values())):.4f} ({result['runtime_s']} s)",
                  flush=True)
    print(f"runtime {time.time() - t_start:.1f} s")
    return 0


def new_names(smoke: bool = False) -> list[tuple[str, int, float]]:
    out = []
    for pdk in ("gf180", "ihp"):
        for rep in (0, 1):
            for lam in (SMOKE["levels"] if smoke else LEVELS):
                if (new_dir(pdk, rep, lam, smoke) / "results.json").exists():
                    out.append((pdk, rep, lam))
    return out


def s3_new(args) -> int:
    t0 = time.time()
    names = [f"new:{p}|{r}|{lam}" for p, r, lam in new_names()]
    per_fold = run_s3(names, args.workers, dict(S3_HYPER))
    (OUT / "new").mkdir(parents=True, exist_ok=True)
    (OUT / "new" / "s3.json").write_text(json.dumps(
        {"prereg": rel(PREREG), "hyper": {k: list(v) if isinstance(v, tuple) else v
                                          for k, v in S3_HYPER.items()},
         "per_fold": per_fold, "runtime_s": round(time.time() - t0, 1)}, indent=1) + "\n")
    print(f"runtime {time.time() - t0:.1f} s")
    return 0


def decide(args) -> int:
    rule = frozen_rule()
    smoke = args.smoke
    names = new_names(smoke)
    s3 = {}
    if (OUT / "new" / "s3.json").exists() and not smoke:
        s3 = json.loads((OUT / "new" / "s3.json").read_text())["per_fold"]
    ds, fold_level, meta = {}, {}, {}
    for pdk, rep, lam in names:
        res = json.loads((new_dir(pdk, rep, lam, smoke) / "results.json").read_text())
        rows = res["per_fold"]
        key = f"{pdk}|{rep}|{lam}"
        for r in rows:
            r["S3"] = s3.get(f"new:{key}", {}).get(r["fold"], float("nan"))
        fold_level[key] = rows
        ds[key] = dataset_summary(rows)
        ds[key]["start_mean"] = float(np.nanmean(list(res["C0_start"]["per_device"].values())))
        ds[key]["c0_refit_mean"] = float(np.nanmean(list(res["C0_refit"]["per_device"].values())))
        meta[key] = {"pdk": pdk, "rep": rep, "lambda": lam}
    valid = {k: v for k, v in ds.items() if v["n_left_out"] <= MAX_LEFT_OUT}
    excluded = sorted(set(ds) - set(valid))
    xs = [v["S1_lodo"] for v in valid.values()]
    ys = [v["y"] for v in valid.values()]
    rho = spearman(xs, ys)
    acc = sign_accuracy(rule, xs, ys)
    correct = acc["correct"]
    p1 = bool(np.isfinite(rho) and rho <= P1_RHO)
    p2 = bool(correct >= P2_MIN)
    verdict = ("supported" if p1 and p2 else "partially supported" if p1 or p2
               else "not supported")
    secondary = {"correlations": correlations(ds, fold_level),
                 "per_pdk_dataset_spearman_S1_lodo": {
                     p: spearman([v["S1_lodo"] for k, v in valid.items() if meta[k]["pdk"] == p],
                                 [v["y"] for k, v in valid.items() if meta[k]["pdk"] == p])
                     for p in ("gf180", "ihp")},
                 "manipulation": {
                     "spearman_lambda_S1_lodo": spearman([meta[k]["lambda"] for k in ds],
                                                         [v["S1_lodo"] for v in ds.values()]),
                     "spearman_lambda_y": spearman([meta[k]["lambda"] for k in ds],
                                                   [v["y"] for v in ds.values()])},
                 "rule_mae": float(np.mean([abs(rule_predict(rule, x) - y)
                                            for x, y in zip(xs, ys)])) if xs else float("nan"),
                 "perm_p_S1_lodo": spearman_perm_p(xs, ys) if len(xs) >= 3 else float("nan")}
    for k in ("S1_pooled", "S2"):
        kx = [v[k] for v in valid.values()]
        secondary[f"P1_P2_with_{k}"] = {"spearman": spearman(kx, ys)}
    secondary["P1_with_S3_spearman_expected_positive"] = spearman(
        [v["S3"] for v in valid.values()], ys)
    out = {"prereg": rel(PREREG), "frozen_rule": rule, "smoke": bool(smoke),
           "n_datasets": len(ds), "excluded": excluded,
           "P1": {"spearman_S1_lodo_vs_y": rho, "threshold": P1_RHO, "pass": p1},
           "P2": {"correct": correct, "n": len(valid) + len(excluded),
                  "required": P2_MIN, "pass": p2},
           "verdict": verdict, "datasets": {k: {**meta[k], **v} for k, v in ds.items()},
           "secondary": secondary}
    target = OUT / ("smoke_decision.json" if smoke else "decision.json")
    target.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("P1", "P2", "verdict", "excluded")}, indent=1))
    for k, v in ds.items():
        print(f"{k:18s} S1_lodo {v['S1_lodo']:+.3f} S2 {v['S2']:+.3f} y {v['y']:+.4f} "
              f"GMR {v['gmr']:.3f} pred {rule_predict(rule, v['S1_lodo']):+.4f}")
    return 0


def figs(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    retro_res = json.loads((OUT / "retro" / "results.json").read_text())
    dec_path = OUT / "decision.json"
    new = json.loads(dec_path.read_text())["datasets"] if dec_path.exists() else {}
    fd = OUT / "figs"
    fd.mkdir(parents=True, exist_ok=True)
    rule = retro_res["rule_S1_lodo_dataset_ols"]
    written = []
    for k in STATS + ("S3",):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for n, v in retro_res["datasets"].items():
            ax.scatter(v[k], v["y"], color="tab:gray", zorder=3)
            ax.annotate(n, (v[k], v["y"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        for n, v in new.items():
            ax.scatter(v[k], v["y"], marker="s", s=24, zorder=3,
                       color="tab:blue" if v["pdk"] == "gf180" else "tab:orange")
        if k == "S1_lodo":
            xs = np.linspace(*ax.get_xlim(), 50)
            ax.plot(xs, rule["a"] + rule["b"] * xs, color="k", lw=1, label="frozen rule")
            ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(f"{k} (training folds, dataset mean)")
        ax.set_ylabel("ln GMR(M_bias, C0_refit)")
        ax.set_title(f"I7-1: {k} vs held-out log gain (gray: stored datasets; "
                     "squares: new, blue GF180, orange IHP)", fontsize=8)
        p = fd / f"scatter_{k}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(rel(p))
    print(json.dumps({"figures": written}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("retro", "s3_retro", "amend", "run",
                                                      "s3_new", "decide", "figs"))
    ap.add_argument("--pdk", choices=("gf180", "ihp"), default="gf180")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    return {"retro": retro, "s3_retro": s3_retro, "amend": amend, "run": run,
            "s3_new": s3_new, "decide": decide, "figs": figs}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
