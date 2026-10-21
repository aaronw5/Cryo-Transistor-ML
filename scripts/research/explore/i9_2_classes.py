#!/usr/bin/env python3
"""I9-2: function-class ablation for the shared bias-only correction g.

Pre-registration: ``out/research/explore/I9_2/prereg.json`` (``validity``,
``run``, ``replicate``, ``lc`` refuse unless it is committed and unmodified;
``run``, ``replicate`` and ``lc`` also need ``validity.json`` with the dataset
passing V1).  Primary pool only: sky130 77 K, NIST 4 K, TU Delft 4.13 K, IHP
measured 300 K.

Every class is trained on the stored starting-card records of each fold's
training devices (folds of the stored I3-2 / I5-1 / I6-1 / I8-1 runs) and the
held-out card is refit from its stored starting card with g frozen in the
loop, through the refit function of the dataset's own driver (strict paper
RRMS).  Stored rows are reused: C0_refit, M_bias (= mlp16), T1_inloop, the I4-1
B3_d2 / B3_d3 / B4 rows (poly2 / poly3 / gp), and on sky130 77 K the I3-2 F2
width-8 / width-32 / seeds 5-9 rows (mlp8 / mlp32 / mlp16_rep).  mlp16_s1 is
member 0 of the stored M_bias fold model (refit only).

Modes:
  plan       records, folds, stored rows, compute estimate (no simulation) -> plan.json
  smoke      sky130 77 K, 2 held-out devices, reduced settings, all classes
  validity   V1 rerun of poly3 on 2 held-out devices per dataset vs the stored rows;
             training loss of the stored corrections -> validity.json
             (--smoke: 1 device per dataset, reduced settings, adapter check only)
  run        one dataset (--dataset), the classes without stored rows + mlp16_s1
  replicate  mlp16_rep (M_bias recipe, seeds 5-9) on the 4 datasets
  decide     pooled statistics and the rule -> decision.json
  lc         learning curve (--dataset tudelft_4K | nist_4K)
  figs       forest plot, capacity plot, learning curve, training-loss table

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i9_2_classes.py --mode run \
      --dataset nist_4K --workers 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cryoml.explore.e1_correction import Correction, load_devices, make_batch  # noqa: E402
from cryoml.explore.i9_2_classes import (  # noqa: E402
    CLASS_SPECS, CONTRASTS, LC_SIZES, MARGIN, REFERENCE, REPLICATE, bonferroni_alpha,
    correction_from_json_i9, decide, ensemble_member, estimate_cpu, fit_class, lc_classes,
    n_params, n_params_member, pooled_ratio, seeds_of, training_loss, training_subsets)

EX = ROOT / "out/research/explore"
OUT = EX / "I9_2"
PREREG = OUT / "prereg.json"
OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
SMOKE = {"n_held": 2, "steps": 200, "seeds": (0, 1), "gp_total": 300, "gp_maxiter": 20,
         "heldout_nfev": 60, "heldout_irls": 1}
SMOKE_VALIDITY = {"steps": 50, "heldout_nfev": 10, "heldout_irls": 1}
V1_TOL = 1e-9

DATASETS = {
    "sky130_77K": {"kind": "sky130"},
    "nist_4K": {"kind": "nist"},
    "tudelft_4K": {"kind": "tudelft", "T": 4},
    "ihp_meas_300K": {"kind": "ihp_meas", "T": 300},
}
PRIMARY = tuple(DATASETS)


def _json(p: Path) -> dict:
    return json.loads(p.read_text())


def _nan(d: dict) -> dict:
    return {t: (float("nan") if v is None else float(v)) for t, v in d.items()}


# ---------------------------------------------------------------------------
# Stored rows and fold models
# ---------------------------------------------------------------------------
def stored_paths(name: str) -> dict:
    """rows: {label: (results.json, accessor)}; models: {label: (corrections.json,
    key prefix)} (fold model key '<prefix>|<fold>')."""
    k = DATASETS[name]["kind"]
    pd = lambda arm: (lambda r: r["per_device"][arm])
    if k == "sky130":
        i32, i41 = EX / "I3_2/real_dev", EX / "I4_1/real_dev"
        f2 = EX / "I3_2/followup/f2_real"
        var = lambda v: (lambda r: r["variants"][v]["per_device"])
        return {"rows": {"C0_refit": (i32 / "results.json", pd("C0_refit")),
                         "mlp16": (i32 / "results.json", pd("M_bias")),
                         "T1_inloop": (i32 / "results.json", pd("T1_inloop")),
                         "poly2": (i41 / "results.json", pd("B3_d2")),
                         "poly3": (i41 / "results.json", pd("B3_d3")),
                         "gp": (i41 / "results.json", pd("B4")),
                         "mlp8": (f2 / "results.json", var("b_width8")),
                         "mlp32": (f2 / "results.json", var("b_width32")),
                         "mlp16_rep": (f2 / "results.json", var("a_seeds_5_9"))},
                "models": {"mlp16": (i32 / "corrections.json", "M_bias"),
                           "poly2": (i41 / "corrections.json", "B3_d2"),
                           "poly3": (i41 / "corrections.json", "B3_d3"),
                           "gp": (i41 / "corrections.json", "B4"),
                           "mlp8": (f2 / "corrections.json", "b_width8"),
                           "mlp32": (f2 / "corrections.json", "b_width32"),
                           "mlp16_rep": (f2 / "corrections.json", "a_seeds_5_9")}}
    if k == "nist":
        lo, ba = EX / "I8_1/lodo", EX / "I8_1/baselines"
        return {"rows": {"C0_refit": (lo / "results.json", pd("C0_refit")),
                         "mlp16": (lo / "results.json", pd("M_bias")),
                         "T1_inloop": (lo / "results.json", pd("T1_inloop")),
                         "poly3": (ba / "results.json", pd("B3_d3"))},
                "models": {"mlp16": (lo / "corrections.json", "M_bias"),
                           "poly3": (ba / "corrections.json", "B3_d3")}}
    sub = EX / (f"I6_1/lodo_{DATASETS[name]['T']}K" if k == "tudelft"
                else f"I5_1/lodo_{DATASETS[name]['T']}K")
    return {"rows": {"C0_refit": (sub / "results.json", pd("C0_refit")),
                     "mlp16": (sub / "results.json", pd("M_bias")),
                     "T1_inloop": (sub / "results.json", pd("T1_inloop")),
                     "poly3": (sub / "results.json", pd("B3_d3"))},
            "models": {"mlp16": (sub / "corrections.json", "M_bias"),
                       "poly3": (sub / "corrections.json", "B3_d3")}}


def stored_rows(name: str) -> dict[str, dict]:
    cache, out = {}, {}
    for label, (path, get) in stored_paths(name)["rows"].items():
        if path not in cache:
            cache[path] = _json(path)
        out[label] = _nan(get(cache[path]))
    return out


_MODELS: dict = {}


def stored_model(name: str, label: str, fold_id: str):
    path, prefix = stored_paths(name)["models"][label]
    if path not in _MODELS:
        _MODELS[path] = _json(path)
    return correction_from_json_i9(_MODELS[path][f"{prefix}|{fold_id}"])


def load_records(name: str) -> dict:
    """{"recs": finite DEV starting records, "excluded": tags of non-finite
    records (scored as failed), "fold": {tag: fold id}, "folds", "train":
    fold -> training tags in the stored training order, "family", "clusters"}."""
    k = DATASETS[name]["kind"]
    excluded = []
    if k == "sky130":
        from i2_1_joint import load_records as lr
        recs, _, _ = lr(argparse.Namespace(mode="real", pdk="gf180"))
        fold = {r.tag: r.tag for r in recs}
        clusters = None
    elif k == "ihp_meas":
        recs = load_devices(EX / f"I5_1/start/records_{DATASETS[name]['T']}K")
        fold = {r.tag: r.tag for r in recs}
        clusters = None
    elif k == "tudelft":
        recs = load_devices(EX / f"I6_1/start/records_{DATASETS[name]['T']}K")
        fold = {r.tag: r.meta["group"] for r in recs}
        clusters = dict(fold)
    else:
        from cryoml.explore.i8_1_nist import split_finite
        recs, excluded = split_finite(load_devices(EX / "I8_1/start/records"))
        fold = {r.tag: r.tag for r in recs}
        clusters = None
    by = {r.tag: r for r in recs}
    folds = sorted(set(fold.values()))
    # stored training order (the loss sums points in this order): load order,
    # except I8-1 (sorted by tag)
    order = sorted(by) if k == "nist" else [r.tag for r in recs]
    if k == "ihp_meas":        # I5-1: training within class (LV / HV)
        train = {f: [t for t in order if t != f and by[t].meta["cls"] == by[f].meta["cls"]]
                 for f in folds}
    else:
        train = {f: [t for t in order if fold[t] != f] for f in folds}
    return {"recs": recs, "by_tag": by, "excluded": excluded, "fold": fold, "folds": folds,
            "train": train, "family": {r.tag: r.family for r in recs}, "clusters": clusters}


def load_devices_for(name: str, tags: list[str]) -> dict:
    """Raw measured devices (and NIST bin info) needed by the refit path."""
    k = DATASETS[name]["kind"]
    if k == "sky130":
        return {t: (None, None) for t in tags}
    if k == "ihp_meas":
        import i5_1_ihp_real as drv
        devs = {d.tag: d for d in drv.dev_devices()}
        return {t: (devs[t], None) for t in tags}
    if k == "tudelft":
        import i6_1_tudelft as drv
        devs = {d.tag: d for d in drv.dev_devices(False)}
        return {t: (devs[t], None) for t in tags}
    import i8_1_nist as drv
    devs = {d.tag: d for d in drv.dev_devices()[0]}
    bins = _json(EX / "I8_1/start/bins.json")["devices"]
    return {t: (devs[t], bins[t]) for t in tags}


# ---------------------------------------------------------------------------
# Worker tasks
# ---------------------------------------------------------------------------
def train_task(task):
    """(key, class, training records, overrides, seed or None) -> (key, corr,
    seconds, n_points, trace).  seed: one member of a multi-seed MLP class."""
    import torch
    torch.set_num_threads(1)
    key, cls, recs, ov, seed = task
    t0 = time.time()
    trace: list = []
    if seed is not None:
        from cryoml.explore.i3_2_bias import fit_bias
        hyper = {"hidden": CLASS_SPECS[cls]["hidden"], "seeds": (seed,)}
        if "steps" in ov:
            hyper["steps"] = ov["steps"]
        corr = fit_bias("M_bias", recs, hyper=hyper, threads=1, trace=trace)
        corr.arm = cls
    else:
        corr = fit_class(cls, recs, ov, threads=1, trace=trace)
    return key, corr, round(time.time() - t0, 3), int(len(make_batch(recs).sim)), trace


def loss_task(task):
    """(key, corr, training records) -> (key, E1 training loss without penalty)."""
    import torch
    torch.set_num_threads(1)
    key, corr, recs = task
    return key, training_loss(corr, recs)


def refit_task(task):
    """Held-out refit with g frozen, through the dataset driver's refit path."""
    name, label, rec, dev, binfo, corr, opts = task
    kind = DATASETS[name]["kind"]
    if kind == "sky130":
        import i4_1_baselines as drv
        row = drv.refit_task((label, rec, {"kind": "corr", "corr": corr}, opts))
    elif kind == "ihp_meas":
        import i5_1_ihp_real as drv
        row = drv.refit_task((label, "corr", {"corr": corr}, rec, dev, opts))
    elif kind == "tudelft":
        import i6_1_tudelft as drv
        row = drv.refit_task((label, "corr", {"corr": corr}, rec, dev, opts))
    else:
        import i8_1_nist as drv
        row = drv.refit_task((label, "corr", {"corr": corr}, rec, dev, binfo, opts))
    row["rrms"] = float(row["rrms"])
    row["dataset"] = name
    return row


def merge_members(parts: list):
    """Merge 1-member MLP corrections of one (class, fold) in seed order."""
    from cryoml.explore.i3_2_bias import BiasCorrection
    first = parts[0]
    if not all(isinstance(p, BiasCorrection) for p in parts):
        raise RuntimeError("merge_members: only multi-seed MLP members can be merged")
    for p in parts[1:]:
        if not (np.array_equal(p.mu, first.mu) and np.array_equal(p.sd, first.sd)):
            raise RuntimeError("members do not share feature scaling")
    return BiasCorrection(first.arm, first.mu, first.sd,
                          nets=[n for p in parts for n in p.nets], cols=first.cols)


def train_tasks_for(cls: str, key_base: str, recs: list, ov: dict) -> list:
    if CLASS_SPECS[cls]["kind"] == "mlp" and seeds_of(cls) > 1:
        seeds = CLASS_SPECS[cls]["seeds"] if cls == REPLICATE else ov.get("seeds", CLASS_SPECS[cls]["seeds"])
        return [(f"{key_base}|seed{s}", cls, recs, ov, s) for s in seeds]
    return [(key_base, cls, recs, ov, None)]


HEAVY = {"gp": 0, "mlp32": 1, "mlp16": 2, "mlp16_rep": 2, "mlp8": 3}


def train_all(pool, tasks: list, recs_of: dict) -> tuple[dict, dict]:
    """Run training tasks, merge members, compute the training loss of every
    merged correction.  ``recs_of``: key base -> training records."""
    tasks = sorted(tasks, key=lambda t: HEAVY.get(t[1], 9))
    out = pool.map(train_task, tasks, chunksize=1)
    parts, rows = {}, {}
    for key, corr, sec, npts, trace in out:
        base = key.rsplit("|seed", 1)[0] if "|seed" in key else key
        parts.setdefault(base, []).append((key, corr))
        rows[key] = {"seconds": sec, "n_points": npts, "trace": trace}
    corrs = {}
    for base, lst in parts.items():
        lst.sort(key=lambda kc: int(kc[0].rsplit("|seed", 1)[1]) if "|seed" in kc[0] else 0)
        corrs[base] = merge_members([c for _, c in lst]) if len(lst) > 1 else lst[0][1]
    losses = dict(pool.map(loss_task, [(b, c, recs_of[b]) for b, c in corrs.items()], chunksize=1))
    return corrs, {"tasks": rows, "final_training_loss": losses}


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def require_prereg() -> dict:
    """prereg.json committed, unmodified and an ancestor of HEAD."""
    rel = str(PREREG.relative_to(ROOT))
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]).returncode == 0
    if not (PREREG.exists() and tracked and clean):
        raise SystemExit(f"{rel} is not committed: the pre-registration is not in force")
    sha = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    return {"prereg": rel, "commit": sha}


def require_validity(names) -> None:
    p = OUT / "validity.json"
    if not p.exists():
        raise SystemExit("validity.json missing: run --mode validity first")
    v = _json(p)
    bad = [n for n in names if not v["datasets"].get(n, {}).get("V1_pass")]
    if bad:
        raise SystemExit(f"V1 failed or missing for {bad}; these datasets are excluded")


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------
def run_validity(args) -> int:
    t_start = time.time()
    smoke = bool(args.smoke)
    guard = None if smoke else require_prereg()
    ov = {"steps": SMOKE_VALIDITY["steps"]} if smoke else {}
    opts = dict(OPTS)
    if smoke:
        opts.update(heldout_nfev=SMOKE_VALIDITY["heldout_nfev"],
                    heldout_irls=SMOKE_VALIDITY["heldout_irls"])
    n_dev = 1 if smoke else 2
    result = {"guard": guard, "smoke": smoke, "options": opts, "overrides": ov,
              "tolerance": V1_TOL, "datasets": {}}
    with get_context("spawn").Pool(args.workers) as pool:
        tasks, refit_specs, recs_of, data_of, loss_specs = [], [], {}, {}, []
        for name in PRIMARY:
            data = load_records(name)
            data_of[name] = data
            held = sorted(data["by_tag"])[:n_dev]
            devs = load_devices_for(name, held)
            for t in held:
                f = data["fold"][t]
                base = f"{name}|poly3|{f}"
                if base not in recs_of:  # held devices can share a fold (TU Delft groups)
                    recs_of[base] = [data["by_tag"][x] for x in data["train"][f]]
                    tasks += train_tasks_for("poly3", base, recs_of[base], ov)
                refit_specs.append((name, base, data["by_tag"][t], devs[t]))
            folds = data["folds"][:1] if smoke else data["folds"]
            for label in stored_paths(name)["models"]:
                for f in folds:
                    loss_specs.append((f"{name}|{label}|{f}", stored_model(name, label, f),
                                       [data["by_tag"][x] for x in data["train"][f]]))
        corrs, train_info = train_all(pool, tasks, recs_of)
        refits = [(name, "V1_poly3", rec, *dv, corrs[base], opts)
                  for name, base, rec, dv in refit_specs]
        rows = pool.map(refit_task, refits, chunksize=1)
        stored_losses = dict(pool.map(loss_task, loss_specs, chunksize=1))
    for name in PRIMARY:
        stored = stored_rows(name)
        mine = [r for r in rows if r["dataset"] == name]
        diffs = {r["tag"]: abs(r["rrms"] - stored["poly3"][r["tag"]]) for r in mine}
        coef = {}
        for name_, base, rec, _ in refit_specs:
            if name_ != name:
                continue
            f = base.split("|")[2]
            st = stored_model(name, "poly3", f)
            coef[f] = float(np.max(np.abs(corrs[base].coef - st.coef)))
        mx = float(max(diffs.values()))
        result["datasets"][name] = {
            "V1_rerun_rrms": {r["tag"]: r["rrms"] for r in mine},
            "V1_stored_rrms": {t: stored["poly3"][t] for t in diffs},
            "V1_abs_diff": diffs, "V1_max_abs_diff": mx,
            "V1_max_abs_coef_diff_vs_stored_model": coef,
            "V1_pass": bool(mx <= V1_TOL) and not smoke,
            "stored_rows_finite_of_heldout": {
                lab: f"{sum(np.isfinite(v.get(t, np.nan)) for t in data_of[name]['by_tag'])}/"
                     f"{len(data_of[name]['by_tag'])}" for lab, v in stored.items()},
            "stored_training_loss": {k.split("|", 1)[1]: v for k, v in stored_losses.items()
                                     if k.startswith(name + "|")}}
        print(f"{name:14s} V1 max |diff| {mx:.3e} pass {result['datasets'][name]['V1_pass']}",
              flush=True)
    result["train"] = train_info
    result["runtime_s"] = round(time.time() - t_start, 1)
    target = OUT / ("smoke_validity.json" if smoke else "validity.json")
    target.write_text(json.dumps(result, indent=1) + "\n")
    print(f"runtime {result['runtime_s']} s")
    return 0


# ---------------------------------------------------------------------------
# run / smoke / replicate
# ---------------------------------------------------------------------------
def classes_to_run(name: str, stored: dict, smoke: bool) -> tuple[list, list]:
    """(classes to train, classes taken from the stored M_bias ensemble)."""
    skip = {"mlp16_s1", REPLICATE}
    if smoke:
        return [c for c in CLASS_SPECS if c not in skip], ["mlp16_s1"]
    return [c for c in CLASS_SPECS if c not in stored and c not in skip], ["mlp16_s1"]


def held_and_folds(data: dict, smoke: bool) -> tuple[list, list]:
    folds = data["folds"]
    if smoke:     # one nMOS and one pMOS held-out device (sky130 LODO)
        fam = data["family"]
        folds = [next(f for f in folds if fam[f] == "nmos"), next(f for f in folds if fam[f] == "pmos")]
    return [t for t in sorted(data["by_tag"]) if data["fold"][t] in folds], folds


def write_outputs(target: Path, result: dict, corrs: dict, rows: dict) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    (target / "corrections.json").write_text(json.dumps({k: c.to_json() for k, c in corrs.items()}) + "\n")
    (target / "cards.json").write_text(json.dumps(
        {lab: {t: r["params"] for t, r in v.items()} for lab, v in rows.items()}) + "\n")


def run_dataset(args, name: str, guard: dict | None) -> int:
    t_start = time.time()
    smoke = bool(args.smoke)
    data = load_records(name)
    stored = stored_rows(name)
    ov = {} if not smoke else {k: SMOKE[k] for k in ("steps", "seeds", "gp_total", "gp_maxiter")}
    opts = dict(OPTS)
    if smoke:
        opts.update(heldout_nfev=SMOKE["heldout_nfev"], heldout_irls=SMOKE["heldout_irls"])
    held, folds = held_and_folds(data, smoke)
    train_cls, member_cls = classes_to_run(name, stored, smoke)
    print(f"I9_2 {name}: {len(folds)} folds, {len(held)} held-out, train {train_cls}, "
          f"from stored M_bias {member_cls}, opts {opts}, overrides {ov}", flush=True)
    by = data["by_tag"]
    tasks, recs_of = [], {}
    for f in folds:
        recs = [by[t] for t in data["train"][f]]
        for c in train_cls:
            recs_of[f"{c}|{f}"] = recs
            tasks += train_tasks_for(c, f"{c}|{f}", recs, ov)
    devs = load_devices_for(name, held)
    timing = {}
    with get_context("spawn").Pool(args.workers) as pool:
        t0 = time.time()
        corrs, train_info = train_all(pool, tasks, recs_of)
        timing["train_s"] = round(time.time() - t0, 1)
        timing["train_n"] = len(tasks)
        for f in folds:
            for c in member_cls:
                src = corrs.get(f"mlp16|{f}") if smoke else stored_model(name, "mlp16", f)
                corrs[f"{c}|{f}"] = ensemble_member(src, 0, arm=c)
        refits = [(name, c, by[t], *devs[t], corrs[f"{c}|{data['fold'][t]}"], opts)
                  for t in held for c in train_cls + member_cls]
        refits.sort(key=lambda r: 0 if r[1] == "gp" else 1)
        t0 = time.time()
        rows_l = pool.map(refit_task, refits, chunksize=1)
        timing["refit_s"] = round(time.time() - t0, 1)
        timing["refit_n"] = len(refits)
    rows = {}
    for row in rows_l:
        rows.setdefault(row["label"], {})[row["tag"]] = row
    scores = {lab: {t: r["rrms"] for t, r in v.items()} for lab, v in rows.items()}
    if not smoke:
        for lab, v in stored.items():
            scores.setdefault(lab, {t: v[t] for t in held})
    for lab in scores:
        for t in data["excluded"]:
            scores[lab][t] = float("nan")
    result = {"dataset": name, "smoke": smoke, "guard": guard, "options": opts, "overrides": ov,
              "held_out": held, "folds": folds, "excluded_records": data["excluded"],
              "trained_classes": train_cls, "member_classes": member_cls,
              "stored_classes": sorted(set(stored) - {"C0_refit", "T1_inloop"}),
              "per_device": scores,
              "n_params": {c: n_params(c) for c in CLASS_SPECS},
              "n_params_per_member": {c: n_params_member(c) for c in CLASS_SPECS},
              "train": train_info,
              "refit_rows": {lab: {t: {k: x for k, x in r.items() if k not in ("params", "history")}
                                   for t, r in v.items()} for lab, v in rows.items()},
              "timing": timing, "runtime_s": round(time.time() - t_start, 1)}
    write_outputs(OUT / ("smoke" if smoke else name), result, corrs, rows)
    for lab in sorted(scores):
        v = [scores[lab][t] for t in held]
        print(f"  {lab:16s} n {len(v):3d} mean {np.nanmean(v):.5f} failed {int(np.sum(~np.isfinite(v)))}",
              flush=True)
    print(f"timing {timing} runtime {result['runtime_s']} s", flush=True)
    return 0


def run_replicate(args, guard: dict) -> int:
    """mlp16_rep (seeds 5-9) on every primary dataset; stored F2 rows on sky130."""
    t_start = time.time()
    out = {"guard": guard, "options": OPTS, "class": REPLICATE,
           "seeds": list(CLASS_SPECS[REPLICATE]["seeds"]), "datasets": {}}
    all_corrs, all_rows = {}, {}
    with get_context("spawn").Pool(args.workers) as pool:
        for name in PRIMARY:
            t0 = time.time()
            data = load_records(name)
            stored = stored_rows(name)
            held = sorted(data["by_tag"])
            if REPLICATE in stored:
                scores = {t: stored[REPLICATE][t] for t in held}
                out["datasets"][name] = {"source": "stored (I3-2 F2 a_seeds_5_9)", "per_device": scores}
                continue
            by = data["by_tag"]
            tasks, recs_of = [], {}
            for f in data["folds"]:
                recs_of[f"{REPLICATE}|{f}"] = [by[t] for t in data["train"][f]]
                tasks += train_tasks_for(REPLICATE, f"{REPLICATE}|{f}", recs_of[f"{REPLICATE}|{f}"], {})
            corrs, info = train_all(pool, tasks, recs_of)
            devs = load_devices_for(name, held)
            refits = [(name, REPLICATE, by[t], *devs[t], corrs[f"{REPLICATE}|{data['fold'][t]}"], OPTS)
                      for t in held]
            rows = pool.map(refit_task, refits, chunksize=1)
            scores = {r["tag"]: r["rrms"] for r in rows}
            for t in data["excluded"]:
                scores[t] = float("nan")
            out["datasets"][name] = {"source": "new", "per_device": scores, "train": info,
                                     "runtime_s": round(time.time() - t0, 1)}
            all_corrs.update({f"{name}|{k}": c for k, c in corrs.items()})
            all_rows[name] = {r["tag"]: r for r in rows}
            print(f"{name}: replicate mean {np.nanmean(list(scores.values())):.5f} "
                  f"({out['datasets'][name]['runtime_s']} s)", flush=True)
    out["runtime_s"] = round(time.time() - t_start, 1)
    write_outputs(OUT / "replicate", out, all_corrs, all_rows)
    return 0


# ---------------------------------------------------------------------------
# Plan (no simulation)
# ---------------------------------------------------------------------------
def stored_refit_seconds(name: str) -> dict:
    """Mean single-process seconds of a stored corrected held-out refit
    (M_bias rows) and of a GP refit (I4-1 B4 rows on sky130, else corrected + 30 s)."""
    k = DATASETS[name]["kind"]
    p = {"sky130": EX / "I3_2/real_dev/results.json", "nist": EX / "I8_1/lodo/results.json",
         "tudelft": EX / f"I6_1/lodo_{DATASETS[name].get('T')}K/results.json",
         "ihp_meas": EX / f"I5_1/lodo_{DATASETS[name].get('T')}K/results.json"}[k]
    r = _json(p)
    rows = r.get("heldout_rows") or r.get("rows")
    corr = float(np.mean([x["runtime_s"] for x in rows["M_bias"].values()]))
    gp = corr + 30.0
    if k == "sky130":
        rq = _json(EX / "I4_1/real_dev/results.json")["heldout_rows"]
        gp = float(np.mean([x["runtime_s"] for x in rq["B4"].values()]))
    return {"corrected_refit_s": round(corr, 2), "gp_refit_s": round(gp, 2)}


GP_TRAIN_S = 30.0       # per fold, 2000 points, 200 L-BFGS-B iterations (I4-1 timing)


def train_seconds(cost: dict, cls: str, points: float) -> float:
    if CLASS_SPECS[cls]["kind"] == "gp":
        return max(1.0, GP_TRAIN_S * (min(points, 2000) / 2000) ** 3)
    return cost[cls] * points * 2000 * seeds_of(cls)


def run_plan(args) -> int:
    cost = dict(_json(OUT / "cost_basis.json")["per_point_step_member_s"])
    cost[REPLICATE] = cost["mlp16"]
    plan = {"per_point_step_member_s": cost, "gp_train_s_per_fold": GP_TRAIN_S,
            "workers": args.workers, "datasets": {}, "jobs": {}}
    total = 0.0
    val_cpu = 0.0
    rep_cpu = 0.0
    for name in PRIMARY:
        data = load_records(name)
        stored = stored_rows(name)
        held = sorted(data["by_tag"])
        npts = {f: int(len(make_batch([data["by_tag"][t] for t in data["train"][f]]).sim))
                for f in data["folds"]}
        new, member = classes_to_run(name, stored, False)
        tm = stored_refit_seconds(name)
        refit = sum(len(held) * (tm["gp_refit_s"] if c == "gp" else tm["corrected_refit_s"])
                    for c in new + member)
        by_cls = {c: sum(train_seconds(cost, c, npts[f]) for f in data["folds"]) for c in new}
        train = sum(by_cls.values())
        est = estimate_cpu(0, 0.0, 1, refit + train, workers=args.workers)
        total += est["cpu_s"]
        loss_eval = 0.5 * len(data["folds"]) * (len(new) + len(stored_paths(name)["models"]))
        val_cpu += 2 * tm["corrected_refit_s"] + 2 * train_seconds(cost, "poly3", np.mean(list(npts.values()))) \
            + 0.5 * len(data["folds"]) * len(stored_paths(name)["models"])
        if REPLICATE not in stored:
            rep_cpu += len(held) * tm["corrected_refit_s"] + \
                sum(train_seconds(cost, REPLICATE, npts[f]) for f in data["folds"])
        plan["datasets"][name] = {
            "n_records": len(data["recs"]) + len(data["excluded"]), "n_finite": len(data["recs"]),
            "excluded": data["excluded"], "n_folds": len(data["folds"]), "n_heldout": len(held),
            "train_points_per_fold": {"min": min(npts.values()), "max": max(npts.values()),
                                      "mean": round(float(np.mean(list(npts.values()))), 1)},
            "stored_classes": sorted(set(stored) - {"C0_refit", "T1_inloop"}),
            "new_classes_trained": new, "classes_from_stored_M_bias": member,
            "n_new_refits": len(held) * len(new + member), "refit_timing": tm,
            "refit_cpu_h": round(refit / 3600, 2), "train_cpu_h": round(train / 3600, 2),
            "train_cpu_h_by_class": {c: round(v / 3600, 2) for c, v in by_cls.items()},
            "loss_eval_cpu_s": round(loss_eval, 1), "estimate": est}
        print(f"{name:14s} folds {len(data['folds']):3d} held {len(held):3d} new {new} refit "
              f"{refit / 3600:.2f} CPU-h train {train / 3600:.2f} CPU-h wall {est['wall_min']:.0f} min",
              flush=True)
    plan["jobs"]["validity"] = estimate_cpu(0, 0.0, 1, val_cpu, workers=args.workers)
    plan["jobs"]["replicate"] = estimate_cpu(0, 0.0, 1, rep_cpu, workers=args.workers)
    plan["jobs"]["run_all_4"] = estimate_cpu(0, 0.0, 1, total, workers=args.workers)
    # learning curve: mlp16 + 2 classes; low = 2 cheap classes, high = mlp32 + gp
    lc = {}
    for name in ("tudelft_4K", "nist_4K"):
        d = plan["datasets"][name]
        data = load_records(name)
        ppd = d["train_points_per_fold"]["mean"] / max(1, len(data["train"][data["folds"][0]]))
        tm = stored_refit_seconds(name)
        n_tr = d["n_folds"] * len(LC_SIZES) * 3
        n_ref = d["n_heldout"] * len(LC_SIZES) * 3
        tr = lambda c: sum(d["n_folds"] * 3 * train_seconds(cost, c, ppd * n) for n in LC_SIZES)
        low = n_ref * 3 * tm["corrected_refit_s"] + tr("mlp16") + 2 * tr("poly3")
        high = n_ref * 2 * tm["corrected_refit_s"] + n_ref * tm["gp_refit_s"] + tr("mlp16") + tr("mlp32") + tr("gp")
        lc[name] = {"refits_per_class": n_ref, "trainings_per_class": n_tr,
                    "low_cheap_classes": estimate_cpu(0, 0.0, 1, low, workers=args.workers),
                    "high_mlp32_gp": estimate_cpu(0, 0.0, 1, high, workers=args.workers)}
    plan["jobs"]["learning_curve"] = lc
    plan["total_without_lc"] = estimate_cpu(0, 0.0, 1, total + val_cpu + rep_cpu, workers=args.workers)
    print(json.dumps(plan["jobs"], indent=1), json.dumps(plan["total_without_lc"]))
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------
# Decide
# ---------------------------------------------------------------------------
def collect_scores(out_dir: Path) -> dict:
    res = {n: _json(out_dir / n / "results.json") for n in PRIMARY}
    rep = _json(out_dir / "replicate/results.json")
    sc = {n: {lab: _nan(v) for lab, v in r["per_device"].items()} for n, r in res.items()}
    for n in PRIMARY:
        sc[n][REPLICATE] = _nan(rep["datasets"][n]["per_device"])
    return sc


def run_decide(args) -> int:
    out_dir = Path(args.out_dir)
    sc = collect_scores(out_dir)
    a_adj = bonferroni_alpha()
    clusters = {n: load_records(n)["clusters"] for n in PRIMARY}
    clusters = {n: c for n, c in clusters.items() if c is not None}

    def pool_of(names, method, control, alpha, cl=None):
        return pooled_ratio({n: (sc[n][method], sc[n][control]) for n in names}, clusters=cl,
                            alpha=alpha)
    replicate = pool_of(PRIMARY, REPLICATE, REFERENCE, a_adj)
    replicate95 = pool_of(PRIMARY, REPLICATE, REFERENCE, 0.05)
    primary = {c: pool_of(PRIMARY, c, REFERENCE, a_adj) for c in CONTRASTS}
    nf = lambda n, c: int(np.sum(~np.isfinite(list(sc[n][c].values()))))
    fails = {c: {n: nf(n, c) - nf(n, REFERENCE) for n in PRIMARY} for c in CONTRASTS}
    rule = decide(primary, fails, replicate=replicate)
    vs_c0 = {c: pool_of(PRIMARY, c, "C0_refit", 0.05) for c in CLASS_SPECS}
    per_ds = {n: {c: pool_of([n], c, REFERENCE, 0.05) for c in list(CONTRASTS) + [REPLICATE]}
              for n in PRIMARY}
    gain = {}
    for n in PRIMARY:
        ref = pool_of([n], REFERENCE, "C0_refit", 0.05)
        if ref["ci_high"] < 1.0:
            gain[n] = {c: float(np.log(pool_of([n], c, "C0_refit", 0.05)["ratio"]) / np.log(ref["ratio"]))
                       for c in CLASS_SPECS}
    losses = {}
    for n in PRIMARY:
        r = _json(out_dir / n / "results.json")
        for k, v in r["train"]["final_training_loss"].items():
            losses.setdefault(n, {}).setdefault(k.split("|")[0], []).append(v)
    out = {"prereg": str(PREREG.relative_to(ROOT)), "primary_datasets": list(PRIMARY),
           "alpha_adjusted": a_adj, "margin": MARGIN,
           "replicate_vs_mlp16": {"bonferroni": replicate, "ci95": replicate95},
           "primary_vs_mlp16": primary, "rule": rule,
           "primary_vs_C0_refit": vs_c0, "learning_curve_classes": lc_classes(vs_c0),
           "secondary": {
               "per_dataset_vs_mlp16_with_replicate": per_ds,
               "per_dataset_vs_C0_refit": {n: {c: pool_of([n], c, "C0_refit", 0.05) for c in CLASS_SPECS}
                                           for n in PRIMARY},
               "primary_cluster_bootstrap": {c: pool_of(PRIMARY, c, REFERENCE, a_adj, clusters)
                                             for c in CONTRASTS},
               "margin_sensitivity": {str(m): decide(primary, fails, margin=m, replicate=replicate)["labels"]
                                      for m in (1.02, 1.05)},
               "fraction_of_mlp16_log_gain": gain,
               "final_training_loss_mean_over_folds": {n: {c: float(np.mean(v)) for c, v in d.items()}
                                                       for n, d in losses.items()}}}
    (out_dir / "decision.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"rule": {k: rule[k] for k in ("labels", "notes", "Q1", "Q2_smallest_matching_class",
                                                    "Q2_beats_with_fewer_parameters")},
                      "replicate": {k: replicate[k] for k in ("ratio", "ci_low", "ci_high")},
                      "learning_curve_classes": out["learning_curve_classes"]}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------
def _safe_ratio(a: dict, b: dict) -> dict:
    try:
        return pooled_ratio({"d": (a, b)})
    except ValueError:
        return {"ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}


def lc_summary(scores: dict, main: dict, classes: list) -> dict:
    """Per class and n: per held-out device exp(mean_r ln RRMS); GMR vs
    C0_refit and vs the class at n = all (95% device bootstrap)."""
    out = {}
    for c in classes:
        out[c] = {}
        for n in LC_SIZES:
            labs = [k for k in scores if k.startswith(f"{c}|n{n}|")]
            tags = sorted(set().union(*[scores[k] for k in labs])) if labs else []
            dev = {}
            for t in tags:
                v = np.array([scores[k].get(t, np.nan) for k in labs], float)
                dev[t] = float(np.exp(np.mean(np.log(v)))) if np.all(np.isfinite(v) & (v > 0)) else float("nan")
            out[c][f"n{n}"] = {"vs_C0_refit": _safe_ratio(dev, main["C0_refit"]),
                               "vs_all": _safe_ratio(dev, main[c]), "per_device": dev}
        out[c]["all"] = {"vs_C0_refit": _safe_ratio(main[c], main["C0_refit"])}
    return out


def run_lc(args, guard: dict) -> int:
    t_start = time.time()
    name = args.dataset
    if name not in ("tudelft_4K", "nist_4K"):
        raise SystemExit("learning curve datasets: tudelft_4K, nist_4K")
    classes = [c for c in args.classes.split(",") if c] or \
        _json(OUT / "decision.json")["learning_curve_classes"]
    data = load_records(name)
    by = data["by_tag"]
    held = sorted(by)
    tasks, recs_of, subsets = [], {}, {}
    for fi, f in enumerate(data["folds"]):
        pool = {t: data["family"][t] for t in data["train"][f]}
        for n in LC_SIZES:
            for r, sub in enumerate(training_subsets(pool, n, key=(fi,))):
                for c in classes:
                    base = f"{c}|n{n}|r{r}|{f}"
                    subsets[base] = sub
                    recs_of[base] = [by[t] for t in sub]
                    tasks += train_tasks_for(c, base, recs_of[base], {})
    devs = load_devices_for(name, held)
    with get_context("spawn").Pool(args.workers) as pool_:
        corrs, info = train_all(pool_, tasks, recs_of)
        refits = []
        for t in held:
            f = data["fold"][t]
            for base in [k for k in corrs if k.rsplit("|", 1)[1] == f]:
                c, n, r, _ = base.split("|")
                refits.append((name, f"{c}|{n}|{r}", by[t], *devs[t], corrs[base], dict(OPTS)))
        rows = pool_.map(refit_task, refits, chunksize=1)
    scores = {}
    for row in rows:
        scores.setdefault(row["label"], {})[row["tag"]] = row["rrms"]
    main = {lab: _nan(v) for lab, v in _json(OUT / name / "results.json")["per_device"].items()}
    result = {"dataset": name, "guard": guard, "classes": classes, "sizes": list(LC_SIZES),
              "subsets": subsets, "per_device": scores, "summary": lc_summary(scores, main, classes),
              "train": info, "runtime_s": round(time.time() - t_start, 1)}
    target = OUT / f"lc_{name}"
    write_outputs(target, result, corrs, {})
    print(f"runtime {result['runtime_s']} s")
    return 0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def run_figs(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(args.out_dir)
    dec = _json(out_dir / "decision.json")
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(CONTRASTS, key=n_params)
    # forest: pooled (Bonferroni) and per dataset (95%) GMR(X, mlp16)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(order) + 1.5))
    rep = dec["replicate_vs_mlp16"]["bonferroni"]
    ax.axvspan(rep["ci_low"], rep["ci_high"], color="0.85", label="mlp16_rep CI (seeds 5-9)")
    ax.axvline(1.0, color="k", lw=0.8)
    for x in (1 / MARGIN, MARGIN):
        ax.axvline(x, color="k", lw=0.6, ls=":")
    marks = {"sky130_77K": "o", "nist_4K": "s", "tudelft_4K": "^", "ihp_meas_300K": "D"}
    per = dec["secondary"]["per_dataset_vs_mlp16_with_replicate"]
    for i, c in enumerate(order):
        p = dec["primary_vs_mlp16"][c]
        ax.errorbar(p["ratio"], i, xerr=[[p["ratio"] - p["ci_low"]], [p["ci_high"] - p["ratio"]]],
                    fmt="o", color="C0", ms=6, capsize=3, label="pooled (Bonferroni CI)" if i == 0 else None)
        for j, n in enumerate(PRIMARY):
            ax.plot(per[n][c]["ratio"], i + 0.18 * (j - 1.5) / 1.5 * 0.9, marks[n], color="C1", ms=3.5,
                    alpha=0.7, label=n if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c} ({n_params(c)} / {n_params_member(c)})" for c in order])
    ax.set_xlabel("GMR(class, mlp16 = M_bias) of held-out RRMS")
    ax.set_title("I9-2 primary pool: class vs M_bias")
    ax.set_ylabel("class (trained values: total / per member)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "forest_vs_mlp16.png", dpi=150)
    plt.close(fig)
    # capacity: pooled GMR vs C0_refit against trained values
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in CLASS_SPECS:
        p = dec["primary_vs_C0_refit"][c]
        x = n_params(c)
        ax.errorbar(x, p["ratio"], yerr=[[p["ratio"] - p["ci_low"]], [p["ci_high"] - p["ratio"]]],
                    fmt="o", color="C3" if c in (REFERENCE, REPLICATE) else "C0", capsize=3)
        ax.annotate(c, (x, p["ratio"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("trained values in g (total over members)")
    ax.set_ylabel("pooled GMR(class, C0_refit), 95% CI")
    ax.set_title("I9-2 primary pool: held-out gain against class size")
    fig.tight_layout()
    fig.savefig(fig_dir / "capacity_vs_C0_refit.png", dpi=150)
    plt.close(fig)
    # learning curves
    for name in ("tudelft_4K", "nist_4K"):
        p = out_dir / f"lc_{name}/results.json"
        if not p.exists():
            continue
        lc = _json(p)
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in lc["classes"]:
            s = lc["summary"][c]
            xs = list(LC_SIZES) + [len(load_records(name)["train"][load_records(name)["folds"][0]])]
            ys = [s[f"n{n}"]["vs_C0_refit"] for n in LC_SIZES] + [s["all"]["vs_C0_refit"]]
            ax.errorbar(xs, [y["ratio"] for y in ys],
                        yerr=[[y["ratio"] - y["ci_low"] for y in ys], [y["ci_high"] - y["ratio"] for y in ys]],
                        fmt="o-", capsize=3, label=c)
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("training devices per fold")
        ax.set_ylabel("GMR(class, C0_refit), 95% CI")
        ax.set_title(f"I9-2 learning curve, {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"learning_curve_{name}.png", dpi=150)
        plt.close(fig)
    # training-loss table
    tl = dec["secondary"]["final_training_loss_mean_over_folds"]
    lines = ["| class | " + " | ".join(PRIMARY) + " |", "|---" * (len(PRIMARY) + 1) + "|"]
    for c in sorted({c for d in tl.values() for c in d}, key=lambda c: n_params(c)):
        lines.append(f"| {c} | " + " | ".join(f"{tl[n].get(c, float('nan')):.5f}" for n in PRIMARY) + " |")
    (fig_dir / "training_loss.md").write_text("\n".join(lines) + "\n")
    print(f"figures in {fig_dir}")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("plan", "smoke", "validity", "run", "replicate", "decide",
                                       "lc", "figs"), required=True)
    ap.add_argument("--dataset", choices=PRIMARY, default="sky130_77K")
    ap.add_argument("--classes", default="", help="lc: override the selected classes")
    ap.add_argument("--smoke", action="store_true", help="validity: reduced adapter check")
    ap.add_argument("--out-dir", default=str(OUT), help="decide / figs input and output directory")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.mode == "smoke":
        args.smoke = True
        return run_dataset(args, "sky130_77K", None)
    if args.mode == "plan":
        return run_plan(args)
    if args.mode == "validity":
        return run_validity(args)
    if args.mode == "decide":
        return run_decide(args)
    if args.mode == "figs":
        return run_figs(args)
    guard = require_prereg()
    if args.mode == "run":
        require_validity([args.dataset])
        return run_dataset(args, args.dataset, guard)
    require_validity(PRIMARY)
    if args.mode == "replicate":
        return run_replicate(args, guard)
    return run_lc(args, guard)


if __name__ == "__main__":
    sys.exit(main())
