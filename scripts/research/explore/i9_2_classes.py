#!/usr/bin/env python3
"""I9-2: function-class ablation for the shared bias-only correction g.

Pre-registration draft: ``out/research/explore/I9_2/prereg_draft.json`` (not in
force; ``--mode run`` / ``lc`` refuse without a committed
``out/research/explore/I9_2/prereg.json`` unless ``--smoke``).

Every class is trained on the stored starting-card records of each fold's
training devices (the folds of the stored I3-2 / I5-1 / I6-1 / I8-1 runs) and
the held-out card is refit from its stored starting card with g frozen in the
loop, through the refit function of the dataset's own driver (strict paper
RRMS).  Stored rows are reused: C0_refit, M_bias (= mlp16), T1_inloop, the I4-1
B3_d2 / B3_d3 / B4 rows (poly2 / poly3 / gp) and the I3-2 F2 width-8 / width-32
rows on sky130 77 K.  mlp16_s1 is member 0 of the stored M_bias fold model (no
training, refit only).

Modes:
  plan    load records and stored rows of every dataset (no simulation); fold and
          point counts, stored-row coverage, compute estimate -> I9_2/plan.json
  smoke   sky130 77 K, 2 held-out devices, reduced settings (all 13 classes)
  run     one dataset, pre-registered settings (--dataset)
  decide  pooled statistics and the rule from the run outputs
  lc      learning curve (--dataset tudelft_4K | nist_4K, --classes a,b)

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i9_2_classes.py --mode plan
  PYTHONPATH=src .venv/bin/python scripts/research/explore/i9_2_classes.py --mode smoke --workers 4
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
    CLASS_SPECS, CONTRASTS, MARGIN, REFERENCE, bonferroni_alpha, correction_from_json_i9,
    decide, ensemble_member, estimate_cpu, fit_class, n_params, pooled_ratio,
    select_lc_classes, seeds_of, training_subsets)

EX = ROOT / "out/research/explore"
OUT = EX / "I9_2"
PREREG = OUT / "prereg.json"
OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
SMOKE = {"n_held": 2, "steps": 200, "seeds": (0, 1), "gp_total": 300, "gp_maxiter": 20,
         "heldout_nfev": 60, "heldout_irls": 1}

# ---------------------------------------------------------------------------
# Datasets: fold structure, stored rows, refit path
# ---------------------------------------------------------------------------
# pool: primary = measured, primary temperature, one per independent device set
DATASETS = {
    "sky130_77K": {"kind": "sky130", "pool": "primary"},
    "nist_4K": {"kind": "nist", "pool": "primary"},
    "tudelft_4K": {"kind": "tudelft", "T": 4, "pool": "primary"},
    "ihp_meas_300K": {"kind": "ihp_meas", "T": 300, "pool": "primary"},
    "gf180_c": {"kind": "pdk2", "pdk": "gf180", "tier": "c", "pool": "simulated"},
    "ihp_c": {"kind": "pdk2", "pdk": "ihp", "tier": "c", "pool": "simulated"},
    "tudelft_295K": {"kind": "tudelft", "T": 295, "pool": "other_temperature"},
    "ihp_meas_233K": {"kind": "ihp_meas", "T": 233, "pool": "other_temperature"},
    "ihp_meas_343K": {"kind": "ihp_meas", "T": 343, "pool": "other_temperature"},
    "ihp_meas_398K": {"kind": "ihp_meas", "T": 398, "pool": "other_temperature"},
    "gf180_b": {"kind": "pdk2", "pdk": "gf180", "tier": "b", "pool": "null_control"},
    "ihp_b": {"kind": "pdk2", "pdk": "ihp", "tier": "b", "pool": "null_control"},
}


def _json(p: Path) -> dict:
    return json.loads(p.read_text())


def _nan(d: dict) -> dict:
    return {t: (float("nan") if v is None else float(v)) for t, v in d.items()}


def stored_paths(name: str) -> dict:
    """{label: (results.json, accessor)} of the stored per-device rows and the
    M_bias fold-model file."""
    ds = DATASETS[name]
    k = ds["kind"]
    pd = lambda arm: (lambda r: r["per_device"][arm])
    if k == "sky130":
        i32, i41 = EX / "I3_2/real_dev/results.json", EX / "I4_1/real_dev/results.json"
        f2 = EX / "I3_2/followup/f2_real/results.json"
        return {"rows": {"C0_refit": (i32, pd("C0_refit")), "mlp16": (i32, pd("M_bias")),
                         "T1_inloop": (i32, pd("T1_inloop")), "poly2": (i41, pd("B3_d2")),
                         "poly3": (i41, pd("B3_d3")), "gp": (i41, pd("B4")),
                         "mlp8": (f2, lambda r: r["variants"]["b_width8"]["per_device"]),
                         "mlp32": (f2, lambda r: r["variants"]["b_width32"]["per_device"])},
                "mbias_models": EX / "I3_2/real_dev/corrections.json"}
    if k == "pdk2" and ds["tier"] == "c":
        i32 = EX / f"I3_2/{ds['pdk']}_c_logo/results.json"
        i41 = EX / f"I4_1/{ds['pdk']}_c_logo/results.json"
        return {"rows": {"C0_refit": (i32, pd("C0_refit")), "mlp16": (i32, pd("M_bias")),
                         "T1_inloop": (i32, pd("T1_inloop")), "poly2": (i41, pd("B3_d2")),
                         "poly3": (i41, pd("B3_d3")), "gp": (i41, pd("B4"))},
                "mbias_models": EX / f"I3_2/{ds['pdk']}_c_logo/corrections.json"}
    if k == "pdk2":
        f2 = EX / f"I3_2/followup/f2_{ds['pdk']}_b/results.json"
        return {"rows": {"C0_refit": (f2, lambda r: r["C0_refit"]["per_device"]),
                         "mlp16": (f2, lambda r: r["M_bias"]["per_device"])},
                "mbias_models": EX / f"I3_2/followup/f2_{ds['pdk']}_b/corrections.json"}
    if k == "ihp_meas":
        p = EX / f"I5_1/lodo_{ds['T']}K/results.json"
        return {"rows": {"C0_refit": (p, pd("C0_refit")), "mlp16": (p, pd("M_bias")),
                         "T1_inloop": (p, pd("T1_inloop")), "poly3": (p, pd("B3_d3"))},
                "mbias_models": EX / f"I5_1/lodo_{ds['T']}K/corrections.json"}
    if k == "tudelft":
        p = EX / f"I6_1/lodo_{ds['T']}K/results.json"
        return {"rows": {"C0_refit": (p, pd("C0_refit")), "mlp16": (p, pd("M_bias")),
                         "T1_inloop": (p, pd("T1_inloop")), "poly3": (p, pd("B3_d3"))},
                "mbias_models": EX / f"I6_1/lodo_{ds['T']}K/corrections.json"}
    if k == "nist":
        lo, ba = EX / "I8_1/lodo/results.json", EX / "I8_1/baselines/results.json"
        return {"rows": {"C0_refit": (lo, pd("C0_refit")), "mlp16": (lo, pd("M_bias")),
                         "T1_inloop": (lo, pd("T1_inloop")), "poly3": (ba, pd("B3_d3"))},
                "mbias_models": EX / "I8_1/lodo/corrections.json"}
    raise KeyError(name)


def stored_rows(name: str) -> dict[str, dict]:
    cache, out = {}, {}
    for label, (path, get) in stored_paths(name)["rows"].items():
        if path not in cache:
            cache[path] = _json(path)
        out[label] = _nan(get(cache[path]))
    return out


def load_records(name: str) -> dict:
    """{"recs": finite DEV starting records, "excluded": tags of non-finite
    records (scored as failed), "fold": {tag: fold id}, "train": fold ->
    training tags, "family": {tag: family}, "clusters": {tag: cluster} or None}."""
    ds = DATASETS[name]
    k = ds["kind"]
    excluded = []
    if k == "sky130":
        from i2_1_joint import load_records as lr
        recs, _, _ = lr(argparse.Namespace(mode="real", pdk="gf180"))
        fold = {r.tag: r.tag for r in recs}
        clusters = None
    elif k == "pdk2" and ds["tier"] == "c":
        from i2_1_joint import load_records as lr
        recs, _, _ = lr(argparse.Namespace(mode="pdk2", pdk=ds["pdk"]))
        fold = {r.tag: r.geometry for r in recs}
        clusters = dict(fold)
    elif k == "pdk2":
        rd = EX / f"I3_2/followup/f2_{ds['pdk']}_b/records"
        recs = [r for s in (0, 1, 2) for r in load_devices(rd / f"{ds['pdk']}_b_set{s}")]
        fold = {r.tag: r.geometry for r in recs}
        clusters = dict(fold)
    elif k == "ihp_meas":
        recs = load_devices(EX / f"I5_1/start/records_{ds['T']}K")
        fold = {r.tag: r.tag for r in recs}
        clusters = None
    elif k == "tudelft":
        recs = load_devices(EX / f"I6_1/start/records_{ds['T']}K")
        fold = {r.tag: r.meta["group"] for r in recs}
        clusters = dict(fold)
    elif k == "nist":
        from cryoml.explore.i8_1_nist import split_finite
        recs, excluded = split_finite(load_devices(EX / "I8_1/start/records"))
        fold = {r.tag: r.tag for r in recs}
        clusters = None
    else:
        raise KeyError(name)
    by = {r.tag: r for r in recs}
    folds = sorted(set(fold.values()))
    # training-record order as in the stored runs (the loss sums points in this
    # order): load order, except I8-1 (sorted by tag)
    order = sorted(by) if k == "nist" else [r.tag for r in recs]
    if k == "ihp_meas":        # I5-1: training within class (LV / HV)
        train = {f: [t for t in order if t != f and by[t].meta["cls"] == by[f].meta["cls"]]
                 for f in folds}
    else:
        train = {f: [t for t in order if fold[t] != f] for f in folds}
    return {"recs": recs, "by_tag": by, "excluded": excluded, "fold": fold, "folds": folds,
            "train": train, "family": {r.tag: r.family for r in recs}, "clusters": clusters}


def stored_mbias(name: str, fold_id: str):
    from cryoml.explore.i3_2_bias import correction_from_json
    models = _json(stored_paths(name)["mbias_models"])
    return correction_from_json(models[f"M_bias|{fold_id}"])


def load_devices_for(name: str, tags: list[str]) -> dict:
    """Raw measured devices (and NIST bin info) needed by the refit path."""
    ds = DATASETS[name]
    if ds["kind"] in ("sky130", "pdk2"):
        return {t: (None, None) for t in tags}
    if ds["kind"] == "ihp_meas":
        import i5_1_ihp_real as drv
        devs = {d.tag: d for d in drv.dev_devices()}
        return {t: (devs[t], None) for t in tags}
    if ds["kind"] == "tudelft":
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
    seconds, n_points).  seed: one member of a 5-seed MLP class."""
    import torch
    torch.set_num_threads(1)
    key, cls, recs, ov, seed = task
    t0 = time.time()
    if seed is not None:
        from cryoml.explore.i3_2_bias import fit_bias
        corr = fit_bias("M_bias", recs, hyper={**({"steps": ov["steps"]} if "steps" in ov else {}),
                                               "hidden": CLASS_SPECS[cls]["hidden"],
                                               "seeds": (seed,)}, threads=1)
        corr.arm = cls
    else:
        corr = fit_class(cls, recs, ov, threads=1)
    return key, corr, round(time.time() - t0, 3), int(len(make_batch(recs).sim))


def refit_task(task):
    """Held-out refit with g frozen, through the dataset driver's refit path."""
    name, label, rec, dev, binfo, corr, opts = task
    kind = DATASETS[name]["kind"]
    if kind in ("sky130", "pdk2"):
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
    return row


def merge_members(parts: list):
    """Merge 1-member MLP corrections of one (class, fold) in seed order."""
    from cryoml.explore.i3_2_bias import BiasCorrection
    first = parts[0]
    for p in parts[1:]:
        if not (np.array_equal(p.mu, first.mu) and np.array_equal(p.sd, first.sd)):
            raise RuntimeError("members do not share feature scaling")
    return BiasCorrection(first.arm, first.mu, first.sd,
                          nets=[n for p in parts for n in p.nets], cols=first.cols)


# ---------------------------------------------------------------------------
# One dataset (run / smoke)
# ---------------------------------------------------------------------------
def classes_to_run(name: str, stored: dict, smoke: bool) -> tuple[list, list]:
    """(classes to train, classes taken from the stored M_bias ensemble)."""
    if smoke:
        return [c for c in CLASS_SPECS if c != "mlp16_s1"], ["mlp16_s1"]
    new = [c for c in CLASS_SPECS if c not in stored and c != "mlp16_s1"]
    return new, ["mlp16_s1"]


def run_dataset(args, name: str) -> int:
    t_start = time.time()
    smoke = bool(args.smoke)
    data = load_records(name)
    stored = stored_rows(name)
    ov = {} if not smoke else {k: SMOKE[k] for k in ("steps", "seeds", "gp_total", "gp_maxiter")}
    opts = dict(OPTS)
    if smoke:
        opts.update(heldout_nfev=SMOKE["heldout_nfev"], heldout_irls=SMOKE["heldout_irls"])
    folds = data["folds"]
    if smoke:     # one nMOS and one pMOS held-out device (sky130 LODO)
        fam = data["family"]
        folds = [next(f for f in folds if fam[f] == "nmos"), next(f for f in folds if fam[f] == "pmos")]
    held = [t for t in sorted(data["by_tag"]) if data["fold"][t] in folds]
    train_cls, member_cls = classes_to_run(name, stored, smoke)
    print(f"I9_2 {name}: {len(folds)} folds, {len(held)} held-out, train {train_cls}, "
          f"from stored M_bias {member_cls}, opts {opts}, overrides {ov}", flush=True)
    by = data["by_tag"]
    tasks = []
    for f in folds:
        recs = [by[t] for t in data["train"][f]]
        for c in train_cls:
            if CLASS_SPECS[c]["kind"] == "mlp" and seeds_of(c) > 1:
                for s in ov.get("seeds", CLASS_SPECS[c]["seeds"]):
                    tasks.append((f"{c}|{f}|{s}", c, recs, ov, s))
            else:
                tasks.append((f"{c}|{f}", c, recs, ov, None))
    heavy = {"gp": 0, "mlp64": 1, "mlp32": 2}
    tasks.sort(key=lambda t: heavy.get(t[1], 9))
    devs = load_devices_for(name, held)
    timing, train_rows = {}, {}
    with get_context("spawn").Pool(args.workers) as pool:
        t0 = time.time()
        out = pool.map(train_task, tasks, chunksize=1)
        timing["train_s"] = round(time.time() - t0, 1)
        timing["train_n"] = len(tasks)
        parts: dict = {}
        for key, corr, sec, npts in out:
            cls, f = key.split("|")[:2]
            parts.setdefault((cls, f), []).append((key, corr))
            train_rows[key] = {"class": cls, "fold": f, "seconds": sec, "n_points": npts,
                               "steps": ov.get("steps", 2000)}
        corrs = {}
        for (cls, f), lst in parts.items():
            lst.sort(key=lambda kc: int(kc[0].split("|")[2]) if kc[0].count("|") == 2 else 0)
            corrs[f"{cls}|{f}"] = merge_members([c for _, c in lst]) if len(lst) > 1 else lst[0][1]
        for f in folds:
            for c in member_cls:
                src = corrs.get(f"mlp16|{f}") if smoke else stored_mbias(name, f)
                corrs[f"{c}|{f}"] = ensemble_member(src, 0, arm=c)
        refits = []
        for t in held:
            f = data["fold"][t]
            for c in train_cls + member_cls:
                refits.append((name, c, by[t], *devs[t], corrs[f"{c}|{f}"], opts))
        if not smoke:   # reproduction check: a stored class rerun on 2 held-out devices
            rc = "poly3" if "poly3" in stored else "C0_refit"
            for t in held[:2]:
                corr = Correction("C0") if rc == "C0_refit" else None
                if corr is None:
                    from cryoml.explore.i4_1_baselines import fit_poly
                    corr = fit_poly("poly3", 3, [by[x] for x in data["train"][data["fold"][t]]])
                refits.append((name, f"repro_{rc}", by[t], *devs[t], corr, opts))
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
    repro = {}
    for lab in [x for x in scores if x.startswith("repro_")]:
        ref = lab[len("repro_"):]
        repro[lab] = float(max(abs(scores[lab][t] - stored[ref][t]) for t in rows[lab]))
    result = {"dataset": name, "smoke": smoke, "prereg": str(PREREG.relative_to(ROOT)),
              "options": opts, "overrides": ov, "held_out": held, "folds": folds,
              "excluded_records": data["excluded"], "trained_classes": train_cls,
              "member_classes": member_cls, "per_device": scores, "reproduction": repro,
              "n_params": {c: n_params(c) for c in CLASS_SPECS},
              "train_rows": train_rows,
              "refit_rows": {lab: {t: {k: x for k, x in r.items() if k not in ("params", "history")}
                                   for t, r in v.items()} for lab, v in rows.items()},
              "timing": timing, "runtime_s": round(time.time() - t_start, 1)}
    target = OUT / ("smoke" if smoke else name)
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    (target / "corrections.json").write_text(json.dumps({k: c.to_json() for k, c in corrs.items()}) + "\n")
    (target / "cards.json").write_text(json.dumps(
        {lab: {t: r["params"] for t, r in v.items()} for lab, v in rows.items()}) + "\n")
    for lab in sorted(scores):
        v = [scores[lab][t] for t in held]
        print(f"  {lab:16s} " + " ".join(f"{x:.5f}" for x in v), flush=True)
    print(f"timing {timing} runtime {result['runtime_s']} s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Plan (no simulation)
# ---------------------------------------------------------------------------
REFIT_S_KEYS = {"sky130": ("I3_2/real_dev", "M_bias"), "nist": ("I8_1/lodo", "M_bias")}


def stored_refit_seconds(name: str) -> dict:
    """Mean single-process seconds of a corrected held-out refit (stored
    M_bias rows) and of a GP refit (stored I4-1 B4 rows, else None)."""
    ds = DATASETS[name]
    k = ds["kind"]
    if k == "sky130":
        p, q = EX / "I3_2/real_dev/results.json", EX / "I4_1/real_dev/results.json"
    elif k == "pdk2" and ds["tier"] == "c":
        p, q = EX / f"I3_2/{ds['pdk']}_c_logo/results.json", EX / f"I4_1/{ds['pdk']}_c_logo/results.json"
    elif k == "pdk2":
        p, q = EX / f"I3_2/{ds['pdk']}_c_logo/results.json", None     # tier b: tier c timing
    elif k == "ihp_meas":
        p, q = EX / f"I5_1/lodo_{ds['T']}K/results.json", None
    elif k == "tudelft":
        p, q = EX / f"I6_1/lodo_{ds['T']}K/results.json", None
    else:
        p, q = EX / "I8_1/lodo/results.json", None
    r = _json(p)
    rows = r.get("heldout_rows") or r.get("rows")
    corr = float(np.mean([x["runtime_s"] for x in rows["M_bias"].values()]))
    gp = None
    if q is not None:
        rq = _json(q)["heldout_rows"]
        gp = float(np.mean([x["runtime_s"] for x in rq["B4"].values()]))
    return {"corrected_refit_s": round(corr, 2), "gp_refit_s": None if gp is None else round(gp, 2)}


def run_plan(args) -> int:
    smoke = OUT / "smoke/results.json"
    cost = {}
    if smoke.exists():       # seconds per (training point x step x member) from the smoke
        sm = _json(smoke)
        for key, r in sm["train_rows"].items():
            c = r["class"]
            if CLASS_SPECS[c]["kind"] == "gp":
                continue
            per = r["seconds"] / (r["n_points"] * r["steps"])
            cost.setdefault(c, []).append(per)
        cost = {c: float(np.median(v)) for c, v in cost.items()}
    gp_train_s = 30.0        # per fold, 2000 points, 200 L-BFGS-B iterations (I4-1 timing)
    plan = {"datasets": {}, "per_point_step_s": cost, "gp_train_s_per_fold": gp_train_s,
            "workers": args.workers}
    tot = {"cpu_s": 0.0}
    for name in DATASETS:
        data = load_records(name)
        stored = stored_rows(name)
        held = sorted(data["by_tag"])
        cover = {lab: (sum(np.isfinite(v.get(t, np.nan)) for t in held), len(held))
                 for lab, v in stored.items()}
        npts = {f: int(len(make_batch([data["by_tag"][t] for t in data["train"][f]]).sim))
                for f in data["folds"]}
        new, member = classes_to_run(name, stored, False)
        timing = stored_refit_seconds(name)
        gp_refit = timing["gp_refit_s"] or timing["corrected_refit_s"] + 30.0
        n_refit = {c: len(held) for c in new + member}
        refit_cpu = sum(len(held) * (gp_refit if c == "gp" else timing["corrected_refit_s"])
                        for c in new + member) + 2 * timing["corrected_refit_s"]
        train_cpu = 0.0
        per_class_train = {}
        for c in new:
            if CLASS_SPECS[c]["kind"] == "gp":
                s = gp_train_s * len(data["folds"])
            elif c in cost:
                s = sum(cost[c] * npts[f] * 2000 * seeds_of(c) for f in data["folds"])
            else:
                s = float("nan")
            per_class_train[c] = round(s, 1)
            train_cpu += s
        est = estimate_cpu(0, 0.0, 1, refit_cpu + train_cpu, workers=args.workers)
        tot["cpu_s"] += est["cpu_s"]
        plan["datasets"][name] = {
            **DATASETS[name], "n_records": len(data["recs"]) + len(data["excluded"]),
            "n_finite": len(data["recs"]), "excluded": data["excluded"],
            "n_folds": len(data["folds"]), "n_heldout": len(held),
            "train_points_per_fold": {"min": min(npts.values()), "max": max(npts.values()),
                                      "mean": round(float(np.mean(list(npts.values()))), 1)},
            "stored_rows_finite_of_heldout": {k: f"{a}/{b}" for k, (a, b) in cover.items()},
            "new_classes_trained": new, "classes_from_stored_M_bias": member,
            "n_new_refits": sum(n_refit.values()) + 2, "refit_timing": timing,
            "refit_cpu_s": round(refit_cpu, 1), "train_cpu_s": round(train_cpu, 1),
            "train_cpu_s_per_class": per_class_train, "estimate": est}
        print(f"{name:14s} folds {len(data['folds']):3d} held {len(held):3d} new {len(new):2d} "
              f"refits {sum(n_refit.values()) + 2:4d} refit {refit_cpu / 3600:.2f} CPU-h train "
              f"{train_cpu / 3600:.2f} CPU-h wall {est['wall_min']:.0f} min", flush=True)
    plan["total"] = estimate_cpu(0, 0.0, 1, tot["cpu_s"], workers=args.workers)
    # learning curve, two classes, n in (2, 4, 8) x 3 subsets
    lc = {}
    for name in ("tudelft_4K", "nist_4K"):
        d = plan["datasets"][name]
        t = stored_refit_seconds(name)["corrected_refit_s"]
        n_ref = d["n_heldout"] * 3 * 3 * 2
        lc[name] = {"refits": n_ref, "trainings_per_class": d["n_folds"] * 9,
                    "estimate_refits_only": estimate_cpu(n_ref, t, 0, 0.0, workers=args.workers)}
    plan["learning_curve"] = lc
    print(json.dumps({"total": plan["total"], "learning_curve": lc}, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------
# Decide (pooled statistics)
# ---------------------------------------------------------------------------
def run_decide(args) -> int:
    res = {n: _json(OUT / n / "results.json") for n in DATASETS if (OUT / n / "results.json").exists()}
    prim = [n for n in DATASETS if DATASETS[n]["pool"] == "primary"]
    missing = [n for n in prim if n not in res]
    if missing:
        raise SystemExit(f"primary datasets missing: {missing}")
    a_adj = bonferroni_alpha()
    clusters = {}
    for n in res:
        d = load_records(n)
        if d["clusters"] is not None:
            clusters[n] = d["clusters"]
    sc = {n: {lab: _nan(v) for lab, v in r["per_device"].items()} for n, r in res.items()}

    def pool_of(names, method, control, alpha, cl=None):
        return pooled_ratio({n: (sc[n][method], sc[n][control]) for n in names}, clusters=cl,
                            alpha=alpha)
    primary = {c: pool_of(prim, c, REFERENCE, a_adj) for c in CONTRASTS}
    fails = {c: {n: int(sum(~np.isfinite(list(sc[n][c].values())))
                        - sum(~np.isfinite(list(sc[n][REFERENCE].values())))) for n in prim}
             for c in CONTRASTS}
    rule = decide(primary, fails)
    vs_c0 = {c: pool_of(prim, c, "C0_refit", 0.05) for c in CLASS_SPECS}
    sim = [n for n in res if DATASETS[n]["pool"] in ("primary", "simulated")]
    secondary = {
        "all_six_vs_M_bias": {c: pool_of(sim, c, REFERENCE, a_adj) for c in CONTRASTS},
        "per_dataset_vs_M_bias": {n: {c: pool_of([n], c, REFERENCE, 0.05) for c in CONTRASTS}
                                  for n in res},
        "per_dataset_vs_C0_refit": {n: {c: pool_of([n], c, "C0_refit", 0.05) for c in CLASS_SPECS}
                                    for n in res},
        "primary_cluster_bootstrap": {c: pool_of(prim, c, REFERENCE, a_adj, clusters)
                                      for c in CONTRASTS},
        "margin_sensitivity": {m: decide(primary, fails, margin=m)["labels"] for m in (1.02, 1.05)},
    }
    out = {"prereg": str(PREREG.relative_to(ROOT)), "primary_datasets": prim,
           "alpha_adjusted": a_adj, "margin": MARGIN, "primary_vs_M_bias": primary,
           "rule": rule, "primary_vs_C0_refit": vs_c0,
           "learning_curve_classes": select_lc_classes(vs_c0), "secondary": secondary}
    (OUT / "decision.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"rule": rule, "learning_curve_classes": out["learning_curve_classes"]}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------
def run_lc(args) -> int:
    name = args.dataset
    if name not in ("tudelft_4K", "nist_4K"):
        raise SystemExit("learning curve datasets: tudelft_4K, nist_4K")
    classes = [c for c in args.classes.split(",") if c] or \
        _json(OUT / "decision.json")["learning_curve_classes"]
    data = load_records(name)
    by = data["by_tag"]
    held = sorted(by)
    tasks, meta = [], {}
    for fi, f in enumerate(data["folds"]):
        pool = {t: data["family"][t] for t in data["train"][f]}
        for n in (2, 4, 8):
            for r, sub in enumerate(training_subsets(pool, n, key=(fi,))):
                for c in classes:
                    key = f"{c}|{f}|n{n}|r{r}"
                    meta[key] = sub
                    tasks.append((key, c, [by[t] for t in sub], {}, None))
    devs = load_devices_for(name, held)
    with get_context("spawn").Pool(args.workers) as pool_:
        out = pool_.map(train_task, tasks, chunksize=1)
        corrs = {k: c for k, c, _, _ in out}
        refits = []
        for t in held:
            f = data["fold"][t]
            for key in [k for k in corrs if k.split("|")[1] == f]:
                c, _, n, r = key.split("|")
                refits.append((name, f"{c}|{n}|{r}", by[t], *devs[t], corrs[key], dict(OPTS)))
        rows = pool_.map(refit_task, refits, chunksize=1)
    scores = {}
    for row in rows:
        scores.setdefault(row["label"], {})[row["tag"]] = row["rrms"]
    target = OUT / f"lc_{name}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(json.dumps(
        {"dataset": name, "classes": classes, "subsets": meta, "per_device": scores}, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------
def require_prereg() -> None:
    """The run and lc modes need prereg.json committed and unmodified."""
    rel = str(PREREG.relative_to(ROOT))
    ok = PREREG.exists() and subprocess.run(
        ["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]).returncode == 0 and \
        subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                       capture_output=True).returncode == 0
    if not ok:
        raise SystemExit(f"{rel} is not committed: the pre-registration is not in force")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("plan", "smoke", "run", "decide", "lc"), required=True)
    ap.add_argument("--dataset", choices=tuple(DATASETS), default="sky130_77K")
    ap.add_argument("--classes", default="", help="lc: override the selected classes")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.smoke = args.mode == "smoke"
    if args.mode == "plan":
        return run_plan(args)
    if args.mode == "smoke":
        return run_dataset(args, "sky130_77K")
    if args.mode == "decide":
        return run_decide(args)
    require_prereg()
    if args.mode == "run":
        return run_dataset(args, args.dataset)
    return run_lc(args)


if __name__ == "__main__":
    sys.exit(main())
