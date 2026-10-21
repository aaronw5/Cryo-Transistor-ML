#!/usr/bin/env python3
"""I11-2: shared bias-only correction with a temperature input, held-out temperature.

Pre-registration: ``out/research/explore/I11_2/prereg.json`` (and ``.md``).
Module: ``cryoml.explore.i11_2_temp``.

Data: the stored DEV starting-card records of I5-1 (IHP SG13G2, 18 DUTs at
233 / 300 / 343 / 398 K) and I6-1 (TU Delft 40 nm, 45 devices in 7 geometry
groups at 4.13 / 295.0 K).  TEST devices are not read.

Modes:
  plan               folds, training-set sizes, stored rows and corrections, finite
                     records, leakage checks (no simulation, no training)
  smoke              reduced run of every path (2 IHP folds at 233 / 343 K, 2 TU Delft
                     devices, 1 seed, 200 steps, refit nfev 60), writes smoke/
  validity           V1 stored C0_refit rows reproduced, V5 stored fold corrections
                     reproduced by retraining (full recipe)
  stats              S1_loto, S3_loto per IHP fold -> prediction.json
  commit_prediction  commit prediction.json alone and push
  lodo_ihp           IHP arms M_bias_T, M_bias_pool, M_bias_near, M_bias_in, lin_T, lin_pool
  lodo_tud           TU Delft arms M_bias_X (= pool = near), M_bias_in
  decide             pre-registered rule, secondary tables -> decision.json

``stats``, ``lodo_*`` and ``decide`` refuse outside --smoke unless
prereg.json is tracked, unmodified, in an ancestor of HEAD and not DRAFT;
``lodo_*`` also require the committed prediction.json.

  PYTHONPATH=src .venv/bin/python -I scripts/research/explore/i11_2_temp.py --mode plan
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.explore import i11_2_temp as k  # noqa: E402

OUT = ROOT / "out/research/explore/I11_2"
PREREG = OUT / "prereg.json"
PREDICTION = OUT / "prediction.json"
I5 = ROOT / "out/research/explore/I5_1"
I6 = ROOT / "out/research/explore/I6_1"
IHP_TEMPS = (233, 300, 343, 398)
TUD_TEMPS = (4, 295)
INTERP = (300, 343)
EXTRAP = (233, 398)
OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
SMOKE = {"seeds": (0,), "steps": 200, "s3_steps": 50, "nfev": 60, "irls": 1,
         "ihp_temps": (233, 343)}
IHP_ARMS = ("M_bias_T", "M_bias_pool", "M_bias_near", "M_bias_in", "lin_T", "lin_pool")
TUD_ARMS = ("M_bias_X", "M_bias_in")
MAX_FAILED = 2
REPRO_TOL = 1e-9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=1, default=default, allow_nan=True) + "\n")


def odir(args) -> Path:
    return OUT / "smoke" if args.smoke else OUT


def key(tag: str, t: int) -> str:
    return f"{tag}|{int(t)}"


def unkey(s: str) -> tuple[str, int]:
    tag, t = s.rsplit("|", 1)
    return tag, int(t)


def guard(path: Path) -> dict:
    from cryoml.explore.i8_1_nist import prediction_guard
    g = prediction_guard(path)
    if g["ok"] and path == PREREG:
        g["status"] = json.loads(path.read_text()).get("status")
        g["ok"] = g["status"] != "DRAFT"
    return g


def require(args, *paths) -> dict:
    if args.smoke:
        return {"smoke": True}
    out = {}
    for p in paths:
        g = guard(p)
        out[p.name] = g
        if not g["ok"]:
            raise SystemExit(f"refused: {p.relative_to(ROOT)} not in force ({g})")
    return out


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def ihp_module():
    from cryoml.explore import i5_1_ihp_real as m5
    return m5


def tud_module():
    from cryoml.explore import i6_1_tudelft as m6
    return m6


def load_ihp(tags=None):
    from cryoml.explore.e1_correction import load_devices
    m5 = ihp_module()
    split = json.loads((I5 / "split.json").read_text())
    dev_tags = sorted(split["DEV"])
    devices = {d.tag: d for d in m5.load_dataset(I5 / "data/ihp_meas.npz",
                                                 json.loads((I5 / "data/manifest.json").read_text()),
                                                 dev_tags)}
    recs = {t: load_devices(I5 / f"start/records_{t}K") for t in IHP_TEMPS}
    for t, rs in recs.items():
        if sorted(r.tag for r in rs) != dev_tags:
            raise RuntimeError(f"IHP {t} K records differ from DEV")
    return devices, recs


def load_tud():
    from cryoml.explore.e1_correction import load_devices
    m6 = tud_module()
    split = json.loads((I6 / "split.json").read_text())
    dev_tags = sorted(split["DEV"])
    devices = {d.tag: d for d in m6.load_dataset(I6 / "data/tudelft40.npz",
                                                 json.loads((I6 / "data/manifest.json").read_text()),
                                                 dev_tags)}
    recs = {t: load_devices(I6 / f"start/records_{t}K") for t in TUD_TEMPS}
    for t, rs in recs.items():
        if sorted(r.tag for r in rs) != dev_tags:
            raise RuntimeError(f"TU Delft {t} K records differ from DEV")
    return devices, recs


def ihp_unit(rec) -> str:
    return rec.tag


def tud_unit(rec) -> str:
    return rec.meta["group"]


def ihp_eligible(held_rec):
    cls = held_rec.meta["cls"]
    return lambda r: r.meta["cls"] == cls


def stored_rows(base: Path, temps, arms=("C0_refit", "M_bias")) -> dict:
    """{arm: {tag|T: RRMS}} from stored lodo results."""
    out = {}
    for t in temps:
        per = json.loads((base / f"lodo_{t}K/results.json").read_text())["per_device"]
        for a in arms:
            for tag, v in per[a].items():
                out.setdefault(a, {})[key(tag, t)] = float(v)
    return out


def stored_x300() -> dict:
    per = json.loads((I5 / "cross/results.json").read_text())["per_temperature"]
    return {key(tag, int(t)): float(v) for t, x in per.items() for tag, v in x["per_device"]["X300"].items()}


def stored_corrections(base: Path, t: int) -> dict:
    from cryoml.explore.i4_1_baselines import correction_from_json_any
    rows = json.loads((base / f"lodo_{t}K/corrections.json").read_text())
    return {kk.split("|", 1)[1]: correction_from_json_any(v) for kk, v in rows.items()
            if kk.startswith("M_bias|")}


# ---------------------------------------------------------------------------
# worker tasks
# ---------------------------------------------------------------------------
def train_task(task):
    kind, arm, recs, hyper = task
    if kind == "mlp_t":
        return k.fit_bias_t(arm, recs, hyper=hyper, threads=1)
    if kind == "mlp":
        from cryoml.explore.i3_2_bias import fit_bias
        corr = fit_bias("M_bias", recs, hyper=hyper, threads=1)
        corr.arm = arm
        return corr
    if kind == "lin_t":
        return k.fit_lin_t(arm, recs, hyper=hyper, threads=1)
    if kind == "lin":
        from cryoml.explore.i4_1_baselines import fit_poly
        return fit_poly(arm, 1, recs, hyper=hyper, threads=1)
    raise KeyError(kind)


def refit_task(task):
    """Held-out refit from the device's starting card at the held-out
    temperature with ``corr`` frozen in the loop (I5-1 / I6-1 refit_task,
    kind 'corr')."""
    from cryoml.explore.e1_correction import score_device
    from cryoml.ngspice_fit import polish
    ds, label, corr, rec, dev, opts = task
    m = ihp_module() if ds == "ihp" else tud_module()
    t0 = time.time()
    n_to0 = getattr(m, "N_TIMEOUT", 0)
    t = int(rec.meta["T_K"])
    start = dict(rec.meta["params"])
    box = m.base_box(dev)
    prob = m.MeasProblem(dev, t, box)
    res = polish(m.corrected_problem(prob, corr), box.params_to_local(start),
                 max_nfev=opts["heldout_nfev"], irls_rounds=opts["heldout_irls"],
                 diff_step=opts["diff_step"])
    params = box.local_to_params(res["local"])
    new = m.to_record(dev, t, prob.simulate_params(params), params)
    after = score_device(new, corr.apply(new))
    row = {"label": label, "tag": rec.tag, "T_K": t, "rrms": after["rrms"],
           "failed": after["failed"], "rrms_start": float(score_device(rec, corr.apply(rec))["rrms"]),
           "n_sim": prob.n_sim, "history": res["history"], "params": params,
           "n_sim_timeout": getattr(m, "N_TIMEOUT", 0) - n_to0,
           "runtime_s": round(time.time() - t0, 2)}
    print(f"  {ds} {label:12s} {rec.tag:28s} {t:4d} K {row['rrms']:.4f} ({row['runtime_s']} s)",
          flush=True)
    return row


def s3_task(task):
    """One inner S3 fold: train (reduced recipe), post-hoc log gain on the
    scored records (T arm evaluated at each scored record's temperature)."""
    from cryoml.explore.i7_1_transfer import s3_fold_value
    fold, arm, inner, scored, hyper = task
    if arm == "M_bias_T":
        corr = k.fit_bias_t(arm, inner, hyper=hyper, threads=1)
        vals = [s3_fold_value(corr.bind(k.temp_kelvin(r)), [r])[0] for r in scored]
    else:
        from cryoml.explore.i3_2_bias import fit_bias
        corr = fit_bias("M_bias", inner, hyper=hyper, threads=1)
        vals = s3_fold_value(corr, scored)
    return fold, arm, vals


def run_pool(workers, fn, tasks):
    if workers <= 1:
        return [fn(t) for t in tasks]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=1)


def train_all(workers, specs: dict, hyper: dict) -> dict:
    """``specs``: name -> (kind, arm, recs).  MLP kinds split by seed and
    merged; returns name -> correction."""
    seeds = tuple(hyper.get("seeds", k.HYPER_BIAS_T["seeds"]))
    tasks, owner = [], []
    for name, (kind, arm, recs) in specs.items():
        if kind in ("mlp_t", "mlp"):
            for s in seeds:
                tasks.append((kind, arm, recs, {**hyper, "seeds": (s,)}))
                owner.append(name)
        else:
            tasks.append((kind, arm, recs, {kk: v for kk, v in hyper.items() if kk == "steps"}))
            owner.append(name)
    order = sorted(range(len(tasks)), key=lambda i: -len(tasks[i][2]))
    res = run_pool(workers, train_task, [tasks[i] for i in order])
    parts: dict = {}
    for i, r in zip(order, res):
        parts.setdefault(owner[i], []).append((tasks[i][3].get("seeds", (0,))[0], r))
    out = {}
    for name, lst in parts.items():
        lst.sort(key=lambda x: x[0])
        corrs = [c for _, c in lst]
        if len(corrs) == 1:
            out[name] = corrs[0]
        elif isinstance(corrs[0], k.TempBiasCorrection):
            out[name] = k.merge_temp_seeds(corrs)
        else:
            from cryoml.explore.i3_2_bias import merge_seed_results
            out[name] = merge_seed_results(1, [0] * len(corrs), corrs)[0]
    return out


def bind_for(corr, rec):
    return corr.bind(k.temp_kelvin(rec)) if hasattr(corr, "bind") else corr


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def ihp_folds(recs, temps=IHP_TEMPS, tags=None):
    by_tag = {r.tag: r for r in recs[300]}
    tags = sorted(by_tag) if tags is None else tags
    return [(tag, t) for t in temps for tag in tags]


def run_plan(args) -> int:
    from cryoml.explore.i8_1_nist import finite_record
    t0 = time.time()
    _, rihp = load_ihp()
    _, rtud = load_tud()
    plan = {"ihp": {}, "tud": {}}
    nonfinite = [key(r.tag, t) for t, rs in rihp.items() for r in rs if not finite_record(r)]
    nonfinite += [key(r.tag, t) for t, rs in rtud.items() for r in rs if not finite_record(r)]
    plan["nonfinite_records"] = nonfinite
    by_tag = {r.tag: r for r in rihp[300]}
    sizes = {}
    for tag, t in ihp_folds(rihp):
        for arm in ("M_bias_T", "M_bias_pool", "M_bias_near", "M_bias_in"):
            tr = k.training_set(rihp, tag, t, arm, ihp_unit, ihp_eligible(by_tag[tag]))
            sizes.setdefault(arm, []).append(len(tr))
    plan["ihp"]["n_folds"] = len(ihp_folds(rihp))
    plan["ihp"]["train_records"] = {a: [min(v), max(v)] for a, v in sizes.items()}
    plan["ihp"]["nearest"] = {t: k.nearest_temp(t, IHP_TEMPS) for t in IHP_TEMPS}
    st = stored_rows(I5, IHP_TEMPS)
    plan["ihp"]["stored_rows"] = {a: len(v) for a, v in st.items()}
    plan["ihp"]["stored_x300"] = len(stored_x300())
    plan["ihp"]["stored_corrections"] = {t: len(stored_corrections(I5, t)) for t in (300, 343)}
    groups = sorted({r.meta["group"] for r in rtud[4]})
    tsz = {}
    for r in rtud[4]:
        for t in TUD_TEMPS:
            for arm in ("M_bias_near", "M_bias_in"):
                tr = k.training_set(rtud, r.meta["group"], t, arm, tud_unit)
                tsz.setdefault(arm, []).append(len(tr))
    plan["tud"]["groups"] = groups
    plan["tud"]["n_records"] = sum(len(v) for v in rtud.values())
    plan["tud"]["train_records"] = {a: [min(v), max(v)] for a, v in tsz.items()}
    sttud = stored_rows(I6, TUD_TEMPS)
    plan["tud"]["stored_rows"] = {a: len(v) for a, v in sttud.items()}
    plan["tud"]["stored_corrections"] = {t: len(stored_corrections(I6, t)) for t in TUD_TEMPS}
    plan["temperatures_K"] = {
        "ihp": {t: sorted({round(k.temp_kelvin(r), 3) for r in rihp[t]}) for t in IHP_TEMPS},
        "tud": {t: [round(min(k.temp_kelvin(r) for r in rtud[t]), 3),
                    round(max(k.temp_kelvin(r) for r in rtud[t]), 3)] for t in TUD_TEMPS}}
    plan["runtime_s"] = round(time.time() - t0, 1)
    write_json(OUT / "plan.json", plan)
    print(json.dumps(plan, indent=1))
    return 0


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------
def run_validity(args) -> int:
    t0 = time.time()
    dihp, rihp = load_ihp()
    dtud, rtud = load_tud()
    from cryoml.explore.e1_correction import Correction
    out = {"what": "V1 stored C0_refit rows reproduced; V5 stored fold corrections reproduced"}
    st_i, st_t = stored_rows(I5, IHP_TEMPS), stored_rows(I6, TUD_TEMPS)
    pick_i = [("hv_nmos_W0p3_L0p5", 233), ("lv_pmos_W0p3_L10", 398),
              ("lv_nmos_W0p15_L0p18", 300), ("hv_pmos_W10_L0p35", 343)]
    pick_t = [(rtud[4][0].tag, 4), (rtud[295][-1].tag, 295)]
    if args.smoke:
        pick_i, pick_t = pick_i[:1], pick_t[:1]
    tasks = []
    for tag, t in pick_i:
        rec = next(r for r in rihp[t] if r.tag == tag)
        tasks.append(("ihp", "C0_refit", Correction("C0"), rec, dihp[tag], OPTS))
    for tag, t in pick_t:
        rec = next(r for r in rtud[t] if r.tag == tag)
        tasks.append(("tud", "C0_refit", Correction("C0"), rec, dtud[tag], OPTS))
    rows = run_pool(args.workers, refit_task, tasks)
    diffs = {}
    for task, row in zip(tasks, rows):
        ref = (st_i if task[0] == "ihp" else st_t)["C0_refit"][key(row["tag"], row["T_K"])]
        diffs[key(row["tag"], row["T_K"])] = abs(row["rrms"] - ref) / ref
    out["V1_c0_refit_rel_diff"] = diffs
    out["V1_ok"] = bool(max(diffs.values()) <= REPRO_TOL)
    # V5: retrain stored fold corrections
    from cryoml.explore.i3_2_bias import HYPER_BIAS
    hyper = dict(HYPER_BIAS) if not args.smoke else {"seeds": (0,), "steps": 200}
    specs = {}
    for tag in ("lv_nmos_W0p15_L0p18", "hv_pmos_W10_L0p35")[: 1 if args.smoke else 2]:
        held = next(r for r in rihp[300] if r.tag == tag)
        tr = [r for r in rihp[300] if r.tag != tag and r.meta["cls"] == held.meta["cls"]]
        specs[f"ihp|{tag}"] = ("mlp", "M_bias", tr)
    g = rtud[4][0].meta["group"]
    if not args.smoke:
        specs[f"tud|{g}"] = ("mlp", "M_bias", [r for r in rtud[4] if r.meta["group"] != g])
    trained = train_all(args.workers, specs, hyper)
    st_ci, st_ct = stored_corrections(I5, 300), stored_corrections(I6, 4)
    v5 = {}
    for name, corr in trained.items():
        ds, unit = name.split("|", 1)
        ref = (st_ci if ds == "ihp" else st_ct)[unit]
        rec = (rihp[300] if ds == "ihp" else rtud[4])[1]
        a = np.concatenate(corr.apply(rec))
        b = np.concatenate(ref.apply(rec))
        v5[name] = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    out["V5_correction_max_rel_diff"] = v5
    out["V5_ok"] = bool(args.smoke or max(v5.values()) <= REPRO_TOL)
    out["rows"] = [{kk: v for kk, v in r.items() if kk != "params"} for r in rows]
    out["runtime_s"] = round(time.time() - t0, 1)
    write_json(odir(args) / "validity.json", out)
    print(json.dumps({kk: v for kk, v in out.items() if kk != "rows"}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# stats (secondary prediction)
# ---------------------------------------------------------------------------
def run_stats(args) -> int:
    from cryoml.explore.i7_1_transfer import device_residual
    guards = require(args, PREREG)
    t0 = time.time()
    _, rihp = load_ihp()
    by_tag = {r.tag: r for r in rihp[300]}
    tags = sorted(by_tag)
    temps = IHP_TEMPS
    hyper = {"seeds": (0,), "steps": SMOKE["s3_steps"] if args.smoke else k.S3_HYPER["steps"]}
    if args.smoke:
        tags = [next(t for t in tags if t.startswith("lv")), next(t for t in tags if t.startswith("hv"))]
        temps = SMOKE["ihp_temps"]
    folds = [(tag, t) for t in temps for tag in tags]
    resid = {key(r.tag, t): device_residual(r) for t, rs in rihp.items() for r in rs}
    per_fold = {}
    s3_tasks = []
    for tag, t in folds:
        tr = k.training_set(rihp, tag, t, "M_bias_pool", ihp_unit, ihp_eligible(by_tag[tag]))
        keys = [key(r.tag, r.meta["T_K"]) for r in tr]
        s1 = k.s1_loto([resid[x] for x in keys], [r.meta["T_K"] for r in tr], [r.tag for r in tr])
        per_fold[key(tag, t)] = {"S1_loto": s1, "n_train": len(tr)}
        for ti, part, inner, scored in k.s3_loto_tasks(tr, ihp_unit):
            for arm in ("M_bias_T", "M_bias_pool"):
                s3_tasks.append((key(tag, t), arm, inner, scored, hyper))
    t_s1 = round(time.time() - t0, 1)
    res = run_pool(args.workers, s3_task, s3_tasks)
    vals: dict = {}
    for fold, arm, v in res:
        vals.setdefault(fold, {}).setdefault(arm, []).extend(v)
    for fold, d in vals.items():
        for arm, v in d.items():
            a = np.array(v, float)
            per_fold[fold][f"S3_{arm}"] = float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
            per_fold[fold][f"S3_{arm}_n_nan"] = int(np.sum(~np.isfinite(a)))
    per_t = {}
    for t in temps:
        rows = [per_fold[key(tag, t)] for tag in tags]
        s1 = float(np.mean([r["S1_loto"] for r in rows]))
        s3t = float(np.nanmean([r["S3_M_bias_T"] for r in rows]))
        s3p = float(np.nanmean([r["S3_M_bias_pool"] for r in rows]))
        y = k.i7_1_rule(s1)
        per_t[str(t)] = {
            "stratum": "interpolation" if t in INTERP else "extrapolation",
            "S1_loto": s1, "I7_1_rule_y": y, "S3_M_bias_T": s3t, "S3_M_bias_pool": s3p,
            "predicted": {"M_bias_T_vs_C0_refit (S3)": "gain" if s3t < 0 else "no gain",
                          "M_bias_pool_vs_C0_refit (S3)": "gain" if s3p < 0 else "no gain",
                          "M_bias_pool_vs_C0_refit (I7-1 rule on S1_loto)": "gain" if y < 0 else "no gain"}}
    pred = {"iteration": "I11-2", "dataset": "IHP SG13G2 measured DEV (18 DUTs)",
            "smoke": bool(args.smoke), "guards": guards,
            "what": "training-only statistics per held-out temperature, computed from the starting-card "
                    "records of the training folds before any I11-2 held-out refit",
            "definitions": {
                "S1_loto": "I7-1 S1 with the shared bin function estimated without the scored record's "
                           "temperature and unit, over the M_bias_pool training records; fold mean",
                "S3_arm": "mean ln(RRMS corrected / RRMS start) of post-hoc g (1 seed, 500 steps) on inner "
                          "leave-one-temperature-out x 2 unit-part folds of the training records",
                "prediction": "gain iff S3 < 0; I7-1 frozen rule y = 0.007619 - 0.485184 S1, gain iff y < 0; "
                              "scored against the sign of the mean over DUTs of ln(arm / C0_refit) at T"},
            "per_temperature": per_t, "per_fold": per_fold, "S3_hyper": hyper,
            "timing": {"s1_s": t_s1, "total_s": round(time.time() - t0, 1), "n_s3_fits": len(s3_tasks)}}
    target = (OUT / "smoke" / "prediction.json") if args.smoke else PREDICTION
    write_json(target, pred)
    print(json.dumps({kk: pred[kk] for kk in ("per_temperature", "timing")}, indent=1))
    return 0


def run_commit_prediction(args) -> int:
    from cryoml.explore.i8_1_nist import _git, prediction_guard
    if args.smoke or not PREDICTION.exists():
        print("no commit in smoke mode" if args.smoke else f"missing {PREDICTION}")
        return 1
    rel = str(PREDICTION.relative_to(ROOT))
    msg = ("I11-2 prediction (pre-registered order step): S1_loto, S3_loto and predicted signs per "
           "held-out temperature from the IHP DEV starting-card records, committed before any "
           "held-out refit\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01FTjyogfhfbXwhjQosGAjKp\n")
    rc, out = _git("add", "--", rel)
    if rc:
        print(out)
        return 1
    rc_d, _ = _git("diff", "--cached", "--quiet", "--", rel)
    if rc_d:
        rc, out = _git("commit", "-m", msg, "--", rel)
        print(out)
        if rc:
            return 1
    rc, out = _git("push")
    print(out if rc == 0 else f"push failed (local commit kept): {out}")
    g = prediction_guard(PREDICTION)
    print(json.dumps(g))
    return 0 if g["ok"] else 1


# ---------------------------------------------------------------------------
# lodo
# ---------------------------------------------------------------------------
def collect(rows_l) -> tuple[dict, dict]:
    rows = {}
    for row in rows_l:
        rows.setdefault(row["label"], {})[key(row["tag"], row["T_K"])] = row
    scores = {a: {kk: r["rrms"] for kk, r in v.items()} for a, v in rows.items()}
    return rows, scores


def save_lodo(target: Path, meta: dict, rows: dict, scores: dict, corrs: dict) -> None:
    write_json(target / "results.json", {
        **meta, "per_record": scores,
        "rows": {a: {kk: {x: y for x, y in r.items() if x != "params"} for kk, r in v.items()}
                 for a, v in rows.items()}})
    write_json(target / "cards.json", {a: {kk: r["params"] for kk, r in v.items()} for a, v in rows.items()})
    (target / "corrections.json").write_text(json.dumps({kk: c.to_json() for kk, c in corrs.items()}) + "\n")


def run_lodo_ihp(args) -> int:
    guards = require(args, PREREG, PREDICTION)
    t_start = time.time()
    devices, rihp = load_ihp()
    by_tag = {r.tag: r for r in rihp[300]}
    tags = sorted(by_tag)
    temps = IHP_TEMPS
    opts = dict(OPTS)
    hyper = {}
    if args.smoke:
        tags = [next(t for t in tags if t.startswith("lv")), next(t for t in tags if t.startswith("hv"))]
        temps = SMOKE["ihp_temps"]
        hyper = {"seeds": SMOKE["seeds"], "steps": SMOKE["steps"]}
        opts.update(heldout_nfev=SMOKE["nfev"], heldout_irls=SMOKE["irls"])
    arms = [a for a in args.arms.split(",") if a] if args.arms else list(IHP_ARMS)
    folds = [(tag, t) for t in temps for tag in tags]
    specs = {}
    kinds = {"M_bias_T": "mlp_t", "M_bias_pool": "mlp", "lin_T": "lin_t", "lin_pool": "lin"}
    for tag, t in folds:
        elig = ihp_eligible(by_tag[tag])
        for arm in arms:
            if arm in kinds:
                specs[f"{arm}|{key(tag, t)}"] = (kinds[arm], arm,
                                                 k.training_set(rihp, tag, t, arm, ihp_unit, elig))
    if "M_bias_in" in arms:
        for tag in tags:
            specs[f"M_bias_in|{tag}"] = ("mlp_t", "M_bias_in",
                                         k.training_set(rihp, tag, 300, "M_bias_in", ihp_unit,
                                                        ihp_eligible(by_tag[tag])))
    print(f"I11_2 lodo_ihp: {len(folds)} folds, arms {arms}, {len(specs)} trained corrections, "
          f"opts {opts}, hyper {hyper}", flush=True)
    t0 = time.time()
    corrs = train_all(args.workers, specs, hyper)
    t_train = round(time.time() - t0, 1)
    print(f"training done ({t_train} s)", flush=True)
    near_src = {t: stored_corrections(I5, t) for t in sorted({k.nearest_temp(x, IHP_TEMPS) for x in temps})}
    tasks = []
    for tag, t in folds:
        rec = next(r for r in rihp[t] if r.tag == tag)
        for arm in arms:
            if arm == "M_bias_near":
                tn = k.nearest_temp(t, IHP_TEMPS)
                corr = near_src[tn][tag]
                corrs[f"M_bias_near|{key(tag, t)}"] = corr
            elif arm == "M_bias_in":
                corr = corrs[f"M_bias_in|{tag}"]
            else:
                corr = corrs[f"{arm}|{key(tag, t)}"]
            tasks.append(("ihp", arm, bind_for(corr, rec), rec, devices[tag], opts))
    t0 = time.time()
    rows_l = run_pool(args.workers, refit_task, tasks)
    t_refit = round(time.time() - t0, 1)
    rows, scores = collect(rows_l)
    meta = {"name": "lodo_ihp", "smoke": bool(args.smoke), "guards": guards, "options": opts,
            "hyper_overrides": hyper, "arms": arms, "folds": [key(*f) for f in folds],
            "nearest": {str(t): k.nearest_temp(t, IHP_TEMPS) for t in IHP_TEMPS},
            "timing": {"train_s": t_train, "refit_s": t_refit, "n_refits": len(tasks),
                       "n_trained": len(specs)},
            "runtime_s": round(time.time() - t_start, 1)}
    save_lodo(odir(args) / "lodo_ihp", meta, rows, scores, corrs)
    for a, v in scores.items():
        vv = np.array(list(v.values()))
        print(f"  {a:12s} mean {np.nanmean(vv):.4f} failed {int(np.sum(~np.isfinite(vv)))}")
    print(f"runtime {meta['runtime_s']} s", flush=True)
    return 0


def run_lodo_tud(args) -> int:
    guards = require(args, PREREG, PREDICTION)
    t_start = time.time()
    devices, rtud = load_tud()
    opts = dict(OPTS)
    hyper = {}
    held = sorted(r.tag for r in rtud[4])
    groups = sorted({r.meta["group"] for r in rtud[4]})
    if args.smoke:
        groups = [next(g for g in groups if "_nmos_" in g), next(g for g in groups if "_pmos_" in g)]
        held = [next(r.tag for r in sorted(rtud[4], key=lambda r: r.tag) if r.meta["group"] == g)
                for g in groups]
        hyper = {"seeds": SMOKE["seeds"], "steps": SMOKE["steps"]}
        opts.update(heldout_nfev=SMOKE["nfev"], heldout_irls=SMOKE["irls"])
    arms = [a for a in args.arms.split(",") if a] if args.arms else list(TUD_ARMS)
    specs = {}
    if "M_bias_in" in arms:
        for g in groups:
            specs[f"M_bias_in|{g}"] = ("mlp_t", "M_bias_in",
                                       k.training_set(rtud, g, 4, "M_bias_in", tud_unit))
    print(f"I11_2 lodo_tud: {len(groups)} groups, {len(held)} devices, arms {arms}, opts {opts}, "
          f"hyper {hyper}", flush=True)
    t0 = time.time()
    corrs = train_all(args.workers, specs, hyper) if specs else {}
    t_train = round(time.time() - t0, 1)
    src = {t: stored_corrections(I6, t) for t in TUD_TEMPS}
    tasks = []
    for t in TUD_TEMPS:
        by_tag = {r.tag: r for r in rtud[t]}
        for tag in held:
            rec = by_tag[tag]
            g = rec.meta["group"]
            for arm in arms:
                if arm == "M_bias_X":
                    tn = k.nearest_temp(t, TUD_TEMPS)
                    k.check_training([r for r in rtud[tn] if r.meta["group"] != g], g, t,
                                     "M_bias_near", tud_unit)
                    corr = src[tn][g]
                    corrs[f"M_bias_X|{key(g, t)}"] = corr
                else:
                    corr = corrs[f"M_bias_in|{g}"]
                tasks.append(("tud", arm, bind_for(corr, rec), rec, devices[tag], opts))
    t0 = time.time()
    rows_l = run_pool(args.workers, refit_task, tasks)
    t_refit = round(time.time() - t0, 1)
    rows, scores = collect(rows_l)
    n_to = sum(r.get("n_sim_timeout", 0) for r in rows_l)
    meta = {"name": "lodo_tud", "smoke": bool(args.smoke), "guards": guards, "options": opts,
            "hyper_overrides": hyper, "arms": arms, "groups": groups, "held_out": held,
            "timing": {"train_s": t_train, "refit_s": t_refit, "n_refits": len(tasks),
                       "n_trained": len(specs)}, "n_sim_timeout": n_to,
            "runtime_s": round(time.time() - t_start, 1)}
    save_lodo(odir(args) / "lodo_tud", meta, rows, scores, corrs)
    for a, v in scores.items():
        vv = np.array(list(v.values()))
        print(f"  {a:12s} mean {np.nanmean(vv):.4f} failed {int(np.sum(~np.isfinite(vv)))}")
    print(f"runtime {meta['runtime_s']} s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def as_pairs(d: dict) -> dict:
    return {unkey(kk): v for kk, v in d.items()}


def contrast(m: dict, c: dict, temps, level, clusters_of=None, unit_of_tag=None) -> dict:
    lr, counts = k.unit_log_ratios(as_pairs(m), as_pairs(c), temps, unit_of_tag)
    cl = None if clusters_of is None else {u: clusters_of(u) for u in lr}
    return {**k.gmr_ci(lr, cl, level=level), **counts}


def run_decide(args) -> int:
    guards = require(args, PREREG, PREDICTION)
    base = odir(args)
    ri = json.loads((base / "lodo_ihp/results.json").read_text())
    rt = json.loads((base / "lodo_tud/results.json").read_text())
    si = ri["per_record"]
    stt = rt["per_record"]
    st_i, st_t = stored_rows(I5, IHP_TEMPS), stored_rows(I6, TUD_TEMPS)
    si = {**si, "C0_refit": st_i["C0_refit"], "M_bias_sameT": st_i["M_bias"]}
    stt = {**stt, "C0_refit": st_t["C0_refit"], "M_bias_sameT": st_t["M_bias"]}
    lvl = k.LEVEL_PRIMARY
    temps_i = sorted({unkey(x)[1] for x in si["M_bias_T"]})
    interp = [t for t in INTERP if t in temps_i]
    extrap = [t for t in EXTRAP if t in temps_i]
    m6 = tud_module()
    primary = {}
    for name, temps in (("interp", interp), ("extrap", extrap)):
        if not temps:
            continue
        for ctl in ("C0_refit", "M_bias_pool", "M_bias_near"):
            primary[f"ihp_{name}_M_bias_T_vs_{ctl}"] = contrast(si["M_bias_T"], si[ctl], temps, lvl)
    for t in TUD_TEMPS:
        primary[f"tud_{t}K_M_bias_X_vs_C0_refit"] = contrast(stt["M_bias_X"], stt["C0_refit"], [t], lvl,
                                                             clusters_of=m6.group_of)
    labels = {}
    for name in ("interp", "extrap"):
        if f"ihp_{name}_M_bias_T_vs_C0_refit" in primary:
            labels[f"ihp_{name}"] = k.label_ihp(primary[f"ihp_{name}_M_bias_T_vs_C0_refit"],
                                                primary[f"ihp_{name}_M_bias_T_vs_M_bias_pool"],
                                                primary[f"ihp_{name}_M_bias_T_vs_M_bias_near"])
    for t in TUD_TEMPS:
        labels[f"tud_{t}K"] = k.label_tud(primary[f"tud_{t}K_M_bias_X_vs_C0_refit"])
    # validity
    val = {}
    vpath = base / "validity.json"
    if vpath.exists():
        v = json.loads(vpath.read_text())
        val["V1_c0_refit_reproduced"] = v["V1_ok"]
        val["V5_corrections_reproduced"] = v["V5_ok"]
    else:
        val["V1_c0_refit_reproduced"] = val["V5_corrections_reproduced"] = False
    x300 = stored_x300()
    near = si.get("M_bias_near", {})
    d2 = [abs(near[kk] - x300[kk]) / x300[kk] for kk in near if kk in x300]
    val["V2_near_equals_X300"] = {"n": len(d2), "max_rel_diff": max(d2) if d2 else None,
                                  "ok": bool(d2) and max(d2) <= REPRO_TOL}
    fails = {}
    for ds, sc, arms in (("ihp", si, ("M_bias_T", "M_bias_pool", "M_bias_near")),
                         ("tud", stt, ("M_bias_X",))):
        for a in arms:
            fails[f"{ds}_{a}"] = int(sum(not np.isfinite(x) for x in sc[a].values()))
    val["V3_failed"] = fails
    val["V3_ok"] = all(v <= MAX_FAILED for v in fails.values())
    val["V4_tud_timeouts"] = rt.get("n_sim_timeout")
    valid = bool(val["V1_c0_refit_reproduced"] and val["V5_corrections_reproduced"]
                 and val["V2_near_equals_X300"]["ok"] and val["V3_ok"])
    dec = k.decide({n: labels[f"ihp_{n}"] for n in ("interp", "extrap") if f"ihp_{n}" in labels}, valid)
    # secondary
    sec = {"per_temperature_95": {}, "lv_hv_95": {}, "tud_device_bootstrap": {}, "aggregates": {}}
    for t in temps_i:
        sec["per_temperature_95"][f"ihp_{t}K"] = {
            f"{a}_vs_C0_refit": contrast(si[a], si["C0_refit"], [t], 0.95)
            for a in si if a != "C0_refit" and any(unkey(x)[1] == t for x in si[a])}
        sec["per_temperature_95"][f"ihp_{t}K"]["M_bias_T_vs_M_bias_in"] = contrast(
            si["M_bias_T"], si["M_bias_in"], [t], 0.95) if "M_bias_in" in si else None
        sec["per_temperature_95"][f"ihp_{t}K"]["lin_T_vs_lin_pool"] = contrast(
            si["lin_T"], si["lin_pool"], [t], 0.95) if "lin_T" in si and "lin_pool" in si else None
    for name, temps in (("interp", interp), ("extrap", extrap)):
        for a in ("M_bias_pool", "M_bias_near", "M_bias_in", "M_bias_sameT", "lin_T", "lin_pool"):
            if a in si:
                sec.setdefault("strata_95", {})[f"ihp_{name}_{a}_vs_C0_refit"] = contrast(
                    si[a], si["C0_refit"], temps, 0.95)
        if "lin_T" in si:
            sec.setdefault("strata_95", {})[f"ihp_{name}_lin_T_vs_lin_pool"] = contrast(
                si["lin_T"], si["lin_pool"], temps, 0.95)
        for cls in ("lv", "hv"):
            sub = {a: {kk: v for kk, v in si[a].items() if kk.startswith(cls)} for a in
                   ("M_bias_T", "C0_refit", "M_bias_pool")}
            sec["lv_hv_95"][f"ihp_{name}_{cls}_M_bias_T_vs_C0_refit"] = contrast(
                sub["M_bias_T"], sub["C0_refit"], temps, 0.95)
    for t in TUD_TEMPS:
        sec["tud_device_bootstrap"][f"{t}K_M_bias_X_vs_C0_refit_primary_level"] = contrast(
            stt["M_bias_X"], stt["C0_refit"], [t], lvl)
        for a in ("M_bias_X", "M_bias_in", "M_bias_sameT"):
            if a in stt:
                sec["tud_device_bootstrap"][f"{t}K_{a}_vs_C0_refit_95"] = contrast(
                    stt[a], stt["C0_refit"], [t], 0.95)
                sec["tud_device_bootstrap"][f"{t}K_{a}_vs_C0_refit_95_cluster"] = contrast(
                    stt[a], stt["C0_refit"], [t], 0.95, clusters_of=m6.group_of)
    for ds, sc, temps in (("ihp", si, temps_i), ("tud", stt, TUD_TEMPS)):
        for t in temps:
            for a, v in sc.items():
                vals = [x for kk, x in v.items() if unkey(kk)[1] == t]
                if vals and (ds == "tud" or a in si):
                    keys_t = sorted(kk for kk in sc["M_bias_T" if ds == "ihp" else "M_bias_X"]
                                    if unkey(kk)[1] == t)
                    vv = np.array([v.get(kk, np.nan) for kk in keys_t], float)
                    sec["aggregates"][f"{ds}_{t}K_{a}"] = {
                        "mean": float(np.mean(vv)) if np.all(np.isfinite(vv)) else float("nan"),
                        "median": float(np.nanmedian(vv)), "n": int(len(vv)),
                        "n_failed": int(np.sum(~np.isfinite(vv)))}
    # prediction scoring
    pred_path = (OUT / "smoke" / "prediction.json") if args.smoke else PREDICTION
    if pred_path.exists():
        pred = json.loads(pred_path.read_text())
        score = {}
        for t, p in pred["per_temperature"].items():
            for name, sign in p["predicted"].items():
                arm = name.split("_vs_")[0]
                if arm not in si:
                    continue
                lr, _ = k.unit_log_ratios(as_pairs(si[arm]), as_pairs(si["C0_refit"]), [int(t)])
                y = float(np.mean(list(lr.values()))) if lr else float("nan")
                obs = "gain" if y < 0 else "no gain"
                score[f"{t}K {name}"] = {"predicted": sign, "observed_ln_gmr": y, "observed": obs,
                                         "correct": sign == obs}
        sec["prediction_scoring"] = {"rows": score, "n_correct": sum(v["correct"] for v in score.values()),
                                     "n": len(score)}
    out = {"iteration": "I11-2", "smoke": bool(args.smoke), "guards": guards,
           "level_primary": lvl, "n_primary": k.N_PRIMARY, "primary": primary, "labels": labels,
           "validity": val, "valid": valid, **dec, "secondary": sec,
           "note": "TU Delft labels are reported and do not enter the action (two temperatures: no "
                   "temperature-input arm is defined without the held-out temperature)"}
    write_json(base / "decision.json", out)
    print(json.dumps({kk: out[kk] for kk in ("labels", "validity", "outcome", "action")}, indent=1))
    for kk, v in primary.items():
        print(f"  {kk:45s} {v['ratio']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}] "
              f"lower {v.get('units_lower')}/{v['n_units']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=("plan", "smoke", "validity", "stats", "commit_prediction", "lodo_ihp",
                             "lodo_tud", "decide"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--arms", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.mode == "smoke":
        args.smoke = True
        args.workers = min(args.workers, 2)
        t0 = time.time()
        timing = {}
        for name, fn in (("validity", run_validity), ("stats", run_stats), ("lodo_ihp", run_lodo_ihp),
                         ("lodo_tud", run_lodo_tud), ("decide", run_decide)):
            t1 = time.time()
            rc = fn(args)
            timing[name] = round(time.time() - t1, 1)
            if rc:
                return rc
        timing["total"] = round(time.time() - t0, 1)
        write_json(OUT / "smoke" / "smoke_timing.json", {"workers": args.workers, "timing": timing})
        print(json.dumps(timing))
        return 0
    return {"plan": run_plan, "validity": run_validity, "stats": run_stats,
            "commit_prediction": run_commit_prediction, "lodo_ihp": run_lodo_ihp,
            "lodo_tud": run_lodo_tud, "decide": run_decide}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
