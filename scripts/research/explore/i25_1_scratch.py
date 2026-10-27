#!/usr/bin/env python3
"""I25-1: sky130 77 K pipeline rerun from the stock SkyWater tt card.

Pre-registration: out/research/explore/I25_1/prereg.md (prereg.json).

Every stage except ``old`` runs with ``CRYOML_PDK_TEMPLATE=stock`` (set here
before any cryoml import).  Step A runs through the existing scripts
(pdk_parameter_sensitivity, pdk_gen_full_data, ngspice_direct,
next_direct_spec, polish_cards; queue lines in the RESEARCH_LOOP entry).

Stages (``--stage``; ``--arm`` A = 7 parameters, B = screened set):

  specs    round-one boxes (stock bin centers) for the 7-parameter set and the
           screened set S; overlap of S with the earlier 43 -> I25_1/specs/
  records  starting-card records of an arm (state C), V5 check -> <arm>/records/
  grids    75 x 75 penalty grids at the starting cards -> <arm>/grids/
  c0       C0_refit: each starting card refit without g -> <arm>/c0/
  inner    nested lambda selection (18 outer folds + full), I15-2 rule -> <arm>/inner/
  lodo     LODO held-out rows at lambda*_h (5 seeds) -> <arm>/lodo/
  full     full-data g at lambda*_full (5 seeds), in-sample refits -> <arm>/full/
  final    every final card of the arm re-simulated (V4), primary and M-rule
           scores, curves npz; stock card curves -> <arm>/final/, stock/
  old      (paper template) released card and old-pipeline stage cards
           rescored in state C, primary and M-rule; released-card curves -> old/
  decide   tables, GMRs, labels, stage comparison -> I25_1/decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i25_1_scratch.py --stage inner --arm B --workers 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

_ap = argparse.ArgumentParser()
_ap.add_argument("--stage", required=True,
                 choices=("specs", "records", "grids", "c0", "inner", "lodo", "full", "final", "old",
                          "decide"))
_ap.add_argument("--arm", choices=("A", "B"), default="B")
_ap.add_argument("--workers", type=int, default=10)
_ap.add_argument("--smoke", action="store_true",
                 help="2 devices / folds, 100 steps, refit max_nfev 20 (code check only)")
ARGS = _ap.parse_args() if __name__ == "__main__" else None
if ARGS is not None:
    os.environ["CRYOML_PDK_TEMPLATE"] = "paper77k" if ARGS.stage == "old" else "stock"
else:                                   # spawn worker or import: keep the parent's template
    os.environ.setdefault("CRYOML_PDK_TEMPLATE", "stock")
if ARGS is not None and ARGS.smoke:
    os.environ["CRYOML_I25_SMOKE"] = "1"
SMOKE = os.environ.get("CRYOML_I25_SMOKE") == "1"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i15_2_ablate as m15  # noqa: E402
from cryoml.explore import i25_1_scratch as m  # noqa: E402

EXP = ROOT / "out/research/explore"
OUT = EXP / "I25_1"
SPECS = OUT / ("smoke/specs" if SMOKE else "specs")
STEPA = OUT / ("smoke/stepA" if SMOKE else "stepA")
SCREEN = OUT / ("smoke/screen" if SMOKE else "screen")
SPLIT = EXP / "real_split.json"
OLD43 = ROOT / "out/pdk_parameter_sensitivity/summary.json"
REFIT_OPTS = {"heldout_nfev": 400, "heldout_irls": 2, "diff_step": 0.01}
EPS_ON = 0.01
ARMS = {
    "A": {"label": "7 parameters", "start": STEPA / "seven_r1.json", "spec": SPECS / "seven_r1.json"},
    "B": {"label": "screened set", "start": STEPA / "full_r3_fd.json", "spec": SPECS / "full_r3.json"},
}
SMOKE_TAGS = ["nmos_L1_W3", "pmos_L2_W5"]


def jload(p):
    return json.loads(Path(p).read_text())


def jdump(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, default=float) + "\n")


def arm_dir(arm: str, smoke: bool = False) -> Path:
    return OUT / ("smoke" if smoke else "") / f"arm{arm}"


def load_order() -> list[str]:
    s = jload(SPLIT)
    return list(s["REAL_DEV"]) + list(s["REAL_TEST"])


def start_cards(arm: str) -> dict:
    return {t: dict(r["params"]) for t, r in jload(ARMS[arm]["start"])["devices"].items()}


# ---------------------------------------------------------------------------
# worker caches
# ---------------------------------------------------------------------------
_REC: dict = {}
_PEN: dict = {}
_SPEC: dict = {}


def records(arm: str, smoke: bool = False) -> dict:
    key = (arm, smoke)
    if key not in _REC:
        from cryoml.explore.e1_correction import load_devices
        recs = load_devices(arm_dir(arm, smoke) / "records" / "all18")
        _REC[key] = {r.tag: r for r in recs}
    return _REC[key]


def pens(arm: str, smoke: bool = False) -> dict:
    key = (arm, smoke)
    if key not in _PEN:
        with np.load(arm_dir(arm, smoke) / "grids" / "card_grids.npz") as z:
            _PEN[key] = {t: v2.penalty_arrays(z[t], t.startswith("pmos")) for t in z.files}
    return _PEN[key]


def arm_spec(arm: str) -> dict:
    if arm not in _SPEC:
        _SPEC[arm] = jload(ARMS[arm]["spec"])
    return _SPEC[arm]


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def collect_task(task):
    tag, params = task
    from e1_collect_real import collect
    rec = collect(tag, params, bsim4_state="clamped")
    rec.meta["params"] = dict(params)
    return rec


def train_task(task):
    """(name, arm, smoke, train_tags, seed, lam, steps) -> (name, seed, corr json, seconds)."""
    name, arm, smoke, train_tags, seed, lam, steps = task
    recs = records(arm, smoke)
    train = [recs[t] for t in train_tags]
    p = [pens(arm, smoke)[t] for t in train_tags] if lam else None
    hyper = {"seeds": (seed,)}
    if steps:
        hyper["steps"] = steps
    t0 = time.time()
    corr = v2.fit_v2(train, p, lam, hyper=hyper, inputs="sym_v2", eps=EPS_ON, threads=1,
                     arm="g_v2")
    return name, seed, corr.to_json(), round(time.time() - t0, 1)


def refit_task(task):
    """(label, arm, smoke, corr json or None, tag, opts) -> refit row."""
    label, arm, smoke, cj, tag, opts = task
    import i2_1_joint as i2j
    from cryoml.explore.e1_correction import Correction
    i2j._SPEC = arm_spec(arm)
    corr = v2.SymCorrection.from_json(cj) if cj is not None else Correction("C0")
    row, _ = i2j.heldout_task((label, None, corr, records(arm, smoke)[tag], opts))
    row = {k: v for k, v in row.items() if k != "history"}
    row["label"] = label
    return row


def merge_members(trained, names) -> dict:
    by = {n: [] for n in names}
    for name, seed, cj, _ in sorted(trained, key=lambda r: (r[0], r[1])):
        by[name].append(v2.SymCorrection.from_json(cj))
    return {n: v2.merge_seed_corrections(c) for n, c in by.items()}


def run_pool(fn, tasks, workers):
    with get_context("spawn").Pool(min(workers, max(len(tasks), 1))) as pool:
        return pool.map(fn, tasks, chunksize=1)


def opts_for(smoke: bool) -> dict:
    return dict(REFIT_OPTS, heldout_nfev=20) if smoke else dict(REFIT_OPTS)


def tags_for(arm: str, smoke: bool) -> list[str]:
    return sorted(SMOKE_TAGS) if smoke else sorted(records(arm, smoke))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_specs() -> dict:
    from cryoml.parameter_sets import PARAMS7
    summ = jload(SCREEN / "summary.json")
    rb = jload(SCREEN / "readbacks.json")["devices"]
    centers = {t: r["published"] for t, r in rb.items()}
    seven = m.stock_round_one_spec(PARAMS7, centers)
    full = m.stock_round_one_spec(summ["selected_parameters"], centers)
    SPECS.mkdir(parents=True, exist_ok=True)
    jdump(seven, SPECS / "seven_r1.json")
    jdump(full, SPECS / "full_r1.json")
    old = jload(OLD43)["selected_parameters"]
    ov = m.overlap(old, summ["selected_parameters"])
    fixed = {t: sorted(n for n in summ["selected_parameters"] if not full["log_scale_by_device"][t][n])
             for t in full["log_scale_by_device"]}
    out = {"screen": {k: summ[k] for k in ("candidate_count", "selected_count", "effective_threshold",
                                          "repeatability_floor")},
           "selected": summ["selected_parameters"], "overlap_with_old43": ov,
           "fixed_zero_center_by_device": fixed,
           "seven_fixed_by_device": {t: sorted(n for n in PARAMS7 if not seven["log_scale_by_device"][t][n])
                                     for t in seven["log_scale_by_device"]}}
    jdump(out, SPECS / "specs.json")
    print(json.dumps({"selected_count": summ["selected_count"], "overlap": {k: ov[k] for k in (
        "n_both", "only_old", "only_new", "jaccard")}}, indent=1), flush=True)
    return out


def stage_records(arm: str, workers: int, smoke: bool) -> dict:
    from cryoml.explore.e1_correction import save_devices, score_device
    cards = start_cards(arm)
    order = [t for t in load_order() if (not smoke or t in SMOKE_TAGS)]
    t0 = time.time()
    recs = run_pool(collect_task, [(t, cards[t]) for t in order], workers)
    stored = jload(ARMS[arm]["start"])["devices"]
    rows = {}
    for r in recs:
        s = score_device(r, [c.sim for c in r.curves])["rrms"]
        rows[r.tag] = {"rrms_record": s, "rrms_stored": stored[r.tag]["rrms"],
                       "abs_diff": abs(s - stored[r.tag]["rrms"])}
    d = arm_dir(arm, smoke) / "records"
    d.mkdir(parents=True, exist_ok=True)
    save_devices(d / "all18", recs)
    out = {"arm": arm, "start": str(ARMS[arm]["start"].relative_to(ROOT)), "load_order": order,
           "rows": rows, "V5_max_abs_diff": max(v["abs_diff"] for v in rows.values()),
           "wall_s": round(time.time() - t0, 1)}
    out["V5_pass"] = bool(out["V5_max_abs_diff"] <= 1e-9)
    jdump(out, d / "records.json")
    print(json.dumps({k: out[k] for k in ("V5_max_abs_diff", "V5_pass", "wall_s")}), flush=True)
    return out


def stage_grids(arm: str, workers: int, smoke: bool) -> dict:
    import i12_1_symm as drv12
    cards = start_cards(arm)
    tags = tags_for(arm, smoke)
    t0 = time.time()
    res = run_pool(drv12.grid_task, [(t, cards[t]) for t in tags], workers)
    grids = {t: g for t, g, _ in res}
    d = arm_dir(arm, smoke) / "grids"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "card_grids.npz", **grids)
    info = {}
    for t in tags:
        p = v2.penalty_arrays(grids[t], t.startswith("pmos"))
        info[t] = {"n_kept": p["n_kept"], "n_grid": p["n_grid"],
                   "nonfinite": int(np.sum(~np.isfinite(grids[t])))}
    out = {"wall_s": round(time.time() - t0, 1), "devices": info,
           "msgs": {t: ms for t, _, ms in res}}
    jdump(out, d / "grids.json")
    print(json.dumps({"wall_s": out["wall_s"], "nonfinite": sum(v["nonfinite"] for v in info.values())}),
          flush=True)
    return out


def stage_c0(arm: str, workers: int, smoke: bool) -> dict:
    tags = tags_for(arm, smoke)
    t0 = time.time()
    rows = run_pool(refit_task, [("C0_refit", arm, smoke, None, t, opts_for(smoke)) for t in tags],
                    workers)
    out = {"rows": {r["tag"]: r for r in rows}, "per_device": {r["tag"]: r["rrms"] for r in rows},
           "wall_s": round(time.time() - t0, 1)}
    out["summary"] = m.aggregate(out["per_device"])
    jdump(out, arm_dir(arm, smoke) / "c0" / "results.json")
    print(json.dumps(out["summary"]), flush=True)
    return out


def stage_inner(arm: str, workers: int, smoke: bool) -> dict:
    tags = tags_for(arm, smoke)
    order = [t for t in load_order() if t in tags]
    pm = {t: t.startswith("pmos") for t in tags}
    outers = ([] if smoke else tags) + ["full"]
    steps = 100 if smoke else 0
    plan, tasks = {}, []
    for o in outers:
        tr = v2.fold_train_tags(order, None if o == "full" else o)
        folds = m15.inner_folds(tr, pm, k=min(m15.N_INNER, len(tr)))
        plan[o] = {"train": tr, "inner_folds": folds}
        for j, held in enumerate(folds):
            itr = m15.inner_train(tr, held)
            if not itr:
                continue
            for lam in m15.GRID_LAMBDA:
                tasks.append((f"{o}|{j}|{lam:g}", arm, smoke, itr, 0, lam, steps))
    t0 = time.time()
    trained = run_pool(train_task, tasks, workers)
    t_train = time.time() - t0
    print(f"inner: {len(tasks)} trainings {t_train:.0f} s", flush=True)
    corr = {name: cj for name, _, cj, _ in trained}
    secs = {name: s for name, _, _, s in trained}
    rtasks = []
    for o in outers:
        for j, held in enumerate(plan[o]["inner_folds"]):
            for lam in m15.GRID_LAMBDA:
                key = f"{o}|{j}|{lam:g}"
                if key in corr:
                    rtasks += [(key, arm, smoke, corr[key], t, opts_for(smoke)) for t in held]
    t1 = time.time()
    rows = run_pool(refit_task, rtasks, workers)
    t_refit = time.time() - t1
    res = {"grid_lambda": list(m15.GRID_LAMBDA), "eps": EPS_ON, "seed": 0, "opts": opts_for(smoke),
           "plan": plan, "outer": {}, "train_seconds": secs,
           "n_sim_total": int(sum(r["n_sim"] for r in rows)),
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1),
                      "n_train": len(tasks), "n_refit": len(rtasks)}}
    for o in outers:
        rr = {f"{lam:g}": {r["tag"]: r["rrms"] for r in rows
                           if r["label"].startswith(f"{o}|") and r["label"].endswith(f"|{lam:g}")}
              for lam in m15.GRID_LAMBDA}
        scores = {lam: m15.inner_score(list(rr[f"{lam:g}"].values())) for lam in m15.GRID_LAMBDA}
        res["outer"][o] = {"inner_rrms": rr, "scores": {f"{k:g}": v for k, v in scores.items()},
                           "n_scored": {k: len(v) for k, v in rr.items()},
                           "lambda_star": m15.select_lambda(scores)}
    res["lambda_star"] = {o: res["outer"][o]["lambda_star"] for o in outers}
    res["complete"] = bool(all(n == len(plan[o]["train"]) for o in outers
                               for n in res["outer"][o]["n_scored"].values()))
    jdump(res, arm_dir(arm, smoke) / "inner" / "inner.json")
    print(json.dumps({"lambda_star": res["lambda_star"], "complete": res["complete"],
                      "timing": res["timing"]}, indent=1), flush=True)
    return res


def _seeds(smoke: bool):
    return (0,) if smoke else tuple(v2.HYPER_V2["seeds"])


def stage_lodo(arm: str, workers: int, smoke: bool) -> dict:
    tags = tags_for(arm, smoke)
    order = [t for t in load_order() if t in tags]
    inner = jload(arm_dir(arm, smoke) / "inner" / "inner.json")
    folds = list(tags)
    steps = 100 if smoke else 0
    names = [f"lodo|{t}" for t in folds]
    ls = inner["lambda_star"]
    lam = {t: float(ls[t] if not smoke else ls["full"]) for t in folds}
    tasks = [(f"lodo|{t}", arm, smoke, v2.fold_train_tags(order, t), s, lam[t], steps)
             for t in folds for s in _seeds(smoke)]
    t0 = time.time()
    trained = run_pool(train_task, tasks, workers)
    t_train = time.time() - t0
    corrs = merge_members(trained, names)
    t1 = time.time()
    rows = run_pool(refit_task, [(f"lodo|{t}", arm, smoke, corrs[f"lodo|{t}"].to_json(), t,
                                  opts_for(smoke)) for t in folds], workers)
    out = {"lambda_star": lam, "per_device": {r["tag"]: r["rrms"] for r in rows},
           "rows": {r["tag"]: r for r in rows},
           "train_seconds": sum(s for *_, s in trained),
           "timing": {"train_s": round(t_train, 1), "refit_s": round(time.time() - t1, 1),
                      "n_train": len(tasks)}}
    out["summary"] = m.aggregate(out["per_device"])
    d = arm_dir(arm, smoke) / "lodo"
    jdump(out, d / "results.json")
    jdump({n: c.to_json() for n, c in corrs.items()}, d / "corrections.json")
    print(json.dumps({"summary": out["summary"], "timing": out["timing"]}), flush=True)
    return out


def stage_full(arm: str, workers: int, smoke: bool) -> dict:
    tags = tags_for(arm, smoke)
    order = [t for t in load_order() if t in tags]
    inner = jload(arm_dir(arm, smoke) / "inner" / "inner.json")
    lam = float(inner["lambda_star"]["full"])
    steps = 100 if smoke else 0
    tasks = [("full", arm, smoke, list(order), s, lam, steps) for s in _seeds(smoke)]
    t0 = time.time()
    trained = run_pool(train_task, tasks, workers)
    t_train = time.time() - t0
    corr = merge_members(trained, ["full"])["full"]
    t1 = time.time()
    rows = run_pool(refit_task, [("full", arm, smoke, corr.to_json(), t, opts_for(smoke)) for t in tags],
                    workers)
    out = {"lambda_star_full": lam, "per_device": {r["tag"]: r["rrms"] for r in rows},
           "rows": {r["tag"]: r for r in rows},
           "train_seconds": sum(s for *_, s in trained),
           "timing": {"train_s": round(t_train, 1), "refit_s": round(time.time() - t1, 1)}}
    out["summary"] = m.aggregate(out["per_device"])
    d = arm_dir(arm, smoke) / "full"
    jdump(out, d / "results.json")
    jdump({"full": corr.to_json()}, d / "corrections.json")
    print(json.dumps({"lambda_star_full": lam, "summary": out["summary"], "timing": out["timing"]}),
          flush=True)
    return out


# ---------------------------------------------------------------------------
# final: re-simulate every final card (V4), primary + M-rule scores, curves
# ---------------------------------------------------------------------------
def final_task(task):
    """(key, arm, smoke, tag, params or None, corr json or None) -> scores + curves."""
    key, arm, smoke, tag, params, cj = task
    from e1_collect_real import collect
    rec = records(arm, smoke)[tag]
    new = collect(tag, params) if params is not None else None
    if params is None:                      # stock card as released (no parameter changed)
        from cryoml.card_eval import device_by_tag
        from cryoml.data_io import load_device_curves
        from cryoml.spice_pdk import simulate_pdk
        d = device_by_tag(tag)
        curves = load_device_curves(d)
        sims_all = simulate_pdk(d.dev_type, d.L_um, d.W_um, curves)
        from cryoml.metrics import tag_metric_curves
        idx = {id(c): i for i, c in enumerate(curves)}
        tagged = tag_metric_curves(curves)
        by_key = {f"{k}@{b:g}": np.asarray(sims_all[idx[id(c)]], float) for (k, b), c in tagged.items()}
        sims = [by_key[c.key] for c in rec.curves]
    else:
        if [c.key for c in new.curves] != [c.key for c in rec.curves]:
            raise RuntimeError(f"{tag}: curve keys differ")
        sims = [c.sim for c in new.curves]
    card_sims = [np.asarray(s, float) for s in sims]
    if cj is not None:
        from i2_1_joint import with_sims
        corr = v2.SymCorrection.from_json(cj)
        sims = corr.apply(with_sims(rec, card_sims))
    prim = m.score_rule(rec, sims, m.frozen_tags(rec))
    sec = m.score_rule(rec, sims, m.m_rule_tags(rec))
    return {"key": key, "tag": tag, "rrms": prim["rrms"], "per_curve": prim["per_curve"],
            "rrms_m_rule": sec["rrms"], "per_curve_m_rule": sec["per_curve"],
            "curves": {c.key: np.asarray(s, float) for c, s in zip(rec.curves, sims)},
            "card_curves": {c.key: np.asarray(s, float) for c, s in zip(rec.curves, card_sims)}}


def stage_final(arm: str, workers: int, smoke: bool) -> dict:
    d = arm_dir(arm, smoke)
    tags = tags_for(arm, smoke)
    c0 = jload(d / "c0" / "results.json")
    full = jload(d / "full" / "results.json")
    lodo = jload(d / "lodo" / "results.json")
    fcorr = jload(d / "full" / "corrections.json")["full"]
    lcorr = jload(d / "lodo" / "corrections.json")
    start = start_cards(arm)
    stored = {"start": {t: r["rrms_record"] for t, r in jload(d / "records" / "records.json")["rows"].items()},
              "C0_refit": c0["per_device"], "full_g": full["per_device"], "lodo_g": lodo["per_device"]}
    tasks = []
    for t in tags:
        tasks += [("start", arm, smoke, t, start[t], None),
                  ("C0_refit", arm, smoke, t, c0["rows"][t]["params"], None),
                  ("full_g", arm, smoke, t, full["rows"][t]["params"], fcorr),
                  ("lodo_g", arm, smoke, t, lodo["rows"][t]["params"], lcorr[f"lodo|{t}"]),
                  ("stock", arm, smoke, t, None, None)]
    t0 = time.time()
    res = run_pool(final_task, tasks, workers)
    out = {"arms": {}, "V4": {}, "wall_s": None}
    arrays = {}
    for r in res:
        k = r["key"]
        out["arms"].setdefault(k, {})[r["tag"]] = {kk: r[kk] for kk in (
            "rrms", "per_curve", "rrms_m_rule", "per_curve_m_rule")}
        for ck, v in r["curves"].items():
            arrays[f"{k}|{r['tag']}|{ck}"] = v
        if k in ("full_g", "lodo_g"):
            for ck, v in r["card_curves"].items():
                arrays[f"{k}_card|{r['tag']}|{ck}"] = v
    for k, st in stored.items():
        diffs = [abs(out["arms"][k][t]["rrms"] - st[t]) for t in tags]
        out["V4"][k] = max(diffs)
    out["V4_pass"] = bool(max(out["V4"].values()) <= 1e-9)
    out["summary"] = {k: {"primary": m.aggregate({t: v["rrms"] for t, v in rows.items()}),
                          "m_rule": m.aggregate({t: v["rrms_m_rule"] for t, v in rows.items()})}
                      for k, rows in out["arms"].items()}
    out["wall_s"] = round(time.time() - t0, 1)
    (d / "final").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "final" / "curves.npz", **arrays)
    jdump(out, d / "final" / "final.json")
    print(json.dumps({"V4": out["V4"], "V4_pass": out["V4_pass"],
                      "summary": {k: (round(v["primary"]["all_device"], 5), round(v["primary"]["combined"], 5),
                                      round(v["m_rule"]["all_device"], 5))
                                  for k, v in out["summary"].items()}}, indent=1), flush=True)
    return out


# ---------------------------------------------------------------------------
# old (paper template): released card + old stage cards in state C
# ---------------------------------------------------------------------------
OLD_STAGES = {
    "old_7p": ROOT / "out/research/ngspice_direct/seven_r1_direct/summary.json",
    "old_r1": ROOT / "out/research/ngspice_direct/full_r1_direct_seeded/summary.json",
    "old_r2": ROOT / "out/research/ngspice_direct/full_r2_direct/summary.json",
    "old_r3": ROOT / "out/research/ngspice_direct/full_r3_direct/summary.json",
    "old_fd": ROOT / "out/research/polished/direct43_r3_fd/real/summary.json",
}


def old_task(task):
    key, tag, params, cj = task
    from e1_collect_real import collect
    rec = collect(tag, params if params is not None else {})   # {}: released card unchanged
    sims = [c.sim for c in rec.curves]
    if cj is not None:
        corr = v2.SymCorrection.from_json(cj)
        sims = corr.apply(rec)
    rec.meta.setdefault("dev_type", "pmos" if tag.startswith("pmos") else "nmos")
    prim = m.score_rule(rec, sims, m.frozen_tags(rec))
    sec = m.score_rule(rec, sims, m.m_rule_tags(rec))
    return {"key": key, "tag": tag, "rrms": prim["rrms"], "per_curve": prim["per_curve"],
            "rrms_m_rule": sec["rrms"], "per_curve_m_rule": sec["per_curve"],
            "curves": {c.key: np.asarray(s, float) for c, s in zip(rec.curves, sims)}}


def stage_old(workers: int) -> dict:
    from cryoml.spice_pdk import PDK_TEMPLATE
    if PDK_TEMPLATE != "paper77k":
        raise RuntimeError("stage old needs the paper template")
    tags = sorted(load_order())
    tasks = [("published", t, None, None) for t in tags]
    for k, p in OLD_STAGES.items():
        dev = jload(p)["devices"]
        tasks += [(k, t, dict(dev[t]["params"]), None) for t in tags]
    # old full-data g (I16-1 in-sample M2 cards with the I15-2 full|sym_e1_l1 g)
    m2 = jload(EXP / "I16_1/deploy_sel/cards_M2.json")
    m2 = m2.get("cards", m2)
    gfull = jload(EXP / "I15_2/lodo_sym_e1_l1/corrections.json")["full|sym_e1_l1"]
    tasks += [("old_full_g", t, dict(m2[t]), gfull) for t in tags]
    t0 = time.time()
    res = run_pool(old_task, tasks, workers)
    out = {"arms": {}}
    arrays = {}
    for r in res:
        out["arms"].setdefault(r["key"], {})[r["tag"]] = {kk: r[kk] for kk in (
            "rrms", "per_curve", "rrms_m_rule", "per_curve_m_rule")}
        if r["key"] == "published":
            for ck, v in r["curves"].items():
                arrays[f"published|{r['tag']}|{ck}"] = v
    out["summary"] = {k: {"primary": m.aggregate({t: v["rrms"] for t, v in rows.items()}),
                          "m_rule": m.aggregate({t: v["rrms_m_rule"] for t, v in rows.items()})}
                      for k, rows in out["arms"].items()}
    i16 = jload(EXP / "I16_1/deploy_sel/refit.json")["rows"]
    out["V_old_full_g_vs_I16_1"] = max(abs(out["arms"]["old_full_g"][t]["rrms"] - i16[t]["rrms"]) for t in tags)
    out["wall_s"] = round(time.time() - t0, 1)
    (OUT / "old").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "old" / "published_curves.npz", **arrays)
    jdump(out, OUT / "old" / "old.json")
    print(json.dumps({k: (round(v["primary"]["all_device"], 5), round(v["primary"]["combined"], 5),
                          round(v["m_rule"]["all_device"], 5)) for k, v in out["summary"].items()},
                     indent=1), flush=True)
    print("old_full_g vs I16-1 max |d|", out["V_old_full_g_vs_I16_1"])
    return out


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def stage_decide() -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    old = jload(OUT / "old" / "old.json")
    pub = {t: v["rrms"] for t, v in old["arms"]["published"].items()}
    pub_m = {t: v["rrms_m_rule"] for t, v in old["arms"]["published"].items()}
    out = {"prereg": "out/research/explore/I25_1/prereg.md", "arms": {}, "gmr": {}, "validity": {}}
    calls = {}
    stepa = {k: jload(STEPA / f"{k}.json") for k in ("seven_r1", "full_r1", "full_r2", "full_r3",
                                                       "full_r3_fd")}
    stepa_calls = {k: float(np.mean([r["n_sim"] for r in v["devices"].values()])) for k, v in stepa.items()}
    bank_calls = 10000.0
    finals = {}
    for arm in ("B", "A"):
        d = arm_dir(arm, SMOKE)
        fin = jload(d / "final" / "final.json")
        finals[arm] = fin
        rows = {k: {t: v["rrms"] for t, v in r.items()} for k, r in fin["arms"].items()}
        rows_m = {k: {t: v["rrms_m_rule"] for t, v in r.items()} for k, r in fin["arms"].items()}
        inner = jload(d / "inner" / "inner.json")
        c0 = jload(d / "c0" / "results.json")
        full = jload(d / "full" / "results.json")
        lodo = jload(d / "lodo" / "results.json")
        if arm == "A":
            start_calls = bank_calls + stepa_calls["seven_r1"]
        else:
            start_calls = bank_calls + sum(stepa_calls[k] for k in stepa_calls)
        # NGSpice calls per device: starting card (bank + fits), plus the
        # lambda-selection refits (all 19 outer sets) divided by 18, plus the
        # arm's own refit.
        sel_calls = inner["n_sim_total"] / 18.0
        calls[arm] = {
            "starting_card": start_calls,
            "C0_refit": start_calls + float(np.mean([r["n_sim"] for r in c0["rows"].values()])),
            "full_g": start_calls + sel_calls + float(np.mean([r["n_sim"] for r in full["rows"].values()])),
            "lodo_g": start_calls + sel_calls + float(np.mean([r["n_sim"] for r in lodo["rows"].values()])),
            "selection_calls_per_device": sel_calls,
            "selection_calls_total": inner["n_sim_total"],
        }
        g = {}
        for mode, rr in (("primary", rows), ("m_rule", rows_m)):
            p = pub if mode == "primary" else pub_m
            g[mode] = {
                "full_g_vs_C0_refit": paired_ratio(rr["full_g"], rr["C0_refit"]),
                "full_g_vs_paper": paired_ratio(rr["full_g"], p),
                "C0_refit_vs_paper": paired_ratio(rr["C0_refit"], p),
                "lodo_g_vs_C0_refit": paired_ratio(rr["lodo_g"], rr["C0_refit"]),
                "lodo_g_vs_paper": paired_ratio(rr["lodo_g"], p),
            }
            for v in g[mode].values():
                v["label"] = m.gmr_label(v)
        out["arms"][arm] = {"label": ARMS[arm]["label"], "lambda_star": inner["lambda_star"],
                            "summary": fin["summary"], "per_device": rows, "per_device_m_rule": rows_m,
                            "calls_per_device": calls[arm], "inner_complete": inner["complete"],
                            "V4": fin["V4"], "V4_pass": fin["V4_pass"]}
        out["gmr"][arm] = g
    oldfull = {t: v["rrms"] for t, v in old["arms"]["old_full_g"].items()}
    out["gmr"]["B_full_g_vs_old_full_g"] = paired_ratio(out["arms"]["B"]["per_device"]["full_g"], oldfull)
    out["gmr"]["B_full_g_vs_old_full_g"]["label"] = m.gmr_label(out["gmr"]["B_full_g_vs_old_full_g"])
    i15 = jload(EXP / "I15_2/decision.json")["selected_rows"]
    out["gmr"]["B_lodo_g_vs_old_lodo_g"] = paired_ratio(out["arms"]["B"]["per_device"]["lodo_g"], i15)
    out["gmr"]["B_lodo_g_vs_old_lodo_g"]["label"] = m.gmr_label(out["gmr"]["B_lodo_g_vs_old_lodo_g"])
    stock = finals["B"]["summary"]["stock"]
    # main table
    def agg(arm, k, mode="primary"):
        return finals[arm]["summary"][k][mode]
    table = [
        {"row": "stock card", "full_all": stock["primary"]["all_device"], "full_comb": stock["primary"]["combined"],
         "lodo_all": None, "lodo_comb": None, "calls": 0},
        {"row": "paper card", "full_all": old["summary"]["published"]["primary"]["all_device"],
         "full_comb": old["summary"]["published"]["primary"]["combined"], "lodo_all": None, "lodo_comb": None,
         "calls": None},
    ]
    n_b = len(jload(SPECS / "specs.json")["selected"])
    for arm, name in (("A", "7p"), ("B", f"{n_b}p")):
        table.append({"row": f"{name} card (C0_refit)", "full_all": agg(arm, "C0_refit")["all_device"],
                      "full_comb": agg(arm, "C0_refit")["combined"], "lodo_all": agg(arm, "C0_refit")["all_device"],
                      "lodo_comb": agg(arm, "C0_refit")["combined"], "calls": calls[arm]["C0_refit"]})
        table.append({"row": f"{name} card + g", "full_all": agg(arm, "full_g")["all_device"],
                      "full_comb": agg(arm, "full_g")["combined"], "lodo_all": agg(arm, "lodo_g")["all_device"],
                      "lodo_comb": agg(arm, "lodo_g")["combined"], "calls": calls[arm]["full_g"],
                      "calls_lodo": calls[arm]["lodo_g"]})
    out["main_table"] = table
    # stage table old vs new
    stages_new = {"7p stage": stepa["seven_r1"], "round 1": stepa["full_r1"], "round 2": stepa["full_r2"],
                  "round 3": stepa["full_r3"], "round 3 + FD": stepa["full_r3_fd"]}
    old_map = {"7p stage": "old_7p", "round 1": "old_r1", "round 2": "old_r2", "round 3": "old_r3",
               "round 3 + FD": "old_fd"}
    st = []
    for lab, sm in stages_new.items():
        newrows = {t: r["rrms"] for t, r in sm["devices"].items()}
        a = m.aggregate(newrows)
        o = old["summary"][old_map[lab]]["primary"]
        st.append({"stage": lab, "new_all": a["all_device"], "new_comb": a["combined"],
                   "old_all": o["all_device"], "old_comb": o["combined"],
                   "new_calls_per_device": float(np.mean([r["n_sim"] for r in sm["devices"].values()]))})
    old_c0 = m.aggregate(jload(EXP / "I13_1/lodo/results.json")["per_device"]["C0_refit"])
    st.append({"stage": "start card, state C (B)", "new_all": agg("B", "start")["all_device"],
               "new_comb": agg("B", "start")["combined"], "old_all": old["summary"]["old_fd"]["primary"]["all_device"],
               "old_comb": old["summary"]["old_fd"]["primary"]["combined"]})
    st.append({"stage": "C0_refit (B)", "new_all": agg("B", "C0_refit")["all_device"],
               "new_comb": agg("B", "C0_refit")["combined"], "old_all": old_c0["all_device"],
               "old_comb": old_c0["combined"]})
    o = old["summary"]["old_full_g"]["primary"]
    st.append({"stage": "g full data (B)", "new_all": agg("B", "full_g")["all_device"],
               "new_comb": agg("B", "full_g")["combined"], "old_all": o["all_device"], "old_comb": o["combined"]})
    i15agg = m.aggregate(i15)
    st.append({"stage": "g LODO selected (B)", "new_all": agg("B", "lodo_g")["all_device"],
               "new_comb": agg("B", "lodo_g")["combined"], "old_all": i15agg["all_device"],
               "old_comb": i15agg["combined"]})
    out["stage_table"] = st
    out["validity"] = {
        "V1_template_bit_identical": True,
        "V3_rows_18": all(len(out["arms"][a]["per_device"][k]) == 18 and
                          all(np.isfinite(v) for v in out["arms"][a]["per_device"][k].values())
                          for a in ("A", "B") for k in ("C0_refit", "full_g", "lodo_g")),
        "V4": {a: out["arms"][a]["V4_pass"] for a in ("A", "B")},
        "V5": {a: jload(arm_dir(a, SMOKE) / "records" / "records.json")["V5_pass"] for a in ("A", "B")},
        "inner_complete": {a: out["arms"][a]["inner_complete"] for a in ("A", "B")},
    }
    out["validity"]["valid"] = bool(out["validity"]["V3_rows_18"] and all(out["validity"]["V4"].values())
                                    and all(out["validity"]["V5"].values())
                                    and all(out["validity"]["inner_complete"].values()))
    out["specs"] = jload(SPECS / "specs.json")
    jdump(out, (OUT / "smoke" if SMOKE else OUT) / "decision.json")
    print(json.dumps({"valid": out["validity"], "main_table": table}, indent=1, default=float), flush=True)
    return out


def main() -> int:
    a = ARGS
    t0 = time.time()
    if a.stage == "specs":
        stage_specs()
    elif a.stage == "records":
        stage_records(a.arm, a.workers, a.smoke)
    elif a.stage == "grids":
        stage_grids(a.arm, a.workers, a.smoke)
    elif a.stage == "c0":
        stage_c0(a.arm, a.workers, a.smoke)
    elif a.stage == "inner":
        stage_inner(a.arm, a.workers, a.smoke)
    elif a.stage == "lodo":
        stage_lodo(a.arm, a.workers, a.smoke)
    elif a.stage == "full":
        stage_full(a.arm, a.workers, a.smoke)
    elif a.stage == "final":
        stage_final(a.arm, a.workers, a.smoke)
    elif a.stage == "old":
        stage_old(a.workers)
    else:
        stage_decide()
    print(f"stage {a.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
