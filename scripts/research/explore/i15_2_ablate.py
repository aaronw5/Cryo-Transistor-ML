#!/usr/bin/env python3
"""I15-2: ablation of the g v2 training changes (symmetric inputs, smooth-abs
EPS, gm/gds penalty lambda) on the 18-fold LODO, and a training-data-only
nested 3-fold selection of lambda.  sky130 77 K, 18 devices, I13-1 state C.

Pre-registration: out/research/explore/I15_2/prereg.md (prereg.json).

Stages (``--stage``):

  validity  V1 trainer reproduces stored member 0 of fold pmos_L2_W5 (g v2 and
            v1 recipes); V2 held-out refit of two stored g v2 fold corrections
  lodo      --variant <name>: 18 fold + full-data corrections (5 seeds) and
            the 18 held-out refits -> I15_2/lodo_<name>/
  inner     nested selection: for every outer fold and the full-data model,
            3 inner folds x 5 lambdas, one member (seed 0), inner held-out
            refits -> I15_2/inner/inner.json
  decide    variant table, contrasts, selected rows, primary GMR, negative
            gm / gds fractions -> I15_2/decision.json

  PYTHONPATH=src .venv/bin/python scripts/research/explore/i15_2_ablate.py --stage lodo --variant sym_e1_l0 --workers 6
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "out/research/explore"
I13 = EXP / "I13_1"
os.environ["CRYOML_E1_RECORDS"] = str(I13 / "records")
os.environ["CRYOML_I3_2_ALL18"] = str(I13 / "all18")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "explore"))

import i12_1_symm as drv12  # noqa: E402  (records, load order, penalty grids, refit)
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore import i15_2_ablate as m  # noqa: E402

OUT = EXP / "I15_2"
PREREG = OUT / "prereg.json"
ALL18, LODO = I13 / "all18", I13 / "lodo"
REFIT_OPTS = dict(drv12.REFIT_OPTS)
SMOKE_FOLDS = ["nmos_L1_W1p6", "pmos_L2_W5"]
jload, jdump = drv12.jload, drv12.jdump


def is_pmos_map() -> dict:
    return {t: d.dev_type == "pmos" for t, d in drv12.devices().items()}


# ---------------------------------------------------------------------------
# training / refit tasks
# ---------------------------------------------------------------------------
def train_task(task):
    """One member: (name, train_tags, seed, hyper, lam, inputs, eps)."""
    name, train_tags, seed, hyper, lam, inputs, eps = task
    recs = drv12.load_records()
    train = [recs[t] for t in train_tags]
    pens = None
    if lam:
        allp = drv12.load_pens()
        pens = [allp[t] for t in train_tags]
    t0 = time.time()
    corr = v2.fit_v2(train, pens, lam, hyper={**hyper, "seeds": (seed,)}, inputs=inputs,
                     eps=v2.EPS if eps is None else eps, threads=1, arm=name.split("|", 1)[0])
    return name, seed, corr, round(time.time() - t0, 1), list(train_tags)


def merge(results, names) -> tuple[dict, dict]:
    by = {n: [] for n in names}
    secs = {n: [] for n in names}
    for name, seed, corr, sec, _ in sorted(results, key=lambda r: (r[0], r[1])):
        by[name].append(corr)
        secs[name].append(sec)
    return {n: v2.merge_seed_corrections(c) for n, c in by.items()}, secs


def refit_task(task):
    """(label, corr, tag, opts) -> held-out refit row (strict paper RRMS)."""
    label, corr, tag, opts = task
    row = drv12.heldout((label, None, corr, drv12.load_records()[tag], opts))
    return {**{k: v for k, v in row.items() if k != "history"}, "label": label}


def stored_corr(variant: str, key_tag: str | None):
    """Stored v1 (BiasCorrection) or g v2 (SymCorrection) fold / full correction."""
    from cryoml.explore.i3_2_bias import correction_from_json
    src, arm = m.STORED[variant]
    js = jload((ALL18 if src == "all18" else LODO) / "corrections.json")
    key = f"{arm}|{key_tag}" if key_tag else f"full|{arm}"
    return correction_from_json(js[key]) if variant == "v1" else v2.SymCorrection.from_json(js[key])


def stored_rows(variant: str) -> dict:
    if variant == "v1":
        return dict(jload(ALL18 / "results.json")["per_device"]["M_bias"])
    if variant == "sym_e1_l10":
        return dict(jload(LODO / "results.json")["per_device"]["M_bias_v2"])
    if variant == "C0_refit":
        return dict(jload(LODO / "results.json")["per_device"]["C0_refit"])
    raise KeyError(variant)


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------
def stage_validity(workers: int, rd: Path) -> dict:
    order = drv12.load_order()
    tr = v2.fold_train_tags(order, "pmos_L2_W5")
    tasks = [("sym_e1_l10|pmos_L2_W5", tr, 0, {}, 10.0, "sym_v2", 0.01),
             ("v1|pmos_L2_W5", tr, 0, {}, 0.0, "abs_v1", None)]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_task, tasks, chunksize=1)
        rows = pool.map(refit_task, [("v2_stored_rerun", stored_corr("sym_e1_l10", t), t, REFIT_OPTS)
                                     for t in ("nmos_L1_W1p6", "pmos_L2_W5")], chunksize=1)
    v1 = {}
    for name, _, corr, sec, _ in trained:
        variant = name.split("|")[0]
        ref = stored_corr(variant, "pmos_L2_W5")
        d = 0.0
        for (w, b), (w0, b0) in zip(corr.nets[0], ref.nets[0]):
            d = max(d, float(np.max(np.abs(np.asarray(w) - np.asarray(w0)))),
                    float(np.max(np.abs(np.asarray(b) - np.asarray(b0)))))
        v1[variant] = {"max_abs_weight_diff_member0": d, "train_s": sec}
    st = stored_rows("sym_e1_l10")
    v2r = {r["tag"]: {"rrms": r["rrms"], "stored": st[r["tag"]], "abs_diff": abs(r["rrms"] - st[r["tag"]])}
           for r in rows}
    out = {"V1": v1, "V2": v2r, "v1_max_weight_diff": max(v["max_abs_weight_diff_member0"] for v in v1.values()),
           "v2_max_rrms_diff": max(v["abs_diff"] for v in v2r.values()), "wall_s": round(time.time() - t0, 1)}
    out["pass"] = bool(out["v1_max_weight_diff"] == 0.0 and out["v2_max_rrms_diff"] <= 1e-9)
    jdump(out, rd / "validity.json")
    print(json.dumps({k: out[k] for k in ("v1_max_weight_diff", "v2_max_rrms_diff", "pass", "wall_s")}), flush=True)
    return out


# ---------------------------------------------------------------------------
# lodo (one variant)
# ---------------------------------------------------------------------------
def stage_lodo(variant: str, workers: int, rd: Path, smoke: bool) -> dict:
    from e1_lodo import aggregate
    inputs, eps, lam = m.VARIANTS[variant]
    if variant in m.STORED:
        raise ValueError(f"{variant} is a stored variant")
    recs = drv12.load_records()
    tags = sorted(recs)
    folds = SMOKE_FOLDS if smoke else tags
    hyper = {"steps": 200} if smoke else {}
    seeds = (0,) if smoke else v2.HYPER_V2["seeds"]
    opts = dict(REFIT_OPTS, heldout_nfev=20) if smoke else dict(REFIT_OPTS)
    order = drv12.load_order()
    names = [f"{variant}|{t}" for t in folds] + [f"full|{variant}"]
    train_of = {f"{variant}|{t}": v2.fold_train_tags(order, t) for t in folds}
    train_of[f"full|{variant}"] = v2.fold_train_tags(order, None)
    tasks = [(n, train_of[n], s, hyper, lam, inputs, eps) for n in names for s in seeds]
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_task, tasks, chunksize=1)
        t_train = time.time() - t0
        corrs, secs = merge(trained, names)
        print(f"lodo {variant}: training {len(tasks)} seed tasks {t_train:.0f} s", flush=True)
        t1 = time.time()
        rows = pool.map(refit_task, [(variant, corrs[f"{variant}|{t}"], t, opts) for t in folds], chunksize=1)
        t_refit = time.time() - t1
    per = {r["tag"]: r["rrms"] for r in rows}
    full18 = len(folds) == 18 and not smoke
    summary = aggregate(v2.select_primary_rows(per, tags)) if full18 else {
        "mean": float(np.mean(list(per.values())))}
    res = {"variant": variant, "inputs": inputs, "eps": eps, "lambda": lam, "smoke": smoke,
           "hyper": v2.HYPER_V2, "hyper_overrides": hyper, "seeds": list(seeds), "opts": opts, "folds": folds,
           "train_order": train_of, "per_device": per, "summary": summary,
           "heldout_rows": {r["tag"]: {k: v for k, v in r.items() if k != "params"} for r in rows},
           "train_seconds_per_member": secs,
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_seed_tasks": len(tasks)}}
    jdump(res, rd / "results.json")
    jdump({n: c.to_json() for n, c in corrs.items()}, rd / "corrections.json")
    jdump({r["tag"]: r["params"] for r in rows}, rd / "heldout_cards.json")
    print(json.dumps({"variant": variant, "summary": summary, "timing": res["timing"]}, indent=1, default=float),
          flush=True)
    return res


# ---------------------------------------------------------------------------
# inner (nested selection of lambda)
# ---------------------------------------------------------------------------
def stage_inner(workers: int, rd: Path, smoke: bool) -> dict:
    recs = drv12.load_records()
    tags = sorted(recs)
    outers = SMOKE_FOLDS if smoke else tags + ["full"]
    hyper = {"steps": 200} if smoke else {}
    opts = dict(REFIT_OPTS, heldout_nfev=20) if smoke else dict(REFIT_OPTS)
    order = drv12.load_order()
    pm = is_pmos_map()
    plan = {}
    tasks = []
    for o in outers:
        tr = v2.fold_train_tags(order, None if o == "full" else o)
        folds = m.inner_folds(tr, pm)
        plan[o] = {"train": tr, "inner_folds": folds}
        for j, held in enumerate(folds):
            for lam in m.GRID_LAMBDA:
                tasks.append((f"{o}|{j}|{lam:g}", m.inner_train(tr, held), 0, hyper, lam, "sym_v2", m.EPS_ON))
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        trained = pool.map(train_task, tasks, chunksize=1)
        t_train = time.time() - t0
        print(f"inner: training {len(tasks)} members {t_train:.0f} s", flush=True)
        corr = {name: c for name, _, c, _, _ in trained}
        secs = {name: s for name, _, _, s, _ in trained}
        rtasks = []
        for o in outers:
            for j, held in enumerate(plan[o]["inner_folds"]):
                for lam in m.GRID_LAMBDA:
                    key = f"{o}|{j}|{lam:g}"
                    rtasks += [(key, corr[key], t, opts) for t in held]
        t1 = time.time()
        rows = pool.map(refit_task, rtasks, chunksize=1)
        t_refit = time.time() - t1
    res = {"smoke": smoke, "grid_lambda": list(m.GRID_LAMBDA), "eps": m.EPS_ON, "seed": 0, "opts": opts,
           "hyper_overrides": hyper, "plan": plan, "outer": {},
           "timing": {"train_s": round(t_train, 1), "refit_s": round(t_refit, 1), "n_train": len(tasks),
                      "n_refit": len(rtasks)}, "train_seconds": secs}
    for o in outers:
        rr = {}
        for lam in m.GRID_LAMBDA:
            vals = {r["tag"]: r["rrms"] for r in rows
                    if r["label"].startswith(f"{o}|") and r["label"].endswith(f"|{lam:g}")}
            rr[f"{lam:g}"] = vals
        scores = {lam: m.inner_score(list(rr[f"{lam:g}"].values())) for lam in m.GRID_LAMBDA}
        n_scored = {f"{lam:g}": len(rr[f"{lam:g}"]) for lam in m.GRID_LAMBDA}
        res["outer"][o] = {"inner_rrms": rr, "scores": {f"{k:g}": v for k, v in scores.items()},
                           "n_scored": n_scored, "lambda_star": m.select_lambda(scores)}
    res["lambda_star"] = {o: res["outer"][o]["lambda_star"] for o in outers}
    res["complete"] = bool(all(n == len(plan[o]["train"]) for o in outers
                               for n in res["outer"][o]["n_scored"].values()))
    jdump(res, rd / "inner.json")
    with gzip.open(rd / "inner_corrections.json.gz", "wt") as fh:
        json.dump({k: c.to_json() for k, c in corr.items()}, fh)
    print(json.dumps({"lambda_star": res["lambda_star"], "complete": res["complete"], "timing": res["timing"]},
                     indent=1), flush=True)
    return res


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def variant_rows(variant: str) -> dict:
    if variant in m.STORED:
        return stored_rows(variant)
    return dict(jload(OUT / f"lodo_{variant}" / "results.json")["per_device"])


def variant_corr(variant: str, tag: str | None):
    if variant in m.STORED:
        return stored_corr(variant, tag)
    js = jload(OUT / f"lodo_{variant}" / "corrections.json")
    return v2.SymCorrection.from_json(js[f"{variant}|{tag}" if tag else f"full|{variant}"])


def neg_fractions(corr_of, tags, pens, order) -> dict:
    """(a) fold training grids (mean over folds), (b) held-out starting-card
    grid (mean over folds and pooled), (c) the full-data model on 18 grids."""
    tr_gm, tr_gds, ho_gm, ho_gds, ho_n_gm, ho_n_gds, ho_n = [], [], [], [], 0, 0, 0
    for t in tags:
        c = corr_of(t)
        a = v2.penalty_value_numpy(c, [pens[u] for u in v2.fold_train_tags(order, t)])
        b = v2.penalty_value_numpy(c, [pens[t]])
        tr_gm.append(a["frac_neg_gm"])
        tr_gds.append(a["frac_neg_gds"])
        ho_gm.append(b["frac_neg_gm"])
        ho_gds.append(b["frac_neg_gds"])
        ho_n_gm += round(b["frac_neg_gm"] * b["n_points"])
        ho_n_gds += round(b["frac_neg_gds"] * b["n_points"])
        ho_n += b["n_points"]
    full = v2.penalty_value_numpy(corr_of(None), [pens[t] for t in tags])
    return {"train_mean_gm": float(np.mean(tr_gm)), "train_mean_gds": float(np.mean(tr_gds)),
            "heldout_mean_gm": float(np.mean(ho_gm)), "heldout_mean_gds": float(np.mean(ho_gds)),
            "heldout_pooled_gm": ho_n_gm / max(ho_n, 1), "heldout_pooled_gds": ho_n_gds / max(ho_n, 1),
            "full_gm": full["frac_neg_gm"], "full_gds": full["frac_neg_gds"], "full_P": full["P"]}


def stage_decide(rd: Path) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    from e1_lodo import aggregate
    tags = sorted(drv12.load_records())
    order = drv12.load_order()
    val = jload(OUT / "validity" / "validity.json")
    inner = jload(OUT / "inner" / "inner.json")
    rows = {v: v2.select_primary_rows(variant_rows(v), tags) for v in m.VARIANTS}
    c0 = v2.select_primary_rows(stored_rows("C0_refit"), tags)
    lam_star = {t: float(inner["lambda_star"][t]) for t in tags}
    sel = {t: rows[m.variant_for_lambda(lam_star[t])][t] for t in tags}
    n_rows_ok = all(len(variant_rows(v)) == 18 for v in m.NEW)
    validity_ok = bool(val["pass"] and n_rows_ok and inner["complete"] and len(inner["lambda_star"]) == 19)
    gmr = lambda a, b: paired_ratio(a, b)       # noqa: E731
    primary = gmr(sel, rows["v1"])
    table = {}
    for v, r in list(rows.items()) + [("selected", sel), ("C0_refit", c0)]:
        table[v] = {**aggregate(r), "n_failed": int(sum(not np.isfinite(x) for x in r.values()))}
        if v != "v1":
            table[v]["gmr_vs_v1"] = gmr(r, rows["v1"])
        if v != "C0_refit":
            table[v]["gmr_vs_C0_refit"] = gmr(r, c0)
    contrasts = {}
    for change, pairs in m.CONTRASTS.items():
        contrasts[change] = []
        for a, b in pairs:
            g = gmr(rows[a], rows[b])
            contrasts[change].append({"method": a, "control": b, **g, "label": m.contrast_label(g)})
    named = sorted(c for c, lst in contrasts.items() if any(x["label"] == "cost" for x in lst))
    curve = [{"method": a, "control": b, **gmr(rows[a], rows[b]),
              "label": m.contrast_label(gmr(rows[a], rows[b]))} for a, b in m.LAMBDA_CURVE]
    pens = drv12.load_pens()
    negs = {}
    for v in m.VARIANTS:
        negs[v] = neg_fractions(lambda t, v=v: variant_corr(v, t), tags, pens, order)
    full_star = m.variant_for_lambda(float(inner["lambda_star"]["full"]))
    negs["selected"] = neg_fractions(
        lambda t: variant_corr(m.variant_for_lambda(lam_star[t]), t) if t else variant_corr(full_star, None),
        tags, pens, order)
    negs["card_only"] = v2.card_negative_fraction([pens[t] for t in tags])
    out = {"prereg": str(PREREG.relative_to(ROOT)),
           "validity": {"V1_V2": val, "V3_rows_18": n_rows_ok, "V3_inner_complete": inner["complete"],
                        "valid": validity_ok},
           "primary_gmr_selected_vs_v1": primary,
           **m.decide_outcome(validity_ok, primary),
           "gmr_selected_vs_C0_refit": gmr(sel, c0),
           "gmr_selected_vs_sym_e1_l10": gmr(sel, rows["sym_e1_l10"]),
           "lambda_star": inner["lambda_star"], "lambda_star_full": inner["lambda_star"]["full"],
           "selected_rows": sel, "variant_table": table, "contrasts": contrasts,
           "changes_named_costly": named, "lambda_curve": curve, "negative_fractions": negs,
           "inner_scores": {o: v["scores"] for o, v in inner["outer"].items()}}
    jdump(out, rd / "decision.json")
    print(json.dumps({"outcome": out["outcome"], "primary": {k: primary[k] for k in ("ratio", "ci_low", "ci_high", "wins")},
                      "lambda_star": inner["lambda_star"], "named": named,
                      "table": {k: (round(v["all_device_mean"], 5), round(v["combined"], 5)) for k, v in table.items()}},
                     indent=1, default=float), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("validity", "lodo", "inner", "decide"))
    ap.add_argument("--variant", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true",
                    help="2 outer folds, 200 steps, refit max_nfev 20, 1 seed (code check; not used for decisions)")
    args = ap.parse_args()
    base = OUT / "smoke" if args.smoke else OUT
    t0 = time.time()
    if args.stage == "validity":
        stage_validity(args.workers, base / "validity")
    elif args.stage == "lodo":
        stage_lodo(args.variant, args.workers, base / f"lodo_{args.variant}", args.smoke)
    elif args.stage == "inner":
        stage_inner(args.workers, base / "inner", args.smoke)
    else:
        stage_decide(base)
    print(f"stage {args.stage} done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
