#!/usr/bin/env python3
"""Report batch for exploration direction I7-1: a training-set statistic S1_lodo
(fraction of the training devices' weighted starting-card residual energy that a
shared bias function, estimated without the scored device, explains) as a
predictor of the held-out log gain y = ln GMR(M_bias, C0_refit) of the I3-2
bias-only correction. Step 2: retrospective calibration on 11 stored datasets
(outcomes already seen), frozen linear rule. Step 3: prospective test on 20 new
simulated datasets (GF180 / IHP, 2 replicates, 5 mixing levels lambda).

Reads stored files only (no NGSpice, no training):

* ``out/research/explore/I7_1/{prereg.json, prereg_amendment_2026-10-20.json,
  decision.json, retro/results.json}``
* ``out/research/explore/I7_1/new/<pdk>_i7_1_r<rep>_l<100 lambda>/results.json``

Every dataset aggregate (S1_lodo, S1_pooled, S2, y, GMR), P1, P2, the verdict,
the secondary statistics (OLS, rule MAE, permutation p, manipulation check,
per-PDK and fold-level Spearman) and the frozen rule are recomputed from the
per-fold values and checked against the stored values. The within-lambda
analysis (table 06) is post hoc and not pre-registered.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i7_1.py --date 2026-10-20
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, RED, DPI, EXP,  # noqa: E402
                                   load, rel, style)
from cryoml.explore.i7_1_transfer import ols, rule_predict, sign_accuracy, spearman, spearman_perm_p  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i7_1"
I7 = EXP / "I7_1"
PREREG = I7 / "prereg.json"
AMEND = I7 / "prereg_amendment_2026-10-20.json"
DEC = I7 / "decision.json"
RETRO = I7 / "retro/results.json"
NEW = I7 / "new"
PDKS = ("gf180", "ihp")
REPS = (0, 1)
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
STATS = ("S1_lodo", "S1_pooled", "S2")
PDK_COLOR = {"gf180": BLUE, "ihp": ORANGE}
PDK_LABEL = {"gf180": "GF180", "ihp": "IHP"}
REP_MARK = {0: "o", 1: "s"}
RETRO_LABEL = {"sky130_77K": "sky130 77 K", "ihp_meas_233K": "IHP meas 233 K", "ihp_meas_300K": "IHP meas 300 K",
               "ihp_meas_343K": "IHP meas 343 K", "ihp_meas_398K": "IHP meas 398 K",
               "tudelft_4K": "TU Delft 4.13 K", "tudelft_295K": "TU Delft 295 K", "gf180_c": "GF180 tier c",
               "ihp_c": "IHP tier c", "gf180_b": "GF180 tier b", "ihp_b": "IHP tier b"}
LABEL_OFFSET = {"gf180_b": (2, 16), "ihp_b": (4, 4), "ihp_c": (-4, 5), "gf180_c": (4, -9),
                "tudelft_295K": (4, 3), "sky130_77K": (4, -9), "ihp_meas_300K": (4, 4),
                "ihp_meas_343K": (-4, -9), "ihp_meas_233K": (-4, -9)}
P1_RHO, P2_MIN, MAX_LEFT_OUT = -0.50, 15, 4


def key_of(pdk: str, rep: int, lam: float) -> str:
    return f"{pdk}|{rep}|{lam}"


def dir_of(pdk: str, rep: int, lam: float) -> Path:
    return NEW / f"{pdk}_i7_1_r{rep}_l{int(round(100 * lam)):03d}"


def shade(color: str, lam: float) -> tuple:
    """PDK colour mixed with white: lambda 0 light, lambda 1 full colour."""
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(color))
    t = 0.72 * (1.0 - lam)
    return tuple(c * (1 - t) + t)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol


def dataset_summary(rows: list[dict]) -> dict:
    """Same aggregation as scripts/research/explore/i7_1_transfer.py dataset_summary."""
    w = np.array([r["n_held"] for r in rows], float)
    out = {"n_folds": len(rows), "n_held": int(w.sum())}
    for k in STATS + ("S3",):
        v = np.array([r.get(k, np.nan) for r in rows], float)
        ok = np.isfinite(v)
        out[k] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
    lr = np.array([x for r in rows for x in r["log_ratios"]], float)
    ok = np.isfinite(lr)
    out["y"] = float(lr[ok].mean())
    out["gmr"] = float(math.exp(out["y"]))
    out["n_left_out"] = int((~ok).sum())
    return out


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    return float(np.sum(x * y) / math.sqrt(np.sum(x ** 2) * np.sum(y ** 2)))


def demean(v: np.ndarray, groups: list) -> np.ndarray:
    """Residual of v after OLS on a categorical variable (subtract the group mean)."""
    out = np.array(v, float)
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        out[idx] = out[idx] - out[idx].mean()
    return out


def strat_perm_p(x, y, groups: list, n_perm: int = 10000, seed: int = 0) -> float:
    """One-sided permutation p for a negative residual Pearson r, y permuted within groups."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xr = demean(x, groups)
    obs = pearson(xr, demean(y, groups))
    rng = np.random.default_rng(seed)
    idx = {g: np.array([i for i, v in enumerate(groups) if v == g]) for g in sorted(set(groups), key=str)}
    hits = 0
    for _ in range(n_perm):
        yp = y.copy()
        for ii in idx.values():
            yp[ii] = y[rng.permutation(ii)]
        hits += pearson(xr, demean(yp, groups)) <= obs + 1e-15
    return float((hits + 1) / (n_perm + 1))


# ================================================================== loading and checks

def load_all() -> dict:
    dec, prereg, amend, retro = load(DEC), load(PREREG), load(AMEND), load(RETRO)
    assert not dec["smoke"] and dec["n_datasets"] == 20 and dec["excluded"] == []
    rule = amend["frozen_rule"]
    assert dec["frozen_rule"] == rule
    # frozen rule = OLS over the 11 retrospective datasets
    rds = retro["datasets"]
    assert len(rds) == 11
    rr = ols([v["S1_lodo"] for v in rds.values()], [v["y"] for v in rds.values()])
    for k in ("a", "b", "r2"):
        assert close(rr[k], rule[k]), k
    assert close(-rule["a"] / rule["b"], rule["zero_crossing_S1_lodo"])
    for n, v in rds.items():
        a = amend["step2_summary"]["per_dataset"][n]
        assert all(close(a[k], v[k]) for k in ("S1_lodo", "S1_pooled", "S2", "y", "gmr")), n
    # new datasets: recompute from per-fold values
    ds, folds = {}, {}
    for pdk in PDKS:
        for rep in REPS:
            for lam in LEVELS:
                key = key_of(pdk, rep, lam)
                res = load(dir_of(pdk, rep, lam) / "results.json")
                assert not res["smoke"] and res["pdk"] == pdk and res["replicate"] == rep and res["lambda"] == lam
                assert res["hyper_overrides"] == {} and res["n_folds"] == 16
                rows = res["per_fold"]
                s = dataset_summary(rows)
                st = dec["datasets"][key]
                for k in STATS + ("y", "gmr"):
                    assert close(s[k], st[k]), (key, k)
                    assert close(s[k], res["summary"][k]), (key, k)
                assert s["n_left_out"] == st["n_left_out"] == 0
                start = float(np.mean(list(res["C0_start"]["per_device"].values())))
                c0 = float(np.mean(list(res["C0_refit"]["per_device"].values())))
                assert close(start, st["start_mean"]) and close(c0, st["c0_refit_mean"])
                assert close(res["M_bias"]["vs_C0_refit"]["ratio"], s["gmr"], 1e-10), key
                s.update(pdk=pdk, rep=rep, lam=lam, start_mean=start, c0_mean=c0,
                         m_mean=res["M_bias"]["summary"]["all_device_mean"],
                         pred=rule_predict(rule, s["S1_lodo"]), S3_stored=st["S3"],
                         ci=(res["M_bias"]["vs_C0_refit"]["ci_low"], res["M_bias"]["vs_C0_refit"]["ci_high"]),
                         wins=res["M_bias"]["vs_C0_refit"]["wins"])
                s["pred_gain"] = s["pred"] < 0
                s["obs_gain"] = s["y"] < 0
                s["correct"] = bool(np.sign(s["pred"]) == np.sign(s["y"]))
                ds[key], folds[key] = s, rows
    keys = list(ds)
    xs = [ds[k]["S1_lodo"] for k in keys]
    ys = [ds[k]["y"] for k in keys]
    lam = [ds[k]["lam"] for k in keys]
    rho = spearman(xs, ys)
    acc = sign_accuracy(rule, xs, ys)
    assert close(rho, dec["P1"]["spearman_S1_lodo_vs_y"]) and dec["P1"]["threshold"] == P1_RHO
    assert acc["correct"] == dec["P2"]["correct"] == sum(ds[k]["correct"] for k in keys) == 16
    assert dec["P2"]["required"] == P2_MIN
    p1, p2 = rho <= P1_RHO, acc["correct"] >= P2_MIN
    verdict = "supported" if p1 and p2 else "partially supported" if p1 or p2 else "not supported"
    assert verdict == dec["verdict"] == "supported"
    sec = dec["secondary"]
    cor = sec["correlations"]
    mine = {}
    for k in STATS:
        kx = [ds[q][k] for q in keys]
        o = ols(kx, ys)
        fx = [r[k] for q in keys for r in folds[q]]
        fy = [r["y"] for q in keys for r in folds[q]]
        within = {q: spearman([r[k] for r in folds[q]], [r["y"] for r in folds[q]]) for q in keys}
        mine[k] = {"dataset_spearman": spearman(kx, ys), "ols": o, "fold_pooled": spearman(fx, fy),
                   "fold_ols": ols(fx, fy), "within": within}
        c = cor[k]
        assert close(mine[k]["dataset_spearman"], c["dataset_spearman"])
        assert close(mine[k]["fold_pooled"], c["fold_spearman_pooled"]) and c["fold_n"] == 320
        assert all(close(o[x], c["dataset_ols"][x]) for x in ("a", "b", "r2"))
        assert all(close(within[q], c["fold_spearman_within"][q]) for q in keys)
    assert close(mine["S1_pooled"]["dataset_spearman"], sec["P1_P2_with_S1_pooled"]["spearman"])
    assert close(mine["S2"]["dataset_spearman"], sec["P1_P2_with_S2"]["spearman"])
    mae = float(np.mean([abs(rule_predict(rule, x) - y) for x, y in zip(xs, ys)]))
    assert close(mae, sec["rule_mae"])
    perm = spearman_perm_p(xs, ys)
    assert close(perm, sec["perm_p_S1_lodo"])
    man = {"S1": spearman(lam, xs), "y": spearman(lam, ys)}
    assert close(man["S1"], sec["manipulation"]["spearman_lambda_S1_lodo"])
    assert close(man["y"], sec["manipulation"]["spearman_lambda_y"])
    per_pdk = {p: spearman([ds[k]["S1_lodo"] for k in keys if ds[k]["pdk"] == p],
                           [ds[k]["y"] for k in keys if ds[k]["pdk"] == p]) for p in PDKS}
    assert all(close(per_pdk[p], sec["per_pdk_dataset_spearman_S1_lodo"][p]) for p in PDKS)
    wrong = [k for k in keys if not ds[k]["correct"]]
    assert len(wrong) == 4 and all(ds[k]["pred"] > 0 and ds[k]["y"] < 0 for k in wrong)
    s3_pending = all(math.isnan(ds[k]["S3_stored"]) for k in keys)
    return {"dec": dec, "prereg": prereg, "amend": amend, "retro": retro, "rule": rule, "ds": ds, "folds": folds,
            "keys": keys, "rho": rho, "acc": acc, "mine": mine, "mae": mae, "perm": perm, "man": man,
            "per_pdk": per_pdk, "wrong": wrong, "s3_pending": s3_pending, "verdict": verdict}


def posthoc(d: dict) -> dict:
    """Post hoc, not pre-registered: does S1_lodo order datasets beyond lambda?"""
    ds, keys = d["ds"], d["keys"]
    x = np.array([ds[k]["S1_lodo"] for k in keys])
    y = np.array([ds[k]["y"] for k in keys])
    lam = [ds[k]["lam"] for k in keys]
    lam_pdk = [(ds[k]["lam"], ds[k]["pdk"]) for k in keys]
    within = {}
    for lv in LEVELS:
        idx = [i for i, v in enumerate(lam) if v == lv]
        within[lv] = {"n": len(idx), "spearman": spearman(x[idx], y[idx]), "pearson": pearson(x[idx], y[idx]),
                      "s1_range": (float(x[idx].min()), float(x[idx].max())),
                      "y_range": (float(y[idx].min()), float(y[idx].max())),
                      "order": [keys[i] for i in sorted(idx, key=lambda i: x[i])]}
    resid = {}
    for name, g, ncat in (("lambda", lam, 5), ("lambda x PDK", lam_pdk, 10)):
        xr, yr = demean(x, g), demean(y, g)
        o = ols(xr, yr)
        resid[name] = {"pearson": pearson(xr, yr), "spearman": spearman(xr, yr), "slope": o["b"],
                       "df": len(x) - ncat - 1, "perm_p": strat_perm_p(x, y, g), "xr": xr, "yr": yr,
                       "ncat": ncat}
    return {"within": within, "resid": resid,
            "mean_within_spearman": float(np.mean([w["spearman"] for w in within.values()]))}


# ================================================================== tables

def t_decision(d: dict) -> rf.Table:
    dec, mine, rule = d["dec"], d["mine"], d["rule"]
    o = mine["S1_lodo"]["ols"]
    rows = [
        {"item": "P1 (decides)", "cond": "Spearman(S1_lodo_D, y_D) over 20 datasets <= -0.50",
         "val": f"{d['rho']:.3f}", "res": "holds"},
        {"item": "P2 (decides)", "cond": "frozen-rule sign correct in >= 15 of 20",
         "val": f"{d['acc']['correct']}/20", "res": "holds"},
        {"item": "verdict", "cond": "supported if P1 and P2; partially supported if one; not supported if neither",
         "val": d["verdict"], "res": ""},
        {"item": "frozen rule (step 2)", "cond": "OLS over the 11 stored datasets, amendment before step 3",
         "val": f"y = {rule['a']:.6f} {rule['b']:+.6f} S1_lodo, R^2 {rule['r2']:.3f}, zero crossing "
                f"{rule['zero_crossing_S1_lodo']:.4f}", "res": "reported"},
        {"item": "prospective OLS", "cond": "OLS over the 20 new datasets",
         "val": f"y = {o['a']:.4f} {o['b']:+.3f} S1_lodo, R^2 {o['r2']:.3f}, zero crossing {o['threshold_S']:.4f}",
         "res": "reported"},
        {"item": "rule MAE", "cond": "mean |a + b S1_lodo_D - y_D| over 20 datasets", "val": f"{d['mae']:.3f}",
         "res": "reported"},
        {"item": "permutation p", "cond": "one-sided, 10000 permutations of y_D, default_rng(0)",
         "val": f"{d['perm']:.1e}", "res": "reported"},
        {"item": "per-PDK Spearman", "cond": "S1_lodo vs y over 10 datasets per PDK",
         "val": f"GF180 {d['per_pdk']['gf180']:.3f}, IHP {d['per_pdk']['ihp']:.3f}", "res": "reported"},
        {"item": "S1_pooled, S2", "cond": "dataset Spearman with y_D (P1 form)",
         "val": f"S1_pooled {mine['S1_pooled']['dataset_spearman']:.3f}, S2 {mine['S2']['dataset_spearman']:.3f}",
         "res": "reported"},
        {"item": "fold level, pooled", "cond": "Spearman(S_f, y_f) over 320 folds",
         "val": ", ".join(f"{k} {mine[k]['fold_pooled']:.3f}" for k in STATS), "res": "reported"},
        {"item": "manipulation check", "cond": "Spearman(lambda, S1_lodo_D), Spearman(lambda, y_D)",
         "val": f"{d['man']['S1']:.3f}, {d['man']['y']:.3f}", "res": "reported"},
        {"item": "S3", "cond": "inner leave-one-group-out post-hoc gain (secondary, rank test only)",
         "val": "pending (queue job I7_1_s3_new)" if d["s3_pending"] else "see decision.json", "res": ""},
    ]
    cols = [{"key": "item", "head": "item", "md": "item", "kind": "text"},
            {"key": "cond", "head": "pre-registered definition", "md": "pre-registered definition", "kind": "text"},
            {"key": "val", "head": "value", "md": "value", "kind": "text"},
            {"key": "res", "head": "evaluated", "md": "evaluated", "kind": "text"}]
    notes = [
        f"Pre-registration {rel(PREREG)} (commit d406939); amendment {rel(AMEND)} (frozen rule, commit 968d973, "
        "written after step 2 and before any step-3 truth set). Decision file "
        f"{rel(DEC)}; every value is recomputed here from the per-fold values in new/*/results.json and checked.",
        "y_D = ln GMR(M_bias, C0_refit) = mean over the 16 held-out targets of ln(RRMS_M_bias / RRMS_C0_refit); "
        "negative = gain. S_D = mean over the 16 leave-one-geometry-out folds of the statistic computed on the "
        "15 training records of the fold. 0 targets left out, 0 datasets excluded.",
    ]
    return rf.Table("01_table_i7_1_decision", "I7-1 prospective test (20 new simulated datasets): pre-registered "
                    "P1, P2, verdict and secondary statistics", cols, rows, notes)


def t_new(d: dict) -> rf.Table:
    rows = []
    for k in d["keys"]:
        s = d["ds"][k]
        rows.append({"pdk": PDK_LABEL[s["pdk"]], "rep": str(s["rep"]), "lam": f"{s['lam']:g}",
                     "s1": f"{s['S1_lodo']:+.3f}", "s1p": f"{s['S1_pooled']:+.3f}", "s2": f"{s['S2']:+.3f}",
                     "y": f"{s['y']:+.4f}", "gmr": f"{s['gmr']:.3f}", "pred": f"{s['pred']:+.4f}",
                     "pg": "gain" if s["pred_gain"] else "no gain", "og": "gain" if s["obs_gain"] else "no gain",
                     "ok": "yes" if s["correct"] else "NO", "c0": s["c0_mean"], "m": s["m_mean"]})
    cols = [{"key": "pdk", "head": "PDK", "md": "PDK", "kind": "text"},
            {"key": "rep", "head": "rep", "md": "replicate", "kind": "rtext"},
            {"key": "lam", "head": "lambda", "md": "lambda", "kind": "rtext"},
            {"key": "s1", "head": "S1_lodo", "md": "S1_lodo", "kind": "rtext", "group": "training statistics"},
            {"key": "s1p", "head": "S1_pooled", "md": "S1_pooled", "kind": "rtext", "group": "training statistics"},
            {"key": "s2", "head": "S2", "md": "S2", "kind": "rtext", "group": "training statistics"},
            {"key": "c0", "head": "C0_refit", "md": "C0_refit mean RRMS", "kind": "rrms",
             "group": "held-out mean RRMS", "family": True},
            {"key": "m", "head": "M_bias", "md": "M_bias mean RRMS", "kind": "rrms",
             "group": "held-out mean RRMS", "family": True},
            {"key": "y", "head": "y", "md": "y = ln GMR(M_bias, C0_refit)", "kind": "rtext", "group": "outcome"},
            {"key": "gmr", "head": "GMR", "md": "GMR(M_bias, C0_refit)", "kind": "rtext", "group": "outcome"},
            {"key": "pred", "head": "rule y", "md": "frozen-rule predicted y", "kind": "rtext",
             "group": "frozen rule", "family": True},
            {"key": "pg", "head": "predicted", "md": "predicted", "kind": "text", "group": "frozen rule",
             "family": True},
            {"key": "og", "head": "observed", "md": "observed", "kind": "text", "group": "frozen rule",
             "family": True},
            {"key": "ok", "head": "correct", "md": "sign correct", "kind": "text", "group": "frozen rule",
             "family": True}]
    notes = [
        "16 targets per dataset (8 nMOS, 8 pMOS geometries), 16 leave-one-geometry-out folds; truth = make_truth "
        "tier-a card x exp(e_d) x (1 + 0.01 N(0,1)), e_d = 0.10 (sqrt(lambda) s_polarity + sqrt(1 - lambda) u_d). "
        "The five lambda levels of one (PDK, replicate) share base cards, s, u_d and noise draws.",
        f"Frozen rule: predicted y = {d['rule']['a']:.6f} {d['rule']['b']:+.6f} S1_lodo; predicted gain iff "
        f"S1_lodo > {d['rule']['zero_crossing_S1_lodo']:.4f}. Sign correct in {d['acc']['correct']}/20. The 4 "
        "errors are datasets with predicted 'no gain' and an observed gain.",
    ]
    return rf.Table("03_table_i7_1_new_datasets", "I7-1 new simulated datasets: training statistics, held-out "
                    "outcome and frozen-rule prediction", cols, rows, notes)


def t_posthoc(d: dict, ph: dict) -> rf.Table:
    rows = []
    for lv, w in ph["within"].items():
        order = ", ".join(f"{PDK_LABEL[d['ds'][k]['pdk']]} r{d['ds'][k]['rep']}" for k in w["order"])
        rows.append({"what": f"within lambda = {lv:g}", "n": str(w["n"]), "sp": f"{w['spearman']:+.3f}",
                     "pe": f"{w['pearson']:+.3f}", "sl": "", "p": "",
                     "info": f"S1_lodo {w['s1_range'][0]:+.3f} to {w['s1_range'][1]:+.3f}; y {w['y_range'][0]:+.3f} "
                             f"to {w['y_range'][1]:+.3f}; order by S1_lodo: {order}"})
    rows.append({"what": "mean of the 5 within-lambda Spearman", "n": "20", "sp": f"{ph['mean_within_spearman']:+.3f}",
                 "pe": "", "sl": "", "p": "", "info": ""})
    for name, r in ph["resid"].items():
        rows.append({"what": f"residual after {name} ({r['ncat']} levels)", "n": "20", "sp": f"{r['spearman']:+.3f}",
                     "pe": f"{r['pearson']:+.3f}", "sl": f"{r['slope']:+.3f}", "p": f"{r['perm_p']:.4f}",
                     "info": f"df {r['df']}; S1_lodo and y each minus their {name} cell mean"})
    cols = [{"key": "what", "head": "subset or model", "md": "subset or model", "kind": "text"},
            {"key": "n", "head": "n", "md": "datasets", "kind": "rtext"},
            {"key": "sp", "head": "Spearman", "md": "Spearman(S1_lodo, y)", "kind": "rtext"},
            {"key": "pe", "head": "Pearson", "md": "Pearson(S1_lodo, y)", "kind": "rtext"},
            {"key": "sl", "head": "OLS slope", "md": "OLS slope of residual y on residual S1_lodo", "kind": "rtext"},
            {"key": "p", "head": "perm. p", "md": "one-sided permutation p (y permuted within cells)",
             "kind": "rtext"},
            {"key": "info", "head": "detail", "md": "detail", "kind": "text"}]
    notes = [
        "Post hoc, not pre-registered. Question: does S1_lodo order the datasets beyond the manipulated lambda? "
        "Within one lambda level the 4 datasets differ in PDK and replicate (base cards, error functions, noise).",
        "Residual rows: S1_lodo and y are each regressed on the categorical variable (OLS with one indicator per "
        "level = subtract the cell mean); the Pearson and Spearman correlations and the OLS slope are computed on "
        "the residuals. Permutation p: 10000 permutations of y within the cells, default_rng(0), statistic = "
        "residual Pearson r, one-sided (negative). The 'lambda x PDK' cells hold 2 datasets (the 2 replicates).",
    ]
    return rf.Table("06_table_i7_1_posthoc_within_lambda", "I7-1 post hoc (not pre-registered): S1_lodo vs y "
                    "within lambda levels and after removing lambda", cols, rows, notes)


def t_retro(d: dict) -> rf.Table:
    retro, rule = d["retro"], d["rule"]
    within = retro["correlations"]["S1_lodo"]["fold_spearman_within"]
    rows = []
    for n, v in retro["datasets"].items():
        p = rule_predict(rule, v["S1_lodo"])
        rows.append({"ds": RETRO_LABEL[n], "f": str(v["n_folds"]), "h": str(v["n_held"]),
                     "s1": f"{v['S1_lodo']:+.3f}", "s1p": f"{v['S1_pooled']:+.3f}", "s2": f"{v['S2']:+.3f}",
                     "y": f"{v['y']:+.4f}", "gmr": f"{v['gmr']:.3f}", "pred": f"{p:+.4f}",
                     "ok": "yes" if np.sign(p) == np.sign(v["y"]) else "NO", "w": f"{within[n]:+.2f}"})
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "f", "head": "folds", "md": "folds", "kind": "rtext"},
            {"key": "h", "head": "held-out", "md": "held-out devices", "kind": "rtext"},
            {"key": "s1", "head": "S1_lodo", "md": "S1_lodo", "kind": "rtext", "group": "training statistics"},
            {"key": "s1p", "head": "S1_pooled", "md": "S1_pooled", "kind": "rtext", "group": "training statistics"},
            {"key": "s2", "head": "S2", "md": "S2", "kind": "rtext", "group": "training statistics"},
            {"key": "y", "head": "y", "md": "y = ln GMR", "kind": "rtext", "group": "outcome", "family": True},
            {"key": "gmr", "head": "GMR", "md": "GMR", "kind": "rtext", "group": "outcome", "family": True},
            {"key": "pred", "head": "rule y", "md": "frozen-rule y (in-sample)", "kind": "rtext",
             "group": "frozen rule (in-sample)"},
            {"key": "ok", "head": "sign", "md": "sign correct", "kind": "text", "group": "frozen rule (in-sample)"},
            {"key": "w", "head": "fold Spearman", "md": "within-dataset fold Spearman(S1_lodo, y_f)",
             "kind": "rtext"}]
    c = retro["correlations"]
    notes = [
        f"Step 2, retrospective calibration on outcomes already seen (exploratory), {rel(RETRO)} (commit 968d973). "
        "The IHP measured temperatures share devices, as do the TU Delft temperatures; the 11 points are not "
        "independent.",
        f"Dataset Spearman with y: S1_lodo {c['S1_lodo']['dataset_spearman']:.3f}, S1_pooled "
        f"{c['S1_pooled']['dataset_spearman']:.3f}, S2 {c['S2']['dataset_spearman']:.3f}. The rule is the OLS line "
        f"over these 11 points (R^2 {rule['r2']:.3f}); in-sample sign correct 10/11, leave-one-dataset-out 10/11.",
    ]
    return rf.Table("08_table_i7_1_retro_datasets", "I7-1 step 2: the 11 stored datasets that calibrate the frozen "
                    "rule", cols, rows, notes)


# ================================================================== figures

def fig_scatter(d: dict, path: Path) -> list[str]:
    ds, rule, retro = d["ds"], d["rule"], d["retro"]["datasets"]
    fig, ax = plt.subplots(figsize=(10.5, 7.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    allx = [v["S1_lodo"] for v in retro.values()] + [v["S1_lodo"] for v in ds.values()]
    xs = np.linspace(min(allx) - 0.05, max(allx) + 0.05, 100)
    o = d["mine"]["S1_lodo"]["ols"]
    ax.plot(xs, rule["a"] + rule["b"] * xs, color=INK, lw=1.4,
            label=f"frozen rule (step 2): y = {rule['a']:.6f} {rule['b']:+.6f} S1_lodo")
    ax.plot(xs, o["a"] + o["b"] * xs, color=MUTED, lw=1.0, ls="--",
            label=f"OLS over the 20 new datasets (not a rule): slope {o['b']:.3f}, R^2 {o['r2']:.3f}")
    ax.axhline(0, color=INK, lw=0.7)
    t = rule["zero_crossing_S1_lodo"]
    ax.axvline(t, color=INK, lw=0.9, ls=":", label=f"zero crossing S1_lodo = {t:.4f}")
    for n, v in retro.items():
        ax.scatter(v["S1_lodo"], v["y"], color=GREY, s=36, zorder=3, edgecolor=BG, lw=0.6)
        off = LABEL_OFFSET.get(n, (4, 3))
        ax.annotate(RETRO_LABEL[n], (v["S1_lodo"], v["y"]), fontsize=7, color=MUTED, xytext=off,
                    textcoords="offset points", ha="right" if off[0] < 0 else "left")
    ax.scatter([], [], color=GREY, s=36, label="stored datasets (step 2, outcomes already seen)")
    for k, v in ds.items():
        ax.scatter(v["S1_lodo"], v["y"], color=shade(PDK_COLOR[v["pdk"]], v["lam"]), marker=REP_MARK[v["rep"]],
                   s=58, zorder=4, edgecolor=PDK_COLOR[v["pdk"]], lw=0.9)
    for k in d["wrong"]:
        v = ds[k]
        ax.scatter(v["S1_lodo"], v["y"], s=210, facecolor="none", edgecolor=RED, lw=1.4, zorder=5)
    for p in PDKS:
        for r in REPS:
            ax.scatter([], [], color=PDK_COLOR[p], marker=REP_MARK[r], s=50,
                       label=f"new {PDK_LABEL[p]} replicate {r} (light to dark: lambda 0 to 1)")
    ax.scatter([], [], s=150, facecolor="none", edgecolor=RED, lw=1.4,
               label="sign of the frozen rule wrong (4 of 20)")
    ax.set_xlabel("S1_lodo (training folds, dataset mean)", fontsize=9)
    ax.set_ylabel("y = ln GMR(M_bias, C0_refit)  (negative = held-out gain)", fontsize=9)
    ax.set_title(f"I7-1: S1_lodo vs held-out log gain. Prospective (20 new simulated datasets): Spearman "
                 f"{d['rho']:.3f} (P1 needs <= -0.50),\nfrozen-rule sign correct {d['acc']['correct']}/20 (P2 needs "
                 f">= 15); verdict {d['verdict']}", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(RETRO), rel(AMEND), rel(DEC)]


def fig_lambda(d: dict, path: Path) -> list[str]:
    ds = d["ds"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, k, lab, rho in ((axs[0], "S1_lodo", "S1_lodo (dataset mean)", d["man"]["S1"]),
                            (axs[1], "y", "y = ln GMR(M_bias, C0_refit)", d["man"]["y"])):
        style(ax)
        for p in PDKS:
            for r in REPS:
                sel = [v for v in ds.values() if v["pdk"] == p and v["rep"] == r]
                off = (-0.012 if p == "gf180" else 0.012) + (-0.004 if r == 0 else 0.004)
                ax.plot([v["lam"] + off for v in sel], [v[k] for v in sel], color=PDK_COLOR[p], lw=1.0,
                        ls="-" if r == 0 else "--", marker=REP_MARK[r], ms=6,
                        label=f"{PDK_LABEL[p]} replicate {r}")
        if k == "y":
            ax.axhline(0, color=INK, lw=0.7)
        else:
            ax.axhline(d["rule"]["zero_crossing_S1_lodo"], color=INK, lw=0.9, ls=":",
                       label=f"rule zero crossing {d['rule']['zero_crossing_S1_lodo']:.4f}")
        ax.set_xticks(LEVELS)
        ax.set_xlabel("lambda (expected shared fraction of the injected error variance)", fontsize=9)
        ax.set_ylabel(lab, fontsize=9)
        ax.set_title(f"Spearman(lambda, {k}) = {rho:.3f} over 20 datasets", fontsize=9.5,
                     color=INK, loc="left")
        ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle("I7-1 manipulation check: S1_lodo and the held-out log gain as functions of the mixing level "
                 "lambda (points offset horizontally by PDK and replicate)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)]


def fig_fold(d: dict, path: Path) -> list[str]:
    ds, keys = d["ds"], d["keys"]
    within = d["mine"]["S1_lodo"]["within"]
    rwithin = d["retro"]["correlations"]["S1_lodo"]["fold_spearman_within"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), dpi=DPI, gridspec_kw={"width_ratios": [20, 11]})
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    xs = np.arange(len(keys))
    ax.bar(xs, [within[k] for k in keys], color=[shade(PDK_COLOR[ds[k]["pdk"]], ds[k]["lam"]) for k in keys],
           edgecolor=[PDK_COLOR[ds[k]["pdk"]] for k in keys], lw=0.8, width=0.75)
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(d["rho"], color=RED, lw=1.0, ls="--",
               label=f"dataset-level Spearman(S1_lodo_D, y_D) = {d['rho']:.3f} (20 datasets)")
    ax.axhline(d["mine"]["S1_lodo"]["fold_pooled"], color=MUTED, lw=1.0, ls=":",
               label=f"pooled fold-level Spearman over 320 folds = {d['mine']['S1_lodo']['fold_pooled']:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{PDK_LABEL[ds[k]['pdk']]} r{ds[k]['rep']} lambda {ds[k]['lam']:g}" for k in keys],
                       fontsize=7, rotation=60, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("within-dataset Spearman(S1_lodo_f, y_f), 16 folds", fontsize=9)
    vals = [within[k] for k in keys]
    ax.set_title(f"New datasets: within-dataset fold-level Spearman positive in {sum(v > 0 for v in vals)}/20 "
                 f"(range {min(vals):.2f} to {max(vals):.2f}); bar shade = lambda", fontsize=9.5, color=INK,
                 loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0.01, 0.3))
    ax.set_axisbelow(True)
    ax = axs[1]
    style(ax)
    names = list(rwithin)
    rv = [rwithin[n] for n in names]
    ax.bar(np.arange(len(names)), rv, color=GREY, width=0.75)
    ax.axhline(0, color=INK, lw=0.7)
    rr = d["retro"]["correlations"]["S1_lodo"]
    ax.axhline(rr["dataset_spearman"], color=RED, lw=1.0, ls="--",
               label=f"dataset-level Spearman = {rr['dataset_spearman']:.3f} (11 datasets)")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([RETRO_LABEL[n] for n in names], fontsize=7, rotation=45, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Stored datasets (step 2): positive in {sum(v > 0 for v in rv)}/11", fontsize=9.5, color=INK,
                 loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(0.01, 0.3))
    ax.set_axisbelow(True)
    fig.suptitle("I7-1 fold level vs dataset level: within a dataset a fold with a higher training S1_lodo has a "
                 "smaller gain (positive Spearman); across datasets a higher S1_lodo goes with a larger gain "
                 "(negative Spearman)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC), rel(RETRO), "out/research/explore/I7_1/new/*/results.json (per_fold)"]


def fig_posthoc(d: dict, ph: dict, path: Path) -> list[str]:
    ds, keys = d["ds"], d["keys"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, name in zip(axs, ("lambda", "lambda x PDK")):
        style(ax)
        r = ph["resid"][name]
        for i, k in enumerate(keys):
            v = ds[k]
            ax.scatter(r["xr"][i], r["yr"][i], color=shade(PDK_COLOR[v["pdk"]], v["lam"]), marker=REP_MARK[v["rep"]],
                       s=50, edgecolor=PDK_COLOR[v["pdk"]], lw=0.8, zorder=3)
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_xlabel(f"S1_lodo minus its {name} cell mean", fontsize=9)
        ax.set_ylabel(f"y minus its {name} cell mean", fontsize=9)
        ax.set_title(f"after {name} ({r['ncat']} cells): Pearson {r['pearson']:+.3f}, Spearman "
                     f"{r['spearman']:+.3f},\nslope {r['slope']:+.3f}, permutation p {r['perm_p']:.4f}", fontsize=9,
                     color=INK, loc="left")
    for p in PDKS:
        for rep in REPS:
            axs[0].scatter([], [], color=PDK_COLOR[p], marker=REP_MARK[rep], s=40,
                           label=f"{PDK_LABEL[p]} replicate {rep} (shade = lambda)")
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("I7-1 post hoc, not pre-registered: residual relation of y on S1_lodo after removing the "
                 "manipulated lambda (left) and lambda x PDK (right)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)]


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)

    d = load_all()
    ph = posthoc(d)
    items, tables = [], []
    new_src = "out/research/explore/I7_1/new/*/results.json"

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(t_decision(d), "Table of the pre-registered P1, P2 and verdict with the secondary statistics (frozen "
              "rule, prospective OLS, rule MAE, permutation p, per-PDK Spearman, S1_pooled and S2, fold-level "
              "Spearman, manipulation check, S3 status)", [rel(PREREG), rel(AMEND), rel(DEC), new_src])
    src = fig_scatter(d, folder / "02_i7_1_scatter_S1_lodo.png")
    items.append(("02_i7_1_scatter_S1_lodo.png", "Scatter of S1_lodo vs y = ln GMR(M_bias, C0_refit): 11 stored "
                  "datasets (grey, labelled), 20 new datasets (GF180 blue, IHP orange, shade = lambda, circle / "
                  "square = replicate 0 / 1), frozen rule line, zero crossing, prospective OLS line, the 4 "
                  "datasets with a wrong rule sign circled", src))
    add_table(t_new(d), "Table of the 20 new datasets: PDK, replicate, lambda, S1_lodo, S1_pooled, S2, C0_refit and "
              "M_bias mean RRMS, y, GMR, frozen-rule prediction, predicted and observed gain, sign correct",
              [rel(DEC), new_src])
    src = fig_lambda(d, folder / "04_i7_1_lambda_manipulation.png")
    items.append(("04_i7_1_lambda_manipulation.png", "Manipulation check: S1_lodo vs lambda and y vs lambda per "
                  "(PDK, replicate)", [rel(DEC), new_src]))
    src = fig_fold(d, folder / "05_i7_1_fold_spearman.png")
    items.append(("05_i7_1_fold_spearman.png", "Within-dataset fold-level Spearman(S1_lodo_f, y_f) for the 20 new "
                  "datasets and the 11 stored datasets, with the dataset-level Spearman", src))
    add_table(t_posthoc(d, ph), "Table (post hoc, not pre-registered): Spearman and Pearson of S1_lodo vs y within "
              "each lambda level, and the residual correlation after removing lambda and lambda x PDK",
              [rel(DEC), new_src])
    src = fig_posthoc(d, ph, folder / "07_i7_1_posthoc_residual.png")
    items.append(("07_i7_1_posthoc_residual.png", "Post hoc, not pre-registered: residual y vs residual S1_lodo "
                  "after removing lambda (left) and lambda x PDK (right)", src))
    add_table(t_retro(d), "Table of step 2: the 11 stored datasets (S1_lodo, S1_pooled, S2, y, GMR, in-sample rule "
              "prediction, within-dataset fold Spearman)", [rel(RETRO), rel(AMEND)])

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i7_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. y = ln GMR(M_bias, C0_refit) over the held-out "
          "targets (strict paper RRMS after the held-out card refit); negative = gain.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    ds, rule, mine = d["ds"], d["rule"], d["mine"]
    wy = [ds[k]["y"] for k in d["wrong"]]
    wx = [ds[k]["S1_lodo"] for k in d["wrong"]]
    wl = ", ".join(f"{PDK_LABEL[ds[k]['pdk']]} r{ds[k]['rep']} lambda {ds[k]['lam']:g} (S1_lodo "
                   f"{ds[k]['S1_lodo']:+.3f}, y {ds[k]['y']:+.3f})" for k in d["wrong"])
    within = [mine["S1_lodo"]["within"][k] for k in d["keys"]]
    o = mine["S1_lodo"]["ols"]
    rl, rlp = ph["resid"]["lambda"], ph["resid"]["lambda x PDK"]
    wtxt = "; ".join(f"lambda {lv:g}: {w['spearman']:+.2f}" for lv, w in ph["within"].items())
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice, no training).", "",
              "## Contents", "",
              "- I7-1: a training-set statistic S1_lodo (fraction of the training devices' weighted starting-card "
              "residual energy explained by a shared function of polarity and normalized bias, estimated without "
              "the scored device) as a predictor of the held-out log gain y = ln GMR(M_bias, C0_refit) of the I3-2 "
              f"bias-only correction. Pre-registration `{rel(PREREG)}` (commit d406939); step 2 retrospective "
              f"calibration on 11 stored datasets (outcomes already seen) `{rel(RETRO)}`; frozen rule in amendment "
              f"`{rel(AMEND)}` (commit 968d973, before any step-3 data); step 3 prospective test on 20 new simulated "
              "datasets (GF180 / IHP x 2 replicates x lambda 0, 0.25, 0.5, 0.75, 1; 16 targets, 16 "
              f"leave-one-geometry-out folds each) `{rel(DEC)}`. Queue jobs I7_1_run_gf180_r0 (1847 s), "
              "I7_1_run_ihp_r0 (1855 s), I7_1_run_gf180_r1 (1743 s), I7_1_run_ihp_r1 (1936 s), I7_1_decide, "
              "I7_1_figs, 2026-10-20.",
              f"- Verdict: {d['verdict']}. P1: Spearman(S1_lodo_D, y_D) = {d['rho']:.3f} over 20 datasets "
              f"(threshold -0.50). P2: frozen-rule sign correct {d['acc']['correct']}/20 (required 15). Permutation "
              f"p for S1_lodo {d['perm']:.0e} (one-sided, 10000 permutations). Per PDK: GF180 "
              f"{d['per_pdk']['gf180']:.3f}, IHP {d['per_pdk']['ihp']:.3f}. S1_pooled and S2 give dataset Spearman "
              f"{mine['S1_pooled']['dataset_spearman']:.3f} and {mine['S2']['dataset_spearman']:.3f}.",
              f"- All 4 errors are datasets the rule predicted 'no gain' (S1_lodo below the zero crossing "
              f"{rule['zero_crossing_S1_lodo']:.4f}) that had a small gain: y between {min(wy):.3f} and "
              f"{max(wy):.3f}, S1_lodo between {min(wx):.3f} and {max(wx):.3f}: {wl}.",
              f"- Magnitude: the prospective OLS slope is {o['b']:.3f} (intercept {o['a']:.4f}, R^2 {o['r2']:.3f}) "
              f"against the frozen slope {rule['b']:.3f}; the rule's mean absolute error on y_D is {d['mae']:.3f}. "
              "The frozen line predicts the sign of y_D, not its magnitude.",
              f"- Manipulation check: Spearman(lambda, S1_lodo_D) = {d['man']['S1']:.3f}, Spearman(lambda, y_D) = "
              f"{d['man']['y']:.3f}.",
              f"- Fold level: within each of the 20 new datasets the fold-level Spearman(S1_lodo_f, y_f) is positive "
              f"({min(within):.2f} to {max(within):.2f}); the fold-level relation has the opposite sign of the "
              "dataset-level relation, as in the retrospective step (positive in 9 of 11 stored datasets). The "
              f"pooled fold-level Spearman over 320 folds is {mine['S1_lodo']['fold_pooled']:.3f}; it mixes "
              "between-dataset and within-dataset variation.",
              f"- Post hoc, not pre-registered (table 06, figure 07): within each lambda level (4 datasets) "
              f"Spearman(S1_lodo, y) is {wtxt} (mean {ph['mean_within_spearman']:+.2f}). After regressing S1_lodo "
              f"and y on lambda as a categorical variable, the residual Pearson r is {rl['pearson']:+.3f} (Spearman "
              f"{rl['spearman']:+.3f}, slope {rl['slope']:+.3f}, df {rl['df']}, within-lambda permutation p "
              f"{rl['perm_p']:.4f}); after lambda x PDK (10 cells of 2 replicates) r is {rlp['pearson']:+.3f} "
              f"(Spearman {rlp['spearman']:+.3f}, permutation p {rlp['perm_p']:.4f}).",
              "- S3 (inner leave-one-group-out post-hoc gain, secondary): pending; the queue jobs I7_1_s3_new, "
              "I7_1_retro_s3, I7_1_decide_s3 and I7_1_figs_s3 had not finished when this batch was generated. S3 "
              "does not enter P1, P2 or the frozen rule.",
              "- Caveat: the synthetic error of step 3 (pre-registration, step3_prospective.error_functions: e_d = "
              "0.10 (sqrt(lambda) s_polarity + sqrt(1 - lambda) u_d), smooth functions of (|Vgs|/VDD, |Vds|/VDD)) "
              "is of exactly the shared-plus-device form that S1_lodo measures, on the same normalized-bias inputs "
              "as M_bias. The prospective test checks the statistic under its own model. The evidence outside this "
              "model is the 11 retrospective datasets (7 measured: sky130 77 K, IHP measured 233, 300, 343, 398 K, "
              "TU Delft 4.13 K and 295 K; 4 simulated tiers b and c), whose outcomes were known when S1_lodo was "
              "defined and the rule was fitted.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    summary = [
        f"I7-1 verdict (prospective, 20 new simulated datasets): {d['verdict']}. P1 Spearman(S1_lodo, y) "
        f"{d['rho']:.3f} (<= -0.50), P2 sign {d['acc']['correct']}/20 (>= 15); permutation p {d['perm']:.0e}; "
        f"S1_pooled {mine['S1_pooled']['dataset_spearman']:.3f}, S2 {mine['S2']['dataset_spearman']:.3f}.",
        f"Prospective OLS slope {o['b']:.3f} vs frozen {rule['b']:.3f}, rule MAE {d['mae']:.3f}: sign predicted, "
        f"not magnitude. 4 errors: predicted no gain, small gain (y {min(wy):.3f} to {max(wy):.3f}).",
        f"Post hoc: within-lambda Spearman mean {ph['mean_within_spearman']:+.2f}; residual r after lambda "
        f"{rl['pearson']:+.3f} (perm. p {rl['perm_p']:.4f}), after lambda x PDK {rlp['pearson']:+.3f}. S3 pending.",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    for lv, w in ph["within"].items():
        print(f" lambda {lv:g}: Spearman {w['spearman']:+.3f} Pearson {w['pearson']:+.3f}")
    for name, r in ph["resid"].items():
        print(f" resid {name}: Pearson {r['pearson']:+.4f} Spearman {r['spearman']:+.4f} slope {r['slope']:+.4f} "
              f"perm p {r['perm_p']:.4f} df {r['df']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
