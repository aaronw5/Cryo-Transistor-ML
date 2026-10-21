#!/usr/bin/env python3
"""Report batch for exploration direction I10-2: training-only per-fold
selection of the class of the shared bias-only correction g (menu none = C0_refit,
lin, gam, mlp8, mlp16) on the stored I9-2 folds of the four primary datasets
(sky130 77 K, NIST 4 K, TU Delft 4.13 K, IHP measured 300 K).  Selectors S_loss
(inner post hoc log ratio), S_refit (inner held-out refits), S_gate (stored
fold-level S3_f), S_cv (stored I9-2 rows of the training devices; not strictly
nested).  Primary rule (in force): stop.  Amendment A1: secondary C1 report.

Reads stored files only; no fit, no training, no NGSpice:

* ``out/research/explore/I10_2/{prereg.json, prereg_amendment_A1.json, plan.json, decision.json}``
* ``out/research/explore/I10_2/inner/<dataset>/results.json`` (S_loss)
* ``out/research/explore/I10_2/refit/<dataset>/results.json`` (S_refit)
* ``out/research/explore/I9_2/<dataset>/results.json`` (held-out rows, through the driver)
* the stored S3_f files (I7-1 retro, I8-1 prediction_v2, I9-1 prediction)

Checks: the S_loss and S_refit fold scores are recomputed from the per-device
inner log ratios and the selections from the scores (``select_class``); the
S_gate and S_cv selections are recomputed from the stored S3_f and I9-2 rows;
every primary GMR (pooled and per dataset, Bonferroni and 95%), the labels, the
rule, the amendment A1 C1 variants, the selection counts, the agreement, the
TU Delft device bootstrap, the tie sensitivity, the S_cv full menu, the I8-2
dataset-level gate, the NIST room-temperature secondary and the oracle
references are recomputed from the per-device rows and compared with
decision.json; point estimates are also recomputed with plain numpy.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i10_2.py --date 2026-10-21
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA, VIOLET, RED, YELLOW,  # noqa: E402
                                   DPI, EXP, load, rel, style)
sys.path.insert(0, str(Path(__file__).resolve().parent / "explore"))
import i10_2_select as drv  # noqa: E402  (I10-2 driver: rows, S3, selectors, statistics)
from cryoml.explore.i10_2_select import (CONTROL, FULL_MENU, MARGIN, MENU, NONE, REFERENCE,  # noqa: E402
                                         SELECTORS, TIE, agreement, assemble, bonferroni_alpha,
                                         class_size, fold_scores, gate_select, gmr, oracle_classes,
                                         overall_decision, select_class, selection_counts, selector_label)

rf = pe.rf
plt = pe.plt

TOPIC = "i10_2"
I10 = EXP / "I10_2"
PREREG = I10 / "prereg.json"
AMEND = I10 / "prereg_amendment_A1.json"
PLAN = I10 / "plan.json"
DEC = I10 / "decision.json"
DS = ("sky130_77K", "nist_4K", "tudelft_4K", "ihp_meas_300K")
DS_LABEL = {"sky130_77K": "sky130 77 K", "nist_4K": "NIST 4 K", "tudelft_4K": "TU Delft 4.13 K",
            "ihp_meas_300K": "IHP 300 K"}
SUBJ = SELECTORS + ("fixed_mlp16",)
S_COLOR = {"S_loss": BLUE, "S_refit": VIOLET, "S_gate": ORANGE, "S_cv": AQUA, "fixed_mlp16": GREY}
S_MARK = {"S_loss": "o", "S_refit": "s", "S_gate": "D", "S_cv": "^", "fixed_mlp16": "v"}
C_COLOR = {NONE: "#d9d7d0", "lin": YELLOW, "gam": ORANGE, "mlp8": AQUA, "mlp16": BLUE}
C_SHORT = {NONE: "-", "lin": "L", "gam": "G", "mlp8": "8", "mlp16": "16"}
COMMITS = {"code": "d16c929", "prereg": "55c4cfc", "interim": "b05acc4", "amend_code": "9cb5c9e",
           "amendment": "7a93dfc"}
# queue jobs (out/research/done.txt): wall time in s
JOBS = {"I10_2_inner": 2372, "I10_2_refit_sky130_77K": 1384, "I10_2_refit_ihp_meas_300K": 1147,
        "I10_2_refit_tudelft_4K": 2017, "I10_2_refit_nist_4K": 3372, "I10_2_decide": 2}
VARIANT_LABEL = {"primary_cluster_bonferroni": "primary: TU Delft cluster bootstrap, Bonferroni",
                 "a_device_bootstrap_bonferroni": "(a) TU Delft device bootstrap, Bonferroni",
                 "b_cluster_95": "(b) TU Delft cluster bootstrap, 95%",
                 "ab_device_bootstrap_95": "(a+b) TU Delft device bootstrap, 95%"}


def close(a, b, tol: float = 1e-12) -> bool:
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def same(a: dict, b: dict) -> bool:
    keys = ("ratio", "ci_low", "ci_high")
    if not all(close(a[k], b[k]) for k in keys):
        return False
    for n, v in a.get("per_dataset", {}).items():
        w = b["per_dataset"][n]
        if not all(close(v[k], w[k]) for k in keys):
            return False
        if any(v[k] != w[k] for k in ("n", "n_used", "wins", "n_method_failed", "n_control_failed") if k in w):
            return False
    return True


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def nplog(ser: dict, ctl: dict) -> float:
    """Plain-numpy point estimate of ln GMR over the finite pairs."""
    t = sorted(ser)
    a = np.array([ser[k] for k in t], float)
    b = np.array([ctl.get(k, np.nan) for k in t], float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.mean(np.log(a[ok] / b[ok])))


# ================================================================== loading and checks

def load_all() -> dict:
    prereg, amend, dec, plan = load(PREREG), load(AMEND), load(DEC), load(PLAN)
    assert prereg["status"].upper().startswith("IN FORCE") and amend["status"].upper().startswith("IN FORCE")
    assert tuple(dec["primary_datasets"]) == DS and tuple(dec["menu"]) == MENU
    assert dec["complete"] and not dec["missing"] and dec["margin"] == MARGIN == 1.03 and dec["tie"] == TIE
    a_adj = bonferroni_alpha()
    assert close(dec["alpha_adjusted"], a_adj)
    assert dec["guard"]["commit"].startswith(COMMITS["prereg"])
    assert dec["amendment_guard"]["commit"].startswith(COMMITS["amendment"])
    assert all(dec["validity"][n]["pass"] for n in DS)
    datas = {n: drv.d92.load_records(n) for n in DS}
    rows = {n: drv.outer_rows(n) for n in DS}
    clusters = {n: d["clusters"] for n, d in datas.items() if d["clusters"] is not None}
    assert list(clusters) == ["tudelft_4K"]
    inner = {n: load(I10 / "inner" / n / "results.json") for n in DS}
    refit = {n: load(I10 / "refit" / n / "results.json") for n in DS}
    sel = {s: {} for s in SELECTORS}
    nfail_refit = {}
    for n in DS:
        d = datas[n]
        # S_loss: scores from the per-device inner log ratios, selections from the scores
        r = inner[n]
        assert not r["smoke"] and r["V_fold_pass"] and max(r["V_fold_lin_max_abs_coef_diff"].values()) == 0.0
        assert sorted(r["folds"]) == sorted(d["folds"])
        for f, v in r["folds"].items():
            tags = [t for part in v["inner_folds"] for t in part]
            sc, bl = fold_scores(v["per_device_log_ratio"], tags, r["start_rrms_finite"])
            assert all(close(sc[c], v["scores"][c]) for c in sc) and bl == v["blocked"], (n, f)
            assert select_class(sc, bl) == v["selected"] == r["selection"][f]
            assert all(select_class(sc, bl, tie=float(x)) == c for x, c in v["tie_sensitivity"].items())
        sel["S_loss"][n] = dict(r["selection"])
        # S_refit
        q = refit[n]
        assert not q["smoke"] and sorted(q["folds"]) == sorted(d["folds"])
        c0 = rows[n][CONTROL]
        nfail_refit[n] = sum(1 for x in q["rows"].values() if not np.isfinite(float(x["rrms"])))
        assert nfail_refit[n] == q["n_failed_refits"] == 0 and len(q["rows"]) == q["n_refits"]
        for f, v in q["folds"].items():
            logs = {c: {} for c in MENU[1:]}
            for c in MENU[1:]:
                for k in v["refit_inner"]:
                    for t in r["folds"][f]["inner_folds"][k]:
                        a, b = float(q["rows"][f"{c}|{f}|k{k}|{t}"]["rrms"]), c0.get(t, np.nan)
                        logs[c][t] = math.log(a / b) if (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) \
                            else float("nan")
            for c in logs:
                assert all(close(logs[c][t], v["per_device_log_ratio"][c][t]) for t in logs[c]), (n, f, c)
            ref_ok = {t: bool(np.isfinite(c0.get(t, np.nan))) for t in v["refit_tags"]}
            sc, bl = fold_scores(logs, v["refit_tags"], ref_ok)
            assert all(close(sc[c], v["scores"][c]) for c in sc) and bl == v["blocked"], (n, f)
            assert select_class(sc, bl) == v["selected"] == q["selection"][f]
        sel["S_refit"][n] = dict(q["selection"])
        # S_gate, S_cv
        s3, _ = drv.stored_s3(n)
        assert sorted(s3) == sorted(d["folds"])
        sel["S_gate"][n] = {f: gate_select(s3[f]) for f in d["folds"]}
        sel["S_cv"][n] = drv.cv_select(rows[n], MENU, d["train"], d["folds"])[0]
    for s in SELECTORS:
        assert sel[s] == dec["selections"][s], s
    series = {s: {n: assemble(sel[s][n], rows[n], datas[n]["fold"], datas[n]["excluded"]) for n in DS}
              for s in SELECTORS}
    # primary contrasts (Bonferroni) and 95%
    prim = {"vs_C0_refit": {}, "vs_mlp16": {}, "vs_S_gate": {}, "labels": {}}
    c95 = {"vs_C0_refit": {}, "vs_mlp16": {}, "vs_S_gate": {}}
    for s in SELECTORS:
        for key, alpha, out in ((a_adj, a_adj, prim), (0.05, 0.05, c95)):
            out["vs_C0_refit"][s] = gmr({n: (series[s][n], rows[n][CONTROL]) for n in DS}, clusters, alpha)
            out["vs_mlp16"][s] = gmr({n: (series[s][n], rows[n][REFERENCE]) for n in DS}, clusters, alpha)
            if s != "S_gate":
                out["vs_S_gate"][s] = gmr({n: (series[s][n], series["S_gate"][n]) for n in DS}, clusters, alpha)
        prim["labels"][s] = selector_label(prim["vs_C0_refit"][s]["per_dataset"], prim["vs_mlp16"][s])
        for k in ("vs_C0_refit", "vs_mlp16", "vs_S_gate"):
            if s in prim[k]:
                assert same(prim[k][s], dec["primary"][k][s]), (k, s)
                assert same(c95[k][s], dec["ci95"][k][s]), (k, s)
        assert prim["labels"][s] == dec["primary"]["labels"][s], s
        # plain numpy point estimates
        pt = math.exp(np.mean([nplog(series[s][n], rows[n][CONTROL]) for n in DS]))
        assert close(pt, prim["vs_C0_refit"][s]["ratio"], 1e-10), s
        pt = math.exp(np.mean([nplog(series[s][n], rows[n][REFERENCE]) for n in DS]))
        assert close(pt, prim["vs_mlp16"][s]["ratio"], 1e-10), s
    rule = overall_decision(prim["labels"], prim["vs_C0_refit"], prim["vs_S_gate"])
    assert rule == dec["rule"] and rule["decision"] == "stop"
    # amendment A1 secondary C1
    subjects = dict(series)
    subjects["fixed_mlp16"] = {n: rows[n][REFERENCE] for n in DS}
    a1 = {}
    for v, spec in drv.C1_VARIANTS.items():
        alpha = a_adj if spec["alpha"] == "bonferroni" else spec["alpha"]
        cl = clusters if spec["tudelft_clusters"] else None
        a1[v] = {}
        for s, ser in subjects.items():
            per = gmr({n: (ser[n], rows[n][CONTROL]) for n in DS}, cl, alpha)
            fail = [n for n, x in per["per_dataset"].items() if not (np.isfinite(x["ci_high"]) and x["ci_high"] <= MARGIN)]
            st = dec["amendment_A1_secondary_C1"][v][s]
            assert (not fail) == st["C1"] and fail == st["failing_datasets"], (v, s)
            for n in DS:
                assert all(close(per["per_dataset"][n][k], st["per_dataset"][n][k]) for k in ("ratio", "ci_low", "ci_high"))
            a1[v][s] = {"C1": not fail, "fail": fail, "per": per["per_dataset"], "pooled": per}
    for s in SELECTORS:      # the primary variant reproduces the primary C1
        assert a1["primary_cluster_bonferroni"][s]["fail"] == prim["labels"][s]["C1_failing_datasets"]
    # secondary
    sec = dec["secondary"]
    counts = {s: {n: selection_counts(sel[s][n], MENU) for n in DS} for s in SELECTORS}
    dcounts = {s: {n: selection_counts({t: sel[s][n][f] for t, f in datas[n]["fold"].items()}, MENU) for n in DS}
               for s in SELECTORS}
    assert counts == sec["selection_counts"] and dcounts == sec["selection_device_counts"]
    agr = {f"{a}|{b}": {n: agreement(sel[a][n], sel[b][n]) for n in DS}
           for i, a in enumerate(SELECTORS) for b in SELECTORS[i + 1:]}
    assert agr == sec["agreement"]
    devb = {s: gmr({n: (series[s][n], rows[n][CONTROL]) for n in DS}, None, a_adj) for s in SELECTORS}
    for s in SELECTORS:
        assert same(devb[s], sec["device_bootstrap_tudelft"][s]), s
    menu_cls = [c for c in MENU if c != NONE]
    fixed = {c: gmr({n: (rows[n][c], rows[n][CONTROL]) for n in DS}, clusters) for c in menu_cls}
    for c in menu_cls:
        assert same(fixed[c], sec["fixed_classes_vs_C0_refit_95"][c]), c
    orc = oracle_classes({n: {c: fixed[c]["per_dataset"][n]["ratio"] for c in menu_cls} for n in DS}, fixed)
    assert orc == sec["oracle"]
    orc_ser = {n: assemble({f: orc["per_dataset"][n] for f in datas[n]["folds"]}, rows[n], datas[n]["fold"],
                           datas[n]["excluded"]) for n in DS}
    refs = {}
    for s in SELECTORS:
        refs[s] = {"pooled": gmr({n: (series[s][n], rows[n][drv.d92_row(orc["pooled"])]) for n in DS}, clusters),
                   "per_dataset": gmr({n: (series[s][n], orc_ser[n]) for n in DS}, clusters)}
        assert same(refs[s]["pooled"], dec["post_hoc_references"][s]["vs_oracle_pooled_class"])
        assert same(refs[s]["per_dataset"], dec["post_hoc_references"][s]["vs_oracle_per_dataset_class"])
    full_sel = {n: drv.cv_select(rows[n], FULL_MENU, datas[n]["train"], datas[n]["folds"])[0] for n in DS}
    full_ser = {n: assemble(full_sel[n], rows[n], datas[n]["fold"], datas[n]["excluded"]) for n in DS}
    full = {"counts": {n: selection_counts(full_sel[n], FULL_MENU) for n in DS},
            "vs_C0_refit": gmr({n: (full_ser[n], rows[n][CONTROL]) for n in DS}, clusters),
            "vs_mlp16": gmr({n: (full_ser[n], rows[n][REFERENCE]) for n in DS}, clusters)}
    assert full["counts"] == sec["S_cv_full_menu"]["selection_counts"]
    assert same(full["vs_C0_refit"], sec["S_cv_full_menu"]["vs_C0_refit"])
    assert same(full["vs_mlp16"], sec["S_cv_full_menu"]["vs_mlp16"])
    gate_ds = {}
    for n in DS:
        s3d = float(np.mean(list(dec["S3"][n]["S3_f"].values())))
        gate_ds[n] = gate_select(s3d)
    assert gate_ds == sec["dataset_level_gate_I8_2"]["class"]
    gser = {n: assemble({f: gate_ds[n] for f in datas[n]["folds"]}, rows[n], datas[n]["fold"], datas[n]["excluded"])
            for n in DS}
    gds = {"vs_C0_refit": gmr({n: (gser[n], rows[n][CONTROL]) for n in DS}, clusters),
           "vs_mlp16": gmr({n: (gser[n], rows[n][REFERENCE]) for n in DS}, clusters)}
    for k in gds:
        assert same(gds[k], sec["dataset_level_gate_I8_2"][k])
    tie = {}
    for s in ("S_loss", "S_refit"):
        src = inner if s == "S_loss" else refit
        for x in drv.TIE_SENSITIVITY:
            sx = {n: {f: src[n]["folds"][f]["tie_sensitivity"][str(x)] for f in datas[n]["folds"]} for n in DS}
            ser = {n: assemble(sx[n], rows[n], datas[n]["fold"], datas[n]["excluded"]) for n in DS}
            tie.setdefault(s, {})[str(x)] = {
                "vs_C0_refit": gmr({n: (ser[n], rows[n][CONTROL]) for n in DS}, clusters, a_adj),
                "vs_mlp16": gmr({n: (ser[n], rows[n][REFERENCE]) for n in DS}, clusters, a_adj)}
    for x in drv.TIE_SENSITIVITY:
        sx = {n: drv.cv_select(rows[n], MENU, datas[n]["train"], datas[n]["folds"], tie=x)[0] for n in DS}
        ser = {n: assemble(sx[n], rows[n], datas[n]["fold"], datas[n]["excluded"]) for n in DS}
        tie.setdefault("S_cv", {})[str(x)] = {
            "vs_C0_refit": gmr({n: (ser[n], rows[n][CONTROL]) for n in DS}, clusters, a_adj),
            "vs_mlp16": gmr({n: (ser[n], rows[n][REFERENCE]) for n in DS}, clusters, a_adj)}
    for s, d in tie.items():
        for x, v in d.items():
            for k in v:
                assert same(v[k], sec["tie_sensitivity"][s][x][k]), (s, x, k)
    rt = drv.nist_rt_data()
    s3rt, _ = drv.stored_s3("nist_rt")
    rt_menu = (NONE, "poly3", "mlp16")
    rt_sel = {"S_gate": {f: gate_select(s3rt[f]) for f in rt["tags"]},
              "S_cv": drv.cv_select(rt["rows"], rt_menu, rt["train"], rt["tags"])[0]}
    nrt = {}
    for s, sl in rt_sel.items():
        ser = assemble(sl, rt["rows"], rt["fold"], rt["excluded"])
        nrt[s] = {"counts": selection_counts(sl, rt_menu),
                  "vs_C0_refit": gmr({"nist_rt": (ser, rt["rows"][CONTROL])}),
                  "vs_mlp16": gmr({"nist_rt": (ser, rt["rows"]["mlp16"])})}
        st = sec["nist_rt_I9_1"][s]
        assert nrt[s]["counts"] == st["selection_counts"] and same(nrt[s]["vs_C0_refit"], st["vs_C0_refit"]) \
            and same(nrt[s]["vs_mlp16"], st["vs_mlp16"])
    assert sec["nist_rt_I9_1"]["c0_lodo_equals_baselines"]
    # cost
    cost = {}
    for n in DS:
        tm = inner[n]["timing"]
        sel_tasks = {k: v for k, v in tm.items() if not k.startswith("check_lin")}
        chk = {k: v for k, v in tm.items() if k.startswith("check_lin")}
        assert len(chk) == len(datas[n]["folds"])
        rr = refit[n]["rows"]
        cost[n] = {"n_train": len(sel_tasks), "train_s": sum(v["seconds"] for v in sel_tasks.values()),
                   "n_check": len(chk), "check_s": sum(v["seconds"] for v in chk.values()),
                   "n_refit": len(rr), "refit_s": sum(float(v["runtime_s"]) for v in rr.values()),
                   "refit_failed": nfail_refit[n], "inner_wall_s": inner[n]["runtime_s"],
                   "refit_wall_s": refit[n]["runtime_s"], "refit_scope": refit[n]["refit_scope"],
                   "n_folds": len(datas[n]["folds"]),
                   "n_inner": sorted({len(v["inner_folds"]) for v in inner[n]["folds"].values()})}
        assert cost[n]["n_train"] == plan["datasets"][n]["n_inner_train_tasks"]
        assert cost[n]["n_refit"] == plan["datasets"][n]["n_inner_refits"]
    nfin = {n: sum(np.isfinite(v) for v in rows[n][CONTROL].values()) for n in DS}
    return {"prereg": prereg, "amend": amend, "dec": dec, "plan": plan, "datas": datas, "rows": rows,
            "sel": sel, "series": series, "prim": prim, "c95": c95, "rule": rule, "a1": a1,
            "counts": counts, "dcounts": dcounts, "agr": agr, "devb": devb, "fixed": fixed, "orc": orc,
            "refs": refs, "full": full, "gate_ds": gate_ds, "gds": gds, "tie": tie, "nrt": nrt, "cost": cost,
            "inner": inner, "refit": refit, "a_adj": a_adj, "nfin": nfin}


# ================================================================== tables

def t_decision(d: dict) -> rf.Table:
    p, rule = d["prim"], d["rule"]
    rows = []
    for s in SELECTORS:
        lab = p["labels"][s]
        g = p["vs_S_gate"].get(s)
        rows.append({"s": s, "c0": ci(p["vs_C0_refit"][s]), "m": ci(p["vs_mlp16"][s]), "g": ci(g) if g else "-",
                     "c1": "holds" if lab["C1_not_worse_than_C0_refit_all_datasets"] else
                     "fails (" + ", ".join(DS_LABEL[n] for n in lab["C1_failing_datasets"]) + ")",
                     "c2": "holds" if lab["C2_noninferior_to_mlp16_pooled"] else "fails",
                     "b": "yes" if lab["beats_mlp16_pooled"] else "no", "lab": lab["label"]})
    rows.append({"s": "PRIMARY RULE (in force)", "c0": "", "m": "", "g": "", "c1": "", "c2": "", "b": "",
                 "lab": f"{rule['decision']} (supported: none)"})
    cols = [{"key": "s", "head": "selector", "md": "selector", "kind": "text"},
            {"key": "c0", "head": "vs C0_refit", "md": "pooled GMR(S, C0_refit) [Bonferroni CI]", "kind": "rtext",
             "group": "pooled GMR [Bonferroni CI]"},
            {"key": "m", "head": "vs mlp16 (C2)", "md": "pooled GMR(S, mlp16) [Bonferroni CI] (C2)", "kind": "rtext",
             "group": "pooled GMR [Bonferroni CI]"},
            {"key": "g", "head": "vs S_gate", "md": "pooled GMR(S, S_gate) [Bonferroni CI]", "kind": "rtext",
             "group": "pooled GMR [Bonferroni CI]"},
            {"key": "c1", "head": "C1 (every dataset)", "md": "C1: per-dataset upper CI of GMR(S, C0_refit) <= 1.03",
             "kind": "text", "group": "criteria (margin 1.03)"},
            {"key": "c2", "head": "C2", "md": "C2: pooled upper CI of GMR(S, mlp16) <= 1.03", "kind": "text",
             "group": "criteria (margin 1.03)"},
            {"key": "b", "head": "beats mlp16", "md": "beats mlp16 (pooled upper < 1)", "kind": "text"},
            {"key": "lab", "head": "label", "md": "label", "kind": "text"}]
    notes = [
        f"PRIMARY RESULT. Pre-registration {rel(PREREG)} in force from commit {COMMITS['prereg']} (code "
        f"{COMMITS['code']}); decision {rel(DEC)}. Every value is recomputed here from the per-device rows and "
        "checked against decision.json.",
        "Held-out score of a device = the stored I9-2 row (held-out card refit with g frozen, strict paper RRMS) of "
        "the class selected in its outer fold from that fold's training devices; none = stored C0_refit row. Pooled "
        "GMR = exp(mean over the 4 datasets of the per-dataset mean log ratio); equal dataset weight; 20000 "
        "stratified resamples, TU Delft resampled by geometry group (7 clusters); level 1 - 0.05/11 (alpha 0.00455).",
        "Rule: supported = C1 and C2; partly supported = one of them; overall 'stop' if no selector is supported: "
        "class selection from training data is not supported on DEV, the deployed rule stays the I8-2 gate between "
        "C0_refit and M_bias, and no class selector goes to TEST.",
        "S_cv is not strictly nested: the stored row of a training device was produced with a g trained on folds "
        "that include fold f's starting-card records. The amendment A1 secondary C1 report is in table 07 and does "
        "not change any label or the rule.",
    ]
    return rf.Table("01_table_i10_2_decision", "I10-2 PRIMARY RESULT (in-force rule): stop. Class selection for the "
                    "shared bias-only correction from training data, per selector", cols, rows, notes)


def t_cost(d: dict) -> rf.Table:
    c, p = d["cost"], d["prim"]
    rows = []
    tot = {"train_s": 0.0, "refit_s": 0.0, "check_s": 0.0, "n_train": 0, "n_refit": 0}
    for n in DS:
        x = c[n]
        for k in tot:
            tot[k] += x[k]
        rows.append({"ds": DS_LABEL[n], "f": f"{x['n_folds']} / {','.join(map(str, x['n_inner']))}",
                     "tr": f"{x['n_train']} / {x['train_s'] / 3600:.2f}", "rf": f"{x['n_refit']} / {x['refit_s'] / 3600:.2f}",
                     "sc": x["refit_scope"], "fl": str(x["refit_failed"]), "lo": f"{x['train_s'] / 3600:.2f}",
                     "re": f"{(x['train_s'] + x['refit_s']) / 3600:.2f}",
                     "w": f"{x['inner_wall_s']:.0f} / {x['refit_wall_s']:.0f}",
                     "ck": f"{x['check_s']:.0f}"})
    rows.append({"ds": "total", "f": "90 outer", "tr": f"{tot['n_train']} / {tot['train_s'] / 3600:.2f}",
                 "rf": f"{tot['n_refit']} / {tot['refit_s'] / 3600:.2f}", "sc": "", "fl": "0",
                 "lo": f"{tot['train_s'] / 3600:.2f}", "re": f"{(tot['train_s'] + tot['refit_s']) / 3600:.2f}",
                 "w": f"{JOBS['I10_2_inner']} / {sum(v for k, v in JOBS.items() if 'refit' in k)}",
                 "ck": f"{tot['check_s']:.0f}"})
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "f", "head": "outer / inner folds", "md": "outer folds / inner folds per outer", "kind": "rtext"},
            {"key": "tr", "head": "trainings / CPU-h", "md": "inner trainings / CPU-h", "kind": "rtext",
             "group": "inner jobs"},
            {"key": "rf", "head": "refits / CPU-h", "md": "inner refits / CPU-h", "kind": "rtext", "group": "inner jobs"},
            {"key": "sc", "head": "refit scope", "md": "refit scope", "kind": "text", "group": "inner jobs"},
            {"key": "fl", "head": "failed", "md": "failed inner refits", "kind": "rtext", "group": "inner jobs"},
            {"key": "lo", "head": "S_loss", "md": "S_loss CPU-h", "kind": "rtext", "group": "new CPU-h per selector"},
            {"key": "re", "head": "S_refit", "md": "S_refit CPU-h (training + refits)", "kind": "rtext",
             "group": "new CPU-h per selector"},
            {"key": "w", "head": "wall s (inner / refit)", "md": "wall s (inner / refit)", "kind": "rtext"},
            {"key": "ck", "head": "V_fold CPU s", "md": "V_fold check CPU s", "kind": "rtext"}]
    s3 = d["dec"]["S3"]["sky130_77K"]["S3_hyper"]
    notes = [
        "CPU time = sum of the per-task seconds stored in inner/<dataset>/results.json (timing; 1 torch thread per "
        "task) and refit/<dataset>/results.json (rows runtime_s); 6 workers. Wall: the job's own runtime_s; the "
        f"queue times are I10_2_inner {JOBS['I10_2_inner']} s (4 datasets), refits " +
        ", ".join(f"{k.split('refit_')[1]} {v} s" for k, v in JOBS.items() if "refit" in k) +
        f", decide {JOBS['I10_2_decide']} s. Plan: 3.73 CPU-h training, 13.15 CPU-h refits.",
        "S_gate and S_cv: 0 new CPU-h in I10-2. S_gate reads the stored fold-level S3_f (M_bias recipe, seeds "
        f"{s3['seeds']}, {s3['steps']} steps; computed in I7-1 retro / I8-1). S_cv reads the stored I9-2 held-out "
        "rows of every menu class; at TEST it needs those rows (leave-one-group-out runs with refits of 4 classes) "
        "on each DEV set.",
        "V_fold: lin retrained on each outer training set reproduces the stored I9-2 lin fold model (max |coefficient "
        "difference| 0 on all 4 datasets); not part of a selector's cost.",
    ]
    return rf.Table("06_table_i10_2_cost", "I10-2 compute per selector and dataset (new CPU time; S_gate and S_cv "
                    "use stored values)", cols, rows, notes)


def t_a1(d: dict) -> rf.Table:
    a1 = d["a1"]
    rows = []
    for v in drv.C1_VARIANTS:
        r = {"v": VARIANT_LABEL[v]}
        for s in SUBJ:
            x = a1[v][s]
            t = x["per"]["tudelft_4K"]
            r[s] = ("pass" if x["C1"] else "fail: " + ", ".join(DS_LABEL[n].split()[0] for n in x["fail"])) + \
                f" | {t['ci_high']:.3f}"
        rows.append(r)
    ihp = {s: a1["ab_device_bootstrap_95"][s]["per"]["ihp_meas_300K"] for s in SUBJ}
    cols = [{"key": "v", "head": "C1 variant", "md": "C1 variant", "kind": "text"}] + \
           [{"key": s, "head": s, "md": f"{s}: C1 | TU Delft upper CI", "kind": "text",
             "group": "C1 (every dataset upper CI of GMR(S, C0_refit) <= 1.03) | TU Delft upper CI"} for s in SUBJ]
    notes = [
        "SECONDARY (amendment A1, in force from commit "
        f"{COMMITS['amendment']}, committed before I10_2_decide and before any S_loss / S_refit outer score). This "
        "table does not change the primary labels or the rule (table 01). The first row is the primary C1.",
        "TU Delft per-dataset GMR(S, C0_refit): " + "; ".join(
            f"{s} {a1['primary_cluster_bonferroni'][s]['per']['tudelft_4K']['ratio']:.3f}" for s in SUBJ) +
        ". IHP 300 K under (a+b): " + "; ".join(f"{s} {ci(ihp[s])}" for s in SUBJ) + ".",
        "Under (a+b) S_cv passes C1 on every dataset; fixed mlp16 (M_bias in every fold) fails C1 under every "
        "variant (IHP 300 K in all four, TU Delft in three). Fail entries name the failing datasets.",
    ]
    return rf.Table("07_table_i10_2_A1_secondary", "I10-2 SECONDARY (amendment A1): criterion C1 under the TU Delft "
                    "device bootstrap and at 95%, for the 4 selectors and fixed mlp16", cols, rows, notes)


def t_secondary(d: dict) -> rf.Table:
    rows = []
    c95, p = d["c95"], d["prim"]
    for s in SELECTORS:
        r = {"item": f"{s} per dataset vs C0_refit (95%)"}
        for n in DS:
            r[n] = ci(c95["vs_C0_refit"][s]["per_dataset"][n])
        rows.append(r)
    for s in SELECTORS:
        r = {"item": f"{s} per dataset vs mlp16 (95%)"}
        for n in DS:
            r[n] = ci(c95["vs_mlp16"][s]["per_dataset"][n])
        rows.append(r)
    for c in ("lin", "gam", "mlp8", "mlp16"):
        r = {"item": f"fixed {c} vs C0_refit (95%)"}
        for n in DS:
            r[n] = ci(d["fixed"][c]["per_dataset"][n])
        r["pooled"] = ci(d["fixed"][c])
        rows.append(r)
    for s in SELECTORS:
        rows[SELECTORS.index(s)]["pooled"] = ci(c95["vs_C0_refit"][s])
        rows[len(SELECTORS) + SELECTORS.index(s)]["pooled"] = ci(c95["vs_mlp16"][s])
    orc = d["orc"]
    rows.append({"item": "oracle class (post hoc)", **{n: orc["per_dataset"][n] for n in DS},
                 "pooled": orc["pooled"]})
    for s in SELECTORS:
        rows.append({"item": f"{s} vs per-dataset oracle (post hoc, 95%)",
                     **{n: ci(d["refs"][s]["per_dataset"]["per_dataset"][n]) for n in DS},
                     "pooled": ci(d["refs"][s]["per_dataset"])})
    full = d["full"]
    rows.append({"item": "S_cv full 13-option menu (95%): picks / pooled vs C0_refit",
                 **{n: ", ".join(f"{c} {k}" for c, k in full["counts"][n].items() if k) for n in DS},
                 "pooled": ci(full["vs_C0_refit"])})
    rows.append({"item": "I8-2 dataset-level gate (95%): class / pooled vs C0_refit",
                 **{n: d["gate_ds"][n] for n in DS}, "pooled": ci(d["gds"]["vs_C0_refit"])})
    for s in ("S_loss", "S_refit", "S_cv"):
        rows.append({"item": f"{s} tie sensitivity (Bonferroni), pooled vs C0_refit / vs mlp16",
                     **{n: "" for n in DS},
                     "pooled": "; ".join(f"TIE {x}: {v['vs_C0_refit']['ratio']:.3f} / {v['vs_mlp16']['ratio']:.3f}"
                                         for x, v in d["tie"][s].items())})
    for s in SELECTORS:
        rows.append({"item": f"{s} TU Delft device bootstrap (Bonferroni)", **{n: "" for n in DS},
                     "tudelft_4K": ci(d["devb"][s]["per_dataset"]["tudelft_4K"]), "pooled": ci(d["devb"][s])})
    for s, v in d["nrt"].items():
        rows.append({"item": f"NIST room temperature (I9-1) {s}, menu none/poly3/mlp16 (95%)",
                     **{n: "" for n in DS}, "nist_4K": f"mlp16 {v['counts']['mlp16']}/87 folds",
                     "pooled": f"vs C0_refit {ci(v['vs_C0_refit'])}"})
    cols = [{"key": "item", "head": "item", "md": "item", "kind": "text"}] + \
           [{"key": n, "head": DS_LABEL[n], "md": DS_LABEL[n], "kind": "rtext"} for n in DS] + \
           [{"key": "pooled", "head": "pooled", "md": "pooled", "kind": "rtext"}]
    notes = ["Secondary analyses of the pre-registration (descriptive, 95% unless marked; TU Delft cluster bootstrap "
             "unless marked). Oracle rows use held-out outcomes and cannot be deployed (D15). NIST room temperature "
             "row: the 'NIST 4 K' column holds the I9-1 NIST room-temperature selection count (87 folds).",
             "All values recomputed from the per-device rows and checked against decision.json (secondary, "
             "post_hoc_references)."]
    return rf.Table("08_table_i10_2_secondary", "I10-2 secondary analyses: per-dataset GMR, fixed classes, oracle "
                    "references, S_cv full menu, I8-2 gate, tie sensitivity, TU Delft device bootstrap, NIST room "
                    "temperature", cols, rows, notes)


# ================================================================== figures

def fig_forest(d: dict, path: Path) -> list[str]:
    p, c95, a1 = d["prim"], d["c95"], d["a1"]
    groups = list(DS) + ["pooled"]
    fig, ax = plt.subplots(figsize=(13, 11), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    y, ticks, labels = 0.0, [], []
    xlo, xhi = 0.5, 1.3
    for g in groups:
        ax.text(xlo * 1.01, y + 0.15, DS_LABEL.get(g, "pooled over 4 datasets (equal weight)"), fontsize=10,
                fontweight="bold", color=INK, va="bottom")
        y -= 0.35
        for s in SUBJ:
            if g == "pooled":
                if s == "fixed_mlp16":
                    b = a1["primary_cluster_bonferroni"][s]["pooled"]
                    w = d["fixed"]["mlp16"]
                else:
                    b, w = p["vs_C0_refit"][s], c95["vs_C0_refit"][s]
            else:
                b = a1["primary_cluster_bonferroni"][s]["per"][g]
                w = a1["b_cluster_95"][s]["per"][g]
            col = S_COLOR[s]
            ax.plot([w["ci_low"], min(w["ci_high"], xhi)], [y - 0.15, y - 0.15], color=col, lw=1.0, alpha=0.7, zorder=3)
            ax.errorbar(b["ratio"], y, xerr=[[b["ratio"] - b["ci_low"]], [b["ci_high"] - b["ratio"]]],
                        fmt=S_MARK[s], color=col, ms=6.5, capsize=3, lw=2.4, zorder=5,
                        mfc=col if s != "fixed_mlp16" else BG)
            fail = g != "pooled" and b["ci_high"] > MARGIN
            clip = "  (CI drawn to 1.3)" if b["ci_high"] > xhi else ""
            ax.annotate(f"{ci(b)}" + ("  C1 fails" if fail else "") + clip, (xhi * 1.02, y), fontsize=8,
                        color=RED if fail else INK, va="center", annotation_clip=False,
                        fontweight="bold" if fail else "normal")
            ticks.append(y)
            labels.append(s)
            y -= 0.42
        y -= 0.35
    ax.axvline(1.0, color=INK, lw=0.9)
    ax.axvline(MARGIN, color=RED, lw=1.0, ls="--")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xscale("log")
    xt = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.03, 1.1, 1.2, 1.3]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:g}" for t in xt], fontsize=8)
    ax.minorticks_off()
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(y + 0.2, 0.7)
    ax.set_xlabel("GMR(S, C0_refit) of held-out RRMS (log axis; < 1 = selector lower than no correction)", fontsize=9)
    hs = [plt.Line2D([], [], color=S_COLOR[s], marker=S_MARK[s], lw=2.4, mfc=S_COLOR[s] if s != "fixed_mlp16" else BG,
                     label=s if s != "fixed_mlp16" else "fixed mlp16 (M_bias in every fold; reference)") for s in SUBJ]
    hs += [plt.Line2D([], [], color=INK, lw=2.4, label="thick: Bonferroni CI (level 1 - 0.05/11; primary, decides C1)"),
           plt.Line2D([], [], color=INK, lw=1.0, alpha=0.55, label="thin, below: 95% CI (descriptive)"),
           plt.Line2D([], [], color=RED, lw=1.0, ls="--", label="C1 margin 1.03")]
    ax.legend(handles=hs, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.55, -0.06), ncol=3)
    fig.suptitle("I10-2: GMR of each selector against C0_refit per dataset and pooled (TU Delft: 7-cluster bootstrap). "
                 "C1 needs every per-dataset Bonferroni upper bound <= 1.03", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.1, right=0.76, top=0.95, bottom=0.12)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)] + [rel(EXP / "I9_2" / n / "results.json") for n in DS]


def fig_pooled(d: dict, path: Path) -> list[str]:
    p = d["prim"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    specs = (("vs_C0_refit", "GMR(S, C0_refit), pooled", SELECTORS, None),
             ("vs_mlp16", "GMR(S, mlp16), pooled (C2: upper <= 1.03)", SELECTORS, MARGIN),
             ("vs_S_gate", "GMR(S, S_gate), pooled", ("S_loss", "S_refit", "S_cv"), None))
    for ax, (key, ttl, subj, line) in zip(axs, specs):
        style(ax)
        ys = list(range(len(subj)))[::-1]
        for yy, s in zip(ys, subj):
            b = p[key][s]
            ax.errorbar(b["ratio"], yy, xerr=[[b["ratio"] - b["ci_low"]], [b["ci_high"] - b["ratio"]]],
                        fmt=S_MARK[s], color=S_COLOR[s], ms=7, capsize=4, lw=2.2)
            ax.annotate(ci(b), (b["ci_high"], yy + 0.22), fontsize=8, color=INK, ha="right")
        ax.axvline(1.0, color=INK, lw=0.9)
        if line:
            ax.axvline(line, color=RED, lw=1.0, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels(subj, fontsize=9)
        ax.set_ylim(-0.6, len(subj) - 0.3)
        lo = min(p[key][s]["ci_low"] for s in subj)
        hi = max(max(p[key][s]["ci_high"] for s in subj), line or 1.0)
        ax.set_xlim(lo - 0.03, hi + 0.03)
        ax.set_title(ttl, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("GMR (< 1 = selector lower)", fontsize=9)
    fig.suptitle("I10-2 primary pooled contrasts (11 at the Bonferroni level 1 - 0.05/11; equal dataset weight; "
                 "TU Delft cluster bootstrap); red dashed = C2 margin 1.03", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.14, wspace=0.3)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)]


def fig_selections(d: dict, path: Path) -> list[str]:
    sel, datas, counts = d["sel"], d["datas"], d["counts"]
    widths = [max(len(datas[n]["folds"]), 9) for n in DS]
    fig = plt.figure(figsize=(19, 7.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    top = fig.add_gridspec(1, 4, width_ratios=widths, left=0.05, right=0.99, top=0.88, bottom=0.6, wspace=0.12)
    bot = fig.add_gridspec(1, 4, left=0.05, right=0.99, top=0.47, bottom=0.14, wspace=0.12)
    cmap = ListedColormap([C_COLOR[c] for c in MENU])
    for j, n in enumerate(DS):
        ax = fig.add_subplot(top[0, j])
        folds = datas[n]["folds"]
        mat = np.array([[MENU.index(sel[s][n][f]) for f in folds] for s in SELECTORS])
        ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=len(MENU) - 0.5, aspect="auto")
        for i in range(mat.shape[0]):
            for k in range(mat.shape[1]):
                ax.text(k, i, C_SHORT[MENU[mat[i, k]]], ha="center", va="center", fontsize=6.5 if len(folds) > 20 else 8,
                        color=INK)
        ax.set_xticks(np.arange(len(folds) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(SELECTORS) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=BG, lw=1.2)
        ax.tick_params(which="both", length=0)
        ax.set_xticks([])
        ax.set_yticks(range(len(SELECTORS)))
        ax.set_yticklabels(SELECTORS if j == 0 else [], fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{DS_LABEL[n]}\n{len(folds)} outer folds ({'groups' if n == 'tudelft_4K' else 'devices'})",
                     fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("outer fold", fontsize=8, color=MUTED)
        # count table
        bx = fig.add_subplot(bot[0, j])
        cm = np.array([[counts[s][n][c] for c in MENU] for s in SELECTORS], float)
        bx.imshow(cm / len(folds), cmap=ListedColormap(plt.get_cmap("Greys")(np.linspace(0.02, 0.55, 64))),
                  vmin=0, vmax=1, aspect="auto")
        for i in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                bx.text(k, i, f"{int(cm[i, k])}", ha="center", va="center", fontsize=9, color=INK,
                        fontweight="bold" if cm[i, k] == cm[i].max() and cm[i, k] > 0 else "normal")
        bx.set_xticks(range(len(MENU)))
        bx.set_xticklabels([f"{c}\n({class_size(c)})" for c in MENU], fontsize=8)
        bx.set_yticks(range(len(SELECTORS)))
        bx.set_yticklabels(SELECTORS if j == 0 else [], fontsize=9)
        bx.tick_params(length=0)
        for sp in bx.spines.values():
            sp.set_visible(False)
        bx.set_title(f"{DS_LABEL[n]}: folds per class (of {len(folds)})\noracle class {d['orc']['per_dataset'][n]} "
                     "(post hoc)",
                     fontsize=9, color=INK, loc="left")
    hs = [Patch(facecolor=C_COLOR[c], label=f"{C_SHORT[c]} = {c} ({class_size(c)} trained values)") for c in MENU]
    fig.legend(handles=hs, loc="lower center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("I10-2: class selected in each outer fold from the fold's training devices (top) and selection "
                 "counts per selector and class (bottom; bold = most frequent)", fontsize=10.5, color=INK)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)] + [rel(I10 / "inner" / n / "results.json") for n in DS] + \
        [rel(I10 / "refit" / n / "results.json") for n in DS]


def fig_agreement(d: dict, path: Path) -> list[str]:
    sel, agr, cost, p = d["sel"], d["agr"], d["cost"], d["prim"]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), dpi=DPI, gridspec_kw={"width_ratios": [1.0, 1.35, 1.1]})
    fig.patch.set_facecolor(BG)
    # (a) confusion S_loss vs S_refit over all outer folds
    ax = axs[0]
    cm = np.zeros((len(MENU), len(MENU)), int)
    for n in DS:
        for f in d["datas"][n]["folds"]:
            cm[MENU.index(sel["S_loss"][n][f]), MENU.index(sel["S_refit"][n][f])] += 1
    ax.imshow(np.log1p(cm), cmap=ListedColormap(plt.get_cmap("Blues")(np.linspace(0.0, 0.7, 64))), aspect="equal")
    for i in range(len(MENU)):
        for k in range(len(MENU)):
            ax.text(k, i, str(cm[i, k]) if cm[i, k] else "", ha="center", va="center", fontsize=10, color=INK,
                    fontweight="bold" if i == k else "normal")
    ax.set_xticks(range(len(MENU)))
    ax.set_xticklabels(MENU, fontsize=9)
    ax.set_yticks(range(len(MENU)))
    ax.set_yticklabels(MENU, fontsize=9)
    ax.set_xlabel("S_refit selection", fontsize=9)
    ax.set_ylabel("S_loss selection", fontsize=9)
    ax.tick_params(length=0)
    tot = sum(len(d["datas"][n]["folds"]) for n in DS)
    ax.set_title(f"(a) S_loss vs S_refit, all {tot} outer folds: same pick {int(np.trace(cm))}/{tot}",
                 fontsize=9.5, color=INK, loc="left")
    # (b) agreement fraction per dataset
    ax = axs[1]
    style(ax)
    pairs = ("S_loss|S_refit", "S_loss|S_cv", "S_refit|S_cv")
    pc = {"S_loss|S_refit": VIOLET, "S_loss|S_cv": BLUE, "S_refit|S_cv": AQUA}
    w = 0.26
    for i, pr in enumerate(pairs):
        xs = np.arange(len(DS)) + (i - 1) * w
        vals = [agr[pr][n]["fraction"] for n in DS]
        ax.bar(xs, vals, width=w * 0.92, color=pc[pr], label=pr.replace("|", " vs "))
        for x, n in zip(xs, DS):
            a = agr[pr][n]
            ax.text(x, a["fraction"] + 0.015, f"{a['same']}/{a['n']}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels([DS_LABEL[n] for n in DS], fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("fraction of outer folds with the same class", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3)
    ax.set_title("(b) agreement of the multi-class selectors per dataset", fontsize=9.5, color=INK, loc="left")
    # (c) cost
    ax = axs[2]
    style(ax)
    tr = sum(cost[n]["train_s"] for n in DS) / 3600
    rr = sum(cost[n]["refit_s"] for n in DS) / 3600
    subj = ("S_gate", "S_cv", "S_loss", "S_refit")
    ys = np.arange(len(subj))[::-1]
    trv = {"S_gate": 0.0, "S_cv": 0.0, "S_loss": tr, "S_refit": tr}
    rfv = {"S_gate": 0.0, "S_cv": 0.0, "S_loss": 0.0, "S_refit": rr}
    ax.barh(ys, [trv[s] for s in subj], color=BLUE, height=0.55, label="inner training")
    ax.barh(ys, [rfv[s] for s in subj], left=[trv[s] for s in subj], color=VIOLET, height=0.55,
            label="inner held-out refits")
    for yy, s in zip(ys, subj):
        tot_s = trv[s] + rfv[s]
        ax.text(tot_s + 0.15, yy, f"{tot_s:.2f} CPU-h; GMR vs C0_refit {p['vs_C0_refit'][s]['ratio']:.3f}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(subj, fontsize=9)
    ax.set_xlim(0, (tr + rr) * 1.9)
    ax.set_xlabel("new CPU-h in I10-2 (4 datasets)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("(c) compute per selector (S_gate, S_cv: stored values)", fontsize=9.5, color=INK, loc="left")
    fig.suptitle("I10-2: S_loss (inner post hoc log ratio, no refit) against S_refit (inner held-out refits): "
                 "selections, agreement and cost", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.12, wspace=0.32)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)] + [rel(I10 / "inner" / n / "results.json") for n in DS] + \
        [rel(I10 / "refit" / n / "results.json") for n in DS]


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = load_all()
    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(t_decision(d), "Table of the primary result (in-force rule): per selector the pooled GMR against "
              "C0_refit, mlp16 and S_gate with Bonferroni CIs, criteria C1 and C2, label, and the overall rule "
              "output 'stop'", [rel(PREREG), rel(DEC)])
    items.append(("02_i10_2_forest_vs_C0_refit.png", "Forest plot of GMR(S, C0_refit) per selector and dataset and "
                  "pooled, Bonferroni CI (thick) and 95% CI (thin), fixed mlp16 as reference, C1 margin 1.03",
                  fig_forest(d, folder / "02_i10_2_forest_vs_C0_refit.png")))
    items.append(("03_i10_2_pooled_contrasts.png", "Pooled primary contrasts: GMR(S, C0_refit), GMR(S, mlp16) "
                  "with the C2 margin, GMR(S, S_gate), Bonferroni CIs",
                  fig_pooled(d, folder / "03_i10_2_pooled_contrasts.png")))
    items.append(("04_i10_2_selections.png", "Class selected in every outer fold per selector and dataset, and "
                  "selection counts per class", fig_selections(d, folder / "04_i10_2_selections.png")))
    items.append(("05_i10_2_loss_vs_refit_agreement.png", "S_loss vs S_refit: confusion of the selections over "
                  "90 outer folds, same-pick fraction per dataset for the multi-class selector pairs, and new CPU-h "
                  "per selector", fig_agreement(d, folder / "05_i10_2_loss_vs_refit_agreement.png")))
    add_table(t_cost(d), "Table of compute: inner trainings and refits, CPU-h per selector and dataset, wall times",
              [rel(I10 / "inner" / n / "results.json") for n in DS] + [rel(I10 / "refit" / n / "results.json")
                                                                      for n in DS] + [rel(PLAN)])
    add_table(t_a1(d), "Table of the amendment A1 secondary C1 report (separately labelled; does not change the "
              "primary result): C1 per variant for the 4 selectors and fixed mlp16, with the TU Delft upper CI",
              [rel(AMEND), rel(DEC)])
    add_table(t_secondary(d), "Table of the secondary analyses: per-dataset GMR at 95%, fixed classes, post hoc "
              "oracle references, S_cv full menu, I8-2 dataset-level gate, tie sensitivity, TU Delft device "
              "bootstrap, NIST room temperature", [rel(DEC)])

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i10_2.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. GMR = geometric mean ratio of held-out strict paper "
          "RRMS (stored I9-2 rows of the selected class); < 1 = the first method is lower.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    p, a1, cost, cnt = d["prim"], d["a1"], d["cost"], d["counts"]
    tr = sum(cost[n]["train_s"] for n in DS) / 3600
    rr = sum(cost[n]["refit_s"] for n in DS) / 3600
    ag = d["agr"]["S_loss|S_refit"]
    same_n = sum(ag[n]["same"] for n in DS)
    tot_n = sum(ag[n]["n"] for n in DS)

    def cstr(s, n):
        return ", ".join(f"{c} {k}" for c, k in cnt[s][n].items() if k)

    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no fit, no training, no NGSpice).", "",
              "## Contents", "",
              "- I10-2: selection of the class of the shared bias-only correction g (menu none = C0_refit, lin, gam, "
              "mlp8, mlp16) in each stored I9-2 outer fold from that fold's training devices only, on sky130 77 K "
              "(12 folds), NIST 4 K (53), TU Delft 4.13 K (7 geometry groups), IHP measured 300 K (18). Held-out "
              "score = stored I9-2 row of the selected class. Pre-registration "
              f"`{rel(PREREG)}` (in force from {COMMITS['prereg']}, code {COMMITS['code']}); amendment A1 "
              f"`{rel(AMEND)}` (in force from {COMMITS['amendment']}, code {COMMITS['amend_code']}); decision "
              f"`{rel(DEC)}`. Queue jobs: " + ", ".join(f"{k} ({v} s)" for k, v in JOBS.items()) + ".",
              f"- PRIMARY (in-force rule): {d['rule']['decision']}; no selector is supported. Labels: " +
              "; ".join(f"{s} {p['labels'][s]['label']} (C1 fails on " +
                        ", ".join(DS_LABEL[n] for n in p['labels'][s]['C1_failing_datasets']) +
                        f"; C2 {'holds' if p['labels'][s]['C2_noninferior_to_mlp16_pooled'] else 'fails'})"
                        for s in SELECTORS) + ".",
              "- Pooled GMR (Bonferroni CI) vs C0_refit / vs mlp16 / vs S_gate: " +
              "; ".join(f"{s} {ci(p['vs_C0_refit'][s])} / {ci(p['vs_mlp16'][s])}" +
                        (f" / {ci(p['vs_S_gate'][s])}" if s in p["vs_S_gate"] else "") for s in SELECTORS) + ".",
              "- TU Delft 4.13 K GMR(S, C0_refit) (Bonferroni, cluster): " +
              "; ".join(f"{s} {ci(p['vs_C0_refit'][s]['per_dataset']['tudelft_4K'])}" for s in SELECTORS) +
              f". IHP 300 K S_refit {ci(p['vs_C0_refit']['S_refit']['per_dataset']['ihp_meas_300K'])}.",
              "- Selections (outer folds): " + " | ".join(
                  f"{s}: " + "; ".join(f"{DS_LABEL[n]} {cstr(s, n)}" for n in DS) for s in SELECTORS) + ".",
              f"- S_loss vs S_refit: same class in {same_n}/{tot_n} outer folds (" +
              ", ".join(f"{DS_LABEL[n]} {ag[n]['same']}/{ag[n]['n']}" for n in DS) + f"). New compute: S_loss "
              f"{tr:.2f} CPU-h (inner training), S_refit {tr + rr:.2f} CPU-h (training + {rr:.2f} CPU-h refits), "
              "S_gate and S_cv 0 (stored values). 0 failed inner refits; V_fold difference 0 on all 4 datasets.",
              "- SECONDARY (amendment A1; does not change the primary result): C1 under (a+b) TU Delft device "
              "bootstrap at 95%: " + "; ".join(
                  f"{s} {'pass' if a1['ab_device_bootstrap_95'][s]['C1'] else 'fail (' + ', '.join(DS_LABEL[n] for n in a1['ab_device_bootstrap_95'][s]['fail']) + ')'}"
                  for s in SUBJ) + ". Fixed mlp16 fails C1 under every variant.",
              "- S_cv is not strictly nested (D9). Oracle references are post hoc (D15).",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    summary = [
        "I10-2 class selection for g from training data (menu none / lin / gam / mlp8 / mlp16, 4 datasets, 11 "
        "Bonferroni contrasts): PRIMARY rule 'stop', no selector supported. S_loss / S_refit / S_cv partly supported "
        f"(C2 holds; vs mlp16 {p['vs_mlp16']['S_loss']['ratio']:.3f} / {p['vs_mlp16']['S_refit']['ratio']:.3f} / "
        f"{p['vs_mlp16']['S_cv']['ratio']:.3f}); S_gate not supported ({ci(p['vs_mlp16']['S_gate'])}).",
        "C1 fails on TU Delft 4.13 K for every selector (S_loss "
        f"{ci(p['vs_C0_refit']['S_loss']['per_dataset']['tudelft_4K'])}, 7-cluster bootstrap); S_refit also on "
        f"IHP 300 K ({p['vs_C0_refit']['S_refit']['per_dataset']['ihp_meas_300K']['ratio']:.3f}). gam selected on "
        "sky130 and NIST by S_loss / S_refit / S_cv.",
        "Secondary A1: under the TU Delft device bootstrap at 95% S_cv passes C1; fixed mlp16 fails C1 under every "
        f"variant. S_loss vs S_refit same pick {same_n}/{tot_n} folds; S_refit cost {tr + rr:.1f} CPU-h vs S_loss "
        f"{tr:.1f}.",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    print("cost", {n: {k: round(v, 1) if isinstance(v, float) else v for k, v in cost[n].items()} for n in DS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
