#!/usr/bin/env python3
"""Report batch for exploration direction I9-2: function-class ablation of the
shared bias-only correction g (I = I_card x exp(g)) on the primary pool of four
measured datasets (sky130 77 K, NIST 4 K, TU Delft 4.13 K, IHP measured 300 K).
11 classes against the reference mlp16 (= stored I3-2 M_bias), pooled
equal-weight GMR with Bonferroni CIs (level 1 - 0.05/11), margin 1.03,
per-dataset limit 1.05, reference replicate mlp16_rep (seeds 5-9), learning
curve (mlp16, gam, mlp8) on TU Delft 4.13 K and NIST 4 K.

Reads stored files only; no fit, no training:

* ``out/research/explore/I9_2/{prereg.json, plan.json, validity.json, decision.json}``
* ``out/research/explore/I9_2/<dataset>/results.json`` (4 datasets), ``replicate/results.json``
* ``out/research/explore/I9_2/lc_{tudelft_4K, nist_4K}/results.json``
* ``out/research/explore/I9_2/examples/examples.json`` (NGSpice re-simulation of stored
  cards for 2 NIST 4 K devices, ``scripts/research/explore/i9_2_examples.py``)
* the stored source rows (I3-2, I4-1, I3-2 F2, I5-1, I6-1, I8-1) through the
  driver's ``stored_rows`` (check V4)

Checks: the stored rows used in I9-2 equal the source files; every pooled and
per-dataset GMR (vs mlp16 at the Bonferroni level and 95%, vs C0_refit, cluster
bootstrap, replicate) is recomputed from the per-device rows with
``pooled_ratio`` and compared with decision.json; the point estimates are also
recomputed with plain numpy; the labels, Q1, Q2, the margin-sensitivity labels,
the gain fractions, the mean final training losses and the learning-curve
summaries are recomputed and compared; validity V1 and the example RRMS are
checked.  Items marked "post hoc" are not pre-registered.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i9_2.py --date 2026-10-21
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA, VIOLET, RED, DPI,  # noqa: E402
                                   EXP, load, rel, style)
import plot_explore_i8_1 as p81  # noqa: E402  (I-V colors, comparison helpers)
sys.path.insert(0, str(Path(__file__).resolve().parent / "explore"))
import i9_2_classes as drv  # noqa: E402  (stored rows, records, folds)
from cryoml.explore.i9_2_classes import (CLASS_SPECS, CONTRASTS, LC_SIZES, MARGIN, PER_DATASET_MAX,  # noqa: E402
                                         REFERENCE, REPLICATE, bonferroni_alpha, decide, lc_classes,
                                         n_params, n_params_member, pooled_ratio)

rf = pe.rf
plt = pe.plt

TOPIC = "i9_2"
I9 = EXP / "I9_2"
PREREG = I9 / "prereg.json"
PREREG_MD = I9 / "prereg.md"
PLAN = I9 / "plan.json"
VALID = I9 / "validity.json"
DEC = I9 / "decision.json"
REP = I9 / "replicate/results.json"
EXAMPLES = I9 / "examples/examples.json"
DS = ("sky130_77K", "nist_4K", "tudelft_4K", "ihp_meas_300K")
DS_LABEL = {"sky130_77K": "sky130 77 K", "nist_4K": "NIST 4 K", "tudelft_4K": "TU Delft 4.13 K",
            "ihp_meas_300K": "IHP 300 K"}
DS_N = {"sky130_77K": 12, "nist_4K": 53, "tudelft_4K": 45, "ihp_meas_300K": 18}
DS_COLOR = {"sky130_77K": BLUE, "nist_4K": ORANGE, "tudelft_4K": AQUA, "ihp_meas_300K": VIOLET}
DS_MARK = {"sky130_77K": "o", "nist_4K": "s", "tudelft_4K": "^", "ihp_meas_300K": "D"}
LC_DS = ("tudelft_4K", "nist_4K")
LC_COLOR = {"mlp16": BLUE, "gam": ORANGE, "mlp8": AQUA}
COMMITS = {"draft": "a96ba07", "code": "dea00cc", "prereg": "02c6996", "fix": "a864f77", "outputs": "09f60c9"}
# queue jobs (out/research/done.txt, after the requeue at 03:35): wall time in s
JOBS = {"I9_2_validity": 31, "I9_2_run_sky130_77K": 186, "I9_2_run_nist_4K": 3282, "I9_2_run_tudelft_4K": 2330,
        "I9_2_run_ihp_meas_300K": 492, "I9_2_replicate": 1429, "I9_2_decide": 1, "I9_2_lc_tudelft_4K": 2923,
        "I9_2_lc_nist_4K": 3701, "I9_2_figs": 2}
ALL_CLASSES = tuple(CLASS_SPECS)       # 11 contrasts + mlp16 + mlp16_rep
ORDER = sorted(CONTRASTS, key=lambda c: (n_params(c), c))
XMAX = 1.75
DIV = LinearSegmentedColormap.from_list("i9_2_div", [BLUE, "#f0efea", RED])

close = p81.close
nanify = p81.nanify


def same(a: dict, b: dict) -> bool:
    keys = ("ratio", "ci_low", "ci_high")
    if not all(close(a[k], b[k]) for k in keys):
        return False
    for n, v in a.get("per_dataset", {}).items():
        w = b["per_dataset"][n]
        if not all(close(v[k], w[k]) for k in keys) or \
                any(v[k] != w[k] for k in ("n", "n_used", "wins", "n_method_failed", "n_control_failed")):
            return False
    return True


def ci(c: dict, wins: bool = False) -> str:
    s = f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
    return s + (f", {c['wins']}/{c['n_used']}" if wins else "")


def plabel(c: str) -> str:
    return f"{c} ({n_params(c)} / {n_params_member(c)})"


# ================================================================== loading and checks

def load_all() -> dict:
    prereg, dec, val, rep, plan = load(PREREG), load(DEC), load(VALID), load(REP), load(PLAN)
    assert prereg["status"].upper().startswith("IN FORCE")
    assert tuple(dec["primary_datasets"]) == DS and dec["margin"] == MARGIN == 1.03
    assert close(dec["alpha_adjusted"], bonferroni_alpha()) and len(CONTRASTS) == 11
    res = {n: load(I9 / n / "results.json") for n in DS}
    for n in DS:
        assert not res[n]["smoke"] and res[n]["guard"] is not None
    # V1 and validity guard
    for n in DS:
        v = val["datasets"][n]
        assert v["V1_pass"] and v["V1_max_abs_diff"] <= 1e-9, n
    assert not val["smoke"]
    # V4: stored rows used are the stored values
    sc = {}
    for n in DS:
        stored = drv.stored_rows(n)
        per = {lab: nanify(v) for lab, v in res[n]["per_device"].items()}
        for lab, v in stored.items():
            assert all(close(per[lab][t], v[t]) for t in per[lab] if t in v and t not in res[n]["excluded_records"]), \
                (n, lab)
        per[REPLICATE] = nanify(rep["datasets"][n]["per_device"])
        if REPLICATE in stored:
            assert all(close(per[REPLICATE][t], stored[REPLICATE][t]) for t in stored[REPLICATE])
        for c in ALL_CLASSES + ("C0_refit",):
            assert c in per, (n, c)
            assert len(per[c]) == DS_N[n] + len(res[n]["excluded_records"]), (n, c)
        sc[n] = per
    assert len(res["nist_4K"]["excluded_records"]) == 4
    a_adj = bonferroni_alpha()
    clusters = {"tudelft_4K": drv.load_records("tudelft_4K")["clusters"]}

    def pool(names, m, c, alpha, cl=None):
        return pooled_ratio({n: (sc[n][m], sc[n][c]) for n in names}, clusters=cl, alpha=alpha)

    def lr(n, m, c):
        a = np.array([sc[n][m][t] for t in sorted(sc[n][m])])
        b = np.array([sc[n][c][t] for t in sorted(sc[n][m])])
        ok = np.isfinite(a) & np.isfinite(b)
        return np.log(a[ok] / b[ok])
    # pooled vs mlp16 (Bonferroni), replicate, cluster bootstrap
    prim = {c: pool(DS, c, REFERENCE, a_adj) for c in CONTRASTS}
    for c in CONTRASTS:
        assert same(prim[c], dec["primary_vs_mlp16"][c]), c
        pt = math.exp(np.mean([lr(n, c, REFERENCE).mean() for n in DS]))
        assert close(pt, prim[c]["ratio"]), c
    repl = pool(DS, REPLICATE, REFERENCE, a_adj)
    repl95 = pool(DS, REPLICATE, REFERENCE, 0.05)
    assert same(repl, dec["replicate_vs_mlp16"]["bonferroni"]) and same(repl95, dec["replicate_vs_mlp16"]["ci95"])
    clus = {c: pool(DS, c, REFERENCE, a_adj, clusters) for c in CONTRASTS}
    for c in CONTRASTS:
        assert same(clus[c], dec["secondary"]["primary_cluster_bootstrap"][c]), c
    vs_c0 = {c: pool(DS, c, "C0_refit", 0.05) for c in ALL_CLASSES}
    for c in ALL_CLASSES:
        assert same(vs_c0[c], dec["primary_vs_C0_refit"][c]), c
    per_m = {n: {c: pool([n], c, REFERENCE, 0.05) for c in CONTRASTS + (REPLICATE,)} for n in DS}
    per_c = {n: {c: pool([n], c, "C0_refit", 0.05) for c in ALL_CLASSES} for n in DS}
    for n in DS:
        for c in CONTRASTS + (REPLICATE,):
            assert same(per_m[n][c], dec["secondary"]["per_dataset_vs_mlp16_with_replicate"][n][c]), (n, c)
        for c in ALL_CLASSES:
            assert same(per_c[n][c], dec["secondary"]["per_dataset_vs_C0_refit"][n][c]), (n, c)
    # rule, Q1, Q2, margin sensitivity
    nf = lambda n, c: int(np.sum(~np.isfinite(list(sc[n][c].values()))))
    fails = {c: {n: nf(n, c) - nf(n, REFERENCE) for n in DS} for c in CONTRASTS}
    rule = decide(prim, fails, replicate=repl)
    sr = dec["rule"]
    for k in ("labels", "notes", "Q1", "Q2_smallest_matching_class", "Q2_candidates",
              "Q2_beats_with_fewer_parameters", "n_params", "n_params_per_member"):
        assert rule[k] == sr[k], k
    assert rule["Q1"] == "class-dependent" and rule["Q2_smallest_matching_class"] == "mlp16_s1"
    sens = {str(m): decide(prim, fails, margin=m, replicate=repl)["labels"] for m in (1.02, 1.05)}
    assert sens == dec["secondary"]["margin_sensitivity"]
    sens_q2 = {str(m): decide(prim, fails, margin=m, replicate=repl)["Q2_smallest_matching_class"]
               for m in (1.02, 1.05)}
    clus_labels = decide(clus, fails, replicate=repl)["labels"]
    # learning-curve classes
    assert lc_classes(vs_c0) == dec["learning_curve_classes"] == ["mlp16", "gam", "mlp8"]
    # S5 gain fraction
    gain = {}
    for n in DS:
        ref = per_c[n][REFERENCE]
        if ref["ci_high"] < 1:
            gain[n] = {c: math.log(per_c[n][c]["ratio"]) / math.log(ref["ratio"]) for c in ALL_CLASSES}
    assert sorted(gain) == sorted(dec["secondary"]["fraction_of_mlp16_log_gain"]) == ["nist_4K", "sky130_77K"]
    for n in gain:
        for c in ALL_CLASSES:
            assert close(gain[n][c], dec["secondary"]["fraction_of_mlp16_log_gain"][n][c]), (n, c)
    # training losses: new classes (run), stored classes (validity), replicate (replicate run)
    tl, tl_med = {}, {}
    for n in DS:
        acc = {}
        for k, v in res[n]["train"]["final_training_loss"].items():
            acc.setdefault(k.split("|")[0], []).append(v)
        for c, v in acc.items():
            assert close(float(np.mean(v)), dec["secondary"]["final_training_loss_mean_over_folds"][n][c]), (n, c)
            assert len(v) == len(res[n]["folds"])
        for k, v in val["datasets"][n]["stored_training_loss"].items():
            acc.setdefault(k.split("|")[0], []).append(v)
        rr = rep["datasets"][n]
        if "train" in rr:
            acc[REPLICATE] = list(rr["train"]["final_training_loss"].values())
        tl[n] = {c: float(np.mean(v)) for c, v in acc.items()}
        tl_med[n] = {c: float(np.median(v)) for c, v in acc.items()}
    # learning curves: recompute the per-device geometric means and summaries
    lc = {}
    for n in LC_DS:
        r = load(I9 / f"lc_{n}/results.json")
        assert r["classes"] == ["mlp16", "gam", "mlp8"] and tuple(r["sizes"]) == LC_SIZES
        main = sc[n]
        summ = {}
        for c in r["classes"]:
            summ[c] = {}
            for k in LC_SIZES:
                labs = [f"{c}|n{k}|r{i}" for i in range(3)]
                assert all(lab in r["per_device"] for lab in labs)
                dev = {}
                for t in sorted(r["per_device"][labs[0]]):
                    v = np.array([r["per_device"][lab][t] for lab in labs], float)
                    dev[t] = float(np.exp(np.mean(np.log(v)))) if np.all(np.isfinite(v) & (v > 0)) else float("nan")
                st = r["summary"][c][f"n{k}"]
                assert all(close(dev[t], st["per_device"][t]) for t in dev)
                a = pooled_ratio({"d": (dev, main["C0_refit"])})
                b = pooled_ratio({"d": (dev, main[c])})
                assert same({**a, "per_dataset": {}}, st["vs_C0_refit"]) and same({**b, "per_dataset": {}}, st["vs_all"])
                summ[c][k] = {"vs_C0_refit": a, "vs_all": b, "dev": dev}
            alls = pooled_ratio({"d": (main[c], main["C0_refit"])})
            assert same({**alls, "per_dataset": {}}, r["summary"][c]["all"]["vs_C0_refit"])
            assert same({**alls, "per_dataset": {}}, {**per_c[n][c], "per_dataset": {}})
            summ[c]["all"] = {"vs_C0_refit": alls}
        # class vs class per n (post hoc): gam vs mlp16, mlp8 vs mlp16
        cross = {k: {c: pooled_ratio({"d": (summ[c][k]["dev"], summ["mlp16"][k]["dev"])}) for c in ("gam", "mlp8")}
                 for k in LC_SIZES}
        recs = drv.load_records(n)
        ntr = sorted({len(v) for v in recs["train"].values()})
        lc[n] = {"res": r, "summ": summ, "cross": cross, "n_train_all": ntr, "runtime_s": r["runtime_s"],
                 "n_refits": sum(len(v) for v in r["per_device"].values())}
    # means per class and dataset (finite devices)
    means = {n: {c: float(np.nanmean(list(sc[n][c].values()))) for c in ALL_CLASSES + ("C0_refit", "T1_inloop")}
             for n in DS}
    nfail = {n: {c: nf(n, c) for c in ALL_CLASSES + ("C0_refit",)} for n in DS}
    return {"prereg": prereg, "dec": dec, "val": val, "rep": rep, "plan": plan, "res": res, "sc": sc,
            "prim": prim, "repl": repl, "repl95": repl95, "clus": clus, "clus_labels": clus_labels,
            "vs_c0": vs_c0, "per_m": per_m, "per_c": per_c, "rule": rule, "sens": sens, "sens_q2": sens_q2,
            "fails": fails, "gain": gain, "tl": tl, "tl_med": tl_med, "lc": lc, "means": means, "nfail": nfail}


# ================================================================== tables

def t_decision(d: dict) -> rf.Table:
    rule, prim, clus, vs = d["rule"], d["prim"], d["clus"], d["vs_c0"]
    rows = []
    for c in ORDER:
        pmax = max(prim[c]["per_dataset"][n]["ratio"] for n in DS)
        worst = max(DS, key=lambda n: prim[c]["per_dataset"][n]["ratio"])
        cand = "yes" if c in rule["Q2_candidates"] else "no"
        rows.append({"c": c, "np": f"{n_params(c)} / {n_params_member(c)}", "g": ci(prim[c]),
                     "lab": rule["labels"][c], "cl": f"[{clus[c]['ci_low']:.3f}, {clus[c]['ci_high']:.3f}]",
                     "mx": f"{pmax:.3f} ({DS_LABEL[worst]})", "q2": cand, "c0": ci(vs[c])})
    rows.append({"c": "mlp16_rep (replicate)", "np": "1765 / 353", "g": ci(d["repl"]), "lab": "(reference noise)",
                 "cl": "", "mx": f"{max(d['repl']['per_dataset'][n]['ratio'] for n in DS):.3f}", "q2": "",
                 "c0": ci(vs[REPLICATE])})
    rows.append({"c": "mlp16 = M_bias (reference)", "np": "1765 / 353", "g": "1", "lab": "", "cl": "", "mx": "",
                 "q2": "", "c0": ci(vs[REFERENCE])})
    cols = [{"key": "c", "head": "class", "md": "class", "kind": "text"},
            {"key": "np", "head": "trained values", "md": "trained values (total / per member)", "kind": "rtext"},
            {"key": "g", "head": "GMR [Bonferroni CI]", "md": "pooled GMR(X, mlp16) [Bonferroni CI]",
             "kind": "rtext", "group": "pooled vs mlp16, 4 datasets"},
            {"key": "lab", "head": "label", "md": "label (margin 1.03)", "kind": "text",
             "group": "pooled vs mlp16, 4 datasets"},
            {"key": "cl", "head": "cluster CI", "md": "cluster-bootstrap CI (S3)", "kind": "rtext",
             "group": "pooled vs mlp16, 4 datasets"},
            {"key": "mx", "head": "max per-dataset GMR", "md": "max per-dataset point GMR (limit 1.05)",
             "kind": "rtext", "group": "Q2"},
            {"key": "q2", "head": "Q2 cand.", "md": "Q2 candidate", "kind": "text", "group": "Q2"},
            {"key": "c0", "head": "GMR(X, C0_refit) [95% CI]", "md": "pooled GMR(X, C0_refit) [95% CI]",
             "kind": "rtext"}]
    notes = [
        f"Pre-registration {rel(PREREG)} in force from commit {COMMITS['prereg']} (code {COMMITS['code']}); "
        f"decision {rel(DEC)}. Every value is recomputed here from the per-device rows and checked against "
        "decision.json.",
        "Pooled GMR = exp(mean over the 4 datasets of the per-dataset mean ln(RRMS_X / RRMS_mlp16)) over held-out "
        "devices (strict paper RRMS after the held-out card refit with g frozen in the loop); equal dataset weight; "
        "CI from 20000 stratified device resamples at level 1 - 0.05/11 (tails 0.227%). Cluster CI: TU Delft "
        "geometry groups resampled.",
        f"Q1: {rule['Q1']} (poly2, poly3, poly4, gp 'worse'). Q2: candidates {', '.join(rule['Q2_candidates'])}; "
        f"smallest matching class {rule['Q2_smallest_matching_class']} (353 trained values); beats with fewer "
        f"parameters: {'yes' if rule['Q2_beats_with_fewer_parameters'] else 'no'}. Replicate notes: none "
        "(no 'beats' / 'worse' class lies inside the replicate CI).",
        "Q2 also needs point GMR(X, mlp16) <= 1.05 on every dataset and no more failed devices than mlp16 "
        "(NIST 4 K: 4 records with non-finite starting points scored as failed in every class).",
    ]
    return rf.Table("01_table_i9_2_decision", "I9-2 function-class ablation: pre-registered labels, Q1 and Q2 "
                    "(pooled over sky130 77 K, NIST 4 K, TU Delft 4.13 K, IHP 300 K)", cols, rows, notes)


def t_per_dataset(d: dict) -> rf.Table:
    rows = []
    for c in ("C0_refit", "mlp16") + tuple(ORDER) + (REPLICATE,):
        r = {"c": c}
        for n in DS:
            m = d["means"][n][c]
            if c == "C0_refit":
                r[n] = f"{m:.4f}"
            elif c == "mlp16":
                r[n] = f"{m:.4f} | {d['per_c'][n]['mlp16']['ratio']:.3f}"
            else:
                g = d["per_m"][n][c]
                star = "*" if (g["ci_low"] > 1 or g["ci_high"] < 1) else ""
                r[n] = f"{m:.4f} | {d['per_c'][n][c]['ratio']:.3f} | {g['ratio']:.3f}{star}"
        rows.append(r)
    cols = [{"key": "c", "head": "class", "md": "class", "kind": "text"}] + \
           [{"key": n, "head": f"{DS_LABEL[n]} ({DS_N[n]})", "md": f"{DS_LABEL[n]} ({DS_N[n]} devices)",
             "kind": "rtext", "group": "mean RRMS | GMR vs C0_refit | GMR vs mlp16"} for n in DS]
    notes = ["Mean RRMS over the finite held-out devices; GMR(X, C0_refit) and GMR(X, mlp16) per dataset with "
             "device-bootstrap 95% CIs in decision.json (secondary S2, descriptive); * = the 95% CI of GMR(X, mlp16) "
             "excludes 1. NIST 4 K: 53 of 57 devices scored (4 failed in every class). TU Delft CIs: device "
             "bootstrap (the cluster bootstrap is in table 01).",
             "Stored rows: C0_refit, mlp16, poly3 on all datasets; on sky130 77 K also poly2, gp (I4-1), mlp8, mlp32, "
             "mlp16_rep (I3-2 F2). Every other row is new in I9-2."]
    return rf.Table("04_table_i9_2_per_dataset", "I9-2 per dataset: mean held-out RRMS and GMR against C0_refit and "
                    "mlp16", cols, rows, notes)


def t_lc(d: dict) -> rf.Table:
    rows = []
    for n in LC_DS:
        L = d["lc"][n]
        for c in ("mlp16", "gam", "mlp8"):
            s = L["summ"][c]
            r = {"ds": DS_LABEL[n], "c": c}
            for k in LC_SIZES:
                r[f"n{k}"] = ci(s[k]["vs_C0_refit"])
                r[f"a{k}"] = f"{s[k]['vs_all']['ratio']:.3f}"
            r["all"] = ci(s["all"]["vs_C0_refit"])
            rows.append(r)
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "c", "head": "class", "md": "class", "kind": "text"}] + \
           [{"key": f"n{k}", "head": f"n = {k}", "md": f"n = {k}", "kind": "rtext",
             "group": "GMR(class_n, C0_refit) [95% CI]"} for k in LC_SIZES] + \
           [{"key": "all", "head": "n = all", "md": "n = all", "kind": "rtext",
             "group": "GMR(class_n, C0_refit) [95% CI]"}] + \
           [{"key": f"a{k}", "head": f"n = {k}", "md": f"vs all, n = {k}", "kind": "rtext",
             "group": "GMR(class_n, class_all)"} for k in LC_SIZES]
    nt = {n: d["lc"][n]["n_train_all"] for n in LC_DS}
    notes = [f"n = training devices per fold (polarity-stratified subsets, 3 per fold and n, "
             f"default_rng([20261021, n, r, fold])); n = all is the main run (TU Delft {nt['tudelft_4K'][0]}-"
             f"{nt['tudelft_4K'][-1]} devices, NIST {nt['nist_4K'][0]}). Per held-out device the RRMS is "
             "exp(mean over the 3 subsets of ln RRMS). Descriptive (Q3); no decision.",
             f"Refits: TU Delft {d['lc']['tudelft_4K']['n_refits']}, NIST {d['lc']['nist_4K']['n_refits']} "
             f"({d['lc']['tudelft_4K']['runtime_s']:.0f} s and {d['lc']['nist_4K']['runtime_s']:.0f} s wall). Classes "
             "chosen by decision.json before the lc jobs: mlp16 plus the two lowest pooled GMR(X, C0_refit) "
             "(gam 0.889, mlp8 0.904)."]
    return rf.Table("08_table_i9_2_learning_curve", "I9-2 learning curve (Q3, secondary): held-out GMR against the "
                    "number of training devices", cols, rows, notes)


def t_margin(d: dict) -> rf.Table:
    rows = []
    for c in ORDER:
        rows.append({"c": c, "g": ci(d["prim"][c]),
                     "m2": d["sens"]["1.02"][c], "m3": d["rule"]["labels"][c], "m5": d["sens"]["1.05"][c],
                     "cl": d["clus_labels"][c], "cci": f"[{d['clus'][c]['ci_low']:.4f}, {d['clus'][c]['ci_high']:.4f}]"})
    rows.append({"c": "Q2 smallest matching class", "g": "", "m2": d["sens_q2"]["1.02"],
                 "m3": d["rule"]["Q2_smallest_matching_class"], "m5": d["sens_q2"]["1.05"], "cl": "", "cci": ""})
    cols = [{"key": "c", "head": "class", "md": "class", "kind": "text"},
            {"key": "g", "head": "GMR [Bonferroni CI]", "md": "pooled GMR(X, mlp16) [Bonferroni CI]",
             "kind": "rtext"},
            {"key": "m2", "head": "1.02", "md": "label, margin 1.02", "kind": "text", "group": "label at margin"},
            {"key": "m3", "head": "1.03 (rule)", "md": "label, margin 1.03 (pre-registered)", "kind": "text",
             "group": "label at margin"},
            {"key": "m5", "head": "1.05", "md": "label, margin 1.05", "kind": "text", "group": "label at margin"},
            {"key": "cl", "head": "cluster CI, 1.03", "md": "label from the cluster-bootstrap CI, margin 1.03",
             "kind": "text"},
            {"key": "cci", "head": "cluster CI", "md": "cluster-bootstrap Bonferroni CI", "kind": "rtext"}]
    diff = [c for c in ORDER if d["clus_labels"][c] != d["rule"]["labels"][c]]
    assert diff == ["poly4"]
    notes = ["S4 (pre-registered): labels at margins 1.02 and 1.05 from the same Bonferroni CIs. Between 1.02 and 1.05 "
             "only gam (inconclusive -> non-inferior at 1.05) and mlp8 (non-inferior -> equivalent at 1.05) change "
             "label; Q2 is mlp16_s1 at every margin.",
             "S3 (pre-registered interval; the label from it is post hoc): with TU Delft geometry groups resampled as "
             f"clusters, poly4 changes from 'worse' to 'inconclusive' (cluster CI lower bound "
             f"{d['clus']['poly4']['ci_low']:.4f}); every other class keeps its label."]
    return rf.Table("11_table_i9_2_margin_sensitivity", "I9-2 margin sensitivity (S4) and cluster bootstrap (S3)",
                    cols, rows, notes)


def t_validity(d: dict) -> rf.Table:
    val, res = d["val"], d["res"]
    rows = []
    for n in DS:
        v = val["datasets"][n]
        r = res[n]
        to = sum(x.get("n_sim_timeout", 0) for lab in r["refit_rows"].values() for x in lab.values())
        rows.append({"ds": DS_LABEL[n], "v1": f"{v['V1_max_abs_diff']:.1e} ({len(v['V1_abs_diff'])} devices)",
                     "coef": f"{max(v['V1_max_abs_coef_diff_vs_stored_model'].values()):.1e}",
                     "v2": f"{d['nfail'][n]['mlp16']} (all classes {min(d['nfail'][n].values())}-"
                           f"{max(d['nfail'][n].values())})",
                     "v3": str(to), "folds": str(len(r["folds"])),
                     "tr": f"{r['timing']['train_n']} / {r['timing']['train_s']:.0f} s",
                     "rf": f"{r['timing']['refit_n']} / {r['timing']['refit_s']:.0f} s",
                     "job": f"{JOBS['I9_2_run_' + n]} s"})
    cols = [{"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "v1", "head": "V1 max |RRMS diff|", "md": "V1 poly3 rerun max |RRMS - stored|", "kind": "rtext"},
            {"key": "coef", "head": "V1 max |coef diff|", "md": "V1 max |coefficient - stored|", "kind": "rtext"},
            {"key": "v2", "head": "V2 failed", "md": "V2 failed devices mlp16 (range over classes)", "kind": "rtext"},
            {"key": "v3", "head": "V3 timeouts", "md": "V3 simulator timeouts (run refits)", "kind": "rtext"},
            {"key": "folds", "head": "folds", "md": "folds", "kind": "rtext"},
            {"key": "tr", "head": "training tasks / wall", "md": "training tasks / wall", "kind": "rtext",
             "group": "run job"},
            {"key": "rf", "head": "refits / wall", "md": "held-out refits / wall", "kind": "rtext", "group": "run job"},
            {"key": "job", "head": "queue job", "md": "queue job wall", "kind": "rtext", "group": "run job"}]
    gpf = [t for t, v in d["sc"]["tudelft_4K"]["gp"].items() if not math.isfinite(v)]
    assert len(gpf) == 2 and all(d["nfail"]["tudelft_4K"][c] == 0 for c in ALL_CLASSES if c != "gp")
    notes = [f"V2: NIST 4 K 4 failed devices in every class (non-finite starting records, amendment A1 of I8-1); "
             f"TU Delft 4.13 K gp 2 failed devices ({', '.join(gpf)}; every Id-Vd curve non-finite after the "
             "gp-corrected refit), every other class 0. gp is not a Q2 candidate (label 'worse').",
             f"Validity job (V1, {val['runtime_s']} s): poly3 retrained and refit on the first 2 held-out devices "
             "(sorted tags) of each dataset; tolerance 1e-9; all 4 datasets pass. V4: every stored row used equals "
             "the stored source file (checked here); mlp16_s1 = member 0 of the stored M_bias fold model.",
             f"Deviation: the I9_2 jobs failed at 03:14 (I9_2_validity rc=1: two TU Delft held-out devices that share "
             f"a fold queued the poly3 training task twice and the merge step combined them as MLP seed members). Fix "
             f"{COMMITS['fix']}; jobs requeued unchanged at 03:35. No I9-2 result existed before the fix "
             "(docs/RESEARCH_LOOP.md, 'I9-2 validity failure and requeue').",
             f"Other jobs: replicate {JOBS['I9_2_replicate']} s, decide {JOBS['I9_2_decide']} s, lc TU Delft "
             f"{JOBS['I9_2_lc_tudelft_4K']} s, lc NIST {JOBS['I9_2_lc_nist_4K']} s; total queue wall "
             f"{sum(JOBS.values()) / 3600:.2f} h (pre-registered estimate without the learning curve 147 min, "
             f"measured {(sum(JOBS.values()) - JOBS['I9_2_lc_tudelft_4K'] - JOBS['I9_2_lc_nist_4K']) / 60:.0f} min)."]
    return rf.Table("12_table_i9_2_validity", "I9-2 validity checks V1-V4, deviation and compute", cols, rows, notes)


# ================================================================== figures

def fig_forest(d: dict, path: Path) -> list[str]:
    prim, clus, repl, per = d["prim"], d["clus"], d["repl"], d["per_m"]
    fig, ax = plt.subplots(figsize=(12.5, 8.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    ax.axvspan(repl["ci_low"], repl["ci_high"], color="#e4e3df", zorder=0,
               label=f"mlp16_rep (seeds 5-9) Bonferroni CI [{repl['ci_low']:.3f}, {repl['ci_high']:.3f}]")
    ax.axvline(1.0, color=INK, lw=0.9)
    for x in (1 / MARGIN, MARGIN):
        ax.axvline(x, color=INK, lw=0.8, ls="--")
    ax.axvline(PER_DATASET_MAX, color=MUTED, lw=0.8, ls=":")
    ys = list(range(len(ORDER)))[::-1]
    for y, c in zip(ys, ORDER):
        p, q = prim[c], clus[c]
        lab = d["rule"]["labels"][c]
        col = {"worse": RED, "equivalent": BLUE, "non-inferior": BLUE}.get(lab, INK)
        ax.plot([q["ci_low"], q["ci_high"]], [y - 0.22] * 2, color=col, lw=1.0, alpha=0.6)
        ax.errorbar(p["ratio"], y, xerr=[[p["ratio"] - p["ci_low"]], [p["ci_high"] - p["ratio"]]], fmt="o",
                    color=col, ms=7, capsize=4, lw=2.0, zorder=5)
        ax.annotate(f"{p['ratio']:.3f} [{p['ci_low']:.3f}, {p['ci_high']:.3f}]  {lab}", (XMAX * 1.03, y), fontsize=8,
                    color=INK, va="center", annotation_clip=False)
        for j, n in enumerate(DS):
            ax.scatter(min(per[n][c]["ratio"], XMAX * 0.995), y + 0.16 + 0.07 * (j - 1.5), marker=DS_MARK[n], s=18,
                       color=DS_COLOR[n], alpha=0.85, zorder=4, lw=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([plabel(c) for c in ORDER], fontsize=8.5)
    ax.set_xscale("log")
    ticks = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.3g}" for t in ticks], fontsize=8)
    ax.minorticks_off()
    ax.set_xlim(0.78, XMAX)
    ax.set_xlabel("GMR(class, mlp16) of held-out RRMS (log axis; < 1 = class lower; per-dataset points above "
                  f"{XMAX} drawn at the right edge: gp TU Delft 2.609)", fontsize=9)
    ax.set_ylabel("class (trained values: total / per member), sorted by total", fontsize=9)
    ax.errorbar([], [], xerr=[], fmt="o", color=INK, capsize=4, label="pooled, Bonferroni CI (level 1 - 0.05/11)")
    ax.plot([], [], color=INK, lw=1.0, alpha=0.6, label="pooled, cluster-bootstrap CI (TU Delft groups)")
    for n in DS:
        ax.scatter([], [], marker=DS_MARK[n], s=18, color=DS_COLOR[n], label=f"{DS_LABEL[n]} point GMR")
    ax.plot([], [], color=INK, ls="--", lw=0.8, label="margin 1/1.03, 1.03")
    ax.plot([], [], color=MUTED, ls=":", lw=0.8, label="per-dataset limit 1.05 (Q2)")
    ax.legend(fontsize=7.6, frameon=False, loc="upper center", bbox_to_anchor=(0.6, -0.09), ncol=3)
    fig.suptitle("I9-2: pooled GMR of each class against mlp16 (M_bias) over 4 measured datasets; red = 'worse', "
                 "blue = 'equivalent' / 'non-inferior', black = 'inconclusive'", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.17, right=0.74, top=0.93, bottom=0.2)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)] + [rel(I9 / n / "results.json") for n in DS] + [rel(REP)]


def fig_heatmap(d: dict, path: Path) -> list[str]:
    rows_m = list(ORDER) + [REPLICATE]
    rows_c = ["mlp16"] + list(ORDER) + [REPLICATE]
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    lim = math.log(1.5)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    for ax, rows, key, ttl in ((axs[0], rows_m, "m", "GMR(class, mlp16)"),
                               (axs[1], rows_c, "c", "GMR(class, C0_refit)")):
        cols = list(DS) + ["pooled"]
        mat = np.zeros((len(rows), len(cols)))
        txt = [[""] * len(cols) for _ in rows]
        for i, c in enumerate(rows):
            for j, n in enumerate(cols):
                if n == "pooled":
                    g = (d["repl"] if c == REPLICATE else d["prim"][c]) if key == "m" else d["vs_c0"][c]
                else:
                    g = d["per_m"][n][c] if key == "m" else d["per_c"][n][c]
                mat[i, j] = math.log(g["ratio"])
                sig = g["ci_low"] > 1 or g["ci_high"] < 1
                txt[i][j] = f"{g['ratio']:.3f}" + ("*" if sig else "")
        im = ax.imshow(mat, cmap=DIV, norm=norm, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(cols)):
                ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=8.3,
                        color=INK, fontweight="bold" if txt[i][j].endswith("*") else "normal")
        ax.axvline(len(DS) - 0.5, color=BG, lw=3)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([DS_LABEL.get(n, "pooled") for n in cols], fontsize=8.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([plabel(c) for c in rows], fontsize=8.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(ttl + (" (pooled: Bonferroni CI)" if key == "m" else " (pooled: 95% CI)"),
                     fontsize=10, color=INK, loc="left")
    cb = fig.colorbar(im, cax=fig.add_axes([0.92, 0.2, 0.012, 0.6]))
    tk = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5]
    cb.set_ticks([math.log(t) for t in tk])
    cb.set_ticklabels([f"{t:g}" for t in tk])
    cb.set_label("GMR (blue = class lower, red = class higher; colour clipped at 0.67 / 1.5)", fontsize=8.5)
    fig.suptitle("I9-2: held-out GMR per dataset and pooled; * and bold = CI excludes 1 (per dataset 95% device "
                 "bootstrap; pooled vs mlp16 Bonferroni, pooled vs C0_refit 95%)", fontsize=10.5, color=INK)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08, wspace=0.42)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)]


def fig_capacity(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(1, 5, figsize=(22, 6.2), dpi=DPI, sharey=False)
    fig.patch.set_facecolor(BG)
    classes = list(ORDER) + ["mlp16", REPLICATE]
    for ax, n in zip(axs, ("pooled",) + DS):
        style(ax)
        for c in classes:
            g = d["vs_c0"][c] if n == "pooled" else d["per_c"][n][c]
            x = n_params(c) * (1.12 if c == REPLICATE else 1.0)
            col = RED if c in ("mlp16", REPLICATE) else (ORANGE if c in ("poly2", "poly3", "poly4", "poly5")
                                                         else (AQUA if c == "gam" else BLUE if c.startswith("mlp")
                                                               else MUTED))
            ax.errorbar(x, g["ratio"], yerr=[[g["ratio"] - g["ci_low"]], [g["ci_high"] - g["ratio"]]], fmt="o",
                        color=col, ms=5, capsize=2.5, lw=1.1)
            if c != REPLICATE:
                ax.annotate(c, (x, g["ratio"]), xytext=(4, 2), textcoords="offset points", fontsize=6.6, color=INK)
        ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="y", which="minor", labelsize=7, colors=MUTED)
        ax.set_xlabel("trained values in g (total over members)", fontsize=8.5)
        ttl = "pooled (equal weight), 95% CI" if n == "pooled" else f"{DS_LABEL[n]} ({DS_N[n]}), 95% CI"
        ax.set_title(ttl, fontsize=9.5, color=INK, loc="left")
    axs[0].set_ylabel("GMR(class, C0_refit) of held-out RRMS (log axis; < 1 = gain)", fontsize=9)
    h = axs[0]
    for col, lab in ((MUTED, "const, lin, gp"), (ORANGE, "poly2-poly5"), (AQUA, "gam"),
                     (BLUE, "mlp8, mlp16_s1, mlp32"), (RED, "mlp16 (M_bias), mlp16_rep")):
        h.errorbar([], [], yerr=[], fmt="o", color=col, label=lab)
    h.legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("I9-2 capacity curve (pre-registered): held-out gain over C0_refit against the number of trained "
                 "values, pooled and per dataset", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)]


def fig_train_vs_heldout(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, n in zip(axs.flat, DS):
        style(ax)
        tl = d["tl"][n]
        for c, v in sorted(tl.items(), key=lambda kv: n_params(kv[0])):
            if c == "gp":
                continue
            g = d["per_c"][n][c]
            col = RED if c in ("mlp16", REPLICATE) else (ORANGE if c.startswith("poly") else
                                                         AQUA if c == "gam" else BLUE if c.startswith("mlp")
                                                         else MUTED)
            ax.errorbar(v, g["ratio"], yerr=[[g["ratio"] - g["ci_low"]], [g["ci_high"] - g["ratio"]]], fmt="o",
                        color=col, ms=5, capsize=2.5, lw=1.0, alpha=0.9)
            ax.annotate(c, (v, g["ratio"]), xytext=(5, 2), textcoords="offset points", fontsize=7.2, color=INK)
        ax.axhline(1.0, color=INK, lw=0.8)
        g = d["per_c"][n]["gp"]
        gp_txt = (f"gp (not drawn): training loss {tl['gp']:.4g} (median over folds {d['tl_med'][n]['gp']:.4g}), "
                  f"GMR {ci(g)}")
        ax.set_xlabel("final E1 training loss, mean over folds", fontsize=8.5)
        ax.set_ylabel("GMR(class, C0_refit) of held-out RRMS, 95% CI", fontsize=8.5)
        ax.set_title(f"{DS_LABEL[n]} ({DS_N[n]}); mlp16_s1 has no stored training loss\n{gp_txt}", fontsize=9,
                     color=INK, loc="left")
    fig.suptitle("I9-2 (S7): final training loss (mean over training devices of the paper per-device RRMS of I_sim x "
                 "exp(g) on the fold's starting records, no penalty)\nagainst the held-out gain over C0_refit; colours: "
                 "grey const / lin, orange poly2-poly5, green gam, blue mlp8 / mlp32, red mlp16 / mlp16_rep",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC), rel(VALID), rel(REP)] + [rel(I9 / n / "results.json") for n in DS]


def fig_lc(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    xs = list(range(len(LC_SIZES) + 1))
    for j, n in enumerate(LC_DS):
        L = d["lc"][n]
        nt = L["n_train_all"]
        xl = [str(k) for k in LC_SIZES] + [f"all ({nt[0]}-{nt[-1]})" if len(nt) > 1 else f"all ({nt[0]})"]
        for i, key in enumerate(("vs_C0_refit", "vs_all")):
            ax = axs[i, j]
            style(ax)
            for k, c in enumerate(("mlp16", "gam", "mlp8")):
                s = L["summ"][c]
                ks = list(LC_SIZES) + (["all"] if key == "vs_C0_refit" else [])
                ys = [s[q][key] for q in ks]
                off = (k - 1) * 0.08
                ax.errorbar([x + off for x in xs[:len(ks)]], [y["ratio"] for y in ys],
                            yerr=[[y["ratio"] - y["ci_low"] for y in ys], [y["ci_high"] - y["ratio"] for y in ys]],
                            fmt="o-", color=LC_COLOR[c], capsize=3, ms=5, lw=1.4, label=c)
            ax.axhline(1.0, color=INK, lw=0.8)
            ax.set_xticks(xs[:len(LC_SIZES) + (1 if key == "vs_C0_refit" else 0)])
            ax.set_xticklabels(xl[:len(LC_SIZES) + (1 if key == "vs_C0_refit" else 0)], fontsize=8.5)
            ax.set_xlabel("training devices per fold", fontsize=9)
            ax.set_ylabel("GMR(class_n, C0_refit), 95% CI" if key == "vs_C0_refit"
                          else "GMR(class_n, class at n = all), 95% CI", fontsize=9)
            ax.set_title(f"{DS_LABEL[n]} ({DS_N[n]} held-out devices): "
                         + ("against C0_refit" if key == "vs_C0_refit" else "against the same class trained on all "
                            "devices"), fontsize=9.5, color=INK, loc="left")
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle("I9-2 learning curve (Q3, secondary, descriptive): classes mlp16, gam, mlp8; 3 polarity-stratified "
                 "training subsets per fold and n", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I9 / f"lc_{n}/results.json") for n in LC_DS] + [rel(I9 / n / "results.json") for n in LC_DS]


def fig_gain(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.0), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    classes = list(ORDER) + ["mlp16", REPLICATE]
    for ax, n in zip(axs, ("sky130_77K", "nist_4K")):
        style(ax)
        v = d["gain"][n]
        ys = np.arange(len(classes))[::-1]
        for y, c in zip(ys, classes):
            col = RED if c in ("mlp16", REPLICATE) else (ORANGE if c.startswith("poly") else AQUA if c == "gam"
                                                         else BLUE if c.startswith("mlp") else MUTED)
            ax.barh(y, v[c], color=col, height=0.62)
            ax.annotate(f"{v[c]:.2f}", (max(v[c], 0) + 0.03, y), va="center", fontsize=7.5, color=INK)
        ax.axvline(0, color=INK, lw=0.8)
        ax.axvline(1, color=INK, lw=0.8, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([plabel(c) for c in classes], fontsize=8.5)
        ref = d["per_c"][n]["mlp16"]
        ax.set_title(f"{DS_LABEL[n]}: GMR(mlp16, C0_refit) {ci(ref)}", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("ln GMR(class, C0_refit) / ln GMR(mlp16, C0_refit)", fontsize=9)
        ax.set_xlim(-0.75, 2.1)
    fig.suptitle("I9-2 (S5): fraction of the mlp16 (M_bias) held-out log gain kept by each class;\ndefined on the "
                 "datasets where the mlp16 CI upper bound is < 1 (TU Delft 4.13 K 1.004 and IHP 300 K 1.214 are not)",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC)]


def fig_replicate(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(1, 5, figsize=(20, 4.6), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    cls = (REPLICATE, "mlp16_s1", "mlp8", "mlp32", "gam")
    for ax, n in zip(axs, ("pooled",) + DS):
        style(ax)
        ys = np.arange(len(cls))[::-1]
        for y, c in zip(ys, cls):
            if n == "pooled":
                g = d["repl"] if c == REPLICATE else d["prim"][c]
            else:
                g = d["per_m"][n][c]
            col = RED if c == REPLICATE else (AQUA if c == "gam" else BLUE)
            ax.errorbar(g["ratio"], y, xerr=[[g["ratio"] - g["ci_low"]], [g["ci_high"] - g["ratio"]]], fmt="o",
                        color=col, ms=6, capsize=3, lw=1.5)
            ax.annotate(f"{g['ratio']:.3f}", (g["ratio"], y + 0.22), ha="center", fontsize=7.3, color=INK)
        if n == "pooled":
            ax.axvspan(d["repl"]["ci_low"], d["repl"]["ci_high"], color="#e4e3df", zorder=0)
        else:
            r = d["per_m"][n][REPLICATE]
            ax.axvspan(r["ci_low"], r["ci_high"], color="#e4e3df", zorder=0)
        ax.axvline(1, color=INK, lw=0.8)
        for x in (1 / MARGIN, MARGIN):
            ax.axvline(x, color=INK, lw=0.7, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels(cls, fontsize=8.5)
        ax.set_xlabel("GMR(class, mlp16)", fontsize=8.5)
        ax.set_title("pooled, Bonferroni CI" if n == "pooled" else f"{DS_LABEL[n]}, 95% CI", fontsize=9.5,
                     color=INK, loc="left")
    fig.suptitle("I9-2 seed-noise reference (D3): mlp16_rep (M_bias recipe, seeds 5-9) against mlp16 (seeds 0-4), "
                 "with the MLP classes and gam; grey band = replicate CI; dashed = margin 1/1.03, 1.03",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DEC), rel(REP)]


IV_ARMS = (("C0_refit", "-", 1.1, "C0_refit refit card (NGSpice)"),
           ("mlp16", "--", 1.3, "mlp16 (M_bias): refit card NGSpice x exp(g)"),
           ("poly5", ":", 1.6, "poly5 (Legendre, 42 values): refit card NGSpice x exp(g)"),
           ("gam", "-.", 1.3, "gam (66 values): refit card NGSpice x exp(g)"))


def fig_iv(tag: str, ex: dict, d: dict, path: Path, why: str) -> list[str]:
    dv = ex["devices"][tag]
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for row, kind in enumerate(("idvg", "idvd")):
        cs = [c for c in dv["curves"] if c["kind"] == kind]
        for col, scale in enumerate(("log", "linear") if kind == "idvg" else ("linear", "log")):
            ax = axs[row, col]
            style(ax)
            for k, c in enumerate(cs):
                color = p81.IV_COLORS[k % len(p81.IV_COLORS)]
                x = np.abs(np.asarray(c["vg"] if kind == "idvg" else c["vd"]))
                meas = np.abs(np.asarray(c["meas"]))
                step = 3 if len(x) > 100 else 1
                ax.scatter(x[::step], meas[::step], s=9, color=color, alpha=0.9 if c["included"] else 0.3,
                           zorder=3, lw=0)
                for arm, ls, lw, _ in IV_ARMS:
                    ax.plot(x, np.abs(np.asarray(c["sim"][arm])), color=color, ls=ls, lw=lw)
                bias = f"|Vd| {abs(c['fixed_V']):.2g} V" if kind == "idvg" else f"|Vg| {abs(c['fixed_V']):.2g} V"
                ax.plot([], [], color=color, marker="o", ms=4, ls="",
                        label=bias + ("" if c["included"] else " (not in RRMS)"))
            if scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-13)
            ax.set_xlabel("|Vgs| (V)" if kind == "idvg" else "|Vds| (V)", fontsize=9)
            ax.set_ylabel("|Id| (A)", fontsize=9)
            ax.set_title(f"{'Id-Vg' if kind == 'idvg' else 'Id-Vd'}, {scale} scale", fontsize=9.5, color=INK,
                         loc="left")
            if col == 0:
                ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)
    h = axs[0, 1]
    for arm, ls, lw, lab in IV_ARMS:
        h.plot([], [], color=INK, ls=ls, lw=lw, label=f"{lab}, RRMS {d['sc']['nist_4K'][arm][tag]:.4f}")
    h.scatter([], [], s=12, color=INK, label="measured (4 K; every 3rd point on 181-point sweeps)")
    h.legend(fontsize=7.3, frameon=False, loc="upper left")
    fig.suptitle(f"I9-2 NIST 4 K {tag} ({'nMOS' if dv['polarity'] == 'nmos' else 'pMOS'}, W {dv['W_um']:g} um, "
                 f"L {dv['L_um']:g} um, bin {dv['bin']}); ln(poly5 / mlp16) {dv['log_poly5_mlp16']:+.3f}; {why}\n"
                 "ngspice-41 arm64, 77 K card and deck, mult = m = 1; stored held-out cards re-simulated, exp(g) of "
                 "the fold's stored correction applied in Python (RRMS reproduced, relative difference "
                 f"{max(v['rel_diff'] for v in ex['checks'][tag].values()):.0e})", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(EXAMPLES), rel(I9 / "nist_4K/results.json"), rel(I9 / "nist_4K/cards.json"),
            rel(EXP / "I8_1/lodo/cards.json")]


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)

    d = load_all()
    ex = load(EXAMPLES)
    assert ex["max_rel_diff"] == 0.0 and len(ex["devices"]) == 2
    for tag, ch in ex["checks"].items():
        for a, v in ch.items():
            assert close(v["stored"], d["sc"]["nist_4K"][a][tag]), (tag, a)
    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(t_decision(d), "Table of the pre-registered result per class: trained values, pooled GMR(X, mlp16) "
              "with the Bonferroni CI, label, cluster-bootstrap CI, largest per-dataset point GMR, Q2 candidacy and "
              "pooled GMR(X, C0_refit); replicate and reference rows; Q1 and Q2", [rel(PREREG), rel(DEC)])
    items.append(("02_i9_2_forest.png", "Forest plot of the pooled GMR(X, mlp16) with Bonferroni CIs (thick) and "
                  "cluster-bootstrap CIs (thin), per-dataset point GMRs, the margin lines 1/1.03 and 1.03, the "
                  "per-dataset limit 1.05 and the replicate CI band", fig_forest(d, folder / "02_i9_2_forest.png")))
    items.append(("03_i9_2_heatmap.png", "Heatmaps of GMR(X, mlp16) and GMR(X, C0_refit) per class and dataset, "
                  "with the pooled column; bold * = CI excludes 1", fig_heatmap(d, folder / "03_i9_2_heatmap.png")))
    add_table(t_per_dataset(d), "Table per dataset and class: mean held-out RRMS, GMR vs C0_refit and GMR vs mlp16",
              [rel(DEC)] + [rel(I9 / n / "results.json") for n in DS] + [rel(REP)])
    items.append(("05_i9_2_capacity.png", "Capacity curve: GMR(X, C0_refit) with 95% CI against the number of "
                  "trained values (log axis), pooled and per dataset", fig_capacity(d, folder / "05_i9_2_capacity.png")))
    items.append(("06_i9_2_train_vs_heldout.png", "Final E1 training loss (mean over folds) against the held-out "
                  "GMR(X, C0_refit) per dataset (S7; post hoc pairing)",
                  fig_train_vs_heldout(d, folder / "06_i9_2_train_vs_heldout.png")))
    items.append(("07_i9_2_learning_curve.png", "Learning curves (Q3) on TU Delft 4.13 K and NIST 4 K for mlp16, "
                  "gam and mlp8: GMR against C0_refit (top) and against the same class trained on all devices "
                  "(bottom) for n = 2, 4, 8 training devices per fold", fig_lc(d, folder / "07_i9_2_learning_curve.png")))
    add_table(t_lc(d), "Table of the learning curve: GMR(class_n, C0_refit) with CI and GMR(class_n, class_all)",
              [rel(I9 / f"lc_{n}/results.json") for n in LC_DS])
    items.append(("09_i9_2_gain_fraction.png", "Fraction of the mlp16 log gain over C0_refit kept by each class (S5) "
                  "on sky130 77 K and NIST 4 K", fig_gain(d, folder / "09_i9_2_gain_fraction.png")))
    items.append(("10_i9_2_replicate.png", "Seed-noise reference: GMR(mlp16_rep, mlp16) with CI, pooled (Bonferroni) "
                  "and per dataset (95%), beside mlp16_s1, mlp8, mlp32 and gam",
                  fig_replicate(d, folder / "10_i9_2_replicate.png")))
    add_table(t_margin(d), "Table of the labels at margins 1.02, 1.03 and 1.05 (S4) and of the labels from the "
              "cluster-bootstrap CI (S3)", [rel(DEC)])
    add_table(t_validity(d), "Table of validity V1-V4 per dataset, the deviation (validity failure at 03:14, fix "
              "a864f77, requeue 03:35) and compute", [rel(VALID)] + [rel(I9 / n / "results.json") for n in DS])
    whys = {}
    tags = list(ex["devices"])
    whys[tags[0]] = "lowest ln(poly5 / mlp16) of the 53 NIST devices"
    whys[tags[1]] = "highest ln(poly5 / mlp16) of the 53 NIST devices"
    for k, tag in enumerate(tags):
        name = f"{13 + k:02d}_i9_2_iv_{tag}.png"
        src = fig_iv(tag, ex, d, folder / name, whys[tag])
        items.append((name, f"Example I-V, NIST 4 K {tag} ({whys[tag]}): measured points, NGSpice C0_refit (solid), "
                      "and the refit cards of mlp16 (dashed), poly5 (dotted) and gam (dash-dot), each NGSpice output "
                      "x exp(g) applied in Python; Id-Vg log / linear, Id-Vd linear / log", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i9_2.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. GMR = geometric mean ratio of held-out strict paper "
          "RRMS after the held-out card refit with g frozen in the loop; < 1 = the first class is lower.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    rule, prim, vs, pc, lc = d["rule"], d["prim"], d["vs_c0"], d["per_c"], d["lc"]
    lab = rule["labels"]
    tdl, nis = lc["tudelft_4K"]["summ"], lc["nist_4K"]["summ"]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no fit, no training). The example I-V curves "
              "(figures 13-14) are NGSpice re-simulations of stored held-out cards by "
              f"`scripts/research/explore/i9_2_examples.py` ({rel(EXAMPLES)}, {ex['runtime_s']} s; RRMS reproduced "
              "with relative difference 0).", "",
              "## Contents", "",
              "- I9-2: function-class ablation of the shared bias-only correction g (I = I_card x exp(g), inputs "
              "is_pmos, |Vgs|/VDD, |Vds|/VDD) on the 4 measured datasets of the primary pool: sky130 77 K (12 devices, "
              "LODO), NIST 4 K (53 of 57 scored, LODO), TU Delft 4.13 K (45, leave-one-group-out, 7 groups), IHP "
              "measured 300 K (18, LODO within LV / HV). 11 classes against mlp16 (= stored I3-2 M_bias). "
              f"Pre-registration `{rel(PREREG)}` (in force from commit {COMMITS['prereg']}, code {COMMITS['code']}); "
              f"decision `{rel(DEC)}`. Queue jobs: " + ", ".join(f"{k} ({v} s)" for k, v in JOBS.items()) + ".",
              f"- Deviation: the I9_2 jobs failed at 03:14 on a validity bug (duplicate poly3 fold task on TU Delft); "
              f"fix {COMMITS['fix']}; requeued unchanged at 03:35; no I9-2 result existed before the fix "
              "(docs/RESEARCH_LOOP.md, 'I9-2 validity failure and requeue').",
              "- Validity: V1 poly3 rerun max |RRMS - stored| 0 on all 4 datasets; V2 failed devices: 4 on NIST in "
              "every class (non-finite starting records); gp 2 more on TU Delft (t40_nmos_W0p12_L0p04_d152, "
              "t40_nmos_W1_L0p04_d199: every Id-Vd curve of the gp-corrected refit card non-finite; GMR(gp, mlp16) on "
              "TU Delft uses the 43 others); every other class and dataset 0; V3 0 simulator timeouts in the run "
              "refits; V4 stored rows equal the source files (checked here).",
              f"- Q1: {rule['Q1']}. Labels (pooled GMR(X, mlp16), Bonferroni CI): " +
              "; ".join(f"{c} {prim[c]['ratio']:.3f} {lab[c]}" for c in ORDER) + ".",
              f"- Replicate mlp16_rep (seeds 5-9) vs mlp16: {ci(d['repl'])} (Bonferroni), {ci(d['repl95'])} (95%); "
              "no replicate note applies.",
              f"- Q2: candidates {', '.join(rule['Q2_candidates'])}; smallest matching class "
              f"{rule['Q2_smallest_matching_class']} (353 trained values, 1 seed); no class beats mlp16 with fewer "
              "parameters. gam (66 values, pooled 0.970) is not a candidate: its CI upper bound is 1.030 and its IHP "
              f"300 K GMR is {prim['gam']['per_dataset']['ihp_meas_300K']['ratio']:.3f} (> 1.05).",
              f"- Per dataset: on NIST 4 K poly4 / poly5 / gam exceed the mlp16 log gain (fractions "
              f"{d['gain']['nist_4K']['poly4']:.2f} / {d['gain']['nist_4K']['poly5']:.2f} / "
              f"{d['gain']['nist_4K']['gam']:.2f}; GMR vs mlp16 {d['per_m']['nist_4K']['poly4']['ratio']:.3f} / "
              f"{d['per_m']['nist_4K']['poly5']['ratio']:.3f} / {d['per_m']['nist_4K']['gam']['ratio']:.3f}). On IHP "
              "300 K every class with more trained values than lin has a GMR(X, C0_refit) above 1: " +
              ", ".join(f"{c} {pc['ihp_meas_300K'][c]['ratio']:.3f}" for c in ORDER if n_params(c) > 6) +
              f"; const {pc['ihp_meas_300K']['const']['ratio']:.3f}, lin {pc['ihp_meas_300K']['lin']['ratio']:.3f}.",
              f"- Pooled vs C0_refit (95%): mlp16 {ci(vs['mlp16'])}, gam {ci(vs['gam'])}, mlp8 {ci(vs['mlp8'])}, "
              f"poly4 {ci(vs['poly4'])}, poly3 {ci(vs['poly3'])}, gp {ci(vs['gp'])}.",
              f"- Learning curve (Q3): TU Delft GMR(X, C0_refit) at n = 2 / 4 / 8 / all: mlp16 " +
              " / ".join(f"{tdl['mlp16'][k]['vs_C0_refit']['ratio']:.3f}" for k in LC_SIZES + ("all",)) + ", gam " +
              " / ".join(f"{tdl['gam'][k]['vs_C0_refit']['ratio']:.3f}" for k in LC_SIZES + ("all",)) + "; NIST mlp16 " +
              " / ".join(f"{nis['mlp16'][k]['vs_C0_refit']['ratio']:.3f}" for k in LC_SIZES + ("all",)) + ", gam " +
              " / ".join(f"{nis['gam'][k]['vs_C0_refit']['ratio']:.3f}" for k in LC_SIZES + ("all",)) + ".",
              f"- Margin sensitivity (S4): at 1.02 every label equals the 1.03 label; at 1.05 gam becomes non-inferior "
              f"and mlp8 equivalent; Q2 is mlp16_s1 at every margin. Cluster bootstrap (S3, TU Delft groups): poly4 CI "
              f"[{d['clus']['poly4']['ci_low']:.4f}, {d['clus']['poly4']['ci_high']:.4f}] includes 1; every other "
              "class has the same label from the cluster CI.",
              "- Training loss (S7): gam has the lowest mean final training loss on all 4 datasets; on IHP 300 K its "
              f"held-out GMR(gam, C0_refit) is {pc['ihp_meas_300K']['gam']['ratio']:.3f}. gp has mean training losses "
              f"{d['tl']['nist_4K']['gp']:.2f} (NIST) and {d['tl']['tudelft_4K']['gp']:.2f} (TU Delft) against "
              f"0.028-0.063 for every other class on these two datasets (median over folds {d['tl_med']['nist_4K']['gp']:.3f} and "
              f"{d['tl_med']['tudelft_4K']['gp']:.3f}). mlp16_s1 has no stored training loss (refit-only class).",
              "- Post hoc (not pre-registered): the labels from the cluster-bootstrap CI, the training-loss vs held-out "
              "pairing (figure 06), the learning-curve class-vs-class values and the example I-V figures.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    summary = [
        f"I9-2 function-class ablation of g (4 measured datasets, pooled equal weight, Bonferroni over 11 contrasts): "
        f"Q1 {rule['Q1']}; mlp16_s1 equivalent {prim['mlp16_s1']['ratio']:.3f}, mlp32 equivalent "
        f"{prim['mlp32']['ratio']:.3f}, mlp8 non-inferior {prim['mlp8']['ratio']:.3f}; poly2 / poly3 / poly4 / gp "
        f"worse ({prim['poly2']['ratio']:.3f} / {prim['poly3']['ratio']:.3f} / {prim['poly4']['ratio']:.3f} / "
        f"{prim['gp']['ratio']:.3f}); const, lin, poly5, gam inconclusive.",
        f"Q2 smallest matching class mlp16_s1 (353 values); no class beats mlp16 with fewer parameters. Replicate "
        f"{ci(d['repl'])}.",
        f"Per dataset: NIST 4 K poly4 / poly5 / gam below mlp16; IHP 300 K every flexible class above C0_refit. "
        f"Learning curve: gam gains more than mlp16 with more devices on both datasets.",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    for n in LC_DS:
        print(n, "train sizes", lc[n]["n_train_all"],
              {k: {c: round(v["ratio"], 3) for c, v in lc[n]["cross"][k].items()} for k in LC_SIZES})
    print("tl", {n: {c: round(v, 4) for c, v in d["tl"][n].items()} for n in DS})
    print("tl median gp", {n: round(d["tl_med"][n].get("gp", float("nan")), 4) for n in DS})
    print("sens q2", d["sens_q2"], "clus labels == labels", d["clus_labels"] == rule["labels"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
