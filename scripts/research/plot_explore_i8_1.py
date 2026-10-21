#!/usr/bin/env python3
"""Report batch for exploration direction I8-1: prospective DEV test on the NIST
mds2-2997 sky130 transistors at 4 K (MPW-5 tile; nfet_01v8_lvt and
pfet_01v8_lvt, 57 DEV devices, 77 K paper card with an explicit bin per device,
deck at 77 K). P_a: sign of y_D = ln GMR(M_bias, C0_refit) from the frozen I7-1
S1_lodo rule (prediction.json, committed before any held-out refit). P_b:
GMR(M_bias, C0_refit) CI upper < 1 with R2 / R3. P_c (secondary): sign(S3_D)
(prediction_v2.json, amendment A1).

Reads stored files only; no NGSpice fit, no training:

* ``out/research/explore/I8_1/{prereg.json, prereg_amendment_2026-10-20.json,
  prediction.json, prediction_v2.json, decision.json, split.json}``
* ``out/research/explore/I8_1/start/{summary.json, bins.json}``
* ``out/research/explore/I8_1/{lodo, baselines}/{results.json, corrections.json}``
* ``out/research/explore/I8_1/examples/examples.json`` (NGSpice re-simulation of
  stored cards for 4 example devices, ``scripts/research/explore/i8_1_examples.py``)
* the I7-1 stored and simulated datasets through ``plot_explore_i7_1.load_all``
  (which recomputes and checks them), and the I3-2 sky130 77 K corrections
  ``out/research/explore/I3_2/real_dev/corrections.json`` (figure 06)

Checks: every per-device score in decision / lodo / baselines agrees across
files; aggregates, every paired ratio (all, covered, nMOS, pMOS), y_D, the
per-family y, P_a / P_b / P_c, the validity conditions, the gates, the S1 / S3
dataset values (mean of the per-fold values) and the predicted y are recomputed
and compared with the stored values. Items marked "post hoc" are not
pre-registered.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i8_1.py --date 2026-10-21
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
import plot_explore_i7_1 as p71  # noqa: E402  (I7-1 datasets, scatter panel)
from cryoml.explore import i8_1_nist as m  # noqa: E402
from cryoml.explore.i3_2_bias import paired_ratio  # noqa: E402
from cryoml.explore.i4_1_baselines import correction_from_json_any, g_grid  # noqa: E402
from cryoml.explore.i7_1_transfer import rule_predict, spearman  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i8_1"
I8 = EXP / "I8_1"
PREREG = I8 / "prereg.json"
AMEND = I8 / "prereg_amendment_2026-10-20.json"
PRED1 = I8 / "prediction.json"
PRED2 = I8 / "prediction_v2.json"
DEC = I8 / "decision.json"
SPLIT = I8 / "split.json"
START = I8 / "start/summary.json"
BINS = I8 / "start/bins.json"
LODO = I8 / "lodo/results.json"
LODO_CORR = I8 / "lodo/corrections.json"
BASE = I8 / "baselines/results.json"
BASE_CORR = I8 / "baselines/corrections.json"
EXAMPLES = I8 / "examples/examples.json"
I32_CORR = EXP / "I3_2/real_dev/corrections.json"
COMMITS = {"prereg": "d022c32", "prediction": "0c184f8", "amendment": "5ba2f8e", "prediction_v2": "4db5a7b"}
# queue jobs (out/research/done.txt): wall time in s
JOBS = {"I8_1_start": 411, "I8_1_stats": 1264, "I8_1_stats_v2": 992, "I8_1_lodo": 1016,
        "I8_1_baselines": 393, "I8_1_decide": 3}
ARMS = ("C0", "C0_refit", "T1_inloop", "M_bias", "B1", "B2", "B3_d3")
ARM_LABEL = {"C0": "C0 starting card (no refit)", "C0_refit": "C0_refit (control)",
             "T1_inloop": "T1_inloop (per-curve gain, non-deployable)", "M_bias": "M_bias (primary)",
             "B1": "B1 wider BSIM4 set", "B2": "B2 Rext + self-heating", "B3_d3": "B3_d3 cubic polynomial g"}
FAM_COLOR = {"nmos": BLUE, "pmos": ORANGE}
FAM_LABEL = {"nmos": "nMOS (nfet_01v8_lvt)", "pmos": "pMOS (pfet_01v8_lvt)"}
NIST_COLOR = RED
VDD = 1.8
T1_PLUS = 0.005


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol * max(1.0, abs(b))


def fam_of(tag: str) -> str:
    return "pmos" if tag.startswith("pfet") else "nmos"


def nanify(per: dict) -> dict:
    return {t: (float("nan") if v is None else float(v)) for t, v in per.items()}


def ci_text(c: dict, wins: bool = True) -> str:
    s = f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
    return s + (f", {c['wins']}/{c['n_used']}" if wins else "")


def same_cmp(a: dict, b: dict) -> bool:
    return all(close(a[k], b[k]) for k in ("ratio", "ci_low", "ci_high")) and \
        all(a[k] == b[k] for k in ("n", "n_used", "wins", "n_method_failed", "n_control_failed"))


# ================================================================== loading and checks

def load_all() -> dict:
    prereg, amend, pred1, pred2, dec = load(PREREG), load(AMEND), load(PRED1), load(PRED2), load(DEC)
    split, start, bins = load(SPLIT), load(START), load(BINS)["devices"]
    lodo, base = load(LODO), load(BASE)
    assert not dec["smoke"] and not lodo["smoke"] and not base["smoke"] and not start["smoke"]
    tags = sorted(split["DEV"])
    assert len(tags) == 57 and sorted(bins) == tags and lodo["held_out"] == tags == base["held_out"]
    excl = dec["excluded_start_failures"]
    assert excl == pred2["excluded_records"] == lodo["excluded_records"] == base["excluded_records"]
    assert len(excl) == 4 and all(t.startswith("nfet") and not bins[t]["covered"] for t in excl)
    assert sorted(amend["finding"]["per_device"]) == excl
    finite = [t for t in tags if t not in excl]
    # per-device scores: lodo arms, baseline arms; C0_refit / M_bias / C0 identical in both files
    sc = {a: nanify(v) for a, v in lodo["per_device"].items()}
    bsc = {a: nanify(v) for a, v in base["per_device"].items()}
    for a in ("C0", "C0_refit", "M_bias"):
        assert all(close(sc[a][t], bsc[a][t]) for t in tags), a
    for a in ("B1", "B2", "B3_d3"):
        sc[a] = bsc[a]
    for a in ARMS:
        assert sorted(sc[a]) == tags and all(math.isnan(sc[a][t]) for t in excl), a
        assert all(math.isfinite(sc[a][t]) for t in finite), a
    # C0 = starting-card score in start/summary.json
    assert all(close(sc["C0"][t], start["rows"][t]["start_rrms"]) for t in finite)
    assert all(start["rows"][t]["start_rrms"] is None for t in excl)
    # aggregates
    fam = {t: fam_of(t) for t in tags}
    cov = [t for t in tags if bins[t]["covered"]]
    assert cov == lodo["covered"] == base["covered"] and len(cov) == 25
    agg = {a: m.aggregate(sc[a], fam) for a in ARMS}
    for a in ARMS:
        st = (lodo if a in lodo["summary"]["all"] else base)["summary"]["all"][a]
        for k in ("n", "n_failed", "valid_only_mean", "median", "combined"):
            assert close(agg[a][k], st[k]), (a, k)
    for a in ("C0", "C0_refit", "M_bias", "T1_inloop"):
        assert close(agg[a]["valid_only_mean"], dec["P_b"]["means_valid_only"][a]), a
    fam_mean = {a: {f: float(np.mean([sc[a][t] for t in finite if fam[t] == f])) for f in ("nmos", "pmos")}
                for a in ARMS}
    # paired ratios, every subset and pair stored in lodo / baselines
    subsets = {"all": tags, "covered": cov, "nmos": [t for t in tags if fam[t] == "nmos"],
               "pmos": [t for t in tags if fam[t] == "pmos"]}
    for res in (lodo, base):
        for sname, sel in subsets.items():
            for key, stored in res["comparisons"][sname].items():
                a, b = key.split("_vs_")
                mine = paired_ratio({t: sc[a][t] for t in sel}, {t: sc[b][t] for t in sel})
                assert same_cmp(mine, stored), (sname, key)
    cmp = {}
    for a in ARMS:
        if a != "C0_refit":
            cmp[f"{a}|C0_refit"] = paired_ratio({t: sc[a][t] for t in tags}, {t: sc["C0_refit"][t] for t in tags})
        if a != "M_bias":
            cmp[f"M_bias|{a}"] = paired_ratio({t: sc["M_bias"][t] for t in tags}, {t: sc[a][t] for t in tags})
    assert same_cmp(cmp["M_bias|C0_refit"], dec["P_b"]["gmr"])
    for x in ("B1", "B2", "B3_d3"):
        assert same_cmp(cmp[f"M_bias|{x}"], dec["secondary"]["baselines"][f"M_bias_vs_{x}"]), x
        assert same_cmp(cmp[f"{x}|C0_refit"], dec["secondary"]["baselines"][f"{x}_vs_C0_refit"]), x
    # subsets (decision D2 and prereg secondary: per family, per bin stratum)
    forced = [t for t in tags if not bins[t]["covered"]]
    sub_sets = {"all 57 DEV": tags, "nMOS": subsets["nmos"], "pMOS": subsets["pmos"],
                "native bin (covered, D2)": cov, "forced nearest bin": forced,
                "nMOS native bin": [t for t in cov if fam[t] == "nmos"],
                "nMOS forced bin": [t for t in forced if fam[t] == "nmos"],
                "pMOS native bin": [t for t in cov if fam[t] == "pmos"],
                "pMOS forced bin": [t for t in forced if fam[t] == "pmos"]}
    subs = {k: {"cmp": paired_ratio({t: sc["M_bias"][t] for t in v}, {t: sc["C0_refit"][t] for t in v}),
                "n_dev": len(v), "n_excl": sum(t in excl for t in v),
                "b3": paired_ratio({t: sc["B3_d3"][t] for t in v}, {t: sc["C0_refit"][t] for t in v}),
                "mb3": paired_ratio({t: sc["M_bias"][t] for t in v}, {t: sc["B3_d3"][t] for t in v})}
            for k, v in sub_sets.items()}
    assert same_cmp(subs["native bin (covered, D2)"]["cmp"], dec["secondary"]["covered_subset"])
    # y_D, per family, P_a, P_c, P_b, validity, gates
    lr = {t: math.log(sc["M_bias"][t] / sc["C0_refit"][t]) for t in finite}
    y = float(np.mean(list(lr.values())))
    assert close(y, dec["P_a"]["y_D"]) and close(math.exp(y), dec["P_b"]["gmr"]["ratio"])
    yf = {f: float(np.mean([lr[t] for t in lr if fam[t] == f])) for f in ("nmos", "pmos")}
    for f in yf:
        assert close(yf[f], dec["secondary"]["per_family"][f]["y"])
        assert dec["secondary"]["per_family"][f]["n"] == sum(fam[t] == f for t in lr)
    rule = p71.load(EXP / "I7_1/prereg_amendment_2026-10-20.json")["frozen_rule"]
    assert rule == pred1["frozen_rule"] == pred2["frozen_rule"] == dec["rule"]
    # S statistics: dataset value = mean of the per-fold values
    assert sorted(pred1["per_fold"]) == tags and sorted(pred2["per_fold"]) == finite
    s1v1 = float(np.mean([pred1["per_fold"][t]["S1_lodo"] for t in tags]))
    s1v2 = float(np.mean([pred2["per_fold"][t]["S1_lodo"] for t in finite]))
    s3v2 = float(np.mean([pred2["per_fold"][t]["S3"] for t in finite]))
    assert close(s1v1, pred1["dataset"]["S1_lodo"]) and close(s1v2, pred2["dataset"]["S1_lodo"])
    assert close(s3v2, pred2["dataset"]["S3"]) and pred1["dataset"]["S3"] is None
    for k in ("S1_pooled", "S2"):
        assert close(float(np.mean([pred1["per_fold"][t][k] for t in tags])), pred1["dataset"][k])
        assert close(float(np.mean([pred2["per_fold"][t][k] for t in finite])), pred2["dataset"][k])
    pa_y = rule_predict(rule, s1v1)
    assert close(pa_y, pred1["dataset"]["predicted_y"]) and close(pa_y, dec["P_a"]["predicted_y"])
    assert close(rule_predict(rule, s1v2), dec["P_a_secondary_recomputed"]["predicted_y"])
    sign = lambda v: "gain" if v < 0 else "no gain"
    assert pred1["P_a"]["predicted_sign"] == sign(pa_y) == dec["P_a"]["predicted"] == "gain"
    assert dec["P_a"]["observed"] == sign(y) == "gain" and dec["P_a"]["correct"]
    assert close(abs(y - pa_y), dec["P_a"]["abs_error_y"])
    assert pred2["P_c"]["predicted_sign"] == sign(s3v2) == "gain" and dec["P_c"]["correct"]
    assert close(dec["P_c"]["S3_D"], s3v2)
    for f in ("nmos", "pmos"):
        pf = pred2["per_family_secondary"][f]
        sel = [t for t in finite if fam[t] == f]
        assert close(pf["S1_lodo"], float(np.mean([pred2["per_fold"][t]["S1_lodo_family"] for t in sel])))
        assert close(pf["S3"], float(np.mean([pred2["per_fold"][t]["S3_family"] for t in sel])))
        assert dec["secondary"]["per_family"][f]["rule_correct"] == (pf["P_a_predicted_sign"] == sign(yf[f]))
        assert dec["secondary"]["per_family"][f]["S3_correct"] == (pf["P_c_predicted_sign"] == sign(yf[f]))
    gmr = cmp["M_bias|C0_refit"]
    r1 = gmr["ci_high"] < 1
    r2 = agg["M_bias"]["valid_only_mean"] <= agg["T1_inloop"]["valid_only_mean"] + T1_PLUS
    r3 = all(math.isfinite(sc["M_bias"][t]) for t in finite)
    pb = "supported" if (r1 and r2 and r3) else "gain, inconclusive" if r1 else \
        "harm" if gmr["ci_low"] > 1 else "not supported"
    assert (r1, r2, r3) == (dec["P_b"]["R1"], dec["P_b"]["R2"], dec["P_b"]["R3"]) and pb == dec["P_b"]["label"]
    n_ok_start = sum(1 for t in tags if start["rows"][t]["start_rrms"] is not None and start["rows"][t]["n_curves"] >= 2)
    v1 = n_ok_start >= 0.9 * len(tags)
    v2 = all(agg[a]["n_failed"] <= 0.1 * len(tags) for a in ("C0_refit", "M_bias")) and gmr["n_used"] >= 45
    assert v1 == dec["validity"]["V1"] and v2 == dec["validity"]["V2"] and dec["validity"]["V3_prospective"]
    assert dec["validity"]["V4_n_timeout_start"] == start["n_sim_timeout"] == 1
    assert dec["validity"]["V4_dropped"] == [] and lodo["n_sim_timeout"] == 0 == base["n_sim_timeout"]
    timeout_tag = [t for t in tags if start["rows"][t]["n_sim_timeout"]]
    assert timeout_tag == ["pfet_01v8_lvt_W1_L0p35_m28p6"]
    for g, choice in pred2["gates"].items():
        assert choice == "M_bias" and dec["secondary"]["gates"][g]["choice"] == choice
        assert close(dec["secondary"]["gates"][g]["gmr_vs_C0_refit"], math.exp(y))
    g1_v1 = pred1["gates"]["G1"]
    # guards recorded in lodo / baselines, commit hashes
    for res in (lodo, base):
        g = res["prediction_guard"]
        assert g["ok"] and g["commit"].startswith(COMMITS["prediction"]) and g["commit_v2"].startswith(COMMITS["prediction_v2"])
        assert g["amendment"]["commit"].startswith(COMMITS["amendment"])
    assert dec["prediction_commit"].startswith(COMMITS["prediction"])
    assert dec["prediction_v2_commit"].startswith(COMMITS["prediction_v2"])
    assert dec["checks"]["S1_v1_abs_diff"] == 0.0 and dec["checks"]["S1_v2_abs_diff"] == 0.0
    # magnitude: frozen rule's in-sample residual RMS over the 11 stored datasets (I7-1 step 2)
    d71 = p71.load_all()
    rms_rule = float(np.sqrt(np.mean([v["S1_lodo"] ** 2 for v in d71["resid"].values()])))
    # fold level (prereg secondary, computed here): Spearman(S1_lodo_f, y_f) over finite devices
    fold_rho = {"v1": spearman([pred1["per_fold"][t]["S1_lodo"] for t in finite], [lr[t] for t in finite]),
                "v2": spearman([pred2["per_fold"][t]["S1_lodo"] for t in finite], [lr[t] for t in finite]),
                "S3": spearman([pred2["per_fold"][t]["S3"] for t in finite], [lr[t] for t in finite])}
    # B3_d3 share of the log gain
    share = math.log(cmp["B3_d3|C0_refit"]["ratio"]) / math.log(gmr["ratio"])
    # stock card over the finite devices
    stock = {t: start["rows"][t]["stock_rrms"] for t in tags}
    return {"prereg": prereg, "amend": amend, "pred1": pred1, "pred2": pred2, "dec": dec, "split": split,
            "start": start, "bins": bins, "lodo": lodo, "base": base, "tags": tags, "finite": finite,
            "excl": excl, "sc": sc, "fam": fam, "cov": cov, "forced": forced, "agg": agg, "fam_mean": fam_mean,
            "cmp": cmp, "subs": subs, "lr": lr, "y": y, "yf": yf, "rule": rule, "s1v1": s1v1, "s1v2": s1v2,
            "s3v2": s3v2, "pa_y": pa_y, "pb": pb, "r": (r1, r2, r3), "v": (v1, v2), "n_ok_start": n_ok_start,
            "timeout_tag": timeout_tag[0], "g1_v1": g1_v1, "d71": d71, "rms_rule": rms_rule,
            "fold_rho": fold_rho, "share": share, "stock": stock}


# ================================================================== tables

def t_decision(d: dict) -> rf.Table:
    dec, p1, p2, agg = d["dec"], d["pred1"], d["pred2"], d["agg"]
    r1, r2, r3 = d["r"]
    g = d["cmp"]["M_bias|C0_refit"]
    pf = p2["per_family_secondary"]
    within = abs(d["y"] - d["pa_y"]) <= d["rms_rule"]
    fr = d["fold_rho"]
    rows = [
        {"item": "P_a (decides)", "cond": "sign of y_D vs the frozen I7-1 rule y = a + b S1_lodo_D, S1_lodo_D from "
         f"prediction.json (commit {COMMITS['prediction']}, before any held-out refit)",
         "val": f"S1_lodo_D {d['s1v1']:.5f} -> predicted y {d['pa_y']:+.4f} (gain); observed y_D {d['y']:+.4f} "
                f"(GMR {math.exp(d['y']):.3f})", "res": "correct"},
        {"item": "P_a magnitude", "cond": "|y_D - predicted y| and whether it is within the frozen rule's in-sample "
         "residual RMS over the 11 stored I7-1 datasets",
         "val": f"{abs(d['y'] - d['pa_y']):.4f}; residual RMS {d['rms_rule']:.4f}", "res": "within" if within else "outside"},
        {"item": "P_a, A1 recomputed (secondary)", "cond": "S1_lodo_D over the 53 finite starting records "
         "(prediction_v2.json)", "val": f"S1_lodo_D {d['s1v2']:.5f} -> predicted y {rule_predict(d['rule'], d['s1v2']):+.4f} (gain)",
         "res": "correct"},
        {"item": "P_b (decides)", "cond": "R1: device-bootstrap 95% CI upper of GMR(M_bias, C0_refit) < 1; R2: M_bias "
         "mean <= T1_inloop mean + 0.005 (finite starts); R3: 0 M_bias failures among finite starts",
         "val": f"R1 {ci_text(g)} wins; R2 {agg['M_bias']['valid_only_mean']:.5f} vs "
                f"{agg['T1_inloop']['valid_only_mean']:.5f} + 0.005; R3 0 of {len(d['finite'])}",
         "res": d["pb"]},
        {"item": "P_c (secondary, prospective)", "cond": f"gain iff S3_D < 0; S3_D from prediction_v2.json (commit "
         f"{COMMITS['prediction_v2']}); the prediction.json P_c is void (S3 NaN)",
         "val": f"S3_D {d['s3v2']:+.4f} -> gain; observed gain", "res": "correct"},
        {"item": "validity V1", "cond": ">= 90% of the 57 DEV devices with a finite starting-card RRMS and >= 2 "
         "included curves", "val": f"{d['n_ok_start']}/57 (4 start failures, amendment A1)", "res": "holds"},
        {"item": "validity V2", "cond": "C0_refit and M_bias <= 10% failed devices and >= 45 paired devices",
         "val": f"failed C0_refit {agg['C0_refit']['n_failed']}, M_bias {agg['M_bias']['n_failed']}; paired "
                f"{g['n_used']}", "res": "holds"},
        {"item": "validity V3", "cond": "prediction.json committed before any lodo output; commit an ancestor of the "
         "results commit", "val": f"lodo / baselines guard: prediction {COMMITS['prediction']}, prediction_v2 "
                                  f"{COMMITS['prediction_v2']}, amendment {COMMITS['amendment']} tracked, unmodified, "
                                  "ancestors of HEAD", "res": "holds (prospective)"},
        {"item": "validity V4", "cond": "ngspice timeouts reported; devices with a timeout in an arm dropped",
         "val": f"start job 1 ({d['timeout_tag']}, during its starting fit; record finite); lodo 0, baselines 0; "
                "dropped 0", "res": "reported"},
        {"item": "amendment A1", "cond": f"commit {COMMITS['amendment']}, after the prediction commit and before any "
         "held-out outcome", "val": "4 nMOS starting records with a NaN Id-Vg curve (forced bins 0, 1, 1, 4; binned "
                                    "PCLM < 0 stops BSIM4 at the first analysis): excluded from every training set and "
                                    "S fold, scored as failed in every arm", "res": "applied"},
        {"item": "per family (secondary)", "cond": "sign(y_fam) vs the frozen rule on S1_lodo_fam and vs sign(S3_fam) "
         "(prediction_v2.json)",
         "val": "; ".join(f"{'nMOS' if f == 'nmos' else 'pMOS'} y {d['yf'][f]:+.4f} ({dec['secondary']['per_family'][f]['n']} "
                          f"devices), rule {pf[f]['predicted_y_rule']:+.4f}, S3 {pf[f]['S3']:+.4f}"
                          for f in ("nmos", "pmos")), "res": "correct / correct (both)"},
        {"item": "covered subset (D2 secondary)", "cond": "GMR(M_bias, C0_refit) on the 25 bin-covered DEV devices",
         "val": ci_text(d["subs"]["native bin (covered, D2)"]["cmp"]) + " wins", "res": "reported"},
        {"item": "gates G, G1 (I8-2 definitions)", "cond": "G: M_bias iff S3_D < 0; G1: M_bias iff frozen-rule y < 0",
         "val": f"both choose M_bias (prediction_v2); GMR(G, C0_refit) {math.exp(d['y']):.3f}; prediction.json G1 "
                f"{d['g1_v1']}", "res": "chose the lower arm"},
        {"item": "S1_pooled, S2", "cond": "dataset values (reported)",
         "val": f"prediction.json {p1['dataset']['S1_pooled']:.4f}, {p1['dataset']['S2']:.4f}; prediction_v2 "
                f"{p2['dataset']['S1_pooled']:.4f}, {p2['dataset']['S2']:.4f}", "res": "reported"},
        {"item": "fold level (secondary, computed here)", "cond": "Spearman(S_f, y_f) over the 53 scored devices "
         "(y_f = ln(M_bias / C0_refit) of the fold's held-out device)",
         "val": f"S1_lodo v1 {fr['v1']:+.3f}, v2 {fr['v2']:+.3f}; S3 {fr['S3']:+.3f}", "res": "reported"},
    ]
    cols = [{"key": "item", "head": "item", "md": "item", "kind": "text"},
            {"key": "cond", "head": "pre-registered definition", "md": "pre-registered definition", "kind": "text"},
            {"key": "val", "head": "value", "md": "value", "kind": "text"},
            {"key": "res", "head": "outcome", "md": "outcome", "kind": "text"}]
    notes = [
        f"Pre-registration {rel(PREREG)} (in force from commit {COMMITS['prereg']}); prediction {rel(PRED1)} (commit "
        f"{COMMITS['prediction']}); amendment A1 {rel(AMEND)} (commit {COMMITS['amendment']}); prediction_v2 "
        f"{rel(PRED2)} (commit {COMMITS['prediction_v2']}); decision {rel(DEC)}. Every value is recomputed here from "
        "the per-device and per-fold values and checked against the stored files.",
        "y_D = mean over the 53 scored DEV devices of ln(RRMS_M_bias / RRMS_C0_refit) (strict paper RRMS after the "
        "held-out card refit, leave-one-device-out); negative = gain. Frozen rule (I7-1 amendment): y = "
        f"{d['rule']['a']:.6f} {d['rule']['b']:+.6f} S1_lodo, gain iff S1_lodo > {d['rule']['zero_crossing_S1_lodo']:.4f}.",
        "Order: prereg in force -> start (411 s) -> stats (1264 s) -> prediction commit -> lodo stopped before output "
        "(NaN corrections from the 4 failed records) -> amendment A1 commit -> stats_v2 (992 s) -> prediction_v2 "
        "commit -> lodo (1016 s) -> baselines (393 s) -> decide.",
    ]
    return rf.Table("01_table_i8_1_decision", "I8-1 NIST sky130 4 K DEV prospective test: P_a, P_b, P_c, validity, "
                    "amendment A1 and secondary outcomes", cols, rows, notes)


def t_subsets(d: dict) -> rf.Table:
    rows = []
    for k, s in d["subs"].items():
        c, b3, mb3 = s["cmp"], s["b3"], s["mb3"]
        rows.append({"sub": k, "n": str(s["n_dev"]), "ex": str(s["n_excl"]), "used": str(c["n_used"]),
                     "y": f"{math.log(c['ratio']):+.4f}", "gmr": f"{c['ratio']:.3f}",
                     "ci": f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]", "w": f"{c['wins']}/{c['n_used']}",
                     "b3": f"{b3['ratio']:.3f} [{b3['ci_low']:.3f}, {b3['ci_high']:.3f}]",
                     "mb3": f"{mb3['ratio']:.3f} [{mb3['ci_low']:.3f}, {mb3['ci_high']:.3f}]"})
    cols = [{"key": "sub", "head": "subset", "md": "subset", "kind": "text"},
            {"key": "n", "head": "DEV", "md": "DEV devices", "kind": "rtext"},
            {"key": "ex", "head": "A1 excl.", "md": "start failures (A1)", "kind": "rtext"},
            {"key": "used", "head": "paired", "md": "paired devices", "kind": "rtext"},
            {"key": "y", "head": "y", "md": "y = ln GMR(M_bias, C0_refit)", "kind": "rtext",
             "group": "GMR(M_bias, C0_refit)"},
            {"key": "gmr", "head": "GMR", "md": "GMR(M_bias, C0_refit)", "kind": "rtext", "group": "GMR(M_bias, C0_refit)"},
            {"key": "ci", "head": "95% CI", "md": "device-bootstrap 95% CI", "kind": "rtext",
             "group": "GMR(M_bias, C0_refit)"},
            {"key": "w", "head": "wins", "md": "wins", "kind": "rtext", "group": "GMR(M_bias, C0_refit)"},
            {"key": "b3", "head": "GMR(B3_d3, C0_refit)", "md": "GMR(B3_d3, C0_refit) [CI]", "kind": "rtext",
             "family": True},
            {"key": "mb3", "head": "GMR(M_bias, B3_d3)", "md": "GMR(M_bias, B3_d3) [CI]", "kind": "rtext",
             "family": True}]
    notes = [
        "Pre-registered secondary (not deciding): per family and per bin stratum (native bin = covered geometry, "
        "decision D2; forced nearest bin otherwise). The covered-subset row equals decision.json secondary."
        "covered_subset. The 4 start failures are nMOS forced-bin devices. The family x stratum rows and the B3_d3 "
        "columns are computed here (post hoc subdivision of the pre-registered strata).",
        "Device-bootstrap CI: 10000 resamples, numpy default_rng(0) (i3_2_bias.paired_ratio).",
    ]
    return rf.Table("04_table_i8_1_subsets", "I8-1 GMR(M_bias, C0_refit) per family and bin stratum, with B3_d3",
                    cols, rows, notes)


def t_arms(d: dict) -> rf.Table:
    agg, cmp, fm = d["agg"], d["cmp"], d["fam_mean"]
    stock = [d["stock"][t] for t in d["finite"]]
    rows = [{"arm": "stock 77 K card (no fit)", "nf": f"{len(stock)}/0", "mean": float(np.mean(stock)),
             "med": float(np.median(stock)), "comb": float(np.mean([np.mean([d['stock'][t] for t in d['finite']
                                                                          if d['fam'][t] == f]) for f in ('nmos', 'pmos')])),
             "n": float(np.mean([d['stock'][t] for t in d['finite'] if d['fam'][t] == 'nmos'])),
             "p": float(np.mean([d['stock'][t] for t in d['finite'] if d['fam'][t] == 'pmos'])),
             "vc": "", "vm": ""}]
    for a in ARMS:
        g = agg[a]
        vc = "1 (control)" if a == "C0_refit" else ci_text(cmp[f"{a}|C0_refit"])
        vm = "1" if a == "M_bias" else ci_text(cmp[f"M_bias|{a}"])
        rows.append({"arm": ARM_LABEL[a], "nf": f"{g['n'] - g['n_failed']}/{g['n_failed']}",
                     "mean": g["valid_only_mean"], "med": g["median"], "comb": g["combined"],
                     "n": fm[a]["nmos"], "p": fm[a]["pmos"], "vc": vc, "vm": vm})
    cols = [{"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "nf", "head": "finite/failed", "md": "devices finite / failed", "kind": "rtext"},
            {"key": "mean", "head": "mean", "md": "mean RRMS (finite)", "kind": "rrms", "group": "RRMS, 53 finite"},
            {"key": "med", "head": "median", "md": "median RRMS", "kind": "rrms", "group": "RRMS, 53 finite"},
            {"key": "comb", "head": "combined", "md": "combined (nMOS / pMOS family means)", "kind": "rrms",
             "group": "RRMS, 53 finite"},
            {"key": "n", "head": "nMOS", "md": "nMOS mean (29)", "kind": "rrms", "group": "family mean", "family": True},
            {"key": "p", "head": "pMOS", "md": "pMOS mean (24)", "kind": "rrms", "group": "family mean", "family": True},
            {"key": "vc", "head": "GMR(arm, C0_refit) [CI], wins", "md": "GMR(arm, C0_refit) [95% CI], wins",
             "kind": "rtext", "group": "paired, 53 devices"},
            {"key": "vm", "head": "GMR(M_bias, arm) [CI], wins", "md": "GMR(M_bias, arm) [95% CI], wins",
             "kind": "rtext", "group": "paired, 53 devices"}]
    b3 = d["cmp"]["B3_d3|C0_refit"]
    rext = [r["rext_ohm"] for t, r in d["base"]["rows"]["B2"].items()]
    nb1 = [r["n_extra_active"] for r in d["base"]["rows"]["B1"].values()]
    notes = [
        f"Sources {rel(LODO)}, {rel(BASE)}, {rel(START)}. Every arm scores a fresh ngspice-41 simulation of its refit "
        "card (M_bias, T1_inloop, B3_d3: multiplied by exp(g) of the frozen fold correction); strict paper RRMS with "
        "the frozen measured inclusion. Means over the 53 devices with a finite starting record (the 4 start failures "
        "are NaN in every arm, amendment A1). Stock card: start/summary.json stock_rrms over the same 53 devices.",
        "B3_d3 (I4-1 baseline, i4_1_baselines.fit_poly): g = tanh(P_pol(u, v)), P_pol a polynomial of total degree 3 "
        "in the standardized u = |Vgs|/VDD, v = |Vds|/VDD (10 monomials), one coefficient vector per polarity (20 "
        "coefficients); same loss as M_bias (mean paper RRMS of the corrected training records), L2 1e-4 on the "
        "non-constant coefficients, Adam 3e-3, 2000 full-batch steps, zero initialization, one fit; held-out refit "
        "with g in the loop. M_bias: MLP 3-16-16-1 (353 weights) on (is_pmos, u, v), 5 seeds, g = mean of members.",
        f"ln GMR(B3_d3, C0_refit) / ln GMR(M_bias, C0_refit) = {math.log(b3['ratio']):.4f} / "
        f"{d['y']:.4f} = {d['share']:.3f}.",
        f"B1: {len(d['base']['b1_screen']['selected'])} extra parameters kept by the edge screen "
        f"({', '.join(d['base']['b1_screen']['selected'])}); extra active per device {min(nb1)}-{max(nb1)}. B2: fitted "
        f"Rext per side median {np.median(rext):.1f} ohm (range {min(rext):.1f}-{max(rext):.1f}). B1 / B2 / B3_d3 are "
        "reported only (not in P_a, P_b or validity).",
    ]
    return rf.Table("05_table_i8_1_arms", "I8-1 arms: RRMS aggregates and paired GMR (57 DEV devices, 53 scored)",
                    cols, rows, notes)


# ================================================================== figures

def fig_scatter(d: dict, path: Path) -> list[str]:
    d71, rule = d["d71"], d["rule"]
    fig, axs = plt.subplots(1, 2, figsize=(17, 7.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, stat in zip(axs, ("S1_lodo", "S3")):
        p71.scatter_panel(ax, d71, stat)
        if stat == "S1_lodo":
            lo, hi = ax.get_xlim()
            xs = np.linspace(lo, max(hi, d["s1v2"] + 0.03), 50)
            ax.plot(xs, rule["a"] + rule["b"] * xs, color=INK, lw=1.4)
            ax.scatter(d["s1v1"], d["y"], marker="*", s=330, color=NIST_COLOR, edgecolor=INK, lw=0.7, zorder=7,
                       label=f"NIST sky130 4 K DEV (I8-1, prospective): S1_lodo {d['s1v1']:.4f}, observed y "
                             f"{d['y']:+.4f}")
            ax.scatter(d["s1v1"], d["pa_y"], marker="D", s=70, facecolor="none", edgecolor=NIST_COLOR, lw=1.6,
                       zorder=7, label=f"NIST prediction on the frozen line (prediction.json): y {d['pa_y']:+.4f}")
            ax.plot([d["s1v1"]] * 2, [d["pa_y"], d["y"]], color=NIST_COLOR, lw=1.0, zorder=6)
            ax.scatter(d["s1v2"], d["y"], marker="*", s=120, facecolor="none", edgecolor=NIST_COLOR, lw=0.9, zorder=6,
                       label=f"NIST, S1_lodo recomputed under A1 (secondary): {d['s1v2']:.4f}")
            ax.set_xlim(lo, max(hi, d["s1v2"] + 0.03))
            ax.legend(fontsize=6.8, frameon=False, loc="lower left")
        else:
            ro = d71["rmine"]["S3"]["ols"]
            ref = ro["a"] + ro["b"] * d["s3v2"]
            ax.scatter(d["s3v2"], d["y"], marker="*", s=330, color=NIST_COLOR, edgecolor=INK, lw=0.7, zorder=7,
                       label=f"NIST sky130 4 K DEV (I8-1): S3 {d['s3v2']:+.4f} (prediction_v2.json), observed y "
                             f"{d['y']:+.4f}")
            ax.scatter(d["s3v2"], ref, marker="D", s=70, facecolor="none", edgecolor=NIST_COLOR, lw=1.6, zorder=7,
                       label=f"post hoc reference: stored-data S3 OLS line at S3_D, y {ref:+.4f} (P_c predicts the "
                             "sign only)")
            ax.plot([d["s3v2"]] * 2, [ref, d["y"]], color=NIST_COLOR, lw=1.0, zorder=6)
            ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    fig.suptitle("I8-1: the NIST sky130 4 K DEV dataset added to the I7-1 S1_lodo (left, frozen rule) and S3 (right) "
                 "scatter. Grey: 11 stored datasets; blue / orange: 20 I7-1 simulated datasets; red star: NIST 4 K",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(PRED1), rel(PRED2), rel(DEC), rel(EXP / "I7_1/retro/results.json"), rel(EXP / "I7_1/decision.json")]


def fig_per_device(d: dict, path: Path) -> list[str]:
    lr, bins, fam = d["lr"], d["bins"], d["fam"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4), dpi=DPI, gridspec_kw={"width_ratios": [10, 10, 8]})
    fig.patch.set_facecolor(BG)
    ys = list(lr.values())
    for ax, key, lab in ((axs[0], "L_um", "L (um)"), (axs[1], "W_um", "W (um)")):
        style(ax)
        for t, v in lr.items():
            b = bins[t]
            x = d["split_info"][t][key]
            col = FAM_COLOR[fam[t]]
            ax.scatter(x, v, s=42, marker="o" if fam[t] == "nmos" else "s", zorder=4, lw=1.1,
                       facecolor=col if b["covered"] else "none", edgecolor=col)
        for t in d["excl"]:
            ax.scatter(d["split_info"][t][key], max(ys) + 0.05, marker="x", s=40, color=RED, zorder=4)
        ax.axhline(0, color=INK, lw=0.7)
        ax.axhline(d["y"], color=MUTED, lw=1.0, ls="--")
        ax.set_xscale("log")
        ticks = []
        for v in sorted({d["split_info"][t][key] for t in d["tags"]}):
            if not ticks or v > 1.05 * ticks[-1]:     # W 3 and 3.01: one tick
                ticks.append(v)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=7, rotation=90)
        ax.minorticks_off()
        ax.set_ylim(min(ys) - 0.05, max(ys) + 0.12)
        ax.set_xlabel(lab, fontsize=9)
        ax.set_ylabel("ln(RRMS_M_bias / RRMS_C0_refit) per device (negative = gain)", fontsize=9)
    h = axs[0]
    for f in ("nmos", "pmos"):
        mk = "o" if f == "nmos" else "s"
        h.scatter([], [], marker=mk, s=42, color=FAM_COLOR[f], label=f"{FAM_LABEL[f]}, native bin (covered)")
        h.scatter([], [], marker=mk, s=42, facecolor="none", edgecolor=FAM_COLOR[f], lw=1.1,
                  label=f"{FAM_LABEL[f]}, forced nearest bin")
    h.scatter([], [], marker="x", s=40, color=RED, label="start failure (A1), no value; plotted at the top")
    h.plot([], [], color=MUTED, ls="--", label=f"y_D = {d['y']:+.4f}")
    h.legend(fontsize=7, frameon=True, framealpha=0.9, facecolor=BG, edgecolor="none", loc="upper center",
             bbox_to_anchor=(0.45, 0.93))
    axs[0].set_title(f"per-device log ratio vs L; {sum(v < 0 for v in ys)}/{len(ys)} devices below 0", fontsize=9.5,
                     color=INK, loc="left")
    axs[1].set_title("per-device log ratio vs W", fontsize=9.5, color=INK, loc="left")
    ax = axs[2]
    style(ax)
    names = list(d["subs"])
    for i, k in enumerate(names):
        c = d["subs"][k]["cmp"]
        yy = len(names) - 1 - i
        col = INK if i < 5 else MUTED
        ax.plot([c["ci_low"], c["ci_high"]], [yy, yy], color=col, lw=1.6)
        ax.scatter(c["ratio"], yy, color=col, s=30, zorder=4)
        ax.annotate(f"{c['ratio']:.3f}  {c['wins']}/{c['n_used']}", (c["ci_high"], yy), xytext=(5, -3),
                    textcoords="offset points", fontsize=7.2, color=col)
    ax.axvline(1, color=INK, lw=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n + (" (post hoc)" if i >= 5 else "") for i, n in enumerate(names)][::-1], fontsize=7.5)
    ax.set_xlabel("GMR(M_bias, C0_refit) with device-bootstrap 95% CI; wins", fontsize=9)
    ax.set_xlim(0.6, 1.08)
    ax.set_title("pre-registered strata (black) and family x stratum (grey)", fontsize=9.5, color=INK, loc="left")
    fig.suptitle("I8-1: per-device held-out log ratio of M_bias to C0_refit vs geometry, and GMR per family and bin "
                 "stratum (53 scored DEV devices)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO), rel(BINS), rel(DEC)]


def g_maps(path_json: Path, arm: str) -> dict:
    rows = load(path_json)
    out = {}
    for pol, flag in (("nmos", 0.0), ("pmos", 1.0)):
        gs = [g_grid(correction_from_json_any(r), flag, n=61) for k, r in rows.items() if k.startswith(arm + "|")]
        out[pol] = np.array(gs)
    return out


def g_points(d: dict) -> dict:
    """Raw features of the trimmed included points of the 53 finite DEV starting records, per polarity."""
    from cryoml.explore.e1_correction import load_devices, make_batch
    recs = [r for r in load_devices(I8 / "start/records") if r.tag in set(d["finite"])]
    assert len(recs) == 53
    phi = make_batch(recs).phi
    return {"nmos": phi[phi[:, 0] == 0.0], "pmos": phi[phi[:, 0] == 1.0]}


def g_at(path_json: Path, arm: str, phi: np.ndarray) -> np.ndarray:
    rows = load(path_json)
    return np.mean([correction_from_json_any(r).g(phi) for k, r in rows.items() if k.startswith(arm + "|")], axis=0)


def fig_g(d: dict, path: Path) -> list[str]:
    mb, b3, i32 = g_maps(LODO_CORR, "M_bias"), g_maps(BASE_CORR, "B3_d3"), g_maps(I32_CORR, "M_bias")
    assert mb["nmos"].shape[0] == 53 and b3["nmos"].shape[0] == 53 and i32["nmos"].shape[0] == 12
    rows = [("NIST 4 K, M_bias (I8-1)", mb, "53 LODO folds"),
            ("NIST 4 K, B3_d3 (I8-1)", b3, "53 LODO folds"),
            ("sky130 77 K REAL_DEV, M_bias (I3-2)", i32, "12 LODO folds")]
    lim = 0.5
    fig = plt.figure(figsize=(13, 17), dpi=DPI)
    fig.patch.set_facecolor(BG)
    gsp = fig.add_gridspec(3, 2, left=0.07, right=0.86, top=0.93, bottom=0.04, hspace=0.36, wspace=0.22)
    a = np.linspace(0, 1, 61)
    stats = {}
    pts = g_points(d)
    for i, (lab, gm, nf) in enumerate(rows):
        for j, pol in enumerate(("nmos", "pmos")):
            ax = fig.add_subplot(gsp[i, j])
            g = gm[pol].mean(axis=0)
            spread = gm[pol].max(axis=0) - gm[pol].min(axis=0)
            im = ax.pcolormesh(a, a, g.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
            cs = ax.contour(a, a, np.exp(g.T), levels=[0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5], colors=INK,
                            linewidths=0.6)
            ax.clabel(cs, fontsize=7, fmt="%.2f")
            if i < 2:   # NIST training points: trimmed included points of the 53 finite DEV records
                ax.scatter(pts[pol][:, 1], pts[pol][:, 2], s=1.5, color=INK, alpha=0.25, lw=0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("|Vgs| / VDD", fontsize=8.5)
            ax.set_ylabel("|Vds| / VDD", fontsize=8.5)
            ax.set_title(f"{lab}, {'nMOS' if pol == 'nmos' else 'pMOS'}: mean g over {nf}\nexp(mean g) "
                         f"{np.exp(g.min()):.2f}-{np.exp(g.max()):.2f}; max fold spread of g {spread.max():.3f}",
                         fontsize=8.5, color=INK, loc="left")
            stats[(i, pol)] = g
    for pol in ("nmos", "pmos"):
        phi = pts[pol]
        gm_ = g_at(LODO_CORR, "M_bias", phi)
        gb_ = g_at(BASE_CORR, "B3_d3", phi)
        gi_ = g_at(I32_CORR, "M_bias", phi)
        stats[f"n_pts_{pol}"] = int(len(phi))
        stats[f"pts_rms_mb_{pol}"] = float(np.sqrt(np.mean(gm_ ** 2)))
        stats[f"pts_rms_b3_{pol}"] = float(np.sqrt(np.mean(gb_ ** 2)))
        stats[f"pts_rms_mb_b3_{pol}"] = float(np.sqrt(np.mean((gm_ - gb_) ** 2)))
        stats[f"pts_rms_i32_{pol}"] = float(np.sqrt(np.mean(gi_ ** 2)))
        stats[f"pts_rms_mb_i32_{pol}"] = float(np.sqrt(np.mean((gm_ - gi_) ** 2)))
        stats[f"pts_corr_mb_b3_{pol}"] = float(np.corrcoef(gm_, gb_)[0, 1])
        stats[f"pts_corr_mb_i32_{pol}"] = float(np.corrcoef(gm_, gi_)[0, 1])
    cax = fig.add_axes([0.89, 0.35, 0.018, 0.3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("g (I = I_card x exp(g)); contours: exp(g)", fontsize=9)
    fig.text(0.07, 0.975, "I8-1 learned correction g over the normalized bias plane per polarity (mean of the stored "
             "fold corrections; no training).", fontsize=11, color=INK)
    fig.text(0.07, 0.957, "NIST VDD 1.8 V, sky130 77 K VDD 1.85 V. Grey dots (rows 1-2): trimmed included points of "
             "the 53 NIST DEV training records; outside them g is extrapolated.", fontsize=9.5, color=MUTED)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    d["g_stats"] = {k: v for k, v in stats.items() if isinstance(k, str)}
    return [rel(LODO_CORR), rel(BASE_CORR), rel(I32_CORR), rel(I8 / "start/records.json")]


IV_COLORS = ["#1f4e9c", "#3a7d44", "#c2571a", "#7b3fa0", "#b8860b", "#2a9d8f", "#9b2226", "#495057", "#0077b6"]


def fig_iv(tag: str, ex: dict, d: dict, path: Path) -> list[str]:
    dv = ex["devices"][tag]
    sc = d["sc"]
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    arms = (("C0", ":", 1.0, "C0 starting card (NGSpice)"),
            ("C0_refit", "-", 1.2, "C0_refit refit card (NGSpice)"),
            ("M_bias", "--", 1.3, "M_bias: refit card NGSpice output x exp(g) (applied in Python)"))
    for row, kind in enumerate(("idvg", "idvd")):
        cs = [c for c in dv["curves"] if c["kind"] == kind]
        for col, scale in enumerate(("log", "linear") if kind == "idvg" else ("linear", "log")):
            ax = axs[row, col]
            style(ax)
            for k, c in enumerate(cs):
                color = IV_COLORS[k % len(IV_COLORS)]
                x = np.abs(np.asarray(c["vg"] if kind == "idvg" else c["vd"]))
                meas = np.abs(np.asarray(c["meas"]))
                step = 3 if len(x) > 100 else 1
                ax.scatter(x[::step], meas[::step], s=9, color=color, alpha=0.9 if c["included"] else 0.3,
                           zorder=3, lw=0)
                for arm, ls, lw, _ in arms:
                    ax.plot(x, np.abs(np.asarray(c["sim"][arm])), color=color, ls=ls, lw=lw)
                bias = f"|Vd| {abs(c['fixed_V']):.2g} V" if kind == "idvg" else f"|Vg| {abs(c['fixed_V']):.2g} V"
                ax.plot([], [], color=color, marker="o", ms=4, ls="",
                        label=bias + ("" if c["included"] else " (not in RRMS)"))
            if scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-13)
            ax.set_xlabel("|Vgs| (V)" if kind == "idvg" else "|Vds| (V)", fontsize=9)
            ax.set_ylabel("|Id| (A)", fontsize=9)
            ax.set_title(f"{'Id-Vg' if kind == 'idvg' else 'Id-Vd'}, {scale} scale", fontsize=9.5, color=INK, loc="left")
            if col == 0:
                ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)
    h = axs[0, 1]
    for arm, ls, lw, lab in arms:
        h.plot([], [], color=INK, ls=ls, lw=lw, label=f"{lab}, RRMS {sc[arm][tag]:.4f}")
    h.scatter([], [], s=12, color=INK, label="measured (4 K; every 3rd point on 181-point sweeps)")
    h.legend(fontsize=7.3, frameon=False, loc="upper left")
    fig.suptitle(f"I8-1 {tag} ({'nMOS' if dv['polarity'] == 'nmos' else 'pMOS'}, W {dv['W_um']:g} um, L {dv['L_um']:g} "
                 f"um, bin {dv['bin']} {'native' if dv['covered'] else 'forced, stretch L x%.2f W x%.2f' % tuple(dv['stretch'])}"
                 f"); ln(M_bias / C0_refit) {d['lr'][tag]:+.3f}\nngspice-41 arm64, 77 K card and deck, mult = m = 1; "
                 "stored cards re-simulated (RRMS reproduced, relative difference "
                 f"{max(v['rel_diff'] for v in ex['checks'][tag].values()):.0e})",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(EXAMPLES), rel(LODO), rel(I8 / "lodo/cards.json"), rel(LODO_CORR)]


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)

    d = load_all()
    mani = load(I8 / "data/manifest.json")
    d["split_info"] = {r["tag"]: r for r in mani["devices"]}
    ex = load(EXAMPLES)
    assert ex["max_rel_diff"] == 0.0 and len(ex["devices"]) == 4
    for tag, ch in ex["checks"].items():
        for a, v in ch.items():
            assert close(v["stored"], d["sc"][a][tag]), (tag, a)
    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(t_decision(d), "Table of the pre-registered P_a, P_b, P_c, validity V1-V4, amendment A1 and the "
              "secondary outcomes (magnitude, A1-recomputed S1_lodo, per family, covered subset, gates, S1_pooled / "
              "S2, fold-level Spearman), with the commit hashes of the pre-registration, prediction, amendment and "
              "prediction_v2", [rel(PREREG), rel(AMEND), rel(PRED1), rel(PRED2), rel(DEC), rel(LODO)])
    src = fig_scatter(d, folder / "02_i8_1_scatter_S1_lodo_S3.png")
    items.append(("02_i8_1_scatter_S1_lodo_S3.png", "Two scatter panels from I7-1 (S1_lodo with the frozen rule, "
                  "left; S3 with the stored-data OLS line, right) with the NIST sky130 4 K DEV dataset added (red "
                  "star = observed y_D; open diamond = frozen-rule prediction on the left, post hoc OLS reference on "
                  "the right; small open star = A1-recomputed S1_lodo)", src))
    src = fig_per_device(d, folder / "03_i8_1_per_device_log_ratio.png")
    items.append(("03_i8_1_per_device_log_ratio.png", "Per-device ln(RRMS_M_bias / RRMS_C0_refit) vs L and W "
                  "(nMOS circles, pMOS squares; filled = native bin, open = forced nearest bin; red x = start "
                  "failure), and GMR(M_bias, C0_refit) with CI per family and bin stratum (family x stratum rows "
                  "post hoc)", src))
    add_table(t_subsets(d), "Table of GMR(M_bias, C0_refit) per family and bin stratum, with GMR(B3_d3, "
              "C0_refit) and GMR(M_bias, B3_d3)", [rel(LODO), rel(BASE), rel(BINS), rel(DEC)])
    add_table(t_arms(d), "Table of the arms (stock card, C0, C0_refit, T1_inloop, M_bias, B1, B2, B3_d3): finite / "
              "failed devices, mean, median, combined, family means, GMR vs C0_refit and GMR(M_bias, arm) with CI",
              [rel(LODO), rel(BASE), rel(START)])
    src = fig_g(d, folder / "06_i8_1_g_map.png")
    gs = d["g_stats"]
    items.append(("06_i8_1_g_map.png", "Mean learned g over the stored leave-one-device-out fold corrections on the "
                  "(|Vgs|/VDD, |Vds|/VDD) plane per polarity: NIST 4 K M_bias (53 folds), NIST 4 K B3_d3 (53 folds), "
                  "sky130 77 K REAL_DEV M_bias from I3-2 (12 folds); contours of exp(g); NIST measured curve "
                  "training points shown as grey dots", src))
    picks = list(ex["devices"])
    for k, tag in enumerate(picks):
        name = f"{7 + k:02d}_i8_1_iv_{tag}.png"
        src = fig_iv(tag, ex, d, folder / name)
        dv = ex["devices"][tag]
        items.append((name, f"Example I-V, {tag} ({'nMOS' if dv['polarity'] == 'nmos' else 'pMOS'}, "
                      f"{'native' if dv['covered'] else 'forced'} bin): measured points, NGSpice C0 (dotted), "
                      "C0_refit (solid) and M_bias (refit card NGSpice output x exp(g) applied in Python, dashed); "
                      "Id-Vg log / linear, Id-Vd linear / log. Device chosen as the one nearest the median "
                      "log ratio of its family and bin stratum", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i8_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. y = ln GMR(M_bias, C0_refit) over the scored DEV "
          "devices (strict paper RRMS after the held-out card refit); negative = gain.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    g, cmp, agg, subs = d["cmp"]["M_bias|C0_refit"], d["cmp"], d["agg"], d["subs"]
    cv, fo = subs["native bin (covered, D2)"]["cmp"], subs["forced nearest bin"]["cmp"]
    nm, pm = subs["nMOS"]["cmp"], subs["pMOS"]["cmp"]
    b3 = cmp["B3_d3|C0_refit"]
    fr = d["fold_rho"]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice fit, no training). The example I-V "
              "curves (figures 07-10) are NGSpice re-simulations of stored cards by "
              f"`scripts/research/explore/i8_1_examples.py` ({rel(EXAMPLES)}, {ex['runtime_s']} s; RRMS reproduced "
              "with relative difference 0).", "",
              "## Contents", "",
              "- I8-1: prospective DEV test on the NIST mds2-2997 sky130 transistors measured at 4 K (MPW-5 tile; "
              "nfet_01v8_lvt 33, pfet_01v8_lvt 24 DEV devices; 77 K paper card with an explicit bin per device, "
              "native for 25 covered geometries, nearest bin in (ln L, ln W) for 32; deck at 77 K; mult = m = 1; "
              "Vb = 0 curves; leave-one-device-out). Pre-registration "
              f"`{rel(PREREG)}` (commit {COMMITS['prereg']}); prediction `{rel(PRED1)}` (commit "
              f"{COMMITS['prediction']}, before any held-out refit); amendment A1 `{rel(AMEND)}` (commit "
              f"{COMMITS['amendment']}); prediction_v2 `{rel(PRED2)}` (commit {COMMITS['prediction_v2']}); decision "
              f"`{rel(DEC)}`. Queue jobs: " + ", ".join(f"{k} ({v} s)" for k, v in JOBS.items()) + "; the first "
              "I8_1_lodo run was stopped before writing output (A1).",
              f"- Validity: V1 {d['n_ok_start']}/57 finite starts, V2 failed C0_refit / M_bias 4 / 4, paired "
              f"{g['n_used']}, V3 prospective (guards recorded in lodo and baselines results), V4 1 start-job timeout "
              f"({d['timeout_tag']}; its record is finite), 0 arm timeouts.",
              "- Amendment A1: 4 nMOS forced-bin starting records have a NaN Id-Vg curve at |Vd| = 0.1 V (the forced "
              "bin's binned PCLM is negative; BSIM4 parameter checking stops the first analysis of the ngspice "
              "process). They are excluded from every training set and S fold and scored as failed in every arm.",
              f"- P_a (frozen I7-1 rule, prediction.json): S1_lodo_D {d['s1v1']:.5f}, predicted y {d['pa_y']:+.4f} "
              f"(gain); observed y_D {d['y']:+.4f}: correct. |error| {abs(d['y'] - d['pa_y']):.4f} (frozen rule's "
              f"in-sample residual RMS {d['rms_rule']:.4f}). A1-recomputed S1_lodo {d['s1v2']:.5f}, predicted y "
              f"{rule_predict(d['rule'], d['s1v2']):+.4f}: correct.",
              f"- P_b: GMR(M_bias, C0_refit) {ci_text(g)} wins; R2 M_bias {agg['M_bias']['valid_only_mean']:.5f} vs "
              f"T1_inloop {agg['T1_inloop']['valid_only_mean']:.5f} (+0.005); R3 0 M_bias failures: {d['pb']}.",
              f"- P_c (prediction_v2.json): S3_D {d['s3v2']:+.4f}, predicted gain: correct. The prediction.json P_c "
              "('no gain', from S3 = NaN) is void under A1.",
              f"- Per family: nMOS GMR {ci_text(nm)}, y {d['yf']['nmos']:+.4f}; pMOS GMR {ci_text(pm)}, y "
              f"{d['yf']['pmos']:+.4f}; rule and S3 signs correct in both. Covered subset (25 devices, D2 secondary): "
              f"{ci_text(cv)}; forced-bin devices: {ci_text(fo)}.",
              f"- Comparators: M_bias vs T1_inloop {ci_text(cmp['M_bias|T1_inloop'])}; T1_inloop vs C0_refit "
              f"{ci_text(cmp['T1_inloop|C0_refit'])}; C0_refit vs C0 {ci_text(paired_ratio({t: d['sc']['C0_refit'][t] for t in d['tags']}, {t: d['sc']['C0'][t] for t in d['tags']}))}.",
              f"- Baselines (reported, not deciding): M_bias vs B1 {ci_text(cmp['M_bias|B1'])}, vs B2 "
              f"{ci_text(cmp['M_bias|B2'])}, vs B3_d3 {ci_text(cmp['M_bias|B3_d3'])}. GMR(B3_d3, C0_refit) "
              f"{ci_text(b3)}: the cubic polynomial correction gives {d['share']:.0%} of M_bias's log gain over "
              "C0_refit; the MLP form adds a factor "
              f"{cmp['M_bias|B3_d3']['ratio']:.3f} over it (CI upper {cmp['M_bias|B3_d3']['ci_high']:.4f}).",
              "- B3_d3 (I4-1 baseline B3 at degree 3, `cryoml.explore.i4_1_baselines.fit_poly` / `PolyCorrection`): "
              "the same multiplicative form I = I_card x exp(g) and the same training loss as M_bias (mean paper RRMS "
              "of the corrected training records), with g = tanh(P_pol(u, v)), P_pol a polynomial of total degree 3 in "
              "the standardized u = |Vgs|/VDD and v = |Vds|/VDD (10 monomials), one coefficient vector per polarity "
              "(20 coefficients), L2 1e-4 on the non-constant coefficients, Adam 3e-3, 2000 full-batch steps, zero "
              "initialization, float64, one fit; trained on the fold's 52 training records and frozen; the held-out "
              "card refit runs with g in the loop (same settings as M_bias). M_bias: MLP 3-16-16-1 tanh (353 weights) "
              "on (is_pmos, u, v), 5-seed ensemble mean of g.",
              f"- Learned g (figure 06, stored fold corrections, evaluated at the trimmed included points of the "
              f"53 NIST DEV records: {gs['n_pts_nmos']} nMOS, {gs['n_pts_pmos']} pMOS points): RMS of mean g, M_bias "
              f"{gs['pts_rms_mb_nmos']:.3f} / {gs['pts_rms_mb_pmos']:.3f}, B3_d3 {gs['pts_rms_b3_nmos']:.3f} / "
              f"{gs['pts_rms_b3_pmos']:.3f}; RMS of M_bias minus B3_d3 {gs['pts_rms_mb_b3_nmos']:.3f} / "
              f"{gs['pts_rms_mb_b3_pmos']:.3f}, Pearson r {gs['pts_corr_mb_b3_nmos']:.3f} / "
              f"{gs['pts_corr_mb_b3_pmos']:.3f}; sky130 77 K I3-2 M_bias at the same points RMS "
              f"{gs['pts_rms_i32_nmos']:.3f} / {gs['pts_rms_i32_pmos']:.3f}, RMS of NIST minus 77 K "
              f"{gs['pts_rms_mb_i32_nmos']:.3f} / {gs['pts_rms_mb_i32_pmos']:.3f}, Pearson r "
              f"{gs['pts_corr_mb_i32_nmos']:.3f} / {gs['pts_corr_mb_i32_pmos']:.3f} (nMOS / pMOS). Away from the "
              "training points (low |Vgs| on pMOS) B3_d3 and M_bias differ by more (figure 06).",
              f"- Fold level (pre-registered secondary, computed here): Spearman(S1_lodo_f, y_f) {fr['v1']:+.3f} "
              f"(prediction.json folds), {fr['v2']:+.3f} (prediction_v2 folds); Spearman(S3_f, y_f) {fr['S3']:+.3f}.",
              "- Post hoc (not pre-registered): the family x bin-stratum rows of table 04 and figure 03, the B3_d3 "
              "columns of table 04, the S3 OLS reference point in figure 02, the g maps (figure 06) and the example "
              "I-V figures.",
              "- Limitations: the deck runs at 77 K (kT/q, band gap and n_i at 77 K) for 4 K data; 32 of 57 devices "
              "use a forced bin outside its geometry range; one dataset gives one binary P_a outcome.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    summary = [
        f"I8-1 NIST sky130 4 K DEV (prospective, 57 devices, 53 scored; A1: 4 nMOS forced-bin start failures): P_a "
        f"correct (predicted y {d['pa_y']:+.3f}, observed {d['y']:+.3f}); P_b {d['pb']}, GMR(M_bias, C0_refit) "
        f"{ci_text(g)}; P_c correct (S3 {d['s3v2']:+.3f}).",
        f"nMOS {nm['ratio']:.3f}, pMOS {pm['ratio']:.3f}, covered subset {cv['ratio']:.3f} [{cv['ci_low']:.3f}, "
        f"{cv['ci_high']:.3f}], forced bin {fo['ratio']:.3f}. M_bias vs T1_inloop {cmp['M_bias|T1_inloop']['ratio']:.3f}.",
        f"Baselines: M_bias vs B1 {cmp['M_bias|B1']['ratio']:.3f}, B2 {cmp['M_bias|B2']['ratio']:.3f}, B3_d3 "
        f"{ci_text(cmp['M_bias|B3_d3'])}; B3_d3 vs C0_refit {b3['ratio']:.3f} ({d['share']:.0%} of the log gain).",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    print("g stats", {k: round(v, 4) for k, v in gs.items()})
    print("fold rho", fr, "rms_rule", d["rms_rule"], "share", d["share"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
