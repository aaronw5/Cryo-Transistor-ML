#!/usr/bin/env python3
"""Report batch for exploration direction I9-1: second prospective DEV test on
the NIST mds2-2997 sky130 transistors measured at 4 K (MPW-5 tile; families
nfet_01v8 and pfet_01v8, 87 DEV devices) with the open sky130 room-temperature
tt cards (open_pdks a918dc7, native bin for every device, deck at 30 C, the
cards' tnom). P_a: sign of y_D = ln GMR(M_bias, C0_refit) from the frozen I7-1
S1_lodo rule (prediction.json, committed before any held-out refit). P_b:
GMR(M_bias, C0_refit) CI upper < 1 with R2 / R3. P_c (secondary): sign(S3_D).
Pre-registered secondaries: S_B3 = GMR(M_bias, B3_d3) and S_transfer = arm
M_bias_lvt (M_bias trained once on the 53 finite I8-1 DEV starting records).

Reads stored files only; no NGSpice fit, no training:

* ``out/research/explore/I9_1/{prereg.json, prediction.json, decision.json,
  split.json, data/manifest.json}``
* ``out/research/explore/I9_1/start/{summary.json, bins.json, records.*}``
* ``out/research/explore/I9_1/{lodo, baselines}/{results.json, corrections.json}``
* ``out/research/explore/I9_1/examples/examples.json`` (NGSpice re-simulation of
  stored cards for 3 example devices, ``scripts/research/explore/i9_1_examples.py``)
* I8-1 stored outputs (``out/research/explore/I8_1/{decision, prediction,
  prediction_v2}.json``, ``{start, lodo, baselines}/*.json``) for the I8-1 point
  and the I8-1 vs I9-1 table
* the I7-1 stored and simulated datasets through ``plot_explore_i7_1.load_all``

Checks: every per-device score in decision / lodo / baselines agrees across
files; aggregates, every stored paired ratio (all, covered, nMOS, pMOS), y_D,
the per-family y, P_a / P_b / P_c, the validity conditions, the gates, the
transfer and baseline comparisons, and the S1 / S3 dataset and per-family
values (mean of the per-fold values) and the predicted y are recomputed and
compared with the stored values. Items marked "post hoc" are not
pre-registered.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i9_1.py --date 2026-10-21
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
import plot_explore_i8_1 as p81  # noqa: E402  (g maps, I-V colors)
from cryoml.explore import i8_1_nist as m8  # noqa: E402
from cryoml.explore.i3_2_bias import paired_ratio  # noqa: E402
from cryoml.explore.i7_1_transfer import rule_predict, spearman  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i9_1"
I9 = EXP / "I9_1"
I8 = EXP / "I8_1"
PREREG = I9 / "prereg.json"
PRED = I9 / "prediction.json"
DEC = I9 / "decision.json"
SPLIT = I9 / "split.json"
START = I9 / "start/summary.json"
BINS = I9 / "start/bins.json"
LODO = I9 / "lodo/results.json"
LODO_CORR = I9 / "lodo/corrections.json"
BASE = I9 / "baselines/results.json"
BASE_CORR = I9 / "baselines/corrections.json"
EXAMPLES = I9 / "examples/examples.json"
MANIFEST = I9 / "data/manifest.json"
COMMITS = {"draft": "5246240", "prereg": "007759b", "decide_note": "13421fe", "prediction": "e937ca2",
           "start_outputs": "c813f6f"}
# queue jobs (out/research/done.txt): wall time in s
JOBS = {"I9_1_start": 472, "I9_1_stats": 4352, "I9_1_commit_prediction": 2, "I9_1_lodo": 2435,
        "I9_1_baselines": 509, "I9_1_decide": 4}
ARMS = ("C0", "C0_refit", "T1_inloop", "M_bias", "M_bias_lvt", "B1", "B2", "B3_d3")
ARM_LABEL = {"C0": "C0 starting card (no refit)", "C0_refit": "C0_refit (control)",
             "T1_inloop": "T1_inloop (per-curve gain, non-deployable)", "M_bias": "M_bias (primary)",
             "M_bias_lvt": "M_bias_lvt (trained on I8-1 lvt DEV, transfer)",
             "B1": "B1 wider BSIM4 set", "B2": "B2 Rext + self-heating", "B3_d3": "B3_d3 cubic polynomial g"}
FAM_COLOR = {"nmos": BLUE, "pmos": ORANGE}
FAM_LABEL = {"nmos": "nMOS (nfet_01v8)", "pmos": "pMOS (pfet_01v8)"}
FAM_SHORT = {"nmos": "nMOS", "pmos": "pMOS", "all": "all"}
I9_COLOR = RED
I8_COLOR = "#7b3fa0"
T1_PLUS = 0.005
N_DEV = 87
CAL_RANGE = (-0.37891158353132853, 0.5040862674260564)

close = p81.close
fam_of = p81.fam_of
nanify = p81.nanify
ci_text = p81.ci_text
same_cmp = p81.same_cmp


def ci_short(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def sign(v: float) -> str:
    return "gain" if v < 0 else "no gain"


# ================================================================== loading and checks

def load_i8() -> dict:
    """I8-1 stored values used for the comparison point and table (not rechecked here; the I8-1 report
    batch checks them)."""
    dec, p1, p2 = load(I8 / "decision.json"), load(I8 / "prediction.json"), load(I8 / "prediction_v2.json")
    lodo, base, start = load(I8 / "lodo/results.json"), load(I8 / "baselines/results.json"), \
        load(I8 / "start/summary.json")
    cmp = dict(lodo["comparisons"]["all"])
    cmp.update(base["comparisons"]["all"])
    fam = {f: {**lodo["comparisons"][f], **base["comparisons"][f]} for f in ("nmos", "pmos")}
    return {"dec": dec, "p1": p1, "p2": p2, "lodo": lodo, "base": base, "start": start, "cmp": cmp, "fam": fam}


def load_all() -> dict:
    prereg, pred, dec = load(PREREG), load(PRED), load(DEC)
    split, start, bins = load(SPLIT), load(START), load(BINS)["devices"]
    lodo, base = load(LODO), load(BASE)
    assert not dec["smoke"] and not lodo["smoke"] and not base["smoke"] and not start["smoke"] and not pred["smoke"]
    assert prereg["status"].upper().startswith("IN FORCE")
    tags = sorted(split["DEV"])
    assert len(tags) == N_DEV and sorted(bins) == tags and lodo["held_out"] == tags == base["held_out"]
    assert all(bins[t]["covered"] for t in tags) and lodo["covered"] == tags == base["covered"]
    excl = dec["excluded_start_failures"]
    assert excl == [] == pred["excluded_records"] == lodo["excluded_records"] == base["excluded_records"]
    assert start["non_finite_records"] == [] and start["n_sim_timeout"] == 0
    # per-device scores: lodo arms, baseline arms; C0 / C0_refit / M_bias identical in both files
    sc = {a: nanify(v) for a, v in lodo["per_device"].items()}
    bsc = {a: nanify(v) for a, v in base["per_device"].items()}
    for a in ("C0", "C0_refit", "M_bias"):
        assert all(close(sc[a][t], bsc[a][t]) for t in tags), a
    for a in ("B1", "B2", "B3_d3"):
        sc[a] = bsc[a]
    for a in ARMS:
        assert sorted(sc[a]) == tags and all(math.isfinite(sc[a][t]) for t in tags), a
    assert all(close(sc["C0"][t], start["rows"][t]["start_rrms"]) for t in tags)
    fam = {t: fam_of(t) for t in tags}
    agg = {a: m8.aggregate(sc[a], fam) for a in ARMS}
    for a in ARMS:
        st = (lodo if a in lodo["summary"]["all"] else base)["summary"]["all"][a]
        for k in ("n", "n_failed", "valid_only_mean", "median", "combined"):
            assert close(agg[a][k], st[k]), (a, k)
    for a in ("C0", "C0_refit", "M_bias", "T1_inloop", "M_bias_lvt"):
        assert close(agg[a]["valid_only_mean"], dec["P_b"]["means_valid_only"][a]), a
    fam_mean = {a: {f: float(np.mean([sc[a][t] for t in tags if fam[t] == f])) for f in ("nmos", "pmos")}
                for a in ARMS}
    # paired ratios: every subset and pair stored in lodo / baselines
    subsets = {"all": tags, "covered": tags, "nmos": [t for t in tags if fam[t] == "nmos"],
               "pmos": [t for t in tags if fam[t] == "pmos"]}
    for res in (lodo, base):
        for sname, sel in subsets.items():
            for key, stored in res["comparisons"][sname].items():
                a, b = key.split("_vs_")
                mine = paired_ratio({t: sc[a][t] for t in sel}, {t: sc[b][t] for t in sel})
                assert same_cmp(mine, stored), (sname, key)
    cmp = {s: {} for s in ("all", "nmos", "pmos")}
    for s in cmp:
        sel = subsets[s]
        for a in ARMS:
            if a != "C0_refit":
                cmp[s][f"{a}|C0_refit"] = paired_ratio({t: sc[a][t] for t in sel}, {t: sc["C0_refit"][t] for t in sel})
            if a != "M_bias":
                cmp[s][f"M_bias|{a}"] = paired_ratio({t: sc["M_bias"][t] for t in sel}, {t: sc[a][t] for t in sel})
        cmp[s]["M_bias_lvt|M_bias"] = paired_ratio({t: sc["M_bias_lvt"][t] for t in sel},
                                                   {t: sc["M_bias"][t] for t in sel})
    ca = cmp["all"]
    assert same_cmp(ca["M_bias|C0_refit"], dec["P_b"]["gmr"])
    for x in ("B1", "B2", "B3_d3"):
        assert same_cmp(ca[f"M_bias|{x}"], dec["secondary"]["baselines"][f"M_bias_vs_{x}"]), x
        assert same_cmp(ca[f"{x}|C0_refit"], dec["secondary"]["baselines"][f"{x}_vs_C0_refit"]), x
    assert same_cmp(ca["M_bias|B3_d3"], dec["secondary"]["M_bias_vs_B3_d3"]["GMR_M_bias_vs_B3_d3"])
    tr = dec["secondary"]["transfer_M_bias_lvt"]
    assert same_cmp(ca["M_bias_lvt|C0_refit"], tr["M_bias_lvt_vs_C0_refit"])
    assert same_cmp(ca["M_bias_lvt|M_bias"], tr["M_bias_lvt_vs_M_bias"])
    assert tr["M_bias_lvt_vs_C0_refit"]["reading"] == "A lower" and tr["M_bias_lvt_vs_M_bias"]["reading"] == "A higher"
    assert dec["secondary"]["M_bias_vs_B3_d3"]["reading"] == "A lower"
    assert dec["secondary"]["covered_subset"] == "all scored devices covered"
    # y_D, per family, P_a, P_c, P_b, validity, gates
    lr = {t: math.log(sc["M_bias"][t] / sc["C0_refit"][t]) for t in tags}
    y = float(np.mean(list(lr.values())))
    assert close(y, dec["P_a"]["y_D"]) and close(math.exp(y), dec["P_b"]["gmr"]["ratio"])
    yf = {f: float(np.mean([lr[t] for t in tags if fam[t] == f])) for f in ("nmos", "pmos")}
    for f in yf:
        assert close(yf[f], dec["secondary"]["per_family"][f]["y"])
        assert dec["secondary"]["per_family"][f]["n"] == sum(fam[t] == f for t in tags)
    rule = p71.load(EXP / "I7_1/prereg_amendment_2026-10-20.json")["frozen_rule"]
    assert rule == pred["frozen_rule"] == dec["rule"]
    assert sorted(pred["per_fold"]) == tags and pred["n_folds"] == N_DEV
    pf_ = pred["per_fold"]
    ds = {k: float(np.mean([pf_[t][k] for t in tags])) for k in ("S1_lodo", "S1_pooled", "S2", "S3")}
    for k, v in ds.items():
        assert close(v, pred["dataset"][k]), k
    s1, s3 = ds["S1_lodo"], ds["S3"]
    pa_y = rule_predict(rule, s1)
    assert close(pa_y, pred["dataset"]["predicted_y"]) and close(pa_y, dec["P_a"]["predicted_y"])
    assert pred["P_a"]["predicted_sign"] == sign(pa_y) == dec["P_a"]["predicted"] == "gain"
    assert dec["P_a"]["observed"] == sign(y) == "gain" and dec["P_a"]["correct"]
    assert close(abs(y - pa_y), dec["P_a"]["abs_error_y"])
    inside = CAL_RANGE[0] <= s1 <= CAL_RANGE[1]
    assert dec["P_a"]["calibration_range_note"]["inside"] == inside is True
    assert pred["P_c"]["predicted_sign"] == sign(s3) == "gain" and dec["P_c"]["correct"]
    assert close(dec["P_c"]["S3_D"], s3)
    pfam = pred["per_family_secondary"]
    for f in ("nmos", "pmos"):
        sel = [t for t in tags if fam[t] == f]
        assert close(pfam[f]["S1_lodo"], float(np.mean([pf_[t]["S1_lodo_family"] for t in sel])))
        assert close(pfam[f]["S3"], float(np.mean([pf_[t]["S3_family"] for t in sel])))
        assert close(pfam[f]["predicted_y_rule"], rule_predict(rule, pfam[f]["S1_lodo"]))
        assert dec["secondary"]["per_family"][f]["rule_correct"] == (pfam[f]["P_a_predicted_sign"] == sign(yf[f]))
        assert dec["secondary"]["per_family"][f]["S3_correct"] == (pfam[f]["P_c_predicted_sign"] == sign(yf[f]))
    assert dec["secondary"]["per_family"]["pmos"]["S3_correct"] is False
    g = ca["M_bias|C0_refit"]
    r1 = g["ci_high"] < 1
    r2 = agg["M_bias"]["valid_only_mean"] <= agg["T1_inloop"]["valid_only_mean"] + T1_PLUS
    r3 = all(math.isfinite(sc["M_bias"][t]) for t in tags)
    pb = "supported" if (r1 and r2 and r3) else "gain, inconclusive" if r1 else \
        "harm" if g["ci_low"] > 1 else "not supported"
    assert (r1, r2, r3) == (dec["P_b"]["R1"], dec["P_b"]["R2"], dec["P_b"]["R3"]) and pb == dec["P_b"]["label"]
    n_ok_start = sum(1 for t in tags if start["rows"][t]["start_rrms"] is not None
                     and start["rows"][t]["finite_record"] and start["rows"][t]["n_curves"] >= 2)
    v1 = n_ok_start >= 0.9 * N_DEV
    v2 = all(agg[a]["n_failed"] <= 0.1 * N_DEV for a in ("C0_refit", "M_bias")) and g["n_used"] >= 70
    vd = dec["validity"]
    assert v1 == vd["V1"] and v2 == vd["V2"] and vd["V3_prospective"] and vd["valid"] and vd["n_paired"] == N_DEV
    assert vd["V4_n_timeout_start"] == 0 and vd["V4_dropped"] == [] and lodo["n_sim_timeout"] == 0 == base["n_sim_timeout"]
    for gname, choice in pred["gates"].items():
        assert choice == "M_bias" and dec["secondary"]["gates"][gname]["choice"] == choice
        assert close(dec["secondary"]["gates"][gname]["gmr_vs_C0_refit"], math.exp(y))
    for res in (lodo, base):
        gd = res["prediction_guard"]
        assert gd["ok"] and gd["commit"].startswith(COMMITS["prediction"])
        assert gd["prereg"]["commit"].startswith(COMMITS["prereg"]) and gd["prereg_in_force"]
    assert dec["prediction_commit"].startswith(COMMITS["prediction"])
    assert dec["checks"]["S1_abs_diff"] == 0.0 and dec["checks"]["excluded_equal"]
    d71 = p71.load_all()
    rms_rule = float(np.sqrt(np.mean([v["S1_lodo"] ** 2 for v in d71["resid"].values()])))
    fold_rho = {"S1": spearman([pf_[t]["S1_lodo"] for t in tags], [lr[t] for t in tags]),
                "S3": spearman([pf_[t]["S3"] for t in tags], [lr[t] for t in tags])}
    for f in ("nmos", "pmos"):
        sel = [t for t in tags if fam[t] == f]
        fold_rho[f"S1_{f}"] = spearman([pf_[t]["S1_lodo_family"] for t in sel], [lr[t] for t in sel])
        fold_rho[f"S3_{f}"] = spearman([pf_[t]["S3_family"] for t in sel], [lr[t] for t in sel])
    share = math.log(ca["B3_d3|C0_refit"]["ratio"]) / math.log(g["ratio"])
    assert close(share, dec["secondary"]["M_bias_vs_B3_d3"]["log_gain_fraction_B3_d3"])
    share_f = {f: math.log(cmp[f]["B3_d3|C0_refit"]["ratio"]) / math.log(cmp[f]["M_bias|C0_refit"]["ratio"])
               for f in ("nmos", "pmos")}
    stock = {t: start["rows"][t]["stock_rrms"] for t in tags}
    assert close(float(np.mean(list(stock.values()))), start["stock"]["valid_only_mean"])
    mani = load(MANIFEST)
    info = {r["tag"]: r for r in mani["devices"]}
    n_inc = {t: int(info[t]["n_included_measured_rule"]) for t in tags}
    assert all(n_inc[t] == start["rows"][t]["n_curves"] for t in tags)
    return {"prereg": prereg, "pred": pred, "dec": dec, "split": split, "start": start, "bins": bins,
            "lodo": lodo, "base": base, "tags": tags, "sc": sc, "fam": fam, "agg": agg, "fam_mean": fam_mean,
            "cmp": cmp, "lr": lr, "y": y, "yf": yf, "rule": rule, "ds": ds, "s1": s1, "s3": s3, "pa_y": pa_y,
            "pb": pb, "r": (r1, r2, r3), "v": (v1, v2), "n_ok_start": n_ok_start, "inside": inside, "d71": d71,
            "rms_rule": rms_rule, "fold_rho": fold_rho, "share": share, "share_f": share_f, "stock": stock,
            "info": info, "n_inc": n_inc, "i8": load_i8()}


# ================================================================== tables

def t_decision(d: dict) -> rf.Table:
    dec, pred, agg = d["dec"], d["pred"], d["agg"]
    g = d["cmp"]["all"]["M_bias|C0_refit"]
    pf = pred["per_family_secondary"]
    fs = dec["secondary"]["per_family"]
    within = abs(d["y"] - d["pa_y"]) <= d["rms_rule"]
    ca = d["cmp"]["all"]
    rows = [
        {"item": "P_a (decides)", "cond": "sign of y_D vs the frozen I7-1 rule y = a + b S1_lodo_D, S1_lodo_D from "
         f"prediction.json (commit {COMMITS['prediction']}, before any held-out refit)",
         "val": f"S1_lodo_D {d['s1']:.5f} -> predicted y {d['pa_y']:+.4f} (gain); observed y_D {d['y']:+.4f} "
                f"(GMR {math.exp(d['y']):.3f})", "res": "correct"},
        {"item": "P_a calibration-range note (Q6)", "cond": "S1_lodo_D inside the I7-1 calibration range of the 11 "
         f"stored datasets [{CAL_RANGE[0]:.3f}, {CAL_RANGE[1]:.3f}]", "val": f"{d['s1']:.4f}",
         "res": "inside" if d["inside"] else "outside"},
        {"item": "P_a magnitude", "cond": "|y_D - predicted y| and whether it is within the frozen rule's in-sample "
         "residual RMS over the 11 stored I7-1 datasets",
         "val": f"{abs(d['y'] - d['pa_y']):.4f}; residual RMS {d['rms_rule']:.4f}",
         "res": "within" if within else "outside"},
        {"item": "P_b (decides)", "cond": "R1: device-bootstrap 95% CI upper of GMR(M_bias, C0_refit) < 1; R2: M_bias "
         "mean <= T1_inloop mean + 0.005; R3: 0 M_bias failures",
         "val": f"R1 {ci_text(g)} wins; R2 {agg['M_bias']['valid_only_mean']:.5f} vs "
                f"{agg['T1_inloop']['valid_only_mean']:.5f} + 0.005; R3 0 of {N_DEV}", "res": d["pb"]},
        {"item": "P_c (secondary, prospective)", "cond": f"gain iff S3_D < 0; S3_D from prediction.json (commit "
         f"{COMMITS['prediction']})", "val": f"S3_D {d['s3']:+.4f} -> gain; observed gain", "res": "correct"},
        {"item": "S_B3 (secondary)", "cond": "GMR(M_bias, B3_d3) with CI; 'M_bias lower' iff CI upper < 1",
         "val": f"{ci_text(ca['M_bias|B3_d3'])} wins; B3_d3 share of the log gain {d['share']:.3f}",
         "res": "M_bias lower"},
        {"item": "S_transfer (secondary)", "cond": "arm M_bias_lvt (M_bias trained once on the 53 finite I8-1 DEV "
         "starting records): GMR vs C0_refit and vs M_bias with CI",
         "val": f"vs C0_refit {ci_text(ca['M_bias_lvt|C0_refit'])}; vs M_bias {ci_text(ca['M_bias_lvt|M_bias'])}",
         "res": "lower than C0_refit, higher than M_bias"},
        {"item": "validity V1", "cond": ">= 90% of the 87 DEV devices with a finite starting record and >= 2 "
         "included curves", "val": f"{d['n_ok_start']}/{N_DEV}", "res": "holds"},
        {"item": "validity V2", "cond": "C0_refit and M_bias <= 10% failed devices and >= 70 paired devices",
         "val": f"failed C0_refit {agg['C0_refit']['n_failed']}, M_bias {agg['M_bias']['n_failed']}; paired "
                f"{g['n_used']}", "res": "holds"},
        {"item": "validity V3", "cond": "prediction.json committed before any lodo output; commit an ancestor of the "
         "results commit", "val": f"lodo / baselines guard: prereg {COMMITS['prereg']} in force, prediction "
                                  f"{COMMITS['prediction']} tracked, unmodified, ancestors of HEAD",
         "res": "holds (prospective)"},
        {"item": "validity V4", "cond": "ngspice timeouts reported; devices with a timeout in an arm dropped",
         "val": "start 0, lodo 0, baselines 0; dropped 0", "res": "reported"},
        {"item": "per family (secondary)", "cond": "sign(y_fam) vs the frozen rule on S1_lodo_fam and vs sign(S3_fam)",
         "val": "; ".join(f"{FAM_SHORT[f]} y {d['yf'][f]:+.4f} ({fs[f]['n']} devices), rule {pf[f]['predicted_y_rule']:+.4f}"
                          f" ({'correct' if fs[f]['rule_correct'] else 'wrong'}), S3 {pf[f]['S3']:+.4f} "
                          f"({'correct' if fs[f]['S3_correct'] else 'wrong'})" for f in ("nmos", "pmos")),
         "res": "rule 2/2, S3 1/2"},
        {"item": "gates G, G1 (I8-2 definitions)", "cond": "G: M_bias iff S3_D < 0; G1: M_bias iff frozen-rule y < 0",
         "val": f"both choose M_bias; GMR(G, C0_refit) {math.exp(d['y']):.3f}", "res": "chose the lower arm"},
        {"item": "S1_pooled, S2", "cond": "dataset values (reported)",
         "val": f"{d['ds']['S1_pooled']:.4f}, {d['ds']['S2']:.4f}", "res": "reported"},
        {"item": "fold level (computed here)", "cond": "Spearman(S_f, y_f) over the 87 folds (y_f = ln(M_bias / "
         "C0_refit) of the fold's held-out device)", "val": f"S1_lodo {d['fold_rho']['S1']:+.3f}; S3 "
                                                            f"{d['fold_rho']['S3']:+.3f}", "res": "reported"},
    ]
    cols = [{"key": "item", "head": "item", "md": "item", "kind": "text"},
            {"key": "cond", "head": "pre-registered definition", "md": "pre-registered definition", "kind": "text"},
            {"key": "val", "head": "value", "md": "value", "kind": "text"},
            {"key": "res", "head": "outcome", "md": "outcome", "kind": "text"}]
    notes = [
        f"Pre-registration {rel(PREREG)} (in force from commit {COMMITS['prereg']}); prediction {rel(PRED)} (commit "
        f"{COMMITS['prediction']}); decision {rel(DEC)}. Every value is recomputed here from the per-device and "
        "per-fold values and checked against the stored files.",
        "y_D = mean over the 87 DEV devices of ln(RRMS_M_bias / RRMS_C0_refit) (strict paper RRMS after the held-out "
        "card refit, leave-one-device-out); negative = gain. Frozen rule (I7-1 amendment): y = "
        f"{d['rule']['a']:.6f} {d['rule']['b']:+.6f} S1_lodo, gain iff S1_lodo > {d['rule']['zero_crossing_S1_lodo']:.4f}.",
        "Order: draft (5246240) -> prereg in force (007759b) -> start (472 s) -> stats (4352 s) -> prediction commit "
        "(e937ca2) -> lodo (2435 s) -> baselines (509 s) -> decide (4 s).",
    ]
    return rf.Table("01_table_i9_1_decision", "I9-1 NIST sky130 4 K DEV room-temperature card prospective test: P_a, "
                    "P_b, P_c, validity and secondary outcomes", cols, rows, notes)


def t_arms(d: dict) -> rf.Table:
    agg, ca, fm = d["agg"], d["cmp"]["all"], d["fam_mean"]
    st = list(d["stock"].values())
    nm = [d["stock"][t] for t in d["tags"] if d["fam"][t] == "nmos"]
    pm = [d["stock"][t] for t in d["tags"] if d["fam"][t] == "pmos"]
    rows = [{"arm": "stock room-temperature card (no fit)", "nf": f"{len(st)}/0", "mean": float(np.mean(st)),
             "med": float(np.median(st)), "comb": float(np.mean([np.mean(nm), np.mean(pm)])),
             "n": float(np.mean(nm)), "p": float(np.mean(pm)), "vc": "", "vm": ""}]
    for a in ARMS:
        g = agg[a]
        vc = "1 (control)" if a == "C0_refit" else ci_text(ca[f"{a}|C0_refit"])
        vm = "1" if a == "M_bias" else ci_text(ca[f"M_bias|{a}"])
        rows.append({"arm": ARM_LABEL[a], "nf": f"{g['n'] - g['n_failed']}/{g['n_failed']}",
                     "mean": g["valid_only_mean"], "med": g["median"], "comb": g["combined"],
                     "n": fm[a]["nmos"], "p": fm[a]["pmos"], "vc": vc, "vm": vm})
    cols = [{"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "nf", "head": "finite/failed", "md": "devices finite / failed", "kind": "rtext"},
            {"key": "mean", "head": "mean", "md": "mean RRMS", "kind": "rrms", "group": "RRMS, 87 DEV"},
            {"key": "med", "head": "median", "md": "median RRMS", "kind": "rrms", "group": "RRMS, 87 DEV"},
            {"key": "comb", "head": "combined", "md": "combined (nMOS / pMOS family means)", "kind": "rrms",
             "group": "RRMS, 87 DEV"},
            {"key": "n", "head": "nMOS", "md": "nMOS mean (48)", "kind": "rrms", "group": "family mean", "family": True},
            {"key": "p", "head": "pMOS", "md": "pMOS mean (39)", "kind": "rrms", "group": "family mean", "family": True},
            {"key": "vc", "head": "GMR(arm, C0_refit) [CI], wins", "md": "GMR(arm, C0_refit) [95% CI], wins",
             "kind": "rtext", "group": "paired, 87 devices"},
            {"key": "vm", "head": "GMR(M_bias, arm) [CI], wins", "md": "GMR(M_bias, arm) [95% CI], wins",
             "kind": "rtext", "group": "paired, 87 devices"}]
    rext = [r["rext_ohm"] for r in d["base"]["rows"]["B2"].values()]
    nb1 = [r["n_extra_active"] for r in d["base"]["rows"]["B1"].values()]
    sel = d["base"]["b1_screen"]["selected"]
    notes = [
        f"Sources {rel(LODO)}, {rel(BASE)}, {rel(START)}. Every arm scores a fresh ngspice-41 simulation of its refit "
        "card (M_bias, M_bias_lvt, T1_inloop, B3_d3: multiplied by exp(g) of the frozen correction); strict paper "
        "RRMS with the frozen measured inclusion. 0 failed devices in every arm. Stock card: start/summary.json "
        "stock_rrms.",
        "M_bias: I3-2 recipe, MLP 3-16-16-1 tanh on (is_pmos, |Vgs|/VDD, |Vds|/VDD), 5 seeds, trained per fold on the "
        "86 other DEV starting records. M_bias_lvt: the same recipe trained once on the 53 finite I8-1 DEV starting "
        "records (nfet_01v8_lvt / pfet_01v8_lvt, 77 K card, 77 K deck), applied unchanged. B3_d3: g = tanh(P_pol(u, "
        "v)), total degree 3, 20 coefficients, one fit per fold.",
        f"B1: {len(sel)} extra parameters kept by the edge screen ({', '.join(sel)}); extra active per device "
        f"{min(nb1)}-{max(nb1)}. B2: fitted Rext per side median {np.median(rext):.3f} ohm (range {min(rext):.1e}-"
        f"{max(rext):.0f}). B1 / B2 / B3_d3 are reported only (not in P_a, P_b or validity).",
    ]
    return rf.Table("04_table_i9_1_arms", "I9-1 arms: RRMS aggregates and paired GMR (87 DEV devices)",
                    cols, rows, notes)


def t_family(d: dict) -> rf.Table:
    pred, sc, fam, cmp = d["pred"], d["sc"], d["fam"], d["cmp"]
    pf = pred["per_family_secondary"]
    fs = d["dec"]["secondary"]["per_family"]
    rows = []
    for f in ("nmos", "pmos", "all"):
        sel = [t for t in d["tags"] if f == "all" or fam[t] == f]
        c = cmp[f]
        if f == "all":
            s1, s3, py = d["s1"], d["s3"], d["pa_y"]
            rc, s3c, yv = "correct", "correct", d["y"]
        else:
            s1, s3, py = pf[f]["S1_lodo"], pf[f]["S3"], pf[f]["predicted_y_rule"]
            rc = "correct" if fs[f]["rule_correct"] else "wrong"
            s3c = "correct" if fs[f]["S3_correct"] else "wrong"
            yv = d["yf"][f]
        rows.append({
            "f": {"nmos": "nMOS (nfet_01v8)", "pmos": "pMOS (pfet_01v8)", "all": "all DEV"}[f],
            "n": str(len(sel)), "inc": str(sum(d["n_inc"][t] for t in sel)),
            "c0r": float(np.mean([sc["C0_refit"][t] for t in sel])),
            "mb": float(np.mean([sc["M_bias"][t] for t in sel])),
            "s1": f"{s1:.4f}", "py": f"{py:+.4f}", "y": f"{yv:+.4f}", "rc": rc,
            "s3": f"{s3:+.4f}", "s3c": s3c,
            "gmr": ci_text(c["M_bias|C0_refit"]),
            "b3": ci_short(c["B3_d3|C0_refit"]), "mb3": ci_short(c["M_bias|B3_d3"]),
            "lvt": ci_short(c["M_bias_lvt|C0_refit"])})
    cols = [{"key": "f", "head": "family", "md": "family", "kind": "text"},
            {"key": "n", "head": "DEV", "md": "DEV devices", "kind": "rtext"},
            {"key": "inc", "head": "curves", "md": "included curves", "kind": "rtext"},
            {"key": "c0r", "head": "C0_refit", "md": "C0_refit mean RRMS", "kind": "rrms", "group": "mean RRMS"},
            {"key": "mb", "head": "M_bias", "md": "M_bias mean RRMS", "kind": "rrms", "group": "mean RRMS"},
            {"key": "s1", "head": "S1_lodo", "md": "S1_lodo (prediction.json)", "kind": "rtext",
             "group": "frozen I7-1 rule"},
            {"key": "py", "head": "pred. y", "md": "rule-predicted y", "kind": "rtext", "group": "frozen I7-1 rule"},
            {"key": "y", "head": "obs. y", "md": "observed y", "kind": "rtext", "group": "frozen I7-1 rule"},
            {"key": "rc", "head": "sign", "md": "rule sign", "kind": "text", "group": "frozen I7-1 rule"},
            {"key": "s3", "head": "S3", "md": "S3 (prediction.json)", "kind": "rtext", "group": "S3"},
            {"key": "s3c", "head": "sign", "md": "S3 sign", "kind": "text", "group": "S3"},
            {"key": "gmr", "head": "GMR(M_bias, C0_refit) [CI], wins", "md": "GMR(M_bias, C0_refit) [95% CI], wins",
             "kind": "rtext"},
            {"key": "b3", "head": "GMR(B3_d3, C0_refit)", "md": "GMR(B3_d3, C0_refit) [CI]", "kind": "rtext",
             "family": True},
            {"key": "mb3", "head": "GMR(M_bias, B3_d3)", "md": "GMR(M_bias, B3_d3) [CI]", "kind": "rtext",
             "family": True},
            {"key": "lvt", "head": "GMR(M_bias_lvt, C0_refit)", "md": "GMR(M_bias_lvt, C0_refit) [CI]",
             "kind": "rtext", "family": True}]
    notes = [
        "Pre-registered secondary (not deciding): per-family S1_lodo and S3 are the means over the family's folds of "
        "S1_lodo_family and S3_family in prediction.json (the statistic computed on the fold's training records of "
        "that family); the rule-predicted y uses the frozen I7-1 rule. The all-DEV row repeats P_a / P_c.",
        "Per-family GMR columns: device-bootstrap CI over the family's devices, 10000 resamples, numpy "
        "default_rng(0) (i3_2_bias.paired_ratio); the B3_d3 and M_bias_lvt columns per family are computed here.",
        f"B3_d3 share of the M_bias log gain over C0_refit: nMOS {d['share_f']['nmos']:.3f}, pMOS "
        f"{d['share_f']['pmos']:.3f}, all {d['share']:.3f}.",
    ]
    return rf.Table("06_table_i9_1_family", "I9-1 per family: S1_lodo rule, S3 sign, GMR(M_bias, C0_refit) and "
                    "secondary arms", cols, rows, notes)


def t_compare(d: dict) -> rf.Table:
    i8 = d["i8"]
    dec8, p1, p2, c8, f8 = i8["dec"], i8["p1"], i8["p2"], i8["cmp"], i8["fam"]
    ca = d["cmp"]["all"]
    pf8, pf9 = p2["per_family_secondary"], d["pred"]["per_family_secondary"]
    fs8, fs9 = dec8["secondary"]["per_family"], d["dec"]["secondary"]["per_family"]
    s8, s9 = i8["start"], d["start"]
    l8 = i8["lodo"]["summary"]["all"]

    def fam_txt(fs, pf, f, cmpf):
        c = cmpf
        return (f"y {fs[f]['y']:+.3f}; GMR {c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] "
                f"{c['wins']}/{c['n_used']}; rule {pf[f]['predicted_y_rule']:+.3f} "
                f"({'correct' if fs[f]['rule_correct'] else 'wrong'}); S3 {pf[f]['S3']:+.3f} "
                f"({'correct' if fs[f]['S3_correct'] else 'wrong'})")

    rows = [
        {"item": "devices", "a": "nfet_01v8_lvt / pfet_01v8_lvt; DEV 57 (33 / 24), 53 scored (4 start failures, A1)",
         "b": f"nfet_01v8 / pfet_01v8; DEV {N_DEV} (48 / 39), {N_DEV} scored"},
        {"item": "card", "a": "77 K paper card, explicit bin (25 native, 32 forced nearest bin)",
         "b": "open sky130 tt room-temperature card (open_pdks a918dc7), native bin for all 87"},
        {"item": "deck temperature", "a": "77 K", "b": "30 C (cards' tnom)"},
        {"item": "measured temperature", "a": "4 K", "b": "4 K"},
        {"item": "stock card mean RRMS", "a": f"{s8['stock']['valid_only_mean']:.4f}",
         "b": f"{s9['stock']['valid_only_mean']:.4f}"},
        {"item": "C0 / C0_refit / M_bias mean RRMS",
         "a": f"{l8['C0']['valid_only_mean']:.4f} / {l8['C0_refit']['valid_only_mean']:.4f} / "
              f"{l8['M_bias']['valid_only_mean']:.4f}",
         "b": f"{d['agg']['C0']['valid_only_mean']:.4f} / {d['agg']['C0_refit']['valid_only_mean']:.4f} / "
              f"{d['agg']['M_bias']['valid_only_mean']:.4f}"},
        {"item": "S1_lodo_D (P_a input)", "a": f"{p1['dataset']['S1_lodo']:.4f} (prediction.json)",
         "b": f"{d['s1']:.4f}"},
        {"item": "P_a: predicted y / observed y", "a": f"{dec8['P_a']['predicted_y']:+.4f} / {dec8['P_a']['y_D']:+.4f}; "
         f"|error| {dec8['P_a']['abs_error_y']:.4f}; correct",
         "b": f"{d['pa_y']:+.4f} / {d['y']:+.4f}; |error| {abs(d['y'] - d['pa_y']):.4f}; correct"},
        {"item": "P_b: GMR(M_bias, C0_refit)", "a": ci_text(c8["M_bias_vs_C0_refit"]) + "; supported",
         "b": ci_text(ca["M_bias|C0_refit"]) + "; supported"},
        {"item": "P_c: S3_D", "a": f"{p2['dataset']['S3']:+.4f} (prediction_v2.json); correct",
         "b": f"{d['s3']:+.4f}; correct"},
        {"item": "nMOS", "a": fam_txt(fs8, pf8, "nmos", f8["nmos"]["M_bias_vs_C0_refit"]),
         "b": fam_txt(fs9, pf9, "nmos", d["cmp"]["nmos"]["M_bias|C0_refit"])},
        {"item": "pMOS", "a": fam_txt(fs8, pf8, "pmos", f8["pmos"]["M_bias_vs_C0_refit"]),
         "b": fam_txt(fs9, pf9, "pmos", d["cmp"]["pmos"]["M_bias|C0_refit"])},
        {"item": "M_bias vs T1_inloop", "a": ci_text(c8["M_bias_vs_T1_inloop"]), "b": ci_text(ca["M_bias|T1_inloop"])},
        {"item": "M_bias vs B1 / B2", "a": f"{ci_short(c8['M_bias_vs_B1'])} / {ci_short(c8['M_bias_vs_B2'])}",
         "b": f"{ci_short(ca['M_bias|B1'])} / {ci_short(ca['M_bias|B2'])}"},
        {"item": "M_bias vs B3_d3 (S_B3)", "a": ci_text(c8["M_bias_vs_B3_d3"]), "b": ci_text(ca["M_bias|B3_d3"])},
        {"item": "B3_d3 share of the log gain",
         "a": f"{math.log(c8['B3_d3_vs_C0_refit']['ratio']) / math.log(c8['M_bias_vs_C0_refit']['ratio']):.3f}",
         "b": f"{d['share']:.3f}"},
        {"item": "M_bias_lvt (transfer)", "a": "not run (training source)",
         "b": f"vs C0_refit {ci_short(ca['M_bias_lvt|C0_refit'])}; vs M_bias {ci_short(ca['M_bias_lvt|M_bias'])}"},
        {"item": "queue wall time", "a": "start 411 s, stats 1264 + 992 s, lodo 1016 s, baselines 393 s",
         "b": f"start {JOBS['I9_1_start']} s, stats {JOBS['I9_1_stats']} s, lodo {JOBS['I9_1_lodo']} s, "
              f"baselines {JOBS['I9_1_baselines']} s"},
    ]
    cols = [{"key": "item", "head": "item", "md": "item", "kind": "text"},
            {"key": "a", "head": "I8-1 (lvt families, 77 K card)", "md": "I8-1", "kind": "text"},
            {"key": "b", "head": "I9-1 (regular-Vt families, room-temperature card)", "md": "I9-1", "kind": "text"}]
    notes = [
        f"I8-1 values from {rel(I8 / 'decision.json')}, {rel(I8 / 'prediction.json')}, "
        f"{rel(I8 / 'prediction_v2.json')}, {rel(I8 / 'lodo/results.json')}, {rel(I8 / 'baselines/results.json')}, "
        f"{rel(I8 / 'start/summary.json')} (checked by plot_explore_i8_1.py); I9-1 values recomputed here.",
        "Both: NIST mds2-2997 MPW-5 tile at 4 K, Vb = 0 curves (9 per device), measured-only inclusion, the 43-parameter "
        "box, leave-one-device-out, strict paper RRMS, ngspice-41 arm64, mult = m = 1. Means over the scored devices.",
        "I8-1 per-family S1_lodo / S3 values come from prediction_v2.json (amendment A1); I9-1 from prediction.json.",
    ]
    return rf.Table("10_table_i9_1_vs_i8_1", "I8-1 (NIST 4 K, lvt, 77 K card) vs I9-1 (NIST 4 K, regular Vt, "
                    "room-temperature card)", cols, rows, notes)


# ================================================================== figures

def fig_scatter(d: dict, path: Path) -> list[str]:
    d71, rule, i8 = d["d71"], d["rule"], d["i8"]
    pf = d["pred"]["per_family_secondary"]
    s1_8, y8, py8 = i8["p1"]["dataset"]["S1_lodo"], i8["dec"]["P_a"]["y_D"], i8["dec"]["P_a"]["predicted_y"]
    s3_8 = i8["p2"]["dataset"]["S3"]
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, stat in zip(axs, ("S1_lodo", "S3")):
        p71.scatter_panel(ax, d71, stat)
        if stat == "S1_lodo":
            lo, hi = ax.get_xlim()
            ax.axvspan(CAL_RANGE[0], CAL_RANGE[1], color=GREY, alpha=0.10, lw=0, zorder=0)
            ax.scatter([], [], marker="s", s=60, color=GREY, alpha=0.25,
                       label=f"I7-1 calibration range of S1_lodo [{CAL_RANGE[0]:.3f}, {CAL_RANGE[1]:.3f}]")
            ax.scatter(s1_8, y8, marker="*", s=260, color=I8_COLOR, edgecolor=INK, lw=0.6, zorder=7,
                       label=f"I8-1 NIST 4 K lvt, 77 K card: S1_lodo {s1_8:.4f}, observed y {y8:+.4f}")
            ax.scatter(s1_8, py8, marker="D", s=55, facecolor="none", edgecolor=I8_COLOR, lw=1.4, zorder=7)
            ax.plot([s1_8] * 2, [py8, y8], color=I8_COLOR, lw=1.0, zorder=6)
            ax.scatter(d["s1"], d["y"], marker="*", s=360, color=I9_COLOR, edgecolor=INK, lw=0.7, zorder=8,
                       label=f"I9-1 NIST 4 K regular Vt, room-temperature card (prospective): S1_lodo {d['s1']:.4f}, "
                             f"observed y {d['y']:+.4f}")
            ax.scatter(d["s1"], d["pa_y"], marker="D", s=70, facecolor="none", edgecolor=I9_COLOR, lw=1.6, zorder=8,
                       label=f"open diamonds: frozen-rule predictions (I9-1 y {d['pa_y']:+.4f}, I8-1 {py8:+.4f})")
            ax.plot([d["s1"]] * 2, [d["pa_y"], d["y"]], color=I9_COLOR, lw=1.1, zorder=6)
            for f, mk in (("nmos", "o"), ("pmos", "s")):
                ax.scatter(pf[f]["S1_lodo"], d["yf"][f], marker=mk, s=60, facecolor="none", edgecolor=I9_COLOR,
                           lw=1.3, zorder=7)
                ax.annotate(f"I9-1 {FAM_SHORT[f]}", (pf[f]["S1_lodo"], d["yf"][f]), xytext=(6, -3),
                            textcoords="offset points", fontsize=7, color=I9_COLOR)
            ax.scatter([], [], marker="o", s=50, facecolor="none", edgecolor=I9_COLOR, lw=1.3,
                       label="I9-1 per family (secondary): S1_lodo_fam vs observed y_fam (circle nMOS, square pMOS)")
            ax.set_xlim(lo, max(hi, pf["nmos"]["S1_lodo"] + 0.05))
            ax.legend(fontsize=6.6, frameon=False, loc="lower left")
        else:
            ro = d71["rmine"]["S3"]["ols"]
            ax.scatter(s3_8, y8, marker="*", s=260, color=I8_COLOR, edgecolor=INK, lw=0.6, zorder=7,
                       label=f"I8-1 NIST 4 K lvt: S3 {s3_8:+.4f} (prediction_v2.json), observed y {y8:+.4f}")
            ax.scatter(d["s3"], d["y"], marker="*", s=360, color=I9_COLOR, edgecolor=INK, lw=0.7, zorder=8,
                       label=f"I9-1 NIST 4 K regular Vt: S3 {d['s3']:+.4f} (prediction.json), observed y {d['y']:+.4f}")
            ref = ro["a"] + ro["b"] * d["s3"]
            ax.scatter(d["s3"], ref, marker="D", s=70, facecolor="none", edgecolor=I9_COLOR, lw=1.6, zorder=8,
                       label=f"post hoc reference: stored-data S3 OLS line at the I9-1 S3, y {ref:+.4f} (P_c predicts "
                             "the sign only)")
            for f, mk in (("nmos", "o"), ("pmos", "s")):
                ax.scatter(pf[f]["S3"], d["yf"][f], marker=mk, s=60, facecolor="none", edgecolor=I9_COLOR,
                           lw=1.3, zorder=7)
                ax.annotate(f"I9-1 {FAM_SHORT[f]}" + (" (S3 sign wrong)" if f == "pmos" else ""),
                            (pf[f]["S3"], d["yf"][f]), xytext=(6, -3), textcoords="offset points", fontsize=7,
                            color=I9_COLOR)
            ax.scatter([], [], marker="o", s=50, facecolor="none", edgecolor=I9_COLOR, lw=1.3,
                       label="I9-1 per family (secondary): S3_fam vs observed y_fam")
            ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    fig.suptitle("I9-1: the NIST sky130 4 K regular-Vt DEV dataset (room-temperature card) and I8-1 added to the I7-1 "
                 "S1_lodo (left, frozen rule) and S3 (right) scatter.\nGrey: 11 stored datasets; blue / orange: 20 I7-1 "
                 "simulated datasets; violet star: I8-1; red star: I9-1", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(PRED), rel(DEC), rel(I8 / "decision.json"), rel(I8 / "prediction_v2.json"),
            rel(EXP / "I7_1/retro/results.json"), rel(EXP / "I7_1/decision.json")]


def fig_per_device(d: dict, path: Path) -> list[str]:
    lr, fam, info, sc = d["lr"], d["fam"], d["info"], d["sc"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ys = list(lr.values())
    xs_all = {"L_um": lambda t: info[t]["L_um"], "W_um": lambda t: info[t]["W_um"],
              "c0r": lambda t: sc["C0_refit"][t]}
    for ax, key, lab in ((axs[0], "L_um", "L (um)"), (axs[1], "W_um", "W (um)"),
                         (axs[2], "c0r", "C0_refit RRMS of the device")):
        style(ax)
        for t, v in lr.items():
            col = FAM_COLOR[fam[t]]
            ax.scatter(xs_all[key](t), v, s=40, marker="o" if fam[t] == "nmos" else "s", zorder=4, lw=1.0,
                       facecolor=col, edgecolor=INK if v > 0 else col, alpha=0.85)
        ax.axhline(0, color=INK, lw=0.7)
        ax.axhline(d["y"], color=MUTED, lw=1.0, ls="--")
        for f in ("nmos", "pmos"):
            ax.axhline(d["yf"][f], color=FAM_COLOR[f], lw=0.8, ls=":")
        ax.set_xscale("log")
        if key != "c0r":
            ticks = sorted({xs_all[key](t) for t in d["tags"]})
            keep = []
            for v in ticks:
                if not keep or v > 1.08 * keep[-1]:
                    keep.append(v)
            ax.set_xticks(keep)
            ax.set_xticklabels([f"{v:g}" for v in keep], fontsize=7, rotation=90)
            ax.minorticks_off()
        ax.set_xlabel(lab, fontsize=9)
        ax.set_ylabel("ln(RRMS_M_bias / RRMS_C0_refit) per device (negative = gain)", fontsize=9)
    worst = sorted(lr, key=lambda t: -lr[t])[:3]
    for t in worst:
        axs[2].annotate(t.replace("pfet_01v8_", "p ").replace("nfet_01v8_", "n "), (sc["C0_refit"][t], lr[t]),
                        xytext=(6, -2), textcoords="offset points", fontsize=7, color=MUTED)
    h = axs[0]
    for f in ("nmos", "pmos"):
        h.scatter([], [], marker="o" if f == "nmos" else "s", s=40, color=FAM_COLOR[f], label=FAM_LABEL[f])
        h.plot([], [], color=FAM_COLOR[f], ls=":", lw=0.8, label=f"{FAM_SHORT[f]} y = {d['yf'][f]:+.4f}")
    h.plot([], [], color=MUTED, ls="--", label=f"y_D = {d['y']:+.4f} (87 devices)")
    h.scatter([], [], s=40, facecolor="white", edgecolor=INK, label="black edge: device above 0 (M_bias higher)")
    h.legend(fontsize=7, frameon=True, framealpha=0.9, facecolor=BG, edgecolor="none", loc="upper left")
    n_up = sum(v > 0 for v in ys)
    axs[0].set_title(f"vs L; {len(ys) - n_up}/{len(ys)} devices below 0", fontsize=9.5, color=INK, loc="left")
    axs[1].set_title("vs W", fontsize=9.5, color=INK, loc="left")
    axs[2].set_title("vs the device's C0_refit RRMS (post hoc); 3 largest log ratios labeled", fontsize=9.5,
                     color=INK, loc="left")
    fig.suptitle("I9-1: per-device held-out log ratio of M_bias to C0_refit (87 DEV devices, NIST sky130 4 K, "
                 "room-temperature card; every device in a native bin)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO), rel(MANIFEST), rel(DEC)]


FOREST = (("M_bias|C0_refit", "M_bias vs C0_refit (P_b)"),
          ("M_bias|T1_inloop", "M_bias vs T1_inloop"),
          ("M_bias|B1", "M_bias vs B1"),
          ("M_bias|B2", "M_bias vs B2"),
          ("M_bias|B3_d3", "M_bias vs B3_d3 (S_B3)"),
          ("M_bias|M_bias_lvt", "M_bias vs M_bias_lvt"),
          ("C0_refit|C0", "C0_refit vs C0"),
          ("T1_inloop|C0_refit", "T1_inloop vs C0_refit"),
          ("B1|C0_refit", "B1 vs C0_refit"),
          ("B2|C0_refit", "B2 vs C0_refit"),
          ("B3_d3|C0_refit", "B3_d3 vs C0_refit"),
          ("M_bias_lvt|C0_refit", "M_bias_lvt vs C0_refit (S_transfer)"))


def get_cmp(d: dict, s: str, key: str) -> dict:
    if key == "C0_refit|C0":
        sel = [t for t in d["tags"] if s == "all" or d["fam"][t] == s]
        return paired_ratio({t: d["sc"]["C0_refit"][t] for t in sel}, {t: d["sc"]["C0"][t] for t in sel})
    return d["cmp"][s][key]


def fig_forest(d: dict, path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(12.5, 9.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    offs = {"all": 0.0, "nmos": -0.24, "pmos": 0.24}
    cols = {"all": INK, "nmos": BLUE, "pmos": ORANGE}
    n = len(FOREST)
    for i, (key, lab) in enumerate(FOREST):
        yy = n - 1 - i
        for s in ("all", "nmos", "pmos"):
            c = get_cmp(d, s, key)
            y0 = yy + offs[s]
            ax.plot([c["ci_low"], c["ci_high"]], [y0, y0], color=cols[s], lw=1.8 if s == "all" else 1.2)
            ax.scatter(c["ratio"], y0, color=cols[s], s=34 if s == "all" else 20, zorder=4,
                       marker="o" if s != "pmos" else "s")
            if s == "all":
                ax.annotate(f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] {c['wins']}/{c['n_used']}",
                            (max(c["ci_high"], 1.0), y0), xytext=(8, -3), textcoords="offset points", fontsize=7.5,
                            color=INK)
        if i == 5:
            ax.axhline(yy - 0.5, color=GREY, lw=0.6)
    ax.axvline(1, color=INK, lw=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for _, lab in FOREST][::-1], fontsize=8.5)
    ax.set_xlim(0.6, 1.5)
    ax.set_xlabel("paired GMR(A, B) = exp(mean ln(RRMS_A / RRMS_B)), device-bootstrap 95% CI; below 1: A lower",
                  fontsize=9)
    for s, lab in (("all", "all 87 DEV (value, CI, wins of A)"), ("nmos", "nMOS (48)"), ("pmos", "pMOS (39)")):
        ax.plot([], [], color=cols[s], marker="o" if s != "pmos" else "s", label=lab)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("I9-1: paired comparisons of every arm with M_bias (top block) and with C0_refit (bottom block)",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO), rel(BASE), rel(DEC)]


def fig_s1_s3(d: dict, path: Path) -> list[str]:
    rule, i8 = d["rule"], d["i8"]
    pf9, pf8 = d["pred"]["per_family_secondary"], i8["p2"]["per_family_secondary"]
    fs8 = i8["dec"]["secondary"]["per_family"]
    pts = []   # (label, S1, S3, y_obs, color, marker)
    pts.append(("I9-1 all", d["s1"], d["s3"], d["y"], I9_COLOR, "*"))
    for f, mk in (("nmos", "o"), ("pmos", "s")):
        pts.append((f"I9-1 {FAM_SHORT[f]}", pf9[f]["S1_lodo"], pf9[f]["S3"], d["yf"][f], I9_COLOR, mk))
    pts.append(("I8-1 all", i8["p1"]["dataset"]["S1_lodo"], i8["p2"]["dataset"]["S3"], i8["dec"]["P_a"]["y_D"],
                I8_COLOR, "*"))
    for f, mk in (("nmos", "o"), ("pmos", "s")):
        pts.append((f"I8-1 {FAM_SHORT[f]}", pf8[f]["S1_lodo"], pf8[f]["S3"], fs8[f]["y"], I8_COLOR, mk))
    loff = {"I8-1 all": (-48, 8), "I8-1 nMOS": (-60, -14), "I8-1 pMOS": (8, -12)}
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), dpi=DPI)
    fig.patch.set_facecolor(BG)
    # (a) S1_lodo vs y with the frozen rule
    ax = axs[0, 0]
    style(ax)
    xs = np.linspace(-0.05, 0.7, 50)
    ax.plot(xs, rule["a"] + rule["b"] * xs, color=INK, lw=1.3, label="frozen I7-1 rule")
    ax.axvline(rule["zero_crossing_S1_lodo"], color=INK, lw=0.8, ls=":", label="rule zero crossing")
    ax.axhline(0, color=INK, lw=0.6)
    for lab, s1, _, y, col, mk in pts:
        ax.scatter(s1, y, color=col, marker=mk, s=240 if mk == "*" else 70, edgecolor=INK, lw=0.6, zorder=5)
        ax.scatter(s1, rule_predict(rule, s1), facecolor="none", edgecolor=col, marker="D", s=45, lw=1.2, zorder=5)
        ax.plot([s1, s1], [rule_predict(rule, s1), y], color=col, lw=0.8)
        ax.annotate(lab, (s1, y), xytext=loff.get(lab, (7, -3)), textcoords="offset points", fontsize=7.5,
                    color=col)
    ax.set_xlabel("S1_lodo (dataset or family value from the prediction file)", fontsize=9)
    ax.set_ylabel("observed y = ln GMR(M_bias, C0_refit)", fontsize=9)
    n_ok1 = sum((rule_predict(rule, s1) < 0) == (y < 0) for _, s1, _, y, _, _ in pts)
    ax.set_title("(a) S1_lodo: filled = observed y, open diamond = frozen-rule prediction;\nrule sign correct on "
                 f"{n_ok1} of {len(pts)} points", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    # (b) S3 vs y, sign quadrants
    ax = axs[0, 1]
    style(ax)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=0.8, ls=":")
    ax.axvspan(-0.25, 0, ymin=0, ymax=1, color=BLUE, alpha=0.04, lw=0)
    for lab, _, s3, y, col, mk in pts:
        ok = (s3 < 0) == (y < 0)
        ax.scatter(s3, y, color=col, marker=mk, s=240 if mk == "*" else 70, edgecolor=INK, lw=0.6, zorder=5)
        if not ok:
            ax.scatter(s3, y, s=320, facecolor="none", edgecolor=RED, lw=1.3, zorder=6)
        ax.annotate(lab + ("" if ok else " (S3 sign wrong)"), (s3, y), xytext=loff.get(lab, (7, -3)),
                    textcoords="offset points",
                    fontsize=7.5, color=col)
    ax.set_xlim(-0.2, 0.08)
    ax.set_xlabel("S3 (dataset or family value; predicts gain iff S3 < 0, shaded)", fontsize=9)
    ax.set_ylabel("observed y", fontsize=9)
    n_ok3 = sum((s3 < 0) == (y < 0) for _, _, s3, y, _, _ in pts)
    cp = d["cmp"]["pmos"]["M_bias|C0_refit"]
    ax.set_title(f"(b) S3: sign correct on {n_ok3} of {len(pts)} points; I9-1 pMOS S3 {pf9['pmos']['S3']:+.3f} "
                 f"predicted no gain,\nobserved y {d['yf']['pmos']:+.3f} (GMR {cp['ratio']:.3f}, CI "
                 f"[{cp['ci_low']:.3f}, {cp['ci_high']:.3f}])", fontsize=9.5, color=INK, loc="left")
    # (c), (d) fold level, I9-1
    pf_, lr, fam = d["pred"]["per_fold"], d["lr"], d["fam"]
    fr = d["fold_rho"]
    for ax, key, lab in ((axs[1, 0], "S1_lodo_family", "S1_lodo_family of the fold"),
                         (axs[1, 1], "S3_family", "S3_family of the fold")):
        style(ax)
        for t in d["tags"]:
            ax.scatter(pf_[t][key], lr[t], color=FAM_COLOR[fam[t]], marker="o" if fam[t] == "nmos" else "s", s=26,
                       alpha=0.8, lw=0)
        ax.axhline(0, color=INK, lw=0.6)
        st = "S1" if key.startswith("S1") else "S3"
        ax.set_xlabel(lab, fontsize=9)
        ax.set_ylabel("y_f = ln(RRMS_M_bias / RRMS_C0_refit) of the held-out device", fontsize=9)
        ax.set_title(f"({'c' if st == 'S1' else 'd'}) I9-1 fold level (post hoc): Spearman within nMOS "
                     f"{fr[st + '_nmos']:+.3f}, within pMOS {fr[st + '_pmos']:+.3f}", fontsize=9.5, color=INK,
                     loc="left")
        for f in ("nmos", "pmos"):
            ax.scatter([], [], color=FAM_COLOR[f], marker="o" if f == "nmos" else "s", s=26, label=FAM_LABEL[f])
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("I9-1 and I8-1: S1_lodo (frozen rule) and S3 predictions against the observed log ratio, for the "
                 "dataset and per family (secondary), and the I9-1 fold-level values", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(PRED), rel(DEC), rel(LODO), rel(I8 / "prediction.json"), rel(I8 / "prediction_v2.json"),
            rel(I8 / "decision.json")]


def fig_transfer(d: dict, path: Path) -> list[str]:
    sc, fam = d["sc"], d["fam"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), dpi=DPI, gridspec_kw={"width_ratios": [1, 1.1]})
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    xs, ys = [], []
    for t in d["tags"]:
        x = math.log(sc["M_bias"][t] / sc["C0_refit"][t])
        y = math.log(sc["M_bias_lvt"][t] / sc["C0_refit"][t])
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, color=FAM_COLOR[fam[t]], marker="o" if fam[t] == "nmos" else "s", s=30, alpha=0.85, lw=0)
    lo, hi = min(xs + ys) - 0.05, max(xs + ys) + 0.05
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls="--", label="M_bias_lvt = M_bias")
    ax.axhline(0, color=INK, lw=0.5)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    for f in ("nmos", "pmos"):
        ax.scatter([], [], color=FAM_COLOR[f], marker="o" if f == "nmos" else "s", s=30, label=FAM_LABEL[f])
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlabel("ln(RRMS_M_bias / RRMS_C0_refit)  (M_bias trained on the 86 other I9-1 DEV devices)", fontsize=9)
    ax.set_ylabel("ln(RRMS_M_bias_lvt / RRMS_C0_refit)  (trained once on 53 I8-1 lvt DEV records)", fontsize=9)
    ax.set_title("(a) per device; below the dashed line: M_bias_lvt lower than M_bias", fontsize=9.5, color=INK,
                 loc="left")
    ax = axs[1]
    style(ax)
    rows = (("M_bias_lvt|C0_refit", "M_bias_lvt vs C0_refit"), ("M_bias|C0_refit", "M_bias vs C0_refit"),
            ("M_bias_lvt|M_bias", "M_bias_lvt vs M_bias"))
    cols = {"all": INK, "nmos": BLUE, "pmos": ORANGE}
    offs = {"all": 0.0, "nmos": -0.22, "pmos": 0.22}
    for i, (key, lab) in enumerate(rows):
        yy = len(rows) - 1 - i
        for s in ("all", "nmos", "pmos"):
            c = d["cmp"][s][key]
            y0 = yy + offs[s]
            ax.plot([c["ci_low"], c["ci_high"]], [y0, y0], color=cols[s], lw=1.6)
            ax.scatter(c["ratio"], y0, color=cols[s], s=26, zorder=4, marker="o" if s != "pmos" else "s")
            ax.annotate(f"{FAM_SHORT[s]} {c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] "
                        f"{c['wins']}/{c['n_used']}", (c["ci_high"], y0), xytext=(6, -3),
                        textcoords="offset points", fontsize=7.2, color=cols[s])
    ax.axvline(1, color=INK, lw=0.7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows][::-1], fontsize=8.5)
    ax.set_xlim(0.6, 1.6)
    ax.set_xlabel("paired GMR with device-bootstrap 95% CI; wins of the first arm", fontsize=9)
    ax.set_title("(b) S_transfer: all 87 (black), nMOS 48 (blue), pMOS 39 (orange)", fontsize=9.5, color=INK,
                 loc="left")
    fig.suptitle("I9-1 transfer arm M_bias_lvt: the M_bias recipe trained once on the I8-1 NIST 4 K lvt DEV records "
                 "(77 K card) and applied to the I9-1 regular-Vt devices (room-temperature card)", fontsize=10.5,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO), rel(DEC)]


def g_points(d: dict) -> dict:
    from cryoml.explore.e1_correction import load_devices, make_batch
    recs = load_devices(I9 / "start/records")
    assert len(recs) == N_DEV
    phi = make_batch(recs).phi
    return {"nmos": phi[phi[:, 0] == 0.0], "pmos": phi[phi[:, 0] == 1.0]}


def fig_g(d: dict, path: Path) -> list[str]:
    mb, b3, lvt = p81.g_maps(LODO_CORR, "M_bias"), p81.g_maps(BASE_CORR, "B3_d3"), p81.g_maps(LODO_CORR, "M_bias_lvt")
    assert mb["nmos"].shape[0] == N_DEV and b3["nmos"].shape[0] == N_DEV and lvt["nmos"].shape[0] == 1
    rows = [("I9-1 M_bias", mb, f"{N_DEV} LODO folds"), ("I9-1 B3_d3", b3, f"{N_DEV} LODO folds"),
            ("M_bias_lvt (I8-1 lvt DEV training)", lvt, "1 correction")]
    lim = 0.5
    fig = plt.figure(figsize=(13, 17), dpi=DPI)
    fig.patch.set_facecolor(BG)
    gsp = fig.add_gridspec(3, 2, left=0.07, right=0.86, top=0.93, bottom=0.04, hspace=0.36, wspace=0.22)
    a = np.linspace(0, 1, 61)
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
            ax.scatter(pts[pol][:, 1], pts[pol][:, 2], s=1.5, color=INK, alpha=0.2, lw=0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("|Vgs| / VDD", fontsize=8.5)
            ax.set_ylabel("|Vds| / VDD", fontsize=8.5)
            ax.set_title(f"{lab}, {FAM_SHORT[pol]}: mean g over {nf}\nexp(mean g) {np.exp(g.min()):.2f}-"
                         f"{np.exp(g.max()):.2f}; max fold spread of g {spread.max():.3f}", fontsize=8.5, color=INK,
                         loc="left")
    st = {}
    for pol in ("nmos", "pmos"):
        phi = pts[pol]
        gm_ = p81.g_at(LODO_CORR, "M_bias", phi)
        gb_ = p81.g_at(BASE_CORR, "B3_d3", phi)
        gl_ = p81.g_at(LODO_CORR, "M_bias_lvt", phi)
        st[f"n_pts_{pol}"] = int(len(phi))
        st[f"rms_mb_{pol}"] = float(np.sqrt(np.mean(gm_ ** 2)))
        st[f"rms_b3_{pol}"] = float(np.sqrt(np.mean(gb_ ** 2)))
        st[f"rms_lvt_{pol}"] = float(np.sqrt(np.mean(gl_ ** 2)))
        st[f"rms_mb_b3_{pol}"] = float(np.sqrt(np.mean((gm_ - gb_) ** 2)))
        st[f"rms_mb_lvt_{pol}"] = float(np.sqrt(np.mean((gm_ - gl_) ** 2)))
        st[f"r_mb_b3_{pol}"] = float(np.corrcoef(gm_, gb_)[0, 1])
        st[f"r_mb_lvt_{pol}"] = float(np.corrcoef(gm_, gl_)[0, 1])
    cax = fig.add_axes([0.89, 0.35, 0.018, 0.3])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("g (I = I_card x exp(g)); contours: exp(g)", fontsize=9)
    fig.text(0.07, 0.975, "I9-1 learned correction g over the normalized bias plane per polarity (mean of the stored "
             "corrections; no training).", fontsize=11, color=INK)
    fig.text(0.07, 0.957, f"VDD 1.8 V. Grey dots: trimmed included points of the {N_DEV} I9-1 DEV starting records; "
             "outside them g is extrapolated. Row 3 was trained on I8-1 lvt records.", fontsize=9.5, color=MUTED)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    d["g_stats"] = st
    return [rel(LODO_CORR), rel(BASE_CORR), rel(I9 / "start/records.json")]


def fig_iv(tag: str, ex: dict, d: dict, path: Path, why: str) -> list[str]:
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
                color = p81.IV_COLORS[k % len(p81.IV_COLORS)]
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
    stock = ex["stock_scores"][tag]["score_device"]
    fig.suptitle(f"I9-1 {tag} ({FAM_SHORT[dv['polarity']]}, W {dv['W_um']:g} um, L {dv['L_um']:g} um, native bin "
                 f"{dv['bin']}); ln(M_bias / C0_refit) {d['lr'][tag]:+.3f}; {why}\nngspice-41 arm64, open sky130 tt "
                 f"room-temperature card, deck 30 C, mult = m = 1; stock card RRMS {stock:.3f} (not drawn); stored "
                 f"cards re-simulated (RRMS reproduced, relative difference "
                 f"{max(v['rel_diff'] for v in ex['checks'][tag].values()):.0e})", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(EXAMPLES), rel(LODO), rel(I9 / "lodo/cards.json"), rel(LODO_CORR)]


# ================================================================== main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)

    d = load_all()
    ex = load(EXAMPLES)
    assert ex["max_rel_diff"] == 0.0 and len(ex["devices"]) == 3
    for tag, ch in ex["checks"].items():
        for a, v in ch.items():
            assert close(v["stored"], d["sc"][a][tag]), (tag, a)
        assert close(ex["stock_scores"][tag]["score_device"], d["stock"][tag])
    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(t_decision(d), "Table of the pre-registered P_a (with the Q6 calibration-range note), P_b, P_c, "
              "S_B3, S_transfer, validity V1-V4 and secondary outcomes (magnitude, per family, gates, S1_pooled / "
              "S2, fold-level Spearman), with the commit hashes of the pre-registration and the prediction",
              [rel(PREREG), rel(PRED), rel(DEC), rel(LODO)])
    src = fig_scatter(d, folder / "02_i9_1_scatter_S1_lodo_S3.png")
    items.append(("02_i9_1_scatter_S1_lodo_S3.png", "Two scatter panels from I7-1 (S1_lodo with the frozen rule and "
                  "the I7-1 calibration range shaded, left; S3 with the stored-data OLS line, right) with I8-1 (violet "
                  "star) and I9-1 (red star = observed y_D; open diamond = frozen-rule prediction on the left, post "
                  "hoc OLS reference on the right; open circle / square = I9-1 nMOS / pMOS family values)", src))
    src = fig_per_device(d, folder / "03_i9_1_per_device_log_ratio.png")
    items.append(("03_i9_1_per_device_log_ratio.png", "Per-device ln(RRMS_M_bias / RRMS_C0_refit) vs L, W and the "
                  "device's C0_refit RRMS (nMOS circles, pMOS squares; dotted lines = family means, dashed = y_D)",
                  src))
    add_table(t_arms(d), "Table of the arms (stock card, C0, C0_refit, T1_inloop, M_bias, M_bias_lvt, B1, B2, "
              "B3_d3): finite / failed devices, mean, median, combined, family means, GMR vs C0_refit and "
              "GMR(M_bias, arm) with CI", [rel(LODO), rel(BASE), rel(START)])
    src = fig_forest(d, folder / "05_i9_1_gmr_forest.png")
    items.append(("05_i9_1_gmr_forest.png", "Forest plot of paired GMR with device-bootstrap 95% CI: M_bias against "
                  "C0_refit, T1_inloop, B1, B2, B3_d3 and M_bias_lvt, and each arm against C0_refit; all 87 DEV "
                  "devices, nMOS and pMOS", src))
    add_table(t_family(d), "Table per family (nMOS, pMOS, all): devices, included curves, C0_refit and M_bias mean "
              "RRMS, S1_lodo, rule-predicted and observed y, rule and S3 sign outcome, GMR(M_bias, C0_refit), "
              "GMR(B3_d3, C0_refit), GMR(M_bias, B3_d3), GMR(M_bias_lvt, C0_refit)",
              [rel(PRED), rel(DEC), rel(LODO), rel(BASE)])
    src = fig_s1_s3(d, folder / "07_i9_1_S1_S3_per_family.png")
    items.append(("07_i9_1_S1_S3_per_family.png", "S1_lodo with the frozen rule (a) and S3 (b) against the observed "
                  "y for I9-1 and I8-1, dataset and per family; I9-1 fold-level S1_lodo_family and S3_family against "
                  "the per-device log ratio (c, d)", src))
    src = fig_transfer(d, folder / "08_i9_1_transfer_M_bias_lvt.png")
    items.append(("08_i9_1_transfer_M_bias_lvt.png", "Transfer arm M_bias_lvt: per-device log ratio vs M_bias (a) "
                  "and GMR with CI vs C0_refit and vs M_bias per family (b)", src))
    src = fig_g(d, folder / "09_i9_1_g_map.png")
    gs = d["g_stats"]
    items.append(("09_i9_1_g_map.png", "Mean learned g on the (|Vgs|/VDD, |Vds|/VDD) plane per polarity: I9-1 M_bias "
                  "(87 folds), I9-1 B3_d3 (87 folds), M_bias_lvt (one correction trained on I8-1 lvt records); "
                  "contours of exp(g); I9-1 training points as grey dots", src))
    add_table(t_compare(d), "Table comparing I8-1 (NIST 4 K lvt families, 77 K card) and I9-1 (NIST 4 K regular-Vt "
              "families, room-temperature card): data, card, RRMS, P_a, P_b, P_c, per family, comparators, "
              "baselines, transfer, compute", [rel(DEC), rel(I8 / "decision.json"), rel(I8 / "lodo/results.json"),
                                               rel(I8 / "baselines/results.json")])
    whys = {}
    lrs = d["lr"]
    for tag in ex["devices"]:
        f = d["fam"][tag]
        sel = {t: v for t, v in lrs.items() if d["fam"][t] == f}
        if tag == max(lrs, key=lambda t: lrs[t]):
            whys[tag] = "largest log ratio of the 87 devices"
        else:
            whys[tag] = f"nearest the {FAM_SHORT[f]} median log ratio ({float(np.median(list(sel.values()))):+.3f})"
    for k, tag in enumerate(ex["devices"]):
        name = f"{11 + k:02d}_i9_1_iv_{tag}.png"
        src = fig_iv(tag, ex, d, folder / name, whys[tag])
        dv = ex["devices"][tag]
        items.append((name, f"Example I-V, {tag} ({FAM_SHORT[dv['polarity']]}; {whys[tag]}): measured points, "
                      "NGSpice C0 (dotted), C0_refit (solid) and M_bias (refit card NGSpice output x exp(g) applied "
                      "in Python, dashed); Id-Vg log / linear, Id-Vd linear / log", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i9_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. y = ln GMR(M_bias, C0_refit) over the 87 DEV devices "
          "(strict paper RRMS after the held-out card refit); negative = gain.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    ca, agg, fr = d["cmp"]["all"], d["agg"], d["fold_rho"]
    g = ca["M_bias|C0_refit"]
    nm, pm = d["cmp"]["nmos"]["M_bias|C0_refit"], d["cmp"]["pmos"]["M_bias|C0_refit"]
    b3 = ca["B3_d3|C0_refit"]
    lvt_c, lvt_m = ca["M_bias_lvt|C0_refit"], ca["M_bias_lvt|M_bias"]
    worst = sorted(lrs, key=lambda t: -lrs[t])
    n_up = [t for t in worst if lrs[t] > 0]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice fit, no training). The example I-V "
              "curves (figures 11-13) are NGSpice re-simulations of stored cards by "
              f"`scripts/research/explore/i9_1_examples.py` ({rel(EXAMPLES)}, {ex['runtime_s']} s; RRMS reproduced "
              "with relative difference 0).", "",
              "## Contents", "",
              "- I9-1: second prospective DEV test on the NIST mds2-2997 sky130 transistors measured at 4 K (MPW-5 "
              "tile; nfet_01v8 48, pfet_01v8 39 DEV devices; 6 pMOS with < 2 included curves excluded before the "
              "split) with the open sky130 room-temperature tt cards (open_pdks a918dc7; native bin for every "
              "device; deck at 30 C, the cards' tnom; mult = m = 1; Vb = 0 curves; leave-one-device-out). "
              f"Pre-registration `{rel(PREREG)}` (in force from commit {COMMITS['prereg']}); prediction "
              f"`{rel(PRED)}` (commit {COMMITS['prediction']}, before any held-out refit); decision `{rel(DEC)}`. "
              "Queue jobs: " + ", ".join(f"{k} ({v} s)" for k, v in JOBS.items()) + ".",
              f"- Validity: V1 {d['n_ok_start']}/{N_DEV} finite starts, V2 failed C0_refit / M_bias 0 / 0, paired "
              f"{g['n_used']}, V3 prospective (guards recorded in lodo and baselines results), V4 0 timeouts.",
              f"- Stock card mean RRMS {d['start']['stock']['valid_only_mean']:.3f}; starting cards (C0) "
              f"{agg['C0']['valid_only_mean']:.4f}.",
              f"- P_a (frozen I7-1 rule, prediction.json): S1_lodo_D {d['s1']:.5f}, inside the I7-1 calibration range "
              f"[{CAL_RANGE[0]:.3f}, {CAL_RANGE[1]:.3f}]; predicted y {d['pa_y']:+.4f} (gain); observed y_D "
              f"{d['y']:+.4f}: correct. |error| {abs(d['y'] - d['pa_y']):.4f} (frozen rule's in-sample residual RMS "
              f"{d['rms_rule']:.4f}).",
              f"- P_b: GMR(M_bias, C0_refit) {ci_text(g)} wins; R2 M_bias {agg['M_bias']['valid_only_mean']:.5f} vs "
              f"T1_inloop {agg['T1_inloop']['valid_only_mean']:.5f} (+0.005); R3 0 M_bias failures: {d['pb']}.",
              f"- P_c: S3_D {d['s3']:+.4f}, predicted gain: correct.",
              f"- Per family (secondary): nMOS GMR {ci_text(nm)}, y {d['yf']['nmos']:+.4f}, rule and S3 signs correct; "
              f"pMOS GMR {ci_text(pm)}, y {d['yf']['pmos']:+.4f}, rule sign correct (predicted y "
              f"{d['pred']['per_family_secondary']['pmos']['predicted_y_rule']:+.4f}), S3 sign wrong (S3_pmos "
              f"{d['pred']['per_family_secondary']['pmos']['S3']:+.4f} predicted no gain). The pMOS GMR CI contains 1.",
              f"- Comparators: M_bias vs T1_inloop {ci_text(ca['M_bias|T1_inloop'])}; T1_inloop vs C0_refit "
              f"{ci_text(ca['T1_inloop|C0_refit'])}; C0_refit vs C0 {ci_text(get_cmp(d, 'all', 'C0_refit|C0'))}.",
              f"- S_B3 and baselines (reported, not deciding): M_bias vs B3_d3 {ci_text(ca['M_bias|B3_d3'])} (M_bias "
              f"lower); GMR(B3_d3, C0_refit) {ci_text(b3)}: the cubic polynomial correction gives {d['share']:.0%} of "
              f"M_bias's log gain (nMOS {d['share_f']['nmos']:.0%}, pMOS {d['share_f']['pmos']:.0%}). M_bias vs B1 "
              f"{ci_text(ca['M_bias|B1'])}, vs B2 {ci_text(ca['M_bias|B2'])}; B1 vs C0_refit "
              f"{ci_text(ca['B1|C0_refit'])}; B2 vs C0_refit {ci_text(ca['B2|C0_refit'])}.",
              f"- S_transfer: M_bias_lvt (trained once on the 53 finite I8-1 lvt DEV records, 77 K card) vs C0_refit "
              f"{ci_text(lvt_c)}; vs M_bias {ci_text(lvt_m)}. nMOS vs M_bias "
              f"{ci_short(d['cmp']['nmos']['M_bias_lvt|M_bias'])}, pMOS vs M_bias "
              f"{ci_short(d['cmp']['pmos']['M_bias_lvt|M_bias'])}.",
              f"- Devices with M_bias above C0_refit: {len(n_up)} of {N_DEV}; the 6 largest log ratios are pMOS with "
              "L >= 4 um with C0_refit RRMS " + f"{min(d['sc']['C0_refit'][t] for t in worst[:6]):.4f}-"
              f"{max(d['sc']['C0_refit'][t] for t in worst[:6]):.4f} (all-device median "
              f"{agg['C0_refit']['median']:.4f}): " + ", ".join(f"{t} {lrs[t]:+.2f}" for t in worst[:6]) +
              " (figure 03, post hoc).",
              f"- Learned g (figure 09, stored corrections, at the trimmed included points of the {N_DEV} DEV records: "
              f"{gs['n_pts_nmos']} nMOS, {gs['n_pts_pmos']} pMOS points): RMS of mean g, M_bias {gs['rms_mb_nmos']:.3f} "
              f"/ {gs['rms_mb_pmos']:.3f}, B3_d3 {gs['rms_b3_nmos']:.3f} / {gs['rms_b3_pmos']:.3f}, M_bias_lvt "
              f"{gs['rms_lvt_nmos']:.3f} / {gs['rms_lvt_pmos']:.3f}; Pearson r M_bias vs B3_d3 "
              f"{gs['r_mb_b3_nmos']:.3f} / {gs['r_mb_b3_pmos']:.3f}, M_bias vs M_bias_lvt {gs['r_mb_lvt_nmos']:.3f} / "
              f"{gs['r_mb_lvt_pmos']:.3f} (nMOS / pMOS).",
              f"- Fold level (computed here): Spearman(S1_lodo_f, y_f) {fr['S1']:+.3f}, Spearman(S3_f, y_f) "
              f"{fr['S3']:+.3f} over the 87 folds; within family S1_lodo_family nMOS {fr['S1_nmos']:+.3f}, pMOS "
              f"{fr['S1_pmos']:+.3f}; S3_family nMOS {fr['S3_nmos']:+.3f}, pMOS {fr['S3_pmos']:+.3f}.",
              "- Post hoc (not pre-registered): the per-family B3_d3 and M_bias_lvt columns, the per-device panels vs "
              "C0_refit RRMS, the S3 OLS reference point, the fold-level panels, the g maps and the example I-V figures.",
              "- Limitations: the deck runs at 30 C for 4 K data (kT/q 26 mV; the stock card's smallest subthreshold "
              "swing is 81-135 mV/decade on the smoke devices, measured 6-46); one dataset gives one binary P_a outcome.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    summary = [
        f"I9-1 NIST sky130 4 K regular-Vt DEV, room-temperature card (prospective, 87 devices, 0 failures): P_a "
        f"correct (S1_lodo {d['s1']:.3f} inside the calibration range; predicted y {d['pa_y']:+.3f}, observed "
        f"{d['y']:+.3f}); P_b {d['pb']}, GMR(M_bias, C0_refit) {ci_text(g)}; P_c correct (S3 {d['s3']:+.3f}).",
        f"nMOS {nm['ratio']:.3f} (rule and S3 correct), pMOS {pm['ratio']:.3f} [{pm['ci_low']:.3f}, "
        f"{pm['ci_high']:.3f}] (rule correct, S3 wrong). M_bias vs B3_d3 {ci_short(ca['M_bias|B3_d3'])} "
        f"({d['share']:.0%} of the log gain from B3_d3).",
        f"Transfer M_bias_lvt vs C0_refit {lvt_c['ratio']:.3f}, vs M_bias {lvt_m['ratio']:.3f}. M_bias vs B1 "
        f"{ca['M_bias|B1']['ratio']:.3f}, B2 {ca['M_bias|B2']['ratio']:.3f}, T1_inloop {ca['M_bias|T1_inloop']['ratio']:.3f}.",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    print("g stats", {k: round(v, 4) for k, v in gs.items()})
    print("fold rho", {k: round(v, 3) for k, v in fr.items()}, "rms_rule", d["rms_rule"], "share", d["share"],
          d["share_f"])
    print("devices above 0:", len(n_up), [(t, round(lrs[t], 3)) for t in n_up])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
