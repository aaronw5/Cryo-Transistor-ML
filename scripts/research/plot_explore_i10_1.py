#!/usr/bin/env python3
"""Report batch for exploration direction I10-1: zero-shot prediction of a
device's 4 K I-V from its room-temperature (RT) data (NIST ETest
nfet_g5v0d10v5 / pfet_g5v0d10v5 DEV, 67 devices; TU Delft 40 nm I6-1 DEV, 45
devices, replication).  Decision: "invalid" (V2 failed on NIST); every K label
on NIST is descriptive.

Reads stored files only; no NGSpice fit, no training:

* ``out/research/explore/I10_1/{prereg.json, decision.json, data/manifest.json}``
* ``out/research/explore/I10_1/lodo_{nist,tud}/results.json``,
  ``start_{nist,tud}/summary.json``
* ``out/research/explore/I10_1/diagnosis/diagnosis.json`` (post hoc failure
  diagnosis, ``scripts/research/explore/i10_1_diagnose.py``)
* ``out/research/explore/I10_1/examples/examples.json`` (NGSpice re-simulation of
  stored cards for 2 example devices, ``scripts/research/explore/i10_1_examples.py``)

Checks: the per-arm aggregates, the validity counts (V1, V2 failed per arm,
paired per contrast), the K1-K5 GMR with Bonferroni / 95% / cluster CIs (all,
nMOS, pMOS), the labels, the outcome and the gap fractions are recomputed from
the per-device scores and compared with results.json and decision.json, for
NIST and TU Delft.  The diagnosis counts are checked against the failed sets.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i10_1.py --date 2026-10-21
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
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, RED, AQUA, VIOLET, YELLOW,  # noqa: E402
                                   DPI, EXP, load, rel, style)
import plot_explore_i8_1 as p81  # noqa: E402  (I-V colors, helpers)
from cryoml.explore import i10_1_rt4k as m  # noqa: E402
from cryoml.explore import i8_1_nist as m8  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i10_1"
I10 = EXP / "I10_1"
PREREG = I10 / "prereg.json"
DEC = I10 / "decision.json"
MANIFEST = I10 / "data/manifest.json"
DIAG = I10 / "diagnosis/diagnosis.json"
EXAMPLES = I10 / "examples/examples.json"
LODO = {"nist": I10 / "lodo_nist/results.json", "tud": I10 / "lodo_tud/results.json"}
START = {"nist": I10 / "start_nist/summary.json", "tud": I10 / "start_tud/summary.json"}
COMMITS = {"feasibility": "a108203", "prereg": "17c96a1", "code_smoke": "eeaed11"}
JOBS = {"I10_1_start_nist": 2335, "I10_1_start_tud": 751, "I10_1_lodo_nist": 2100, "I10_1_lodo_tud": 488,
        "I10_1_decide": 0}
CONTRASTS = (("K1", "A1", "A0"), ("K2", "A1", "A0_rtdeck"), ("K3", "A1", "R_pop"), ("K4", "A2", "A1"),
             ("K5", "A3", "A1"))
PRIMARY_ARMS = ("A0", "A0_rtdeck", "A1", "A2", "A3", "R_pop")
BONF = 1 - 0.05 / 5
N_NIST, N_TUD = 67, 45
ARMS = {"nist": ("A0", "A0_rtdeck", "A1", "A2", "A3", "R_pop", "R_oracle", "A1_L", "R_stock4k", "R_stock_rtdeck",
                 "V_ind", "V_mean", "V_sub7", "V_deck"),
        "tud": ("A0", "A0_rtdeck", "A1", "A2", "A3", "R_pop", "R_oracle", "A1_L", "R_stock4k", "R_stock_rtdeck",
                "V_mean", "V_sub7", "V_i6")}
ARM_LABEL = {"A0": "A0: C0_RT at the 4 K deck", "A0_rtdeck": "A0_rtdeck: C0_RT at the RT deck",
             "A1": "A1: C0_RT + median shift", "A2": "A2: A1 x exp(g) (M_bias)",
             "A3": "A3: A1 + ridge term", "R_pop": "R_pop: median training C4",
             "R_oracle": "R_oracle: own C4 (uses 4 K data)", "A1_L": "A1_L: A1 + geometry ridge",
             "R_stock4k": "R_stock4k: stock card, 4 K deck", "R_stock_rtdeck": "R_stock_rtdeck: stock card, RT deck",
             "V_ind": "V_ind: C4 from stock", "V_mean": "V_mean: mean shift", "V_sub7": "V_sub7: 7-parameter shift",
             "V_deck": "V_deck: C4 and A1 at -269.15 C", "V_i6": "V_i6: stored I6-1 4 K cards"}
K_LABEL = {"K1": "K1 GMR(A1, A0) (primary)", "K2": "K2 GMR(A1, A0_rtdeck)", "K3": "K3 GMR(A1, R_pop)",
           "K4": "K4 GMR(A2, A1)", "K5": "K5 GMR(A3, A1)"}
ARM_COLOR = {"A0": GREY, "A0_rtdeck": MUTED, "A1": BLUE, "A2": AQUA, "A3": VIOLET, "R_pop": YELLOW,
             "R_oracle": RED}
FAM_SHORT = {"nmos": "nMOS", "pmos": "pMOS", "all": "all"}
DS_LABEL = {"nist": "NIST ETest g5v0d10v5 (67 DEV, decides)", "tud": "TU Delft 40 nm (45 DEV, replication)"}

close = p81.close
nanify = p81.nanify


def ci_text(c: dict, wins: bool = True) -> str:
    s = f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
    return s + (f", {c['wins']}/{c['n_used']}" if wins else "")


def same(a: dict, b: dict) -> bool:
    return all(close(a[k], b[k]) for k in ("ratio", "ci_low", "ci_high")) and \
        all(a[k] == b[k] for k in ("n", "n_used", "wins", "n_method_failed", "n_control_failed"))


def short(tag: str) -> str:
    return tag.replace("etest_", "").replace("_g5v0d10v5", "")


# ================================================================== loading and checks

def load_ds(name: str, dec: dict) -> dict:
    res, start = load(LODO[name]), load(START[name])
    assert not res["smoke"] and not start["smoke"]
    tags = res["devices"]
    assert len(tags) == (N_NIST if name == "nist" else N_TUD)
    sc = {a: nanify(v) for a, v in res["per_device"].items()}
    pol = {t: ("pmos" if ("pfet" in t or "_pmos_" in t) else "nmos") for t in tags}
    agg = {a: m8.aggregate(sc[a], pol) for a in sc}
    for a in sc:
        for k in ("n", "n_failed", "valid_only_mean", "median", "combined"):
            assert close(agg[a][k], res["summary"][a][k]), (name, a, k)
    clusters = {t: g for g in res["groups"] for t in tags if res["folds"][g]["held"].count(t)}
    assert sorted(clusters) == sorted(tags)
    cmp = {}
    for s in ("all", "nmos", "pmos"):
        sel = [t for t in tags if s == "all" or pol[t] == s]
        cmp[s] = {}
        for k, a, c in CONTRASTS:
            ma, mc = {t: sc[a][t] for t in sel}, {t: sc[c][t] for t in sel}
            cmp[s][k] = {"bonferroni": m.gmr_ci(ma, mc, level=BONF), "level95": m.gmr_ci(ma, mc, level=0.95),
                         "cluster_bonferroni": m.gmr_ci(ma, mc, clusters=clusters, level=BONF)}
            for lv in cmp[s][k]:
                assert same(cmp[s][k][lv], res["comparisons"][s][k][lv]), (name, s, k, lv)
    dd = dec["datasets"][name]
    labels = {k: m.ci_label(cmp["all"][k]["bonferroni"]) for k, _, _ in CONTRASTS}
    assert labels == dd["labels"], (name, labels)
    for k, _, _ in CONTRASTS:
        for lv in ("bonferroni", "level95", "cluster_bonferroni"):
            assert same(cmp["all"][k][lv], dd["contrasts"][k][lv])
        for f in ("nmos", "pmos"):
            assert same(cmp[f][k]["bonferroni"], dd["per_family"][f][k])
    for a, v in dd["means_valid_only"].items():
        assert close(agg[a]["valid_only_mean"], v), (name, a)
    # validity
    rows4 = start["fits"]["C4"]["rows"]
    ok_start = [t for t in tags if t not in res["excluded_A1_rule"] and (rows4[t]["n_curves"] or 0) >= 2]
    failed = {a: sum(1 for t in ok_start if not math.isfinite(sc[a][t])) for a in PRIMARY_ARMS}
    paired = {k: cmp["all"][k]["bonferroni"]["n_used"] for k, _, _ in CONTRASTS}
    n = len(tags)
    v1 = len(ok_start) >= 0.9 * n
    v2 = all(failed[a] <= 0.1 * n for a in PRIMARY_ARMS) and all(p >= math.ceil(0.8 * n) for p in paired.values())
    val = dd["validity"]
    assert (v1, v2, len(ok_start)) == (val["V1"], val["V2"], val["n_finite_start"]) and failed == val["failed"] \
        and paired == val["paired"], name
    assert dd["valid"] == (v1 and v2)
    # gap fraction toward the oracle: ln GMR(arm, A0) / ln GMR(R_oracle, A0) (decision.json definition)
    sec = res["comparisons"]["all"]
    ln_or = math.log(cmp["all"]["K1"]["bonferroni"]["ratio"]) - math.log(sec["S_A1_vs_R_oracle"]["bonferroni"]["ratio"])
    gap = {"A1": math.log(cmp["all"]["K1"]["bonferroni"]["ratio"]) / ln_or,
           "A2": math.log(sec["S_A2_vs_A0"]["bonferroni"]["ratio"]) / ln_or,
           "A3": math.log(sec["S_A3_vs_A0"]["bonferroni"]["ratio"]) / ln_or}
    for a, v in gap.items():
        assert close(v, dd["gap_fraction_toward_oracle"][f"gap_fraction_{a}"]), (name, a)
    fam_mean = {a: {f: (float(np.mean([x for t, x in sc[a].items() if pol[t] == f and math.isfinite(x)]))
                        if any(pol[t] == f and math.isfinite(x) for t, x in sc[a].items()) else float("nan"))
                    for f in ("nmos", "pmos")} for a in sc}
    fam_fail = {a: {f: sum(1 for t, x in sc[a].items() if pol[t] == f and not math.isfinite(x))
                    for f in ("nmos", "pmos")} for a in sc}
    return {"res": res, "start": start, "tags": tags, "sc": sc, "pol": pol, "agg": agg, "cmp": cmp, "labels": labels,
            "ok_start": ok_start, "failed": failed, "paired": paired, "v": (v1, v2), "gap": gap, "clusters": clusters,
            "fam_mean": fam_mean, "fam_fail": fam_fail, "sec": sec, "dd": dd}


def load_all() -> dict:
    prereg, dec = load(PREREG), load(DEC)
    assert prereg["status"].upper().startswith("IN FORCE") and not dec["smoke"]
    assert dec["outcome"] == "invalid" and dec["guard"]["ok"] and dec["guard"]["commit"].startswith(COMMITS["prereg"])
    d = {"prereg": prereg, "dec": dec, "nist": load_ds("nist", dec), "tud": load_ds("tud", dec)}
    n, t = d["nist"], d["tud"]
    assert not all(n["v"]) and all(t["v"])
    for k, _, _ in CONTRASTS:
        agree = bool(np.sign(math.log(t["cmp"]["all"][k]["bonferroni"]["ratio"]))
                     == np.sign(math.log(n["cmp"]["all"][k]["bonferroni"]["ratio"])))
        assert agree == dec["tud_direction_agrees"][k]
    # diagnosis
    dg = load(DIAG)
    sc = n["sc"]
    excl = n["res"]["excluded_A1_rule"]
    assert dg["excluded_A1_rule"] == excl
    for arm in ("A1", "A3", "R_pop"):
        fails = sorted(x for x in n["tags"] if x not in excl and not math.isfinite(sc[arm][x]))
        assert sorted(r["tag"] for r in dg["op_runs"] if r["arm"] == arm) == fails, arm
        ag = dg["agreement"][arm]
        assert ag["failed_not_predicted"] == [] and ag["failed_and_predicted"] == len(fails)
    a2 = sorted(x for x in n["tags"] if x not in excl and not math.isfinite(sc["A2"][x]))
    a1 = sorted(x for x in n["tags"] if x not in excl and not math.isfinite(sc["A1"][x]))
    assert a1 == a2
    assert all(len(r["fatal"]) == 1 for r in dg["op_runs"])
    for x in dg["agreement"]["R_pop"]["predicted_not_failed"]:
        e = dg["analytic"]["R_pop"][x]
        assert abs(e["lpeb_eff"] + e["Leff_m"]) <= 1e-9 * e["Leff_m"]
    d["diag"] = dg
    d["manifest"] = load(MANIFEST)
    d["rows"] = {r["tag"]: r for r in d["manifest"]["devices"]}
    return d


# ================================================================== tables

def t_decision(d: dict) -> rf.Table:
    dec, n, t = d["dec"], d["nist"], d["tud"]
    lim_f, lim_p = 0.1 * N_NIST, math.ceil(0.8 * N_NIST)
    fail_txt = ", ".join(f"{a} {n['failed'][a]}" for a in PRIMARY_ARMS)
    pair_txt = ", ".join(f"{k} {n['paired'][k]}" for k, _, _ in CONTRASTS)
    bad_arms = [a for a in PRIMARY_ARMS if n["failed"][a] > lim_f]
    bad_k = [k for k, _, _ in CONTRASTS if n["paired"][k] < lim_p]
    rows = [
        {"item": "OUTCOME", "cond": "invalid iff V1 or V2 fails on NIST DEV; action 'stop: fix and re-register'",
         "val": "V2 fails on NIST", "res": "INVALID (stopped by rule)"},
        {"item": "validity V1 (NIST)", "cond": ">= 90% of 67 with finite C0_RT and C4 records and >= 2 included 4 K "
         "curves", "val": f"{len(n['ok_start'])}/{N_NIST} (A1 rule excluded 1: {short(n['res']['excluded_A1_rule'][0])},"
                          " non-finite C0_RT)", "res": "holds"},
        {"item": "validity V2 (NIST): failed per arm", "cond": f"A0, A0_rtdeck, A1, A2, A3, R_pop each <= 10% failed "
         f"(<= {lim_f:.1f} of 67; counted over the {len(n['ok_start'])} V1 devices)", "val": fail_txt,
         "res": "FAILS: " + ", ".join(f"{a} {n['failed'][a]} = {n['failed'][a] / N_NIST:.1%}" for a in bad_arms)},
        {"item": "validity V2 (NIST): paired per contrast", "cond": f"every K contrast >= 80% paired (>= {lim_p} of 67)",
         "val": pair_txt, "res": "FAILS: " + ", ".join(f"{k} {n['paired'][k]} = {n['paired'][k] / N_NIST:.0%}"
                                                       for k in bad_k)},
        {"item": "validity V3", "cond": "prereg.json tracked, unmodified, in an ancestor of HEAD (start, lodo, decide)",
         "val": f"prereg {COMMITS['prereg']} in force; guard ok in lodo and decide", "res": "holds"},
        {"item": "validity V4", "cond": "ngspice timeouts reported", "val": "start 0, lodo NIST 0, lodo TU Delft 0",
         "res": "reported"},
    ]
    for k, a, c in CONTRASTS:
        b = n["cmp"]["all"][k]["bonferroni"]
        rows.append({"item": f"{k} (NIST, descriptive)", "cond": f"GMR({a}, {c}), device bootstrap, 99% (Bonferroni) CI",
                     "val": ci_text(b), "res": f"{n['labels'][k]} (descriptive)"})
    rows += [
        {"item": "TEST consequences", "cond": "TEST primary arm, confirmatory set, variant",
         "val": "none (outcome invalid); variant rule: no variant with 95% CI upper < 1 vs A1", "res": "TEST stays locked"},
        {"item": "TU Delft replication", "cond": "same contrasts; changes no label", "val": "V1 45/45, V2 holds (A3 2 "
         "failed); labels " + ", ".join(f"{k} {t['labels'][k]}" for k, _, _ in CONTRASTS),
         "res": "valid; direction agrees with NIST on " + ", ".join(k for k, v in dec["tud_direction_agrees"].items() if v)},
    ]
    cols = [{"key": "item", "head": "item", "md": "item", "kind": "text"},
            {"key": "cond", "head": "pre-registered definition", "md": "pre-registered definition", "kind": "text"},
            {"key": "val", "head": "value", "md": "value", "kind": "text"},
            {"key": "res", "head": "outcome", "md": "outcome", "kind": "text"}]
    notes = [
        f"Pre-registration {rel(PREREG)} (in force from {COMMITS['prereg']}); decision {rel(DEC)}: outcome "
        f"'{dec['outcome']}', action '{dec['action']}'. Every value is recomputed here from the per-device scores in "
        "lodo_nist/results.json and lodo_tud/results.json and checked against decision.json.",
        "The A1, A2, A3 failures are the same 8 devices (A2 multiplies the A1 output by exp(g)); with the A1-rule "
        "device each method arm has 9 non-finite scores of 67. Failure cause: table 08.",
        "Because the run is invalid, the K labels are descriptive: they are not the pre-registered decision.",
    ]
    return rf.Table("01_table_i10_1_decision", "I10-1 decision and validity (NIST DEV decides; TU Delft replication)",
                    cols, rows, notes)


def t_arms(d: dict) -> rf.Table:
    n = d["nist"]
    dg = d["diag"]
    cause = {}
    for r in dg["op_runs"]:
        key = "Pclm <= 0" if r["fatal"][0].startswith("Fatal: Pclm") else "Lpeb < -Leff"
        cause.setdefault(r["arm"], {}).setdefault(key, 0)
        cause[r["arm"]][key] += 1
    cause["A2"] = cause["A1"]
    rows = []
    for a in ARMS["nist"]:
        g, fm, ff = n["agg"][a], n["fam_mean"][a], n["fam_fail"][a]
        extra = g["n_failed"] - 1
        why = "1 A1-rule device" if g["n_failed"] >= 1 else ""
        if a in cause:
            why += "; " + ", ".join(f"{v} {k}" for k, v in cause[a].items())
        elif a in ("A1_L", "V_ind", "V_mean"):
            ag = dg["agreement"][a]
            why += f"; {ag['failed_and_predicted']} Pclm <= 0 or Lpeb < -Leff (analytic)"
            if ag["failed_not_predicted"]:
                why += f", {len(ag['failed_not_predicted'])} other"
        elif a == "V_deck" and extra:
            why += f"; {extra} at the -269.15 C deck (2 Pclm <= 0 analytic, 12 not diagnosed)"
        vs_rt = "" if a == "A0_rtdeck" else ci_text(m.gmr_ci({t: n["sc"][a][t] for t in n["tags"]},
                                                              {t: n["sc"]["A0_rtdeck"][t] for t in n["tags"]}))
        rows.append({"arm": ARM_LABEL[a], "nf": f"{g['n'] - g['n_failed']}/{g['n_failed']}",
                     "ff": f"{ff['nmos']} / {ff['pmos']}", "mean": g["valid_only_mean"], "med": g["median"],
                     "comb": g["combined"], "nm": fm["nmos"], "pm": fm["pmos"], "vs": vs_rt, "why": why})
    cols = [{"key": "arm", "head": "arm", "md": "arm", "kind": "text"},
            {"key": "nf", "head": "finite/failed", "md": "finite / failed (67)", "kind": "rtext"},
            {"key": "ff", "head": "failed nMOS/pMOS", "md": "failed nMOS / pMOS", "kind": "rtext"},
            {"key": "mean", "head": "mean", "md": "mean RRMS (finite)", "kind": "rrms", "group": "RRMS, finite devices"},
            {"key": "med", "head": "median", "md": "median RRMS", "kind": "rrms", "group": "RRMS, finite devices"},
            {"key": "comb", "head": "combined", "md": "combined (nMOS / pMOS means)", "kind": "rrms",
             "group": "RRMS, finite devices"},
            {"key": "nm", "head": "nMOS", "md": "nMOS mean (36)", "kind": "rrms", "group": "family mean"},
            {"key": "pm", "head": "pMOS", "md": "pMOS mean (31)", "kind": "rrms", "group": "family mean"},
            {"key": "vs", "head": "GMR(arm, A0_rtdeck) [95% CI], wins", "md": "GMR(arm, A0_rtdeck) [95% CI], wins",
             "kind": "rtext"},
            {"key": "why", "head": "failures", "md": "failure causes", "kind": "text"}]
    notes = [
        f"Source {rel(LODO['nist'])}; strict paper RRMS on the held-out device's 4 K Vb = 0 curves, measured-only "
        "inclusion frozen from the measured 4 K current. Means over finite devices only (the device sets differ "
        "between arms). GMR(arm, A0_rtdeck) is computed here (post hoc, device bootstrap 95%).",
        f"Failure causes: {rel(DIAG)} (ngspice .op messages for A1, A3, R_pop; analytic BSIM4 binning check for the "
        "other arms). The C0_RT card of etest_pfet_g5v0d10v5_W0p42_L8_m8395p10 has Pclm <= 0 (A1 rule: failed in "
        "every arm).",
    ]
    return rf.Table("02_table_i10_1_arms", "I10-1 NIST arms: RRMS aggregates and failures (67 DEV devices)",
                    cols, rows, notes)


def t_family(d: dict) -> rf.Table:
    rows = []
    for name in ("nist", "tud"):
        x = d[name]
        for f in ("nmos", "pmos", "all"):
            sel = [t for t in x["tags"] if f == "all" or x["pol"][t] == f]
            row = {"ds": "NIST" if name == "nist" else "TU Delft", "fam": FAM_SHORT[f], "n": str(len(sel))}
            for a in ("A0_rtdeck", "A1", "A2", "R_pop", "R_oracle"):
                v = [x["sc"][a][t] for t in sel if math.isfinite(x["sc"][a][t])]
                row[a] = f"{np.mean(v):.3f} ({len(sel) - len(v)})"
            for k, _, _ in CONTRASTS:
                row[k] = ci_text(x["cmp"][f][k]["bonferroni"])
            rows.append(row)
    cols = [{"key": "ds", "head": "data", "md": "dataset", "kind": "text"},
            {"key": "fam", "head": "family", "md": "family", "kind": "text"},
            {"key": "n", "head": "n", "md": "devices", "kind": "rtext"}]
    cols += [{"key": a, "head": a, "md": f"{a} mean (failed)", "kind": "rtext", "group": "mean RRMS (failed)"}
             for a in ("A0_rtdeck", "A1", "A2", "R_pop", "R_oracle")]
    cols += [{"key": k, "head": k, "md": f"{k} GMR [99% CI], wins", "kind": "rtext",
              "group": "paired GMR, device bootstrap, 99% CI, wins / paired"} for k, _, _ in CONTRASTS]
    notes = ["K1 GMR(A1, A0), K2 GMR(A1, A0_rtdeck), K3 GMR(A1, R_pop), K4 GMR(A2, A1), K5 GMR(A3, A1). Per-family "
             "rows are pre-registered secondaries. NIST rows are descriptive (run invalid).",
             "TU Delft: A0 uses kT/q only (I6-1 card). TU Delft pMOS has 2 geometry groups, so each pMOS fold trains "
             "on the other group only; the 8 devices of pfet W1 L0.04 have A1 RRMS 28.7-46.8 (A0_rtdeck 0.18-0.24), "
             "which gives the pMOS K1 and K2 GMR of about 10.7."]
    return rf.Table("04_table_i10_1_family", "I10-1 per family: mean RRMS and K1-K5 (NIST descriptive; TU Delft "
                    "replication)", cols, rows, notes)


def t_tud(d: dict) -> rf.Table:
    n, t = d["nist"], d["tud"]
    rows = []
    for k, a, c in CONTRASTS:
        tb, tc, nb = t["cmp"]["all"][k]["bonferroni"], t["cmp"]["all"][k]["cluster_bonferroni"], \
            n["cmp"]["all"][k]["bonferroni"]
        rows.append({"k": K_LABEL[k], "tdev": ci_text(tb), "tclu": ci_text(tc, wins=False),
                     "tlab": t["labels"][k], "tclab": m.ci_label(tc), "nist": ci_text(nb),
                     "nlab": n["labels"][k] + " (descr.)",
                     "agree": "yes" if d["dec"]["tud_direction_agrees"][k] else "no"})
    vt = t["dd"]["validity"]
    rows.append({"k": "validity", "tdev": f"V1 {vt['n_finite_start']}/{N_TUD}; V2 failed "
                 + ", ".join(f"{a} {vt['failed'][a]}" for a in PRIMARY_ARMS), "tclu": "",
                 "tlab": "valid", "tclab": "", "nist": "V2 fails", "nlab": "invalid", "agree": ""})
    mt, mn = t["dd"]["means_valid_only"], n["dd"]["means_valid_only"]
    rows.append({"k": "mean RRMS A0 / A0_rtdeck / A1 / A2 / R_pop / R_oracle",
                 "tdev": " / ".join(f"{mt[a]:.3f}" for a in ("A0", "A0_rtdeck", "A1", "A2", "R_pop", "R_oracle")),
                 "tclu": "", "tlab": "", "tclab": "",
                 "nist": " / ".join(f"{mn[a]:.3f}" for a in ("A0", "A0_rtdeck", "A1", "A2", "R_pop", "R_oracle")),
                 "nlab": "", "agree": ""})
    rows.append({"k": "median RRMS A0_rtdeck / A1 / A2 / R_oracle",
                 "tdev": " / ".join(f"{t['agg'][a]['median']:.3f}" for a in ("A0_rtdeck", "A1", "A2", "R_oracle")),
                 "tclu": "", "tlab": "", "tclab": "",
                 "nist": " / ".join(f"{n['agg'][a]['median']:.3f}" for a in ("A0_rtdeck", "A1", "A2", "R_oracle")),
                 "nlab": "", "agree": ""})
    rows.append({"k": "gap fraction toward R_oracle A1 / A2 / A3",
                 "tdev": " / ".join(f"{t['gap'][a]:+.3f}" for a in ("A1", "A2", "A3")), "tclu": "", "tlab": "",
                 "tclab": "", "nist": " / ".join(f"{n['gap'][a]:+.3f}" for a in ("A1", "A2", "A3")), "nlab": "",
                 "agree": ""})
    cols = [{"key": "k", "head": "contrast", "md": "contrast", "kind": "text"},
            {"key": "tdev", "head": "TU Delft device CI", "md": "TU Delft GMR [99% device CI], wins", "kind": "rtext",
             "group": "TU Delft 4.13 K (valid)"},
            {"key": "tclu", "head": "TU Delft cluster CI", "md": "TU Delft [99% cluster CI, 7 groups]", "kind": "rtext",
             "group": "TU Delft 4.13 K (valid)"},
            {"key": "tlab", "head": "label", "md": "TU Delft label (device CI)", "kind": "text",
             "group": "TU Delft 4.13 K (valid)"},
            {"key": "tclab", "head": "cluster label", "md": "TU Delft label (cluster CI)", "kind": "text",
             "group": "TU Delft 4.13 K (valid)"},
            {"key": "nist", "head": "NIST", "md": "NIST GMR [99% CI], wins", "kind": "rtext", "group": "NIST (invalid)"},
            {"key": "nlab", "head": "label", "md": "NIST label", "kind": "text", "group": "NIST (invalid)"},
            {"key": "agree", "head": "same direction", "md": "same direction", "kind": "text"}]
    notes = ["TU Delft: I6-1 DEV devices, stored 295 K cards (C0_RT), C4 warm start from them at the 295 K stage deck; "
             "A0 at 4.13 K changes kT/q only. Labels from the device-bootstrap CI (as decision.json); cluster labels "
             "are computed here. The replication changes no label.",
             "Gap fraction = ln GMR(arm, A0) / ln GMR(R_oracle, A0) (decision.json; 1 = R_oracle, 0 = A0)."]
    return rf.Table("06_table_i10_1_tud_replication", "I10-1 TU Delft replication next to NIST", cols, rows, notes)


def t_failures(d: dict) -> rf.Table:
    dg = d["diag"]
    by = {}
    for r in dg["op_runs"]:
        by.setdefault(r["tag"], {})[r["arm"]] = r
    rows = []

    def cause(r):
        if r is None:
            return ""
        e = r["analytic"]
        if r["fatal"][0].startswith("Fatal: Pclm"):
            return f"Pclm {e['pclm_eff']:+.3f}"
        return f"Lpeb/Leff {e['lpeb_over_leff']:.3f}"

    for tag in sorted(by, key=lambda x: (d["rows"][x]["dev_type"], d["rows"][x]["L_um"], d["rows"][x]["W_um"])):
        rr = by[tag]
        any_r = next(iter(rr.values()))
        c0 = any_r["C0_RT_analytic"]
        bin_txt = "in bin" if any_r["covered"] else f"stretch L x{any_r['stretch_L']:.3f}" + (
            f", W x{any_r['stretch_W']:.2f}" if any_r["stretch_W"] > 1 else "")
        rows.append({"dev": short(tag), "type": "nMOS" if "nfet" in tag else "pMOS",
                     "WL": f"{any_r['W_um']:g} / {any_r['L_um']:g}", "bin": bin_txt,
                     "c0": f"Pclm {c0['pclm_eff']:.3f}; Lpeb/Leff {c0['lpeb_over_leff']:.3f}",
                     "a1": cause(rr.get("A1")), "a3": cause(rr.get("A3")), "rp": cause(rr.get("R_pop")),
                     "c0f": cause(rr.get("C0_RT"))})
    cols = [{"key": "dev", "head": "device", "md": "device", "kind": "text"},
            {"key": "type", "head": "type", "md": "type", "kind": "text"},
            {"key": "WL", "head": "W / L (um)", "md": "W / L (um)", "kind": "rtext"},
            {"key": "bin", "head": "bin", "md": "bin", "kind": "text"},
            {"key": "c0", "head": "C0_RT effective values", "md": "C0_RT effective Pclm; Lpeb / Leff", "kind": "text"},
            {"key": "a1", "head": "A1 = A2", "md": "A1 (= A2) fatal check, effective value", "kind": "text",
             "group": "ngspice fatal check of the stored card (effective value)"},
            {"key": "a3", "head": "A3", "md": "A3 fatal check", "kind": "text",
             "group": "ngspice fatal check of the stored card (effective value)"},
            {"key": "rp", "head": "R_pop", "md": "R_pop fatal check", "kind": "text",
             "group": "ngspice fatal check of the stored card (effective value)"},
            {"key": "c0f", "head": "C0_RT", "md": "C0_RT fatal check", "kind": "text",
             "group": "ngspice fatal check of the stored card (effective value)"}]
    ag = dg["agreement"]
    lp = [r["C0_RT_analytic"]["lpeb_over_leff"] for r in dg["op_runs"] if r["arm"] in ("A1", "A3")
          and r["fatal"][0].startswith("Fatal: Lpeb")]
    pc = [r["C0_RT_analytic"]["pclm_eff"] for r in dg["op_runs"] if r["arm"] in ("A1", "A3")
          and r["fatal"][0].startswith("Fatal: Pclm")]
    notes = [
        f"Source {rel(DIAG)} (post hoc). Each failed record of A1, A3, R_pop and the excluded C0_RT record was run once "
        f"as an ngspice .op ({dg['n_op_runs']} runs, {dg['runtime_s']} s); every run stops with one BSIM4 parameter-check "
        "fatal: 'Pclm = ... is not positive' or 'Lpeb = ... is less than -Leff'. Lpeb/Leff below -1 is fatal.",
        "Effective value = P0 + lP/Leff + wP/Weff + pP/(Leff Weff) of the device's bin (binunit 2). The analytic check "
        "predicts every failure of A1 (8/8), A2 (= A1), A3 (8/8), R_pop (12/12), A1_L "
        f"({ag['A1_L']['failed_and_predicted']}/{ag['A1_L']['n_failed']}), V_ind "
        f"({ag['V_ind']['failed_and_predicted']}/{ag['V_ind']['n_failed']}), C0_RT (1/1); V_sub7 (7 parameters, no "
        "PCLM / LPEB) has 0 failures. R_pop: 6 more cards sit at Lpeb = -Leff (relative 5e-11; the check is strict, "
        "finite).",
        "Mechanism: the I8-1 box bounds the base parameter P0 (log scale, sign kept), not the size-dependent value. "
        f"For the listed geometries the C0_RT effective value lies near the limit (A1 / A3 Lpeb failures, nMOS L "
        f"0.40-0.45 um, bin-stretched: Lpeb/Leff {min(lp):.3f} to {max(lp):.3f} at C0_RT; A1 / A3 Pclm failures, pMOS: "
        f"Pclm {min(pc):.3f} to {max(pc):.3f} at C0_RT, with negative l / w terms), and the training median shift "
        "(or the R_pop coordinates) moves P0 across it.",
    ]
    return rf.Table("08_table_i10_1_failures", "I10-1 failure diagnosis: failed NIST records and the BSIM4 check that "
                    "stops ngspice", cols, rows, notes)


# ================================================================== figures

def fig_forest(d: dict, path: Path) -> list[str]:
    n = d["nist"]
    keys = [(k, K_LABEL[k]) for k, _, _ in CONTRASTS]
    var = [("V_V_ind", "V_ind vs A1"), ("V_V_mean", "V_mean vs A1"), ("V_V_sub7", "V_sub7 vs A1"),
           ("V_V_deck", "V_deck vs A1"), ("S_A1_L_vs_A1", "A1_L vs A1"), ("S_A2_vs_R_pop", "A2 vs R_pop"),
           ("S_A1_vs_R_oracle", "A1 vs R_oracle")]
    fig, ax = plt.subplots(figsize=(12.5, 8.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    offs = {"all": 0.0, "nmos": -0.24, "pmos": 0.24}
    cols = {"all": INK, "nmos": BLUE, "pmos": ORANGE}
    items = keys + var
    nn = len(items)
    res = n["res"]["comparisons"]
    for i, (key, lab) in enumerate(items):
        yy = nn - 1 - i
        for s in ("all", "nmos", "pmos"):
            c = n["cmp"][s][key]["bonferroni"] if key.startswith("K") else res[s][key]["bonferroni"]
            y0 = yy + offs[s]
            ax.plot([c["ci_low"], c["ci_high"]], [y0, y0], color=cols[s], lw=1.8 if s == "all" else 1.2)
            ax.scatter(c["ratio"], y0, color=cols[s], s=34 if s == "all" else 20, zorder=4,
                       marker="o" if s != "pmos" else "s")
            if s == "all":
                lab_txt = f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}] {c['wins']}/{c['n_used']}"
                if key.startswith("K"):
                    lab_txt += f"  ({n['labels'][key]}, descriptive)"
                ax.annotate(lab_txt, (max(c["ci_high"], 1.0), y0), xytext=(8, -3), textcoords="offset points",
                            fontsize=7.5, color=INK)
        if i == len(keys) - 1:
            ax.axhline(yy - 0.5, color=GREY, lw=0.6)
    ax.axvline(1, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 40)
    ax.set_yticks(range(nn))
    ax.set_yticklabels([lab for _, lab in items][::-1], fontsize=8.5)
    ax.set_xlabel("paired GMR(A, B) = exp(mean ln(RRMS_A / RRMS_B)), device bootstrap, 99% (Bonferroni) CI; "
                  "below 1: A lower", fontsize=9)
    for s, lab in (("all", "all NIST DEV (value, CI, wins / paired)"), ("nmos", "nMOS (36)"), ("pmos", "pMOS (31)")):
        ax.plot([], [], color=cols[s], marker="o" if s != "pmos" else "s", label=lab)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.text(0.99, 0.985, "DESCRIPTIVE: run invalid (V2 failed); no label is a decision",
            transform=ax.transAxes, ha="right", va="top", fontsize=10, color=RED, fontweight="bold")
    ax.set_title("I10-1 NIST DEV: pre-registered contrasts K1-K5 (top) and secondary / variant contrasts (bottom)",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO["nist"]), rel(DEC)]


def fig_replication(d: dict, path: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    nk = len(CONTRASTS)
    series = (("nist", "bonferroni", 0.22, RED, "o", "NIST, device CI (invalid run, descriptive)"),
              ("tud", "bonferroni", 0.0, BLUE, "s", "TU Delft, device CI (valid)"),
              ("tud", "cluster_bonferroni", -0.22, AQUA, "D", "TU Delft, cluster CI (7 groups)"))
    for i, (k, _, _) in enumerate(CONTRASTS):
        yy = nk - 1 - i
        for name, lv, off, col, mk, _ in series:
            c = d[name]["cmp"]["all"][k][lv]
            ax.plot([c["ci_low"], c["ci_high"]], [yy + off, yy + off], color=col, lw=1.6)
            ax.scatter(c["ratio"], yy + off, color=col, marker=mk, s=30, zorder=4)
            ax.annotate(f"{c['ratio']:.3f}", (c["ci_high"], yy + off), xytext=(5, -3), textcoords="offset points",
                        fontsize=7.2, color=col)
    for _, _, _, col, mk, lab in series:
        ax.plot([], [], color=col, marker=mk, label=lab)
    ax.axvline(1, color=INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_yticks(range(nk))
    ax.set_yticklabels([K_LABEL[k] for k, _, _ in CONTRASTS][::-1], fontsize=8.5)
    ax.set_xlabel("paired GMR, 99% (Bonferroni) CI, log scale; below 1: method lower", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    t = d["tud"]["labels"]
    ax.set_title("I10-1 TU Delft replication (labels " + ", ".join(f"{k} {t[k]}" for k in t) + ") next to NIST",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO["tud"]), rel(LODO["nist"]), rel(DEC)]


def fig_gap(d: dict, path: Path) -> list[str]:
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.2), dpi=DPI, gridspec_kw={"width_ratios": [2.2, 2.2, 1.2]})
    fig.patch.set_facecolor(BG)
    arms = ("A0", "A0_rtdeck", "A1", "A2", "A3", "R_pop", "R_oracle")
    rng = np.random.default_rng(0)
    for ax, name in zip(axs[:2], ("nist", "tud")):
        style(ax)
        x = d[name]
        for i, a in enumerate(arms):
            v = np.array([x["sc"][a][t] for t in x["tags"]])
            ok = np.isfinite(v)
            jit = rng.uniform(-0.18, 0.18, ok.sum())
            mk = np.array(["s" if x["pol"][t] == "pmos" else "o" for t in x["tags"]])[ok]
            for sym in ("o", "s"):
                sel = mk == sym
                ax.scatter(i + jit[sel], v[ok][sel], s=11, color=ARM_COLOR[a], alpha=0.65, lw=0, marker=sym)
            ax.plot([i - 0.3, i + 0.3], [np.mean(v[ok])] * 2, color=INK, lw=1.6)
            ax.plot([i - 0.3, i + 0.3], [np.median(v[ok])] * 2, color=INK, lw=1.0, ls=":")
            nf = int((~ok).sum())
            ax.annotate(f"mean {np.mean(v[ok]):.3f}\nmedian {np.median(v[ok]):.3f}" + (f"\nfailed {nf}" if nf else ""),
                        (i, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=6.8, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, fontsize=8, rotation=20)
        ax.set_ylabel("held-out 4 K RRMS (strict paper RRMS)", fontsize=9)
        ax.set_title(("(a) " if name == "nist" else "(b) ") + DS_LABEL[name] + (" - invalid run" if name == "nist"
                                                                                 else ""),
                     fontsize=9.5, color=INK, loc="left", pad=40)
    ax = axs[2]
    style(ax)
    for j, (name, col) in enumerate((("nist", RED), ("tud", BLUE))):
        g = d[name]["gap"]
        for i, a in enumerate(("A1", "A2", "A3")):
            ax.bar(i + (j - 0.5) * 0.36, g[a], width=0.34, color=col, alpha=0.85)
            ax.annotate(f"{g[a]:+.2f}", (i + (j - 0.5) * 0.36, g[a]), xytext=(0, 3 if g[a] >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    from matplotlib.patches import Patch
    hs = [Patch(color=RED, alpha=0.85, label="NIST (descriptive)"), Patch(color=BLUE, alpha=0.85, label="TU Delft")]
    ax.axhline(0, color=INK, lw=0.7)
    ax.axhline(1, color=RED, lw=0.7, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels(("A1", "A2", "A3"), fontsize=9)
    ax.set_ylim(-0.5, 1.1)
    ax.set_ylabel("ln GMR(arm, A0) / ln GMR(R_oracle, A0)", fontsize=9)
    ax.legend(handles=hs, fontsize=8, frameon=False, loc="upper left")
    ax.set_title("(c) gap fraction toward R_oracle\n(0 = A0, 1 = R_oracle)", fontsize=9.5, color=INK, loc="left")
    fig.suptitle("I10-1: per-device held-out 4 K RRMS by arm (solid = mean, dotted = median over finite devices; "
                 "circles nMOS, squares pMOS) and gap fraction toward the oracle", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(LODO["nist"]), rel(LODO["tud"]), rel(DEC)]


def fig_margins(d: dict, path: Path) -> list[str]:
    dg = d["diag"]
    an = dg["analytic"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    tags = sorted(an["A1"])
    for ax, (key, lim, ylab, title, types) in zip(axs, (
            ("lpeb_over_leff", -1.0, "effective Lpeb / Leff (fatal below -1)", "(a) LPEB, nfet_g5v0d10v5", ("nfet",)),
            ("pclm_eff", 0.0, "effective Pclm (fatal at <= 0)", "(b) PCLM, both types", ("nfet", "pfet")))):
        style(ax)
        for t in tags:
            if not any(x in t for x in types):
                continue
            L = d["rows"][t]["L_um"]
            c0, a1, a3, rp = (an[a][t][key] for a in ("C0_RT", "A1", "A3", "R_pop"))
            mk = "s" if "pfet" in t else "o"
            fail = an["A1"][t]["failed"]
            ax.plot([L, L], [c0, a1], color=RED if fail else GREY, lw=0.8, alpha=0.8)
            ax.scatter(L, c0, s=16, color=MUTED, marker=mk, zorder=3)
            ax.scatter(L, a1, s=22, color=RED if fail else BLUE, marker=mk, zorder=4)
            ax.scatter(L, rp, s=14, facecolor="none", edgecolor=YELLOW if not an["R_pop"][t]["failed"] else RED,
                       marker=mk, zorder=4, lw=0.9)
        ax.axhline(lim, color=RED, lw=0.9, ls="--")
        ax.set_xscale("log")
        if key == "pclm_eff":
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlabel("drawn L (um)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    h = axs[1]
    h.scatter([], [], s=16, color=MUTED, label="C0_RT card (A0_rtdeck)")
    h.scatter([], [], s=22, color=BLUE, label="A1 card, finite")
    h.scatter([], [], s=22, color=RED, label="A1 card, failed (line from C0_RT)")
    h.scatter([], [], s=14, facecolor="none", edgecolor=YELLOW, label="R_pop card, finite")
    h.scatter([], [], s=14, facecolor="none", edgecolor=RED, label="R_pop card, failed")
    h.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("I10-1 NIST: size-dependent BSIM4 values of the C0_RT, A1 and R_pop cards (analytic, from the bin "
                 "coefficients; post hoc diagnosis); circles nMOS, squares pMOS", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(DIAG), rel(MANIFEST)]


def fig_iv(tag: str, ex: dict, d: dict, path: Path, why: str) -> list[str]:
    dv = ex["devices"][tag]
    sc = d["nist"]["sc"]
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    arms = (("A0_rtdeck", ":", 1.1, "A0_rtdeck: C0_RT card at the RT deck (NGSpice)"),
            ("A1", "-", 1.2, "A1: C0_RT + median training shift (NGSpice)"),
            ("A2", "--", 1.3, "A2: A1 NGSpice output x exp(g) (applied in Python)"),
            ("R_oracle", "-.", 1.1, "R_oracle: own C4 card, uses 4 K data (NGSpice)"))
    for row, kind in enumerate(("idvg", "idvd")):
        cs = [c for c in dv["curves"] if c["kind"] == kind]
        for col, scale in enumerate(("log", "linear") if kind == "idvg" else ("linear", "log")):
            ax = axs[row, col]
            style(ax)
            for k, c in enumerate(cs):
                color = p81.IV_COLORS[k % len(p81.IV_COLORS)]
                x = np.abs(np.asarray(c["vg"] if kind == "idvg" else c["vd"]))
                meas = np.abs(np.asarray(c["meas"]))
                step = 4 if len(x) > 100 else 1
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
    h.scatter([], [], s=12, color=INK, label="measured at 4 K (every 4th point on 331- / 133-point sweeps)")
    h.legend(fontsize=7.3, frameon=False, loc="upper left")
    bin_txt = f"bin {dv['bin']}" + ("" if dv["covered"] else " (stretched)")
    fig.suptitle(f"I10-1 {short(tag)} ({FAM_SHORT[dv['polarity']]}, W {dv['W_um']:g} um, L {dv['L_um']:g} um, "
                 f"{bin_txt}); ln(A1 / A0_rtdeck) {dv['log_ratio_A1_A0_rtdeck']:+.3f}; {why}\n"
                 "ngspice-41 arm64, open sky130 tt card, deck 30 C; stored cards re-simulated (RRMS reproduced, relative "
                 f"difference {max(v['rel_diff'] for v in ex['checks'][tag].values()):.0e}); NIST run invalid "
                 "(descriptive)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(EXAMPLES), rel(LODO["nist"]), rel(I10 / "lodo_nist/cards.json"),
            rel(I10 / "lodo_nist/corrections.json.gz")]


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
            assert close(v["stored"], d["nist"]["sc"][a][tag]), (tag, a)
    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(t_decision(d), "Table of the outcome (INVALID, stopped by rule), validity V1-V4 with the V2 failed "
              "counts per arm and paired counts per contrast, the descriptive K1-K5 values and the TU Delft "
              "replication labels", [rel(PREREG), rel(DEC), rel(LODO["nist"]), rel(START["nist"])])
    add_table(t_arms(d), "Table of the NIST arms: finite / failed devices (per family), mean, median, combined and "
              "family-mean RRMS, GMR vs A0_rtdeck (post hoc) and the failure causes", [rel(LODO["nist"]), rel(DIAG)])
    src = fig_forest(d, folder / "03_i10_1_forest_K1_K5.png")
    items.append(("03_i10_1_forest_K1_K5.png", "Forest plot (log scale) of K1-K5 and of the variant / secondary "
                  "contrasts on NIST, all / nMOS / pMOS, 99% (Bonferroni) device-bootstrap CI; marked descriptive "
                  "because the run is invalid", src))
    add_table(t_family(d), "Table per family (nMOS, pMOS, all) for NIST and TU Delft: mean RRMS with failed counts "
              "for A0_rtdeck, A1, A2, R_pop, R_oracle, and K1-K5 GMR with 99% CI",
              [rel(LODO["nist"]), rel(LODO["tud"]), rel(DEC)])
    src = fig_replication(d, folder / "05_i10_1_tud_replication.png")
    items.append(("05_i10_1_tud_replication.png", "K1-K5 on TU Delft (device and 7-group cluster CI, valid) next to "
                  "NIST (device CI, descriptive), log scale", src))
    add_table(t_tud(d), "Table of the TU Delft replication: K1-K5 with device and cluster CIs and labels, NIST values, "
              "direction agreement, validity, mean / median RRMS and gap fractions",
              [rel(LODO["tud"]), rel(LODO["nist"]), rel(DEC)])
    src = fig_gap(d, folder / "07_i10_1_gap_to_oracle.png")
    items.append(("07_i10_1_gap_to_oracle.png", "Per-device held-out 4 K RRMS by arm (A0, A0_rtdeck, A1, A2, A3, R_pop, "
                  "R_oracle) for NIST (a) and TU Delft (b), and the gap fraction toward R_oracle of A1, A2, A3 (c)",
                  src))
    add_table(t_failures(d), "Table of the failure diagnosis: every failed NIST device with its bin, the C0_RT "
              "effective Pclm and Lpeb / Leff, and the BSIM4 fatal check of the A1 (= A2), A3, R_pop and C0_RT cards",
              [rel(DIAG), rel(MANIFEST)])
    src = fig_margins(d, folder / "09_i10_1_bsim4_check_margins.png")
    items.append(("09_i10_1_bsim4_check_margins.png", "Effective (size-dependent) Lpeb / Leff of nMOS (a) and Pclm of "
                  "both types (b) for the C0_RT, A1 and R_pop cards vs L, with the BSIM4 fatal limits; failed cards "
                  "in red", src))
    for k, tag in enumerate(ex["devices"]):
        dv = ex["devices"][tag]
        fam = dv["polarity"]
        sel = [math.log(d["nist"]["sc"]["A1"][t] / d["nist"]["sc"]["A0_rtdeck"][t]) for t in d["nist"]["tags"]
               if d["nist"]["pol"][t] == fam and all(math.isfinite(d["nist"]["sc"][a][t])
                                                     for a in ("A0_rtdeck", "A1", "A2", "R_oracle"))]
        why = f"nearest the {FAM_SHORT[fam]} median ln(A1 / A0_rtdeck) ({float(np.median(sel)):+.3f})"
        name = f"{10 + k:02d}_i10_1_iv_{short(tag)}.png"
        src = fig_iv(tag, ex, d, folder / name, why)
        items.append((name, f"Example I-V, {short(tag)} ({FAM_SHORT[fam]}; {why}): measured 4 K points, NGSpice "
                      "A0_rtdeck (dotted), A1 (solid), A2 (A1 output x exp(g), dashed) and R_oracle (dash-dot); Id-Vg "
                      "log / linear, Id-Vd linear / log", src))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i10_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. NIST DEV decides; the run is invalid (V2), so the NIST K "
          "labels are descriptive. Strict paper RRMS on the held-out 4 K Vb = 0 curves.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    n, t = d["nist"], d["tud"]
    dg = d["diag"]
    cn = n["cmp"]["all"]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice fit, no training). The example I-V curves "
              "(figures 10-11) are NGSpice re-simulations of stored cards by "
              f"`scripts/research/explore/i10_1_examples.py` ({rel(EXAMPLES)}, {ex['runtime_s']} s; RRMS reproduced "
              "with relative difference 0). The failure diagnosis (table 08, figure 09) is from "
              f"`scripts/research/explore/i10_1_diagnose.py` ({rel(DIAG)}; {dg['n_op_runs']} ngspice .op runs, "
              f"{dg['runtime_s']} s; post hoc).", "",
              "## Contents", "",
              "- I10-1: prediction of a device's 4 K I-V from its room-temperature data and training devices with both "
              "temperatures. NIST ETest nfet_g5v0d10v5 (36) / pfet_g5v0d10v5 (31) DEV devices paired with the google "
              "sky130 raw-data RT files (decides); TU Delft 40 nm I6-1 DEV, 45 devices (replication). Pre-registration "
              f"`{rel(PREREG)}` (in force from {COMMITS['prereg']}); decision `{rel(DEC)}`. Queue jobs: "
              + ", ".join(f"{k} ({v} s)" for k, v in JOBS.items()) + ".",
              f"- Outcome: INVALID (stopped by rule; action '{d['dec']['action']}'). V1 holds ({len(n['ok_start'])}/67). "
              "V2 fails: failed A0 0, A0_rtdeck 0, A1 8, A2 8, A3 8, R_pop 12 (limit 6.7 = 10% of 67); paired "
              + ", ".join(f"{k} {n['paired'][k]}" for k, _, _ in CONTRASTS) + " (limit 54 = 80% of 67; K3 fails). "
              "V3 holds, V4 0 timeouts.",
              "- Failure cause (post hoc): every failed record stops ngspice at a BSIM4 parameter check of the "
              "size-dependent value: A1 (= A2) 6 Lpeb < -Leff (nMOS L 0.40-0.45 um, bin-stretched) and 2 Pclm <= 0 "
              "(pMOS); A3 6 Lpeb, 2 Pclm; R_pop 12 Pclm; the A1-rule device's C0_RT card Pclm <= 0. The shift is "
              "applied to the base parameter P0 inside the I8-1 box; the bin's l / w / p terms carry the effective "
              "value across the limit. An analytic check from the bin coefficients predicts every failure of the "
              "primary arms with no simulation. V_sub7 (7 parameters, no PCLM / LPEB) has 0 failures.",
              "- Descriptive NIST values (finite devices): mean RRMS "
              + ", ".join(f"{a} {n['agg'][a]['valid_only_mean']:.3f}" for a in ("A0", "A0_rtdeck", "A1", "A2", "A3",
                                                                               "R_pop", "R_oracle"))
              + "; median A0_rtdeck " + f"{n['agg']['A0_rtdeck']['median']:.3f}, A1 {n['agg']['A1']['median']:.3f}, "
              f"R_oracle {n['agg']['R_oracle']['median']:.3f}.",
              "- Descriptive NIST contrasts (99% CI): " + "; ".join(f"{k} {ci_text(cn[k]['bonferroni'])} {n['labels'][k]}"
                                                                    for k, _, _ in CONTRASTS) + ". Gap fraction toward "
              f"R_oracle A1 {n['gap']['A1']:.2f}, A2 {n['gap']['A2']:.2f}, A3 {n['gap']['A3']:.2f}.",
              "- TU Delft (valid): " + "; ".join(f"{k} {ci_text(t['cmp']['all'][k]['bonferroni'])} {t['labels'][k]}"
                                                for k, _, _ in CONTRASTS) + f". Mean RRMS A0_rtdeck "
              f"{t['agg']['A0_rtdeck']['valid_only_mean']:.3f}, A1 {t['agg']['A1']['valid_only_mean']:.3f}, R_oracle "
              f"{t['agg']['R_oracle']['valid_only_mean']:.3f}; gap fraction A1 {t['gap']['A1']:+.2f}. Direction agrees "
              "with NIST on " + ", ".join(k for k, v in d["dec"]["tud_direction_agrees"].items() if v) + ".",
              "- Post hoc (not pre-registered): the failure diagnosis, GMR vs A0_rtdeck in table 02, the TU Delft "
              "cluster labels, the per-device strip plots and the example I-V figures.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    summary = [
        "I10-1 RT -> 4 K (NIST g5v0d10v5 DEV, 67 devices): INVALID, stopped by rule. V2 failed A1 / A2 / A3 8 each, "
        f"R_pop 12 (limit 6.7); K3 paired {n['paired']['K3']} < 54.",
        "Cause: shifted cards fail BSIM4 checks of the size-dependent values (A1 6 Lpeb < -Leff on bin-stretched nMOS "
        "L 0.40-0.45 um, 2 Pclm <= 0; R_pop 12 Pclm <= 0); predicted analytically for every primary-arm failure.",
        f"Descriptive: mean A1 {n['agg']['A1']['valid_only_mean']:.3f}, A0_rtdeck "
        f"{n['agg']['A0_rtdeck']['valid_only_mean']:.3f}, R_pop {n['agg']['R_pop']['valid_only_mean']:.3f}, R_oracle "
        f"{n['agg']['R_oracle']['valid_only_mean']:.3f}; K2 {ci_text(cn['K2']['bonferroni'], wins=False)}, K3 "
        f"{ci_text(cn['K3']['bonferroni'], wins=False)} inconclusive. TU Delft (valid): K2 higher, K3 lower.",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
