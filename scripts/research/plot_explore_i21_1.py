#!/usr/bin/env python3
"""Report batch for I21-1: (A) NIST Part B of the selected g v2 full-data
element, (B) TU Delft v1-recipe S3_D on state-C records, (C) selected g v2
recipe on IHP SG13G2 233 / 343 / 398 K.

Reads stored files under ``out/research/explore/I21_1/`` (prereg,
partA_<ds>/{summary, q12, q3, m2, accuracy, compile}.json and arrays,
s3tud/s3tud.json, ihp_<T>K/{base, inner, sel, pens, decision.json},
decision.json) and the I17-2 / I18-2 / I19-1 / I20-1 stored files they refer
to.

NGSpice: the Part A I-V figures re-simulate the stock card, the M2 card
(card only) and the M2 card with the OSDI element (ngspice-41 OSDI build);
the Part C I-V figures re-simulate the stock PSP103 card and the C0_refit,
M_bias v1 and selected held-out cards (ngspice-46 + psp103.osdi).  Every
re-simulated RRMS is checked against the stored row.

  .venv/bin/python -I scripts/research/plot_explore_i21_1.py --date 2026-10-23
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA, VIOLET, RED, DPI,  # noqa: E402
                                   EXP, load, rel, style)
from matplotlib.lines import Line2D  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i21_1"
I21 = EXP / "I21_1"
DSS = {"nist4": "NIST lvt (I8-1, 77 K card)", "nistrt": "NIST RT card (I9-1, room-temperature card)"}
TEMPS = (233, 343, 398)
COMMITS = {"prereg": "1808a01", "code": "0688e64"}
IV_COLORS = ["#1f4e9c", "#3a7d44", "#c2571a", "#7b3fa0", "#b8860b", "#2a9d8f", "#9b2226", "#495057", "#0077b6"]


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def e(x) -> str:
    return f"{x:.2e}" if x is not None and np.isfinite(x) else str(x)


def check(a: float, b: float, what: str, tol: float = 1e-9) -> None:
    if not abs(a - b) <= tol:
        raise AssertionError(f"{what}: {a} vs {b}")


# ---------------------------------------------------------------------------
# load and recompute
# ---------------------------------------------------------------------------
def load_all() -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    from cryoml.explore import i17_2_ihp as m17b
    from cryoml.explore import i21_1_close as m
    d = {"prereg": load(I21 / "prereg.json"), "dec": load(I21 / "decision.json"),
         "s3": load(I21 / "s3tud/s3tud.json")}
    n = 0
    for ds in m.DATASETS:
        B = I21 / f"partA_{ds}"
        d[ds] = {k: load(B / f"{k}.json") for k in ("summary", "q12", "q3", "m2", "accuracy", "compile")}
        S = d[ds]["summary"]
        dec = m.decide_part_a(S["values"])
        assert dec["deploy"] == S["decision"]["deploy"]
        q12 = list(d[ds]["q12"]["devices"].values())
        check(S["values"]["q1_vds0_max"], max(r["vds0"]["tight"]["elem_max"] for r in q12), f"{ds} vds0")
        assert S["values"]["q2_gds_sign"] == sum(sum(r["q2_gds_curves_introduced"]) for r in q12)
        assert S["values"]["q2_gm_sign"] == sum(sum(r["q2_gm_curves_introduced"]) for r in q12)
        n += 4
    for t, r in d["s3"]["per_temperature"].items():
        v = [x for x in r["per_record"].values()]
        check(float(np.nanmean(v)), r["S3_D_state_C"], f"S3 {t}", 1e-12)
        n += 1
    for T in TEMPS:
        B = I21 / f"ihp_{T}K"
        D = load(B / "decision.json")
        d[T] = {"dec": D, "base": load(B / "base/results.json"), "sel": load(B / "sel/results.json"),
                "inner": load(B / "inner/inner.json"), "pens": load(B / "pens/pens.json")}
        R = D["rows"]
        g = paired_ratio(R["selected"], R["C0_refit"])
        for k in ("ratio", "ci_low", "ci_high"):
            check(g[k], D["primary_gmr_selected_vs_C0_refit"][k], f"{T} primary {k}", 1e-12)
        g = paired_ratio(R["selected"], R["M_bias_v1"])
        for k in ("ratio", "ci_low", "ci_high"):
            check(g[k], D["secondary_gmr_selected_vs_v1"][k], f"{T} secondary {k}", 1e-12)
        assert m17b.primary_label(D["primary_gmr_selected_vs_C0_refit"], D["n_failed_selected_C0"]) == D["outcome"]
        assert m17b.secondary_label(D["secondary_gmr_selected_vs_v1"]) == D["secondary"]
        st = load(EXP / f"I5_1/lodo_{T}K/results.json")["per_device"]
        for a, k in (("C0_refit", "C0_refit"), ("M_bias_v1", "M_bias")):
            for tag, v in R[a].items():
                check(v, st[k][tag], f"{T} {a} {tag} vs I5-1")
        n += 8 + 36
    d["n_checked"] = n
    return d


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_setup(d: dict) -> rf.Table:
    P = d["prereg"]
    A = P["part_A"]
    rows = [{"c": "pre-registration", "v": f"out/research/explore/I21_1/prereg.md (commit {COMMITS['prereg']}), "
                                         f"code {COMMITS['code']}; DEV only, TEST locked; no amendment of a755e18"},
            {"c": "Part A models", "v": f"I20-1 full-data g v2, lambda*_full {A['models']['lambda_full_V']:g} V, "
                                         f"VDD {A['models']['vdd_V']:g} V, EPS {A['models']['eps_V']:g} V"},
            {"c": "Part A decks", "v": "; ".join(f"{k}: {v['n']} devices, {v['deck']}" for k, v in A["datasets"].items())},
            {"c": "Part A simulator", "v": A["simulator"]},
            {"c": "Part A gate current", "v": A["gate_current"]},
            {"c": "Part A Q3 device rule", "v": A["q3"]["device_rule"]},
            {"c": "Part B", "v": P["part_B"]["definition"] + "; records " + P["part_B"]["records"]},
            {"c": "Part C", "v": f"IHP SG13G2 {', '.join(str(t) for t in P['part_C']['temperatures_K'])} K, "
                                 f"{P['part_C']['setup']}; {P['part_C']['change_from_I17_2']}"}]
    return rf.Table("01_table_i21_1_setup", "I21-1 setup (pre-registered)", [col("c", "item", kind="text"),
                    col("v", "value", kind="text")], rows, [])


def t_a_validity(d: dict) -> rf.Table:
    rows = []
    for ds in ("nist4", "nistrt"):
        S, M2 = d[ds]["summary"], d[ds]["m2"]
        rows.append({"ds": DSS[ds], "va1": e(S["accuracy"]["va1_max_diff"]),
                     "va2": f"{len(M2['rows'])} rows, {M2['n_failed']} failed",
                     "va3": str(S["unit_tests"]["passed"]), "va4": f"{S['n_devices']} rows, {len(d[ds]['q3']['rows'])} Q3",
                     "va5": str(S["dev_check"]["ok"]),
                     "v": "valid" if S["validity"]["valid"] else "invalid: " + "; ".join(S["validity"]["fails"])})
    return rf.Table("02_table_i21_1_a_validity", "Part A validity (VA1-VA5)",
                    [col("ds", "dataset", kind="text"), col("va1", "VA1 max |card-only deck - M2 row| (<= 1e-9)"),
                     col("va2", "VA2 M2 refits"), col("va3", "VA3 unit tests"), col("va4", "VA4 rows"),
                     col("va5", "VA5 DEV only"), col("v", "result", kind="text")], rows,
                    ["VA1: the card-only measured-grid deck built by the Part B library code (ngspice-41 OSDI build) "
                     "on every M2 card, x exp(g) in Python, against the M2 in-sample row computed by the I20-1 "
                     "simulator (canonical ngspice-41)."])


THR_ROWS = (("q1_tight_viol", "Q1 equivalence violations (tight, fwd + exchanged)", "0", "core", "d"),
            ("q1_bad", "Q1 convergence messages + non-finite (element decks)", "0", "core", "d"),
            ("q1_vds0_max", "Q1 |I(V(d,s) = 0)| tight, raw element (A)", "<= 1e-12", "core", "e"),
            ("g_text_max", "Verilog-A g vs Python", "<= 1e-9", "core", "e"),
            ("g_sym_text_max", "Verilog-A g exchanged vs forward", "<= 1e-12", "core", "e"),
            ("swap_asym_max", "element-introduced exchange asymmetry", "<= 1e-3", "core", "e"),
            ("q2_ac_max", "Q2 gm / gds error vs element identity", "<= 1e-3", "core", "e"),
            ("cap_max", "Q2 capacitance error", "<= 1e-3", "core", "e"),
            ("q2_gm_sign", "gm curves with introduced sign change (strong inversion)", "0", "core", "d"),
            ("q3_fail", "Q3 circuits failing only with the element", "0", "core", "d"),
            ("q1_default_max", "default-tolerance relative difference (|I| > 1 nA)", "<= 1e-3", "restriction", "e"),
            ("q2_gds_sign", "gds curves with introduced sign change", "0", "restriction", "d"),
            ("runtime_ratio_max", "runtime ratio element / card (grid, circuits)", "<= 2", "restriction", "f"))


def t_a_checks(d: dict) -> rf.Table:
    rows = []
    for k, lab, thr, kind, fmt in THR_ROWS:
        r = {"c": lab, "k": kind, "t": thr}
        for ds in ("nist4", "nistrt"):
            v = d[ds]["summary"]["values"][k]
            txt = str(int(v)) if fmt == "d" else (f"{v:.2f}" if fmt == "f" else e(v))
            if k == "q2_gds_sign":
                txt += f" of {d[ds]['summary']['n_gds_curves']}"
            if k == "q2_gm_sign":
                txt += f" of {d[ds]['summary']['n_gm_curves']}"
            r[ds] = txt
        rows.append(r)
    rows.append({"c": "label (decide_deploy, I17-1 V(d,s) = 0 rule)", "k": "-", "t": "-",
                 **{ds: d[ds]["summary"]["label"] for ds in ("nist4", "nistrt")}})
    pr = d["prereg"]["predictions"]
    rows.append({"c": "predicted label", "k": "-", "t": "-", **{ds: pr[f"A label {ds}"] for ds in ("nist4", "nistrt")}})
    notes = []
    for ds in ("nist4", "nistrt"):
        S = d[ds]["summary"]
        notes.append(f"{DSS[ds]}: core failures: {', '.join(S['decision']['core_fail']) or 'none'}; restrictions: "
                     f"{', '.join(S['decision']['restrictions']) or 'none'}. V(d,s) = 0 raw element at default / r4 "
                     f"tolerance (reported): {e(S['vds0']['default']['elem_max'])} / {e(S['vds0']['r4']['elem_max'])} A; "
                     f"element-added current at tight tolerance {e(S['vds0_added_tight_max'])} A. Replica non-channel "
                     f"outputs max over {S['replica_nonchannel_n_values']} printed values: "
                     f"{e(max(S['replica_nonchannel_max'].values()) if S['replica_nonchannel_max'] else float('nan'))} A.")
    return rf.Table("03_table_i21_1_a_checks", "Part A core checks and restrictions vs thresholds (I17-1 rules)",
                    [col("c", "check", kind="text"), col("k", "type"), col("t", "threshold"),
                     col("nist4", "NIST lvt"), col("nistrt", "NIST RT card")], rows, notes)


def t_a_q3(d: dict) -> rf.Table:
    rows = []
    for ds in ("nist4", "nistrt"):
        Q = d[ds]["q3"]["rows"]
        for name in ("inv_A_vtc", "inv_B_vtc", "ro5_A", "ro5_B", "mirror_n", "mirror_p"):
            r = {"ds": DSS[ds].split(" (")[0], "c": name}
            for arm in ("published", "C0_refit", "M2_card", "M2_full"):
                x = Q[f"{name}|{arm}"]
                if not x["ok"]:
                    v = "fail" + (f" ({x['msgs'][0][:30]})" if x["msgs"] else
                                  (f" ({x.get('n_periods', 0)} periods)" if "n_periods" in x else ""))
                elif name.startswith("inv"):
                    v = f"V_M {x['vm']:.3f} V, gain {x['gain_max']:.1f}"
                elif name.startswith("ro5"):
                    v = f"{x['freq_hz'] / 1e6:.0f} MHz"
                else:
                    v = f"ratio {x['ratio_at_0p9']:.4f}"
                r[arm] = v
            rr = d[ds]["summary"]["runtime_circuits"].get(name)
            r["rt"] = f"{rr:.2f}" if rr is not None else "-"
            rows.append(r)
    notes = []
    for ds in ("nist4", "nistrt"):
        dv = d[ds]["q3"]["devices"]
        b = {r: d[ds]["q3"]["rows"]["inv_A_vtc|published"]["bins"].get(r) for r in dv}
        b.update(d[ds]["q3"]["rows"]["inv_B_vtc|published"]["bins"])
        notes.append(f"{DSS[ds]} circuit devices (role -> device, bin): "
                     + "; ".join(f"{k} -> {v} (bin {b.get(k)})" for k, v in dv.items()) + ".")
    notes.append("Supply 1.8 V. Q3 core count = circuits failing with M2_full while M2_card passes. Ring: tran with "
                 "the I14-1 settings, frequency of rising V_DD/2 crossings after 30% of tstop; failure: convergence "
                 "message, non-finite output or < 5 periods. Mirror ratio: |I_out| / I_ref at V_out = 0.9 V.")
    return rf.Table("04_table_i21_1_a_q3", "Part A Q3 circuits (sky130 decks, NIST cards)",
                    [col("ds", "dataset", kind="text"), col("c", "circuit", kind="text"), col("published", "published"),
                     col("C0_refit", "C0_refit"), col("M2_card", "M2 card"), col("M2_full", "M2 card + element"),
                     col("rt", "runtime element / card")], rows, notes)


def t_a_gds(d: dict) -> rf.Table:
    from cryoml.explore import i21_1_close as m
    rows = []
    for ds in ("nist4", "nistrt"):
        S = d[ds]["summary"]
        q12 = d[ds]["q12"]["devices"]
        n_dev = len(q12)
        for k, vg in enumerate(m.GDS_VGS):
            devs = [t for t, r in q12.items() if r["q2_gds_curves_introduced"][k]]
            pts = sum(r["q2_gds_points_introduced_dense"][k] for r in q12.values())
            rows.append({"ds": DSS[ds].split(" (")[0], "vg": f"{vg:g}", "n": f"{len(devs)} of {n_dev}",
                         "p": str(pts), "nm": str(sum(1 for t in devs if t.startswith("n"))),
                         "pm": str(sum(1 for t in devs if t.startswith("p")))})
        rows.append({"ds": DSS[ds].split(" (")[0], "vg": "all", "n": f"{len(S['gds_curves_introduced'])} of "
                     f"{S['n_gds_curves']}", "p": "-", "nm": "-", "pm": "-"})
    return rf.Table("05_table_i21_1_a_gds", "Part A gds curves with an introduced sign change (restriction)",
                    [col("ds", "dataset", kind="text"), col("vg", "|V(g,s)| (V)"), col("n", "curves (devices)"),
                     col("nm", "nMOS"), col("pm", "pMOS"), col("p", "dense 1 mV points with a sign change")], rows,
                    ["A curve counts when the element's gds (AC 1 MHz or the 1 mV dense sweep) changes sign at "
                     "|V(d,s)| >= 0.05 V where the card's gds does not. lambda 0 V: the penalty on negative "
                     "gm / gds is off in training (I20-1 full-data selection)."])


def t_a_acc(d: dict) -> rf.Table:
    rows = []
    for ds in ("nist4", "nistrt"):
        A = d[ds]["accuracy"]
        rows.append({"ds": DSS[ds], "s": f"{A['mean_stored_python']:.5f}", "c": f"{A['mean_card_python']:.5f}",
                     "pc": f"{A['mean_python_channel']:.5f}", "n": f"{A['mean_native']:.5f}",
                     "x": e(A["max_abs_native_minus_stored"]), "m": str(A["n_msgs"])})
    return rf.Table("06_table_i21_1_a_accuracy", "Part A element-deck accuracy of the M2 pairs (reported)",
                    [col("ds", "dataset", kind="text"), col("s", "M2 row (Python, I20-1 simulator)"),
                     col("c", "card-only deck x exp(g)"), col("pc", "card + replica: I_card + (exp(g)-1) I_rep"),
                     col("n", "native element deck"), col("x", "max per-device |native - M2 row|"),
                     col("m", "messages")], rows,
                    ["Mean in-sample RRMS of the M2 cards (full-data g in the loop) on the measured grid. The Python "
                     "rows scale the whole card current by exp(g); the element scales the replica channel current "
                     "only, so the difference is the non-channel current share."])


def t_b(d: dict) -> rf.Table:
    rows = []
    pr = d["prereg"]["predictions"]
    for t, lab in (("4", "4.13 K"), ("295", "295.0 K")):
        r = d["s3"]["per_temperature"][t]
        rows.append({"t": lab, "l": f"{r['S3_D_state_L']:+.4f}", "c": f"{r['S3_D_state_C']:+.4f}",
                     "dl": r["decision_state_L"], "dc": r["decision_state_C"], "s": str(r["sign_differs"]),
                     "p": pr[f"B sign S3_D state C {t}K"] + (": correct" if r["S3_D_state_C"] < 0 else ": wrong"),
                     "v": "valid" if r["validity"]["valid"] else "invalid",
                     "vb1": e(max(x["diff"] for x in d["s3"]["per_temperature"][t]["state_L_check"].values())
                              if "state_L_check" in d["s3"]["per_temperature"][t] else float("nan"))})
    return rf.Table("07_table_i21_1_b_s3", "Part B: TU Delft v1-recipe S3_D, state-L (I19-1) vs state-C records",
                    [col("t", "temperature"), col("l", "S3_D state L (I19-1)"), col("c", "S3_D state C"),
                     col("dl", "decision L"), col("dc", "decision C (apply iff S3_D < 0)"), col("s", "sign differs"),
                     col("p", "predicted sign"), col("vb1", "VB1 state-L check max |diff|"), col("v", "validity")],
                    rows,
                    ["Report only; no amendment of the committed I19-1 TEST gate decisions (a755e18). The frozen-method "
                     "rule (I20-1) reads the v1-recipe S3_D on state-C records; the committed TU Delft primary "
                     "decisions are C0_refit (g v2-recipe S3_D +0.008 / +0.264). Realised DEV ln GMR(selected g v2, "
                     "C0_refit): 0.826 / 0.829 (I18-2, gain)."])


def t_c_validity(d: dict) -> rf.Table:
    rows = []
    for T in TEMPS:
        D, B = d[T]["dec"], d[T]["base"]
        rows.append({"t": f"{T} K", "c0": e(B["vc1"]["c0_max"]), "mb": e(B["vc1"]["mb_max"]),
                     "u": str(B["unit_tests"]["passed"]),
                     "r": ", ".join(f"{k} {len(v)}" for k, v in D["rows"].items()),
                     "i": f"{d[T]['inner']['complete']} ({d[T]['inner']['timing']['n_train']} trainings, "
                          f"{d[T]['inner']['timing']['n_refit']} refits)", "g": str(d[T]["pens"]["v4_ok"]),
                     "v": "valid" if D["validity"]["valid"] else "invalid: " + "; ".join(D["validity"]["fails"])})
    return rf.Table("08_table_i21_1_c_validity", "Part C validity (VC1-VC5) per temperature",
                    [col("t", "T"), col("c0", "VC1 C0_refit max |d|"), col("mb", "VC1 M_bias v1 max |d|"),
                     col("u", "VC2 unit tests"), col("r", "VC3 rows"), col("i", "VC3 inner selection"),
                     col("g", "VC4 grids"), col("v", "result", kind="text")], rows,
                    ["VC1: reruns against I5_1/lodo_<T>K/results.json (threshold 1e-9). VC5 (DEV only) is part of "
                     "the result column."])


def t_c_rules(d: dict) -> rf.Table:
    rows = []
    pr = d["prereg"]["predictions"]
    i17 = load(EXP / "I17_2/decision.json")
    for T in TEMPS:
        D = d[T]["dec"]
        p, s, g1 = D["primary_gmr_selected_vs_C0_refit"], D["secondary_gmr_selected_vs_v1"], D["gmr_v1_vs_C0_refit"]
        pc = D["prediction_check"]
        rows.append({"t": f"{T} K", "p": f"{ci(p)} {p['wins']}/18", "pl": D["outcome"],
                     "ppl": pr["C primary label"][str(T)] + (" (correct)" if pc["primary label"] else " (wrong)"),
                     "s": f"{ci(s)} {s['wins']}/18", "sl": D["secondary"],
                     "psl": "correct" if pc["secondary label"] else "wrong",
                     "sg": ("correct" if pc["ln GMR(selected, C0_refit)"] else "wrong") + " / "
                           + ("correct" if pc["ln GMR(selected, M_bias v1)"] else "wrong"),
                     "v1": f"{ci(g1)} {g1['wins']}/18"})
    p, s, g1 = (i17["primary_gmr_selected_vs_C0_refit"], i17["secondary_gmr_selected_vs_v1"],
                i17["gmr_v1_vs_C0_refit"])
    rows.append({"t": "300 K (I17-2, stored)", "p": f"{ci(p)} {p['wins']}/18", "pl": i17["outcome"], "ppl": "-",
                 "s": f"{ci(s)} {s['wins']}/18", "sl": i17["secondary"], "psl": "-", "sg": "-",
                 "v1": f"{ci(g1)} {g1['wins']}/18"})
    return rf.Table("09_table_i21_1_c_rules", "Part C pre-registered rules per temperature: selected g v2 on IHP SG13G2",
                    [col("t", "T"), col("p", "GMR(selected, C0_refit) lower/n"), col("pl", "primary label", kind="text"),
                     col("ppl", "predicted label", kind="text"), col("s", "GMR(selected, M_bias v1) lower/n"),
                     col("sl", "secondary label", kind="text"), col("psl", "secondary prediction"),
                     col("sg", "predicted signs (vs C0 / vs v1)"), col("v1", "GMR(M_bias v1, C0_refit) (reported)")],
                    rows, ["Device bootstrap 10,000 resamples, seed 0, 95% CI, 18 DEV DUTs. Primary: gain replicated "
                           "iff CI upper < 1 and 0 failed; worse iff CI lower > 1; else inconclusive. Secondary: "
                           "non-inferior iff CI upper <= 1.05."])


def t_c_arms(d: dict) -> rf.Table:
    rows = []
    for T in TEMPS:
        D = d[T]["dec"]
        for a in ("C0_refit", "M_bias_v1", "selected"):
            tb = D["table"][a]
            r = {"t": f"{T} K", "a": a, "m": f"{tb['all_device_mean']:.4f}", "c": f"{tb['combined']:.4f}",
                 "md": f"{tb['median']:.4f}"}
            r["cl"] = ci(tb["gmr_vs_C0_refit_cluster"]) if a != "C0_refit" else "-"
            for c in ("lv", "hv"):
                r[c] = f"{D['by_class'][c]['agg'][a]['all_device_mean']:.4f}"
            rows.append(r)
        cnt = D["lambda_star_counts"]
        rows.append({"t": f"{T} K", "a": "lambda* (folds)", "m": ", ".join(f"{k} V x{v}" for k, v in
                                                                            sorted(cnt.items(), key=lambda x: float(x[0]))),
                     "c": "", "md": "", "cl": "", "lv": "", "hv": ""})
    return rf.Table("10_table_i21_1_c_arms", "Part C arms per temperature (RRMS all-device / combined / median, LV / HV)",
                    [col("t", "T"), col("a", "arm", kind="text"), col("m", "all-device"), col("c", "combined"),
                     col("md", "median"), col("lv", "LV mean"), col("hv", "HV mean"),
                     col("cl", "GMR vs C0_refit, 12-cluster CI")], rows,
                    ["Clusters: class and W/L (I17-2 definition). lambda*: I17-2 nested selection per outer fold."])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_forest(d: dict, path: Path) -> list[str]:
    i17 = load(EXP / "I17_2/decision.json")
    rows = []
    for T in (233, 300, 343, 398):
        D = i17 if T == 300 else d[T]["dec"]
        rows.append((T, D["primary_gmr_selected_vs_C0_refit"], D["gmr_v1_vs_C0_refit"], D["secondary_gmr_selected_vs_v1"]))
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    lab = (("GMR(selected g v2, C0_refit)", AQUA), ("GMR(M_bias v1, C0_refit)", ORANGE),
           ("GMR(selected g v2, M_bias v1)", VIOLET))
    for i, (T, *gs) in enumerate(rows):
        for j, (g, (l, c)) in enumerate(zip(gs, lab)):
            y = i * 4 + j
            ax.plot([g["ci_low"], g["ci_high"]], [y, y], color=c, lw=2)
            ax.plot(g["ratio"], y, "o", color=c, ms=6, label=l if i == 0 else None)
            ax.text(g["ci_high"] + 0.01, y, f"{g['ratio']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}]",
                    va="center", fontsize=7.5, color=INK)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.axvline(1.05, color=GREY, lw=0.8, ls=":")
    ax.set_yticks([i * 4 + 1 for i in range(len(rows))])
    ax.set_yticklabels([f"{T} K" + (" (I17-2)" if T == 300 else " (I21-1)") for T, *_ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("geometric-mean RRMS ratio (95% device-bootstrap CI); dashed 1, dotted 1.05 (non-inferiority margin)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("IHP SG13G2, 18 DEV DUTs, LODO within class: selected g v2 vs C0_refit and M_bias v1 per temperature",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21 / f"ihp_{T}K/decision.json") for T in TEMPS] + [rel(EXP / "I17_2/decision.json")]


def fig_c_per_device(d: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, T in zip(axes, TEMPS):
        style(ax)
        R = d[T]["dec"]["rows"]
        tags = sorted(R["C0_refit"], key=lambda t: (t.split("_")[0] != "lv", t))
        y = np.arange(len(tags))
        ls = [math.log(R["selected"][t] / R["C0_refit"][t]) for t in tags]
        lv = [math.log(R["M_bias_v1"][t] / R["C0_refit"][t]) for t in tags]
        ax.barh(y - 0.2, ls, height=0.4, color=AQUA, label="ln(selected g v2 / C0_refit)")
        ax.barh(y + 0.2, lv, height=0.4, color=ORANGE, label="ln(M_bias v1 / C0_refit)")
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(tags, fontsize=7)
        ax.set_title(f"{T} K", fontsize=9.5, color=INK)
        ax.set_xlabel("per-DUT log RRMS ratio")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle("Part C per-DUT log ratios vs C0_refit (negative: lower RRMS than C0_refit)", fontsize=9.6, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21 / f"ihp_{T}K/decision.json") for T in TEMPS]


def fig_b(d: dict, path: Path) -> list[str]:
    from cryoml.explore.e1_correction import load_devices
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, (t, lab) in zip(axes, (("4", "4.13 K"), ("295", "295.0 K"))):
        style(ax)
        r = d["s3"]["per_temperature"][t]
        Lp = load(EXP / f"I19_1/s3/tud{t}_v1.json")["per_record"]
        recs = load_devices(EXP / f"I18_2/start/records_{t}K")
        grp = {x.tag: x.meta["group"] for x in recs}
        groups = sorted(set(grp.values()))
        for k, g in enumerate(groups):
            tags = [x for x in r["per_record"] if grp[x] == g]
            ax.scatter([Lp[x] for x in tags], [r["per_record"][x] for x in tags], s=18,
                       color=IV_COLORS[k % len(IV_COLORS)], label=g.replace("t40_", ""))
        lim = [min(min(Lp.values()), min(r["per_record"].values())) - 0.05,
               max(max(Lp.values()), max(r["per_record"].values())) + 0.05]
        ax.plot(lim, lim, color=GREY, lw=0.8, ls="--")
        ax.axhline(0, color=GREY, lw=0.6)
        ax.axvline(0, color=GREY, lw=0.6)
        ax.set_xlabel("per-record ln(RRMS corrected / RRMS start), state L records (I19-1)")
        ax.set_ylabel("same, state C records (I21-1)")
        ax.set_title(f"TU Delft {lab}: S3_D L {r['S3_D_state_L']:+.4f} -> C {r['S3_D_state_C']:+.4f}", fontsize=9.5,
                     color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Part B: v1-recipe S3_D inner fits (seed 0, 500 steps, leave-one-group-out) on the I6-1 state-L and the "
                 "I18-2 state-C starting records", fontsize=9.6, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21 / "s3tud/s3tud.json"), rel(EXP / "I19_1/s3/tud4_v1.json"), rel(EXP / "I19_1/s3/tud295_v1.json")]


def fig_a_gds(d: dict, ds: str, path: Path) -> list[str]:
    """Dense gds of the element vs the card for the first introduced curves."""
    from cryoml.explore import i11_1_deploy as m1
    from cryoml.explore import i21_1_close as m
    S = d[ds]["summary"]
    picks = S["gds_curves_introduced"][:4] or []
    if not picks:
        picks = [[next(iter(d[ds]["q12"]["devices"])), m.GDS_VGS[-1]]]
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks) + 1, 4.8), dpi=DPI, squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax, (tag, vg) in zip(axes[0], picks):
        style(ax)
        z = np.load(I21 / f"partA_{ds}/arrays/q12_{tag}.npz")
        k = list(m.GDS_VGS).index(vg)
        dc, dk = z[f"dense_card_gds{k}"], z[f"dense_elem_gds{k}"]
        o = np.argsort(np.abs(dc[:, 0]))
        x = np.abs(dc[o, 0])
        gc, ge = m1.central_diff(dc[o, 0], dc[o, 1]), m1.central_diff(dc[o, 0], dk[o, 1])
        sgn = 1.0 if tag.startswith("n") else 1.0
        ax.plot(x, sgn * gc, color=BLUE, lw=1.2, label="M2 card only")
        ax.plot(x, sgn * ge, color=AQUA, lw=1.2, label="M2 card + element (NGSpice OSDI)")
        ax.axhline(0, color=GREY, lw=0.7)
        neg = (np.sign(ge) != np.sign(gc)) & (x >= 0.05)
        ax.scatter(x[neg], ge[neg], s=6, color=RED, zorder=4, label="sign differs from card")
        ax.set_xlabel("|V(d,s)| (V)")
        ax.set_ylabel("dI_D / dV_D (S), 1 mV central differences")
        ax.set_title(f"{tag}\n|V(g,s)| = {vg:g} V", fontsize=8.5, color=INK, loc="left")
        ax.set_yscale("symlog", linthresh=max(1e-9, float(np.nanmax(np.abs(gc))) * 1e-3))
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Part A {DSS[ds]}: gds of the deployed element vs the card on curves with an introduced sign change "
                 f"({len(S['gds_curves_introduced'])} of {S['n_gds_curves']} curves; restriction)", fontsize=9.4,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21 / f"partA_{ds}/summary.json"), f"out/research/explore/I21_1/partA_{ds}/arrays/q12_<tag>.npz (not in git)"]


def fig_a_q3(d: dict, ds: str, path: Path) -> list[str]:
    arms = (("published", GREY, "--"), ("C0_refit", BLUE, "-"), ("M2_card", ORANGE, "-"), ("M2_full", AQUA, ":"))
    names = ("inv_A_vtc", "inv_B_vtc", "ro5_A", "ro5_B", "mirror_n", "mirror_p")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    Q = d[ds]["q3"]["rows"]
    for ax, name in zip(axes.flat, names):
        style(ax)
        for arm, c, ls in arms:
            p = I21 / f"partA_{ds}/arrays/q3_{name}_{arm}.npz"
            if not p.exists():
                continue
            rows = np.load(p)["rows"]
            if rows.size == 0:
                continue
            ok = Q[f"{name}|{arm}"]["ok"]
            lab = {"published": "published card", "C0_refit": "C0_refit card", "M2_card": "M2 card",
                   "M2_full": "M2 card + element"}[arm] + ("" if ok else " (fail)")
            if name.startswith("ro5"):
                ax.plot(rows[:, 0] * 1e9, rows[:, 1], color=c, ls=ls, lw=1.0, label=lab)
            elif name.startswith("mirror"):
                ax.plot(rows[:, 0], rows[:, 1] * 1e6, color=c, ls=ls, lw=1.1, label=lab)
            else:
                ax.plot(rows[:, 0], rows[:, 1], color=c, ls=ls, lw=1.1, label=lab)
        if name.startswith("ro5"):
            ax.set_xlabel("t (ns)")
            ax.set_ylabel("V(o1) (V)")
            st = Q[f"{name}|M2_full"].get("freq_hz")
            ax.set_xlim(0, 3 if name == "ro5_A" else 15)
        elif name.startswith("mirror"):
            ax.set_xlabel("V_out (V)")
            ax.set_ylabel("|I_out| (uA), I_ref 10 uA")
        else:
            ax.set_xlabel("V_in (V)")
            ax.set_ylabel("V_out (V)")
        ax.set_title(name, fontsize=9.5, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Part A Q3 circuits, {DSS[ds]}: sky130 decks at 1.8 V on the nearest NIST DEV devices "
                 f"(devices: {', '.join(f'{k} -> {v}' for k, v in d[ds]['q3']['devices'].items())})", fontsize=8.6,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21 / f"partA_{ds}/q3.json"), f"out/research/explore/I21_1/partA_{ds}/arrays/q3_*.npz (not in git)"]


def a_resim(d: dict, ds: str, tag: str) -> dict:
    """Measured-grid NGSpice of the stock card, the M2 card (card only) and
    the M2 card with the OSDI element; RRMS checked against the stored rows."""
    os.environ.pop("CRYOML_I21_1_SMOKE", None)
    import i21_1_close as drv
    from cryoml.explore import i8_1_nist as b8
    from cryoml.explore.e1_correction import score_device
    dev, bi = drv.d20.devices(ds)[tag], drv.d20.bins(ds)[tag]
    params = load(I21 / f"partA_{ds}/cards_M2.json")[tag]
    corr = drv.full_corr(ds)
    prob = drv.d20.problem(ds, tag, "C")
    stock = prob.simulate_params(None)
    card = drv.measured_run(ds, tag, params, "card")
    elem = drv.measured_run(ds, tag, params, "elem")
    rc = b8.to_record(dev, bi, card["sims"], params)
    re_ = b8.to_record(dev, bi, elem["sims"], params)
    sims = {"stock": stock, "M2_card": card["sims"], "M2_python": corr.apply(rc), "M2_elem": elem["sims"]}
    sc = {"stock": float(np.mean(prob.curve_rrms(stock))),
          "M2_card": float(score_device(rc, card["sims"])["rrms"]),
          "M2_python": float(score_device(rc, sims["M2_python"])["rrms"]),
          "M2_elem": float(score_device(re_, elem["sims"])["rrms"])}
    A = d[ds]["accuracy"]["rows"][tag]
    check(sc["M2_python"], d[ds]["m2"]["rows"][tag]["rrms"], f"{ds} {tag} M2 python")
    check(sc["M2_elem"], A["rrms_native"], f"{ds} {tag} native")
    check(sc["M2_card"], A["rrms_card_only"], f"{ds} {tag} card only")
    return {"rec": rc, "sims": sims, "score": sc, "dev": dev, "bin": bi}


def fig_a_iv(d: dict, ds: str, tag: str, why: str, path: Path) -> list[str]:
    r = a_resim(d, ds, tag)
    rec, sc = r["rec"], r["score"]
    arms = (("stock", GREY, (0, (4, 2)), 1.0), ("M2_card", BLUE, "-", 1.0), ("M2_python", ORANGE, "--", 1.3),
            ("M2_elem", AQUA, ":", 1.6))
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for row, kind in enumerate(("idvg", "idvd")):
        idx = [i for i, c in enumerate(rec.curves) if c.kind == kind]
        for colm, scale in enumerate(("log", "linear") if kind == "idvg" else ("linear", "log")):
            ax = axs[row, colm]
            style(ax)
            for k, i in enumerate(idx):
                c = rec.curves[i]
                color = IV_COLORS[k % len(IV_COLORS)]
                x = np.abs(np.asarray(c.vg if kind == "idvg" else c.vd, float))
                meas = np.abs(np.asarray(c.target, float))
                step = 3 if len(x) > 100 else 1
                ax.scatter(x[::step], meas[::step], s=9, color=color, alpha=0.9 if c.included else 0.3, zorder=3, lw=0)
                for a, colr, ls, lw in arms:
                    ax.plot(x, np.abs(np.asarray(r["sims"][a][i], float)), color=colr, ls=ls, lw=lw,
                            alpha=0.9 if c.included else 0.35)
                ax.plot([], [], color=color, marker="o", ms=4, ls="", label=c.key + ("" if c.included else " (not in RRMS)"))
            if scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(bottom=1e-13)
            ax.set_xlabel("|Vgs| (V)" if kind == "idvg" else "|Vds| (V)", fontsize=9)
            ax.set_ylabel("|Id| (A)", fontsize=9)
            ax.set_title(f"{'Id-Vg' if kind == 'idvg' else 'Id-Vd'}, {scale} scale", fontsize=9.5, color=INK, loc="left")
            if colm == 0:
                ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)
    lab = {"stock": "NGSpice, stock card (published sky130 card of the dataset, no fit)",
           "M2_card": "NGSpice, M2 card only (no correction)",
           "M2_python": "NGSpice M2 card x exp(g) in Python (as scored in training)",
           "M2_elem": "NGSpice M2 card + OSDI element (deployed form, ngspice-41 OSDI)"}
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=4,
                      label="measured 4 K (colour per curve; faded: not in RRMS; every 3rd point on 181-point sweeps)")]
    handles += [Line2D([], [], color=c, ls=ls, lw=lw, label=f"{lab[a]}: RRMS {sc[a]:.4f}") for a, c, ls, lw in arms]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.8, frameon=False)
    dv = r["dev"]
    fig.suptitle(f"I21-1 Part A {DSS[ds]} {tag} ({dv.polarity}, W {dv.W_um:g} um, L {dv.L_um:g} um, bin "
                 f"{r['bin']['bin']}): {why}\nM2 card = state-C starting card refit with the full-data g v2 "
                 "(lambda 0 V) in the loop; every RRMS equals the stored row", fontsize=9.4, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I21 / f"partA_{ds}/cards_M2.json"), rel(I21 / f"partA_{ds}/accuracy.json"),
            rel(I21 / f"partA_{ds}/m2.json")]


def c_resim(d: dict, T: int, tag: str) -> dict:
    os.environ["CRYOML_I21_1_IHP_T"] = str(T)
    import i21_1_close as drv
    drv.bind_ihp(T)
    from cryoml.explore import i5_1_ihp_real as m5
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i3_2_bias import correction_from_json
    from cryoml.explore.i12_1_symm import SymCorrection
    B = I21 / f"ihp_{T}K"
    D = d[T]["dec"]
    cards = load(B / "base/heldout_cards.json")
    bc = load(B / "base/corrections.json")
    sel = d[T]["sel"]
    dev = drv.d17.devices()[tag]
    prob = m5.MeasProblem(dev, T, m5.base_box(dev))
    stock = m5.stock_params(dev, m5.BASE_SET)
    sims, sc = {}, {}
    s0 = prob.simulate_params(stock)
    rec0 = m5.to_record(dev, T, s0, stock)
    sims["stock"], sc["stock"] = s0, score_device(rec0, s0)["rrms"]
    c0 = dict(cards[f"C0_refit|{tag}"])
    s = prob.simulate_params(c0)
    sims["C0_refit"], sc["C0_refit"] = s, score_device(m5.to_record(dev, T, s, c0), s)["rrms"]
    check(sc["C0_refit"], D["rows"]["C0_refit"][tag], f"{T} {tag} C0_refit")
    p1 = dict(cards[f"M_bias|{tag}"])
    r1 = m5.to_record(dev, T, prob.simulate_params(p1), p1)
    g1 = correction_from_json(bc[f"M_bias|{tag}"]).apply(r1)
    sims["M_bias_v1"], sc["M_bias_v1"] = g1, score_device(r1, g1)["rrms"]
    check(sc["M_bias_v1"], D["rows"]["M_bias_v1"][tag], f"{T} {tag} v1")
    ps = dict(load(B / "sel/heldout_cards.json")[tag])
    corr = SymCorrection.from_json(load(B / "sel/corrections.json")[sel["names"][tag]])
    rs = m5.to_record(dev, T, prob.simulate_params(ps), ps)
    gs = corr.apply(rs)
    sims["selected"], sc["selected"] = gs, score_device(rs, gs)["rrms"]
    check(sc["selected"], D["rows"]["selected"][tag], f"{T} {tag} selected")
    return {"rec": rec0, "sims": sims, "score": sc, "lam": sel["lambda_star"][tag]}


def fig_c_iv(d: dict, T: int, tag: str, why: str, path: Path) -> list[str]:
    r = c_resim(d, T, tag)
    rec = r["rec"]
    lines = (("stock", GREY, "--"), ("C0_refit", BLUE, "-"), ("M_bias_v1", ORANGE, "-"), ("selected", AQUA, "-"))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, kind in zip(axes, ("idvd", "idvg")):
        style(ax)
        for i, c in enumerate(rec.curves):
            if c.kind != kind:
                continue
            xv = np.abs(c.vd if kind == "idvd" else c.vg)
            alpha = 1.0 if c.included else 0.35
            meas = np.abs(c.target)
            s = slice(c.start, None)
            ax.scatter(xv[s], meas[s], s=7, color=INK, zorder=4, alpha=alpha, lw=0)
            if c.start:
                ax.scatter(xv[:c.start], meas[:c.start], s=7, facecolor="none", edgecolor=GREY, lw=0.5, zorder=4,
                           alpha=alpha)
            for a, colr, ls in lines:
                ax.plot(xv, np.abs(r["sims"][a][i]), color=colr, ls=ls, lw=1.1, alpha=alpha, zorder=3)
        if kind == "idvg":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-13)
            ax.set_xlabel("|Vgs| (V)")
            ax.set_title("Id-Vg (log scale)", fontsize=9.5)
        else:
            ax.set_xlabel("|Vds| (V)")
            ax.set_title("Id-Vd", fontsize=9.5)
        ax.set_ylabel("|Id| (A)")
    sc = r["score"]
    handles = [
        Line2D([], [], marker="o", ls="", color=INK, ms=4,
               label=f"measured {T} K (filled: scored points; hollow: trimmed before the metric; faded: excluded curve)"),
        Line2D([], [], color=GREY, ls="--", lw=1.2, label=f"NGSpice, stock PDK card (no fit): RRMS {sc['stock']:.4f}"),
        Line2D([], [], color=BLUE, lw=1.2,
               label=f"NGSpice re-simulated, C0_refit held-out card (no correction): RRMS {sc['C0_refit']:.4f}"),
        Line2D([], [], color=ORANGE, lw=1.2,
               label=f"NGSpice re-simulated, M_bias v1 held-out card x exp(fold g): RRMS {sc['M_bias_v1']:.4f}"),
        Line2D([], [], color=AQUA, lw=1.2,
               label=f"NGSpice re-simulated, selected g v2 held-out card x exp(fold g) (lambda* {r['lam']:g} V): RRMS "
                     f"{sc['selected']:.4f}")]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag} (IHP SG13G2 DEV, {T} K; {why}): held-out cards (LODO within class) re-simulated in ngspice-46 "
                 "+ PSP103 OSDI; g applied in Python (no PSP103 element).\nRRMS equals the stored rows.",
                 fontsize=9.2, color=INK)
    fig.tight_layout(rect=(0, 0.2, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    B = I21 / f"ihp_{T}K"
    return [rel(B / "base/heldout_cards.json"), rel(B / "sel/heldout_cards.json"), rel(B / "sel/corrections.json"),
            "I5_1/data/ihp_meas.npz (not in git)"]


def c_iv_tags(d: dict, T: int) -> list[tuple[str, str]]:
    R = d[T]["dec"]["rows"]
    lr = {t: math.log(R["selected"][t] / R["C0_refit"][t]) for t in R["C0_refit"]}
    lo, hi = min(lr, key=lr.get), max(lr, key=lr.get)
    return [(lo, f"lowest ln(selected / C0_refit) {lr[lo]:+.3f}"), (hi, f"highest ln(selected / C0_refit) {lr[hi]:+.3f}")]


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = load_all()
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    pa = [rel(I21 / f"partA_{ds}/summary.json") for ds in ("nist4", "nistrt")]
    pc = [rel(I21 / f"ihp_{T}K/decision.json") for T in TEMPS]
    add_table(t_setup(d), "I21-1 setup (pre-registered)", [rel(I21 / "prereg.json")])
    add_table(t_a_validity(d), "Part A validity", pa + [rel(I21 / "partA_nist4/m2.json"), rel(I21 / "partA_nistrt/m2.json")])
    add_table(t_a_checks(d), "Part A core checks and restrictions vs thresholds, labels and predictions", pa)
    add_table(t_a_q3(d), "Part A Q3 circuit metrics per arm and circuit runtime ratios",
              [rel(I21 / "partA_nist4/q3.json"), rel(I21 / "partA_nistrt/q3.json")])
    add_table(t_a_gds(d), "Part A gds curves with an introduced sign change per |V(g,s)|",
              [rel(I21 / "partA_nist4/q12.json"), rel(I21 / "partA_nistrt/q12.json")])
    add_table(t_a_acc(d), "Part A element-deck accuracy of the M2 pairs",
              [rel(I21 / "partA_nist4/accuracy.json"), rel(I21 / "partA_nistrt/accuracy.json")])
    add_table(t_b(d), "Part B TU Delft v1-recipe S3_D, state L vs state C", [rel(I21 / "s3tud/s3tud.json")])
    add_table(t_c_validity(d), "Part C validity per temperature", pc)
    add_table(t_c_rules(d), "Part C primary / secondary labels per temperature, predictions", pc + [rel(EXP / "I17_2/decision.json")])
    add_table(t_c_arms(d), "Part C arm aggregates, LV / HV, cluster CIs, lambda* counts", pc)
    k = len(items) + 1
    figs = [("forest", fig_forest, "Part C forest plot: GMR(selected g v2, C0_refit), GMR(M_bias v1, C0_refit), "
                                   "GMR(selected g v2, M_bias v1) at 233 / 300 / 343 / 398 K"),
            ("c_per_device", fig_c_per_device, "Part C per-DUT log ratios vs C0_refit per temperature"),
            ("b_s3", fig_b, "Part B per-record S3 log ratios, state L vs state C, per geometry group")]
    for name, fn, desc in figs:
        png = f"{k:02d}_i21_1_{name}.png"
        items.append((png, desc, fn(d, folder / png)))
        k += 1
    for ds in ("nist4", "nistrt"):
        png = f"{k:02d}_i21_1_a_gds_{ds}.png"
        items.append((png, f"Part A {DSS[ds]}: dense gds of the element vs the card on curves with an introduced sign "
                           "change", fig_a_gds(d, ds, folder / png)))
        k += 1
        png = f"{k:02d}_i21_1_a_q3_{ds}.png"
        items.append((png, f"Part A {DSS[ds]}: Q3 circuit outputs per arm", fig_a_q3(d, ds, folder / png)))
        k += 1
    for ds in ("nist4", "nistrt"):
        ct = d[ds]["q3"]["devices"]
        for role in ("nmos_L1_W1p6", "pmos_L2_W5"):
            tag = ct[role]
            png = f"{k:02d}_i21_1_a_iv_{ds}_{tag}.png"
            why = f"Q3 circuit device for the sky130 role {role}"
            items.append((png, f"Part A I-V of {tag} ({DSS[ds]}): measured 4 K points, NGSpice stock card, M2 card only, "
                               "M2 card x exp(g) (Python) and M2 card + OSDI element (NGSpice)",
                          fig_a_iv(d, ds, tag, why, folder / png)))
            k += 1
    for T in TEMPS:
        for tag, why in c_iv_tags(d, T):
            png = f"{k:02d}_i21_1_c_iv_{T}K_{tag}.png"
            items.append((png, f"Part C I-V of {tag} at {T} K ({why}): measured points, NGSpice stock card, re-simulated "
                               "C0_refit, M_bias v1 and selected g v2 held-out cards",
                          fig_c_iv(d, T, tag, why, folder / png)))
            k += 1
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_explore_i21_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    lines = []
    for ds in ("nist4", "nistrt"):
        S = d[ds]["summary"]
        lines.append(f"Part A {DSS[ds]}: {S['label']}; core failures {', '.join(S['decision']['core_fail']) or 'none'}; "
                     f"restrictions {', '.join(S['decision']['restrictions']) or 'none'}; gds curves "
                     f"{len(S['gds_curves_introduced'])} of {S['n_gds_curves']}.")
    for t, lab in (("4", "4.13 K"), ("295", "295.0 K")):
        r = d["s3"]["per_temperature"][t]
        lines.append(f"Part B TU Delft {lab}: v1-recipe S3_D state L {r['S3_D_state_L']:+.4f} -> state C "
                     f"{r['S3_D_state_C']:+.4f} (sign differs: {r['sign_differs']}; decision {r['decision_state_C']}).")
    for T in TEMPS:
        D = d[T]["dec"]
        lines.append(f"Part C IHP {T} K: GMR(selected g v2, C0_refit) {ci(D['primary_gmr_selected_vs_C0_refit'])} "
                     f"({D['outcome']}); GMR(selected, M_bias v1) {ci(D['secondary_gmr_selected_vs_v1'])} "
                     f"({D['secondary']}).")
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. NGSpice runs: Part A I-V re-simulation (stock, M2 card, M2 card + OSDI "
              f"element) and Part C I-V re-simulation (stock, C0_refit, M_bias v1, selected); every re-simulated RRMS "
              f"checked against the stored rows; {d['n_checked']} stored values recomputed.", "",
              "## Contents", "",
              "- I21-1: open items that need no user decision. Pre-registration "
              f"`out/research/explore/I21_1/prereg.md` ({COMMITS['prereg']}). DEV only; no amendment of a755e18.",
              *[f"- {x}" for x in lines], "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = ["I21-1 open items without a user decision: NIST Part B, TU Delft S3_D in state C, IHP 233 / 343 / 398 K "
               "selected g v2."] + lines
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
