#!/usr/bin/env python3
"""Report batch for I17-1: re-registered deployment test (Part B, I16-1 rules
except the Q1 V(d,s) = 0 check, evaluated at tight tolerance only; serpentine
Q1 grids) of the I15-2 selected bias-only correction (symmetric g, penalty
lambda selected per fold; deployed full-data model lambda* = 1 V,
``full|sym_e1_l1``); sky130 77 K, 18 devices.

Reads stored files:

* ``out/research/explore/I17_1/{prereg.json, decision.json}``
* ``I17_1/validity/{validity,unit_tests}.json``, ``I17_1/accuracy/accuracy.json``
* ``I17_1/deploy_sel/{compile,refit,q12,q3,summary,restrictions,runtime}.json``
  and ``arrays/q12_<tag>.npz`` (dense gds sweeps)
* ``I15_1/deploy_{v2,v1}/{q12,q3,summary}.json``, ``I15_1/deploy_v2/arrays``,
  ``I15_1/accuracy/accuracy.json`` (I15-1 g v2, lambda 10 V, and v1 reference)
* ``I15_2/decision.json`` (selected rows, lambda*), I13-1 state-C rows

NGSpice: the I-V figures re-simulate the published card (upstream committed
sweeps), the I13-1 C0_refit card, the v1 and g v2 held-out cards (element
deck with the I15-1 fold modules) and the selected held-out cards (card x
exp(g) in Python and element deck with the I17-1 fold module); every RRMS is
checked against the stored rows.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i17_1.py --date 2026-10-22
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, RED, VIOLET, AQUA, DPI,  # noqa: E402
                                   EXP, ROOT, load, rel, style)
from cryoml.explore.e1_correction import score_device  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i17_1"
I17 = EXP / "I17_1"
I15 = EXP / "I15_1"
I152 = EXP / "I15_2"
I13 = EXP / "I13_1"
F = {"prereg": I17 / "prereg.json", "decision": I17 / "decision.json",
     "val": I17 / "validity/validity.json", "unit": I17 / "validity/unit_tests.json",
     "acc": I17 / "accuracy/accuracy.json", "lodo": I13 / "lodo/results.json",
     "all18": I13 / "all18/results.json", "i152": I152 / "decision.json",
     "acc15": I15 / "accuracy/accuracy.json", "i16_vds0": EXP / "I16_1/diagnosis/vds0.json",
     "i16_dec": EXP / "I16_1/decision.json"}
for _k in ("compile", "refit", "q12", "q3", "summary", "restrictions", "runtime"):
    F[f"sel_{_k}"] = I17 / f"deploy_sel/{_k}.json"
for _a in ("v2", "v1"):
    for _k in ("q12", "q3", "summary"):
        F[f"{_a}_{_k}"] = I15 / f"deploy_{_a}/{_k}.json"
COMMITS = {"prereg": "3cfc9a7", "code": "86fcbf8"}
SEL = AQUA
V2C = VIOLET


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def e(x: float) -> str:
    return f"{x:.2e}" if (x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4)) else f"{x:.4g}"


def agg_s(a: dict) -> str:
    return f"{a['all_device_mean']:.5f} / {a['combined']:.5f} / {a['median']:.5f}"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
CHECKS = (  # key, label, threshold text, pass(value), kind
    ("q1_tight_viol", "Q1 equivalence: points violating |dI| <= 1e-6 |I_py| + 1e-13 A (tight, both benches)", "0",
     lambda v: v == 0, "core"),
    ("q1_bad", "Q1 runs: convergence messages + non-finite values + failed runs (element decks)", "0",
     lambda v: v == 0, "core"),
    ("q1_vds0_max", "Q1 max |I_elem(Vds = 0)| (A), tight tolerance only (I17-1 rule)", "<= 1e-12",
     lambda v: v <= 1e-12, "core"),
    ("g_text_max", "Verilog-A g vs Python, max |dg|", "<= 1e-9", lambda v: v <= 1e-9, "core"),
    ("g_sym_text_max", "Verilog-A g, exchanged vs forward terminals, max |dg|", "<= 1e-12",
     lambda v: v <= 1e-12, "core"),
    ("swap_asym_max", "exchange asymmetry, element-introduced max |d ln|", "<= 1e-3", lambda v: v <= 1e-3, "core"),
    ("q2_ac_max", "Q2 gm / gds vs element identity, max normalized difference", "<= 1e-3",
     lambda v: v <= 1e-3, "core"),
    ("cap_max", "Q2 capacitances (Y_dd, Y_dg, Y_gg), max relative error", "<= 1e-3", lambda v: v <= 1e-3, "core"),
    ("q2_gm_sign", "Q2 gm curves with an introduced sign change in strong inversion (of 54)", "0",
     lambda v: v == 0, "core"),
    ("q3_fail", "Q3 circuits failing only with the element", "0", lambda v: v == 0, "core"),
    ("q1_default_max", "default tolerance, max relative difference (|I| > 1 nA)", "<= 1e-3 (restriction)",
     lambda v: v <= 1e-3, "restriction"),
    ("q2_gds_sign", "gds curves with an introduced sign change (of 90)", "0 (restriction)", lambda v: v == 0,
     "restriction"),
    ("runtime_ratio_max", "wall-time ratio element / card only, max (grid and circuits)", "<= 2 (restriction)",
     lambda v: v <= 2.0, "restriction"),
)


def t_partb(d: dict) -> rf.Table:
    rows = []
    tight = {a: d["i16_vds0"]["arms"][a]["summary"]["tight"]["elem_max"] for a in ("v2_I15_1", "v1_I15_1")}
    for key, lab, thr, ok, kind in CHECKS:
        a = d["sel_summary"]["values_I17_1_rule"][key]
        b, c = d["v2_summary"]["values"][key], d["v1_summary"]["values"][key]
        if key == "q1_vds0_max":
            b, c = tight["v2_I15_1"], tight["v1_I15_1"]
        fm = (lambda x: f"{x:.4e}") if key == "q1_vds0_max" else e
        rows.append({"chk": lab, "kind": kind, "thr": thr, "s": fm(a), "ps": "pass" if ok(a) else "FAIL",
                     "v2": f"{fm(b)} ({'pass' if ok(b) else 'FAIL'})",
                     "v1": f"{fm(c)} ({'pass' if ok(c) else 'FAIL'})"})
    vd = d["sel_summary"]["vds0"]["summary"]
    rows.append({"chk": "reported, no rule: Q1 max |I_elem(Vds = 0)| (A), default tolerance", "kind": "reported",
                 "thr": "(I16-1: <= 1e-12)", "s": f"{vd['default']['elem_max']:.4e}",
                 "ps": f"{vd['default']['n_above']} of {vd['default']['n_points']} points > 1e-12 A",
                 "v2": f"{d['i16_vds0']['arms']['v2_I15_1']['summary']['default']['elem_max']:.4e}",
                 "v1": f"{d['i16_vds0']['arms']['v1_I15_1']['summary']['default']['elem_max']:.4e}"})
    rows.append({"chk": "label (I17-1 rule)", "kind": "", "thr": "", "s": d["sel_summary"]["decision"]["deploy"],
                 "ps": "", "v2": d["v2_summary"]["decision"]["deploy"], "v1": d["v1_summary"]["decision"]["deploy"]})
    rows.append({"chk": "label under the unchanged I16-1 rule (max of default and tight), same run", "kind": "",
                 "thr": "", "s": d["sel_summary"]["decision_I16_1_rule"]["deploy"], "ps": "", "v2": "", "v1": ""})
    s = d["sel_summary"]
    notes = ["The Q1 V(d,s) = 0 check was re-registered after the observed I16-1 failure (tight tolerance only, "
             "threshold 1e-12 A kept); the I16-1 outcome stays 'not deployable'. I15-1 columns of that row: tight-"
             "tolerance values from I16_1/diagnosis/vds0.json.",
             "selected: I15-2 full-data correction sym_e1_l1 (symmetric inputs, EPS 0.01 V, penalty lambda 1 V = "
             "full-data lambda*) with its M2 cards (refit from the starting card with g in the loop, state C, "
             f"{s['refit_in_sample']['n_failed']} failed). I15-1 columns (not rerun): g v2 (lambda 10 V) and the "
             "M_bias v1 reference arm. Same element (channel-current replica, gmin VCCS, module "
             "I(d, s) <+ (exp(g) - 1) I(sense)), serpentine Q1 grids.",
             "Classification (prereg, kept from I15-1): an introduced gds sign change is a restriction; an introduced "
             "gm sign change in strong inversion is a core failure.",
             f"Q2 with the I12-1 formula exp(g) Re Y_card + I_card exp(g) dg/dV on the same runs: max "
             f"{s['q2_err_i12_formula_max']:.3g}.",
             f"Runtime: DC grid {min(s['runtime_grid'].values()):.2f}-{max(s['runtime_grid'].values()):.2f}; "
             "circuits " + ", ".join(f"{k} {v:.2f}" for k, v in s["runtime_circuits"].items())
             + "; " + (", ".join(s["runtime_circuits_excluded"]) or "none") + " excluded (I12-1 amendment A1)."]
    return rf.Table("01_table_i17_1_part_b", "I17-1 Part B checks vs pre-registered thresholds: selected correction "
                    "(outcome), I15-1 g v2 and M_bias v1 values",
                    [col("chk", "check", kind="text"), col("kind", "type", kind="text"), col("thr", "threshold"),
                     col("s", "selected value"), col("ps", "selected"), col("v2", "I15-1 g v2 (lambda 10 V)"),
                     col("v1", "I15-1 v1 reference")], rows, notes)


def t_validity(d: dict) -> rf.Table:
    V, U = d["val"], d["unit"]
    dec = d["decision"]
    v4 = V["v4"]
    rows = [
        {"c": "V1 card-only library deck x exp(fold g) vs I15-2 selected rows (18 rows)", "t": "<= 1e-9",
         "v": e(V["v1_max_rrms_diff"]), "p": "pass" if V["v1_max_rrms_diff"] <= 1e-9 and V["v1_n_rows"] == 18
         else "FAIL"},
        {"c": "V2 unit tests (" + ", ".join(x.split(".")[-1] for x in U["files"]) + ")", "t": "all pass",
         "v": U["tail"][-1] if U["tail"] else "", "p": "pass" if U["passed"] else "FAIL"},
        {"c": "V3 element-deck accuracy rows / failed M2 refits", "t": "18 / 0",
         "v": f"{len(d['acc']['native'])} / {d['sel_summary']['refit_in_sample']['n_failed']}",
         "p": "pass" if (len(d["acc"]["native"]) == 18 and d["sel_summary"]["refit_in_sample"]["n_failed"] == 0)
         else "FAIL"},
        {"c": "V4 deployed g: inputs / EPS / lambda; I15-2 full-data lambda*", "t": "sym_v2 / 0.01 / 1; 1",
         "v": f"{v4['corr_inputs']} / {v4['corr_eps']} / {v4['results_lambda']}; {v4['lambda_star_full']}",
         "p": "pass" if v4["ok"] else "FAIL"},
    ]
    notes = ["Validity rule (prereg): any V fail -> invalid. Result: " + ("valid." if dec["validity"]["valid"]
                                                                          else "invalid.")]
    return rf.Table("02_table_i17_1_validity", "I17-1 validity",
                    [col("c", "check", kind="text"), col("t", "criterion"), col("v", "value"), col("p", "result")],
                    rows, notes)


def t_devices(d: dict) -> rf.Table:
    Q = d["sel_q12"]["devices"]
    R = d["sel_refit"]
    rows = []
    for t in sorted(Q):
        r = Q[t]
        msgs = sum(len(v) for k, v in r["msgs"].items() if k.endswith("_elem"))
        n = R["native"][t]
        rows.append({"dev": t, "q1": f"{max(r['q1_tight_fwd_max_ratio'], r['q1_tight_swap_max_ratio']):.2f}",
                     "as": e(r["swap_asym_element_max"]), "q2": e(max(r["q2_err_gm"], r["q2_err_gds"])),
                     "cap": e(r["cap_max"]), "def": e(r["q1_default_max_rel_1nA"]),
                     "gm": str(sum(r["q2_gm_curves_introduced"])), "gds": str(sum(r["q2_gds_curves_introduced"])),
                     "msg": str(msgs), "rt": f"{d['sel_summary']['runtime_grid'][t]:.2f}",
                     "m2": f"{n['rrms_python']:.4f} / {n['rrms_native']:.4f}"})
    notes = ["q1: max |dI| / (1e-6 |I_py| + 1e-13 A), tight, forward and exchanged (pass <= 1); asym: element-"
             "introduced |d ln(I_swap / I_fwd)|; q2: max normalized gm / gds difference vs the element identity; cap: "
             "capacitance error; def: default-tolerance relative difference (|I| > 1 nA); gm: gm curves with an "
             "introduced sign change in strong inversion (of 3, core); gds: gds curves with an introduced sign change "
             "(of 5, restriction); msg: convergence lines in element runs; rt: DC grid wall-time ratio; M2 RRMS "
             "in-sample: Python (I_card exp(g)) / SPICE-native element."]
    return rf.Table("03_table_i17_1_devices", "I17-1 Part B per device, selected correction with its M2 card",
                    [col("dev", "device", kind="text"), col("q1", "Q1 ratio"), col("as", "asym"), col("q2", "Q2"),
                     col("cap", "cap"), col("def", "default tol"), col("gm", "gm sign"), col("gds", "gds sign"),
                     col("msg", "msgs"), col("rt", "runtime"), col("m2", "M2 RRMS py / native")], rows, notes)


def t_accuracy(d: dict) -> rf.Table:
    A = d["acc"]
    R = A["summary_reference"]
    G = A["gmr_native"]
    rows = [
        {"arm": "selected (I15-2 rows; Python, I_card exp(g))", "v": agg_s(A["summary_stored"]), "x": ""},
        {"arm": "selected, element deck (fold OSDI module + held-out card)", "v": agg_s(A["summary_native"]),
         "x": f"max |native - stored| {A['max_abs_native_minus_stored']:.1e}; Python rerun max |d| "
              f"{A['max_abs_python_rerun_minus_stored']:.1e}"},
        {"arm": "M_bias v1 (I13-1 rows)", "v": agg_s(R["v1_stored"]), "x": ""},
        {"arm": "M_bias v1, element deck (I15-1)", "v": agg_s(R["v1_element_I15_1"]), "x": ""},
        {"arm": "g v2 lambda 10 V, element deck (I15-1)", "v": agg_s(R["v2_element_I15_1"]), "x": ""},
        {"arm": "C0_refit (no correction)", "v": agg_s(R["C0_refit"]), "x": ""},
        {"arm": "GMR(selected element, v1 rows)", "v": ci(G["sel_native_vs_v1_stored"]),
         "x": f"{G['sel_native_vs_v1_stored']['wins']}/18 lower"},
        {"arm": "GMR(selected element, v1 element)", "v": ci(G["sel_native_vs_v1_element_I15_1"]),
         "x": f"{G['sel_native_vs_v1_element_I15_1']['wins']}/18 lower"},
        {"arm": "GMR(selected element, g v2 element)", "v": ci(G["sel_native_vs_v2_element_I15_1"]),
         "x": f"{G['sel_native_vs_v2_element_I15_1']['wins']}/18 lower"},
        {"arm": "GMR(selected element, C0_refit)", "v": ci(G["sel_native_vs_C0_refit"]),
         "x": f"{G['sel_native_vs_C0_refit']['wins']}/18 lower"},
        {"arm": "I15-2 primary GMR(selected rows, v1 rows) (registered, fixed)",
         "v": ci(d["i152"]["primary_gmr_selected_vs_v1"]),
         "x": f"{d['i152']['primary_gmr_selected_vs_v1']['wins']}/18 lower; {d['i152']['outcome']}"},
    ]
    notes = ["Values: all-device / combined / median (strict paper RRMS, published-card curve inclusion). Element "
             "deck: per device h the fold correction of sym_e1_l{lambda*_h} (lambda* 1 V on 10 folds, 2 V on 6, 0 V on "
             "pmos_L4_W7, 10 V on pmos_L8_W5) as an OSDI element with its held-out card, "
             "I = I_card + (exp(g) - 1) I_rep.",
             f"Element vs channel model I_card + (exp(g) - 1) I_rep from the card + replica run: max |dI| / (1e-3 |I| "
             f"+ 1e-12 A) {A['max_native_vs_channel_ratio']:.2f}; convergence lines {A['n_msgs']}.",
             "GMR: device bootstrap 10,000, 95% CI. The accuracy result of the selected variant is the I15-2 primary "
             "(not rerun); the element-deck values are reported without a rule."]
    return rf.Table("04_table_i17_1_accuracy", "I17-1 element-deck accuracy of the selected held-out pairs vs the "
                    "I15-2 rows, 18 devices",
                    [col("arm", "arm / contrast", kind="text"), col("v", "value"), col("x", "detail", kind="text")],
                    rows, notes)


def t_gds(d: dict) -> rf.Table:
    S = d["sel_restrictions"]["sel"]["per_device"]
    Rv = d["sel_restrictions"].get("ref_v2_I15_1", {}).get("per_device", {})
    rows = []
    for t in sorted(S):
        r = {"dev": t}
        for k, c in enumerate(S[t]["gds"]):
            txt = f"{c['n_neg_elem']} / {c['n_neg_python']}"
            r[f"g{k}"] = ("I: " if c["introduced"] else "") + txt
        rv = Rv.get(t)
        r["v2"] = (f"{rv['n_gds_introduced']} ({sum(c['n_neg_elem'] for c in rv['gds'])})" if rv else "-")
        r["n"] = str(S[t]["n_gds_introduced"])
        rows.append(r)
    tot = d["sel_restrictions"]["sel"]["totals"]
    totv = d["sel_restrictions"].get("ref_v2_I15_1", {}).get("totals", {})
    vg = [f"{c['vgs']:+.2f}" for c in S[sorted(S)[0]]["gds"]]
    notes = [f"Per V(g,s) (nMOS sign shown for the first device; pMOS negative): points of the 1 mV dense sweep with "
             f"dI/dV(d,s) < 0 at |V(d,s)| >= 0.05 V, element / Python I_card exp(g) of the card sweep; 'I': "
             f"introduced sign change by the pre-registered curve rule (card has no negative value below -1e-12 S, "
             f"element has; AC or dense). The card alone has {tot['n_gds_neg_points_card']} negative points.",
             f"Totals selected: {tot['n_gds_curves_introduced']} of 90 curves introduced, {tot['n_gds_neg_points_elem']} "
             f"element points, {tot['n_gds_neg_points_python']} Python points. I15-1 g v2 (lambda 10 V): "
             f"{totv.get('n_gds_curves_introduced', '-')} curves, {totv.get('n_gds_neg_points_elem', '-')} points "
             f"(column 'g v2': curves introduced (points)). Classification: restriction (prereg)."]
    cols = [col("dev", "device", kind="text")] + [col(f"g{k}", f"|Vgs| {abs(float(v)):.2f} V") for k, v in
                                                  enumerate(vg)] + [col("n", "curves introduced"),
                                                                    col("v2", "g v2 (lambda 10 V)")]
    return rf.Table("07_table_i17_1_gds", "I17-1 gds restriction items per device: negative output-conductance "
                    "points, selected correction (element / I_card exp(g)) and g v2 reference", cols, rows, notes)


def t_gm(d: dict) -> rf.Table:
    S = d["sel_restrictions"]["sel"]["per_device"]
    Rv = d["sel_restrictions"].get("ref_v2_I15_1", {}).get("per_device", {})
    rows = []
    for t in sorted(S):
        r = {"dev": t, "vt": f"{S[t]['vt_card']:.3f}"}
        for k, c in enumerate(S[t]["gm"]):
            r[f"m{k}"] = (("I: " if c["introduced"] else "")
                          + f"{c['n_neg_elem_si']} / {c['n_neg_card_si']} / {c['n_neg_python_si']}")
        rv = Rv.get(t)
        r["v2"] = (f"{rv['n_gm_introduced']} ({sum(c['n_neg_elem_si'] for c in rv['gm'])})" if rv else "-")
        r["n"] = str(S[t]["n_gm_introduced"])
        rows.append(r)
    tot = d["sel_restrictions"]["sel"]["totals"]
    vd = [abs(c["vds"]) for c in S[sorted(S)[0]]["gm"]]
    notes = ["Per |V(d,s)|: points of the 1 mV dense V(g,s) sweep in strong inversion (|V(g,s)| >= V_T + 0.2 V; V_T "
             "of the card at |V(d,s)| = 0.05 V) with dI/dV(g,s) < 0, element / card / Python I_card exp(g); 'I': "
             "introduced sign change by the curve rule (card has no negative gm below -1e-12 S, element has). "
             "Negative gm of the card itself (mobility degradation at high |V(g,s)|) is not an introduced change.",
             f"Totals selected: {tot['n_gm_curves_introduced']} of 54 curves introduced (core check); element "
             f"{tot['n_gm_neg_points_elem_si']}, card {tot['n_gm_neg_points_card_si']}, Python "
             f"{tot['n_gm_neg_points_python_si']} points. Column 'g v2': I15-1 lambda 10 V, curves introduced "
             "(element points)."]
    cols = ([col("dev", "device", kind="text"), col("vt", "V_T card (V)")]
            + [col(f"m{k}", f"|Vds| {v:.3g} V") for k, v in enumerate(vd)]
            + [col("n", "curves introduced"), col("v2", "g v2 (lambda 10 V)")])
    return rf.Table("08_table_i17_1_gm", "I17-1 gm items per device (core check): negative transconductance points "
                    "in strong inversion, element / card / I_card exp(g)", cols, rows, notes)


def t_q3(d: dict) -> rf.Table:
    rows = []
    Qs, Qv = d["sel_q3"]["rows"], d["v2_q3"]["rows"]

    def fmt(r):
        if not r["ok"]:
            return "fail"
        if "vm" in r:
            return f"{r['vm']:.4f} V"
        if "freq_hz" in r:
            return f"{r['freq_hz'] / 1e9:.3f} GHz"
        return f"{r['ratio_at_0p9']:.3f}"
    for name in ("inv_A_vtc", "inv_B_vtc", "ro5_A", "ro5_B", "mirror_n", "mirror_p"):
        rows.append({"c": name, "p": fmt(Qs[f"{name}|published"]), "c0": fmt(Qs[f"{name}|C0_refit"]),
                     "mc": fmt(Qs[f"{name}|M2_card"]), "mf": fmt(Qs[f"{name}|M2_full"]),
                     "v2": f"{fmt(Qv[f'{name}|M2_card'])} / {fmt(Qv[f'{name}|M2_full'])}"})
    notes = ["inverters: switching voltage V_M; ring oscillators: frequency; mirrors: Iout / Iref at 0.9 V. "
             "M2 card: refit with the selected full-data g in the loop, simulated without the element; M2 card + "
             "element: with it. Last column: I15-1 g v2 M2 card / M2 card + element. 'fail' at card level is not a "
             "Q3 failure (I12-1 amendment A1)."]
    return rf.Table("11_table_i17_1_circuits", "I17-1 Q3 circuits (default tolerance)",
                    [col("c", "circuit", kind="text"), col("p", "published"), col("c0", "C0_refit"),
                     col("mc", "M2 card"), col("mf", "M2 card + element"), col("v2", "I15-1 g v2 (card / element)")],
                    rows, notes)


def t_restrict(d: dict) -> rf.Table:
    R, T, S = d["sel_restrictions"], d["sel_runtime"], d["sel_summary"]
    q = R["q1_max_rel_1nA"]
    over = [t for t, v in q.items() if v["default"] > 1e-3]
    tot = R["sel"]["totals"]
    rows = [{"item": "default-tolerance difference, max relative (|I| > 1 nA)",
             "v": f"{max(v['default'] for v in q.values()):.2e}",
             "x": f"{len(over)} of 18 devices above 1e-3; r4 max {max(v['r4'] for v in q.values()):.1e}; tight max "
                  f"{max(v['tight'] for v in q.values()):.1e}"},
            {"item": "gds curves with an introduced sign change (of 90)", "v": str(tot["n_gds_curves_introduced"]),
             "x": ", ".join(tot["devices_with_gds_introduced"]) or "none"},
            {"item": "gm curves with an introduced sign change (of 54; core)", "v": str(tot["n_gm_curves_introduced"]),
             "x": ", ".join(tot["devices_with_gm_introduced"]) or "none"}]
    rt = S["runtime_circuits"]
    rows.append({"item": "runtime, DC grid ratio (18 devices)",
                 "v": f"{min(S['runtime_grid'].values()):.2f}-{max(S['runtime_grid'].values()):.2f}",
                 "x": "circuits: " + ", ".join(f"{k} {v:.2f}" for k, v in rt.items())})
    w = {k: float(np.median(v["wall"])) for k, v in T["rows"].items()}
    rows.append({"item": "ro5_A decomposition (M2 cards), wall s card / card + replica / element",
                 "v": f"{w['card']:.3f} / {w['card_rep']:.3f} / {w['elem']:.3f}",
                 "x": f"rusage load s {T['load_s']['card']:.3f} / {T['load_s']['card_rep']:.3f} / "
                      f"{T['load_s']['elem']:.3f}"})
    notes = ["Restriction items (reported). The module text has the same network size as I15-1 (5 seeds x "
             "3-16-16-1 tanh); runtime differences to I15-1 come from the M2 cards and machine load."]
    return rf.Table("12_table_i17_1_restrictions", "I17-1 restriction items of the selected correction",
                    [col("item", "item", kind="text"), col("v", "value"), col("x", "detail", kind="text")],
                    rows, notes)


def t_vds0(d: dict) -> rf.Table:
    """I17-1 run vs the I16-1 run and the I15-1 arms: element current at
    V(d,s) = 0 per tolerance; which tolerance enters the rule."""
    I17v = d["sel_summary"]["vds0"]["summary"]
    V16 = d["i16_vds0"]["arms"]
    rows = []
    srcs = (("I17-1 run, selected", {t: {**I17v[t], "n_above_1e-12": I17v[t]["n_above"]} for t in I17v}),
            ("I16-1 run, selected", V16["sel"]["summary"]),
            ("I15-1 g v2, lambda 10 V", V16["v2_I15_1"]["summary"]),
            ("I15-1 M_bias v1", V16["v1_I15_1"]["summary"]))
    for lab, S in srcs:
        for tol in ("default", "r4", "tight"):
            sm = S[tol]
            rule16 = "yes" if tol in ("default", "tight") else "no"
            rule17 = "yes" if tol == "tight" else "no"
            rows.append({"arm": lab if tol == "default" else "", "tol": tol, "el": f"{sm['elem_max']:.4e}",
                         "at": f"{sm['at'][0]}, |V(g,s)| {sm['at'][1]:.3f} V", "cr": f"{sm['card_rep_max']:.2e}",
                         "n": f"{sm['n_above_1e-12']} / {sm['n_points']}", "r16": rule16, "r17": rule17})
    notes = ["The check was changed after the observed I16-1 failure (I17-1 prereg, commit 3cfc9a7): I16-1 took the "
             "max over the default and tight forward Q1 grids; I17-1 uses the tight grid only, threshold 1e-12 A "
             "kept. The I16-1 outcome stays 'not deployable'. Default: ngspice options not set (abstol 1e-12 A, "
             "reltol 1e-3); r4: reltol 1e-4 (in neither rule); tight: reltol 1e-9, abstol 1e-18 A, vntol 1e-12 V.",
             "Reason registered for the change: at default tolerance the threshold equals abstol, the current "
             "tolerance of the solver's convergence test, so the check there does not separate an element current "
             "from the solver tolerance; at tight tolerance abstol is 1e-18 A.",
             f"I17-1 label: {d['sel_summary']['decision']['deploy']}; the same run under the I16-1 rule: "
             f"{d['sel_summary']['decision_I16_1_rule']['deploy']}."]
    return rf.Table("13_table_i17_1_vds0", "I17-1 re-registered check: element current at V(d,s) = 0 per tolerance, "
                    "I17-1 run vs I16-1 run and I15-1 arms",
                    [col("arm", "arm / run", kind="text"), col("tol", "tolerance", kind="text"),
                     col("el", "max |I_elem| (A)"), col("at", "at", kind="text"),
                     col("cr", "max |I_card+rep| (A)"), col("n", "points > 1e-12 A"),
                     col("r16", "in I16-1 rule"), col("r17", "in I17-1 rule")], rows, notes)


def t_vs_i16(d: dict) -> rf.Table:
    """I17-1 rerun vs I16-1 run: every Part B value and the labels."""
    a, b = d["sel_summary"], d["i16_sel_summary"]
    rows = []
    for key, lab, thr, ok, kind in CHECKS:
        va, vb = a["values"][key], b["values"][key]
        rows.append({"k": key, "i16": e(va if False else vb), "i17": e(va), "d": "0" if va == vb else e(va - vb)})
    rows.append({"k": "M2 in-sample RRMS (element)", "i16": f"{b['refit_in_sample']['mean_rrms_M2_native']:.6f}",
                 "i17": f"{a['refit_in_sample']['mean_rrms_M2_native']:.6f}",
                 "d": e(a['refit_in_sample']['mean_rrms_M2_native'] - b['refit_in_sample']['mean_rrms_M2_native'])})
    rows.append({"k": "label under the I16-1 rule", "i16": b["decision"]["deploy"],
                 "i17": a["decision_I16_1_rule"]["deploy"], "d": ""})
    rows.append({"k": "label under the I17-1 rule", "i16": "(not registered)", "i17": a["decision"]["deploy"], "d": ""})
    notes = ["Values as computed by the I14-1 summary (q1_vds0_max: max of default and tight, the I16-1 "
             "definition). Every stage was rerun into I17_1 (M2 refits, decks, circuits); the runtime ratio "
             "differs by machine load."]
    return rf.Table("14_table_i17_1_vs_i16_1", "I17-1 rerun vs the I16-1 run: Part B values and labels",
                    [col("k", "value", kind="text"), col("i16", "I16-1 run"), col("i17", "I17-1 run"),
                     col("d", "I17-1 - I16-1")], rows, notes)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_accuracy(d: dict, path: Path) -> list[str]:
    A = d["acc"]["rows"]
    tags = sorted(A)
    x = np.arange(len(tags))
    v1 = d["all18"]["per_device"]["M_bias"]
    c0 = d["lodo"]["per_device"]["C0_refit"]
    v2e = d["acc15"]["per_arm"]["v2"]["native"]
    fig, axs = plt.subplots(2, 1, figsize=(14, 8.2), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    ax.scatter(x, [c0[t] for t in tags], marker="o", color=BLUE, s=16, label="C0_refit (no correction)")
    ax.scatter(x - 0.15, [v1[t] for t in tags], marker="D", color=ORANGE, s=16, label="M_bias v1 (I13-1 rows)")
    ax.scatter(x + 0.15, [v2e[t] for t in tags], marker="v", color=V2C, s=16,
               label="g v2, lambda 10 V, element deck (I15-1)")
    ax.scatter(x, [A[t]["stored_python"] for t in tags], marker="^", facecolor="none", edgecolor=SEL, s=40, lw=1.0,
               label="selected, I15-2 row (Python, I_card exp(g))")
    ax.scatter(x, [A[t]["rrms_native"] for t in tags], marker="^", color=SEL, s=14,
               label="selected, element deck (fold OSDI module, I17-1)")
    for i, t in enumerate(tags):
        ax.annotate(f"{A[t]['lambda_star']:g}", (x[i], A[t]["rrms_native"]), xytext=(0, 7),
                    textcoords="offset points", fontsize=6.5, color=MUTED, ha="center")
    ax.set_yscale("log")
    ax.set_ylabel("held-out RRMS (log)")
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")
    ax.set_title("number above each selected point: lambda*_h (V) of the fold", fontsize=8.5, color=MUTED)
    ax = axs[1]
    style(ax)
    ax.scatter(x, [A[t]["native_minus_stored"] for t in tags], marker="^", color=SEL, s=26)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("element deck - I15-2 row")
    ax.set_title("difference = RRMS effect of (exp(g) - 1)(I_card - I_rep): substrate current, junction gmin current "
                 "and the body-voltage effect", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=7.5)
    fig.suptitle("I17-1 element-deck accuracy of the selected held-out pairs, 18 devices (LODO, state C)",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["acc"]), rel(F["all18"]), rel(F["lodo"]), rel(F["acc15"])]


def fig_checks(d: dict, path: Path) -> list[str]:
    keys = (("swap_asym_element_max", "exchange asymmetry (element)", 1e-3),
            ("q2", "Q2 gm / gds vs element identity", 1e-3),
            ("cap_max", "capacitance error", 1e-3),
            ("q1", "Q1 tight ratio |dI| / (1e-6 |I| + 1e-13 A)", 1.0),
            ("q1_default_max_rel_1nA", "default tolerance, relative (restriction)", 1e-3))
    fig, axs = plt.subplots(1, 5, figsize=(17, 5.8), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    tags = sorted(d["sel_q12"]["devices"])
    y = np.arange(len(tags))
    for ax, (k, lab, thr) in zip(axs, keys):
        style(ax)
        for arm, colr, mk, off, name in (("sel", SEL, "^", -0.15, "selected (lambda 1 V)"),
                                         ("v2", V2C, "v", 0.15, "I15-1 g v2 (lambda 10 V)")):
            Q = d[f"{arm}_q12"]["devices"]
            if k == "q2":
                vals = [max(Q[t]["q2_err_gm"], Q[t]["q2_err_gds"]) for t in tags]
            elif k == "q1":
                vals = [max(Q[t]["q1_tight_fwd_max_ratio"], Q[t]["q1_tight_swap_max_ratio"]) for t in tags]
            else:
                vals = [Q[t][k] for t in tags]
            ax.scatter(np.maximum(vals, 1e-18), y + off, marker=mk, color=colr, s=22, label=name)
        ax.axvline(thr, color=RED, lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(tags, fontsize=7.5)
    fig.suptitle("I17-1 Part B checks per device (M2 cards, serpentine Q1 grids); dashed: pre-registered threshold",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["sel_q12"]), rel(F["v2_q12"])]


def fig_signs(d: dict, path: Path) -> list[str]:
    S = d["sel_restrictions"]["sel"]["per_device"]
    Rv = d["sel_restrictions"].get("ref_v2_I15_1", {}).get("per_device", {})
    tags = sorted(S)
    y = np.arange(len(tags))
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.6), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    ns = [sum(c["n_neg_elem"] for c in S[t]["gds"]) for t in tags]
    nv = [sum(c["n_neg_elem"] for c in Rv[t]["gds"]) if t in Rv else 0 for t in tags]
    ax.barh(y - 0.2, ns, height=0.38, color=SEL, label="selected (lambda 1 V), element")
    ax.barh(y + 0.2, nv, height=0.38, color=V2C, label="I15-1 g v2 (lambda 10 V), element")
    for i, t in enumerate(tags):
        k = S[t]["n_gds_introduced"]
        if k:
            ax.annotate(f"{k} curve{'s' if k > 1 else ''}", (ns[i], y[i] - 0.2), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color=INK, va="center")
    ax.set_xlabel("dense-sweep points with dI/dV(d,s) < 0, |V(d,s)| >= 0.05 V (5 V(g,s) curves, 1 mV)")
    ax.set_title("gds (restriction); label: curves with an introduced sign change", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax = axs[1]
    style(ax)
    me = [sum(c["n_neg_elem_si"] for c in S[t]["gm"]) for t in tags]
    mc = [sum(c["n_neg_card_si"] for c in S[t]["gm"]) for t in tags]
    ax.barh(y - 0.2, me, height=0.38, color=SEL, label="selected, element")
    ax.barh(y + 0.2, mc, height=0.38, color=GREY, label="card alone (M2 card)")
    for i, t in enumerate(tags):
        k = S[t]["n_gm_introduced"]
        if k:
            ax.annotate(f"{k} introduced", (me[i], y[i] - 0.2), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=RED, va="center")
    ax.set_xlabel("dense-sweep points with dI/dV(g,s) < 0 in strong inversion (3 V(d,s) curves)")
    ax.set_title("gm (core check: an introduced sign change fails)", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(tags, fontsize=7.5)
    fig.suptitle("I17-1 gds and gm sign items per device (M2 cards, tight tolerance)", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["sel_restrictions"])]


def fig_gds_curves(d: dict, path: Path) -> list[str]:
    """dI/dV(d,s) of the gds curves with an introduced sign change: card,
    selected element, I15-1 g v2 element (same V(g,s); different M2 cards)."""
    S = d["sel_restrictions"]["sel"]["per_device"]
    cur = [(t, k, c) for t in sorted(S) for k, c in enumerate(S[t]["gds"]) if c["introduced"]]
    if not cur:
        cur = [(t, len(S[t]["gds"]) - 1, S[t]["gds"][-1]) for t in sorted(S)
               if sum(c["n_neg_elem"] for c in S[t]["gds"])][:2]
    cur = cur[:6]
    n = max(len(cur), 1)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow), dpi=DPI, squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax in axs.flat:
        ax.set_visible(False)
    for ax, (t, k, c) in zip(axs.flat, cur):
        ax.set_visible(True)
        style(ax)
        for rd, key, colr, lab in ((I17 / "deploy_sel/arrays", "dense_card", GREY, "M2 card (selected), card alone"),
                                   (I17 / "deploy_sel/arrays", "dense_elem", SEL, "M2 card + selected element"),
                                   (I15 / "deploy_v2/arrays", "dense_elem", V2C, "I15-1 M2 card + g v2 element")):
            z = np.load(rd / f"q12_{t}.npz")
            a = z[f"{key}_gds{k}"]
            o = np.argsort(np.abs(a[:, 0]))
            x, i_ = a[o, 0], a[o, 1]
            gds = np.gradient(i_, x)
            ax.plot(np.abs(x), gds, color=colr, lw=1.1, label=lab)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xlabel("|V(d,s)| (V)")
        ax.set_ylabel("dI/dV(d,s) (S)")
        ax.set_title(f"{t}, V(g,s) = {c['vgs']:+.2f} V: {c['n_neg_elem']} points < 0", fontsize=8.5)
        ax.legend(fontsize=6.8, frameon=False)
    fig.suptitle("I17-1 gds curves with an introduced sign change (1 mV dense sweeps, tight tolerance)",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return ["out/research/explore/I17_1/deploy_sel/arrays/q12_<tag>.npz (not in git)",
            "out/research/explore/I15_1/deploy_v2/arrays/q12_<tag>.npz (not in git)", rel(F["sel_restrictions"])]


def iv_tags(d: dict) -> list[str]:
    S = d["sel_restrictions"]["sel"]["totals"]
    tags = list(S["devices_with_gds_introduced"][:2])
    for t in ("pmos_L8_W5", "pmos_L4_W7", "nmos_L1_W3", "nmos_L1_W1p6"):
        if len(tags) >= 4:
            break
        if t not in tags:
            tags.append(t)
    return tags


def resim(d: dict, tags) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/research/explore"))
    import i17_1_vds0 as drv17
    from cryoml.explore import i14_1_chan as m
    from plot_explore_i3_2 import published_rrms
    drv = drv17.drv14
    recs = drv.load_records()
    pub, pub_curves = published_rrms({t: recs[t] for t in tags})
    held_c0 = load(I13 / "all18/heldout_cards.json")["C0_refit"]
    A = d["acc"]["rows"]
    A15 = d["acc15"]["rows"]
    out = {}
    for tag in tags:
        rec = recs[tag]
        pm = drv.devices()[tag].dev_type == "pmos"
        c0 = drv.measured_run(tag, dict(held_c0[tag]), "card")["sims"]
        sims, sc = {"C0_refit": c0}, {"C0_refit": score_device(rec, c0)["rrms"]}
        assert abs(sc["C0_refit"] - d["lodo"]["per_device"]["C0_refit"][tag]) < 1e-9
        for arm, osdi_dir, ref in (("v1", I15 / "accuracy", A15), ("v2", I15 / "accuracy", A15),
                                   ("sel", I17 / "accuracy", None)):
            corr = drv.corr_fold(arm, tag)
            card = drv.heldout_card(arm, tag)
            el = drv.measured_run(tag, card, "elem", drv.fold_osdi(osdi_dir, arm, tag), m.module_name(corr))
            sims[f"{arm}_native"] = el["sims"]
            sc[f"{arm}_native"] = score_device(rec, el["sims"])["rrms"]
            want = ref[f"{arm}|{tag}"]["rrms_native"] if ref is not None else A[tag]["rrms_native"]
            assert abs(sc[f"{arm}_native"] - want) < 1e-9, (tag, arm, sc[f"{arm}_native"], want)
            if arm == "sel":
                cr = drv.measured_run(tag, card, "card")
                py = [np.asarray(ic) * np.exp(corr.g_v(pm, c.vg, c.vd)) for ic, c in zip(cr["sims"], rec.curves)]
                sims["sel_py"] = py
                sc["sel_py"] = score_device(rec, py)["rrms"]
                assert abs(sc["sel_py"] - A[tag]["stored_python"]) < 1e-9, (tag, sc["sel_py"])
        out[tag] = {"rec": rec, "pub": pub_curves[tag], "pub_rrms": pub[tag], "sims": sims, "score": sc,
                    "lam": A[tag]["lambda_star"]}
    return out


def fig_iv(d: dict, tag: str, path: Path) -> list[str]:
    r = d["iv"][tag]
    rec = r["rec"]
    lines = (("C0_refit", BLUE, "-"), ("v1_native", ORANGE, "-"), ("v2_native", V2C, "-"),
             ("sel_py", SEL, ":"), ("sel_native", SEL, "-"))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), dpi=DPI)
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
            ax.plot(xv, np.abs(r["pub"][i]), color=GREY, ls="--", lw=1.2, alpha=alpha, zorder=2)
            for a, colr, ls in lines:
                ax.plot(xv, np.abs(r["sims"][a][i]), color=colr, ls=ls, lw=1.1, alpha=alpha, zorder=3)
            lab = c.key.split("@")[1]
            ax.annotate(f"{'|Vgs|' if kind == 'idvd' else '|Vds|'} {lab}" + ("" if c.included else " (excluded)"),
                        (xv[-1], max(abs(meas[-1]), 1e-14)), fontsize=6.8, color=MUTED, xytext=(3, 0),
                        textcoords="offset points", va="center")
        if kind == "idvg":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-12)
            ax.set_xlabel("|Vgs| (V)")
            ax.set_title("Id-Vg (log scale)", fontsize=9.5)
        else:
            ax.set_xlabel("|Vds| (V)")
            ax.set_title("Id-Vd", fontsize=9.5)
        ax.set_ylabel("|Id| (A)")
    sc = r["score"]
    handles = [
        Line2D([], [], marker="o", ls="", color=INK, ms=4,
               label="measured 77 K (filled: scored points; hollow: trimmed before the metric; faded: excluded curve)"),
        Line2D([], [], color=GREY, ls="--", lw=1.2,
               label=f"NGSpice, published 77 K card (upstream committed sweep): RRMS {r['pub_rrms']:.4f}"),
        Line2D([], [], color=BLUE, lw=1.2,
               label=f"NGSpice re-simulated, C0_refit held-out card (no correction): RRMS {sc['C0_refit']:.4f}"),
        Line2D([], [], color=ORANGE, lw=1.2,
               label=f"NGSpice element deck, M_bias v1 held-out card + v1 fold module: RRMS {sc['v1_native']:.4f}"),
        Line2D([], [], color=V2C, lw=1.2,
               label=f"NGSpice element deck, g v2 (lambda 10 V) held-out card + fold module: RRMS "
                     f"{sc['v2_native']:.4f}"),
        Line2D([], [], color=SEL, ls=":", lw=1.2,
               label=f"NGSpice re-simulated, selected held-out card x exp(g) in Python (I15-2 row): RRMS "
                     f"{sc['sel_py']:.4f}"),
        Line2D([], [], color=SEL, lw=1.2,
               label=f"NGSpice element deck, selected held-out card + fold module (lambda* {r['lam']:g} V): RRMS "
                     f"{sc['sel_native']:.4f}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag}: held-out pairs (LODO, state C) re-simulated in ngspice-41; element deck = card + "
                 "channel-current replica + OSDI module.\nAll curves simulated with the BSIM4 clamp written into the "
                 "card; RRMS equals the stored rows.", fontsize=9.4, color=INK)
    fig.tight_layout(rect=(0, 0.23, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I13 / "records/real_dev.npz"), rel(I13 / "records/real_test.npz"),
            rel(I13 / "all18/heldout_cards.json"), rel(I13 / "lodo/heldout_cards.json"),
            "I15_2/lodo_sym_e1_l{lambda*}/{corrections,heldout_cards}.json", "upstream committed {vd,vg}_sweep/*.txt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = {k: load(p) for k, p in F.items()}
    d["i16_sel_summary"] = load(EXP / "I16_1/deploy_sel/summary.json")
    tags = iv_tags(d)
    d["iv"] = resim(d, tags)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    add_table(t_partb(d), "Part B checks vs thresholds, selected correction (outcome); I15-1 g v2 and v1 values",
              [rel(F["sel_summary"]), rel(F["v2_summary"]), rel(F["v1_summary"])])
    add_table(t_validity(d), "Validity V1-V4", [rel(F["val"]), rel(F["unit"])])
    add_table(t_devices(d), "Part B per device for the selected correction", [rel(F["sel_q12"]), rel(F["sel_refit"])])
    add_table(t_accuracy(d), "Element-deck accuracy of the selected held-out pairs vs the I15-2 rows, aggregates "
              "and GMRs", [rel(F["acc"]), rel(F["i152"]), rel(F["acc15"])])
    add_fig("05_i17_1_accuracy_per_device.png", fig_accuracy,
            "Per-device held-out RRMS: selected (I15-2 row and element deck), v1, g v2 element, C0_refit")
    add_fig("06_i17_1_checks_per_device.png", fig_checks, "Part B checks per device vs thresholds, selected and g v2")
    add_table(t_gds(d), "gds restriction items per device (negative output-conductance points, curves introduced)",
              [rel(F["sel_restrictions"])])
    add_table(t_gm(d), "gm items per device (strong inversion; core check)", [rel(F["sel_restrictions"])])
    add_fig("09_i17_1_signs_per_device.png", fig_signs, "gds and gm sign items per device, selected vs g v2 / card")
    add_fig("10_i17_1_gds_curves.png", fig_gds_curves,
            "dI/dV(d,s) of the gds curves with an introduced sign change: card, selected element, g v2 element")
    add_table(t_q3(d), "Q3 circuits", [rel(F["sel_q3"]), rel(F["v2_q3"])])
    add_table(t_restrict(d), "Restriction items: tolerance, sign changes, runtime",
              [rel(F["sel_restrictions"]), rel(F["sel_runtime"]), rel(F["sel_summary"])])
    add_table(t_vds0(d), "Re-registered V(d,s) = 0 check: element current per tolerance, I17-1 vs I16-1 runs "
              "and I15-1 arms", [rel(F["sel_summary"]), rel(F["i16_vds0"])])
    add_table(t_vs_i16(d), "I17-1 rerun vs I16-1 run: Part B values and labels",
              [rel(F["sel_summary"]), rel(EXP / "I16_1/deploy_sel/summary.json")])
    for k, tag in enumerate(tags):
        name = f"{15 + k:02d}_i17_1_iv_{tag}.png"
        items.append((name, f"I-V of {tag}: measured 77 K points, NGSpice of the published card, and NGSpice "
                            "re-simulated C0_refit card, v1 and g v2 element decks, selected held-out card x exp(g) "
                            "(Python) and selected element deck", fig_iv(d, tag, folder / name)))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i17_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    D, A = d["decision"], d["acc"]
    s = d["sel_summary"]["values"]
    tot = d["sel_restrictions"]["sel"]["totals"]
    readme = [
        f"# Report batch {args.date} · {TOPIC}", "",
        f"Generated {now} by {gen}. NGSpice runs: re-simulation of the published, C0_refit, v1, g v2 and selected "
        f"held-out cards for the {len(tags)} I-V devices (checked against the stored rows); everything else from stored "
        "files.", "",
        "## Contents", "",
        "- I17-1: re-registered deployment test (Part B, I16-1 rules except the V(d,s) = 0 check) of the I15-2 selected bias-only "
        "correction: symmetric inputs, EPS 0.01 V, penalty lambda selected per fold by nested cross-validation; "
        "deployed full-data model lambda* = 1 V (`full|sym_e1_l1`). Pre-registration "
        f"`out/research/explore/I17_1/prereg.md`, commit {COMMITS['prereg']}; code {COMMITS['code']}.",
        f"- Outcome (pre-registered rule): **{D['outcome']}**; label {d['sel_summary']['decision']['deploy']}; core "
        f"failures: {', '.join(d['sel_summary']['decision']['core_fail']) or 'none'}. Q1 runs {s['q1_bad']}, Q1 "
        f"violations {s['q1_tight_viol']}, exchange asymmetry {s['swap_asym_max']:.1e}, Q2 {s['q2_ac_max']:.1e}, "
        f"capacitances {s['cap_max']:.1e}, gm curves {s['q2_gm_sign']} of 54. Restrictions: "
        + ("; ".join(d["sel_summary"]["decision"]["restrictions"]) or "none")
        + f" (default tolerance {s['q1_default_max']:.2e}, {s['q2_gds_sign']} of 90 gds curves: "
        + (", ".join(tot["devices_with_gds_introduced"]) or "none") + f", runtime {s['runtime_ratio_max']:.2f}).",
        "- The Q1 V(d,s) = 0 check was re-registered after the observed I16-1 failure: tight tolerance only, threshold "
        "1e-12 A kept (tables 01, 13). The I16-1 outcome stays 'not deployable'. I17-1 run: tight maximum "
        f"{d['sel_summary']['vds0']['summary']['tight']['elem_max']:.2e} A; default tolerance (reported, no rule) "
        f"{d['sel_summary']['vds0']['summary']['default']['elem_max']:.4e} A at "
        f"{d['sel_summary']['vds0']['summary']['default']['at'][0]}; label under the unchanged I16-1 rule: "
        f"{d['sel_summary']['decision_I16_1_rule']['deploy']}.",
        f"- Element-deck accuracy (selected held-out pairs): all-device {A['summary_native']['all_device_mean']:.5f} "
        f"vs I15-2 rows {A['summary_stored']['all_device_mean']:.5f} (max per-device |d| "
        f"{A['max_abs_native_minus_stored']:.1e}); GMR(element, v1 rows) {ci(A['gmr_native']['sel_native_vs_v1_stored'])}.",
        "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I17-1 Part B of the I15-2 selected correction (full-data lambda* 1 V), V(d,s) = 0 check at tight tolerance "
               f"only (re-registered after the I16-1 failure): {D['outcome']} (pre-registered rule); label {d['sel_summary']['decision']['deploy']}.",
               f"gm curves {s['q2_gm_sign']} of 54; restrictions: default tolerance {s['q1_default_max']:.1e}, "
               f"{s['q2_gds_sign']} of 90 gds curves, runtime {s['runtime_ratio_max']:.2f}.",
               f"Element-deck RRMS {A['summary_native']['all_device_mean']:.4f} (I15-2 rows "
               f"{A['summary_stored']['all_device_mean']:.4f})."]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
