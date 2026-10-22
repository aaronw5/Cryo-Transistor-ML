#!/usr/bin/env python3
"""Report batch for I14-1: Part B deployment test of g v2 with a
channel-current replica, BSIM4 clamp state C in every deck, element-identity
Q2 reference; M_bias v1 full-data g as reference arm; sky130 77 K, 18 devices.

Reads stored files:

* ``out/research/explore/I14_1/{prereg.json, decision.json}``
* ``I14_1/repcheck/repcheck.json``, ``I14_1/validity/{validity,unit_tests}.json``
* ``I14_1/accuracy/accuracy.json`` (element-deck RRMS of the held-out pairs)
* ``I14_1/deploy_{v2,v1}/{compile,refit,q12,q3,summary}.json``
* ``I14_1/diagnosis/diagnosis.json`` (post hoc)
* I13-1 state-C inputs (records, corrections, held-out cards, rows)

NGSpice: the I-V figures re-simulate the published card (upstream committed
sweeps), the I13-1 C0_refit card and the v2 / v1 held-out cards (card + replica
deck, element deck with the fold OSDI module); every RRMS is checked against
the stored rows.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i14_1.py --date 2026-10-22
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_explore_e2_e3_e4 as pe  # noqa: E402  (helpers, style)
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, RED, VIOLET, AQUA, DPI,  # noqa: E402
                                   EXP, ROOT, load, rel, style)
from cryoml.explore import i12_1_symm as v2  # noqa: E402
from cryoml.explore.e1_correction import score_device  # noqa: E402

from matplotlib.lines import Line2D  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i14_1"
I14 = EXP / "I14_1"
I13 = EXP / "I13_1"
F = {"prereg": I14 / "prereg.json", "decision": I14 / "decision.json", "rep": I14 / "repcheck/repcheck.json",
     "val": I14 / "validity/validity.json", "unit": I14 / "validity/unit_tests.json",
     "acc": I14 / "accuracy/accuracy.json", "diag": I14 / "diagnosis/diagnosis.json",
     "lodo": I13 / "lodo/results.json"}
for _a in ("v2", "v1"):
    for _k in ("compile", "refit", "q12", "q3", "summary"):
        F[f"{_a}_{_k}"] = I14 / f"deploy_{_a}/{_k}.json"
COMMITS = {"prereg": "d637500", "code": "f40c6b7"}
IV_TAGS = ("nmos_L100_W100", "nmos_L20_W0p64", "nmos_L8_W1p6", "pmos_L2_W5")


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def e(x: float) -> str:
    return f"{x:.2e}" if (x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4)) else f"{x:.4g}"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
CHECKS = (  # key, label, threshold text, pass(value), kind
    ("q1_tight_viol", "Q1 equivalence: points violating |dI| <= 1e-6 |I_py| + 1e-13 A (tight, both benches)", "0",
     lambda v: v == 0, "core"),
    ("q1_bad", "Q1 runs: convergence messages + non-finite values (element runs)", "0", lambda v: v == 0, "core"),
    ("q1_vds0_max", "Q1 max |I_elem(Vds = 0)| (A)", "<= 1e-12", lambda v: v <= 1e-12, "core"),
    ("g_text_max", "Verilog-A g vs Python, max |dg|", "<= 1e-9", lambda v: v <= 1e-9, "core"),
    ("g_sym_text_max", "Verilog-A g, exchanged vs forward terminals, max |dg|", "<= 1e-12",
     lambda v: v <= 1e-12, "core"),
    ("swap_asym_max", "exchange asymmetry, element-introduced max |d ln|", "<= 1e-3", lambda v: v <= 1e-3, "core"),
    ("q2_ac_max", "Q2 gm / gds vs element identity, max normalized difference", "<= 1e-3",
     lambda v: v <= 1e-3, "core"),
    ("cap_max", "Q2 capacitances (Y_dd, Y_dg, Y_gg), max relative error", "<= 1e-3", lambda v: v <= 1e-3, "core"),
    ("q2_gm_sign", "Q2 gm curves with an introduced sign change (of 54)", "0", lambda v: v == 0, "core"),
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
    for key, lab, thr, ok, kind in CHECKS:
        a, b = d["v2_summary"]["values"][key], d["v1_summary"]["values"][key]
        rows.append({"chk": lab, "kind": kind, "thr": thr, "v2": e(a), "p2": "pass" if ok(a) else "FAIL",
                     "v1": e(b), "p1": "pass" if ok(b) else "FAIL"})
    rows.append({"chk": "label", "kind": "", "thr": "", "v2": d["v2_summary"]["decision"]["deploy"], "p2": "",
                 "v1": d["v1_summary"]["decision"]["deploy"], "p1": ""})
    s2 = d["v2_summary"]
    notes = ["v2: I13-1 full-data g v2 (state C) with its M2 cards (refit from the starting card with g in the loop, "
             "state C, 0 failed); v1: reference arm, I13-1 full-data M_bias v1 g (abs() inputs) with its own M2 cards; "
             "same element (channel-current replica, gmin VCCS, module I(d, s) <+ (exp(g) - 1) I(sense)).",
             f"Q1 runs (v2): the 2 counted lines are 'dynamic gmin stepping failed' and 'true gmin stepping failed' "
             f"in one element run (nmos_L8_W1p6, exchanged bench, tight); the card + replica reference run of the "
             f"forward bench printed the same two lines. Q1 equivalence on that run: 0 violations.",
             f"Q2 with the I12-1 formula exp(g) Re Y_card + I_card exp(g) dg/dV on the same runs: max "
             f"{s2['q2_err_i12_formula_max']:.3g} (v2).",
             f"Runtime: DC grid {min(s2['runtime_grid'].values()):.2f}-{max(s2['runtime_grid'].values()):.2f}; "
             "circuits " + ", ".join(f"{k} {v:.2f}" for k, v in s2["runtime_circuits"].items())
             + "; ro5_B excluded (fails at card level, I12-1 amendment A1)."]
    return rf.Table("01_table_i14_1_part_b", "I14-1 Part B checks vs pre-registered thresholds: g v2 (outcome) and "
                    "M_bias v1 (reference arm)",
                    [col("chk", "check", kind="text"), col("kind", "type", kind="text"), col("thr", "threshold"),
                     col("v2", "g v2 value"), col("p2", "g v2"), col("v1", "v1 reference value"), col("p1", "v1")],
                    rows, notes)


def t_validity(d: dict) -> rf.Table:
    V, U, R = d["val"], d["unit"], d["rep"]["summary"]
    dec = d["decision"]
    rows = [
        {"c": "V1 card-only library deck x exp(fold g) vs I13-1 rows (36 rows)", "t": "<= 1e-9",
         "v": e(V["v1_max_rrms_diff"]), "p": "pass" if V["v1_max_rrms_diff"] <= 1e-9 else "FAIL"},
        {"c": "V2 unit tests (" + ", ".join(x.split(".")[-1] for x in U["files"]) + ")", "t": "all pass",
         "v": U["tail"][-1] if U["tail"] else "", "p": "pass" if U["passed"] else "FAIL"},
        {"c": "V3 element-deck accuracy rows per arm / failed M2 refits", "t": "18 / 0",
         "v": f"{len(d['acc']['per_arm']['v2']['native'])}, {len(d['acc']['per_arm']['v1']['native'])} / "
              f"{d['v2_summary']['refit_in_sample']['n_failed']}, {d['v1_summary']['refit_in_sample']['n_failed']}",
         "p": "pass" if dec["validity"]["valid"] else "FAIL"},
    ]
    for lab, key in (("published", "published"), ("start (C)", "start_C"), ("v2 held-out (C)", "v2_heldout_C"),
                     ("v1 held-out (C)", "v1_heldout_C")):
        s = R[key]
        rows.append({"c": f"replica pre-check, {lab}: P1 non-channel max / P2 max abs (A) / P3 forward max rel / "
                          f"P3 devices <= 1e-12",
                     "t": "1e-12 relative requested",
                     "v": f"{s['P1_rep_nonchannel_max']:.1g} / {s['P2_max_abs']:.2g} / {s['P3_rep_id_fwd_max_rel']:.2g}"
                          f" / {len(s['devices_P3_rep_id_fwd_le_1e-12'])} of 18",
                     "p": "met in part"})
    notes = ["Validity rule (prereg): any V fail -> invalid. Outcome: valid.",
             "Replica pre-check (before registration): P1 replica isub, igidl, igisl, gate currents; P2 sensed current "
             "vs the replica's own BSIM4 channel current; P3 replica channel current vs the card's channel current "
             "(forward quadrant, |I| > 1 nA). The card's own Isub and junction currents flow through its body resistor "
             "network (rbodymod = 1) and change its channel current; the replica without them does not see that."]
    return rf.Table("02_table_i14_1_validity", "I14-1 validity and replica pre-check",
                    [col("c", "check", kind="text"), col("t", "criterion"), col("v", "value"), col("p", "result")],
                    rows, notes)


def t_devices(d: dict) -> rf.Table:
    Q = d["v2_q12"]["devices"]
    R = d["v2_refit"]
    rows = []
    for t in sorted(Q):
        r = Q[t]
        gds = sum(r["q2_gds_curves_introduced"])
        msgs = sum(len(v) for k, v in r["msgs"].items() if k.endswith("_elem"))
        n = R["native"][t]
        rows.append({"dev": t, "q1": f"{max(r['q1_tight_fwd_max_ratio'], r['q1_tight_swap_max_ratio']):.2f}",
                     "as": e(r["swap_asym_element_max"]), "asc": e(r["swap_asym_card_max"]),
                     "q2": e(max(r["q2_err_gm"], r["q2_err_gds"])), "cap": e(r["cap_max"]),
                     "def": e(r["q1_default_max_rel_1nA"]), "gds": str(gds), "msg": str(msgs),
                     "rt": f"{d['v2_summary']['runtime_grid'][t]:.2f}",
                     "m2": f"{n['rrms_python']:.4f} / {n['rrms_native']:.4f}"})
    notes = ["q1: max |dI| / (1e-6 |I_py| + 1e-13 A), tight, forward and exchanged (pass <= 1); asym: element-introduced "
             "|d ln(I_swap / I_fwd)|, card: the card's own; q2: max normalized gm / gds difference vs the element "
             "identity; cap: capacitance error; def: default-tolerance relative difference (|I| > 1 nA); gds: gds curves "
             "with an introduced sign change (of 5); msg: convergence lines in element runs; rt: DC grid wall-time "
             "ratio; M2 RRMS in-sample: Python (I_card exp(g)) / SPICE-native element."]
    return rf.Table("03_table_i14_1_devices_v2", "I14-1 Part B per device, g v2 with its M2 card",
                    [col("dev", "device", kind="text"), col("q1", "Q1 ratio"), col("as", "asym"),
                     col("asc", "asym card"), col("q2", "Q2"), col("cap", "cap"), col("def", "default tol"),
                     col("gds", "gds sign"), col("msg", "msgs"), col("rt", "runtime"),
                     col("m2", "M2 RRMS py / native")], rows, notes)


def t_accuracy(d: dict) -> rf.Table:
    A = d["acc"]
    rows = []
    for arm, lab in (("v2", "M_bias v2 (g v2)"), ("v1", "M_bias v1 (reference)")):
        pa = A["per_arm"][arm]
        s, n = pa["summary_stored"], pa["summary_native"]
        rows.append({"arm": lab, "st": f"{s['all_device_mean']:.5f} / {s['combined']:.5f} / {s['median']:.5f}",
                     "na": f"{n['all_device_mean']:.5f} / {n['combined']:.5f} / {n['median']:.5f}",
                     "d": f"{n['all_device_mean'] - s['all_device_mean']:+.1e}",
                     "mx": f"{pa['max_abs_native_minus_stored']:.1e}",
                     "ch": f"{pa['max_native_vs_channel_ratio']:.2f}"})
    G = A["gmr_native"]
    L = d["lodo"]["comparisons"]
    rows.append({"arm": "GMR(v2, C0_refit): stored / native", "st": ci(L["A1_M_bias_v2_vs_C0_refit"]),
                 "na": ci(G["v2_native_vs_C0_refit"]), "d": "", "mx": "", "ch": ""})
    rows.append({"arm": "GMR(v2, v1): stored / native v2 vs stored v1", "st": ci(L["A2_M_bias_v2_vs_M_bias_v1"]),
                 "na": ci(G["v2_native_vs_v1_stored"]), "d": "", "mx": "", "ch": ""})
    rows.append({"arm": "GMR(v2, v1): native vs native", "st": "", "na": ci(G["v2_native_vs_v1_native"]), "d": "",
                 "mx": "", "ch": ""})
    rows.append({"arm": "GMR(v1, C0_refit): stored / native", "st": ci(L["M_bias_v1_vs_C0_refit"]),
                 "na": ci(G["v1_native_vs_C0_refit"]), "d": "", "mx": "", "ch": ""})
    notes = ["stored: I13-1 state-C rows (Python, I_card exp(g)); the card-only library deck reproduces them (V1, 0.0). "
             "native: element deck (fold g as OSDI module with the channel-current replica, I13-1 held-out card), "
             "I = I_card + (exp(g) - 1) I_rep. all-device / combined / median.",
             "max |d|: per-device |native - stored|. ch: native vs channel model I_card + (exp(g) - 1) I_rep from the "
             "card + replica run, max |dI| / (1e-3 |I| + 1e-12 A) at default tolerance.",
             "Registered accuracy flags stay the I13-1 values: A1 passes (CI upper < 1), A2 fails (CI upper > 1.05). "
             "GMR: device bootstrap 10,000, 95% CI."]
    return rf.Table("04_table_i14_1_accuracy", "I14-1 element-deck accuracy of the held-out pairs vs the I13-1 rows",
                    [col("arm", "arm / contrast", kind="text"), col("st", "stored (Python)"),
                     col("na", "element deck (native)"), col("d", "all-device diff"), col("mx", "max |d|"),
                     col("ch", "native vs channel model")], rows, notes)


def t_diag(d: dict) -> rf.Table:
    D = d["diag"]
    rows = []
    for k, v in sorted(D["messages"].items()):
        tag, var = k.split("|")
        for bench, msgs in v.items():
            rows.append({"dev": tag, "var": var, "b": bench, "m": "; ".join(msgs)})
    notes = ["Post hoc (scripts/research/explore/i14_1_diagnosis.py; the recorded outcome is unchanged): tight forward and "
             "exchanged grids of every M2 card (g v2) in six variants: card only; card + channel replica + gmin VCCS "
             "(I14-1 reference); the same without the VCCS; card + I12-1 replica (junction and substrate currents kept); "
             "I14-1 element; element without the VCCS.",
             "No message in the card-only and I12-1-replica variants of any device. The warnings of nmos_L8_W1p6 occur "
             "with the channel-current replica with and without the VCCS and with and without the module.",
             f"Q1 of the I14-1 element against its card + replica reference: 0 violations on all 18 devices; the variant "
             f"without the VCCS on nmos_L0p25_W1p6 (source stepping failed) has 1735 violating points."]
    return rf.Table("07_table_i14_1_diagnosis", "I14-1 diagnosis: convergence warnings by deck variant (post hoc)",
                    [col("dev", "device", kind="text"), col("var", "variant", kind="text"), col("b", "bench"),
                     col("m", "messages", kind="text")], rows, notes)


def t_q3(d: dict) -> rf.Table:
    rows = []
    for arm in ("v2", "v1"):
        Q = d[f"{arm}_q3"]["rows"]
        for name in ("inv_A_vtc", "inv_B_vtc", "ro5_A", "ro5_B", "mirror_n", "mirror_p"):
            vals = []
            for a in ("published", "C0_refit", "M2_card", "M2_full"):
                r = Q[f"{name}|{a}"]
                if not r["ok"]:
                    vals.append("fail")
                elif "vm" in r:
                    vals.append(f"{r['vm']:.4f} V")
                elif "freq_hz" in r:
                    vals.append(f"{r['freq_hz'] / 1e9:.3f} GHz")
                else:
                    vals.append(f"{r['ratio_at_0p9']:.3f}")
            rows.append({"arm": arm if name == "inv_A_vtc" else "", "c": name, "p": vals[0], "c0": vals[1],
                         "mc": vals[2], "mf": vals[3]})
    notes = ["inverters: switching voltage V_M; ring oscillators: frequency; mirrors: Iout / Iref at 0.9 V. "
             "ro5_B fails at card level for C0_refit and M2_card (timestep too small; I12-1 amendment A1)."]
    return rf.Table("08_table_i14_1_circuits", "I14-1 Q3 circuits (default tolerance)",
                    [col("arm", "arm", kind="text"), col("c", "circuit", kind="text"), col("p", "published"),
                     col("c0", "C0_refit"), col("mc", "M2 card"), col("mf", "M2 card + element")], rows, notes)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_accuracy(d: dict, path: Path) -> list[str]:
    A = d["acc"]["rows"]
    tags = sorted({k.split("|")[1] for k in A})
    x = np.arange(len(tags))
    fig, axs = plt.subplots(2, 1, figsize=(14, 8.2), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    ax = axs[0]
    style(ax)
    for arm, colr, mk, off in (("v2", AQUA, "^", -0.12), ("v1", ORANGE, "D", 0.12)):
        ax.scatter(x + off, [A[f"{arm}|{t}"]["stored_python"] for t in tags], marker=mk, facecolor="none",
                   edgecolor=colr, s=34, lw=1.0, label=f"M_bias {arm}, stored I13-1 row (Python, I_card exp(g))")
        ax.scatter(x + off, [A[f"{arm}|{t}"]["rrms_native"] for t in tags], marker=mk, color=colr, s=14,
                   label=f"M_bias {arm}, element deck (OSDI, channel-current replica)")
    c0 = d["lodo"]["per_device"]["C0_refit"]
    ax.scatter(x, [c0[t] for t in tags], marker="o", color=BLUE, s=14, label="C0_refit (no correction)")
    ax.set_yscale("log")
    ax.set_ylabel("held-out RRMS (log)")
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="upper left")
    ax = axs[1]
    style(ax)
    for arm, colr, mk, off in (("v2", AQUA, "^", -0.12), ("v1", ORANGE, "D", 0.12)):
        ax.scatter(x + off, [A[f"{arm}|{t}"]["native_minus_stored"] for t in tags], marker=mk, color=colr, s=26,
                   label=f"M_bias {arm}")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("element deck - stored row")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("difference = RRMS effect of (exp(g) - 1)(I_card - I_rep): substrate current, junction gmin current "
                 "and the body-voltage effect", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=60, ha="right", fontsize=7.5)
    fig.suptitle("I14-1 element-deck accuracy of the held-out pairs (fold g + I13-1 held-out card), 18 devices",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["acc"]), rel(F["lodo"])]


def fig_checks(d: dict, path: Path) -> list[str]:
    keys = (("swap_asym_element_max", "exchange asymmetry (element)", 1e-3),
            ("q2", "Q2 gm / gds vs element identity", 1e-3),
            ("cap_max", "capacitance error", 1e-3),
            ("q1", "Q1 tight ratio |dI| / (1e-6 |I| + 1e-13 A)", 1.0))
    fig, axs = plt.subplots(1, 4, figsize=(16, 5.6), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    tags = sorted(d["v2_q12"]["devices"])
    y = np.arange(len(tags))
    for ax, (k, lab, thr) in zip(axs, keys):
        style(ax)
        for arm, colr, mk, off in (("v2", AQUA, "^", -0.15), ("v1", ORANGE, "D", 0.15)):
            Q = d[f"{arm}_q12"]["devices"]
            if k == "q2":
                vals = [max(Q[t]["q2_err_gm"], Q[t]["q2_err_gds"]) for t in tags]
            elif k == "q1":
                vals = [max(Q[t]["q1_tight_fwd_max_ratio"], Q[t]["q1_tight_swap_max_ratio"]) for t in tags]
            else:
                vals = [Q[t][k] for t in tags]
            ax.scatter(np.maximum(vals, 1e-18), y + off, marker=mk, color=colr, s=22,
                       label="g v2" if arm == "v2" else "v1 reference")
        if k == "swap_asym_element_max":
            ax.scatter([max(d["v2_q12"]["devices"][t]["swap_asym_card_max"], 1e-18) for t in tags], y, marker="|",
                       color=GREY, s=40, label="card's own asymmetry")
        ax.axvline(thr, color=RED, lw=1.0, ls="--")
        ax.set_xscale("log")
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(tags, fontsize=7.5)
    fig.suptitle("I14-1 Part B checks per device (M2 cards, tight tolerance); dashed: pre-registered threshold",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["v2_q12"]), rel(F["v1_q12"])]


def resim(d: dict) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/research/explore"))
    os.environ["CRYOML_E1_RECORDS"] = str(I13 / "records")
    import i14_1_chan as drv
    from cryoml.explore import i14_1_chan as m
    from plot_explore_i3_2 import published_rrms
    recs = drv.load_records()
    pub, pub_curves = published_rrms({t: recs[t] for t in IV_TAGS})
    held_c0 = load(I13 / "all18/heldout_cards.json")["C0_refit"]
    A = d["acc"]["rows"]
    out = {}
    for tag in IV_TAGS:
        rec = recs[tag]
        pm = drv.devices()[tag].dev_type == "pmos"
        c0 = drv.measured_run(tag, dict(held_c0[tag]), "card")["sims"]
        sims, sc = {"C0_refit": c0}, {"C0_refit": score_device(rec, c0)["rrms"]}
        assert abs(sc["C0_refit"] - d["lodo"]["per_device"]["C0_refit"][tag]) < 1e-9
        for arm in ("v2", "v1"):
            corr = drv.corr_fold(arm, tag)
            card = drv.heldout_card(arm, tag)
            cr = drv.measured_run(tag, card, "card_rep")
            el = drv.measured_run(tag, card, "elem", drv.fold_osdi(I14 / "accuracy", arm, tag), m.module_name(corr))
            py = [np.asarray(ic) * np.exp(corr.g_v(pm, c.vg, c.vd)) for ic, c in zip(cr["sims"], rec.curves)]
            sims[f"{arm}_py"], sims[f"{arm}_native"] = py, el["sims"]
            sc[f"{arm}_py"] = score_device(rec, py)["rrms"]
            sc[f"{arm}_native"] = score_device(rec, el["sims"])["rrms"]
            assert abs(sc[f"{arm}_py"] - A[f"{arm}|{tag}"]["stored_python"]) < 1e-9, (tag, arm)
            assert abs(sc[f"{arm}_native"] - A[f"{arm}|{tag}"]["rrms_native"]) < 1e-9, (tag, arm)
        out[tag] = {"rec": rec, "pub": pub_curves[tag], "pub_rrms": pub[tag], "sims": sims, "score": sc}
    return out


def fig_iv(d: dict, tag: str, path: Path) -> list[str]:
    r = d["iv"][tag]
    rec = r["rec"]
    lines = (("C0_refit", BLUE, "-"), ("v2_py", AQUA, ":"), ("v2_native", AQUA, "-"), ("v1_native", ORANGE, "-"))
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
               label=f"NGSpice re-simulated, I13-1 C0_refit held-out card (no correction): RRMS {sc['C0_refit']:.4f}"),
        Line2D([], [], color=AQUA, ls=":", lw=1.2,
               label=f"NGSpice re-simulated, v2 held-out card x exp(g v2 fold) in Python (I13-1 row): RRMS "
                     f"{sc['v2_py']:.4f}"),
        Line2D([], [], color=AQUA, lw=1.2,
               label=f"NGSpice element deck, v2 held-out card + g v2 fold module (channel-current replica): RRMS "
                     f"{sc['v2_native']:.4f}"),
        Line2D([], [], color=ORANGE, lw=1.2,
               label=f"NGSpice element deck, v1 held-out card + g v1 fold module (reference arm): RRMS "
                     f"{sc['v1_native']:.4f}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag}: held-out pairs (LODO, I13-1 state C) re-simulated in ngspice-41; element deck = card + "
                 "channel-current replica + OSDI module.\nAll curves simulated with the BSIM4 clamp written into the "
                 "card; RRMS equals the stored rows.", fontsize=9.4, color=INK)
    fig.tight_layout(rect=(0, 0.2, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I13 / "records/real_dev.npz"), rel(I13 / "records/real_test.npz"),
            rel(I13 / "all18/heldout_cards.json"), rel(I13 / "lodo/heldout_cards.json"),
            rel(I13 / "lodo/corrections.json") + f" (M_bias_v2|{tag})",
            rel(I13 / "all18/corrections.json") + f" (M_bias|{tag})", "upstream committed {vd,vg}_sweep/*.txt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = {k: load(p) for k, p in F.items()}
    d["iv"] = resim(d)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    add_table(t_partb(d), "Part B checks vs thresholds, g v2 (outcome) and M_bias v1 (reference arm), labels",
              [rel(F["v2_summary"]), rel(F["v1_summary"])])
    add_table(t_validity(d), "Validity V1-V3 and the replica pre-check", [rel(F["val"]), rel(F["unit"]), rel(F["rep"])])
    add_table(t_devices(d), "Part B per device for g v2", [rel(F["v2_q12"]), rel(F["v2_refit"])])
    add_table(t_accuracy(d), "Element-deck accuracy of the held-out pairs vs the I13-1 rows, aggregates and GMRs",
              [rel(F["acc"]), rel(F["lodo"])])
    add_fig("05_i14_1_accuracy_per_device.png", fig_accuracy,
            "Per-device held-out RRMS, stored vs element deck, and their difference")
    add_fig("06_i14_1_checks_per_device.png", fig_checks, "Part B checks per device vs thresholds, v2 and v1")
    add_table(t_diag(d), "Post-hoc diagnosis of the convergence warnings by deck variant", [rel(F["diag"])])
    add_table(t_q3(d), "Q3 circuits, both arms", [rel(F["v2_q3"]), rel(F["v1_q3"])])
    for k, tag in enumerate(IV_TAGS):
        name = f"{9 + k:02d}_i14_1_iv_{tag}.png"
        items.append((name, f"I-V of {tag}: measured 77 K points, NGSpice of the published card, and NGSpice "
                            "re-simulated C0_refit card, v2 held-out card x exp(g) (Python), v2 element deck and v1 "
                            "element deck", fig_iv(d, tag, folder / name)))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i14_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    D, A = d["decision"], d["acc"]["per_arm"]
    s2 = d["v2_summary"]["values"]
    readme = [
        f"# Report batch {args.date} · {TOPIC}", "",
        f"Generated {now} by {gen}. NGSpice runs: re-simulation of the published, C0_refit and v2 / v1 held-out cards "
        f"for the {len(IV_TAGS)} I-V devices (checked against the stored rows); everything else from stored files.", "",
        "## Contents", "",
        "- I14-1: re-registered Part B of g v2 (pre-registration `out/research/explore/I14_1/prereg.md`, commit "
        f"{COMMITS['prereg']}; code {COMMITS['code']}) with (1) the b4check.c clamp written into the card in every "
        "multi-analysis deck builder, (2) an element that scales only the replica's channel current (no substrate, "
        "GIDL / GISL, junction or gate current; gmin VCCS), (3) the element-identity Q2 reference.",
        f"- Outcome (pre-registered rule): **{D['outcome']}**; action '{D['action']}'. Validity: valid. The only failed "
        "core check is 'Q1 runs': 2 convergence lines ('dynamic gmin stepping failed', 'true gmin stepping failed') in "
        "1 of 126 element decks (7 per device; nmos_L8_W1p6, exchanged bench, tight). Every other core check passes: Q1 0 violations, "
        f"exchange asymmetry {s2['swap_asym_max']:.1e}, Q2 {s2['q2_ac_max']:.1e}, capacitances {s2['cap_max']:.1e}, "
        f"0 gm sign changes, 0 circuit failures. Restrictions: default tolerance {s2['q1_default_max']:.2e}, "
        f"{s2['q2_gds_sign']} gds curves, runtime {s2['runtime_ratio_max']:.2f}.",
        "- Diagnosis (post hoc): the warnings come from the channel-current replica (zero junction saturation current), "
        "with and without the gmin VCCS and the module; no card-only or I12-1-replica run prints them.",
        f"- Element-deck accuracy (held-out pairs, I13-1): v2 all-device {A['v2']['summary_native']['all_device_mean']:.5f}"
        f" vs stored {A['v2']['summary_stored']['all_device_mean']:.5f} (max per-device |d| "
        f"{A['v2']['max_abs_native_minus_stored']:.1e}); v1 {A['v1']['summary_native']['all_device_mean']:.5f} vs "
        f"{A['v1']['summary_stored']['all_device_mean']:.5f} (max {A['v1']['max_abs_native_minus_stored']:.1e}).",
        f"- Reference arm M_bias v1: {d['v1_summary']['decision']['deploy']} (failed: "
        + "; ".join(d["v1_summary"]["decision"]["core_fail"]) + ").",
        "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I14-1 Part B of g v2 with a channel-current replica, clamp state C, element-identity Q2: "
               f"{D['outcome']} (pre-registered rule).",
               f"Only failed core check: 2 gmin-stepping warning lines in 1 of 126 element decks (nmos_L8_W1p6); "
               f"asymmetry {s2['swap_asym_max']:.1e}, Q2 {s2['q2_ac_max']:.1e}, Q1 0 violations.",
               f"Element-deck RRMS v2 {A['v2']['summary_native']['all_device_mean']:.4f} (stored "
               f"{A['v2']['summary_stored']['all_device_mean']:.4f})."]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
