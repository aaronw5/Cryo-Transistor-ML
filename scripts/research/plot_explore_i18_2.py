#!/usr/bin/env python3
"""Report batch for I18-2: the selected g v2 recipe on measured TU Delft 40-nm
MOSFETs (I6-1 data, BSIM4 CMOSedu card), 4.13 K and 295.0 K, 45 DEV devices,
leave-one-geometry-group-out, BSIM4 clamp state C; Part B (I17-1 rules,
amendment A1) with ngspice-41 OSDI.

Reads stored files under ``out/research/explore/I18_2``.  NGSpice
(ngspice-41, state C): the I-V figures re-simulate the stock card, the
C0_refit, M_bias v1 and selected held-out cards (g applied in Python); every
RRMS is checked against the stored rows.

  .venv/bin/python -I scripts/research/plot_explore_i18_2.py --date 2026-10-22
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))
import plot_explore_e2_e3_e4 as pe  # noqa: E402
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA, VIOLET, RED, DPI,  # noqa: E402,F401
                                   EXP, load, rel, style)
from matplotlib.lines import Line2D  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i18_2"
I18 = EXP / "I18_2"
LAMS = (0.0, 1.0, 2.0, 5.0, 10.0)
TEMPS = ("4", "295")
TLAB = {"4": "4.13 K", "295": "295.0 K"}
COMMITS = {"prereg": "2454b15", "amendment": "1795e10", "code": "254e77d"}
ARMS = ("C0_refit", "M_bias_v1", "sym_e1_l0", "sym_e1_l1", "sym_e1_l2", "sym_e1_l5", "sym_e1_l10", "selected")
COLOR = {"C0_refit": BLUE, "M_bias_v1": ORANGE, "selected": AQUA, "stock": GREY}


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext"):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": None}


def check(a: float, b: float, what: str, tol: float = 1e-9) -> None:
    if not abs(a - b) <= tol:
        raise AssertionError(f"{what}: {a} vs {b}")


def group_of(t: str) -> str:
    return t.rsplit("_d", 1)[0]


def short(t: str) -> str:
    return t.replace("t40_", "")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_setup(d: dict) -> rf.Table:
    P, S = d["prereg"], d["start"]
    rows = [{"c": "data", "v": P["data"]["split"]}, {"c": "card", "v": P["card"]},
            {"c": "simulator", "v": P["simulator"]}, {"c": "folds", "v": P["folds"]},
            {"c": "penalty grid", "v": P["penalty_grid"]["axes"]},
            {"c": "BSIM4 clamp", "v": P["bsim4_state"]["rule"]}]
    for t in TEMPS:
        s = S["per_temperature"][t]
        rows.append({"c": f"{TLAB[t]}: starting cards clamped (a2 > 1)", "v": f"{s['n_clamped']} / {len(s['rows'])}"})
        rows.append({"c": f"{TLAB[t]}: starting-card RRMS state L -> state C (all-device)",
                     "v": f"{s['start_L']['all_device_mean']:.5f} -> {s['start_C']['all_device_mean']:.5f}"})
        rows.append({"c": f"{TLAB[t]}: b4check messages (unclamped cards)",
                     "v": f"{len(s['b4check_messages_unclamped'])} distinct lines, all 'A2 ... larger than 1'"
                     if all("A2" in x for x in s["b4check_messages_unclamped"])
                     else "; ".join(s["b4check_messages_unclamped"][:4])})
        rows.append({"c": f"{TLAB[t]}: b4check messages after the clamp", "v": str(len(s["b4check_messages_clamped"]))})
    rows.append({"c": "amendment A1", "v": d["amend"]["changes"]["part_B_vds0_check"]["amended"]})
    rows.append({"c": "Q3 circuits", "v": P["part_B"]["Q3"]})
    return rf.Table("01_table_i18_2_setup", "I18-2 setup, BSIM4 clamp state C audit, amendment A1",
                    [col("c", "item", kind="text"), col("v", "value", kind="text")], rows,
                    [f"Pre-registration commit {COMMITS['prereg']}; amendment A1 commit {COMMITS['amendment']}."])


def t_validity(d: dict) -> rf.Table:
    D, S = d["decision"], d["start"]
    v = D["validity"]
    b4 = d["base"]["4"]
    rows = [{"c": "V1a state-L re-simulation of 90 I6-1 starting cards, max |d RRMS|", "t": "<= 1e-9",
             "v": f"{S['v1a_max']:.1e} (n {S['v1a_n']})"},
            {"c": "V1b state C = state L on cards with a2 in [0.01, 1]", "t": "identical", "v":
             f"{'identical' if S['v1b_max'] == 0 else 'differ'} (n {S['v1b_n']})"},
            {"c": "V1c state-L refits vs I6-1 rows (C0_refit, M_bias v1; 2 devices, 4.13 K)", "t": "<= 1e-9",
             "v": f"{b4['v1c_max']:.1e} (n {b4['v1c_n']})"},
            {"c": "V2 unit tests", "t": "pass", "v": str(all(d["base"][t]["unit_tests_passed"] for t in TEMPS))},
            {"c": "V3 rows per arm and temperature", "t": "45",
             "v": ", ".join(sorted({str(len(x)) for t in TEMPS for x in D["per_temperature"][t]["rows"].values()}))},
            {"c": "V3 inner / full selection complete", "t": "yes",
             "v": ", ".join(f"{TLAB[t]} {d['inner'][t]['complete']} / {d['full'][t]['complete']}" for t in TEMPS)},
            {"c": "V4 penalty grids 45 x 45 finite, forward corner > 0", "t": "90", "v": str(d["pens"]["v4_ok"])}]
    for t in TEMPS:
        if t in d["partB"]:
            vb = d["partB"][t]["validity"]
            rows.append({"c": f"VB Part B {TLAB[t]} (45 M2 refits, 0 failed; unit tests; 45 rows)", "t": "pass",
                         "v": "pass" if vb["valid"] else "; ".join(vb["fails"])})
    return rf.Table("02_table_i18_2_validity", "I18-2 validity", [col("c", "check", kind="text"), col("t", "criterion"),
                    col("v", "value", kind="text")], rows,
                    ["Result: " + ("valid." if v["valid"] else "invalid: " + "; ".join(v["fails"]))])


def t_rules(d: dict) -> rf.Table:
    D = d["decision"]
    rows = []
    for t in TEMPS:
        x = D["per_temperature"][t]
        pr, pc = D["predictions"], x["prediction_signs_correct"]
        p, s, g1 = x["primary_gmr_selected_vs_C0_refit"], x["secondary_gmr_selected_vs_v1"], x["gmr_v1_vs_C0_refit"]
        n = p["n"]
        rows += [
            {"T": TLAB[t], "q": "primary GMR(selected, C0_refit)", "v": ci(p), "w": f"{p['wins']}/{n}",
             "o": x.get("outcome", ""), "pr": f"{pr['ln GMR(selected, C0_refit)'][t]}: "
             + ("correct" if pc["ln GMR(selected, C0_refit)"] else "wrong")},
            {"T": TLAB[t], "q": "secondary GMR(selected, M_bias v1)", "v": ci(s), "w": f"{s['wins']}/{n}",
             "o": x.get("secondary", ""), "pr": f"{pr['ln GMR(selected, M_bias v1)'][t]}: "
             + ("correct" if pc["ln GMR(selected, M_bias v1)"] else "wrong")},
            {"T": TLAB[t], "q": "GMR(M_bias v1, C0_refit) (reported)", "v": ci(g1), "w": f"{g1['wins']}/{n}",
             "o": "-", "pr": f"{pr['ln GMR(M_bias v1, C0_refit)'][t]}: "
             + ("correct" if pc["ln GMR(M_bias v1, C0_refit)"] else "wrong")}]
        if t in d["partB"]:
            b = d["partB"][t]
            rows.append({"T": TLAB[t], "q": "Part B label", "v": "-", "w": "-",
                         "o": b["decision"]["label_full"] if b["validity"]["valid"] else "invalid",
                         "pr": f"{pr['part_B'][t]}: " + ("correct" if b["decision"]["label_full"] == pr["part_B"][t]
                                                         else "wrong")})
    return rf.Table("03_table_i18_2_rules", "I18-2 pre-registered rules per temperature: values, labels, predicted signs",
                    [col("T", "T", kind="text"), col("q", "quantity", kind="text"), col("v", "GMR [95% CI]"),
                     col("w", "lower"), col("o", "label", kind="text"), col("pr", "prediction", kind="text")], rows,
                    ["GMR: exp(mean ln(method / control)) over 45 DEV devices; cluster bootstrap over the 7 geometry "
                     "groups, 10,000 resamples, 95% CI. Primary: CI upper < 1 and 0 failed: gain replicated; CI lower "
                     "> 1: worse; else inconclusive. Secondary: CI upper <= 1.05: non-inferior."])


def t_arms(d: dict) -> rf.Table:
    D = d["decision"]
    rows = []
    for t in TEMPS:
        x = D["per_temperature"][t]
        for a in ARMS:
            r = x["table"][a]
            rows.append({"T": TLAB[t], "a": a, "m": f"{r['all_device_mean']:.5f}", "c": f"{r['combined']:.5f}",
                         "md": f"{r['median']:.5f}",
                         "g0": ci(r["gmr_vs_C0_refit_cluster"]) if "gmr_vs_C0_refit_cluster" in r else "1",
                         "g0d": ci(r["gmr_vs_C0_refit_device"]) if "gmr_vs_C0_refit_device" in r else "-",
                         "g1": ci(r["gmr_vs_v1_cluster"]) if "gmr_vs_v1_cluster" in r else "1"})
        st = x["state_L_I6_1"]
        rows.append({"T": TLAB[t], "a": "I6-1 stored (state L) C0_refit / M_bias",
                     "m": f"{st['C0_refit']['all_device_mean']:.5f} / {st['M_bias']['all_device_mean']:.5f}",
                     "c": "", "md": "", "g0": "", "g0d": "", "g1": ""})
    return rf.Table("04_table_i18_2_arms", "I18-2 held-out RRMS of every arm (45 TU Delft DEV devices, state C)",
                    [col("T", "T", kind="text"), col("a", "arm", kind="text"), col("m", "all-device"),
                     col("c", "combined"), col("md", "median"), col("g0", "GMR vs C0_refit (cluster CI)"),
                     col("g0d", "device CI"), col("g1", "GMR vs M_bias v1 (cluster CI)")], rows,
                    ["Strict paper RRMS on the I6-1 frozen measured inclusion; held-out refit from the I6-1 starting "
                     "card with g frozen (max_nfev 400, irls 2), clamp state C."])


def t_groups(d: dict) -> rf.Table:
    D = d["decision"]
    rows = []
    for t in TEMPS:
        x = D["per_temperature"][t]
        for g, a in x["per_group"].items():
            rows.append({"T": TLAB[t], "g": short(g), "n": str(a["C0_refit"]["n"]),
                         "c0": f"{a['C0_refit']['all_device_mean']:.5f}", "v1": f"{a['M_bias_v1']['all_device_mean']:.5f}",
                         "s": f"{a['selected']['all_device_mean']:.5f}", "l": f"{x['lambda_star'][g]:g}"})
    return rf.Table("05_table_i18_2_groups", "I18-2 per geometry group (outer fold): mean held-out RRMS and lambda*",
                    [col("T", "T", kind="text"), col("g", "group", kind="text"), col("n", "devices"),
                     col("c0", "C0_refit"), col("v1", "M_bias v1"), col("s", "selected"), col("l", "lambda* (V)")],
                    rows, [])


def t_lambda(d: dict) -> rf.Table:
    D = d["decision"]
    rows = []
    for t in TEMPS:
        x = D["per_temperature"][t]
        for g in sorted(x["inner_scores"]):
            sc = x["inner_scores"][g]
            rows.append({"T": TLAB[t], "g": short(g), **{f"l{int(lam)}": f"{sc[f'{lam:g}']:.4f}" for lam in LAMS},
                         "s": f"{x['lambda_star'][g]:g}"})
        sc = x["full_scores"]
        rows.append({"T": TLAB[t], "g": "full data (Part B model)",
                     **{f"l{int(lam)}": f"{sc[f'{lam:g}']:.4f}" for lam in LAMS}, "s": f"{x['lambda_star_full']:g}"})
    return rf.Table("06_table_i18_2_lambda", "I18-2 nested selection: inner score (mean ln RRMS of the inner held-out "
                    "refits) per outer fold and lambda",
                    [col("T", "T", kind="text"), col("g", "held-out group", kind="text")]
                    + [col(f"l{int(lam)}", f"lambda {lam:g} V") for lam in LAMS] + [col("s", "lambda* (V)")], rows,
                    ["Inner folds: training groups sorted by (polarity, name), group i -> fold i mod 3; one member "
                     "(seed 0); ties to the larger lambda."])


def t_neg(d: dict) -> rf.Table:
    rows = []
    for t in TEMPS:
        N = d["decision"]["per_temperature"][t]["negative_fractions"]
        for a in ("M_bias_v1", "sym_e1_l0", "sym_e1_l1", "sym_e1_l2", "sym_e1_l5", "sym_e1_l10", "selected"):
            n = N[a]
            rows.append({"T": TLAB[t], "a": a, "hgds": f"{100 * n['heldout_mean_gds']:.2f}",
                         "hgm": f"{100 * n['heldout_mean_gm']:.2f}", "tgds": f"{100 * n['train_mean_gds']:.2f}",
                         "tgm": f"{100 * n['train_mean_gm']:.2f}"})
        c = N["card_only"]
        rows.append({"T": TLAB[t], "a": "card only", "hgds": f"{100 * c['frac_neg_gds_card']:.3f}",
                     "hgm": f"{100 * c['frac_neg_gm_card']:.3f}", "tgds": "", "tgm": ""})
    return rf.Table("07_table_i18_2_negative", "I18-2 negative corrected gds / gm on the penalty grids (% of points)",
                    [col("T", "T", kind="text"), col("a", "arm", kind="text"), col("hgds", "held-out gds < 0"),
                     col("hgm", "held-out gm < 0"), col("tgds", "training gds < 0"), col("tgm", "training gm < 0")],
                    rows, ["Grid 45 x 45, forward quadrant 0 .. 1.1 V, forward current > 1 nA; mean over folds."])


def t_partb(d: dict) -> rf.Table:
    from cryoml.explore.i12_1_symm import THR
    keys = [("q1_tight_viol", "Q1 tight equivalence violations", "0", "core"),
            ("q1_bad", "Q1 element runs: messages + non-finite", "0", "core"),
            ("q1_vds0_max", "Q1 element-added |I| at V(d,s) = 0, tight (A1)", f"<= {THR['q1_vds0_abs']:g} A", "core"),
            ("g_text_max", "Verilog-A g vs Python", f"<= {THR['g_text']:g}", "core"),
            ("g_sym_text_max", "Verilog-A g exchanged vs forward", f"<= {THR['g_sym_text']:g}", "core"),
            ("swap_asym_max", "exchange asymmetry", f"<= {THR['swap_asym']:g}", "core"),
            ("q2_ac_max", "Q2 derivative vs element identity", f"<= {THR['q2_ac']:g}", "core"),
            ("cap_max", "Q2 capacitance error", f"<= {THR['cap']:g}", "core"),
            ("q2_gm_sign", "gm curves with introduced sign change", "0", "core"),
            ("q1_default_max", "default-tolerance relative difference", f"<= {THR['q1_default_rel']:g}", "restriction"),
            ("q2_gds_sign", "gds curves with introduced sign change", "0", "restriction"),
            ("runtime_ratio_max", "runtime ratio, DC grid", f"<= {THR['runtime_ratio']:g}", "restriction")]
    rows = []
    for k, lab, thr, kind in keys:
        r = {"c": lab, "k": kind, "t": thr}
        for t in TEMPS:
            v = d["partB"].get(t, {}).get("values", {}).get(k)
            r[f"v{t}"] = "-" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))
        rows.append(r)
    extra = [("vds0_raw_element_tight_max", "raw element |I| at V(d,s) = 0, tight (reported)"),
             ("vds0_card_max", "card + replica |I| at V(d,s) = 0 (reported)"),
             ("nonchannel_share_fwd_max", "card non-channel share of |I|, forward (reported)")]
    for k, lab in extra:
        rows.append({"c": lab, "k": "reported", "t": "-",
                     **{f"v{t}": (f"{d['partB'][t][k]:.3g}" if t in d["partB"] else "-") for t in TEMPS}})
    rows.append({"c": "Q3 circuits", "k": "core", "t": "-", **{f"v{t}": "not run" for t in TEMPS}})
    rows.append({"c": "label", "k": "", "t": "", **{f"v{t}": (d["partB"][t]["decision"]["label_full"]
                                                               if t in d["partB"] else "-") for t in TEMPS}})
    notes = []
    for t in TEMPS:
        if t in d["partB"]:
            b = d["partB"][t]
            notes.append(f"{TLAB[t]}: lambda*_full {b['lambda_star_full']:g} V; core failures "
                         f"{b['decision']['core_fail'] or 'none'}; restrictions {b['decision']['restrictions'] or 'none'};"
                         f" gds curves {b['gds_curves_introduced']}; replica non-channel outputs max "
                         f"{max(b['replica_nonchannel_max'].values()) if b['replica_nonchannel_max'] else 'n/a'}.")
    return rf.Table("08_table_i18_2_partb", "I18-2 Part B (I17-1 rules, amendment A1) per temperature, 45 devices",
                    [col("c", "check", kind="text"), col("k", "kind"), col("t", "threshold"),
                     col("v4", "4.13 K"), col("v295", "295.0 K")], rows, notes)


def t_partb_acc(d: dict) -> rf.Table:
    rows = []
    for t in TEMPS:
        a = d.get("acc", {}).get(t)
        if not a:
            continue
        for k, lab in (("M2", "M2 cards + full-data g (in sample)"), ("sel", "selected held-out pairs")):
            x = a[k]
            rows.append({"T": TLAB[t], "p": lab, "n": f"{x['mean_native']:.5f}", "py": f"{x['mean_python_rerun']:.5f}",
                         "st": f"{x['mean_stored_python']:.5f}" if x["mean_stored_python"] is not None else "-",
                         "cd": f"{x['mean_card_only']:.5f}", "dm": f"{x['max_abs_native_minus_stored']:.1e}"
                         if x["max_abs_native_minus_stored"] is not None else "-", "ms": str(x["n_msgs"])})
    return rf.Table("09_table_i18_2_partb_accuracy", "I18-2 element-deck RRMS (ngspice-41 OSDI) vs Python rows "
                    "(reported, no rule)", [col("T", "T", kind="text"), col("p", "pairs", kind="text"),
                                           col("n", "element deck"), col("py", "Python rerun"),
                                           col("st", "stored Python"), col("cd", "card only"),
                                           col("dm", "max |element - stored|"), col("ms", "messages")], rows,
                    ["All-device means over 45 devices. Element deck: card + replica + OSDI module on the measured "
                     "sweeps; Python: card x exp(g)."])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_per_device(d: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, t in zip(axes, TEMPS):
        style(ax)
        x = d["decision"]["per_temperature"][t]
        R = x["rows"]
        tags = sorted(R["C0_refit"])
        off = np.linspace(-0.25, 0.25, 3)
        for j, a in enumerate(("C0_refit", "M_bias_v1", "selected")):
            ax.scatter(np.arange(len(tags)) + off[j], [R[a][g] for g in tags], color=COLOR[a], s=18, zorder=3,
                       label=f"{a} (all-device {x['table'][a]['all_device_mean']:.4f})")
        for k in range(1, len(tags)):
            if group_of(tags[k]) != group_of(tags[k - 1]):
                ax.axvline(k - 0.5, color=MUTED, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(tags)))
        ax.set_xticklabels([short(g) for g in tags], rotation=60, ha="right", fontsize=6.5)
        ax.set_ylabel("held-out RRMS (log)", fontsize=9)
        ax.set_title(f"I18-2 TU Delft {TLAB[t]}: held-out RRMS per DEV device (vertical lines: geometry groups)",
                     fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I18 / "decision.json")]


def fig_ratio(d: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, t in zip(axes, TEMPS):
        style(ax)
        x = d["decision"]["per_temperature"][t]
        R = x["rows"]
        gs = sorted(x["per_group"])
        for j, a in enumerate(("M_bias_v1", "selected")):
            for k, g in enumerate(gs):
                tags = [u for u in R["C0_refit"] if group_of(u) == g]
                y = [R[a][u] / R["C0_refit"][u] for u in tags]
                ax.scatter(np.full(len(y), k + (-0.15 if j == 0 else 0.15)), y, color=COLOR[a], s=16,
                           label=a if k == 0 else None)
        ax.axhline(1, color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(gs)))
        ax.set_xticklabels([f"{short(g)}\nlambda* {x['lambda_star'][g]:g}" for g in gs], rotation=30, ha="right",
                           fontsize=7.5)
        ax.set_ylabel("RRMS / C0_refit RRMS (log)")
        p = x["primary_gmr_selected_vs_C0_refit"]
        ax.set_title(f"{TLAB[t]}: GMR(selected, C0_refit) {ci(p)}", fontsize=9.5, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("I18-2 per-device ratio to C0_refit by geometry group (outer fold)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I18 / "decision.json")]


def fig_lambda_curve(d: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, t in zip(axes, TEMPS):
        style(ax)
        T = d["decision"]["per_temperature"][t]["table"]
        xs = np.arange(len(LAMS))
        for key, colr, lab in (("gmr_vs_C0_refit_cluster", BLUE, "GMR vs C0_refit"),
                               ("gmr_vs_v1_cluster", ORANGE, "GMR vs M_bias v1")):
            g = [T[f"sym_e1_l{int(lam)}"][key] for lam in LAMS] + [T["selected"][key]]
            xx = list(xs) + [len(LAMS) + 0.3]
            ax.errorbar(xx, [z["ratio"] for z in g], yerr=[[z["ratio"] - z["ci_low"] for z in g],
                                                           [z["ci_high"] - z["ratio"] for z in g]],
                        color=colr, marker="o", capsize=3, label=lab, ls="none")
        ax.axhline(1, color=INK, lw=0.8)
        ax.axhline(1.05, color=GREY, lw=0.8, ls="--")
        ax.set_xticks(list(xs) + [len(LAMS) + 0.3])
        ax.set_xticklabels([f"{lam:g}" for lam in LAMS] + ["selected"])
        ax.set_xlabel("penalty lambda (V)")
        ax.set_ylabel("GMR [95% cluster CI]")
        ax.set_title(f"{TLAB[t]}", fontsize=10, color=INK, loc="left")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("I18-2 lambda curve (dashed: margin 1.05)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I18 / "decision.json")]


def fig_partb(d: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for t, colr in zip(TEMPS, (BLUE, ORANGE)):
        if t not in d["q12"]:
            continue
        D = d["q12"][t]["devices"]
        tags = sorted(D)
        x = np.arange(len(tags))
        axes[0].scatter(x, [D[u]["q1_default_max_rel_1nA"] for u in tags], color=colr, s=12, label=TLAB[t])
        axes[1].scatter(x, [np.median(D[u]["wall"]["grid_fwd_default_elem"]) / np.median(D[u]["wall"]["grid_fwd_default_card"])
                            for u in tags], color=colr, s=12, label=TLAB[t])
        axes[2].scatter(x, [max(D[u]["q1_vds0_added_by_tol"]["tight"], 1e-30) for u in tags], color=colr, s=12,
                        label=f"{TLAB[t]} element-added")
        axes[2].scatter(x, [max(D[u]["q1_vds0_by_tol"]["tight"], 1e-30) for u in tags], color=colr, s=12, marker="x",
                        label=f"{TLAB[t]} raw element run")
    for ax, (thr, lab) in zip(axes, ((1e-3, "default-tolerance max relative difference (|I| > 1 nA)"),
                                     (2.0, "runtime ratio element / card, DC grid"),
                                     (1e-12, "|I| at V(d,s) = 0, tight (A)"))):
        style(ax)
        ax.axhline(thr, color=RED, lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("device index (sorted tag)")
        ax.set_title(lab, fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("I18-2 Part B per device (dashed: threshold)", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I18 / "partB_4K/q12.json"), rel(I18 / "partB_295K/q12.json")]


# ---------------------------------------------------------------------------
# I-V (NGSpice re-simulation, state C)
# ---------------------------------------------------------------------------
def iv_tags(d: dict, t: str) -> list[str]:
    R = d["decision"]["per_temperature"][t]["rows"]
    tags = sorted(R["C0_refit"])
    ratio = {u: R["selected"][u] / R["C0_refit"][u] for u in tags}
    s = sorted(tags, key=lambda u: ratio[u])
    return [s[0], s[-1]]


def resim(base: Path, dec_t: dict, t: int, tags) -> dict:
    """Re-simulate stock, C0_refit, M_bias v1 (x exp g) and selected (x exp g)
    held-out cards of ``tags`` at temperature ``t`` (state C); checks RRMS."""
    import i18_2_tud as drv
    from cryoml.explore import i6_1_tudelft as m6
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i3_2_bias import correction_from_json
    from cryoml.explore.i12_1_symm import SymCorrection
    cards = load(base / f"base_{t}K/heldout_cards.json")
    bc = load(base / f"base_{t}K/corrections.json")
    out = {}
    for tag in tags:
        dev = drv.devices()[tag]
        curves = m6.fit_curves(dev, t)
        temp = dev.temps_K[t]
        g = group_of(tag)
        sims, sc = {}, {}
        stock = m6.stock_params(dev, m6.BASE_SET)
        s0 = drv.sim(dev, curves, temp, stock)
        rec0 = m6.to_record(dev, t, s0, stock)
        sims["stock"], sc["stock"] = s0, score_device(rec0, s0)["rrms"]
        c0 = dict(cards[f"C0_refit|{tag}"])
        s = drv.sim(dev, curves, temp, c0)
        sims["C0_refit"], sc["C0_refit"] = s, score_device(m6.to_record(dev, t, s, c0), s)["rrms"]
        check(sc["C0_refit"], dec_t["rows"]["C0_refit"][tag], f"{tag} C0_refit")
        p1 = dict(cards[f"M_bias|{tag}"])
        r1 = m6.to_record(dev, t, drv.sim(dev, curves, temp, p1), p1)
        g1 = correction_from_json(bc[f"M_bias|{g}"]).apply(r1)
        sims["M_bias_v1"], sc["M_bias_v1"] = g1, score_device(r1, g1)["rrms"]
        check(sc["M_bias_v1"], dec_t["rows"]["M_bias_v1"][tag], f"{tag} v1")
        lam = dec_t["lambda_star"][g]
        v = f"sym_e1_l{int(lam)}"
        ps = dict(load(base / f"lodo_{t}K_{v}/heldout_cards.json")[tag])
        corr = SymCorrection.from_json(load(base / f"lodo_{t}K_{v}/corrections.json")[f"{v}|{g}"])
        rs = m6.to_record(dev, t, drv.sim(dev, curves, temp, ps), ps)
        gs = corr.apply(rs)
        sims["selected"], sc["selected"] = gs, score_device(rs, gs)["rrms"]
        check(sc["selected"], dec_t["rows"]["selected"][tag], f"{tag} selected")
        out[tag] = {"rec": rec0, "sims": sims, "score": sc, "lam": lam}
    return out


def fig_iv(r: dict, tag: str, tlab: str, path: Path, extra_title: str = "",
           labels: dict | None = None) -> None:
    rec = r["rec"]
    lab = {"stock": "NGSpice, stock CMOSedu card (no fit)",
           "C0_refit": "NGSpice re-simulated, C0_refit held-out card (no correction)",
           "M_bias_v1": "NGSpice re-simulated, M_bias v1 held-out card x exp(fold g)",
           "selected": f"NGSpice re-simulated, selected held-out card x exp(fold g) (lambda* {r['lam']:g} V)"}
    lab.update(labels or {})
    lines = (("stock", GREY, "--"), ("C0_refit", BLUE, "-"), ("M_bias_v1", ORANGE, "-"), ("selected", AQUA, "-"))
    lines = tuple(x for x in lines if x[0] in lab and lab[x[0]])
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, kind in zip(axes, ("idvd", "idvg")):
        style(ax)
        for i, c in enumerate(rec.curves):
            if c.kind != kind:
                continue
            xv = np.abs(c.vd if kind == "idvd" else c.vg)
            alpha = 1.0 if c.included else 0.3
            meas = np.abs(c.target)
            s = slice(c.start, None)
            ax.scatter(xv[s], meas[s], s=6, color=INK, zorder=4, alpha=alpha, lw=0)
            if c.start:
                ax.scatter(xv[:c.start], meas[:c.start], s=6, facecolor="none", edgecolor=GREY, lw=0.5, zorder=4,
                           alpha=alpha)
            for a, colr, ls in lines:
                ax.plot(xv, np.abs(r["sims"][a][i]), color=colr, ls=ls, lw=1.0, alpha=alpha, zorder=3)
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
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=4,
                      label=f"measured {tlab} (filled: scored points; hollow: trimmed before the metric; faded: "
                            "excluded curve)")]
    for a, colr, ls in lines:
        handles.append(Line2D([], [], color=colr, ls=ls, lw=1.2, label=f"{lab[a]}: RRMS {sc[a]:.4f}"))
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag} (TU Delft 40 nm DEV, {tlab}): held-out cards (leave-one-group-out) re-simulated in "
                 f"ngspice-41, BSIM4 clamp state C; g applied in Python. RRMS equals the stored rows.{extra_title}",
                 fontsize=9.2, color=INK)
    fig.tight_layout(rect=(0, 0.2, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def main() -> int:
    global I18
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--smoke-out", default="")
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    if args.smoke_out:
        I18 = EXP / "I18_2/smoke"
        os.environ["CRYOML_I18_2_SMOKE"] = "1"
        folder = Path(args.smoke_out)
    import i18_2_tud as drv
    drv._SMOKE["on"] = bool(args.smoke_out)
    folder.mkdir(parents=True, exist_ok=True)
    d = {"prereg": load(EXP / "I18_2/prereg.json"), "amend": load(EXP / "I18_2/prereg_amendment_A1.json"),
         "decision": load(I18 / "decision.json"), "start": load(I18 / "start/start.json"),
         "pens": load(I18 / "pens/pens.json"),
         "base": {t: load(I18 / f"base_{t}K/results.json") for t in TEMPS},
         "inner": {t: load(I18 / f"inner_{t}K/inner.json") for t in TEMPS},
         "full": {t: load(I18 / f"full_{t}K/full.json") for t in TEMPS},
         "partB": {t: load(I18 / f"partB_{t}K/summary.json") for t in TEMPS
                   if (I18 / f"partB_{t}K/summary.json").exists()},
         "q12": {t: load(I18 / f"partB_{t}K/q12.json") for t in TEMPS if (I18 / f"partB_{t}K/q12.json").exists()},
         "acc": {t: load(I18 / f"partB_{t}K/accuracy.json") for t in TEMPS
                 if (I18 / f"partB_{t}K/accuracy.json").exists()}}
    from cryoml.explore.i3_2_bias import paired_ratio
    D = d["decision"]
    for t in TEMPS:
        x = D["per_temperature"][t]
        sel = {u: load(I18 / f"lodo_{t}K_sym_e1_l{int(x['lambda_star'][group_of(u)])}/results.json")["per_device"][u]
               for u in x["rows"]["C0_refit"]}
        for u in sel:
            check(sel[u], x["selected_rows"][u], f"selected row {u}", 0.0)
        g = paired_ratio(sel, x["rows"]["C0_refit"], clusters={u: group_of(u) for u in sel})
        check(g["ratio"], x["primary_gmr_selected_vs_C0_refit"]["ratio"], "primary GMR", 1e-12)
        check(g["ci_high"], x["primary_gmr_selected_vs_C0_refit"]["ci_high"], "primary CI", 1e-12)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    dj = rel(I18 / "decision.json")
    add_table(t_setup(d), "Setup, clamp state C audit, amendment A1, Q3 status", [rel(I18 / "start/start.json")])
    add_table(t_validity(d), "Validity V1-V4 and VB", [rel(I18 / "start/start.json"), dj])
    add_table(t_rules(d), "Primary / secondary GMR, Part B labels vs the pre-registered rules and predictions", [dj])
    add_table(t_arms(d), "Held-out RRMS of every arm, GMR with cluster and device CIs", [dj])
    add_table(t_groups(d), "Per geometry group", [dj])
    add_table(t_lambda(d), "Nested lambda selection and the full-data selection", [dj])
    add_table(t_neg(d), "Negative corrected gds / gm fractions", [dj])
    if d["partB"]:
        add_table(t_partb(d), "Part B core checks, restrictions and labels", ["I18_2/partB_<T>K/summary.json"])
        add_table(t_partb_acc(d), "Part B element-deck accuracy", ["I18_2/partB_<T>K/accuracy.json"])
    add_fig("10_i18_2_per_device.png", fig_per_device, "Per-device held-out RRMS (C0_refit, M_bias v1, selected)")
    add_fig("11_i18_2_ratio_by_group.png", fig_ratio, "Per-device ratio to C0_refit by group with lambda*")
    add_fig("12_i18_2_lambda_curve.png", fig_lambda_curve, "GMR vs lambda with cluster CIs")
    if d["q12"]:
        add_fig("13_i18_2_partb_per_device.png", fig_partb, "Part B per device: default tolerance, runtime, "
                "V(d,s) = 0 current")
    k = 14
    for t in TEMPS:
        tags = iv_tags(d, t)
        res = resim(I18, D["per_temperature"][t], int(t), tags)
        for tag in tags:
            name = f"{k:02d}_i18_2_iv_{t}K_{short(tag)}.png"
            fig_iv(res[tag], tag, TLAB[t], folder / name)
            items.append((name, f"I-V of {tag} at {TLAB[t]}: measured points, NGSpice of the stock card and "
                                "re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g))",
                          [f"I18_2/base_{t}K/heldout_cards.json", f"I18_2/lodo_{t}K_*/heldout_cards.json",
                           "I6_1/data/tudelft40.npz (not in git)"]))
            k += 1
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_explore_i18_2.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    summ = []
    for t in TEMPS:
        x = D["per_temperature"][t]
        p, s = x["primary_gmr_selected_vs_C0_refit"], x["secondary_gmr_selected_vs_v1"]
        lb = d["partB"][t]["decision"]["label_full"] if t in d["partB"] else "Part B pending"
        summ.append(f"I18-2 TU Delft {TLAB[t]} (45 DEV, 7 groups): {x.get('outcome')}; {x.get('secondary')}; "
                    f"GMR(selected, C0_refit) {ci(p)}; GMR(selected, v1) {ci(s)}; Part B: {lb}.")
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. NGSpice runs: re-simulation of the stock, C0_refit, M_bias v1 and selected "
              "cards for the I-V devices (checked against the stored rows); everything else from stored files.", "",
              "## Contents", "",
              "- I18-2: selected g v2 recipe on measured TU Delft 40 nm (I6-1 data), 4.13 K and 295.0 K, 45 DEV "
              f"devices, leave-one-group-out, BSIM4 clamp state C; Part B with I17-1 rules (amendment A1). Pre-"
              f"registration commit {COMMITS['prereg']}, amendment {COMMITS['amendment']}, code {COMMITS['code']}.",
              *[f"- {x}" for x in summ], "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    if not args.smoke_out:
        rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summ)
    print("\n".join(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
