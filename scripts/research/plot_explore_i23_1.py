#!/usr/bin/env python3
"""Report batch for I23-1: (A) call-matched IHP control for B1, (B) g v2 with
lambda fixed at 1 V (no nested selection) on the 9 measured DEV datasets.

Reads ``out/research/explore/I23_1/`` (prereg, partA_<ihp>/, verify_<ds>.json,
fixl1_<ds>/, decision.json), the I22-1 rows / cards / cost table, and the
stored lambda = 1 V and selected g v2 files.

NGSpice: the I-V figures re-simulate the published / stock card and the held-out
cards (Part A: C0_refit, C0_call, B1, B1_cap; Part B: C0_refit, selected g v2,
fixed lambda 1 V) with the simulator and state of each dataset; g is applied in
Python.  Every re-simulated RRMS is checked against the stored row.

  .venv/bin/python -I scripts/research/plot_explore_i23_1.py --date 2026-10-27
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "explore"))
import plot_explore_i22_1 as p22  # noqa: E402  (style, simulation, I-V drawing)
from plot_explore_e2_e3_e4 import (BG, INK, GREY, BLUE, ORANGE, AQUA, VIOLET, RED, DPI,  # noqa: E402
                                   EXP, load, rel, style)
from matplotlib.lines import Line2D  # noqa: E402

from cryoml.explore import i22_1_baselines as m22  # noqa: E402
from cryoml.explore import i23_1_budget as m  # noqa: E402

rf = p22.rf
plt = p22.plt
ci = p22.ci
col = p22.col
check = p22.check
kfmt = p22.kfmt

TOPIC = "i23_1"
I23 = EXP / "I23_1"
I22 = EXP / "I22_1"
COMMITS = {"prereg": "a9aa227", "code": "73ce90e"}
A_COLOR = {"C0_refit": BLUE, "C0_call": "#1d3557", "B1": VIOLET, "B1_cap": "#c8b6ff", "selected_g_v2": AQUA}
A_LABEL = {"C0_refit": "C0_refit (max_nfev 400)", "C0_call": "C0_call (calls = B1)", "B1": "B1 wider card set",
           "B1_cap": "B1_cap (calls = C0_refit)", "selected_g_v2": "selected g v2"}
B_COLOR = {"fixed_l1": RED, "selected_g_v2": AQUA, "C0_refit": BLUE, "M_bias_v1": ORANGE, "B1": VIOLET}
B_LABEL = {"fixed_l1": "fixed lambda 1 V", "selected_g_v2": "selected g v2 (nested lambda)", "C0_refit": "C0_refit",
           "M_bias_v1": "M_bias v1", "B1": "B1 wider card set"}
A_CMP = (("B1_vs_C0_call", "GMR(B1, C0_call) (primary)"), ("C0_call_vs_C0_refit", "GMR(C0_call, C0_refit)"),
         ("B1_vs_C0_refit", "GMR(B1, C0_refit) (I22-1)"), ("B1_cap_vs_C0_refit", "GMR(B1_cap, C0_refit)"),
         ("B1_vs_B1_cap", "GMR(B1, B1_cap)"), ("C0_call_vs_selected_g_v2", "GMR(C0_call, selected g v2)"))
NAME = {**m22.NAMES}


def is_p(t: str) -> bool:
    return "pmos" in t or t.startswith("pfet")


# ---------------------------------------------------------------------------
def load_all() -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    d = {"prereg": load(I23 / "prereg.json"), "dec": load(I23 / "decision.json")}
    A, B = d["dec"]["part_A"], d["dec"]["part_B"]
    n = 0
    for ds in m.IHP:
        R = A["rows"][ds]
        g = paired_ratio(R["B1"], R["C0_call"])
        for k in ("ratio", "ci_low", "ci_high"):
            check(g[k], A["datasets"][ds]["comparisons"]["B1_vs_C0_call"][k], f"{ds} B1 vs C0_call {k}", 1e-12)
            n += 1
        i22 = load(I22 / f"partA_{ds}/results.json")["per_device"]
        for t in R["B1"]:
            check(R["B1"][t], i22["B1"][t], f"{ds} B1 {t}", 0.0)
            check(R["C0_refit"][t], i22["C0_refit"][t], f"{ds} C0 {t}", 1e-12)
            n += 2
    for ds in m.DATASETS:
        R = B["rows"][ds]
        cl = None
        if ds.startswith("tud"):
            from cryoml.explore.i6_1_tudelft import group_of
            cl = {t: group_of(t) for t in R["C0_refit"]}
        g = paired_ratio(R["fixed_l1"], R["selected_g_v2"], cl)
        for k in ("ratio", "ci_low", "ci_high"):
            check(g[k], B["datasets"][ds]["comparisons"]["fixed_vs_selected"][k], f"{ds} fixed vs sel {k}", 1e-12)
            n += 1
        s22 = load(I22 / "decision.json")["part_A"]["rows"][ds]["selected_g_v2"]
        for t, v in s22.items():
            check(v, R["selected_g_v2"][t], f"{ds} selected {t}", 0.0)
            n += 1
    d["n_checked"] = n
    d["partA"] = {ds: load(I23 / f"partA_{ds}/results.json") for ds in m.IHP}
    d["verify"] = {ds: load(I23 / f"verify_{ds}.json") for ds in m.REUSED}
    return d


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_setup(d: dict) -> rf.Table:
    P = d["prereg"]
    A, B = P["part_A"], P["part_B"]
    rows = [{"c": "pre-registration", "v": f"out/research/explore/I23_1/prereg.md (commit {COMMITS['prereg']}); code "
                                         f"{COMMITS['code']}; DEV only, TEST locked; no amendment of a755e18"},
            {"c": "Part A data", "v": f"{', '.join(A['datasets'])}; {A['records']}; {A['simulator']}"},
            {"c": "Part A call unit", "v": A["call_unit"]},
            {"c": "Part A arms", "v": "; ".join(f"{k}: {v}" for k, v in A["arms"].items())},
            {"c": "Part A capped refit", "v": A["capped_refit"]},
            {"c": "Part A rule", "v": "; ".join(f"{k}: {v}" for k, v in A["rule"].items())},
            {"c": "Part B arm", "v": B["arm"]},
            {"c": "Part B rows reused", "v": "; ".join(f"{k}: {v}" for k, v in B["reused"].items())},
            {"c": "Part B rows new", "v": "; ".join(f"{k}: {v}" for k, v in B["new"].items())},
            {"c": "Part B rule", "v": f"gain set {', '.join(B['gain_set'])}; " +
                                      "; ".join(f"{k}: {v}" for k, v in B["rule"].items())}]
    return rf.Table("01_table_i23_1_setup", "I23-1 setup (pre-registered)",
                    [col("c", "item", kind="text"), col("v", "value", kind="text")], rows, [])


def t_a_validity(d: dict) -> rf.Table:
    rows = []
    for ds in m.IHP:
        v = d["dec"]["part_A"]["datasets"][ds]["validity"]
        va3 = v["VA3_detail"]
        rows.append({"ds": NAME[ds], "va1": f"{v['VA1_max_abs_diff']:.1e}; n_sim equal {v['VA1_n_sim_equal']}",
                     "va2": ", ".join(f"{a} {k}" for a, k in v["VA2_n_failed"].items() if k) or "0 failed",
                     "va3": f"C0_call over limit {len(va3['C0_call']['over'])}, B1_cap over limit "
                            f"{len(va3['B1_cap']['over'])}",
                     "va4": str(v["VA4"]), "va5": str(v["VA5_unit_tests"]), "res": "valid" if v["valid"] else "invalid"})
    return rf.Table("02_table_i23_1_a_validity", "Part A validity per temperature (VA1-VA5)",
                    [col("ds", "temperature", kind="text"),
                     col("va1", "VA1 max |C0_refit rerun via polish_capped - stored| (<= 1e-12)"),
                     col("va2", "VA2 non-finite rows"), col("va3", "VA3 calls <= limit"), col("va4", "VA4 DEV only"),
                     col("va5", "VA5 unit tests"), col("res", "result", kind="text")], rows,
                    ["VA1: the I22-1 C0_refit refit through polish_capped with no limit against the stored rows."])


def t_a_rrms(d: dict) -> rf.Table:
    rows = []
    arms = ("C0_refit", "C0_call", "B1", "B1_cap", "selected_g_v2")
    for ds in m.IHP:
        s = d["dec"]["part_A"]["datasets"][ds]["summary"]
        rows.append({"ds": NAME[ds], **{a: f"{s[a]['all_device_mean']:.5f} / {s[a]['combined']:.5f}" for a in arms}})
    return rf.Table("03_table_i23_1_a_rrms", "Part A held-out RRMS per arm (18 DUTs per temperature): all-device "
                    "mean / combined (mean of the nMOS and pMOS family means)",
                    [col("ds", "temperature", kind="text")] + [col(a, A_LABEL[a]) for a in arms], rows,
                    ["C0_refit, B1: stored I22-1 rows; selected g v2: stored rows; C0_call, B1_cap: I23-1. Strict "
                     "paper RRMS, frozen curve inclusion."])


def t_a_gmr(d: dict) -> rf.Table:
    A = d["dec"]["part_A"]
    rows = []
    for ds in m.IHP:
        a = A["datasets"][ds]
        rows.append({"ds": NAME[ds], **{k: f"{ci(a['comparisons'][k])} {a['comparisons'][k]['wins']}/"
                                            f"{a['comparisons'][k]['n_used']}; {a['labels'][k]}" for k, _ in A_CMP}})
    rows.append({"ds": "pooled exp(mean ln GMR), 4 temperatures",
                 **{k: f"{A['pooled_over_4_temperatures'][k]:.3f}" for k, _ in A_CMP}})
    notes = [f"Rule outcome: {A['rule']['label']} (temperatures with CI upper < 1: "
             f"{', '.join(A['rule']['gain_temperatures']) or 'none'})."]
    P = d["dec"]["predictions"]
    notes.append(f"P1 {P['P1']}; P2 {P['P2']}; P3 {P['P3']}.")
    return rf.Table("04_table_i23_1_a_gmr", "Part A GMR with 95% bootstrap CI, DUTs lower / paired, label (X better: "
                    "CI upper < 1; Y better: CI lower > 1)",
                    [col("ds", "temperature", kind="text")] + [col(k, h) for k, h in A_CMP], rows, notes)


def t_a_calls(d: dict) -> rf.Table:
    rows = []
    for ds in m.IHP:
        c = d["dec"]["part_A"]["datasets"][ds]["calls"]
        rows.append({"ds": NAME[ds], "b1": f"{c['B1_refit_calls_per_device']:.0f}",
                     "sc": kfmt(c["B1_screen_calls_per_device"]),
                     "cc": f"{c['C0_call_calls_per_device']:.0f}", "fr": f"{c['C0_call']['mean_fraction_used']:.3f}",
                     "nc": f"{c['n_capped']['C0_call']}/18", "c0": f"{c['C0_refit_calls_per_device']:.0f}",
                     "bc": f"{c['B1_cap_calls_per_device']:.0f}", "nb": f"{c['n_capped']['B1_cap']}/18"})
    return rf.Table("05_table_i23_1_a_calls", "Part A NGSpice calls per DUT (held-out refit, incl. start, Jacobian "
                    "columns, IRLS and final re-simulations)",
                    [col("ds", "temperature", kind="text"), col("b1", "B1 refit calls (= C0_call limit)"),
                     col("sc", "B1 screen calls / DUT (stored I5-1, not in the limit)"),
                     col("cc", "C0_call calls used"), col("fr", "C0_call mean fraction of limit"),
                     col("nc", "C0_call DUTs at the limit"), col("c0", "C0_refit calls (= B1_cap limit)"),
                     col("bc", "B1_cap calls used"), col("nb", "B1_cap DUTs at the limit")], rows,
                    ["A DUT 'at the limit' stopped because the call limit was reached; otherwise TRF ended by its own "
                     "criteria or its iteration budget (max_nfev) before the limit."])


def t_b_validity(d: dict) -> rf.Table:
    rows = []
    for ds in m.DATASETS:
        b = d["dec"]["part_B"]["datasets"][ds]
        v = b["validity"]
        rows.append({"ds": NAME[ds], "src": "; ".join(b["sources"]), "nr": f"{b['n_reused']} / {b['n_new']}",
                     "vb1": (f"{v['VB1']} ({v['VB1_max_abs_diff']:.1e}; {', '.join(v['VB1_samples'])})"
                             if "VB1" in v else "-"),
                     "vb2": (f"{v['VB2']} ({v['VB2_overlap_n']} folds; max {v['VB2_max_abs_diff']:.1e})"
                             if v.get("VB2_overlap_n") else ("True (no overlap fold)" if "VB2" in v else "-")),
                     "vb3": f"{v['VB3']} ({v['VB3_n_failed']} failed)", "vb4": str(v["VB4"]),
                     "res": "valid" if v["valid"] else "invalid"})
    return rf.Table("06_table_i23_1_b_validity", "Part B row sources and validity per dataset (VB1-VB5)",
                    [col("ds", "dataset", kind="text"), col("src", "fixed lambda 1 V rows from", kind="text"),
                     col("nr", "devices reused / new"), col("vb1", "VB1 reuse sample reproduces (<= 1e-9)", kind="text"),
                     col("vb2", "VB2 new rows = selected rows on lambda* = 1 V folds", kind="text"),
                     col("vb3", "VB3 complete"), col("vb4", "VB4 DEV only"), col("res", "result", kind="text")], rows,
                    ["VB1: the sampled folds were retrained (5 members, no cache) and refit; difference to the stored "
                     "row. Unit tests (VB5) passed in every stage."])


def t_b_rrms(d: dict) -> rf.Table:
    rows = []
    arms = ("C0_refit", "M_bias_v1", "selected_g_v2", "fixed_l1")
    for ds in m.DATASETS:
        b = d["dec"]["part_B"]["datasets"][ds]
        s = b["summary"]
        rows.append({"ds": NAME[ds], "n": f"{b['n']} ({b['unit']})",
                     **{a: f"{s[a]['all_device_mean']:.4f} / {s[a]['combined']:.4f}" for a in arms}})
    return rf.Table("07_table_i23_1_b_rrms", "Part B held-out RRMS per arm: all-device mean / combined (mean of the "
                    "nMOS and pMOS family means)",
                    [col("ds", "dataset", kind="text"), col("n", "n (CI unit)")] +
                    [col(a, B_LABEL[a]) for a in arms], rows, [])


def t_b_gmr(d: dict) -> rf.Table:
    B = d["dec"]["part_B"]
    rows = []
    for ds in m.DATASETS:
        b = B["datasets"][ds]
        c = b["comparisons"]
        rows.append({"ds": NAME[ds] + (" (G)" if ds in m.GAIN_SET else ""),
                     "fs": f"{ci(c['fixed_vs_selected'])} {c['fixed_vs_selected']['wins']}/"
                           f"{c['fixed_vs_selected']['n_used']}; {b['label_vs_selected']}" +
                           ("; NI" if b["non_inferior_to_selected"] else ""),
                     "fc": f"{ci(c['fixed_vs_C0_refit'])}; {b['label_vs_C0_refit']}",
                     "sc": f"{ci(c['selected_vs_C0_refit'])}; {b['label_selected_vs_C0_refit']}",
                     "fv": ci(c["fixed_vs_M_bias_v1"])})
    P = B["pooled"]
    rows.append({"ds": "pooled exp(mean ln GMR), 9 datasets", "fs": f"{P['fixed_l1_vs_selected']:.3f}",
                 "fc": f"{P['fixed_l1_vs_C0_refit']:.3f}", "sc": f"{P['selected_vs_C0_refit']:.3f}", "fv": "-"})
    pr = d["dec"]["predictions"]
    notes = [f"Rule outcome: {B['rule']['label']}" + (f" (failing: {', '.join(B['rule']['failing'])})"
                                                      if B["rule"]["failing"] else "") + ".",
             f"Non-inferior (CI upper < 1.05) on: {', '.join(B['non_inferior_all_9']) or 'none'}.",
             f"I22-1 pooled vs C0_refit: B1 {P['I22_1_pooled_vs_C0_refit']['B1']:.3f}, B2 "
             f"{P['I22_1_pooled_vs_C0_refit']['B2']:.3f}, B3_d3 {P['I22_1_pooled_vs_C0_refit']['B3_d3']:.3f}.",
             f"P4 {pr['P4']}; P5 {pr['P5']}; P6 {pr['P6']}."]
    return rf.Table("08_table_i23_1_b_gmr", "Part B GMR with 95% bootstrap CI (TU Delft: 7-group cluster), devices "
                    "lower / paired, labels; NI: fixed lambda non-inferior to selected (CI upper < 1.05)",
                    [col("ds", "dataset", kind="text"), col("fs", "GMR(fixed lambda 1 V, selected g v2)"),
                     col("fc", "GMR(fixed lambda 1 V, C0_refit)"), col("sc", "GMR(selected g v2, C0_refit)"),
                     col("fv", "GMR(fixed lambda 1 V, M_bias v1)")], rows, notes)


def t_b_cost(d: dict) -> rf.Table:
    rows = []
    for ds in m.DATASETS:
        c = d["dec"]["part_B"]["datasets"][ds]["cost"]
        rows.append({"ds": NAME[ds], "st": kfmt(c["starting_card_calls_per_device"]),
                     "c0": f"{c['C0_refit_refit_calls_mean']:.0f}", "fx": f"{c['fixed_l1_refit_calls_mean']:.0f}",
                     "fr": f"{c['fixed_l1_ratio_to_C0_refit']:.2f}", "se": kfmt(c["selected_total_calls_per_device"]),
                     "sr": f"{c['selected_ratio_to_C0_refit']:.0f}", "b1": kfmt(c["B1_total_calls_per_device"]),
                     "tr": str(c["fixed_l1_trainings"]), "cpu": kfmt(c["train_cpu_s"])})
    return rf.Table("09_table_i23_1_b_cost", "Part B NGSpice calls per held-out device (fixed lambda 1 V: held-out "
                    "refit only, no selection calls) next to the I22-1 cost table",
                    [col("ds", "dataset", kind="text"), col("st", "starting card calls / device (shared)"),
                     col("c0", "C0_refit"), col("fx", "fixed lambda 1 V"), col("fr", "fixed / C0_refit"),
                     col("se", "selected g v2 arm total (I22-1)"), col("sr", "selected / C0_refit"),
                     col("b1", "B1 arm total (I22-1)"), col("tr", "fixed lambda trainings"),
                     col("cpu", "fixed lambda training CPU s")], rows,
                    ["Selected g v2 totals include the nested lambda selection inner refits (I22-1 Part B; estimated "
                     "where not stored). Penalty grids (one DC deck per device) are shared by both g v2 arms and not "
                     "converted to calls."])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_forest_a(d: dict, path: Path) -> list[str]:
    A = d["dec"]["part_A"]["datasets"]
    cmps = [k for k, _ in A_CMP]
    colors = {"B1_vs_C0_call": VIOLET, "C0_call_vs_C0_refit": "#1d3557", "B1_vs_C0_refit": "#9d4edd",
              "B1_cap_vs_C0_refit": "#c8b6ff", "B1_vs_B1_cap": GREY, "C0_call_vs_selected_g_v2": AQUA}
    fig, ax = plt.subplots(figsize=(12, 8.3), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    k = len(cmps) + 1
    for i, ds in enumerate(m.IHP):
        for j, c in enumerate(cmps):
            v = A[ds]["comparisons"][c]
            y = i * k + j
            ax.plot([v["ci_low"], v["ci_high"]], [y, y], color=colors[c], lw=2.2 if j == 0 else 1.6)
            ax.plot(v["ratio"], y, "o", color=colors[c], ms=5 if j == 0 else 4,
                    label=dict(A_CMP)[c] if i == 0 else None)
            ax.text(v["ci_high"] * 1.02, y, f"{v['ratio']:.3f}", va="center", fontsize=6.8, color=INK)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks([i * k + (len(cmps) - 1) / 2 for i in range(len(m.IHP))])
    ax.set_yticklabels([NAME[ds] for ds in m.IHP], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("GMR, 95% bootstrap CI over 18 DUTs (log scale); < 1: first arm has lower held-out RRMS")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.set_title("Part A: B1 vs C0_refit at equal NGSpice calls (C0_call) and reverse control (B1_cap), IHP SG13G2",
                 fontsize=9.6, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I23 / "decision.json")]


def fig_forest_b(d: dict, path: Path) -> list[str]:
    B = d["dec"]["part_B"]["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    style(ax)
    for i, ds in enumerate(m.DATASETS):
        c = B[ds]["comparisons"]["fixed_vs_selected"]
        ax.plot([c["ci_low"], c["ci_high"]], [i, i], color=RED, lw=2)
        ax.plot(c["ratio"], i, "o", color=RED, ms=5)
        ax.text(max(c["ci_high"], 1.05) * 1.01, i, f"{c['ratio']:.3f}", va="center", fontsize=7, color=INK)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.axvline(1.05, color=GREY, lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("GMR(fixed lambda 1 V, selected g v2) (log); dashed 1, dotted 1.05 (non-inferiority margin)")
    ax.set_title("fixed lambda 1 V vs selected g v2", fontsize=9.5, color=INK)
    ax = axes[1]
    style(ax)
    for i, ds in enumerate(m.DATASETS):
        for j, (k, colr, lab) in enumerate((("fixed_vs_C0_refit", RED, "fixed lambda 1 V"),
                                            ("selected_vs_C0_refit", AQUA, "selected g v2"))):
            c = B[ds]["comparisons"][k]
            y = i + (j - 0.5) * 0.3
            ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=colr, lw=1.8)
            ax.plot(c["ratio"], y, "o", color=colr, ms=4, label=lab if i == 0 else None)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("GMR(arm, C0_refit) (log); < 1: lower held-out RRMS than C0_refit")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("both g v2 arms vs C0_refit", fontsize=9.5, color=INK)
    axes[0].set_yticks(range(len(m.DATASETS)))
    axes[0].set_yticklabels([NAME[ds] + (" (G)" if ds in m.GAIN_SET else "") for ds in m.DATASETS], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Part B: g v2 with lambda fixed at 1 V (no nested selection) per dataset, 95% bootstrap CI (TU Delft: "
                 "7-group cluster); G: gain set of the rule", fontsize=9.6, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I23 / "decision.json")]


def fig_cost(d: dict, path: Path) -> list[str]:
    B = d["dec"]["part_B"]["datasets"]
    fig, ax = plt.subplots(figsize=(12.5, 6.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    arms = (("C0_refit", "C0_refit_refit_calls_mean"), ("fixed_l1", "fixed_l1_refit_calls_mean"),
            ("B1", "B1_total_calls_per_device"), ("selected_g_v2", "selected_total_calls_per_device"))
    w = 0.2
    for j, (a, key) in enumerate(arms):
        v = [B[ds]["cost"][key] for ds in m.DATASETS]
        x = np.arange(len(m.DATASETS)) + (j - 1.5) * w
        ax.bar(x, v, width=w, color=B_COLOR[a], label=B_LABEL[a])
        for xx, vv in zip(x, v):
            ax.text(xx, vv * 1.08, f"{vv:,.0f}", ha="center", fontsize=6, rotation=90, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(m.DATASETS)))
    ax.set_xticklabels([NAME[ds] for ds in m.DATASETS], fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("NGSpice calls per held-out device (log)")
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=4)
    ax.set_title("Part B: NGSpice calls per held-out device; fixed lambda 1 V has no selection calls (selected g v2 "
                 "and B1 totals from I22-1)", fontsize=9.4, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I23 / "decision.json"), rel(I22 / "cost.json")]


def fig_a_per_device(d: dict, path: Path) -> list[str]:
    A = d["dec"]["part_A"]["rows"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    tags = sorted(A[m.IHP[0]]["B1"], key=lambda t: (t[:2], is_p(t), t))
    y = np.arange(len(tags))
    for ax, ds in zip(axes, m.IHP):
        style(ax)
        R = A[ds]
        for j, (a, b, colr, lab) in enumerate((("B1", "C0_call", VIOLET, "ln(B1 / C0_call)"),
                                               ("B1", "C0_refit", "#9d4edd", "ln(B1 / C0_refit)"),
                                               ("C0_call", "C0_refit", "#1d3557", "ln(C0_call / C0_refit)"))):
            v = [math.log(R[a][t] / R[b][t]) for t in tags]
            ax.plot(v, y + (j - 1) * 0.2, "o", ms=3.4, color=colr, label=lab)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_title(NAME[ds], fontsize=9, color=INK)
        ax.set_xlabel("per-DUT log RRMS ratio")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(tags, fontsize=7)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Part A per-DUT log ratios (< 0: first arm lower RRMS)", fontsize=9.6, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I23 / "decision.json")]


def fig_b_per_device(d: dict, ds: str, path: Path) -> list[str]:
    R = d["dec"]["part_B"]["rows"][ds]
    tags = sorted(R["C0_refit"], key=lambda t: (is_p(t), t))
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.17 * len(tags) + 1.5)), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    y = np.arange(len(tags))
    for j, (a, b, colr, lab) in enumerate((("fixed_l1", "selected_g_v2", RED, "ln(fixed lambda 1 V / selected g v2)"),
                                           ("fixed_l1", "C0_refit", "#e5989b", "ln(fixed lambda 1 V / C0_refit)"),
                                           ("selected_g_v2", "C0_refit", AQUA, "ln(selected g v2 / C0_refit)"))):
        v = [math.log(R[a][t] / R[b][t]) for t in tags]
        ax.plot(v, y + (j - 1) * 0.18, "o", ms=3, color=colr, label=lab)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tags, fontsize=5.5 if len(tags) > 40 else 7)
    ax.invert_yaxis()
    ax.set_xlabel("per-device log RRMS ratio (< 0: first arm lower RRMS)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title(f"Part B {NAME[ds]}: per-device ratios", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I23 / "decision.json")]


# ---------------------------------------------------------------------------
# I-V re-simulation
# ---------------------------------------------------------------------------
def published(ds: str, tag: str, rec):
    from cryoml.explore.e1_correction import score_device
    if ds == "sky130":
        from plot_explore_i3_2 import published_rrms
        pub, pubc = published_rrms({tag: rec})
        return pubc[tag], pub[tag]
    r0 = p22.simulate_card(ds, tag, None)
    s = [c.sim for c in r0.curves]
    return s, score_device(r0, s)["rrms"]


def fixed_card_corr(ds: str, tag: str):
    from cryoml.explore.i12_1_symm import SymCorrection
    from cryoml.explore.i6_1_tudelft import group_of
    if ds in ("sky130", "ihp300"):
        B = EXP / ("I15_2" if ds == "sky130" else "I17_2") / "lodo_sym_e1_l1"
        key = f"sym_e1_l1|{tag}"
    elif ds.startswith("tud"):
        B, key = EXP / f"I18_2/lodo_{ds[3:]}K_sym_e1_l1", f"sym_e1_l1|{group_of(tag)}"
    else:
        B, key = I23 / f"fixl1_{ds}", f"fixl1|{tag}"
        if ds == "nist4" and tag not in load(B / "results.json")["per_device"]:
            B, key = EXP / "I20_1/lodo_nist4", f"selected|{tag}"
    return dict(load(B / "heldout_cards.json")[tag]), SymCorrection.from_json(load(B / "corrections.json")[key])


def fig_iv_a(d: dict, ds: str, tag: str, why: str, path: Path) -> list[str]:
    import i22_1_baselines as drv
    from cryoml.explore.e1_correction import score_device
    R = d["dec"]["part_A"]["rows"][ds]
    rec = drv.recs(ds)[tag]
    sims, sc = {}, {}
    sims["published"], sc["published"] = published(ds, tag, rec)
    c22 = load(I22 / f"partA_{ds}/heldout_cards.json")
    c23 = load(I23 / f"partA_{ds}/heldout_cards.json")
    for a, card in (("C0_refit", c22[f"C0_refit|{tag}"]), ("B1", c22[f"B1|{tag}"]),
                    ("C0_call", c23[f"C0_call|{tag}"]), ("B1_cap", c23[f"B1_cap|{tag}"])):
        new = p22.simulate_card(ds, tag, dict(card))
        s = [c.sim for c in new.curves]
        sims[a], sc[a] = s, score_device(new, s)["rrms"]
        check(sc[a], R[a][tag], f"{ds} {tag} {a}")
    calls = d["partA"][ds]["n_sim"]
    lines = [("published", GREY, "--"), ("C0_refit", BLUE, "-"), ("C0_call", "#1d3557", "-."), ("B1", VIOLET, "-"),
             ("B1_cap", "#c8b6ff", ":")]
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=4,
                      label=f"measured {ds[3:]} K (filled: scored; hollow: trimmed before the metric; faded: excluded "
                            "curve)"),
               Line2D([], [], color=GREY, ls="--", lw=1.2, label=f"NGSpice, stock card (no fit): RRMS "
                                                                 f"{sc['published']:.4f}"),
               Line2D([], [], color=BLUE, lw=1.2, label=f"NGSpice re-simulated, C0_refit held-out card ("
                                                        f"{d['partA'][ds]['N_C0'][tag]} calls): RRMS {sc['C0_refit']:.4f}"),
               Line2D([], [], color="#1d3557", ls="-.", lw=1.2,
                      label=f"NGSpice re-simulated, C0_call held-out card ({calls['C0_call'][tag]} calls, limit "
                            f"{d['partA'][ds]['N_B1'][tag]}): RRMS {sc['C0_call']:.4f}"),
               Line2D([], [], color=VIOLET, lw=1.2, label=f"NGSpice re-simulated, B1 wider-card held-out card ("
                                                          f"{d['partA'][ds]['N_B1'][tag]} calls): RRMS {sc['B1']:.4f}"),
               Line2D([], [], color="#c8b6ff", ls=":", lw=1.4,
                      label=f"NGSpice re-simulated, B1_cap held-out card ({calls['B1_cap'][tag]} calls): RRMS "
                            f"{sc['B1_cap']:.4f}")]
    p22.draw_iv(rec, sims, lines, handles,
                f"{tag} ({NAME[ds]}, DEV; {why}):\n"
                f"held-out cards re-simulated ({p22.SIMNAME['ihp']}). RRMS equals "
                "the stored rows.", path)
    return [rel(I22 / f"partA_{ds}/heldout_cards.json"), rel(I23 / f"partA_{ds}/heldout_cards.json")]


def fig_iv_b(d: dict, ds: str, tag: str, why: str, path: Path) -> list[str]:
    import i22_1_baselines as drv
    from cryoml.explore.e1_correction import score_device
    R = d["dec"]["part_B"]["rows"][ds]
    rec = drv.recs(ds)[tag]
    sims, sc = {}, {}
    sims["published"], sc["published"] = published(ds, tag, rec)
    new = p22.simulate_card(ds, tag, dict(load(I22 / f"partA_{ds}/heldout_cards.json")[f"C0_refit|{tag}"]))
    sims["C0_refit"] = [c.sim for c in new.curves]
    sc["C0_refit"] = score_device(new, sims["C0_refit"])["rrms"]
    check(sc["C0_refit"], R["C0_refit"][tag], f"{ds} {tag} C0_refit")
    p, corr, lam = p22.g2_card_corr(ds, tag)
    new = p22.simulate_card(ds, tag, p)
    sims["selected_g_v2"] = corr.apply(new)
    sc["selected_g_v2"] = score_device(new, sims["selected_g_v2"])["rrms"]
    check(sc["selected_g_v2"], R["selected_g_v2"][tag], f"{ds} {tag} selected")
    p, corr = fixed_card_corr(ds, tag)
    new = p22.simulate_card(ds, tag, p)
    sims["fixed_l1"] = corr.apply(new)
    sc["fixed_l1"] = score_device(new, sims["fixed_l1"])["rrms"]
    check(sc["fixed_l1"], R["fixed_l1"][tag], f"{ds} {tag} fixed")
    pub_lab = ("NGSpice, published 77 K card (upstream committed sweep)" if ds == "sky130"
               else "NGSpice, stock card (no fit)")
    meas_lab = {"sky130": "77 K", "nist4": "4 K", "nistrt": "4 K"}.get(ds, f"{ds[3:]} K")
    lines = [("published", GREY, "--"), ("C0_refit", BLUE, "-"), ("selected_g_v2", AQUA, "-"), ("fixed_l1", RED, "-.")]
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=4,
                      label=f"measured {meas_lab} (filled: scored; hollow: trimmed before the metric; faded: excluded "
                            "curve)"),
               Line2D([], [], color=GREY, ls="--", lw=1.2, label=f"{pub_lab}: RRMS {sc['published']:.4f}"),
               Line2D([], [], color=BLUE, lw=1.2, label=f"NGSpice re-simulated, C0_refit held-out card: RRMS "
                                                        f"{sc['C0_refit']:.4f}"),
               Line2D([], [], color=AQUA, lw=1.2, label=f"NGSpice re-simulated, selected g v2 held-out card x exp(fold "
                                                        f"g) (lambda* {lam:g} V): RRMS {sc['selected_g_v2']:.4f}"),
               Line2D([], [], color=RED, ls="-.", lw=1.2,
                      label=f"NGSpice re-simulated, fixed lambda 1 V held-out card x exp(fold g): RRMS "
                            f"{sc['fixed_l1']:.4f}")]
    k = "ihp" if ds.startswith("ihp") else "tud" if ds.startswith("tud") else ds
    p22.draw_iv(rec, sims, lines, handles,
                f"{tag} ({NAME[ds]}, DEV; {why}):\n"
                f"held-out cards re-simulated ({p22.SIMNAME[k]}); g applied in "
                "Python. RRMS equals the stored rows.", path)
    return ["stored C0_refit (I22-1), selected g v2 and fixed lambda 1 V held-out cards and corrections"]


def iv_choices_a(d: dict) -> list[tuple[str, str, str]]:
    A = d["dec"]["part_A"]["rows"]
    lr = {(ds, t): math.log(A[ds]["B1"][t] / A[ds]["C0_call"][t]) for ds in m.IHP for t in A[ds]["B1"]}
    lo = min(lr, key=lr.get)
    hi = max(lr, key=lr.get)
    return [(lo[0], lo[1], f"lowest ln(B1 / C0_call) {lr[lo]:+.3f}"),
            (hi[0], hi[1], f"highest ln(B1 / C0_call) {lr[hi]:+.3f}")]


def iv_choices_b(d: dict) -> list[tuple[str, str, str]]:
    B = d["dec"]["part_B"]["rows"]
    out = []
    for ds in ("sky130", "nist4", "nistrt", "ihp233", "tud4"):
        R = B[ds]
        lr = {t: math.log(R["fixed_l1"][t] / R["selected_g_v2"][t]) for t in R["fixed_l1"]
              if np.isfinite(R["fixed_l1"][t])}
        hi = max(lr, key=lr.get)
        out.append((ds, hi, f"highest ln(fixed lambda 1 V / selected) {lr[hi]:+.3f}"))
        if ds in m.GAIN_SET:
            lo = min(lr, key=lr.get)
            out.append((ds, lo, f"lowest ln(fixed lambda 1 V / selected) {lr[lo]:+.3f}"))
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--no-iv", action="store_true")
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    d = load_all()
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    dec = [rel(I23 / "decision.json")]
    add_table(t_setup(d), "I23-1 setup (pre-registered)", [rel(I23 / "prereg.json")])
    add_table(t_a_validity(d), "Part A validity per temperature", dec)
    add_table(t_a_rrms(d), "Part A RRMS per arm and temperature", dec)
    add_table(t_a_gmr(d), "Part A GMR with CI and labels, rule outcome, pooled", dec)
    add_table(t_a_calls(d), "Part A NGSpice calls per DUT", dec)
    add_table(t_b_validity(d), "Part B row sources and validity", dec + [rel(I23 / f"verify_{x}.json") for x in m.REUSED])
    add_table(t_b_rrms(d), "Part B RRMS per arm and dataset", dec)
    add_table(t_b_gmr(d), "Part B GMR with CI and labels, rule outcome, pooled", dec)
    add_table(t_b_cost(d), "Part B NGSpice calls per held-out device next to the I22-1 cost table",
              dec + [rel(I22 / "cost.json")])
    k = len(items) + 1
    for name, fn, desc in (("forest_a", fig_forest_a, "Part A forest plot: B1 vs call-matched C0_refit and reverse "
                                                     "control per temperature"),
                           ("forest_b", fig_forest_b, "Part B forest plot: fixed lambda 1 V vs selected g v2, and both "
                                                     "vs C0_refit, per dataset"),
                           ("cost_b", fig_cost, "Part B NGSpice calls per held-out device per arm and dataset"),
                           ("per_device_a", fig_a_per_device, "Part A per-DUT log ratios")):
        png = f"{k:02d}_i23_1_{name}.png"
        items.append((png, desc, fn(d, folder / png)))
        k += 1
    for ds in m.GAIN_SET:
        png = f"{k:02d}_i23_1_per_device_b_{ds}.png"
        items.append((png, f"Part B {NAME[ds]}: per-device log ratios", fig_b_per_device(d, ds, folder / png)))
        k += 1
    if not args.no_iv:
        for ds, tag, why in iv_choices_a(d):
            png = f"{k:02d}_i23_1_iv_a_{ds}_{tag}.png"
            items.append((png, f"Part A I-V of {tag} ({NAME[ds]}; {why}): measured points, NGSpice stock card, "
                               "re-simulated C0_refit, C0_call, B1 and B1_cap held-out cards",
                          fig_iv_a(d, ds, tag, why, folder / png)))
            k += 1
        for ds, tag, why in iv_choices_b(d):
            png = f"{k:02d}_i23_1_iv_b_{ds}_{tag}.png"
            items.append((png, f"Part B I-V of {tag} ({NAME[ds]}; {why}): measured points, NGSpice published / stock "
                               "card, re-simulated C0_refit, selected g v2 and fixed lambda 1 V held-out cards",
                          fig_iv_b(d, ds, tag, why, folder / png)))
            k += 1
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_explore_i23_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    A, B = d["dec"]["part_A"], d["dec"]["part_B"]
    lines = [f"Part A rule: {A['rule']['label']}."]
    for ds in m.IHP:
        c = A["datasets"][ds]["comparisons"]
        lines.append(f"{NAME[ds]}: GMR(B1, C0_call) {ci(c['B1_vs_C0_call'])}; GMR(C0_call, C0_refit) "
                     f"{ci(c['C0_call_vs_C0_refit'])}; GMR(B1_cap, C0_refit) {ci(c['B1_cap_vs_C0_refit'])}")
    lines.append(f"Part B rule: {B['rule']['label']}.")
    for ds in m.DATASETS:
        c = B["datasets"][ds]["comparisons"]
        lines.append(f"{NAME[ds]}: GMR(fixed, selected) {ci(c['fixed_vs_selected'])}; GMR(fixed, C0_refit) "
                     f"{ci(c['fixed_vs_C0_refit'])}")
    lines.append(f"Pooled vs C0_refit: fixed lambda 1 V {B['pooled']['fixed_l1_vs_C0_refit']:.3f}, selected g v2 "
                 f"{B['pooled']['selected_vs_C0_refit']:.3f}.")
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. NGSpice runs: I-V re-simulation of the published / stock cards and the "
              f"held-out cards; every re-simulated RRMS checked against the stored rows; {d['n_checked']} stored "
              "values recomputed.", "", "## Contents", "",
              f"- I23-1 call-matched IHP control for B1 and fixed-lambda g v2. Pre-registration "
              f"`out/research/explore/I23_1/prereg.md` ({COMMITS['prereg']}). DEV only.",
              *[f"- {x}" for x in lines], "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items],
                    ["I23-1 call-matched IHP control and fixed-lambda g v2."] + lines)
    print("\n".join(lines))
    print("\n".join(str(folder / i[0]) for i in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
