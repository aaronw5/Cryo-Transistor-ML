#!/usr/bin/env python3
"""Report batch for I24-1: (A) three-way dataset-level gate {C0_refit, selected
g v2, B1} on the 9 measured DEV datasets (retrospective), (C) TEST prediction
(proposed amendment only), (B) selected g v2 on top of B1 on IHP SG13G2.

Reads ``out/research/explore/I24_1/`` (prereg, partA/logo_<ds>.json,
partB/ihp_<T>K/, decision.json) and the I22-1 rows / cards.

NGSpice: the I-V figures re-simulate the stock card and the held-out cards
(C0_refit, B1, B1_cont, B1+g v2, selected g v2) with ngspice-46 + psp103.osdi;
g is applied in Python.  Every re-simulated RRMS is checked against the stored
row.

  .venv/bin/python -I scripts/research/plot_explore_i24_1.py --date 2026-10-28
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
from cryoml.explore import i24_1_gate3 as m  # noqa: E402
from cryoml.explore.i3_2_bias import paired_ratio  # noqa: E402

rf = p22.rf
plt = p22.plt
ci = p22.ci
col = p22.col
check = p22.check
kfmt = p22.kfmt

TOPIC = "i24_1"
I24 = EXP / "I24_1"
I22 = EXP / "I22_1"
COMMITS = {"prereg": "9ffe9f6", "code": "8fcf823"}
DS = m22.DATASETS
NAME = {**m22.NAMES}
TEMPS = ("233", "300", "343", "398")
ARM = {"C0_refit": "C0_refit", "g": "selected g v2", "B1": "B1"}
SERIES_A = (("g always", AQUA), ("B1 always", VIOLET), ("two-way frozen gate", ORANGE), ("three-way gate", RED))
B_COLOR = {"B1": VIOLET, "B1_cont": "#c8b6ff", "B1+g v2": RED, "C0_refit": BLUE, "selected g v2": AQUA}
B_CMP = ("B1+g v2 vs B1", "B1+g v2 vs C0_refit", "B1+g v2 vs B1_cont", "B1_cont vs B1", "B1+g v2 vs selected g v2",
         "B1 vs C0_refit")


def is_p(t: str) -> bool:
    return "pmos" in t or t.startswith("pfet")


def clusters(ds: str, tags):
    if ds.startswith("tud"):
        from cryoml.explore.i6_1_tudelft import group_of
        return {t: group_of(t) for t in tags}
    return None


# ---------------------------------------------------------------------------
def load_all() -> dict:
    d = {"prereg": load(I24 / "prereg.json"), "dec": load(I24 / "decision.json")}
    A, B = d["dec"]["part_A"], d["dec"]["part_B"]
    n = 0
    for ds in DS:
        a = A["datasets"][ds]
        R = a["rows"]
        i22 = load(I22 / f"partA_{ds}/results.json")["per_device"]
        for t in R["B1"]:
            check(R["B1"][t], i22["B1"][t], f"{ds} B1 {t}", 0.0)
            n += 1
        L = load(I24 / f"partA/logo_{ds}.json")
        check(a["S_B1"], float(np.mean([L["per_record"][t]["log_ratio"] for t in L["per_record"]])), f"{ds} S_B1",
              1e-12)
        n += 1
        g = paired_ratio(R[a["choice"]["arm"]], R["C0_refit"], clusters(ds, R["C0_refit"]))
        for k in ("ratio", "ci_low", "ci_high"):
            check(g[k], a["gmr_vs_C0_refit"]["three-way gate"][k], f"{ds} three-way {k}", 1e-12)
            n += 1
    lg = [math.log(A["datasets"][ds]["gmr_vs_C0_refit"]["three-way gate"]["ratio"]) for ds in DS]
    check(math.exp(np.mean(lg)), A["pooled"]["three-way gate"]["pooled"], "pooled three-way", 1e-12)
    n += 1
    for T in TEMPS:
        x = B["per_temperature"][T]
        g = paired_ratio(x["rows"]["B1+g v2"], x["rows"]["B1"])
        for k in ("ratio", "ci_low", "ci_high"):
            check(g[k], x["comparisons"]["B1+g v2 vs B1"][k], f"{T} B1+g vs B1 {k}", 1e-12)
            n += 1
    d["n_checked"] = n
    return d


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_setup(d: dict) -> rf.Table:
    P = d["prereg"]
    A, B, C = P["part_A"], P["part_B"], P["part_C"]
    rows = [{"c": "pre-registration", "v": f"out/research/explore/I24_1/prereg.md (commit {COMMITS['prereg']}); code "
                                         f"{COMMITS['code']}; DEV only, TEST locked; no TEST decision file written"},
            {"c": "Part A arms", "v": "; ".join(f"{k}: {v}" for k, v in A["arms"].items())},
            {"c": "S_g", "v": A["S_g"]}, {"c": "S_B1", "v": A["S_B1"]}, {"c": "choice rule", "v": A["rule"]},
            {"c": "sensitivity", "v": "; ".join(f"{k}: {v}" for k, v in A["sensitivity"].items())},
            {"c": "Part A outcome rule", "v": "; ".join(f"{k}: {v}" for k, v in A["outcome_rule"].items()
                                                        if k != "labels")},
            {"c": "Part A limitation", "v": A["limitation"]},
            {"c": "Part C", "v": f"{C['decision']}; amendment: {C['amendment']}"},
            {"c": "Part B records / refit", "v": f"{B['records']}; {B['refit']}"},
            {"c": "Part B arms", "v": "; ".join(B["arms"])},
            {"c": "Part B rule", "v": "; ".join(f"{k}: {v}" for k, v in B["rule"].items())}]
    return rf.Table("01_table_i24_1_setup", "I24-1 setup (pre-registered)",
                    [col("c", "item", kind="text"), col("v", "value", kind="text")], rows, [])


def t_a_validity(d: dict) -> rf.Table:
    rows = []
    for ds in DS:
        a = d["dec"]["part_A"]["datasets"][ds]
        v = a["validity"]
        rows.append({"ds": NAME[ds], "va1": f"{v['VA1']} ({v['VA1_S_g']:+.4f})", "va2": str(v["VA2"]),
                     "va3": (f"{v['VA3']} ({v['VA3_max']:.1e})" if v["VA3_required"] else "not required"),
                     "va4": f"{v['VA4']} ({a['S_B1_n_nonfinite']} non-finite)", "va5": str(v["VA5"]),
                     "va6": str(v["VA6"]), "res": "valid" if v["valid"] else "invalid"})
    return rf.Table("02_table_i24_1_a_validity", "Part A validity per dataset (VA1-VA6)",
                    [col("ds", "dataset", kind="text"), col("va1", "VA1 S_g = freeze draft (< 5e-5)"),
                     col("va2", "VA2 screen reproduces"), col("va3", "VA3 B1 rerun = I22-1 (<= 1e-12)"),
                     col("va4", "VA4 complete"), col("va5", "VA5 DEV only"), col("va6", "VA6 unit tests"),
                     col("res", "result", kind="text")], rows,
                    ["VA3 is required only on datasets with at least one new LOGO refit (2 devices rerun with the "
                     "full selection)."])


def t_a_stats(d: dict) -> rf.Table:
    rows = []
    for ds in DS:
        a = d["dec"]["part_A"]["datasets"][ds]
        lg = a["logo"]
        rows.append({"ds": NAME[ds], "sg": f"{a['S_g']:+.4f}", "yg": f"{a['yhat_g']:+.4f}", "sb": f"{a['S_B1']:+.4f}",
                     "sbf": f"{a['S_B1_full_screen']:+.4f}",
                     "lo": f"{lg['n_equal']} / {lg['n_new_refits']} / {lg['n_empty']}",
                     "two": ARM[a["two_way"]], "three": ARM[a["choice"]["arm"]],
                     "s1": ARM[a["S1"]], "s2": ARM[a["S2"]],
                     "s3": ", ".join(f"{ARM[k]} {v}" for k, v in a["S3_counts"].items() if v),
                     "or": ARM[a["oracle"]]})
    return rf.Table("03_table_i24_1_a_stats", "Part A gate statistics and choices per dataset (DEV)",
                    [col("ds", "dataset", kind="text"), col("sg", "S_g (frozen S3_D)"),
                     col("yg", "yhat_g = -0.0215 + 1.217 S_g"), col("sb", "S_B1 (LOGO screen)"),
                     col("sbf", "S_B1 (I22-1 full screen, S2)"),
                     col("lo", "records: LOGO = full / new refit / empty"), col("two", "two-way frozen gate"),
                     col("three", "three-way gate (primary)"), col("s1", "S1 raw minimum"),
                     col("s2", "S2 full-screen S_B1"), col("s3", "S3 leave-group-out (devices per arm)"),
                     col("or", "oracle (post hoc)")], rows,
                    ["S_g: v1-recipe S3_D, state C (frozen rule). S_B1: mean over DEV records of ln(RRMS(B1 with the "
                     "screen of the other groups) / RRMS(C0_refit)). Candidate iff statistic < 0; both candidates: "
                     "yhat_g vs S_B1, lower wins, ties to B1."])


def t_a_gmr(d: dict) -> rf.Table:
    A = d["dec"]["part_A"]
    keys = ("g always", "B1 always", "two-way frozen gate", "three-way gate", "S3 leave-group-out gate",
            "oracle (post hoc)")
    rows = []
    for ds in DS:
        g = A["datasets"][ds]["gmr_vs_C0_refit"]
        rows.append({"ds": NAME[ds], "u": A["datasets"][ds]["unit"],
                     **{k: f"{ci(g[k])} {g[k]['wins']}/{g[k]['n_used']}" for k in keys}})
    P = A["pooled"]
    rows.append({"ds": "pooled exp(mean ln GMR), 9 datasets", "u": "stratified",
                 **{k: ci({"ratio": P[k]["pooled"], "ci_low": P[k]["ci_low"], "ci_high": P[k]["ci_high"]})
                    for k in keys}})
    return rf.Table("04_table_i24_1_a_gmr", "Part A GMR vs C0_refit with 95% bootstrap CI, devices lower / paired "
                    "(TU Delft: 7-group cluster); last row pooled over the 9 datasets",
                    [col("ds", "dataset", kind="text"), col("u", "CI unit")] + [col(k, k) for k in keys], rows,
                    ["B1 rows: I22-1 (screen on all DEV devices). Pooled CI: stratified bootstrap, decisions fixed."])


def t_a_pooled(d: dict) -> rf.Table:
    A = d["dec"]["part_A"]
    rows = [{"k": k, "v": ci({"ratio": v["pooled"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]})}
            for k, v in A["pooled"].items()]
    rows += [{"k": f"paired pooled ratio {k}", "v": ci(v)} for k, v in A["paired_pooled"].items()]
    o = A["outcome"]
    notes = [f"Rule outcome: {o['label']} (R1 {o['R1']}" + (f", fails {', '.join(o['R1_fail'])}" if o["R1_fail"] else "")
             + f"; R2 {o['R2']}; R3 {o['R3']}).",
             f"B1 not buildable on: {', '.join(A['not_buildable']) or 'none'}.",
             "P: " + "; ".join(f"{k} {v}" for k, v in d["dec"]["predictions"].items() if not isinstance(v, dict))]
    return rf.Table("05_table_i24_1_a_pooled", "Part A pooled GMR vs C0_refit over 9 datasets [95% stratified "
                    "bootstrap CI] and paired pooled ratios",
                    [col("k", "series", kind="text"), col("v", "pooled GMR [95% CI]")], rows, notes)


def t_a_rrms(d: dict) -> rf.Table:
    A = d["dec"]["part_A"]
    keys = ("C0_refit", "g always", "B1 always", "two-way frozen gate", "three-way gate")
    rows = []
    for ds in DS:
        ag = A["datasets"][ds]["aggregates"]
        rows.append({"ds": NAME[ds], **{k: f"{ag[k]['all_device_mean']:.4f} / {ag[k]['combined']:.4f}" for k in keys}})
    return rf.Table("06_table_i24_1_a_rrms", "Part A held-out RRMS per series: all-device mean / combined (mean of "
                    "the nMOS and pMOS family means)", [col("ds", "dataset", kind="text")] +
                    [col(k, k) for k in keys], rows, ["Strict paper RRMS, frozen curve inclusion."])


def t_c(d: dict) -> rf.Table:
    C = d["dec"]["part_C"]
    rows = [{"s": r["test_split"], "n": str(r["n_test"]), "c": f"{r['committed_a755e18']} ({r['committed_arm']})",
             "two": r["two_way_frozen"], "three": r["three_way"], "sg": f"{r['S_g']:+.4f}", "sb": f"{r['S_B1']:+.4f}",
             "df": "yes" if r["differs_from_a755e18"] else "no"} for r in C["rows"]]
    notes = ["Proposed amendment (not committed; user decision): " +
             ("; ".join(C["proposed_amendment"]) if C["proposed_amendment"] else "none"),
             "TEST data not opened; decisions from DEV statistics only."]
    return rf.Table("07_table_i24_1_c_test", "Part C: TEST gate decisions, committed (a755e18) vs two-way frozen "
                    "rule vs three-way gate",
                    [col("s", "TEST split", kind="text"), col("n", "n TEST"), col("c", "committed a755e18"),
                     col("two", "two-way frozen"), col("three", "three-way (I24-1)"), col("sg", "S_g (DEV)"),
                     col("sb", "S_B1 (DEV)"), col("df", "differs from a755e18")], rows, notes)


def t_b_validity(d: dict) -> rf.Table:
    rows = []
    for T in TEMPS:
        v = d["dec"]["part_B"]["per_temperature"][T]["validity"]
        rows.append({"t": f"{T} K", "vb1": f"{v['VB1']} ({v['VB1_max']:.1e})", "vb2": str(v["VB2"]),
                     "vb3": str(v["VB3"]), "vb4": ", ".join(f"{a} {k}" for a, k in v["n_failed"].items() if k) or
                     "0 failed", "vb5": str(v["VB5"]), "vb6": str(v["VB6"]),
                     "res": "valid" if v["valid"] else "invalid"})
    return rf.Table("08_table_i24_1_b_validity", "Part B validity per temperature (VB1-VB6; VB7 checked once: "
                    "difference 0 on 2 DUTs)",
                    [col("t", "temperature", kind="text"), col("vb1", "VB1 B1 records = I22-1 B1 rows"),
                     col("vb2", "VB2 grids"), col("vb3", "VB3 inner complete"), col("vb4", "VB4 non-finite rows"),
                     col("vb5", "VB5 unit tests"), col("vb6", "VB6 DEV only"), col("res", "result", kind="text")],
                    rows, [])


def t_b_rrms(d: dict) -> rf.Table:
    rows = []
    arms = ("C0_refit", "selected g v2", "B1", "B1_cont", "B1+g v2")
    for T in TEMPS:
        ag = d["dec"]["part_B"]["per_temperature"][T]["aggregates"]
        rows.append({"t": f"{T} K", **{a: f"{ag[a]['all_device_mean']:.5f} / {ag[a]['combined']:.5f}" for a in arms}})
    return rf.Table("09_table_i24_1_b_rrms", "Part B held-out RRMS per arm (18 DUTs): all-device mean / combined",
                    [col("t", "temperature", kind="text")] + [col(a, a) for a in arms], rows,
                    ["B1 and C0_refit: stored I22-1 rows; selected g v2: stored rows (starting cards); B1_cont and "
                     "B1+g v2: I24-1 refits of the B1 card (B1 box, same budget)."])


def t_b_gmr(d: dict) -> rf.Table:
    B = d["dec"]["part_B"]
    rows = []
    for T in TEMPS:
        x = B["per_temperature"][T]
        rows.append({"t": f"{T} K", **{k: f"{ci(x['comparisons'][k])} {x['comparisons'][k]['wins']}/"
                                            f"{x['comparisons'][k]['n_used']}; {x['labels'][k]}" for k in B_CMP},
                     "lam": ", ".join(f"{k} V x{v}" for k, v in sorted(x["lambda_star_counts"].items(),
                                                                         key=lambda z: float(z[0])))})
    notes = [f"Rule outcome: {B['outcome']['label']}."]
    return rf.Table("10_table_i24_1_b_gmr", "Part B GMR with 95% bootstrap CI (device), DUTs lower / paired, label "
                    "(gain: CI upper < 1; worse: CI lower > 1)",
                    [col("t", "temperature", kind="text")] + [col(k, f"GMR({k.replace(' vs ', ', ')})") for k in B_CMP]
                    + [col("lam", "lambda* (folds)")], rows, notes)


def t_b_calls(d: dict) -> rf.Table:
    rows = []
    for T in TEMPS:
        c = d["dec"]["part_B"]["per_temperature"][T]["calls_mean"]
        rows.append({"t": f"{T} K", "c0": f"{c['C0_refit']:.0f}", "b1": f"{c['B1_refit']:.0f}",
                     "sel": kfmt(c["selection"]), "ho": f"{c['heldout']:.0f}", "tot": kfmt(c["total"]),
                     "r": f"{c['total'] / c['C0_refit']:.0f}"})
    return rf.Table("11_table_i24_1_b_calls", "Part B NGSpice calls per held-out DUT (mean over 18)",
                    [col("t", "temperature", kind="text"), col("c0", "C0_refit refit"), col("b1", "B1 refit (I22-1)"),
                     col("sel", "nested selection (inner B1-box refits of the outer fold)"),
                     col("ho", "B1+g v2 held-out refit"), col("tot", "B1+g v2 arm total"),
                     col("r", "total / C0_refit")], rows,
                    ["Penalty grids (one 75 x 75 DC deck per DUT) and the B1 screen (173 calls per DUT, 300 K, "
                     "stored) are not included."])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_forest_a(d: dict, path: Path) -> list[str]:
    A = d["dec"]["part_A"]
    fig, ax = plt.subplots(figsize=(12, 9.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    k = len(SERIES_A) + 1
    labels = []
    for i, ds in enumerate(list(DS) + ["pooled"]):
        for j, (s, colr) in enumerate(SERIES_A):
            if ds == "pooled":
                v = A["pooled"][s]
                v = {"ratio": v["pooled"], "ci_low": v["ci_low"], "ci_high": v["ci_high"]}
            else:
                v = A["datasets"][ds]["gmr_vs_C0_refit"][s]
            y = i * k + j
            ax.plot([v["ci_low"], v["ci_high"]], [y, y], color=colr, lw=2.0 if s == "three-way gate" else 1.5)
            ax.plot(v["ratio"], y, "D" if ds == "pooled" else "o", color=colr, ms=5,
                    label=s if i == 0 else None)
            ax.text(max(v["ci_high"], v["ratio"]) * 1.02, y, f"{v['ratio']:.3f}", va="center", fontsize=6.6, color=INK)
        if ds == "pooled":
            labels.append("pooled, 9 datasets\n(stratified bootstrap)")
        else:
            a = A["datasets"][ds]
            labels.append(f"{NAME[ds]}\nthree-way: {ARM[a['choice']['arm']]}")
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks([i * k + (len(SERIES_A) - 1) / 2 for i in range(len(DS) + 1)])
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("GMR vs C0_refit, 95% bootstrap CI (log scale); < 1: lower held-out RRMS than C0_refit")
    ax.legend(fontsize=7.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4)
    ax.set_title("Part A: fixed arms, two-way frozen gate and three-way gate vs C0_refit per DEV dataset "
                 "(retrospective)", fontsize=9.6, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I24 / "decision.json")]


def fig_pooled(d: dict, path: Path) -> list[str]:
    A = d["dec"]["part_A"]
    keys = list(A["pooled"])
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    colors = {"three-way gate": RED, "two-way frozen gate": ORANGE, "g always": AQUA, "B1 always": VIOLET}
    for i, k in enumerate(keys):
        v = A["pooled"][k]
        c = colors.get(k, GREY)
        ax.plot([v["ci_low"], v["ci_high"]], [i, i], color=c, lw=2)
        ax.plot(v["pooled"], i, "o", color=c, ms=5)
        ax.text(v["ci_high"] * 1.01, i, f"{v['pooled']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]", va="center",
                fontsize=7, color=INK)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("pooled exp(mean over 9 datasets of ln GMR vs C0_refit), 95% stratified bootstrap CI "
                  "(decisions fixed)")
    p = A["paired_pooled"]["three-way / two-way"]
    ax.set_title(f"Part A pooled GMR vs C0_refit; three-way / two-way {p['ratio']:.3f} [{p['ci_low']:.3f}, "
                 f"{p['ci_high']:.3f}]; outcome: {A['outcome']['label']}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I24 / "decision.json")]


def fig_stat_scatter(d: dict, path: Path) -> list[str]:
    A = d["dec"]["part_A"]["datasets"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    for ax, (arm, key, xlab, colr) in zip(axes, (("B1", "S_B1", "S_B1 (LOGO screen, DEV)", VIOLET),
                                                 ("g", "S_g", "S_g = frozen S3_D (v1 recipe, state C)", AQUA))):
        style(ax)
        xs, ys = [], []
        for ds in DS:
            x = A[ds][key]
            y = math.log(A[ds]["gmr_vs_C0_refit"]["B1 always" if arm == "B1" else "g always"]["ratio"])
            xs.append(x)
            ys.append(y)
            ax.plot(x, y, "o", color=colr, ms=6)
            ax.annotate(ds, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7, color=INK)
        lim = [min(xs + ys) - 0.03, max(xs + ys) + 0.03]
        ax.plot(lim, lim, color=GREY, lw=0.8, ls=":")
        if arm == "g":
            xx = np.linspace(lim[0], lim[1], 10)
            ax.plot(xx, m.OLS_A + m.OLS_B * xx, color=AQUA, lw=0.8, ls="--", label="I19-1 OLS yhat_g")
            ax.legend(fontsize=7.5, frameon=False)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel(xlab)
        ax.set_ylabel(f"realised ln GMR({ARM[arm]}, C0_refit), DEV")
    fig.suptitle("Part A: gate statistic vs realised ln GMR per dataset (B1 rows: I22-1; S_B1 and the B1 rows share "
                 "the held-out refits)", fontsize=9.4, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I24 / "decision.json")]


def fig_forest_b(d: dict, path: Path) -> list[str]:
    B = d["dec"]["part_B"]["per_temperature"]
    colors = {"B1+g v2 vs B1": RED, "B1+g v2 vs C0_refit": BLUE, "B1+g v2 vs B1_cont": "#e5989b",
              "B1_cont vs B1": "#c8b6ff", "B1+g v2 vs selected g v2": AQUA, "B1 vs C0_refit": VIOLET}
    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    k = len(B_CMP) + 1
    for i, T in enumerate(TEMPS):
        for j, c in enumerate(B_CMP):
            v = B[T]["comparisons"][c]
            y = i * k + j
            ax.plot([v["ci_low"], v["ci_high"]], [y, y], color=colors[c], lw=2.2 if j == 0 else 1.5)
            ax.plot(v["ratio"], y, "o", color=colors[c], ms=5 if j == 0 else 4, label=f"GMR({c.replace(' vs ', ', ')})"
                    if i == 0 else None)
            ax.text(v["ci_high"] * 1.02, y, f"{v['ratio']:.3f}", va="center", fontsize=6.8, color=INK)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks([i * k + (len(B_CMP) - 1) / 2 for i in range(len(TEMPS))])
    ax.set_yticklabels([f"IHP SG13G2 {T} K" for T in TEMPS], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("GMR, 95% bootstrap CI over 18 DUTs (log scale); < 1: first arm has lower held-out RRMS")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.set_title(f"Part B: selected g v2 on top of B1, IHP SG13G2; outcome: {d['dec']['part_B']['outcome']['label']}",
                 fontsize=9.6, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I24 / "decision.json")]


def fig_b_per_device(d: dict, path: Path) -> list[str]:
    B = d["dec"]["part_B"]["per_temperature"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    tags = sorted(B[TEMPS[0]]["rows"]["B1"], key=lambda t: (t[:2], is_p(t), t))
    y = np.arange(len(tags))
    for ax, T in zip(axes, TEMPS):
        style(ax)
        R = B[T]["rows"]
        for j, (a, b, colr, lab) in enumerate((("B1+g v2", "B1", RED, "ln(B1+g v2 / B1)"),
                                               ("B1+g v2", "C0_refit", BLUE, "ln(B1+g v2 / C0_refit)"),
                                               ("B1_cont", "B1", "#c8b6ff", "ln(B1_cont / B1)"))):
            v = [math.log(R[a][t] / R[b][t]) if np.isfinite(R[a][t]) else np.nan for t in tags]
            ax.plot(v, y + (j - 1) * 0.2, "o", ms=3.4, color=colr, label=lab)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_title(f"IHP SG13G2 {T} K", fontsize=9, color=INK)
        ax.set_xlabel("per-DUT log RRMS ratio")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(tags, fontsize=7)
    axes[0].invert_yaxis()
    axes[-1].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Part B per-DUT log ratios (< 0: first arm lower RRMS)", fontsize=9.6, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I24 / "decision.json")]


def fig_iv_b(d: dict, T: str, tag: str, why: str, path: Path) -> list[str]:
    import i22_1_baselines as drv
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i12_1_symm import SymCorrection
    ds = f"ihp{T}"
    R = d["dec"]["part_B"]["per_temperature"][T]["rows"]
    rec = drv.recs(ds)[tag]
    sims, sc = {}, {}
    r0 = p22.simulate_card(ds, tag, None)
    sims["published"] = [c.sim for c in r0.curves]
    sc["published"] = score_device(r0, sims["published"])["rrms"]
    c22 = load(I22 / f"partA_{ds}/heldout_cards.json")
    for a, card in (("C0_refit", c22[f"C0_refit|{tag}"]), ("B1", c22[f"B1|{tag}"])):
        new = p22.simulate_card(ds, tag, dict(card))
        sims[a] = [c.sim for c in new.curves]
        sc[a] = score_device(new, sims[a])["rrms"]
        check(sc[a], R[a][tag], f"{ds} {tag} {a}")
    S = I24 / f"partB/ihp_{T}K/sel"
    sel = load(S / "results.json")
    corr = SymCorrection.from_json(load(S / "corrections.json")[sel["names"][tag]])
    new = p22.simulate_card(ds, tag, dict(load(S / "heldout_cards.json")[tag]))
    sims["B1+g v2"] = corr.apply(new)
    sc["B1+g v2"] = score_device(new, sims["B1+g v2"])["rrms"]
    check(sc["B1+g v2"], R["B1+g v2"][tag], f"{ds} {tag} B1+g v2")
    p, c2, lam2 = p22.g2_card_corr(ds, tag)
    new = p22.simulate_card(ds, tag, p)
    sims["selected g v2"] = c2.apply(new)
    sc["selected g v2"] = score_device(new, sims["selected g v2"])["rrms"]
    check(sc["selected g v2"], R["selected g v2"][tag], f"{ds} {tag} selected g v2")
    lam = sel["lambda_star"][tag]
    lines = [("published", GREY, "--"), ("C0_refit", BLUE, "-"), ("selected g v2", AQUA, ":"), ("B1", VIOLET, "-"),
             ("B1+g v2", RED, "-.")]
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=4,
                      label=f"measured {T} K (filled: scored; hollow: trimmed before the metric; faded: excluded "
                            "curve)"),
               Line2D([], [], color=GREY, ls="--", lw=1.2, label=f"NGSpice, stock card (no fit): RRMS "
                                                                 f"{sc['published']:.4f}"),
               Line2D([], [], color=BLUE, lw=1.2, label=f"NGSpice re-simulated, C0_refit held-out card: RRMS "
                                                        f"{sc['C0_refit']:.4f}"),
               Line2D([], [], color=AQUA, ls=":", lw=1.4,
                      label=f"NGSpice re-simulated, selected g v2 held-out card x exp(fold g) (lambda* {lam2:g} V): "
                            f"RRMS {sc['selected g v2']:.4f}"),
               Line2D([], [], color=VIOLET, lw=1.2, label=f"NGSpice re-simulated, B1 wider-card held-out card: RRMS "
                                                          f"{sc['B1']:.4f}"),
               Line2D([], [], color=RED, ls="-.", lw=1.2,
                      label=f"NGSpice re-simulated, B1+g v2 held-out card x exp(fold g) (lambda* {lam:g} V): RRMS "
                            f"{sc['B1+g v2']:.4f}")]
    p22.draw_iv(rec, sims, lines, handles,
                f"{tag} (IHP SG13G2 {T} K, DEV; {why}):\nheld-out cards re-simulated ({p22.SIMNAME['ihp']}); g "
                "applied in Python. RRMS equals the stored rows.", path)
    return [rel(I22 / f"partA_{ds}/heldout_cards.json"), rel(S / "heldout_cards.json"), rel(S / "corrections.json")]


def iv_choices_b(d: dict) -> list[tuple[str, str, str]]:
    B = d["dec"]["part_B"]["per_temperature"]
    out = []
    for T in TEMPS:
        R = B[T]["rows"]
        lr = {t: math.log(R["B1+g v2"][t] / R["B1"][t]) for t in R["B1"] if np.isfinite(R["B1+g v2"][t])}
        lo, hi = min(lr, key=lr.get), max(lr, key=lr.get)
        out.append((T, lo, f"lowest ln(B1+g v2 / B1) {lr[lo]:+.3f}"))
        out.append((T, hi, f"highest ln(B1+g v2 / B1) {lr[hi]:+.3f}"))
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

    dec = [rel(I24 / "decision.json")]
    add_table(t_setup(d), "I24-1 setup (pre-registered)", [rel(I24 / "prereg.json")])
    add_table(t_a_validity(d), "Part A validity per dataset", dec)
    add_table(t_a_stats(d), "Part A gate statistics and choices", dec + [rel(I24 / "partA")])
    add_table(t_a_gmr(d), "Part A GMR vs C0_refit per dataset and pooled", dec)
    add_table(t_a_pooled(d), "Part A pooled GMR, paired pooled ratios and rule outcome", dec)
    add_table(t_a_rrms(d), "Part A RRMS per series", dec)
    add_table(t_c(d), "Part C TEST decisions and proposed amendment", dec + ["out/research/explore/I19_1/"
                                                                          "test_gate_decisions.json"])
    add_table(t_b_validity(d), "Part B validity per temperature", dec)
    add_table(t_b_rrms(d), "Part B RRMS per arm", dec)
    add_table(t_b_gmr(d), "Part B GMR with labels and lambda* counts", dec)
    add_table(t_b_calls(d), "Part B NGSpice calls per held-out DUT", dec)
    k = len(items) + 1
    for name, fn, desc in (("forest_a", fig_forest_a, "Part A forest plot: GMR vs C0_refit of g always, B1 always, "
                                                     "two-way and three-way gate per dataset and pooled"),
                           ("pooled_a", fig_pooled, "Part A pooled GMR vs C0_refit per series with stratified CI"),
                           ("stat_scatter_a", fig_stat_scatter, "Part A gate statistic vs realised ln GMR per dataset"),
                           ("forest_b", fig_forest_b, "Part B forest plot per temperature"),
                           ("per_device_b", fig_b_per_device, "Part B per-DUT log ratios")):
        png = f"{k:02d}_i24_1_{name}.png"
        items.append((png, desc, fn(d, folder / png)))
        k += 1
    if not args.no_iv:
        for T, tag, why in iv_choices_b(d):
            png = f"{k:02d}_i24_1_iv_b_ihp{T}_{tag}.png"
            items.append((png, f"Part B I-V of {tag} (IHP {T} K; {why}): measured points, NGSpice stock card, "
                               "re-simulated C0_refit, selected g v2, B1 and B1+g v2 held-out cards",
                          fig_iv_b(d, T, tag, why, folder / png)))
            k += 1
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_explore_i24_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    A, B, C = d["dec"]["part_A"], d["dec"]["part_B"], d["dec"]["part_C"]
    P = A["pooled"]
    lines = [f"Part A outcome: {A['outcome']['label']}.",
             "Three-way choices: " + ", ".join(f"{ds} {ARM[A['datasets'][ds]['choice']['arm']]}" for ds in DS) + ".",
             "Pooled GMR vs C0_refit: " + ", ".join(f"{k} {P[k]['pooled']:.3f} [{P[k]['ci_low']:.3f}, "
                                                     f"{P[k]['ci_high']:.3f}]" for k in
                                                     ("three-way gate", "two-way frozen gate", "B1 always",
                                                      "g always", "S3 leave-group-out gate")) + ".",
             f"Paired pooled three-way / two-way: {ci(A['paired_pooled']['three-way / two-way'])}.",
             "Part C proposed amendment (not committed): " + ("; ".join(C["proposed_amendment"]) or "none") + ".",
             f"Part B outcome: {B['outcome']['label']}."]
    for T in TEMPS:
        c = B["per_temperature"][T]["comparisons"]
        lines.append(f"IHP {T} K: GMR(B1+g v2, B1) {ci(c['B1+g v2 vs B1'])}; GMR(B1+g v2, C0_refit) "
                     f"{ci(c['B1+g v2 vs C0_refit'])}; GMR(B1_cont, B1) {ci(c['B1_cont vs B1'])}")
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. NGSpice runs: I-V re-simulation of the stock cards and the held-out cards; "
              f"every re-simulated RRMS checked against the stored rows; {d['n_checked']} stored values recomputed.",
              "", "## Contents", "",
              f"- I24-1 three-way dataset-level gate and g v2 on top of B1. Pre-registration "
              f"`out/research/explore/I24_1/prereg.md` ({COMMITS['prereg']}). DEV only.",
              *[f"- {x}" for x in lines], "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items],
                    ["I24-1 three-way dataset-level gate and g v2 on top of B1."] + lines)
    print("\n".join(lines))
    print("\n".join(str(folder / i[0]) for i in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
