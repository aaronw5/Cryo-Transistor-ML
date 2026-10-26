#!/usr/bin/env python3
"""Report batch for I22-1: matched-budget baselines.

(A) correction-stage baselines B1 (wider card set), B2 (extrinsic resistance
and self-heating), B3_d3 (cubic polynomial g), B4 (GP discrepancy) under the
selected g v2 protocol on the 9 measured DEV datasets; (B) NGSpice calls per
arm; (C) CMA-ES and TuRBO-1 at the C0_refit NGSpice-call budget on sky130
77 K.

Reads ``out/research/explore/I22_1/`` (prereg, screen_<ds>.json,
partA_<ds>/, partC/, cost.json, decision.json) and the stored C0_refit,
M_bias v1 and selected g v2 files they refer to.

NGSpice: the I-V figures re-simulate the published / stock card and the
held-out cards (C0_refit, selected g v2, B1, B2, B3_d3, B4; Part C: LS, CMA-ES
and TuRBO cards at B_d) with the simulator and state of each dataset; g is
applied in Python.  Every re-simulated RRMS is checked against the stored row.

  .venv/bin/python -I scripts/research/plot_explore_i22_1.py --date 2026-10-26
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
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
os.environ.setdefault("CRYOML_E1_RECORDS", str(ROOT / "out/research/explore/I13_1/records"))
import plot_explore_e2_e3_e4 as pe  # noqa: E402
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, AQUA, VIOLET, RED, DPI,  # noqa: E402
                                   EXP, load, rel, style)
from matplotlib.lines import Line2D  # noqa: E402

from cryoml.explore import i22_1_baselines as m  # noqa: E402

rf = pe.rf
plt = pe.plt

TOPIC = "i22_1"
I22 = EXP / "I22_1"
COMMITS = {"prereg": "02113d4", "code": "422551a"}
ARM_COLOR = {"C0_refit": BLUE, "M_bias_v1": ORANGE, "selected_g_v2": AQUA, "B1": VIOLET, "B2": "#b8860b",
             "B3_d3": RED, "B4": "#495057", "published": GREY}
ARM_LABEL = {"C0_refit": "C0_refit", "M_bias_v1": "M_bias v1", "selected_g_v2": "selected g v2",
             "B1": "B1 wider card set", "B2": "B2 R_ext + self-heating", "B3_d3": "B3_d3 cubic g",
             "B4": "B4 GP discrepancy"}
METH_COLOR = {"LS": BLUE, "cma_primary": RED, "cma_secondary": "#e5989b", "turbo_primary": VIOLET,
              "turbo_secondary": "#c8b6ff", "LS_secondary": "#8ecae6"}
METH_LABEL = {"cma_primary": "CMA-ES sigma0 0.1", "cma_secondary": "CMA-ES sigma0 0.03",
              "turbo_primary": "TuRBO-1 L_init 0.1", "turbo_secondary": "TuRBO-1 L_init 0.03"}


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext", group=None):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": group}


def check(a: float, b: float, what: str, tol: float = 1e-9) -> None:
    if not (abs(a - b) <= tol or (not np.isfinite(a) and not np.isfinite(b))):
        raise AssertionError(f"{what}: {a} vs {b}")


def kfmt(x) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:,.0f}"


# ---------------------------------------------------------------------------
# load + recompute
# ---------------------------------------------------------------------------
def load_all() -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    d = {"prereg": load(I22 / "prereg.json"), "dec": load(I22 / "decision.json"), "cost": load(I22 / "cost.json")}
    A = d["dec"]["part_A"]
    n = 0
    for ds in m.DATASETS:
        R = A["rows"][ds]
        cl = None
        if ds.startswith("tud"):
            from cryoml.explore.i6_1_tudelft import group_of
            cl = {t: group_of(t) for t in R["C0_refit"]}
        for x in m.BASELINES:
            g = paired_ratio(R[x], R["selected_g_v2"], cl)
            s = A["datasets"][ds]["vs_g_v2"][x]
            for k in ("ratio", "ci_low", "ci_high"):
                check(g[k], s[k], f"{ds} {x} vs g v2 {k}", 1e-12)
                n += 1
            g = paired_ratio(R[x], R["C0_refit"], cl)
            check(g["ratio"], A["datasets"][ds]["vs_C0_refit"][x]["ratio"], f"{ds} {x} vs C0", 1e-12)
            n += 1
        # stored comparison rows equal the freeze-draft rows
        fd = load(EXP / "I15_2/decision.json") if ds == "sky130" else None
        if fd is not None:
            for t, v in fd["selected_rows"].items():
                check(float(v), R["selected_g_v2"][t], f"sky130 selected {t}", 1e-15)
                n += 1
    d["n_checked"] = n
    d["screens"] = {ds: load(I22 / f"screen_{ds}.json") for ds in m.DATASETS}
    d["partA"] = {ds: load(I22 / f"partA_{ds}/results.json") for ds in m.DATASETS}
    C = I22 / "partC"
    d["ls"] = load(C / "ls.json")
    d["opt"] = {f"{a}_{b}": load(C / f"{a}_{b}.json") for a in ("cma", "turbo") for b in ("primary", "secondary")
                if (C / f"{a}_{b}.json").exists()}
    return d


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_setup(d: dict) -> rf.Table:
    P = d["prereg"]
    rows = [{"c": "pre-registration", "v": f"out/research/explore/I22_1/prereg.md (commit {COMMITS['prereg']}); "
                                         f"code {COMMITS['code']}; amendment A1 (before any run): TuRBO runs stop at "
                                         "B_d; DEV only, TEST locked; no amendment of a755e18"},
            {"c": "Part A datasets", "v": "; ".join(f"{k}: {v['records']}, {v['folds']}"
                                                    for k, v in P["part_A"]["datasets"].items())},
            {"c": "Part A arms", "v": "; ".join(f"{k}: {v}" for k, v in P["part_A"]["arms"].items())},
            {"c": "Part A comparison rows", "v": P["part_A"]["comparison_rows"]},
            {"c": "Part A outcome rule", "v": P["part_A"]["outcome_rule"]},
            {"c": "Part B", "v": P["part_B"]["unit"] + "; report only"},
            {"c": "Part C", "v": f"{P['part_C']['problem']}; objective {P['part_C']['objective']}; "
                                 f"budget {P['part_C']['budget']} (A1: TuRBO to B_d); {P['part_C']['statistic']}"}]
    return rf.Table("01_table_i22_1_setup", "I22-1 setup (pre-registered)",
                    [col("c", "item", kind="text"), col("v", "value", kind="text")], rows, [])


def t_validity(d: dict) -> rf.Table:
    rows = []
    A = d["dec"]["part_A"]["datasets"]
    for ds in m.DATASETS:
        v = A[ds]["validity"]
        va5 = v.get("VA5_ihp_vs_I5_1")
        rows.append({"ds": m.NAMES[ds], "va1": f"{v['VA1_c0_max_abs_diff']:.1e}",
                     "va2": ", ".join(f"{a} {k}" for a, k in v["VA2_n_failed"].items() if k) or "0 failed",
                     "va3": str(v["VA3_unit_tests"]), "va4": str(v["VA4"]),
                     "va5": ", ".join(f"{a} {x:.1e}" for a, x in va5.items()) if va5 else "-",
                     "res": "valid" if v["valid"] else "invalid"})
    return rf.Table("02_table_i22_1_a_validity", "Part A validity per dataset (VA1-VA5)",
                    [col("ds", "dataset", kind="text"), col("va1", "VA1 max |C0_refit rerun - stored| (<= 1e-12)"),
                     col("va2", "VA2 non-finite rows"), col("va3", "VA3 unit tests"), col("va4", "VA4 DEV only"),
                     col("va5", "VA5 IHP rerun vs I5-1 (reported)", kind="text"), col("res", "result", kind="text")],
                    rows, ["VA1: the C0_refit rerun through the I22-1 refit path against the stored C0_refit rows of "
                           "the g v2 comparison (state C; IHP no clamp)."])


def t_rrms(d: dict) -> rf.Table:
    rows = []
    A = d["dec"]["part_A"]["datasets"]
    arms = ("C0_refit", "M_bias_v1", "selected_g_v2", "B1", "B2", "B3_d3", "B4")
    for ds in m.DATASETS:
        s = A[ds]["summary"]
        r = {"ds": m.NAMES[ds], "n": f"{A[ds]['n']} ({A[ds]['unit']})"}
        for a in arms:
            r[a] = f"{s[a]['all_device_mean']:.4f} / {s[a]['combined']:.4f}" + (
                f" ({s[a]['n_failed']} failed)" if s[a]["n_failed"] else "")
        rows.append(r)
    return rf.Table("03_table_i22_1_a_rrms", "Part A held-out RRMS per arm: all-device mean / combined (mean of the "
                    "nMOS and pMOS family means)",
                    [col("ds", "dataset", kind="text"), col("n", "n (CI unit)")] +
                    [col(a, ARM_LABEL[a]) for a in arms], rows,
                    ["C0_refit, M_bias v1, selected g v2: stored rows (freeze-draft loader). B1-B4: I22-1 rows. Strict "
                     "paper RRMS, frozen curve inclusion; a failed (non-finite) device is left out of the means and "
                     "counted."])


def t_vs_g(d: dict) -> rf.Table:
    rows = []
    A = d["dec"]["part_A"]["datasets"]
    for ds in m.DATASETS:
        r = {"ds": m.NAMES[ds] + (" (G)" if ds in m.GAIN_SET else "")}
        for x in m.BASELINES:
            c = A[ds]["vs_g_v2"][x]
            lab = A[ds]["label_vs_g_v2"][x].replace(f"{x} better than g v2", "X better").replace(
                f"g v2 better than {x}", "g v2 better")
            r[x] = f"{ci(c)} {c['wins']}/{c['n_used']}; {lab}" + ("; NI" if A[ds]["non_inferior_to_g_v2"][x] else "")
        rows.append(r)
    return rf.Table("04_table_i22_1_a_vs_g_v2", "Part A GMR(baseline, selected g v2) with 95% bootstrap CI, devices "
                    "lower / paired, label; NI: non-inferior (CI upper < 1.05)",
                    [col("ds", "dataset", kind="text")] + [col(x, ARM_LABEL[x]) for x in m.BASELINES], rows,
                    ["GMR < 1: the baseline has lower RRMS than selected g v2. Bootstrap 10000, seed 0; unit device "
                     "(TU Delft: 7 geometry groups). (G): gain set of the outcome rule."])


def t_vs_c0(d: dict) -> rf.Table:
    rows = []
    A = d["dec"]["part_A"]["datasets"]
    arms = ("selected_g_v2", "M_bias_v1") + m.BASELINES
    for ds in m.DATASETS:
        r = {"ds": m.NAMES[ds]}
        for a in arms:
            c = A[ds]["vs_C0_refit"][a]
            r[a] = f"{ci(c)} {c['wins']}/{c['n_used']}; {A[ds]['label_vs_C0_refit'][a]}"
        rows.append(r)
    pooled = d["dec"]["part_A"]["pooled_gmr_vs_C0_refit"]
    rows.append({"ds": "pooled exp(mean ln GMR), 9 datasets", **{a: f"{pooled[a]:.3f}" for a in arms}})
    return rf.Table("05_table_i22_1_a_vs_c0", "Part A GMR(arm, C0_refit) with 95% bootstrap CI and label "
                    "(gain / worse / inconclusive); pooled over the 9 datasets",
                    [col("ds", "dataset", kind="text")] + [col(a, ARM_LABEL[a]) for a in arms], rows, [])


def t_outcome(d: dict) -> rf.Table:
    D = d["dec"]
    oc = D["part_A"]["outcome"]
    rows = []
    for x in m.BASELINES:
        pb = oc["per_baseline"][x]
        rows.append({"x": ARM_LABEL[x], "ni": ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in pb["non_inferior"].items()),
                     "gb": ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in pb["g_v2_better"].items()),
                     "mt": "yes" if pb["matches_g_v2"] else "no", "ah": "yes" if pb["g_v2_ahead"] else "no"})
    notes = [f"Outcome: {oc['label']}."]
    for k, v in D["predictions"].items():
        notes.append(f"{k}: {v}")
    return rf.Table("06_table_i22_1_a_outcome", "Part A outcome rule over the gain set G = {sky130, NIST lvt, NIST RT "
                    "card}, and prediction scores",
                    [col("x", "baseline", kind="text"), col("ni", "non-inferior to g v2 (CI upper < 1.05)", kind="text"),
                     col("gb", "g v2 better (CI lower > 1)", kind="text"), col("mt", "matches g v2"),
                     col("ah", "g v2 ahead")], rows, notes)


def t_cost(d: dict) -> rf.Table:
    rows = []
    C = d["cost"]["datasets"]
    for ds in m.DATASETS:
        c = C[ds]
        first = True
        for a in ("C0_refit", "M_bias_v1", "selected_g_v2", "B1", "B2", "B3_d3", "B4"):
            r = c["arms"][a]
            pre = r["pre_calls_total"] / c["n"]
            note = ""
            if a == "selected_g_v2":
                note = (f"{r['selection_refits']} selection refits ({'stored' if 'stored' in r['pre_note'] else 'estimated'})"
                        f"; {r['penalty_grid_decks']} grid decks x {r['penalty_grid_points']} points; full-data "
                        f"selection {r['full_data_selection_refits']} refits")
            elif a == "B1":
                note = r.get("pre_note", "")
            rows.append({"ds": m.NAMES[ds] if first else "", "st": kfmt(c["starting_cards"]["per_device_mean"]) if first else "",
                         "a": ARM_LABEL[a], "pre": kfmt(pre), "ref": kfmt(r["refit_calls_mean"]),
                         "tot": kfmt(r["total_calls_per_device"]), "rat": f"{r['ratio_to_C0_refit_refit_calls']:.1f}",
                         "tr": str(r["trainings"]),
                         "cpu": kfmt(r["train_cpu_s"]) if r.get("train_cpu_s") is not None else "not stored",
                         "note": note})
            first = False
    return rf.Table("07_table_i22_1_b_cost", "Part B NGSpice calls per held-out device by arm (one call = one "
                    "simulation of the device's measured bias set), trainings and training CPU seconds",
                    [col("ds", "dataset", kind="text"), col("st", "starting card calls / device (shared)"),
                     col("a", "arm", kind="text"), col("pre", "pre-refit calls / device"),
                     col("ref", "held-out refit calls / device"), col("tot", "arm total / device"),
                     col("rat", "arm total / C0_refit"), col("tr", "trainings"), col("cpu", "training CPU s"),
                     col("note", "note", kind="text")], rows,
                    ["Pre-refit calls: B1 edge screen; selected g v2 nested lambda selection inner refits for the LODO "
                     "evaluation (NIST: stored n_sim; other datasets: inner refits x median held-out n_sim, estimate). "
                     "S3_D gate: 0 calls (one v1 training per fold, 500 steps). Penalty grids are one DC deck per "
                     "device and are not converted to calls. M_bias v1 training CPU seconds are not stored per member."])


def t_partc(d: dict) -> rf.Table:
    C = d["dec"]["part_C"]
    rows = [{"m": "LS (TRF refit, max_nfev 400)", "b": f"{np.mean(list(C['budget'].values())):.0f}",
             "v": ci(C["LS"]["vs_start"]), "g": "1", "lab": "reference", "s": "-",
             "w": f"{np.mean([r['seconds'] for r in d['ls']['primary'].values()]):.1f}"}]
    for k, v in C["methods"].items():
        rows.append({"m": METH_LABEL[k], "b": f"{np.mean(list(C['budget'].values())):.0f}" +
                     (" (runs to 4 B_d)" if v.get("budget_factor") == 4 else ""),
                     "v": ci(v["vs_start"]), "g": ci(v["vs_LS"]) + f" {v['vs_LS']['wins']}/{v['vs_LS']['n_used']}",
                     "lab": v["label"] + (" (primary)" if k.endswith("primary") else " (secondary config)"),
                     "s": ", ".join(f"{s}: {r:.3f}" for s, r in v["per_seed_vs_LS"].items()),
                     "w": f"{v['run_seconds_mean']:.1f}"})
    notes = [f"Validity: VC1 {C['validity']['VC1']} (max |LS - stored C0_refit| {C['validity']['VC1_max_abs_diff']:.1e}), "
             f"VC2 {C['validity']['VC2']}, VC3 {C['validity']['VC3_unit_tests']}."]
    sec = [f"{METH_LABEL[k]} at 4 B_d vs LS max_nfev 1600: {ci(v['vs_LS_4B_secondary'])}"
           for k, v in C["methods"].items() if "vs_LS_4B_secondary" in v]
    notes += sec
    return rf.Table("08_table_i22_1_c", "Part C sky130 77 K (18 devices): best-so-far paper RRMS at the matched "
                    "NGSpice-call budget B_d (= LS n_sim per device), geometric mean over 3 seeds",
                    [col("m", "method", kind="text"), col("b", "mean B_d (calls)"),
                     col("v", "GMR vs starting card [CI]"), col("g", "GMR vs LS [CI], devices lower"),
                     col("lab", "label", kind="text"), col("s", "per-seed GMR vs LS", kind="text"),
                     col("w", "mean run time (s)")], rows, notes)


def t_b1(d: dict) -> rf.Table:
    rows = []
    for ds in m.DATASETS:
        s = d["screens"][ds]
        if "selected_by_class" in s:
            sel = "; ".join(f"{c}: {len(v)}" for c, v in s["selected_by_class"].items())
            names = "stored I5-1 per-class screen"
        else:
            sel = f"{len(s['selected'])} of {len(s['candidates'])}"
            names = ", ".join(s["selected"])
        A = d["partA"][ds]
        b1 = [r for k, r in A["rows"].items() if k.startswith("B1|")]
        b2 = [r for k, r in A["rows"].items() if k.startswith("B2|")]
        rows.append({"ds": m.NAMES[ds], "sel": sel, "names": names,
                     "na": f"{np.median([r.get('n_active', np.nan) for r in b1]):.0f}",
                     "nf": f"{np.median([r.get('max_nfev', np.nan) for r in b1]):.0f}",
                     "rx": f"{np.median([r.get('rext_ohm', np.nan) / r.get('rext_max_ohm', np.nan) for r in b2]):.2f}",
                     "hk": f"{np.median([r.get('heat_kappa', np.nan) for r in b2]):+.3f}"})
    return rf.Table("09_table_i22_1_a_b1_b2", "Part A B1 screen selections (state C) and B2 fitted elements",
                    [col("ds", "dataset", kind="text"), col("sel", "B1 extras selected"),
                     col("names", "B1 names", kind="text"), col("na", "B1 median active parameters"),
                     col("nf", "B1 median max_nfev (matched)"), col("rx", "B2 median R_ext / upper bound"),
                     col("hk", "B2 median kappa")], rows, [])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_forest_g(d: dict, path: Path) -> list[str]:
    A = d["dec"]["part_A"]["datasets"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.2), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, x in zip(axes, m.BASELINES):
        style(ax)
        for i, ds in enumerate(m.DATASETS):
            c = A[ds]["vs_g_v2"][x]
            colr = ARM_COLOR[x]
            ax.plot([c["ci_low"], c["ci_high"]], [i, i], color=colr, lw=2)
            ax.plot(c["ratio"], i, "o", color=colr, ms=5)
            ax.text(max(c["ci_high"], 1.0) * 1.02, i, f"{c['ratio']:.3f}", va="center", fontsize=7, color=INK)
        ax.axvline(1.0, color=GREY, lw=1, ls="--")
        ax.axvline(1.05, color=GREY, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"GMR({ARM_LABEL[x]}, selected g v2)", fontsize=9, color=INK)
        ax.set_xlabel("ratio (log); dashed 1, dotted 1.05")
    axes[0].set_yticks(range(len(m.DATASETS)))
    axes[0].set_yticklabels([m.NAMES[ds] + (" (G)" if ds in m.GAIN_SET else "") for ds in m.DATASETS], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Part A: baselines vs selected g v2 per dataset (95% bootstrap CI; > 1: higher RRMS than g v2); "
                 "G: gain set of the outcome rule", fontsize=9.6, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I22 / "decision.json")]


def fig_forest_c0(d: dict, path: Path) -> list[str]:
    A = d["dec"]["part_A"]["datasets"]
    arms = ("selected_g_v2", "M_bias_v1") + m.BASELINES
    fig, ax = plt.subplots(figsize=(12, 9.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    k = len(arms) + 1
    for i, ds in enumerate(m.DATASETS):
        for j, a in enumerate(arms):
            c = A[ds]["vs_C0_refit"][a]
            y = i * k + j
            ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=ARM_COLOR[a], lw=1.8)
            ax.plot(c["ratio"], y, "o", color=ARM_COLOR[a], ms=4, label=ARM_LABEL[a] if i == 0 else None)
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks([i * k + (len(arms) - 1) / 2 for i in range(len(m.DATASETS))])
    ax.set_yticklabels([m.NAMES[ds] for ds in m.DATASETS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("GMR(arm, C0_refit), 95% bootstrap CI (log scale); < 1: lower held-out RRMS than C0_refit")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Part A: every arm vs C0_refit per dataset", fontsize=9.6, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I22 / "decision.json")]


def fig_cost(d: dict, path: Path) -> list[str]:
    C = d["cost"]["datasets"]
    arms = ("C0_refit", "M_bias_v1", "selected_g_v2", "B1", "B2", "B3_d3", "B4")
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), dpi=DPI, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax, ds in zip(axes.ravel(), m.DATASETS):
        style(ax)
        c = C[ds]
        y = np.arange(len(arms))
        ref = np.array([c["arms"][a]["refit_calls_mean"] for a in arms])
        pre = np.array([c["arms"][a]["pre_calls_total"] / c["n"] for a in arms])
        ax.barh(y, ref, color=[ARM_COLOR[a] for a in arms], label="held-out refit")
        ax.barh(y, pre, left=ref, color=[ARM_COLOR[a] for a in arms], alpha=0.35, hatch="//", label="pre-refit")
        for yy, (r_, p_) in enumerate(zip(ref, pre)):
            ax.text(r_ + p_, yy, f" {r_ + p_:,.0f}", va="center", fontsize=7, color=INK)
        ax.axvline(c["starting_cards"]["per_device_mean"], color=GREY, ls=":", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([ARM_LABEL[a] for a in arms], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_title(f"{m.NAMES[ds]} (dotted: starting-card calls / device "
                     f"{c['starting_cards']['per_device_mean']:,.0f})", fontsize=8.2, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("NGSpice calls per held-out device (log)")
    h = [Line2D([], [], color=GREY, lw=6, label="held-out refit calls"),
         Line2D([], [], color=GREY, lw=6, alpha=0.35, label="pre-refit calls (B1 screen; g v2 nested selection)")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle("Part B: NGSpice calls per held-out device and arm (selected g v2 selection calls estimated where "
                 "not stored; see the table)", fontsize=9.8, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I22 / "cost.json")]


def _curve(values, f0, B, xs, carry: bool = False):
    """best-so-far / f0 at round(x * B) calls; beyond the last call NaN, or the
    last value when ``carry`` (an LS run that stopped before the horizon)."""
    b = m.best_so_far(values)
    out = []
    for x in xs:
        k = int(round(x * B))
        out.append(b[min(max(k, 1), len(b)) - 1] / f0 if (k <= len(b) or carry) else np.nan)
    return np.array(out)


def fig_partc_curves(d: dict, path: Path) -> list[str]:
    ls = d["ls"]
    tags = ls["tags"]
    xs = np.linspace(0.0, 4.0, 161)[1:]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    style(ax)

    def gm(curves):
        a = np.array(curves)
        with np.errstate(invalid="ignore"):
            return np.exp(np.nanmean(np.log(a), axis=0)) if len(a) else None

    f0 = {t: ls["primary"][t]["trace"][0] for t in tags}
    B = {t: ls["budget"][t] for t in tags}
    c = gm([_curve(ls["primary"][t]["trace"], f0[t], B[t], xs) for t in tags])
    ax.plot(xs, c, color=METH_COLOR["LS"], lw=2, label="LS (TRF, max_nfev 400) = C0_refit")
    c = gm([_curve(ls["secondary"][t]["trace"], f0[t], B[t], xs, carry=True) for t in tags])
    ax.plot(xs, c, color=METH_COLOR["LS_secondary"], lw=1.6, ls="--", label="LS (TRF, max_nfev 1600)")
    for k, R in d["opt"].items():
        cs = [_curve(r["values"], f0[r["tag"]], r["budget"], xs) for r in R["rows"]]
        ax.plot(xs, gm(cs), color=METH_COLOR[k], lw=1.8 if k.endswith("primary") else 1.2,
                ls="-" if k.endswith("primary") else "-.", label=METH_LABEL[k] + " (3 seeds)")
    ax.axvline(1.0, color=GREY, lw=1, ls=":")
    ax.set_xlabel("NGSpice calls / B_d (B_d = LS calls on device d)")
    ax.set_ylabel("geo. mean of best-so-far RRMS / start RRMS")
    ax.set_title("best-so-far paper RRMS vs NGSpice calls (sky130 77 K, 18 devices, C0_refit stage;\n"
                 "geometric mean over devices and seeds; LS max_nfev 1600 held at its final value after it stops)",
                 fontsize=9.2)
    ax.legend(fontsize=7.8, frameon=False)
    ax = axes[1]
    style(ax)
    C = d["dec"]["part_C"]
    order = sorted(tags, key=lambda t: C["LS"]["value"][t])
    y = np.arange(len(order))
    for k, v in C["methods"].items():
        r = [v["value"][t] / C["LS"]["value"][t] for t in order]
        ax.plot(r, y, "o" if k.endswith("primary") else "x", color=METH_COLOR[k], ms=5, label=METH_LABEL[k])
    ax.axvline(1.0, color=GREY, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=7)
    ax.set_xlabel("RRMS at B_d / LS RRMS (seed geometric mean); > 1: higher than LS")
    ax.set_title("per device at the matched budget", fontsize=9.2)
    ax.legend(fontsize=7.8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I22 / "partC/ls.json")] + [rel(I22 / f"partC/{k}.json") for k in d["opt"]]


def fig_per_device(d: dict, ds: str, path: Path) -> list[str]:
    R = d["dec"]["part_A"]["rows"][ds]
    tags = sorted(R["C0_refit"], key=lambda t: (is_p(t), t))
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.18 * len(tags) + 1.5)), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    y = np.arange(len(tags))
    arms = ("C0_refit", "M_bias_v1") + m.BASELINES
    for j, a in enumerate(arms):
        v = [math.log(R[a][t] / R["selected_g_v2"][t]) if np.isfinite(R[a][t]) else np.nan for t in tags]
        ax.plot(v, y + (j - 2.5) * 0.11, "o", ms=3, color=ARM_COLOR[a], label=f"ln({ARM_LABEL[a]} / selected g v2)")
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(tags, fontsize=6 if len(tags) > 40 else 7)
    ax.invert_yaxis()
    ax.set_xlabel("per-device log RRMS ratio vs selected g v2 (> 0: higher RRMS than g v2)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title(f"Part A {m.NAMES[ds]}: per-device ratios", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I22 / "decision.json")]


def is_p(t: str) -> bool:
    return "pmos" in t or t.startswith("pfet")


# ---------------------------------------------------------------------------
# I-V re-simulation
# ---------------------------------------------------------------------------
def g2_card_corr(ds: str, tag: str):
    from cryoml.explore.i12_1_symm import SymCorrection
    from cryoml.explore.i6_1_tudelft import group_of
    if ds == "sky130":
        lam = float(load(EXP / "I15_2/decision.json")["lambda_star"][tag])
        if lam == 10.0:
            B, key = EXP / "I13_1/lodo", f"M_bias_v2|{tag}"
        else:
            B, key = EXP / f"I15_2/lodo_sym_e1_l{lam:g}", f"sym_e1_l{lam:g}|{tag}"
        return dict(load(B / "heldout_cards.json")[tag]), SymCorrection.from_json(load(B / "corrections.json")[key]), lam
    if ds.startswith("nist"):
        B = EXP / f"I20_1/lodo_{ds}"
        lam = load(B / "results.json")["lambda_by_device"][tag]
        return (dict(load(B / "heldout_cards.json")[tag]),
                SymCorrection.from_json(load(B / "corrections.json")[f"selected|{tag}"]), lam)
    if ds == "ihp300":
        lam = float(load(EXP / "I17_2/decision.json")["lambda_star"][tag])
        B = EXP / f"I17_2/lodo_sym_e1_l{lam:g}"
        return dict(load(B / "heldout_cards.json")[tag]), SymCorrection.from_json(
            load(B / "corrections.json")[f"sym_e1_l{lam:g}|{tag}"]), lam
    T = int(ds[3:])
    if ds.startswith("ihp"):
        B = EXP / f"I21_1/ihp_{T}K/sel"
        sel = load(B / "results.json")
        return dict(load(B / "heldout_cards.json")[tag]), SymCorrection.from_json(
            load(B / "corrections.json")[sel["names"][tag]]), float(sel["lambda_star"][tag])
    g = group_of(tag)
    lam = float(load(EXP / "I18_2/decision.json")["per_temperature"][str(T)]["lambda_star"][g])
    B = EXP / f"I18_2/lodo_{T}K_sym_e1_l{lam:g}"
    return dict(load(B / "heldout_cards.json")[tag]), SymCorrection.from_json(
        load(B / "corrections.json")[f"sym_e1_l{lam:g}|{g}"]), lam


def simulate_card(ds: str, tag: str, params: dict | None):
    """(record, raw sims) of ``params`` (None: published / stock card) with the
    dataset's refit problem (state C); B2 elements applied if present."""
    import i22_1_baselines as drv
    from cryoml.explore.i4_1_baselines import B2_NAMES, heat_factor, measured_pmax
    rec = drv.recs(ds)[tag]
    k = drv.kind(ds)
    card = None if params is None else {a: b for a, b in params.items() if a not in B2_NAMES}
    rext = 0.0 if params is None else float(params.get("rext_ohm", 0.0))
    kappa = 0.0 if params is None else float(params.get("heat_kappa", 0.0))
    if k == "sky130":
        box, start, make, sim_fn = drv.d4.base_setup(rec)
        base = make(box)
        sims = sim_fn(base)(card, rext)
        curves = base.curves
        to_rec = lambda s: drv.d4.record_from_sims(rec, base, s, params or {})  # noqa: E731
    elif k == "nist":
        from cryoml.explore import i8_1_nist as b8
        from cryoml.explore import i9_1_nist_rt as b9
        dev, binfo = drv.d20.devices(ds)[tag], drv.d20.bins(ds)[tag]
        cls = b8.MeasProblem if ds == "nist4" else b9.MeasProblemRT
        prob = cls(dev, binfo, binfo["stock"], b8.base_box(binfo["stock"]))
        sims = prob.simulate_params(card, rext)
        curves = prob.curves
        to_rec = lambda s: b8.to_record(dev, binfo, s, params)  # noqa: E731
    elif k == "ihp":
        from cryoml.explore import i5_1_ihp_real as m5
        dev = drv.ihp_devices()[tag]
        prob = m5.MeasProblem(dev, int(ds[3:]), m5.base_box(dev))
        p = m5.stock_params(dev, m5.BASE_SET) if card is None else card
        sims = prob.simulate_params(p, rext)
        curves = prob.curves
        to_rec = lambda s: m5.to_record(dev, int(ds[3:]), s, p)  # noqa: E731
    else:
        from cryoml.explore import i6_1_tudelft as m6
        dev = drv.d18.devices()[tag]
        prob = drv.d18.ProblemC(dev, int(ds[3:]), m6.base_box(dev))
        p = m6.stock_params(dev, m6.BASE_SET) if card is None else card
        sims = prob.simulate_params(p, rext)
        curves = prob.curves
        to_rec = lambda s: m6.to_record(dev, int(ds[3:]), s, p)  # noqa: E731
    sims = [np.asarray(s, float) for s in sims]
    if kappa != 0.0:
        pmax = measured_pmax(rec)
        sims = [s * heat_factor(np.asarray(c.Vd, float)[:len(s)], s, kappa, pmax) for c, s in zip(curves, sims)]
    return to_rec(sims)


def resim_device(d: dict, ds: str, tag: str) -> dict:
    import i22_1_baselines as drv
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i4_1_baselines import correction_from_json_any
    from cryoml.explore.i6_1_tudelft import group_of
    R = d["dec"]["part_A"]["rows"][ds]
    rec = drv.recs(ds)[tag]
    sims, sc = {}, {}
    if ds == "sky130":
        from plot_explore_i3_2 import published_rrms
        pub, pubc = published_rrms({tag: rec})
        sims["published"], sc["published"] = pubc[tag], pub[tag]
    else:
        r0 = simulate_card(ds, tag, None)
        sims["published"] = [c.sim for c in r0.curves]
        sc["published"] = score_device(r0, sims["published"])["rrms"]
    cards = load(I22 / f"partA_{ds}/heldout_cards.json")
    b3 = load(I22 / f"partA_{ds}/corrections_B3_d3.json")
    with gzip.open(I22 / f"partA_{ds}/corrections_B4.json.gz", "rt") as fh:
        b4 = json.load(fh)
    fold = group_of(tag) if ds.startswith("tud") else tag
    for a in ("C0_refit", "B1", "B2", "B3_d3", "B4"):
        new = simulate_card(ds, tag, dict(cards[f"{a}|{tag}"]))
        if a == "B3_d3":
            s = correction_from_json_any(b3[f"B3_d3|{fold}"]).apply(new)
        elif a == "B4":
            s = correction_from_json_any(b4[f"B4|{fold}"]).apply(new)
        else:
            s = [c.sim for c in new.curves]
        sims[a], sc[a] = s, score_device(new, s)["rrms"]
        check(sc[a], R[a if a != "C0_refit" else "C0_refit_rerun"][tag], f"{ds} {tag} {a}")
    p, corr, lam = g2_card_corr(ds, tag)
    new = simulate_card(ds, tag, p)
    s = corr.apply(new)
    sims["selected_g_v2"], sc["selected_g_v2"] = s, score_device(new, s)["rrms"]
    check(sc["selected_g_v2"], R["selected_g_v2"][tag], f"{ds} {tag} selected g v2")
    return {"rec": rec, "sims": sims, "score": sc, "lam": lam}


def draw_iv(rec, sims: dict, lines: list, handles: list, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.9), dpi=DPI)
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
                ax.plot(xv, np.abs(np.asarray(sims[a][i], float)), color=colr, ls=ls, lw=1.1, alpha=alpha, zorder=3)
        if kind == "idvg":
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-13)
            ax.set_xlabel("|Vgs| (V)")
            ax.set_title("Id-Vg (log scale)", fontsize=9.5)
        else:
            ax.set_xlabel("|Vds| (V)")
            ax.set_title("Id-Vd", fontsize=9.5)
        ax.set_ylabel("|Id| (A)")
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=7.8, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=9.2, color=INK)
    fig.tight_layout(rect=(0, 0.27, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


SIMNAME = {"sky130": "ngspice-41 arm64, state C", "nist4": "ngspice-41 arm64, 77 K lvt card, state C",
           "nistrt": "ngspice-41 arm64, room-temperature card, state C", "ihp": "ngspice-46 + psp103.osdi",
           "tud": "ngspice-41 arm64, CMOSedu BSIM4 card, state C"}


def fig_iv(d: dict, ds: str, tag: str, why: str, path: Path) -> list[str]:
    r = resim_device(d, ds, tag)
    sc = r["score"]
    pub_lab = ("NGSpice, published 77 K card (upstream committed sweep)" if ds == "sky130"
               else "NGSpice, stock card (no fit)")
    lines = [("published", GREY, "--"), ("C0_refit", BLUE, "-"), ("selected_g_v2", AQUA, "-"),
             ("B1", VIOLET, "-."), ("B2", ARM_COLOR["B2"], "-."), ("B3_d3", RED, ":"), ("B4", ARM_COLOR["B4"], ":")]
    meas_lab = {"sky130": "77 K", "nist4": "4 K", "nistrt": "4 K"}.get(ds, f"{ds[3:]} K")
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=4,
                      label=f"measured {meas_lab} (filled: scored points; hollow: trimmed before the metric; faded: "
                            "excluded curve)"),
               Line2D([], [], color=GREY, ls="--", lw=1.2, label=f"{pub_lab}: RRMS {sc['published']:.4f}"),
               Line2D([], [], color=BLUE, lw=1.2, label=f"NGSpice re-simulated, C0_refit held-out card: RRMS "
                                                        f"{sc['C0_refit']:.4f}"),
               Line2D([], [], color=AQUA, lw=1.2, label=f"NGSpice re-simulated, selected g v2 held-out card x exp(fold g) "
                                                        f"(lambda* {r['lam']:g} V): RRMS {sc['selected_g_v2']:.4f}"),
               Line2D([], [], color=VIOLET, ls="-.", lw=1.2, label=f"NGSpice re-simulated, B1 wider-card held-out card: "
                                                                   f"RRMS {sc['B1']:.4f}"),
               Line2D([], [], color=ARM_COLOR["B2"], ls="-.", lw=1.2,
                      label=f"NGSpice re-simulated, B2 held-out card with R_ext, x self-heating factor: RRMS "
                            f"{sc['B2']:.4f}"),
               Line2D([], [], color=RED, ls=":", lw=1.4, label=f"NGSpice re-simulated, B3_d3 held-out card x exp(fold "
                                                               f"cubic g): RRMS {sc['B3_d3']:.4f}"),
               Line2D([], [], color=ARM_COLOR["B4"], ls=":", lw=1.4,
                      label=f"NGSpice re-simulated, B4 held-out card x exp(fold GP mean): RRMS {sc['B4']:.4f}")]
    k = "ihp" if ds.startswith("ihp") else "tud" if ds.startswith("tud") else ds
    draw_iv(r["rec"], r["sims"], lines, handles,
            f"{tag} ({m.NAMES[ds]}, DEV; {why}): held-out cards re-simulated ({SIMNAME[k]}); g applied in Python. "
            "RRMS equals the stored rows.", path)
    return [rel(I22 / f"partA_{ds}/heldout_cards.json"), rel(I22 / f"partA_{ds}/corrections_B3_d3.json"),
            rel(I22 / f"partA_{ds}/corrections_B4.json.gz"), "stored selected g v2 held-out cards and corrections"]


def fig_iv_partc(d: dict, tag: str, path: Path) -> list[str]:
    import i22_1_baselines as drv
    from cryoml.explore.e1_correction import score_device
    from plot_explore_i3_2 import published_rrms
    rec = drv.recs("sky130")[tag]
    pub, pubc = published_rrms({tag: rec})
    sims, sc = {"published": pubc[tag]}, {"published": pub[tag]}
    ls = d["ls"]["primary"][tag]
    new = simulate_card("sky130", tag, dict(ls["params"]))
    sims["LS"] = [c.sim for c in new.curves]
    sc["LS"] = score_device(new, sims["LS"])["rrms"]
    check(sc["LS"], ls["best_traced"], f"Part C LS {tag}")
    for k in ("cma_primary", "turbo_primary"):
        if k not in d["opt"]:
            continue
        r = next(x for x in d["opt"][k]["rows"] if x["tag"] == tag and x["seed"] == 1)
        new = simulate_card("sky130", tag, dict(r["params_at_budget"]))
        sims[k] = [c.sim for c in new.curves]
        sc[k] = score_device(new, sims[k])["rrms"]
        check(sc[k], r["value_at_budget"], f"Part C {k} {tag}")
    lines = [("published", GREY, "--"), ("LS", BLUE, "-")] + [(k, METH_COLOR[k], "-.") for k in sims
                                                                if k.endswith("primary")]
    handles = [Line2D([], [], marker="o", ls="", color=INK, ms=4,
                      label="measured 77 K (filled: scored points; hollow: trimmed before the metric; faded: excluded "
                            "curve)"),
               Line2D([], [], color=GREY, ls="--", lw=1.2,
                      label=f"NGSpice, published 77 K card (upstream committed sweep): RRMS {sc['published']:.4f}"),
               Line2D([], [], color=BLUE, lw=1.2, label=f"NGSpice re-simulated, LS (TRF refit = C0_refit) card after "
                                                        f"{d['ls']['budget'][tag]} calls: RRMS {sc['LS']:.4f}")]
    for k in sims:
        if k.endswith("primary"):
            handles.append(Line2D([], [], color=METH_COLOR[k], ls="-.", lw=1.2,
                                  label=f"NGSpice re-simulated, {METH_LABEL[k]} best card after "
                                        f"{d['ls']['budget'][tag]} calls (seed 1): RRMS {sc[k]:.4f}"))
    draw_iv(rec, sims, lines, handles, f"{tag} (sky130 77 K, Part C): cards at the matched NGSpice-call budget, "
            "re-simulated in ngspice-41 (state C). RRMS equals the traced values.", path)
    return [rel(I22 / "partC/ls.json"), rel(I22 / "partC/cma_primary.json"), rel(I22 / "partC/turbo_primary.json")]


def iv_choices(d: dict) -> list[tuple[str, str, str]]:
    """Per selected dataset: the device where the best baseline (lowest RRMS of
    B1-B4) has the largest log advantage over selected g v2; on the gain set
    also the device with the largest g v2 advantage over the best baseline."""
    out = []
    for ds in ("sky130", "nist4", "nistrt", "ihp300", "tud4", "tud295"):
        R = d["dec"]["part_A"]["rows"][ds]
        lr = {}
        for t in R["C0_refit"]:
            best = min((R[x][t] for x in m.BASELINES if np.isfinite(R[x][t])), default=np.nan)
            lr[t] = math.log(best / R["selected_g_v2"][t])
        lo = min(lr, key=lr.get)
        out.append((ds, lo, f"lowest ln(best baseline / g v2) {lr[lo]:+.3f}"))
        if ds in m.GAIN_SET:
            hi = max(lr, key=lr.get)
            out.append((ds, hi, f"highest ln(best baseline / g v2) {lr[hi]:+.3f}"))
    return out


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

    dec = [rel(I22 / "decision.json")]
    add_table(t_setup(d), "I22-1 setup (pre-registered)", [rel(I22 / "prereg.json")])
    add_table(t_validity(d), "Part A validity per dataset", dec)
    add_table(t_rrms(d), "Part A RRMS per arm and dataset", dec)
    add_table(t_vs_g(d), "Part A GMR of each baseline vs selected g v2 with labels", dec)
    add_table(t_vs_c0(d), "Part A GMR of every arm vs C0_refit with labels, pooled", dec)
    add_table(t_outcome(d), "Part A outcome rule and prediction scores", dec)
    add_table(t_cost(d), "Part B NGSpice calls, trainings and training CPU per arm and dataset", [rel(I22 / "cost.json")])
    add_table(t_partc(d), "Part C matched-budget comparison of LS, CMA-ES and TuRBO-1", [rel(I22 / "partC/ls.json")])
    add_table(t_b1(d), "Part A B1 screen selections and B2 fitted elements",
              [rel(I22 / f"screen_{ds}.json") for ds in m.DATASETS])
    k = len(items) + 1
    for name, fn, desc in (("forest_vs_g_v2", fig_forest_g, "Part A forest plot: GMR(baseline, selected g v2) per "
                                                             "dataset and baseline"),
                           ("forest_vs_c0", fig_forest_c0, "Part A forest plot: GMR(arm, C0_refit) per dataset"),
                           ("cost", fig_cost, "Part B NGSpice calls per held-out device and arm"),
                           ("partc_curves", fig_partc_curves, "Part C best-so-far RRMS vs NGSpice calls / B_d and "
                                                             "per-device ratios at B_d")):
        png = f"{k:02d}_i22_1_{name}.png"
        items.append((png, desc, fn(d, folder / png)))
        k += 1
    for ds in ("sky130", "nist4", "nistrt"):
        png = f"{k:02d}_i22_1_per_device_{ds}.png"
        items.append((png, f"Part A {m.NAMES[ds]}: per-device log ratios vs selected g v2",
                      fig_per_device(d, ds, folder / png)))
        k += 1
    for ds, tag, why in iv_choices(d):
        png = f"{k:02d}_i22_1_iv_{ds}_{tag}.png"
        items.append((png, f"I-V of {tag} ({m.NAMES[ds]}; {why}): measured points, NGSpice published / stock card, "
                           "re-simulated C0_refit, selected g v2, B1, B2, B3_d3, B4 held-out cards",
                      fig_iv(d, ds, tag, why, folder / png)))
        k += 1
    C = d["dec"]["part_C"]
    if "cma_primary" in C["methods"]:
        lr = {t: math.log(C["methods"]["cma_primary"]["value"][t] / C["LS"]["value"][t]) for t in C["tags"]}
        tag = max(lr, key=lr.get)
        png = f"{k:02d}_i22_1_iv_partc_{tag}.png"
        items.append((png, f"Part C I-V of {tag} (highest ln(CMA-ES / LS) at B_d): measured points, published card, LS, "
                           "CMA-ES and TuRBO cards at the matched budget", fig_iv_partc(d, tag, folder / png)))
        k += 1
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_explore_i22_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    A = d["dec"]["part_A"]
    lines = [f"Part A outcome: {A['outcome']['label']}."]
    for ds in m.DATASETS:
        a = A["datasets"][ds]
        lines.append(f"{m.NAMES[ds]}: " + "; ".join(f"{x} vs g v2 {ci(a['vs_g_v2'][x])}, vs C0_refit "
                                                    f"{ci(a['vs_C0_refit'][x])}" for x in m.BASELINES))
    for kk, v in C["methods"].items():
        lines.append(f"Part C {METH_LABEL[kk]}: GMR vs LS at B_d {ci(v['vs_LS'])} ({v['label']}).")
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. NGSpice runs: I-V re-simulation of the published / stock cards and the "
              f"held-out cards of every arm; every re-simulated RRMS checked against the stored rows; "
              f"{d['n_checked']} stored values recomputed.", "", "## Contents", "",
              f"- I22-1 matched-budget baselines. Pre-registration `out/research/explore/I22_1/prereg.md` "
              f"({COMMITS['prereg']}); amendment A1 (before any run). DEV only.",
              *[f"- {x}" for x in lines], "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], ["I22-1 matched-budget baselines."] + lines)
    print("\n".join(lines))
    print("\n".join(str(folder / i[0]) for i in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
