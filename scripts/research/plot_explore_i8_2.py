#!/usr/bin/env python3
"""Report batch for I8-2: (A) pre-registered TEST predictions of the sign of
ln GMR(M_bias, C0_refit) from the DEV statistics S1_lodo (frozen I7-1 rule) and
S3; (B) the S3 gate G and the S1_lodo gate G1 evaluated retrospectively on the
11 stored and the 20 I7-1 simulated datasets.

Reads stored files only (no NGSpice, no training):

* ``out/research/explore/I8_2/prereg_test_predictions.json`` (Part A)
* ``out/research/explore/I8_2/gate_results.json`` (Part B)

Every per-dataset gate value (choice, GMR(G, C0_refit), GMR(G, M_bias)) and
every pooled value is recomputed from the stored y_D, S3(D), S1_lodo(D) and the
frozen rule, and checked against gate_results.json.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i8_2.py --date 2026-10-20
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
from plot_explore_e2_e3_e4 import (BG, INK, MUTED, GREY, BLUE, ORANGE, DPI, EXP,  # noqa: E402
                                   load, rel, style)

rf = pe.rf
plt = pe.plt

TOPIC = "i8_2"
I8 = EXP / "I8_2"
PRED = I8 / "prereg_test_predictions.json"
GATE = I8 / "gate_results.json"
GROUPS = ("stored_11", "measured_7", "new_20", "all_31")
GROUP_LABEL = {"stored_11": "11 stored datasets", "measured_7": "7 measured (subset of stored)",
               "new_20": "20 I7-1 simulated datasets", "all_31": "all 31 datasets"}
ARMS = ("G", "G1", "M_bias", "C0_refit", "oracle_reference", "diagnostic_fold_gate_S3",
        "diagnostic_fold_gate_S1")
ARM_LABEL = {"G": "G (S3 gate)", "G1": "G1 (S1_lodo rule gate)", "M_bias": "M_bias always",
             "C0_refit": "C0_refit always", "oracle_reference": "dataset oracle (reference)",
             "diagnostic_fold_gate_S3": "fold-level S3 gate (diagnostic)",
             "diagnostic_fold_gate_S1": "fold-level S1_lodo gate (diagnostic)"}
RETRO_LABEL = {"sky130_77K": "sky130 77 K", "ihp_meas_233K": "IHP meas 233 K",
               "ihp_meas_300K": "IHP meas 300 K", "ihp_meas_343K": "IHP meas 343 K",
               "ihp_meas_398K": "IHP meas 398 K", "tudelft_4K": "TU Delft 4.13 K",
               "tudelft_295K": "TU Delft 295 K", "gf180_c": "GF180 tier c", "ihp_c": "IHP tier c",
               "gf180_b": "GF180 tier b", "ihp_b": "IHP tier b"}


def label(name: str) -> str:
    if name in RETRO_LABEL:
        return RETRO_LABEL[name]
    pdk, rep, lam = name.split("|")
    return f"{'GF180' if pdk == 'gf180' else 'IHP'} r{rep} lambda {float(lam):g}"


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def recompute(g: dict, rule: dict) -> dict:
    """Recompute choices and pooled values from y, S3, S1_lodo; check stored values."""
    ds = g["datasets"]
    n_checked = 0
    for n, v in ds.items():
        y = v["y_M_bias_vs_C0"]
        ry = rule["a"] + rule["b"] * v["S1_lodo"]
        for arm, choose in (("G", v["S3"] < 0), ("G1", ry < 0)):
            lv = y if choose else 0.0
            assert v[arm]["choice"] == ("M_bias" if choose else "C0_refit"), (n, arm)
            assert close(v[arm]["ln_gmr_vs_C0"], lv) and close(v[arm]["ln_gmr_vs_M_bias"], lv - y)
            n_checked += 1
        assert close(v["rule_y"], ry)
    pooled = {}
    for grp, p in g["pooled"].items():
        names = {"stored_11": [n for n in ds if ds[n]["group"] == "stored"],
                 "measured_7": [n for n in ds if n.startswith(("sky130", "ihp_meas", "tudelft"))],
                 "new_20": [n for n in ds if ds[n]["group"] == "new"],
                 "all_31": list(ds)}[grp]
        assert len(names) == p["n_datasets"]
        for arm in ARMS:
            if arm == "M_bias":
                lv = [ds[n]["y_M_bias_vs_C0"] for n in names]
            elif arm == "C0_refit":
                lv = [0.0] * len(names)
            else:
                lv = [ds[n][arm]["ln_gmr_vs_C0"] for n in names]
            assert close(p[arm]["mean_ln_gmr_vs_C0"], float(np.mean(lv)))
            assert p[arm]["n_lose_to_C0_gt_2pct"] == sum(math.exp(x) > 1.02 for x in lv)
            assert close(p[arm]["worst_vs_C0"]["gmr"], math.exp(max(lv)))
            n_checked += 3
        pooled[grp] = p
    return {"n_checked": n_checked}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def t_predictions(pr: dict) -> rf.Table:
    rows = []
    for r in pr["predictions"]:
        rows.append({"ds": r["dataset"], "nt": str(r["n_test_devices"]), "nd": str(r["n_dev_devices"]),
                     "s1": f"{r['S1_lodo_stored']:+.4f}", "ry": f"{r['predicted_y_rule']:+.4f}",
                     "rg": f"{r['predicted_gmr_rule']:.3f}", "p1": r["S1_predicts"],
                     "s3": f"{r['S3_stored']:+.4f}", "p3": r["S3_predicts"],
                     "s1t": f"{r['secondary_S1_lodo_test_training_set']:+.4f}",
                     "p1t": r["secondary_S1_predicts"],
                     "dev": f"{r['dev_dataset_outcome_already_seen']['gmr_dev']:.3f}"})
    cols = [{"key": "ds", "head": "TEST dataset", "md": "TEST dataset", "kind": "text"},
            {"key": "nt", "head": "n TEST", "md": "n TEST devices", "kind": "rtext"},
            {"key": "nd", "head": "n DEV", "md": "n DEV devices", "kind": "rtext"},
            {"key": "s1", "head": "S1_lodo", "md": "S1_lodo (DEV, stored)", "kind": "rtext",
             "group": "frozen I7-1 rule", "family": True},
            {"key": "ry", "head": "rule y", "md": "predicted y", "kind": "rtext",
             "group": "frozen I7-1 rule", "family": True},
            {"key": "rg", "head": "rule GMR", "md": "predicted GMR", "kind": "rtext",
             "group": "frozen I7-1 rule", "family": True},
            {"key": "p1", "head": "S1 sign", "md": "S1_lodo prediction", "kind": "text",
             "group": "frozen I7-1 rule", "family": True},
            {"key": "s3", "head": "S3", "md": "S3 (DEV, stored)", "kind": "rtext", "group": "S3 < 0 rule"},
            {"key": "p3", "head": "S3 sign", "md": "S3 prediction", "kind": "text", "group": "S3 < 0 rule"},
            {"key": "s1t", "head": "S1_lodo", "md": "S1_lodo on TEST training set (secondary)",
             "kind": "rtext", "group": "secondary", "family": True},
            {"key": "p1t", "head": "sign", "md": "secondary prediction", "kind": "text",
             "group": "secondary", "family": True},
            {"key": "dev", "head": "DEV GMR", "md": "DEV GMR(M_bias, C0_refit), already seen",
             "kind": "rtext"}]
    fr = pr["frozen_rule"]
    notes = [
        f"Pre-registered {pr['written']} before any TEST curve was loaded, simulated or fitted "
        f"({rel(PRED)}; code at git {pr['code']['git_head_at_run'][:7]}). The final TEST run is planned "
        "for about 2026-12-10. 'gain' = predicted ln GMR(M_bias, C0_refit) < 0 on the TEST devices.",
        f"Frozen I7-1 rule: predicted y = {fr['a']:.6f} {fr['b']:+.6f} S1_lodo; gain iff S1_lodo > "
        f"{fr['zero_crossing_S1_lodo']:.4f}. S3 rule: gain iff S3 < 0. Primary S values = stored "
        "dataset-level DEV values (mean over the DEV leave-one-out folds). Secondary: S1_lodo on the TEST "
        "run's training set (all DEV devices; IHP per class, weighted 6 LV / 4 HV TEST DUTs).",
        "TEST scoring: M_bias (I3-2 recipe, 5 seeds, 2000 steps) trained on all DEV starting-card records "
        "(IHP: DEV DUTs of the same class and temperature), TEST card refit with and without M_bias in the "
        "loop (polish max_nfev 400, irls 2, diff_step 0.01) from the TEST starting card, strict paper RRMS, "
        "y = mean ln(RRMS_M_bias / RRMS_C0_refit) over the TEST devices.",
        "TEST data were not opened: every record read is in the DEV list of its split file. The DEV "
        "outcomes (last column) were known when the rule was fitted. The 4 IHP temperatures share DUTs and "
        "the 2 TU Delft temperatures share devices (3 device sets).",
    ]
    return rf.Table("01_table_i8_2_test_predictions", "I8-2 Part A: pre-registered TEST predictions of the "
                    "sign of ln GMR(M_bias, C0_refit)", cols, rows, notes)


def t_gate_datasets(g: dict) -> rf.Table:
    rows = []
    for n, v in g["datasets"].items():
        fg = v["diagnostic_fold_gate_S3"]
        rows.append({"grp": "stored" if v["group"] == "stored" else "new (I7-1)", "ds": label(n),
                     "n": str(v["n_devices"]), "s3": f"{v['S3']:+.4f}", "s1": f"{v['S1_lodo']:+.4f}",
                     "gm": f"{v['gmr_M_bias_vs_C0']:.3f}",
                     "gc": "M_bias" if v["G"]["choice"] == "M_bias" else "C0_refit",
                     "gc0": f"{v['G']['gmr_vs_C0']:.3f}", "gm0": f"{v['G']['gmr_vs_M_bias']:.3f}",
                     "g1c": "M_bias" if v["G1"]["choice"] == "M_bias" else "C0_refit",
                     "g1c0": f"{v['G1']['gmr_vs_C0']:.3f}", "g1m0": f"{v['G1']['gmr_vs_M_bias']:.3f}",
                     "fg": f"{fg['gmr_vs_C0']:.3f}",
                     "fd": f"{fg['n_folds_decision_differs_from_dataset_gate']}/{v['n_folds']}"})
    cols = [{"key": "grp", "head": "set", "md": "set", "kind": "text"},
            {"key": "ds", "head": "dataset", "md": "dataset", "kind": "text"},
            {"key": "n", "head": "n", "md": "held-out devices", "kind": "rtext"},
            {"key": "s3", "head": "S3", "md": "S3(D)", "kind": "rtext", "group": "statistics"},
            {"key": "s1", "head": "S1_lodo", "md": "S1_lodo(D)", "kind": "rtext", "group": "statistics"},
            {"key": "gm", "head": "M_bias/C0", "md": "GMR(M_bias, C0_refit)", "kind": "rtext",
             "group": "outcome"},
            {"key": "gc", "head": "choice", "md": "G choice", "kind": "text", "group": "G: S3 gate",
             "family": True},
            {"key": "gc0", "head": "vs C0", "md": "GMR(G, C0_refit)", "kind": "rtext", "group": "G: S3 gate",
             "family": True},
            {"key": "gm0", "head": "vs M_bias", "md": "GMR(G, M_bias)", "kind": "rtext",
             "group": "G: S3 gate", "family": True},
            {"key": "g1c", "head": "choice", "md": "G1 choice", "kind": "text", "group": "G1: S1_lodo rule"},
            {"key": "g1c0", "head": "vs C0", "md": "GMR(G1, C0_refit)", "kind": "rtext",
             "group": "G1: S1_lodo rule"},
            {"key": "g1m0", "head": "vs M_bias", "md": "GMR(G1, M_bias)", "kind": "rtext",
             "group": "G1: S1_lodo rule"},
            {"key": "fg", "head": "vs C0", "md": "fold-level S3 gate GMR vs C0_refit (diagnostic)",
             "kind": "rtext", "group": "fold-level S3 gate", "family": True},
            {"key": "fd", "head": "folds differ", "md": "folds whose decision differs from G",
             "kind": "rtext", "group": "fold-level S3 gate", "family": True}]
    notes = [
        "Retrospective: all outcomes were known before the gates were defined. G: M_bias on dataset D iff "
        "S3(D) < 0, else C0_refit; G1: M_bias iff the frozen I7-1 rule predicts a gain (S1_lodo(D) > "
        "0.0157). One choice per dataset from training-device statistics; GMR(G, C0_refit) = GMR(M_bias, "
        "C0_refit) if G chooses M_bias, else 1.",
        "S3 on the 11 stored datasets was computed after their outcomes were known; G1's rule was fitted on "
        "the 11 stored datasets (in-sample there) and frozen before the 20 simulated datasets were generated.",
        "Fold-level S3 gate (diagnostic): the decision for the held-out devices of an outer fold uses that "
        "fold's training-set S3_f, which excludes the scored devices. G's dataset-level S3(D) averages S3_f "
        "over all folds, so a scored device's own starting-card record enters G's decision through the other "
        "folds.",
    ]
    return rf.Table("02_table_i8_2_gate_datasets", "I8-2 Part B (retrospective): S3 gate G and S1_lodo gate G1 "
                    "per dataset", cols, rows, notes)


def worst(w: dict) -> str:
    """Largest GMR with its dataset; no dataset name when the largest value is 1 (ties)."""
    if abs(w["gmr"] - 1.0) < 1e-12:
        return "1.000 (no dataset above 1)"
    return f"{w['gmr']:.3f} ({label(w['dataset'])})"


def t_gate_pooled(g: dict) -> rf.Table:
    rows = []
    for grp in GROUPS:
        p = g["pooled"][grp]
        for arm in ARMS:
            a = p[arm]
            row = {"grp": GROUP_LABEL[grp], "arm": ARM_LABEL[arm],
                   "mc0": f"{a['mean_ln_gmr_vs_C0']:+.4f}", "gc0": f"{a['gmr_of_mean_vs_C0']:.3f}",
                   "wc0": worst(a["worst_vs_C0"]),
                   "lc0": f"{a['n_lose_to_C0_gt_2pct']}/{p['n_datasets']}",
                   "mm": f"{a['mean_ln_gmr_vs_M_bias']:+.4f}",
                   "wm": worst(a["worst_vs_M_bias"]),
                   "lm": f"{a['n_lose_to_M_bias_gt_2pct']}/{p['n_datasets']}"}
            if arm in ("G", "G1"):
                row["ch"] = f"{a['n_choose_M_bias']} M_bias, {a['n_correct_choice']} correct"
            rows.append(row)
    cols = [{"key": "grp", "head": "datasets", "md": "datasets", "kind": "text"},
            {"key": "arm", "head": "method", "md": "method", "kind": "text"},
            {"key": "ch", "head": "choices", "md": "choices (M_bias, sign correct)", "kind": "text"},
            {"key": "mc0", "head": "mean ln GMR", "md": "mean ln GMR(., C0_refit)", "kind": "rtext",
             "group": "vs C0_refit", "family": True},
            {"key": "gc0", "head": "exp(mean)", "md": "exp(mean ln GMR(., C0_refit))", "kind": "rtext",
             "group": "vs C0_refit", "family": True},
            {"key": "wc0", "head": "worst dataset", "md": "worst GMR(., C0_refit)", "kind": "text",
             "group": "vs C0_refit", "family": True},
            {"key": "lc0", "head": "> 1.02", "md": "datasets GMR(., C0_refit) > 1.02", "kind": "rtext",
             "group": "vs C0_refit", "family": True},
            {"key": "mm", "head": "mean ln GMR", "md": "mean ln GMR(., M_bias)", "kind": "rtext",
             "group": "vs M_bias"},
            {"key": "wm", "head": "worst dataset", "md": "worst GMR(., M_bias)", "kind": "text",
             "group": "vs M_bias"},
            {"key": "lm", "head": "> 1.02", "md": "datasets GMR(., M_bias) > 1.02", "kind": "rtext",
             "group": "vs M_bias"}]
    lk = g["leakage"]["observed"]
    near = lk["near_threshold"]
    notes = [
        "Retrospective. Pooled = equal weight per dataset. 'worst dataset' = largest GMR. The dataset oracle "
        "chooses with the outcome known; it is a reference bound, not a method. Fold-level gates are "
        "diagnostics (decision per outer fold from that fold's training-set statistic).",
        "Leakage of G: S3(D) contains each scored device's own starting-card record through the folds in "
        "which it is a training device (about 1/n_devices of S3(D)); the held-out refit outcomes are not used. "
        f"Datasets with |S3(D)| < 0.01: {', '.join(label(n) for n in near['datasets'])}; there a flipped "
        f"decision changes ln GMR(G, C0_refit) by at most |y_D| <= {near['max_abs_y']:.3f}. On TEST the "
        "gate uses DEV devices only and has no leakage of this kind.",
        "The 4 IHP measured temperatures share DUTs and the 2 TU Delft temperatures share devices; the 7 "
        "measured datasets come from 3 device sets.",
    ]
    return rf.Table("03_table_i8_2_gate_pooled", "I8-2 Part B (retrospective): pooled gate results over "
                    "datasets", cols, rows, notes)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def fig_gate(g: dict, path: Path) -> None:
    ds = g["datasets"]
    names = list(ds)
    fig, ax = plt.subplots(figsize=(10.5, 11.0), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    ypos = {n: i for i, n in enumerate(names)}
    x_m = [ds[n]["gmr_M_bias_vs_C0"] for n in names]
    x_g = [ds[n]["G"]["gmr_vs_C0"] for n in names]
    x_g1 = [ds[n]["G1"]["gmr_vs_C0"] for n in names]
    yy = np.array([ypos[n] for n in names], float)
    ax.axvline(1.0, color=INK, lw=0.9)
    ax.axvline(1.02, color=MUTED, lw=0.8, ls=":")
    ax.axhline(10.5, color=GREY, lw=0.6)
    ax.scatter(x_m, yy, s=70, facecolor="none", edgecolor=GREY, lw=1.3, zorder=3,
               label="M_bias on every dataset")
    ax.scatter(x_g, yy - 0.18, s=34, color=BLUE, zorder=4, label="G: M_bias iff S3(D) < 0")
    ax.scatter(x_g1, yy + 0.18, s=34, color=ORANGE, marker="s", zorder=4,
               label="G1: M_bias iff frozen S1_lodo rule predicts a gain")
    ax.set_xscale("log")
    ticks = [0.2, 0.3, 0.5, 0.7, 1.0, 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.16, 1.45)
    ax.set_yticks(yy)
    ax.set_yticklabels([label(n) for n in names], fontsize=7.5)
    ax.invert_yaxis()
    ax.text(1.42, 5.0, "11 stored datasets", rotation=90, va="center", ha="right", fontsize=8, color=MUTED)
    ax.text(1.42, 20.5, "20 I7-1 simulated datasets", rotation=90, va="center", ha="right", fontsize=8,
            color=MUTED)
    ax.text(1.021, len(names) - 0.2, "1.02", fontsize=7, color=MUTED, va="bottom")
    ax.set_xlabel("GMR(method, C0_refit) over the held-out devices of the dataset (log scale; < 1 = lower "
                  "RRMS than C0_refit)", fontsize=9)
    p = g["pooled"]["all_31"]
    ax.set_title("I8-2 (retrospective): S3 gate G and S1_lodo gate G1 vs C0_refit per dataset.\n"
                 f"All 31: exp(mean ln GMR) G {p['G']['gmr_of_mean_vs_C0']:.3f}, G1 "
                 f"{p['G1']['gmr_of_mean_vs_C0']:.3f}, M_bias {p['M_bias']['gmr_of_mean_vs_C0']:.3f}; "
                 f"datasets > 1.02: G {p['G']['n_lose_to_C0_gt_2pct']}, G1 {p['G1']['n_lose_to_C0_gt_2pct']}, "
                 f"M_bias {p['M_bias']['n_lose_to_C0_gt_2pct']}", fontsize=9.5, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    pr, g = load(PRED), load(GATE)
    chk = recompute(g, pr["frozen_rule"])
    items, tables = [], []

    def add_table(t: rf.Table, desc: str, srcs: list[str]) -> None:
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    add_table(t_predictions(pr), "Table of the pre-registered TEST predictions: per TEST dataset the number of "
              "TEST and DEV devices, stored DEV S1_lodo, frozen-rule predicted y and GMR, S1_lodo sign, stored "
              "DEV S3 and its sign, secondary S1_lodo on the TEST training set, DEV GMR (already seen)",
              [rel(PRED)])
    add_table(t_gate_datasets(g), "Table of the S3 gate G and the S1_lodo gate G1 per dataset (11 stored, 20 "
              "simulated): S3, S1_lodo, GMR(M_bias, C0_refit), choice, GMR vs C0_refit and vs M_bias, fold-level "
              "S3 gate diagnostic", [rel(GATE)])
    add_table(t_gate_pooled(g), "Table of pooled gate results (11 stored, 7 measured, 20 simulated, 31): mean "
              "ln GMR, worst dataset and number of datasets above 1.02 vs C0_refit and vs M_bias for G, G1, "
              "M_bias, C0_refit, the dataset oracle and the fold-level gates", [rel(GATE)])
    fig_gate(g, folder / "04_i8_2_gate_per_dataset.png")
    items.append(("04_i8_2_gate_per_dataset.png", "Dot plot of GMR(method, C0_refit) per dataset for M_bias on "
                  "every dataset (grey ring), the S3 gate G (blue) and the S1_lodo gate G1 (orange), with the "
                  "1.0 and 1.02 lines", [rel(GATE)]))

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i8_2.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "",
          f"Generated {now} by {gen} from stored files only. y = ln GMR(M_bias, C0_refit) over the held-out "
          "devices (strict paper RRMS after the held-out card refit); negative = gain.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))

    P = g["pooled"]
    lk = g["leakage"]["observed"]
    preds = pr["predictions"]
    pred_txt = "; ".join(f"{r['dataset']}: {r['S1_predicts']} (rule y {r['predicted_y_rule']:+.3f}), S3 "
                         f"{r['S3_predicts']}" for r in preds)

    def pl(grp: str, arm: str) -> str:
        a = P[grp][arm]
        return (f"exp(mean ln GMR vs C0_refit) {a['gmr_of_mean_vs_C0']:.3f}, worst {a['worst_vs_C0']['gmr']:.3f}, "
                f"> 1.02 on {a['n_lose_to_C0_gt_2pct']}/{P[grp]['n_datasets']}; vs M_bias exp(mean) "
                f"{math.exp(a['mean_ln_gmr_vs_M_bias']):.3f}, worst {worst(a['worst_vs_M_bias'])}, "
                f"> 1.02 on {a['n_lose_to_M_bias_gt_2pct']}/{P[grp]['n_datasets']}")

    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen} from stored files only (no NGSpice, no training). {chk['n_checked']} "
              "stored gate values recomputed and checked.", "",
              "## Contents", "",
              f"- Part A, pre-registration (`{rel(PRED)}`, code git {pr['code']['git_head_at_run'][:7]}, written "
              f"{pr['written']}, before any TEST curve was loaded): predicted sign of ln GMR(M_bias, C0_refit) on "
              f"the 7 TEST datasets from the stored DEV S1_lodo (frozen I7-1 rule) and the stored DEV S3. "
              f"{pred_txt}. S1_lodo and S3 agree on all 7. TEST data were not opened.",
              f"- Part B, retrospective (`{rel(GATE)}`; outcomes known before the gates were defined). G: M_bias "
              "on a dataset iff S3(D) < 0, else C0_refit; G1: the same with the frozen S1_lodo rule.",
              f"- 11 stored datasets: G {pl('stored_11', 'G')}. G1 {pl('stored_11', 'G1')}. M_bias always "
              f"{pl('stored_11', 'M_bias')}.",
              f"- 20 simulated datasets: G {pl('new_20', 'G')}. G1 {pl('new_20', 'G1')}. M_bias always "
              f"{pl('new_20', 'M_bias')}.",
              f"- All 31: G {pl('all_31', 'G')}. G1 {pl('all_31', 'G1')}. Dataset oracle (reference) exp(mean) "
              f"{P['all_31']['oracle_reference']['gmr_of_mean_vs_C0']:.3f}.",
              "- Leakage: S3(D) and S1_lodo(D) average fold statistics over all outer folds, so each scored "
              "device's own starting-card record enters the gate decision used for it (about 1/n_devices of the "
              "value); the held-out refit outcomes do not. Fold-level S3 gate (decision from the fold's own "
              "training set, which excludes the scored device): decisions differ from G in "
              f"{len(lk['stored_11']['datasets_with_any_fold_decision_different'])}/11 stored and "
              f"{len(lk['new_20']['datasets_with_any_fold_decision_different'])}/20 simulated datasets; this "
              "difference contains fold-level variability of S3_f as well as leakage. Datasets with |S3(D)| < "
              f"0.01: {', '.join(label(n) for n in lk['near_threshold']['datasets'])}; a flipped decision there "
              f"changes the dataset ln GMR by at most {lk['near_threshold']['max_abs_y']:.3f}. On TEST the gate "
              "uses DEV devices only.",
              "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))

    summary = [
        "I8-2 Part A (pre-registered before TEST, git " + pr["code"]["git_head_at_run"][:7] + "): predicted "
        "sign of ln GMR(M_bias, C0_refit) on TEST: sky130 77 K gain, IHP 233/300/343/398 K loss, TU Delft "
        "4.13/295 K gain; S1_lodo (frozen rule) and S3 agree on all 7.",
        f"I8-2 Part B (retrospective), all 31 datasets: G (S3 gate) exp(mean ln GMR vs C0_refit) "
        f"{P['all_31']['G']['gmr_of_mean_vs_C0']:.3f}, > 1.02 on {P['all_31']['G']['n_lose_to_C0_gt_2pct']}/31; "
        f"G1 {P['all_31']['G1']['gmr_of_mean_vs_C0']:.3f}, {P['all_31']['G1']['n_lose_to_C0_gt_2pct']}/31; "
        f"M_bias always {P['all_31']['M_bias']['gmr_of_mean_vs_C0']:.3f}, "
        f"{P['all_31']['M_bias']['n_lose_to_C0_gt_2pct']}/31.",
        f"vs M_bias: G worst {P['all_31']['G']['worst_vs_M_bias']['gmr']:.3f} "
        f"({label(P['all_31']['G']['worst_vs_M_bias']['dataset'])}), G1 worst "
        f"{P['all_31']['G1']['worst_vs_M_bias']['gmr']:.3f} "
        f"({label(P['all_31']['G1']['worst_vs_M_bias']['dataset'])}).",
    ]
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    print(f"checked {chk['n_checked']} values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
