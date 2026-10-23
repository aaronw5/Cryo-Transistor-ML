#!/usr/bin/env python3
"""Report batch for I19-1: dataset-level gate (one decision per dataset from
S3_D computed on all DEV devices of the dataset).

Reads ``out/research/explore/I19_1/{prereg.json, decision.json,
test_gate_decisions.json, s3/}`` and the I18-1 rows; recomputes every S3_D
from the per-record values, every gated_D GMR from the stored rows and every
simulated decision, and checks them against decision.json.  No NGSpice run,
no training.

  .venv/bin/python -I scripts/research/plot_explore_i19_1.py --date 2026-10-23
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
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

rf = pe.rf
plt = pe.plt

TOPIC = "i19_1"
I19 = EXP / "I19_1"
COMMITS = {"prereg": "45b4f37", "code": "4da313c"}
NAMES = {"sky130_77K": "sky130 77 K (all 18)", "nist_4K": "NIST sky130 4 K lvt (I8-1)",
         "nist_rt_card_4K": "NIST sky130 4 K, RT card (I9-1)", "ihp_300K": "IHP SG13G2 300 K",
         "ihp_233K": "IHP SG13G2 233 K", "ihp_343K": "IHP SG13G2 343 K", "ihp_398K": "IHP SG13G2 398 K",
         "tudelft_4K": "TU Delft 4.13 K", "tudelft_295K": "TU Delft 295.0 K"}


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext"):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": None}


def check(a: float, b: float, what: str, tol: float = 1e-12) -> None:
    if not abs(a - b) <= tol:
        raise AssertionError(f"{what}: {a} vs {b}")


def recompute(D: dict) -> int:
    """S3_D from per-record values, gated_D rows and GMRs from stored rows,
    simulated decisions; returns the number of checked values."""
    import i19_1_dataset_gate as drv
    from cryoml.explore import i19_1_dataset_gate as m
    from cryoml.explore.i3_2_bias import paired_ratio
    n = 0
    ds = drv.g18.datasets(fallback_tud=False)
    for name, x in D["part_R"]["datasets"].items():
        f = load(drv.part_file(drv.PRIMARY_PART[name], False))
        s = m.s3_dataset(list(f["per_record"].values()))
        check(s, f["S3_D"], f"{name} S3_D file")
        check(s, x["S3_D"], f"{name} S3_D")
        d = ds[name]
        rows = m.gated_rows(d["corr"], d["c0"], m.apply_gate(s))
        g = paired_ratio(rows, d["c0"], clusters=d["clusters"])
        for k in ("ratio", "ci_low", "ci_high"):
            check(g[k], x["gmr_gated_D_vs_C0_refit"][k], f"{name} {k}")
        n += 5
    r = m.part_r_rule({k: v["gmr_gated_D_vs_C0_refit"]["ratio"] for k, v in D["part_R"]["datasets"].items()},
                      {k: v["gmr_correction_vs_C0_refit"]["ratio"] for k, v in D["part_R"]["datasets"].items()})
    check(r["pooled_gated_D"], D["part_R"]["rule"]["pooled_gated_D"], "pooled")
    n += 1
    for k, v in D["part_C"]["datasets"].items():
        f = load(drv.part_file(f"sim:{k}", False))
        check(m.s3_dataset(list(f["per_record"].values())), v["S3_D"], f"sim {k}")
        if m.sign_correct(v["S3_D"], v["y"]) != v["correct"]:
            raise AssertionError(k)
        n += 2
    T = load(I19 / "test_gate_decisions.json")
    for row in T["decisions"]:
        check(row["S3_D"], D["part_R"]["datasets"][row["name"]]["S3_D"], f"TEST {row['name']}")
        n += 1
    return n


def t_rule(D: dict, P: dict) -> rf.Table:
    v, r, c = D["validity"], D["part_R"]["rule"], D["part_C"]["rule"]
    rows = [{"c": "validity", "v": "valid" if v["valid"] else "invalid: " + "; ".join(v["fails"])},
            {"c": "V2 S3 code reproduces stored fold values (3 folds)", "v": f"max |diff| {max(D['v2_values']):.1e}"},
            {"c": "R1: GMR(gated_D, C0_refit) <= 1 on each of 9 DEV datasets",
             "v": "holds" if r["R1_never_worse"] else "fails on " + ", ".join(r["R1_fail_datasets"])},
            {"c": "R2: pooled gated_D below pooled correction always",
             "v": f"{'holds' if r['R2_pooled_below_correction'] else 'fails'} "
                  f"({r['pooled_gated_D']:.3f} vs {r['pooled_correction_always']:.3f})"},
            {"c": "Part R outcome (retrospective)", "v": r["outcome"]},
            {"c": "C: simulated sign accuracy >= 19/20",
             "v": f"{'holds' if c['holds'] else 'fails'} ({c['correct']}/{c['n']}; constant apply "
                  f"{c['constant_apply_correct']}/{c['n']}; wrong: {', '.join(c['wrong']) or 'none'})"},
            {"c": "overall outcome (pre-registered)", "v": D["outcome"]},
            {"c": "threshold (fixed before any S3_D)", "v": f"{P['decision']['threshold']} (strict S3_D < 0)"}]
    return rf.Table("01_table_i19_1_rule", "I19-1 pre-registered rules, validity and outcome",
                    [col("c", "item", kind="text"), col("v", "value", kind="text")], rows,
                    [f"Pre-registration commit {COMMITS['prereg']}; code {COMMITS['code']}. Part R is retrospective: "
                     "DEV outcomes and the I8-2 gate G values were known."])


def t_part_r(D: dict) -> rf.Table:
    rows = []
    for name, x in D["part_R"]["datasets"].items():
        rows.append({"d": NAMES[name], "a": x["arm"], "s": f"{x['S3_D']:+.4f}", "dec": x["decision"],
                     "sens": x["decision_sensitivity_0.02455"],
                     "g8": f"{x['I8_2_gate_G']['S3_fold_mean']:+.3f} ({x['I8_2_gate_G']['decision']})",
                     "g": ci(x["gmr_gated_D_vs_C0_refit"]),
                     "pf": f"{x['per_fold_gate_I18_1']['ratio']:.3f} ({x['per_fold_gate_I18_1']['folds_apply']})",
                     "gc": ci(x["gmr_correction_vs_C0_refit"]),
                     "ok": "yes" if x["decision_matches_realised_sign"] else "no"})
    return rf.Table("02_table_i19_1_part_r", "I19-1 Part R (retrospective): dataset-level decision and GMR per DEV "
                    "dataset",
                    [col("d", "dataset", kind="text"), col("a", "gated arm", kind="text"), col("s", "S3_D"),
                     col("dec", "decision"), col("sens", "at +0.02455"), col("g8", "I8-2 G: fold-mean S3"),
                     col("g", "GMR(gated_D, C0_refit) [95% CI]"), col("pf", "per-fold gate GMR (folds)"),
                     col("gc", "GMR(correction always, C0_refit)"), col("ok", "decision = realised sign")], rows,
                    ["S3_D: mean over all DEV records of ln(RRMS(start x exp(g)) / RRMS(start)), g trained with the "
                     "gated arm's recipe (seed 0, 500 steps) on the LOGO fold that holds out the record's group "
                     "(IHP: same class). Decision: apply iff S3_D < 0. Per-fold gate: I18-1."])


def t_pooled(D: dict) -> rf.Table:
    p = D["part_R"]["pooled"]
    w = p["without_tudelft"]
    rows = [{"m": "gated_D (I19-1)", "a": f"{p['gated_D']:.3f}", "b": f"{w['gated_D']:.3f}"},
            {"m": "per-fold gate (I18-1)", "a": f"{p['per_fold_gate_I18_1']:.3f}", "b": f"{w['per_fold_gate_I18_1']:.3f}"},
            {"m": "correction always", "a": f"{p['correction_always']:.3f}", "b": f"{w['correction_always']:.3f}"},
            {"m": "I8-2 gate G (fold-mean S3 < 0)", "a": f"{p['I8_2_gate_G']:.3f}", "b": "-"},
            {"m": "post hoc dataset oracle (outcome known)", "a": f"{p['oracle_post_hoc']:.3f}", "b": "-"},
            {"m": "decision = realised sign of ln GMR(correction, C0_refit)",
             "a": D["part_R"]["decision_matches_realised_sign"], "b": "-"},
            {"m": "v1 secondary: gated_D / v1 always",
             "a": f"{D['part_R_secondary_v1']['pooled_gated_D']:.3f} / "
                  f"{D['part_R_secondary_v1']['pooled_correction_always']:.3f}", "b": "-"}]
    return rf.Table("03_table_i19_1_pooled", "I19-1 Part R pooled exp(mean ln GMR vs C0_refit), equal weight per "
                    "dataset",
                    [col("m", "method", kind="text"), col("a", "9 datasets"), col("b", "7 (without TU Delft)")], rows,
                    ["The 4 IHP temperatures share DUTs and the 2 TU Delft temperatures share devices."])


def t_secondary(D: dict) -> rf.Table:
    rows = []
    for name, x in D["part_R_secondary_v1"]["datasets"].items():
        rows.append({"d": NAMES[name], "s": f"{x['S3_D']:+.4f}", "dec": x["decision"],
                     "g": ci(x["gmr_gated_D_vs_C0_refit"]), "gc": ci(x["gmr_correction_vs_C0_refit"]),
                     "ok": "yes" if x["decision_matches_realised_sign"] else "no"})
    return rf.Table("04_table_i19_1_secondary_v1", "I19-1 secondary: dataset-level gate with M_bias v1 and the v1 "
                    "recipe (reported, no rule)",
                    [col("d", "dataset", kind="text"), col("s", "S3_D (v1)"), col("dec", "decision"),
                     col("g", "GMR(gated_D, C0_refit)"), col("gc", "GMR(v1 always, C0_refit)"),
                     col("ok", "decision = realised sign")], rows,
                    ["sky130: state-C records and rows (I13-1); TU Delft: I6-1 state-L records and rows; other "
                     "datasets: the primary v1 values."])


def t_test(T: dict) -> rf.Table:
    rows = []
    for r in T["decisions"]:
        v1, i8 = r.get("secondary_v1"), r.get("I8_2")
        rows.append({"d": r["test_split"], "n": str(r["n_test"]), "nd": str(r["n_dev_records_used"]), "a": r["arm"],
                     "s": f"{r['S3_D']:+.4f}", "dec": r["decision"], "sens": r["decision_sensitivity_0.02455"],
                     "v1": f"{v1['S3_D']:+.4f} ({v1['decision']})" if v1 else "-",
                     "i8": f"{i8['S3_predicts']} / {i8['S1_predicts']}" if i8 else
                     (f"DEV P_c {r['dev_P_c']['predicted_sign']}" if "dev_P_c" in r else "-")})
    dis = T["disagreements_with_I8_2"]
    note = ("Disagreements with I8-2 (I8-2 predicts the sign for M_bias v1): "
            + ("; ".join(f"{d['name']} I19-1 {d['I19_1']} {d['I19_1_decision']} vs I8-2 {d['I8_2']} "
                         f"{d['I8_2_prediction']}" for d in dis) if dis else "none") + ".")
    return rf.Table("05_table_i19_1_test_decisions", "I19-1 Part P: committed TEST gate decisions (TEST not "
                    "evaluated)",
                    [col("d", "TEST split", kind="text"), col("n", "n TEST"), col("nd", "DEV records"),
                     col("a", "arm", kind="text"), col("s", "S3_D (DEV)"), col("dec", "decision"),
                     col("sens", "at +0.02455"), col("v1", "v1 S3_D (decision)"),
                     col("i8", "I8-2 S3 / S1 prediction", kind="text")], rows,
                    [f"Committed in its own commit before the decide step; written at git HEAD "
                     f"{T['git_head_at_run'][:7]}. {note}"])


def t_sim(D: dict) -> rf.Table:
    rows = []
    for k, v in D["part_C"]["datasets"].items():
        rows.append({"d": k.replace("|", " r").replace(" r0 r", " r0 lambda ").replace(" r1 r", " r1 lambda "),
                     "s": f"{v['S3_D']:+.4f}", "f": f"{v['S3_fold_mean_I7_1']:+.4f}", "y": f"{v['y']:+.4f}",
                     "dec": v["decision"], "ok": "yes" if v["correct"] else "NO"})
    s = D["part_C"]["summary"]
    return rf.Table("06_table_i19_1_part_c", "I19-1 Part C: 20 simulated I7-1 datasets, S3_D (v1) vs realised y_D",
                    [col("d", "dataset (PDK, replicate, lambda)", kind="text"), col("s", "S3_D"),
                     col("f", "I7-1 fold-mean S3"), col("y", "y_D = ln GMR(M_bias, C0_refit)"), col("dec", "decision"),
                     col("ok", "correct")], rows,
                    [f"Spearman(S3_D, y_D) {s['spearman_S3_D_y']:+.3f}; OLS y = {s['ols_y_on_S3_D']['a']:+.4f} + "
                     f"{s['ols_y_on_S3_D']['b']:.3f} S3_D (R^2 {s['ols_y_on_S3_D']['r2']:.3f}); pooled GMR gated_D "
                     f"{s['pooled_gated_D']:.3f}, M_bias always {s['pooled_M_bias_always']:.3f}, oracle "
                     f"{s['pooled_oracle']:.3f}; sensitivity threshold +0.02455: {s['sensitivity_correct']}/20 correct."])


def fig_forest(D: dict, path: Path) -> list[str]:
    R = D["part_R"]["datasets"]
    names = list(R)
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    y = np.arange(len(names))[::-1]
    series = (("gmr_correction_vs_C0_refit", ORANGE, "correction always", 0.22),
              ("per_fold_gate_I18_1", VIOLET, "per-fold gate (I18-1, S3_f < 0)", 0.0),
              ("gmr_gated_D_vs_C0_refit", AQUA, "dataset-level gate (I19-1, S3_D < 0)", -0.22))
    for key, colr, lab, dy in series:
        g = [R[n][key] for n in names]
        ax.errorbar([z["ratio"] for z in g], y + dy, xerr=[[z["ratio"] - z["ci_low"] for z in g],
                                                            [z["ci_high"] - z["ratio"] for z in g]],
                    fmt="o", color=colr, capsize=3, label=lab, ms=5)
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{NAMES[n]}\nS3_D {R[n]['S3_D']:+.3f}: {R[n]['decision']}" for n in names], fontsize=8)
    ax.set_xlabel("GMR vs C0_refit [95% CI] (log scale; below 1: lower held-out RRMS)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    p = D["part_R"]["pooled"]
    ax.set_title(f"I19-1 Part R (retrospective), 9 DEV datasets; pooled: dataset gate {p['gated_D']:.3f}, per-fold "
                 f"gate {p['per_fold_gate_I18_1']:.3f}, correction always {p['correction_always']:.3f}",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I19 / "decision.json")]


def fig_scatter(D: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    style(ax)
    R = D["part_R"]["datasets"]
    S = D["part_R_secondary_v1"]["datasets"]
    for n, x in R.items():
        yv = x["realised_ln_gmr_correction"]
        ax.scatter([x["S3_D"]], [yv], color=AQUA if x["apply"] else RED, s=36, zorder=3)
        ax.annotate(NAMES[n], (x["S3_D"], yv), fontsize=7, color=INK, xytext=(4, 3), textcoords="offset points")
    for n, x in S.items():
        if x["S3_D"] != R.get(n, {}).get("S3_D"):
            ax.scatter([x["S3_D"]], [x["realised_ln_gmr_correction"]], facecolors="none",
                       edgecolors=AQUA if x["apply"] else RED, s=36, zorder=3)
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=1.0, label="threshold 0")
    ax.axvline(0.02455, color=GREY, lw=0.8, ls="--", label="sensitivity +0.02455 (no rule)")
    ax.set_xlabel("S3_D (all DEV devices of the dataset)")
    ax.set_ylabel("realised ln GMR(correction, C0_refit), DEV LOGO")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("Measured DEV datasets (filled: primary arm; hollow: v1 secondary)\naqua: apply, red: C0_refit",
                 fontsize=9, color=INK, loc="left")
    ax = axes[1]
    style(ax)
    C = D["part_C"]["datasets"]
    for k, v in C.items():
        ax.scatter([v["S3_D"]], [v["y"]], color=(BLUE if k.startswith("gf180") else ORANGE),
                   marker="o" if v["correct"] else "x", s=30, zorder=3)
    ax.scatter([], [], color=BLUE, label="GF180")
    ax.scatter([], [], color=ORANGE, label="IHP")
    ax.scatter([], [], color=GREY, marker="x", label="decision wrong")
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=1.0)
    ax.axvline(0.02455, color=GREY, lw=0.8, ls="--")
    s = D["part_C"]["summary"]
    c = D["part_C"]["rule"]
    ax.set_xlabel("S3_D (v1 recipe, all 16 devices)")
    ax.set_ylabel("y_D = ln GMR(M_bias, C0_refit), LOGO")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title(f"20 simulated I7-1 datasets: sign correct {c['correct']}/20 (rule >= 19); Spearman "
                 f"{s['spearman_S3_D_y']:+.3f}", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I19 / "decision.json"), "I19_1/s3/*.json", "I7_1/decision.json"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    D, P, T = load(I19 / "decision.json"), load(I19 / "prereg.json"), load(I19 / "test_gate_decisions.json")
    n_checks = recompute(D)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    dj = rel(I19 / "decision.json")
    add_table(t_rule(D, P), "Rules R1 / R2 / C, validity, outcome", [dj, rel(I19 / "prereg.json")])
    add_table(t_part_r(D), "Part R per dataset: S3_D, decision, GMR vs per-fold gate and correction always", [dj])
    add_table(t_pooled(D), "Part R pooled GMR per method", [dj])
    add_table(t_secondary(D), "Secondary v1 dataset-level gate", [dj])
    add_table(t_test(T), "Part P committed TEST gate decisions and I8-2 disagreements",
              [rel(I19 / "test_gate_decisions.json")])
    add_table(t_sim(D), "Part C per simulated dataset", [dj])
    for name, fn, desc in (("07_i19_1_forest.png", fig_forest, "GMR vs C0_refit per DEV dataset: dataset-level gate, "
                            "per-fold gate, correction always"),
                           ("08_i19_1_s3d_scatter.png", fig_scatter, "S3_D vs realised gain with the threshold, "
                            "measured and simulated datasets")):
        items.append((name, desc, fn(D, folder / name)))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_explore_i19_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    r, c = D["part_R"]["rule"], D["part_C"]["rule"]
    summ = [f"I19-1 dataset-level gate: {D['outcome']}. Part R (retrospective): {r['outcome']}; pooled GMR gated_D "
            f"{r['pooled_gated_D']:.3f} vs correction always {r['pooled_correction_always']:.3f} and per-fold gate "
            f"{D['part_R']['pooled']['per_fold_gate_I18_1']:.3f}. Part C: {c['correct']}/20 sign correct.",
            "Part P TEST decisions: " + ", ".join(f"{x['name']} {x['decision']}" for x in T["decisions"]) + "."]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. {n_checks} recomputed values checked against decision.json and the S3_D "
              "files. No NGSpice runs, no training.", "", "## Contents", "",
              *[f"- {x}" for x in summ], "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summ)
    print("\n".join(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
