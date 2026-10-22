#!/usr/bin/env python3
"""Report batch for I18-1: gated correction ("selected g v2 applied only if
the training-only S3 gate predicts a gain, else C0_refit") on 9 measured DEV
datasets.

Reads ``out/research/explore/I18_1/{prereg.json, decision.json, s3/*.json}``
and the stored rows it names; recomputes every gated row and GMR from the
per-device rows and the per-fold S3 values and checks them against
decision.json.  NGSpice (ngspice-41, state C): the I-V figures re-simulate
the stock card and the C0_refit / selected held-out cards of TU Delft devices
(I18-2), one device per temperature from a fold where the gate applied the
correction and one where it chose C0_refit, when both exist.

  .venv/bin/python -I scripts/research/plot_explore_i18_1.py --date 2026-10-22
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

TOPIC = "i18_1"
I18 = EXP / "I18_1"
COMMITS = {"prereg": "b384a42", "code": "8fc4f32"}
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
    """Recompute gated rows and GMR(gated, C0_refit) from stored rows; returns checks."""
    import i18_1_gate as drv
    from cryoml.explore import i18_1_gate as m
    from cryoml.explore.i3_2_bias import paired_ratio
    ds = drv.datasets(fallback_tud=D["tudelft_source"] != "I18-2")
    n = 0
    for name, d in ds.items():
        rows, dec = m.gated_rows(d["corr"], d["c0"], d["fold_of"], d["s3"])
        g = paired_ratio(rows, d["c0"], clusters=d["clusters"])
        x = D["datasets"][name]
        check(g["ratio"], x["gmr_gated_vs_C0_refit"]["ratio"], f"{name} GMR")
        check(g["ci_low"], x["gmr_gated_vs_C0_refit"]["ci_low"], f"{name} CI low")
        check(g["ci_high"], x["gmr_gated_vs_C0_refit"]["ci_high"], f"{name} CI high")
        if sum(dec.values()) != x["folds_apply"]:
            raise AssertionError(name)
        for t, v in rows.items():
            check(v, x["gated_rows"][t], f"{name} row {t}", 0.0)
        n += 3 + len(rows)
    return n


def t_rule(D: dict, P: dict) -> rf.Table:
    v, r = D["validity"], D["rule"]
    rows = [{"c": "validity", "v": "valid" if v["valid"] else "invalid: " + "; ".join(v["fails"])},
            {"c": "C1 never worse: CI lower of GMR(gated, C0_refit) <= 1 on all 9 datasets",
             "v": "holds" if r["C1_never_worse"] else "fails on " + ", ".join(r["C1_fail_datasets"])},
            {"c": "C2 sky130 gain: CI upper < 1 on sky130 77 K",
             "v": "holds" if r["C2_sky130_gain"] else "fails"},
            {"c": "outcome (pre-registered rule)", "v": D["outcome"]},
            {"c": "predicted outcome", "v": P["predictions"]["outcome"]},
            {"c": "TU Delft rows", "v": D["tudelft_source"]},
            {"c": "gate", "v": P["gate"]["threshold"]},
            {"c": "sign convention", "v": P["gate"]["sign_convention"]},
            {"c": "pooled exp(mean ln GMR) over 9 datasets (reported): gated / correction always",
             "v": f"{D['pooled_exp_mean_ln_gmr_gated']:.3f} / {D['pooled_exp_mean_ln_gmr_correction']:.3f}"}]
    return rf.Table("01_table_i18_1_rule", "I18-1 pre-registered rule, validity and outcome",
                    [col("c", "item", kind="text"), col("v", "value", kind="text")], rows,
                    [f"Pre-registration commit {COMMITS['prereg']}; code {COMMITS['code']}."])


def t_datasets(D: dict, P: dict) -> rf.Table:
    rows = []
    pr = P["predictions"]
    for name, x in D["datasets"].items():
        pk = name if name in pr else ("ihp_233K_343K_398K" if name in ("ihp_233K", "ihp_343K", "ihp_398K") else None)
        g = x["gmr_gated_vs_C0_refit"]
        rows.append({"d": NAMES[name], "a": x["arm"], "u": x["unit"],
                     "f": f"{x['folds_apply']}/{x['n_folds']} ({x['devices_apply']}/{x['n_rows']} dev)",
                     "p": pr[pk]["decisions"] if pk else "-",
                     "g": ci(g), "lo": "yes" if g["ci_low"] <= 1 else "NO",
                     "gc": ci(x["gmr_correction_vs_C0_refit"]), "go": ci(x["gmr_gated_vs_C0_refit_other_unit"])})
    return rf.Table("02_table_i18_1_datasets", "I18-1 per dataset: gate decisions and GMR(gated, C0_refit)",
                    [col("d", "dataset", kind="text"), col("a", "gated arm", kind="text"), col("u", "unit"),
                     col("f", "folds applied"), col("p", "predicted", kind="text"),
                     col("g", "GMR(gated, C0_refit) [95% CI]"), col("lo", "CI lower <= 1"),
                     col("gc", "GMR(correction always, C0_refit)"), col("go", "gated, other unit CI")], rows,
                    ["Gate: correction row for a fold's held-out devices iff S3_f < 0 (training devices only); "
                     "otherwise the C0_refit row. Other unit: device bootstrap for TU Delft, per-fold cluster "
                     "(= device) elsewhere."])


def t_secondary(D: dict) -> rf.Table:
    rows = []
    for name, x in D["secondary_v1_gate"].items():
        rows.append({"d": NAMES.get(name, name), "f": f"{x['folds_apply']}/{x['n_folds']}",
                     "g": ci(x["gmr_gated_vs_C0_refit"]), "gc": ci(x["gmr_correction_vs_C0_refit"]),
                     "dg": f"{x['dataset_gate']['S3_D']:+.3f} -> {x['dataset_gate']['gmr_vs_C0_refit']:.3f}",
                     "sp": f"{x['spearman_s3_y']:+.2f}" if np.isfinite(x["spearman_s3_y"]) else "-"})
    return rf.Table("03_table_i18_1_secondary_v1", "I18-1 secondary: the same gate with M_bias v1 and the v1-recipe S3 "
                    "(reported, no rule)",
                    [col("d", "dataset", kind="text"), col("f", "folds applied"),
                     col("g", "GMR(v1-gated, C0_refit)"), col("gc", "GMR(v1 always, C0_refit)"),
                     col("dg", "dataset-level gate S3_D -> GMR"), col("sp", "Spearman(S3_f, y_f)")], rows,
                    ["sky130: state-C rows (I13-1), S3_v1 recomputed on 18 folds; TU Delft: I6-1 stored rows (state "
                     "L) and stored I7-1 S3; other datasets: stored rows and stored S3."])


def t_folds(D: dict) -> rf.Table:
    rows = []
    for name in ("sky130_77K", "ihp_300K", "tudelft_4K", "tudelft_295K"):
        x = D["datasets"][name]
        s = np.array(list(x["s3"].values()))
        rows.append({"d": NAMES[name], "n": str(len(s)), "neg": str(int(np.sum(s < 0))),
                     "r": f"[{s.min():+.3f}, {s.max():+.3f}]", "dg": f"{x['dataset_gate']['S3_D']:+.3f}",
                     "sp": f"{x['spearman_s3_y']:+.2f}" if np.isfinite(x["spearman_s3_y"]) else "-",
                     "or": f"{x['oracle_post_hoc']:.3f}"})
    return rf.Table("04_table_i18_1_s3_v2", "I18-1 S3_v2 per fold (g v2 recipe at lambda*_f, seed 0, 500 steps)",
                    [col("d", "dataset", kind="text"), col("n", "folds"), col("neg", "S3_f < 0"),
                     col("r", "range"), col("dg", "S3_D (weighted mean)"), col("sp", "Spearman(S3_f, y_f)"),
                     col("or", "post hoc oracle GMR")], rows,
                    ["y_f: mean held-out ln(RRMS correction / RRMS C0_refit) of fold f. Oracle: min(1, GMR(correction, "
                     "C0_refit)) chosen with the outcome known (reference only)."])


def fig_forest(D: dict, path: Path) -> list[str]:
    names = list(D["datasets"])
    fig, ax = plt.subplots(figsize=(11, 6), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    y = np.arange(len(names))[::-1]
    for k, (key, colr, lab, dy) in enumerate((("gmr_correction_vs_C0_refit", ORANGE, "correction always", 0.15),
                                              ("gmr_gated_vs_C0_refit", AQUA, "gated (S3_f < 0)", -0.15))):
        g = [D["datasets"][n][key] for n in names]
        ax.errorbar([z["ratio"] for z in g], y + dy, xerr=[[z["ratio"] - z["ci_low"] for z in g],
                                                            [z["ci_high"] - z["ratio"] for z in g]],
                    fmt="o", color=colr, capsize=3, label=lab)
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{NAMES[n]}\n{D['datasets'][n]['folds_apply']}/{D['datasets'][n]['n_folds']} folds applied"
                        for n in names], fontsize=8)
    ax.set_xlabel("GMR vs C0_refit [95% CI] (log scale; below 1: lower held-out RRMS)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_title(f"I18-1 gated correction per dataset; outcome: {D['outcome']}", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I18 / "decision.json")]


def fig_s3(D: dict, path: Path) -> list[str]:
    names = list(D["datasets"])
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    for k, n in enumerate(names):
        s = np.array(list(D["datasets"][n]["s3"].values()))
        ax.scatter(np.full(len(s), k) + np.random.default_rng(k).uniform(-0.15, 0.15, len(s)), s,
                   color=np.where(s < 0, AQUA, RED), s=12)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([NAMES[n] for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("S3_f (mean ln RRMS ratio, inner held-out training records)")
    ax.set_title("I18-1 gate statistic per outer fold (aqua: S3_f < 0, correction applied; red: C0_refit)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I18 / "decision.json"), "I18_1/s3/*.json"]


def fig_s3_scatter(D: dict, path: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    style(ax)
    for name, colr in (("sky130_77K", BLUE), ("ihp_300K", ORANGE)):
        c = D["s3_v2_vs_v1"].get(name, {})
        if c:
            ax.scatter([v[1] for v in c.values()], [v[0] for v in c.values()], color=colr, s=16, label=NAMES[name])
    lim = ax.get_xlim()
    ax.plot(lim, lim, color=GREY, lw=0.8, ls="--")
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("S3_v1 (v1 recipe)")
    ax.set_ylabel("S3_v2 (g v2 recipe at lambda*_f)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("S3 per fold: g v2 recipe vs v1 recipe", fontsize=9.5, color=INK, loc="left")
    ax = axes[1]
    style(ax)
    for name, colr in (("sky130_77K", BLUE), ("ihp_300K", ORANGE), ("tudelft_4K", AQUA), ("tudelft_295K", VIOLET)):
        x = D["datasets"][name]
        f = [k for k in x["s3"] if k in x["fold_y"]]
        ax.scatter([x["s3"][k] for k in f], [x["fold_y"][k] for k in f], color=colr, s=16,
                   label=f"{NAMES[name]} (Spearman {x['spearman_s3_y']:+.2f})")
    ax.axhline(0, color=INK, lw=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("S3_f (g v2)")
    ax.set_ylabel("y_f = mean held-out ln(RRMS selected / RRMS C0_refit)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Gate statistic vs held-out outcome per fold", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I18 / "decision.json")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    folder.mkdir(parents=True, exist_ok=True)
    D, P = load(I18 / "decision.json"), load(I18 / "prereg.json")
    n_checks = recompute(D)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    dj = rel(I18 / "decision.json")
    add_table(t_rule(D, P), "Rule C1 / C2, validity, outcome", [dj, rel(I18 / "prereg.json")])
    add_table(t_datasets(D, P), "Per dataset: arm, unit, gate decisions vs prediction, GMR(gated, C0_refit)", [dj])
    add_table(t_secondary(D), "Secondary v1 gate on every dataset with v1 rows", [dj])
    add_table(t_folds(D), "S3_v2 per fold summary, Spearman with the fold outcome, oracle reference", [dj])
    for name, fn, desc in (("05_i18_1_forest.png", fig_forest, "GMR vs C0_refit per dataset: gated vs correction "
                            "always"),
                           ("06_i18_1_s3_per_fold.png", fig_s3, "S3_f per outer fold and dataset with the threshold"),
                           ("07_i18_1_s3_scatter.png", fig_s3_scatter, "S3_v2 vs S3_v1, and S3_f vs fold outcome")):
        items.append((name, desc, fn(D, folder / name)))
    # I-V: TU Delft (I18-2 rows), one device per temperature from a gated-on and a gated-off fold
    if D["tudelft_source"] == "I18-2":
        import plot_explore_i18_2 as p2
        dec2 = load(EXP / "I18_2/decision.json")
        k = 8
        for t in ("4", "295"):
            x = D["datasets"][f"tudelft_{t}K"]
            picks = []
            for on in (True, False):
                folds = [f for f, v in x["decisions"].items() if v == on]
                if folds:
                    tags = sorted(u for u in x["gated_rows"] if p2.group_of(u) in folds)
                    picks.append((tags[0], on))
            res = p2.resim(EXP / "I18_2", dec2["per_temperature"][t], int(t), [u for u, _ in picks])
            for tag, on in picks:
                name = f"{k:02d}_i18_1_iv_tudelft_{t}K_{p2.short(tag)}.png"
                lab = ("gated: selected g v2 applied (S3_f "
                       f"{x['s3'][p2.group_of(tag)]:+.3f} < 0)" if on else
                       f"gated: C0_refit kept (S3_f {x['s3'][p2.group_of(tag)]:+.3f} >= 0)")
                p2.fig_iv(res[tag], tag, p2.TLAB[t], folder / name, extra_title=f"\nI18-1 {lab}.",
                          labels={"M_bias_v1": ""})
                items.append((name, f"I-V of {tag} ({p2.TLAB[t]}), {lab}: measured points, NGSpice stock card, "
                                    "re-simulated C0_refit and selected held-out cards",
                              ["I18_2/base_*/heldout_cards.json", "I18_2/lodo_*/heldout_cards.json"]))
                k += 1
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_explore_i18_1.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    sky = D["datasets"]["sky130_77K"]["gmr_gated_vs_C0_refit"]
    summ = [f"I18-1 gated correction over 9 measured DEV datasets: {D['outcome']} (C1 "
            f"{'holds' if D['rule']['C1_never_worse'] else 'fails'}, C2 {'holds' if D['rule']['C2_sky130_gain'] else 'fails'}).",
            f"sky130 GMR(gated, C0_refit) {ci(sky)}; folds applied: "
            + ", ".join(f"{NAMES[n]} {x['folds_apply']}/{x['n_folds']}" for n, x in D["datasets"].items()) + "."]
    readme = [f"# Report batch {args.date} · {TOPIC}", "",
              f"Generated {now} by {gen}. {n_checks} recomputed values checked against decision.json. NGSpice runs: "
              "re-simulation of TU Delft I-V devices only.", "", "## Contents", "",
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
