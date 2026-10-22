#!/usr/bin/env python3
"""Report batch for I17-2: second-PDK replication of the selected g v2 recipe
on measured IHP SG13G2 MOSFETs (PSP103, I5-1 data), 300 K, 18 DEV DUTs,
leave-one-DUT-out within class; accuracy only, Part B not run.

Reads stored files: ``out/research/explore/I17_2/{prereg.json, decision.json,
pens/pens.json, base/*, lodo_sym_e1_l*/*, inner/inner.json}`` and the I5-1
split, dataset and stored rows.

NGSpice (ngspice-46 + psp103.osdi, I5-1 setup): the I-V figures re-simulate
the stock PDK card, the C0_refit held-out card, the M_bias v1 held-out card
(x exp(fold g) in Python) and the selected held-out card (x exp(fold g));
every RRMS is checked against the stored rows.

  .venv/bin/python -I scripts/research/plot_explore_i17_2.py --date 2026-10-22
"""

from __future__ import annotations

import argparse
import datetime as dt
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

TOPIC = "i17_2"
I17 = EXP / "I17_2"
I5 = EXP / "I5_1"
F = {"prereg": I17 / "prereg.json", "decision": I17 / "decision.json", "pens": I17 / "pens/pens.json",
     "base": I17 / "base/results.json", "inner": I17 / "inner/inner.json", "split": I5 / "split.json",
     "i5": I5 / "lodo_300K/results.json"}
LAMS = (0.0, 1.0, 2.0, 5.0, 10.0)
COMMITS = {"prereg": "c071472", "code": "bd660cb"}
ARMS = ("C0_refit", "M_bias_v1", "sym_e1_l0", "sym_e1_l1", "sym_e1_l2", "sym_e1_l5", "sym_e1_l10", "selected")
COLOR = {"C0_refit": BLUE, "M_bias_v1": ORANGE, "selected": AQUA, "sym_e1_l0": "#9aa5b1", "sym_e1_l1": "#7c8a99",
         "sym_e1_l2": "#5f6f80", "sym_e1_l5": "#475566", "sym_e1_l10": VIOLET}


def ci(c: dict) -> str:
    return f"{c['ratio']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def col(key, head, md=None, kind="rtext"):
    return {"key": key, "head": head, "md": md or head, "kind": kind, "group": None}


def dev_key(t: str):
    return (t.split("_")[0] != "lv", t.split("_")[1], t)


def check(a: float, b: float, what: str, tol: float = 1e-9) -> None:
    if not abs(a - b) <= tol:
        raise AssertionError(f"{what}: {a} vs {b}")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def t_choice(d: dict) -> rf.Table:
    P = d["prereg"]
    rows = [
        {"c": "dataset chosen", "v": P["dataset"]["chosen"]},
        {"c": "reason", "v": P["dataset"]["why"]},
        {"c": "candidates", "v": "; ".join(f"{k}: {v}" for k, v in P["dataset"]["candidates"].items())},
        {"c": "simulator", "v": P["inputs"]["simulator"]},
        {"c": "BSIM4 clamp", "v": P["inputs"]["bsim4_clamp"]},
        {"c": "folds", "v": P["folds"]},
        {"c": "penalty grid", "v": P["penalty_grid"]["axes"] + f"; points with forward current > "
                                   f"{P['penalty_grid']['floor_A']:g} A"},
        {"c": "Part B", "v": "not run"},
    ]
    rows += [{"c": f"Part B reason {i + 1}", "v": r} for i, r in enumerate(P["part_B"]["reasons"])]
    return rf.Table("01_table_i17_2_setup", "I17-2 dataset choice, setup and why Part B was not run (pre-registered)",
                    [col("c", "item", kind="text"), col("v", "value", kind="text")], rows,
                    [f"Pre-registration out/research/explore/I17_2/prereg.md, commit {COMMITS['prereg']}."])


def t_validity(d: dict) -> rf.Table:
    D, B = d["decision"], d["base"]
    v = D["validity"]
    n_rows = {a: len(D["rows"][a]) for a in ARMS}
    rows = [
        {"c": "V1 C0_refit rerun vs I5-1 rows (18), max |d|", "t": "<= 1e-9", "v": f"{B['v1']['c0_max']:.1e}"},
        {"c": "V1 M_bias v1 rerun vs I5-1 rows (18), max |d|", "t": "<= 1e-9", "v": f"{B['v1']['mb_max']:.1e}"},
        {"c": "V2 unit tests", "t": "all pass", "v": str(B["unit_tests_passed"])},
        {"c": "V3 rows per arm", "t": "18", "v": ", ".join(f"{a} {n}" for a, n in n_rows.items())},
        {"c": "V3 inner selection complete", "t": "18 x 3 x 5", "v": str(d["inner"]["complete"])
         + f" ({d['inner']['timing']['n_train']} trainings, {d['inner']['timing']['n_refit']} inner refits)"},
        {"c": "V4 penalty grids 75 x 75 finite, forward corner current > 0", "t": "18", "v": str(d["pens"]["v4_ok"])},
    ]
    return rf.Table("02_table_i17_2_validity", "I17-2 validity", [col("c", "check", kind="text"),
                    col("t", "criterion"), col("v", "value", kind="text")], rows,
                    ["Result: " + ("valid." if v["valid"] else "invalid: " + "; ".join(v["fails"]))])


def t_rules(d: dict) -> rf.Table:
    D = d["decision"]
    p, s, g1 = D["primary_gmr_selected_vs_C0_refit"], D["secondary_gmr_selected_vs_v1"], D["gmr_v1_vs_C0_refit"]
    pr, pc = D["predictions"], D["prediction_signs_correct"]
    rows = [
        {"q": "primary: GMR(selected, C0_refit)", "v": ci(p), "w": f"{p['wins']}/18",
         "r": "CI upper < 1 and 0 failed: replicated; CI lower > 1: worse; else inconclusive",
         "o": D["outcome"], "pr": f"ln {pr['ln GMR(selected, C0_refit)']}: "
                                   + ("correct" if pc["ln GMR(selected, C0_refit)"] else "wrong")},
        {"q": "secondary: GMR(selected, M_bias v1)", "v": ci(s), "w": f"{s['wins']}/18",
         "r": "CI upper <= 1.05: non-inferior", "o": D.get("secondary", ""),
         "pr": f"ln {pr['ln GMR(selected, M_bias v1)']}: " + ("correct" if pc["ln GMR(selected, M_bias v1)"]
                                                              else "wrong")},
        {"q": "GMR(M_bias v1, C0_refit) (reported)", "v": ci(g1), "w": f"{g1['wins']}/18", "r": "-", "o": "-",
         "pr": f"ln {pr['ln GMR(M_bias v1, C0_refit)']}: " + ("correct" if pc["ln GMR(M_bias v1, C0_refit)"]
                                                              else "wrong")},
        {"q": "I5-1 stored GMR(M_bias, C0_refit)", "v": ci(D["I5_1_stored"]["M_bias_vs_C0_refit"]),
         "w": f"{D['I5_1_stored']['M_bias_vs_C0_refit']['wins']}/18", "r": "-", "o": "-", "pr": "-"},
        {"q": "Part B", "v": "-", "w": "-", "r": "-", "o": "not run", "pr": "-"},
    ]
    return rf.Table("03_table_i17_2_rules", "I17-2 pre-registered rules: values, labels and predicted signs",
                    [col("q", "quantity", kind="text"), col("v", "GMR [95% CI]"), col("w", "lower"),
                     col("r", "rule", kind="text"), col("o", "label", kind="text"),
                     col("pr", "predicted sign", kind="text")], rows,
                    ["GMR: exp(mean ln(method / control)) over 18 DEV DUTs, device bootstrap 10,000, 95% CI. "
                     f"Failed DUTs (selected, C0_refit): {D['n_failed_selected_C0']}."])


def t_arms(d: dict) -> rf.Table:
    D = d["decision"]
    rows = []
    for a in ARMS:
        t = D["table"][a]
        r = {"a": a, "m": f"{t['all_device_mean']:.5f}", "c": f"{t['combined']:.5f}", "md": f"{t['median']:.5f}",
             "g0": ci(t["gmr_vs_C0_refit"]) if "gmr_vs_C0_refit" in t else "1",
             "g0c": ci(t["gmr_vs_C0_refit_cluster"]) if "gmr_vs_C0_refit_cluster" in t else "-",
             "g1": ci(t["gmr_vs_v1"]) if "gmr_vs_v1" in t else "1",
             "g1c": ci(t["gmr_vs_v1_cluster"]) if "gmr_vs_v1_cluster" in t else "-"}
        rows.append(r)
    st = D["I5_1_stored"]
    rows.append({"a": "I5-1 stored C0_refit / M_bias", "m": f"{st['C0_refit']['all_device_mean']:.5f} / "
                 f"{st['M_bias']['all_device_mean']:.5f}", "c": "", "md": "", "g0": "", "g0c": "", "g1": "", "g1c": ""})
    return rf.Table("04_table_i17_2_arms", "I17-2 held-out RRMS of every arm, 18 IHP DEV DUTs, 300 K (LODO within "
                    "class)", [col("a", "arm", kind="text"), col("m", "all-device"), col("c", "combined"),
                               col("md", "median"), col("g0", "GMR vs C0_refit"),
                               col("g0c", "cluster CI"), col("g1", "GMR vs M_bias v1"), col("g1c", "cluster CI ")],
                    rows, ["Strict paper RRMS on the I5-1 frozen measured inclusion. Cluster bootstrap: clusters = "
                           f"class and W/L ({len(set(D['clusters'].values()))} clusters); reported, not deciding. "
                           "sym_e1_l{lambda}: symmetric inputs, EPS 0.01 V, penalty lambda (V), VDD = class supply."])


def t_class(d: dict) -> rf.Table:
    D = d["decision"]
    rows = []
    for c, lab in (("lv", "LV (1.2 V)"), ("hv", "HV (3.3 V)")):
        b = D["by_class"][c]
        rows.append({"c": lab, "n": str(b["n"]), "c0": f"{b['agg']['C0_refit']['all_device_mean']:.5f}",
                     "v1": f"{b['agg']['M_bias_v1']['all_device_mean']:.5f}",
                     "s": f"{b['agg']['selected']['all_device_mean']:.5f}",
                     "g0": ci(b["selected_vs_C0_refit"]) + f" {b['selected_vs_C0_refit']['wins']}/{b['n']}",
                     "g1": ci(b["selected_vs_v1"]) + f" {b['selected_vs_v1']['wins']}/{b['n']}",
                     "gv": ci(b["v1_vs_C0_refit"]) + f" {b['v1_vs_C0_refit']['wins']}/{b['n']}"})
    return rf.Table("05_table_i17_2_class", "I17-2 LV / HV split (reported, no rule)",
                    [col("c", "class", kind="text"), col("n", "DUTs"), col("c0", "C0_refit"), col("v1", "M_bias v1"),
                     col("s", "selected"), col("g0", "GMR(selected, C0_refit)"), col("g1", "GMR(selected, v1)"),
                     col("gv", "GMR(v1, C0_refit)")], rows, ["Device bootstrap within class."])


def t_lambda(d: dict) -> rf.Table:
    D, I = d["decision"], d["inner"]
    rows = []
    for t in sorted(D["lambda_star"], key=dev_key):
        sc = I["outer"][t]["scores"]
        rows.append({"t": t, "n": str(len(I["plan"][t]["train"])),
                     **{f"l{int(lam)}": f"{sc[f'{lam:g}']:.4f}" for lam in LAMS},
                     "s": f"{D['lambda_star'][t]:g}", "r": f"{D['selected_rows'][t]:.5f}"})
    cnt = ", ".join(f"{k} V: {v}" for k, v in sorted(D["lambda_star_counts"].items(), key=lambda kv: float(kv[0])))
    return rf.Table("06_table_i17_2_lambda", "I17-2 nested selection: inner score (mean ln RRMS of inner held-out "
                    "refits) per outer fold and lambda, lambda* and selected row",
                    [col("t", "held-out DUT", kind="text"), col("n", "training DUTs")]
                    + [col(f"l{int(lam)}", f"lambda {lam:g} V") for lam in LAMS]
                    + [col("s", "lambda* (V)"), col("r", "selected RRMS")], rows,
                    [f"lambda* counts: {cnt}. Inner folds: training DUTs sorted by (polarity, tag), i mod 3; one "
                     "member (seed 0); ties to the larger lambda."])


def t_neg(d: dict) -> rf.Table:
    N = d["decision"]["negative_fractions"]
    rows = []
    for a in ("M_bias_v1", "sym_e1_l0", "sym_e1_l1", "sym_e1_l2", "sym_e1_l5", "sym_e1_l10", "selected"):
        if a not in N:
            continue
        n = N[a]
        rows.append({"a": a, "hgds": f"{100 * n['heldout_mean_gds']:.2f}", "hgm": f"{100 * n['heldout_mean_gm']:.2f}",
                     "tgds": f"{100 * n['train_mean_gds']:.2f}", "tgm": f"{100 * n['train_mean_gm']:.2f}"})
    c = N["card_only"]
    rows.append({"a": "card only (starting cards)", "hgds": f"{100 * c['frac_neg_gds_card']:.3f}",
                 "hgm": f"{100 * c['frac_neg_gm_card']:.3f}", "tgds": "", "tgm": ""})
    return rf.Table("07_table_i17_2_negative", "I17-2 negative corrected gds / gm on the penalty grids (% of points)",
                    [col("a", "arm", kind="text"), col("hgds", "held-out gds < 0"), col("hgm", "held-out gm < 0"),
                     col("tgds", "training gds < 0"), col("tgm", "training gm < 0")], rows,
                    ["Fold model on the held-out DUT's starting-card grid (held-out) and on its training grids, "
                     "mean over folds; grid 75 x 75 forward quadrant to the class supply, forward current > 1 nA."])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_per_device(d: dict, path: Path) -> list[str]:
    R = d["decision"]["rows"]
    tags = sorted(R["C0_refit"], key=dev_key)
    arms = ("C0_refit", "M_bias_v1", "sym_e1_l10", "selected")
    fig, ax = plt.subplots(figsize=(14, 5.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    off = np.linspace(-0.25, 0.25, len(arms))
    for j, a in enumerate(arms):
        ax.scatter(np.arange(len(tags)) + off[j], [R[a][t] for t in tags], color=COLOR[a], s=28, zorder=3,
                   label=f"{a} (all-device {d['decision']['table'][a]['all_device_mean']:.4f})")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
    nlv = sum(t.startswith("lv_") for t in tags)
    ax.axvline(nlv - 0.5, color=MUTED, lw=1.2)
    ax.set_ylabel("held-out strict paper RRMS (log scale)", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.32))
    ax.set_title("I17-2 IHP SG13G2 300 K, per DEV DUT (left: LV 1.2 V, right: HV 3.3 V): held-out RRMS after the "
                 "card refit with g frozen", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["decision"])]


def fig_ratio(d: dict, path: Path) -> list[str]:
    R = d["decision"]["rows"]
    tags = sorted(R["C0_refit"], key=dev_key)
    fig, ax = plt.subplots(figsize=(14, 5.4), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    x = np.arange(len(tags))
    for a, dx in (("M_bias_v1", -0.15), ("selected", 0.15)):
        y = [R[a][t] / R["C0_refit"][t] for t in tags]
        ax.bar(x + dx, np.array(y) - 1, bottom=1, width=0.28, color=COLOR[a],
               label=f"{a} / C0_refit (GMR {d['decision']['table'][a]['gmr_vs_C0_refit']['ratio']:.3f})")
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
    nlv = sum(t.startswith("lv_") for t in tags)
    ax.axvline(nlv - 0.5, color=MUTED, lw=1.2)
    for t_, lam in zip(x, [d["decision"]["lambda_star"][t] for t in tags]):
        ax.text(t_, ax.get_ylim()[1], f"{lam:g}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_ylabel("RRMS ratio to C0_refit (below 1: lower error)", fontsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.32), ncol=2)
    ax.set_title("I17-2 per-DUT ratio to C0_refit; numbers above: lambda* (V) of the selected row", fontsize=10,
                 color=INK, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["decision"])]


def fig_lambda_curve(d: dict, path: Path) -> list[str]:
    T = d["decision"]["table"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=DPI)
    fig.patch.set_facecolor(BG)
    style(ax)
    xs = np.arange(len(LAMS))
    for key, colr, lab in (("gmr_vs_C0_refit", BLUE, "GMR vs C0_refit"), ("gmr_vs_v1", ORANGE, "GMR vs M_bias v1")):
        g = [T[f"sym_e1_l{int(lam)}"][key] for lam in LAMS]
        ax.errorbar(xs, [x["ratio"] for x in g], yerr=[[x["ratio"] - x["ci_low"] for x in g],
                                                       [x["ci_high"] - x["ratio"] for x in g]],
                    color=colr, marker="o", capsize=3, label=lab)
        s = T["selected"][key]
        ax.errorbar([len(LAMS) + 0.3], [s["ratio"]], yerr=[[s["ratio"] - s["ci_low"]], [s["ci_high"] - s["ratio"]]],
                    color=colr, marker="s", capsize=3)
    ax.axhline(1, color=INK, lw=0.8)
    ax.axhline(1.05, color=GREY, lw=0.8, ls="--")
    ax.set_xticks(list(xs) + [len(LAMS) + 0.3])
    ax.set_xticklabels([f"{lam:g}" for lam in LAMS] + ["selected"])
    ax.set_xlabel("penalty lambda (V)")
    ax.set_ylabel("GMR [95% CI]")
    ax.legend(fontsize=8.5, frameon=False)
    ax.set_title("I17-2 lambda curve, 18 IHP DEV DUTs (dashed: margin 1.05)", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(F["decision"])]


def iv_tags(d: dict) -> list[str]:
    R = d["decision"]["rows"]
    tags = sorted(R["C0_refit"])
    ratio = {t: R["selected"][t] / R["C0_refit"][t] for t in tags}
    out = []
    for cls in ("lv", "hv"):
        sub = sorted((t for t in tags if t.startswith(cls)), key=lambda t: ratio[t])
        out += [t for t in (sub[0], sub[-1]) if t not in out]
    return out


def resim(d: dict, tags) -> dict:
    import i17_2_ihp as drv
    from cryoml.explore import i5_1_ihp_real as m5
    from cryoml.explore.e1_correction import score_device
    from cryoml.explore.i3_2_bias import correction_from_json
    from cryoml.explore.i12_1_symm import SymCorrection
    D = d["decision"]
    cards = load(I17 / "base/heldout_cards.json")
    bc = load(I17 / "base/corrections.json")
    out = {}
    for tag in tags:
        dev = drv.devices()[tag]
        prob = m5.MeasProblem(dev, 300, m5.base_box(dev))
        stock = m5.stock_params(dev, m5.BASE_SET)
        sims, sc = {}, {}
        s0 = prob.simulate_params(stock)
        rec0 = m5.to_record(dev, 300, s0, stock)
        sims["stock"], sc["stock"] = s0, score_device(rec0, s0)["rrms"]
        c0 = dict(cards[f"C0_refit|{tag}"])
        s = prob.simulate_params(c0)
        sims["C0_refit"], sc["C0_refit"] = s, score_device(m5.to_record(dev, 300, s, c0), s)["rrms"]
        check(sc["C0_refit"], D["rows"]["C0_refit"][tag], f"{tag} C0_refit")
        p1 = dict(cards[f"M_bias|{tag}"])
        r1 = m5.to_record(dev, 300, prob.simulate_params(p1), p1)
        g1 = correction_from_json(bc[f"M_bias|{tag}"]).apply(r1)
        sims["M_bias_v1"], sc["M_bias_v1"] = g1, score_device(r1, g1)["rrms"]
        check(sc["M_bias_v1"], D["rows"]["M_bias_v1"][tag], f"{tag} v1")
        lam = D["lambda_star"][tag]
        v = f"sym_e1_l{int(lam)}"
        ps = dict(load(I17 / f"lodo_{v}/heldout_cards.json")[tag])
        corr = SymCorrection.from_json(load(I17 / f"lodo_{v}/corrections.json")[f"{v}|{tag}"])
        rs = m5.to_record(dev, 300, prob.simulate_params(ps), ps)
        gs = corr.apply(rs)
        sims["selected"], sc["selected"] = gs, score_device(rs, gs)["rrms"]
        check(sc["selected"], D["rows"]["selected"][tag], f"{tag} selected")
        out[tag] = {"rec": rec0, "sims": sims, "score": sc, "lam": lam}
    return out


def fig_iv(d: dict, tag: str, path: Path) -> list[str]:
    r = d["iv"][tag]
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
               label="measured 300 K (filled: scored points; hollow: trimmed before the metric; faded: excluded curve)"),
        Line2D([], [], color=GREY, ls="--", lw=1.2, label=f"NGSpice, stock PDK card (no fit): RRMS {sc['stock']:.4f}"),
        Line2D([], [], color=BLUE, lw=1.2,
               label=f"NGSpice re-simulated, C0_refit held-out card (no correction): RRMS {sc['C0_refit']:.4f}"),
        Line2D([], [], color=ORANGE, lw=1.2,
               label=f"NGSpice re-simulated, M_bias v1 held-out card x exp(fold g): RRMS {sc['M_bias_v1']:.4f}"),
        Line2D([], [], color=AQUA, lw=1.2,
               label=f"NGSpice re-simulated, selected held-out card x exp(fold g) (lambda* {r['lam']:g} V): RRMS "
                     f"{sc['selected']:.4f}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"{tag} (IHP SG13G2 DEV, 300 K): held-out cards (LODO within class) re-simulated in ngspice-46 + "
                 "PSP103 OSDI; g applied in Python (no element: Part B not run).\nRRMS equals the stored rows.",
                 fontsize=9.4, color=INK)
    fig.tight_layout(rect=(0, 0.2, 1, 0.93))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return [rel(I17 / "base/heldout_cards.json"), "I17_2/lodo_sym_e1_l{lambda*}/{corrections,heldout_cards}.json",
            "I5_1/data/ihp_meas.npz (not in git)"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--smoke-out", default="", help="code check on I17_2/smoke; write here, no index update")
    args = ap.parse_args()
    global I17
    folder = rf.REPORT_ROOT / f"{args.date}_{TOPIC}"
    if args.smoke_out:
        I17 = EXP / "I17_2/smoke"
        for k in ("decision", "pens", "base", "inner"):
            F[k] = I17 / Path(*F[k].parts[-2:]) if k != "decision" else I17 / "decision.json"
        folder = Path(args.smoke_out)
    folder.mkdir(parents=True, exist_ok=True)
    d = {k: load(p) for k, p in F.items()}
    # recompute the primary GMR and the selected rows from stored per-device rows
    from cryoml.explore.i3_2_bias import paired_ratio
    D = d["decision"]
    sel = {t: load(I17 / f"lodo_sym_e1_l{int(D['lambda_star'][t])}/results.json")["per_device"][t]
           for t in D["lambda_star"]}
    for t in sel:
        check(sel[t], D["selected_rows"][t], f"selected row {t}", 0.0)
    g = paired_ratio(sel, d["base"]["per_device"]["C0_refit"])
    check(g["ratio"], D["primary_gmr_selected_vs_C0_refit"]["ratio"], "primary GMR", 1e-12)
    tags = iv_tags(d)
    d["iv"] = resim(d, tags)
    items, tables = [], []

    def add_table(t, desc, srcs):
        rf.render_table(t, folder / f"{t.name}.png")
        tables.append(t)
        items.append((f"{t.name}.png", desc, srcs))

    def add_fig(name, fn, desc):
        items.append((name, desc, fn(d, folder / name)))

    add_table(t_choice(d), "Dataset choice, setup and the reasons Part B was not run", [rel(F["prereg"])])
    add_table(t_validity(d), "Validity V1-V4", [rel(F["base"]), rel(F["inner"]), rel(F["pens"])])
    add_table(t_rules(d), "Primary and secondary GMR vs the pre-registered rules, predicted signs",
              [rel(F["decision"])])
    add_table(t_arms(d), "Held-out RRMS of every arm, GMR vs C0_refit and vs M_bias v1 (device and cluster CI)",
              [rel(F["decision"])])
    add_table(t_class(d), "LV / HV split", [rel(F["decision"])])
    add_table(t_lambda(d), "Nested lambda selection per outer fold", [rel(F["inner"]), rel(F["decision"])])
    add_table(t_neg(d), "Negative corrected gds / gm fractions on the penalty grids", [rel(F["decision"])])
    add_fig("08_i17_2_per_device.png", fig_per_device, "Per-DUT held-out RRMS: C0_refit, M_bias v1, lambda 10, "
            "selected")
    add_fig("09_i17_2_ratio_per_device.png", fig_ratio, "Per-DUT RRMS ratio to C0_refit (v1, selected) with lambda*")
    add_fig("10_i17_2_lambda_curve.png", fig_lambda_curve, "GMR vs lambda (vs C0_refit and vs v1) and the selected "
            "variant")
    for k, tag in enumerate(tags):
        name = f"{11 + k:02d}_i17_2_iv_{tag}.png"
        items.append((name, f"I-V of {tag}: measured 300 K points, NGSpice of the stock PDK card, and NGSpice "
                            "re-simulated C0_refit, M_bias v1 and selected held-out cards (x exp(g))",
                      fig_iv(d, tag, folder / name)))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    gen = f"`.venv/bin/python -I scripts/research/plot_explore_i17_2.py --date {args.date}`"
    md = [f"# Tables, {args.date} {TOPIC}", "", f"Generated {now} by {gen}.", ""]
    for t in tables:
        md += [rf.table_markdown(t), ""]
    (folder / "tables.md").write_text("\n".join(md))
    p, s = D["primary_gmr_selected_vs_C0_refit"], D["secondary_gmr_selected_vs_v1"]
    T = D["table"]
    readme = [
        f"# Report batch {args.date} · {TOPIC}", "",
        f"Generated {now} by {gen}. NGSpice runs: re-simulation of the stock, C0_refit, M_bias v1 and selected cards "
        f"for the {len(tags)} I-V DUTs (checked against the stored rows); everything else from stored files.", "",
        "## Contents", "",
        "- I17-2: second-PDK replication of the selected g v2 recipe on measured IHP SG13G2 (PSP103, I5-1 data), "
        "300 K, 18 DEV DUTs, LODO within class. Pre-registration `out/research/explore/I17_2/prereg.md`, commit "
        f"{COMMITS['prereg']}; code {COMMITS['code']}.",
        f"- Outcome (pre-registered rule): **{D['outcome']}**; secondary: {D.get('secondary')}; Part B not run. "
        f"GMR(selected, C0_refit) {ci(p)} ({p['wins']}/18 lower); GMR(selected, M_bias v1) {ci(s)} "
        f"({s['wins']}/18 lower).",
        f"- All-device RRMS: C0_refit {T['C0_refit']['all_device_mean']:.5f}, M_bias v1 "
        f"{T['M_bias_v1']['all_device_mean']:.5f}, selected {T['selected']['all_device_mean']:.5f}; lambda* counts "
        + ", ".join(f"{k} V: {v}" for k, v in sorted(D["lambda_star_counts"].items(), key=lambda kv: float(kv[0])))
        + ".", "", "## Figures and tables", ""]
    for png, desc, srcs in items:
        readme.append(f"- `{png}`: {desc}; sources: " + ", ".join(f"`{x}`" for x in srcs) + ".")
    readme += ["", "- `tables.md`: the tables above in markdown.", ""]
    (folder / "README.md").write_text("\n".join(readme))
    summary = [f"I17-2 IHP SG13G2 300 K (18 DEV DUTs): {D['outcome']}; {D.get('secondary')}; Part B not run.",
               f"GMR(selected, C0_refit) {ci(p)}; GMR(selected, v1) {ci(s)}.",
               f"All-device C0_refit {T['C0_refit']['all_device_mean']:.4f}, v1 {T['M_bias_v1']['all_device_mean']:.4f},"
               f" selected {T['selected']['all_device_mean']:.4f}."]
    if not args.smoke_out:
        rf.update_index(args.date, TOPIC, folder, [i[0] for i in items], summary)
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
