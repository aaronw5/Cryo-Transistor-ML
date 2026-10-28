#!/usr/bin/env python3
"""Report batch for I25-1 (amendment A3): the paper's 77 K task with every
released multiplier set to 1, fitted from 3 random starts per device with the
multi-round direct NGSpice fit; arm A (7 parameters) and arm B (screened set),
with and without g v2.

Reads ``out/research/explore/I25_1/`` (decision.json, arm{A,B}/final/,
arm{A,B}/records/, old/, specs/, stepA/).  The I-V curves are the NGSpice
re-simulations of the final cards stored by ``i25_1_scratch.py --stage final``
(V4: each reproduces its stored RRMS) and ``--stage old`` (released card);
g is applied in Python.  Every plotted RRMS is recomputed from the stored
curves and checked against decision.json.

Writes ``out/research/report/<date>_i25_1/`` (tables, figures, iv/ with all 18
devices for both arms) and a pptx deck ``slides/i25_1_from_scratch.pptx``.

  PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i25_1.py --date 2026-10-28
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import report_figs as rf  # noqa: E402
from plot_explore_e2_e3_e4 import (AQUA, BG, BLUE, DPI, GREY, INK, ORANGE, RED, VIOLET,  # noqa: E402
                                   style)

from cryoml.explore import i25_1_scratch as m  # noqa: E402
from cryoml.explore.e1_correction import load_devices  # noqa: E402

plt = rf.plt
I25 = ROOT / "out/research/explore/I25_1"
OLD = I25 / "old"
ARM_NAME = {"A": "7 parameters", "B": "screened set"}
C_PUB, C_CARD, C_G, C_STOCK = GREY, BLUE, RED, ORANGE


def devlab(t: str) -> str:
    return ("nMOS " if t.startswith("nmos") else "pMOS ") + rf.device_label(t)


def jload(p):
    return json.loads(Path(p).read_text())


def col(key, head, kind="rtext", group=None):
    return {"key": key, "head": head, "md": head, "kind": kind, "group": group}


def f4(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.4f}"


def ci(g):
    return f"{g['ratio']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}]"


def kint(x):
    return "—" if x is None else f"{x:,.0f}"


def check(a, b, what, tol=1e-9):
    if not (abs(a - b) <= tol or (np.isnan(a) and np.isnan(b))):
        raise RuntimeError(f"check failed: {what}: {a} vs {b}")


class Batch:
    def __init__(self, out: Path):
        self.out, self.tables, self.files, self.n = out, [], [], 0

    def table(self, t: rf.Table):
        self.n += 1
        name = f"{self.n:02d}_{t.name}.png"
        rf.render_table(t, self.out / name)
        self.tables.append(rf.table_markdown(t))
        self.files.append(name)
        return name

    def fig(self, name, fig):
        self.n += 1
        fn = f"{self.n:02d}_{name}.png"
        fig.savefig(self.out / fn, dpi=DPI, facecolor="white")
        plt.close(fig)
        self.files.append(fn)
        return fn


# ---------------------------------------------------------------------------
def load_all() -> dict:
    d = {"dec": jload(I25 / "decision.json"), "old": jload(OLD / "old.json"),
         "specs": jload(I25 / "specs/specs.json")}
    d["final"] = {a: jload(I25 / f"arm{a}/final/final.json") for a in "AB"}
    d["curves"] = {a: dict(np.load(I25 / f"arm{a}/final/curves.npz")) for a in "AB"}
    d["pub_curves"] = dict(np.load(OLD / "published_curves.npz"))
    d["recs"] = {a: {r.tag: r for r in load_devices(I25 / f"arm{a}/records/all18")} for a in "AB"}
    # recompute primary RRMS from the stored curves and check
    for a in "AB":
        for k, rows in d["final"][a]["arms"].items():
            for t, row in rows.items():
                rec = d["recs"][a][t]
                sims = [d["curves"][a][f"{k}|{t}|{c.key}"] for c in rec.curves]
                check(m.score_rule(rec, sims, m.frozen_tags(rec))["rrms"], row["rrms"], f"{a} {k} {t}")
                check(m.score_rule(rec, sims, m.m_rule_tags(rec))["rrms"], row["rrms_m_rule"], f"{a} {k} {t} M")
    for t, row in d["old"]["arms"]["published"].items():
        if t not in d["recs"]["B"]:
            continue
        rec = d["recs"]["B"][t]
        sims = [d["pub_curves"][f"published|{t}|{c.key}"] for c in rec.curves]
        check(m.score_rule(rec, sims, m.frozen_tags(rec))["rrms"], row["rrms"], f"published {t}")
    return d


def t_main(d, mode="primary", lodo=True) -> rf.Table:
    dec, fin = d["dec"], d["final"]
    ref = dec["reference"]
    nb = len(d["specs"]["selected"])
    pubs = ref["paper"][mode]
    rows = [{"row": "stock SkyWater tt card at 77 K (no fit)", "fa": f4(ref["stock_77K"][mode]["all_device"]),
             "fc": f4(ref["stock_77K"][mode]["combined"]), "la": "", "lc": "", "calls": "0"},
            {"row": "released 77 K card, multipliers = 1 (no fit)", "fa": f4(ref["mult1_no_fit"][mode]["all_device"]),
             "fc": f4(ref["mult1_no_fit"][mode]["combined"]), "la": "", "lc": "", "calls": "0"},
            {"row": "paper card (released, tuned)", "fa": f4(pubs["all_device"]), "fc": f4(pubs["combined"]),
             "la": "", "lc": "", "calls": "—"}]
    for a, name in (("A", "7p"), ("B", f"{nb}p")):
        s = fin[a]["summary"]
        c = dec["arms"][a]["calls_per_device"]
        rows.append({"row": f"{name} card fit, no g (C0_refit)", "fa": f4(s["C0_refit"][mode]["all_device"]),
                     "fc": f4(s["C0_refit"][mode]["combined"]), "la": f4(s["C0_refit"][mode]["all_device"]),
                     "lc": f4(s["C0_refit"][mode]["combined"]), "calls": kint(c["C0_refit"])})
        rows.append({"row": f"{name} card fit + g v2", "fa": f4(s["full_g"][mode]["all_device"]),
                     "fc": f4(s["full_g"][mode]["combined"]), "la": f4(s["lodo_g"][mode]["all_device"]),
                     "lc": f4(s["lodo_g"][mode]["combined"]),
                     "calls": f"{kint(c['full_g'])} / {kint(c['lodo_g'])}" if lodo else kint(c["full_g"])})
    label = "primary: paper RRMS, released-card curve set" if mode == "primary" else \
        "secondary: paper RRMS, M-rule curve set (measured current only)"
    cols = [col("row", "card", "text"), col("fa", "all-device", group="full data (18 fitted)"),
            col("fc", "combined", group="full data (18 fitted)")]
    if lodo:
        cols += [col("la", "all-device", group="LODO held-out g"), col("lc", "combined", group="LODO held-out g")]
    cols.append(col("calls", "NGSpice calls / device"))
    notes = ["Task: the paper's 18 devices, card files, 7 parameters and RRMS; every released multiplier set to 1; "
             "3 random starts per device, best of 3 by training RRMS; multi-round direct NGSpice fit (rounds 1-3 + "
             "FD).", "full data: g trained on all 18 records (lambda by the nested procedure on all 18), each card "
             "refit with g in the loop; in-sample RRMS.",
             "combined = mean of the nMOS and pMOS family means; all-device = mean over 18 devices.",
             "calls: Step A summed over the 3 starts + lambda-selection refits / 18 + own refit."]
    if lodo:
        notes.insert(2, "LODO: g trained on the other 17 devices (secondary); C0_refit has no g, so it is the "
                        "same in both column groups.")
    return rf.Table(f"table_main_{mode}{'' if lodo else '_full'}", f"I25-1: RRMS, {label}", cols, rows, notes)


def t_stages(d) -> rf.Table:
    rows = []
    for r in d["dec"]["stage_table"]:
        rows.append({"s": r["stage"], "aa": f4(r.get("A_all")), "ac": f4(r.get("A_comb")),
                     "ba": f4(r.get("B_all")), "bc": f4(r.get("B_comb")),
                     "oa": f4(r.get("old_all")), "oc": f4(r.get("old_comb"))})
    nb = len(d["specs"]["selected"])
    return rf.Table("table_stages", "RRMS after every round (selected start of each device), new 7p and "
                    f"{nb}p chains vs the old 43p chain, state C, primary curve set",
                    [col("s", "stage", "text"), col("aa", "all-device", group="7p, multipliers 1, random"),
                     col("ac", "combined", group="7p, multipliers 1, random"),
                     col("ba", "all-device", group=f"{nb}p, multipliers 1, random"),
                     col("bc", "combined", group=f"{nb}p, multipliers 1, random"),
                     col("oa", "all-device", group="old 43p chain"), col("oc", "combined", group="old 43p chain")],
                    rows,
                    ["Old chain: multiplier-free centers with the pmos_L0p35_W1p6 named-multiplier bug, round 1 seeded "
                     "by the old 7p card; cards rescored in state C. Old C0_refit = I13-1 rows; old g full data = I16-1 "
                     "in-sample rows; old g LODO = I15-2 selected rows."])


def t_spread(d) -> rf.Table:
    rows = []
    for t in sorted(d["dec"]["arms"]["A"]["multistart"]["spread"], key=rf.device_key):
        r = {"dev": devlab(t)}
        for a in "AB":
            sp_ = d["dec"]["arms"][a]["multistart"]["spread"][t]
            r[a] = f"{sp_['min']:.4f} / {sp_['median']:.4f} / {sp_['max']:.4f} (s{sp_['start']})"
        rows.append(r)
    for a in "AB":
        ps = d["dec"]["arms"][a]["multistart"]["per_start"]
        fin = d["dec"]["arms"][a]["multistart"].get("per_start_finite") or [18] * len(ps)
        rows.append({"dev": f"all-device, start 0 / 1 / 2 (arm {a})",
                     "A" if a == "A" else "B": " / ".join(
                         f4(x["all_device_mean_rrms"]) if np.isfinite(x["all_device_mean_rrms"])
                         else f"non-finite ({n}/18 finite)" for x, n in zip(ps, fin))})
    nb = len(d["specs"]["selected"])
    return rf.Table("table_multistart", "Spread over the 3 random starts per device after round 3 + FD: "
                    "min / median / max training RRMS (selected start)",
                    [col("dev", "device", "text"), col("A", "7p"), col("B", f"{nb}p")], rows,
                    ["Starts: local coordinates uniform in [-1, 1] (log-uniform in the box), seeds 20261027 + 1000 k + "
                     "device index. Selection: lowest training RRMS."])


def t_gmr(d) -> rf.Table:
    rows = []
    names = {"full_g_vs_C0_refit": "card + g vs card (C0_refit), full data",
             "full_g_vs_paper": "card + g vs paper card, full data",
             "C0_refit_vs_paper": "card (C0_refit) vs paper card",
             "lodo_g_vs_C0_refit": "card + g (LODO) vs card (C0_refit)",
             "lodo_g_vs_paper": "card + g (LODO) vs paper card"}
    for a in "AB":
        for k, lab in names.items():
            gp, gs = d["dec"]["gmr"][a]["primary"][k], d["dec"]["gmr"][a]["m_rule"][k]
            rows.append({"arm": ARM_NAME[a], "cmp": lab, "p": ci(gp), "pw": f"{gp['wins']}/18",
                         "pl": gp["label"], "s": ci(gs), "sl": gs["label"]})
    for k, lab in (("A_full_g_vs_B_full_g", "7p + g vs screened set + g, full data"),
                   ("B_full_g_vs_old_full_g", "screened set + g vs old 43p + g, full data"),
                   ("B_lodo_g_vs_old_lodo_g", "screened set + g vs old 43p + g, LODO")):
        g = d["dec"]["gmr"][k]
        rows.append({"arm": "", "cmp": lab, "p": ci(g), "pw": f"{g['wins']}/18", "pl": g["label"],
                     "s": "", "sl": ""})
    return rf.Table("table_gmr", "Geometric-mean RRMS ratio [95% device-bootstrap CI]; lower = first card better",
                    [col("arm", "arm", "text"), col("cmp", "comparison", "text"),
                     col("p", "GMR", group="primary curve set"), col("pw", "wins", group="primary curve set"),
                     col("pl", "label", group="primary curve set"), col("s", "GMR", group="M-rule curve set"),
                     col("sl", "label", group="M-rule curve set")], rows,
                    ["label: lower if CI upper < 1, higher if CI lower > 1, else inconclusive (pre-registered)."])


def t_per_device(d, a) -> rf.Table:
    fin = d["final"][a]["arms"]
    pub = d["old"]["arms"]["published"]
    lam = d["dec"]["arms"][a]["lambda_star"]
    st = d["dec"]["reference"]
    rows = []
    for t in sorted(fin["C0_refit"], key=rf.device_key):
        rows.append({"dev": devlab(t), "m1": f4(fin["mult1"][t]["rrms"]), "pub": f4(pub[t]["rrms"]),
                     "start": f4(fin["start"][t]["rrms"]), "c0": f4(fin["C0_refit"][t]["rrms"]),
                     "fg": f4(fin["full_g"][t]["rrms"]), "lg": f4(fin["lodo_g"][t]["rrms"]),
                     "lam": f"{lam.get(t, lam['full']):g}"})
    s = d["final"][a]["summary"]
    rows.append({"dev": "all-device mean", **{k: f4(s[kk]["primary"]["all_device"]) for k, kk in (
        ("m1", "mult1"), ("start", "start"), ("c0", "C0_refit"), ("fg", "full_g"), ("lg", "lodo_g"))},
        "pub": f4(st["paper"]["primary"]["all_device"]), "lam": f"{lam['full']:g} (full)"})
    rows.append({"dev": "combined", **{k: f4(s[kk]["primary"]["combined"]) for k, kk in (
        ("m1", "mult1"), ("start", "start"), ("c0", "C0_refit"), ("fg", "full_g"), ("lg", "lodo_g"))},
        "pub": f4(st["paper"]["primary"]["combined"]), "lam": ""})
    return rf.Table(f"table_per_device_arm{a}", f"Per-device RRMS, arm {a} ({ARM_NAME[a]}), primary curve set",
                    [col("dev", "device", "text"), col("m1", "multipliers 1, no fit"), col("pub", "paper card"),
                     col("start", "Step A card"), col("c0", "card fit (C0_refit)"),
                     col("fg", "card + g, full data"), col("lg", "card + g, LODO"), col("lam", "lambda* (V)")],
                    rows, ["Every value is the NGSpice re-simulation of the stored card (g applied in Python)."])


def t_params(d) -> rf.Table:
    ov = d["specs"]["overlap_with_old43"]
    rows = [{"k": "screened set (around the multiplier-1 card)", "v": f"{len(d['specs']['selected'])}: "
             + ", ".join(d["specs"]["selected"])},
            {"k": "in both", "v": f"{ov['n_both']}"},
            {"k": "old 43 only", "v": ", ".join(ov["only_old"]) or "none"},
            {"k": "new only", "v": ", ".join(ov["only_new"]) or "none"},
            {"k": "Jaccard", "v": f"{ov['jaccard']:.3f}"}]
    return rf.Table("table_parameter_set", "Parameter screen around the multiplier-1 card (115 candidates, "
                    "unchanged rule) vs the earlier 43 (screen around the released tuned card)",
                    [col("k", "item", "text"), col("v", "value", "text")], rows)


def fig_stages(d):
    st = [r for r in d["dec"]["stage_table"] if r["stage"].startswith("round") or r["stage"] in
          ("C0_refit", "g full data")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(st))
    nb = len(d["specs"]["selected"])
    for key, c, lab in (("old_all", GREY, "old 43p chain (released-card centers)"),
                        ("A_all", ORANGE, "7p, multipliers 1, random starts"),
                        ("B_all", BLUE, f"{nb}p, multipliers 1, random starts")):
        ax.plot(x, [r.get(key, np.nan) for r in st], "o-", color=c, label=lab)
    ax.axhline(d["dec"]["reference"]["paper"]["primary"]["all_device"], color=INK, ls="--", lw=1,
               label="paper card 0.2751")
    ax.set_xticks(x, [r["stage"] for r in st], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("all-device RRMS (primary)")
    ax.set_yscale("log")
    style(ax)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("RRMS after every round, then card refit and card + g (full data)", fontsize=11)
    fig.tight_layout()
    return fig


def fig_forest(d):
    items = []
    for a in "AB":
        for k, lab in (("full_g_vs_C0_refit", "card + g vs card, full data"),
                       ("full_g_vs_paper", "card + g vs paper, full data"),
                       ("C0_refit_vs_paper", "card vs paper"),
                       ("lodo_g_vs_C0_refit", "card + g vs card, LODO")):
            items.append((f"arm {a} ({ARM_NAME[a]}): {lab}", d["dec"]["gmr"][a]["primary"][k]))
    items.append(("arm B + g vs old 43p + g, full data", d["dec"]["gmr"]["B_full_g_vs_old_full_g"]))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(items) + 1.2))
    for i, (lab, g) in enumerate(items):
        y = len(items) - 1 - i
        c = BLUE if g["label"] == "lower" else (RED if g["label"] == "higher" else GREY)
        ax.plot([g["ci_low"], g["ci_high"]], [y, y], color=c, lw=2)
        ax.plot(g["ratio"], y, "o", color=c)
        ax.text(1.02, y, f"{g['ratio']:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_yticks(range(len(items)), [lab for lab, _ in items][::-1], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("geometric-mean RRMS ratio (95% device bootstrap CI)")
    style(ax)
    fig.tight_layout()
    return fig


def fig_per_device(d):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    pub = d["old"]["arms"]["published"]
    for ax, a in zip(axs, "AB"):
        fin = d["final"][a]["arms"]
        tags = sorted(fin["C0_refit"], key=rf.device_key)
        x = np.arange(len(tags))
        for k, c, mk, lab in (("mult1", C_STOCK, "v", "multipliers 1, no fit"), (None, C_PUB, "s", "paper card"),
                              ("C0_refit", C_CARD, "o", "card fit"), ("full_g", C_G, "D", "card + g (full data)")):
            vals = [pub[t]["rrms"] if k is None else fin[k][t]["rrms"] for t in tags]
            ax.plot(x, vals, mk, color=c, label=lab, ms=5)
        ax.set_xticks(x, [devlab(t) for t in tags], rotation=70, fontsize=7)
        ax.set_yscale("log")
        ax.set_title(f"arm {a}: {ARM_NAME[a]}", fontsize=10)
        style(ax)
    axs[0].set_ylabel("RRMS (primary)")
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def fig_iv(d, a, tag):
    rec = d["recs"][a][tag]
    fin = d["final"][a]["arms"]
    cv = d["curves"][a]
    pub = d["old"]["arms"]["published"][tag]
    keys = [c.key for c in rec.curves]
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    axs = axs.ravel()
    for i, c in enumerate(rec.curves):
        ax = axs[i]
        x = np.abs(c.vd if c.kind == "idvd" else c.vg)
        meas = np.abs(np.asarray(c.target, float))
        ax.plot(x, meas, ".", color=INK, ms=3, label="measured 77 K")
        ax.plot(x, np.abs(d["pub_curves"][f"published|{tag}|{c.key}"]), "-", color=C_PUB, lw=1.4,
                label="paper card, NGSpice")
        ax.plot(x, np.abs(cv[f"C0_refit|{tag}|{c.key}"]), "-", color=C_CARD, lw=1.4,
                label="our card fit, NGSpice")
        ax.plot(x, np.abs(cv[f"full_g|{tag}|{c.key}"]), "--", color=C_G, lw=1.4,
                label="our card fit + g, NGSpice x exp(g)")
        if c.kind == "idvg":
            ax.set_yscale("log")
            pos = meas[meas > 0]
            if len(pos):
                ax.set_ylim(max(pos.min() * 0.3, 1e-13), pos.max() * 3)
        sc = c.key in m.frozen_tags(rec)

        def r(row):
            v = row["per_curve"].get(c.key)
            return "—" if v is None else f"{v:.3f}"
        ttl = (f"{c.key}  RRMS paper {r(pub)} | card {r(fin['C0_refit'][tag])} | +g {r(fin['full_g'][tag])}"
               if sc else f"{c.key}  (not in the scored set)")
        ax.set_title(ttl, fontsize=7.5)
        ax.set_xlabel("|V_DS| (V)" if c.kind == "idvd" else "|V_GS| (V)", fontsize=7)
        style(ax)
    for j in range(len(keys), len(axs)):
        axs[j].axis("off")
    h, lab = axs[0].get_legend_handles_labels()
    axs[-1].legend(h, lab, loc="center", fontsize=9, frameon=False)
    s = (f"{devlab(tag)}, arm {a} ({ARM_NAME[a]}): RRMS paper {pub['rrms']:.4f}, our card fit "
         f"{fin['C0_refit'][tag]['rrms']:.4f}, our card + g {fin['full_g'][tag]['rrms']:.4f} (full data); "
         f"multipliers 1 without fit {fin['mult1'][tag]['rrms']:.3f}")
    fig.suptitle(s, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def summary_lines(d) -> list[str]:
    nb = len(d["specs"]["selected"])
    out = ["I25-1 the paper's task (released 77 K card files, every multiplier = 1, paper RRMS) fitted from 3 random "
           f"starts per device with the multi-round direct NGSpice fit; arm A 7 parameters, arm B {nb} parameters."]
    for a in "AB":
        fs = d["final"][a]["summary"]
        g = d["dec"]["gmr"][a]["primary"]
        out.append(f"arm {a} ({ARM_NAME[a]}): card fit {fs['C0_refit']['primary']['all_device']:.4f} / "
                   f"{fs['C0_refit']['primary']['combined']:.4f}; card + g full data "
                   f"{fs['full_g']['primary']['all_device']:.4f} / {fs['full_g']['primary']['combined']:.4f}, "
                   f"LODO {fs['lodo_g']['primary']['all_device']:.4f} / {fs['lodo_g']['primary']['combined']:.4f} "
                   f"(all-device / combined); GMR(card + g, card) full data {ci(g['full_g_vs_C0_refit'])}; "
                   f"GMR(card + g, paper) {ci(g['full_g_vs_paper'])}; GMR(card, paper) {ci(g['C0_refit_vs_paper'])}.")
    r = d["dec"]["reference"]
    out.append(f"references: paper card {r['paper']['primary']['all_device']:.4f} / "
               f"{r['paper']['primary']['combined']:.4f}; multipliers 1 no fit "
               f"{r['mult1_no_fit']['primary']['all_device']:.4f} / {r['mult1_no_fit']['primary']['combined']:.4f}; "
               f"stock tt card at 77 K {r['stock_77K']['primary']['all_device']:.4f} / "
               f"{r['stock_77K']['primary']['combined']:.4f}.")
    return out


def deck(batch: Batch, d: dict, iv_files: dict, path: Path):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    def slide(title, img=None, text=None):
        s = prs.slides.add_slide(blank)
        tb = s.shapes.add_textbox(Inches(0.4), Inches(0.2), Inches(12.5), Inches(0.7))
        tb.text_frame.text = title
        tb.text_frame.paragraphs[0].font.size = Pt(24)
        tb.text_frame.paragraphs[0].font.bold = True
        if text:
            t2 = s.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12.3), Inches(6))
            t2.text_frame.word_wrap = True
            for k, line in enumerate(text):
                p = t2.text_frame.paragraphs[0] if k == 0 else t2.text_frame.add_paragraph()
                p.text = line
                p.font.size = Pt(16)
        if img:
            from PIL import Image
            w, h = Image.open(img).size
            maxw, maxh = 12.5, 6.3
            sc = min(maxw / (w / 100), maxh / (h / 100))
            s.shapes.add_picture(str(img), Inches(0.4), Inches(1.0), Inches(w / 100 * sc), Inches(h / 100 * sc))
    dec = d["dec"]
    nb = len(d["specs"]["selected"])
    slide("I25-1: the paper's 77 K fit, redone with our method from random starts", text=[
        "Task = the paper's: the same 18 measured devices at 77 K, the paper's 77 K card files (its bins, tnom "
        "-196.15 C, temperature coefficients 0), the paper's RRMS and curve set.",
        "Every tuned value of the paper removed: all released multipliers set to 1 (named .param multipliers "
        "included).",
        "Fit: multi-round direct NGSpice fit. Round 1 from a random point in the box (3 random starts per "
        "device); each round: CMA-ES then trust-region least squares on NGSpice residuals; rounds 2-3 re-center "
        "the box on the previous card; finite-difference polish after round 3. Per device the start with the "
        "lowest training RRMS.",
        f"Arm A: the paper's 7 parameters (VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0). Arm B: {nb} parameters "
        "from the 115-candidate screen around the multiplier-1 card.",
        "g: learned current correction I = I_card x exp(g(V_GS, V_DS, polarity)), trained on all 18 fitted "
        "devices; each card refit with g in the loop.",
        "RRMS: mean over scored curves of RMS(I_sim - I_meas) / mean |I_meas| (trimmed); all-device = mean over "
        "18 devices; combined = mean of the nMOS and pMOS family means."])
    for key, title in (("table_main_primary_full", "RRMS per card (all 18 devices fitted)"),
                       ("table_stages", "RRMS after every round"), ("fig_stages", "RRMS after every round"),
                       ("table_multistart", "Spread over the 3 random starts"),
                       ("fig_per_device", "Per-device RRMS"),
                       ("table_per_device_armA", "Per-device RRMS, 7 parameters"),
                       ("table_per_device_armB", f"Per-device RRMS, {nb} parameters"),
                       ("table_parameter_set", f"Parameter set of arm B ({nb})")):
        for f in batch.files:
            if f.endswith(key + ".png"):
                slide(title, batch.out / f)
    for a in "AB":
        for t in sorted(iv_files[a], key=rf.device_key):
            slide(f"I-V, {devlab(t)}, {ARM_NAME[a]}", iv_files[a][t])
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    ap.add_argument("--no-iv", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="read I25_1/smoke (code check)")
    a = ap.parse_args()
    global I25
    if a.smoke:
        I25 = I25 / "smoke"
        a.date = "smoke"
    out = ROOT / "out/research/report" / f"{a.date}_i25_1"
    (out / "iv").mkdir(parents=True, exist_ok=True)
    d = load_all()
    b = Batch(out)
    b.table(t_main(d, "primary"))
    b.table(t_main(d, "primary", lodo=False))
    b.table(t_main(d, "m_rule"))
    b.table(t_stages(d))
    b.table(t_spread(d))
    b.fig("fig_stages", fig_stages(d))
    b.table(t_gmr(d))
    b.fig("fig_forest", fig_forest(d))
    b.fig("fig_per_device", fig_per_device(d))
    b.table(t_per_device(d, "A"))
    b.table(t_per_device(d, "B"))
    b.table(t_params(d))
    iv = {"A": {}, "B": {}}
    if not a.no_iv:
        for arm in "AB":
            for t in sorted(d["recs"][arm], key=rf.device_key):
                fn = out / "iv" / f"iv_arm{arm}_{t}.png"
                f = fig_iv(d, arm, t)
                f.savefig(fn, dpi=110, facecolor="white")
                plt.close(f)
                iv[arm][t] = fn
    (out / "tables.md").write_text("\n".join(b.tables))
    src = "out/research/explore/I25_1/decision.json, I25_1/arm{A,B}/final/final.json, I25_1/old/old.json"
    desc = {"table_main_primary": "Main table, primary curve set: stock card at 77 K, multiplier-1 card, paper "
                                  "card, 7p and screened-set card fits without and with g, full data and LODO, "
                                  "NGSpice calls per device",
            "table_main_primary_full": "Main table, primary curve set, full data only (slide version)",
            "table_main_m_rule": "Main table, secondary M-rule curve set",
            "table_stages": "RRMS after every round of the selected start, 7p and screened-set chains vs the old "
                            "43p chain, then C0_refit and g",
            "table_multistart": "Spread over the 3 random starts per device (min / median / max)",
            "fig_stages": "RRMS after every round (log scale)",
            "table_gmr": "GMR with 95% device-bootstrap CI and pre-registered labels, both curve sets",
            "fig_forest": "Forest plot of the primary GMRs",
            "fig_per_device": "Per-device RRMS: multiplier-1 card, paper, card fit, card fit + g (full data), both arms",
            "table_per_device_armA": "Per-device RRMS, arm A (7 parameters)",
            "table_per_device_armB": "Per-device RRMS, arm B (screened set)",
            "table_parameter_set": "Screened set around the multiplier-1 card vs the earlier 43"}
    lines = []
    for f in b.files:
        key = f.split("_", 1)[1][:-4]
        lines.append(f"- `{f}`: {desc.get(key, key)}; sources: {src}.")
    (out / "README.md").write_text(
        f"# Report batch {a.date} · i25_1\n\nScript `scripts/research/plot_explore_i25_1.py`. Curves are the "
        "NGSpice re-simulations stored by `i25_1_scratch.py --stage final/old` (each reproduces its stored RRMS); "
        "every plotted RRMS recomputed from them and checked against decision.json. I-V figures of all 18 devices "
        "for both arms in `iv/iv_arm{A,B}_<device>.png` (measured points, paper card NGSpice, from-scratch card "
        "fit NGSpice, from-scratch card fit + g NGSpice x exp(g); per-curve RRMS in the panel titles). Deck "
        "`slides/i25_1_from_scratch.pptx` (not in git).\n\n## Contents\n\n" + "\n".join(lines) + "\n")
    if not a.smoke:
        rf.update_index(a.date, "i25_1", out, list(b.files), summary_lines(d))
    deck(b, d, iv, ROOT / "slides" / ("i25_1_smoke.pptx" if a.smoke else "i25_1_from_scratch.pptx"))
    print(json.dumps({"out": str(out), "files": b.files, "n_iv": sum(len(v) for v in iv.values())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
