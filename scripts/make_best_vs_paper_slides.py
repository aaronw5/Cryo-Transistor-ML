#!/usr/bin/env python3
"""Deck: the current draft method (starting card -> held-out card refit with the
selected bias-only correction g v2 in the loop) against the published 77 K
card, sky130 77 K, 18 measured devices, leave-one-device-out, BSIM4 state C.

Stages (run in this order; each later stage reads the files of the earlier ones):

  --resim   NGSpice re-simulation, 18 devices, no fitting and no training:
            published card (upstream committed sweeps), C0_refit held-out card
            (I13-1), selected g v2 held-out card + fold OSDI module in the
            element deck (I17-1 modules), and the same card x exp(g) in Python.
            Every RRMS is asserted against the stored rows (|d| < 1e-9; published
            card < 1e-8, the tolerance of plot_explore_i3_2.published_rrms).
            Output: out/research/explore/best_vs_paper/{curves.npz, check.json}
  --plots   figures and equation images -> slides/best_vs_paper_plots/
  (always)  deck -> slides/cryo_ml_best_vs_paper.pptx

Stored inputs (numbers on the slides are read from these files or recomputed
from their per-device rows; nothing is typed in by hand except labels):
  out/research/reference_scores.json                     published card rows
  out/research/explore/I13_1/{summary.json, lodo/results.json, all18/results.json}
  out/research/explore/I15_2/decision.json               selected g v2 rows, lambda*
  out/research/explore/I15_1/decision.json               v1 element deployment label
  out/research/explore/I17_1/{decision.json, accuracy/accuracy.json}
  out/research/explore/I22_1/{decision.json, cost.json}
  out/research/explore/I23_1/decision.json
  out/research/explore/I20_1/decision.json, I17_2/decision.json,
      I21_1/ihp_{233,343,398}K/decision.json, I18_2/decision.json
  out/research/explore/mbias_deck/method_numbers.json    starting-card stages, refit counts
  slides/mbias_plots/eq_rrms.png, eq_dev.png              metric equations (reused)

  PYTHONPATH=src .venv/bin/python scripts/make_best_vs_paper_slides.py --resim --plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EXP = ROOT / "out/research/explore"
OUTD = EXP / "best_vs_paper"
PLOTS = ROOT / "slides/best_vs_paper_plots"
MB_PLOTS = ROOT / "slides/mbias_plots"
DECK = ROOT / "slides/cryo_ml_best_vs_paper.pptx"
REF = ROOT / "out/research/reference_scores.json"
TOL = {"published": 1e-8, "C0_refit": 1e-9, "sel_native": 1e-9, "sel_py": 1e-9}


def jload(p):
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------------------
# stage 1: NGSpice re-simulation
# ---------------------------------------------------------------------------
def resim() -> None:
    sys.path.insert(0, str(ROOT / "scripts/research"))
    sys.path.insert(0, str(ROOT / "scripts/research/explore"))
    import i17_1_vds0 as drv17
    from cryoml.explore import i14_1_chan as m
    from cryoml.explore.e1_correction import score_device
    from plot_explore_i3_2 import published_rrms
    drv = drv17.drv14
    recs = drv.load_records()
    acc = jload(EXP / "I17_1/accuracy/accuracy.json")["rows"]
    c0_rows = jload(EXP / "I13_1/lodo/results.json")["per_device"]["C0_refit"]
    held_c0 = jload(EXP / "I13_1/all18/heldout_cards.json")["C0_refit"]
    tags = sorted(acc)
    assert len(tags) == 18
    pub, pub_curves = published_rrms({t: recs[t] for t in tags})
    arrays, check = {}, {"tags": tags, "devices": {}}
    for tag in tags:
        rec = recs[tag]
        pm = drv.devices()[tag].dev_type == "pmos"
        c0 = drv.measured_run(tag, dict(held_c0[tag]), "card")
        corr = drv.corr_fold("sel", tag)
        card = drv.heldout_card("sel", tag)
        el = drv.measured_run(tag, card, "elem", drv.fold_osdi(EXP / "I17_1/accuracy", "sel", tag),
                              m.module_name(corr))
        cr = drv.measured_run(tag, card, "card")
        py = [np.asarray(ic) * np.exp(corr.g_v(pm, c.vg, c.vd)) for ic, c in zip(cr["sims"], rec.curves)]
        sims = {"published": pub_curves[tag], "C0_refit": c0["sims"], "sel_native": el["sims"],
                "sel_py": py}
        sc = {k: float(score_device(rec, v)["rrms"]) for k, v in sims.items()}
        want = {"published": jload(REF)["released"]["devices"][tag]["rrms"], "C0_refit": c0_rows[tag],
                "sel_native": acc[tag]["rrms_native"], "sel_py": acc[tag]["stored_python"]}
        diff = {k: abs(sc[k] - want[k]) for k in sc}
        for k, d in diff.items():
            assert d < TOL[k], (tag, k, sc[k], want[k])
        check["devices"][tag] = {"scores": sc, "stored": want, "abs_diff": diff,
                                 "msgs": {"C0_refit": c0["msgs"], "sel_native": el["msgs"],
                                          "sel_card": cr["msgs"]},
                                 "lambda_star": acc[tag]["lambda_star"]}
        for i, c in enumerate(rec.curves):
            p = f"{tag}|{i}|"
            arrays[p + "vg"] = np.asarray(c.vg, float)
            arrays[p + "vd"] = np.asarray(c.vd, float)
            arrays[p + "meas"] = np.asarray(c.target, float)
            for k, v in sims.items():
                arrays[p + k] = np.asarray(v[i], float)
            check["devices"][tag].setdefault("curves", []).append(
                {"key": c.key, "kind": c.kind, "start": int(c.start), "included": bool(c.included)})
        print(f"{tag}: " + ", ".join(f"{k} {v:.5f}" for k, v in sc.items()), flush=True)
    check["max_abs_diff"] = {k: max(check["devices"][t]["abs_diff"][k] for t in tags)
                             for k in ("published", "C0_refit", "sel_native", "sel_py")}
    check["n_msgs"] = sum(bool(v) for t in tags for v in check["devices"][t]["msgs"].values())
    check["simulator"] = "ngspice-41 OSDI build (~/cryo-ng41/arm64-osdi), BSIM4 state C"
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTD / "curves.npz", **arrays)
    (OUTD / "check.json").write_text(json.dumps(check, indent=1))
    print("max |re-simulated - stored|:", check["max_abs_diff"], "messages:", check["n_msgs"])


# ---------------------------------------------------------------------------
# numbers (every value read from a stored file or recomputed from stored rows)
# ---------------------------------------------------------------------------
def agg(d: dict) -> dict:
    v = np.array([d[t] for t in sorted(d)], float)
    n = [d[t] for t in d if t.startswith("nmos")]
    p = [d[t] for t in d if t.startswith("pmos")]
    return {"all": float(v.mean()), "combined": float((np.mean(n) + np.mean(p)) / 2),
            "nmos": float(np.mean(n)), "pmos": float(np.mean(p)), "median": float(np.median(v))}


def close(a: float, b: float, tol: float = 1e-12) -> None:
    assert abs(a - b) <= tol, (a, b)


def numbers() -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    N: dict = {}
    ref = jload(REF)
    N["ref"] = ref
    rows = {
        "published": {t: float(v["rrms"]) for t, v in ref["released"]["devices"].items()},
        "paper_seven": {t: float(v["rrms"]) for t, v in ref["paper_seven"]["devices"].items()},
        "paper_full43": {t: float(v["rrms"]) for t, v in ref["paper_full43"]["devices"].items()},
    }
    s13 = jload(EXP / "I13_1/summary.json")
    assert s13["state"] == "C"
    rows["start"] = {t: float(v["new"]) for t, v in s13["per_device"]["start"].items()}
    lodo = jload(EXP / "I13_1/lodo/results.json")
    a18 = jload(EXP / "I13_1/all18/results.json")
    rows["C0_refit"] = dict(lodo["per_device"]["C0_refit"])
    rows["T1_inloop"] = dict(a18["per_device"]["T1_inloop"])
    rows["M_bias_v1"] = dict(a18["per_device"]["M_bias"])
    d152 = jload(EXP / "I15_2/decision.json")
    rows["sel"] = dict(d152["selected_rows"])
    acc = jload(EXP / "I17_1/accuracy/accuracy.json")
    rows["sel_native"] = dict(acc["native"])
    N["rows"] = rows
    N["agg"] = {k: agg(v) for k, v in rows.items()}
    vt = d152["variant_table"]
    # stored aggregates agree with the recomputed ones
    close(N["agg"]["published"]["all"], ref["released"]["all_device_mean_rrms"])
    close(N["agg"]["published"]["combined"], ref["released"]["combined_rrms"])
    close(N["agg"]["start"]["all"], s13["table"]["start"]["new"]["all_device_mean"])
    close(N["agg"]["start"]["combined"], s13["table"]["start"]["new"]["combined"])
    for k, key in (("C0_refit", "C0_refit"), ("M_bias_v1", "v1"), ("sel", "selected")):
        close(N["agg"][k]["all"], vt[key]["all_device_mean"])
        close(N["agg"][k]["combined"], vt[key]["combined"])
    close(N["agg"]["T1_inloop"]["all"], a18["summary"]["T1_inloop"]["all_device_mean"])
    close(N["agg"]["sel_native"]["all"], acc["summary_native"]["all_device_mean"])
    close(N["agg"]["sel_native"]["combined"], acc["summary_native"]["combined"])
    N["lambda_star"] = d152["lambda_star"]
    N["inner_full"] = d152["inner_scores"]["full"]
    N["vt"] = vt
    N["acc"] = acc
    # paired ratios: stored ones are recomputed and checked; the others are new
    g = {}
    g["sel_vs_C0"] = paired_ratio(rows["sel"], rows["C0_refit"])
    for k in ("ratio", "ci_low", "ci_high", "wins"):
        close(g["sel_vs_C0"][k], d152["gmr_selected_vs_C0_refit"][k])
    g["sel_vs_v1"] = paired_ratio(rows["sel"], rows["M_bias_v1"])
    for k in ("ratio", "ci_low", "ci_high", "wins"):
        close(g["sel_vs_v1"][k], d152["primary_gmr_selected_vs_v1"][k])
    g["native_vs_C0"] = acc["gmr_native"]["sel_native_vs_C0_refit"]
    g["sel_vs_pub"] = paired_ratio(rows["sel"], rows["published"])
    g["sel_vs_start"] = paired_ratio(rows["sel"], rows["start"])
    g["C0_vs_pub"] = paired_ratio(rows["C0_refit"], rows["published"])
    g["start_vs_pub"] = paired_ratio(rows["start"], rows["published"])
    g["native_vs_sel"] = paired_ratio(rows["sel_native"], rows["sel"])
    N["gmr"] = g
    N["check"] = jload(OUTD / "check.json")
    N["mn"] = jload(EXP / "mbias_deck/method_numbers.json")
    N["cost"] = jload(EXP / "I22_1/cost.json")["datasets"]
    N["i22"] = jload(EXP / "I22_1/decision.json")
    N["i23"] = jload(EXP / "I23_1/decision.json")
    N["i17"] = jload(EXP / "I17_1/decision.json")
    N["i15_1"] = jload(EXP / "I15_1/decision.json")
    N["restr"] = jload(EXP / "I17_1/deploy_sel/restrictions.json")
    N["n_sim"] = {"C0_refit": list(jload(EXP / "I22_1/partA_sky130/results.json")["n_sim"]["C0_refit"].values())}
    close(float(np.mean(N["n_sim"]["C0_refit"])),
          jload(EXP / "I22_1/cost.json")["datasets"]["sky130"]["arms"]["C0_refit"]["refit_calls_mean"])
    N["n_sim"]["T1_inloop"] = [r["n_sim"] for r in a18["heldout_rows"]["T1_inloop"].values()]
    N["n_sim"]["M_bias_v1"] = [r["n_sim"] for r in a18["heldout_rows"]["M_bias"].values()]
    return N


# ---------------------------------------------------------------------------
# stage 2: figures and equation images
# ---------------------------------------------------------------------------
C_MEAS, C_PUB, C_C0, C_SEL = "#111111", "#8a8a8a", "#2a6fdb", "#d1495b"


def pretty(tag: str) -> str:
    fam, l, w = tag.split("_")
    return f"{tag}  (L {l[1:].replace('p', '.')} um, W {w[1:].replace('p', '.')} um)"


def curves_of(arr, check, tag):
    out = []
    for i, c in enumerate(check["devices"][tag]["curves"]):
        p = f"{tag}|{i}|"
        out.append({**c, **{k: arr[p + k] for k in ("vg", "vd", "meas", "published", "C0_refit",
                                                     "sel_native")}})
    return out


def fig_iv_page(tags, arr, check, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(tags), figsize=(13.0, 6.05), dpi=160)
    for j, tag in enumerate(tags):
        sc = check["devices"][tag]["scores"]
        for r, kind in enumerate(("idvd", "idvg")):
            ax = axes[r, j]
            for c in curves_of(arr, check, tag):
                if c["kind"] != kind:
                    continue
                x = np.abs(c["vd"] if kind == "idvd" else c["vg"])
                al = 1.0 if c["included"] else 0.3
                meas = np.abs(c["meas"])
                s = slice(c["start"], None)
                ax.scatter(x[s][::2], meas[s][::2], s=5, color=C_MEAS, lw=0, alpha=al, zorder=4)
                ax.plot(x, np.abs(c["published"]), color=C_PUB, ls="--", lw=1.1, alpha=al, zorder=2)
                ax.plot(x, np.abs(c["C0_refit"]), color=C_C0, lw=0.9, alpha=al, zorder=3)
                ax.plot(x, np.abs(c["sel_native"]), color=C_SEL, lw=1.3, alpha=al, zorder=3)
            if kind == "idvg":
                ax.set_yscale("log")
                lo = min(np.min(np.abs(c["meas"])[np.abs(c["meas"]) > 0])
                         for c in curves_of(arr, check, tag) if c["kind"] == "idvg")
                ax.set_ylim(bottom=max(lo / 3, 1e-13))
                ax.set_xlabel("|Vgs| (V)", fontsize=9)
            else:
                ax.set_xlabel("|Vds| (V)", fontsize=9)
                ax.set_title(f"{pretty(tag)}\nRRMS: published {sc['published']:.4f}, C0_refit "
                             f"{sc['C0_refit']:.4f}, method {sc['sel_native']:.4f}", fontsize=9)
            ax.set_ylabel("|Id| (A)", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25, lw=0.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C_MEAS, ms=3.5, label="measured 77 K (scored points)"),
         Line2D([], [], color=C_PUB, ls="--", lw=1.2, label="NGSpice, published 77 K card"),
         Line2D([], [], color=C_C0, lw=1.0, label="NGSpice, C0_refit held-out card"),
         Line2D([], [], color=C_SEL, lw=1.4, label="NGSpice element deck, method (g v2 held-out card + fold module)")]
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.6, w_pad=1.0)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def fig_paired(N, path):
    import matplotlib.pyplot as plt
    g = N["gmr"]
    rowsf = [("g v2 / published card", g["sel_vs_pub"]),
             ("g v2 / starting card", g["sel_vs_start"]),
             ("g v2 / C0_refit", g["sel_vs_C0"]),
             ("g v2 element deck / C0_refit", g["native_vs_C0"]),
             ("g v2 / M_bias v1", g["sel_vs_v1"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=160, gridspec_kw={"width_ratios": [1, 1.25]})
    for i, (lab, v) in enumerate(rowsf):
        y = len(rowsf) - 1 - i
        a1.plot([v["ci_low"], v["ci_high"]], [y, y], color=C_SEL, lw=2)
        a1.plot(v["ratio"], y, "o", color=C_SEL, ms=6)
        a1.annotate(f"{v['ratio']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]  {v['wins']}/{v['n_used']} lower",
                    (v["ci_high"], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    a1.axvline(1, color="#555", lw=0.8, ls=":")
    a1.set_yticks(range(len(rowsf)))
    a1.set_yticklabels([r[0] for r in rowsf][::-1], fontsize=9.5)
    a1.set_xscale("log")
    a1.set_xlim(0.15, 2.5)
    a1.set_xticks([0.2, 0.3, 0.5, 0.7, 1, 1.5])
    a1.set_xticklabels(["0.2", "0.3", "0.5", "0.7", "1", "1.5"], fontsize=9)
    a1.set_xlabel("GMR (log scale), 95% CI", fontsize=9.5)
    a1.set_title("Paired GMR over 18 devices", fontsize=10)
    R = N["rows"]
    tags = sorted(R["sel"])
    y = np.arange(len(tags))[::-1]
    a2.scatter([R["published"][t] for t in tags], y, marker="s", color=C_PUB, s=22, label="published card")
    a2.scatter([R["C0_refit"][t] for t in tags], y, marker="o", color=C_C0, s=22, label="C0_refit")
    a2.scatter([R["sel"][t] for t in tags], y, marker="D", color=C_SEL, s=22, label="g v2 (method)")
    a2.set_yticks(y)
    a2.set_yticklabels(tags, fontsize=8.5)
    a2.set_xscale("log")
    a2.set_xlabel("device RRMS (log scale)", fontsize=9.5)
    a2.minorticks_off()
    a2.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    a2.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
    a2.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               title="Per-device RRMS", title_fontsize=10)
    a2.grid(alpha=0.25, lw=0.5)
    a2.tick_params(labelsize=8.5)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


EQS = {
    "score": r"$I_{\mathrm{model}} = I_{\mathrm{card}}\;\exp(g(x))$   (training, refit, scoring)",
    "elem": r"$I(d,s) = I_{\mathrm{card}} + (\exp(g) - 1)\, I_{\mathrm{ch}}$   (deployed element)",
    "in1": r"$s = +1$ (nMOS), $-1$ (pMOS);   $a = s\,V_{gs}$,   $b = s\,V_{gd}$,   "
           r"$m = \sqrt{V_{ds}^2 + \mathrm{EPS}^2}$",
    "in2": r"$x_1 = \dfrac{a + b + m}{2\,\mathrm{VDD}}$,   $x_2 = \dfrac{m}{\mathrm{VDD}}$,   "
           r"$x = (\mathrm{is\_pmos},\, x_1,\, x_2)$",
    "g": r"$g(x) = \frac{1}{5}\sum_{j=1}^{5} \tanh(h_j(z))$,   $z$: $x_1, x_2$ standardized "
         r"(training mean, std)",
    "loss": r"$L = \frac{1}{D}\sum_{d=1}^{D} \mathrm{RRMS}_d(I_{\mathrm{card}}\, e^{g}) + "
            r"10^{-4}\,\|W\|^2 + \lambda\, P$",
    "pen": r"$P = \frac{1}{D}\sum_{d}\, \mathrm{mean}\,[\mathrm{relu}(-r_{gm}) + \mathrm{relu}(-r_{gds})]$,   "
           r"$r_{gm} = \frac{g_m}{I} + \frac{\partial g}{\partial u_g}$,   "
           r"$r_{gds} = \frac{g_{ds}}{I} + \frac{\partial g}{\partial u_d}$",
    "gmr": r"$\mathrm{GMR}(A, B) = \exp\left(\frac{1}{n}\sum_{d=1}^{n} "
           r"\ln \frac{\mathrm{RRMS}_{A,d}}{\mathrm{RRMS}_{B,d}}\right)$",
}


def render_eqs() -> dict:
    import matplotlib.pyplot as plt
    sizes = {}
    for name, tex in EQS.items():
        fig = plt.figure(figsize=(0.1, 0.1), dpi=300)
        fig.text(0, 0, tex, fontsize=20, color="#202428")
        p = PLOTS / f"eq_{name}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=True)
        plt.close(fig)
        from PIL import Image
        with Image.open(p) as im:
            sizes[name] = [im.width / 300, im.height / 300]
    (PLOTS / "eq_sizes.json").write_text(json.dumps(sizes, indent=1))
    return sizes


def plots(N) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    arr = np.load(OUTD / "curves.npz")
    check = N["check"]
    tags = check["tags"]
    for k in range(6):
        fig_iv_page(tags[3 * k:3 * k + 3], arr, check, PLOTS / f"iv_page{k + 1}.png")
    fig_paired(N, PLOTS / "paired.png")
    render_eqs()


# ---------------------------------------------------------------------------
# stage 3: deck
# ---------------------------------------------------------------------------
def build(N) -> None:
    import make_mbias_slides as mb
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Inches, Pt

    new_slide, add_lines, add_table, add_picture = mb.new_slide, mb.add_lines, mb.add_table, mb.add_picture
    add_box, arrow = mb.add_box, mb.arrow
    eqs = jload(PLOTS / "eq_sizes.json")
    R, A, G = N["rows"], N["agg"], N["gmr"]
    mn = N["mn"]
    cost = N["cost"]["sky130"]
    tags = sorted(R["sel"])

    def f4(x):
        return f"{x:.4f}"

    def ci(v, n=3):
        return f"{v['ratio']:.{n}f} [{v['ci_low']:.{n}f}, {v['ci_high']:.{n}f}]"

    def eq(slide, name, left, top, size=16, src=PLOTS):
        w, h = eqs[name] if src == PLOTS else mn["eq_sizes_in"][name]
        w, h = w * size / 20, h * size / 20
        if left + w > 13.2:
            mb.FIT_WARNINGS.append(f"equation {name}: width {w:.2f} in at left {left:.2f}")
        slide.shapes.add_picture(str(src / f"eq_{name}.png"), Inches(left), Inches(top), Inches(w), Inches(h))
        return w, h

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 title ------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    box = s.shapes.add_textbox(Inches(0.7), Inches(1.9), Inches(11.9), Inches(1.3))
    tf = box.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    run = para.add_run()
    run.text = "SKY130 77 K: card refit with the bias-only correction g v2, compared with the published 77 K card"
    run.font.size, run.font.bold, run.font.color.rgb = Pt(30), True, mb.INK
    b = add_lines(s, [
        f"Published 77 K card: all-device RRMS {f4(A['published']['all'])}, combined {f4(A['published']['combined'])}",
        f"Method (starting card, held-out refit with g v2, LODO): all-device RRMS {f4(A['sel']['all'])}, "
        f"combined {f4(A['sel']['combined'])}",
        f"18 measured devices (9 nMOS, 9 pMOS); ngspice-41; BSIM4 state C; all rows DEV, method draft not frozen; "
        "2026-10-27",
    ], left=0.9, top=3.45, width=11.5, height=1.95, size=19)
    for p_ in b.text_frame.paragraphs:
        p_.alignment = PP_ALIGN.CENTER
    add_lines(s, [
        "RRMS: relative RMS current error of the paper (slide 2). Combined: mean of the nMOS and pMOS family means. "
        "LODO: leave-one-device-out. State C: BSIM4 A2 clamp written into the card (slide 2). "
        "g v2: learned bias-dependent current correction (slide 5). C0_refit: card refit without g (slide 4).",
    ], left=0.9, top=5.4, width=11.5, height=1.3, size=13)
    for p_ in s.shapes[-1].text_frame.paragraphs:
        p_.alignment = PP_ALIGN.CENTER
        for r_ in p_.runs:
            r_.font.color.rgb = mb.MUTED

    # 2 data and metric ---------------------------------------------------------
    s = new_slide(prs, "Data and metric: 18 devices, 11 curves each, strict paper RRMS", size=24)
    nc = mn["records"]["n_curves"]

    def lw(t):
        _, l, w = t.split("_")
        return f"{l[1:].replace('p', '.')} x {w[1:].replace('p', '.')}"
    nt = [t for t in tags if t.startswith("nmos")]
    pt = [t for t in tags if t.startswith("pmos")]
    rows = [["nMOS L x W (um)", "scored curves", "pMOS L x W (um)", "scored curves"]]
    for a_, b_ in zip(nt, pt):
        rows.append([lw(a_), str(nc[a_]), lw(b_), str(nc[b_])])
    add_table(s, rows, [1.6, 1.15, 1.6, 1.15], left=0.45, top=1.05, size=12, row_h=0.33, align_right_from=9)
    keys = [c["key"] for c in N["check"]["devices"][tags[0]]["curves"]]
    vds = ", ".join(k.split("@")[1] for k in keys if k.startswith("idvg"))
    vgs = ", ".join(k.split("@")[1] for k in keys if k.startswith("idvd"))
    ref = N["ref"]["released"]
    add_lines(s, [
        "Devices: SkyWater SKY130 nfet_01v8_lvt and pfet_01v8_lvt, measured at 77 K (data of the paper).",
        f"Curves per device: Id-Vg at |Vds| = {vds} V; Id-Vd at |Vgs| = {vgs} V.",
        f"Scored curves: frozen to the published card's inclusion set ({min(nc.values())}-{max(nc.values())} "
        f"per device); {mn['records']['total']:,} scored points in total.",
        "All-device RRMS: mean over the 18 devices. Combined RRMS: (nMOS mean + pMOS mean) / 2.",
        f"Published card: all-device {f4(ref['all_device_mean_rrms'])}, combined {f4(ref['combined_rrms'])} "
        f"(nMOS {f4(ref['nmos_rrms'])}, pMOS {f4(ref['pmos_rrms'])}).",
        "Simulator: ngspice-41, arm64 build. State C: a2 > 1 written as a2 = 1, a1 = 0; a2 < 0.01 as 0.01 "
        "(ngspice-41 clamps A2 only in the first analysis per model).",
    ], left=6.2, top=1.0, width=6.9, height=3.9, size=13)
    eq(s, "rrms", 0.5, 4.75, size=15, src=MB_PLOTS)
    eq(s, "dev", 0.5, 5.6, size=15, src=MB_PLOTS)
    add_lines(s, ["n_c: points of curve c after the trim (points below the device current floor at the start of the "
                  "sweep removed); den_c: mean |I_meas| of those points."],
              left=0.45, top=6.5, width=12.4, height=0.8, size=13)

    # 3 pipeline ------------------------------------------------------------------
    s = new_slide(prs, "Pipeline: published card to deployed element (one LODO fold k)", size=24)
    st = mn["stages"]["stages"]
    seven = st[1]["calls"]
    bw, bh = 2.95, 1.55
    add_box(s, f"Published 77 K card\n(paper)\nRRMS {f4(A['published']['all'])} / "
               f"{f4(A['published']['combined'])}", 0.35, 1.2, bw, bh, size=12, fill=RGBColor(236, 236, 236))
    add_box(s, f"Starting card, device k\n43 parameters, direct NGSpice fit\nCMA-ES + TRF, 3 rounds + FD\n"
               f"{cost['starting_cards']['per_device_mean']:,.0f} calls / device\nRRMS {f4(A['start']['all'])} / "
               f"{f4(A['start']['combined'])}", 3.65, 1.2, bw, bh + 0.35, size=12)
    add_box(s, f"C0_refit (control)\nTRF refit of device k, g = 0\nmax_nfev 400\nRRMS {f4(A['C0_refit']['all'])} / "
               f"{f4(A['C0_refit']['combined'])}", 7.0, 0.95, bw, bh, size=12)
    add_box(s, f"Method: refit with g v2\nTRF refit of device k,\nI_card x exp(g) in residuals, g frozen\n"
               f"RRMS {f4(A['sel']['all'])} / {f4(A['sel']['combined'])}", 7.0, 2.85, bw, bh + 0.2, size=12,
            fill=RGBColor(252, 236, 226))
    add_box(s, f"Deployed element\ncard + channel replica\n+ OSDI module (Verilog-A)\nelement deck RRMS\n"
               f"{f4(A['sel_native']['all'])} / {f4(A['sel_native']['combined'])}", 10.3, 2.85, 2.75, bh + 0.2,
            size=12, fill=RGBColor(252, 236, 226))
    add_box(s, "g v2, fold k\ntrained on the starting-card records\nof the 17 other devices;\n"
               "penalty weight lambda by nested\ninner CV (slide 6)", 3.65, 3.55, bw, bh, size=12, fill=RGBColor(252, 236, 226))
    arrow(s, 3.33, 1.2 + bh / 2, 3.62, 1.2 + bh / 2)
    arrow(s, 6.63, 1.75, 6.97, 1.7)
    arrow(s, 6.63, 2.4, 6.97, 3.3)
    arrow(s, 6.63, 3.55 + bh / 2, 6.97, 3.75)
    arrow(s, 9.98, 2.85 + (bh + 0.2) / 2, 10.27, 2.85 + (bh + 0.2) / 2)
    add_lines(s, [
        "RRMS values: all-device / combined, held-out device k, 18 folds. Starting cards were fitted per device "
        "before the LODO folds (device k's own measured curves).",
        f"Starting-card calls: rounds 1-3 and FD polish ({cost['starting_cards']['per_device_mean']:,.0f}, "
        f"I22-1 cost.json); the 7-parameter stage that initializes round 1 adds {seven:,.0f} calls / device.",
        "TRF: scipy least_squares(method='trf'), bounded local coordinates. FD: forward-difference Jacobian.",
    ], left=0.45, top=5.45, width=12.4, height=1.9, size=13)

    # 4 starting cards and C0_refit -------------------------------------------------
    s = new_slide(prs, "Starting cards and C0_refit: stages, budget, RRMS", size=24)
    rows = [["Stage", "Parameters", "Optimizer", "NGSpice calls / device", "All-device", "Combined"]]
    rows.append(["Published 77 K card", "-", "-", "-", f4(A["published"]["all"]), f4(A["published"]["combined"])])
    for r_ in st[1:]:
        o = r_["options"]
        if "cmaes_evals" in o:
            opt = f"CMA-ES {o['cmaes_evals']} evals, then TRF max_nfev {o['polish_nfev']}"
        else:
            opt = f"TRF max_nfev {o['max_nfev']}, IRLS {o['irls']}, diff_step {o['diff_step']}"
        lab = {"round 3 + FD": "round 3 + FD = starting card"}.get(r_["label"], r_["label"])
        rows.append([lab, "7" if r_["label"].startswith("7") else "43", opt, f"{r_['calls']:,.0f}",
                     f4(r_["all_device"]), f4(r_["combined"])])
    ns = N["n_sim"]["C0_refit"]
    rows.append(["C0_refit (held-out, LODO)", f"{mn['box']['n_active_min']}-{mn['box']['n_active_max']} active of 43",
                 "TRF max_nfev 400, IRLS 2, diff_step 0.01", f"{np.mean(ns):.1f} ({min(ns)}-{max(ns)})",
                 f4(A["C0_refit"]["all"]), f4(A["C0_refit"]["combined"])])
    rows.append(["Method refit (g v2 in loop)", f"{mn['box']['n_active_min']}-{mn['box']['n_active_max']} active of 43",
                 "same as C0_refit", f"{cost['arms']['selected_g_v2']['refit_calls_mean']:.1f}",
                 f4(A["sel"]["all"]), f4(A["sel"]["combined"])])
    add_table(s, rows, [3.0, 1.85, 3.9, 1.65, 1.1, 1.1], left=0.4, top=1.0, size=12, row_h=0.36,
              align_right_from=3)
    grp = mn["screen"]["groups"]
    allp = [p_ for v in grp.values() for p_ in v]
    add_lines(s, [
        f"43 parameters ({mn['screen']['n_candidates']} candidates, one-at-a-time sensitivity screen): "
        + ", ".join(allp) + ".",
        "Box: local coordinates u in [-1, 1]; parameters with value 0 or below the box activity test are held "
        f"fixed, so {mn['box']['n_active_min']}-{mn['box']['n_active_max']} are active per device.",
        "Starting-card stages are CMA-ES (sigma0 0.3, seed 20261015) then TRF on the NGSpice RRMS objective; "
        "the round-3 + FD card is the starting card of every later arm.",
    ], left=0.45, top=4.4, width=12.4, height=2.9, size=12.5)

    # 5 g v2 definition --------------------------------------------------------------
    s = new_slide(prs, "Correction g v2: inputs, network, loss", size=24)
    y = 0.95
    for name, size_ in (("score", 15), ("elem", 15), ("in1", 15), ("in2", 15), ("g", 15), ("loss", 15),
                        ("pen", 15)):
        _, h = eq(s, name, 0.45, y, size=size_)
        y += h + 0.12
    rows = [["Item", "Value"],
            ["I_ch", "channel current of a replica of the card instance"],
            ["EPS", "0.01 V (smooth |Vds|)"],
            ["VDD (input scale)", "1.85 V"],
            ["s, a, b, m", "polarity sign; s Vgs; s Vgd; smooth |Vds|"],
            ["x1, x2", "unchanged when drain and source are exchanged"],
            ["h_j network", "3-16-16-1, tanh, last layer zero-initialized"],
            ["ensemble", "5 members, seeds 0-4; g = member mean"],
            ["optimizer", "Adam, lr 3e-3, 2000 full-batch steps"],
            ["training data", "starting-card records of the 17 training devices (cards fixed)"],
            ["P grid", "25 mV, 75 x 75, 0-1.85 V, forward quadrant, I_card > 1e-9 A"],
            ["u_g, u_d", "s Vgs, s Vds; gm, gds of the card by central differences"],
            ["lambda", "penalty weight (V); grid {0, 1, 2, 5, 10} V, slide 6"]]
    rows += [["D, W", "number of training devices; weight matrices"],
             ["P", "mean hinge on negative gm / I and gds / I of I_card exp(g) (1/V)"]]
    add_table(s, rows, [1.35, 3.75], left=8.0, top=0.95, size=10.5, row_h=0.33, align_right_from=9)

    # 6 lambda selection -------------------------------------------------------------
    s = new_slide(prs, "Penalty weight lambda: nested inner cross-validation per fold", size=24)
    lam = N["lambda_star"]
    grid = [0.0, 1.0, 2.0, 5.0, 10.0]
    cnt = {g_: sum(1 for t in tags if lam[t] == g_) for g_ in grid}
    vt = N["vt"]
    rows = [["lambda (V)"] + [f"{g_:g}" for g_ in grid] + ["selected"]]
    rows.append(["folds with lambda* = value"] + [str(cnt[g_]) for g_ in grid] + ["18"])
    rows.append(["full-data inner score (mean ln RRMS)"] + [f"{N['inner_full'][str(int(g_))]:.4f}" for g_ in grid]
                + [f"lambda* {lam['full']:g} V"])
    rows.append(["LODO all-device RRMS, fixed lambda (post hoc)"]
                + [f4(vt[f"sym_e1_l{int(g_)}"]["all_device_mean"]) for g_ in grid] + [f4(vt["selected"]["all_device_mean"])])
    rows.append(["LODO combined RRMS, fixed lambda (post hoc)"]
                + [f4(vt[f"sym_e1_l{int(g_)}"]["combined"]) for g_ in grid] + [f4(vt["selected"]["combined"])])
    add_table(s, rows, [4.3, 1.1, 1.1, 1.1, 1.1, 1.1, 1.55], left=0.45, top=1.0, size=12.5, row_h=0.42)
    add_lines(s, [
        "Per outer fold k: the 17 training devices are sorted (nMOS first, then tag) and split into 3 inner folds "
        "(device i -> inner fold i mod 3).",
        "Per inner fold and lambda: one member (seed 0) trained on the other 2 inner folds; inner held-out devices "
        "refit with g frozen (outer refit settings).",
        "Inner score: mean ln RRMS over the inner held-out devices. lambda* = argmin; ties within 1e-12 go to the "
        "larger lambda. Device k is not used.",
        f"Full-data lambda* = {lam['full']:g} V: the same selection over all 18 devices; it defines the module of the "
        "deployment checks (slide 17), not the LODO rows.",
        "Fixed-lambda rows are reported after the fact; they did not enter the selection.",
    ], top=3.35, height=3.9, size=14)

    # 7 evaluation protocol -----------------------------------------------------------
    s = new_slide(prs, "Evaluation protocol: leave-one-device-out (LODO), 18 folds", size=24)
    rows = [["Stage", "Uses data of held-out device k", "Detail"],
            ["Starting card of k", "yes", "direct NGSpice fit to k's measured curves, before the folds"],
            ["g v2 training (fold k)", "no", "starting-card records of the 17 training devices"],
            ["Input standardization, penalty grid", "no", "training devices only"],
            ["lambda selection (fold k)", "no", "inner CV over the 17 training devices"],
            ["Held-out refit of k", "yes", "k's measured curves, g frozen; max_nfev 400"],
            ["Score of k", "yes", "fresh NGSpice run of the kept card x exp(g); frozen inclusion set"],
            ["Full-data lambda* and module", "yes", "deployment checks only (slide 17)"],
            ["Dataset gate S3_D (apply g or not)", "yes", "uses all DEV records incl. k (I19-1); sky130: apply"],
            ["Method design (inputs, EPS, penalty, grid)", "yes", "chosen on these 18 devices since I3-2 (DEV)"]]
    add_table(s, rows, [3.7, 2.6, 6.2], left=0.45, top=1.0, size=13, row_h=0.42, align_right_from=9)
    add_lines(s, [
        "Held-out device k contributes: its own starting card, its measured curves to its refit, and its score.",
        "Held-out device k does not contribute: g, its input scaling, its penalty, or the lambda of fold k.",
        "The 6 former REAL_TEST sky130 devices were opened on 2026-10-21 by user decision; no locked sky130 TEST "
        "exists.",
    ], top=5.35, height=1.9, size=14)

    # 8 main results ----------------------------------------------------------------------
    s = new_slide(prs, "Results: all 18 devices, LODO, state C", size=24)
    a18n = N["n_sim"]
    sel_c = cost["arms"]["selected_g_v2"]
    rl = N["i15_1"]["reference_arm_v1_deploy"]
    pap = N["ref"]

    def gs(key):
        return ci(paired_ratio_cached(N, "sel", key))
    rows = [["Card / arm", "Circuit element", "All-device RRMS", "Combined RRMS", "GMR(g v2, row) [95% CI]",
             "NGSpice calls / device"]]
    rows += [
        ["Published 77 K card (paper)", "card", f4(A["published"]["all"]), f4(A["published"]["combined"]),
         gs("published"), "-"],
        ["Paper per-device cards, 7 parameters", "card", f4(pap["paper_seven"]["all_device_mean_rrms"]),
         f4(pap["paper_seven"]["combined_rrms"]), gs("paper_seven"), "-"],
        ["Paper per-device cards, 43 parameters", "card", f4(pap["paper_full43"]["all_device_mean_rrms"]),
         f4(pap["paper_full43"]["combined_rrms"]), gs("paper_full43"), "-"],
        ["Starting card (43 parameters)", "card", f4(A["start"]["all"]), f4(A["start"]["combined"]), gs("start"),
         f"{cost['starting_cards']['per_device_mean']:,.0f}"],
        ["C0_refit (refit, g = 0)", "card", f4(A["C0_refit"]["all"]), f4(A["C0_refit"]["combined"]), gs("C0_refit"),
         f"+{np.mean(a18n['C0_refit']):.1f}"],
        ["T1_inloop (refit, one gain per curve)", "no: defined only at measured biases", f4(A["T1_inloop"]["all"]),
         f4(A["T1_inloop"]["combined"]), gs("T1_inloop"), f"+{np.mean(a18n['T1_inloop']):.1f}"],
        ["M_bias v1 (refit, g v1)", "no: not deployable (I15-1)", f4(A["M_bias_v1"]["all"]),
         f4(A["M_bias_v1"]["combined"]), gs("M_bias_v1"), f"+{np.mean(a18n['M_bias_v1']):.1f}"],
        ["Selected g v2 (method), Python rows", "deployable with restrictions (I17-1)", f4(A["sel"]["all"]),
         f4(A["sel"]["combined"]), "1",
         f"+{sel_c['refit_calls_mean']:.1f} (+{sel_c['total_calls_per_device'] - sel_c['refit_calls_mean']:,.0f} "
         "selection, estimate)"],
        ["Selected g v2, NGSpice element deck", "same element", f4(A["sel_native"]["all"]),
         f4(A["sel_native"]["combined"]), f"{G['native_vs_sel']['ratio']:.4f}", "as above"],
    ]
    add_table(s, rows, [3.35, 2.75, 1.25, 1.25, 2.1, 2.0], left=0.3, top=0.95, size=11.5, row_h=0.42,
              align_right_from=2)
    add_lines(s, [
        f"M_bias v1 has a lower RRMS than selected g v2 ({f4(A['M_bias_v1']['all'])} vs {f4(A['sel']['all'])}; "
        f"GMR(g v2, v1) {ci(G['sel_vs_v1'])}), but v1 is not deployable as a circuit element: "
        + "; ".join(rl["core_fail"]) + " (I15-1 core failures).",
        "GMR(g v2, row): paired geometric-mean ratio of g v2 to the row over the 18 devices (slide 9); 95% percentile "
        "bootstrap CI, 10,000 resamples, seed 0. Below 1: g v2 has the lower RRMS.",
        "Paper per-device cards: reference_scores.json paper_seven / paper_full43, scored with the same metric.",
    ], top=5.3, height=2.0, size=12.5)

    # 9 paired comparison --------------------------------------------------------------
    s = new_slide(prs, "Paired comparison: GMR of g v2 against the published card and controls", size=24)
    add_picture(s, PLOTS / "paired.png", left=0.3, top=0.9, max_w=12.7, max_h=4.9)
    eq(s, "gmr", 0.6, 5.95, size=15)
    gp = G["sel_vs_pub"]
    add_lines(s, [
        f"g v2 vs published card: GMR {ci(gp)}, {gp['wins']}/{gp['n_used']} devices lower (computed here from the "
        "stored rows). g v2 vs C0_refit: stored in I15-2.",
    ], left=6.6, top=5.85, width=6.5, height=1.4, size=13)

    # 10 per-device table ----------------------------------------------------------------
    s = new_slide(prs, "Per-device RRMS: published card, starting card, C0_refit, g v2", size=24)
    rows = [["Device", "Curves", "lambda* (V)", "Published card", "Starting card", "C0_refit", "g v2 (Python)",
             "g v2 (element deck)", "g v2 / C0_refit", "g v2 / published"]]
    for t in tags:
        rows.append([t, str(nc[t]), f"{lam[t]:g}", f4(R["published"][t]), f4(R["start"][t]), f4(R["C0_refit"][t]),
                     f4(R["sel"][t]), f4(R["sel_native"][t]), f"{R['sel'][t] / R['C0_refit'][t]:.3f}",
                     f"{R['sel'][t] / R['published'][t]:.3f}"])
    for lab, key in (("All-device mean", "all"), ("Combined", "combined")):
        rows.append([lab, "", "", f4(A["published"][key]), f4(A["start"][key]), f4(A["C0_refit"][key]),
                     f4(A["sel"][key]), f4(A["sel_native"][key]), "", ""])
    add_table(s, rows, [2.05, 0.75, 1.0, 1.3, 1.25, 1.1, 1.2, 1.35, 1.2, 1.25], left=0.35, top=0.88, size=10.5,
              row_h=0.29, align_right_from=1)

    # 11-16 I-V ---------------------------------------------------------------------------------
    acc = N["acc"]
    for k in range(6):
        grp_t = N["check"]["tags"][3 * k:3 * k + 3]
        s = new_slide(prs, f"I-V, held-out devices {3 * k + 1}-{3 * k + 3} of 18: "
                           f"{', '.join(grp_t)}", size=20)
        add_picture(s, PLOTS / f"iv_page{k + 1}.png", left=0.15, top=0.78, max_w=13.0, max_h=6.05)
        add_lines(s, [
            "All curves are NGSpice runs (ngspice-41 OSDI build, state C), re-simulated by this script. Method: g v2 "
            "held-out card + fold OSDI module in the element deck; RRMS equal to the stored I17-1 element-deck rows "
            f"(difference 0) and within {acc['max_abs_native_minus_stored']:.1e} of the I15-2 Python rows. Faded: "
            "curves outside the scored set. Top: Id-Vd; bottom: Id-Vg, log scale.",
        ], left=0.3, top=6.78, width=12.8, height=0.7, size=10.5)

    # 17 deployment ------------------------------------------------------------------------------
    s = new_slide(prs, "Deployment checks of the g v2 element (I17-1, full-data lambda* 1 V)", size=24)
    v = N["i17"]["values_I17_1_rule"]
    th = N["i17"]["thresholds"]
    pdv = N["restr"]["sel"]["per_device"]
    n_gds = sum(len(pdv[t]["gds"]) for t in pdv)
    n_gm = sum(len(pdv[t].get("gm", [])) for t in pdv)
    rows = [["Check", "Type", "Value", "Threshold", "Result"],
            ["Q1 equivalence, tight tolerance (points violating)", "core", f"{v['q1_tight_viol']}", "0", "pass"],
            ["Q1 convergence messages / non-finite / failed runs", "core", f"{v['q1_bad']}", "0", "pass"],
            ["Q1 |I(Vds = 0)|, tight tolerance (A)", "core", f"{v['q1_vds0_max']:.2e}", f"<= {th['q1_vds0_abs']:.0e}",
             "pass"],
            ["Verilog-A g vs Python g", "core", f"{v['g_text_max']:.1e}", f"<= {th['g_text']:.0e}", "pass"],
            ["Verilog-A g, exchanged vs forward terminals", "core", f"{v['g_sym_text_max']:.1e}",
             f"<= {th['g_sym_text']:.0e}", "pass"],
            ["terminal-exchange asymmetry |d ln I|", "core", f"{v['swap_asym_max']:.1e}", f"<= {th['swap_asym']:.0e}",
             "pass"],
            ["Q2 gm / gds vs element identity (AC, tight)", "core", f"{v['q2_ac_max']:.1e}", f"<= {th['q2_ac']:.0e}",
             "pass"],
            ["Q2 capacitances vs card", "core", f"{v['cap_max']:.1e}", f"<= {th['cap']:.0e}", "pass"],
            ["gm sign change in strong inversion (curves)", "core", f"{v['q2_gm_sign']} of {n_gm}", "0", "pass"],
            ["Q3 circuits failing only with the element", "core", f"{v['q3_fail']}", "0", "pass"],
            ["default-tolerance relative difference, |I| > 1 nA", "restriction", f"{v['q1_default_max']:.2e}",
             f"restriction if > {th['q1_default_rel']:.0e}", "restriction"],
            ["gds sign change introduced (curves)", "restriction", f"{v['q2_gds_sign']} of {n_gds}", "restriction if > 0",
             "restriction"],
            ["runtime ratio element / card (max)", "restriction", f"{v['runtime_ratio_max']:.2f}",
             f"restriction if > {th['runtime_ratio']:g}", "restriction"]]
    add_table(s, rows, [4.9, 1.2, 1.5, 1.6, 1.3], left=0.45, top=0.95, size=11.5, row_h=0.36, align_right_from=2)
    devs = N["restr"]["sel"]["totals"]["devices_with_gds_introduced"]
    add_lines(s, [
        f"Label: {N['i17']['label']}. Q1 grid: Vgs 0-1.85 V x Vds -1.85..1.85 V, 25 mV. Q3: inverters, "
        "5-stage ring oscillators, current mirrors.",
        f"gds sign changes on {', '.join(devs)} (long-channel pMOS); the same points appear in I_card x exp(g), "
        "so they come from g. Max runtime ratio on ro5_A.",
        "The Vds = 0 check was re-registered at tight tolerance after an observed failure at default tolerance "
        f"(I16-1: {N['i17']['values']['q1_vds0_max']:.4e} A, 1 point); I16-1 stays 'not deployable'.",
    ], top=5.98, height=1.5, size=12)

    # 18 cost ------------------------------------------------------------------------------------
    s = new_slide(prs, "Cost: NGSpice calls per held-out device, sky130 (I22-1, I23-1)", size=24)
    ar = cost["arms"]
    fx = N["i23"]["part_B"]["datasets"]["sky130"]
    fxc = fx["cost"]
    rows = [["Arm", "Refit calls / device", "Other calls / device", "Total / device", "x C0_refit", "g trainings",
             "Training CPU s"],
            ["Starting card (once per device)", "-", f"{cost['starting_cards']['per_device_mean']:,.0f}",
             f"{cost['starting_cards']['per_device_mean']:,.0f}", "-", "-", "-"],
            ["C0_refit", f"{ar['C0_refit']['refit_calls_mean']:.1f}", "0", f"{ar['C0_refit']['total_calls_per_device']:.1f}",
             "1.00", "0", "0"],
            ["M_bias v1", f"{ar['M_bias_v1']['refit_calls_mean']:.1f}", "0",
             f"{ar['M_bias_v1']['total_calls_per_device']:.1f}", f"{ar['M_bias_v1']['ratio_to_C0_refit_refit_calls']:.2f}",
             str(ar["M_bias_v1"]["trainings"]), "-"],
            ["Selected g v2 (nested lambda selection)", f"{sel_c['refit_calls_mean']:.1f}",
             f"{sel_c['pre_calls_total'] / cost['n']:,.0f} (estimate)", f"{sel_c['total_calls_per_device']:,.0f}",
             f"{sel_c['ratio_to_C0_refit_refit_calls']:.1f}", str(sel_c["trainings"]), f"{sel_c['train_cpu_s']:,.0f}"],
            ["g v2, fixed lambda 1 V (I23-1)", f"{fxc['fixed_l1_refit_calls_mean']:.1f}", f"{fxc['fixed_l1_pre_calls']}",
             f"{fxc['fixed_l1_refit_calls_mean']:.1f}", f"{fxc['fixed_l1_ratio_to_C0_refit']:.2f}",
             str(fxc["fixed_l1_trainings"]), f"{fxc['train_cpu_s']:,.0f}"]]
    add_table(s, rows, [3.6, 1.55, 2.1, 1.45, 1.1, 1.15, 1.35], left=0.4, top=1.0, size=12, row_h=0.42,
              align_right_from=1)
    fb = N["i23"]["part_B"]
    cmp_ = fx["comparisons"]
    nrt = fb["datasets"]["nistrt"]["comparisons"]["fixed_vs_selected"]
    n4 = fb["datasets"]["nist4"]["comparisons"]["fixed_vs_selected"]
    add_lines(s, [
        f"Selection calls: {sel_c['selection_refits']:,} inner refits over 18 folds, estimated as inner refits x median "
        f"held-out calls (cost.json); "
        f"{(sel_c['total_calls_per_device'] - sel_c['refit_calls_mean']) / sel_c['total_calls_per_device']:.1%} of the "
        "g v2 total. Full-data lambda*: {sel_c['full_data_selection_refits']} "
        f"refits, {sel_c['full_data_selection_calls']:,.0f} calls, once.",
        f"Fixed lambda 1 V vs selected g v2: sky130 GMR {ci(cmp_['fixed_vs_selected'])}; NIST lvt {ci(n4)}; "
        f"NIST RT card {ci(nrt)}.",
        f"I23-1 outcome: '{fb['rule']['label']}' (fails on NIST RT card, where the selection chose 0 V for all 87 "
        "devices); the nested selection stays in the draft method.",
        "One NGSpice call: one simulation of a device's measured bias set.",
    ], top=3.75, height=3.5, size=13.5)

    # 19 baselines ---------------------------------------------------------------------------------
    s = new_slide(prs, "Baselines at matched protocol, sky130 (I22-1)", size=24)
    sk = N["i22"]["part_A"]["datasets"]["sky130"]
    sm, vc, vg = sk["summary"], sk["vs_C0_refit"], sk["vs_g_v2"]
    names = {"B1": f"B1 wider card set (+{len(sk['b1_selected'])} BSIM4 parameters)", "B2": "B2 extrinsic resistance + self-heating",
             "B3_d3": "B3 cubic polynomial g", "B4": "B4 Gaussian-process discrepancy"}
    rows = [["Correction-stage arm", "All-device", "Combined", "GMR vs C0_refit", "GMR(arm, g v2)",
             "Calls x C0_refit"]]
    rows.append(["Selected g v2", f4(sm["selected_g_v2"]["all_device_mean"]), f4(sm["selected_g_v2"]["combined"]),
                 ci(vc["selected_g_v2"]), "1", f"{ar['selected_g_v2']['ratio_to_C0_refit_refit_calls']:.1f}"])
    for k_, lab in names.items():
        rows.append([lab, f4(sm[k_]["all_device_mean"]), f4(sm[k_]["combined"]), ci(vc[k_]), ci(vg[k_]),
                     f"{ar[k_]['ratio_to_C0_refit_refit_calls']:.2f}"])
    add_table(s, rows, [4.0, 1.15, 1.15, 2.4, 2.4, 1.2], left=0.4, top=0.95, size=12, row_h=0.38,
              align_right_from=1)
    pc = N["i22"]["part_C"]
    meth = pc["methods"]
    b_ = list(pc["budget"].values())
    rows = [["Parameter-extraction arm (from the starting card, budget = C0_refit calls)", "GMR(arm, TRF refit)",
             "Label"],
            ["CMA-ES, sigma0 0.1", ci(meth["cma_primary"]["vs_LS"]), meth["cma_primary"]["label"]],
            ["CMA-ES, sigma0 0.03", ci(meth["cma_secondary"]["vs_LS"]), meth["cma_secondary"]["label"]],
            ["TuRBO-1 (BoTorch), L_init 0.1", ci(meth["turbo_primary"]["vs_LS"]), meth["turbo_primary"]["label"]],
            ["TuRBO-1 (BoTorch), L_init 0.03", ci(meth["turbo_secondary"]["vs_LS"]), meth["turbo_secondary"]["label"]]]
    add_table(s, rows, [6.4, 2.5, 1.7], left=0.4, top=3.45, size=12, row_h=0.38, align_right_from=1)
    add_lines(s, [
        f"I22-1 outcome: '{N['i22']['part_A']['outcome']['label']}'. Same protocol for all arms: state C, g v2 folds and "
        "training order, held-out refit max_nfev 400; B1 / B2 matched per TRF iteration.",
        f"Part C budget per device: {min(b_)}-{max(b_)} NGSpice calls (the C0_refit calls of that device); TRF refit = "
        "C0_refit card. GMR > 1: the arm has the higher RRMS.",
    ], top=5.5, height=1.8, size=13)

    # 20 other datasets ------------------------------------------------------------------------------
    s = new_slide(prs, "Other datasets (DEV): same g v2 recipe, GMR vs C0_refit", size=24)
    i20 = jload(EXP / "I20_1/decision.json")["per_dataset"]
    i172 = jload(EXP / "I17_2/decision.json")
    i182 = jload(EXP / "I18_2/decision.json")["per_temperature"]
    ihp = {T: jload(EXP / f"I21_1/ihp_{T}K/decision.json") for T in (233, 343, 398)}
    b1 = N["i22"]["part_A"]["datasets"]

    def tab(t):
        return (f"{f4(t['C0_refit']['all_device_mean'])} / {f4(t['C0_refit']['combined'])}",
                f"{f4(t['selected']['all_device_mean'])} / {f4(t['selected']['combined'])}")

    rows = [["Dataset", "n", "C0_refit RRMS", "g v2 RRMS", "GMR(g v2, C0_refit) [95% CI]", "Statement"]]
    for k_, lab, note in (("nist4", "NIST sky130 4 K, lvt (77 K card)", "gain; element not deployable (I21-1)"),
                          ("nistrt", "NIST sky130 4 K, RT card", "gain; element not deployable (I21-1)")):
        d_ = i20[k_]
        g_ = d_["primary_gmr_selected_vs_C0_refit"]
        rows.append([lab, str(g_["n"]), *tab(d_["table"]), f"{ci(g_)} {g_['wins']}/{g_['n_used']}", note])
    g_ = i172["primary_gmr_selected_vs_C0_refit"]
    rows.append(["IHP SG13G2 300 K", str(g_["n"]), *tab(i172["table"]), f"{ci(g_)} {g_['wins']}/{g_['n_used']}",
                 f"worse; B1 vs C0_refit {b1['ihp300']['vs_C0_refit']['B1']['ratio']:.3f}"])
    for T in (233, 343, 398):
        g_ = ihp[T]["primary_gmr_selected_vs_C0_refit"]
        lab_ = "worse" if g_["ci_low"] > 1 else "inconclusive"
        rows.append([f"IHP SG13G2 {T} K", str(g_["n"]), *tab(ihp[T]["table"]), f"{ci(g_)} {g_['wins']}/{g_['n_used']}",
                     f"{lab_}; B1 vs C0_refit {b1[f'ihp{T}']['vs_C0_refit']['B1']['ratio']:.3f}"])
    for T, lab in (("4", "TU Delft 40 nm, 4.13 K"), ("295", "TU Delft 40 nm, 295.0 K")):
        g_ = i182[T]["primary_gmr_selected_vs_C0_refit"]
        rows.append([lab, f"{g_['n']} (7 groups)", *tab(i182[T]["table"]), f"{ci(g_)} {g_['wins']}/{g_['n_used']}",
                     "inconclusive (cluster CI includes 1)"])
    add_table(s, rows, [3.0, 1.15, 1.65, 1.65, 2.85, 2.35], left=0.35, top=0.95, size=11, row_h=0.44,
              align_right_from=1)
    add_lines(s, [
        "IHP: PSP103 cards, 12-parameter refit; the dataset gate keeps C0_refit at all 4 temperatures; no element "
        "exists for PSP103. B1 (wider PSP103 card set) gains over C0_refit at all 4 temperatures (I22-1, I23-1).",
        "TU Delft: generic CMOSedu BSIM4 example card, not a foundry PDK; CI by 7 geometry groups.",
        "NIST: devices of the NIST sky130 4 K data set; RT card = room-temperature sky130 card. Element checks "
        "(I21-1) fail core checks on both.",
    ], top=5.05, height=2.3, size=12.5)

    # 21 limitations ---------------------------------------------------------------------------------------
    s = new_slide(prs, "Limitations", size=24)
    fb_rt = nrt
    i23a = N["i23"]["part_A"]
    add_lines(s, ["- " + x for x in [
        "No locked TEST result: every number in this deck is DEV. The method draft is not frozen "
        "(planned about 2026-12-10).",
        "sky130 77 K: method choices (inputs, EPS, penalty, lambda grid) were made on these 18 devices; the 6 "
        "former TEST devices were opened on 2026-10-21.",
        f"IHP SG13G2: no gain; 300 K GMR {ci(i172['primary_gmr_selected_vs_C0_refit'])}; B1 (wider card set) is "
        "better than g v2 at all 4 temperatures; no PSP103 element.",
        "NIST (lvt, RT card): RRMS gain, but the element is not deployable on either dataset (I21-1 core failures).",
        f"sky130 element restrictions: default-tolerance difference {v['q1_default_max']:.2e}, gds sign changes "
        f"{v['q2_gds_sign']} of {n_gds} curves, runtime ratio up to {v['runtime_ratio_max']:.2f}.",
        f"Cost: nested lambda selection is {sel_c['ratio_to_C0_refit_refit_calls']:.0f}x the C0_refit NGSpice calls "
        f"({sel_c['total_calls_per_device']:,.0f} per device); fixed lambda 1 V costs "
        f"{fxc['fixed_l1_ratio_to_C0_refit']:.2f}x but is not shown non-inferior on NIST RT card ({ci(fb_rt)}).",
        f"Starting cards: {cost['starting_cards']['per_device_mean']:,.0f} NGSpice calls per device; no "
        "matched-budget comparison of this stage (NPE and neural-operator baselines not run).",
        "TU Delft: generic example card; result inconclusive with the 7-group cluster CI.",
        f"M_bias v1 has a lower sky130 RRMS ({f4(A['M_bias_v1']['all'])}) than g v2 ({f4(A['sel']['all'])}) but is "
        "not deployable as a circuit element.",
    ]], top=1.0, height=6.3, size=15)

    DECK.parent.mkdir(parents=True, exist_ok=True)
    prs.save(DECK)
    print(f"wrote {DECK} ({len(prs.slides._sldIdLst)} slides)")
    for w in mb.FIT_WARNINGS:
        print("FIT:", w)


_PR_CACHE: dict = {}


def paired_ratio_cached(N, a: str, b: str) -> dict:
    from cryoml.explore.i3_2_bias import paired_ratio
    if (a, b) not in _PR_CACHE:
        _PR_CACHE[(a, b)] = paired_ratio(N["rows"][a], N["rows"][b])
    return _PR_CACHE[(a, b)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resim", action="store_true")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    if a.resim:
        resim()
    N = numbers()
    if a.plots:
        plots(N)
    build(N)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
