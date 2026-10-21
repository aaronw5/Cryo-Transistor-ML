#!/usr/bin/env python3
"""Deck: 43-parameter card refit with the shared bias-only correction M_bias
(I3-2 recipe), sky130 77 K, all 18 measured devices, leave-one-device-out.

Inputs (all stored; nothing is fitted here):
  out/research/explore/I3_2/all18/results.json        all-18 LODO run
  out/research/explore/I3_2/real_dev/results.json     pre-registered 12-device run
  out/research/explore/mbias_deck/sky130/check.json   NGSpice re-simulation
      (scripts/research/explore/mbias_deck_resim.py --dataset sky130)
  slides/mbias_plots/*.png  (scripts/research/explore/mbias_deck_plots.py)
  stored per-device RRMS of the comparison methods (performance table)

  PYTHONPATH=src .venv/bin/python scripts/make_mbias_slides.py
Output: slides/cryo_ml_mbias_real_data.pptx
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "slides" / "cryo_ml_mbias_real_data.pptx"
PLOTS = ROOT / "slides" / "mbias_plots"
EXP = ROOT / "out/research/explore"
ALL18 = EXP / "I3_2/all18"
DEV12 = EXP / "I3_2/real_dev"
CHECK = EXP / "mbias_deck/sky130/check.json"
INK = RGBColor(32, 36, 40)
MUTED = RGBColor(90, 96, 102)
HEAD_FILL = RGBColor(232, 236, 240)
GROUP_FILL = RGBColor(246, 247, 248)


def jload(p: Path):
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------
def agg(d: dict) -> dict:
    v = np.array(list(d.values()), float)
    n = [x for t, x in d.items() if t.startswith("nmos")]
    p = [x for t, x in d.items() if t.startswith("pmos")]
    return {"n": len(v), "all": float(v.mean()), "combined": float((np.mean(n) + np.mean(p)) / 2),
            "median": float(np.median(v))}


def per_device_ml(name: str) -> dict:
    out = {}
    for f in glob.glob(str(ROOT / "out" / name / "ml_*.json")):
        r = jload(f)
        out[r["device"]] = float(r["rrms"])
    return out


def summary_rrms(path: Path, key: str = "rrms") -> dict:
    return {t: float(v[key]) for t, v in jload(path)["devices"].items()}


def performance_rows() -> list[dict]:
    ref = jload(ROOT / "out/research/reference_scores.json")
    refd = lambda k: {t: float(v["rrms"]) for t, v in ref[k]["devices"].items()}
    nd = ROOT / "out/research/ngspice_direct"
    pol = ROOT / "out/research/polished"
    a18 = jload(ALL18 / "results.json")
    d12full = jload(DEV12 / "results.json")
    d12 = d12full["per_device"]
    n12 = {lab: float(np.mean([r["n_sim"] for r in v.values()]))
           for lab, v in d12full["heldout_rows"].items()}
    hr = a18["heldout_rows"]
    nsim = {lab: float(np.mean([r["n_sim"] for r in hr[lab].values()])) for lab in hr}
    rt = {lab: float(np.mean([r["runtime_s"] for r in hr[lab].values()])) for lab in hr}
    s = lambda p: jload(p)
    rows = [
        {"group": "References (18 devices)"},
        {"m": "Published 77 K card (Beall et al.), strict scorer", "p": "published", "d": refd("released"),
         "cost": "-", "src": "reference_scores.json"},
        {"m": "Paper, 7 parameters", "p": "7", "d": refd("paper_seven"), "cost": "-",
         "src": "reference_scores.json"},
        {"m": "Paper, 43 parameters", "p": "43", "d": refd("paper_full43"), "cost": "-",
         "src": "reference_scores.json"},
        {"group": "Fixed ML series, July setup (18 devices)"},
        {"m": "direct MLP forward pass", "p": "7", "d": per_device_ml("pdk_direct_mlp"),
         "cost": "1 forward pass", "src": "out/pdk_direct_mlp"},
        {"m": "emu_search", "p": "7", "d": per_device_ml("pdk_ml_emu_raw"),
         "cost": "10k-card bank/device", "src": "out/pdk_ml_emu_raw"},
        {"m": "emu_search+fd", "p": "7", "d": per_device_ml("pdk_ml_emu"),
         "cost": "bank + FD; 2420 s run", "src": "out/pdk_ml_emu"},
        {"group": "Cold-start loop, October (18 devices)"},
        {"m": "Cold paper pipeline, round 4", "p": "7",
         "d": summary_rrms(pol / "pipe_r4_fd/real/summary.json", "before_rrms"),
         "cost": "40k bank sims/device", "src": "polished/pipe_r4_fd (before)"},
        {"m": "Cold paper pipeline, round 4 (a)", "p": "43",
         "d": summary_rrms(pol / "pipe43_r4_fixed_fd/real/summary.json", "before_rrms"),
         "cost": "40k bank sims/device", "src": "pipe_fixed/full_r4"},
        {"m": "Direct NGSpice (CMA-ES + TRF)", "p": "7",
         "d": summary_rrms(nd / "seven_r1_direct/summary.json"),
         "cost": f"{s(nd / 'seven_r1_direct/summary.json')['mean_ngspice_calls_per_device']:.0f} calls/device",
         "src": "ngspice_direct/seven_r1_direct"},
    ]
    for r, name in ((1, "full_r1_direct_seeded"), (2, "full_r2_direct"), (3, "full_r3_direct")):
        p = nd / name / "summary.json"
        rows.append({"m": f"Direct NGSpice, round {r}", "p": "43", "d": summary_rrms(p),
                     "cost": f"{s(p)['mean_ngspice_calls_per_device']:.0f} calls/device",
                     "src": f"ngspice_direct/{name}"})
    p = pol / "direct43_r3_fd/real/summary.json"
    rows.append({"m": "Direct NGSpice, round 3 + FD (starting card)", "p": "43",
                 "d": summary_rrms(p),
                 "cost": f"+{s(p)['mean_ngspice_calls_per_device']:.0f} calls/device",
                 "src": "polished/direct43_r3_fd"})
    rows += [
        {"group": "This deck: leave-one-device-out, all 18 devices"},
        {"m": "C0_refit (refit, g = 0)", "p": "43", "d": a18["per_device"]["C0_refit"],
         "cost": f"+{nsim['C0_refit']:.0f} calls/device", "src": "I3_2/all18"},
        {"m": "T1_inloop (refit, per-curve gain table) (b)", "p": "43 + table",
         "d": a18["per_device"]["T1_inloop"],
         "cost": f"+{nsim['T1_inloop']:.0f} calls/device", "src": "I3_2/all18"},
        {"m": "M_bias (refit with shared g)", "p": "43 + 1765 shared (c)",
         "d": a18["per_device"]["M_bias"],
         "cost": f"+{nsim['M_bias']:.0f} calls/device", "src": "I3_2/all18"},
        {"group": "Pre-registered I3-2 run, 12 devices, training on 11 (d)"},
        {"m": "Published 77 K card, same 12 devices", "p": "published",
         "d": {t: v for t, v in refd("released").items() if t in d12["M_bias"]}, "cost": "-",
         "src": "reference_scores.json", "n12": True},
        {"m": "C0_refit", "p": "43", "d": d12["C0_refit"], "cost": f"+{n12['C0_refit_rerun']:.0f} calls/device",
         "src": "I3_2/real_dev", "n12": True},
        {"m": "T1_inloop (b)", "p": "43 + table", "d": d12["T1_inloop"],
         "cost": f"+{n12['T1_inloop_rerun']:.0f} calls/device",
         "src": "I3_2/real_dev", "n12": True},
        {"m": "M_bias", "p": "43 + 1765 shared", "d": d12["M_bias"],
         "cost": f"+{n12['M_bias']:.0f} calls/device",
         "src": "I3_2/real_dev", "n12": True},
    ]
    for r in rows:
        if "d" in r:
            r["a"] = agg(r["d"])
    return rows, nsim, rt, a18


# ---------------------------------------------------------------------------
# pptx helpers (layout of make_paper_style_slides.py)
# ---------------------------------------------------------------------------
def new_slide(prs, title: str, size: int = 26):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.7))
    para = box.text_frame.paragraphs[0]
    run = para.add_run()
    run.text = title
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = INK
    return slide


def add_lines(slide, lines, left=0.6, top=1.15, width=12.1, height=5.9, size=17):
    body = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = body.text_frame
    frame.word_wrap = True
    for i, line in enumerate(lines):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = INK
        para.space_after = Pt(4)
    return body


def add_table(slide, rows, widths, left=0.45, top=1.05, size=11, row_h=0.27,
              head_rows=1, group_rows=(), align_right_from=1):
    n_r, n_c = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_r, n_c, Inches(left), Inches(top),
                                   Inches(sum(widths)), Inches(row_h * n_r))
    table = shape.table
    for j, w in enumerate(widths):
        table.columns[j].width = Inches(w)
    for i in range(n_r):
        table.rows[i].height = Inches(row_h)
        for j in range(n_c):
            cell = table.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.01)
            cell.text = str(rows[i][j])
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            para.font.color.rgb = INK
            para.alignment = PP_ALIGN.RIGHT if (j >= align_right_from and i >= head_rows
                                                and i not in group_rows) else PP_ALIGN.LEFT
            cell.fill.solid()
            if i < head_rows:
                para.font.bold = True
                cell.fill.fore_color.rgb = HEAD_FILL
            elif i in group_rows:
                para.font.bold = True
                para.font.italic = True
                cell.fill.fore_color.rgb = GROUP_FILL
            else:
                cell.fill.fore_color.rgb = RGBColor(255, 255, 255)
    for i in group_rows:
        table.cell(i, 0).merge(table.cell(i, n_c - 1))
    return table


def add_picture(slide, path: Path, left=0.4, top=0.95, max_w=12.5, max_h=6.3):
    with Image.open(path) as im:
        ratio = im.width / im.height
    w = max_w
    h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio
    x = left + (max_w - w) / 2
    slide.shapes.add_picture(str(path), Inches(x), Inches(top), Inches(w), Inches(h))


def add_box(slide, text, left, top, width, height, size=13, fill=RGBColor(240, 244, 248)):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = RGBColor(120, 128, 136)
    tf = shp.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = INK
        run.font.bold = i == 0
    return shp


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2),
                                   Inches(y2))
    c.line.color.rgb = RGBColor(90, 96, 102)
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd",
                          {"type": "triangle"})
    ln.append(tail)


def f4(x) -> str:
    return f"{x:.4f}"


def n_msgs(check) -> int:
    return sum(bool(check["devices"][t]["msgs"].get("M_bias_native")) for t in check["tags"])


def n_nonfinite(check) -> int:
    return sum(not np.isfinite(check["devices"][t]["scores"]["M_bias_native"])
               for t in check["tags"])


# ---------------------------------------------------------------------------
# slides
# ---------------------------------------------------------------------------
def title_slide(prs, a18):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.8), Inches(2.4), Inches(11.7), Inches(1.0))
    para = box.text_frame.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    run = para.add_run()
    run.text = "SKY130 at 77 K: 43-parameter card refit with a shared bias-only correction"
    run.font.size = Pt(30)
    run.font.bold = True
    run.font.color.rgb = INK
    s = a18["summary"]
    add_lines(slide, [
        "Measured I-V of 18 transistors; held-out device RRMS from NGSpice "
        "(leave-one-device-out)",
        f"All-device RRMS: released card 0.2751, paper 43 parameters 0.2038, "
        f"M_bias {s['M_bias']['all_device_mean']:.4f}",
        "2026-10-21",
    ], left=1.0, top=3.55, width=11.3, height=1.6, size=18).text_frame.paragraphs[0]
    for para in slide.shapes[-1].text_frame.paragraphs:
        para.alignment = PP_ALIGN.CENTER
        for r in para.runs:
            r.font.color.rgb = MUTED


def data_slide(prs, check):
    tags = check["tags"]
    n_n = sum(t.startswith("nmos") for t in tags)
    slide = new_slide(prs, "Data and metric")
    add_lines(slide, [
        f"Devices: SkyWater SKY130 130 nm, measured at 77 K (Beall et al.); {len(tags)} transistors, "
        f"{n_n} nMOS (nfet_01v8_lvt) and {len(tags) - n_n} pMOS (pfet_01v8_lvt), L 0.15-100 um, "
        "W 0.42-100 um.",
        "Curves per device: 11. Id-Vg at |Vds| = 0.01, 0.37, 0.74, 1.11, 1.48, 1.85 V; "
        "Id-Vd at |Vgs| = 0.37, 0.74, 1.11, 1.48, 1.85 V.",
        "Curve RRMS = RMS(I_sim - I_meas) / mean|I_meas| over the trimmed sweep points (paper "
        "rrmsCalc.py). Device RRMS = mean over included curves.",
        "Curve inclusion is frozen to the set the released card includes (9-11 curves per device). "
        "The upstream dynamic inclusion is not used here.",
        "All-device = mean of the 18 device values. Combined = (mean nMOS + mean pMOS) / 2. "
        "Paper numbers below are all-device.",
        "Simulator: ngspice-41 (arm64 build; reproduces the reference scores exactly). "
        "Every RRMS on these slides is from an NGSpice simulation.",
        "Reference, 18 devices: released card 0.2751 (combined 0.2595); paper, 7 parameters 0.2284 "
        "(0.2135); paper, 43 parameters 0.2038 (0.1905).",
        "Device split: the 18 devices are the 12 former DEV and 6 former TEST devices. The TEST "
        "devices were opened on 2026-10-21 by user decision.",
    ], size=16)


def method_slide(prs):
    slide = new_slide(prs, "Method: card refit with a shared correction g, leave-one-device-out")
    y, h = 1.15, 1.25
    boxes = [
        ("Starting card\n43 BSIM4 parameters per device\n(direct NGSpice fit, round 3 + FD)", 0.4),
        ("Train g\non the 17 other devices'\nstarting-card records", 3.55),
        ("Refit held-out card\n43 parameters, TRF in NGSpice\nwith g fixed in the residual", 6.7),
        ("Score held-out device\nstrict paper RRMS,\nNGSpice x exp(g)", 9.85),
    ]
    for text, x in boxes:
        add_box(slide, text, x, y, 2.95, h)
    for x in (3.35, 6.5, 9.65):
        arrow(slide, x, y + h / 2, x + 0.2, y + h / 2)
    add_lines(slide, [
        "Model:  I_D = I_card(V; theta) x exp(g),   g = mean over 5 seeds of "
        "tanh(MLP 3-16-16-1(is_pmos, |Vgs|/VDD, |Vds|/VDD)),   VDD = 1.85 V",
        "g inputs carry no geometry. One g is shared by all devices of both polarities. "
        "353 weights per seed, 1765 in total.",
        "Training of g: loss = mean strict paper RRMS of I_card x exp(g) over the training records, "
        "+ 1e-4 L2; Adam, learning rate 3e-3, 2000 steps; seeds 0-4. Cards fixed during training.",
        "Held-out refit: start from the device's starting card; bounded TRF on the 43 parameters, "
        "max_nfev 400, 2 IRLS rounds, diff_step 0.01; residual = I_card x exp(g) - I_meas.",
        "Folds: 18. In each fold the held-out device enters neither the training of g nor any "
        "other fitted quantity.",
        "Controls with the same refit:  C0_refit, g = 0.  T1_inloop, g replaced by one constant per "
        "(polarity, curve tag), e.g. pMOS idvg@0.37, 21 values fitted on the training devices; "
        "defined only at the measured bias settings.",
        "Recipe: pre-registered for the 12-device run (I3-2, 2026-10-20); applied unchanged to "
        "all 18 devices.",
    ], top=2.65, height=4.6, size=15)


def performance_slide(prs, rows):
    slide = new_slide(prs, "Performance across methods, sky130 77 K measured data", size=24)
    head = ["Method", "Fitted parameters", "n", "All-device", "Combined", "Median",
            "NGSpice cost", "Source"]
    table, groups = [head], []
    for r in rows:
        if "group" in r:
            groups.append(len(table))
            table.append([r["group"]] + [""] * 7)
            continue
        a = r["a"]
        table.append([r["m"], r["p"], str(a["n"]), f4(a["all"]), f4(a["combined"]),
                      f4(a["median"]), r["cost"], r["src"]])
    add_table(slide, table, [3.55, 1.6, 0.35, 0.9, 0.9, 0.8, 1.85, 2.45], left=0.2, top=0.85,
              size=9, row_h=0.215, group_rows=groups, align_right_from=2)
    add_lines(slide, [
        "Fixed procedures applied to every device; no per-device choice between methods. "
        "Rows with n = 12 compare only with each other.  "
        "(a) Re-scored with failed-curve rejection (stored before the fix: 0.1513).  "
        "(b) One constant g per (polarity, curve tag), 21 values; defined only at the measured biases.  "
        "(c) 43 card parameters per device plus 5 x 353 network weights shared by all devices.  "
        "(d) Pre-registered I3-2 run on the 12 former DEV devices.  Cost: NGSpice simulations; "
        "'+' = on top of the starting card; direct NGSpice rounds: calls in that round.",
    ], top=6.55, height=0.9, size=9.5)


def per_device_slide(prs, check, a18):
    pd = a18["per_device"]
    tags = check["tags"]
    test = set(jload(EXP / "real_split.json")["REAL_TEST"])
    stored = {t: check["devices"][t]["stored"] for t in tags}
    cols = (("published", lambda t: stored[t]["published"]),
            ("start", lambda t: stored[t]["start"]),
            ("C0_refit", lambda t: pd["C0_refit"][t]),
            ("T1_inloop", lambda t: pd["T1_inloop"][t]),
            ("M_bias", lambda t: pd["M_bias"][t]))
    slide = new_slide(prs, "Per-device held-out RRMS, 18 devices", size=24)
    head = ["Device", "Published card", "Starting card", "C0_refit", "T1_inloop", "M_bias"]
    table = [head]
    for t in tags:
        table.append([t + (" *" if t in test else "")] + [f4(f(t)) for _, f in cols])
    vals = {k: {t: f(t) for t in tags} for k, f in cols}
    for lab, key in (("All-device", "all"), ("Combined", "combined"), ("Median", "median")):
        table.append([lab] + [f4(agg(vals[k])[key]) for k, _ in cols])
    add_table(slide, table, [2.4, 1.45, 1.45, 1.45, 1.45, 1.45], left=0.5, top=0.85, size=10.5,
              row_h=0.262)
    add_lines(slide, [
        "* former TEST device (opened 2026-10-21).",
        "Starting card: direct NGSpice 43-parameter fit (round 3 + FD); "
        "every refit starts from it.",
        "C0_refit, T1_inloop, M_bias: held-out refit, leave-one-device-out.",
        f"M_bias below C0_refit on "
        f"{sum(pd['M_bias'][t] < pd['C0_refit'][t] for t in tags)} of {len(tags)} devices.",
        "Published-card RRMS: reference_scores.json; reproduced by the deck re-simulation.",
    ], left=10.25, top=1.0, width=2.95, height=5.5, size=12)


def dev12_slide(prs, a18):
    d12 = jload(DEV12 / "results.json")["per_device"]
    pd = a18["per_device"]
    ref = jload(ROOT / "out/research/reference_scores.json")["released"]["devices"]
    tags12 = sorted(d12["M_bias"])
    slide = new_slide(prs, "Earlier pre-registered run: 12 devices, training on 11", size=24)
    head = ["Same 12 devices", "Published card", "C0_refit", "T1_inloop", "M_bias"]
    rows = [head]
    for lab, src in (("I3-2 run (2026-10-20), g trained on 11 devices", d12),
                     ("All-18 run, g trained on 17 devices", pd)):
        for conv in ("all", "combined"):
            rows.append([f"{lab}, {'all-device' if conv == 'all' else 'combined'}",
                         f4(agg({t: ref[t]['rrms'] for t in tags12})[conv])]
                        + [f4(agg({t: src[k][t] for t in tags12})[conv])
                           for k in ("C0_refit", "T1_inloop", "M_bias")])
    add_table(slide, rows, [5.3, 1.6, 1.6, 1.6, 1.6], left=0.5, top=1.1, size=13, row_h=0.42)
    add_lines(slide, [
        "The I3-2 run was pre-registered on the 12 former DEV devices; the 6 TEST devices were "
        "not loaded then.",
        "On 2026-10-21 the user opened the 6 TEST devices. The all-18 run uses the same recipe "
        "with 17 training devices per fold.",
        "C0_refit does not depend on the training devices: on the 12 devices the two runs give "
        "the same values (max |difference| "
        f"{a18['reproduction'].get('C0_refit_vs_stored_I2_1_on_REAL_DEV', float('nan')):.1e}).",
    ], top=3.6, height=2.5, size=15)


def bar_slide(prs):
    slide = new_slide(prs, "Per-device RRMS, 18 devices (log scale)", size=24)
    add_picture(slide, PLOTS / "rrms_per_device.png", top=0.95, max_h=6.2)


def deploy_slide(prs, check):
    dv = check["devices"]
    tags = check["tags"]
    nat = {t: dv[t]["scores"]["M_bias_native"] for t in tags}
    py = {t: dv[t]["scores"]["M_bias_python"] for t in tags}
    st = {t: dv[t]["stored"]["M_bias_python"] for t in tags}
    dmax = max(abs(nat[t] - py[t]) for t in tags)
    f1c = jload(EXP / "I3_2/followup/f1c/summary.json")
    fa = f1c["aggregates"]
    slide = new_slide(prs, "Deployment: g as a Verilog-A element in NGSpice", size=24)
    add_lines(slide, [
        "Export: g written as Verilog-A (module i3_2_bias_corr, explicit tanh layers, 5-seed mean). "
        "Element between the external drain d and the transistor drain di:  "
        "I(d, s) += (exp(g) - 1) x I(d -> di).",
        "Compile: OpenVAF 23.5.0 -> OSDI 0.3. Simulator: ngspice-41 built with --enable-osdi "
        "(same source and flags as the canonical build; uncorrected currents bit-identical).",
        "Netlist:  M1 di g s b <card>;  NCORR d g s di corrmodel;  pre_osdi <file>.osdi in .control.",
    ], top=1.0, height=2.2, size=15)
    rows = [["Run", "Devices", "SPICE-native (Verilog-A)", "NGSpice card x exp(g), Python",
             "Max per-device |difference|"],
            ["All-18 run, 18 fold models", "18",
             f"{agg(nat)['all']:.6f} ({agg(nat)['combined']:.6f})",
             f"{agg(py)['all']:.6f} ({agg(py)['combined']:.6f})", f"{dmax:.1e}"],
            ["I3-2 run, 12 fold models (F1c)", "12",
             f"{fa['native_ng41osdi']['all_device_mean']:.6f} "
             f"({fa['native_ng41osdi']['combined']:.6f})",
             f"{fa['stored_M_bias']['all_device_mean']:.6f} ({fa['stored_M_bias']['combined']:.6f})",
             f"{f1c['checks']['F1.3_end_to_end']['max_abs_delta_device']:.1e}"]]
    add_table(slide, rows, [3.2, 0.9, 2.75, 2.75, 2.3], left=0.45, top=3.3, size=12, row_h=0.42,
              align_right_from=1)
    add_lines(slide, [
        "Values: all-device RRMS (combined).  Python value on the deck re-simulation vs stored "
        f"all-18 value: max per-device |difference| {max(abs(py[t] - st[t]) for t in tags):.1e}.",
        "I3-2 F1c: SPICE-native and Python currents differ by up to 2.45e-3 relative, at a "
        "2.0e-12 A point (pmos_L0p5_W0p42, Id-Vg at |Vds| = 1.85 V); 0 convergence messages, "
        "0 non-finite currents.",
        f"All-18 SPICE-native runs: {n_msgs(check)} devices with convergence messages, "
        f"{n_nonfinite(check)} non-finite scored values.",
    ], top=4.75, height=2.0, size=14)


def gmap_slide(prs):
    slide = new_slide(prs, "Learned g, full model trained on all 18 devices", size=24)
    add_picture(slide, PLOTS / "g_map.png", top=0.95, max_h=5.6)
    add_lines(slide, [
        "The full model is the one written for deployment; held-out scores use the 18 fold "
        "models (each without its device). exp(g) is the factor applied to the card current.",
    ], top=6.6, height=0.7, size=13)


def iv_slides(prs, check):
    test = set(jload(EXP / "real_split.json")["REAL_TEST"])
    for t in check["tags"]:
        pol, l_s, w_s = t.split("_")
        f = lambda s: s[1:].replace("p", ".")
        title = (f"{t}:  {pol[0].upper()}MOS, L = {f(l_s)} um, W = {f(w_s)} um"
                 + ("  (former TEST device)" if t in test else ""))
        slide = new_slide(prs, title, size=20)
        add_picture(slide, PLOTS / f"iv_{t}.png", top=0.8, max_h=5.75)
        add_lines(slide, [
            "Points: measured. Lines: NGSpice (ngspice-41). M_bias line: refit card with the "
            "fold Verilog-A g in the netlist (OSDI), g trained without this device. "
            "Faded: curves outside the frozen inclusion set (not scored). "
            "RRMS: strict paper metric for this device.",
        ], top=6.75, height=0.6, size=11)


def main() -> None:
    check = jload(CHECK)
    rows, _nsim, _rt, a18 = performance_rows()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    title_slide(prs, a18)
    data_slide(prs, check)
    method_slide(prs)
    performance_slide(prs, rows)
    per_device_slide(prs, check, a18)
    dev12_slide(prs, a18)
    bar_slide(prs)
    deploy_slide(prs, check)
    gmap_slide(prs)
    iv_slides(prs, check)
    DECK.parent.mkdir(parents=True, exist_ok=True)
    prs.save(DECK)
    print(f"wrote {DECK} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
