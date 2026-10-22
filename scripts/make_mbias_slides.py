#!/usr/bin/env python3
"""Deck: 43-parameter card refit with the shared bias-only correction M_bias
(I3-2 recipe), sky130 77 K, all 18 measured devices, leave-one-device-out.

Inputs (all stored; nothing is fitted here):
  out/research/explore/I13_1/all18/results.json       all-18 LODO run, BSIM4 clamp
      state C (I13-1 rerun of the I3-2 recipe; replaces I3_2/all18)
  out/research/explore/I13_1/summary.json             old (state L) vs new (state C)
  out/research/explore/I15_2/decision.json, I16_1/{decision.json, accuracy/accuracy.json,
      deploy_sel/{summary,restrictions}.json}, I15_1/deploy_{v2,v1}/summary.json
      (g v2 with the selected penalty weight: accuracy and deployment checks)
  out/research/explore/I3_2/real_dev/results.json     pre-registered 12-device run
  out/research/explore/mbias_deck/sky130/check.json   NGSpice re-simulation
      (scripts/research/explore/mbias_deck_resim.py --dataset sky130)
  slides/mbias_plots/*.png  (scripts/research/explore/mbias_deck_plots.py)
  slides/mbias_plots/m_*.png, eq_*.png and
  out/research/explore/mbias_deck/method_numbers.json
      (scripts/research/explore/mbias_method_plots.py; method section)
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
ALL18 = EXP / "I13_1/all18"        # I13-1: state-C rerun of the I3-2 recipe
I13 = EXP / "I13_1"
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
    start_c = {t: float(v["new"]) for t, v in jload(I13 / "summary.json")["per_device"]["start"].items()}
    rows.append({"m": "Direct NGSpice, round 3 + FD (starting card), state C (e)", "p": "43",
                 "d": start_c,
                 "cost": f"+{s(p)['mean_ngspice_calls_per_device']:.0f} calls/device",
                 "src": "polished/direct43_r3_fd"})
    rows += [
        {"group": "This deck: leave-one-device-out, all 18 devices, BSIM4 clamp state C"},
        {"m": "C0_refit (refit, g = 0)", "p": "43", "d": a18["per_device"]["C0_refit"],
         "cost": f"+{nsim['C0_refit']:.0f} calls/device", "src": "I13_1/all18"},
        {"m": "T1_inloop (refit, per-curve gain table) (b)", "p": "43 + table",
         "d": a18["per_device"]["T1_inloop"],
         "cost": f"+{nsim['T1_inloop']:.0f} calls/device", "src": "I13_1/all18"},
        {"m": "M_bias (refit with shared g)", "p": "43 + 1765 shared (c)",
         "d": a18["per_device"]["M_bias"],
         "cost": f"+{nsim['M_bias']:.0f} calls/device", "src": "I13_1/all18"},
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


FIT_WARNINGS: list[str] = []


def text_height(lines, width: float, size: float) -> float:
    """Estimated height (in) of ``lines`` at ``size`` pt in a box ``width`` in wide
    (mean glyph width 0.5 em, line height 1.2 em, 4 pt paragraph spacing)."""
    per_line = max(1, int((width - 0.2) * 72 / (0.5 * size)))
    n = sum(max(1, -(-len(line) // per_line)) for line in lines)
    return n * 1.2 * size / 72 + len(lines) * 4 / 72 + 0.1


def add_lines(slide, lines, left=0.6, top=1.15, width=12.1, height=5.9, size=17):
    need = text_height(lines, width, size)
    if need > height or top + height > 7.5 or left + width > 13.34:
        FIT_WARNINGS.append(f"'{slide.shapes[0].text_frame.text[:50]}': text needs {need:.2f} in, "
                            f"box {height:.2f} in at top {top:.2f}")
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
    return x, w, h


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
        "2026-10-22 (BSIM4 clamp state C, I13-1)",
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
        "Every RRMS on these slides is from an NGSpice simulation, with the BSIM4 A2 clamp "
        "written into the card (state C, slide 'BSIM4 A2 clamp').",
        "Reference, 18 devices: released card 0.2751 (combined 0.2595); paper, 7 parameters 0.2284 "
        "(0.2135); paper, 43 parameters 0.2038 (0.1905).",
        "Device split: the 18 devices are the 12 former DEV and 6 former TEST devices. The TEST "
        "devices were opened on 2026-10-21 by user decision.",
    ], size=16)


# ---------------------------------------------------------------------------
# method section (figures: scripts/research/explore/mbias_method_plots.py)
# ---------------------------------------------------------------------------
NUM = EXP / "mbias_deck/method_numbers.json"


def add_eq(slide, name: str, left: float, top: float, size: float = 16, nums=None):
    """Equation PNG (mathtext, rendered at 20 pt) placed at ``size`` pt."""
    w, h = nums["eq_sizes_in"][name]
    w, h = w * size / 20, h * size / 20
    if left + w > 13.2:
        FIT_WARNINGS.append(f"equation {name}: width {w:.2f} in at left {left:.2f}")
    slide.shapes.add_picture(str(PLOTS / f"eq_{name}.png"), Inches(left), Inches(top),
                             Inches(w), Inches(h))
    return w, h


def overview_slide(prs, nums, a18):
    st = nums["stages"]["stages"]
    sims_start = sum(r["calls"] for r in st if r["calls"])
    rf = nums["refit"]
    tm = nums["timing_all18"]
    slide = new_slide(prs, "Pipeline: measured I-V to held-out RRMS, 18 folds", size=24)
    w, h = 3.7, 1.35
    row1 = [("Measured I-V, 77 K\n18 devices x 11 curves\n9-11 curves scored per device", 0.45),
            (f"Starting cards (43 parameters)\ndirect NGSpice fit, 5 stages\n"
             f"{sims_start:,.0f} simulations per device", 4.8),
            (f"Records\nstarting card simulated once\non the measured grid: "
             f"{nums['records']['total']:,} points", 9.15)]
    for text, x in row1:
        add_box(slide, text, x, 1.0, w, h, size=13)
    for x in (4.15, 8.5):
        arrow(slide, x + 0.03, 1.0 + h / 2, x + 0.62, 1.0 + h / 2)
    y2 = 3.05
    rec = nums["records"]
    row2 = [(f"Train g (fold k)\n17 devices, {rec['train_min']:,}-{rec['train_max']:,} points\n"
             f"5 seeds, Python only", 0.45),
            (f"Refit card of device k\n{nums['box']['n_active_min']}-{nums['box']['n_active_max']} "
             f"active of 43 parameters, TRF\n"
             f"g fixed; {rf['n_sim']['M_bias'][0]}-{rf['n_sim']['M_bias'][1]} simulations", 4.8),
            ("Score device k\nNGSpice(final card) x exp(g)\nstrict paper RRMS", 9.15)]
    for text, x in row2:
        add_box(slide, text, x, y2, w, h, size=13, fill=RGBColor(252, 236, 226))
    for x in (4.15, 8.5):
        arrow(slide, x + 0.03, y2 + h / 2, x + 0.62, y2 + h / 2)
    arrow(slide, 11.0, 1.0 + h, 2.3, y2)
    add_lines(slide, ["repeat for k = 1 ... 18 (leave-one-device-out)"], left=0.45, top=y2 + h + 0.05,
              width=12.4, height=0.45, size=13)
    add_eq(slide, "model", 0.6, 5.05, size=17, nums=nums)
    s = a18["summary"]
    add_lines(slide, [
        f"Run time (all 18 folds, process pool): g training {tm['train_s']:.0f} s "
        f"({tm['train_n_seed_tasks']} seed tasks = 19 models x 5 seeds), held-out refits "
        f"{tm['heldout_refit_s']:.0f} s ({tm['heldout_refit_n']} refits = 3 arms x 18), "
        f"total {tm['runtime_s']:.0f} s.",
        f"All-device RRMS, held-out: starting card {st[-1]['all_device']:.4f}; "
        f"C0_refit (g = 0) {s['C0_refit']['all_device_mean']:.4f}; "
        f"M_bias {s['M_bias']['all_device_mean']:.4f}.",
    ], top=5.7, height=1.3, size=14)


def data_metric_slide(prs, nums):
    d = nums["data"]
    exc = [c["key"] for c in d["curves"] if not c["included"]]
    slide = new_slide(prs, f"Measured curves, trim and curve RRMS (example {d['tag']})", size=24)
    add_picture(slide, PLOTS / "m_data.png", left=0.4, top=0.85, max_w=8.6, max_h=3.6)
    add_eq(slide, "rrms", 0.6, 4.6, size=15, nums=nums)
    add_eq(slide, "dev", 0.6, 5.45, size=15, nums=nums)
    add_lines(slide, [
        "Trim: points before the first |I_meas| above the device floor are removed "
        "(nMOS floor 0; pMOS floor per geometry, rrmsCalc.py). n_c = points left.",
        "den_c and n_c are computed from the measured curve only.",
        f"{d['tag']}: {len(d['curves']) - len(exc)} of {len(d['curves'])} curves scored; "
        f"not in the inclusion set: {', '.join(exc)}.",
    ], left=9.15, top=0.95, width=3.95, height=3.6, size=13)
    add_lines(slide, [
        "Device RRMS = mean of the curve RRMS over the frozen inclusion set; "
        "a non-finite included curve makes the device RRMS NaN (failed).",
    ], top=6.1, height=0.8, size=13)


def screen_slide(prs, nums):
    sc = nums["screen"]
    slide = new_slide(prs, "Starting cards (1/4): 43 parameters from a sensitivity screen", size=24)
    add_picture(slide, PLOTS / "m_screen.png", left=0.3, top=0.85, max_w=12.7, max_h=3.6)
    names = {"process_and_effective_geometry": "process, geometry",
             "threshold_and_short_channel": "threshold, short channel",
             "mobility_velocity_and_resistance": "mobility, velocity, resistance",
             "subthreshold_and_dibl": "subthreshold, DIBL",
             "output_conductance_and_high_field": "output conductance, high field"}
    rows = [["Group", "n", "Parameters"]]
    for g, ps in sc["groups"].items():
        rows.append([names.get(g, g), str(len(ps)), " ".join(ps)])
    add_table(slide, rows, [2.7, 0.4, 4.6], left=0.4, top=4.6, size=10, row_h=0.36,
              align_right_from=1)
    add_lines(slide, [
        f"Candidates: {sc['n_candidates']} continuous BSIM4 DC parameters "
        f"({sc['n_excluded_pre_screen']} integer selectors, bin bounds, capacitance-only and "
        "noise-only parameters excluded before).",
        f"Each candidate: +/-{sc['fraction']:.0%} of its published-card value (family median if "
        "the value is 0), one at a time, on all 18 devices.",
        "Response per curve: RMS(I_pert - I_pub) / mean|I_meas|.",
        f"Kept if the response is >= {sc['threshold']:.0e} on any curve of any device; "
        "the 7 paper parameters are always kept.",
    ], left=8.35, top=4.55, width=4.85, height=2.85, size=12)


def box_slide(prs, nums):
    bx = nums["box"]
    slide = new_slide(prs, "Starting cards (2/4): parameter box and box coordinates", size=24)
    add_picture(slide, PLOTS / "m_box.png", left=0.3, top=0.85, max_w=12.7, max_h=3.55)
    add_eq(slide, "box", 0.6, 4.55, size=16, nums=nums)
    add_lines(slide, [
        "c_k round 1: released 77 K value / its multiplier in the 77 K card "
        "(multiplier-free stock SkyWater value).",
        "h_k round 1: vth0 ln 3; u0, nfactor, vsat, delta, rdsw, eta0 ln 100; "
        "the 36 screened parameters ln 3.",
        "Rounds 2, 3: c_k = previous round's card. |u_k| >= 0.9: h_k kept (box extends past the "
        "old bound); otherwise h_k / 2, minimum ln 1.05.",
        "Fixed (held at c_k): parameters with value 0 on that device (vfb, prwg, a0, b0, b1, "
        "lpe0, lpeb on some devices), and parameters whose box width is below the 1e-14 activity "
        "test of the box code: ub on all 18 devices (|ub| ~ 1e-18), b1 on 5 devices. "
        f"Active: {bx['n_active_min']}-{bx['n_active_max']} per device.",
    ], top=5.1, height=2.3, size=13)


def search_slide(prs, nums):
    st = nums["stages"]
    rows = [["Run", "Par.", "Starts", "CMA-ES evals", "TRF max_nfev, IRLS", "Sim./device",
             "Wall s", "All-device"]]
    spec = {"7 parameters": ("7", "box centre + 4 bank samples"),
            "43 p., round 1": ("43", "box centre; 7-par. card, others at centre"),
            "43 p., round 2": ("43", "previous card"),
            "43 p., round 3": ("43", "previous card"),
            "round 3 + FD": ("43", "round-3 card")}
    for r in st["stages"][1:]:
        o = r["options"]
        cma = f"{o['cmaes_evals']}" if "cmaes_evals" in o else "-"
        trf = (f"{o['polish_nfev']}, {o['irls']}" if "polish_nfev" in o
               else f"{o['max_nfev']}, {o['irls']}")
        rows.append([r["label"], spec[r["label"]][0], spec[r["label"]][1], cma, trf,
                     f"{r['calls']:,.0f}", f"{r['wall_s']:,.0f}", f4(r["all_device"])])
    slide = new_slide(prs, "Starting cards (3/4): CMA-ES, then TRF, in NGSpice", size=24)
    add_table(slide, rows, [1.9, 0.55, 2.85, 1.3, 1.75, 1.2, 0.9, 1.15], left=0.45, top=0.95,
              size=12, row_h=0.36, align_right_from=3)
    ex = st["example_r1"]
    cb = st["cma_better_than_best_start"]
    add_lines(slide, [
        "Objective for every stage: mean included-curve RRMS of the NGSpice card "
        "(frozen inclusion set).",
        "Starts: each start is simulated; CMA-ES runs from the best one. Bank samples: the 4 "
        "lowest-RRMS samples of the 7-parameter round-1 NGSpice bank, pairwise RMS distance "
        ">= 0.35 in u.",
        "CMA-ES on the active u in [-1, 1]: sigma0 = 0.3, population 4 + floor(3 ln n_active), "
        "seed 20261015.",
        "TRF: bounded least squares on the residuals of the refit slide (forward-difference "
        "Jacobian, step 0.01 in u), starting from the better of start and CMA-ES; the best card "
        "seen in the run is kept.",
        f"CMA-ES result below the best start: {cb['7 parameters']}/18 (7 par.), "
        f"{cb['43 p., round 1']}/18, {cb['43 p., round 2']}/18, {cb['43 p., round 3']}/18 "
        "(rounds 1-3).",
        f"Example {ex['tag']}, round 1: box centre {ex['start_scores'][0]:.4f}; 7-par. card "
        f"{ex['start_scores'][1]:.4f}; CMA-ES best {ex['cmaes_best']:.4f}; TRF from the 7-par. "
        f"card {ex['polish_history'][1]:.4f}, {ex['polish_history'][2]:.4f} "
        f"(IRLS rounds 1, 2); {ex['n_sim']:,} simulations.",
    ], top=3.3, height=4.1, size=15)


def stages_slide(prs, nums):
    st = nums["stages"]["stages"]
    slide = new_slide(prs, "Starting cards (4/4): RRMS and cost per stage", size=24)
    add_picture(slide, PLOTS / "m_stages.png", left=0.3, top=0.9, max_w=12.7, max_h=4.4)
    fd = st[-1]
    add_lines(slide, [
        f"Starting card = round 3 + FD: all-device {fd['all_device']:.4f}, combined "
        f"{fd['combined']:.4f}. Sum over the five stages: "
        f"{sum(r['calls'] for r in st if r['calls']):,.0f} NGSpice simulations per device.",
        "Published-card RRMS: reference_scores.json (strict scorer, same frozen inclusion).",
    ], top=5.2, height=1.4, size=14)


def records_slide(prs, nums):
    r = nums["records"]
    lr = r["ln_ratio"]
    slide = new_slide(prs, f"Training data for g: 18 devices, {r['total']:,} points", size=24)
    add_picture(slide, PLOTS / "m_records.png", left=0.3, top=0.9, max_w=12.7, max_h=4.2)
    add_lines(slide, [
        "Record of a device: its starting card simulated once on the measured bias grid; per point "
        "(is_pmos, Vgs, Vds, I_card, I_meas, curve, den_c). Kept: included curves, points after "
        "the trim.",
        f"Per fold: the 17 training devices give {r['train_min']:,}-{r['train_max']:,} points.",
        f"Median |ln(I_meas / I_card)|: nMOS {lr['nmos']['median_abs_ln_ratio']:.3f}, pMOS "
        f"{lr['pmos']['median_abs_ln_ratio']:.3f} (points with |I| < 1e-13 A left out of the "
        "colour plot only).",
    ], top=5.05, height=2.0, size=13)


def network_slide(prs, nums):
    t = nums["training"]
    slide = new_slide(prs, "g: network and ensemble", size=24)
    add_picture(slide, PLOTS / "m_network.png", left=0.2, top=0.8, max_w=8.0, max_h=3.1)
    y = 3.95
    for name, size in (("model", 16), ("x", 16), ("z", 13), ("g", 16)):
        _, h = add_eq(slide, name, 0.5, y, size=size, nums=nums)
        y += h + 0.12
    add_lines(slide, [
        "f_s: Linear(3,16), tanh, Linear(16,16), tanh, Linear(16,1). 353 weights per seed, "
        "1765 for 5 seeds.",
        "Last layer initialised to 0, so g = 0 at step 0.",
        "Inputs: no geometry, no current, no curve label. The same g is applied to every "
        "curve of a device.",
        "One g per fold: trained on 17 devices, applied to the held-out device.",
        f"Fold {t['held']}: mu = ({t['mu'][1]:.3f}, {t['mu'][2]:.3f}), sigma = "
        f"({t['sd'][1]:.3f}, {t['sd'][2]:.3f}) for the two voltage inputs.",
    ], left=8.35, top=0.95, width=4.85, height=5.0, size=13)


def training_slide(prs, nums):
    t = nums["training"]
    fl = list(t["final_loss_per_seed"].values())
    slide = new_slide(prs, "Training g: loss and optimizer", size=24)
    add_eq(slide, "loss", 0.5, 0.95, size=15, nums=nums)
    add_picture(slide, PLOTS / "m_training.png", left=0.3, top=2.0, max_w=8.2, max_h=3.2)
    add_lines(slide, [
        "Sum over training devices d, their included curves c and trimmed points p. "
        "L2 on the weight matrices only.",
        "I_card fixed (starting cards): no simulation during training.",
        "Full batch, float64, Adam, learning rate 3e-3, 2000 steps; seeds 0-4, "
        "torch.manual_seed(s); seeds trained independently.",
        f"Fold {t['held']} ({t['n_train_points']:,} points): RRMS term {t['loss_g0']:.4f} at "
        f"g = 0; {min(fl):.4f}-{max(fl):.4f} per seed; {t['loss_ensemble']:.4f} for the "
        "5-seed mean g.",
        f"Std of g across seeds at the training points: median {t['seed_spread_g_std_median']:.4f}, "
        f"max {t['seed_spread_g_std_max']:.3f}.",
        f"Trace: re-run of the stored fold model's trainer (fit_bias, training records in load "
        f"order, {t['runtime_s']:.0f} s); max |g - stored fold g| = {t['max_abs_g_vs_stored']:.1g}, "
        f"max |weight difference| = {t.get('max_abs_weight_vs_stored', float('nan')):.1g}.",
    ], left=8.65, top=1.95, width=4.55, height=5.3, size=12.5)


def gfold_slide(prs, nums):
    g = nums["g_folds"]
    slide = new_slide(prs, "g of the 18 fold models at the held-out points", size=24)
    add_picture(slide, PLOTS / "m_g_folds.png", left=0.3, top=0.9, max_w=12.7, max_h=4.3)
    add_lines(slide, [
        f"Range of g at the held-out scored points: nMOS [{g['nmos']['min']:.3f}, "
        f"{g['nmos']['max']:.3f}], median |g| {g['nmos']['median_abs']:.4f}; pMOS "
        f"[{g['pmos']['min']:.3f}, {g['pmos']['max']:.3f}], median |g| "
        f"{g['pmos']['median_abs']:.4f}.",
        "Each fold model is evaluated only on its own held-out device. Full model (black): trained "
        "on all 18 devices; used for the Verilog-A export, not for scores.",
    ], top=5.15, height=1.8, size=13)


def refit_slide(prs, nums):
    rf = nums["refit"]
    h = rf["history_example"]
    slide = new_slide(prs, "Refit of the held-out card with g fixed", size=24)
    add_eq(slide, "resid", 0.5, 0.95, size=15, nums=nums)
    add_eq(slide, "irls", 0.5, 1.75, size=14, nums=nums)
    add_picture(slide, PLOTS / "m_refit.png", left=0.3, top=2.3, max_w=12.7, max_h=3.75)
    nm = rf["n_sim"]["M_bias"]
    add_lines(slide, [
        "scipy least_squares, method trf, u in [-1, 1] of the round-3 box, start = starting "
        "card. Jacobian: forward difference, step 0.01 in u, n_active + 1 simulations. "
        "Budget max_nfev = 400: per IRLS round floor(400 / 2 / (n_active + 2)) = 4-5 residual "
        "evaluations.",
        f"Simulations per device: {nm[0]}-{nm[1]}, mean {nm[2]:.0f}; "
        f"{rf['runtime_s']['M_bias']:.0f} s. Kept: best card seen (mean RRMS with g). "
        f"{rf['example']}: {h[0]:.4f} -> {h[1]:.4f} -> {h[2]:.4f}.",
    ], top=6.1, height=1.35, size=12.5)


def refit_effect_slide(prs, nums):
    m = nums["refit_effect"]["means"]
    slide = new_slide(prs, "Refit versus no refit, same fold g", size=24)
    add_picture(slide, PLOTS / "m_refit_effect.png", left=0.3, top=0.9, max_w=12.7, max_h=4.9)
    add_lines(slide, [
        f"No refit: starting card x exp(g) {m['start_x_g']:.4f}. Refit with g: {m['M_bias']:.4f}. "
        f"Without g: starting card {m['start']:.4f}, C0_refit {m['C0_refit']:.4f}.",
    ], top=5.95, height=1.0, size=14)


def loso_slide(prs, nums):
    tm = nums["timing_all18"]
    slide = new_slide(prs, "Leave-one-device-out loop over 18 devices", size=24)
    add_picture(slide, PLOTS / "m_loso.png", left=0.3, top=0.85, max_w=6.4, max_h=6.4)
    add_lines(slide, [
        "Fold k (k = 1 ... 18), device k held out:",
        "1  mu, sigma and g (5 seeds) from the 17 training records.",
        "2  T1 table from the same 17 records.",
        "3  Three refits of device k from its starting card: g (M_bias), T1 table (T1_inloop), "
        "g = 0 (C0_refit). Same refit options.",
        "4  Score each refit card (next slide).",
        "Device k enters none of mu, sigma, g, the T1 table.",
        f"Models trained: 18 fold models + 1 full model (all 18 devices, export only) = 19 x 5 "
        f"seeds = {tm['train_n_seed_tasks']} tasks, {tm['train_s']:.0f} s. T1 tables "
        f"{tm['t1_fit_s']:.0f} s. Refits {tm['heldout_refit_n']}, {tm['heldout_refit_s']:.0f} s.",
        "Recipe pre-registered for the 12-device run (I3-2); applied unchanged to 18 devices.",
    ], left=6.9, top=1.0, width=6.3, height=6.2, size=14)


def controls_slide(prs, nums):
    c = nums["controls"]
    slide = new_slide(prs, "Scoring and controls: C0_refit, T1_inloop", size=24)
    add_picture(slide, PLOTS / "m_controls.png", left=0.3, top=0.85, max_w=7.6, max_h=3.3)
    add_eq(slide, "t1", 0.5, 4.3, size=14, nums=nums)
    add_lines(slide, [
        "Score: final card simulated in NGSpice on the measured grid, multiplied by exp(g), strict "
        "paper RRMS on the frozen inclusion set.",
        "C0_refit: same refit with g = 0.",
        "T1_inloop: g = one constant per (polarity, curve tag), fitted on the 17 training records "
        "with the same loss (Adam, lr 2e-2, 1000 steps); 21 values (20 in the fold of "
        "pmos_L0p35_W0p55, the only device with pMOS idvd@0.37 included; that curve gets g = 0). "
        "Defined only at the measured bias settings.",
    ], left=0.45, top=4.8, width=7.5, height=2.6, size=12.5)
    rows = [["Paired ratio", "Geo. mean", "95% CI", "Lower"]]
    for k, lab in (("M_bias_vs_C0_refit", "M_bias / C0_refit"),
                   ("M_bias_vs_T1_inloop", "M_bias / T1_inloop"),
                   ("T1_inloop_vs_C0_refit", "T1_inloop / C0_refit")):
        v = c[k]
        rows.append([lab, f"{v['ratio']:.3f}", f"[{v['ci_low']:.3f}, {v['ci_high']:.3f}]",
                     f"{v['wins']}/{v['n_used']}"])
    add_table(slide, rows, [2.0, 1.1, 1.45, 0.65], left=8.05, top=1.0, size=11.5, row_h=0.36)
    add_lines(slide, [
        "Geo. mean = exp(mean over devices of ln(RRMS_a / RRMS_b)). CI: percentile, 10,000 "
        "bootstrap resamples of the 18 devices. Lower = devices with RRMS_a < RRMS_b.",
    ], left=8.1, top=2.6, width=5.1, height=1.5, size=12)


def va_slide(prs, nums, check):
    ex = nums["export"]
    dv = check["devices"]
    dmax = max(abs(dv[t]["scores"]["M_bias_native"] - dv[t]["scores"]["M_bias_python"])
               for t in check["tags"])
    slide = new_slide(prs, "Verilog-A element for g in ngspice-41", size=24)
    add_picture(slide, PLOTS / "m_va.png", left=0.2, top=0.85, max_w=6.3, max_h=4.4)
    add_eq(slide, "va", 0.5, 5.35, size=15, nums=nums)
    add_lines(slide, [
        "Ports (d, g, s, di). The transistor drain is di. Branch d -> di: "
        "V(sense) <+ RS * I(sense), RS = 0 (zero-volt source; its current is I_card).",
        "I(d, s) <+ (exp(g) - 1) * I(sense).",
        "g from V(g, s) and V(d, s) through abs(): pMOS bias and reverse-mode bias map onto the "
        "forward range. is_pmos is a module parameter.",
        f"Each seed's tanh layers written out; 5 seeds; {ex['lines']} lines for the full model.",
        "OpenVAF 23.5.0 -> OSDI; ngspice-41 built with --enable-osdi; 'pre_osdi' in .control.",
        f"Checks: g of the Verilog-A arithmetic vs Python, 200 random bias points: max |dg| "
        f"{ex['max_abs_g_diff_va_vs_python']:.1e}. Device RRMS, SPICE-native vs NGSpice card x "
        f"exp(g) in Python: max |d| {dmax:.1e} over 18 devices.",
    ], left=6.75, top=0.95, width=6.45, height=6.4, size=13)


def cost_slide(prs, nums):
    st = nums["stages"]["stages"]
    tm = nums["timing_all18"]
    rf = nums["refit"]
    rows = [["Stage", "NGSpice simulations", "Wall time (run)", "Output"]]
    rows.append(["Sensitivity screen, 115 candidates", "-", "256 s", "43 parameters"])
    for r in st[1:]:
        rows.append([f"Starting cards: {r['label']}", f"{r['calls']:,.0f} per device",
                     f"{r['wall_s']:,.0f} s", f"all-device {r['all_device']:.4f}"])
    rows += [
        ["Records (starting card on the measured grid)", "1 per device", "-", "25,086 points"],
        ["T1 tables, 18 folds", "0", f"{tm['t1_fit_s']:.0f} s", "18 tables"],
        ["g training, 19 models x 5 seeds", "0", f"{tm['train_s']:.0f} s", "19 models"],
        ["Held-out refits, 3 arms x 18", f"{rf['n_sim']['M_bias'][2]:.0f} per refit (mean, M_bias)",
         f"{tm['heldout_refit_s']:.0f} s", "54 cards"],
    ]
    slide = new_slide(prs, "Cost per stage", size=24)
    add_table(slide, rows, [4.6, 3.0, 1.8, 2.6], left=0.6, top=1.0, size=13, row_h=0.42,
              align_right_from=1)
    add_lines(slide, [
        "Wall time: elapsed time of each run (process pool; the pool size is not stored in the "
        "outputs). Simulations: mean per device over 18 devices.",
    ], top=1.0 + 0.42 * len(rows) + 0.25, height=0.9, size=13)


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
        "'+' = on top of the starting card; direct NGSpice rounds: calls in that round.  "
        "(e) Starting card scored with the BSIM4 A2 clamp written into the card (I13-1); the earlier "
        "rows are the stored scores (old deck, state L; 5 devices have a2 > 1 in the 43-parameter cards).",
    ], top=6.55, height=0.9, size=9)


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
    for lab, src in (("I3-2 run (2026-10-20, state L), g trained on 11 devices", d12),
                     ("All-18 run (state C), g trained on 17 devices", pd)):
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
        "C0_refit does not depend on the training devices: on the 12 devices the two runs differ only "
        "through the BSIM4 clamp state (state L vs C, devices with a2 > 1; max |difference| "
        f"{a18['reproduction'].get('C0_refit_vs_stored_I2_1_on_REAL_DEV', float('nan')):.1e}).",
    ], top=3.6, height=2.5, size=15)


def clamp_slide(prs):
    """I13-1: BSIM4 A2 clamp state, issue and effect (old state L vs new state C)."""
    s = jload(I13 / "summary.json")
    T, G = s["table"], s["gmr"]
    slide = new_slide(prs, "BSIM4 A2 clamp: one model state per card (I13-1)", size=24)
    add_lines(slide, [
        "ngspice-41 b4check.c checks each BSIM4 model once per process (static model name) and sets "
        "A2 = 1 and A1 = 0 when the size-dependent A2 > 1; b4temp.c rebuilds the size-dependent "
        "parameters at every analysis.",
        "The measured-curve deck runs one dc analysis per curve in one process. For a card with "
        "a2 > 1 the first curve was simulated with A2 = 1, the other curves with the card a2 "
        "(state L, all stored 43-parameter scores before I13-1).",
        "a2 > 1 in every 43-parameter arm on nmos_L0p19_W7 (1.877), nmos_L0p25_W1p6 (1.045), "
        "nmos_L20_W0p64 (2.002), nmos_L8_W1p6 (1.062), pmos_L0p35_W1p6 (1.253). Published cards: a2 <= 0.467.",
        "Fix: the clamp is written into the card (state C: every curve as in a fresh process). "
        "All numbers in this deck are state C; refit and training reruns, starting cards not refit.",
    ], top=0.95, height=2.6, size=13.5)
    lab = {"published": "Published 77 K card", "start": "Starting card (not refit)",
           "C0_refit": "C0_refit", "T1_inloop": "T1_inloop", "M_bias_v1": "M_bias",
           "M_bias_v2": "M_bias v2 (symmetric inputs, I12-1)"}
    rows = [["Arm (18 devices)", "Old, state L: all / comb / median", "New, state C: all / comb / median",
             "New - old (all-device)"]]
    for a in ("published", "start", "C0_refit", "T1_inloop", "M_bias_v1", "M_bias_v2"):
        o, n = T[a]["old"], T[a]["new"]
        rows.append([lab[a], f"{o['all_device_mean']:.4f} / {o['combined']:.4f} / {o['median']:.4f}",
                     f"{n['all_device_mean']:.4f} / {n['combined']:.4f} / {n['median']:.4f}",
                     f"{n['all_device_mean'] - o['all_device_mean']:+.4f}"])
    for k, name in (("M_bias_v1_vs_C0_refit", "GMR M_bias / C0_refit"),
                    ("M_bias_v1_vs_T1_inloop", "GMR M_bias / T1_inloop")):
        o, n = G[k]["old"], G[k]["new"]
        rows.append([name, f"{o['ratio']:.3f} [{o['ci_low']:.3f}, {o['ci_high']:.3f}], {o['wins']}/18",
                     f"{n['ratio']:.3f} [{n['ci_low']:.3f}, {n['ci_high']:.3f}], {n['wins']}/18", ""])
    add_table(slide, rows, [3.3, 3.4, 3.4, 1.9], left=0.6, top=3.6, size=11.5, row_h=0.36,
              align_right_from=1)


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
            ["I3-2 run, 12 fold models (F1c, state L)", "12",
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


def selected_variant_slide(prs):
    """g v2 with the nested-selected penalty weight (I12-1, I14-1, I15-2, I16-1):
    definition and 18-device accuracy."""
    d152 = jload(EXP / "I15_2/decision.json")
    acc16 = jload(EXP / "I16_1/accuracy/accuracy.json")
    a18 = jload(ALL18 / "results.json")["per_device"]
    lodo = jload(I13 / "lodo/results.json")["per_device"]
    ref = jload(ROOT / "out/research/reference_scores.json")["released"]["devices"]
    from cryoml.explore.i3_2_bias import paired_ratio
    v1 = a18["M_bias"]
    arms = (("Published 77 K card", {t: float(v["rrms"]) for t, v in ref.items()}),
            ("C0_refit (refit, g = 0)", lodo["C0_refit"]),
            ("M_bias v1 (abs inputs, no penalty)", v1),
            ("g v2, lambda = 10 V (I12-1)", lodo["M_bias_v2"]),
            ("g v2, selected lambda (I15-2 rows)", d152["selected_rows"]),
            ("g v2, selected lambda, element deck (I16-1)", acc16["native"]))
    rows = [["Arm (18 devices, held-out)", "All-device", "Combined", "Median", "GMR vs M_bias v1 [95% CI]",
             "Lower than v1"]]
    for name, dd in arms:
        a = agg(dd)
        if dd is v1:
            g, w = "1", "-"
        else:
            p = paired_ratio(dd, v1)
            g, w = f"{p['ratio']:.3f} [{p['ci_low']:.3f}, {p['ci_high']:.3f}]", f"{p['wins']}/18"
        rows.append([name, f4(a["all"]), f4(a["combined"]), f4(a["median"]), g, w])
    ls = d152["lambda_star"]
    cnt = {}
    for t, v in ls.items():
        if t != "full":
            cnt[v] = cnt.get(v, 0) + 1
    slide = new_slide(prs, "g v2: symmetric inputs and a penalty weight selected on training devices", size=24)
    add_lines(slide, [
        "Inputs (d/s symmetric): x1 = (s V(g,s) + s V(g,d) + m) / (2 VDD), x2 = m / VDD, m = sqrt(V(d,s)^2 + EPS^2), "
        "EPS = 0.01 V, s = +1 (nMOS) / -1 (pMOS), VDD = 1.8 V. Exchanging d and s leaves x1 and x2 unchanged. "
        "On forward-bias training data x1 = |V(g,s)| / VDD and x2 = |V(d,s)| / VDD (the v1 inputs) for EPS = 0.",
        "Training: v1 loss + lambda x P. P = mean over training cards and 25 mV forward-grid points of "
        "relu(-(gm/I + dg/dV(g,s))) + relu(-(gds/I + dg/dV(d,s))), the negative part of the corrected gm and gds.",
        "Element: I(d, s) <+ (exp(g) - 1) I_rep. I_rep is the drain current of a replica BSIM4 instance (same card, "
        "no charge, no substrate, GIDL, gate or junction current), sensed by a 0 V source; a -gmin VCCS removes "
        "the replica junction gmin current.",
        f"lambda selection, training devices only: per held-out device, 3 inner folds of the 17 training devices, "
        f"lambda in {{0, 1, 2, 5, 10}} V, lambda* = argmin mean ln RRMS of the inner held-out refits. lambda* = "
        + ", ".join(f"{k:g} V on {v}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]))
        + f" of 18 folds; full-data model (deployed) lambda* = {ls['full']:g} V.",
    ], top=0.85, height=3.35, size=12.5)
    add_table(slide, rows, [3.9, 1.15, 1.15, 1.15, 2.6, 1.3], left=0.6, top=4.25, size=11.5, row_h=0.33,
              align_right_from=1)
    add_lines(slide, [
        "Strict paper RRMS, curve inclusion frozen to the published card; leave-one-device-out, state C. "
        "GMR: geometric-mean ratio, device bootstrap 10,000. Element deck: fold model as an OSDI module with "
        f"its held-out card (max per-device |element - Python row| {acc16['max_abs_native_minus_stored']:.1e}).",
    ], top=4.25 + 0.33 * len(rows) + 0.08, height=0.75, size=10.5)


def selected_deploy_slide(prs):
    """I16-1 Part B checks of the deployed selected model vs I15-1 g v2 and v1."""
    dec = jload(EXP / "I16_1/decision.json")
    s16 = jload(EXP / "I16_1/deploy_sel/summary.json")
    rest = jload(EXP / "I16_1/deploy_sel/restrictions.json")
    s2 = jload(EXP / "I15_1/deploy_v2/summary.json")
    s1 = jload(EXP / "I15_1/deploy_v1/summary.json")
    e = lambda x: f"{x:.1e}" if (x != 0 and (abs(x) < 1e-2 or abs(x) >= 1e4)) else f"{x:.3g}"
    checks = (("q1_tight_viol", "Q1 equivalence violations, tight", "0", lambda v: v == 0),
              ("q1_bad", "Q1 runs: convergence lines / non-finite", "0", lambda v: v == 0),
              ("q1_vds0_max", "|I(Vds = 0)| (A)", "<= 1e-12", lambda v: v <= 1e-12),
              ("g_text_max", "Verilog-A g vs Python", "<= 1e-9", lambda v: v <= 1e-9),
              ("g_sym_text_max", "Verilog-A g, d/s exchanged", "<= 1e-12", lambda v: v <= 1e-12),
              ("swap_asym_max", "exchange asymmetry", "<= 1e-3", lambda v: v <= 1e-3),
              ("q2_ac_max", "Q2 gm / gds vs element identity", "<= 1e-3", lambda v: v <= 1e-3),
              ("cap_max", "Q2 capacitances", "<= 1e-3", lambda v: v <= 1e-3),
              ("q2_gm_sign", "gm curves with a sign change (of 54)", "0", lambda v: v == 0),
              ("q3_fail", "circuits failing only with the element", "0", lambda v: v == 0),
              ("q1_default_max", "default tolerance, relative (restriction)", "<= 1e-3", lambda v: v <= 1e-3),
              ("q2_gds_sign", "gds curves with a sign change, of 90 (restriction)", "0", lambda v: v == 0),
              ("runtime_ratio_max", "wall-time ratio element / card (restriction)", "<= 2", lambda v: v <= 2))
    rows = [["Check", "Threshold", "g v2, selected lambda (I16-1)", "g v2, lambda 10 V (I15-1)",
             "M_bias v1 (I15-1)"]]
    for k, lab, thr, ok in checks:
        fm = (lambda x: f"{x:.4e}") if k == "q1_vds0_max" else e
        bad = "restriction" if "(restriction)" in lab else "FAIL"
        rows.append([lab, thr] + [f"{fm(s['values'][k])} {'pass' if ok(s['values'][k]) else bad}"
                                  for s in (s16, s2, s1)])
    rows.append(["Label", ""] + [s["decision"]["deploy"] for s in (s16, s2, s1)])
    slide = new_slide(prs, "Deployment checks of the selected g v2 (I16-1, pre-registered)", size=24)
    add_table(slide, rows, [4.3, 1.1, 2.55, 2.55, 2.0], left=0.4, top=0.8, size=10.5, row_h=0.27,
              align_right_from=1)
    tot = rest["sel"]["totals"]
    gds_dev = [f"{t} ({v['n_gds_introduced']})" for t, v in sorted(rest["sel"]["per_device"].items())
               if v["n_gds_introduced"]]
    intro = [c for v in rest["sel"]["per_device"].values() for c in v["gds"] if c["introduced"]]
    n_el, n_py, n_cd = (sum(c[k] for c in intro) for k in ("n_neg_elem", "n_neg_python", "n_neg_card"))
    add_lines(slide, [
        f"Outcome (pre-registered rule): {dec['outcome']}; label {s16['decision']['deploy']}. Core failures: "
        + (", ".join(s16["decision"]["core_fail"]) or "none") + ". Restrictions: "
        + ("; ".join(s16["decision"]["restrictions"]) or "none") + ".",
        f"gds curves with an introduced sign change: {tot['n_gds_curves_introduced']} of 90 ("
        + (", ".join(gds_dev) or "none") + f"). On these curves the element has {n_el} points with dI/dV(d,s) < 0, "
        f"I_card exp(g) of the card sweep {n_py}, the card alone {n_cd}: the sign changes come from g. "
        "A gds sign change is a restriction, a gm sign change in strong inversion a core failure (I15-1).",
        f"Failed core check: |I(Vds = 0)| = {s16['values']['q1_vds0_max']:.4e} A at 1 of 2700 points (pmos_L8_W0p84, "
        "|Vgs| = 0.825 V, default tolerance, ngspice abstol 1e-12 A); card + replica run at that point 8e-24 A; "
        "tight-tolerance maximum 4.9e-17 A (post hoc diagnosis).",
        "M2 card: 43-parameter card refit from the starting card with the full-data g in the loop. Q1 grids: "
        "V(g,s), V(d,s) in 25 mV steps, V(d,s) swept in serpentine order; tight: reltol 1e-9.",
    ], top=0.8 + 0.27 * len(rows) + 0.1, height=2.4, size=11)


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
    nums = jload(NUM)
    title_slide(prs, a18)
    data_slide(prs, check)
    overview_slide(prs, nums, a18)
    data_metric_slide(prs, nums)
    screen_slide(prs, nums)
    box_slide(prs, nums)
    search_slide(prs, nums)
    stages_slide(prs, nums)
    records_slide(prs, nums)
    network_slide(prs, nums)
    training_slide(prs, nums)
    gfold_slide(prs, nums)
    refit_slide(prs, nums)
    refit_effect_slide(prs, nums)
    loso_slide(prs, nums)
    controls_slide(prs, nums)
    va_slide(prs, nums, check)
    cost_slide(prs, nums)
    performance_slide(prs, rows)
    per_device_slide(prs, check, a18)
    clamp_slide(prs)
    dev12_slide(prs, a18)
    bar_slide(prs)
    deploy_slide(prs, check)
    selected_variant_slide(prs)
    selected_deploy_slide(prs)
    gmap_slide(prs)
    iv_slides(prs, check)
    DECK.parent.mkdir(parents=True, exist_ok=True)
    prs.save(DECK)
    print(f"wrote {DECK} ({len(prs.slides._sldIdLst)} slides)")
    for w in FIT_WARNINGS:
        print("FIT:", w)


if __name__ == "__main__":
    main()
