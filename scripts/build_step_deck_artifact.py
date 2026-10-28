#!/usr/bin/env python3
"""Build the step-by-step Slides artifact (https://claude.ai/artifact/VXzexFpptoGiwaSzpVF57r): basic workflow, card only.

The deck shows the parameter-extraction workflow for the unchanged model (standard BSIM4 in NGSpice 41; only parameter
values change), step by step, one idea per slide.  The final card of each parameter set is the best Step A start per
device refit once more by least squares (stored as C0_refit in out/research/explore/I25_1/arm{A,B}/c0).

Inputs: out/research/explore/I25_1/decision.json, I25_1/stepA/*.json, I25_1/screen_mult1/summary.json,
out/research/explore/step_deck/numbers_i25_card.json (written by scripts/make_step_deck_i25.py).  Figures are uploaded
to the artifact's asset store first; their /_blob/ ids are in BLOB, with the local file in LOCAL (used by --render).

  python3 scripts/build_step_deck_artifact.py [OUT_ROOT]            # write project/deck.json + project/slides/*.html
  python3 scripts/build_step_deck_artifact.py [OUT_ROOT] --render=DIR  # also write standalone HTML per slide to DIR
                                                                       # for a headless-browser layout check
Publish (Artifact tool): url = the artifact, root = OUT_ROOT, file_path = OUT_ROOT/project/deck.json,
files = every project/slides/<id>.html, plus null for removed slide files.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I25 = ROOT / "out/research/explore/I25_1"
REPO_PROJECT = ROOT / "slides/step_deck_artifact/project"
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT_ROOT = Path(ARGS[0]) if ARGS else ROOT / "slides/step_deck_artifact"
RENDER = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--render=")), None)

FIG = ROOT / "figs/step_deck"
BLOB = {  # asset ids in the artifact; LOCAL = the uploaded file
    "goal": ("0118aec709e8834e49204c883c014ab5", FIG / "i25_goal_stock.png"),
    "measured": ("1844405a05e791cedc2e08e8e320a132", FIG / "s01_measured.png"),
    "eq_rrms": ("42728e3500a3277bc0e28dbb5d75edc7", FIG / "eq_rrms.png"),
    "start": ("94d8dcf27e1e1ecb3c06b51b8b0d461f", FIG / "i25_start_card.png"),
    "paper_ex": ("db3b79dceb6be9f0c2e1cc1e7b33aa61", FIG / "i25_ex_paper.png"),
    "starts": ("48e34cebc391f0d4cd32a444ed3d7a23", FIG / "i25_ex_starts.png"),
    "cma": ("90ef5423cc737d41a66b67e21865d9ae", FIG / "i25_ex_cma.png"),
    "ls": ("0d9d39c0810f7704af40fe475b196da8", FIG / "i25_ex_ls.png"),
    "rounds_ex": ("febb23cdf0036fb4ef80be978f23c410", FIG / "i25_ex_rounds.png"),
    "screen": ("0b3894a877f07e1a4d766d99e3553279", FIG / "i25_screen.png"),
    "perdev": ("cc5157dee20ebc5a72e5119064b55397", FIG / "i25_per_device_card.png"),
    "iv_7p_1": ("7e82729a0da6a0cc451b90109b917312", FIG / "i25_iv_card_7p_1.png"),
    "iv_7p_2": ("674df2715072be671d840a25d40799c0", FIG / "i25_iv_card_7p_2.png"),
    "iv_7p_3": ("ab51bb2e3b74a7155fa8b7b0e823bc44", FIG / "i25_iv_card_7p_3.png"),
    "iv_45p_1": ("b74c6312641b315427349ba3818bfece", FIG / "i25_iv_card_45p_1.png"),
    "iv_45p_2": ("ec4ebbda9a6d6e78909bc18e09a25f9f", FIG / "i25_iv_card_45p_2.png"),
    "iv_45p_3": ("52914a90fb7c2bf54b9af407dbebb897", FIG / "i25_iv_card_45p_3.png"),
}
ORDER = ["title", "w01-goal", "w02-data", "w03-model", "w04-paper", "w05-workflow", "w05b-pseudo", "w05c-eval",
         "w06-step1", "w07-step2",
         "w08-step3", "w09-step4", "w10-step5", "w11-params45", "w12-results", "w13-perdevice",
         "s11-iv7-1", "s11-iv7-2", "s11-iv7-3", "s12-iv45-1", "s12-iv45-2", "s12-iv45-3"]
OLD = ["b1-transistor", "b2-model", "b3-fit", "s01-measured", "s02-stock", "s03-paper", "s04-setup", "s04-stepA",
       "s04b-params", "s04c-why", "s04d-45vs43", "s09-results", "s09b-gmr", "s09c-stages", "s10-perdevice",
       "s10b-nmos", "s10c-pmos", "s13-remaining", "s13b-lowvds", "s15-cost", "s16-sources", "s17-limits"]
SECTIONS = {
    "s1": {"description": "Title", "start": "title"},
    "s2": {"description": "Goal, measured data, model and error, the paper's card", "start": "w01-goal"},
    "s3": {"description": "Our workflow: nested loops, pseudocode, one evaluation, steps 1-5", "start": "w05-workflow"},
    "s4": {"description": "Results: table, per device, I-V curves", "start": "w12-results"},
}

INK, MUTED, PAPER, CARD, LINE = "#1f252b", "#5a6168", "#f6f5f1", "#ffffff", "#c9ced3"
BLUE, GREEN, ORANGE = "#2a6fdb", "#2e9e6a", "#d9822b"
F = "font-family:'IBM Plex Sans', Arial, sans-serif"
SEC = (f"background:{PAPER};color:{INK};{F};padding:64px 96px 56px;display:flex;flex-direction:column;"
       "gap:{g}px")
H2 = f'<h2 style="font-size:52px;font-weight:600;line-height:1.1;color:{INK}">{{}}</h2>'


def ul(items, size=30, lh=1.35, extra=""):
    return (f'<ul style="font-size:{size}px;line-height:{lh};color:{INK}{extra}">'
            + "".join(f"<li>{i}</li>" for i in items) + "</ul>")


def img(key, alt, w, h):
    return (f'<div style="display:flex;flex-direction:row;justify-content:center;background:{CARD};'
            f'border:1px solid {LINE}"><img src="/_blob/{BLOB[key][0]}" alt="{alt}" '
            f'style="width:{w}px;height:{h}px;object-fit:contain"></div>')


def table(head, widths, rows, size=28, hl=(), width=1728):
    s = (f'<table style="width:{width}px;font-size:{size}px;color:{INK};background:{CARD};'
         f'border:1px solid {LINE}">')
    s += '<tr style="background:#e4e7ea">' + "".join(f'<th style="width:{w}%">{h}</th>' for h, w in zip(head, widths)) + "</tr>"
    for i, r in enumerate(rows):
        bg = ' style="background:#fbe8de"' if i in hl else ""
        s += f"<tr{bg}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return s + "</table>"


def small(text):
    return f'<p style="font-size:24px;line-height:1.3;color:{MUTED}">{text}</p>'


def sec(sid, body, notes, gap=28, style=None):
    st = style or SEC.format(g=gap)
    return f'<section id="{sid}" style="{st}">\n' + "\n".join(body) + f"\n<aside>{notes}</aside>\n</section>\n"


def svg_boxes(boxes, arrows, w, h, label):
    """boxes: (x, y, bw, bh, [lines], stroke, fill); arrows: (x1, y1, x2, y2[, text, tx, ty]).  Plain SVG
    (rect, line, polygon, text), font sizes >= 24 px."""
    out = [f'<svg aria-label="{label}" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
           f'xmlns="http://www.w3.org/2000/svg">']
    for x, y, bw, bh, lines, stroke, fill in boxes:
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{fill}" stroke="{stroke}" '
                   f'stroke-width="3"/>')
        n = len(lines)
        for i, (t, size, bold) in enumerate(lines):
            ty = y + bh / 2 + (i - (n - 1) / 2) * (size * 1.25) + size * 0.35
            fw = ' font-weight="600"' if bold else ""
            out.append(f'<text x="{x + bw / 2}" y="{ty:.0f}" text-anchor="middle" font-size="{size}"{fw} '
                       f'font-family="IBM Plex Sans, Arial, sans-serif" fill="{INK}">{t}</text>')
    for a in arrows:
        x1, y1, x2, y2 = a[:4]
        dash = ' stroke-dasharray="10 8"' if len(a) > 7 and a[7] else ""
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4a5259" stroke-width="3"{dash}/>')
        import math
        if len(a) > 8 and not a[8]:
            continue
        ang = math.atan2(y2 - y1, x2 - x1)
        p = [(x2, y2), (x2 - 16 * math.cos(ang) + 8 * math.sin(ang), y2 - 16 * math.sin(ang) - 8 * math.cos(ang)),
             (x2 - 16 * math.cos(ang) - 8 * math.sin(ang), y2 - 16 * math.sin(ang) + 8 * math.cos(ang))]
        out.append('<polygon points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in p) + '" fill="#4a5259"/>')
        if len(a) > 4 and a[4]:  # noqa: SIM102
            out.append(f'<text x="{a[5]}" y="{a[6]}" text-anchor="middle" font-size="26" '
                       f'font-family="IBM Plex Sans, Arial, sans-serif" fill="{MUTED}">{a[4]}</text>')
    out.append("</svg>")
    return "".join(out)


def f4(x):
    return f"{x:.4f}"


def kint(x):
    return f"{x:,.0f}"


def load():
    d = json.loads((I25 / "decision.json").read_text())
    assert d["validity"]["valid"], "I25-1 decision.json not valid"
    nums = json.loads((ROOT / "out/research/explore/step_deck/numbers_i25_card.json").read_text())
    screen = json.loads((I25 / "screen_mult1/summary.json").read_text())
    rounds = {}
    for arm, pre in (("A", "seven"), ("B", "full")):
        r = {"cma": [], "ls": [], "fd": []}
        for k in range(3):
            for v in json.loads((I25 / f"stepA/{pre}_s{k}_fd.json").read_text())["devices"].values():
                r["fd"].append(v["n_sim"])
        for st in ("r1", "r2", "r3"):
            for k in range(3):
                for v in json.loads((I25 / f"stepA/{pre}_s{k}_{st}.json").read_text())["devices"].values():
                    r["cma"].append(v["cmaes_evals"])
                    r["ls"].append(v["n_sim"] - v["cmaes_evals"] - (v.get("n_sim_start_draws") or 0))
        log = json.loads((I25 / f"stepA/{pre}_chain_log.json").read_text())
        red = {k: v["draws"] for k, v in log["pairs"].items() if not k.startswith("center") and v.get("draws", 0) > 0}
        r["redraw_pairs"] = len(red)
        r["cma_mean"] = sum(r["cma"]) / len(r["cma"])
        r["ls_mean"] = sum(r["ls"]) / len(r["ls"])
        r["fd_mean"] = sum(r["fd"]) / len(r["fd"])
        rounds[arm] = r
    return d, nums, screen, rounds


def build() -> dict:
    d, nums, screen, rnd = load()
    A, B = d["arms"]["A"], d["arms"]["B"]
    ref = d["reference"]
    stock, m1, pap = ref["stock_77K"]["primary"], ref["mult1_no_fit"]["primary"], ref["paper"]["primary"]
    cA_, cB_ = A["summary"]["C0_refit"]["primary"], B["summary"]["C0_refit"]["primary"]
    cA, cB = A["calls_per_device"], B["calls_per_device"]
    rA, rB = d["gmr"]["A"]["primary"]["C0_refit_vs_paper"], d["gmr"]["B"]["primary"]["C0_refit_vs_paper"]
    st = {r["stage"]: r for r in d["stage_table"]}
    spec = d["specs"]
    nfit = {t: 45 - len(v) for t, v in spec["fixed_zero_center_by_device"].items()}
    wf = nums["workflow"]
    ov = spec["overlap_with_old43"]

    # checks of every number written below against the brief / sources
    for k, (a, c, row) in {"paper": (0.2751, 0.2595, pap), "7p": (0.1546, 0.1458, cA_),
                           "45p": (0.1094, 0.1026, cB_)}.items():
        assert round(row["all_device"], 4) == a and round(row["combined"], 4) == c, (k, row)
    assert round(m1["all_device"], 1) == 12.1 and round(stock["all_device"], 2) == 3.42
    assert (round(rA["ratio"], 2), round(rB["ratio"], 2)) == (0.54, 0.36) and rA["wins"] == rB["wins"] == 18
    assert screen["candidate_count"] == 115 and screen["selected_count"] == 45
    assert screen["effective_threshold"] == 1e-4 and screen["perturbation_fraction"] == 0.1
    assert ov["only_old"] == ["cdscd"] and sorted(ov["only_new"]) == ["cdsc", "dvt0w", "pdiblc1"] and ov["n_old"] == 43
    assert (rnd["A"]["redraw_pairs"], rnd["B"]["redraw_pairs"]) == (19, 22)
    assert wf["spread_45p_gt10pct"] == 15
    S = {}

    S["title"] = sec("title", [
        '<h1 style="font-size:80px;font-weight:600;line-height:1.12;color:#f1f0eb">Fitting BSIM4 parameters to '
        'SKY130 transistors at 77 K</h1>',
        '<div style="display:flex;flex-direction:column;gap:16px">' + "".join(
            f'<p style="font-size:34px;line-height:1.3">{t}</p>' for t in [
                "18 measured transistors (8 nMOS, 10 pMOS), simulated in NGSpice",
                "Compared with the paper's published 77 K card",
                '<span style="color:#b8bec4">2026-10-28</span>']) + "</div>"],
        "This talk shows how we fit BSIM4 model parameters to measured 77 K data of 18 SkyWater 130 nm transistors. "
        "The model and the simulator stay unchanged; only parameter values change. The result is compared with the "
        "card published in the paper.",
        style=f"background:{INK};color:#f1f0eb;{F};padding:128px 160px;display:flex;flex-direction:column;"
              "justify-content:center;gap:56px")

    S["w01-goal"] = sec("w01-goal", [
        H2.format("Goal: simulated curves that match measurement"),
        img("goal", "pMOS L 0.35 um x W 5 um at 77 K: measured points and the stock SkyWater card simulated in "
                    "NGSpice", 1640, 636),
        ul(["Card: file of BSIM4 parameter values, one set per transistor size",
            "Stock SkyWater card was fitted at room temperature; here simulated at 77 K",
            "Goal: parameter values so that NGSpice matches the 77 K measurement"])],
        "A transistor model card is a list of parameter values for the BSIM4 equations. The stock SkyWater card was "
        "made for room temperature. Simulated at 77 K, its current differs from the measurement, as in this pMOS. "
        "The goal is to find parameter values that make the simulation match the 77 K data.", gap=24)

    S["w02-data"] = sec("w02-data", [
        H2.format("Measured data: 18 transistors at 77 K"),
        img("measured", "Measured Id-Vd and Id-Vg of pMOS L 0.35 um x W 5 um at 77 K", 1640, 560),
        ul(["8 nMOS, 10 pMOS; channel length L 0.15-100 um, width W 0.42-100 um",
            "Id-Vd: drain current vs drain voltage, 5 gate voltages",
            "Id-Vg: drain current vs gate voltage, 6 drain voltages",
            "Faded points: below the paper's current floor, not used in the error"])],
        "The data are the paper's 77 K measurements. Each transistor has five Id-Vd curves and six Id-Vg curves. "
        "For pMOS the voltages and currents are negative; plots show absolute values. Points below the current floor "
        "are not used.", gap=22)

    model_svg = svg_boxes(
        [(0, 20, 330, 130, [("Card", 30, True), ("BSIM4 parameter values", 26, False)], BLUE, CARD),
         (430, 20, 330, 130, [("NGSpice", 30, True), ("circuit simulator", 26, False)], BLUE, CARD),
         (860, 20, 360, 130, [("Simulated curves", 30, True), ("Id at measured voltages", 26, False)], BLUE, CARD),
         (860, 200, 360, 110, [("Measured curves", 30, True), ("11 per device", 26, False)], INK, CARD),
         (1330, 105, 398, 130, [("Error: RRMS", 30, True), ("simulated vs measured", 26, False)], "#c8384e",
          "#fbe8de")],
        [(330, 85, 426, 85), (760, 85, 856, 85), (1220, 85, 1326, 150), (1220, 255, 1326, 195)],
        1728, 320, "Card to NGSpice to simulated curves; simulated and measured curves give the error RRMS")
    S["w03-model"] = sec("w03-model", [
        H2.format("Model and error measure"),
        model_svg,
        f'<div style="display:flex;flex-direction:row;gap:40px;align-items:center">'
        f'<div style="background:{CARD};border:1px solid {LINE};padding:12px"><img src="/_blob/{BLOB["eq_rrms"][0]}" '
        f'alt="RRMS of a curve and of a device" style="width:720px;height:156px;object-fit:contain"></div>'
        + ul(["RRMS of curve c: RMS current error / mean measured current",
              "Device RRMS: mean over its curves",
              "All-device RRMS: mean over the 18 devices",
              "Same definition and curves as the paper"], 28, extra=";width:960px") + "</div>"],
        "The card goes into NGSpice, which simulates the drain current at the measured voltages. The error of one "
        "curve is the RMS difference between simulated and measured current, divided by the mean measured current. "
        "A device's RRMS is the mean over its curves, and the all-device RRMS is the mean over 18 devices. This is "
        "the paper's definition, unchanged.", gap=36)

    seven = [("VTH0", "threshold voltage"), ("U0", "low-field mobility"), ("NFACTOR", "subthreshold slope factor"),
             ("VSAT", "saturation velocity"), ("DELTA", "linear-to-saturation smoothing"),
             ("RDSW", "source/drain series resistance"), ("ETA0", "threshold shift with drain voltage")]
    ex = wf["example"]
    S["w04-paper"] = sec("w04-paper", [
        H2.format("The paper's card: 7 tuned parameters"),
        '<div style="display:flex;flex-direction:row;gap:32px;align-items:center">'
        + table(["Parameter", "Meaning"], [36, 64], [[f"<b>{p}</b>", m] for p, m in seven], 24, width=600)
        + img("paper_ex", "pMOS L 0.35 um x W 5 um: measured points and the paper card simulated in NGSpice",
              1060, 410) + "</div>",
        ul(["The paper changed these 7 parameters per device; all others unchanged",
            f"Figure: one pMOS (L 0.35 um x W 5 um), paper card; device RRMS {ex['paper_rrms']:.3f}",
            f"All 18 devices: all-device RRMS {f4(pap['all_device'])}",
            "Our first run fits the same 7 parameters"], 28)],
        "The paper's card changes seven BSIM4 parameters for each device and leaves the rest of the card file "
        "unchanged. The figure shows the paper card on the pMOS that the step slides use as the example. Over the "
        "18 devices the paper card has an all-device RRMS of 0.2751. Our first run fits the same seven parameters.",
        gap=30)

    # ---------------------------------------------------------------- run counts per box (means over devices)
    def r10(v):
        return f"{int(round(v / 10.0) * 10):,}"
    cma = (rnd["A"]["cma_mean"], rnd["B"]["cma_mean"])
    lsr = (rnd["A"]["ls_mean"], rnd["B"]["ls_mean"])
    fdr = (rnd["A"]["fd_mean"], rnd["B"]["fd_mean"])
    ref_ = (cA["C0_refit"] - cA["starting_card"], cB["C0_refit"] - cB["starting_card"])
    runs = {"cma": f"{r10(cma[0])} / {r10(cma[1])} runs", "ls": f"{r10(lsr[0])} / {r10(lsr[1])} runs",
            "fd": f"{r10(fdr[0])} / {r10(fdr[1])} runs", "ref": f"{r10(ref_[0])} / {r10(ref_[1])} runs"}

    def T(x, y, text, size=26, bold=False, anchor="start", color=INK):
        fw = ' font-weight="600"' if bold else ""
        return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}"{fw} '
                f'font-family="IBM Plex Sans, Arial, sans-serif" fill="{color}">{text}</text>')

    def R(x, y, w, h, stroke, fill=CARD, dash=False, rx=10):
        d = ' stroke-dasharray="12 8"' if dash else ""
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="3"{d}/>')

    def A_(x1, y1, x2, y2):
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        p = [(x2, y2), (x2 - 16 * math.cos(ang) + 8 * math.sin(ang), y2 - 16 * math.sin(ang) - 8 * math.cos(ang)),
             (x2 - 16 * math.cos(ang) - 8 * math.sin(ang), y2 - 16 * math.sin(ang) + 8 * math.cos(ang))]
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4a5259" stroke-width="3"/>'
                '<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in p) + '" fill="#4a5259"/>')

    def B(x, y, w, h, lines, stroke=BLUE, fill=CARD):
        out = [R(x, y, w, h, stroke, fill)]
        n = len(lines)
        for i, (t, sz, bold) in enumerate(lines):
            ty = y + h / 2 + (i - (n - 1) / 2) * (sz * 1.25) + sz * 0.35
            out.append(T(x + w / 2, f"{ty:.0f}", t, sz, bold, "middle"))
        return "".join(out)

    def svg(w, h, label, parts):
        return (f'<svg aria-label="{label}" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
                'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>")

    nest = svg(1728, 560, "Nested loops: for each device, for each of 3 starts, rounds 1-3 of range, CMA-ES and "
                          "least squares, one more least-squares pass, keep the best start, final refit", [
        R(2, 2, 1724, 556, INK),
        T(24, 40, "For each of the 18 devices (fitted separately)", 28, True),
        R(22, 62, 1120, 480, BLUE, "#f3f7fd", dash=True),
        T(42, 98, "For each of 3 random starts", 26, True, color=BLUE),
        B(42, 150, 250, 170, [("Random start", 26, True), ("in range R1;", 24, False),
                              ("redraw if NGSpice", 24, False), ("fails", 24, False)]),
        A_(292, 235, 334, 235),
        R(338, 116, 784, 250, "#4a5259", "#ffffff"),
        T(358, 150, "Rounds 1, 2, 3", 26, True),
        B(358, 170, 220, 170, [("Set range", 26, True), ("R1; later:", 24, False), ("centred on last", 24, False),
                               ("card, half width", 24, False)]),
        A_(578, 255, 616, 255),
        B(620, 170, 230, 170, [("Global search", 26, True), ("CMA-ES", 24, False), (runs["cma"], 24, False)]),
        A_(850, 255, 884, 255),
        B(888, 170, 220, 170, [("Least squares", 26, True), (runs["ls"], 24, False)]),
        A_(730, 366, 730, 402),
        B(560, 404, 340, 130, [("Least squares again", 26, True), ("after round 3", 24, False),
                               (runs["fd"], 24, False)]),
        A_(1142, 300, 1186, 300),
        B(1190, 120, 516, 110, [("Keep the start with", 26, True), ("the lowest device RRMS", 26, True)], INK),
        A_(1448, 230, 1448, 266),
        B(1190, 270, 516, 110, [("Final refit: least squares", 26, True), (runs["ref"], 24, False)], INK),
        A_(1448, 380, 1448, 416),
        B(1190, 420, 516, 100, [("Final card", 28, True)], "#c8384e", "#fbe8de"),
    ])
    S["w05-workflow"] = sec("w05-workflow", [
        H2.format("Our workflow: nested loops"),
        nest,
        ul(["Run counts: mean NGSpice runs per box, 7 / 45 parameters",
            f"Total per device, 3 starts included: {kint(cA['C0_refit'])} / {kint(cB['C0_refit'])} NGSpice runs",
            "Step slides use one pMOS (L 0.35 um x W 5 um), 45 parameters"], 28)],
        "Each device is fitted on its own. For each device there are three random starts. Each start goes through "
        "three rounds; a round sets the parameter range, runs the global search, then least squares. After round 3 "
        "one more least-squares pass follows. Then the start with the lowest RRMS is kept and refitted once more by "
        "least squares; the result is the final card. The numbers are mean NGSpice runs per box.", gap=22)

    pc = [("for device in 18 devices:", "fitted separately"),
          ("&nbsp;&nbsp;for start in 1, 2, 3:", ""),
          ("&nbsp;&nbsp;&nbsp;&nbsp;range = R1 around the start card", "step 1"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;x = random card in range", "step 2; redraw if NGSpice fails"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;for round in 1, 2, 3:", ""),
          ("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = CMA-ES(device RRMS, x, range)", f"step 3; {runs['cma']}"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = least_squares(residuals, x, range)", f"step 4; {runs['ls']}"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if round &lt; 3: range = centred on x, half width", "step 5"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;x = least_squares(residuals, x, range)", runs["fd"]),
          ("&nbsp;&nbsp;best = start with the lowest device RRMS", ""),
          ("&nbsp;&nbsp;final card = least_squares(residuals, best, range)", runs["ref"])]
    mono = "font-family:'Courier New', Courier, monospace"
    code = (f'<div style="display:flex;flex-direction:column;gap:6px;background:{CARD};border:1px solid {LINE};'
            f'padding:28px 32px">' + "".join(
                f'<div style="display:flex;flex-direction:row"><p style="font-size:28px;line-height:1.3;{mono};'
                f'width:1060px">{a}</p><p style="font-size:26px;line-height:1.3;color:{MUTED}">'
                f'{"# " + b if b else ""}</p></div>' for a, b in pc) + "</div>")
    S["w05b-pseudo"] = sec("w05b-pseudo", [
        H2.format("The same procedure as pseudocode"),
        code,
        ul(["Run counts: mean NGSpice runs, 7 / 45 parameters",
            "range: allowed interval of each parameter, on a log scale"], 26)],
        "This is the same procedure line by line. The comments give the step number used on the following slides "
        "and the mean number of NGSpice runs. The range is the allowed interval of each parameter; it is "
        "narrowed in rounds 2 and 3.", gap=24)

    ev = svg(1728, 150, "One evaluation: parameter values, write card, NGSpice, 11 simulated curves, compare with "
                        "measured, error", [
        B(0, 20, 250, 110, [("Parameter", 26, True), ("values", 26, True)]),
        A_(250, 75, 282, 75),
        B(286, 20, 230, 110, [("Write", 26, True), ("card file", 26, True)]),
        A_(516, 75, 548, 75),
        B(552, 20, 250, 110, [("NGSpice", 26, True), ("1 run", 24, False)]),
        A_(802, 75, 834, 75),
        B(838, 20, 270, 110, [("11 simulated", 26, True), ("curves", 26, True)]),
        A_(1108, 75, 1140, 75),
        B(1144, 20, 280, 110, [("Compare with", 26, True), ("measured curves", 26, True)], INK),
        A_(1424, 75, 1456, 75),
        B(1460, 20, 268, 110, [("Error", 28, True)], "#c8384e", "#fbe8de"),
    ])
    S["w05c-eval"] = sec("w05c-eval", [
        H2.format("One evaluation, used by both optimizers"),
        ev,
        ul(["CMA-ES minimizes one number per card: the device RRMS",
            "Least squares minimizes Σ r_i² over all scored points i of the device",
            "r_i = w_c (I_sim,i − I_meas,i) / (den_c "
            "√(n_c C)): den_c = mean |I_meas| of curve c, n_c its "
            "points, C curves",
            "w_c = 1 in pass 1; pass 2: w_c ∝ 1/√RRMS_c. "
            "With w_c = 1, Σ r_i² = mean of RRMS_c²"], 28)],
        "Every evaluation writes the parameter values into the card, runs NGSpice once for all measured biases of "
        "the device, and compares the 11 simulated curves with the measured ones. CMA-ES uses the device RRMS, a "
        "single number. Least squares uses the vector of point residuals; with unit weights its sum of squares is "
        "the mean of the squared curve RRMS. Least squares runs two passes; the second weights each curve by one "
        "over the square root of its RRMS. A curve NGSpice cannot simulate is given a fixed error of 10.", gap=30)

    # ---------------------------------------------------------------- step strip and input/does/output block
    steps = ["1 Start card", "2 Random starts", "3 Global search", "4 Least squares", "5 Rounds, best start"]

    def strip(cur):
        parts, x, w = [], 0, 320
        for i, lab in enumerate(steps, 1):
            on = i == cur
            parts.append(R(x, 4, w, 60, "#c8384e" if on else LINE, "#fbe8de" if on else CARD, rx=8))
            parts.append(T(x + w / 2, 43, lab, 26, on, "middle", INK if on else MUTED))
            if i < 5:
                parts.append(A_(x + w + 2, 34, x + w + 26, 34))
            x += w + 32
        return svg(1728, 68, f"Workflow steps, step {cur} highlighted", parts)

    def io(inp, does, outp):
        return (f'<div style="display:flex;flex-direction:column;gap:4px;background:{CARD};border:1px solid {LINE};'
                f'padding:14px 24px">' + "".join(
                    f'<p style="font-size:26px;line-height:1.3"><b>{k}:</b> {v}</p>'
                    for k, v in (("Input", inp), ("Does", does), ("Output", outp))) + "</div>")

    STEP = SEC.format(g=14).replace("padding:64px 96px 56px", "padding:48px 96px 40px")

    S["w06-step1"] = sec("w06-step1", [
        H2.format("Step 1: start card"), strip(1),
        io("paper card file with the tuned values removed (all multipliers = 1)",
           "uses its parameter values as range centres; range R1 per parameter (step 2)",
           f"start card; all-device RRMS {m1['all_device']:.1f}"),
        img("start", "pMOS L 0.35 um x W 5 um: measured points and the start card simulated in NGSpice", 1400, 540),
        ul(["Figure: one pMOS (L 0.35 um x W 5 um), start card vs measured"], 26)],
        "The start card is the paper's card file with every tuned value set back to its base value, that is, all "
        "multipliers equal to 1. It keeps the file's 77 K settings and contains no fitted value from the paper. "
        f"Its all-device RRMS is {m1['all_device']:.1f}. Its values are the centres of the parameter ranges.",
        style=STEP)

    sr = ex["start_rrms"]
    S["w07-step2"] = sec("w07-step2", [
        H2.format("Step 2: three random start points"), strip(2),
        io("start card and range R1: VTH0 and added parameters 1/3-3 x, other 6 of the 7: 1/100-100 x",
           "draws each free parameter at random (uniform on a log scale); redraws if NGSpice fails",
           "3 start cards per device"),
        img("starts", "pMOS L 0.35 um x W 5 um: measured Id-Vd and Id-Vg at 1.85 V and the 3 random start cards "
                      "simulated in NGSpice", 1400, 540),
        ul([f"Figure: starts 1 and 2 give almost no current (RRMS {sr[0]:.2f}); start 3: {sr[2]:.1f}",
            "Reason for 3 starts: a search from one point can stop in a local minimum"], 26)],
        "Each device gets three random start cards inside range R1. A parameter is drawn uniformly on a log scale "
        "between its lower and upper limit. If NGSpice cannot simulate the card, for example because the effective "
        "mobility becomes negative, a new card is drawn. In the example, starts 1 and 2 give almost no current and "
        "start 3 gives a current with the wrong shape.", style=STEP)

    S["w08-step3"] = sec("w08-step3", [
        H2.format("Step 3: global search (CMA-ES)"), strip(3),
        io("start card (round 1) or previous card (rounds 2, 3) and the range",
           f"per generation: draws {ex['popsize']} cards (this pMOS), runs each, moves centre and spread toward "
           "lower device RRMS",
           f"lowest-RRMS card after {runs['cma'].replace(' runs', '')} runs (7 / 45 parameters)"),
        img("cma", "Round 1 CMA-ES on pMOS L 0.35 um x W 5 um: device RRMS of every candidate against NGSpice runs, "
                   "best so far per start; sampled VTH0 and U0 by generation", 1400, 540),
        ul(["Left: each dot one card (one run); line: lowest device RRMS so far",
            "Right: VTH0 and U0 drawn for start 2; later generations (yellow) cluster"], 26)],
        "The global search is CMA-ES. Each generation it draws a set of cards from a normal distribution in the "
        "scaled range, runs NGSpice for each, and moves the centre and spread of the distribution toward the cards "
        "with the lowest device RRMS. The left plot shows every card of round 1 and the lowest RRMS so far for the "
        "three starts. The right plot shows the sampled values of two parameters, which contract over the "
        f"generations. Lowest RRMS after round-1 CMA-ES: {ex['cmaes_best'][0]:.3f}, {ex['cmaes_best'][1]:.3f}, "
        f"{ex['cmaes_best'][2]:.3f}.", style=STEP)

    n_ = ex["n_active"]
    jac = svg(1728, 96, "Least-squares iteration: card, n plus 1 NGSpice runs, slope matrix, step", [
        B(0, 8, 300, 80, [("Card x", 26, True), (f"{n_} free parameters", 24, False)]),
        A_(300, 48, 336, 48),
        B(340, 8, 470, 80, [(f"{n_} + 1 NGSpice runs", 26, True), ("x, and x with one parameter changed", 22, False)]),
        A_(810, 48, 846, 48),
        B(850, 8, 430, 80, [("Slope matrix", 26, True), ("change of each r_i per parameter", 22, False)]),
        A_(1280, 48, 1316, 48),
        B(1320, 8, 408, 80, [("Step that lowers", 26, True), ("the sum of r_i² (+1 run)", 22, False)]),
    ])
    S["w09-step4"] = sec("w09-step4", [
        H2.format("Step 4: local refinement (least squares)"), strip(4),
        io("card from step 3 and the range",
           "slopes from small changes of each parameter; steps that lower the sum of squared residuals; 2 passes",
           f"refined card after {runs['ls'].replace(' runs', '')} runs (7 / 45 parameters)"),
        jac,
        img("ls", "Round 1 least squares on pMOS L 0.35 um x W 5 um: device RRMS after each iteration for 3 starts; "
                  "RRMS of each curve of start 2 before and after", 1300, 440),
        ul([f"Left: device RRMS per iteration; 1 iteration = {n_} + 2 runs",
            "Right: start 2, each curve before (grey) and after (blue)"], 26)],
        "Least squares refines the card from the global search. Each iteration runs NGSpice once at the current "
        "card and once with each free parameter changed by a small step, which gives the slope matrix, and then "
        "tries a step; that is the number of free parameters plus two runs. Some curves go down and some go up, "
        "because the method lowers the summed squared error over all points. Start 2 goes from "
        f"{ex['cmaes_best'][1]:.3f} to {ex['round1_rrms'][1]:.3f} in round 1.", style=STEP)

    rr = ex["rounds"]
    sel = rr["selected_start"]
    S["w10-step5"] = sec("w10-step5", [
        H2.format("Step 5: rounds 2, 3; keep the best start"), strip(5),
        io("card after round r",
           "range centred on it, half width on a log scale; repeat steps 3-4; after round 3 one more least "
           "squares; keep the best start; refit",
           f"final card; this pMOS: device RRMS {rr['final']:.3f}"),
        img("rounds_ex", "pMOS L 0.35 um x W 5 um: allowed range and value found for VTH0, U0, RDSW in rounds 1-3; "
                         "device RRMS after each round for the 3 starts", 1400, 556),
        ul([f"Left: range (bar) and value found (dot), start {sel + 1}; a value near the edge keeps the width",
            f"Right: starts end at {rr['rrms']['0'][-1]:.3f}, {rr['rrms']['1'][-1]:.3f}, {rr['rrms']['2'][-1]:.3f};"
            f" start {sel + 1} kept"], 26)],
        "In rounds 2 and 3 each parameter's range is centred on the value found in the previous round and its "
        "width is halved on a log scale; a parameter that ended within ten percent of a range edge keeps its "
        "width. After round 3 one more least-squares pass runs. The start with the lowest device RRMS is kept and "
        "refitted once more by least squares; that is the final card.", style=STEP)

    sc = wf["screen"]
    act = [v["n"] for v in wf["active_45p"].values()]
    assert (min(act), max(act)) == (37, 41)
    S["w11-params45"] = sec("w11-params45", [
        H2.format("Second run: 45 parameters from a screen"),
        img("screen", "Screen of 115 BSIM4 parameters: current change for a 10 % parameter change, threshold, kept "
                      "parameters, the paper's 7", 1660, 640),
        ul([f"Reason: with the paper's 7 parameters the all-device RRMS stays at {f4(cA_['all_device'])}",
            "Screen: change each of 115 parameters by 10 %; keep if the current changes by more than 0.01 %",
            f"45 kept (blue); fitted per device: {min(act)}-{max(act)} (zero and values &lt; 1e-14, e.g. UB, "
            "stay fixed)",
            "Earlier 43-parameter screen: CDSC, DVT0W, PDIBLC1 added here, CDSCD dropped"], 28)],
        "With the paper's seven parameters the error stops falling at 0.1546. For the second run we choose more "
        "parameters with a screen: each of 115 BSIM4 parameters is changed by ten percent, and it is kept if the "
        "simulated current changes by more than one part in ten thousand of the measured current. 45 pass. A "
        "parameter whose value is zero in a device's card stays zero, and the fit code also keeps values below "
        "about 1e-14 fixed: UB on all devices, B1 and CDSC on some. So 37 to 41 are fitted per device. Compared "
        "with an earlier screen that gave 43, three parameters were added and one removed.", gap=22)

    S["w12-results"] = sec("w12-results", [
        H2.format("Results: our cards and the paper card"),
        table(["Card (NGSpice)", "All-device RRMS", "Combined RRMS", "NGSpice runs per device"],
              [40, 20, 20, 20], [
            ["Paper card", f4(pap["all_device"]), f4(pap["combined"]), "-"],
            ["Our card, 7 parameters", f4(cA_["all_device"]), f4(cA_["combined"]), kint(cA["C0_refit"])],
            ["Our card, 45 parameters", f4(cB_["all_device"]), f4(cB_["combined"]), kint(cB["C0_refit"])]],
              32, hl=(1, 2)),
        ul(["All three: standard BSIM4 in NGSpice; only parameter values differ",
            f"Per device, our RRMS divided by the paper card's RRMS: typical value "
            f"{rA['ratio']:.2f} (7 parameters), {rB['ratio']:.2f} (45)",
            f"Lower than the paper card on {rA['wins']} of 18 devices (7) and {rB['wins']} of 18 (45)",
            "Combined RRMS: mean of the nMOS mean and the pMOS mean"]),
        small("Typical value = geometric mean over the 18 devices. NGSpice runs include all 3 starts.")],
        "This table compares the three cards on the same 18 devices and the same curves. With the same seven "
        "parameters as the paper, our card has about half the paper card's error on a typical device, and it is "
        "lower on every device. With 45 parameters it is about a third. All cards are the standard BSIM4 model in "
        "NGSpice.", gap=30)

    big = "pmos_L0p35_W0p55"
    pdA = A["per_device"]["C0_refit"]
    S["w13-perdevice"] = sec("w13-perdevice", [
        H2.format("RRMS per device"),
        '<div style="display:flex;flex-direction:row;gap:36px;align-items:center">'
        + img("perdev", "Per-device RRMS, log scale: paper card, our 7-parameter card, our 45-parameter card",
              1290, 800)
        + ul([f"Both our cards lower than the paper card on 18 of 18 devices",
              "Largest error: pMOS L 0.35 um x W 0.55 um"], 30, extra=";width:390px") + "</div>"],
        "Each device has three bars: the paper card, our seven-parameter card, and our 45-parameter card. Our cards "
        "are lower than the paper card on every device. The largest remaining error is the narrow pMOS with "
        f"L 0.35 um and W 0.55 um (RRMS {pdA[big] if not isinstance(pdA[big], dict) else pdA[big]['rrms']:.2f} with "
        "7 parameters).", gap=24)

    grp = ["nMOS L 0.15-8 um", "nMOS L 20, 100 um; pMOS L 0.35, 0.5 um", "pMOS L 0.5-8 um"]
    for arm, short, pre, col in (("A", "7p", "s11-iv7", "green"), ("B", "45p", "s12-iv45", "blue")):
        n = "7" if arm == "A" else "45"
        for k in range(3):
            sid = f"{pre}-{k + 1}"
            S[sid] = sec(sid, [
                f'<h2 style="font-size:48px;font-weight:600;line-height:1.1;color:{INK}">I-V curves, {n} parameters: '
                f'devices {6 * k + 1}-{6 * k + 6}</h2>',
                img(f"iv_{short}_{k + 1}", f"Id-Vd and Id-Vg, {grp[k]}: measured, paper card (NGSpice), our "
                                           f"{n}-parameter card (NGSpice)", 1760, 836),
                f'<p style="font-size:26px;line-height:1.3">{grp[k]}. Left of each pair: Id-Vd; right: Id-Vg, log '
                f'scale. Black: measured; grey dashed: paper card; {col}: our card.</p>'],
                f"These are devices {6 * k + 1} to {6 * k + 6} with our {n}-parameter card. Black points are "
                f"measured; the grey dashed line is the paper card and the {col} line our card, both simulated in "
                "NGSpice. The panel title gives each device's RRMS for both cards.",
                style=f"background:{PAPER};color:{INK};{F};padding:40px 64px 32px;display:flex;"
                      "flex-direction:column;gap:12px")
    return S


def render_html(sid: str, html: str) -> str:
    """Standalone page for a headless-browser layout check (local image files instead of /_blob/ ids)."""
    for key, (bid, path) in BLOB.items():
        html = html.replace(f"/_blob/{bid}", path.as_uri())
    html = re.sub(r"<aside>.*?</aside>", "", html, flags=re.S)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap' "
            "rel='stylesheet'><style>*{margin:0;box-sizing:border-box} body{width:1920px;height:1080px;"
            "overflow:hidden} section{width:1920px;height:1080px;position:relative;overflow:hidden} "
            "ul{padding-left:1.2em} td,th{padding:6px 12px;text-align:left;border:1px solid #c9ced3}"
            "</style></head><body>" + html + "</body></html>")


def main() -> int:
    S = build()
    assert list(S) == ORDER or set(S) == set(ORDER), set(S) ^ set(ORDER)
    proj = OUT_ROOT / "project"
    sd = proj / "slides"
    sd.mkdir(parents=True, exist_ok=True)
    for k, v in S.items():
        (sd / f"{k}.html").write_text(v)
    for k in OLD:
        p = sd / f"{k}.html"
        if p.exists():
            p.unlink()
    deck = json.loads((REPO_PROJECT / "deck.json").read_text())
    deck["title"] = "Fitting BSIM4 parameters to SKY130 transistors at 77 K"
    deck["order"] = ORDER
    deck["sections"] = SECTIONS
    deck["cover"] = "title"
    (proj / "deck.json").write_text(json.dumps(deck, indent=1))
    if RENDER:
        rd = Path(RENDER)
        rd.mkdir(parents=True, exist_ok=True)
        for i, k in enumerate(ORDER):
            (rd / f"{i + 1:02d}_{k}.html").write_text(render_html(k, S[k]))
    print(f"{len(ORDER)} slides written to {sd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
