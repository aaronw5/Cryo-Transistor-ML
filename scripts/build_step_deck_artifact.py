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
    "paper_ex": ("2ced443a287cf34fd2e0ab9b6d66295f", FIG / "i25_ex_paper.png"),
    "starts": ("48e34cebc391f0d4cd32a444ed3d7a23", FIG / "i25_ex_starts.png"),
    "cma": ("90ef5423cc737d41a66b67e21865d9ae", FIG / "i25_ex_cma.png"),
    "ls": ("6e6e5eb5d482d2834fa36b6fdeb92486", FIG / "i25_ex_ls.png"),
    "rounds_ex": ("5f34ff7d93a4e4ad83061b9f566625bc", FIG / "i25_ex_rounds.png"),
    "optimizers": ("7fdaa39dfd82d86b26154e1544d829c7", FIG / "i25_ex_optimizers.png"),
    "screen": ("0b3894a877f07e1a4d766d99e3553279", FIG / "i25_screen.png"),
    "perdev": ("d31f1cdb1f5cfd9b708df2e510854fd5", FIG / "i25_per_device_card.png"),
    "iv_7p_1": ("0b524c16213b5bebea45f8b9cae84b4e", FIG / "i25_iv_card_7p_1.png"),
    "iv_7p_2": ("2c223869301929f15353e49de144870f", FIG / "i25_iv_card_7p_2.png"),
    "iv_7p_3": ("b8b8fd124e13e49b24162295158acac9", FIG / "i25_iv_card_7p_3.png"),
    "iv_45p_1": ("3bef28518e314338e7efd15169f2f06d", FIG / "i25_iv_card_45p_1.png"),
    "iv_45p_2": ("080f622d1773c1cd5d130c9f273b7037", FIG / "i25_iv_card_45p_2.png"),
    "iv_45p_3": ("7097cb56c4cb709b0a57a9f4c3f01202", FIG / "i25_iv_card_45p_3.png"),
}
ORDER = ["title", "w01-goal", "w02-data", "w03-model", "w04-paper", "w05c-eval", "w05d-optimizers", "w05-workflow",
         "w05b-pseudo", "w06-step1", "w07-step2", "w08-step3", "w09-step4", "w10-step5", "w11-step6", "w11b-step7",
         "w12-params45", "w13-results", "w14-perdevice",
         "s11-iv7-1", "s11-iv7-2", "s11-iv7-3", "s12-iv45-1", "s12-iv45-2", "s12-iv45-3"]
OLD = ["w11-params45", "w12-results", "w13-perdevice", "b1-transistor", "b2-model", "b3-fit", "s01-measured", "s02-stock", "s03-paper", "s04-setup", "s04-stepA",
       "s04b-params", "s04c-why", "s04d-45vs43", "s09-results", "s09b-gmr", "s09c-stages", "s10-perdevice",
       "s10b-nmos", "s10c-pmos", "s13-remaining", "s13b-lowvds", "s15-cost", "s16-sources", "s17-limits"]
SECTIONS = {
    "s1": {"description": "Title", "start": "title"},
    "s2": {"description": "Stock card, measured data, card and RRMS, Beall et al. card", "start": "w01-goal"},
    "s3": {"description": "NGSpice run, optimizers CMA-ES and TRF, fitting procedure, steps 1-7", "start": "w05c-eval"},
    "s4": {"description": "Run 2 screen, results, per-device RRMS, I-V curves", "start": "w12-params45"},
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

    # ---------------------------------------------------------------- svg helpers
    def T(x, y, text, size=26, bold=False, anchor="start", color=INK):
        fw = ' font-weight="600"' if bold else ""
        return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}"{fw} '
                f'font-family="IBM Plex Sans, Arial, sans-serif" fill="{color}">{text}</text>')

    def R(x, y, w, h, stroke, fill=CARD, dash=False, rx=10):
        d_ = ' stroke-dasharray="12 8"' if dash else ""
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="3"{d_}/>')

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

    def r10(v):
        return f"{int(round(v / 10.0) * 10):,}"
    cma = (rnd["A"]["cma_mean"], rnd["B"]["cma_mean"])
    lsr = (rnd["A"]["ls_mean"], rnd["B"]["ls_mean"])
    fdr = (rnd["A"]["fd_mean"], rnd["B"]["fd_mean"])
    ref_ = (cA["C0_refit"] - cA["starting_card"], cB["C0_refit"] - cB["starting_card"])
    runs = {"cma": f"{r10(cma[0])} / {r10(cma[1])}", "ls": f"{r10(lsr[0])} / {r10(lsr[1])}",
            "fd": f"{r10(fdr[0])} / {r10(fdr[1])}", "ref": f"{r10(ref_[0])} / {r10(ref_[1])}"}
    assert runs == {"cma": "2,990 / 8,010", "ls": "160 / 480", "fd": "150 / 330", "ref": "120 / 300"}, runs
    ex = wf["example"]
    sr = ex["start_rrms"]
    rr = ex["rounds"]
    fd = rr["fd"]
    n_ = ex["n_active"]
    op = wf["optimizers"]
    assert ex["popsize"] == 15 and n_ == 40

    # ---------------------------------------------------------------- title, background
    S["title"] = sec("title", [
        '<h1 style="font-size:80px;font-weight:600;line-height:1.12;color:#f1f0eb">BSIM4 parameter fit, SKY130, '
        '77 K</h1>',
        '<div style="display:flex;flex-direction:column;gap:16px">' + "".join(
            f'<p style="font-size:34px;line-height:1.3">{t}</p>' for t in [
                "18 measured transistors (8 nMOS, 10 pMOS), simulated in NGSpice",
                "Reference: Beall et al. card (published 77 K card)",
                '<span style="color:#b8bec4">2026-10-28</span>']) + "</div>"],
        "BSIM4 parameters are fitted to 77 K measurements of 18 SkyWater 130 nm transistors. Model and simulator are "
        "unchanged; only parameter values change.",
        style=f"background:{INK};color:#f1f0eb;{F};padding:128px 160px;display:flex;flex-direction:column;"
              "justify-content:center;gap:56px")

    S["w01-goal"] = sec("w01-goal", [
        H2.format("Stock card at 77 K"),
        img("goal", "pMOS L 0.35 um x W 5 um at 77 K: measured points and the stock SkyWater card simulated in "
                    "NGSpice", 1640, 636),
        ul(["A card holds the BSIM4 parameter values of one transistor size.",
            "The stock SkyWater card was fitted at room temperature.",
            "The fit changes parameter values to reproduce the 77 K curves."])],
        "Black points are the measured pMOS, L 0.35 um, W 5 um, at 77 K. Orange lines are the stock card simulated "
        "in NGSpice at 77 K.", gap=24)

    S["w02-data"] = sec("w02-data", [
        H2.format("Measured data, 18 transistors"),
        img("measured", "Measured Id-Vd and Id-Vg of pMOS L 0.35 um x W 5 um at 77 K", 1640, 560),
        ul(["8 nMOS, 10 pMOS; L 0.15-100 um, W 0.42-100 um.",
            "Each device has 5 Id-Vd and 6 Id-Vg curves.",
            "Faded points lie below the current floor and are not scored."])],
        "The example device is a pMOS with L 0.35 um and W 5 um. pMOS voltages and currents are negative; the plots "
        "show absolute values.", gap=22)

    model_svg = svg_boxes(
        [(0, 20, 330, 130, [("Card", 30, True), ("BSIM4 parameter values", 26, False)], BLUE, CARD),
         (430, 20, 330, 130, [("NGSpice", 30, True), ("circuit simulator", 26, False)], BLUE, CARD),
         (860, 20, 360, 130, [("Simulated curves", 30, True), ("Id at measured voltages", 26, False)], BLUE, CARD),
         (860, 200, 360, 110, [("Measured curves", 30, True), ("11 per device", 26, False)], INK, CARD),
         (1330, 105, 398, 130, [("RRMS", 30, True), ("simulated vs measured", 26, False)], "#c8384e",
          "#fbe8de")],
        [(330, 85, 426, 85), (760, 85, 856, 85), (1220, 85, 1326, 150), (1220, 255, 1326, 195)],
        1728, 320, "Card to NGSpice to simulated curves; simulated and measured curves give the RRMS")
    S["w03-model"] = sec("w03-model", [
        H2.format("Card, NGSpice, RRMS"),
        model_svg,
        f'<div style="display:flex;flex-direction:row;gap:40px;align-items:center">'
        f'<div style="background:{CARD};border:1px solid {LINE};padding:12px"><img src="/_blob/{BLOB["eq_rrms"][0]}" '
        f'alt="RRMS of a curve and of a device" style="width:720px;height:156px;object-fit:contain"></div>'
        + ul(["Curve RRMS = RMS current error / mean |I_meas|.",
              "Device RRMS = mean of its curve RRMS values.",
              "All-device RRMS = mean over the 18 devices."], 28, extra=";width:960px") + "</div>"],
        "Definition and scored curves are the same as used for the Beall et al. card.", gap=36)

    seven = [("VTH0", "threshold voltage"), ("U0", "low-field mobility"), ("NFACTOR", "subthreshold slope factor"),
             ("VSAT", "saturation velocity"), ("DELTA", "linear-saturation smoothing"),
             ("RDSW", "source/drain resistance"), ("ETA0", "threshold shift with Vds")]
    S["w04-paper"] = sec("w04-paper", [
        H2.format("Beall et al. card: 7 multipliers"),
        '<div style="display:flex;flex-direction:row;gap:32px;align-items:center">'
        + table(["Parameter", "Meaning"], [36, 64], [[f"<b>{p}</b>", m] for p, m in seven], 24, width=600)
        + img("paper_ex", "pMOS L 0.35 um x W 5 um: measured points and the Beall et al. card simulated in NGSpice",
              1060, 410) + "</div>",
        ul(["One fitted multiplier per parameter and device; other values unchanged.",
            f"Device RRMS, example pMOS: {ex['paper_rrms']:.3f}. All-device RRMS: {f4(pap['all_device'])}.",
            "Run 1 fits the same 7 parameters."], 28)],
        "The figure shows the Beall et al. card on the example pMOS, simulated in NGSpice.", gap=30)

    # ---------------------------------------------------------------- evaluation, optimizers, procedure
    ev = svg(1728, 150, "One NGSpice run: parameter values, card file, NGSpice, 11 simulated curves, comparison with "
                        "measured curves, error", [
        B(0, 20, 250, 110, [("Parameter", 26, True), ("values", 26, True)]),
        A_(250, 75, 282, 75),
        B(286, 20, 230, 110, [("Card", 26, True), ("file", 26, True)]),
        A_(516, 75, 548, 75),
        B(552, 20, 250, 110, [("NGSpice", 26, True), ("1 run", 24, False)]),
        A_(802, 75, 834, 75),
        B(838, 20, 270, 110, [("11 simulated", 26, True), ("curves", 26, True)]),
        A_(1108, 75, 1140, 75),
        B(1144, 20, 280, 110, [("Minus measured", 26, True), ("curves", 26, True)], INK),
        A_(1424, 75, 1456, 75),
        B(1460, 20, 268, 110, [("Error", 28, True)], "#c8384e", "#fbe8de"),
    ])
    S["w05c-eval"] = sec("w05c-eval", [
        H2.format("One NGSpice run"),
        ev,
        ul(["One run simulates all 11 curves of one device for one card.",
            "CMA-ES uses one number per run: the device RRMS.",
            "TRF uses one residual per measured point."], 30)],
        "Point residual = (I_sim - I_meas) / mean |I_meas| of its curve, divided by the square root of (points in "
        "the curve x curves of the device). The sum of squared residuals then equals the mean squared curve RRMS. "
        "A curve NGSpice cannot simulate counts as RRMS 10.", gap=36)

    S["w05d-optimizers"] = sec("w05d-optimizers", [
        H2.format("Two optimizers: CMA-ES and TRF"),
        table(["", "CMA-ES (Python cma)", "TRF (SciPy least_squares)"], [22, 37, 41], [
            ["Loss", "device RRMS", "sum of squared point residuals"],
            ["Derivatives", "none", "finite differences (FD): 1 run per parameter"],
            ["Iteration", "sample 9 (7 p.) or 14-15 (45 p.) cards, simulate, move mean and covariance toward "
                          "lowest loss",
             "FD Jacobian, step inside a trust region, keep if loss drops"],
            ["Runs per call, 7 / 45 p.", runs["cma"], runs["ls"]],
            ["Starts from", "random card or previous card", "CMA-ES result"]], 23),
        img("optimizers", "pMOS L 0.35 um x W 5 um, 45 parameters: lowest device RRMS against NGSpice runs for "
                          "CMA-ES and FD least squares from the same 3 random start cards", 1080, 428),
        ul(["FD least squares = TRF with FD Jacobian, 2 passes.",
            "Pass 2 weights each curve by 1/√(curve RRMS).",
            f"From the random starts: TRF {op['0']['ls_end']:.2f}, {op['1']['ls_end']:.2f}, "
            f"{op['2']['ls_end']:.2f}; CMA-ES {op['0']['cma_end']:.3f}, {op['1']['cma_end']:.3f}, "
            f"{op['2']['cma_end']:.3f}."], 24)],
        "CMA-ES is the covariance matrix adaptation evolution strategy. TRF is the trust-region reflective "
        "least-squares method; FD shifts each parameter by 0.01 in scaled units, 1/200 of its log range. The figure "
        "is a separate run of TRF alone from the 3 random start cards of the example pMOS.", gap=14)

    nest = svg(1728, 560, "Fitting procedure for one device: 3 random start cards; rounds 1-3 of CMA-ES, FD least "
                          "squares and range update; FD polish; keep the best start; final FD refit; final card", [
        R(2, 2, 1724, 556, INK),
        T(24, 40, "One device; repeated for 18 devices", 28, True),
        R(22, 62, 1120, 480, BLUE, "#f3f7fd", dash=True),
        T(42, 98, "Each of 3 starts", 26, True, color=BLUE),
        B(42, 150, 250, 170, [("Random start", 26, True), ("card in the", 24, False), ("round-1 range", 24, False)]),
        A_(292, 235, 334, 235),
        R(338, 116, 784, 250, "#4a5259", "#ffffff"),
        T(358, 150, "Rounds 1, 2, 3", 26, True),
        B(358, 170, 230, 170, [("CMA-ES", 26, True), (runs["cma"], 24, False)]),
        A_(588, 255, 620, 255),
        B(624, 170, 230, 170, [("FD least", 26, True), ("squares", 26, True), (runs["ls"], 24, False)]),
        A_(854, 255, 886, 255),
        B(890, 170, 216, 170, [("Range:", 26, True), ("centred on", 24, False), ("result, half", 24, False),
                               ("width", 24, False)]),
        A_(730, 366, 730, 402),
        B(560, 406, 340, 120, [("FD polish", 26, True), (runs["fd"], 24, False)]),
        A_(1142, 300, 1186, 300),
        B(1190, 120, 516, 110, [("Keep the start with", 26, True), ("lowest device RRMS", 26, True)], INK),
        A_(1448, 230, 1448, 266),
        B(1190, 270, 516, 110, [("Final FD refit", 26, True), (runs["ref"], 24, False)], INK),
        A_(1448, 380, 1448, 416),
        B(1190, 420, 516, 100, [("Final card", 28, True)], "#c8384e", "#fbe8de"),
    ])
    S["w05-workflow"] = sec("w05-workflow", [
        H2.format("Fitting procedure, one device"),
        nest,
        ul(["Mean NGSpice runs per call, 7 / 45 parameters.",
            f"Total per device, 3 starts: {kint(cA['C0_refit'])} / {kint(cB['C0_refit'])}."], 28)],
        "Run 1 fits the 7 Beall et al. parameters; run 2 fits 45. FD least squares runs inside each round, as the "
        "FD polish after round 3, and as the final FD refit.", gap=22)

    pc = [("for device in 18 devices:", ""),
          ("&nbsp;&nbsp;for start in 1, 2, 3:", ""),
          ("&nbsp;&nbsp;&nbsp;&nbsp;range = round-1 range", "step 1"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;x = random card in range", "step 2"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;for round in 1, 2, 3:", "round"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = cmaes(loss=device_rrms, x, range)", f"step 3; {runs['cma']}"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = trf(loss=sum_sq_residuals, jac=fd, x, range)",
           f"step 4; {runs['ls']}"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if round &lt; 3: range = centred on x, half width", "step 5"),
          ("&nbsp;&nbsp;&nbsp;&nbsp;x = trf(loss=sum_sq_residuals, jac=fd, x, range)",
           f"step 6, FD polish; {runs['fd']}"),
          ("&nbsp;&nbsp;best = start with lowest device_rrms", "step 7"),
          ("&nbsp;&nbsp;card = trf(loss=sum_sq_residuals, jac=fd, best, range)",
           f"step 7, final FD refit; {runs['ref']}")]
    mono = "font-family:'Courier New', Courier, monospace"
    code = (f'<div style="display:flex;flex-direction:column;gap:6px;background:{CARD};border:1px solid {LINE};'
            f'padding:24px 28px">' + "".join(
                f'<div style="display:flex;flex-direction:row"><p style="font-size:26px;line-height:1.3;{mono};'
                f'width:1080px">{a}</p><p style="font-size:24px;line-height:1.3;color:{MUTED}">'
                f'{"# " + b if b else ""}</p></div>' for a, b in pc) + "</div>")
    S["w05b-pseudo"] = sec("w05b-pseudo", [
        H2.format("Procedure as pseudocode"),
        code,
        ul(["Run counts: mean NGSpice runs per call, 7 / 45 parameters.",
            "range: allowed interval per parameter, log scale."], 26)],
        "fd: Jacobian by finite differences in NGSpice. trf: SciPy least_squares, method trf, 2 passes.", gap=24)

    # ---------------------------------------------------------------- steps
    steps = ["1 Start card", "2 Random starts", "3 CMA-ES", "4 FD least sq.", "5 Rounds 2, 3", "6 FD polish",
             "7 Keep, refit"]

    def strip(cur):
        parts, x, w = [], 0, 222
        for i, lab in enumerate(steps, 1):
            on = i == cur
            parts.append(R(x, 4, w, 60, "#c8384e" if on else LINE, "#fbe8de" if on else CARD, rx=8))
            parts.append(T(x + w / 2, 43, lab, 23, on, "middle", INK if on else MUTED))
            if i < 7:
                parts.append(A_(x + w + 2, 34, x + w + 22, 34))
            x += w + 29
        return svg(1728, 68, f"Steps 1 to 7, step {cur} highlighted", parts)

    def io(inp, outp):
        return (f'<table style="width:1728px;font-size:25px;color:{INK};background:{CARD};border:1px solid {LINE}">'
                f'<tr><th style="width:12%">Input</th><td>{inp}</td></tr>'
                f'<tr><th style="width:12%">Output</th><td>{outp}</td></tr></table>')

    STEP = SEC.format(g=14).replace("padding:64px 96px 56px", "padding:48px 96px 40px")

    S["w06-step1"] = sec("w06-step1", [
        H2.format("Step 1: start card"), strip(1),
        io("Beall et al. card file", "start card: same file, all multipliers = 1"),
        img("start", "pMOS L 0.35 um x W 5 um: measured points and the start card simulated in NGSpice", 1400, 560),
        ul(["The Beall et al. card file stores each fitted value as base value x multiplier.",
            "All multipliers are set to 1; the 77 K settings of the file stay.",
            "The start-card values are the centres of the round-1 ranges in step 2.",
            f"All-device RRMS of the start card: {m1['all_device']:.1f}."], 25)],
        "The figure shows the start card and the measurement for the example pMOS.", style=STEP)

    S["w07-step2"] = sec("w07-step2", [
        H2.format("Step 2: random start cards"), strip(2),
        io("start card; range VTH0 and 38 added parameters 1/3-3x, other 6 of 7 1/100-100x",
           "3 random start cards per device"),
        img("starts", "pMOS L 0.35 um x W 5 um: measured Id-Vd and Id-Vg at 1.85 V and the 3 random start cards "
                      "simulated in NGSpice", 1400, 540),
        ul(["Each free parameter gets a random value between its limits, uniform on a log scale.",
            "NGSpice simulates the card; if a scored curve fails, a new card is drawn (mean 1.6 / 2.2 runs per "
            "start).",
            f"Example: starts 1 and 2 give currents below 1e-11 A (device RRMS {sr[0]:.2f}); start 3 has device RRMS "
            f"{sr[2]:.1f}."], 25)],
        "Starts 1 and 2 give currents below 1e-11 A on both curves shown.", style=STEP)

    S["w08-step3"] = sec("w08-step3", [
        H2.format("Step 3: CMA-ES, loss device RRMS"), strip(3),
        io("start card or previous card; range", "card with the lowest device RRMS"),
        img("cma", "Round 1 CMA-ES on pMOS L 0.35 um x W 5 um: device RRMS of every candidate against NGSpice runs, "
                   "lowest so far per start; sampled VTH0 and U0 by generation", 1400, 540),
        ul(["CMA-ES keeps a normal distribution (mean, covariance) over the scaled parameters.",
            f"One generation: sample {ex['popsize']} cards, run NGSpice for each, compute each device RRMS.",
            "The mean moves toward the lowest-RRMS cards; the covariance is re-estimated from them.",
            f"Example, round 1, {ex['cmaes_evals'][0]:,} runs per start: device RRMS {sr[0]:.2f}, {sr[1]:.2f}, "
            f"{sr[2]:.1f} to {ex['cmaes_best'][0]:.3f}, {ex['cmaes_best'][1]:.3f}, {ex['cmaes_best'][2]:.3f}."],
           24)],
        "The left plot shows the device RRMS of every sampled card and the lowest value so far. The right plot shows "
        "the sampled VTH0 and U0 of start 2, coloured by generation.", style=STEP)

    jac = svg(1728, 96, "One TRF iteration: card, n plus 1 NGSpice runs, Jacobian, trial step", [
        B(0, 8, 300, 80, [("Card x", 26, True), (f"{n_} free parameters", 24, False)]),
        A_(300, 48, 336, 48),
        B(340, 8, 470, 80, [(f"{n_} + 1 NGSpice runs", 26, True), ("x, and x with one parameter shifted", 22, False)]),
        A_(810, 48, 846, 48),
        B(850, 8, 430, 80, [("Jacobian", 26, True), ("change of each residual", 22, False)]),
        A_(1280, 48, 1316, 48),
        B(1320, 8, 408, 80, [("Trial step, 1 run", 26, True), ("kept if the loss drops", 22, False)]),
    ])
    S["w09-step4"] = sec("w09-step4", [
        H2.format("Step 4: FD least squares (TRF)"), strip(4),
        io("card after CMA-ES; range", "card with the lowest device RRMS seen"),
        jac,
        img("ls", "Round 1 FD least squares on pMOS L 0.35 um x W 5 um: device RRMS per iteration for 3 starts; "
                  "RRMS of each curve of start 2 before and after", 1180, 390),
        ul(["Residual of a point: (simulated - measured current) / mean measured current of its curve.",
            f"Loss: sum of squared residuals. Jacobian: NGSpice reruns with each of the {n_} parameters shifted by "
            "1/200 of its log range.",
            "TRF solves the linearized problem inside a trust region and tries the step; kept if the loss drops.",
            "Two passes; pass 2 weights each curve by 1/√(curve RRMS). Example, start 2, round 1: "
            f"{ex['cmaes_best'][1]:.3f} to {ex['round1_rrms'][1]:.3f} in {ex['ls_runs'][1]} runs."], 24)],
        f"One iteration costs {n_} + 2 NGSpice runs: 1 at x, {n_} shifted, 1 trial step. The right plot shows each "
        "curve RRMS of start 2 after CMA-ES and after FD least squares.", style=STEP)

    S["w10-step5"] = sec("w10-step5", [
        H2.format("Step 5: rounds 2 and 3"), strip(5),
        io("card after the previous round", "card after round 3"),
        img("rounds_ex", "pMOS L 0.35 um x W 5 um: allowed range and value found for VTH0, U0, RDSW in rounds 1-3; "
                         "device RRMS after each round for the 3 starts", 1400, 556),
        ul(["Rounds 2 and 3 start CMA-ES from the card of the previous round.",
            "Each range is re-centred on the previous card and halved in width on a log scale.",
            "A parameter in the outer 5 % of its range keeps its width, so its range extends past the old limit.",
            f"Example, start 2: device RRMS {rr['rrms']['1'][0]:.3f}, {rr['rrms']['1'][1]:.3f}, "
            f"{rr['rrms']['1'][2]:.3f} after rounds 1, 2, 3."], 24)],
        "Each round repeats steps 3 and 4 in the new range. The left plot shows the range and the value found for "
        "VTH0, U0 and RDSW of start 2.", style=STEP)

    S["w11-step6"] = sec("w11-step6", [
        H2.format("Step 6: FD polish"), strip(6),
        io("card after round 3, each start", "polished card, each start"),
        table(["Device RRMS", "Example start 1", "start 2", "start 3", "All devices, 45 p.", "All devices, 7 p."],
              [24, 16, 12, 12, 18, 18], [
            ["After round 3", f"{rr['rrms']['0'][2]:.4f}", f"{rr['rrms']['1'][2]:.4f}", f"{rr['rrms']['2'][2]:.4f}",
             f4(st["round 3"]["B_all"]), f"{st['round 3']['A_all']:.6f}"],
            ["After FD polish", f"{fd['0']:.4f}", f"{fd['1']:.4f}", f"{fd['2']:.4f}",
             f4(st["round 3 + FD"]["B_all"]), f"{st['round 3 + FD']['A_all']:.6f}"]], 28),
        ul(["After round 3, each start gets one more FD least-squares call (TRF, 2 passes) in its round-3 range.",
            f"Up to 400 NGSpice runs; mean {runs['fd']} (7 / 45 parameters)."], 26)],
        "The FD polish is the same TRF routine as step 4, run once on the card from round 3.", style=STEP)

    S["w11b-step7"] = sec("w11b-step7", [
        H2.format("Step 7: best start, final FD refit"), strip(7),
        io("3 polished cards", "final card"),
        table(["Device RRMS", "Example start 1", "start 2", "start 3", "All devices, 45 p.", "All devices, 7 p."],
              [24, 16, 12, 12, 18, 18], [
            ["After FD polish", f"{fd['0']:.4f}", f"{fd['1']:.4f}", f"{fd['2']:.4f}",
             f4(st["round 3 + FD"]["B_all"]), f"{st['round 3 + FD']['A_all']:.6f}"],
            ["Final FD refit", "-", f"{rr['final']:.4f} (kept)", "-", f4(cB_["all_device"]),
             f"{cA_['all_device']:.6f}"]], 28, hl=(1,)),
        ul(["The start with the lowest device RRMS after the FD polish is kept.",
            f"The kept card gets one more FD least-squares call (up to 400 runs, mean {runs['ref']}).",
            f"Example: start 2 kept ({fd['1']:.4f}); final card {rr['final']:.4f}."], 26)],
        "The final FD refit is the same TRF routine, run once on the kept card in its round-3 range.", style=STEP)

    sc = wf["screen"]
    act = [v["n"] for v in wf["active_45p"].values()]
    assert (min(act), max(act)) == (37, 41)
    S["w12-params45"] = sec("w12-params45", [
        H2.format("Run 2: 45 screened parameters"),
        img("screen", "Screen of 115 BSIM4 parameters: current change for a 10 % parameter change, threshold, kept "
                      "parameters, the 7 Beall et al. parameters", 1660, 640),
        ul(["Each of 115 parameters changed by ±10 %, one at a time.",
            "Kept if current change exceeds 1e-4 x mean measured current.",
            f"{min(act)}-{max(act)} of 45 fitted per device; zero or tiny values fixed."], 26)],
        f"With 7 parameters the all-device RRMS is {f4(cA_['all_device'])}. The current change is the RMS over each "
        "curve divided by the curve's mean measured current, maximum over all curves and 18 devices. UB (values "
        "near 1e-18) stays fixed in the fit.", gap=22)

    # ---------------------------------------------------------------- results
    S["w13-results"] = sec("w13-results", [
        H2.format("Results, 18 devices"),
        table(["Card (NGSpice)", "All-device RRMS", "Combined RRMS", "NGSpice runs per device"],
              [40, 20, 20, 20], [
            ["Beall et al. card", f4(pap["all_device"]), f4(pap["combined"]), "-"],
            ["Fitted card, 7 parameters", f4(cA_["all_device"]), f4(cA_["combined"]), kint(cA["C0_refit"])],
            ["Fitted card, 45 parameters", f4(cB_["all_device"]), f4(cB_["combined"]), kint(cB["C0_refit"])]],
              32, hl=(1, 2)),
        ul([f"Fitted / Beall et al. RRMS per device: {rA['ratio']:.2f} (7 p.), {rB['ratio']:.2f} (45 p.).",
            f"Lower than the Beall et al. card on {rA['wins']} of 18 devices, both runs.",
            "Combined RRMS = mean of nMOS mean and pMOS mean."]),
        small("0.54, 0.36: geometric mean over 18 devices. Runs include all 3 starts.")],
        "All three cards are BSIM4 in NGSpice with different parameter values.", gap=30)

    big = "pmos_L0p35_W0p55"
    S["w14-perdevice"] = sec("w14-perdevice", [
        H2.format("Device RRMS, three cards"),
        '<div style="display:flex;flex-direction:row;gap:36px;align-items:center">'
        + img("perdev", "Per-device RRMS, log scale: Beall et al. card, fitted 7-parameter card, fitted 45-parameter "
                        "card", 1290, 800)
        + ul(["Both fitted cards below the Beall et al. card on 18 of 18.",
              "Largest RRMS: pMOS L 0.35 um x W 0.55 um."], 30, extra=";width:390px") + "</div>"],
        "Bars are device RRMS on a log scale, three cards per device.", gap=24)

    grp = ["nMOS L 0.15-8 um", "nMOS L 20, 100 um; pMOS L 0.35, 0.5 um", "pMOS L 0.5-8 um"]
    for arm, short, pre, col in (("A", "7p", "s11-iv7", "green"), ("B", "45p", "s12-iv45", "blue")):
        n = "7" if arm == "A" else "45"
        for k in range(3):
            sid = f"{pre}-{k + 1}"
            S[sid] = sec(sid, [
                f'<h2 style="font-size:48px;font-weight:600;line-height:1.1;color:{INK}">I-V, {n} parameters, '
                f'devices {6 * k + 1}-{6 * k + 6}</h2>',
                img(f"iv_{short}_{k + 1}", f"Id-Vd and Id-Vg, {grp[k]}: measured, Beall et al. card (NGSpice), fitted "
                                           f"{n}-parameter card (NGSpice)", 1760, 836),
                f'<p style="font-size:26px;line-height:1.3">{grp[k]}. Each pair: Id-Vd left, Id-Vg right (log). '
                f'Black measured, grey dashed Beall et al. card, {col} fitted card.</p>'],
                "Panel titles give the device RRMS of both cards. Both cards are simulated in NGSpice.",
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
    deck["title"] = "BSIM4 parameter fit, SKY130, 77 K"
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
