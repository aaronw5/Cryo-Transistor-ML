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
    "goal": ("4fd811034f85d747ce71c54af5238f2f", FIG / "i25_goal_stock.png"),
    "measured": ("1844405a05e791cedc2e08e8e320a132", FIG / "s01_measured.png"),
    "eq_rrms": ("42728e3500a3277bc0e28dbb5d75edc7", FIG / "eq_rrms.png"),
    "start": ("523c4132bdc0c8d791e52e95ff4458ef", FIG / "i25_start_card.png"),
    "trace": ("0d8e318e0c507dc6cdfd762b27d42f60", FIG / "i25_trace.png"),
    "rounds": ("d5ebd30a930e6dc23707e73fa4597873", FIG / "i25_rounds_starts.png"),
    "perdev": ("cc5157dee20ebc5a72e5119064b55397", FIG / "i25_per_device_card.png"),
    "iv_7p_1": ("7e82729a0da6a0cc451b90109b917312", FIG / "i25_iv_card_7p_1.png"),
    "iv_7p_2": ("674df2715072be671d840a25d40799c0", FIG / "i25_iv_card_7p_2.png"),
    "iv_7p_3": ("ab51bb2e3b74a7155fa8b7b0e823bc44", FIG / "i25_iv_card_7p_3.png"),
    "iv_45p_1": ("b74c6312641b315427349ba3818bfece", FIG / "i25_iv_card_45p_1.png"),
    "iv_45p_2": ("ec4ebbda9a6d6e78909bc18e09a25f9f", FIG / "i25_iv_card_45p_2.png"),
    "iv_45p_3": ("52914a90fb7c2bf54b9af407dbebb897", FIG / "i25_iv_card_45p_3.png"),
}
ORDER = ["title", "w01-goal", "w02-data", "w03-model", "w04-paper", "w05-workflow", "w06-step1", "w07-step2",
         "w08-step3", "w09-step4", "w10-step5", "w11-params45", "w12-results", "w13-perdevice",
         "s11-iv7-1", "s11-iv7-2", "s11-iv7-3", "s12-iv45-1", "s12-iv45-2", "s12-iv45-3"]
OLD = ["b1-transistor", "b2-model", "b3-fit", "s01-measured", "s02-stock", "s03-paper", "s04-setup", "s04-stepA",
       "s04b-params", "s04c-why", "s04d-45vs43", "s09-results", "s09b-gmr", "s09c-stages", "s10-perdevice",
       "s10b-nmos", "s10c-pmos", "s13-remaining", "s13b-lowvds", "s15-cost", "s16-sources", "s17-limits"]
SECTIONS = {
    "s1": {"description": "Title", "start": "title"},
    "s2": {"description": "Goal, measured data, model and error, the paper's card", "start": "w01-goal"},
    "s3": {"description": "Our workflow, step by step", "start": "w05-workflow"},
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
        r = {"cma": [], "ls": []}
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
    assert (min(nfit.values()), max(nfit.values())) == (39, 42)
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
             ("VSAT", "saturation velocity"), ("DELTA", "smoothing of the linear-to-saturation transition"),
             ("RDSW", "source/drain series resistance"), ("ETA0", "threshold shift with drain voltage (DIBL)")]
    S["w04-paper"] = sec("w04-paper", [
        H2.format("The paper's card: 7 tuned parameters"),
        '<div style="display:flex;flex-direction:row;gap:48px;align-items:center">'
        + table(["Parameter", "Meaning"], [30, 70], [[f"<b>{p}</b>", m] for p, m in seven], 28, width=1040)
        + f'<div style="display:flex;flex-direction:column;gap:12px;background:{CARD};border:1px solid {LINE};'
          f'border-radius:8px;padding:36px;width:560px"><p style="font-size:28px;color:{MUTED}">All-device RRMS, '
          f'paper card</p><p style="font-size:80px;font-weight:600;line-height:1.1">{f4(pap["all_device"])}</p>'
          f'<p style="font-size:26px;color:{MUTED}">NGSpice, 18 devices</p></div></div>',
        ul(["The paper changed these 7 parameters per device; all others unchanged",
            "Our first run fits the same 7 parameters"])],
        "The paper's card changes seven BSIM4 parameters for each device and leaves the rest of the card file "
        "unchanged. Simulated in NGSpice on the 18 devices, it gives an all-device RRMS of 0.2751. Our first run "
        "fits the same seven parameters, so the two cards can be compared directly.", gap=36)

    wf_svg = svg_boxes(
        [(0, 40, 240, 150, [("Start card", 28, True), ("tuned values", 24, False), ("removed", 24, False)], INK, CARD),
         (290, 40, 240, 150, [("3 random", 28, True), ("start points", 28, True)], BLUE, CARD),
         (580, 40, 260, 150, [("Global search", 28, True), ("CMA-ES", 26, False)], BLUE, CARD),
         (890, 40, 260, 150, [("Local refinement", 28, True), ("least squares", 26, False)], BLUE, CARD),
         (1200, 40, 240, 150, [("Keep best", 28, True), ("start", 28, True)], BLUE, CARD),
         (1490, 40, 238, 150, [("Final card", 28, True)], "#c8384e", "#fbe8de")],
        [(240, 115, 286, 115), (530, 115, 576, 115), (840, 115, 886, 115), (1150, 115, 1196, 115),
         (1440, 115, 1486, 115),
         (1020, 190, 1020, 250, "", 0, 0, True, False), (710, 250, 1020, 250, "", 0, 0, True, False),
         (710, 250, 710, 194, "rounds 2 and 3: narrower range", 865, 290, True)],
        1728, 310, "Workflow: start card, 3 random starts, global search, local refinement, repeated twice with a "
                   "narrower range, keep the best start, final card")
    S["w05-workflow"] = sec("w05-workflow", [
        H2.format("Our workflow"),
        wf_svg,
        ul(["Done separately for each of the 18 devices",
            "Run 1: the paper's 7 parameters; run 2: 45 parameters",
            "Every error value comes from an NGSpice simulation",
            "NGSpice run: one simulation of all measured curves of one device"])],
        "This is the whole procedure; the next slides go through it step by step. Each device is fitted on its own. "
        "We run it twice, once with the paper's seven parameters and once with 45 parameters. Every error value in "
        "the search comes from an NGSpice simulation.", gap=44)

    S["w06-step1"] = sec("w06-step1", [
        H2.format("Step 1: start card"),
        img("start", "pMOS L 0.35 um x W 5 um: measured points and the start card simulated in NGSpice", 1640, 636),
        ul(["Start card: paper card file with the tuned values removed (all multipliers = 1)",
            "Keeps the file's 77 K settings; contains no fitted value from the paper",
            f"All-device RRMS of the start card: {m1['all_device']:.1f}"])],
        "We start from the paper's card file with every tuned value set back to the base value, that is, all "
        "multipliers equal to 1. The file keeps its 77 K settings but no fitted value from the paper. Its "
        f"all-device RRMS is {m1['all_device']:.1f}.", gap=24)

    S["w07-step2"] = sec("w07-step2", [
        H2.format("Step 2: three random start points"),
        table(["Parameters", "Allowed range"], [50, 50], [
            ["VTH0 (threshold voltage)", "1/3 to 3 times the start-card value"],
            ["the other 6 of the paper's 7", "1/100 to 100 times the start-card value"],
            ["the other 38 of the 45", "1/3 to 3 times the start-card value"]], 30),
        ul(["Each parameter drawn at random in its range, uniform on a log scale",
            "Reason: a search from one point can stop in a local minimum",
            "If NGSpice cannot simulate a start (mobility below 0), a new one is drawn",
            f"New draws needed: {rnd['A']['redraw_pairs']} of 54 starts (7 parameters), "
            f"{rnd['B']['redraw_pairs']} of 54 (45)"])],
        "Each device gets three random start points inside the allowed range. Several starts are used because a "
        "search from a single point can end in a local minimum. Some random values give a card that NGSpice cannot "
        "simulate, because the effective mobility for that transistor size becomes negative; such a start is "
        "replaced by a new random draw.", gap=36)

    S["w08-step3"] = sec("w08-step3", [
        H2.format("Step 3: global search (CMA-ES)"),
        img("trace", "One device, 45 parameters, 3 starts: device RRMS against NGSpice runs; circles after global "
                     "search, squares after least squares", 1640, 656),
        ul(["CMA-ES: samples many parameter sets, moves the sampling toward lower error",
            f"Per start and round: about {rnd['A']['cma_mean'] / 1000:.0f},000 NGSpice runs (7 parameters), "
            f"{rnd['B']['cma_mean'] / 1000:.0f},000 (45)",
            "Figure: one pMOS, 45 parameters; circles = after global search"], 28)],
        "The global search uses CMA-ES, an evolution strategy. It samples many parameter sets inside the range, "
        "and moves its sampling distribution toward the sets with the lowest error. It needs no slopes. In this "
        "pMOS example the first global search lowers the RRMS from about 1 to about 0.12 to 0.17.", gap=22)

    S["w09-step4"] = sec("w09-step4", [
        H2.format("Step 4: local refinement (least squares)"),
        img("trace", "Same device: squares mark the result after each least-squares refinement", 1640, 656),
        ul(["Change each parameter slightly, run NGSpice, get the slope of the error",
            "Step to parameter values with lower squared current error; repeat",
            f"Per start and round: about {rnd['A']['ls_mean']:.0f} NGSpice runs (7 parameters), "
            f"{rnd['B']['ls_mean']:.0f} (45)"], 28)],
        "After the global search, a least-squares method refines the best card. It changes each parameter by a "
        "small amount, runs NGSpice, and uses the change in error as a slope to pick the next step. This needs far "
        "fewer runs than the global search. The squares in the figure show the result after each refinement.",
        gap=22)

    S["w10-step5"] = sec("w10-step5", [
        H2.format("Step 5: repeat twice, keep the best start"),
        img("rounds", "Left: all-device RRMS after each round, 7 and 45 parameters. Right: RRMS of the 3 starts per "
                      "device, 45 parameters", 1660, 640),
        ul(["Rounds 2 and 3: range centred on the previous card, half as wide (log scale)",
            "Per device: keep the start with the lowest RRMS, refine it once more",
            f"45 parameters: the 3 starts differ by more than 10 % on {wf['spread_45p_gt10pct']} of 18 devices"],
           28)],
        "Steps 3 and 4 run three times. In rounds 2 and 3 the range is centred on the previous card and narrowed. "
        "After round 3, each device keeps the start with the lowest error, and that card is refined once more by "
        f"least squares. With 45 parameters the all-device RRMS goes from {f4(st['round 1']['B_all'])} after round 1 "
        f"to {f4(cB_['all_device'])} for the final card; with 7 parameters it stops changing after round 2.", gap=22)

    groups = [("Threshold voltage, short / narrow channel", "VTH0, VFB, K1, K2, K3, K3B, W0, LPE0, LPEB, DVT0, DVT1, "
                                                            "DVT0W, DVT1W"),
              ("Oxide, doping, geometry", "TOXE, TOXM, XJ, NDEP, NSD, DWG"),
              ("Mobility", "U0, UA, UB"),
              ("Velocity saturation, bulk charge", "VSAT, A0, A2, AGS, B0, B1, DELTA"),
              ("Series resistance", "RDSW, PRWG, WR, RSH"),
              ("Subthreshold", "VOFF, NFACTOR, CIT, CDSC"),
              ("Drain-voltage effects, output resistance", "ETA0, DSUB, PCLM, PDIBLC1, PDIBLC2, DROUT"),
              ("Substrate current", "ALPHA0, BETA0")]
    allp = sorted(p.strip().lower() for _, g in groups for p in g.split(","))
    assert allp == sorted(spec["selected"]) and len(allp) == 45, allp
    S["w11-params45"] = sec("w11-params45", [
        H2.format("Second run: 45 parameters from a screen"),
        table(["Role", "Parameters (45)"], [38, 62], [[r, p] for r, p in groups], 24),
        ul([f"Reason: with the paper's 7 parameters the all-device RRMS stays at {f4(cA_['all_device'])}",
            "Screen: change each of 115 parameters by 10 %; keep if the current changes by more than 0.01 %",
            f"A parameter that is 0 in a device's card stays 0: {min(nfit.values())}-{max(nfit.values())} fitted "
            "per device",
            "Earlier 43-parameter screen: here CDSC, DVT0W, PDIBLC1 added, CDSCD removed"], 26)],
        "With the paper's seven parameters the error stops falling at 0.1546. For the second run we choose more "
        "parameters with a screen: each of 115 BSIM4 parameters is changed by ten percent, and it is kept if the "
        "simulated current changes by more than one part in ten thousand of the measured current. 45 pass. Compared "
        "with an earlier screen that gave 43, three parameters were added and one removed.", gap=24)

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
