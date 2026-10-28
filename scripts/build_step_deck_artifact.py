#!/usr/bin/env python3
"""Build the step-by-step Slides artifact (https://claude.ai/artifact/VXzexFpptoGiwaSzpVF57r), card-only revision.

The deck covers parameter extraction for the unchanged model: the standard BSIM4 card in NGSpice 41, only parameter
values change.  No correction g appears on any slide.  The final card of each arm is the Step A card (round 3 + FD)
refit by TRF without any correction, stored as C0_refit (I25_1/arm{A,B}/c0; NGSpice re-runs in final/).

Writes project/deck.json and project/slides/<id>.html under OUT_ROOT (default: slides/step_deck_artifact).
Every result number is read from out/research/explore/I25_1/decision.json (plus the per-round call counts in
I25_1/stepA/*.json, the screen summary I25_1/screen_mult1/summary.json, and the figure statistics in
out/research/explore/step_deck/numbers_i25_card.json written by scripts/make_step_deck_i25.py).  Figures are
uploaded to the artifact's asset store first; their /_blob/ ids are listed in BLOB below.  Slides kept unchanged
from the previous revision (b1-transistor, b2-model, s01-measured, b3-fit) are copied from the existing project
folder.

Publish (Artifact tool): url = the artifact, root = OUT_ROOT, file_path = OUT_ROOT/project/deck.json,
files = every project/slides/<id>.html, plus `null` for removed slide files.

  python3 scripts/build_step_deck_artifact.py [OUT_ROOT]
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
I25 = ROOT / "out/research/explore/I25_1"
REPO_PROJECT = ROOT / "slides/step_deck_artifact/project"
OUT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "slides/step_deck_artifact"

BLOB = {  # uploaded 2026-10-28 (figs/step_deck/i25_*card*.png, i25_err_split.png, i25_lowvds_card.png)
    "stock": "bb63a57cb18977aa487e2bb59fd97c6d",
    "stages": "0dd55ea5ba963f99447704742891ce59", "perdev": "cc5157dee20ebc5a72e5119064b55397",
    "split": "b61989ebd1d1d5bf9731aae13d9fb34b", "lowvds": "eaaf8fcced6894a919f026c5dc866d47",
    "iv_7p_1": "7e82729a0da6a0cc451b90109b917312", "iv_7p_2": "674df2715072be671d840a25d40799c0",
    "iv_7p_3": "ab51bb2e3b74a7155fa8b7b0e823bc44", "iv_45p_1": "b74c6312641b315427349ba3818bfece",
    "iv_45p_2": "ec4ebbda9a6d6e78909bc18e09a25f9f", "iv_45p_3": "52914a90fb7c2bf54b9af407dbebb897",
}
KEEP = ["b1-transistor", "b2-model", "s01-measured", "b3-fit"]
ORDER = ["title", "b1-transistor", "b2-model", "s01-measured", "s02-stock", "s03-paper", "b3-fit",
         "s04-setup", "s04c-why", "s04b-params", "s04d-45vs43", "s04-stepA",
         "s09-results", "s09b-gmr", "s09c-stages", "s10-perdevice", "s10b-nmos", "s10c-pmos",
         "s11-iv7-1", "s11-iv7-2", "s11-iv7-3", "s12-iv45-1", "s12-iv45-2", "s12-iv45-3",
         "s13-remaining", "s13b-lowvds", "s15-cost", "s16-sources", "s17-limits"]
REMOVED = ["s05-stepB", "s05c-whyg", "s06-stepC", "s07-stepD", "s14-element", "s14b-checks"]
SECTIONS = {
    "s1": {"description": "Title", "start": "title"},
    "s2": {"description": "Background: I-V curves, card, NGSpice, RRMS; measured data", "start": "b1-transistor"},
    "s3": {"description": "Stock card and multiplier-1 start card at 77 K, paper card", "start": "s02-stock"},
    "s4": {"description": "Card fit: setup, design choices, parameter sets, RRMS by round", "start": "b3-fit"},
    "s5": {"description": "Results, card only: tables, GMR, per device, I-V curves for 7 and 45 parameters",
           "start": "s09-results"},
    "s6": {"description": "Where the card-only fit still differs from the measurement", "start": "s13-remaining"},
    "s7": {"description": "Cost, paper inputs, limits", "start": "s15-cost"},
}
CARD_ONLY = "Card only: the standard BSIM4 model in NGSpice 41, only the parameter values changed"

F = "font-family:'IBM Plex Sans', Arial, sans-serif"
SEC = f"background:#f6f5f1;color:#1f252b;{F};padding:56px 96px 48px;display:flex;flex-direction:column;gap:{{g}}px"
H2 = '<h2 style="font-size:48px;font-weight:600;line-height:1.1;color:#1f252b">{}</h2>'


def ul(items, size=28, lh=1.35, extra=""):
    return (f'<ul style="font-size:{size}px;line-height:{lh};color:#1f252b{extra}">'
            + "".join(f"<li>{i}</li>" for i in items) + "</ul>")


def img(blob, alt, w, h, pad=0):
    p = f";padding:{pad}px" if pad else ""
    return (f'<div style="display:flex;flex-direction:row;justify-content:center;background:#ffffff;'
            f'border:1px solid #c9ced3{p}"><img src="/_blob/{BLOB[blob]}" alt="{alt}" '
            f'style="width:{w}px;height:{h}px;object-fit:contain"></div>')


def table(head, widths, rows, size=28, hl=(), width=1728):
    s = f'<table style="width:{width}px;font-size:{size}px;color:#1f252b;background:#ffffff;border:1px solid #c9ced3">'
    s += '<tr style="background:#e4e7ea">' + "".join(f'<th style="width:{w}%">{h}</th>' for h, w in zip(head, widths)) + "</tr>"
    for i, r in enumerate(rows):
        bg = ' style="background:#fbe8de"' if i in hl else ""
        s += f"<tr{bg}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return s + "</table>"


def sec(sid, body, notes, gap=20, style=None):
    st = style or SEC.format(g=gap)
    return f'<section id="{sid}" style="{st}">\n' + "\n".join(body) + f"\n<aside>{notes}</aside>\n</section>\n"


def note_line(text):
    return f'<p style="font-size:26px;line-height:1.3;color:#5a6168">{text}</p>'


def f4(x):
    return f"{x:.4f}"


def ci(g):
    return f"{g['ratio']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}]"


def kint(x):
    return f"{x:,.0f}"


def pretty(tag):
    fam, l_, w_ = tag.split("_")
    return f"{'nMOS' if fam == 'nmos' else 'pMOS'} {l_[1:].replace('p', '.')} x {w_[1:].replace('p', '.')}"


def dev_order(tags):
    def key(t):
        fam, l_, w_ = t.split("_")
        return (fam == "pmos", float(l_[1:].replace("p", ".")), float(w_[1:].replace("p", ".")))
    return sorted(tags, key=key)


def load():
    d = json.loads((I25 / "decision.json").read_text())
    assert d["validity"]["valid"], "I25-1 decision.json not valid"
    old = json.loads((I25 / "old/old.json").read_text())
    nums = json.loads((ROOT / "out/research/explore/step_deck/numbers_i25_card.json").read_text())
    screen = json.loads((I25 / "screen_mult1/summary.json").read_text())
    rounds = {}
    for arm, pre in (("A", "seven"), ("B", "full")):
        r = {}
        for st in ("r1", "r2", "r3", "fd"):
            tot = 0
            for k in range(3):
                j = json.loads((I25 / f"stepA/{pre}_s{k}_{st}.json").read_text())
                tot += sum(v.get("n_sim", 0) for v in j["devices"].values())
            r[st] = tot / 18
        log = json.loads((I25 / f"stepA/{pre}_chain_log.json").read_text())
        red = {k: v["draws"] for k, v in log["pairs"].items() if not k.startswith("center") and v.get("draws", 0) > 0}
        r["redraw_pairs"], r["redraw_max"], r["wall_s"] = len(red), max(red.values()), log["wall_s"]
        rounds[arm] = r
    return d, old, nums, screen, rounds


def build() -> dict:
    d, old, nums, screen, rnd = load()
    A, B = d["arms"]["A"], d["arms"]["B"]
    ref = d["reference"]
    stock, m1, pap = ref["stock_77K"]["primary"], ref["mult1_no_fit"]["primary"], ref["paper"]["primary"]
    cA_, cB_ = A["summary"]["C0_refit"]["primary"], B["summary"]["C0_refit"]["primary"]
    cA, cB = A["calls_per_device"], B["calls_per_device"]
    gA, gB = d["gmr"]["A"]["primary"], d["gmr"]["B"]["primary"]
    gx = nums["gmr_extra"]
    st = {r["stage"]: r for r in d["stage_table"]}
    pub = {t: r["rrms"] for t, r in old["arms"]["published"].items()}
    pd = {a: {t: (v["rrms"] if isinstance(v, dict) else v) for t, v in d["arms"][a]["per_device"]["C0_refit"].items()}
          for a in "AB"}
    tags = dev_order(list(pd["A"]))
    spec = d["specs"]
    fixed = spec["fixed_zero_center_by_device"]
    nfit = {t: 45 - len(v) for t, v in fixed.items()}
    ms = {a: d["arms"][a]["multistart"] for a in "AB"}
    split, low = nums["error_split"], nums["lowvds"]["summary"]
    sp_ = {a: {t: v["max"] / v["min"] for t, v in ms[a]["spread"].items() if v["max"] != float("inf")} for a in "AB"}
    S = {}

    # -- checks against the numbers stated in the brief (fail loudly on mismatch)
    brief = {"stock": (3.4176, 3.1077, stock), "mult1": (12.1010, 10.9593, m1), "paper": (0.2751, 0.2595, pap),
             "7p": (0.1546, 0.1458, cA_), "45p": (0.1094, 0.1026, cB_)}
    for k, (a, c, row) in brief.items():
        assert round(row["all_device"], 4) == a and round(row["combined"], 4) == c, (k, row)
    assert (round(gA["C0_refit_vs_paper"]["ratio"], 3), round(gB["C0_refit_vs_paper"]["ratio"], 3)) == (0.536, 0.365)
    assert gA["C0_refit_vs_paper"]["wins"] == 18 and gB["C0_refit_vs_paper"]["wins"] == 18
    assert [round(cA["C0_refit"]), round(cB["C0_refit"])] == [28953, 77696]
    sm = screen["parameters"]
    assert screen["candidate_count"] == 115 and screen["selected_count"] == 45
    assert screen["effective_threshold"] == 1e-4 and screen["perturbation_fraction"] == 0.1
    ov = spec["overlap_with_old43"]
    assert ov["n_both"] == 42 and ov["only_old"] == ["cdscd"] and sorted(ov["only_new"]) == ["cdsc", "dvt0w", "pdiblc1"]
    assert not sm["cdscd"]["selected"] and sm["cdscd"]["max_full_grid_response"] < 1e-4

    def agg_row(name, row, calls):
        fm = lambda v: f4(v) if v < 1 else f"{v:.3f}"   # noqa: E731
        return [name, fm(row["all_device"]), fm(row["combined"]), fm(row["nmos"]), fm(row["pmos"]), calls]

    # ------------------------------------------------------------------ title
    S["title"] = sec("title", [
        '<h1 style="font-size:72px;font-weight:600;line-height:1.15;color:#f1f0eb">SKY130, 77 K: BSIM4 card '
        'parameters fitted from multiplier-1 cards, 7 and 45 parameters</h1>',
        '<div style="display:flex;flex-direction:column;gap:14px">' + "".join(
            f'<p style="font-size:32px;line-height:1.3">{t}</p>' for t in [
                "Devices: 18 (8 nMOS, 10 pMOS), measured at 77 K; simulator: NGSpice 41; model: standard BSIM4, "
                "only parameter values change",
                "Start: paper's 77 K card files with every tuned multiplier set to 1; 3 random starts per device",
                f'<span style="color:#f2a68f">All-device / combined RRMS: paper card {f4(pap["all_device"])} / '
                f'{f4(pap["combined"])}; our 7-parameter card {f4(cA_["all_device"])} / {f4(cA_["combined"])}; '
                f'our 45-parameter card {f4(cB_["all_device"])} / {f4(cB_["combined"])}</span>',
                '<span style="color:#b8bec4">2026-10-28 (run I25-1)</span>']) + "</div>"],
        "This deck covers fitting BSIM4 card parameters to measured SkyWater 130 nm I-V data at 77 K. The model is "
        "the standard BSIM4 in NGSpice 41; only parameter values change. The fit starts from the paper's 77 K card "
        "files with every tuned value removed. There are 18 measured devices, 8 nMOS and 10 pMOS. Every number comes "
        "from NGSpice 41 runs scored with the paper's RRMS metric. Two parameter sets are fitted: the paper's 7 "
        "parameters and a 45-parameter set from a sensitivity screen.",
        style=f"background:#1f252b;color:#f1f0eb;{F};padding:128px 160px;display:flex;flex-direction:column;"
              "justify-content:center;gap:48px")

    # ------------------------------------------------------------------ stock / paper
    S["s02-stock"] = sec("s02-stock", [
        H2.format("Stock card and multiplier-1 card at 77 K"),
        img("stock", "pMOS L 0.35 um W 5 um: measured; stock card at 77 K (orange); paper's 77 K card file with "
                     "multipliers 1 (purple)", 1700, 540),
        ul([f"Orange: stock sky130A tt card, fitted at 300 K (tnom 30 C), run at 77 K; 18 devices: RRMS "
            f"{stock['all_device']:.3f} all-device, {stock['combined']:.3f} combined (nMOS / pMOS mean "
            f"{stock['nmos']:.3f} / {stock['pmos']:.3f})",
            "Purple: paper's 77 K card files (18 custom bins, tnom -196.15 C, temperature coefficients 0) with every "
            "tuned multiplier = 1; this is the fit start",
            f"Multiplier-1 card, 18 devices: {m1['all_device']:.3f} all-device, {m1['combined']:.3f} combined "
            f"(nMOS / pMOS mean {m1['nmos']:.3f} / {m1['pmos']:.3f})",
            "Device in figure: pMOS L 0.35 x W 5 um (L, W = channel length, width in um)"], 26)],
        "The stock SkyWater card was fitted at room temperature. Run at 77 K it misses the measured pMOS currents by "
        "large factors; the pMOS mean RRMS is 5.9. The purple curves use the paper's 77 K card files, which set the "
        "nominal temperature to 77 K and the temperature coefficients to zero, with every tuned multiplier set back "
        "to 1. No tuned value from the paper is left in this card. It scores 12.1 all-device, and it is the card the "
        "fit starts from.", gap=18)

    S["s03-paper"] = sec("s03-paper", [
        H2.format("Paper card (reference)"),
        '<div style="display:flex;flex-direction:row;gap:32px">' + "".join(
            f'<div style="flex:{fl};display:flex;flex-direction:column;gap:8px;background:#ffffff;border:1px solid '
            f'#c9ced3;border-radius:8px;padding:32px"><p style="font-size:28px;color:#5a6168">{a}</p>'
            f'<p style="font-size:64px;font-weight:600;line-height:1.1">{b}</p></div>'
            for a, b, fl in (("All-device RRMS", f4(pap["all_device"]), 1), ("Combined RRMS", f4(pap["combined"]), 1),
                             ("nMOS / pMOS mean", f"{f4(pap['nmos'])} / {f4(pap['pmos'])}", 1.6))) + "</div>",
        ul(["Tuned per device: VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0 (7 parameters)",
            "Card form: base value x multiplier, in the paper's 77 K card files",
            "Run here: NGSpice 41, 18 devices, same RRMS and scored curve set as every other row",
            "All-device = mean over 18 devices; combined = (nMOS mean + pMOS mean) / 2"])],
        "The paper published one 77 K card with seven tuned BSIM4 parameters per device, written as multipliers on "
        "base values. Re-run in NGSpice 41 on the same 18 devices it scores 0.2751 all-device and 0.2595 combined. "
        "This card fixes the scored curve set used for every row in this deck. Our fit uses the same card files "
        "with these multipliers set to 1, and fits the same 7 parameters, then a larger set of 45.",
        style=SEC.format(g=48).replace("padding:56px 96px 48px", "padding:72px 96px 72px"))

    # ------------------------------------------------------------------ Step A setup and design choices
    S["s04-setup"] = sec("s04-setup", [
        H2.format("Step A setup: start card, random starts, rounds, final card"),
        table(["Item", "Setting"], [24, 76], [
            ["Start card", "paper's 77 K card files, every tuned multiplier = 1 (RRMS 12.10 all-device)"],
            ["Box (log)", "center = multiplier-1 value; VTH0 x/÷ 3, other 6 of the 7 x/÷ 100, all others x/÷ 3"],
            ["Starts", "3 per device: each fitted parameter drawn log-uniform in its box (seeds 20261027 + 1000 k + i)"],
            ["Round 1", "CMA-ES from the start point (7p: 3,000 evaluations; 45p: 8,000), then TRF"],
            ["Rounds 2, 3", "box re-centered on the previous card, width x 0.5 (edge coordinates keep width); "
                            "CMA-ES + TRF"],
            ["FD polish", "TRF with finite-difference Jacobian, max 400 calls"],
            ["Selection", "per device, the start with the lowest RRMS on its own data after FD"],
            ["Final card", "selected card refit by TRF (max 400 calls); stored as C0_refit; every result below"],
        ], 26, hl=(7,)),
        ul(["CMA-ES = evolutionary search; TRF = trust-region least squares; 1 NGSpice call = 1 run of all measured "
            "biases of 1 device",
            "x/÷ k = from center / k to center x k"], 26)],
        "Step A fits the card in NGSpice. The fit starts from the paper's card files with every tuned multiplier "
        "set to 1, so no tuned value from the paper is used. Each device gets three random start points inside the "
        "box. Each start runs three rounds of CMA-ES followed by TRF; rounds 2 and 3 re-center the box on the "
        "previous card and halve its width. A finite-difference polish follows. Per device, the start with the "
        "lowest RRMS on that device's data is kept, and that card is refit once more by TRF. This final card is the "
        "card in every result table and I-V figure. The next slide gives the reason for each choice and the "
        "evidence from this run.", gap=18)

    sB_ = ms["B"]["per_start"]
    S["s04c-why"] = sec("s04c-why", [
        H2.format("Step A choices: reason and evidence from this run"),
        table(["Choice", "Reason", "Evidence (all-device RRMS, 18 devices)"], [17, 31, 52], [
            ["Start from multiplier-1 card files",
             "fit result should not depend on the paper's tuned values",
             f"vs the earlier run started from the paper card file with tuned values (round 3 + FD, paired GMR): "
             f"45p vs earlier 43p {ci(gx['B_start_vs_old43_fd'])}, lower on {gx['B_start_vs_old43_fd']['wins']}/18; "
             f"7p vs earlier 7p {ci(gx['A_start_vs_old7p'])}, lower on {gx['A_start_vs_old7p']['wins']}/18"],
            ["3 random starts, best per device",
             "one start can stop in a local minimum; 3 results show the start dependence",
             f"45p: best / worst start differ by &gt; 10 % on {sum(v > 1.1 for v in sp_['B'].values())} of 18 devices; "
             f"one start for all devices {sB_[0]['all_device_mean_rrms']:.4f} / {sB_[1]['all_device_mean_rrms']:.4f} / "
             f"{sB_[2]['all_device_mean_rrms']:.4f} vs best per device {f4(st['round 3 + FD']['B_all'])}; "
             f"7p: &gt; 10 % on {sum(v > 1.1 for v in sp_['A'].values())} of {len(sp_['A'])}, box center start "
             f"{f4(ref['A_center_start_chain']['all_device'])} vs {f4(st['round 3 + FD']['A_all'])}"],
            ["Redraw a start NGSpice cannot run",
             "a low U0 draw with the bin's negative LU0 / WU0 / PU0 gives effective u0 &lt;= 0; BSIM4 stops",
             f"redrawn on {rnd['A']['redraw_pairs']} of 54 device-start pairs (7p), {rnd['B']['redraw_pairs']} of 54 "
             f"(45p); at most {rnd['A']['redraw_max']} / {rnd['B']['redraw_max']} redraws"],
            ["Rounds 2, 3: re-centered box, width x 0.5",
             "next search covers the region around the previous card at higher sample density",
             f"45p: {f4(st['round 1']['B_all'])} -&gt; {f4(st['round 2']['B_all'])} -&gt; {f4(st['round 3']['B_all'])}; "
             f"7p: {f4(st['round 1']['A_all'])} -&gt; {f4(st['round 2']['A_all'])} -&gt; {f4(st['round 3']['A_all'])}"],
            ["FD polish",
             "least-squares steps with a finite-difference Jacobian after the derivative-free search",
             f"45p: {f4(st['round 3']['B_all'])} -&gt; {f4(st['round 3 + FD']['B_all'])}; 7p: change "
             f"{abs(st['round 3']['A_all'] - st['round 3 + FD']['A_all']):.0e}; "
             f"{kint(rnd['A']['fd'])} / {kint(rnd['B']['fd'])} calls per device"],
        ], 22)],
        "Each row gives one Step A choice, the reason for it, and what this run shows. Starting from the "
        "multiplier-1 card removes the paper's tuned values from the fit; against the earlier run that started from "
        "the paper's card file with tuned values, the paired ratio is 0.95 for both parameter sets, and the 45-"
        "parameter interval contains 1. Three random starts matter for the 45-parameter set: the best and worst "
        "start differ by more than ten percent on 15 devices, and any single start for all devices is 0.118 to "
        "0.135 against 0.1096 for the best start per device. Some draws give a card NGSpice cannot run and are "
        "redrawn from the same random stream. Re-centering and narrowing the box lowers the 45-parameter RRMS in "
        "rounds 2 and 3; the 7-parameter fit stops changing after round 2. The finite-difference polish changes the "
        "45-parameter RRMS by 0.0006.", gap=18)

    # ------------------------------------------------------------------ parameter sets
    seven = [("VTH0", "threshold voltage of a long channel at Vbs = 0"), ("U0", "low-field mobility"),
             ("NFACTOR", "subthreshold swing factor"), ("VSAT", "carrier saturation velocity"),
             ("DELTA", "smoothing of the linear-to-saturation transition (effective Vds)"),
             ("RDSW", "source/drain series resistance per unit width"),
             ("ETA0", "DIBL coefficient in subthreshold (Vth shift with Vds)")]
    groups = [("Oxide, doping, geometry", "TOXE, TOXM, XJ, NDEP, NSD, DWG"),
              ("Threshold voltage, body effect, short / narrow channel",
               "VTH0, VFB, K1, K2, K3, K3B, W0, LPE0, LPEB, DVT0, DVT1, DVT0W, DVT1W"),
              ("Mobility", "U0, UA, UB"),
              ("Velocity saturation, bulk charge, linear-saturation", "VSAT, A0, A2, AGS, B0, B1, DELTA"),
              ("Series resistance", "RDSW, PRWG, WR, RSH"),
              ("Subthreshold", "VOFF, NFACTOR, CIT, CDSC"),
              ("DIBL, output resistance", "ETA0, DSUB, PCLM, PDIBLC1, PDIBLC2, DROUT"),
              ("Impact ionization (substrate current)", "ALPHA0, BETA0")]
    allp = sorted(p.strip().lower() for _, g in groups for p in g.split(","))
    assert allp == sorted(spec["selected"]) and len(allp) == 45, allp
    nm_fit = sorted({nfit[t] for t in nfit if t.startswith("nmos")})
    pm_fit = sorted({nfit[t] for t in nfit if t.startswith("pmos")})
    S["s04b-params"] = sec("s04b-params", [
        H2.format("Parameter sets: the paper's 7 and the screened 45"),
        '<div style="display:flex;flex-direction:row;gap:28px">'
        + table(["7 (paper)", "BSIM4 meaning"], [26, 74], [[f"<b>{p}</b>", m] for p, m in seven], 22, width=720)
        + table(["45 (screen), by role", "Parameters"], [44, 56], [[r, p] for r, p in groups], 22, width=980)
        + "</div>",
        ul([f"Fitted per device: a parameter whose value is 0 in the device's bin stays 0 (log box): nMOS "
            f"{nm_fit[-1]} of 45 (B0, B1, DVT0W, PRWG, VFB at 0; nMOS 0.15 x 1.6 also A0: {nm_fit[0]}); pMOS "
            f"{pm_fit[-1]} (LPE0, LPEB, PDIBLC1 at 0) or {pm_fit[0]} (also B0: pMOS 2 x 5, 4 x 7, 8 x 1.6, 8 x 5)",
            "The 7 are inside the 45; all 7 fitted on all 18 devices"], 24)],
        "The left table gives the seven parameters the paper tuned, with their BSIM4 meaning. The right table groups "
        "the 45 parameters of the screened set by the part of the model they act on. "
        "Meanings of the other 38: TOXE electrical oxide thickness; TOXM oxide thickness at which parameters were "
        "extracted; XJ source/drain junction depth; NDEP channel doping at the depletion edge; NSD source/drain doping; "
        "DWG gate-bias dependence of effective width; VFB flat-band voltage; K1, K2 first- and second-order body-effect "
        "coefficients; K3 narrow-width coefficient; K3B body dependence of K3; W0 narrow-width parameter; LPE0, LPEB "
        "lateral non-uniform doping and its body dependence; DVT0, DVT1 short-channel coefficients of Vth; DVT0W, DVT1W "
        "narrow-width coefficients of Vth at small L; UA, UB first- and second-order mobility degradation by the "
        "vertical field; A0 channel-length dependence of the bulk charge effect; A2 second non-saturation factor; AGS "
        "gate-bias coefficient of the bulk charge factor; B0, B1 width coefficient and width offset of the bulk charge "
        "effect; PRWG gate-bias dependence of the series resistance; WR width offset for the series resistance; RSH "
        "source/drain sheet resistance; VOFF subthreshold offset voltage; CIT interface trap capacitance; CDSC "
        "source/drain-to-channel coupling capacitance; DSUB DIBL exponent in subthreshold; PCLM channel length "
        "modulation; PDIBLC1, PDIBLC2 first and second DIBL corrections of the output resistance; DROUT length "
        "dependence of that correction; ALPHA0, BETA0 impact ionization current parameters. The box is in log space, "
        "so a parameter with value zero in a device's bin is not fitted on that device: 39 to 42 of the 45 are fitted "
        "per device.", gap=16)

    resp = {p: sm[p]["max_full_grid_response"] for p in ("cdsc", "dvt0w", "pdiblc1", "cdscd")}
    act = {p: sm[p]["active_devices"] for p in ("cdsc", "dvt0w", "pdiblc1")}
    assert act["cdsc"] == ["pmos_L0p35_W5"] and act["dvt0w"] == ["pmos_L0p35_W0p55"] and len(act["pdiblc1"]) == 5
    assert all("dvt0w" in fixed[t] for t in fixed if t.startswith("nmos"))
    assert all("pdiblc1" in fixed[t] for t in fixed if t.startswith("pmos"))
    assert not any("cdsc" in v for v in fixed.values())
    S["s04d-45vs43"] = sec("s04d-45vs43", [
        H2.format("45-parameter set: what changed from the earlier 43"),
        ul([f"Screen: each of {screen['candidate_count']} BSIM4 DC candidates changed by +/-10 % one at a time; response "
            f"= RMS current change / mean |I_meas| per curve; kept if &gt; {screen['effective_threshold']:g} on any "
            f"curve of any device; the 7 always kept: {screen['selected_count']} of {screen['candidate_count']} kept",
            "Earlier 43: same rule around the paper card (tuned values); now: around the multiplier-1 card",
            f"Shared: {ov['n_both']}; added 3, dropped 1; net +2"], 26),
        table(["Parameter", "Change", "BSIM4 meaning", "Max screen response", "Fitted on"], [13, 9, 42, 18, 18], [
            ["CDSC", "added", "coupling capacitance between source/drain and channel; enters the subthreshold swing "
                              "factor", f"{resp['cdsc']:.1e} (1 device: pMOS 0.35 x 5)", "all 18 devices"],
            ["DVT0W", "added", "first coefficient of the narrow-width effect on Vth at small channel length",
             f"{resp['dvt0w']:.1e} (1 device: pMOS 0.35 x 0.55)", "10 pMOS (0 in nMOS bins)"],
            ["PDIBLC1", "added", "first coefficient of the DIBL correction to the output resistance",
             f"{resp['pdiblc1']:.1f} (5 pMOS, L 0.35-0.5 um)", "8 nMOS (0 in pMOS bins)"],
            ["CDSCD", "dropped", "drain-bias dependence of CDSC",
             f"{resp['cdscd']:.1e} (below {screen['effective_threshold']:g})", "not fitted"]], 24)],
        "The 45-parameter set comes from the same sensitivity screen that gave the earlier 43, re-run around the "
        "multiplier-1 card instead of the paper card. 42 parameters are in both sets. Three are added: CDSC and DVT0W "
        "pass the threshold on one device each, just above the limit, and PDIBLC1 passes on the five short pMOS "
        "devices. CDSCD falls below the threshold and is dropped. Because the fit box is logarithmic, a parameter "
        "whose value is zero in a bin stays at zero: PDIBLC1 is zero in every pMOS bin, so it is fitted only on the "
        "nMOS devices, although the screen found it on pMOS; DVT0W is zero in every nMOS bin and is fitted only on "
        "pMOS.", gap=20)

    S["s04-stepA"] = sec("s04-stepA", [
        H2.format("Step A: all-device RRMS after each round, card only"),
        img("stages", "All-device RRMS per round for our 7-parameter and 45-parameter cards, then the final card "
                      "refit; paper card as a dotted line; log scale", 1500, 580),
        ul([f"Best start per device, all-device / combined: 7p round 1 {f4(st['round 1']['A_all'])} / "
            f"{f4(st['round 1']['A_comb'])}, final card {f4(cA_['all_device'])} / {f4(cA_['combined'])}; 45p round 1 "
            f"{f4(st['round 1']['B_all'])} / {f4(st['round 1']['B_comb'])}, final card {f4(cB_['all_device'])} / "
            f"{f4(cB_['combined'])}",
            "Final card = selected Step A card refit by TRF (C0_refit); every curve is an NGSpice run"], 26)],
        "This plot shows the all-device RRMS of the card after each stage. With the paper's seven parameters, the "
        "RRMS goes from 0.1706 after round 1 to 0.1546 after round 2 and does not change further. With 45 parameters "
        "it goes from 0.1354 to 0.1096 after round 3 and the finite-difference polish, and to 0.1094 after the final "
        "refit. The paper card is at 0.2751.", gap=18)

    # ------------------------------------------------------------------ results, card only
    S["s09-results"] = sec("s09-results", [
        H2.format("Results, 18 devices, card only"),
        note_line(CARD_ONLY),
        table(["Card (NGSpice)", "All-device", "Combined", "nMOS mean", "pMOS mean", "NGSpice calls / device"],
              [34, 12, 12, 12, 12, 18], [
            agg_row("Stock card, 77 K", stock, "0"),
            agg_row("Paper's 77 K card, multipliers = 1", m1, "0"),
            agg_row("Paper card", pap, "-"),
            agg_row("Our 7-parameter card", cA_, kint(cA["C0_refit"])),
            agg_row("Our 45-parameter card", cB_, kint(cB["C0_refit"]))], 28, hl=(3, 4)),
        ul(["Our cards: final card of each set (Step A over 3 starts + final TRF refit); same 7 parameters as the "
            "paper, or the 45-parameter set",
            "All-device = mean of 18 devices; combined = (nMOS mean + pMOS mean) / 2; combined is compared only "
            "with combined",
            "Calls: NGSpice runs per device, summed over the 3 starts, incl. the final refit"], 26)],
        "This table lists every card on the same 18 devices and the same scored curve set. All rows are the standard "
        "BSIM4 model in NGSpice; only the parameter values differ. Starting from the multiplier-1 card at 12.1, the "
        "fit of the paper's seven parameters reaches 0.1546 all-device and 0.1458 combined, against 0.2751 and "
        "0.2595 for the paper card. The 45-parameter fit reaches 0.1094 and 0.1026. The last column is the number of "
        "NGSpice runs per device.", gap=22)

    grA, grB, gab = gA["C0_refit_vs_paper"], gB["C0_refit_vs_paper"], gx["A_card_vs_B_card"]
    S["s09b-gmr"] = sec("s09b-gmr", [
        H2.format("Paired device ratios (GMR), card only"),
        note_line(CARD_ONLY),
        table(["Comparison", "GMR [95% CI]", "devices with first lower"], [54, 28, 18], [
            ["Our 7-parameter card vs paper card (same 7 parameters)", ci(grA), f"{grA['wins']}/18"],
            ["Our 45-parameter card vs paper card", ci(grB), f"{grB['wins']}/18"],
            ["Our 7-parameter card vs our 45-parameter card", ci(gab), f"{gab['wins']}/18"]], 28, hl=(0,)),
        ul(["GMR(A, B) = geometric mean over the 18 devices of RRMS_A / RRMS_B; &lt; 1: A lower",
            "95% CI = device bootstrap, 10,000 resamples",
            "Rows 1-2 from decision.json; row 3 computed with the same bootstrap code (i3_2_bias.paired_ratio)"], 26)],
        "The geometric mean ratio compares two cards device by device. Our seven-parameter card, which changes the "
        "same seven parameters as the paper, has 0.54 times the paper card's RRMS and is lower on all 18 devices. The "
        "45-parameter card has 0.36 times the paper card's RRMS, lower on all 18. The seven-parameter card is 1.47 "
        "times the 45-parameter card and lower on none of the 18 devices.", gap=24)

    order = ["round 1", "round 2", "round 3", "round 3 + FD", "C0_refit"]
    names = ["Round 1", "Round 2", "Round 3", "Round 3 + FD", "Final card (TRF refit)"]
    calls = {"round 1": (rnd["A"]["r1"], rnd["B"]["r1"]), "round 2": (rnd["A"]["r2"], rnd["B"]["r2"]),
             "round 3": (rnd["A"]["r3"], rnd["B"]["r3"]), "round 3 + FD": (rnd["A"]["fd"], rnd["B"]["fd"]),
             "C0_refit": (cA["C0_refit"] - cA["starting_card"], cB["C0_refit"] - cB["starting_card"])}
    S["s09c-stages"] = sec("s09c-stages", [
        H2.format("RRMS by stage, 7 and 45 parameters, card only"),
        table(["Stage", "7p all-device", "7p combined", "45p all-device", "45p combined", "calls 7p / 45p"],
              [24, 14, 14, 14, 14, 20],
              [[n, f4(st[s]["A_all"]), f4(st[s]["A_comb"]), f4(st[s]["B_all"]), f4(st[s]["B_comb"]),
                f"{kint(calls[s][0])} / {kint(calls[s][1])}"] for n, s in zip(names, order)], 28, hl=(4,)),
        ul(["Rounds: best start per device; calls per device summed over the 3 starts",
            f"7p reference chain from the box center (no random start): {f4(ref['A_center_start_chain']['all_device'])}"
            f" / {f4(ref['A_center_start_chain']['combined'])}"], 26)],
        "This table lists all-device and combined RRMS after every stage for both parameter sets, with the NGSpice "
        "calls each stage costs per device. The seven-parameter fit stops improving after round 2. The 45-parameter "
        "fit improves in every round. The final refit changes little because it starts from the selected Step A card "
        "with the same objective.", gap=24)

    big = "pmos_L0p35_W0p55"
    S["s10-perdevice"] = sec("s10-perdevice", [
        H2.format("Per-device RRMS, card only"),
        '<div style="display:flex;flex-direction:row;gap:32px;align-items:center">'
        + img("perdev", "Per-device RRMS bars, log scale: paper card, our 7-parameter card, our 45-parameter card",
              1300, 820)
        + ul([f"Our 7p card below the paper card on {grA['wins']} of 18 devices; 45p on {grB['wins']} of 18",
              f"45p below 7p on {18 - gab['wins']} of 18",
              f"Largest: pMOS 0.35 x 0.55 um: paper {f4(pub[big])}, 7p {f4(pd['A'][big])}, 45p {f4(pd['B'][big])}",
              "Tables: next two slides"], 28, extra=";width:390px") + "</div>"],
        "Each device has three bars on a log scale: the paper card, our seven-parameter card, and our 45-parameter "
        "card, all NGSpice runs. Both of our cards are below the paper card on every device, and the 45-parameter "
        "card is below the seven-parameter card on every device. The largest remaining error is the narrow pMOS "
        "L 0.35 x W 0.55 um.", gap=20)

    def dev_rows(fam):
        rows = [[pretty(t), f4(pub[t]), f4(pd["A"][t]), f4(pd["B"][t])] for t in tags if t.startswith(fam)]
        rows.append([f"{'nMOS' if fam == 'nmos' else 'pMOS'} mean", f4(pap[fam]), f4(cA_[fam]), f4(cB_[fam])])
        return rows
    head = ["Device, L x W (um)", "Paper card", "Our 7-parameter card", "Our 45-parameter card"]
    nm = [pd[a][t] for a in "AB" for t in tags if t.startswith("nmos")]
    S["s10b-nmos"] = sec("s10b-nmos", [
        H2.format("Per-device RRMS, nMOS, card only"),
        table(head, [34, 22, 22, 22], dev_rows("nmos"), 28, hl=(8,)),
        ul(["All values: NGSpice runs of the stored cards, paper RRMS on the frozen curve set"], 26)],
        f"These are the eight nMOS devices. The paper card is between {min(pub[t] for t in tags if t.startswith('nmos')):.2f} "
        f"and {max(pub[t] for t in tags if t.startswith('nmos')):.2f}. Our cards are between {min(nm):.3f} and "
        f"{max(nm):.3f}. The nMOS mean is the first half of the combined value.", gap=24)
    S["s10c-pmos"] = sec("s10c-pmos", [
        H2.format("Per-device RRMS, pMOS, card only"),
        table(head, [34, 22, 22, 22], dev_rows("pmos"), 26, hl=(10,))],
        "These are the ten pMOS devices. pMOS errors are larger than nMOS for every card. The narrow pMOS L 0.35 x W "
        "0.55 um has the largest error in every column.", gap=24)

    grp = ["nMOS L 0.15-8 um", "nMOS L 20, 100 um; pMOS L 0.35, 0.5 x 0.42 um", "pMOS L 0.5 x 0.64 - 8 um"]
    for arm, short, pre, col in (("A", "7p", "s11-iv7", "green"), ("B", "45p", "s12-iv45", "blue")):
        n = "7" if arm == "A" else "45"
        for k in range(3):
            sid = f"{pre}-{k + 1}"
            S[sid] = sec(sid, [
                f'<h2 style="font-size:44px;font-weight:600;line-height:1.1;color:#1f252b">I-V, our {n}-parameter '
                f'card vs paper card ({k + 1}/3)</h2>',
                img(f"iv_{short}_{k + 1}", f"Id-Vd and Id-Vg, {grp[k]}: measured, paper card (NGSpice), our "
                                           f"{n}-parameter card (NGSpice)", 1724, 820),
                f'<p style="font-size:28px;line-height:1.3">{grp[k]}. Left: Id-Vd; right: Id-Vg (log). Black: '
                f'measured; grey dashed: paper card (NGSpice); {col}: our {n}-parameter card (NGSpice). Panel '
                f'title: device RRMS</p>'],
                f"These panels show six devices with our {n}-parameter card. Black points are measured; faded points "
                f"are below the current floor or on curves the paper card does not score. The grey dashed line is "
                f"the paper card and the {col} line our {n}-parameter card; both are NGSpice runs of the standard "
                f"BSIM4 model with the respective parameter values. The panel title gives each device's RRMS for both "
                f"cards.",
                style=f"background:#f6f5f1;color:#1f252b;{F};padding:40px 64px 36px;display:flex;"
                      "flex-direction:column;gap:14px")

    # ------------------------------------------------------------------ remaining error
    sa, sb = split["A"], split["B"]
    pl = "pmos|idvg_low"
    S["s13-remaining"] = sec("s13-remaining", [
        H2.format("Where the card-only fit still differs from the measurement"),
        img("split", "Left: pMOS L 0.35 x W 5 um Id-Vg at |Vds| 0.01 V, measured, paper card, our 7p and 45p "
                     "cards. Right: contribution of each device type and curve class to the all-device RRMS", 1700, 676),
        ul([f"Contribution of curve c of device d = RRMS_c / C_d / 18 (C_d = scored curves of d); the rows add up to "
            f"the all-device RRMS",
            f"pMOS Id-Vg at |Vds| 0.01 V (10 curves): {sa['contribution'][pl]:.4f} of {f4(sa['all_device'])} "
            f"({sa['share'][pl]:.0%}, 7p), {sb['contribution'][pl]:.4f} of {f4(sb['all_device'])} "
            f"({sb['share'][pl]:.0%}, 45p); mean curve RRMS {sa['mean_curve_rrms'][pl]:.2f} / "
            f"{sb['mean_curve_rrms'][pl]:.2f}"], 24)],
        "The right panel splits the all-device RRMS of each final card into device type and curve class. Because "
        "device RRMS is the mean of its curves and the all-device value is the mean of devices, the rows add up "
        "exactly. One curve class carries the largest part for both parameter sets: the pMOS Id-Vg at a drain "
        "voltage of 0.01 V, with 44 percent of the seven-parameter RRMS and 38 percent of the 45-parameter RRMS. "
        "Its mean curve RRMS is 1.18 and 0.73. The left panel shows this curve on one pMOS: both fitted cards give "
        "more current than measured at high gate voltage.", gap=16)

    lA, lB, lM = low["A"], low["B"], low["meas"]
    S["s13b-lowvds"] = sec("s13b-lowvds", [
        H2.format("pMOS at low |Vds|: fitted cards vs measured"),
        img("lowvds", "Left: ln(I_card / I_meas) on the Id-Vg at |Vds| 0.01 V, |Vgs| 1.81-1.85 V, all 18 devices, "
                      "7p and 45p. Right: ln(I_meas / I_card) along the pMOS Id-Vd at |Vgs| 1.85 V", 1500, 596),
        ul([f"|Vds| 0.01 V, |Vgs| 1.81-1.85 V: ln(I_card / I_meas) pMOS +{lA['pmos_min']:.2f} to +{lA['pmos_max']:.2f} "
            f"(7p), +{lB['pmos_min']:.2f} to +{lB['pmos_max']:.2f} (45p), card above measured on {lA['pmos_pos']} of 10 "
            f"(7p) and {lB['pmos_pos']} of 10 (45p); nMOS {lA['nmos_min']:+.2f} to {lA['nmos_max']:+.2f} / {lB['nmos_min']:+.2f} to "
            f"{lB['nmos_max']:+.2f}",
            f"Id-Vd at |Vgs| 1.85 V, |Vds| = 0.2 VDD: card below measured on {lA['onset_pos']['0.2']} of "
            f"10 pMOS (7p) and {lB['onset_pos']['0.2']} of 10 (45p); conductance ratio G(0.15 VDD) / G(0.03 VDD), G = |Id| / "
            f"|Vds|: measured {lM['g_ratio_min']:.2f}-{lM['g_ratio_max']:.2f} (&gt; 1 on {lM['g_ratio_gt1']} of 10), "
            f"7p card {lA['g_ratio_min']:.2f}-{lA['g_ratio_max']:.2f}, 45p card {lB['g_ratio_min']:.2f}-"
            f"{lB['g_ratio_max']:.2f} (&gt; 1 on {lA['g_ratio_gt1']} and {lB['g_ratio_gt1']} of 10)",
            "Same bias in two measured sweeps (|Vgs| 1.85 V, |Vds| 0.01 V): Id-Vd / Id-Vg current pMOS 0.28-1.13 "
            "(&lt; 1 on 9 of 10), nMOS 1.15-1.31; one card value cannot match both sweeps there"], 23)],
        "On all ten pMOS devices both fitted cards give more current than measured in the Id-Vg sweep at a drain "
        "voltage of 0.01 V; on nMOS the difference is small and of both signs. Along the pMOS Id-Vd curve at the "
        "highest gate voltage, the cards are below the measurement at 0.1 to 0.3 of the supply voltage. Together "
        "this means the measured pMOS channel conductance rises with drain voltage at small drain voltage, while "
        "the conductance of both fitted cards falls: with the fitted 7 or 45 parameters the card does not produce "
        "this shape. Going from 7 to 45 parameters lowers the excess at 0.01 V but does not remove it. Part of the "
        "difference is in the data: at the same bias point the measured Id-Vd and Id-Vg sweeps of the pMOS differ "
        "by up to a factor 3.6, so no single card value matches both.", gap=14)

    # ------------------------------------------------------------------ cost, sources, limits
    S["s15-cost"] = sec("s15-cost", [
        H2.format("Cost in NGSpice calls, card only"),
        table(["Stage, per device", "7 parameters", "45 parameters"], [50, 25, 25], [
            ["Step A start draws (redraw rule)",
             kint(cA["starting_card"] - sum(rnd["A"][k] for k in ("r1", "r2", "r3", "fd"))),
             kint(cB["starting_card"] - sum(rnd["B"][k] for k in ("r1", "r2", "r3", "fd")))],
            ["Step A round 1 / 2 / 3 (3 starts)", f"{kint(rnd['A']['r1'])} / {kint(rnd['A']['r2'])} / {kint(rnd['A']['r3'])}",
             f"{kint(rnd['B']['r1'])} / {kint(rnd['B']['r2'])} / {kint(rnd['B']['r3'])}"],
            ["Step A FD polish (3 starts)", kint(rnd["A"]["fd"]), kint(rnd["B"]["fd"])],
            ["Step A total", kint(cA["starting_card"]), kint(cB["starting_card"])],
            ["Final TRF refit", kint(cA["C0_refit"] - cA["starting_card"]), kint(cB["C0_refit"] - cB["starting_card"])],
            ["Total = final card", kint(cA["C0_refit"]), kint(cB["C0_refit"])]], 28, hl=(5,)),
        ul(["1 call = 1 NGSpice run of all measured biases of 1 device (0.04-0.09 s)",
            f"Step A wall time, 18 devices x 3 starts, 10 workers: {rnd['A']['wall_s'] / 3600:.1f} h (7p), "
            f"{rnd['B']['wall_s'] / 3600:.1f} h (45p)"], 26)],
        "Cost is counted in NGSpice calls per device. Step A runs three starts per device, each through three rounds "
        "and a polish. The seven-parameter set costs about 29,000 calls per device and the 45-parameter set about "
        "78,000, of which the final refit is a few hundred.", gap=22)

    S["s16-sources"] = sec("s16-sources", [
        H2.format("Paper inputs used"),
        table(["Item", "Source", "Used how"], [22, 46, 32], [
            ["Card file structure", "paper 77 K files: 18 custom bins, tnom -196.15 C, temperature coefficients 0, "
                                    "version 4.82", "template, every tuned multiplier = 1"],
            ["Tuned values", "paper multipliers (incl. named multipliers of pMOS 0.35 x 1.6, bin 0)", "not used"],
            ["7-parameter list", "paper", "first parameter set"],
            ["45-parameter set", "screen of 115 candidates around the multiplier-1 card", "second parameter set"],
            ["Box widths", "fixed rule; set with the paper's multiplier range (1.75e-4 to 49) known", "Step A boxes"],
            ["Scored curve set", "curves scored with the paper card", "same for all rows"],
            ["Data, RRMS, current floors", "paper", "task definition"]], 26)],
        "This run removes every tuned value of the paper from the start: all multipliers in the card files are set "
        "to 1. What remains from the paper is the card file structure with its 77 K settings, the list of seven "
        "parameters, the scored curve set, the data and the metric. One input was set with the paper's result "
        "known: the box widths, chosen in October when the range of the paper's multipliers was known.", gap=24)

    S["s17-limits"] = sec("s17-limits", [
        H2.format("Limits"),
        table(["Item", "Value"], [36, 64], [
            ["RRMS in this deck", "fit error on the 18 measured devices; cards fitted on the same data"],
            ["Method design choices", "box widths, round schedule, number of starts chosen in earlier runs on these "
                                      "18 devices"],
            ["Box widths", "set with the paper's multiplier range known"],
            ["Random starts", f"redrawn on {rnd['A']['redraw_pairs']} / {rnd['B']['redraw_pairs']} of 54 device-start "
                              "pairs (7p / 45p); 7p starts 0 and 2: 1 device without finite RRMS each"],
            ["Largest remaining error", f"pMOS 0.35 x 0.55 um: {f4(pd['A'][big])} (7p), {f4(pd['B'][big])} (45p)"],
            ["pMOS Id-Vg at |Vds| 0.01 V", f"{sa['share'][pl]:.0%} (7p) / {sb['share'][pl]:.0%} (45p) of all-device "
                                           "RRMS; measured Id-Vd and Id-Vg sweeps differ at that bias"],
            ["Fitted values", "bin base values (sky130 bins add L, W, P terms); physical range of the effective "
                              "values not checked"],
            ["Step A at matched call budget", "not compared with other fitting methods"],
            ["Other data sets (NIST 4 K, IHP 300 K)", "not run with this pipeline"]], 26)],
        "All RRMS values in this deck measure how well the cards fit the 18 measured devices; there is no separate "
        "test set, because the goal is to fit measured data. The fit settings were chosen on these same devices in "
        "earlier runs, and the box widths were set with the paper's multiplier range known. Some random starts had "
        "to be redrawn because NGSpice could not run them. The pMOS Id-Vg at 0.01 V carries the largest part of the "
        "remaining error, and the measured sweeps disagree at that bias. The fitted numbers are bin base values; "
        "whether the resulting effective values are in a physical range has not been checked. Step A has not been "
        "compared with other fitting methods at the same budget.",
        gap=24)

    return S


def main() -> int:
    S = build()
    proj = OUT_ROOT / "project"
    sd = proj / "slides"
    sd.mkdir(parents=True, exist_ok=True)
    src = REPO_PROJECT
    for k in KEEP:
        if (src / "slides" / f"{k}.html").resolve() != (sd / f"{k}.html").resolve():
            shutil.copy(src / "slides" / f"{k}.html", sd / f"{k}.html")
    for k, v in S.items():
        (sd / f"{k}.html").write_text(v)
    for k in REMOVED:
        p = sd / f"{k}.html"
        if p.exists():
            p.unlink()
    deck = json.loads((src / "deck.json").read_text())
    deck["title"] = "SKY130, 77 K: BSIM4 card parameters fitted from multiplier-1 cards"
    deck["order"] = ORDER
    deck["sections"] = SECTIONS
    deck["cover"] = "title"
    (proj / "deck.json").write_text(json.dumps(deck, indent=1))
    missing = [k for k in ORDER if not (sd / f"{k}.html").exists()]
    assert not missing, missing
    assert set(S) | set(KEEP) == set(ORDER), set(ORDER) ^ (set(S) | set(KEEP))
    print(f"{len(ORDER)} slides written to {sd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
