#!/usr/bin/env python3
"""Build the step-by-step Slides artifact (https://claude.ai/artifact/VXzexFpptoGiwaSzpVF57r), I25-1 revision.

Writes project/deck.json and project/slides/<id>.html under OUT_ROOT (default: slides/step_deck_artifact).
Every result number is read from out/research/explore/I25_1/decision.json (plus the per-round call counts in
I25_1/stepA/*.json and the figure statistics in out/research/explore/step_deck/numbers_i25.json written by
scripts/make_step_deck_i25.py). Figures are uploaded to the artifact's asset store first; their /_blob/ ids
are listed in BLOB below. Slides kept unchanged from the previous revision (b1-transistor, b2-model,
s01-measured, b3-fit) are copied from the existing project folder.

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

BLOB = {  # uploaded 2026-10-28 (figs/step_deck/i25_*.png) and earlier revisions (equations, basics)
    "stock": "bb63a57cb18977aa487e2bb59fd97c6d", "stages": "11b8c7fcd553290ef0555a6a5b91e5c0",
    "residual": "c6e535e4b4f18a4e4118a328fe46aeec", "lowvds": "979b95a916d8484d34a2b39d913f7afa",
    "perdev": "446eedac07c816aa1e2feb55cdf21e37",
    "iv_7p_1": "68334a95e85a1d44fe9f3bb593d9c5c2", "iv_7p_2": "0c0854e1c1f788ad6a63cc771b37a9ea",
    "iv_7p_3": "50d6ef01e9790ddd65d32b3d80e7a9c4", "iv_45p_1": "73c82bd7218b64520203707560a83a12",
    "iv_45p_2": "33d34b5cc48d35a4d4c9bb025f691e28", "iv_45p_3": "7a98d6884ac2a4303964eb8dbae8491f",
    "eq_rrms": "42728e3500a3277bc0e28dbb5d75edc7", "eq_model": "55a5e5ce6b105b9475eb2ff7bd151ea3",
    "eq_inputs": "6ea62251db2662876f8fbf0f5d1e4864", "eq_g": "508cb1226f4c8bce4c25cf740d916920",
    "eq_loss": "123d21e1cb5fbe20decf6424ba1f6b81", "eq_pen": "46e611598f4b8011c405025b4ce4e000",
    "eq_elem": "bd3f4e491727862ad5e1e61f690b7925",
}
KEEP = ["b1-transistor", "b2-model", "s01-measured", "b3-fit"]
ORDER = ["title", "b1-transistor", "b2-model", "s01-measured", "s02-stock", "s03-paper", "b3-fit",
         "s04-setup", "s04b-params", "s04-stepA", "s05-stepB", "s05c-whyg", "s06-stepC", "s07-stepD",
         "s09-results", "s09b-gmr", "s09c-stages", "s10-perdevice", "s10b-nmos", "s10c-pmos",
         "s11-iv7-1", "s11-iv7-2", "s11-iv7-3", "s12-iv45-1", "s12-iv45-2", "s12-iv45-3",
         "s14-element", "s14b-checks", "s15-cost", "s16-sources", "s17-limits"]
REMOVED = ["s05b-cardonly", "s11-iv1", "s12-iv2", "s13-iv3"]
SECTIONS = {
    "s1": {"description": "Title", "start": "title"},
    "s2": {"description": "Background: I-V curves, card, NGSpice, RRMS; measured data", "start": "b1-transistor"},
    "s3": {"description": "Stock card and multiplier-1 start card at 77 K, paper card", "start": "s02-stock"},
    "s4": {"description": "Card fit from multiplier-1 cards: setup, parameter sets, RRMS by round",
           "start": "b3-fit"},
    "s5": {"description": "Remaining error, pMOS low-|Vds| current, correction g, penalty and lambda",
           "start": "s05-stepB"},
    "s6": {"description": "Results on 18 devices: tables, GMR, per device, I-V curves for 7 and 45 parameters",
           "start": "s09-results"},
    "s7": {"description": "Verilog-A element, cost, paper inputs, limits", "start": "s14-element"},
}

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


def table(head, widths, rows, size=28, hl=()):
    s = f'<table style="width:1728px;font-size:{size}px;color:#1f252b;background:#ffffff;border:1px solid #c9ced3">'
    s += '<tr style="background:#e4e7ea">' + "".join(f'<th style="width:{w}%">{h}</th>' for h, w in zip(head, widths)) + "</tr>"
    for i, r in enumerate(rows):
        bg = ' style="background:#fbe8de"' if i in hl else ""
        s += f"<tr{bg}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return s + "</table>"


def sec(sid, body, notes, gap=20, style=None):
    st = style or SEC.format(g=gap)
    return f'<section id="{sid}" style="{st}">\n' + "\n".join(body) + f"\n<aside>{notes}</aside>\n</section>\n"


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
    nums = json.loads((ROOT / "out/research/explore/step_deck/numbers_i25.json").read_text())
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
    return d, old, nums, rounds


def build() -> dict:
    d, old, nums, rnd = load()
    A, B = d["arms"]["A"], d["arms"]["B"]
    ref = d["reference"]
    stock, m1, pap = ref["stock_77K"]["primary"], ref["mult1_no_fit"]["primary"], ref["paper"]["primary"]
    sA, sB = A["summary"], B["summary"]
    cA, cB = A["calls_per_device"], B["calls_per_device"]
    gA, gB = d["gmr"]["A"]["primary"], d["gmr"]["B"]["primary"]
    gAB = d["gmr"]["A_full_g_vs_B_full_g"]
    gold = d["gmr"]["B_full_g_vs_old_full_g"]
    st = {r["stage"]: r for r in d["stage_table"]}
    pub = {t: r["rrms"] for t, r in old["arms"]["published"].items()}
    pd = {a: {k: {t: (v["rrms"] if isinstance(v, dict) else v) for t, v in d["arms"][a]["per_device"][k].items()}
              for k in ("mult1", "C0_refit", "full_g")} for a in "AB"}
    tags = dev_order(list(pd["A"]["C0_refit"]))
    spec = d["specs"]
    nact = [45 - len(v) for v in spec["fixed_zero_center_by_device"].values()]
    ms = {a: d["arms"][a]["multistart"] for a in "AB"}
    low = nums["pmos_lowvds_45p"]["summary"]
    split = nums["gain_split"]
    S = {}

    # -- checks against the numbers in the task brief (fail loudly on mismatch)
    brief = {"stock": (3.4176, 3.1077, stock), "mult1": (12.1010, 10.9593, m1), "paper": (0.2751, 0.2595, pap),
             "7p": (0.1546, 0.1458, sA["C0_refit"]["primary"]), "7p+g": (0.0783, 0.0757, sA["full_g"]["primary"]),
             "45p": (0.1094, 0.1026, sB["C0_refit"]["primary"]), "45p+g": (0.0667, 0.0635, sB["full_g"]["primary"])}
    for k, (a, c, row) in brief.items():
        assert round(row["all_device"], 4) == a and round(row["combined"], 4) == c, (k, row)
    assert [round(cA["C0_refit"]), round(cA["full_g"]), round(cB["C0_refit"]), round(cB["full_g"])] == \
        [28953, 50185, 77696, 106986]

    def agg_row(name, row, calls):
        return [name, f4(row["all_device"]) if row["all_device"] < 1 else f"{row['all_device']:.3f}",
                f4(row["combined"]) if row["combined"] < 1 else f"{row['combined']:.3f}",
                f4(row["nmos"]) if row["nmos"] < 1 else f"{row['nmos']:.3f}",
                f4(row["pmos"]) if row["pmos"] < 1 else f"{row['pmos']:.3f}", calls]

    S["title"] = sec("title", [
        '<h1 style="font-size:72px;font-weight:600;line-height:1.15;color:#f1f0eb">SKY130, 77 K: card fit from '
        'multiplier-1 cards, 7 and 45 parameters, with correction g</h1>',
        '<div style="display:flex;flex-direction:column;gap:14px">' + "".join(
            f'<p style="font-size:32px;line-height:1.3">{t}</p>' for t in [
                "Devices: 18 (8 nMOS, 10 pMOS), measured at 77 K; simulator: NGSpice 41",
                "Start: paper's 77 K card files with every tuned multiplier set to 1; 3 random starts per device",
                f'<span style="color:#f2a68f">All-device / combined RRMS: paper card {f4(pap["all_device"])} / '
                f'{f4(pap["combined"])}; 7-parameter card + g {f4(sA["full_g"]["primary"]["all_device"])} / '
                f'{f4(sA["full_g"]["primary"]["combined"])}; 45-parameter card + g '
                f'{f4(sB["full_g"]["primary"]["all_device"])} / {f4(sB["full_g"]["primary"]["combined"])}</span>',
                '<span style="color:#b8bec4">2026-10-28 (run I25-1)</span>']) + "</div>"],
        "This deck covers fitting transistor model cards to measured SkyWater 130 nm I-V data at 77 K, starting "
        "from the paper's 77 K card files with every tuned value removed. There are 18 measured devices, 8 nMOS and "
        "10 pMOS. Every number comes from NGSpice 41 runs scored with the paper's RRMS metric. All 18 devices are "
        "fitted. Two parameter sets are fitted: the paper's 7 parameters and a 45-parameter set from a sensitivity "
        "screen. Each card is then combined with a learned current correction g.",
        style=f"background:#1f252b;color:#f1f0eb;{F};padding:128px 160px;display:flex;flex-direction:column;"
              "justify-content:center;gap:48px")

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

    S["s04-setup"] = sec("s04-setup", [
        H2.format("Step A setup: start card, random starts, rounds"),
        table(["Item", "Setting"], [26, 74], [
            ["Start card", "paper's 77 K card files, every tuned multiplier = 1 (RRMS 12.10 all-device)"],
            ["Box (log)", "center = multiplier-1 value; VTH0 x/÷ 3, other 6 of the 7 x/÷ 100, all others x/÷ 3"],
            ["Starts", "3 per device: each fitted parameter drawn log-uniform in its box (seeds 20261027 + 1000 k + i)"],
            ["Round 1", "CMA-ES from the start point (7p: 3,000 evaluations; 45p: 8,000), then TRF"],
            ["Rounds 2, 3", "box re-centered on the previous card, width x 0.5 (edge coordinates keep width); "
                            "CMA-ES + TRF"],
            ["FD polish", "TRF with finite-difference Jacobian, max 400 calls"],
            ["Selection", "per device, the start with the lowest RRMS on its own data after FD"],
        ], 26),
        ul([f"Redraw rule: a start is redrawn until NGSpice simulates every scored curve; redrawn on "
            f"{rnd['A']['redraw_pairs']} of 54 device-start pairs (7p), {rnd['B']['redraw_pairs']} of 54 (45p); "
            f"at most {rnd['A']['redraw_max']} / {rnd['B']['redraw_max']} redraws",
            "Cause: a low U0 draw plus negative size terms LU0 / WU0 / PU0 in the bin gives effective u0 &lt;= 0; "
            "BSIM4 aborts the run",
            "CMA-ES = evolutionary search; TRF = trust-region least squares; 1 NGSpice call = 1 run of all measured "
            "biases of 1 device"], 26)],
        "Step A fits the card in NGSpice. The fit starts from the paper's card files with every tuned multiplier "
        "set to 1, so no tuned value from the paper is used. Each device gets three random start points inside the "
        "box. Each start runs three rounds of CMA-ES followed by TRF; rounds 2 and 3 re-center the box on the "
        "previous card and halve its width. A finite-difference polish follows. Per device, the start with the "
        "lowest RRMS on that device's data is kept. Some random draws give a card NGSpice cannot run: a low U0 "
        "combined with the bin's negative length, width and area terms gives a non-positive effective mobility, and "
        "BSIM4 stops. Such a start is redrawn from the same random stream.", gap=18)

    ov = spec["overlap_with_old43"]
    S["s04b-params"] = sec("s04b-params", [
        H2.format("Parameter sets: 7 and 45"),
        table(["Set", "Parameters", "Selection"], [14, 62, 24], [
            ["7", "VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0", "the paper's set"],
            [f"{spec['screen']['selected_count']}", ", ".join(p.upper() for p in spec["selected"]),
             f"screen of {spec['screen']['candidate_count']} candidates"],
        ], 24),
        ul([f"Screen: each candidate changed +/-10 % one at a time around the multiplier-1 card; kept if the "
            f"current change / mean |I_meas| exceeds {spec['screen']['effective_threshold']:g} on any device; the 7 "
            f"are always kept",
            f"Overlap with the earlier 43-parameter set (screened around the paper card): {ov['n_both']} shared; "
            f"only in 43: {', '.join(p.upper() for p in ov['only_old'])}; only in 45: "
            f"{', '.join(p.upper() for p in ov['only_new'])}",
            f"Parameters with value 0 in a bin stay fixed: {min(nact)}-{max(nact)} of 45 fitted per device",
            "VTH0 = threshold voltage; U0 = mobility; VSAT = saturation velocity; RDSW = source/drain resistance; "
            "NFACTOR, ETA0, DELTA = subthreshold swing, DIBL, linear-saturation transition"], 26)],
        "The first arm fits exactly the paper's seven parameters. The second arm fits the set chosen by a "
        "sensitivity screen run on the multiplier-1 card: every one of 115 BSIM4 candidates is changed by ten "
        "percent and kept if the simulated current moves by more than one part in ten thousand of the measured "
        "mean current on any device. 45 parameters pass, 42 of them shared with the 43-parameter set used in the "
        "earlier run. Parameters whose stock value is zero in a device's bin are not fitted, so each device fits "
        "39 to 42 of the 45.", gap=18)

    S["s04-stepA"] = sec("s04-stepA", [
        H2.format("Step A: RRMS after each round"),
        img("stages", "All-device RRMS per round for the 7-parameter and 45-parameter card fits, then card refit "
                      "without and with g, log scale", 1500, 560),
        ul([f"Best start per device, all-device / combined: 7p round 1 {f4(st['round 1']['A_all'])} / "
            f"{f4(st['round 1']['A_comb'])}, after FD {f4(st['round 3 + FD']['A_all'])} / "
            f"{f4(st['round 3 + FD']['A_comb'])}; 45p round 1 {f4(st['round 1']['B_all'])} / "
            f"{f4(st['round 1']['B_comb'])}, after FD {f4(st['round 3 + FD']['B_all'])} / "
            f"{f4(st['round 3 + FD']['B_comb'])}",
            f"45p, single start for all devices, all-device: start 0 / 1 / 2 = "
            f"{ms['B']['per_start'][0]['all_device_mean_rrms']:.4f} / "
            f"{ms['B']['per_start'][1]['all_device_mean_rrms']:.4f} / "
            f"{ms['B']['per_start'][2]['all_device_mean_rrms']:.4f}; 7p start 1 "
            f"{ms['A']['per_start'][1]['all_device_mean_rrms']:.4f} (starts 0, 2: one device without finite RRMS)",
            f"7p rounds 2-3 change all-device RRMS by {abs(st['round 2']['A_all'] - st['round 3 + FD']['A_all']):.1e}"],
           26)],
        "This is the card fit alone, before g. With the paper's seven parameters, all-device RRMS goes from 0.1706 "
        "after round 1 to 0.1546 after round 2 and does not change further. With 45 parameters it goes from 0.1354 "
        "to 0.1096 after round 3 and the finite-difference polish. The last two points of each line are the card "
        "refit without g and with g, shown in detail later. Picking each device's best of three starts matters for "
        "the 45-parameter arm: any single start for all devices gives 0.118 to 0.135.", gap=18)

    S["s05-stepB"] = sec("s05-stepB", [
        H2.format("Step B: remaining error after the card fit"),
        img("residual", "Residual after the 45-parameter card fit, pMOS L 0.35 um W 5 um: top (I_card - I_meas)/den_c, "
                        "bottom ln(I_meas/I_card)", 1460, 660),
        ul([f"Device: pMOS L 0.35 x W 5 um; 45-parameter card fit, no g; device RRMS "
            f"{f4(nums['example_rrms']['45p_card'])}",
            "Top: (I_card - I_meas) / den_c, den_c = mean |I_meas| of curve c (the RRMS error term); bottom: "
            "ln(I_meas / I_card) = the target of the correction g"], 26)],
        "After the card fit the error left on each curve has a smooth shape in voltage. The top row shows the error "
        "in the units RRMS uses. The bottom row shows the log ratio of measured to card current, which is the "
        "quantity the correction g is trained to reproduce. For this pMOS the largest errors are at low drain "
        "voltage: the Id-Vg curve at |Vds| 0.01 V and the start of the Id-Vd curves.", gap=18)

    sp_ = split["B"]
    sp7 = split["A"]
    top_lab = {"pmos|idvg@0.01": "pMOS Id-Vg at |Vds| 0.01 V", "pmos|idvg@0.37": "pMOS Id-Vg at |Vds| 0.37 V"}
    S["s05c-whyg"] = sec("s05c-whyg", [
        H2.format("pMOS at |Vds| = 0.01 V: card current above measured"),
        img("lowvds", "Id-Vg at |Vds| 0.01 V for the 10 pMOS devices: measured, 45-parameter card fit, card fit + g",
            1600, 560),
        ul([f"45p card at |Vgs| 1.81-1.85 V: ln(I_card / I_meas) = +{low['card_min']:.2f} to +{low['card_max']:.2f}, "
            f"mean +{low['card_mean']:.2f}, card above measured on {low['card_pos']} of {low['n']}; card + g mean "
            f"{low['cardg_mean']:+.2f}",
            f"Share of the all-device RRMS drop from g carried by pMOS: {sp_['family']['pmos']:.2f} (45p), "
            f"{sp7['family']['pmos']:.2f} (7p); by the pMOS |Vds| 0.01 V curve alone: "
            f"{dict(sp_['top'])['pmos|idvg@0.01']:.2f} (45p), {dict(sp7['top'])['pmos|idvg@0.01']:.2f} (7p)",
            "Earlier analysis (g_why, previous cards): same sign on 10 of 10 pMOS; measured conductance Id/Vds rises "
            "from 0.03 to 0.15 VDD, the card's does not; refit with g leaves the pattern in the card"], 24)],
        "This slide shows why a correction on top of the card is needed. On all ten pMOS devices the fitted card "
        "gives too much current in the linear region at very low drain voltage. "
        "The fitted card leaves this excess on all ten devices. The correction g removes most "
        "of it, and this one curve family carries more than half of the RRMS reduction that g gives. The earlier "
        "g_why analysis on the previous cards found the same pattern: the measured pMOS conductance rises with "
        "drain voltage at the start of the Id-Vd curve, which the BSIM4 card does not produce.", gap=16)

    S["s06-stepC"] = sec("s06-stepC", [
        H2.format("Step C: correction g"),
        '<div style="display:flex;flex-direction:column;gap:14px;background:#ffffff;border:1px solid #c9ced3;padding:24px">'
        f'<img src="/_blob/{BLOB["eq_model"]}" alt="I = I_card exp(g(x1, x2))" style="width:800px;height:74px;object-fit:contain">'
        f'<img src="/_blob/{BLOB["eq_inputs"]}" alt="a, b, m, x1, x2 definitions" style="width:1400px;height:98px;object-fit:contain">'
        f'<img src="/_blob/{BLOB["eq_g"]}" alt="g = mean of 5 tanh(h_j)" style="width:760px;height:106px;object-fit:contain">'
        f'<img src="/_blob/{BLOB["eq_loss"]}" alt="training loss L" style="width:1000px;height:106px;object-fit:contain"></div>',
        ul(["I_card = NGSpice current of the device card; exp(g) = current factor, |g| &lt; 1; g = 0 gives I = I_card",
            "s = +1 (nMOS), -1 (pMOS); Vgd = Vgs - Vds; EPS = 0.01 V; VDD = 1.85 V; x1, x2 unchanged under drain-source exchange",
            "h_j = MLP 3-16-16-1 (weighted sums + tanh) on is_pmos and z1, z2 = standardized x1, x2; seeds 0-4",
            "D = 18 devices, Step A cards fixed; W = network weights; Adam, rate 3e-3, 2,000 steps, L2 1e-4; one g per arm",
            "Then each card is refit with g fixed (TRF, max 400 calls, Step A box)"], 26)],
        "The correction multiplies the card current by exp(g), where g is a small neural network of the two bias "
        "inputs x1 and x2. The inputs are built so that swapping drain and source gives the same value, which a "
        "circuit simulator needs. g is the mean of five networks trained from seeds 0 to 4, each squashed by tanh, "
        "so the factor stays between about 0.37 and 2.7. The loss is the mean device RRMS over the 18 devices with g "
        "applied, plus weight decay and the penalty on the next slide. One g is trained per parameter arm; then each "
        "device's card is refit with that g in the loop.", gap=18)

    S["s07-stepD"] = sec("s07-stepD", [
        H2.format("Step D: penalty and lambda"),
        img("eq_pen", "penalty P", 1500, 104, pad=28),
        ul(["gm = dId/dVgs, gds = dId/dVds; gm / I = d ln I / d|Vgs|, gds / I = d ln I / d|Vds| (1/V); P = mean "
            "negative part of both",
            "Grid: 25 mV steps, 75 x 75 points, 0-1.85 V, forward bias, I_card &gt; 1 nA",
            "lambda = penalty weight; grid {0, 1, 2, 5, 10} V",
            "Selection: 18 devices in 3 groups; train g on 2 groups, refit the 3rd group's cards with g fixed; "
            "lambda* = minimum mean ln RRMS (tie: larger lambda); 15 trainings, 90 refits",
            f"lambda* = {A['lambda_star']['full']:g} V (7p), {B['lambda_star']['full']:g} V (45p)"], 28, 1.4)],
        "A correction that fits the curves can still make gm or gds negative between the measured biases, which a "
        "circuit simulator does not accept. The penalty P measures the negative part of both log-slopes of the "
        "corrected current on a dense bias grid. lambda sets its weight in the loss. lambda is chosen from five "
        "values by three-group cross-validation over the 18 devices; the chosen value is 5 V for both arms.",
        gap=28)

    S["s09-results"] = sec("s09-results", [
        H2.format("Results, 18 devices, full data"),
        table(["Card", "All-device", "Combined", "nMOS mean", "pMOS mean", "NGSpice calls / device"],
              [34, 12, 12, 12, 12, 18], [
            agg_row("Stock card, 77 K", stock, "0"),
            agg_row("Paper's 77 K card, multipliers = 1", m1, "0"),
            agg_row("Paper card", pap, "-"),
            agg_row("7p card fit", sA["C0_refit"]["primary"], kint(cA["C0_refit"])),
            agg_row("7p card fit + g", sA["full_g"]["primary"], kint(cA["full_g"])),
            agg_row("45p card fit", sB["C0_refit"]["primary"], kint(cB["C0_refit"])),
            agg_row("45p card fit + g", sB["full_g"]["primary"], kint(cB["full_g"]))], 28, hl=(4, 6)),
        ul(["7p / 45p = the paper's 7 parameters / the 45-parameter set; card fit = Step A card refit without g "
            "(TRF, max 400 calls)",
            "+ g: g trained on all 18 devices, each card refit with g fixed; RRMS = NGSpice current x exp(g)",
            "Calls: Step A over 3 starts + lambda selection / 18 + own refit"], 26)],
        "This table collects every card on the same 18 devices and the same scored curve set. Starting from the "
        "multiplier-1 card at 12.1, the seven-parameter fit reaches 0.1546 all-device and the 45-parameter fit "
        "0.1094; the paper card is at 0.2751. Adding g and refitting gives 0.0783 with seven parameters and 0.0667 "
        "with 45. Combined values are in their own column and are compared only with each other. The last column is "
        "the number of NGSpice runs per device, summed over the three starts.", gap=22)

    def grow(lab, a, b, wa, wb):
        return [lab, ci(a), f"{a['wins']}/18", ci(b), f"{b['wins']}/18"]
    S["s09b-gmr"] = sec("s09b-gmr", [
        H2.format("Paired device ratios (GMR)"),
        table(["Comparison", "7p: GMR [95% CI]", "lower", "45p: GMR [95% CI]", "lower"], [34, 22, 11, 22, 11], [
            grow("card + g vs card", gA["full_g_vs_C0_refit"], gB["full_g_vs_C0_refit"], 0, 0),
            grow("card + g vs paper card", gA["full_g_vs_paper"], gB["full_g_vs_paper"], 0, 0),
            grow("card vs paper card", gA["C0_refit_vs_paper"], gB["C0_refit_vs_paper"], 0, 0)], 28),
        table(["Comparison", "GMR [95% CI]", "first lower"], [56, 28, 16], [
            ["7p card + g vs 45p card + g", ci(gAB), f"{gAB['wins']}/18"],
            ["45p card + g vs earlier 43p card + g (start: paper card file with tuned values)", ci(gold),
             f"{gold['wins']}/18"]], 28),
        ul(["GMR(A, B) = geometric mean over the 18 devices of RRMS_A / RRMS_B; &lt; 1: A lower",
            "95% CI = device bootstrap, 10,000 resamples; lower = devices with RRMS_A &lt; RRMS_B"], 26)],
        "The geometric mean ratio compares two cards device by device. g lowers the RRMS on all 18 devices in both "
        "arms: by a factor 0.52 with seven parameters and 0.63 with 45. Against the paper card, the card with g is "
        "at 0.28 and 0.23 of the paper's RRMS. With g, the seven-parameter card is 1.21 times the 45-parameter card, "
        "lower on 2 of 18 devices. The last row compares with the earlier run that started from the paper's card "
        "file with its tuned values: the interval contains 1.", gap=24)

    order = ["round 1", "round 2", "round 3", "round 3 + FD", "C0_refit", "g full data"]
    names = ["Round 1", "Round 2", "Round 3", "Round 3 + FD", "Card refit, no g", "Card refit + g"]
    calls = {"round 1": (rnd["A"]["r1"], rnd["B"]["r1"]), "round 2": (rnd["A"]["r2"], rnd["B"]["r2"]),
             "round 3": (rnd["A"]["r3"], rnd["B"]["r3"]), "round 3 + FD": (rnd["A"]["fd"], rnd["B"]["fd"]),
             "C0_refit": (cA["C0_refit"] - cA["starting_card"], cB["C0_refit"] - cB["starting_card"]),
             "g full data": (cA["full_g"] - cA["starting_card"], cB["full_g"] - cB["starting_card"])}
    S["s09c-stages"] = sec("s09c-stages", [
        H2.format("RRMS by stage, 7 and 45 parameters"),
        table(["Stage", "7p all-device", "7p combined", "45p all-device", "45p combined", "calls 7p / 45p"],
              [22, 14, 14, 14, 14, 22],
              [[n, f4(st[s]["A_all"]), f4(st[s]["A_comb"]), f4(st[s]["B_all"]), f4(st[s]["B_comb"]),
                f"{kint(calls[s][0])} / {kint(calls[s][1])}"] for n, s in zip(names, order)], 28, hl=(5,)),
        ul(["Rounds: best start per device; calls per device summed over the 3 starts",
            "Card refit + g calls = lambda selection / 18 + own refit",
            f"7p reference chain from the box center (no random start): {f4(ref['A_center_start_chain']['all_device'])}"
            f" / {f4(ref['A_center_start_chain']['combined'])}"], 26)],
        "This table lists all-device and combined RRMS after every stage for both arms, with the NGSpice calls each "
        "stage costs per device. The seven-parameter fit stops improving after round 2. The 45-parameter fit "
        "improves in every round. The card refit without g changes little because it starts from the Step A card "
        "with the same objective. The large step is g with the card refit, which costs mostly the lambda selection.",
        gap=24)

    S["s10-perdevice"] = sec("s10-perdevice", [
        H2.format("Per-device RRMS"),
        '<div style="display:flex;flex-direction:row;gap:32px;align-items:center">'
        + img("perdev", "Per-device RRMS bars, log scale: paper card, 7p card fit, 7p + g, 45p card fit, 45p + g",
              1300, 820)
        + ul([f"g lowers RRMS on 18 of 18 devices in both arms",
              f"Largest with g: pMOS 0.35 x 0.55 um, {f4(pd['A']['full_g']['pmos_L0p35_W0p55'])} (7p), "
              f"{f4(pd['B']['full_g']['pmos_L0p35_W0p55'])} (45p)",
              "Tables: next two slides"], 28, extra=";width:390px") + "</div>"],
        "Each device has five bars on a log scale: paper card, seven-parameter card fit, seven-parameter fit with g, "
        "45-parameter card fit, and 45-parameter fit with g. Every fit is below the paper card on every device, and "
        "g lowers the RRMS on every device in both arms. The largest remaining error is the narrow pMOS L 0.35 x W "
        "0.55 um, at about 0.3 with g.", gap=20)

    def dev_rows(fam):
        rows = []
        for t in [t for t in tags if t.startswith(fam)]:
            rows.append([pretty(t), f4(pub[t]), f4(pd["A"]["C0_refit"][t]), f4(pd["A"]["full_g"][t]),
                         f4(pd["B"]["C0_refit"][t]), f4(pd["B"]["full_g"][t])])
        rows.append([f"{'nMOS' if fam == 'nmos' else 'pMOS'} mean", f4(pap[fam]), f4(sA["C0_refit"]["primary"][fam]),
                     f4(sA["full_g"]["primary"][fam]), f4(sB["C0_refit"]["primary"][fam]),
                     f4(sB["full_g"]["primary"][fam])])
        return rows
    head = ["Device, L x W (um)", "Paper card", "7p card", "7p + g", "45p card", "45p + g"]
    S["s10b-nmos"] = sec("s10b-nmos", [
        H2.format("Per-device RRMS, nMOS"),
        table(head, [30, 14, 14, 14, 14, 14], dev_rows("nmos"), 28, hl=(8,)),
        ul(["7p / 45p card = card fit without g; + g = card refit with g fixed; all values NGSpice re-runs of the stored cards"], 26)],
        "These are the eight nMOS devices. The paper card is between 0.06 and 0.16. The seven-parameter fit with g "
        "is between 0.02 and 0.11, and the 45-parameter fit with g between 0.017 and 0.08. The nMOS mean is the "
        "first half of the combined value.", gap=24)
    S["s10c-pmos"] = sec("s10c-pmos", [
        H2.format("Per-device RRMS, pMOS"),
        table(head, [30, 14, 14, 14, 14, 14], dev_rows("pmos"), 26, hl=(10,))],
        "These are the ten pMOS devices. pMOS errors are larger than nMOS for every card. g has its largest effect "
        "here: for example pMOS L 0.35 x W 5 um goes from 0.1575 to 0.0573 with seven parameters. The narrow "
        "pMOS L 0.35 x W 0.55 um stays the largest error in every column.", gap=24)

    groups = ["nMOS L 0.15-8 um", "nMOS L 20, 100 um; pMOS L 0.35, 0.5 x 0.42 um", "pMOS L 0.5 x 0.64 - 8 um"]
    for arm, short, pre, col in (("A", "7p", "s11-iv7", "green"), ("B", "45p", "s12-iv45", "blue")):
        n = "7" if arm == "A" else "45"
        for k in range(3):
            sid = f"{pre}-{k + 1}"
            S[sid] = sec(sid, [
                f'<h2 style="font-size:44px;font-weight:600;line-height:1.1;color:#1f252b">I-V, {n}-parameter card '
                f'fit and + g ({k + 1}/3)</h2>',
                img(f"iv_{short}_{k + 1}", f"Id-Vd and Id-Vg, {groups[k]}: measured, paper card NGSpice, {n}-parameter "
                                           f"card fit NGSpice, card fit + g", 1724, 820),
                f'<p style="font-size:28px;line-height:1.3">{groups[k]}. Left: Id-Vd; right: Id-Vg (log). Panel '
                f'title: device RRMS. Grey dashed: paper card; {col}: {n}p card; red: {n}p card + g</p>'],
                f"These panels show six devices with the {n}-parameter card fit. Black points are measured; faded "
                f"points are below the current floor or on curves the paper card does not score. The grey dashed line "
                f"is the paper card and the {col} line the {n}-parameter card, both NGSpice runs. The red line is the "
                f"same card refit with g, computed as the NGSpice current times exp(g). The panel title gives each "
                f"device's RRMS for the paper card, the card, and the card with g.",
                style=f"background:#f6f5f1;color:#1f252b;{F};padding:40px 64px 36px;display:flex;"
                      "flex-direction:column;gap:14px")

    S["s14-element"] = sec("s14-element", [
        H2.format("Verilog-A element"),
        img("eq_elem", "I(d,s) = I_card + (e^g - 1) I_ch", 1000, 106, pad=32),
        ul(["I_card = card transistor current; I_ch = channel current of a card-transistor copy",
            "Verilog-A = language for analog device models; compiled with OpenVAF, loaded through OSDI (NGSpice 41 "
            "compiled-model interface)",
            "Checks (next slide): run I17-1 on the earlier 43p card + g (lambda 1 V); element RRMS = Python RRMS within 1.8e-4 per device",
            "Not rerun on the I25-1 cards and g"], 28, 1.45)],
        "To use g in circuit simulation, it is written as a Verilog-A element that adds (exp(g) - 1) times the "
        "channel current to the card transistor. The element is compiled with OpenVAF and loaded into NGSpice 41. "
        "The checks on the next slide were run on the earlier 43-parameter card with g; the same element code takes "
        "the new cards and g, but the checks have not been repeated for them.", gap=36)

    S["s14b-checks"] = sec("s14b-checks", [
        H2.format("Element checks (I17-1, earlier 43p card + g)"),
        table(["Check (I17-1)", "Value", "Limit", "Result"], [58, 14, 12, 16], [
            ["Element vs card x exp(g), tight tolerance: points off", "0", "0", "pass"],
            ["Convergence messages, non-finite, failed runs", "0", "0", "pass"],
            ["|I| at Vds = 0 (A)", "4.9e-17", "1e-12", "pass"],
            ["Verilog-A g - Python g", "2.8e-16", "1e-9", "pass"],
            ["Drain-source exchange, |d ln I|", "2.2e-5", "1e-3", "pass"],
            ["AC gm, gds vs element equation (relative)", "1.1e-5", "1e-3", "pass"],
            ["Capacitances vs card (relative)", "3.3e-14", "1e-3", "pass"],
            ["Inverters, 5-stage ring oscillators, mirrors failing", "0", "0", "pass"],
            ["Default-tolerance difference, |I| &gt; 1 nA (relative)", "1.8e-3", "1e-3", "restriction"],
            ["gds sign changes from g (curves)", "6 of 90", "0", "restriction"],
            ["Run time, element / card (max)", "4.60", "2", "restriction"]], 28, hl=(8, 9, 10)),
        ul(["gds sign changes: pMOS 2 x 5, 8 x 0.84, 8 x 1.6, 8 x 5 um; also in Python card x exp(g)",
            "Label (I17-1): deployable with restrictions"], 26)],
        "The element passed the eight core checks on the earlier cards: agreement with the Python model, zero "
        "current at Vds = 0, drain-source symmetry, AC small-signal values, capacitances and small circuits. Three "
        "items exceeded their limits and are reported as restrictions: a 1.8e-3 difference at default simulator "
        "tolerance, gds sign changes on 6 of 90 curves, and up to 4.6 times the card's run time.", gap=20)

    S["s15-cost"] = sec("s15-cost", [
        H2.format("Cost in NGSpice calls"),
        table(["Stage, per device", "7 parameters", "45 parameters"], [50, 25, 25], [
            ["Step A start draws (redraw rule)",
             kint(cA["starting_card"] - sum(rnd["A"][k] for k in ("r1", "r2", "r3", "fd"))),
             kint(cB["starting_card"] - sum(rnd["B"][k] for k in ("r1", "r2", "r3", "fd")))],
            ["Step A round 1 / 2 / 3 (3 starts)", f"{kint(rnd['A']['r1'])} / {kint(rnd['A']['r2'])} / {kint(rnd['A']['r3'])}",
             f"{kint(rnd['B']['r1'])} / {kint(rnd['B']['r2'])} / {kint(rnd['B']['r3'])}"],
            ["Step A FD polish (3 starts)", kint(rnd["A"]["fd"]), kint(rnd["B"]["fd"])],
            ["Step A total", kint(cA["starting_card"]), kint(cB["starting_card"])],
            ["Card fit = Step A + refit without g", kint(cA["C0_refit"]), kint(cB["C0_refit"])],
            ["lambda selection / 18", kint(cA["selection_calls_per_device"]), kint(cB["selection_calls_per_device"])],
            ["Card refit + g", kint(cA["full_g"] - cA["starting_card"] - cA["selection_calls_per_device"]),
             kint(cB["full_g"] - cB["starting_card"] - cB["selection_calls_per_device"])],
            ["Card fit + g = Step A + lambda selection / 18 + refit + g", kint(cA["full_g"]), kint(cB["full_g"])]],
            26, hl=(4, 7)),
        ul(["1 call = 1 NGSpice run of all measured biases of 1 device (0.04-0.09 s)",
            f"Step A wall time, 18 devices x 3 starts, 10 workers: {rnd['A']['wall_s'] / 3600:.1f} h (7p), "
            f"{rnd['B']['wall_s'] / 3600:.1f} h (45p)",
            f"lambda selection totals {kint(cA['selection_calls_total'])} / {kint(cB['selection_calls_total'])} calls "
            "over all nested selections run in I25-1; the full-data g uses the 18-device one"], 26)],
        "Cost is counted in NGSpice calls per device. Step A runs three starts per device, each through three "
        "rounds and a polish. The seven-parameter arm costs about 29,000 calls per device, the 45-parameter arm "
        "about 77,000. The lambda selection for g adds 21,000 and 29,000 calls per device as run, because it "
        "trains and refits many times. The final card refit with g costs a few hundred calls.", gap=22)

    S["s16-sources"] = sec("s16-sources", [
        H2.format("Paper inputs used"),
        table(["Item", "Source", "Used how"], [22, 46, 32], [
            ["Card file structure", "paper 77 K files: 18 custom bins, tnom -196.15 C, temperature coefficients 0, "
                                    "version 4.82", "template, every tuned multiplier = 1"],
            ["Tuned values", "paper multipliers (incl. named multipliers of pMOS 0.35 x 1.6, bin 0)", "not used"],
            ["7-parameter list", "paper", "arm 1 parameters"],
            ["45-parameter set", "screen of 115 candidates around the multiplier-1 card", "arm 2 parameters"],
            ["Box widths", "fixed rule; set with the paper's multiplier range (1.75e-4 to 49) known", "Step A boxes"],
            ["Scored curve set", "curves scored with the paper card", "same for all rows"],
            ["Data, RRMS, current floors", "paper", "task definition"]], 26)],
        "This run removes every tuned value of the paper from the start: all multipliers in the card files are set "
        "to 1. What remains from the paper is the card file structure with its 77 K settings, the list of seven "
        "parameters, the scored curve set, the data and the metric. One input was set with the paper's result "
        "known: the box widths, chosen in October when the range of the paper's multipliers was known.", gap=24)

    S["s17-limits"] = sec("s17-limits", [
        H2.format("Limits"),
        table(["Item", "Value"], [40, 60], [
            ["RRMS in this deck", "fit error on the 18 measured devices; cards and g fitted on the same data"],
            ["Method design choices", "g inputs, EPS, penalty, lambda grid chosen on these 18 devices (earlier runs)"],
            ["Box widths", "set with the paper's multiplier range known"],
            ["Random starts", f"redrawn on {rnd['A']['redraw_pairs']} / {rnd['B']['redraw_pairs']} of 54 device-start "
                              "pairs (7p / 45p); 7p starts 0 and 2: 1 device without finite RRMS each"],
            ["Element checks", "run on the earlier 43p card + g; not rerun on these cards"],
            ["Largest remaining error", f"pMOS 0.35 x 0.55 um: {f4(pd['A']['full_g']['pmos_L0p35_W0p55'])} (7p + g), "
                                        f"{f4(pd['B']['full_g']['pmos_L0p35_W0p55'])} (45p + g)"],
            ["Step A at matched call budget", "not compared with other fitting methods"],
            ["Other data sets (NIST 4 K, IHP 300 K)", "not run with this pipeline"]], 28)],
        "All RRMS values in this deck measure how well the cards and g fit the 18 measured devices; there is no "
        "separate test set, because the goal is to fit measured data. The design of g was chosen on these same "
        "devices in earlier runs, and the box widths were set with the paper's multiplier range known. Some random "
        "starts had to be redrawn because NGSpice could not run them. The Verilog-A element checks were not "
        "repeated for the new cards. Step A has not been compared with other fitting methods at the same budget.",
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
    deck["title"] = "SKY130, 77 K: card fit from multiplier-1 cards, with correction g"
    deck["order"] = ORDER
    deck["sections"] = SECTIONS
    deck["cover"] = "title"
    (proj / "deck.json").write_text(json.dumps(deck, indent=1))
    missing = [k for k in ORDER if not (sd / f"{k}.html").exists()]
    assert not missing, missing
    print(f"{len(ORDER)} slides written to {sd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
