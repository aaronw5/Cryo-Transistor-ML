#!/usr/bin/env python3
"""Minimal deck: method note, then one paper-style I-V slide per device."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "out/figs/iv_paper_style"
DECK = ROOT / "slides" / "cryo_ml_paper_style.pptx"
OUT_EQ = ROOT / "slides" / "eq"


def means() -> dict[str, float]:
    base = json.loads((ROOT / "out/pdk_baseline/pdk_baseline.json").read_text())
    seven = json.loads((ROOT / "out/pdk_foundation_segmented_seven_rrms_extract"
                        "/summary.json").read_text())
    full = json.loads((ROOT / "out/pdk_foundation_lowdrain_extract"
                       "/summary.json").read_text())
    low = json.loads((ROOT / "out/tables/low_drain_diagnostic.json").read_text())
    return {
        "paper": float(np.mean([r["rrms"] for r in base["devices"]])),
        "seven": seven["methods"]["noise_aware_fd"]["all_device_mean_rrms"],
        "full": full["methods"]["noise_aware_fd"]["all_device_mean_rrms"],
        "paper_low": low["all_device_means"]["paper"]["low_drain_rrms"],
        "full_low": low["all_device_means"]["constrained"]["low_drain_rrms"],
    }


def equation_image(latex: str, path: Path, *, fontsize: int = 26) -> Path:
    """Render one equation to PNG; python-pptx cannot typeset maths itself."""
    fig = plt.figure(figsize=(0.01, 0.01))
    text = fig.text(0, 0, latex, fontsize=fontsize)
    fig.savefig(path, dpi=200, bbox_inches="tight",
                bbox_extra_artists=[text], transparent=True)
    plt.close(fig)
    return path


def equation_slide(prs, title: str, blocks: list[tuple[str, str]]) -> None:
    """Slide of caption/equation pairs, equations rendered as images."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35),
                                   Inches(12.1), Inches(0.8))
    para = box.text_frame.paragraphs[0]
    run = para.add_run()
    run.text = title
    run.font.size = Pt(30)
    run.font.bold = True
    top = 1.35
    for index, (caption, latex) in enumerate(blocks):
        label = slide.shapes.add_textbox(Inches(0.8), Inches(top),
                                         Inches(11.7), Inches(0.4))
        run = label.text_frame.paragraphs[0].add_run()
        run.text = caption
        run.font.size = Pt(17)
        top += 0.45
        image = OUT_EQ / f"eq_{abs(hash((title, index))) % 10**8}.png"
        equation_image(latex, image)
        width, height = Image.open(image).size
        scale = min(10.5 / (width / 200), 1.0 / (height / 200))
        slide.shapes.add_picture(str(image), Inches(1.2), Inches(top),
                                 Inches(width / 200 * scale),
                                 Inches(height / 200 * scale))
        top += height / 200 * scale + 0.42


def rrms_chart(path: Path) -> None:
    """Per-device RRMS for the three cards, all 18 devices."""
    base = json.loads((ROOT / "out/pdk_baseline/pdk_baseline.json").read_text())
    seven = json.loads((ROOT / "out/pdk_foundation_segmented_seven_rrms_extract"
                        "/summary.json").read_text())
    full = json.loads((ROOT / "out/pdk_foundation_lowdrain_extract"
                       "/summary.json").read_text())
    devices = [row["device"] for row in base["devices"]]
    series = (
        ("paper card", {r["device"]: r["rrms"] for r in base["devices"]},
         "#0072B2"),
        ("7-parameter ML",
         {r["device"]: r["methods"]["noise_aware_fd"]["rrms"]
          for r in seven["devices"]}, "#009E73"),
        ("43-parameter ML",
         {r["device"]: r["methods"]["noise_aware_fd"]["rrms"]
          for r in full["devices"]}, "#D55E00"),
    )
    x = np.arange(len(devices))
    width = 0.27
    fig, ax = plt.subplots(figsize=(14, 5.4), layout="constrained")
    for index, (label, values, color) in enumerate(series):
        heights = [values[d] for d in devices]
        ax.bar(x + (index - 1) * width, heights, width, color=color,
               label=f"{label}   mean {np.mean(heights):.4f}")
    ax.set_xticks(x, devices, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel("RRMS")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def text_slide(prs, title: str, lines: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.5),
                                   Inches(12.1), Inches(1.0))
    para = box.text_frame.paragraphs[0]
    para.text = title
    para.runs[0].font.size = Pt(30)
    para.runs[0].font.bold = True
    body = slide.shapes.add_textbox(Inches(0.6), Inches(1.6),
                                    Inches(12.1), Inches(5.2))
    frame = body.text_frame
    frame.word_wrap = True
    for index, line in enumerate(lines):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        run = para.add_run()          # blank lines carry no run to style
        run.text = line
        run.font.size = Pt(19)


def figure_slide(prs, title: str, image: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(0.4), Inches(0.15),
                                   Inches(12.5), Inches(0.5))
    para = box.text_frame.paragraphs[0]
    para.text = title
    para.runs[0].font.size = Pt(20)
    para.runs[0].font.bold = True
    width, height = Image.open(image).size
    scale = min(12.5 / (width / 170), 5.9 / (height / 170))
    slide.shapes.add_picture(str(image), Inches(0.4), Inches(0.85),
                             Inches(width / 170 * scale),
                             Inches(height / 170 * scale))


def main() -> None:
    m = means()
    OUT_EQ.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    text_slide(prs, "Method 1 of 3:  which parameters, and where to look", [
        "Screen.  Each continuous BSIM4 DC parameter is perturbed one at a "
        "time in NGSpice on all 18 devices.  A parameter is kept if its "
        "normalised current response exceeds 1e-4 on any device.",
        "    115 candidates  ->  43 kept.  The 7 of the previous method are a "
        "subset, so the comparison is nested.",
        "",
        "Box.  Sampling is centred on each device's own 77 K card value, "
        "resolved through BSIM binning (8 bins per polarity).",
        "    Default window  +/-10%.",
        "    DELTA and AGS  +/-80%:  NGSpice probes show the values that set "
        "the triode/saturation knee lie outside +/-10%, so the correct shape "
        "was unreachable rather than unfound.",
        "    Parameters published as zero stay pinned, since a multiplicative "
        "window around zero has no width.  PRWG is in this state on every "
        "nMOS card.",
    ])
    text_slide(prs, "Method 2 of 3:  data and surrogate", [
        "Data.  10,000 Latin-hypercube points per device, every one simulated "
        "in NGSpice.  180,000 cards, 2,046 current points each, ~2 h.",
        "    50% global, 35% in one newly opened region, 15% in two jointly.",
        "",
        "Surrogate.  A single network for all 18 devices:",
        "    in:   43 parameters, polarity, log L, log W",
        "    out:  all 2,046 currents",
        "    4 x 1024 LeakyReLU, Adam, batch 512, 200 epochs, per-device "
        "80/10/10 split.",
        "",
        "It is used only to make the search affordable: one gradient needs ~40 "
        "objective evaluations, which is microseconds through the network and "
        "hours through NGSpice.  No reported number comes from it.",
    ])
    equation_slide(prs, "Method 3 of 3:  metric, loss and selection", [
        ("Per curve, against measured data, on the points the 77 K card "
         "admits:",
         r"$\mathrm{RRMS}_c=\dfrac{\sqrt{\frac{1}{N_c}\sum_i"
         r"\left(I^{\mathrm{sim}}_i-I^{\mathrm{meas}}_i\right)^2}}"
         r"{\frac{1}{N_c}\sum_i\left|I^{\mathrm{meas}}_i\right|}"
         r"\qquad\mathrm{RRMS}_{\mathrm{dev}}=\frac{1}{C}"
         r"\sum_{c=1}^{C}\mathrm{RRMS}_c$"),
        ("Surrogate loss.  Each curve is split into 4 normalised sweep "
         "sections; sections and curves carry equal weight, plus a worst-"
         "section term:",
         r"$\mathcal{L}=\frac{1}{CS}\sum_{c,s}\left[\ell^{\log}_{cs}"
         r"+\lambda\,\ell^{\mathrm{lin}}_{cs}\right]"
         r"+\frac{1}{4}\max_{c,s}\ell_{cs}$"),
        ("Selection.  Constrained, not reweighted:",
         r"$\min_{\theta}\ \mathrm{RRMS}_{\mathrm{dev}}(\theta)"
         r"\quad\mathrm{s.t.}\quad"
         r"\mathrm{RRMS}_{\mathrm{low}}(\theta)\leq "
         r"\mathrm{RRMS}_{\mathrm{low}}(\theta_{77\mathrm{K}})$"),
    ])
    text_slide(prs, "Why a constraint and not a weighting", [
        "Low-drain region:  |Vds| < 0.6 V on output curves at |Vgs| >= 1.48 V "
        "-- triode at strong gate drive.",
        "",
        "Two reweightings were tried first and both failed, on measurement:",
        "    Absolute section weights chase the numerically largest sections, "
        "which are transfer-curve upper ends, not the triode region.  "
        "All-device RRMS 0.1823 -> 0.2016, low drain 0.0585 -> 0.0813.",
        "    A ratio-to-reference objective is degenerate: the 77 K card "
        "scores exactly 1.0 in every section, so any card better somewhere and "
        "worse elsewhere loses on the worst-ratio term.  It reselects the 77 K "
        "card.",
        "",
        "The constraint has neither failure mode: a candidate worse than the "
        "77 K card in that region is discarded outright, and the optimiser is "
        "free everywhere else.",
        "",
        "Finite differences on the surrogate produced the winning card on 4 of "
        "18 devices; 14 selected a sampled card the search could not beat.  A "
        "finite-difference stage against measured data in real NGSpice was "
        "implemented and rejected: 0.1823 -> 0.2016.",
    ])
    text_slide(prs, "Summary", [
        f"All-device RRMS:  paper {m['paper']:.4f}  ->  "
        f"7-parameter {m['seven']:.4f}  ->  43-parameter {m['full']:.4f}",
        f"    43-parameter is {100 * (1 - m['full'] / m['paper']):.0f}% better "
        f"than the paper cards, {100 * (1 - m['full'] / m['seven']):.0f}% "
        "better than the 7-parameter method.",
        "",
        f"Low-drain triode RRMS (|Vd| < 0.6 V, |Vg| >= 1.48 V):  "
        f"paper {m['paper_low']:.4f}  ->  43-parameter {m['full_low']:.4f}",
        "    0 of 18 devices worse than its paper card. Before the "
        "constraint, 12 of 18 were worse.",
        "",
        "What changed from the 7-parameter method:",
        "    Parameters:  7  ->  43, chosen by NGSpice sensitivity screen, "
        "not by hand.",
        "    Sampling box:  DELTA and AGS widened 8x; their required values "
        "sit outside +/-10% of the foundry value.",
        "    Selection:  hard constraint that no device regresses against its "
        "paper card in the low-drain triode region.",
        "",
        "Cost:  +1.8% all-device RRMS versus the unconstrained 43-parameter "
        "fit (0.1823 -> 0.1857). High-bias RRMS improved anyway "
        "(0.0601 -> 0.0581).",
    ])
    chart = ROOT / "out/figs/iv_paper_style/_rrms_all_devices.png"
    rrms_chart(chart)
    figure_slide(prs, "RRMS, all 18 devices", chart)
    for path in sorted(FIGS.glob("*.png")):
        if path.name.startswith("_"):
            continue
        figure_slide(prs, path.stem, path)
    DECK.parent.mkdir(parents=True, exist_ok=True)
    prs.save(DECK)
    print(f"wrote {DECK} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
