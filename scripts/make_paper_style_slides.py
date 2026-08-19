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
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    text_slide(prs, "43-parameter method, step by step", [
        "1. Screen.  Perturb every continuous BSIM4 DC parameter in NGSpice on "
        "all 18 devices; keep those whose response clears the simulator "
        "repeatability floor.  43 of 115 survive.  The 7 tuned by the previous "
        "method are a subset.",
        "2. Box.  +/-10% around each device's foundry value, except DELTA and "
        "AGS, widened 8x: probes show the values needed for the "
        "triode/saturation knee sit outside +/-10%, so the correct shape was "
        "unreachable, not merely unfound.",
        "3. Data.  10,000 NGSpice cards per device (180,000 simulations): half "
        "global Latin hypercube, the rest in the newly opened regions, singly "
        "and in pairs.",
        "4. Surrogate.  One network for all devices: 43 parameters plus "
        "polarity, log L and log W in, all 2,046 currents out.  Four hidden "
        "layers of 1,024.  Loss is per curve and per sweep section.",
        "5. Search.  Finite differences on the frozen surrogate from several "
        "starts: best stored cards, section-best, low-drain-feasible, the "
        "foundry card, and the previous result.",
        "6. Validate.  Every candidate is re-simulated in real NGSpice.  No "
        "reported number comes from the surrogate.",
        "7. Select.  Discard any candidate worse than that device's foundry "
        "card in the low-drain triode region, then take the lowest RRMS among "
        "the survivors.",
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
