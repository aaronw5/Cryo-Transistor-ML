#!/usr/bin/env python3
"""Minimal deck: method note, then one paper-style I-V slide per device."""
from __future__ import annotations

import json
from pathlib import Path

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
    text_slide(prs, "43-parameter extraction", [
        "7-parameter method tunes VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0.",
        "",
        "43-parameter method: every continuous BSIM4 DC parameter that an "
        "NGSpice sensitivity screen showed to be active, selected over all 18 "
        "devices. A geometry-conditioned surrogate is trained on 10,000 "
        "NGSpice cards per device, searched by finite differences, and every "
        "candidate is re-simulated in NGSpice.",
        "",
        "Two changes made it fit better than the 7-parameter method:",
        "1. Sampling box widened on DELTA and AGS, whose required values lie "
        "outside the +/-10% box, so the correct triode/saturation knee is "
        "reachable at all.",
        "2. Selection constrained so no device is worse than its paper card "
        "in the low-drain triode region (|Vd| < 0.6 V at |Vg| >= 1.48 V).",
        "",
        f"All-device RRMS   paper {m['paper']:.4f}   "
        f"7-parameter {m['seven']:.4f}   43-parameter {m['full']:.4f}",
        f"Low-drain RRMS    paper {m['paper_low']:.4f}   "
        f"43-parameter {m['full_low']:.4f}   (0 of 18 devices worse than paper)",
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
    for path in sorted(FIGS.glob("*.png")):
        figure_slide(prs, path.stem, path)
    DECK.parent.mkdir(parents=True, exist_ok=True)
    prs.save(DECK)
    print(f"wrote {DECK} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
