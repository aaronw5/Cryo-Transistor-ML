#!/usr/bin/env python3
"""PNG table of the pdk2 refit floors written by ``floors.py``.

Rows: PDK x fitted set; columns: mean tier a/b/c floor, c/b floor ratio, and
the noise-only floor (tier b / tier c truths).  All values are means over the
devices in ``<dir>/floors_<pdk>.json`` (paper RRMS).

  .venv/bin/python -I scripts/research/pdk2/floors_table.py \
      out/research/pdk2/floors/2026-10-16_final
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from cryoml.pdk2 import boxes  # noqa: E402

PDKS = (("gf180", "GF180MCU"), ("ihp", "IHP SG13G2"))
SETS = ("seven", "wide")
HEADER = ("PDK", "fitted set", "tier a floor", "tier b floor", "tier c floor",
          "c/b ratio", "noise-only floor (b / c)")
INK, MUTED, RULE = "#1f1f1f", "#5f6368", "#c4c7c5"


def table_rows(folder: Path) -> tuple[list[list[str]], list[dict]]:
    rows, reports = [], []
    for pdk, label in PDKS:
        report = json.loads((folder / f"floors_{pdk}.json").read_text())
        reports.append(report)
        summary = report["summary"]
        for set_name in SETS:
            a = summary.get(f"a/{set_name}")
            b, c = summary[f"b/{set_name}"], summary[f"c/{set_name}"]
            rows.append([
                label, f"{set_name} ({len(boxes.PDK_SETS[pdk][set_name])})",
                f"{a['mean_floor']:.4f}" if a else "not run",
                f"{b['mean_floor']:.4f}", f"{c['mean_floor']:.4f}",
                f"{c['mean_floor'] / b['mean_floor']:.2f}",
                f"{b['mean_noise']:.4f} / {c['mean_noise']:.4f}"])
    return rows, reports


def main(argv: list[str]) -> int:
    folder = Path(argv[0])
    rows, reports = table_rows(folder)
    (n_dev,), (nfev,), (irls,) = ({len(r["devices"]) for r in reports},
                                  {r["nfev"] for r in reports}, {r["irls"] for r in reports})
    caption = (f"Mean paper RRMS, {n_dev} devices per PDK, NGSpice polish ({nfev} nfev, "
               f"{irls} IRLS rounds) from the truth's fitted-parameter projection; 1 truth "
               "set per tier, default make_truth settings; tier a wide not run (0 by "
               "construction).")
    fig = plt.figure(figsize=(12.0, 2.3), dpi=200)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    widths = (0.14, 0.12, 0.13, 0.13, 0.13, 0.12, 0.23)
    xs = [sum(widths[:i]) for i in range(len(widths))]
    top, step = 0.86, 0.15
    for x, w, text in zip(xs, widths, HEADER):
        right = text not in ("PDK", "fitted set")
        ax.text(x + (w - 0.01 if right else 0), top, text, ha="right" if right else "left",
                va="center", fontsize=9.5, color=MUTED, fontweight="bold")
    ax.plot([0, 1], [top - step / 2] * 2, color=INK, lw=0.8)
    for i, row in enumerate(rows):
        y = top - step * (i + 1)
        for j, (x, w, text) in enumerate(zip(xs, widths, row)):
            right = j >= 2
            ax.text(x + (w - 0.01 if right else 0), y,
                    "" if (j == 0 and i % 2) else text,
                    ha="right" if right else "left", va="center", fontsize=10,
                    color=MUTED if text == "not run" else INK,
                    family="DejaVu Sans Mono" if right and text != "not run" else None)
        if i % 2:
            ax.plot([0, 1], [y - step / 2] * 2, color=RULE, lw=0.6)
    ax.text(0, top - step * (len(rows) + 1) + 0.03, caption, ha="left", va="center",
            fontsize=8.5, color=MUTED, wrap=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(top - step * (len(rows) + 1.4), 1)
    out = folder / "floors_table.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    print(out)
    for row in rows:
        print(" | ".join(row))
    print(caption)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
