#!/usr/bin/env python3
"""Paper-style I-V figures: measured points against three model cards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import curve_trim  # noqa: E402
from cryoml.regions import masked_rrms  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

# Output-curve sweep split, so the triode, knee and saturation regions are
# reported separately instead of averaging into one whole-curve number.
VDS_REGIONS = ((0.0, 0.6), (0.6, 1.2), (1.2, np.inf))
VDS_LABELS = ("$|V_{DS}|$<0.6V", "$|V_{DS}|$ 0.6-1.2V",
              "$|V_{DS}|$>1.2V")

CARDS = (
    ("paper card", "out/pdk_baseline", "sim_", "#0072B2",
     "out/pdk_baseline/pdk_baseline.json", None),
    ("7-parameter ML", "out/pdk_foundation_segmented_seven_rrms_extract",
     "fd_sim_", "#009E73",
     "out/pdk_foundation_segmented_seven_rrms_extract/summary.json",
     "noise_aware_fd"),
    ("43-parameter, unconstrained", "out/pdk_foundation_widened_extract",
     "fd_sim_", "#CC79A7",
     "out/pdk_foundation_widened_extract/summary.json", "noise_aware_fd"),
    ("43-parameter, constrained", "out/pdk_foundation_lowdrain_extract",
     "fd_sim_", "#D55E00",
     "out/pdk_foundation_lowdrain_extract/summary.json", "noise_aware_fd"),
)


def card_rrms(summary_path: str, method: str | None) -> dict[str, float]:
    """Per-device reported RRMS, so each legend entry carries its own score."""
    payload = json.loads((ROOT / summary_path).read_text())
    if method is None:
        return {row["device"]: float(row["rrms"])
                for row in payload["devices"]}
    return {row["device"]: float(row["methods"][method]["rrms"])
            for row in payload["devices"]}


def load(directory: str, prefix: str, tag: str, count: int):
    path = ROOT / directory / f"sims_{tag}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return [np.asarray(data[f"{prefix}{i}"], dtype=np.float64)
                for i in range(count)]


def region_values(device, curves, sims, included: set[str]):
    """Output-curve RRMS per fixed gate bias and |Vds| region."""
    active = [name for name, _d, _p, _c, _s, _m in CARDS
              if sims[name] is not None]
    rows = []
    for index, curve in enumerate(curves):
        tag = f"{curve.kind}@{abs(float(curve.fixed)):g}"
        if curve.kind != "idvd" or tag not in included:
            continue
        start, cleaned, denominator = curve_trim(
            device.dev_type, device.L_um, device.W_um, curve.kind, curve.Id)
        if denominator <= 0:
            continue
        sweep = np.abs(np.asarray(curve.Vd, dtype=np.float64))[start:]
        target = np.asarray(cleaned[start:], dtype=np.float64)
        cells = []
        for low, high in VDS_REGIONS:
            mask = (sweep >= low) & (sweep < high)
            cells.append([
                masked_rrms(np.asarray(sims[name][index],
                                       dtype=np.float64)[start:],
                            target, mask, denominator)
                for name in active])
        rows.append((abs(float(curve.fixed)), cells))
    rows.sort()
    return active, rows


def draw_region_table(ax, device, curves, sims, included: set[str]) -> None:
    """Per-region RRMS table: groups separated, best card bold in each group."""
    short = {"paper card": "paper", "7-parameter ML": "7-par",
             "43-parameter, unconstrained": "43 uncon",
             "43-parameter, constrained": "43 constr"}
    group_tint = ("#ffffff", "#f2f2f2", "#ffffff")
    active, rows = region_values(device, curves, sims, included)
    ax.axis("off")
    if not rows:
        return
    width = len(active)
    columns, column_group = [], []
    for group, label in enumerate(VDS_LABELS):
        if group:
            columns.append("")
            column_group.append(None)          # spacer between groups
        for name in active:
            columns.append(f"{label}\n{short[name]}")
            column_group.append(group)
    text, best = [], []
    for _bias, cells in rows:
        line, flags = [], []
        for group, values in enumerate(cells):
            if group:
                line.append("")
                flags.append(False)
            finite = [v for v in values if np.isfinite(v)]
            winner = min(finite) if finite else None
            for value in values:
                line.append(f"{value:.4f}" if np.isfinite(value) else "-")
                flags.append(winner is not None and value == winner)
        text.append(line)
        best.append(flags)
    table = ax.table(
        cellText=text, colLabels=columns,
        rowLabels=[f"$V_{{GS}}$ = {bias:.2f} V" for bias, _c in rows],
        cellLoc="center", rowLoc="right", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, column), cell in table.get_celld().items():
        group = column_group[column] if 0 <= column < len(column_group) else None
        if group is None and column >= 0:
            cell.set_linewidth(0)              # spacer carries no border
            cell.set_facecolor("white")
            cell.set_width(cell.get_width() * 0.35)
            continue
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#b0b0b0")
        cell.set_facecolor(group_tint[group] if group is not None else "white")
        if row == 0 or column < 0:
            cell.get_text().set_fontweight("bold")
        elif best[row - 1][column]:
            cell.get_text().set_fontweight("bold")
            cell.set_facecolor("#d7ead7")
    ax.set_title("output-curve RRMS by fixed gate bias and $|V_{DS}|$ region "
                 "(best in bold)", fontsize=10, pad=2)


def plot_device(device, out_dir: Path) -> Path:
    tag = tag_key = device_tag(device.dev_type, device.L_um, device.W_um)
    curves = load_device_curves(device)
    sims = {name: load(d, p, tag, len(curves))
            for name, d, p, _c, _s, _m in CARDS}
    scores = {name: card_rrms(summary, method).get(tag)
              for name, _d, _p, _c, summary, method in CARDS}
    fig = plt.figure(figsize=(16.5, 7.0), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[2.5, 1.0])
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    table_ax = fig.add_subplot(grid[1, :])
    for index, curve in enumerate(curves):
        sweep = np.abs(np.asarray(
            curve.Vd if curve.kind == "idvd" else curve.Vg, dtype=np.float64))
        measured = np.abs(np.asarray(curve.Id, dtype=np.float64)) * 1e6
        panels = [0] if curve.kind == "idvd" else [1, 2]
        for panel in panels:
            axes[panel].plot(sweep, measured, "o", mfc="none", mec="black",
                             ms=2.6, mew=0.5, zorder=1,
                             label="measured" if index == 0 else None)
        for (name, _d, _p, color, _s, _m) in CARDS:
            if sims[name] is None:
                continue
            model = np.abs(np.asarray(sims[name][index],
                                      dtype=np.float64)) * 1e6
            count = min(len(sweep), len(model))
            for panel in panels:
                score = scores[name]
                label = (f"{name}   RRMS {score:.4f}"
                         if score is not None else name)
                axes[panel].plot(sweep[:count], model[:count], "-",
                                 color=color, lw=1.1, zorder=2,
                                 label=label if index == 0 else None)
    axes[0].set_title("Output characteristics")
    axes[0].set_xlabel("|$V_{DS}$| (V)")
    axes[1].set_title("Transfer characteristics")
    axes[1].set_xlabel("|$V_{GS}$| (V)")
    axes[2].set_title("Transfer characteristics (log)")
    axes[2].set_xlabel("|$V_{GS}$| (V)")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_ylabel("|$I_D$| ($\\mu$A)")
        ax.grid(True, alpha=0.25)
    included = {tag for tag, row in json.loads(
        (ROOT / "out/pdk_baseline/pdk_baseline.json").read_text()
    )["per_curve"][tag_key].items() if row.get("included")}
    draw_region_table(table_ax, device, curves, sims, included)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=5,
               fontsize=11, handlelength=3.2, columnspacing=2.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "out/figs/iv_paper_style")
    args = parser.parse_args()
    for device in PAPER_DEVICES:
        print(plot_device(device, args.out_dir))


if __name__ == "__main__":
    main()
