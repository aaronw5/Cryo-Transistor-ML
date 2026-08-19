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
VDS_LABELS = ("|Vds|<0.6V", "|Vds|0.6-1.2V", "|Vds|>1.2V")

CARDS = (
    ("paper card", "out/pdk_baseline", "sim_", "#0072B2",
     "out/pdk_baseline/pdk_baseline.json", None),
    ("7-parameter ML", "out/pdk_foundation_segmented_seven_rrms_extract",
     "fd_sim_", "#009E73",
     "out/pdk_foundation_segmented_seven_rrms_extract/summary.json",
     "noise_aware_fd"),
    ("43-parameter ML", "out/pdk_foundation_lowdrain_extract",
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


def region_table(device, curves, sims, included: set[str]) -> str:
    """Output-curve RRMS per |Vds| region, broken out by fixed gate bias."""
    short = {"paper card": "paper", "7-parameter ML": "7-par",
             "43-parameter ML": "43-par"}
    active = [name for name, _d, _p, _c, _s, _m in CARDS
              if sims[name] is not None]
    header = f"{'':>12s}" + "".join(
        f"{label:>{7 * len(active) + 3}s}" for label in VDS_LABELS)
    subhead = f"{'':>12s}" + "".join(
        "".join(f"{short[name]:>7s}" for name in active) + "   "
        for _ in VDS_LABELS)
    lines = ["output-curve RRMS, by fixed gate bias and |Vds| region:",
             header, subhead]
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
        cells = ""
        for low, high in VDS_REGIONS:
            mask = (sweep >= low) & (sweep < high)
            for name in active:
                model = np.asarray(sims[name][index],
                                   dtype=np.float64)[start:]
                value = masked_rrms(model, target, mask, denominator)
                cells += (f"{value:7.4f}" if np.isfinite(value)
                          else f"{'-':>7s}")
            cells += "   "
        rows.append((abs(float(curve.fixed)),
                     f"Vgs={abs(float(curve.fixed)):.2f}V".rjust(12) + cells))
    lines.extend(row for _bias, row in sorted(rows))
    return "\n".join(lines)


def plot_device(device, out_dir: Path) -> Path:
    tag = tag_key = device_tag(device.dev_type, device.L_um, device.W_um)
    curves = load_device_curves(device)
    sims = {name: load(d, p, tag, len(curves))
            for name, d, p, _c, _s, _m in CARDS}
    scores = {name: card_rrms(summary, method).get(tag)
              for name, _d, _p, _c, summary, method in CARDS}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for index, curve in enumerate(curves):
        sweep = np.abs(np.asarray(
            curve.Vd if curve.kind == "idvd" else curve.Vg, dtype=np.float64))
        measured = np.abs(np.asarray(curve.Id, dtype=np.float64)) * 1e6
        panels = [0] if curve.kind == "idvd" else [1, 2]
        for panel in panels:
            axes[panel].plot(sweep, measured, "o", mfc="none", mec="black",
                             ms=3, mew=0.6, zorder=1,
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
                                 color=color, lw=1.8, zorder=2,
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
    fig.text(0.5, -0.02, region_table(device, curves, sims, included),
             ha="center", va="top", family="monospace", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4,
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
