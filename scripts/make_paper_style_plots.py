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
from cryoml.utils import device_tag  # noqa: E402

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


def plot_device(device, out_dir: Path) -> Path:
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
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
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4,
               fontsize=11, handlelength=3.2, columnspacing=2.0)
    fig.suptitle(f"{device.dev_type} L={device.L_um} $\\mu$m, "
                 f"W={device.W_um} $\\mu$m", fontsize=13)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.png"
    fig.savefig(path, dpi=170)
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
