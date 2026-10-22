#!/usr/bin/env python3
"""Figures for the M_bias deck (scripts/make_mbias_slides.py), sky130 77 K,
all 18 devices, from the NGSpice re-simulation written by
mbias_deck_resim.py (out/research/explore/mbias_deck/sky130/).

  iv_<tag>.png          paper-style I-V of one device: output, transfer
                        (linear), transfer (log); measured points, published
                        77 K card, C0_refit card, M_bias (SPICE-native: refit
                        card with the fold Verilog-A correction in ngspice-41)
  rrms_per_device.png   per-device RRMS, published / C0_refit / T1_inloop / M_bias
  g_map.png             g of the full M_bias model (trained on all 18 devices)

  PYTHONPATH=src .venv/bin/python scripts/research/explore/mbias_deck_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

SRC = ROOT / "out/research/explore/mbias_deck/sky130"
ALL18 = ROOT / "out/research/explore/I13_1/all18"     # I13-1 state-C rerun of the I3-2 recipe
OUT = ROOT / "slides" / "mbias_plots"
VDD = 1.85

# (key in curves.npz, legend name, colour, line style)
SERIES = (("published", "published 77 K card", "#0072B2", "--"),
          ("C0_refit", "C0_refit card (43 par., no g)", "#009E73", "-"),
          ("M_bias_native", "M_bias (43 par. + g, Verilog-A in ngspice-41)", "#D55E00", "-"))


def save(fig, path: Path, dpi: int) -> None:
    """PNG with an adaptive 256-colour palette (keeps the deck below 5 MB)."""
    from PIL import Image
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    with Image.open(path) as im:
        im.convert("RGB").quantize(colors=256).save(path, optimize=True)


def test_tags() -> set[str]:
    return set(json.loads((ROOT / "out/research/explore/real_split.json").read_text())["REAL_TEST"])


def device_label(tag: str) -> str:
    pol, l_s, w_s = tag.split("_")
    f = lambda s: s[1:].replace("p", ".")
    return f"{pol[0].upper()}MOS  L = {f(l_s)} um, W = {f(w_s)} um"


def iv_figure(tag: str, dev: dict, z) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), layout="constrained")
    sc = dev["scores"]
    for i, c in enumerate(dev["curves"]):
        kind = c["kind"]
        x = np.abs(z[f"{tag}|{i}|{'vd' if kind == 'idvd' else 'vg'}"])
        meas = np.abs(z[f"{tag}|{i}|meas"]) * 1e6
        alpha = 1.0 if c["included"] else 0.3
        panels = [0] if kind == "idvd" else [1, 2]
        for p in panels:
            axes[p].plot(x, meas, "o", mfc="none", mec="black", ms=2.8, mew=0.5, alpha=alpha,
                         zorder=1)
            for key, _name, col, ls in SERIES:
                axes[p].plot(x, np.abs(z[f"{tag}|{i}|{key}"]) * 1e6, ls, color=col, lw=1.2,
                             alpha=alpha, zorder=2)
    pos = np.concatenate([np.abs(z[f"{tag}|{i}|meas"])[c["start"]:] for i, c in
                          enumerate(dev["curves"]) if c["kind"] == "idvg"]) * 1e6
    pos = pos[pos > 0]
    axes[2].set_yscale("log")
    axes[2].set_ylim(max(np.percentile(pos, 1) * 0.3, 1e-7), pos.max() * 3)
    axes[0].set_title("Output, Id-Vd at |Vgs| = 0.37-1.85 V")
    axes[1].set_title("Transfer, Id-Vg at |Vds| = 0.01-1.85 V")
    axes[2].set_title("Transfer (log)")
    axes[0].set_xlabel("|$V_{DS}$| (V)")
    for ax in axes[1:]:
        ax.set_xlabel("|$V_{GS}$| (V)")
    for ax in axes:
        ax.set_ylabel("|$I_D$| ($\\mu$A)")
        ax.grid(True, alpha=0.25)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", mfc="none", mec="black", ms=5, label="measured 77 K")]
    handles += [Line2D([], [], color=col, ls=ls, lw=1.6, label=f"{name}   RRMS {sc[key]:.4f}")
                for key, name, col, ls in SERIES]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=11,
               handlelength=3.0, columnspacing=1.8)
    path = OUT / f"iv_{tag}.png"
    save(fig, path, 130)
    return path


def rrms_chart(check: dict) -> Path:
    res = json.loads((ALL18 / "results.json").read_text())["per_device"]
    tags = check["tags"]
    tt = test_tags()
    pub = {t: check["devices"][t]["stored"]["published"] for t in tags}
    series = (("published 77 K card", pub, "#0072B2"),
              ("C0_refit", res["C0_refit"], "#009E73"),
              ("T1_inloop", res["T1_inloop"], "#CC79A7"),
              ("M_bias", res["M_bias"], "#D55E00"))
    x = np.arange(len(tags))
    w = 0.2
    fig, ax = plt.subplots(figsize=(14, 5.4), layout="constrained")
    for k, (name, vals, col) in enumerate(series):
        h = [vals[t] for t in tags]
        ax.bar(x + (k - 1.5) * w, h, w, color=col, label=f"{name}   all-device {np.mean(h):.4f}")
    ax.set_xticks(x, [t + (" *" if t in tt else "") for t in tags], rotation=55, ha="right",
                  fontsize=9)
    ax.set_ylabel("held-out device RRMS")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=11)
    ax.set_title("* former TEST devices (opened 2026-10-21)", fontsize=10, loc="right")
    path = OUT / "rrms_per_device.png"
    save(fig, path, 170)
    return path


def g_map() -> Path:
    from cryoml.explore.i3_2_bias import correction_from_json
    corr = correction_from_json(json.loads((ALL18 / "corrections.json").read_text())["full|M_bias"])
    n = 121
    v = np.linspace(0, VDD, n)
    vg, vd = np.meshgrid(v, v)

    def g(pmos, a, b):
        phi = np.column_stack([np.full(a.size, float(pmos)), a.ravel() / VDD, b.ravel() / VDD,
                               np.zeros(a.size), np.zeros(a.size), np.zeros(a.size)])
        return corr.g(phi).reshape(a.shape)

    maps = {p: g(p, vg, vd) for p in (0, 1)}
    lim = max(np.abs(m).max() for m in maps.values())
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for ax, p in zip(axes[:2], (0, 1)):
        im = ax.pcolormesh(vg, vd, maps[p], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        cs = ax.contour(vg, vd, maps[p], levels=[-0.2, -0.1, 0, 0.1, 0.2], colors="k",
                        linewidths=0.7)
        ax.clabel(cs, fontsize=8, fmt="%.1f")
        ax.set_title(f"{'pMOS' if p else 'nMOS'}: g range [{maps[p].min():.2f}, {maps[p].max():.2f}]")
        ax.set_xlabel("|$V_{GS}$| (V)")
        ax.set_ylabel("|$V_{DS}$| (V)")
    fig.colorbar(im, ax=axes[:2], label="g  (current multiplier exp(g))", shrink=0.9)
    ax = axes[2]
    cols = ("#009E73", "#0072B2", "#E69F00", "#D55E00")
    for vds, col in zip((0.01, 0.37, 1.11, 1.85), cols):
        for p, ls in ((0, "-"), (1, "--")):
            gg = g(p, v, np.full(n, vds))
            ax.plot(v, np.exp(gg), ls, color=col, lw=1.4,
                    label=f"{'pMOS' if p else 'nMOS'}, |Vds| = {vds:.2f} V")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.set_xlabel("|$V_{GS}$| (V)")
    ax.set_ylabel("exp(g)")
    ax.set_title("exp(g) along |Vgs|")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    path = OUT / "g_map.png"
    save(fig, path, 170)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    check = json.loads((SRC / "check.json").read_text())
    with np.load(SRC / "curves.npz") as zf:
        z = {k: zf[k] for k in zf.files}
    written = [iv_figure(t, check["devices"][t], z) for t in check["tags"]]
    written += [rrms_chart(check), g_map()]
    print("\n".join(str(p.relative_to(ROOT)) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
