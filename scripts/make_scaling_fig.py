#!/usr/bin/env python3
"""Fit and plot scaling laws from out/scaling/results.csv.

Top row:    emulator val MSE vs training-set size / parameter count, and
            surrogate search loss vs number of search starts (log-log, with
            power-law fits on the geometric mean across devices).
Bottom row: final NGSpice-validated RRMS (after short polish) on the same
            axes — the metric that actually matters, including its floor.

Output: figs/scaling_laws.png
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cryoml.config import FIGS_DIR, OUT_DIR  # noqa: E402

plt.rcParams.update({"font.size": 9, "figure.dpi": 200, "axes.grid": True,
                     "grid.alpha": 0.3})


def power_fit(x, y):
    """Fit y = a * x^(-alpha); returns (alpha, fitted y over x)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    return -slope, np.exp(intercept + slope * lx)


def main() -> int:
    rows = list(csv.DictReader(open(OUT_DIR / "scaling" / "results.csv")))
    for r in rows:
        for k in ("n_data", "n_params", "n_starts"):
            r[k] = int(r[k])
        for k in ("emu_val", "search_loss", "rrms_raw", "rrms_polished"):
            r[k] = float(r[k])

    devices = sorted({r["device"] for r in rows})
    colors = {d: plt.cm.tab10(i) for i, d in enumerate(devices)}

    panels = [
        ("data", "n_data", "training simulations", "emu_val",
         "NN error on unseen test simulations", True),
        ("capacity", "n_params", "NN weights (model size)", "emu_val",
         "NN error on unseen test simulations", True),
        ("search", "n_starts", "search starting points", "search_loss",
         "best error found (NN-predicted)", True),
        ("data", "n_data", "training simulations", "rrms_polished",
         "final fitting error (real NGSpice)", False),
        ("capacity", "n_params", "NN weights (model size)", "rrms_polished",
         "final fitting error (real NGSpice)", False),
        ("search", "n_starts", "search starting points", "rrms_polished",
         "final fitting error (real NGSpice)", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    for ax, (sweep, xk, xlabel, yk, ylabel, fit) in zip(axes.flat, panels):
        sub = [r for r in rows if r["sweep"] == sweep]
        # include the shared reference cell in every sweep
        ref = [r for r in rows if r["n_data"] == 6000
               and r["arch"] == "512x512x512x512" and r["n_starts"] == 2048]
        seen = {(r["device"], r[xk]) for r in sub}
        sub += [r for r in ref if (r["device"], r[xk]) not in seen]

        by_x = defaultdict(list)
        for d in devices:
            pts = sorted([(r[xk], r[yk]) for r in sub if r["device"] == d])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", color=colors[d], lw=1, markersize=3,
                    alpha=0.65, label=d)
            for x, y in pts:
                by_x[x].append(y)
        xs = sorted(by_x)
        gmean = [float(np.exp(np.mean(np.log(np.maximum(by_x[x], 1e-12)))))
                 for x in xs]
        ax.plot(xs, gmean, "k--", lw=2, label="average of 4 transistors")
        ax.set_xscale("log")
        if fit:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6.5)
    fig.suptitle("Scaling behavior of the ML method's search stage "
                 "(no feedback rounds; 4 test transistors; axes are "
                 "logarithmic)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "scaling_laws.png")
    print(f"wrote {FIGS_DIR / 'scaling_laws.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
