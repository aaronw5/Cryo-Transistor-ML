#!/usr/bin/env python3
"""DEV anytime curves grouped by method family, surrogate vs NGSpice-only.

One panel per family (full-Jacobian hybrid TR, sketched Jacobians, budget-aware
racing). In each panel the surrogate arm is solid and the NGSpice-only control
is dashed (all-device mean over the DEV targets); CMA-ES + TRF is drawn in
grey in every panel as the shared baseline.

  .venv/bin/python scripts/research/plot_dev_families.py \
      --compare out/research/dev/compare.json --out <png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = [
    ("Full Jacobian (hybrid TR)", "hybrid_surrogate", "hybrid_none"),
    ("Sketched Jacobian, k = 8", "sketch_surrogate_k8", "sketch_none_k8"),
    ("Budget-aware racing", "anytime_sur_presearch_race", "anytime_none_race"),
]
BASELINE = "cmaes_trf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", default="out/research/dev/compare.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = json.loads(Path(args.compare).read_text())["arms"]
    ink, muted, accent, base = "#0b0b0b", "#52514e", "#2a78d6", "#9a9892"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=150, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (title, sur, none) in zip(axes, FAMILIES):
        ax.set_facecolor("#fcfcfb")
        for name, style, color, label in (
                (BASELINE, "-", base, "CMA-ES + TRF"),
                (none, "--", accent, "NGSpice only"),
                (sur, "-", accent, "with surrogate")):
            arm = arms.get(name)
            if not arm or "curve" not in arm:
                continue
            c = arm["curve"]
            ax.plot(c["grid"], c["mean"], ls=style, color=color, lw=2,
                    label=f"{label} (n={c['n_targets']})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=ink)
        ax.set_xlabel("NGSpice calls per target (Jacobian probes included)",
                      fontsize=8, color=ink)
        ax.grid(True, which="major", color="#e4e3df", lw=0.6)
        ax.tick_params(colors=muted, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0].set_ylabel("all-device mean paper RRMS, best so far", color=ink)
    fig.suptitle("DEV, 36 simulated targets (sets 0 and 1): surrogate arm vs "
                 "NGSpice-only control, by method family", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
