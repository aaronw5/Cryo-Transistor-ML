#!/usr/bin/env python3
"""Per-device ensemble across tandem seeds + best-of-all-methods final.

Every tandem seed validates its per-device card in real NGSpice, so taking the
per-device minimum across seeds is selection by the same paper-exact NGSpice
score the pipeline uses internally (equivalent to more search starts). Total
NGSpice budget across N<=6 seeds stays under the budget-matched CMA control
(~520 evals/device/seed; CMA control = 8,500 evals/device).

Reports the N-seed tandem ensemble and the best-of-all (tandem ensemble vs the
existing ml_final per-device winners), plus per-family means, vs the CMA-8500
control."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cryoml.config import OUT_DIR, OUT_TABLES  # noqa: E402

SEED_DIRS = [
    "pdk_mlp_tandem_full",   # seed 0
    "pdk_mlp_tandem_seed1",
    "pdk_mlp_tandem_seed2",
    "pdk_mlp_tandem_seed3",
    "pdk_mlp_tandem_seed4",
    "pdk_mlp_tandem_seed5",
]


def load_dir(d: str, prefix: str = "ml_") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in glob.glob(str(OUT_DIR / d / f"{prefix}*.json")):
        r = json.load(open(f))
        if "device" in r and np.isfinite(r.get("rrms", np.nan)):
            out[r["device"]] = r
    return out


def main() -> int:
    seeds = {d: load_dir(d) for d in SEED_DIRS if (OUT_DIR / d).exists()}
    seeds = {d: v for d, v in seeds.items() if v}
    mlfinal = load_dir("pdk_ml_final_perdev")
    cma = {}
    for f in glob.glob(str(OUT_DIR / "pdk_cma" / "fd_*.json")):
        r = json.load(open(f))
        if "device" in r:
            cma[r["device"]] = r["rrms"]

    devs = sorted(next(iter(seeds.values())).keys())
    print(f"tandem seeds used: {list(seeds.keys())}\n")
    hdr = f"{'device':22s} " + " ".join(f"s{i}" for i in range(len(seeds)))
    hdr += f"  {'ens':>6s} {'mlfin':>6s} {'cma':>6s} {'BEST':>6s}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for dv in devs:
        per_seed = [seeds[d][dv]["rrms"] for d in seeds]
        ens = min(per_seed)
        ens_dir = min(seeds, key=lambda d: seeds[d][dv]["rrms"])
        mf = mlfinal.get(dv, {}).get("rrms", np.inf)
        best = min(ens, mf)
        best_src = ens_dir if ens <= mf else "ml_final"
        rows.append(dict(device=dv,
                         dev_type="nmos" if dv.startswith("nmos") else "pmos",
                         ensemble=ens, ensemble_src=ens_dir,
                         ml_final=mf if np.isfinite(mf) else None,
                         best=best, best_src=best_src,
                         cma=cma.get(dv)))
        cells = " ".join(f"{v:.3f}" for v in per_seed)
        print(f"{dv:22s} {cells}  {ens:6.3f} "
              f"{(mf if np.isfinite(mf) else float('nan')):6.3f} "
              f"{cma.get(dv, float('nan')):6.3f} {best:6.3f}")
    print("-" * len(hdr))

    def mean(key, filt=lambda r: True):
        vals = [r[key] for r in rows if filt(r) and r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    nmos = lambda r: r["dev_type"] == "nmos"  # noqa: E731
    pmos = lambda r: r["dev_type"] == "pmos"  # noqa: E731
    summary = {
        "n_seeds": len(seeds),
        "n_devices": len(rows),
        "tandem_ensemble_mean": mean("ensemble"),
        "best_of_all_mean": mean("best"),
        "ml_final_mean": mean("ml_final"),
        "cma8500_mean": mean("cma"),
        "ensemble_nmos": mean("ensemble", nmos),
        "ensemble_pmos": mean("ensemble", pmos),
        "best_nmos": mean("best", nmos),
        "best_pmos": mean("best", pmos),
        "best_wins_vs_cma": int(sum(
            1 for r in rows if r["cma"] is not None and r["best"] < r["cma"] - 1e-8)),
        "best_ties_vs_cma": int(sum(
            1 for r in rows if r["cma"] is not None and abs(r["best"] - r["cma"]) <= 1e-8)),
        "best_losses_vs_cma": int(sum(
            1 for r in rows if r["cma"] is not None and r["best"] > r["cma"] + 1e-8)),
        "devices": rows,
    }
    print(f"tandem {len(seeds)}-seed ensemble : {summary['tandem_ensemble_mean']:.4f}  "
          f"(nmos {summary['ensemble_nmos']:.4f} / pmos {summary['ensemble_pmos']:.4f})")
    print(f"ml_final per-device      : {summary['ml_final_mean']:.4f}")
    print(f"cma-8500 control         : {summary['cma8500_mean']:.4f}")
    print(f"BEST-of-all (ens+mlfinal): {summary['best_of_all_mean']:.4f}  "
          f"(nmos {summary['best_nmos']:.4f} / pmos {summary['best_pmos']:.4f})")
    print(f"BEST vs cma: {summary['best_wins_vs_cma']} wins / "
          f"{summary['best_ties_vs_cma']} ties / {summary['best_losses_vs_cma']} losses")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    out = OUT_TABLES / "tandem_ensemble.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
