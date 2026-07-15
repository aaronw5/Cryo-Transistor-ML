#!/usr/bin/env python3
"""Materialize the fixed surrogate-search stages for reporting.

The extraction run validates several candidates per device and stores every
method's parameters. To avoid per-device method cherry-picking in the
reported comparisons, this script re-simulates two CONSISTENT stages for
every device:

  emu_search       multistart inverse search through the frozen surrogate
  emu_search+fd    the same surrogate-search family, FD-polished

— and writes each as a standalone method directory (ml_<tag>.json +
sims_<tag>.npz) compatible with pdk_compare.py / make_figs.py:

  out/pdk_ml_emu_raw  emu_search
  out/pdk_ml_emu      emu_search+fd

  python scripts/make_ml_variants.py --src out/pdk_surrogate_final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR, OUT_TABLES, ensure_dirs  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("make_ml_variants")

VARIANTS = {
    "pdk_ml_emu_raw": "emu_search",
    "pdk_ml_emu": "emu_search+fd",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(OUT_DIR / "pdk_surrogate_final"))
    args = ap.parse_args()
    ensure_dirs()
    src = Path(args.src)

    source_records = {}
    for d in PAPER_DEVICES:
        tag = device_tag(d.dev_type, d.L_um, d.W_um)
        path = src / f"ml_{tag}.json"
        if not path.exists():
            raise RuntimeError(f"incomplete extraction: missing {path}")
        rec = json.load(open(path))
        if rec.get("box_mode") != "lhc10":
            raise RuntimeError(
                f"{tag}: expected current lhc10 result, got "
                f"box_mode={rec.get('box_mode')!r}"
            )
        if not rec.get("include_tags"):
            raise RuntimeError(
                f"{tag}: missing the baseline-frozen curve-inclusion set"
            )
        source_records[tag] = rec

    variant_summaries = {}
    for out_name, primary in VARIANTS.items():
        out_dir = OUT_DIR / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for d in PAPER_DEVICES:
            tag = device_tag(d.dev_type, d.L_um, d.W_um)
            rec = source_records[tag]
            pbm = rec["params_by_method"]
            if primary not in pbm:
                raise RuntimeError(
                    f"{tag}: fixed method {primary!r} is missing; refusing "
                    "to substitute a different method"
                )
            params = pbm[primary]
            curves = load_device_curves(d)
            sims = simulate_pdk(d.dev_type, d.L_um, d.W_um, curves,
                                params=params,
                                bin_index=int(rec["bin_index"]))
            include = set(rec.get("include_tags") or [])
            fixed = score_device_new(d.dev_type, d.L_um, d.W_um, curves,
                                     sims, include_tags=include)
            official = score_device_new(d.dev_type, d.L_um, d.W_um, curves,
                                        sims)
            out_rec = {
                "device": tag, "dev_type": d.dev_type,
                "L_um": d.L_um, "W_um": d.W_um,
                "bin_index": rec["bin_index"],
                "box_mode": rec["box_mode"],
                "paper_reported": d.paper_rrms,
                "method": primary,
                "selection_policy": "one fixed surrogate stage across all 18 devices",
                "production_config": rec.get("production_config"),
                "source_selection_ignored": rec.get("best_method"),
                "include_tags": sorted(include),
                "params": params,
                "params_by_method": {primary: params},
                "best_method": primary,
                "rrms": float(fixed["rrms"]),
                "rrms_official": float(official["rrms"]),
                "sigma_official": float(official["sigma"]),
                "n_curves_official": int(official["n_curves"]),
            }
            if primary == "emu_search+fd":
                out_rec["source_fd_attempts"] = rec.get("fd_attempts", [])
            json.dump(out_rec, open(out_dir / f"ml_{tag}.json", "w"),
                      indent=2)
            np.savez(out_dir / f"sims_{tag}.npz",
                     **{f"sim_{i}": np.asarray(s)
                        for i, s in enumerate(sims)})
            rows.append(out_rec)
            logger.info("%-22s %-16s rrms=%.4f official=%.4f",
                        tag, primary, out_rec["rrms"],
                        out_rec["rrms_official"])
        summary = {
            "method": primary,
            "n_devices": len(rows),
            "mean_rrms": float(np.mean([r["rrms"] for r in rows])),
            "nmos_mean": float(np.mean([r["rrms"] for r in rows
                                        if r["dev_type"] == "nmos"])),
            "pmos_mean": float(np.mean([r["rrms"] for r in rows
                                        if r["dev_type"] == "pmos"])),
            "selection_policy": "one fixed method across all 18 devices",
        }
        summary["combined_rrms"] = (summary["nmos_mean"]
                                    + summary["pmos_mean"]) / 2.0
        json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
        variant_summaries[primary] = summary
        print(out_name, json.dumps(summary))

    selected_counts = {}
    for rec in source_records.values():
        key = rec.get("best_method", "missing")
        selected_counts[key] = selected_counts.get(key, 0) + 1
    audit = {
        "policy": "fixed surrogate-search stages across all 18 devices; no "
                  "per-device method selection in reporting",
        "source_best_method_is_diagnostic_only": True,
        "source_best_method_counts": selected_counts,
        "all_variants_complete": all(
            summary["n_devices"] == len(PAPER_DEVICES)
            for summary in variant_summaries.values()
        ),
        "variants": variant_summaries,
        "fd_polish_ablation_all_device_mean": {
            "surrogate_delta": (
                variant_summaries["emu_search"]["mean_rrms"]
                - variant_summaries["emu_search+fd"]["mean_rrms"]
            ),
        },
    }
    if not audit["all_variants_complete"]:
        raise RuntimeError("fixed-method audit is incomplete")
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    json.dump(audit, open(OUT_TABLES / "fixed_method_audit.json", "w"),
              indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
