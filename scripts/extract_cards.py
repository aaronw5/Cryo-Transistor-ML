#!/usr/bin/env python3
"""Fit 77 K model cards of the paper devices with NGSpice in the loop.

Runs the card-extraction procedure of ``cryoml.extraction`` (I25-1 main
method: 3 random starts x 3 rounds of CMA-ES + TRF least squares, FD polish,
best start, final FD refit) on the multiplier-1 template of the Beall et al.
77 K card files, and writes

  <out>/devices/<tag>.json   final card, device RRMS after every stage of the
                             kept start, RRMS of every start, NGSpice run
                             counts, all start records and ranges
  <out>/summary.json         per-device final RRMS, all-device mean RRMS and
                             combined RRMS (mean of the nMOS and pMOS family
                             means), the same for every stage of the kept start,
                             run counts, configuration
  <out>/cards/               corner card files with the fitted values in each
                             device's bin and a library file
                             (``sky130_77k_extracted.lib.spice``, section
                             ``tt_77k``); every device is re-simulated from
                             the library and its RRMS is stored in the summary

RRMS is the paper metric with the published card's frozen curve set
(``card_eval``); the upstream dynamic-inclusion value is kept as
``official_dynamic_*``.

Usage (one compute-heavy job at a time; check ``ps`` / ``uptime`` first):

  PYTHONPATH=src .venv/bin/python scripts/extract_cards.py --params 7 \\
      --workers 10 --out out/extraction/seven
  PYTHONPATH=src .venv/bin/python scripts/extract_cards.py --params 45 \\
      --devices nmos_L1_W3,pmos_L2_W5 --workers 6 --out out/extraction/full45

Cost: about 29,000 (7 parameters) or 78,000 (45 parameters) NGSpice runs per
device; I25-1 took about 1 h (7) and 3.4 h (45) for 18 devices on 10 workers.
"""
from __future__ import annotations

import os

# Before any cryoml import: spice_pdk reads the card template at import, and
# every worker uses one BLAS thread.
os.environ["CRYOML_PDK_TEMPLATE"] = "mult1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml import extraction as ex  # noqa: E402
from cryoml.card_eval import aggregate  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402

ALL_TAGS = [device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", type=int, choices=sorted(ex.CONFIGS), required=True,
                    help="parameter set: 7 (Beall et al.) or 45 (I25-1 screen)")
    ap.add_argument("--devices", default="all",
                    help="comma-separated device tags, or 'all' (default: the 18 paper devices)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True, help="output directory")
    return ap.parse_args()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")


def main() -> int:
    args = parse_args()
    config = ex.CONFIGS[args.params]
    tags = ALL_TAGS if args.devices == "all" else args.devices.split(",")
    unknown = sorted(set(tags) - set(ALL_TAGS))
    if unknown:
        raise SystemExit(f"unknown devices: {unknown}")
    # pMOS devices take longer; submitting them first keeps the pool busy.
    tags = sorted(tags, key=lambda t: (not t.startswith("pmos"), ALL_TAGS.index(t)))
    out = Path(args.out)
    t0 = time.time()

    ranges = {t: ex.initial_range(config.parameters, ex.template_values(t, config.parameters))
              for t in tags}
    with get_context("spawn").Pool(max(1, args.workers)) as pool:
        results = ex.fit_devices(tags, config, ranges,
                                 map_fn=lambda fn, jobs: pool.map(fn, jobs, chunksize=1),
                                 log=lambda msg: print(msg, flush=True))
    fit_s = round(time.time() - t0, 1)

    for tag, res in results.items():
        write_json(res.to_json(), out / "devices" / f"{tag}.json")

    lib_path = ex.write_card_library({t: r.card for t, r in results.items()}, out / "cards")
    library_rrms = {t: ex.score_library(lib_path, t) for t in tags}

    def agg(values: dict[str, dict]) -> dict:
        return aggregate({t: values[t] for t in ALL_TAGS if t in values})

    final_rows = {t: {"rrms": r.final.rrms, "official_dynamic_rrms": r.final.official_dynamic_rrms}
                  for t, r in results.items()}
    stage_names = list(next(iter(results.values())).stage_rrms)
    stages = {name: agg({t: {"rrms": r.stage_rrms[name]} for t, r in results.items()})
              for name in stage_names}
    for row in stages.values():
        row.pop("official_dynamic_all_device_mean")      # not recorded per stage
    summary = {
        "method": "cryoml.extraction (I25-1 card fit, no learned correction)",
        "config": config.to_json(),
        "devices": {t: {"rrms": results[t].final.rrms,
                        "official_dynamic_rrms": results[t].final.official_dynamic_rrms,
                        "library_rrms": library_rrms[t],
                        "selected_start": results[t].selected_start,
                        "selected_eligible": results[t].selected_eligible,
                        "stage_rrms": results[t].stage_rrms,
                        "ngspice_runs": results[t].ngspice_runs} for t in ALL_TAGS if t in results},
        **agg(final_rows),
        "stages_kept_start": stages,
        "library": str(lib_path),
        # 0 when both values are equal (also both inf).
        "library_max_abs_diff": max(0.0 if library_rrms[t] == results[t].final.rrms
                                    else abs(library_rrms[t] - results[t].final.rrms)
                                    for t in tags),
        "mean_ngspice_runs_per_device": float(sum(r.ngspice_runs["total"] for r in results.values())
                                              / len(results)),
        "workers": args.workers,
        "wall_s": {"fit": fit_s, "total": round(time.time() - t0, 1)},
    }
    write_json(summary, out / "summary.json")
    print(json.dumps({k: summary[k] for k in ("all_device_mean_rrms", "combined_rrms",
                                              "nmos_rrms", "pmos_rrms", "library_max_abs_diff",
                                              "mean_ngspice_runs_per_device", "wall_s")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
