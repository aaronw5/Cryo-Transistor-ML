#!/usr/bin/env python3
"""Export and revalidate a single, globally selected fixed ML method."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import OUT_DIR  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.spice_pdk import (LIB_FILE, PDK77K_DIR, _CORNER_BASENAME,  # noqa: E402
                              _patch_params_in_bin, ensure_pdk77k, simulate_pdk)
from cryoml.utils import device_tag  # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", required=True,
        help="the fixed pdk_ml_emu directory from make_ml_variants.py; do "
             "not pass a mixed extraction-source directory",
    )
    ap.add_argument(
        "--card-dir", default=str(OUT_DIR / "pdk_ml_selected" / "cards"),
        help="canonical output directory for the globally selected card",
    )
    args = ap.parse_args()

    error = ensure_pdk77k()
    if error:
        raise RuntimeError(error)
    result_dir = Path(args.src)
    card_dir = Path(args.card_dir)
    card_dir.mkdir(parents=True, exist_ok=True)

    bins: dict[tuple[str, int], dict] = {}
    manifest_bins = []
    method_names = set()
    records_seen = set()
    for path in sorted(result_dir.glob("ml_*.json")):
        record = json.load(open(path))
        if "method" not in record:
            raise RuntimeError(
                f"{result_dir} is not a fixed-method directory; run "
                "make_ml_variants.py and export pdk_ml_emu"
            )
        method_names.add(record["method"])
        records_seen.add(record["device"])
        if record.get("box_mode") != "lhc10":
            raise RuntimeError(
                f"{record['device']}: export source is not current lhc10 data"
            )
        if not record["method"].endswith("+fd"):
            raise RuntimeError(
                f"{record['method']} is an unpolished ablation; exported "
                "cards must use the globally selected polished method"
            )
        key = record["dev_type"], int(record["bin_index"])
        params = record["params"]
        if key in bins:
            previous = bins[key]
            if not all(
                np.isclose(params[name], previous["params"][name], rtol=1e-9, atol=1e-12)
                for name in params
            ):
                raise RuntimeError(f"devices in {key} do not share one parameter vector")
            previous["devices"].append(record["device"])
        else:
            bins[key] = {"params": params, "devices": [record["device"]]}
    expected_devices = {
        device_tag(d.dev_type, d.L_um, d.W_um) for d in PAPER_DEVICES
    }
    if records_seen != expected_devices:
        raise RuntimeError(
            "fixed-method source must contain exactly all 18 devices; "
            f"missing={sorted(expected_devices - records_seen)}, "
            f"extra={sorted(records_seen - expected_devices)}"
        )
    if len(bins) != len(PAPER_DEVICES):
        raise RuntimeError(
            f"confirmed setup expects 18 distinct native bins, found {len(bins)}"
        )
    if len(method_names) != 1:
        raise RuntimeError(f"expected one uniform method, found {method_names}")
    if method_names != {"emu_search+fd"}:
        raise RuntimeError(
            "the canonical card is the fixed surrogate-search + FD method; "
            f"refusing to export {method_names}"
        )

    output_cards = {}
    for dev_type in ("nmos", "pmos"):
        source = PDK77K_DIR / _CORNER_BASENAME[dev_type]
        text = source.read_text(errors="replace")
        for (kind, bin_index), entry in sorted(bins.items()):
            if kind == dev_type:
                text = _patch_params_in_bin(text, dev_type, bin_index, entry["params"])
                manifest_bins.append({
                    "dev_type": dev_type,
                    "bin_index": bin_index,
                    **entry,
                })
        output = card_dir / _CORNER_BASENAME[dev_type]
        output.write_text(text)
        output_cards[dev_type] = output

    library = LIB_FILE.read_text()
    for dev_type, output in output_cards.items():
        library = library.replace(
            (PDK77K_DIR / _CORNER_BASENAME[dev_type]).as_posix(),
            output.as_posix(),
        )
    library_path = card_dir / "sky130_77k_ml.lib.spice"
    library_path.write_text(library)

    baseline = {
        row["device"]: row
        for row in json.load(
            open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json")
        )["devices"]
    }
    scores, saved_differences = [], []
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        curves = load_device_curves(device)
        simulations = simulate_pdk(
            device.dev_type, device.L_um, device.W_um, curves,
            library_path=library_path,
        )
        rec = json.load(open(result_dir / f"ml_{tag}.json"))
        include = set(rec.get("include_tags") or [])
        score = float(score_device_new(
            device.dev_type, device.L_um, device.W_um, curves, simulations,
            include_tags=include)["rrms"])
        saved = rec["rrms"]
        scores.append(score)
        saved_differences.append(abs(score - saved))
    nmos = [s for s, d in zip(scores, PAPER_DEVICES) if d.dev_type == "nmos"]
    pmos = [s for s, d in zip(scores, PAPER_DEVICES) if d.dev_type == "pmos"]
    manifest = {
        "source_directory": str(result_dir),
        "uniform_method": next(iter(method_names)),
        "validated_mean_rrms": float(np.mean(scores)),
        "validated_nmos_rrms": float(np.mean(nmos)),
        "validated_pmos_rrms": float(np.mean(pmos)),
        "validated_combined_rrms": float((np.mean(nmos) + np.mean(pmos)) / 2),
        "validated_wins_vs_paper_params_ngspice": int(sum(
            score < baseline[device_tag(d.dev_type, d.L_um, d.W_um)]["rrms"]
            for score, d in zip(scores, PAPER_DEVICES)
        )),
        "max_saved_score_difference": float(max(saved_differences)),
        "bins": manifest_bins,
    }
    json.dump(manifest, open(card_dir / "manifest.json", "w"), indent=2)
    print(f"wrote deployable cards to {card_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
