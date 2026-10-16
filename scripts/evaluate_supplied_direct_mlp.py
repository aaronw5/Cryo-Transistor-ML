#!/usr/bin/env python3
"""Evaluate the supplied one-shot inverse-MLP prediction in NGSpice.

The prediction is a seven-value vector labeled with the ``*_nom`` names used
by the upstream SKY130 77 K extraction.  Its checkpoint, device assignment,
and parameter-label convention were not supplied.  This script provisionally
assigns it to nMOS L=1 um, W=3 um (model bin 4) and makes a second explicit
assumption: each output corresponds to the factor embedded in that released
model-bin expression.  It
rescales the NGSpice readbacks by ``predicted / model4_factor``.  Results must
be interpreted as an assumed-mapping diagnostic, not a controlled MLP score.

Outputs are deliberately separate from the controlled all-device experiments:

* ``out/supplied_direct_mlp/sims_nmos_L1_W3.npz`` -- all 11 fresh NGSpice
  curves, under keys ``direct_sim_0`` ... ``direct_sim_10``;
* ``out/supplied_direct_mlp/result_nmos_L1_W3.json`` -- supplied multipliers,
  effective values, the frozen-inclusion RRMS, and per-curve scores.

This is a single-device comparison, not an all-device baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import find_device  # noqa: E402
from cryoml.metrics import score_device_new  # noqa: E402
from cryoml.pdk_extract import PARAMS7, read_bin_params  # noqa: E402
from cryoml.spice_pdk import find_bin_index, simulate_pdk  # noqa: E402
from cryoml.utils import device_tag  # noqa: E402
from pdk_ml_extract import baseline_include_tags  # noqa: E402


DEVICE = ("nmos", 1.0, 3.0)

# Values supplied for the direct I--V -> nominal-parameter MLP comparison.
PREDICTED_NOMINAL = {
    "vth0": 1.3514227867126465,
    "vsat": 2.346750259399414,
    "rdsw": 2.7952463626861572,
    "u0": 8.016054153442383,
    "nfactor": 22.230966567993164,
    "eta0": 2.436180830001831,
    "delta": 8.145480155944824,
}

# Multipliers encoded in the released nfet model-4 expressions for the
# L=1 um, W=3 um bin.  These are read from the card itself rather than the
# auxiliary upstream ``nominal_lib.py`` table; that table duplicates the U0
# value into NFACTOR for this bin, while the released card uses 8.73855.
RELEASED_NOMINAL = {
    "vth0": 1.29339,
    "vsat": 0.000175058,
    "rdsw": 2.78088,
    "u0": 11.342,
    "nfactor": 8.73855,
    "eta0": 1.0,
    "delta": 5.0,
}


def main() -> None:
    device = find_device(*DEVICE)
    if device is None:
        raise RuntimeError(f"paper device {DEVICE} is unavailable")
    tag = device_tag(device.dev_type, device.L_um, device.W_um)
    curves = load_device_curves(device)
    bin_index = find_bin_index(device.dev_type, device.L_um, device.W_um)
    if bin_index != 4:
        raise RuntimeError(f"expected model bin 4 for {tag}, found {bin_index}")

    released_effective = read_bin_params(
        device.dev_type, device.L_um, device.W_um, bin_index)
    predicted_effective = {
        parameter: (
            released_effective[parameter]
            * PREDICTED_NOMINAL[parameter]
            / RELEASED_NOMINAL[parameter]
        )
        for parameter in PARAMS7
    }
    simulations = simulate_pdk(
        device.dev_type,
        device.L_um,
        device.W_um,
        curves,
        params=predicted_effective,
        bin_index=bin_index,
    )
    frozen = score_device_new(
        device.dev_type,
        device.L_um,
        device.W_um,
        curves,
        simulations,
        include_tags=baseline_include_tags(tag),
    )
    dynamic = score_device_new(
        device.dev_type, device.L_um, device.W_um, curves, simulations)

    out_dir = ROOT / "out" / "supplied_direct_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"sims_{tag}.npz",
        **{f"direct_sim_{i}": values
           for i, values in enumerate(simulations)},
    )
    payload = {
        "comparison_scope": "single device",
        "interpretation": "assumed model-4-factor mapping",
        "device_assignment": "provisional nmos L=1 um W=3 um assignment",
        "mapping_limitation": (
            "checkpoint and parameter-label convention were not supplied; "
            "the VSAT output and model-4 factor are on very different scales"
        ),
        "device": tag,
        "dev_type": device.dev_type,
        "L_um": device.L_um,
        "W_um": device.W_um,
        "bin_index": bin_index,
        "input_output_description": (
            "direct inverse MLP: measured I-V vector to seven nominal "
            "parameter multipliers"
        ),
        "predicted_nominal_multipliers": PREDICTED_NOMINAL,
        "released_nominal_multipliers": RELEASED_NOMINAL,
        "released_effective_bsim4": released_effective,
        "predicted_effective_bsim4": predicted_effective,
        "frozen_inclusion_score": frozen,
        "dynamic_inclusion_score": dynamic,
        "simulator": "NGSpice 41",
    }
    (out_dir / f"result_{tag}.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "device": tag,
        "frozen_rrms": frozen["rrms"],
        "dynamic_rrms": dynamic["rrms"],
        "n_curves": frozen["n_curves"],
        "output": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
