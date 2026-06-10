#!/usr/bin/env python3
"""Verify the local simulator against the corrected repository's saved sweeps."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import (CORRECTED_REPO_DIR, OUT_DIR, OUT_TABLES,  # noqa: E402
                           PAPER_REPO_DIR, ensure_dirs)
from cryoml.data_io import Curve  # noqa: E402
from cryoml.spice_pdk import simulate_pdk  # noqa: E402


def corrected_reference_curves() -> tuple[list[Curve], list[np.ndarray]]:
    root = CORRECTED_REPO_DIR / "ngspice-skywater-sims"
    curves: list[Curve] = []
    references: list[np.ndarray] = []
    for path in sorted((root / "vgsSweep_out").glob("*.txt")):
        values = np.loadtxt(path)
        vg = values[:, 1]
        vd = float(values[0, 5])
        current = values[:, 3]
        curves.append(Curve("idvg", vd, vg, np.full_like(vg, vd), current))
        references.append(current)
    for path in sorted((root / "vdsSweep_out").glob("*.txt")):
        values = np.loadtxt(path)
        vg = float(values[0, 1])
        vd = values[:, 5]
        current = values[:, 3]
        curves.append(Curve("idvd", vg, np.full_like(vd, vg), vd, current))
        references.append(current)
    if len(curves) != 11:
        raise RuntimeError(f"expected 11 corrected-repository sweeps, found {len(curves)}")
    return curves, references


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    ensure_dirs()
    curves, references = corrected_reference_curves()
    simulations = simulate_pdk("nmos", 0.15, 1.6, curves)
    max_abs = max(
        float(np.max(np.abs(sim - reference)))
        for sim, reference in zip(simulations, references)
    )
    max_peak_relative = max(
        float(np.max(np.abs(sim - reference)) / max(np.max(np.abs(reference)), 1e-9))
        for sim, reference in zip(simulations, references)
    )

    baseline = json.load(open(OUT_DIR / "pdk_baseline" / "pdk_baseline.json"))
    result = {
        "ngspice_version": subprocess.run(
            ["ngspice", "--version"], capture_output=True, check=True, text=True
        ).stdout.splitlines()[1].strip(),
        "corrected_repo_commit": git_head(CORRECTED_REPO_DIR),
        "measurement_repo_commit": git_head(PAPER_REPO_DIR),
        "corrected_cards_match_paper_cards": {
            device: (
                (CORRECTED_REPO_DIR / "compressed-cryo-files" / filename).read_bytes()
                == (PAPER_REPO_DIR / "cryo_models" / filename).read_bytes()
            )
            for device, filename in {
                "nmos": "sky130_fd_pr__nfet_01v8_lvt__tt_77k.corner.spice",
                "pmos": "sky130_fd_pr__pfet_01v8_lvt__tt_77k.corner.spice",
            }.items()
        },
        "corrected_reference_sweeps": len(curves),
        "corrected_reference_max_abs_A": max_abs,
        "corrected_reference_max_peak_relative_with_1nA_floor": max_peak_relative,
        **baseline["summary"],
    }
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(OUT_TABLES / "simulator_verification.json", "w"), indent=2)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
