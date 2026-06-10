"""Project-wide paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUT_DIR = ROOT / "out"
OUT_TABLES = OUT_DIR / "tables"

FIGS_DIR = ROOT / "figs"

PAPER_REPO_URL = (
    "https://github.com/UTA-Advanced-Detector-Technologies/"
    "Skywater-130nm-77K-Cryogenic-Models"
)
PAPER_REPO_DIR = RAW_DIR / "Skywater-130nm-77K-Cryogenic-Models"

CORRECTED_REPO_URL = "https://github.com/ogzamour/CryoSkywater130nm_CorrectedForNgspice"
CORRECTED_REPO_DIR = RAW_DIR / "CryoSkywater130nm_CorrectedForNgspice"

# Temperature of the cryogenic dataset (Kelvin).
TEMP_K = 77.0

# The 7 BSIM4 parameters the paper tunes.
TUNED_PARAMS = ("vth0", "u0", "nfactor", "vsat", "delta", "rdsw", "eta0")


def ensure_dirs() -> None:
    """Create all output/data directories if they do not exist."""
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        OUT_DIR / "pdk_baseline",
        OUT_DIR / "pdk_fd",
        OUT_DIR / "pdk_cma",
        OUT_DIR / "pdk_ml",
        OUT_TABLES,
        FIGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
