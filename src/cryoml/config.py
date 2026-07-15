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

# The confirmed simulation setup (updated pFET card, eval decks, rrmsCalc
# metric, LHC Monte Carlo). Vendored sparsely — the heavy montecarlo outputs
# are excluded; .upstream_commit records the pinned revision.
NEW_REPO_URL = "https://github.com/ogzamour/CryoPDK_Skywater130nm_ML"
NEW_REPO_DIR = RAW_DIR / "CryoPDK_Skywater130nm_ML"
NEW_REPO_COMMIT = "39b1e518e25120104225b8fa19f4cfc61a6766b3"

# The confirmed simulator binary (new repo README pins ngspice-41). Resolution
# order: NGSPICE_BIN env var, the known local ngspice-41 install, PATH.
_NG41 = Path.home() / "cryo-ng41" / "mm" / "envs" / "ng41" / "bin" / "ngspice"


def resolve_ngspice_bin() -> str:
    import os

    env = os.environ.get("NGSPICE_BIN")
    if env:
        return env
    if _NG41.exists():
        return str(_NG41)
    return "ngspice"

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
        OUT_TABLES,
        FIGS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
