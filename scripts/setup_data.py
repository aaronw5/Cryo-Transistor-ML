#!/usr/bin/env python3
"""Fetch measured data and the corrected NGSpice model repository."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure src/ is importable when run from a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryoml.config import (CORRECTED_REPO_DIR, CORRECTED_REPO_URL, PAPER_REPO_DIR,
                           PAPER_REPO_URL, PROCESSED_DIR, ensure_dirs)  # noqa: E402
from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.paper_data import discover_paper_paths  # noqa: E402
from cryoml.utils import get_logger  # noqa: E402

logger = get_logger("setup_data")


def clone_or_update_repo(url: str, dest: Path) -> bool:
    if dest.exists():
        if not (dest / ".git").exists():
            logger.warning("%s exists but isn't a git repo; leaving as-is", dest)
            return True
        logger.info("%s already cloned, attempting `git pull`", dest.name)
        try:
            subprocess.run(
                ["git", "-C", str(dest), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("git pull failed (offline?): %s", e.stderr.decode(errors="replace"))
            return True   # keep what we have
    git = shutil.which("git")
    if not git:
        logger.error("git not found on PATH")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("cloning %s -> %s", url, dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("git clone failed: %s", e)
        return False
    return True


def write_devices_csv(repo_root: Path) -> Path:
    out = PROCESSED_DIR / "devices.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dev_type", "L_um", "W_um", "source_folder", "paper_rrms", "paper_sigma"])
        for d in PAPER_DEVICES:
            src = str(Path(load_device_curves(d, repo_root)[0].path).parent)
            w.writerow([d.dev_type, d.L_um, d.W_um, src, d.paper_rrms or "", d.paper_sigma or ""])
    logger.info("wrote %s", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-clone", action="store_true",
                    help="do not update either required repository")
    args = ap.parse_args()

    ensure_dirs()
    if not args.skip_clone:
        clone_or_update_repo(PAPER_REPO_URL, PAPER_REPO_DIR)
        clone_or_update_repo(CORRECTED_REPO_URL, CORRECTED_REPO_DIR)

    paths = discover_paper_paths(PAPER_REPO_DIR)
    if not paths.data_files:
        raise RuntimeError(f"measured paper data unavailable under {PAPER_REPO_DIR}")
    corrected_corners = CORRECTED_REPO_DIR / "compressed-cryo-files"
    if not corrected_corners.exists():
        raise RuntimeError(f"corrected model repository unavailable under {CORRECTED_REPO_DIR}")
    logger.info("found %d measured-data files in %s",
                len(paths.data_files), PAPER_REPO_DIR)
    write_devices_csv(PAPER_REPO_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
