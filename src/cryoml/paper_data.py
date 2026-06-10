"""Helpers for locating measured files inside the cloned paper repository."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import PAPER_REPO_DIR
from .utils import get_logger

logger = get_logger("cryoml.paper_data")


@dataclass(frozen=True)
class PaperPaths:
    repo_root: Path
    data_files: tuple[Path, ...]


_DATA_PATTERNS = (
    "**/*.csv",
    "**/*.tsv",
    "**/*.dat",
    "**/*.txt",
)


def _glob_many(root: Path, patterns: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def discover_paper_paths(repo_root: Path | None = None) -> PaperPaths:
    """Locate measurement files inside the paper repository."""
    root = repo_root if repo_root is not None else PAPER_REPO_DIR
    root = Path(root)
    if not root.exists():
        logger.warning("paper repo not found at %s — run scripts/setup_data.py", root)
        return PaperPaths(root, ())

    data_files = _glob_many(root, _DATA_PATTERNS)
    # Filter obvious non-data noise (READMEs etc.) from the data set.
    data_files = [p for p in data_files if not p.name.lower().startswith("readme")]
    return PaperPaths(root, tuple(data_files))


# Match SkyWater repo's directory naming, e.g. ``nmos_FET_len_0p15_wid_1p6``.
# Also accept the simpler ``nmos_L0.15_W1.6`` style.
_GEOM_RE_PAPER = re.compile(
    r"(?P<type>nmos|pmos)_FET_len_(?P<L>[\d.p]+)_wid_(?P<W>[\d.p]+)",
    re.IGNORECASE,
)
_GEOM_RE_SIMPLE = re.compile(
    r"(?P<type>n|p)?mos?[_\-]?l(?P<L>[\d.]+)(?:u|um)?[_\-]?w(?P<W>[\d.]+)(?:u|um)?",
    re.IGNORECASE,
)


def _parse_paper_num(s: str) -> float:
    # The repo uses 'p' as the decimal point (e.g. ``0p15`` -> 0.15).
    return float(s.replace("p", "."))


def parse_geometry_from_name(name: str) -> tuple[str, float, float] | None:
    """Best-effort parse of a filename / dirname.

    Supports both ``nmos_FET_len_0p15_wid_1p6`` (paper repo) and
    ``nmos_L0.15_W1.6`` (simple) styles.
    """
    m = _GEOM_RE_PAPER.search(name)
    if m:
        try:
            return (
                m.group("type").lower(),
                _parse_paper_num(m.group("L")),
                _parse_paper_num(m.group("W")),
            )
        except ValueError:
            pass
    m = _GEOM_RE_SIMPLE.search(name)
    if not m:
        return None
    dt = m.group("type")
    dev_type = "nmos" if (dt is None or dt.lower() == "n") else "pmos"
    try:
        return dev_type, float(m.group("L")), float(m.group("W"))
    except ValueError:
        return None
