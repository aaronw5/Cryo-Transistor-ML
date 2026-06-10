"""Load the paper's measured I-V curves for the Table 6 devices."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import PAPER_REPO_DIR
from .devices import Device, PAPER_DEVICES
from .paper_data import discover_paper_paths, parse_geometry_from_name
from .utils import get_logger

logger = get_logger("cryoml.data_io")


@dataclass
class Curve:
    """A measured I-V sweep on a single device."""

    kind: str                # "idvg" or "idvd"
    fixed: float             # VDS (for idvg) or VGS (for idvd)
    Vg: np.ndarray
    Vd: np.ndarray
    Id: np.ndarray
    path: str = ""
    source: str = "measured"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.Id))


_FILENAME_RE = re.compile(
    r"(?P<kind>idvg|idvd)_(?:Vd|Vg)(?P<fixed>[\d.p\-]+)\.csv$",
    re.IGNORECASE,
)

_MEASUREMENT_DIR_OVERRIDES = {
    ("pmos", 0.35, 1.6): "pmos_FET_len_0.35_wid_1.6",
}


def _load_csv_curves(path: Path, dev_type: str) -> list[Curve]:
    """Parse a paper-repo CSV.

    Each CSV holds one I-V curve. The fixed bias and kind are encoded in
    the filename (e.g. ``idvg_Vd0p01.csv`` is an Id-Vg sweep at VDS=0.01).

    Columns in the file are: ``Vd_src, Vg_src, Id, Ig`` (header row).
    """
    fn_match = _FILENAME_RE.search(path.name)
    if not fn_match:
        # Fall back to single-curve no-name parsing.
        kind = None
        fixed = None
    else:
        kind = fn_match.group("kind").lower()
        try:
            fixed = float(fn_match.group("fixed").replace("p", "."))
        except ValueError:
            fixed = None
        # For pMOS the fixed voltage is positive in the filename but negative
        # in the physical sweep — handled below via the sign flip on V columns.

    rows: list[list[str]] = []
    with path.open("r") as f:
        reader = csv.reader(f)
        header: list[str] | None = None
        for row in reader:
            if not row:
                continue
            row = [c.strip() for c in row]
            if header is None:
                if any(_is_non_numeric(c) for c in row):
                    header = [c.lower().strip() for c in row]
                    continue
                header = ["vd_src", "vg_src", "id", "ig"]
            rows.append(row)
    if not rows or header is None:
        return []

    def col(*names) -> int | None:
        for n in names:
            for i, h in enumerate(header):
                if h == n.lower():
                    return i
        return None

    iVd = col("vd_src", "vd", "vds", "v_d")
    iVg = col("vg_src", "vg", "vgs", "v_g")
    iId = col("id", "ids", "i_d", "drain")
    if iVd is None or iVg is None or iId is None:
        logger.debug("skipping %s — header %s missing Vd/Vg/Id", path, header)
        return []

    Vd_list, Vg_list, Id_list = [], [], []
    for r in rows:
        try:
            Vd_list.append(float(r[iVd]))
            Vg_list.append(float(r[iVg]))
            Id_list.append(float(r[iId]))
        except (ValueError, IndexError):
            continue
    if not Vd_list:
        return []

    Vd_arr = np.array(Vd_list, dtype=np.float64)
    Vg_arr = np.array(Vg_list, dtype=np.float64)
    Id_arr = np.array(Id_list, dtype=np.float64)

    # The measured data is *unsigned* — VG, VD, Id are all reported as
    # positive numbers even for pMOS. Convert to the physical convention
    # (negative VG, VD for pMOS) so the SPICE harness operates correctly,
    # but keep Id positive in the paper's "magnitude" convention.
    if dev_type == "pmos":
        Vd_arr = -np.abs(Vd_arr)
        Vg_arr = -np.abs(Vg_arr)
        Id_arr = np.abs(Id_arr)
        if fixed is not None:
            fixed = -abs(fixed)

    # If kind wasn't inferable from the filename, detect from variance.
    if kind is None:
        kind = "idvg" if np.std(Vg_arr) >= np.std(Vd_arr) else "idvd"
    if fixed is None:
        fixed = float(np.median(Vd_arr) if kind == "idvg" else np.median(Vg_arr))

    return [
        Curve(
            kind=kind,
            fixed=float(fixed),
            Vg=Vg_arr,
            Vd=Vd_arr,
            Id=Id_arr,
            path=str(path),
            source="measured",
        )
    ]


def _is_non_numeric(s: str) -> bool:
    try:
        float(s)
        return False
    except (TypeError, ValueError):
        return True


def load_device_curves(
    dev: Device,
    repo_root: Path | None = None,
) -> list[Curve]:
    """Find and load every curve file that belongs to ``dev``.

    The paper repo lays data out as ``cryo_data/<geom_dir>/<sweep>.csv`` —
    the geometry lives in the *directory* name, not the file name. We try
    both.

    Missing measured data is an error because generated targets would make
    the extraction comparison invalid.
    """
    paths = discover_paper_paths(repo_root or PAPER_REPO_DIR)
    matched: list[Path] = []
    required_dir = _MEASUREMENT_DIR_OVERRIDES.get(
        (dev.dev_type, dev.L_um, dev.W_um)
    )
    for p in paths.data_files:
        # First try parent dir (paper repo convention), then filename.
        geom = parse_geometry_from_name(p.parent.name) or parse_geometry_from_name(p.name)
        if geom is None:
            continue
        dt, L, W = geom
        if (dt == dev.dev_type
                and abs(L - dev.L_um) < 1e-3
                and abs(W - dev.W_um) < 1e-3
                and (required_dir is None or p.parent.name == required_dir)):
            matched.append(p)

    curves: list[Curve] = []
    for p in matched:
        curves.extend(_load_csv_curves(p, dev.dev_type))

    if not curves:
        raise RuntimeError(
            f"no measured data for {dev.dev_type} L={dev.L_um:g} W={dev.W_um:g} "
            f"under {repo_root or PAPER_REPO_DIR}"
        )
    return curves


def all_device_curves(
    devices: Iterable[Device] = PAPER_DEVICES,
) -> dict[tuple[str, float, float], list[Curve]]:
    return {
        (d.dev_type, d.L_um, d.W_um): load_device_curves(d)
        for d in devices
    }
