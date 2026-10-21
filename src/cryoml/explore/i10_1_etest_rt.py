"""I10-1 feasibility: pair the NIST mds2-2997 ETest 4 K files with the
room-temperature files of the same module/pad IDs in
``google/skywater-pdk-sky130-raw-data``.

4 K side: ``i8_1_nist.discover_etest`` (``ETEST_ROOT/Sky130-4k``, names
``sky130_fd_pr__<type>_w<W>u_l<L>u_m<m>(<module>_<a>_<b>_<IDVG|IDVD>)_4K.mdm``).
Room-temperature side: the file of the same name without ``_4K`` under
``sky130_fd_pr/cells/<type>/`` of the raw-data repository at ``RT_COMMIT``,
local copy ``RT_ROOT`` (gitignored; fetched by ``fetch``).  Files are read as
IC-CAP MDM text with ``i5_1_ihp_real.parse_mdm``; nothing in the downloads is
executed.  This module fits nothing.
"""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

from ..config import ROOT
from . import i8_1_nist as base
from .i5_1_ihp_real import parse_mdm

RT_REPO = "google/skywater-pdk-sky130-raw-data"
RT_COMMIT = "1f7214f2f81a0629303947a07e1ec00a28ae8a33"     # main, 2023-02-08
RT_ROOT = ROOT / "out" / "research" / "data_raw" / "google_sky130_raw_data" / "files"
CORE_TYPES = ("nfet_01v8", "nfet_01v8_lvt", "pfet_01v8", "pfet_01v8_lvt", "pfet_01v8_hvt")
THICK_TYPES = ("nfet_g5v0d10v5", "pfet_g5v0d10v5")
SUFFIX_4K = "_4K.mdm"


def rt_name(name_4k: str) -> str:
    """Room-temperature file name of a 4 K ETest file name."""
    if not name_4k.endswith(SUFFIX_4K):
        raise ValueError(f"not a 4 K ETest file name: {name_4k}")
    return name_4k[: -len(SUFFIX_4K)] + ".mdm"


def rt_relpath(dev_type: str, name_4k: str) -> str:
    """Path of the room-temperature file inside the raw-data repository."""
    return f"sky130_fd_pr/cells/{dev_type}/{rt_name(name_4k)}"


def raw_url(relpath: str, commit: str = RT_COMMIT) -> str:
    return (f"https://raw.githubusercontent.com/{RT_REPO}/{commit}/"
            + urllib.parse.quote(relpath, safe="/()"))


def git_blob_sha1(data: bytes) -> str:
    """Git blob id of ``data`` (compare with the GitHub tree listing)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def rt_relpaths(devices: Iterable[base.NistDevice]) -> list[str]:
    out = []
    for dev in devices:
        for name in ("IDVG", "IDVD"):
            if name in dev.files:
                out.append(rt_relpath(dev.dev_type, Path(dev.files[name]).name))
    return out


def fetch(relpaths: Sequence[str], dest: Path = RT_ROOT, commit: str = RT_COMMIT,
          blob_sha: dict[str, str] | None = None, timeout_s: float = 60.0) -> dict:
    """Download ``relpaths`` from the raw-data repository at ``commit`` into
    ``dest`` (existing files are kept).  With ``blob_sha`` (relpath -> git blob
    id from the tree listing) every file is checked; a mismatch raises."""
    n_new = 0
    for rel in relpaths:
        out = Path(dest) / rel
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(raw_url(rel, commit), timeout=timeout_s) as r:
                out.write_bytes(r.read())
            n_new += 1
        if blob_sha is not None and git_blob_sha1(out.read_bytes()) != blob_sha[rel]:
            raise ValueError(f"{rel}: git blob id differs from the tree listing")
    return {"n": len(relpaths), "n_downloaded": n_new, "commit": commit}


def load_rt(dev: base.NistDevice, root: Path = RT_ROOT) -> base.NistDevice:
    """Room-temperature ``NistDevice`` with the tag, type and geometry of the
    4 K device ``dev``; curves keyed by ``i8_1_nist.assign_keys``."""
    rt = base.NistDevice(dev.tag, "google_raw_data_RT", dev.dev_type, dev.polarity,
                         dev.module, dev.position, dev.W_um, dev.L_um)
    for name, kind in (("IDVG", "idvg"), ("IDVD", "idvd")):
        if name not in dev.files:
            rt.flags.append(f"missing {name}")
            continue
        rel = rt_relpath(dev.dev_type, Path(dev.files[name]).name)
        path = Path(root) / rel
        if not path.exists():
            rt.flags.append(f"missing RT {name}")
            continue
        mdm = parse_mdm(path)
        rt.files[name] = rel
        rt.inputs[name] = base._inputs_row(mdm)
        rt.curves.extend(base.file_curves(mdm, kind))
    base.assign_keys(rt.curves)
    return rt


def pair_etest(types: Sequence[str] | None = None, root_4k: Path = base.ETEST_ROOT,
               root_rt: Path = RT_ROOT) -> list[tuple[base.NistDevice, base.NistDevice]]:
    """(4 K device, room-temperature device) pairs with both files present at
    both temperatures, optionally limited to ``types``."""
    out = []
    for dev in base.discover_etest(root_4k):
        if types is not None and dev.dev_type not in types:
            continue
        rt = load_rt(dev, root_rt)
        if len(dev.files) == 2 and len(rt.files) == 2:
            out.append((dev, rt))
    return out
