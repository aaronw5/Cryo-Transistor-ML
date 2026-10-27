"""Cold-start parameter boxes that do not use the published 77 K multipliers.

The released 77 K corner cards write each tuned BSIM4 value as
``{base_expression * multiplier}``.  The base expression is the stock
SkyWater value for that bin, so dividing the evaluated published value by the
multiplier recovers the multiplier-free card.  A cold start centers the search
on that card (every multiplier equal to one) and searches log-uniformly in
multiplier space, so the earlier extraction never sets the center, the bounds,
or the fallback candidate.

Round specs use the same JSON layout as the sensitivity summary consumed by
``scripts/pdk_gen_full_data.py`` (``selected_parameters``,
``published_by_device``, ``bounds_by_device``), plus ``log_scale_by_device``.
``published_by_device`` holds the round's box center, not the released card.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .config import OUT_DIR
from .devices import PAPER_DEVICES
from .parameter_sets import PARAMS7
from . import spice_pdk as _sp
from .spice_pdk import PDK77K_DIR, _CORNER_BASENAME, _bin_spans, ensure_pdk77k
from .utils import device_tag

SENSITIVITY_SUMMARY = OUT_DIR / "pdk_parameter_sensitivity" / "summary.json"
BASELINE_JSON = OUT_DIR / "pdk_baseline" / "pdk_baseline.json"

_MULTIPLIER_RE = re.compile(
    r"^\+\s*(\w+)\s*=\s*\{(.*)\*\s*([0-9.eE+\-]+)\s*\}\s*$", re.MULTILINE)
# Named multipliers (I25-1 fix): some released bins write ``{expr*vth0_nom}``
# with a file-level ``.param vth0_nom = 2.08113`` (pmos_L0p35_W1p6, bin 0).
# Both brace and single-quote expression forms occur (``+ub='...*ub_nom'``).
_NAMED_MULTIPLIER_RE = re.compile(
    r"^\+\s*(\w+)\s*=\s*[{'](.*)\*\s*([A-Za-z_]\w*)\s*[}']\s*$", re.MULTILINE)
_PARAM_RE = re.compile(r"^\.param\s+(\w+)\s*=\s*([0-9.eE+\-]+)\s*$",
                       re.MULTILINE | re.IGNORECASE)

# Round-one half-widths in natural-log multiplier units.  Threshold voltage
# is searched within a factor of three; the other paper parameters within a
# factor of 100 (the released card uses factors from 1.75e-4 to 49); the
# secondary sensitivity-screened parameters within a factor of three.
COLD_HALF_WIDTH = {"vth0": math.log(3.0)}
CORE_HALF_WIDTH = math.log(100.0)
SECONDARY_HALF_WIDTH = math.log(3.0)
MIN_HALF_WIDTH = math.log(1.05)


def bin_multipliers(dev_type: str, bin_index: int) -> dict[str, float]:
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    text = (PDK77K_DIR / _CORNER_BASENAME[dev_type]).read_text(errors="replace")
    return multipliers_from_text(text, dev_type, bin_index)


def multipliers_from_text(text: str, dev_type: str, bin_index: int) -> dict[str, float]:
    """Multiplier of every ``{base * multiplier}`` line of one bin; the
    multiplier is a number or a name defined by a file-level ``.param``."""
    named = {k.lower(): float(v) for k, v in _PARAM_RE.findall(text)}
    for index, start, stop in _bin_spans(text, dev_type):
        if index == bin_index:
            span = text[start:stop]
            out = {name.lower(): float(value) for name, _expr, value
                   in _MULTIPLIER_RE.findall(span)}
            for name, _expr, ref in _NAMED_MULTIPLIER_RE.findall(span):
                if ref.lower() not in named:
                    raise ValueError(f"{name}: multiplier {ref} has no .param value")
                out[name.lower()] = named[ref.lower()]
            return out
    raise ValueError(f"bin {bin_index} not found for {dev_type}")


STOCK_BINS_JSON = _sp.STOCK_DIR / "device_bins.json"


def device_bins() -> dict[str, int]:
    """Bin of every paper device.  Paper template: the released card's bin
    (pdk_baseline.json).  Stock template: the bin NGSpice selects for the
    native L/W (``spice_pdk.find_bin_index``), cached in
    ``data/processed/pdk_stock/device_bins.json``."""
    if _sp.PDK_TEMPLATE == "stock":
        return stock_device_bins()
    baseline = json.loads(BASELINE_JSON.read_text())
    return {row["device"]: int(row["bin_index"]) for row in baseline["devices"]}


def stock_device_bins() -> dict[str, int]:
    if STOCK_BINS_JSON.exists():
        return {k: int(v) for k, v in json.loads(STOCK_BINS_JSON.read_text()).items()}
    bins = {}
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        b = _sp.find_bin_index(device.dev_type, device.L_um, device.W_um)
        if b is None:
            raise RuntimeError(f"no stock bin for {tag}")
        bins[tag] = int(b)
    STOCK_BINS_JSON.parent.mkdir(parents=True, exist_ok=True)
    STOCK_BINS_JSON.write_text(json.dumps(bins, indent=1) + "\n")
    return bins


def cold_centers(names: tuple[str, ...]) -> dict[str, dict]:
    """Multiplier-free value of every parameter for every paper device."""
    summary = json.loads(SENSITIVITY_SUMMARY.read_text())
    bins = device_bins()
    out = {}
    for device in PAPER_DEVICES:
        tag = device_tag(device.dev_type, device.L_um, device.W_um)
        published = summary["published_by_device"][tag]
        multipliers = bin_multipliers(device.dev_type, bins[tag])
        values, used = {}, {}
        for name in names:
            multiplier = multipliers.get(name, 1.0)
            values[name] = float(published[name]) / multiplier
            used[name] = multiplier
        out[tag] = {"center": values, "released_multipliers": used,
                    "bin_index": bins[tag]}
    return out


def half_width(name: str) -> float:
    if name in COLD_HALF_WIDTH:
        return COLD_HALF_WIDTH[name]
    return CORE_HALF_WIDTH if name in PARAMS7 else SECONDARY_HALF_WIDTH


def spec_from_centers(names: tuple[str, ...], centers: dict[str, dict],
                      widths: dict[str, dict[str, float]],
                      extra: dict | None = None) -> dict:
    """Symmetric log boxes; zero-valued parameters stay fixed (inactive)."""
    published, bounds, log_scale = {}, {}, {}
    for tag, row in centers.items():
        published[tag], bounds[tag], log_scale[tag] = {}, {}, {}
        for name in names:
            value = float(row[name])
            width = float(widths[tag][name])
            if value == 0.0 or not math.isfinite(value):
                published[tag][name] = 0.0 if not math.isfinite(value) else value
                bounds[tag][name] = [value, value]
                log_scale[tag][name] = False
                continue
            lo = value * math.exp(-width)
            hi = value * math.exp(width)
            published[tag][name] = value
            bounds[tag][name] = [min(lo, hi), max(lo, hi)]
            log_scale[tag][name] = True
    return {
        "schema_version": 1,
        "kind": "cold_start_round_spec",
        "selected_parameters": list(names),
        "published_by_device": published,
        "bounds_by_device": bounds,
        "log_scale_by_device": log_scale,
        "half_width_by_device": widths,
        **(extra or {}),
    }


def round_one_spec(names: tuple[str, ...]) -> dict:
    centers = cold_centers(names)
    widths = {tag: {name: half_width(name) for name in names}
              for tag in centers}
    spec = spec_from_centers(
        names, {tag: row["center"] for tag, row in centers.items()}, widths,
        extra={"round": 1,
               "center_rule": "released 77 K card divided by its multiplier"})
    spec["released_multipliers_by_device"] = {
        tag: row["released_multipliers"] for tag, row in centers.items()}
    return spec


def next_round_spec(previous: dict, selected: dict[str, dict[str, float]],
                    selected_local: dict[str, dict[str, float]], *,
                    shrink: float = 0.5, edge: float = 0.9,
                    round_index: int) -> dict:
    """Re-center on each device's selected card.

    Interior coordinates shrink their log half-width; coordinates at the box
    edge keep it, so the next box extends past the old bound in that
    direction.  The rule is identical for every device and parameter.
    """
    names = tuple(previous["selected_parameters"])
    widths = {}
    for tag, old in previous["half_width_by_device"].items():
        widths[tag] = {}
        for name in names:
            u = abs(float(selected_local[tag].get(name, 0.0)))
            width = float(old[name])
            widths[tag][name] = (width if u >= edge
                                 else max(MIN_HALF_WIDTH, shrink * width))
    return spec_from_centers(
        names, selected, widths,
        extra={"round": round_index,
               "center_rule": "previous round's selected NGSpice-validated card",
               "shrink": shrink, "edge": edge})


def write_spec(spec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1) + "\n")
