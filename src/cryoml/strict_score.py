"""Strict device status for the paper RRMS: never average over missing curves.

``metrics.device_rrms_new`` (and everything built on it: ``score_device_new``,
``fps.score_real``, ``fps.score_canonical``, ``card_eval.score_card``)
averages only the *finite* included curves.  When an included metric curve
fails to simulate (non-finite NGSpice output) the upstream ``rrmsCalc.py``
gives NaN for the device, whereas the port silently reports the mean of the
surviving curves.  A device whose included curve fails therefore has no
valid paper RRMS.

This module keeps RRMS itself unchanged.  It only classifies a device record
as ``valid`` (every frozen-inclusion curve finite) or ``failed`` and offers
two explicit aggregate conventions per run:

* ``valid``: all-device mean and combined (mean of the nMOS/pMOS family
  means) over the valid devices only, with the number of failed devices
  always reported next to it;
* ``penalized``: every device kept, each failed included curve scored at
  ``PENALTY`` (= ``ngspice_fit.PENALTY`` = 10, the optimizers' convention).
  This is labelled the *penalized convention*; it is not the paper metric.

Accepted record shapes (all read-only):

* ``per_curve`` in the full ``device_rrms_new`` form
  ``{tag: {"rrms", "included", "mean_abs_meas"}}`` (cold-loop extract records,
  DEV arms, swap search, simgap);
* ``per_curve`` in the flat ``card_eval.score_card`` form ``{tag: rrms}``
  holding the scored curves only (ngspice_direct, reference scores);
* FD-polish records without ``per_curve`` (``polish_cards`` summaries,
  posterior ``polished``): ``failed_curves`` when recorded, otherwise inferred
  from the penalized polish objective ``history[-1]`` against the stored
  drop-non-finite ``rrms`` of the same card (exact when both are present:
  ``n * history[-1] = (n - k) * rrms + PENALTY * k``).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from .config import OUT_DIR

# Mirrors ngspice_fit.PENALTY / temp_fit.PENALTY (checked by the unit test);
# not imported so result readers need neither scipy nor the simulator stack.
PENALTY = 10.0

BASELINE_JSON = OUT_DIR / "pdk_baseline" / "pdk_baseline.json"

VALID = "valid"
FAILED = "failed"
UNKNOWN = "unknown"

PENALIZED_LABEL = (f"penalized convention: each failed included curve at RRMS "
                   f"{PENALTY:g} (ngspice_fit.PENALTY); not the paper metric")
VALID_LABEL = "valid-only: mean over devices whose included curves all simulate"


@lru_cache(maxsize=4)
def _baseline(path: str) -> dict[str, frozenset[str]]:
    per_curve = json.loads(Path(path).read_text())["per_curve"]
    return {tag: frozenset(t for t, v in rows.items() if v.get("included"))
            for tag, rows in per_curve.items()}


def include_tags(tag: str, baseline: str | Path = BASELINE_JSON) -> frozenset[str]:
    """The frozen published-baseline inclusion set of a device."""
    return _baseline(str(baseline))[tag]


def _finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def family_of(tag: str) -> str:
    return "nmos" if tag.split("/")[-1].startswith("nmos") else "pmos"


def device_status(per_curve: Mapping | None, include: Iterable[str]) -> dict:
    """Strict status of one device from its ``per_curve`` and inclusion set.

    Returns ``status`` (valid/failed/unknown), ``n_included``, ``n_finite``,
    ``n_failed``, ``failed_tags`` (tag -> reason: ``nonfinite`` = scored as
    NaN/inf/None, ``not_scored`` = absent or not marked included), and
    ``strict_rrms`` (the paper mean when valid, NaN when failed) and
    ``penalized_rrms``.  An included tag whose target curve is identically
    zero (``mean_abs_meas == 0``, possible only for simulated targets) has no
    defined RRMS for any card and is left out of ``n_included``.
    """
    include = set(include)
    if per_curve is None:
        return {"status": UNKNOWN, "basis": "none", "n_included": len(include),
                "n_finite": None, "n_failed": None, "failed_tags": {},
                "strict_rrms": math.nan, "penalized_rrms": math.nan}
    values, failed, zero_target = [], {}, []
    for tag in sorted(include):
        entry = per_curve.get(tag)
        if isinstance(entry, Mapping):
            if (not entry.get("included")
                    and entry.get("mean_abs_meas") == 0.0
                    and entry.get("reason") is None):
                zero_target.append(tag)
                continue
            if not entry.get("included"):
                failed[tag] = "not_scored"
                continue
            value = entry.get("rrms")
        else:
            if tag not in per_curve:
                failed[tag] = "not_scored"
                continue
            value = entry
        if _finite(value):
            values.append(float(value))
        else:
            failed[tag] = "nonfinite"
    n = len(values) + len(failed)
    k = len(failed)
    strict = (sum(values) / len(values)) if values and not k else math.nan
    penalized = ((sum(values) + PENALTY * k) / n) if n else math.nan
    return {"status": FAILED if k else VALID, "basis": "per_curve",
            "n_included": n, "n_finite": len(values), "n_failed": k,
            "failed_tags": failed, "zero_target_tags": zero_target,
            "strict_rrms": strict, "penalized_rrms": penalized}


def polish_status(record: Mapping, n_included: int) -> dict:
    """Strict status of an FD-polish record without ``per_curve``.

    Uses ``failed_curves`` when the record carries it, otherwise infers the
    count from the penalized objective of the final card (``history[-1]``)
    and its stored drop-non-finite ``rrms``."""
    rrms = record.get("rrms")
    history = record.get("history") or []
    objective = history[-1] if history and _finite(history[-1]) else None
    if "failed_curves" in record:
        k = int(record["failed_curves"])
        n = int(record.get("n_included_curves", n_included))
        basis = "failed_curves_field"
    else:
        n = n_included
        if objective is None:
            return {"status": UNKNOWN, "basis": "none", "n_included": n,
                    "n_finite": None, "n_failed": None, "failed_tags": {},
                    "strict_rrms": math.nan, "penalized_rrms": math.nan}
        basis = "polish_history"
        if not _finite(rrms):
            # A NaN/inf device score (strict metric, or every curve failed):
            # failed, count unknown; history[-1] is the penalized mean itself.
            return {"status": FAILED, "basis": basis, "n_included": n,
                    "n_finite": None, "n_failed": None, "failed_tags": {},
                    "strict_rrms": math.nan, "penalized_rrms": float(objective)}
        k = int(round(n * (objective - rrms) / (PENALTY - rrms)))
        k = min(max(k, 0), n)
    finite_mean = float(rrms) if _finite(rrms) else math.nan
    if n and (k == n or _finite(finite_mean)):
        penalized = ((finite_mean * (n - k) if n > k else 0.0) + PENALTY * k) / n
    else:
        penalized = float(objective) if objective is not None else math.nan
    return {"status": FAILED if k else VALID, "basis": basis, "n_included": n,
            "n_finite": n - k, "n_failed": k, "failed_tags": {},
            "strict_rrms": finite_mean if not k else math.nan,
            "penalized_rrms": penalized}


def record_status(record: Mapping, tag: str,
                  include: Iterable[str] | None = None) -> dict:
    """Strict status of any per-device result record (see module doc)."""
    include = include_tags(tag) if include is None else include
    if isinstance(record.get("per_curve"), Mapping):
        return device_status(record["per_curve"], include)
    if "history" in record or "failed_curves" in record:
        return polish_status(record, len(set(include)))
    return device_status(None, include)


def _family_means(values: Mapping[str, float]) -> dict:
    nmos = [v for t, v in values.items() if family_of(t) == "nmos"]
    pmos = [v for t, v in values.items() if family_of(t) == "pmos"]
    out = {"all_device_mean_rrms": (sum(values.values()) / len(values)
                                    if values else math.nan),
           "nmos_rrms": sum(nmos) / len(nmos) if nmos else math.nan,
           "pmos_rrms": sum(pmos) / len(pmos) if pmos else math.nan}
    out["combined_rrms"] = ((out["nmos_rrms"] + out["pmos_rrms"]) / 2
                            if nmos and pmos else math.nan)
    return out


def aggregate(statuses: Mapping[str, Mapping],
              reported: Mapping[str, float] | None = None) -> dict:
    """Run-level strict aggregates from ``{device: status}``.

    ``valid`` means and combined use the valid devices only; ``penalized``
    keeps every device with failed curves at ``PENALTY``.  ``reported`` (the
    run's own per-device values, optional) is averaged as stored, for an
    old-vs-strict comparison.  Devices whose status is unknown are counted
    and left out of both conventions."""
    valid = {t: s["strict_rrms"] for t, s in statuses.items()
             if s["status"] == VALID}
    failed = sorted(t for t, s in statuses.items() if s["status"] == FAILED)
    unknown = sorted(t for t, s in statuses.items() if s["status"] == UNKNOWN)
    penal = {t: s["penalized_rrms"] for t, s in statuses.items()
             if s["status"] != UNKNOWN}
    out = {"n_devices": len(statuses), "n_valid_devices": len(valid),
           "n_failed_devices": len(failed), "failed_devices": failed,
           "n_failed_curves": {t: statuses[t]["n_failed"] for t in failed},
           "n_unknown_devices": len(unknown), "unknown_devices": unknown,
           "valid": {**_family_means(valid), "n_devices": len(valid),
                     "convention": VALID_LABEL},
           "penalized": {**_family_means(penal), "n_devices": len(penal),
                         "convention": PENALIZED_LABEL}}
    if reported is not None:
        vals = {t: float(v) for t, v in reported.items() if _finite(v)}
        out["reported"] = {**_family_means(vals), "n_devices": len(vals),
                           "convention": "as stored (non-finite curves dropped)"}
    return out
