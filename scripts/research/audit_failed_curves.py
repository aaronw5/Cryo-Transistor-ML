#!/usr/bin/env python3
"""Audit every reported result record for included metric curves that failed.

``metrics.device_rrms_new`` averages only the finite included curves, so a
card whose NGSpice sweep fails on an included curve is reported with the mean
of the surviving curves (upstream ``rrmsCalc.py`` would give NaN).  This
script reads result files only (no NGSpice, no training) and classifies every
device record with ``cryoml.strict_score`` against the frozen
published-baseline inclusion set (``out/pdk_baseline/pdk_baseline.json``):

* records with ``per_curve`` (cold-loop extract, swap search, simgap, DEV
  arms, ngspice_direct, reference scores) are counted directly; the stored
  sims npz (``fd_sim_*`` / ``inverse_sim_*``) are cross-checked where present;
* FD-polish records (``polished/*``, posterior ``polished``) are classified
  from the penalized polish objective ``history[-1]`` vs the stored ``rrms``;
  their start card from ``history[0]`` vs ``before_rrms`` (posterior: vs
  ``best_rrms``);
* the temperature study already scores failed curves at PENALTY (temp_fit),
  so it is only screened for per-temperature values at or above the penalty
  floor ``PENALTY / n_curves``.

Writes ``out/research/audit/failed_curves.json`` and ``failed_curves.md``.

    PYTHONPATH=src .venv/bin/python scripts/research/audit_failed_curves.py
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.strict_score import (FAILED, PENALIZED_LABEL, PENALTY,  # noqa: E402
                                 UNKNOWN, VALID_LABEL, aggregate,
                                 include_tags, polish_status, record_status)

OUT = ROOT / "out"
RESEARCH = OUT / "research"
AUDIT = RESEARCH / "audit"
TAG_RE = re.compile(r"(nmos|pmos)_L[\dp]+_W[\dp]+$")
NPZ_PREFIX = {"surrogate_fd": "fd_sim", "inverse_autoencoder": "inverse_sim"}


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def clean(o):
    """JSON-safe copy (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return clean(o.item())
    return o


RUNS: list[dict] = []


def add_run(name: str, group: str, statuses: dict, reported: dict,
            sources: list[str], headline: bool, note: str = "",
            extra: dict | None = None) -> None:
    agg = aggregate(statuses, reported)
    RUNS.append({"run": name, "group": group, "headline_candidate": headline,
                 "sources": sources, "note": note,
                 "devices": {t: {k: s.get(k) for k in (
                     "status", "basis", "n_included", "n_failed",
                     "failed_tags", "strict_rrms", "penalized_rrms")}
                     | {"reported_rrms": reported.get(t)}
                     for t, s in sorted(statuses.items())},
                 **agg, **(extra or {})})


# ------------------------------------------------------------- npz crosscheck

_CURVE_INDEX: dict[str, dict[str, int]] = {}


def curve_index(tag: str) -> dict[str, int]:
    """metric tag -> curve index in load_device_curves order (geometry only;
    the same for simulated targets)."""
    if tag not in _CURVE_INDEX:
        from cryoml.card_eval import device_by_tag
        from cryoml.data_io import load_device_curves
        from cryoml.metrics import tag_metric_curves
        curves = load_device_curves(device_by_tag(tag))
        index = {id(c): i for i, c in enumerate(curves)}
        _CURVE_INDEX[tag] = {f"{k}@{b:g}": index[id(c)] for (k, b), c
                             in tag_metric_curves(curves).items()}
    return _CURVE_INDEX[tag]


def npz_failed(path: Path, prefix: str, tag: str) -> list[str] | None:
    """Included tags whose stored simulation has any non-finite point."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as z:
        index = curve_index(tag)
        bad = []
        for t in sorted(include_tags(tag)):
            key = f"{prefix}_{index[t]}"
            if key not in z.files:
                bad.append(t)
            elif not np.all(np.isfinite(np.asarray(z[key], dtype=np.float64))):
                bad.append(t)
        return bad


NPZ_CHECKS: list[dict] = []


# --------------------------------------------------------------- record dirs

def ml_dir(directory: Path, group: str, label: str, headline: bool,
           note: str = "", check_npz: bool = False) -> None:
    files = sorted(directory.glob("ml_*.json"))
    if not files:
        return
    by_method: dict[str, tuple[dict, dict]] = {}
    for f in files:
        row = json.loads(f.read_text())
        tag = row["device"]
        for method, rec in row.get("methods", {}).items():
            st, rp = by_method.setdefault(method, ({}, {}))
            st[tag] = record_status(rec, tag)
            rp[tag] = rec.get("rrms")
            if check_npz and method in NPZ_PREFIX:
                bad = npz_failed(directory / f"sims_{tag}.npz",
                                 NPZ_PREFIX[method], tag)
                if bad is not None:
                    from_record = sorted(st[tag]["failed_tags"])
                    NPZ_CHECKS.append({
                        "run": f"{label} {method}", "device": tag,
                        "npz_nonfinite_included": bad,
                        "record_failed_tags": from_record,
                        "agree": bad == from_record})
    for method, (st, rp) in by_method.items():
        add_run(f"{label} {method}", group, st, rp, [rel(directory)],
                headline, note)


def direct_summary(path: Path, group: str, label: str, headline: bool,
                   note: str = "") -> None:
    d = json.loads(path.read_text())
    st = {t: record_status(r, t) for t, r in d.get("devices", {}).items()}
    rp = {t: r.get("rrms") for t, r in d.get("devices", {}).items()}
    add_run(label, group, st, rp, [rel(path)], headline, note)


def polished_summary(path: Path, label: str, headline: bool) -> None:
    d = json.loads(path.read_text())
    after, start, rp_after, rp_start = {}, {}, {}, {}
    for t, r in d.get("devices", {}).items():
        n = len(include_tags(t))
        after[t] = record_status(r, t)
        rp_after[t] = r.get("rrms")
        # Summaries written before polish_cards recorded before_failed_curves
        # store the *penalized* start objective as before_rrms (== history[0]),
        # so the start card's status is only taken from the explicit field;
        # otherwise it is audited through the source method's own records.
        if "before_failed_curves" in r:
            start[t] = polish_status({"rrms": r.get("before_rrms"),
                                      "failed_curves": r["before_failed_curves"],
                                      "n_included_curves": r.get(
                                          "n_included_curves", n)}, n)
            rp_start[t] = r.get("before_rrms")
    src = [rel(path)] + ([d["cards"]] if d.get("cards") else [])
    note = (f"cards {d.get('cards')} method {d.get('method')}; "
            f"target_override {d.get('target_override')}")
    add_run(f"{label} [after FD]", "polished", after, rp_after, src, headline,
            note)
    if start:
        add_run(f"{label} [start card]", "polished", start, rp_start, src,
                False, note + "; start card = the method's card")


def posterior_eval(path: Path, label: str) -> None:
    d = json.loads(path.read_text())
    pol, best, rp_pol, rp_best = {}, {}, {}, {}
    for t, r in d.get("devices", {}).items():
        n = len(include_tags(t))
        p = r.get("polished")
        if p:
            pol[t] = record_status(p, t)
            rp_pol[t] = p.get("rrms")
            hist = p.get("history") or []
            if hist:
                best[t] = polish_status({"rrms": r.get("best_rrms"),
                                         "history": hist[:1]}, n)
                rp_best[t] = r.get("best_rrms")
    if pol:
        add_run(f"{label} [polished]", "posterior", pol, rp_pol, [rel(path)],
                not label.startswith("posterior smoke"))
    if best:
        add_run(f"{label} [best-of-k card]", "posterior", best, rp_best,
                [rel(path)], not label.startswith("posterior smoke"),
                "best-of-k card status inferred from the polish start objective")


def temp_screen() -> list[dict]:
    rows = []
    for f in sorted((RESEARCH / "temp").glob("*/set_*.json")):
        d = json.loads(f.read_text())
        suspects = []
        for t, r in d.get("devices", {}).items():
            n = len(include_tags(t))
            floor = PENALTY / n
            for stage in ("raw", "fd"):
                for temp, v in (r.get(stage, {}).get("per_temp") or {}).items():
                    if v is not None and (not math.isfinite(v) or v >= floor - 1e-12):
                        suspects.append({"device": t, "stage": stage,
                                         "temp_K": temp, "value": v,
                                         "penalty_floor": floor})
        rows.append({"file": rel(f), "n_suspect": len(suspects),
                     "suspects": suspects})
    return rows


# Headline rows of docs/RESEARCH_LOOP.md and docs/site/content.json -> run.
HEADLINES = [
    ("pipeline 43, round 1, raw (swap / misrank / rounds claims)",
     "cold_loop full r1 surrogate_fd"),
    ("pipeline 43, round 1, + FD", "polished pipe43_r1_fd/real [after FD]"),
    ("pipeline 43, round 4, raw (rounds / cryo-43 claims)",
     "cold_loop full r4 surrogate_fd"),
    ("pipeline 43, round 4, + FD", "polished pipe43_r4_fd/real [after FD]"),
    ("A tandem 43, round 2", "cold_loop full r2 inverse_autoencoder"),
    ("A tandem 43, round 3 (logged as nan)", "cold_loop full r3 inverse_autoencoder"),
    ("A tandem 43, round 4", "cold_loop full r4 inverse_autoencoder"),
    ("A tandem 7, round 1", "cold_loop seven r1 inverse_autoencoder"),
    ("pipeline 43 r4 + FD, sim set 0", "polished pipe43_r4_fd/sim0 [after FD]"),
    ("pipeline 43 r4 + FD, sim set 2", "polished pipe43_r4_fd/sim2 [after FD]"),
    ("DEV pipeline r1, set_0", "dev pipeline_r1/set_0 surrogate_fd"),
    ("DEV pipeline r1, set_1", "dev pipeline_r1/set_1 surrogate_fd"),
    ("DEV pipeline r1 + FD, set_0", "polished dev_pipe_r1_fd/set_0 [after FD]"),
    ("DEV pipeline r1 + FD, set_1", "polished dev_pipe_r1_fd/set_1 [after FD]"),
    ("pipeline 7, round 4, raw", "cold_loop seven r4 surrogate_fd"),
    ("pipeline 7, round 4, + FD", "polished pipe_r4_fd/real [after FD]"),
    ("NGSpice swap 43, round 1", "swap full_round_1 surrogate_fd"),
    ("NGSpice swap 43, round 1, + FD", "polished swap_full_r1_fd/real [after FD]"),
    ("direct NGSpice 43, round 3", "direct full_r3_direct"),
    ("direct NGSpice 43, round 3, + FD", "polished direct43_r3_fd/real [after FD]"),
]


def headline_rows() -> list[dict]:
    by_name = {r["run"]: r for r in RUNS}
    rows = []
    for label, name in HEADLINES:
        r = by_name.get(name)
        if r is None:
            continue
        rows.append({"headline": label, "run": name,
                     "n_failed_devices": r["n_failed_devices"],
                     "failed_devices": r["failed_devices"],
                     **{f"{conv}_{key}": r[conv][f"{key}_rrms"]
                        for conv in ("reported", "valid", "penalized")
                        for key in ("all_device_mean", "combined")}})
    return rows


# ---------------------------------------------------------------------- main

def main() -> int:
    for pset in ("seven", "full"):
        for k in (1, 2, 3, 4):
            d = OUT / "cold_loop" / pset / f"round_{k}" / "extract"
            ml_dir(d, "cold_loop", f"cold_loop {pset} r{k}", True,
                   check_npz=True)
    for d in sorted((RESEARCH / "swap_search").glob("*")):
        ml_dir(d, "swap_search", f"swap {d.name}", True)
    for p in sorted((RESEARCH / "ngspice_direct").rglob("summary.json")):
        name = rel(p.parent).replace("out/research/ngspice_direct/", "")
        headline = name != "smoke"
        direct_summary(p, "ngspice_direct", f"direct {name}", headline,
                       "" if headline else "smoke run")
    ref = RESEARCH / "reference_scores.json"
    if ref.is_file():
        d = json.loads(ref.read_text())
        for key, block in d.items():
            if isinstance(block, dict) and "devices" in block:
                st = {t: record_status(r, t) for t, r in block["devices"].items()}
                rp = {t: r.get("rrms") for t, r in block["devices"].items()}
                add_run(f"reference {key}", "reference", st, rp, [rel(ref)],
                        True)
    for p in sorted((RESEARCH / "polished").glob("*/*/summary.json")):
        name = rel(p.parent).replace("out/research/polished/", "")
        polished_summary(p, f"polished {name}", True)
    for p in sorted((RESEARCH / "posterior").glob("*/eval_*.json")):
        posterior_eval(p, f"posterior {p.parent.name}/{p.stem}")
    for d in sorted((RESEARCH / "simgap").glob("*/set_*")):
        ml_dir(d, "simgap", f"simgap {d.parent.name}/{d.name}", True,
               check_npz=True)
    dev = RESEARCH / "dev"
    for d in sorted(dev.glob("pipeline_r1/set_*")):
        ml_dir(d, "dev", f"dev pipeline_r1/{d.name}", True, check_npz=True)
    for d in sorted(dev.glob("*/B*/set_*")):
        ml_dir(d, "dev", f"dev {d.parent.parent.name}/{d.parent.name}/{d.name}",
               True)
    for d in sorted(list(dev.glob("smoke/*")) + list(dev.glob("verify_n1c/*"))):
        ml_dir(d, "dev_smoke", f"dev {d.parent.name}/{d.name}", False,
               "smoke / verification run, not a headline")
    temp = temp_screen()

    affected = [r for r in RUNS if r["n_failed_devices"]]
    npz_disagree = [c for c in NPZ_CHECKS if not c["agree"]]
    report = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "include_source": "out/pdk_baseline/pdk_baseline.json (frozen inclusion)",
        "penalty": PENALTY, "conventions": {"valid": VALID_LABEL,
                                            "penalized": PENALIZED_LABEL},
        "n_runs": len(RUNS), "n_runs_with_failed_devices": len(affected),
        "runs": RUNS, "npz_crosscheck": {"n_checked": len(NPZ_CHECKS),
                                         "n_disagree": len(npz_disagree),
                                         "disagreements": npz_disagree},
        "temperature_screen": temp,
        "headlines": headline_rows(),
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "failed_curves.json").write_text(
        json.dumps(clean(report), indent=1) + "\n")
    (AUDIT / "failed_curves.md").write_text(markdown(report))
    print(f"runs audited {len(RUNS)}; with failed devices {len(affected)}; "
          f"npz checks {len(NPZ_CHECKS)} ({len(npz_disagree)} disagree)")
    for r in affected:
        print(f"  {r['run']}: " + ", ".join(
            f"{t} ({r['n_failed_curves'][t]}/{r['devices'][t]['n_included']})"
            for t in r["failed_devices"]))
    return 0


def f4(v) -> str:
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) \
        else f"{v:.4f}"


def markdown(report: dict) -> str:
    runs = report["runs"]
    affected = [r for r in runs if r["n_failed_devices"]]
    lines = [
        "# Failed included metric curves: audit", "",
        f"Generated {report['generated']} by "
        "`scripts/research/audit_failed_curves.py` from result files only.",
        "",
        "A device is **failed** when any curve of its frozen published-baseline "
        "inclusion set is non-finite (or not scored) in its record. "
        "`metrics.device_rrms_new` silently averages the surviving curves; "
        "upstream `rrmsCalc.py` would give NaN. RRMS itself is unchanged here.",
        "",
        f"- valid: {VALID_LABEL}.",
        f"- penalized: {PENALIZED_LABEL}.",
        "",
        f"Runs audited: {len(runs)}; runs with failed devices: {len(affected)}. "
        f"Stored-sims cross-check: {report['npz_crosscheck']['n_checked']} device "
        f"records, {report['npz_crosscheck']['n_disagree']} disagreements.", "",
        "## Runs with failed devices", "",
        "| run | failed devices (failed / included curves) | reported all-device "
        "| valid-only all-device (n) | penalized all-device | basis |",
        "|---|---|---:|---:|---:|---|"]
    for r in affected:
        devs = "; ".join(f"{t} ({r['n_failed_curves'][t]}/"
                         f"{r['devices'][t]['n_included']})"
                         for t in r["failed_devices"])
        bases = sorted({r["devices"][t]["basis"] for t in r["failed_devices"]})
        lines.append(
            f"| {r['run']}{'' if r['headline_candidate'] else ' *(non-headline)*'}"
            f" | {devs} | {f4(r['reported']['all_device_mean_rrms'])} | "
            f"{f4(r['valid']['all_device_mean_rrms'])} ({r['n_valid_devices']}) | "
            f"{f4(r['penalized']['all_device_mean_rrms'])} | {', '.join(bases)} |")
    lines += ["", "## Headline numbers: stored vs strict", "",
              "all-device (combined). valid-only averages the valid devices; "
              "penalized keeps every device with failed curves at 10.", "",
              "| headline | failed dev. | stored | valid-only | penalized |",
              "|---|---:|---:|---:|---:|"]
    for h in report["headlines"]:
        lines.append(
            f"| {h['headline']} | {h['n_failed_devices']} | "
            + " | ".join(f"{f4(h[c + '_all_device_mean'])} ({f4(h[c + '_combined'])})"
                         for c in ("reported", "valid", "penalized")) + " |")
    unknown = [r for r in runs if r["n_unknown_devices"]]
    if unknown:
        lines += ["", "## Runs with devices that could not be classified", ""]
        lines += [f"- {r['run']}: {', '.join(r['unknown_devices'])}"
                  for r in unknown]
    if report["npz_crosscheck"]["disagreements"]:
        lines += ["", "## Stored sims vs record disagreements", ""]
        lines += [f"- {c['run']} {c['device']}: npz non-finite "
                  f"{c['npz_nonfinite_included']}, record failed "
                  f"{c['record_failed_tags']}"
                  for c in report["npz_crosscheck"]["disagreements"]]
    lines += ["", "## Temperature study (already penalized, screened only)", "",
              "temp_fit scores a failed curve at PENALTY, so these values are "
              "never silently dropped. Device-temperature values at or above "
              "the penalty floor PENALTY / n_curves:", ""]
    for t in report["temperature_screen"]:
        lines.append(f"- `{t['file']}`: {t['n_suspect']}"
                     + ("" if not t["suspects"] else " (" + "; ".join(
                         f"{s['device']} {s['stage']} {s['temp_K']} K "
                         f"{s['value']:.3g}" for s in t["suspects"][:6]) + ")"))
    clean_runs = [r["run"] for r in runs if not r["n_failed_devices"]
                  and not r["n_unknown_devices"]]
    lines += ["", f"## Runs with every device valid ({len(clean_runs)})", "",
              ", ".join(clean_runs), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
