#!/usr/bin/env python3
"""Polish extracted cards with finite differences against measured data.

The surrogate search proposes a card; this stage refines it using fresh
NGSpice simulations only.  Residuals are computed from the measured I-V curves
and the real simulator, never from the emulator, so the optimizer cannot lower
an objective by exploiting surrogate error.  Residuals are per curve *and* per
normalized sweep section, so a card cannot buy an aggregate gain by damaging a
single sweep region.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cryoml.data_io import load_device_curves  # noqa: E402
from cryoml.devices import PAPER_DEVICES  # noqa: E402
from cryoml.metrics import (clean_current, curve_trim,  # noqa: E402
                            score_device_new)
from cryoml.noise_aware import fit_curve_noise_model  # noqa: E402
from cryoml.regions import (FINE_SECTION_EDGES,  # noqa: E402
                            section_spans)
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import device_tag, get_logger  # noqa: E402

logger = get_logger("real_fd_polish")
EDGES = FINE_SECTION_EDGES
PENALTY = 1e3


def build_layout(device, curves, include_tags: set[str]) -> list[dict]:
    """Per-curve trimmed measurement, denominator and section boundaries."""
    layout = []
    for index, curve in enumerate(curves):
        tag = f"{curve.kind}@{abs(float(curve.fixed)):g}"
        if tag not in include_tags:
            continue
        start, cleaned, denominator = curve_trim(
            device.dev_type, device.L_um, device.W_um, curve.kind, curve.Id)
        if denominator <= 0:
            continue
        target = np.asarray(cleaned[start:], dtype=np.float64)
        count = len(target)
        spans = section_spans(count, EDGES)
        sweep = np.asarray(
            curve.Vd if curve.kind == "idvd" else curve.Vg,
            dtype=np.float64)[start:]
        model = fit_curve_noise_model(
            curve.kind, sweep, clean_current(cleaned)[start:])
        layout.append({"index": index, "tag": tag, "start": start,
                       "target": target, "denominator": float(denominator),
                       "spans": spans,
                       "censored": np.asarray(model.censored_mask, dtype=bool),
                       "detection": float(model.detection_limit)})
    return layout


def curve_deviation(simulated: np.ndarray, row: dict, *,
                    censor: bool) -> np.ndarray:
    """Pointwise deviation, optionally censored below the detection band.

    Where the measurement sits under the instrument detection limit it carries
    no information about which card is better, so reproducing its exact noise
    plateau is not a fitting target.  Those points are penalized one-sidedly:
    only for putting real current where the measurement says there is none.
    The reported paper RRMS is never censored.
    """
    target = row["target"]
    count = min(len(simulated), len(target))
    simulated, target = simulated[:count], target[:count]
    deviation = simulated - target
    if not censor:
        return deviation
    censored = row["censored"][:count]
    excess = np.maximum(np.abs(simulated) - row["detection"], 0.0)
    return np.where(censored, excess, deviation)


def section_values(simulated: np.ndarray, row: dict, *,
                   censor: bool) -> list[float]:
    deviation = curve_deviation(simulated, row, censor=censor)
    count = len(deviation)
    values = []
    for lo, hi in row["spans"]:
        hi = min(hi, count)
        if hi <= lo:
            values.append(PENALTY)
            continue
        rrms = float(np.sqrt(np.mean(deviation[lo:hi] ** 2))
                     / row["denominator"])
        values.append(rrms if np.isfinite(rrms) else PENALTY)
    return values


def section_residuals(simulations, layout: list[dict], *,
                      censor: bool = True) -> np.ndarray:
    """Residual vector whose sum of squares is the mean section RRMS."""
    values = []
    for row in layout:
        simulated = np.asarray(
            simulations[row["index"]], dtype=np.float64)[row["start"]:]
        values.extend(section_values(simulated, row, censor=censor))
    total = max(len(values), 1)
    return np.asarray(values, dtype=np.float64) / np.sqrt(total)


def polish_device(job) -> dict:
    (tag, dev_type, L_um, W_um, bin_index, names, start_params, lo, hi,
     active, include_tags, max_nfev, trust_radius, diff_step,
     accept_on, censor, overrides) = job
    device = next(row for row in PAPER_DEVICES
                  if device_tag(row.dev_type, row.L_um, row.W_um) == tag)
    curves = load_device_curves(device)
    layout = build_layout(device, curves, set(include_tags))
    lo = np.asarray(lo, dtype=np.float64).copy()
    hi = np.asarray(hi, dtype=np.float64).copy()
    start = np.asarray(start_params, dtype=np.float64)
    active = np.asarray(active, dtype=bool).copy()
    # Unfreeze coordinates whose published value is zero, which the
    # multiplicative box collapses to a single point.  PRWG is the clearest
    # case: every pMOS card carries 0.052 and fits the triode region well,
    # while every nMOS card is pinned at exactly zero.
    for name, (new_lo, new_hi) in (overrides or {}).items():
        if name not in names:
            continue
        index = names.index(name)
        lo[index] = min(lo[index], float(new_lo))
        hi[index] = max(hi[index], float(new_hi))
        if hi[index] > lo[index]:
            active[index] = True
    span = np.where(hi > lo, hi - lo, 1.0)
    # Optimize in normalized local coordinates.  The physical box spans many
    # orders of magnitude (toxe ~4e-9 against ndep ~1.7e17), so a single
    # relative step is meaningless and least_squares declares convergence on
    # the scaled variable vector almost immediately.  In local units every
    # coordinate is O(1) and one absolute step is valid for all of them.
    start_local = 2.0 * (start - lo) / span - 1.0

    def to_params(free_local: np.ndarray) -> dict:
        local = start_local.copy()
        local[active] = np.clip(free_local, -1.0, 1.0)
        values = lo + (local + 1.0) * 0.5 * span
        return {name: float(value) for name, value in zip(names, values)}

    calls = 0

    def residual(free: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        try:
            simulations = simulate_pdk(
                dev_type, L_um, W_um, curves, params=to_params(free),
                bin_index=bin_index)
        except Exception:  # a non-convergent probe is a bad point, not a crash
            return np.full((len(EDGES) - 1) * max(len(layout), 1), PENALTY)
        if not all(np.isfinite(np.asarray(values)).all()
                   for values in simulations):
            return np.full((len(EDGES) - 1) * max(len(layout), 1), PENALTY)
        return section_residuals(simulations, layout, censor=censor)

    def score(params: dict) -> dict:
        simulations = simulate_pdk(dev_type, L_um, W_um, curves,
                                   params=params, bin_index=bin_index)
        paper = score_device_new(dev_type, L_um, W_um, curves, simulations,
                                 include_tags=set(include_tags))
        return {"rrms": float(paper["rrms"]),
                "per_curve": paper["per_curve"],
                "sections": {
                    row["tag"]: section_values(
                        np.asarray(simulations[row["index"]],
                                   dtype=np.float64)[row["start"]:],
                        row, censor=False)
                    for row in layout},
                "signal_sections": {
                    row["tag"]: section_values(
                        np.asarray(simulations[row["index"]],
                                   dtype=np.float64)[row["start"]:],
                        row, censor=censor)
                    for row in layout},
                "simulations": simulations}

    t0 = time.time()
    before = score(to_params(start_local[active]))
    centre = start_local[active]
    bounds = (np.maximum(-1.0, centre - 2.0 * trust_radius),
              np.minimum(1.0, centre + 2.0 * trust_radius))
    error = None

    def jacobian(free_local: np.ndarray) -> np.ndarray:
        """Absolute two-point Jacobian in local units.

        SciPy's ``diff_step`` is relative to the current value, which collapses
        to nothing for a coordinate sitting near zero -- exactly where the
        newly unfrozen parameters live.
        """
        base = np.asarray(residual(free_local), dtype=np.float64)
        columns = np.empty((base.size, free_local.size), dtype=np.float64)
        for index in range(free_local.size):
            shifted = free_local.copy()
            step = diff_step if shifted[index] + diff_step <= 1.0 else -diff_step
            shifted[index] += step
            columns[:, index] = (
                np.asarray(residual(shifted), dtype=np.float64) - base) / step
        return columns

    try:
        solution = least_squares(
            residual, np.clip(centre, bounds[0], bounds[1]),
            bounds=bounds, method="trf", jac=jacobian,
            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=max_nfev)
        after_params = to_params(solution.x)
        after = score(after_params)
        success, message = bool(solution.success), str(solution.message)
    except Exception as exc:  # noqa: BLE001
        after_params, after = to_params(start_local[active]), before
        success, message, error = False, "", f"{type(exc).__name__}: {exc}"

    # Accept on the objective the search actually minimizes.  Judging a
    # section-targeted refinement by overall RRMS repeats the mistake that
    # made the surrogate search useless: optimize one quantity, then discard
    # the result unless a different quantity improved.  Both are recorded.
    def section_objective(scored: dict) -> float:
        flat = [value for values in scored["signal_sections"].values()
                for value in values]
        if not flat:
            return float("inf")
        return float(0.75 * np.mean(flat) + 0.25 * np.max(flat))

    before_objective = section_objective(before)
    after_objective = section_objective(after)
    if accept_on == "overall":
        improved = after["rrms"] < before["rrms"]
    else:
        improved = after_objective < before_objective
    chosen = after if improved else before
    chosen_params = (after_params if improved
                     else to_params(start_local[active]))
    return {
        "device": tag, "dev_type": dev_type,
        "before_rrms": before["rrms"], "after_rrms": after["rrms"],
        "before_section_objective": before_objective,
        "after_section_objective": after_objective,
        "accept_on": accept_on,
        "kept": "fd_endpoint" if improved else "start",
        "rrms": chosen["rrms"], "params": chosen_params,
        "sections_before": before["sections"],
        "sections_after": after["sections"],
        "signal_sections_before": before["signal_sections"],
        "signal_sections_after": after["signal_sections"],
        "sections": chosen["sections"],
        "per_curve": chosen["per_curve"],
        "n_active": int(active.sum()), "residual_calls": calls,
        "max_nfev": max_nfev, "trust_radius": trust_radius,
        "success": success, "message": message, "error": error,
        "runtime_s": round(time.time() - t0, 1),
        "simulations": [np.asarray(v).tolist() for v in chosen["simulations"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True,
                        help="extraction summary providing the start cards")
    parser.add_argument("--method", default="noise_aware_fd")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/pdk_synth_full_widened"),
                        help="dataset defining parameter names and bounds")
    parser.add_argument("--baseline", type=Path,
                        default=Path("out/pdk_baseline/pdk_baseline.json"))
    parser.add_argument("--devices", default=None)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--trust-radius", type=float, default=0.35,
                        help="fraction of each parameter's box span the "
                             "search may move from its start")
    parser.add_argument("--diff-step", type=float, default=0.02)
    parser.add_argument(
        "--unfreeze", default="",
        help="polarity:name:lo:hi specs unfreezing structurally pinned "
             "coordinates, e.g. nmos:prwg:-0.4:0.4")
    parser.add_argument(
        "--no-censor", action="store_true",
        help="fit sub-detection measurement points literally (control)")
    parser.add_argument(
        "--accept-on", choices=("section", "overall"), default="section",
        help="quantity deciding whether the refined card replaces its start; "
             "'section' matches what the residual minimizes")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("out/pdk_real_fd_polish"))
    args = parser.parse_args()

    overrides: dict[str, dict] = {}
    for item in (part.strip() for part in args.unfreeze.split(",")):
        if not item:
            continue
        polarity, name, low, high = item.split(":")
        overrides.setdefault(polarity, {})[name] = (float(low), float(high))

    summary = json.loads(args.summary.read_text())
    baseline = json.loads(args.baseline.read_text())
    include = {tag: [curve for curve, row in values.items()
                     if row.get("included")]
               for tag, values in baseline["per_curve"].items()}
    bins = {row["device"]: int(row["bin_index"])
            for row in baseline["devices"]}
    wanted = (set(args.devices.split(",")) if args.devices else None)

    jobs = []
    for record in summary["devices"]:
        tag = record["device"]
        if wanted and tag not in wanted:
            continue
        with np.load(args.data_dir / f"{tag}.npz", allow_pickle=True) as data:
            names = [str(value) for value in data["parameter_names"]]
            lo = np.asarray(data["lo"], dtype=np.float64)
            hi = np.asarray(data["hi"], dtype=np.float64)
            active = np.asarray(data["active"], dtype=bool)
        card = record["methods"][args.method]["params"]
        start = np.asarray([float(card[name]) for name in names],
                           dtype=np.float64)
        jobs.append((tag, record["dev_type"], record["L_um"], record["W_um"],
                     bins[tag], names, start, lo, hi, active, include[tag],
                     args.max_nfev, args.trust_radius, args.diff_step,
                     args.accept_on, not args.no_censor,
                     overrides.get(record["dev_type"], {})))

    if args.workers == 1:
        results = [polish_device(job) for job in jobs]
    else:
        with get_context("spawn").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(polish_device, jobs))
    results.sort(key=lambda row: row["device"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for row in results:
        simulations = row.pop("simulations")
        np.savez_compressed(
            args.out_dir / f"sims_{row['device']}.npz",
            **{f"fd_sim_{index}": np.asarray(values)
               for index, values in enumerate(simulations)})
        logger.info("%-22s %.5f -> %.5f (%s, %d calls, %.0fs)",
                    row["device"], row["before_rrms"], row["after_rrms"],
                    row["kept"], row["residual_calls"], row["runtime_s"])
    payload = {
        "source_summary": str(args.summary), "source_method": args.method,
        "data_dir": str(args.data_dir),
        "residual": ("per curve and per normalized sweep section, from fresh "
                     "NGSpice against measured curves; no surrogate"),
        "max_nfev": args.max_nfev, "trust_radius": args.trust_radius,
        "accept_on": args.accept_on, "censored": not args.no_censor,
        "unfreeze": args.unfreeze, "section_edges": list(EDGES),
        "all_device_mean_before": float(np.mean(
            [row["before_rrms"] for row in results])),
        "all_device_mean_after": float(np.mean(
            [row["rrms"] for row in results])),
        "n_improved": int(sum(row["kept"] == "fd_endpoint"
                              for row in results)),
        "devices": results,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "devices"},
                     indent=2))


if __name__ == "__main__":
    main()
