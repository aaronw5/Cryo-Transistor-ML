#!/usr/bin/env python3
"""Noise-aware finite-difference extraction on a frozen foundation model.

This is the staged post-baseline experiment. It does not retrain or overwrite
the controlled foundation checkpoints. Optimization uses only general curve
labels and a data-derived measurement noise model; all finalists are scored by
fresh NGSpice with the unchanged paper metric.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import foundation_parameter_study as fps  # noqa: E402
from cryoml.config import OUT_DIR, OUT_TABLES, ensure_dirs  # noqa: E402
from cryoml.devices import PAPER_DEVICES, parse_device_list  # noqa: E402
from cryoml.foundation import (FoundationSurrogate, InverseAutoencoder,  # noqa: E402
                               signed_log)
from cryoml.metrics import clean_current, curve_trim, family_totals  # noqa: E402
from cryoml.noise_aware import (balanced_regime_objective,  # noqa: E402
                                fit_curve_noise_model,
                                noise_aware_curve_loss,
                                reported_bias_rrms,
                                segmented_curve_loss,
                                segment_rrms_within_budget,
                                within_noise_aware_budget,
                                within_tradeoff_budget)
from cryoml.spice_pdk import simulate_pdk  # noqa: E402
from cryoml.utils import get_logger, set_seed  # noqa: E402

logger = get_logger("noise_aware_foundation_extract")


def load_models(parameter_set: str, device: torch.device,
                forward_path: Path | None = None,
                inverse_path: Path | None = None,
                data_dir: Path | None = None,
                cache_dir: Path | None = None):
    default_data, default_cache, baseline_out = fps.default_paths(
        parameter_set)
    # The parameter box, its bounds and the Latin-hypercube starts all come
    # from the dataset, so a surrogate trained on a widened box must be
    # searched against that same cache.  Silently falling back to the default
    # narrow cache would confine the search to the old bounds.
    data_dir = Path(data_dir) if data_dir else default_data
    cache_dir = Path(cache_dir) if cache_dir else default_cache
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    forward_path = forward_path or (baseline_out / "forward_model.pt")
    inverse_path = inverse_path or (baseline_out / "inverse_model.pt")
    forward_checkpoint = torch.load(
        forward_path, map_location=device,
        weights_only=False)
    inverse_checkpoint = torch.load(
        inverse_path, map_location=device,
        weights_only=False)
    expected = (metadata["n_parameters"], metadata["n_points"],
                metadata["parameter_names"])
    for path, checkpoint in ((forward_path, forward_checkpoint),
                             (inverse_path, inverse_checkpoint)):
        actual = (checkpoint["n_parameters"], checkpoint["n_points"],
                  checkpoint["parameter_names"])
        if actual != expected:
            raise RuntimeError(f"incompatible checkpoint {path}")
    forward = FoundationSurrogate(
        metadata["n_parameters"], metadata["n_points"],
        tuple(forward_checkpoint["hidden"])).to(device)
    forward.load_state_dict(forward_checkpoint["state"])
    forward.eval()
    inverse = InverseAutoencoder(
        inverse_checkpoint.get("n_iv_features", metadata["n_points"]),
        metadata["n_parameters"],
        tuple(inverse_checkpoint["hidden"])).to(device)
    inverse.load_state_dict(inverse_checkpoint["state"])
    inverse.eval()
    tensors = fps.training_tensors(cache_dir, device)
    model_sources = {
        "forward_checkpoint": str(forward_path.resolve()),
        "inverse_checkpoint": str(inverse_path.resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "cache_dir": str(Path(cache_dir).resolve()),
    }
    return (data_dir, cache_dir, baseline_out, metadata, forward, inverse,
            tensors, model_sources)


from cryoml.regions import (LOW_DRAIN_GATE_MIN_V,  # noqa: E402
                            LOW_DRAIN_MAX_V, is_low_drain_curve,
                            low_drain_mask)


def low_drain_layout(context: fps.DeviceContext) -> list[dict]:
    """Output-curve rows restricted to the low-drain triode region.

    This is the region where every nMOS card was worse than its published
    card: small |Vd| at strong gate drive.  Whole-curve and quartile scores
    dilute it, so it is carried as its own quantity.
    """
    rows = []
    for desc in context.canonical_curves:
        if desc["tag"] not in context.include_tags or desc["kind"] != "idvd":
            continue
        source = int(desc["source_curve_index"])
        curve = context.curves[source]
        if not is_low_drain_curve(curve.kind, curve.fixed,
                                  LOW_DRAIN_GATE_MIN_V):
            continue
        span = slice(desc["start"], desc["stop"])
        start, cleaned, denominator = curve_trim(
            context.device.dev_type, context.device.L_um, context.device.W_um,
            desc["kind"], context.measured_canonical[span])
        if denominator <= 0:
            continue
        sweep = np.abs(np.asarray(curve.Vd, dtype=np.float64))[start:]
        target = np.asarray(cleaned[start:], dtype=np.float64)
        count = min(len(sweep), len(target))
        mask = low_drain_mask(sweep[:count], LOW_DRAIN_MAX_V)
        if not mask.any():
            continue
        rows.append({"desc": desc, "trim": start, "target": target[:count],
                     "denominator": float(denominator), "mask": mask})
    return rows


def low_drain_rrms(currents: np.ndarray, layout: list[dict]) -> np.ndarray:
    """Mean low-drain RRMS for one canonical vector or a stack of them."""
    currents = np.asarray(currents, dtype=np.float64)
    single = currents.ndim == 1
    if single:
        currents = currents[None, :]
    columns = []
    for row in layout:
        desc = row["desc"]
        start = desc["start"] + row["trim"]
        block = currents[:, start:desc["stop"]]
        count = min(block.shape[1], len(row["target"]), len(row["mask"]))
        mask = row["mask"][:count]
        residual = block[:, :count][:, mask] - row["target"][:count][mask]
        columns.append(np.sqrt(np.mean(residual ** 2, axis=1))
                       / row["denominator"])
    if not columns:
        values = np.full(len(currents), np.nan)
    else:
        values = np.mean(np.stack(columns, axis=1), axis=1)
    return float(values[0]) if single else values


def low_drain_constrained_starts(context: fps.DeviceContext, n_starts: int,
                                 reference: float) -> list[tuple[str, np.ndarray]]:
    """Best stored cards by RRMS subject to the low-drain constraint.

    Reweighting the objective does not work here: absolute section weights are
    outvoted by numerically larger sections elsewhere, and a ratio-to-published
    objective is degenerate because the published card scores 1.0 everywhere by
    construction.  A constraint has neither failure mode.
    """
    layout = low_drain_layout(context)
    if not layout or not np.isfinite(reference):
        return []
    samples = np.asarray(context.current_samples, dtype=np.float64)
    low = low_drain_rrms(samples, layout)
    overall = np.mean(fps.curve_rrms_matrix(
        samples, fps.curve_objective_layout(context)), axis=1)
    feasible = np.flatnonzero(np.isfinite(low) & np.isfinite(overall)
                              & (low <= reference))
    if not feasible.size:
        return []
    ranked = feasible[np.argsort(overall[feasible])]
    starts, active = [], context.box.active
    for index in ranked:
        candidate = np.asarray(context.local_samples[index], dtype=np.float64)
        if (not starts or not np.any(active)
                or min(np.sqrt(np.mean(
                    (candidate[active] - prior[active]) ** 2))
                       for _, prior in starts) >= 0.35):
            starts.append((f"lhc_low_drain_{len(starts)}", candidate))
        if len(starts) >= n_starts:
            break
    return starts


def curve_models(context: fps.DeviceContext) -> dict[str, dict]:
    models = {}
    for desc in context.canonical_curves:
        if desc["tag"] not in context.include_tags:
            continue
        span = slice(desc["start"], desc["stop"])
        start, cleaned, denominator = curve_trim(
            context.device.dev_type, context.device.L_um,
            context.device.W_um, desc["kind"],
            context.measured_canonical[span])
        if denominator <= 0:
            continue
        source = int(desc["source_curve_index"])
        measured_curve = context.curves[source]
        sweep = (measured_curve.Vd if desc["kind"] == "idvd"
                 else measured_curve.Vg)
        sweep = np.asarray(sweep, dtype=np.float64)[start:]
        target = clean_current(cleaned)[start:]
        models[desc["tag"]] = {
            "desc": desc, "trim_start": start,
            "model": fit_curve_noise_model(desc["kind"], sweep, target),
        }
    return models


def noise_score_canonical(current: np.ndarray,
                          models: dict[str, dict]) -> dict:
    losses, components, segment_rrms = {}, {}, {}
    for tag, item in models.items():
        desc = item["desc"]
        start = desc["start"] + item["trim_start"]
        simulated = np.asarray(current[start:desc["stop"]], dtype=np.float64)
        result = segmented_curve_loss(simulated, item["model"])
        losses[tag] = result["total"]
        components[tag] = result
        segment_rrms[tag] = result["segments"]
    aggregate = balanced_regime_objective(losses)
    return {**aggregate, "curve_components": components,
            "segment_rrms": segment_rrms,
            "noise_models": {tag: item["model"].diagnostic()
                             for tag, item in models.items()}}


def regime_pair(score: dict) -> tuple[float, float]:
    ordinary = [value for name, value in score["groups"].items()
                if name.endswith("_ordinary")]
    high = [value for name, value in score["groups"].items()
            if name.endswith("_high")]
    if not ordinary or not high:
        raise ValueError("both ordinary and high regime groups are required")
    return float(np.mean(ordinary)), float(np.mean(high))


def canonical_from_real(context: fps.DeviceContext,
                        simulations: list[np.ndarray]) -> np.ndarray:
    blocks = []
    for desc in context.canonical_curves:
        source = int(desc["source_curve_index"])
        blocks.append(np.asarray(simulations[source], dtype=np.float64))
    return np.concatenate(blocks)


def inverse_local(inverse: InverseAutoencoder, context: fps.DeviceContext,
                  tensors: fps.TrainingTensors,
                  device: torch.device) -> np.ndarray:
    normalized = (
        signed_log(context.measured_canonical).astype(np.float32)
        - tensors.slog_mean.detach().cpu().numpy()
    ) / tensors.slog_std.detach().cpu().numpy()
    with torch.no_grad():
        normalized_tensor = torch.tensor(
            normalized[None, :], dtype=torch.float32, device=device)
        current_tensor = torch.tensor(
            context.measured_canonical[None, :], dtype=torch.float32,
            device=device)
        device_ids = torch.tensor(
            [context.device_index], dtype=torch.long, device=device)
        iv_features = fps.inverse_iv_features(
            inverse, normalized_tensor, current_tensor, device_ids, tensors)
        local = inverse(
            iv_features,
            torch.tensor(context.geometry[None, :], dtype=torch.float32,
                         device=device))[0].detach().cpu().numpy()
    local = np.asarray(local, dtype=np.float64)
    local[~context.box.active] = 0.0
    return local


def extra_physical_starts(context: fps.DeviceContext, parameter_set: str,
                          data_dir: Path) -> list[tuple[str, np.ndarray]]:
    starts = []
    # Always include the controlled foundation result.  The revised selector
    # therefore measures a bounded tradeoff from an existing real-NGSpice
    # result, not merely from the published card.
    _data_dir, _cache_dir, baseline_out = fps.default_paths(parameter_set)
    baseline_path = baseline_out / f"ml_{context.tag}.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        params = baseline["methods"]["surrogate_fd"]["params"]
        starts.append(("controlled_baseline_fd",
                       context.box.params_to_local(params)))
    if parameter_set == "seven":
        path = OUT_TABLES / "seven_general_loss_ablation.json"
        if path.exists():
            payload = json.loads(path.read_text())
            record = next(row for row in payload["devices"]
                          if row["device"] == context.tag)
            for policy in ("capped_regime", "attainable_regret",
                           "high_voltage"):
                params = record["policies"][policy]["params"]
                starts.append((f"lhc_{policy}",
                               context.box.params_to_local(params)))
    else:
        path = OUT_TABLES / "parameter_surface_diagnostic.json"
        if path.exists() and _diagnostic_matches_box(parameter_set, data_dir):
            payload = json.loads(path.read_text())
            record = next(row for row in payload["devices"]
                          if row["device"] == context.tag)
            full = record["parameter_sets"]["full"]
            for policy in ("best_paper_rrms_sample",
                           "best_high_voltage_sample"):
                starts.append((f"lhc_{policy}", np.asarray(
                    full[policy]["local"], dtype=np.float64)))
    return starts


def _diagnostic_matches_box(parameter_set: str, data_dir: Path) -> bool:
    """Reject stored *local* coordinates that came from a different box.

    A local coordinate is only meaningful together with the bounds that
    normalized it.  Reusing one from a narrower box silently relocates the
    start: local -0.876 means delta 0.0456 in the original box but 0.0150 in
    the widened one.  The surface diagnostic stores no physical parameters and
    no provenance, so it may only be trusted when the extraction is running on
    the very dataset it was computed from.
    """
    default_data, _cache, _baseline = fps.default_paths(parameter_set)
    return Path(data_dir).resolve() == Path(default_data).resolve()


def section_objective_starts(context: fps.DeviceContext,
                             n_starts: int) -> list[tuple[str, np.ndarray]]:
    """Rank stored physical samples by the sweep-section objective.

    ``diverse_oracle_starts`` ranks by mean whole-curve RRMS, so a card that is
    excellent inside individual sweep regions but only average overall is never
    proposed.  Those are exactly the cards the section guard is meant to
    select, so they must enter the candidate pool.  This uses only general
    curve/section structure and no device identity.
    """
    models = curve_models(context)
    samples = np.asarray(context.current_samples, dtype=np.float64)
    # Scoring every stored sample with the full noise objective costs minutes
    # per device.  Rank first with a vectorized section-RRMS proxy that uses
    # the same four normalized sweep regions and the same denominators, then
    # pay for the exact objective only on the shortlist.
    proxy = np.zeros(len(samples), dtype=np.float64)
    worst = np.zeros(len(samples), dtype=np.float64)
    n_curves = 0
    for item in models.values():
        desc = item["desc"]
        start = desc["start"] + item["trim_start"]
        block = samples[:, start:desc["stop"]]
        target = item["model"].target_magnitude
        count = min(block.shape[1], len(target))
        block, target = block[:, :count], target[:count]
        denominator = max(float(item["model"].denominator), 1e-30)
        edges = (0.0, 0.2, 0.4, 0.7, 1.0)
        section_scores = []
        for left, right in zip(edges[:-1], edges[1:]):
            lo = int(round(left * count))
            hi = min(count, max(lo + 1, int(round(right * count))))
            section_scores.append(np.sqrt(np.mean(
                (np.abs(block[:, lo:hi]) - target[lo:hi]) ** 2,
                axis=1)) / denominator)
        stacked = np.stack(section_scores, axis=1)
        proxy += stacked.mean(axis=1)
        worst = np.maximum(worst, stacked.max(axis=1))
        n_curves += 1
    proxy = 0.75 * proxy / max(n_curves, 1) + 0.25 * worst
    shortlist = np.argsort(proxy)[:200]
    exact = np.array([
        noise_score_canonical(samples[index], models)["objective"]
        for index in shortlist])
    ranked = shortlist[np.argsort(exact)]
    starts, active = [], context.box.active
    for index in ranked:
        candidate = np.asarray(context.local_samples[index], dtype=np.float64)
        if (not starts or not np.any(active)
                or min(np.sqrt(np.mean(
                    (candidate[active] - prior[active]) ** 2))
                       for _, prior in starts) >= 0.35):
            starts.append((f"lhc_section_{len(starts)}", candidate))
        if len(starts) >= n_starts:
            break
    return starts


def extract_device(context: fps.DeviceContext, parameter_set: str,
                   forward: FoundationSurrogate,
                   inverse: InverseAutoencoder,
                   tensors: fps.TrainingTensors,
                   device: torch.device, *, n_starts: int,
                   n_validate: int, maxiter: int,
                   diff_step: float, regularization: float,
                   guard_tolerance: float,
                   guard_absolute: float,
                   curve_guard_relative: float,
                   curve_guard_absolute: float,
                   segment_guard_relative: float,
                   segment_guard_absolute: float,
                   data_dir: Path = None,
                   trust_radius: float = 0.25,
                   low_drain_tolerance: float = 0.0,
                   guard_mode: str = "reported",
                   noise_guard_absolute: float = 0.0005,
                   noise_curve_guard_absolute: float = 0.002,
                   noise_segment_guard_absolute: float = 0.002,
                   ) -> tuple[dict, list]:
    t0 = time.time()
    models = curve_models(context)
    drain_layout = low_drain_layout(context)
    published_simulations = simulate_pdk(
        context.device.dev_type, context.device.L_um, context.device.W_um,
        context.curves, params=dict(context.box.published),
        bin_index=context.bin_index)
    published_low_drain = low_drain_rrms(
        canonical_from_real(context, published_simulations), drain_layout)
    ordinary_starts, oracle = fps.diverse_oracle_starts(
        context, max(1, n_starts))
    starts = [(f"ordinary_lhc_{index}", local)
              for index, local in enumerate(ordinary_starts)]
    starts.extend(extra_physical_starts(context, parameter_set, data_dir))
    starts.extend(section_objective_starts(context, max(1, n_starts)))
    starts.extend(low_drain_constrained_starts(
        context, max(1, n_starts), published_low_drain))
    starts.append(("inverse_prediction",
                   inverse_local(inverse, context, tensors, device)))
    starts.append(("published", context.box.local_published))
    unique, seen = [], set()
    for name, local in starts:
        local = np.asarray(local, dtype=np.float64)
        key = np.round(local, 12).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((name, local))
    starts = unique
    active = np.flatnonzero(context.box.active)
    candidates = []
    for start_name, local_start in starts:
        local_start = local_start.copy()
        for objective_mode in ("balanced", "high"):
            def objective(active_values):
                local = local_start.copy()
                local[active] = np.asarray(active_values, dtype=np.float64)
                predicted = fps.predict_surrogate(
                    forward, local, context, tensors, device)[0]
                score = noise_score_canonical(predicted, models)
                if objective_mode == "high":
                    _ordinary, value = regime_pair(score)
                else:
                    value = score["objective"]
                if regularization > 0 and len(active):
                    value += regularization * float(np.mean(
                        np.asarray(active_values, dtype=np.float64) ** 2))
                return value

            if len(active):
                # Trust region.  The surrogate is only reliable near cards it
                # was trained around; with the full box as bounds, L-BFGS-B
                # reliably lowers the emulator objective while raising true
                # NGSpice error, because it walks to where the emulator is
                # confidently wrong.  Constraining the step keeps the search
                # where the local derivatives were validated.
                centre = local_start[active]
                limits = [
                    (max(-1.0, float(value) - trust_radius),
                     min(1.0, float(value) + trust_radius))
                    for value in centre
                ]
                result = minimize(
                    objective, centre, method="L-BFGS-B",
                    jac="2-point", bounds=limits,
                    options={"maxiter": maxiter, "maxls": 20,
                             "finite_diff_rel_step": diff_step})
                local = local_start.copy()
                local[active] = result.x
                nfev = int(result.nfev)
                success, message = bool(result.success), str(result.message)
            else:
                local, nfev = local_start, 1
                success, message = True, "no active parameters"
            predicted = fps.predict_surrogate(
                forward, local, context, tensors, device)[0]
            candidates.append({
                "start": start_name, "kind": "fd_endpoint",
                "objective_mode": objective_mode, "local": local,
                "surrogate_noise": noise_score_canonical(predicted, models),
                "surrogate_paper": fps.score_canonical(context, predicted),
                "nfev": nfev, "success": success, "message": message,
            })
        # Preserve each physical start as a validation candidate; local
        # surrogate gradients can be wrong outside dense training regions.
        start_prediction = fps.predict_surrogate(
            forward, local_start, context, tensors, device)[0]
        candidates.append({
            "start": start_name, "kind": "physical_start",
            "objective_mode": "physical_start",
            "local": local_start,
            "surrogate_noise": noise_score_canonical(
                start_prediction, models),
            "surrogate_paper": fps.score_canonical(context, start_prediction),
            "nfev": 0, "success": True, "message": "unoptimized start",
        })
    candidates.sort(key=lambda row: row["surrogate_noise"]["objective"])
    physical = [row for row in candidates if row["kind"] == "physical_start"]
    finalists = list(physical)
    for objective_mode in ("balanced", "high"):
        endpoints = [row for row in candidates
                     if row["kind"] == "fd_endpoint"
                     and row["objective_mode"] == objective_mode]
        if objective_mode == "high":
            endpoints.sort(key=lambda row: regime_pair(
                row["surrogate_noise"])[1])
        else:
            endpoints.sort(key=lambda row: row["surrogate_noise"]["objective"])
        finalists.extend(endpoints[:max(1, n_validate)])
    validations, seen = [], set()
    for row in finalists:
        key = np.round(row["local"], 12).tobytes()
        if key in seen:
            continue
        seen.add(key)
        params = context.box.local_to_params(row["local"])
        simulations = simulate_pdk(
            context.device.dev_type, context.device.L_um,
            context.device.W_um, context.curves, params=params,
            bin_index=context.bin_index)
        canonical = canonical_from_real(context, simulations)
        validations.append({
            **row, "params": params, "simulations": simulations,
            "real_noise": noise_score_canonical(canonical, models),
            "real_paper": fps.score_real(context, simulations),
            "low_drain_rrms": low_drain_rrms(canonical, drain_layout),
        })
    reference_pool = [row for row in validations
                      if np.isfinite(row.get("low_drain_rrms", np.nan))
                      and np.isfinite(published_low_drain)
                      and row["low_drain_rrms"] <= published_low_drain]
    reference = min(reference_pool or validations,
                    key=lambda row: row["real_paper"]["rrms"])
    published = next(row for row in validations
                     if row["start"] == "published"
                     and row["kind"] == "physical_start")
    reference_paper = reference["real_paper"]
    reference_ordinary = reported_bias_rrms(
        reference_paper["per_curve"], high=False)
    def reported_feasible(row: dict) -> bool:
        """Veto on the raw paper scorer, including sub-detection points."""
        return (within_tradeoff_budget(
            row["real_paper"], reference_paper,
            relative_budget=guard_tolerance,
            absolute_budget=guard_absolute,
            curve_relative_budget=curve_guard_relative,
            curve_absolute_budget=curve_guard_absolute)
            and segment_rrms_within_budget(
                row["real_noise"]["segment_rrms"],
                [reference["real_noise"]["segment_rrms"],
                 published["real_noise"]["segment_rrms"]],
                relative_budget=segment_guard_relative,
                absolute_budget=segment_guard_absolute,
                field="reported_rrms"))

    def noise_aware_feasible(row: dict) -> bool:
        """Veto on measurement-aware losses at their own numerical scale.

        The raw section budget is expressed in reported-RRMS units, where a
        typical section is ~0.04.  A measurement-aware section loss is ~0.0008,
        so reusing the same absolute budget would never bind.  Both guards are
        evaluated and both counts are recorded; only the selected mode decides.
        """
        return (within_noise_aware_budget(
            row["real_noise"], reference["real_noise"],
            relative_budget=guard_tolerance,
            absolute_budget=noise_guard_absolute,
            curve_relative_budget=curve_guard_relative,
            curve_absolute_budget=noise_curve_guard_absolute)
            and segment_rrms_within_budget(
                row["real_noise"]["segment_rrms"],
                [reference["real_noise"]["segment_rrms"],
                 published["real_noise"]["segment_rrms"]],
                relative_budget=segment_guard_relative,
                absolute_budget=noise_segment_guard_absolute,
                field="total"))

    # The reference is the incumbent: the best real-NGSpice overall-RRMS
    # candidate, and the card that is used when nothing else qualifies.  It
    # must therefore be feasible by construction.  Anchoring sections on
    # min(reference, published) let the incumbent fail its own guard whenever
    # the published card was better in any single section, which emptied the
    # feasible set and silently reduced the selector to "keep the reference".
    def with_reference(rows: list[dict]) -> list[dict]:
        if any(row is reference for row in rows):
            return rows
        return [reference, *rows]

    # Hard constraint: never worse than the published card in the low-drain
    # triode region.  Applied as a filter before the existing rules, so the
    # selector optimizes freely everywhere else.
    def meets_low_drain(row: dict) -> bool:
        value = row.get("low_drain_rrms")
        if value is None or not np.isfinite(value):
            return True
        if not np.isfinite(published_low_drain):
            return True
        return bool(value <= published_low_drain * (1.0 + low_drain_tolerance))

    constrained = [row for row in validations if meets_low_drain(row)]
    pool = constrained if constrained else validations
    reported_set = with_reference(
        [row for row in pool if reported_feasible(row)])
    noise_set = with_reference(
        [row for row in pool if noise_aware_feasible(row)])
    for candidate_set in (reported_set, noise_set):
        candidate_set.sort(key=lambda row: row["real_noise"]["objective"])

    def build_record(feasible: list[dict], mode: str) -> tuple[dict, list]:
        """Select and describe one card from the shared validated pool.

        Both guard modes see identical starts, finite-difference endpoints and
        fresh NGSpice validations, so the only difference between the two
        emitted cards is which veto was applied.
        """
        best = feasible[0] if feasible else reference
        simulations = best["simulations"]
        paper = best["real_paper"]
        record = {
            "method": "noise_aware_two_point_fd_on_frozen_surrogate",
            "selected_start": best["start"],
            "selected_candidate_kind": best["kind"],
            "selected_objective_mode": best["objective_mode"],
            "params": best["params"], "local": best["local"].tolist(),
            "rrms": float(paper["rrms"]), "sigma": float(paper["sigma"]),
            "rmse_uA": float(paper["rmse_uA"]),
            "high_voltage_rrms": float(paper["high_voltage_rrms"]),
            "official_dynamic_rrms": float(paper["official_dynamic_rrms"]),
            "per_curve": paper["per_curve"],
            "high_voltage_per_curve": paper["high_voltage_per_curve"],
            "low_drain_rrms": (float(best["low_drain_rrms"])
                           if np.isfinite(best.get("low_drain_rrms", np.nan))
                           else None),
        "noise_aware_objective": float(best["real_noise"]["objective"]),
            "segment_rrms": best["real_noise"]["segment_rrms"],
            "surrogate_noise_aware_objective": float(
                best["surrogate_noise"]["objective"]),
            "surrogate_rrms": float(best["surrogate_paper"]["rrms"]),
            "surrogate_high_voltage_rrms": float(
                best["surrogate_paper"]["high_voltage_rrms"]),
            "noise_models": best["real_noise"]["noise_models"],
            "oracle_lhc": oracle, "n_starts": len(starts),
            "n_validated": len(validations), "maxiter": maxiter,
            "diff_step": diff_step, "regularization": regularization,
            "trust_radius": trust_radius,
            "tradeoff_guard": {
                "definition": (
                    "noise-aware selection may spend a bounded amount of "
                    "unchanged reported RRMS relative to the best validated "
                    "real-NGSpice candidate; bounds cover overall, ordinary, "
                    "high, and every included curve"),
                "reference_start": reference["start"],
                "reference_candidate_kind": reference["kind"],
                "relative_budget": guard_tolerance,
                "absolute_budget": guard_absolute,
                "curve_relative_budget": curve_guard_relative,
                "curve_absolute_budget": curve_guard_absolute,
                "segment_relative_budget": segment_guard_relative,
                "segment_absolute_budget": segment_guard_absolute,
                "guard_mode": mode,
                "noise_absolute_budget": noise_guard_absolute,
                "noise_curve_absolute_budget": noise_curve_guard_absolute,
                "noise_segment_absolute_budget": noise_segment_guard_absolute,
                "n_feasible_reported_guard": len(reported_set),
                "n_feasible_noise_aware_guard": len(noise_set),
                "reference_is_always_feasible": True,
                "published_low_drain_rrms": (
                    float(published_low_drain)
                    if np.isfinite(published_low_drain) else None),
                "low_drain_tolerance": low_drain_tolerance,
                "n_meeting_low_drain": len(constrained),
                "low_drain_constraint_met": bool(constrained),
                "segment_anchors": ["best overall real candidate", "published"],
                "reference_rrms": float(reference_paper["rrms"]),
                "reference_ordinary_rrms": reference_ordinary,
                "reference_high_voltage_rrms": float(
                    reference_paper["high_voltage_rrms"]),
                "n_feasible_candidates": len(feasible),
            },
            "validated_candidates": [{
                "start": row["start"], "kind": row["kind"],
                "objective_mode": row["objective_mode"],
                "real_noise_objective": float(row["real_noise"]["objective"]),
                "real_noise_ordinary": regime_pair(row["real_noise"])[0],
                "real_noise_high": regime_pair(row["real_noise"])[1],
                "real_rrms": float(row["real_paper"]["rrms"]),
            "low_drain_rrms": (float(row["low_drain_rrms"])
                               if np.isfinite(row.get("low_drain_rrms", np.nan))
                               else None),
                "real_high_voltage_rrms": float(
                    row["real_paper"]["high_voltage_rrms"]),
                "surrogate_noise_objective": float(
                    row["surrogate_noise"]["objective"]),
                "surrogate_rrms": float(row["surrogate_paper"]["rrms"]),
                "nfev": row["nfev"], "success": row["success"],
            } for row in validations],
            "runtime_s": round(time.time() - t0, 1),
        }
        return record, simulations

    primary_mode = guard_mode
    alternate_mode = ("reported" if primary_mode == "noise_aware"
                      else "noise_aware")
    sets = {"reported": reported_set, "noise_aware": noise_set}
    record, simulations = build_record(sets[primary_mode], primary_mode)
    alternate, alternate_sims = build_record(
        sets[alternate_mode], alternate_mode)
    return record, simulations, alternate, alternate_sims


def aggregate(records: list[dict], method: str) -> dict:
    values = {row["device"]: row["methods"][method] for row in records}
    result = family_totals(values)
    result["all_device_mean_rrms"] = float(np.mean(
        [row["rrms"] for row in values.values()]))
    result["all_device_mean_high_voltage_rrms"] = float(np.mean(
        [row["high_voltage_rrms"] for row in values.values()]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-set", choices=("seven", "full"),
                        required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--forward-checkpoint", type=Path, default=None,
                        help="frozen surrogate checkpoint; defaults to the controlled baseline")
    parser.add_argument("--inverse-checkpoint", type=Path, default=None,
                        help="inverse checkpoint used only for starts/control")
    parser.add_argument("--devices", default=None)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="dataset directory defining the parameter box "
                             "and the Latin-hypercube starts")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="training cache matching --data-dir")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-starts", type=int, default=3)
    parser.add_argument("--n-validate", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--diff-step", type=float, default=0.02)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument(
        "--trust-radius", type=float, default=0.25,
        help="maximum local-coordinate movement of a finite-difference "
             "search from its start; 2.0 restores unconstrained search")
    parser.add_argument("--guard-tolerance", type=float, default=0.05)
    parser.add_argument("--guard-absolute", type=float, default=0.005)
    parser.add_argument("--curve-guard-relative", type=float, default=0.25)
    parser.add_argument("--curve-guard-absolute", type=float, default=0.05)
    parser.add_argument("--segment-guard-relative", type=float, default=0.25)
    parser.add_argument("--segment-guard-absolute", type=float, default=0.02)
    parser.add_argument(
        "--low-drain-tolerance", type=float, default=0.0,
        help="fractional slack on the published-card low-drain constraint")
    parser.add_argument(
        "--guard-mode", choices=("reported", "noise_aware"),
        default="reported",
        help="which feasibility veto selects the card; both are always "
             "evaluated and both counts are recorded")
    parser.add_argument("--noise-guard-absolute", type=float, default=0.0005)
    parser.add_argument("--noise-curve-guard-absolute", type=float,
                        default=0.002)
    parser.add_argument("--noise-segment-guard-absolute", type=float,
                        default=0.002)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    set_seed(args.seed)
    torch_device = torch.device(args.device)
    (data_dir, _cache_dir, _baseline_out, metadata, forward, inverse,
     tensors, model_sources) = load_models(
         args.parameter_set, torch_device,
         forward_path=args.forward_checkpoint,
         inverse_path=args.inverse_checkpoint,
         data_dir=args.data_dir, cache_dir=args.cache_dir)
    out_dir = args.out_dir or (
        OUT_DIR / f"pdk_foundation_noiseaware_{args.parameter_set}")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for selected_device in parse_device_list(args.devices):
        context = fps.build_device_context(
            selected_device, data_dir, metadata, args.parameter_set)
        record_path = out_dir / f"ml_{context.tag}.json"
        sims_path = out_dir / f"sims_{context.tag}.npz"
        if args.resume and record_path.exists() and sims_path.exists():
            records.append(json.loads(record_path.read_text()))
            continue
        fd, fd_sims, alt, alt_sims = extract_device(
            context, args.parameter_set, forward, inverse, tensors,
            torch_device, n_starts=args.n_starts,
            n_validate=args.n_validate, maxiter=args.maxiter,
            diff_step=args.diff_step, regularization=args.regularization,
            guard_tolerance=args.guard_tolerance,
            guard_absolute=args.guard_absolute,
            curve_guard_relative=args.curve_guard_relative,
            curve_guard_absolute=args.curve_guard_absolute,
            segment_guard_relative=args.segment_guard_relative,
            segment_guard_absolute=args.segment_guard_absolute,
            data_dir=data_dir,
            trust_radius=args.trust_radius,
            low_drain_tolerance=args.low_drain_tolerance,
            guard_mode=args.guard_mode,
            noise_guard_absolute=args.noise_guard_absolute,
            noise_curve_guard_absolute=args.noise_curve_guard_absolute,
            noise_segment_guard_absolute=args.noise_segment_guard_absolute)
        direct, direct_sims = fps.inverse_extract(
            context, inverse, forward, tensors, torch_device)
        record = {
            "device": context.tag, "dev_type": selected_device.dev_type,
            "L_um": selected_device.L_um, "W_um": selected_device.W_um,
            "model_sources": model_sources,
            "methods": {"noise_aware_fd": fd,
                        "noise_aware_fd_alternate_guard": alt,
                        "direct_inverse_control": direct},
        }
        np.savez_compressed(
            sims_path,
            **{f"fd_sim_{index}": np.asarray(sim)
               for index, sim in enumerate(fd_sims)},
            **{f"alt_sim_{index}": np.asarray(sim)
               for index, sim in enumerate(alt_sims)},
            **{f"inverse_sim_{index}": np.asarray(sim)
               for index, sim in enumerate(direct_sims)})
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
        logger.info(
            "%-22s %s-guard %.4f high %.4f; %s-guard %.4f high %.4f; "
            "direct %.4f high %.4f",
            context.tag,
            fd["tradeoff_guard"]["guard_mode"], fd["rrms"],
            fd["high_voltage_rrms"],
            alt["tradeoff_guard"]["guard_mode"], alt["rrms"],
            alt["high_voltage_rrms"],
            direct["rrms"], direct["high_voltage_rrms"])
    if len(records) == len(PAPER_DEVICES):
        summary = {
            "parameter_set": args.parameter_set,
            "model_sources": model_sources,
            "loss": "data-derived censoring plus bounded current/slope and balanced regime tail",
            "device_specific_training_groups": False,
            "methods": {
                method: aggregate(records, method)
                for method in ("noise_aware_fd",
                               "noise_aware_fd_alternate_guard",
                               "direct_inverse_control")
            },
            "devices": records,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["methods"], indent=2))
    else:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
