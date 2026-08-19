"""General noise-aware objectives for measured transistor I-V curves.

These routines intentionally know nothing about the fixed evaluation-device
list. They use only a measured sweep, its physical sweep kind, and simulated
current. Reported paper RRMS remains unchanged; this module defines an
optimization loss that cannot gain by reproducing instrument floors or spikes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class CurveNoiseModel:
    kind: str
    sweep: np.ndarray
    target_magnitude: np.ndarray
    detection_limit: float
    noise_sigma: float
    denominator: float
    signal_mask: np.ndarray
    censored_mask: np.ndarray

    def diagnostic(self) -> dict:
        return {
            "kind": self.kind,
            "detection_limit_A": float(self.detection_limit),
            "noise_sigma_A": float(self.noise_sigma),
            "denominator_A": float(self.denominator),
            "signal_fraction": float(np.mean(self.signal_mask)),
            "censored_fraction": float(np.mean(self.censored_mask)),
        }


def _odd_window(count: int, requested: int = 11) -> int:
    window = min(requested, count if count % 2 else count - 1)
    return max(window, 1)


def _robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def fit_curve_noise_model(kind: str, sweep: np.ndarray,
                          measured: np.ndarray,
                          *, sigma_multiplier: float = 4.0,
                          relative_detection_floor: float = 1e-5,
                          minimum_signal_points: int = 5) -> CurveNoiseModel:
    """Estimate a detection band and monotone physical target from one sweep.

    Transfer curves use their initial quarter to estimate an off-state floor.
    Output curves use only their first two points because a high-gate output
    curve acquires real current immediately. Local non-monotone variation
    supplies a robust noise estimate. Points below the resulting band are
    censored rather than fitted to their exact measured value.
    """
    if kind not in {"idvd", "idvg"}:
        raise ValueError(f"unknown curve kind {kind}")
    x = np.abs(np.asarray(sweep, dtype=np.float64))
    raw = np.abs(np.asarray(measured, dtype=np.float64))
    count = min(len(x), len(raw))
    x, raw = x[:count], raw[:count]
    if count < 2:
        raise ValueError("a curve needs at least two points")
    window = _odd_window(count)
    if window >= 5:
        smooth = savgol_filter(raw, window_length=window,
                               polyorder=min(2, window - 1), mode="interp")
    else:
        smooth = raw.copy()
    smooth = np.maximum.accumulate(np.maximum(smooth, 0.0))
    noise_sigma = _robust_sigma(raw - smooth)
    peak = max(float(np.quantile(smooth, 0.99)), 1e-30)
    initial_count = (max(5, count // 4) if kind == "idvg"
                     else min(2, count))
    baseline = float(np.median(raw[:initial_count]))
    detection = max(
        relative_detection_floor * peak,
        baseline + sigma_multiplier * noise_sigma,
        1e-30,
    )
    signal = smooth > detection
    if int(np.sum(signal)) < minimum_signal_points:
        # A curve with no reliable signal remains a censored upper-bound
        # observation. Do not invent a small denominator that gives it
        # unlimited leverage.
        signal = np.zeros(count, dtype=bool)
    censored = ~signal
    denominator = (float(np.mean(smooth[signal])) if np.any(signal)
                   else max(detection, peak))
    denominator = max(denominator, detection, 1e-30)
    return CurveNoiseModel(
        kind=kind, sweep=x, target_magnitude=smooth,
        detection_limit=float(detection), noise_sigma=float(noise_sigma),
        denominator=float(denominator), signal_mask=signal,
        censored_mask=censored,
    )


def pseudo_huber(values: np.ndarray, delta: float = 0.1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if delta <= 0:
        raise ValueError("delta must be positive")
    return delta * delta * (np.sqrt(1.0 + (values / delta) ** 2) - 1.0)


def noise_aware_curve_loss(simulated: np.ndarray, model: CurveNoiseModel,
                           *, huber_delta: float = 0.1,
                           censored_weight: float = 0.25,
                           shape_weight: float = 0.05) -> dict:
    """Return bounded-current and smoothed-slope losses for one curve."""
    simulation = np.abs(np.asarray(simulated, dtype=np.float64))
    count = min(len(simulation), len(model.target_magnitude))
    simulation = simulation[:count]
    target = model.target_magnitude[:count]
    signal = model.signal_mask[:count]
    censored = model.censored_mask[:count]
    denominator = model.denominator

    if np.any(signal):
        error = (simulation[signal] - target[signal]) / denominator
        current_loss = float(np.mean(pseudo_huber(error, huber_delta)))
    else:
        current_loss = 0.0
    if np.any(censored):
        excess = np.maximum(
            simulation[censored] - model.detection_limit, 0.0) / denominator
        censor_loss = float(np.mean(pseudo_huber(excess, huber_delta)))
    else:
        censor_loss = 0.0

    slope_loss = 0.0
    if np.sum(signal) >= 3 and shape_weight > 0:
        x = model.sweep[:count]
        span = max(float(x[-1] - x[0]), 1e-12)
        normalized_x = (x - x[0]) / span
        sim_slope = np.gradient(simulation / denominator, normalized_x)
        target_slope = np.gradient(target / denominator, normalized_x)
        slope_error = sim_slope[signal] - target_slope[signal]
        slope_loss = float(np.mean(pseudo_huber(slope_error, huber_delta)))
    total = (current_loss + censored_weight * censor_loss
             + shape_weight * slope_loss)
    return {
        "total": float(total), "current": current_loss,
        "censored": censor_loss, "shape": slope_loss,
    }


def segmented_curve_loss(simulated: np.ndarray, model: CurveNoiseModel,
                         *, segment_edges: tuple[float, ...] =
                         (0.0, 0.2, 0.4, 0.7, 1.000001),
                         tail_weight: float = 0.25,
                         noise_weight: float = 0.25) -> dict:
    """Balance residuals within a sweep so one endpoint cannot hide a knee.

    Segments use normalized sweep position rather than absolute voltage, so
    the rule applies to arbitrary sweep ranges.  For a 0--1.85 V sweep the
    first two segments end at 0.37 and 0.74 V, matching the linear/knee region
    exposed by the L1/W3 diagnostic without encoding that device identity.
    """
    if len(segment_edges) < 2 or segment_edges[0] != 0.0:
        raise ValueError("segment edges must begin at zero")
    if any(right <= left for left, right in
           zip(segment_edges[:-1], segment_edges[1:])):
        raise ValueError("segment edges must be strictly increasing")
    simulation = np.abs(np.asarray(simulated, dtype=np.float64))
    count = min(len(simulation), len(model.target_magnitude))
    simulation = simulation[:count]
    span = max(float(model.sweep[count - 1] - model.sweep[0]), 1e-12)
    position = (model.sweep[:count] - model.sweep[0]) / span
    segments = {}
    for index, (left, right) in enumerate(
            zip(segment_edges[:-1], segment_edges[1:])):
        mask = ((position >= left) & (position < right))
        if int(np.sum(mask)) < 2:
            continue
        submodel = CurveNoiseModel(
            kind=model.kind, sweep=model.sweep[:count][mask],
            target_magnitude=model.target_magnitude[:count][mask],
            detection_limit=model.detection_limit,
            noise_sigma=model.noise_sigma, denominator=model.denominator,
            signal_mask=model.signal_mask[:count][mask],
            censored_mask=model.censored_mask[:count][mask])
        result = noise_aware_curve_loss(simulation[mask], submodel)
        target = submodel.target_magnitude
        reported_rrms = float(np.sqrt(np.mean(
            (simulation[mask] - target) ** 2)) / submodel.denominator)
        segments[f"s{index}_{left:g}_{min(right, 1.0):g}"] = {
            **result, "reported_rrms": reported_rrms,
            "normalized_sweep_start": float(left),
            "normalized_sweep_stop": float(min(right, 1.0)),
            "n_points": int(np.sum(mask)),
        }
    if not segments:
        raise ValueError("curve contains no valid sweep segments")
    noise_totals = np.asarray([row["total"] for row in segments.values()])
    rrms_totals = np.asarray(
        [row["reported_rrms"] for row in segments.values()])
    noise_mean = float(np.mean(noise_totals))
    noise_tail = float(np.max(noise_totals))
    rrms_mean = float(np.mean(rrms_totals))
    rrms_tail = float(np.max(rrms_totals))
    noise_objective = (
        (1.0 - tail_weight) * noise_mean + tail_weight * noise_tail)
    rrms_objective = (
        (1.0 - tail_weight) * rrms_mean + tail_weight * rrms_tail)
    # RRMS by x-axis section is the primary optimization signal.  The
    # noise-aware term remains a smaller stabilizer for censored/noisy data.
    total = rrms_objective + float(noise_weight) * noise_objective
    return {
        "total": float(total),
        "segment_rrms_mean": rrms_mean,
        "segment_rrms_tail": rrms_tail,
        "segment_rrms_objective": rrms_objective,
        "segment_noise_mean": noise_mean,
        "segment_noise_tail": noise_tail,
        "segment_noise_objective": noise_objective,
        "segments": segments,
    }


def segment_rrms_within_budget(candidate: dict, anchors: list[dict], *,
                               relative_budget: float = 0.25,
                               absolute_budget: float = 0.02,
                               field: str = "reported_rrms") -> bool:
    """Guard every sweep segment against the best supplied incumbent.

    ``field`` selects which per-segment quantity is guarded.  ``reported_rrms``
    is the raw paper-scorer value over every point of the segment, including
    points below the instrument detection band.  ``total`` is the
    measurement-aware loss, which does not penalize a prediction for failing to
    reproduce a censored noise plateau.  The two live on different numerical
    scales, so their budgets are not interchangeable.
    """
    if not anchors:
        raise ValueError("at least one segment anchor is required")
    for curve, candidate_segments in candidate.items():
        for segment, row in candidate_segments.items():
            baselines = [
                anchor[curve][segment][field] for anchor in anchors
                if curve in anchor and segment in anchor[curve]
            ]
            if not baselines:
                continue
            baseline = float(min(baselines))
            margin = max(absolute_budget,
                         relative_budget * max(abs(baseline), 1e-30))
            if float(row[field]) > baseline + margin:
                return False
    return True


def noise_regime_pair(score: dict) -> tuple[float, float]:
    """Mean ordinary and mean high-bias group loss from a noise-aware score."""
    ordinary = [value for name, value in score["groups"].items()
                if name.endswith("_ordinary")]
    high = [value for name, value in score["groups"].items()
            if name.endswith("_high")]
    if not ordinary or not high:
        raise ValueError("both ordinary and high regime groups are required")
    return float(np.mean(ordinary)), float(np.mean(high))


def within_noise_aware_budget(candidate: dict, reference: dict, *,
                              relative_budget: float = 0.05,
                              absolute_budget: float = 0.0005,
                              curve_relative_budget: float = 0.25,
                              curve_absolute_budget: float = 0.002) -> bool:
    """Judge feasibility on measurement-aware losses instead of raw RRMS.

    The reported paper RRMS is unchanged and still reported; it is simply not
    used as the veto.  Its denominator is set by the whole curve, so a sweep
    whose points sit on the instrument floor produces a very large raw RRMS for
    every candidate.  A point that carries no measurable signal cannot
    discriminate between two cards, so it must not be able to reject one.
    """
    if min(relative_budget, absolute_budget, curve_relative_budget,
           curve_absolute_budget) < 0:
        raise ValueError("tradeoff budgets must be non-negative")

    def allowed(value: float, baseline: float, relative: float,
                absolute: float) -> bool:
        margin = max(absolute, relative * max(abs(baseline), 1e-30))
        return value <= baseline + margin

    candidate_ordinary, candidate_high = noise_regime_pair(candidate)
    reference_ordinary, reference_high = noise_regime_pair(reference)
    if not all((
        allowed(float(candidate["objective"]), float(reference["objective"]),
                relative_budget, absolute_budget),
        allowed(candidate_ordinary, reference_ordinary,
                relative_budget, absolute_budget),
        allowed(candidate_high, reference_high,
                relative_budget, absolute_budget),
    )):
        return False
    reference_curves = reference["curve_components"]
    for tag, row in candidate["curve_components"].items():
        baseline = reference_curves.get(tag)
        if baseline is None:
            continue
        if not allowed(float(row["total"]), float(baseline["total"]),
                       curve_relative_budget, curve_absolute_budget):
            return False
    return True


def balanced_regime_objective(curve_losses: dict[str, float],
                              *, high_voltage_floor: float = 1.48,
                              contribution_cap: float = 0.05,
                              tail_fraction: float = 0.25,
                              tail_weight: float = 0.25) -> dict:
    """Balance sweep/regime groups, then add a bounded curve-tail term."""
    if not curve_losses:
        raise ValueError("no curve losses")
    if contribution_cap <= 0:
        raise ValueError("contribution_cap must be positive")
    bounded = {
        tag: float(contribution_cap * np.tanh(value / contribution_cap))
        for tag, value in curve_losses.items()
    }
    groups = {}
    for kind in ("idvd", "idvg"):
        for region in ("ordinary", "high"):
            values = []
            for tag, value in bounded.items():
                if not tag.startswith(kind + "@"):
                    continue
                curve_bias = float(tag.split("@", 1)[1])
                is_high = curve_bias >= high_voltage_floor
                if is_high == (region == "high"):
                    values.append(value)
            if values:
                groups[f"{kind}_{region}"] = float(np.mean(values))
    group_mean = float(np.mean(list(groups.values())))
    values = np.asarray(list(bounded.values()), dtype=np.float64)
    tail_count = max(1, int(np.ceil(len(values) * tail_fraction)))
    tail = float(np.mean(np.partition(values, len(values) - tail_count)
                         [-tail_count:]))
    objective = (1.0 - tail_weight) * group_mean + tail_weight * tail
    return {
        "objective": float(objective), "group_mean": group_mean,
        "tail_mean": tail, "groups": groups, "bounded_curves": bounded,
    }


def reported_bias_rrms(per_curve: dict[str, dict], *, high: bool,
                       high_voltage_floor: float = 1.48) -> float:
    """Return unchanged paper RRMS averaged within one bias regime."""
    values = []
    for tag, row in per_curve.items():
        if not row.get("included") or row.get("rrms") is None:
            continue
        bias = float(tag.split("@", 1)[1])
        if (bias >= high_voltage_floor) == high:
            values.append(float(row["rrms"]))
    if not values:
        raise ValueError("requested bias regime contains no reported curves")
    return float(np.mean(values))


def within_tradeoff_budget(candidate: dict, reference: dict, *,
                           relative_budget: float = 0.05,
                           absolute_budget: float = 0.005,
                           curve_relative_budget: float = 0.25,
                           curve_absolute_budget: float = 0.05) -> bool:
    """Allow a small tradeoff without permitting a hidden curve collapse.

    The score dictionaries contain unchanged reported RRMS.  The rule is
    geometry-agnostic and is recomputed from the curves of any new transistor.
    """
    if min(relative_budget, absolute_budget, curve_relative_budget,
           curve_absolute_budget) < 0:
        raise ValueError("tradeoff budgets must be non-negative")

    def allowed(value: float, baseline: float, relative: float,
                absolute: float) -> bool:
        margin = max(absolute, relative * max(abs(baseline), 1e-30))
        return value <= baseline + margin

    candidate_ordinary = reported_bias_rrms(
        candidate["per_curve"], high=False)
    reference_ordinary = reported_bias_rrms(
        reference["per_curve"], high=False)
    if not all((
        allowed(float(candidate["rrms"]), float(reference["rrms"]),
                relative_budget, absolute_budget),
        allowed(candidate_ordinary, reference_ordinary,
                relative_budget, absolute_budget),
        allowed(float(candidate["high_voltage_rrms"]),
                float(reference["high_voltage_rrms"]),
                relative_budget, absolute_budget),
    )):
        return False
    for tag, reference_row in reference["per_curve"].items():
        candidate_row = candidate["per_curve"].get(tag)
        if (not reference_row.get("included")
                or reference_row.get("rrms") is None
                or candidate_row is None
                or not candidate_row.get("included")
                or candidate_row.get("rrms") is None):
            continue
        if not allowed(float(candidate_row["rrms"]),
                       float(reference_row["rrms"]),
                       curve_relative_budget, curve_absolute_budget):
            return False
    return True
