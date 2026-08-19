"""Shared neural and curve-layout components for foundation studies."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import VD_METRIC_BIASES, VG_METRIC_BIASES

I_REF = 1e-9
CANONICAL_CURVE_TAGS = tuple(
    [("idvd", bias) for bias in VD_METRIC_BIASES]
    + [("idvg", bias) for bias in VG_METRIC_BIASES])


def signed_log(values):
    if isinstance(values, np.ndarray):
        return np.sign(values) * np.log1p(np.abs(values) / I_REF)
    return torch.sign(values) * torch.log1p(torch.abs(values) / I_REF)


def current_from_signed_log(values):
    if isinstance(values, np.ndarray):
        return np.sign(values) * I_REF * np.expm1(np.abs(values))
    # The generated cards remain well below |slog(I)|=20. Clamping only the
    # linear-current branch prevents an early random prediction from turning
    # an off-curve normalization into an unbounded exp() gradient; the
    # standardized signed-log branch still supplies a gradient outside it.
    safe = torch.clamp(torch.abs(values), max=20.0)
    return torch.sign(values) * I_REF * torch.expm1(safe)


def canonical_point_order(kinds, fixeds, slices) -> tuple[np.ndarray, list[dict]]:
    """Return point indices that place every device in one curve order."""
    kinds = [str(value) for value in kinds]
    fixeds = np.abs(np.asarray(fixeds, dtype=np.float64))
    slices = np.asarray(slices, dtype=np.int64)
    point_indices, descriptions, cursor = [], [], 0
    used = set()
    for kind, bias in CANONICAL_CURVE_TAGS:
        matches = [index for index, (candidate_kind, candidate_bias) in
                   enumerate(zip(kinds, fixeds))
                   if candidate_kind == kind and abs(candidate_bias - bias) < 5e-3]
        if len(matches) != 1:
            raise ValueError(
                f"expected one {kind}@{bias:g} curve, found {len(matches)}")
        index = matches[0]
        if index in used:
            raise ValueError("curve layout reused one source curve")
        used.add(index)
        start, stop = (int(value) for value in slices[index])
        indices = np.arange(start, stop, dtype=np.int64)
        point_indices.append(indices)
        descriptions.append({
            "kind": kind, "bias": float(bias), "tag": f"{kind}@{bias:g}",
            "start": cursor, "stop": cursor + len(indices),
            "source_curve_index": index,
        })
        cursor += len(indices)
    return np.concatenate(point_indices), descriptions


def voltage_point_weights(curve_descriptions: list[dict],
                          n_points: int,
                          *, emphasize_high_voltage: bool = False) -> np.ndarray:
    """Return point weights for the forward loss.

    The controlled seven-vs-expanded comparison is uniform by default.  An
    explicit 1x/2x/4x high-voltage ablation is available, but it must not be
    confused with diagnosing whether the parameter surface or surrogate can
    represent the measured strong-inversion curves in the first place.
    """
    weights = np.ones(n_points, dtype=np.float32)
    if not emphasize_high_voltage:
        return weights
    for curve in curve_descriptions:
        multiplier = 4.0 if curve["bias"] >= 1.85 - 1e-6 else (
            2.0 if curve["bias"] >= 1.48 - 1e-6 else 1.0)
        weights[curve["start"]:curve["stop"]] = multiplier
    return weights / float(np.mean(weights))


def high_voltage_point_mask(curve_descriptions: list[dict],
                            n_points: int,
                            floor: float = 1.48) -> np.ndarray:
    mask = np.zeros(n_points, dtype=bool)
    for curve in curve_descriptions:
        if curve["bias"] >= floor - 1e-6:
            mask[curve["start"]:curve["stop"]] = True
    return mask


def mlp(sizes: list[int], activation=nn.GELU) -> nn.Sequential:
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class FoundationSurrogate(nn.Module):
    def __init__(self, n_parameters: int, n_points: int,
                 hidden: tuple[int, ...]):
        super().__init__()
        self.n_parameters = n_parameters
        self.n_points = n_points
        self.network = mlp([n_parameters + 3, *hidden, n_points])

    def forward(self, inputs):
        return self.network(inputs)


class InverseAutoencoder(nn.Module):
    """I-V plus geometry -> bounded local compact-model parameters."""

    def __init__(self, n_iv_features: int, n_parameters: int,
                 hidden: tuple[int, ...]):
        super().__init__()
        self.n_iv_features = int(n_iv_features)
        self.network = mlp(
            [self.n_iv_features + 3, *hidden, n_parameters])

    def forward(self, normalized_iv, geometry):
        inputs = torch.cat([normalized_iv, geometry], dim=1)
        return torch.tanh(self.network(inputs))


def geometry_ranges(devices) -> dict[str, float]:
    log_l = np.log([device.L_um for device in devices])
    log_w = np.log([device.W_um for device in devices])
    return {
        "log_l_min": float(log_l.min()), "log_l_max": float(log_l.max()),
        "log_w_min": float(log_w.min()), "log_w_max": float(log_w.max()),
    }


def _scale_minus_one_one(value: float, lo: float, hi: float) -> float:
    return float(2.0 * (value - lo) / (hi - lo) - 1.0)


def geometry_features(device, ranges: dict[str, float]) -> np.ndarray:
    return np.asarray([
        -1.0 if device.dev_type == "nmos" else 1.0,
        _scale_minus_one_one(
            math.log(device.L_um), ranges["log_l_min"], ranges["log_l_max"]),
        _scale_minus_one_one(
            math.log(device.W_um), ranges["log_w_min"], ranges["log_w_max"]),
    ], dtype=np.float32)


def decode_surrogate_output(normalized_slog: torch.Tensor,
                            slog_mean: torch.Tensor,
                            slog_std: torch.Tensor) -> tuple[torch.Tensor,
                                                             torch.Tensor]:
    slog = normalized_slog * slog_std + slog_mean
    return slog, current_from_signed_log(slog)


def hybrid_forward_loss(prediction_normalized: torch.Tensor,
                        target_normalized: torch.Tensor,
                        target_current: torch.Tensor,
                        device_ids: torch.Tensor,
                        slog_mean: torch.Tensor,
                        slog_std: torch.Tensor,
                        current_scales: torch.Tensor,
                        point_weights: torch.Tensor,
                        linear_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Weighted standardized-log plus curve-normalized linear current loss."""
    weights = point_weights[device_ids]
    weights = weights / torch.mean(weights, dim=1, keepdim=True)
    slog_sq = (prediction_normalized - target_normalized) ** 2
    _prediction_slog, prediction_current = decode_surrogate_output(
        prediction_normalized, slog_mean, slog_std)
    scale = torch.clamp(current_scales[device_ids], min=1e-30)
    normalized_linear_error = (
        (prediction_current - target_current) / scale)
    linear_point_loss = F.smooth_l1_loss(
        normalized_linear_error, torch.zeros_like(normalized_linear_error),
        beta=1.0, reduction="none")
    slog_loss = torch.mean(weights * slog_sq)
    linear_loss = torch.mean(weights * linear_point_loss)
    total = slog_loss + float(linear_weight) * linear_loss
    return total, {"slog": slog_loss, "linear": linear_loss}


def segmented_hybrid_loss(
        prediction_normalized: torch.Tensor,
        target_normalized: torch.Tensor,
        target_current: torch.Tensor,
        device_ids: torch.Tensor,
        slog_mean: torch.Tensor,
        slog_std: torch.Tensor,
        current_scales: torch.Tensor,
        curve_descriptions: list[dict],
        linear_weight: float = 1.0,
        rrms_weight: float = 1.0,
        segment_edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.7, 1.0),
        worst_segment_weight: float = 0.25) -> tuple[torch.Tensor, dict]:
    """Curve- and sweep-segment-balanced reconstruction loss.

    The normalized sweep bins are device-independent.  Averaging within each
    bin before averaging curves prevents a well-fit saturation tail from
    hiding a poor low-sweep knee (and vice versa).  A small worst-segment term
    discourages trading one part of a curve for another.
    """
    if len(segment_edges) < 2 or segment_edges[0] != 0.0:
        raise ValueError("segment_edges must start at zero")
    if segment_edges[-1] != 1.0 or any(
            right <= left for left, right in zip(
                segment_edges[:-1], segment_edges[1:])):
        raise ValueError("segment_edges must increase and end at one")
    _prediction_slog, prediction_current = decode_surrogate_output(
        prediction_normalized, slog_mean, slog_std)
    scale = torch.clamp(current_scales[device_ids], min=1e-30)
    slog_sq = (prediction_normalized - target_normalized) ** 2
    normalized_linear_error = (
        prediction_current - target_current) / scale
    current_error = prediction_current - target_current
    linear_point_loss = F.smooth_l1_loss(
        normalized_linear_error, torch.zeros_like(normalized_linear_error),
        beta=1.0, reduction="none")

    curve_losses = []
    segment_losses = []
    segment_slog_losses = []
    segment_linear_losses = []
    segment_rrms_losses = []
    for curve in curve_descriptions:
        start, stop = int(curve["start"]), int(curve["stop"])
        length = stop - start
        if length <= 0:
            continue
        current_curve_segments = []
        curve_denominator = torch.mean(
            torch.abs(target_current[:, start:stop]), dim=1)
        # A data-derived 60 dB floor prevents a synthetic off-state curve
        # with nearly zero mean current from taking an unbounded RRMS.  The
        # audit shows this only affects the bottom 12% of idvd@0.37 samples
        # and about 1% of idvd@0.74/1.11 samples; signal-bearing curves are
        # unchanged.
        curve_floor = 1e-3 * torch.max(scale[:, start:stop], dim=1).values
        curve_denominator = torch.maximum(
            curve_denominator, torch.clamp(curve_floor, min=1e-30))
        for left, right in zip(segment_edges[:-1], segment_edges[1:]):
            segment_start = start + int(round(left * length))
            segment_stop = start + int(round(right * length))
            segment_stop = max(segment_stop, segment_start + 1)
            segment_stop = min(segment_stop, stop)
            span = slice(segment_start, segment_stop)
            slog_value = torch.mean(slog_sq[:, span])
            linear_value = torch.mean(linear_point_loss[:, span])
            sample_rrms = (
                torch.sqrt(torch.mean(current_error[:, span] ** 2, dim=1)
                           + 1e-30)
                / curve_denominator)
            # Optimize a robust loss *of* each section RRMS.  This preserves
            # a distinct RRMS signal per region while preventing one extreme
            # off-state Monte Carlo sample from destabilizing a batch.
            bounded_sample_rrms = torch.clamp(sample_rrms, max=5.0)
            rrms_value = torch.mean(F.smooth_l1_loss(
                bounded_sample_rrms, torch.zeros_like(bounded_sample_rrms),
                beta=0.25, reduction="none"))
            raw_rrms_value = torch.mean(sample_rrms)
            value = (slog_value + float(linear_weight) * linear_value
                     + float(rrms_weight) * rrms_value)
            current_curve_segments.append(value)
            segment_losses.append(value)
            segment_slog_losses.append(slog_value)
            segment_linear_losses.append(linear_value)
            segment_rrms_losses.append(raw_rrms_value)
        stacked = torch.stack(current_curve_segments)
        curve_losses.append(
            torch.mean(stacked)
            + float(worst_segment_weight) * torch.max(stacked))
    if not curve_losses:
        raise ValueError("curve_descriptions contains no non-empty curves")
    loss = torch.mean(torch.stack(curve_losses))
    return loss, {
        "slog": torch.mean(torch.stack(segment_slog_losses)),
        "linear": torch.mean(torch.stack(segment_linear_losses)),
        "segment_rrms": torch.mean(torch.stack(segment_rrms_losses)),
        "mean_segment": torch.mean(torch.stack(segment_losses)),
        "worst_segment": torch.max(torch.stack(segment_losses)),
    }
