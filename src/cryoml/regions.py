"""Sweep-region primitives shared by the extraction, diagnostic and plot code.

A whole-curve RRMS hides *where* a card is wrong: it can be several times
better above the knee and much worse below it and still look improved.  Every
stage that needs to resolve position along a sweep uses the helpers here, so
the section boundaries and the low-drain definition cannot drift apart between
the loss, the guard, the diagnostics and the figures.
"""
from __future__ import annotations

import numpy as np

# Quartile-style split used by the surrogate loss and the segment tables.
SECTION_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.7, 1.0)

# Finer split for measured-card refinement: the triode region needs more than
# one term or it is outvoted by numerically larger sections elsewhere.  On a
# 0-2 V drain sweep this is 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-1.0, 1.0-1.5, 1.5-2 V.
FINE_SECTION_EDGES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)

# Low-drain triode region: small |Vd| at strong gate drive.
LOW_DRAIN_MAX_V = 0.6
LOW_DRAIN_GATE_MIN_V = 1.48


def section_spans(count: int,
                  edges: tuple[float, ...] = SECTION_EDGES) -> list[tuple[int, int]]:
    """Index spans for each normalized sweep section of a curve."""
    if count <= 0:
        return []
    spans = []
    for left, right in zip(edges[:-1], edges[1:]):
        low = int(round(left * count))
        high = min(count, max(low + 1, int(round(right * count))))
        spans.append((low, high))
    return spans


def section_rrms(simulated: np.ndarray, target: np.ndarray,
                 denominator: float,
                 edges: tuple[float, ...] = SECTION_EDGES,
                 penalty: float = 1e3) -> list[float]:
    """RRMS within each sweep section, using the unchanged curve denominator."""
    simulated = np.asarray(simulated, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    count = min(len(simulated), len(target))
    simulated, target = simulated[:count], target[:count]
    values = []
    for low, high in section_spans(count, edges):
        high = min(high, count)
        if high <= low or denominator <= 0:
            values.append(penalty)
            continue
        value = float(np.sqrt(np.mean(
            (simulated[low:high] - target[low:high]) ** 2)) / denominator)
        values.append(value if np.isfinite(value) else penalty)
    return values


def section_rrms_matrix(simulated: np.ndarray, target: np.ndarray,
                        denominator: float,
                        edges: tuple[float, ...] = SECTION_EDGES) -> np.ndarray:
    """Vectorized :func:`section_rrms` over a stack of simulated curves."""
    simulated = np.asarray(simulated, dtype=np.float64)
    if simulated.ndim == 1:
        simulated = simulated[None, :]
    target = np.asarray(target, dtype=np.float64)
    count = min(simulated.shape[1], len(target))
    simulated, target = simulated[:, :count], target[:count]
    columns = [
        np.sqrt(np.mean(
            (simulated[:, low:high] - target[low:high]) ** 2, axis=1))
        / max(denominator, 1e-30)
        for low, high in section_spans(count, edges)
    ]
    return np.stack(columns, axis=1)


def low_drain_mask(sweep: np.ndarray,
                   drain_max: float = LOW_DRAIN_MAX_V) -> np.ndarray:
    """Points below ``drain_max`` on an output-curve sweep."""
    return np.abs(np.asarray(sweep, dtype=np.float64)) < drain_max


def is_low_drain_curve(kind: str, fixed: float,
                       gate_min: float = LOW_DRAIN_GATE_MIN_V) -> bool:
    """True for output curves taken at strong gate drive."""
    return kind == "idvd" and abs(float(fixed)) >= gate_min


def masked_rrms(simulated: np.ndarray, target: np.ndarray,
                mask: np.ndarray, denominator: float) -> float:
    """RRMS over a subset of a curve, or NaN when the subset is empty."""
    simulated = np.asarray(simulated, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    count = min(len(simulated), len(target), len(mask))
    selected = np.asarray(mask, dtype=bool)[:count]
    if not selected.any() or denominator <= 0:
        return float("nan")
    residual = simulated[:count][selected] - target[:count][selected]
    return float(np.sqrt(np.mean(residual ** 2)) / denominator)
