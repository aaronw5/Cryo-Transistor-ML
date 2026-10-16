"""Simulated 77-300 K temperature benchmark.

Only 77 K measurements exist, so temperature methods are evaluated on
simulated devices whose truth is known.  The synthetic world:

* the card is the 77 K card with ``tnom = 27 C``; with the released
  ``tnom = -196.15 C`` the junction model diverges above ~150 K;
* the card's own BSIM4 temperature coefficients stay zero (as released), so
  all temperature dependence beyond BSIM4's intrinsic kT/q, bandgap and
  junction physics comes from the seven paper parameters, each written as a
  ``temper`` expression

      p(T) = p_stock * m ** (w(T) ** g),   w(T) = (27 C - T) / (27 C + 196.15 C)

  clipped at w >= 0, so p(77 K) = p_stock * m (the cryogenic multiplier) and
  p(300.15 K) = p_stock (the SkyWater card's value);
* a card is the 14-vector (log m_p, log g_p) for p in the seven parameters.

The 77 K multipliers are anchored on the cold-loop seven-parameter fits to
the real measurements; ``g`` sets how the parameter moves between the ends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cold_start import cold_centers
from .parameter_sets import PARAMS7

TEMPS_OBSERVED = (77.0, 150.0, 225.0, 300.0)
TEMPS_HELDOUT = (110.0, 190.0, 260.0)
TEMPS_ALL = tuple(sorted(TEMPS_OBSERVED + TEMPS_HELDOUT))
T_COLD_C, T_STOCK_C = -196.15, 27.0
NAMES = tuple(f"m_{p}" for p in PARAMS7) + tuple(f"g_{p}" for p in PARAMS7)


def temperature_feature(temp_K: float | np.ndarray):
    """Temperature scaled to [-1, 1] over 77-300 K."""
    return 2.0 * (np.asarray(temp_K, dtype=np.float64) - 77.0) / 223.0 - 1.0


@dataclass
class TempBox:
    """Uniform box in z = (log m, log g) for one device."""
    tag: str
    stock: dict[str, float]
    lo: np.ndarray
    hi: np.ndarray

    def z_to_local(self, z):
        return 2.0 * (np.asarray(z) - self.lo) / (self.hi - self.lo) - 1.0

    def local_to_z(self, local):
        return self.lo + (np.asarray(local) + 1.0) * 0.5 * (self.hi - self.lo)


def card_params(stock: dict[str, float], z: np.ndarray) -> dict[str, str | float]:
    """NGSpice parameter overrides for card ``z`` (log m then log g)."""
    z = np.asarray(z, dtype=np.float64)
    out: dict[str, str | float] = {"tnom": T_STOCK_C}
    span = T_STOCK_C - T_COLD_C
    for i, name in enumerate(PARAMS7):
        m, g = float(np.exp(z[i])), float(np.exp(z[i + len(PARAMS7)]))
        out[name] = (f"{{{stock[name]:.10g}*pow({m:.10g},"
                     f"pow(max(({T_STOCK_C:g}-temper)/{span:g},0),{g:.8g}))}}")
    return out


def parameters_at(stock: dict[str, float], z: np.ndarray, temp_K: float) -> dict:
    """Numeric parameter values of card ``z`` at one temperature."""
    w = max((T_STOCK_C - (temp_K - 273.15)) / (T_STOCK_C - T_COLD_C), 0.0)
    return {name: stock[name] * float(np.exp(z[i])) ** (w ** float(np.exp(z[i + 7])))
            for i, name in enumerate(PARAMS7)}


def device_boxes(anchor_spec: Path, m_scale: float = 2.0,
                 g_range: tuple[float, float] = (0.25, 4.0)) -> dict[str, TempBox]:
    """Boxes centered on a cold-loop seven-parameter spec's centers.

    ``m_scale`` multiplies the spec's log half-widths for the cryogenic
    multipliers; ``g`` is log-uniform over ``g_range`` for every device.
    """
    spec = json.loads(Path(anchor_spec).read_text())
    stock_all = cold_centers(tuple(PARAMS7))
    boxes = {}
    for tag, center in spec["published_by_device"].items():
        stock = stock_all[tag]["center"]
        log_m = np.array([np.log(center[p] / stock[p]) for p in PARAMS7])
        half = np.array([spec["half_width_by_device"][tag][p]
                         for p in PARAMS7]) * m_scale
        lo = np.concatenate([log_m - half, np.full(7, np.log(g_range[0]))])
        hi = np.concatenate([log_m + half, np.full(7, np.log(g_range[1]))])
        boxes[tag] = TempBox(tag, stock, lo, hi)
    return boxes
