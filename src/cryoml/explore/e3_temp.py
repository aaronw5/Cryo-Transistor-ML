"""E3: temperature generalization under misspecification (simulated, 77-300 K).

World.  Every card is the SkyWater 77 K corner bin of a REAL_DEV geometry
with ``tnom = 27 C`` and the seven paper parameters set to numeric values at
the simulation temperature; ``x = log(p / p_stock)`` is the per-temperature
parameter vector (``p_stock`` from ``cryoml.temperature``).  Two truth tiers:

* ``W`` (well-specified): ``x(T) = mu * w(T)^g`` (the
  ``cryoml.temperature`` family; ``w = (300.15 K - T) / 223.15 K``).
* ``M`` (misspecified), with three effects the fitted laws cannot represent:
  - law shape: ``x(T) = c + (mu - c) * s(T) + b * 4 w (1 - w)``, where
    ``s`` is a logistic step in T with knee ``Tk`` and width ``tau``,
    normalised to ``s(77 K) = 1`` and ``s(300.15 K) = 0`` (an inflection
    point the power law ``w^g`` does not have), and ``b`` is a
    non-monotone bump on vth0, u0 and rdsw;
  - 300 K offset: ``x(300.15 K) = c != 0`` (the ``cryoml.temperature``
    family fixes the 300 K card at stock);
  - extrinsic series resistance outside the card on drain and source,
    ``R(T) = R300 * (1 + alpha (T - 300.15 K))`` per side, simulated with
    ``sky130_misspec.instance_lines``.

Fits never see the series resistors; every fitted card is simulated with
``spice_pdk.simulate_pdk`` / ``spice_batch.simulate_pdk_multi``.

Scoring: frozen-inclusion paper RRMS per temperature against the truth
curves (``temp_fit.TempProblem`` layout), strict: a non-finite included
curve makes that (device, temperature) value NaN.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..card_eval import device_by_tag
from ..cold_start import device_bins
from ..config import OUT_DIR
from ..data_io import load_device_curves
from ..parameter_sets import PARAMS7
from ..sky130_misspec import instance_lines
from ..spice_batch import simulate_pdk_multi
from ..spice_pdk import (LIB_FILE, PDK77K_DIR, _CORNER_BASENAME, _NGSPICE_BIN,
                         _batch_deck, _instance_line, _patch_params_in_bin,
                         _read_curves, ensure_pdk77k, simulate_pdk)
from ..temperature import device_boxes, temperature_feature

ANCHOR = OUT_DIR / "cold_loop" / "seven" / "round_4" / "spec.json"
T_COLD, T_HOT = 77.0, 300.15
TEMPS_DENSE = (77.0, 90.0, 100.0, 110.0, 130.0, 150.0, 170.0, 190.0, 210.0,
               225.0, 245.0, 260.0, 280.0, 300.0)
TEMPS_HELDOUT = (110.0, 190.0, 260.0)
OBS_SETS = {"O4": (77.0, 150.0, 225.0, 300.0), "O2": (77.0, 300.0)}
TEMPS_TEST = tuple(sorted(set(TEMPS_HELDOUT) | set(OBS_SETS["O4"])))
N_PAR = len(PARAMS7)
BUMP_SCALE = {"vth0": 0.05, "u0": 0.15, "rdsw": 0.15}
FD_STEP = 0.01


def w_of(temp_K) -> np.ndarray:
    return np.clip((T_HOT - np.asarray(temp_K, dtype=np.float64)) / (T_HOT - T_COLD),
                   0.0, 1.0)


# ------------------------------------------------------------------ truth

@lru_cache(maxsize=None)
def stock_and_box(tag: str):
    """Stock values and log-multiplier bounds (1.5x anchor box) per device."""
    box = device_boxes(ANCHOR, 1.5, (0.25, 4.0))[tag]
    stock = np.array([box.stock[p] for p in PARAMS7], dtype=np.float64)
    return stock, box.lo[:N_PAR].copy(), box.hi[:N_PAR].copy()


U0_SWITCH_MARGIN = 0.02     # in x = log(u0 / u0_stock), about 2 % below u0 = 1


def u0_switch_x(tag: str) -> float:
    """x_u0 at which the card value u0 reaches 1.  BSIM4 reads u0 > 1 as
    cm^2/Vs and divides it by 1e4, so the drain current drops by about 1e4
    across this value (I2-0 diagnosis, nmos_L1_W3)."""
    stock, _, _ = stock_and_box(tag)
    return float(math.log(1.0 / abs(stock[PARAMS7.index("u0")])))


def x_bounds(tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Fit bounds on x: the 1.5x box of log m, widened to contain 0, +-0.5,
    with the u0 upper bound kept below the BSIM4 u0 unit switch."""
    _, lo, hi = stock_and_box(tag)
    lo, hi = np.minimum(lo, 0.0) - 0.5, np.maximum(hi, 0.0) + 0.5
    k = PARAMS7.index("u0")
    hi[k] = min(hi[k], u0_switch_x(tag) - U0_SWITCH_MARGIN)
    return lo, hi


def u0_upper_x(tag: str) -> float:
    """Upper fit bound of x_u0 (``u0_switch_x - U0_SWITCH_MARGIN`` or lower)."""
    return float(x_bounds(tag)[1][PARAMS7.index("u0")])


def at_or_above_u0_switch(tag: str, x) -> bool:
    """True when the card with parameter vector ``x`` has u0 >= 1."""
    return bool(np.asarray(x, dtype=np.float64)[PARAMS7.index("u0")] >= u0_switch_x(tag))


def clip_u0(tag: str, x) -> tuple[np.ndarray, bool]:
    """``x`` with x_u0 clipped to the upper fit bound; second value is True
    when the clip changed x.  Only x_u0 and only its upper side are changed."""
    out = np.array(x, dtype=np.float64)
    k = PARAMS7.index("u0")
    upper = u0_upper_x(tag)
    clipped = bool(out[k] > upper)
    if clipped:
        out[k] = upper
    return out, clipped


def truth_u0_max(truth: dict, temps=TEMPS_DENSE) -> float:
    """Largest x_u0 of the truth over ``temps``."""
    k = PARAMS7.index("u0")
    return float(max(truth_x(truth, t)[k] for t in temps))


def crosses_u0_switch(truth: dict, temps=TEMPS_DENSE) -> bool:
    """True when the truth's x_u0 is at or above ``u0_switch_x - U0_SWITCH_MARGIN``
    at any of ``temps``."""
    return truth_u0_max(truth, temps) >= u0_switch_x(truth["tag"]) - U0_SWITCH_MARGIN


def draw_truth(tag: str, tier: str, rng: np.random.Generator,
               max_attempts: int = 50) -> dict:
    """One truth device whose u0 stays below the BSIM4 unit switch at every
    dense temperature (rejection sampling from the same generator stream;
    the number of rejected draws is recorded)."""
    for attempt in range(max_attempts):
        out = _draw_truth_once(tag, tier, rng)
        if not crosses_u0_switch(out):
            out["u0_switch_rejections"] = attempt
            return out
    raise RuntimeError(f"{tag}: no truth draw below the u0 unit switch "
                       f"in {max_attempts} attempts")


def _draw_truth_once(tag: str, tier: str, rng: np.random.Generator) -> dict:
    """One truth device.  mu is uniform in the 0.75x anchor box (as the
    existing temperature truths); the remaining draws follow the tier."""
    _, lo, hi = stock_and_box(tag)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo) / 1.5 * 0.75
    mu = centre + half * rng.uniform(-1, 1, N_PAR)
    out = {"tag": tag, "tier": tier, "mu": mu.tolist()}
    if tier == "W":
        out["g"] = np.exp(rng.uniform(math.log(1 / 3), math.log(3), N_PAR)).tolist()
        return out
    if tier != "M":
        raise ValueError(tier)
    knee = rng.uniform(120.0, 230.0)
    out["c"] = (rng.uniform(-1, 1, N_PAR)
                * np.array([0.05 if p == "vth0" else 0.10 for p in PARAMS7])).tolist()
    out["Tk"] = (knee + rng.uniform(-25.0, 25.0, N_PAR)).tolist()
    out["tau"] = rng.uniform(10.0, 35.0, N_PAR).tolist()
    out["bump"] = [float(rng.uniform(-1, 1) * BUMP_SCALE.get(p, 0.0)) for p in PARAMS7]
    stock, _, _ = stock_and_box(tag)
    device = device_by_tag(tag)
    rdsw = abs(stock[PARAMS7.index("rdsw")])
    out["rs300_ohm"] = float(rng.uniform(0.3, 1.0) * 0.5 * rdsw / device.W_um)
    out["alpha_per_K"] = float(rng.uniform(0.002, 0.0035))
    return out


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.asarray(v, dtype=np.float64)))


def truth_x(truth: dict, temp_K: float) -> np.ndarray:
    mu = np.asarray(truth["mu"])
    w = float(w_of(temp_K))
    if truth["tier"] == "W":
        return mu * w ** np.asarray(truth["g"])
    c, tk, tau = (np.asarray(truth[k]) for k in ("c", "Tk", "tau"))
    top, bottom = _sigmoid((tk - T_COLD) / tau), _sigmoid((tk - T_HOT) / tau)
    s = (_sigmoid((tk - temp_K) / tau) - bottom) / (top - bottom)
    return c + (mu - c) * s + np.asarray(truth["bump"]) * 4.0 * w * (1.0 - w)


def truth_rs(truth: dict, temp_K: float) -> float:
    if truth["tier"] != "M":
        return 0.0
    return float(truth["rs300_ohm"] * (1.0 + truth["alpha_per_K"] * (temp_K - T_HOT)))


def card(tag: str, x: np.ndarray) -> dict:
    stock, _, _ = stock_and_box(tag)
    values = stock * np.exp(np.asarray(x, dtype=np.float64))
    return {"tnom": 27.0, **{p: float(v) for p, v in zip(PARAMS7, values)}}


@lru_cache(maxsize=None)
def device_ctx(tag: str):
    device = device_by_tag(tag)
    return device, load_device_curves(device), device_bins()[tag]


def simulate_truth_at(tag: str, truth: dict, temp_K: float,
                      timeout_s: float = 20.0) -> np.ndarray:
    """Truth currents (concatenated curves) at one temperature, with the
    tier's extrinsic resistors.  With ``rs = 0`` this is ``simulate_pdk``."""
    device, curves, bin_index = device_ctx(tag)
    params = card(tag, truth_x(truth, temp_K))
    rs = truth_rs(truth, temp_K)
    if rs <= 0:
        return np.concatenate(simulate_pdk(device.dev_type, device.L_um, device.W_um,
                                           curves, params=params, bin_index=bin_index,
                                           temp_K=temp_K, timeout_s=timeout_s))
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    with tempfile.TemporaryDirectory(prefix="cryoml_e3_") as td_s:
        td = Path(td_s)
        base = PDK77K_DIR / _CORNER_BASENAME[device.dev_type]
        corner = td / _CORNER_BASENAME[device.dev_type]
        corner.write_text(_patch_params_in_bin(base.read_text(errors="replace"),
                                               device.dev_type, bin_index, params))
        lib_path = td / "lib77k.spice"
        lib_path.write_text(LIB_FILE.read_text().replace(base.as_posix(), corner.as_posix()))
        deck = _batch_deck(device.dev_type, device.L_um, device.W_um, curves, lib_path,
                           td, temp_K)
        stock_line = _instance_line(device.dev_type, device.L_um, device.W_um)
        if deck.count(stock_line) != 1:
            raise RuntimeError("instance line not found once in deck")
        deck = deck.replace(stock_line, instance_lines(device.dev_type, device.L_um,
                                                       device.W_um, rs))
        (td / "deck.sp").write_text(deck)
        try:
            proc = subprocess.run([_NGSPICE_BIN, "-b", str(td / "deck.sp")], cwd=str(td),
                                  capture_output=True, timeout=timeout_s, text=True,
                                  check=False)
            fatal = "could not find a valid modelname" in (proc.stdout + proc.stderr).lower()
        except subprocess.TimeoutExpired:
            fatal = True
        return np.concatenate(_read_curves(td, "raw_", curves, device.dev_type, fatal))


# ---------------------------------------------------------------- scoring

class TempScorer:
    """Frozen-inclusion paper RRMS layout of one device at one temperature
    (the ``temp_fit.TempProblem`` layout with a single temperature)."""

    def __init__(self, tag: str, temp_K: float, target: np.ndarray):
        from ..temp_fit import TempProblem
        box = device_boxes(ANCHOR, 1.5, (0.25, 4.0))[tag]
        self.problem = TempProblem(tag, box, {temp_K: np.asarray(target)}, (temp_K,))
        self.tag, self.temp = tag, temp_K
        self.layout = self.problem.layout
        self.n_curves = self.problem.n_curves

    def strict_ok(self, sims) -> bool:
        return all(np.all(np.isfinite(np.asarray(sims[r["index"]])[r["start"]:]))
                   for r in self.layout)

    def rrms(self, sims) -> float:
        """Mean included-curve RRMS; NaN if any included curve failed."""
        if not self.strict_ok(sims):
            return float("nan")
        return float(np.mean(self.problem.curve_rrms(sims)))

    def residual(self, sims) -> np.ndarray | None:
        if not self.strict_ok(sims):
            return None
        return self.problem.residual_vector(sims, np.ones(self.n_curves))


def simulate_cards(tag: str, xs: list, temp_K: float) -> list[list[np.ndarray]]:
    device, curves, bin_index = device_ctx(tag)
    cards = [card(tag, x) for x in xs]
    if len(cards) == 1:
        return [simulate_pdk(device.dev_type, device.L_um, device.W_um, curves,
                             params=cards[0], bin_index=bin_index, temp_K=temp_K,
                             timeout_s=20)]
    return simulate_pdk_multi(device.dev_type, device.L_um, device.W_um, curves, cards,
                              bin_index=bin_index, method="parallel", temp_K=temp_K,
                              timeout_s=20)


def score_card(tag: str, x: np.ndarray, temp_K: float, target: np.ndarray) -> float:
    scorer = TempScorer(tag, temp_K, target)
    return scorer.rrms(simulate_cards(tag, [np.asarray(x)], temp_K)[0])


# ------------------------------------------------- per-temperature LM fit

@dataclass
class Counter:
    n_sim: int = 0


def _jacobian(tag, scorer, x, temp_K, counter):
    """Residual and forward-FD Jacobian in x from one batched process."""
    xs = [x] + [x + FD_STEP * np.eye(N_PAR)[j] for j in range(N_PAR)]
    sims = simulate_cards(tag, xs, temp_K)
    counter.n_sim += len(xs)
    r0 = scorer.residual(sims[0])
    if r0 is None:
        return None, None
    J = np.zeros((r0.size, N_PAR))
    for j in range(N_PAR):
        rj = scorer.residual(sims[1 + j])
        if rj is not None:
            J[:, j] = (rj - r0) / FD_STEP
    return r0, J


def lm_fit(residual_and_jac, residual_only, theta0, lo, hi, max_iter=25,
           max_step=0.4):
    """Projected Levenberg-Marquardt on ``0.5 |r|^2`` within [lo, hi]."""
    theta = np.clip(np.asarray(theta0, dtype=np.float64), lo, hi)
    r, J = residual_and_jac(theta)
    if r is None:
        return {"theta": theta, "ssr": float("inf"), "J": None, "iters": 0}
    ssr, lam, it, small = float(r @ r), 1e-2, 0, 0
    for it in range(1, max_iter + 1):
        A, gvec = J.T @ J, J.T @ r
        accepted = False
        for _ in range(8):
            step = -np.linalg.solve(A + lam * (np.diag(np.diag(A)) + 1e-9 * np.trace(A)
                                               / len(theta) * np.eye(len(theta))
                                               + 1e-12 * np.eye(len(theta))), gvec)
            norm = np.max(np.abs(step))
            if norm > max_step:
                step *= max_step / norm
            trial = np.clip(theta + step, lo, hi)
            rt = residual_only(trial)
            if rt is not None and float(rt @ rt) < ssr:
                accepted = True
                break
            lam *= 4.0
        if not accepted:
            break
        gain = (ssr - float(rt @ rt)) / max(ssr, 1e-300)
        theta = trial
        lam = max(lam / 3.0, 1e-7)
        r, J = residual_and_jac(theta)
        if r is None:
            break
        ssr = float(r @ r)
        small = small + 1 if gain < 1e-3 else 0
        if small >= 2:
            break
    return {"theta": theta, "ssr": ssr, "J": J, "iters": it}


def fit_temperature(tag: str, temp_K: float, target: np.ndarray, starts: list,
                    max_iter: int = 25) -> dict:
    """Seven-parameter fit at one temperature from each start; keeps the
    lowest sum of squares.  Returns x, RRMS, H = J^T J (residual units)."""
    scorer = TempScorer(tag, temp_K, target)
    counter = Counter()
    lo, hi = x_bounds(tag)

    def rj(x):
        return _jacobian(tag, scorer, x, temp_K, counter)

    def ro(x):
        counter.n_sim += 1
        return scorer.residual(simulate_cards(tag, [x], temp_K)[0])

    best = None
    for start in starts:
        result = lm_fit(rj, ro, start, lo, hi, max_iter=max_iter)
        if best is None or result["ssr"] < best["ssr"]:
            best = result
    x = best["theta"]
    J = best["J"]
    H = (J.T @ J) if J is not None else np.full((N_PAR, N_PAR), np.nan)
    counter.n_sim += 1
    value = scorer.rrms(simulate_cards(tag, [x], temp_K)[0])
    return {"x": x.tolist(), "rrms": value, "ssr": best["ssr"], "H": H.tolist(),
            "n_sim": counter.n_sim, "iters": best["iters"]}


BANK = Path(__file__).resolve().parents[3] / "data" / "processed" / "temp_bank" / "v1"
BANK_N = 6000


@lru_cache(maxsize=1)
def _bank(tag: str):
    """First BANK_N samples of the existing temperature bank (the
    ``cryoml.temperature`` family in the 1.5x box): x at each bank
    temperature and the simulated currents there."""
    with np.load(BANK / f"{tag}.npz") as data:
        ok = data["ok"][:BANK_N]
        z = data["Z"][:BANK_N][ok]
        ids = data["IDS"][:BANK_N][ok]
        temps = tuple(float(t) for t in data["temps"])
    xs = {t: z[:, :N_PAR] * w_of(t) ** np.exp(z[:, N_PAR:]) for t in temps}
    return temps, xs, ids


def bank_starts(tag: str, temp_K: float, target: np.ndarray, n: int,
                min_distance: float = 0.3) -> list[np.ndarray]:
    """The ``n`` lowest-RRMS distinct bank samples at the bank temperature
    nearest ``temp_K`` (scored against ``target`` without NGSpice)."""
    temps, xs, ids = _bank(tag)
    j = int(np.argmin([abs(t - temp_K) for t in temps]))
    scorer = TempScorer(tag, temp_K, target)
    n_curves = len(scorer.problem.curves)
    columns = []
    for row in scorer.layout:
        lo, hi = scorer.problem.bounds[row["index"] % n_curves]
        block = ids[:, j, lo + row["start"]:hi].astype(np.float64)
        columns.append(np.sqrt(np.mean((block - row["target"][None]) ** 2, 1))
                       / row["denominator"])
    scores = np.nan_to_num(np.mean(np.stack(columns, 1), 1), nan=np.inf)
    chosen = []
    for index in np.argsort(scores):
        x = xs[temps[j]][index]
        if not chosen or min(np.sqrt(np.mean((x - c) ** 2)) for c in chosen) >= min_distance:
            chosen.append(x)
        if len(chosen) >= n:
            break
    return chosen


def fit_sequence(tag: str, targets: dict, temps: tuple, max_iter: int = 20,
                 n_bank_cold: int = 2) -> dict:
    """Per-temperature fits over ``temps`` (ascending): the lowest
    temperature from the ``n_bank_cold`` best bank samples; every later
    temperature from the previous fit and the best bank sample at the
    nearest bank temperature."""
    out = {}
    previous = None
    for temp in sorted(temps):
        if previous is None:
            starts = bank_starts(tag, temp, targets[temp], n_bank_cold)
        else:
            starts = [np.asarray(previous)] + bank_starts(tag, temp, targets[temp], 1)
        row = fit_temperature(tag, temp, targets[temp], starts, max_iter)
        out[f"{temp:g}"] = row
        previous = row["x"]
    return out


# ------------------------------------------------------- parametric laws

LAW_VARIANTS = {
    # name: (free mu, free c, free lg, default lg source)
    "par14": dict(c=False, g="free"),      # cryoml.temperature family
    "par14_g1": dict(c=False, g="one"),
    "par3": dict(c=True, g="free"),
    "par3_g1": dict(c=True, g="one"),
    "par3_pop": dict(c=True, g="pop"),
}


def law_x(theta_mu, theta_c, lg, temp_K):
    wg = float(w_of(temp_K)) ** np.exp(lg)
    return theta_c + (theta_mu - theta_c) * wg


def fit_law(tag: str, targets: dict, temps: tuple, variant: str, init: dict,
            pop_lg: np.ndarray | None = None, max_iter: int = 20) -> dict:
    """Fit a parametric temperature law jointly on ``temps`` (NGSpice LM,
    Jacobian by chain rule from batched per-temperature FD Jacobians).
    ``init`` holds per-temperature fits (``fit_sequence`` rows) used for the
    starting point: mu = x(77 K), c = x(300 K) and lg by a grid search on
    the H-weighted per-temperature fits."""
    spec = LAW_VARIANTS[variant]
    scorers = {t: TempScorer(tag, t, targets[t]) for t in temps}
    counter = Counter()
    xs = {t: np.asarray(init[f"{t:g}"]["x"]) for t in temps}
    mu0 = xs[min(temps)]
    c0 = xs[max(temps)] if spec["c"] else np.zeros(N_PAR)
    if spec["g"] == "one":
        lg0 = np.zeros(N_PAR)
    elif spec["g"] == "pop":
        lg0 = np.asarray(pop_lg, dtype=np.float64)
    else:
        grid = np.linspace(math.log(1 / 6), math.log(6), 41)
        lg0 = np.zeros(N_PAR)
        middle = [t for t in temps if t not in (min(temps), max(temps))]
        for k in range(N_PAR):
            if not middle:
                break
            errs = []
            for lg in grid:
                err = 0.0
                for t in middle:
                    h = max(np.asarray(init[f"{t:g}"]["H"])[k, k], 1e-12)
                    err += h * (law_x(mu0[k], c0[k], lg, t) - xs[t][k]) ** 2
                errs.append(err)
            lg0[k] = grid[int(np.argmin(errs))]
    free = ["mu"] + (["c"] if spec["c"] else []) + (["lg"] if spec["g"] == "free" else [])
    xlo, xhi = x_bounds(tag)
    lo = np.concatenate([xlo] + ([xlo] if spec["c"] else [])
                        + ([np.full(N_PAR, math.log(1 / 6))] if spec["g"] == "free" else []))
    hi = np.concatenate([xhi] + ([xhi] if spec["c"] else [])
                        + ([np.full(N_PAR, math.log(6))] if spec["g"] == "free" else []))

    def unpack(theta):
        parts = dict(zip(free, np.split(theta, len(free))))
        return parts["mu"], parts.get("c", c0 if spec["c"] else np.zeros(N_PAR)), \
            parts.get("lg", lg0)

    theta0 = np.concatenate([mu0] + ([c0] if spec["c"] else [])
                            + ([lg0] if spec["g"] == "free" else []))

    def dx_dtheta(mu, c, lg, t):
        w = float(w_of(t))
        g = np.exp(lg)
        wg = w ** g
        blocks = [np.diag(wg)]
        if spec["c"]:
            blocks.append(np.diag(1.0 - wg))
        if spec["g"] == "free":
            dlg = (mu - c) * wg * (math.log(w) if w > 0 else 0.0) * g
            blocks.append(np.diag(dlg))
        return np.concatenate(blocks, 1)

    def rj(theta):
        mu, c, lg = unpack(theta)
        rs, Js = [], []
        for t in temps:
            r, J = _jacobian(tag, scorers[t], law_x(mu, c, lg, t), t, counter)
            if r is None:
                return None, None
            rs.append(r)
            Js.append(J @ dx_dtheta(mu, c, lg, t))
        return np.concatenate(rs), np.concatenate(Js)

    def ro(theta):
        mu, c, lg = unpack(theta)
        rs = []
        for t in temps:
            counter.n_sim += 1
            r = scorers[t].residual(simulate_cards(tag, [law_x(mu, c, lg, t)], t)[0])
            if r is None:
                return None
            rs.append(r)
        return np.concatenate(rs)

    result = lm_fit(rj, ro, theta0, lo, hi, max_iter=max_iter)
    mu, c, lg = unpack(result["theta"])
    return {"mu": mu.tolist(), "c": np.asarray(c).tolist(), "lg": np.asarray(lg).tolist(),
            "ssr": result["ssr"], "iters": result["iters"], "n_sim": counter.n_sim}


def pooled_lg(train_fits: list[dict], temps_fit: tuple) -> np.ndarray:
    """Population-shared exponent per parameter for ``par3_pop``: grid search
    of lg_k minimising the H-weighted squared distance between the law
    (mu = x(77 K), c = x(300 K) of each training device) and the training
    devices' per-temperature fits at ``temps_fit``."""
    grid = np.linspace(math.log(1 / 6), math.log(6), 81)
    out = np.zeros(N_PAR)
    for k in range(N_PAR):
        errs = np.zeros(len(grid))
        for fits in train_fits:
            mu, c = fits["77"]["x"][k], fits["300"]["x"][k]
            for t in temps_fit:
                if t in (77.0, 300.0):
                    continue
                row = fits[f"{t:g}"]
                h = max(np.asarray(row["H"])[k, k], 1e-12)
                h /= max(np.trace(np.asarray(row["H"])) / N_PAR, 1e-12)
                for i, lg in enumerate(grid):
                    errs[i] += h * (law_x(mu, c, lg, t) - row["x"][k]) ** 2
        out[k] = grid[int(np.argmin(errs))]
    return out


# ------------------------------------------------------ interpolation

def linear_interp(anchor_temps, anchor_x, temp_K) -> np.ndarray:
    anchor_x = np.asarray(anchor_x)
    return np.array([np.interp(temp_K, anchor_temps, anchor_x[:, k]) for k in range(N_PAR)])


def phi(anchor_temps, temp_K) -> float:
    """Zero at every anchor temperature, max 1 over [77, 300] K."""
    grid = np.linspace(T_COLD, 300.0, 447)

    def raw(t):
        return np.prod([abs(temperature_feature(t) - temperature_feature(a))
                        for a in anchor_temps], axis=0)
    return float(raw(temp_K) / np.max(raw(grid)))
