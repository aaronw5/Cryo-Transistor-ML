"""I11-2: shared bias-only correction with a temperature input, held-out temperature.

``I_corr = I_card * exp(g)``.  Arms (pre-registration
``out/research/explore/I11_2/prereg.json``):

* ``M_bias_T``: ``g(is_pmos, |Vgs|/VDD, |Vds|/VDD, t)``, ``t = ln(T / 300 K)``;
  the I3-2 ``M_bias`` recipe with a fourth standardized input
  (``TempBiasCorrection``, ``fit_bias_t``).  At the held-out record the
  network is evaluated at the record's temperature (``bind``).
* ``M_bias_pool``: the unchanged I3-2 ``M_bias`` (no temperature input)
  trained on the pooled training temperatures.
* ``M_bias_near``: ``M_bias`` trained at the training temperature nearest in
  ``ln T`` (``nearest_temp``).
* ``M_bias_in``: ``M_bias_T`` trained on every temperature, the held-out
  device excluded (reference only).
* ``lin_T`` / ``lin_pool`` (IHP secondary): ``g = tanh(c_p + a_p u + b_p v
  [+ d_p t])`` per polarity (``TempLinCorrection``, ``fit_lin_t``;
  ``lin_pool`` is the I9-2 ``lin`` class).

Fold construction (``training_set``) keeps every record of the held-out
device (IHP: the DUT; TU Delft: its geometry group) and, except for
``M_bias_in``, every record at the held-out temperature out of training
(``check_training``).

Statistics: per-unit mean log ratio over a temperature stratum
(``unit_log_ratios``), geometric-mean ratio with a device or cluster
bootstrap percentile CI at a given level (``gmr_ci``), the labels and the
decision of the pre-registered rule (``label_ihp``, ``label_tud``,
``decide``), and the training-only statistics S1_loto / S3_loto
(``s1_loto``, ``s3_loto_tasks``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .e1_correction import OUT_SCALE, Correction, DeviceRec, _loss_fn, make_batch

T_REF_K = 300.0
COLS = (0, 1, 2)                          # is_pmos, |Vgs|/VDD, |Vds|/VDD
HYPER_BIAS_T = {"hidden": 16, "layers": 2, "l2": 1e-4, "lr": 3e-3, "steps": 2000,
                "seeds": (0, 1, 2, 3, 4)}   # = i3_2_bias.HYPER_BIAS
HYPER_LIN_T = {"l2": 1e-4, "lr": 3e-3, "steps": 2000}   # = i4_1_baselines.HYPER_POLY
S3_HYPER = {"seeds": (0,), "steps": 500}                 # = i7_1_transfer.S3_HYPER
I7_1_RULE = {"intercept": 0.007619, "slope": -0.485184}  # frozen I7-1 rule, y = a + b S1

N_BOOT = 10000
ALPHA = 0.05
N_PRIMARY = 8
LEVEL_PRIMARY = 1.0 - ALPHA / N_PRIMARY


# ---------------------------------------------------------------------------
# Temperature of a record
# ---------------------------------------------------------------------------
def temp_kelvin(rec: DeviceRec) -> float:
    """Measured temperature (K) of a starting-card record: TU Delft
    ``temp_stage_K``, IHP ``temp_C`` + 273.15, else the label ``T_K``."""
    meta = rec.meta
    if meta.get("temp_stage_K") is not None:
        return float(meta["temp_stage_K"])
    if meta.get("temp_C") is not None:
        return float(meta["temp_C"]) + 273.15
    return float(meta["T_K"])


def t_feature(temp_k: float) -> float:
    return math.log(float(temp_k) / T_REF_K)


def nearest_temp(t_held: int, temps) -> int:
    """Training temperature label nearest to ``t_held`` in ln T (ties: the
    lower temperature)."""
    cand = sorted(int(t) for t in temps if int(t) != int(t_held))
    if not cand:
        raise ValueError("no training temperature")
    return min(cand, key=lambda t: (abs(math.log(t / t_held)), t))


# ---------------------------------------------------------------------------
# Corrections with a temperature input
# ---------------------------------------------------------------------------
@dataclass
class TempBiasCorrection(Correction):
    """MLP on (phi[:, cols], t).  ``t_value`` is the bound temperature
    feature; ``g`` refuses to run unbound."""
    cols: tuple = COLS
    t_value: float | None = None

    def bind(self, temp_k: float) -> "TempBiasCorrection":
        return TempBiasCorrection(self.arm, self.mu, self.sd, nets=self.nets, cols=self.cols,
                                  t_value=t_feature(temp_k))

    def g(self, phi, group_tag=None, group_kind=None) -> np.ndarray:
        if self.t_value is None:
            raise RuntimeError(f"{self.arm}: temperature not bound")
        phi = np.asarray(phi, dtype=np.float64)
        x = np.column_stack([phi[:, list(self.cols)], np.full(len(phi), self.t_value)])
        return Correction.g(self, x)

    def to_json(self) -> dict:
        return {**Correction.to_json(self), "kind": "mlp_t", "cols": list(self.cols),
                "t_value": self.t_value, "t_ref_k": T_REF_K}

    @classmethod
    def from_json(cls, row: dict) -> "TempBiasCorrection":
        base = Correction.from_json(row)
        return cls(base.arm, base.mu, base.sd, nets=base.nets,
                   cols=tuple(int(c) for c in row["cols"]), t_value=row.get("t_value"))


@dataclass
class TempLinCorrection(Correction):
    """g = OUT_SCALE * tanh(c_p + a_p u + b_p v + d_p t), standardized u, v, t,
    coefficients ``coef[polarity]`` = (c, a, b, d)."""
    coef: np.ndarray | None = None
    t_value: float | None = None

    def bind(self, temp_k: float) -> "TempLinCorrection":
        return TempLinCorrection(self.arm, self.mu, self.sd, coef=self.coef,
                                 t_value=t_feature(temp_k))

    def g(self, phi, group_tag=None, group_kind=None) -> np.ndarray:
        if self.t_value is None:
            raise RuntimeError(f"{self.arm}: temperature not bound")
        phi = np.asarray(phi, dtype=np.float64)
        f = lin_features(phi[:, 1], phi[:, 2], np.full(len(phi), self.t_value), self.mu, self.sd)
        p = phi[:, 0]
        h = (1.0 - p) * (f @ self.coef[0]) + p * (f @ self.coef[1])
        return OUT_SCALE * np.tanh(h)

    def to_json(self) -> dict:
        return {"arm": self.arm, "kind": "lin_t", "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "coef": self.coef.tolist(), "t_value": self.t_value, "t_ref_k": T_REF_K,
                "out_scale": OUT_SCALE}

    @classmethod
    def from_json(cls, row: dict) -> "TempLinCorrection":
        return cls(row["arm"], np.asarray(row["mu"], float), np.asarray(row["sd"], float),
                   coef=np.asarray(row["coef"], float), t_value=row.get("t_value"))


def lin_features(u, v, t, mu, sd) -> np.ndarray:
    return np.column_stack([np.ones(len(u)), (u - mu[0]) / sd[0], (v - mu[1]) / sd[1],
                            (t - mu[2]) / sd[2]])


def correction_from_json_i11(row: dict) -> Correction:
    kind = row.get("kind")
    if kind == "mlp_t":
        return TempBiasCorrection.from_json(row)
    if kind == "lin_t":
        return TempLinCorrection.from_json(row)
    from .i4_1_baselines import correction_from_json_any
    return correction_from_json_any(row)


def point_t_feature(batch, train: list[DeviceRec]) -> np.ndarray:
    """Temperature feature of every batch point (from its record)."""
    tf = np.array([t_feature(temp_kelvin(r)) for r in train])
    return tf[batch.curve_dev[batch.curve_id]]


def train_mlp(x: np.ndarray, loss, hp: dict, first_binary: bool = True,
              trace: list | None = None) -> tuple:
    """The ``i3_2_bias.fit_bias`` training loop on the feature matrix ``x``:
    standardization with the training-point mean / std (column 0 kept 0/1
    when ``first_binary``), zero-initialized last layer, per-member
    ``torch.manual_seed(seed)``, L2 on weight matrices, full-batch Adam,
    float64.  Returns (mu, sd, nets)."""
    import torch
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    if first_binary:
        mu[0], sd[0] = 0.0, 1.0
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = torch.tensor((x - mu) / sd, dtype=torch.float64)
    nets = []
    for seed in hp["seeds"]:
        torch.manual_seed(seed)
        dims = [z.shape[1]] + [hp["hidden"]] * hp["layers"] + [1]
        mods = [torch.nn.Linear(a, b).double() for a, b in zip(dims[:-1], dims[1:])]
        torch.nn.init.zeros_(mods[-1].weight)
        torch.nn.init.zeros_(mods[-1].bias)
        params = [p for m in mods for p in m.parameters()]
        opt = torch.optim.Adam(params, lr=hp["lr"])

        def forward(v):
            for i, m in enumerate(mods):
                v = m(v)
                if i < len(mods) - 1:
                    v = torch.tanh(v)
            return OUT_SCALE * torch.tanh(v[:, 0])
        for step in range(hp["steps"]):
            opt.zero_grad()
            value = loss(forward(z)) + hp["l2"] * sum((m.weight ** 2).sum() for m in mods)
            value.backward()
            opt.step()
            if trace is not None and (step + 1) % 250 == 0:
                trace.append((int(seed), step + 1, float(value)))
        nets.append([(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                     for m in mods])
    return mu, sd, nets


def fit_bias_t(arm: str, train: list[DeviceRec], hyper: dict | None = None,
               threads: int = 1, trace: list | None = None) -> TempBiasCorrection:
    """``M_bias`` recipe with the temperature feature as a fourth input
    (unbound result; call ``bind`` before use)."""
    import torch
    torch.set_num_threads(threads)
    hp = {**HYPER_BIAS_T, **(hyper or {})}
    batch = make_batch(train)
    x = np.column_stack([batch.phi[:, list(COLS)], point_t_feature(batch, train)])
    mu, sd, nets = train_mlp(x, _loss_fn(batch), hp, trace=trace)
    return TempBiasCorrection(arm, mu, sd, nets=nets, cols=COLS)


def fit_lin_t(arm: str, train: list[DeviceRec], hyper: dict | None = None,
              threads: int = 1) -> TempLinCorrection:
    """``i4_1_baselines.fit_poly`` (degree 1) with the temperature feature as
    a third standardized term."""
    import torch
    torch.set_num_threads(threads)
    hp = {**HYPER_LIN_T, **(hyper or {})}
    batch = make_batch(train)
    loss = _loss_fn(batch)
    x = np.column_stack([batch.phi[:, 1], batch.phi[:, 2], point_t_feature(batch, train)])
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    f = torch.tensor(lin_features(x[:, 0], x[:, 1], x[:, 2], mu, sd), dtype=torch.float64)
    p = torch.tensor(batch.phi[:, 0], dtype=torch.float64)
    coef = torch.zeros((2, f.shape[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([coef], lr=hp["lr"])
    for _ in range(hp["steps"]):
        opt.zero_grad()
        h = (1.0 - p) * (f @ coef[0]) + p * (f @ coef[1])
        value = loss(OUT_SCALE * torch.tanh(h)) + hp["l2"] * (coef[:, 1:] ** 2).sum()
        value.backward()
        opt.step()
    return TempLinCorrection(arm, mu, sd, coef=coef.detach().numpy().copy())


def merge_temp_seeds(parts: list[TempBiasCorrection]) -> TempBiasCorrection:
    """Join per-seed ``fit_bias_t`` results (same standardization) into one
    ensemble (equal to one ``fit_bias_t`` call with all seeds)."""
    m = parts[0]
    for c in parts[1:]:
        if not (np.array_equal(c.mu, m.mu) and np.array_equal(c.sd, m.sd) and c.cols == m.cols):
            raise RuntimeError("per-seed corrections do not share feature scaling")
    return TempBiasCorrection(m.arm, m.mu, m.sd, nets=[n for c in parts for n in c.nets],
                              cols=m.cols)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
ARMS_TRAINED_OUT_T = ("M_bias_T", "M_bias_pool", "M_bias_near", "lin_T", "lin_pool")


def training_set(recs_by_t: dict, held_unit: str, held_t: int, arm: str, unit_of,
                 eligible=None) -> list[DeviceRec]:
    """Training records of a fold.  ``recs_by_t``: T label -> records;
    ``unit_of(rec)``: held-out unit of a record (IHP DUT tag, TU Delft
    geometry group); ``eligible(rec)``: class filter (IHP: same class).
    Records of ``held_unit`` never enter; the held-out temperature enters
    only for ``M_bias_in``; ``M_bias_near`` uses the nearest temperature."""
    temps = sorted(int(t) for t in recs_by_t)
    if arm == "M_bias_in":
        use = temps
    elif arm == "M_bias_near":
        use = [nearest_temp(held_t, temps)]
    elif arm in ARMS_TRAINED_OUT_T:
        use = [t for t in temps if t != int(held_t)]
    else:
        raise KeyError(arm)
    out = [r for t in use for r in recs_by_t[t]
           if unit_of(r) != held_unit and (eligible is None or eligible(r))]
    check_training(out, held_unit, held_t, arm, unit_of)
    return out


def check_training(train: list[DeviceRec], held_unit: str, held_t: int, arm: str, unit_of) -> None:
    if not train:
        raise RuntimeError(f"{arm}: empty training set")
    if any(unit_of(r) == held_unit for r in train):
        raise RuntimeError(f"{arm}: held-out unit {held_unit} in training")
    if arm != "M_bias_in" and any(int(r.meta["T_K"]) == int(held_t) for r in train):
        raise RuntimeError(f"{arm}: held-out temperature {held_t} in training")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def unit_log_ratios(method: dict, control: dict, temps, unit_of_tag=None) -> tuple[dict, dict]:
    """``method`` / ``control``: {(tag, T): RRMS}.  Per unit (default: the
    tag) the mean of ln(method / control) over the finite positive pairs at
    ``temps``; also counts of pairs and of failed values."""
    temps = {int(t) for t in temps}
    unit = unit_of_tag or (lambda tag: tag)
    acc: dict = {}
    counts = {"pairs": 0, "used": 0, "method_failed": 0, "control_failed": 0, "wins": 0}
    for key in sorted(set(method) & set(control)):
        tag, t = key
        if int(t) not in temps:
            continue
        a, b = float(method[key]), float(control[key])
        counts["pairs"] += 1
        counts["method_failed"] += int(not np.isfinite(a))
        counts["control_failed"] += int(not np.isfinite(b))
        if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
            continue
        counts["used"] += 1
        counts["wins"] += int(a < b)
        acc.setdefault(unit(tag), []).append(math.log(a / b))
    return {u: float(np.mean(v)) for u, v in acc.items()}, counts


def gmr_ci(unit_lr: dict, clusters: dict | None = None, level: float = 0.95,
           n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """exp(mean of the unit log ratios) with a percentile bootstrap CI at
    ``level`` over units, or over clusters (``clusters``: unit -> cluster;
    a resampled cluster contributes all its units; mean = sum / count)."""
    units = sorted(unit_lr)
    lr = np.array([unit_lr[u] for u in units], dtype=float)
    out = {"n_units": len(units), "level": float(level)}
    if not len(lr):
        return {**out, "ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    out["ratio"] = float(np.exp(lr.mean()))
    out["units_lower"] = int(np.sum(lr < 0))
    rng = np.random.default_rng(seed)
    if clusters is None:
        idx = rng.integers(0, len(lr), size=(n_boot, len(lr)))
        boots = lr[idx].mean(axis=1)
        out["resample_unit"] = "unit"
    else:
        names = sorted({clusters[u] for u in units})
        members = [np.array([i for i, u in enumerate(units) if clusters[u] == c]) for c in names]
        sums = np.array([lr[m].sum() for m in members])
        sizes = np.array([len(m) for m in members], dtype=float)
        pick = rng.integers(0, len(names), size=(n_boot, len(names)))
        boots = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
        out["resample_unit"] = f"cluster ({len(names)})"
    q = (1.0 - level) / 2.0
    lo, hi = np.percentile(boots, [100 * q, 100 * (1 - q)])
    out["ci_low"], out["ci_high"] = float(np.exp(lo)), float(np.exp(hi))
    out["n_boot"] = int(n_boot)
    return out


def ci_label(c: dict) -> str:
    """'lower' (CI upper < 1), 'higher' (CI lower > 1), else 'inconclusive'."""
    if not np.isfinite(c.get("ci_high", np.nan)):
        return "undefined"
    if c["ci_high"] < 1.0:
        return "lower"
    if c["ci_low"] > 1.0:
        return "higher"
    return "inconclusive"


def label_ihp(vs_c0: dict, vs_pool: dict, vs_near: dict) -> str:
    """Label of one IHP stratum (interpolation or extrapolation)."""
    l0 = ci_label(vs_c0)
    if l0 == "lower":
        if ci_label(vs_pool) == "lower" and ci_label(vs_near) == "lower":
            return "supported"
        return "transfer without temperature input"
    if l0 == "higher":
        return "loss"
    return "no gain"


def label_tud(vs_c0: dict) -> str:
    return {"lower": "transfer gain", "higher": "transfer loss"}.get(ci_label(vs_c0), "inconclusive")


def decide(ihp_labels: dict, valid: bool) -> dict:
    """Pre-registered rule.  ``ihp_labels``: {'interp': label, 'extrap':
    label}.  TU Delft labels are reported and do not enter the action."""
    if not valid:
        return {"outcome": "invalid", "action": "stop: fix and re-register"}
    labels = set(ihp_labels.values())
    if "supported" in labels:
        return {"outcome": "supported",
                "action": "continue: pre-register the temperature-input correction on a second "
                          "multi-temperature dataset"}
    if "transfer without temperature input" in labels:
        return {"outcome": "transfer without temperature input",
                "action": "continue, reduced: the pooled correction without a temperature input "
                          "is the cross-temperature form; no temperature-input work"}
    return {"outcome": "not supported",
            "action": "stop: no cross-temperature work on the shared bias-only correction"}


# ---------------------------------------------------------------------------
# Training-only statistics (secondary prediction)
# ---------------------------------------------------------------------------
def s1_loto(resids: list, temps: list, units: list) -> float:
    """S1 with the shared function estimated without the scored record's
    temperature and without its unit: 1 - sum_r E_r(m_-r) / sum_r E_r(0)
    over the training records ``resids`` (``i7_1_transfer.DevResid``)."""
    from .i7_1_transfer import N_BINS
    num = den = 0.0
    n = N_BINS * N_BINS
    for i, d in enumerate(resids):
        W = np.zeros(n)
        R = np.zeros(n)
        for j, e in enumerate(resids):
            if e.pmos == d.pmos and temps[j] != temps[i] and units[j] != units[i]:
                W += e.W
                R += e.R
        m = np.divide(R, W, out=np.zeros_like(R), where=W > 0)
        num += d.energy(m)
        den += d.energy()
    return float(1.0 - num / den) if den > 0 else float("nan")


def s3_loto_tasks(train: list[DeviceRec], unit_of, n_parts: int = 2) -> list[tuple]:
    """Inner folds of a training set, leave-one-temperature-out crossed with
    ``n_parts`` interleaved parts of the sorted units: (inner temperature,
    part, inner-training records, scored records).  The inner training set
    holds no record at the inner temperature and no record of a scored unit."""
    temps = sorted({int(r.meta["T_K"]) for r in train})
    units = sorted({unit_of(r) for r in train})
    part = {u: i % n_parts for i, u in enumerate(units)}
    out = []
    for t in temps:
        for k in range(n_parts):
            scored = [r for r in train if int(r.meta["T_K"]) == t and part[unit_of(r)] == k]
            inner = [r for r in train if int(r.meta["T_K"]) != t and part[unit_of(r)] != k]
            if scored and inner:
                out.append((t, k, inner, scored))
    return out


def i7_1_rule(s1: float) -> float:
    return I7_1_RULE["intercept"] + I7_1_RULE["slope"] * float(s1)
