"""I10-2: selection of the function class of the shared bias-only correction
from training data only.

The I9-2 classes of ``g`` (``I_corr = I_card * exp(g)``) plus ``none`` (no
correction; the held-out score is the stored ``C0_refit`` row) form a class
menu.  In every outer fold of the stored I9-2 runs a selector chooses one class
from the fold's training devices only; the held-out score of the fold's
devices is the stored I9-2 row of the chosen class (the class was trained on
the same fold's training devices and the held-out card refit with it frozen).
Pre-registration: ``out/research/explore/I10_2/prereg.json``.

Selectors (``SELECTORS``):

* ``S_loss``: inner folds on the outer fold's training devices
  (``inner_folds``); every menu class is trained on the inner-training
  devices (MLP classes: member seed 0 only) and scored post hoc on the inner
  held-out devices' starting-card records, ln(RRMS of I_sim x exp(g) / RRMS of
  I_sim) (``i7_1_transfer.s3_fold_value``); no refit.
* ``S_refit``: the same inner corrections; the inner held-out cards are refit
  with ``g`` frozen (I9-2 refit path) and scored by ln(RRMS_refit /
  RRMS_C0_refit), C0_refit being the stored uncorrected refit of the device.
* ``S_gate``: the I8-2 gate at fold level: ``mlp16`` if the stored S3_f of the
  fold is < 0, else ``none``.
* ``S_cv``: the stored outer rows of the fold's training devices, ln(RRMS_X /
  RRMS_C0_refit); no new computation.  The rows of a training device d were
  produced with a ``g`` whose training set contained the outer fold's own
  devices (their starting-card records), so S_cv is not strictly nested.

Selection (``select_class``): the class with the lowest mean inner log ratio;
``none`` scores 0; classes within ``TIE`` of the lowest value count as tied
and the class with fewer trained values is chosen.  A class with a non-finite
inner value on a device whose reference value is finite is not selectable in
that fold.
"""

from __future__ import annotations

import math

import numpy as np

from .i9_2_classes import CLASS_SPECS, REPLICATE, n_params, pooled_ratio

NONE = "none"
MENU = (NONE, "lin", "gam", "mlp8", "mlp16")
FULL_MENU = (NONE,) + tuple(c for c in CLASS_SPECS if c != REPLICATE)
SELECTORS = ("S_loss", "S_refit", "S_gate", "S_cv")
COMPUTE_ORDER = ("S_gate", "S_cv", "S_loss", "S_refit")   # cheapest first (tie-break)
MULTI_CLASS = ("S_loss", "S_refit", "S_cv")
REFERENCE = "mlp16"
CONTROL = "C0_refit"
TIE = 0.005                  # tie tolerance on the mean inner log ratio
INNER_K = 4                  # grouped inner folds when the training set has many groups
INNER_LOGO_MAX = 12          # inner leave-one-group-out up to this many training groups
INNER_SEEDS = (0,)           # MLP classes in the inner loop: member seed 0 only
MARGIN = 1.03
ALPHA = 0.05
N_PRIMARY_CONTRASTS = 11     # 4 x GMR(S, C0_refit) + 4 x GMR(S, mlp16) + 3 x GMR(S, S_gate)
SELECT_TIE = 0.005           # overall rule: pooled-GMR tie tolerance between selectors


def class_size(name: str) -> int:
    """Trained values of the exported g (``none``: 0)."""
    return 0 if name == NONE else n_params(name)


def row_key(name: str) -> str:
    """Name of the stored per-device row of a menu class."""
    return CONTROL if name == NONE else name


def bonferroni_alpha(n: int = N_PRIMARY_CONTRASTS, alpha: float = ALPHA) -> float:
    return alpha / n


# ---------------------------------------------------------------------------
# Inner folds
# ---------------------------------------------------------------------------
def inner_folds(train: list[str], group: dict, family: dict, k: int = INNER_K,
                logo_max: int = INNER_LOGO_MAX) -> list[list[str]]:
    """Inner folds of an outer fold's training devices ``train`` (stored
    training order kept inside each fold).

    Groups (``group[tag]``: the device itself, or the TU Delft geometry group)
    are sorted by (family of the group's first device, group name).  With at
    most ``logo_max`` groups every group is one inner fold (leave-one-group-
    out); otherwise group i goes to inner fold i mod ``k``."""
    first: dict = {}
    for t in train:
        first.setdefault(group[t], t)
    gs = sorted(first, key=lambda g: (family[first[g]], g))
    parts = [[g] for g in gs] if len(gs) <= logo_max else [gs[i::k] for i in range(k)]
    return [[t for t in train if group[t] in set(p)] for p in parts]


def refit_inner_index(fold_pos: int, n_inner: int) -> int:
    """Inner fold whose held-out devices get S_refit refits when only one
    inner fold is refit (NIST 4 K): outer fold position mod number of inner folds."""
    return fold_pos % n_inner


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def fold_scores(logs: dict, tags: list[str], reference_ok: dict | None = None) -> tuple[dict, dict]:
    """Mean inner log ratio per class over ``tags`` and the classes that are
    not selectable.

    ``logs``: {class: {tag: ln ratio (NaN if the class value is non-finite)}};
    ``reference_ok``: {tag: bool} whether the reference value (starting record
    or C0_refit) of the tag is finite (default: all True).  A class with a
    non-finite value on a tag with a finite reference is not selectable; tags
    with a non-finite reference are ignored."""
    scores, blocked = {}, {}
    for c, d in logs.items():
        vals, bad = [], 0
        for t in tags:
            if reference_ok is not None and not reference_ok.get(t, True):
                continue
            v = d.get(t, float("nan"))
            if np.isfinite(v):
                vals.append(v)
            else:
                bad += 1
        scores[c] = float(np.mean(vals)) if vals else float("nan")
        if bad:
            blocked[c] = bad
    return scores, blocked


def select_class(scores: dict, blocked=(), tie: float = TIE) -> str:
    """Class with the lowest score; ``none`` scores 0 and is always
    selectable; scores within ``tie`` of the lowest are tied and the class
    with fewer trained values wins (then the lower score, then the name)."""
    cand = {c: float(v) for c, v in scores.items()
            if c != NONE and c not in set(blocked) and np.isfinite(v)}
    cand[NONE] = 0.0
    best = min(cand.values())
    tied = [c for c, v in cand.items() if v <= best + tie]
    return min(tied, key=lambda c: (class_size(c), cand[c], c))


def gate_select(s3_f: float) -> str:
    """I8-2 gate at fold level: mlp16 iff S3_f < 0."""
    return REFERENCE if (np.isfinite(s3_f) and s3_f < 0) else NONE


def cv_logs(rows: dict, menu, tags: list[str]) -> tuple[dict, dict]:
    """S_cv inputs from stored outer rows: {class: {tag: ln(RRMS_X / RRMS_C0)}}
    and {tag: C0_refit finite}."""
    c0 = rows[CONTROL]
    ok = {t: bool(np.isfinite(c0.get(t, np.nan)) and c0.get(t, 0) > 0) for t in tags}
    logs = {}
    for c in menu:
        if c == NONE:
            continue
        logs[c] = {}
        for t in tags:
            a = rows[row_key(c)].get(t, np.nan)
            logs[c][t] = float(math.log(a / c0[t])) if ok[t] and np.isfinite(a) and a > 0 else float("nan")
    return logs, ok


def assemble(selection: dict, rows: dict, fold_of: dict, extra_nan=()) -> dict:
    """Per-device held-out RRMS of a selector: device t gets the stored row of
    the class selected in its outer fold (``none`` -> C0_refit); tags in
    ``extra_nan`` (non-finite starting records) are NaN."""
    out = {}
    for t, f in fold_of.items():
        out[t] = float(rows[row_key(selection[f])].get(t, np.nan))
    for t in extra_nan:
        out[t] = float("nan")
    return out


def selection_counts(selection: dict, menu=MENU) -> dict:
    out = {c: 0 for c in menu}
    for c in selection.values():
        out[c] = out.get(c, 0) + 1
    return out


def agreement(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    same = sum(a[k] == b[k] for k in keys)
    return {"n": len(keys), "same": int(same), "fraction": same / len(keys) if keys else float("nan")}


# ---------------------------------------------------------------------------
# Statistics and the rule
# ---------------------------------------------------------------------------
def gmr(data: dict, clusters: dict | None = None, alpha: float = ALPHA) -> dict:
    """``pooled_ratio`` (equal dataset weight; stratified device or cluster
    bootstrap, 20000 resamples, default_rng(0))."""
    return pooled_ratio(data, clusters=clusters, alpha=alpha)


def selector_label(per_ds_vs_c0: dict, pooled_vs_ref: dict, margin: float = MARGIN) -> dict:
    """Criteria of one selector.

    C1 (not worse than C0_refit): on every primary dataset the upper bound
    of the (Bonferroni) CI of GMR(S, C0_refit) is <= ``margin``.
    C2 (close to M_bias): the upper bound of the pooled (Bonferroni) CI of
    GMR(S, mlp16) is <= ``margin``.
    'supported' if C1 and C2, 'partly supported' if one holds, else 'not
    supported'.  'beats mlp16' if the pooled upper bound is < 1."""
    c1 = all(np.isfinite(d["ci_high"]) and d["ci_high"] <= margin for d in per_ds_vs_c0.values())
    c2 = bool(np.isfinite(pooled_vs_ref["ci_high"]) and pooled_vs_ref["ci_high"] <= margin)
    label = "supported" if (c1 and c2) else ("partly supported" if (c1 or c2) else "not supported")
    return {"C1_not_worse_than_C0_refit_all_datasets": bool(c1),
            "C1_failing_datasets": [n for n, d in per_ds_vs_c0.items()
                                    if not (np.isfinite(d["ci_high"]) and d["ci_high"] <= margin)],
            "C2_noninferior_to_mlp16_pooled": c2,
            "beats_mlp16_pooled": bool(pooled_vs_ref["ci_high"] < 1.0),
            "label": label}


def overall_decision(labels: dict, pooled_vs_c0: dict, vs_gate: dict,
                     tie: float = SELECT_TIE) -> dict:
    """Overall rule.

    ``labels``: {selector: selector_label(...)}; ``pooled_vs_c0``:
    {selector: pooled GMR(S, C0_refit)}; ``vs_gate``: {multi-class selector:
    pooled Bonferroni GMR(S, S_gate)}.

    * 'stop' if no selector is 'supported'.
    * 'continue: class selection adds to the gate' if a supported multi-class
      selector beats S_gate (upper bound of GMR(S, S_gate) < 1); the frozen
      TEST selector is the one of those with the lowest pooled GMR(S, S_gate).
    * else 'continue: gate only' if S_gate is supported (frozen: S_gate).
    * else 'continue: multi-class selector' with the supported selector of the
      lowest pooled GMR(S, C0_refit) (ties within ``tie``: cheaper selector,
      ``COMPUTE_ORDER``)."""
    sup = [s for s in labels if labels[s]["label"] == "supported"]
    if not sup:
        return {"decision": "stop", "frozen_selector": None, "supported": []}
    beat = [s for s in sup if s in vs_gate and vs_gate[s]["ci_high"] < 1.0]
    if beat:
        pick = min(beat, key=lambda s: (vs_gate[s]["ratio"], COMPUTE_ORDER.index(s)))
        return {"decision": "continue: class selection adds to the gate", "frozen_selector": pick,
                "supported": sup, "beats_S_gate": beat}
    if "S_gate" in sup:
        return {"decision": "continue: gate only", "frozen_selector": "S_gate", "supported": sup,
                "beats_S_gate": []}
    best = min(pooled_vs_c0[s]["ratio"] for s in sup)
    tied = [s for s in sup if pooled_vs_c0[s]["ratio"] <= best + tie]
    pick = min(tied, key=COMPUTE_ORDER.index)
    return {"decision": "continue: multi-class selector", "frozen_selector": pick, "supported": sup,
            "beats_S_gate": []}


def oracle_classes(per_ds_vs_c0: dict, pooled_vs_c0: dict) -> dict:
    """Post hoc references: the class with the lowest pooled GMR(X, C0_refit)
    over the menu (``none`` = 1), and per dataset the class with the lowest
    GMR(X, C0_refit)."""
    pooled = {c: v["ratio"] for c, v in pooled_vs_c0.items()}
    pooled[NONE] = 1.0
    per = {}
    for n, d in per_ds_vs_c0.items():
        r = {c: v for c, v in d.items()}
        r[NONE] = 1.0
        per[n] = min(r, key=lambda c: (r[c], class_size(c)))
    return {"pooled": min(pooled, key=lambda c: (pooled[c], class_size(c))), "per_dataset": per}
