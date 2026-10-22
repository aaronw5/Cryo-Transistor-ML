"""Many model cards per NGSpice process, and NGSpice finite-difference Jacobians.

``simulate_pdk`` launches one NGSpice process per card (~10 ms process start,
~5 ms library parse, ~12 ms of DC sweeps, a few ms of Python/file overhead).
``simulate_pdk_multi`` runs K cards in one process.

``method="altermod"`` (default, exact): one circuit; before each card's sweeps
every overridden parameter is set with ``altermod`` on the instance's bin
model.  Each card's curves are bit-identical to separate ``simulate_pdk``
calls.  Three details make that true:

* the value is written as ``<integer mantissa>e<exponent>`` of the same
  ``%.10g`` text ``simulate_pdk`` patches into the card, so NGSpice's
  front-end number parser computes ``mantissa * pow(10, e)`` exactly like the
  netlist parser (``INPevaluate``).  Plain ``%.10g`` text can differ by one
  ulp, which visibly moves sub-``abstol`` off-state currents;
* ``destroy all`` before each card, so a failed sweep cannot pick up the
  previous card's plot (a fresh process has no earlier plot either);
* a one-device check circuit runs a single ``op`` between cards.  BSIM4 checks
  a model's parameters only once per process (a static model-name cache in
  ``BSIM4checkModel``); without the reset a fatal card (e.g. ``delta < 0``)
  would skip the check that aborts it in a fresh process.

Cards whose sweep hangs or makes NGSpice exit are isolated: every card writes
a progress marker, a watchdog kills a process that makes no progress for
``timeout_s``, the stalled card is reported as ``simulate_pdk`` would report
it (all-NaN after a timeout), and the remaining cards restart in a new
process.

``method="parallel"`` (near-exact, ~2x faster again): K renamed copies of the
device, each with its own patched bin and a 0 V sense source, solved by one
``.dc`` sweep.  Newton iterations then stop only when *all* copies have
converged, so each copy gets a few extra iterations: strong-inversion
currents agree with ``simulate_pdk`` to ~1e-7 relative (~1e-8 of the curve
maximum); sub-``abstol`` off-state points differ more.  Any failed or
incomplete sweep makes the whole batch fall back to ``altermod``.

``method="separate"`` loops over ``simulate_pdk`` (the reference).

BSIM4 model-check clamp state (I14-1): ``bsim4_state="clamped"`` (default)
writes the ``b4check.c`` A1/A2 clamp into every card
(``spice_pdk.bsim4_clamp_params``) before any method runs, so every curve of
every card is simulated in the fresh-process state, as
``simulate_pdk(..., bsim4_state="clamped")``.  ``"legacy"`` reproduces the
decks used before I14-1 (altermod: the first curve of each card after the
check reset clamped, later curves with the card values).
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import TEMP_K
from .spice_pdk import (_CORNER_BASENAME, _DEFAULT_TIMEOUT_S, _MODEL,
                        _NGSPICE_BIN, _SUBCKT, LIB_FILE, PDK77K_DIR,
                        _batch_deck, _bin_spans, _instance_line,
                        _patch_params_in_bin, _read_curves, _sweep_lines,
                        bsim4_clamp_params, ensure_pdk77k, find_bin_index,
                        simulate_pdk)

_FATAL_TEXT = "could not find a valid modelname"
_NUM_RE = re.compile(r"([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?")
_CHECK_CIRCUIT = ("* BSIM4 model-check cache reset\n"
                  "Mngchk 0 0 0 0 ngchk_dummy l=1u w=1u\n"
                  ".model ngchk_dummy nmos level=54 version=4.82\n"
                  ".end\n")
_NATIVE_BIN: dict[tuple[str, float, float], int | None] = {}


# ---------------------------------------------------------------------------
# Number literals that NGSpice's two parsers read identically
# ---------------------------------------------------------------------------
def _split_literal(value: float) -> tuple[str, str, int, bool]:
    text = f"{value:.10g}"            # exactly what _patch_params_in_bin writes
    match = _NUM_RE.fullmatch(text)
    if not match or not (match.group(2) or match.group(3)):
        raise ValueError(f"cannot write {value!r} as an NGSpice literal")
    sign = "-" if match.group(1) == "-" else ""
    frac = match.group(3) or ""
    digits = ((match.group(2) or "") + frac).lstrip("0") or "0"
    exponent = (int(match.group(4)) if match.group(4) else 0) - len(frac)
    plain_int = match.group(3) is None and match.group(4) is None
    return sign, digits, exponent, plain_int


def ngspice_literal(value: float) -> str:
    """``%.10g`` of ``value`` rewritten as ``<int>e<exp>`` for ``altermod``."""
    sign, digits, exponent, plain_int = _split_literal(value)
    return sign + digits if plain_int else f"{sign}{digits}e{exponent}"


def ngspice_value(value: float) -> float:
    """The double NGSpice's netlist parser (``INPevaluate``) assigns to the
    ``%.10g`` text of ``value`` — the parameter value actually simulated."""
    sign, digits, exponent, plain_int = _split_literal(value)
    mantissa = float(int(digits)) * (-1.0 if sign else 1.0)
    return mantissa if plain_int else mantissa * math.pow(10.0, exponent)


# ---------------------------------------------------------------------------
# Process runner with a per-card progress watchdog
# ---------------------------------------------------------------------------
def _progress(path: Path) -> list[int]:
    try:
        return [int(t) for t in path.read_text().split()]
    except (OSError, ValueError):
        return []


def _run_watched(deck: Path, cwd: Path, progress: Path,
                 timeout_s: float) -> tuple[bool, str]:
    """Run ``ngspice -b deck``; kill it if ``progress`` stops growing for
    ``timeout_s``.  Returns (killed, lower-cased log)."""
    log_path = cwd / "ngspice.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen([_NGSPICE_BIN, "-b", str(deck)], cwd=str(cwd),
                                stdout=log, stderr=subprocess.STDOUT)
    done, killed = threading.Event(), threading.Event()

    def watch() -> None:
        poll = min(0.25, max(0.01, timeout_s / 20.0))
        last_count, last_time = -1, time.monotonic()
        while not done.wait(poll):
            count = len(_progress(progress))
            now = time.monotonic()
            if count != last_count:
                last_count, last_time = count, now
            elif now - last_time > timeout_s:
                killed.set()
                proc.kill()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    proc.wait()
    done.set()
    watcher.join()
    return killed.is_set(), log_path.read_text(errors="replace").lower()


# ---------------------------------------------------------------------------
# altermod: exact
# ---------------------------------------------------------------------------
def _altermod_once(dev_type: str, L_um: float, W_um: float, curves: list,
                   cards: list[dict], bin_idx: int | None, temp_K: float,
                   timeout_s: float) -> tuple[list, int]:
    """Run ``cards`` (all altering a superset of earlier cards' keys) in one
    process.  Returns (results for the cards that finished or stalled, index
    of the stalled card or len(cards))."""
    model = f"xm1:{_MODEL[dev_type]}.{bin_idx}"
    with tempfile.TemporaryDirectory(prefix="cryoml_pdk_multi_") as td_s:
        td = Path(td_s)
        progress = td / "progress.txt"
        check = td / "modelcheck.sp"
        check.write_text(_CHECK_CIRCUIT)
        head = _batch_deck(dev_type, L_um, W_um, [], LIB_FILE, td, temp_K)
        tail = "quit\n.endc\n.end\n"
        assert head.endswith(tail)
        # After ``source`` the check circuit is circuit 1, the device 2.
        lines = [head[:-len(tail)], f"source {check.as_posix()}\n", "setcirc 2\n"]
        for k, params in enumerate(cards):
            # altermod re-runs CKTtemp when the last sweep ended above 0 V;
            # the check cache still names this model then, so half-updated
            # cards are never parameter-checked.  The cache is reset only
            # once the whole card is in place, right before its sweeps.
            lines += [f"altermod {model} {name} = {ngspice_literal(value)}\n"
                      for name, value in params.items()]
            lines += ["setcirc 1\n", "op\n", "setcirc 2\n", "destroy all\n"]
            lines += _sweep_lines(curves, td, f"c{k}_")
            lines.append(f"echo {k} >> {progress.as_posix()}\n")
        lines.append(tail)
        deck = td / "deck_multi.sp"
        deck.write_text("".join(lines))
        killed, log = _run_watched(deck, td, progress, timeout_s)
        fatal = _FATAL_TEXT in log
        n_done = len(_progress(progress))
        out = [_read_curves(td, f"c{k}_", curves, dev_type, fatal)
               for k in range(n_done)]
        if n_done < len(cards):
            # Timeout -> all NaN (simulate_pdk's TimeoutExpired path); an
            # NGSpice exit/crash keeps whatever that card had written.
            out.append(_read_curves(td, f"c{n_done}_", curves, dev_type,
                                    fatal or killed))
    return out, n_done


def _altermod_segment(dev_type, L_um, W_um, curves, cards, bin_idx, temp_K,
                      timeout_s) -> list:
    results: list = []
    while len(results) < len(cards):
        start = len(results)
        out, _ = _altermod_once(dev_type, L_um, W_um, curves, cards[start:],
                                bin_idx, temp_K, timeout_s)
        results += out          # finished cards + the stalled card, if any
    return results


def _segments(params_list: list) -> list[tuple[str, list[int]]]:
    """Group consecutive cards that can share one altermod process: each
    card must override every key altered before it in the process (altered
    values cannot be restored to the card's own expression values)."""
    segments: list[tuple[str, list[int]]] = []
    altered: set[str] | None = None
    for index, params in enumerate(params_list):
        params = params or {}
        if any(isinstance(v, str) for v in params.values()):
            segments.append(("separate", [index]))
            altered = None
            continue
        keys = {k.lower() for k in params}
        if altered is None or not keys >= altered:
            segments.append(("altermod", []))
            altered = set()
        segments[-1][1].append(index)
        altered |= keys
    return segments


# ---------------------------------------------------------------------------
# parallel: K device copies in one circuit (near-exact)
# ---------------------------------------------------------------------------
def _native_bin(dev_type: str, L_um: float, W_um: float) -> int | None:
    key = (dev_type, round(L_um, 6), round(W_um, 6))
    if key not in _NATIVE_BIN:
        _NATIVE_BIN[key] = find_bin_index(dev_type, L_um, W_um)
    return _NATIVE_BIN[key]


def _copies_corner(dev_type: str, bin_idx: int, cards: list[dict]) -> str:
    base = (PDK77K_DIR / _CORNER_BASENAME[dev_type]).read_text(errors="replace")
    subckt = _SUBCKT[dev_type]
    head = re.search(rf"^\.subckt\s+{re.escape(subckt)}\b", base, re.M | re.I)
    ends = re.search(rf"^\.ends\s+{re.escape(subckt)}\b.*$", base, re.M | re.I)
    body = base[head.start():ends.end()]
    first_model = _bin_spans(body, dev_type)[0][1]
    parts = [base[:head.start()]]
    for k, params in enumerate(cards):
        patched = (_patch_params_in_bin(body, dev_type, bin_idx, params)
                   if params else body)
        span = next(s for s in _bin_spans(patched, dev_type) if s[0] == bin_idx)
        model = re.split(r"^\.ends\b", patched[span[1]:span[2]],
                         flags=re.M | re.I)[0].rstrip("\n") + "\n"
        prelude = re.sub(rf"^\.subckt\s+{re.escape(subckt)}\b",
                         f".subckt {subckt}__c{k}", body[:first_model],
                         count=1, flags=re.M | re.I)
        parts.append(prelude + model + f".ends {subckt}__c{k}\n\n")
    return "".join(parts)


def _parallel(dev_type, L_um, W_um, curves, cards, bin_idx, temp_K,
              timeout_s) -> list | None:
    """All cards in one circuit; None if anything failed (caller falls back)."""
    if not cards or bin_idx is None or bin_idx != _native_bin(dev_type, L_um, W_um):
        return None
    K = len(cards)
    subckt = _SUBCKT[dev_type]
    with tempfile.TemporaryDirectory(prefix="cryoml_pdk_par_") as td_s:
        td = Path(td_s)
        corner = td / _CORNER_BASENAME[dev_type]
        corner.write_text(_copies_corner(dev_type, bin_idx, cards))
        lib = td / "lib77k.spice"
        lib.write_text(LIB_FILE.read_text().replace(
            (PDK77K_DIR / _CORNER_BASENAME[dev_type]).as_posix(),
            corner.as_posix()))
        head = _batch_deck(dev_type, L_um, W_um, [], lib, td, temp_K)
        instance = _instance_line(dev_type, L_um, W_um)
        assert instance.startswith(f"Xm1 nd ng 0 0 {subckt} ")
        rest = instance[len(f"Xm1 nd ng 0 0 {subckt} "):]
        devices = "".join(f"Xm{k} nd{k} ng 0 0 {subckt}__c{k} {rest}"
                          f"VS{k} nd nd{k} DC 0\n" for k in range(K))
        head = head.replace(instance, devices, 1)
        tail = "quit\n.endc\n.end\n"
        lines = [head[:-len(tail)]]
        groups = [list(range(g, min(g + 32, K))) for g in range(0, K, 32)]
        for i, curve in enumerate(curves):
            alter, sweep, _ = _sweep_lines([curve], td, "unused_")
            lines += [alter, sweep]
            for g, members in enumerate(groups):
                vecs = " ".join(f"i(VS{k})" for k in members)
                lines.append(f"wrdata {(td / f'p{i}_{g}.dat').as_posix()} {vecs}\n")
        lines.append(tail)
        deck = td / "deck_parallel.sp"
        deck.write_text("".join(lines))
        try:
            proc = subprocess.run([_NGSPICE_BIN, "-b", str(deck)], cwd=str(td),
                                  capture_output=True, text=True, check=False,
                                  timeout=timeout_s * max(1.0, K / 4.0))
        except subprocess.TimeoutExpired:
            return None
        log = (proc.stdout + proc.stderr).lower()
        if proc.returncode or _FATAL_TEXT in log or "aborted" in log:
            return None
        out = [[None] * len(curves) for _ in range(K)]
        for i, curve in enumerate(curves):
            target = np.asarray(curve.Vg if curve.kind == "idvg" else curve.Vd,
                                dtype=np.float64)
            for g, members in enumerate(groups):
                path = td / f"p{i}_{g}.dat"
                if not path.exists():
                    return None
                flat = np.array(path.read_text().split(), dtype=np.float64)
                width = 2 * len(members)
                if flat.size == 0 or flat.size % width:
                    return None
                table = flat.reshape(-1, width)
                sweep_v = table[:, 0]
                span = max(abs(float(target.max()) - float(target.min())), 1e-12)
                if (not np.all(np.isfinite(table))
                        or sweep_v.min() > target.min() + 1e-9 * span
                        or sweep_v.max() < target.max() - 1e-9 * span):
                    return None
                order = np.argsort(sweep_v)
                for j, k in enumerate(members):
                    values = np.interp(target, sweep_v[order],
                                       table[order, 2 * j + 1])
                    out[k][i] = -values if dev_type == "pmos" else values
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def simulate_pdk_multi(
    dev_type: str,
    L_um: float,
    W_um: float,
    curves: Iterable,
    params_list: Sequence[dict[str, float] | None],
    bin_index: int | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    *,
    method: str = "altermod",
    temp_K: float = TEMP_K,
    workers: int = 1,
    bsim4_state: str = "clamped",
) -> list[list[np.ndarray]]:
    """``[simulate_pdk(..., params=p, bin_index=bin_index,
    bsim4_state=bsim4_state) for p in params_list]`` with many cards per
    NGSpice process.

    ``method``: ``"altermod"`` (bit-identical, default), ``"parallel"``
    (near-exact, faster; falls back to altermod on any failure) or
    ``"separate"`` (one process per card).  ``timeout_s`` is per card.
    ``workers`` > 1 splits the cards into contiguous chunks run as concurrent
    NGSpice processes (threads; each chunk is independent).
    ``bsim4_state``: "clamped" (default; the b4check.c clamp written into
    every card) or "legacy" (no clamp written; reproduction only).
    """
    err = ensure_pdk77k()
    if err:
        raise RuntimeError(err)
    if method not in ("altermod", "parallel", "separate"):
        raise ValueError(f"unknown method {method!r}")
    if bsim4_state not in ("clamped", "legacy"):
        raise ValueError(f"bsim4_state {bsim4_state!r} not in ('clamped', 'legacy')")
    curves = list(curves)
    params_list = [dict(p) if p else None for p in params_list]
    if not params_list:
        return []
    needs_bin = any(params_list)
    bin_idx = bin_index
    if needs_bin and bin_idx is None:
        bin_idx = find_bin_index(dev_type, L_um, W_um)
        if bin_idx is None:
            raise ValueError(f"no bin for {dev_type} L={L_um} W={W_um}")
    if bsim4_state == "clamped":
        params_list = [bsim4_clamp_params(dev_type, bin_idx, p)[0] if p else None
                       for p in params_list]

    def separate(index: int) -> list[np.ndarray]:
        return simulate_pdk(dev_type, L_um, W_um, curves,
                            params=params_list[index], temp_K=temp_K,
                            timeout_s=timeout_s, bin_index=bin_idx,
                            bsim4_state=bsim4_state)

    def run_chunk(indices: list[int]) -> list[list[np.ndarray]]:
        chunk = [params_list[i] for i in indices]
        if method == "separate":
            return [separate(i) for i in indices]
        if method == "parallel" and all(
                p and not any(isinstance(v, str) for v in p.values())
                for p in chunk):
            got = _parallel(dev_type, L_um, W_um, curves, chunk, bin_idx,
                            temp_K, timeout_s)
            if got is not None:
                return got
        results: dict[int, list] = {}
        for kind, members in _segments(chunk):
            if kind == "separate":
                results[members[0]] = separate(indices[members[0]])
                continue
            out = _altermod_segment(dev_type, L_um, W_um, curves,
                                    [chunk[m] or {} for m in members],
                                    bin_idx, temp_K, timeout_s)
            results.update(zip(members, out))
        return [results[m] for m in range(len(chunk))]

    order = list(range(len(params_list)))
    workers = max(1, min(int(workers), len(order)))
    if workers == 1:
        return run_chunk(order)
    chunks = [list(c) for c in np.array_split(order, workers) if len(c)]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        parts = list(pool.map(run_chunk, chunks))
    return [row for part in parts for row in part]


def ngspice_jacobian(
    dev,
    curves: Iterable,
    params: dict[str, float],
    names: Sequence[str],
    rel_step: float = 1e-2,
    *,
    scheme: str = "forward",
    abs_step: float | None = None,
    bin_index: int | None = None,
    method: str = "altermod",
    workers: int = 1,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    temp_K: float = TEMP_K,
    full_output: bool = False,
    bsim4_state: str = "clamped",
):
    """Finite-difference Jacobian d(I-V points)/d(card parameters) via NGSpice.

    ``dev`` has ``dev_type``, ``L_um``, ``W_um``; ``params`` is the full card
    override (every name in ``names`` must be a key).  Rows are the
    ``simulate_pdk`` points of ``curves`` concatenated in order (the same
    sign convention: positive drain current); columns follow ``names``.

    Parameter j moves by ``h_j = rel_step * |p_j|`` (``abs_step``, default
    ``rel_step``, when ``p_j == 0``).  The divisor is the difference of the
    values NGSpice actually parses (``%.10g`` text), so card rounding never
    biases a column.  ``scheme="forward"`` simulates 1 + n cards,
    ``"central"`` 1 + 2n (the base card is always run to return ``f0``).

    Returns ``J`` (n_points x n_names), or ``(J, f0, steps)`` with
    ``full_output=True``.  Failed simulations give NaN entries.
    ``bsim4_state`` as ``simulate_pdk_multi`` (default "clamped": a column of
    a parameter that the clamp overrides, e.g. a2 > 1, is 0).
    """
    if scheme not in ("forward", "central"):
        raise ValueError(f"unknown scheme {scheme!r}")
    curves = list(curves)
    names = list(names)
    missing = [n for n in names if n not in params]
    if missing:
        raise KeyError(f"names without a base value in params: {missing}")
    base = {k: float(v) for k, v in params.items()}
    zero_step = rel_step if abs_step is None else abs_step
    cards = [base]
    plus, minus, steps = [], [], []
    for name in names:
        value = base[name]
        h = rel_step * abs(value) if value != 0.0 else zero_step
        up = {**base, name: value + h}
        cards.append(up)
        plus.append(ngspice_value(value + h))
        if scheme == "central":
            down = {**base, name: value - h}
            cards.append(down)
            minus.append(ngspice_value(value - h))
        else:
            minus.append(ngspice_value(value))
    for p_hi, p_lo, name in zip(plus, minus, names):
        if not p_hi > p_lo:
            raise ValueError(f"step for {name} vanishes at 10 significant digits")
        steps.append(p_hi - p_lo)
    sims = simulate_pdk_multi(dev.dev_type, dev.L_um, dev.W_um, curves, cards,
                              bin_index=bin_index, timeout_s=timeout_s,
                              method=method, temp_K=temp_K, workers=workers,
                              bsim4_state=bsim4_state)
    flat = [np.concatenate([np.asarray(c, dtype=np.float64) for c in s])
            for s in sims]
    f0 = flat[0]
    jac = np.empty((f0.size, len(names)), dtype=np.float64)
    for j, step in enumerate(steps):
        if scheme == "central":
            jac[:, j] = (flat[1 + 2 * j] - flat[2 + 2 * j]) / step
        else:
            jac[:, j] = (flat[1 + j] - f0) / step
    if full_output:
        return jac, f0, np.asarray(steps)
    return jac
