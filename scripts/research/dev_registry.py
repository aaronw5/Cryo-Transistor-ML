#!/usr/bin/env python3
"""DEV arm registry: command templates, per-tier registries, queue lines.

Every research node registers its DEV arms in one JSON file under
``out/research/dev/arms.d/`` (``{"arms": [...]}``; each entry in the
``out/research/dev/arms.json`` format plus ``command``, a template with the
fields ``{override}``, ``{budget}``, ``{out}`` and, for drivers with a
``--seed`` option, ``{seed}``; ``seed_default`` is the seed of the base run).
``out/research/dev/arms.json`` itself is never edited.

Subcommands (``PYTHONPATH=src .venv/bin/python scripts/research/dev_registry.py``):

* ``init-iter1``: write ``arms.d/iter1.json``, the 13 iteration-1 arms with
  their ``arms.json`` fields and the command templates (refuses to overwrite
  without ``--force``).
* ``build --tier a``: ``out/research/dev/arms_iter2.json`` = the
  ``arms.json`` entries plus every ``arms.d`` entry, one entry per label.
  Every declaration of a label (``arms.json`` or any ``arms.d`` file) must
  agree on the functional keys (``FUNCTIONAL``: directories, method key,
  mode, budgets, expect, commands, tiers); later declarations only add keys,
  and documentation keys that differ (note, node, ...) are kept under
  ``annotations``.  Fragments written with ``command_template`` ({targets},
  {dir}) and ``{dev_dir}`` in ``dir_template`` are accepted (``fill``
  supplies both field sets); an entry with ``tiers`` is used only in those
  tiers.
* ``build --tier c``: ``out/research/dev_sky130c/arms.json``: the same arms
  with the ``out/research/dev/`` prefix of every ``dir_template`` replaced by
  ``out/research/dev_sky130c/``, targets
  ``out/research/sim_targets_sky130c/{set}.npz``, sets [set_0, set_1] and
  only the ``bank_oracle`` reference (the ``pipeline_r1`` references need a
  separate ``foundation_parameter_study`` run on these targets).
* ``--seeds 1,2`` (build) adds replicate arms ``<arm>__seed<n>`` with
  directories ``out/research/<dev|dev_sky130c>/<arm>__seed<n>/B{budget}/{set}``
  for every arm whose template has ``{seed}``.
* ``check-commands --tier a``: fill each template (top budget, set_0,
  ``seed_default``), run the driver's own argument parsing and option
  construction up to its process pool (the pool is replaced by a stub that
  records the jobs; no fit runs, no simulation), and compare the options with
  ``out/research/dev/<arm>/B<top>/set_0/summary.json`` on every key except
  ``out``.
* ``queue-lines --tier {a,c} --arms L1,... [--budgets ...] --sets 0,1
  [--seeds 1,2]``: print ``<name> | <command>`` lines,
  name ``dev_<tier>_<label>_B<b>_s<k>``; never writes ``out/research/queue.txt``.
* ``plan-sky130c``: the n2a queue plan (calibration, generation, floors,
  floor reference, tier-c arms, finalize/compare) in run order, written to
  ``out/research/dev_sky130c/queue_lines.txt`` for the orchestrator.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

DEV = ROOT / "out" / "research" / "dev"
DEV_C = ROOT / "out" / "research" / "dev_sky130c"
ARMS_D = DEV / "arms.d"
BASE_REGISTRY = DEV / "arms.json"
QUEUE = ROOT / "out" / "research" / "queue.txt"
OVERRIDE_ENV = "CRYOML_TARGET_OVERRIDE"
TIERS = {
    "a": {"registry": DEV / "arms_iter2.json", "prefix": "out/research/dev/",
          "targets_template": "out/research/sim_targets/{set}.npz"},
    "c": {"registry": DEV_C / "arms.json", "prefix": "out/research/dev_sky130c/",
          "targets_template": "out/research/sim_targets_sky130c/{set}.npz"},
}
DEV_SETS = ("set_0", "set_1")

PREFIX = ("{override} .venv/bin/python scripts/research/%s "
          "--round-dir out/cold_loop/full/round_1 "
          "--bank data/processed/cold_loop/full_r1 --set full")
SUFFIX = "--budget {budget} --out {out}"
# label: (driver, flags, seed_default or None when the driver has no --seed)
ITER1 = {
    "hybrid_surrogate": ("hybrid_tr.py", "--model surrogate", None),
    "hybrid_none": ("hybrid_tr.py", "--model none", None),
    "cmaes_trf": ("cmaes_trace.py", "--method cmaes_trf", 20261016),
    "lq_cmaes": ("cmaes_trace.py", "--method lq_cmaes", 20261016),
    "sketch_surrogate_k8": ("sketch_tr.py", "--model surrogate --k 8", 0),
    "sketch_none_k8": ("sketch_tr.py", "--model none --k 8", 0),
    "sketch_surrogate_k4": ("sketch_tr.py", "--model surrogate --k 4", 0),
    "sketch_surrogate_k16": ("sketch_tr.py", "--model surrogate --k 16", 0),
    "sketch_fullfd_surrogate": ("sketch_tr.py", "--model surrogate --refresh-every 0", 0),
    "sketch_fullfd_none": ("sketch_tr.py", "--model none --refresh-every 0", 0),
    "anytime_sur_presearch_equal": (
        "anytime_tr.py", "--model surrogate --presearch on --schedule equal "
        "--restarts off", None),
    "anytime_sur_presearch_race": (
        "anytime_tr.py", "--model surrogate --presearch on --schedule race "
        "--restarts on", None),
    "anytime_none_race": (
        "anytime_tr.py", "--model none --presearch off --schedule race "
        "--restarts on", None),
}


def template(driver: str, flags: str, seed_default) -> str:
    seed = " --seed {seed}" if seed_default is not None else ""
    return f"{PREFIX % driver} {flags}{seed} {SUFFIX}"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(path)


# ------------------------------------------------------------------ arms.d

def init_iter1(force: bool) -> int:
    path = ARMS_D / "iter1.json"
    if path.exists() and not force:
        print(f"{rel(path)} exists (use --force to rewrite)")
        return 1
    base = {a["label"]: a for a in json.loads(BASE_REGISTRY.read_text())["arms"]}
    arms = []
    for label, (driver, flags, seed) in ITER1.items():
        entry = copy.deepcopy(base[label])
        entry["command"] = template(driver, flags, seed)
        if seed is not None:
            entry["seed_default"] = seed
        arms.append(entry)
    write_json(path, {
        "node": "iteration 1 (n1a, n1b, n1c)",
        "description": "Command templates of the 13 iteration-1 DEV arms; the "
                       "other fields equal out/research/dev/arms.json. Fields: "
                       "{override} (CRYOML_TARGET_OVERRIDE=<targets npz>), "
                       "{budget}, {out}, {seed} (drivers with --seed only; "
                       "seed_default = the base run's seed).",
        "arms": arms})
    print(f"wrote {rel(path)} ({len(arms)} arms)")
    return 0


def arms_d_entries() -> list[tuple[Path, dict]]:
    out = []
    for path in sorted(ARMS_D.glob("*.json")):
        data = json.loads(path.read_text())
        for entry in (data["arms"] if isinstance(data, dict) else data):
            out.append((path, entry))
    return out


def seeded(arm: dict, n: int, prefix: str) -> dict:
    label = arm["label"]
    entry = copy.deepcopy(arm)
    entry.update({"label": f"{label}__seed{n}", "replicate_of": label, "seed": n,
                  "dir_template": f"{prefix}{label}__seed{n}/B{{budget}}/{{set}}"})
    entry.pop("seed_default", None)
    return entry


# Keys that decide which runs an arm reads and how they are checked; every
# declaration of a label must agree on them.  Other keys (note, node,
# control, ...) are documentation: the first declaration is kept and later
# differing values are listed under "annotations".
FUNCTIONAL = ("dir_template", "method_key", "mode", "budgets", "expect", "kind",
              "calls_key", "summary_template", "base", "command",
              "command_template", "tiers")


def command_of(arm: dict) -> str | None:
    """``command`` (this tool's fields) or ``command_template`` (fragments
    written with {targets} / {dir}); ``fill`` supplies both field sets."""
    return arm.get("command") or arm.get("command_template")


def normalized(entry: dict) -> dict:
    """An arms.d entry with a tier-dependent ``{dev_dir}`` in its
    ``dir_template`` resolved to tier a (``out/research/dev``)."""
    entry = copy.deepcopy(entry)
    if "{dev_dir}" in entry.get("dir_template", ""):
        entry["dir_template"] = entry["dir_template"].replace(
            "{dev_dir}", TIERS["a"]["prefix"].rstrip("/"))
    return entry


def merged_arms(seeds: list[int]) -> tuple[dict, list[dict], list[str]]:
    """``arms.json`` plus every ``arms.d`` entry, tier-a directories."""
    base = json.loads(BASE_REGISTRY.read_text())
    arms = [copy.deepcopy(a) for a in base["arms"]]
    by_label = {a["label"]: a for a in arms}
    origin = {a["label"]: "out/research/dev/arms.json" for a in arms}
    files = []
    for path, raw in arms_d_entries():
        entry = normalized(raw)
        label = entry["label"]
        files.append(rel(path))
        if label in by_label:
            existing = by_label[label]
            clash = sorted(k for k in FUNCTIONAL if k in entry and k in existing
                           and existing[k] != entry[k])
            if clash:
                raise SystemExit(f"{rel(path)}: {label!r} differs from "
                                 f"{origin[label]} on {clash}")
            docs = {k: v for k, v in entry.items()
                    if k not in FUNCTIONAL and k in existing and existing[k] != v}
            if docs:
                existing.setdefault("annotations", {})[rel(path)] = docs
            existing.update({k: v for k, v in entry.items() if k not in existing})
            existing.setdefault("registered_in", [origin[label]]).append(rel(path))
        else:
            entry["registered_in"] = [rel(path)]
            arms.append(entry)
            by_label[label] = entry
            origin[label] = rel(path)
    for n in seeds:
        for arm in list(arms):
            if (arm.get("mode") != "reference" and "{seed}" in (command_of(arm) or "")
                    and "replicate_of" not in arm
                    and f"{arm['label']}__seed{n}" not in by_label):
                arms.append(seeded(arm, n, TIERS["a"]["prefix"]))
                by_label[arms[-1]["label"]] = arms[-1]
    for arm in arms:
        if arm.get("mode") != "reference" and not arm["dir_template"].startswith(
                TIERS["a"]["prefix"]):
            raise SystemExit(f"{arm['label']}: dir_template outside out/research/dev/")
    return base, arms, sorted(set(files))


def in_tier(arm: dict, tier: str) -> bool:
    return tier in arm.get("tiers", ("a", "c"))


def registry_a(seeds: list[int]) -> dict:
    base, arms, files = merged_arms(seeds)
    return {**{k: v for k, v in base.items() if k != "arms"},
            "description": base.get("description", "") + " Iteration-2 registry: "
            "arms.json plus out/research/dev/arms.d/*.json (scripts/research/"
            "dev_registry.py build --tier a).",
            "arms_d": files, "seeds": seeds,
            "arms": [a for a in arms if in_tier(a, "a")]}


def registry_c(seeds: list[int]) -> dict:
    base, merged, files = merged_arms(seeds)
    prefix_a, prefix_c = TIERS["a"]["prefix"], TIERS["c"]["prefix"]
    arms, omitted = [], []
    for arm in merged:
        arm = copy.deepcopy(arm)
        if arm.get("mode") == "reference":
            if arm.get("kind") == "bank_oracle":
                arms.append(arm)
            else:
                omitted.append(arm["label"])
            continue
        if not in_tier(arm, "c"):
            continue
        arm["dir_template"] = prefix_c + arm["dir_template"][len(prefix_a):]
        arms.append(arm)
    # Fragments that also list their own tier-c entries must agree with the
    # prefix rule (reported, not merged).
    derived = {a["label"]: a["dir_template"] for a in arms if "dir_template" in a}
    for path in sorted(ARMS_D.glob("*.json")):
        data = json.loads(path.read_text())
        for entry in (data.get("tier_c", {}).get("arms", [])
                      if isinstance(data, dict) else []):
            if derived.get(entry["label"], entry.get("dir_template")) != entry.get(
                    "dir_template"):
                raise SystemExit(f"{rel(path)} tier_c {entry['label']!r}: "
                                 f"{entry.get('dir_template')} != {derived[entry['label']]}")
    return {"description": "DEV registry, tier c (sky130c): misspecified SkyWater "
                           "targets paired with tier a (out/research/sim_targets_sky130c, "
                           "scripts/research/make_sky130_misspec.py) on the full "
                           "round-1 43-parameter box. Never add set_2 or real-device "
                           "runs here (TEST).",
            "tier": "sky130c", "sets": list(DEV_SETS), "bank": base["bank"],
            "round_dir": base["round_dir"],
            "targets_template": TIERS["c"]["targets_template"],
            "floors": "out/research/sim_targets_sky130c/floors.json",
            "omitted_references": {label: "needs a separate foundation_parameter_"
                                   "study / polish_cards run on the tier-c targets"
                                   for label in omitted},
            "source": "out/research/dev/arms.json + arms.d (dev_registry.py build "
                      "--tier c)", "arms_d": files, "seeds": seeds, "arms": arms}


def build(tier: str, seeds: list[int]) -> int:
    reg = registry_a(seeds) if tier == "a" else registry_c(seeds)
    path = TIERS[tier]["registry"]
    write_json(path, reg)
    n_arms = sum(1 for a in reg["arms"] if a.get("mode") != "reference")
    refs = [a["label"] for a in reg["arms"] if a.get("mode") == "reference"]
    print(f"wrote {rel(path)}: {n_arms} arms, references {refs}")
    return 0


def load_registry(tier: str) -> dict:
    path = TIERS[tier]["registry"]
    if not path.exists():
        raise SystemExit(f"{rel(path)} missing: run build --tier {tier}")
    return json.loads(path.read_text())


# ---------------------------------------------------------- check-commands

class _Captured(Exception):
    pass


class _StubPool:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, jobs):
        raise _Captured(list(jobs))

    imap = map = imap_unordered


class _StubContext:
    def Pool(self, *args, **kwargs):  # noqa: N802 (multiprocessing API)
        return _StubPool()


def driver_options(command: str, override_path: str) -> dict:
    """The options dict a driver builds from ``command`` (filled template,
    without the override prefix), captured at its process pool."""
    tokens = shlex.split(command)
    script = next(t for t in tokens if t.endswith(".py"))
    argv = tokens[tokens.index(script):]
    module = importlib.import_module(Path(script).stem)
    saved = (sys.argv, getattr(module, "get_context"), os.environ.get(OVERRIDE_ENV))
    sys.argv = argv
    module.get_context = lambda *a, **k: _StubContext()
    os.environ[OVERRIDE_ENV] = override_path
    try:
        module.main()
    except _Captured as captured:
        jobs = captured.args[0]
        return dict(jobs[0][1])
    finally:
        sys.argv, module.get_context = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop(OVERRIDE_ENV, None)
        else:
            os.environ[OVERRIDE_ENV] = saved[2]
    raise RuntimeError(f"{script}: main() returned before its process pool")


def fill(arm: dict, tier: str, set_name: str, budget: int, seed=None,
         out: str | None = None) -> str:
    """Fill an arm's template: {override} (``CRYOML_TARGET_OVERRIDE=<npz>``),
    {targets} (the npz), {budget}, {out} / {dir} (the run directory),
    {seed}, {set}, {dev_dir}."""
    targets = TIERS[tier]["targets_template"].format(set=set_name)
    if seed is None:
        seed = arm.get("seed", arm.get("seed_default"))
    directory = out or arm["dir_template"].format(set=set_name, budget=budget)
    return command_of(arm).format(
        override=f"{OVERRIDE_ENV}={targets}", targets=targets, budget=budget,
        out=directory, dir=directory, seed=seed, set=set_name,
        dev_dir=TIERS[tier]["prefix"].rstrip("/"))


def check_commands(tier: str) -> int:
    reg = load_registry(tier)
    failed = checked = 0
    with tempfile.TemporaryDirectory(prefix="dev_registry_check_") as td:
        for arm in reg["arms"]:
            if arm.get("mode") == "reference" or not command_of(arm):
                continue
            top = max(int(b) for b in arm["budgets"])
            summary = ROOT / arm["dir_template"].format(set="set_0", budget=top) / "summary.json"
            if not summary.exists():
                print(f"[SKIP] {arm['label']}: no {rel(summary)}")
                continue
            stored = json.loads(summary.read_text())["options"]
            out = str(Path(td) / arm["label"])
            command = fill(arm, tier, "set_0", top, out=out)
            override = command.split()[0].split("=", 1)[1]
            try:
                options = driver_options(command.split(" ", 1)[1],
                                         str(ROOT / override))
            except Exception as exc:  # a driver without the pool pattern
                failed += 1
                checked += 1
                print(f"[FAIL] {arm['label']} B{top}: options not captured ({exc!r})")
                continue
            keys = sorted((set(stored) | set(options)) - {"out"})
            diff = {k: (stored.get(k, "<absent>"), options.get(k, "<absent>"))
                    for k in keys if stored.get(k, "<absent>") != options.get(k, "<absent>")}
            checked += 1
            if diff:
                failed += 1
                print(f"[FAIL] {arm['label']} B{top}: {diff}")
            else:
                print(f"[PASS] {arm['label']} B{top}: {len(keys)} option keys equal "
                      f"({rel(summary)})")
    print(f"{checked} checked, {failed} failed")
    return 1 if failed or not checked else 0


# -------------------------------------------------------------- queue lines

def arm_lines(tier: str, labels: list[str], budgets: list[int] | None,
              sets: list[int], seeds: list[int]) -> list[str]:
    if any(k not in (0, 1) for k in sets):
        raise SystemExit("only DEV sets 0 and 1 (set_2 is TEST)")
    reg = load_registry(tier)
    arms = {a["label"]: a for a in reg["arms"]}
    lines = []
    for label in labels:
        arm = arms.get(label)
        if arm is None or arm.get("mode") == "reference" or not command_of(arm):
            raise SystemExit(f"{label!r}: no command template in {rel(TIERS[tier]['registry'])}")
        for b in (budgets or sorted(int(x) for x in arm["budgets"])):
            if b not in [int(x) for x in arm["budgets"]]:
                raise SystemExit(f"{label}: budget {b} not registered "
                                 f"({arm['budgets']}); register it in arms.d first")
            for n in (seeds or [None]):
                if n is not None and "{seed}" not in command_of(arm):
                    raise SystemExit(f"{label}: driver has no --seed option")
                entry = arm if n is None else seeded(arm, n, TIERS[tier]["prefix"])
                for k in sets:
                    name = f"dev_{tier}_{entry['label']}_B{b}_s{k}"
                    lines.append(f"{name} | " + fill(entry, tier, f"set_{k}", b, seed=n))
    return lines


def ints(text: str | None) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()] if text else []


def emit(lines: list[str], write: str | None) -> None:
    for line in lines:
        print(line)
    if write:
        path = Path(write) if Path(write).is_absolute() else ROOT / write
        if path.resolve() == QUEUE.resolve():
            raise SystemExit("refusing to write out/research/queue.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        print(f"# wrote {len(lines)} lines to {rel(path)}", file=sys.stderr)


MAIN_ARMS = ("sketch_surrogate_k8", "sketch_none_k8", "hybrid_surrogate",
             "hybrid_none", "anytime_sur_presearch_race", "anytime_none_race",
             "cmaes_trf", "lq_cmaes")
SCREEN_5000 = ("sketch_surrogate_k4", "sketch_surrogate_k16",
               "sketch_fullfd_surrogate", "sketch_fullfd_none")
CALIB_DEVICES = ("nmos_L0p15_W1p6,nmos_L1_W3,nmos_L100_W100,"
                 "pmos_L0p35_W1p6,pmos_L2_W5,pmos_L8_W0p84")
PY = ".venv/bin/python"
TARGETS_C = "out/research/sim_targets_sky130c"
REF_C = "out/research/dev_sky130c/floor_ref/sketch_none_k8_B20000"


def plan_sky130c() -> list[str]:
    lines = []
    for s in ("0.5", "1", "2"):
        lines.append(
            f"sky130c_calib_s{s} | {PY} scripts/research/make_sky130_misspec.py "
            f"--sets 0 --devices {CALIB_DEVICES} --strength {s} --workers 6 "
            f"--out {TARGETS_C}/calib/s{s} && {PY} scripts/research/sky130c_floor.py "
            f"--calibrate --targets {TARGETS_C}/calib/s{s} --sets 0 --max-nfev 1500 "
            f"--workers 6")
    lines.append(f"sky130c_calib_next | {PY} scripts/research/sky130c_floor.py "
                 f"--calibrate-next --targets {TARGETS_C}/calib --workers 6")
    lines.append(f"sky130c_gen | {PY} scripts/research/make_sky130_misspec.py "
                 f"--sets 0,1 --strength auto --workers 10 --out {TARGETS_C}")
    lines.append(f"sky130c_floor | {PY} scripts/research/sky130c_floor.py "
                 f"--targets {TARGETS_C} --sets 0,1 --max-nfev 3000 --workers 10")
    for k in (0, 1):
        lines.append(
            f"dev_c_floorref_s{k} | {OVERRIDE_ENV}={TARGETS_C}/set_{k}.npz {PY} "
            "scripts/research/sketch_tr.py --round-dir out/cold_loop/full/round_1 "
            "--bank data/processed/cold_loop/full_r1 --set full --model none --k 8 "
            f"--budget 20000 --out {REF_C}/set_{k}")
    lines += arm_lines("c", list(MAIN_ARMS), [500, 1000, 2000, 5000], [0, 1], [])
    lines += arm_lines("c", list(SCREEN_5000), [5000], [0, 1], [])
    lines += arm_lines("c", ["anytime_sur_presearch_equal"], [1000, 5000], [0, 1], [])
    lines += [
        f"sky130c_finalize | {PY} scripts/research/sky130c_floor.py --finalize "
        f"--targets {TARGETS_C} --reference {REF_C}",
        f"dev_registry_build | {PY} scripts/research/dev_registry.py build --tier a "
        f"&& {PY} scripts/research/dev_registry.py build --tier c",
        f"dev_compare_iter2 | {PY} scripts/research/dev_compare.py --registry "
        "out/research/dev/arms_iter2.json --out-dir out/research/dev/compare_iter2",
        f"dev_compare_sky130c | {PY} scripts/research/dev_compare.py --registry "
        "out/research/dev_sky130c/arms.json --out-dir out/research/dev_sky130c/compare",
        f"dev_compare_floor_sky130c | {PY} scripts/research/dev_compare_floor.py "
        "--registry-c out/research/dev_sky130c/arms.json --registry-a "
        "out/research/dev/arms_iter2.json --floors "
        f"{TARGETS_C}/floors.json --out out/research/dev_sky130c/compare_floor"]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init-iter1")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("build")
    p.add_argument("--tier", choices=("a", "c"), required=True)
    p.add_argument("--seeds", default=None)
    p = sub.add_parser("check-commands")
    p.add_argument("--tier", choices=("a", "c"), default="a")
    p = sub.add_parser("queue-lines")
    p.add_argument("--tier", choices=("a", "c"), required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--budgets", default=None)
    p.add_argument("--sets", default="0,1")
    p.add_argument("--seeds", default=None)
    p.add_argument("--write", default=None,
                   help="also write the lines to this file (never queue.txt)")
    p = sub.add_parser("plan-sky130c")
    p.add_argument("--write", default="out/research/dev_sky130c/queue_lines.txt")
    args = parser.parse_args()
    if args.cmd == "init-iter1":
        return init_iter1(args.force)
    if args.cmd == "build":
        return build(args.tier, ints(args.seeds))
    if args.cmd == "check-commands":
        return check_commands(args.tier)
    if args.cmd == "queue-lines":
        emit(arm_lines(args.tier, args.arms.split(","), ints(args.budgets) or None,
                       ints(args.sets), ints(args.seeds)), args.write)
        return 0
    if args.cmd == "plan-sky130c":
        emit(plan_sky130c(), args.write)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
