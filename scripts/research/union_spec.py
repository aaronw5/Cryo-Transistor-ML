#!/usr/bin/env python3
"""Union of several round specs (per device, per parameter: min lo, max hi),
log-scaled and centered geometrically -- the search box of a model trained on
the pooled banks of those rounds.

  PYTHONPATH=src .venv/bin/python scripts/research/union_spec.py OUT SPEC [SPEC...]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    out, paths = Path(sys.argv[1]), sys.argv[2:]
    specs = [json.loads(Path(p).read_text()) for p in paths]
    names = specs[0]["selected_parameters"]
    union = {"schema_version": 1, "kind": "cold_start_union_spec",
             "selected_parameters": names, "sources": paths,
             "published_by_device": {}, "bounds_by_device": {},
             "log_scale_by_device": {}}
    for tag in specs[0]["bounds_by_device"]:
        pub, bounds, logs = {}, {}, {}
        for name in names:
            lo = min(s["bounds_by_device"][tag][name][0] for s in specs)
            hi = max(s["bounds_by_device"][tag][name][1] for s in specs)
            same_sign = lo * hi > 0
            bounds[name] = [lo, hi]
            logs[name] = bool(same_sign and lo != hi)
            pub[name] = (math.copysign(math.sqrt(lo * hi), lo) if same_sign
                         else 0.5 * (lo + hi))
        union["published_by_device"][tag] = pub
        union["bounds_by_device"][tag] = bounds
        union["log_scale_by_device"][tag] = logs
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(union, indent=1) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
