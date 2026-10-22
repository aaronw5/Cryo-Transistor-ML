"""ro5_A wall time: card, card_rep, elem with replica variants (M2 v2 cards)."""
import sys, numpy as np
from pathlib import Path
ROOT = Path("/Users/anrunw/Documents/cryo-ml")
sys.path.insert(0, str(ROOT / "scripts/research/explore"))
import i14_1_chan as drv
from cryoml.explore import i11_1_deploy as m1
from cryoml.explore import i14_1_chan as m
BASE = dict(m.REPLICA_CHAN)
cards = drv.jload(drv.OUT / "deploy_v2" / "cards_M2.json")
devs = drv.devices()
name, kind, tags, st = [c for c in drv.CIRCUITS if c[0] == sys.argv[1]][0]
osdi = drv.arm_osdi(drv.OUT / "deploy_v2", "v2")
for var in sys.argv[2].split(","):
    m.REPLICA_CHAN.clear(); m.REPLICA_CHAN.update(BASE)
    if var.startswith("rb0"): m.REPLICA_CHAN["rbodymod"] = 0
    for mode in ("card", "card_rep", "elem"):
        if mode == "card" and var != "i14": continue
        with m1.tempdir("x15c_") as td_s:
            td = Path(td_s)
            lib = m.patched_library(td, {devs[t].dev_type: [(devs[t].bin, cards[t])] for t in tags})
            f = td / "o.dat"
            if kind == "ring":
                body = m.ring_body(devs[tags[0]], devs[tags[1]], mode, m.v2.VA_MODULE_V2)
                ctrl = m1.ring_control(f, st["tstep"], st["tstop"])
            else:
                body = m.vtc_body(devs[tags[0]], devs[tags[1]], mode, m.v2.VA_MODULE_V2); ctrl = m1.vtc_control(f)
            if var.endswith("_gdirect"):
                import re
                body = re.sub(r"^E(\w+)g (\S+)_rg 0 (\S+) 0 1\n", "", body, flags=re.M)
                # replace replica gate node with circuit gate: Xname r line uses <p>rg
                lines = []
                for ln in body.splitlines():
                    mm = re.match(r"^X(\w+)r (\S+) (\S+)_rg (.*)$", ln)
                    if mm: pass
                    lines.append(ln)
                body = "\n".join(lines) + "\n"
            ws = []
            for _ in range(3):
                r = m1.run_deck(m.deck(name, lib, body, ctrl, osdi if mode == "elem" else None, numdgt=12), td, "c.sp", timeout_s=120)
                ws.append(r["wall_s"])
            rows = m1.read_wrdata(f, 2)
            print(var, mode, [round(w, 3) for w in ws], r["msgs"][:1], rows.shape, flush=True)
