import sys, re
from pathlib import Path
ROOT = Path("/Users/anrunw/Documents/cryo-ml")
sys.path.insert(0, str(ROOT / "scripts/research/explore"))
import i14_1_chan as drv
from cryoml.explore import i11_1_deploy as m1
from cryoml.explore import i14_1_chan as m
cards = drv.jload(drv.OUT / "deploy_v2" / "cards_M2.json")
devs = drv.devices()
name, kind, tags, st = [c for c in drv.CIRCUITS if c[0] == "ro5_A"][0]
osdi = drv.arm_osdi(drv.OUT / "deploy_v2", "v2")
for mode in ("card", "card_rep", "elem"):
    with m1.tempdir("x15c_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {devs[t].dev_type: [(devs[t].bin, cards[t])] for t in tags})
        f = td / "o.dat"
        body = m.ring_body(devs[tags[0]], devs[tags[1]], mode, m.v2.VA_MODULE_V2)
        ctrl = m1.ring_control(f, st["tstep"], st["tstop"]) + ["rusage all"]
        r = m1.run_deck(m.deck(name, lib, body, ctrl, osdi if mode == "elem" else None, numdgt=12), td, "c.sp", timeout_s=120)
        keep = [l for l in r["text"].splitlines() if re.search(r"(?i)(tran.*(iter|points|time)|load time|matrix|equations|total analysis|reordering|lu)", l)]
        print(mode, round(r["wall_s"],3)); print("\n".join(keep[:25]))
