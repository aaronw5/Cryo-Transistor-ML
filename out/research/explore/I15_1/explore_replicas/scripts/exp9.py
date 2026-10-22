import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/Users/anrunw/Documents/cryo-ml/scripts/research/explore")
import i14_1_chan as drv
from cryoml.explore import i11_1_deploy as m1
from cryoml.explore import i14_1_chan as m
tag = sys.argv[1]
dev = drv.devices()[tag]
m2 = drv.jload(drv.OUT / "deploy_v2" / "cards_M2.json")
td = Path(sys.argv[2]); td.mkdir(exist_ok=True)
lib = m.patched_library(td, {dev.dev_type: [(dev.bin, m2[tag])]})
osdi = drv.arm_osdi(drv.OUT / "deploy_v2", "v2")
for extra in ("", "itl2=100", "itl2=200", "itl2=500", "itl1=500 itl2=500"):
    for swap, mode in ((False, "card_rep"), (True, "elem"), (False, "elem"), (True, "card_rep")):
        f = td / "o.dat"
        sense = mode == "card_rep"
        r = m1.run_deck(m.deck("x", lib, m.device_body(dev, mode, swap=swap), m.grid_control(dev, f, swap, sense),
                               osdi if mode == "elem" else None, options=m1.TIGHT + (" " + extra if extra else "")), td, "x.sp", timeout_s=300)
        rows = m1.read_wrdata(f, 4 if sense else 2)
        if extra == "": base = rows
        print(extra or "default-itl", swap, mode, r["msgs"], rows.shape, round(r["wall_s"], 2))
