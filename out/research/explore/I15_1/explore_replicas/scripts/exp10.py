import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "/Users/anrunw/Documents/cryo-ml/scripts/research/explore")
import i14_1_chan as drv
from cryoml.explore import i11_1_deploy as m1
from cryoml.explore import i14_1_chan as m
tag = "nmos_L8_W1p6"
dev = drv.devices()[tag]
m2 = drv.jload(drv.OUT / "deploy_v2" / "cards_M2.json")
td = Path(sys.argv[1]); td.mkdir(exist_ok=True)
lib = m.patched_library(td, {dev.dev_type: [(dev.bin, m2[tag])]})
for ctrl in (["alter VG dc = 1.85", "alter VD dc = -1.85", "op", "echo A"],
             ["alter VG dc = 1.85", "dc VD 1.85 -1.85 -3.7", "echo B"],
             ["alter VG dc = 1.85", "dc VD -1.85 1.85 0.025", "echo C"],
             ["alter VG dc = 1.85", "dc VD 1.85 -1.85 -0.025", "echo D"],
             ["alter VG dc = 1.85", "dc VD -1.8 1.85 0.025", "echo E"],
             ["alter VG dc = 1.8", "dc VD -1.85 1.85 0.025", "echo F"]):
    r = m1.run_deck(m.deck("x", lib, m.device_body(dev, "card_rep"), ctrl, None, options=m1.TIGHT), td, "x.sp", timeout_s=300)
    print(ctrl[-1], r["msgs"])
