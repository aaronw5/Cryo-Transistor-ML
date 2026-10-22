import sys
from pathlib import Path
sys.path.insert(0, "/Users/anrunw/Documents/cryo-ml/scripts/research/explore")
import i14_1_chan as drv
from cryoml.explore import i11_1_deploy as m1
from cryoml.explore import i14_1_chan as m
tag = sys.argv[1]; swap = sys.argv[2] == "swap"; mode = sys.argv[3]
dev = drv.devices()[tag]
m2 = drv.jload(drv.OUT / "deploy_v2" / "cards_M2.json")
osdi = drv.arm_osdi(drv.OUT / "deploy_v2", "v2") if mode == "elem" else None
td = Path(sys.argv[4]); td.mkdir(exist_ok=True)
lib = m.patched_library(td, {dev.dev_type: [(dev.bin, m2[tag])]})
s = -1.0 if dev.dev_type == "pmos" else 1.0
ctrl = []
for k, vg in enumerate(m1.GRID_VGS):
    ctrl += [f"alter VG dc = {s*vg:.6g}", f"echo VGPOINT {vg}", f"dc VD {m1.GRID_VDS[0]:.6g} {m1.GRID_VDS[-1]:.6g} {m1.STEP:.6g}"]
r = m1.run_deck(m.deck("x", lib, m.device_body(dev, mode, swap=swap), ctrl, osdi, options=m1.TIGHT), td, "x.sp", timeout_s=300)
for l in r["text"].splitlines():
    if "VGPOINT" in l or "stepping" in l.lower(): print(l)
# single op at first point
ctrl = [f"alter VG dc = 0", f"alter VD dc = {m1.GRID_VDS[0]}", "op", "echo OP1done", f"alter VD dc = 0", "op", "echo OP2done", f"alter VD dc = -1.0", "op", "echo OP3done"]
r = m1.run_deck(m.deck("y", lib, m.device_body(dev, mode, swap=swap), ctrl, osdi, options=m1.TIGHT), td, "y.sp", timeout_s=300)
for l in r["text"].splitlines():
    if "done" in l or "stepping" in l.lower(): print(l)
