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
s = -1.0 if dev.dev_type == "pmos" else 1.0
def serp(f, sense):
    lines = []
    vecs = "-i(VD)" + (" i(Vm1R)" if sense else "")
    for k, vg in enumerate(m1.GRID_VGS):
        lines.append(f"alter VG dc = {s*vg:.6g}")
        a, b = (m1.GRID_VDS[0], m1.GRID_VDS[-1]) if k % 2 == 0 else (m1.GRID_VDS[-1], m1.GRID_VDS[0])
        st = m1.STEP if k % 2 == 0 else -m1.STEP
        lines += [f"dc VD {a:.6g} {b:.6g} {st:.6g}", f"wrdata {f.as_posix()}_{k}.dat {vecs}"]
    return lines
for swap, mode in ((False, "card_rep"), (True, "elem"), (False, "elem"), (True, "card_rep")):
    f = td / "o"
    sense = mode == "card_rep"
    r = m1.run_deck(m.deck("x", lib, m.device_body(dev, mode, swap=swap), serp(f, sense), osdi if mode == "elem" else None, options=m1.TIGHT), td, "x.sp", timeout_s=300)
    rows = [m1.read_wrdata(Path(f"{f}_{k}.dat"), 4 if sense else 2) for k in range(len(m1.GRID_VGS))]
    g = np.array([np.sort(r_[:, :2], axis=0)[:, 1] if False else r_[np.argsort(r_[:, 0]), 1] for r_ in rows])
    # nested reference
    f2 = td / "n.dat"
    r2 = m1.run_deck(m.deck("y", lib, m.device_body(dev, mode, swap=swap), m.grid_control(dev, f2, swap, sense), osdi if mode == "elem" else None, options=m1.TIGHT), td, "y.sp", timeout_s=300)
    gn = m1.grid_from_rows(m1.read_wrdata(f2, 4 if sense else 2)[:, :2])
    d = np.abs(g - gn) / (1e-6 * np.abs(gn) + 1e-13)
    print(swap, mode, "serp msgs", r["msgs"], "nested msgs", r2["msgs"], "max eq ratio serp vs nested %.3g" % np.nanmax(d), round(r["wall_s"], 2), round(r2["wall_s"], 2))
