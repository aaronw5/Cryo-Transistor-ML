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
# trivial module: g = 0 constant, same ports
va = """`include "disciplines.vams"
module i12_1_bias_corr_v2(d, g, s, rx, ri);
  inout d, g, s, rx, ri; electrical d, g, s, rx, ri;
  branch (rx, ri) sense;
  parameter real is_pmos = 0; parameter real RS = 0.0;
  real ib;
  analog begin
    V(sense) <+ RS * I(sense);
    ib = I(sense);
    I(d, s) <+ (exp(1e-3 * V(g, s)) - 1.0) * ib;
  end
endmodule
"""
td0 = Path(sys.argv[1]); td0.mkdir(exist_ok=True)

m.REPLICA_CHAN["rbodymod"] = 0
for lab, os_ in (("elem_v2", osdi), ("elem_exp", td0 / "v2_exp.osdi")):
    with m1.tempdir("x15c_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {devs[t].dev_type: [(devs[t].bin, cards[t])] for t in tags})
        f = td / "o.dat"
        body = m.ring_body(devs[tags[0]], devs[tags[1]], "elem", m.v2.VA_MODULE_V2)
        ws = []
        for _ in range(3):
            r = m1.run_deck(m.deck(name, lib, body, m1.ring_control(f, st["tstep"], st["tstop"]), os_, numdgt=12), td, "c.sp", timeout_s=120)
            ws.append(round(r["wall_s"], 3))
        print(lab, ws, r["msgs"][:1])
