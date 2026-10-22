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
m.REPLICA_CHAN["rbodymod"] = 0
def gdirect(body):
    # drop gate and body VCVS; replica gate -> circuit gate, replica body -> circuit body
    out=[]
    eg = dict(re.findall(r"^E(\w+)g (\S+) 0 (\S+) 0 1$", body, re.M) and [(a, b) for a, b, c in re.findall(r"^E(\w+)g (\S+) 0 (\S+) 0 1$", body, re.M)])
    gmap = {n: c for _, n, c in re.findall(r"^E(\w+)g (\S+) 0 (\S+) 0 1$", body, re.M)}
    bmap = {n: c for _, n, c in re.findall(r"^E(\w+)b (\S+) 0 (\S+) 0 1$", body, re.M)} if "B" in sys.argv[1] else {}
    for ln in body.splitlines():
        if re.match(r"^E\w+g ", ln) and gmap: continue
        if re.match(r"^E\w+b ", ln) and bmap: continue
        toks = ln.split(" ")
        toks = [gmap.get(t, bmap.get(t, t)) for t in toks]
        out.append(" ".join(toks))
    return "\n".join(out) + "\n"
for lab in ("elem", "elem_direct"):
    with m1.tempdir("x15c_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {devs[t].dev_type: [(devs[t].bin, cards[t])] for t in tags})
        f = td / "o.dat"
        body = m.ring_body(devs[tags[0]], devs[tags[1]], "elem", m.v2.VA_MODULE_V2)
        if lab == "elem_direct": body = gdirect(body)
        ws = []
        for _ in range(3):
            r = m1.run_deck(m.deck(name, lib, body, m1.ring_control(f, st["tstep"], st["tstop"]) + ["rusage all"], osdi, numdgt=12), td, "c.sp", timeout_s=120)
            ws.append(round(r["wall_s"], 3))
        eqs = [l for l in r["text"].splitlines() if "Circuit Equations" in l or "Transient iterations =" in l]
        print(lab, ws, r["msgs"][:1], eqs)
