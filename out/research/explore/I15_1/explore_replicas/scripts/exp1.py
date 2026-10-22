"""Replica variants on nmos_L8_W1p6 M2 v2 card: convergence messages, tight fwd/swap."""
import sys, json
from pathlib import Path
from multiprocessing import get_context
ROOT = Path("/Users/anrunw/Documents/cryo-ml")
sys.path.insert(0, str(ROOT / "scripts/research/explore"))
import i14_1_chan as drv
from cryoml.explore import i11_1_deploy as m1
from cryoml.explore import i14_1_chan as m
BASE = dict(m.REPLICA_CHAN)
JS = ("jss", "jsd", "jsws", "jswd", "jswgs", "jswgd")
def variant(name):
    r = dict(BASE)
    if name == "i14": pass
    elif name == "js_card":
        for k in JS: r.pop(k)
    elif name.startswith("js_scale"):
        f = float(name.split("_")[2])
        # card values scaled: jss=jsd=0.00275*f, jsws=jswd=6e-10*f, jswgs=jswgd=card? -> set
        r.update(jss=float(f"{0.00275*f:.6g}"), jsd=float(f"{0.00275*f:.6g}"), jsws=float(f"{6e-10*f:.6g}"), jswd=float(f"{6e-10*f:.6g}"), jswgs=0, jswgd=0)
    elif name == "rb0":
        r["rbodymod"] = 0
    elif name == "rb0_js_card":
        for k in JS: r.pop(k)
        r["rbodymod"] = 0
    elif name == "isub_on":
        for k in ("alpha0","alpha1","beta0","lalpha0","walpha0","palpha0","lalpha1","walpha1","palpha1","lbeta0","wbeta0","pbeta0"): r.pop(k)
    return r
def task(t):
    tag, params, name, mode, swap = t
    m.REPLICA_CHAN.clear(); m.REPLICA_CHAN.update(variant(name))
    dev = drv.devices()[tag]
    osdi = drv.arm_osdi(drv.OUT / "deploy_v2", "v2") if mode == "elem" else None
    sense = mode == "card_rep"
    with m1.tempdir("x15_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]})
        f = td / "o.dat"
        r = m1.run_deck(m.deck("x", lib, m.device_body(dev, mode, swap=swap), m.grid_control(dev, f, swap, sense),
                               osdi, options=m1.TIGHT), td, "x.sp", timeout_s=300)
        rows = m1.read_wrdata(f, 4 if sense else 2)
    return (tag, name, mode, swap, r["msgs"], int(rows.shape[0]), round(r["wall_s"], 2))
if __name__ == "__main__":
    cards = drv.jload(drv.OUT / "deploy_v2" / "cards_M2.json")
    names = sys.argv[1].split(",")
    tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ["nmos_L8_W1p6"]
    tasks = [(t, cards[t], n, mo, sw) for t in tags for n in names for mo in ("card_rep", "elem") for sw in (False, True)]
    with get_context("spawn").Pool(4) as p:
        for row in p.imap(task, tasks):
            print(row, flush=True)
