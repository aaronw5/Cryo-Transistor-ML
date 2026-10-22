"""Replica variants: convergence messages (grid fwd/swap x 3 tol x card_rep/elem) and
sense vs replica BSIM4 channel current (tight, card_rep), 18 devices x card sets."""
import sys, json, re
import numpy as np
from pathlib import Path
from multiprocessing import get_context
ROOT = Path("/Users/anrunw/Documents/cryo-ml")
sys.path.insert(0, str(ROOT / "scripts/research/explore"))
import i14_1_chan as drv
from cryoml.explore import i11_1_deploy as m1
from cryoml.explore import i14_1_chan as m
from cryoml import spice_pdk as sp
BASE = dict(m.REPLICA_CHAN)
TOLS = (("default", ""), ("r4", m1.R4), ("tight", m1.TIGHT))
def variant(name):
    r = dict(BASE); route = False; fd = fs = 0.0
    parts = name.split("+")
    for p in parts:
        if p == "route": route = True
        elif p == "rb0": r["rbodymod"] = 0
        elif p.startswith("jd"): fd = float(p[2:])
        elif p.startswith("js"): fs = float(p[2:])
    def v(x): return float(f"{x:.6g}")
    if fd: r.update(jsd=v(0.00275*fd), jswd=v(6e-10*fd))
    if fs: r.update(jss=v(0.00275*fs), jsws=v(6e-10*fs))
    return r, route
def patch_body(body, route):
    if not route: return body
    out = []
    for ln in body.splitlines():
        mm = re.match(r"^E(\w+)b (\S+)_rb 0 (\S+) 0 1$", ln)
        if mm:
            nm, p, b = mm.group(1), mm.group(2), mm.group(3)
            # drain copy node of this instance: E<nm>d <p>_rx 0 <d> 0 1
            d = re.search(rf"^E{nm}d {p}_rx 0 (\S+) 0 1$", body, re.M).group(1)
            out.append(f"E{nm}b {p}_rb {p}_ri {b} {d} 1")
            out.append(f"G{nm}s {p}_ri 0 {p}_rs {p}_rb {m.GMIN:.17g}")
            continue
        if re.match(r"^G\w+c ", ln): continue
        out.append(ln)
    return "\n".join(out) + "\n"
def task(t):
    tag, label, params, name, osdi = t
    rep, route = variant(name)
    dev = drv.devices()[tag]
    pm = dev.dev_type == "pmos"; s = -1.0 if pm else 1.0
    sub = sp._SUBCKT[dev.dev_type]
    res = {"tag": tag, "label": label, "name": name, "msgs": {}}
    with m1.tempdir("x15v_") as td_s:
        td = Path(td_s)
        lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]}, rep)
        for swap in (False, True):
            for tol, opt in TOLS:
                for mode in ("card_rep", "elem"):
                    lab = f"{'swap' if swap else 'fwd'}_{tol}_{mode}"
                    f = td / f"{lab}.dat"
                    body = patch_body(m.device_body(dev, mode, swap=swap), route)
                    sense = mode == "card_rep"
                    ctrl = m.grid_control(dev, f, swap, sense)
                    if sense and tol == "tight":
                        r_ = f"@m.xm1r.m{sub}[id]"
                        ctrl = ["save all " + r_, ctrl[0], ctrl[1] + " " + r_]
                    r = m1.run_deck(m.deck(lab, lib, body, ctrl, Path(osdi) if mode == "elem" else None, options=opt), td, f"{lab}.sp", timeout_s=300)
                    res["msgs"][lab] = r["msgs"]
                    if sense and tol == "tight":
                        rows = m1.read_wrdata(f, 6)
                        if rows.shape[0] == 11175:
                            vd = rows[:, 0]
                            vds_rep = -vd if swap else vd
                            ich = m.drain_channel_current(rows[:, 5], vds_rep, pm)
                            isense = rows[:, 3]
                            fq = (s * vd) > 0
                            d = np.abs(isense - ich)
                            big = np.abs(ich) > 1e-9
                            res[f"{lab}_P2_fwdq_eq"] = float(np.max((d / (1e-6*np.abs(ich) + 1e-13))[fq]))
                            res[f"{lab}_P2_revq_eq"] = float(np.max((d / (1e-6*np.abs(ich) + 1e-13))[~fq]))
                            res[f"{lab}_P2_revq_maxabs"] = float(np.max(d[~fq]))
                            res[f"{lab}_P2_fwdq_maxabs"] = float(np.max(d[fq]))
    return res
if __name__ == "__main__":
    names = sys.argv[1].split(",")
    sets = sys.argv[2].split(",")
    tags = sorted(drv.devices()) if len(sys.argv) < 4 or sys.argv[3] == "all" else sys.argv[3].split(",")
    osdi = str(drv.OUT / "deploy_v2" / "osdi" / "v2_full.osdi")
    sc = drv.start_cards(); m2 = drv.jload(drv.OUT / "deploy_v2" / "cards_M2.json")
    getc = {"published": lambda t: None, "start_C": lambda t: sc[t], "v2_heldout_C": lambda t: drv.heldout_card("v2", t),
            "v1_heldout_C": lambda t: drv.heldout_card("v1", t), "M2_I14": lambda t: m2[t]}
    tasks = [(t, lab, getc[lab](t), n, osdi) for n in names for lab in sets for t in tags]
    with get_context("spawn").Pool(6) as p:
        rows = p.map(task, tasks, chunksize=1)
    summ = {}
    for n in names:
        rr = [r for r in rows if r["name"] == n]
        msgs = [f"{r['label']}|{r['tag']}|{k}" for r in rr for k, v in r["msgs"].items() if v]
        keys = [k for k in rr[0] if "_P2_" in k]
        summ[n] = {"n_decks": sum(len(r["msgs"]) for r in rr), "msgs": msgs,
                   **{k: max(r.get(k, float("nan")) for r in rr) for k in keys}}
    print(json.dumps(summ, indent=1))
    Path(sys.argv[4] if len(sys.argv) > 4 else "/dev/null").write_text(json.dumps(summ, indent=1)) if len(sys.argv) > 4 else None
