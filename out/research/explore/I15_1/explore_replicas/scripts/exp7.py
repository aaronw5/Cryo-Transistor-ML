"""Channel replica with junctions + junction replica (u0 ~ 0) subtracted by a CCCS."""
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
JS = ("jss", "jsd", "jsws", "jswd", "jswgs", "jswgd")
TOLS = (("default", ""), ("r4", m1.R4), ("tight", m1.TIGHT))
def reps(name):
    r = dict(BASE)
    if name.startswith("card"):
        for k in JS: r.pop(k)
    else:
        f = float(name[1:])
        v = lambda x: float(f"{x:.6g}")
        r.update(jss=v(0.00275*f), jsd=v(0.00275*f), jsws=v(6e-10*f), jswd=v(6e-10*f))
    j = {**r, "u0": 1e-30, "lu0": 0, "wu0": 0, "pu0": 0}
    return r, j
def library(td, dev, params, name):
    r, j = reps(name)
    lib = m.patched_library(td, {dev.dev_type: [(dev.bin, params)]}, r)
    text = lib.read_text()
    adds = []
    for dt in ("nmos", "pmos"):
        base = sp.PDK77K_DIR / sp._CORNER_BASENAME[dt]
        patched = Path(td) / sp._CORNER_BASENAME[dt]
        src = patched if patched.exists() and patched.as_posix() in text else base
        rep = m.replica_corner_chan(src.read_text(errors="replace"), dt, j)
        rep = rep.replace(f"{sp._SUBCKT[dt]}_ch", f"{sp._SUBCKT[dt]}_jn").replace(f"{sp._MODEL[dt]}_ch", f"{sp._MODEL[dt]}_jn")
        p = Path(td) / f"replica_jn_{dt}.spice"; p.write_text(rep)
        adds.append(f'.include "{p.as_posix()}"\n')
    lib.write_text(text.replace(".endl", "".join(adds) + ".endl"))
    return lib
def patch_body(body):
    out = []
    for ln in body.splitlines():
        if re.match(r"^G\w+c ", ln): continue        # gmin cancels in the difference
        out.append(ln)
        mm = re.match(r"^X(\w+)r (\S+)_ri (\S+)_rg (\S+)_rs (\S+)_rb (\S+)_ch (.*)$", ln)
        if mm:
            nm, p, sub, rest = mm.group(1), mm.group(2), mm.group(6), mm.group(7)
            out.append(f"X{nm}j {p}_jd {p}_rg {p}_rs {p}_rb {sub}_jn {rest}")
            out.append(f"V{nm}J {p}_rx {p}_jd DC 0")
            out.append(f"F{nm}j 0 {p}_ri V{nm}J 1")
    return "\n".join(out) + "\n"
def task(t):
    tag, label, params, name, osdi = t
    dev = drv.devices()[tag]
    pm = dev.dev_type == "pmos"; s = -1.0 if pm else 1.0
    sub = sp._SUBCKT[dev.dev_type]
    res = {"tag": tag, "label": label, "name": name, "msgs": {}}
    with m1.tempdir("x15j_") as td_s:
        td = Path(td_s)
        lib = library(td, dev, params, name)
        for swap in (False, True):
            for tol, opt in TOLS:
                for mode in ("card_rep", "elem"):
                    lab = f"{'swap' if swap else 'fwd'}_{tol}_{mode}"
                    f = td / f"{lab}.dat"
                    body = patch_body(m.device_body(dev, mode, swap=swap))
                    sense = mode == "card_rep"
                    ctrl = m.grid_control(dev, f, swap, sense)
                    if sense and tol == "tight":
                        r_ = f"@m.xm1r.m{sub}[id] @m.xm1j.m{sub}[id]"
                        ctrl = ["save all " + r_, ctrl[0], ctrl[1] + " " + r_]
                    r = m1.run_deck(m.deck(lab, lib, body, ctrl, Path(osdi) if mode == "elem" else None, options=opt), td, f"{lab}.sp", timeout_s=300)
                    res["msgs"][lab] = r["msgs"]
                    if sense and tol == "tight":
                        rows = m1.read_wrdata(f, 8)
                        if rows.shape[0] == 11175:
                            vd = rows[:, 0]
                            vds_rep = -vd if swap else vd
                            ich = m.drain_channel_current(rows[:, 5], vds_rep, pm)
                            res[f"{lab}_jn_id_max"] = float(np.max(np.abs(rows[:, 7])))
                            isense = rows[:, 3]
                            fq = (s * vd) > 0
                            d = np.abs(isense - ich)
                            e = d / (1e-6*np.abs(ich) + 1e-13)
                            res[f"{lab}_P2_fwdq_eq"] = float(np.max(e[fq])); res[f"{lab}_P2_revq_eq"] = float(np.max(e[~fq]))
                            res[f"{lab}_P2_fwdq_maxabs"] = float(np.max(d[fq])); res[f"{lab}_P2_revq_maxabs"] = float(np.max(d[~fq]))
                        else:
                            res[f"{lab}_rows"] = int(rows.shape[0])
    return res
if __name__ == "__main__":
    names = sys.argv[1].split(","); sets = sys.argv[2].split(",")
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
        keys = sorted({k for r in rr for k in r if ("_P2_" in k or "_jn_" in k or "_rows" in k)})
        summ[n] = {"n_decks": sum(len(r["msgs"]) for r in rr), "msgs": msgs,
                   **{k: max(r.get(k, float("nan")) for r in rr) for k in keys}}
    print(json.dumps(summ, indent=1))
    if len(sys.argv) > 4: Path(sys.argv[4]).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
