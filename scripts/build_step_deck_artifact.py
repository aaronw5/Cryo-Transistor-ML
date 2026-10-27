from pathlib import Path
D=Path('/private/tmp/claude-501/-Users-anrunw-Documents-cryo-ml/5aadb445-5fac-407c-a0cc-7cb848012626/scratchpad/best_vs_paper_slides/project/slides')
F="font-family:'IBM Plex Sans', Arial, sans-serif"
SEC=f"background:#f6f5f1;color:#1f252b;{F};padding:56px 96px 48px;display:flex;flex-direction:column;gap:20px"
H='<h2 style="font-size:48px;font-weight:600;line-height:1.1;color:#1f252b">{}</h2>'
def P(t,size=26,extra=""): return f'<p style="font-size:{size}px;line-height:1.3{extra}">{t}</p>'
def col(items,size=26,gap=6,width=None):
    w=f";width:{width}px" if width else ""
    return f'<div style="display:flex;flex-direction:column;gap:{gap}px{w}">'+"".join(P(i,size) for i in items)+'</div>'
def img(blob,alt,w,h,boxw=None):
    bw=f";width:{boxw}px" if boxw else ""
    return f'<div style="display:flex;flex-direction:row;justify-content:center;background:#ffffff;border:1px solid #c9ced3{bw}"><img src="/_blob/{blob}" alt="{alt}" style="width:{w}px;height:{h}px;object-fit:contain"></div>'
def table(head,widths,rows,size=24,hl=()):
    right=lambda i: i>0
    s=f'<table style="width:1728px;font-size:{size}px;color:#1f252b;background:#ffffff;border:1px solid #c9ced3">'
    s+='<tr style="background:#e4e7ea">'+"".join(f'<th style="width:{w}%">{h}</th>' for h,w in zip(head,widths))+'</tr>'
    for i,r in enumerate(rows):
        bg=' style="background:#fbe8de"' if i in hl else ''
        s+=f'<tr{bg}>'+"".join(f'<td>{c}</td>' for c in r)+'</tr>'
    return s+'</table>'
def sec(i,body,style=SEC): return f'<section id="{i}" style="{style}">\n'+"\n".join(body)+'\n</section>\n'
S={}
S['title']=sec('title',[
 '<h1 style="font-size:72px;font-weight:600;line-height:1.15;color:#f1f0eb">SKY130, 77 K: card fit from the stock card</h1>',
 col(["Devices: 18 (8 nMOS, 10 pMOS)","Simulator: NGSpice 41","Steps: stock card, A card fit, B residual, C correction g, D penalty, E held-out test",
      '<span style="color:#f2a68f">All-device RRMS: paper card 0.2751; card refit + g 0.0739</span>',
      '<span style="color:#b8bec4">Split: development only, no locked test. 2026-10-27</span>'],30,14)],
 f"background:#1f252b;color:#f1f0eb;{F};padding:128px 160px;display:flex;flex-direction:column;justify-content:center;gap:48px")
S['s01-measured']=sec('s01-measured',[H.format("Measured data"),
 img("1844405a05e791cedc2e08e8e320a132","Measured Id-Vd and Id-Vg, pMOS L 0.35 um, W 5 um, 77 K",1700,560),
 col(["Figure: pMOS, L 0.35 um, W 5 um; faded = below current floor, not scored",
      "Devices: sky130 nfet_01v8_lvt (8), pfet_01v8_lvt (10); 77 K",
      "Curves per device: Id-Vd at |Vgs| 0.37-1.85 V (5); Id-Vg at |Vds| 0.01-1.85 V (6)",
      "nMOS L x W (um): 0.15 x 1.6, 0.19 x 7, 0.25 x 1.6, 1 x 1.6, 1 x 3, 8 x 1.6, 20 x 0.64, 100 x 100",
      "pMOS L x W (um): 0.35 x 0.55, 0.35 x 1.6, 0.35 x 5, 0.5 x 0.42, 0.5 x 0.64, 2 x 5, 4 x 7, 8 x 0.84, 8 x 1.6, 8 x 5",
      "Scored: 175 curves, 25,086 points"],24)])
S['s02-stock']=sec('s02-stock',[H.format("Stock card, 77 K"),
 img("8a1390560d953aa342fca35147e0716f","pMOS L 0.35 um W 5 um: measured; stock card at 77 K (orange); stock values in 77 K card file (purple)",1700,470),
 '<div style="display:flex;flex-direction:row;gap:36px">'+
 col(["Orange = stock sky130A tt card, tnom 30 C, run at -196.15 C",
      "RRMS, 18 devices: 3.418 all-device, 3.108 combined",
      "RRMS, nMOS / pMOS mean: 0.318 / 5.897",
      "Purple = stock values in 77 K card file, tnom -196.15 C (Step A start)",
      "RRMS, 18 devices: 11.601 / 10.509"],24,8,900)+
 '<div style="display:flex;flex-direction:column;gap:8px;width:792px"><img src="/_blob/42728e3500a3277bc0e28dbb5d75edc7" alt="RRMS_c and device RRMS formula" style="width:780px;height:168px;object-fit:contain">'+
 "".join(P(t,24) for t in ["c = scored curve; n_c = its points; C = scored curves per device",
   "Scored set = curves scored with the published card (fixed for all rows)",
   "All-device = mean of 18; combined = (nMOS mean + pMOS mean) / 2"])+'</div></div>'],
 SEC.replace("gap:20px","gap:18px"))
S['s03-paper']=sec('s03-paper',[H.format("Paper card (reference)"),
 col(["Tuned parameters per device: VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0",
      "Card form: stock value x multiplier"],28,8),
 '<div style="display:flex;flex-direction:row;gap:32px">'+"".join(
  f'<div style="flex:{fl};display:flex;flex-direction:column;gap:8px;background:#ffffff;border:1px solid #c9ced3;border-radius:8px;padding:32px"><p style="font-size:28px;color:#5a6168">{a}</p><p style="font-size:64px;font-weight:600;line-height:1.1">{b}</p></div>'
  for a,b,fl in (("All-device RRMS","0.2751",1),("Combined RRMS","0.2595",1),("nMOS / pMOS mean","0.1198 / 0.3992",1.6)))+'</div>',
 col(["Run: NGSpice 41, 18 devices, same RRMS and curve set",
      "Paper inputs in this pipeline: slide &#8220;Paper inputs used&#8221;"],26,8)],
 SEC.replace("gap:20px","gap:40px").replace("padding:56px 96px 48px","padding:72px 96px 72px"))
S['s04-stepA']=sec('s04-stepA',[H.format("Step A: card fit"),
 '<div style="display:flex;flex-direction:row;gap:28px">'+img("b6cb1fe890ab9508e1ced935297083a2","All-device and combined RRMS per fit stage, log scale",1100,470,1110)+
 col(["Input: measured curves; stock values in 77 K card file",
      "Operation: minimize device RRMS; 1 evaluation = 1 NGSpice run",
      "Output: starting card per device",
      "RRMS after FD: 0.1044",
      "RRMS after a2 rule: 0.1109 (combined 0.1049)",
      "Calls per device: 47,502 + 10,000 bank"],24,10,590)+'</div>',
 col(["Parameters: 7 (VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0), then 43 = 7 + 36 from a 115-candidate sensitivity screen",
      "Box (log): VTH0 x/÷ 3; other 6 x/÷ 100; other 36 x/÷ 3",
      "Starts: box center + 4 best of 10,000 Latin-hypercube runs (7-parameter stage)",
      "Optimizer: CMA-ES (3,000 / 8,000 evaluations, seed 20261015), then TRF",
      "Rounds 2-3: re-centered box, width x 0.5 except at box edge",
      "FD = TRF with finite-difference Jacobian, max 400 calls",
      "a2 rule = a2 &gt; 1 set to 1, a1 = 0 (BSIM4 check); 5 devices changed",
      '<span style="color:#a23b2c">Exception: pMOS L 0.35 x W 1.6 um, start = paper values for 5 of 7 parameters</span>'],24,4)],
 SEC.replace("gap:20px","gap:16px"))
S['s05-stepB']=sec('s05-stepB',[H.format("Step B: residual"),
 '<div style="display:flex;flex-direction:row;gap:28px">'+img("21fd658c0d4421dbef72e4724f405888","Residual after card refit, pMOS L 0.35 um W 5 um: top (I_card - I_meas)/den_c, bottom ln(I_meas/I_card)",1100,500,1110)+
 col(["Device: pMOS L 0.35 x W 5 um","Card: refit, no g; RRMS 0.1122",
      "Top: (I_card - I_meas) / den_c","den_c = mean |I_meas| of curve c",
      "Bottom: ln(I_meas / I_card) = target of g"],24,10,590)+'</div>',
 table(["Card-only fit, 18 devices","All-device RRMS","GMR vs refit, no g"],[56,20,24],[
  ["Step A round 2 / round 3 / FD","0.1094 / 0.1049 / 0.1044","-"],
  ["Refit, no g","0.1051","1"],
  ["Refit + 19 BSIM4 parameters","0.1054","1.020 [0.999, 1.056]"],
  ["Refit + series resistance, self-heating","0.1049","1.013 [0.994, 1.045]"],
  ["CMA-ES or TuRBO instead of TRF, same calls","-","1.002 to 1.048 (vs TRF)"]],24),
 P("GMR(A, B) = geometric mean over devices of RRMS_A / RRMS_B; [ ] = 95% bootstrap interval",24,";color:#5a6168")],
 SEC.replace("gap:20px","gap:16px"))
S['s06-stepC']=sec('s06-stepC',[H.format("Step C: correction g"),
 '<div style="display:flex;flex-direction:column;gap:10px;background:#ffffff;border:1px solid #c9ced3;padding:20px">'+
 '<img src="/_blob/55a5e5ce6b105b9475eb2ff7bd151ea3" alt="I = I_card exp(g(x1, x2))" style="width:700px;height:64px;object-fit:contain">'
 '<img src="/_blob/6ea62251db2662876f8fbf0f5d1e4864" alt="a, b, m, x1, x2 definitions" style="width:1300px;height:90px;object-fit:contain">'
 '<img src="/_blob/508cb1226f4c8bce4c25cf740d916920" alt="g = mean of 5 tanh(h_j)" style="width:680px;height:96px;object-fit:contain">'
 '<img src="/_blob/123d21e1cb5fbe20decf6424ba1f6b81" alt="training loss L" style="width:900px;height:96px;object-fit:contain"></div>',
 '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 40px">'+"".join(P(t,24) for t in [
  "I_card = NGSpice current of the device card (A)","s = +1 (nMOS), -1 (pMOS); Vgd = Vgs - Vds (V)",
  "EPS = 0.01 V","VDD = 1.85 V",
  "x1 = |Vgs| / VDD, x2 = |Vds| / VDD (forward bias)","x1, x2: unchanged under drain-source exchange",
  "z1, z2 = x1, x2 standardized on training devices","is_pmos = 0 or 1",
  "h_j = network 3-16-16-1, tanh; j = 1..5, seeds 0-4","|g| &lt; 1",
  "D = 17 training devices; their starting cards fixed","W = network weights; Adam, rate 3e-3, 2,000 steps"])+'</div>',
 P("Input: 17 other devices. Output: g for device k. RRMS: 0.1051 (no g) -&gt; 0.0739 (Step E)",24)],
 SEC.replace("gap:20px","gap:16px"))
S['s07-stepD']=sec('s07-stepD',[H.format("Step D: penalty, lambda"),
 '<div style="display:flex;flex-direction:row;background:#ffffff;border:1px solid #c9ced3;padding:16px"><img src="/_blob/46e611598f4b8011c405025b4ce4e000" alt="penalty P" style="width:1300px;height:90px;object-fit:contain"></div>',
 col(["d ln I / d|Vgs| = gm / I; d ln I / d|Vds| = gds / I (1/V)",
      "Grid: 25 mV, 75 x 75, 0-1.85 V, forward bias, I_card &gt; 1 nA",
      "lambda grid: {0, 1, 2, 5, 10} V",
      "Inner CV per fold k: 17 devices in 3 groups; train on 2, refit 3rd with g fixed",
      "Inner score = mean ln RRMS; lambda* = minimum; tie -&gt; larger lambda; device k excluded"],24,4),
 table(["lambda (V)","0","1","2","5","10","per fold"],[40,10,10,10,10,10,10],[
  ["folds choosing lambda","1","10","6","0","1","18"],
  ["all-device RRMS, fixed lambda","0.0736","0.0731","0.0726","0.0734","0.0772","0.0739"],
  ["combined RRMS, fixed lambda","0.0696","0.0692","0.0687","0.0696","0.0732","0.0700"]],24),
 col(["Fixed-lambda rows: computed after selection","Inner refits: 1,530; trainings: 360",
      "lambda* on all 18 devices: 1 V (circuit element)"],24,4)],
 SEC.replace("gap:20px","gap:16px"))
S['s08-stepE']=sec('s08-stepE',[H.format("Step E: held-out test"),
 P("Fold k: device k held out; 17 devices train g; k refit with g fixed, then scored. 18 folds.",26),
 table(["Stage","Device k data","Detail"],[34,16,50],[
  ["Starting card of k (Step A)","yes","k curves, before the folds"],
  ["g of fold k (Step C)","no","17 other devices, their input scaling"],
  ["lambda of fold k (Step D)","no","inner groups of 17 other devices"],
  ["Card refit of k, g fixed","yes","TRF, max 400 calls; 37-40 of 43 parameters free"],
  ["Score of k","yes","new NGSpice run, refit card x exp(g); fixed curve set"],
  ["Control: card refit, no g","yes","same refit, g = 0"],
  ["Method design (inputs, EPS, penalty, lambda grid)","yes","chosen on these 18 devices"]],24),
 '<div style="display:flex;flex-direction:row;gap:32px">'+"".join(
  f'<div style="flex:1;display:flex;flex-direction:column;gap:6px;background:#ffffff;border:1px solid #c9ced3;border-radius:8px;padding:24px"><p style="font-size:24px;color:#5a6168">{a}</p><p style="font-size:48px;font-weight:600;line-height:1.1;color:{c}">{b}</p></div>'
  for a,b,c in (("Card refit, no g (all / combined)","0.1051 / 0.0984","#2a6fdb"),("Card refit + g (all / combined)","0.0739 / 0.0700","#c8384e"),("GMR(+ g, no g); devices lower","0.743; 17 of 18","#1f252b")))+'</div>',
 P("GMR 95% interval: [0.654, 0.855]",24)])
S['s09-results']=sec('s09-results',[H.format("Results"),
 table(["Card","All-device","Combined","nMOS mean","pMOS mean","Device held out"],[40,12,12,12,12,12],[
  ["Stock card, 77 K","3.418","3.108","0.318","5.897","-"],
  ["Stock values, 77 K card file (Step A start)","11.601","10.509","0.685","20.333","-"],
  ["Paper card","0.2751","0.2595","0.1198","0.3992","-"],
  ["Starting card (Step A)","0.1109","0.1049","0.0510","0.1588","no"],
  ["Card refit, no g","0.1051","0.0984","0.0381","0.1587","yes"],
  ["Card refit + g","0.0739","0.0700","0.0345","0.1054","yes"],
  ["Card refit + g, NGSpice element deck","0.0739","0.0700","0.0345","0.1054","yes"]],26,hl=(5,6)),
 col(["GMR(refit + g, paper card): 0.249 [0.205, 0.305]; 18 of 18 lower",
      "GMR(refit, no g, paper card): 0.335 [0.285, 0.392]; 18 of 18 lower",
      "Device held out = device curves not used for g or lambda",
      "Element deck = Verilog-A element in NGSpice; per-device difference &lt;= 2.0e-4"],26,8)])
S['s10-perdevice']=sec('s10-perdevice',[H.format("Per-device RRMS"),
 img("c3e03d578ef949ab4a2ed1f6593e04c7","Per-device RRMS bars, log scale: paper card, card refit no g, card refit + g",1700,780),
 P("Higher with g: nMOS L 0.15 x W 1.6 um (0.0193 -&gt; 0.0320). Maximum with g: pMOS L 0.35 x W 0.55 um (0.336)",24)],
 SEC.replace("gap:20px","gap:18px"))
for sid,blob,n,devs in (("s11-iv1","0f4fcae61a99dd7b7d92f01c9e342356",1,"nMOS L 0.15-8 um"),("s12-iv2","600592c187bb51554cff38a0c7ee0ad3",2,"nMOS L 20, 100 um; pMOS L 0.35, 0.5 um"),("s13-iv3","a1008ac73397212672956261323f02c7",3,"pMOS L 0.5-8 um")):
    S[sid]=sec(sid,['<h2 style="font-size:44px;font-weight:600;line-height:1.1;color:#1f252b">I-V curves (%d/3)</h2>'%n,
     img(blob,f"Id-Vd and Id-Vg, {devs}: measured, paper card NGSpice, card refit + g element deck NGSpice",1724,820),
     P(f"Devices: {devs}. Left: Id-Vd; right: Id-Vg (log). Faded: not scored. All lines: NGSpice re-runs; RRMS = stored value",24)],
     f"background:#f6f5f1;color:#1f252b;{F};padding:40px 64px 36px;display:flex;flex-direction:column;gap:14px")
S['s14-element']=sec('s14-element',[H.format("Verilog-A element"),
 '<div style="display:flex;flex-direction:row;gap:32px;align-items:center"><div style="background:#ffffff;border:1px solid #c9ced3;padding:12px"><img src="/_blob/bd3f4e491727862ad5e1e61f690b7925" alt="I(d,s) = I_card + (e^g - 1) I_ch" style="width:620px;height:66px;object-fit:contain"></div>'+
 col(["I_ch = channel current of a card-transistor copy","Build: Verilog-A, OpenVAF, OSDI, NGSpice 41","g: 18 devices, lambda = 1 V"],24,4,1060)+'</div>',
 table(["Check (I17-1)","Value","Limit","Result"],[58,16,14,12],[
  ["Element vs card x exp(g), tight tolerance: points off","0","0","pass"],
  ["Convergence messages, non-finite, failed runs","0","0","pass"],
  ["|I| at Vds = 0 (A)","4.9e-17","1e-12","pass"],
  ["Verilog-A g - Python g","2.8e-16","1e-9","pass"],
  ["Drain-source exchange, |d ln I|","2.2e-5","1e-3","pass"],
  ["AC gm, gds vs element equation (relative)","1.1e-5","1e-3","pass"],
  ["Capacitances vs card (relative)","3.3e-14","1e-3","pass"],
  ["Inverters, 5-stage ring oscillators, mirrors failing","0","0","pass"],
  ["Default-tolerance difference, |I| &gt; 1 nA (relative)","1.8e-3","1e-3","restriction"],
  ["gds sign changes from g (curves)","6 of 90","0","restriction"],
  ["Run time, element / card (max)","4.60","2","restriction"]],24,hl=(8,9,10)),
 col(["Label: deployable with restrictions","gds sign changes: pMOS 2 x 5, 8 x 0.84, 8 x 1.6, 8 x 5 um; also in Python card x exp(g)",
      "gm sign changes, strong inversion: 0 of 54 curves"],24,4)],
 SEC.replace("gap:20px","gap:14px"))
S['s15-cost']=sec('s15-cost',[H.format("Cost"),
 P("1 call = 1 NGSpice run of all measured biases of one device",26),
 table(["Step","Calls per device","Other"],[52,20,28],[
  ["Step A: sample bank (7-parameter starts)","10,000","Latin hypercube"],
  ["Step A: 7 parameters","3,283","CMA-ES + TRF"],
  ["Step A: 43 parameters, rounds 1 / 2 / 3","21,213 / 14,287 / 8,405","CMA-ES + TRF"],
  ["Step A: FD","314","TRF"],
  ["Step D: lambda selection (estimate)","27,880","360 trainings; 28,110 CPU s"],
  ["Step E: card refit + g","311",""],
  ["Step E: card refit, no g","316",""],
  ["Total (Steps A, D, E)","85,693",""]],26,hl=(7,)),
 col(["Step D / refit without g: 89x","Step D estimate = 1,530 inner refits x median refit calls",
      "Fixed lambda 1 V vs per-fold lambda, GMR: 0.988 [0.962, 1.014] (sky130); 1.030 [1.002, 1.058] (NIST RT card, 87)",
      "I23-1 label: fixed lambda 1 V not shown non-inferior"],26,8)])
S['s16-sources']=sec('s16-sources',[H.format("Paper inputs used"),
 table(["Item","Source","Value"],[24,40,36],[
  ["Card file, all runs","paper 77 K file: tnom -196.15 C, temperature coefficients 0, version 4.82, 1 bin per device","RRMS 11.60 (stock values); 3.42 (stock card)"],
  ["Start, pMOS 0.35 x 1.6 um","paper tuned multipliers: VTH0 x2.08, U0 x3.42, NFACTOR x13.5, VSAT x9,999.9, RDSW x1.1e-5","start RRMS 0.3776 = paper card"],
  ["Start, other 17 devices","stock values (multipliers removed)","difference to stock &lt; 0.1 %"],
  ["43-parameter set","screen of 115 candidates, +/-10 % around paper card","not measured"],
  ["Bin, pMOS 0.35 x 0.55 um","paper file bin 5 = stock bin W 0.42-0.55 um","not measured"],
  ["Scored curve set","curves scored with paper card","same for all rows"],
  ["Box width x/÷ 100","paper multipliers 1.75e-4 to 49","not measured"],
  ["Data, RRMS, current floors, device count m","paper","task definition"]],24)])
S['s17-limits']=sec('s17-limits',[H.format("Limits"),
 table(["Item","Source","Value"],[30,30,40],[
  ["Locked test","-","none; 18 devices = development set"],
  ["Method design choices","these 18 devices","inputs, EPS, penalty, lambda grid"],
  ["Former test devices","opened 2026-10-21","6"],
  ["Paper inputs","slide &#8220;Paper inputs used&#8221;","1 start, 43-set screen, curve set"],
  ["Rerun from stock card","estimate","~57,500 calls per affected device + Steps C-E"],
  ["NIST 4 K, lvt card","I20-1","GMR 0.803 [0.757, 0.857]; 50 of 53"],
  ["NIST 4 K, RT card","I20-1","GMR 0.811 [0.761, 0.871]; 77 of 87"],
  ["NIST element","I21-1","core checks failed"],
  ["IHP SG13G2 300 K","I17-2","GMR 1.255 [1.026, 1.600]"],
  ["IHP 233 / 343 / 398 K","I21-1","GMR 1.084 / 1.188 / 1.260"],
  ["lambda selection cost","I22-1","89x refit without g"],
  ["Element restrictions","I17-1","1.8e-3; 6 of 90 gds curves; 4.6x run time"],
  ["Step A at matched budget","-","not compared"]],24)])
def deftable(rows,size=26,w=(28,72)):
    return table(["Term","Definition"],list(w),rows,size)
S['b1-transistor']=sec('b1-transistor',[H.format("Transistor I-V basics"),
 deftable([
  ["MOSFET","transistor; terminals gate, drain, source, body"],
  ["nMOS / pMOS","electron / hole channel; pMOS voltages and currents negative"],
  ["L, W","channel length, width (um)"],
  ["Vgs, Vds","gate-source, drain-source voltage (V)"],
  ["Id","drain current (A)"],
  ["Id-Vd curve","Id vs Vds at fixed Vgs"],
  ["Id-Vg curve","Id vs Vgs at fixed Vds; log scale shows subthreshold current"],
  ["gm, gds","dId/dVgs, dId/dVds (S)"],
  ["77 K","liquid-nitrogen temperature (-196.15 C)"],
  ["300 K","room temperature (27 C)"]],26)])
S['b2-model']=sec('b2-model',[H.format("Compact model, card, simulator"),
 deftable([
  ["Compact model","equations: Id from Vgs, Vds, L, W, temperature"],
  ["BSIM4","compact model used by the SkyWater 130 nm process"],
  ["Parameter","number in the BSIM4 equations, e.g. VTH0 = threshold voltage (V)"],
  ["Model card (card)","file of BSIM4 parameter values"],
  ["Bin","card section for one L, W range"],
  ["PDK","foundry model files; here SkyWater sky130A"],
  ["Stock card","PDK card as released; fitted at 300 K (tnom 30 C)"],
  ["tnom","temperature at which card values hold"],
  ["NGSpice","open-source circuit simulator; evaluates the card"],
  ["Stock card at 77 K","all-device RRMS 3.418 (slide &#8220;Stock card, 77 K&#8221;)"]],26)])
loop_box=lambda t,x: f'<div style="position:absolute;left:{x}px;top:250px;width:330px;background:#ffffff;border:2px solid #2a6fdb;border-radius:8px;padding:20px"><p style="font-size:26px;line-height:1.3;text-align:center">{t}</p></div>'
S['b3-fit']=sec('b3-fit',[H.format("NGSpice fit"),
 loop_box("1. Card parameters p",96)+loop_box("2. NGSpice run: simulated curves",546)+loop_box("3. RRMS vs measured curves",996)+loop_box("4. Optimizer: new p",1446),
 '<x-connector x1="426" y1="310" x2="546" y2="310" head="end" style="color:#4a5259;border-width:3px"></x-connector>',
 '<x-connector x1="876" y1="310" x2="996" y2="310" head="end" style="color:#4a5259;border-width:3px"></x-connector>',
 '<x-connector x1="1326" y1="310" x2="1446" y2="310" head="end" style="color:#4a5259;border-width:3px"></x-connector>',
 '<x-connector x1="1611" y1="372" x2="1611" y2="440" head="none" style="color:#4a5259;border-width:3px;border-style:dashed"></x-connector>',
 '<x-connector x1="1611" y1="440" x2="261" y2="440" head="none" style="color:#4a5259;border-width:3px;border-style:dashed"></x-connector>',
 '<x-connector x1="261" y1="440" x2="261" y2="372" head="end" style="color:#4a5259;border-width:3px;border-style:dashed"></x-connector>',
 '<div style="position:absolute;left:96px;top:480px;width:1728px">'+deftable([
  ["NGSpice fit","repeat steps 1-4; objective = device RRMS"],
  ["NGSpice call","1 simulation of all measured curves of 1 device"],
  ["Box","allowed range per parameter"],
  ["CMA-ES","evolutionary search: sample p, keep best, update sampling distribution"],
  ["TRF","trust-region least squares (SciPy least_squares)"],
  ["FD","finite-difference Jacobian: 1 call per parameter per step"],
  ["Latin hypercube","spread-out random sample of the box"]],24)+'</div>'])
S['s19-glossary']=sec('s19-glossary',[H.format("Glossary"),
 '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 40px">'+"".join(P(t,24) for t in [
  "Id, Vgs, Vds = drain current, gate and drain voltages","gm, gds = dId/dVgs, dId/dVds",
  "BSIM4 = transistor compact model","Card = file of BSIM4 parameter values",
  "Bin = card section for one L, W range","PDK = foundry model files",
  "NGSpice = open-source circuit simulator","NGSpice call = 1 simulation of 1 device",
  "RRMS = relative RMS current error (slide &#8220;Stock card, 77 K&#8221;)","All-device = mean over 18 devices",
  "Combined = (nMOS mean + pMOS mean) / 2","Starting card = Step A output",
  "Card refit = TRF fit of a held-out device","g = learned correction, I = I_card exp(g)",
  "MLP = stacked weighted sums + nonlinear functions","lambda = penalty weight (V)",
  "LODO = leave one device out","Fold = one LODO split",
  "GMR = geometric mean of RRMS ratios","CMA-ES = evolutionary search",
  "TRF = trust-region least squares","FD = finite-difference Jacobian",
  "Verilog-A = language for analog device models","OSDI = compiled-model interface of NGSpice"])+'</div>'],
 SEC.replace("gap:20px","gap:16px"))
# Step C: replace two grid entries with MLP and exp(g) definitions
S['s06-stepC']=S['s06-stepC'].replace("z1, z2 = x1, x2 standardized on training devices","MLP = stacked weighted sums + tanh; h_j = MLP 3-16-16-1").replace(
 "h_j = network 3-16-16-1, tanh; j = 1..5, seeds 0-4","exp(g) = current factor; g = 0 -&gt; I = I_card").replace(
 "is_pmos = 0 or 1","z1, z2 = standardized x1, x2; is_pmos = 0 or 1; seeds 0-4").replace(
 "x1, x2: unchanged under drain-source exchange","x1, x2 (dimensionless); same under drain-source exchange")
S['s07-stepD']=S['s07-stepD'].replace("d ln I / d|Vgs| = gm / I; d ln I / d|Vds| = gds / I (1/V)",
 "gm = dId/dVgs, gds = dId/dVds; d ln I / d|V| = gm / I, gds / I (1/V)").replace(
 "lambda grid: {0, 1, 2, 5, 10} V","lambda = penalty weight; grid {0, 1, 2, 5, 10} V").replace(
 "Inner CV per fold k","Cross-validation (CV) = train on part, score on the rest. Inner CV, fold k")
boxes='<div style="display:flex;flex-direction:column;gap:8px"><div style="display:flex;flex-direction:row;gap:8px">'+"".join(
 f'<div style="width:72px;height:56px;background:{"#c8384e" if i==5 else "#2a6fdb"};border-radius:6px"></div>' for i in range(18))+'</div>'+ \
 P('<span style="color:#2a6fdb">blue = 17 training devices</span>; <span style="color:#c8384e">red = test device k</span>',24)+'</div>'
S['s08-stepE']=S['s08-stepE'].replace(P("Fold k: device k held out; 17 devices train g; k refit with g fixed, then scored. 18 folds.",26),
 col(["Held-out test = score on data not used in training","LODO = leave one device out; fold k = device k held out; 18 folds"],24,4)+boxes)
S['s14-element']=S['s14-element'].replace("Build: Verilog-A, OpenVAF, OSDI, NGSpice 41",
 "Verilog-A = language for analog models; OpenVAF = its compiler; OSDI = compiled-model interface of NGSpice 41")

for k,v in S.items(): (D/f"{k}.html").write_text(v)
print(len(S))
