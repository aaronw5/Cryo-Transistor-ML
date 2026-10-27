# I25-1 amendment A3 (user instructions, 2026-10-27), committed alone before every affected run

User instructions, verbatim: "I think we shoudl use random as starting point….."; "We shoudl aim for that low temp
fit…"; "Basically we want to fit SXACTLY what the paper fit, jsut with our method. That's it"; "No start this now.
Both 7 and 43". Runs done before A3: template check, screen around the stock card (45 selected), specs, 7p bank
(stopped at 15:34, incomplete, not used). No fit, training or refit had run. The stock-template results are not used
by any A3 stage; the stock-template code stays (default template unchanged) and its stages are not run.

## Task (replaces prereg sections 1, 3, 4, 5)

- Template: the released 77 K card files (paper structure: its 18 custom bins, tnom -196.15 C, temperature
  coefficients 0, version 4.82, base expressions and rdw / rsw = 1e-12 as written) with every released multiplier set
  to 1: each `{base*number}` line gets multiplier 1, and every `.param <name>_nom` used as a named multiplier
  (pmos bin 0: vth0_nom 2.08113, u0_nom 3.41512, nfactor_nom 13.5087, vsat_nom 9999.92, rdsw_nom 1.11407e-05, and
  the others) is set to 1. Template id `mult1` (`CRYOML_PDK_TEMPLATE=mult1`, files `data/processed/pdk77k_mult1/`),
  bins = the released card's bins (as the paper). State C. Check: `cold_start.bin_multipliers` returns 1 for every
  multiplier of every bin of the mult1 files; the named-multiplier fix (prereg section 2) is in force.
- Metric and curve set: the paper's RRMS with the released card's curve set (primary); M-rule secondary unchanged.
- Arm A: the paper's 7 parameters (VTH0, U0, NFACTOR, VSAT, DELTA, RDSW, ETA0). Arm B (extension): the set selected
  by the unchanged 115-candidate screen around the mult1 card (size as selected), queued after arm A.
- Boxes: centers = mult1 card values (showmod readback); widths = the registered cold-start rule (VTH0 x/÷3, other
  PARAMS7 x/÷100, others x/÷3). The box bounds use the multiplier-1 value; the start does not.
- Starts: 3 random starts per device, k = 0, 1, 2: local coordinates uniform in [-1, 1] on every active coordinate
  (log-uniform inside the box), `numpy.random.default_rng(20261027 + 1000 k + i)` with i the device's index in
  `PAPER_DEVICES`. Unfitted parameters: mult1 card values.
- Step A per start: arm A: CMA-ES 3000 evaluations centered on the random point (sigma0 0.3, seed 20261015), TRF polish
  max_nfev 400, IRLS 2. Arm B: round 1 CMA-ES 8000 centered on the random point + polish 600; rounds 2 and 3 with the
  registered re-centering rule on that start's card (CMA-ES 8000 + polish 600); FD polish (max_nfev 400, diff_step
  0.01, IRLS 2) in that start's round-3 box. No bank (it only supplied starts) and no seeding of arm B by arm A.
- Multistart output per device: the start with the lowest training RRMS (primary metric on the device's own data,
  after the last stage); same rule for every device. Reported: min / median / max over the 3 starts per device,
  NGSpice calls summed over the 3 starts. Reference (arm A only): one run from the box center (mult1 card), not used
  for selection.
- g v2 pipeline (prereg section 6) unchanged on the multistart cards: refit box = arm A the 7p box; arm B the selected
  start's round-3 box. Primary = full-data fit of the 18 devices; LODO secondary (reported in tables, not in slides).
- Reference rows: paper card 0.2751 / 0.2595; stock tt card at 77 K 3.418 / 3.108 (stored step_deck curves); mult1
  card with no fit (new). Old-pipeline stage rows as registered.
- Queue: arm A Step A and g stages first, then the arm B screen, Step A and g stages, then decide.

Estimated time: arm A Step A about 0.3 h (3 x 3,300 calls x 18), arm A g about 1.5 h; arm B screen 0.1 h, Step A
about 3.5-4 h (3 starts x about 44,000 calls x 18 at 0.04-0.05 s per call, 10 workers), arm B g about 1.5 h; total
about 7-8 h.
