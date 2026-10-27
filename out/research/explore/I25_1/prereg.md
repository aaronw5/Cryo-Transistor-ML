# I25-1 pre-registration (IN FORCE): sky130 77 K pipeline rerun from the stock SkyWater card, 7-parameter and screened-set arms, with and without g v2

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-27), committed alone, before any I25-1 code beyond the template check below, before any I25-1 fit,
screen, bank, training or refit. Base commit b299dc9.

User instructions this study implements (2026-10-27): "don't start from paper cards, start from scratch" and "Do it"
(full rerun with no tuned value or card-file structure from the released 77 K cards); "there's no held out set here
cause the goal is to be able to fit real data" (primary = full-data fit of the 18 measured devices; LODO kept as
secondary); "Also do the same stuff with 7 parameters" (7-parameter arm with and without g); "include the iv curve
and stuff for ALL devices for both parameter counts". These were received before this file was committed and are
part of the registration (not amendments).

Provenance audit that motivates the study (b299dc9; audit scripts in the session scratchpad): every earlier
simulation used the released 77 K card file as template (tnom -196.15 C, temperature coefficients 0, version 4.82,
one custom bin per device); `cold_start._MULTIPLIER_RE` did not match the named multipliers of pmos_L0p35_W1p6 bin 0
(`{...*vth0_nom}`, `.param vth0_nom = 2.08113`), so that device kept the released tuned VTH0, U0, NFACTOR, VSAT, RDSW
(and the quote-form `'...*ub_nom'`, `'...*pclm_nom'` lines); the 43-parameter screen ran around the released card;
the box widths were set with the released multiplier range known; the scored curve set is the released card's set.

Known before registration: stock tt card at 77 K (no parameter changed, frozen curve set) all-device 3.4176 /
combined 3.1077 (nMOS 0.3184, pMOS 5.8970); released (paper) card 0.2751 / 0.2595. Template check (done before this
file, no fit): the I25-1 stock template below reproduces the curves of the full `sky130.lib.spice` section `tt` on
all 18 devices with maximum relative difference 0 (bit-identical), 0.07-0.09 s per device simulation.

## 1. Template

- Stock SkyWater sky130A tt models of `nfet_01v8_lvt` and `pfet_01v8_lvt`: `libs.ref/sky130_fd_pr/spice/*_01v8_lvt__tt.pm3.spice`
  copied byte for byte (tnom 30 C, version 4.5, stock temperature coefficients, 88 nFET / 48 pFET L/W bins), the
  `.param` block of the stock `__tt.corner.spice` wrappers (junction multipliers), stock `parameters/lod.spice`, stock
  mismatch corners, `mc_mm_switch = mc_pr_switch = 0`. Wrapper `data/processed/pdk_stock/sky130_stock_tt.lib.spice`.
- Temperature: the deck convention of every earlier run and of the upstream `sweeps.spice` decks,
  `.options temp=-196.15` (77 K). With the stock card NGSpice applies the BSIM4 temperature equations from tnom 30 C.
- Bin: NGSpice native L/W selection; the selected bin per device is read with the sentinel method of
  `spice_pdk.find_bin_index` and parameters are written into that bin. Bins: nmos L0.15/W1.6 55, L0.19/W7 6,
  L0.25/W1.6 53, L1/W1.6 51, L1/W3 35, L8/W1.6 48, L20/W0.64 64, L100/W100 0; pmos L0.35/W0.55 39, L0.35/W1.6 31,
  L0.35/W5 15, L0.5/W0.42 46, L0.5/W0.64 38, L2/W5 11, L4/W7 2, L8/W0.84 33, L8/W1.6 25, L8/W5 9.
- Code: `CRYOML_PDK_TEMPLATE=stock` selects the template in `src/cryoml/spice_pdk.py` (default `paper77k`, old path
  unchanged); stock pm3 lines carry several `key = value` tokens, written by a token patcher that keeps the other
  tokens (l/w/p size terms). BSIM4 state C (I13-1: b4check.c A1/A2 clamp written into the card) for every
  simulation; the stock bins have a1 = 0, a2 = 0.387 (n) / 0.467 (p) and no l/w/p A1/A2 terms.
- Same measured data, deck instance lines (sweeps.spice parasitics, per-device mult), metric code.

## 2. Named-multiplier fix

`cold_start.bin_multipliers` also reads `{expr*name}` and `'expr*name'` with a file-level `.param name = value`.
Unit test: pmos_L0p35_W1p6 bin 0 returns vth0 2.08113, u0 3.41512, nfactor 13.5087, vsat 9999.92, rdsw 1.11407e-05.
The stock template has no multipliers; the fix changes no I25-1 number.

## 3. Parameter screen

`scripts/pdk_parameter_sensitivity.py`, unchanged rule, run on the stock template at 77 K: 115 candidates
(`MAX_DC_CANDIDATES`), center = stock bin value (showmod readback), +/-10 % one-at-a-time step (family-median scale
for zero values; retry scales 1, 0.5, 0.25, 0.1), response = RMS current change / measured mean |I| on every curve,
threshold max(1e-4, 10 x repeatability); keep a candidate active on any device; PARAMS7 mandatory. The selected set
S (size N, whatever the rule gives) is the "screened set" of arm B. Reported: N, S, overlap with the earlier 43
(intersection, only-old, only-new, Jaccard).

## 4. Box widths and centers

Rule kept from the existing cold-start code (`cold_start.half_width`), not re-derived: log-symmetric boxes, VTH0
x/÷3, the other six PARAMS7 x/÷100, every other parameter x/÷3; rounds 2-3 re-center on the previous card, interior
half-widths x0.5 (minimum x/÷1.05), coordinates at |u| >= 0.9 keep their width. Why: a fixed prior choice applied
unchanged to every device, so no width is tuned on the stock-card result (limitation: the widths were set in October
with the released multiplier range known). Centers: stock bin values (showmod readback); a zero stock value is fixed.
Reported: fraction of active coordinates at |u| >= 0.9 per round.

## 5. Step A from the stock card (all 18 devices; same optimizers, budgets, seeds as the old pipeline)

1. 7-parameter stage: 10,000-point LHC NGSpice bank in the 7-parameter box (`pdk_gen_full_data.py`, seed
   20260818, global LHC); `ngspice_direct.py`: starts = center + 4 diverse best bank samples (min distance 0.35);
   CMA-ES 3000 evaluations, sigma0 0.3, seed 20261015; TRF polish max_nfev 400, IRLS 2.
2. Screened-set round 1: box centered on the stock card; extra start = the 7-parameter card; CMA-ES 8000, polish
   600, IRLS 2, seed 20261015, no bank.
3. Rounds 2 and 3: `next_direct_spec` rule; same budgets.
4. FD polish of round 3 (`polish_cards.py`, max_nfev 400, diff_step 0.01, IRLS 2, round-3 box).
Starting cards: arm A (7 parameters) = stage-1 card; arm B (screened set) = round 3 + FD card.

## 6. g v2 pipeline per arm (selected g v2 recipe of I15-2 / I16-1)

Records: the 18 starting cards simulated on the measured grid in state C (`e1_collect_real.collect`); record load
order = REAL_DEV then REAL_TEST of `real_split.json` (as before). Penalty grids: 75 x 75 forward grid at the starting
card (I12-1 grid code, ngspice-41 arm64). g: symmetric inputs, EPS 0.01 V, 3-16-16-1 tanh, 5 seeds, Adam 3e-3, 2000
steps, L2 1e-4, gm/gds penalty weight lambda. Refit: the arm's card with g in the loop, TRF max_nfev 400, IRLS 2,
diff_step 0.01, box = the arm's last box (arm A: 7-parameter box; arm B: round-3 box), start = the starting card.

- Nested lambda selection (I15-2 rule): for each outer set (each of the 18 LODO training sets and the full 18-device
  set), 3 inner folds (sorted by polarity, tag; i mod 3), one member (seed 0) per inner fold and lambda in
  {0, 1, 2, 5, 10} V, inner held-out refits; lambda* = argmin mean ln RRMS, ties to the larger lambda.
- PRIMARY, full data (user instruction): g trained on all 18 records (5 seeds) at lambda*_full; every card refit with
  that g in the loop; in-sample RRMS. Control: C0_refit = each starting card refit without g (same refit settings;
  no g, so in-sample = held-out).
- SECONDARY, LODO held-out: for device h, g trained on the other 17 records (5 seeds) at lambda*_h, card of h refit
  with that g. Labelled as the check that g does not fit per-device noise and as prediction at an unmeasured size.
  Only the lambda*_h member set is trained per fold (I15-2 defined the selected row as the LODO row at lambda*_h; the
  rows of the other lambdas are not needed).
- Order: arm B jobs, then arm A jobs; one heavy job at a time through the queue.

## 7. Scoring

Primary: the paper RRMS (`metrics.device_rrms_new`) with curve inclusion frozen to the released card's set (as all
earlier rows, comparable with 0.2751). Secondary: card-independent inclusion "M-rule": nMOS the upstream rule
(measured mean |I| thresholds, unchanged); pMOS the upstream thresholds (1e-10 A idvd, 1e-11 A idvg) applied to the
measured |I| at the last trimmed point instead of the simulated one. The M-rule set differs from the frozen set on
pmos_L0p5_W0p42 (+idvd@0.37, +idvd@0.74), pmos_L8_W0p84 (+idvd@0.74), pmos_L8_W5 (+idvd@0.74). Both reported,
all-device mean and combined (mean of the nMOS and pMOS family means), never compared across conventions.

## 8. Comparisons and labels

Rows: stock card; paper card; arm A card (C0_refit); arm A + g; arm B card (C0_refit); arm B + g; columns full-data
all-device / combined, LODO all-device / combined, NGSpice calls per device. Old (released-template) pipeline vs
from-scratch at every stage (7p stage, rounds 1-3, FD, start state C, C0_refit, g full-data (I16-1 in-sample rows),
g LODO selected (I15-2 rows)); the old 7p stage cards rescored in state C. GMR, device bootstrap (10,000, paired
ratio of `i3_2_bias.paired_ratio`), for each arm and for full-data and LODO: (arm + g) vs arm C0_refit; (arm + g)
vs paper card; arm C0_refit vs paper card; arm B + g vs old 43p + g. Label per GMR: "lower" if CI upper < 1, "higher"
if CI lower > 1, else "inconclusive". No decision depends on a label; the outcome lists them.

## 9. Validity

V1 template bit-identical to the full stock lib on 18 devices (done, 0). V2 unit tests pass (template switch, token
patcher, bins, named-multiplier fix). V3 every arm has 18 finite rows. V4 every final card re-simulated from its
stored parameters reproduces its stored RRMS (|d| <= 1e-9; g arms with g applied). V5 records' card-only score
equals `card_eval.score_card` on the starting card. A failed check makes the run invalid; the cause is reported.

## 10. Optional last stage

Verilog-A element checks of the arm-B full-data g (I17-1 rules, OSDI build) if time allows; label reported only.

## Outputs

`out/research/explore/I25_1/` (specs, screen, step A summaries, records, inner, lodo, full, c0, final cards, curves
npz (gitignored), decision.json); report batch `out/research/report/2026-10-2x_i25_1/` with I-V figures of all 18
devices for both arms (measured points, paper card NGSpice, arm card NGSpice, arm card + g NGSpice re-simulation,
per-curve RRMS), per-device tables, stage table, forest plot; site claim; RESEARCH_LOOP outcome entry.
Estimated cost: step A about 1.0 M NGSpice calls (3 h at 10 workers); g pipeline per arm 380 trainings and 1,674
refits (about 2.5 h); total about 8-9 h.
