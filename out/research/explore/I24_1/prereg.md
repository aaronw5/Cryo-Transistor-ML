# I24-1 pre-registration (IN FORCE): (A) three-way dataset-level gate {C0_refit, selected g v2, B1}, (B) selected g v2 on top of B1 on IHP SG13G2

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-27), committed alone, before any I24-1 code, simulation, training, refit or gate-statistic value.
Base commit 1b89401. DEV data only: every TEST split stays locked (no TEST tag is loaded; every loaded tag is checked
against the DEV / TEST lists of its split file, as in I22-1 / I23-1; sky130 77 K has no locked TEST since the
2026-10-21 user decision). This study does not commit an amended TEST decision file. If the Part C TEST prediction
differs from the committed I19-1 decisions (a755e18), the difference is written as a proposed amendment in the
RESEARCH_LOOP outcome entry only (user decision).

Origin: I23-1 (prereg a9aa227, outcome 1b89401). The user did not answer the I23-1 decision push; the pushed default
applies: add baseline B1 (wider card set) as a third choice of the dataset-level gate, and test g v2 on top of B1.

Results known when this was written (all DEV, stored):
- GMR(X, C0_refit) [95% CI] (I22-1 decision, device bootstrap; TU Delft 7-group cluster):
  B1: sky130 1.020 [0.999, 1.056], NIST lvt 1.032 [1.019, 1.049], NIST RT card 1.021 [1.013, 1.029], IHP 233 / 300 /
  343 / 398 K 0.794 [0.659, 0.925] / 0.775 [0.653, 0.903] / 0.695 [0.579, 0.816] / 0.809 [0.704, 0.914], TU Delft
  4.13 / 295.0 K 1.064 [1.029, 1.116] / 1.110 [1.017, 1.250].
  Selected g v2: sky130 0.743, NIST lvt 0.803, NIST RT card 0.811, IHP 1.084 / 1.255 / 1.188 / 1.260, TU Delft 0.826 /
  0.829. Pooled exp(mean ln GMR vs C0_refit) over 9 datasets: B1 0.913, selected g v2 0.957; two-way gate under the
  frozen rule 0.923 (method-freeze draft).
- Frozen-rule gate statistic (v1-recipe S3_D, state C; method-freeze draft 1.E): sky130 -0.1777, NIST lvt -0.1172,
  NIST RT card -0.0730, IHP 233 / 300 / 343 / 398 K +0.1866 / +0.3114 / +0.3669 / +0.3667, TU Delft 4.13 / 295.0 K
  +0.0055 / +0.2537. Two-way decisions: apply g v2 on sky130 and both NIST datasets, C0_refit on IHP x4 and TU Delft
  x2; equal to the committed a755e18 decisions on the 8 TEST splits.
- I19-1 Part C OLS of the realised ln GMR on S3_D over 20 simulated datasets (v1 recipe): y = -0.021454 + 1.217286
  S3_D (R^2 0.993).
- B1 screens (I22-1, state C, all DEV devices of each dataset; IHP: stored I5-1 per-class screen on the 300 K starting
  cards, used at all four temperatures): selected extras sky130 19, NIST lvt 18, NIST RT card 19, TU Delft 17 / 16,
  IHP lv 75 / hv 76. The per-device screen responses are stored. No leave-one-group-out screen selection, no
  S_B1 value and no I24-1 refit was computed before this file.
- I22-1 IHP B1 held-out refits: 85-87 active parameters (12-parameter base box + screened extras), max_nfev from
  `matched_nfev`, about 1,900-2,100 NGSpice calls per DUT (6.0-6.6x C0_refit).

The parts are evaluated separately; each has its own validity checks and rule.

## Part A: three-way dataset-level gate (retrospective, 9 measured DEV datasets)

Datasets: sky130 77 K (18, LODO), NIST lvt (53), NIST RT card (87), IHP SG13G2 233 / 300 / 343 / 398 K (18 each),
TU Delft 4.13 / 295.0 K (45 each, 7 geometry groups). Arms and rows (no new rows except the B1 LOGO refits below):
- C0_refit: stored rows (freeze-draft loader `plot_freeze_draft.datasets()`, as I22-1).
- g: selected g v2, stored rows (same loader).
- B1: I22-1 Part A B1 rows (`I22_1/partA_<ds>/results.json`; screen on all DEV devices).

Gate statistics (DEV only; same inner groups as S3_D: sky130, NIST and IHP the device (IHP: training = the other DEV
DUTs of the same class), TU Delft the geometry group):
- S_g,D: the frozen-rule S3_D (v1 recipe, state C), read from the stored files (sky130 `I19_1/s3/sky130_v1.json`,
  NIST `I20_1/s3/{nist4,nistrt}_v1.json`, IHP `I19_1/s3/ihp{233,300,343,398}_v1.json`, TU Delft
  `I21_1/s3tud/s3tud.json` `S3_D_state_C`). Not recomputed.
- S_B1,D = mean over all DEV records r of ln(RRMS(B1_{-k(r)}(r)) / RRMS(C0_refit(r))), where k(r) is the inner group
  of r and B1_{-k} is the held-out B1 refit of r (I22-1 refit protocol through `i22_1_baselines.refit` with kind
  "b1": state C, starting record, extras at the card values, max_nfev = `matched_nfev`, IRLS 2, diff_step 0.01,
  strict paper RRMS with frozen inclusion) whose extras are the edge-screen selection (`screen_select`, threshold
  1e-4) over the stored per-device screen entries of the inner-training devices of group k only (sky130: the other
  17 devices; NIST: the other 52 / 86; IHP: the other DEV DUTs of the same class, 300 K I5-1 screen entries; TU
  Delft: the devices of the other 6 groups). NIST: per-device extra values from the I22-1 screen file, restricted to
  the LOGO selection. A record whose LOGO selection equals the I22-1 selection reuses its I22-1 B1 row (identical
  computation; VA3 checks the refit path). A record whose LOGO selection is empty gets log ratio 0 (B1 = C0_refit).
  Non-finite log ratios are left out of the mean and counted; if more than 10% of the records of a dataset are
  non-finite, S_B1,D is invalid there.

Primary choice rule (three-way gate), fixed before any S_B1 value:
1. Candidates: g iff S_g,D < 0 (the frozen rule, unchanged); B1 iff S_B1,D < 0 (strict; same threshold 0).
2. No candidate: C0_refit on every device of the dataset. One candidate: that arm.
3. Both candidates: put S_g,D on the scale of ln GMR vs C0_refit with the I19-1 Part C OLS, yhat_g = -0.021454 +
   1.217286 S_g,D (S_B1,D is already a held-out ln GMR vs C0_refit); choose g iff yhat_g < S_B1,D, B1 iff
   S_B1,D < yhat_g; equal values go to B1 (lower NGSpice cost: 2-7x C0_refit vs 28-89x for g).
4. B1 not buildable on a dataset (no stored per-device screen, or S_B1,D invalid): the two-way frozen rule there. The
   report lists every such dataset (expected: none; B1 rows exist on all 9).

Sensitivity rules (reported only, no role in the outcome):
- S1 raw minimum: the arm with the smallest of {C0_refit: 0, g: S_g,D, B1: S_B1,D} (strict; ties to C0_refit, then B1).
- S2 full screen: S_B1,D computed from the I22-1 B1 rows (screen on all DEV devices) instead of the LOGO rows.
- S3 leave-group-out gate (per device): for each record d, both statistics are recomputed without the records of
  d's inner group (S_g from the stored per-record v1 log ratios, S_B1 from the LOGO log ratios), the primary rule is
  applied, and d gets the chosen arm's row. This removes the scored device's own log ratio from its decision.

Realised outcome: on each dataset, the rows of the chosen arm on every DEV device (B1 rows = I22-1 rows; secondary:
the LOGO rows). GMR(arm, C0_refit) with `paired_ratio` (95% percentile bootstrap, 10,000 resamples, seed 0; unit
device, TU Delft 7-group cluster). Pooled GMR = exp(mean over the 9 datasets of ln GMR). Pooled CI: stratified
bootstrap, each replicate resamples devices (TU Delft clusters) within every dataset independently, the same indices
for every arm, per-dataset GMR then pooled; 10,000 replicates, seed 0; gate decisions are held fixed in the
replicates (the CI does not include decision uncertainty). Reported for: C0_refit, g always, B1 always, two-way
frozen gate, three-way gate (primary), S1, S2, S3, and the post hoc per-dataset oracle (best of the three arms by
realised GMR; report only). Paired pooled ratios with CI: three-way / two-way, three-way / B1 always, three-way / g
always. Aggregates per arm and dataset: all-device mean and combined (mean of nMOS / pMOS family means) RRMS.

Rule (Part A outcome):
- R1: GMR(three-way, C0_refit) <= 1 (point) on all 9 datasets.
- R2: pooled ratio three-way / two-way frozen gate with CI upper < 1.
- R3: pooled three-way <= pooled B1 always and <= pooled g always (point).
- "three-way gate supported (retrospective)" iff R1, R2 and R3; "partially supported (retrospective)" iff R2 holds
  and R1 or R3 fails; "not supported (retrospective)" iff R2 fails.
Stated limitation (fixed now): S_B1,D and the realised B1 rows come from the same held-out refits (identical where
the LOGO screen equals the full screen), so on datasets where B1 is chosen R1 holds nearly by construction, and the
retrospective part does not measure how well S_B1 predicts the B1 gain on new devices. S3 is the less circular check;
the locked TEST splits are the test.

Validity (Part A):
- VA1: the S_g,D values read equal the method-freeze-draft values listed above (|difference| < 5e-5).
- VA2: `screen_select` over all DEV devices reproduces the I22-1 selections (sky130, NIST x2, TU Delft x2) and the
  I5-1 per-class selections (IHP) exactly.
- VA3: on each dataset with at least one new LOGO refit, the I22-1 B1 refit with the full selection is rerun on 2
  devices in the same job and reproduces the I22-1 B1 rows (|difference| <= 1e-12).
- VA4: S_B1,D finite with at most 10% non-finite records on every dataset; every DEV device has a row in every arm.
- VA5: every loaded tag is in the DEV list and not in the TEST list of its split file.
- VA6: unit tests pass.

## Part C: TEST prediction (no TEST data loaded)

For each of the 8 locked TEST splits with a DEV counterpart (NIST lvt 38, NIST RT card 58, IHP 10 DUTs x 4
temperatures, TU Delft 24 x 2), the three-way decision is the primary-rule decision of the DEV dataset (statistics
from all DEV records; IHP TEST B1 arm: the stored I5-1 300 K per-class DEV screen). Each decision is compared with the
committed a755e18 decision and with the two-way frozen-rule decision. Any difference is written as a proposed
amendment in the RESEARCH_LOOP outcome entry; no TEST decision file is written or committed.

## Part B: selected g v2 on top of B1, IHP SG13G2 233 / 300 / 343 / 398 K (18 DEV DUTs each)

Starting records (the "C0" of this pipeline): the B1 card of each DUT, i.e. the I22-1 B1 held-out card
(`I22_1/partA_ihp<T>/heldout_cards.json`, key `B1|<tag>`), simulated on the measured bias set with the I5-1
`MeasProblem` on the B1 box (`extend_box(base_box, b1_extras(dev, I5-1 class selection))`) and stored as an E1
record (`to_record`). Penalty grids: the B1 card on the I17-2 75 x 75 forward-quadrant grid (0 .. class VDD).
g v2 recipe: I17-2 / I21-1 unchanged (sym_v2 inputs, EPS, HYPER_V2, 5 seeds, lambda grid `GRID_LAMBDA`, nested inner
selection `inner_folds` / `inner_score` / `select_lambda`, LODO within class, training order, per-fold lambda*),
trained on the B1 records. Held-out refits (inner and outer): start = the DUT's B1 card, B1 box, max_nfev =
`matched_nfev(12, n_active(B1 box), 400, 2)`, IRLS 2, diff_step 0.01, g in the loop (`corrected_problem`), strict
paper RRMS of I_card x exp(g) with the frozen inclusion (the I5-1 `refit_task` "corr" path on the B1 box).
Ngspice-46 + psp103.osdi as in I17-2 / I21-1 / I22-1 / I23-1.

Arms: B1 (I22-1 rows); B1_cont (the same refit of the B1 card with g = 0, same budget; equal-budget control);
B1+g v2 (selected lambda*); C0_refit (stored rows); selected g v2 (stored rows, starting cards).

Primary: GMR(B1+g v2, B1) per temperature (`paired_ratio`, device bootstrap); label gain (CI upper < 1), worse (CI
lower > 1), inconclusive. Rule: "g v2 adds to B1 on IHP" iff CI upper < 1 at all 4 temperatures; "g v2 worse than
B1 on IHP" iff CI lower > 1 at all 4; otherwise "g v2 gain over B1 not shown". Secondary (report only): GMR(B1+g v2,
C0_refit), GMR(B1+g v2, B1_cont), GMR(B1_cont, B1), GMR(B1+g v2, selected g v2); lambda* counts; RRMS all-device /
combined per arm; NGSpice calls per held-out DUT (arm total including the nested selection, I22-1 cost convention).

Validity (Part B, per temperature):
- VB1: the B1 records score equal to the I22-1 B1 rows (|difference| <= 1e-12) on 18 / 18 DUTs.
- VB2: penalty grids finite, 75 x 75, forward corner current > 0, 18 / 18.
- VB3: inner selection complete (18 outer folds, 3 inner folds x 5 lambda trainings each, every inner refit scored).
- VB4: at most 10% non-finite rows in each new arm.
- VB5: unit tests pass. VB6: DEV only.
- VB7 (checked once, before the main runs): the B1-box refit path with g = 0 from the starting record and the I22-1
  extras reproduces the I22-1 B1 row on 2 DUTs at 300 K (|difference| <= 1e-9). A failure is recorded as an
  amendment before any main run.

Not included: sky130 77 K. The B1 cards exist (I22-1), but the nested selection on sky130 runs about 1,530 inner
refits (I15-2: 8,991 s with the 43-parameter cards) at about 1.7x the C0_refit calls with B1 (I22-1: 9,885 vs 5,683
calls), plus a new sky130 B1 record / penalty-grid path; estimated 4-5 h of compute. B1 does not gain over C0_refit
on sky130 (1.020). Left as an option for the user.

## Predictions

- P1: three-way choices: B1 on IHP 233 / 300 / 343 / 398 K; g on sky130, NIST lvt, NIST RT card; C0_refit on TU Delft
  4.13 and 295.0 K.
- P2: the LOGO screen selection equals the I22-1 selection on at least 75% of the records of every dataset except TU
  Delft.
- P3: Part A outcome "three-way gate supported (retrospective)".
- P4: pooled GMR(three-way, C0_refit) < 0.85.
- P5: Part C: the three-way TEST decision differs from a755e18 on the 4 IHP splits only (B1 instead of C0_refit).
- P6: Part B outcome "g v2 gain over B1 not shown", with GMR(B1+g v2, B1) > 1 (point) at >= 3 temperatures.
- P7: GMR(B1_cont, B1) within [0.99, 1.01] at all 4 temperatures.

## Order and compute

1. This file + RESEARCH_LOOP entry, committed alone; push.
2. Code (`src/cryoml/explore/i24_1_gate3.py`, `scripts/research/explore/i24_1_gate3.py`), unit tests, smoke, VB7;
   commit, push.
3. Queue (one job at a time, <= 10 workers, BLAS / OpenMP threads 1, `torch.set_num_threads(1)`): Part A LOGO
   refits (all 9 datasets, VA3 reruns), Part B per temperature (records, pens, base (B1_cont), inner, sel, decide),
   decide.
4. decision.json, report batch, site claim, RESEARCH_LOOP outcome, freeze-draft update, memory; commit, push.

Estimate: Part A at most 371 B1 refits (all records, worst case), <= 1.5 h; Part B about 3 h per temperature (270
trainings ~1,200 s; 730 B1-box inner refits ~7,000-9,500 s at 10 workers; selected arm ~15 min), about 12 h total.
