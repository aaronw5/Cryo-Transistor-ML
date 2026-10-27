# I23-1 pre-registration (IN FORCE): (A) call-matched IHP control for B1, (B) fixed-lambda g v2 arm on the 9 measured DEV datasets

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-26), committed alone, before any I23-1 code, simulation, training or refit. Base commit 95e1502.
DEV data only: every TEST split stays locked (no TEST tag is loaded; every loaded tag is checked against the DEV /
TEST lists of its split file, as in I22-1; sky130 77 K has no locked TEST since the 2026-10-21 user decision). This
study writes no amendment of the committed I19-1 TEST gate decisions (a755e18).

Origin: I22-1 (prereg 02113d4, outcome 95e1502). The user did not answer the I22-1 decision push; the two pushed
default options apply: (A) a C0_refit control on IHP at the NGSpice-call count of B1, and (B) a g v2 arm with the
penalty weight fixed at lambda = 1 V (no nested selection) on all 9 DEV datasets.

Results known when this was written (all DEV, stored):
- I22-1 IHP (233 / 300 / 343 / 398 K, 18 DUTs each): GMR(B1, C0_refit) 0.794 [0.659, 0.925] / 0.775 [0.653, 0.903] /
  0.695 [0.579, 0.816] / 0.809 [0.704, 0.914]; GMR(B1, selected g v2) 0.733 / 0.617 / 0.585 / 0.642. B1 held-out
  refit NGSpice calls per device 287-2440 (sum 34,313-35,234 per temperature; B1 has 85 active parameters, the
  12-parameter PSP103 base box plus 73 screened extras; max_nfev 2436 = `matched_nfev`); C0_refit 71-396 per device
  (sum 5,516-5,841; 12 active parameters, max_nfev 400). B1 / C0_refit call ratio 6.0-6.6.
- I22-1 cost (Part B): selected g v2 arm 28-89x the C0_refit calls per held-out device; nested lambda selection
  96-99% of that.
- Stored lambda = 1 V held-out arms (5-member ensembles, same folds and training order as the selected arm, state C):
  sky130 `I15_2/lodo_sym_e1_l1` (all-device 0.07313; GMR vs C0_refit 0.734 [0.649, 0.841]; selected 0.07391, 0.743
  [0.654, 0.855]); IHP 300 K `I17_2/lodo_sym_e1_l1` (all-device 0.0200; selected 0.0205; C0_refit 0.0191); TU Delft
  `I18_2/lodo_{4K,295K}_sym_e1_l1` (all-device 0.0507 / 0.0220; selected 0.0484 / 0.0215; C0_refit 0.0602 /
  0.0265). The selected rows equal the lambda = 1 V rows on the folds with lambda* = 1 V (sky130 10 of 18 folds,
  IHP 300 K 3 of 18; difference 0, checked while writing this file). No paired GMR(fixed lambda, selected) value was
  computed before this file. Folds with lambda* = 1 V: IHP 233 K 4 / 18, 343 K 8 / 18, 398 K 2 / 18 (I21-1); NIST
  lvt 17 / 53 (selection blocks 4 and 5, I20-1); NIST RT card 0 / 87; TU Delft 0 / 7 groups at both temperatures.

The two parts are independent; each has its own validity checks and rule.

## Part A: B1 vs C0_refit at equal NGSpice calls on IHP SG13G2 (233 / 300 / 343 / 398 K)

Data, records and folds: as I22-1 Part A for `ihp233`, `ihp300`, `ihp343`, `ihp398` (I5-1 state-C starting records
`I5_1/start/records_<T>K`; PSP103 has no A2 clamp; ngspice-46 + psp103.osdi via the I5-1 `MeasProblem`; 18 DEV DUTs
per temperature; LODO within class). The held-out refit of every arm uses the I5-1 `refit_task` through the I22-1
driver (`i22_1_baselines.refit`), strict paper RRMS with the frozen curve inclusion of the starting record.

Call accounting: one NGSpice call = one simulation of the DUT's measured bias set, counted by `MeasProblem.n_sim`
exactly as in the I22-1 rows (start simulation, every residual and Jacobian column, the IRLS re-simulation after
each round, and the final re-simulation of the kept card). N_B1(d) = the I22-1 stored B1 `n_sim` of DUT d
(`I22_1/partA_ihp<T>/results.json`, `n_sim.B1`); N_C0(d) = the I22-1 stored C0_refit `n_sim`. The B1 edge-screen
calls (stored I5-1 screen, 1 + 2 x active candidates per DUT, counted once at 300 K) are not part of N_B1; they are
reported next to the result.

Capped refit: `polish_capped` = `ngspice_fit.polish` (TRF, irls 2, diff_step 0.01, best card kept) with a hard limit
on the NGSpice calls made inside the refit: when the limit is reached the running TRF round stops, the remaining
IRLS rounds are skipped, and the best card seen so far is kept. The limit is N - 1 calls inside polish, so the
recorded `n_sim` (including the final re-simulation) is at most N. Without a binding limit `polish_capped` is
identical to `polish` (unit test; and VA1 below).

Arms (per temperature, 18 DUTs):
- **C0_refit (rerun)**: the I22-1 C0_refit refit (max_nfev 400) through `polish_capped` with no binding limit.
  Validity VA1 only.
- **C0_call** (primary control): C0_refit (12-parameter base box, no correction) with max_nfev = N_B1(d) (TRF budget
  per IRLS round max(2, N_B1(d) // 2 // 14)) and call limit N_B1(d).
- **B1** (stored): the I22-1 B1 rows (`I22_1/partA_ihp<T>/results.json`, equal to the I5-1 rows).
- **B1_cap** (reverse control, secondary): B1 (same extras, `matched_nfev` from max_nfev 400) with call limit
  N_C0(d).

Statistics: GMR = exp(mean ln ratio) over the 18 paired DUTs, 95% percentile bootstrap CI, 10000 resamples, seed 0,
resampling unit DUT (`i3_2_bias.paired_ratio`). All-device mean and combined (mean of nMOS and pMOS family means)
RRMS per arm. Labels for GMR(X, Y): **X better** (CI upper < 1), **Y better** (CI lower > 1), **inconclusive**.

Primary comparison per temperature: GMR(B1, C0_call).

### Rule (Part A)

- **"B1 gain survives the call match"** iff CI upper of GMR(B1, C0_call) < 1 at all 4 temperatures.
- **"B1 gain not shown at equal calls"** iff CI upper of GMR(B1, C0_call) >= 1 at all 4 temperatures.
- **"partial"** otherwise (the temperatures with CI upper < 1 are listed).

Implication, fixed now:
- "survives": B1 (wider PSP103 card set, I5-1 screen) is entered in the method-freeze draft as a candidate IHP arm
  and gate option, as a user decision (a third gate choice besides "apply g v2" and C0_refit); the frozen method is
  not changed by this study.
- "not shown": the I22-1 IHP B1 gain is recorded as not separable from its 6x call budget; B1 is not proposed as an
  IHP arm; the freeze draft records C0_call as the IHP call-matched control.
- "partial": reported per temperature; B1 stays an open item, not proposed as an arm.

Reported without a rule: GMR(C0_call, C0_refit) per temperature (gain of C0_refit from 6x calls), GMR(C0_call,
selected g v2), GMR(B1_cap, C0_refit) and GMR(B1, B1_cap); calls used per DUT (sum, fraction of the limit, number of
DUTs where the limit binds); pooled exp(mean ln GMR) over the 4 temperatures for each comparison.

### Validity (per temperature)

- VA1: the C0_refit rerun through `polish_capped` reproduces the stored I22-1 C0_refit row on every DUT (max
  |difference| <= 1e-12) and its `n_sim` equals the stored `n_sim`.
- VA2: every arm has 18 rows; non-finite RRMS rows <= 10% per arm.
- VA3: n_sim(C0_call, d) <= N_B1(d) and n_sim(B1_cap, d) <= N_C0(d) for every DUT.
- VA4: every loaded tag is a DEV tag (`I5_1/split.json`).
- VA5: unit tests of the new code and reused modules pass.
An invalid temperature is reported and left out of the rule; with an invalid temperature the rule is "not
evaluated".

## Part B: g v2 with lambda fixed at 1 V (no nested selection) on the 9 measured DEV datasets

Arm **fixed_l1**: for every outer fold, the 5-member g v2 ensemble (seeds 0-4, the selected recipe of the
method-freeze draft section 1.C: symmetric inputs, EPS 0.01 V, 3-16-16-1 tanh network, Adam lr 3e-3, 2000 steps,
L2 1e-4, the dataset's penalty grids) trained with lambda = 1 V on the fold's training records in the same order as
the selected g v2 arm, frozen in the held-out refit (polish max_nfev 400, irls 2, diff_step 0.01), strict paper RRMS,
frozen inclusion. Datasets, records, folds, simulators and state C exactly as the selected g v2 rows they are
compared with (I22-1 table): sky130 77 K (18, LODO), NIST lvt (53, LODO), NIST RT card (87, LODO), IHP 233 / 300 /
343 / 398 K (18 each, LODO within class), TU Delft 4.13 / 295.0 K (45 each, LOGO over 7 geometry groups, cluster
bootstrap). No lambda selection is run; the nested-selection inner refits are not part of this arm.

Row sources:
- **Reused stored rows** (same recipe, folds, training order and state as the selected arm, because the selected
  rows were composed from them): sky130 `I15_2/lodo_sym_e1_l1`; IHP 300 K `I17_2/lodo_sym_e1_l1`; TU Delft
  `I18_2/lodo_4K_sym_e1_l1`, `I18_2/lodo_295K_sym_e1_l1`; NIST lvt: the 17 devices with lambda* = 1 V in
  `I20_1/lodo_nist4` (their I20-1 training ran at lambda = 1 V on the same LODO training set).
- **New runs**: IHP 233 / 343 / 398 K, all 18 folds each (drivers of I17-2 / I21-1: `i17_2_ihp.train_sym_task`,
  `refit_task`, penalty grids `I21_1/ihp_<T>K/pens`); NIST lvt, the 36 devices with lambda* = 0 V; NIST RT card,
  all 87 devices (I20-1 driver: `train_v2_task`, `refit_v2_task`, penalty grids `I20_1/pens`, state C; I23-1 cache
  directory, not the I20-1 cache).
- **Reproduction checks of reused rows (VB1)**: retrain (no cache) and refit a sample: sky130 folds nmos_L1_W1p6 and
  pmos_L2_W5; IHP 300 K folds lv_nmos_W0p15_L0p18 and hv_pmos_W0p6_L0p5; TU Delft group t40_nmos_W0p36_L0p04 at both
  temperatures (its 2 devices); NIST lvt the first nMOS and the first pMOS (sorted tag) of the 17 reused devices.
  Pass: |RRMS rerun - stored| <= 1e-9 on every sampled device. If a dataset fails, its reused rows are replaced by a
  full rerun of that dataset's fixed_l1 arm (reported as such).
- **Overlap checks of new runs (VB2)**: on the folds where the stored lambda* = 1 V (IHP 233 K 4, 343 K 8, 398 K 2
  folds), the new fixed_l1 row must equal the stored selected row (|difference| <= 1e-9).

Comparison rows: selected g v2, C0_refit and M_bias v1 as loaded by `plot_freeze_draft.datasets()` (the I22-1
comparison rows).

Statistics: as Part A (GMR, device bootstrap 10000, seed 0; TU Delft the 7 geometry groups as cluster units);
all-device mean and combined RRMS per arm and dataset; pooled exp(mean ln GMR vs C0_refit) over the 9 datasets
(equal weight per dataset), for fixed_l1 next to the I22-1 pooled values (selected g v2 0.957, B1 0.913).

Per dataset: GMR(fixed_l1, selected g v2) with the labels **fixed better** (CI upper < 1), **selected better** (CI
lower > 1), **inconclusive**, and the flag **non-inferior** iff CI upper < 1.05 (margin of I15-2 / I20-1 / I21-1 /
I22-1); GMR(fixed_l1, C0_refit) with the labels gain / worse / inconclusive; GMR(fixed_l1, M_bias v1) reported.

### Rule (Part B)

Gain set G = {sky130, NIST lvt, NIST RT card} (the datasets where the selected g v2 gains over C0_refit; I22-1).
- **"fixed lambda 1 V non-inferior to the selected g v2"** iff fixed_l1 is non-inferior on all 3 datasets of G.
- **"fixed lambda 1 V not shown non-inferior"** otherwise (the failing datasets are listed).
If a dataset of G is invalid the outcome is "not evaluated". Datasets outside G are reported without a rule.

Implication, fixed now: "non-inferior" -> the method-freeze draft lists "g v2 with lambda fixed at 1 V" as a proposed
replacement of the nested lambda selection (user decision; the frozen method is not changed by this study), with
its NGSpice-call cost next to the selected arm; "not shown" -> the nested selection stays in the draft and the
fixed-lambda arm is reported as a cost / accuracy trade-off.

### Cost (Part B, report only)

NGSpice calls per held-out device of fixed_l1: held-out refit `n_sim` (from the stored or new rows) + 0 selection
calls; penalty-grid decks as the selected arm (counted as decks, not calls); trainings = 5 per fold; training CPU
seconds as stored or measured. Reported next to the I22-1 cost table (`I22_1/cost.json`: C0_refit, selected g v2,
B1) as the ratio to the C0_refit refit calls.

### Validity (per dataset)

- VB1 (reused datasets): reproduction sample as above.
- VB2 (new IHP runs): overlap check as above.
- VB3: fixed_l1 has a row for every DEV device; non-finite rows <= 10%.
- VB4: every loaded tag is a DEV tag of the dataset's split file.
- VB5: unit tests of the new code and reused modules pass.

## Predictions (scored in the outcome entry)

- P1 Part A rule: "B1 gain survives the call match".
- P2 GMR(C0_call, C0_refit) < 1 by point estimate at all 4 temperatures.
- P3 GMR(B1_cap, C0_refit) > 1 by point estimate at 3 or more temperatures.
- P4 Part B rule: "fixed lambda 1 V non-inferior to the selected g v2".
- P5 fixed_l1 gains over C0_refit on sky130, NIST lvt and NIST RT card; is not a gain at any IHP temperature.
- P6 fixed_l1 NGSpice calls per held-out device within 1.1x of C0_refit on every dataset.

## Order of operations

1. This file, `prereg.json` and the RESEARCH_LOOP pre-registration entry, committed alone; push.
2. Code (`src/cryoml/explore/i23_1_budget.py`, `scripts/research/explore/i23_1_budget.py`), unit tests, smoke (2 DUTs
   / folds per stage, reduced budgets, output under `I23_1/smoke`); commit.
3. Queue, one job at a time (<= 10 workers; OMP / MKL / OpenBLAS / vecLib threads 1 and `torch.set_num_threads(1)`
   in every worker): Part A per temperature; Part B reproduction checks; IHP fixed_l1 per temperature; NIST lvt;
   NIST RT card; decide.
4. decision.json, report batch `out/research/report/2026-10-<dd>_i23_1/`, site claim, RESEARCH_LOOP outcome, freeze
   draft (open item 12, IHP and cost rows), memory status line; commit and push.

Compute estimate: Part A about 30 min (C0_call about 35,000 calls per temperature). Part B: IHP about 15 min per
temperature; NIST lvt 180 member trainings at about 330 s (about 1.7 h at 10 workers); NIST RT card 435 member
trainings at an estimated 450-550 s (about 6 h at 10 workers); reproduction checks about 20 min.
