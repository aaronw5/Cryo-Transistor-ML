# I16-1 pre-registration (IN FORCE): deployment test (Part B, I15-1 rules) of the I15-2 selected bias-only correction (symmetric g, nested-selected penalty lambda; full-data lambda* = 1 V); sky130 77 K, 18 devices, state C

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-22), committed alone, before any I16-1 code or run.

## Background

I15-1 tested g v2 (symmetric inputs, EPS = 0.01 V, penalty lambda = 10 V; `I13_1/lodo` `full|M_bias_v2`) with the
I14-1 channel-current replica element and serpentine Q1 grids: every core check passed; restrictions default
tolerance 1.81e-3, 2 of 90 gds curves (pmos_L8_W0p84, pmos_L8_W1p6 at V(g,s) = -1.85 V; the same points in
I_card exp(g), so the sign change comes from g), runtime ro5_A 4.30; label 'deployable with restrictions'. I15-2
selected lambda per outer fold by nested 3-fold cross-validation on the training devices: lambda* = 1 V on 10 folds,
2 V on 6, 0 V on pmos_L4_W7, 10 V on pmos_L8_W5; full-data lambda* = 1 V. Primary GMR(selected, M_bias v1)
0.997 [0.962, 1.027]: non-inferior. The I15-2 penalty-grid fractions of negative corrected gds are higher at
lambda = 1 V than at 10 V (full-data model on the 18 starting-card grids: 1.59 % vs 0.23 %; held-out grids, selected
vs lambda 10: 4.7 % vs 1.3 %). A lambda = 1 V model has not been tested in a simulator.

Done before this registration: no I16-1 run. No deck, compile, M2 refit or accuracy run with a lambda = 1 V, 2 V or
0 V correction. The I15-2 rows, corrections and held-out cards are the inputs.

## Model under test (fixed)

- Deployed g (Part B): the full-data correction of the I15-2 variant sym_e1_l1 (`I15_2/lodo_sym_e1_l1/corrections.json`
  `full|sym_e1_l1`; 5 seeds, 18 training devices, symmetric inputs, EPS 0.01 V, lambda 1 V). Same Verilog-A
  module text generator (`i14_1_chan.to_verilog_a`), module name and element as I15-1.
- Held-out pairs (accuracy, reported): for device h, the fold correction and held-out card of the variant
  sym_e1_l{lambda*_h}: lambda* = 1 V (`I15_2/lodo_sym_e1_l1`), 2 V (`I15_2/lodo_sym_e1_l2`), 0 V
  (`I15_2/lodo_sym_e1_l0`), 10 V (stored g v2: `I13_1/lodo` `M_bias_v2|<tag>`, `heldout_cards.json`). lambda*_h from
  `I15_2/decision.json` `lambda_star`. Python rows: `I15_2/decision.json` `selected_rows`.

## Part B (I15-1 procedure, checks, thresholds and label unchanged)

Replica (I14-1 `REPLICA_CHAN` with the gmin VCCS), module, references, benches, tolerances and serpentine Q1 grid
sweep (`i15_1_serp.grid_control_serp`; a grid run with a row count other than 11175 is a failed run) as I15-1.
Card M2 per device: refit from the starting card (direct43_r3_fd) with the full-data sym_e1_l1 g frozen in the loop
(TRF, max_nfev 400, irls 2, diff_step 0.01, state C); in-sample.

Core checks (any fail -> not deployable):

| check | threshold |
|---|---|
| Q1 equivalence, tight, forward and exchanged bench, reference I_card +- (exp(g) - 1) I_rep from the card + replica run of the same order and tolerance | 0 points with \|dI\| > 1e-6 \|I_py\| + 1e-13 A |
| Q1 runs (grids, AC, dense element decks) | 0 convergence messages, 0 non-finite values, 0 failed runs |
| Q1 \|I(V(d,s) = 0)\|, default and tight | <= 1e-12 A |
| Verilog-A g vs Python (2000 points, `default_rng(20261021)`) | <= 1e-9; exchanged vs forward <= 1e-12 |
| exchange asymmetry (forward operation, \|I\| > 1 nA, tight) | <= 1e-3 |
| Q2 derivatives vs the element identity (tight) | <= 1e-3 |
| Q2 capacitances | <= 1e-3 |
| Q2 gm sign: gm curves (3 V(d,s) x 18 devices = 54) with an introduced sign change in strong inversion (\|V(g,s)\| >= V_T + 0.2 V; AC or 1 mV dense sweep) | 0 of 54 |
| Q3 circuits (inv_A, inv_B, ro5_A, mirror_n, mirror_p) failing only with the element (I12-1 amendment A1) | 0 |

Restrictions (reported; do not change the deployable / not deployable outcome):

| restriction | condition |
|---|---|
| default tolerance | Q1 max relative difference at \|I_py\| > 1 nA > 1e-3 |
| gds sign | >= 1 of 90 gds curves (5 V(g,s) x 18 devices) with an introduced sign change (\|V(d,s)\| >= 0.05 V) |
| runtime | wall-time ratio element / card > 2 (DC grid or circuits) |

**Classification of sign changes (kept from I15-1):** an introduced gds sign change is a **restriction**, not a
core failure; an introduced gm sign change in strong inversion is a **core failure**. Expectation stated before the
run (not a rule): more gds curves with an introduced sign change than the 2 of 90 at lambda = 10 V (I15-1), since
the lambda = 1 V full-data model has 1.59 % negative corrected gds on the penalty grids against 0.23 % at 10 V.

Label: any core fail -> not deployable; core pass and >= 1 restriction -> deployable with restrictions; else
deployable (simulator-level).

### Restriction items (reported per device; stage `restrict`)

- Q1 max relative difference at |I_py| > 1 nA per device at default, r4 and tight tolerance.
- gds, per device and V(g,s) in {0.6, 0.9, 1.2, 1.5, 1.85} V (x sign): introduced-sign-change flag (AC or dense) and
  the number of 1 mV dense-sweep points with dI/dV(d,s) < 0 at |V(d,s)| >= 0.05 V for the element, the card and the
  Python product I_card exp(g) of the card sweep; V(d,s) range of the negative points; g range on the curve.
- gm, per device and V(d,s) in {0.05, 0.925, 1.85} V (x sign): introduced-sign-change flag (core item) and the number
  of dense-sweep points with dI/dV(g,s) < 0 in strong inversion (|V(g,s)| >= V_T + 0.2 V) and over the whole sweep,
  for the element, the card and I_card exp(g).
- The same gds and gm counts for the I15-1 g v2 (lambda 10 V) arm from its stored dense arrays (reference column).
- Runtime: per-device grid ratios and circuit ratios; ro5_A decomposition (card, card + replica, element; wall time,
  rusage load time) on the I16-1 M2 cards.

## Accuracy (fixed from I15-2; element deck reported without a rule)

Registered accuracy result: I15-2 primary GMR(selected, M_bias v1) 0.997 [0.962, 1.027], 7/18 lower, non-inferior
(margin 1.05). Not rerun. Reported: element-deck RRMS (strict paper RRMS, published-card curve inclusion) of the 18
held-out pairs (fold g of sym_e1_l{lambda*_h} as an OSDI element with its held-out card), the Python rerun
(I_card exp(g)) and the channel-model Python rerun (I_card + (exp(g) - 1) I_rep) from card + replica runs, per-device
differences to the I15-2 Python rows, all-device / combined / median, GMR(element selected, M_bias v1 stored),
GMR(element selected, C0_refit), GMR(element selected, I15-1 element g v2), device bootstrap as I15-2.

## Reference values (not rerun)

I15-1 g v2 arm (lambda 10 V, label 'deployable with restrictions') and I15-1 v1 reference arm ('not deployable'):
values from `I15_1/deploy_v2/summary.json`, `I15_1/deploy_v1/summary.json`, `I15_1/deploy_v2/restrictions.json`.

## Validity (checked first; any fail -> invalid)

| check | pass |
|---|---|
| V1 card-only library deck (OSDI build, clamp written) x exp(fold g) of the 18 selected held-out pairs reproduces the I15-2 `selected_rows` | max \|RRMS difference\| <= 1e-9 |
| V2 unit tests (I15-1 list + `tests.test_i15_2_ablate` + `tests.test_i16_1_sel`) | all pass |
| V3 rows | 18 element-deck accuracy rows; 0 failed M2 refits |
| V4 deployed g | the full-data correction loaded for Part B has inputs sym_v2, EPS 0.01 V and lambda 1 V in `lodo_sym_e1_l1/results.json`, and I15-2 `lambda_star.full` = 1 V |

## Decision

| condition | outcome | action |
|---|---|---|
| any validity check fails | invalid | stop: fix and re-register |
| Part B core checks all pass | deployable (label 'deployable with restrictions' or 'deployable (simulator-level)'; restrictions listed) | the I15-2 selected variant is the deployable add-on candidate; report restrictions |
| any core check fails | not deployable | stop; diagnose the failed core checks |

## Order and compute

This prereg commit (alone) -> code + unit tests commit -> smoke (2 devices nmos_L1_W1p6 and pmos_L2_W5, refit
max_nfev 20; not used for decisions) -> queue, one heavy job at a time: I16_1_validity (about 10 s), I16_1_accuracy
(about 20 s), I16_1_deploy (about 4 min), I16_1_restrict (about 15 s), I16_1_runtime (about 15 s), I16_1_decide. The
queue is empty at registration (QUEUE EMPTY 07:22:48).

## Outputs

`out/research/explore/I16_1/{validity, accuracy, deploy_sel, decision.json}`; arrays (`*.npz`) and OSDI binaries not
in git; report batch `out/research/report/<date>_i16_1/`, site claim `i16-1` and RESEARCH_LOOP entry after the
decision.
