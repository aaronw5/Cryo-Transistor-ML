# I25-1 element checks (prereg section 10): pre-registration (IN FORCE)

In force from the commit that adds this file (2026-10-28), committed alone, before any element-check code or run.
I25-1 prereg section 10 lists these checks as an optional last stage for arm B; this note runs them on both arms
(user instruction, 2026-10-28). The I25-1 outcome and every I25-1 number stay unchanged; the checks below are a
separate result.

## Models under test (one Part B run per arm)

| | arm A (7 parameters) | arm B (45 parameters) |
|---|---|---|
| deployed g | `I25_1/armA/full/corrections.json` `full` | `I25_1/armB/full/corrections.json` `full` |
| g recipe | sym_v2 inputs, EPS 0.01 V, lambda*_full 5 V (`inner/inner.json`) | same |
| M2 cards (card + g pairs) | `armA/full/results.json` `rows[tag].params` | `armB/full/results.json` `rows[tag].params` |
| LODO pairs (accuracy only) | `armA/lodo` cards + `lodo|<tag>` fold g | `armB/lodo` cards + `lodo|<tag>` fold g |
| C0_refit cards (Q3 reference) | `armA/c0/results.json` | `armB/c0/results.json` |

The M2 cards are the I25-1 full-data in-sample refits (state C, g frozen in the loop, TRF max_nfev 400, irls 2,
diff_step 0.01; the I17-1 M2 refit options). They are not refit again. Template `CRYOML_PDK_TEMPLATE=mult1` (the
released 77 K card files with every multiplier set to 1, `data/processed/pdk77k_mult1/`), BSIM4 clamp written
(state C). Simulator: ngspice-41 OSDI build `/Users/anrunw/cryo-ng41/arm64-osdi/bin/ngspice`, OpenVAF 23.5
`/Users/anrunw/cryo-ng41/openvaf/bin/openvaf`; Verilog-A sense branch `V(sense) <+ RS*I(sense)` with RS = 0.

## Procedure (I17-1, unchanged)

Code reused: the I14-1 / I15-1 / I16-1 / I17-1 drivers with the inputs above rebound. Channel-current replica
`REPLICA_CHAN` with the gmin VCCS, VCVS terminal copies, OSDI element `I_card + (exp(g) - 1) I_rep`; serpentine Q1
grids (forward and exchanged bench; default, r4 and tight tolerance); Q2 AC at the op points and dense 1 mV gm / gds
sweeps; Q3 circuits inv_A_vtc, inv_B_vtc, ro5_A, ro5_B, mirror_n, mirror_p with arms template card (multiplier-1
card; replaces 'published'), C0_refit, M2_card, M2_full. Q1 V(d,s) = 0 check at tight tolerance only (I17-1 rule).

## Core checks (any fail -> not deployable), thresholds copied from I17-1

| check | threshold |
|---|---|
| Q1 equivalence, tight, forward and exchanged bench | 0 points with \|dI\| > 1e-6 \|I_py\| + 1e-13 A |
| Q1 runs (grids, AC, dense element decks) | 0 convergence messages, 0 non-finite values, 0 failed runs |
| Q1 \|I(V(d,s) = 0)\|, tight tolerance only (reltol 1e-9, abstol 1e-18 A, vntol 1e-12 V) | <= 1e-12 A |
| Verilog-A g vs Python (2000 random points) | <= 1e-9; exchanged vs forward <= 1e-12 |
| exchange asymmetry (forward operation, \|I\| > 1 nA, tight) | <= 1e-3 |
| Q2 derivatives vs the element identity (tight) | <= 1e-3 |
| Q2 capacitances | <= 1e-3 |
| Q2 gm curves with an introduced sign change in strong inversion | 0 of 54 |
| Q3 circuits failing only with the element (M2_full fails, M2_card passes) | 0 |

Restrictions (reported in the label): Q1 default-tolerance relative difference (\|I\| > 1 nA) > 1e-3; gds curves with
an introduced sign change > 0 of 90; runtime ratio (element / card, DC grid and circuits) > 2.

Label per arm: any core fail -> not deployable; core pass and >= 1 restriction -> deployable with restrictions;
else deployable (simulator-level).

## Validity (checked first; any fail -> invalid for that arm)

| check | pass |
|---|---|
| V1 card-only library deck (OSDI build, clamp written) x exp(g) reproduces the stored I25-1 rows (`full_g` 18 rows and `lodo_g` 18 rows, primary frozen curve set) | max \|RRMS difference\| <= 1e-9 |
| V2 unit tests (I17-1 list + `tests.test_i25_1_scratch`) | all pass |
| V3 rows | 18 finite element-deck rows per row type (full, LODO); 18 M2 cards |
| V4 deployed g | sym_v2 / EPS 0.01 V / lambda 5 V; equals `inner.json` `lambda_star.full` |

## Reported without a rule

Element-deck RRMS (paper RRMS, frozen curve set) of the full-data pairs and of the LODO pairs, all-device and
combined (mean of the nMOS and pMOS family means), with the difference to the stored Python rows; the
V(d,s) = 0 values at default and r4 tolerance and the label under the I16-1 rule (max of default and tight);
gds / gm restriction items per device; ro5_A runtime decomposition. Expectation (not a rule): I17-1 on the old
43-parameter card + g (lambda 1 V) gave 'deployable with restrictions' (default tolerance 1.84e-3, 6 of 90 gds
curves, runtime 4.60); the same label is expected for both arms here.

## Order and compute

This note (alone) -> code -> smoke (2 devices, not used for decisions) -> per arm, one heavy job at a time:
validity, accuracy, deploy (compile, M2 rows, q12, q3, report), restrict, runtime, decide. Outputs
`out/research/explore/I25_1/element/arm{A,B}/`; arrays and OSDI binaries not in git.
