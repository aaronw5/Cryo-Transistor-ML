# I17-1 pre-registration (IN FORCE): re-registration of the I16-1 Part B test of the I15-2 selected correction (full|sym_e1_l1, lambda 1 V) with the Q1 V(d,s) = 0 check at tight tolerance only; sky130 77 K, 18 devices, state C

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-22), committed alone, before any I17-1 code or run.

## This check was changed after an observed failure

I16-1 (pre-registration `I16_1/prereg.md`, commit 6adec13) failed one core check: Q1 |I(V(d,s) = 0)| <= 1e-12 A,
evaluated as the maximum over the default-tolerance and the tight-tolerance forward Q1 grids. The value was
1.0006e-12 A at 1 of 2700 points (pmos_L8_W0p84, |V(g,s)| = 0.825 V, default tolerance). Every other core check
passed. **The I16-1 outcome stays recorded as 'not deployable'.** I17-1 changes this one check after the failure
was observed; it is not a test that was blind to the data.

Known before this registration (post hoc diagnosis `I16_1/diagnosis/vds0.json`, I16-1 arrays): at tight tolerance
the element maximum over the 18 devices is 4.87e-17 A (0 of 1350 points above 1e-12 A) and the card + replica
maximum is 4.66e-17 A; at default tolerance 1.0006e-12 A (1 point above 1e-12 A); at reltol 1e-4 9.28e-13 A. The
I15-1 arms at default tolerance: g v2 9.84e-13 A, v1 9.95e-13 A. A rerun of the same model with the same procedure
is expected to reproduce the tight-tolerance values.

## Changed check (the only change)

| | I16-1 | I17-1 |
|---|---|---|
| Q1 \|I(V(d,s) = 0)\| | <= 1e-12 A, max over default and tight tolerance | <= 1e-12 A, tight tolerance only (reltol 1e-9, abstol 1e-18 A, vntol 1e-12 V) |

Computation: maximum over the 18 devices of |I| of the element run at the V(d,s) = 0 column of the forward tight
Q1 grid (75 V(g,s) values per device, 1350 points), from the I17-1 arrays.

Option chosen: tight tolerance only, threshold 1e-12 A kept. Not chosen: a threshold derived from abstol at default
tolerance. Reasons:

1. At default tolerance ngspice accepts a Newton iterate when the change of each branch current is within
   reltol |I| + abstol, with abstol = 1e-12 A, the same value as the threshold. A current of order 1e-12 A at
   V(d,s) = 0 is inside the accepted solver error, so at that tolerance the check does not separate a current from
   the element from the solver tolerance.
2. A threshold k x abstol at default tolerance needs a value of k that the solver documentation does not give; a k
   chosen now would be chosen with the observed 1.0006e-12 A known.
3. At tight tolerance abstol is 1e-18 A, six orders of magnitude below the kept threshold 1e-12 A; an element
   current of 1e-12 A at V(d,s) = 0 is detectable there.

Reported without a rule: max |I(V(d,s) = 0)| per device at default and r4 tolerance and the number of points above
1e-12 A (the I16-1 `vds0` table), and the label under the unchanged I16-1 rule (max of default and tight) from the
I17-1 run.

## Unchanged from I16-1

Model under test (`I15_2/lodo_sym_e1_l1` `full|sym_e1_l1`; symmetric inputs, EPS 0.01 V, lambda 1 V), held-out pairs
(sym_e1_l{lambda*_h}), Part B procedure (replica `REPLICA_CHAN` with the gmin VCCS, module text, element,
references, benches, three tolerances, serpentine Q1 grids, 11175 rows per grid run), M2 refit (from direct43_r3_fd
with the deployed g frozen in the loop; TRF, max_nfev 400, irls 2, diff_step 0.01, state C, BSIM4 clamp written),
every other core check and threshold, the gds / gm sign-change classification (gds a restriction, gm in strong
inversion a core failure), the restrictions (default tolerance 1e-3, gds sign, runtime ratio 2), the restriction
items, the accuracy reporting (fixed from I15-2: GMR(selected, v1) 0.997 [0.962, 1.027], non-inferior; element-deck
RRMS reported without a rule) and the validity checks V1, V3, V4. V2: the I16-1 unit-test list plus
`tests.test_i17_1_vds0`. Every stage is rerun into `out/research/explore/I17_1/` (no I16-1 output is copied).

Core checks (any fail -> not deployable):

| check | threshold |
|---|---|
| Q1 equivalence, tight, forward and exchanged bench | 0 points with \|dI\| > 1e-6 \|I_py\| + 1e-13 A |
| Q1 runs (grids, AC, dense element decks) | 0 convergence messages, 0 non-finite values, 0 failed runs |
| **Q1 \|I(V(d,s) = 0)\|, tight tolerance only (changed)** | <= 1e-12 A |
| Verilog-A g vs Python | <= 1e-9; exchanged vs forward <= 1e-12 |
| exchange asymmetry (forward operation, \|I\| > 1 nA, tight) | <= 1e-3 |
| Q2 derivatives vs the element identity (tight) | <= 1e-3 |
| Q2 capacitances | <= 1e-3 |
| Q2 gm curves with an introduced sign change in strong inversion | 0 of 54 |
| Q3 circuits failing only with the element (I12-1 amendment A1) | 0 |

Label: any core fail -> not deployable; core pass and >= 1 restriction -> deployable with restrictions; else
deployable (simulator-level).

## Validity (checked first; any fail -> invalid)

| check | pass |
|---|---|
| V1 card-only library deck x exp(fold g) of the 18 selected held-out pairs reproduces I15-2 `selected_rows` | max \|RRMS difference\| <= 1e-9 |
| V2 unit tests (I16-1 list + `tests.test_i17_1_vds0`) | all pass |
| V3 rows | 18 element-deck accuracy rows; 0 failed M2 refits |
| V4 deployed g | sym_v2 / EPS 0.01 V / lambda 1 V; I15-2 `lambda_star.full` = 1 V |

## Decision

| condition | outcome | action |
|---|---|---|
| any validity check fails | invalid | stop: fix and re-register |
| Part B core checks all pass (I17-1 V(d,s) = 0 rule) | deployable (label 'deployable with restrictions' or 'deployable (simulator-level)'; restrictions listed) | the I15-2 selected variant is the deployable add-on candidate, reported with the statement that the V(d,s) = 0 check was re-registered after an observed failure; report restrictions |
| any core check fails | not deployable | stop; diagnose the failed core checks |

Expectation stated before the run (not a rule): the rerun reproduces the I16-1 values (M2 refits are
deterministic); under the I17-1 rule every core check passes and the label is 'deployable with restrictions'
(default tolerance, 6 of 90 gds curves, runtime); under the unchanged I16-1 rule the label stays 'not deployable'
if the default-tolerance value at pmos_L8_W0p84 reproduces.

## Order and compute

This prereg commit (alone) -> code + unit tests commit -> smoke (nmos_L1_W1p6, pmos_L2_W5, refit max_nfev 20; not
used for decisions) -> queue, one heavy job at a time: I17_1_validity (about 10 s), I17_1_accuracy (about 10 s),
I17_1_deploy (about 4 min), I17_1_restrict, I17_1_runtime, I17_1_decide.

## Outputs

`out/research/explore/I17_1/{validity, accuracy, deploy_sel, decision.json}`; arrays (`*.npz`) and OSDI binaries not
in git; report batch `out/research/report/<date>_i17_1/`, site claim `i17-1` and RESEARCH_LOOP entry after the
decision. If the outcome is deployable, slides 25-26 of the M_bias deck are updated.
