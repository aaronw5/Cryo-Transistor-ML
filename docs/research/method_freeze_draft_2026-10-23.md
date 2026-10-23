# Method freeze and TEST plan: draft (2026-10-23)

Status: **proposed, not frozen.** The final freeze is planned for about 2026-12-10. This file is not a
pre-registration. No TEST split was opened or evaluated for this draft; no NGSpice run and no training were done
for it. Every number below is copied from a stored file named next to it or recomputed from stored per-device rows
by `scripts/research/plot_freeze_draft.py` (88 values checked against the stored decision files).

Sources read: `docs/RESEARCH_LOOP.md` (I3-2 to I19-1), `out/research/explore/I19_1/test_gate_decisions.md`,
`out/research/explore/I8_2/prereg_test_predictions.md` (commit 08d23e3), `out/research/explore/real_split.json`,
the split files of I5-1, I6-1, I8-1, I9-1, the decision files listed in section 2, and the code cited in section 1.

## 1. Method specification

The method has five parts: (A) a starting card per device, (B) a BSIM4 card state, (C) a shared bias-only
correction g trained on training devices, (D) a held-out card refit with g frozen in the loop, (E) a dataset-level
gate that decides whether g is applied to a dataset, and (F) a Verilog-A / OSDI element that implements
I = I_card + (exp(g) - 1) I_channel in the simulator, with deployment checks.

### 1.A Starting cards (sky130 77 K: direct43_r3_fd)

| item | value | source |
|---|---|---|
| card | upstream sky130 77 K cards (nFET lvt card; pFET `update_sky130_fd_pr__pfet_01v8_lvt__tt_77k.corner.spice`), ngspice-41 arm64 `~/cryo-ng41/arm64` | `CLAUDE.md` |
| fitted parameters (43) | toxe, toxm, xj, ndep, nsd, dwg, rsh, vth0, k1, k2, k3, dvt0, dvt1, dvt1w, w0, k3b, vfb, lpe0, lpeb, vsat, ua, ub, rdsw, prwg, wr, u0, a0, a2, ags, b0, b1, voff, nfactor, cit, cdscd, eta0, dsub, pclm, pdiblc2, drout, delta, alpha0, beta0 | `out/cold_loop/full_direct/round_3/spec.json` (`selected_parameters`) |
| box | round-3 box of that spec (local coordinates in [-1, 1]) | same file |
| direct fit | chain `full_r1_direct_seeded` -> `full_r2_direct` -> `full_r3_direct`; round 3: CMA-ES 8000 evaluations (sigma0 0.3), 4 bank starts, polish max_nfev 600, irls 2, seed 20261015; 8404.6 NGSpice calls per device | `out/research/ngspice_direct/full_r3_direct/summary.json` |
| FD polish | `ngspice_fit.polish`, max_nfev 400, diff_step 0.01, irls 2; 313.8 NGSpice calls per device | `out/research/polished/direct43_r3_fd/real/summary.json` |
| stored score | all-device 0.10437 (state L); in state C 0.1109 / combined 0.1049 | same file; `out/research/explore/I13_1/summary.json` |
| records | one NGSpice simulation of each starting card on the measured bias grid | `scripts/research/explore/e1_collect_real.py`; `out/research/explore/E1/real_dev`, `E1/real_test.json` |

Other datasets (DEV evidence only; their TEST starting cards follow the same per-dataset recipe):
NIST lvt (I8-1) and NIST room-temperature card (I9-1): `polish` from the stock bin, max_nfev 3000, irls 3,
diff_step 0.01, box D6 of `out/research/explore/I8_1/prereg.md` (VTH0 x/ 3; U0, NFACTOR, VSAT, DELTA, RDSW, ETA0
x/ 100; other 36 x/ 3); IHP SG13G2: 12-parameter PSP103 set, polish from stock, max_nfev 3000, irls 3
(`out/research/explore/I5_1/prereg.json`); TU Delft: 43-parameter set on the CMOSedu BSIM4 card, polish from stock,
max_nfev 3000, irls 3 (`out/research/explore/I6_1/prereg.json`), rescored in state C by I18-2.

### 1.B BSIM4 state C (A1/A2 clamp written into the card)

- ngspice-41 `b4check.c` clamps a size-dependent A2 > 1 to 1 (and A1 to 0) only in the first analysis per model
  name per process; `b4temp.c` rebuilds the size-dependent values at every analysis. A multi-analysis deck therefore
  used two models for one card when a2 > 1 (diagnosis: I12-1, `out/research/explore/I12_1/diagnosis/diagnosis.json`).
- State C: every deck writes the clamp into the card: a2 > 1 -> a2 = 1 and a1 = 0; a2 < 0.01 -> 0.01.
  `spice_pdk.simulate_pdk(..., bsim4_state="clamped")` (default), `spice_pdk.bsim4_clamp_params`; code 837d07d
  (I13-1), extended to every multi-analysis deck builder in f40c6b7 (I14-1); TU Delft card text
  `i18_2_tud.clamp_params` (254e77d).
- Required property (unit-tested, `tests/test_i13_1_clamp.py`): published cards and cards with 0.01 <= a2 <= 1 give
  identical scores in state C and in the legacy state L (difference 0).
- Not yet applied: the NIST decks (`i8_1_nist.simulate`, `i9_1_nist_rt.simulate`) run all sweeps of a device in one
  ngspice process and do not write the clamp. Stored base a2 > 1 on 16 of 57 I8-1 starting cards and 34 of 87 I9-1
  starting cards (file check of `I8_1/start/summary.json`, `I9_1/start/summary.json`; the effective size-dependent
  value with the bin's l/w/p terms was not computed). See open item 4.1.

### 1.C Correction g v2, selected recipe

| item | value | source |
|---|---|---|
| form | I = I_card x exp(g), g = mean over 5 ensemble members of OUT_SCALE x tanh(h(z)), OUT_SCALE = 1 | `src/cryoml/explore/i12_1_symm.py` (`SymCorrection.g_x`); `e1_correction.OUT_SCALE` |
| inputs | s = +1 nMOS / -1 pMOS; a = s V(g,s), b = s V(g,d), m = sqrt(V(d,s)^2 + EPS^2); x = (is_pmos 0/1, x1 = (a + b + m) / (2 VDD), x2 = m / VDD) | `i12_1_symm.sym_inputs` |
| EPS | 0.01 V (smooth abs) | `i12_1_symm.EPS`; I15-2 contrast EPS 0 -> 0.01: GMR 0.994 [0.982, 1.003] |
| VDD (input scale) | sky130 1.85 V; IHP class supply 1.2 V (LV) / 3.3 V (HV); TU Delft 1.1 V | `VDD_NORM`; `i17_2_ihp`; `i18_2_tud.VDD` |
| standardization | x1, x2 by training-point mean / std; is_pmos kept 0/1 | `fit_v2` |
| network | 3-16-16-1, tanh hidden layers, last layer zero-initialized (weights and bias), float64 | `HYPER_V2`, `fit_v2` |
| seeds | 0, 1, 2, 3, 4 (one member each; g = member mean) | `HYPER_V2` |
| loss | E1 loss: mean over training devices of the per-device paper RRMS of I_sim x exp(g) against the measured current (curve RRMS = sqrt(mean squared error) / curve denominator, averaged over the device's included curves) + 1e-4 x sum of squared weight-matrix entries + lambda x P | `e1_correction._loss_fn`, `fit_v2` |
| penalty P | on a 25 mV forward-quadrant grid of each training device's starting card (sky130 75 x 75, 0..1.85 V; IHP 75 x 75 to the class supply; TU Delft 45 x 45, 0..1.1 V), points with u_d > 0 and forward card current > 1e-9 A: r_gm = gm_card / I_card + dg/du_g, r_gds = gds_card / I_card + dg/du_d (card terms by central differences, dg/du by autograd); P = mean over devices of mean(relu(-r_gm) + relu(-r_gds)) (1/V) | `i12_1_symm.penalty_arrays`, `fit_v2`; `i18_2_tud` (45 x 45) |
| penalty grid of lambda | {0, 1, 2, 5, 10} V | `i15_2_ablate.GRID_LAMBDA` |
| nested lambda selection | per outer fold: training devices sorted by (polarity, nMOS first; tag), device i -> inner fold i mod 3; per inner fold and lambda one member (seed 0) trained on the other inner folds (training records in load order), inner held-out refits with g frozen (outer refit settings); score = mean ln RRMS over inner held-out devices (non-finite counts as 0); lambda* = argmin, ties within 1e-12 to the larger lambda. TU Delft: inner folds over geometry groups. Full-data lambda*: the same selection over all training devices of the dataset | `i15_2_ablate.inner_folds`, `inner_score`, `select_lambda`; `i18_2_tud` |
| optimizer | Adam, lr 3e-3, 2000 full-batch steps per member | `HYPER_V2` |
| training data | post hoc on the starting-card records of the training devices (cards fixed); training records in load order (I12-1 amendment A2, `i12_1_symm.fold_train_tags`) | `fit_v2`; 28b6c14 |
| commits | I12-1 prereg 3619a2b, code e7bf9b6, A2 28b6c14; I15-2 prereg 7924961, code d9e79ac | `docs/RESEARCH_LOOP.md` |

Selected lambda values stored: sky130 full-data lambda* 1 V (per fold 1 V x10, 2 V x6, 0 V x1, 10 V x1;
`I15_2/decision.json`); IHP 300 K per fold 0 V x2, 1 V x3, 2 V x6, 5 V x2, 10 V x5, no full-data value stored
(`I17_2/decision.json`); TU Delft full-data lambda* 10 V (4.13 K), 5 V (295.0 K) (`I18_2/decision.json`).

### 1.D Held-out card refit with g frozen

| item | value | source |
|---|---|---|
| optimizer | `ngspice_fit.polish`: scipy `least_squares(method="trf")` on the active parameters in local coordinates, bounds [-1, 1], xtol = ftol = gtol = 1e-10 | `src/cryoml/ngspice_fit.py` |
| settings | max_nfev 400, irls_rounds 2, diff_step 0.01 | I3-2 recipe; `I8_2/prereg_test_predictions.md` |
| Jacobian | forward differences with an absolute step of 0.01 in local units (sign flipped at the upper bound) | `polish.jacobian` |
| budget per IRLS round | max(2, max_nfev // irls // (n_active + 2)) TRF evaluations | `polish` |
| IRLS weights | per curve 1 / sqrt(max(curve RRMS, 1e-3)), normalized so that sum of squared weights = n_curves | `polish` |
| kept card | the card with the lowest mean included-curve RRMS seen in any residual call | `polish` |
| correction in the loop | residuals of I_sim(card) x exp(g) against the measured curves, g frozen | `CorrectedProblem` (I12-1); I3-2 `i3_2_bias` |
| score | fresh NGSpice simulation of the kept card, x exp(g), strict paper RRMS: mean over the frozen included curves of trimmed RMSE / trimmed mean measured current | `e1_correction.score_device` |
| C0_refit | the same refit without g (control arm) | I3-2 |

### 1.E Dataset-level S3 gate

- Definition (I19-1, prereg 45b4f37, code 4da313c, `src/cryoml/explore/i19_1_dataset_gate.py`):
  S3_D = mean over all DEV records r of the dataset of ln(RRMS(r, I_start x exp(g_{-k(r)})) / RRMS(r, I_start)),
  where g_{-k(r)} is trained with the gated arm's recipe (seed 0, 500 steps; g v2: lambda = lambda*_k of fold k) on
  the training set of the leave-one-group-out fold that holds out the record's group k(r) (IHP: same class), and is
  applied to the starting-card record without a refit.
- Decision: apply the correction to every device of the dataset iff S3_D < 0 (strict); otherwise C0_refit for every
  device. Threshold 0 was fixed before any S3_D value (`THRESHOLD = 0.0`). The I7-1 OLS zero crossing +0.02455 is a
  reported sensitivity column only (`SENS_THRESHOLD`).
- Leakage stated in I19-1: the scored DEV devices' own starting records enter S3_D (through the folds where they are
  training devices). At TEST the gate uses DEV records only.

### 1.F Verilog-A element and deployment checks

Element construction (I14-1, code f40c6b7, `src/cryoml/explore/i14_1_chan.py`; module text
`i12_1_symm.to_verilog_a_v2`):

1. Card instance unchanged.
2. Channel-current replica: the same subcircuit and bins (sky130: suffix `_ch`; TU Delft: top-level `.model i6card_ch`,
   `i18_2_partb.REPLICA`) with `REPLICA_ZERO` (xpart = -1; cgso = cgdo = cgbo = cgsl = cgdl = cf = 0; cjs = cjd = cjsws
   = cjswd = cjswgs = cjswgd = 0) and `REPLICA_CHAN` (alpha0 = alpha1 = beta0 = 0 with their l/w/p terms; agidl = agisl
   = 0 with l/w/p terms; jss = jsd = jsws = jswd = jswgs = jswgd = 0; igcmod = igbmod = 0). Both card and replica carry
   the state-C clamp.
3. Replica terminals driven by unit-gain VCVS copies of V(d), V(g), V(s), V(b); a VCCS of value -m x gmin
   (m = 1, gmin = 1e-12 S, ngspice default; decks may not set gmin) between the replica drain and body copies cancels
   the junction gmin current.
4. OSDI module `N<name> d g s rx ri <model>`: sense branch `V(sense) <+ RS * I(sense)` with RS = 0 (a 0 V source in
   series with the replica drain), output `I(d, s) <+ (exp(g) - 1) * I(sense)`; g evaluated from the terminal
   voltages with the same inputs as in training (EPS and VDD are literals in the module).
5. Build: OpenVAF 23.5.0 (OSDI 0.3) at `~/cryo-ng41/openvaf/bin/openvaf --target aarch64-apple-darwin`; ngspice-41
   built with `--enable-osdi` at `~/cryo-ng41/arm64-osdi/bin/ngspice` (bit-identical to the canonical build on 36828
   of 36828 published-card currents; `out/tables/simulator_compare_ng41_arm64_vs_arm64_osdi.json`; I3-2 F1c, 3713710).
   IHP (PSP103) uses ngspice-46 + psp103.osdi, OSDI 0.4; no element exists for it (open item 4.2).

Deployment checks (I12-1 `decide_deploy`, thresholds `i12_1_symm.THR`; Q1 grids swept in serpentine order, I15-1
`i15_1_serp.py`, prereg 04c7c2e; V(d,s) = 0 check at tight tolerance only, I17-1 `i17_1_vds0.py`, prereg 3cfc9a7).
Tolerances: default (ngspice defaults), r4 (reltol 1e-4, reported), tight (reltol 1e-9, abstol 1e-18 A,
vntol 1e-12 V). sky130 Q1 grid V(g,s) 0..1.85 V x V(d,s) -1.85..1.85 V, 25 mV; TU Delft 0..1.1 V x -1.1..1.1 V.

| check | type | threshold |
|---|---|---|
| Q1 equivalence, forward and exchanged-terminal grids, tight; reference I_card + (exp(g) - 1) I_rep | core | 0 points with \|dI\| > 1e-6 \|I\| + 1e-13 A |
| Q1 convergence messages, non-finite values, failed runs in element decks | core | 0 |
| Q1 \|I(V(d,s) = 0)\| on the tight forward grid | core | <= 1e-12 A |
| same, cards with gate current (igcmod / igbmod = 1): evaluated on the element-added current \|I_elem - I_card\| | core | <= 1e-12 A (I18-2 amendment A1, 1795e10) |
| Verilog-A g vs Python g | core | <= 1e-9 |
| Verilog-A g, exchanged vs forward terminals | core | <= 1e-12 |
| element-introduced terminal-exchange asymmetry \|d ln(I_swap / I_fwd)\|, forward, \|I\| > 1 nA | core | <= 1e-3 |
| Q2 normalized gm / gds difference vs the element identity Re Y_card + (exp(g) - 1) Re Y_rep + I_rep exp(g) dg/dV (AC 1 MHz, tight) | core | <= 1e-3 |
| Q2 capacitances: \|Im Y_elem - Im Y_card\| / max(\|Im Y_card\|, 1e-15 S) for Y_dd, Y_dg, Y_gg | core | <= 1e-3 |
| gm curves with an introduced sign change in strong inversion (sky130: 54 curves) | core | 0 |
| Q3 circuits failing with the element while the card alone passes (inv_A, inv_B, ro5_A, ro5_B, mirror_n, mirror_p; ro5_B fails at card level, I12-1 amendment A1) | core | 0 |
| default-tolerance relative difference, \|I\| > 1 nA | restriction | > 1e-3 |
| gds curves with an introduced sign change (sky130: 90 curves) | restriction | > 0 |
| runtime ratio element / card (DC grid and circuits) | restriction | > 2 |

Label: any core failure -> not deployable; else any restriction -> deployable with restrictions; else deployable
(simulator-level). The V(d,s) = 0 rule was changed after an observed failure (I16-1, 1.0006e-12 A at default
tolerance on 1 of 2700 points); I16-1 stays "not deployable" and this change is reported with every use.

Stored Part B results of the selected recipe:

- sky130 77 K, full-data lambda 1 V (I17-1, `out/research/explore/I17_1/decision.json`): deployable with
  restrictions. Core values: Q1 0 violations, 0 messages, V(d,s) = 0 tight max 4.87e-17 A, g text 2.8e-16,
  exchanged 1.7e-16, asymmetry 2.2e-5, Q2 1.1e-5, capacitance 3.3e-14, gm 0 of 54, Q3 0. Restrictions: default
  tolerance 1.84e-3; 6 of 90 gds curves (pmos_L2_W5, pmos_L8_W0p84 x2, pmos_L8_W1p6 x2, pmos_L8_W5; the same points
  appear in I_card x exp(g) of the card sweep, so they come from g); runtime ratio up to 4.60 (ro5_A).
- TU Delft, full-data lambda 10 V at 4.13 K, 5 V at 295.0 K (I18-2, `out/research/explore/I18_2/partB_decision.json`):
  4.13 K not deployable (Q3 not run): one core failure, 3 convergence lines (dynamic gmin, true gmin, source
  stepping) in the dense element deck of t40_nmos_W1_L0p04_d192; the card-only dense deck of the same M2 card prints
  the same 3 lines. 295.0 K deployable with restrictions (Q3 not run): default tolerance 1.46e-3, 1 gds curve
  (t40_nmos_W1_L1_d134). Runtime ratio 1.95 / 1.93.

## 2. DEV evidence

Table PNG: `out/research/report/2026-10-23_freeze_draft/01_table_freeze_draft_dev_evidence.png` (generated by
`.venv/bin/python -I scripts/research/plot_freeze_draft.py --date 2026-10-23`). RRMS columns: all-device mean /
combined (mean of the nMOS and pMOS family means). GMR = exp(mean ln ratio) with 95% percentile bootstrap CI
(10000 resamples, seed 0; unit device; TU Delft 7 geometry groups), followed by devices lower / devices paired.

| dataset | n (unit) | C0_refit | M_bias v1 | selected g v2 | gated (I19-1) | GMR(v1, C0_refit) | GMR(selected, C0_refit) | GMR(gated, C0_refit) | GMR(selected, v1) | I19-1 decision (S3_D) | Part B label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sky130 77 K (all 18, LODO) | 18 (device) | 0.1051 / 0.0984 | 0.0735 / 0.0695 | 0.0739 / 0.0700 | 0.0739 / 0.0700 | 0.745 [0.660, 0.850] 17/18 | 0.743 [0.654, 0.855] 17/18 | 0.743 [0.654, 0.855] 17/18 | 0.997 [0.962, 1.027] 7/18 | apply (-0.133) | deployable with restrictions (I17-1, lambda 1 V) |
| NIST sky130 4 K lvt (I8-1) | 53 (device) | 0.0400 / 0.0428 | 0.0315 / 0.0337 | not run | 0.0315 / 0.0337 | 0.806 [0.758, 0.864] 48/53 | - | 0.806 [0.758, 0.864] 48/53 | - | apply (-0.103) | not tested (g v2 not run; v1 element not deployable, I11-1) |
| NIST sky130 4 K, RT card (I9-1) | 87 (device) | 0.0555 / 0.0580 | 0.0455 / 0.0480 | not run | 0.0455 / 0.0480 | 0.797 [0.748, 0.859] 77/87 | - | 0.797 [0.748, 0.859] 77/87 | - | apply (-0.069) | not tested |
| IHP SG13G2 300 K | 18 (device) | 0.0191 / 0.0191 | 0.0203 / 0.0203 | 0.0205 / 0.0205 | 0.0191 / 0.0191 | 1.214 [1.008, 1.537] 9/18 | 1.255 [1.026, 1.600] 9/18 | 1.000 [1.000, 1.000] 0/18 | 1.034 [0.990, 1.107] 7/18 | C0_refit (+0.308) | Part B not run (PSP103) |
| IHP SG13G2 233 K | 18 (device) | 0.0273 / 0.0273 | 0.0281 / 0.0281 | not run | 0.0273 / 0.0273 | 1.106 [0.908, 1.425] 11/18 | - | 1.000 [1.000, 1.000] 0/18 | - | C0_refit (+0.187) | Part B not run (PSP103) |
| IHP SG13G2 343 K | 18 (device) | 0.0190 / 0.0190 | 0.0194 / 0.0194 | not run | 0.0190 / 0.0190 | 1.181 [0.983, 1.502] 8/18 | - | 1.000 [1.000, 1.000] 0/18 | - | C0_refit (+0.367) | Part B not run (PSP103) |
| IHP SG13G2 398 K | 18 (device) | 0.0197 / 0.0197 | 0.0202 / 0.0202 | not run | 0.0197 / 0.0197 | 1.272 [1.030, 1.627] 11/18 | - | 1.000 [1.000, 1.000] 0/18 | - | C0_refit (+0.367) | Part B not run (PSP103) |
| TU Delft 4.13 K | 45 (7 groups) | 0.0602 / 0.0532 | 0.0534 / 0.0491 | 0.0484 / 0.0449 | 0.0602 / 0.0532 | 0.909 [0.766, 1.077] 30/45 | 0.826 [0.669, 1.014] 30/45 | 1.000 [1.000, 1.000] 0/45 | 0.908 [0.848, 0.965] 33/45 | C0_refit (+0.008) | not deployable (Q3 not run; lambda 10 V) |
| TU Delft 295.0 K | 45 (7 groups) | 0.0265 / 0.0251 | 0.0208 / 0.0210 | 0.0215 / 0.0218 | 0.0265 / 0.0251 | 0.811 [0.532, 1.190] 29/45 | 0.829 [0.523, 1.247] 29/45 | 1.000 [1.000, 1.000] 0/45 | 1.022 [0.975, 1.056] 22/45 | C0_refit (+0.264) | deployable with restrictions (Q3 not run; lambda 5 V) |

Pooled exp(mean ln GMR vs C0_refit), equal weight over the 9 rows: gated 0.921 (`I19_1/decision.json`
`part_R.rule.pooled_gated_D`), correction always 0.958, M_bias v1 always 0.963 (computed by the script), per-fold
gate (I18-1) 0.929.

Row sources (per-device rows; the stored decision files hold the GMRs that the script reproduces):

- sky130: `out/research/explore/I13_1/lodo/results.json` (C0_refit, state C), `I13_1/all18/results.json` (M_bias v1,
  state C), `I15_2/decision.json` (`selected_rows`, primary and C0_refit GMRs), `I17_1/decision.json` (Part B).
- NIST: `I8_1/lodo/results.json`, `I8_1/decision.json` (P_b); `I9_1/lodo/results.json`, `I9_1/decision.json` (P_b).
  4 I8-1 devices with non-finite starting records (amendment A1) are not paired.
- IHP 300 K: `I17_2/base/results.json` (C0_refit), `I5_1/lodo_300K/results.json` (M_bias v1), `I17_2/decision.json`.
  IHP 233 / 343 / 398 K: `I5_1/lodo_{233,343,398}K/results.json`.
- TU Delft: `I18_2/decision.json` (`per_temperature.{4,295}.rows`, `selected_rows`, GMRs),
  `I18_2/partB_decision.json`.
- Gate: `I19_1/decision.json` (`part_R.datasets.*`).

Evidence status: every row is DEV. sky130 rows were used for method design from I3-2 on; NIST rows (I8-1, I9-1)
were prospective for M_bias v1 when run; IHP and TU Delft rows were seen before the gate was defined; I19-1 Part R
is retrospective.

## 3. TEST plan (proposal)

### 3.1 Locked TEST splits

| TEST split | devices | temperatures | DEV counterpart | committed I19-1 decision (a755e18) | gated arm |
|---|---|---|---|---|---|
| NIST 4 K lvt (I8-1 split, seed 20261021) | 38: nfet_01v8_lvt 22 (covered 8, bin-uncovered 14), pfet_01v8_lvt 16 (covered 9, uncovered 7) | 4 K (deck 77 K card at 77 K) | 53 DEV | apply (S3_D -0.1027) | M_bias v1 |
| NIST 4 K room-temperature card (I9-1 split, seed 20261022) | 58: nfet_01v8 32, pfet_01v8 26 | 4 K (deck 30 C) | 87 DEV | apply (S3_D -0.0689) | M_bias v1 |
| IHP SG13G2 (I5-1 split, seed 20261020) | 10 DUTs: hv_nmos_W10_L10, hv_nmos_W2_L3, hv_pmos_W0p3_L0p35, hv_pmos_W10_L10, lv_nmos_W0p15_L0p13, lv_nmos_W10_L0p13, lv_nmos_W10_L0p18, lv_pmos_W0p15_L0p13, lv_pmos_W10_L0p13, lv_pmos_W10_L0p18 | 233 / 300 / 343 / 398 K (4 splits, same DUTs) | 18 DEV | C0_refit at all 4 (S3_D +0.1866 / +0.3078 / +0.3669 / +0.3667) | v1 (233, 343, 398 K); selected g v2 (300 K) |
| TU Delft 40 nm (I6-1 split, seed 20261020) | 24: t40_nmos_W1_L0p36 (8), t40_nmos_W4_L4 (8), t40_pmos_W4_L4 (8); 3 geometry groups | 4.13 K and 295.0 K (2 splits, same devices) | 45 DEV | C0_refit at both (S3_D +0.0080 / +0.2640) | selected g v2 (state C) |

Not in the plan:

- sky130 77 K: no locked TEST. The 6 REAL_TEST devices of `real_split.json` (nmos_L0p19_W7, nmos_L1_W1p6,
  nmos_L8_W1p6, pmos_L0p35_W1p6, pmos_L0p5_W0p64, pmos_L8_W0p84) were opened by user decision on 2026-10-21; all 18
  devices are reported by LODO. The I8-2 sky130 prediction cannot be checked on held-out data.
- I10-1 TEST (25 NIST ETest 1.8 V core devices): locked; I10-1 stopped (invalid run, not re-registered), so no TEST
  run is planned for it.

The 8 splits come from 4 device sets (the IHP temperatures share DUTs; the TU Delft temperatures share devices).

### 3.2 What is run per split

1. TEST starting cards: per TEST device (and temperature) with the dataset's DEV starting-card recipe (section 1.A),
   no correction. One NGSpice record per starting card.
2. Correction trained once per split on all DEV starting-card records (IHP: same class and temperature):
   M_bias v1 (I3-2 recipe) on every split; selected g v2 on IHP 300 K and TU Delft with the full-data lambda*
   (TU Delft 10 V / 5 V stored; IHP 300 K full-data selection over all 18 DEV DUTs, to be run on DEV before the
   freeze).
3. Held-out refit of each TEST device (section 1.D) for each arm: C0_refit, M_bias v1, selected g v2 (where defined).
   All arms run on every split, also where the committed decision is C0_refit, because decision correctness needs
   the sign of the TEST ln GMR(correction, C0_refit).
4. Gated arm = the committed decision of a755e18 (no recomputation of S3_D on TEST data).
5. Score: strict paper RRMS per device, frozen measured inclusion of the dataset.

### 3.3 Contrasts, bootstrap unit, decision rules (proposed; final wording in the TEST pre-registration)

- Primary per split: GMR(gated, C0_refit) with the 95% CI. Where the decision is C0_refit this is 1 by construction;
  the primary contrast is informative on the two NIST splits.
- Decision correctness per split (I19-1 rule, `test_gate_decisions.md`): correct iff the decision equals the sign of
  the TEST ln GMR(correction arm, C0_refit) (apply <-> negative).
- Secondary: GMR(correction always, C0_refit) per split; GMR(selected g v2, M_bias v1) on IHP 300 K and TU Delft;
  pooled exp(mean ln GMR vs C0_refit) over the 8 splits for gated, correction always, v1 always.
- Bootstrap unit: device for NIST and IHP. TU Delft TEST has 3 geometry groups, too few for a cluster percentile
  bootstrap; proposal: device bootstrap as the deciding unit, 3-group cluster bootstrap reported (open item 4.7).
- Proposed rule: T1 (NIST, both splits) CI upper of GMR(gated, C0_refit) < 1 with 0 failed devices beyond the
  starting-card failures; T2 pooled gated <= pooled correction always over the 8 splits; T3 decision correctness
  reported as k/8 with no threshold (8 splits from 4 device sets). Labels: confirmed (T1 on both NIST splits and
  T2), partly confirmed (T1 on one NIST split, or T1 without T2), not confirmed (T1 on neither); invalid if more than
  10% of TEST devices of a split fail in any arm or the TEST guard fails.

### 3.4 Committed predictions to be scored

| split | I8-2 (08d23e3): sign of ln GMR(M_bias v1, C0_refit), S1 rule / S3 | I19-1 (a755e18): primary decision | I19-1 v1 secondary decision |
|---|---|---|---|
| NIST 4 K lvt | none (DEV P_c gain, `I8_1/prediction_v2.json`) | apply | - |
| NIST RT card | none (DEV P_c gain, `I9_1/prediction.json`) | apply | - |
| IHP 233 K | loss / loss | C0_refit | - |
| IHP 300 K | loss / loss | C0_refit (g v2 arm) | C0_refit (S3_D +0.3114) |
| IHP 343 K | loss / loss | C0_refit | - |
| IHP 398 K | loss / loss | C0_refit | - |
| TU Delft 4.13 K | gain / gain | C0_refit (g v2 arm) | apply (S3_D -0.0292) |
| TU Delft 295.0 K | gain / gain | C0_refit (g v2 arm) | apply (S3_D -0.0510) |
| sky130 77 K | gain / gain (void: TEST opened) | - | - |

The I8-2 and I19-1 primary predictions disagree on TU Delft at both temperatures (I8-2 predicts a gain for M_bias v1;
I19-1 keeps C0_refit with the g v2 arm). Both are scored as written: I8-2 against the v1 arm, I19-1 primary against
the g v2 arm, I19-1 v1 secondary against the v1 arm.

### 3.5 Order of operations

1. Close the open items of section 4 that change the method (or record them as limitations) on DEV only.
2. Method freeze commit: this document revised to status "frozen", with the code commit of every part.
3. TEST-mode code and unit tests: per-dataset `test` modes that load TEST tags only after the TEST pre-registration
   is tracked, unmodified and an ancestor of HEAD (the guard pattern of I8-1 / I9-1 / I10-1); smoke on DEV tags only.
4. TEST pre-registration committed alone (rules of 3.3, predictions of 3.4 by reference, compute estimate).
5. TEST run, one queue job at a time: starting cards, training, refits per split.
6. Decide (the pre-registered rule), then report batch and site update. No method change after step 4; a change
   after TEST is reported as a post hoc analysis.

### 3.6 Compute estimate (from DEV wall times, 6 workers, one job at a time)

| split | basis | estimate |
|---|---|---|
| NIST lvt | start 411 s / 57 devices; LODO 1016 s / 53 folds (training and refits) | about 15-20 min |
| NIST RT card | start 472 s / 87; LODO 2435 s / 87 | about 30-35 min |
| IHP, 4 temperatures | I5-1 start about 18 min (18 DUTs x 4 T), LODO about 10 min per T; I17-2 inner selection 2984 s / 18 folds | about 40-50 min |
| TU Delft, 2 temperatures | I6-1 start 1083 s (45 x 2); I18-2 base 653 s / 45, full-data 5-seed training per lambda about 2-5 min | about 45-60 min |
| total | | about 2.2-2.8 h wall; no Part B rerun |

Part B is not part of the TEST run: under the committed decisions no TEST split applies the g v2 element (NIST
applies M_bias v1, IHP and TU Delft keep C0_refit). The element enters TEST only through the secondary
"correction always" contrast on IHP 300 K and TU Delft.

### 3.7 Reported regardless of outcome

Per split: n devices, failures per arm, mean RRMS (all-device and combined) for C0_refit, M_bias v1, selected g v2,
gated; GMR(gated, C0_refit), GMR(correction always, C0_refit), GMR(selected, v1) with CIs; decision correctness;
I8-2 and I19-1 prediction scores; pooled values; per-device rows; I-V figures with measured points, the starting
card, C0_refit and the correction-arm curves re-simulated in NGSpice; NGSpice calls and wall time per arm.

## 4. Open items before the freeze

1. **NIST clamp state.** The NIST decks do not write the BSIM4 A2 clamp; stored base a2 > 1 on 16/57 (I8-1) and 34/87
   (I9-1) starting cards. The NIST DEV rows and the committed NIST S3_D values (-0.103, -0.069) come from these decks.
   Needed: an I13-1-style audit (states L / C / U) of the NIST rows and a decision on which state the TEST run uses;
   if state C changes S3_D sign, a dated amendment to the Part P decisions.
2. **IHP PSP103 element.** Part B not run: PSP103 contributes Qg, Qb, Qd unconditionally, so a replica carries
   displacement current; ngspice-41 loads OSDI 0.3 only; no IHP circuits. On DEV the gate keeps C0_refit on all IHP
   temperatures, so the method on IHP is the card refit alone.
3. **TU Delft 4.13 K Part B failure.** 3 convergence lines in one dense element deck (t40_nmos_W1_L0p04_d192); the
   card-only deck of the same M2 card prints the same lines. The rule counts element decks. Options: re-register the
   check to count only messages absent from the card-only deck, or record 4.13 K as not deployable.
4. **Circuits for 1 V cards.** Q3 was not run on TU Delft (sky130 circuits are at 1.8 V; card 1 V, measured range
   1.1 V). Needed: 1.1 V inverter, ring oscillator and mirror decks for the TU Delft card.
5. **gds restrictions.** sky130 lambda 1 V: 6 of 90 gds curves with an introduced sign change (long-channel pMOS at
   |V(g,s)| 0.9-1.85 V); lambda 10 V: 2 of 90; TU Delft 295.0 K: 1 curve. These come from g (same points in
   I_card x exp(g)). Options: report as a restriction; a lambda floor; a monotone output constraint (new
   pre-registration).
6. **Runtime overhead.** Element / card runtime up to 4.30-4.60 on ro5_A, 1.8-2.1 on the DC grid; the OSDI network
   evaluation is about 0.6 s of the 0.89 s added on ro5_A (I15-1). No runtime variant registered.
7. **TU Delft TEST bootstrap unit.** 3 geometry groups on TEST; the DEV analyses used a 7-group cluster bootstrap.
   The TEST pre-registration has to fix the unit (proposal in 3.3).
8. **g v2 recipe not run on 5 of 9 DEV datasets.** NIST (both) and IHP 233 / 343 / 398 K have M_bias v1 rows only, so
   the gate applies M_bias v1 there, and M_bias v1 failed the deployment checks on sky130 (I11-1; reference arm of
   I14-1 / I15-1). The frozen method has one correction recipe only if g v2 is run on these datasets (with new
   S3_D values and amended Part P decisions) or if the paper states two recipes.
9. **TU Delft gate miss.** With the g v2 recipe S3_D is +0.008 / +0.264 while the DEV correction lowers the error
   (0.826 / 0.829); the v1 recipe gives -0.029 / -0.051 and the realised sign on 9/9 DEV datasets. Which recipe
   computes S3_D for a g v2 arm is part of the freeze.
10. **IHP 300 K full-data lambda.** Not stored (I17-2 has per-fold lambda* only); needed for the TEST g v2 arm.
11. **Pending user question.** Whether to open the IHP and TU Delft TEST devices (asked 2026-10-21, no reply). If they
    are opened, their I8-2 and I19-1 predictions become void as for sky130, and the locked TEST is NIST only (96
    devices, one PDK). The 2026-10-21 user statement concerned the sky130 77 K devices; whether it covers the NIST
    sky130 4 K TEST splits is part of the same question.
12. **ICML target gaps.**
    - Second PDK: the correction does not transfer to IHP SG13G2 (worse than C0_refit at 300 K: 1.255 [1.026, 1.600]);
      TU Delft uses a generic CMOSedu example card, not a foundry PDK, and its result is inconclusive with the
      cluster CI; GF180 has simulated data only. The gain is shown on sky130 data (77 K and NIST 4 K).
    - Matched-budget baselines: the held-out refit budget is matched across arms (max_nfev 400), but B1 (wider card
      set), B2 (extrinsic resistance and self-heating), B3_d3 (cubic polynomial g) and B4 (GP) were run with the v1
      recipe on sky130 REAL_DEV, NIST, IHP 300 K and TU Delft only, not with the selected g v2 recipe; B3_d3 is close
      to M_bias on NIST (0.981 [0.961, 1.000] and 0.955 [0.926, 0.985]). The memory-listed ML baselines (NPE,
      TuRBO / BO, CMA-ES, neural operators) are for the parameter-extraction stage and have no matched-budget runs
      in the current method.
    - Training and gate cost (g training, S3_D inner fits, nested lambda selection) are not reported in NGSpice
      calls next to the refit budget.
    - Dependence: 8 TEST splits from 4 device sets; the paper's TEST claims rest on NIST (2 splits, sky130 4 K) for
      the applied correction.
