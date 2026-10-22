# Tables, 2026-10-22 i14_1

Generated 2026-10-22 01:19 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i14_1.py --date 2026-10-22`.

### I14-1 Part B checks vs pre-registered thresholds: g v2 (outcome) and M_bias v1 (reference arm)

| check | type | threshold | g v2 value | g v2 | v1 reference value | v1 |
|---|---|---:|---:|---:|---:|---:|
| Q1 equivalence: points violating |dI| <= 1e-6 |I_py| + 1e-13 A (tight, both benches) | core | 0 | 0 | pass | 0 | pass |
| Q1 runs: convergence messages + non-finite values (element runs) | core | 0 | 2 | FAIL | 0 | pass |
| Q1 max |I_elem(Vds = 0)| (A) | core | <= 1e-12 | 9.84e-13 | pass | 1.00e-12 | FAIL |
| Verilog-A g vs Python, max |dg| | core | <= 1e-9 | 2.50e-16 | pass | 2.88e-16 | pass |
| Verilog-A g, exchanged vs forward terminals, max |dg| | core | <= 1e-12 | 3.33e-16 | pass | 0.1344 | FAIL |
| exchange asymmetry, element-introduced max |d ln| | core | <= 1e-3 | 1.68e-05 | pass | 0.1252 | FAIL |
| Q2 gm / gds vs element identity, max normalized difference | core | <= 1e-3 | 1.24e-05 | pass | 0.006607 | FAIL |
| Q2 capacitances (Y_dd, Y_dg, Y_gg), max relative error | core | <= 1e-3 | 1.09e-12 | pass | 1.04e-13 | pass |
| Q2 gm curves with an introduced sign change (of 54) | core | 0 | 0 | pass | 1 | FAIL |
| Q3 circuits failing only with the element | core | 0 | 0 | pass | 0 | pass |
| default tolerance, max relative difference (|I| > 1 nA) | restriction | <= 1e-3 (restriction) | 0.001805 | FAIL | 0.001926 | FAIL |
| gds curves with an introduced sign change (of 90) | restriction | 0 (restriction) | 2 | FAIL | 28 | FAIL |
| wall-time ratio element / card only, max (grid and circuits) | restriction | <= 2 (restriction) | 4.587 | FAIL | 4.479 | FAIL |
| label |  |  | not deployable |  | not deployable |  |

- v2: I13-1 full-data g v2 (state C) with its M2 cards (refit from the starting card with g in the loop, state C, 0 failed); v1: reference arm, I13-1 full-data M_bias v1 g (abs() inputs) with its own M2 cards; same element (channel-current replica, gmin VCCS, module I(d, s) <+ (exp(g) - 1) I(sense)).
- Q1 runs (v2): the 2 counted lines are 'dynamic gmin stepping failed' and 'true gmin stepping failed' in one element run (nmos_L8_W1p6, exchanged bench, tight); the card + replica reference run of the forward bench printed the same two lines. Q1 equivalence on that run: 0 violations.
- Q2 with the I12-1 formula exp(g) Re Y_card + I_card exp(g) dg/dV on the same runs: max 0.0217 (v2).
- Runtime: DC grid 1.90-2.00; circuits inv_A_vtc 1.76, inv_B_vtc 1.78, ro5_A 4.59, mirror_n 1.23, mirror_p 1.28; ro5_B excluded (fails at card level, I12-1 amendment A1).


### I14-1 validity and replica pre-check

| check | criterion | value | result |
|---|---:|---:|---:|
| V1 card-only library deck x exp(fold g) vs I13-1 rows (36 rows) | <= 1e-9 | 0 | pass |
| V2 unit tests (test_i14_1_chan, test_i13_1_clamp, test_spice_batch, test_sky130_misspec, test_i12_1_symm) | all pass | OK | pass |
| V3 element-deck accuracy rows per arm / failed M2 refits | 18 / 0 | 18, 18 / 0, 0 | pass |
| replica pre-check, published: P1 non-channel max / P2 max abs (A) / P3 forward max rel / P3 devices <= 1e-12 | 1e-12 relative requested | 0 / 2.9e-13 / 1.1e-05 / 2 of 18 | met in part |
| replica pre-check, start (C): P1 non-channel max / P2 max abs (A) / P3 forward max rel / P3 devices <= 1e-12 | 1e-12 relative requested | 0 / 3.6e-13 / 0.0012 / 6 of 18 | met in part |
| replica pre-check, v2 held-out (C): P1 non-channel max / P2 max abs (A) / P3 forward max rel / P3 devices <= 1e-12 | 1e-12 relative requested | 0 / 1.7e-12 / 0.0011 / 5 of 18 | met in part |
| replica pre-check, v1 held-out (C): P1 non-channel max / P2 max abs (A) / P3 forward max rel / P3 devices <= 1e-12 | 1e-12 relative requested | 0 / 4.8e-13 / 0.0012 / 6 of 18 | met in part |

- Validity rule (prereg): any V fail -> invalid. Outcome: valid.
- Replica pre-check (before registration): P1 replica isub, igidl, igisl, gate currents; P2 sensed current vs the replica's own BSIM4 channel current; P3 replica channel current vs the card's channel current (forward quadrant, |I| > 1 nA). The card's own Isub and junction currents flow through its body resistor network (rbodymod = 1) and change its channel current; the replica without them does not see that.


### I14-1 Part B per device, g v2 with its M2 card

| device | Q1 ratio | asym | asym card | Q2 | cap | default tol | gds sign | msgs | runtime | M2 RRMS py / native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.05 | 1.74e-06 | 5.05e-07 | 3.91e-07 | 1.08e-15 | 0.001161 | 0 | 0 | 1.92 | 0.0285 / 0.0285 |
| nmos_L0p19_W7 | 0.22 | 1.28e-05 | 5.31e-07 | 9.22e-07 | 1.62e-15 | 0.001184 | 0 | 0 | 1.96 | 0.0199 / 0.0200 |
| nmos_L0p25_W1p6 | 0.06 | 2.93e-06 | 9.96e-08 | 1.65e-06 | 1.37e-15 | 9.94e-04 | 0 | 0 | 1.95 | 0.0188 / 0.0188 |
| nmos_L100_W100 | 0.35 | 6.66e-07 | 3.09e-07 | 2.61e-07 | 3.06e-14 | 9.96e-04 | 0 | 0 | 1.94 | 0.0696 / 0.0695 |
| nmos_L1_W1p6 | 0.10 | 1.63e-06 | 2.17e-07 | 2.56e-06 | 2.22e-15 | 0.001056 | 0 | 0 | 1.98 | 0.0285 / 0.0285 |
| nmos_L1_W3 | 0.05 | 7.82e-07 | 6.67e-08 | 3.99e-07 | 2.99e-15 | 0.001035 | 0 | 0 | 1.96 | 0.0491 / 0.0491 |
| nmos_L20_W0p64 | 0.03 | 1.80e-06 | 8.86e-07 | 3.17e-07 | 7.85e-15 | 0.001192 | 0 | 0 | 1.98 | 0.0502 / 0.0500 |
| nmos_L8_W1p6 | 0.10 | 6.72e-06 | 6.73e-08 | 5.07e-06 | 5.02e-15 | 0.00105 | 0 | 2 | 1.94 | 0.0222 / 0.0222 |
| pmos_L0p35_W0p55 | 0.01 | 3.59e-07 | 1.70e-07 | 1.51e-07 | 1.28e-15 | 0.001653 | 0 | 0 | 1.92 | 0.3433 / 0.3433 |
| pmos_L0p35_W1p6 | 0.00 | 1.20e-09 | 2.02e-08 | 5.39e-10 | 0 | 0.001349 | 0 | 0 | 2.00 | 0.1199 / 0.1199 |
| pmos_L0p35_W5 | 0.22 | 1.68e-05 | 2.44e-07 | 2.01e-06 | 2.91e-15 | 0.001805 | 0 | 0 | 1.97 | 0.0620 / 0.0620 |
| pmos_L0p5_W0p42 | 0.02 | 6.89e-07 | 1.08e-07 | 3.33e-07 | 1.09e-12 | 0.001583 | 0 | 0 | 1.99 | 0.1065 / 0.1065 |
| pmos_L0p5_W0p64 | 0.02 | 6.75e-07 | 9.15e-08 | 3.17e-07 | 2.18e-15 | 0.001723 | 0 | 0 | 1.94 | 0.1221 / 0.1221 |
| pmos_L2_W5 | 0.00 | 7.13e-14 | 6.56e-14 | 1.89e-09 | 2.21e-16 | 0.001041 | 0 | 0 | 1.97 | 0.0483 / 0.0483 |
| pmos_L4_W7 | 0.21 | 6.56e-06 | 9.76e-08 | 8.50e-06 | 2.38e-15 | 0.001326 | 0 | 0 | 1.93 | 0.0568 / 0.0568 |
| pmos_L8_W0p84 | 0.03 | 7.79e-07 | 5.52e-08 | 8.62e-07 | 3.05e-15 | 0.001065 | 1 | 0 | 1.98 | 0.0870 / 0.0870 |
| pmos_L8_W1p6 | 0.03 | 1.46e-06 | 6.18e-08 | 8.50e-07 | 1.54e-14 | 9.91e-04 | 1 | 0 | 1.90 | 0.0817 / 0.0817 |
| pmos_L8_W5 | 0.21 | 1.06e-05 | 8.21e-08 | 1.24e-05 | 2.12e-15 | 0.001086 | 0 | 0 | 1.91 | 0.0461 / 0.0461 |

- q1: max |dI| / (1e-6 |I_py| + 1e-13 A), tight, forward and exchanged (pass <= 1); asym: element-introduced |d ln(I_swap / I_fwd)|, card: the card's own; q2: max normalized gm / gds difference vs the element identity; cap: capacitance error; def: default-tolerance relative difference (|I| > 1 nA); gds: gds curves with an introduced sign change (of 5); msg: convergence lines in element runs; rt: DC grid wall-time ratio; M2 RRMS in-sample: Python (I_card exp(g)) / SPICE-native element.


### I14-1 element-deck accuracy of the held-out pairs vs the I13-1 rows

| arm / contrast | stored (Python) | element deck (native) | all-device diff | max |d| | native vs channel model |
|---|---:|---:|---:|---:|---:|
| M_bias v2 (g v2) | 0.07716 / 0.07323 / 0.05437 | 0.07714 / 0.07321 / 0.05428 | -1.8e-05 | 1.8e-04 | 1.00 |
| M_bias v1 (reference) | 0.07348 / 0.06953 / 0.06370 | 0.07346 / 0.06951 / 0.06358 | -1.4e-05 | 2.5e-04 | 1.00 |
| GMR(v2, C0_refit): stored / native | 0.785 [0.693, 0.896] | 0.785 [0.693, 0.896] |  |  |  |
| GMR(v2, v1): stored / native v2 vs stored v1 | 1.053 [0.966, 1.137] | 1.053 [0.966, 1.137] |  |  |  |
| GMR(v2, v1): native vs native |  | 1.053 [0.966, 1.137] |  |  |  |
| GMR(v1, C0_refit): stored / native | 0.745 [0.660, 0.850] | 0.745 [0.659, 0.850] |  |  |  |

- stored: I13-1 state-C rows (Python, I_card exp(g)); the card-only library deck reproduces them (V1, 0.0). native: element deck (fold g as OSDI module with the channel-current replica, I13-1 held-out card), I = I_card + (exp(g) - 1) I_rep. all-device / combined / median.
- max |d|: per-device |native - stored|. ch: native vs channel model I_card + (exp(g) - 1) I_rep from the card + replica run, max |dI| / (1e-3 |I| + 1e-12 A) at default tolerance.
- Registered accuracy flags stay the I13-1 values: A1 passes (CI upper < 1), A2 fails (CI upper > 1.05). GMR: device bootstrap 10,000, 95% CI.


### I14-1 diagnosis: convergence warnings by deck variant (post hoc)

| device | variant | bench | messages |
|---|---|---:|---|
| nmos_L0p25_W1p6 | elem_novccs | fwd | warning: dynamic gmin stepping failed; warning: true gmin stepping failed; supplies reduced to 100.0000% warning: source stepping failed |
| nmos_L8_W1p6 | card_rep | fwd | warning: dynamic gmin stepping failed; warning: true gmin stepping failed |
| nmos_L8_W1p6 | card_rep_novccs | fwd | warning: dynamic gmin stepping failed; warning: true gmin stepping failed |
| nmos_L8_W1p6 | elem | swap | warning: dynamic gmin stepping failed; warning: true gmin stepping failed |
| nmos_L8_W1p6 | elem_novccs | swap | warning: dynamic gmin stepping failed; warning: true gmin stepping failed |

- Post hoc (scripts/research/explore/i14_1_diagnosis.py; the recorded outcome is unchanged): tight forward and exchanged grids of every M2 card (g v2) in six variants: card only; card + channel replica + gmin VCCS (I14-1 reference); the same without the VCCS; card + I12-1 replica (junction and substrate currents kept); I14-1 element; element without the VCCS.
- No message in the card-only and I12-1-replica variants of any device. The warnings of nmos_L8_W1p6 occur with the channel-current replica with and without the VCCS and with and without the module.
- Q1 of the I14-1 element against its card + replica reference: 0 violations on all 18 devices; the variant without the VCCS on nmos_L0p25_W1p6 (source stepping failed) has 1735 violating points.


### I14-1 Q3 circuits (default tolerance)

| arm | circuit | published | C0_refit | M2 card | M2 card + element |
|---|---|---:|---:|---:|---:|
| v2 | inv_A_vtc | 0.7466 V | 0.7593 V | 0.7596 V | 0.7600 V |
|  | inv_B_vtc | 0.6417 V | 0.6491 V | 0.6491 V | 0.6499 V |
|  | ro5_A | 5.322 GHz | 5.287 GHz | 5.291 GHz | 5.353 GHz |
|  | ro5_B | 0.276 GHz | fail | fail | fail |
|  | mirror_n | 1.059 | 1.022 | 1.022 | 1.024 |
|  | mirror_p | 0.938 | 0.992 | 0.992 | 0.984 |
| v1 | inv_A_vtc | 0.7466 V | 0.7593 V | 0.7595 V | 0.7603 V |
|  | inv_B_vtc | 0.6417 V | 0.6491 V | 0.6484 V | 0.6492 V |
|  | ro5_A | 5.322 GHz | 5.287 GHz | 5.296 GHz | 5.371 GHz |
|  | ro5_B | 0.276 GHz | fail | fail | fail |
|  | mirror_n | 1.059 | 1.022 | 1.022 | 1.033 |
|  | mirror_p | 0.938 | 0.992 | 0.992 | 0.974 |

- inverters: switching voltage V_M; ring oscillators: frequency; mirrors: Iout / Iref at 0.9 V. ro5_B fails at card level for C0_refit and M2_card (timestep too small; I12-1 amendment A1).

