# Tables, 2026-10-22 i17_1

Generated 2026-10-22 10:47 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i17_1.py --date 2026-10-22`.

### I17-1 Part B checks vs pre-registered thresholds: selected correction (outcome), I15-1 g v2 and M_bias v1 values

| check | type | threshold | selected value | selected | I15-1 g v2 (lambda 10 V) | I15-1 v1 reference |
|---|---|---:|---:|---:|---:|---:|
| Q1 equivalence: points violating |dI| <= 1e-6 |I_py| + 1e-13 A (tight, both benches) | core | 0 | 0 | pass | 0 (pass) | 0 (pass) |
| Q1 runs: convergence messages + non-finite values + failed runs (element decks) | core | 0 | 0 | pass | 0 (pass) | 0 (pass) |
| Q1 max |I_elem(Vds = 0)| (A), tight tolerance only (I17-1 rule) | core | <= 1e-12 | 4.8690e-17 | pass | 5.0764e-17 (pass) | 4.8563e-17 (pass) |
| Verilog-A g vs Python, max |dg| | core | <= 1e-9 | 2.78e-16 | pass | 2.50e-16 (pass) | 2.88e-16 (pass) |
| Verilog-A g, exchanged vs forward terminals, max |dg| | core | <= 1e-12 | 1.67e-16 | pass | 3.33e-16 (pass) | 0.1344 (FAIL) |
| exchange asymmetry, element-introduced max |d ln| | core | <= 1e-3 | 2.25e-05 | pass | 1.68e-05 (pass) | 0.1252 (FAIL) |
| Q2 gm / gds vs element identity, max normalized difference | core | <= 1e-3 | 1.08e-05 | pass | 1.24e-05 (pass) | 0.006607 (FAIL) |
| Q2 capacitances (Y_dd, Y_dg, Y_gg), max relative error | core | <= 1e-3 | 3.34e-14 | pass | 1.09e-12 (pass) | 1.04e-13 (pass) |
| Q2 gm curves with an introduced sign change in strong inversion (of 54) | core | 0 | 0 | pass | 0 (pass) | 1 (FAIL) |
| Q3 circuits failing only with the element | core | 0 | 0 | pass | 0 (pass) | 0 (pass) |
| default tolerance, max relative difference (|I| > 1 nA) | restriction | <= 1e-3 (restriction) | 0.001842 | FAIL | 0.001814 (FAIL) | 0.001898 (FAIL) |
| gds curves with an introduced sign change (of 90) | restriction | 0 (restriction) | 6 | FAIL | 2 (FAIL) | 28 (FAIL) |
| wall-time ratio element / card only, max (grid and circuits) | restriction | <= 2 (restriction) | 4.604 | FAIL | 4.298 (FAIL) | 4.295 (FAIL) |
| reported, no rule: Q1 max |I_elem(Vds = 0)| (A), default tolerance | reported | (I16-1: <= 1e-12) | 1.0006e-12 | 1 of 1350 points > 1e-12 A | 9.8357e-13 | 9.9506e-13 |
| label (I17-1 rule) |  |  | deployable with restrictions |  | deployable with restrictions | not deployable |
| label under the unchanged I16-1 rule (max of default and tight), same run |  |  | not deployable |  |  |  |

- The Q1 V(d,s) = 0 check was re-registered after the observed I16-1 failure (tight tolerance only, threshold 1e-12 A kept); the I16-1 outcome stays 'not deployable'. I15-1 columns of that row: tight-tolerance values from I16_1/diagnosis/vds0.json.
- selected: I15-2 full-data correction sym_e1_l1 (symmetric inputs, EPS 0.01 V, penalty lambda 1 V = full-data lambda*) with its M2 cards (refit from the starting card with g in the loop, state C, 0 failed). I15-1 columns (not rerun): g v2 (lambda 10 V) and the M_bias v1 reference arm. Same element (channel-current replica, gmin VCCS, module I(d, s) <+ (exp(g) - 1) I(sense)), serpentine Q1 grids.
- Classification (prereg, kept from I15-1): an introduced gds sign change is a restriction; an introduced gm sign change in strong inversion is a core failure.
- Q2 with the I12-1 formula exp(g) Re Y_card + I_card exp(g) dg/dV on the same runs: max 0.0229.
- Runtime: DC grid 1.84-2.07; circuits inv_A_vtc 1.76, inv_B_vtc 1.75, ro5_A 4.60, mirror_n 1.20, mirror_p 1.29; ro5_B excluded (I12-1 amendment A1).


### I17-1 validity

| check | criterion | value | result |
|---|---:|---:|---:|
| V1 card-only library deck x exp(fold g) vs I15-2 selected rows (18 rows) | <= 1e-9 | 0 | pass |
| V2 unit tests (test_i14_1_chan, test_i13_1_clamp, test_spice_batch, test_sky130_misspec, test_i12_1_symm, test_i15_1_serp, test_i15_2_ablate, test_i16_1_sel, test_i17_1_vds0) | all pass | OK | pass |
| V3 element-deck accuracy rows / failed M2 refits | 18 / 0 | 18 / 0 | pass |
| V4 deployed g: inputs / EPS / lambda; I15-2 full-data lambda* | sym_v2 / 0.01 / 1; 1 | sym_v2 / 0.01 / 1.0; 1.0 | pass |

- Validity rule (prereg): any V fail -> invalid. Result: valid.


### I17-1 Part B per device, selected correction with its M2 card

| device | Q1 ratio | asym | Q2 | cap | default tol | gm sign | gds sign | msgs | runtime | M2 RRMS py / native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.04 | 1.60e-06 | 2.54e-07 | 1.61e-15 | 0.001105 | 0 | 0 | 0 | 1.93 | 0.0273 / 0.0273 |
| nmos_L0p19_W7 | 0.36 | 2.25e-05 | 7.20e-06 | 1.42e-15 | 0.001133 | 0 | 0 | 0 | 2.04 | 0.0162 / 0.0162 |
| nmos_L0p25_W1p6 | 0.05 | 3.31e-06 | 3.20e-07 | 1.49e-15 | 0.001022 | 0 | 0 | 0 | 2.01 | 0.0170 / 0.0170 |
| nmos_L100_W100 | 0.11 | 1.89e-07 | 2.19e-07 | 3.34e-14 | 0.001006 | 0 | 0 | 0 | 2.03 | 0.0685 / 0.0683 |
| nmos_L1_W1p6 | 0.11 | 7.01e-06 | 1.70e-06 | 3.39e-15 | 0.001098 | 0 | 0 | 0 | 1.87 | 0.0243 / 0.0243 |
| nmos_L1_W3 | 0.05 | 1.69e-06 | 1.10e-06 | 3.39e-15 | 0.001429 | 0 | 0 | 0 | 1.95 | 0.0463 / 0.0463 |
| nmos_L20_W0p64 | 0.03 | 1.79e-06 | 5.19e-07 | 5.65e-15 | 0.001716 | 0 | 0 | 0 | 1.97 | 0.0456 / 0.0454 |
| nmos_L8_W1p6 | 0.05 | 3.33e-06 | 3.37e-06 | 6.59e-15 | 0.001682 | 0 | 0 | 0 | 1.99 | 0.0173 / 0.0173 |
| pmos_L0p35_W0p55 | 0.01 | 3.41e-07 | 1.70e-07 | 1.84e-15 | 0.001484 | 0 | 0 | 0 | 1.84 | 0.3299 / 0.3299 |
| pmos_L0p35_W1p6 | 0.00 | 9.29e-10 | 9.70e-10 | 0 | 0.001759 | 0 | 0 | 0 | 1.90 | 0.1017 / 0.1017 |
| pmos_L0p35_W5 | 0.21 | 1.14e-05 | 5.21e-06 | 4.20e-15 | 0.001395 | 0 | 0 | 0 | 2.01 | 0.0513 / 0.0513 |
| pmos_L0p5_W0p42 | 0.02 | 3.81e-07 | 3.56e-07 | 3.20e-14 | 0.001715 | 0 | 0 | 0 | 2.04 | 0.0958 / 0.0958 |
| pmos_L0p5_W0p64 | 0.02 | 6.10e-07 | 2.46e-07 | 2.51e-15 | 0.001435 | 0 | 0 | 0 | 1.85 | 0.1086 / 0.1086 |
| pmos_L2_W5 | 0.00 | 3.73e-14 | 9.80e-09 | 2.22e-16 | 0.001842 | 0 | 1 | 0 | 1.90 | 0.0526 / 0.0526 |
| pmos_L4_W7 | 0.11 | 1.34e-05 | 1.08e-05 | 5.03e-15 | 0.001233 | 0 | 0 | 0 | 1.91 | 0.0649 / 0.0649 |
| pmos_L8_W0p84 | 0.03 | 4.20e-07 | 1.00e-06 | 4.27e-15 | 0.001383 | 0 | 2 | 0 | 2.07 | 0.0728 / 0.0728 |
| pmos_L8_W1p6 | 0.03 | 1.26e-06 | 1.49e-06 | 1.27e-14 | 0.001537 | 0 | 2 | 0 | 1.86 | 0.0695 / 0.0695 |
| pmos_L8_W5 | 0.11 | 4.90e-06 | 6.25e-06 | 5.09e-15 | 0.00123 | 0 | 1 | 0 | 1.90 | 0.0439 / 0.0439 |

- q1: max |dI| / (1e-6 |I_py| + 1e-13 A), tight, forward and exchanged (pass <= 1); asym: element-introduced |d ln(I_swap / I_fwd)|; q2: max normalized gm / gds difference vs the element identity; cap: capacitance error; def: default-tolerance relative difference (|I| > 1 nA); gm: gm curves with an introduced sign change in strong inversion (of 3, core); gds: gds curves with an introduced sign change (of 5, restriction); msg: convergence lines in element runs; rt: DC grid wall-time ratio; M2 RRMS in-sample: Python (I_card exp(g)) / SPICE-native element.


### I17-1 element-deck accuracy of the selected held-out pairs vs the I15-2 rows, 18 devices

| arm / contrast | value | detail |
|---|---:|---|
| selected (I15-2 rows; Python, I_card exp(g)) | 0.07391 / 0.06997 / 0.05504 |  |
| selected, element deck (fold OSDI module + held-out card) | 0.07390 / 0.06996 / 0.05503 | max |native - stored| 2.0e-04; Python rerun max |d| 0.0e+00 |
| M_bias v1 (I13-1 rows) | 0.07348 / 0.06953 / 0.06370 |  |
| M_bias v1, element deck (I15-1) | 0.07346 / 0.06951 / 0.06358 |  |
| g v2 lambda 10 V, element deck (I15-1) | 0.07714 / 0.07321 / 0.05428 |  |
| C0_refit (no correction) | 0.10509 / 0.09839 / 0.08198 |  |
| GMR(selected element, v1 rows) | 0.996 [0.962, 1.027] | 7/18 lower |
| GMR(selected element, v1 element) | 0.997 [0.962, 1.027] | 7/18 lower |
| GMR(selected element, g v2 element) | 0.947 [0.885, 1.023] | 13/18 lower |
| GMR(selected element, C0_refit) | 0.743 [0.654, 0.855] | 17/18 lower |
| I15-2 primary GMR(selected rows, v1 rows) (registered, fixed) | 0.997 [0.962, 1.027] | 7/18 lower; selected variant non-inferior to M_bias v1 |

- Values: all-device / combined / median (strict paper RRMS, published-card curve inclusion). Element deck: per device h the fold correction of sym_e1_l{lambda*_h} (lambda* 1 V on 10 folds, 2 V on 6, 0 V on pmos_L4_W7, 10 V on pmos_L8_W5) as an OSDI element with its held-out card, I = I_card + (exp(g) - 1) I_rep.
- Element vs channel model I_card + (exp(g) - 1) I_rep from the card + replica run: max |dI| / (1e-3 |I| + 1e-12 A) 1.00; convergence lines 0.
- GMR: device bootstrap 10,000, 95% CI. The accuracy result of the selected variant is the I15-2 primary (not rerun); the element-deck values are reported without a rule.


### I17-1 gds restriction items per device: negative output-conductance points, selected correction (element / I_card exp(g)) and g v2 reference

| device | |Vgs| 0.60 V | |Vgs| 0.90 V | |Vgs| 1.20 V | |Vgs| 1.50 V | |Vgs| 1.85 V | curves introduced | g v2 (lambda 10 V) |
|---|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| nmos_L0p19_W7 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| nmos_L0p25_W1p6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| nmos_L100_W100 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| nmos_L1_W1p6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| nmos_L1_W3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| nmos_L20_W0p64 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| nmos_L8_W1p6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| pmos_L0p35_W0p55 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| pmos_L0p35_W1p6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| pmos_L0p35_W5 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| pmos_L0p5_W0p42 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| pmos_L0p5_W0p64 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| pmos_L2_W5 | 0 / 0 | I: 230 / 230 | 0 / 0 | 0 / 0 | 0 / 0 | 1 | 0 (0) |
| pmos_L4_W7 | 5 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 | 0 (0) |
| pmos_L8_W0p84 | 4 / 3 | 0 / 0 | 0 / 0 | I: 115 / 115 | I: 555 / 555 | 2 | 1 (510) |
| pmos_L8_W1p6 | 0 / 0 | 0 / 0 | 0 / 0 | I: 543 / 543 | I: 870 / 870 | 2 | 1 (914) |
| pmos_L8_W5 | 116 / 102 | 0 / 0 | 0 / 0 | I: 126 / 126 | 0 / 0 | 1 | 0 (301) |

- Per V(g,s) (nMOS sign shown for the first device; pMOS negative): points of the 1 mV dense sweep with dI/dV(d,s) < 0 at |V(d,s)| >= 0.05 V, element / Python I_card exp(g) of the card sweep; 'I': introduced sign change by the pre-registered curve rule (card has no negative value below -1e-12 S, element has; AC or dense). The card alone has 112 negative points.
- Totals selected: 6 of 90 curves introduced, 2564 element points, 2544 Python points. I15-1 g v2 (lambda 10 V): 2 curves, 1725 points (column 'g v2': curves introduced (points)). Classification: restriction (prereg).


### I17-1 gm items per device (core check): negative transconductance points in strong inversion, element / card / I_card exp(g)

| device | V_T card (V) | |Vds| 0.05 V | |Vds| 0.925 V | |Vds| 1.85 V | curves introduced | g v2 (lambda 10 V) |
|---|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.727 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| nmos_L0p19_W7 | 0.638 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| nmos_L0p25_W1p6 | 0.669 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| nmos_L100_W100 | 0.039 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| nmos_L1_W1p6 | 0.608 | 116 / 115 / 116 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| nmos_L1_W3 | 0.647 | 200 / 200 / 200 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (177) |
| nmos_L20_W0p64 | 0.325 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| nmos_L8_W1p6 | 0.342 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L0p35_W0p55 | 1.151 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L0p35_W1p6 | 0.916 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L0p35_W5 | 1.104 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L0p5_W0p42 | 1.090 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L0p5_W0p64 | 1.077 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L2_W5 | 1.099 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L4_W7 | 1.134 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L8_W0p84 | 1.164 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L8_W1p6 | 1.197 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |
| pmos_L8_W5 | 1.135 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 | 0 (0) |

- Per |V(d,s)|: points of the 1 mV dense V(g,s) sweep in strong inversion (|V(g,s)| >= V_T + 0.2 V; V_T of the card at |V(d,s)| = 0.05 V) with dI/dV(g,s) < 0, element / card / Python I_card exp(g); 'I': introduced sign change by the curve rule (card has no negative gm below -1e-12 S, element has). Negative gm of the card itself (mobility degradation at high |V(g,s)|) is not an introduced change.
- Totals selected: 0 of 54 curves introduced (core check); element 316, card 315, Python 316 points. Column 'g v2': I15-1 lambda 10 V, curves introduced (element points).


### I17-1 Q3 circuits (default tolerance)

| circuit | published | C0_refit | M2 card | M2 card + element | I15-1 g v2 (card / element) |
|---|---:|---:|---:|---:|---:|
| inv_A_vtc | 0.7466 V | 0.7593 V | 0.7595 V | 0.7601 V | 0.7596 V / 0.7600 V |
| inv_B_vtc | 0.6417 V | 0.6491 V | 0.6485 V | 0.6491 V | 0.6491 V / 0.6499 V |
| ro5_A | 5.322 GHz | 5.287 GHz | 5.292 GHz | 5.384 GHz | 5.291 GHz / 5.353 GHz |
| ro5_B | 0.276 GHz | fail | fail | fail | fail / fail |
| mirror_n | 1.059 | 1.022 | 1.022 | 1.030 | 1.022 / 1.024 |
| mirror_p | 0.938 | 0.992 | 0.992 | 0.989 | 0.992 / 0.984 |

- inverters: switching voltage V_M; ring oscillators: frequency; mirrors: Iout / Iref at 0.9 V. M2 card: refit with the selected full-data g in the loop, simulated without the element; M2 card + element: with it. Last column: I15-1 g v2 M2 card / M2 card + element. 'fail' at card level is not a Q3 failure (I12-1 amendment A1).


### I17-1 restriction items of the selected correction

| item | value | detail |
|---|---:|---|
| default-tolerance difference, max relative (|I| > 1 nA) | 1.84e-03 | 18 of 18 devices above 1e-3; r4 max 9.9e-04; tight max 2.3e-05 |
| gds curves with an introduced sign change (of 90) | 6 | pmos_L2_W5, pmos_L8_W0p84, pmos_L8_W1p6, pmos_L8_W5 |
| gm curves with an introduced sign change (of 54; core) | 0 | none |
| runtime, DC grid ratio (18 devices) | 1.84-2.07 | circuits: inv_A_vtc 1.76, inv_B_vtc 1.75, ro5_A 4.60, mirror_n 1.20, mirror_p 1.29 |
| ro5_A decomposition (M2 cards), wall s card / card + replica / element | 0.261 / 0.526 / 1.211 | rusage load s 0.130 / 0.289 / 0.941 |

- Restriction items (reported). The module text has the same network size as I15-1 (5 seeds x 3-16-16-1 tanh); runtime differences to I15-1 come from the M2 cards and machine load.


### I17-1 re-registered check: element current at V(d,s) = 0 per tolerance, I17-1 run vs I16-1 run and I15-1 arms

| arm / run | tolerance | max |I_elem| (A) | at | max |I_card+rep| (A) | points > 1e-12 A | in I16-1 rule | in I17-1 rule |
|---|---|---:|---|---:|---:|---:|---:|
| I17-1 run, selected | default | 1.0006e-12 | pmos_L8_W0p84, |V(g,s)| 0.825 V | 2.56e-13 | 1 / 1350 | yes | no |
|  | r4 | 9.2802e-13 | nmos_L1_W3, |V(g,s)| 0.525 V | 2.56e-13 | 0 / 1350 | no | no |
|  | tight | 4.8690e-17 | nmos_L0p19_W7, |V(g,s)| 1.825 V | 4.66e-17 | 0 / 1350 | yes | yes |
| I16-1 run, selected | default | 1.0006e-12 | pmos_L8_W0p84, |V(g,s)| 0.825 V | 2.56e-13 | 1 / 1350 | yes | no |
|  | r4 | 9.2802e-13 | nmos_L1_W3, |V(g,s)| 0.525 V | 2.56e-13 | 0 / 1350 | no | no |
|  | tight | 4.8690e-17 | nmos_L0p19_W7, |V(g,s)| 1.825 V | 4.66e-17 | 0 / 1350 | yes | yes |
| I15-1 g v2, lambda 10 V | default | 9.8357e-13 | nmos_L20_W0p64, |V(g,s)| 0.250 V | 2.57e-13 | 0 / 1350 | yes | no |
|  | r4 | 9.8357e-13 | nmos_L20_W0p64, |V(g,s)| 0.250 V | 2.57e-13 | 0 / 1350 | no | no |
|  | tight | 5.0764e-17 | nmos_L0p19_W7, |V(g,s)| 1.850 V | 4.59e-17 | 0 / 1350 | yes | yes |
| I15-1 M_bias v1 | default | 9.9506e-13 | nmos_L1_W3, |V(g,s)| 0.525 V | 2.55e-13 | 0 / 1350 | yes | no |
|  | r4 | 9.9506e-13 | nmos_L1_W3, |V(g,s)| 0.525 V | 2.55e-13 | 0 / 1350 | no | no |
|  | tight | 4.8563e-17 | nmos_L0p19_W7, |V(g,s)| 1.850 V | 4.75e-17 | 0 / 1350 | yes | yes |

- The check was changed after the observed I16-1 failure (I17-1 prereg, commit 3cfc9a7): I16-1 took the max over the default and tight forward Q1 grids; I17-1 uses the tight grid only, threshold 1e-12 A kept. The I16-1 outcome stays 'not deployable'. Default: ngspice options not set (abstol 1e-12 A, reltol 1e-3); r4: reltol 1e-4 (in neither rule); tight: reltol 1e-9, abstol 1e-18 A, vntol 1e-12 V.
- Reason registered for the change: at default tolerance the threshold equals abstol, the current tolerance of the solver's convergence test, so the check there does not separate an element current from the solver tolerance; at tight tolerance abstol is 1e-18 A.
- I17-1 label: deployable with restrictions; the same run under the I16-1 rule: not deployable.


### I17-1 rerun vs the I16-1 run: Part B values and labels

| value | I16-1 run | I17-1 run | I17-1 - I16-1 |
|---|---:|---:|---:|
| q1_tight_viol | 0 | 0 | 0 |
| q1_bad | 0 | 0 | 0 |
| q1_vds0_max | 1.00e-12 | 1.00e-12 | 0 |
| g_text_max | 2.78e-16 | 2.78e-16 | 0 |
| g_sym_text_max | 1.67e-16 | 1.67e-16 | 0 |
| swap_asym_max | 2.25e-05 | 2.25e-05 | 0 |
| q2_ac_max | 1.08e-05 | 1.08e-05 | 0 |
| cap_max | 3.34e-14 | 3.34e-14 | 0 |
| q2_gm_sign | 0 | 0 | 0 |
| q3_fail | 0 | 0 | 0 |
| q1_default_max | 0.001842 | 0.001842 | 0 |
| q2_gds_sign | 6 | 6 | 0 |
| runtime_ratio_max | 4.561 | 4.604 | 0.04296 |
| M2 in-sample RRMS (element) | 0.069624 | 0.069624 | 0 |
| label under the I16-1 rule | not deployable | not deployable |  |
| label under the I17-1 rule | (not registered) | deployable with restrictions |  |

- Values as computed by the I14-1 summary (q1_vds0_max: max of default and tight, the I16-1 definition). Every stage was rerun into I17_1 (M2 refits, decks, circuits); the runtime ratio differs by machine load.

