# Tables, 2026-10-28 i25_1_element

Generated 2026-10-28 03:00 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i25_1_element.py --date 2026-10-28`.

### Element checks (I17-1 rules): I25-1 arms A and B, old 43p card + g

| check | type | limit | arm A, 7p + g | arm B, 45p + g | old 43p + g (I17-1) |
|---|---|---:|---:|---:|---:|
| Q1 element vs I_card + (e^g - 1) I_rep, tight, both benches: points with |dI| > 1e-6 |I_py| + 1e-13 A | core | 0 | 43 (FAIL) | 2.26e+04 (FAIL) | 0 (pass) |
| Q1 element runs: convergence messages + non-finite values + failed runs | core | 0 | 0 (pass) | 2.29e+04 (FAIL) | 0 (pass) |
| Q1 max |I_elem(V(d,s) = 0)| (A), tight tolerance | core | <= 1e-12 | 4.68e-17 (pass) | inf (FAIL) | 4.87e-17 (pass) |
| Verilog-A g vs Python g, max |dg| (2000 points) | core | <= 1e-9 | 3.33e-16 (pass) | 3.05e-16 (pass) | 2.78e-16 (pass) |
| Verilog-A g, exchanged vs forward terminals, max |dg| | core | <= 1e-12 | 4.44e-16 (pass) | 2.22e-16 (pass) | 1.67e-16 (pass) |
| d-s exchange, element-introduced max |d ln I| (|I| > 1 nA) | core | <= 1e-3 | 3.13e-05 (pass) | 8.51e-05 (pass) | 2.25e-05 (pass) |
| Q2 AC gm, gds vs element identity, max normalized difference | core | <= 1e-3 | 3.23e-05 (pass) | 9.66e-05 (pass) | 1.08e-05 (pass) |
| Q2 capacitances (Y_dd, Y_dg, Y_gg) vs card, max relative error | core | <= 1e-3 | 1.14e-14 (pass) | 3.26e-14 (pass) | 3.34e-14 (pass) |
| gm curves with a sign change introduced in strong inversion (of 54) | core | 0 | 0 (pass) | 2 (FAIL) | 0 (pass) |
| Q3 circuits failing only with the element (of 6) | core | 0 | 0 (pass) | 0 (pass) | 0 (pass) |
| default tolerance, max relative difference (|I| > 1 nA) | restriction | <= 1e-3 | 0.001982 (restriction) | 0.001588 (restriction) | 0.001842 (restriction) |
| gds curves with a sign change introduced (of 90) | restriction | 0 | 6 (restriction) | 10 (restriction) | 6 (restriction) |
| wall-time ratio element / card, max (DC grid and circuits) | restriction | <= 2 | 4.494 (restriction) | 4.305 (restriction) | 4.604 (restriction) |
| label (I17-1 rule) |  |  | not deployable | not deployable | deployable with restrictions |
| reported, no rule: label under the I16-1 V(d,s) = 0 rule (max of default and tight) |  |  | not deployable | not deployable | not deployable |

- arm A: I25-1 7-parameter full-data card + g; arm B: 45-parameter full-data card + g; g = full-data g v2 (sym_v2 inputs, EPS 0.01 V, lambda*_full 5 V); cards = I25-1 in-sample refits with g in the loop (multiplier-1 template, state C). Old 43p + g: I17-1 run (lambda 1 V), out/research/explore/I17_1.
- Element: I(d, s) = I_card + (exp(g) - 1) I_rep, I_rep = sensed channel current of a charge-free card replica; Verilog-A sense branch V(sense) <+ RS * I(sense), RS = 0; OpenVAF 23.5, ngspice-41 OSDI build.
- core fail -> not deployable; restriction -> deployable with restrictions (I17-1 thresholds).


### Validity V1-V4 per arm

| arm | check | criterion | value | result |
|---|---|---:|---:|---:|
| A | V1 card-only deck (OSDI build) x exp(g) vs stored I25-1 rows (18 full + 18 LODO) | <= 1e-9 | 0 (36 rows) | pass |
| A | V2 unit tests (I17-1 list + test_i25_1_scratch) | all pass | OK | pass |
| A | V3 finite element-deck rows full / LODO; M2 cards; failed | 18 / 18; 18; 0 | 18 / 18; 18; 0 | pass |
| A | V4 deployed g: inputs / EPS / lambda*_full (results, inner) | sym_v2 / 0.01 / 5, 5 | sym_v2 / 0.01 / 5, 5 | pass |
| B | V1 card-only deck (OSDI build) x exp(g) vs stored I25-1 rows (18 full + 18 LODO) | <= 1e-9 | 0 (36 rows) | pass |
| B | V2 unit tests (I17-1 list + test_i25_1_scratch) | all pass | OK | pass |
| B | V3 finite element-deck rows full / LODO; M2 cards; failed | 18 / 18; 18; 0 | 18 / 18; 18; 0 | pass |
| B | V4 deployed g: inputs / EPS / lambda*_full (results, inner) | sym_v2 / 0.01 / 5, 5 | sym_v2 / 0.01 / 5, 5 | pass |

- arm A: valid; arm B: valid. Unit tests run with the default template; every NGSpice stage with CRYOML_PDK_TEMPLATE=mult1.


### Element-deck RRMS vs stored Python rows

| pair set | element deck | stored Python | channel model | max |d| per device | msgs |
|---|---:|---:|---:|---:|---:|
| arm A, 7p card + g, full data (card + g fitted on 18) | 0.0783 / 0.0757 | 0.0783 / 0.0757 | 0.0783 / 0.0757 | 3.42e-05 | 0 |
| arm A, 7p card + g, LODO (g without the device) | 0.0823 / 0.0795 | 0.0823 / 0.0795 | 0.0823 / 0.0795 | 3.33e-05 | 0 |
| arm B, 45p card + g, full data (card + g fitted on 18) | 0.0667 / 0.0635 | 0.0667 / 0.0635 | 0.0667 / 0.0635 | 2.30e-05 | 0 |
| arm B, 45p card + g, LODO (g without the device) | 0.0693 / 0.0661 | 0.0693 / 0.0661 | 0.0693 / 0.0661 | 5.52e-05 | 0 |

- RRMS: paper RRMS with the released card's curve set (frozen); values all-device / combined (combined = mean of the nMOS and pMOS family means).
- element deck: NGSpice with the OSDI element; stored Python: I25-1 row, NGSpice card current x exp(g); channel model: I_card + (exp(g) - 1) I_rep from the card + replica deck; d: max per-device |element - stored|.


### Element checks per device, arm A (7p card + g)

| device | Q1 ratio | Vds0 (A) | asym | Q2 | cap | default tol | gm | gds | msgs | runtime | RRMS py / element |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.03 | 1.20e-17 | 1.03e-06 | 1.44e-07 | 8.39e-16 | 0.001575 | 0 | 0 | 0 | 1.96 | 0.0200 / 0.0200 |
| nmos_L0p19_W7 | 0.11 | 4.68e-17 | 4.02e-06 | 1.75e-06 | 1.31e-15 | 0.001226 | 0 | 0 | 0 | 2.21 | 0.0214 / 0.0214 |
| nmos_L0p25_W1p6 | 0.03 | 9.90e-18 | 4.00e-07 | 2.01e-07 | 1.54e-15 | 0.001157 | 0 | 0 | 0 | 2.21 | 0.0217 / 0.0217 |
| nmos_L100_W100 | 1.57 | 3.84e-18 | 3.13e-05 | 3.23e-05 | 1.14e-14 | 0.001284 | 0 | 0 | 0 | 2.24 | 0.1144 / 0.1144 |
| nmos_L1_W1p6 | 0.03 | 4.51e-18 | 5.28e-07 | 3.22e-07 | 2.62e-15 | 0.001181 | 0 | 0 | 0 | 1.96 | 0.0384 / 0.0384 |
| nmos_L1_W3 | 0.06 | 1.07e-17 | 1.12e-06 | 8.31e-07 | 2.46e-15 | 9.95e-04 | 0 | 0 | 0 | 1.75 | 0.0970 / 0.0970 |
| nmos_L20_W0p64 | 0.01 | 1.83e-19 | 3.49e-07 | 3.25e-07 | 8.27e-15 | 0.001389 | 0 | 0 | 0 | 1.97 | 0.0712 / 0.0712 |
| nmos_L8_W1p6 | 0.03 | 8.03e-19 | 5.40e-07 | 7.93e-07 | 5.14e-15 | 0.001094 | 0 | 0 | 0 | 1.81 | 0.0333 / 0.0333 |
| pmos_L0p35_W0p55 | 0.01 | 7.56e-20 | 3.87e-07 | 1.98e-07 | 2.70e-15 | 0.001098 | 0 | 0 | 0 | 1.87 | 0.3217 / 0.3217 |
| pmos_L0p35_W1p6 | 0.00 | 7.71e-19 | 3.47e-14 | 5.16e-10 | 1.17e-16 | 0.001507 | 0 | 0 | 0 | 1.93 | 0.0848 / 0.0848 |
| pmos_L0p35_W5 | 0.06 | 1.53e-18 | 2.68e-06 | 1.62e-06 | 1.91e-15 | 0.001593 | 0 | 0 | 0 | 2.09 | 0.0573 / 0.0573 |
| pmos_L0p5_W0p42 | 0.01 | 1.11e-19 | 1.94e-07 | 1.94e-07 | 2.68e-15 | 0.001053 | 0 | 0 | 0 | 2.06 | 0.0895 / 0.0895 |
| pmos_L0p5_W0p64 | 0.01 | 1.09e-19 | 4.37e-07 | 3.04e-07 | 1.68e-15 | 0.001651 | 0 | 0 | 0 | 1.93 | 0.0956 / 0.0956 |
| pmos_L2_W5 | 0.00 | 5.55e-19 | 3.91e-14 | 1.15e-09 | 1.52e-16 | 0.00168 | 0 | 0 | 0 | 1.98 | 0.0511 / 0.0511 |
| pmos_L4_W7 | 0.11 | 5.15e-19 | 5.76e-06 | 7.47e-06 | 3.35e-15 | 0.001982 | 0 | 1 | 0 | 1.94 | 0.0721 / 0.0721 |
| pmos_L8_W0p84 | 0.02 | 1.75e-20 | 3.82e-07 | 9.72e-07 | 3.89e-15 | 0.001763 | 0 | 3 | 0 | 2.05 | 0.0915 / 0.0915 |
| pmos_L8_W1p6 | 0.03 | 3.71e-20 | 9.63e-07 | 2.52e-06 | 7.98e-15 | 0.001682 | 0 | 0 | 0 | 1.90 | 0.0749 / 0.0749 |
| pmos_L8_W5 | 0.05 | 1.50e-19 | 1.59e-06 | 6.09e-06 | 7.88e-15 | 0.001622 | 0 | 2 | 0 | 1.94 | 0.0534 / 0.0534 |

- q1: max |dI| / (1e-6 |I_py| + 1e-13 A), tight, forward and exchanged (pass <= 1); Vds0: max |I_elem(V(d,s) = 0)| (A), tight; asym: element-introduced |d ln(I_swap / I_fwd)|; q2: max normalized gm / gds difference; cap: capacitance error; def: default-tolerance relative difference (|I| > 1 nA); gm: gm curves with an introduced sign change in strong inversion (of 3); gds: gds curves with an introduced sign change (of 5); msgs: convergence lines in element runs; runtime: DC grid wall-time ratio element / card; RRMS: stored Python / element deck, full-data pair.


### Element checks per device, arm B (45p card + g)

| device | Q1 ratio | Vds0 (A) | asym | Q2 | cap | default tol | gm | gds | msgs | runtime | RRMS py / element |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.05 | 1.20e-17 | 1.65e-06 | 2.80e-07 | 1.30e-15 | 0.001021 | 0 | 0 | 0 | 1.94 | 0.0169 / 0.0169 |
| nmos_L0p19_W7 | 0.07 | 4.90e-17 | 1.19e-06 | 2.23e-06 | 2.28e-15 | 0.001525 | 0 | 0 | 0 | 1.97 | 0.0196 / 0.0196 |
| nmos_L0p25_W1p6 | 0.02 | 1.02e-17 | 5.63e-07 | 1.35e-07 | 1.50e-15 | 0.001391 | 0 | 0 | 0 | 1.98 | 0.0205 / 0.0205 |
| nmos_L100_W100 | 4.78 | 4.01e-18 | 8.51e-05 | 9.66e-05 | 3.26e-14 | 0.001294 | 1 | 0 | 0 | 1.95 | 0.0794 / 0.0795 |
| nmos_L1_W1p6 | 0.05 | 4.77e-18 | 2.78e-06 | 1.08e-06 | 2.08e-15 | 0.001381 | 0 | 0 | 0 | 2.05 | 0.0311 / 0.0311 |
| nmos_L1_W3 | 0.03 | 9.27e-18 | 1.06e-06 | 1.10e-06 | 3.26e-15 | 0.001398 | 0 | 0 | 0 | 2.24 | 0.0419 / 0.0419 |
| nmos_L20_W0p64 | inf | inf | 0 | 2.59e-07 | 1.13e-14 | 0.001078 | 1 | 1 | 539 | 1.89 | 0.0433 / 0.0433 |
| nmos_L8_W1p6 | 0.01 | 6.25e-19 | 8.33e-07 | 4.03e-07 | 8.07e-15 | 9.99e-04 | 0 | 1 | 0 | 1.97 | 0.0272 / 0.0272 |
| pmos_L0p35_W0p55 | 0.02 | 7.62e-20 | 8.29e-07 | 1.93e-07 | 5.43e-15 | 0.001588 | 0 | 0 | 0 | 2.02 | 0.3032 / 0.3032 |
| pmos_L0p35_W1p6 | 0.00 | 8.86e-19 | 7.26e-14 | 8.00e-10 | 2.05e-16 | 0.001506 | 0 | 0 | 0 | 1.91 | 0.0849 / 0.0849 |
| pmos_L0p35_W5 | 0.04 | 1.48e-18 | 1.48e-06 | 1.61e-06 | 1.78e-15 | 0.001563 | 0 | 0 | 0 | 1.98 | 0.0548 / 0.0548 |
| pmos_L0p5_W0p42 | 0.02 | 6.09e-19 | 7.11e-07 | 1.67e-07 | 5.03e-15 | 0.001269 | 0 | 0 | 0 | 1.88 | 0.0671 / 0.0672 |
| pmos_L0p5_W0p64 | 0.05 | 9.71e-20 | 3.09e-06 | 1.59e-06 | 1.36e-15 | 0.001549 | 0 | 0 | 0 | 2.04 | 0.0949 / 0.0949 |
| pmos_L2_W5 | 0.00 | 5.23e-19 | 5.71e-14 | 2.21e-09 | 0 | 0.001537 | 0 | 0 | 0 | 2.08 | 0.0395 / 0.0395 |
| pmos_L4_W7 | 0.04 | 9.65e-19 | 7.84e-07 | 1.03e-06 | 2.02e-15 | 0.001396 | 0 | 2 | 0 | 1.84 | 0.0687 / 0.0687 |
| pmos_L8_W0p84 | 0.01 | 2.48e-19 | 2.85e-07 | 3.44e-07 | 4.11e-15 | 0.001467 | 0 | 2 | 0 | 2.06 | 0.0686 / 0.0686 |
| pmos_L8_W1p6 | 0.01 | 9.70e-19 | 3.01e-07 | 3.74e-07 | 3.27e-15 | 0.001448 | 0 | 3 | 0 | 2.03 | 0.0926 / 0.0926 |
| pmos_L8_W5 | 0.05 | 6.91e-19 | 2.37e-06 | 3.23e-06 | 4.85e-15 | 0.001283 | 0 | 1 | 0 | 1.81 | 0.0456 / 0.0456 |

- q1: max |dI| / (1e-6 |I_py| + 1e-13 A), tight, forward and exchanged (pass <= 1); Vds0: max |I_elem(V(d,s) = 0)| (A), tight; asym: element-introduced |d ln(I_swap / I_fwd)|; q2: max normalized gm / gds difference; cap: capacitance error; def: default-tolerance relative difference (|I| > 1 nA); gm: gm curves with an introduced sign change in strong inversion (of 3); gds: gds curves with an introduced sign change (of 5); msgs: convergence lines in element runs; runtime: DC grid wall-time ratio element / card; RRMS: stored Python / element deck, full-data pair.


### Current at V(d,s) = 0 per tolerance

| tolerance | use | arm A, 7p + g | arm B, 45p + g | old 43p + g (I17-1) |
|---|---|---:|---:|---:|
| tight | core check (I17-1) | 4.676e-17 at nmos_L0p19_W7, |Vgs| 1.82 V; 0 of 1350 | inf at nmos_L20_W0p64, |Vgs| nan V; 0 of 1350 | 4.869e-17; 0 of 1350 |
| default | reported | 9.858e-13 at pmos_L0p5_W0p42, |Vgs| 0.625 V; 0 of 1350 | 9.521e-13 at pmos_L8_W0p84, |Vgs| 0.8 V; 0 of 1350 | 1.001e-12; 1 of 1350 |
| r4 | reported | 9.858e-13 at pmos_L0p5_W0p42, |Vgs| 0.625 V; 0 of 1350 | 9.521e-13 at pmos_L8_W0p84, |Vgs| 0.8 V; 0 of 1350 | 9.280e-13; 0 of 1350 |

- max |I_elem| at V(d,s) = 0 over 18 devices x 75 V(g,s) values; 'k of n' = points above 1e-12 A. Tolerances: default (ngspice abstol 1e-12 A, reltol 1e-3); r4 reltol 1e-4; tight reltol 1e-9, abstol 1e-18 A, vntol 1e-12 V.


### Q3 circuits (1.8 V supply)

| circuit | arm A | arm B |
|---|---|---|
| inv_A_vtc | mult1: ok Vm 0.798 V; card: ok Vm 0.756 V; M2 card: ok Vm 0.755 V; M2 + elem: ok Vm 0.756 V | mult1: ok Vm 0.798 V; card: ok Vm 0.761 V; M2 card: ok Vm 0.760 V; M2 + elem: ok Vm 0.761 V |
| inv_B_vtc | mult1: ok Vm 0.805 V; card: ok Vm 0.653 V; M2 card: ok Vm 0.653 V; M2 + elem: ok Vm 0.653 V | mult1: ok Vm 0.805 V; card: ok Vm 0.639 V; M2 card: ok Vm 0.639 V; M2 + elem: ok Vm 0.640 V |
| ro5_A | mult1: ok 3.081 GHz; card: ok 5.369 GHz; M2 card: ok 5.370 GHz; M2 + elem: ok 5.506 GHz | mult1: ok 3.081 GHz; card: ok 7.165 GHz; M2 card: ok 7.101 GHz; M2 + elem: ok 7.212 GHz |
| ro5_B | mult1: fail; card: ok 0.258 GHz; M2 card: ok 0.252 GHz; M2 + elem: ok 0.257 GHz | mult1: fail; card: ok 0.545 GHz; M2 card: ok 0.542 GHz; M2 + elem: ok 0.548 GHz |
| mirror_n | mult1: ok Iout/Iref(0.9 V) 1.022; card: ok Iout/Iref(0.9 V) 1.045; M2 card: ok Iout/Iref(0.9 V) 1.046; M2 + elem: ok Iout/Iref(0.9 V) 1.049 | mult1: ok Iout/Iref(0.9 V) 1.022; card: ok Iout/Iref(0.9 V) 1.036; M2 card: ok Iout/Iref(0.9 V) 1.036; M2 + elem: ok Iout/Iref(0.9 V) 1.048 |
| mirror_p | mult1: ok Iout/Iref(0.9 V) 1.001; card: ok Iout/Iref(0.9 V) 0.952; M2 card: ok Iout/Iref(0.9 V) 0.961; M2 + elem: ok Iout/Iref(0.9 V) 0.969 | mult1: ok Iout/Iref(0.9 V) 1.001; card: ok Iout/Iref(0.9 V) 0.980; M2 card: ok Iout/Iref(0.9 V) 0.980; M2 + elem: ok Iout/Iref(0.9 V) 0.975 |

- Q3 runs per circuit: multiplier-1 template card (mult1), I25-1 C0_refit card (card), M2 card = card fitted with g, without the element (M2 card), and with the element (M2 + elem). Core check: M2 + elem fails while M2 card passes. inv: inverter VTC (Vm = switching voltage); ro5: 5-stage ring oscillator; mirror: current mirror, Iref 10 uA.


### Restriction items per arm

| arm | item | value |
|---|---|---|
| A | default tolerance > 1e-3: devices; max | 17 of 18; 0.001982 |
| A | gds curves with an introduced sign change | 6 of 90: pmos_L4_W7 Vgs -1.85 V (757 pts), pmos_L8_W0p84 Vgs -1.2 V (108 pts), pmos_L8_W0p84 Vgs -1.5 V (771 pts), pmos_L8_W0p84 Vgs -1.85 V (848 pts), pmos_L8_W5 Vgs -1.5 V (244 pts), pmos_L8_W5 Vgs -1.85 V (563 pts) |
| A | negative gds points: element / card x exp(g) / card | 3291 / 3291 / 0 |
| A | runtime element / card: DC grid; circuits | 1.75-2.24; inv_A_vtc 1.77, inv_B_vtc 1.76, ro5_A 4.49, ro5_B 3.65, mirror_n 1.21, mirror_p 1.26 |
| A | ro5_A wall time card + replica / card; element / card | 2.03; 4.65 |
| B | default tolerance > 1e-3: devices; max | 17 of 18; 0.001588 |
| B | gds curves with an introduced sign change | 10 of 90: nmos_L8_W1p6 Vgs 1.85 V (21 pts), pmos_L4_W7 Vgs -1.5 V (97 pts), pmos_L4_W7 Vgs -1.85 V (298 pts), pmos_L8_W0p84 Vgs -1.2 V (236 pts), pmos_L8_W0p84 Vgs -1.5 V (100 pts), pmos_L8_W1p6 Vgs -1.2 V (184 pts), pmos_L8_W1p6 Vgs -1.5 V (157 pts), pmos_L8_W1p6 Vgs -1.85 V (145 pts), pmos_L8_W5 Vgs -1.5 V (99 pts) |
| B | negative gds points: element / card x exp(g) / card | 1509 / 1519 / 182 |
| B | runtime element / card: DC grid; circuits | 1.81-2.24; inv_A_vtc 1.73, inv_B_vtc 1.78, ro5_A 4.30, ro5_B 4.19, mirror_n 1.21, mirror_p 1.27 |
| B | ro5_A wall time card + replica / card; element / card | 1.96; 4.36 |

- Restriction items (I17-1): reported in the label, not a core failure. card x exp(g): Python model on the dense card sweep.


### Core failures by device (diagnosis, reported)

| arm | device | check | value |
|---|---|---|---|
| A | nmos_L100_W100 | Q1 equivalence, tight, forward bench | 26 points; |dI| <= 1.68e-13 A (ratio <= 1.57); |I| 1.8e-12-5.0e-08 A, |Vgs| 0-0.425 V, |Vds| 1.52-1.85 V |
| A | nmos_L100_W100 | Q1 equivalence, tight, exchanged bench | 17 points; |dI| <= 1.63e-13 A (ratio <= 1.56); |I| 1.8e-12-5.0e-08 A, |Vgs| 0.125-0.425 V, |Vds| 1.52-1.85 V |
| B | nmos_L100_W100 | Q1 equivalence, tight, forward bench | 154 points; |dI| <= 5.57e-13 A (ratio <= 4.78); |I| 1.4e-12-1.6e-07 A, |Vgs| 0-0.425 V, |Vds| 0.725-1.85 V |
| B | nmos_L100_W100 | Q1 equivalence, tight, exchanged bench | 145 points; |dI| <= 5.66e-13 A (ratio <= 4.15); |I| 1.4e-12-2.7e-07 A, |Vgs| 0.075-0.425 V, |Vds| 0.7-1.85 V |
| B | nmos_L100_W100 | gm sign change in strong inversion, |Vds| 0.05 V | negative gm points: element 51, card x exp(g) 51, card 0 (of 1170) |
| B | nmos_L20_W0p64 | Q1 runs, V(d,s) = 0, Q1 equivalence (tight) | convergence lines: element tight grids 167 / 271, card + replica tight 153; exchanged card + replica tight run timed out (300 s); every tight element grid point non-finite (22350) |
| B | nmos_L20_W0p64 | card only, no replica, no element (diagnostic) | tight forward grid convergence lines: M2 card 4, C0_refit card 4, multiplier-1 card 0; default tolerance 0 / 0 / 0 |
| B | nmos_L20_W0p64 | gm curve |Vds| 1.85 V, gds curve |Vgs| 1.85 V | dense element sweep incomplete (1724 / 1372 of 1851 points); counted as introduced by the rule |

- Reported without a rule (the label follows the pre-registered checks). ratio = |dI| / (1e-6 |I| + 1e-13 A). Old 43p card + g (I17-1): nmos_L100_W100 0 points, ratio <= 0.11.

