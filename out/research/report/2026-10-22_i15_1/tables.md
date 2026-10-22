# Tables, 2026-10-22 i15_1

Generated 2026-10-22 02:16 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i15_1.py --date 2026-10-22`.

### I15-1 Part B checks vs pre-registered thresholds: g v2 (outcome), I14-1 g v2 values, M_bias v1 (reference arm)

| check | type | threshold | g v2 value | g v2 | I14-1 g v2 | v1 reference value | v1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Q1 equivalence: points violating |dI| <= 1e-6 |I_py| + 1e-13 A (tight, both benches) | core | 0 | 0 | pass | 0 (pass) | 0 | pass |
| Q1 runs: convergence messages + non-finite values (element runs) | core | 0 | 0 | pass | 2 (FAIL) | 0 | pass |
| Q1 max |I_elem(Vds = 0)| (A) | core | <= 1e-12 | 9.84e-13 | pass | 9.84e-13 (pass) | 9.95e-13 | pass |
| Verilog-A g vs Python, max |dg| | core | <= 1e-9 | 2.50e-16 | pass | 2.50e-16 (pass) | 2.88e-16 | pass |
| Verilog-A g, exchanged vs forward terminals, max |dg| | core | <= 1e-12 | 3.33e-16 | pass | 3.33e-16 (pass) | 0.1344 | FAIL |
| exchange asymmetry, element-introduced max |d ln| | core | <= 1e-3 | 1.68e-05 | pass | 1.68e-05 (pass) | 0.1252 | FAIL |
| Q2 gm / gds vs element identity, max normalized difference | core | <= 1e-3 | 1.24e-05 | pass | 1.24e-05 (pass) | 0.006607 | FAIL |
| Q2 capacitances (Y_dd, Y_dg, Y_gg), max relative error | core | <= 1e-3 | 1.09e-12 | pass | 1.09e-12 (pass) | 1.04e-13 | pass |
| Q2 gm curves with an introduced sign change (of 54) | core | 0 | 0 | pass | 0 (pass) | 1 | FAIL |
| Q3 circuits failing only with the element | core | 0 | 0 | pass | 0 (pass) | 0 | pass |
| default tolerance, max relative difference (|I| > 1 nA) | restriction | <= 1e-3 (restriction) | 0.001814 | FAIL | 0.001805 (FAIL) | 0.001898 | FAIL |
| gds curves with an introduced sign change (of 90) | restriction | 0 (restriction) | 2 | FAIL | 2 (FAIL) | 28 | FAIL |
| wall-time ratio element / card only, max (grid and circuits) | restriction | <= 2 (restriction) | 4.298 | FAIL | 4.587 (FAIL) | 4.295 | FAIL |
| label |  |  | deployable with restrictions |  | not deployable | not deployable |  |

- v2: I13-1 full-data g v2 (state C) with its M2 cards (refit from the starting card with g in the loop, state C, 0 failed); v1: reference arm, I13-1 full-data M_bias v1 g (abs() inputs) with its own M2 cards; same element (channel-current replica, gmin VCCS, module I(d, s) <+ (exp(g) - 1) I(sense)).
- I15-1 change (test procedure): every Q1 grid deck sweeps V(d) up on even V(g) lines and down on odd lines (no 3.7 V restart jump); AC, dense, measured-curve and circuit decks as I14-1. I14-1 column: the I14-1 v2 values (nested sweeps).
- Q2 with the I12-1 formula exp(g) Re Y_card + I_card exp(g) dg/dV on the same runs: max 0.0217 (v2).
- Runtime: DC grid 1.84-2.08; circuits inv_A_vtc 1.76, inv_B_vtc 1.95, ro5_A 4.30, mirror_n 1.19, mirror_p 1.30; ro5_B excluded (fails at card level, I12-1 amendment A1).


### I15-1 validity

| check | criterion | value | result |
|---|---:|---:|---:|
| V1 card-only library deck x exp(fold g) vs I13-1 rows (36 rows) | <= 1e-9 | 0 | pass |
| V2 unit tests (test_i14_1_chan, test_i13_1_clamp, test_spice_batch, test_sky130_misspec, test_i12_1_symm, test_i15_1_serp) | all pass | OK | pass |
| V3 element-deck accuracy rows per arm / failed M2 refits | 18 / 0 | 18, 18 / 0, 0 | pass |

- Validity rule (prereg): any V fail -> invalid. Outcome: valid.


### I15-1 Part B per device, g v2 with its M2 card

| device | Q1 ratio | asym | asym card | Q2 | cap | default tol | gds sign | msgs | runtime | M2 RRMS py / native |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 0.05 | 1.74e-06 | 5.05e-07 | 3.91e-07 | 1.08e-15 | 0.00117 | 0 | 0 | 1.89 | 0.0285 / 0.0285 |
| nmos_L0p19_W7 | 0.22 | 1.12e-05 | 5.31e-07 | 9.22e-07 | 1.62e-15 | 0.001184 | 0 | 0 | 2.01 | 0.0199 / 0.0200 |
| nmos_L0p25_W1p6 | 0.06 | 2.74e-06 | 1.01e-07 | 1.65e-06 | 1.37e-15 | 0.001005 | 0 | 0 | 2.08 | 0.0188 / 0.0188 |
| nmos_L100_W100 | 0.37 | 6.47e-07 | 3.00e-07 | 2.61e-07 | 3.06e-14 | 9.95e-04 | 0 | 0 | 2.03 | 0.0696 / 0.0695 |
| nmos_L1_W1p6 | 0.10 | 1.63e-06 | 2.17e-07 | 2.56e-06 | 2.22e-15 | 0.001056 | 0 | 0 | 2.01 | 0.0285 / 0.0285 |
| nmos_L1_W3 | 0.05 | 7.82e-07 | 6.67e-08 | 3.99e-07 | 2.99e-15 | 0.001238 | 0 | 0 | 2.03 | 0.0491 / 0.0491 |
| nmos_L20_W0p64 | 0.03 | 1.80e-06 | 8.86e-07 | 3.17e-07 | 7.85e-15 | 0.001523 | 0 | 0 | 2.00 | 0.0502 / 0.0500 |
| nmos_L8_W1p6 | 0.08 | 6.72e-06 | 6.73e-08 | 5.07e-06 | 5.02e-15 | 0.001218 | 0 | 0 | 2.03 | 0.0222 / 0.0222 |
| pmos_L0p35_W0p55 | 0.01 | 3.59e-07 | 1.70e-07 | 1.51e-07 | 1.28e-15 | 0.001548 | 0 | 0 | 1.84 | 0.3433 / 0.3433 |
| pmos_L0p35_W1p6 | 0.00 | 1.34e-09 | 2.02e-08 | 5.39e-10 | 0 | 0.001343 | 0 | 0 | 1.92 | 0.1199 / 0.1199 |
| pmos_L0p35_W5 | 0.22 | 1.68e-05 | 2.53e-07 | 2.01e-06 | 2.91e-15 | 0.001805 | 0 | 0 | 2.02 | 0.0620 / 0.0620 |
| pmos_L0p5_W0p42 | 0.02 | 6.92e-07 | 1.08e-07 | 3.33e-07 | 1.09e-12 | 0.001563 | 0 | 0 | 1.96 | 0.1065 / 0.1065 |
| pmos_L0p5_W0p64 | 0.02 | 6.75e-07 | 1.05e-07 | 3.17e-07 | 2.18e-15 | 0.001723 | 0 | 0 | 1.92 | 0.1221 / 0.1221 |
| pmos_L2_W5 | 0.00 | 6.82e-14 | 6.93e-14 | 1.89e-09 | 2.21e-16 | 0.001039 | 0 | 0 | 1.94 | 0.0483 / 0.0483 |
| pmos_L4_W7 | 0.21 | 8.44e-06 | 9.78e-08 | 8.50e-06 | 2.38e-15 | 0.001814 | 0 | 0 | 1.85 | 0.0568 / 0.0568 |
| pmos_L8_W0p84 | 0.03 | 7.79e-07 | 6.45e-08 | 8.62e-07 | 3.05e-15 | 0.001065 | 1 | 0 | 2.04 | 0.0870 / 0.0870 |
| pmos_L8_W1p6 | 0.03 | 1.54e-06 | 6.07e-08 | 8.50e-07 | 1.54e-14 | 9.91e-04 | 1 | 0 | 1.84 | 0.0817 / 0.0817 |
| pmos_L8_W5 | 0.21 | 5.88e-06 | 8.22e-08 | 1.24e-05 | 2.12e-15 | 0.00117 | 0 | 0 | 1.87 | 0.0461 / 0.0461 |

- q1: max |dI| / (1e-6 |I_py| + 1e-13 A), tight, forward and exchanged (pass <= 1); asym: element-introduced |d ln(I_swap / I_fwd)|, card: the card's own; q2: max normalized gm / gds difference vs the element identity; cap: capacitance error; def: default-tolerance relative difference (|I| > 1 nA); gds: gds curves with an introduced sign change (of 5); msg: convergence lines in element runs; rt: DC grid wall-time ratio; M2 RRMS in-sample: Python (I_card exp(g)) / SPICE-native element.


### I15-1 element-deck accuracy of the held-out pairs vs the I13-1 rows

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


### I15-1 pre-registration: replica variants and the serpentine Q1 sweep, decks with convergence warnings

| variant | decks | decks with warnings | where | reverse-quadrant junction current in sense |
|---|---:|---:|---|---:|
| I14-1 replica (zero junction saturation current) | 864 | 0 | - | 1.7e-12 A |
| junction saturation current = card values x 1e-20 | 864 | 1 | v2_heldout_C|pmos_L0p35_W5|fwd_tight_elem | 0.039 A |
| card values x 1e-10 | 864 | 1 | start_C|pmos_L8_W5|fwd_tight_elem | 0.046 A |
| card values x 1 | 864 | 1 | v1_heldout_C|pmos_L0p5_W0p64|fwd_tight_elem | 0.052 A |
| rbodymod = 0 replica | 1008 | 3 | rb0|start_C|pmos_L8_W0p84|fwd_tight_card_rep; rb0|v2_heldout_C|nmos_L0p25_W1p6|swap_tight_card_rep; rb0|v2_heldout_C|nmos_L0p25_W1p6|swap_tight_elem | - |
| serpentine Q1 grid, I14-1 replica (registered) | 1296 | 0 (nested: 0) | - | - |
| nmos_L8_W1p6 I14-1 M2 card (disclosed), nested / serpentine | 18 / 18 | 2 / 0 | nested_fwd_tight_card_rep; nested_swap_tight_elem | - |

- Pre-registration work (I15-1 prereg 'done before'; 4 card sets x 18 devices: published, start state C, I13-1 v2 and v1 held-out cards; forward / exchanged x default / r4 / tight x card + replica / element). Last column: max |sensed current - replica BSIM4 channel current| in the reverse quadrant (forward bench, tight): the forward-biased drain-junction current that a small-saturation-current replica adds to the sensed current.
- Exact subtraction with a second, junction-only replica (u0 = 1e-30) through a CCCS: exact to 5.7e-5 of the Q1 tolerance (exchanged bench) but the forward tight card + replica run on the nmos_L8_W1p6 M2 card aborts (timestep too small): the sense branch carries the difference of two ~0.04 A currents.
- Serpentine - nested, tight: max |d| / (1e-6 |I| + 1e-13 A) element 2.10; default-tolerance element max relative 2.0e-03 (solver tolerance).


### I15-1 Q3 circuits (default tolerance)

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


### I15-1 restriction items of g v2 (reported)

| item | value | detail |
|---|---:|---|
| default-tolerance difference, max relative (|I| > 1 nA) | 1.81e-03 | 16 of 18 devices above 1e-3; r4 max 9.9e-04; tight max 1.8e-05 |
| gds sign change: pmos_L8_W0p84, V(g,s) = -1.85 V | 506 points dI/dV(d,s) < 0 | card 0, I_card exp(g) 506; g in [-0.306, 0.115] |
| gds sign change: pmos_L8_W1p6, V(g,s) = -1.85 V | 914 points dI/dV(d,s) < 0 | card 0, I_card exp(g) 914; g in [-0.306, 0.115] |
| runtime, DC grid ratio (18 devices) | 1.84-2.08 | circuits: inv_A_vtc 1.76, inv_B_vtc 1.95, ro5_A 4.30, mirror_n 1.19, mirror_p 1.30 |
| ro5_A decomposition (M2 cards), wall s card / card + replica / element | 0.259 / 0.524 / 1.212 | rusage load s 0.129 / 0.288 / 0.945; iterations 23335 / 26998 / 28427 |

- Restriction items (reported, no rule). gds: 1 mV dense sweeps, tight; points with |V(d,s)| >= 0.05 V. Runtime: pre-registration measurement on the I14-1 M2 cards: an element with a trivial module (same ports, g = 1e-3 V(g,s)) took 0.52 s against 1.13 s for the g v2 module: the network evaluation in the OSDI module is the largest part of the added time; no runtime-reduction variant was registered.

