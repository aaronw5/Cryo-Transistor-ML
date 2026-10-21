# Tables, 2026-10-21 i11_1

Generated 2026-10-21 18:14 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i11_1.py --date 2026-10-21` from stored files only.

### I11-1 pre-registered checks and outcome

| Q | check | value | threshold | result |
|---|---|---:|---:|---|
| Q1 | tight max rel. diff., |I| > 1 nA (core) | 2.59e-05 | <= 1e-05 | fail |
| Q1 | convergence msgs + non-finite, element runs (core) | 0 | 0 | pass |
| Q1 | max |I_elem(Vds = 0)| (core) | 9.59e-13 A | <= 1e-12 A | pass |
| Q2 | AC derivative error gm, gds, Im Y (core) | 6.35e-03 | <= 1e-03 | fail |
| Q2 | gm curves with introduced sign change (core) | 1 | 0 of 54 | fail |
| Q3 | circuits failing only with the element (core) | 0 | 0 of 6 | pass |
| Q1 | default-tolerance max rel. diff. (restriction) | 1.94e-03 | <= 1e-03 | restriction |
| Q1 | terminal-exchange asymmetry of the element (restriction) | 0.117 | <= 1e-03 | restriction |
| Q2 | drain displacement current changed (restriction) | yes | no | restriction |
| Q2 | gds curves with introduced sign change (restriction) | 27 | 0 of 90 | restriction |
| Q4 | max wall-time ratio element / card (restriction) | 3.25 | <= 2 | restriction |
| Q3 | card-only circuit runs not ok (reported) | 2 | — | ro5_B|C0_refit, ro5_B|M_card |

- Outcome (pre-registered rule, prereg commit da03859): not deployable.
- Core checks failed: Q1 tight-tolerance equivalence; Q2 derivative agreement; Q2 gm sign change in strong inversion.
- Restrictions: default tolerance above 1e-3 (tightened reltol needed for equivalence); not source/drain symmetric (g uses the instance source terminal); drain displacement current scaled by exp(g); gds sign changes introduced; runtime overhead > 2x.
- 18 devices, M_full card (43-parameter card refit with the full-data g in the loop) + full-data g; tight = reltol 1e-9, abstol 1e-18, vntol 1e-12; no measured circuit data (Q3 is a simulator-level check).


### Per-device Q1 / Q2 values (M_full card + full-data g)

| device | tight | 1e-4 | default | >=1uA | Vds0 A | swap | gm/Ydg | gds/Ydd | gm | gds | Cdd ratio | /exp(g) | g min/max | RRMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nmos_L0p15_W1p6 | 1.77e-06 | 4.03e-04 | 9.93e-04 | 9.93e-04 | 9.36e-13 | 0.043 | 4.05e-10 | 2.02e-07 | 0 | 0 | 0.942-1.033 | 1.49e-11 | -0.060/+0.033 | 0.0208 |
| nmos_L0p19_W7 | 2.59e-05 | 6.78e-04 | 1.09e-03 | 1.09e-03 | 7.15e-13 | 0.043 | 3.14e-10 | 5.63e-06 | 0 | 0 | 0.942-1.033 | 1.50e-11 | -0.060/+0.033 | 0.0175 |
| nmos_L0p25_W1p6 | 8.20e-06 | 3.80e-04 | 9.98e-04 | 9.98e-04 | 9.12e-13 | 0.043 | 1.71e-10 | 2.98e-08 | 0 | 0 | 0.942-1.033 | 1.49e-11 | -0.060/+0.033 | 0.0160 |
| nmos_L100_W100 | 4.67e-07 | 1.39e-04 | 1.76e-03 | 1.76e-03 | 3.59e-18 | 0.043 | 6.35e-03 | 3.18e-07 | 1 | 0 | 0.942-1.033 | 1.49e-11 | -0.060/+0.033 | 0.0673 |
| nmos_L1_W1p6 | 1.09e-05 | 6.36e-04 | 9.97e-04 | 9.97e-04 | 3.98e-13 | 0.043 | 1.78e-09 | 1.30e-07 | 0 | 0 | 0.942-1.033 | 1.48e-11 | -0.060/+0.033 | 0.0236 |
| nmos_L1_W3 | 3.63e-06 | 5.94e-04 | 1.36e-03 | 1.00e-03 | 8.00e-13 | 0.043 | 2.50e-10 | 4.23e-07 | 0 | 0 | 0.942-1.033 | 1.47e-11 | -0.060/+0.033 | 0.0471 |
| nmos_L20_W0p64 | 8.71e-07 | 7.74e-04 | 1.01e-03 | 1.01e-03 | 7.26e-13 | 0.043 | 7.22e-10 | 3.34e-07 | 0 | 0 | 0.942-1.033 | 1.49e-11 | -0.060/+0.033 | 0.0390 |
| nmos_L8_W1p6 | 3.87e-06 | 6.33e-04 | 1.47e-03 | 9.93e-04 | 7.23e-19 | 0.043 | 1.26e-08 | 3.59e-06 | 0 | 2 | 0.942-1.033 | 1.51e-11 | -0.060/+0.033 | 0.0178 |
| pmos_L0p35_W0p55 | 3.67e-07 | 8.39e-04 | 1.30e-03 | 1.25e-03 | 7.28e-13 | 0.117 | 4.35e-10 | 1.09e-07 | 0 | 0 | 0.688-1.222 | 2.58e-11 | -0.374/+0.201 | 0.3250 |
| pmos_L0p35_W1p6 | 9.22e-10 | 9.86e-04 | 1.39e-03 | 1.00e-03 | 6.99e-13 | 0.117 | 3.83e-11 | 1.15e-09 | 0 | 0 | 0.688-1.222 | 2.58e-11 | -0.374/+0.201 | 0.0984 |
| pmos_L0p35_W5 | 1.42e-05 | 9.66e-04 | 1.42e-03 | 1.42e-03 | 9.47e-13 | 0.117 | 1.79e-08 | 5.39e-06 | 0 | 0 | 0.688-1.222 | 2.59e-11 | -0.374/+0.201 | 0.0569 |
| pmos_L0p5_W0p42 | 7.08e-07 | 9.58e-04 | 1.45e-03 | 1.26e-03 | 7.65e-13 | 0.117 | 4.37e-10 | 9.09e-08 | 0 | 4 | 0.688-1.222 | 2.59e-11 | -0.374/+0.201 | 0.0932 |
| pmos_L0p5_W0p64 | 1.13e-06 | 9.49e-04 | 1.83e-03 | 1.83e-03 | 8.91e-13 | 0.117 | 1.54e-09 | 1.42e-07 | 0 | 0 | 0.688-1.222 | 2.59e-11 | -0.374/+0.201 | 0.1065 |
| pmos_L2_W5 | 2.60e-11 | 1.04e-03 | 1.75e-03 | 1.00e-03 | 6.40e-13 | 0.117 | 3.82e-11 | 5.88e-07 | 0 | 5 | 0.688-1.222 | 2.58e-11 | -0.374/+0.201 | 0.0610 |
| pmos_L4_W7 | 1.05e-05 | 8.16e-04 | 1.85e-03 | 1.37e-03 | 6.76e-13 | 0.117 | 1.39e-08 | 7.26e-06 | 0 | 5 | 0.688-1.222 | 2.60e-11 | -0.374/+0.201 | 0.0693 |
| pmos_L8_W0p84 | 1.54e-06 | 8.55e-04 | 1.82e-03 | 1.82e-03 | 8.08e-13 | 0.117 | 2.78e-10 | 3.68e-07 | 0 | 4 | 0.688-1.222 | 2.59e-11 | -0.374/+0.201 | 0.0702 |
| pmos_L8_W1p6 | 1.67e-06 | 7.20e-04 | 1.76e-03 | 1.71e-03 | 9.06e-13 | 0.117 | 2.76e-09 | 1.11e-06 | 0 | 4 | 0.688-1.222 | 2.58e-11 | -0.374/+0.201 | 0.0761 |
| pmos_L8_W5 | 8.46e-06 | 7.05e-04 | 1.94e-03 | 1.42e-03 | 9.59e-13 | 0.117 | 4.92e-09 | 4.40e-06 | 0 | 3 | 0.688-1.222 | 2.60e-11 | -0.374/+0.201 | 0.0489 |

- q1: max |I_elem - I_py| / |I_py| on the 75 x 149 grid, |I_py| > 1 nA; columns tight / reltol 1e-4 / default, and default at |I| >= 1 uA. Vds0: max |I_elem| at Vds = 0 (default and tight).
- swap: max |ln(I_swap/I_fwd)_elem - ln(I_swap/I_fwd)_card|, forward operation (terminal exchange).
- AC err: max normalized difference of the OSDI AC admittance vs exp(g) Y_card + I exp(g) dg/dV (real and imaginary parts); gm / gds sign: curves (of 3 / 5) with an introduced negative value.
- Cdd ratio: Im Y_dd(elem) / Im Y_dd(card) range; /exp(g): max |ratio / exp(g) - 1|.
- g range on the grid (incl. Vds < 0); RRMS: in-sample M_full SPICE-native RRMS on the measured curves.


### Q3 circuits: outputs per arm

| circuit | devices | arm | output | ok | first message | wall s |
|---|---|---|---|---:|---|---:|
| inv_A_vtc | nmos_L0p15_W1p6 + pmos_L0p35_W1p6 | published | V_M 0.7466 V, max gain 25.9 | yes | — | 0.051 |
| inv_A_vtc | nmos_L0p15_W1p6 + pmos_L0p35_W1p6 | C0_refit | V_M 0.7593 V, max gain 24.6 | yes | — | 0.046 |
| inv_A_vtc | nmos_L0p15_W1p6 + pmos_L0p35_W1p6 | M_card | V_M 0.7592 V, max gain 25.1 | yes | — | 0.043 |
| inv_A_vtc | nmos_L0p15_W1p6 + pmos_L0p35_W1p6 | M_full | V_M 0.7598 V, max gain 28.9 | yes | — | 0.067 |
| inv_B_vtc | nmos_L1_W1p6 + pmos_L2_W5 | published | V_M 0.6417 V, max gain 61.8 | yes | — | 0.050 |
| inv_B_vtc | nmos_L1_W1p6 + pmos_L2_W5 | C0_refit | V_M 0.6491 V, max gain 184.8 | yes | — | 0.045 |
| inv_B_vtc | nmos_L1_W1p6 + pmos_L2_W5 | M_card | V_M 0.6504 V, max gain 187.9 | yes | — | 0.049 |
| inv_B_vtc | nmos_L1_W1p6 + pmos_L2_W5 | M_full | V_M 0.6510 V, max gain 254.1 | yes | — | 0.078 |
| ro5_A | nmos_L0p15_W1p6 + pmos_L0p35_W1p6 | published | 5.3217 GHz, 36 periods | yes | — | 0.419 |
| ro5_A | nmos_L0p15_W1p6 + pmos_L0p35_W1p6 | C0_refit | 5.2868 GHz, 36 periods | yes | — | 0.409 |
| ro5_A | nmos_L0p15_W1p6 + pmos_L0p35_W1p6 | M_card | 5.2971 GHz, 36 periods | yes | — | 0.406 |
| ro5_A | nmos_L0p15_W1p6 + pmos_L0p35_W1p6 | M_full | 5.3820 GHz, 37 periods | yes | — | 1.317 |
| ro5_B | nmos_L1_W1p6 + pmos_L2_W5 | published | 0.2758 GHz, 9 periods | yes | — | 0.367 |
| ro5_B | nmos_L1_W1p6 + pmos_L2_W5 | C0_refit | stopped (1128 points) | no | doanalyses: tran:  timestep too small; time = 4.64014e-09, t | 0.163 |
| ro5_B | nmos_L1_W1p6 + pmos_L2_W5 | M_card | stopped (666 points) | no | doanalyses: tran:  timestep too small; time = 2.48289e-09, t | 0.142 |
| ro5_B | nmos_L1_W1p6 + pmos_L2_W5 | M_full | stopped (87 points) | no | doanalyses: tran:  timestep too small; time = 2.03028e-10, t | 0.182 |
| mirror_n | nmos_L1_W1p6 | published | ratio 1.0594 @0.9 V, 1.00000 @V(ref) 0.705 V | yes | — | 0.031 |
| mirror_n | nmos_L1_W1p6 | C0_refit | ratio 1.0225 @0.9 V, 1.00000 @V(ref) 0.730 V | yes | — | 0.031 |
| mirror_n | nmos_L1_W1p6 | M_card | ratio 1.0228 @0.9 V, 1.00000 @V(ref) 0.730 V | yes | — | 0.030 |
| mirror_n | nmos_L1_W1p6 | M_full | ratio 1.0330 @0.9 V, 0.99995 @V(ref) 0.731 V | yes | — | 0.039 |
| mirror_p | pmos_L2_W5 | published | ratio 0.9375 @0.9 V, 1.00000 @V(ref) 0.515 V | yes | — | 0.033 |
| mirror_p | pmos_L2_W5 | C0_refit | ratio 0.9920 @0.9 V, 1.00000 @V(ref) 0.526 V | yes | — | 0.035 |
| mirror_p | pmos_L2_W5 | M_card | ratio 0.9920 @0.9 V, 1.00000 @V(ref) 0.526 V | yes | — | 0.034 |
| mirror_p | pmos_L2_W5 | M_full | ratio 0.9692 @0.9 V, 1.00001 @V(ref) 0.527 V | yes | — | 0.038 |

- VDD 1.8 V, 77 K, default tolerances, default sky130 capacitances. Arms: published card; C0_refit card (stored all18 run); M_card = M_full card without g; M_full = M_full card + g (OSDI element).
- No measured circuit data: these are simulator-level consistency and convergence checks, not accuracy claims. ok = no convergence message, finite output, ring >= 5 periods.


### Post hoc diagnosis of the failed core checks and the gds restriction

| device | |Vgs| V | card S | element S | measured S | negative range V | min gm/gds S |
|---|---:|---:|---:|---:|---:|---:|
| nmos_L8_W1p6 | 0.90 | 1.97e-07 | 1.56e-07 | — | 0.570-0.749 | -4.77e-07 |
| nmos_L8_W1p6 | 1.20 | 5.37e-07 | 4.46e-07 | — | 0.662-0.810 | -1.38e-06 |
| pmos_L0p5_W0p42 | 0.60 | 1.02e-11 | 9.65e-12 | — | 1.460-1.850 | -4.22e-11 |
| pmos_L0p5_W0p42 | 0.90 | 7.50e-10 | 2.65e-10 | — | 0.441-1.850 | -2.90e-09 |
| pmos_L0p5_W0p42 | 1.20 | 1.48e-07 | 8.51e-08 | — | 1.473-1.850 | -1.79e-07 |
| pmos_L0p5_W0p42 | 1.50 | 1.75e-06 | 1.34e-06 | 2.61e-06 | 1.608-1.850 | -6.78e-07 |
| pmos_L2_W5 | 0.60 | 1.19e-12 | 1.16e-12 | — | 1.683-1.850 | -1.59e-12 |
| pmos_L2_W5 | 0.90 | 1.12e-10 | -7.87e-11 | — | 0.357-1.850 | -1.43e-09 |
| pmos_L2_W5 | 1.20 | 3.90e-08 | -4.15e-08 | — | 0.375-1.850 | -7.16e-07 |
| pmos_L2_W5 | 1.50 | 1.13e-06 | -2.32e-08 | 1.88e-06 | 0.508-1.850 | -6.22e-06 |
| pmos_L2_W5 | 1.85 | 6.67e-06 | 6.13e-06 | 8.71e-06 | 1.635-1.850 | -3.71e-06 |
| pmos_L4_W7 | 0.60 | 1.26e-11 | 1.26e-11 | — | 0.371-1.849 | -9.07e-12 |
| pmos_L4_W7 | 0.90 | 1.21e-09 | 1.04e-09 | — | 0.346-1.850 | -8.22e-10 |
| pmos_L4_W7 | 1.20 | 9.11e-08 | 3.52e-08 | — | 0.344-1.850 | -6.15e-07 |
| pmos_L4_W7 | 1.50 | 1.09e-06 | 2.25e-07 | 5.52e-07 | 0.441-1.850 | -1.10e-05 |
| pmos_L4_W7 | 1.85 | 6.21e-06 | 5.76e-06 | 4.38e-06 | 1.690-1.850 | -2.15e-06 |
| pmos_L8_W0p84 | 0.90 | 3.66e-11 | 3.24e-11 | — | 0.341-0.518 | -2.04e-11 |
| pmos_L8_W0p84 | 1.20 | 2.56e-09 | -1.13e-11 | — | 0.323-1.850 | -3.61e-08 |
| pmos_L8_W0p84 | 1.50 | 1.16e-08 | -2.26e-08 | 1.79e-08 | 0.556-1.850 | -4.36e-07 |
| pmos_L8_W0p84 | 1.85 | 3.22e-08 | 1.32e-08 | 8.98e-08 | 0.823-1.850 | -2.29e-07 |
| pmos_L8_W1p6 | 0.90 | 1.18e-10 | 9.24e-11 | — | 0.314-1.850 | -2.44e-10 |
| pmos_L8_W1p6 | 1.20 | 6.75e-09 | 2.43e-09 | — | 0.326-1.850 | -5.72e-08 |
| pmos_L8_W1p6 | 1.50 | 1.86e-08 | -4.98e-08 | 3.12e-08 | 0.412-1.850 | -1.81e-06 |
| pmos_L8_W1p6 | 1.85 | 2.96e-08 | -7.31e-09 | 1.86e-07 | 0.692-1.850 | -2.88e-06 |
| pmos_L8_W5 | 1.20 | 3.33e-08 | 1.55e-08 | — | 0.333-1.850 | -2.16e-07 |
| pmos_L8_W5 | 1.50 | 2.33e-07 | -2.73e-08 | 1.41e-07 | 0.441-1.850 | -4.27e-06 |
| pmos_L8_W5 | 1.85 | 1.12e-06 | 9.85e-07 | 6.84e-07 | 1.590-1.850 | -1.00e-06 |
| nmos_L100_W100 (gm) | |Vds| 0.05 | — | — | — | |Vgs| 1.757-1.850 | -5.48e-06 |

- Post hoc (after the pre-registered decision; does not change it).
- gds rows: curves with an introduced negative gds; average output conductance = slope of a linear fit of |I_D| over |Vds| 1.0-1.85 V for the card, the element and the measured curve at the nearest measured |Vgs| (within 0.04 V); range = |Vds| interval with gds < -1e-12 S on the 1 mV sweep.
- gm row: the gm curve with an introduced sign change (|Vgs| interval and minimum gm).
- Q1 tight points above 1e-5: nmos_L0p19_W7 6 points, |I| <= 2.6e-09 A, |dI| <= 2.6e-14 A; nmos_L1_W1p6 1 points, |I| <= 1.2e-09 A, |dI| <= 1.3e-14 A; pmos_L0p35_W5 13 points, |I| <= 2.2e-09 A, |dI| <= 2.8e-14 A; pmos_L4_W7 1 points, |I| <= 1.6e-09 A, |dI| <= 1.6e-14 A; max rel. diff. at |I| > 10 nA 1.5e-06 (gmin 1e-15 and nobypass do not remove the floor).
- Q2: the only AC point above 1e-3 lies on the line Vgs = 0 (abs() kink; OpenVAF takes a one-sided derivative, the Python reference a central difference); max off the lines Vgs = 0 and Vds = 0 7.3e-06.

