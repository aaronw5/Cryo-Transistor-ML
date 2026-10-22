# Tables, 2026-10-22 i17_2

Generated 2026-10-22 12:51 by `.venv/bin/python -I scripts/research/plot_explore_i17_2.py --date 2026-10-22`.

### I17-2 dataset choice, setup and why Part B was not run (pre-registered)

| item | value |
|---|---|
| dataset chosen | IHP SG13G2 (I5-1 data), 300 K, DEV 18 DUTs from I5_1/split.json; TEST locked |
| reason | only measured dataset with a non-sky130 foundry PDK card; GF180 has no measured data; TU Delft uses a generic example card |
| candidates | IHP: measured, PSP103 OSDI 0.4, ngspice-46; TU_Delft: measured, BSIM4 CMOSedu example card, ngspice-41; GF180: simulated only |
| simulator | Homebrew ngspice-46 + data/raw/pdk2/ihp_sg13g2/osdi/psp103.osdi (I5-1) |
| BSIM4 clamp | not applicable (PSP103) |
| folds | leave-one-DUT-out within class (LV 9 / HV 7 training DUTs), record load order |
| penalty grid | u_g, u_d in {0, VDD/74, ..., VDD} (75 x 75), forward quadrant, Vb = 0, 300 K, starting card; points with forward current > 1e-09 A |
| Part B | not run |
| Part B reason 1 | BSIM4 channel replica does not apply; PSP103 replica needs SWIGATE=SWIMPACT=SWGIDL=SWJUNCAP=0, zero overlap/fringe charge parameters, VCCS for the DI-BP and SI-BP gmin conductances |
| Part B reason 2 | PSP103 intrinsic charges Qg, Qb, Qd are contributed unconditionally (PSP103_module.include 1788-1790): the sensed replica current includes displacement current, the Q2 capacitance check fails by construction; needs an edited PSP103 Verilog-A, a different element |
| Part B reason 3 | ngspice-41 OSDI build loads OSDI 0.3 only; psp103.osdi is OSDI 0.4 |
| Part B reason 4 | Q3 circuits are sky130 netlists |

- Pre-registration out/research/explore/I17_2/prereg.md, commit c071472.


### I17-2 validity

| check | criterion | value |
|---|---:|---|
| V1 C0_refit rerun vs I5-1 rows (18), max |d| | <= 1e-9 | 0.0e+00 |
| V1 M_bias v1 rerun vs I5-1 rows (18), max |d| | <= 1e-9 | 0.0e+00 |
| V2 unit tests | all pass | True |
| V3 rows per arm | 18 | C0_refit 18, M_bias_v1 18, sym_e1_l0 18, sym_e1_l1 18, sym_e1_l2 18, sym_e1_l5 18, sym_e1_l10 18, selected 18 |
| V3 inner selection complete | 18 x 3 x 5 | True (270 trainings, 730 inner refits) |
| V4 penalty grids 75 x 75 finite, forward corner current > 0 | 18 | True |

- Result: valid.


### I17-2 pre-registered rules: values, labels and predicted signs

| quantity | GMR [95% CI] | lower | rule | label | predicted sign |
|---|---:|---:|---|---|---|
| primary: GMR(selected, C0_refit) | 1.255 [1.026, 1.600] | 9/18 | CI upper < 1 and 0 failed: replicated; CI lower > 1: worse; else inconclusive | worse than C0_refit | ln > 0: correct |
| secondary: GMR(selected, M_bias v1) | 1.034 [0.990, 1.107] | 7/18 | CI upper <= 1.05: non-inferior | not shown non-inferior to M_bias v1 | ln < 0: wrong |
| GMR(M_bias v1, C0_refit) (reported) | 1.214 [1.008, 1.537] | 9/18 | - | - | ln > 0: correct |
| I5-1 stored GMR(M_bias, C0_refit) | 1.214 [1.008, 1.537] | 9/18 | - | - | - |
| Part B | - | - | - | not run | - |

- GMR: exp(mean ln(method / control)) over 18 DEV DUTs, device bootstrap 10,000, 95% CI. Failed DUTs (selected, C0_refit): 0.


### I17-2 held-out RRMS of every arm, 18 IHP DEV DUTs, 300 K (LODO within class)

| arm | all-device | combined | median | GMR vs C0_refit | cluster CI | GMR vs M_bias v1 | cluster CI  |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0_refit | 0.01906 | 0.01906 | 0.01336 | 1 | - | 0.823 [0.650, 0.992] | 0.823 [0.655, 1.011] |
| M_bias_v1 | 0.02029 | 0.02029 | 0.01424 | 1.214 [1.008, 1.537] | 1.214 [0.989, 1.527] | 1 | - |
| sym_e1_l0 | 0.02033 | 0.02033 | 0.01423 | 1.218 [1.007, 1.546] | 1.218 [0.990, 1.533] | 1.003 [0.996, 1.013] | 1.003 [0.997, 1.010] |
| sym_e1_l1 | 0.01999 | 0.01999 | 0.01413 | 1.209 [0.998, 1.536] | 1.209 [0.980, 1.523] | 0.996 [0.976, 1.014] | 0.996 [0.972, 1.015] |
| sym_e1_l2 | 0.02025 | 0.02025 | 0.01435 | 1.208 [1.001, 1.531] | 1.208 [0.982, 1.521] | 0.995 [0.982, 1.003] | 0.995 [0.981, 1.003] |
| sym_e1_l5 | 0.02014 | 0.02014 | 0.01431 | 1.233 [1.003, 1.578] | 1.233 [0.985, 1.550] | 1.015 [0.972, 1.085] | 1.015 [0.967, 1.080] |
| sym_e1_l10 | 0.02015 | 0.02015 | 0.01435 | 1.244 [1.013, 1.590] | 1.244 [0.984, 1.569] | 1.024 [0.975, 1.100] | 1.024 [0.966, 1.109] |
| selected | 0.02045 | 0.02045 | 0.01443 | 1.255 [1.026, 1.600] | 1.255 [1.000, 1.578] | 1.034 [0.990, 1.107] | 1.034 [0.985, 1.117] |
| I5-1 stored C0_refit / M_bias | 0.01906 / 0.02029 |  |  |  |  |  |  |

- Strict paper RRMS on the I5-1 frozen measured inclusion. Cluster bootstrap: clusters = class and W/L (12 clusters); reported, not deciding. sym_e1_l{lambda}: symmetric inputs, EPS 0.01 V, penalty lambda (V), VDD = class supply.


### I17-2 LV / HV split (reported, no rule)

| class | DUTs | C0_refit | M_bias v1 | selected | GMR(selected, C0_refit) | GMR(selected, v1) | GMR(v1, C0_refit) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LV (1.2 V) | 10 | 0.02137 | 0.02245 | 0.02278 | 1.275 [0.961, 1.879] 6/10 | 1.070 [0.998, 1.194] 4/10 | 1.192 [0.938, 1.712] 6/10 |
| HV (3.3 V) | 8 | 0.01617 | 0.01759 | 0.01754 | 1.232 [0.974, 1.636] 3/8 | 0.990 [0.964, 1.012] 3/8 | 1.244 [0.991, 1.643] 3/8 |

- Device bootstrap within class.


### I17-2 nested selection: inner score (mean ln RRMS of inner held-out refits) per outer fold and lambda, lambda* and selected row

| held-out DUT | training DUTs | lambda 0 V | lambda 1 V | lambda 2 V | lambda 5 V | lambda 10 V | lambda* (V) | selected RRMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| lv_nmos_W0p15_L0p18 | 9 | -4.2314 | -4.2661 | -4.2734 | -4.2367 | -4.2262 | 2 | 0.03190 |
| lv_nmos_W0p15_L10 | 9 | -3.9771 | -3.9809 | -3.9769 | -3.9886 | -3.9980 | 10 | 0.00750 |
| lv_nmos_W0p3_L0p13 | 9 | -4.3107 | -4.3162 | -4.3183 | -4.3093 | -4.3141 | 2 | 0.02094 |
| lv_nmos_W0p3_L10 | 9 | -4.0610 | -4.0592 | -4.0811 | -4.0801 | -4.0887 | 10 | 0.01187 |
| lv_nmos_W10_L10 | 9 | -4.1069 | -4.1055 | -4.0969 | -4.0913 | -4.1056 | 0 | 0.01423 |
| lv_pmos_W0p15_L0p18 | 9 | -4.1948 | -4.2675 | -4.2635 | -4.2534 | -4.2698 | 10 | 0.04760 |
| lv_pmos_W0p15_L10 | 9 | -3.7971 | -3.8015 | -3.7921 | -3.8068 | -3.8183 | 10 | 0.00280 |
| lv_pmos_W0p3_L0p13 | 9 | -4.5937 | -4.6055 | -4.5913 | -4.5701 | -4.5830 | 1 | 0.07063 |
| lv_pmos_W0p3_L10 | 9 | -3.8199 | -3.8918 | -3.8673 | -3.8590 | -3.8706 | 1 | 0.00264 |
| lv_pmos_W10_L10 | 9 | -3.9015 | -3.9492 | -3.9367 | -3.9548 | -3.9270 | 5 | 0.01767 |
| hv_nmos_W0p3_L0p5 | 7 | -4.1272 | -4.1138 | -4.1055 | -4.1275 | -4.1224 | 5 | 0.01856 |
| hv_nmos_W0p3_L10 | 7 | -3.9588 | -3.9280 | -3.9559 | -3.9348 | -3.9501 | 0 | 0.00580 |
| hv_nmos_W0p6_L1p2 | 7 | -3.9549 | -3.9559 | -4.0148 | -4.0131 | -4.0061 | 2 | 0.00892 |
| hv_nmos_W10_L0p5 | 7 | -4.3838 | -4.3925 | -4.3927 | -4.3683 | -4.3667 | 2 | 0.04510 |
| hv_pmos_W0p3_L10 | 7 | -4.1986 | -4.2025 | -4.2081 | -4.1959 | -4.2032 | 2 | 0.00924 |
| hv_pmos_W0p6_L0p5 | 7 | -4.3147 | -4.3191 | -4.3189 | -4.3089 | -4.3324 | 10 | 0.01462 |
| hv_pmos_W10_L0p35 | 7 | -4.3999 | -4.4675 | -4.4650 | -4.3944 | -4.4614 | 1 | 0.03040 |
| hv_pmos_W2_L1p2 | 7 | -4.0587 | -4.0530 | -4.1134 | -4.1125 | -4.0566 | 2 | 0.00768 |

- lambda* counts: 0 V: 2, 1 V: 3, 2 V: 6, 5 V: 2, 10 V: 5. Inner folds: training DUTs sorted by (polarity, tag), i mod 3; one member (seed 0); ties to the larger lambda.


### I17-2 negative corrected gds / gm on the penalty grids (% of points)

| arm | held-out gds < 0 | held-out gm < 0 | training gds < 0 | training gm < 0 |
|---|---:|---:|---:|---:|
| M_bias_v1 | 2.89 | 0.00 | 1.99 | 0.00 |
| sym_e1_l0 | 2.88 | 0.00 | 1.99 | 0.00 |
| sym_e1_l1 | 2.89 | 0.00 | 1.27 | 0.00 |
| sym_e1_l2 | 2.87 | 0.00 | 0.90 | 0.00 |
| sym_e1_l5 | 2.87 | 0.00 | 0.45 | 0.00 |
| sym_e1_l10 | 2.84 | 0.00 | 0.21 | 0.00 |
| selected | 2.81 | 0.00 | 0.88 | 0.00 |
| card only (starting cards) | 0.002 | 0.000 |  |  |

- Fold model on the held-out DUT's starting-card grid (held-out) and on its training grids, mean over folds; grid 75 x 75 forward quadrant to the class supply, forward current > 1 nA.

