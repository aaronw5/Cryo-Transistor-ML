# Tables, 2026-10-27 i23_1

Generated 2026-10-27 09:19 by `.venv/bin/python -I scripts/research/plot_explore_i23_1.py --date 2026-10-27`.

### I23-1 setup (pre-registered)

| item | value |
|---|---|
| pre-registration | out/research/explore/I23_1/prereg.md (commit a9aa227); code 73ce90e; DEV only, TEST locked; no amendment of a755e18 |
| Part A data | ihp233, ihp300, ihp343, ihp398; I5_1/start/records_<T>K (state C, PSP103 no clamp); ngspice-46 + psp103.osdi (I5-1 MeasProblem) |
| Part A call unit | MeasProblem.n_sim of the held-out refit (incl. start, Jacobian columns, IRLS re-simulations, final re-simulation) |
| Part A arms | C0_refit_rerun: max_nfev 400, no binding limit (VA1); C0_call: base box (12 active), max_nfev N_B1(d), call limit N_B1(d); B1: stored I22-1 rows; B1_cap: I22-1 B1 (matched_nfev from 400), call limit N_C0(d) (secondary) |
| Part A capped refit | polish_capped: polish with at most N-1 calls inside polish; at the limit the running TRF round stops, remaining IRLS rounds skipped, best card kept |
| Part A rule | survives: CI upper of GMR(B1, C0_call) < 1 at all 4 temperatures; not_shown: CI upper >= 1 at all 4 temperatures; partial: otherwise; invalid_temperature: rule not evaluated |
| Part B arm | fixed_l1: 5-member g v2 ensemble per outer fold, lambda = 1 V, selected recipe otherwise unchanged, same folds / training order / state as the selected arm; held-out refit max_nfev 400 irls 2 diff_step 0.01 |
| Part B rows reused | sky130: I15_2/lodo_sym_e1_l1; ihp300: I17_2/lodo_sym_e1_l1; tud4: I18_2/lodo_4K_sym_e1_l1; tud295: I18_2/lodo_295K_sym_e1_l1; nist4: I20_1/lodo_nist4, the 17 devices with lambda* = 1 V |
| Part B rows new | ihp233: 18 folds; ihp343: 18 folds; ihp398: 18 folds; nist4: 36 devices with lambda* = 0 V; nistrt: 87 devices |
| Part B rule | gain set sky130, nist4, nistrt; non_inferior: fixed_l1 non-inferior to selected on all 3 of G; not_shown: otherwise; invalid_G: not evaluated |


### Part A validity per temperature (VA1-VA5)

| temperature | VA1 max |C0_refit rerun via polish_capped - stored| (<= 1e-12) | VA2 non-finite rows | VA3 calls <= limit | VA4 DEV only | VA5 unit tests | result |
|---|---:|---:|---:|---:|---:|---|
| IHP SG13G2 233 K | 0.0e+00; n_sim equal True | 0 failed | C0_call over limit 0, B1_cap over limit 0 | True | True | valid |
| IHP SG13G2 300 K | 0.0e+00; n_sim equal True | 0 failed | C0_call over limit 0, B1_cap over limit 0 | True | True | valid |
| IHP SG13G2 343 K | 0.0e+00; n_sim equal True | 0 failed | C0_call over limit 0, B1_cap over limit 0 | True | True | valid |
| IHP SG13G2 398 K | 0.0e+00; n_sim equal True | 0 failed | C0_call over limit 0, B1_cap over limit 0 | True | True | valid |

- VA1: the I22-1 C0_refit refit through polish_capped with no limit against the stored rows.


### Part A held-out RRMS per arm (18 DUTs per temperature): all-device mean / combined (mean of the nMOS and pMOS family means)

| temperature | C0_refit (max_nfev 400) | C0_call (calls = B1) | B1 wider card set | B1_cap (calls = C0_refit) | selected g v2 |
|---|---:|---:|---:|---:|---:|
| IHP SG13G2 233 K | 0.02734 / 0.02734 | 0.02730 / 0.02730 | 0.02346 / 0.02346 | 0.02776 / 0.02776 | 0.02787 / 0.02787 |
| IHP SG13G2 300 K | 0.01906 / 0.01906 | 0.01903 / 0.01903 | 0.01602 / 0.01602 | 0.01892 / 0.01892 | 0.02045 / 0.02045 |
| IHP SG13G2 343 K | 0.01900 / 0.01900 | 0.01899 / 0.01899 | 0.01531 / 0.01531 | 0.01969 / 0.01969 | 0.01935 / 0.01935 |
| IHP SG13G2 398 K | 0.01973 / 0.01973 | 0.01971 / 0.01971 | 0.01835 / 0.01835 | 0.02093 / 0.02093 | 0.02017 / 0.02017 |

- C0_refit, B1: stored I22-1 rows; selected g v2: stored rows; C0_call, B1_cap: I23-1. Strict paper RRMS, frozen curve inclusion.


### Part A GMR with 95% bootstrap CI, DUTs lower / paired, label (X better: CI upper < 1; Y better: CI lower > 1)

| temperature | GMR(B1, C0_call) (primary) | GMR(C0_call, C0_refit) | GMR(B1, C0_refit) (I22-1) | GMR(B1_cap, C0_refit) | GMR(B1, B1_cap) | GMR(C0_call, selected g v2) |
|---|---:|---:|---:|---:|---:|---:|
| IHP SG13G2 233 K | 0.796 [0.661, 0.927] 15/18; gain | 0.998 [0.996, 0.999] 12/18; gain | 0.794 [0.659, 0.925] 16/18; gain | 0.977 [0.891, 1.035] 11/18; inconclusive | 0.813 [0.717, 0.910] 18/18; gain | 0.921 [0.728, 1.111] 7/18; inconclusive |
| IHP SG13G2 300 K | 0.776 [0.654, 0.905] 16/18; gain | 0.998 [0.997, 1.000] 16/18; gain | 0.775 [0.653, 0.903] 17/18; gain | 0.933 [0.868, 0.991] 14/18; gain | 0.830 [0.727, 0.933] 18/18; gain | 0.795 [0.625, 0.973] 9/18; gain |
| IHP SG13G2 343 K | 0.696 [0.580, 0.818] 17/18; gain | 0.999 [0.998, 1.000] 14/18; gain | 0.695 [0.579, 0.816] 17/18; gain | 0.944 [0.880, 1.011] 13/18; inconclusive | 0.736 [0.637, 0.841] 18/18; gain | 0.841 [0.664, 1.009] 8/18; inconclusive |
| IHP SG13G2 398 K | 0.811 [0.706, 0.916] 16/18; gain | 0.998 [0.996, 0.999] 13/18; gain | 0.809 [0.704, 0.914] 17/18; gain | 0.960 [0.899, 1.019] 13/18; inconclusive | 0.843 [0.758, 0.925] 18/18; gain | 0.792 [0.621, 0.974] 7/18; gain |
| pooled exp(mean ln GMR), 4 temperatures | 0.768 | 0.998 | 0.767 | 0.953 | 0.805 | 0.836 |

- Rule outcome: B1 gain survives the call match (temperatures with CI upper < 1: ihp233, ihp300, ihp343, ihp398).
- P1 True; P2 {'ihp233': True, 'ihp300': True, 'ihp343': True, 'ihp398': True}; P3 False.


### Part A NGSpice calls per DUT (held-out refit, incl. start, Jacobian columns, IRLS and final re-simulations)

| temperature | B1 refit calls (= C0_call limit) | B1 screen calls / DUT (stored I5-1, not in the limit) | C0_call calls used | C0_call mean fraction of limit | C0_call DUTs at the limit | C0_refit calls (= B1_cap limit) | B1_cap calls used | B1_cap DUTs at the limit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IHP SG13G2 233 K | 1938 | 173 | 1206 | 0.581 | 0/18 | 313 | 313 | 18/18 |
| IHP SG13G2 300 K | 1957 | 173 | 1436 | 0.664 | 1/18 | 324 | 324 | 18/18 |
| IHP SG13G2 343 K | 1906 | 173 | 1199 | 0.563 | 1/18 | 306 | 306 | 18/18 |
| IHP SG13G2 398 K | 1940 | 173 | 1341 | 0.631 | 1/18 | 321 | 321 | 18/18 |

- A DUT 'at the limit' stopped because the call limit was reached; otherwise TRF ended by its own criteria or its iteration budget (max_nfev) before the limit.


### Part B row sources and validity per dataset (VB1-VB5)

| dataset | fixed lambda 1 V rows from | devices reused / new | VB1 reuse sample reproduces (<= 1e-9) | VB2 new rows = selected rows on lambda* = 1 V folds | VB3 complete | VB4 DEV only | result |
|---|---|---:|---|---|---:|---:|---|
| sky130 77 K (18, LODO) | out/research/explore/I15_2/lodo_sym_e1_l1/results.json | 18 / 0 | True (0.0e+00; nmos_L1_W1p6, pmos_L2_W5) | - | True (0 failed) | True | valid |
| NIST sky130 4 K lvt (53) | out/research/explore/I20_1/lodo_nist4/results.json; out/research/explore/I23_1/fixl1_nist4/results.json | 17 / 36 | True (0.0e+00; nfet_01v8_lvt_W0p64_L0p18_m21p3, pfet_01v8_lvt_W0p42_L0p5_m27p2) | True (no overlap fold) | True (0 failed) | True | valid |
| NIST sky130 4 K RT card (87) | out/research/explore/I23_1/fixl1_nistrt/results.json | 0 / 87 | - | True (no overlap fold) | True (0 failed) | True | valid |
| IHP SG13G2 233 K | out/research/explore/I23_1/fixl1_ihp233/results.json | 0 / 18 | - | True (4 folds; max 0.0e+00) | True (0 failed) | True | valid |
| IHP SG13G2 300 K | out/research/explore/I17_2/lodo_sym_e1_l1/results.json | 18 / 0 | True (0.0e+00; lv_nmos_W0p15_L0p18, hv_pmos_W0p6_L0p5) | - | True (0 failed) | True | valid |
| IHP SG13G2 343 K | out/research/explore/I23_1/fixl1_ihp343/results.json | 0 / 18 | - | True (8 folds; max 0.0e+00) | True (0 failed) | True | valid |
| IHP SG13G2 398 K | out/research/explore/I23_1/fixl1_ihp398/results.json | 0 / 18 | - | True (2 folds; max 0.0e+00) | True (0 failed) | True | valid |
| TU Delft 4.13 K | out/research/explore/I18_2/lodo_4K_sym_e1_l1/results.json | 45 / 0 | True (0.0e+00; t40_nmos_W0p36_L0p04) | - | True (0 failed) | True | valid |
| TU Delft 295.0 K | out/research/explore/I18_2/lodo_295K_sym_e1_l1/results.json | 45 / 0 | True (0.0e+00; t40_nmos_W0p36_L0p04) | - | True (0 failed) | True | valid |

- VB1: the sampled folds were retrained (5 members, no cache) and refit; difference to the stored row. Unit tests (VB5) passed in every stage.


### Part B held-out RRMS per arm: all-device mean / combined (mean of the nMOS and pMOS family means)

| dataset | n (CI unit) | C0_refit | M_bias v1 | selected g v2 (nested lambda) | fixed lambda 1 V |
|---|---:|---:|---:|---:|---:|
| sky130 77 K (18, LODO) | 18 (device) | 0.1051 / 0.0984 | 0.0735 / 0.0695 | 0.0739 / 0.0700 | 0.0731 / 0.0692 |
| NIST sky130 4 K lvt (53) | 53 (device) | 0.0399 / 0.0427 | 0.0312 / 0.0334 | 0.0318 / 0.0340 | 0.0331 / 0.0354 |
| NIST sky130 4 K RT card (87) | 87 (device) | 0.0567 / 0.0593 | 0.0466 / 0.0491 | 0.0466 / 0.0491 | 0.0493 / 0.0520 |
| IHP SG13G2 233 K | 18 (device) | 0.0273 / 0.0273 | 0.0281 / 0.0281 | 0.0279 / 0.0279 | 0.0280 / 0.0280 |
| IHP SG13G2 300 K | 18 (device) | 0.0191 / 0.0191 | 0.0203 / 0.0203 | 0.0205 / 0.0205 | 0.0200 / 0.0200 |
| IHP SG13G2 343 K | 18 (device) | 0.0190 / 0.0190 | 0.0194 / 0.0194 | 0.0194 / 0.0194 | 0.0195 / 0.0195 |
| IHP SG13G2 398 K | 18 (device) | 0.0197 / 0.0197 | 0.0202 / 0.0202 | 0.0202 / 0.0202 | 0.0202 / 0.0202 |
| TU Delft 4.13 K | 45 (cluster (7)) | 0.0602 / 0.0532 | 0.0534 / 0.0491 | 0.0484 / 0.0449 | 0.0507 / 0.0472 |
| TU Delft 295.0 K | 45 (cluster (7)) | 0.0265 / 0.0251 | 0.0208 / 0.0210 | 0.0215 / 0.0218 | 0.0220 / 0.0222 |


### Part B GMR with 95% bootstrap CI (TU Delft: 7-group cluster), devices lower / paired, labels; NI: fixed lambda non-inferior to selected (CI upper < 1.05)

| dataset | GMR(fixed lambda 1 V, selected g v2) | GMR(fixed lambda 1 V, C0_refit) | GMR(selected g v2, C0_refit) | GMR(fixed lambda 1 V, M_bias v1) |
|---|---:|---:|---:|---:|
| sky130 77 K (18, LODO) (G) | 0.988 [0.962, 1.014] 6/18; inconclusive; NI | 0.734 [0.649, 0.841]; gain | 0.743 [0.654, 0.855]; gain | 0.985 [0.948, 1.019] |
| NIST sky130 4 K lvt (53) (G) | 1.018 [0.991, 1.044] 14/53; inconclusive; NI | 0.817 [0.782, 0.860]; gain | 0.803 [0.757, 0.857]; gain | 1.024 [0.990, 1.056] |
| NIST sky130 4 K RT card (87) (G) | 1.030 [1.002, 1.058] 23/87; selected better | 0.835 [0.797, 0.883]; gain | 0.811 [0.761, 0.871]; gain | 1.032 [1.000, 1.062] |
| IHP SG13G2 233 K | 1.012 [1.000, 1.027] 8/18; selected better; NI | 1.097 [0.905, 1.400]; inconclusive | 1.084 [0.898, 1.371]; inconclusive | 0.992 [0.971, 1.011] |
| IHP SG13G2 300 K | 0.963 [0.911, 1.000] 9/18; inconclusive; NI | 1.209 [0.998, 1.536]; inconclusive | 1.255 [1.026, 1.600]; worse | 0.996 [0.976, 1.014] |
| IHP SG13G2 343 K | 1.015 [1.004, 1.030] 1/18; selected better; NI | 1.206 [0.995, 1.541]; inconclusive | 1.188 [0.990, 1.503]; inconclusive | 1.021 [0.995, 1.053] |
| IHP SG13G2 398 K | 1.004 [0.992, 1.015] 8/18; inconclusive; NI | 1.265 [1.026, 1.617]; worse | 1.260 [1.026, 1.607]; worse | 0.994 [0.983, 1.004] |
| TU Delft 4.13 K | 1.070 [1.025, 1.137] 9/45; selected better | 0.884 [0.726, 1.077]; inconclusive | 0.826 [0.669, 1.014]; inconclusive | 0.972 [0.918, 1.018] |
| TU Delft 295.0 K | 1.030 [0.980, 1.094] 13/45; inconclusive | 0.854 [0.561, 1.237]; inconclusive | 0.829 [0.523, 1.247]; inconclusive | 1.053 [1.003, 1.107] |
| pooled exp(mean ln GMR), 9 datasets | 1.014 | 0.971 | 0.957 | - |

- Rule outcome: fixed lambda 1 V not shown non-inferior (failing: nistrt).
- Non-inferior (CI upper < 1.05) on: sky130, nist4, ihp233, ihp300, ihp343, ihp398.
- I22-1 pooled vs C0_refit: B1 0.913, B2 0.954, B3_d3 1.058.
- P4 False; P5 {'sky130': True, 'nist4': True, 'nistrt': True, 'ihp233': True, 'ihp300': True, 'ihp343': True, 'ihp398': True}; P6 {'sky130': True, 'nist4': True, 'nistrt': True, 'ihp233': True, 'ihp300': True, 'ihp343': True, 'ihp398': True, 'tud4': True, 'tud295': False}.


### Part B NGSpice calls per held-out device (fixed lambda 1 V: held-out refit only, no selection calls) next to the I22-1 cost table

| dataset | starting card calls / device (shared) | C0_refit | fixed lambda 1 V | fixed / C0_refit | selected g v2 arm total (I22-1) | selected / C0_refit | B1 arm total (I22-1) | fixed lambda trainings | fixed lambda training CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sky130 77 K (18, LODO) | 44,219 | 316 | 317 | 1.01 | 28,191 | 89 | 657 | 90 | 11,390 |
| NIST sky130 4 K lvt (53) | 1,569 | 238 | 260 | 1.09 | 6,683 | 28 | 641 | 265 | 97,091 |
| NIST sky130 4 K RT card (87) | 1,573 | 219 | 231 | 1.06 | 6,232 | 28 | 635 | 435 | 258,190 |
| IHP SG13G2 233 K | 1,389 | 313 | 316 | 1.01 | 14,258 | 46 | 1,938 | 90 | 6,572 |
| IHP SG13G2 300 K | 1,368 | 324 | 317 | 0.98 | 16,379 | 50 | 2,131 | 90 | 5,114 |
| IHP SG13G2 343 K | 1,053 | 306 | 330 | 1.08 | 16,127 | 53 | 1,906 | 90 | 7,117 |
| IHP SG13G2 398 K | 1,273 | 321 | 328 | 1.02 | 16,375 | 51 | 1,940 | 90 | 7,501 |
| TU Delft 4.13 K | 1,846 | 210 | 219 | 1.04 | 7,504 | 36 | 517 | 35 | 5,621 |
| TU Delft 295.0 K | 2,128 | 208 | 242 | 1.16 | 8,652 | 42 | 576 | 35 | 7,026 |

- Selected g v2 totals include the nested lambda selection inner refits (I22-1 Part B; estimated where not stored). Penalty grids (one DC deck per device) are shared by both g v2 arms and not converted to calls.

