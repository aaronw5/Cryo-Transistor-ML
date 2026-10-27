# g_why: where the learned correction g lowers the RRMS, and why not on IHP

Date 2026-10-27. Script `scripts/research/explore/g_why.py` (stored records and stored corrections only; no
NGSpice run, no training; CPU, single thread, about 6 s). Numbers below are copied from `results.json` in this
directory. Interpretations are labeled **Interpretation** and cite the number they rest on.

## Inputs

| item | source |
|---|---|
| sky130 77 K records | `I13_1/records/real_dev` + `real_test` (18 devices, state C, starting cards direct43_r3_fd; the training data of g) |
| sky130 full-data g | `I15_2/lodo_sym_e1_l1/corrections.json` key `full\|sym_e1_l1` (lambda 1 V; the model of I16-1 / I17-1) |
| sky130 LODO g | per fold at lambda*_k of `I15_2/decision.json` (lambda 10 V = `I13_1/lodo` `M_bias_v2`) |
| sky130 refit cards (M2) | `I17_1/deploy_sel/arrays/q12_<tag>.npz` `grid_fwd_default_card` (25 mV grid), interpolated to the measured biases |
| NIST 4 K lvt | `I20_1/start/records_nist4` (53 DEV devices with a stored LODO g); full-data g `I20_1/select_nist4/full_corrections.json` (`full\|sym_e1_l0`); LODO g `I20_1/lodo_nist4` `selected\|<tag>` |
| IHP SG13G2 300 K | `I5_1/start/records_300K` (18 devices); LODO g `I17_2/lodo_sym_e1_l<lambda*_k>`; no full-data g is stored |

Definitions.
- Residual r = ln(|I_meas| / |I_card|) at points with |I_meas| > floor (sky130 1e-8 A, NIST and IHP 1e-10 A) and
  |Vds| >= 5 mV. I_card is the starting card unless marked "refit M2".
- Metric weight w = |I_meas| / curve denominator on the trimmed points of included curves (0 elsewhere). To first
  order (I_card e^g - I_meas) / den = w (g - r), so w^2-weighted statistics of r measure what a bias-only g can
  remove from the paper RRMS (same expansion as `shared_discrepancy.py`).
- Regions per device, from the card Vth (max-gm extrapolation on the low-|Vds| Id-Vg of the card): Vov = |Vgs| - Vth_card;
  sub: Vov < -0.1 V; near: |Vov| <= 0.1 V; SI_lin: Vov > 0.1 V and |Vds| < Vov; SI_sat: Vov > 0.1 V and |Vds| >= Vov.
  Card Vth (sky130): nMOS 0.518-0.792 V (median 0.690), pMOS 1.029-1.239 V (median 1.120).
- Curve classes (sky130): Id-Vg low |Vds| = idvg@0.01; Id-Vg high |Vds| = idvg@0.37 ... @1.85; Id-Vd = idvd@*.
- RRMS reduction split: per included curve, r0 - r1 = sum over regions of dS_R / (n den (sqrt(S0) + sqrt(S1))),
  S = mean squared error; exact (the split reproduces each device's RRMS change to 1e-9, asserted in the script).
- Shared-pattern statistics: per family, residual averaged in bins of 0.04 VDD x 0.04 VDD over (|Vgs|/VDD, |Vds|/VDD)
  ("raw grid", the inputs g sees) or (Vov/VDD, |Vds|/VDD) ("Vov grid"); bins with >= 3 devices. Shared fraction =
  1 - SS(r - m) / SS(r), m = w^2-weighted mean over devices per bin, in-sample or leave-one-device-out (LOO);
  "removed by g" = 1 - SS(r - g) / SS(r). Pairwise correlation = Pearson r of two devices' bin values.

Check of the refit-card interpolation: interpolated M2 card x g gives all-device RRMS 0.0705 against the stored
0.0696 (max per-device difference 0.0030); M2 card alone 0.1092 against the stored 0.1083.

## 1. g over the bias plane (sky130 full-data model)

Figure `01_g_full_sky130_maps.png` (heat maps, cuts vs |Vgs| and vs |Vds|; Vth_card median dashed, range dotted,
near-threshold band shaded, |Vds| = Vov line dash-dot).

- nMOS: |g| <= 0.12 everywhere on 0..1.85 V; g = -0.12 at subthreshold, low |Vds|; +0.02 to +0.04 in strong inversion.
- pMOS: g = -0.21 to -0.37 along |Vds| = 0.01 V (all |Vgs|), rising to 0 at |Vds| ~ 0.1 VDD, positive +0.06 to +0.20
  at |Vds| = 0.2-0.4 V for |Vgs| >= 1.48 V (linear region, just below the |Vds| = Vov line), 0 in saturation
  (|g| < 0.01 for |Vds| > 0.9 V).

## 2. Where the residual is and where the RRMS gain comes from (sky130)

RRMS without refit (g applied to the starting records; all-device / combined):

| dataset | n | start card | start x g_full (no refit) | start x g_LODO (no refit) | GMR LODO/start, lower/n |
|---|---:|---:|---:|---:|---:|
| sky130 77 K | 18 | 0.1109 / 0.1049 (n 0.0510, p 0.1588) | 0.0782 / 0.0748 (n 0.0443, p 0.1052) | 0.0836 / 0.0801 (n 0.0486, p 0.1116) | 0.801, 14/18 |
| NIST 4 K lvt (DEV) | 53 | 0.0418 / 0.0446 (n 0.0145, p 0.0748) | 0.0341 / 0.0364 (n 0.0121, p 0.0606) | 0.0354 / 0.0378 (n 0.0124, p 0.0632) | 0.856, 49/53 |
| IHP 300 K | 18 | 0.0218 / 0.0218 (n 0.0187, p 0.0248) | no full-data model stored | 0.0237 / 0.0237 (n 0.0197, p 0.0276) | 1.404, 9/18 |

Reference rows (stored): C0_refit LODO 0.1051; selected g v2 LODO with refit 0.0739; full-data deployment M2
(in-sample refit with g in the loop) 0.0696, its card alone 0.1083.

Split of the all-device RRMS reduction 0.1109 -> 0.0782 (total 0.0327) by family, curve class and region (shares of
the total; sum = 1). Figure `03_rrms_reduction_split_sky130.png`.

| family | curve class | sub | near | SI_lin | SI_sat | row total |
|---|---|---:|---:|---:|---:|---:|
| nmos | IdVg_lowVds | -0.000 | +0.000 | +0.021 | +0.000 | +0.022 |
| nmos | IdVg_highVds | -0.000 | +0.000 | +0.033 | +0.019 | +0.052 |
| nmos | IdVd | +0.000 | +0.001 | +0.011 | +0.005 | +0.017 |
| pmos | IdVg_lowVds | +0.000 | +0.002 | +0.617 | +0.000 | +0.619 |
| pmos | IdVg_highVds | +0.000 | +0.000 | +0.223 | +0.009 | +0.232 |
| pmos | IdVd | -0.001 | +0.011 | +0.043 | +0.006 | +0.059 |

Largest curves: tags: pmos|idvg@0.01 0.619, pmos|idvg@0.37 0.227, pmos|idvd@1.85 0.047, nmos|idvg@0.01 0.022, nmos|idvg@0.74 0.018, nmos|idvg@0.37 0.015

- pMOS carries 0.910 of the reduction; nMOS 0.090. pMOS idvg@0.01 alone carries 0.619, pMOS idvg@0.37 0.227.
- Subthreshold carries |share| <= 0.001 in every class; near-threshold 0.014 in total; SI_lin 0.948; SI_sat 0.039.

Log residual by region (per device mean, then mean / SD over devices):


sky130
| family | region | devices | points | mean ln residual (mean / SD over devices) | after g | RMS ln residual (mean over devices) | metric-weighted mean (mean / SD) |
|---|---|---:|---:|---:|---:|---:|---:|
| nmos | sub | 7 | 425 | +0.792 / 2.068 | +0.802 / 2.067 | 1.799 | +1.236 / 1.914 |
| nmos | near | 8 | 1767 | -0.072 / 0.329 | -0.063 / 0.329 | 0.546 | +0.125 / 0.336 |
| nmos | SI_lin | 8 | 3930 | +0.008 / 0.016 | -0.001 / 0.016 | 0.080 | +0.021 / 0.018 |
| nmos | SI_sat | 8 | 6279 | +0.001 / 0.025 | -0.009 / 0.024 | 0.049 | +0.017 / 0.019 |
| pmos | sub | 10 | 2487 | +0.666 / 1.444 | +0.681 / 1.452 | 1.039 | +0.173 / 0.351 |
| pmos | near | 10 | 2445 | -0.036 / 0.140 | -0.010 / 0.142 | 0.297 | +0.033 / 0.084 |
| pmos | SI_lin | 10 | 2049 | +0.014 / 0.094 | +0.074 / 0.098 | 0.339 | -0.009 / 0.084 |
| pmos | SI_sat | 10 | 5382 | +0.030 / 0.037 | +0.025 / 0.037 | 0.073 | +0.015 / 0.020 |

nist4
| family | region | devices | points | mean ln residual (mean / SD over devices) | after g | RMS ln residual (mean over devices) | metric-weighted mean (mean / SD) |
|---|---|---:|---:|---:|---:|---:|---:|
| nmos | sub | 15 | 606 | -5.471 / 2.339 | -5.519 / 2.335 | 5.945 | -3.270 / 3.269 |
| nmos | near | 29 | 2174 | -1.742 / 1.851 | -1.764 / 1.855 | 2.332 | +0.007 / 0.058 |
| nmos | SI_lin | 29 | 6454 | -0.000 / 0.004 | +0.000 / 0.004 | 0.021 | +0.003 / 0.005 |
| nmos | SI_sat | 29 | 9532 | +0.004 / 0.005 | +0.003 / 0.005 | 0.022 | +0.002 / 0.007 |
| pmos | sub | 11 | 1066 | -1.276 / 4.266 | -1.256 / 4.266 | 4.147 | -1.699 / 3.738 |
| pmos | near | 24 | 2439 | -3.371 / 1.819 | -3.323 / 1.826 | 3.903 | -0.323 / 0.303 |
| pmos | SI_lin | 24 | 2183 | -0.044 / 0.078 | -0.027 / 0.075 | 0.289 | -0.006 / 0.025 |
| pmos | SI_sat | 24 | 4647 | +0.035 / 0.044 | +0.011 / 0.043 | 0.115 | +0.024 / 0.018 |

ihp300
| family | region | devices | points | mean ln residual (mean / SD over devices) | after g | RMS ln residual (mean over devices) | metric-weighted mean (mean / SD) |
|---|---|---:|---:|---:|---:|---:|---:|
| nmos | sub | 8 | 429 | +0.405 / 0.701 | +0.406 / 0.703 | 0.584 | +0.223 / 0.518 |
| nmos | near | 9 | 666 | -0.023 / 0.279 | -0.023 / 0.281 | 0.235 | -0.048 / 0.100 |
| nmos | SI_lin | 9 | 1000 | +0.000 / 0.005 | -0.000 / 0.006 | 0.020 | +0.003 / 0.004 |
| nmos | SI_sat | 9 | 1217 | +0.002 / 0.006 | +0.002 / 0.006 | 0.021 | -0.001 / 0.002 |
| pmos | sub | 8 | 274 | -0.126 / 0.364 | -0.128 / 0.365 | 0.297 | -0.040 / 0.250 |
| pmos | near | 9 | 567 | -0.017 / 0.097 | -0.021 / 0.097 | 0.093 | -0.002 / 0.081 |
| pmos | SI_lin | 9 | 1027 | -0.008 / 0.011 | -0.002 / 0.013 | 0.027 | -0.004 / 0.011 |
| pmos | SI_sat | 9 | 1424 | +0.017 / 0.039 | +0.011 / 0.041 | 0.039 | +0.011 / 0.022 |

- The largest log residuals are in subthreshold (sky130 nMOS mean +0.79, SD 2.07 over devices; NIST nMOS -5.47);
  g changes them by <= 0.05. The metric weight there is near 0, so the RRMS does not see them.
- In strong inversion the mean residual is small (|mean| <= 0.03) but has shape along the curves: see section 3.

Consistency of the residual across devices (figures `05_shared_pattern_by_dataset.png`,
`06_device_residual_maps_sky130.png`, `09_device_residual_maps_ihp300k.png`):

| dataset | family | devices | bins | RMS weighted residual | shared fraction in-sample | shared fraction LOO | removed by g (g used) | mean pair corr | LOO shared, Vov grid | mean pair corr, Vov grid | unweighted LOO shared | unweighted pair corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sky130 77 K | nmos | 8 | 189 | 0.052 | 0.438 | 0.242 | 0.292 (full); LODO 0.178 | 0.206 | -2.080 | 0.008 | -0.170 | 0.196 |
| sky130 77 K | pmos | 10 | 157 | 0.153 | 0.652 | 0.447 | 0.555 (full); LODO 0.510 | 0.479 | 0.469 | 0.383 | -0.253 | 0.469 |
| NIST 4 K | nmos | 29 | 128 | 0.016 | 0.483 | 0.445 | 0.291 (full); LODO 0.282 | 0.736 | 0.556 | 0.812 | 0.807 | 0.937 |
| NIST 4 K | pmos | 24 | 104 | 0.084 | 0.602 | 0.555 | 0.410 (full); LODO 0.337 | 0.736 | 0.709 | 0.707 | 0.514 | 0.716 |
| IHP 300 K | nmos | 9 | 293 | 0.044 | 0.393 | -0.409 | -0.020 (LODO) | -0.064 | 0.093 | 0.083 | -0.004 | -0.030 |
| IHP 300 K | pmos | 9 | 291 | 0.033 | 0.462 | -0.049 | 0.024 (LODO) | -0.028 | 0.081 | 0.004 | -0.207 | -0.050 |

IHP by class (weighted raw grid; LOO shared fraction, mean pair correlation, devices): IHP by class: {'hv_nmos': (-0.655, 0.62, 4), 'hv_pmos': (0.039, 0.191, 4), 'lv_nmos': (-0.713, -0.047, 5), 'lv_pmos': (-0.08, -0.119, 5)}

- sky130 pMOS: 0.447 of the metric-weighted residual sum of squares is predicted by the mean of the other pMOS
  devices at the same bias; full-data g removes 0.555 (in-sample), LODO g 0.510. nMOS: 0.242 LOO, g removes 0.292
  (full) / 0.178 (LODO).
- The raw grid gives a higher LOO shared fraction than the Vov grid for sky130 nMOS (0.242 vs -2.080) and the same
  for pMOS (0.447 vs 0.469). **Interpretation:** the shared component is tied to absolute bias, not to the
  threshold-relative bias; this matches g having no Vth input.

## 3. Which card mismatch g removes (sky130)

Residual on the low-|Vds| Id-Vg at |Vgs|/VDD = 0.8 / 0.9 / 1.0 (mean ± SD over devices; g at the same points; count of
devices with r < 0):

| dataset | family | 0.8 VDD | 0.9 VDD | 1.0 VDD |
|---|---|---:|---:|---:|
| sky130 | nmos | +0.024±0.052 (g +0.035, neg 4/8) | +0.056±0.076 (g +0.036, neg 1/8) | +0.060±0.086 (g +0.037, neg 3/8) |
| sky130 | pmos | -0.370±0.186 (g -0.281, neg 10/10) | -0.382±0.248 (g -0.333, neg 10/10) | -0.411±0.347 (g -0.374, neg 10/10) |
| sky130 refit M2 | nmos | +0.066±0.042 (g +0.035, neg 0/8) | +0.092±0.066 (g +0.036, neg 0/8) | +0.089±0.089 (g +0.037, neg 1/8) |
| sky130 refit M2 | pmos | -0.393±0.189 (g -0.281, neg 10/10) | -0.409±0.270 (g -0.333, neg 10/10) | -0.441±0.344 (g -0.374, neg 10/10) |
| nist4 | nmos | +0.009±0.010 (g +0.008, neg 4/29) | +0.005±0.008 (g +0.003, neg 4/29) | -0.007±0.011 (g -0.014, neg 23/29) |
| nist4 | pmos | -0.081±0.095 (g -0.087, neg 21/24) | -0.044±0.054 (g -0.038, neg 22/24) | -0.027±0.025 (g -0.000, neg 21/24) |
| ihp300 | nmos | -0.013±0.019 (g -0.004, neg 7/9) | -0.011±0.014 (g -0.005, neg 8/9) | -0.007±0.011 (g -0.005, neg 6/9) |
| ihp300 | pmos | -0.002±0.014 (g -0.008, neg 3/9) | -0.001±0.023 (g -0.006, neg 4/9) | -0.002±0.030 (g -0.005, neg 6/9) |

Residual on the top-|Vgs| Id-Vd at |Vds|/VDD (mean ± SD over devices; g at the same points):

| dataset | family | 0.03 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | 0.75 | 1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sky130 | nmos | -0.11±0.05 (g +0.02) | -0.12±0.06 (g +0.01) | -0.11±0.04 (g -0.02) | -0.08±0.05 (g -0.05) | -0.05±0.04 (g -0.05) | -0.00±0.03 (g -0.02) | +0.03±0.03 (g +0.04) | +0.02±0.03 (g +0.02) | +0.02±0.03 (g +0.02) |
| sky130 | pmos | +0.10±0.17 (g -0.29) | +0.17±0.14 (g -0.20) | +0.23±0.11 (g +0.01) | +0.25±0.09 (g +0.16) | +0.24±0.09 (g +0.20) | +0.18±0.10 (g +0.11) | +0.06±0.09 (g +0.01) | +0.01±0.02 (g +0.00) | -0.01±0.04 (g -0.00) |
| sky130 refit M2 | nmos | -0.10±0.05 (g +0.02) | -0.11±0.06 (g +0.01) | -0.11±0.03 (g -0.02) | -0.08±0.03 (g -0.05) | -0.06±0.03 (g -0.05) | -0.01±0.02 (g -0.02) | +0.02±0.01 (g +0.04) | +0.01±0.01 (g +0.02) | +0.01±0.01 (g +0.02) |
| sky130 refit M2 | pmos | +0.06±0.15 (g -0.29) | +0.14±0.13 (g -0.20) | +0.21±0.08 (g +0.01) | +0.22±0.06 (g +0.16) | +0.21±0.06 (g +0.20) | +0.16±0.09 (g +0.11) | +0.05±0.08 (g +0.01) | +0.01±0.02 (g +0.00) | -0.01±0.04 (g -0.00) |
| nist4 | nmos | +0.01±0.03 (g -0.02) | -0.00±0.01 (g -0.01) | -0.02±0.02 (g -0.01) | -0.03±0.02 (g -0.01) | -0.03±0.02 (g -0.01) | -0.01±0.01 (g -0.01) | +0.01±0.01 (g +0.00) | +0.00±0.01 (g -0.00) | -0.00±0.01 (g -0.00) |
| nist4 | pmos | -0.28±0.11 (g -0.02) | -0.07±0.03 (g -0.01) | +0.13±0.07 (g +0.04) | +0.20±0.10 (g +0.07) | +0.22±0.11 (g +0.10) | +0.21±0.11 (g +0.12) | +0.10±0.08 (g +0.07) | +0.03±0.04 (g +0.01) | -0.01±0.01 (g -0.01) |
| ihp300 | nmos | n/a | -0.00±0.01 (g -0.01) | -0.00±0.01 (g -0.00) | +0.00±0.01 (g -0.00) | +0.00±0.01 (g +0.00) | +0.01±0.02 (g +0.01) | +0.01±0.01 (g +0.01) | +0.00±0.01 (g +0.00) | -0.01±0.01 (g -0.00) |
| ihp300 | pmos | n/a | -0.01±0.04 (g -0.00) | -0.01±0.03 (g -0.00) | -0.01±0.03 (g -0.00) | -0.01±0.03 (g +0.00) | -0.01±0.02 (g +0.00) | -0.01±0.02 (g +0.00) | -0.01±0.02 (g -0.00) | -0.02±0.02 (g -0.00) |

Device metrics, measured vs card vs card x exp(g) (figure `04_device_metrics_sky130.png`; SS = minimum swing over a
4-point window with 3 x floor <= |I| <= 0.3 I(Vth); kT/q ln10 = 15.3 mV/dec at 77 K, 0.79 at 4 K, 59.5 at 300 K;
gm by a Savitzky-Golay filter of about 0.09 V because the measured sky130 curves carry 5-10 % point noise in strong
inversion; the ELR Vth of the measured sky130 curves is noise-limited, the constant-current Vth at
max(1e-7 W/L, 30 x floor) A is the robust one; G = I/Vds on the top-|Vgs| Id-Vd):


sky130
| family | metric | n | measured (median) | card (median) | card x g (median) | card - meas: mean / MAE | card x g - meas: mean / MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| nmos | SS low Vds (mV/dec) | 6 | 57.5 | 84.8 | 84.5 | +79.9 / 79.9 | +78.3 / 78.3 |
| nmos | SS high Vds (mV/dec) | 4 | 45.1 | 80 | 80 | +62.3 / 62.3 | +62.1 / 62.1 |
| nmos | SS low Vds, no floor (mV/dec) | 7 | 36.3 | 87 | 86.5 | +101 / 101 | +99.1 / 99.1 |
| nmos | Vth ELR (V) | 8 | 0.745 | 0.69 | 0.691 | -0.156 / 0.162 | -0.155 / 0.16 |
| nmos | Vth constant current (V) | 8 | 0.679 | 0.68 | 0.679 | -0.00658 / 0.0544 | -0.00899 / 0.0515 |
| nmos | Vov at gm peak (V) | 8 | 0.295 | 0.205 | 0.209 | -0.0877 / 0.107 | -0.0855 / 0.108 |
| nmos | gm(VDD)/gm_max | 8 | 0.181 | 0.0786 | 0.0795 | -0.24 / 0.311 | -0.24 / 0.312 |
| nmos | I(VDD, low Vds) (ln ratio to meas) | 8 | 2.74e-05 | 2.28e-05 | 2.37e-05 | -0.0599 / 0.0723 | -0.0232 / 0.0631 |
| nmos | I(VDD, VDD) (ln ratio to meas) | 8 | 0.000941 | 0.000923 | 0.000944 | -0.0207 / 0.0269 | +0.00105 / 0.0229 |
| nmos | DIBL (mV/V) | 7 | 34 | 67.8 | 67.4 | +30.6 / 36.2 | +28.9 / 37.1 |
| nmos | gds/I sat, top Vgs (1/V) | 8 | 0.038 | 0.061 | 0.058 | +0.00805 / 0.0208 | +0.00502 / 0.0192 |
| nmos | gds/I sat, mid Vgs (1/V) | 8 | 0.0937 | 0.0878 | 0.0871 | +0.0257 / 0.0356 | +0.025 / 0.0358 |
| nmos | gds(0.85-1 VDD)/gds(0.55-0.7 VDD) | 8 | 0.592 | 0.54 | 0.981 | -0.176 / 0.417 | +0.247 / 0.726 |
| nmos | G(0.15VDD)/G(0.03VDD) | 8 | 0.749 | 0.781 | 0.731 | -0.0199 / 0.0369 | -0.0694 / 0.0694 |
| pmos | SS low Vds (mV/dec) | 2 | 57.3 | 180 | 182 | +122 / 122 | +125 / 125 |
| pmos | SS high Vds (mV/dec) | 8 | 120 | 149 | 149 | +24.9 / 27.3 | +24.9 / 27.3 |
| pmos | SS low Vds, no floor (mV/dec) | 10 | 51.6 | 133 | 133 | +78.6 / 81.2 | +78.4 / 81.1 |
| pmos | Vth ELR (V) | 10 | 1.45 | 1.12 | 1.11 | -0.325 / 0.325 | -0.343 / 0.343 |
| pmos | Vth constant current (V) | 7 | 1.24 | 1.21 | 1.23 | -0.152 / 0.155 | -0.0898 / 0.098 |
| pmos | Vov at gm peak (V) | 10 | 0.203 | 0.244 | 0.231 | +0.0511 / 0.122 | +0.0436 / 0.118 |
| pmos | gm(VDD)/gm_max | 10 | 0.408 | 0.63 | 0.473 | +0.276 / 0.424 | +0.155 / 0.427 |
| pmos | I(VDD, low Vds) (ln ratio to meas) | 10 | 5.45e-07 | 7.39e-07 | 5.08e-07 | +0.411 / 0.411 | +0.0368 / 0.252 |
| pmos | I(VDD, VDD) (ln ratio to meas) | 10 | 5.84e-05 | 6.15e-05 | 6.15e-05 | +0.00894 / 0.0278 | +0.00856 / 0.0277 |
| pmos | DIBL (mV/V) | 7 | 195 | 103 | 140 | -85.6 / 85.6 | -51.4 / 54.2 |
| pmos | gds/I sat, top Vgs (1/V) | 10 | 0.0798 | 0.0408 | 0.0381 | +0.0531 / 0.0721 | +0.0504 / 0.0715 |
| pmos | gds/I sat, mid Vgs (1/V) | 10 | 0.176 | 0.0789 | 0.0714 | +0.0266 / 0.108 | +0.0192 / 0.105 |
| pmos | gds(0.85-1 VDD)/gds(0.55-0.7 VDD) | 8 | 0.484 | 0.35 | 0.405 | -0.188 / 0.338 | -0.168 / 0.335 |
| pmos | G(0.15VDD)/G(0.03VDD) | 10 | 1.11 | 0.971 | 1.51 | -0.166 / 0.168 | +0.37 / 0.37 |

nist4
| family | metric | n | measured (median) | card (median) | card x g (median) | card - meas: mean / MAE | card x g - meas: mean / MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| nmos | SS low Vds (mV/dec) | 28 | 17.7 | 78.5 | 78.8 | +62.3 / 62.3 | +62.5 / 62.5 |
| nmos | SS high Vds (mV/dec) | 29 | 16.7 | 77.3 | 77.8 | +60.4 / 60.4 | +60.8 / 60.8 |
| nmos | SS low Vds, no floor (mV/dec) | 29 | 8.44 | 76.5 | 76.8 | +66.8 / 66.8 | +67.1 / 67.1 |
| nmos | Vth ELR (V) | 29 | 0.701 | 0.706 | 0.708 | +0.00183 / 0.00368 | +0.00297 / 0.00434 |
| nmos | Vth constant current (V) | 29 | 0.675 | 0.629 | 0.63 | -0.0489 / 0.0489 | -0.0484 / 0.0485 |
| nmos | Vov at gm peak (V) | 29 | 0.225 | 0.219 | 0.227 | -0.00735 / 0.0171 | -0.00366 / 0.0169 |
| nmos | gm(VDD)/gm_max | 29 | 0.244 | 0.253 | 0.181 | +0.0498 / 0.0524 | -0.0245 / 0.0362 |
| nmos | I(VDD, low Vds) (ln ratio to meas) | 29 | 0.00011 | 0.000111 | 0.000109 | +0.00723 / 0.0103 | -0.00685 / 0.0108 |
| nmos | I(VDD, VDD) (ln ratio to meas) | 29 | 0.000442 | 0.000444 | 0.000443 | +0.00129 / 0.00598 | -0.00244 / 0.00447 |
| nmos | DIBL (mV/V) | 29 | 16.9 | 38.8 | 41 | +16.7 / 17.9 | +19.3 / 20.3 |
| nmos | gds/I sat, top Vgs (1/V) | 29 | 0.0347 | 0.0445 | 0.0362 | +0.00206 / 0.00701 | -0.00627 / 0.00774 |
| nmos | gds/I sat, mid Vgs (1/V) | 29 | 0.0775 | 0.09 | 0.0967 | +7.11e-05 / 0.0277 | +0.00685 / 0.0289 |
| nmos | gds(0.85-1 VDD)/gds(0.55-0.7 VDD) | 29 | 0.57 | 0.756 | 0.825 | +0.0306 / 0.142 | +0.0875 / 0.169 |
| nmos | G(0.15VDD)/G(0.03VDD) | 29 | 0.742 | 0.814 | 0.818 | +0.0284 / 0.0387 | +0.033 / 0.0413 |
| pmos | SS low Vds (mV/dec) | 14 | 24.7 | 95.9 | 94.7 | +65.6 / 73 | +64.5 / 71.9 |
| pmos | SS high Vds (mV/dec) | 19 | 27.5 | 99.7 | 99.8 | +59 / 63.8 | +59.5 / 64.2 |
| pmos | SS low Vds, no floor (mV/dec) | 24 | 17.3 | 107 | 105 | +87.1 / 87.1 | +85.8 / 85.8 |
| pmos | Vth ELR (V) | 24 | 1.24 | 1.24 | 1.24 | -0.0171 / 0.022 | -0.00352 / 0.0214 |
| pmos | Vth constant current (V) | 24 | 1.26 | 1.18 | 1.19 | -0.101 / 0.102 | -0.0924 / 0.093 |
| pmos | Vov at gm peak (V) | 24 | 0.191 | 0.251 | 0.27 | +0.0241 / 0.0592 | +0.0448 / 0.0748 |
| pmos | gm(VDD)/gm_max | 24 | 0.739 | 0.733 | 0.828 | -0.0144 / 0.112 | +0.0728 / 0.14 |
| pmos | I(VDD, low Vds) (ln ratio to meas) | 24 | 1.3e-05 | 1.3e-05 | 1.3e-05 | +0.0272 / 0.0289 | +0.0267 / 0.0285 |
| pmos | I(VDD, VDD) (ln ratio to meas) | 24 | 6.49e-05 | 6.44e-05 | 6.39e-05 | +0.0077 / 0.0107 | -0.00149 / 0.00965 |
| pmos | DIBL (mV/V) | 24 | 9.12 | 22.5 | 27.3 | -3.92 / 20.5 | +1.06 / 22.7 |
| pmos | gds/I sat, top Vgs (1/V) | 24 | 0.0373 | 0.109 | 0.0511 | +0.0931 / 0.0937 | +0.0354 / 0.0742 |
| pmos | gds/I sat, mid Vgs (1/V) | 24 | 0.0783 | 0.0773 | 0.0265 | +0.0585 / 0.101 | +0.00814 / 0.0983 |
| pmos | gds(0.85-1 VDD)/gds(0.55-0.7 VDD) | 18 | 0.374 | 0.408 | 0.717 | +0.0576 / 0.129 | +0.756 / 0.819 |
| pmos | G(0.15VDD)/G(0.03VDD) | 24 | 1.4 | 0.893 | 0.982 | -0.553 / 0.553 | -0.464 / 0.464 |

ihp300
| family | metric | n | measured (median) | card (median) | card x g (median) | card - meas: mean / MAE | card x g - meas: mean / MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| nmos | SS low Vds (mV/dec) | 5 | 83.4 | 78.2 | 78.1 | -3.25 / 3.25 | -3.26 / 3.26 |
| nmos | SS high Vds (mV/dec) | 6 | 81 | 78 | 78 | -3.71 / 3.71 | -3.72 / 3.72 |
| nmos | SS low Vds, no floor (mV/dec) | 9 | 73.6 | 70.9 | 70.9 | -4.61 / 8.43 | -4.62 / 8.45 |
| nmos | Vth ELR (V) | 9 | 0.546 | 0.536 | 0.536 | -0.00211 / 0.00531 | -0.0038 / 0.00582 |
| nmos | Vth constant current (V) | 9 | 0.471 | 0.478 | 0.478 | +0.00301 / 0.0156 | +0.00292 / 0.0155 |
| nmos | Vov at gm peak (V) | 9 | 0.352 | 0.352 | 0.353 | +0.0188 / 0.0374 | +0.0149 / 0.041 |
| nmos | gm(VDD)/gm_max | 9 | 0.45 | 0.499 | 0.469 | -0.0193 / 0.0568 | -0.0262 / 0.0599 |
| nmos | I(VDD, low Vds) (ln ratio to meas) | 9 | 1.56e-05 | 1.52e-05 | 1.53e-05 | +0.00345 / 0.00921 | -0.00303 / 0.00893 |
| nmos | I(VDD, VDD) (ln ratio to meas) | 9 | 0.000149 | 0.000151 | 0.000151 | +0.00602 / 0.00702 | +0.00194 / 0.00783 |
| nmos | DIBL (mV/V) | 9 | 17.2 | 14.4 | 14.2 | -1.69 / 2.76 | -1.61 / 2.67 |
| nmos | gds/I sat, top Vgs (1/V) | 9 | 0.113 | 0.135 | 0.114 | +0.0217 / 0.0217 | +0.00784 / 0.0203 |
| nmos | gds/I sat, mid Vgs (1/V) | 9 | 0.0464 | 0.0386 | 0.0361 | +0.00549 / 0.0196 | +0.00498 / 0.0216 |
| nmos | gds(0.85-1 VDD)/gds(0.55-0.7 VDD) | 9 | 0.167 | 0.182 | 0.149 | +0.0183 / 0.0317 | -0.00184 / 0.0381 |
| nmos | G(0.15VDD)/G(0.03VDD) | 9 | 0.924 | 0.932 | 0.933 | -0.00441 / 0.00776 | +0.00163 / 0.00825 |
| pmos | SS low Vds (mV/dec) | 3 | 92.7 | 90.7 | 90.8 | +1.33 / 6.64 | +1.36 / 6.52 |
| pmos | SS high Vds (mV/dec) | 5 | 91.5 | 90.9 | 90.9 | +3.5 / 6.42 | +3.51 / 6.46 |
| pmos | SS low Vds, no floor (mV/dec) | 9 | 76 | 83.3 | 83.4 | +4.36 / 8.68 | +4.32 / 8.69 |
| pmos | Vth ELR (V) | 9 | 0.425 | 0.459 | 0.46 | +0.00305 / 0.0109 | +0.00316 / 0.0111 |
| pmos | Vth constant current (V) | 9 | 0.419 | 0.429 | 0.429 | -0.00343 / 0.00571 | -0.0023 / 0.00606 |
| pmos | Vov at gm peak (V) | 9 | 0.298 | 0.291 | 0.29 | +0.00806 / 0.0174 | +0.0024 / 0.0121 |
| pmos | gm(VDD)/gm_max | 9 | 0.567 | 0.572 | 0.571 | +0.0089 / 0.0423 | +0.0132 / 0.0402 |
| pmos | I(VDD, low Vds) (ln ratio to meas) | 9 | 4.59e-06 | 4.84e-06 | 4.79e-06 | +0.00524 / 0.0251 | +0.000532 / 0.0303 |
| pmos | I(VDD, VDD) (ln ratio to meas) | 9 | 4.27e-05 | 4.36e-05 | 4.36e-05 | +0.0131 / 0.0139 | +0.0089 / 0.0153 |
| pmos | DIBL (mV/V) | 9 | 10.9 | 9.34 | 11.7 | -1.31 / 4.16 | +0.277 / 4.09 |
| pmos | gds/I sat, top Vgs (1/V) | 9 | 0.0467 | 0.0576 | 0.0532 | +0.0139 / 0.0139 | +0.0082 / 0.0109 |
| pmos | gds/I sat, mid Vgs (1/V) | 9 | 0.0197 | 0.00666 | 0.0263 | -0.0174 / 0.0174 | -0.00488 / 0.0146 |
| pmos | gds(0.85-1 VDD)/gds(0.55-0.7 VDD) | 9 | 0.127 | 0.175 | 0.142 | +0.00817 / 0.0395 | -0.0162 / 0.0589 |
| pmos | G(0.15VDD)/G(0.03VDD) | 9 | 0.918 | 0.926 | 0.925 | +0.00127 / 0.00672 | +0.00521 / 0.00617 |

Findings (sky130):
- pMOS linear-region current at |Vds| = 0.01 V: the card is high by ln ratio +0.411 at |Vgs| = VDD (10/10 devices
  card > measured); card x exp(g) +0.037 (MAE 0.411 -> 0.252). Saturation current at (VDD, VDD) is already within
  +0.009 and g leaves it (+0.009). Figure `10_idvg_lowvds_all_devices_sky130.png`.
- pMOS Id-Vd onset: the residual changes sign along |Vds| at top |Vgs|: card high at 0.01 V, card low by +0.23 to +0.25
  at 0.1-0.2 VDD (10/10 devices r > 0 at 0.1-0.3 VDD), 0 at VDD. The measured conductance ratio
  G(0.15 VDD) / G(0.03 VDD) is 0.98-1.34 (7/10 > 1); the card gives 0.94-1.00 (0/10 > 1); card x exp(g) gives
  > 1 on 10/10 (median 1.51, overshoot against measured median 1.11). g reproduces the sign pattern (g -0.29 at
  0.03 VDD, +0.20 at 0.2 VDD). Figure `12_idvd_onset_pmos_sky130.png`.
- The refit card M2 (refit with g in the loop) keeps the same residual pattern: low-|Vds| pMOS -0.39 to -0.44, Id-Vd
  onset +0.21 to +0.22 at 0.1-0.2 VDD (table above, rows "refit M2"; figure `13_residual_profiles_sky130_refit_M2.png`).
  The refit does not absorb this component into the 43 parameters; g carries it.
- Subthreshold slope: measured SS (nMOS median 57.5 mV/dec low |Vds|, 45.1 high |Vds|) is 2.9-3.8 x kT/q ln10 at 77 K;
  the card SS is larger (84.8 / 80.0). g changes the median card SS by <= 2.5 mV/dec. **Interpretation:** the
  SS mismatch is real and is not what g fixes; the subthreshold region carries |share| <= 0.001 of the RRMS reduction.
- Mobility / gm shape: gm(VDD)/gm_max nMOS measured 0.18, card 0.079, card x g 0.080 (unchanged); pMOS measured 0.41,
  card 0.63, card x g 0.47 (mean error +0.28 -> +0.15, MAE 0.42 -> 0.43). Measured gm is noise-limited.
- Output conductance in saturation (gds/I at top |Vgs|): nMOS card error +0.008 -> +0.005 1/V with g; pMOS +0.053 ->
  +0.050. g leaves it. No kink metric: gds(0.85-1 VDD)/gds(0.55-0.7 VDD) is < 1 in the measured medians (0.59 nMOS,
  0.48 pMOS), i.e. no rise of gds at high |Vds|; g raises the nMOS value to 0.98 (error MAE 0.42 -> 0.73), a side
  effect, not a fix.
- Measurement consistency (same bias point in two sweeps, |Vgs| = 1.85 V, |Vds| = 0.01 V): measured Id-Vd / Id-Vg
  ratio nMOS 1.15-1.31 (8/8 > 1), pMOS 0.28-1.13 (8/10 < 0.85). The card and g are single-valued at that point, so this
  part of the residual is not removable by any bias-only model. NIST: ratio 1.00 on 24/24 pMOS (0.1 V point).

**Interpretation (tied to the numbers above):** g lowers the sky130 RRMS mainly by correcting one shape that the
BSIM4 card does not produce: in pMOS, the current at small |Vds| is lower than the card (ln -0.41 at 0.01 V) and the
current at 0.1-0.3 VDD is higher (+0.24), i.e. the measured channel conductance rises with |Vds| (G ratio > 1 on 7/10)
while the card's falls or stays flat (<= 1.00 on 10/10). The pattern is shared across the 10 pMOS (10/10 signs agree),
so a bias-only g learned on other devices predicts it (LODO g removes 0.510 of the weighted residual sum of squares).
Possible physical causes, not tested here: a source/drain access resistance that decreases with |Vds| at cryogenic
temperature (dopant freeze-out in the p-type extensions, or a non-ohmic contact). The nMOS part of the gain is small
(0.090 of the reduction) because the nMOS residual is small (weighted RMS 0.052) and less shared (pair correlation 0.21).

## 4. NIST 4 K and IHP 300 K

Profiles: `07_residual_profiles_nist4k.png`, `08_residual_profiles_ihp300k.png` (sky130: `11_residual_profiles_sky130.png`).

- NIST 4 K pMOS shows the same Id-Vd onset sign pattern as sky130 pMOS: r = -0.28 at 0.03 VDD (24/24 negative), +0.20 to
  +0.22 at 0.15-0.3 VDD (24/24 positive at 0.1-0.2 VDD); measured G(0.15 VDD)/G(0.03 VDD) 0.98-2.05 (22/24 > 1), card
  0.83-0.95 (0/24 > 1). NIST nMOS: |r| <= 0.03 on the same curve, G ratio < 1 (0/29). The full-data NIST g
  (lambda 0 V) reproduces about half of the pMOS hump (g +0.10 vs r +0.22 at 0.2 VDD); RRMS 0.0418 -> 0.0341 without
  refit; pMOS gds/I error at top |Vgs| +0.093 -> +0.035 1/V.
- NIST 4 K subthreshold: measured SS nMOS 17.7 mV/dec (22 x kT/q ln10 at 4 K), card 78.5; residual in subthreshold
  -5.47 (nMOS). g changes it by 0.05; the metric weight there is near 0.
- NIST residual is the most shared of the three datasets (pair correlation 0.736 both families; LOO shared fraction
  0.445 nMOS, 0.555 pMOS).
- IHP 300 K: no dataset-wide pattern. Weighted residual RMS 0.044 nMOS / 0.033 pMOS (sky130 0.052 / 0.153); mean
  pairwise correlation -0.064 / -0.028; LOO shared fraction -0.409 / -0.049 (the other devices' mean predicts worse
  than zero). Within class (4-5 devices) the LOO fraction is -0.713 to +0.039. Strong-inversion residual |mean| <= 0.017
  and SD <= 0.041 per region; Id-Vd onset |r| <= 0.016 at every |Vds| point; G ratio < 1 on 18/18 measured and card.
  The residual that remains is near/below threshold and device-specific in sign (nMOS sub +0.41 ± 0.70, figure 08).
  LODO g applied without refit raises the RRMS 0.0218 -> 0.0237 (GMR 1.404, 9/18 lower).

**Interpretation (tied to the numbers above):** g helps where the card has a residual shape that recurs across devices
at the same bias (sky130 77 K pMOS and NIST 4 K pMOS: superlinear Id-Vd onset, 10/10 and 24/24 sign agreement). On IHP
300 K the PSP103 starting cards leave a smaller metric-weighted residual (RMS 0.033-0.044 vs 0.153 for sky130 pMOS)
whose bias pattern is not shared between devices (pair correlation <= 0), so a g trained on the other devices adds
error to the held-out device. This agrees with the stored IHP S3_D values (+0.31 at 300 K) and the gate decision
C0_refit.

## Figures

| file | content |
|---|---|
| 01_g_full_sky130_maps.png | g heat map and line cuts, nMOS / pMOS, regions from card Vth |
| 02_residual_vs_vov_sky130.png | all points: r and g vs Vov (Id-Vg) and vs abs(Vds) (Id-Vd) |
| 03_rrms_reduction_split_sky130.png | RRMS reduction share by region x curve class |
| 04_device_metrics_sky130.png | SS, gm peak, gm(VDD)/gm_max, gds/I, DIBL, G ratio: model vs measured |
| 05_shared_pattern_by_dataset.png | weighted residual RMS, LOO shared fraction, pair correlation; 3 datasets |
| 06_device_residual_maps_sky130.png | per-device binned residual, sky130 |
| 07_residual_profiles_nist4k.png | residual and g along three curve classes, NIST 4 K |
| 08_residual_profiles_ihp300k.png | same, IHP 300 K (LODO g) |
| 09_device_residual_maps_ihp300k.png | per-device binned residual, IHP 300 K |
| 10_idvg_lowvds_all_devices_sky130.png | Id-Vg at 0.01 V: measured, card, card x g, 18 devices |
| 11_residual_profiles_sky130.png | residual and g along three curve classes, sky130 starting cards |
| 12_idvd_onset_pmos_sky130.png | top-abs(Vgs) Id-Vd below 0.7 V, 10 pMOS |
| 13_residual_profiles_sky130_refit_M2.png | as 11, for the refit M2 cards of the deployment |
