# Tables, 2026-10-20 i6_1

Generated 2026-10-20 14:42 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i6_1.py --date 2026-10-20` from stored files only. RRMS is the strict paper metric with the inclusion frozen from the measured current (pdk2 rule). I6-1 values are on the 45 TU Delft DEV devices (development evidence); TEST is not loaded.

### I6-1 dataset: TU Delft 40-nm CMOS, campaign 1, device groups, split, included curves per temperature and stock-card RRMS

| group (polarity, W, L) | type | W (um) | L (um) | copies | device ids | die (sample) | array (sub-block) | split | included curves 4.13 K | included curves 295.0 K | stock card RRMS 4.13 K (group mean) | stock card RRMS 295.0 K (group mean) |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| t40_nmos_W0p12_L0p04 | nmos | 0.12 | 0.04 | 8 | 152-159 | 3 | LC | DEV | 183/184 | 181/184 | 10.1456 | 12.5338 |
| t40_nmos_W0p12_L0p05 | nmos | 0.12 | 0.05 | 3 | 160-162 | 3 | LC | DEV | 69/69 | 68/69 | 2.5894 | 0.9536 |
| t40_nmos_W0p36_L0p04 | nmos | 0.36 | 0.04 | 2 | 190-191 | 3 | LC | DEV | 46/46 | 46/46 | 10.5099 | 18.5344 |
| t40_nmos_W1_L0p04 | nmos | 1 | 0.04 | 8 | 192-199 | 3 | HC | DEV | 184/184 | 184/184 | 7.1955 | 12.4292 |
| t40_nmos_W1_L1 | nmos | 1 | 1 | 8 | 128-135 | 3 | LC | DEV | 184/184 | 184/184 | 1.2482 | 0.9887 |
| t40_pmos_W1_L0p04 | pmos | 1 | 0.04 | 8 | 392-399 | 5 | HC | DEV | 184/184 | 184/184 | 5.5977 | 9.1892 |
| t40_pmos_W1_L1 | pmos | 1 | 1 | 8 | 328-335 | 5 | LC | DEV | 184/184 | 184/184 | 0.9318 | 0.7298 |
| t40_nmos_W1_L0p36 | nmos | 1 | 0.36 | 8 | 144-151 | 3 | LC | TEST | 184/184 | 184/184 | — | — |
| t40_nmos_W4_L4 | nmos | 4 | 4 | 8 | 136-143 | 3 | LC | TEST | 184/184 | 184/184 | — | — |
| t40_pmos_W4_L4 | pmos | 4 | 4 | 8 | 336-343 | 5 | LC | TEST | 176/184 | 184/184 | — | — |
| DEV total (45 devices) |  |  |  |  |  |  |  | DEV | 1034/1035 | 1031/1035 | 5.1053 | 7.2643 |
| TEST total (24 devices) |  |  |  |  |  |  |  | TEST | 544/552 | 552/552 | — | — |

- Source: 4TU.ResearchData doi:10.4121/c41c26f7-586f-48ca-8abe-9127b2d97c60.v1 (CC0), Data underlying the publication: Cryogenic Characterization of Low-Frequency Noise in 40-nm CMOS (G. Kiene, S. Ilik, L. Mastrodomenico, M. Babaie, F. Sebastiano, TU Delft); main.zip: repository_for_upload_slim/measurement_automation/main/results/dc/*.npy (1,717 files; I3-1 download, 16.3 MB on disk); local copy not committed; read by cryoml.explore.i6_1_tudelft.read_npy_dict (unpickler restricted to numpy reconstruction globals) under python -I.
- Selection: campaign 1, 69 low-Vt core devices (NF = 1) with a campaign-1 file at both file temperatures; nMOS on die 3, pMOS on die 5; 'array' is the dataset's sub-block (LC low-current, HC high-current). Stage temperature (mean of the file's start and end readings): label 4 = 4.13-4.14 K, label 295 = 295.00-295.01 K; deck temperature = stage temperature of the file.
- Curves per device and temperature: 12 Id-Vg (idvg@d0..d11; Vd 0.05 V and 11 values to 1.1 V) and 11 Id-Vd (idvd@g0..g10; Vd 0-1.1 V), 111 points each. Vg sweep start: nMOS 0.45 V (4.13 K), 0.3 V (295 K); pMOS -0.6 V, -0.4 V. No deep subthreshold data.
- Inclusion frozen from the measured current only (pdk2 rule): trim the leading points up to the first |I| > 1e-10 A; include iff the trimmed mean |I| > 3e-7 A (Id-Vd) or > 2e-7 A (Id-Vg). 3161 of 3174 curves included, 22-23 per device and temperature.
- Float64 currents; smallest nonzero |Id| 4.87e-14 A; no exact zeros; 308 points near Vd = 0 with sign opposite to the bias; no compliance stored; largest |Id| 0.873 mA; no curve repeats its maximum.
- Split out/research/explore/I6_1/split.json (seed 20261020, by W/L pair; written before any fit): TEST = W/L 4/4 (both polarities) and nMOS 1/0.36, 24 devices; DEV 45 devices in 7 groups. TEST curves are not loaded or simulated by any I6-1 mode. Stock card RRMS: DEV only (no fit), from out/research/explore/I6_1/stock/stock_rrms.json.


### I6-1 card and box: CMOSedu 50-nm BSIM4 card, the 43-parameter set, stock values, box per polarity and starting-fit bound hits

| parameter | nMOS stock value | nMOS box | pMOS stock value | pMOS box | starting fits at a bound, 4.13 K (of 45) | starting fits at a bound, 295.0 K (of 45) |
|---|---:|---:|---:|---:|---:|---:|
| TOXE | 1.4e-09 | log [1.4e-10, 1.4e-08] | 1.4e-09 | log [1.4e-10, 1.4e-08] | 0 | 0 |
| TOXM | 1.4e-09 | log [1.4e-10, 1.4e-08] | 1.4e-09 | log [1.4e-10, 1.4e-08] | 0 | 0 |
| XJ | 2e-08 | log [2e-09, 2e-07] | 2e-08 | log [2e-09, 2e-07] | 0 | 0 |
| NDEP | 2.8e+18 | log [2.8e+17, 2.8e+19] | 2.8e+18 | log [2.8e+17, 2.8e+19] | 0 | 0 |
| NSD | 1e+20 | log [1e+19, 1e+21] | 1e+20 | log [1e+19, 1e+21] | 0 | 0 |
| DWG | 0 | inactive | 0 | inactive | 0 | 0 |
| RSH | 3 | log [0.3, 30] | 3 | log [0.3, 30] | 1 | 0 |
| VTH0 | 0.22 | +0.22 +/- 0.5 V | -0.22 | -0.22 +/- 0.5 V | 0 | 0 |
| K1 | 0.35 | log [0.035, 3.5] | 0.39 | log [0.039, 3.9] | 0 | 0 |
| K2 | 0.05 | log [0.005, 0.5] | 0.05 | log [0.005, 0.5] | 0 | 0 |
| K3 | 0 | inactive | 0 | inactive | 0 | 0 |
| DVT0 | 2.8 | log [0.28, 28] | 3.9 | log [0.39, 39] | 0 | 0 |
| DVT1 | 0.52 | log [0.052, 5.2] | 0.635 | log [0.0635, 6.35] | 1 | 0 |
| DVT1W | 0 | inactive | 0 | inactive | 0 | 0 |
| W0 | 2.5e-06 | log [2.5e-07, 2.5e-05] | 2.5e-06 | log [2.5e-07, 2.5e-05] | 3 | 2 |
| K3B | 0 | inactive | 0 | inactive | 0 | 0 |
| VFB | -0.55 | -0.55 +/- 0.5 V | 0.55 | +0.55 +/- 0.5 V | 0 | 0 |
| LPE0 | 5.75e-08 | log [5.75e-09, 5.75e-07] | 5.75e-08 | log [5.75e-09, 5.75e-07] | 0 | 0 |
| LPEB | 2.3e-10 | log [2.3e-11, 2.3e-09] | 2.3e-10 | log [2.3e-11, 2.3e-09] | 0 | 0 |
| VSAT | 1.1e+05 | log [1.1e+04, 1.1e+06] | 9e+04 | log [9e+03, 9e+05] | 0 | 0 |
| UA | 1.6e-10 | log [1.6e-11, 1.6e-09] | 1.6e-09 | log [1.6e-10, 1.6e-08] | 0 | 0 |
| UB | 1.1e-17 | inactive | 8e-18 | inactive | 0 | 0 |
| RDSW | 150 | log [15, 1.5e+03] | 250 | log [25, 2.5e+03] | 0 | 0 |
| PRWG | 0 | inactive | 3.22e-08 | log [3.22e-09, 3.22e-07] | 0 | 0 |
| WR | 1 | log [0.1, 10] | 1 | log [0.1, 10] | 0 | 1 |
| U0 | 0.032 | log [0.0032, 0.32] | 0.0095 | log [0.00095, 0.095] | 0 | 0 |
| A0 | 2 | log [0.2, 20] | 1.2 | log [0.12, 12] | 0 | 0 |
| A2 | 1 | log [0.1, 10] | 1 | log [0.1, 10] | 0 | 0 |
| AGS | 1e-20 | inactive | 1e-20 | inactive | 0 | 0 |
| B0 | -1e-20 | lin [-1e-20, 1e-07] | -1e-20 | lin [-1e-20, 1e-07] | 1 | 4 |
| B1 | 0 | lin [0, 1e-07] | 0 | lin [0, 1e-07] | 8 | 1 |
| VOFF | -0.15 | -0.15 +/- 0.3 V | -0.15 | -0.15 +/- 0.3 V | 2 | 0 |
| NFACTOR | 1.2 | log [0.12, 12] | 2 | log [0.2, 20] | 0 | 0 |
| CIT | 0 | lin [0, 0.0025] | 0 | lin [0, 0.0025] | 5 | 6 |
| CDSCD | 0 | lin [0, 0.0025] | 6.1e-08 | log [6.1e-09, 6.1e-07] | 5 | 6 |
| ETA0 | 0.15 | log [0.015, 1.5] | 0.15 | log [0.015, 1.5] | 0 | 0 |
| DSUB | 2 | log [0.2, 20] | 0.7 | log [0.07, 7] | 0 | 0 |
| PCLM | 0.18 | log [0.018, 1.8] | 0.55 | log [0.055, 5.5] | 0 | 0 |
| PDIBLC2 | 0.022 | log [0.0022, 0.22] | 0.0055 | log [0.00055, 0.055] | 0 | 0 |
| DROUT | 0.45 | log [0.045, 4.5] | 0.56 | log [0.056, 5.6] | 0 | 0 |
| DELTA | 0.01 | log [0.001, 0.1] | 0.014 | log [0.0014, 0.14] | 0 | 0 |
| ALPHA0 | 0.074 | log [0.0074, 0.74] | 0.074 | log [0.0074, 0.74] | 1 | 0 |
| BETA0 | 30 | log [3, 300] | 30 | log [3, 300] | 2 | 1 |

- Card out/research/explore/I6_1/card/ngspice41_ro_17_4_cmosedu50nm_bsim4.mod: BSIM4 level 54 version 4.5.0, N1 (nMOS) and P1 (pMOS) copied verbatim from ngspice-41 examples/mos/ro_17_4.cir ('Short channel models from CMOS Circuit Design, Layout, and Simulation, 50nm BSIM4 models VDD=1V', CMOSedu.com, R. J. Baker); ngspice Modified BSD license. Not a card of the measured 40-nm process. lint 12 nm, no xl (Leff 16 nm at L = 40 nm), toxe 1.4 nm, TNOM 27 C.
- Temperature: deck .options temp = stage temperature; kt1, kt1l, kt2, ute, ua1, ub1, uc1, at, prt set to 0, so the card values apply at the deck temperature and temperature enters only through the built-in kT/q, band-gap and intrinsic-carrier terms. One card is fitted per device and temperature. BSIM4 is not designed for 4.2 K.
- Box (cryoml.explore.i6_1_tudelft.base_box, the same for the starting fit and every held-out refit): VTH0, VFB stock +/- 0.5 V, VOFF +/- 0.3 V; other nonzero values log |v|/10..10|v|; zero values: B0, B1 [0, 1e-7] m, CIT, CDSCD [0, 2.5e-3] F/m^2, else inactive. UB has a box but is inactive under the ParameterBox rule (as in the sky130 43-set fits). Active: nMOS 36, pMOS 37 (PRWG is zero in N1).
- B1 (wider set): universe of 54 BSIM4 names after amendment out/research/explore/I6_1/prereg_amendment_2026-10-20.json (arms.B1 universe: 54 names: DVT0W is removed); edge screen per temperature kept 16 names at 4.13 K and 16 at 295.0 K, the same set: a1, cdsc, dvtp0, dvtp1, fprout, minv, pdiblc1, pdits, pditsd, pditsl, phin, pscbe1, pscbe2, pvag, rdswmin, voffl. Amendment 2: each ngspice process is stopped after 10 s (cryoml.explore.i6_1_tudelft.TIMEOUT_S.
- Simulator: ngspice-41 native arm64 build, one process per device and card, 23 sweeps on the measured grid, 0.03 s per run. 'at a bound' counts DEV starting cards (45 per temperature, both polarities) with the parameter at a box edge after the starting fit.


### I6-1 box-width trial before the pre-registration: starting-fit RRMS per IRLS round, box factor 2 vs 10

| device | temperature | box factor | RRMS stock card | RRMS round 1 | RRMS round 2 | RRMS round 3 | simulations | parameters at a bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| t40_nmos_W1_L1_d128 | 4.13 K | 2 | 1.2145 | 0.0882 | 0.0795 | 0.0788 | 1412 | toxe, rdsw, eta0 |
| t40_nmos_W1_L1_d128 | 4.13 K | 10 | 1.2145 | 0.0768 | 0.0734 | 0.0729 | 1116 | wr |
| t40_nmos_W1_L1_d128 | 295.0 K | 2 | 0.9874 | 0.0771 | 0.0725 | 0.0697 | 1486 | ndep, nsd, drout |
| t40_nmos_W1_L1_d128 | 295.0 K | 10 | 0.9874 | 0.0507 | 0.0368 | 0.0289 | 1375 | none |
| t40_nmos_W1_L0p04_d192 | 4.13 K | 2 | 5.8528 | 0.0712 | 0.0648 | 0.0511 | 1375 | cdscd |
| t40_nmos_W1_L0p04_d192 | 4.13 K | 10 | 5.8528 | 0.0725 | 0.0566 | 0.0430 | 1116 | cdscd |
| t40_nmos_W1_L0p04_d192 | 295.0 K | 2 | 10.5536 | 0.0372 | 0.0265 | 0.0264 | 1153 | none |
| t40_nmos_W1_L0p04_d192 | 295.0 K | 10 | 10.5536 | 0.0372 | 0.0265 | 0.0264 | 1153 | none |
| t40_pmos_W1_L0p04_d392 | 4.13 K | 2 | 5.5890 | 0.0413 | 0.0389 | 0.0358 | 1180 | b0, delta |
| t40_pmos_W1_L0p04_d392 | 4.13 K | 10 | 5.5890 | 0.0424 | 0.0395 | 0.0391 | 838 | none |
| t40_pmos_W1_L0p04_d392 | 295.0 K | 2 | 9.0411 | 0.0294 | 0.0239 | 0.0220 | 1408 | none |
| t40_pmos_W1_L0p04_d392 | 295.0 K | 10 | 9.0411 | 0.0597 | 0.0172 | 0.0170 | 686 | none |

- out/research/explore/I6_1/box_trial.json: written before the pre-registration; ngspice_fit.polish from the stock card, max_nfev 1500, irls 3, diff_step 0.01, 3 DEV devices at both temperatures. Factor 2: log |v|/2..2|v|; U0, VSAT, RDSW |v|/4..4|v|; VTH0, VFB +/- 0.3 V, VOFF +/- 0.2 V. Factor 10: log |v|/10..10|v|; VTH0, VFB +/- 0.5 V, VOFF +/- 0.3 V.
- The factor-10 box was adopted for the starting cards and all refits. No correction was fitted in this trial.


### I6-1 stock-card, starting-card and C0_refit RRMS per temperature and DEV group

| temperature | group | devices | stock mean | stock median | starting card mean | starting card median | C0_refit mean | GMR(start, C0_refit) [95% CI] | devices with start < C0_refit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.13 K | all DEV | 45 | 5.1053 | 5.3436 | 0.0628 | 0.0437 | 0.0575 | 1.082 [1.052, 1.120] | 0/45 |
| 4.13 K | nMOS 0.12/0.04 | 8 | 10.1456 | 10.4812 | 0.0877 | 0.0522 | 0.0793 |  |  |
| 4.13 K | nMOS 0.12/0.05 | 3 | 2.5894 | 2.5607 | 0.1555 | 0.1623 | 0.1513 |  |  |
| 4.13 K | nMOS 0.36/0.04 | 2 | 10.5099 | 10.5099 | 0.0835 | 0.0835 | 0.0589 |  |  |
| 4.13 K | nMOS 1/0.04 | 8 | 7.1955 | 7.1063 | 0.0599 | 0.0609 | 0.0530 |  |  |
| 4.13 K | nMOS 1/1 | 8 | 1.2482 | 1.2309 | 0.0679 | 0.0692 | 0.0624 |  |  |
| 4.13 K | pMOS 1/0.04 | 8 | 5.5977 | 5.5846 | 0.0306 | 0.0288 | 0.0305 |  |  |
| 4.13 K | pMOS 1/1 | 8 | 0.9318 | 0.9288 | 0.0282 | 0.0297 | 0.0267 |  |  |
| 295.0 K | all DEV | 45 | 7.2643 | 8.6387 | 0.0271 | 0.0227 | 0.0255 | 1.045 [1.019, 1.082] | 0/45 |
| 295.0 K | nMOS 0.12/0.04 | 8 | 12.5338 | 13.6622 | 0.0455 | 0.0419 | 0.0399 |  |  |
| 295.0 K | nMOS 0.12/0.05 | 3 | 0.9536 | 0.9608 | 0.0268 | 0.0287 | 0.0265 |  |  |
| 295.0 K | nMOS 0.36/0.04 | 2 | 18.5344 | 18.5344 | 0.0328 | 0.0328 | 0.0327 |  |  |
| 295.0 K | nMOS 1/0.04 | 8 | 12.4292 | 12.0284 | 0.0352 | 0.0357 | 0.0345 |  |  |
| 295.0 K | nMOS 1/1 | 8 | 0.9887 | 0.9880 | 0.0150 | 0.0127 | 0.0142 |  |  |
| 295.0 K | pMOS 1/0.04 | 8 | 9.1892 | 9.1786 | 0.0212 | 0.0186 | 0.0204 |  |  |
| 295.0 K | pMOS 1/1 | 8 | 0.7298 | 0.7303 | 0.0171 | 0.0190 | 0.0166 |  |  |

- Strict paper RRMS (per included curve RMSE over the trimmed points / trimmed mean |I_measured|, device value = mean over included curves) with the frozen measured inclusion; group and all-DEV means over devices.
- Starting card: ngspice_fit.polish from the stock card on the 43-parameter box, max_nfev 3000, irls 3, diff_step 0.01; one per DEV device and temperature (queue job I6_1_start, 1083 s, 0 simulator timeouts). C0_refit starts from it (max_nfev 400, irls 2).
- The stock card is a generic 50-nm card; its RRMS is 0.66-23.1 per device and temperature (threshold and current level of another process). The starting-card all-DEV mean at 4.13 K is 2.32 times the 295.0 K value.


### I6-1 decision conditions per temperature (4.13 K decides, 295.0 K is the control)

| temperature | R1: GMR(M_bias, C0_refit) [device 95% CI] | cluster 95% CI (7 groups) | devices with M_bias < C0_refit | R1 | R2: M_bias mean vs T1_inloop mean + 0.005 | R2 | R3: M_bias failed devices | R3 | GMR(M_bias, B1) [device CI] [cluster CI] wins | GMR(M_bias, B2) [device CI] [cluster CI] wins | GMR(M_bias, B3_d3) [device CI] [cluster CI] wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.13 K (decides) | 0.939 [0.877, 1.004] | [0.811, 1.081] | 27/45 | not met | 0.0533 vs 0.0584 | met | 0 | met | 0.885 [0.819, 0.955] [0.737, 1.049] 28/45 (beaten) | 0.954 [0.871, 1.054] [0.773, 1.178] 28/45 (not beaten) | 0.983 [0.956, 1.015] [0.953, 1.012] 29/45 (not beaten) |
| 295.0 K (control) | 0.797 [0.688, 0.935] | [0.576, 1.087] | 35/45 | met | 0.0196 vs 0.0283 | met | 0 | met | 0.769 [0.660, 0.904] [0.554, 1.052] 35/45 (beaten) | 0.788 [0.674, 0.928] [0.560, 1.085] 35/45 (beaten) | 0.896 [0.839, 0.955] [0.822, 0.962] 34/45 (beaten) |

- Pre-registration out/research/explore/I6_1/prereg.json (commit a967f04, before any correction fit and before the starting-card fits); amendment out/research/explore/I6_1/prereg_amendment_2026-10-20.json (after the smoke runs, before any full run): DVT0W removed from the B1 universe, 10 s simulator timeout.
- Conditions per temperature over the 45 DEV devices: R1 device-bootstrap CI upper of GMR(M_bias, C0_refit) < 1; R2 M_bias all-device mean <= T1_inloop mean + 0.005; R3 0 failed M_bias devices; baseline X beaten if the device-bootstrap CI upper of GMR(M_bias, X) < 1. The prediction: R1-R3 and all baselines beaten at 4.13 K, R1 not met at 295.0 K.
- At 4.13 K R1 is not met (CI upper 1.004); R2 and R3 are met; B1 is beaten, B2 and B3_d3 are not. At 295.0 K R1, R2, R3 are met and every baseline is beaten, which is the opposite of the 295 K prediction.
- GMR(A, B) = exp(mean ln(RRMS_A / RRMS_B)) over the 45 DEV devices; device bootstrap 10000 resamples, default_rng(0); cluster bootstrap resamples the 7 (polarity, W, L) groups (reported, not deciding; the 2-8 copies of one geometry are not independent). wins = devices with A < B. Recomputed here and checked against out/research/explore/I6_1/decision.json.


### I6-1 outcome logic: premise P, temperature contrast K and the pre-registered outcome list

| item | pre-registered condition | value | evaluated | selected |
|---|---|---|---|---|
| premise P | C0_refit mean at 4.13 K >= 1.5 x C0_refit mean at 295.0 K | 0.0575 vs 1.5 x 0.0255 = 0.0383 (ratio 2.25) | holds |  |
| contrast K | device CI upper of K < 1, K = GMR of (M_bias / C0_refit at 4.13 K) / (M_bias / C0_refit at 295.0 K) | 1.177 [1.016, 1.348], cluster [0.897, 1.502]; 13/45 devices with the 4.13 K ratio below the 295.0 K ratio | fails |  |
| outcome: refuted | R1 fails at 4 K |  | holds | rule output |
| outcome: inconclusive | R1 holds at 4 K, R2 or R3 fails at 4 K |  | fails |  |
| outcome: matched by baselines | R1-R3 hold at 4 K, at least one of B1, B2, B3_d3 not beaten at 4 K |  | fails |  |
| outcome: supported | R1-R3 hold and all baselines beaten at 4 K; R1 fails at 295 K; P holds |  | fails |  |
| outcome: supported, premise not met | as 'supported' but P fails |  | fails |  |
| outcome: temperature-graded | R1-R3 hold and all baselines beaten at 4 K; R1 holds at 295 K; K holds |  | fails |  |
| outcome: not temperature-specific | R1-R3 hold and all baselines beaten at 4 K; R1 holds at 295 K; K fails |  | fails |  |

- The outcome list is evaluated in the pre-registered order and the first label whose condition holds is the rule output (scripts/research/explore/i6_1_tudelft.py, function outcome). Each condition is also evaluated on its own here. Rule output: refuted (4.2 K R1 fails: the correction does not lower held-out error where the compact model is used far outside its validity range).
- Pre-registered interpretation of 'refuted': the correction does not lower held-out error even where BSIM4 is used outside its validity range; together with I5-1 the I3-2 gain is not explained by model-form error of this kind.
- At 295.0 K the correction lowers the held-out RRMS (R1 met). K > 1 means the M_bias / C0_refit ratio is larger at 4.13 K than at 295.0 K, so the correction lowers the error more at 295.0 K.


### I6-1 results per temperature for every arm (45 DEV devices), with GMR against M_bias and against C0_refit, device and cluster CIs

| temperature | arm | definition | all-device | nMOS (29) | pMOS (16) | combined | median | failed | GMR(M_bias, X) [device 95% CI] | GMR(M_bias, X) cluster 95% CI | devices with M_bias < X | GMR(X, C0_refit) [device 95% CI] | GMR(X, C0_refit) cluster 95% CI | devices with X < C0_refit |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.13 K | start | starting card (43-set fit from the stock card), no held-out refit | 0.0628 | 0.0813 | 0.0294 | 0.0553 | 0.0437 | 0/45 | 0.867 [0.803, 0.936] | [0.725, 1.027] | 29/45 | 1.082 [1.052, 1.120] | [1.044, 1.127] | 0/45 |
| 4.13 K | C0_refit | control: held-out card refit, no correction | 0.0575 | 0.0734 | 0.0286 | 0.0510 | 0.0408 | 0/45 | 0.939 [0.877, 1.004] | [0.811, 1.081] | 27/45 |  |  |  |
| 4.13 K | M_bias | I3-2 correction, refit with g in the loop (method under test) | 0.0533 | 0.0644 | 0.0331 | 0.0488 | 0.0373 | 0/45 |  |  |  | 0.939 [0.877, 1.004] | [0.811, 1.081] | 27/45 |
| 4.13 K | T1_inloop | per-(polarity, curve key) gain table (not deployable) | 0.0534 | 0.0660 | 0.0305 | 0.0483 | 0.0374 | 0/45 | 1.014 [0.964, 1.067] | [0.926, 1.101] | 20/45 | 0.926 [0.884, 0.971] | [0.851, 1.016] | 30/45 |
| 4.13 K | B1 | wider BSIM4 parameter set (base + 16 screened extras), no correction | 0.0612 | 0.0791 | 0.0290 | 0.0540 | 0.0437 | 0/45 | 0.885 [0.819, 0.955] | [0.737, 1.049] | 28/45 | 1.060 [1.033, 1.095] | [1.024, 1.108] | 11/45 |
| 4.13 K | B2 | extrinsic series resistance + self-heating factor, no learned term | 0.0582 | 0.0757 | 0.0263 | 0.0510 | 0.0424 | 0/45 | 0.954 [0.871, 1.054] | [0.773, 1.178] | 28/45 | 0.984 [0.935, 1.024] | [0.911, 1.054] | 24/45 |
| 4.13 K | B3_d3 | shared polynomial g, total degree 3 (20 coefficients) | 0.0539 | 0.0660 | 0.0321 | 0.0490 | 0.0381 | 0/45 | 0.983 [0.956, 1.015] | [0.953, 1.012] | 29/45 | 0.955 [0.902, 1.014] | [0.847, 1.071] | 28/45 |
| 4.13 K | B5 | per-device correction trained on the held-out device (in-sample reference) | 0.0279 | 0.0395 | 0.0068 | 0.0232 | 0.0162 | 0/45 | 2.624 [2.249, 3.076] | [1.798, 3.768] | 0/45 | 0.358 [0.315, 0.407] | [0.278, 0.471] | 45/45 |
| 295.0 K | start | starting card (43-set fit from the stock card), no held-out refit | 0.0271 | 0.0314 | 0.0191 | 0.0253 | 0.0227 | 0/45 | 0.763 [0.655, 0.898] | [0.550, 1.047] | 35/45 | 1.045 [1.019, 1.082] | [1.021, 1.082] | 0/45 |
| 295.0 K | C0_refit | control: held-out card refit, no correction | 0.0255 | 0.0294 | 0.0185 | 0.0239 | 0.0226 | 0/45 | 0.797 [0.688, 0.935] | [0.576, 1.087] | 35/45 |  |  |  |
| 295.0 K | M_bias | I3-2 correction, refit with g in the loop (method under test) | 0.0196 | 0.0211 | 0.0168 | 0.0190 | 0.0175 | 0/45 |  |  |  | 0.797 [0.688, 0.935] | [0.576, 1.087] | 35/45 |
| 295.0 K | T1_inloop | per-(polarity, curve key) gain table (not deployable) | 0.0233 | 0.0257 | 0.0189 | 0.0223 | 0.0199 | 0/45 | 0.830 [0.757, 0.913] | [0.680, 0.993] | 34/45 | 0.961 [0.901, 1.029] | [0.845, 1.097] | 27/45 |
| 295.0 K | B1 | wider BSIM4 parameter set (base + 16 screened extras), no correction | 0.0268 | 0.0310 | 0.0191 | 0.0251 | 0.0226 | 0/45 | 0.769 [0.660, 0.904] | [0.554, 1.052] | 35/45 | 1.037 [1.012, 1.073] | [1.014, 1.070] | 12/45 |
| 295.0 K | B2 | extrinsic series resistance + self-heating factor, no learned term | 0.0264 | 0.0308 | 0.0183 | 0.0246 | 0.0222 | 0/45 | 0.788 [0.674, 0.928] | [0.560, 1.085] | 35/45 | 1.012 [0.981, 1.050] | [0.970, 1.054] | 30/45 |
| 295.0 K | B3_d3 | shared polynomial g, total degree 3 (20 coefficients) | 0.0215 | 0.0236 | 0.0177 | 0.0206 | 0.0191 | 0/45 | 0.896 [0.839, 0.955] | [0.822, 0.962] | 34/45 | 0.889 [0.798, 1.005] | [0.693, 1.140] | 33/45 |
| 295.0 K | B5 | per-device correction trained on the held-out device (in-sample reference) | 0.0051 | 0.0060 | 0.0034 | 0.0047 | 0.0053 | 0/45 | 3.660 [3.247, 4.147] | [2.987, 4.248] | 0/45 | 0.218 [0.187, 0.256] | [0.178, 0.304] | 45/45 |

- Per temperature, leave-one-geometry-out over the 7 DEV groups: corrections are trained on the DEV devices of the other 6 groups (both polarities, one network, VDD 1.1 V); every device of the held-out group is refit from its starting card on the 43-parameter box (max_nfev 400, irls 2, diff_step 0.01) and scored on a fresh ngspice-41 simulation of the refit card, multiplied by exp(g) for M_bias, T1_inloop, B3_d3 and B5.
- all-device = mean over 45 devices (29 nMOS, 16 pMOS); combined = mean of the nMOS and pMOS family means. start = the starting card without a held-out refit. 0 failed devices and 0 simulator timeouts in every arm.
- B1: base box plus the 16 screened extras (max_nfev 540-550). The B1 refit returned the starting card unchanged (no accepted step) for 16/45 devices at 4.13 K and 13/45 at 295.0 K; B1 is above C0_refit on 34/45 and 33/45 devices.
- B2: fitted R_ext / box upper median 8.2e-06 (4.13 K) and 9.9e-06 (295.0 K), maximum 0.070 / 0.051; kappa range [-0.099, +0.047] / [-0.018, +0.073].
- B5 is trained on the held-out device's own measured curves and is an in-sample reference, not a held-out method. Queue jobs I6_1_lodo_4 (1201 s; training 419 s, 315 refits 781 s) and I6_1_lodo_295 (1185 s), 2026-10-20.


### I6-1 per-group (fold) means of every held-out arm and GMR(M_bias, C0_refit) within the group

| temperature | held-out group | devices | C0_refit | M_bias | T1_inloop | B1 | B2 | B3_d3 | B5 | GMR(M_bias, C0_refit) within group | devices with M_bias < C0_refit | lowest of B1, B2, B3_d3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.13 K | nMOS 0.12/0.04 | 8 | 0.0793 | 0.0700 | 0.0726 | 0.0847 | 0.0826 | 0.0727 | 0.0457 | 0.797 | 6/8 | B3_d3 0.0727 |
| 4.13 K | nMOS 0.12/0.05 | 3 | 0.1513 | 0.1414 | 0.1452 | 0.1516 | 0.1515 | 0.1426 | 0.0936 | 0.912 | 3/3 | B3_d3 0.1426 |
| 4.13 K | nMOS 0.36/0.04 | 2 | 0.0589 | 0.0479 | 0.0576 | 0.0823 | 0.0574 | 0.0490 | 0.0361 | 0.804 | 2/2 | B3_d3 0.0490 |
| 4.13 K | nMOS 1/0.04 | 8 | 0.0530 | 0.0414 | 0.0454 | 0.0596 | 0.0587 | 0.0418 | 0.0191 | 0.754 | 7/8 | B3_d3 0.0418 |
| 4.13 K | nMOS 1/1 | 8 | 0.0624 | 0.0570 | 0.0525 | 0.0649 | 0.0620 | 0.0590 | 0.0344 | 0.930 | 7/8 | B3_d3 0.0590 |
| 4.13 K | pMOS 1/0.04 | 8 | 0.0305 | 0.0342 | 0.0333 | 0.0303 | 0.0295 | 0.0336 | 0.0071 | 1.137 | 2/8 | B2 0.0295 |
| 4.13 K | pMOS 1/1 | 8 | 0.0267 | 0.0320 | 0.0277 | 0.0277 | 0.0231 | 0.0307 | 0.0064 | 1.204 | 0/8 | B2 0.0231 |
| 295.0 K | nMOS 0.12/0.04 | 8 | 0.0399 | 0.0274 | 0.0337 | 0.0447 | 0.0440 | 0.0310 | 0.0063 | 0.612 | 7/8 | B3_d3 0.0310 |
| 295.0 K | nMOS 0.12/0.05 | 3 | 0.0265 | 0.0106 | 0.0207 | 0.0268 | 0.0264 | 0.0149 | 0.0052 | 0.406 | 3/3 | B3_d3 0.0149 |
| 295.0 K | nMOS 0.36/0.04 | 2 | 0.0327 | 0.0147 | 0.0252 | 0.0326 | 0.0326 | 0.0186 | 0.0059 | 0.403 | 2/2 | B3_d3 0.0186 |
| 295.0 K | nMOS 1/0.04 | 8 | 0.0345 | 0.0202 | 0.0281 | 0.0348 | 0.0349 | 0.0229 | 0.0059 | 0.578 | 8/8 | B3_d3 0.0229 |
| 295.0 K | nMOS 1/1 | 8 | 0.0142 | 0.0214 | 0.0174 | 0.0149 | 0.0148 | 0.0213 | 0.0063 | 1.548 | 1/8 | B2 0.0148 |
| 295.0 K | pMOS 1/0.04 | 8 | 0.0204 | 0.0182 | 0.0220 | 0.0212 | 0.0209 | 0.0184 | 0.0037 | 0.890 | 8/8 | B3_d3 0.0184 |
| 295.0 K | pMOS 1/1 | 8 | 0.0166 | 0.0154 | 0.0158 | 0.0169 | 0.0157 | 0.0170 | 0.0032 | 1.009 | 6/8 | B2 0.0157 |

- Each row is one fold: the held-out group's devices, with corrections trained on the other 6 DEV groups. Per-group means are reported, not deciding (pre-registration). The within-group GMR has no CI (2-8 copies of one geometry). Means checked against out/research/explore/I6_1/decision.json per_group.
- At 4.13 K M_bias is above C0_refit in both pMOS groups (1/0.04 and 1/1) and below it in the 5 nMOS groups; at 295.0 K M_bias is above C0_refit in the nMOS 1/1 group only.


### I6-1 temperature contrast K and the fraction of the C0_refit -> B5 gap closed by M_bias per temperature

| temperature | C0_refit mean | M_bias mean | B5 mean (in-sample) | GMR(M_bias, C0_refit) [device CI] [cluster CI] | GMR(B5, C0_refit) [device CI] [cluster CI] | fraction of C0->B5 gap closed by M_bias (means) | fraction of C0->B5 gap closed by M_bias (log GMR) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4.13 K | 0.0575 | 0.0533 | 0.0279 | 0.939 [0.877, 1.004] [0.811, 1.081] | 0.358 [0.315, 0.407] [0.278, 0.471] | +0.142 | +0.062 |
| 295.0 K | 0.0255 | 0.0196 | 0.0051 | 0.797 [0.688, 0.935] [0.576, 1.087] | 0.218 [0.187, 0.256] [0.178, 0.304] | +0.290 | +0.149 |
| K (4.13 K over 295.0 K) | — | — | — | 1.177 [1.016, 1.348] [0.897, 1.502] | 13/45 devices: 4.13 K ratio < 295.0 K ratio |  |  |

- fraction of mean gap = (C0_refit - M_bias) / (C0_refit - B5); fraction of log GMR = ln GMR(M_bias, C0_refit) / ln GMR(B5, C0_refit). B5 is the in-sample reference (correction trained on the held-out device), so the gap is the part of the C0_refit error that a bias-only correction of this form removes when it sees the device.
- K = exp(mean_i [ln(M4_i / C4_i) - ln(M295_i / C295_i)]) over the 45 DEV devices (pre-registered temperature contrast); the K row lists the device CI, then the cluster CI. K > 1: the M_bias / C0_refit ratio is larger at 4.13 K.


### Cross-dataset summary of the I3-2 bias-only correction: sky130 77 K (I3-2 / I4-1), IHP SG13G2 233-398 K (I5-1), TU Delft 40 nm 4.13 K and 295.0 K (I6-1)

| dataset | devices | card, refit parameters, simulator | card calibrated for the measured condition | C0_refit mean | M_bias mean | GMR(M_bias, C0_refit) [device 95% CI] wins | GMR(M_bias, C0_refit) cluster 95% CI | lowest-mean baseline of B1, B2, B3_d3 | GMR(M_bias, best baseline) [device 95% CI] | B5 mean (in-sample) | fraction of C0->B5 gap closed (means) | fraction of C0->B5 gap closed (log GMR) | pre-registered rule output |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sky130 77 K REAL_DEV (I3-2 / I4-1) | 12 | BSIM4, 43 parameters, ngspice-41 | yes: the paper's 77 K sky130 card, refit | 0.1072 | 0.0806 | 0.769 [0.666, 0.886] 11/12 | = device CI | B3_d3 0.0956 | 0.844 [0.751, 0.949] | 0.0410 | +0.403 | +0.298 | I3-2 CONTINUE; I4-1 supported against baselines |
| IHP SG13G2 233 K DEV (I5-1) | 18 | PSP103, 12 parameters, NGSpice-46 | yes: foundry PSP103 card at a rated temperature | 0.0273 | 0.0281 | 1.106 [0.908, 1.425] 11/18 | = device CI | B1 0.0235 | 1.393 [1.077, 1.876] | 0.0084 | -0.037 | -0.087 | not replicated; matched by B1, B2, B3_d3 |
| IHP SG13G2 300 K DEV (I5-1, decides) | 18 | PSP103, 12 parameters, NGSpice-46 | yes: foundry PSP103 card at a rated temperature | 0.0191 | 0.0203 | 1.214 [1.008, 1.537] 9/18 | = device CI | B1 0.0160 | 1.567 [1.261, 1.981] | 0.0054 | -0.090 | -0.185 | not replicated; matched by B1, B2 |
| IHP SG13G2 343 K DEV (I5-1) | 18 | PSP103, 12 parameters, NGSpice-46 | yes: foundry PSP103 card at a rated temperature | 0.0190 | 0.0194 | 1.181 [0.983, 1.502] 8/18 | = device CI | B2 0.0150 | 1.328 [1.066, 1.723] | 0.0047 | -0.027 | -0.152 | not replicated; matched by B1, B2 |
| IHP SG13G2 398 K DEV (I5-1) | 18 | PSP103, 12 parameters, NGSpice-46 | yes: foundry PSP103 card at a rated temperature | 0.0197 | 0.0202 | 1.272 [1.030, 1.627] 11/18 | = device CI | B2 0.0151 | 1.435 [1.151, 1.844] | 0.0037 | -0.032 | -0.205 | not replicated; matched by B1, B2, B3_d3 |
| TU Delft 40 nm 4.13 K DEV (I6-1, decides) | 45 | BSIM4, 43-set (36/37 active), ngspice-41 | no: generic ngspice example card (CMOSedu 50 nm), temperature coefficients zeroed | 0.0575 | 0.0533 | 0.939 [0.877, 1.004] 27/45 | [0.811, 1.081] | B3_d3 0.0539 | 0.983 [0.956, 1.015] | 0.0279 | +0.142 | +0.062 | refuted (R1 not met; baselines beaten: B1) |
| TU Delft 40 nm 295.0 K DEV (I6-1, control) | 45 | BSIM4, 43-set (36/37 active), ngspice-41 | no: generic ngspice example card (CMOSedu 50 nm), temperature coefficients zeroed | 0.0255 | 0.0196 | 0.797 [0.688, 0.935] 35/45 | [0.576, 1.087] | B3_d3 0.0215 | 0.896 [0.839, 0.955] | 0.0051 | +0.290 | +0.149 | R1 met; baselines beaten: B1, B2, B3_d3 |

- Sources: sky130 out/research/explore/I4_1/real_dev/results.json (12 REAL_DEV devices, 5 nMOS + 7 pMOS; inclusion frozen from the published-card baseline); IHP out/research/explore/I5_1/lodo_233K/results.json, out/research/explore/I5_1/lodo_300K/results.json, out/research/explore/I5_1/lodo_343K/results.json, out/research/explore/I5_1/lodo_398K/results.json, out/research/explore/I5_1/decision.json (18 DEV DUTs; inclusion frozen from the measured current); TU Delft out/research/explore/I6_1/lodo_4K/results.json, out/research/explore/I6_1/lodo_295K/results.json (45 DEV devices in 7 groups; inclusion frozen from the measured current).
- Same M_bias recipe (I3-2, unchanged) and held-out refit settings (max_nfev 400, irls 2) in every row. Folds: sky130 leave-one-device-out; IHP leave-one-DUT-out within class (LV / HV); TU Delft leave-one-geometry-out (both polarities in one training pool).
- Card calibration: sky130 77 K uses the paper's 77 K card (parameters published for that temperature) as the refit start; IHP uses the foundry PSP103 card, which the PDK supports at 233-398 K; TU Delft uses a generic 50-nm example card that is not a card of the measured process, with the temperature coefficients set to 0, at 4.13 K and 295.0 K. Every row refits the card per device before scoring.
- best baseline = lowest all-device mean of B1, B2, B3_d3 (the baselines common to I4-1, I5-1 and I6-1). sky130 also has B3_d2 0.1034 and B4 0.1317, both above B3_d3. B1 and B2 differ between datasets (sky130 B1 19 screened BSIM4 extras, IHP 70-73 PSP extras, TU Delft 16 BSIM4 extras).
- cluster CI: TU Delft resamples the 7 geometry groups; sky130 and IHP have one device per geometry, so the device bootstrap is the cluster bootstrap there ('= device CI'). fraction of mean gap = (C0_refit - M_bias) / (C0_refit - B5); fraction of log GMR = ln GMR(M_bias, C0_refit) / ln GMR(B5, C0_refit); negative: M_bias above C0_refit.

