# Tables, 2026-10-23 i20_1

Generated 2026-10-23 13:16 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i20_1.py --date 2026-10-23`.

### I20-1 audit: stored NIST cards rescored per BSIM4 clamp state

| dataset | arm | devices | A2_eff > 1 | C != L | state L (stored) | state C | state U | max |L - stored| |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NIST lvt | stock card | 57 | 0 | 0 | 0.6775 / 0.6944 | 0.6775 / 0.6944 | 0.5396 / 0.5752 | 0 |
|  | start (C0, no refit) | 57 | 16 | 14 | 0.0411 / 0.0440 | 0.0418 / 0.0446 | 0.0396 / 0.0444 | 0 |
|  | C0_refit | 53 | 20 | 20 | 0.0400 / 0.0428 | 0.0407 / 0.0435 | 0.0400 / 0.0428 | 0 |
|  | T1_inloop | 53 | 22 | 22 | 0.0349 / 0.0372 | 0.0356 / 0.0379 | 0.0349 / 0.0373 | 0 |
|  | M_bias v1 | 53 | 13 | 13 | 0.0315 / 0.0337 | 0.0321 / 0.0342 | 0.0315 / 0.0337 | 0 |
|  | B1 | 53 | 14 | 14 | 0.0407 / 0.0436 | 0.0413 / 0.0441 | 0.0407 / 0.0436 | 0 |
|  | B2 | 53 | 18 | 17 | 0.0390 / 0.0417 | 0.0396 / 0.0423 | 0.0390 / 0.0417 | 0 |
|  | B3_d3 | 53 | 16 | 16 | 0.0316 / 0.0338 | 0.0325 / 0.0345 | 0.0316 / 0.0338 | 0 |
|  | I9_2:const | 53 | 20 | 20 | 0.0399 / 0.0427 | 0.0406 / 0.0433 | 0.0399 / 0.0428 | 0 |
|  | I9_2:gam | 53 | 14 | 14 | 0.0254 / 0.0271 | 0.0261 / 0.0277 | 0.0254 / 0.0271 | 0 |
|  | I9_2:gp | 53 | 10 | 10 | 0.0463 / 0.0501 | 0.0468 / 0.0506 | 0.0464 / 0.0501 | 0 |
|  | I9_2:lin | 53 | 18 | 18 | 0.0400 / 0.0428 | 0.0406 / 0.0434 | 0.0400 / 0.0428 | 0 |
|  | I9_2:mlp16_s1 | 53 | 11 | 11 | 0.0313 / 0.0335 | 0.0318 / 0.0339 | 0.0313 / 0.0335 | 0 |
|  | I9_2:mlp32 | 53 | 13 | 13 | 0.0315 / 0.0337 | 0.0322 / 0.0343 | 0.0315 / 0.0337 | 0 |
|  | I9_2:mlp8 | 53 | 14 | 14 | 0.0311 / 0.0333 | 0.0318 / 0.0339 | 0.0311 / 0.0333 | 0 |
|  | I9_2:poly2 | 53 | 16 | 16 | 0.0354 / 0.0377 | 0.0360 / 0.0383 | 0.0354 / 0.0377 | 0 |
|  | I9_2:poly4 | 53 | 8 | 8 | 0.0279 / 0.0298 | 0.0285 / 0.0303 | 0.0280 / 0.0298 | 0 |
|  | I9_2:poly5 | 53 | 10 | 10 | 0.0267 / 0.0285 | 0.0272 / 0.0290 | 0.0267 / 0.0285 | 0 |
|  | I9_2:mlp16_rep | 53 | 13 | 13 | 0.0310 / 0.0332 | 0.0317 / 0.0338 | 0.0311 / 0.0332 | 0 |
| NIST RT card | stock card | 87 | 0 | 0 | 1.6270 / 1.7458 | 1.6270 / 1.7458 | 1.6270 / 1.7458 | 0 |
|  | start (C0, no refit) | 87 | 35 | 35 | 0.0576 / 0.0603 | 0.0614 / 0.0642 | 0.0596 / 0.0625 | 0 |
|  | C0_refit | 87 | 41 | 41 | 0.0555 / 0.0580 | 0.0603 / 0.0631 | 0.0582 / 0.0610 | 0 |
|  | T1_inloop | 87 | 41 | 41 | 0.0531 / 0.0554 | 0.0575 / 0.0601 | 0.0555 / 0.0581 | 0 |
|  | M_bias v1 | 87 | 35 | 35 | 0.0455 / 0.0480 | 0.0498 / 0.0524 | 0.0476 / 0.0503 | 0 |
|  | M_bias_lvt (I9-1 transfer) | 87 | 36 | 36 | 0.0475 / 0.0495 | 0.0518 / 0.0540 | 0.0494 / 0.0516 | 0 |
|  | B1 | 87 | 35 | 35 | 0.0567 / 0.0593 | 0.0603 / 0.0631 | 0.0586 / 0.0614 | 0 |
|  | B2 | 87 | 37 | 37 | 0.0552 / 0.0577 | 0.0591 / 0.0618 | 0.0572 / 0.0600 | 0 |
|  | B3_d3 | 87 | 34 | 34 | 0.0459 / 0.0480 | 0.0500 / 0.0522 | 0.0478 / 0.0501 | 0 |

- Stored cards and stored corrections of every arm, no refit and no retraining, rescored per clamp state. L: stored deck (first analysis clamped by b4check.c, later ones with the card values); C: clamp written into the card; U: warm-up op (every curve with the card values). RRMS: mean over devices with a finite value / combined (mean of the nMOS and pMOS family means); NIST lvt arms cover the 53 finite-start devices (stock and start: 57, 4 failed in L and C by I8-1 amendment A1).
- V1: state L reproduces every stored row of every arm (max |diff| 0.0e+00 on start / C0_refit / M_bias v1; 0 on all other arms). V2 (amendment A1): C = L = U on 1231 unclamped cards without a fatal model check; C = L and U != L on 7 cards with 'Pclm is not positive' (5 stock, 2 start, I8-1 A1 devices). V3: b4check.c printed A2 = computed A2_eff on every clamped card (max rel 4.9e-06); no A2 message on unclamped cards; state-C curve 1 = state-L curve 1 (max |diff| 0e+00).
- Other size-dependent b4check.c clamps on stored cards: 0. Clamped cards: A2_eff > 1 (no card has A2_eff < 0.01). 'C != L': devices whose RRMS differs between the states.


### I20-1 audit: paired GMRs of the stored arms per clamp state

| dataset | contrast | state L | state C | state U |
|---|---|---:|---:|---:|
| NIST lvt | GMR(M_bias v1, C0_refit) | 0.806 [0.758, 0.864] 48/53 | 0.797 [0.749, 0.855] 48/53 | 0.805 [0.757, 0.863] 48/53 |
|  | GMR(B3_d3, C0_refit) | 0.822 [0.768, 0.885] 47/53 | 0.828 [0.773, 0.893] 47/53 | 0.821 [0.768, 0.885] 47/53 |
|  | GMR(M_bias v1, B3_d3) | 0.981 [0.961, 1.000] 27/53 | 0.962 [0.931, 0.990] 30/53 | 0.981 [0.961, 1.000] 28/53 |
| NIST RT card | GMR(M_bias v1, C0_refit) | 0.797 [0.748, 0.859] 77/87 | 0.814 [0.766, 0.876] 77/87 | 0.798 [0.749, 0.860] 77/87 |
|  | GMR(B3_d3, C0_refit) | 0.835 [0.782, 0.900] 77/87 | 0.848 [0.795, 0.915] 77/87 | 0.835 [0.782, 0.901] 77/87 |
|  | GMR(M_bias v1, B3_d3) | 0.955 [0.926, 0.985] 55/87 | 0.960 [0.931, 0.989] 54/87 | 0.956 [0.927, 0.985] 54/87 |

- Stored cards and corrections rescored (no refit); state L equals the stored I8-1 / I9-1 values. Device bootstrap 10,000, seed 0, 95% percentile CI; wins = devices with the first arm lower.


### I20-1 headline: NIST DEV arms, stored (state L) vs rerun in state C, and the selected g v2 recipe

| dataset (devices) | arm | old (L): all / comb / median | new (C): all / comb / median | GMR vs C0_refit, old | GMR vs C0_refit, new |
|---|---|---:|---:|---:|---:|
| NIST lvt (53) | start (not refit) | 0.0411 / 0.0440 / 0.0181 | 0.0418 / 0.0446 / 0.0201 |  |  |
|  | C0_refit | 0.0400 / 0.0428 / 0.0178 | 0.0399 / 0.0427 / 0.0178 |  |  |
|  | M_bias v1 | 0.0315 / 0.0337 / 0.0164 | 0.0312 / 0.0334 / 0.0165 | 0.806 [0.758, 0.864] 48/53 | 0.798 [0.750, 0.857] 50/53 |
|  | selected g v2 (new) | - | 0.0318 / 0.0340 / 0.0165 | - | 0.803 [0.757, 0.857] 50/53 |
|  | GMR(selected g v2, M_bias v1) |  |  |  | 1.006 [0.986, 1.024] 18/53 |
| NIST RT card (87) | start (not refit) | 0.0576 / 0.0603 / 0.0366 | 0.0614 / 0.0642 / 0.0387 |  |  |
|  | C0_refit | 0.0555 / 0.0580 / 0.0357 | 0.0567 / 0.0593 / 0.0360 |  |  |
|  | M_bias v1 | 0.0455 / 0.0480 / 0.0280 | 0.0466 / 0.0491 / 0.0272 | 0.797 [0.748, 0.859] 77/87 | 0.809 [0.759, 0.873] 78/87 |
|  | selected g v2 (new) | - | 0.0466 / 0.0491 / 0.0275 | - | 0.811 [0.761, 0.871] 77/87 |
|  | GMR(selected g v2, M_bias v1) |  |  |  | 1.002 [0.989, 1.015] 40/87 |

- old: stored I8-1 / I9-1 rows (state L). new: rerun in state C (starting cards re-simulated in C, not refit; C0_refit and M_bias v1 retrained / refit with the I8-1 / I9-1 recipe; selected g v2 = symmetric inputs, EPS 0.01 V, VDD 1.8 V, lambda* per device from the 6-block nested selection, 5 seeds, LODO, held-out refit with g frozen).
- RRMS: all-device mean over the finite-start devices / combined (mean of nMOS and pMOS family means) / median; strict paper RRMS with the frozen measured inclusion. GMR: device bootstrap 10,000, 95% CI, wins = devices with the first arm lower / devices paired.


### I20-1 pre-registered outcome per NIST dataset (state C)

| dataset | GMR(selected g v2, C0_refit) | primary label | GMR(selected g v2, M_bias v1) | secondary label | failed rows | lambda* (devices) | full-data lambda* |
|---|---:|---|---:|---|---:|---|---:|
| NIST lvt | 0.803 [0.757, 0.857] | gain over C0_refit replicated | 1.006 [0.986, 1.024] | non-inferior to M_bias v1 | 0 | 0 V x36, 1 V x17 | 0 V |
| NIST RT card | 0.811 [0.761, 0.871] | gain over C0_refit replicated | 1.002 [0.989, 1.015] | non-inferior to M_bias v1 | 0 | 0 V x87 | 0 V |

- Validity: valid; V7 grids True, V8 unclamped starting records identical True.
- Primary label (I17-2 rule): gain over C0_refit replicated iff CI upper < 1 and 0 failed rows; worse iff CI lower > 1; else inconclusive. Secondary: non-inferior to M_bias v1 iff CI upper <= 1.05.
- lambda*: per device = lambda* of its selection block (6 blocks; I15-2 inner rule inside each block); full: selection over all finite DEV devices (for TEST).


### I20-1 dataset-level gate on NIST: S3_D values and decisions

| dataset | committed (a755e18) | v1 recipe, state C | g v2 recipe | frozen rule | realised GMR(sel., C0_refit) | sign = realised |
|---|---:|---:|---:|---|---:|---|
| NIST lvt | -0.1027 apply | -0.1172 apply | -0.1163 apply | apply | 0.803 (gain) | v1 yes, g v2 yes |
| NIST RT card | -0.0689 apply | -0.0730 apply | -0.0706 apply | apply | 0.811 (gain) | v1 yes, g v2 yes |

- S3_D (I19-1 definition): mean over the finite DEV records of ln(RRMS(start x exp(g_-k)) / RRMS(start)), g_-k trained on all other devices (seed 0, 500 steps), no refit; apply iff S3_D < 0. committed: I19-1 Part P (a755e18), v1 recipe, state-L records. v1 state C: same recipe on the state-C records. g v2: selected recipe with lambda*_k.
- Frozen-method rule (pre-registered in I20-1, open item 9): the v1-recipe S3_D on the state-C records decides whether the selected g v2 correction is applied.
- Sign check against the realised GMR(selected g v2, C0_refit): v1 state C 2/2, g v2 recipe 2/2 (retrospective DEV rows before this iteration: 4/4 and 2/4).
- Amendment of the committed TEST gate decisions: not needed (no decision differs from the committed one).


### I20-1 per family (state C)

| dataset | family (devices) | C0_refit | M_bias v1 | selected g v2 | GMR(v1, C0_refit) | GMR(selected, C0_refit) |
|---|---|---:|---:|---:|---:|---:|
| NIST lvt | nmos (29) | 0.0130 | 0.0104 | 0.0105 | 0.800 [0.773, 0.827] 29 | 0.808 [0.783, 0.834] 29 |
|  | pmos (24) | 0.0725 | 0.0564 | 0.0575 | 0.797 [0.697, 0.930] 21 | 0.798 [0.705, 0.919] 21 |
| NIST RT card | nmos (48) | 0.0341 | 0.0250 | 0.0250 | 0.705 [0.678, 0.734] 47 | 0.708 [0.681, 0.738] 47 |
|  | pmos (39) | 0.0844 | 0.0732 | 0.0732 | 0.960 [0.855, 1.106] 31 | 0.958 [0.858, 1.098] 30 |

- State C rows; family means over the finite-start devices; GMR device bootstrap within the family; wins = devices with the arm below C0_refit.


### I20-1 nested lambda selection (6 blocks and full data)

| dataset | selection | devices scored | training devices | lambda 0 V | lambda 1 V | lambda 2 V | lambda 5 V | lambda 10 V | lambda* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NIST lvt | block 0 | 9 | 44 | -3.8298 | -3.8035 | -3.8041 | -3.7925 | -3.7825 | 0 |
|  | block 1 | 9 | 44 | -3.8288 | -3.7951 | -3.7919 | -3.7966 | -3.7919 | 0 |
|  | block 2 | 9 | 44 | -3.8715 | -3.8650 | -3.8313 | -3.7947 | -3.7604 | 0 |
|  | block 3 | 9 | 44 | -3.8467 | -3.8368 | -3.8157 | -3.7972 | -3.7461 | 0 |
|  | block 4 | 9 | 44 | -3.8478 | -3.8511 | -3.8210 | -3.7809 | -3.7542 | 1 |
|  | block 5 | 8 | 45 | -3.8887 | -3.9011 | -3.8491 | -3.8032 | -3.8040 | 1 |
|  | full | 53 | 53 | -3.8891 | -3.8473 | -3.8282 | -3.7848 | -3.7643 | 0 |
| NIST RT card | block 0 | 15 | 72 | -3.4889 | -3.4317 | -3.4306 | -3.4283 | -3.4311 | 0 |
|  | block 1 | 15 | 72 | -3.4384 | -3.4059 | -3.3972 | -3.3935 | -3.3833 | 0 |
|  | block 2 | 15 | 72 | -3.4684 | -3.4084 | -3.4034 | -3.3943 | -3.3910 | 0 |
|  | block 3 | 14 | 73 | -3.4704 | -3.4154 | -3.3926 | -3.3819 | -3.3768 | 0 |
|  | block 4 | 14 | 73 | -3.4587 | -3.3912 | -3.3742 | -3.3550 | -3.3530 | 0 |
|  | block 5 | 14 | 73 | -3.4921 | -3.4280 | -3.4046 | -3.4028 | -3.3977 | 0 |
|  | full | 87 | 87 | -3.4835 | -3.3981 | -3.3943 | -3.3859 | -3.3659 | 0 |

- Inner score = mean ln RRMS over the inner held-out devices (one member, seed 0, per inner fold and lambda; inner held-out refit with g frozen, state C). lambda* = argmin (ties to the larger lambda). Block j: devices scored with lambda*_j; its selection uses only the devices outside block j ('training devices'). full: all finite DEV devices.

