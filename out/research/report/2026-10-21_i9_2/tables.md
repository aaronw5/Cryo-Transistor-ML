# Tables, 2026-10-21 i9_2

Generated 2026-10-21 08:43 by `PYTHONPATH=src .venv/bin/python scripts/research/plot_explore_i9_2.py --date 2026-10-21` from stored files only. GMR = geometric mean ratio of held-out strict paper RRMS after the held-out card refit with g frozen in the loop; < 1 = the first class is lower.

### I9-2 function-class ablation: pre-registered labels, Q1 and Q2 (pooled over sky130 77 K, NIST 4 K, TU Delft 4.13 K, IHP 300 K)

| class | trained values (total / per member) | pooled GMR(X, mlp16) [Bonferroni CI] | label (margin 1.03) | cluster-bootstrap CI (S3) | max per-dataset point GMR (limit 1.05) | Q2 candidate | pooled GMR(X, C0_refit) [95% CI] |
|---|---:|---:|---|---:|---:|---|---:|
| const | 2 / 2 | 1.095 [0.981, 1.198] | inconclusive | [0.978, 1.218] | 1.301 (sky130 77 K) | no | 1.004 [1.000, 1.008] |
| lin | 6 / 6 | 1.065 [0.948, 1.169] | inconclusive | [0.941, 1.187] | 1.304 (sky130 77 K) | no | 0.976 [0.958, 0.995] |
| poly2 | 12 / 12 | 1.103 [1.041, 1.169] | worse | [1.033, 1.178] | 1.268 (sky130 77 K) | no | 1.011 [0.967, 1.066] |
| poly3 | 20 / 20 | 1.086 [1.029, 1.150] | worse | [1.029, 1.150] | 1.185 (sky130 77 K) | no | 0.996 [0.938, 1.072] |
| poly4 | 30 / 30 | 1.060 [1.001, 1.131] | worse | [1.000, 1.134] | 1.247 (IHP 300 K) | no | 0.972 [0.902, 1.057] |
| poly5 | 42 / 42 | 1.062 [0.978, 1.154] | inconclusive | [0.968, 1.166] | 1.496 (IHP 300 K) | no | 0.973 [0.883, 1.080] |
| gam | 66 / 66 | 0.970 [0.922, 1.030] | inconclusive | [0.920, 1.033] | 1.211 (IHP 300 K) | no | 0.889 [0.819, 0.973] |
| mlp16_s1 | 353 / 353 | 0.999 [0.982, 1.015] | equivalent | [0.981, 1.015] | 1.012 (sky130 77 K) | yes | 0.916 [0.857, 0.987] |
| mlp8 | 565 / 113 | 0.986 [0.966, 1.006] | non-inferior | [0.965, 1.007] | 0.989 (IHP 300 K) | yes | 0.904 [0.850, 0.966] |
| gp | 2006 / 2006 | 1.483 [1.319, 1.684] | worse | [1.237, 1.821] | 2.609 (TU Delft 4.13 K) | no | 1.367 [1.245, 1.505] |
| mlp32 | 6085 / 1217 | 1.003 [0.987, 1.019] | equivalent | [0.988, 1.019] | 1.011 (sky130 77 K) | no | 0.919 [0.864, 0.986] |
| mlp16_rep (replicate) | 1765 / 353 | 0.997 [0.982, 1.013] | (reference noise) |  | 1.010 |  | 0.914 [0.855, 0.984] |
| mlp16 = M_bias (reference) | 1765 / 353 | 1 |  |  |  |  | 0.917 [0.859, 0.986] |

- Pre-registration out/research/explore/I9_2/prereg.json in force from commit 02c6996 (code dea00cc); decision out/research/explore/I9_2/decision.json. Every value is recomputed here from the per-device rows and checked against decision.json.
- Pooled GMR = exp(mean over the 4 datasets of the per-dataset mean ln(RRMS_X / RRMS_mlp16)) over held-out devices (strict paper RRMS after the held-out card refit with g frozen in the loop); equal dataset weight; CI from 20000 stratified device resamples at level 1 - 0.05/11 (tails 0.227%). Cluster CI: TU Delft geometry groups resampled.
- Q1: class-dependent (poly2, poly3, poly4, gp 'worse'). Q2: candidates mlp16_s1, mlp8; smallest matching class mlp16_s1 (353 trained values); beats with fewer parameters: no. Replicate notes: none (no 'beats' / 'worse' class lies inside the replicate CI).
- Q2 also needs point GMR(X, mlp16) <= 1.05 on every dataset and no more failed devices than mlp16 (NIST 4 K: 4 records with non-finite starting points scored as failed in every class).


### I9-2 per dataset: mean held-out RRMS and GMR against C0_refit and mlp16

| class | sky130 77 K (12 devices) | NIST 4 K (53 devices) | TU Delft 4.13 K (45 devices) | IHP 300 K (18 devices) |
|---|---:|---:|---:|---:|
| C0_refit | 0.1072 | 0.0400 | 0.0575 | 0.0191 |
| mlp16 | 0.0806 | 0.769 | 0.0315 | 0.806 | 0.0533 | 0.939 | 0.0203 | 1.214 |
| const | 0.1072 | 1.000 | 1.301* | 0.0399 | 1.001 | 1.242* | 0.0582 | 1.015 | 1.081* | 0.0191 | 1.000 | 0.824* |
| lin | 0.1073 | 1.002 | 1.304* | 0.0400 | 0.998 | 1.238* | 0.0542 | 0.922 | 0.983 | 0.0192 | 0.983 | 0.809* |
| poly2 | 0.1034 | 0.975 | 1.268* | 0.0354 | 0.939 | 1.165* | 0.0545 | 0.949 | 1.011 | 0.0203 | 1.201 | 0.989 |
| poly3 | 0.0956 | 0.911 | 1.185* | 0.0316 | 0.822 | 1.019 | 0.0539 | 0.955 | 1.018 | 0.0214 | 1.375 | 1.133* |
| poly4 | 0.0864 | 0.846 | 1.101* | 0.0279 | 0.734 | 0.911* | 0.0530 | 0.948 | 1.011 | 0.0228 | 1.514 | 1.247* |
| poly5 | 0.0811 | 0.787 | 1.024 | 0.0267 | 0.691 | 0.857* | 0.0518 | 0.908 | 0.967 | 0.0246 | 1.816 | 1.496* |
| gam | 0.0771 | 0.727 | 0.946* | 0.0254 | 0.670 | 0.831* | 0.0503 | 0.874 | 0.931* | 0.0225 | 1.471 | 1.211* |
| mlp16_s1 | 0.0808 | 0.778 | 1.012 | 0.0313 | 0.809 | 1.004 | 0.0522 | 0.931 | 0.992 | 0.0200 | 1.198 | 0.987 |
| mlp8 | 0.0805 | 0.759 | 0.988 | 0.0311 | 0.797 | 0.989 | 0.0525 | 0.919 | 0.979* | 0.0201 | 1.202 | 0.989 |
| gp | 0.1317 | 1.168 | 1.520* | 0.0463 | 0.844 | 1.047 | 0.1615 | 2.499 | 2.609* | 0.0210 | 1.416 | 1.166 |
| mlp32 | 0.0814 | 0.777 | 1.011 | 0.0315 | 0.807 | 1.001 | 0.0528 | 0.937 | 0.998 | 0.0202 | 1.215 | 1.001 |
| mlp16_rep | 0.0805 | 0.771 | 1.003 | 0.0310 | 0.792 | 0.982* | 0.0529 | 0.931 | 0.992 | 0.0201 | 1.226 | 1.010 |

- Mean RRMS over the finite held-out devices; GMR(X, C0_refit) and GMR(X, mlp16) per dataset with device-bootstrap 95% CIs in decision.json (secondary S2, descriptive); * = the 95% CI of GMR(X, mlp16) excludes 1. NIST 4 K: 53 of 57 devices scored (4 failed in every class). TU Delft CIs: device bootstrap (the cluster bootstrap is in table 01).
- Stored rows: C0_refit, mlp16, poly3 on all datasets; on sky130 77 K also poly2, gp (I4-1), mlp8, mlp32, mlp16_rep (I3-2 F2). Every other row is new in I9-2.


### I9-2 learning curve (Q3, secondary): held-out GMR against the number of training devices

| dataset | class | n = 2 | n = 4 | n = 8 | n = all | vs all, n = 2 | vs all, n = 4 | vs all, n = 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TU Delft 4.13 K | mlp16 | 1.004 [0.945, 1.068] | 0.966 [0.907, 1.029] | 0.933 [0.874, 0.997] | 0.939 [0.877, 1.004] | 1.070 | 1.029 | 0.994 |
| TU Delft 4.13 K | gam | 1.017 [0.951, 1.086] | 0.984 [0.919, 1.055] | 0.915 [0.844, 0.989] | 0.874 [0.795, 0.959] | 1.163 | 1.126 | 1.047 |
| TU Delft 4.13 K | mlp8 | 1.005 [0.950, 1.063] | 0.962 [0.907, 1.023] | 0.926 [0.869, 0.987] | 0.919 [0.864, 0.979] | 1.093 | 1.047 | 1.008 |
| NIST 4 K | mlp16 | 0.947 [0.866, 1.045] | 0.846 [0.780, 0.928] | 0.825 [0.772, 0.887] | 0.806 [0.758, 0.865] | 1.175 | 1.050 | 1.023 |
| NIST 4 K | gam | 0.912 [0.825, 1.014] | 0.769 [0.695, 0.857] | 0.717 [0.651, 0.793] | 0.670 [0.614, 0.734] | 1.361 | 1.148 | 1.070 |
| NIST 4 K | mlp8 | 0.936 [0.854, 1.034] | 0.838 [0.773, 0.917] | 0.814 [0.759, 0.879] | 0.797 [0.746, 0.858] | 1.174 | 1.051 | 1.022 |

- n = training devices per fold (polarity-stratified subsets, 3 per fold and n, default_rng([20261021, n, r, fold])); n = all is the main run (TU Delft 37-43 devices, NIST 52). Per held-out device the RRMS is exp(mean over the 3 subsets of ln RRMS). Descriptive (Q3); no decision.
- Refits: TU Delft 1215, NIST 1431 (2923 s and 3700 s wall). Classes chosen by decision.json before the lc jobs: mlp16 plus the two lowest pooled GMR(X, C0_refit) (gam 0.889, mlp8 0.904).


### I9-2 margin sensitivity (S4) and cluster bootstrap (S3)

| class | pooled GMR(X, mlp16) [Bonferroni CI] | label, margin 1.02 | label, margin 1.03 (pre-registered) | label, margin 1.05 | label from the cluster-bootstrap CI, margin 1.03 | cluster-bootstrap Bonferroni CI |
|---|---:|---|---|---|---|---:|
| const | 1.095 [0.981, 1.198] | inconclusive | inconclusive | inconclusive | inconclusive | [0.9780, 1.2178] |
| lin | 1.065 [0.948, 1.169] | inconclusive | inconclusive | inconclusive | inconclusive | [0.9409, 1.1875] |
| poly2 | 1.103 [1.041, 1.169] | worse | worse | worse | worse | [1.0334, 1.1785] |
| poly3 | 1.086 [1.029, 1.150] | worse | worse | worse | worse | [1.0292, 1.1502] |
| poly4 | 1.060 [1.001, 1.131] | worse | worse | worse | inconclusive | [0.9996, 1.1337] |
| poly5 | 1.062 [0.978, 1.154] | inconclusive | inconclusive | inconclusive | inconclusive | [0.9683, 1.1661] |
| gam | 0.970 [0.922, 1.030] | inconclusive | inconclusive | non-inferior | inconclusive | [0.9198, 1.0332] |
| mlp16_s1 | 0.999 [0.982, 1.015] | equivalent | equivalent | equivalent | equivalent | [0.9808, 1.0149] |
| mlp8 | 0.986 [0.966, 1.006] | non-inferior | non-inferior | equivalent | non-inferior | [0.9654, 1.0070] |
| gp | 1.483 [1.319, 1.684] | worse | worse | worse | worse | [1.2367, 1.8214] |
| mlp32 | 1.003 [0.987, 1.019] | equivalent | equivalent | equivalent | equivalent | [0.9878, 1.0191] |
| Q2 smallest matching class |  | mlp16_s1 | mlp16_s1 | mlp16_s1 |  |  |

- S4 (pre-registered): labels at margins 1.02 and 1.05 from the same Bonferroni CIs. Between 1.02 and 1.05 only gam (inconclusive -> non-inferior at 1.05) and mlp8 (non-inferior -> equivalent at 1.05) change label; Q2 is mlp16_s1 at every margin.
- S3 (pre-registered interval; the label from it is post hoc): with TU Delft geometry groups resampled as clusters, poly4 changes from 'worse' to 'inconclusive' (cluster CI lower bound 0.9996); every other class keeps its label.


### I9-2 validity checks V1-V4, deviation and compute

| dataset | V1 poly3 rerun max |RRMS - stored| | V1 max |coefficient - stored| | V2 failed devices mlp16 (range over classes) | V3 simulator timeouts (run refits) | folds | training tasks / wall | held-out refits / wall | queue job wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sky130 77 K | 0.0e+00 (2 devices) | 0.0e+00 | 0 (all classes 0-0) | 0 | 12 | 60 / 25 s | 72 / 160 s | 186 s |
| NIST 4 K | 0.0e+00 (2 devices) | 0.0e+00 | 4 (all classes 4-4) | 0 | 53 | 901 / 2388 s | 530 / 894 s | 3282 s |
| TU Delft 4.13 K | 0.0e+00 (2 devices) | 0.0e+00 | 0 (all classes 0-2) | 0 | 7 | 119 / 1148 s | 450 / 1182 s | 2330 s |
| IHP 300 K | 0.0e+00 (2 devices) | 0.0e+00 | 0 (all classes 0-0) | 0 | 18 | 306 / 152 s | 180 / 338 s | 492 s |

- V2: NIST 4 K 4 failed devices in every class (non-finite starting records, amendment A1 of I8-1); TU Delft 4.13 K gp 2 failed devices (t40_nmos_W0p12_L0p04_d152, t40_nmos_W1_L0p04_d199; every Id-Vd curve non-finite after the gp-corrected refit), every other class 0. gp is not a Q2 candidate (label 'worse').
- Validity job (V1, 31.4 s): poly3 retrained and refit on the first 2 held-out devices (sorted tags) of each dataset; tolerance 1e-9; all 4 datasets pass. V4: every stored row used equals the stored source file (checked here); mlp16_s1 = member 0 of the stored M_bias fold model.
- Deviation: the I9_2 jobs failed at 03:14 (I9_2_validity rc=1: two TU Delft held-out devices that share a fold queued the poly3 training task twice and the merge step combined them as MLP seed members). Fix a864f77; jobs requeued unchanged at 03:35. No I9-2 result existed before the fix (docs/RESEARCH_LOOP.md, 'I9-2 validity failure and requeue').
- Other jobs: replicate 1429 s, decide 1 s, lc TU Delft 2923 s, lc NIST 3701 s; total queue wall 3.99 h (pre-registered estimate without the learning curve 147 min, measured 129 min).

