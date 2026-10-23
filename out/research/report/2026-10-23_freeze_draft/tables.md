# Tables, 2026-10-23 freeze_draft

Generated 2026-10-23 06:38 by `.venv/bin/python -I scripts/research/plot_freeze_draft.py --date 2026-10-23`.

### Method-freeze draft (proposed, not frozen): DEV evidence per measured dataset

| dataset | n (CI unit) | RRMS all-device / combined: C0_refit | RRMS all-device / combined: M_bias v1 | RRMS all-device / combined: selected g v2 | RRMS all-device / combined: gated (I19-1) | GMR vs C0_refit [95% CI] lower/n: v1 | GMR vs C0_refit [95% CI] lower/n: selected g v2 | GMR vs C0_refit [95% CI] lower/n: gated (I19-1) | GMR(selected, v1) | I19-1 decision (S3_D) | Part B deployment label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sky130 77 K (all 18, LODO) | 18 (device) | 0.1051 / 0.0984 | 0.0735 / 0.0695 | 0.0739 / 0.0700 | 0.0739 / 0.0700 | 0.745 [0.660, 0.850] 17/18 | 0.743 [0.654, 0.855] 17/18 | 0.743 [0.654, 0.855] 17/18 | 0.997 [0.962, 1.027] 7/18 | apply (-0.133) | deployable with restrictions (I17-1, lambda 1 V full-data model) |
| NIST sky130 4 K lvt (I8-1) | 53 (device) | 0.0400 / 0.0428 | 0.0315 / 0.0337 | not run | 0.0315 / 0.0337 | 0.806 [0.758, 0.864] 48/53 | - | 0.806 [0.758, 0.864] 48/53 | - | apply (-0.103) | not tested (g v2 not run; v1 element not deployable, I11-1) |
| NIST sky130 4 K, RT card (I9-1) | 87 (device) | 0.0555 / 0.0580 | 0.0455 / 0.0480 | not run | 0.0455 / 0.0480 | 0.797 [0.748, 0.859] 77/87 | - | 0.797 [0.748, 0.859] 77/87 | - | apply (-0.069) | not tested (g v2 not run; v1 element not deployable, I11-1) |
| IHP SG13G2 300 K | 18 (device) | 0.0191 / 0.0191 | 0.0203 / 0.0203 | 0.0205 / 0.0205 | 0.0191 / 0.0191 | 1.214 [1.008, 1.537] 9/18 | 1.255 [1.026, 1.600] 9/18 | 1.000 [1.000, 1.000] 0/18 | 1.034 [0.990, 1.107] 7/18 | C0_refit (+0.308) | Part B not run (PSP103 replica, I17-2) |
| IHP SG13G2 233 K | 18 (device) | 0.0273 / 0.0273 | 0.0281 / 0.0281 | not run | 0.0273 / 0.0273 | 1.106 [0.908, 1.425] 11/18 | - | 1.000 [1.000, 1.000] 0/18 | - | C0_refit (+0.187) | Part B not run (PSP103 replica) |
| IHP SG13G2 343 K | 18 (device) | 0.0190 / 0.0190 | 0.0194 / 0.0194 | not run | 0.0190 / 0.0190 | 1.181 [0.983, 1.502] 8/18 | - | 1.000 [1.000, 1.000] 0/18 | - | C0_refit (+0.367) | Part B not run (PSP103 replica) |
| IHP SG13G2 398 K | 18 (device) | 0.0197 / 0.0197 | 0.0202 / 0.0202 | not run | 0.0197 / 0.0197 | 1.272 [1.030, 1.627] 11/18 | - | 1.000 [1.000, 1.000] 0/18 | - | C0_refit (+0.367) | Part B not run (PSP103 replica) |
| TU Delft 4.13 K | 45 (7 groups) | 0.0602 / 0.0532 | 0.0534 / 0.0491 | 0.0484 / 0.0449 | 0.0602 / 0.0532 | 0.909 [0.766, 1.077] 30/45 | 0.826 [0.669, 1.014] 30/45 | 1.000 [1.000, 1.000] 0/45 | 0.908 [0.848, 0.965] 33/45 | C0_refit (+0.008) | not deployable (Q3 not run) (I18-2, lambda 10 V full-data model) |
| TU Delft 295.0 K | 45 (7 groups) | 0.0265 / 0.0251 | 0.0208 / 0.0210 | 0.0215 / 0.0218 | 0.0265 / 0.0251 | 0.811 [0.532, 1.190] 29/45 | 0.829 [0.523, 1.247] 29/45 | 1.000 [1.000, 1.000] 0/45 | 1.022 [0.975, 1.056] 22/45 | C0_refit (+0.264) | deployable with restrictions (Q3 not run) (I18-2, lambda 5 V full-data model) |

- DEV only; sky130 77 K is the all-18 LODO set (no locked TEST since the 2026-10-21 user decision). RRMS: strict paper RRMS of the held-out refit card (x exp(g) for correction arms); all-device = mean over the paired devices, combined = mean of the nMOS and pMOS family means. GMR = exp(mean ln ratio), percentile bootstrap 10000 resamples, seed 0; unit device, TU Delft 7 geometry groups. 'lower/n' = devices with a lower RRMS than the reference.
- gated (I19-1): one decision per dataset, apply the correction arm iff S3_D < 0 (S3_D from all DEV records, gated arm's recipe). Correction arm: selected g v2 on sky130, IHP 300 K and TU Delft; M_bias v1 elsewhere (g v2 not run there). Pooled exp(mean ln GMR vs C0_refit) over the 9 rows: gated 0.921, v1 0.963.
- sky130 v1 and selected rows: I13-1 state C (BSIM4 A2 clamp written in the card); TU Delft rows: I18-2 state C; NIST rows (I8-1, I9-1): own multi-analysis deck, clamp state not audited; IHP: PSP103 (no BSIM4 clamp).

