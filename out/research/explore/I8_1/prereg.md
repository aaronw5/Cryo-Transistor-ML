# I8-1 pre-registration (IN FORCE): NIST mds2-2997 sky130 at 4 K, prospective DEV test

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds
this version. Written before any card fit on these data, before any correction was trained and before any S
statistic was computed. Smoke runs of the fitting modes are run after this commit (reduced settings, `smoke_*`
directories) and are not used for any decision.

## Questions

- P_a: does the I7-1 frozen rule (y_D = 0.007619 - 0.485184 S1_lodo_D; gain iff S1_lodo_D > 0.0157), applied to
  S1_lodo from the DEV training folds, predict the sign of y_D = ln GMR(M_bias, C0_refit)?
- P_b: does M_bias (I3-2 recipe) lower held-out error relative to C0_refit?
- P_c (secondary): does sign(S3_D) predict the sign of y_D?

## Decisions (closing the draft open questions)

| id | decision | reason |
|---|---|---|
| D1 | families nfet_01v8_lvt (nthin_lvt) and pfet_01v8_lvt (pthin_lvt) | device types of the 77 K paper cards |
| D2 | primary: all 57 DEV devices, explicit bin per device (native bin if covered, nearest bin otherwise); pre-registered secondary: the same statistics on the 25 bin-covered DEV devices from the same results; stretch factor per device reported | the pruned 77 K cards hold only the 18 paper-device bins; 25 devices alone give a wide CI |
| D3 | deck temperature 77 K (card tnom) | no extrapolation of the BSIM4 temperature equations to 4 K; the refit absorbs the difference. Limitation: kT/q, band gap and n_i stay at 77 K; the 4 K subthreshold slope is reproduced only through fitted parameters |
| D4 | measured-only inclusion (pdk2 rule, as I5-1 / I6-1), stored on the C0 starting-card records, used by every arm | depends on the measured current only |
| D5 | Vb = 0 curves only (9 per device) | deck body at ground; same as the I7-1 calibration datasets |
| D6 | cold_start round-one box around the showmod stock value of the bin: VTH0 x/ 3; U0, NFACTOR, VSAT, DELTA, RDSW, ETA0 x/ 100; other 36 x/ 3; log, sign kept; zero values inactive | sky130 rule of the 43-parameter fits |
| D7 | S1_lodo from the starting-card records of the training devices (frozen I7-1 definition) | the frozen rule was calibrated on it |
| D8 | B1, B2, B3_d3 (I4-1 recipes), separate job, reported only | size context for any M_bias gain |
| D9 | VDD 1.8 V for the correction inputs | largest \|V\| of every sweep |
| D10 | P_a binary dataset-level; also registered: predicted y_D, per-family (nMOS-only, pMOS-only) S1_lodo, S3 and predicted signs | one dataset gives one binary outcome |
| D11 | I8-2 gates G (S3_D < 0 -> M_bias) and G1 (frozen rule) as secondary predictions | no extra compute |

## Data and split

| item | value |
|---|---|
| source | NIST mds2-2997 v1.1.2, `MPW-5 Test Tile.zip`, sha256 equal to the NIST metadata value |
| devices | nfet_01v8_lvt 56 (W 0.42-7 um, L 0.15-8 um), pfet_01v8_lvt 40 (W 0.42-7 um, L 0.35-20 um); excluded nfet_01v8_lvt_W1p65_L0p25_m23p4 |
| curves | Id-Vg at \|Vd\| 0.1, 1.8 V (181 points); Id-Vd at \|Vg\| 0-1.8 V in 0.3 V steps (91 points) |
| split | DEV 57 (nMOS 33, pMOS 24; 25 bin-covered), TEST 38; TEST not loaded before the December TEST run |

## Simulation and fits

- ngspice-41 arm64, canonical 77 K library, corner file patched (explicit bin, parameter values), deck at 77 K,
  instance with nf = 1, 0.29-based ad/as/pd/ps, nrd/nrs, mult = m = 1 for every device.
- Stock: the bin's 43 parameters written as their showmod readback values.
- Starting cards: polish from stock, max_nfev 3000 (NGSpice runs), irls 3, diff_step 0.01.
- Arms: C0 (reported), C0_refit (max_nfev 400, irls 2), M_bias (I3-2 recipe, 5 seeds, 2000 steps, refit in the loop),
  T1_inloop; baselines B1 (screened extras), B2, B3_d3 reported only.
- Folds: leave-one-device-out over the 57 DEV devices; both polarities in the training pool.
- Metric: strict paper RRMS, frozen measured inclusion, fresh NGSpice simulation of every refit card.

## Order of operations

1. Commit this file alone and push. Then code and smoke runs.
2. `I8_1_start` -> `start/bins.json`, `start/records`, `start/summary.json`.
3. `I8_1_stats` -> `prediction.json` (S1_lodo, S1_pooled, S2, S3, predicted y_D and signs, per-family values, G /
   G1 choices, sha256 of the record files) from `start/records` only.
4. `I8_1_commit_prediction` commits `prediction.json` alone and pushes; fails if the file is missing.
5. `I8_1_lodo` refuses to start unless `prediction.json` is tracked, unmodified and in an ancestor commit of HEAD.
6. `I8_1_baselines` (same guard).
7. `I8_1_decide` checks the recomputed S1_lodo_D against `prediction.json`.

## Criteria

| item | rule |
|---|---|
| P_a | correct iff sign(y_D) = sign(0.007619 - 0.485184 S1_lodo_D) |
| P_b supported | R1 CI upper of GMR(M_bias, C0_refit) < 1; R2 M_bias mean <= T1_inloop mean + 0.005; R3 0 failed |
| P_b other labels | gain, inconclusive (R1 only); not supported (CI contains 1); harm (CI lower > 1) |
| P_c | correct iff sign(y_D) = sign(S3_D) |
| secondary | covered subset (25), \|y_D - predicted y_D\|, per-family signs, gates G / G1, baselines |
| V1 | >= 90% of DEV devices with a finite starting RRMS and >= 2 included curves |
| V2 | <= 10% failed devices per arm and >= 45 paired devices |
| V3 | `prediction.json` committed before any lodo output; else P_a, P_c "not prospective" |
| V4 | 0 simulator timeouts, or timed-out devices dropped from every arm |

## Compute estimate

About 5-7 CPU-h, 60-80 min on 6 workers (starting fits about 1.4 CPU-h, S3 1-2 CPU-h, LODO 1.4-2.2 CPU-h, baselines
about 1 CPU-h). Refined after the smoke.
