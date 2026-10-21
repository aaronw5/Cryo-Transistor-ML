# I8-1 pre-registration (DRAFT, not in force): NIST mds2-2997 sky130 at 4 K, prospective DEV test

Machine-readable version: `prereg.json` (same content). Status: draft for review. The open questions at the end
must be closed before this file is committed as the pre-registration. No card has been fit to these data, no
correction trained, no S statistic computed.

## Questions

- P_a: does the I7-1 frozen rule (y_D = 0.007619 - 0.485184 S1_lodo_D; gain iff S1_lodo_D > 0.0157), applied to
  S1_lodo from the DEV training folds, predict the sign of ln GMR(M_bias, C0_refit)?
- P_b: does M_bias (I3-2 recipe) lower held-out error relative to C0_refit (device-bootstrap CI upper bound of
  GMR < 1)?
- P_c (secondary): does sign(S3_D) predict the sign of ln GMR(M_bias, C0_refit)?

## Data

| item | value |
|---|---|
| source | NIST mds2-2997 v1.1.2, `MPW-5 Test Tile.zip`, 39,471,626 bytes, sha256 `3a2cd00d...932a03e3` (equal to the NIST metadata value) |
| measurement | 4 K chuck temperature, Agilent 4156B, source grounded; drain compliance 0.1 A |
| families | nfet_01v8_lvt (nthin_lvt, 56 devices, W 0.42-7 um, L 0.15-8 um); pfet_01v8_lvt (pthin_lvt, 40 devices, W 0.42-7 um, L 0.35-20 um); one device per W/L |
| excluded | nfet_01v8_lvt_W1p65_L0p25_m23p4 (readme "23_4 Issue"; gate-control ratio 12.1) |
| curves | Vb = 0: Id-Vg at \|Vd\| 0.1, 1.8 V (181 points); Id-Vd at \|Vg\| 0-1.8 V in 0.3 V steps (7 curves, 91 points) |
| inclusion (draft) | measured-only pdk2 rule (trim 1e-10 A; 3e-7 A Id-Vd, 2e-7 A Id-Vg); DEV nMOS 198 curves (6 per device), pMOS 102 (2-6) |
| split | seed 20261021; strata family x bin coverage; blocks of 5 sorted by (L, W), 3 DEV / 2 TEST; DEV 57 (nMOS 33, pMOS 24), TEST 38 (22, 16) |

## Model, cards, arms

- Cards: the 77 K paper cards (nfet_01v8_lvt original, pfet_01v8_lvt upstream update), ngspice-41 arm64,
  `spice_pdk.simulate_pdk` deck.
- The 77 K cards hold only the 18 paper-device bins: 21 / 56 nMOS and 22 / 40 pMOS geometries are inside a bin.
  Draft: forced nearest bin (log L, log W distance; largest DEV distance factor 3.0) for the others.
- Deck temperature (draft): 77 K (= card tnom).
- Starting cards: 43-parameter set, cold_start round-one box around the bin values (draft), polish from the stock
  card, max_nfev 3000, irls 3.
- Arms: C0 (starting card, no refit; reported), C0_refit (held-out refit, max_nfev 400, irls 2), M_bias (I3-2
  recipe, VDD 1.8 V, 5 seeds, 2000 steps, refit in the loop), T1_inloop.
- Folds: leave-one-device-out over the 57 DEV devices, both polarities in the training pool.
- Metric: strict paper RRMS, frozen inclusion, fresh NGSpice simulation of every refit card.

## Order of operations

1. Close the open questions; commit this file alone and push. No fit before this commit.
2. `I8_1_start`: starting cards of the 57 DEV devices -> `start/records`.
3. `I8_1_stats`: S1_lodo, S1_pooled, S2 from `start/records` only; `I8_1_s3`: S3 from `start/records` only;
   write `prediction.json` (values, predicted signs, code commit, sha256 of the record files).
4. Commit `prediction.json` alone and push. The lodo mode refuses to start unless `prediction.json` is tracked,
   unmodified, and in an ancestor commit of HEAD; it records that commit in `results.json`.
5. `I8_1_lodo`: C0, C0_refit, M_bias, T1_inloop.
6. `I8_1_decide`: P_a, P_b, P_c, validity; recomputes S1_lodo_D and checks it against `prediction.json`.

## Criteria

| item | rule |
|---|---|
| P_a | correct iff sign(y_D) = sign(0.007619 - 0.485184 S1_lodo_D) |
| P_b supported | R1 CI upper of GMR(M_bias, C0_refit) < 1; R2 M_bias mean <= T1_inloop mean + 0.005; R3 0 failed |
| P_b other labels | gain, inconclusive (R1 only); not supported (CI contains 1); harm (CI lower > 1) |
| P_c | correct iff sign(y_D) = sign(S3_D) |
| V1 | >= 90% of DEV devices with a finite starting RRMS and >= 2 included curves |
| V2 | <= 10% failed devices per arm and >= 45 paired devices |
| V3 | `prediction.json` committed before any lodo output; else P_a, P_c "not prospective" |
| V4 | 0 simulator timeouts, or timed-out devices dropped from every arm |

## Compute estimate

Smoke: 0.023-0.025 s per 9-sweep simulation (stock card; 77 K and 4.2 K decks). Scaled from I6-1 / I7-1 job
times: starting fits about 1 CPU-h, S1 statistics about 10 s, S3 1.3-5 CPU-h (1596 distinct inner training
sets), LODO 1.8-2.6 CPU-h; total 4-9 CPU-h, about 45-90 min on 6 workers. Not queued.

## Open questions

| id | question | draft choice |
|---|---|---|
| Q1 | the nMOS 77 K card is nfet_01v8_lvt, not nfet_01v8 | nthin_lvt (56); nthin (80) needs a room-temperature card |
| Q2 | 32 of 57 DEV geometries outside the 77 K card bins | forced nearest bin; alternatives: covered only (DEV 25), full room-temperature lvt cards |
| Q3 | deck temperature | 77 K (card tnom); alternative 4.2 K (stock subthreshold at the GMIN floor) |
| Q4 | inclusion rule | measured-only pdk2 rule; alternative rrmsCalc rule with pMOS inclusion frozen from the C0 start card |
| Q5 | curve set | Vb = 0 only (9 curves) |
| Q6 | fit box | cold_start round-one rule; alternative I6-1 rule |
| Q7 | records for S1_lodo | starting-card records of training devices (frozen I7-1 definition), not C0_refit cards |
| Q8 | baselines B1, B2, B3_d3 | not included (about +1 CPU-h) |
| Q9 | VDD for M_bias inputs | 1.8 V |
| Q10 | one dataset gives one P_a outcome | per-family values reported, not deciding |
| Q11 | I8-2 gate G (M_bias iff S3_D < 0) as a secondary prediction | not included; no extra compute |
