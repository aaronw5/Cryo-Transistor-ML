# I10-1 pre-registration (IN FORCE): 4 K I-V from room-temperature data

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
version, committed alone (2026-10-21). Done before it: feasibility check (a108203), code and 21 unit tests, NIST DEV
dataset (`data/manifest.json`), stock-card scan of the 77 checked NIST devices (finiteness and timing, no RRMS), card
checks, and the reduced-setting smoke of every mode on 6 NIST and 7 TU Delft DEV devices (`smoke_runs.json`, commit
eeaed11; not used for any decision except the addition of K2, decision D14).

## Question

Can a device's 4 K I-V be predicted without any of its 4 K data, from its room-temperature (RT) data and what
training devices with both temperatures teach?

## Data and split

| item | value |
|---|---|
| DEV, NIST | ETest nfet_g5v0d10v5 (36) / pfet_g5v0d10v5 (31), 4 K paired by module/pad ID with google sky130 raw-data RT files (1f7214f); 61 geometry groups; 30 inside a bin, 37 bin-stretched |
| NIST exclusions | 4 fail device checks (gate control < 100 at 4 K or RT); 10 (L 0.30-0.40 um) whose stretched-bin stock card fails BSIM4 parameter checks at 30 C (D10) |
| NIST curves | 4 K Vb = 0: Id-Vg at \|Vd\| 0.1, 3.3 V (331 points), Id-Vd at \|Vg\| 0-3.3 V in 0.55 V steps (133 points); RT inside \|V\| <= 3.3 V: Id-Vg at 0.1 V, Id-Vd at \|Vg\| 0-3 V (67 points) |
| DEV, TU Delft | I6-1 DEV, 45 devices in 7 geometry groups; stored I6-1 295 K and 4.13 K cards (reproduce their records, difference 0) |
| TEST | 25 ETest 1.8 V core devices (5 types x 5); locked; no I10-1 mode reads them |
| TEST prior exposure | the feasibility check read the TEST files once for descriptive numbers and simulated the stock card on 2 TEST devices; no fit, no I10-1 arm scored |

## Fits and arms

Cards: open_pdks a918dc7 sky130 tt per type, single-bin card (equal to the open_pdks library), 43-parameter set,
I8-1 box. NIST deck 30 C (tnom); TU Delft deck = stage temperature.

| name | definition |
|---|---|
| C0_RT | RT card: NIST polish from stock on the RT curves (30 C, nfev 3000, irls 3); TU Delft stored 295 K card |
| C4 | 4 K card, warm start from C0_RT, at the RT deck (training shifts and R_oracle) |
| A0 | C0_RT at the 4 K deck (NIST -269.15 C, BSIM4 temperature equations; TU Delft 4.13 K, kT/q only) |
| A0_rtdeck | C0_RT at the RT deck (no temperature change) |
| A1 | C0_RT + median over training devices of (z_C4 - z_C0_RT) in local coordinates, clipped |
| A2 | A1 x exp(g), g = M_bias trained per fold on nested out-of-fold A1 records (both polarities), no held-out refit |
| A3 | A1 + ridge term on (log L, log W, 3 RT descriptors, RT card coordinates), lambda per coordinate by inner LOO |
| A1_L | secondary: A1 + ridge term on (log L, log W) |
| R_pop | median training C4 coordinates on the held-out device's bin |
| R_oracle | the device's own C4 (uses its 4 K data; upper reference) |
| R_stock4k, R_stock_rtdeck | stock card at the 4 K / RT deck |
| variants | V_ind (C4 from stock), V_mean, V_sub7 (7 parameters), V_deck (C4 and A1 at -269.15 C); TU Delft V_i6 (stored I6-1 4 K cards) |

Folds: leave-one-geometry-out; training within type (NIST) / polarity (TU Delft). Score: strict paper RRMS on the
held-out device's 4 K Vb = 0 curves, measured-only inclusion frozen from the measured 4 K current.
A1 rule (in advance): a device with a non-finite C0_RT or C4 record is excluded from every training set and scored
as failed in every arm; variant and nested records likewise for their own use.

## Contrasts and decision (NIST DEV)

K1 (primary) GMR(A1, A0); K2 GMR(A1, A0_rtdeck); K3 GMR(A1, R_pop); K4 GMR(A2, A1); K5 GMR(A3, A1). Paired GMR,
device bootstrap 10,000, two-sided Bonferroni level 0.99. Label lower / higher / inconclusive.

| outcome | condition | action |
|---|---|---|
| supported | K1, K2, K3 lower | continue: TEST protocol in December, confirmatory K1, K2, K3 |
| partly supported | K1 lower, K2 or K3 not lower | continue, reduced: TEST confirmatory set = K1 and the lower ones of K2, K3 |
| not supported | K1 not lower | stop: no I10-1 TEST run; TEST stays locked |
| invalid | V1 or V2 fails | stop: fix and re-register |

TEST primary arm: A1, or A2 / A3 if K4 / K5 is lower (the lower GMR). TEST configuration: the variant with the lowest
GMR(V, A1) whose 95% CI upper bound is < 1, else the primary configuration. TU Delft: same contrasts with device and
cluster CIs, reported as replication; it changes no label.

Validity: V1 >= 90% finite C0_RT and C4 records with >= 2 included 4 K curves; V2 arms A0, A0_rtdeck, A1, A2, A3,
R_pop <= 10% failed and every K contrast >= 80% paired; V3 guard on this file in start, lodo, decide; V4 timeouts
reported.

## TEST protocol (December)

25 core devices, same checks and exclusions, RT fit on the full RT range (0-1.8 V), leave-one-device-out within type
(4 training devices), arms A0, A0_rtdeck, A1, A2 (g from the other 24 core devices), A3 (pooled per polarity with
type indicators), R_pop, R_oracle, R_stock4k; confirmatory contrasts from the DEV decision with Bonferroni over that
set; device bootstrap over 25 devices; 5-type cluster CI and per-type GMRs reported. Wording: 'confirmed' iff every
confirmatory contrast is lower.

## Decisions

| id | decision | reason |
|---|---|---|
| D1 | cards fitted at the RT deck; -269.15 C convention as V_deck | I9-1 fits at 30 C worked; at -269.15 C the stock subthreshold current is at the gmin floor |
| D2 | C4 warm start from C0_RT; V_ind | local difference; independent fits of 34-37 active parameters can differ by compensating combinations |
| D3 | median in local coordinates, clipped; V_mean | robust to bound hits; multiplicative shift comparable across bins |
| D4 | all active parameters; V_sub7 | no DEV evidence yet on which parameters carry the shift |
| D5 | leave-one-geometry-out | 6 NIST geometries have 2 copies; A3 uses geometry features |
| D6 | A2 on nested out-of-fold A1 records, M_bias unchanged, no refit | g must learn a prediction residual; I9-2 found no better smaller class |
| D7 | ridge, median intercept, inner LOO lambda; A1_L | about 30 devices per type vs about 42 features; lambda = infinity equals A1 |
| D8 | R_pop with the median | same statistic as A1 |
| D9 | >= 2 included 4 K and RT curves | as I9-1 D12; excluded none |
| D10 | exclude 10 devices with non-finite stock cards at 30 C | such records stay non-finite after the fit (I8-1); TEST devices all lie in a bin |
| D11 | A1 rule in advance | I8-1 amendment A1 |
| D12 | split as recommended by the feasibility check | TEST shares tile and card source with DEV NIST; core types used by no DEV choice |
| D13 | decision on NIST; TU Delft replication | TEST is NIST sky130; TU Delft A0 is kT/q only, 7 clusters |
| D14 | K1-K5, Bonferroni 0.99; K2 added after the smoke | smoke A0 RRMS about 7 vs A0_rtdeck about 0.8; the decision should not rest on K1 alone; the rule becomes stricter |
| D15 | variant selection at 95%, at most one | the primary configuration stays unless a variant is clearly better |

## Compute and queue

Estimate on 6 workers: start NIST 30-40 min (268 fits), start TU Delft about 9 min, lodo NIST 25-35 min (61 folds x
5 M_bias seeds), lodo TU Delft 7-8 min, decide seconds; total about 75-95 min. Queue: I10_1_start_nist,
I10_1_start_tud, I10_1_lodo_nist, I10_1_lodo_tud, I10_1_decide.

## Limitations

DEV devices are thick-oxide 3.3 V devices with their own cards, TEST devices thin-oxide 1.8 V; the RT fit has 4-5
curves and a 1e-9 A floor; die identity of the RT and 4 K measurements is not documented; 37 of 67 NIST DEV devices
use a stretched bin; A0 at -269.15 C uses BSIM4 temperature equations outside their calibrated range; TEST has 4
training devices per type with 2-3 distinct L.
