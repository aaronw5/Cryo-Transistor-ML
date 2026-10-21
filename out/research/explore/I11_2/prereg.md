# I11-2 pre-registration: shared bias-only correction with a temperature input, held-out temperature

Status: IN FORCE (2026-10-21). Code commit 02df93c. The machine-readable file is `prereg.json`; this file restates it.

## Question

Does a shared bias-only correction g(is_pmos, |Vgs|/VDD, |Vds|/VDD, T), trained on starting-card records of other
devices at other temperatures, lower the held-out RRMS of a device at a temperature absent from training, compared
with the refit card without g (C0_refit) and with the same correction without the temperature input?

## Data (DEV only, decision D1)

- IHP SG13G2 (I5-1): 18 DEV DUTs (LV 10, HV 8) at 233.15 / 300.15 / 343.15 / 398.15 K, stored starting-card records.
- TU Delft 40 nm (I6-1): 45 DEV devices in 7 geometry groups at 4.13 K and 295.0 K, stored starting-card records.
- TEST devices of both datasets are not read.

## Protocol

- Held-out record = (device, temperature). Training excludes every record of the held-out unit (IHP DUT at all
  temperatures; TU Delft geometry group at both temperatures) and every record at the held-out temperature (except
  the reference arm M_bias_in). The code raises on a violation.
- Held-out card: the device's own stored starting card at the held-out temperature, refit with g frozen in the loop
  (polish, max_nfev 400, irls 2, diff_step 0.01, base box); strict paper RRMS of I_card x exp(g).
- Temperature input t = ln(T / 300 K), measured temperature, standardized.

## Arms

| arm | definition | datasets |
|---|---|---|
| C0_refit | stored I5-1 / I6-1 rows (no g) | both |
| M_bias_T | 4-16-16-1 MLP on (is_pmos, u, v, t), M_bias recipe, other units x other temperatures | IHP |
| M_bias_pool | M_bias (no t) on the same records | IHP |
| M_bias_near | stored I5-1 fold M_bias at the nearest training temperature in ln T (233->300, 300->343, 343->300, 398->343) | IHP |
| M_bias_X | stored I6-1 fold M_bias at the other temperature (= pool = near) | TU Delft |
| M_bias_in | M_bias_T trained on all temperatures, held-out unit excluded (reference) | both |
| M_bias_sameT | stored I5-1 / I6-1 M_bias (reference) | both |
| lin_T, lin_pool | linear g per polarity with / without t (secondary) | IHP |

## Primary contrasts (Bonferroni over 8, two-sided percentile level 0.99375)

IHP (device bootstrap over 18 DUTs; per DUT the mean log ratio over the stratum's two temperatures):
C1-C3 interpolation (300, 343 K) GMR(M_bias_T, X) for X = C0_refit, M_bias_pool, M_bias_near;
C4-C6 extrapolation (233, 398 K), same three.
TU Delft (cluster bootstrap over 7 groups): C7 GMR(M_bias_X, C0_refit) with 4 K held out; C8 with 295 K held out.

## Rule

- IHP stratum label: supported (C vs C0_refit, vs pool and vs near all with CI upper < 1); transfer without
  temperature input (vs C0_refit lower, not both others lower); loss (vs C0_refit CI lower > 1); no gain (otherwise).
- TU Delft label: transfer gain / transfer loss / inconclusive; reported, not in the action (limitation L1).
- Validity: V1 C0_refit reruns of 6 stored rows (relative difference <= 1e-9); V2 M_bias_near rows at 233 / 343 K
  equal the stored X300 rows (<= 1e-9); V3 at most 2 failed records per primary arm per dataset; V5 retrained fold
  corrections equal the stored ones (<= 1e-9); V4 TU Delft timeouts reported.
- Decision: invalid -> stop, fix and re-register; some stratum supported -> continue (pre-register on a second
  multi-temperature dataset); else some stratum "transfer without temperature input" -> continue, reduced (pooled g
  without temperature input; no temperature-input work); else -> stop (no cross-temperature work on the shared
  bias-only correction).

## Secondary prediction (committed before any refit)

Order: stats -> commit_prediction -> lodo. Per IHP held-out temperature: S1_loto (I7-1 S1 with the shared function
estimated without the scored record's temperature and DUT) and S3_loto (post-hoc gain of a 1-seed 500-step g on
inner leave-one-temperature-out x 2 DUT-part folds) for M_bias_T and M_bias_pool. Predictions: gain iff S3 < 0;
M_bias_pool gain iff the frozen I7-1 rule on S1_loto gives y < 0. 12 predictions, reported as k of 12. TU Delft: no
prediction (one training temperature).

## Decisions

D1 DEV only (TEST reading cannot be undone and would void the I8-2 TEST predictions of 6 dataset-temperatures); D2
crossed exclusion; D3 stored folds (IHP DUT within class, TU Delft geometry group); D4 t = ln(T/300 K); D5 mlp16
primary, lin pair secondary, gam not added; D6 reuse stored nearest-temperature fold corrections; D7 no TU Delft
M_bias_T; D8 Bonferroni over 8, TU Delft cluster bootstrap primary; D9 no card clamp (no card shift; refits inside
the stored base box, 0 failed records in I5-1 / I6-1); D10 equal weight per record. Reasons in `prereg.json`.

## Limitations

L1 TU Delft has 2 temperatures (transfer only). L2 IHP spans 233-398 K and its same-temperature M_bias was above
C0_refit at every temperature (I5-1). L3 18 IHP DUTs. L4 stored C0_refit / M_bias rows were seen before this file.

## Compute and queue

Smoke (2 workers, reduced): 86 s end to end. Estimate on 6 workers: about 50-75 min. Queue: I11_2_validity,
I11_2_stats, I11_2_commit_prediction, I11_2_lodo_ihp, I11_2_lodo_tud, I11_2_decide.
