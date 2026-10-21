# I9-1 pre-registration (DRAFT, not in force): NIST mds2-2997 sky130 4 K, nfet_01v8 / pfet_01v8, room-temperature starting cards

Machine-readable version: `prereg.json` (authoritative if the two differ). Status DRAFT: the open questions below
are closed by the coordinator before the file is put in force; the lodo and baselines modes refuse to start while
the status starts with DRAFT. Done before this draft: card provenance and checks (`cards.json`), dataset
(`data/manifest.json`), DEV / TEST split (`split.json`), stock-card scan of the 87 DEV devices (finiteness and
timing, no RRMS) and the reduced-setting smoke of every mode on 5 DEV devices (`smoke_runs.json`, not used for any
decision).

## Questions

- P_a: does the I7-1 frozen rule (y_D = 0.007619 - 0.485184 S1_lodo_D; gain iff S1_lodo_D > 0.0157), applied to
  S1_lodo from the DEV training folds, predict the sign of y_D = ln GMR(M_bias, C0_refit)?
- P_b: does M_bias (I3-2 recipe) lower held-out error relative to C0_refit? Supported iff the CI upper bound of
  GMR(M_bias, C0_refit) < 1 (R1), M_bias mean <= T1_inloop mean + 0.005 (R2), 0 failed M_bias devices (R3).
- P_c (secondary): does sign(S3_D) predict the sign of y_D?
- S_B3 (pre-registered secondary): GMR(M_bias, B3_d3) with CI; reading 'M_bias lower' iff the CI upper bound < 1.
  I8-1: 0.981 [0.961, 1.000]; B3_d3 reached 91% of the M_bias log gain.
- S_transfer (pre-registered secondary): arm M_bias_lvt, M_bias trained once on the 53 finite I8-1 DEV starting
  records (lvt families, 77 K cards), applied with the held-out refit to these devices; GMR vs C0_refit and vs
  M_bias with CIs.

## Data, split, cards

| item | value |
|---|---|
| families | nfet_01v8 (nthin, 80 devices, W 0.36-7 um, L 0.15-20 um), pfet_01v8 (pthin, 71 devices with files, W 0.42-7, L 0.15-20); disjoint from I8-1 |
| curves | Vb = 0, 9 per device (I8-1 set) |
| exclusion (D12, draft) | 6 pMOS with < 2 measured-rule included curves (L 8-20 um; Id-Vd at \|Vg\| <= 1.2 V below 1e-12 A) |
| split | I8-1 procedure, seed 20261022: DEV 87 (nMOS 48, pMOS 39), TEST 58 (32, 26); TEST not opened |
| included curves, DEV | nMOS 288 (6 per device), pMOS 161 (2-6 per device) |
| cards | open_pdks a918dc7 (Volare build on disk) sky130 tt: nfet_01v8 tt.pm3 (180 bins), pfet_01v8 tt.corner + tt.pm3 (108 bins), mismatch corners, lod.spice; tnom 30 C; sha256 in `cards.json`; no download |
| bin coverage | all 151 geometries and all 87 DEV devices inside a bin (native selection); no forced bins |
| card handling | one assignment per `+` line (keeps l / w / p binning terms when a parameter is written); per run a card with the device's bin only; equal to the full card (difference 0) and to the unmodified open_pdks library (<= 1.7e-34 A) on 5 smoke devices |
| deck temperature (D3, draft) | 30 C (cards' tnom) |

## Pipeline

As I8-1 (`prereg.json` sections cards, arms, evaluation, statistics), with amendment A1 built in: a DEV device
whose starting record has a non-finite simulated point is excluded from every training set and S fold, not refit,
and scored NaN in every arm. Arms C0, C0_refit, M_bias (primary), T1_inloop, M_bias_lvt (secondary); baselines B1,
B2, B3_d3. Validity: V1 >= 90% finite starts with >= 2 included curves; V2 <= 10% failed per arm and >= 70 paired
devices; V3 prediction.json committed before any lodo output; V4 timeouts reported, arm timeouts dropped.

## Limitations

- Deck at 30 C: kT/q 26 mV, smallest model subthreshold swing n x 60 mV/decade; measured smallest swing of the smoke
  devices 6-46 mV/decade (stock card at 30 C: 81-135). I8-1 had the same limitation with n x 15 mV/decade.
- The I7-1 rule was calibrated on 11 stored datasets with S1_lodo -0.379 to 0.504 (checked on 20 simulated
  datasets and I8-1, 0.502); S1_lodo_D from room-temperature starting cards may lie outside that range.
- Regular-Vt pMOS at 4 K: few included curves per device (mean 4.1), 6 devices excluded.

## Compute estimate

Scaled from I8-1 job times (6 workers, similar 0.025 s per simulation): start about 11 min, stats about 75 min
(3741 S3 pair fits), lodo about 44 min, baselines about 11 min; total about 2.3 h wall, about 14 CPU-h. Not queued.

## Open questions

| id | question |
|---|---|
| Q1 | deck temperature 30 C (draft, the cards' tnom) or 27 C (brief)? 27 C changes stock currents by 0.0-0.8% at \|Vg\| = \|Vd\| = 1.8 V and 0.1-10% at \|Vg\| = 0.9 V. 4.2 K options recorded in the smoke, not scored |
| Q2 | exclude the 6 low-current pMOS before the split (draft) or keep them as V1 failures (DEV 91, TEST 60)? |
| Q3 | keep the full S3 pair design (about 75 min) or subsample inner devices? |
| Q4 | keep or omit the transfer arm M_bias_lvt (about 3 min)? |
| Q5 | V2 minimum of 70 paired devices |
| Q6 | state P_a unconditionally, or only if S1_lodo_D lies within the I7-1 calibration range (-0.379 to 0.504)? |
