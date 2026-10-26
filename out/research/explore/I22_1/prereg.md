# I22-1 pre-registration (IN FORCE): matched-budget baselines; (A) correction-stage baselines B1, B2, B3_d3, B4 under the selected g v2 protocol on the 9 measured DEV datasets, (B) NGSpice-call and wall-time accounting per arm, (C) CMA-ES and TuRBO at the C0_refit NGSpice-call budget on sky130 77 K

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-26), committed alone, before any I22-1 code, simulation, training or refit. Base commit ca81768.
DEV data only: every TEST split stays locked (no TEST tag is loaded; every loaded tag is checked against the DEV /
TEST lists of its split file; sky130 77 K has no locked TEST since the 2026-10-21 user decision, all 18 devices are
DEV). This study writes no amendment of the committed I19-1 TEST gate decisions (a755e18).

Origin: method-freeze draft open item 12 (ICML target gaps): B1, B2, B3_d3 and B4 were run with the v1 recipe (I4-1,
I5-1, I6-1, I8-1, I9-1; state L on sky130, NIST and TU Delft; sky130 on 12 REAL_DEV devices; B4 on sky130 only), not
with the selected g v2 protocol; the cost of g training, S3_D and nested lambda selection is not reported in NGSpice
calls; parameter-extraction-stage baselines (CMA-ES, BO / TuRBO) have no matched-budget runs.

Results known when this was written (all v1-era, DEV): I4-1 sky130 REAL_DEV (12) GMR(M_bias v1, X) B1 0.770, B2
0.773, B3_d3 0.844, B4 0.658; I8-1 NIST lvt B1 0.787, B2 0.811, B3_d3 0.981 [0.961, 1.000]; I9-1 NIST RT card B1
0.780, B2 0.804, B3_d3 0.955 [0.926, 0.985]; I5-1 IHP GMR(B1, C0_refit) 0.794 / 0.775 / 0.695 / 0.809 (233 / 300 /
343 / 398 K), B2 0.916 / 0.946 / 0.889 / 0.886, B3_d3 1.238 / 1.375 / 1.311 / 1.364; I6-1 TU Delft GMR(M_bias v1, X)
4.13 K B1 0.885, B2 0.954, B3_d3 0.983; 295.0 K B1 0.769, B2 0.788, B3_d3 0.896. No B4 row exists for NIST, IHP
measured or TU Delft. No I22-1 value exists.

The three parts are independent; each has its own validity checks. Part A has the outcome rule; Part B is a report
table without a rule; Part C has its own labels.

## Part A: correction-stage baselines under the selected g v2 protocol

### Datasets, records, folds (identical to the selected g v2 rows they are compared with)

| key | dataset | DEV devices | starting records (state C) | folds | training records of a fold |
|---|---|---|---|---|---|
| sky130 | sky130 77 K | 18 | `I13_1/records` (real_dev + real_test; direct43_r3_fd cards) | LODO, 18 | the other 17, `i12_1_symm.fold_train_tags` (load order of `i3_2_bias.load_records_all18`) |
| nist4 | NIST sky130 4 K lvt (I8-1) | 53 (finite starting records) | `I20_1/start/records_nist4` | LODO, 53 | the other 52, sorted tag order (I20-1 `finite_order`) |
| nistrt | NIST sky130 4 K, RT card (I9-1) | 87 | `I20_1/start/records_nistrt` | LODO, 87 | the other 86, sorted tag order |
| ihp233, ihp300, ihp343, ihp398 | IHP SG13G2 233 / 300 / 343 / 398 K | 18 each | `I5_1/start/records_<T>K` (PSP103 has no A2 clamp; records unchanged) | LODO within class, 18 | same-class DUTs, `i17_2_ihp.fold_train` (load order) |
| tud4, tud295 | TU Delft 40 nm 4.13 / 295.0 K | 45 each | `I18_2/start/records_<T>K` | LOGO, 7 geometry groups | other groups, `i18_2_tud.fold_train` (I6-1 load order) |

### Arms

Every arm refits each held-out card from its state-C starting card with `ngspice_fit.polish` (TRF, irls 2,
diff_step 0.01) and is scored with the strict paper RRMS (`e1_correction.score_device`, frozen curve inclusion of
the starting record). Simulator and state: sky130 `simulate_pdk` (default `bsim4_state="clamped"`, ngspice-41
arm64); NIST `i8_1_nist.simulate` / `i9_1_nist_rt.simulate` (default clamped); TU Delft I18-2 `ProblemC` (state C);
IHP ngspice-46 + psp103.osdi (I5-1 `MeasProblem`).

- **C0_refit (rerun)**: no correction, max_nfev 400. Used for validity VA1 and for the cost table; the comparisons
  use the stored C0_refit rows (equal to the rerun if VA1 holds).
- **B1 wider card set** (I4-1 definition): base box plus the extra card parameters kept by the I4-1 edge screen
  (threshold 1e-4; a candidate is kept if the edge response at either box edge is >= 1e-4 on at least one starting
  card; non-finite responses count as 0). The screen is recomputed on the state-C starting records of all DEV devices
  of the dataset: sky130 on all 18 (universe: `out/pdk_parameter_sensitivity/summary.json` parameters minus the base
  box, `BSIM4_ZERO_RANGE`, card values from `readbacks.json`), NIST per dataset (I8-1 / I9-1 universe and per-device
  readbacks), TU Delft per temperature (I6-1 universe). IHP: the stored I5-1 per-class screen at 300 K
  (`I5_1/start/b1_screen.json`; records, universe and simulator unchanged). max_nfev = `matched_nfev` (the same
  per-IRLS-round TRF budget as the base box).
- **B2 extrinsic resistance and self-heating** (I4-1 definition): base box plus rext_ohm in [0, 4 x series resistance
  per side of the starting card / W] (netlist resistor on drain and source) and heat_kappa in [-0.5, 0.5]
  (I x exp(kappa |V(d,s) I| / P_max)), both starting at 0; max_nfev = `matched_nfev`.
- **B3_d3 cubic polynomial g** (I4-1 definition): g = tanh(P_pol(u, v)), total degree 3 in standardized
  (|V(g,s)|/VDD, |V(d,s)|/VDD), separate coefficients per polarity; E1 loss + 1e-4 x squared non-constant
  coefficients; Adam lr 3e-3, 2000 steps, zero initialization, float64; trained per fold on the fold's training
  records (order as in the table); frozen in the held-out refit (max_nfev 400).
- **B4 GP discrepancy** (I4-1 definition): GP posterior mean of ln|I_meas| - ln|I_sim| on (is_pmos, |V(g,s)|/VDD,
  |V(d,s)|/VDD), squared-exponential ARD kernel, constant mean, type-II ML (L-BFGS-B, maxiter 200); 2000 training
  points per fold (equal share per training device, `default_rng(0)`, devices in the fold's training order); frozen
  in the held-out refit (max_nfev 400).

VDD of the inputs: the record's `vdd` (as in the stored v1-era runs). No hyperparameter of B1-B4 is tuned in I22-1.

### Comparison rows (stored, not rerun)

C0_refit, M_bias v1 and selected g v2 per-device rows as listed in the method-freeze draft section 2 and loaded by
`scripts/research/plot_freeze_draft.py` `datasets()`: sky130 `I13_1/lodo` (C0_refit), `I15_2/decision.json`
`selected_rows`; NIST `I20_1/base_<ds>` (C0_refit), `I20_1/lodo_<ds>` (selected); IHP 300 K `I17_2/base` (C0_refit),
`I17_2/decision.json` `selected_rows`; IHP 233 / 343 / 398 K `I5_1/lodo_<T>K` (C0_refit), `I21_1/ihp_<T>K/decision.json`
`rows.selected`; TU Delft `I18_2/decision.json` `per_temperature.<T>.rows.C0_refit`, `selected_rows`.

### Statistics

GMR(X, Y) = exp(mean over paired devices of ln(RRMS_X / RRMS_Y)), 95% percentile bootstrap CI, 10000 resamples,
seed 0 (`i3_2_bias.paired_ratio`); resampling unit device, TU Delft the 7 geometry groups (cluster). Devices with a
non-finite value in either arm are left out of the ratio and counted. All-device mean and combined RRMS (mean of the
nMOS and pMOS family means) per arm and dataset. Pooled exp(mean ln GMR vs C0_refit) over the 9 datasets per arm
(the freeze-draft statistic), equal weight per dataset.

Labels per dataset and baseline X:
- vs C0_refit: **gain** (CI upper < 1), **worse** (CI lower > 1), **inconclusive** otherwise.
- vs selected g v2: **X better than g v2** (CI upper of GMR(X, g v2) < 1), **g v2 better than X** (CI lower > 1),
  **inconclusive** otherwise; flag **X non-inferior to g v2** iff CI upper < 1.05 (the I20-1 / I21-1 margin).

### Outcome rule

Gain set G = the datasets where the stored selected g v2 rows gain over C0_refit (CI upper < 1): sky130 (0.743
[0.654, 0.855]), NIST lvt (0.803 [0.757, 0.857]), NIST RT card (0.811 [0.761, 0.871]).
- Baseline X **matches g v2** iff X is non-inferior to g v2 on all 3 datasets of G.
- **g v2 ahead of X** iff "g v2 better than X" on all 3 datasets of G.
- Outcome: **"selected g v2 not matched by a correction-stage baseline"** iff no X matches g v2; otherwise
  **"matched by X"** (list). Reported without a rule: for each dataset outside G, the baselines that gain over
  C0_refit and the label vs g v2.

### Validity (per dataset; a dataset that fails is reported invalid and left out of the outcome; if a dataset of G
is invalid the outcome is "not evaluated")

- VA1: the C0_refit rerun reproduces the stored C0_refit row on every device (max |difference| <= 1e-12).
- VA2: every arm has a row for every DEV device; non-finite RRMS rows <= 10% of the devices per arm.
- VA3: unit tests of the new code and of the reused modules pass.
- VA4: every loaded tag is a DEV tag of the dataset's split file (sky130: the 18 tags of `real_split.json`).
- VA5 (reported, not a validity condition): the IHP B1, B2, B3_d3 reruns vs the stored I5-1 rows (same protocol;
  difference expected 0).

## Part B: NGSpice-call and wall-time accounting (report only, no rule)

One NGSpice call = one simulation of a device's full measured bias set (the unit counted by `polish` `n_sim`).
Per dataset and arm (C0_refit, M_bias v1, selected g v2, B1, B2, B3_d3, B4):
1. Starting cards (shared by every arm): stored per-device calls where a summary holds them (sky130 direct fit 8404.6
   + FD polish 313.8 per device; NIST / IHP / TU Delft `polish` max_nfev 3000 runs: stored n_sim if present, else
   reported as "not stored").
2. Arm-specific calls before the held-out refit: B1 screen (2 calls per active candidate per device + 1 reference);
   g v2 penalty-grid decks (one DC sweep deck per DEV device, reported with its point count, not converted to calls);
   g v2 nested lambda selection inner refits (stored n_sim where stored; otherwise the number of inner refits x the
   median n_sim of the dataset's selected g v2 held-out refits, labeled "estimate"); S3_D gate: 0 NGSpice calls.
3. Held-out refit calls: sum of n_sim over the DEV devices (I22-1 rows for C0_refit and B1-B4; stored rows for M_bias
   v1 and selected g v2 where stored, else labeled).
4. Wall time: training, selection and refit seconds as stored or measured, with the worker count.
Prediction P7 (scored, no rule): the g v2 nested selection uses more NGSpice calls than all held-out refits of the
g v2 arm on every dataset where it was run.

## Part C: parameter-extraction-stage baselines at the C0_refit NGSpice-call budget (sky130 77 K, 18 devices)

- Problem: the C0_refit stage. Each device's state-C starting card (direct43_r3_fd), the 43-parameter round-3 box
  in local coordinates [-1, 1] (`cold_loop.box_for`), `DeviceProblem` (ngspice-41 arm64, state C). Objective f =
  device paper RRMS = mean over the frozen included curves of the curve RRMS (the quantity `polish` keeps the best
  card by). A non-finite simulation counts as RRMS 10.
- **LS** (reference): the C0_refit `polish` (TRF, max_nfev 400, irls 2, diff_step 0.01) with every NGSpice call
  traced (best-so-far f after each call). Budget B_d = n_sim of this run on device d. Secondary trace: the same with
  max_nfev 1600 (reported, no rule).
- **CMA-ES**: pycma 4.4.4, x0 = starting card, sigma0 = 0.1 (local units), bounds [-1, 1] (pycma BoundTransform),
  default population (15 for 43 parameters), objective ln f, seeds 1, 2, 3. The starting card is the first call.
  Secondary configuration: sigma0 = 0.03.
- **TuRBO-1** (Eriksson et al. 2019; BoTorch 0.18.1 + GPyTorch, installed into the venv for this part): inputs scaled
  to [0, 1]^43 from the local box; SingleTaskGP with BoTorch default priors, outcome standardized, objective -ln f;
  initial design = starting card + 9 scrambled Sobol points in the initial trust region centered at the starting
  card; batch q = 4 by Thompson sampling over 5000 candidates (perturbation probability min(1, 20 / 43)); trust
  region centered at the best point, side lengths scaled by the GP lengthscales (TuRBO weights); L_init = 0.1
  (normalized units; = 0.2 local units), L_min 0.5^7, L_max 1.6, success tolerance 3, failure tolerance ceil(max(4 /
  q, 43 / q)) = 11; on L < L_min the trust region restarts at L_init around the best point. Seeds 1, 2, 3. Secondary
  configuration: L_init = 0.03.
- Every CMA-ES and TuRBO run uses 4 B_d NGSpice calls; the best-so-far value at B_d is the matched-budget result, the
  value at 4 B_d is secondary. The best card at B_d is re-simulated once (not counted) and must reproduce the traced
  value (validity).
- Statistic: per device and method, value = exp(mean over the 3 seeds of ln best-so-far f at B_d). GMR(method, LS)
  over 18 devices, device bootstrap (10000, seed 0). Labels (primary configurations): **LS better** (CI lower > 1),
  **method better** (CI upper < 1), **inconclusive**. Reported: GMR vs the starting card, secondary configurations,
  4 B_d values vs the max_nfev 1600 LS trace, best-so-far curves (geometric mean over devices of f / f_start vs
  calls / B_d), wall time per run (incl. GP fitting).
- Validity VC1: the traced LS run reproduces the stored C0_refit row on every device (max |difference| <= 1e-12);
  VC2: every run used exactly its budget and the re-simulated best card reproduces the traced value (|difference|
  <= 1e-12); VC3 unit tests pass. Compute estimate: under 2 h (8 workers), so Part C is run in full scope.

## Predictions (scored in the outcome entry)

- P1 sky130: g v2 better than B1, B2, B3_d3 and B4.
- P2 NIST lvt and NIST RT card: g v2 better than B1, B2 and B4; B3_d3 non-inferior to g v2 on both.
- P3 IHP (each temperature): B1 gains over C0_refit and is better than g v2.
- P4 TU Delft (each temperature): g v2 better than B1 by point estimate; every label vs g v2 inconclusive.
- P5 Outcome: "selected g v2 not matched by a correction-stage baseline".
- P6 Part C: LS better than CMA-ES and than TuRBO at the matched budget (primary configurations).
- P7 Part B as stated above.

## Order of operations

1. This file and `prereg.json`, and the RESEARCH_LOOP pre-registration entry, committed alone; push.
2. Code (`src/cryoml/explore/i22_1_baselines.py`, `scripts/research/explore/i22_1_baselines.py`), unit tests, smoke
   (2 devices per dataset, reduced budgets, output under `I22_1/smoke`); BoTorch installed; commit.
3. Queue, one job at a time: B1 screens, Part A per dataset, Part C, cost table, decide.
4. decision.json, report batch `out/research/report/2026-10-<dd>_i22_1/`, site claim, RESEARCH_LOOP outcome, freeze
   draft open item 12, memory status line; commit and push.

Compute estimate: Part A about 2-3 h (8 workers; v1-era runs of B1 / B2 / B3_d3 took 6-13 min per dataset, B4 GP
training about 1 min per fold), Part B bookkeeping only, Part C about 1-2 h.
