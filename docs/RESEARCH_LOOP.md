# Cold-start research loop (started 2026-10-15)

Goal (user): fit the 77 K data as well as possible — better than the paper
(all-device RRMS 0.2038 with 43 parameters, 0.2284 with seven; released card
0.2751) — and later at other temperatures. ML preferred, not required.

All numbers are all-device mean RRMS with the frozen published-baseline curve
inclusion unless stated; NGSpice re-simulation is always the final score.

## Environment change

The pinned x86_64 conda-forge ngspice-41 no longer runs (macOS 27, no
Rosetta). A native arm64 build of the same ngspice-41 release is at
`~/cryo-ng41/arm64/bin/ngspice` and is preferred by `cryoml.config`.
It reproduces the reference scores exactly
(`out/research/reference_scores.json`):

| card | all-device | nMOS | pMOS |
|---|---:|---:|---:|
| released (Beall et al.) | 0.27505 | 0.1198 | 0.3992 |
| paper, seven parameters | 0.22844 | 0.0788 | 0.3481 |
| paper, 43 parameters | 0.20380 | 0.0704 | 0.3105 |
| cold start: every released multiplier = 1 | 11.6006 | 0.6847 | 20.3333 |

`verify_simulator.py` agrees with the upstream sweeps to <=1e-11 A
(`out/tables/simulator_verification_arm64_ng41.json`).

## What "cold start" means here

The released 77 K card writes each tuned value as `{base * multiplier}`;
the base is the stock SkyWater value.  The paper's banks, starts and fallback
were all centered on the released (multiplied) card.  The cold loop instead
centers round 1 on the multiplier-free card and searches log-uniformly:
vth0 within x3, the other six paper parameters within x100, the remaining
sensitivity-screened parameters within x3 (`src/cryoml/cold_start.py`).
Each later round re-centers every device's box on its selected,
NGSpice-validated card; interior coordinates halve their log width, edge
coordinates keep it so the box moves outward (same rule for all devices).
The surrogate is always trained from random initialization.

## Methods under test

- Paper pipeline (cold): bank -> surrogate -> multistart surrogate FD ->
  NGSpice validation (`scripts/research/cold_loop.py`).
- A, tandem autoencoder: the study script's `inverse_autoencoder`, run in
  every cold round.
- B, conditional flow-matching posterior and K-head winner-takes-all model in
  a round-independent log-multiplier coordinate, pooled over rounds
  (`src/cryoml/posterior.py`, `scripts/research/posterior_study.py`).
  Evaluation: K samples -> NGSpice -> best -> optional NGSpice polish.
  Measured curves are out of distribution for any bank, so training uses
  discrepancy-transfer augmentation (another device's measured-vs-closest-
  simulation signed-log gap, never the device's own) and samples are clipped
  to the device's prior support.
- Non-ML reference: CMA-ES + TRF directly on NGSpice
  (`scripts/research/ngspice_direct.py`).

Heavy jobs run one at a time via `scripts/research/queue_runner.sh`
(`out/research/queue.txt`, `done.txt`, `logs/`).

## Results log

(appended per round)

## Temperature plan (user direction, 2026-10-15)

User: for other temperatures, simulate several temperatures to build the
dataset rather than waiting for measurements. Facts that shape this:

- The 77 K card has `tnom=-196.15` and zeroed temperature coefficients
  (kt1 = ute = 0), so it cannot be extrapolated; changing `.temp` on it does
  nothing meaningful.
- The stock SkyWater card (tnom = 27 C) carries real BSIM4 temperature
  coefficients and is itself a fit to 300 K silicon, so it serves as the 300 K
  anchor; the 77 K measurements are the cold anchor.

Plan: banks sweep `.temp` (e.g. 77, 100, 150, 200, 250, 300 K) as an input
alongside the card parameters, including the BSIM4 temperature coefficients
(kt1, kt2, ute, ua1, ub1, uc1, at, prt) as learnable parameters.  Train
temperature-conditioned models (I-V, T -> parameters; and a T-aware
surrogate).  Target: one card at tnom = 27 C whose temperature coefficients
reproduce the stock card at 300 K and the measurements at 77 K.  Caveat:
BSIM4's temperature laws are not built for cryogenic range, so intermediate
temperatures are a model prediction, not validated, until measured data at
another temperature exist.

## Results: seven parameters, cold start (2026-10-15)

All scores re-checked independently with `scripts/research/verify_cold_results.py`
(`out/research/verify_seven_r4.json`); they match the pipeline exactly.

| method (fixed procedure, all 18 devices) | all-device | combined | nMOS | pMOS | NGSpice calls/device |
|---|---:|---:|---:|---:|---:|
| released card | 0.2751 | — | 0.1198 | 0.3992 | — |
| paper, seven (hot) | 0.2284 | — | 0.0788 | 0.3481 | — |
| paper, 43 (hot) | 0.2038 | — | 0.0704 | 0.3105 | — |
| cold paper pipeline, round 1 | 0.2367 | 0.2251 | 0.1211 | 0.3291 | 10k bank |
| cold paper pipeline, round 2 | 0.1921 | 0.1820 | 0.0917 | 0.2723 | 20k cumulative |
| cold paper pipeline, round 3 | 0.1758 | 0.1660 | 0.0774 | 0.2545 | 30k |
| cold paper pipeline, round 4 | **0.1697** | **0.1602** | 0.0746 | 0.2457 | 40k |
| direct NGSpice (CMA-ES + TRF), round-1 box | **0.1676** | **0.1575** | 0.0665 | 0.2484 | 3.3k (291 s wall) |
| A: tandem autoencoder, round 4 | 0.6857 | 0.6859 | 0.6874 | 0.6843 | 0 |
| B: flow posterior, best of 256 + polish (r1) | 0.3719 | — | — | — | ~700 |
| B: 16-head WTA, best of 16 + polish (r1) | 0.5122 | — | — | — | ~400 |

Observations:

- Seven parameters, cold, beat the paper's 43-parameter hot result
  (0.2038). The paper's own boxes (±10% of released) were the binding
  limit, not the method: round-1 cold boxes allow ×100.
- Which candidate the paper pipeline keeps (NGSpice-validated among FD
  endpoints, top bank samples, fallback): rounds 1–4 = FD endpoint
  11, 8, 5, 3 of 18. By round 4 the winner is usually a raw NGSpice bank
  sample, so the iterated loop behaves mostly like an NGSpice
  cross-entropy search; the surrogate's FD contributes less each round.
  (The paper's own vb_seven/vb_full runs also kept bank samples on 11/18.)
- Direct NGSpice matches the 4-round pipeline with ~12× fewer
  simulations. Many direct parameters sit at the ×0.01/×100 box edges
  (eta0, vsat, rdsw, delta), and rdsw reaches ×2·10^4 of released on some
  devices: fits may rely on non-physical compensation. Needs a physicality
  check before any card is presented.
- A and B are far behind on measured data. B recovers simulated
  parameters well, so the gap is the simulation-to-measurement mismatch,
  not the model.

Queued next: 43 parameters by direct NGSpice seeded from the seven-parameter
fit and fully cold; a 3× longer seven-parameter direct run (convergence
check).

## Held-out simulated devices: simulation-vs-measurement gap (user request)

Every method also runs, unchanged, on held-out simulated devices: for each
of three sets, one card per paper device drawn log-uniformly from the round-4
seven-parameter box (clipped into the round-1 box), simulated in NGSpice on
that device's measured bias grid (`scripts/research/make_sim_targets.py`,
`out/research/sim_targets/`). No bank contains these cards. The targets
are realistic: a true card scores a median RRMS of about 0.45 against the
real device it imitates. `CRYOML_TARGET_OVERRIDE` substitutes them for the
measurements in `load_device_curves` and in every bank `meas` reader; bias
grids, frozen curve inclusion and the metric are unchanged (true card scores
exactly 0). `scripts/research/simgap_table.py` builds the table: the real
all-device RRMS, the mean sim RRMS over sets, and median parameter error.
Low sim RRMS with high real RRMS = model-form/measurement gap; high sim
RRMS = the method itself fails to invert the simulator.

Caveat: the pipeline's sim column reuses its round-4 stage (surrogate and
bank centered on the real-device fits; targets are drawn in that box), so it
is optimistic relative to a full four-round run per target. Direct NGSpice
and B use only round-1 information.

## Results: gap table, 43 parameters, matched budgets (2026-10-15 11:30)

Sim = mean over three held-out simulated-device sets (18 devices each).
Re-scored independently (`verify_direct.py`, `verify_cold_results.py`).

| method | real all-device | real combined | real median | sim all-device | sim median | param err (sim) |
|---|---:|---:|---:|---:|---:|---:|
| cold paper pipeline, round 4 (surrogate+FD) | 0.1697 | 0.1602 | 0.1585 | 0.0157 | 0.0141 | ×1.04 |
| A: tandem autoencoder, round 4 | 0.6857 | 0.6859 | 0.6344 | 32.41* | 0.0258 | ×1.05 |
| direct NGSpice, seven, round-1 box | 0.1676 | 0.1575 | 0.1460 | 0.0191 | 0.0000 | ×1.00 |
| B: flow r1, best of 256 + polish | 0.3719 | 0.3514 | 0.2670 | 0.3812 | 0.1839 | ×1.91 |
| B: WTA r1, best of 16 + polish | 0.5122 | 0.4912 | 0.3509 | 0.6932 | 0.2468 | ×2.09 |
| direct NGSpice, 43, seeded from seven fit | **0.1201** | **0.1126** | 0.1071 | pending | | |
| direct NGSpice, 43, fully cold | 0.1269 | 0.1186 | | | | |

\* one device (nmos L1/W3) explodes to RRMS 500–1200 in two sets; the median
device is fine.

Reading the gap:

- NGSpice-in-the-loop methods (pipeline, direct) invert the simulator almost
  perfectly (sim ≈ 0.015–0.02, parameters within 4%). Their real-data error
  of ~0.17 is therefore the seven-parameter model-form/measurement gap, not
  an optimization failure — which is why 43 parameters cut it to 0.12.
- A inverts simulations well on the median device but collapses on measured
  curves: its problem is purely simulation-to-measurement transfer.
- B is as bad on simulations as on measurements: the model itself fails to
  invert the round-1 (×100) prior. Next: pool rounds 1–4 (≈650k cards),
  150 epochs.
- B's NGSpice polish rarely moves (TRF terminates in ~25 sims): its best
  samples sit in local basins; only CMA-ES escapes them.

Other runs:

- Seven-parameter direct with 3× budget (7.9k calls): 0.16708 vs 0.16757 —
  converged; seven parameters cannot go much below ~0.167.
- Matched budget (1k CMA + 300 TRF): bank-sample starts 0.1777 (1505 calls)
  vs flow-best-sample start 0.1782 (1416 calls, plus the flow's 257): the
  flow start gives no saving.
- 43-parameter direct fits leave ~6 of 43 parameters per device at the box
  edge (mostly secondary ones held to ×3: dwg, k3b, pdiblc2, xj, prwg, k2).
  Round 2 re-centers with the cold-loop rule (running).

Temperature note: the stock SkyWater PDK is local under
`data/raw/pdk/volare/.../sky130A`. The 77 K fits move nfactor ×26 from stock;
BSIM4 has no temperature law for nfactor (and u0 ×6.6 would need ute ≈ −1.4),
so a single 77–300 K card needs temperature-dependent parameter expressions
(ngspice evaluates `temper` in model expressions), not just stock temperature
coefficients.

## Uniform NGSpice FD polish for every method (user request, 2026-10-15 17:30)

Bug found first: `ngspice_fit.polish` used SciPy's relative `diff_step`/`xtol`;
a start at the box center (local = 0, as in B's ×1.5 polish box) stopped
after one rejected step without moving. Fixed with the absolute-step local
Jacobian of `scripts/real_fd_polish.py` (`xtol=ftol=gtol=1e-10`). Direct
NGSpice runs were unaffected (they polish from CMA-ES endpoints).

Every method's cards then get the identical final stage
(`scripts/research/polish_cards.py`): bounded TRF on fixed-inclusion per-curve
NGSpice residuals against the target curves, ~400 simulations/device budget,
in the method's own search box (pooled B: union of the round 1–4 boxes,
`union_spec.py`; a round-1-only box clipped its cards). Real and all three
held-out simulated sets.

| method + NGSpice FD | real all-device | real combined | real median | sim all-device | sim median | FD calls/device |
|---|---:|---:|---:|---:|---:|---:|
| cold paper pipeline, seven, round 4 | **0.1643** | **0.1548** | 0.1513 | 0.0003 | 0.0000 | 270 |
| direct NGSpice, seven | 0.1670 | 0.1570 | 0.1446 | 0.0190 | 0.0000 | 187 |
| B flow pooled r1–4, best of 256 | 0.1728 | 0.1632 | 0.1662 | 0.0942 | 0.0001 | 248 |
| B flow r1, best of 256 | 0.2209 | 0.2067 | 0.1664 | 0.1546 | 0.0000 | 239 |
| B WTA pooled, best of 16 | 0.2268 | 0.2242 | 0.1658 | 0.0697 | 0.0000 | 251 |
| A tandem, round 4 | 0.2270 | 0.2245 | 0.1746 | 0.6112* | 0.0000 | 298 |
| B WTA r1, best of 16 | 0.2587 | 0.2396 | 0.1512 | 0.1108 | 0.0001 | 248 |
| direct NGSpice, 43, seeded (round 1) | 0.1191 | 0.1117 | 0.1071 | 0.0073 | 0.0000 | 319 |
| direct NGSpice, 43, round 3 | **0.1044** | **0.0976** | — | not run | — | 314 |

\* the nmos L1/W3 outlier from A's raw prediction survives FD.

Before → after FD (real all-device): pipeline 0.1697→0.1643, direct-7
0.1676→0.1670, B flow pooled 0.2922→0.1728, B flow r1 0.3801→0.2209,
B WTA pooled 0.4608→0.2268, A 0.6857→0.2270, B WTA r1 0.5148→0.2587,
direct-43 0.1201→0.1191, direct-43 r3 0.1049→0.1044.

Reading: after the same FD, the ML paper pipeline (surrogate) is the best
seven-parameter method, slightly ahead of direct NGSpice. FD rescues A and B
substantially (all recover the median simulated device exactly), but their
real-data means stay behind because FD is local: devices where the proposal
lands in the wrong basin stay there. Pooling rounds for B moves flow+FD from
0.221 to 0.173. With 43 parameters, direct NGSpice rounds 1→2→3 give
0.1201→0.1094→0.1049 (re-scored independently). Seven-parameter model-form
floor ≈ 0.164; 43 parameters is where the remaining gain is.

Next: cold 43-parameter paper pipeline (rounds 1–4) is running, so the ML
pipeline can be compared at 43 parameters.

## Cold 43-parameter paper pipeline, round 1, and why it trails direct NGSpice

Round 1 (`cold_loop.py --set full`, 10k bank/device, ~40% non-converged
samples in the ×100 box): pipeline 0.2375 all-device (combined 0.2281); tandem
A 0.5922. Direct NGSpice in the same box: 0.1269 cold, 0.1201 seeded.

Difference between the two procedures: the paper pipeline spends its NGSpice
budget on a 10k-card random bank, trains a neural surrogate (params → I–V),
searches on the surrogate (8 starts, surrogate FD), and uses NGSpice only to
check the final candidates. Direct NGSpice spends ~3.3k simulations/device
inside the optimizer (CMA-ES then TRF), all near the solution.

Diagnostic (`scripts/research/surrogate_gap.py`,
`out/research/surrogate_gap_full_round_1.json`, inference only): the round-1
surrogate predicts paper RRMS 0.203 for its own picks (NGSpice 0.2375) and
**0.507** for the direct-fit cards (NGSpice 0.127). On 17/18 devices it ranks
its own pick better while NGSpice ranks the direct card better. The good cards
sit on the box faces (max |local| ≈ 1), where the LHC is sparse and failures
cluster. So at 43 parameters the bottleneck is surrogate accuracy where it
matters, not the search. (At seven parameters the surrogate was accurate:
predicted 0.235 vs NGSpice 0.2367 in round 1.)

Queued: `surrogate_arch_study.py` on the round-1 cache: paper 4×1024 MLP,
6×2048 MLP, residual MLP (27M, 139M), DeepONet (branch ResMLP, trunk over
bias point), and a conditioned FNO along each curve's sweep axis
(`src/cryoml/operators.py`; truncated-DFT layers as GEMMs, verified equal to
FFT to 4e-7). Same split, loss and recipe; metrics are held-out loss,
emulator curve NRMSE, and whether the surrogate predicts/ranks the known
NGSpice scores of the pipeline and direct-fit cards. Learning curves at
1/8, 1/4, 1/2 of the data separate "too small" from "too little data".

## Simulated temperature benchmark (77–300 K)

Only 77 K measurements exist, so temperature methods are scored on simulated
devices (`src/cryoml/temperature.py`, `temp_fit.py`,
`scripts/research/temp_bank.py`, `temp_study.py`).

* The released 77 K card cannot be heated: with `tnom = -196.15 C` the
  junction model gives ~0.8 A at 150 K and NGSpice hangs at 300 K. With
  `tnom = 27 C` all 18 devices simulate cleanly from 77 to 300 K.
* ngspice-41 evaluates `temper` in model-parameter expressions, and
  `option temp=` inside one control block re-evaluates them, so all seven
  temperatures run in one process (identical to separate decks; 0.12 s for
  7 temperatures × 11 curves).
* Card: each paper parameter p(T) = p_stock · m^(w(T)^g),
  w = (27 C − T)/(223.15 K), clipped at 0. p(77 K) = p_stock·m,
  p(300.15 K) = p_stock (checked: equals the stock card to 3e-11 A). 14 numbers
  per device: log m and log g for the seven parameters. BSIM4's own
  temperature coefficients stay zero as released.
* 77 K multipliers anchored on the cold seven-parameter round-4 box (the real
  device neighbourhood); bank = 10k LHC/device in 1.5× that box with
  g ∈ [1/4, 4]; truths = 3 sets × 18 devices in 0.75× with g ∈ [1/3, 3].
* Observed temperatures 77, 150, 225, 300 K; held out 110, 190, 260 K.
  Score = frozen-inclusion paper RRMS vs the simulated truth at each
  temperature (all-device mean and combined), plus parameter error.
* Methods (same box, bank starts, NGSpice FD polish): direct NGSpice
  (CMA-ES 1500 + TRF 400), temperature-conditioned surrogate + gradient
  multistart + NGSpice validation (+FD) for paper MLP, ResMLP, DeepONet
  (temperature in the trunk) and FNO, and a one-pass direct MLP (+FD as a
  diagnostic).

Reference re-check after the `spice_pdk.py` multi-temperature change:
released card 0.27505 / nMOS 0.1198 / pMOS 0.3992 (unchanged); simulator
verification passes.

## NGSpice-swap controls for every surrogate ablation (2026-10-16, queued)

Each surrogate study gets a control row in which the surrogate is replaced by
NGSpice and nothing else changes: same 8 diverse bank starts, same TRF
2-point search (max_nfev 100, diff_step 0.02, reg 1e-4), same NGSpice
validation of top-4 endpoints + 4 starts + fallback, same FD polish. The gap
to that row is pure surrogate error.

- Pipelines: `scripts/research/swap_search.py` calls
  `finite_difference_surrogate_extract` unchanged with an NGSpice evaluator
  (round trip vs bank ≤5e-6 relative; failed points penalized at curve
  RRMS 10). Queued: seven r1, full r1, and FD polish of full r1 for both the
  swap and the pipeline.
- Architecture study: every trained model also runs that search end to end
  (`"end_to_end"` in its report; CPU spawn pool, identical results to the
  sequential path); the full-r1 swap row is the shared ceiling, also for
  the learning curves.
- Temperature study: the surrogate search is now the same pipeline search
  (Adam multistart removed, since it cannot be matched on NGSpice);
  `temp_study.py ngspice_search` is the swap row.
- Correction: direct NGSpice 43-parameter fits used ~21.9k NGSpice calls per
  device (TRF Jacobian evaluations included), not ~3.3k.

## Cold 43-parameter paper pipeline, rounds 1-4 (finished 2026-10-16 05:39, 12.4 h)

| round | surrogate+search all-device | combined | nMOS | pMOS | forward val loss | A (tandem) all-device |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.2375 | 0.2281 | 0.1435 | 0.3128 | 0.0322 | 0.592 |
| 2 | 0.1965 | 0.1875 | 0.1058 | 0.2691 | 0.0207 | 5.83 |
| 3 | 0.1716 | 0.1631 | 0.0865 | 0.2396 | 0.0113 | nan |
| 4 | **0.1513** | **0.1427** | 0.0652 | 0.2201 | 0.0085 | 1.70 |

Beats paper-43 (0.2038) and the cold seven-parameter pipeline r4 (0.1697),
still behind direct NGSpice 43 r3 (0.1049). Surrogate accuracy improves
every round as the box narrows and convergence rises. A is unstable at 43.

## Gradients through NGSpice (2026-10-16 scoping)

- Per-call cost ~31 ms: 5 ms process start, 8.5 ms library parse, ~11 ms
  Newton solving, ~6 ms output and Python I/O.
- `src/cryoml/spice_batch.py`: `simulate_pdk_multi` (altermod, bit-identical
  to separate calls on all 18 devices, failure-isolated) and
  `ngspice_jacobian`. The 43-parameter forward Jacobian takes 1.37 s → 0.65 s exact
  (0.30 s with the near-exact one-circuit "parallel" mode). `.sens` is wrong
  for delta/rdsw/eta0/pclm/dwg and in saturation/subthreshold; unusable.
- Differentiable BSIM4 port scoping: v4.8.2, rdsmod=0 but rbodymod=1 and
  rsh give 5 internal nodes, so the port needs a batched Newton solve plus
  implicit gradients. About 600 lines of b4ld DC + 1,150 lines of b4temp (half
  binning boilerplate). Estimate 2-3 weeks; parity is limited to ~1e-4 by
  ngspice tolerances. Not started.
- Side finding: BSIM4 divides u0 by 1e4 when u0 > 1 (cm² units). The
  cold round-1 box (u0 ×100) crosses that for ~14% of nMOS samples, a
  10^4 mobility cliff inside the box.

## NGSpice-swap and FD results (2026-10-16)

Real measurements, all-device (combined):

| method | raw | +FD | NGSpice calls/device |
|---|---:|---:|---:|
| pipeline 7, round 1 (surrogate search) | 0.2367 | | bank 10k |
| same search, NGSpice instead of surrogate, 7, r1 | 0.1887 (0.1767) | | 1.9k |
| pipeline 43, round 1 | 0.2375 (0.2281) | 0.1824 (0.1730) | bank 10k |
| same search, NGSpice instead of surrogate, 43, r1 | **0.1125 (0.1053)** | **0.1106 (0.1036)** | 19.9k |
| pipeline 43, round 4 | 0.1513 (0.1427) | 0.1465 (0.1380) | 4 × 10k banks |
| direct NGSpice (CMA+TRF) 43 r1 cold / r3 | 0.1269 / 0.1049 | — / 0.1044 | 21.9k / round |

The surrogate costs 0.125 RRMS at 43 parameters and 0.048 at seven in round 1.
The pipeline's 8-start TRF search on NGSpice beats CMA-ES+TRF in the same box.
The earlier claim that "the seven-parameter surrogate is accurate" held only for its own picks.

Simulated column, pipeline 43 r4 + FD: set 1 0.113. Sets 0 and 2 have mean
4.9 and 5.0 from one device (nmos_L1_W3 RRMS ~87, the same device that
exploded for A). Median ≈ 0.09-0.10. The 7-parameter sim truths may lie
outside the 43-parameter round-4 box. This needs a check.

Temperature benchmark (3 sets × 18 simulated devices, well-specified), observed
/ held-out all-device: direct CMA 0.0129/0.0362 (+FD 0.0015/0.0117);
NGSpice pipeline-search 0.0005/0.0021 (+FD 0.0000/0.0004); paper-MLP
surrogate search 0.0051/0.0081 (+FD 0.0000/0.0007). 110 K is the
hardest held-out temperature. Parameter error stays 0.16-0.27 (local RMS) even
at RRMS 1e-4, so the g exponents are weakly identifiable. The
well-specified task is too easy to separate methods, so it needs a misspecified tier.

## Research loop iteration 1 (2026-10-16, AI-Scientist-style)

Ideation lenses: method designer, literature scout, ICML reviewer. Nodes
implemented and adversarially verified:
- n1a: sketched-Jacobian TR (k NGSpice directional probes chosen from the
  surrogate Jacobian, Broyden secant, full batched Jacobian on demand); the
  r_s = 0 control is a random-subspace/Broyden Gauss-Newton TR.
- n1b: budget-aware allocation (zero-call surrogate presearch, racing with
  optimistic-bound drops, surrogate restarts).
- n1c: DEV harness (36 targets = sim sets 0 and 1, budgets 500/1000/2000/5000)
  with CMA-ES+TRF and lq-CMA-ES call-trace baselines and the pipeline r1 row.
81 DEV jobs queued, about 6-9 h. The real-device hybrid runs were removed from the
queue: real devices are TEST data and run once per frozen method. Tree:
out/research/tree/tree.json.

## DEV iteration 1 results (2026-10-16, 36 well-specified simulated targets)

All-device mean RRMS at fixed NGSpice budgets (every call counted):
B500: racing+surrogate 0.0287, sketch+surrogate 0.0313, sketch-only 0.0370,
hybrid+surrogate 0.0463, CMA-ES+TRF 0.0565, hybrid-only 0.0579.
B5000: sketch-only 0.0021, hybrid-only 0.0058, sketch+surrogate 0.0091,
CMA-ES+TRF 0.0109, hybrid+surrogate 0.0114.
Paired surrogate minus NGSpice-only (95% bootstrap CI): hybrid B500 -0.0116
[-0.0230, -0.0024], B5000 +0.0056 [0.0018, 0.0112]; sketch B500 -0.0058
[-0.0126, 0.0003], B5000 +0.0069 [0.0024, 0.0133].
Reading: the surrogate lowers error below ~500 calls and raises it from about
2000 calls on these zero-residual targets. Next: a misspecified DEV set (GF180/IHP
tier c) and a surrogate weight that decays as the trust region contracts.
Report: out/research/report/2026-10-16_dev_iteration1/.

## Pipeline 43 re-scored with failed-curve rejection (2026-10-16 22:18)

Final selection rerun with the round-1 and round-4 surrogates (out/research/pipe_fixed/).
All 18 devices are valid. Round 1: 0.2382 all-device (combined 0.2289); stored was 0.2375,
and nmos_L0p15_W1p6 changes 0.2035 → 0.2159. Round 4: 0.1522 (combined 0.1437); stored was
0.1513, and nmos_L8_W1p6 changes 0.0263 → 0.0431. The round-4 box still reflects the
round-3 pick made before the fix. TODO: update site claims icml-swap, cryo-rounds, cryo-43, cryo-fd.

## Research loop iteration 2 (2026-10-17)

Nodes (all adversarially verified after fixes; 288 DEV jobs queued, about 17 h):
- n2a: SkyWater tier-c misspecified DEV targets (sky130c sets 0 and 1) with
  calibrated mismatch strength, refit floors, and a DEV registry so every arm
  runs on tier a and tier c.
- n2b: per-curve remainder-weighted corrected model. lambda_j is estimated by least
  squares from the NGSpice second-order remainder at trial points already
  simulated (no extra calls). It keeps first-order consistency, and lambda=0 and
  lambda=1 reproduce the iteration-1 NGSpice-only and surrogate arms.
- n2c: DFO-LS-style interpolation Gauss-Newton trust-region baseline
  (validated on the Moré-Garbow-Hillstrom set) and paired statistics.

### 2026-10-17 07:20: sky130 tier-c calibration

Calibration means of the polish floor on 6 devices: strength 0.5 → 0.0343, 1 → 0.0404,
2 → 0.0545. The pre-registered range was 0.06-0.12. Strength 4 could not be drawn: every
nmos_L1_W3 draw put u0 above 1 (BSIM4 unit switch) and was rejected. The rule (closest to
0.09 among the evaluated strengths) selects strength 2; the range was not reached. The 84
tier-c jobs that failed in cascade were requeued with --strength 2.

## Architecture study, 43 parameters, round 1 (finished 2026-10-17 07:12)

End-to-end with the pipeline's search, on the measured devices (diagnostic ablation, run before
the DEV/TEST protocol), all-device RRMS (test loss): paper MLP 0.2348 (0.0331), wide MLP 0.2463
(0.0324), ResMLP 0.2308 (0.0291), large ResMLP 0.2366 (0.0323), DeepONet 0.2274 (0.0326), FNO
0.2549 (0.0259). Same search with NGSpice in place of the surrogate: 0.1125. Across the six
architectures the end-to-end range is 0.227-0.255. The architecture with the lowest test loss
(FNO) has the highest end-to-end RRMS.
Report: out/research/report/2026-10-17_arch_study/.

## Iteration-2 DEV results (ablation; reported 2026-10-19)

B500 / B5000. Tier a: RRMS. Tier c (misspecified sky130, strength 2): excess over the per-target floor.
sketch+surrogate 0.0334/0.0082 and 0.0221/0.0047; sketch NGSpice-only 0.0367/0.0024 and 0.0225/0.0022;
lambda per-curve 0.0317/0.0062 and 0.0200/0.0039; hybrid+surrogate 0.0463/0.0114 and 0.0304/0.0064;
hybrid NGSpice-only 0.0579/0.0058 and 0.0401/0.0041; DFO-LS-style 0.0579/0.0076 and 0.0406/0.0062.
Pre-registered lambda decision: negative (no arm passed rules A1-A4). No TEST run.
The search-efficiency line ends here as an ablation (see memory exploration-phase).
Fixes: dev_compare tolerates a <=1e-4 fresh-vs-trace gap only on batched-probe bests (WARN);
temp_study FNO OOM (low-memory kill) fixed; rerun queued as temp_sur_fno_r.

## Exploration E1 stage 2 (2026-10-19 10:08): correction inside the refit, REAL_DEV LODO

Each held-out card was re-polished (TRF, 400 nfev) with that fold's correction in the residual.
All-device mean (combined): no-correction re-polish C0 0.1072 (0.0981); deployable MLP correction M
0.0871 (0.0842); per-tag gain table T1 (not deployable) 0.0820 (0.0760); 0 failed devices.
Pre-registered checks: M <= 0.090 yes; M < T1 mean no; M beats T1 on >= 8/12 no; M beats C0 on
>= 8/12 yes. Stage 2 does not pass (it requires M to beat T1). M reduces held-out RRMS by 19% relative to
the physics-only re-polish when the card is refit with the correction in the loop. In stage 1
(fixed card), the same correction increased RRMS.

## Exploration E1 simulated benchmark (2026-10-19 10:16): leave-one-geometry-out, post-hoc

GF180 (BSIM4) and IHP SG13G2 (PSP103) tier c, sets 0-2, 48 targets per PDK, 16 geometry folds
(all three draws of the held-out geometry removed from training), pre-registered hyperparameters.
The correction is applied post hoc to the fitted card's simulation; no in-loop refit was run on the
simulated sets. Post hoc is the setting in which M also failed on REAL_DEV (stage 1).
All-device mean (equal to combined; 24 nMOS, 24 pMOS): GF180 C0 0.0474, T1 0.0420, T2 0.0463,
LIN 0.0443, M 0.0496 (M < C0 on 19/48); IHP C0 0.1012, T1 0.0957, T2 0.0999, LIN 0.0987,
M 0.1553 (M < C0 on 13/48). criterion_S (M <= 0.90 x C0 and M < T1) fails on both PDKs; it is
secondary and not used for the E1 decision. Files: `out/research/explore/E1/gf180_c_logo/`,
`out/research/explore/E1/ihp_c_logo/`. Report batch: `out/research/report/2026-10-19_explore_e1_lc/`.

## Surrogate learning curves `archlc_full_r1` (finished 2026-10-19 10:05)

`scripts/research/surrogate_arch_study.py`, paper MLP, DeepONet and FNO trained on 1/8, 1/4 and 1/2
of the round-1 43-parameter training split (10,732 / 21,463 / 42,926 samples; fixed permutation,
seed 20260818); the full-split points (85,853) are the `arch_full_r1` reports. Same loss, recipe
and end-to-end search as the architecture study; the selected card is re-simulated in NGSpice on the
18 measured devices (diagnostic ablation, not used for method selection). First attempt killed
(rc 143) on 2026-10-18; the restart trained `fno_f0.5` and filled the remaining end-to-end runs.

| surrogate | test loss, 1/8 -> 1 | end-to-end all-device, 1/8, 1/4, 1/2, 1 | combined, 1/8 -> 1 | bank-start selections, 1/8 -> 1 |
|---|---|---|---|---|
| paper MLP | 0.1579 -> 0.0331 | 0.2588, 0.2599, 0.2504, 0.2348 | 0.2483 -> 0.2248 | 17 -> 11 |
| DeepONet | 0.1683 -> 0.0326 | 0.2581, 0.2497, 0.2360, 0.2274 | 0.2477 -> 0.2193 | 17 -> 9 |
| FNO | 0.1394 -> 0.0259 | 0.2494, 0.2549, 0.2468, 0.2549 | 0.2397 -> 0.2448 | 15 -> 16 |

Same search with NGSpice in place of the surrogate: 0.1125 all-device (0.1053 combined). Test loss
falls 4.8-5.4x over the 8x data range; end-to-end RRMS falls by 0.024 (paper MLP) and 0.031
(DeepONet) and rises by 0.005 (FNO), and stays 0.11-0.15 above the NGSpice swap at every size.
0 failed devices in all 12 runs. Report batch: `out/research/report/2026-10-19_explore_e1_lc/`.

## Exploration E2 decision run (2026-10-19 11:52): one-pass network with output augmentation

Queue jobs `E2_full` (11:26-11:40) and `E2_fd_sim` (11:40-11:52), seeds 0-2, `scripts/research/explore/e2_direct.py`.
REAL_DEV raw (one forward pass + one NGSpice validation), penalized mean over 12 devices, seed mean per device,
all-device (combined): N0 11.241 (9.738), M 6.265 (5.513), S 19.581 (16.955), nearest bank sample 0.905 (0.979);
failed device-seed pairs 9/36, 6/36, 4/36, 0/12. Medians: N0 5.490, M 2.795, S 1.621, nearest bank sample 0.396.
Pre-registered checks: C1 no (S / N0 = 1.742); C2 no (gap ratio 1.754); C3 no (S lower on 6/12); C4 guard yes
(tier a 14.89 vs 20.56); K1 met; K2 not met. Verdict: KILL. The seed-0 light run (S / N0 = 1.145) had the same
check outcomes; the three-seed run confirms it. Attribution check R(S) <= 0.90 x R(M): no (3.126).
FD polish (100 nfev), REAL_DEV all-device: N0 1.067, M 0.521, S 0.711, nearest bank sample 0.310; D (direct
NGSpice 7 + FD) 0.1735; released card 0.2848. Tier a raw mean: nearest bank sample 0.286, trained arms 6.12-14.89.
The penalized means are dominated by pmos_L0p35_W0p55 and pmos_L0p35_W5. E2 is stopped.
Report batch: `out/research/report/2026-10-19_explore_e2_e3_e4/`.

## Exploration E3 decision run (2026-10-19 12:23): temperature law under misspecification

Queue jobs `E3_full_M` (11:52-12:07, decision) and `E3_full_W` (12:07-12:23, secondary). Tier M: 72 training and
36 test simulated devices on the 12 REAL_DEV geometries; held-out 110/190/260 K; 0 failed cells.
All-device mean held-out RRMS: O4 lrn 9.801, par3 (control) 9.123, pt_interp 10.118, par3_pop 7.376;
O2 lrn 7.102, par3_pop (control) 6.858, pt_interp 9.592; floor 0.0136.
R_O4 = -0.074 (CI -0.090 to 0.504), R_O2 = -0.036 (CI -0.055 to 0.288); lrn vs pt_interp O4 0.031, O2 0.260.
R_S < 0.10 for both S: verdict KILL. E3 is stopped.
One test device, nmos_L1_W3__2, has held-out RRMS 226-353 for every method (110 K cell 676-1060; floor 0.007)
and determines the means. Diagnostics, not decision inputs: medians O4 lrn 0.168 vs par3 0.312 (lrn lower on
27/36), O2 0.354 vs 0.406 (21/36); means without that device O4 lrn 0.267 vs par3 0.423, O2 0.484 vs 0.610.
Tier W (truth inside the parametric family): O4 lrn 0.113 vs par3 0.067 (R -0.632); O2 lrn 0.490 vs par3_pop
0.746 (R 0.343); rule output INCONCLUSIVE.

## Exploration E4, 43 parameters (2026-10-19 13:35): held-out bias curves with intervals

Queue jobs `E4_p43_vd` (decision), `E4_p43_sparse`, `E4_p43_vg` (12:23-13:35), 12 REAL_DEV devices, about
22,000 NGSpice calls per device. `e4_analyze.py --select feasible` on sparse and vg was run for this report
(reads fitted records only); feasible equals strict on all three splits.
vd held-out all-device (combined): single 0.4177 (0.3751), ens 0.4418 (0.3968), Laplace 500.2 (428.8),
learned 10.87 (9.82), crossinterp control 0.0977 (0.0897); 0 failed devices.
Checks on vd: ens C1 yes (coverage 0.744, width 1.006 <= 1.671), C2 no (6/12); Laplace C1 no (coverage 0.900,
width 7.44); learned C1 no (coverage 0.963, width 5846), C2 no, C3 no. Verdict: PARTIAL_CALIBRATION_ONLY
(continue only as a calibration component of another direction).
Secondary: sparse single 0.3945, ens coverage 0.676, crossinterp 0.2718, rule output KILL; vg single 0.2070,
ens C1 yes C2 no, learned 0.0664 (11/12, C2 yes) with coverage 0.965 (C1 no) and above crossinterp 0.0466
(C3 no), rule output PARTIAL_CALIBRATION_ONLY.
crossinterp is lower than single on 12/12 (vd), 11/12 (sparse), 12/12 (vg). The learned model's inner
leave-one-device-out MSE is about 7.5e6 (vd) and 4.5e6 (sparse) in every fold with pmos_L0p35_W0p55 in training.
Seven-parameter runs: p7 vd and p7 vg finished, p7 sparse running; reported when complete.

## Exploration E4, seven parameters (2026-10-19 14:19): held-out bias curves, secondary runs

Queue jobs `E4_p7_vd` (13:35-13:50), `E4_p7_vg` (13:50-14:09), `E4_p7_sparse` (14:09-14:19), seven-parameter box
`out/cold_loop/seven/round_1/spec.json`, 12 REAL_DEV devices, about 13,000-14,000 NGSpice calls per device, same
budgets and held-out-bias feasibility penalty as p43. prereg.json lists p7 vd and p7 vg as secondary without a
decision; sparse is secondary by amendment 1. `e4_analyze.py --run p7 --select feasible` was run for the report
(fitted records only); feasible equals strict on all three splits. 0 failed single-card devices.
Held-out all-device (combined): vd single 1.4536 (1.2764), ens 1.5017 (1.3178), Laplace 1146.5 (982.8),
learned 0.8253 (0.7558), crossinterp 0.1607 (0.1433); sparse single 8.5733 (7.3723), ens 8.7947, learned 1.9220,
crossinterp 0.6817 (0.6031); vg single 0.2527 (0.2249), ens 0.2589, Laplace 0.3125, learned 0.0555 (0.0573),
crossinterp 0.0436 (0.0401).
Rule outputs with the pre-registered decide(), reporting only: vd KILL (coverage ens 0.650, Laplace 0.608, learned
0.980 with width 78.5; learned C2 yes, 10/12, C3 no); sparse KILL; vg PARTIAL_CALIBRATION_ONLY (ens and Laplace C1
yes, coverage 0.757 and 0.766, C2 no, 1/12; learned C2 yes, 11/12, C1 no, C3 no). The E4 decision (p43 vd,
PARTIAL_CALIBRATION_ONLY) is unchanged.
p7 single is above p43 single on vd 12/12, sparse 11/12, vg 7/12 devices. p7 means are determined by single devices:
vd pmos_L0p35_W0p55 7.01, sparse pmos_L0p35_W5 96.0 (single card). crossinterp is below single on 12/12 in every p7
split. Learned inner leave-one-device-out MSE on p7 vd: 1.16e8 in every fold with pmos_L0p35_W5 in training.
Spearman (single held-out RRMS vs sloppy fraction), p7: vd 0.35, sparse 0.57 (p 0.051), vg -0.83 (p 0.001).
Report batch: `out/research/report/2026-10-19_explore_e4p7_e5/`.

## Exploration E5 decision run (2026-10-19 14:30): shared discrepancy

Queue jobs `E5_stage1` (14:19-14:29, 625 s; 3000 steps, seeds 0-4) and `E5_stage2` (14:29-14:30, 64 s),
`scripts/research/explore/e5_shared_discrepancy.py`. REAL_DEV leave-one-device-out, 0 failed devices in every arm.
43-parameter base (decision), all-device (combined): A0 0.1073 (0.0981), A1 const 0.1221 (0.1110), A2 MLP 0.2004
(0.2024), A4 no geometry 0.1681 (0.1625), A3 in-sample oracle 0.0343 (0.0322).
Stage-1 rule: reduction (A0 - A2) / A0 = -0.868, A2 < A0 on 1/12: KILL (light run: -0.686, 2/12).
Seven-parameter base (secondary): A0 0.1735, A2 0.2252 (-0.298, 3/12), A3 0.0419.
A2 raises nMOS from 0.0433 to 0.2147 (nmos_L20_W0p64 0.049 -> 0.551); A4 is below A2 and below A0 on 5/12.
Stage 2 (diagnostic, stage 1 not AMBIGUOUS): control polish 0.1072, joint refit with the fold's A2 delta 0.1235
(reduction -0.152, 3/12), joint-refit card without delta 0.1890. E5 is stopped.
Recorded deviations from the prereg text: AdamW, per-device weight renormalization, MPS float32.
Report batch: `out/research/report/2026-10-19_explore_e4p7_e5/`.

## Exploration iteration 1 complete (2026-10-19)

Rule outputs on REAL_DEV: E1 stage 1 and stage 2 fail their checks; E2 KILL; E3 KILL; E4 PARTIAL_CALIBRATION_ONLY
(continue only as a calibration component of another direction); E5 KILL. No exploration job is queued.

## Iteration 2, I2-0 (2026-10-19): diagnosis of the devices that determined the iteration-1 means

`scripts/research/explore/i2_0_diagnose.py`; stored records plus about 40 single-device NGSpice runs; no stored
verdict changed. Output: `out/research/explore/I2_0/` (diagnosis.md, JSON tables, PNGs).
A. pmos_L0p35_W5 and pmos_L0p35_W0p55 are the only REAL_DEV devices with included curves whose denominator is
below 1e-3 of the device's largest (ranges 5106 and 8409; other devices 37-342): W5 idvd@0.74 (1.2e-7 A) and
W0p55 idvd@0.37 and idvd@0.74 (4.3e-9 and 1.8e-8 A; 1 nA and 10 nA SMU steps, 8 and 3 distinct values).
The failing predictions over-predict subthreshold current: E2 cards with nfactor at the box upper edge (top curve
idvd@0.74, 79-98 % of the W5 curve sum); E4 p43 Laplace = mean of 64 draws, 2 draws above RRMS 10 (max 1.15e6,
condition 1.4e16); E4 p7 sparse card extrapolated to 8e-5 A at |Vg| 0.74 V; E4 learned target divided by a
1e-12 A predicted scale (inner MSE 7.5e6 and 1.16e8). E5 means are set by nMOS (nmos_L20_W0p64 0.049 -> 0.551).
Re-simulation reproduces all stored values; no pipeline bug.
B. nmos_L1_W3__2 (E3 tier M): truth u0 is 1.28 at 77 K and 1.10 at 110 K, above the BSIM4 u0 > 1 unit switch
(divided by 1e4), and 0.54 at 150 K. Every method's 110 K card is below the switch (lrn 0.833) and gives
RRMS 676-1060; the lrn card with only u0 set to the truth value gives 0.156. Pipeline bug in
`src/cryoml/explore/e3_temp.py` (no u0-switch guard; 3 training truths also cross); proposed patch
`out/research/explore/I2_0/proposed_e3_u0_switch.patch`, not applied.
C. Diagnostic only: rule outputs without the devices: E1, E4, E5 unchanged; E2 KILL -> INCONCLUSIVE without both
pMOS devices (S/N0 0.534, wins 6/10, C4 fails); E3 KILL -> CONTINUE without nmos_L1_W3__2 (R_O4 0.369,
CI 0.168-0.533, lrn vs pt_interp 0.278). Proposals for new pre-registrations: E3 rerun after the u0 fix; a paired
per-device statistic (geometric-mean RRMS ratio with bootstrap CI) as the decision aggregate; a data-quality and
denominator-ratio secondary score fixed from measured data only; Laplace MAP/median point predictor and a
device-level scale for the E4 learned target.

## Iteration 2, E3b (2026-10-19): E3 rerun with the u0 unit-switch fix (pre-registered, queued)

The E3 verdict KILL stands. E3b is a new probe motivated by I2-0 part B. Pre-registration
`out/research/explore/E3b/prereg.json` (commit 787e8aa, written before any E3b run): hypothesis, methods, metric
and decision rule copied verbatim from the E3 pre-registration; the rule was not changed after the I2-0
without-device diagnostic. Changes: `e3_temp.draw_truth` rejects truths with x_u0 >= switch - 0.02 at any dense
temperature (tier M: 4 truths regenerated, 7 rejected draws; tier W: 2 truths, 2 rejected draws; all other truths
identical to E3); the x_u0 fit bound is 0.02 below the switch (changes nmos_L1_W3 4.874 -> 3.635 and
nmos_L0p25_W1p6 4.213 -> 3.638); `--study E3b` clips x_u0 of evaluated cards to that bound (law and interpolation
cards stay inside the bound by construction; learned cards are not bounded) and records the counts; a truth,
fitted card or evaluated card at or above the switch makes the run INVALID; a secondary, non-deciding paired
geometric-mean ratio lrn / control with bootstrap 95% CI and win count is reported.
Smoke (`runs/smoke/tier_M`, nmos_L1_W3 and pmos_L2_W5, n_train 2, n_test 1, 7-temperature grid, law_iter 3,
1 learn seed, 2 workers, 102 s): 0 truths, 0/46 fitted, 0/210 predicted and 0/216 evaluated cards at or above
the switch, 0 clips, 4 rejected draws (2 truths regenerated); pipeline check only, not a decision input.
Queued after the existing jobs: `E3b_full_M` (decision) and `E3b_full_W` (secondary), same settings as
E3_full_M/W; estimated 15-17 min each.

## Iteration 2, I2-0 follow-up (2026-10-19): u0 fix applied

After user approval, the proposed patch from I2-0 part B was applied in commit 40545e0
(`src/cryoml/explore/e3_temp.py`: truth rejection at the BSIM4 u0 switch, x_u0 fit bound 0.02 below it, helpers;
`scripts/research/explore/e3_temp_misspec.py --study E3b`; 14 unit tests). The E3b pre-registration is commit
787e8aa. The I2-0 part-C values (devices removed after the fact) are diagnostic only; no stored verdict changed.

## Iteration 2, I2-1 (2026-10-19 22:25): correction trained jointly with the training cards

Queue jobs `i2_1_real` (17:59-18:35, 2152 s), `i2_1_gf180` (18:35-20:24, 6534 s), `i2_1_ihp` (20:24-22:25, 7246 s),
`scripts/research/explore/i2_1_joint.py`, pre-registration `out/research/explore/I2_1/prereg.json` (commit 97127c9).
REAL_DEV leave-one-device-out, K = 3 rounds, held-out card refit with the frozen correction; 0 failed devices in
every arm. All-device (combined): C0_refit 0.1072 (0.0981), M_post 0.0871 (0.0842), M_joint 0.0903 (0.0884),
T1_joint 0.0803 (0.0744), ORACLE_joint (in-sample) 0.0368 (0.0344).
Checks: M_joint <= 0.080 no; M_joint < M_post on 6/12 (>= 8 no); M_joint <= T1_joint + 0.005 no; 0 failed yes: KILL.
T1_joint < C0 on 11/12; M_joint < C0 on 7/12.
Round diagnostics (not decision inputs): training-device RRMS with g_r on c_r, mean over the 12 folds, 0.0609 ->
0.0583 -> 0.0562 -> 0.0541 (fold nmos_L0p15_W1p6: 0.0629 -> 0.0580); post-hoc held-out RRMS of g_r on the starting
card 0.1276 -> 0.1501; held-out after the refit with g_0..g_3 0.0871, 0.0893, 0.0914, 0.0903.
Simulated (secondary), tier c leave-one-geometry-out, 48 targets each: GF180 C0 0.0457, M_post 0.0389, M_joint
0.0381 (ratio 0.834, check_S pass; M_joint < M_post on 37/48); IHP C0 0.1011, M_post 0.1019, M_joint 0.0915
(ratio 0.904, check_S fail; 27/48). Identifiability: g correlates most with the injected series-resistance
component (r 0.35/0.32 GF180, 0.20/0.27 IHP for M_post/M_joint); OLS R^2 on the four components 0.18/0.21 (GF180),
0.04/0.07 (IHP). I2-1 is stopped.
Report batch: `out/research/report/2026-10-20_explore_i2_e3b/` (08-14).

## Iteration 2, E3b (2026-10-19 22:57): E3 rerun with the u0 fix (simulated only)

Queue jobs `E3b_full_M` (22:25-22:40, 910 s, decision) and `E3b_full_W` (22:40-22:57, 989 s, secondary).
u0 check valid on both tiers: 0/108 truths, 0/1332 fitted, 0/4032 predicted and 0/4140 evaluated cards at or above
the switch; 0 clipped cards; tier M 4 truths regenerated (7 rejected draws), tier W 2 (2).
Tier M, 36 simulated test devices, held-out 110/190/260 K: O4 lrn 0.2778, par3 0.4197, pt_interp 0.3689, floor
0.0135; R_O4 0.338 [0.160, 0.495], lrn vs pt_interp 0.247, 0 failed cells: O4 meets all conditions. O2 lrn 0.4945,
par3_pop 0.6052, pt_interp 0.9579; R_O2 0.183 [0.019, 0.281]. Rule output CONTINUE.
Secondary (not decisive): lrn vs lrn_mean O4 0.033 [-0.154, 0.161], O2 0.375 [0.031, 0.533]; lrn vs lrn_noH O4
0.008, O2 -0.013; gap closed to floor O4 0.349, O2 0.187; paired geometric-mean ratio lrn/control O4 0.639
[0.516, 0.793] (29/36), O2 0.899 [0.784, 1.035] (23/36). Per temperature (O4 lrn / par3): 110 K 0.366 / 0.684,
190 K 0.395 / 0.375, 260 K 0.072 / 0.200. lrn_param has 1 failed cell of 108.
Tier W (secondary): R_O4 -0.667 (lrn 0.1139 vs par3 0.0666, 35 devices, 1 failed lrn cell), R_O2 0.333 [-0.034,
0.462]: INCONCLUSIVE; the E3 pre-registration expected the parametric law to win on tier W.
E3 vs E3b, tier M O4: lrn 9.801 -> 0.278, par3 9.123 -> 0.420. The E3 KILL stands; E3b is a separate
pre-registered probe. Simulated devices only; no measured multi-temperature I-V exists. In O4 lrn does not
separate from lrn_mean; a continuation needs a law-shift test.
Report batch: `out/research/report/2026-10-20_explore_i2_e3b/` (15-19).

## Temperature benchmark table complete (2026-10-19 23:13)

`temp_sur_fno_r` (22:56-23:13, 1002 s, rc 0) loaded the FNO checkpoint trained in the killed job `temp_sur_fno`
(25 epochs, 14402 s, rc 137 after training), ran the held-out emulator and validation passes with the --eval-batch
memory fix (commit 168cf9f) and completed sets 0-2. `temp_table` wrote `out/research/temp/table.json`.
Held-out all-device raw / + FD (simulated, well-specified, 3 sets x 18 devices, 0/54 failed records per row):
direct 0.0362 / 0.0117; direct MLP 0.0707 / 0.0006; NGSpice search 0.0021 / 0.0004; surrogate paper MLP
0.0081 / 0.0007, ResMLP 0.0105 / 0.0003, DeepONet 0.0120 / 0.0022, FNO 0.0227 / 0.0012.
Report batch: `out/research/report/2026-10-20_explore_i2_e3b/20_table_temp_benchmark.png`.

## Exploration iteration 2 complete (2026-10-20)

I2-0 diagnosis complete (u0 fix applied); I2-1 KILL; E3b CONTINUE on simulated tier M (tier W INCONCLUSIVE).
Iteration 3 is being designed.

## Iteration 3, I3-2 (2026-10-20 01:55): bias-only correction with the held-out card refit in the loop

Pre-registration `out/research/explore/I3_2/prereg.json` (commit f7f6378); code commit 4c7c66b
(`src/cryoml/explore/i3_2_bias.py`, `scripts/research/explore/i3_2_bias.py`). Queue jobs `I3_2_real` (00:44-00:52,
477 s), `I3_2_gf180` (00:52-01:22, 1820 s), `I3_2_ihp` (01:22-01:55, 1944 s), `I3_2_decide` (01:55).
Correction I_corr = I_sim · exp(g), g = mean over 5 seeds of tanh(MLP 3-16-16-1 of is_pmos, |Vgs|/VDD, |Vds|/VDD),
trained post-hoc on the E1 starting-card records of the training devices (loss = mean paper RRMS + 1e-4 L2, Adam
3e-3, 2000 steps); held-out 43-parameter card refit from direct43_r3_fd with g frozen in the loop (max_nfev 400,
irls 2), strict frozen-inclusion RRMS.
REAL_DEV leave-one-device-out, all-device (combined), 0 failed in every arm: C0_refit 0.1072 (0.0981), M_bias
0.0806 (0.0749), M_bias_geom 0.0914 (0.0879), T1_inloop 0.0820 (0.0760). R(M_bias, C0_refit) 0.769 [0.666, 0.886],
11/12; R(M_bias, T1_inloop) 0.968 [0.920, 1.011], 7/12; R(M_bias_geom, M_bias) 1.163 [0.992, 1.413], 4/12.
C0_refit and T1_inloop reruns reproduce the stored rows (max |difference| 0).
Simulated tier c, 48 targets each: GF180 C0_refit 0.0457, M_bias 0.0399, T1_inloop 0.0391, R 0.880 [0.837, 0.921],
42/48; IHP C0_refit 0.1011, M_bias 0.0871, T1_inloop 0.0921, R 0.885 [0.789, 1.005], 32/48; 0 failed.
Checks: R1 met, R2 met (0.08055 <= 0.08698), R3 met, R4 met (GF180 and IHP R <= 0.90). Rule output CONTINUE
(`out/research/explore/I3_2/decision.json`). REAL_DEV values are development evidence (REAL_DEV reused by E1, I2-1
and the I3-2 design; the input set was chosen after I2-1). M_bias does not separate from T1_inloop. The full
12-device M_bias is written as Verilog-A text (207 lines, not compiled). REAL_TEST not loaded.
Report batch: `out/research/report/2026-10-20_i3_2_bias/` (process diagram, decision, results, per device, g
surface, I-V of pmos_L0p35_W0p55, pmos_L2_W5, nmos_L20_W0p64, caveats); site claim `i3-2`.

## I3-2 follow-up F1/F2 (2026-10-20): pre-registered, code and smoke done, queued

Pre-registration `out/research/explore/I3_2/followup_prereg.json` (commit 4ae726a), amendment
`followup_prereg_amendment_2026-10-20.json` (commit 4f4e7ef), code commit 5fb08d6. The follow-ups do not change
the I3-2 decision. F1: compile the exported M_bias Verilog-A (12 fold models + full model) with OpenVAF-Reloaded
and run it around the BSIM4 instance in NGSpice. ngspice-41 arm64 has no OSDI (configured without it), so F1 uses
ngspice-46 and first compares ngspice-46 with ngspice-41 on the uncorrected cards. The exported sense line
`V(sense) <+ 0.0;` compiles but conducts no current in ngspice-46; F1 uses `V(sense) <+ RS * I(sense);` with
RS = 0. Smoke (2 devices): ngspice-46 = ngspice-41 exactly (difference 0.0); SPICE-native vs Python-corrected
max relative difference 9.7e-4 at default reltol 1e-3 and 1.0e-8 at reltol 1e-9; SPICE-native minus stored
M_bias RRMS <= 9.7e-6. F2: REAL_DEV seed sets {5-9, 10-14, 15-19}, widths 8 and 32, 3-16-1, g trained on
round-1 43-parameter cards; GF180/IHP tier b sets 0-2 (tier c has only sets 0-2). Queue jobs I3_2_F1,
I3_2_F2_real, I3_2_F2_gf180_b, I3_2_F2_ihp_b, I3_2_F2_summary (about 65-80 min after the I3_1 jobs).
REAL_TEST not loaded.

## Iteration 3, I3-1 (2026-10-20 02:58): hybrid temperature law, rule output KILL

Pre-registration `out/research/explore/I3_1/prereg.json` (commit c877413) and `prereg_addendum.json`; code commit
c59ce14. Queue jobs `I3_1_full_M` (01:55-02:22, 1629 s), `I3_1_full_W` (02:22-02:40), `I3_1_full_H` (02:40-02:58),
`I3_1_decide` (02:58). Simulated devices only, 36 test devices per tier, setting O4, per-device held-out RRMS =
mean over 110/190/260 K, GMR with eps 0.01 and 4000-resample bootstrap CI.
hyb = per-device par3 law + learned population residual. GMR(hyb, par3): M 0.741 [0.610, 0.901] (27/36),
W 1.739 [1.432, 2.132] (6/36), H 1.224 [1.062, 1.422] (13/36). GMR(hyb, lrn_mean): M 1.119 [0.951, 1.312],
H 1.306. GMR(hyb, pt_interp): M 0.885, H 1.763. GMR(hyb, hyb_mean): M 0.791, W 0.750, H 0.906.
Held-out all-device mean, hyb / par3 / pt_interp / lrn / lrn_mean: M 0.3086 / 0.4197 / 0.3689 / 0.2778 / 0.2872;
W 0.0935 / 0.0666 / 0.1019 / 0.1139 / 0.1507; H 0.2197 / 0.1838 / 0.1654 / 0.1572 / 0.2178.
Conditions: M1 met, M2 not met, M3 met, W1 not met, H1 not met, H2 not met, F met; KILL conditions W >= 1.50 and
H >= 1.00 hit. Rule output KILL (`out/research/explore/I3_1/runs/full/decision.json`). u0 check valid on all tiers.
hyb anchor-temperature mean RRMS 0.213 / 0.089 / 0.178 (M / W / H) vs par3 0.171 / 0.038 / 0.081.
lrn (E3b method, unchanged; tier M and W values identical to E3b): GMR vs par3 M 0.650, W 2.603, H 0.752
[0.640, 0.886]; vs lrn_mean H 0.802 [0.697, 0.919]; vs pt_interp H 1.083 [1.010, 1.164] (15/36 lower).
E3-rule output for lrn: M CONTINUE, W INCONCLUSIVE, H INCONCLUSIVE (R_O4 0.145 < 0.25; reduction vs pt_interp
0.049 < 0.10). Scaling arm (tier M, 3 / 6 / 12 draws per geometry): lrn 0.2644 / 0.2778 / 0.2468, hyb
0.3333 / 0.3086 / 0.3102. The e3b site claim carries a caveat with the tier H and W lrn values.

## I3-2 follow-up F1/F2 (2026-10-20 03:49): results

Queue jobs `I3_2_F1` (02:58-02:59), `I3_2_F2_real` (02:59-03:15), `I3_2_F2_gf180_b` (03:15-03:31),
`I3_2_F2_ihp_b` (03:31-03:49), `I3_2_F2_summary` (03:49). The I3-2 rule output (CONTINUE) is unchanged.
F1.0 pass: 13 models compile; g from the emitted lines matches Python within 5.3e-11.
F1.2: SPICE-native (ngspice-46 + OSDI) vs Python correction on ngspice-46, per-device |dRRMS| <= 1.4e-5 on 12
devices (<= 1.4e-9 at reltol 1e-9); max relative current difference 2.4e-3 > 1e-3 on 3 pMOS devices, so F1.2 is
recorded as fail; 0 convergence messages.
F1.1 and F1.3 fail: ngspice-46 and ngspice-41 differ for the uncorrected cards on pmos_L0p35_W0p55 (published card
RRMS 2.6351 vs 0.7270) and pmos_L8_W5 (0.8670 vs 0.3146); on these two devices ngspice-46 returns identical
currents for the published, starting, C0_refit and M_bias refit cards. The other 10 devices agree exactly.
SPICE-native all-device 0.3205 vs stored M_bias 0.0806. F1b (OSDI-enabled ngspice-41) is in progress.
F2 REAL_DEV, all-device (combined), GMR vs C0_refit: seeds 5-9 0.0805 (0.0748) 0.771; seeds 10-14 0.0813 (0.0754)
0.779; seeds 15-19 0.0804 (0.0748) 0.771; width 8 0.0805 (0.0748) 0.759; width 32 0.0814 (0.0755) 0.777; 3-16-1
0.0804 (0.0747) 0.770; round-1 training cards 0.0828 (0.0769) 0.832 [0.710, 0.996]. All CI upper bounds < 1,
0 failed; seed, capacity and training-card checks pass. GMR vs I3-2 M_bias 0.988-1.083, every CI includes 1.
F2(d) tier b, 48 targets: GMR(M_bias, C0_refit) GF180 0.996 [0.986, 1.006] (26/48), IHP 0.985 [0.975, 0.995]
(28/48); R4 rule not met on either PDK: F2(d) fails. All-target means M_bias / C0_refit: GF180 0.0133 / 0.0133,
IHP 0.0138 / 0.0141; seven-parameter refit floors (3 devices, set 0): GF180 0.0136, IHP 0.0146.
Report batch: `out/research/report/2026-10-20_i3_1_i3_2f/` (`scripts/research/plot_explore_i3_1_i3_2f.py`); site
claims `i3-1` (refuted, new), `e3b` (caveat added), `i3-2` (follow-up section, status unchanged).

## I3-2 follow-up F1b/F1c (2026-10-20 06:00): SPICE-native runs on ngspice-41

Simulator: ngspice-41 rebuilt from the same source with `--enable-osdi` (otherwise the canonical configure flags;
`make CXXFLAGS="-O2 -Wno-invalid-specialization"`), build dir `~/cryo-ng41/src/ngspice-41/build-arm64-osdi`,
prefix `~/cryo-ng41/arm64-osdi`. The canonical build `~/cryo-ng41/arm64` is unchanged. Reproduction:
`scripts/verify_simulator.py` per-device output identical to the canonical file
(`out/tables/simulator_verification_arm64_ng41_osdi.json`); published card, 18 devices, 36828 of 36828 currents
bit-identical to the canonical binary (`scripts/research/compare_ngspice_bins.py`,
`out/tables/simulator_compare_ng41_arm64_vs_arm64_osdi.json`).
F1b (amendment `followup_prereg_amendment_F1b_2026-10-20.json`): F1.0 pass, F1.1 pass (max relative difference
0.0, max |dRRMS| 0.0). F1.2 and F1.3 fail: ngspice-41 accepts only OSDI 0.3 and OpenVAF-Reloaded v24 emits OSDI
0.4, so no .osdi file loaded and every SPICE-native current is NaN. The first F1b run reported F1.3 as pass with
NaN deltas; the check code now counts a non-finite statistic as a failure, and F1b was run again.
F1c (amendment `followup_prereg_amendment_F1c_2026-10-20.json`): F1b with OpenVAF 23.5.0 (OSDI 0.3) built from
source at `~/cryo-ng41/openvaf` (rust 1.64, Homebrew llvm@15; two link-step patches recorded in the amendment).
F1.0 pass. F1.1 pass (OSDI build vs canonical ngspice-41: max relative current difference 0.0, max |dRRMS| 0.0,
stored values reproduced with difference 0.0). F1.2 fail: max relative difference SPICE-native vs Python on the
same simulator 2.45e-3 (fold, pmos_L0p5_W0p42, idvg@1.85 at 2.0e-12 A) and 1.43e-3 (full model) > 1e-3;
pmos_L0p35_W0p55 1.23e-3 and pmos_L8_W1p6 1.45e-3 also exceed 1e-3; the other 9 devices are below 1e-3;
max per-device |dRRMS| 1.24e-5; 0 convergence messages, 0 non-finite currents. F1.3 pass: max per-device
|native - stored M_bias| 1.24e-5 (limit 2e-4), all-device |delta| 5.3e-6 (limit 1e-4).
SPICE-native all-device (combined): 0.080555 (0.074863) vs stored M_bias 0.080550 (0.074858); C0_refit card
0.107187 (0.098051); published card 0.284764 (0.260833). The two pMOS devices that failed in F1 now give
pmos_L0p35_W0p55 0.349646 (stored 0.349639) and pmos_L8_W5 0.042485 (stored 0.042479).
Figures: `out/research/explore/I3_2/followup/f1c_iv_{pmos_L0p35_W0p55,pmos_L2_W5,nmos_L20_W0p64}.png`.
Exporter: `to_verilog_a_bias` now emits `V(sense) <+ RS * I(sense);` with `parameter real RS = 0.0;` and names
the polarity parameter `is_pmos` (`pmos` is a reserved Verilog-AMS keyword); `legacy=True` reproduces the stored
I3-2 text. Corrected export `out/research/explore/I3_2/real_dev/m_bias_sky130_dev_full_sense_rs.va`: 0 compiler
warnings with OpenVAF-Reloaded v24 and OpenVAF 23.5.0; in the ngspice-41 OSDI build its currents equal the F1c
full-model SPICE-native currents (max relative difference 0.0 on pmos_L2_W5, nmos_L20_W0p64, pmos_L0p35_W0p55).
The stored `m_bias_sky130_dev_full.va` is unchanged.

## I4-1 baselines for I3-2, I4-2 interpretation (2026-10-20 06:15): pre-registered, smoke run, queued

Pre-registration `out/research/explore/I4_1/prereg.json` (commit e57afce, committed alone before any B1-B5
held-out result); amendment `prereg_amendment_2026-10-20.json` corrects one parenthetical (REAL_DEV base-box TRF
budget per IRLS round is 4 or 5, not 4; rule unchanged). Code `src/cryoml/explore/i4_1_baselines.py`,
driver `scripts/research/explore/i4_1_baselines.py`, tests `tests/test_i4_1_baselines.py` (13 tests).
`spice_pdk.simulate_pdk` has a new `rs_ext_ohm` argument (default 0 gives the same deck text as before).
Arms, all with the held-out card refit in the loop and the I3-2 folds and settings: B1 wider card parameter set;
B2 extrinsic series resistance per side (netlist resistors) plus a self-heating factor
exp(kappa |Vds I| / P_max), both fitted per device; B3_d2 / B3_d3 shared polynomial g of total degree 2 / 3 in
(|Vgs|, |Vds|) per polarity; B4 GP posterior mean of ln|I_meas / I_sim| on (is_pmos, |Vgs|/VDD, |Vds|/VDD),
ARD squared-exponential kernel plus white noise, type-II ML; B5 per-device M_bias-form MLP fitted on the
held-out device (in-sample reference). B1 and B2 get the same TRF iterations per IRLS round as C0_refit.
B1 extras from an edge screen of the starting cards (`b1_screen.json`; response >= 1e-4 at a box edge):
sky130 19 (a1, alpha1, cdsc, dvt0w, dvtp0, etab, fprout, minv, pdiblc1, pdits, phin, pscbe1, pscbe2, pvag, rbpb,
rdswmin, ud, up, voffl), GF180 35, IHP 71. GF180 B1 contains 9 of the 12 parameters the tier c truth perturbs and
IHP B1 all 9; the tier c truths also contain an extrinsic resistance of the B2 form.
Rule: for each X in {B1, B2, B3_d2, B3_d3, B4}, X is beaten if the REAL_DEV bootstrap 95% CI upper bound of
GMR(M_bias, X) is < 1, or GMR(M_bias, X) < 0.95 on both GF180 and IHP; "supported against baselines" if all are
beaten, otherwise "matched by baseline X" with the list.
Smoke (1 fold; B3/B5 200 steps, B5 1 seed; not used for any conclusion). REAL_DEV nmos_L0p15_W1p6 (45 s):
C0_refit_rerun 0.019338 (stored 0.019338, |diff| 0.0), B1 0.019341, B2 0.019341, B3_d2 0.018626, B3_d3 0.018350,
B4 0.036776, B5 0.015044, M_bias (stored) 0.023513. IHP geometry nmos_L0p13_W0p2, 3 targets (133 s):
all-device C0_refit_rerun 0.19533 (|diff| 0.0 vs stored), B1 0.04315, B2 0.05341, B3_d2 0.16173,
B3_d3 0.15981, B4 0.14632, B5 0.05399, M_bias (stored) 0.15920. B1 and B2 start from the C0 start current
(first objective equal to C0's).
Estimated wall time: real 10 min, GF180 30 min, IHP 42 min, decide and analyze 1 min (about 85 min).
Queue: `I4_1_real`, `I4_1_gf180`, `I4_1_ihp`, `I4_1_decide`, `I4_2_analyze`.
