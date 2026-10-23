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

## I4-1 results, I4-2 analysis, report batch (2026-10-20 07:16)

Queue jobs `I4_1_real` (06:14-06:22, 485 s), `I4_1_gf180` (06:22-06:42, 1172 s), `I4_1_ihp` (06:42-07:16,
2065 s), `I4_1_decide` (07:16), `I4_2_analyze` (07:16, 1 s). C0_refit_rerun reproduces the stored C0_refit rows
(max |difference| 0.0) on all three datasets; 0 failed devices or targets in every arm.
REAL_DEV all-device (combined): C0_refit 0.1072 (0.0981), B1 0.1069 (0.0978), B2 0.1063 (0.0973),
B3_d2 0.1034 (0.0948), B3_d3 0.0956 (0.0879), B4 0.1317 (0.1199), M_bias 0.0806 (0.0749), T1_inloop 0.0820
(0.0760), B5 (in-sample) 0.0410 (0.0386).
GMR(M_bias, X) on REAL_DEV [95% CI], wins: B1 0.770 [0.667, 0.887] 11/12; B2 0.773 [0.670, 0.891] 11/12;
B3_d2 0.789 [0.684, 0.907] 11/12; B3_d3 0.844 [0.751, 0.949] 11/12; B4 0.658 [0.581, 0.753] 12/12.
GF180 / IHP GMR(M_bias, X): B1 0.897 / 1.644; B2 1.333 / 1.806; B3_d2 0.994 / 0.986; B3_d3 0.997 / 0.985;
B4 0.954 / 0.848. All-target means GF180 / IHP: C0_refit 0.0457 / 0.1011, B1 0.0454 / 0.0403, B2 0.0272 /
0.0381, M_bias 0.0399 / 0.0871, B5 0.0239 / 0.0274. The pre-registration states that the tier c truths contain an
extrinsic resistance of the B2 form and that B1 contains 9/12 (GF180) and 9/9 (IHP) truth-perturbed parameters.
Rule: every baseline is beaten through the REAL_DEV condition (CI upper bound < 1); the simulated condition
(GMR < 0.95 on both PDKs) is not met for any baseline. Rule output: supported against baselines
(`out/research/explore/I4_1/decision.json`), development evidence (REAL_DEV reused). REAL_TEST not loaded.
Fraction of the C0_refit -> B5 gap closed by M_bias (means / log GMR): REAL_DEV 0.403 / 0.298, GF180
0.266 / 0.238, IHP 0.191 / 0.121.
I4-2 (diagnostic): g fold spread (mean std over the grid) 0.003-0.011; REAL_DEV pMOS linear region mean g -0.211;
curve-mean g vs T1 per-tag gain Pearson r 0.965 (REAL_DEV), 0.871 (GF180), 0.842 (IHP); by curve kind Id-Vd /
Id-Vg 0.565 / 0.970, 0.966 / 0.660, 0.983 / 0.435.
Report batch `out/research/report/2026-10-20_i4/` (`scripts/research/plot_explore_i4.py`; recomputes every
aggregate, paired ratio and the rule output from the stored per-device values): I4-1 decision, forest plot,
per-dataset tables, truth/B1/B2 table, B5 gap table, REAL_DEV per device; I4-2 g surfaces, analysis table,
per-tag T1 table, T1 scatter; I3-2 F1b/F1c checks, F1c per device, F1c I-V figures. The F1 (ngspice-46) I-V
figures `out/research/explore/I3_2/followup/f1_iv_*.png` are committed and labelled as F1 in the batch README.
Site: new claims `i4-1` (in progress, "supported against baselines (development set)") and `i4-2` (verified,
diagnostic); claim `i3-2` follow-up section updated with F1b/F1c and the exporter fix, the ngspice-46 caveat
replaced; Exploration status block updated. Site rebuilt, not published.

## I5-1: I3-2 bias-only correction on IHP SG13G2 measured MOSFETs (2026-10-20 07:55)

Pre-registration `out/research/explore/I5_1/prereg.json` (committed alone before any card fit, 9267a90).
Data: IHP-Open-PDK `ihp-sg13g2/libs.doc/meas/MOS` (commit 5e6d592e, Apache-2.0, IC-CAP MDM text, read as
text under `python -I`): 28 DUTs (LV 8 nMOS + 8 pMOS on the same 8 W/L; HV 6 + 6), 233/300/343/398 K,
dc_idvg (3 Vd x 5 or 7 Vb), dc_idvd and dc_idvd_vth (Vb = 0 and Vbmin). Dataset
`out/research/explore/I5_1/data/ihp_meas.npz` (not committed) and `manifest.json` (committed).
Fitted curves: the 11 Vb = 0 curves per DUT and temperature (idvg@d0..d2, idvd@g0..g4, idvd@t0..t2, keys by
position). Inclusion: pdk2 target-frozen rule (trim at the first |I| > 1e-10 A; include iff trimmed mean |I| >
3e-7 A Id-Vd / 2e-7 A Id-Vg): 946 of 1232 curves, 3 to 10 per DUT and temperature. Quantization: up to 6
significant digits, smallest nonzero |id| 2e-14 A; largest |id| 0.061 of the 0.1 A drain compliance; off-current
floor 2e-14 to 6.3e-12 A.
Simulator: ngspice-46 + psp103.osdi, PDK .model block as a top-level card (LV and HV), header AD/AS/PD/PS, bulk
source, header temperature; matches the PDK subcircuit through cornerMOS{lv,hv}.lib mos_tt within 4.4e-8
relative on all 28 x 4 x 11 curves. Stock-card RRMS (no fit), all-device mean DEV / TEST / all 28:
233 K 0.348 / 0.117 / 0.265; 300 K 0.190 / 0.119 / 0.165; 343 K 0.153 / 0.127 / 0.144; 398 K 0.135 / 0.140 /
0.137. HV nMOS W10/L0.5 is 1.60 at 300 K (near-threshold Id-Vd curves). I-V check plots of 6 DEV DUTs:
`out/research/explore/I5_1/stock/figs/`.
Split (seed 20261020; LV by geometry, both polarities; HV 2 of 6 per polarity): DEV 18 (LV 10, HV 8),
TEST 10, `out/research/explore/I5_1/split.json`; the fitting modes read only DEV.
Cards: 12-parameter pdk2 IHP wide set; box VFBO/DPHIBO/DPHIBL stock +/- 0.2 V, others log |v|/2..2|v| (HV pMOS
CTL = 0: [0, 0.05]); starting cards = polish from stock, max_nfev 3000, irls 3. Arms per temperature (LODO over
DEV, training within class, VDD 1.2 V LV / 3.3 V HV): C0_refit, M_bias (I3-2 recipe unchanged), T1_inloop, B1
(screened PSP extras), B2, B3_d3, B5; X300 cross-temperature arm (300 K fold g applied at 233/343/398 K).
Rule at 300 K over 18 DEV DUTs: R1 CI upper of GMR(M_bias, C0_refit) < 1; R2 M_bias mean <= T1_inloop mean +
0.005; R3 0 failed -> "replicated"; baselines B1, B2, B3_d3 beaten if CI upper of GMR(M_bias, X) < 1 ->
"supported against baselines".
Smoke (300 K, start max_nfev 100, 2 folds, 1 seed, 200 steps; not used for any conclusion): start 0.190 ->
0.042 (300 K DEV); folds lv_nmos_W0p15_L0p18 / hv_nmos_W0p3_L0p5: C0_refit 0.0313 / 0.0181, M_bias 0.0337 /
0.0221, T1_inloop 0.0314 / 0.0525, B1 0.0171 / 0.0121, B2 0.0200 / 0.0105, B3_d3 0.0341 / 0.0217, B5 0.0105 /
0.0228; 70 s. Cross and decide smoke ran. Simulation 0.03 s per card.
Estimated wall time: start 18 min, screen 1 min, lodo 4 x 10 min, cross 3 min, decide/figs 1 min (about 65 min).
Queue: `I5_1_start`, `I5_1_screen`, `I5_1_lodo_300`, `I5_1_lodo_233`, `I5_1_lodo_343`, `I5_1_lodo_398`,
`I5_1_cross`, `I5_1_decide`, `I5_1_figs`.

## I5-1 outcome (2026-10-20 08:32): IHP SG13G2 real data, not replicated

Pre-registered rule (out/research/explore/I5_1/prereg.json), 300 K, 18 DEV DUTs, LODO. R1 fails: GMR(M_bias, C0_refit) 1.214 [1.008, 1.537], 9/18 lower. R2 and R3 hold. All-device means: C0_refit 0.0191, M_bias 0.0203, T1_inloop 0.0215, B1 0.0160, B2 0.0176, B3_d3 0.0214, B5 (in-sample) 0.0054. Baseline verdict: matched by B1, B2. Secondary temperatures (233, 343, 398 K) also not replicated. The starting PSP103 fits here are about 5x lower than sky130 77 K C0_refit (0.107). Report batch and site update pending.

## I5-1 report batch and site (2026-10-20 10:42)

Report batch `out/research/report/2026-10-20_i5_1/` (`scripts/research/plot_explore_i5_1.py`, stored files only;
recomputes every aggregate, paired ratio, per-class ratio, X300 ratio and the R1-R3 / baseline rule outputs from
the stored per-DUT values and checks them against `lodo_*K/results.json`, `cross/results.json` and
`decision.json`). Contents: dataset table (28 DUTs, split, included curves per temperature: DEV 146/143/142/140
of 198 at 233/300/343/398 K), stock vs starting-card RRMS (DEV 300 K 0.1899 -> 0.0218, C0_refit 0.0191),
decision per temperature, results per temperature for every arm, LV/HV class split, forest plot, per-DUT
300 K, per-DUT ratio to C0_refit by temperature, X300 table, sky130 77 K vs IHP comparison, four I-V figures.
LV/HV at 300 K, GMR(M_bias, C0_refit): LV 1.192 [0.938, 1.712] 6/10, HV 1.244 [0.991, 1.643] 3/8; GMR(B1,
C0_refit) LV 0.821, HV 0.721. sky130 77 K vs IHP 300 K: C0_refit 0.1072 vs 0.0191, GMR(M_bias, C0_refit) 0.769
vs 1.214, fraction of the C0_refit -> B5 gap closed by M_bias (means / log GMR) 0.403 / 0.298 vs -0.090 /
-0.185. X300 GMR vs C0_refit: 233 K 1.116, 343 K 1.197, 398 K 1.341.
I-V figures: the stored `figs/iv_*_300K.png` contain measured points, the C0_refit card and the M_bias refit
card x exp(g), not the stock-card curve. The batch figures 11-13 stack the stored stock-card figure
(`stock/figs/stock_*_300K.png`) above the refit figure; hv_pmos_W10_L0p35 (figure 14) has no stored stock-card
figure. No simulation was run for the batch.
Site: new claim `i5-1` (refuted, "not replicated on IHP SG13G2 room-temperature data"); caveats linking i5-1
added to claims `i3-2` and `i4-1`; Exploration status block updated. Site rebuilt, not published.

## I6-1 pre-registration and pipeline (2026-10-20 12:10): TU Delft 40-nm CMOS, 4.2 K vs 295 K

Question: does the I3-2 bias-only correction lower held-out error where the compact model has a model-form error a
local card refit cannot absorb (BSIM4 at 4.2 K), and not where the card fits (same devices at 295 K)?
Pre-registration `out/research/explore/I6_1/prereg.json` (commit a967f04, before any correction fit and before the
starting-card fits); amendment `prereg_amendment_2026-10-20.json` (after the smoke runs, before any full run): DVT0W
removed from the B1 universe (DVT1W = 0 in the card makes the BSIM4 narrow-width term 0/0; DVT0W > 0 gave no DC
solution), simulator timeout 10 s per ngspice process.
Data: doi:10.4121/c41c26f7-586f-48ca-8abe-9127b2d97c60.v1 (CC0), I3-1 download (results/dc, 16.3 MB), read with an
unpickler restricted to numpy reconstruction globals. Campaign 1, 69 low-Vt devices with files at 4.13-4.14 K and
295.0 K stage temperature; 10 (polarity, W, L) groups of 2-8 copies (nMOS die 3, pMOS die 5). Per device and
temperature 12 Id-Vg (Vd = 0.05 V and 11 values to 1.1 V) and 11 Id-Vd (Vd 0-1.1 V), 111 points; Vg starts at
0.45 V (nMOS 4 K), 0.30 V (nMOS 295 K), -0.60 / -0.40 V (pMOS). Float64 currents, smallest nonzero |Id| 4.9e-14 A,
no stored compliance, largest |Id| 0.87 mA, no clipped curve. Inclusion (pdk2 rule, measured only): 3161 of 3174
curves. Manifest `out/research/explore/I6_1/data/manifest.json`.
Card: BSIM4 4.5.0 N1/P1 from ngspice-41 `examples/mos/ro_17_4.cir` (CMOSedu 50 nm, VDD 1 V; ngspice Modified BSD),
copied verbatim to `I6_1/card/`; temperature coefficients (kt1, kt1l, kt2, ute, ua1, ub1, uc1, at, prt) set to 0;
deck temperature = stage temperature; one card fitted per temperature. PTM 45 nm not used (host does not resolve,
no license statement found); the ngspice BSIM4 QA card has xl = -30 nm (non-positive Leff at L = 40 nm).
Simulator: ngspice-41 arm64, 0.03 s per device and card (23 sweeps).
Fitted set: the 43-parameter sky130 set; box VTH0, VFB +/- 0.5 V, VOFF +/- 0.3 V, other nonzero values log
|v|/10..10|v| (chosen after a 3-device trial, `box_trial.json`), CIT, CDSCD, B0, B1 zero ranges; 36 (nMOS) / 37
(pMOS) active (UB inactive under the ParameterBox rule, as in sky130).
Split (seed 20261020, by W/L pair): TEST = W/L 4/4 (both polarities) and nMOS 1/0.36, 24 devices; DEV 45 devices in
7 groups (nMOS 0.12/0.04, 0.12/0.05, 0.36/0.04, 1/0.04, 1/1; pMOS 1/0.04, 1/1).
DEV stock-card RRMS: 4 K 5.105 (median 5.344), 295 K 7.264 (median 8.639).
Starting cards (queue job I6_1_start, polish max_nfev 3000, irls 3; 1083 s, 0 simulator timeouts): 4 K 0.0628
(median 0.0437, combined 0.0553), 295 K 0.0271 (median 0.0227, combined 0.0253); per group 4 K 0.028-0.156,
295 K 0.015-0.046.
Arms per temperature, leave-one-geometry-out over the 7 DEV groups (training on the other 6 groups, both
polarities, VDD 1.1 V): C0_refit, M_bias (I3-2 recipe unchanged), T1_inloop, B1, B2, B3_d3, B5.
Rule: at 4 K R1 (device-bootstrap CI upper of GMR(M_bias, C0_refit) < 1), R2 (M_bias mean <= T1_inloop + 0.005),
R3 (0 failed) and every B1, B2, B3_d3 beaten (CI upper of GMR(M_bias, X) < 1); at 295 K R1 fails; premise P:
C0_refit mean at 4 K >= 1.5 x at 295 K; contrast K = GMR of (M_bias/C0_refit at 4 K) / (same at 295 K).
Outcomes: refuted (4 K R1 fails), inconclusive (4 K R2 or R3 fails), matched by baselines, supported, supported
with premise not met, temperature-graded (R1 at both, K CI upper < 1), not temperature-specific (R1 at both, K
fails). Group-cluster bootstrap CIs reported, not deciding.
Smoke (one device per DEV group, start max_nfev 100, 2 folds, 1 seed, 200 steps; 107 s; not used for any
conclusion): runs end to end through decide and figs.
Queue: I6_1_start (done), I6_1_screen, I6_1_lodo_4, I6_1_lodo_295, I6_1_decide, I6_1_figs. Estimated remaining
wall time about 60-75 min (screen 2 min, each LODO 25-35 min).

## I6-1 outcome (2026-10-20 12:47): TU Delft 40 nm, 4.13 K vs 295.0 K, refuted

Pre-registered rule (out/research/explore/I6_1/prereg.json and amendment), 45 DEV devices in 7 geometry groups,
leave-one-geometry-out, CMOSedu 50-nm BSIM4 example card refit per device, ngspice-41. Outputs committed in 74f4fc4
(B1 screen: the same 16 extras at both temperatures; LODO 4 K 1201 s, 295 K 1185 s; 0 simulator timeouts; 0 failed
devices in every arm).
4.13 K: R1 not met: GMR(M_bias, C0_refit) 0.939 [0.877, 1.004], cluster [0.811, 1.081], 27/45 lower. R2 and R3 met.
All-device: C0_refit 0.0575, M_bias 0.0533, T1_inloop 0.0534, B1 0.0612, B2 0.0582, B3_d3 0.0539, B5 (in-sample)
0.0279. GMR(M_bias, X): B1 0.885 [0.819, 0.955] (beaten), B2 0.954 [0.871, 1.054], B3_d3 0.983 [0.956, 1.015] (not
beaten). M_bias is above C0_refit in both pMOS groups and below it in the 5 nMOS groups.
295.0 K: GMR(M_bias, C0_refit) 0.797 [0.688, 0.935], cluster [0.576, 1.087], 35/45; C0_refit 0.0255, M_bias 0.0196,
T1_inloop 0.0233, B1 0.0268, B2 0.0264, B3_d3 0.0215, B5 0.0051; B1, B2, B3_d3 beaten.
Contrast K (4 K ratio over 295 K ratio) 1.177 [1.016, 1.348], cluster [0.897, 1.502]. Premise P holds (0.0575 vs
1.5 x 0.0255). Rule output: refuted (`out/research/explore/I6_1/decision.json`). DEV only; TEST not loaded.

## I6-1 report batch and site (2026-10-20 14:45)

Report batch `out/research/report/2026-10-20_i6_1/` (`scripts/research/plot_explore_i6_1.py`, stored files only;
recomputes every aggregate for all devices and per group, every paired ratio with the device and the cluster
bootstrap, K, P and the outcome label, and checks them against `lodo_{4,295}K/results.json`, `decision.json`,
`start/summary.json` and `stock/stock_rrms.json`). Contents: dataset table (10 groups, split, included curves),
card and box table (43 parameters, stock value and box per polarity, starting-fit bound hits), box trial, stock /
start / C0_refit per group, decision per temperature, outcome logic, results per arm with device and cluster CIs,
per-group table, contrast and gap table, forest plot, per-device RRMS, per-device ratio 4.13 K vs 295.0 K,
cross-dataset table, five I-V figures (stored stock-card and refit figures at both temperatures; solid C0_refit,
dashed M_bias refit card x exp(g)).
Fraction of the C0_refit -> B5 gap closed by M_bias (means / log GMR): TU Delft 4.13 K +0.142 / +0.062, 295.0 K
+0.290 / +0.149; sky130 77 K +0.403 / +0.298; IHP 300 K -0.090 / -0.185.
B1 returned the starting card unchanged for 16/45 (4.13 K) and 13/45 (295.0 K) devices and is above C0_refit on
34/45 and 33/45 devices.
Site: new claim `i6-1` (refuted; headline gives the 4.13 K and 295.0 K GMRs); caveats linking i6-1 added to claims
`i3-2`, `i4-1`, `i5-1`; Exploration status block updated with the I6-1 item and the cross-dataset table
(`build_site.py` now renders a block, such as a batch_table, as a status item). Site rebuilt, not published.

## I7-1 pre-registration, retrospective calibration and prospective queue (2026-10-20 14:55): training-set statistic for the transfer of the shared bias-only correction

Question: does a statistic S computed from the training devices of a fold only (no held-out information) predict
the held-out log gain y = ln(M_bias / C0_refit) of the I3-2 bias-only correction?
Pre-registration `out/research/explore/I7_1/prereg.json` (commit d406939, before any S value was computed; the
dataset-level outcomes of the stored datasets were known, so step 2 is retrospective and exploratory).
Residual of a training device at an included trimmed point of its starting-card record: r = clip(ln(|I_target| /
|I_sim|), -1, 1), weight w = (|I_target| / den)^2 / (n_curve points x n_curves), so sum w (r - g)^2 is the
first-order form of the M_bias loss term of the device. Common grid: 0.1-wide bins of (|Vgs|/VDD, |Vds|/VDD), 12 x
12, per polarity; shared function m = weighted bin mean.
S1_lodo (primary): 1 - sum_d E_d(m without d) / sum_d E_d(0) over the fold's training devices. Secondary:
S1_pooled (m includes d), S2 (mean same-polarity pairwise weighted uncentered cosine of bin-mean residuals), S3
(inner leave-one-group-out post-hoc gain of M_bias with 1 seed and 500 steps). S1_lodo chosen as primary because it
measures the shared part of the fitted-card residual on the M_bias inputs, uses the M_bias loss weights, estimates
transfer to an unseen device (no in-sample inflation), and needs no training or simulation.
Code: `src/cryoml/explore/i7_1_transfer.py`, `scripts/research/explore/i7_1_transfer.py` (modes retro, s3_retro,
amend, run, s3_new, decide, figs), `tests/test_i7_1_transfer.py` (12 cases).

Step 2 (retrospective; 11 stored datasets, 162 folds, fold structure and training sets as in the stored runs;
`retro/results.json`, commit 968d973). Dataset S = fold mean weighted by held-out devices; y_D = ln GMR.

| dataset | folds | held-out | S1_lodo | S1_pooled | S2 | y_D | GMR | within-dataset fold Spearman (S1_lodo, y_f) |
|---|---|---|---|---|---|---|---|---|
| sky130 77 K | 12 | 12 | +0.322 | +0.647 | +0.679 | -0.2631 | 0.769 | +0.69 |
| IHP meas 233 K | 18 | 18 | -0.314 | +0.297 | +0.228 | +0.1012 | 1.106 | +0.11 |
| IHP meas 300 K | 18 | 18 | -0.223 | +0.371 | +0.321 | +0.1943 | 1.214 | +0.30 |
| IHP meas 343 K | 18 | 18 | -0.331 | +0.336 | +0.340 | +0.1664 | 1.181 | -0.10 |
| IHP meas 398 K | 18 | 18 | -0.379 | +0.534 | +0.314 | +0.2405 | 1.272 | +0.11 |
| TU Delft 4.13 K | 7 | 45 | +0.504 | +0.531 | +0.781 | -0.0634 | 0.939 | -0.57 |
| TU Delft 295 K | 7 | 45 | +0.323 | +0.374 | +0.476 | -0.2265 | 0.797 | +0.79 |
| GF180 tier c | 16 | 48 | +0.248 | +0.307 | +0.435 | -0.1275 | 0.880 | +0.72 |
| IHP tier c | 16 | 48 | +0.224 | +0.302 | +0.373 | -0.1219 | 0.885 | +0.53 |
| GF180 tier b | 16 | 48 | +0.012 | +0.042 | +0.076 | -0.0039 | 0.996 | +0.79 |
| IHP tier b | 16 | 48 | +0.032 | +0.064 | +0.140 | -0.0148 | 0.985 | +0.94 |

Dataset-level Spearman with y_D (11 datasets): S1_lodo -0.873, S1_pooled -0.145, S2 -0.636. Pooled fold-level
Spearman (162 folds): S1_lodo -0.333, S1_pooled -0.150, S2 -0.287. Within datasets the fold-level Spearman of
S1_lodo with y_f is positive in 9 of 11 datasets: a fold whose training set has a higher S1_lodo has a smaller gain
(when a device with a device-specific residual is held out, the remaining training devices share more of their
residual, and the held-out device is the one the shared correction fits least). S1_lodo predicts at dataset level,
not per fold. TU Delft 4.13 K has the largest S1_lodo (0.504) and a small gain (0.939).
Frozen rule (OLS over the 11 datasets, amendment `prereg_amendment_2026-10-20.json`, commit 968d973, before any
step-3 data): y_D = 0.007619 - 0.485184 S1_lodo_D, R^2 0.785; predicted gain iff S1_lodo_D > 0.0157. In-sample
sign 10 of 11 (GF180 tier b wrong: S1_lodo 0.012, y -0.0039); leave-one-dataset-out sign 10 of 11. The IHP
measured temperatures share devices, as do the TU Delft temperatures; the 11 points are not independent.
S3 retrospective values are a queued job and do not change the rule.

Step 3 (prospective, pre-registered): new truth sets from the GF180 / IHP generators, 2 PDKs x 2 replicates x 5
mixing levels lambda (0, 0.25, 0.5, 0.75, 1) x 16 devices. Base = make_truth tier-a draw (seven fitted parameters
in their box, other values stock, 27 C); I_obs = I_clean exp(e_d) (1 + 0.01 N(0,1)), e_d = 0.10 (sqrt(lambda)
s_polarity + sqrt(1 - lambda) u_d), s and u_d random Fourier feature functions of (|Vgs|/VDD, |Vds|/VDD) (16
features, length scale 0.35, unit RMS on a 21 x 21 grid); levels of one (PDK, replicate) share base cards, s, u_d
and noise. Pipeline unchanged from I3-2 / F2 (starting fits wide box nfev 300 irls 3; LOGO 16 folds; M_bias 5
seeds 2000 steps; C0_refit and M_bias refits nfev 400 irls 2). Truth under `out/research/pdk2/{truth,ihp_truth}/
i7_1_r{r}_l{lambda}`; results `out/research/explore/I7_1/new/`.
Prediction: P1 Spearman(S1_lodo_D, y_D) <= -0.50 over the 20 datasets; P2 frozen-rule sign correct in >= 15 of 20;
supported (both), partially supported (one), not supported (neither). Datasets with > 4 of 16 targets left out are
excluded from P1 and counted as wrong in P2.
Smoke (4 devices, lambda 0 and 1, 1 seed, 200 steps; tiers `i7_1smoke_*`; code check only, not used for any
conclusion): GF180 r0 73 s, IHP r1 117 s; decide and S3 (50 steps, 2 folds of 3 datasets, 15 s) run end to end.
Queue (appended to `out/research/queue.txt`, runner started): I7_1_run_gf180_r0, I7_1_run_ihp_r0,
I7_1_run_gf180_r1, I7_1_run_ihp_r1, I7_1_decide, I7_1_figs, I7_1_s3_retro, I7_1_s3_new, I7_1_retro_s3,
I7_1_decide_s3, I7_1_figs_s3. Estimated wall time: run jobs 25-30 min each (about 1 h 50 min), S3 retro about
35 min, S3 new about 50 min; total about 3 h 15 min. The decision (I7_1_decide) is available after the four run
jobs; the S3 jobs add secondary values only.

## I7-2 pre-registration and pipeline (2026-10-20 15:30): hierarchical correction, few-shot adaptation on K curves

Question: can a low-dimensional per-device adjustment of the shared bias-only correction, fitted with the card on a
few curves of the held-out device, lower the error on that device's other curves? (In I3-2 / I5-1 / I6-1 the
in-sample per-device correction B5 was about 2x lower than every shared arm.)
Pre-registration `out/research/explore/I7_2/prereg.json` (commit be7d69e, before any I7-2 code was run on data).
Form: g_d = g_shared + phi . delta_d, phi = (1, a, b, a^2, ab, b^2), a = 2|Vgs|/VDD - 1, b = 2|Vds|/VDD - 1
(6 coefficients); g_shared = the stored fold M_bias corrections of I3-2 (sky130), I5-1 (IHP 300 K), I6-1 (TU Delft
4 K, 295 K), not retrained. Prior per fold: delta_j of each training device fitted on its starting record (card
fixed), N(m, S), S = 0.5 Sigma_hat + 0.5 diag + 1e-6 I. MAP on the K curves: J = mean curve RRMS + mu |z|^2,
z = whitened delta; mu chosen per fold and K on the training devices (card fixed, leave-one-out prior) from
{0, 1e-5 .. 1, inf}.
Split rule: included Id-Vg by |Vd|, Id-Vd by |Vg|; K2a (top Id-Vg, top Id-Vd), K2b (lowest Id-Vg, middle Id-Vd),
K4a (K2a + K2b), K4b (Id-Vg and Id-Vd at the 1/3 and 2/3 positions); a draw needs distinct picks and >= 2 held-out
curves. Undefined draws: 6 (IHP K4). Splits: sky130 48, IHP 66, TU Delft 180 per temperature.
Per split: start = lower-K-curve-RRMS of (box-center card, polarity median of the training starting cards in local
coordinates); base fit on the K curves (polish 1200 / irls 2, no correction); sky130 uses the round-1 43-parameter
box (center = released card), because the round-3 box center is a card fitted to all curves. Arms from the base card,
polish 400 / irls 2 on the K curves: a_C0, b_shared, c_hier (primary, joint card + z, prior residual, nfev matched
to b), c_ml (mu = 0), d_scratch (B5 recipe on the K curves), e_crossinterp (E4 cross-curve residual on a's card).
Score: strict paper RRMS on the held-out curves only.
Rule: C1 CI upper of GMR(c, b) < 1 on >= 3 of 4 datasets; C2 CI upper of GMR(c, a) < 1 on all 4; C3 GMR(c, d) < 1 on
all 4 and CI upper < 1 on >= 3; C4 c failures <= 10% and >= 8 paired devices per dataset. Outcomes in order:
invalid (C4 fails on >= 2), supported_continue (C1-C4), delta_no_gain_stop (GMR(c, b) >= 0.98 on >= 2),
sharing_no_gain (C1, C2, not C3), inconclusive.
Dry check of the mu selection (fold 1 of sky130, IHP, TU 295 K, K = 2): selected mu inf, 0.003, 1.0; the
card-fixed criterion decreases with mu on all three, so the prior mean is preferred over a K-curve fit of delta
when the training card is fitted to all curves.
Smoke (queue job I7_2_smoke, 65 s; one device per dataset, K2a only, base nfev 100, arm nfev 60, d 1 seed 200
steps; not used for any conclusion): runs end to end for all four datasets and decide. Held-out RRMS a / b / c /
d / e: sky130 nmos_L0p15_W1p6 0.669 / 0.663 / 0.660 / 0.551 / 0.203; IHP hv_nmos_W0p3_L0p5 0.164 / 0.166 / 0.165 /
0.105 / 0.156; TU 4 K d152 0.548 / 0.611 / 0.864 / 1.021 / 0.385; TU 295 K d152 0.042 / 0.148 / 0.030 / 0.027 /
0.030. 0 simulator timeouts. Simulation cost 0.022 s (sky130, IHP) and 0.014 s (TU Delft) per K-curve call.
Queue: I7_2_sky130, I7_2_ihp300, I7_2_tud4, I7_2_tud295, I7_2_decide (8 workers), after the queued I7-1 jobs.
Estimated wall time of the four runs 1.2-2 h (about 3000 simulations and two MLP fits per split); no draw reduction.

## I7-1 outcome, report batch and site (2026-10-20 17:45): S1_lodo predicts the sign of the held-out gain on 20 new simulated datasets, supported

Step 3 (prospective; queue jobs I7_1_run_gf180_r0 1847 s, I7_1_run_ihp_r0 1855 s, I7_1_run_gf180_r1 1743 s,
I7_1_run_ihp_r1 1936 s, I7_1_decide; `out/research/explore/I7_1/decision.json`): 20 datasets (GF180 / IHP x 2
replicates x lambda 0, 0.25, 0.5, 0.75, 1), 16 targets and 16 leave-one-geometry-out folds each, 0 targets left out,
0 datasets excluded.
P1: Spearman(S1_lodo_D, y_D) = -0.970 (threshold -0.50), holds. P2: frozen-rule sign correct 16/20 (required 15),
holds. Verdict: supported. Permutation p 1e-4 (one-sided, 10000 permutations); per PDK GF180 -0.988, IHP -0.976;
S1_pooled -0.974, S2 -0.979; pooled fold-level Spearman over 320 folds S1_lodo -0.751.
The 4 sign errors are all datasets the rule predicted 'no gain' that had a small gain: GF180 r0 lambda 0 (S1_lodo
-0.103, y -0.045), GF180 r1 lambda 0.25 (+0.006, -0.039), IHP r1 lambda 0 (-0.050, -0.131), IHP r1 lambda 0.25
(-0.001, -0.037).
Magnitude: prospective OLS y = -0.0102 - 1.419 S1_lodo (R^2 0.864) vs frozen slope -0.485; rule MAE 0.357. The
frozen line predicts the sign of y_D, not its magnitude.
Manipulation check: Spearman(lambda, S1_lodo_D) 0.981, Spearman(lambda, y_D) -0.956.
Fold level: within each of the 20 new datasets Spearman(S1_lodo_f, y_f) is positive (0.28 to 0.96), the opposite
sign of the dataset-level relation, as in step 2 (positive in 9 of 11 stored datasets).
Post hoc, not pre-registered: within each lambda level (4 datasets) Spearman(S1_lodo, y) is -0.80, -1.00, -0.80,
-0.80, +0.40 for lambda 0, 0.25, 0.5, 0.75, 1 (mean -0.60). After regressing S1_lodo and y on lambda as a
categorical variable the residual Pearson r is -0.590 (Spearman -0.696, slope -0.659, df 14; permutation of y
within lambda levels, 10000, p 0.0037); after lambda x PDK (10 cells of 2 replicates) r -0.751 (Spearman -0.749,
p 0.0020).
Caveat: the synthetic error (prereg step3_prospective.error_functions) is of exactly the shared-plus-device form that
S1_lodo measures, on the M_bias inputs, so step 3 checks the statistic under its own model. The evidence outside
that model is the 11 retrospective datasets (7 measured, 4 simulated tiers), whose outcomes were known when S1_lodo
was defined and the rule was fitted.
S3: pending (I7_1_s3_new running; I7_1_retro_s3, I7_1_decide_s3, I7_1_figs_s3 queued). S3 does not enter P1, P2 or
the rule.
Report batch `out/research/report/2026-10-20_i7_1/` (`scripts/research/plot_explore_i7_1.py`, stored files only;
recomputes every dataset aggregate, P1, P2, the verdict, secondary statistics and the frozen rule from the per-fold
values and checks them against the stored values): decision table, S1_lodo scatter (stored and new datasets, frozen
rule, zero crossing, prospective OLS, errors circled), new-dataset table, lambda manipulation panel, fold-level
Spearman bars, post-hoc within-lambda table and residual scatter, step-2 table.
Site: new claim `i7-1` (status in progress; headline "supported (simulated, prospective); S3 pending") in the
Exploration tab; status block gains the I7-1 item and the I7-1 decision table. Site rebuilt, not published.

## I7-1 S3 outcome (2026-10-20 18:40): secondary statistic S3 on the 20 new and the 11 stored datasets

S3 (pre-registered secondary; rank test only, expected positive correlation with y, no frozen rule; does not enter
P1, P2 or the verdict): per fold, mean over the fold's training records of ln(RRMS with correction / RRMS of the
starting card), the correction = M_bias reduced to 1 seed and 500 steps, trained on the fold's training records
minus the record's inner group, applied without a card refit. Queue jobs I7_1_s3_retro (2410 s, 1760 inner fits;
`out/research/explore/I7_1/retro/s3.json`), I7_1_s3_new (3442 s, 4800 inner fits; `new/s3.json`), I7_1_retro_s3,
I7_1_decide_s3, I7_1_figs_s3 (finished 18:31). `decision.json` and `retro/results.json` were rewritten; all
non-S3 values are unchanged (checked against the previous commit). P1, P2 and the verdict ('supported') are
unchanged.
New datasets (20): Spearman(S3_D, y_D) +0.985; OLS y = -0.0298 + 1.214 S3 (R^2 0.992); sign(S3_D) = sign(y_D) in
19/20 (exception GF180 r0 lambda 0: S3 +0.0062, y -0.0449). For comparison S1_lodo: Spearman -0.970, OLS R^2 0.864,
frozen-rule sign 16/20.
Stored datasets (11; outcomes already seen, S3 computed after the rule was frozen): Spearman +0.927; OLS y =
-0.0957 + 0.797 S3 (R^2 0.862); sign agreement 9/11 (exceptions IHP tier c: S3 +0.037, y -0.122; GF180 tier b:
S3 +0.0015, y -0.0039). S1_lodo on the same 11: Spearman -0.873, rule R^2 0.785, in-sample sign 10/11.
Fold level: new datasets pooled Spearman(S3_f, y_f) +0.736 over 320 folds (fold OLS slope +1.173, R^2 0.727);
within-dataset Spearman negative in 20/20 (-0.97 to -0.30, mean -0.81). Stored datasets pooled +0.277 over 162
folds; within negative in 9/11 (mean -0.35; positive for IHP measured 343 K and 398 K). S3 does not keep the
positive sign within a dataset: it reverses at fold level, as S1_lodo does (S1_lodo within-dataset positive in
20/20 new datasets).
TU Delft 4.13 K: residual from the frozen S1_lodo rule +0.174 (largest of the 11; RMS 0.075); residual from the
stored-data S3 OLS line +0.058 (RMS 0.060) and from the new-data S3 OLS line +0.006. Removing TU Delft 4.13 K
changes the stored-data OLS slope from -0.485 to -0.609 for S1_lodo and from 0.797 to 0.821 for S3.
Cost: S3 used 2409 s + 3442 s of queue time (about 98 min) for 31 datasets; S1_lodo, S1_pooled and S2 on the 162
stored folds take 2.1 s with no training.
Report batch `out/research/report/2026-10-20_i7_1/` rerun (`scripts/research/plot_explore_i7_1.py`; the
consistency checks now also recompute S3 per dataset from the per-fold files and check the S3 correlations, OLS
fits and within-dataset values for the new and the stored datasets): table 01 gains S3 rows (new, stored, fold
level, compute), tables 03 and 08 gain S3 columns (08 also S3 sign and within-dataset S3 Spearman), new figure
`09_i7_1_scatter_S1_lodo_vs_S3.png` (S1_lodo and S3 scatter panels side by side). Site: claim `i7-1` headline
"supported (simulated, prospective); S3 secondary done", S3 numbers, compute comparison, figure 09; status block
text updated. Site rebuilt, not published.

## I7-2 outcome, report batch and site (2026-10-20 19:50): hierarchical correction on K curves, inconclusive

Queue jobs I7_2_sky130 (540 s, 48 splits), I7_2_ihp300 (610 s, 66 splits; 6 K4 draws undefined), I7_2_tud4
(1266 s, 180 splits; 2 simulator timeouts), I7_2_tud295 (1234 s, 180 splits), I7_2_decide (finished 19:32);
`out/research/explore/I7_2/decision.json`. Preregistered configuration, no draw reduction, DEV devices only.
Rule output: inconclusive. C1 GMR(c_hier, b_shared), CI upper < 1 on 2/4 (needs 3): sky130 0.992 [0.929, 1.066],
IHP 300 K 0.904 [0.803, 1.024], TU Delft 4.13 K 0.912 [0.863, 0.958], 295 K 0.939 [0.904, 0.973]. C2
GMR(c_hier, a_C0), CI upper < 1 on 3/4 (needs 4): 0.766, 1.025 (IHP), 0.818, 0.758. C3 GMR(c_hier, d_scratch) point
< 1 on 1/4 (needs 4): 0.912 [0.840, 0.994], 1.253, 1.153, 1.121. C4 holds on 4/4 (sky130: pmos_L0p35_W0p55 failed
in every arm, draws K2b and K4b; 11 paired for c vs b and a, 10 for c vs d). Stop condition GMR(c, b) >= 0.98 on
1/4 (sky130), not 2.
Not in the rule: GMR(c_hier, e_crossinterp) 1.036 / 1.102 / 1.397 / 1.125 (sky130 / IHP / TU 4.13 K / TU 295 K);
GMR(e_crossinterp, a_C0) 0.713 / 0.930 / 0.585 / 0.673 (e is the lowest arm on both TU Delft datasets);
GMR(c_hier, c_ml) 1.126 / 0.930 / 1.005 / 1.065 (the learned prior is not lower than the flat prior); GMR(b_shared,
a_C0) 0.772 / 1.134 / 0.897 / 0.807; GMR(d_scratch, a_C0) 0.811 / 0.818 / 0.709 / 0.676.
Per K (geometric mean over the K's two draws): GMR(c, d) at K = 2 0.897 / 1.148 / 0.851 / 0.984, at K = 4 0.959 /
1.392 / 1.562 / 1.278: c_hier is ahead of d_scratch at K = 2 on 3/4 datasets and behind at K = 4 on 3/4. At K = 4
the two lowest arms are d_scratch and e_crossinterp on IHP and both TU Delft datasets. GMR(c, b) at K = 2 0.982 /
0.949 / 0.943 / 0.986, at K = 4 1.002 / 0.800 / 0.882 / 0.894.
Learned prior: mu = inf (delta = fold prior mean, z = 0) in sky130 44/48 splits, TU Delft 4.13 K 26/180, IHP 0/66,
TU Delft 295 K 0/180. Median RMS over the held-out points of phi . delta (c's correction minus b's) for c_hier /
c_ml and RMS of g_shared: sky130 0.0148 / 0.0341 / 0.0440, IHP 0.0074 / 0.0144 / 0.0071, TU 4.13 K 0.0072 / 0.0171 /
0.0699, TU 295 K 0.0052 / 0.0110 / 0.0208.
Report batch `out/research/report/2026-10-20_i7_2/` (`scripts/research/plot_explore_i7_2.py`, stored files only;
recomputes the curve split of every row from the records, per-device values, aggregates, paired ratios with device
and TU Delft cluster bootstrap, the decision, each fold's prior, selected mu and every stored delta, and checks them
against the stored files): decision table, forest plot, arm-mean table, per-draw / per-K table and figure, prior
table and figure, example-device figures for 2 devices per dataset. The example figures show measured points and
exp(phi . delta) only: the arm I-V curves are not stored (results.json keeps RRMS, cards.json the cards; d_scratch's
MLP is not stored) and were not re-simulated.
Site: new claim `i7-2` (status in progress; headline "inconclusive (pre-registered rule)") in the Exploration tab;
status block: iteration 7 complete, I7-2 item and the I7-2 decision table. Site rebuilt, not published.

### I7-2 example I-V curves (2026-10-20 19:50)

Queue jobs I7_2_examples (6 s) and I7_2_figs (6 s), script scripts/research/explore/i7_2_examples.py (commit dd55131). The 8 example device/draw pairs in figures 08-11 of batch 2026-10-20_i7_2 are re-simulated with NGSpice from each arm's stored final card (sky130 ngspice-41 arm64, IHP ngspice-46 + psp103.osdi, TU Delft ngspice-41 arm64); corrections are applied in Python to the NGSpice output (b, c, c_ml, d: x exp(g); e: + crossing residual). d's per-device MLP is refit with the run's seeds and gives the stored result exactly. Held-out and adaptation RRMS recomputed from these curves match results.json with relative difference 0 for every arm. Figures 08-11 now show the arm curves.

## I8-2 TEST predictions (pre-registered) and S3 gate (retrospective) (2026-10-20 22:45)

Part A, pre-registration (commits 3543948 code, 08d23e3 predictions; pushed before Part B):
`out/research/explore/I8_2/prereg_test_predictions.{json,md}` from `scripts/research/explore/i8_2_gate.py --mode
prereg`. Predicted sign of ln GMR(M_bias, C0_refit) on the 7 TEST datasets (final TEST run about 2026-12-10) from
the stored dataset-level DEV statistics: frozen I7-1 rule y = 0.007619 - 0.485184 S1_lodo and S3 < 0. sky130 77 K
REAL_TEST (6 devices): S1_lodo +0.3216, rule y -0.148 (gain), S3 -0.113 (gain). IHP SG13G2 TEST (10 DUTs) 233 /
300 / 343 / 398 K: rule y +0.160 / +0.116 / +0.168 / +0.191 (loss), S3 +0.254 / +0.366 / +0.375 / +0.395 (loss).
TU Delft TEST (24 devices) 4.13 / 295 K: rule y -0.237 / -0.149 (gain), S3 -0.033 / -0.041 (gain). S1_lodo and S3
agree on 7/7; the signs equal the DEV outcomes. Secondary S1_lodo on the TEST run's training set (all DEV devices;
IHP per class) has the same sign on 7/7. TEST scoring written down: M_bias (I3-2 recipe, 5 seeds, 2000 steps)
trained on all DEV starting-card records (IHP: same class and temperature), TEST starting cards (sky130: stored
43-parameter direct-fit cards; IHP / TU Delft: polish from stock, max_nfev 3000, irls 3), held-out refit with and
without M_bias (max_nfev 400, irls 2, diff_step 0.01), strict paper RRMS, y = mean ln ratio over TEST devices. No
TEST mode exists in the code yet. TEST data were not opened: the script reads DEV records, stored DEV statistics and
the split tag lists; every loaded record tag is in the DEV list (checked); stored S1_lodo and S3 reproduced with
difference 0.

Part B, retrospective (outcomes known before the gates were defined): `--mode gate`,
`out/research/explore/I8_2/gate_results.json`. G: on a dataset, M_bias iff S3(D) < 0, else C0_refit (one choice
per dataset from training-device statistics, not a per-device best-of); G1: the same with the frozen S1_lodo rule.
Evaluated from the stored per-device log ratios (checked against the per-device RRMS files and stored y, S3,
S1_lodo); no NGSpice, no training. exp(mean ln GMR vs C0_refit) (datasets above 1.02, worst):
11 stored: G 0.939 (0 above 1.02, worst 1.000), G1 0.928 (0, 1.000), M_bias always 0.989 (4, 1.272), dataset oracle 0.928.
7 measured: G 0.924 (0 above 1.02, worst 1.000), G1 0.924 (0, 1.000), M_bias always 1.022 (4, 1.272), dataset oracle 0.924.
20 simulated: G 0.590 (0 above 1.02, worst 1.000), G1 0.596 (0, 1.000), M_bias always 0.591 (2, 1.073), dataset oracle 0.588.
All 31: G 0.695 (0 above 1.02, worst 1.000), G1 0.697 (0, 1.000), M_bias always 0.710 (6, 1.272), dataset oracle 0.692.
G and G1 do not choose M_bias on any dataset where M_bias is worse than C0_refit. Versus M_bias always, G is worse
by more than 2% on 2/31 (IHP tier c 1.130, GF180 r0 lambda 0 1.046) and G1 on
4/31 (IHP r1 lambda 0 1.140, GF180 r0 lambda 0 1.046, GF180 r1 lambda 0.25 1.040, IHP r1 lambda 0.25
1.037). S3 on the stored datasets was computed with outcomes known; G1's rule was fitted on the 11 stored datasets.
Leakage: S3(D) averages S3_f over all outer folds, so a scored device's own starting-card record enters G's
decision through the folds where it is a training device (about 1/n_devices of S3(D)); the held-out refit outcomes
are not used. Diagnostic fold-level S3 gate (decision from S3_f of the device's own fold, which excludes it):
decisions differ from G in 4/11 stored (tudelft_4K, tudelft_295K, ihp_c, gf180_b) and 4/20 simulated
datasets; all 31 exp(mean) 0.712, 5/31 above 1.02 (the fold-level difference also contains
fold-level variability of S3_f; within a dataset S3_f and y_f have the opposite rank relation, I7-1). Datasets with
|S3(D)| < 0.01 (cut chosen post hoc): GF180 tier b, IHP tier b, GF180 r0 lambda 0, IHP r1 lambda 0.25; a flipped
decision there changes the dataset ln GMR by at most |y_D| <= 0.045. On TEST the gate uses DEV devices only.
Report batch `out/research/report/2026-10-20_i8_2/` (`scripts/research/plot_explore_i8_2.py`, stored files only,
146 values recomputed and checked): table 01 TEST predictions, table 02 gate per dataset, table 03 pooled, figure
04 per-dataset GMR(., C0_refit). Site: new claim `i8-2` (in progress) in the Exploration tab, status item and the
Part A table. Site rebuilt, not published.

## I8-1 ingest, split, smoke and pre-registration draft (2026-10-20 23:00): NIST mds2-2997 sky130 at 4 K

Data: NIST mds2-2997 (doi:10.18434/mds2-2997, record version 1.1.2), `MPW-5 Test Tile.zip` (39,471,626 bytes) and
`ETest Tile.zip` (3,807,177 bytes), sha256 equal to the NIST metadata values (`out/research/explore/I8_1/data/
sources.json`); local copies and extracted files under `out/research/data_raw/nist_mds2_2997/` (gitignored), read
as MDM text and as xlsx XML (stdlib reader; openpyxl is not installed in .venv) under `python -I`.
Code: `src/cryoml/explore/i8_1_nist.py`, `scripts/research/explore/i8_1_nist.py` (modes sources, inventory, data,
split, smoke), `tests/test_i8_1_nist.py` (5 cases).
Inventory (`inventory.json`, `inventory.md`): MPW-5 287 transistors (nfet_01v8 80, pfet_01v8 71, nfet_01v8_lvt 56,
pfet_01v8_lvt 40, nfet_03v3_nvt 40), one device per W/L within each type; ETest 114 devices. MPW-5 1.8 V types:
Id-Vg at |Vd| 0.1 and 1.8 V, 5 Vb values, 181 points; Id-Vd at |Vg| 0-1.8 V in 0.3 V steps, Vb 0 and 1 V magnitude,
91 points. Readme flags: nthin_lvt 23_4 "Issue" (gate-control ratio 12.1, excluded), pthin 17_6 "gate not
connected" (no files). No non-finite values, no all-zero curve, no drain-compliance hit, no identical current data.
Families: nfet_01v8_lvt and pfet_01v8_lvt, the device types of the 77 K paper cards (the nMOS card is the lvt type).
The pruned 77 K cards cover 21 of 56 nMOS and 22 of 40 pMOS geometries; the others need a bin choice (draft: forced
nearest bin; forcing the native bin reproduces native selection exactly).
Split (`split.json`, seed 20261021, strata family x bin coverage, blocks of 5 by (L, W), 3 DEV / 2 TEST): DEV 57
(nMOS 33, pMOS 24), TEST 38 (22, 16), excluded 1. Measured-rule inclusion on DEV: 198 nMOS curves (6 per device),
102 pMOS curves (2-6).
Smoke (`smoke/smoke.json`; stock 77 K card, no fit, no RRMS): one DEV device per family with its native bin and one
with the forced nearest bin, deck at 77 K and 4.2 K: 9 of 9 curves finite with the measured array shapes and
current sign; 0.023-0.025 s per simulation. At 4.2 K the stock-card subthreshold currents drop to the 1.8e-12 A
GMIN floor.
Pre-registration draft (`prereg.json`, `prereg.md`; not in force, not queued): P_a frozen I7-1 rule on DEV S1_lodo
(starting-card records), P_b GMR(M_bias, C0_refit) CI upper < 1 with R2 / R3, P_c sign(S3_D); prediction.json
committed before the LODO job; 11 open questions (card bins, deck temperature, inclusion rule, box, curve set,
baselines, VDD, S1 records). Compute estimate 4-9 CPU-h, about 45-90 min on 6 workers.

## I8-1 pre-registration in force, pipeline, smoke, queue (2026-10-20 23:10)

Pre-registration `out/research/explore/I8_1/prereg.json` in force (commit d022c32, committed alone before any fit on
these data). Decisions D1-D11 close the draft questions: nfet_01v8_lvt / pfet_01v8_lvt; all 57 DEV devices with an
explicit bin per device (native bin for the 25 covered geometries, nearest bin in (ln L, ln W) otherwise), the
25-device covered subset as a pre-registered secondary from the same results; deck at 77 K (limitation: kT/q, band
gap and n_i stay at 77 K); measured-only inclusion stored on the starting-card records; Vb = 0; cold_start round-one
box around the showmod stock values; S1_lodo from starting-card records (I7-1 definition); B1 / B2 / B3_d3 reported
only; VDD 1.8 V; predicted y_D, per-family values and the I8-2 gates G / G1 registered as secondary predictions.
Code: `cryoml.explore.i8_1_nist` (bins, deck with mult = m = 1 for every device, showmod readback, box, problem,
records, guard), driver modes start, stats, commit_prediction, lodo, baselines, decide; tests 8 cases.
Smoke (`smoke_*`, `smoke_runs.json`; 4 DEV devices, reduced settings, not used for any decision): all modes run end
to end; 0 timeouts, 0 failed devices; 0.024 s per NGSpice run; the lodo guard reports an untracked prediction and
refuses without --smoke. Queue: I8_1_start, I8_1_stats, I8_1_commit_prediction, I8_1_lodo, I8_1_baselines,
I8_1_decide. Estimated wall time about 70 min on 6 workers (start 12 min, stats 22 min, lodo 25 min, baselines 10
min).

## I8-1 failed starting records and amendment A1 (2026-10-20 23:55)

The full run (queue jobs I8_1_start 411 s, I8_1_stats 1264 s, I8_1_commit_prediction -> 0c184f8) produced 4 starting
records with NaN simulated points: nfet_01v8_lvt W0.55/L0.25, W3.01/L0.15, W7/L0.15, W7/L2 (forced bins 0, 1, 1, 4,
stretch factors up to 2.0 x 2.33). Cause: for these bin-uncovered geometries the forced bin's binned PCLM is negative
(-0.34, -0.13, -0.03, -0.12); BSIM4 4.8.2 parameter checking stops the first analysis of the ngspice process ("Pclm
is not positive"), so the first sweep (Id-Vg at Vd = 0.1 V) has no output while the other 8 sweeps run. The stock
card fails the same way; the starting fits kept the penalty for that curve (objective about 1.67-1.70). Not a code
bug; the starting fits are unchanged. fit_bias trains on every record of a training set, so one NaN record made
every correction NaN: S3_D and every S3 inner value in prediction.json are null, and its P_c ('no gain') and gate G
('C0_refit') came from NaN < 0 being false. The I8_1_lodo job was stopped before writing output.
S1 statistics were finite but included the 4 records (device_residual keeps their finite curves). Committed:
S1_lodo 0.50247, S1_pooled 0.54795, S2 0.61573; with the 4 records excluded (53 folds): 0.50712, 0.55282, 0.64218;
frozen-rule sign 'gain' in both (predicted y -0.2362 / -0.2385).
Amendment A1 (`prereg_amendment_2026-10-20.json`, commit 5ba2f8e, before any held-out outcome): records with any
non-finite simulated point are excluded from every training set and S fold and scored as failed in every arm; R2
means and R3 over devices with a finite starting record; V4 read as written (1 start-job timeout,
pfet_01v8_lvt_W1_L0p35_m28p6, reported; arm timeouts drop devices); P_a primary stays the committed value, the
recomputed S1 is secondary; P_c, G and the per-family S3 values of prediction.json are void and come from
prediction_v2.json. Code: worker processes hold only finite records (split_finite), stats_v2 writes
prediction_v2.json and refuses non-finite S3 values, lodo / baselines require prediction.json, prediction_v2.json and
the amendment committed, decide checks the S1 values (v1 and v2) and the excluded sets; unit test with a NaN record
(9 tests). Smoke rerun with one failed device (smoke_runs.json, amendment_A1_smoke). Queue: I8_1_stats_v2,
I8_1_commit_prediction_v2, I8_1_lodo, I8_1_baselines, I8_1_decide.

## I8-1 outcome (2026-10-21 00:45): NIST mds2-2997 sky130 4 K DEV, prospective

Queue (after amendment A1): I8_1_stats_v2 992 s, I8_1_commit_prediction_v2 -> 4db5a7b, I8_1_lodo 1016 s (guard: prediction
0c184f8, prediction_v2 4db5a7b, amendment 5ba2f8e tracked, unmodified, ancestors of HEAD), I8_1_baselines 393 s,
I8_1_decide 3 s. Outputs `out/research/explore/I8_1/{lodo, baselines}/{results, cards, corrections}.json`,
`decision.json`.
Validity: V1 53/57 finite starts; V2 failed C0_refit 4, M_bias 4 (the A1 start failures), 53 paired; V3 prospective;
V4 1 start-job timeout (pfet_01v8_lvt_W1_L0p35_m28p6, record finite), 0 timeouts in lodo and baselines, 0 dropped.
P_a (primary, frozen I7-1 rule on the committed S1_lodo_D 0.50247): predicted y -0.2362 (gain), observed y_D
-0.2157: correct; |error| 0.0205 (frozen rule in-sample residual RMS 0.0751). A1-recomputed S1_lodo 0.50712,
predicted y -0.2384: correct. P_c (prediction_v2 S3_D -0.1081, gain): correct. P_b: GMR(M_bias, C0_refit) 0.806
[0.758, 0.864], 48/53; R2 M_bias mean 0.03151 vs T1_inloop 0.03490 + 0.005; R3 0 M_bias failures: supported.
Secondary: nMOS 0.808 [0.780, 0.840] 28/29, pMOS 0.804 [0.707, 0.934] 20/24 (rule and S3 signs correct in both);
covered subset (25) 0.801 [0.739, 0.881] 23/25; forced bin 0.811 [0.742, 0.898] 25/28. Gates G and G1 choose M_bias.
M_bias vs T1_inloop 0.887 [0.852, 0.930]; T1_inloop vs C0_refit 0.908; C0_refit vs C0 0.969. Fold-level
Spearman(S1_lodo_f, y_f) +0.706 (prediction.json folds), +0.734 (prediction_v2 folds); Spearman(S3_f, y_f) -0.381.
Baselines (reported): M_bias vs B1 0.787 [0.739, 0.845], vs B2 0.811 [0.759, 0.874], vs B3_d3 0.981 [0.961, 1.000]
27/53; B3_d3 vs C0_refit 0.822 [0.768, 0.885]. B3_d3 is the I4-1 baseline B3 at degree 3
(`i4_1_baselines.fit_poly`): same form I = I_card x exp(g) and same loss as M_bias, g = tanh(P_pol(u, v)) with a
polynomial of total degree 3 in the standardized u = |Vgs|/VDD, v = |Vds|/VDD (10 monomials), one coefficient
vector per polarity (20 coefficients), L2 1e-4, Adam 3e-3, 2000 steps, one fit, held-out refit with g in the loop.
ln GMR(B3_d3, C0_refit) / ln GMR(M_bias, C0_refit) = 0.911: the cubic polynomial gives 91% of the log gain; the MLP
form adds a factor 0.981 over it on these data. At the trimmed included training points the mean fold g of M_bias
and B3_d3 have Pearson r 0.94 (nMOS) / 0.95 (pMOS), RMS difference 0.005 / 0.039.
Report batch `out/research/report/2026-10-21_i8_1/` (`scripts/research/plot_explore_i8_1.py`; recomputes and checks
every stored aggregate, paired ratio, y, P_a / P_b / P_c, validity, gates and the S1 / S3 dataset values): table 01
decision, figure 02 I7-1 S1_lodo / S3 scatter with the NIST point, figure 03 per-device log ratio vs L, W and GMR per
stratum, table 04 subsets, table 05 arms, figure 06 mean fold g maps (NIST M_bias, NIST B3_d3, sky130 77 K I3-2
M_bias), figures 07-10 example I-V. Example curves: `scripts/research/explore/i8_1_examples.py` re-simulates the
stored C0, C0_refit and M_bias cards of 4 devices (per family and bin stratum, the device nearest the stratum
median log ratio; 0.4 s, `examples/examples.json`); RRMS reproduced with relative difference 0. Post hoc (not
pre-registered): family x stratum rows, B3_d3 subset columns, the S3 OLS reference point, g maps, I-V figures.
Site: new claim `i8-1` (in progress; headline "P_a correct, P_b supported (NIST sky130 4 K DEV, prospective)") in the
Exploration tab, status item and decision table. Site rebuilt, not published.

## I9-2 pre-registration draft, code and smoke (2026-10-21 01:10): function-class ablation for the shared bias-only correction

Draft `out/research/explore/I9_2/prereg_draft.json` / `.md` (not in force; `--mode run` and `lc` refuse until
`prereg.json` is committed). Question: does the held-out gain of g in I = I_card x exp(g) depend on the function class,
and does a class with fewer trained values than M_bias (1765 over 5 members) match or beat it? Classes: const (2), lin
(6), poly2-poly5 (12 / 20 / 30 / 42), gam (8 cubic B-splines per input, rank-1 interaction, 66), mlp8 / mlp16 (=
M_bias) / mlp32 / mlp64 (5 seeds; 565 / 1765 / 6085 / 22405), mlp16_s1 (member 0 of the stored M_bias, 353), gp (I4-1
B4). Protocol of the stored runs (folds, starting-card records, E1 loss, Adam 3e-3, 2000 steps, L2 1e-4, held-out refit
max_nfev 400 irls 2 with g frozen, strict paper RRMS) through each dataset driver's refit function. 12 datasets:
primary pool sky130 77 K, NIST 4 K, TU Delft 4.13 K, IHP measured 300 K; secondary GF180 / IHP tier c, other
temperatures, tier b null control. Stored rows reused (C0_refit, M_bias, T1_inloop, B3_d2 / B3_d3 / B4, F2 widths 8 /
32). Rule: pooled equal-weight GMR(X, M_bias) over the primary pool, stratified device bootstrap, Bonferroni over 12
contrasts, margin 1.03; Q1 class-dependent if some class beats or is worse; Q2 smallest non-inferior class with fewer
trained values, point GMR <= 1.05 on every primary dataset, no extra failures. Learning curve (secondary): best two
classes by pooled GMR vs C0_refit, n_train 2 / 4 / 8 / all, 3 polarity-stratified subsets, TU Delft 4.13 K and NIST.
Code: `src/cryoml/explore/i9_2_classes.py`, `scripts/research/explore/i9_2_classes.py` (modes plan, smoke, run, decide,
lc), `tests/test_i9_2_classes.py` (19 tests). Plan (`plan.json`, no simulation): every dataset's records, folds and
stored rows load; stored rows finite on all held-out devices. Compute estimate: 70 CPU-h for all 12 datasets (about
13.7 h wall on 6 workers), primary pool 19 CPU-h (3.8 h); mlp32 + mlp64 5-seed training is about 45 of the 70 CPU-h;
learning curve about 1.2 h. Smoke (sky130 77 K, 2 held-out devices, 200 steps, 2 seeds, GP 300 points, refit nfev 60;
30 s; code check only). Not queued.

## I9-1 ingest, split, smoke and pre-registration draft (2026-10-21 01:05): NIST mds2-2997 sky130 4 K, nfet_01v8 / pfet_01v8

Second prospective real-data test on the NIST MPW-5 tile, families not used in I8-1: nfet_01v8 (nthin, 80 devices,
W 0.36-7 um, L 0.15-20 um) and pfet_01v8 (pthin, 71 devices with files). No 77 K card exists for these types; the
starting cards are the open sky130 room-temperature tt cards of the Volare open_pdks build already on disk
(a918dc7; nfet_01v8 tt.pm3 180 bins, pfet_01v8 tt.corner + tt.pm3 108 bins, tnom 30 C; sha256 in
`out/research/explore/I9_1/cards.json`; no download).
Code: `src/cryoml/explore/i9_1_nist_rt.py` (card text: one assignment per `+` line so a written parameter keeps
its l / w / p binning terms; per-run card with the device's bin only; native bin via eta0-sentinel readback; deck,
showmod readback, `MeasProblemRT`), `scripts/research/explore/i9_1_nist_rt.py` (modes cards, data, split, smoke,
start, stats, commit_prediction, lodo, baselines, decide; I8-1 pipeline with amendment A1 built in; lodo and
baselines refuse to start while prereg.json has status DRAFT), `tests/test_i9_1_nist_rt.py` (6 cases).
Bin coverage: all 151 geometries inside a bin of the full cards (no forced bins). Card checks on 5 DEV devices: the
single-bin card equals the full rewritten card (difference 0) and the unmodified open_pdks `sky130.lib.spice` tt
library (<= 1.7e-34 A).
Inclusion (measured-only rule): nMOS 6 curves per device; pMOS 0-6. Draft decision D12 excludes the 6 pMOS with
fewer than 2 included curves (L 8-20 um). Split (I8-1 procedure, seed 20261022): DEV 87 (nMOS 48, pMOS 39), TEST 58
(32, 26). TEST not opened.
Smoke (`smoke/smoke.json`, `smoke_runs.json`; not used for any decision): stock-card scan of the 87 DEV devices at
30 C, 0 non-finite curves, 0 fatal lines, median 0.025 s per simulation, so no failed-start device exists for the
smoke; reduced-setting runs of start, stats, lodo (with the transfer arm M_bias_lvt trained on the I8-1 DEV
records), baselines and decide on 5 DEV devices: 0 failures, 0 timeouts. Deck-temperature records (not scored): 27 C
changes stock currents by 0.0-0.8% at |Vg| = |Vd| = 1.8 V; 4.2 K with the card's temperature equations raises that
current x 1.4-12.8 and moves the threshold (current at |Vg| = 0.9 V 0-37% of the 30 C value); 4.2 K with the
temperature coefficients and their binning terms at 0 gives x 0.94-0.99. Smallest subthreshold swing: measured
6-46 mV/decade, stock card at 30 C 81-135 mV/decade.
Pre-registration draft (`prereg.json`, `prereg.md`, status DRAFT): P_a, P_b, P_c as I8-1; pre-registered
secondaries S_B3 = GMR(M_bias, B3_d3) with CI, S_transfer = arm M_bias_lvt; V2 minimum 70 paired devices. Open
questions: deck temperature 30 C (tnom) vs 27 C, the D12 exclusion, S3 cost, the transfer arm, V2 threshold, P_a
outside the I7-1 calibration range (S1_lodo -0.379 to 0.504). Compute estimate about 2.3 h wall on 6 workers (S3
about 75 min). Not queued.

## I9-2 pre-registration in force, queue (2026-10-21 01:25)

Pre-registration `out/research/explore/I9_2/prereg.json` / `.md` in force (commit 02c6996, committed alone before any I9-2
held-out result; code commit dea00cc). Decisions on the draft questions (each with its reason in the file): D1 primary
pool only (sky130 77 K, NIST 4 K, TU Delft 4.13 K, IHP measured 300 K); tier c and other temperatures would be a later,
separately pre-registered run; tier b dropped; I7-1 datasets excluded. D2 margin 1.03, per-dataset limit 1.05. D3
reference replicate mlp16_rep (M_bias recipe, seeds 5-9; stored F2 a_seeds_5_9 rows on sky130), GMR(mlp16_rep, mlp16)
reported beside every class; a 'beats' / 'worse' class whose pooled GMR lies inside the replicate's CI gets a note,
label unchanged. D4 parameter count = total over members, per-member count reported. D5 gam unchanged. D6 poly4 / poly5
in the Legendre product basis on the standardized inputs (poly2 / poly3 stay the stored B3 monomial fits); final
training loss for every class and fold, objective trace every 250 steps (new optional `trace` argument of fit_bias /
fit_poly; no numerical change, unit-tested). D7 mlp64 dropped. D8 learning curve = mlp16 + the two best classes. D9
equal dataset weight. 11 contrasts, Bonferroni level 1 - 0.05/11.
Code checks: smoke rerun (52 s, 2 workers, while I9_1_start ran); reduced validity run through the sky130 / NIST / TU
Delft / IHP refit adapters (7.5 s); decide and figs run on synthetic inputs; 22 unit tests.
Compute estimate (`plan.json`, 6 workers): validity 1 min; run jobs sky130 4 min, NIST 66 min, TU Delft 41 min, IHP 10
min (10.2 CPU-h); replicate 26 min (2.2 CPU-h); learning curve 1.7-4.4 h wall depending on the selected classes
(8.7-22.4 CPU-h). Queue (after the I9-1 jobs): I9_2_validity, I9_2_run_{sky130_77K, nist_4K, tudelft_4K,
ihp_meas_300K}, I9_2_replicate, I9_2_decide, I9_2_lc_tudelft_4K, I9_2_lc_nist_4K, I9_2_figs.

## I9-1 outcome (2026-10-21 03:45): NIST mds2-2997 sky130 4 K, nfet_01v8 / pfet_01v8, room-temperature card, prospective

Pre-registration `out/research/explore/I9_1/prereg.json` / `.md` in force from 007759b (committed alone before any
full-setting fit; decisions Q1 deck 30 C, Q2 6 low-current pMOS excluded before the split, Q3 full S3 pair design, Q4
M_bias_lvt secondary, Q5 V2 minimum 70 paired devices, Q6 calibration-range note with P_a). Queue: I9_1_start 472 s
(87 DEV starting cards; 0 non-finite records, 0 ngspice timeouts; stock mean RRMS 1.627 -> start 0.0576; c813f6f),
I9_1_stats 4352 s, I9_1_commit_prediction -> e937ca2, I9_1_lodo 2435 s (guard: prereg 007759b in force, prediction
e937ca2 tracked, unmodified, ancestors of HEAD), I9_1_baselines 509 s, I9_1_decide 4 s. Outputs
`out/research/explore/I9_1/{lodo, baselines}/{results, cards, corrections}.json`, `decision.json`.
Validity: V1 87/87 finite starts; V2 failed C0_refit 0, M_bias 0, 87 paired; V3 prospective; V4 0 timeouts, 0 dropped.
P_a (frozen I7-1 rule on S1_lodo_D 0.29484, inside the I7-1 calibration range [-0.379, 0.504]): predicted y -0.1354
(gain), observed y_D -0.2266: correct; |error| 0.0911 (frozen rule in-sample residual RMS 0.0751). P_c (S3_D -0.0710,
gain): correct. P_b: GMR(M_bias, C0_refit) 0.797 [0.748, 0.859], 77/87; R2 M_bias mean 0.04554 vs T1_inloop 0.05310 +
0.005; R3 0 M_bias failures: supported.
Per family (secondary): nMOS y -0.361, GMR 0.697 [0.672, 0.724] 47/48, rule and S3 signs correct; pMOS y -0.061, GMR
0.941 [0.842, 1.085] 30/39, rule sign correct (predicted y -0.082), S3 sign wrong (S3_pmos +0.037 predicted no gain).
Gates G and G1 choose M_bias. 10/87 devices have M_bias above C0_refit; the 6 largest log ratios are pMOS with L >= 4 um
and C0_refit RRMS 0.004-0.023 (largest pfet_01v8_W3_L20_m17p7, +1.71).
Comparators: M_bias vs T1_inloop 0.815 [0.782, 0.851]; T1_inloop vs C0_refit 0.978 [0.941, 1.029]; C0_refit vs C0
0.968. Baselines (reported): M_bias vs B1 0.780 [0.732, 0.839], vs B2 0.804 [0.755, 0.866]; B1 vs C0_refit 1.023, B2 vs
C0_refit 0.991. S_B3: M_bias vs B3_d3 0.955 [0.926, 0.985] 55/87, CI upper < 1 (I8-1: 0.981 [0.961, 1.000]); B3_d3 vs
C0_refit 0.835 [0.782, 0.900], 80% of the M_bias log gain (nMOS 72%, pMOS 134%; pMOS M_bias vs B3_d3 1.021).
S_transfer: M_bias_lvt (M_bias recipe trained once on the 53 finite I8-1 lvt DEV starting records, 77 K card) vs
C0_refit 0.887 [0.828, 0.962] 79/87; vs M_bias 1.112 [1.071, 1.156] 29/87 (nMOS 1.220, pMOS 0.993).
Fold level (computed in the report): Spearman(S1_lodo_f, y_f) +0.231, Spearman(S3_f, y_f) -0.604 over 87 folds.
Report batch `out/research/report/2026-10-21_i9_1/` (`scripts/research/plot_explore_i9_1.py`; recomputes and checks
every stored aggregate, paired ratio, y, P_a / P_b / P_c, validity, gates, the transfer and baseline comparisons and
the S1 / S3 dataset and per-family values): table 01 decision, figure 02 I7-1 S1_lodo / S3 scatter with the I8-1 and
I9-1 points, figure 03 per-device log ratio vs L, W and C0_refit RRMS, table 04 arms, figure 05 GMR forest plot, table
06 per family, figure 07 S1_lodo / S3 per family and fold level, figure 08 transfer arm, figure 09 mean g maps (I9-1
M_bias, I9-1 B3_d3, M_bias_lvt), table 10 I8-1 vs I9-1, figures 11-13 example I-V. Example curves:
`scripts/research/explore/i9_1_examples.py` re-simulates the stored stock, C0, C0_refit and M_bias cards of 3 devices
(nMOS and pMOS nearest the family median log ratio, and the device with the largest log ratio; 0.5 s,
`examples/examples.json`); RRMS reproduced with relative difference 0 (stock card RRMS equal to start/summary.json).
Post hoc (not pre-registered): per-family B3_d3 and M_bias_lvt columns, the per-device panel vs C0_refit RRMS, the S3
OLS reference point, fold-level panels, g maps, I-V figures.
Site: new claim `i9-1` (verified; headline "P_a correct, P_b supported (NIST sky130 4 K regular-Vt DEV,
room-temperature card, prospective)") in the Exploration tab, status item and decision table. Site rebuilt, not
published.

### I9-2 validity failure and requeue (2026-10-21 03:14-03:35)

The I9_2 jobs failed at 03:14 (I9_2_validity rc=1; the later I9_2 jobs exited rc=1 in 0-1 s). Cause: two held-out TU
Delft devices that share one fold queued the (dataset, poly3, fold) training task twice, and the merge step then
combined the two results as if they were MLP seed members. Fix a864f77: each (dataset, poly3, fold) is trained once;
merge_members refuses non-MLP members; unit test added. No I9-2 result was produced before the fix. The jobs were
requeued unchanged at 03:35 (I9_2_validity rc=0, 31 s).

## I9-2 outcome (2026-10-21 08:50): function-class ablation for the shared bias-only correction, primary pool

Pre-registration `out/research/explore/I9_2/prereg.json` / `.md` in force from 02c6996 (code dea00cc). The jobs failed
once at 03:14 on a validity bug and were requeued unchanged at 03:35 after fix a864f77 (see "I9-2 validity failure and
requeue" above; no I9-2 result existed before the fix). Queue after the requeue: I9_2_validity 31 s, run sky130 77 K
186 s, NIST 4 K 3282 s, TU Delft 4.13 K 2330 s, IHP 300 K 492 s, replicate 1429 s, decide 1 s, lc TU Delft 2923 s,
lc NIST 3701 s, figs 2 s (3.99 h; 129 min without the learning curve against the 147 min estimate). Outputs
`out/research/explore/I9_2/{validity, decision}.json`, `{sky130_77K, nist_4K, tudelft_4K, ihp_meas_300K, replicate,
lc_tudelft_4K, lc_nist_4K}/{results, cards, corrections}.json` (09f60c9 and the report commit).
Validity: V1 poly3 rerun max |RRMS - stored| 0 (and coefficient difference 0) on all 4 datasets; V2 failed devices 4 on
NIST 4 K in every class (non-finite starting records), gp 2 more on TU Delft 4.13 K (t40_nmos_W0p12_L0p04_d152,
t40_nmos_W1_L0p04_d199: every Id-Vd curve non-finite after the gp-corrected refit), all other classes 0; V3 0 simulator
timeouts in the run refits; V4 stored rows equal the source files (checked in the report).
Labels (pooled GMR(X, mlp16), Bonferroni CI at 1 - 0.05/11, margin 1.03): const 1.095 [0.981, 1.198] inconclusive; lin
1.065 [0.948, 1.169] inconclusive; poly2 1.103 [1.041, 1.169] worse; poly3 1.086 [1.029, 1.150] worse; poly4 1.060
[1.001, 1.131] worse; poly5 1.062 [0.978, 1.154] inconclusive; gam 0.970 [0.922, 1.030] inconclusive; mlp16_s1 0.999
[0.982, 1.015] equivalent; mlp8 0.986 [0.966, 1.006] non-inferior; gp 1.483 [1.319, 1.684] worse; mlp32 1.003 [0.987,
1.019] equivalent. Replicate mlp16_rep (seeds 5-9) 0.997 [0.982, 1.013]; no replicate note applies.
Q1: class-dependent. Q2: candidates mlp16_s1, mlp8; smallest matching class mlp16_s1 (353 trained values); no class
beats mlp16 with fewer parameters. gam is not a candidate (CI upper 1.030; IHP 300 K point GMR 1.211 > 1.05).
Per dataset (95%, descriptive): NIST 4 K GMR vs mlp16 poly4 0.911, poly5 0.857, gam 0.831 (fractions of the mlp16 log
gain 1.43 / 1.72 / 1.86); IHP 300 K GMR vs C0_refit above 1 for every class with more than 6 trained values (poly2
1.201 ... poly5 1.816, gam 1.471, mlp16 1.214), const 1.000, lin 0.983; GMR vs mlp16 const 0.824, lin 0.809.
Pooled vs C0_refit (95%): gam 0.889 [0.819, 0.973], mlp8 0.904, mlp16 0.917 [0.859, 0.986], poly3 0.996, gp 1.367.
Learning-curve classes from decision.json: mlp16, gam, mlp8.
S4 margin sensitivity: labels at 1.02 equal those at 1.03; at 1.05 gam non-inferior, mlp8 equivalent; Q2 mlp16_s1 at
every margin. S3 cluster bootstrap (TU Delft groups): poly4 CI [0.9996, 1.134] includes 1 (label from it, post hoc:
inconclusive); the other labels are unchanged. S7: gam has the lowest mean final training loss on all 4 datasets; gp
mean training loss 1.27 (NIST) and 3.91 (TU Delft) against 0.028-0.063 for the other classes.
Q3 learning curve (GMR(X, C0_refit) at n = 2 / 4 / 8 / all training devices per fold): TU Delft mlp16 1.004 / 0.966 /
0.933 / 0.939, gam 1.017 / 0.984 / 0.915 / 0.874, mlp8 1.005 / 0.963 / 0.926 / 0.919; NIST mlp16 0.947 / 0.846 / 0.825
/ 0.806, gam 0.912 / 0.769 / 0.717 / 0.670, mlp8 0.936 / 0.838 / 0.814 / 0.797.
Report batch `out/research/report/2026-10-21_i9_2/` (`scripts/research/plot_explore_i9_2.py`; recomputes every pooled
and per-dataset GMR, labels, Q1, Q2, margin-sensitivity labels, gain fractions, training-loss means and learning-curve
summaries from the per-device rows and checks them against decision.json and the lc results): table 01 decision,
figure 02 forest plot, figure 03 per-dataset heatmaps, table 04 per dataset, figure 05 capacity curve, figure 06
training loss vs held-out, figure 07 learning curves, table 08 learning curve, figure 09 gain fraction, figure 10
replicate, table 11 margin sensitivity and cluster bootstrap, table 12 validity and compute, figures 13-14 example I-V.
Example curves: `scripts/research/explore/i9_2_examples.py` re-simulates the stored C0_refit, mlp16, poly5 and gam
held-out cards of the NIST 4 K devices with the lowest (pfet_01v8_lvt_W1_L4_m29p2, -0.588) and highest
(nfet_01v8_lvt_W1_L8_m23p1, +0.195) ln(poly5 / mlp16) (0.4 s, `examples/examples.json`); RRMS reproduced with relative
difference 0.
Post hoc (not pre-registered): labels from the cluster-bootstrap CI, the training-loss vs held-out pairing, the
learning-curve class-vs-class values, the I-V figures.
Site: new claim `i9-2` (verified; headline "Q1 class-dependent; Q2 smallest matching class mlp16_s1 (353 values); no
class beats M_bias with fewer parameters") in the Exploration tab, status item and decision table. Site rebuilt, not
published.

## I10-2 pre-registration in force, interim stored-row selectors, queue (2026-10-21 09:50): class selection for the shared bias-only correction

Question: can the class of g (menu none = C0_refit, lin, gam, mlp8, mlp16) be selected per dataset from training data
only, so that the selected method is not worse than C0_refit on any dataset and close to the best fixed class?
Retrospective DEV analysis on the stored I9-2 folds of sky130 77 K, NIST 4 K, TU Delft 4.13 K, IHP measured 300 K (menu,
datasets, I9-2 rows and I8-2 S3 values seen before); prospective test at the December TEST stage (protocol in the prereg).
Code d16c929 (`src/cryoml/explore/i10_2_select.py`, `scripts/research/explore/i10_2_select.py`, 19 unit tests, smoke 18 s
on 2 sky130 folds, V_fold difference 0, decide path on synthetic rows only). Pre-registration
`out/research/explore/I10_2/prereg.{json,md}` in force from 55c4cfc (committed alone).
Decisions (reasons in the file): D1 menu M5 (0 / 6 / 66 / 565 / 1765 trained values), full 13-option menu for S_cv as a
secondary; D2 selection per stored outer fold from that fold's training devices; D3 held-out score = stored I9-2 row of
the selected class, no outer refit; D4 inner leave-one-group-out up to 12 groups (sky130, TU Delft, IHP), else 4 grouped
folds (NIST); D5 inner MLP classes seed 0 only (I9-2 mlp16_s1 equivalent to mlp16); D6 S_loss = post hoc mean ln(RRMS
corrected / RRMS start) on inner held-out starting records; D7 S_refit = inner held-out refits with g frozen (outer
settings) vs the stored C0_refit, all inner devices except NIST (one inner fold of 4); D8 S_gate = fold-level S3_f < 0 ->
mlp16; D9 S_cv = stored rows of the fold's training devices (not strictly nested: the fold's starting records are in the
other folds' training sets); D10 argmin, tie 0.005 -> fewer trained values, failed inner value blocks the class; D11
equal dataset weight, TU Delft cluster bootstrap; D12 11 contrasts at 1 - 0.05/11 (GMR vs C0_refit x 4, vs mlp16 x 4, vs
S_gate x 3); D13 margin 1.03; D14 NIST room temperature (I9-1) secondary with menu {none, poly3, mlp16}; D15 oracle classes
post hoc only. Rule: C1 = every dataset upper Bonferroni CI of GMR(S, C0_refit) <= 1.03; C2 = pooled upper CI of GMR(S,
mlp16) <= 1.03; supported = C1 and C2; overall stop / continue (class selection adds to the gate if a supported
multi-class selector has GMR(S, S_gate) upper < 1; else gate only; else multi-class selector).
Interim (`--mode stored`, 1.4 s, `interim.json`; S_loss and S_refit pending, rule not applied):
S_gate (selections mlp16 on sky130 12/12, NIST 53/53, TU Delft 5/7 groups; none on IHP 18/18): pooled GMR vs C0_refit
0.893 [0.836, 0.951], vs mlp16 0.974 [0.885, 1.049]; per dataset vs C0_refit sky130 0.769 [0.623, 0.942], NIST 0.806 [0.740,
0.896], TU Delft 1.027 [0.911, 1.144], IHP 1.000 [1.000, 1.000]; C1 fails on TU Delft, C2 fails; label not supported.
S_cv (gam on sky130 12/12 and NIST 53/53; TU Delft gam 6/7, lin 1/7; IHP lin 10/18, none 8/18): pooled vs C0_refit 0.814
[0.742, 0.888], vs mlp16 0.888 [0.797, 0.964] (beats mlp16), vs S_gate 0.912 [0.865, 0.948]; per dataset vs C0_refit
sky130 0.727 [0.580, 0.911], NIST 0.670 [0.590, 0.767], TU Delft 0.939 [0.736, 1.164], IHP 0.962 [0.891, 1.009]; C1 fails on
TU Delft only (7-group cluster bootstrap at the Bonferroni level; device bootstrap 95% 0.939 [0.825, 1.050]); label partly
supported. Secondary: S_cv full menu 0.822 vs C0_refit; I8-2 dataset-level gate 0.873; NIST room temperature S_gate and
S_cv both select mlp16 on 87/87 folds, GMR vs C0_refit 0.797 [0.749, 0.859]; post hoc S_cv vs per-dataset oracle 1.013
[0.984, 1.052].
Queue (after the runner was idle; started 09:47): I10_2_inner (about 45 min), I10_2_refit_{sky130_77K, ihp_meas_300K,
tudelft_4K, nist_4K} (about 23 / 19 / 36 / 76 min), I10_2_decide; plan 3.73 CPU-h training + 13.15 CPU-h refits, about
3.3 h wall on 6 workers.

## I10-1 pre-registration in force, code, smoke, queue (2026-10-21 10:00): 4 K I-V from room-temperature data

Question: can a device's 4 K I-V be predicted without any of its 4 K data, from its room-temperature (RT) data and
what training devices with both temperatures teach? Pre-registration `out/research/explore/I10_1/prereg.json` / `.md`
in force from 17c96a1 (committed alone, before any full-setting fit on the I10-1 DEV data); code, manifest and smoke
eeaed11; feasibility a108203 (`docs/research/i10_1_feasibility_2026-10-21.md`).
Code: `src/cryoml/explore/i10_1_rt4k.py` (open_pdks tt card handling for any sky130 FET type, generalized from I9-1;
DEV-type ETest discovery that parses only DEV-type file names; RT restriction to |V| <= 3.3 V; RT descriptors; shift in
ParameterBox local coordinates; ridge shift with inner leave-one-out lambda; GMR with a CI level),
`scripts/research/explore/i10_1_rt4k.py` (modes data, smoke, start, lodo, decide; start, lodo and decide refuse outside
--smoke unless prereg.json is tracked, unmodified, in an ancestor of HEAD and not DRAFT), `tests/test_i10_1_rt4k.py`
(12 cases; 15 with the I10-1 pairing test, 21 with the I9-1 card tests; the "21 unit tests" in the code commit and in
prereg.json count all three files).
DEV: NIST ETest nfet_g5v0d10v5 36 / pfet_g5v0d10v5 31 devices (61 geometry groups, 37 bin-stretched), 4 K paired with
the google sky130 raw-data RT files, RT fit on |V| <= 3.3 V; 14 of 81 excluded (4 device checks; 10 with L 0.30-0.40 um
whose stretched-bin stock card fails BSIM4 parameter checks at 30 C: Lpeb < -Leff, Pclm <= 0, Drout < 0). TU Delft I6-1
DEV 45 devices with the stored 295 K and 4.13 K cards (reproduce their records, difference 0). TEST: 25 ETest 1.8 V core
devices, locked; no I10-1 mode reads them (the feasibility check read them once for descriptive numbers; no fit, no
arm scored). Card checks: single-bin card equals the full card and the open_pdks library (relative difference <=
2.8e-18) on 2 DEV devices and on a synthetic grid for the 5 core types.
Fits: C0_RT (RT card, deck 30 C) and C4 (4 K card at the RT deck, warm start from C0_RT); variants C4_ind (from stock),
C4_deck (deck -269.15 C); TU Delft C4 warm start from the stored 295 K card at the 295 K stage deck.
Arms (leave-one-geometry-out, training within type / polarity): A0 C0_RT at the 4 K deck (NIST BSIM4 temperature
equations at -269.15 C; TU Delft kT/q only); A0_rtdeck C0_RT at the RT deck; A1 C0_RT + median shift of (C4 - C0_RT)
in local coordinates; A2 A1 x exp(g), M_bias trained on nested out-of-fold A1 records, no held-out refit; A3 A1 + ridge
term on geometry, 3 RT descriptors and the RT card coordinates; A1_L (geometry-only ridge); references R_pop (median
training C4 on the device's bin), R_oracle (own C4), stock card at both decks; variants V_ind, V_mean, V_sub7, V_deck
(TU Delft V_i6). Score: strict paper RRMS on the held-out 4 K Vb = 0 curves, measured-only inclusion; A1 rule in
advance.
Decision (NIST DEV; TU Delft replication only): K1 GMR(A1, A0) primary, K2 GMR(A1, A0_rtdeck), K3 GMR(A1, R_pop), K4
GMR(A2, A1), K5 GMR(A3, A1); device bootstrap, Bonferroni level 0.99. Supported (K1-K3 lower): continue to the
December TEST protocol with K1-K3; partly supported (K1 lower only with some of K2, K3): continue, reduced; not
supported (K1 not lower): stop, TEST stays locked; invalid (V1 or V2 fails): stop. TEST primary arm A1 unless K4 / K5
is lower; one variant is adopted for TEST if its 95% CI vs A1 lies below 1. K2 was added after the smoke (A0 RRMS about
7 vs A0_rtdeck about 0.8 on 6 NIST devices) so that the decision does not rest on K1 alone (D14).
Smoke (`smoke_runs.json`; 6 NIST and 7 TU Delft DEV devices, start nfev 100, M_bias 1 seed 200 steps, 2 workers; not
used for any decision except D14): every mode runs end to end, 0 non-finite start records, 0 simulator timeouts;
start 21 s (NIST) / 15 s (TU Delft), lodo 5 s / 11 s.
Queue (inserted ahead of the I10-2 refit jobs, after the running I10_2_inner): I10_1_start_nist (268 fits, about 30-40
min), I10_1_start_tud (45 fits, about 9 min), I10_1_lodo_nist (61 folds x 5 M_bias seeds, about 25-35 min),
I10_1_lodo_tud (about 7-8 min), I10_1_decide; total about 75-95 min.

### I10-2 amendment A1 (2026-10-21 10:15)

Amendment `out/research/explore/I10_2/prereg_amendment_A1.json` in force from 7a93dfc (committed alone; code 9cb5c9e),
before I10_2_decide and before any S_loss / S_refit outer score was computed or read (known before: the interim S_gate /
S_cv results and the sky130 S_loss selections in the inner log). Secondary report of C1 under (a) the TU Delft device
bootstrap, (b) 95% per-dataset CIs, (a+b), for the 4 selectors and fixed mlp16; the primary rule and labels are
unchanged; decide prints both, labelled, and refuses without the committed amendment. Reason: the 7-group TU Delft cluster
bootstrap at 1 - 0.05/11 gives CIs too wide for C1 to pass for most selectors. Limitation: no new NIST room-temperature
class runs in I10-2 (menu there stays {none, poly3, mlp16}). TEST planning (December): DEV runs of lin / gam / mlp8 /
mlp16 on the datasets I9-2 did not cover (IHP 233 / 343 / 398 K, TU Delft 295 K, NIST TEST counterparts) are needed for
a frozen multi-class selector and will be pre-registered separately. Queue: the I10-1 jobs were inserted ahead of the
I10-2 refits (order after I10_2_inner: I10_1_start_nist, I10_1_start_tud, I10_1_lodo_nist, I10_1_lodo_tud, I10_1_decide,
then I10_2_refit x 4, I10_2_decide); one job at a time.

## I10-1 outcome (2026-10-21 12:50): 4 K I-V from room-temperature data, invalid run (stopped by rule)

Queue: I10_1_start_nist 2335 s, I10_1_start_tud 751 s, I10_1_lodo_nist 2100 s, I10_1_lodo_tud 488 s, I10_1_decide 0 s
(94.6 min; estimate 75-95 min). Outputs `out/research/explore/I10_1/{decision.json, start_nist, start_tud, lodo_nist,
lodo_tud}` (record .npz files and `work/` nested records gitignored).
Decision (`decision.json`): outcome "invalid", action "stop: fix and re-register". NIST DEV: V1 holds (66/67; A1 rule
excluded etest_pfet_g5v0d10v5_W0p42_L8_m8395p10, non-finite C0_RT). V2 fails: failed A0 0, A0_rtdeck 0, A1 8, A2 8, A3
8, R_pop 12 (limit 10% of 67 = 6.7; A1 / A2 / A3 11.9%, R_pop 17.9%); paired K1 58, K2 58, K3 51, K4 58, K5 55 (limit 54;
K3 76%). V3 holds (guard ok in start, lodo, decide), V4 0 timeouts. TEST stays locked; no TEST arm or variant.
TU Delft replication: valid (V1 45/45; A3 2 failed, others 0).

Diagnosis (post hoc; `scripts/research/explore/i10_1_diagnose.py`, `diagnosis/diagnosis.json`; one ngspice .op per
failed A1 / A3 / R_pop record and for the excluded C0_RT record, 29 runs, 0.9 s; analytic check of every stored card).
Every failed record stops ngspice at a BSIM4 parameter check of a size-dependent value; no other error.
- A1 (A2 uses the A1 card; the A2 failures are the same 8 devices): 6 "Lpeb < -Leff" (nfet W 0.75 / 1 / 3 / 20 at L 0.4
  and W 0.75 / 20 at L 0.45; bin-stretched, bin L min 0.5 um) and 2 "Pclm <= 0" (pfet W0.42 L20, in bin; pfet W0.75
  L0.4, stretched).
- A3: 6 Lpeb (nMOS L 0.40-0.45 um), 2 Pclm (pMOS). R_pop: 12 Pclm <= 0 (4 nMOS, 8 pMOS). C0_RT of the A1-rule device:
  Pclm <= 0 (a non-included RT curve was non-finite, so the RT fit did not see it).
- Mechanism: the I8-1 box bounds the base parameter P0 (log scale, sign kept); the bin's l / w / p terms give the
  effective value P = P0 + lP/Leff + wP/Weff + pP/(Leff Weff) (binunit 2). At C0_RT the failing nMOS devices already
  have Lpeb/Leff -0.82 to -1.00 and the failing pMOS devices Pclm 0.10-0.53; the training median shift of the base
  value (or the R_pop coordinates) moves the effective value past the limit. This is the mechanism of the 10 devices
  with L 0.30-0.40 um excluded before the pre-registration (D10), at L 0.40-0.45 um.
- The analytic check (no simulation) predicts every failure of A1 8/8, A3 8/8, R_pop 12/12, A1_L 10/10, V_ind 13/13 and
  C0_RT 1/1; 6 more R_pop cards sit at Lpeb = -Leff (relative 5e-11, finite because the check is strict). V_sub7 (7
  parameters, no PCLM / LPEB) has 0 failures. V_mean has 1 and V_deck 12 failures not explained by these two checks
  (variants only; not diagnosed).
- Class: (b) the shift moves parameters outside BSIM4's valid range; not a code bug (the validity counts, contrasts and
  labels recompute from the per-device scores). Fixable by a pre-specifiable projection: after every shift (A1, A1_L,
  A3, R_pop, variants, nested A1 records) and inside the start polish, clamp P0 of PCLM and LPEB (and the other BSIM4
  fatal checks evaluated the same way, e.g. LPE0, DROUT) so that the effective value keeps a fixed margin (for example
  Pclm_eff >= 0.01, Lpeb_eff >= -0.9 Leff); computed from the card text, no simulation.

Descriptive NIST values (not decisions): mean RRMS over finite devices A0 4.023, A0_rtdeck 0.460, A1 0.411, A2 0.379, A3
0.468, R_pop 0.381, R_oracle 0.066 (medians A0_rtdeck 0.330, A1 0.301, R_oracle 0.031). K1 0.299 [0.211, 0.412] lower,
K2 0.871 [0.730, 1.053] inconclusive, K3 0.950 [0.824, 1.082] inconclusive, K4 0.819 [0.745, 0.890] lower, K5 1.143
[0.887, 1.468] inconclusive (99% CI). Per family K2: nMOS 0.628, pMOS 1.235. Gap fraction toward R_oracle A1 0.35, A2
0.41, A3 0.30. Variants vs A1: V_ind 1.450, V_mean 1.570, V_sub7 1.113, V_deck 2.243 (none adopted).
TU Delft (valid): K1 1.359 inconclusive, K2 2.418 [1.182, 5.597] higher, K3 0.297 lower, K4 0.838 lower, K5 1.722
higher; mean RRMS A0_rtdeck 0.407, A1 6.471, R_oracle 0.064. TU Delft pMOS has 2 geometry groups, so each pMOS fold
trains on the other group only; the 8 pfet W1 L0.04 devices have A1 RRMS 28.7-46.8. Direction agrees with NIST on K3,
K4, K5.

Cost of a re-registered run (estimate, not run): projection code and unit tests; compute on 6 workers about 40 min
(lodo NIST 35 min with the stored start fits, refit of the one C0_RT card, decide; TU Delft unchanged) or about 95 min
for a full rerun of start and lodo on both datasets (the measured 94.6 min).
What a re-registered run can change: it adds the 8 A1 failures (15 for K3) to 58 (51) paired devices. With the observed
spread (SD of ln ratio 0.553 for K2, 0.381 for K3; normal approximation) K2 becomes "lower" only if the added devices
have GMR(A1, A0_rtdeck) below about 0.64, and K3 only if GMR(A1, R_pop) on the added devices is below about 0.70. The
added devices are bin-stretched short-L devices whose projected cards sit at the check limits. K1 rests on A0, whose
mean RRMS is 4.02 (BSIM4 temperature equations at -269.15 C); K1 "lower" does not separate A1 from A0_rtdeck.

Recommendation: stop I10-1; do not re-register. Reasons: (1) on the 58 scored NIST devices A1 (0.411) is close to
A0_rtdeck (0.460) and R_pop (0.381), K2 and K3 are inconclusive, and A1 closes 35% of the log gap to R_oracle (0.066);
(2) on the valid TU Delft replication A1 is above A0_rtdeck (K2 2.418, higher); (3) the NIST DEV values are now seen, so
a re-registered DEV run is not blind; the best expected outcome is "partly supported" with K1 alone, whose control is
A0. Kept from I10-1: the BSIM4 effective-parameter check, which any later card-shift method needs before its
pre-registration, and K4 (A2 below A1 on both datasets, 0.819 / 0.838), consistent with the I8-1 / I9-1 M_bias results.
The TEST devices stay locked.

Report batch `out/research/report/2026-10-21_i10_1/` (`scripts/research/plot_explore_i10_1.py`; recomputes aggregates,
validity counts, K1-K5 GMR and CIs per family and cluster, labels and gap fractions for NIST and TU Delft and checks them
against results.json and decision.json): table 01 decision and validity, table 02 arms, figure 03 forest K1-K5
(descriptive), table 04 per family, figure 05 and table 06 TU Delft replication, figure 07 per-device RRMS and gap to
oracle, table 08 failure diagnosis, figure 09 effective Lpeb / Pclm, figures 10-11 example I-V
(`scripts/research/explore/i10_1_examples.py`: stored A0_rtdeck, A1, A2, R_oracle cards of
etest_nfet_g5v0d10v5_W20_L0p5_m8363p5 and etest_pfet_g5v0d10v5_W1_L2_m8364p10, nearest the type median ln(A1 /
A0_rtdeck); 0.6 s; RRMS reproduced with relative difference 0).
Site: new claim `i10-1` (status "invalid run (stopped by rule)", a new status value in build_site.py) in the Exploration
tab, status item and decision table. Site rebuilt, not published.

## I10-2 outcome (2026-10-21 14:45): class selection for the shared bias-only correction, stop (pre-registered rule)

Queue: I10_2_inner 2372 s (sky130 77 K 453 s, NIST 4 K 995 s, TU Delft 4.13 K 810 s, IHP 300 K 114 s), I10_2_refit
sky130 77 K 1384 s, IHP 300 K 1147 s, TU Delft 4.13 K 2017 s, NIST 4 K 3372 s, I10_2_decide 2 s (the I10-1 jobs ran
between the inner job and the refits). Outputs `out/research/explore/I10_2/{decision.json, decision.md,
inner/<dataset>/{results, corrections}.json, refit/<dataset>/results.json}`. Pre-registration in force from 55c4cfc;
amendment A1 in force from 7a93dfc (decide refused without it; guard commits recorded in decision.json).
Validity: V_rows pass on all 4 datasets; V_fold max |lin coefficient - stored I9-2 lin fold model| 0 on all 4
datasets; V_s3 S3 fold keys equal the I9-2 folds; 0 failed inner refits of 4948 (no class blocked in any fold).
Primary (in-force rule): decision "stop", frozen selector none, supported none. Pooled GMR (Bonferroni CI, level 1 -
0.05/11) vs C0_refit / vs mlp16 / vs S_gate:
- S_loss 0.838 [0.778, 0.900] / 0.914 [0.827, 0.992] / 0.938 [0.917, 0.958]; partly supported (C2 holds; C1 fails on
  TU Delft 0.993 [0.871, 1.089]).
- S_refit 0.834 [0.776, 0.897] / 0.910 [0.828, 0.979] / 0.934 [0.912, 0.959]; partly supported (C1 fails on TU Delft
  0.976 [0.877, 1.089] and IHP 300 K 1.020 [0.994, 1.092]).
- S_gate 0.893 [0.836, 0.951] / 0.974 [0.885, 1.049]; not supported (C1 fails on TU Delft 1.027 [0.911, 1.144]; C2
  fails).
- S_cv 0.814 [0.742, 0.888] / 0.888 [0.797, 0.964] / 0.912 [0.865, 0.948]; partly supported (C1 fails on TU Delft 0.939
  [0.736, 1.164]); not strictly nested (D9).
Per dataset vs C0_refit: sky130 77 K S_loss 0.741, S_refit and S_cv 0.727, S_gate 0.769; NIST 4 K S_loss, S_refit, S_cv
0.670, S_gate 0.806; IHP 300 K S_loss and S_gate 1.000 (none in every fold), S_cv 0.962 [0.891, 1.009].
Selections (outer folds): sky130 S_loss gam 10 / mlp8 2, S_refit and S_cv gam 12, S_gate mlp16 12; NIST S_loss, S_refit,
S_cv gam 53, S_gate mlp16 53; TU Delft (7 groups) S_loss none 2 / gam 2 / mlp8 3, S_refit lin 3 / gam 3 / mlp8 1, S_cv lin
1 / gam 6, S_gate none 2 / mlp16 5; IHP (18) S_loss and S_gate none 18, S_refit none 9 / lin 9, S_cv none 8 / lin 10. No
selector picks mlp16 except S_gate, whose menu is {none, mlp16}.
S_loss vs S_refit: same class in 75/90 outer folds (sky130 10/12, NIST 53/53, TU Delft 3/7, IHP 9/18); S_refit vs S_cv
82/90. New compute: S_loss 3.89 CPU-h (2128 inner trainings), S_refit 17.04 CPU-h (plus 4948 inner refits, 13.15 CPU-h;
plan 3.73 + 13.15); S_gate and S_cv 0 (stored values). On these 4 datasets the refit-based score changes the pooled GMR
from 0.838 to 0.834 at 4.4 times the compute.
Secondary (amendment A1; the primary label and rule are unchanged), C1 per variant: primary cluster Bonferroni, (a)
TU Delft device bootstrap Bonferroni and (b) cluster 95%: every selector and fixed mlp16 fail (TU Delft; S_refit and
fixed mlp16 also IHP). (a+b) device bootstrap 95%: S_cv passes (TU Delft 0.939 [0.862, 1.016], IHP 0.962 [0.914,
1.001]); S_loss fails on TU Delft (upper 1.037), S_refit on IHP (1.021), S_gate on TU Delft (1.071), fixed mlp16 on IHP
(1.214 [1.008, 1.539]). Fixed mlp16 fails C1 under every variant.
Other secondaries (95%, descriptive): tie sensitivity (TIE 0 / 0.01 / 0.02) pooled vs C0_refit S_loss 0.841 / 0.834 /
0.842, S_refit 0.832 / 0.834 / 0.832, S_cv 0.814 / 0.824 / 0.824; S_cv full 13-option menu 0.822 (TU Delft picks poly5
once); I8-2 dataset-level gate 0.873; TU Delft device bootstrap pooled S_loss 0.838 [0.784, 0.898]; NIST room
temperature (I9-1; menu none / poly3 / mlp16): S_gate and S_cv mlp16 87/87, 0.797 [0.749, 0.859]. Post hoc: oracle class
gam pooled, per dataset gam / gam / gam / lin; GMR vs the per-dataset oracle S_cv 1.013 [0.984, 1.052], S_refit 1.038,
S_loss 1.042, S_gate 1.111; fixed gam vs C0_refit 0.889 [0.808, 0.982].
Consequence (pre-registered): selection of the class from training data is not supported on DEV; the deployed rule stays
the I8-2 gate between C0_refit and M_bias; no class selector goes to TEST, and the DEV runs of lin / gam / mlp8 / mlp16
on the datasets I9-2 did not cover are not needed for I10-2.
Report batch `out/research/report/2026-10-21_i10_2/` (`scripts/research/plot_explore_i10_2.py`; recomputes the S_loss /
S_refit fold scores from the per-device inner log ratios and the selections from the scores, the S_gate / S_cv
selections from the stored S3_f and I9-2 rows, and every primary and secondary GMR, label, the rule, the A1 variants,
counts, agreement, tie sensitivity, NIST room temperature and oracle references from the per-device rows; all equal to
decision.json): table 01 primary result, figure 02 forest vs C0_refit per dataset, figure 03 pooled contrasts, figure
04 selections per fold and counts, figure 05 S_loss vs S_refit agreement and cost, table 06 compute, table 07 amendment
A1 secondary C1, table 08 secondary analyses.
Site: new claim `i10-2` (status "refuted"; headline "stop (pre-registered rule): no selector supported") in the
Exploration tab, status item and primary table. Site rebuilt, not published.

## sky130 77 K REAL_TEST opened by user decision; I3-2 recipe on all 18 devices (2026-10-21 16:28)

User decision (2026-10-21, verbatim): "Nahhh don't lock the real data the point is the fit the real data". The 6
sky130 77 K REAL_TEST devices (nmos_L0p19_W7, nmos_L1_W1p6, nmos_L8_W1p6, pmos_L0p35_W1p6, pmos_L0p5_W0p64,
pmos_L8_W0p84) are opened; the December TEST-stage prediction for sky130 77 K (I8-2 Part A) can no longer be checked
on held-out data. The other datasets' TEST splits are unchanged.
Starting-card records of the 6 devices: `scripts/research/explore/e1_collect_real.py --split test`
(`out/research/explore/E1/real_test.json`; strict RRMS equal to `card_eval.score_card`, all 6).
Run: `scripts/research/explore/i3_2_bias.py --mode real --devices all18 --name all18 --arms M_bias --workers 6`
(new flag; the default `--devices dev` path is unchanged). Leave-one-device-out over 18 devices, I3-2 recipe
unchanged (g trained on the other 17 starting-card records, 5 seeds, 2000 steps; held-out refit from direct43_r3_fd
with g in the loop, max_nfev 400, irls 2, diff_step 0.01); C0_refit and T1_inloop are new runs (T1 per fold fitted on
the other 17 records with `e1_correction.fit_arm`; on a REAL_DEV fold this reproduces the stored E1 table exactly).
Not pre-registered. 526 s (training 330 s, 54 refits 191 s), 0 failed devices.
All-device (combined) [median]: published card 0.2751 (0.2595) [0.2304]; starting card 0.1044 (0.0976); C0_refit
0.1043 (0.0975) [0.0820]; T1_inloop 0.0744 (0.0703) [0.0672]; M_bias 0.0732 (0.0691) [0.0643]. M_bias below
C0_refit on 16/18 devices; GMR(M_bias, C0_refit) 0.755 [0.673, 0.852]; GMR(M_bias, T1_inloop) 0.974 [0.945, 1.001],
11/18. The 6 former TEST devices: C0_refit 0.0985, T1_inloop 0.0589, M_bias 0.0590. On the 12 REAL_DEV devices:
C0_refit 0.1072 (difference to the stored I2-1 rows 0.0), M_bias 0.0803 (I3-2, trained on 11: 0.0806), T1_inloop
0.0822 (0.0820). Full model trained on all 18: `out/research/explore/I3_2/all18/m_bias_sky130_all18_full.va`
(209 lines, max |g difference| to Python 2.4e-11). Outputs `out/research/explore/I3_2/all18/{results, corrections,
heldout_cards}.json`, log `I3_2/logs/all18.log`.
Re-simulation for the slide deck (`scripts/research/explore/mbias_deck_resim.py --dataset sky130`, 4 workers, 24 s):
published, starting, C0_refit and M_bias (Python x exp(g)) RRMS reproduced with max |difference| 0.0 on 18/18; the 18
fold models compiled to OSDI (OpenVAF 23.5.0) and run in the ngspice-41 OSDI build give all-device 0.073171 vs
0.073163 (max per-device |difference| 1.7e-5). Deck `slides/cryo_ml_mbias_real_data.pptx` (27 slides,
`scripts/make_mbias_slides.py`, figures `scripts/research/explore/mbias_deck_plots.py`).

## I11-2 pre-registration in force, code, smoke, queue (2026-10-21 17:47): temperature-input bias-only correction, held-out temperature

Started by the user's default choice (no reply by 17:00). Question: does a shared bias-only correction
g(is_pmos, |Vgs|/VDD, |Vds|/VDD, T), trained on other devices at other temperatures, lower the held-out RRMS of a
device at a temperature absent from training, compared with C0_refit and with the same correction without the
temperature input? Pre-registration `out/research/explore/I11_2/prereg.{json,md}` in force from 964a828 (committed
alone, before any I11-2 held-out refit); code 02df93c (`src/cryoml/explore/i11_2_temp.py`,
`scripts/research/explore/i11_2_temp.py`, `tests/test_i11_2_temp.py`, 15 tests).
Data (D1, DEV only): IHP SG13G2 I5-1 DEV 18 DUTs at 233.15 / 300.15 / 343.15 / 398.15 K and TU Delft I6-1 DEV 45
devices (7 geometry groups) at 4.13 / 295.0 K, stored starting-card records. The TEST devices of both datasets are not
read: reading them cannot be undone and would void the I8-2 TEST predictions of 6 dataset-temperatures.
Protocol: held-out (device, temperature) record; training excludes the held-out unit (IHP DUT, within class; TU Delft
geometry group) at every temperature and every record at the held-out temperature (except the reference M_bias_in);
the code raises on a violation. Held-out card: the device's own starting card at the held-out temperature refit with
g frozen in the loop (max_nfev 400, irls 2, diff_step 0.01), strict paper RRMS. t = ln(T / 300 K), standardized.
Arms: M_bias_T (4-16-16-1 MLP, M_bias recipe with t as fourth input; IHP), M_bias_pool (M_bias on the same records),
M_bias_near (stored I5-1 fold M_bias at the nearest training temperature in ln T: 233->300, 300->343, 343->300,
398->343), TU Delft M_bias_X (stored I6-1 fold M_bias of the other temperature; with 2 temperatures pool = near = X),
references M_bias_in (all temperatures, unit excluded) and M_bias_sameT (stored), IHP secondary lin_T / lin_pool.
Primary contrasts (Bonferroni over 8, level 0.99375): IHP interpolation (300, 343 K) and extrapolation (233, 398 K)
GMR(M_bias_T, X) for X = C0_refit, M_bias_pool, M_bias_near (device bootstrap over DUTs, per-DUT mean over the
stratum's temperatures); TU Delft GMR(M_bias_X, C0_refit) at 4 K and at 295 K held out (cluster bootstrap over 7
groups). Labels: IHP stratum supported (all three lower) / transfer without temperature input / loss / no gain; TU
Delft transfer gain / loss / inconclusive (reported, not in the action: with one training temperature a temperature
input is constant in training). Decision: invalid -> stop, fix and re-register; supported -> continue (second
multi-temperature dataset); transfer without temperature input -> continue, reduced (pooled g, no temperature
input); otherwise stop. Validity: V1 6 stored C0_refit rows reproduced, V2 M_bias_near at 233 / 343 K equals the
stored X300 rows, V3 <= 2 failed records per primary arm and dataset, V5 retrained stored fold corrections reproduced.
No card clamp (D9): no card is shifted, the held-out card is refit inside the stored base box (0 failed records in
I5-1 / I6-1); the I10-1 failure mechanism (shifted effective BSIM4 values) does not arise.
Secondary prediction (stats -> commit_prediction -> lodo, as I8-1 / I9-1): per IHP held-out temperature S1_loto (I7-1
S1 with the shared function estimated without the scored record's temperature and DUT) and S3_loto (1 seed, 500
steps, inner leave-one-temperature-out x 2 DUT parts) for M_bias_T and M_bias_pool; 12 predicted signs. No TU Delft
prediction.
Known before the pre-registration: I5-1 GMR(M_bias, C0_refit) 1.106 / 1.214 / 1.181 / 1.272 at 233 / 300 / 343 /
398 K and X300 1.116 / 1.197 / 1.341; I6-1 0.939 (4.13 K) / 0.797 (295 K); I9-2 / I10-2 IHP 300 K lin 0.983, gam
1.471.
Smoke (2 workers, 1 seed, 200 steps, refit nfev 60; 2 IHP DUTs at 233 / 343 K, 2 TU Delft devices; 86 s; code check
only): every mode runs, 0 failed records, V1 relative difference 0 on 2 records; M_bias_near at full settings equals
the stored X300 rows on 2 records (difference 0.0).
Queue (runner started 17:47): I11_2_validity, I11_2_stats, I11_2_commit_prediction, I11_2_lodo_ihp (954 training
tasks, 432 refits), I11_2_lodo_tud (35 training tasks, 180 refits), I11_2_decide; estimate about 50-75 min on 6 workers.

## I11-1 (2026-10-21 18:40): M_bias as an ngspice-41 OSDI add-on, sky130 77 K, 18 devices, not deployable (pre-registered rule)

Started by the user's default choice (no reply by 17:00). Pre-registration `out/research/explore/I11_1/prereg.md` /
`prereg.json` (in force from da03859, committed alone after a 2-device smoke). Code `src/cryoml/explore/i11_1_deploy.py`,
`scripts/research/explore/i11_1_deploy.py`, tests `tests/test_i11_1_deploy.py`. Element: the stored full-data Verilog-A
`I3_2/all18/m_bias_sky130_all18_full.va` compiled with OpenVAF 23.5.0 (g of the emitted text vs Python 2.5e-11), BSIM4
drain on an internal node, module `N d g s di`, total drain current I(sense) x exp(g), ngspice-41 OSDI build. Card
M_full: per device the 43-parameter card refit from direct43_r3_fd with the full-data g in the loop (I3-2 recipe; 171 s,
0 failed); in-sample all-device RRMS SPICE-native 0.06971, Python 0.06970, same card without g 0.1057 (in-sample, not
comparable with the LODO 0.0732). Run 3.4 min, 4 workers, directly (an I11-2 queue job with 6 workers ran at the same
time; the wall times in Q4 were measured under that load).
Outcome (rule): **not deployable**. Core failures: Q1 tight max relative difference 2.59e-5 > 1e-5 (4 devices; points at
1.0-2.6 nA with |dI| <= 2.8e-14 A; above 10 nA max 1.5e-6; gmin 1e-15 and nobypass do not remove it, post hoc); Q2 AC
difference 6.35e-3 > 1e-3 at one point, nmos_L100_W100 at Vgs = 0 (abs() derivative jump; OSDI one-sided, Python
central difference), max 7.3e-6 off the lines Vgs = 0 and Vds = 0 (post hoc); Q2 gm sign change introduced on 1 of 54
curves (nmos_L100_W100, |Vds| 0.05 V, |Vgs| 1.757-1.85 V, gm down to -5.5e-6 S). Passed: 0 convergence messages and 0
non-finite values in element runs; max |I(Vds = 0)| 9.6e-13 A; 0 circuits failing only with the element.
Restrictions: default-tolerance difference 1.94e-3 (reltol 1e-4: 1.04e-3); drain/source exchange changes ln I by up to
0.043 (nMOS) / 0.117 (pMOS), equal to the value predicted from g (g uses the instance source terminal and abs(V(d,s)));
Im Y_dd and Im Y_dg of the element are exp(g) times the card value (max deviation 3e-11), Im Y_gg unchanged: the element
scales the whole drain-terminal current including displacement and junction currents; 27 of 90 gds curves acquire
negative gds (pmos_L2_W5 5, pmos_L4_W7 5, pmos_L8_W0p84 4, pmos_L8_W1p6 4, pmos_L0p5_W0p42 4, pmos_L8_W5 3, nmos_L8_W1p6 2;
where measured, the average slope over |Vds| 1.0-1.85 V is positive, post hoc); wall-time ratio DC grid 1.58-1.87,
circuits 1.11-3.25 (ro5_A).
Q3 (simulator-level, no measured circuit data; published / C0_refit / M_card / M_full): inv_A V_M 0.7466 / 0.7593 /
0.7592 / 0.7598 V; ro5_A 5.322 / 5.287 / 5.297 / 5.382 GHz; mirror_p Iout/Iref at 0.9 V 0.938 / 0.992 / 0.992 / 0.969;
ro5_B (nmos_L1_W1p6 + pmos_L2_W5) stops with 'timestep too small' for C0_refit, M_card and M_full (card-level; removing the
k1 or k2 change of the pmos_L2_W5 C0_refit card removes it, smoke diagnosis); the published card runs (0.276 GHz).
Next actions (each needs a new pre-registration): source/drain-symmetric g input; exp(g) applied to the channel current
only (correction inside a Verilog-A BSIM4); training penalty on negative gm / gds; smooth abs(); equivalence criterion
as |dI| <= 1e-6 |I| + 1e-13 A. Report batch `out/research/report/2026-10-21_i11_1/`
(`scripts/research/plot_explore_i11_1.py`); site claims `i11-1` (refuted) and `i3-2-all18` (the all-18 LODO result,
verified, not pre-registered).

## I11-2 outcome (2026-10-21): temperature-input bias-only correction, held-out temperature, invalid run (V2), stopped by rule

Queue (6 workers): validity 109 s (V1 6/6 stored C0_refit rows, difference 0; V5 3/3 stored fold corrections,
difference 0), stats and commit_prediction (812d14c, before any held-out refit), lodo_ihp 1754 s (306 trained
corrections, 432 refits), lodo_tud 1454 s (180 refits, 0 timeouts), decide. 0 failed records in every arm.
Outcome (rule): **invalid**, action 'stop: fix and re-register'. V2 failed: the decide check compared 54 M_bias_near
rows with stored I5-1 X300 rows, max relative difference 0.263.
Diagnosis (post hoc, `scripts/research/explore/i11_2_v2_diagnosis.py` -> `out/research/explore/I11_2/v2_diagnosis.json`):
prereg.json states V2 over the 36 records at 233 / 343 K. `run_decide` selected every M_bias_near row whose key exists
in the stored X300 rows; I5-1 stored X300 rows at 233, 343 and 398 K, so the 18 rows at 398 K entered. At 398 K
M_bias_near uses the I5-1 lodo_343K fold correction (398 -> 343 by the prereg definition) and X300 the lodo_300K one,
so the two are different refits. 233 / 343 K: 36/36 rows equal (difference 0). All 72 M_bias_near rows use the
stored fold correction of the prereg mapping (233->300, 300->343, 343->300, 398->343). Two single-record refits of
hv_pmos_W0p3_L10 at 398 K (largest difference, 28 s): lodo_343K correction 0.006829 = stored M_bias_near, lodo_300K
correction 0.009270 = stored X300 (both differences 0). The smoke run passed V2 because it used only 233 / 343 K.
Affected: the validity flag only. The near-arm records are the pre-registered arm; no arm row and no contrast uses
X300, so the contrasts vs C0_refit, vs M_bias_pool and vs M_bias_near are computed from correct rows (descriptive
because the run is invalid). With V2 as written in prereg.json the rule would give 'not supported' (stop: no
cross-temperature work on the shared bias-only correction); post hoc, the recorded outcome stays invalid.
Descriptive primary contrasts (Bonferroni CI, level 0.99375): IHP interpolation (300, 343 K) M_bias_T vs C0_refit
1.121 [0.928, 1.477], vs M_bias_pool 1.008 [0.990, 1.029], vs M_bias_near 0.923 [0.846, 0.989]; extrapolation (233,
398 K) 1.209 [0.968, 1.614], 1.008 [0.964, 1.053], 1.010 [0.954, 1.062]; TU Delft M_bias_X vs C0_refit 4.13 K held
out 0.978 [0.885, 1.128], 295 K held out 1.460 [0.669, 2.631] (cluster bootstrap). Labels: no gain / no gain /
inconclusive / inconclusive. Per IHP temperature (95%): M_bias_T vs C0_refit 1.154 / 1.133 / 1.110 / 1.267 at 233 /
300 / 343 / 398 K; lin_pool 1.000 / 0.960 / 0.960 / 0.997. TU Delft mean RRMS C0_refit / M_bias_X / M_bias_sameT:
4.13 K 0.0575 / 0.0558 / 0.0533, 295 K 0.0255 / 0.0353 / 0.0196.
Secondary prediction: 12 of 12 predicted signs correct (all 'no gain'; observed ln GMR vs C0_refit +0.090 to +0.237).
Recommendation: stop I11-2 without re-registration. A re-registration would change only the V2 row selection (restrict
to temperatures whose nearest training temperature is 300 K), reuse the same arm rows, and give 'not supported'; the
rows are no longer unseen, and the descriptive values (M_bias_T above C0_refit at every IHP temperature, equal to
M_bias_pool within 1%) and the committed prediction agree. If the decide code is reused, the V2 selection should be
fixed and covered by a unit test. Report batch `out/research/report/2026-10-21_i11_2/`
(`scripts/research/plot_explore_i11_2.py`); site claim `i11-2` (invalid). `lodo_ihp/corrections.json` stored gzipped
(`corrections.json.gz`, 4.2 MB).

## I12-1 pre-registration in force (2026-10-21 22:10): bias-only correction g v2 as a deployable OSDI add-on, sky130 77 K, 18 devices

Pre-registration `out/research/explore/I12_1/prereg.md` / `prereg.json`, committed alone before any v2 training or
I12-1 held-out refit. g v2 changes from M_bias v1, addressing the I11-1 failures: (1) d/s-symmetric inputs
x1 = (s V(g,s) + s V(g,d) + m) / (2 VDD), x2 = m / VDD, m = sqrt(V(d,s)^2 + EPS^2); (2) exp(g) multiplies only the
static drain current: the element senses a charge-free replica of the card (xpart = -1, zero overlap / fringing /
junction capacitance parameters, terminals driven by unit-gain VCVS copies) and adds (exp(g) - 1) I_rep from d to s;
(3) penalty lambda x P on negative corrected gm / gds on a 25 mV forward-quadrant grid of each training card,
lambda = 10 V (rule: lambda x P(v1 full) = 10 x 0.00873 exceeds the v1 in-sample loss 0.0732); (4) EPS = 0.01 V;
(5) equivalence |dI| <= 1e-6 |I| + 1e-13 A per point. Everything else as M_bias v1. Part A (LODO): A1 GMR(M_bias_v2,
C0_refit) CI upper < 1 with 0 failed devices; A2 non-inferiority GMR(M_bias_v2, M_bias_v1) CI upper <= 1.05 (device
bootstrap, stored all-18 rows). Part B: I11-1 deployment checks on the full-data g v2 with M2 cards; exchange asymmetry,
capacitance and Verilog-A g checks are core. Validity V1 (trainer reproduces 2 stored fold corrections), V2 (held-out
refit reproduces 2 stored rows), V3 (18 rows). Outcome: invalid / no gain / supported / deployable with accuracy below
v1 / not deployable. Before registration: replica check on the published card (DC equal to 12 digits, imaginary AC
current 0) and the v1 penalty calibration (8.4% of grid points with negative corrected gds, 0.20% gm; card alone
0.13% gm, 0 gds).

## I12-1 code, unit tests, smoke, amendment A1 (2026-10-21 22:20)

Code `src/cryoml/explore/i12_1_symm.py` (inputs, `SymCorrection`, trainer `fit_v2` with the gm/gds penalty, Verilog-A
v2 export and evaluator, charge-free replica library, add-on netlist, Python reference model, row selections,
decision), driver `scripts/research/explore/i12_1_symm.py` (stages grids, calib, validity, lodo, deploy, decide),
tests `tests/test_i12_1_symm.py` (26 tests; they include the V1 / V2 / primary / Q3 row selections against the
prereg.json wording and a bitwise reproduction of `i3_2_bias.fit_bias` by `fit_v2` with v1 inputs and lambda 0).
Pre-checks: one full-length seed of the V1 fold pmos_L2_W5 reproduces the stored member 0 exactly (max |dW| 0);
`CorrectedProblem` feeds the held-out refit the same signed voltages as the stored records (difference 0, 2 devices).
Smoke (`out/research/explore/I12_1/smoke/`; 2 folds, 1 seed, 200 steps, refit max_nfev 20; Part B on nmos_L1_W1p6 and
pmos_L2_W5; not used for decisions): all stages run; Verilog-A g vs Python 1.4e-16, exchanged vs forward 1.4e-16;
Q1 tight 0 violations (max |dI| / (1e-6 |I| + 1e-13 A) 0.10); exchange asymmetry 5.3e-5; capacitance error 0 (Y_dd,
Y_dg, Y_gg); AC derivative error 9.0e-7; 0 gm curves and 2 of 10 gds curves (pmos_L2_W5) with an introduced sign
change; default-tolerance difference 1.6e-3; grid wall-time ratio 1.37, ro5_A 2.76; training penalty of the smoke
full model 0.00049 /V (negative corrected gds on 0.35% of grid points). Amendment A1 (`prereg_amendment_A1.json`,
before any full run): ro5_B fails at card level (C0_refit and M2_card 'timestep too small'; the M2_full run reaches the
120 s timeout), so the Q4 circuit ratio uses only circuits where both M2_card and M2_full pass, and repeats after a
failed run are skipped.

## I12-1 validity run 1 invalid (V1), amendment A2 and requeue (2026-10-21 22:20)

Queue run 1: I12_1_grids 1 s, I12_1_validity 49 s: V1 failed (max |g_new - g_stored| 1.85e-3 on M_bias|nmos_L0p15_W1p6,
1.17e-3 on M_bias|pmos_L2_W5; threshold 1e-9), V2 passed (difference 0.0 on both devices). Outcome by rule: invalid,
action 'stop: fix and re-register'. I12_1_lodo was stopped by hand during training; it wrote no output. Cause: the
driver built the training sets in sorted tag order, the stored all-18 corrections used the record load order (REAL_DEV,
then REAL_TEST); the order sets the summation order of the loss. In load order the trainer reproduces stored members
exactly (4 single-seed checks, max |dW| 0). Amendment A2 (`prereg_amendment_A2.json`, committed with the fix and before
any LODO output): every training set in load order (`fold_train_tags`, unit-tested); validity, lodo, deploy and decide
rerun with the rules unchanged. Run-1 validity kept in `validity_run1_invalid/`.
Queue (runner restarted): I12_1_validity (about 1 min), I12_1_lodo (95 seed tasks at about 100 s plus 18 refits on 6
workers, about 30 min), I12_1_deploy (about 10 min), I12_1_decide.
Run 2 (after A2): I12_1_validity 49 s, V1 max |g_new - g_stored| 0.0 on both folds, V2 |RRMS difference| 0.0 on both
devices (valid so far; V3 is checked on the LODO rows). I12_1_lodo started 22:08:44. (Correction to A2: lodo run 1 ran
22:05:24-22:06:45.)

## I12-1 outcome (2026-10-21 22:39): g v2 deployable add-on, sky130 77 K, 18 devices, not deployable (pre-registered rule)

Queue (after amendment A2): I12_1_lodo 95 seed tasks (training 1649 s) + 18 held-out refits (53 s), 0 failed devices;
I12_1_deploy compile 16 s, M2 refit 94 s, q12 15 s, q3 4 s; I12_1_decide. Validity: V1 0.0, V2 0.0, V3 18 rows: valid.
Training penalty of the full-data g v2 0.00024 /V (v1 0.00873); negative corrected gds on 0.25% and gm on 0.07% of
the training grid points.
Part A (LODO, all-device / combined / median): M_bias_v2 0.0774 / 0.0732 / 0.0534; M_bias_v1 0.0732 / 0.0691 / 0.0643;
T1_inloop 0.0744 / 0.0703 / 0.0672; C0_refit 0.1043 / 0.0975 / 0.0820; published 0.2751 / 0.2595 / 0.2304. A1
GMR(v2, C0_refit) 0.787 [0.703, 0.884], 16/18 lower: passes. A2 GMR(v2, v1) 1.042 [0.967, 1.111], 5/18 lower: CI upper
above 1.05, fails. Reported: GMR(v2, T1_inloop) 1.015 [0.930, 1.096]. Largest per-device v2 / v1 ratios: pmos_L0p35_W1p6
1.224, nmos_L1_W1p6 1.194, pmos_L8_W0p84 1.193; smallest pmos_L2_W5 0.693, pmos_L4_W7 0.750.
Part B (M2 cards + full-data g v2; M2 in-sample RRMS Python 0.0751, SPICE-native 0.0801, card only 0.1055): **not
deployable**. Passed core checks: Q1 tight 0 points violating |dI| <= 1e-6 |I| + 1e-13 A (forward and exchanged
benches, 18 devices); 0 convergence messages, 0 non-finite values; max |I(Vds = 0)| 9.6e-13 A; Verilog-A g vs Python
2.2e-16, exchanged vs forward 1.7e-16; 0 of 54 gm curves with an introduced sign change; 0 circuits failing only with the
element. Failed core checks: exchange asymmetry 1.35e-2 (nmos_L100_W100; nmos_L20_W0p64 6.5e-3); Q2 derivative
difference 2.31 (nmos_L20_W0p64; also nmos_L0p19_W7, nmos_L0p25_W1p6, nmos_L100_W100, nmos_L8_W1p6); Q2 capacitance
error 1.77e-3 (nmos_L20_W0p64, nmos_L0p19_W7). Restrictions: default-tolerance difference 1.8e-3 (13 of 18 devices above
1e-3); 2 of 90 gds curves with an introduced sign change (pmos_L8_W0p84, pmos_L8_W1p6, |Vgs| 1.85 V); wall-time ratio
DC grid 1.91-2.00, ro5_A 4.72. Q3 (published / C0_refit / M2_card / M2_full): inv_A V_M 0.7466 / 0.7593 / 0.7592 /
0.7596 V; ro5_A 5.322 / 5.287 / 5.301 / 5.335 GHz; mirror_p Iout/Iref at 0.9 V 0.938 / 0.992 / 0.992 / 0.989; ro5_B
fails at card level (amendment A1).
Outcome (rule): **not deployable**; action 'stop; diagnose the failed core checks'.
Diagnosis (post hoc, `scripts/research/explore/i12_1_diagnosis.py` -> `out/research/explore/I12_1/diagnosis/
diagnosis.json`, 18 s of single-process ngspice runs; the recorded outcome stays 'not deployable'):
(1) Q2 capacitances and the Q2 derivative failures on nmos_L0p19_W7, nmos_L0p25_W1p6, nmos_L20_W0p64, nmos_L8_W1p6:
cause in the reference run, not in the element. ngspice-41 `b4check.c` checks each BSIM4 model name once per process
(static `modname`) and clamps a size-dependent A2 > 1 to 1 (A1 to 0); `b4temp.c` rebuilds the size-dependent parameters
at every analysis. A deck with one BSIM4 model therefore uses A2 = 1 in its first analysis and the card value in every
later one; in the element deck the card and replica model names alternate, so the check and the clamp run in every
analysis. The card-only AC reference (600 op + AC in one process) used the unclamped card; the element deck the clamped
card. The four M2 cards have a2 1.888 / 1.042 / 2.003 / 1.030; two op at the same bias in one card-only process differ
by up to 0.094 (published cards 0; cards with a2 <= 1 0); on nmos_L0p19_W7 the a2 value alone produces the drift and
removing it removes the drift. Card op current of the stored AC deck vs the DC grid: up to 0.183. With the card-only
reference rerun with the replica instance present (matched check state): capacitance error max 9.9e-13, derivative
difference on the four devices max 2.8e-6.
(2) Q2 derivative on nmos_L100_W100 (1.67e-2): cause in the registered reference formula exp(g) Re Y_card + I exp(g)
dg/dV, which takes Re Y_card at 1 MHz as the static conductance. With rbodymod = 1 (body resistor network) and the
100 um x 100 um capacitances, Re Y_dg at Vgs 0, Vds 0.05 V is 5.797e-7 S at 1 MHz and 4.699e-7 S at 100 Hz
(rbodymod = 0: 4.699e-7 S at 1 MHz). The charge-free replica contributes its static conductance. With the element
identity Re Y_card + (exp(g) - 1) Re Y_rep + I_rep exp(g) dg/dV (matched state) the max difference over 18 devices is
6.6e-6.
(3) Exchange asymmetry: cause in the element. The replica drain-terminal current contains the impact-ionization
substrate current (drain to body) in the forward bench; in the exchanged bench this current flows at the module s
terminal and is not scaled. At the nmos_L100_W100 maximum (Vgs 0, Vds 1.5 V, I 7.1e-7 A) the drain non-channel current
J = |I_fwd,card| - |i(VZ)| is 1.82e-7 A (J/I 0.26); alpha0 = 0 lowers the card current by 1.82e-7 A. The predicted
asymmetry |1 - exp(-g)| J / I is 0.0136 (observed 0.0135) and 0.0065 for nmos_L20_W0p64 (observed 0.0065). The
pre-registered bound assumed J = gmin x V <= 1.85e-12 A.
Why the smoke passed: nmos_L1_W1p6 and pmos_L2_W5 have a2 <= 1, J/I <= 1e-3 and Re Y_card at 1 MHz equal to the static
value.
Side finding (affects stored results): the A2 clamp state applies to every multi-analysis card-only deck, including the
measured-curve batch deck (`spice_pdk._batch_deck`, one dc analysis per curve). For cards with a2 > 1 (the four nMOS
devices above, in the direct43_r3_fd starting cards and in the refit cards) the first curve is simulated with A2 = 1 and
the other curves with the card value. All stored 43-parameter scores (I3-2 and later) were produced with this deck
order, so they are reproducible, but the card does not define a single model for these devices. The SPICE-native M2
RRMS (0.0801, element deck: clamp in every analysis) differs from the Python value (0.0751) partly for this reason.
Recommended next actions (each needs a new pre-registration): (a) make the card unambiguous: bound a2 <= 1 in the
43-parameter box (or run each curve in a fresh process / set the clamp explicitly) and rescore the stored arms; (b)
restate the Q2 reference with one model-check state for card-only and element runs and the replica's static
conductance; (c) scale only the channel current: sense I_ds of the replica without the substrate and junction currents
(for example a second replica with alpha0 = beta0 = 0 and the junction currents off, or the source-terminal current
combined with the drain-terminal current), then rerun Part B; A2 (accuracy cost 1.042 vs v1) stays as measured.
Report batch `out/research/report/2026-10-21_i12_1/` (`scripts/research/plot_explore_i12_1.py`; 9 PNGs; the I-V figures
re-simulate the stored held-out cards and reproduce the stored LODO rows); site claim `i12-1` (refuted).

## I13-1 pre-registration in force (2026-10-21 23:59): BSIM4 A2 clamp state of the 43-parameter cards, audit and fix

Pre-registration `out/research/explore/I13_1/prereg.md` / `prereg.json`, committed alone. Clamp states: L (legacy
batch deck, first curve with A2 = 1, later curves with the card a2), C (the b4check.c clamp written into the card:
a2 > 1 -> a2 = 1, a1 = 0; every curve in the fresh-process state), U (warm-up op, every curve with the card a2; audit
only). Fix rule: a card is valid only if every curve is evaluated in one clamp state; option (b), state C, as the new
`simulate_pdk` default. Required: published cards and cards with 0.01 <= a2 <= 1 give identical scores in C and L
(difference 0, unit-tested). Audit: the six stored arms (published, start, C0_refit, T1_inloop, M_bias v1, M_bias v2)
rescored in L, C and U, per device and all-device / combined / median; validity V1 (L reproduces stored rows), V2
(C = L = U for unclamped cards), V3 (18 rows). Other b4check.c clamps on size-dependent parameters listed per stored
card. Rerun in state C: start-card records, I3-2 all18 recipe, I12-1 LODO recipe; start cards not refit. Before
registration: code draft and a mechanism check on two start cards; stored a2 > 1 on 5 devices (nmos_L0p19_W7,
nmos_L0p25_W1p6, nmos_L20_W0p64, nmos_L8_W1p6, pmos_L0p35_W1p6) in every 43-parameter arm.

## I13-1 outcome (2026-10-22 00:40): BSIM4 A2 clamp state of the 43-parameter cards, audit and fix

Code `837d07d` (`simulate_pdk(..., bsim4_state)`: "clamped" default writes the b4check.c clamp into the card,
a2 > 1 -> a2 = 1, a1 = 0, a2 < 0.01 -> 0.01; "legacy" reproduces the old batch deck; "unclamped" runs a warm-up op;
`bsim4_clamp_params`, `bsim4_check_messages`; driver `scripts/research/explore/i13_1_clamp.py`; 9 unit tests in
`tests/test_i13_1_clamp.py`). The 77 K bins have a1 = 0, no l/w/p A1/A2 terms and 0.01 <= a2 <= 1 (max 0.467), and
b4ld.c is the only user of A1/A2, so state C equals the fresh-process state.
Audit (`I13_1/audit/audit.json`, 8 s): V1 state L reproduces every stored row of the six arms (max |diff| 0.0,
published included); V2 the 83 cards with a2 in [0.01, 1] give C = L = U (0.0); V3 18 rows per arm and state. a2 > 1
in every 43-parameter arm on nmos_L0p19_W7 (start 1.877), nmos_L0p25_W1p6 (1.045), nmos_L20_W0p64 (2.002),
nmos_L8_W1p6 (1.062), pmos_L0p35_W1p6 (1.253). Stored cards rescored, all-device stored / L / C / U: start 0.1044 /
0.1044 / 0.1109 / 0.1045; C0_refit 0.1043 / 0.1043 / 0.1108 / 0.1044; T1_inloop 0.0744 / 0.0744 / 0.0785 / 0.0745;
M_bias v1 0.0732 / 0.0732 / 0.0775 / 0.0732; M_bias v2 0.0774 / 0.0774 / 0.0821 / 0.0774. Per device in state C the
4 nMOS stored cards score 0.0263-0.0797 against 0.0153-0.0491 in L (largest: nmos_L0p19_W7 start 0.0155 -> 0.0699);
pmos_L0p35_W1p6 differs below 1e-4. Other b4check.c clamps on size-dependent parameters (ckappas / ckappad < 0.02,
prwg < 0, rdsw < 0, rds0 < 0, rdswmin < 0, xn < 3 and lc < 0 with vtl given): no stored card hits one; after writing
the A2 clamp no clamp message remains. Warnings printed without a clamp: Eta0 < 0, Pdibl1 < 0, (B1 + Weff) small,
TNOIMOD = 1.
Reruns in state C (queue): start records (5 devices changed, 13 identical); I3-2 all18 recipe 600 s; I12-1 LODO recipe
1694 s; start cards not refit. C0_refit is identical to the stored rows on the 13 devices with a2 <= 1.
Headline all-18 (all-device / combined / median), old -> new: published 0.2751 / 0.2595 / 0.2304 (unchanged); start
0.1044 / 0.0976 / 0.0823 -> 0.1109 / 0.1049 / 0.0823; C0_refit 0.1043 / 0.0975 / 0.0820 -> 0.1051 / 0.0984 / 0.0820;
T1_inloop 0.0744 / 0.0703 / 0.0672 -> 0.0753 / 0.0713 / 0.0680; M_bias v1 0.0732 / 0.0691 / 0.0643 -> 0.0735 /
0.0695 / 0.0637; M_bias v2 0.0774 / 0.0732 / 0.0534 -> 0.0772 / 0.0732 / 0.0544. GMR(M_bias v1, C0_refit) 0.755
[0.673, 0.852] 16/18 -> 0.745 [0.660, 0.850] 17/18; GMR(M_bias v2, C0_refit) 0.787 [0.703, 0.884] -> 0.785 [0.693,
0.896] (I12-1 A1 passes); GMR(M_bias v2, M_bias v1) 1.042 [0.967, 1.111] -> 1.053 [0.966, 1.137] (I12-1 A2 fails,
margin 1.05); GMR(M_bias v1, T1_inloop) 0.974 [0.945, 1.001] -> 0.960 [0.934, 0.986]. The state-C rows
(`I13_1/all18`, `I13_1/lodo`, `I13_1/summary.json`) replace the stored I3-2 all18 and I12-1 LODO rows as the
headline; I12-1 Part B was not rerun. Report batch `out/research/report/2026-10-22_i13_1/`
(`scripts/research/plot_explore_i13_1.py`; 10 PNGs; the I-V figures re-simulate the start card in L and C and the
new held-out cards, each RRMS checked against the stored rows); site claim `i13-1`. Not changed: modules with their
own multi-analysis decks that do not call `simulate_pdk` (`spice_batch.simulate_pdk_multi`, `sky130_misspec`) and
the I12-1 OSDI element decks.

## I14-1 pre-registration in force (2026-10-22 01:05): Part B of g v2 with a channel-current replica, one model-check state, element-identity Q2 reference

Pre-registration `out/research/explore/I14_1/prereg.md` / `prereg.json`, commit d637500 (alone). Inputs fixed from
I13-1 state C (records, v2 fold / full corrections and held-out cards, v1 fold / full corrections and held-out cards,
C0_refit cards); nothing is trained. Fix 1: the b4check.c A1/A2 clamp is written into the card by every deck builder
with more than one analysis per process (`spice_batch.simulate_pdk_multi` / `ngspice_jacobian`,
`sky130_misspec.simulate_truth`, `i3_2_followup.simulate_sky130`, `i11_1_deploy.patched_library`,
`i12_1_symm.patched_library_v2`, `e3_temp.simulate_truth_at`; "legacy" kept for reproduction, used by
`i12_1_diagnosis.py`). Fix 2: the sensed replica has no charge (I12-1 set), alpha0 = alpha1 = beta0 = 0 with the
l/w/p terms (the nMOS bins carry nonzero lalpha0 / walpha0 / palpha0 / lbeta0 / wbeta0 / pbeta0), agidl = agisl = 0
with l/w/p terms, jss = jsd = jsws = jswd = jswgs = jswgd = 0, igcmod = igbmod = 0; a VCCS of -gmin between the replica
drain and body copies removes the junction gmin current (BSIM4 instance m = 1 in these decks: the X-line mult / m of
the six pMOS devices with 2 or 3 does not reach the BSIM4 instance). Fix 3: Q2 reference Re Y_card + (exp(g) - 1)
Re Y_rep + I_rep exp(g) dg/dV from one card + replica run. Part B rules unchanged from I12-1; M2 cards refit in state C
with the full-data g; reference arm M_bias v1 full-data g (module with the v1 abs() inputs, same element), reported,
not the outcome. Accuracy flags fixed by I13-1 (A1 passes 0.785 [0.693, 0.896]; A2 fails 1.053 [0.966, 1.137]);
element-deck RRMS of the held-out pairs reported without a rule. Reachable outcomes: invalid / not deployable /
deployable, accuracy below v1.
Replica pre-check before registration (`repcheck/repcheck.json`; 18 devices x published, start C, v2 held-out C, v1
held-out C; I11-1 grid, tight, 17 digits, 8 s): P1 replica isub, igidl, igisl and gate currents exactly 0, junction
current = gmin x V to 6e-16 A; P2 sensed current vs the replica's own BSIM4 `id` max 1.7e-12 A (solver tolerance;
relative up to 1.1e-4 at 1 nA); P3 replica channel vs card channel within 1e-12 relative on every forward point for 2,
6, 5, 6 of 18 devices per set, max 1.2e-3 forward (nmos_L100_W100, card Isub up to 1.2e-4 A) and up to 218 in the
reverse quadrant (forward-biased drain junction). The requested criterion (replica drain current = card channel
current within 1e-12 relative) is therefore met only in part: the card's own substrate and junction currents flow
through its rbodymod = 1 body resistor network (5 x 50 ohm) and change its internal body voltage and channel current;
the replica without these currents has a different internal body voltage, and a replica with them carries them in
its drain-terminal current. Registered consequence: the element scales the replica channel current, and every Part B
reference uses I_rep measured in a card + replica run.

## I14-1 code, unit tests, smoke, queue (2026-10-22 01:07)

Code f40c6b7: `src/cryoml/explore/i14_1_chan.py` (replica, gmin VCCS, card / card_rep / elem netlists, v1 Verilog-A,
element-identity references, validity rule), driver `scripts/research/explore/i14_1_chan.py` (repcheck, validity,
accuracy, deploy --arm v2|v1, decide), `tests/test_i14_1_chan.py` (16 tests; 68 with the I13-1, spice_batch,
sky130_misspec and I12-1 suites, all pass). Smoke (`I14_1/smoke/`, nmos_L1_W1p6 and pmos_L2_W5, refit max_nfev 20; not
used for decisions): all stages run; V1 0.0 on 4 rows; v2 core checks pass on the 2 devices (exchange asymmetry
3.8e-6, Q2 1.3e-6, capacitance 2.1e-15), restrictions default tolerance 1.06e-3 and runtime 4.6; v1 reference arm
fails the exchanged Verilog-A g check (0.134) and the exchange asymmetry (0.125), as expected for abs() inputs.
Queue: I14_1_validity, I14_1_accuracy, I14_1_deploy_v2, I14_1_deploy_v1, I14_1_decide.

## I14-1 outcome (2026-10-22 01:16): Part B of g v2 with the channel-current replica, not deployable (pre-registered rule)

Queue: I14_1_validity 6 s, I14_1_accuracy 13 s (36 OSDI compiles 9 s), I14_1_deploy_v2 187 s (M2 refit 147 s, 0
failed), I14_1_deploy_v1 306 s, I14_1_decide. Validity: V1 card-only library deck x exp(fold g) reproduces the 36
I13-1 rows (max |diff| 0.0); V2 68 unit tests pass; V3 18 rows per arm, 0 failed M2 refits: valid.
Part B, g v2 (M2 cards, in-sample M2 RRMS Python 0.0756, element deck 0.0756, card only 0.1081). Core checks: Q1
equivalence 0 violating points (tight, forward and exchanged, reference I_card + (exp(g) - 1) I_rep); Q1 runs **2
convergence lines** ('dynamic gmin stepping failed', 'true gmin stepping failed') in 1 of 126 element decks
(nmos_L8_W1p6, exchanged bench, tight; 0 non-finite values): **fails**; max |I(Vds = 0)| 9.8e-13 A; Verilog-A g vs
Python 2.5e-16, exchanged vs forward 3.3e-16; exchange asymmetry 1.7e-5 (I12-1: 1.35e-2); Q2 derivative vs the
element identity 1.2e-5 (I12-1: 2.31; the I12-1 formula on the same runs: 0.0217); capacitance error 1.1e-12 (I12-1:
1.8e-3); 0 of 54 gm curves with an introduced sign change; 0 circuits failing only with the element (ro5_B fails at
card level, amendment A1). Restrictions: default tolerance 1.8e-3; 2 of 90 gds curves (pmos_L8_W0p84, pmos_L8_W1p6,
|Vgs| 1.85 V); runtime DC grid 1.90-2.00, ro5_A 4.59. Q3 (published / C0_refit / M2_card / M2_full): mirror_n Iout /
Iref at 0.9 V 1.059 / 1.022 / 1.022 / 1.024; mirror_p 0.938 / 0.992 / 0.992 / 0.984.
Outcome (rule): **not deployable**; action 'stop; diagnose the failed core checks'. Accuracy flags (I13-1): A1 passes,
A2 fails.
Element-deck accuracy of the held-out pairs (reported): v2 all-device / combined 0.07714 / 0.07321 vs stored 0.07716 /
0.07323 (max per-device |d| 1.8e-4, nmos_L100_W100 and nmos_L20_W0p64); v1 0.07346 / 0.06951 vs 0.07348 / 0.06953 (max
2.5e-4); GMR(native v2, C0_refit) 0.785 [0.693, 0.896], GMR(native v2, native v1) 1.053 [0.966, 1.137]: the element's
DC change (substrate current, junction gmin term, body-voltage effect) moves no aggregate by more than 2e-5.
Reference arm M_bias v1 (full-data g, abs() inputs, same element): not deployable (exchanged Verilog-A g 0.134,
exchange asymmetry 0.125, Q2 6.6e-3, 1 gm sign change, max |I(Vds = 0)| 1.0e-12 A; 28 gds curves, default tolerance
1.9e-3, runtime 4.48).
Diagnosis (post hoc, `scripts/research/explore/i14_1_diagnosis.py` -> `I14_1/diagnosis/diagnosis.json`, 11 s; the
recorded outcome stays 'not deployable'): tight forward and exchanged grids of all 18 M2 cards in six deck variants.
No message in the card-only and the I12-1-replica variants. The two warnings on nmos_L8_W1p6 occur with the
channel-current replica in the card + replica deck (forward) and in the element deck (exchanged), with and without the
gmin VCCS: the cause is the replica with zero junction saturation current (its operating-point solution needs the
convergence aids), not the module and not the VCCS. Without the VCCS, nmos_L0p25_W1p6 also fails source stepping and
the element run violates Q1 on 1735 points; with the VCCS (I14-1 element) Q1 has 0 violations on all 18 devices.
Possible next step (needs a new pre-registration): a replica whose junction currents stay nonzero but are sensed
separately is not available from BSIM4 terminal currents (pre-check); options are a replica with small nonzero
saturation currents compensated in the module, or counting 'gmin stepping failed' warnings as a restriction when the
solution converges (rule change).
Report batch `out/research/report/2026-10-22_i14_1/` (`scripts/research/plot_explore_i14_1.py`; 12 PNGs; the I-V
figures re-simulate the published, C0_refit and v2 / v1 held-out cards and reproduce the stored rows); site claim
`i14-1` (refuted).

## I15-1 pre-registration in force (2026-10-22 01:57): re-registration of the I14-1 Part B test of g v2 with the Q1 grid swept in serpentine order

Pre-registration `out/research/explore/I15_1/prereg.md` / `prereg.json`, commit 04c7c2e (alone). Work before
registration (not used for decisions): (1) the two I14-1 warning decks split per V(g) line: the warnings occur only at
the restart of the inner sweep from V(d) = +1.85 V to -1.85 V when V(g) steps to 1.85 V; isolated ops at that bias and
sweeps that start there print none; itl1 / itl2 up to 500 leave the warnings unchanged. (2) Requested fix, a replica
with small junction saturation currents: card values x 1e-20 removes the nmos_L8_W1p6 warnings but adds 1 warning deck
in 864 on the 4 card sets x 18 devices (I14-1 replica: 0); x 1e-10 and x 1 also 1 each; x 1e-60 does not remove them;
rbodymod = 0 adds 3 of 1008. Without compensation the sensed current of these replicas contains the forward-biased
drain-junction current in the reverse quadrant (up to 0.039 A). Exact subtraction through a junction-only replica
(u0 = 1e-30) and a CCCS is exact to 5.7e-5 of the Q1 tolerance but the forward tight card + replica run aborts on
the nmos_L8_W1p6 M2 card (difference of two ~0.04 A currents in the sense branch); body-current routing gives
warnings. Result: no replica change met the pre-condition 'convergence unchanged elsewhere'. (3) Registered change:
every Q1 grid deck (card, card + replica, element; both benches; three tolerances; Q4 grid) sweeps V(d) up on even
and down on odd V(g) lines (one dc per line, rows appended and reordered); replica, module, references, thresholds and
rule as I14-1. Pre-check: 0 / 0 warning decks over 1296 decks per order (nested / serpentine), nmos_L8_W1p6 M2
(disclosed) nested 2 / serpentine 0; serpentine - nested tight max 2.1 x (1e-6 |I| + 1e-13 A). (4) Runtime
decomposition on ro5_A: card 0.24 s, card + replica 0.48 s, element 1.13 s, element with a trivial module 0.52 s; the
network evaluation in the OSDI module is about 0.6 s of the 0.89 s added; replica-side reductions <= 2.5 % without new
warnings; OpenVAF --target_cpu crashes; no runtime variant registered.

## I15-1 outcome (2026-10-22 02:09): Part B of g v2 with serpentine Q1 grids, deployable with restrictions; outcome 'deployable, accuracy below v1 (not non-inferior)' (pre-registered rule)

Code 6df4cab (`src/cryoml/explore/i15_1_serp.py`, `scripts/research/explore/i15_1_chan.py`, 3 unit tests). Queue:
I15_1_validity 7 s, I15_1_accuracy 14 s, I15_1_deploy_v2 194 s, I15_1_deploy_v1 313 s, I15_1_restrict 7 s,
I15_1_decide. Validity: V1 0.0 (36 rows), unit tests pass, 18 rows per arm, 0 failed M2 refits; M2 v2 cards identical
to the I14-1 ones (max relative difference 0). Part B v2: Q1 0 violations, **Q1 runs 0** (I14-1: 2), max |I(Vds = 0)|
9.8e-13 A, Verilog-A g 2.5e-16 / exchanged 3.3e-16, exchange asymmetry 1.7e-5, Q2 vs element identity 1.2e-5,
capacitance 1.1e-12, 0 of 54 gm curves, 0 circuits failing only with the element: every core check passes.
Restrictions: default-tolerance difference 1.81e-3 (16 of 18 devices above 1e-3; r4 max 9.9e-4, tight 1.8e-5);
2 of 90 gds curves (pmos_L8_W0p84 506 points, pmos_L8_W1p6 914 points of dI/dV(d,s) < 0 at V(g,s) = -1.85 V,
|V(d,s)| 0.77-0.97 V; the card alone has none and I_card exp(g) of the same sweep has the same points: the sign change
comes from g, not from the element); runtime DC grid 1.84-2.08, circuits inv_A 1.76, inv_B 1.95, ro5_A 4.30,
mirror_n 1.19, mirror_p 1.30 (ro5_A on the M2 cards: card 0.26 s, card + replica 0.52 s, element 1.21 s; load time
0.13 / 0.29 / 0.94 s). Label **deployable with restrictions**; outcome **deployable, accuracy below v1 (not
non-inferior)**; action 'stop v2; report the accuracy cost of the deployment constraints' (A2: GMR(v2, v1) 1.053
[0.966, 1.137]). Element-deck accuracy (unchanged decks, reproduces I14-1): v2 0.07714 / 0.07321 vs stored 0.07716 /
0.07323; v1 0.07346 / 0.06951. Reference arm v1: not deployable (exchanged g 0.134, asymmetry 0.125, Q2 6.6e-3, 1 gm
sign change; 28 gds curves). Q3: mirror_p Iout / Iref at 0.9 V published 0.938, C0_refit 0.992, M2 card 0.992, M2 +
element 0.984; ro5_A 5.322 / 5.287 / 5.291 / 5.353 GHz. Limitation (stated in the prereg): nested sweeps with the
3.7 V restart at tight tolerance can print gmin-stepping warnings for the element deck. Report batch
`out/research/report/2026-10-22_i15_1/` (`scripts/research/plot_explore_i15_1.py`, 13 PNGs; I-V figures re-simulate
the published, C0_refit and v2 / v1 held-out cards, RRMS checked against the stored rows); site claim `i15-1`.

## I15-2 pre-registration in force (2026-10-22 02:05): which g v2 change costs accuracy; ablation and training-data-only selection of lambda

Pre-registration `out/research/explore/I15_2/prereg.md` / `prereg.json`, commit 7924961 (alone). New 18-fold LODO
variants (I12-1 recipe, 5 seeds, held-out refit in state C): sym_e0_l0, sym_e1_l0, sym_e0_l10, sym_e1_l1, sym_e1_l2,
sym_e1_l5 (symmetric inputs, EPS 0 or 0.01 V, lambda in V); stored v1 and g v2 (= sym_e1_l10). Contrasts with
pre-registered labels (cost: CI lower > 1; no cost beyond the margin: CI upper <= 1.05; else inconclusive): input form
GMR(sym_e0_l0, v1); EPS at lambda 0 and 10; penalty at EPS 0 and 0.01; lambda curve. Selection: per outer fold, 3
inner folds (sorted by polarity, tag; i mod 3), one member (seed 0) per inner fold and lambda in {0, 1, 2, 5, 10},
inner held-out refits, lambda* = argmin mean ln RRMS (ties to the larger lambda); the selected row of device h is
the sym_e1_l{lambda*_h} LODO row. Primary GMR(selected, M_bias v1), device bootstrap, non-inferiority margin 1.05.
Negative gm / gds grid fractions for every variant. Code d9e79ac (8 unit tests); smoke (2 folds, 200 steps, not used
for decisions) ran all stages; queue: I15_2_validity, six lodo variants, I15_2_inner (about 2 h), I15_2_decide.

## I15-2 outcome (2026-10-22 07:23): which g v2 change costs accuracy; nested lambda selection non-inferior to M_bias v1 (pre-registered rule)

Queue: I15_2_validity 110 s (V1 seed 0 of fold pmos_L2_W5 reproduces the stored member 0 for the g v2 and v1 recipes,
max weight difference 0; V2 refit of the stored g v2 fold corrections of nmos_L1_W1p6 and pmos_L2_W5, difference 0);
LODO variants 453-456 s (lambda 0) and 2010-2131 s (lambda > 0); I15_2_inner 8990 s (285 single-member trainings
3183 s, 1620 inner refits 5807 s, complete); I15_2_decide 38 s. Valid.
Variants, all-device / combined / median, GMR vs M_bias v1 [95% CI] (devices lower): v1 0.0735 / 0.0695 / 0.0637;
sym_e0_l0 0.0740 / 0.0700 / 0.0636, 1.006 [0.997, 1.015] (3); sym_e1_l0 0.0736 / 0.0696 / 0.0624, 0.999 [0.985, 1.013]
(7); sym_e0_l10 0.0775 / 0.0735 / 0.0557, 1.055 [0.970, 1.135] (3); sym_e1_l1 0.0731 / 0.0692 / 0.0554, 0.985 [0.948,
1.019] (8); sym_e1_l2 0.0726 / 0.0688 / 0.0549, 0.973 [0.931, 1.013] (9); sym_e1_l5 0.0734 / 0.0696 / 0.0549, 0.984
[0.926, 1.041] (8); g v2 = sym_e1_l10 0.0772 / 0.0732 / 0.0544, 1.053 [0.966, 1.137] (4); C0_refit 0.1051 / 0.0984.
Contrasts (labels by rule): input form abs -> sym 1.006 [0.997, 1.015] no cost beyond the margin; EPS 0 -> 0.01 0.994
[0.982, 1.003] (lambda 0) and 0.998 [0.978, 1.018] (lambda 10), no cost beyond the margin; penalty 0 -> 10 V 1.049
[0.963, 1.130] (EPS 0) and 1.054 [0.968, 1.136] (EPS 0.01), inconclusive. No change is named as costing accuracy
(no CI lower bound above 1); the point estimates place the g v2 difference in the lambda = 10 V penalty; lambda 1-5 V
give GMR vs v1 0.973-0.985.
Selection: lambda* = 1 V on 10 folds, 2 V on 6, 0 V on pmos_L4_W7, 10 V on pmos_L8_W5; full-data lambda* 1 V.
Primary GMR(selected, v1) **0.997 [0.962, 1.027]**, 7/18 lower: **selected variant non-inferior to M_bias v1**
(CI upper <= 1.05; not below 1). Selected all-device / combined / median 0.0739 / 0.0700 / 0.0550; GMR(selected,
C0_refit) 0.743 [0.654, 0.855] 17/18; GMR(selected, g v2) 0.947 [0.885, 1.023] 13/18.
Negative corrected gds / gm on the penalty grids (% of points; fold models on the held-out starting-card grid, mean
over folds): v1 11.5 / 0.19; lambda 0 11.4 / 0.19; lambda 1 4.4 / 0.14; 2 3.5 / 0.13; 5 1.4 / 0.12; 10 1.3 / 0.11;
selected 4.7 / 0.14; on the training grids 8.1 (v1), 1.9 (lambda 1), 0.27 (lambda 10), selected 1.9; card only gds 0,
gm 0.13. Deployment of a lambda = 1 V model was not tested (I15-1 tested lambda 10 V; at lambda 10 V the two gds
sign changes of I15-1 come from g). Report batch `out/research/report/2026-10-22_i15_2/`
(`scripts/research/plot_explore_i15_2.py`, 12 PNGs; I-V figures re-simulate the held-out cards of C0_refit, v1,
lambda 0, g v2 and the selected variant, RRMS checked against the stored rows); site claim `i15-2`.

## M_bias deck training-trace fix (2026-10-22 02:16)

The deck's training-trace slide re-ran the fold pmos_L2_W5 training with its own loop over tag-sorted records
(max |g - stored fold g| 1.8e-3). `scripts/research/explore/mbias_method_plots.py` now calls the trainer of the
stored model (`i3_2_bias.fit_bias`, new per-step `trace_cb`, update sequence unchanged; 36 I12-1 / I15-2 unit tests
pass) with the training records in load order: max |g - stored g| 0, max weight difference 0 (72 s). Deck regenerated
(43 slides), commit e92946a.

## I16-1 pre-registration in force (2026-10-22 07:27): deployment test of the I15-2 selected correction

Pre-registration `out/research/explore/I16_1/prereg.md` / `prereg.json`, commit 6adec13 (alone). Model under test:
the full-data correction of the I15-2 variant sym_e1_l1 (symmetric inputs, EPS 0.01 V, lambda 1 V = I15-2 full-data
lambda*), `I15_2/lodo_sym_e1_l1` `full|sym_e1_l1`. Part B procedure, replica, module, references, serpentine Q1
grids, core checks, restrictions and thresholds of I15-1; gds sign change a restriction, gm sign change in strong
inversion a core failure (I15-1 classification). Expectation stated before the run (not a rule): more than 2 of 90
gds curves (full-data negative gds on the penalty grids 1.59 % at lambda 1 V vs 0.23 % at 10 V). Accuracy fixed from
I15-2 (GMR(selected, v1) 0.997 [0.962, 1.027]); element-deck RRMS of the 18 selected held-out pairs (fold g of
sym_e1_l{lambda*_h} + its held-out card) reported against the I15-2 rows. Validity V1 card-only deck reproduces the
I15-2 selected rows, V2 unit tests, V3 18 rows / 0 failed M2 refits, V4 deployed g. Outcomes: invalid / deployable
(restrictions listed) / not deployable. Code 64c1e54 (`src/cryoml/explore/i16_1_sel.py`,
`scripts/research/explore/i16_1_sel.py`, 6 unit tests); smoke (nmos_L1_W1p6, pmos_L2_W5, refit max_nfev 20; not used
for decisions) ran all stages.

## I16-1 outcome (2026-10-22 07:37): selected correction (lambda 1 V), not deployable (pre-registered rule)

Queue: I16_1_validity 6 s, I16_1_accuracy 6 s, I16_1_deploy 197 s (0 failed M2 refits), I16_1_restrict 1 s,
I16_1_runtime 6 s, I16_1_decide. Validity: V1 max |RRMS card-only deck - I15-2 selected row| 0.0 (18 rows); unit
tests pass; V3 18 rows, 0 failed; V4 deployed g sym_v2 / EPS 0.01 / lambda 1, I15-2 full-data lambda* 1: valid.
Part B (M2 cards, in-sample M2 RRMS Python 0.0696, element 0.0696, card only 0.1083). Core checks: Q1 equivalence 0
violating points; Q1 runs 0 messages, 0 non-finite, 0 failed runs; **Q1 max |I(V(d,s) = 0)| 1.0006e-12 A > 1e-12 A:
fails**; Verilog-A g 2.8e-16, exchanged 1.7e-16; exchange asymmetry 2.2e-5; Q2 vs element identity 1.1e-5;
capacitances 3.3e-14; gm curves with an introduced sign change 0 of 54; Q3 0 circuits failing only with the element
(ro5_B fails at card level, amendment A1). Restrictions: default tolerance 1.84e-3 (18 of 18 devices above 1e-3; r4
max 9.9e-4, tight 2.3e-5); **6 of 90 gds curves** (pmos_L2_W5 V(g,s) -0.9 V 230 points; pmos_L8_W0p84 -1.5 V 115,
-1.85 V 555; pmos_L8_W1p6 -1.5 V 543, -1.85 V 870; pmos_L8_W5 -1.5 V 126; the card alone has none on these curves and
I_card exp(g) of the card sweep has the same points: the sign changes come from g; I15-1 lambda 10 V: 2 of 90);
runtime DC grid 1.80-2.09, inv_A 1.75, inv_B 1.71, ro5_A 4.56, mirror_n 1.21, mirror_p 1.25 (ro5_A wall card / card +
replica / element ratio 4.66; rusage load 0.13 / 0.29 / 0.94 s). gm items: element and card have the same negative
strong-inversion points (316 / 315; card mobility degradation), none introduced. Label **not deployable**; outcome
**not deployable**; action 'stop; diagnose the failed core checks'.
Element-deck accuracy (reported, no rule): all-device / combined / median 0.07390 / 0.06996 / 0.05503 vs I15-2 rows
0.07391 / 0.06997 / 0.05504 (max per-device |d| 2.0e-4); GMR(element, v1 rows) 0.997 [0.962, 1.027];
GMR(element, I15-1 g v2 element) 0.947 [0.885, 1.023]; GMR(element, C0_refit) 0.743 [0.654, 0.855].
Q3: mirror_p Iout / Iref at 0.9 V published 0.938, C0_refit 0.992, M2 card 0.992, M2 + element 0.989; ro5_A M2 card
5.292 GHz, M2 + element 5.384 GHz.
Diagnosis (post hoc, `--stage vds0` -> `I16_1/diagnosis/vds0.json`; the recorded outcome stays 'not deployable'): the
failure is 1 of 2700 V(d,s) = 0 points of the default and tight forward grids: pmos_L8_W0p84, |V(g,s)| = 0.825 V,
default tolerance (no options; ngspice abstol 1e-12 A), element 1.0006e-12 A where the card + replica run of the same
grid gives 8.1e-24 A. The same device gives 3.5e-13 A at reltol 1e-4 and 8.3e-19 A at tight tolerance; the tight
maximum over 18 devices is 4.9e-17 A. The I15-1 arms on the same check: g v2 9.84e-13 A, v1 9.95e-13 A (both at
default tolerance). The threshold equals the default abstol, so at default tolerance the check compares the solver's
current tolerance with itself. Possible next step (needs a new pre-registration): evaluate the V(d,s) = 0 check at
tight tolerance only, or as a restriction at default tolerance.
Report batch `out/research/report/2026-10-22_i16_1/` (`scripts/research/plot_explore_i16_1.py`, 17 PNGs; the I-V
figures re-simulate the published, C0_refit, v1, g v2 and selected held-out cards and reproduce the stored rows); site
claim `i16-1` (refuted).

## I17-1 pre-registration in force (2026-10-22 10:35): re-registration of the I16-1 Part B test with the V(d,s) = 0 check at tight tolerance only

Pre-registration `out/research/explore/I17_1/prereg.md` / `prereg.json`, commit 3cfc9a7 (alone). The check was changed
after an observed failure: I16-1 failed only 'Q1 |I(V(d,s) = 0)| <= 1e-12 A, max over default and tight tolerance'
(1.0006e-12 A at 1 of 2700 points); the I16-1 outcome stays 'not deployable'. I17-1 evaluates this check on the tight
forward Q1 grid only (reltol 1e-9, abstol 1e-18 A, vntol 1e-12 V), threshold 1e-12 A kept. Option chosen over an
abstol-derived threshold because (1) at default tolerance the threshold equals ngspice abstol, the current tolerance of
the convergence test, so the check there does not separate an element current from the solver tolerance; (2) a factor
k x abstol would be chosen with the observed value known; (3) at tight tolerance abstol is six orders of magnitude below
the threshold. Known before registration: the I16-1 tight maximum 4.87e-17 A. Every other I16-1 rule, procedure,
threshold and validity check unchanged; every stage rerun into `I17_1`. Reported without a rule: default and r4 values
and the label under the I16-1 rule. Code 86fcbf8 (`src/cryoml/explore/i17_1_vds0.py`,
`scripts/research/explore/i17_1_vds0.py`, 3 unit tests); smoke (nmos_L1_W1p6, pmos_L2_W5, refit max_nfev 20) ran all
stages.

## I17-1 outcome (2026-10-22 10:45): selected correction deployable with restrictions under the re-registered check

Queue: I17_1_validity 6 s, I17_1_accuracy 6 s, I17_1_deploy 197 s (0 failed M2 refits), I17_1_restrict 1 s,
I17_1_runtime 6 s, I17_1_decide. Valid (V1 max |RRMS card-only deck - I15-2 selected row| 0.0 on 18 rows; unit tests
pass; V3 18 rows / 0 failed; V4 sym_v2 / EPS 0.01 / lambda 1, lambda*_full 1).
Core checks: Q1 equivalence 0 violating points; Q1 runs 0 messages, 0 non-finite, 0 failed; **Q1 |I(V(d,s) = 0)| tight
max 4.87e-17 A (nmos_L0p19_W7, |V(g,s)| 1.825 V; 0 of 1350 points above 1e-12 A): pass**; Verilog-A g 2.8e-16,
exchanged 1.7e-16; exchange asymmetry 2.2e-5; Q2 1.1e-5; capacitances 3.3e-14; gm 0 of 54; Q3 0 circuits failing only
with the element. Reported without a rule: default tolerance 1.0006e-12 A (pmos_L8_W0p84, 0.825 V; 1 of 1350 points),
r4 9.28e-13 A. Restrictions: default tolerance 1.84e-3, 6 of 90 gds curves (pmos_L2_W5, pmos_L8_W0p84 x2,
pmos_L8_W1p6 x2, pmos_L8_W5), runtime DC grid 1.84-2.07, inv_A 1.76, inv_B 1.75, ro5_A 4.60, mirror_n 1.20,
mirror_p 1.29. Label **deployable with restrictions**; outcome **deployable** (re-registered check). The same run
under the unchanged I16-1 rule: not deployable. Every Part B value equals the I16-1 run except the runtime ratio
(4.56 -> 4.60); M2 in-sample element RRMS 0.069624 in both. Element-deck accuracy (reported): 0.07390 / 0.06996 /
0.05503 (all-device / combined / median); GMR(element, v1) 0.997 [0.962, 1.027] 7/18; GMR(element, C0_refit) 0.743
[0.654, 0.855] 17/18. Report batch `out/research/report/2026-10-22_i17_1/` (`scripts/research/plot_explore_i17_1.py`,
18 PNGs; I-V figures re-simulate the published, C0_refit, v1, g v2 and selected held-out cards and reproduce the
stored rows); site claim `i17-1` (verified; the check change is stated), caveat added to `i16-1`.

## I17-2 pre-registration in force (2026-10-22 10:52): second-PDK replication of the selected g v2 recipe on measured IHP SG13G2

Pre-registration `out/research/explore/I17_2/prereg.md` / `prereg.json`, commit c071472 (alone). Dataset: IHP SG13G2
(I5-1 data, PSP103, 300 K, 18 DEV DUTs: LV 10, HV 8; TEST 10 locked). Chosen over TU Delft (measured, BSIM4, but a
generic CMOSedu example card, not a foundry PDK) and GF180 (no measured data). Simulator: ngspice-46 + psp103.osdi
(I5-1; ngspice-41 OSDI build loads OSDI 0.3 only); BSIM4 clamp not applicable. Arms, LODO within class (I5-1 folds),
held-out refit from the I5-1 starting card with g frozen (max_nfev 400, irls 2): C0_refit, M_bias v1 (I3-2 recipe,
class VDD), sym_e1_l{0,1,2,5,10} (symmetric inputs, EPS 0.01 V, VDD = class supply 1.2 / 3.3 V, penalty on 75 x 75
forward-quadrant starting-card grids to the class supply), selected (nested 3-fold lambda selection per outer fold,
I15-2 rule). Primary GMR(selected, C0_refit), device bootstrap: CI upper < 1 and 0 failed -> gain replicated; CI lower
> 1 -> worse than C0_refit; else inconclusive. Secondary GMR(selected, M_bias v1), margin 1.05. Predicted signs: ln
GMR(selected, C0_refit) > 0, ln GMR(selected, v1) < 0, ln GMR(v1, C0_refit) > 0. Part B not run: a PSP103 replica
needs SWIGATE = SWIMPACT = SWGIDL = SWJUNCAP = 0, zero overlap / fringe charges and a VCCS for the DI-BP / SI-BP gmin;
PSP103 contributes the intrinsic charges Qg, Qb, Qd unconditionally, so the sensed replica current carries
displacement current and the Q2 capacitance check fails by construction without an edited PSP103 Verilog-A; plus OSDI
0.4 vs ngspice-41 and no IHP circuits. Text error in the prereg (reported quantity only): it states '9 clusters of
DEV'; the geometry clusters are 12 (LV 5, HV 7); the code uses 12.
Code bd660cb (`src/cryoml/explore/i17_2_ihp.py`, `scripts/research/explore/i17_2_ihp.py`, 10 unit tests;
`i12_1_symm.fit_v2` `vdd` and `penalty_arrays` `grid` options, defaults unchanged). Smoke (2 folds, 200 steps, 1 seed,
refit max_nfev 20) ran all stages; one full-budget C0_refit refit reproduced the I5-1 row exactly.

## I17-2 outcome (2026-10-22 12:50): IHP SG13G2 300 K, selected g v2 worse than C0_refit (pre-registered rule); Part B not run

Queue: I17_2_pens 2 s, I17_2_base 119 s, I17_2_lodo_l0 82 s, l1 915 s, l2 / l5 / l10 about 915 s each, I17_2_inner
2984 s (270 single-member trainings 1408 s, 730 inner refits), I17_2_decide 17 s. Valid: V1 C0_refit and M_bias v1
reruns equal the I5-1 rows (max difference 0 on 18 each); unit tests pass; 18 rows per arm; inner complete; 18
penalty grids finite (kept points 4213-5545 per DUT).
All-device / median: C0_refit 0.01906 / 0.01336; M_bias v1 0.02029 / 0.01424; sym_e1_l0 0.02033; l1 0.01999; l2
0.02025; l5 0.02014; l10 0.02015; selected 0.02045 / 0.01443. GMR vs C0_refit [device CI] (lower): v1 1.214 [1.008,
1.537] (9); l0 1.218 [1.007, 1.546]; l1 1.209 [0.998, 1.536] (10); l2 1.208 [1.001, 1.531]; l5 1.233 [1.003, 1.578];
l10 1.244 [1.013, 1.590]; **selected 1.255 [1.026, 1.600] (9/18): worse than C0_refit**. Secondary GMR(selected, v1)
**1.034 [0.990, 1.107] (7/18): not shown non-inferior**. Geometry-cluster CIs (12 clusters): selected vs C0_refit
[1.000, 1.578], the other arms span 1. LV: selected vs C0_refit 1.275 [0.961, 1.879], vs v1 1.070 [0.998, 1.194];
HV: 1.232 [0.974, 1.636], vs v1 0.990 [0.964, 1.012]. lambda*: 0 V on 2 folds, 1 V on 3, 2 V on 6, 5 V on 2, 10 V
on 5. Negative corrected gds on the held-out grids 2.81-2.89 % for every arm (v1 2.89 %); on the training grids v1
1.99 %, lambda 1 1.27 %, 2 0.90 %, 5 0.45 %, 10 0.21 %, selected 0.88 %; card only 0.002 %; gm 0 everywhere.
Predicted signs: ln GMR(selected, C0_refit) > 0 correct; ln GMR(selected, v1) < 0 wrong (observed > 0); ln GMR(v1,
C0_refit) > 0 correct. Part B not run (prereg reasons). Report batch `out/research/report/2026-10-22_i17_2/`
(`scripts/research/plot_explore_i17_2.py`, 14 PNGs: setup and Part B reasons, validity, rules, arms with device and
cluster CIs, LV / HV, lambda selection, negative fractions, per-DUT RRMS and ratios, lambda curve, I-V of
lv_nmos_W0p3_L10, lv_pmos_W10_L10, hv_pmos_W0p6_L0p5, hv_pmos_W0p3_L10 with measured points, stock-card NGSpice and
re-simulated C0_refit / v1 / selected cards; RRMS equal to the stored rows); site claim `i17-2` (refuted), caveat on
`i5-1`.

## I18-1 pre-registration in force (2026-10-22 15:30): gated correction on every measured DEV dataset with stored rows

Pre-registration `out/research/explore/I18_1/prereg.md` / `prereg.json`, commit b384a42 (alone, before any S3 value
with the g v2 recipe and before any I18-2 run). Method: for each outer fold f, the held-out devices get the stored
correction-arm row if S3_f < 0 (strict), else the C0_refit row; S3_f = mean over the fold's training records of
ln(RRMS(start x exp(g_{-k})) / RRMS(start)), g_{-k} trained with the gated arm's recipe (seed 0, 500 steps) on the
training records outside the record's inner group (device; geometry group for TU Delft), applied without refit (I7-1
definition). S3_f < 0 = predicted gain. Recipe matched to the arm: g v2 datasets (sky130 77 K all-18 with the I15-2
selected rows, IHP 300 K with the I17-2 selected rows, TU Delft 4.13 / 295.0 K with the I18-2 selected rows) use
S3_v2 (`fit_v2`, sym_v2, EPS 0.01 V, dataset supply, lambda = lambda*_f of the outer fold, penalty grids of the
inner-training devices), recomputed; M_bias v1 datasets (NIST 4 K I8-1, NIST 4 K room-temperature card I9-1, IHP
233 / 343 / 398 K) use the stored v1-recipe S3_f. Bootstrap unit: device; 7 geometry groups for TU Delft. Rule: C1
CI lower of GMR(gated, C0_refit) <= 1 on all 9 datasets, C2 CI upper < 1 on sky130; supported iff both. Validity:
V1 row counts (18 / 53 / 87 / 18 x 4 / 45 x 2) and selected rows equal the stored ones; V2 the I18-1 S3 code with
the v1 recipe reproduces the stored I7-1 S3_f on 2 IHP 300 K folds; V3 S3_v2 finite on every fold; V4 unit tests.
Predicted decisions: sky130 18/18, NIST 53/53 and 87/87 (determined from stored S3_f), IHP 300 K 0/18, IHP 233 /
343 / 398 K 0/18 (determined), TU Delft 5/7 and 3/7 groups; predicted outcome supported. The v1-arm datasets are
retrospective (S3_f, decisions and rows stored; I8-2 fold gate and I10-2 S_gate values known); TU Delft g v2 is
prospective. Code 8fc4f32 (`src/cryoml/explore/i18_1_gate.py`, `scripts/research/explore/i18_1_gate.py`, 9 unit
tests); smoke (2 folds, 50 steps) ran.
Queue I18_1_s3 (2035 s, commit after the run): V2 difference 0.0 on both folds; sky130 S3_v2 < 0 on 18/18 folds
(range -0.214 .. -0.100; 306 inner fits 1509 s); sky130 S3_v1 < 0 on 18/18 (-0.190 .. -0.147); IHP 300 K S3_v2 < 0
on 0/18 (0.184 .. 0.642). The TU Delft S3_v2 (I18_1_s3_tud) and I18_1_decide run after the I18-2 jobs.

## I18-2 pre-registration in force (2026-10-22 16:00): selected g v2 recipe on measured TU Delft 40 nm, Part B with ngspice-41 OSDI

Pre-registration `out/research/explore/I18_2/prereg.md` / `prereg.json`, commit 2454b15 (alone). Data I6-1 (45 DEV
devices, 7 geometry groups, TEST 24 locked), 4.13 K and 295.0 K, BSIM4 CMOSedu card, ngspice-41. BSIM4 clamp state C
(change from I6-1): the I6-1 starting cards have a2 > 1 on 25/45 (4.13 K) and 20/45 (295.0 K); every deck writes
the b4check.c clamp. Arms per temperature, leave-one-group-out: C0_refit, M_bias v1, sym_e1_l{0,1,2,5,10} (VDD
1.1 V), selected (inner 3-fold selection over the training groups). Penalty grid 45 x 45 to 1.1 V (25 mV step).
Primary GMR(selected, C0_refit) with the 7-group cluster bootstrap (gain replicated / worse / inconclusive);
secondary GMR(selected, v1) margin 1.05. Part B per temperature on the full-data model at lambda*_full (selection
over all 7 groups) with M2 cards: TU Delft channel replica `i6card_ch` (charge-free; alpha0 = alpha1 = beta0 = 0,
agidl = agisl = 0, junction saturation currents 0, igcmod = igbmod = 0), VCVS copies, -gmin VCCS, OSDI module;
serpentine Q1 grids V(g,s) 0 .. 1.1 V x V(d,s) -1.1 .. 1.1 V; Q2 op points scaled to 1.1 V; I17-1 rules; Q3 not run
(sky130 circuits at 1.8 V, card 1 V). Predictions: ln GMR(selected, C0_refit) < 0 at both temperatures, ln
GMR(selected, v1) > 0 at both, primary inconclusive at 4.13 K and gain replicated at 295.0 K, Part B deployable
with restrictions (Q3 not run) at both.
Amendment A1 (commit 1795e10, alone, before any full run): the Part B V(d,s) = 0 check is evaluated on the
element-added current |I_elem - I_card| (tight forward grid, 1e-12 A). Reason: the card has igcmod = igbmod = 1;
in the smoke run the card alone gave 5.96e-11 A at V(d,s) = 0 (gate tunneling) and the element added 1.8e-20 A.
Code 254e77d (`src/cryoml/explore/i18_2_tud.py`, `i18_2_partb.py`, drivers `scripts/research/explore/i18_2_tud.py`,
`i18_2_partb.py`, 11 unit tests). Smoke (4 groups x 1 device, 200 steps, Part B on 2 devices; not used for
decisions) ran every stage; it found the serpentine default-tolerance grid file appended over the runtime repeats
(fixed before the code commit). Queue: I18_2_start (V1a max |diff| 0 on 90 cards, V1b identical on 45 unclamped
cards; state-C starting RRMS 4.13 K 0.0628 -> 0.0722, 295.0 K 0.0271 -> 0.0463), I18_2_pens (90 grids, V4 ok),
I18_2_base_4 (V1c |diff| 0.0 on 4 rows), then base_295, lodo x 10, inner x 2, full x 2, decide, I18_1_s3_tud,
I18_1_decide, partB x 2, decideB.

## I18-2 outcome (2026-10-22 23:15): TU Delft 40 nm, selected g v2 inconclusive at both temperatures; Part B not deployable at 4.13 K, deployable with restrictions at 295.0 K (Q3 not run)

Queue: I18_2_start 6 s, pens 2 s, base_4 653 s, lodo_4 l0 572 s, l1 / l2 / l5 / l10 1054-1082 s, inner_4 4293 s (105
trainings 1707 s, 1350 inner refits 2586 s), full_4 about 16 min, base_295, lodo_295 x 5, inner_295 (training 2093
s), full_295 1115 s, decide 21 s; partB_4 158 s, partB_295 159 s. Outputs `out/research/explore/I18_2/{start, pens,
base_*, lodo_*, inner_*, full_*, partB_*, decision.json, partB_decision.json}`.
Validity: V1a max |diff| 0 (90 cards), V1b state C = state L on the 45 unclamped cards, V1c 0.0 on 4 rows (state-L
refits of C0_refit and the stored I6-1 v1 fold correction), unit tests pass, 45 rows per arm and temperature, inner
and full selection complete, 90 grids finite; VB 0 failed M2 refits, 45 Part B rows: valid. Clamp state C: 25/45
(4.13 K) and 20/45 (295.0 K) starting cards clamped; starting RRMS state L -> C 0.0628 -> 0.0722 and 0.0271 ->
0.0463; after the clamp no b4check message remains.
All-device / combined / median (state C): 4.13 K C0_refit 0.0602 / 0.0532 / 0.0441, M_bias v1 0.0534 / 0.0491 /
0.0396, sym_e1_l0 0.0509, l1 0.0507, l2 0.0476, l5 0.0492, l10 0.0483, selected 0.0484 / 0.0449 / 0.0374; 295.0 K
C0_refit 0.0265 / 0.0251 / 0.0256, v1 0.0208 / 0.0210 / 0.0184, l0 0.0211, l1 0.0220, l2 0.0219, l5 0.0212, l10
0.0218, selected 0.0215 / 0.0218 / 0.0188. I6-1 stored (state L): 4.13 K C0_refit 0.0575, M_bias 0.0533; 295.0 K
0.0255, 0.0196.
Primary (7-group cluster bootstrap): 4.13 K GMR(selected, C0_refit) **0.826 [0.669, 1.014]**, 30/45 lower:
**inconclusive** (device CI [0.756, 0.905]); 295.0 K **0.829 [0.523, 1.247]**, 29/45: **inconclusive** (device CI
[0.691, 0.999]). Secondary GMR(selected, v1): 4.13 K 0.908 [0.848, 0.965] non-inferior (CI upper below 1); 295.0 K
1.022 [0.975, 1.056] not shown non-inferior. GMR(v1, C0_refit) 0.909 [0.766, 1.077] and 0.811 [0.532, 1.190].
lambda*: 4.13 K 10 V on 6 groups, 5 V on 1, lambda*_full 10 V; 295.0 K 5 V x3, 1 V x2, 2 V x1, 10 V x1, lambda*_full
5 V. Contrast K (selected / C0_refit, 4.13 K over 295.0 K) 0.997. Predicted signs 5 of 6 correct (selected vs v1 at
4.13 K predicted > 0, observed 0.908).
Part B (ngspice-41 OSDI, M2 cards, 45 devices per temperature; amendment A1 V(d,s) = 0 check on the element-added
current): 4.13 K **not deployable (Q3 not run)**: the only core failure is 'Q1 convergence / non-finite', 3 lines
(dynamic gmin, true gmin and source stepping failed) in the dense_elem deck of t40_nmos_W1_L0p04_d192; the
card-only dense deck of the same M2 card prints the same 3 lines (rule counts element decks; outcome stays not
deployable). Other core values: tight equivalence 0 violations, element-added |I(Vds = 0)| 9.7e-19 A (raw element
and card both 9.58e-9 A, gate tunneling), g text 4.4e-16, exchanged 1.7e-15, asymmetry 3.8e-10, Q2 1.0e-7,
capacitance 0, gm 0; restrictions default tolerance 1.64e-3; runtime 1.95. 295.0 K **deployable with restrictions
(Q3 not run)**: every core check passes; restrictions default tolerance 1.46e-3 and 1 gds curve
(t40_nmos_W1_L1_d134, |V(g,s)| 0.725 V); runtime 1.93. Replica non-channel outputs 0 at the op points. Element-deck
RRMS vs Python rows: M2 0.04108 vs 0.04108 (4.13 K), 0.01662 vs 0.01662 (295.0 K); selected held-out pairs 0.04836
vs 0.04835 and 0.02150 vs 0.02149 (max per-device |d| 2.5e-4). Part B predictions 1 of 2 correct.
Report batch `out/research/report/2026-10-22_i18_2/` (`scripts/research/plot_explore_i18_2.py`, 17 PNGs; the I-V
figures re-simulate the stock, C0_refit, v1 and selected cards in state C and reproduce the stored rows); site claim
`i18-2` (refuted: accuracy inconclusive, Part B not deployable at 4.13 K).

## I18-1 outcome (2026-10-22 23:15): gated correction, supported by the pre-registered rule; on TU Delft the gate chose against the outcome

Queue: I18_1_s3 2035 s, I18_1_s3_tud 536 s (84 inner fits); I18_1_decide failed once (KeyError: the 4 NIST 4 K
records excluded by I8-1 amendment A1 have null rows and no S3); fix: devices with non-finite rows in both arms and
no fold S3 are left out (53 paired, as registered); decide rerun by hand. Valid (V1 row counts 18 / 53 / 87 / 18 x 4
/ 45 x 2 and selected rows equal; V2 difference 0 on 2 folds; V3 S3_v2 finite on every fold; V4 tests pass). TU
Delft rows from I18-2 (no fallback).
S3_v2 per fold: sky130 < 0 on 18/18 (-0.214 .. -0.100), IHP 300 K 0/18 (0.184 .. 0.642), TU Delft 4.13 K 2/7 (-0.039
.. 0.091; applied on nMOS W1/L1 and pMOS W1/L1), 295.0 K 1/7 (-0.106 .. 0.471; applied on pMOS W1/L1).
GMR(gated, C0_refit) [95% CI], folds applied: sky130 0.743 [0.654, 0.855] 18/18; NIST 4 K 0.806 [0.758, 0.864]
53/53; NIST RT card 0.797 [0.748, 0.859] 87/87; IHP 300 / 233 / 343 / 398 K 1.000 [1.000, 1.000] 0/18 each; TU
Delft 4.13 K 0.981 [0.863, 1.097] 2/7; 295.0 K 1.099 [1.000, 1.327] 1/7. Rule: C1 holds (every CI lower <= 1), C2
holds: **supported**. The TU Delft 295.0 K CI lower bound equals 1.000 (resamples without the applied group give 1);
the device-bootstrap CI is [1.029, 1.190]. Correction always vs C0_refit: TU Delft 0.826 / 0.829; gated vs
correction always 1.188 / 1.326. Within-dataset Spearman(S3_f, y_f): TU Delft -0.21 / -0.68, sky130 -0.37, NIST
-0.38 / -0.60: the fold-level gate keeps C0_refit on the TU Delft groups with the largest held-out gains (295.0 K
nMOS W0.12 groups, y_f -0.58 .. -1.08, S3_f 0.34 .. 0.47) and applies the correction to the group where it raised
the error (pMOS W1/L1, y_f +0.53). Pooled exp(mean ln GMR) over 9 datasets: gated 0.929, correction always 0.958.
Secondary v1 gate: sky130 0.745, NIST 0.806 / 0.797, IHP 1.000 x 4, TU Delft (I6-1 state L) 1.027 / 1.040.
Predictions: decisions correct on 7/9 datasets (TU Delft predicted 5/7 and 3/7 groups, observed 2/7 and 1/7);
GMR signs: TU Delft 4.13 K < 1 correct, 295.0 K predicted < 1, observed 1.099; outcome predicted supported.
Report batch `out/research/report/2026-10-22_i18_1/` (`scripts/research/plot_explore_i18_1.py`, 11 PNGs; every
gated row and GMR recomputed from the stored rows and S3 values; I-V figures of TU Delft devices in a gated-on and
a gated-off group per temperature); site claim `i18-1` (verified, with the 295.0 K bound and the within-dataset sign
reversal as caveats). Site rebuilt, not published.

## I19-1 pre-registration in force (2026-10-23 02:30): dataset-level gate, one decision per dataset from S3_D on all DEV devices

Iteration 19, default option 1 (no user reply). Pre-registration `out/research/explore/I19_1/prereg.md` / `prereg.json`,
commit 45b4f37 (alone, before any I19-1 code or S3_D value). Method: S3_D = mean over all DEV records r of the dataset
of ln(RRMS(r, I_start x exp(g_{-k(r)})) / RRMS(r, I_start)), g_{-k(r)} trained with the gated arm's recipe (I18-1 arms;
seed 0, 500 steps) on the training set of the LOGO fold that holds out the group k(r) (IHP: same class), lambda =
lambda*_k of that fold for g v2; the correction is applied to every device of the dataset iff S3_D < 0, else
C0_refit. Threshold 0 fixed before any S3_D: same threshold as I8-2 G and I18-1; the I7-1 OLS zero crossing
(+0.02455) was fitted to the fold-mean S3 with the v1 recipe and is reported as a sensitivity column only.
Part R (retrospective; DEV outcomes and I8-2 G values known): R1 GMR(gated_D, C0_refit) <= 1 on each of the 9 I18-1
datasets; R2 pooled exp(mean ln GMR) below correction always (0.958). Part P: TEST gate decisions for the 8 locked TEST
splits with a DEV counterpart (NIST 4 K lvt 38, NIST RT card 58, IHP 10 x 4 temperatures, TU Delft 24 x 2), committed
alone before the decide step; TEST not evaluated. Part C: sign accuracy on the 20 I7-1 simulated datasets (v1) at
least 19/20 (constant apply gets 18/20). Outcome supported iff R and C. Validity V1 rows equal I18-1, V2 S3 code
reproduces 3 stored fold values, V3 complete, V4 unit tests, V5 TEST closed (every loaded tag in the DEV list).
Code 4da313c (`src/cryoml/explore/i19_1_dataset_gate.py`, `scripts/research/explore/i19_1_dataset_gate.py`, 12 unit
tests); smoke 0f7734a (2 fits per part, 50 steps).

## I19-1 outcome (2026-10-23 03:00): dataset-level gate supported by the pre-registered rule; on TU Delft the g v2 gate keeps C0_refit where the correction lowers the error

Queue: I19_1_s3_real 556 s (327 inner fits, incl. V2 checks), I19_1_testgate 0 s, I19_1_s3_sim 289 s (320 fits),
I19_1_decide, I19_1_figs. Valid: V1 rows 18 / 53 / 87 / 18 x 4 / 45 x 2 equal to I18-1, V2 difference 0.0 on 3 folds
(IHP 300 K v1 x 2, sky130 g v2 x 1), V3 complete, V4 12 tests pass, V5 all loaded tags in DEV lists.
Part P (commit a755e18, before decide; TEST not opened): NIST 4 K lvt TEST apply (S3_D -0.103), NIST RT card TEST
apply (-0.069), IHP TEST 233 / 300 (g v2) / 343 / 398 K C0_refit (+0.187 / +0.308 / +0.367 / +0.367), TU Delft TEST
4.13 / 295.0 K (g v2) C0_refit (+0.008 / +0.264); v1 secondary: IHP 300 K C0_refit (+0.311), TU Delft apply (-0.029 /
-0.051). Disagreements with the I8-2 committed TEST predictions (M_bias v1): TU Delft primary decisions at both
temperatures (I8-2 S3 and S1 rule predict gain); none for the v1 secondary. At +0.02455 the TU Delft 4.13 K primary
decision is apply.
Part R (retrospective): S3_D sky130 -0.133, NIST -0.103 / -0.069, IHP 300 K +0.308 (hv +0.225, lv +0.374), 233 /
343 / 398 K +0.187 / +0.367 / +0.367, TU Delft g v2 +0.008 / +0.264. GMR(gated_D, C0_refit): sky130 0.743 [0.654,
0.855], NIST 0.806 / 0.797, IHP 1.000 x 4, TU Delft 1.000 / 1.000. R1 holds (9/9 <= 1), R2 holds: pooled gated_D 0.921
vs correction always 0.958 (per-fold gate 0.929, I8-2 G 0.921, post hoc oracle 0.883; without TU Delft 0.900 /
0.999 / 0.900). Part R supported (retrospective). The decisions equal the I8-2 gate G on 9/9. Decision = realised
sign of ln GMR(correction, C0_refit) on 7/9 (TU Delft g v2: correction always 0.826 / 0.829, gate keeps C0_refit).
v1 secondary: S3_D sky130 -0.178, IHP 300 K +0.311, TU Delft (I6-1 state-L records) -0.029 / -0.051; pooled gated_D
0.892 vs v1 always 0.965; decision = realised sign 9/9 (TU Delft applied: 0.939 / 0.797).
Part C: 19/20 correct (wrong GF180 r0 lambda 0: S3_D +0.0003, y -0.045): holds. Spearman(S3_D, y_D) +0.985; OLS y =
-0.021 + 1.217 S3_D (R^2 0.993, zero crossing +0.0176); S3_D vs I7-1 fold-mean S3 Spearman 1.000; pooled gated_D
0.590, M_bias always 0.591, oracle 0.588; at +0.02455: 20/20.
Outcome: **supported** (Part R retrospective, Part C). With the g v2 recipe on TU Delft, S3_D is positive at both
temperatures while the correction lowers the held-out error; the dataset-level gate avoids the per-fold losses of
I18-1 (0.981 / 1.099 -> 1.000 / 1.000) but does not recover the gain. The v1 recipe gives the realised sign on all 9.
Report batch `out/research/report/2026-10-23_i19_1/` (`scripts/research/plot_explore_i19_1.py`; 6 tables, forest
plot, S3_D vs realised gain scatter; 94 values recomputed); site claim `i19-1` (verified, caveats: retrospective
Part R, DEV leakage of starting records, TU Delft gain missed). Site rebuilt, not published.

## Method-freeze draft and TEST plan (2026-10-23): proposed, not frozen

Default option run (no user reply); documents only, no TEST split opened, no NGSpice run, no training. Draft
`docs/research/method_freeze_draft_2026-10-23.md` (status "proposed, not frozen"; final freeze about 2026-12-10):
(1) method specification with code and commit per part (direct43_r3_fd starting cards, BSIM4 state C, selected g v2
recipe with nested lambda selection, held-out TRF refit, I19-1 dataset-level S3 gate, channel-replica OSDI element,
deployment checks with I17-1 and I18-2 A1 rules); (2) DEV evidence table for the 9 measured DEV datasets; (3) TEST
plan for the 8 locked splits (NIST lvt 38, NIST RT card 58, IHP 10 DUTs x 4 temperatures, TU Delft 24 x 2) with the
committed I19-1 decisions (a755e18) and I8-2 predictions (08d23e3) to be scored, proposed rules, order of operations
and a compute estimate of about 2.2-2.8 h; (4) open items. New finding from a file check: the NIST decks
(`i8_1_nist`, `i9_1_nist_rt`) run all sweeps in one ngspice process without the A2 clamp, and the stored base a2 > 1 on
16/57 (I8-1) and 34/87 (I9-1) starting cards; the NIST rows were not audited for the clamp state (open item 4.1).
Pooled exp(mean ln GMR vs C0_refit) over the 9 DEV datasets: gated 0.921, correction always 0.958, M_bias v1 always
0.963 (TU Delft v1 rows in state C). Table batch `out/research/report/2026-10-23_freeze_draft/`
(`scripts/research/plot_freeze_draft.py`; 88 values checked against the stored decision files).

## I20-1 pre-registration in force (2026-10-23 06:55): NIST deck A2 clamp audit and fix; selected g v2 recipe on both NIST DEV datasets

Pre-registration `out/research/explore/I20_1/prereg.md` / `prereg.json`, commit a516fe3 (alone, before any I20-1 RRMS
value). Part A: every stored NIST DEV arm (I8-1: stock, start, C0_refit, T1_inloop, M_bias v1, B1, B2, B3_d3, the 10
I9-2 class arms and mlp16_rep; I9-1: the same without I9-2 plus M_bias_lvt) rescored in states L / C / U with
validity V1 (L reproduces the stored rows), V2 (C = L = U on unclamped cards), V3 (computed A2_eff = b4check.c value;
state-C curve 1 = state-L curve 1). Effective A2: 77 K bins have no size terms; room-temperature pMOS bins carry
la2 / wa2 / pa2 (b4temp.c binning, binunit 2). Part B: both NIST simulators default to state C. Part C: state-C
starting records, C0_refit and M_bias v1 LODO, S3_D v1. Part D: selected g v2 recipe (VDD 1.8 V, 73 x 73 grid at
25 mV), lambda selection with 6 blocks instead of per outer fold (the I15-2 per-fold selection needs 795 / 1,305
trainings and 13,780 / 37,410 refits), 5-seed LODO at lambda*_h, S3_D g v2. Part E: primary GMR(selected g v2,
C0_refit) per dataset (I17-2 label), secondary vs M_bias v1 margin 1.05. Part F: a dated amendment of the I19-1 TEST
gate decisions (a755e18) iff any of the four NIST decisions (S3_D v1 state C, S3_D g v2) differs from the committed
one; frozen-method S3_D recipe fixed before the NIST values: the v1 recipe on state-C records decides whether the
selected g v2 correction is applied (retrospective sign agreement with the realised g v2 gain 4/4 vs 2/4 for the
g v2-recipe S3_D, known when the rule was chosen). Code 572e39e (driver `scripts/research/explore/i20_1_nist_clamp.py`,
17 unit tests in `tests/test_i20_1_nist_clamp.py`); smoke on 6 devices per dataset. Amendment A1 (1f14eda, after the
audit, before any rerun result): V2 is read on unclamped cards without a fatal model check (7 cards of I8-1 A1 devices
with "Pclm is not positive" have U != L because the warm-up op takes the fatal check; C = L on all 1238).
