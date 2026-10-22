# I17-2 pre-registration (IN FORCE): second-PDK replication of the selected g v2 recipe on measured IHP SG13G2 MOSFETs (PSP103), 300 K, 18 DEV DUTs, leave-one-DUT-out; accuracy only, Part B not run

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-22), committed alone, before any I17-2 code or run.

## Dataset choice

Candidates with measured non-sky130 DEV data in this repository (`real_split.json` covers sky130 only; the other
splits are `I5_1/split.json` and `I6_1/split.json`):

| dataset | measured | compact model / simulator | DEV / TEST | usable for |
|---|---|---|---|---|
| IHP SG13G2 (I5-1), IHP-Open-PDK `libs.doc/meas/MOS` | yes, 233 / 300 / 343 / 398 K | PSP103 via OSDI (`psp103.osdi`, OSDI 0.4), Homebrew ngspice-46 | 18 / 10 DUTs | accuracy part; Part B needs a PSP replica (below) |
| TU Delft 40 nm (I6-1) | yes, 4.13 / 295 K | BSIM4 (CMOSedu 50-nm example card, not a foundry PDK), ngspice-41 | 45 / 24 devices | accuracy and Part B with the BSIM4 replica |
| GF180MCU (pdk2) | no (simulated targets only) | BSIM4, ngspice-41 | - | not usable (no measured data) |

Chosen: **IHP SG13G2, 300 K** (I5-1 primary temperature). Reasons: it is the only measured dataset with a foundry
PDK model card other than sky130 (TU Delft uses a generic example card, so it is not a second PDK); the task
preference is IHP or GF180 if measured, and GF180 has no measured data. Cost of the choice: the model is PSP103, so
the deployable element (BSIM4 channel replica) does not transfer, and the simulator is ngspice-46 (I5-1 setup).

## Data, cards, simulator (I5-1, unchanged)

- 18 DEV DUTs (`I5_1/split.json`: LV 10 = 5 geometries x 2 polarities; HV 8), 300 K (header TEMP 27 C); TEST 10
  DUTs stay locked (not loaded). 11 Vb = 0 curves per DUT; I5-1 frozen measured inclusion (143 of 198 at 300 K).
- Card: the PDK `.model` block as a top-level PSP103 card; fitted set: the 12-parameter pdk2 IHP wide set, I5-1 box.
  Starting cards: `I5_1/start/records_300K` (stock polish, max_nfev 3000, irls 3).
- Simulator: Homebrew ngspice-46 + `psp103.osdi` (I5-1; matches the PDK subcircuit within 4.4e-8 relative). Not
  ngspice-41: the ngspice-41 OSDI build loads OSDI 0.3 only and the PDK `psp103.osdi` is OSDI 0.4; a recompile with
  OpenVAF 23.5 has not been validated. Check before registration (not an outcome): re-simulation of the stored
  starting-card curves of 3 DEV DUTs (lv_nmos_W0p15_L0p18, hv_pmos_W10_L0p35, lv_pmos_W10_L10) gives max relative
  difference 0.
- BSIM4 clamp (`bsim4_state` clamped): not applicable; the IHP decks contain no BSIM4 instance (stated, not run).
- Metric: strict paper RRMS on the included curves (I5-1 `score`), NaN if an included curve is non-finite.

## Arms (leave-one-DUT-out within class, as I5-1)

Held-out DUT h; training = the other DEV DUTs of the same class (LV 9, HV 7), both polarities, in record load
order. Held-out refit for every arm: polish from the starting card with g frozen in the loop (max_nfev 400, irls 2,
diff_step 0.01), I5-1 `refit_task` kind 'corr'.

| arm | definition |
|---|---|
| C0_refit | g = 0 |
| M_bias v1 | I3-2 recipe (`i3_2_bias.fit_bias`, inputs is_pmos, \|V(g,s)\| / VDD, \|V(d,s)\| / VDD with VDD = class supply 1.2 / 3.3 V; hidden 16, 2 layers, l2 1e-4, lr 3e-3, 2000 steps, seeds 0-4) |
| sym_e1_l{lambda}, lambda in {0, 1, 2, 5, 10} V | g v2 recipe (`i12_1_symm.fit_v2`): symmetric inputs x1 = (s V(g,s) + s V(g,d) + m) / (2 VDD), x2 = m / VDD, m = sqrt(V(d,s)^2 + EPS^2), EPS = 0.01 V, VDD = class supply; same hyperparameters and seeds; loss + lambda P |
| selected | per outer fold the row of sym_e1_l{lambda*_h} |

Penalty P (I12-1 definition): mean over training DUTs of the mean over penalty points of relu(-(gm/I + dg/dV(g,s)))
+ relu(-(gds/I + dg/dV(d,s))). Penalty grid per DUT: the starting card simulated (ngspice-46, Vb = 0, 300 K) on the
forward quadrant u_g = s V(g,s), u_d = s V(d,s) in {0, VDD/74, ..., VDD} (75 x 75, the sky130 point count; step
16.2 mV LV, 44.6 mV HV); points with u_d > 0 and forward current > 1e-9 A; gm/I and gds/I by central differences.
Code change: `fit_v2` and `penalty_arrays` gain optional `vdd` / grid-step arguments whose defaults keep the sky130
behaviour (unit tests).

Nested selection of lambda (I15-2 rule): per outer fold, the training DUTs sorted by (polarity, nMOS first; tag) go
to inner fold i mod 3; for each inner fold and lambda one member (seed 0) is trained on the other training DUTs and
the inner held-out DUTs are refit with it; lambda* = argmin over lambda of the mean ln RRMS of the inner held-out
refits (non-finite counts as ln 1 = 0); ties within 1e-12 to the larger lambda. No full-data model (Part B not run).

## Rules

Primary: GMR(selected, C0_refit) over the 18 DUTs, device bootstrap 10,000, 95% CI (`i3_2_bias.paired_ratio`).

| condition | primary label |
|---|---|
| CI upper < 1 and 0 failed DUTs in selected and C0_refit | gain over C0_refit replicated |
| CI lower > 1 | worse than C0_refit |
| otherwise | inconclusive |

Secondary: GMR(selected, M_bias v1), margin 1.05: CI upper <= 1.05 -> non-inferior to M_bias v1; otherwise not
shown non-inferior.

Reported without a rule: every arm's all-device / combined / median; GMR of each arm vs C0_refit and vs v1;
geometry-cluster bootstrap CIs (cluster = class and W/L; LV nMOS and pMOS of one W/L share a cluster: 9 clusters of
DEV, of which LV 5 (2 DUTs each), HV 7 (HV W0p3_L10 nMOS and pMOS share a cluster)); LV / HV split; lambda* counts;
negative corrected gm / gds fractions on the held-out and training penalty grids (I15-2 definition); I5-1 stored
rows for comparison.

Predictions (signs, stated before any run; not rules):

| quantity | predicted sign of ln GMR | basis |
|---|---|---|
| GMR(selected, C0_refit) | > 0 (selected above C0_refit) | I5-1 GMR(M_bias v1, C0_refit) 1.214 [1.008, 1.537] at 300 K; I8-2 committed prediction 'IHP no gain' |
| GMR(selected, M_bias v1) | < 0 (selected below v1) | the penalty and the inner selection limit g where v1 overfits the training DUTs |
| GMR(M_bias v1 rerun, C0_refit rerun) | > 0 | I5-1 |

## Part B: not run

The I17-1 Part B (rules, thresholds and procedure) cannot run on this dataset without changing the element:

1. Replica. The BSIM4 channel replica (`i14_1_chan.REPLICA_CHAN`) does not apply to PSP103. A PSP103 replica needs
   SWIGATE = 0, SWIMPACT = 0, SWGIDL = 0 (gate, impact-ionization and GIDL currents off), SWJUNCAP = 0 (junction
   currents and charges off), zero overlap / fringe charge parameters (CGOV, CGBOV, CFR and their geometry-scaling
   forms), and a VCCS cancelling the two gmin conductances PSP103 adds from DI and SI to BP
   (`PSP103_module.include` lines 1726-1727).
2. Intrinsic charge. PSP103 contributes the intrinsic charges Qg, Qb, Qd unconditionally (`PSP103_module.include`
   lines 1788-1790, no switch); no parameter removes them. The sensed replica current would contain displacement
   current in AC and transient analyses, and I(d, s) <+ (exp(g) - 1) I(sense) would scale it, so the registered Q2
   capacitance check (Im Y_elem = Im Y_card within 1e-3) and the transient circuits would fail by construction. A
   fix needs an edited PSP103 Verilog-A (a replica module without ddt contributions, or exp(g) applied to the
   channel current inside the model): a different element from the registered one.
3. Simulator. The ngspice-41 OSDI build loads OSDI 0.3 only; `psp103.osdi` is OSDI 0.4.
4. Q3 circuits are sky130 netlists; IHP circuits are not defined.

Part B is recorded as **not run**; no deployment label is given for this PDK.

## Validity (checked first; any fail -> invalid)

| check | pass |
|---|---|
| V1 C0_refit and M_bias v1 rerun rows reproduce `I5_1/lodo_300K/results.json` per_device C0_refit and M_bias | max \|RRMS difference\| <= 1e-9 on 18 DUTs each |
| V2 unit tests (`tests.test_i17_2_ihp`, `tests.test_i12_1_symm`, `tests.test_i15_2_ablate`) | all pass |
| V3 rows | 18 rows in every arm; inner selection complete (18 outer folds x 3 inner folds x 5 lambdas trained; every inner held-out DUT scored) |
| V4 penalty grids | 18 grids of 75 x 75 finite currents; forward current sign positive at u_g = u_d = VDD |

## Decision

| condition | outcome |
|---|---|
| any validity check fails | invalid (stop: fix and re-register) |
| valid | primary label (gain over C0_refit replicated / worse than C0_refit / inconclusive) + secondary label (non-inferior / not shown non-inferior to M_bias v1); Part B not run |

## Order and compute

This prereg commit (alone) -> code + unit tests commit -> smoke (2 outer folds lv_nmos_W0p15_L0p18 and
hv_nmos_W0p3_L0p5, 200 steps, 1 seed, refit max_nfev 20; not used for decisions) -> queue, one heavy job at a time:
I17_2_pens (penalty grids, about 1 min), I17_2_base (C0_refit + M_bias v1 + V1, about 10 min), I17_2_lodo_l{0,1,2,5,10}
(about 10-30 min each), I17_2_inner (about 1 h), I17_2_decide.

## Outputs

`out/research/explore/I17_2/{pens, base, lodo_sym_e1_l*, inner, decision.json}`; arrays (`*.npz`) and the
measured dataset not in git; report batch `out/research/report/<date>_i17_2/` (tables vs rules, per-DUT plots, I-V
figures with measured points, stock-card NGSpice and re-simulated C0_refit / v1 / selected cards), site claim
`i17-2`, RESEARCH_LOOP entry.
