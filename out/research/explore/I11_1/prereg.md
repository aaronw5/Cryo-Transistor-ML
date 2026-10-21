# I11-1 pre-registration (IN FORCE): 43-parameter card + shared bias-only correction as a compact-model add-on in ngspice-41 + OSDI, sky130 77 K

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds
this file (2026-10-21). Done before it: code draft (`src/cryoml/explore/i11_1_deploy.py`,
`scripts/research/explore/i11_1_deploy.py`, not committed), OpenVAF 23.5.0 compile of the stored full-data model
and a 2-device smoke (`runs/smoke/`, nmos_L1_W1p6 and pmos_L2_W5, refit max_nfev 20; not used for any decision).
Started by the user's default choice (no reply by 17:00). All 18 sky130 77 K devices are open (user decision
2026-10-21). There is no measured circuit data: Q3 and Q4 are simulator-level consistency, convergence and
runtime checks, not accuracy claims.

## Questions

- Q1 device-level equivalence over the bias plane: ngspice-41 OSDI build with the compiled correction element vs
  the Python model I_card x exp(g) on a dense grid beyond the measured curves; convergence; I_D at Vds = 0;
  g outside the measured bias points; behaviour for negative Vds and for exchanged drain/source terminals.
- Q2 small signal: gm, gds and the imaginary (capacitive) admittances from ngspice AC at op points vs the Python
  model; finite differences on 1 mV sweeps; gm / gds sign changes and kinks; whether capacitances change.
- Q3 circuits at 77 K (inverter VTC, 5-stage ring oscillator, 1:1 current mirror): convergence and the
  differences between published card, C0_refit card, M_full card alone and M_full card + g.
- Q4 ngspice wall time with and without the element.

## Configuration

| item | value |
|---|---|
| correction | the stored full-data M_bias model trained on all 18 starting-card records (`I3_2/all18/corrections.json` `full\|M_bias`, I3-2 recipe); Verilog-A `I3_2/all18/m_bias_sky130_all18_full.va` unchanged (sense line `V(sense) <+ RS * I(sense)`, RS = 0, parameter `is_pmos`, VDD = 1.85 normalization) |
| compiler / simulator | OpenVAF 23.5.0 (`~/cryo-ng41/openvaf/bin/openvaf --target aarch64-apple-darwin`), ngspice-41 OSDI build `~/cryo-ng41/arm64-osdi/bin/ngspice` (bit-identical to the canonical binary on the published card, I3-2 F1c); card-only runs use the same binary without `pre_osdi` |
| element | sky130 lvt BSIM4 subcircuit with its drain at an internal node di; module instance `N d g s di`; total drain current I(sense) x exp(g), g = g(is_pmos, \|V(g,s)\|/1.85, \|V(d,s)\|/1.85) |
| card (decision Q1 of the task) | **M_full card**: per device, the 43-parameter card refit from the direct43_r3_fd starting card with the full-data g frozen in the loop (I3-2 refit recipe: max_nfev 400, irls 2, diff_step 0.01). Reason: a deployed add-on is one correction module plus cards fitted with that module; the stored M_bias LODO card of a device was refit with its fold g (trained without the device), so pairing it with the full-data g combines two different corrections. The M_full RRMS on the measured curves is in-sample (the correction saw the device) and is reported as such, next to the LODO value 0.0732 |
| instance geometry | `spice_pdk._instance_line` of each measured device (ad/as/pd/ps/nrd/nrs, per-device mult); 77 K (temp -196.15 C) |
| tolerances | default (reltol 1e-3, abstol 1e-12, vntol 1e-6); r4 (reltol 1e-4, descriptive); tight (reltol 1e-9, abstol 1e-18, vntol 1e-12) |

## Q1 design

- Grid: Vgs 0 to 1.85 V in 25 mV steps (75 values; pMOS 0 to -1.85 V) x Vds -1.85 to +1.85 V in 25 mV steps (149
  values), nested DC sweep (Vds inner), source and body at 0. All 18 devices. Runs: card only and card + element
  at default, r4 and tight tolerance.
- Python reference: the card-only current of the same tolerance x exp(g) at the instance terminal voltages.
- Statistic: max |I_elem - I_py| / |I_py| over grid points with |I_py| > 1 nA (also split into forward /
  reverse Vds and |I| >= 1 uA).
- Terminal exchange: the same grid with the instance's drain and source exchanged (BSIM drain chain and module d at
  ground, module s at the swept node); asymmetry statistic max |ln(I_swap/I_fwd)_elem - ln(I_swap/I_fwd)_card| in
  forward operation, |I| > 1 nA (the card is symmetric by construction; the element evaluates g with the
  instance source terminal, so the expected value is |g(Vgs - Vds, Vds) - g(Vgs, Vds)|).
- I_D at Vds = 0: max |I_elem(Vds = 0)| over Vgs (default and tight).
- g outside the measured bias points: range of g on the grid (including Vds < 0, where abs() folds Vds onto
  forward bias) vs the range at the measured points; g on the Vds = 0 line.

## Q2 design

- Op points per device (signed for pMOS): gm sets |Vds| in {0.05, 0.925, 1.85} V x |Vgs| on the 25 mV grid; gds
  sets |Vgs| in {0.6, 0.9, 1.2, 1.5, 1.85} V x |Vds| on the 25 mV grid (600 points). At each: `op`, then AC at
  1 MHz with unit excitation on the gate (gm, Im Y_dg) or on the drain (gds, Im Y_dd); gate current Im Y_gg.
  Tight tolerance, card only and card + element.
- Python model: Y = exp(g) Y_card + I_card exp(g) dg/dV (dg/dV by central differences of g, h = 1e-5 V).
- Statistic: max |Y_elem - Y_py| / max(|Y_py|, 1e-9 S) (real parts; imaginary parts with floor 1e-15 S) over
  points with |I_py| > 1 nA.
- Capacitance: ratios Im Y_dd(elem)/Im Y_dd(card), Im Y_dg(elem)/Im Y_dg(card) vs exp(g); Im Y_gg(elem)/Im Y_gg(card).
  Expected from the netlist: I(sense) is the total BSIM drain-terminal current including its displacement current,
  so the element scales the drain displacement current by exp(g); the gate admittance is unchanged.
- 1 mV sweeps (tight): Id-Vg at the three gm |Vds|, Id-Vd at the five gds |Vgs|, card only and card + element;
  central-difference gm/gds of the element current and of the Python model vs the AC values (descriptive).
- Sign rule (curve level): strong inversion is |Vgs| >= V_T + 0.2 V, V_T = card-only |Vgs| where |I_D| = 1e-7 W_eff/L
  A at |Vds| = 0.05 V. A gm curve has an introduced sign change if the element curve has gm < -1e-12 S in strong
  inversion (AC points or 1 mV sweep) and the card-only curve has none. Same rule for gds curves at |Vds| >= 0.05 V.
- Kink at Vgs = 0 from abs(): jump I exp(g) 2 |dg/d|Vgs|| (absolute and relative to the card gm), descriptive.
  At Vds = 0 the jump is zero because I_card = 0.

## Q3 design

| circuit | devices (measured geometries) | analysis | outputs |
|---|---|---|---|
| inv_A_vtc | nmos_L0p15_W1p6, pmos_L0p35_W1p6 | DC Vin 0-1.8 V, 1 mV | V_M (Vout = Vin), max \|dVout/dVin\| |
| inv_B_vtc | nmos_L1_W1p6, pmos_L2_W5 | as above | as above |
| ro5_A | inv_A pair, 5 stages, each loaded by the next stage | tran 1 ps / 10 ns, .ic alternating | frequency (rising 0.9 V crossings after 3 ns), periods |
| ro5_B | inv_B pair | tran 5 ps / 50 ns | after 15 ns |
| mirror_n | 2 x nmos_L1_W1p6, Iref 10 uA | DC Vout 0-1.8 V, 10 mV | Iout/Iref at Vout = 0.9 V and at Vout = V(ref) |
| mirror_p | 2 x pmos_L2_W5, Iref 10 uA, source at VDD | as above | as above |

VDD 1.8 V; default tolerances; default sky130 capacitances (BSIM4 intrinsic, overlap, junction from ad/as/pd/ps).
Arms: published card, C0_refit card (stored all18 run), M_card (M_full card, no element), M_full (M_full card +
element). The geometries are measured devices; the pMOS lvt minimum L is 0.35 um, so the pairs are not matched
sky130 standard cells. Smoke note: with the C0_refit card of pmos_L2_W5, ro5_B stops with "timestep too small"
without any element (card-level transient failure; removing the k1 or the k2 change of that card removes it);
ro5_B is kept and card-only failures are reported.

## Q4 design

Wall time of the ngspice process (Python `perf_counter`), median of 3 repeats: default-tolerance grid deck
(card only vs card + element, 18 devices) and the six circuits (M_card vs M_full). Ratio element/card only.

## Decision rule

| check | pass condition |
|---|---|
| Q1 core | tight max relative difference <= 1e-5 (\|I_py\| > 1 nA) on all 18 devices; 0 convergence messages and 0 non-finite values in element runs; max \|I_elem(Vds = 0)\| <= 1e-12 A |
| Q2 core | normalized AC derivative difference <= 1e-3 (gm, gds, Im Y_dg, Im Y_dd); 0 gm curves with an introduced sign change in strong inversion |
| Q3 core | 0 circuits that fail with the element (convergence message, non-finite output, timeout 120 s, or < 5 ring periods) while the M_card run of the same circuit passes |

Restrictions (each listed in the outcome when it applies): default-tolerance Q1 difference > 1e-3; terminal-exchange
asymmetry > 1e-3; drain displacement current changed by the element (max \|Im Y_dd ratio - 1\| > 1e-3); >= 1 gds
curve with an introduced sign change; wall-time ratio > 2 for any device sweep or circuit.

Outcome: any core check fails -> **not deployable**; all core checks pass and >= 1 restriction -> **deployable
with restrictions**; otherwise **deployable (simulator-level)**. Circuit differences between arms are reported
descriptively. The tight threshold 1e-5 was set after the smoke (smoke maximum 2.5e-6 at 1.7 nA, absolute
difference about 4e-15 A); the other thresholds are those of the task statement or set before the smoke.

## Expected restrictions (stated before the full run)

From the module text: (a) g uses abs(V(d,s)) and V(g,s) of the instance terminals, so for exchanged terminals the
element is not symmetric; (b) the element multiplies the total drain-terminal current, including displacement and
drain-body junction currents; (c) at default reltol the OSDI and Python currents differ at the reltol / abstol
level. The 2-device smoke also showed negative gds introduced on pmos_L2_W5 and a wall-time ratio of about 3.4 on
ro5_A.

## Outputs

`runs/full/{compile, refit, cards_M_full, q12, q3, summary, decision, timing}.json`, arrays (`*.npz`, not in git),
report batch `out/research/report/2026-10-21_i11_1/` (`scripts/research/plot_explore_i11_1.py`), site claim
`i11-1`.

## Compute

Light: refit 18 x about 10 s, q12 about 18 x 5 s, q3 24 runs; 4 workers; no queue job running at registration.
