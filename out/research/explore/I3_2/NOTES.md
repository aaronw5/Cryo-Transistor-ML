# I3_2 notes

Pre-registration: `prereg.json` (committed alone before any M_bias result existed).

## Design

- g form: `I_corr = I_sim * exp(g)`, `g = tanh(h(x))`, h = MLP 3-16-16-1 (M_bias, x = is_pmos,
  |Vgs|/VDD, |Vds|/VDD) or 5-16-16-1 (M_bias_geom, x adds log10 L, log10 W). tanh hidden units,
  zero-initialized last layer, L2 1e-4 on weight matrices, Adam lr 3e-3, 2000 full-batch steps,
  float64, 5 seeds (0-4) averaged. This is the E1 M recipe with the input columns and the hidden
  width changed. A unit test checks that `fit_bias` with all six E1 columns and width 32
  reproduces `e1_correction.fit_arm("M")` to 1e-12.
- Training data: E1 stored training records (starting cards' simulations). No joint rounds.
- Held-out refit: the I2-1 held-out task (`scripts/research/explore/i2_1_joint.py`,
  `heldout_task`): `ngspice_fit.polish` from the starting card, max_nfev 400, irls 2,
  diff_step 0.01, correction in the loop, best card re-simulated and scored with
  `e1_correction.score_device`.
- Stored rows reused:
  - C0_refit: I2-1 `real_dev` / `{pdk}_c_logo` `per_device.C0_refit`.
  - T1_inloop (REAL_DEV): E1 stage 2, `out/research/explore/E1/real_dev_refit/results.json`
    `rows.T1` (0.08198). It is used instead of I2-1 T1_joint (0.08035) because it is trained
    post-hoc on the starting cards, as M_bias is. T1_joint is reported as a reference row.
  - T1_inloop (GF180/IHP): new held-out refit with the E1 leave-one-geometry-out fold T1.
  - The full runs re-run C0_refit (all modes) and T1_inloop (REAL_DEV) as reproduction checks.
- Decision statistic: paired geometric-mean ratio with bootstrap 95% CI (10000 resamples,
  default_rng(0); device resampling on REAL_DEV, geometry-cluster resampling on GF180/IHP).
- Verilog-A: `to_verilog_a_bias` writes the network as explicit tanh assignments of pmos,
  abs(V(g,s))/VDD, abs(V(d,s))/VDD (and log(L), log(W) for M_bias_geom; Verilog-A log is base 10).
  `eval_verilog_a_g` evaluates the emitted assignment lines in Python; the real-mode run checks
  the exported full model on 200 random points. Not compiled.

## Files

- `src/cryoml/explore/i3_2_bias.py`: `BiasCorrection`, `fit_bias`, per-seed split/merge,
  `paired_ratio`, `to_verilog_a_bias`, `eval_verilog_a_g`.
- `scripts/research/explore/i3_2_bias.py`: modes `real`, `pdk2`, `decide`, `export`.
- `tests/test_i3_2_bias.py`: 10 unittest cases.

## Smoke runs (code check only; 1 seed, 200 steps, full refit settings)

| run | folds | held-out | runtime | M_bias mean | M_bias_geom mean | C0_refit mean | T1_inloop mean | C0 rerun max diff | T1 rerun max diff |
|---|---|---|---|---|---|---|---|---|---|
| smoke_real | 2 | nmos_L0p15_W1p6, nmos_L0p25_W1p6 | 44 s | 0.02504 | 0.02983 | 0.01908 | 0.02014 | 0.0 | 0.0 |
| smoke_gf180 | 1 | 3 targets of one geometry | 56 s | 0.10696 | 0.10552 | 0.12164 | 0.11068 (new) | 0.0 | - |

The exported smoke_real Verilog-A text (`smoke_real/m_bias_sky130_dev_full.va`, 67 lines,
1 member) matches `BiasCorrection.g` to 6.6e-12. These numbers come from reduced training and
2 or 1 folds and are not evidence for or against the hypothesis.

## Queue (appended to out/research/queue.txt; runner left stopped)

```
I3_2_real | .venv/bin/python scripts/research/explore/i3_2_bias.py --mode real --name real_dev --workers 4
I3_2_gf180 | .venv/bin/python scripts/research/explore/i3_2_bias.py --mode pdk2 --pdk gf180 --name gf180_c_logo --workers 6
I3_2_ihp | .venv/bin/python scripts/research/explore/i3_2_bias.py --mode pdk2 --pdk ihp --name ihp_c_logo --workers 6
I3_2_decide | .venv/bin/python scripts/research/explore/i3_2_bias.py --mode decide
```

Wall-time estimate (from smoke timings and the I2-1 full runs): real about 6 min (125 seed
tasks, 48 refits), GF180 about 30-40 min (160 seed tasks of about 50-70 s, 192 refits),
IHP about 35-45 min; total about 75-90 min.
