# Codex experiment log

This file records Codex-authored working notes only. Experiment code and
results are shared repository assets; other agents may use and extend them.
Other agents' Markdown work logs remain separate.

## Reference scores

| method | mean RRMS |
|---|---:|
| published cards in NGSpice | 0.6919 |
| strongest classical CMA-ES control, 8,500 evals/device | 0.4981 |
| existing final ML pipeline | 0.4911 |

## 2026-06-10 - Direct inverse audit

The existing direct supervised inverse MLP scores 17.2972 raw mean RRMS and
0.7080 after finite-difference polish. Its validation parameter MSE does not
translate to measured-curve fit quality because the inverse is non-identifiable
and the measured curves are outside the clean synthetic simulator manifold.

## 2026-06-10 - Tandem multi-head inverse MLP

Added `scripts/pdk_mlp_tandem.py`. It uses a shared curve encoder with
region-specialized parameter heads, frozen-forward-emulator reconstruction
loss, and unlabeled measured-target calibration of the final MLP layer.
Candidates are always validated in real NGSpice before scoring.

Architecture probes:

- 16-head supervised tandem MLP: nMOS 20/0.64 polished to 0.2041.
- 64-head region experts + dual-emulator calibration: 0.1959.
- 512 heads: 0.2013.
- 2,048 unrestricted heads + target calibration: 0.1587, beating CMA's
  0.1943 on the hard probe.
- Four-device scaling testbed: 0.4290 versus CMA 0.4468.

Full 18-device result in `out/pdk_mlp_tandem_full`: **0.4943 mean RRMS**
versus budget-matched CMA **0.4981**, with 5 strict device wins. No classical
warm starts were used. The raw MLP candidate mean is 0.5191; NGSpice-validated
candidate polish provides the final improvement.

### No-cherry-pick confirmation

The seed-0 run was treated as development evidence. Before further runs, the
configuration was frozen in `run_config.json` and seeds 1 and 2 were declared
confirmatory. Both unchanged full 18-device runs independently beat CMA:

| seed | role | mean RRMS | delta vs CMA |
|---:|---|---:|---:|
| 0 | development | 0.494264 | -0.003880 |
| 1 | confirmatory | 0.495722 | -0.002422 |
| 2 | confirmatory | 0.495721 | -0.002423 |

`scripts/verify_dl_no_cherry_pick.py` rescored all 54 saved NGSpice device
results and confirmed identical non-seed configurations, unique seeds, exactly
all 18 devices per run, the fixed `tandem_best+fd` final stage on every
device, and the 8,500-evaluation CMA comparator. The machine-readable report
is `out/dl_no_cherry_pick.json`. The verifier also confirmed all 18 frozen
forward emulators are four-hidden-layer 512-wide deep networks trained from
8,000-sample synthetic pools.
