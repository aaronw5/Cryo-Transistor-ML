# I13-1 pre-registration (IN FORCE): BSIM4 A2 clamp state of the 43-parameter cards, audit and fix, sky130 77 K, 18 devices

Machine-readable version: `prereg.json` (authoritative if the two differ). In force from the commit that adds this
file (2026-10-21), committed alone.

Done before it (not committed, not used for any decision): code draft in `src/cryoml/spice_pdk.py`
(`bsim4_state` argument of `simulate_pdk`, `bsim4_clamp_params`, `bsim4_check_messages`); a mechanism check on two
starting cards (direct43_r3_fd): nmos_L0p19_W7 (a2 1.877): state C curve 1 equals the legacy curve 1 exactly, state U
curves 2-11 equal the legacy curves 2-11 exactly, legacy curves 2-11 differ from state C by up to 0.27 (relative
current); nmos_L1_W1p6 (a2 0.888): C, L and U identical. A listing of the stored a2 values (no simulation): a2 > 1 in
the start, C0_refit, T1_inloop, M_bias v1 and M_bias v2 cards of nmos_L0p19_W7, nmos_L0p25_W1p6, nmos_L20_W0p64,
nmos_L8_W1p6 and pmos_L0p35_W1p6 (5 devices; the I12-1 diagnosis named the 4 nMOS); no stored card has a2 < 0.01.

## Cause (I12-1 diagnosis)

ngspice-41 `b4check.c` (`BSIM4checkModel`) runs once per process per model name (static `modname`) and clamps the
size-dependent A2 > 1 to A2 = 1, A1 = 0 (and A2 < 0.01 to 0.01). `b4temp.c` rebuilds the size-dependent parameters
at every analysis. `spice_pdk._batch_deck` runs one dc analysis per curve in one process, so for a card with
a2 > 1 the first curve in deck order is simulated with A2 = 1 and the other curves with the card value.

## Clamp states (definitions)

- **L (legacy)**: the deck as used for all stored 43-parameter scores (first curve clamped, later curves not).
- **C (clamped)**: every curve in the state ngspice applies in a fresh single-analysis process. Implementation: the
  clamp is written into the card before simulation (a2 > 1 -> a2 = 1 and a1 = 0; a2 < 0.01 -> a2 = 0.01). The 77 K
  bins have no la1/wa1/pa1/la2/wa2/pa2 terms and a1 = 0 in every bin, so the size-dependent A2 equals the card a2
  and A1 = 0 already; `b4ld.c` is the only user of A1/A2, so state C is the fresh-process state exactly. A card that
  the check does not clamp is written unchanged (identical deck text).
- **U (unclamped)**: one `op` before the curves, so every curve uses the card a2 (audit only).

## Fix rule

A card is valid only if every curve is evaluated in one clamp state. **Chosen option (b): score every curve in state
C** (the fresh-process state), `simulate_pdk(..., bsim4_state="clamped")` as the new default. This leaves the paper
metric (strict paper RRMS, curve inclusion frozen to the published card) and the published-card baseline unchanged:
the published bins have a2 in [0.01, 1] (max 0.467) and the 7-parameter paper pipeline does not change a2. Required:
the published cards and every card with 0.01 <= a2 <= 1 give identical scores under C and L (difference 0),
unit-tested (deck text identity and a simulation test). Option (a) (bound a2 <= 1 in the 43-parameter box) is not
used; in state C a value a2 > 1 has the same effect as a2 = 1, a1 = 0.

Other b4check.c clamps on size-dependent parameters (ckappas/ckappad < 0.02, prwg < 0, rdsw < 0, rds0 < 0,
rdswmin < 0, xn < 3 and lc < 0 when vtl is given) are listed in the audit with `bsim4_check_messages` (one `op` in a
fresh process per stored card). A stored card that triggers one of them is reported; if any does, the fix is extended
to that clamp before the reruns (amendment).

## Audit quantities

Arms (stored cards, 18 devices each): published (bin card), start (direct43_r3_fd), C0_refit, T1_inloop, M_bias v1
(`I3_2/all18`), M_bias v2 (`I12_1/lodo`). For M_bias v1 / v2 the stored fold corrections are applied; for T1_inloop
the fold T1 tables are refitted from the stored E1 records (deterministic, `e1_correction.fit_arm`). Each arm row is
rescored in L, C and U: per-device strict paper RRMS, per-device differences C - L and U - L, all-device mean,
combined (mean of the nMOS and pMOS family means), median.
Validity: (V1) L reproduces every stored row (max |diff| <= 1e-12); (V2) C = L = U exactly for every card that the
check does not clamp (published and a2 in [0.01, 1]); (V3) 18 rows per arm and state.

## Rerun in state C (heavy jobs through the queue, one at a time)

1. Records: the 18 starting-card records (E1 `real_dev` + `real_test`) re-collected in state C ->
   `I13_1/records/`. Required: sims of devices whose start card is not clamped identical to the stored records.
   The starting cards are not refit (fixed input; their state-C score is the start row).
2. I3-2 all18 recipe unchanged (fold T1 fits; M_bias v1 3-16-16-1, 5 seeds; held-out refits C0_refit, T1_inloop,
   M_bias with max_nfev 400, irls 2, diff_step 0.01; strict paper RRMS) on the state-C records with state-C
   simulation -> `I13_1/all18/`.
3. I12-1 LODO recipe unchanged (g v2, lambda 10 V, EPS 0.01 V, 5 seeds, same held-out refit, the stored penalty grids:
   one nested dc analysis per device, i.e. already state C) -> `I13_1/lodo/`, with C0_refit / T1_inloop / M_bias v1
   rows from `I13_1/all18/`.

## Reported (headline all-18 table, state C)

published / start / C0_refit / T1_inloop / M_bias v1 / M_bias v2: all-device mean, combined, median; GMR M_bias v1 vs
C0_refit (device bootstrap, 10000 resamples, `i3_2_bias.paired_ratio`), GMR M_bias v2 vs C0_refit and vs M_bias v1
(the I12-1 A1 / A2 quantities, thresholds unchanged: A1 CI upper < 1, A2 CI upper <= 1.05; reported, I12-1 Part B not
rerun). Old (stored, state L) vs new values and per-device differences. Both aggregate conventions are reported; RRMS
is not redefined.
