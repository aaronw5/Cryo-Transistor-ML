# I20-1 amendment A1 (2026-10-23): reading of audit validity V2 on cards with a fatal model check

Written after the audit job (`I20_1/audit/audit.json`), the state-C starting records and the state-C S3_D (v1
recipe) were computed, and before any C0_refit / M_bias v1 / g v2 rerun result. It changes no rerun quantity,
no contrast, no label rule and no gate rule.

Observation. Pre-registered V2: C = L = U (difference 0) for every card with 0.01 <= A2_eff <= 1. Result: C = L on
all 1238 such cards; U differs from L on 7 of them. The 7 cards are 5 stock cards and 2 starting cards of the
I8-1 devices nfet_01v8_lvt W0.55/L0.25, W3.01/L0.15, W5/L2, W7/L0.15, W7/L2, whose model check prints
"Fatal: Pclm = ... is not positive" (binned PCLM < 0). In states L and C the fatal check stops the first analysis
(first curve NaN; I8-1 amendment A1). In state U the warm-up `op` is the first analysis, so the fatal check stops
the `op` and every curve runs. The difference comes from the fatal PCLM check, not from the A2 clamp. Four of these
devices are excluded from every arm by I8-1 amendment A1; W5/L2 has a finite starting record (its fitted card has
PCLM > 0) and only its stock card is affected.

Reading used from here on. V2 is evaluated on the unclamped cards without a fatal model-check line (1231 cards);
on the 7 fatal-check cards C = L is required and U is reported. The unrestricted V2 result (7 cards with U != L)
is reported next to it. `i20_1_nist_clamp.decide_validity` receives the restricted V2 value.
