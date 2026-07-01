#!/bin/bash
# Continue the tandem training: additional confirmatory seeds, identical config
# except --seed. Validation uses ngspice-46 (the validated backend).
set -e
cd /Users/anrunw/Documents/cryo-ml
unset NGSPICE_BIN
for s in 3 4 5; do
  echo "===== tandem seed $s started $(date) ====="
  .venv/bin/python scripts/pdk_mlp_tandem.py --seed $s \
      --out-dir out/pdk_mlp_tandem_seed$s --resume \
      > out/pdk_mlp_tandem_seed${s}_run.log 2>&1
  echo "===== tandem seed $s done $(date) mean: ====="
  .venv/bin/python3 -c "import json; print(json.load(open('out/pdk_mlp_tandem_seed$s/summary.json'))['mean_rrms'])"
done
echo "ALL SEEDS DONE $(date)"
