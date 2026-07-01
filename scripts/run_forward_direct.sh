#!/bin/bash
# Train the direct IV->params extractor in both modes (validation = ngspice-46).
set -e
cd /Users/anrunw/Documents/cryo-ml
unset NGSPICE_BIN
echo "===== NO-SURROGATE (pure param-distance loss) $(date) ====="
.venv/bin/python scripts/pdk_forward_direct.py --device mps --recon-weight 0 \
    --out-dir out/pdk_fwd_nosurr --resume > out/pdk_fwd_nosurr_run.log 2>&1
echo "no-surrogate done $(date)"
.venv/bin/python3 -c "import json;print(json.load(open('out/pdk_fwd_nosurr/summary.json')))"
echo "===== WITH-SURROGATE (param-distance + emulator reconstruction) $(date) ====="
.venv/bin/python scripts/pdk_forward_direct.py --device mps --recon-weight 1.0 \
    --emu-dir out/pdk_ml2 --out-dir out/pdk_fwd_surr --resume > out/pdk_fwd_surr_run.log 2>&1
echo "with-surrogate done $(date)"
.venv/bin/python3 -c "import json;print(json.load(open('out/pdk_fwd_surr/summary.json')))"
echo "ALL FORWARD MODES DONE $(date)"
