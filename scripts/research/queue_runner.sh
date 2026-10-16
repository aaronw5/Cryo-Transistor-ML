#!/bin/zsh
# Run heavy research jobs strictly one at a time (repo resource policy).
# Jobs are lines in out/research/queue.txt; finished lines move to done.txt.
# Waits for any already-running heavy job first; exits when the queue is empty.
cd "$(dirname "$0")/../.."
Q=out/research/queue.txt; D=out/research/done.txt; L=out/research/logs
mkdir -p $L; touch $Q $D
while pgrep -f "pdk_gen_full_data.py|foundation_parameter_study.py|posterior_study.py|ngspice_direct.py|cold_loop.py" >/dev/null; do sleep 30; done
while true; do
  job=$(grep -v '^\s*$' $Q | grep -v '^#' | head -1)
  [ -z "$job" ] && break
  name=$(echo "$job" | cut -d'|' -f1 | tr -d ' ')
  cmd=$(echo "$job" | cut -d'|' -f2-)
  echo "$(date '+%F %T') START $name" >> $D
  start=$(date +%s)
  PYTHONPATH=src zsh -c "$cmd" > $L/$name.log 2>&1
  rc=$?
  echo "$(date '+%F %T') END $name rc=$rc $(( $(date +%s) - start ))s" >> $D
  # drop the first matching line from the queue
  python3 - "$Q" "$job" <<'PY'
import sys
path, job = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines()
for i, line in enumerate(lines):
    if line == job:
        del lines[i]; break
open(path, "w").write("\n".join(lines) + ("\n" if lines else ""))
PY
done
echo "$(date '+%F %T') QUEUE EMPTY" >> $D
