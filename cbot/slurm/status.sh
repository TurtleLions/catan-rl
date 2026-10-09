#!/bin/bash
# One-screen status of the 4-player loop and its jobs:
#   cbot/slurm/status.sh [lines of loop4.log, default 8]
cd "$(dirname "$0")/../.."
N=${1:-8}

echo "== jobs"
squeue -u "$USER" -o "%.8i %.13j %.9T %.10M %.8N %R" | sed 's/^/  /'

if [ -f runs/loop4.leader ]; then
  age=$(( $(date +%s) - $(stat -c %Y runs/loop4.leader) ))
  echo "== leader: $(cat runs/loop4.leader) (lease refreshed ${age}s ago)"
else
  echo "== leader: none"
fi

echo "== loop4.log (last $N)"
tail -n "$N" runs/loop4.log 2>/dev/null | cut -c1-200 | sed 's/^/  /'

task=$(cat runs/gen4_task.json 2>/dev/null)
if [ -z "$task" ]; then
  exit 0
fi
out=$(.venv/bin/python -c 'import json, sys; t = json.loads(sys.argv[1]); print(t["out"], -(-t["games"] // t["chunk"]))' "$task")
read -r dir nchunk <<< "$out"
done=$(ls "$dir" 2>/dev/null | grep -cE '^c[0-9]+\.(npz|empty)$')
claimed=$(ls "$dir/claims" 2>/dev/null | grep -cE '^[0-9]+$')
state=running; [ -f "$dir/DONE" ] && state=DONE
echo "== generation $dir: $done/$nchunk chunks written, $claimed in progress ($state)"

# Chunk lines from the leader's gen log and every job's Slurm output.
# Format: [date time] host:job:pid <dir>: chunk k written; ...
gen=runs/gen_$(basename "$dir").log
helpers=$(squeue -h -u "$USER" -n catan4 -t RUNNING -o "runs/slurm-%i.out" | tr '\n' ' ')
# shellcheck disable=SC2086
grep -h "$dir: chunk .* written" "$gen" $helpers 2>/dev/null | .venv/bin/python -c '
import sys, time
from datetime import datetime
now = time.time()
hosts = {}
for line in sys.stdin:
    ts = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp()
    host = line[22:].split(":", 1)[0].split(".")[0]
    h = hosts.setdefault(host, [0, 0, 0])
    h[0] += 1
    h[1] = max(h[1], ts)
    h[2] += now - ts < 3600
for host, (n, last, hour) in sorted(hosts.items()):
    print(f"  {host:10s} {n:4d} chunks this round, {hour:3d} in the last hour, last {(now - last) / 60:5.1f} min ago")
'
