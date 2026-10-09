#!/bin/bash
# Start the 4-player loop on every node, or top up the nodes that are missing.
# Run on the login node, from the repo:
#   cbot/launch.sh                  # start whatever isn't running yet
#   cbot/launch.sh --stop dunes     # stop the process on a node without Slurm
#   cbot/launch.sh --stop-all       # cancel the Slurm jobs and stop those processes
# Env: SLURM_NODES (default "matthew monster khai"): one catan4 job each, unless
#      one is already pending or running there; jobs wait for their node.
#      PLAIN_NODES (default "dunes"): nodes without Slurm, reached with ssh; a
#      background cbot/run_node.sh each (pid in runs/pids/<name>.pid, output in
#      runs/node-<name>.out), unless it is already running.
# Which node leads: cbot/node.py (dunes, then matthew, by default).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
SLURM_NODES=${SLURM_NODES-matthew monster khai}
PLAIN_NODES=${PLAIN_NODES-dunes}
mkdir -p runs/pids

plain_alive() {  # node
  local pid
  pid=$(cat "runs/pids/$1.pid" 2>/dev/null) || return 1
  ssh -n "$1" "kill -0 $pid" 2>/dev/null
}

plain_stop() {  # node: SIGTERM lets it release the leader lease first
  if plain_alive "$1"; then
    ssh -n "$1" "kill -TERM $(cat "runs/pids/$1.pid")"
    echo "$1: stopped"
  else
    echo "$1: not running"
  fi
}

case "${1:-}" in
  --stop)
    plain_stop "$2"
    exit 0
    ;;
  --stop-all)
    scancel -u "$USER" -n catan4 && echo "slurm: catan4 jobs cancelled"
    for n in $PLAIN_NODES; do plain_stop "$n"; done
    exit 0
    ;;
  "") ;;
  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac

for n in $SLURM_NODES; do
  if squeue -h -u "$USER" -n catan4 -o "%n %N" | tr ' ,' '\n\n' | grep -qx "$n"; then
    echo "$n: catan4 job already pending or running"
  else
    echo "$n: $(sbatch -w "$n" cbot/slurm/loop4.sbatch)"
  fi
done

for n in $PLAIN_NODES; do
  if plain_alive "$n"; then
    echo "$n: already running (pid $(cat "runs/pids/$n.pid"))"
    continue
  fi
  ssh -n "$n" "cd '$ROOT' && { setsid nohup cbot/run_node.sh >> 'runs/node-$n.out' 2>&1 < /dev/null & echo \$! > 'runs/pids/$n.pid'; }"
  sleep 2
  if plain_alive "$n"; then
    echo "$n: started (pid $(cat "runs/pids/$n.pid"), output runs/node-$n.out)"
  else
    echo "$n: failed to start; see runs/node-$n.out"
  fi
done
