#!/bin/bash
# Start this node's cbot.node process (lead or help the 4-player loop), from a
# fresh code snapshot, after preflight checks. Used by cbot/slurm/loop4.sbatch
# and, on nodes without Slurm, by cbot/launch.sh (in the background, with its
# pid in runs/pids/<name>.pid; stop it with cbot/launch.sh --stop <name>).
#   cbot/run_node.sh            # run from the repo root
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
# (nproc obeys OMP_NUM_THREADS, which the Python workers set to 1.)
export P=${P:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)}
NAME=$(hostname -s)
TAG=${SLURM_JOB_ID:-$NAME-$$}

# Run from a per-run snapshot of the code, so a `git pull` meanwhile can't
# change what later steps import. A requeued Slurm job keeps its id and
# refreshes the snapshot. Data, models, runs and the venv stay shared.
SNAP=$ROOT/jobs/$TAG
mkdir -p "$SNAP"
rm -r -f "$SNAP/cbot"
cp -r "$ROOT/cbot" "$SNAP/"
for x in .venv data models runs; do ln -sfn "$ROOT/$x" "$SNAP/$x"; done
cd "$SNAP"
echo "== code snapshot $SNAP at $(git -C "$ROOT" rev-parse --short HEAD)"
echo "== $NAME $(date) restart=${SLURM_RESTART_COUNT:-0} procs=$P mem=$(awk '/MemTotal/ {printf "%.0f", $2 / 1048576}' /proc/meminfo)GB"
.venv/bin/python -c 'import sys, numpy, torch; print(sys.version.split()[0], numpy.__version__, torch.__version__)'

echo "== preflight: data checksums"
sha256sum -c --quiet data/MANIFEST.sha256
echo "== preflight: cbot.fast equivalence (expect 1/1 identical)"
# (No `tee /dev/stderr`: reopening the output file truncates it.)
out=$(.venv/bin/python -m cbot.check_fast --games 1 --table AB,AB,F,F)
echo "$out"
grep -q "identical games 1/1" <<< "$out"

exec .venv/bin/python -m cbot.node --procs "$P"
