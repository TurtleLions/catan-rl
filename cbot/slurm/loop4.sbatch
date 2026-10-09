#!/bin/bash
# 4-player expert-iteration loop, one job per node; all jobs are identical:
#   sbatch -w matthew cbot/slurm/loop4.sbatch
#   sbatch -w monster cbot/slurm/loop4.sbatch
#   sbatch -w khai    cbot/slurm/loop4.sbatch
# One job leads (runs cbot/loop4.sh: generation, training, evals); the others
# play generation chunks and take over the round if the leader goes away (see
# cbot/node.py). Any subset of nodes works, including one.
# Rounds and settings: cbot/loop4.conf, overridden live by runs/loop4.conf.
# Preempted jobs are requeued and rejoin with the repo's current code (the
# snapshot is refreshed on start). Status: cbot/slurm/status.sh
# Output: runs/slurm-<jobid>.out (appended across restarts), runs/loop4.log.
#SBATCH --job-name=catan4
#SBATCH --account=scc
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=runs/slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export P=$(nproc)

# Run from a per-job snapshot of the code, so a `git pull` during the job can't
# change what later steps import. A requeued job keeps its id and refreshes the
# snapshot. Data, models, runs and the venv stay shared.
ROOT=$PWD
SNAP=$ROOT/jobs/$SLURM_JOB_ID
mkdir -p "$SNAP"
rm -r -f "$SNAP/cbot"
cp -r "$ROOT/cbot" "$SNAP/"
for x in .venv data models runs; do ln -sfn "$ROOT/$x" "$SNAP/$x"; done
cd "$SNAP"
echo "== code snapshot $SNAP at $(git -C "$ROOT" rev-parse --short HEAD)"

echo "== $(hostname) $(date) restart=${SLURM_RESTART_COUNT:-0} cpus=$P nproc=$(nproc) mem=${SLURM_MEM_PER_NODE:-?}MB"
.venv/bin/python -c 'import sys, numpy, torch; print(sys.version.split()[0], numpy.__version__, torch.__version__)'

echo "== preflight: data checksums"
sha256sum -c --quiet data/MANIFEST.sha256
echo "== preflight: cbot.fast equivalence (expect 1/1 identical)"
# (No `tee /dev/stderr`: reopening the output file truncates it.)
out=$(.venv/bin/python -m cbot.check_fast --games 1 --table AB,AB,F,F)
echo "$out"
grep -q "identical games 1/1" <<< "$out"

exec .venv/bin/python -m cbot.node --procs "$P"
