#!/bin/bash
# 4-player expert-iteration loop on one compute node.
#   sbatch cbot/slurm/loop4.sbatch
# Rounds and settings: cbot/loop4.conf, overridden live by runs/loop4.conf.
# If preempted, Slurm requeues the job; the loop resumes where it stopped and
# the restarted job runs the repo's current code (snapshot refreshed on start).
# Output: runs/slurm-<jobid>.out (appended across restarts), runs/loop4.log.
#SBATCH --job-name=catan4
#SBATCH --partition=long
#SBATCH --nodelist=matthew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --output=runs/slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --dependency=singleton
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export P=${SLURM_CPUS_PER_TASK:-64}

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

exec cbot/loop4.sh
