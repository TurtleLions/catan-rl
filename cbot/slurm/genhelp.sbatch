#!/bin/bash
# Helper for the 4-player loop: plays chunks of each round's data generation on
# another node, alongside the loop job (cbot/slurm/loop4.sbatch). One per node:
#   sbatch -w monster cbot/slurm/genhelp.sbatch
#   sbatch -w khai cbot/slurm/genhelp.sbatch
# Uses every core of the node. Idle during training and evals; exits after 3
# hours with no generation to do (e.g. once the loop is finished). Requeued if
# preempted; its unfinished chunks are taken over by other jobs after 15 min.
#SBATCH --job-name=catan4-help
#SBATCH --account=scc
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH --output=runs/slurm-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
P=$(nproc)

# Per-job code snapshot (see loop4.sbatch), refreshed on requeue.
ROOT=$PWD
SNAP=$ROOT/jobs/$SLURM_JOB_ID
mkdir -p "$SNAP"
rm -r -f "$SNAP/cbot"
cp -r "$ROOT/cbot" "$SNAP/"
for x in .venv data models runs; do ln -sfn "$ROOT/$x" "$SNAP/$x"; done
cd "$SNAP"
echo "== helper $(hostname) $(date) restart=${SLURM_RESTART_COUNT:-0} procs=$P code $(git -C "$ROOT" rev-parse --short HEAD)"

echo "== preflight: cbot.fast equivalence (expect 1/1 identical)"
out=$(.venv/bin/python -m cbot.check_fast --games 1 --table AB,AB,F,F)
echo "$out"
grep -q "identical games 1/1" <<< "$out"

exec .venv/bin/python -m cbot.genpool help --task runs/gen4_task.json --procs "$P"
