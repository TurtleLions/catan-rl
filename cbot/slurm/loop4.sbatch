#!/bin/bash
# 4-player expert-iteration loop on one compute node.
#   sbatch cbot/slurm/loop4.sbatch                      # 4 rounds from round 1
#   ROUNDS=2 FIRST=3 sbatch cbot/slurm/loop4.sbatch     # env is passed through
# Output: runs/slurm-<jobid>.out, runs/loop4.log (one line per step).
#SBATCH --job-name=catan4
#SBATCH --partition=long
#SBATCH --nodelist=matthew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --output=runs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export P=${SLURM_CPUS_PER_TASK:-64}

echo "== $(hostname) $(date) cpus=$P nproc=$(nproc) mem=${SLURM_MEM_PER_NODE:-?}MB"
.venv/bin/python -c 'import sys, numpy, torch; print(sys.version.split()[0], numpy.__version__, torch.__version__)'

echo "== preflight: data checksums"
sha256sum -c --quiet data/MANIFEST.sha256
echo "== preflight: cbot.fast equivalence (expect 1/1 identical)"
.venv/bin/python -m cbot.check_fast --games 1 --table AB,AB,F,F | tee /dev/stderr | grep -q "identical games 1/1"

exec cbot/loop4.sh
