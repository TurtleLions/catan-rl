#!/bin/bash
# Precise evals of finished 4-player models vs three AlphaBetas, one after another:
#   sbatch -w matthew cbot/slurm/evalset.sbatch 4000 M0 M1 M2
# Each name is a models/<name>.txt ensemble spec. Fresh seeds (40M), full opening.
# Results: runs/evalset/<name>_<games>.log (last line) and runs/evalset.jsonl.
# Models with a finished log are skipped, so a requeued job continues.
#SBATCH --job-name=catan4-eval
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
P=$(nproc)
GAMES=$1; shift
mkdir -p runs/evalset
for name in "$@"; do
  logf=runs/evalset/${name}_${GAMES}.log
  if grep -q " = " "$logf" 2>/dev/null; then continue; fi
  .venv/bin/python -m cbot.arena "NN:$(cat "models/$name.txt"):initf=2" AB AB AB --games "$GAMES" \
    --procs "$P" --seed0 40000000 --log runs/evalset.jsonl > "$logf" 2>&1
  echo "$name: $(tail -n 1 "$logf")"
done
