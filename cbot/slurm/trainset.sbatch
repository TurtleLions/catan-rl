#!/bin/bash
# Train a 4-player ensemble on chosen data the way the loop does (3 nets at
# lam .5 + 3 at lam .3), then evaluate it vs three AlphaBetas on the evalset
# seeds (40M, 4000 games):
#   sbatch -w matthew cbot/slurm/trainset.sbatch M4w data/m0 data/m0b data/m1s ...
# Writes models/<name>.txt and runs/evalset/<name>_4000.log. Rerunnable: finished
# nets and evals are kept. SEQ=1 trains three nets at a time even with 160+ GB
# (each net holds the whole dataset, ~2x that while loading). STAGGER=<s> sets the
# delay between nets' starts (default 60); with a lot of data, 1200 keeps their loads
# from overlapping. HIDDEN=512,256 trains wider nets (default 256,128). P=<n>
# sets the cores to use (default all), e.g. to run two of these side by side
# on a node without Slurm:
#   SLURM_SUBMIT_DIR=$PWD P=64 setsid nohup bash cbot/slurm/trainset.sbatch NAME dirs... > runs/NAME.out 2>&1 &
#SBATCH --job-name=catan4-train
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
P=${P:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)}
NAME=$1; shift
export PLAYERS=4
echo "== $(hostname) $(date) training $NAME on $*"

if [ ! -f "models/$NAME.txt" ]; then
  mem_gb=$(( ${SLURM_MEM_PER_NODE:-$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 ))} / 1024 ))
  if [ "$mem_gb" -ge 160 ] && [ "${SEQ:-0}" != 1 ]; then
    T=$(( P / 6 ))
    STAGGER=${STAGGER:-60} cbot/train_ens.sh "${NAME}a" 3 0.5 "$T" "$@" > "runs/train_${NAME}a.log" 2>&1 &
    sleep 30
    STAGGER=${STAGGER:-60} cbot/train_ens.sh "${NAME}b" 3 0.3 "$T" "$@" > "runs/train_${NAME}b.log" 2>&1 &
    wait
  else
    T=$(( P / 3 ))
    STAGGER=${STAGGER:-60} cbot/train_ens.sh "${NAME}a" 3 0.5 "$T" "$@" > "runs/train_${NAME}a.log" 2>&1
    STAGGER=${STAGGER:-60} cbot/train_ens.sh "${NAME}b" 3 0.3 "$T" "$@" > "runs/train_${NAME}b.log" 2>&1
  fi
  echo "$(cat "models/${NAME}a.txt")+$(cat "models/${NAME}b.txt")" > "models/$NAME.txt"
fi
echo "== val losses $(grep -h 'best val' runs/train_${NAME}a_s*.log runs/train_${NAME}b_s*.log | awk '{print $3}' | tr '\n' ' ')"

mkdir -p runs/evalset
logf=runs/evalset/${NAME}_4000.log
if ! grep -q " = " "$logf" 2>/dev/null; then
  .venv/bin/python -m cbot.arena "NN:$(cat "models/$NAME.txt"):initf=2" AB AB AB --games 4000 \
    --procs "$P" --seed0 40000000 --log runs/evalset.jsonl > "$logf" 2>&1
fi
echo "== $NAME: $(tail -n 1 "$logf")"
