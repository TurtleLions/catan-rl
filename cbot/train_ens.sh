#!/bin/bash
# Train N value-net seeds in parallel and write the ensemble spec to models/NAME.txt.
#   cbot/train_ens.sh NAME N LAM THREADS data/r0 data/r1 ...
set -e
cd "$(dirname "$0")/.."
NAME=$1; N=$2; LAM=$3; THREADS=$4; shift 4
paths=()
for i in $(seq 1 "$N"); do
  out=models/${NAME}_s$i.npz
  .venv/bin/python -m cbot.train --data "$@" --out "$out" --epochs 3 --threads "$THREADS" \
    --hidden 256,128 --cols "${COLS:-compact}" --eval-every 2000 --lam "$LAM" > runs/train_${NAME}_s$i.log 2>&1 &
  paths+=("$out")
done
wait
(IFS=+; echo "${paths[*]}") > models/$NAME.txt
grep -h "best val" runs/train_${NAME}_s*.log
cat models/$NAME.txt
