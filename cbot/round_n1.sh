#!/bin/bash
# Native self-play round n1, ~7 hours: games by the native tree search with
# N0 (round n0's winner) + P1 for 5.25 h (400 sims, cpuct .3, root noise .25,
# p-record .2), then
#   N1  = 5 value nets (256-128, lam .3, 3 epochs) on n1 alone,
#   N1b = 3 value nets, same recipe, on n0 + n1 (only if memory allows),
# each with P1 vs AB on unused seeds 130M (4000 games), and the best one
# head-to-head vs N0 + P1 on seeds 140M.
# Trainers start only when enough memory is free (the machine is shared).
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
memfree() { awk '/MemAvailable/{print int($2 / 1048576)}' /proc/meminfo; }
wait_mem() {  # GB, max minutes
  local t=0
  while [ "$(memfree)" -lt "$1" ]; do
    [ "$t" -ge "$2" ] && return 1
    sleep 60; t=$((t + 1))
  done
}
N0=$(cat models/N0.txt)
S="sims=400:cpuct=0.3"

log "generating n1"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/n1 --games 400000 --seed0 300000000 --threads 32 --minutes 315 \
  --model "$N0" --policy models/P1.npz --search sims=400,cpuct=0.3,noise=0.25 --p-record 0.2 >> runs/gen_n1.log 2>&1
log "$(tail -n 2 runs/gen_n1.log | head -n 1)"

train() {  # name, number of nets, GB needed, threads, data...
  local name=$1 n=$2 gb=$3 th=$4 paths=() i out
  shift 4
  for i in $(seq 1 "$n"); do
    out=models/${name}_s$i.npz
    paths+=("$out")
    [ -f "$out" ] && continue
    if ! wait_mem "$gb" 30; then log "$name: skipping net $i, only $(memfree) GB free"; continue; fi
    .venv/bin/python -m cbot.train --data "$@" --out "$out" --epochs 3 --threads "$th" --hidden 256,128 \
      --cols compact --eval-every 2000 --lam 0.3 > runs/train_${name}_s$i.log 2>&1 &
    sleep 120  # past the loading peak before checking memory again
  done
  wait
  paths=($(ls "${paths[@]}" 2>/dev/null))
  [ ${#paths[@]} -gt 0 ] && (IFS=+; echo "${paths[*]}") > models/$name.txt
  grep -h "best val" runs/train_${name}_s*.log
}
log "training N1 (n1 only)"
[ -f models/N1.txt ] || train N1 5 16 6 data/n1
log "training N1b (n0 + n1)"
[ -f models/N1b.txt ] || train N1b 3 30 10 data/n0 data/n1

arena() {  # name, spec A, spec B, seed0
  local out=runs/evalset/n1_$1.log
  if ! grep -q " = " "$out" 2>/dev/null; then
    OMP_NUM_THREADS=1 .venv/bin/python -m cbot.arena "$2" "$3" --games 4000 --procs 31 --seed0 $4 \
      --log runs/evalset.jsonl > "$out" 2>&1
  fi
  log "$1: $(tail -n 1 "$out" | sed 's/.*: \([0-9]*\/[0-9]* = .*\)/\1/')"
}
best=""; bestw=0
for v in N0 N1 N1b; do
  [ -f models/$v.txt ] || continue
  spec="NMCTS:$(cat models/$v.txt):initf=2:$S:policy=models/P1.npz"
  arena "${v}_P1_vsAB" "$spec" AB 130000000
  w=$(tail -n 1 runs/evalset/n1_${v}_P1_vsAB.log | sed 's/.*: \([0-9]*\)\/.*/\1/')
  if [ "$v" != N0 ] && [ "$w" -gt "$bestw" ]; then best=$spec; bestw=$w; bestname=${v}_P1; fi
done
log "best new: $bestname ($bestw/4000)"
arena "${bestname}_vs_N0_P1" "$best" "NMCTS:$N0:initf=2:$S:policy=models/P1.npz" 140000000
log "round n1 done"
