#!/bin/bash
# 4-player native self-play, bootstrap round q0, ~3 h: 50k games, every seat
# the native tree search with A20big (the released 4-player value committee)
# and no policy (uniform priors), 400 sims, cpuct 1, root noise .25,
# p-record .25; then
#   P4a = 4-player policy on q0's visit counts,
#   Q0  = 5 value nets (179 -> 512 -> 256 -> 4, lam .3, 3 epochs) on q0,
# and each combination vs three AlphaBetas on unused seeds 310M (1000 games,
# seats rotated; chance is 25%), A20big's expectimax (the released bot) as
# the baseline on the same seeds.
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
A=$(cat models/A20big.txt)

log "generating q0"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/q0 --games 50000 --seed0 400000000 --threads 32 --players 4 \
  --model "$A" --search sims=400,cpuct=1.0,noise=0.25 --p-record 0.25 >> runs/gen_q0.log 2>&1
log "$(tail -n 1 runs/gen_q0.log)"

# The policy trainer (12 threads) and the five value trainers (4 threads each)
# run side by side on the 32 cores; a value trainer holds ~7 GB and the policy
# trainer ~10 GB, so each starts only when that much memory is free. The
# training and arena wall times go to runs/q0_times.txt (round q1 sizes its
# self-play by them).
memfree() { awk '/MemAvailable/{print int($2 / 1048576)}' /proc/meminfo; }
wait_mem() {  # GB; gives up after 2 h
  local t=0
  while [ "$(memfree)" -lt "$1" ]; do
    [ "$t" -ge 120 ] && { log "only $(memfree) GB free for 2 h, stopping"; exit 1; }
    sleep 60; t=$((t + 1))
  done
}
t_train=$(date +%s)
log "training P4a and Q0"
if [ ! -f models/P4a.npz ]; then
  wait_mem 12
  .venv/bin/python -m cbot.train_policy --data data/q0 --out models/P4a.npz --threads 12 > runs/train_P4a.log 2>&1 &
  sleep 60
fi
paths=()
for i in 1 2 3 4 5; do
  out=models/Q0_s$i.npz
  paths+=("$out")
  [ -f "$out" ] && continue
  wait_mem 9
  .venv/bin/python -m cbot.train --data data/q0 --out "$out" --epochs 3 --threads 4 --hidden 512,256 \
    --cols compact --eval-every 2000 --lam 0.3 --players 4 > runs/train_Q0_s$i.log 2>&1 &
  sleep 60  # past the loading peak before checking memory again
done
wait
[ -f models/P4a.npz ] || { log "models/P4a.npz missing"; exit 1; }
for out in "${paths[@]}"; do [ -f "$out" ] || { log "$out missing"; exit 1; }; done
[ -f models/Q0.txt ] || (IFS=+; echo "${paths[*]}") > models/Q0.txt
grep -h "best val" runs/train_P4a.log
grep -h "best val" runs/train_Q0_s*.log
echo "train $(( $(date +%s) - t_train ))" > runs/q0_times.txt

arena() {  # name, spec, seed0
  local out=runs/eval4/q0_$1.log
  if ! grep -q " = " "$out" 2>/dev/null; then
    OMP_NUM_THREADS=1 .venv/bin/python -m cbot.arena "$2" AB AB AB --games 1000 --procs 31 --seed0 $3 \
      --log runs/evalset.jsonl > "$out" 2>&1
  fi
  log "$1: $(tail -n 1 "$out" | sed 's/.*: \([0-9]*\/[0-9]* = .*\)/\1/')"
}
t_arena=$(date +%s)
arena base "NN:$A:initf=2" 310000000
for v in A20big Q0; do
  for c in 0.3 1.0; do
    arena "${v}_P4a_c$c" "NMCTS:$(cat models/$v.txt):initf=2:sims=400:cpuct=$c:policy=models/P4a.npz" 310000000
  done
done
echo "arena $(( ($(date +%s) - t_arena) / 5 ))" >> runs/q0_times.txt
log "round q0 done"
