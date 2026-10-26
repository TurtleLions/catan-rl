#!/bin/bash
# 3-player round t1, after t0 (cbot/round_t0.sh), as round q0 for 4 players:
# the native tree search with T0 and no policy (400 sims, cpuct 1, root noise
# .25, p-record .25) plays itself for $T1_MIN minutes (default 120), then
#   P3a = 3-player policy on the visit counts,
#   T1  = 5 value nets (512-256, lam .3, 3 epochs) on t1,
# side by side, and each combination vs two AlphaBetas on seeds 350M (1000
# games; chance 33%), T0's expectimax as the baseline. Training and arena wall
# times go to runs/t1_times.txt (round t2 sizes its self-play by them).
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
. cbot/round_lib.sh

log "generating t1"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/t1 --games 300000 --seed0 710000000 --threads 32 --players 3 \
  --model "$(cat models/T0.txt)" --search sims=400,cpuct=1.0,noise=0.25 --p-record 0.25 \
  --minutes "${T1_MIN:-120}" >> runs/gen_t1.log 2>&1
log "$(tail -n 1 runs/gen_t1.log)"

t=$(date +%s)
train_round T1 P3a 3 0.3 data/t1
echo "train $(( $(date +%s) - t ))" > runs/t1_times.txt

t=$(date +%s)
mkdir -p runs/eval3
arena runs/eval3/t1_base.log 350000000 "NN:$(cat models/T0.txt):initf=2" AB AB
for v in T0 T1; do
  for c in 0.3 1.0; do
    arena "runs/eval3/t1_${v}_P3a_c$c.log" 350000000 \
      "NMCTS:$(cat models/$v.txt):initf=2:sims=400:cpuct=$c:policy=models/P3a.npz" AB AB
  done
done
echo "arena $(( ($(date +%s) - t) / 5 ))" >> runs/t1_times.txt
log "round t1 done"
