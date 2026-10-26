#!/bin/bash
# 3-player bootstrap, round t0 (~1.5 h): there is no 3-player net to search
# with, so 60k games between AlphaBetas (and a few ValueFunction players, for
# variety) give the first value nets T0 (5 nets, 142 -> 512 -> 256 -> 3,
# trained on the winners, lam 1). Then vs two AlphaBetas on seeds 340M (1000
# games, seats rotated; chance is 33%): T0's expectimax (the 4-player bot's
# first search) and the native tree search with T0 and no policy.
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
. cbot/round_lib.sh

log "generating t0 (AlphaBeta games)"
nice -n 5 .venv/bin/python -m cbot.gen --out data/t0 --games 60000 --procs 32 --seed0 700000000 \
  --mix "AB,AB,AB:3 AB,AB,F:2 AB,F,F:1" --p-record 0.2 >> runs/gen_t0.log 2>&1
log "$(tail -n 1 runs/gen_t0.log)"

train_round T0 - 3 1 data/t0

mkdir -p runs/eval3
arena runs/eval3/t0_T0_nn.log 340000000 "NN:$(cat models/T0.txt):initf=2" AB AB
arena runs/eval3/t0_T0_mcts.log 340000000 "NMCTS:$(cat models/T0.txt):initf=2:sims=400:cpuct=1.0" AB AB
log "round t0 done"
