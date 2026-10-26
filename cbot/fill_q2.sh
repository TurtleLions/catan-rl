#!/bin/bash
# Keeps the cores on 4-player self-play until $DEADLINE (default 07:00): the
# best bot so far (runs/q_best.txt: value committee, policy, cpuct; A20big with
# no policy if there is none) plays into data/q2 (seeds 600M on, root noise
# .25, p-record .25), the start of the next round's data. Resumable.
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
end=$(date -d "${DEADLINE:-07:00}" +%s)
min=$(( (end - $(date +%s)) / 60 - 2 ))
[ "$min" -ge 10 ] || { log "under 10 min left, not starting"; exit 0; }
if [ -s runs/q_best.txt ]; then
  { read -r v; read -r p; read -r c; } < runs/q_best.txt
  pol=(--policy "$p")
else
  v=A20big; c=1.0; pol=()
fi
log "self-play into data/q2 for $min min: $v ${pol[*]} cpuct $c"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/q2 --games 1000000 --seed0 600000000 --threads 32 --players 4 \
  --model "$(cat models/$v.txt)" "${pol[@]}" --search sims=400,cpuct=$c,noise=0.25 --p-record 0.25 \
  --minutes "$min" >> runs/gen_q2.log 2>&1
log "$(tail -n 1 runs/gen_q2.log)"
