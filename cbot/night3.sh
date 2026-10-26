#!/bin/bash
# The 3-player run, done by $DEADLINE (default 07:00): rounds t0 -> t1 -> t2
# (cbot/round_t*.sh), then self-play by the best bot so far into data/t3 until
# the deadline (the start of the next round's data); a failed round goes
# straight to that self-play. At the deadline anything still running is
# stopped. Logs: runs/night3.log (the chain and the fill), runs/round_t*.log.
#
#     nohup cbot/night3.sh > runs/night3.log 2>&1 < /dev/null &
#     DEADLINE=18:00 nohup cbot/night3.sh ...    # another end time
cd "$(dirname "$0")/.."
# Its own process group, so the stop reaches everything it started.
[ "$(ps -o pgid= $$ | tr -d ' ')" = "$$" ] || exec setsid "$PWD/cbot/night3.sh" "$@"
. cbot/round_lib.sh
log "3-player run until $(date -d "@$END" '+%m-%d %H:%M')"

# The stop: this script's process group (rounds, trainers, self-play, arenas).
( sleep $(( END - $(date +%s) )); log "deadline: stopping"; kill -- -$$ ) &
stopper=$!

ok=1
for r in t0 t1 t2; do
  cbot/round_$r.sh > "runs/round_$r.log" 2>&1
  grep -q "round $r done" "runs/round_$r.log" || { log "round $r failed: $(tail -n 1 "runs/round_$r.log")"; ok=0; break; }
  log "round $r done"
done

# Fill: the best bot so far (T0 without a policy if t1 never finished).
min=$(( (END - $(date +%s)) / 60 - 2 ))
if [ "$min" -ge 10 ] && [ -f models/T0.txt ]; then
  if [ -s runs/t_best.txt ]; then
    { read -r v; read -r p; read -r c; } < runs/t_best.txt
    pol=(--policy "$p")
  else
    v=T0; c=1.0; pol=()
  fi
  log "self-play into data/t3 for $min min: $v ${pol[*]} cpuct $c"
  nice -n 5 .venv/bin/python -m cbot.ngen --out data/t3 --games 1000000 --seed0 730000000 --threads 32 --players 3 \
    --model "$(cat models/$v.txt)" "${pol[@]}" --search sims=400,cpuct=$c,noise=0.25 --p-record 0.25 \
    --minutes "$min" >> runs/gen_t3.log 2>&1
  log "$(tail -n 1 runs/gen_t3.log)"
fi
log "done (rounds ok: $ok)"
kill $stopper 2>/dev/null
