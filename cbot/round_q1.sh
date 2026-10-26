#!/bin/bash
# 4-player native self-play round q1, ~7 h, after q0 (cbot/round_q0.sh): the
# best NMCTS combination of q0 by wins vs three AlphaBetas plays 150k games
# against itself (root noise .25, p-record .1: ~21M rows, so a value trainer
# fits in ~8 GB beside other work), then
#   P4b = policy on q1's visit counts,
#   Q1  = 5 value nets (512-256, lam .3, 3 epochs) on q1,
# each new combination vs three AlphaBetas on unused seeds 320M (1000 games;
# q0's best on the same seeds), and the better one vs three copies of q0's
# best on seeds 330M (1000 games, seats rotated: 25% = equal strength).
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
memfree() { awk '/MemAvailable/{print int($2 / 1048576)}' /proc/meminfo; }
wait_mem() {  # GB; gives up after 2 h
  local t=0
  while [ "$(memfree)" -lt "$1" ]; do
    [ "$t" -ge 120 ] && { log "only $(memfree) GB free for 2 h, stopping"; exit 1; }
    sleep 60; t=$((t + 1))
  done
}
wins() { tail -n 1 "$1" | sed -n 's/.*: \([0-9]*\)\/[0-9]* = .*/\1/p'; }

# q0's best NMCTS combination (value committee, policy, cpuct) by wins.
best=""; bw=-1
for v in A20big Q0; do
  for c in 0.3 1.0; do
    w=$(wins runs/eval4/q0_${v}_P4a_c$c.log)
    [ -n "$w" ] && [ "$w" -gt "$bw" ] && { bw=$w; bv=$v; bc=$c; }
  done
done
[ "$bw" -ge 0 ] || { log "no q0 results"; exit 1; }
G="NMCTS:$(cat models/$bv.txt):initf=2:sims=400:cpuct=$bc:policy=models/P4a.npz"
log "generator: $bv + P4a, cpuct $bc ($bw/1000 vs 3 AB in q0)"

log "generating q1"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/q1 --games 150000 --seed0 500000000 --threads 32 --players 4 \
  --model "$(cat models/$bv.txt)" --policy models/P4a.npz --search sims=400,cpuct=$bc,noise=0.25 --p-record 0.1 \
  >> runs/gen_q1.log 2>&1
log "$(tail -n 1 runs/gen_q1.log)"

log "training P4b"
if [ ! -f models/P4b.npz ]; then
  wait_mem 14
  .venv/bin/python -m cbot.train_policy --data data/q1 --out models/P4b.npz --threads 30 > runs/train_P4b.log 2>&1
fi
grep -h "best val" runs/train_P4b.log
log "training Q1"
paths=()
for i in 1 2 3 4 5; do
  out=models/Q1_s$i.npz
  paths+=("$out")
  [ -f "$out" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  while [ "$(memfree)" -lt 10 ] && [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n; done
  wait_mem 10
  .venv/bin/python -m cbot.train --data data/q1 --out "$out" --epochs 3 --threads 10 --hidden 512,256 \
    --cols compact --eval-every 2000 --lam 0.3 --players 4 > runs/train_Q1_s$i.log 2>&1 &
  sleep 120  # past the loading peak before checking memory again
done
wait
for out in "${paths[@]}"; do [ -f "$out" ] || { log "$out missing"; exit 1; }; done
[ -f models/Q1.txt ] || (IFS=+; echo "${paths[*]}") > models/Q1.txt
grep -h "best val" runs/train_Q1_s*.log

arena() {  # name, seed0, specs...
  local out=runs/eval4/q1_$1.log seed=$2
  shift 2
  if ! grep -q " = " "$out" 2>/dev/null; then
    OMP_NUM_THREADS=1 .venv/bin/python -m cbot.arena "$@" --games 1000 --procs 31 --seed0 $seed \
      --log runs/evalset.jsonl > "$out" 2>&1
  fi
  log "$1: $(tail -n 1 "$out" | sed 's/.*: \([0-9]*\/[0-9]* = .*\)/\1/')"
}
arena "${bv}_P4a_vsAB" 320000000 "$G" AB AB AB
nb=""; nw=-1
for p in P4a P4b; do
  [ "$p" = P4a ] && [ "$bv" = Q1 ] && continue
  spec="NMCTS:$(cat models/Q1.txt):initf=2:sims=400:cpuct=$bc:policy=models/$p.npz"
  arena "Q1_${p}_vsAB" 320000000 "$spec" AB AB AB
  w=$(wins runs/eval4/q1_Q1_${p}_vsAB.log)
  [ "$w" -gt "$nw" ] && { nw=$w; nb=$spec; nn=Q1_$p; }
done
spec="NMCTS:$(cat models/$bv.txt):initf=2:sims=400:cpuct=$bc:policy=models/P4b.npz"
arena "${bv}_P4b_vsAB" 320000000 "$spec" AB AB AB
w=$(wins runs/eval4/q1_${bv}_P4b_vsAB.log)
[ "$w" -gt "$nw" ] && { nw=$w; nb=$spec; nn=${bv}_P4b; }
log "best new: $nn ($nw/1000)"
arena "${nn}_vs_3x_q0best" 330000000 "$nb" "$G" "$G" "$G"
log "round q1 done"
