#!/bin/bash
# Native self-play round n0: 200k games by the native tree search (FINAL value
# committee, policy P1, 400 sims, cpuct .3, root noise .25), then
#   N0    = 5 value nets (256-128, lam .3, 3 epochs) on n0 alone,
#   N0all = 4 value nets, same recipe (4, not 5: memory), on r0-r6 + p0 + n0,
#   P3    = policy on n0's visit counts,
# each combination vs AB on unused seeds 110M (4000 games, native search), and
# the best one head-to-head vs the current bot (FINAL + P1) on seeds 120M.
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
F=$(cat models/FINAL.txt)
S="sims=400:cpuct=0.3"

log "generating n0"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/n0 --games 200000 --seed0 200000000 --threads 32 \
  --model "$F" --policy models/P1.npz --search sims=400,cpuct=0.3,noise=0.25 --p-record 0.25 >> runs/gen_n0.log 2>&1
log "$(tail -n 1 runs/gen_n0.log)"

export HIDDEN=256,128 EPOCHS=3 STAGGER=120
log "training N0 (n0 only) and P3"
[ -f models/N0.txt ] || cbot/train_ens.sh N0 5 0.3 5 data/n0 > runs/train_N0.out 2>&1 &
[ -f models/P3.npz ] || .venv/bin/python -m cbot.train_policy --data data/n0 --out models/P3.npz --threads 7 > runs/train_P3.log 2>&1 &
wait
grep -h "best val" runs/train_N0_s*.log; grep -h "best val" runs/train_P3.log
log "training N0all (r0-r6, p0, n0)"
[ -f models/N0all.txt ] || STAGGER=300 cbot/train_ens.sh N0all 4 0.3 8 data/r0 data/r1 data/r2 data/r3 data/r4 data/r5 data/r6 data/p0 data/n0 > runs/train_N0all.out 2>&1
grep -h "best val" runs/train_N0all_s*.log

arena() {  # name, spec A, spec B, seed0
  local out=runs/evalset/n0_$1.log
  if ! grep -q " = " "$out" 2>/dev/null; then
    OMP_NUM_THREADS=1 .venv/bin/python -m cbot.arena "$2" "$3" --games 4000 --procs 31 --seed0 $4 \
      --log runs/evalset.jsonl > "$out" 2>&1
  fi
  log "$1: $(tail -n 1 "$out" | sed 's/.*: \([0-9]*\/[0-9]* = .*\)/\1/')"
}
best=""; bestw=0
for v in FINAL N0 N0all; do
  for p in P1 P3; do
    [ "$v$p" = "FINALP3" ] && continue
    spec="NMCTS:$(cat models/$v.txt):initf=2:$S:policy=models/$p.npz"
    arena "${v}_${p}_vsAB" "$spec" AB 110000000
    w=$(tail -n 1 runs/evalset/n0_${v}_${p}_vsAB.log | sed 's/.*: \([0-9]*\)\/.*/\1/')
    if [ "$v" != FINAL ] && [ "$w" -gt "$bestw" ]; then best=$spec; bestw=$w; bestname=${v}_${p}; fi
  done
done
log "best new: $bestname ($bestw/4000)"
arena "${bestname}_vs_FINAL_P1" "$best" "NMCTS:$F:initf=2:$S:policy=models/P1.npz" 120000000
log "round n0 done"
