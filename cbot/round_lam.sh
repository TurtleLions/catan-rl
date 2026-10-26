#!/bin/bash
# Lean on the search's value: N1's recipe (5 nets 256-128, 3 epochs, data/n1)
# with lower lam (target = lam * outcome + (1 - lam) * search value):
#   L15 = lam .15 (all rows; rows without a search value keep the outcome),
#   L0  = lam 0, searched rows only (58% of n1),
# then each with P1 in the native search vs AB on unused seeds 170M (4000
# games, N1 = lam .3 as the baseline on the same seeds), and the better one
# head-to-head vs N1 + P1 on seeds 180M. 10 trainers x 3 threads, ~6 GB each.
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
S="sims=400:cpuct=0.3"
N1=$(cat models/N1.txt)

train() {  # name, lam, extra trainer flags
  local name=$1 lam=$2 paths=() i out
  for i in 1 2 3 4 5; do
    out=models/${name}_s$i.npz
    paths+=("$out")
    [ -f "$out" ] && continue
    .venv/bin/python -m cbot.train --data data/n1 --out "$out" --epochs 3 --threads 3 --hidden 256,128 \
      --cols compact --eval-every 2000 --lam "$lam" $3 > runs/train_${name}_s$i.log 2>&1 &
    sleep 30  # stagger the loading peaks
  done
}
log "training L15 and L0"
[ -f models/L15.txt ] || train L15 0.15
[ -f models/L0.txt ] || train L0 0 --searched-only
wait
for name in L15 L0; do
  [ -f models/$name.txt ] || ls models/${name}_s[1-5].npz | paste -sd+ > models/$name.txt
  log "$name: $(grep -h "best val" runs/train_${name}_s*.log | paste -sd' ')"
done

arena() {  # name, spec A, spec B, seed0
  local out=runs/evalset/lam_$1.log
  if ! grep -q " = " "$out" 2>/dev/null; then
    OMP_NUM_THREADS=1 .venv/bin/python -m cbot.arena "$2" "$3" --games 4000 --procs 31 --seed0 $4 \
      --log runs/evalset.jsonl > "$out" 2>&1
  fi
  log "$1: $(tail -n 1 "$out" | sed 's/.*: \([0-9]*\/[0-9]* = .*\)/\1/')"
}
best=""; bestw=0
for v in N1 L15 L0; do
  spec="NMCTS:$(cat models/$v.txt):initf=2:$S:policy=models/P1.npz"
  arena "${v}_P1_vsAB" "$spec" AB 170000000
  w=$(tail -n 1 runs/evalset/lam_${v}_P1_vsAB.log | sed 's/.*: \([0-9]*\)\/.*/\1/')
  if [ "$v" != N1 ] && [ "$w" -gt "$bestw" ]; then best=$spec; bestw=$w; bestname=${v}_P1; fi
done
log "best new: $bestname ($bestw/4000)"
arena "${bestname}_vs_N1_P1" "$best" "NMCTS:$N1:initf=2:$S:policy=models/P1.npz" 180000000
log "lam test done"
