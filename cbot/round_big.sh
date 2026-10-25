#!/bin/bash
# Bigger value nets on all native self-play data (n0 + n1, ~50M positions), ~3 h:
#   B5  = 4 nets 512-256       (~25 min each on 16 threads),
#   B10 = 4 nets 1024-512-256  (~50 min each),
# lam .3, 3 epochs, then each committee with P1 in the native search vs AB on
# unused seeds 150M (4000 games, N1 as the baseline on the same seeds), and the
# better one head-to-head vs N1 + P1 (the current bot) on seeds 160M.
# A trainer holds ~11 GB; trainers start only when that much memory is free
# (the machine is shared), at most two at a time. Steps whose output exists are
# skipped, so a rerun continues.
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
memfree() { awk '/MemAvailable/{print int($2 / 1048576)}' /proc/meminfo; }
DATA="data/n0 data/n1"
GB=13; PAR=2; TH=16
S="sims=400:cpuct=0.3"
N1=$(cat models/N1.txt)

train() {  # name, number of nets, hidden
  local name=$1 n=$2 hidden=$3 paths=() i out waited=0
  for i in $(seq 1 "$n"); do
    out=models/${name}_s$i.npz
    paths+=("$out")
    [ -f "$out" ] && continue
    # Wait for a free slot and the memory; with nothing running, give up after 2 h.
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ] || [ "$(memfree)" -lt "$GB" ]; do
      if [ "$(jobs -rp | wc -l)" -gt 0 ]; then wait -n; continue; fi
      if [ "$waited" -ge 120 ]; then log "$name: only $(memfree) GB free for 2 h, stopping"; exit 1; fi
      sleep 60; waited=$((waited + 1))
    done
    log "$name: net $i ($(memfree) GB free)"
    .venv/bin/python -m cbot.train --data $DATA --out "$out" --epochs 3 --threads "$TH" --hidden "$hidden" \
      --cols compact --eval-every 2000 --lam 0.3 > runs/train_${name}_s$i.log 2>&1 &
    sleep 120  # past the loading peak before checking memory again
  done
  wait
  for out in "${paths[@]}"; do [ -f "$out" ] || { log "$name: $out missing"; exit 1; }; done
  (IFS=+; echo "${paths[*]}") > models/$name.txt
  grep -h "best val" runs/train_${name}_s*.log
}
[ -f models/B5.txt ] || train B5 4 512,256
[ -f models/B10.txt ] || train B10 4 1024,512,256

arena() {  # name, spec A, spec B, seed0
  local out=runs/evalset/big_$1.log
  if ! grep -q " = " "$out" 2>/dev/null; then
    OMP_NUM_THREADS=1 .venv/bin/python -m cbot.arena "$2" "$3" --games 4000 --procs 31 --seed0 $4 \
      --log runs/evalset.jsonl > "$out" 2>&1
  fi
  log "$1: $(tail -n 1 "$out" | sed 's/.*: \([0-9]*\/[0-9]* = .*\)/\1/')"
}
best=""; bestw=0
for v in N1 B5 B10; do
  spec="NMCTS:$(cat models/$v.txt):initf=2:$S:policy=models/P1.npz"
  arena "${v}_P1_vsAB" "$spec" AB 150000000
  w=$(tail -n 1 runs/evalset/big_${v}_P1_vsAB.log | sed 's/.*: \([0-9]*\)\/.*/\1/')
  if [ "$v" != N1 ] && [ "$w" -gt "$bestw" ]; then best=$spec; bestw=$w; bestname=${v}_P1; fi
done
log "best new: $bestname ($bestw/4000)"
arena "${bestname}_vs_N1_P1" "$best" "NMCTS:$N1:initf=2:$S:policy=models/P1.npz" 160000000
log "big nets done"
