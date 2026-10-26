#!/bin/bash
# 4-player native self-play round q1, after q0 (cbot/round_q0.sh), done by
# $DEADLINE (default 07:00): the best NMCTS combination of q0 by wins vs three
# AlphaBetas plays against itself (root noise .25, p-record .25) for as long
# as leaves time for training and arenas (from q0's wall times in
# runs/q0_times.txt), then on q0 + q1 together
#   P4b = policy on the visit counts,
#   Q1  = 5 value nets (512-256, lam .3, 3 epochs),
# side by side; each new combination vs three AlphaBetas on unused seeds 320M
# (1000 games; q0's best on the same seeds), and the better one vs three
# copies of q0's best on seeds 330M (1000 games, seats rotated: 25% = equal
# strength). An arena that would end past the deadline is skipped. The best
# bot (value committee, policy, cpuct) goes to runs/q_best.txt.
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

printf '%s\n' "$bv" models/P4a.npz "$bc" > runs/q_best.txt

# Self-play time g from the time left R: training scales with the rows (q0's
# 50k games plus q1's r*g), so R = g + Tt*(1 + r*g/50000) + 5 arenas + 20 min.
end=$(date -d "${DEADLINE:-07:00}" +%s)
read -r _ Tt < <(grep train runs/q0_times.txt)
read -r _ Ta < <(grep arena runs/q0_times.txt)
r=$(awk '/games\/h/{x=$4} END{print x / 3600 * 0.85}' runs/gen_q0.log)  # games/s; a policy slows it a little
R=$(( end - $(date +%s) - 1200 ))
g=$(awk -v R=$R -v T=$Tt -v A=$Ta -v r=$r 'BEGIN{print int((R - T - 5 * A) / (1 + T * r / 50000))}')
log "time left ${R}s; q0 training ${Tt}s, arena ${Ta}s; self-play ${g}s at ${r} games/s"
[ "$g" -ge 1800 ] || { log "under 30 min of self-play would fit, stopping"; exit 1; }

log "generating q1"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/q1 --games 300000 --seed0 500000000 --threads 32 --players 4 \
  --model "$(cat models/$bv.txt)" --policy models/P4a.npz --search sims=400,cpuct=$bc,noise=0.25 --p-record 0.25 \
  --minutes $(( g / 60 )) >> runs/gen_q1.log 2>&1
log "$(tail -n 1 runs/gen_q1.log)"

# Memory per trainer from the rows: a value trainer holds X (float16, 179
# columns) and its copies, the policy trainer the dense visit targets (351
# float32 per decision) twice; one that does not fit waits for a running one.
read -r rows dec < <(.venv/bin/python -c "
import glob, numpy as np
r = d = 0
for f in glob.glob('data/q0/shard*.npz') + glob.glob('data/q1/shard*.npz'):
    z = np.load(f); r += len(z['y']); d += len(np.unique(z['pr']))
print(r, d)")
need_v=$(awk -v n=$rows -v f=${VMEM:-1.3} 'BEGIN{print int(n * 179 * 2 * f / 1e9) + 2}')
need_p=$(awk -v n=$dec -v f=${PMEM:-2.5} 'BEGIN{print int(n * 351 * 4 * f / 1e9) + 3}')
log "training P4b and Q1: $rows rows, $dec decisions; $need_v GB per value trainer, $need_p GB for the policy"
fit() { while [ "$(memfree)" -lt "$1" ] && [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n; done; wait_mem "$1"; }
if [ ! -f models/P4b.npz ]; then
  fit "$need_p"
  .venv/bin/python -m cbot.train_policy --data data/q0 data/q1 --out models/P4b.npz --threads 12 \
    > runs/train_P4b.log 2>&1 &
  sleep 120
fi
paths=()
for i in 1 2 3 4 5; do
  out=models/Q1_s$i.npz
  paths+=("$out")
  [ -f "$out" ] && continue
  fit "$need_v"
  .venv/bin/python -m cbot.train --data data/q0 data/q1 --out "$out" --epochs 3 --threads 4 --hidden 512,256 \
    --cols compact --eval-every 2000 --lam 0.3 --players 4 > runs/train_Q1_s$i.log 2>&1 &
  sleep 120  # past the loading peak before checking memory again
done
wait
[ -f models/P4b.npz ] || { log "models/P4b.npz missing"; exit 1; }
for out in "${paths[@]}"; do [ -f "$out" ] || { log "$out missing"; exit 1; }; done
[ -f models/Q1.txt ] || (IFS=+; echo "${paths[*]}") > models/Q1.txt
grep -h "best val" runs/train_P4b.log
grep -h "best val" runs/train_Q1_s*.log

arena() {  # name, seed0, specs...
  local out=runs/eval4/q1_$1.log seed=$2
  shift 2
  if ! grep -q " = " "$out" 2>/dev/null; then
    [ $(( $(date +%s) + Ta )) -le "$end" ] || { log "$1: skipped, past the deadline"; return; }
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
  [ -n "$w" ] && [ "$w" -gt "$nw" ] && { nw=$w; nb=$spec; nn=Q1_$p; nv=Q1; np=models/$p.npz; }
done
spec="NMCTS:$(cat models/$bv.txt):initf=2:sims=400:cpuct=$bc:policy=models/P4b.npz"
arena "${bv}_P4b_vsAB" 320000000 "$spec" AB AB AB
w=$(wins runs/eval4/q1_${bv}_P4b_vsAB.log)
[ -n "$w" ] && [ "$w" -gt "$nw" ] && { nw=$w; nb=$spec; nn=${bv}_P4b; nv=$bv; np=models/P4b.npz; }
[ "$nw" -ge 0 ] || { log "no q1 results"; exit 1; }
log "best new: $nn ($nw/1000)"
arena "${nn}_vs_3x_q0best" 330000000 "$nb" "$G" "$G" "$G"
w=$(wins runs/eval4/q1_${nn}_vs_3x_q0best.log)
[ -n "$w" ] && [ "$w" -gt 250 ] && printf '%s\n' "$nv" "$np" "$bc" > runs/q_best.txt
log "round q1 done; best bot $(paste -sd' ' runs/q_best.txt)"
