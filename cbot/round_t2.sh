#!/bin/bash
# 3-player round t2, after t1 (cbot/round_t1.sh), as round q1 for 4 players,
# done by $DEADLINE (default 07:00): the best combination of t1 by wins vs two
# AlphaBetas plays itself (root noise .25, p-record .25) for as long as leaves
# time for training and arenas (from t1's wall times), then on t1 + t2
#   P3b = policy on the visit counts,
#   T2  = 5 value nets (512-256, lam .3, 3 epochs),
# side by side; each new combination vs two AlphaBetas on seeds 360M (1000
# games; t1's best on the same seeds), and the better one vs two copies of
# t1's best on seeds 370M (seats rotated: 33% = equal strength). An arena that
# would end past the deadline is skipped. The best bot (value committee,
# policy, cpuct) goes to runs/t_best.txt.
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
. cbot/round_lib.sh

best=""; bw=-1
for v in T0 T1; do
  for c in 0.3 1.0; do
    w=$(wins "runs/eval3/t1_${v}_P3a_c$c.log")
    [ -n "$w" ] && [ "$w" -gt "$bw" ] && { bw=$w; bv=$v; bc=$c; }
  done
done
[ "$bw" -ge 0 ] || { log "no t1 results"; exit 1; }
G="NMCTS:$(cat models/$bv.txt):initf=2:sims=400:cpuct=$bc:policy=models/P3a.npz"
log "generator: $bv + P3a, cpuct $bc ($bw/1000 vs 2 AB in t1)"
printf '%s\n' "$bv" models/P3a.npz "$bc" > runs/t_best.txt

# Self-play time g from the time left R: training scales with the rows (t1's
# n1 games plus t2's r*g; doubled, as five value trainers may not fit at once),
# so R = g + 2Tt*(1 + r*g/n1) + 5 arenas + 20 min.
read -r _ Tt < <(grep train runs/t1_times.txt)
read -r _ Ta < <(grep arena runs/t1_times.txt)
ARENA_S=$Ta
n1=$(awk '/^done/{x=$2} END{print x}' runs/gen_t1.log)
r=$(awk '/games\/h/{x=$4} END{print x / 3600 * 0.85}' runs/gen_t1.log)  # games/s; a policy slows it a little
R=$(( END - $(date +%s) - 1200 ))
g=$(awk -v R=$R -v T=$(( 2 * Tt )) -v A="$Ta" -v r="$r" -v n="$n1" 'BEGIN{print int((R - T - 5 * A) / (1 + T * r / n))}')
log "time left ${R}s; t1 training ${Tt}s on $n1 games, arena ${Ta}s; self-play ${g}s at ${r} games/s"
[ "$g" -ge 1800 ] || { log "under 30 min of self-play would fit, stopping"; exit 1; }

log "generating t2"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/t2 --games 600000 --seed0 720000000 --threads 32 --players 3 \
  --model "$(cat models/$bv.txt)" --policy models/P3a.npz --search sims=400,cpuct=$bc,noise=0.25 --p-record 0.25 \
  --minutes $(( g / 60 )) >> runs/gen_t2.log 2>&1
log "$(tail -n 1 runs/gen_t2.log)"

train_round T2 P3b 3 0.3 data/t1 data/t2

arena "runs/eval3/t2_${bv}_P3a_vsAB.log" 360000000 "$G" AB AB
nb=""; nw=-1
for p in P3a P3b; do
  spec="NMCTS:$(cat models/T2.txt):initf=2:sims=400:cpuct=$bc:policy=models/$p.npz"
  arena "runs/eval3/t2_T2_${p}_vsAB.log" 360000000 "$spec" AB AB
  w=$(wins "runs/eval3/t2_T2_${p}_vsAB.log")
  [ -n "$w" ] && [ "$w" -gt "$nw" ] && { nw=$w; nb=$spec; nn=T2_$p; nv=T2; np=models/$p.npz; }
done
spec="NMCTS:$(cat models/$bv.txt):initf=2:sims=400:cpuct=$bc:policy=models/P3b.npz"
arena "runs/eval3/t2_${bv}_P3b_vsAB.log" 360000000 "$spec" AB AB
w=$(wins "runs/eval3/t2_${bv}_P3b_vsAB.log")
[ -n "$w" ] && [ "$w" -gt "$nw" ] && { nw=$w; nb=$spec; nn=${bv}_P3b; nv=$bv; np=models/P3b.npz; }
[ "$nw" -ge 0 ] || { log "no t2 results"; exit 1; }
log "best new: $nn ($nw/1000)"
arena "runs/eval3/t2_${nn}_vs_2x_t1best.log" 370000000 "$nb" "$G" "$G"
w=$(wins "runs/eval3/t2_${nn}_vs_2x_t1best.log")
[ -n "$w" ] && [ "$w" -gt 333 ] && printf '%s\n' "$nv" "$np" "$bc" > runs/t_best.txt
log "round t2 done; best bot $(paste -sd' ' runs/t_best.txt)"
