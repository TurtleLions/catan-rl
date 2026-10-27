#!/bin/bash
# 4-player round q2, after q1 (cbot/round_q1.sh): q1's best bot (runs/q_best.txt)
# plays itself into data/q2 (root noise .25, p-record .25; resumes the games
# fill_q2.sh put there) for $Q2_MIN minutes (default 150), then on q1 + q2
#   P4c = policy on the visit counts,
#   Q2  = 5 value nets (512-256, lam .3, 3 epochs),
# side by side; each new combination vs three AlphaBetas on unused seeds 480M
# (1000 games; q1's best on the same seeds), and the better one vs three
# copies of q1's best on seeds 490M (seats rotated: 25% = equal strength).
# The best bot goes to runs/q_best.txt.
# Steps whose output exists are skipped, so a rerun continues.
cd "$(dirname "$0")/.."
. cbot/round_lib.sh

{ read -r bv; read -r bp; read -r bc; } < runs/q_best.txt
G="NMCTS:$(cat models/$bv.txt):initf=2:sims=400:cpuct=$bc:policy=$bp"
log "generator: $bv + $(basename "$bp" .npz), cpuct $bc"

log "generating q2 for ${Q2_MIN:-150} min"
nice -n 5 .venv/bin/python -m cbot.ngen --out data/q2 --games 1000000 --seed0 600000000 --threads 32 --players 4 \
  --model "$(cat models/$bv.txt)" --policy "$bp" --search sims=400,cpuct=$bc,noise=0.25 --p-record 0.25 \
  --minutes "${Q2_MIN:-150}" >> runs/gen_q2.log 2>&1
log "$(tail -n 1 runs/gen_q2.log)"

train_round Q2 P4c 4 0.3 data/q1 data/q2

ARENA_S=300
arena "runs/eval4/q2_${bv}_$(basename "$bp" .npz)_vsAB.log" 480000000 "$G" AB AB AB
nb=""; nw=-1
for p in P4a P4c; do
  for v in Q2 "$bv"; do
    [ "$v" = "$bv" ] && [ "models/$p.npz" = "$bp" ] && continue  # q1's best, above
    [ "$v" = "$bv" ] && [ "$p" = P4a ] && continue  # no new net
    spec="NMCTS:$(cat models/$v.txt):initf=2:sims=400:cpuct=$bc:policy=models/$p.npz"
    arena "runs/eval4/q2_${v}_${p}_vsAB.log" 480000000 "$spec" AB AB AB
    w=$(wins "runs/eval4/q2_${v}_${p}_vsAB.log")
    [ -n "$w" ] && [ "$w" -gt "$nw" ] && { nw=$w; nb=$spec; nn=${v}_$p; nv=$v; np=models/$p.npz; }
  done
done
[ "$nw" -ge 0 ] || { log "no q2 results"; exit 1; }
log "best new: $nn ($nw/1000)"
arena "runs/eval4/q2_${nn}_vs_3x_q1best.log" 490000000 "$nb" "$G" "$G" "$G"
w=$(wins "runs/eval4/q2_${nn}_vs_3x_q1best.log")
[ -n "$w" ] && [ "$w" -gt 250 ] && printf '%s\n' "$nv" "$np" "$bc" > runs/q_best.txt
log "round q2 done; best bot $(paste -sd' ' runs/q_best.txt)"
