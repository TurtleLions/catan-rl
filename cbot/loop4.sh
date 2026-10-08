#!/bin/bash
# 4-player expert iteration: generate -> train -> evaluate, round after round.
#   cbot/loop4.sh
#
# Settings come from cbot/loop4.conf, overridden by runs/loop4.conf if present.
# runs/loop4.conf is re-read at the start of every round, so a running (or
# requeued) job picks up edits to it, e.g. LAST_ROUND or a self-play MIX.
# Env: P (workers/threads, default nproc).
#
# Round r writes data/m<r>s, trains models/M<r> (3 nets at lam .5 + 3 at lam .3)
# on round 0 plus the last two rounds, and evaluates it vs three AlphaBetas on
# the fixed development seeds (20M), plus optionally head-to-head vs the current
# best (seeds 30M). See loop4.conf for how the next generator is chosen.
# State: models/BEST4.txt (ensemble spec), runs/best4_rate.txt, runs/loop4.jsonl,
# runs/loop4.log.
#
# Resumable at any point (for Slurm preemption + requeue): the round to run is
# the one after the last round with a gate line in runs/loop4.log; generation
# resumes from its shards, training keeps finished nets, and a finished eval is
# read back from runs/eval_M<r>.rate.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-$(nproc)}
PY=.venv/bin/python
export PLAYERS=4

[ -f models/BEST4.txt ] || cp models/M0.txt models/BEST4.txt
[ -f runs/best4_rate.txt ] || echo 0.629 > runs/best4_rate.txt  # M0, full opening, seeds 20M

log() { echo "[$(date '+%F %T')] $*" | tee -a runs/loop4.log; }

# Gate lines: "round <r>: M<r> generates next" / "... more than 3 points below" (older
# runs) and "round <r>: complete" (this version).
next_round() {
  local last
  last=$(grep -oE 'round [0-9]+: (M[0-9]+ generates next|M[0-9]+ more than|complete)' runs/loop4.log 2>/dev/null \
    | awk '{sub(":", "", $2); print $2}' | sort -n | tail -n 1)
  echo $(( ${last:-0} + 1 ))
}

# Win rate of the first spec from an arena run; appends to runs/loop4.jsonl.
arena_rate() {
  local seed0=$1 games=$2 logf=$3; shift 3
  $PY -m cbot.arena "$@" --games "$games" --procs "$P" --seed0 "$seed0" --log runs/loop4.jsonl > "$logf" 2>&1
  $PY -c 'import json; print(json.loads(open("runs/loop4.jsonl").readlines()[-1])["a_winrate"])'
}

while :; do
  source cbot/loop4.conf
  if [ -f runs/loop4.conf ]; then source runs/loop4.conf; fi
  r=$(next_round)
  if [ "$r" -gt "$LAST_ROUND" ]; then
    log "loop done (next round $r > LAST_ROUND $LAST_ROUND)"
    break
  fi
  best=$(cat models/BEST4.txt)
  best_rate=$(cat runs/best4_rate.txt)
  out=data/m${r}s

  # 1. Generate with the current best bot (opening shortlist for speed).
  if [ ! -f "$out/DONE" ]; then
    B="NN:$best:initf=2:openk=15"
    log "round $r: generating $GAMES games into $out with $best, mix $MIX"
    $PY -m cbot.gen --out "$out" --games "$GAMES" --procs "$P" --p-record 0.2 \
      --seed0 $((100000000 + r * 1000000)) --eps 0,0.03 --shard-games 1000 \
      --mix "${MIX//BEST/$B}" >> "runs/gen_m${r}s.log" 2>&1
    touch "$out/DONE"
  fi

  # 2. Train on round 0 plus rounds r-1 and r (all data dirs for those rounds).
  if [ ! -f "models/M$r.txt" ]; then
    dirs=(data/m0 data/m0b)
    for k in $((r - 1)) "$r"; do
      if [ "$k" -lt 1 ]; then continue; fi
      for d in data/m${k} data/m${k}b data/m${k}s; do
        if [ -d "$d" ]; then dirs+=("$d"); fi
      done
    done
    log "round $r: training M$r on ${dirs[*]}"
    T=$(( P / 6 ))
    STAGGER=60 cbot/train_ens.sh "M${r}a" 3 0.5 "$T" "${dirs[@]}" > "runs/train_M${r}a.log" 2>&1 &
    sleep 30
    STAGGER=60 cbot/train_ens.sh "M${r}b" 3 0.3 "$T" "${dirs[@]}" > "runs/train_M${r}b.log" 2>&1 &
    wait
    echo "$(cat "models/M${r}a.txt")+$(cat "models/M${r}b.txt")" > "models/M$r.txt"
    log "round $r: val losses $(grep -h 'best val' runs/train_M${r}a_s*.log runs/train_M${r}b_s*.log | awk '{print $3}' | tr '\n' ' ')"
  fi

  # 3. Evaluate vs three AlphaBetas (full opening, development seeds) and,
  #    if H2H_GAMES > 0, vs three copies of the current best (25% = equal).
  cand=$(cat "models/M$r.txt")
  if [ ! -f "runs/eval_M$r.rate" ]; then
    rate=$(arena_rate 20000000 "$EVAL_GAMES" "runs/eval_M$r.log" "NN:$cand:initf=2" AB AB AB)
    echo "$rate" > "runs/eval_M$r.rate"
    log "round $r: M$r vs 3AB = $rate (best so far $best_rate) | $(tail -n 1 "runs/eval_M$r.log")"
  fi
  rate=$(cat "runs/eval_M$r.rate")
  h2h=""
  if [ "$H2H_GAMES" -gt 0 ]; then
    if [ ! -f "runs/h2h_M$r.rate" ]; then
      bb="NN:$best:initf=2"
      h=$(arena_rate 30000000 "$H2H_GAMES" "runs/h2h_M$r.log" "NN:$cand:initf=2" "$bb" "$bb" "$bb")
      echo "$h" > "runs/h2h_M$r.rate"
      log "round $r: M$r vs 3x best ($best) = $h | $(tail -n 1 "runs/h2h_M$r.log")"
    fi
    h2h=$(cat "runs/h2h_M$r.rate")
  fi

  # 4. Next round's generator.
  if [ "$GATE" = h2h ] && [ -n "$h2h" ]; then
    ok=$($PY -c "print(int($h2h >= $H2H_MIN))")
    why="h2h $h2h vs min $H2H_MIN"
  else
    ok=$($PY -c "print(int($rate >= $best_rate - 0.03))")
    why="vs 3AB $rate vs best $best_rate - 0.03"
  fi
  if [ "$ok" = 1 ]; then
    echo "$cand" > models/BEST4.txt
    $PY -c "print(max($rate, $best_rate))" > runs/best4_rate.txt
    log "round $r: M$r adopted as generator ($why)"
  else
    log "round $r: M$r rejected, keeping $best ($why)"
  fi
  log "round $r: complete"
done
