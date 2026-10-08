#!/bin/bash
# Final evaluation of a bot spec on fresh seeds (never used during development).
#   cbot/final_eval.sh "NN:<ensemble>:initf=2" [procs]
set -e
cd "$(dirname "$0")/.."
BOT=$1; P=${2:-30}
LOG=runs/final.jsonl
for opp_games in "AB 2000" "ABfix 1000" "F 1000" "AB:3 400"; do
  set -- $opp_games
  .venv/bin/python -m cbot.arena "$BOT" "$1" --games "$2" --procs "$P" --log "$LOG" --seed0 50000000 | tail -n 1
done
