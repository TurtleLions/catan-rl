#!/bin/bash
# Overnight: round q2 (4 players, $Q2_MIN minutes of self-play), then round t2
# (3 players, self-play sized to end by $DEADLINE, default 07:00). Training and
# arenas always finish, past the deadline if they must (LATE=1).
cd "$(dirname "$0")/.."
export LATE=1
cbot/round_q2.sh >> runs/round_q2.log 2>&1 || echo "[$(date '+%m-%d %H:%M')] round q2 failed" >> runs/round_q2.log
cbot/round_t2.sh >> runs/round_t2.log 2>&1 || echo "[$(date '+%m-%d %H:%M')] round t2 failed" >> runs/round_t2.log
echo "[$(date '+%m-%d %H:%M')] night done" >> runs/round_t2.log
