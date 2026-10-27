# Helpers for the round scripts (sourced): logging, memory waits, arenas and
# the deadline. $DEADLINE (default 07:00) is the next such time from now.
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
memfree() { awk '/MemAvailable/{print int($2 / 1048576)}' /proc/meminfo; }
wait_mem() {  # GB; gives up after 2 h
  local t=0
  while [ "$(memfree)" -lt "$1" ]; do
    [ "$t" -ge 120 ] && { log "only $(memfree) GB free for 2 h, stopping"; exit 1; }
    sleep 60; t=$((t + 1))
  done
}
# Waits for that much free memory, first for running jobs to finish if they hold it.
fit() { while [ "$(memfree)" -lt "$1" ] && [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n; done; wait_mem "$1"; }
wins() { tail -n 1 "$1" 2>/dev/null | sed -n 's/.*: \([0-9]*\)\/[0-9]* = .*/\1/p'; }
deadline() {
  local e
  e=$(date -d "${DEADLINE:-07:00}" +%s)
  [ "$e" -gt "$(date +%s)" ] || e=$(date -d "tomorrow ${DEADLINE:-07:00}" +%s)
  echo "$e"
}
END=$(deadline)
# arena log seed0 specs...: 1000 games, skipped if done or if it would end past
# the deadline ($ARENA_S seconds per arena, default 300) unless LATE=1.
arena() {
  local out=$1 seed=$2
  shift 2
  if ! grep -q " = " "$out" 2>/dev/null; then
    [ "${LATE:-0}" = 1 ] || [ $(( $(date +%s) + ${ARENA_S:-300} )) -le "$END" ] || { log "$(basename "$out" .log): skipped, past the deadline"; return; }
    OMP_NUM_THREADS=1 .venv/bin/python -m cbot.arena "$@" --games 1000 --procs 31 --seed0 "$seed" \
      --log runs/evalset.jsonl > "$out" 2>&1
  fi
  log "$(basename "$out" .log): $(tail -n 1 "$out" | sed 's/.*: \([0-9]*\/[0-9]* = [0-9.]* \[[-0-9., ]*\]\).*/\1/')"
}
# Value trainers (5, 512-256, 3 epochs, 4 threads each) and a policy trainer
# (12 threads) side by side, each started when its memory (from the rows) is
# free: train_round <value name> <policy name> <players> <lam> <dirs...>
train_round() {
  local v=$1 p=$2 n=$3 lam=$4
  shift 4
  local rows dec need_v need_p
  read -r rows dec < <(.venv/bin/python -c "
import glob, sys, numpy as np
r = d = 0
for a in sys.argv[1:]:
    for f in glob.glob(a + '/shard*.npz'):
        z = np.load(f); r += len(z['y']); d += len(np.unique(z['pr']))
print(r, d)" "$@")
  need_v=$(awk -v r="$rows" -v k=$((31 + 37 * n)) 'BEGIN{print int(r * k * 2 * 1.1 / 1e9) + 2}')  # k features
  need_p=$(awk -v n="$dec" 'BEGIN{print int(n * 351 * 4 * 3.0 / 1e9) + 3}')
  if [ "$p" = - ]; then
    log "training $v: $rows rows; $need_v GB per value trainer"
  else
    log "training $p and $v: $rows rows, $dec decisions; $need_v GB per value trainer, $need_p GB for the policy"
  fi
  if [ "$p" != - ] && [ ! -f "models/$p.npz" ]; then
    fit "$need_p"
    .venv/bin/python -m cbot.train_policy --data "$@" --out "models/$p.npz" --threads 12 > "runs/train_$p.log" 2>&1 &
    sleep 120
  fi
  local paths=() i out
  for i in 1 2 3 4 5; do
    out=models/${v}_s$i.npz
    paths+=("$out")
    [ -f "$out" ] && continue
    fit "$need_v"
    .venv/bin/python -m cbot.train --data "$@" --out "$out" --epochs 3 --threads 4 --hidden 512,256 \
      --cols compact --eval-every 2000 --lam "$lam" --players "$n" > "runs/train_${v}_s$i.log" 2>&1 &
    sleep 120  # past the loading peak before checking memory again
  done
  wait
  [ "$p" = - ] || [ -f "models/$p.npz" ] || { log "models/$p.npz missing"; exit 1; }
  for out in "${paths[@]}"; do [ -f "$out" ] || { log "$out missing"; exit 1; }; done
  [ -f "models/$v.txt" ] || (IFS=+; echo "${paths[*]}") > "models/$v.txt"
  [ "$p" = - ] || grep -h "best val" "runs/train_$p.log"
  grep -h "best val" runs/train_"${v}"_s*.log
}
