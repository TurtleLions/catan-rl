#!/bin/bash
# One-time environment setup on a new machine, run from anywhere in the repo:
#   cbot/setup_server.sh [python]      # python defaults to python3; needs 3.12+
#
# Creates .venv with Catanatron (vendor/, pinned to the commit all results used),
# numpy and CPU-only torch, then runs smoke tests: the cbot.fast equivalence
# check (games must match stock Catanatron move for move) and a short 4-player
# arena that reports per-core speed.
set -e
cd "$(dirname "$0")/.."
PY=${1:-python3}
CATANATRON_COMMIT=ecf9311

"$PY" -c 'import sys; assert sys.version_info >= (3, 12), "need Python 3.12+, got " + sys.version'

if [ ! -d vendor/catanatron ]; then
  git clone https://github.com/bcollazo/catanatron vendor/catanatron
fi
git -C vendor/catanatron checkout -q "$CATANATRON_COMMIT"

if [ ! -x .venv/bin/python ]; then
  if ! "$PY" -m venv .venv; then
    echo "venv creation failed (Debian/Ubuntu: install python3-venv, or pass a different python)." >&2
    exit 1
  fi
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e vendor/catanatron
.venv/bin/python -m pip install -q numpy
.venv/bin/python -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu

echo "== versions"
.venv/bin/python -c 'import sys, numpy, torch; print(sys.version.split()[0], "numpy", numpy.__version__, "torch", torch.__version__)'
echo "cores: $(nproc)"

echo "== cbot.fast equivalence vs stock Catanatron (expect 2/2 identical)"
.venv/bin/python -m cbot.check_fast --games 2 --table AB,AB,F,F

echo "== 4-player arena smoke test (bot vs 3 AlphaBeta, 1 game per core)"
N=$(nproc)
.venv/bin/python -m cbot.arena "NN:$(cat models/M0.txt):initf=2:openk=15" AB AB AB \
  --games $(( N / 4 * 4 )) --procs "$N" --seed0 90000000 --log /dev/null
