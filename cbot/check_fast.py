"""Check that cbot.fast leaves games unchanged, and time it.

    python -m cbot.check_fast --games 4 --table AB,AB,F,F

Plays the same seeds with stock Catanatron (CBOT_NOFAST=1) and with the
speedups, in subprocesses, and compares every action of every game.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def play(table, seeds):
    from catanatron import Color, Game

    from cbot.players import make_player

    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    out, t = [], time.perf_counter()
    for s in seeds:
        g = Game([make_player(sp, c) for sp, c in zip(table, colors)], seed=s)
        g.play()
        out.append([str(r) for r in g.state.action_records])
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--table", default="AB,AB,F,F")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    table, seeds = a.table.split(","), list(range(a.seed0, a.seed0 + a.games))
    if a.child:
        print(json.dumps(play(table, seeds)))
        return
    res = {}
    for name, extra in (("stock", {"CBOT_NOFAST": "1"}), ("fast", {})):
        # Fixed hash seed: set iteration order (and so stock games) varies per process otherwise.
        env = {**os.environ, "PYTHONHASHSEED": "0", **extra}
        cmd = [sys.executable, "-m", "cbot.check_fast", "--child", "--games", str(a.games),
               "--seed0", str(a.seed0), "--table", a.table]
        res[name] = json.loads(subprocess.run(cmd, env=env, capture_output=True, check=True, text=True).stdout)
    same = sum(x == y for x, y in zip(res["stock"][0], res["fast"][0]))
    print(f"{a.table}: identical games {same}/{a.games}; "
          f"stock {res['stock'][1]:.1f}s fast {res['fast'][1]:.1f}s "
          f"speedup {res['stock'][1] / res['fast'][1]:.2f}x")


if __name__ == "__main__":
    main()
