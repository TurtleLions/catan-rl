"""Parallel arena with Wilson confidence intervals.

Usage:
    python -m cbot.arena A B --games 1000 --procs 30          # 1v1
    python -m cbot.arena A B C D --games 1000 --procs 30      # 4-player table

Player specs are resolved by cbot.players.make_player (e.g. AB, F, ABfix, NN:path).
A is the player under test; win rates are A's. Each seed is played once per
rotation of the seating (twice in 1v1, four times with four players), so A sits
in every seat on the same map.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import multiprocessing as mp
import time

from catanatron import Color, Game


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def play_one(args):
    specs, seed, rot = args
    from cbot.players import make_player

    players = [make_player(sp, c) for sp, c in zip(specs, COLORS)]
    # Seating is rng.sample(players) with the game seed, so rotating the input
    # order under the same seed rotates the seats on an identical map.
    game = Game(players[rot:] + players[:rot], seed=seed)
    times = {p.color: [0.0, 0] for p in players}

    def decide(player, g, actions):
        t = time.perf_counter()
        act = player.decide(g, actions)
        rec = times[player.color]
        rec[0] += time.perf_counter() - t
        rec[1] += 1
        return act

    winner = game.play(decide_fn=decide)
    other = [c for c in COLORS[1 : len(specs)]]
    return {
        "a_won": winner == Color.RED,
        "b_won": winner in other,
        "a_first": game.state.colors[0] == Color.RED,
        "a_seat": game.state.colors.index(Color.RED),
        "turns": game.state.num_turns,
        "a_time": times[Color.RED],
        "b_time": [sum(times[c][0] for c in other), sum(times[c][1] for c in other)],
    }


def run(specs, games, procs, seed0=0, quiet=False):
    n = len(specs)
    jobs = [(specs, seed0 + i, r) for i in range(games // n) for r in range(n)]
    spec_a = specs[0]
    res = []
    t0 = time.time()
    with mp.Pool(procs) as pool:
        for r in pool.imap_unordered(play_one, jobs, chunksize=1):
            res.append(r)
            if not quiet and len(res) % 200 == 0:
                k = sum(x["a_won"] for x in res)
                print(f"  {len(res)}/{len(jobs)} {spec_a} wins {k / len(res):.3f}", flush=True)
    return summarize(specs, res, time.time() - t0)


def summarize(specs, res, elapsed):
    spec_a, spec_b = specs[0], ",".join(specs[1:])
    n = len(res)
    k = sum(r["a_won"] for r in res)
    draws = sum(not r["a_won"] and not r["b_won"] for r in res)
    lo, hi = wilson(k, n)
    first = [r for r in res if r["a_first"]]
    second = [r for r in res if not r["a_first"]]
    at = sum(r["a_time"][0] for r in res) / max(1, sum(r["a_time"][1] for r in res))
    bt = sum(r["b_time"][0] for r in res) / max(1, sum(r["b_time"][1] for r in res))
    out = {
        "a": spec_a,
        "b": spec_b,
        "games": n,
        "a_wins": k,
        "draws": draws,
        "a_winrate": k / n,
        "ci95": (lo, hi),
        "a_first_winrate": sum(r["a_won"] for r in first) / max(1, len(first)),
        "a_second_winrate": sum(r["a_won"] for r in second) / max(1, len(second)),
        "a_seat_winrate": [
            sum(r["a_won"] for r in res if r["a_seat"] == s)
            / max(1, sum(r["a_seat"] == s for r in res))
            for s in range(len(specs))
        ],
        "a_ms_per_decision": at * 1000,
        "b_ms_per_decision": bt * 1000,
        "avg_turns": sum(r["turns"] for r in res) / n,
        "elapsed_s": elapsed,
    }
    print(
        f"{spec_a} vs {spec_b}: {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}] "
        f"(by seat {' '.join(f'{w:.3f}' for w in out['a_seat_winrate'])}, "
        f"draws {draws}) ms/dec {out['a_ms_per_decision']:.1f} vs {out['b_ms_per_decision']:.1f} "
        f"[{elapsed:.0f}s]",
        flush=True,
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=10_000_000)
    ap.add_argument("--log", default="results.jsonl")
    args = ap.parse_args()
    out = run(args.specs, args.games, args.procs, args.seed0)
    import json

    with open(args.log, "a") as f:
        f.write(json.dumps(out) + "\n")
