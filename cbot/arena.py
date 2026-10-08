"""Parallel 1v1 arena with Wilson confidence intervals.

Usage:
    python -m cbot.arena A B --games 1000 --procs 30

Player specs are resolved by cbot.players.make_player (e.g. AB, F, ABfix, NN:path).
Each seed is played twice with the seat order swapped, so neither bot gets the
first-move advantage more often.
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


def play_one(args):
    spec_a, spec_b, seed, a_first = args
    from cbot.players import make_player

    a = make_player(spec_a, Color.RED)
    b = make_player(spec_b, Color.BLUE)
    # Seating is rng.sample(players) with the game seed, so swapping the input
    # order under the same seed swaps the seats on an identical map.
    game = Game([a, b] if a_first else [b, a], seed=seed)
    a_first = game.state.colors[0] == Color.RED
    times = {Color.RED: [0.0, 0], Color.BLUE: [0.0, 0]}

    def decide(player, g, actions):
        t = time.perf_counter()
        act = player.decide(g, actions)
        rec = times[player.color]
        rec[0] += time.perf_counter() - t
        rec[1] += 1
        return act

    winner = game.play(decide_fn=decide)
    return {
        "a_won": winner == Color.RED,
        "b_won": winner == Color.BLUE,
        "a_first": a_first,
        "turns": game.state.num_turns,
        "a_time": times[Color.RED],
        "b_time": times[Color.BLUE],
    }


def run(spec_a, spec_b, games, procs, seed0=0, quiet=False):
    jobs = []
    for i in range(games // 2):
        jobs.append((spec_a, spec_b, seed0 + i, True))
        jobs.append((spec_a, spec_b, seed0 + i, False))
    res = []
    t0 = time.time()
    with mp.Pool(procs) as pool:
        for r in pool.imap_unordered(play_one, jobs, chunksize=1):
            res.append(r)
            if not quiet and len(res) % 200 == 0:
                k = sum(x["a_won"] for x in res)
                print(f"  {len(res)}/{len(jobs)} {spec_a} wins {k / len(res):.3f}", flush=True)
    return summarize(spec_a, spec_b, res, time.time() - t0)


def summarize(spec_a, spec_b, res, elapsed):
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
        "a_ms_per_decision": at * 1000,
        "b_ms_per_decision": bt * 1000,
        "avg_turns": sum(r["turns"] for r in res) / n,
        "elapsed_s": elapsed,
    }
    print(
        f"{spec_a} vs {spec_b}: {k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}] "
        f"(first {out['a_first_winrate']:.3f}, second {out['a_second_winrate']:.3f}, "
        f"draws {draws}) ms/dec {out['a_ms_per_decision']:.1f} vs {out['b_ms_per_decision']:.1f} "
        f"[{elapsed:.0f}s]",
        flush=True,
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=10_000_000)
    ap.add_argument("--log", default="results.jsonl")
    args = ap.parse_args()
    out = run(args.a, args.b, args.games, args.procs, args.seed0)
    import json

    with open(args.log, "a") as f:
        f.write(json.dumps(out) + "\n")
