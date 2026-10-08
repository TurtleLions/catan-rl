"""How often does the opening shortlist (openk) keep the full method's choice?

    python -m cbot.check_openk MODEL --positions 200 --ks 5,10,15 --players 4

Samples opening positions (ValueFunction players, random stopping point) and
compares NNSearchPlayer's opening choice with and without the shortlist.
"""

import argparse
import multiprocessing as mp
import random
import time


def one(args):
    model, seed, ks, n = args
    from catanatron import Color, Game

    from cbot.players import make_player
    from cbot.search import NNSearchPlayer

    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE][:n]
    rng = random.Random(seed)
    g = Game([make_player("F", c) for c in colors], seed=seed)
    stop = rng.randrange(2 * n)  # settlement placements before ours
    placed = 0
    while True:
        if g.state.current_prompt.name == "BUILD_INITIAL_SETTLEMENT":
            if placed == stop:
                break
            placed += 1
        g.play_tick()
    me = g.state.current_color()
    out = {}
    for k in [0] + ks:
        p = NNSearchPlayer(me, model, initf=2, openk=k)
        t = time.perf_counter()
        a = p.decide(g, g.playable_actions)
        out[k] = (a, time.perf_counter() - t)
    return {k: (out[k][0] == out[0][0], out[k][1]) for k in out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--ks", default="5,10,15")
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]
    jobs = [(a.model, 7_000_000 + i, ks, a.players) for i in range(a.positions)]
    with mp.Pool(a.procs) as pool:
        res = pool.map(one, jobs, chunksize=1)
    for k in [0] + ks:
        agree = sum(r[k][0] for r in res) / len(res)
        t = sum(r[k][1] for r in res) / len(res)
        print(f"openk={k or 'all'}: same choice {agree:.3f}  {t * 1000:.0f} ms/decision")


if __name__ == "__main__":
    main()
