"""Generate (state features, outcome) training data from 1v1 games.

    python -m cbot.gen --out data/r0 --games 60000 --mix "AB,F:3 AB,AB:2 F,F:1 ABfix,AB:2"

Each recorded state is stored from both players' perspectives with label
1 if that player went on to win. Optional epsilon-random moves add diversity.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import random
import time

import numpy as np

from catanatron import Color, Game


def parse_mix(mix):
    out = []
    for item in mix.split():
        pair, w = item.rsplit(":", 1) if item.count(":") and item.rsplit(":", 1)[1].isdigit() else (item, "1")
        a, b = pair.split(",")
        out.append((a, b, int(w)))
    return out


def play_record(args):
    spec_a, spec_b, seed, eps, p_record = args
    from cbot import feats
    from cbot.players import make_player

    rng = random.Random(seed * 7 + 1)
    a = make_player(spec_a, Color.RED)
    b = make_player(spec_b, Color.BLUE)
    game = Game([a, b], seed=seed)
    rows, who, sv = [], [], []

    def decide(player, g, actions):
        rec = rng.random() < p_record
        if rec:
            for c in (Color.RED, Color.BLUE):
                rows.append(feats.encode(g, c))
                who.append(c)
                sv.append(np.nan)
        if len(actions) > 1 and rng.random() < eps:
            return rng.choice(actions)
        act = player.decide(g, actions)
        v = getattr(player, "last_value", None)
        if rec and v is not None:
            # Search value from the mover's view; 1v1 so the other seat gets 1-v.
            for k in (-2, -1):
                sv[k] = v if who[k] == player.color else 1.0 - v
        return act

    winner = game.play(decide_fn=decide)
    if winner is None or not rows:
        return None
    X = np.stack(rows).astype(np.float16)
    y = np.array([c == winner for c in who], dtype=np.int8)
    return X, y, np.array(sv, dtype=np.float32), np.full(len(y), seed, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=30)
    ap.add_argument("--mix", default="AB,F:3 AB,AB:2 F,F:1 ABfix,AB:2")
    ap.add_argument("--eps", default="0,0.03,0.1")
    ap.add_argument("--p-record", type=float, default=0.2)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--shard-games", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mix = parse_mix(args.mix)
    weights = [w for _, _, w in mix]
    epss = [float(e) for e in args.eps.split(",")]
    rng = random.Random(args.seed0)
    jobs = []
    for i in range(args.games):
        a, b, _ = rng.choices(mix, weights)[0]
        if rng.random() < 0.5:
            a, b = b, a
        jobs.append((a, b, args.seed0 + i, rng.choice(epss), args.p_record))

    Xs, ys, vs, gs, shard, done = [], [], [], [], 0, 0
    t0 = time.time()

    def flush():
        nonlocal Xs, ys, vs, gs, shard
        if Xs:
            np.savez(
                os.path.join(args.out, f"shard{shard:03d}.npz"),
                X=np.concatenate(Xs),
                y=np.concatenate(ys),
                v=np.concatenate(vs),
                g=np.concatenate(gs),
            )
            shard += 1
            Xs, ys, vs, gs = [], [], [], []

    with mp.Pool(args.procs) as pool:
        for r in pool.imap_unordered(play_record, jobs, chunksize=4):
            done += 1
            if r is not None:
                Xs.append(r[0])
                ys.append(r[1])
                vs.append(r[2])
                gs.append(r[3])
            if done % args.shard_games == 0:
                flush()
                print(f"{done}/{args.games} games, {time.time() - t0:.0f}s", flush=True)
    flush()
    print(f"done {done} games in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
