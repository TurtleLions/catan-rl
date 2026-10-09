"""Generate (state features, outcome) training data from self-play games.

    python -m cbot.gen --out data/r0 --games 60000 --mix "AB,F:3 AB,AB:2 F,F:1 ABfix,AB:2"
    python -m cbot.gen --out data/m0 --games 40000 --mix "AB,F,F,F:2 F,F,F,F:1"

Each recorded state is stored from every player's perspective. 1v1 shards hold
y = 1 if that player went on to win and v = the mover's search value mapped to
that player. Multiplayer shards hold y = the winner's seat relative to that
player (0 = itself) and V = the search's per-seat win probabilities, also
relative. Optional epsilon-random moves add diversity.

Resumable: rerunning with the same arguments skips games whose seeds are
already in the output's shards (shards are written atomically).
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
        out.append((tuple(pair.split(",")), int(w)))
    return out


COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def play_record(args):
    specs, seed, eps, p_record = args
    from cbot import feats
    from cbot.players import make_player

    rng = random.Random(seed * 7 + 1)
    game = Game([make_player(sp, c) for sp, c in zip(specs, COLORS)], seed=seed)
    colors = game.state.colors  # seat order
    n = len(colors)
    rows, who, sv = [], [], []

    def decide(player, g, actions):
        rec = rng.random() < p_record
        if rec:
            for c in colors:
                rows.append(feats.encode(g, c))
                who.append(colors.index(c))
                sv.append(np.full(n, np.nan))
        if len(actions) > 1 and rng.random() < eps:
            return rng.choice(actions)
        act = player.decide(g, actions)
        vec = getattr(player, "last_vec", None)
        if rec and vec is not None:
            # Per-seat search values, rotated so each row is relative to its player.
            for k in range(-n, 0):
                sv[k] = np.roll(vec, -who[k])
        return act

    winner = game.play(decide_fn=decide)
    if winner is None or not rows:
        return None
    X = np.stack(rows).astype(np.float16)
    w = colors.index(winner)
    who = np.array(who)
    V = np.array(sv, dtype=np.float32)
    if n == 2:
        y, V = (who == w).astype(np.int8), V[:, 0]
    else:
        y = ((w - who) % n).astype(np.int8)
    return X, y, V, np.full(len(y), seed, dtype=np.int64)


def make_jobs(games, seed0, mix, eps, p_record):
    """Game i: (seat specs, seed seed0 + i, eps, p_record). Drawn for every game
    from one rng, so any subset (resume, chunks) gets the same tables."""
    mix = parse_mix(mix)
    weights = [w for _, w in mix]
    epss = [float(e) for e in eps.split(",")]
    rng = random.Random(seed0)
    jobs = []
    for i in range(games):
        specs = list(rng.choices(mix, weights)[0][0])
        rng.shuffle(specs)
        jobs.append((specs, seed0 + i, rng.choice(epss), p_record))
    return jobs


def legacy_seeds(out):
    """Seeds of games already in shard*.npz files written by main()."""
    seen = set()
    for f in sorted(os.listdir(out)):
        if f.startswith("shard") and f.endswith(".npz") and ".tmp" not in f:
            seen.update(np.load(os.path.join(out, f))["g"].tolist())
    return seen


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
    have = sorted(f for f in os.listdir(args.out) if f.startswith("shard") and f.endswith(".npz") and ".tmp" not in f)
    seen = set()
    for f in list(have):
        path = os.path.join(args.out, f)
        try:
            seen.update(np.load(path)["g"].tolist())
        except Exception:  # cut off mid-write by a pre-resume version; replay its games
            os.replace(path, path + ".bad")
            have.remove(f)
    jobs = make_jobs(args.games, args.seed0, args.mix, args.eps, args.p_record)
    jobs = [j for j in jobs if j[1] not in seen]
    if seen:
        print(f"resuming: {len(have)} shards, {len(seen)} games recorded, {len(jobs)} to play", flush=True)

    Xs, ys, vs, gs, shard, done = [], [], [], [], 1 + max((int(f[5:8]) for f in have), default=-1), args.games - len(jobs)
    t0 = time.time()

    def flush():
        nonlocal Xs, ys, vs, gs, shard
        if Xs:
            path = os.path.join(args.out, f"shard{shard:03d}.npz")
            np.savez(
                path + ".tmp.npz",
                X=np.concatenate(Xs),
                y=np.concatenate(ys),
                v=np.concatenate(vs),
                g=np.concatenate(gs),
            )
            os.replace(path + ".tmp.npz", path)
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
            if len(Xs) >= args.shard_games or done == args.games:
                flush()
                print(f"{done}/{args.games} games, {time.time() - t0:.0f}s", flush=True)
    flush()
    print(f"done {done} games in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
