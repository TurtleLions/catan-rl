"""Self-play data from the native engine: both seats play the native tree
search (cbot/engine), in shards like cbot.gen writes.

    python -m cbot.ngen --out data/n0 --games 20000 --threads 32 \
        --model "$(cat models/FINAL.txt)" --policy models/P1.npz \
        --search sims=400,cpuct=0.3,noise=0.25

Games run in C++ threads (no Python per move), each on its own random map with
seed seed0 + i, every seat played by the search (--players 2 to 4). Shards
hold X (float16), y, v, g and the policy triples pr, pa, pv, pn, as cbot.gen's
do (multiplayer: y the winner's seat and v the search values, both counted
from the row's seat; pa in native numbers, cbot.native.index); games that
hit the turn limit are dropped.
Resumable: rerunning skips seeds already in the output's shards. --minutes
stops early.
"""

import argparse
import os
import time

import numpy as np

from cbot import native
from cbot.gen import legacy_seeds


def parse_search(s):
    out = {}
    for kv in filter(None, s.split(",")):
        k, v = kv.split("=")
        out[k] = float(v) if "." in v else int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--search", default="sims=400,cpuct=0.3,noise=0.25")
    ap.add_argument("--p-record", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=0, help="games per call (default 4 per thread)")
    ap.add_argument("--shard-games", type=int, default=2000)
    ap.add_argument("--minutes", type=float, default=0, help="stop starting new games after this long")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    seen = legacy_seeds(a.out)
    seeds = [s for s in range(a.seed0, a.seed0 + a.games) if s not in seen]
    shard = 1 + max((int(f[5:8]) for f in os.listdir(a.out) if f.startswith("shard") and f.endswith(".npz")
                     and ".tmp" not in f), default=-1)
    if seen:
        print(f"resuming: {len(seen)} games recorded, {len(seeds)} to play", flush=True)
    nets = native.get_nets(a.model, a.policy)
    prm = parse_search(a.search)
    chunk = a.chunk or 4 * a.threads
    parts, ngames, done, dropped, t0 = [], 0, a.games - len(seeds), 0, time.time()

    def flush():
        nonlocal parts, ngames, shard
        if not parts:
            return
        offs = np.cumsum([0] + [len(p["y"]) for p in parts[:-1]])
        path = os.path.join(a.out, f"shard{shard:03d}.npz")
        np.savez(
            path + ".tmp.npz",
            X=np.concatenate([p["X"] for p in parts]),
            y=np.concatenate([p["y"] for p in parts]),
            v=np.concatenate([p["v"] for p in parts]),
            g=np.concatenate([p["g"] for p in parts]),
            pr=np.concatenate([p["pr"] + o for p, o in zip(parts, offs)]),
            pa=np.concatenate([p["pa"] for p in parts]),
            pv=np.concatenate([p["pv"] for p in parts]),
            pn=np.concatenate([p["pn"] for p in parts]),
        )
        os.replace(path + ".tmp.npz", path)
        shard += 1
        parts, ngames = [], 0

    for i in range(0, len(seeds), chunk):
        if a.minutes and time.time() - t0 > 60 * a.minutes:
            print(f"time limit: stopping at {done} games", flush=True)
            break
        batch = seeds[i : i + chunk]
        r = native._engine.selfplay(nets, prm, batch, a.threads, a.p_record, a.eps, a.players)
        r["X"] = r["X"].astype(np.float16)
        dropped += r.pop("unfinished")
        parts.append(r)
        ngames += len(batch)
        done += len(batch)
        if ngames >= a.shard_games:
            flush()
        el = time.time() - t0
        print(f"{done}/{a.games} games, {el:.0f}s, {3600 * (done - a.games + len(seeds)) / el:.0f} games/h"
              f"{f', {dropped} unfinished' if dropped else ''}", flush=True)
    flush()
    print(f"done {done} games in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
