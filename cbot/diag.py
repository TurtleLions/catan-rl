"""Diagnostic: production advantage right after the opening vs game result."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
import sys

import numpy as np
from catanatron import Color, Game
from catanatron.features import build_production_features

PROD = build_production_features(True)


def one(args):
    spec_a, spec_b, seed, a_first = args
    from cbot.players import make_player

    a, b = make_player(spec_a, Color.RED), make_player(spec_b, Color.BLUE)
    g = Game([a, b] if a_first else [b, a], seed=seed)
    while g.state.is_initial_build_phase:
        g.play_tick()
    f = PROD(g, Color.RED)
    mine = sum(v for k, v in f.items() if "_P0_" in k)
    theirs = sum(v for k, v in f.items() if "_P1_" in k)
    w = g.play()
    return mine - theirs, w == Color.RED, g.state.num_turns


if __name__ == "__main__":
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    jobs = [(a, b, 40_000_000 + i, f) for i in range(n // 2) for f in (True, False)]
    with mp.Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 16) as p:
        res = p.map(one, jobs, chunksize=2)
    d = np.array([r[0] for r in res]); w = np.array([r[1] for r in res])
    print(f"overall {w.mean():.3f}  mean prod adv {d.mean():+.4f}  P(adv>0) {(d > 0).mean():.3f}")
    for lo, hi in [(-1, -0.05), (-0.05, -0.02), (-0.02, 0.02), (0.02, 0.05), (0.05, 1)]:
        m = (d > lo) & (d <= hi)
        if m.sum():
            print(f"  adv in ({lo:+.2f},{hi:+.2f}]: n={m.sum():4d} win {w[m].mean():.3f}")
