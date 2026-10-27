"""Compare generated data directories, to spot a round whose data looks off.

    python -m cbot.datastats data/m1s data/m2s data/m3s data/m4s

Per directory: games, duplicate games, rows per game, share of rows without a
search value, how well the generator's search values predicted the winner
(cross-entropy on rows that have them; 1.386 = no better than a coin among
four), and the input features whose mean differs most from the first
directory's (in units of the first directory's standard deviation).
"""

import argparse
import glob
import os

import numpy as np


def stats(d, sample=400_000, seed=0):
    files = sorted(f for f in glob.glob(f"{d}/*.npz") if not f.endswith(".tmp.npz"))
    g, rows, nan, xent, nv, Xs = [], 0, 0, 0.0, 0, []
    rng = np.random.default_rng(seed)
    per = max(1, sample // max(1, len(files)))
    for f in files:
        z = np.load(f)
        y, v = z["y"], z["v"]
        g.append(np.unique(z["g"]))
        rows += len(y)
        has = ~np.isnan(v if v.ndim == 1 else v[:, 0])
        nan += int((~has).sum())
        if has.any():
            if v.ndim == 2:
                p = v[has][np.arange(has.sum()), y[has]]
            else:
                p = np.where(y[has] == 1, v[has], 1 - v[has])
            xent += float(-np.log(np.clip(p, 1e-4, 1)).sum())
            nv += int(has.sum())
        X = z["X"]
        Xs.append(X[rng.choice(len(X), min(per, len(X)), replace=False)].astype(np.float32))
    g = np.concatenate(g) if g else np.array([])
    games = len(np.unique(g))
    return {
        "files": len(files),
        "games": games,
        "dup_games": len(g) - games,
        "rows": rows,
        "rows_per_game": rows / max(1, games),
        "value_xent": xent / nv if nv else float("nan"),
        "no_search_value": nan / max(1, rows),
        "X": np.concatenate(Xs) if Xs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    names = None
    try:
        from cbot.feats import feature_names
        names = feature_names(4)
    except Exception:
        pass

    ref = None
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"{d}: missing")
            continue
        s = stats(d)
        print(
            f"{d}: {s['files']} files, {s['games']} games ({s['dup_games']} duplicated), "
            f"{s['rows']} rows ({s['rows_per_game']:.0f}/game), "
            f"no search value {s['no_search_value']:.3f}, search value xent {s['value_xent']:.3f}"
        )
        X = s["X"]
        if X is None:
            continue
        if ref is None:
            ref = (X.mean(0), X.std(0) + 1e-6)
            continue
        z = (X.mean(0) - ref[0]) / ref[1]
        top = np.argsort(-np.abs(z))[: args.top]

        def label(i):
            return names[i] if names is not None and i < len(names) else f"x{i}"

        print("   most shifted vs first: " + ", ".join(f"{label(i)} {z[i]:+.2f}" for i in top))


if __name__ == "__main__":
    main()
