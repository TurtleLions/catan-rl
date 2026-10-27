"""Check the native engine against Catanatron, move by move.

    python -m cbot.check_native --games 200 --table R,R
    python -m cbot.check_native --games 50 --table AB,F --procs 32
    python -m cbot.check_native --games 50 --table AB,F,R,F --procs 32

Plays Catanatron games (2 to 4 players) and, at every decision, converts the
position to the native engine and compares
  * the legal moves (as native numbers, cbot.native.index),
  * every seat's feature vector with cbot.feats.encode,
  * base_fn (the heuristic behind ValueFunctionPlayer and AlphaBeta),
  * the native position after playing Catanatron's move (with its dice,
    card or steal) with Catanatron's position after it.
Prints the first few mismatches and a summary.
"""

import argparse
import multiprocessing as mp

import numpy as np


def _diff(a, b, path=""):
    if isinstance(a, dict):
        out = []
        for k in a:
            out += _diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, (list, tuple)) and a and isinstance(a[0], (list, tuple, dict)):
        if len(a) != len(b):
            return [f"{path}: {a} != {b}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}[{i}]")
        return out
    a = list(a) if isinstance(a, tuple) else a
    b = list(b) if isinstance(b, tuple) else b
    if isinstance(a, list):
        a, b = [list(x) if isinstance(x, tuple) else x for x in a], [list(x) if isinstance(x, tuple) else x for x in b]
    return [] if a == b else [f"{path}: {a} != {b}"]


def check_game(args):
    table, seed = args
    from catanatron import Color, Game
    from catanatron.players.value import base_fn

    from cbot import feats, native
    from cbot.players import make_player

    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE][: len(table)]
    game = Game([make_player(sp, c) for sp, c in zip(table, colors)], seed=seed)
    fn = base_fn()
    stats = {"positions": 0, "moves": 0, "legal": 0, "feats": 0, "base": 0, "next": 0, "maxdiff": 0.0}
    errors = []
    names = list(feats.feature_names(len(table)))
    if len(table) == 2:
        names += [f"PIP{i}" for i in range(54)] + [f"OPP{i}" for i in range(5)]

    def err(kind, msg):
        stats[kind] += 1
        if len(errors) < 5:
            errors.append(f"seed {seed} turn {game.state.num_turns} {kind}: {msg}")

    while game.winning_color() is None and game.state.num_turns < 1000:
        st = game.state
        d = native.to_dict(game)
        g = native._engine.Game.from_dict(d)
        stats["positions"] += 1
        bad = _diff(d, g.to_dict())
        if bad:
            err("next", "conversion " + "; ".join(bad[:3]))
        want = sorted(native.index(game, a) for a in game.playable_actions)
        got = sorted(g.legal())
        if want != got:
            err("legal", f"prompt {st.current_prompt} catanatron {want} native {got}")
        for seat, c in enumerate(st.colors):
            x, y = feats.encode(game, c), g.encode(seat)
            dx = np.abs(x - y)
            stats["maxdiff"] = max(stats["maxdiff"], float(dx.max()))
            if dx.max() > 1e-5:
                i = np.argsort(-dx)[:4]
                err("feats", ", ".join(f"{names[j]} {x[j]:g}/{y[j]:g}" for j in i if dx[j] > 1e-5))
            b0, b1 = fn(game, c), g.base_value(seat)
            if abs(b0 - b1) > 1e-6 * max(1.0, abs(b0)):
                err("base", f"seat {seat}: {b0!r} vs {b1!r}")
        player = st.current_player()
        action = player.decide(game, game.playable_actions)
        move = native.index(game, action)
        rec = game.execute(action)
        stats["moves"] += 1
        g.apply(move, native.outcome_of(rec))
        bad = _diff(native.to_dict(game), g.to_dict())
        if bad:
            err("next", f"after {action}: " + "; ".join(bad[:4]))
        if (game.winning_color() is None) != (g.winner() < 0):
            err("next", "winner differs")
    return stats, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--table", default="R,R")
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    table = a.table.split(",")
    jobs = [(table, a.seed0 + i) for i in range(a.games)]
    tot, errs = {}, []
    with mp.Pool(a.procs) as pool:
        for stats, errors in pool.imap_unordered(check_game, jobs):
            for k, v in stats.items():
                tot[k] = max(tot.get(k, 0), v) if k == "maxdiff" else tot.get(k, 0) + v
            errs += errors
    for e in errs[:20]:
        print(e)
    print(f"{a.table}: {a.games} games, {tot['positions']} positions, {tot['moves']} moves; mismatches: "
          f"legal {tot['legal']}, features {tot['feats']}, base_fn {tot['base']}, next state {tot['next']}; "
          f"max feature diff {tot['maxdiff']:.2g}")


if __name__ == "__main__":
    main()
