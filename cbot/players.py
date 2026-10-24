"""Player factory for the arena: spec string -> Player.

Specs:
    AB                AlphaBetaPlayer defaults (depth 2, no pruning, base value fn)
    AB:3              AlphaBeta depth 3
    ABfix[:d]         AlphaBeta with the chance-node bug fixed (cbot.chance)
    F                 ValueFunctionPlayer
    R / W             Random / WeightedRandom
    NN:path[:k=v,...] cbot.search.NNSearchPlayer with weights at path
    MCTS:path[:k=v,...] cbot.mcts.MCTSPlayer (PUCT search) with weights at path
    NMCTS:path[:k=v,...] the same search run by the native engine (cbot.native, 1v1)
"""

from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.player import RandomPlayer
import catanatron.players.minimax as minimax

from cbot import chance, fast  # noqa: F401  (fast: engine speedups)


class FixedAlphaBetaPlayer(AlphaBetaPlayer):
    def decide(self, game, playable_actions):
        orig = minimax.expand_spectrum
        minimax.expand_spectrum = chance.expand_spectrum
        try:
            return super().decide(game, playable_actions)
        finally:
            minimax.expand_spectrum = orig


def _kv(parts):
    out = {}
    for p in parts:
        k, v = p.split("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        out[k] = v
    return out


def make_player(spec, color):
    parts = spec.split(":")
    name = parts[0]
    if name in ("AB", "ABfix"):
        cls = AlphaBetaPlayer if name == "AB" else FixedAlphaBetaPlayer
        depth = int(parts[1]) if len(parts) > 1 else 2
        return cls(color, AlphaBetaPlayer.Params(depth=depth))
    if name == "F":
        return ValueFunctionPlayer(color)
    if name == "W":
        return WeightedRandomPlayer(color)
    if name == "R":
        return RandomPlayer(color)
    if name == "NN":
        from cbot.search import NNSearchPlayer

        return NNSearchPlayer(color, parts[1], **_kv(parts[2:]))
    if name == "MCTS":
        from cbot.mcts import MCTSPlayer

        return MCTSPlayer(color, parts[1], **_kv(parts[2:]))
    if name == "NMCTS":
        from cbot.native import NativeMCTSPlayer

        return NativeMCTSPlayer(color, parts[1], **_kv(parts[2:]))
    raise ValueError(spec)
