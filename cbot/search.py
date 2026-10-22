"""Expectimax search over Catanatron with a learned P(win) leaf evaluator.

With more than two players the search is max-n: every node carries a vector of
win probabilities (one per seat) and the player to move picks the child that is
best for itself. In 1v1 this is ordinary minimax.

Differences from Catanatron's AlphaBetaPlayer:
  * leaves are scored by an MLP trained on game outcomes (cbot.train), not the
    hand-weighted base_fn;
  * chance nodes (dice, dev-card draws, robber steals) produce the outcome
    they are weighted for (cbot.chance);
  * the tree is built first and all leaves are evaluated in one batch.
"""

import random

import numpy as np

from catanatron.models.player import Player
from catanatron.models.enums import ActionPrompt, ActionType
from cbot import fast  # noqa: F401  (patch Catanatron before taking references)
from catanatron.players.value import ValueFunctionPlayer, base_fn
from catanatron.players.tree_search_utils import list_prunned_actions

from cbot import belief, chance, feats


class ValueNet:
    def __init__(self, path):
        d = np.load(path)
        self.cols = d["cols"] if "cols" in d else None
        self.derived = bool(d["derived"]) if "derived" in d else False
        self.mean = d["mean"]
        self.std = d["std"]
        self.players = int(d["players"]) if "players" in d else 2
        self.layers = []
        i = 0
        while f"W{i}" in d:
            self.layers.append((d[f"W{i}"], d[f"b{i}"]))
            i += 1
        # Fold normalization into the first layer.
        W0, b0 = self.layers[0]
        W0n = W0 / self.std[:, None]
        b0n = b0 - (self.mean / self.std) @ W0
        self.layers[0] = (W0n.astype(np.float32), b0n.astype(np.float32))

    def __call__(self, X):
        """1v1 nets: P(P0 wins), shape (n,). Multiplayer nets: softmax over
        seats relative to P0, shape (n, players)."""
        h = X if self.cols is None else X[:, self.cols]
        if self.derived:
            h = np.concatenate([h, feats.derive(X)], axis=1)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                np.maximum(h, 0, out=h)
        if h.shape[1] > 1:
            h = np.exp(h - h.max(1, keepdims=True))
            return h / h.sum(1, keepdims=True)
        return 1.0 / (1.0 + np.exp(-np.clip(h[:, 0], -30, 30)))


_NETS = {}


class Ensemble:
    def __init__(self, nets):
        self.nets = nets

    def __call__(self, X):
        return sum(n(X) for n in self.nets) / len(self.nets)


def get_net(path):
    """`a.npz` or an ensemble `a.npz+b.npz` (mean of win probabilities)."""
    if path not in _NETS:
        parts = path.split("+")
        _NETS[path] = (
            ValueNet(path) if len(parts) == 1 else Ensemble([get_net(p) for p in parts])
        )
    return _NETS[path]


class NNSearchPlayer(Player):
    LABEL = "NNSearch"

    def __init__(
        self, color, model_path, depth=2, prune=1, maxleaves=4000,
        place=0, topk=6, mix=0.5, initf=0, qroll=0, sym=0, openk=0, fair=1,
    ):
        super().__init__(color)
        self.net = get_net(model_path)
        self.depth = int(depth)
        self.prune = int(prune)
        self.maxleaves = int(maxleaves)
        # Opening placement by simulation: `place` rollouts for each of the
        # net's `topk` settlement candidates; score = mix*net + (1-mix)*rollouts.
        self.place = int(place)
        self.topk = int(topk)
        self.mix = float(mix)
        self.rng = random.Random()
        # Roll extension: a leaf whose only action is ROLL is expanded over the
        # 11 dice outcomes instead of being scored before the dice.
        self.qroll = int(qroll)
        # Symmetrized leaf value: average P(me wins) with 1 - P(opponent wins).
        self.sym = int(sym)
        # Opening handling: 0 = plain search, 1 = ValueFunctionPlayer,
        # 2 = "opening completion": finish the opening with ValueFunction proxies
        # for both seats, then score the first mid-game position with the net.
        self.initmode = int(initf)
        self.initf = ValueFunctionPlayer(color) if self.initmode else None
        # Opening completion only for the `openk` best candidates by base_fn
        # (0 = all candidates).
        self.openk = int(openk)
        # Fair play: search a copy of the game in which the other hands are
        # what this seat can know (cbot.belief), not the real ones. 0 searches
        # the real game, every hand face up (how the bots up to A20big were
        # trained and evaluated).
        self.fair = int(fair)

    def _actions(self, game, root):
        if self.prune and not root:
            return list_prunned_actions(game)
        return game.playable_actions

    def decide(self, game, playable_actions):
        self.last_value = None
        if len(playable_actions) == 1:
            return playable_actions[0]
        if self.fair:
            game = belief.determinize(game, self.color, self.rng)
        if self.initmode == 2 and game.state.is_initial_build_phase:
            return self._complete_opening(game, playable_actions)
        if self.initf is not None and game.state.is_initial_build_phase:
            if (
                self.place
                and game.state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
            ):
                return self._place_by_rollouts(game, self._base_shortlist(game))
            return self.initf.decide(game, playable_actions)
        leaves = []
        tree = self._build(game, self.depth, leaves, root=True)
        if leaves:
            vals = self._eval(leaves)
        else:
            vals = np.zeros(0)
        me = self._me(game)
        scored = [
            (sum(p * self._value(n, vals) for n, p in outcomes), a)
            for a, outcomes in tree[1]
        ]
        scored.sort(key=lambda x: -x[0][me])
        # last_vec: win probability per seat (game.state.colors order).
        self.last_vec = scored[0][0]
        self.last_value = float(self.last_vec[me])
        scored = [(v[me], a) for v, a in scored]
        if (
            self.place
            and game.state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT
        ):
            return self._place_by_rollouts(game, scored[: self.topk])
        return scored[0][1]

    def _complete_opening(self, game, actions):
        if self.openk and len(actions) > self.openk:
            actions = [a for _, a in self._base_shortlist(game, self.openk)]
        finals = []
        for a in actions:
            g = game.copy()
            g.execute(a, validate_action=False)
            while g.state.is_initial_build_phase and g.winning_color() is None:
                c = g.state.current_color()
                if c not in _ROLLOUT_POLICY:
                    _ROLLOUT_POLICY[c] = ValueFunctionPlayer(c)
                g.execute(_ROLLOUT_POLICY[c].decide(g, g.playable_actions), validate_action=False)
            finals.append(g)
        vals = self._eval(finals)[:, self._me(game)]
        return actions[int(np.argmax(vals))]

    def _base_shortlist(self, game, k=None):
        fn = base_fn()
        scored = []
        for a in game.playable_actions:
            g = game.copy()
            g.execute(a, validate_action=False)
            scored.append((fn(g, self.color), a))
        scored.sort(key=lambda x: -x[0])
        # mix=0 => pure rollout score; the base_fn scale is not a probability.
        return [(0.0, a) for _, a in scored[: k or self.topk]]

    def _place_by_rollouts(self, game, cands):
        best_a, best_s = None, -1.0
        for v, a in cands:
            wins = 0
            for _ in range(self.place):
                wins += rollout(game, a, self.color, self.rng)
            s = self.mix * v + (1 - self.mix) * wins / self.place
            if s > best_s:
                best_a, best_s = a, s
        return best_a

    def _me(self, game):
        return game.state.colors.index(self.color)

    def _eval(self, games):
        """Win probability per seat, shape (len(games), players), in
        game.state.colors order."""
        X = np.stack([feats.encode(g, self.color) for g in games])
        vals = self.net(X)
        n = len(games[0].state.colors)
        me = self._me(games[0])
        if vals.ndim == 1:
            if self.sym:
                opp = [c for c in games[0].state.colors if c != self.color][0]
                Xo = np.stack([feats.encode(g, opp) for g in games])
                vals = 0.5 * (vals + 1.0 - self.net(Xo))
            rel = np.stack([vals, 1.0 - vals], axis=1)
        else:
            rel = vals
        # Relative seat j is absolute seat (me + j) % n.
        return np.roll(rel, me, axis=1)

    def _build(self, game, depth, leaves, root=False):
        w = game.winning_color()
        if w is not None:
            v = np.zeros(len(game.state.colors))
            v[game.state.colors.index(w)] = 1.0
            return ("T", v)
        if depth == 0 or (not root and len(leaves) >= self.maxleaves):
            acts = game.playable_actions
            if (
                self.qroll
                and depth == 0
                and len(acts) == 1
                and acts[0].action_type == ActionType.ROLL
            ):
                outs = chance.execute_spectrum(game, acts[0])
                kids = []
                for g, p in outs:
                    leaves.append(g)
                    kids.append((("L", len(leaves) - 1), p))
                return (game.state.current_player_index, [(acts[0], kids)])
            leaves.append(game)
            return ("L", len(leaves) - 1)
        actions = self._actions(game, root)
        children = []
        for a in actions:
            outs = chance.execute_spectrum(game, a)
            children.append(
                (a, [(self._build(g, depth - 1, leaves), p) for g, p in outs])
            )
        # Decision node, labeled with the mover's seat.
        return (game.state.current_player_index, children)

    def _value(self, node, vals):
        """Win-probability vector (one entry per seat) of a tree node."""
        kind = node[0]
        if kind == "L":
            return vals[node[1]]
        if kind == "T":
            return node[1]
        best = None
        for _, outcomes in node[1]:
            v = sum(p * self._value(n, vals) for n, p in outcomes)
            if best is None or v[kind] > best[kind]:
                best = v
        return best


_ROLLOUT_POLICY = {}


def rollout(game, action, me, rng, eps=0.05, turns_limit=400):
    """Play `action`, then finish the game with ValueFunctionPlayer on both
    sides. Uses a private RNG and a reshuffled dev deck so the simulation
    neither knows nor disturbs the real game's future."""
    g = game.copy()
    r = random.Random(rng.random())
    g.random = r
    g.state.random = r
    r.shuffle(g.state.development_listdeck)
    g.execute(action, validate_action=False)
    for c in g.state.colors:
        if c not in _ROLLOUT_POLICY:
            _ROLLOUT_POLICY[c] = ValueFunctionPlayer(c)

    while g.winning_color() is None and g.state.num_turns < turns_limit:
        acts = g.playable_actions
        if len(acts) > 1 and r.random() < eps:
            a = r.choice(acts)
        else:
            a = _ROLLOUT_POLICY[g.state.current_color()].decide(g, acts)
        g.execute(a, validate_action=False)
    return 1 if g.winning_color() == me else 0
