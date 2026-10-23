"""Policy-guided Monte Carlo tree search (PUCT, as in AlphaZero) over Catanatron.

The value net (cbot.search) scores the positions the search reaches and the
policy net (cbot.policy) gives each move's prior. A simulation walks down the
tree by

    argmax_a  Q(a)[mover] + c * P(a) * sqrt(N) / (1 + n(a))

until it reaches a position not yet scored, scores it, and adds that value to
every move on the way. Unlike the fixed-depth expectimax (NNSearchPlayer), the
search spends its budget on the lines that look best, to whatever depth they
need.

Chance moves (dice, development cards, steals) expand into all outcomes with
their probabilities (cbot.chance); a simulation takes the outcome furthest
below its share of the move's visits, so each move's Q is a near-stratified
average over outcomes. A position with one legal move is passed through
without being scored. Simulations run in batches with a virtual loss, so the
nets score several positions per call.

Everything else (fair determinization, opening completion, trade prompts) is
inherited from NNSearchPlayer.
"""

import math

import numpy as np

from cbot import chance, feats, policy as policy_mod
from cbot.search import NNSearchPlayer

NEW, PENDING, OPEN, OVER = range(4)


class _Node:
    __slots__ = ("game", "mover", "state", "value", "actions", "P", "N", "W", "kids", "Q0", "level")

    def __init__(self, game, level=0):
        self.game = game
        self.level = level  # decisions with a choice between the root and here
        self.mover = game.state.current_player_index
        self.state = NEW
        self.value = None  # the net's value vector, or the result once OVER
        self.actions = None
        self.P = None  # priors per move
        self.N = None  # visits per move
        self.W = None  # summed value vectors per move, (moves, players)
        self.kids = None  # per move: None, or [[node, prob, visits], ...]
        self.Q0 = None  # with qinit: each move's value by the net, one move ahead

    def open(self, actions, P, players):
        self.state = OPEN
        self.actions = actions
        self.P = P
        self.N = np.zeros(len(actions))
        self.W = np.zeros((len(actions), players))
        self.kids = [None] * len(actions)


class MCTSPlayer(NNSearchPlayer):
    LABEL = "MCTS"

    def __init__(
        self, color, model_path, sims=200, cpuct=1.5, batch=8, ptemp=1.0,
        pmix=0.0, fpu=0.0, noise=0.0, alpha=0.3, pick=0, qinit=0, **kw,
    ):
        policy = kw.pop("policy", None)
        super().__init__(color, model_path, **kw)
        self.sims = int(sims)
        self.cpuct = float(cpuct)
        self.batch = int(batch)
        # Priors: softmax(policy logits / ptemp), mixed with `pmix` uniform.
        # Without a policy net every move gets the same prior.
        self.pnet = policy_mod.get_policy(policy) if policy else None
        self.ptemp = float(ptemp)
        self.pmix = float(pmix)
        # First-play urgency: an unvisited move counts as the position's
        # value minus `fpu`, for the player to move.
        self.fpu = float(fpu)
        # qinit=k: when a position at one of the first k levels of choices is
        # scored, every move from it is played and scored too (an exact
        # expectation over chance outcomes, like one level of the
        # expectimax), and that value counts as the move's first visit. The
        # tree then only decides which lines to look at further.
        self.qinit = int(qinit)
        # Self-play exploration: Dirichlet noise on the root priors, and
        # pick=1 plays a move drawn by visit counts instead of the most visited.
        self.noise = float(noise)
        self.alpha = float(alpha)
        self.pick = int(pick)
        self.nprng = np.random.default_rng(self.rng.getrandbits(32))
        # last_visits: (action, visits) of every root move of the last
        # search, for training the policy on visit counts (cbot.gen).
        self.last_visits = None

    def decide(self, game, playable_actions):
        self.last_visits = None
        return super().decide(game, playable_actions)

    def _root_search(self, game):
        players = len(game.state.colors)
        root = _Node(game)
        self._score([root])
        if root.state == OVER:
            return [(root.value, a) for a in game.playable_actions]
        if self.noise:
            d = self.nprng.dirichlet([self.alpha] * len(root.actions))
            root.P = (1 - self.noise) * root.P + self.noise * d
        loss = [_loss(m, players) for m in range(players)]
        done = 0
        while done < self.sims:
            walks = [self._descend(root, loss) for _ in range(min(self.batch, self.sims - done))]
            done += len(walks)
            fresh = {id(leaf): leaf for _, leaf in walks if leaf.state == PENDING}
            if fresh:
                self._score(list(fresh.values()))
            for path, leaf in walks:
                for node, a in path:
                    node.W[a] += leaf.value - loss[node.mover]

        N = root.N
        if root.Q0 is not None:
            Q = (root.W + root.Q0) / (N + 1)[:, None]
        else:
            Q = np.where(N[:, None] > 0, root.W / np.maximum(N, 1)[:, None], root.value - 1.0)
        if self.pick:
            first = int(self.nprng.choice(len(N), p=N / N.sum()))
        else:
            first = int(np.lexsort((Q[:, root.mover], N))[-1])
        self.last_visits = [(a, int(k)) for a, k in zip(root.actions, N)]
        rest = sorted((i for i in range(len(N)) if i != first), key=lambda i: -Q[i, root.mover])
        return [(Q[i], root.actions[i]) for i in [first] + rest]

    def _descend(self, node, loss):
        """Walk down by PUCT, adding a virtual loss to each move taken, to a
        node that is over or waiting to be scored. Returns [(node, move)] and
        that node."""
        path = []
        while True:
            if node.state == NEW:
                acts = node.game.playable_actions
                if node.game.winning_color() is not None or len(acts) > 1:
                    node.state = PENDING
                else:
                    node.open(acts, np.ones(1), len(node.game.state.colors))
            if node.state != OPEN:
                return path, node
            a = self._select(node)
            path.append((node, a))
            node.N[a] += 1
            node.W[a] += loss[node.mover]
            node = self._child(node, a)

    def _select(self, node):
        if len(node.actions) == 1:
            return 0
        m = node.mover
        if node.Q0 is not None:
            q = (node.W[:, m] + node.Q0[:, m]) / (node.N + 1)
        else:
            q = np.where(node.N > 0, node.W[:, m] / np.maximum(node.N, 1), node.value[m] - self.fpu)
        u = self.cpuct * node.P * math.sqrt(node.N.sum() + 1) / (1 + node.N)
        return int(np.argmax(q + u))

    def _child(self, node, a):
        if node.kids[a] is None:
            self._spectrum(node, a)
        outs = node.kids[a]
        if len(outs) == 1:
            o = outs[0]
        else:
            tot = sum(o[2] for o in outs) + 1
            o = max(outs, key=lambda o: o[1] - o[2] / tot)
        o[2] += 1
        return o[0]

    def _score(self, nodes):
        """Value, and priors over the moves, of each node."""
        live = self._values(nodes)
        if not live:
            return
        games = [nd.game for nd in live]
        X = np.stack([feats.encode(g, self.color) for g in games])
        for nd, P in zip(live, self._priors(games, X)):
            nd.open(nd.game.playable_actions, P, len(nd.value))
        near = [nd for nd in live if nd.level < self.qinit]
        for nd in near:
            for a in range(len(nd.actions)):
                self._spectrum(nd, a)
        self._values([o[0] for nd in near for outs in nd.kids for o in outs])
        for nd in near:
            nd.Q0 = np.stack([sum(p * o.value for o, p, _ in outs) for outs in nd.kids])

    def _spectrum(self, node, a):
        level = node.level + (len(node.actions) > 1)
        node.kids[a] = [
            [_Node(g, level), p, 0] for g, p in chance.execute_spectrum(node.game, node.actions[a])
        ]

    def _values(self, nodes):
        """Sets the value of each node that has none: the result if the game
        is over, else the net's. Returns the nodes still in play."""
        live, todo = [], []
        for nd in nodes:
            w = nd.game.winning_color()
            if w is not None:
                nd.value = np.zeros(len(nd.game.state.colors))
                nd.value[nd.game.state.colors.index(w)] = 1.0
                nd.state = OVER
                continue
            live.append(nd)
            if nd.value is None:
                todo.append(nd)
        if todo:
            for nd, v in zip(todo, self._eval([nd.game for nd in todo])):
                nd.value = v
        return live

    def _priors(self, games, X):
        """X: the games' features for this seat (reused where it is to move)."""
        acts = [g.playable_actions for g in games]
        if self.pnet is None:
            return [np.full(len(a), 1.0 / len(a)) for a in acts]
        Xm = np.stack([
            x if g.state.current_color() == self.color else feats.encode(g, g.state.current_color())
            for g, x in zip(games, X)
        ])
        logits = self.pnet(Xm)
        out = []
        for a, lg in zip(acts, logits):
            idx = np.array([policy_mod.index(x) for x in a])
            known = idx >= 0
            P = np.full(len(a), 1.0 / len(a))  # moves without a number (player trades)
            if known.any():
                z = lg[idx[known]] / self.ptemp
                z = np.exp(z - z.max())
                P[known] = z / z.sum() * known.mean()
            P = P / P.sum()
            if self.pmix:
                P = (1 - self.pmix) * P + self.pmix / len(a)
            out.append(P)
        return out


def _loss(mover, n):
    """A lost simulation, from the mover's point of view."""
    v = np.full(n, 1.0 / (n - 1))
    v[mover] = 0.0
    return v
