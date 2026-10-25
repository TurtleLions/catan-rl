"""Catanatron's web server with our bot as a seat ("CBOT").

Setup, once (Catanatron checked out in vendor/catanatron as in RESULTS.md):

    .venv/bin/pip install flask flask_cors flask_sqlalchemy sqlalchemy
    git -C vendor/catanatron apply ../../cbot/catanatron_web.patch
    cp cbot/web_tiles/*.svg vendor/catanatron/ui/src/assets/
    docker build -t cbot-ui vendor/catanatron/ui    # the UI needs Node 24

The patch: games are shown from the human's seat (other hands as counts),
the robber asks whom to rob, a robbed player sees the card they lost,
trades with the other players (offer from the Trade menu after rolling;
answer or pick a partner in the toolbar),
standard rules by default (10 points, discard over 7, no friendly robber,
you against three CBOTs), no full-information analysis panel, and a lighter
theme; web_tiles has plain hexes in the standard tile colors.

Run:

    .venv/bin/flask --app cbot.web run --host 127.0.0.1 --port 5001
    docker run -d --name cbot-ui -p 127.0.0.1:3001:3000 cbot-ui

and play at localhost:3001. Games are kept in SQLite (runs/web.sqlite)
unless DATABASE_URL is set.
"""

import dataclasses
import os

from catanatron.models.player import Player
from catanatron.registry import REGISTRY

from cbot.players import make_player

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Cbot(Player):
    """Search with a learned value net. Against three players: the A20big
    ensemble (70.0% vs three AlphaBetas at depth 2, hidden cards guessed; depth 3
    gained A20 about two points at five times the time). One on one: native tree
    search with the N1 committee and policy P1 (90.1% vs AlphaBeta)."""

    LABEL = "CBOT"

    @dataclasses.dataclass(frozen=True)
    class Params:
        model: str = "A20big"  # models/<model>.txt holds the NN spec
        model1v1: str = "BOT1V1"  # the same, for two-player games
        depth: int = 3  # about 250 ms a decision, a few seconds at most
        trade: int = 1  # offer trades to other players (answering is always on)

    def __init__(self, color, params=None):
        super().__init__(color, params)
        self.bot = None  # built at the first decision, when the number of players is known

    def _make(self, players):
        name = self.params.model1v1 if players == 2 else self.params.model
        with open(os.path.join(ROOT, "models", f"{name}.txt")) as f:
            spec = f.read().strip()
        if spec.split(":")[0] not in ("NN", "MCTS", "NMCTS"):  # a bare list of nets, e.g. models/M0.txt
            spec = f"NN:{spec}:initf=2"
        spec = spec.replace("models/", os.path.join(ROOT, "models") + "/")
        if spec.startswith("NN:"):
            spec += f":depth={self.params.depth}:trade={self.params.trade}"
        try:
            return make_player(spec, self.color)
        except ImportError:  # native engine not built (cbot/engine/build.sh): same search in Python
            return make_player(spec.replace("NMCTS:", "MCTS:", 1), self.color)

    def decide(self, game, playable_actions):
        if self.bot is None:
            self.bot = self._make(len(game.state.colors))
        return self.bot.decide(game, playable_actions)


def create_app():
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{os.path.join(ROOT, 'runs', 'web.sqlite')}")
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    from catanatron.web import create_app as catanatron_app

    app = catanatron_app()
    REGISTRY.register("CBOT", Cbot, replace=True)
    return app
