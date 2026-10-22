"""A policy over Catanatron's fixed action list: which moves are worth searching.

Moves are numbered by catanatron.gym's action array for the base map, with
the robber's victim recorded as "an opponent" rather than by color, so one
numbering serves every seat. Moves outside that list (player trades) have no
number and are always searched.

The net (cbot.train_policy) maps the mover's features to one logit per move;
the search expands only the top few legal moves below the root.
"""

import importlib.util
import os

import numpy as np

import catanatron
from catanatron.models.enums import ActionType
from catanatron.models.player import Color

from cbot import feats
from cbot.search import ValueNet

# catanatron.gym's package imports gymnasium; load just its action list.
_spec = importlib.util.spec_from_file_location(
    "_catan_action_space",
    os.path.join(os.path.dirname(catanatron.__file__), "gym", "envs", "action_space.py"),
)
_action_space = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_action_space)
get_action_array = _action_space.get_action_array

_OPP = Color.RED  # stands for any victim
_ARRAY = get_action_array((_OPP,), "BASE")
_INDEX = {a: i for i, a in enumerate(_ARRAY)}
SIZE = len(_ARRAY)


def index(action):
    """The move's number, or -1 if it has none."""
    t, v = action.action_type, action.value
    if t == ActionType.MOVE_ROBBER:
        v = (v[0], None if v[1] is None else _OPP)
    elif t == ActionType.BUILD_ROAD:
        v = tuple(sorted(v))
    return _INDEX.get((t, v), -1)


class PolicyNet(ValueNet):
    """Same file format as a value net; the output is one logit per move."""

    def __call__(self, X):
        return self.logits(X)


_POLICIES = {}


def get_policy(path):
    if path not in _POLICIES:
        _POLICIES[path] = PolicyNet(path)
    return _POLICIES[path]


def top(net, game, actions, k):
    """The k legal moves the policy rates highest for the player to move.
    Moves without a number are kept on top of those."""
    idx = np.array([index(a) for a in actions])
    known = idx >= 0
    if known.sum() <= k:
        return actions
    x = feats.encode(game, game.state.current_color())[None, :]
    logits = net(x)[0]
    order = np.argsort(-logits[idx[known]])[:k]
    kept = [a for a, ok in zip(actions, known) if ok]
    return [a for a, ok in zip(actions, known) if not ok] + [kept[i] for i in order]
