"""Chance-node expansion with the outcomes actually applied.

Catanatron's tree_search_utils.execute_spectrum (as of ecf9311) puts the
intended dice roll / dev card in action.value, which apply_action ignores, so
every dev-card branch draws the real top card and every roll branch draws a
fresh random roll (see upstream PR #385). Here the outcome goes through
ActionRecord.result, which apply_action does honor.
"""

from collections import defaultdict

from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    ActionRecord,
    ActionType,
)
from catanatron.state_functions import (
    get_dev_cards_in_hand,
    get_enemy_colors,
    get_player_freqdeck,
)
from catanatron.players.tree_search_utils import DETERMINISTIC_ACTIONS

# P(sum) for 2..12, and one representative dice pair per sum.
ROLLS = [(s, (s // 2, s - s // 2), (6 - abs(s - 7)) / 36.0) for s in range(2, 13)]


def _exec(game, action, result=None):
    g = game.copy()
    rec = None if result is None else ActionRecord(action=action, result=result)
    g.execute(action, validate_action=False, action_record=rec)
    return g


def execute_spectrum(game, action):
    t = action.action_type
    if t in DETERMINISTIC_ACTIONS:
        return [(_exec(game, action), 1.0)]
    if t == ActionType.ROLL:
        return [(_exec(game, action, dice), p) for _, dice, p in ROLLS]
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        # Unseen cards from the buyer's perspective: deck + enemy hands.
        deck = list(game.state.development_listdeck)
        for color in get_enemy_colors(game.state.colors, action.color):
            for card in DEVELOPMENT_CARDS:
                deck += [card] * get_dev_cards_in_hand(game.state, color, card)
        out = []
        for card in set(deck):
            try:
                g = _exec(game, action, card)
            except Exception:
                # Card only exists in an enemy hand; draw the real top card instead.
                g = _exec(game, action)
            out.append((g, deck.count(card) / len(deck)))
        return out
    if t == ActionType.MOVE_ROBBER:
        _, robbed = action.value
        if robbed is None:
            return [(_exec(game, action), 1.0)]
        hand = get_player_freqdeck(game.state, robbed)
        total = sum(hand)
        if total == 0:
            return [(_exec(game, action), 1.0)]
        return [
            (_exec(game, action, res), n / total)
            for res, n in zip(RESOURCES, hand)
            if n > 0
        ]
    raise RuntimeError(f"Unknown ActionType {t}")


def expand_spectrum(game, actions):
    children = defaultdict(list)
    for a in actions:
        children[a] = execute_spectrum(game, a)
    return children
