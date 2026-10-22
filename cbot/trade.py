"""Trades between players.

Catanatron's sequence: after rolling, the player to move offers a bundle of
cards for another (OFFER_TRADE); every other player accepts or rejects in seat
order; the offerer then confirms with one of those who accepted or cancels.
Offers are not in playable_actions (they can be any bundle), so a bot draws
its own from offers().

Importing this module fixes two things in Catanatron's version:
  * the offerer was asked to answer their own offer when not seated last;
  * an offer of cards the offerer does not hold was accepted.
"""

from catanatron import apply_action as A
from catanatron import game as game_mod
from catanatron.models.enums import RESOURCES, Action, ActionPrompt, ActionRecord, ActionType
from catanatron.state_functions import (
    get_player_freqdeck,
    player_freqdeck_add,
    player_freqdeck_subtract,
    player_has_rolled,
)

MAX_OFFERS = 2  # a bot's offers per turn; the same offer is not made twice


def _next_responder(state, after):
    """The next seat after `after` to answer the current offer, or None."""
    offerer = state.current_trade[10]
    return next((i for i in range(after + 1, len(state.colors)) if i != offerer), None)


def apply_offer_trade(state, action):
    state.is_resolving_trade = True
    state.current_trade = (*action.value[:10], state.current_turn_index)
    state.acceptees = tuple(False for _ in state.colors)
    state.current_player_index = _next_responder(state, -1)
    state.current_prompt = ActionPrompt.DECIDE_TRADE
    return ActionRecord(action=action, result=None)


def _after_answer(state):
    nxt = _next_responder(state, state.current_player_index)
    if nxt is not None:
        state.current_player_index = nxt
        return
    state.current_player_index = state.current_turn_index
    if any(state.acceptees):
        state.current_prompt = ActionPrompt.DECIDE_ACCEPTEES
    else:
        A.reset_trading_state(state)
        state.current_prompt = ActionPrompt.PLAY_TURN


def apply_accept_trade(state, action):
    acceptees = list(state.acceptees)
    acceptees[state.colors.index(action.color)] = True
    state.acceptees = tuple(acceptees)
    _after_answer(state)
    return ActionRecord(action=action, result=None)


def apply_reject_trade(state, action):
    _after_answer(state)
    return ActionRecord(action=action, result=None)


_is_valid_action = game_mod.is_valid_action


def is_valid_action(playable_actions, state, action):
    if action.action_type == ActionType.OFFER_TRADE:
        hand = get_player_freqdeck(state, action.color)
        if any(n > h for n, h in zip(action.value[:5], hand)):
            return False
    return _is_valid_action(playable_actions, state, action)


A.apply_offer_trade = apply_offer_trade
A.apply_accept_trade = apply_accept_trade
A.apply_reject_trade = apply_reject_trade
game_mod.is_valid_action = is_valid_action


def offers_this_turn(state, color):
    """Values (10-tuples) of the offers `color` has made since rolling."""
    out = []
    for rec in reversed(state.action_records):
        a = rec.action
        if a.action_type == ActionType.ROLL:
            break
        if a.action_type == ActionType.OFFER_TRADE and a.color == color:
            out.append(tuple(a.value[:10]))
    return out


def offers(game, color):
    """The offers a bot considers: one card for one, or two of a kind for one,
    of a resource someone else may hold (not counting offers already made
    this turn)."""
    st = game.state
    if (
        st.current_prompt != ActionPrompt.PLAY_TURN
        or st.current_color() != color
        or st.is_initial_build_phase
        or not player_has_rolled(st, color)
    ):
        return []
    made = offers_this_turn(st, color)
    if len(made) >= MAX_OFFERS:
        return []
    hand = get_player_freqdeck(st, color)
    out = []
    for g, give in enumerate(RESOURCES):
        for k in (1, 2):
            if hand[g] < k:
                continue
            for w, want in enumerate(RESOURCES):
                # Cards of `want` outside the bank and this hand: public.
                if w == g or 19 - st.resource_freqdeck[w] - hand[w] <= 0:
                    continue
                v = [0] * 10
                v[g], v[5 + w] = k, 1
                if tuple(v) not in made:
                    out.append(Action(color, ActionType.OFFER_TRADE, tuple(v)))
    return out


def _settle(g):
    st = g.state
    A.reset_trading_state(st)
    st.current_player_index = st.current_turn_index
    st.current_prompt = ActionPrompt.PLAY_TURN
    g.playable_actions = game_mod.generate_playable_actions(st)
    return g


def traded(game, offerer, partner, value):
    """A copy of `game` after `offerer` and `partner` swap the cards of the
    offer `value`, back at the offerer's turn."""
    g = game.copy()
    give, get = list(value[:5]), list(value[5:10])
    player_freqdeck_subtract(g.state, offerer, give)
    player_freqdeck_add(g.state, offerer, get)
    player_freqdeck_subtract(g.state, partner, get)
    player_freqdeck_add(g.state, partner, give)
    return _settle(g)


def untraded(game):
    """A copy of `game` with the offer under way called off."""
    return _settle(game.copy())


def holds(game, color, cards):
    return all(n <= h for n, h in zip(cards, get_player_freqdeck(game.state, color)))
