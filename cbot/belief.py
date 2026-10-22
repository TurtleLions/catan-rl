"""The other hands as a player at the table knows them, dealt as one guess.

A copy of the real game holds every hand face up, and the search plays its
moves out on copies. determinize(game, me, rng) returns a copy in which what
`me` cannot see is replaced by a guess consistent with what `me` has seen:

  * Resource cards are counted from the game's public record, the way a
    careful player counts them: dice payouts, builds, trades, discards and
    monopolies are public; the card in a steal between two other players is
    not. The record is replayed on a shadow state in which each such card is
    drawn from the victim's hand in the shadow. A draw that later proves
    impossible (a hand going negative, a hand total or monopoly that
    disagrees with the real one, an offer of cards the offerer would not
    hold) is redrawn, going back to earlier steals when needed, so the guess
    always explains the whole record.
  * Development cards in opponents' hands are dealt again from everything
    `me` has not seen (the deck plus those hands), so each opponent keeps
    their number of cards; victory points follow the cards dealt.

The replay is kept per game and extended on the next decision, so each
decision only replays the actions since the last one.
"""

import random
from collections import OrderedDict

from catanatron.apply_action import apply_action
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES, ActionRecord, ActionType
from catanatron.models.player import SimplePlayer
from catanatron.state import State

from cbot import fast  # noqa: F401  (patch Catanatron before taking references)

VP_CARD = "VICTORY_POINT"
PLAYABLE = ("KNIGHT", "MONOPOLY", "YEAR_OF_PLENTY", "ROAD_BUILDING")
DEALS = 1000  # tries at a deal of development cards that leaves the game unfinished
_TRACKERS = OrderedDict()  # (game id, color) -> _Tracker; a few games at a time
_KEEP = 16
_SHOWS = (ActionType.OFFER_TRADE, ActionType.ACCEPT_TRADE)


class _Inconsistent(Exception):
    pass


def _hand(state, color):
    k = f"P{state.color_to_index[color]}"
    return [state.player_state[f"{k}_{r}_IN_HAND"] for r in RESOURCES]


class _Tracker:
    def __init__(self, game, me, rng):
        self.me, self.rng = me, rng
        self.reset(game)

    def reset(self, game):
        st = game.state
        s = State(
            [SimplePlayer(c) for c in st.colors],
            st.board.map,
            discard_limit=st.discard_limit,
            friendly_robber=st.friendly_robber,
            rng=random.Random(0),
        )
        # State() seats the players in random order; use the game's.
        s.players = [SimplePlayer(c) for c in st.colors]
        s.colors = tuple(st.colors)
        s.color_to_index = {c: i for i, c in enumerate(s.colors)}
        self.state, self.n = s, 0
        # One entry per hidden steal replayed: [record index, shadow state
        # before it, cards not tried yet]. A contradiction later on goes back
        # to the latest steal with a card left to try (depth-first search).
        self.stack, self.forced = [], None

    def _order(self, hand):
        """The cards the victim may have lost, in a random order weighted by
        how many of each the victim holds (weighted sampling without
        replacement)."""
        keys = [(self.rng.random() ** (1.0 / n), r) for r, n in zip(RESOURCES, hand) if n]
        return [r for _, r in sorted(keys, reverse=True)]

    def _backtrack(self):
        while self.stack:
            i, snap, rest = self.stack[-1]
            if rest:
                self.state, self.n = snap.copy(), i
                self.forced = (i, rest.pop(0))
                return
            self.stack.pop()
        raise _Inconsistent

    def advance(self, records):
        while self.n < len(records):
            rec = records[self.n]
            a = rec.action
            if (
                a.action_type == ActionType.MOVE_ROBBER
                and a.value[1] is not None
                and self.me not in (a.color, a.value[1])
            ):
                if self.forced is not None and self.forced[0] == self.n:
                    card, self.forced = self.forced[1], None
                else:
                    order = self._order(_hand(self.state, a.value[1]))
                    if not order:
                        self._backtrack()
                        continue
                    self.stack.append([self.n, self.state.copy(), order[1:]])
                    card = order[0]
                rec = ActionRecord(action=a, result=card)
            s = self.state
            try:
                apply_action(s, a, rec)
            except (AssertionError, ValueError):
                self._backtrack()
                continue
            if any(v < 0 for c in s.colors for v in _hand(s, c)):
                self._backtrack()
                continue
            # Offering cards, or accepting an offer, shows that one holds them.
            if a.action_type in _SHOWS:
                need = a.value[:5] if a.action_type == ActionType.OFFER_TRADE else a.value[5:10]
                if any(n > h for n, h in zip(need, _hand(s, a.color))):
                    self._backtrack()
                    continue
            self.n += 1

    def hands(self, game):
        """Each player's resource hand in the shadow, once it agrees with
        everything public about the real one."""
        st = game.state
        records = st.action_records
        if self.n > len(records):
            self.reset(game)
        while True:
            self.advance(records)
            s = self.state
            ok = s.resource_freqdeck == st.resource_freqdeck
            out = {}
            for c in st.colors:
                h, real = _hand(s, c), _hand(st, c)
                ok = ok and sum(h) == sum(real) and (c != self.me or h == real)
                out[c] = h
            if ok:
                return out
            self._backtrack()  # e.g. a monopoly that took a different number of cards


def _deal_by_counts(game, me, rng):
    """Fallback guess: the opponents' cards pooled and dealt by hand size.
    Uses only what is public (hand sizes and, through the bank, how many of
    each resource the opponents hold together)."""
    st = game.state
    opps = [c for c in st.colors if c != me]
    pool = [r for c in opps for r, n in zip(RESOURCES, _hand(st, c)) for _ in range(n)]
    rng.shuffle(pool)
    out = {me: _hand(st, me)}
    for c in opps:
        n = sum(_hand(st, c))
        cards, pool = pool[:n], pool[n:]
        out[c] = [cards.count(r) for r in RESOURCES]
    return out


def resource_hands(game, me, rng):
    key = (game.id, me)
    tr = _TRACKERS.get(key)
    if tr is None:
        tr = _TRACKERS[key] = _Tracker(game, me, rng)
        while len(_TRACKERS) > _KEEP:
            _TRACKERS.popitem(last=False)
    _TRACKERS.move_to_end(key)
    try:
        return tr.hands(game)
    except _Inconsistent:
        # No guess explains the record (should not happen): start over next
        # time, and guess from the public counts alone this time.
        tr.reset(game)
        return _deal_by_counts(game, me, rng)


def determinize(game, me, rng):
    """A copy of `game` with the hands `me` cannot see replaced by a guess."""
    hands = resource_hands(game, me, rng)
    g = game.copy()
    st = g.state
    ps = st.player_state
    opps = [c for c in st.colors if c != me]

    for c in opps:
        k = f"P{st.color_to_index[c]}"
        for r, n in zip(RESOURCES, hands[c]):
            ps[f"{k}_{r}_IN_HAND"] = n

    pool, sizes = list(st.development_listdeck), {}
    for c in opps:
        k = f"P{st.color_to_index[c]}"
        sizes[c] = 0
        for card in DEVELOPMENT_CARDS:
            n = ps[f"{k}_{card}_IN_HAND"]
            pool += [card] * n
            sizes[c] += n
    # The game is not over, so nobody holds enough victory-point cards to
    # have won: deal again until that holds (conditioning on it). Should that
    # keep failing, deal the other cards first.
    room = {
        c: g.vps_to_win - 1 - ps[f"P{st.color_to_index[c]}_VICTORY_POINTS"] for c in opps
    }
    for _ in range(DEALS):
        rng.shuffle(pool)
        i, ok = 0, True
        for c in opps:
            ok = ok and pool[i : i + sizes[c]].count(VP_CARD) <= room[c]
            i += sizes[c]
        if ok:
            break
    else:
        rng.shuffle(pool)
        pool.sort(key=lambda card: card == VP_CARD)
    for c in opps:
        k = f"P{st.color_to_index[c]}"
        dealt, pool = pool[: sizes[c]], pool[sizes[c] :]
        for card in DEVELOPMENT_CARDS:
            ps[f"{k}_{card}_IN_HAND"] = dealt.count(card)
        for card in PLAYABLE:
            ps[f"{k}_{card}_OWNED_AT_START"] = card in dealt
        ps[f"{k}_ACTUAL_VICTORY_POINTS"] = ps[f"{k}_VICTORY_POINTS"] + dealt.count(VP_CARD)
    st.development_listdeck = pool
    return g
