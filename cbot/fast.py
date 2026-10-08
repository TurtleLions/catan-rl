"""Drop-in speedups for Catanatron's hot paths (profiled in 4-player games).

Every replacement returns exactly what the original returns, in the same
summation order, so games replay move for move (checked by cbot/check_fast.py).
Installed on import of cbot.players; set CBOT_NOFAST=1 to run stock Catanatron.

  * Board.copy / State.copy: structural copies instead of pickle round-trips
    and copy.deepcopy.
  * production features: per-map table lookups instead of ~5M lru_cache calls
    per game (get_node_production's 1000-entry cache also thrashes).
  * reachability features: buildings/roads grouped by owner once per call,
    enemy sets by set difference instead of one full scan per player.
  * base_fn (AlphaBeta's and ValueFunction's heuristic): computes only what it
    uses -- production once (the original computes it twice), reachability for
    the scoring player at 0 and 1 roads (the original computes all players at
    0-2 roads), and reads the hand directly.
  * board-feature cache: reachability results are cached per board position.
    Boards carry a Zobrist fingerprint of their buildings and roads, updated by
    the three methods that change them; the cache hangs off the map object so
    it never outlives its game. Most leaves of a search tree share a board.
  * lazy legal moves: Game.execute no longer lists the next player's legal
    moves up front; Game.playable_actions computes them on first access. Search
    leaves are scored but never expanded, so most lists were never read.
"""

import os
import random
from collections import defaultdict

from catanatron import features as F
from catanatron import game as game_mod
from catanatron import state as state_mod
from catanatron.models import board as board_mod
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.players import value as value_mod

_orig = {}

# --- Board fingerprint (Zobrist hashing) ------------------------------------

_ZKEYS = {}


def _zkey(*key):
    z = _ZKEYS.get(key)
    if z is None:
        z = _ZKEYS[key] = random.Random(repr(key)).getrandbits(64)
    return z


def _board_zob(board):
    z = getattr(board, "_zob", None)
    if z is None:
        # Board built some other way (fresh or deserialized): fingerprint it.
        z = 0
        for n, (c, kind) in board.buildings.items():
            z ^= _zkey("B", n, c.value, kind)
        for e, c in board.roads.items():
            if e[0] < e[1]:
                z ^= _zkey("R", e, c.value)
        board._zob = z
    return z


def _build_settlement(self, color, node_id, initial_build_phase=False):
    z = _board_zob(self)
    out = _orig["build_settlement"](self, color, node_id, initial_build_phase)
    self._zob = z ^ _zkey("B", node_id, color.value, SETTLEMENT)
    return out


def _build_road(self, color, edge):
    z = _board_zob(self)
    out = _orig["build_road"](self, color, edge)
    self._zob = z ^ _zkey("R", (min(edge), max(edge)), color.value)
    return out


def _build_city(self, color, node_id):
    z = _board_zob(self)
    out = _orig["build_city"](self, color, node_id)
    self._zob = (
        z
        ^ _zkey("B", node_id, color.value, SETTLEMENT)
        ^ _zkey("B", node_id, color.value, CITY)
    )
    return out


# --- Lazy legal moves -------------------------------------------------------

_STALE = object()


def _get_playable(self):
    pa = self.__dict__.get("_pa", _STALE)
    if pa is _STALE:
        pa = self._pa = game_mod.generate_playable_actions(self.state)
    return pa


def _set_playable(self, value):
    self._pa = value


def _execute(self, action, validate_action=True, action_record=None):
    if validate_action and not game_mod.is_valid_action(
        self.playable_actions, self.state, action
    ):
        raise ValueError(
            f"{action} not playable right now. playable_actions={self.playable_actions}"
        )
    action_record = game_mod.apply_action(self.state, action, action_record)
    self._pa = _STALE
    return action_record


def _game_copy(self):
    g = game_mod.Game(players=[], initialize=False)
    g.seed = self.seed
    g.random = self.random
    g.id = self.id
    g.vps_to_win = self.vps_to_win
    g.friendly_robber = self.friendly_robber
    g.state = self.state.copy()
    # Same state, so the (possibly not yet computed) move list carries over.
    g._pa = self.__dict__.get("_pa", _STALE)
    return g


_CACHE_MAX = 200_000


def _board_cache(board):
    """Per-map cache dict (dies with the map, so entries never leak across games)."""
    m = board.map
    c = getattr(m, "_cbot_cache", None)
    if c is None or len(c) > _CACHE_MAX:
        c = m._cbot_cache = {}
    return c


def _board_copy(self):
    board = board_mod.Board(self.map, initialize=False)
    board.map = self.map
    board.buildings = self.buildings.copy()
    board.roads = self.roads.copy()
    board.connected_components = defaultdict(list, {
        c: [set(s) for s in comps] for c, comps in self.connected_components.items()
    })
    board.board_buildable_ids = self.board_buildable_ids.copy()
    board.road_lengths = self.road_lengths.copy()
    board.road_color = self.road_color
    board.road_length = self.road_length
    board._zob = _board_zob(self)
    board.robber_coordinate = self.robber_coordinate
    board.buildable_subgraph = self.buildable_subgraph
    # Cache values are never mutated in place (the caches are reset by
    # reassignment), so copying the dicts is enough.
    board.buildable_edges_cache = dict(self.buildable_edges_cache)
    board.player_port_resources_cache = dict(self.player_port_resources_cache)
    return board


_PROD = {}


def _prod_table(catan_map, robber):
    """node_id -> per-resource production, as get_node_production computes it."""
    key = (id(catan_map), robber)
    t = _PROD.get(key)
    if t is None:
        base = F.get_node_production.__wrapped__
        t = {
            n: tuple(base(catan_map, n, r, robber) for r in RESOURCES)
            for n in catan_map.adjacent_tiles
        }
        # Keep the map alive so its id can't be reused by another map.
        _PROD[key] = t = (catan_map, t)
    return t[1]


def _build_production_features(consider_robber):
    prefix = "EFFECTIVE_" if consider_robber else "TOTAL_"

    def production_features(game, p0_color):
        st = game.state
        board = st.board
        table = _prod_table(board.map, board.robber_coordinate if consider_robber else None)
        players = F.iter_players(st.colors, p0_color)
        owned = [
            (i, st.buildings_by_color[c][SETTLEMENT], st.buildings_by_color[c][CITY])
            for i, c in players
        ]
        features = {}
        for ri, resource in enumerate(RESOURCES):
            for i, settlements, cities in owned:
                production = 0
                for n in settlements:
                    production += table[n][ri]
                for n in cities:
                    production += 2 * table[n][ri]
                features[f"{prefix}P{i}_{resource}_PRODUCTION"] = production
        return features

    return production_features


def _owners(board):
    """(all occupied nodes, all roads, nodes by color, roads by color)."""
    nodes_by, roads_by = defaultdict(set), defaultdict(set)
    for k, v in board.buildings.items():
        if v is not None:
            nodes_by[v[0]].add(k)
    for k, c in board.roads.items():
        if c is not None:
            roads_by[c].add(k)
    all_nodes = set().union(*nodes_by.values())
    all_roads = set().union(*roads_by.values())
    return all_nodes, all_roads, nodes_by, roads_by


def _enemy_sets(owners, color):
    all_nodes, all_roads, nodes_by, roads_by = owners
    return (
        frozenset(all_nodes.difference(nodes_by.get(color, ()))),
        frozenset(all_roads.difference(roads_by.get(color, ()))),
    )


def _reachability_features(game, p0_color, levels=F.REACHABLE_FEATURES_MAX):
    """Cached by board fingerprint. Callers only read the returned dict."""
    board = game.state.board
    cache = _board_cache(board)
    key = ("reach", _board_zob(board), p0_color, game.state.colors, levels)
    out = cache.get(key)
    if out is None:
        out = cache[key] = _reachability_uncached(game, p0_color, levels)
    return out


def _reachability_uncached(game, p0_color, levels):
    features = {}
    board = game.state.board
    owners = _owners(board)

    board_buildable = board.buildable_node_ids(p0_color, True)
    for i, color in F.iter_players(game.state.colors, p0_color):
        owned_or_buildable = F.get_owned_or_buildable(game, color, board_buildable)

        zero_nodes = F.get_zero_nodes(game, color)
        production = F.count_production(
            frozenset(owned_or_buildable.intersection(zero_nodes)), board.map
        )
        for resource in RESOURCES:
            features[f"P{i}_0_ROAD_REACHABLE_{resource}"] = production[resource]

        enemy_nodes, enemy_roads = _enemy_sets(owners, color)
        for level, level_nodes, paths in F.iter_level_nodes(
            enemy_nodes, enemy_roads, levels, frozenset(zero_nodes)
        ):
            production = F.count_production(
                frozenset(owned_or_buildable.intersection(level_nodes)), board.map
            )
            for resource in RESOURCES:
                features[f"P{i}_{level}_ROAD_REACHABLE_{resource}"] = production[resource]

    return features


def _reach01(game, p0_color):
    """base_fn's reachable production at 0 and 1 roads for p0, cached by board."""
    board = game.state.board
    cache = _board_cache(board)
    key = ("r01", _board_zob(board), p0_color)
    out = cache.get(key)
    if out is not None:
        return out
    board_buildable = board.buildable_node_ids(p0_color, True)
    owned_or_buildable = F.get_owned_or_buildable(game, p0_color, board_buildable)
    zero_nodes = F.get_zero_nodes(game, p0_color)
    prod = F.count_production(
        frozenset(owned_or_buildable.intersection(zero_nodes)), board.map
    )
    r0 = sum([prod[r] for r in RESOURCES])
    enemy_nodes, enemy_roads = _enemy_sets(_owners(board), p0_color)
    _, level_nodes, _ = F.iter_level_nodes(
        enemy_nodes, enemy_roads, 2, frozenset(zero_nodes)
    )[0]
    prod = F.count_production(
        frozenset(owned_or_buildable.intersection(level_nodes)), board.map
    )
    r1 = sum([prod[r] for r in RESOURCES])
    out = cache[key] = (r0, r1)
    return out


def _base_fn(params=value_mod.DEFAULT_WEIGHTS):
    production_features = _build_production_features(True)

    def fn(game, p0_color):
        state = game.state
        board = state.board
        sample = production_features(game, p0_color)
        production = value_mod.value_production(sample, "P0")
        enemy_production = value_mod.value_production(sample, "P1", False)

        key = player_key(state, p0_color)
        longest_road_length = get_longest_road_length(state, p0_color)

        reachable_production_at_zero, reachable_production_at_one = _reach01(game, p0_color)

        ps = state.player_state
        distance_to_city = (
            max(2 - ps[f"{key}_WHEAT_IN_HAND"], 0)
            + max(3 - ps[f"{key}_ORE_IN_HAND"], 0)
        ) / 5.0
        distance_to_settlement = (
            max(1 - ps[f"{key}_WHEAT_IN_HAND"], 0)
            + max(1 - ps[f"{key}_SHEEP_IN_HAND"], 0)
            + max(1 - ps[f"{key}_BRICK_IN_HAND"], 0)
            + max(1 - ps[f"{key}_WOOD_IN_HAND"], 0)
        ) / 4.0
        hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

        num_in_hand = player_num_resource_cards(state, p0_color)
        discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0

        buildings = state.buildings_by_color[p0_color]
        owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
        owned_tiles = set()
        for n in owned_nodes:
            owned_tiles.update(board.map.adjacent_tiles[n])
        num_tiles = len(owned_tiles)

        num_buildable_nodes = len(board.buildable_node_ids(p0_color))
        longest_road_factor = params["longest_road"] if num_buildable_nodes == 0 else 0.1

        return float(
            ps[f"{key}_VICTORY_POINTS"] * params["public_vps"]
            + production * params["production"]
            + enemy_production * params["enemy_production"]
            + reachable_production_at_zero * params["reachable_production_0"]
            + reachable_production_at_one * params["reachable_production_1"]
            + hand_synergy * params["hand_synergy"]
            + num_buildable_nodes * params["buildable_nodes"]
            + num_tiles * params["num_tiles"]
            + num_in_hand * params["hand_resources"]
            + discard_penalty
            + longest_road_length * longest_road_factor
            + player_num_dev_cards(state, p0_color) * params["hand_devs"]
            + get_played_dev_cards(state, p0_color, "KNIGHT") * params["army_size"]
        )

    return fn


def install():
    if _orig or os.environ.get("CBOT_NOFAST"):
        return
    _orig["Board.copy"] = board_mod.Board.copy
    _orig["State.copy"] = orig_state_copy = state_mod.State.copy

    # State.copy pickles buildings_by_color; run the original with a cheap
    # stand-in for that field, then set it properly.
    def _state_copy_fast(self):
        bbc = self.buildings_by_color
        self.buildings_by_color = None
        try:
            s = orig_state_copy(self)
        finally:
            self.buildings_by_color = bbc
        s.buildings_by_color = {
            c: defaultdict(list, {k: list(v) for k, v in d.items()})
            for c, d in bbc.items()
        }
        return s

    for name in ("build_settlement", "build_road", "build_city"):
        _orig[name] = getattr(board_mod.Board, name)
    board_mod.Board.build_settlement = _build_settlement
    board_mod.Board.build_road = _build_road
    board_mod.Board.build_city = _build_city
    board_mod.Board.copy = _board_copy
    game_mod.Game.playable_actions = property(_get_playable, _set_playable)
    game_mod.Game.execute = _execute
    game_mod.Game.copy = _game_copy
    state_mod.State.copy = _state_copy_fast
    F.build_production_features = _build_production_features
    value_mod.build_production_features = _build_production_features
    F.reachability_features = _reachability_features
    value_mod.reachability_features = _reachability_features
    value_mod.base_fn = _base_fn


install()
