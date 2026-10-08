"""State -> fixed-length float32 vector, from one player's perspective (P0).

Only uses information P0 could know at the table: own hand in full, opponent
hand/dev totals and played cards, and the public board. (Catanatron's own
search bots read the full state; this encoder does not.)
"""

import functools

import numpy as np

from catanatron import features as F
from catanatron.models.enums import ActionPrompt, RESOURCES
from catanatron.models.map import number_probability

_production = F.build_production_features(True)
_PROMPTS = list(ActionPrompt)
NUM_NODES = 54


@functools.lru_cache(maxsize=64)
def _node_pips(catan_map):
    pips = np.zeros(NUM_NODES, dtype=np.float32)
    for node_id, tiles in catan_map.adjacent_tiles.items():
        if node_id < NUM_NODES:
            pips[node_id] = sum(
                number_probability(t.number) for t in tiles if t.resource is not None
            )
    return pips


def _sample(game, p0):
    s = F.create_sample(game, p0)
    s.update(_production(game, p0))
    s.update(F.reachability_features(game, p0, 2))
    st = game.state
    s["X_MY_TURN"] = st.current_color() == p0
    s["X_I_AM_FIRST"] = st.colors[0] == p0
    s["X_NUM_TURNS"] = st.num_turns / 100.0
    s["X_INITIAL_PHASE"] = st.is_initial_build_phase
    for p in _PROMPTS:
        s[f"X_PROMPT_{p.name}"] = st.current_prompt == p
    for i, color in F.iter_players(st.colors, p0):
        s[f"X_P{i}_BUILDABLE"] = len(st.board.buildable_node_ids(color)) / 10.0
    return s


@functools.lru_cache(maxsize=4)
def feature_names():
    from catanatron import Color, Game
    from catanatron.models.player import RandomPlayer

    g = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=0)
    return tuple(sorted(_sample(g, Color.RED).keys()))


NUM_OPP = len(RESOURCES)


def _opp_hand(game, p0):
    """Opponent resource-hand composition (what a perfect card counter knows)."""
    st = game.state
    opp = [c for c in st.colors if c != p0][0]
    key = f"P{st.color_to_index[opp]}"
    return np.array(
        [st.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES], dtype=np.float32
    )


def encode(game, p0):
    """Layout: [sorted feature_names() | 54 node pips | 5 opponent hand]. The
    opponent-hand block is appended last so models trained on the older,
    shorter vectors (which store explicit column indices) still work."""
    s = _sample(game, p0)
    names = feature_names()
    base = np.fromiter((float(s[k]) for k in names), dtype=np.float32, count=len(names))
    return np.concatenate([base, _node_pips(game.state.board.map), _opp_hand(game, p0)])


def dim():
    return len(feature_names()) + NUM_NODES + NUM_OPP


# --- Features derived from the full vector (usable on stored data too) ------

# Base-map port slots are fixed; only their type is shuffled.
PORT_NODES = [(25, 26), (28, 29), (32, 33), (35, 36), (38, 39), (40, 44), (45, 47), (48, 49), (52, 53)]
PORT_TYPES = list(RESOURCES) + ["THREE_TO_ONE"]
NUM_DERIVED = 2 * len(PORT_TYPES) + len(RESOURCES)


@functools.lru_cache(maxsize=1)
def _derive_index():
    idx = {n: i for i, n in enumerate(feature_names())}
    port_type = np.array(
        [[idx[f"PORT{p}_IS_{t}"] for t in PORT_TYPES] for p in range(len(PORT_NODES))]
    )  # (9, 6)
    owner = np.array(
        [
            [
                [idx[f"NODE{n}_P{i}_SETTLEMENT"], idx[f"NODE{n}_P{i}_CITY"]]
                for n in nodes
            ]
            for nodes in PORT_NODES
            for i in (0, 1)
        ]
    ).reshape(len(PORT_NODES), 2, 2, 2)  # port, player, node, building
    tiles = sorted({int(n[4:].split("_")[0]) for n in idx if n.startswith("TILE")})
    tile_res = np.array([[idx[f"TILE{t}_IS_{r}"] for r in RESOURCES] for t in tiles])
    tile_p = np.array([idx[f"TILE{t}_PROBA"] for t in tiles])
    return port_type, owner, tile_res, tile_p


def derive(X):
    """(n, dim) full vectors -> (n, NUM_DERIVED): per-player port ownership by
    type, and total board production probability per resource."""
    port_type, owner, tile_res, tile_p = _derive_index()
    X = X.astype(np.float32, copy=False)
    has = X[:, owner].reshape(len(X), len(PORT_NODES), 2, 4).max(-1)  # n, port, player
    types = X[:, port_type]  # n, port, type
    own = np.einsum("npi,npt->nit", has, types).clip(0, 1).reshape(len(X), -1)
    board = np.einsum("ntr,nt->nr", X[:, tile_res], X[:, tile_p])
    return np.concatenate([own, board], axis=1)
