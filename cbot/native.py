"""The native 1v1 engine (cbot/engine, built into cbot._engine) seen from
Catanatron: a Catanatron game converted to a native one, native move numbers
mapped back to Catanatron actions, and native features in cbot.feats order.

    from cbot import native
    g = native.from_game(game)            # same position, natively
    g.legal(), g.apply(move), g.encode(seat)

Build the module with cbot/engine/build.sh.
"""

import numpy as np

from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES, SETTLEMENT, CITY, ActionPrompt
from catanatron.models.board import get_edges

from cbot import _engine, feats, policy

DIM = _engine.DIM
_PROMPTS = list(ActionPrompt)
_EDGES = get_edges()
_DEVS = DEVELOPMENT_CARDS  # KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT
_OWNED = DEVELOPMENT_CARDS[:4]


def canonical_names():
    """Feature names in the order catan.h's encode_canonical writes them."""
    n = []
    n.append("P0_ACTUAL_VPS")
    for i in range(2):
        n += [f"P{i}_{k}" for k in ("PUBLIC_VPS", "HAS_ARMY", "HAS_ROAD", "ROADS_LEFT", "SETTLEMENTS_LEFT",
                                    "CITIES_LEFT", "HAS_ROLLED", "LONGEST_ROAD_LENGTH")]
    n += [f"P0_{r}_IN_HAND" for r in RESOURCES] + [f"P0_{d}_IN_HAND" for d in _DEVS]
    n.append("P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN")
    for i in range(2):
        n += [f"P{i}_{d}_PLAYED" for d in _DEVS[:4]] + [f"P{i}_NUM_RESOURCES_IN_HAND", f"P{i}_NUM_DEVS_IN_HAND"]
    for t in range(19):
        n += [f"TILE{t}_IS_{r}" for r in RESOURCES] + [f"TILE{t}_IS_DESERT", f"TILE{t}_PROBA", f"TILE{t}_HAS_ROBBER"]
    for p in range(9):
        n += [f"PORT{p}_IS_{r}" for r in RESOURCES] + [f"PORT{p}_IS_THREE_TO_ONE"]
    for i in range(2):
        for node in range(54):
            n += [f"NODE{node}_P{i}_SETTLEMENT", f"NODE{node}_P{i}_CITY"]
        n += [f"EDGE{e}_P{i}_ROAD" for e in _EDGES]
    n += ["BANK_DEV_CARDS", "IS_MOVING_ROBBER", "IS_DISCARDING"] + [f"BANK_{r}" for r in RESOURCES]
    n += [f"EFFECTIVE_P{i}_{r}_PRODUCTION" for r in RESOURCES for i in range(2)]
    n += [f"P{i}_{lv}_ROAD_REACHABLE_{r}" for i in range(2) for lv in range(3) for r in RESOURCES]
    n += ["X_MY_TURN", "X_I_AM_FIRST", "X_NUM_TURNS", "X_INITIAL_PHASE"]
    n += [f"X_PROMPT_{p.name}" for p in _PROMPTS]
    n += [f"X_P{i}_BUILDABLE" for i in range(2)]
    return n


def _install_layout():
    names = canonical_names()
    order = {k: i for i, k in enumerate(feats.feature_names())}
    assert sorted(names) == sorted(order), "native features differ from cbot.feats"
    perm = [order[k] for k in names] + list(range(len(names), DIM))
    _engine.set_layout(np.array(perm, dtype=np.int32))


_install_layout()


def _res(r):
    return -1 if r is None else RESOURCES.index(r)


def to_dict(game):
    """The native state dict (catan.h State) of a 1v1 Catanatron game."""
    st = game.state
    assert len(st.colors) == 2, "the native engine is 1v1 only"
    assert st.current_prompt not in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES)
    seat = {c: i for i, c in enumerate(st.colors)}
    board = st.board
    m = board.map
    ps = st.player_state
    players = []
    for i, c in enumerate(st.colors):
        k = f"P{i}_"
        players.append({
            "vp": ps[k + "VICTORY_POINTS"], "avp": ps[k + "ACTUAL_VICTORY_POINTS"],
            "roads_av": ps[k + "ROADS_AVAILABLE"], "setts_av": ps[k + "SETTLEMENTS_AVAILABLE"],
            "cities_av": ps[k + "CITIES_AVAILABLE"], "has_road": ps[k + "HAS_ROAD"],
            "has_army": ps[k + "HAS_ARMY"], "has_rolled": ps[k + "HAS_ROLLED"],
            "played_dev": ps[k + "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
            "lr_len": ps[k + "LONGEST_ROAD_LENGTH"],
            "owned_start": [bool(ps[f"{k}{d}_OWNED_AT_START"]) for d in _OWNED],
            "res": [ps[f"{k}{r}_IN_HAND"] for r in RESOURCES],
            "dev": [ps[f"{k}{d}_IN_HAND"] for d in _DEVS],
            "played": [ps[f"{k}PLAYED_{d}"] for d in _DEVS],
            "setts": list(st.buildings_by_color[c][SETTLEMENT]),
            "cities": list(st.buildings_by_color[c][CITY]),
        })
    bowner, btype = [-1] * 54, [0] * 54
    for n, (c, kind) in board.buildings.items():
        bowner[n], btype[n] = seat[c], 1 if kind == SETTLEMENT else 2
    road = [seat[board.roads[e]] if e in board.roads else -1 for e in _EDGES]
    return {
        "map": {
            "res": [_res(m.tiles_by_id[t].resource) for t in range(19)],
            "num": [m.tiles_by_id[t].number or 0 for t in range(19)],
            "port": [_res(m.ports_by_id[p].resource) for p in range(9)],
        },
        "players": players,
        "bowner": bowner, "btype": btype, "road": road,
        "comps": [[sorted(s) for s in board.connected_components.get(c, [])] for c in st.colors],
        "buildable": sorted(board.board_buildable_ids),
        "rl": [(seat[c], n) for c, n in board.road_lengths.items()],
        "road_color": -1 if board.road_color is None else seat[board.road_color],
        "road_length": board.road_length,
        "robber": m.land_tiles[board.robber_coordinate].id,
        "bank": list(st.resource_freqdeck),
        "deck": [_DEVS.index(d) for d in st.development_listdeck],
        "num_turns": st.num_turns, "cur": st.current_player_index, "turn": st.current_turn_index,
        "prompt": _PROMPTS.index(st.current_prompt),
        "initial": st.is_initial_build_phase, "discarding": st.is_discarding,
        "moving_knight": st.is_moving_knight, "road_building": st.is_road_building,
        "free_roads": st.free_roads_available, "discard": list(st.discard_counts),
        "vps_to_win": game.vps_to_win, "discard_limit": st.discard_limit,
    }


def from_game(game, seed=1):
    return _engine.Game.from_dict(to_dict(game), seed)


def action_of(game, move):
    """The Catanatron action (of game.playable_actions) with native number `move`."""
    for a in game.playable_actions:
        if policy.index(a) == move:
            return a
    raise ValueError(f"move {move} is not playable")


def outcome_of(record):
    """The native outcome of a Catanatron ActionRecord (dice sum, card, stolen resource)."""
    from catanatron.models.enums import ActionType

    t, r = record.action.action_type, record.result
    if t == ActionType.ROLL:
        return r[0] + r[1]
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        return _DEVS.index(r)
    if t == ActionType.MOVE_ROBBER and r is not None:
        return RESOURCES.index(r)
    return -1


# --- Nets and the tree-search player ----------------------------------------

_NETS = {}


def _mlp(net):
    """A cbot.search.ValueNet (or PolicyNet) as the native MLP description."""
    if net.derived:
        raise ValueError("nets with derived features are not supported natively")
    cols = np.arange(DIM) if net.cols is None else net.cols
    return {"cols": [int(c) for c in cols], "layers": [(W, b) for W, b in net.layers]}


def get_nets(model_path, policy_path=None):
    """Native nets for a value model (`a.npz+b.npz` committee) and a policy."""
    key = (model_path, policy_path)
    if key not in _NETS:
        from cbot.search import Ensemble, get_net
        from cbot.policy import get_policy

        v = get_net(model_path)
        members = v.nets if isinstance(v, Ensemble) else [v]
        if any(m.players != 2 or not hasattr(m, "layers") for m in members):
            raise ValueError("native search needs 1v1 MLP value nets")
        pol = _mlp(get_policy(policy_path)) if policy_path else None
        _NETS[key] = _engine.Nets([_mlp(m) for m in members], pol)
    return _NETS[key]


_PARAMS = ("sims", "cpuct", "batch", "ptemp", "pmix", "fpu", "noise", "alpha", "pick", "initf", "fair")


def search_params(kw):
    return {k: kw[k] for k in _PARAMS if k in kw}


from cbot.mcts import MCTSPlayer  # noqa: E402


class NativeMCTSPlayer(MCTSPlayer):
    """cbot.mcts.MCTSPlayer's search run by the native engine (spec NMCTS:...).
    Same parameters and defaults; trade prompts (from human players) are
    answered by the Python player."""

    LABEL = "NativeMCTS"

    def __init__(self, color, model_path, **kw):
        kw.setdefault("initf", 2)
        super().__init__(color, model_path, **kw)
        prm = {k: getattr(self, k) for k in ("sims", "cpuct", "batch", "ptemp", "pmix", "fpu", "noise", "alpha", "pick")}
        prm.update(initf=self.initmode, fair=self.fair)
        if self.initmode not in (0, 2):
            raise ValueError("native search supports initf=0 or 2")
        self.native = _engine.Searcher(get_nets(model_path, kw.get("policy")), prm, self.rng.getrandbits(63))

    def decide(self, game, playable_actions):
        if game.state.current_prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
            return super().decide(game, playable_actions)
        self.last_value = self.last_scores = self.last_vec = self.last_visits = None
        if len(playable_actions) == 1:
            return playable_actions[0]
        me = game.state.colors.index(self.color)
        d = self.native.decide(from_game(game), me)
        acts = {policy.index(a): a for a in playable_actions}
        self.last_vec = np.array(d["vec"])
        self.last_value = float(self.last_vec[me])
        self.last_scores = [(acts[m], q) for m, q in zip(d["moves"], d["q"])]
        if any(n >= 0 for n in d["visits"]):
            self.last_visits = [(acts[m], n) for m, n in zip(d["moves"], d["visits"])]
        return acts[d["move"]]
