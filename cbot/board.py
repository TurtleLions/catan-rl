"""A board-shaped value net: message passing over the 54 corners of the map.

The plain MLP on the raw board one-hots memorizes games: the tile layout
identifies the game, so it learns "this map -> this winner". Here every
corner is described by what is on and around it (buildings, port, the
production of its tiles, robber, roads) and the same weights apply at every
corner, which never learns its own index. Rounds of messages along the 72
edges carry information between neighbors; the corners are then pooled
(mean, max, mine, the opponent's) and joined with the compact features.
Nothing depends on where a corner is, so a rotated or mirrored board gets the
same value.

Inputs are the full 1v1 vectors (cbot.feats.encode), so the stored shards
train it as they are. Training is in torch (cbot.train --arch board); search
uses the numpy forward pass in BoardNet.
"""

import functools

import numpy as np

from cbot import feats

RES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
NODE_DIM = 24  # see node_inputs
EDGE_DIM = 2


@functools.lru_cache(maxsize=1)
def topology():
    """Column indices into the full vector and the fixed adjacency of the base
    map (the same for every game; only tile/port contents are shuffled)."""
    from catanatron import Color, Game
    from catanatron.models.player import RandomPlayer

    idx = {n: i for i, n in enumerate(feats.feature_names())}
    m = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=0).state.board.map
    n_nodes, n_tiles = feats.NUM_NODES, len(m.tiles_by_id)
    t = {}
    t["bld"] = np.array(
        [[idx[f"NODE{v}_P{p}_{b}"] for p in (0, 1) for b in ("SETTLEMENT", "CITY")] for v in range(n_nodes)]
    )
    t["port"] = np.array([[idx[f"PORT{p}_IS_{x}"] for x in feats.PORT_TYPES] for p in range(len(feats.PORT_NODES))])
    node_port = np.zeros((n_nodes, len(feats.PORT_NODES)), np.float32)
    for p, nodes in enumerate(feats.PORT_NODES):
        node_port[list(nodes), p] = 1
    t["node_port"] = node_port
    t["tile_res"] = np.array([[idx[f"TILE{i}_IS_{r}"] for r in RES] for i in range(n_tiles)])
    t["tile_p"] = np.array([idx[f"TILE{i}_PROBA"] for i in range(n_tiles)])
    t["tile_rob"] = np.array([idx[f"TILE{i}_HAS_ROBBER"] for i in range(n_tiles)])
    node_tile = np.zeros((n_nodes, n_tiles), np.float32)
    for i, tile in m.tiles_by_id.items():
        node_tile[[v for v in tile.nodes.values() if v < n_nodes], i] = 1
    t["node_tile"] = node_tile
    edges = sorted({tuple(int(x) for x in n[5:].split(")")[0].split(", ")) for n in idx if n.startswith("EDGE(")})
    t["edge_a"] = np.array([a for a, _ in edges])
    t["edge_b"] = np.array([b for _, b in edges])
    t["edge_road"] = np.array([[idx[f"EDGE({a}, {b})_P{p}_ROAD"] for p in (0, 1)] for a, b in edges])
    inc = [[] for _ in range(n_nodes)]
    for e, (a, b) in enumerate(edges):
        inc[a].append(e)
        inc[b].append(e)
    pad = len(edges)  # index of an all-zero row appended to the edge array
    t["node_edges"] = np.array([es + [pad] * (3 - len(es)) for es in inc])
    t["compact"] = np.array(
        [i for i, n in enumerate(feats.feature_names()) if not n.startswith(("NODE", "EDGE", "TILE", "PORT"))]
    )
    return t


def node_inputs(X, xp=np):
    """(n, full dim) -> per-corner inputs (n, 54, NODE_DIM) and per-edge road
    inputs (n, 72, 2). `xp` is numpy or torch (the same code serves both)."""
    t = topology()
    if xp is np:
        c, cat, ein = (lambda a: a), np.concatenate, np.einsum
    else:
        c, cat, ein = (lambda a: xp.as_tensor(a).to(X.device)), xp.cat, xp.einsum
    n = len(X)
    bld = X[:, c(t["bld"])]  # mine S, C, theirs S, C
    port = ein("np,bpt->bnt", c(t["node_port"]).to(X.dtype) if xp is not np else t["node_port"], X[:, c(t["port"])])
    prod = X[:, c(t["tile_res"])] * X[:, c(t["tile_p"])][:, :, None] * 10.0  # (n, tiles, 5)
    rob = X[:, c(t["tile_rob"])]
    nt = c(t["node_tile"]).to(X.dtype) if xp is not np else t["node_tile"]
    near = ein("nt,btr->bnr", nt, prod)
    robbed = ein("nt,btr->bnr", nt, prod * rob[:, :, None])
    rob_near = ein("nt,bt->bn", nt, rob)[:, :, None]
    land = ein("nt,bt->bn", nt, rob * 0 + 1)[:, :, None] / 3.0  # land tiles around the corner
    roads = X[:, c(t["edge_road"])]  # (n, 72, 2)
    padded = cat([roads, roads[:, :1] * 0], 1)
    node_roads = padded[:, c(t["node_edges"])].sum(2)
    return cat([bld, port, near, robbed, rob_near, land, node_roads], -1), roads


class BoardNet:
    """Numpy forward pass of an exported board net (same interface as
    cbot.search.ValueNet)."""

    players = 2

    def __init__(self, path):
        d = np.load(path)
        self.w = {k: d[k].astype(np.float32) for k in d.files if k not in ("kind",)}
        self.rounds = int(self.w.pop("rounds"))
        self.head = int(self.w.pop("head_layers"))
        t = topology()
        self.ea, self.eb, self.ne = t["edge_a"], t["edge_b"], t["node_edges"]
        self.compact = t["compact"]

    def logits(self, X):
        w, relu = self.w, (lambda a: np.maximum(a, 0, out=a))
        X = X.astype(np.float32, copy=False)
        nodes, roads = node_inputs(X)
        own = nodes[:, :, 0] + nodes[:, :, 1]
        opp = nodes[:, :, 2] + nodes[:, :, 3]
        h = relu((nodes - w["node_mean"]) / w["node_std"] @ w["in_W"] + w["in_b"])
        zero = np.zeros((len(X), 1, h.shape[2]), np.float32)
        for r in range(self.rounds):
            e = relu((h[:, self.ea] + h[:, self.eb]) @ w[f"e{r}_W"] + roads @ w[f"r{r}_W"] + w[f"e{r}_b"])
            agg = np.concatenate([e, zero], 1)[:, self.ne].sum(2)
            h = h + relu(h @ w[f"h{r}_W"] + agg @ w[f"m{r}_W"] + w[f"h{r}_b"])
        pooled = np.concatenate(
            [h.mean(1), h.max(1), np.einsum("bn,bnh->bh", own, h) / 4.0, np.einsum("bn,bnh->bh", opp, h) / 4.0], 1
        )
        g = (X[:, self.compact] - w["mean"]) / w["std"]
        z = np.concatenate([pooled, g], 1)
        for i in range(self.head):
            z = z @ w[f"f{i}_W"] + w[f"f{i}_b"]
            if i < self.head - 1:
                relu(z)
        return z

    def __call__(self, X):
        return 1.0 / (1.0 + np.exp(-np.clip(self.logits(X)[:, 0], -30, 30)))


def is_board_net(path):
    return "kind" in np.load(path).files


# --- torch ------------------------------------------------------------------


def build(mean, std, node_mean, node_std, hidden=32, rounds=3, head=(256, 128), dropout=0.1):
    import torch
    import torch.nn as nn

    t = topology()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rounds = rounds
            for k, v in (("mean", mean), ("std", std), ("node_mean", node_mean), ("node_std", node_std)):
                self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
            for k in ("edge_a", "edge_b", "node_edges", "compact"):
                self.register_buffer(k, torch.as_tensor(t[k], dtype=torch.long), persistent=False)
            self.inp = nn.Linear(NODE_DIM, hidden)
            self.e = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(rounds))
            self.r = nn.ModuleList(nn.Linear(EDGE_DIM, hidden, bias=False) for _ in range(rounds))
            self.h = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(rounds))
            self.m = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(rounds))
            layers, d = [], 4 * hidden + len(t["compact"])
            for k in head:
                layers += [nn.Linear(d, k), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
                d = k
            layers.append(nn.Linear(d, 1))
            self.head = nn.Sequential(*layers)

        def forward(self, X):
            relu = torch.relu
            nodes, roads = node_inputs(X, torch)
            own = nodes[:, :, 0] + nodes[:, :, 1]
            opp = nodes[:, :, 2] + nodes[:, :, 3]
            h = relu(self.inp((nodes - self.node_mean) / self.node_std))
            zero = h.new_zeros(len(X), 1, h.shape[2])
            for r in range(self.rounds):
                e = relu(self.e[r](h[:, self.edge_a] + h[:, self.edge_b]) + self.r[r](roads))
                agg = torch.cat([e, zero], 1)[:, self.node_edges].sum(2)
                h = h + relu(self.h[r](h) + self.m[r](agg))
            pooled = torch.cat(
                [h.mean(1), h.amax(1), torch.einsum("bn,bnh->bh", own, h) / 4.0,
                 torch.einsum("bn,bnh->bh", opp, h) / 4.0], 1
            )
            g = (X[:, self.compact] - self.mean) / self.std
            return self.head(torch.cat([pooled, g], 1))

    return Net()


def node_stats(X, rows=100_000):
    """Mean/std of the per-corner inputs and of the compact features."""
    Xs = X[:rows].astype(np.float32)
    nodes, _ = node_inputs(Xs)
    flat = nodes.reshape(-1, NODE_DIM)
    g = Xs[:, topology()["compact"]]
    return g.mean(0), g.std(0) + 1e-3, flat.mean(0), flat.std(0) + 1e-3


def export(model, path):
    import torch.nn as nn

    arrs = {"kind": np.array("board"), "rounds": np.array(model.rounds)}
    for k in ("mean", "std", "node_mean", "node_std"):
        arrs[k] = getattr(model, k).numpy()
    lin = lambda m: (m.weight.detach().numpy().T, None if m.bias is None else m.bias.detach().numpy())  # noqa: E731
    arrs["in_W"], arrs["in_b"] = lin(model.inp)
    for r in range(model.rounds):
        arrs[f"e{r}_W"], arrs[f"e{r}_b"] = lin(model.e[r])
        arrs[f"r{r}_W"] = lin(model.r[r])[0]
        arrs[f"h{r}_W"], arrs[f"h{r}_b"] = lin(model.h[r])
        arrs[f"m{r}_W"] = lin(model.m[r])[0]
    heads = [m for m in model.head if isinstance(m, nn.Linear)]
    for i, m in enumerate(heads):
        arrs[f"f{i}_W"], arrs[f"f{i}_b"] = lin(m)
    arrs["head_layers"] = np.array(len(heads))
    np.savez(path + ".tmp.npz", **arrs)
    import os

    os.replace(path + ".tmp.npz", path)
