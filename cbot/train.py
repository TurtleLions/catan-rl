"""Train the P(win) value MLP on generated shards.

    python -m cbot.train --data data/r0 [data/r1 ...] --out models/v0.npz

Saves numpy weights + feature normalization for torch-free inference in search.
Multiplayer shards (2-D `v`) train a softmax over seats relative to the
perspective player instead of a single P(win) logit.
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn


def load(dirs, val_frac=0.05, seed=0, lam=1.0, project=None, searched_only=False):
    files = sorted(f for d in dirs for f in glob.glob(f"{d}/*.npz") if not f.endswith(".tmp.npz"))
    # Targets first, then the features into one preallocated float16 matrix:
    # concatenating per-shard parts would need twice the memory at the end.
    keeps, ys = [], []
    players = 2
    for f in files:
        d = np.load(f)
        v = d["v"] if "v" in d else None
        keep = slice(None)
        if searched_only:
            # Only states the generator searched (both seats' rows have a value).
            if v is None:
                keeps.append(None)
                continue
            keep = ~np.isnan(v if v.ndim == 1 else v[:, 0])
            v = v[keep]
        keeps.append(keep)
        if v is not None and v.ndim == 2:
            # Multiplayer: y = winner seat relative to this row's player.
            players = v.shape[1]
            y = np.eye(players, dtype=np.float32)[d["y"][keep]]
            if lam < 1.0:
                has = ~np.isnan(v[:, 0])
                y[has] = lam * y[has] + (1 - lam) * v[has]
        else:
            y = d["y"][keep].astype(np.float32)
            if lam < 1.0 and v is not None:
                # Soft target: blend outcome with the generator's search value.
                y = np.where(np.isnan(v), y, lam * y + (1 - lam) * v).astype(np.float32)
        ys.append(y)
    y = np.concatenate(ys).astype(np.float32)
    X, i = None, 0
    for f, keep in zip(files, keeps):
        if keep is None:
            continue
        part = np.load(f)["X"][keep]
        part = part if project is None else project(part)
        if X is None:
            X = np.empty((len(y), part.shape[1]), np.float16)
        X[i : i + len(part)] = part
        i += len(part)
    assert i == len(y)
    # Each state is stored once per player in consecutive rows; split off the
    # tail in whole blocks so validation states are not mirrored in training.
    n = len(X)
    nval = int(n * val_frac) // players * players
    return X[: n - nval], y[: n - nval], X[n - nval :], y[n - nval :]


def select_cols(kind, players=2):
    from cbot import feats

    if players > 2:
        # Multiplayer shards already hold only the compact features.
        assert kind == "compact", kind
        return np.arange(feats.dim(players))
    names = (
        list(feats.feature_names())
        + [f"PIP{i}" for i in range(feats.NUM_NODES)]
        + [f"OPP_{r}_IN_HAND" for r in range(feats.NUM_OPP)]
    )
    if kind == "all":
        return np.arange(len(names))
    board = ("NODE", "EDGE", "TILE", "PORT", "PIP")
    if kind in ("compact", "compact+"):
        return np.array(
            [i for i, n in enumerate(names) if not n.startswith(board + ("OPP_",))]
        )
    if kind == "compact+opp":
        return np.array([i for i, n in enumerate(names) if not n.startswith(board)])
    if kind == "board":
        # Everything public: the compact features plus the raw board.
        return np.array([i for i, n in enumerate(names) if not n.startswith("OPP_")])
    raise ValueError(kind)


def build(dim, hidden, dropout=0.0, out=1):
    layers, d = [], dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def export(model, mean, std, path, hidden, cols, derived=False, players=2):
    Ws, bs = [], []
    for m in model:
        if isinstance(m, nn.Linear):
            Ws.append(m.weight.detach().numpy().T.astype(np.float32))
            bs.append(m.bias.detach().numpy().astype(np.float32))
    arrs = {
        "mean": mean, "std": std, "cols": cols,
        "derived": np.array(int(derived)), "players": np.array(players),
    }
    for i, (W, b) in enumerate(zip(Ws, bs)):
        arrs[f"W{i}"] = W
        arrs[f"b{i}"] = b
    np.savez(path + ".tmp.npz", **arrs)
    os.replace(path + ".tmp.npz", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", default="512,256,128")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--init", default=None, help="warm start from .pt")
    ap.add_argument("--cols", default="compact")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--searched-only", action="store_true",
                    help="drop states without a search value (for --lam 0)")
    ap.add_argument("--arch", default="mlp", choices=("mlp", "board"),
                    help="board: message passing over the map's corners (cbot.board)")
    ap.add_argument("--node-hidden", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    if args.arch == "board":
        args.cols = "board"
    torch.set_num_threads(args.threads)
    hidden = [int(h) for h in args.hidden.split(",")]

    t0 = time.time()
    cols = select_cols(args.cols, args.players)
    multi = args.players > 2
    use_derived = args.cols == "compact+"

    def project(X, chunk=500_000):
        out = []
        for i in range(0, len(X), chunk):
            part = X[i : i + chunk]
            sel = part[:, cols]
            if use_derived:
                from cbot import feats

                sel = np.concatenate([sel, feats.derive(part).astype(np.float16)], axis=1)
            out.append(sel)
        return np.concatenate(out)

    # Project each shard as it loads so the full 721-wide matrix never sits in memory.
    Xtr, ytr, Xva, yva = load(args.data, lam=args.lam, project=project,
                                  searched_only=args.searched_only)
    assert (ytr.ndim == 2) == multi, "--players does not match the data"
    print(f"loaded train {Xtr.shape} val {Xva.shape} pos {ytr.mean(0)} in {time.time() - t0:.0f}s", flush=True)
    if args.arch == "board":
        from cbot import board

        # The board net normalizes its own inputs; it takes raw vectors.
        model = board.build(*board.node_stats(Xtr), hidden=args.node_hidden, rounds=args.rounds,
                            head=hidden, dropout=args.dropout)
        mean, std = np.zeros(Xtr.shape[1], np.float32), np.ones(Xtr.shape[1], np.float32)
    else:
        mean = Xtr[:200000].astype(np.float32).mean(0)
        std = Xtr[:200000].astype(np.float32).std(0) + 1e-3
        model = build(Xtr.shape[1], hidden, args.dropout, out=args.players if multi else 1)
    print(f"{sum(p.numel() for p in model.parameters())} parameters", flush=True)

    yva_t = torch.from_numpy(yva)
    if args.init:
        model.load_state_dict(torch.load(args.init))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (len(Xtr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    mean_t, std_t = torch.from_numpy(mean), torch.from_numpy(std)
    if multi:
        def lossf(logits, t):
            return -(t * torch.log_softmax(logits, 1)).sum(1).mean()

        def accf(logits, t):
            return (logits.argmax(1) == t.argmax(1)).float().mean()
    else:
        bce = nn.BCEWithLogitsLoss()

        def lossf(logits, t):
            return bce(logits.squeeze(1), t)

        def accf(logits, t):
            return ((logits.squeeze(1) > 0).float() == t).float().mean()

    def evaluate(chunk=65536):
        model.eval()
        with torch.no_grad():
            lv = torch.cat([
                model((torch.from_numpy(Xva[i : i + chunk].astype(np.float32)) - mean_t) / std_t)
                for i in range(0, len(Xva), chunk)
            ])
            vloss = lossf(lv, yva_t).item()
            acc = accf(lv, yva_t).item()
        model.train()
        return vloss, acc

    best = (float("inf"), None)
    step = 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot, cnt = 0.0, 0
        for i in range(0, len(perm) - args.bs + 1, args.bs):
            idx = np.sort(perm[i : i + args.bs])
            xb = (torch.from_numpy(Xtr[idx].astype(np.float32)) - mean_t) / std_t
            yb = torch.from_numpy(ytr[idx])
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            cnt += 1
            step += 1
            if step % args.eval_every == 0:
                vloss, acc = evaluate()
                if vloss < best[0]:
                    best = (vloss, {k: v.clone() for k, v in model.state_dict().items()})
                print(f"ep {ep} step {step} train {tot / cnt:.4f} val {vloss:.4f} acc {acc:.4f} best {best[0]:.4f} [{time.time() - t0:.0f}s]", flush=True)
                tot, cnt = 0.0, 0
    vloss, acc = evaluate()
    if vloss < best[0]:
        best = (vloss, model.state_dict())
    model.load_state_dict(best[1])
    model.eval()
    print(f"best val {best[0]:.4f}", flush=True)

    torch.save(model.state_dict(), args.out.replace(".npz", ".pt"))
    if args.arch == "board":
        board.export(model, args.out)
    else:
        export(model, mean, std, args.out, hidden, cols, use_derived, args.players)
    print("saved", args.out)


if __name__ == "__main__":
    main()
