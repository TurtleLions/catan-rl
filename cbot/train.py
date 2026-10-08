"""Train the P(win) value MLP on generated shards.

    python -m cbot.train --data data/r0 [data/r1 ...] --out models/v0.npz

Saves numpy weights + feature normalization for torch-free inference in search.
"""

import argparse
import glob
import time

import numpy as np
import torch
import torch.nn as nn


def load(dirs, val_frac=0.05, seed=0, lam=1.0, project=None):
    files = sorted(f for d in dirs for f in glob.glob(f"{d}/*.npz"))
    Xs, ys = [], []
    for f in files:
        d = np.load(f)
        Xs.append(d["X"] if project is None else project(d["X"]))
        y = d["y"].astype(np.float32)
        if lam < 1.0 and "v" in d:
            # Soft target: blend outcome with the generator's search value.
            v = d["v"]
            y = np.where(np.isnan(v), y, lam * y + (1 - lam) * v).astype(np.float32)
        ys.append(y)
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(np.float32)
    # Rows come in (RED, BLUE) pairs from the same state; split by pair-blocks of
    # whole shards' tails so validation states are not mirrored in training.
    n = len(X)
    nval = int(n * val_frac) // 2 * 2
    return X[: n - nval], y[: n - nval], X[n - nval :], y[n - nval :]


def select_cols(kind):
    from cbot import feats

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
    raise ValueError(kind)


def build(dim, hidden, dropout=0.0):
    layers, d = [], dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def export(model, mean, std, path, hidden, cols, derived=False):
    Ws, bs = [], []
    for m in model:
        if isinstance(m, nn.Linear):
            Ws.append(m.weight.detach().numpy().T.astype(np.float32))
            bs.append(m.bias.detach().numpy().astype(np.float32))
    arrs = {"mean": mean, "std": std, "cols": cols, "derived": np.array(int(derived))}
    for i, (W, b) in enumerate(zip(Ws, bs)):
        arrs[f"W{i}"] = W
        arrs[f"b{i}"] = b
    np.savez(path, **arrs)


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
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    hidden = [int(h) for h in args.hidden.split(",")]

    t0 = time.time()
    cols = select_cols(args.cols)
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
    Xtr, ytr, Xva, yva = load(args.data, lam=args.lam, project=project)
    print(f"loaded train {Xtr.shape} val {Xva.shape} pos {ytr.mean():.3f} in {time.time() - t0:.0f}s", flush=True)
    mean = Xtr[:200000].astype(np.float32).mean(0)
    std = Xtr[:200000].astype(np.float32).std(0) + 1e-3

    Xva_t = torch.from_numpy((Xva.astype(np.float32) - mean) / std)
    yva_t = torch.from_numpy(yva)
    model = build(Xtr.shape[1], hidden, args.dropout)
    if args.init:
        model.load_state_dict(torch.load(args.init))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (len(Xtr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    mean_t, std_t = torch.from_numpy(mean), torch.from_numpy(std)

    def evaluate():
        model.eval()
        with torch.no_grad():
            lv = model(Xva_t).squeeze(1)
            vloss = lossf(lv, yva_t).item()
            acc = ((lv > 0).float() == yva_t).float().mean().item()
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
            loss = lossf(model(xb).squeeze(1), yb)
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
    export(model, mean, std, args.out, hidden, cols, use_derived)
    print("saved", args.out)


if __name__ == "__main__":
    main()
