"""Train a policy MLP on the policy targets in generated shards (cbot.gen).

    python -m cbot.train_policy --data data/p0 --out models/P0.npz

Each recorded decision gives the search's value of every legal move for the
mover. The target is softmax(value / temp) over the legal moves, or, where a
tree search (cbot.mcts) made the decision, its share of the visits, as in
AlphaZero; the loss is cross-entropy with illegal moves masked out. Validation is on the last games
(whole games, so no position is in both sets) and reports how often the
search's best move is among the policy's top 1-4 moves, the number that
matters for pruning.

The output file has the value-net format (cbot.search.ValueNet) with one
output per move: cbot.policy.SIZE in 1v1, the native engine's numbers
(cbot.native.index, with steals from each seat apart) with more players.
"""

import argparse
import glob
import time

import numpy as np
import torch
import torch.nn as nn

from cbot import policy
from cbot.train import build, export


def moves(players):
    """Policy outputs: cbot.policy's numbers in 1v1, the native engine's otherwise."""
    if players == 2:
        return policy.SIZE
    from cbot import native

    return native._engine.NUM_ACTIONS


def load(dirs, temp):
    X, T, M, G = [], [], [], []
    size = None
    for f in sorted(f for d in dirs for f in glob.glob(f"{d}/*.npz") if not f.endswith(".tmp.npz")):
        d = np.load(f)
        if "pr" not in d.files or len(d["pr"]) == 0:
            continue
        if size is None:
            size = moves(2 if d["v"].ndim == 1 else d["v"].shape[1])
        pr, pa, pv = d["pr"], d["pa"].astype(np.int64), d["pv"]
        rows, inv = np.unique(pr, return_inverse=True)
        vals = np.full((len(rows), size), -np.inf, dtype=np.float32)
        vals[inv, pa] = pv
        mask = np.isfinite(vals)
        z = (vals - vals.max(1, keepdims=True)) / temp
        t = np.where(mask, np.exp(z), 0.0)
        t /= t.sum(1, keepdims=True)
        if "pn" in d.files and (d["pn"] >= 0).any():
            pn = d["pn"]
            visits = np.zeros((len(rows), size), dtype=np.float32)
            visits[inv, pa] = np.maximum(pn, 0)
            tot = visits.sum(1, keepdims=True)
            searched = np.zeros(len(rows), bool)
            searched[inv[pn >= 0]] = True
            searched &= tot[:, 0] > 0
            t[searched] = visits[searched] / tot[searched]
        X.append(d["X"][rows])
        T.append(t.astype(np.float32))
        M.append(mask)
        G.append(d["g"][rows])
    return np.concatenate(X), np.concatenate(T), np.concatenate(M), np.concatenate(G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", default="512,256")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--temp", type=float, default=0.01)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    hidden = [int(h) for h in args.hidden.split(",")]
    t0 = time.time()

    X, T, M, G = load(args.data, args.temp)
    games = np.unique(G)
    val_games = games[-max(1, int(len(games) * args.val_frac)):]
    va = np.isin(G, val_games)
    print(f"{len(X)} decisions from {len(games)} games ({va.sum()} validation), "
          f"{M.sum(1).mean():.1f} legal moves each [{time.time() - t0:.0f}s]", flush=True)
    Xtr, Ttr, Mtr = X[~va], T[~va], M[~va]
    mean = Xtr.astype(np.float32).mean(0)
    std = Xtr.astype(np.float32).std(0) + 1e-3
    norm = lambda x: torch.from_numpy((x.astype(np.float32) - mean) / std)  # noqa: E731
    Xva_t, Tva_t, Mva_t = norm(X[va]), torch.from_numpy(T[va]), torch.from_numpy(M[va])

    model = build(X.shape[1], hidden, args.dropout, out=T.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * max(1, len(Xtr) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    def lossf(logits, t, m):
        logp = torch.log_softmax(logits.masked_fill(~m, -1e9), 1)
        return -(t * logp.masked_fill(~m, 0.0)).sum(1).mean()

    def evaluate():
        model.eval()
        with torch.no_grad():
            lg = model(Xva_t).masked_fill(~Mva_t, -1e9)
            loss = lossf(model(Xva_t), Tva_t, Mva_t).item()
            best = Tva_t.argmax(1)
            rank = (lg > lg.gather(1, best[:, None])).sum(1)  # moves rated above the best
            multi = Mva_t.sum(1) > 1
            recall = [((rank < k) & multi).sum().item() / max(1, multi.sum().item()) for k in (1, 2, 3, 4)]
        model.train()
        return loss, recall

    best = (float("inf"), None, None)
    for ep in range(args.epochs):
        perm = np.random.permutation(len(Xtr))
        tot, cnt = 0.0, 0
        for i in range(0, len(perm) - args.bs + 1, args.bs):
            idx = np.sort(perm[i : i + args.bs])
            loss = lossf(model(norm(Xtr[idx])), torch.from_numpy(Ttr[idx]), torch.from_numpy(Mtr[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            cnt += 1
        vloss, recall = evaluate()
        if vloss < best[0]:
            best = (vloss, {k: v.clone() for k, v in model.state_dict().items()}, recall)
        print(f"ep {ep} train {tot / max(1, cnt):.4f} val {vloss:.4f} best move in top1-4 "
              + " ".join(f"{r:.3f}" for r in recall) + f" [{time.time() - t0:.0f}s]", flush=True)
    model.load_state_dict(best[1])
    model.eval()
    print(f"best val {best[0]:.4f}, best move in top1-4 " + " ".join(f"{r:.3f}" for r in best[2]), flush=True)
    export(model, mean, std, args.out, hidden, np.arange(X.shape[1]))
    print("saved", args.out)


if __name__ == "__main__":
    main()
