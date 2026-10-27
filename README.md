# catan-rl

Bots for Settlers of Catan, built on [Catanatron](https://github.com/bcollazo/catanatron), that beat
Catanatron's strongest built-in bot (AlphaBeta) at 2, 3 and 4 players. They combine tree search with
value and policy networks trained by expert iteration (self-play, then training on the search's own
results), and play only on information a player at the table would have.

## Results

The released bots play with the standard rules: 10 points to win, discard over 7, base map. One seat
is our bot and the others are Catanatron's AlphaBeta. Each map is played once from every seat, and
every evaluation uses seeds that were not used during development. Intervals are 95% Wilson
intervals.

| Players | Bot | Games | Win rate | 95% CI | Chance | Time per move |
|---|---|---|---|---|---|---|
| 2 | N1 + P1 (`models/BOT1V1.txt`) | 4000 | **90.1%** | 89.1-91.0% | 50% | ~11 ms |
| 3 | T1 + P3a (`models/BOT3P.txt`) | 999 | **78.6%** | 75.9-81.0% | 33% | ~14 ms |
| 4 | Q1 + P4a (`models/BOT4P.txt`) | 1000 | **76.6%** | 73.9-79.1% | 25% | ~14 ms |

For scale: Catanatron's own bots are close to even with each other one on one (AlphaBeta against
ValueFunction wins 49.6% over 2000 games), and AlphaBeta in one seat against three ValueFunction
players wins 30.7%.

How the bots got there, against AlphaBeta:

| Bot | 2 players | 3 players | 4 players |
|---|---|---|---|
| Depth-2 expectimax with a value-net committee | 82.3% (4000 games) | 69.2% | 70.2% |
| Tree search with a policy net, first self-play round | 89.4% | 73.0% | 74.0% |
| Tree search, second self-play round (released) | 90.1% | 78.6% | 76.6% |

The one-on-one expectimax figure predates the restriction to public information (its search saw
the real hands for robber steals); every other figure is with hidden information guessed.

Head to head, each second-round bot beats the first: one on one N1 wins 52.4% against N0 (4000
games), and at 4 players Q1 wins 29.1% [26.4, 32.0] against three copies of Q0 (25% is even).

### Trading

The bots trade with other players: they answer offers, choose between players who accept, and at 3
and 4 players they also make offers (one card for one, or two of a kind for one; at most two a
turn). Each option is scored by searching the position after the trade against the position
without it. One seat that offers, 1000 games each:

| Table | With offers | Without | Gain |
|---|---|---|---|
| 4 players, against three copies that only answer | 31.6% | 25% (even) | +6.6 points |
| 4 players, against three of the previous round's bots | 36.8% | 28.9% | +7.9 points |
| 3 players, against two copies that only answer | 36.4% | 33.3% (even) | +3.1 points |
| 3 players, against two of the previous round's bots | 52.1% | 52.1% | none |
| 2 players, against a copy that only answers | 51.6% | 50% (even) | not significant |

Offers are therefore on at 3 and 4 players and off one on one. AlphaBeta cannot take part in trades
(it raises an error when offered one), so trades are off at any table with a Catanatron bot.

## How it works

**Search.** The released bots use PUCT tree search (as in AlphaZero) with max-n backups: each node
holds every seat's chance of winning, and each player moves to maximize their own. A policy net
gives the move priors and a committee of five value nets scores the leaves. Dice, development cards
and robber steals are chance nodes. Search runs in a C++ engine (`cbot/engine`) that reimplements
Catanatron's rules for 2 to 4 players. It was checked move for move against Catanatron over 741,000
positions and is fast enough for about 60,000 searched one-on-one games an hour on 32 threads.

**Hidden information.** Before each decision the search builds one guess of the hidden state, the
way a careful player would (`cbot/belief.py`). Resource hands are counted from the public record; a
card stolen between two other players is drawn from the victim's counted hand, and a guess that
later contradicts the record is redrawn. Opponents' development cards are dealt again from every
card the bot has not seen. The networks themselves see only public features: own hand, other
players' card counts, played development cards and the board.

**Training.** Expert iteration. The current best bot plays itself with exploration noise at the root.
The value nets learn a blend of the game result and the search's value at each position
(`lam * result + (1 - lam) * search value`); the search value is much less noisy than one
dice-driven result. The policy net learns the search's visit counts. The new nets play the old ones
and replace them only if they win. The scripts for each round are in `cbot/round_*.sh`.

**Earlier bots.** Before the tree search, the bots were depth-2 expectimax over Catanatron's own
engine (`cbot/search.py`), with value-net committees trained on AlphaBeta games and then on their
own play. The most important trick there was opening completion (`initf=2`): each candidate
placement is scored by finishing the opening with a fast heuristic player and judging the first real
position with the net. That raised the win rate from 49.5% to 73.0% before any other improvement.

## Findings about Catanatron

- Chance nodes in AlphaBeta's search are wrong: `execute_spectrum` puts the intended dice roll or
  development card in `action.value`, which `apply_action` ignores. Dice branches roll at random and
  development-card branches draw the real top card. Fixing it (`ABfix`) barely changes AlphaBeta's
  strength: fixed against stock wins 51.3% over 2000 games.
- `Game.copy()` shares the game's random number generator, so a search that rolls dice in a copy
  changes the real game's future rolls.
- The ladder in Catanatron's README does not reproduce one on one: AlphaBeta against ValueFunction
  is 49.6%, not 80%.
- In the domestic trade sequence the offering player could be asked to answer their own offer when
  not seated last; `cbot/trade.py` fixes this.

## What did not help

| Idea | Result |
|---|---|
| Depth-3 expectimax one on one | 75.7% against 76.8% at depth 2, 2.5 times slower |
| Raw board one-hot features | Overfits: the net memorizes games |
| Tracking the opponent's exact hand as a network input (1v1) | 74.1% against 73.2%, within noise |
| Wider value nets for the 1v1 expectimax (512-256-128) | Overfit: data-limited, not capacity-limited |
| A second policy net trained on both rounds (4 players) | No better than the first |

## Setup

Requires Linux, Python 3.12+ and g++.

```bash
cbot/setup_server.sh
```

This clones Catanatron into `vendor/catanatron` at commit `ecf9311` (every result above used it),
creates `.venv` with numpy, pybind11 and CPU-only PyTorch, builds the native engine
(`cbot/engine/build.sh`) and runs smoke tests.

Evaluate a bot against AlphaBeta:

```bash
.venv/bin/python -m cbot.arena "$(cat models/BOT1V1.txt)" AB --games 1000 --procs 30
.venv/bin/python -m cbot.arena "$(cat models/BOT3P.txt)" AB AB --games 999 --procs 30
.venv/bin/python -m cbot.arena "$(cat models/BOT4P.txt)" AB AB AB --games 1000 --procs 30
```

Other bot specs are listed in `cbot/players.py`, for example `NN:$(cat models/FINAL.txt):initf=2`
for the earlier one-on-one expectimax bot, or `AB`, `ABfix`, `F` and `R` for Catanatron's bots.

## Play against the bots

`cbot/web.py` adds a CBOT seat to Catanatron's web app. A patch to the UI
(`cbot/catanatron_web.patch`) shows the game from your seat with other hands face down, lets you
choose whom to rob, adds trading with the other players, and defaults to standard rules with you
against three CBOTs. The seat picks the 2-, 3- or 4-player bot to match the table.

```bash
.venv/bin/pip install flask flask_cors flask_sqlalchemy sqlalchemy
git -C vendor/catanatron apply ../../cbot/catanatron_web.patch
cp cbot/web_tiles/*.svg vendor/catanatron/ui/src/assets/
docker build -t cbot-ui vendor/catanatron/ui

.venv/bin/flask --app cbot.web run --host 127.0.0.1 --port 5001
docker run -d --name cbot-ui -p 127.0.0.1:3001:3000 cbot-ui
```

Then open `localhost:3001`.

## Repository layout

| Path | Contents |
|---|---|
| `cbot/engine/` | C++ rules engine and tree search (pybind11 module) |
| `cbot/native.py`, `cbot/mcts.py` | Tree-search players: native and pure Python |
| `cbot/search.py` | Expectimax player; trade answers and offers |
| `cbot/belief.py` | Guessing hidden hands from the public record |
| `cbot/feats.py`, `cbot/policy.py` | Network inputs and the move encoding |
| `cbot/train.py`, `cbot/train_policy.py` | Value and policy training |
| `cbot/ngen.py`, `cbot/gen.py` | Self-play data generation, native and Python |
| `cbot/arena.py`, `cbot/players.py` | Evaluation and bot specs |
| `cbot/trade.py`, `cbot/chance.py`, `cbot/fast.py` | Trades, chance-node fix, speedups for Catanatron |
| `cbot/web.py`, `cbot/catanatron_web.patch`, `cbot/web_tiles/` | Web play |
| `cbot/round_*.sh`, `cbot/round_lib.sh` | Training rounds on one machine |
| `cbot/loop4.sh`, `cbot/node.py`, `cbot/genpool.py`, `cbot/slurm/` | Multi-node training on a Slurm cluster |
| `cbot/check_*.py` | Checks, mainly that the native engine and speedups match Catanatron |
| `models/` | Released weights (`.npz`) and bot specs (`.txt`) |
