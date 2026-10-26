// Native tree search and self-play: cbot/mcts.py's MCTSPlayer on the native
// engine, with the value committee and policy net evaluated in C++. With more
// than two players the search is max-n, as in cbot.mcts: values are vectors
// of win probabilities, one per seat, and each mover maximizes its own.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include "catan.h"

namespace ce {

// LAYOUT[n][k]: where canonical feature k goes in cbot.feats order, n players.
extern int LAYOUT[MAXP + 1][DIM];
inline void encode(const State& s, int p0, float* out) {
  float f[DIM];
  s.encode_canonical(p0, f);
  const int* L = LAYOUT[int(s.np)];
  for (int k = 0, d = dim(s.np); k < d; k++) out[L[k]] = f[k];
}

// --- Nets (cbot.search.ValueNet: normalization folded into the first layer) --

struct Layer {
  int in, out;
  std::vector<float> W, b;  // W row-major (in, out)
};

struct MLP {
  std::vector<int> cols;  // input columns of the full feature vector
  std::vector<Layer> layers;

  int out_dim() const { return layers.back().out; }

  // X: (B, stride) full vectors -> out: (B, out_dim) logits.
  void forward(const float* X, int B, int stride, float* out) const {
    thread_local std::vector<float> h, o;
    int in = int(cols.size());
    h.resize(size_t(B) * in);
    for (int b = 0; b < B; b++)
      for (int i = 0; i < in; i++) h[size_t(b) * in + i] = X[size_t(b) * stride + cols[i]];
    for (size_t li = 0; li < layers.size(); li++) {
      const Layer& L = layers[li];
      o.resize(size_t(B) * L.out);
      for (int b = 0; b < B; b++) std::memcpy(&o[size_t(b) * L.out], L.b.data(), L.out * sizeof(float));
      for (int i = 0; i < L.in; i++) {
        const float* __restrict w = &L.W[size_t(i) * L.out];
        for (int b = 0; b < B; b++) {
          float x = h[size_t(b) * L.in + i];
          if (x == 0.0f) continue;
          float* __restrict y = &o[size_t(b) * L.out];
          for (int j = 0; j < L.out; j++) y[j] += x * w[j];
        }
      }
      bool last = li + 1 == layers.size();
      if (!last)
        for (float& v : o) v = v > 0 ? v : 0;
      h.swap(o);
    }
    std::memcpy(out, h.data(), size_t(B) * out_dim() * sizeof(float));
  }
};

struct Nets {
  // Committee, mean of win probabilities: 1v1 nets give P(perspective seat
  // wins) as one logit, multiplayer nets a softmax over the seats in order
  // from the perspective seat.
  std::vector<MLP> value;
  MLP policy;  // logits over the moves (the first out_dim move numbers)
  bool has_policy = false;

  // Win probability of each seat from the perspective seat on (n per row)
  // for B feature rows of n players (encoded for the perspective seat).
  void values(const float* X, int B, int n, double* out) const {
    thread_local std::vector<float> lg;
    int stride = dim(n);
    for (int i = 0; i < B * n; i++) out[i] = 0;
    for (const MLP& m : value) {
      int k = m.out_dim();
      if (k != (n == 2 ? 1 : n) && k != n) throw std::runtime_error("value net does not match the players");
      lg.resize(size_t(B) * k);
      m.forward(X, B, stride, lg.data());
      for (int b = 0; b < B; b++) {
        double* o = out + size_t(b) * n;
        const float* z = &lg[size_t(b) * k];
        if (k == 1) {
          double p = 1.0 / (1.0 + std::exp(-std::min(30.0, std::max(-30.0, double(z[0])))));
          o[0] += p, o[1] += 1 - p;
        } else {
          double mx = -INFINITY, sum = 0, e[MAXP];
          for (int j = 0; j < k; j++) mx = std::max(mx, double(z[j]));
          for (int j = 0; j < k; j++) sum += e[j] = std::exp(z[j] - mx);
          for (int j = 0; j < k; j++) o[j] += e[j] / sum;
        }
      }
    }
    for (int i = 0; i < B * n; i++) out[i] /= value.size();
  }
};

// --- Hidden information (cbot.belief.determinize, development cards) ---------

// The other seats' development cards dealt again from what `me` has not seen
// (the deck and their hands), each keeping its number of cards and none
// getting enough victory points to have won; the deck is the rest, in a
// random order. Resource hands are left as they are: fully known in 1v1;
// with more players a steal between two others hides its card, which
// cbot.belief guesses before a position reaches the engine (self-play
// searches the real hands).
inline void determinize(State& s, int me, Rng& rng) {
  int8_t pool[25];
  int n = 0, opp[MAXP], size[MAXP], room[MAXP], no = 0;
  for (int i = 0; i < s.ndeck; i++) pool[n++] = s.deck[i];
  for (int k = 1; k < s.np; k++) {
    int c = (me + k) % s.np;
    Player& o = s.p[c];
    opp[no] = c, size[no] = o.ndevs(), room[no] = s.vps_to_win - 1 - o.vp, no++;
    for (int d = 0; d < 5; d++)
      for (int j = 0; j < o.dev[d]; j++) pool[n++] = int8_t(d);
  }
  bool ok = false;
  for (int t = 0; t < 1000 && !ok; t++) {
    rng.shuffle(pool, n);
    ok = true;
    for (int i = 0, j = 0; j < no; i += size[j], j++) {
      int vps = 0;
      for (int x = i; x < i + size[j]; x++) vps += pool[x] == VPC;
      ok = ok && vps <= room[j];
    }
  }
  if (!ok) {
    rng.shuffle(pool, n);
    std::stable_sort(pool, pool + n, [](int8_t a, int8_t b) { return (a == VPC) < (b == VPC); });
  }
  int i = 0;
  for (int j = 0; j < no; j++) {
    Player& o = s.p[opp[j]];
    for (int d = 0; d < 5; d++) o.dev[d] = 0;
    for (int x = 0; x < size[j]; x++) o.dev[pool[i++]]++;
    for (int d = 0; d < 4; d++) o.owned_start[d] = o.dev[d] > 0;
    o.avp = int8_t(o.vp + o.dev[VPC]);
  }
  s.ndeck = int8_t(n - i);
  for (int x = 0; x < s.ndeck; x++) s.deck[x] = pool[i + x];
}

// --- Tree search (cbot.mcts.MCTSPlayer) ----------------------------------------

struct Params {
  int sims = 400;
  double cpuct = 0.3;
  int batch = 8;
  double ptemp = 1.0, pmix = 0.0, fpu = 0.0;
  double noise = 0.0, alpha = 0.3;  // root Dirichlet noise
  int pick = 0;                     // 1: play a move drawn by visit counts
  int initf = 2;                    // 2: opening completion, 0: search the opening too
  int fair = 1;                     // search a determinized copy
};

enum { NEW, PENDING, OPEN, OVER };

struct Outcome {
  int node;
  double p;
  int visits;
};

struct Node {
  State s;
  int mover, level, status = NEW;
  bool has_value = false;
  double value[MAXP];
  std::vector<int> acts;
  std::vector<double> P, N, W;  // W: (moves, MAXP)
  std::vector<std::vector<Outcome>> kids;
};

// What a decision found: the move, and per root move its visits and value
// (for the mover), plus the value vector of the move played.
struct Decision {
  int move = -1;
  bool searched = false;
  std::vector<int> moves, visits;
  std::vector<double> q;
  double vec[MAXP] = {0, 0, 0, 0};
};

struct Searcher {
  const Nets* nets;
  Params prm;
  Rng rng;
  std::vector<Node> tree;
  std::vector<float> X;

  Searcher(const Nets* n, const Params& p, uint64_t seed) : nets(n), prm(p), rng(seed) {}

  Decision decide(const State& real, int me) {
    Decision d;
    int acts[NUM_ACTIONS];
    int na = real.legal(acts);
    if (na == 1) {
      d.move = acts[0];
      return d;
    }
    State s = real;
    if (prm.fair) determinize(s, me, rng);
    d.searched = true;
    if (prm.initf == 2 && s.initial) return complete_opening(s, me, acts, na, d);
    return search(s, me, d);
  }

  // Value vectors (seat order) of states, from `me`'s nets.
  void eval(const std::vector<const State*>& ss, int me, std::vector<std::array<double, MAXP>>& out) {
    int B = int(ss.size());
    if (!B) return;
    int n = ss[0]->np, D = dim(n);
    X.resize(size_t(B) * D);
    for (int b = 0; b < B; b++) encode(*ss[b], me, &X[size_t(b) * D]);
    std::vector<double> v(size_t(B) * n);
    nets->values(X.data(), B, n, v.data());
    out.assign(B, {0, 0, 0, 0});
    for (int b = 0; b < B; b++)
      for (int j = 0; j < n; j++) out[b][(me + j) % n] = v[size_t(b) * n + j];
  }

  // NNSearchPlayer._complete_opening: finish the opening greedily by base_fn
  // for both seats after each candidate, score the first mid-game positions.
  Decision& complete_opening(const State& s, int me, const int* acts, int na, Decision& d) {
    std::vector<State> finals(na, s);
    for (int i = 0; i < na; i++) {
      State& g = finals[i];
      g.apply(acts[i], -1, rng);
      while (g.initial && g.winner() < 0) {
        int a[NUM_ACTIONS];
        int n = g.legal(a), best = a[0];
        if (n > 1) {
          double bv = -INFINITY;
          for (int k = 0; k < n; k++) {
            State h = g;
            h.apply(a[k], -1, rng);
            double v = h.base_value(g.cur);
            if (v > bv) bv = v, best = a[k];
          }
        }
        g.apply(best, -1, rng);
      }
    }
    std::vector<const State*> ps;
    for (auto& g : finals) ps.push_back(&g);
    std::vector<std::array<double, MAXP>> vals;
    eval(ps, me, vals);
    int bi = 0;
    for (int i = 1; i < na; i++)
      if (vals[i][me] > vals[bi][me]) bi = i;
    d.move = acts[bi];
    d.moves.assign(acts, acts + na);
    d.visits.assign(na, -1);
    for (int i = 0; i < na; i++) d.q.push_back(vals[i][me]);
    for (int j = 0; j < MAXP; j++) d.vec[j] = vals[bi][j];
    return d;
  }

  int add(const State& s, int level) {
    tree.emplace_back();
    Node& n = tree.back();
    n.s = s;
    n.mover = s.cur;
    n.level = level;
    return int(tree.size()) - 1;
  }

  void open(int i, const int* acts, int na) {
    Node& n = tree[i];
    n.status = OPEN;
    n.acts.assign(acts, acts + na);
    n.N.assign(na, 0);
    n.W.assign(size_t(na) * MAXP, 0);
    n.kids.assign(na, {});
  }

  // Scores nodes (value, and priors over their moves).
  void score(const std::vector<int>& idx, int me) {
    std::vector<int> live;
    std::vector<const State*> todo;
    std::vector<int> todo_i;
    for (int i : idx) {
      Node& n = tree[i];
      int w = n.s.winner();
      if (w >= 0) {
        for (int k = 0; k < MAXP; k++) n.value[k] = k == w;
        n.has_value = true;
        n.status = OVER;
        continue;
      }
      live.push_back(i);
      if (!n.has_value) todo.push_back(&n.s), todo_i.push_back(i);
    }
    if (!todo.empty()) {
      std::vector<std::array<double, MAXP>> vals;
      eval(todo, me, vals);
      for (size_t k = 0; k < todo_i.size(); k++) {
        Node& n = tree[todo_i[k]];
        for (int j = 0; j < MAXP; j++) n.value[j] = vals[k][j];
        n.has_value = true;
      }
    }
    if (live.empty()) return;
    // Priors from the policy net, on the mover's features.
    int B = int(live.size());
    std::vector<std::vector<int>> acts(B);
    for (int b = 0; b < B; b++) {
      int a[NUM_ACTIONS];
      int na = tree[live[b]].s.legal(a);
      acts[b].assign(a, a + na);
    }
    std::vector<float> logits;
    int K = nets->has_policy ? nets->policy.out_dim() : 0;
    if (nets->has_policy) {
      int D = dim(tree[live[0]].s.np);
      X.resize(size_t(B) * D);
      for (int b = 0; b < B; b++) encode(tree[live[b]].s, tree[live[b]].mover, &X[size_t(b) * D]);
      logits.resize(size_t(B) * K);
      nets->policy.forward(X.data(), B, D, logits.data());
    }
    for (int b = 0; b < B; b++) {
      int i = live[b];
      open(i, acts[b].data(), int(acts[b].size()));
      Node& n = tree[i];
      int na = int(n.acts.size());
      n.P.assign(na, 1.0 / na);
      if (nets->has_policy) {
        // Moves past the policy's numbers (a 1v1 policy's, at a bigger
        // table) keep a uniform share, as moves without a number in cbot.mcts.
        const float* lg = &logits[size_t(b) * K];
        double mx = -INFINITY, sum = 0;
        int known = 0;
        for (int k = 0; k < na; k++)
          if (n.acts[k] < K) mx = std::max(mx, lg[n.acts[k]] / prm.ptemp), known++;
        if (known) {
          for (int k = 0; k < na; k++)
            if (n.acts[k] < K) sum += n.P[k] = std::exp(lg[n.acts[k]] / prm.ptemp - mx);
          double share = double(known) / na, tot = 0;
          for (int k = 0; k < na; k++) {
            if (n.acts[k] < K) n.P[k] = n.P[k] / sum * share;
            tot += n.P[k];
          }
          for (int k = 0; k < na; k++) n.P[k] /= tot;
        }
      }
      if (prm.pmix)
        for (int k = 0; k < na; k++) n.P[k] = (1 - prm.pmix) * n.P[k] + prm.pmix / na;
    }
  }

  int select(const Node& n) const {
    int na = int(n.acts.size());
    if (na == 1) return 0;
    int m = n.mover;
    double tot = 0;
    for (double x : n.N) tot += x;
    double sq = std::sqrt(tot + 1), best = -INFINITY;
    int bi = 0;
    for (int k = 0; k < na; k++) {
      double q = n.N[k] > 0 ? n.W[size_t(k) * MAXP + m] / n.N[k] : n.value[m] - prm.fpu;
      double u = prm.cpuct * n.P[k] * sq / (1 + n.N[k]);
      if (q + u > best) best = q + u, bi = k;
    }
    return bi;
  }

  int child(int i, int k) {
    if (tree[i].kids[k].empty()) {
      int outs[16];
      double probs[16];
      const State s = tree[i].s;
      int a = tree[i].acts[k];
      int n = s.spectrum(a, outs, probs);
      int level = tree[i].level + (tree[i].acts.size() > 1);
      std::vector<Outcome> kids;
      for (int j = 0; j < n; j++) {
        State g = s;
        g.apply_outcome(a, outs[j], rng);
        kids.push_back({add(g, level), probs[j], 0});
      }
      tree[i].kids[k] = std::move(kids);
    }
    auto& outs = tree[i].kids[k];
    Outcome* o = &outs[0];
    if (outs.size() > 1) {
      double tot = 1;
      for (auto& x : outs) tot += x.visits;
      double best = -INFINITY;
      for (auto& x : outs) {
        double v = x.p - x.visits / tot;
        if (v > best) best = v, o = &x;
      }
    }
    o->visits++;
    return o->node;
  }

  // A lost simulation from the mover's view (cbot.mcts._loss): a share of a
  // win for every other seat.
  void add_loss(Node& n, int k, double sign) {
    int np = n.s.np;
    for (int j = 0; j < np; j++)
      if (j != n.mover) n.W[size_t(k) * MAXP + j] += sign / (np - 1);
  }

  // Walks down by PUCT with a virtual loss; returns the leaf, path in `path`.
  int descend(int i, std::vector<std::pair<int, int>>& path) {
    path.clear();
    while (true) {
      Node& n = tree[i];
      if (n.status == NEW) {
        int a[NUM_ACTIONS];
        int na = n.s.legal(a);
        if (n.s.winner() >= 0 || na > 1) {
          n.status = PENDING;
        } else {
          open(i, a, na);
          tree[i].P.assign(1, 1.0);
        }
      }
      if (tree[i].status != OPEN) return i;
      int k = select(tree[i]);
      path.push_back({i, k});
      Node& m = tree[i];
      m.N[k] += 1;
      add_loss(m, k, 1);
      i = child(i, k);
    }
  }

  Decision& search(const State& s, int me, Decision& d) {
    tree.clear();
    tree.reserve(size_t(prm.sims) * 4 + 64);
    int root = add(s, 0);
    score({root}, me);
    Node& r = tree[root];
    int na = int(r.acts.size());
    if (prm.noise > 0) {
      std::vector<double> g(na);
      double sum = 0;
      std::gamma_distribution<double> gd(prm.alpha, 1.0);
      std::mt19937_64 eng(rng.next());
      for (double& x : g) sum += x = gd(eng);
      for (int k = 0; k < na; k++) r.P[k] = (1 - prm.noise) * r.P[k] + prm.noise * g[k] / sum;
    }
    std::vector<std::vector<std::pair<int, int>>> paths(prm.batch);
    std::vector<int> leaves(prm.batch);
    int done = 0;
    while (done < prm.sims) {
      int nb = std::min(prm.batch, prm.sims - done);
      std::vector<int> fresh;
      for (int b = 0; b < nb; b++) {
        leaves[b] = descend(root, paths[b]);
        if (tree[leaves[b]].status == PENDING &&
            std::find(fresh.begin(), fresh.end(), leaves[b]) == fresh.end())
          fresh.push_back(leaves[b]);
      }
      done += nb;
      if (!fresh.empty()) score(fresh, me);
      for (int b = 0; b < nb; b++) {
        const double* v = tree[leaves[b]].value;
        for (auto& [ni, k] : paths[b]) {
          Node& n = tree[ni];
          for (int j = 0; j < MAXP; j++) n.W[size_t(k) * MAXP + j] += v[j];
          add_loss(n, k, -1);
        }
      }
    }
    Node& R = tree[root];
    int m = R.mover;
    std::vector<std::array<double, MAXP>> Q(na);
    for (int k = 0; k < na; k++)
      for (int j = 0; j < MAXP; j++) Q[k][j] = R.N[k] > 0 ? R.W[size_t(k) * MAXP + j] / R.N[k] : R.value[j] - 1.0;
    int first = 0;
    if (prm.pick) {
      double tot = 0;
      for (double x : R.N) tot += x;
      double u = rng.uniform() * tot;
      first = na - 1;
      for (int k = 0; k < na; k++) {
        if (u < R.N[k]) {
          first = k;
          break;
        }
        u -= R.N[k];
      }
    } else {
      for (int k = 1; k < na; k++)
        if (R.N[k] > R.N[first] || (R.N[k] == R.N[first] && Q[k][m] >= Q[first][m])) first = k;
    }
    d.move = R.acts[first];
    d.moves = R.acts;
    for (int k = 0; k < na; k++) d.visits.push_back(int(R.N[k])), d.q.push_back(Q[k][me]);
    for (int j = 0; j < MAXP; j++) d.vec[j] = Q[first][j];
    return d;
  }
};

// --- Self-play (cbot.gen with every seat played by the same searcher) -------

struct GameRecord {
  uint64_t seed;
  int np, winner;
  std::vector<float> X;  // rows of dim(np), every seat per recorded position
  std::vector<int> who;
  std::vector<float> v;  // per row, np search values from the row's seat on (NaN if not searched)
  std::vector<int> pr, pa, pn;
  std::vector<float> pv;
};

inline GameRecord play_game(const Nets* nets, const Params& prm, uint64_t seed, int np, double p_record, double eps) {
  Rng rng(seed);
  Map map = Map::random(rng);
  State s = State::fresh(&map, rng, np);
  std::vector<Searcher> players;
  for (int c = 0; c < np; c++) players.emplace_back(nets, prm, rng.next());
  GameRecord g;
  g.seed = seed, g.np = np;
  const int D = dim(np);
  int acts[NUM_ACTIONS];
  while (!s.over()) {
    bool rec = rng.uniform() < p_record;
    size_t row0 = g.who.size();
    if (rec)
      for (int c = 0; c < np; c++) {
        g.X.resize(g.X.size() + D);
        encode(s, c, &g.X[g.X.size() - D]);
        g.who.push_back(c);
        g.v.resize(g.v.size() + np, NAN);
      }
    int na = s.legal(acts), move;
    if (na > 1 && rng.uniform() < eps) {
      move = acts[rng.below(na)];
    } else {
      int mover = s.cur;
      Decision d = players[mover].decide(s, mover);
      move = d.move;
      if (rec && d.searched) {
        for (size_t k = 0; k < d.moves.size(); k++) {
          g.pr.push_back(int(row0) + mover);
          g.pa.push_back(d.moves[k]);
          g.pv.push_back(float(d.q[k]));
          g.pn.push_back(d.visits[k]);
        }
        for (int c = 0; c < np; c++)
          for (int j = 0; j < np; j++) g.v[(row0 + c) * np + j] = float(d.vec[(c + j) % np]);
      }
    }
    s.apply(move, -1, rng);
  }
  g.winner = s.winner();
  return g;
}

}  // namespace ce
