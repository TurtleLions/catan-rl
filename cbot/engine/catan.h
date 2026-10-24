// Native 1v1 Catan with Catanatron's rules, numbering and features.
//
// Mirrors catanatron (vendor/catanatron, ecf9311) move for move, including its
// quirks: connected road components kept as an ordered list of node sets (a
// cut road's halves keep the cutting settlement's node), longest road
// re-awarded to the longest remaining road when a settlement cuts one, the
// opening's longest-road lengths only published at the first later build, and
// so on. cbot/check_native.py replays Catanatron games through it and compares
// legal moves, states and feature vectors at every step.
//
// Two seats; seat i is colors[i] in Catanatron. Moves are numbered as in
// cbot.policy (catanatron.gym's action array, the robber's victim recorded as
// "the opponent").
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "tables.h"

namespace ce {

constexpr int NODES = 54, EDGES = 72, TILES = 19, PORTS = 9, NP = 2, MAXC = 16;
constexpr int FEATS = 667, PIPS = 54, DIM = FEATS + PIPS + 5;
constexpr int TURNS_LIMIT = 1000;  // catanatron.game.TURNS_LIMIT

enum Res { WOOD, BRICK, SHEEP, WHEAT, ORE };
enum Dev { KNIGHT, YOP, MONO, RB, VPC };  // DEVELOPMENT_CARDS order
// ActionPrompt order; the last two (trades between players) are never reached.
enum Prompt { P_INIT_SETTLE, P_INIT_ROAD, P_PLAY, P_DISCARD, P_ROBBER, P_DECIDE_TRADE, P_DECIDE_ACCEPTEES };
// Types of ACTIONS[i][0] (gen_tables.TYPES).
enum Act { A_ROLL, A_DISCARD, A_ROAD, A_SETTLE, A_CITY, A_BUY, A_KNIGHT, A_YOP, A_RB, A_MONO, A_ROBBER, A_TRADE, A_END };

static const int ROAD_COST[5] = {1, 1, 0, 0, 0};
static const int SETTLE_COST[5] = {1, 1, 1, 1, 0};
static const int CITY_COST[5] = {0, 0, 0, 2, 3};
static const int DEV_COST[5] = {0, 0, 1, 1, 1};
static const int SPIRAL_NUMBERS[18] = {5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11};

inline uint64_t bit(int i) { return uint64_t(1) << i; }
inline int popcount(uint64_t x) { return __builtin_popcountll(x); }

// --- Fixed topology, derived from tables.h ---------------------------------

struct Topo {
  int edge_id[NODES][NODES];
  int node_edges[NODES][3];  // edge to NODE_NBR[n][k]
  uint64_t nbr_mask[NODES];
  // Action numbers by meaning.
  int roll, buy, knight, rb, end;
  int discard[5], mono[5], road[EDGES], settle[NODES], city[NODES];
  int yop[5][6];        // [r1][r2], r2 = 5 for one card
  int robber[TILES][2];  // [tile][steal]
  int trade[5][5][5];   // [give][get][rate]
  double dice[13];      // DICE_PROBAS, summed as Catanatron sums them

  Topo() {
    for (auto& r : edge_id) for (int& x : r) x = -1;
    for (int e = 0; e < EDGES; e++) {
      int a = EDGE_NODES[e][0], b = EDGE_NODES[e][1];
      edge_id[a][b] = edge_id[b][a] = e;
    }
    for (int n = 0; n < NODES; n++) {
      nbr_mask[n] = 0;
      for (int k = 0; k < 3; k++) {
        int v = NODE_NBR[n][k];
        node_edges[n][k] = v < 0 ? -1 : edge_id[n][v];
        if (v >= 0) nbr_mask[n] |= bit(v);
      }
    }
    for (int i = 0; i < NUM_ACTIONS; i++) {
      int t = ACTIONS[i][0], x = ACTIONS[i][1], y = ACTIONS[i][2], z = ACTIONS[i][3];
      switch (t) {
        case A_ROLL: roll = i; break;
        case A_BUY: buy = i; break;
        case A_KNIGHT: knight = i; break;
        case A_RB: rb = i; break;
        case A_END: end = i; break;
        case A_DISCARD: discard[x] = i; break;
        case A_MONO: mono[x] = i; break;
        case A_ROAD: road[x] = i; break;
        case A_SETTLE: settle[x] = i; break;
        case A_CITY: city[x] = i; break;
        case A_YOP: yop[x][y < 0 ? 5 : y] = i; break;
        case A_ROBBER: robber[x][y] = i; break;
        case A_TRADE: trade[x][y][z] = i; break;
      }
    }
    for (double& d : dice) d = 0;
    for (int i = 1; i <= 6; i++)
      for (int j = 1; j <= 6; j++) dice[i + j] += 1.0 / 36;
  }
};

inline const Topo& topo() {
  static const Topo t;
  return t;
}

// --- Map --------------------------------------------------------------------

struct Map {
  int8_t res[TILES];   // resource, -1 desert
  int8_t num[TILES];   // dice number, 0 desert
  int8_t port[PORTS];  // resource, -1 for 3:1
  // Derived.
  int desert;
  double proba[TILES];           // number_probability, 0 for the desert
  double node_prod[NODES][5];    // map.node_production
  float pips[NODES];             // feats._node_pips
  uint64_t port_nodes[6];        // per resource, [5] = 3:1

  void derive() {
    const Topo& T = topo();
    desert = -1;
    for (int t = 0; t < TILES; t++) {
      if (res[t] < 0 && desert < 0) desert = t;
      proba[t] = res[t] < 0 ? 0.0 : T.dice[num[t]];
    }
    for (int n = 0; n < NODES; n++) {
      double pip = 0;
      for (int r = 0; r < 5; r++) node_prod[n][r] = 0;
      for (int k = 0; k < 3; k++) {
        int t = NODE_TILES[n][k];
        if (t < 0 || res[t] < 0) continue;
        node_prod[n][res[t]] += T.dice[num[t]];
        pip += T.dice[num[t]];
      }
      pips[n] = float(pip);
    }
    for (auto& m : port_nodes) m = 0;
    for (int p = 0; p < PORTS; p++) {
      int k = port[p] < 0 ? 5 : port[p];
      port_nodes[k] |= bit(PORT_NODES[p][0]) | bit(PORT_NODES[p][1]);
    }
  }

  // A random base map: shuffled tiles and ports, numbers in the official spiral.
  template <class Rng>
  static Map random(Rng& rng) {
    Map m;
    int tiles[TILES] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, -1};
    int ports[PORTS] = {0, 1, 2, 3, 4, -1, -1, -1, -1};
    rng.shuffle(tiles, TILES);
    rng.shuffle(ports, PORTS);
    for (int t = 0; t < TILES; t++) m.res[t] = tiles[t], m.num[t] = 0;
    for (int p = 0; p < PORTS; p++) m.port[p] = ports[p];
    int i = 0;
    for (int k = 0; k < TILES; k++) {
      int t = SPIRAL[k];
      if (m.res[t] >= 0) m.num[t] = SPIRAL_NUMBERS[i++];
    }
    m.derive();
    return m;
  }
};

// --- Random numbers -----------------------------------------------------------

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed = 1) : s(seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull) {}
  uint64_t next() {  // splitmix64
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  int below(int n) { return int((next() >> 11) * (1.0 / 9007199254740992.0) * n); }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  template <class T>
  void shuffle(T* a, int n) {
    for (int i = n - 1; i > 0; i--) std::swap(a[i], a[below(i + 1)]);
  }
};

// --- State ------------------------------------------------------------------

struct Player {
  int8_t vp, avp;  // VICTORY_POINTS (public), ACTUAL_VICTORY_POINTS
  int8_t roads_av, setts_av, cities_av;
  bool has_road, has_army, has_rolled, played_dev;
  int8_t lr_len;          // LONGEST_ROAD_LENGTH
  bool owned_start[4];    // <card>_OWNED_AT_START, by Dev (not VPC)
  int8_t res[5], dev[5], played[5];
  int8_t setts[5], nsetts;  // buildings_by_color[SETTLEMENT], in order
  int8_t cities[4], ncities;

  int ncards() const { return res[0] + res[1] + res[2] + res[3] + res[4]; }
  int ndevs() const { return dev[0] + dev[1] + dev[2] + dev[3] + dev[4]; }
  bool has(const int* cost) const {
    for (int r = 0; r < 5; r++)
      if (res[r] < cost[r]) return false;
    return true;
  }
  bool can_play(int d) const { return !played_dev && dev[d] >= 1 && owned_start[d]; }
};

struct State {
  const Map* map;
  Player p[NP];
  int8_t bowner[NODES], btype[NODES];  // building owner (-1 none); 1 settlement, 2 city
  int8_t road[EDGES];                  // owner, -1 none
  uint64_t comp[NP][MAXC];             // board.connected_components, in order
  int8_t ncomp[NP];
  uint64_t buildable;                  // board.board_buildable_ids
  int8_t rl[NP], rl_order[NP], n_rl;   // board.road_lengths, in insertion order
  int8_t road_color, road_length;      // board.road_color (-1 none), road_length
  int8_t robber;                       // tile id
  int8_t bank[5];
  int8_t deck[25], ndeck;              // development_listdeck, drawn from the end
  int16_t num_turns;
  int8_t cur, turn, prompt;            // current_player_index, current_turn_index
  bool initial, discarding, moving_knight, road_building;
  int8_t free_roads;
  int8_t discard[NP];                  // discard_counts
  int8_t vps_to_win, discard_limit;

  // A new game on `m` (seat 0 first), deck shuffled with rng.
  template <class R>
  static State fresh(const Map* m, R& rng, int vps_to_win = 10, int discard_limit = 7) {
    State s;
    std::memset(&s, 0, sizeof(s));
    s.map = m;
    for (Player& q : s.p) {
      q.roads_av = 15, q.setts_av = 5, q.cities_av = 4;
    }
    for (int n = 0; n < NODES; n++) s.bowner[n] = -1;
    for (int e = 0; e < EDGES; e++) s.road[e] = -1;
    s.buildable = bit(NODES) - 1;
    s.road_color = -1;
    s.robber = m->desert;
    for (int r = 0; r < 5; r++) s.bank[r] = 19;
    static const int8_t DECK[25] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 3, 3, 2, 2, 4, 4, 4, 4, 4};
    std::memcpy(s.deck, DECK, 25);
    s.ndeck = 25;
    rng.shuffle(s.deck, 25);
    s.prompt = P_INIT_SETTLE;
    s.initial = true;
    s.vps_to_win = vps_to_win, s.discard_limit = discard_limit;
    return s;
  }

  int winner() const {
    int w = -1;
    for (int i = 0; i < NP; i++)
      if (p[i].avp >= vps_to_win) w = i;
    return w;
  }
  bool over() const { return winner() >= 0 || num_turns >= TURNS_LIMIT; }

  // ---- board helpers
  bool enemy_node(int n, int c) const { return bowner[n] >= 0 && bowner[n] != c; }
  int comp_index(int c, int n) const {
    for (int i = 0; i < ncomp[c]; i++)
      if (comp[c][i] & bit(n)) return i;
    return -1;
  }
  uint64_t comp_union(int c) const {
    uint64_t u = 0;
    for (int i = 0; i < ncomp[c]; i++) u |= comp[c][i];
    return u;
  }
  void comp_delete(int c, int i) {
    for (int j = i; j + 1 < ncomp[c]; j++) comp[c][j] = comp[c][j + 1];
    ncomp[c]--;
  }
  void comp_append(int c, uint64_t set) {
    if (ncomp[c] >= MAXC) throw std::runtime_error("too many road components");
    comp[c][ncomp[c]++] = set;
  }
  uint64_t owned_nodes(int c) const {
    uint64_t m = 0;
    for (int i = 0; i < p[c].nsetts; i++) m |= bit(p[c].setts[i]);
    for (int i = 0; i < p[c].ncities; i++) m |= bit(p[c].cities[i]);
    return m;
  }
  uint64_t buildable_nodes(int c) const { return comp_union(c) & buildable; }
  // board.buildable_edges(color): edges at the color's component nodes with no road.
  void buildable_edges(int c, bool* out) const {
    const Topo& T = topo();
    std::memset(out, 0, EDGES);
    uint64_t u = comp_union(c);
    while (u) {
      int n = __builtin_ctzll(u);
      u &= u - 1;
      for (int k = 0; k < 3; k++) {
        int e = T.node_edges[n][k];
        if (e >= 0 && road[e] < 0) out[e] = true;
      }
    }
  }
  bool any_buildable_edge(int c) const {
    bool b[EDGES];
    buildable_edges(c, b);
    for (int e = 0; e < EDGES; e++)
      if (b[e]) return true;
    return false;
  }
  // Port types (resource, 5 = 3:1) at nodes the color has built on.
  int port_mask(int c) const {
    uint64_t own = owned_nodes(c);
    int m = 0;
    for (int k = 0; k < 6; k++)
      if (map->port_nodes[k] & own) m |= 1 << k;
    return m;
  }

  int path_from(int n, unsigned __int128 used, int c) const {
    const Topo& T = topo();
    int best = 0;
    for (int k = 0; k < 3; k++) {
      int v = NODE_NBR[n][k];
      if (v < 0) continue;
      int e = T.node_edges[n][k];
      if (road[e] != c || enemy_node(v, c)) continue;
      unsigned __int128 eb = (unsigned __int128)1 << e;
      if (used & eb) continue;
      best = std::max(best, 1 + path_from(v, used | eb, c));
    }
    return best;
  }
  // board.longest_acyclic_path: the longest trail of c's roads from any node in set.
  int longest_path(uint64_t set, int c) const {
    int best = 0;
    while (set) {
      int n = __builtin_ctzll(set);
      set &= set - 1;
      best = std::max(best, path_from(n, 0, c));
    }
    return best;
  }
  // board.dfs_walk: nodes reached from n along c's roads (enemy nodes included, not passed).
  uint64_t dfs_walk(int n0, int c) const {
    const Topo& T = topo();
    uint64_t visited = 0;
    int agenda[256], na = 0;
    agenda[na++] = n0;
    while (na) {
      int n = agenda[--na];
      visited |= bit(n);
      if (enemy_node(n, c)) continue;
      for (int k = 0; k < 3; k++) {
        int v = NODE_NBR[n][k];
        if (v < 0 || (visited & bit(v))) continue;
        if (road[T.node_edges[n][k]] == c && na < 256) agenda[na++] = v;
      }
    }
    return visited;
  }
  void rl_touch(int c) {  // road_lengths[c] on a defaultdict: inserted on first access
    for (int i = 0; i < n_rl; i++)
      if (rl_order[i] == c) return;
    rl_order[n_rl++] = c;
    rl[c] = 0;
  }

  // Board.build_settlement; returns the previous road_color.
  int board_settle(int c, int n, bool init) {
    const Topo& T = topo();
    bowner[n] = c, btype[n] = 1;
    int prev = road_color;
    if (init) {
      comp_append(c, bit(n));
    } else {
      // Group the node's edges by road color, in adjacency order.
      int colors[3], ncol = 0, edges[3][3], ne[3] = {0, 0, 0};
      for (int k = 0; k < 3; k++) {
        if (NODE_NBR[n][k] < 0) continue;
        int e = T.node_edges[n][k], rc = road[e];
        int j = 0;
        while (j < ncol && colors[j] != rc) j++;
        if (j == ncol) colors[ncol++] = rc;
        edges[j][ne[j]++] = k;
      }
      for (int j = 0; j < ncol; j++) {
        int ec = colors[j];
        if (ec == c || ec < 0) continue;
        if (ne[j] == 2) {
          int a = NODE_NBR[n][edges[j][0]], cc = NODE_NBR[n][edges[j][1]];
          uint64_t aset = dfs_walk(a, ec), cset = dfs_walk(cc, ec);
          int bi = comp_index(ec, n);
          comp_delete(ec, bi);
          comp_append(ec, aset);
          comp_append(ec, cset);
          int best = 0;
          for (int i = 0; i < ncomp[ec]; i++) best = std::max(best, longest_path(comp[ec][i], ec));
          rl[ec] = best;
          int bc = -1, bl = -1;
          for (int i = 0; i < n_rl; i++)
            if (rl[rl_order[i]] > bl) bc = rl_order[i], bl = rl[rl_order[i]];
          road_color = bc, road_length = bl;
        } else if (ne[j] == 1) {
          int bi = comp_index(ec, n);
          if (bi >= 0) comp[ec][bi] &= ~bit(n);
        }
      }
    }
    buildable &= ~(bit(n) | T.nbr_mask[n]);
    return prev;
  }

  // Board.build_road; returns the previous road_color.
  int board_road(int c, int e) {
    road[e] = c;
    int a = EDGE_NODES[e][0], b = EDGE_NODES[e][1];
    int ai = comp_index(c, a), bi = comp_index(c, b);
    int ci;
    if (ai < 0 && !enemy_node(a, c)) {
      ci = bi;
      comp[c][ci] |= bit(a);
    } else if (bi < 0 && !enemy_node(b, c)) {
      ci = ai;
      comp[c][ci] |= bit(b);
    } else if (ai >= 0 && bi >= 0 && ai != bi) {
      comp[c][ai] |= comp[c][bi];
      comp_delete(c, bi);
      ci = ai < bi ? ai : ai - 1;
    } else {
      ci = ai >= 0 ? ai : bi;
    }
    int prev = road_color;
    int cand = longest_path(comp[c][ci], c);
    rl_touch(c);
    rl[c] = std::max<int>(rl[c], cand);
    if (cand >= 5 && cand > road_length) road_color = c, road_length = cand;
    return prev;
  }

  // state_functions.maintain_longest_road.
  void maintain_road(int prev) {
    for (int i = 0; i < n_rl; i++) p[rl_order[i]].lr_len = rl[rl_order[i]];
    if (road_color < 0 || prev == road_color) return;
    Player& w = p[road_color];
    w.has_road = true, w.vp += 2, w.avp += 2;
    if (prev >= 0) {
      Player& l = p[prev];
      l.has_road = false, l.vp -= 2, l.avp -= 2;
    }
  }

  void pay(Player& q, const int* cost) {
    for (int r = 0; r < 5; r++) q.res[r] -= cost[r], bank[r] += cost[r];
  }
  void advance(int dir = 1) {
    cur = int8_t(((cur + dir) % NP + NP) % NP);
    turn = cur;
    num_turns++;
  }
  void play_dev(int c, int d) {
    int prev = -1, prev_size = 0;
    if (d == KNIGHT)
      for (int i = 0; i < NP; i++)
        if (p[i].has_army) {
          prev = i, prev_size = p[i].played[KNIGHT];
          break;
        }
    Player& q = p[c];
    q.dev[d]--;
    q.played_dev = true;
    q.played[d]++;
    if (d != KNIGHT) return;
    int size = q.played[KNIGHT];  // maintain_largest_army
    if (size < 3) return;
    if (prev < 0) {
      q.has_army = true, q.vp += 2, q.avp += 2;
    } else if (prev_size < size && prev != c) {
      q.has_army = true, q.vp += 2, q.avp += 2;
      p[prev].has_army = false, p[prev].vp -= 2, p[prev].avp -= 2;
    }
  }

  // ---- moves

  // Legal moves in Catanatron's order (where it has one); returns the count.
  int legal(int* out) const {
    const Topo& T = topo();
    int n = 0, c = cur;
    const Player& q = p[c];
    bool eb[EDGES];
    switch (prompt) {
      case P_INIT_SETTLE: {
        uint64_t b = buildable;
        while (b) {
          int x = __builtin_ctzll(b);
          b &= b - 1;
          out[n++] = T.settle[x];
        }
        return n;
      }
      case P_INIT_ROAD: {
        int last = q.setts[q.nsetts - 1];
        buildable_edges(c, eb);
        for (int e = 0; e < EDGES; e++)
          if (eb[e] && (EDGE_NODES[e][0] == last || EDGE_NODES[e][1] == last)) out[n++] = T.road[e];
        return n;
      }
      case P_ROBBER: {
        for (int t = 0; t < TILES; t++) {
          if (t == robber) continue;
          bool steal = false;
          for (int k = 0; k < 6; k++) {
            int o = bowner[TILE_NODES[t][k]];
            if (o >= 0 && o != c && p[o].ncards() >= 1) steal = true;
          }
          out[n++] = T.robber[t][steal];
        }
        return n;
      }
      case P_DISCARD: {
        if (discard[c] <= 0) return 0;
        for (int r = 0; r < 5; r++)
          if (q.res[r] > 0) out[n++] = T.discard[r];
        return n;
      }
      case P_PLAY: break;
      default: throw std::runtime_error("trade prompts are not supported");
    }
    if (road_building) {
      if (q.roads_av <= 0) return 0;
      buildable_edges(c, eb);
      for (int e = 0; e < EDGES; e++)
        if (eb[e]) out[n++] = T.road[e];
      return n;
    }
    if (q.can_play(YOP)) {
      bool seen[NUM_ACTIONS] = {};
      for (int i = 0; i < 5; i++)
        for (int j = i; j < 5; j++) {
          bool both = i == j ? bank[i] >= 2 : bank[i] >= 1 && bank[j] >= 1;
          int cands[2], nc = 0;
          if (both) cands[nc++] = T.yop[i][j];
          else {
            if (bank[i] >= 1) cands[nc++] = T.yop[i][5];
            if (bank[j] >= 1) cands[nc++] = T.yop[j][5];
          }
          for (int k = 0; k < nc; k++)
            if (!seen[cands[k]]) seen[cands[k]] = true, out[n++] = cands[k];
        }
    }
    if (q.can_play(MONO))
      for (int r = 0; r < 5; r++) out[n++] = T.mono[r];
    if (q.can_play(KNIGHT)) out[n++] = T.knight;
    if (q.can_play(RB) && q.roads_av > 0 && any_buildable_edge(c)) out[n++] = T.rb;
    if (!q.has_rolled) {
      out[n++] = T.roll;
      return n;
    }
    out[n++] = T.end;
    if (q.roads_av > 0 && q.has(ROAD_COST)) {
      buildable_edges(c, eb);
      for (int e = 0; e < EDGES; e++)
        if (eb[e]) out[n++] = T.road[e];
    }
    if (q.has(SETTLE_COST) && q.setts_av > 0) {
      uint64_t b = buildable_nodes(c);
      while (b) {
        int x = __builtin_ctzll(b);
        b &= b - 1;
        out[n++] = T.settle[x];
      }
    }
    if (q.has(CITY_COST) && q.cities_av > 0)
      for (int i = 0; i < q.nsetts; i++) out[n++] = T.city[q.setts[i]];
    if (q.has(DEV_COST) && ndeck > 0) out[n++] = T.buy;
    int ports = port_mask(c);
    int rate[5];
    for (int r = 0; r < 5; r++) rate[r] = (ports & 32) ? 3 : 4;
    for (int r = 0; r < 5; r++)
      if (ports & (1 << r)) rate[r] = 2;
    for (int r = 0; r < 5; r++) {
      if (q.res[r] < rate[r]) continue;
      for (int j = 0; j < 5; j++)
        if (j != r && bank[j] > 0) out[n++] = T.trade[r][j][rate[r]];
    }
    return n;
  }

  // apply_action. outcome: the dice sum for a roll, the card for a purchase,
  // the resource for a steal; -1 draws it (dice and steals from rng, the
  // card from the top of the deck). Returns false if a given card is not in
  // the deck (nothing is changed then).
  template <class R>
  bool apply(int a, int outcome, R& rng) {
    const int* A = ACTIONS[a];
    int c = cur;
    Player& q = p[c];
    switch (A[0]) {
      case A_END: {
        q.played_dev = false, q.has_rolled = false;
        for (int d = 0; d < 4; d++) q.owned_start[d] = q.dev[d] > 0;
        advance();
        prompt = P_PLAY;
        break;
      }
      case A_SETTLE: {
        int n = A[1];
        if (initial) {
          board_settle(c, n, true);
          q.setts[q.nsetts++] = n, q.setts_av--, q.vp++, q.avp++;
          if (q.nsetts == 2)
            for (int k = 0; k < 3; k++) {
              int t = NODE_TILES[n][k];
              if (t < 0 || map->res[t] < 0) continue;
              bank[map->res[t]]--, q.res[map->res[t]]++;
            }
          prompt = P_INIT_ROAD;
        } else {
          int prev = board_settle(c, n, false);
          q.setts[q.nsetts++] = n, q.setts_av--, q.vp++, q.avp++;
          pay(q, SETTLE_COST);
          maintain_road(prev);
        }
        break;
      }
      case A_ROAD: {
        int e = A[1];
        if (initial) {
          board_road(c, e);
          q.roads_av--;
          int nb = p[0].nsetts + p[1].nsetts;
          if (nb < NP) advance(), prompt = P_INIT_SETTLE;
          else if (nb == NP) prompt = P_INIT_SETTLE;
          else if (nb == 2 * NP) initial = false, prompt = P_PLAY;
          else advance(-1), prompt = P_INIT_SETTLE;
        } else if (road_building && free_roads > 0) {
          int prev = board_road(c, e);
          q.roads_av--;
          maintain_road(prev);
          free_roads--;
          if (free_roads == 0 || q.roads_av <= 0 || !any_buildable_edge(c)) road_building = false, free_roads = 0;
        } else {
          int prev = board_road(c, e);
          q.roads_av--;
          pay(q, ROAD_COST);
          maintain_road(prev);
        }
        break;
      }
      case A_CITY: {
        int n = A[1];
        btype[n] = 2;
        int i = 0;
        while (q.setts[i] != n) i++;
        for (; i + 1 < q.nsetts; i++) q.setts[i] = q.setts[i + 1];
        q.nsetts--;
        q.cities[q.ncities++] = n;
        q.setts_av++, q.cities_av--, q.vp++, q.avp++;
        pay(q, CITY_COST);
        break;
      }
      case A_BUY: {
        int card;
        if (outcome < 0 || deck[ndeck - 1] == outcome) {
          card = deck[--ndeck];
        } else {
          int i = 0;
          while (i < ndeck && deck[i] != outcome) i++;
          if (i == ndeck) return false;
          for (; i + 1 < ndeck; i++) deck[i] = deck[i + 1];
          ndeck--;
          card = outcome;
        }
        q.dev[card]++;
        if (card == VPC) q.avp++;
        pay(q, DEV_COST);
        break;
      }
      case A_ROLL: {
        q.has_rolled = true;
        int number = outcome >= 0 ? outcome : 2 + rng.below(6) + rng.below(6);
        if (number == 7) {
          int first = -1;
          for (int i = 0; i < NP; i++) {
            int nc = p[i].ncards();
            discard[i] = nc > discard_limit ? nc / 2 : 0;
            if (discard[i] > 0 && first < 0) first = i;
          }
          if (first >= 0) {
            cur = first, prompt = P_DISCARD, discarding = true;
          } else {
            for (int i = 0; i < NP; i++) discard[i] = 0;
            prompt = P_ROBBER, moving_knight = true;
          }
        } else {
          int pay_[NP][5] = {}, tot[5] = {};
          for (int t = 0; t < TILES; t++) {
            if (map->num[t] != number || t == robber || map->res[t] < 0) continue;
            int r = map->res[t];
            for (int k = 0; k < 6; k++) {
              int nd = TILE_NODES[t][k], o = bowner[nd];
              if (o < 0) continue;
              pay_[o][r] += btype[nd], tot[r] += btype[nd];
            }
          }
          for (int r = 0; r < 5; r++) {
            if (bank[r] < tot[r]) continue;  // depleted: nobody gets it
            for (int i = 0; i < NP; i++) p[i].res[r] += pay_[i][r], bank[r] -= pay_[i][r];
          }
          prompt = P_PLAY;
        }
        break;
      }
      case A_DISCARD: {
        int r = A[1];
        q.res[r]--, bank[r]++;
        discard[c]--;
        if (discard[c] <= 0) {
          int next = -1;
          for (int i = cur + 1; i < NP; i++)
            if (discard[i] > 0) {
              next = i;
              break;
            }
          if (next >= 0) cur = next;
          else {
            cur = turn, prompt = P_ROBBER, discarding = false, moving_knight = true;
            for (int i = 0; i < NP; i++) discard[i] = 0;
          }
        }
        break;
      }
      case A_ROBBER: {
        if (A[2]) {
          int v = 1 - c;
          int r = outcome;
          if (r < 0) {  // random card from the victim's hand
            int k = rng.below(p[v].ncards());
            r = 0;
            while (k >= p[v].res[r]) k -= p[v].res[r], r++;
          }
          p[v].res[r]--, q.res[r]++;
        }
        robber = A[1];
        prompt = P_PLAY;
        break;
      }
      case A_KNIGHT: {
        play_dev(c, KNIGHT);
        prompt = P_ROBBER;
        break;
      }
      case A_YOP: {
        int r1 = A[1], r2 = A[2];
        q.res[r1]++, bank[r1]--;
        if (r2 >= 0) q.res[r2]++, bank[r2]--;
        play_dev(c, YOP);
        prompt = P_PLAY;
        break;
      }
      case A_MONO: {
        int r = A[1];
        for (int i = 0; i < NP; i++)
          if (i != c) q.res[r] += p[i].res[r], p[i].res[r] = 0;
        play_dev(c, MONO);
        prompt = P_PLAY;
        break;
      }
      case A_RB: {
        play_dev(c, RB);
        road_building = true, free_roads = 2;
        prompt = P_PLAY;
        break;
      }
      case A_TRADE: {
        int give = A[1], get = A[2], rate = A[3];
        q.res[give] -= rate, bank[give] += rate;
        q.res[get]++, bank[get]--;
        prompt = P_PLAY;
        break;
      }
    }
    return true;
  }

  // ---- chance (cbot.chance.execute_spectrum)

  // Outcomes of move a: (outcome for apply, probability).
  int spectrum(int a, int* outs, double* probs) const {
    const int* A = ACTIONS[a];
    if (A[0] == A_ROLL) {
      for (int s = 2; s <= 12; s++) outs[s - 2] = s, probs[s - 2] = (6 - std::abs(s - 7)) / 36.0;
      return 11;
    }
    if (A[0] == A_BUY) {
      // Unseen cards from the buyer's view: the deck plus the other hand. A
      // card not in the deck is drawn from its top instead (as cbot.chance).
      int cnt[5] = {}, tot = 0;
      for (int i = 0; i < ndeck; i++) cnt[deck[i]]++, tot++;
      const Player& o = p[1 - cur];
      for (int d = 0; d < 5; d++) cnt[d] += o.dev[d], tot += o.dev[d];
      int n = 0;
      for (int d = 0; d < 5; d++)
        if (cnt[d]) outs[n] = d, probs[n] = double(cnt[d]) / tot, n++;
      return n;
    }
    if (A[0] == A_ROBBER && A[2]) {
      const Player& v = p[1 - cur];
      int tot = v.ncards(), n = 0;
      if (tot == 0) {
        outs[0] = -1, probs[0] = 1.0;
        return 1;
      }
      for (int r = 0; r < 5; r++)
        if (v.res[r] > 0) outs[n] = r, probs[n] = double(v.res[r]) / tot, n++;
      return n;
    }
    outs[0] = -1, probs[0] = 1.0;
    return 1;
  }

  // Applies a chance outcome from spectrum(); a card missing from the deck is
  // drawn from its top.
  template <class R>
  void apply_outcome(int a, int outcome, R& rng) {
    if (!apply(a, outcome, rng)) apply(a, -1, rng);
  }

  // ---- features (cbot.feats.encode, 1v1 layout, before the sorting into
  // feature_names order; see canonical_names in cbot/native.py)

  void production(int c, bool robber_on, double* out) const {
    // features.build_production_features(True): settlements then 2x cities.
    for (int r = 0; r < 5; r++) out[r] = 0;
    for (int r = 0; r < 5; r++) {
      double s = 0;
      for (int i = 0; i < p[c].nsetts; i++) s += node_production(p[c].setts[i], r, robber_on);
      for (int i = 0; i < p[c].ncities; i++) s += 2 * node_production(p[c].cities[i], r, robber_on);
      out[r] = s;
    }
  }
  double node_production(int n, int r, bool robber_on) const {
    const Topo& T = topo();
    double s = 0;
    for (int k = 0; k < 3; k++) {
      int t = NODE_TILES[n][k];
      if (t < 0 || map->res[t] != r || (robber_on && t == robber)) continue;
      s += T.dice[map->num[t]];
    }
    return s;
  }
  void count_production(uint64_t nodes, double* out) const {
    for (int r = 0; r < 5; r++) out[r] = 0;
    while (nodes) {
      int n = __builtin_ctzll(nodes);
      nodes &= nodes - 1;
      for (int r = 0; r < 5; r++) out[r] += map->node_prod[n][r];
    }
  }
  // features.reachability_features for one color: [level][resource], levels 0-2.
  void reachability(int c, double out[3][5]) const {
    const Topo& T = topo();
    uint64_t oob = owned_nodes(c) | buildable;
    uint64_t zero = comp_union(c);
    count_production(oob & zero, out[0]);
    uint64_t enemy = 0;
    for (int n = 0; n < NODES; n++)
      if (bowner[n] >= 0 && bowner[n] != c) enemy |= bit(n);
    uint64_t last = zero;
    for (int level = 1; level <= 2; level++) {
      uint64_t lv = last, it = last & ~enemy;
      while (it) {
        int n = __builtin_ctzll(it);
        it &= it - 1;
        for (int k = 0; k < 3; k++) {
          int v = NODE_NBR[n][k];
          if (v < 0) continue;
          int ow = road[T.node_edges[n][k]];
          if (ow < 0 || ow == c) lv |= bit(v);
        }
      }
      count_production(oob & lv, out[level]);
      last = lv;
    }
  }

  // Features from seat p0's view, in canonical order (DIM floats).
  void encode_canonical(int p0, float* f) const {
    int k = 0;
    int seat[NP] = {p0, 1 - p0};
    auto put = [&](double x) { f[k++] = float(x); };
    put(p[p0].avp);
    for (int i = 0; i < NP; i++) {
      const Player& q = p[seat[i]];
      put(q.vp), put(q.has_army), put(q.has_road), put(q.roads_av), put(q.setts_av), put(q.cities_av),
          put(q.has_rolled), put(q.lr_len);
    }
    const Player& me = p[p0];
    for (int r = 0; r < 5; r++) put(me.res[r]);
    for (int d = 0; d < 5; d++) put(me.dev[d]);
    put(me.played_dev);
    for (int i = 0; i < NP; i++) {
      const Player& q = p[seat[i]];
      for (int d = 0; d < 4; d++) put(q.played[d]);
      put(q.ncards()), put(q.ndevs());
    }
    for (int t = 0; t < TILES; t++) {
      for (int r = 0; r < 5; r++) put(map->res[t] == r);
      put(map->res[t] < 0), put(map->proba[t]), put(t == robber);
    }
    for (int pt = 0; pt < PORTS; pt++) {
      for (int r = 0; r < 5; r++) put(map->port[pt] == r);
      put(map->port[pt] < 0);
    }
    for (int i = 0; i < NP; i++) {
      for (int n = 0; n < NODES; n++) put(bowner[n] == seat[i] && btype[n] == 1), put(bowner[n] == seat[i] && btype[n] == 2);
      for (int e = 0; e < EDGES; e++) put(road[e] == seat[i]);
    }
    put(ndeck), put(prompt == P_ROBBER), put(prompt == P_DISCARD);
    for (int r = 0; r < 5; r++) put(bank[r]);
    double prod[NP][5];
    for (int i = 0; i < NP; i++) production(seat[i], true, prod[i]);
    for (int r = 0; r < 5; r++)
      for (int i = 0; i < NP; i++) put(prod[i][r]);
    for (int i = 0; i < NP; i++) {
      double reach[3][5];
      reachability(seat[i], reach);
      for (int l = 0; l < 3; l++)
        for (int r = 0; r < 5; r++) put(reach[l][r]);
    }
    put(cur == p0), put(p0 == 0), put(num_turns / 100.0), put(initial);
    for (int pr = 0; pr < 7; pr++) put(prompt == pr);
    for (int i = 0; i < NP; i++) put(popcount(buildable_nodes(seat[i])) / 10.0);
    for (int n = 0; n < NODES; n++) f[k++] = map->pips[n];
    for (int r = 0; r < 5; r++) put(p[1 - p0].res[r]);
  }

  // players.value.base_fn (DEFAULT_WEIGHTS) for seat c.
  double base_value(int c) const {
    double prod[NP][5];
    production(c, true, prod[0]);
    production(1 - c, true, prod[1]);
    static const int ORDER[5] = {WHEAT, ORE, SHEEP, WOOD, BRICK};
    double production = 0, enemy = 0;
    int variety = 0;
    for (int r : ORDER) production += prod[0][r], variety += prod[0][r] != 0, enemy += prod[1][r];
    production += variety * 4 * (2.778 / 100);
    const Player& q = p[c];
    double reach[3][5];
    reachability(c, reach);
    double r0 = 0, r1 = 0;
    for (int r = 0; r < 5; r++) r0 += reach[0][r];
    for (int r = 0; r < 5; r++) r1 += reach[1][r];
    double to_city = (std::max(2 - q.res[WHEAT], 0) + std::max(3 - q.res[ORE], 0)) / 5.0;
    double to_settle = (std::max(1 - q.res[WHEAT], 0) + std::max(1 - q.res[SHEEP], 0) + std::max(1 - q.res[BRICK], 0) +
                        std::max(1 - q.res[WOOD], 0)) / 4.0;
    double synergy = (2 - to_city - to_settle) / 2;
    int in_hand = q.ncards();
    double discard_penalty = in_hand > 7 ? -5 : 0;
    uint32_t tiles = 0;
    uint64_t own = owned_nodes(c);
    while (own) {
      int n = __builtin_ctzll(own);
      own &= own - 1;
      for (int k = 0; k < 3; k++)
        if (NODE_TILES[n][k] >= 0) tiles |= 1u << NODE_TILES[n][k];
    }
    int nbuild = popcount(buildable_nodes(c));
    double lr_factor = nbuild == 0 ? 10 : 0.1;
    return q.vp * 3e14 + production * 1e8 + enemy * -1e8 + r0 * 0 + r1 * 1e4 + synergy * 1e2 + nbuild * 1e3 +
           __builtin_popcount(tiles) * 1 + in_hand * 1 + discard_penalty + q.lr_len * lr_factor + q.ndevs() * 10 +
           q.played[KNIGHT] * 10.1;
  }
};

}  // namespace ce
