// Python bindings: cbot._engine (built by cbot/engine/build.sh).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "catan.h"
#include "search.h"

namespace py = pybind11;
using namespace ce;

namespace ce {
int LAYOUT[MAXP + 1][DIM];  // n players: canonical feature k goes to LAYOUT[n][k] (cbot.feats order)
}

namespace {

bool HAVE_LAYOUT[MAXP + 1] = {};

void need_layout(int n) {
  if (n < 2 || n > MAXP || !HAVE_LAYOUT[n]) throw std::runtime_error("call set_layout first (import cbot.native)");
}

struct PyGame {
  std::shared_ptr<Map> map;
  State s;
  Rng rng;

  PyGame copy() const { return *this; }
};

template <class T>
std::vector<int> vec(const T* a, int n) {
  return std::vector<int>(a, a + n);
}
std::vector<int> bits(uint64_t m) {
  std::vector<int> v;
  for (int i = 0; i < 64; i++)
    if (m >> i & 1) v.push_back(i);
  return v;
}
uint64_t mask(const std::vector<int>& v) {
  uint64_t m = 0;
  for (int i : v) m |= bit(i);
  return m;
}
template <class T>
void fill(T* a, const py::handle& h, int n) {
  auto v = h.cast<std::vector<int>>();
  if (int(v.size()) != n) throw std::runtime_error("bad length");
  for (int i = 0; i < n; i++) a[i] = T(v[i]);
}

py::dict to_dict(const PyGame& g) {
  const State& s = g.s;
  py::dict d, m;
  m["res"] = vec(g.map->res, TILES), m["num"] = vec(g.map->num, TILES), m["port"] = vec(g.map->port, PORTS);
  d["map"] = m;
  py::list players;
  for (int i = 0; i < s.np; i++) {
    const Player& q = s.p[i];
    py::dict p;
    p["vp"] = q.vp, p["avp"] = q.avp, p["roads_av"] = q.roads_av, p["setts_av"] = q.setts_av,
    p["cities_av"] = q.cities_av, p["has_road"] = q.has_road, p["has_army"] = q.has_army,
    p["has_rolled"] = q.has_rolled, p["played_dev"] = q.played_dev, p["lr_len"] = q.lr_len;
    p["owned_start"] = std::vector<bool>(q.owned_start, q.owned_start + 4);
    p["res"] = vec(q.res, 5), p["dev"] = vec(q.dev, 5), p["played"] = vec(q.played, 5);
    p["setts"] = vec(q.setts, q.nsetts), p["cities"] = vec(q.cities, q.ncities);
    players.append(p);
  }
  d["players"] = players;
  d["bowner"] = vec(s.bowner, NODES), d["btype"] = vec(s.btype, NODES), d["road"] = vec(s.road, EDGES);
  py::list comps;
  for (int c = 0; c < s.np; c++) {
    py::list cs;
    for (int i = 0; i < s.ncomp[c]; i++) cs.append(bits(s.comp[c][i]));
    comps.append(cs);
  }
  d["comps"] = comps;
  d["buildable"] = bits(s.buildable);
  py::list rl;
  for (int i = 0; i < s.n_rl; i++) rl.append(py::make_tuple(int(s.rl_order[i]), int(s.rl[s.rl_order[i]])));
  d["rl"] = rl;
  d["road_color"] = s.road_color, d["road_length"] = s.road_length, d["robber"] = s.robber;
  d["bank"] = vec(s.bank, 5), d["deck"] = vec(s.deck, s.ndeck);
  d["num_turns"] = s.num_turns, d["cur"] = s.cur, d["turn"] = s.turn, d["prompt"] = s.prompt;
  d["initial"] = s.initial, d["discarding"] = s.discarding, d["moving_knight"] = s.moving_knight;
  d["road_building"] = s.road_building, d["free_roads"] = s.free_roads, d["discard"] = vec(s.discard, s.np);
  d["vps_to_win"] = s.vps_to_win, d["discard_limit"] = s.discard_limit;
  return d;
}

PyGame from_dict(const py::dict& d, uint64_t seed) {
  PyGame g;
  g.rng = Rng(seed);
  g.map = std::make_shared<Map>();
  py::dict m = d["map"];
  fill(g.map->res, m["res"], TILES), fill(g.map->num, m["num"], TILES), fill(g.map->port, m["port"], PORTS);
  g.map->derive();
  State& s = g.s;
  std::memset(&s, 0, sizeof(s));
  s.map = g.map.get();
  py::list players = d["players"];
  if (players.size() < 2 || players.size() > MAXP) throw std::runtime_error("2 to 4 players");
  s.np = int8_t(players.size());
  for (int i = 0; i < s.np; i++) {
    py::dict p = players[i];
    Player& q = s.p[i];
    q.vp = p["vp"].cast<int>(), q.avp = p["avp"].cast<int>(), q.roads_av = p["roads_av"].cast<int>();
    q.setts_av = p["setts_av"].cast<int>(), q.cities_av = p["cities_av"].cast<int>();
    q.has_road = p["has_road"].cast<bool>(), q.has_army = p["has_army"].cast<bool>();
    q.has_rolled = p["has_rolled"].cast<bool>(), q.played_dev = p["played_dev"].cast<bool>();
    q.lr_len = p["lr_len"].cast<int>();
    auto os = p["owned_start"].cast<std::vector<bool>>();
    for (int k = 0; k < 4; k++) q.owned_start[k] = os[k];
    fill(q.res, p["res"], 5), fill(q.dev, p["dev"], 5), fill(q.played, p["played"], 5);
    auto st = p["setts"].cast<std::vector<int>>(), ci = p["cities"].cast<std::vector<int>>();
    q.nsetts = int8_t(st.size()), q.ncities = int8_t(ci.size());
    for (size_t k = 0; k < st.size(); k++) q.setts[k] = int8_t(st[k]);
    for (size_t k = 0; k < ci.size(); k++) q.cities[k] = int8_t(ci[k]);
  }
  fill(s.bowner, d["bowner"], NODES), fill(s.btype, d["btype"], NODES), fill(s.road, d["road"], EDGES);
  py::list comps = d["comps"];
  for (int c = 0; c < s.np; c++) {
    py::list cs = comps[c];
    s.ncomp[c] = 0;
    for (auto h : cs) s.comp_append(c, mask(h.cast<std::vector<int>>()));
  }
  s.buildable = mask(d["buildable"].cast<std::vector<int>>());
  s.n_rl = 0;
  for (auto h : py::list(d["rl"])) {
    auto t = h.cast<std::pair<int, int>>();
    s.rl_order[s.n_rl++] = int8_t(t.first);
    s.rl[t.first] = int8_t(t.second);
  }
  s.road_color = d["road_color"].cast<int>(), s.road_length = d["road_length"].cast<int>();
  s.robber = d["robber"].cast<int>();
  fill(s.bank, d["bank"], 5);
  auto deck = d["deck"].cast<std::vector<int>>();
  s.ndeck = int8_t(deck.size());
  for (size_t k = 0; k < deck.size(); k++) s.deck[k] = int8_t(deck[k]);
  s.num_turns = d["num_turns"].cast<int>(), s.cur = d["cur"].cast<int>(), s.turn = d["turn"].cast<int>();
  s.prompt = d["prompt"].cast<int>();
  s.initial = d["initial"].cast<bool>(), s.discarding = d["discarding"].cast<bool>();
  s.moving_knight = d["moving_knight"].cast<bool>(), s.road_building = d["road_building"].cast<bool>();
  s.free_roads = d["free_roads"].cast<int>();
  fill(s.discard, d["discard"], s.np);
  s.vps_to_win = d["vps_to_win"].cast<int>(), s.discard_limit = d["discard_limit"].cast<int>();
  return g;
}

py::array_t<float> py_encode(const PyGame& g, int p0) {
  need_layout(g.s.np);
  py::array_t<float> out(dim(g.s.np));
  ce::encode(g.s, p0, out.mutable_data());
  return out;
}

using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

MLP mlp_from(py::dict d) {
  MLP m;
  m.cols = d["cols"].cast<std::vector<int>>();
  for (auto h : py::list(d["layers"])) {
    auto t = h.cast<py::tuple>();
    auto W = t[0].cast<FArray>(), b = t[1].cast<FArray>();
    Layer L;
    L.in = int(W.shape(0)), L.out = int(W.shape(1));
    if (int(b.size()) != L.out) throw std::runtime_error("bias size");
    L.W.assign(W.data(), W.data() + W.size());
    L.b.assign(b.data(), b.data() + b.size());
    m.layers.push_back(std::move(L));
  }
  return m;
}

Params params_from(py::dict d) {
  Params p;
  for (auto item : d) {
    std::string k = item.first.cast<std::string>();
    py::handle v = item.second;
    if (k == "sims") p.sims = v.cast<int>();
    else if (k == "cpuct") p.cpuct = v.cast<double>();
    else if (k == "batch") p.batch = v.cast<int>();
    else if (k == "ptemp") p.ptemp = v.cast<double>();
    else if (k == "pmix") p.pmix = v.cast<double>();
    else if (k == "fpu") p.fpu = v.cast<double>();
    else if (k == "noise") p.noise = v.cast<double>();
    else if (k == "alpha") p.alpha = v.cast<double>();
    else if (k == "pick") p.pick = v.cast<int>();
    else if (k == "initf") p.initf = v.cast<int>();
    else if (k == "fair") p.fair = v.cast<int>();
    else throw std::runtime_error("unknown search parameter " + k);
  }
  return p;
}

struct PyNets {
  std::shared_ptr<Nets> nets;
};

struct PySearcher {
  std::shared_ptr<Nets> nets;
  std::unique_ptr<Searcher> s;
};

py::dict decision_dict(const Decision& d) {
  py::dict o;
  o["move"] = d.move, o["searched"] = d.searched, o["moves"] = d.moves, o["visits"] = d.visits, o["q"] = d.q;
  o["vec"] = std::vector<double>(d.vec, d.vec + MAXP);
  return o;
}

// Plays the games on `threads` threads; returns the finished games' records
// stacked as cbot.gen shards hold them (X float32 here): in 1v1 y = 1 if the
// row's seat won and v its search value; with more players y = the winner's
// seat counted from the row's seat and v the search values from it on.
py::dict selfplay(const PyNets& nets, py::dict params, std::vector<uint64_t> seeds, int threads, double p_record,
                  double eps, int np) {
  need_layout(np);
  Params prm = params_from(params);
  std::vector<GameRecord> out(seeds.size());
  {
    py::gil_scoped_release nogil;
    std::atomic<size_t> next(0);
    std::vector<std::thread> pool;
    std::exception_ptr fail;
    std::mutex mu;
    for (int t = 0; t < threads; t++)
      pool.emplace_back([&] {
        try {
          for (size_t i; (i = next++) < seeds.size();)
            out[i] = play_game(nets.nets.get(), prm, seeds[i], np, p_record, eps);
        } catch (...) {
          std::lock_guard<std::mutex> lk(mu);
          fail = std::current_exception();
          next = seeds.size();
        }
      });
    for (auto& th : pool) th.join();
    if (fail) std::rethrow_exception(fail);
  }
  size_t rows = 0, prows = 0;
  int unfinished = 0;
  for (auto& g : out) {
    if (g.winner < 0) {
      unfinished++;
      continue;
    }
    rows += g.who.size(), prows += g.pr.size();
  }
  const size_t D = dim(np);
  py::array_t<float> X({rows, D});
  py::array_t<int8_t> y(rows);
  py::array_t<float> v = np == 2 ? py::array_t<float>(rows) : py::array_t<float>({rows, size_t(np)});
  py::array_t<int64_t> gs(rows);
  py::array_t<int32_t> pr(prows), pn(prows);
  py::array_t<int16_t> pa(prows);
  py::array_t<float> pv(prows);
  size_t r = 0, q = 0;
  for (auto& g : out) {
    if (g.winner < 0) continue;
    std::memcpy(X.mutable_data() + r * D, g.X.data(), g.X.size() * sizeof(float));
    for (size_t k = 0; k < g.pr.size(); k++, q++) {
      pr.mutable_data()[q] = int32_t(r + g.pr[k]);
      pa.mutable_data()[q] = int16_t(g.pa[k]);
      pv.mutable_data()[q] = g.pv[k];
      pn.mutable_data()[q] = g.pn[k];
    }
    for (size_t k = 0; k < g.who.size(); k++, r++) {
      if (np == 2) {
        y.mutable_data()[r] = int8_t(g.who[k] == g.winner);
        v.mutable_data()[r] = g.v[k * 2];
      } else {
        y.mutable_data()[r] = int8_t((g.winner - g.who[k] + np) % np);
        for (int j = 0; j < np; j++) v.mutable_data()[r * np + j] = g.v[k * np + j];
      }
      gs.mutable_data()[r] = int64_t(g.seed);
    }
  }
  py::dict d;
  d["X"] = X, d["y"] = y, d["v"] = v, d["g"] = gs, d["pr"] = pr, d["pa"] = pa, d["pv"] = pv, d["pn"] = pn;
  d["unfinished"] = unfinished;
  return d;
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
  m.doc() = "Native Catan engine (2 to 4 players) and tree search (cbot/engine).";
  m.attr("DIM") = DIM;
  m.attr("NUM_ACTIONS") = NUM_ACTIONS;
  m.attr("NUM_ACTIONS_1V1") = NUM_ACTIONS_1V1;
  m.def("dim", [](int n) { return dim(n); });
  m.def("set_layout", [](int n, py::array_t<int> perm) {
    if (n < 2 || n > MAXP) throw std::runtime_error("2 to 4 players");
    if (perm.size() != dim(n)) throw std::runtime_error("layout must have dim(n) entries");
    for (int k = 0; k < dim(n); k++) LAYOUT[n][k] = perm.data()[k];
    HAVE_LAYOUT[n] = true;
  });

  py::class_<PyGame>(m, "Game")
      .def_static("from_dict", &from_dict, py::arg("d"), py::arg("seed") = 1)
      .def_static("new", [](uint64_t seed, int players) {
        PyGame g;
        g.rng = Rng(seed);
        g.map = std::make_shared<Map>(Map::random(g.rng));
        g.s = State::fresh(g.map.get(), g.rng, players);
        return g;
      }, py::arg("seed"), py::arg("players") = 2)
      .def("copy", &PyGame::copy)
      .def("to_dict", &to_dict)
      .def("legal", [](const PyGame& g) {
        int a[NUM_ACTIONS];
        int n = g.s.legal(a);
        return std::vector<int>(a, a + n);
      })
      .def("apply", [](PyGame& g, int a, int outcome) { return g.s.apply(a, outcome, g.rng); },
           py::arg("action"), py::arg("outcome") = -1)
      .def("spectrum", [](const PyGame& g, int a) {
        int o[16];
        double p[16];
        int n = g.s.spectrum(a, o, p);
        std::vector<std::pair<int, double>> v;
        for (int i = 0; i < n; i++) v.push_back({o[i], p[i]});
        return v;
      })
      .def("apply_outcome", [](PyGame& g, int a, int o) { g.s.apply_outcome(a, o, g.rng); })
      .def("encode", &py_encode)
      .def("base_value", [](const PyGame& g, int c) { return g.s.base_value(c); })
      .def("winner", [](const PyGame& g) { return g.s.winner(); })
      .def_property_readonly("players", [](const PyGame& g) { return int(g.s.np); })
      .def_property_readonly("cur", [](const PyGame& g) { return int(g.s.cur); })
      .def_property_readonly("num_turns", [](const PyGame& g) { return int(g.s.num_turns); })
      .def_property_readonly("prompt", [](const PyGame& g) { return int(g.s.prompt); })
      .def_property_readonly("initial", [](const PyGame& g) { return g.s.initial; });

  py::class_<PyNets>(m, "Nets")
      .def(py::init([](py::list value, py::object policy) {
             PyNets n;
             n.nets = std::make_shared<Nets>();
             for (auto h : value) n.nets->value.push_back(mlp_from(h.cast<py::dict>()));
             if (!policy.is_none()) n.nets->policy = mlp_from(policy.cast<py::dict>()), n.nets->has_policy = true;
             return n;
           }),
           py::arg("value"), py::arg("policy") = py::none())
      .def("values", [](const PyNets& n, FArray X, int players) {
        int B = int(X.shape(0));
        if (X.shape(1) != dim(players)) throw std::runtime_error("rows must have dim(players) features");
        py::array_t<double> out({B, players});
        n.nets->values(X.data(), B, players, out.mutable_data());
        return out;
      }, py::arg("X"), py::arg("players") = 2)
      .def("policy", [](const PyNets& n, FArray X) {
        int B = int(X.shape(0));
        int K = n.nets->policy.out_dim();
        py::array_t<float> out({B, K});
        n.nets->policy.forward(X.data(), B, int(X.shape(1)), out.mutable_data());
        return out;
      });

  py::class_<PySearcher>(m, "Searcher")
      .def(py::init([](const PyNets& n, py::dict params, uint64_t seed) {
             PySearcher s;
             s.nets = n.nets;
             s.s = std::make_unique<Searcher>(n.nets.get(), params_from(params), seed);
             return s;
           }),
           py::arg("nets"), py::arg("params"), py::arg("seed") = 1)
      .def("decide", [](PySearcher& s, const PyGame& g, int me) {
        need_layout(g.s.np);
        Decision d;
        {
          py::gil_scoped_release nogil;
          d = s.s->decide(g.s, me);
        }
        return decision_dict(d);
      });

  m.def("selfplay", &selfplay, py::arg("nets"), py::arg("params"), py::arg("seeds"), py::arg("threads"),
        py::arg("p_record") = 1.0, py::arg("eps") = 0.0, py::arg("players") = 2);
}
