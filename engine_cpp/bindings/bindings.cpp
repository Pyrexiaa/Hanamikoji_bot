#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // std::vector, std::unordered_map auto-conversion
#include <pybind11/operators.h>  // py::self comparison helpers

#include "hana/agent.h"
#include "hana/constants.h"
#include "hana/enums.h"
#include "hana/game_config.h"
#include "hana/game_runner.h"
#include "hana/game_state.h"
#include "hana/info_state.h"
#include "hana/player_state.h"
#include "hana/policy_table.h"
#include "hana/suit.h"

#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace hana;

// -----------------------------------------------------------------------
// PyAgent — trampoline allowing Python to subclass the C++ Agent ABC.
//
// Every virtual call from C++ to Agent::get_action_distribution is routed
// through here, which asks pybind11 to call the Python override.
// -----------------------------------------------------------------------
using Distribution = std::unordered_map<Action, double>;

class PyAgent : public Agent {
public:
    using Agent::Agent;

    Distribution get_action_distribution(const RoundInfoState& info_state) override {
        PYBIND11_OVERRIDE_PURE(
            Distribution,
            Agent,
            get_action_distribution,
            info_state
        );
    }
};

// -----------------------------------------------------------------------
// Variant Suit registry.
//
// The C++ Suit enum is shared by all variants, but each Python variant
// defines its own Suit enum with variant-specific ranks (GEISHA_1 is rank 2
// in tiny_hana, rank 1 in medium_hana).  Variants register their enum via
// _register_suit_enum(); every suit handed to Python is then the variant's
// own member (so .rank, dict keys and == behave as in Python mode), and
// every function taking a suit also accepts those members.
//
// Objects that leave C++ carry a variant tag (num_suits, 0 = unknown) so
// their suits can be resolved: CppCard/PlayerView store it in a field,
// Action objects in the `_num_suits` instance attribute (py::dynamic_attr),
// which leaves the engine's Action struct, == and hash untouched.
// -----------------------------------------------------------------------
static constexpr int MAX_VARIANT_SUITS = SUIT_CAPACITY;   // values of the C++ Suit enum

struct SuitRegistry {
    // Strong references, intentionally never released (outlive interpreter teardown).
    PyObject* type[MAX_VARIANT_SUITS + 1] = {};
    PyObject* member[MAX_VARIANT_SUITS + 1][MAX_VARIANT_SUITS] = {};
};
static SuitRegistry g_suits;

static void register_suit_enum(py::handle cls) {
    const auto n = static_cast<int>(py::len(cls));
    if (n < 1 || n > MAX_VARIANT_SUITS)
        throw py::value_error("Suit enum must have 1.." + std::to_string(MAX_VARIANT_SUITS) + " members");
    PyObject* members[MAX_VARIANT_SUITS] = {};
    for (int v = 0; v < n; v++)
        members[v] = cls(v).release().ptr();          // Enum lookup by value; raises if missing
    Py_XDECREF(g_suits.type[n]);
    g_suits.type[n] = py::reinterpret_borrow<py::object>(cls).release().ptr();
    for (int v = 0; v < MAX_VARIANT_SUITS; v++) {
        Py_XDECREF(g_suits.member[n][v]);
        g_suits.member[n][v] = members[v];
    }
}

// Variant tag of a registered Python Suit class, or 0.
static int8_t registered_num_suits(py::handle obj) {
    PyObject* t = reinterpret_cast<PyObject*>(Py_TYPE(obj.ptr()));
    for (int n = 1; n <= MAX_VARIANT_SUITS; n++)
        if (g_suits.type[n] == t) return static_cast<int8_t>(n);
    return 0;
}

// C++ Suit → the variant's Python Suit member, or the C++ Suit if untagged.
static py::object suit_out(Suit s, int n) {
    const int v = suit_value(s);
    if (n > 0 && n <= MAX_VARIANT_SUITS && v < n && g_suits.member[n][v])
        return py::reinterpret_borrow<py::object>(g_suits.member[n][v]);
    return py::cast(s);
}

// cpp.Suit or registered variant Suit member → C++ Suit.  Returns false if
// obj is neither; *n_out receives the variant tag (0 for cpp.Suit).
static bool try_suit_in(py::handle obj, Suit* out, int8_t* n_out = nullptr) {
    if (py::isinstance<Suit>(obj)) {
        *out = obj.cast<Suit>();
        if (n_out) *n_out = 0;
        return true;
    }
    const int8_t n = registered_num_suits(obj);
    if (n == 0) return false;
    const int v = obj.attr("value").cast<int>();
    if (v < 0 || v >= n) return false;
    *out = static_cast<Suit>(v);
    if (n_out) *n_out = n;
    return true;
}

static Suit suit_in(py::handle obj, int8_t* n_out = nullptr) {
    Suit s;
    if (!try_suit_in(obj, &s, n_out))
        throw py::type_error("expected a Suit, got " + std::string(py::repr(obj)));
    return s;
}

// -----------------------------------------------------------------------
// CppCard — thin wrapper that duck-types Python's Card dataclass.
//
// `.suit` returns the variant's Python Suit member when the card's variant
// is known (num_suits != 0), otherwise the C++ Suit.  Equality and hash
// use the suit only.
// -----------------------------------------------------------------------
struct CppCard {
    Suit   suit;
    int8_t num_suits = 0;   // variant tag; 0 = unknown

    bool operator==(const CppCard& o) const noexcept { return suit == o.suit; }
    bool operator!=(const CppCard& o) const noexcept { return suit != o.suit; }

    std::string repr() const {
        return std::string("Card(") + suit_name(suit) + ")";
    }
};

static std::vector<CppCard> cards_out(const Suit* suits, int count, int n) {
    std::vector<CppCard> v;
    v.reserve(count);
    for (int i = 0; i < count; i++) v.push_back(CppCard{suits[i], static_cast<int8_t>(n)});
    return v;
}

// PlayerView — PlayerState snapshot plus its variant tag (bound as "PlayerState").
struct PlayerView {
    PlayerState ps;
    int8_t      num_suits;
};

// -----------------------------------------------------------------------
// Action variant tags (see registry comment above).
// -----------------------------------------------------------------------
// Interned attribute name, created once (never released, like g_suits).
static PyObject* tag_name() {
    static PyObject* name = PyUnicode_InternFromString("_num_suits");
    return name;
}

static void set_action_tag(py::handle obj, int n) {
    py::int_ v(n);   // small ints are cached by CPython
    if (PyObject_SetAttr(obj.ptr(), tag_name(), v.ptr()) != 0) throw py::error_already_set();
}

static py::object action_out(const Action& a, int n) {
    py::object o = py::cast(a);
    if (n > 0) set_action_tag(o, n);
    return o;
}

static int action_num_suits(py::handle self) {
    PyObject* v = PyObject_GetAttr(self.ptr(), tag_name());
    if (!v) { PyErr_Clear(); return 0; }
    const long n = PyLong_AsLong(v);
    Py_DECREF(v);
    return static_cast<int>(n);
}

static py::list actions_out(const std::vector<Action>& actions, int n) {
    py::list out(actions.size());
    for (std::size_t i = 0; i < actions.size(); i++)
        out[i] = action_out(actions[i], n);
    return out;
}

// cpp.ActionType, or any enum with a matching .name (e.g. Python ActionType).
static ActionType action_type_in(py::handle obj) {
    if (py::isinstance<ActionType>(obj)) return obj.cast<ActionType>();
    if (py::hasattr(obj, "name")) {
        const std::string name = obj.attr("name").cast<std::string>();
        for (int at = 0; at <= static_cast<int>(ActionType::COMPETITION_RESPONSE); at++)
            if (name == action_type_name(static_cast<ActionType>(at)))
                return static_cast<ActionType>(at);
    }
    throw py::type_error("expected an ActionType, got " + std::string(py::repr(obj)));
}

// Card-like (cpp.Card, or anything with a .suit) → C++ Suit plus variant tag.
static Suit card_suit_in(py::handle card, int8_t* n_out) {
    if (py::isinstance<CppCard>(card)) {
        const CppCard& c = card.cast<const CppCard&>();
        *n_out = c.num_suits;
        return c.suit;
    }
    if (py::hasattr(card, "suit")) return suit_in(card.attr("suit"), n_out);
    throw py::type_error("expected a Card, got " + std::string(py::repr(card)));
}

// Python-style Action(action_type, cards, choice=None) → C++ Action.
// As permissive as engine.models.Action: any card count / choice is
// accepted here and illegal combinations are rejected later by
// execute_action().  For SPLIT/DISCARD with two cards the order is
// canonicalized (lower suit value first) and a non-None choice flipped on
// swap — exactly like Python's Action and the C++ factories, so the result
// equals (== and hash) the factory-built action.  Only values the C++
// struct cannot hold raise ValueError.
static Action action_from_parts(py::handle action_type, py::iterable cards, py::handle choice,
                                int8_t* n_out) {
    const ActionType at = action_type_in(action_type);
    std::vector<Suit> suits;
    int8_t n = 0;
    for (py::handle card : cards) {
        int8_t cn = 0;
        suits.push_back(card_suit_in(card, &cn));
        if (n == 0) n = cn;
    }
    *n_out = n;
    if (suits.size() > static_cast<std::size_t>(MAX_ACTION_SUITS))
        throw py::value_error("an Action holds at most " + std::to_string(MAX_ACTION_SUITS) + " cards");

    int c = -1;   // -1 is the C++ encoding of choice=None
    if (!choice.is_none()) {
        c = choice.cast<int>();
        if (c < 0 || c > 127)
            throw py::value_error("choice must be None or in 0..127, got " + std::to_string(c));
    }

    if ((at == ActionType::SPLIT || at == ActionType::DISCARD) && suits.size() == 2 &&
        suit_value(suits[0]) > suit_value(suits[1])) {
        std::swap(suits[0], suits[1]);
        if (c != -1) c = 1 - c;
    }
    // CHANCE_DEAL hands, TRADE_OFF / GIFT cards, COMPETITION pairs.
    Action::canonicalize_cards(at, suits.data(), static_cast<int>(suits.size()));
    return Action::make_unchecked(at, suits.data(), static_cast<int>(suits.size()), c);
}

// -----------------------------------------------------------------------
// CppRng — thin mt19937 wrapper exposed as hana_engine_cpp.Rng.
//
// Python solvers pass one Rng instance across many sample_chance_action
// calls to get reproducible sampling without recreating a generator each
// time.
// -----------------------------------------------------------------------
struct CppRng {
    std::mt19937 rng;
    CppRng()              : rng(std::random_device{}()) {}
    explicit CppRng(uint64_t seed) : rng(seed) {}
    void seed(uint64_t s) { rng.seed(s); }
};

// -----------------------------------------------------------------------
// hash_combine — Boost-style mix used by GameStateKey.__hash__.
// -----------------------------------------------------------------------
static inline void hash_combine(size_t& h, size_t v) noexcept {
    h ^= v + 0x9e3779b9u + (h << 6) + (h >> 2);
}

// -----------------------------------------------------------------------
// Module definition
// -----------------------------------------------------------------------
PYBIND11_MODULE(hana_engine_cpp, m) {
    m.doc() = "Hana game engine — C++ backend";
    m.attr("ENGINE_VERSION") = 1;
    m.attr("CHANCE_PLAYER")  = CHANCE_PLAYER;
    m.attr("NO_PLAYER")      = NO_PLAYER;

    // -----------------------------------------------------------------
    // Suit enum
    // Mirrors Python's Suit enum; .value == 0..6, .name == "GEISHA_X".
    // -----------------------------------------------------------------
    auto suit_cls = py::enum_<Suit>(m, "Suit")
        .value("GEISHA_1", Suit::GEISHA_1)
        .value("GEISHA_2", Suit::GEISHA_2)
        .value("GEISHA_3", Suit::GEISHA_3)
        .value("GEISHA_4", Suit::GEISHA_4)
        .value("GEISHA_5", Suit::GEISHA_5)
        .value("GEISHA_6", Suit::GEISHA_6)
        .value("GEISHA_7", Suit::GEISHA_7)
        .export_values();

    // Replace pybind11's strict same-type __eq__/__ne__ so a C++ Suit also
    // equals the registered variant Suit member with the same value (the
    // Python side is patched symmetrically by engine.register_suit_enum).
    // __hash__ stays hash(value), matching the patched Python Suit.
    suit_cls.attr("__eq__") = py::cpp_function(
        [](py::handle a, py::handle b) -> py::object {
            Suit sb;
            if (!try_suit_in(b, &sb)) return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            return py::bool_(a.cast<Suit>() == sb);
        }, py::is_method(suit_cls), py::name("__eq__"));
    suit_cls.attr("__ne__") = py::cpp_function(
        [](py::handle a, py::handle b) -> py::object {
            Suit sb;
            if (!try_suit_in(b, &sb)) return py::reinterpret_borrow<py::object>(Py_NotImplemented);
            return py::bool_(a.cast<Suit>() != sb);
        }, py::is_method(suit_cls), py::name("__ne__"));

    m.def("_register_suit_enum", &register_suit_enum, py::arg("suit_enum"),
          "Register a variant's Python Suit enum (keyed by its member count).");

    // -----------------------------------------------------------------
    // ActionType enum
    // .value is an int (0-11); .name matches Python's ActionType.name.
    // Python's ActionType.value is a string ("reserve" etc.) — Phase 10
    // aligns .value via the value_str property added below.
    // -----------------------------------------------------------------
    auto at_cls = py::enum_<ActionType>(m, "ActionType")
        .value("RESERVE",        ActionType::RESERVE)
        .value("DISCARD",        ActionType::DISCARD)
        .value("SPLIT",          ActionType::SPLIT)
        .value("SPLIT_RESPONSE", ActionType::SPLIT_RESPONSE)
        .value("CHANCE_START",   ActionType::CHANCE_START)
        .value("CHANCE_DEAL",    ActionType::CHANCE_DEAL)
        .value("CHANCE_DRAW",    ActionType::CHANCE_DRAW)
        .value("TRADE_OFF",            ActionType::TRADE_OFF)
        .value("GIFT",                 ActionType::GIFT)
        .value("GIFT_RESPONSE",        ActionType::GIFT_RESPONSE)
        .value("COMPETITION",          ActionType::COMPETITION)
        .value("COMPETITION_RESPONSE", ActionType::COMPETITION_RESPONSE)
        .export_values();

    // Add .value_str property for Python-compatible artifact serialization.
    // This does NOT override the built-in .value (which remains an int)
    // but provides `at.value_str == "reserve"` so Phase 12 (policy table)
    // can serialize/deserialize correctly.
    at_cls.def_property_readonly("value_str", [](ActionType t) -> std::string {
        return action_type_value_str(t);
    });

    // -----------------------------------------------------------------
    // RuntimeConfig — opaque game-variant descriptor.
    // Python callers use the static factory methods (tiny_hana / medium_hana)
    // to obtain a config and pass it to GameState / Game constructors.
    // -----------------------------------------------------------------
    py::class_<RuntimeConfig>(m, "RuntimeConfig")
        .def_static("tiny_hana",   []() { return RuntimeConfig::tiny_hana(); })
        .def_static("medium_hana", []() { return RuntimeConfig::medium_hana(); })
        .def_static("full_hana",   []() { return RuntimeConfig::full_hana(); })
        .def_readonly("enumerable",        &RuntimeConfig::enumerable)
        .def_readonly("num_suits",         &RuntimeConfig::num_suits)
        .def_readonly("initial_hand_size", &RuntimeConfig::initial_hand_size)
        .def_readonly("turns_per_round",   &RuntimeConfig::turns_per_round)
        .def_readonly("win_geisha_count",  &RuntimeConfig::win_geisha_count)
        .def_readonly("has_win_points",    &RuntimeConfig::has_win_points)
        .def_readonly("win_favor_points",  &RuntimeConfig::win_favor_points);

    // -----------------------------------------------------------------
    // Card
    // Duck-type compatible with Python's Card dataclass.
    // -----------------------------------------------------------------
    py::class_<CppCard>(m, "Card")
        .def(py::init([](py::handle suit) {
                 int8_t n = 0;
                 const Suit s = suit_in(suit, &n);
                 return CppCard{s, n};
             }), py::arg("suit"))
        .def_property_readonly("suit", [](const CppCard& c) { return suit_out(c.suit, c.num_suits); })
        .def("__eq__",   &CppCard::operator==)
        .def("__ne__",   &CppCard::operator!=)
        .def("__hash__", [](const CppCard& c) {
            return static_cast<py::ssize_t>(static_cast<int>(c.suit));
        })
        .def("__repr__", &CppCard::repr);

    // -----------------------------------------------------------------
    // Action
    // -----------------------------------------------------------------
    using ActionClass = py::class_<Action>;
    ActionClass(m, "Action", py::dynamic_attr())
        // --- Python-style constructor: Action(action_type, cards, choice=None) ---
        // Mirrors engine.models.Action.  Written as a new-style constructor by
        // hand (what py::init expands to) so the new instance can be tagged.
        .def("__init__",
            [](py::detail::value_and_holder& v_h, py::handle action_type,
               py::iterable cards, py::handle choice) {
                int8_t n = 0;
                Action a = action_from_parts(action_type, cards, choice, &n);
                py::detail::initimpl::construct<ActionClass>(v_h, std::move(a), false);
                if (n > 0) set_action_tag(py::handle(reinterpret_cast<PyObject*>(v_h.inst)), n);
            },
            py::detail::is_new_style_constructor(),
            py::arg("action_type"), py::arg("cards") = py::tuple(), py::arg("choice") = py::none())
        // --- Factory methods (accept cpp.Suit or registered variant Suit members) ---
        .def_static("make_reserve",
            [](py::handle suit) {
                int8_t n = 0;
                const Suit s = suit_in(suit, &n);
                return action_out(Action::make_reserve(s), n);
            },
            py::arg("suit"))
        .def_static("make_discard",
            [](py::handle suit_a, py::handle suit_b, int choice) {
                int8_t n = 0;
                const Suit a = suit_in(suit_a, &n);
                const Suit b = suit_in(suit_b);
                return action_out(Action::make_discard(a, b, choice), n);
            },
            py::arg("suit_a"), py::arg("suit_b"), py::arg("choice"))
        .def_static("make_split",
            [](py::handle suit_a, py::handle suit_b) {
                int8_t n = 0;
                const Suit a = suit_in(suit_a, &n);
                const Suit b = suit_in(suit_b);
                return action_out(Action::make_split(a, b), n);
            },
            py::arg("suit_a"), py::arg("suit_b"))
        .def_static("make_split_response",
            &Action::make_split_response,
            py::arg("choice"))
        // --- Full Hanamikoji ---
        .def_static("make_trade_off",
            [](py::handle suit_a, py::handle suit_b) {
                int8_t n = 0;
                const Suit a = suit_in(suit_a, &n);
                return action_out(Action::make_trade_off(a, suit_in(suit_b)), n);
            },
            py::arg("suit_a"), py::arg("suit_b"))
        .def_static("make_gift",
            [](py::handle suit_a, py::handle suit_b, py::handle suit_c) {
                int8_t n = 0;
                const Suit a = suit_in(suit_a, &n);
                return action_out(Action::make_gift(a, suit_in(suit_b), suit_in(suit_c)), n);
            },
            py::arg("suit_a"), py::arg("suit_b"), py::arg("suit_c"))
        .def_static("make_competition",
            [](py::handle a0, py::handle a1, py::handle b0, py::handle b1) {
                int8_t n = 0;
                const Suit s0 = suit_in(a0, &n);
                return action_out(Action::make_competition(s0, suit_in(a1), suit_in(b0), suit_in(b1)), n);
            },
            py::arg("pair0_a"), py::arg("pair0_b"), py::arg("pair1_a"), py::arg("pair1_b"))
        .def_static("make_gift_response",
            [](int choice) {
                if (choice < 0 || choice > 2) throw py::value_error("gift response choice must be 0, 1 or 2");
                return Action::make_gift_response(choice);
            },
            py::arg("choice"))
        .def_static("make_competition_response",
            [](int choice) {
                if (choice < 0 || choice > 1) throw py::value_error("competition response choice must be 0 or 1");
                return Action::make_competition_response(choice);
            },
            py::arg("choice"))
        .def_static("make_chance_start",
            &Action::make_chance_start,
            py::arg("starting_player"))
        .def_static("make_chance_draw",
            [](py::handle suit, int player) {
                int8_t n = 0;
                const Suit s = suit_in(suit, &n);
                return action_out(Action::make_chance_draw(s, player), n);
            },
            py::arg("suit"), py::arg("player"))
        .def_static("make_chance_deal",
            [](py::iterable suits) {
                std::vector<Suit> v;
                int8_t n = 0;
                for (py::handle s : suits) {
                    int8_t sn = 0;
                    v.push_back(suit_in(s, &sn));
                    if (n == 0) n = sn;
                }
                if (v.size() > static_cast<std::size_t>(MAX_ACTION_SUITS))
                    throw py::value_error("CHANCE_DEAL takes at most " + std::to_string(MAX_ACTION_SUITS) + " cards");
                return action_out(Action::make_chance_deal(v.data(), static_cast<int>(v.size())), n);
            },
            py::arg("suits"))
        // --- Properties matching Python's Action dataclass ---
        .def_property_readonly("action_type",
            [](const Action& a) { return a.action_type; })
        .def_property_readonly("cards",
            [](py::handle self) {
                const Action& a = self.cast<const Action&>();
                return py::tuple(py::cast(cards_out(a.suits, a.suit_count, action_num_suits(self))));
            })
        .def_property_readonly("choice",
            [](const Action& a) -> py::object {
                if (a.choice == -1) return py::none();
                return py::int_(static_cast<int>(a.choice));
            })
        // --- Python protocols ---
        .def("__eq__",   &Action::operator==)
        .def("__ne__",   &Action::operator!=)
        .def("__hash__", [](const Action& a) {
            // Cast to ssize_t: pybind11 requires signed hash.
            return static_cast<py::ssize_t>(std::hash<Action>{}(a));
        })
        .def("__repr__", &Action::repr);

    // -----------------------------------------------------------------
    // PlayerState — per-player snapshot exposed with Python-compatible
    // attribute names.  Python code uses `ActionType.X in ps.used_actions`
    // so used_actions returns a list (Python `in` works on sequences).
    // -----------------------------------------------------------------
    py::class_<PlayerView>(m, "PlayerState")
        .def_property_readonly("hand", [](const PlayerView& p) {
            return cards_out(p.ps.hand, p.ps.hand_count, p.num_suits);
        })
        .def_property_readonly("reserved_cards", [](const PlayerView& p) {
            return cards_out(p.ps.reserved, p.ps.reserved_count, p.num_suits);
        })
        .def_property_readonly("collected_cards", [](const PlayerView& p) {
            return cards_out(p.ps.collected, p.ps.collected_count, p.num_suits);
        })
        .def_property_readonly("discarded_revealed", [](const PlayerView& p) {
            return cards_out(p.ps.discarded_rev, p.ps.discarded_rev_count, p.num_suits);
        })
        .def_property_readonly("discarded_hidden", [](const PlayerView& p) {
            return cards_out(p.ps.discarded_hid, p.ps.discarded_hid_count, p.num_suits);
        })
        .def_property_readonly("used_actions", [](const PlayerView& p) {
            std::vector<ActionType> used;
            for (int at = 0; at <= static_cast<int>(ActionType::COMPETITION_RESPONSE); at++) {
                if (p.ps.used_actions & (1u << at))
                    used.push_back(static_cast<ActionType>(at));
            }
            return used;
        });

    // -----------------------------------------------------------------
    // RoundInfoState
    // All query methods mirror Python's RoundInfoState.
    // Callback-based C++ generators are wrapped to return Python lists.
    // -----------------------------------------------------------------
    py::class_<RoundInfoState>(m, "RoundInfoState")
        .def_readonly("player_id",    &RoundInfoState::player_id)
        .def_readonly("round_number", &RoundInfoState::round_number)
        // Core
        .def("__str__",  &RoundInfoState::to_string)
        // Hash of the canonical string that == compares, so info states can be
        // dict keys and set members, as in Python mode.
        .def("__hash__", &RoundInfoState::hash)
        .def("compact_key", [](const RoundInfoState& r) { return py::bytes(r.compact_key()); },
             "The same information as str(), as a few bytes (see engine/compact_key.py).")
        .def("__repr__", [](const RoundInfoState& r) {
            return "RoundInfoState(player=" + std::to_string(r.player_id) +
                   ", round=" + std::to_string(r.round_number) +
                   ", events=" + std::to_string(r.event_count) + ")";
        })
        .def("__eq__", &RoundInfoState::operator==)
        .def("__ne__", &RoundInfoState::operator!=)
        // Legal actions — returns list so Python can iterate multiple times.
        .def("get_legal_actions", [](const RoundInfoState& r) {
            std::vector<Action> v;
            r.get_legal_actions([&](const Action& a) { v.push_back(a); });
            return actions_out(v, r.num_suits);
        })
        // Hand / used actions
        .def("get_current_hand_and_actions", [](const RoundInfoState& r) {
            const HandAndActions ha = r.get_current_hand_and_actions();
            const std::vector<CppCard> hand = cards_out(ha.hand, ha.hand_count, r.num_suits);
            std::vector<ActionType> used;
            for (int at = 0; at <= static_cast<int>(ActionType::COMPETITION_RESPONSE); at++) {
                if (ha.used_actions & (1u << at))
                    used.push_back(static_cast<ActionType>(at));
            }
            return py::make_tuple(hand, used);
        })
        // Split-collected cards
        .def("get_collected_cards", [](const RoundInfoState& r) {
            const CollectedCards cc = r.get_collected_cards();
            return py::make_tuple(cards_out(cc.my_cards,  cc.my_count,  r.num_suits),
                                  cards_out(cc.opp_cards, cc.opp_count, r.num_suits));
        })
        // Favors — {Suit: int8_t}; loops to r.num_suits (3 for tiny, 5 for medium).
        .def("get_favors", [](const RoundInfoState& r) {
            int8_t favors[SUIT_CAPACITY];
            r.get_favors(favors);
            py::dict d;
            for (int i = 0; i < r.num_suits; i++)
                d[suit_out(static_cast<Suit>(i), r.num_suits)] = py::int_(static_cast<int>(favors[i]));
            return d;
        })
        // Reserved cards (my own visible reserved)
        .def("get_my_reserved", [](const RoundInfoState& r) {
            Suit out[2];
            const int n = r.get_my_reserved(out);
            return cards_out(out, n, r.num_suits);
        })
        .def("get_opponent_reserved_count", &RoundInfoState::get_opponent_reserved_count)
        // Discarded cards
        .def("get_discarded_cards", [](const RoundInfoState& r) {
            const DiscardedCards dc = r.get_discarded_cards();
            return py::make_tuple(cards_out(dc.my_revealed,  dc.my_revealed_count,  r.num_suits),
                                  cards_out(dc.my_hidden,    dc.my_hidden_count,    r.num_suits),
                                  cards_out(dc.opp_revealed, dc.opp_revealed_count, r.num_suits));
        })
        // Opponent used actions bitmask → list
        .def("get_opponent_used_actions", [](const RoundInfoState& r) {
            const ActionMask bitmask = r.get_opponent_used_actions();
            std::vector<ActionType> used;
            for (int at = 0; at <= static_cast<int>(ActionType::COMPETITION_RESPONSE); at++) {
                if (bitmask & (1u << at)) used.push_back(static_cast<ActionType>(at));
            }
            return used;
        })
        .def("get_current_turn", &RoundInfoState::get_current_turn)
        // Pending split → None or list[CppCard]
        .def("get_pending_split", [](const RoundInfoState& r) -> py::object {
            const PendingSplit ps = r.get_pending_split();
            if (!ps.has_pending) return py::none();
            return py::cast(cards_out(ps.cards, 2, r.num_suits));
        })
        // Pending offer of any type → None or (ActionType, list[CppCard])
        .def("get_pending_offer", [](const RoundInfoState& r) -> py::object {
            const PendingOffer po = r.get_pending_offer();
            if (!po.has_pending) return py::none();
            return py::make_tuple(po.offer_type, cards_out(po.cards, po.count, r.num_suits));
        })
        // Available action types bitmask → list
        .def("get_available_action_types", [](const RoundInfoState& r) {
            const ActionMask bitmask = r.get_available_action_types();
            std::vector<ActionType> avail;
            for (int at = 0; at <= static_cast<int>(ActionType::COMPETITION_RESPONSE); at++) {
                if (bitmask & (1u << at)) avail.push_back(static_cast<ActionType>(at));
            }
            return avail;
        });

    // -----------------------------------------------------------------
    // RoundResolution — read-only snapshot returned by preview_round_resolution.
    // Mirrors Python's RoundResolutionPreview.
    // -----------------------------------------------------------------
    py::class_<RoundResolution>(m, "RoundResolution")
        .def_property_readonly("p0_reserved_revealed", [](const RoundResolution& r) {
            return cards_out(r.p0_reserved, r.p0_reserved_count, r.num_suits);
        })
        .def_property_readonly("p1_reserved_revealed", [](const RoundResolution& r) {
            return cards_out(r.p1_reserved, r.p1_reserved_count, r.num_suits);
        })
        // Loops to r.num_suits so medium_hana (5 suits) prints all 5 favors.
        .def_property_readonly("final_favors", [](const RoundResolution& r) {
            py::dict d;
            for (int i = 0; i < r.num_suits; i++)
                d[suit_out(static_cast<Suit>(i), r.num_suits)] = py::int_(static_cast<int>(r.final_favors[i]));
            return d;
        })
        .def_property_readonly("winner", [](const RoundResolution& r) -> py::object {
            if (r.winner == static_cast<int8_t>(NO_PLAYER)) return py::none();
            return py::int_(static_cast<int>(r.winner));
        });

    // -----------------------------------------------------------------
    // GameState
    // -----------------------------------------------------------------
    // Deck — only its size, as Python's Deck.total() (the state's renderer and
    // the GUI show how many cards are left undealt).
    py::class_<Deck>(m, "Deck")
        .def("total", &Deck::total);

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def(py::init<const RuntimeConfig&>(), py::arg("config"))
        // Copy: the state is plain data (fixed-size arrays, no pointers), so the
        // C++ copy constructor copies everything, undo history included.
        // engine.GameState's __copy__/__deepcopy__ use it.
        .def(py::init<const GameState&>(), py::arg("other"))
        // Core execution
        .def("execute_action",    &GameState::execute_action,   py::arg("action"))
        .def("is_round_complete", &GameState::is_round_complete)
        .def("get_info_state",    &GameState::get_info_state,   py::arg("player_id"))
        .def("update_favors",     &GameState::update_favors)
        .def("begin_next_round",  &GameState::begin_next_round)
        .def("check_winner",      &GameState::check_winner)
        // phase — string matching Python's phase strings
        .def_property_readonly("phase", [](const GameState& gs) {
            return std::string(phase_name(gs.phase));
        })
        // current_player — None when round complete (mirrors Python property)
        .def_property_readonly("current_player", [](const GameState& gs) -> py::object {
            if (gs.phase == Phase::ROUND_COMPLETE) return py::none();
            return py::int_(gs.current_player());
        })
        // round_count
        .def_property_readonly("round_count", [](const GameState& gs) {
            return gs.round_count;
        })
        // turn_count
        .def_property_readonly("turn_count", [](const GameState& gs) {
            return gs.turn_count;
        })
        // winner — None if no winner, else 0 or 1 (mirrors Python)
        .def_property_readonly("winner", [](const GameState& gs) -> py::object {
            const int w = gs.check_winner();
            if (w == NO_PLAYER) return py::none();
            return py::int_(w);
        })
        // pending_split — None or (offerer_id, [CppCard, CppCard]); SPLIT offers only
        .def_property_readonly("pending_split", [](const GameState& gs) -> py::object {
            if (!gs.has_pending_offer() || gs.pending_offer_type_ != ActionType::SPLIT)
                return py::none();
            const int offerer = static_cast<int>(gs.pending_offer_player_);
            return py::make_tuple(py::int_(offerer),
                                  cards_out(gs.pending_offer_cards_, 2, gs.config_.num_suits));
        })
        // pending_offer — None or (offerer_id, ActionType, (CppCard, ...)); any offer type
        .def_property_readonly("pending_offer", [](const GameState& gs) -> py::object {
            if (!gs.has_pending_offer()) return py::none();
            return py::make_tuple(
                py::int_(static_cast<int>(gs.pending_offer_player_)), gs.pending_offer_type_,
                py::tuple(py::cast(cards_out(gs.pending_offer_cards_, gs.pending_offer_count_,
                                             gs.config_.num_suits))));
        })
        // deck — a copy of the undealt cards (read-only).
        .def_property_readonly("deck", [](const GameState& gs) { return gs.deck; })
        // favors array — {Suit: int}; loops to config_.num_suits.
        // Reading returns a new dict, so editing it changes nothing; assign a dict
        // instead (state.favors = {...}), e.g. to set a round's starting favors.
        // Suits missing from the assigned dict keep their favor.
        .def_property("favors",
            [](const GameState& gs) {
                py::dict d;
                for (int i = 0; i < gs.config_.num_suits; i++)
                    d[suit_out(static_cast<Suit>(i), gs.config_.num_suits)] = py::int_(static_cast<int>(gs.favors[i]));
                return d;
            },
            [](GameState& gs, const py::dict& favors) {
                int8_t values[SUIT_CAPACITY];
                for (int i = 0; i < gs.config_.num_suits; i++) values[i] = gs.favors[i];
                for (auto item : favors) {
                    const int s = static_cast<int>(suit_in(item.first));
                    if (s >= gs.config_.num_suits)
                        throw py::value_error("favors: suit not in this variant");
                    const int v = py::cast<int>(item.second);
                    if (v < -1 || v > 1)
                        throw py::value_error("favors: each value must be -1, 0 or 1");
                    values[s] = static_cast<int8_t>(v);
                }
                for (int i = 0; i < gs.config_.num_suits; i++) gs.favors[i] = values[i];
            })
        // players — list of two PlayerState snapshots (copies, not references).
        .def_property_readonly("players", [](const GameState& gs) {
            const auto n = static_cast<int8_t>(gs.config_.num_suits);
            return std::vector<PlayerView>{PlayerView{gs.players[0], n}, PlayerView{gs.players[1], n}};
        })
        // ---------------------------------------------------------
        // Solver API
        // ---------------------------------------------------------
        // undo_move — reverses the last execute_action call.
        .def("undo_move", &GameState::undo_move)
        // get_legal_actions — returns list; valid at any non-ROUND_COMPLETE node.
        .def("get_legal_actions", [](const GameState& gs) {
            std::vector<Action> v;
            gs.get_legal_actions([&](const Action& a) { v.push_back(a); });
            return actions_out(v, gs.config_.num_suits);
        })
        // get_chance_actions_with_probs — list of (Action, float) pairs.
        // Only valid at chance nodes.
        .def("get_chance_actions_with_probs", [](const GameState& gs) {
            py::list out;
            gs.get_chance_actions_with_probs([&](const Action& a, double p) {
                out.append(py::make_tuple(action_out(a, gs.config_.num_suits), p));
            });
            return out;
        })
        // sample_chance_action — draw one chance action using the supplied Rng.
        // Only valid at chance nodes.
        .def("sample_chance_action", [](const GameState& gs, CppRng& rng) {
            return action_out(gs.sample_chance_action(rng.rng), gs.config_.num_suits);
        }, py::arg("rng"))
        // terminal_payoff — payoff for player_id at a round-complete node.
        .def("terminal_payoff", &GameState::terminal_payoff, py::arg("player_id") = 0)
        // preview_round_resolution — compute end-of-round outcome without mutation.
        .def("preview_round_resolution", &GameState::preview_round_resolution)
        // to_key — value-based hashable snapshot for transposition tables.
        .def("to_key", &GameState::to_key);

    // -----------------------------------------------------------------
    // GameStateKey — value-based hashable snapshot from GameState::to_key().
    // Usable as a Python dict key for transposition tables.
    // -----------------------------------------------------------------
    py::class_<GameState::Key>(m, "GameStateKey")
        .def("__eq__", &GameState::Key::operator==)
        .def("__ne__", &GameState::Key::operator!=)
        .def("__hash__", [](const GameState::Key& k) {
            size_t h = 0;
            hash_combine(h, static_cast<size_t>(k.num_suits));
            hash_combine(h, static_cast<size_t>(k.phase));
            hash_combine(h, static_cast<size_t>(static_cast<uint8_t>(k.raw_current_player)));
            hash_combine(h, static_cast<size_t>(k.turn_count));
            hash_combine(h, static_cast<size_t>(k.round_count));
            hash_combine(h, static_cast<size_t>(static_cast<uint8_t>(k.round_starting_player)));
            hash_combine(h, static_cast<size_t>(static_cast<uint8_t>(k.pending_draw_player)));
            hash_combine(h, static_cast<size_t>(static_cast<uint8_t>(k.winner)));
            hash_combine(h, static_cast<size_t>(static_cast<uint8_t>(k.pending_offer_player)));
            for (int i = 0; i < k.num_suits; i++) {
                hash_combine(h, static_cast<size_t>(k.deck_counts[i]));
                hash_combine(h, static_cast<size_t>(static_cast<uint8_t>(k.favors[i])));
            }
            if (k.pending_offer_player != static_cast<int8_t>(NO_PLAYER)) {
                hash_combine(h, static_cast<size_t>(k.pending_offer_type));
                for (int i = 0; i < k.pending_offer_count; i++)
                    hash_combine(h, static_cast<size_t>(k.pending_offer_cards[i]));
            }
            auto mix_suits = [&h](const Suit* suits, int8_t count) {
                hash_combine(h, static_cast<size_t>(count));
                for (int i = 0; i < count; i++)
                    hash_combine(h, static_cast<size_t>(suits[i]));
            };
            mix_suits(k.p0_hand,          k.p0_hand_count);
            mix_suits(k.p0_reserved,      k.p0_reserved_count);
            mix_suits(k.p0_collected,     k.p0_collected_count);
            mix_suits(k.p0_discarded_rev, k.p0_discarded_rev_count);
            mix_suits(k.p0_discarded_hid, k.p0_discarded_hid_count);
            hash_combine(h, static_cast<size_t>(k.p0_used_actions));
            mix_suits(k.p1_hand,          k.p1_hand_count);
            mix_suits(k.p1_reserved,      k.p1_reserved_count);
            mix_suits(k.p1_collected,     k.p1_collected_count);
            mix_suits(k.p1_discarded_rev, k.p1_discarded_rev_count);
            mix_suits(k.p1_discarded_hid, k.p1_discarded_hid_count);
            hash_combine(h, static_cast<size_t>(k.p1_used_actions));
            return static_cast<py::ssize_t>(h);
        });

    // -----------------------------------------------------------------
    // Rng — seedable mt19937 wrapper for sample_chance_action.
    // -----------------------------------------------------------------
    py::class_<CppRng>(m, "Rng")
        .def(py::init<>(), "Initialise with a random seed from std::random_device.")
        .def(py::init<uint64_t>(), py::arg("seed"),
             "Initialise with an explicit integer seed for reproducibility.")
        .def("seed", &CppRng::seed, py::arg("seed"), "Re-seed the generator.");

    // -----------------------------------------------------------------
    // Agent ABC (with PyAgent trampoline)
    // Python subclasses override get_action_distribution(info_state).
    // -----------------------------------------------------------------
    py::class_<Agent, PyAgent>(m, "Agent")
        .def(py::init<>())
        .def("get_action_distribution",
             &Agent::get_action_distribution,
             py::arg("info_state"));

    // -----------------------------------------------------------------
    // RandomAgent
    // -----------------------------------------------------------------
    py::class_<RandomAgent, Agent>(m, "RandomAgent")
        .def(py::init<>())
        .def("get_action_distribution",
             &RandomAgent::get_action_distribution,
             py::arg("info_state"));

    // -----------------------------------------------------------------
    // GameRunner exposed as Game (mirrors Python's engine.game.Game).
    //
    // py::keep_alive<0, N> keeps constructor argument N alive as long as
    // the constructed object (index 0) is alive — prevents Python GC from
    // destroying an agent while the runner still holds a raw pointer to it.
    // -----------------------------------------------------------------
    py::class_<GameRunner>(m, "Game")
        .def(py::init<Agent&, Agent&, bool, uint64_t, const RuntimeConfig&>(),
             py::arg("agent0"),
             py::arg("agent1"),
             py::arg("verbose") = false,
             py::arg("seed")    = 0,
             py::arg("cfg")     = RuntimeConfig::tiny_hana(),
             py::keep_alive<0, 1>(),   // keep agent0 alive with Game
             py::keep_alive<0, 2>())   // keep agent1 alive with Game
        .def("play_game",     &GameRunner::play_game,     py::arg("max_rounds") = 10)
        .def("play_round",    &GameRunner::play_round)
        .def("play_decision", &GameRunner::play_decision)
        .def_property_readonly("state",       &GameRunner::state,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("game_winner", &GameRunner::game_winner);

    // -----------------------------------------------------------------
    // BinaryPolicyTableWriter (hana/policy_table.h): writes the binary
    // policy tables that engine.PolicyTableAgent.from_artifact reads.
    // Exposed so the C++ writer can be tested against the Python reader.
    // -----------------------------------------------------------------
    py::class_<BinaryPolicyTableWriter>(m, "BinaryPolicyTableWriter")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("path"), py::arg("key_encoding"), py::arg("variant"))
        .def("add", [](BinaryPolicyTableWriter& w, const std::string& key,
                       const PolicyDistribution& dist) { w.add(key, dist); },
             py::arg("key"), py::arg("distribution"),
             "key: str or bytes; distribution: list of (Action, probability).")
        .def("close", &BinaryPolicyTableWriter::close)
        .def("__len__", &BinaryPolicyTableWriter::size);
}
