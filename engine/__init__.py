"""Engine package exports.

Set the environment variable HANA_ENGINE=cpp before importing this package
to activate the C++ backend.  All public names listed in __all__ remain
available regardless of which backend is active.
"""

import os

_CPP_BACKEND = os.environ.get("HANA_ENGINE") == "cpp"

if _CPP_BACKEND:
    import hana_engine_cpp as _cpp

    # ----------------------------------------------------------------
    # Constants
    # ----------------------------------------------------------------
    CHANCE_PLAYER = _cpp.CHANCE_PLAYER
    NO_PLAYER     = _cpp.NO_PLAYER
    Rng           = _cpp.Rng

    # ----------------------------------------------------------------
    # Core types — replaced by C++ equivalents
    # ----------------------------------------------------------------
    ActionType     = _cpp.ActionType
    Action         = _cpp.Action
    Card           = _cpp.Card
    RoundInfoState = _cpp.RoundInfoState
    RandomAgent    = _cpp.RandomAgent

    # ----------------------------------------------------------------
    # Agent — Python subclass of the C++ Agent that behaves like the
    # Python-mode ABC:
    #   * its metaclass combines pybind11's with ABCMeta, so subclasses may
    #     mix in ABC and use @abstractmethod;
    #   * the C++ base is initialised automatically, so a subclass __init__
    #     need not call super().__init__() (pybind11 otherwise raises).
    # The C++ RandomAgent is registered as a virtual subclass.
    # ----------------------------------------------------------------
    from abc import ABCMeta as _ABCMeta, abstractmethod as _abstractmethod

    class _AgentMeta(type(_cpp.Agent), _ABCMeta):
        def __call__(cls, *args, **kwargs):
            if cls.__abstractmethods__:
                raise TypeError(
                    f"Can't instantiate abstract class {cls.__name__} with abstract "
                    f"method(s) {', '.join(sorted(cls.__abstractmethods__))}")
            # type.__call__ skips pybind11's "base __init__ must be called" check;
            # Agent.__init__ below then initialises the C++ base if still needed.
            obj = type.__call__(cls, *args, **kwargs)
            Agent.__init__(obj)
            return obj

    class Agent(_cpp.Agent, metaclass=_AgentMeta):
        """Base class for AI agents using mixed strategies (C++ backend)."""

        def __init__(self):
            if not self.__dict__.get("_cpp_agent_initialised", False):
                _cpp.Agent.__init__(self)
                self.__dict__["_cpp_agent_initialised"] = True

        @_abstractmethod
        def get_action_distribution(self, info_state):
            """Return {legal Action: probability}; probabilities sum to 1."""

    Agent.register(_cpp.RandomAgent)

    # ----------------------------------------------------------------
    # Variant Suit enums
    #
    # The C++ Suit is shared by all variants; each variant's Python Suit
    # carries variant-specific ranks.  Registering a variant's Suit makes
    # the C++ objects hand back its members (so .rank and dict keys work),
    # and makes Python Suit and cpp.Suit compare/hash equal by value —
    # cpp.Suit's side of that lives in bindings.cpp.
    # ----------------------------------------------------------------
    def _suit_eq(self, other):
        if self is other:
            return True
        if isinstance(other, _cpp.Suit):
            return self._value_ == other.value
        return NotImplemented

    def _suit_hash(self):
        return hash(self._value_)   # == hash(cpp.Suit) for the same value

    def register_suit_enum(suit_enum):
        """Register a variant's Suit enum with the C++ backend; returns it."""
        suit_enum.__eq__ = _suit_eq
        suit_enum.__hash__ = _suit_hash
        _cpp._register_suit_enum(suit_enum)
        return suit_enum

    def _py_config_to_cpp_config(config):
        """Map a Python GameConfig to the matching C++ RuntimeConfig.

        Detection is by suit count: 3 suits → tiny_hana, 5 → medium_hana,
        7 → full_hana.  Raises ValueError for unrecognised variants.
        """
        if config is None:
            return _cpp.RuntimeConfig.tiny_hana()
        n = len(list(config.suits))
        if n == 3:
            return _cpp.RuntimeConfig.tiny_hana()
        if n == 5:
            return _cpp.RuntimeConfig.medium_hana()
        if n == 7:
            return _cpp.RuntimeConfig.full_hana()
        raise ValueError(
            f"No C++ RuntimeConfig for a {n}-suit variant. Only tiny_hana (3 suits), "
            "medium_hana (5 suits) and full_hana (7 suits) are supported."
        )

    # ----------------------------------------------------------------
    # GameState — thin wrapper that maps Python GameConfig → RuntimeConfig.
    # ----------------------------------------------------------------
    class GameState(_cpp.GameState):
        def __init__(self, config=None):
            super().__init__(_py_config_to_cpp_config(config))
            self.config = config

        # copy.copy / copy.deepcopy, as in Python mode: the C++ copy constructor
        # copies the whole state (it holds no references), keeping the subclass.
        def __copy__(self):
            new = type(self).__new__(type(self))
            _cpp.GameState.__init__(new, self)
            new.config = self.config
            return new

        def __deepcopy__(self, memo):
            return self.__copy__()

        def pretty_print(self, box_width: int = 58) -> str:
            from .render import pretty_print_game_state
            return pretty_print_game_state(self, box_width)

    # ----------------------------------------------------------------
    # Game — matches Python's Game(agent0, agent1, config, verbose, seed)
    # positional signature used by tiny_hana.Game.__init__.
    # ----------------------------------------------------------------
    class Game(_cpp.Game):
        def __init__(self, agent0, agent1, config=None,
                     verbose=False, seed=None):
            import random as _random
            seed_val = seed if seed is not None else _random.getrandbits(64)
            super().__init__(agent0, agent1, verbose, int(seed_val),
                             _py_config_to_cpp_config(config))
            self.config  = config
            self.agents  = [agent0, agent1]
            self.verbose = verbose

    # ----------------------------------------------------------------
    # Python-only utilities — still imported from their source modules.
    # These are available at import time; runtime behaviour when combined
    # with C++ game objects is deferred to Phase 12 (policy artifacts).
    # ----------------------------------------------------------------
    from .constants import PLAYER_1, PLAYER_2
    from .enums import ObservationType
    from .events import (
        _freeze_favors,
        ObservationEvent,
        RoundStartEvent,
        DrawCardEvent,
        OwnReserveEvent,
        OpponentReserveEvent,
        OwnDiscardEvent,
        OpponentDiscardEvent,
        OwnSplitEvent,
        OpponentSplitEvent,
        SplitChoiceEvent,
        RoundEndEvent,
        OwnTradeOffEvent,
        OpponentTradeOffEvent,
        OwnGiftEvent,
        OpponentGiftEvent,
        GiftChoiceEvent,
        OwnCompetitionEvent,
        OpponentCompetitionEvent,
        CompetitionChoiceEvent,
    )
    from .models import GameConfig, require_enumerable
    from .state import PlayerState

    # ----------------------------------------------------------------
    # Policy-table system — C++-aware overrides.
    #
    # The on-disk artifact format is backend-agnostic (action types
    # stored as strings, suits as integers).  What changes per backend
    # is the decode step: Python mode returns Python Action objects;
    # C++ mode returns cpp.Action objects so GameRunner can use them.
    # ----------------------------------------------------------------
    from .policy_table import (
        POLICY_TABLE_FORMAT,
        POLICY_TABLE_KEY_ENCODING,
        MARKOV_POLICY_TABLE_KEY_ENCODING,
        PolicyTableAgent as _BasePolicyTableAgent,
        _MarkovPolicyKeys,
        markov_policy_key,
        _load_policy_table_artifact_internal,
        dump_policy_table_artifact,          # encoding handles cpp.Action after fix
    )

    def _make_cpp_action(action_type_py, suits, choice):
        """Reconstruct a cpp.Action from internal (py ActionType, suits, choice)."""
        name = action_type_py.name
        cs   = [_cpp.Suit(s.value) for s in suits]
        if name == "RESERVE":
            return _cpp.Action.make_reserve(cs[0])
        if name == "DISCARD":
            return _cpp.Action.make_discard(cs[0], cs[1], choice)
        if name == "SPLIT":
            return _cpp.Action.make_split(cs[0], cs[1])
        if name == "SPLIT_RESPONSE":
            return _cpp.Action.make_split_response(choice)
        if name == "TRADE_OFF":
            return _cpp.Action.make_trade_off(cs[0], cs[1])
        if name == "GIFT":
            return _cpp.Action.make_gift(cs[0], cs[1], cs[2])
        if name == "GIFT_RESPONSE":
            return _cpp.Action.make_gift_response(choice)
        if name == "COMPETITION":
            return _cpp.Action.make_competition(cs[0], cs[1], cs[2], cs[3])
        if name == "COMPETITION_RESPONSE":
            return _cpp.Action.make_competition_response(choice)
        raise ValueError(
            f"Action type '{name}' cannot be decoded to cpp.Action."
        )

    def _decode_dist_cpp(encoded_dist):
        """Decode an internal distribution tuple to {cpp.Action: float}."""
        return {
            _make_cpp_action(at, suits, choice): prob
            for (at, suits, choice), prob in encoded_dist
        }

    class PolicyTableAgent(_BasePolicyTableAgent, Agent):
        """PolicyTableAgent that returns cpp.Action objects.

        Inherits all encoding, artifact save/load, and from_artifact logic
        from the Python base.  Only the decode step is overridden.  Also a
        C++ Agent, so it can be passed to the C++ Game.
        """

        @classmethod
        def from_artifact(cls, *args, **kwargs):
            # The base builds the agent with cls.__new__ (no __init__), which
            # would leave the C++ Agent base uninitialised.
            agent = super().from_artifact(*args, **kwargs)
            Agent.__init__(agent)
            return agent

        def get_action_distribution(self, info_state):
            key = self._lookup_key(info_state)
            if key not in self._table:
                return self._missing_key(info_state, key)
            return _decode_dist_cpp(self._table[key])

    def load_policy_table_artifact(path, suits_enum, *, expected_variant=None):
        """Load a JSON policy-table artifact, returning cpp.Action objects."""
        encoded_table = _load_policy_table_artifact_internal(
            path, suits_enum, expected_variant=expected_variant)
        return {key: _decode_dist_cpp(dist) for key, dist in encoded_table.items()}

    class MarkovPolicyTableAgent(_MarkovPolicyKeys, PolicyTableAgent):
        """Round-independent policy table for Markov (stationary) strategies.

        C++-backend counterpart of engine.policy_table.MarkovPolicyTableAgent,
        whose docstring explains why the round number is not part of the key.
        """

    from .policy_table import _CompactPolicyKeys

    class CompactPolicyTableAgent(_CompactPolicyKeys, PolicyTableAgent):
        """Policy table keyed by compact keys.

        C++-backend counterpart of engine.policy_table.CompactPolicyTableAgent.
        """

else:
    from .constants import CHANCE_PLAYER, PLAYER_1, PLAYER_2
    from .enums import ActionType, ObservationType
    from .models import GameConfig, Card, Action, require_enumerable
    from .events import (
        _freeze_favors,
        ObservationEvent,
        RoundStartEvent,
        DrawCardEvent,
        OwnReserveEvent,
        OpponentReserveEvent,
        OwnDiscardEvent,
        OpponentDiscardEvent,
        OwnSplitEvent,
        OpponentSplitEvent,
        SplitChoiceEvent,
        RoundEndEvent,
        OwnTradeOffEvent,
        OpponentTradeOffEvent,
        OwnGiftEvent,
        OpponentGiftEvent,
        GiftChoiceEvent,
        OwnCompetitionEvent,
        OpponentCompetitionEvent,
        CompetitionChoiceEvent,
    )
    from .info_state import RoundInfoState
    from .state import PlayerState, GameState
    from .agents import Agent, RandomAgent
    from .game import Game
    from .policy_table import (
        POLICY_TABLE_FORMAT,
        POLICY_TABLE_KEY_ENCODING,
        PolicyTableAgent,
        dump_policy_table_artifact,
        load_policy_table_artifact,
        MARKOV_POLICY_TABLE_KEY_ENCODING,
        MarkovPolicyTableAgent,
        markov_policy_key,
        CompactPolicyTableAgent,
    )

    def register_suit_enum(suit_enum):
        """No-op in Python mode (see the C++ branch); returns suit_enum."""
        return suit_enum

# Compact information-state keys: pure Python, so the same for both backends.
from .compact_key import compact_key_to_info_state, compact_key_to_string, string_to_compact_key

# Binary policy tables: pure Python, so the same for both backends.
from .binary_policy_table import (
    BINARY_POLICY_TABLE_FORMAT,
    COMPACT_POLICY_TABLE_KEY_ENCODING,
    write_binary_policy_table,
)
from .policy_table import policy_table_key_encoding


def load_policy_table_agent(path, suits_enum, *, expected_variant=None, fallback=None):
    """Load a policy-table file as an agent, whatever kind it is.

    Reads JSON or binary files, and picks the agent class the file's key
    encoding needs: PolicyTableAgent (keys are str(info_state)),
    MarkovPolicyTableAgent (keys without the round number) or
    CompactPolicyTableAgent (compact keys). `fallback` is the agent that plays
    information states the table does not list; without one, they raise.
    """
    classes = {
        POLICY_TABLE_KEY_ENCODING: PolicyTableAgent,
        MARKOV_POLICY_TABLE_KEY_ENCODING: MarkovPolicyTableAgent,
        COMPACT_POLICY_TABLE_KEY_ENCODING: CompactPolicyTableAgent,
    }
    key_encoding = policy_table_key_encoding(path)
    if key_encoding not in classes:
        raise ValueError(f"{path} is not a policy table with a known key encoding ({key_encoding!r}).")
    return classes[key_encoding].from_artifact(path, suits_enum, expected_variant=expected_variant,
                                               fallback=fallback)


__all__ = [
    "CHANCE_PLAYER",
    "PLAYER_1",
    "PLAYER_2",
    "ActionType",
    "ObservationType",
    "GameConfig",
    "require_enumerable",
    "Card",
    "Action",
    "_freeze_favors",
    "ObservationEvent",
    "RoundStartEvent",
    "DrawCardEvent",
    "OwnReserveEvent",
    "OpponentReserveEvent",
    "OwnDiscardEvent",
    "OpponentDiscardEvent",
    "OwnSplitEvent",
    "OpponentSplitEvent",
    "SplitChoiceEvent",
    "RoundEndEvent",
    "OwnTradeOffEvent",
    "OpponentTradeOffEvent",
    "OwnGiftEvent",
    "OpponentGiftEvent",
    "GiftChoiceEvent",
    "OwnCompetitionEvent",
    "OpponentCompetitionEvent",
    "CompetitionChoiceEvent",
    "RoundInfoState",
    "PlayerState",
    "GameState",
    "Agent",
    "RandomAgent",
    "Game",
    "POLICY_TABLE_FORMAT",
    "POLICY_TABLE_KEY_ENCODING",
    "PolicyTableAgent",
    "dump_policy_table_artifact",
    "load_policy_table_artifact",
    "MARKOV_POLICY_TABLE_KEY_ENCODING",
    "MarkovPolicyTableAgent",
    "markov_policy_key",
    "register_suit_enum",
    "compact_key_to_string",
    "string_to_compact_key",
    "compact_key_to_info_state",
    "CompactPolicyTableAgent",
    "BINARY_POLICY_TABLE_FORMAT",
    "COMPACT_POLICY_TABLE_KEY_ENCODING",
    "write_binary_policy_table",
    "policy_table_key_encoding",
    "load_policy_table_agent",
]

if _CPP_BACKEND:
    __all__ += ["NO_PLAYER", "Rng"]
