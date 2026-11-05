"""Policy-table artifact helpers and an artifact-backed agent."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .agents import Agent
from .binary_policy_table import (
    COMPACT_POLICY_TABLE_KEY_ENCODING,
    BinaryPolicyTable,
    read_header as _read_binary_header,
    write_binary_policy_table,
)
from .compact_key import compact_key_to_string, string_to_compact_key
from .enums import ActionType
from .info_state import RoundInfoState
from .models import Action, Card


POLICY_TABLE_FORMAT = "policy_table_v1"
POLICY_TABLE_KEY_ENCODING = "round_info_state_str_v1"
# Round-independent keys for Markov (stationary) strategies; see MarkovPolicyTableAgent.
MARKOV_POLICY_TABLE_KEY_ENCODING = "markov_round_info_state_str_v1"

_EPS = 1e-6
# Player actions a policy table may hold: (number of cards, allowed choices),
# where None means choice must be None.
_POLICY_ACTION_SHAPES = {
    ActionType.RESERVE: (1, None),
    ActionType.DISCARD: (2, (0, 1)),
    ActionType.SPLIT: (2, None),
    ActionType.SPLIT_RESPONSE: (0, (0, 1)),
    ActionType.TRADE_OFF: (2, None),
    ActionType.GIFT: (3, None),
    ActionType.GIFT_RESPONSE: (0, (0, 1, 2)),
    ActionType.COMPETITION: (4, None),
    ActionType.COMPETITION_RESPONSE: (0, (0, 1)),
}
_POLICY_ACTION_TYPES = frozenset(_POLICY_ACTION_SHAPES)
_ACTION_TYPE_BY_VALUE = {action_type.value: action_type for action_type in ActionType}

_InternalEncodedAction = Tuple[ActionType, Tuple[Any, ...], Optional[int]]
_InternalEncodedDistribution = Tuple[Tuple[_InternalEncodedAction, float], ...]
_InternalEncodedPolicyTable = Dict[str, _InternalEncodedDistribution]

# Name-keyed lookup used to normalise cpp.ActionType → Python ActionType.
# cpp.ActionType has the same .name strings but .value is an int, not "reserve".
_ACTION_TYPE_BY_NAME = {at.name: at for at in ActionType}


def _coerce_policy_key(raw_key: Union[str, bytes, RoundInfoState]) -> str:
    if isinstance(raw_key, str):
        return raw_key
    if isinstance(raw_key, bytes):   # a compact key (engine/compact_key.py)
        return compact_key_to_string(raw_key)
    # Duck-type check: works for both Python RoundInfoState and C++ RoundInfoState.
    if hasattr(raw_key, "player_id") and hasattr(raw_key, "round_number"):
        return str(raw_key)
    raise TypeError(f"Policy key must be str or RoundInfoState, got {type(raw_key).__name__}.")


# str(RoundInfoState) starts with "p{player_id}r{round_number}", followed by
# "|<event>" for each event of the current round.
_ROUND_PREFIX = re.compile(r"^p(\d+)r\d+(?=\||$)")


def markov_policy_key(raw_key: Union[str, RoundInfoState]) -> str:
    """Round-independent policy key: str(RoundInfoState) without the round number.

        "p1r2|ROUND_START@t0(...)|DRAW_CARD@t0(card=2)"
            -> "p1|ROUND_START@t0(...)|DRAW_CARD@t0(card=2)"

    Only valid for Markov (stationary) strategies — see MarkovPolicyTableAgent.
    Nothing else is lost: the remaining events still record everything the
    player has seen this round, including the round's starting favors and
    starting player.
    """
    key = _coerce_policy_key(raw_key)
    match = _ROUND_PREFIX.match(key)
    if match is None:
        raise ValueError(f"Not a RoundInfoState key (expected 'p<id>r<round>...'): {key!r}")
    return f"p{match.group(1)}" + key[match.end():]


def _validate_policy_action_shape(
    action_type: ActionType,
    cards: Sequence[Any],
    choice: Optional[int],
    *,
    context: str,
) -> None:
    if action_type not in _POLICY_ACTION_TYPES:
        raise ValueError(f"{context}: unsupported action type '{action_type.value}' for policy tables.")

    n_cards, choices = _POLICY_ACTION_SHAPES[action_type]
    name = action_type.value
    if len(cards) != n_cards:
        if n_cards == 0:
            raise ValueError(f"{context}: {name} action must not include cards.")
        raise ValueError(f"{context}: {name} action must contain exactly {n_cards} card(s).")
    if choices is None:
        if choice is not None:
            raise ValueError(f"{context}: {name} action must not set choice.")
    elif choice not in choices:
        raise ValueError(f"{context}: {name} action choice must be one of {list(choices)}.")


def _encode_action(action: Any, *, context: str) -> _InternalEncodedAction:
    # Accept both Python Action and cpp.Action (duck-typed).
    if not (hasattr(action, "action_type") and hasattr(action, "cards")
            and hasattr(action, "choice")):
        raise TypeError(
            f"{context}: expected Action-like object, got {type(action).__name__}.")

    # Normalise action_type to Python ActionType so that .value is always a string
    # ("reserve", …) regardless of whether the source is Python or cpp.ActionType.
    at = action.action_type
    if not isinstance(at, ActionType):
        try:
            at = _ACTION_TYPE_BY_NAME[at.name]
        except KeyError:
            raise ValueError(f"{context}: unknown action type name '{at.name}'.")

    cards: List[Any] = []
    for idx, card in enumerate(action.cards):
        # Accept both Python Card and CppCard (duck-typed).
        if not hasattr(card, "suit"):
            raise TypeError(f"{context}: card {idx} is missing a '.suit' attribute.")
        cards.append(card.suit)

    _validate_policy_action_shape(at, cards, action.choice, context=context)
    return (at, tuple(cards), action.choice)


def _encode_distribution(raw_distribution: Mapping[Action, float], *, context: str) -> _InternalEncodedDistribution:
    if not isinstance(raw_distribution, Mapping):
        raise TypeError(f"{context}: distribution must be a mapping.")
    if len(raw_distribution) == 0:
        raise ValueError(f"{context}: distribution cannot be empty.")

    encoded: List[Tuple[_InternalEncodedAction, float]] = []
    total_prob = 0.0

    for idx, (action, prob) in enumerate(raw_distribution.items()):
        entry_context = f"{context} entry {idx}"
        encoded_action = _encode_action(action, context=entry_context)
        try:
            weight = float(prob)
        except (TypeError, ValueError):
            raise ValueError(f"{entry_context}: probability must be numeric.")
        if weight < 0:
            raise ValueError(f"{entry_context}: probability must be non-negative.")
        encoded.append((encoded_action, weight))
        total_prob += weight

    if total_prob <= 0:
        raise ValueError(f"{context}: distribution total probability must be positive.")
    if abs(total_prob - 1.0) > _EPS:
        raise ValueError(f"{context}: probabilities must sum to 1.0, got {total_prob:.6f}.")

    return tuple(encoded)


def _encode_strategy_map(
    strategy_map: Mapping[Union[str, RoundInfoState], Mapping[Action, float]],
    key_fn=_coerce_policy_key,
) -> _InternalEncodedPolicyTable:
    if not isinstance(strategy_map, Mapping):
        raise TypeError(f"strategy_map must be a mapping, got {type(strategy_map).__name__}.")

    encoded_table: _InternalEncodedPolicyTable = {}
    for raw_key, raw_distribution in strategy_map.items():
        key = key_fn(raw_key)
        if key in encoded_table:
            raise ValueError(f"Duplicate policy key after string coercion: {key}")
        encoded_table[key] = _encode_distribution(raw_distribution, context=f"state '{key}'")

    return encoded_table


def _decode_distribution(encoded_distribution: _InternalEncodedDistribution) -> Dict[Action, float]:
    return {
        Action(action_type, [Card(suit) for suit in cards], choice=choice): prob
        for (action_type, cards, choice), prob in encoded_distribution
    }


def _serialize_suit(suit: Any, *, context: str) -> Any:
    if not hasattr(suit, "value"):
        raise ValueError(f"{context}: suit '{suit}' is missing '.value' for artifact serialization.")
    return suit.value


def _build_artifact_payload(
    encoded_table: _InternalEncodedPolicyTable,
    *,
    variant: Optional[str],
    key_encoding: str = POLICY_TABLE_KEY_ENCODING,
) -> Dict[str, Any]:
    entries: Dict[str, List[Dict[str, Any]]] = {}
    for key, distribution in encoded_table.items():
        encoded_distribution: List[Dict[str, Any]] = []
        for idx, ((action_type, cards, choice), prob) in enumerate(distribution):
            context = f"state '{key}' entry {idx}"
            encoded_distribution.append(
                {
                    "action_type": action_type.value,
                    "cards": [_serialize_suit(suit, context=context) for suit in cards],
                    "choice": choice,
                    "prob": prob,
                }
            )
        entries[key] = encoded_distribution

    return {
        "format": POLICY_TABLE_FORMAT,
        "key_encoding": key_encoding,
        "variant": variant,
        "entries": entries,
    }


def dump_policy_table_artifact(
    path: str,
    strategy_map: Mapping[Union[str, RoundInfoState], Mapping[Action, float]],
    *,
    variant: Optional[str] = None,
) -> None:
    """Serialize a strategy table to JSON artifact format."""
    encoded_table = _encode_strategy_map(strategy_map)
    payload = _build_artifact_payload(encoded_table, variant=variant)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)


def _decode_distribution_from_payload(
    key: str,
    raw_distribution: Any,
    suit_lookup: Mapping[Any, Any],
) -> _InternalEncodedDistribution:
    if not isinstance(raw_distribution, list):
        raise ValueError(f"state '{key}': distribution must be a list.")
    if len(raw_distribution) == 0:
        raise ValueError(f"state '{key}': distribution cannot be empty.")

    encoded_distribution: List[Tuple[_InternalEncodedAction, float]] = []
    total_prob = 0.0

    for idx, raw_entry in enumerate(raw_distribution):
        context = f"state '{key}' entry {idx}"
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{context}: entry must be an object.")

        action_type_value = raw_entry.get("action_type")
        if action_type_value not in _ACTION_TYPE_BY_VALUE:
            raise ValueError(f"{context}: unknown action_type '{action_type_value}'.")
        action_type = _ACTION_TYPE_BY_VALUE[action_type_value]

        raw_cards = raw_entry.get("cards")
        if not isinstance(raw_cards, list):
            raise ValueError(f"{context}: cards must be a list.")

        cards: List[Any] = []
        for card_idx, raw_suit_value in enumerate(raw_cards):
            if raw_suit_value not in suit_lookup:
                raise ValueError(f"{context}: unknown suit value at cards[{card_idx}] -> {raw_suit_value}.")
            cards.append(suit_lookup[raw_suit_value])

        choice = raw_entry.get("choice")
        _validate_policy_action_shape(action_type, cards, choice, context=context)

        raw_prob = raw_entry.get("prob")
        try:
            prob = float(raw_prob)
        except (TypeError, ValueError):
            raise ValueError(f"{context}: prob must be numeric.")
        if prob < 0:
            raise ValueError(f"{context}: prob must be non-negative.")

        encoded_distribution.append(((action_type, tuple(cards), choice), prob))
        total_prob += prob

    if total_prob <= 0:
        raise ValueError(f"state '{key}': distribution total probability must be positive.")
    if abs(total_prob - 1.0) > _EPS:
        raise ValueError(f"state '{key}': probabilities must sum to 1.0, got {total_prob:.6f}.")

    return tuple(encoded_distribution)


def _load_policy_table_artifact_internal(
    path: str,
    suits_enum: type,
    *,
    expected_variant: Optional[str] = None,
    key_encoding: str = POLICY_TABLE_KEY_ENCODING,
) -> Mapping[Any, _InternalEncodedDistribution]:
    """The encoded table of a JSON artifact, or of a binary policy table
    (binary_policy_table.py), which is decoded entry by entry on lookup."""
    if _read_binary_header(path) is not None:
        return BinaryPolicyTable.load(path, suits_enum, expected_variant=expected_variant,
                                      key_encoding=key_encoding)

    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise ValueError("Artifact root must be an object.")

    artifact_format = payload.get("format")
    if artifact_format != POLICY_TABLE_FORMAT:
        raise ValueError(f"Unsupported artifact format '{artifact_format}', expected '{POLICY_TABLE_FORMAT}'.")

    artifact_key_encoding = payload.get("key_encoding")
    if artifact_key_encoding != key_encoding:
        raise ValueError(
            f"Unsupported key_encoding '{artifact_key_encoding}', expected '{key_encoding}'."
        )

    variant = payload.get("variant")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"Artifact variant '{variant}' does not match expected '{expected_variant}'.")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, dict):
        raise ValueError("Artifact 'entries' must be an object mapping key -> distribution.")

    suit_lookup = {suit.value: suit for suit in suits_enum}
    if not suit_lookup:
        raise ValueError("suits_enum must define at least one suit.")

    encoded_table: _InternalEncodedPolicyTable = {}
    for key, raw_distribution in raw_entries.items():
        if not isinstance(key, str):
            raise ValueError("Artifact entries keys must be strings.")
        encoded_table[key] = _decode_distribution_from_payload(key, raw_distribution, suit_lookup)

    return encoded_table


def policy_table_key_encoding(path: str) -> Optional[str]:
    """The key encoding of a policy-table file (JSON or binary), or None if
    `path` is not a policy table."""
    header = _read_binary_header(path)
    if header is not None:
        return header["key_encoding"]
    with open(path, "rb") as fp:
        if fp.read(1) != b"{":
            return None
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return payload.get("key_encoding") if isinstance(payload, dict) else None


def load_policy_table_artifact(
    path: str,
    suits_enum: type,
    *,
    expected_variant: Optional[str] = None,
) -> Dict[str, Dict[Action, float]]:
    """Load a policy-table artifact (JSON or binary) into an in-memory strategy mapping."""
    encoded_table = _load_policy_table_artifact_internal(path, suits_enum, expected_variant=expected_variant)
    return {key: _decode_distribution(distribution) for key, distribution in encoded_table.items()}


class PolicyTableAgent(Agent):
    """Agent that serves a precomputed policy table keyed by RoundInfoState string."""

    # How info states become table keys (when building the table and at
    # lookup), and the matching artifact tag.  Overridden by MarkovPolicyTableAgent.
    KEY_ENCODING = POLICY_TABLE_KEY_ENCODING

    @staticmethod
    def _policy_key(raw_key: Union[str, RoundInfoState]) -> str:
        return _coerce_policy_key(raw_key)

    @staticmethod
    def _lookup_key(info_state: Any) -> str:
        return str(info_state)

    def __init__(
        self,
        strategy_map: Mapping[Union[str, RoundInfoState], Mapping[Action, float]],
        fallback: Optional[Agent] = None,
    ):
        """`fallback`, if given, plays info states the table has no entry for
        (e.g. a partial table for full Hanamikoji, whose tree is far too large
        to tabulate).  Without it, a missing state raises ValueError."""
        self._table = _encode_strategy_map(strategy_map, key_fn=self._policy_key)
        self._fallback = fallback

    @classmethod
    def from_artifact(
        cls,
        path: str,
        suits_enum: type,
        *,
        expected_variant: Optional[str] = None,
        fallback: Optional[Agent] = None,
    ) -> "PolicyTableAgent":
        encoded_table = _load_policy_table_artifact_internal(
            path,
            suits_enum,
            expected_variant=expected_variant,
            key_encoding=cls.KEY_ENCODING,
        )
        agent = cls.__new__(cls)
        agent._table = encoded_table
        agent._fallback = fallback
        return agent

    def dump_artifact(self, path: str, *, variant: Optional[str] = None) -> None:
        """Write this agent's internal policy table to JSON artifact format."""
        payload = _build_artifact_payload(self._table, variant=variant, key_encoding=self.KEY_ENCODING)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)

    def dump_binary_artifact(self, path: str, *, variant: Optional[str] = None,
                             metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Write this agent's policy table as a binary policy table (binary_policy_table.py)."""
        write_binary_policy_table(path, self._table.items(), key_encoding=self.KEY_ENCODING,
                                  variant=variant, metadata=metadata)

    @classmethod
    def table_key(cls, raw_key: Union[str, bytes, RoundInfoState]) -> Union[str, bytes]:
        """The key this kind of table uses for an info state, its string or its compact key."""
        return cls._policy_key(raw_key)

    def table_entry(self, raw_key: Union[str, bytes, RoundInfoState]) -> Optional[_InternalEncodedDistribution]:
        """The table's encoded distribution ((action_type, suits, choice), prob)
        for an info state, or None if the table does not list it (the
        fallback is not consulted)."""
        key = self.table_key(raw_key)
        return self._table[key] if key in self._table else None

    def _missing_key(self, info_state: Any, key: str) -> Any:
        """Distribution for an info state not in the table: the fallback's, or raise."""
        fallback = getattr(self, "_fallback", None)   # subclasses may skip __init__
        if fallback is None:
            raise ValueError(f"Policy table missing state key: {key}")
        return fallback.get_action_distribution(info_state)

    def get_action_distribution(self, info_state: RoundInfoState) -> Dict[Action, float]:
        key = self._lookup_key(info_state)
        if key not in self._table:
            return self._missing_key(info_state, key)
        return _decode_distribution(self._table[key])


class _MarkovPolicyKeys:
    """Mixin: round-independent keys (shared by both engine backends' agents)."""

    KEY_ENCODING = MARKOV_POLICY_TABLE_KEY_ENCODING

    @staticmethod
    def _policy_key(raw_key: Union[str, RoundInfoState]) -> str:
        return markov_policy_key(raw_key)

    @staticmethod
    def _lookup_key(info_state: Any) -> str:
        return markov_policy_key(str(info_state))


class MarkovPolicyTableAgent(_MarkovPolicyKeys, PolicyTableAgent):
    """Policy table for a Markov (stationary) strategy: one strategy per info
    state, independent of the round number.

    In the Markov-game model of Hanamikoji, the state carried between rounds
    is (player who starts the round, favor state); hands and the deck are
    re-dealt every round.  A Markov strategy is stationary: it depends only on
    that state and on what has happened in the current round, never on how
    many rounds have been played.  It plays round 5 from a given
    (starting player, favors) exactly as it would play round 1 from there.

    The table therefore stores, and looks up, str(RoundInfoState) with the
    round number removed (markov_policy_key); the round's starting favors and
    starting player remain part of the key.  This is what lets the Markov
    solver's subgames, which are always built as round 1, serve every round.
    It would be wrong for a non-stationary (round-dependent) strategy — use
    PolicyTableAgent for those.

    Artifacts are tagged key_encoding = MARKOV_POLICY_TABLE_KEY_ENCODING;
    PolicyTableAgent and MarkovPolicyTableAgent refuse each other's files.
    """


class _CompactPolicyKeys:
    """Mixin: compact keys (engine/compact_key.py), shared by both engine backends' agents."""

    KEY_ENCODING = COMPACT_POLICY_TABLE_KEY_ENCODING

    @staticmethod
    def _policy_key(raw_key: Union[str, bytes, RoundInfoState]) -> bytes:
        if isinstance(raw_key, bytes):
            return raw_key
        if isinstance(raw_key, str):
            return string_to_compact_key(raw_key)
        if hasattr(raw_key, "compact_key"):
            return raw_key.compact_key()
        raise TypeError(f"Policy key must be bytes, str or RoundInfoState, got {type(raw_key).__name__}.")

    @staticmethod
    def _lookup_key(info_state: Any) -> bytes:
        return info_state.compact_key()

    def dump_artifact(self, path: str, *, variant: Optional[str] = None) -> None:
        raise ValueError("Compact keys are bytes, which JSON artifacts cannot hold; "
                         "use dump_binary_artifact.")


class CompactPolicyTableAgent(_CompactPolicyKeys, PolicyTableAgent):
    """PolicyTableAgent keyed by compact keys instead of str(RoundInfoState).

    The same strategy as a PolicyTableAgent with the same entries (each
    compact key converts exactly to its string); its tables are smaller,
    which matters for millions of entries. Only binary artifacts
    (dump_binary_artifact; tagged COMPACT_POLICY_TABLE_KEY_ENCODING) can hold
    them. Keys given as strings or info states are converted.
    """
