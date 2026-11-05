"""Compact information-state keys.

A compact key holds exactly the information of str(RoundInfoState), as a
few bytes instead of a few hundred characters, for solvers that store
millions of information sets. It is an alternative, not a replacement:
str(RoundInfoState) remains the key of policy-table artifacts and agents.
Both directions convert exactly:

    compact_key_to_string(info_state.compact_key()) == str(info_state)
    string_to_compact_key(str(info_state)) == info_state.compact_key()

Encoding (every value is one byte unless stated):

    player_id, round_number, then for each event:
    event code, turn, then the event's fields in declaration order, which is
    also their order in the event's string:
        i  small int (chooser, choice index, starting player)
        s  suit value;  2 / 3: that many suit values;  p: two pairs, 4 values
        l  variable-length suit list: length, then suit values
        f  favors: number of suits, then owner + 1 for each suit (0, 1, 2)
        a  action set: 16-bit mask, bit k = k-th ActionType, low byte first

The event code is the position in _EVENT_LAYOUTS, which is the order of the
C++ Event variant (engine_cpp/include/hana/events.h); the C++ engine writes
the same bytes in RoundInfoState::compact_key().

Keys only grow by appending, as strings do: a player's key at any state is a
prefix of their key at every later state of the round.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Callable, List, Tuple

from . import events as ev
from .enums import ActionType

# Bit k of the action mask is the k-th ActionType (the C++ ActionType values).
_ACTION_TYPES = list(ActionType)

# (string name, event class, kinds of the fields after `turn`), indexed by event code.
_EVENT_LAYOUTS = [
    ("ROUND_START", ev.RoundStartEvent, "lfia"),
    ("DRAW_CARD", ev.DrawCardEvent, "s"),
    ("OWN_RESERVE", ev.OwnReserveEvent, "s"),
    ("OPPONENT_RESERVE", ev.OpponentReserveEvent, ""),
    ("OWN_SPLIT", ev.OwnSplitEvent, "2"),
    ("OPPONENT_SPLIT", ev.OpponentSplitEvent, "2"),
    ("SPLIT_CHOICE", ev.SplitChoiceEvent, "iiss"),
    ("OWN_DISCARD", ev.OwnDiscardEvent, "ss"),
    ("OPPONENT_DISCARD", ev.OpponentDiscardEvent, "s"),
    ("ROUND_END", ev.RoundEndEvent, "llf"),
    ("OWN_TRADE_OFF", ev.OwnTradeOffEvent, "2"),
    ("OPPONENT_TRADE_OFF", ev.OpponentTradeOffEvent, ""),
    ("OWN_GIFT", ev.OwnGiftEvent, "3"),
    ("OPPONENT_GIFT", ev.OpponentGiftEvent, "3"),
    ("GIFT_CHOICE", ev.GiftChoiceEvent, "iis2"),
    ("OWN_COMPETITION", ev.OwnCompetitionEvent, "p"),
    ("OPPONENT_COMPETITION", ev.OpponentCompetitionEvent, "p"),
    ("COMPETITION_CHOICE", ev.CompetitionChoiceEvent, "ii22"),
]

_CODE_BY_CLASS = {cls: code for code, (_, cls, _) in enumerate(_EVENT_LAYOUTS)}
_CODE_BY_NAME = {name: code for code, (name, _, _) in enumerate(_EVENT_LAYOUTS)}
_FIELD_NAMES = {cls: [f.name for f in dataclasses.fields(cls)][1:]   # after `turn`
                for _, cls, _ in _EVENT_LAYOUTS}


def _value(x: Any) -> int:
    """Suit (any variant) or int -> int."""
    return x.value if hasattr(x, "value") else int(x)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _encode_field(kind: str, value: Any, out: List[int]) -> None:
    if kind in "is":
        out.append(_value(value))
    elif kind in "23":
        out.extend(_value(v) for v in value)
    elif kind == "p":
        out.extend(_value(v) for pair in value for v in pair)
    elif kind == "l":
        out.append(len(value))
        out.extend(_value(v) for v in value)
    elif kind == "f":
        out.append(len(value))
        for index, (suit, owner) in enumerate(value):
            if _value(suit) != index:
                raise ValueError(f"favors must list suits 0..n-1 in order, got {value!r}")
            out.append(owner + 1)
    elif kind == "a":
        mask = 0
        for action_type in value:
            mask |= 1 << _ACTION_TYPES.index(action_type)
        out.extend((mask & 0xFF, mask >> 8))
    else:
        raise AssertionError(kind)


def encode_event(event: ev.ObservationEvent, out: List[int]) -> None:
    """Append an event's code, turn and fields to `out`."""
    code = _CODE_BY_CLASS[type(event)]
    out.extend((code, event.turn))
    for kind, name in zip(_EVENT_LAYOUTS[code][2], _FIELD_NAMES[type(event)]):
        _encode_field(kind, getattr(event, name), out)


def encode(player_id: int, round_number: int, events) -> bytes:
    out = [player_id, round_number]
    for event in events:
        encode_event(event, out)
    return bytes(out)


# ---------------------------------------------------------------------------
# Compact key -> string
# ---------------------------------------------------------------------------

def _decode_field(kind: str, key: bytes, pos: int, suit: Callable[[int], Any]) -> Tuple[Any, int]:
    if kind == "i":
        return key[pos], pos + 1
    if kind == "s":
        return suit(key[pos]), pos + 1
    if kind in "23":
        n = int(kind)
        return tuple(suit(v) for v in key[pos:pos + n]), pos + n
    if kind == "p":
        return (tuple(suit(v) for v in key[pos:pos + 2]),
                tuple(suit(v) for v in key[pos + 2:pos + 4])), pos + 4
    if kind == "l":
        n = key[pos]
        return tuple(suit(v) for v in key[pos + 1:pos + 1 + n]), pos + 1 + n
    if kind == "f":
        n = key[pos]
        return tuple((suit(i), key[pos + 1 + i] - 1) for i in range(n)), pos + 1 + n
    if kind == "a":
        mask = key[pos] | (key[pos + 1] << 8)
        return frozenset(t for k, t in enumerate(_ACTION_TYPES) if mask >> k & 1), pos + 2
    raise AssertionError(kind)


def decode_events(key: bytes, suit: Callable[[int], Any] = int
                  ) -> Tuple[int, int, List[ev.ObservationEvent]]:
    """(player_id, round_number, events) of a compact key.

    Suits are built from their values by `suit`: ints by default, which
    format exactly as the original suits (all compact_key_to_string needs),
    or a variant's Suit enum (compact_key_to_info_state).
    """
    events = []
    pos = 2
    while pos < len(key):
        code, turn = key[pos], key[pos + 1]
        _, cls, kinds = _EVENT_LAYOUTS[code]
        pos += 2
        values = []
        for kind in kinds:
            value, pos = _decode_field(kind, key, pos, suit)
            values.append(value)
        events.append(cls(turn, *values))
    return key[0], key[1], events


def compact_key_to_string(key: bytes) -> str:
    """The str(RoundInfoState) of a compact key."""
    player_id, round_number, events = decode_events(key)
    return "|".join([f"p{player_id}r{round_number}"] + [str(event) for event in events])


def compact_key_to_info_state(key: bytes, suits_enum: type):
    """The Python-engine RoundInfoState of a compact key, with suits of `suits_enum`.

    It answers every query a live info state does (legal actions, hand,
    favors, ...), so a key alone is enough to ask an agent what it would play.
    Used for both backends: the C++ RoundInfoState cannot be built from events.
    """
    from .info_state import RoundInfoState   # info_state imports this module

    player_id, round_number, events = decode_events(key, suit=suits_enum)
    return RoundInfoState(player_id, round_number, tuple(events))


# ---------------------------------------------------------------------------
# String -> compact key
# ---------------------------------------------------------------------------

_HEADER = re.compile(r"p(\d+)r(\d+)$")
_EVENT = re.compile(r"([A-Z_]+)@t(\d+)(?:\((.*)\))?$")


def _split_top_level(text: str) -> List[str]:
    """Split on commas that are not inside brackets."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    if text:
        parts.append(text[start:])
    return parts


def _parse_value(kind: str, text: str) -> Any:
    if kind in "is":
        return int(text)
    inner = _split_top_level(text[1:-1])   # strip the brackets
    if kind in "23l":
        return tuple(int(v) for v in inner)
    if kind == "p":
        return tuple(_parse_value("2", pair) for pair in inner)
    if kind == "f":
        return tuple(tuple(int(v) for v in item.split(":")) for item in inner)
    if kind == "a":
        return frozenset(ActionType(v) for v in inner)
    raise AssertionError(kind)


def string_to_compact_key(text: str) -> bytes:
    """The compact key of a str(RoundInfoState).

    Raises ValueError if the string is not one the engine produces.
    """
    header, *event_texts = text.split("|")
    match = _HEADER.match(header)
    if not match:
        raise ValueError(f"not an information-state string: {text[:40]!r}")
    events = []
    for event_text in event_texts:
        match_event = _EVENT.match(event_text)
        if not match_event or match_event.group(1) not in _CODE_BY_NAME:
            raise ValueError(f"unknown event {event_text!r}")
        name, turn, body = match_event.groups()
        _, cls, kinds = _EVENT_LAYOUTS[_CODE_BY_NAME[name]]
        items = _split_top_level(body or "")
        if len(items) != len(kinds):
            raise ValueError(f"unexpected fields in {event_text!r}")
        values = [_parse_value(kind, item.split("=", 1)[1]) for kind, item in zip(kinds, items)]
        event = cls(int(turn), *values)
        if str(event) != event_text:
            raise ValueError(f"cannot reproduce {event_text!r} (got {str(event)!r})")
        events.append(event)
    return encode(int(match.group(1)), int(match.group(2)), events)
