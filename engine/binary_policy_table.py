"""Binary policy tables: the policy_table_v1 content in a compact file.

A policy table maps information-state keys to action distributions. The
JSON artifact (policy_table.py) is readable but large: a medium_hana
strategy has millions of entries. This binary file holds the same content,
with keys either as strings or as compact keys (engine/compact_key.py), and
is written the same way from Python (`write_binary_policy_table`) and C++
(engine_cpp/include/hana/policy_table.h). PolicyTableAgent.from_artifact
reads both kinds of file.

A table may be partial: it lists only the information sets it has. What to
play elsewhere is decided by whoever loads it (PolicyTableAgent's `fallback`,
or raise).

File layout (all integers unsigned, little-endian):

    line 1   JSON header ending in "\\n", for example
             {"format": "binary_policy_table_v1", "key_encoding": "...", "variant": "medium_hana"}
             with an optional "metadata" object (anything the writer wants to record)
    then one record per information set, to the end of the file:
             u32  key length, then the key bytes
                  (UTF-8 for the string encodings, raw bytes for compact keys)
             u8   number of actions (at least 1)
             per action:
                  u8   action type (the C++ ActionType value)
                  u8   number of cards, then one u8 suit value per card
                  u8   choice, 255 for none
                  f64  probability

Each distribution lists only actions the table may play; probabilities are
non-negative and sum to 1. Actions are written out, not given as positions
in a legal-action list, so the file does not depend on the order in which an
engine lists legal actions.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple, Union

from .enums import ActionType

BINARY_POLICY_TABLE_FORMAT = "binary_policy_table_v1"
# Key encodings a binary table may use. The two string encodings are the
# JSON artifact's; compact keys only exist in binary files.
_STRING_KEY_ENCODINGS = ("round_info_state_str_v1", "markov_round_info_state_str_v1")
COMPACT_POLICY_TABLE_KEY_ENCODING = "round_info_state_compact_v1"

_NO_CHOICE = 255
_MAX_HEADER_LENGTH = 1 << 16
# Byte k of an action type is the k-th ActionType (the C++ ActionType values),
# as in compact keys.
_ACTION_TYPES = list(ActionType)
# By name, so that both engines' ActionType enums are accepted.
_ACTION_TYPE_CODE = {action_type.name: code for code, action_type in enumerate(_ACTION_TYPES)}

Key = Union[str, bytes]
# (action type, suits, choice) and its probability; the same encoded form as
# policy_table.py uses for its tables.
EncodedAction = Tuple[ActionType, Tuple[Any, ...], Optional[int]]
EncodedDistribution = Tuple[Tuple[EncodedAction, float], ...]


def _key_to_bytes(key: Key, key_encoding: str) -> bytes:
    if key_encoding == COMPACT_POLICY_TABLE_KEY_ENCODING:
        if not isinstance(key, bytes):
            raise TypeError(f"compact keys must be bytes, got {type(key).__name__}")
        return key
    if not isinstance(key, str):
        raise TypeError(f"{key_encoding} keys must be str, got {type(key).__name__}")
    return key.encode("utf-8")


def _key_from_bytes(raw: bytes, key_encoding: str) -> Key:
    return raw if key_encoding == COMPACT_POLICY_TABLE_KEY_ENCODING else raw.decode("utf-8")


def _suit_value(suit: Any) -> int:
    return suit.value if hasattr(suit, "value") else int(suit)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def encode_record_body(distribution: EncodedDistribution) -> bytes:
    """The bytes of a record after its key: number of actions, then each action."""
    out = bytearray([len(distribution)])
    for (action_type, suits, choice), prob in distribution:
        out.append(_ACTION_TYPE_CODE[action_type.name])
        out.append(len(suits))
        out.extend(_suit_value(suit) for suit in suits)
        out.append(_NO_CHOICE if choice is None else choice)
        out += struct.pack("<d", prob)
    return bytes(out)


def write_binary_policy_table(path: str, entries: Iterable[Tuple[Key, EncodedDistribution]], *,
                              key_encoding: str, variant: Optional[str] = None,
                              metadata: Optional[Mapping[str, Any]] = None) -> int:
    """Write (key, distribution) pairs to a binary policy table; returns how many.

    `entries` may be a generator, so a large table never has to exist in
    memory in this encoded form. Keys must be unique.
    """
    if key_encoding not in _STRING_KEY_ENCODINGS + (COMPACT_POLICY_TABLE_KEY_ENCODING,):
        raise ValueError(f"unknown key_encoding {key_encoding!r}")
    header: Dict[str, Any] = {"format": BINARY_POLICY_TABLE_FORMAT,
                              "key_encoding": key_encoding, "variant": variant}
    if metadata is not None:
        header["metadata"] = dict(metadata)
    count = 0
    with open(path, "wb") as f:
        f.write(json.dumps(header, sort_keys=True).encode("utf-8") + b"\n")
        for key, distribution in entries:
            raw_key = _key_to_bytes(key, key_encoding)
            f.write(struct.pack("<I", len(raw_key)))
            f.write(raw_key)
            f.write(encode_record_body(distribution))
            count += 1
    return count


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_header(path: str) -> Optional[Dict[str, Any]]:
    """The header of a binary policy table, or None if `path` is not one."""
    with open(path, "rb") as f:
        line = f.readline(_MAX_HEADER_LENGTH)
    try:
        header = json.loads(line)
    except (UnicodeDecodeError, ValueError):
        return None
    if isinstance(header, dict) and header.get("format") == BINARY_POLICY_TABLE_FORMAT:
        return header
    return None


def is_binary_policy_table(path: str) -> bool:
    return read_header(path) is not None


def _read_records(data: bytes, start: int) -> Iterator[Tuple[bytes, bytes]]:
    """(key bytes, record body bytes) of every record, checking their framing."""
    pos = start
    while pos < len(data):
        if pos + 4 > len(data):
            raise ValueError("truncated record (key length)")
        (key_length,) = struct.unpack_from("<I", data, pos)
        key_start = pos + 4
        body_start = key_start + key_length
        if body_start >= len(data):
            raise ValueError("truncated record (key)")
        pos = body_start + 1
        for _ in range(data[body_start]):
            if pos + 2 > len(data):
                raise ValueError("truncated record (action)")
            pos += 2 + data[pos + 1] + 1 + 8   # type, card count, cards, choice, probability
        if pos > len(data):
            raise ValueError("truncated record (action)")
        yield data[key_start:body_start], data[body_start:pos]


def decode_record_body(body: bytes, suit_lookup: Mapping[int, Any]) -> EncodedDistribution:
    """Inverse of encode_record_body, with suits from `suit_lookup` (value -> suit)."""
    distribution: List[Tuple[EncodedAction, float]] = []
    pos = 1
    for _ in range(body[0]):
        action_type = _ACTION_TYPES[body[pos]]
        num_cards = body[pos + 1]
        suits = tuple(suit_lookup[value] for value in body[pos + 2:pos + 2 + num_cards])
        pos += 2 + num_cards
        choice = None if body[pos] == _NO_CHOICE else body[pos]
        (prob,) = struct.unpack_from("<d", body, pos + 1)
        pos += 9
        distribution.append(((action_type, suits, choice), prob))
    return tuple(distribution)


class BinaryPolicyTable(Mapping):
    """A loaded binary policy table: a read-only mapping key -> encoded distribution.

    Each record is checked when the file is loaded, but kept as bytes and
    decoded only when looked up, so a table of millions of entries stays
    small in memory. Usable wherever policy_table.py expects an encoded table.
    """

    def __init__(self, records: Dict[Key, bytes], suit_lookup: Mapping[int, Any],
                 key_encoding: str, variant: Optional[str], metadata: Optional[dict]):
        self._records = records
        self._suit_lookup = suit_lookup
        self.key_encoding = key_encoding
        self.variant = variant
        self.metadata = metadata

    @classmethod
    def load(cls, path: str, suits_enum: type, *, expected_variant: Optional[str] = None,
             key_encoding: Optional[str] = None) -> "BinaryPolicyTable":
        """Read and check a binary policy table.

        Raises ValueError if the file is not a binary policy table, its key
        encoding or variant differ from the expected ones, or any record is
        malformed (unknown action type or suit, wrong number of cards,
        probabilities not summing to 1, duplicate keys).
        """
        from .policy_table import _EPS, _validate_policy_action_shape   # policy_table imports this module

        header = read_header(path)
        if header is None:
            raise ValueError(f"{path} is not a binary policy table.")
        if key_encoding is not None and header["key_encoding"] != key_encoding:
            raise ValueError(f"Unsupported key_encoding '{header['key_encoding']}', expected '{key_encoding}'.")
        if expected_variant is not None and header.get("variant") != expected_variant:
            raise ValueError(f"Artifact variant '{header.get('variant')}' does not match "
                             f"expected '{expected_variant}'.")
        suit_lookup = {suit.value: suit for suit in suits_enum}

        with open(path, "rb") as f:
            data = f.read()
        records: Dict[Key, bytes] = {}
        for raw_key, body in _read_records(data, data.index(b"\n") + 1):
            key = _key_from_bytes(raw_key, header["key_encoding"])
            context = f"state {key!r}"
            if key in records:
                raise ValueError(f"{context}: duplicate key")
            if body[0] == 0:
                raise ValueError(f"{context}: distribution cannot be empty.")
            try:
                distribution = decode_record_body(body, suit_lookup)
            except (IndexError, KeyError) as error:
                raise ValueError(f"{context}: unknown action type or suit ({error!r})")
            total = 0.0
            for index, ((action_type, suits, choice), prob) in enumerate(distribution):
                _validate_policy_action_shape(action_type, suits, choice, context=f"{context} entry {index}")
                if not prob >= 0.0:
                    raise ValueError(f"{context} entry {index}: prob must be non-negative.")
                total += prob
            if abs(total - 1.0) > _EPS:
                raise ValueError(f"{context}: probabilities must sum to 1.0, got {total:.6f}.")
            records[key] = body
        return cls(records, suit_lookup, header["key_encoding"], header.get("variant"),
                   header.get("metadata"))

    def __getitem__(self, key: Key) -> EncodedDistribution:
        return decode_record_body(self._records[key], self._suit_lookup)

    def __contains__(self, key: object) -> bool:
        return key in self._records

    def __iter__(self) -> Iterator[Key]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)
