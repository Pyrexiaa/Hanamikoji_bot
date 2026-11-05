"""Observation event model and event helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple

from .enums import ActionType, ObservationType


def _freeze_favors(favors: Dict[Any, int]) -> Tuple[Tuple[Any, int], ...]:
    """Convert favors dict to sorted tuple for hashability."""

    return tuple(sorted(favors.items(), key=lambda x: x[0].value))


@dataclass(frozen=True)
class ObservationEvent:
    """Base class for observable events. Do not instantiate directly."""

    turn: int

    @property
    def event_type(self) -> ObservationType:
        raise NotImplementedError("Subclasses must implement event_type")

    @staticmethod
    def _fmt_value(value: Any) -> str:
        """Format enum-like values deterministically for key strings."""

        return str(value.value) if hasattr(value, "value") else str(value)

    @classmethod
    def _fmt_values(cls, values: Tuple[Any, ...]) -> str:
        """Format tuple/list values deterministically for key strings."""

        return ",".join(cls._fmt_value(v) for v in values)

    @classmethod
    def _fmt_favors(cls, favors: Tuple[Tuple[Any, int], ...]) -> str:
        """Format favor tuples deterministically for key strings."""

        return ",".join(f"{cls._fmt_value(suit)}:{owner}" for suit, owner in favors)


@dataclass(frozen=True)
class RoundStartEvent(ObservationEvent):
    """Emitted at the start of each round."""

    initial_hand: Tuple[Any, ...]
    favors: Tuple[Tuple[Any, int], ...]
    starting_player: int
    available_actions: FrozenSet[ActionType]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.ROUND_START

    def __str__(self) -> str:
        actions = ",".join(sorted(action.value for action in self.available_actions))
        return (
            f"{self.event_type.name}@t{self.turn}"
            f"(hand=[{self._fmt_values(self.initial_hand)}],"
            f"favors=[{self._fmt_favors(self.favors)}],"
            f"start={self.starting_player},"
            f"actions=[{actions}])"
        )


@dataclass(frozen=True)
class DrawCardEvent(ObservationEvent):
    """Emitted when the player draws a card."""

    card: Any

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.DRAW_CARD

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(card={self._fmt_value(self.card)})"


@dataclass(frozen=True)
class OwnReserveEvent(ObservationEvent):
    """Emitted when the player reserves a card."""

    card: Any

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OWN_RESERVE

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(card={self._fmt_value(self.card)})"


@dataclass(frozen=True)
class OpponentReserveEvent(ObservationEvent):
    """Emitted when opponent reserves a card (card hidden)."""

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OPPONENT_RESERVE

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}"


@dataclass(frozen=True)
class OwnDiscardEvent(ObservationEvent):
    """Emitted when the player discards cards."""

    revealed_card: Any
    hidden_card: Any

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OWN_DISCARD

    def __str__(self) -> str:
        return (
            f"{self.event_type.name}@t{self.turn}"
            f"(revealed={self._fmt_value(self.revealed_card)},"
            f"hidden={self._fmt_value(self.hidden_card)})"
        )


@dataclass(frozen=True)
class OpponentDiscardEvent(ObservationEvent):
    """Emitted when opponent discards (only revealed card visible)."""

    revealed_card: Any

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OPPONENT_DISCARD

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(revealed={self._fmt_value(self.revealed_card)})"


@dataclass(frozen=True)
class OwnSplitEvent(ObservationEvent):
    """Emitted when the player offers a split."""

    cards: Tuple[Any, Any]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OWN_SPLIT

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(cards=[{self._fmt_values(self.cards)}])"


@dataclass(frozen=True)
class OpponentSplitEvent(ObservationEvent):
    """Emitted when opponent offers a split."""

    cards: Tuple[Any, Any]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OPPONENT_SPLIT

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(cards=[{self._fmt_values(self.cards)}])"


@dataclass(frozen=True)
class SplitChoiceEvent(ObservationEvent):
    """Emitted when a split choice is made."""

    chooser: int
    choice_index: int
    chooser_gets: Any
    offerer_gets: Any

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.SPLIT_CHOICE

    def __str__(self) -> str:
        return (
            f"{self.event_type.name}@t{self.turn}"
            f"(chooser={self.chooser},choice={self.choice_index},"
            f"chooser_gets={self._fmt_value(self.chooser_gets)},"
            f"offerer_gets={self._fmt_value(self.offerer_gets)})"
        )


@dataclass(frozen=True)
class RoundEndEvent(ObservationEvent):
    """Emitted at the end of each round."""

    p0_reserved_revealed: Tuple[Any, ...]
    p1_reserved_revealed: Tuple[Any, ...]
    final_favors: Tuple[Tuple[Any, int], ...]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.ROUND_END

    def __str__(self) -> str:
        return (
            f"{self.event_type.name}@t{self.turn}"
            f"(p0_reserved=[{self._fmt_values(self.p0_reserved_revealed)}],"
            f"p1_reserved=[{self._fmt_values(self.p1_reserved_revealed)}],"
            f"favors=[{self._fmt_favors(self.final_favors)}])"
        )



# ---------------------------------------------------------------------------
# Full Hanamikoji events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OwnTradeOffEvent(ObservationEvent):
    """Emitted when the player discards two cards face down (trade-off)."""

    cards: Tuple[Any, Any]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OWN_TRADE_OFF

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(cards=[{self._fmt_values(self.cards)}])"


@dataclass(frozen=True)
class OpponentTradeOffEvent(ObservationEvent):
    """Emitted when the opponent trades off (both cards hidden)."""

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OPPONENT_TRADE_OFF

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}"


@dataclass(frozen=True)
class OwnGiftEvent(ObservationEvent):
    """Emitted when the player offers a gift of three cards."""

    cards: Tuple[Any, Any, Any]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OWN_GIFT

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(cards=[{self._fmt_values(self.cards)}])"


@dataclass(frozen=True)
class OpponentGiftEvent(ObservationEvent):
    """Emitted when the opponent offers a gift of three cards."""

    cards: Tuple[Any, Any, Any]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OPPONENT_GIFT

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(cards=[{self._fmt_values(self.cards)}])"


@dataclass(frozen=True)
class GiftChoiceEvent(ObservationEvent):
    """Emitted when the chooser takes one card of a gift; the offerer keeps two."""

    chooser: int
    choice_index: int
    chooser_gets: Any
    offerer_gets: Tuple[Any, Any]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.GIFT_CHOICE

    def __str__(self) -> str:
        return (
            f"{self.event_type.name}@t{self.turn}"
            f"(chooser={self.chooser},choice={self.choice_index},"
            f"chooser_gets={self._fmt_value(self.chooser_gets)},"
            f"offerer_gets=[{self._fmt_values(self.offerer_gets)}])"
        )


@dataclass(frozen=True)
class OwnCompetitionEvent(ObservationEvent):
    """Emitted when the player offers two pairs (competition)."""

    pairs: Tuple[Tuple[Any, Any], Tuple[Any, Any]]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OWN_COMPETITION

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(pairs={_fmt_pairs(self.pairs)})"


@dataclass(frozen=True)
class OpponentCompetitionEvent(ObservationEvent):
    """Emitted when the opponent offers two pairs (competition)."""

    pairs: Tuple[Tuple[Any, Any], Tuple[Any, Any]]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.OPPONENT_COMPETITION

    def __str__(self) -> str:
        return f"{self.event_type.name}@t{self.turn}(pairs={_fmt_pairs(self.pairs)})"


@dataclass(frozen=True)
class CompetitionChoiceEvent(ObservationEvent):
    """Emitted when the chooser takes one pair; the offerer gets the other."""

    chooser: int
    choice_index: int
    chooser_gets: Tuple[Any, Any]
    offerer_gets: Tuple[Any, Any]

    @property
    def event_type(self) -> ObservationType:
        return ObservationType.COMPETITION_CHOICE

    def __str__(self) -> str:
        return (
            f"{self.event_type.name}@t{self.turn}"
            f"(chooser={self.chooser},choice={self.choice_index},"
            f"chooser_gets=[{self._fmt_values(self.chooser_gets)}],"
            f"offerer_gets=[{self._fmt_values(self.offerer_gets)}])"
        )


def _fmt_pairs(pairs: Tuple[Tuple[Any, Any], ...]) -> str:
    """'[[a,b],[c,d]]' for a competition's two pairs."""
    return "[" + ",".join(f"[{ObservationEvent._fmt_values(pair)}]" for pair in pairs) + "]"
